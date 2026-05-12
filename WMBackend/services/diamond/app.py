from __future__ import annotations

import base64
import io
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIAMOND_ROOT = PROJECT_ROOT / "vendors" / "diamond"
DIAMOND_SRC = DIAMOND_ROOT / "src"
DIAMOND_CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "diamond"

if str(DIAMOND_SRC) not in sys.path:
    sys.path.insert(0, str(DIAMOND_SRC))

from hydra import compose, initialize_config_dir  # type: ignore
from hydra.utils import instantiate  # type: ignore
from huggingface_hub import snapshot_download  # type: ignore
from omegaconf import OmegaConf  # type: ignore

from agent import Agent  # type: ignore
from envs import WorldModelEnv  # type: ignore
from csgo.action_processing import MOUSE_X_POSSIBLES, MOUSE_Y_POSSIBLES  # type: ignore

OmegaConf.register_new_resolver("eval", eval)


class LoadRequest(BaseModel):
    model_id: Optional[str] = "diamond"


class StartRequest(BaseModel):
    init_image_base64: Optional[str] = None


class StepRequest(BaseModel):
    session_id: str
    action: Dict[str, Any]


class ResetRequest(BaseModel):
    session_id: str
    init_image_base64: Optional[str] = None


class RandomImageRequest(BaseModel):
    dataset_id: str = "CSGO"


@dataclass
class Runtime:
    agent: Any
    wm_env: Any
    device: torch.device
    session_id: Optional[str] = None


class DiamondRuntimeService:
    def __init__(self) -> None:
        self.runtime: Optional[Runtime] = None
        self.mouse_gain_x = float(os.getenv("WM_DIAMOND_MOUSE_GAIN_X", "300.0"))
        self.mouse_gain_y = float(os.getenv("WM_DIAMOND_MOUSE_GAIN_Y", "120.0"))
        self.movement_step_repeat = max(1, int(os.getenv("WM_DIAMOND_MOVEMENT_STEP_REPEAT", "3")))

    def load(self) -> Dict[str, Any]:
        print("[service] load requested", flush=True)
        if self.runtime is not None:
            print("[service] already loaded", flush=True)
            return {
                "model_id": "diamond",
                "status": "already_loaded",
                "device": str(self.runtime.device),
            }

        with initialize_config_dir(version_base="1.3", config_dir=str(DIAMOND_ROOT / "config")):
            cfg = compose(config_name="trainer")

        path_ckpt, spawn_dir = self._resolve_assets()
        cfg.agent = OmegaConf.load(DIAMOND_ROOT / "config" / "agent" / "csgo.yaml")
        cfg.env = OmegaConf.load(DIAMOND_ROOT / "config" / "env" / "csgo.yaml")

        device = self._pick_device()
        num_actions = int(cfg.env.num_actions)

        agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(device).eval()
        agent.load(path_ckpt)

        seq_length = int(cfg.agent.denoiser.inner_model.num_steps_conditioning)
        if agent.upsampler is not None:
            seq_length = max(seq_length, int(cfg.agent.upsampler.inner_model.num_steps_conditioning))

        wm_env_cfg = instantiate(cfg.world_model_env, num_batches_to_preload=1)
        wm_env = WorldModelEnv(
            agent.denoiser,
            agent.upsampler,
            agent.rew_end_model,
            spawn_dir,
            1,
            seq_length,
            wm_env_cfg,
            return_denoising_trajectory=False,
        )

        self.runtime = Runtime(agent=agent, wm_env=wm_env, device=device)
        print(f"[service] load done device={device}", flush=True)
        return {
            "model_id": "diamond",
            "status": "loaded",
            "device": str(device),
            "checkpoint": str(path_ckpt),
            "spawn_dir": str(spawn_dir),
        }

    def start_session(self, init_image_base64: Optional[str]) -> Dict[str, Any]:
        print("[service] start_session requested", flush=True)
        runtime = self._require_runtime()
        init_image_bytes = self._decode_image(init_image_base64)

        obs, _ = runtime.wm_env.reset()
        if init_image_bytes is not None:
            obs = self._inject_init_image(runtime, init_image_bytes)

        runtime.session_id = str(uuid.uuid4())
        print(f"[service] start_session done session_id={runtime.session_id}", flush=True)
        return {
            "session_id": runtime.session_id,
            "frame_base64": self._obs_to_base64(obs),
        }

    def reset_session(self, session_id: str, init_image_base64: Optional[str]) -> Dict[str, Any]:
        print(f"[service] reset_session requested session_id={session_id}", flush=True)
        runtime = self._require_runtime()
        self._require_session(runtime, session_id)
        init_image_bytes = self._decode_image(init_image_base64)

        obs, _ = runtime.wm_env.reset()
        if init_image_bytes is not None:
            obs = self._inject_init_image(runtime, init_image_bytes)

        return {
            "session_id": runtime.session_id,
            "frame_base64": self._obs_to_base64(obs),
        }

    @torch.no_grad()
    def step(self, session_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        runtime = self._require_runtime()
        self._require_session(runtime, session_id)

        act_vec = self._encode_action_vector(action, runtime.device)
        n_step = (
            self.movement_step_repeat
            if self._movement_only_repeatable(action)
            else 1
        )
        next_obs = rew = end = trunc = info = None
        for _ in range(n_step):
            next_obs, rew, end, trunc, info = runtime.wm_env.step(act_vec)

        return {
            "session_id": session_id,
            "frame_base64": self._obs_to_base64(next_obs),
            "reward": float(rew.item()),
            "ended": bool(end.item()),
            "truncated": bool(trunc.item()),
            "extra": {"info_keys": sorted(list(info.keys()))},
        }

    def random_spawn_image(self, dataset_id: str) -> Dict[str, Any]:
        print(f"[service] random_spawn_image dataset_id={dataset_id}", flush=True)
        if dataset_id.lower() != "csgo":
            raise RuntimeError(f"Unsupported dataset for diamond service: {dataset_id}")

        spawn_dir = self._resolve_spawn_dir()
        candidates = [d for d in spawn_dir.iterdir() if d.is_dir() and (d / "full_res.npy").exists()]
        if not candidates:
            raise RuntimeError(f"No spawn samples with full_res.npy found in {spawn_dir}")

        sample_dir = candidates[np.random.randint(0, len(candidates))]
        full_res = np.load(sample_dir / "full_res.npy")
        frame = self._pick_random_frame(full_res)
        data_url = self._frame_to_data_url(frame)
        return {
            "dataset_id": "CSGO",
            "file": str(sample_dir / "full_res.npy"),
            "image_base64": data_url,
        }

    def health(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "model_id": "diamond",
            "ready": self.runtime is not None,
            "session_id": None if self.runtime is None else self.runtime.session_id,
        }

    def _decode_image(self, payload: Optional[str]) -> Optional[bytes]:
        if not payload:
            return None
        if "," in payload:
            payload = payload.split(",", 1)[1]
        return base64.b64decode(payload)

    def _resolve_assets(self) -> Tuple[Path, Path]:
        env_ckpt = os.getenv("DIAMOND_CKPT_PATH")
        env_spawn = os.getenv("DIAMOND_SPAWN_DIR")
        if env_ckpt and env_spawn:
            return Path(env_ckpt), Path(env_spawn)

        local_ckpt = DIAMOND_CHECKPOINT_ROOT / "csgo" / "model" / "csgo.pt"
        local_spawn = DIAMOND_CHECKPOINT_ROOT / "csgo" / "spawn"
        if local_ckpt.exists() and local_spawn.exists():
            return local_ckpt, local_spawn

        path_hf = Path(snapshot_download(repo_id="eloialonso/diamond", allow_patterns="csgo/*"))
        return path_hf / "csgo/model/csgo.pt", path_hf / "csgo/spawn"

    def _resolve_spawn_dir(self) -> Path:
        env_spawn = os.getenv("DIAMOND_SPAWN_DIR")
        if env_spawn:
            return Path(env_spawn)

        local_spawn = DIAMOND_CHECKPOINT_ROOT / "csgo" / "spawn"
        if local_spawn.exists():
            return local_spawn

        path_hf = Path(snapshot_download(repo_id="eloialonso/diamond", allow_patterns="csgo/spawn/*"))
        return path_hf / "csgo/spawn"

    def _pick_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _require_runtime(self) -> Runtime:
        if self.runtime is None:
            raise RuntimeError("Model is not loaded. Call /load first.")
        return self.runtime

    def _require_session(self, runtime: Runtime, session_id: str) -> None:
        if runtime.session_id is None:
            raise RuntimeError("Session is not started. Call /sessions/start first.")
        if runtime.session_id != session_id:
            raise RuntimeError("Unknown or expired session_id")

    def _inject_init_image(self, runtime: Runtime, init_image_bytes: bytes) -> torch.Tensor:
        img = Image.open(io.BytesIO(init_image_bytes)).convert("RGB")

        low_h = int(runtime.wm_env.obs_buffer.shape[-2])
        low_w = int(runtime.wm_env.obs_buffer.shape[-1])
        full_h = int(runtime.wm_env.obs_full_res_buffer.shape[-2])
        full_w = int(runtime.wm_env.obs_full_res_buffer.shape[-1])

        low = self._image_to_tensor(img, low_w, low_h, runtime.device)
        full = self._image_to_tensor(img, full_w, full_h, runtime.device)

        runtime.wm_env.obs_buffer[:] = low.unsqueeze(0).unsqueeze(0)
        runtime.wm_env.obs_full_res_buffer[:] = full.unsqueeze(0).unsqueeze(0)

        neutral_action = self._encode_action_vector({}, runtime.device)
        runtime.wm_env.act_buffer[:] = neutral_action

        return runtime.wm_env.obs_full_res_buffer[:, -1]

    def _image_to_tensor(self, image: Image.Image, width: int, height: int, device: torch.device) -> torch.Tensor:
        arr = np.asarray(image.resize((width, height), resample=Image.BICUBIC), dtype=np.float32)
        t = torch.from_numpy(arr).permute(2, 0, 1).to(device)
        return t.div(255.0).mul(2.0).sub(1.0)

    def _obs_to_base64(self, obs: torch.Tensor) -> str:
        arr = (
            obs[0]
            .detach()
            .clamp(-1, 1)
            .add(1)
            .div(2)
            .mul(255)
            .byte()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        buffer = io.BytesIO()
        Image.fromarray(arr).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _pick_random_frame(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 4:
            idx = np.random.randint(0, arr.shape[0])
            frame = arr[idx]
        elif arr.ndim == 3:
            frame = arr
        else:
            raise RuntimeError(f"Unsupported full_res.npy shape: {arr.shape}")

        if frame.ndim != 3:
            raise RuntimeError(f"Unsupported frame shape: {frame.shape}")
        if frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
            frame = np.transpose(frame, (1, 2, 0))

        if frame.dtype != np.uint8:
            f = frame.astype(np.float32)
            if f.min() >= -1.0 and f.max() <= 1.0:
                f = (f + 1.0) * 127.5
            elif f.min() >= 0.0 and f.max() <= 1.0:
                f = f * 255.0
            frame = np.clip(f, 0.0, 255.0).astype(np.uint8)
        return frame

    def _frame_to_data_url(self, frame: np.ndarray) -> str:
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def _encode_action_vector(self, action: Dict[str, Any], device: torch.device) -> torch.Tensor:
        keys = torch.zeros(11, dtype=torch.float32, device=device)
        if action.get("w"):
            keys[0] = 1
        if action.get("a"):
            keys[1] = 1
        if action.get("s"):
            keys[2] = 1
        if action.get("d"):
            keys[3] = 1
        if action.get("space"):
            keys[4] = 1
        if action.get("ctrl"):
            keys[5] = 1
        if action.get("shift"):
            keys[6] = 1
        weapon = int(action.get("weapon", 0) or 0)
        if weapon == 1:
            keys[7] = 1
        elif weapon == 2:
            keys[8] = 1
        elif weapon == 3:
            keys[9] = 1
        if action.get("reload"):
            keys[10] = 1

        l_click = torch.tensor([1.0 if action.get("l_click") else 0.0], device=device)
        r_click = torch.tensor([1.0 if action.get("r_click") else 0.0], device=device)

        raw_x, raw_y = self._mouse_raw_from_unified(action)
        mx = min(MOUSE_X_POSSIBLES, key=lambda v: abs(v - raw_x))
        my = min(MOUSE_Y_POSSIBLES, key=lambda v: abs(v - raw_y))

        mouse_x = torch.zeros(len(MOUSE_X_POSSIBLES), dtype=torch.float32, device=device)
        mouse_y = torch.zeros(len(MOUSE_Y_POSSIBLES), dtype=torch.float32, device=device)
        mouse_x[MOUSE_X_POSSIBLES.index(mx)] = 1
        mouse_y[MOUSE_Y_POSSIBLES.index(my)] = 1

        return torch.cat([keys, l_click, r_click, mouse_x, mouse_y], dim=0)

    def _movement_only_repeatable(self, action: Dict[str, Any]) -> bool:
        """Repeat env steps only for pure WASD (no look). Camera / mouse deltas stay single-step."""
        dx = float(action.get("camera_dx", 0.0) or 0.0)
        dy = float(action.get("camera_dy", 0.0) or 0.0)
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            return False
        return bool(
            action.get("w")
            or action.get("a")
            or action.get("s")
            or action.get("d")
        )

    def _mouse_raw_from_unified(self, action: Dict[str, Any]) -> Tuple[float, float]:
        # Match unified sweep: cu/cd/cl/cr map to camera_dy / camera_dx.
        # Previous sign flipped pitch and yaw vs user expectation.
        raw_x = float(action.get("camera_dx", 0.0) or 0.0) * self.mouse_gain_x
        raw_y = float(action.get("camera_dy", 0.0) or 0.0) * self.mouse_gain_y
        return raw_x, raw_y


svc = DiamondRuntimeService()
app = FastAPI(title="WMFactory Diamond Service")


@app.post("/health")
def health() -> Dict[str, Any]:
    return svc.health()


@app.post("/load")
def load(_: LoadRequest) -> Dict[str, Any]:
    try:
        return svc.load()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/sessions/start")
def start(req: StartRequest) -> Dict[str, Any]:
    try:
        return svc.start_session(req.init_image_base64)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/sessions/step")
def step(req: StepRequest) -> Dict[str, Any]:
    try:
        return svc.step(req.session_id, req.action)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/sessions/reset")
def reset(req: ResetRequest) -> Dict[str, Any]:
    try:
        return svc.reset_session(req.session_id, req.init_image_base64)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/datasets/random-image")
def random_image(req: RandomImageRequest) -> Dict[str, Any]:
    try:
        return svc.random_spawn_image(req.dataset_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
