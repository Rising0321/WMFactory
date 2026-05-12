from __future__ import annotations

import base64
from contextlib import nullcontext
import io
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torchvision.transforms.functional as TF
from fastapi import FastAPI, HTTPException
from omegaconf import OmegaConf
from PIL import Image
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VID2WORLD_ROOT = PROJECT_ROOT / "vendors" / "vid2world"
VID2WORLD_CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "vid2world"
OPENCLIP_HF_CACHE_ROOT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K"
)

if str(VID2WORLD_ROOT) not in sys.path:
    sys.path.insert(0, str(VID2WORLD_ROOT))
if str(VID2WORLD_ROOT / "main") not in sys.path:
    sys.path.insert(0, str(VID2WORLD_ROOT / "main"))

from utils.utils import instantiate_from_config  # type: ignore
from main.utils_train import load_checkpoints, load_checkpoints_causal  # type: ignore


DEFAULT_CONFIG = VID2WORLD_ROOT / "configs" / "game" / "config_csgo_test_long_rollout.yaml"
DEFAULT_CKPT = VID2WORLD_CHECKPOINT_ROOT / "Vid2World-CSGO" / "model_checkpoint_100000.ckpt"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "csgo_processed_min" / "full_res"
DEFAULT_RESOLUTION = (320, 512)
MOUSE_X_POSSIBLES = [
    -1000,
    -500,
    -300,
    -200,
    -100,
    -60,
    -30,
    -20,
    -10,
    -4,
    -2,
    0,
    2,
    4,
    10,
    20,
    30,
    60,
    100,
    200,
    300,
    500,
    1000,
]
MOUSE_Y_POSSIBLES = [-200, -100, -50, -20, -10, -4, -2, 0, 2, 4, 10, 20, 50, 100, 200]


class LoadRequest(BaseModel):
    model_id: Optional[str] = "vid2world"


class StartRequest(BaseModel):
    init_image_base64: Optional[str] = None
    seed_meta: Optional[Dict[str, Any]] = None


class StepRequest(BaseModel):
    session_id: str
    action: Dict[str, Any]


class ResetRequest(BaseModel):
    session_id: str
    init_image_base64: Optional[str] = None
    seed_meta: Optional[Dict[str, Any]] = None


class RandomImageRequest(BaseModel):
    dataset_id: str = "CSGO"


@dataclass
class SessionState:
    session_id: str
    video_history: torch.Tensor
    action_history: torch.Tensor
    init_frame_base64: str
    synthetic_history: bool
    generated_steps: int


@dataclass
class Runtime:
    model: Any
    device: torch.device
    history_steps: int
    action_dim: int
    ddim_steps: int
    timestep_spacing: str
    guidance_scale: float
    guidance_rescale: float
    fps: int
    mouse_gain_x: float
    mouse_gain_y: float
    session: Optional[SessionState] = None


class Vid2WorldRuntimeService:
    def __init__(self) -> None:
        os.environ.setdefault("HF_ENDPOINT", os.getenv("WM_HF_ENDPOINT", "https://hf-mirror.com"))
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = ""
        os.environ["HTTP_PROXY"] = ""
        os.environ["HTTPS_PROXY"] = ""
        if self._has_local_openclip_cache():
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        self.runtime: Optional[Runtime] = None
        self._step_lock = threading.Lock()

    def health(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "model_id": "vid2world",
            "ready": self.runtime is not None,
            "session_id": None if self.runtime is None or self.runtime.session is None else self.runtime.session.session_id,
        }

    def load(self) -> Dict[str, Any]:
        if self.runtime is not None:
            return {
                "model_id": "vid2world",
                "status": "already_loaded",
                "device": str(self.runtime.device),
                "ddim_steps": self.runtime.ddim_steps,
                "guidance_scale": self.runtime.guidance_scale,
            }

        if not torch.cuda.is_available():
            raise RuntimeError("Vid2World requires CUDA GPU")

        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

        config_path = Path(os.getenv("WM_VID2WORLD_CONFIG", str(DEFAULT_CONFIG)))
        ckpt_path = Path(os.getenv("WM_VID2WORLD_CKPT", str(DEFAULT_CKPT)))
        if not config_path.exists():
            raise RuntimeError(f"Missing Vid2World config: {config_path}")
        if not ckpt_path.exists():
            raise RuntimeError(f"Missing Vid2World checkpoint: {ckpt_path}")

        config = OmegaConf.load(config_path)
        config.model.pretrained_checkpoint = str(ckpt_path)

        device = torch.device(os.getenv("WM_VID2WORLD_DEVICE", "cuda:0"))
        model = instantiate_from_config(config.model)
        if config.model.params.unet_config.params.use_causal_attention:
            model = load_checkpoints_causal(model, config.model)
        else:
            model = load_checkpoints(model, config.model)
        if model.rescale_betas_zero_snr:
            model.register_schedule(
                given_betas=model.given_betas,
                beta_schedule=model.beta_schedule,
                timesteps=model.timesteps,
                linear_start=model.linear_start,
                linear_end=model.linear_end,
                cosine_s=model.cosine_s,
            )
        model = model.to(device).eval()

        history_steps = int(os.getenv("WM_VID2WORLD_HISTORY_STEPS", "9"))
        ddim_steps = int(os.getenv("WM_VID2WORLD_DDIM_STEPS", "50"))
        timestep_spacing = os.getenv("WM_VID2WORLD_TIMESTEP_SPACING", "uniform_trailing")
        guidance_scale = float(os.getenv("WM_VID2WORLD_GUIDANCE_SCALE", "2.5"))
        guidance_rescale = float(os.getenv("WM_VID2WORLD_GUIDANCE_RESCALE", "0.7"))
        fps = int(os.getenv("WM_VID2WORLD_FPS", "3"))
        mouse_gain_x = float(os.getenv("WM_VID2WORLD_MOUSE_GAIN_X", "200"))
        mouse_gain_y = float(os.getenv("WM_VID2WORLD_MOUSE_GAIN_Y", "100"))
        action_dim = int(config.model.params.unet_config.params.action_dim)

        self.runtime = Runtime(
            model=model,
            device=device,
            history_steps=history_steps,
            action_dim=action_dim,
            ddim_steps=ddim_steps,
            timestep_spacing=timestep_spacing,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            fps=fps,
            mouse_gain_x=mouse_gain_x,
            mouse_gain_y=mouse_gain_y,
        )
        return {
            "model_id": "vid2world",
            "status": "loaded",
            "device": str(device),
            "ddim_steps": ddim_steps,
            "timestep_spacing": timestep_spacing,
            "guidance_scale": guidance_scale,
            "guidance_rescale": guidance_rescale,
        }

    def start_session(self, init_image_base64: Optional[str], seed_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        runtime = self._require_runtime()
        synthetic_history = False
        if seed_meta is not None:
            history, action_history = self._history_and_actions_from_seed_meta(seed_meta, runtime.history_steps)
            frame_b64 = self._frame_to_base64(history[:, -1])
        else:
            init_bytes = self._decode_image(init_image_base64)
            if init_bytes is None:
                history, action_history, frame_b64 = self._random_history_clip(runtime)
            else:
                history = self._history_from_image_bytes(init_bytes, runtime.history_steps)
                action_history = self._neutral_action(runtime.device).unsqueeze(0).repeat(runtime.history_steps, 1).cpu()
                frame_b64 = self._frame_to_base64(history[:, -1])
                synthetic_history = True

        runtime.session = SessionState(
            session_id=str(uuid.uuid4()),
            video_history=history.unsqueeze(0).to(runtime.device, non_blocking=True),
            action_history=action_history.unsqueeze(0).to(runtime.device, non_blocking=True),
            init_frame_base64=frame_b64,
            synthetic_history=synthetic_history,
            generated_steps=0,
        )
        return {"session_id": runtime.session.session_id, "frame_base64": frame_b64}

    def reset_session(self, session_id: str, init_image_base64: Optional[str], seed_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        runtime = self._require_runtime()
        session = self._require_session(runtime, session_id)
        synthetic_history = False
        if seed_meta is not None:
            history, action_history = self._history_and_actions_from_seed_meta(seed_meta, runtime.history_steps)
            frame_b64 = self._frame_to_base64(history[:, -1])
        else:
            init_bytes = self._decode_image(init_image_base64)
            if init_bytes is None:
                history, action_history, frame_b64 = self._random_history_clip(runtime)
            else:
                history = self._history_from_image_bytes(init_bytes, runtime.history_steps)
                action_history = self._neutral_action(runtime.device).unsqueeze(0).repeat(runtime.history_steps, 1).cpu()
                frame_b64 = self._frame_to_base64(history[:, -1])
                synthetic_history = True

        session.video_history = history.unsqueeze(0).to(runtime.device, non_blocking=True)
        session.action_history = action_history.unsqueeze(0).to(runtime.device, non_blocking=True)
        session.init_frame_base64 = frame_b64
        session.synthetic_history = synthetic_history
        session.generated_steps = 0
        return {"session_id": session.session_id, "frame_base64": frame_b64}

    @torch.inference_mode()
    def step(self, session_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        runtime = self._require_runtime()
        session = self._require_session(runtime, session_id)

        use_autocast = os.getenv("WM_VID2WORLD_USE_AUTOCAST", "0") == "1"
        autocast_context = torch.amp.autocast("cuda", dtype=torch.float16) if use_autocast else nullcontext()
        with self._step_lock, autocast_context:
            started = time.perf_counter()
            next_action = self._map_action(action, runtime).view(1, 1, -1)
            placeholder = session.video_history[:, :, -1:, :, :]
            step_video = torch.cat([session.video_history, placeholder], dim=2)
            step_action = torch.cat([session.action_history, next_action], dim=1)
            step_batch = {
                "video": step_video,
                "action": step_action,
                "caption": [""],
                "path": ["vid2world_session"],
                "fps": torch.tensor([runtime.fps], device=runtime.device),
                "frame_stride": torch.tensor([1], device=runtime.device),
            }
            logs = runtime.model.log_images(
                step_batch,
                sample=True,
                ddim_steps=runtime.ddim_steps,
                unconditional_guidance_scale=runtime.guidance_scale,
                ar=True,
                ar_noise_schedule=1,
                cond_frame=runtime.history_steps,
                guidance_rescale=runtime.guidance_rescale,
                sampled_img_num=1,
                timestep_spacing=runtime.timestep_spacing,
            )
            next_frame = logs["samples"][:, :, -1:, :, :]
            session.video_history = torch.cat([session.video_history[:, :, 1:, :, :], next_frame], dim=2)
            session.action_history = torch.cat([session.action_history[:, 1:, :], next_action], dim=1)
            session.generated_steps += 1
            latency_ms = int((time.perf_counter() - started) * 1000)

        return {
            "session_id": session.session_id,
            "frame_base64": self._frame_to_base64(next_frame[0, :, 0]),
            "reward": 0.0,
            "ended": False,
            "truncated": False,
            "extra": {
                "latency_ms": latency_ms,
                "ddim_steps": runtime.ddim_steps,
                "timestep_spacing": runtime.timestep_spacing,
                "guidance_scale": runtime.guidance_scale,
                "synthetic_history": session.synthetic_history,
                "generated_steps": session.generated_steps,
            },
        }

    def random_dataset_image(self, dataset_id: str) -> Dict[str, Any]:
        if dataset_id.lower() != "csgo":
            raise RuntimeError(f"Unsupported dataset for Vid2World: {dataset_id}")
        history_steps = self.runtime.history_steps if self.runtime is not None else int(os.getenv("WM_VID2WORLD_HISTORY_STEPS", "9"))
        history, _, meta = self._random_history_clip_for_length(history_steps)
        preview = history[:, -1]
        return {
            "dataset_id": "CSGO",
            "file": str(meta["file"]),
            "image_base64": f"data:image/png;base64,{self._frame_to_base64(preview)}",
            "extra": {"seed_meta": meta},
        }

    def _random_history_clip(self, runtime: Runtime) -> Tuple[torch.Tensor, torch.Tensor, str]:
        history, actions, _ = self._random_history_clip_for_length(runtime.history_steps)
        return history, actions, self._frame_to_base64(history[:, -1])

    def _random_history_clip_for_length(self, history_steps: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        data_dir = Path(os.getenv("WM_VID2WORLD_DATA_DIR", str(DEFAULT_DATA_DIR)))
        files = sorted(p for p in data_dir.rglob("*.hdf5") if p.is_file())
        if not files:
            raise RuntimeError(f"No CSGO hdf5 files found under {data_dir}")
        path = files[np.random.randint(0, len(files))]
        max_start = 1000 - history_steps
        start_idx = int(np.random.randint(0, max_start + 1))
        history, actions = self._load_history_actions(path, start_idx, history_steps)
        return history, actions, {"file": str(path), "start_idx": int(start_idx)}

    def _history_and_actions_from_seed_meta(self, seed_meta: Dict[str, Any], history_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = Path(str(seed_meta["file"]))
        start_idx = int(seed_meta["start_idx"])
        if not path.exists():
            raise RuntimeError(f"Seed meta file does not exist: {path}")
        return self._load_history_actions(path, start_idx, history_steps)

    def _load_history_actions(self, path: Path, start_idx: int, history_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        with h5py.File(path, "r") as handle:
            frames = [
                self._preprocess_csgo_frame(handle[f"frame_{idx}_x"][()])
                for idx in range(start_idx, start_idx + history_steps)
            ]
            actions = [
                torch.from_numpy(np.asarray(handle[f"frame_{idx}_y"][()], dtype=np.float32))
                for idx in range(start_idx, start_idx + history_steps)
            ]
        return torch.stack(frames, dim=1), torch.stack(actions, dim=0)

    def _history_from_image_bytes(self, image_bytes: bytes, history_steps: int) -> torch.Tensor:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self._preprocess_image(image)
        return tensor.unsqueeze(1).repeat(1, history_steps, 1, 1)

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image, dtype=np.float32)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        tensor = TF.resize(tensor, [275, 512], antialias=True)
        tensor = TF.center_crop(tensor, list(DEFAULT_RESOLUTION))
        tensor = (tensor / 255.0 - 0.5) * 2.0
        return tensor

    def _preprocess_csgo_frame(self, frame: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(frame.astype(np.float32)).flip(2).permute(2, 0, 1)
        tensor = TF.resize(tensor, [275, 512], antialias=True)
        tensor = TF.center_crop(tensor, list(DEFAULT_RESOLUTION))
        tensor = (tensor / 255.0 - 0.5) * 2.0
        return tensor

    def _neutral_action(self, device: torch.device) -> torch.Tensor:
        action = torch.zeros(51, dtype=torch.float32, device=device)
        action[13 + 11] = 1.0
        action[13 + 23 + 7] = 1.0
        return action

    def _map_action(self, action: Dict[str, Any], runtime: Runtime) -> torch.Tensor:
        result = self._neutral_action(runtime.device)

        if action.get("w"):
            result[0] = 1.0
        if action.get("a"):
            result[1] = 1.0
        if action.get("s"):
            result[2] = 1.0
        if action.get("d"):
            result[3] = 1.0
        if action.get("space"):
            result[4] = 1.0
        if action.get("ctrl"):
            result[5] = 1.0
        if action.get("shift"):
            result[6] = 1.0

        weapon = int(action.get("weapon", 0) or 0)
        if weapon in (1, 2, 3):
            result[6 + weapon] = 1.0
        if action.get("reload"):
            result[10] = 1.0

        result[11] = 1.0 if action.get("l_click") else 0.0
        result[12] = 1.0 if action.get("r_click") else 0.0

        dx = float(action.get("camera_dx", 0.0) or 0.0)
        dy = float(action.get("camera_dy", 0.0) or 0.0)
        mouse_x = self._nearest_bucket(dx * runtime.mouse_gain_x, MOUSE_X_POSSIBLES)
        mouse_y = self._nearest_bucket(dy * runtime.mouse_gain_y, MOUSE_Y_POSSIBLES)
        result[13:13 + len(MOUSE_X_POSSIBLES)] = 0.0
        result[13 + len(MOUSE_X_POSSIBLES):] = 0.0
        result[13 + MOUSE_X_POSSIBLES.index(mouse_x)] = 1.0
        result[13 + len(MOUSE_X_POSSIBLES) + MOUSE_Y_POSSIBLES.index(mouse_y)] = 1.0
        return result

    def _nearest_bucket(self, value: float, buckets: List[int]) -> int:
        return min(buckets, key=lambda bucket: abs(bucket - value))

    def _frame_to_base64(self, frame: torch.Tensor) -> str:
        array = (
            frame.detach()
            .float()
            .clamp(-1.0, 1.0)
            .add(1.0)
            .div(2.0)
            .mul(255.0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        buf = io.BytesIO()
        Image.fromarray(array).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _decode_image(self, payload: Optional[str]) -> Optional[bytes]:
        if not payload:
            return None
        if "," in payload:
            payload = payload.split(",", 1)[1]
        return base64.b64decode(payload)

    def _require_runtime(self) -> Runtime:
        if self.runtime is None:
            raise RuntimeError("Model is not loaded. Call /load first.")
        return self.runtime

    def _require_session(self, runtime: Runtime, session_id: str) -> SessionState:
        if runtime.session is None:
            raise RuntimeError("Session is not started. Call /sessions/start first.")
        if runtime.session.session_id != session_id:
            raise RuntimeError("Unknown or expired session_id")
        return runtime.session

    def _has_local_openclip_cache(self) -> bool:
        snapshots_dir = OPENCLIP_HF_CACHE_ROOT / "snapshots"
        if not snapshots_dir.exists():
            return False
        return any((snapshot / "open_clip_pytorch_model.bin").exists() for snapshot in snapshots_dir.iterdir())


service = Vid2WorldRuntimeService()
app = FastAPI(title="Vid2World Service")


@app.post("/health")
def health() -> Dict[str, Any]:
    return service.health()


@app.post("/load")
def load(_: LoadRequest) -> Dict[str, Any]:
    try:
        return service.load()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/sessions/start")
def start(req: StartRequest) -> Dict[str, Any]:
    try:
        return service.start_session(req.init_image_base64, seed_meta=req.seed_meta)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/sessions/reset")
def reset(req: ResetRequest) -> Dict[str, Any]:
    try:
        return service.reset_session(req.session_id, req.init_image_base64, seed_meta=req.seed_meta)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/sessions/step")
def step(req: StepRequest) -> Dict[str, Any]:
    try:
        return service.step(req.session_id, req.action)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/datasets/random-image")
def random_image(req: RandomImageRequest) -> Dict[str, Any]:
    try:
        return service.random_dataset_image(req.dataset_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
