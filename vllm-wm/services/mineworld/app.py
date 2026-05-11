from __future__ import annotations

import base64
import io
import os
import sys
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Optional

import cv2
import numpy as np
import torch
from einops import rearrange
from fastapi import FastAPI, HTTPException
from omegaconf import OmegaConf
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MINEWORLD_ROOT = Path(os.getenv("WM_MINEWORLD_ROOT", str(PROJECT_ROOT / "vendors" / "mineworld"))).resolve()
MINEWORLD_CHECKPOINT_ROOT = Path(
    os.getenv("WM_MINEWORLD_CHECKPOINT_ROOT", str(PROJECT_ROOT / "checkpoints" / "mineworld"))
).resolve()
MINEWORLD_DATA_ROOT = Path(os.getenv("WM_MINEWORLD_DATA_ROOT", str(PROJECT_ROOT / "data" / "mineworld"))).resolve()
if str(MINEWORLD_ROOT) not in sys.path:
    sys.path.insert(0, str(MINEWORLD_ROOT))

from mcdataset import MCDataset  # type: ignore
from utils import load_model  # type: ignore


class LoadRequest(BaseModel):
    model_id: Optional[str] = "mineworld"


class StartRequest(BaseModel):
    init_image_base64: Optional[str] = None


class StepRequest(BaseModel):
    session_id: str
    action: Dict[str, Any]


class ResetRequest(BaseModel):
    session_id: str
    init_image_base64: Optional[str] = None


class RandomImageRequest(BaseModel):
    dataset_id: str = "minecraft"


@dataclass
class SessionState:
    session_id: str
    frame_cache: Deque[torch.Tensor]
    action_cache: Deque[torch.Tensor]
    last_pos: int
    last_frame_b64: str
    selected_weapon: Optional[int] = None


@dataclass
class Runtime:
    model: Any
    tokenizer: Any
    action_map: MCDataset
    context_len: int
    reference_frame: int
    diagd: bool
    window_size: int
    session: Optional[SessionState] = None


class MineWorldRuntimeService:
    AGENT_RESOLUTION = (384, 224)  # (w, h)
    TOKEN_PER_IMAGE = 336
    TOKEN_PER_ACTION = 11

    def __init__(self) -> None:
        self.runtime: Optional[Runtime] = None
        self.scene_path = Path(
            os.getenv("WM_MINEWORLD_SCENE", str(MINEWORLD_DATA_ROOT / "gradio_scene" / "scene.mp4"))
        )
        self.model_ckpt = Path(
            os.getenv("WM_MINEWORLD_CKPT", str(MINEWORLD_CHECKPOINT_ROOT / "1200M_32f.ckpt"))
        )
        self.config_path = Path(
            os.getenv("WM_MINEWORLD_CONFIG", str(MINEWORLD_ROOT / "configs" / "1200M_32f.yaml"))
        )
        self.reference_frame = int(os.getenv("WM_MINEWORLD_REFERENCE_FRAME", "8"))
        self.diagd = os.getenv("WM_MINEWORLD_DIAGD", "1") == "1"
        self.window_size = int(os.getenv("WM_MINEWORLD_WINDOW_SIZE", "4"))

    def _log(self, msg: str) -> None:
        text = f"[service][mineworld] {msg}"
        try:
            print(text, flush=True)
        except BrokenPipeError:
            pass
        except OSError:
            pass

    def load(self) -> Dict[str, Any]:
        if self.runtime is not None:
            return {"model_id": "mineworld", "status": "already_loaded", "device": "cuda"}

        if not torch.cuda.is_available():
            raise RuntimeError("MineWorld requires CUDA for real-time inference.")
        if not self.model_ckpt.exists():
            raise RuntimeError(f"MineWorld checkpoint not found: {self.model_ckpt}")
        if not self.config_path.exists():
            raise RuntimeError(f"MineWorld config not found: {self.config_path}")

        os.chdir(str(MINEWORLD_ROOT))
        self._log(f"loading model (cwd={MINEWORLD_ROOT})...")
        config = OmegaConf.load(self.config_path)
        config.model.params.tokenizer_config.params.config_path = str(MINEWORLD_CHECKPOINT_ROOT / "vae" / "config.json")
        config.model.params.tokenizer_config.params.ckpt_path = str(MINEWORLD_CHECKPOINT_ROOT / "vae" / "vae.ckpt")
        model = load_model(config, str(self.model_ckpt), gpu=True, eval_mode=True)
        tokenizer = model.tokenizer
        context_len = int(
            config.model.params.transformer_config.params.max_position_embeddings
            / (self.TOKEN_PER_ACTION + self.TOKEN_PER_IMAGE)
        )
        if not (4 < self.reference_frame < context_len):
            raise RuntimeError(
                f"Invalid reference_frame={self.reference_frame}, should satisfy 4 < ref < {context_len}"
            )

        self.runtime = Runtime(
            model=model,
            tokenizer=tokenizer,
            action_map=MCDataset(),
            context_len=context_len,
            reference_frame=self.reference_frame,
            diagd=self.diagd,
            window_size=self.window_size,
            session=None,
        )
        self._log("model loaded")
        return {
            "model_id": "mineworld",
            "status": "loaded",
            "device": "cuda",
            "config": str(self.config_path),
            "checkpoint": str(self.model_ckpt),
            "diagd": self.diagd,
        }

    def start_session(self, init_image_base64: Optional[str]) -> Dict[str, Any]:
        runtime = self._require_runtime()
        frame = self._load_init_frame(init_image_base64)
        session = self._init_session_from_frame(runtime, frame)
        runtime.session = session
        return {"session_id": session.session_id, "frame_base64": session.last_frame_b64}

    def reset_session(self, session_id: str, init_image_base64: Optional[str]) -> Dict[str, Any]:
        runtime = self._require_runtime()
        session = self._require_session(runtime, session_id)
        frame = self._load_init_frame(init_image_base64)
        new_session = self._init_session_from_frame(runtime, frame, session_id=session.session_id)
        runtime.session = new_session
        return {"session_id": new_session.session_id, "frame_base64": new_session.last_frame_b64}

    @torch.inference_mode()
    def step(self, session_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        runtime = self._require_runtime()
        session = self._require_session(runtime, session_id)
        model = runtime.model

        if len(session.action_cache) >= runtime.context_len - 1:
            for _ in range(runtime.context_len - runtime.reference_frame):
                session.frame_cache.popleft()
                session.action_cache.popleft()
            model.transformer.refresh_kvcache()
            vis_act = self._build_vis_act(session.frame_cache, session.action_cache)
            _, session.last_pos = model.transformer.prefill_for_gradio(vis_act)
            session.last_pos = int(session.last_pos)

        ongoing_act = self._action_to_token(runtime, action, session).unsqueeze(0).to("cuda")
        session.action_cache.append(ongoing_act)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            if runtime.diagd:
                next_frame, last_pos = model.transformer.diagd_img_token_for_gradio(
                    input_action=ongoing_act,
                    position_id=session.last_pos,
                    max_new_tokens=self.TOKEN_PER_IMAGE,
                    windowsize=runtime.window_size,
                )
            else:
                next_frame, last_pos = model.transformer.decode_img_token_for_gradio(
                    input_action=ongoing_act,
                    position_id=session.last_pos,
                    max_new_tokens=self.TOKEN_PER_IMAGE + 1,
                )

        session.last_pos = int(last_pos[0] if isinstance(last_pos, (list, tuple)) else last_pos)
        next_frame_tokens = torch.cat(next_frame, dim=-1).to("cuda")
        session.frame_cache.append(next_frame_tokens)
        next_frame_np = runtime.tokenizer.token2image(next_frame_tokens)
        session.last_frame_b64 = self._frame_to_png_b64(next_frame_np)

        return {
            "session_id": session.session_id,
            "frame_base64": session.last_frame_b64,
            "reward": 0.0,
            "ended": False,
            "truncated": False,
            "extra": {"diagd": runtime.diagd, "window_size": runtime.window_size, "action": action},
        }

    def random_scene_image(self, dataset_id: str) -> Dict[str, Any]:
        frame = self._load_random_scene_frame()
        png_b64 = self._frame_to_data_url(frame)
        return {
            "dataset_id": dataset_id,
            "file": str(self.scene_path),
            "image_base64": png_b64,
        }

    def health(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "model_id": "mineworld",
            "ready": self.runtime is not None,
            "session_id": None if self.runtime is None or self.runtime.session is None else self.runtime.session.session_id,
        }

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

    def _decode_image(self, payload: Optional[str]) -> Optional[bytes]:
        if not payload:
            return None
        if "," in payload:
            payload = payload.split(",", 1)[1]
        return base64.b64decode(payload)

    def _load_init_frame(self, init_image_base64: Optional[str]) -> np.ndarray:
        img_bytes = self._decode_image(init_image_base64)
        if img_bytes is None:
            return self._load_scene_frame(25)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image = image.resize(self.AGENT_RESOLUTION, resample=Image.BICUBIC)
        return np.asarray(image, dtype=np.uint8).copy()

    def _load_scene_frame(self, frame_idx: int) -> np.ndarray:
        if not self.scene_path.exists():
            raise RuntimeError(f"Scene video not found: {self.scene_path}")
        cap = cv2.VideoCapture(str(self.scene_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {self.scene_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, self.AGENT_RESOLUTION, interpolation=cv2.INTER_LINEAR)
        return np.asarray(np.clip(frame, 0, 255), dtype=np.uint8).copy()

    def _load_random_scene_frame(self) -> np.ndarray:
        if not self.scene_path.exists():
            raise RuntimeError(f"Scene video not found: {self.scene_path}")
        cap = cv2.VideoCapture(str(self.scene_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = np.random.randint(0, max(1, total))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return self._load_scene_frame(25)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, self.AGENT_RESOLUTION, interpolation=cv2.INTER_LINEAR)
        return np.asarray(np.clip(frame, 0, 255), dtype=np.uint8).copy()

    def _frame_to_token(self, runtime: Runtime, frame: np.ndarray) -> torch.Tensor:
        frame_tensor = torch.from_numpy(frame).to("cuda")
        frame_tensor = frame_tensor.permute(2, 0, 1).float().div(255.0)
        frame_tensor = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(frame_tensor)
        frame_tensor = frame_tensor.unsqueeze(0)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            img_tokens = runtime.tokenizer.tokenize_images(frame_tensor)
        img_tokens = rearrange(img_tokens, "(b t) h w -> (b t) (h w)", b=1)
        return img_tokens[0:1].contiguous()

    def _action_to_token(self, runtime: Runtime, action: Dict[str, Any], session: SessionState) -> torch.Tensor:
        camera_scale = float(os.getenv("WM_MINEWORLD_CAMERA_SCALE", "15.0"))
        cam_x = int(round(float(action.get("camera_dx", 0.0)) * camera_scale))
        cam_y = int(round(float(action.get("camera_dy", 0.0)) * camera_scale))
        cam_x = int(np.clip(cam_x, -90, 90))
        cam_y = int(np.clip(cam_y, -90, 90))

        action_dict: Dict[str, Any] = {
            "forward": 1 if bool(action.get("w", False)) else 0,
            "back": 1 if bool(action.get("s", False)) else 0,
            "left": 1 if bool(action.get("a", False)) else 0,
            "right": 1 if bool(action.get("d", False)) else 0,
            "jump": 1 if bool(action.get("space", False)) else 0,
            "sprint": 1 if bool(action.get("ctrl", False)) else 0,
            "sneak": 1 if bool(action.get("shift", False)) else 0,
            "attack": 1 if bool(action.get("l_click", False)) else 0,
            "use": 1 if bool(action.get("r_click", False)) else 0,
            "drop": 1 if bool(action.get("reload", False)) else 0,
            "pickItem": 0,
            "swapHands": 0,
            "camera": np.array((cam_y, cam_x)),
        }
        for i in range(1, 10):
            action_dict[f"hotbar.{i}"] = 0

        weapon = action.get("weapon")
        if weapon is not None:
            try:
                w = int(weapon)
                if 0 <= w <= 8 and session.selected_weapon != w:
                    action_dict[f"hotbar.{w + 1}"] = 1
                    session.selected_weapon = w
            except Exception:
                pass

        tokens = runtime.action_map.get_action_index_from_actiondict(action_dict, action_vocab_offset=8192)
        return torch.tensor(tokens, dtype=torch.long)

    def _build_vis_act(self, frame_cache: Deque[torch.Tensor], action_cache: Deque[torch.Tensor]) -> torch.Tensor:
        frame_iter = list(frame_cache)
        act_iter = list(action_cache)
        vis_act = [torch.cat([img, act], dim=1) for img, act in zip(frame_iter[:-1], act_iter)]
        vis_act.append(frame_iter[-1])
        return torch.cat(vis_act, dim=-1)

    def _init_session_from_frame(self, runtime: Runtime, frame: np.ndarray, session_id: Optional[str] = None) -> SessionState:
        frame_token = self._frame_to_token(runtime, frame)
        noop = self._action_to_token(runtime, {}, SessionState("", deque(), deque(), 0, "")).unsqueeze(0).to("cuda")

        frame_cache: Deque[torch.Tensor] = deque([frame_token.clone() for _ in range(runtime.reference_frame)])
        action_cache: Deque[torch.Tensor] = deque([noop.clone() for _ in range(runtime.reference_frame - 1)])

        runtime.model.transformer.refresh_kvcache()
        vis_act = self._build_vis_act(frame_cache, action_cache)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            _, last_pos = runtime.model.transformer.prefill_for_gradio(vis_act)
        last_pos_int = int(last_pos)

        sid = session_id or str(uuid.uuid4())
        return SessionState(
            session_id=sid,
            frame_cache=frame_cache,
            action_cache=action_cache,
            last_pos=last_pos_int,
            last_frame_b64=self._frame_to_png_b64(frame),
            selected_weapon=None,
        )

    def _frame_to_png_b64(self, frame: np.ndarray) -> str:
        image = Image.fromarray(frame.astype(np.uint8))
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _frame_to_data_url(self, frame: np.ndarray) -> str:
        return f"data:image/png;base64,{self._frame_to_png_b64(frame)}"


service = MineWorldRuntimeService()
app = FastAPI(title="MineWorld Service")


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
        return service.start_session(req.init_image_base64)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/sessions/reset")
def reset(req: ResetRequest) -> Dict[str, Any]:
    try:
        return service.reset_session(req.session_id, req.init_image_base64)
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
        return service.random_scene_image(req.dataset_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
