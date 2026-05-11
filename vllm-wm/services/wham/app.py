from __future__ import annotations

import base64
import io
import os
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from tensordict import TensorDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WHAM_ROOT = PROJECT_ROOT / "vendors" / "wham"
WHAM_CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "wham"

if str(WHAM_ROOT) not in sys.path:
    sys.path.insert(0, str(WHAM_ROOT))

from wham.utils import POS_BINS_BOUNDARIES, load_model_from_checkpoint  # type: ignore


class LoadRequest(BaseModel):
    model_id: Optional[str] = "wham"


class StartRequest(BaseModel):
    init_image_base64: Optional[str] = None


class StepRequest(BaseModel):
    session_id: str
    action: Dict[str, Any]


class ResetRequest(BaseModel):
    session_id: str
    init_image_base64: Optional[str] = None


@dataclass
class SessionState:
    session_id: str
    seed_image_bytes: bytes
    # [1, T, 3, 180, 300], uint8, cuda
    context_images: torch.Tensor
    # [1, T, 16], float32 (buttons + stick bins), cuda
    context_actions: torch.Tensor
    # len(T), each item is either None or token list.
    context_tokens: list[Optional[list[int]]]
    step_index: int = 0


@dataclass
class Runtime:
    model: Any
    device: torch.device
    session: Optional[SessionState] = None


class WhamRuntimeService:
    def __init__(self) -> None:
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = ""
        os.environ["HTTP_PROXY"] = ""
        os.environ["HTTPS_PROXY"] = ""
        os.environ.setdefault("HF_ENDPOINT", os.getenv("WM_HF_ENDPOINT", "https://hf-mirror.com"))
        # WHAM checkpoints are trusted local assets. PyTorch 2.6+ defaults
        # torch.load(..., weights_only=True), which breaks older Lightning
        # checkpoints with pickled numpy metadata.
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

        self.runtime: Optional[Runtime] = None
        self._lock = threading.RLock()

        self.model_path = Path(
            os.getenv("WM_WHAM_MODEL_PATH", str(WHAM_CHECKPOINT_ROOT / "WHAM_200M.ckpt"))
        )
        self.gpu_index = int(os.getenv("WM_WHAM_GPU_INDEX", "0"))
        self.context_length = int(os.getenv("WM_WHAM_CONTEXT_LENGTH", "10"))
        self.temperature = float(os.getenv("WM_WHAM_TEMPERATURE", "0.9"))
        self.top_p = float(os.getenv("WM_WHAM_TOP_P", "1.0"))
        self.step_log_every = int(os.getenv("WM_WHAM_STEP_LOG_EVERY", "20"))
        self._step_counter = 0

    def _log(self, message: str) -> None:
        print(f"[service][wham] {message}", flush=True)

    def health(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "ok": True,
                "model_id": "wham",
                "ready": self.runtime is not None,
                "session_id": None if self.runtime is None or self.runtime.session is None else self.runtime.session.session_id,
                "context_length": self.context_length,
                "model_path": str(self.model_path),
            }

    def load(self) -> Dict[str, Any]:
        with self._lock:
            self._log("load requested")
            if self.runtime is not None:
                return {
                    "model_id": "wham",
                    "status": "already_loaded",
                    "device": str(self.runtime.device),
                    "model_path": str(self.model_path),
                }

            if not self.model_path.exists():
                raise RuntimeError(f"WHAM checkpoint not found: {self.model_path}")
            if not torch.cuda.is_available():
                raise RuntimeError("WHAM requires CUDA GPU")

            device = torch.device(f"cuda:{self.gpu_index}")
            torch.cuda.set_device(device)
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            model = load_model_from_checkpoint(self.model_path).to(device).eval()

            self.runtime = Runtime(model=model, device=device, session=None)
            self._log(f"load done device={device}")
            return {
                "model_id": "wham",
                "status": "loaded",
                "device": str(device),
                "model_path": str(self.model_path),
                "context_length": self.context_length,
                "sampling": {
                    "temperature": self.temperature,
                    "top_k": None,
                    "top_p": self.top_p,
                },
            }

    def start_session(self, init_image_base64: Optional[str]) -> Dict[str, Any]:
        with self._lock:
            runtime = self._require_runtime()
            init_image_bytes = self._decode_image(init_image_base64)
            if init_image_bytes is None:
                raise RuntimeError("init_image_base64 is required for wham start")

            image_t = self._bytes_to_model_image_tensor(init_image_bytes, runtime.device)  # [3,H,W], uint8
            action0 = self._to_wham_action_tensor(self._map_unified_action_to_wham_action(self._neutral_action()), runtime.device)  # [16], float

            context_images = image_t.unsqueeze(0).unsqueeze(0).repeat(1, self.context_length, 1, 1, 1).contiguous()
            context_actions = action0.unsqueeze(0).unsqueeze(0).repeat(1, self.context_length, 1).contiguous()
            context_tokens: list[Optional[list[int]]] = [None for _ in range(self.context_length)]

            session_id = str(uuid.uuid4())
            runtime.session = SessionState(
                session_id=session_id,
                seed_image_bytes=init_image_bytes,
                context_images=context_images,
                context_actions=context_actions,
                context_tokens=context_tokens,
                step_index=0,
            )

            self._log(f"start_session done session_id={session_id}")
            return {"session_id": session_id, "frame_base64": self._tensor_frame_to_png_base64(image_t)}

    def reset_session(self, session_id: str, init_image_base64: Optional[str]) -> Dict[str, Any]:
        with self._lock:
            runtime = self._require_runtime()
            session = self._require_session(runtime, session_id)

            init_image_bytes = self._decode_image(init_image_base64) or session.seed_image_bytes
            # Reuse start path for deterministic reset behavior.
            return self.start_session(self._encode_image(init_image_bytes))

    @torch.inference_mode()
    def step(self, session_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            runtime = self._require_runtime()
            session = self._require_session(runtime, session_id)

            self._step_counter += 1
            if self._step_counter % max(1, self.step_log_every) == 0:
                self._log(f"step #{self._step_counter} session={session_id[:8]}")

            input_action = self._map_unified_action_to_wham_action(action)
            input_action_t = self._to_wham_action_tensor(input_action, runtime.device).unsqueeze(0).unsqueeze(0)  # [1,1,16]
            context_actions = session.context_actions[:, : self.context_length].clone()
            # Apply the current input action to the current prediction instead of
            # waiting one extra step to roll it into the context.
            context_actions[:, -1:] = input_action_t

            context_data = TensorDict(
                {
                    "images": session.context_images[:, : self.context_length],
                    "actions_output": context_actions,
                },
                batch_size=(1, self.context_length),
            )
            predicted_step, predicted_tokens = runtime.model.predict_next_step(
                context_data,
                tokens=[session.context_tokens],
                temperature=self.temperature,
                top_k=None,
                top_p=self.top_p,
                min_tokens_to_keep=1,
            )

            dreamt_image = predicted_step["images"][:, :1]  # [1,1,3,H,W]
            action_to_append = input_action_t

            if session.context_images.shape[1] >= self.context_length:
                session.context_images = session.context_images[:, 1:]
                session.context_actions = session.context_actions[:, 1:]
                session.context_tokens = session.context_tokens[1:]

            session.context_images = torch.cat([session.context_images, dreamt_image], dim=1)
            session.context_actions = torch.cat([session.context_actions, action_to_append], dim=1)

            if isinstance(predicted_tokens, torch.Tensor):
                token_list = predicted_tokens[0, 0].detach().cpu().tolist()
            else:
                token_list = []
            session.context_tokens.append(token_list)
            session.step_index += 1

            frame = dreamt_image[0, 0]
            return {
                "session_id": session_id,
                "frame_base64": self._tensor_frame_to_png_base64(frame),
                "reward": 0.0,
                "ended": False,
                "truncated": False,
                "extra": {
                    "step_index": session.step_index,
                    "context_length": self.context_length,
                },
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

    def _encode_image(self, image_bytes: bytes) -> str:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def _bytes_to_model_image_tensor(self, image_bytes: bytes, device: torch.device) -> torch.Tensor:
        # Follow WHAM run_server preprocessing: RGB -> resize 300x180 -> CHW uint8.
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((300, 180), resample=Image.BICUBIC)
        arr = np.asarray(img, dtype=np.uint8).copy()  # HWC
        chw = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(chw).to(device=device, dtype=torch.uint8)

    def _tensor_frame_to_png_base64(self, frame: torch.Tensor) -> str:
        t = frame.detach().cpu()
        if t.ndim == 4:
            t = t[0]
        if t.dtype == torch.uint8:
            arr = t.permute(1, 2, 0).numpy()
        else:
            a = t.float()
            amin = float(a.min().item())
            amax = float(a.max().item())
            if amin >= -1.01 and amax <= 1.01:
                a = (a + 1.0) * 127.5
            elif amin >= 0.0 and amax <= 1.01:
                a = a * 255.0
            arr = a.clamp(0, 255).byte().permute(1, 2, 0).numpy()
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _neutral_action(self) -> Dict[str, Any]:
        return {
            "w": False,
            "a": False,
            "s": False,
            "d": False,
            "camera_dx": 0.0,
            "camera_dy": 0.0,
            "l_click": False,
            "r_click": False,
            "space": False,
            "ctrl": False,
            "shift": False,
            "weapon": 0,
            "reload": False,
        }

    def _map_unified_action_to_wham_action(self, action: Dict[str, Any]) -> np.ndarray:
        # WHAM expects 16 values: 12 buttons + 4 analog sticks.
        out = np.zeros(16, dtype=np.float32)

        # Button mapping (heuristic, keeps unified controls semantics).
        out[0] = 1.0 if bool(action.get("l_click", False)) else 0.0
        out[1] = 1.0 if bool(action.get("r_click", False)) else 0.0
        out[2] = 1.0 if bool(action.get("space", False)) else 0.0
        out[3] = 1.0 if bool(action.get("ctrl", False)) else 0.0
        out[4] = 1.0 if bool(action.get("shift", False)) else 0.0
        out[5] = 1.0 if bool(action.get("reload", False)) else 0.0

        weapon = int(action.get("weapon", 0) or 0)
        if weapon == 1:
            out[6] = 1.0
        elif weapon == 2:
            out[7] = 1.0
        elif weapon == 3:
            out[8] = 1.0

        # Left stick from WASD.
        left_x = (1.0 if bool(action.get("d", False)) else 0.0) - (1.0 if bool(action.get("a", False)) else 0.0)
        left_y = (1.0 if bool(action.get("w", False)) else 0.0) - (1.0 if bool(action.get("s", False)) else 0.0)
        right_x = float(action.get("camera_dx", 0.0) or 0.0)
        right_y = -float(action.get("camera_dy", 0.0) or 0.0)

        out[12] = float(np.clip(left_x, -1.0, 1.0))
        out[13] = float(np.clip(left_y, -1.0, 1.0))
        out[14] = float(np.clip(right_x, -1.0, 1.0))
        out[15] = float(np.clip(right_y, -1.0, 1.0))
        return out

    def _to_wham_action_tensor(self, action16: np.ndarray, device: torch.device) -> torch.Tensor:
        arr = action16.copy()
        # Match WHAM run_server preprocessing exactly: discretize last 4 stick dimensions to 11 bins.
        arr[-4:] = np.digitize(arr[-4:], bins=POS_BINS_BOUNDARIES) - 1
        return torch.from_numpy(arr).to(device=device, dtype=torch.float32)


svc = WhamRuntimeService()
app = FastAPI(title="WMFactory WHAM Service")


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
