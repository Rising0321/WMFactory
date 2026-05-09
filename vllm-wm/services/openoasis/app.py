from __future__ import annotations

import base64
import io
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from einops import rearrange
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from safetensors.torch import load_model
from torch import autocast
from torchvision.transforms.functional import resize
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPENOASIS_ROOT = PROJECT_ROOT / "vendors" / "openoasis"
OPENOASIS_CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "openoasis"

# Import from open-oasis repo files.
import sys

if str(OPENOASIS_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENOASIS_ROOT))

from dit import DiT_models  # type: ignore
from utils import sigmoid_beta_schedule  # type: ignore
from vae import VAE_models  # type: ignore

ACTION_KEYS = [
    "inventory",
    "ESC",
    "hotbar.1",
    "hotbar.2",
    "hotbar.3",
    "hotbar.4",
    "hotbar.5",
    "hotbar.6",
    "hotbar.7",
    "hotbar.8",
    "hotbar.9",
    "forward",
    "back",
    "left",
    "right",
    "cameraX",
    "cameraY",
    "jump",
    "sneak",
    "sprint",
    "swapHands",
    "attack",
    "use",
    "pickItem",
    "drop",
]
ACTION_IDX = {k: i for i, k in enumerate(ACTION_KEYS)}


class LoadRequest(BaseModel):
    model_id: Optional[str] = "open-oasis"


class StartRequest(BaseModel):
    init_image_base64: Optional[str] = None


class StepRequest(BaseModel):
    session_id: str
    action: Dict[str, Any]


class ResetRequest(BaseModel):
    session_id: str
    init_image_base64: Optional[str] = None


@dataclass
class Runtime:
    model: Any
    vae: Any
    device: torch.device
    scaling_factor: float
    alphas_cumprod: torch.Tensor
    ddim_steps: int
    noise_ids: list[int]
    max_frames: int
    session_id: Optional[str] = None
    # Latent history for generated frames: [B=1, T, C, H, W]
    x: Optional[torch.Tensor] = None
    # Action history aligned with frames: [B=1, T, 25], first frame uses neutral action.
    actions: Optional[torch.Tensor] = None


class OpenOasisRuntimeService:
    def __init__(self) -> None:
        self.runtime: Optional[Runtime] = None

    def health(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "model_id": "open-oasis",
            "ready": self.runtime is not None,
            "session_id": None if self.runtime is None else self.runtime.session_id,
        }

    def load(self) -> Dict[str, Any]:
        print("[service] load requested", flush=True)
        if self.runtime is not None:
            return {
                "model_id": "open-oasis",
                "status": "already_loaded",
                "device": str(self.runtime.device),
                "ddim_steps": self.runtime.ddim_steps,
            }

        if not torch.cuda.is_available():
            raise RuntimeError("open-oasis requires CUDA GPU")

        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

        device = torch.device(os.getenv("WM_OPENOASIS_DEVICE", "cuda:0"))
        oasis_ckpt = Path(
            os.getenv("WM_OPENOASIS_CKPT", str(OPENOASIS_CHECKPOINT_ROOT / "oasis500m.safetensors"))
        )
        vae_ckpt = Path(
            os.getenv("WM_OPENOASIS_VAE_CKPT", str(OPENOASIS_CHECKPOINT_ROOT / "vit-l-20.safetensors"))
        )
        # Keep native Oasis quality setting fixed.
        ddim_steps = 10

        if not oasis_ckpt.exists():
            raise RuntimeError(f"Missing DiT checkpoint: {oasis_ckpt}")
        if not vae_ckpt.exists():
            raise RuntimeError(f"Missing VAE checkpoint: {vae_ckpt}")

        model = DiT_models["DiT-S/2"]()
        load_model(model, str(oasis_ckpt))
        model = model.to(device).eval()

        vae = VAE_models["vit-l-20-shallow-encoder"]()
        load_model(vae, str(vae_ckpt))
        vae = vae.to(device).eval()

        betas = sigmoid_beta_schedule(1000).float().to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod = rearrange(alphas_cumprod, "t -> t 1 1 1")
        noise_ids = torch.linspace(-1, 999, ddim_steps + 1).long().tolist()

        self.runtime = Runtime(
            model=model,
            vae=vae,
            device=device,
            scaling_factor=0.07843137255,
            alphas_cumprod=alphas_cumprod,
            ddim_steps=ddim_steps,
            noise_ids=noise_ids,
            max_frames=int(model.max_frames),
        )
        print(f"[service] load done device={device} ddim_steps={ddim_steps}", flush=True)
        return {
            "model_id": "open-oasis",
            "status": "loaded",
            "device": str(device),
            "ddim_steps": ddim_steps,
            "oasis_ckpt": str(oasis_ckpt),
            "vae_ckpt": str(vae_ckpt),
        }

    def start_session(self, init_image_base64: Optional[str]) -> Dict[str, Any]:
        runtime = self._require_runtime()
        img_bytes = self._decode_image(init_image_base64)
        if img_bytes is None:
            raise RuntimeError("init_image_base64 is required for open-oasis session start")

        x, frame_base64 = self._encode_init_image(runtime, img_bytes)
        actions = torch.zeros((1, 1, len(ACTION_KEYS)), dtype=torch.float32, device=runtime.device)

        runtime.x = x
        runtime.actions = actions
        runtime.session_id = str(uuid.uuid4())
        return {
            "session_id": runtime.session_id,
            "frame_base64": frame_base64,
        }

    def reset_session(self, session_id: str, init_image_base64: Optional[str]) -> Dict[str, Any]:
        runtime = self._require_runtime()
        self._require_session(runtime, session_id)

        img_bytes = self._decode_image(init_image_base64)
        if img_bytes is None:
            raise RuntimeError("init_image_base64 is required for open-oasis reset")

        x, frame_base64 = self._encode_init_image(runtime, img_bytes)
        runtime.x = x
        runtime.actions = torch.zeros((1, 1, len(ACTION_KEYS)), dtype=torch.float32, device=runtime.device)
        return {
            "session_id": runtime.session_id,
            "frame_base64": frame_base64,
        }

    @torch.inference_mode()
    def step(self, session_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        runtime = self._require_runtime()
        self._require_session(runtime, session_id)
        if runtime.x is None or runtime.actions is None:
            raise RuntimeError("Session is not initialized")

        # Append current action for next generated frame.
        action_vec = self._map_action(action, runtime.device)
        runtime.actions = torch.cat([runtime.actions, action_vec], dim=1)

        x = runtime.x
        i = int(x.shape[1])  # index of frame to be generated
        start_frame = max(0, i + 1 - runtime.max_frames)

        chunk = torch.randn((1, 1, *x.shape[-3:]), device=runtime.device)
        chunk = torch.clamp(chunk, -20, 20)
        x = torch.cat([x, chunk], dim=1)

        t_ctx = torch.full((1, i), 14, dtype=torch.long, device=runtime.device)

        for noise_idx in reversed(range(1, runtime.ddim_steps + 1)):
            t_scalar = runtime.noise_ids[noise_idx]
            t_next_scalar = runtime.noise_ids[noise_idx - 1]
            if t_next_scalar < 0:
                t_next_scalar = t_scalar

            t = torch.full((1, 1), t_scalar, dtype=torch.long, device=runtime.device)
            t_next = torch.full((1, 1), t_next_scalar, dtype=torch.long, device=runtime.device)
            t = torch.cat([t_ctx, t], dim=1)[:, start_frame:]
            t_next = torch.cat([t_ctx, t_next], dim=1)[:, start_frame:]

            x_curr = x[:, start_frame:]
            with autocast("cuda", dtype=torch.half):
                v = runtime.model(x_curr, t, runtime.actions[:, start_frame : i + 1])

            x_start = runtime.alphas_cumprod[t].sqrt() * x_curr - (1 - runtime.alphas_cumprod[t]).sqrt() * v
            x_noise = ((1 / runtime.alphas_cumprod[t]).sqrt() * x_curr - x_start) / (1 / runtime.alphas_cumprod[t] - 1).sqrt()
            alpha_next = runtime.alphas_cumprod[t_next]
            alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
            if noise_idx == 1:
                alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])

            x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
            x[:, -1:] = x_pred[:, -1:]

        # Decode only newest frame for low-latency interaction.
        last = x[:, -1:]
        z = rearrange(last, "b t c h w -> (b t) (h w) c")
        frame = (runtime.vae.decode(z / runtime.scaling_factor) + 1) / 2
        frame = torch.clamp(frame, 0, 1)
        frame_base64 = self._frame_tensor_to_png_base64(frame[0])

        # Keep at most model.max_frames history to bound latency/memory.
        if x.shape[1] > runtime.max_frames:
            x = x[:, -runtime.max_frames :]
            runtime.actions = runtime.actions[:, -runtime.max_frames :]

        runtime.x = x

        return {
            "session_id": session_id,
            "frame_base64": frame_base64,
            "reward": 0.0,
            "ended": False,
            "truncated": False,
            "extra": {
                "ddim_steps": runtime.ddim_steps,
                "history_frames": int(runtime.x.shape[1]),
            },
        }

    def _require_runtime(self) -> Runtime:
        if self.runtime is None:
            raise RuntimeError("Model is not loaded. Call /load first.")
        return self.runtime

    def _require_session(self, runtime: Runtime, session_id: str) -> None:
        if runtime.session_id is None:
            raise RuntimeError("Session is not started. Call /sessions/start first.")
        if runtime.session_id != session_id:
            raise RuntimeError("Unknown or expired session_id")

    def _decode_image(self, payload: Optional[str]) -> Optional[bytes]:
        if not payload:
            return None
        if "," in payload:
            payload = payload.split(",", 1)[1]
        return base64.b64decode(payload)

    def _encode_init_image(self, runtime: Runtime, image_bytes: bytes) -> tuple[torch.Tensor, str]:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img, dtype=np.uint8, copy=True)
        t = torch.from_numpy(arr).permute(2, 0, 1)  # C,H,W
        t = resize(t, (360, 640))

        frame_base64 = self._frame_tensor_to_png_base64(t.float().div(255.0))

        x = t.unsqueeze(0).unsqueeze(0).float().div(255.0).to(runtime.device)
        h, w = x.shape[-2:]
        x = rearrange(x, "b t c h w -> (b t) c h w")
        with autocast("cuda", dtype=torch.half):
            x = runtime.vae.encode(x * 2 - 1).mean * runtime.scaling_factor
        x = rearrange(
            x,
            "(b t) (hh ww) c -> b t c hh ww",
            t=1,
            hh=h // runtime.vae.patch_size,
            ww=w // runtime.vae.patch_size,
        )
        return x, frame_base64

    def _frame_tensor_to_png_base64(self, frame_chw: torch.Tensor) -> str:
        # frame_chw in [0,1], shape C,H,W
        arr = frame_chw.detach().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _map_action(self, action: Dict[str, Any], device: torch.device) -> torch.Tensor:
        v = torch.zeros((1, 1, len(ACTION_KEYS)), dtype=torch.float32, device=device)

        if action.get("w"):
            v[0, 0, ACTION_IDX["forward"]] = 1.0
        if action.get("s"):
            v[0, 0, ACTION_IDX["back"]] = 1.0
        if action.get("a"):
            v[0, 0, ACTION_IDX["left"]] = 1.0
        if action.get("d"):
            v[0, 0, ACTION_IDX["right"]] = 1.0

        if action.get("space"):
            v[0, 0, ACTION_IDX["jump"]] = 1.0
        if action.get("ctrl"):
            v[0, 0, ACTION_IDX["sneak"]] = 1.0
        if action.get("shift"):
            v[0, 0, ACTION_IDX["sprint"]] = 1.0

        if action.get("l_click"):
            v[0, 0, ACTION_IDX["attack"]] = 1.0
        if action.get("r_click"):
            v[0, 0, ACTION_IDX["use"]] = 1.0

        weapon = int(action.get("weapon", 0) or 0)
        if 1 <= weapon <= 9:
            v[0, 0, ACTION_IDX[f"hotbar.{weapon}"]] = 1.0

        # Approximate mapping for generic controls that do not exist in Oasis action space.
        if action.get("reload"):
            v[0, 0, ACTION_IDX["drop"]] = 1.0

        dx = float(action.get("camera_dx", 0.0))
        dy = float(action.get("camera_dy", 0.0))
        camera_x = dy
        camera_y = dx
        v[0, 0, ACTION_IDX["cameraX"]] = float(max(-1.0, min(1.0, camera_x)))
        v[0, 0, ACTION_IDX["cameraY"]] = float(max(-1.0, min(1.0, camera_y)))

        return v


svc = OpenOasisRuntimeService()
app = FastAPI(title="WMFactory Open-Oasis Service")


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
