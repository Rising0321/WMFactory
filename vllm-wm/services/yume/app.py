from __future__ import annotations

import base64
import importlib
import io
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
YUME_ROOT = Path(os.getenv("WM_YUME_ROOT", str(PROJECT_ROOT / "vendors" / "yume"))).resolve()
YUME_CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "yume"

if str(YUME_ROOT) not in sys.path:
    sys.path.insert(0, str(YUME_ROOT))


class LoadRequest(BaseModel):
    model_id: Optional[str] = "yume"


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
    seed_path: Path
    seed_frame_b64: str
    started_at: float
    step_count: int = 0
    last_action: Optional[Dict[str, Any]] = None


@dataclass
class Runtime:
    session: Optional[SessionState] = None
    loaded: bool = False
    device: str = "cuda:0"


class YumeRuntimeService:
    def __init__(self) -> None:
        os.environ.setdefault("HF_ENDPOINT", os.getenv("WM_HF_ENDPOINT", "https://hf-mirror.com"))
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = ""
        os.environ["HTTP_PROXY"] = ""
        os.environ["HTTPS_PROXY"] = ""

        self.runtime = Runtime()
        self._lock = threading.Lock()
        self.yume = importlib.import_module("webapp_single_gpu")
        self.yume.CKPT_DIR = str(
            Path(os.getenv("WM_YUME_CKPT_DIR", str(YUME_CHECKPOINT_ROOT / "Yume-5B-720P"))).resolve()
        )
        self.yume.INTERNVL_PATH = str(
            Path(os.getenv("WM_YUME_INTERNVL_PATH", str(YUME_CHECKPOINT_ROOT / "InternVL3-2B-Instruct"))).resolve()
        )
        self.yume.OUTPUT_DIR = str(
            Path(os.getenv("WM_YUME_OUTPUT_DIR", str(PROJECT_ROOT / "outputs" / "yume"))).resolve()
        )
        os.makedirs(self.yume.OUTPUT_DIR, exist_ok=True)

        self.session_root = Path(
            os.getenv("WM_YUME_SESSION_DIR", str(PROJECT_ROOT / "outputs" / "yume" / "sessions"))
        ).resolve()
        self.session_root.mkdir(parents=True, exist_ok=True)

        self.prompt = os.getenv(
            "WM_YUME_PROMPT",
            "Continue this realistic first-person world naturally with coherent motion and scene consistency.",
        )
        self.resolution = os.getenv("WM_YUME_RESOLUTION", "704x1280")
        self.fps = int(os.getenv("WM_YUME_FPS", "16"))
        self.sample_steps = int(os.getenv("WM_YUME_SAMPLE_STEPS", "4"))
        self.sample_num = int(os.getenv("WM_YUME_SAMPLE_NUM", "1"))
        self.frame_zero = int(os.getenv("WM_YUME_FRAME_ZERO", "32"))
        self.shift = float(os.getenv("WM_YUME_SHIFT", "5.0"))
        self.seed = int(os.getenv("WM_YUME_SEED", "43"))
        self.memory_optimization = os.getenv("WM_YUME_MEMORY_OPTIMIZATION", "1") == "1"
        self.vae_memory_optimization = os.getenv("WM_YUME_VAE_MEMORY_OPTIMIZATION", "1") == "1"
        self.refine_from_image = os.getenv("WM_YUME_REFINE_FROM_IMAGE", "0") == "1"
        self.camera_deadzone = float(os.getenv("WM_YUME_CAMERA_DEADZONE", "0.18"))
        # Unified frontend already inverts the generic camera stick axes for
        # non-MineWorld models. YUME needs the opposite interpretation, so
        # compensate here by default.
        self.invert_camera_x = os.getenv("WM_YUME_INVERT_CAMERA_X", "1") == "1"
        self.invert_camera_y = os.getenv("WM_YUME_INVERT_CAMERA_Y", "1") == "1"

    def _log(self, message: str) -> None:
        print(f"[service][yume] {message}", flush=True)

    def health(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "model_id": "yume",
            "ready": self.runtime.loaded,
            "session_id": None if self.runtime.session is None else self.runtime.session.session_id,
        }

    def load(self) -> Dict[str, Any]:
        with self._lock:
            self._log("load requested")
            if self.runtime.loaded:
                return {
                    "model_id": "yume",
                    "status": "already_loaded",
                    "device": self.runtime.device,
                    "resolution": self.resolution,
                    "sample_steps": self.sample_steps,
                }

            self.yume.load_wan()
            device_id = getattr(self.yume, "DEVICE_ID", 0)
            self.runtime.loaded = True
            self.runtime.device = f"cuda:{device_id}"
            self._log(f"load done device={self.runtime.device}")
            return {
                "model_id": "yume",
                "status": "loaded",
                "device": self.runtime.device,
                "resolution": self.resolution,
                "sample_steps": self.sample_steps,
            }

    def start_session(self, init_image_base64: Optional[str]) -> Dict[str, Any]:
        init_image_bytes = self._decode_image(init_image_base64)
        if init_image_bytes is None:
            raise RuntimeError("init_image_base64 is required for YUME start")

        if not self.runtime.loaded:
            self.load()

        with self._lock:
            self._reset_yume_context()
            session_id = str(uuid.uuid4())
            seed_path, seed_frame_b64 = self._write_seed_image(session_id, init_image_bytes)
            self.runtime.session = SessionState(
                session_id=session_id,
                seed_path=seed_path,
                seed_frame_b64=seed_frame_b64,
                started_at=time.time(),
            )
            self._log(f"start_session done session_id={session_id}")
            return {"session_id": session_id, "frame_base64": seed_frame_b64}

    def reset_session(self, session_id: str, init_image_base64: Optional[str]) -> Dict[str, Any]:
        session = self._require_session(session_id)
        init_image_bytes = self._decode_image(init_image_base64)

        with self._lock:
            self._reset_yume_context()
            if init_image_bytes is not None:
                seed_path, seed_frame_b64 = self._write_seed_image(session.session_id, init_image_bytes)
                session.seed_path = seed_path
                session.seed_frame_b64 = seed_frame_b64
            session.step_count = 0
            session.last_action = None
            session.started_at = time.time()
            self._log(f"reset_session done session_id={session.session_id}")
            return {"session_id": session.session_id, "frame_base64": session.seed_frame_b64}

    def step(self, session_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        session = self._require_session(session_id)
        if not self.runtime.loaded:
            self.load()

        with self._lock:
            t0 = time.perf_counter()
            move_key = self._movement_from_action(action)
            camera_key = self._camera_from_action(action)
            continue_from_last = session.step_count > 0

            gen_args = self.yume.LongGenArgs(
                prompt=self.prompt,
                jpg_path=None if continue_from_last else str(session.seed_path),
                output_dir=self.yume.OUTPUT_DIR,
                fps=self.fps,
                sample_steps=self.sample_steps,
                sample_num=self.sample_num,
                frame_zero=self.frame_zero,
                shift=self.shift,
                seed=self.seed,
                continue_from_last=continue_from_last,
                refine_from_image=self.refine_from_image,
                caption_path=None,
                mode="I2V",
                resolution=self.resolution,
                memory_optimization=self.memory_optimization,
                vae_memory_optimization=self.vae_memory_optimization,
                camera_movement1=move_key,
                camera_movement2=camera_key,
            )
            video_path, final_prompt = self.yume.long_generate(gen_args)
            frame_b64 = self._latest_frame_base64()

            session.step_count += 1
            session.last_action = dict(action)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            self._log(
                f"step done session_id={session.session_id} step={session.step_count} "
                f"move={move_key} camera={camera_key} latency_ms={latency_ms}"
            )
            return {
                "session_id": session.session_id,
                "frame_base64": frame_b64,
                "reward": 0.0,
                "ended": False,
                "truncated": False,
                "extra": {
                    "latency_ms": latency_ms,
                    "step_count": session.step_count,
                    "video_path": str(video_path),
                    "prompt": final_prompt,
                    "movement_key": move_key,
                    "camera_key": camera_key,
                },
            }

    def _require_session(self, session_id: str) -> SessionState:
        if self.runtime.session is None:
            raise RuntimeError("Session is not started. Call /sessions/start first.")
        if self.runtime.session.session_id != session_id:
            raise RuntimeError("Unknown or expired session_id")
        return self.runtime.session

    def _reset_yume_context(self) -> None:
        self.yume.LAST["last_model_input_latent"] = None
        self.yume.LAST["last_model_input_de"] = None
        self.yume.LAST["frame_total"] = 0
        self.yume.LAST["last_video_path"] = None
        self.yume.LAST["last_prompt"] = ""

    def _decode_image(self, payload: Optional[str]) -> Optional[bytes]:
        if not payload:
            return None
        if "," in payload:
            payload = payload.split(",", 1)[1]
        return base64.b64decode(payload)

    def _write_seed_image(self, session_id: str, init_image_bytes: bytes) -> tuple[Path, str]:
        session_dir = self.session_root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        seed_path = session_dir / "seed.png"

        img = Image.open(io.BytesIO(init_image_bytes)).convert("RGB")
        img.save(seed_path, format="PNG")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        seed_frame_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return seed_path, seed_frame_b64

    def _latest_frame_base64(self) -> str:
        last_video = self.yume.LAST.get("last_model_input_de")
        if last_video is None:
            raise RuntimeError("YUME did not produce a frame")
        frame = last_video[:, -1, :, :].detach().float().clamp(-1, 1)
        frame = frame.add(1.0).div(2.0).mul(255.0).byte().permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _movement_from_action(self, action: Dict[str, Any]) -> str:
        forward = bool(action.get("w")) and not bool(action.get("s"))
        backward = bool(action.get("s")) and not bool(action.get("w"))
        left = bool(action.get("a")) and not bool(action.get("d"))
        right = bool(action.get("d")) and not bool(action.get("a"))

        if forward and left:
            return "W+A"
        if forward and right:
            return "W+D"
        if backward and left:
            return "S+A"
        if backward and right:
            return "S+D"
        if forward:
            return "W"
        if backward:
            return "S"
        if left:
            return "A"
        if right:
            return "D"
        return "None"

    def _camera_from_action(self, action: Dict[str, Any]) -> str:
        dx = float(action.get("camera_dx", 0.0) or 0.0)
        dy = float(action.get("camera_dy", 0.0) or 0.0)

        if self.invert_camera_x:
            dx = -dx
        if self.invert_camera_y:
            dy = -dy

        horiz = ""
        vert = ""

        if dx >= self.camera_deadzone:
            horiz = "→"
        elif dx <= -self.camera_deadzone:
            horiz = "←"

        if dy >= self.camera_deadzone:
            vert = "↓"
        elif dy <= -self.camera_deadzone:
            vert = "↑"

        key = f"{vert}{horiz}"
        return key or "·"


svc = YumeRuntimeService()
app = FastAPI(title="WMFactory YUME Service")


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
