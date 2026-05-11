from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import imageio.v2 as imageio
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVICE_ROOT = PROJECT_ROOT / "services" / "lingbotworldfast"
LINGBOTFAST_CHECKPOINT_ROOT = Path(
    os.getenv(
        "WM_LINGBOTWORLDFAST_CHECKPOINT_ROOT",
        str(PROJECT_ROOT / "checkpoints" / "lingbotworld" / "lingbot-world-base-cam"),
    )
).resolve()
LINGBOTFAST_ROOT = Path(
    os.getenv("WM_LINGBOTWORLDFAST_ROOT", str(PROJECT_ROOT / "vendors" / "lingbotworldfast"))
).resolve()
RUNNER_PATH = SERVICE_ROOT / "run_fast_infer.py"
PERSISTENT_WORKER_PATH = SERVICE_ROOT / "persistent_worker.py"

if str(LINGBOTFAST_ROOT) not in sys.path:
    sys.path.insert(0, str(LINGBOTFAST_ROOT))

from wan.utils.wasd_ijkl_to_c2ws import generate_and_save_trajectory, pad_frame_num_to_4n_plus_1  # noqa: E402


class LoadRequest(BaseModel):
    model_id: Optional[str] = "lingbot-world-fast"


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
    session_dir: Path
    current_frame_path: Path
    current_frame_b64: str
    current_pose: np.ndarray
    current_video_path: Optional[Path]
    started_at: float
    step_count: int = 0
    last_action: Optional[Dict[str, Any]] = None
    last_keys: Optional[List[str]] = None


@dataclass
class Runtime:
    loaded: bool = False
    session: Optional[SessionState] = None
    num_procs: int = 1
    visible_devices: str = ""
    mode: str = "persistent-torchrun"
    worker_proc: Optional[subprocess.Popen[str]] = None
    worker_log_fp: Optional[Any] = None


class LingBotWorldFastRuntimeService:
    def __init__(self) -> None:
        os.environ.setdefault("HF_ENDPOINT", os.getenv("WM_HF_ENDPOINT", "https://hf-mirror.com"))
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = ""
        os.environ["HTTP_PROXY"] = ""
        os.environ["HTTPS_PROXY"] = ""

        if not torch.cuda.is_available():
            raise RuntimeError("LingBot-World-Fast requires CUDA")

        self.runtime = Runtime()
        self._lock = threading.Lock()
        self.session_root = Path(
            os.getenv(
                "WM_LINGBOTWORLDFAST_SESSION_DIR",
                str(PROJECT_ROOT / "outputs" / "lingbotworldfast" / "sessions"),
            )
        ).resolve()
        self.session_root.mkdir(parents=True, exist_ok=True)
        self.worker_root = Path(
            os.getenv(
                "WM_LINGBOTWORLDFAST_WORKER_DIR",
                str(PROJECT_ROOT / "outputs" / "lingbotworldfast" / "worker"),
            )
        ).resolve()
        self.worker_root.mkdir(parents=True, exist_ok=True)
        self.worker_request_dir = self.worker_root / "requests"
        self.worker_request_dir.mkdir(parents=True, exist_ok=True)
        self.worker_log_path = self.worker_root / "persistent_worker.log"
        self.worker_ready_path = self.worker_request_dir / "ready.json"

        self.service_python = Path(
            os.getenv("WM_LINGBOTWORLDFAST_SERVICE_PYTHON", sys.executable)
        ).resolve()
        self.prompt = os.getenv(
            "WM_LINGBOTWORLDFAST_PROMPT",
            (
                "Continue this first-person world naturally with coherent egomotion, "
                "stable geometry, realistic scene persistence, and smooth camera motion."
            ),
        )
        self.frame_num = pad_frame_num_to_4n_plus_1(int(os.getenv("WM_LINGBOTWORLDFAST_FRAME_NUM", "9")))
        self.size = os.getenv("WM_LINGBOTWORLDFAST_SIZE", "480*832")
        self.shift = float(os.getenv("WM_LINGBOTWORLDFAST_SHIFT", "10.0"))
        self.seed = int(os.getenv("WM_LINGBOTWORLDFAST_SEED", "42"))
        self.offload_model = self._parse_bool(os.getenv("WM_LINGBOTWORLDFAST_OFFLOAD_MODEL"), default=False)
        self.t5_cpu = self._parse_bool(os.getenv("WM_LINGBOTWORLDFAST_T5_CPU"), default=True)
        self.convert_model_dtype = self._parse_bool(
            os.getenv("WM_LINGBOTWORLDFAST_CONVERT_MODEL_DTYPE"),
            default=False,
        )
        self.max_attention_size = self._parse_optional_int(os.getenv("WM_LINGBOTWORLDFAST_MAX_ATTENTION_SIZE"))
        self.subprocess_timeout = int(os.getenv("WM_LINGBOTWORLDFAST_SUBPROCESS_TIMEOUT", "7200"))
        self.worker_start_timeout = int(os.getenv("WM_LINGBOTWORLDFAST_WORKER_START_TIMEOUT", "1800"))

        self.camera_deadzone = float(os.getenv("WM_LINGBOTWORLDFAST_CAMERA_DEADZONE", "0.08"))
        self.invert_yaw = self._parse_bool(os.getenv("WM_LINGBOTWORLDFAST_INVERT_YAW"), default=False)
        self.invert_pitch = self._parse_bool(os.getenv("WM_LINGBOTWORLDFAST_INVERT_PITCH"), default=False)
        self.repeat_keys = max(1, int(os.getenv("WM_LINGBOTWORLDFAST_REPEAT_KEYS", str(self.frame_num - 1))))

        intrinsics_raw = os.getenv("WM_LINGBOTWORLDFAST_BASE_INTRINSICS", "415.5298,415.6922,415.77786,239.77779")
        intrinsics_values = [float(part.strip()) for part in intrinsics_raw.split(",") if part.strip()]
        if len(intrinsics_values) != 4:
            raise ValueError("WM_LINGBOTWORLDFAST_BASE_INTRINSICS must contain 4 comma-separated floats")
        self.base_intrinsics = np.asarray(intrinsics_values, dtype=np.float32)

    def _parse_bool(self, raw: Optional[str], *, default: bool) -> bool:
        if raw is None or raw == "":
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    def _parse_optional_int(self, raw: Optional[str]) -> Optional[int]:
        if raw is None or raw == "":
            return None
        return int(raw)

    def _log(self, message: str) -> None:
        print(f"[service][lingbot-world-fast] {message}", flush=True)

    def _format_exception(self, exc: Exception) -> str:
        return f"{exc}\nTraceback:\n{traceback.format_exc()}"

    def _visible_cuda_count(self) -> int:
        raw = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
        if not raw:
            return torch.cuda.device_count()
        return len([part for part in raw.split(",") if part.strip()])

    def _visible_cuda_devices(self) -> str:
        raw = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
        if raw:
            return raw
        count = torch.cuda.device_count()
        return ",".join(str(index) for index in range(count))

    def _resolve_num_procs(self) -> int:
        visible_count = self._visible_cuda_count()
        default = 4 if visible_count >= 4 else (2 if visible_count >= 2 else 1)
        num_procs = int(os.getenv("WM_LINGBOTWORLDFAST_NUM_PROCS", str(default)))
        if num_procs < 1:
            raise RuntimeError("WM_LINGBOTWORLDFAST_NUM_PROCS must be >= 1")
        if num_procs > visible_count:
            raise RuntimeError(
                f"WM_LINGBOTWORLDFAST_NUM_PROCS={num_procs} exceeds visible CUDA devices ({visible_count})"
            )
        return num_procs

    def _required_paths(self) -> list[Path]:
        return [
            LINGBOTFAST_CHECKPOINT_ROOT / "models_t5_umt5-xxl-enc-bf16.pth",
            LINGBOTFAST_CHECKPOINT_ROOT / "Wan2.1_VAE.pth",
            LINGBOTFAST_CHECKPOINT_ROOT / "google" / "umt5-xxl",
            LINGBOTFAST_CHECKPOINT_ROOT / "lingbot_world_fast" / "config.json",
            LINGBOTFAST_CHECKPOINT_ROOT / "lingbot_world_fast" / "diffusion_pytorch_model.safetensors.index.json",
            PERSISTENT_WORKER_PATH,
        ]

    def health(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "model_id": "lingbot-world-fast",
            "ready": self.runtime.loaded,
            "mode": self.runtime.mode,
            "num_procs": self.runtime.num_procs,
            "visible_devices": self.runtime.visible_devices,
            "session_id": None if self.runtime.session is None else self.runtime.session.session_id,
        }

    def load(self) -> Dict[str, Any]:
        with self._lock:
            self._log("load requested")
            missing = [str(path) for path in self._required_paths() if not path.exists()]
            if missing:
                raise RuntimeError("Missing LingBot-World-Fast assets:\n" + "\n".join(missing))

            self.runtime.num_procs = self._resolve_num_procs()
            self.runtime.visible_devices = self._visible_cuda_devices()
            self._ensure_worker()
            self.runtime.loaded = True
            self._log(
                "load done "
                f"mode={self.runtime.mode} num_procs={self.runtime.num_procs} "
                f"visible_devices={self.runtime.visible_devices or '<all>'}"
            )
            return {
                "model_id": "lingbot-world-fast",
                "status": "loaded",
                "mode": self.runtime.mode,
                "num_procs": self.runtime.num_procs,
                "visible_devices": self.runtime.visible_devices,
                "frame_num": self.frame_num,
                "size": self.size,
            }

    def start_session(self, init_image_base64: Optional[str]) -> Dict[str, Any]:
        init_image_bytes = self._decode_image(init_image_base64)
        if init_image_bytes is None:
            raise RuntimeError("init_image_base64 is required for LingBot-World-Fast start")
        if not self.runtime.loaded:
            self.load()

        with self._lock:
            session_id = str(uuid.uuid4())
            session_dir = self.session_root / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            current_frame_path = session_dir / "current.png"
            _, frame_b64 = self._write_current_image(init_image_bytes, current_frame_path)
            self.runtime.session = SessionState(
                session_id=session_id,
                session_dir=session_dir,
                current_frame_path=current_frame_path,
                current_frame_b64=frame_b64,
                current_pose=np.eye(4, dtype=np.float32),
                current_video_path=None,
                started_at=time.time(),
            )
            self._log(f"start_session done session_id={session_id}")
            return {"session_id": session_id, "frame_base64": frame_b64}

    def reset_session(self, session_id: str, init_image_base64: Optional[str]) -> Dict[str, Any]:
        session = self._require_session(session_id)
        init_image_bytes = self._decode_image(init_image_base64)
        if init_image_bytes is None:
            raise RuntimeError("init_image_base64 is required for LingBot-World-Fast reset")

        with self._lock:
            _, frame_b64 = self._write_current_image(init_image_bytes, session.current_frame_path)
            session.current_frame_b64 = frame_b64
            session.current_pose = np.eye(4, dtype=np.float32)
            session.current_video_path = None
            session.step_count = 0
            session.started_at = time.time()
            session.last_action = None
            session.last_keys = None
            self._log(f"reset_session done session_id={session_id}")
            return {"session_id": session.session_id, "frame_base64": frame_b64}

    def step(self, session_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        session = self._require_session(session_id)
        if not self.runtime.loaded:
            self.load()

        with self._lock:
            keys = self._keys_from_action(action)
            frame_keys = [keys[:] for _ in range(self.repeat_keys)]
            relative_poses = np.asarray(generate_and_save_trajectory(frame_keys), dtype=np.float32)
            poses = np.asarray([session.current_pose @ rel_pose for rel_pose in relative_poses], dtype=np.float32)
            intrinsics = np.repeat(self.base_intrinsics[None, :], poses.shape[0], axis=0).astype(np.float32)

            chunk_dir = session.session_dir / f"chunk_{session.step_count:04d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            np.save(chunk_dir / "poses.npy", poses)
            np.save(chunk_dir / "intrinsics.npy", intrinsics)

            video_path = chunk_dir / "chunk.mp4"
            log_path = chunk_dir / "generate_fast.log"

            t0 = time.perf_counter()
            self._run_generation(
                image_path=session.current_frame_path,
                action_path=chunk_dir,
                video_path=video_path,
                log_path=log_path,
                step_seed=self.seed + session.step_count,
            )
            next_image, frame_b64 = self._extract_last_frame(video_path)
            next_image.save(session.current_frame_path, format="PNG")

            session.current_frame_b64 = frame_b64
            session.current_video_path = video_path
            session.current_pose = poses[-1].astype(np.float32)
            session.step_count += 1
            session.last_action = dict(action)
            session.last_keys = keys[:]

            latency_ms = int((time.perf_counter() - t0) * 1000)
            self._log(
                f"step done session_id={session.session_id} step_count={session.step_count} "
                f"keys={keys} latency_ms={latency_ms}"
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
                    "keys": keys,
                    "video_path": str(video_path),
                    "runner_log": str(log_path),
                    "poses_path": str(chunk_dir / "poses.npy"),
                    "intrinsics_path": str(chunk_dir / "intrinsics.npy"),
                },
            }

    def _run_generation(
        self,
        *,
        image_path: Path,
        action_path: Path,
        video_path: Path,
        log_path: Path,
        step_seed: int,
    ) -> None:
        self._ensure_worker()
        request_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex}"
        request_path = self.worker_request_dir / f"request_{request_id}.json"
        response_path = self.worker_request_dir / f"response_{request_id}.json"
        payload = {
            "type": "generate",
            "request_id": request_id,
            "image_path": str(image_path),
            "action_path": str(action_path),
            "output_path": str(video_path),
            "prompt": self.prompt,
            "frame_num": self.frame_num,
            "size": self.size,
            "shift": self.shift,
            "seed": step_seed,
            "offload_model": self.offload_model,
            "max_attention_size": self.max_attention_size,
        }
        tmp_path = request_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(request_path)

        deadline = time.time() + self.subprocess_timeout
        while time.time() < deadline:
            proc = self.runtime.worker_proc
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(
                    f"LingBot-World-Fast persistent worker exited with code {proc.returncode}\n"
                    f"Worker log tail:\n{self._tail_file(self.worker_log_path)}"
                )
            if response_path.exists():
                response = json.loads(response_path.read_text(encoding="utf-8"))
                response_path.unlink(missing_ok=True)
                if not response.get("ok", False):
                    log_path.write_text(
                        response.get("traceback") or response.get("error", "unknown worker error"),
                        encoding="utf-8",
                    )
                    raise RuntimeError(
                        "LingBot-World-Fast persistent worker request failed\n"
                        f"Worker error: {response.get('error')}\n"
                        f"Worker log tail:\n{self._tail_file(self.worker_log_path)}"
                    )
                break
            time.sleep(0.5)
        else:
            raise TimeoutError(
                f"Timed out waiting for LingBot-World-Fast persistent worker response {request_id}\n"
                f"Worker log tail:\n{self._tail_file(self.worker_log_path)}"
            )

        if not video_path.exists():
            raise RuntimeError(
                f"LingBot-World-Fast persistent worker completed but did not produce video: {video_path}\n"
                f"Worker log tail:\n{self._tail_file(self.worker_log_path)}"
            )

        log_path.write_text(
            f"request_id={request_id}\nworker_log={self.worker_log_path}\nvideo_path={video_path}\n",
            encoding="utf-8",
        )

    def _build_worker_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["http_proxy"] = ""
        env["https_proxy"] = ""
        env["HTTP_PROXY"] = ""
        env["HTTPS_PROXY"] = ""
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("HF_ENDPOINT", os.getenv("WM_HF_ENDPOINT", "https://hf-mirror.com"))
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env["PYTHONPATH"] = f"{LINGBOTFAST_ROOT}{os.pathsep}" + env.get("PYTHONPATH", "")
        return env

    def _ensure_worker(self) -> None:
        proc = self.runtime.worker_proc
        if proc is not None and proc.poll() is None and self.worker_ready_path.exists():
            return

        if proc is not None and proc.poll() is not None:
            self._log(f"persistent worker exited code={proc.returncode}, restarting")

        for path in self.worker_request_dir.glob("request_*.json"):
            path.unlink(missing_ok=True)
        for path in self.worker_request_dir.glob("response_*.json"):
            path.unlink(missing_ok=True)
        self.worker_ready_path.unlink(missing_ok=True)

        cmd = [
            str(self.service_python),
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(self.runtime.num_procs),
            str(PERSISTENT_WORKER_PATH),
            "--ckpt_dir",
            str(LINGBOTFAST_CHECKPOINT_ROOT),
            "--request_dir",
            str(self.worker_request_dir),
        ]
        if self.runtime.worker_log_fp is not None:
            try:
                self.runtime.worker_log_fp.close()
            except Exception:
                pass
        self.worker_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.runtime.worker_log_fp = self.worker_log_path.open("w", encoding="utf-8")
        self._log(f"starting persistent torchrun worker: {' '.join(cmd)}")
        self.runtime.worker_proc = subprocess.Popen(
            cmd,
            cwd=str(SERVICE_ROOT),
            env=self._build_worker_env(),
            stdout=self.runtime.worker_log_fp,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

        deadline = time.time() + self.worker_start_timeout
        while time.time() < deadline:
            proc = self.runtime.worker_proc
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(
                    f"LingBot-World-Fast persistent worker exited during startup code={proc.returncode}\n"
                    f"Worker log tail:\n{self._tail_file(self.worker_log_path)}"
                )
            if self.worker_ready_path.exists():
                self._log("persistent worker ready")
                return
            time.sleep(1.0)

        raise TimeoutError(
            "Timed out waiting for LingBot-World-Fast persistent worker startup\n"
            f"Worker log tail:\n{self._tail_file(self.worker_log_path)}"
        )

    def shutdown(self) -> None:
        proc = self.runtime.worker_proc
        if proc is None:
            return
        if proc.poll() is None:
            request_id = f"shutdown_{uuid.uuid4().hex}"
            request_path = self.worker_request_dir / f"request_{request_id}.json"
            request_path.write_text(
                json.dumps({"type": "shutdown", "request_id": request_id}, ensure_ascii=False),
                encoding="utf-8",
            )
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        if self.runtime.worker_log_fp is not None:
            try:
                self.runtime.worker_log_fp.close()
            except Exception:
                pass
            self.runtime.worker_log_fp = None

    def _tail_file(self, path: Path, lines: int = 120) -> str:
        try:
            content = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as exc:
            return f"failed to read {path}: {exc}"
        if not content:
            return "<empty>"
        return "\n".join(content[-lines:])

    def _require_session(self, session_id: str) -> SessionState:
        if self.runtime.session is None:
            raise RuntimeError("Session is not started. Call /sessions/start first.")
        if self.runtime.session.session_id != session_id:
            raise RuntimeError("Unknown or expired session_id")
        return self.runtime.session

    def _decode_image(self, payload: Optional[str]) -> Optional[bytes]:
        if not payload:
            return None
        if "," in payload:
            payload = payload.split(",", 1)[1]
        return base64.b64decode(payload)

    def _write_current_image(self, init_image_bytes: bytes, current_frame_path: Path) -> tuple[Image.Image, str]:
        image = Image.open(io.BytesIO(init_image_bytes)).convert("RGB")
        image.save(current_frame_path, format="PNG")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        frame_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return image, frame_b64

    def _keys_from_action(self, action: Dict[str, Any]) -> List[str]:
        keys: List[str] = []
        if bool(action.get("w", False)):
            keys.append("w")
        if bool(action.get("a", False)):
            keys.append("a")
        if bool(action.get("s", False)):
            keys.append("s")
        if bool(action.get("d", False)):
            keys.append("d")

        dx = float(action.get("camera_dx", 0.0) or 0.0)
        dy = float(action.get("camera_dy", 0.0) or 0.0)
        if abs(dx) > self.camera_deadzone:
            yaw_positive = dx > 0
            if self.invert_yaw:
                yaw_positive = not yaw_positive
            keys.append("l" if yaw_positive else "j")
        if abs(dy) > self.camera_deadzone:
            pitch_positive = dy > 0
            if self.invert_pitch:
                pitch_positive = not pitch_positive
            keys.append("k" if pitch_positive else "i")
        return keys

    def _extract_last_frame(self, video_path: Path) -> tuple[Image.Image, str]:
        reader = imageio.get_reader(video_path)
        last_frame = None
        try:
            for frame in reader:
                last_frame = frame
        finally:
            reader.close()
        if last_frame is None:
            raise RuntimeError(f"No frames found in {video_path}")
        image = Image.fromarray(last_frame)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return image, base64.b64encode(buf.getvalue()).decode("utf-8")


svc = LingBotWorldFastRuntimeService()
app = FastAPI(title="WMFactory LingBot-World-Fast Service")


@app.on_event("shutdown")
def shutdown() -> None:
    svc.shutdown()


@app.post("/health")
def health() -> Dict[str, Any]:
    return svc.health()


@app.post("/load")
def load(_: LoadRequest) -> Dict[str, Any]:
    try:
        return svc.load()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=svc._format_exception(exc)) from exc


@app.post("/sessions/start")
def start(req: StartRequest) -> Dict[str, Any]:
    try:
        return svc.start_session(req.init_image_base64)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=svc._format_exception(exc)) from exc


@app.post("/sessions/reset")
def reset(req: ResetRequest) -> Dict[str, Any]:
    try:
        return svc.reset_session(req.session_id, req.init_image_base64)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=svc._format_exception(exc)) from exc


@app.post("/sessions/step")
def step(req: StepRequest) -> Dict[str, Any]:
    try:
        return svc.step(req.session_id, req.action)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=svc._format_exception(exc)) from exc
