from __future__ import annotations

import atexit
import base64
import io
import os
import pty
import select
import hashlib
import socket
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import imageio.v2 as imageio
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "models" / "Matrix-Game-3.0"

PROMPT_TOKEN = "Please input the mouse action"


class LoadRequest(BaseModel):
    model_id: Optional[str] = "matrixgame3"


class StartRequest(BaseModel):
    init_image_base64: Optional[str] = None


class StepRequest(BaseModel):
    session_id: str
    action: Dict[str, Any]


class ResetRequest(BaseModel):
    session_id: str
    init_image_base64: Optional[str] = None


class ProgressRequest(BaseModel):
    request_id: Optional[str] = None


@dataclass
class SessionState:
    session_id: str
    session_dir: Path
    seed_path: Path
    current_frame_path: Path
    current_frame_b64: str
    started_at: float
    step_count: int = 0
    current_video_path: Optional[Path] = None
    final_video_path: Optional[Path] = None
    last_action: Optional[Dict[str, Any]] = None
    last_motion: Optional[Dict[str, Any]] = None
    proc: Optional[subprocess.Popen[bytes]] = None
    master_fd: Optional[int] = None
    reader_thread: Optional[threading.Thread] = None
    output_buffer: str = ""
    prompt_count: int = 0
    saved_final: bool = False
    seed_hash: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)
    output_event: threading.Event = field(default_factory=threading.Event)


@dataclass
class Runtime:
    loaded: bool = False
    session: Optional[SessionState] = None
    preloaded_session: Optional[SessionState] = None


class MatrixGame3RuntimeService:
    def __init__(self) -> None:
        os.environ.setdefault("HF_ENDPOINT", os.getenv("WM_HF_ENDPOINT", "https://hf-mirror.com"))
        self.runtime = Runtime()
        self._lock = threading.Lock()
        self.session_root = Path(
            os.getenv("WM_MATRIXGAME3_SESSION_DIR", str(ROOT / "outputs" / "matrixgame3" / "sessions"))
        ).resolve()
        self.session_root.mkdir(parents=True, exist_ok=True)
        self.preload_image_path = Path(
            os.getenv("WM_MATRIXGAME3_PRELOAD_IMAGE", str(ROOT / "demoImage" / "real.png"))
        ).resolve()

        self.python_bin = os.getenv(
            "WM_MATRIXGAME3_PYTHON",
            str(ROOT / "venvs" / "Matrix-Game-3" / "bin" / "python"),
        )
        self.ckpt_dir = Path(
            os.getenv(
                "WM_MATRIXGAME3_CKPT_DIR",
                str(MODEL_ROOT / "Matrix-Game-3.0"),
            )
        ).resolve()
        self.output_dir = Path(
            os.getenv("WM_MATRIXGAME3_OUTPUT_DIR", str(MODEL_ROOT / "output" / "frontend"))
        ).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cuda_visible_devices = os.getenv("WM_MATRIXGAME3_CUDA_VISIBLE_DEVICES", "0,1")
        self.num_gpus = int(os.getenv("WM_MATRIXGAME3_NUM_GPUS", "2"))
        self.prompt = os.getenv(
            "WM_MATRIXGAME3_PROMPT",
            "Continue this first-person world naturally from the input image with coherent egomotion, stable geometry, realistic lighting, high detail, and smooth camera motion.",
        )
        self.size = os.getenv("WM_MATRIXGAME3_SIZE", "704*1280")
        self.num_iterations = int(os.getenv("WM_MATRIXGAME3_NUM_ITERATIONS", "100"))
        self.num_inference_steps = int(os.getenv("WM_MATRIXGAME3_NUM_INFERENCE_STEPS", "50"))
        self.seed = int(os.getenv("WM_MATRIXGAME3_SEED", "42"))
        self.start_timeout = float(os.getenv("WM_MATRIXGAME3_START_TIMEOUT", "1800"))
        self.step_timeout = float(os.getenv("WM_MATRIXGAME3_STEP_TIMEOUT", "900"))
        self.camera_deadzone = float(os.getenv("WM_MATRIXGAME3_CAMERA_DEADZONE", "0.08"))
        self.camera_scale = float(os.getenv("WM_MATRIXGAME3_CAMERA_SCALE", "0.1"))
        self.use_base_model = os.getenv("WM_MATRIXGAME3_USE_BASE_MODEL", "1") == "1"
        self._progress_lock = threading.Lock()
        self._progress: Dict[str, Any] = {
            "request_id": None,
            "phase": "idle",
            "message": "idle",
            "active": False,
            "session_id": None,
            "updated_at": time.time(),
        }
        self._progress_logs: Dict[str, list[str]] = {}

        atexit.register(self._cleanup)

    def _log(self, message: str) -> None:
        text = f"[service][matrixgame3] {message}"
        self._append_log(text)
        print(text, flush=True)

    def _append_log(self, message: str) -> None:
        with self._progress_lock:
            request_id = self._progress.get("request_id")
            if request_id is not None:
                logs = list(self._progress_logs.get(request_id, []))
                logs.append(message)
                self._progress_logs[request_id] = logs[-200:]
            self._progress["updated_at"] = time.time()

    def _set_progress(
        self,
        *,
        phase: str,
        message: str,
        active: bool,
        session_id: Optional[str],
        request_id: Optional[str] = None,
    ) -> None:
        with self._progress_lock:
            if request_id is not None:
                self._progress["request_id"] = request_id
                self._progress_logs[request_id] = []
            self._progress["phase"] = phase
            self._progress["message"] = message
            self._progress["active"] = active
            self._progress["session_id"] = session_id
            self._progress["updated_at"] = time.time()

    def progress(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        with self._progress_lock:
            p = dict(self._progress)
            active_request_id = p.get("request_id")
            if request_id is not None:
                p["logs"] = list(self._progress_logs.get(request_id, []))
            elif active_request_id is not None:
                p["logs"] = list(self._progress_logs.get(active_request_id, []))
            else:
                p["logs"] = []
        if request_id and p.get("request_id") and request_id != p.get("request_id") and request_id not in self._progress_logs:
            return {
                "request_id": request_id,
                "phase": "unknown",
                "message": "request_id not found",
                "active": False,
                "session_id": None,
                "logs": [],
                "updated_at": time.time(),
            }
        return p

    def health(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "model_id": "matrixgame3",
            "ready": self.runtime.loaded,
            "session_id": None if self.runtime.session is None else self.runtime.session.session_id,
        }

    def load(self) -> Dict[str, Any]:
        with self._lock:
            if self.runtime.loaded and self.runtime.preloaded_session is not None:
                return {
                    "model_id": "matrixgame3",
                    "status": "already_loaded",
                    "device": f"cuda:{self.cuda_visible_devices}",
                    "num_gpus": self.num_gpus,
                    "native_steps": self.num_inference_steps,
                    "max_iterations": self.num_iterations,
                    "request_id": self._progress.get("request_id"),
                }
            request_id = str(uuid.uuid4())
            self._set_progress(
                request_id=request_id,
                phase="validating",
                message="validating Matrix-Game-3.0 environment",
                active=True,
                session_id=None,
            )
            if not Path(self.python_bin).exists():
                raise RuntimeError(f"Matrix-Game-3.0 python not found: {self.python_bin}")
            if not self.ckpt_dir.exists():
                raise RuntimeError(f"Matrix-Game-3.0 checkpoint dir not found: {self.ckpt_dir}")
            if not self.preload_image_path.exists():
                raise RuntimeError(f"Matrix-Game-3.0 preload image not found: {self.preload_image_path}")
            self.runtime.loaded = True
            self._terminate_session(self.runtime.preloaded_session)
            preload_bytes = self.preload_image_path.read_bytes()
            preload_session = self._create_session(
                session_id=f"preload-{uuid.uuid4()}",
                init_image_bytes=preload_bytes,
            )
            self.runtime.preloaded_session = preload_session
            self._set_progress(
                phase="loading",
                message="spawning Matrix-Game-3.0 preload worker",
                active=True,
                session_id=preload_session.session_id,
            )
            self._spawn_interactive_process(preload_session)
            self._set_progress(
                phase="warming",
                message="waiting for Matrix-Game-3.0 interactive prompt",
                active=True,
                session_id=preload_session.session_id,
            )
            self._wait_for_prompt(preload_session, expected_count=1, timeout=self.start_timeout)
            self._set_progress(
                phase="ready",
                message="Matrix-Game-3.0 preload worker is ready",
                active=False,
                session_id=preload_session.session_id,
            )
            self._log("load done")
            return {
                "model_id": "matrixgame3",
                "status": "loaded",
                "device": f"cuda:{self.cuda_visible_devices}",
                "num_gpus": self.num_gpus,
                "native_steps": self.num_inference_steps,
                "max_iterations": self.num_iterations,
                "request_id": request_id,
            }

    def start_session(self, init_image_base64: Optional[str]) -> Dict[str, Any]:
        init_image_bytes = self._decode_image(init_image_base64)
        if init_image_bytes is None:
            raise RuntimeError("init_image_base64 is required for Matrix-Game-3.0 start")
        if not self.runtime.loaded:
            self.load()

        with self._lock:
            request_id = str(uuid.uuid4())
            seed_hash = self._hash_bytes(init_image_bytes)
            self._set_progress(
                request_id=request_id,
                phase="starting",
                message="starting Matrix-Game-3.0 session",
                active=True,
                session_id=None,
            )
            preload_session = self.runtime.preloaded_session
            if preload_session is not None and preload_session.seed_hash == seed_hash:
                self._terminate_session(self.runtime.session)
                self.runtime.session = preload_session
                self.runtime.preloaded_session = None
                self._set_progress(
                    phase="ready",
                    message="reused preloaded Matrix-Game-3.0 worker",
                    active=False,
                    session_id=preload_session.session_id,
                )
                self._log(f"start_session reused preload session_id={preload_session.session_id}")
                return {"session_id": preload_session.session_id, "frame_base64": preload_session.current_frame_b64}

            self._terminate_session(self.runtime.session)
            session_id = str(uuid.uuid4())
            session = self._create_session(session_id=session_id, init_image_bytes=init_image_bytes)
            self.runtime.session = session
            self._terminate_session(self.runtime.preloaded_session)
            self.runtime.preloaded_session = None
            self._set_progress(
                phase="loading",
                message="spawning Matrix-Game-3.0 session worker",
                active=True,
                session_id=session_id,
            )
            self._spawn_interactive_process(session)
            self._set_progress(
                phase="warming",
                message="waiting for Matrix-Game-3.0 session prompt",
                active=True,
                session_id=session_id,
            )
            self._wait_for_prompt(session, expected_count=1, timeout=self.start_timeout)
            self._set_progress(
                phase="ready",
                message="Matrix-Game-3.0 session ready",
                active=False,
                session_id=session_id,
            )
            self._log(f"start_session ready session_id={session_id}")
            return {"session_id": session_id, "frame_base64": session.current_frame_b64, "request_id": request_id}

    def reset_session(self, session_id: str, init_image_base64: Optional[str]) -> Dict[str, Any]:
        session = self._require_session(session_id)
        init_image_bytes = self._decode_image(init_image_base64)
        if init_image_bytes is None:
            init_image_bytes = session.current_frame_path.read_bytes()
        self._terminate_session(session)
        return self.start_session(self._encode_image(init_image_bytes))

    def step(self, session_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        session = self._require_session(session_id)
        if session.step_count >= self.num_iterations:
            return {
                "session_id": session.session_id,
                "frame_base64": session.current_frame_b64,
                "reward": 0.0,
                "ended": True,
                "truncated": False,
                "extra": {"reason": "max_iterations_reached", "step_count": session.step_count},
            }

        with self._lock:
            proc = self._require_proc(session)
            if proc.poll() is not None:
                raise RuntimeError(f"Matrix-Game-3.0 process exited early with code {proc.returncode}")

            t0 = time.perf_counter()
            keyboard_token, movement_key = self._keyboard_token(action)
            mouse_token, camera_key, motion = self._mouse_token(action)
            chunk_index = session.step_count
            expected_prompt_count = min(chunk_index + 2, self.num_iterations)
            chunk_video_path = self.output_dir / f"{session.session_id}_current_iteration_{chunk_index}.mp4"
            self._set_progress(
                phase="stepping",
                message=f"running Matrix-Game-3.0 step {chunk_index + 1}",
                active=True,
                session_id=session.session_id,
            )

            self._write_stdin(session, f"{mouse_token}\n{keyboard_token}\n")
            self._wait_for_file(chunk_video_path, timeout=self.step_timeout)
            ended = (chunk_index + 1) >= self.num_iterations
            if not ended:
                self._wait_for_prompt(session, expected_count=expected_prompt_count, timeout=self.step_timeout)
            else:
                self._wait_for_final_save(session, timeout=self.step_timeout)

            frame_b64 = self._extract_last_frame_base64(session, chunk_video_path)
            session.current_frame_b64 = frame_b64
            session.current_video_path = chunk_video_path
            session.step_count += 1
            session.last_action = dict(action)
            session.last_motion = dict(motion)

            latency_ms = int((time.perf_counter() - t0) * 1000)
            self._log(
                f"step done session_id={session.session_id} step={session.step_count} "
                f"move={movement_key} camera={camera_key} latency_ms={latency_ms}"
            )
            self._set_progress(
                phase="ready",
                message=f"Matrix-Game-3.0 step {session.step_count} completed",
                active=False,
                session_id=session.session_id,
            )
            return {
                "session_id": session.session_id,
                "frame_base64": frame_b64,
                "reward": 0.0,
                "ended": ended,
                "truncated": False,
                "extra": {
                    "latency_ms": latency_ms,
                    "step_count": session.step_count,
                    "movement_key": movement_key,
                    "camera_key": camera_key,
                    "video_path": str(chunk_video_path),
                    "final_video_path": str(session.final_video_path) if ended and session.final_video_path else None,
                    "native_mouse_input": mouse_token,
                    "native_keyboard_input": keyboard_token,
                    "motion": motion,
                },
            }

    def _spawn_interactive_process(self, session: SessionState) -> None:
        master_fd, slave_fd = pty.openpty()
        master_port = self._pick_free_port()
        cmd = [
            self.python_bin,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={self.num_gpus}",
            "--master_addr",
            "127.0.0.1",
            "--master_port",
            str(master_port),
            "generate.py",
            "--size",
            self.size,
            "--dit_fsdp",
            "--t5_fsdp",
            "--ulysses_size",
            str(self.num_gpus),
            "--ckpt_dir",
            str(self.ckpt_dir),
            "--fa_version",
            "2",
            "--use_int8",
            "--num_iterations",
            str(self.num_iterations),
            "--num_inference_steps",
            str(self.num_inference_steps),
            "--image",
            str(session.seed_path),
            "--prompt",
            self.prompt,
            "--save_name",
            session.session_id,
            "--seed",
            str(self.seed),
            "--lightvae_pruning_rate",
            "0.75",
            "--vae_type",
            "mg_lightvae_v2",
            "--output_dir",
            str(self.output_dir),
            "--interactive",
        ]
        if self.use_base_model:
            cmd.append("--use_base_model")

        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        env["WAN_VAE_SEGMENT_SIZE"] = "1"
        env["WM_MATRIXGAME3_DRAW_CONTROLS"] = os.getenv("WM_MATRIXGAME3_DRAW_CONTROLS", "0")
        env["OMP_NUM_THREADS"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        env["http_proxy"] = ""
        env["https_proxy"] = ""
        env["HTTP_PROXY"] = ""
        env["HTTPS_PROXY"] = ""
        env.setdefault("HF_ENDPOINT", os.getenv("WM_HF_ENDPOINT", "https://hf-mirror.com"))
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = str(master_port)

        proc = subprocess.Popen(
            cmd,
            cwd=str(MODEL_ROOT),
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            env=env,
            start_new_session=True,
            close_fds=True,
        )
        os.close(slave_fd)

        session.proc = proc
        session.master_fd = master_fd
        session.reader_thread = threading.Thread(target=self._reader_loop, args=(session,), daemon=True)
        session.reader_thread.start()
        self._log(f"spawned interactive worker pid={proc.pid} session_id={session.session_id}")

    def _reader_loop(self, session: SessionState) -> None:
        assert session.master_fd is not None
        fd = session.master_fd
        try:
            while True:
                ready, _, _ = select.select([fd], [], [], 0.5)
                if not ready:
                    proc = session.proc
                    if proc is not None and proc.poll() is not None:
                        break
                    continue
                try:
                    chunk = os.read(fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="replace")
                with session.lock:
                    session.output_buffer += text
                    session.prompt_count = session.output_buffer.count(PROMPT_TOKEN)
                    if "Saved concatenated video with" in session.output_buffer:
                        session.saved_final = True
                session.output_event.set()
                for line in text.splitlines():
                    if line.strip():
                        self._log(line.rstrip())
        finally:
            session.output_event.set()

    def _wait_for_prompt(self, session: SessionState, expected_count: int, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            proc = self._require_proc(session)
            if proc.poll() is not None:
                raise RuntimeError(self._format_crash(session, proc.returncode))
            with session.lock:
                if session.prompt_count >= expected_count:
                    return
            session.output_event.wait(timeout=1.0)
            session.output_event.clear()
        raise RuntimeError(f"Timed out waiting for Matrix-Game-3.0 prompt #{expected_count}")

    def _wait_for_file(self, path: Path, timeout: float) -> None:
        deadline = time.time() + timeout
        last_size = -1
        stable_rounds = 0
        while time.time() < deadline:
            if path.exists():
                size = path.stat().st_size
                if size > 0 and size == last_size:
                    stable_rounds += 1
                    if stable_rounds >= 2:
                        return
                else:
                    stable_rounds = 0
                    last_size = size
            time.sleep(1.0)
        raise RuntimeError(f"Timed out waiting for Matrix-Game-3.0 chunk output: {path}")

    def _wait_for_final_save(self, session: SessionState, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            proc = self._require_proc(session)
            with session.lock:
                if session.saved_final:
                    return
            if session.final_video_path is not None and session.final_video_path.exists() and session.final_video_path.stat().st_size > 0:
                return
            if proc.poll() is not None:
                if session.final_video_path is not None and session.final_video_path.exists():
                    return
                raise RuntimeError(self._format_crash(session, proc.returncode))
            session.output_event.wait(timeout=1.0)
            session.output_event.clear()
        raise RuntimeError("Timed out waiting for Matrix-Game-3.0 final video save")

    def _write_stdin(self, session: SessionState, text: str) -> None:
        if session.master_fd is None:
            raise RuntimeError("Matrix-Game-3.0 session stdin is not available")
        os.write(session.master_fd, text.encode("utf-8"))

    def _keyboard_token(self, action: Dict[str, Any]) -> tuple[str, str]:
        forward = bool(action.get("w")) and not bool(action.get("s"))
        backward = bool(action.get("s")) and not bool(action.get("w"))
        left = bool(action.get("a")) and not bool(action.get("d"))
        right = bool(action.get("d")) and not bool(action.get("a"))

        token = ""
        if forward:
            token += "w"
        elif backward:
            token += "s"
        if left:
            token += "a"
        elif right:
            token += "d"
        if not token:
            token = "q"

        movement_key = token.upper() if token != "q" else "Q"
        return token, movement_key

    def _mouse_token(self, action: Dict[str, Any]) -> tuple[str, str, Dict[str, float]]:
        camera_dx = float(action.get("camera_dx", 0.0) or 0.0)
        camera_dy = float(action.get("camera_dy", 0.0) or 0.0)

        if abs(camera_dx) <= self.camera_deadzone:
            camera_dx = 0.0
        if abs(camera_dy) <= self.camera_deadzone:
            camera_dy = 0.0

        pitch = max(-1.0, min(1.0, camera_dy)) * self.camera_scale
        yaw = max(-1.0, min(1.0, camera_dx)) * self.camera_scale
        if abs(pitch) < 1e-6 and abs(yaw) < 1e-6:
            return "u", "U", {"pitch": 0.0, "yaw": 0.0}

        token = f"{pitch:.4f},{yaw:.4f}"
        camera_key = self._camera_label(pitch, yaw)
        return token, camera_key, {"pitch": pitch, "yaw": yaw}

    def _camera_label(self, pitch: float, yaw: float) -> str:
        vertical = ""
        horizontal = ""
        if pitch > 1e-6:
            vertical = "I"
        elif pitch < -1e-6:
            vertical = "K"
        if yaw > 1e-6:
            horizontal = "L"
        elif yaw < -1e-6:
            horizontal = "J"
        return (vertical + horizontal) or "U"

    def _extract_last_frame_base64(self, session: SessionState, video_path: Path) -> str:
        reader = imageio.get_reader(str(video_path))
        try:
            last_frame = None
            for frame in reader:
                last_frame = frame
            if last_frame is None:
                raise RuntimeError(f"No frames found in video: {video_path}")
        finally:
            reader.close()

        image = Image.fromarray(last_frame).convert("RGB")
        image.save(session.current_frame_path, format="PNG")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _format_crash(self, session: SessionState, returncode: Optional[int]) -> str:
        with session.lock:
            tail = session.output_buffer[-4000:]
        return f"Matrix-Game-3.0 process exited with code {returncode}. Recent output:\n{tail}"

    def _terminate_session(self, session: Optional[SessionState]) -> None:
        if session is None:
            return
        proc = session.proc
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
        if session.master_fd is not None:
            try:
                os.close(session.master_fd)
            except OSError:
                pass
            session.master_fd = None
        session.proc = None

    def _cleanup(self) -> None:
        self._terminate_session(self.runtime.session)
        self._terminate_session(self.runtime.preloaded_session)

    def _decode_image(self, payload: Optional[str]) -> Optional[bytes]:
        if not payload:
            return None
        if "," in payload:
            payload = payload.split(",", 1)[1]
        return base64.b64decode(payload)

    def _encode_image(self, payload: bytes) -> str:
        return f"data:image/png;base64,{base64.b64encode(payload).decode('utf-8')}"

    def _write_png(self, init_image_bytes: bytes, seed_path: Path, current_frame_path: Path) -> str:
        image = Image.open(io.BytesIO(init_image_bytes)).convert("RGB")
        image.save(seed_path, format="PNG")
        image.save(current_frame_path, format="PNG")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _create_session(self, session_id: str, init_image_bytes: bytes) -> SessionState:
        session_dir = self.session_root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        seed_path = session_dir / "seed.png"
        current_frame_path = session_dir / "current.png"
        frame_b64 = self._write_png(init_image_bytes, seed_path, current_frame_path)
        return SessionState(
            session_id=session_id,
            session_dir=session_dir,
            seed_path=seed_path,
            current_frame_path=current_frame_path,
            current_frame_b64=frame_b64,
            started_at=time.time(),
            final_video_path=self.output_dir / f"{session_id}.mp4",
            seed_hash=self._hash_bytes(init_image_bytes),
        )

    def _hash_bytes(self, payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    def _pick_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _require_session(self, session_id: str) -> SessionState:
        if self.runtime.session is None:
            raise RuntimeError("Session is not started. Call /sessions/start first.")
        if self.runtime.session.session_id != session_id:
            raise RuntimeError("Unknown or expired session_id")
        return self.runtime.session

    def _require_proc(self, session: SessionState) -> subprocess.Popen[bytes]:
        if session.proc is None:
            raise RuntimeError("Matrix-Game-3.0 subprocess is not running")
        return session.proc


service = MatrixGame3RuntimeService()
app = FastAPI(title="Matrix-Game-3.0 Runtime Service")


@app.post("/health")
def health() -> Dict[str, Any]:
    return service.health()


@app.post("/load")
def load(req: LoadRequest) -> Dict[str, Any]:
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


@app.post("/sessions/progress")
def progress(req: ProgressRequest) -> Dict[str, Any]:
    try:
        return service.progress(req.request_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
