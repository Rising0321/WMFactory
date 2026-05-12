from __future__ import annotations

import base64
import gc
import io
import os
import socket
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from scipy.spatial.transform import Rotation as R


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVICE_ROOT = PROJECT_ROOT / "services" / "worldplay"
WORLDPLAY_ROOT = Path(os.getenv("WM_WORLDPLAY_ROOT", str(PROJECT_ROOT / "vendors" / "worldplay"))).resolve()
WORLDPLAY_CHECKPOINT_ROOT = Path(
    os.getenv("WM_WORLDPLAY_CHECKPOINT_ROOT", str(PROJECT_ROOT / "checkpoints" / "worldplay"))
).resolve()
WAN_ROOT = WORLDPLAY_ROOT / "wan"

if str(WORLDPLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(WORLDPLAY_ROOT))
if str(WAN_ROOT) not in sys.path:
    sys.path.insert(0, str(WAN_ROOT))

from hyvideo.generate_custom_trajectory import generate_camera_trajectory_local
from wan.inference.helper import CHUNK_SIZE
from wan.models.utils import select_mem_frames_wan


ACTION_MAPPING = {
    (0, 0, 0, 0): 0,
    (1, 0, 0, 0): 1,
    (0, 1, 0, 0): 2,
    (0, 0, 1, 0): 3,
    (0, 0, 0, 1): 4,
    (1, 0, 1, 0): 5,
    (1, 0, 0, 1): 6,
    (0, 1, 1, 0): 7,
    (0, 1, 0, 1): 8,
}
INTRINSIC = [
    [969.6969696969696, 0.0, 960.0],
    [0.0, 969.6969696969696, 540.0],
    [0.0, 0.0, 1.0],
]


class LoadRequest(BaseModel):
    model_id: Optional[str] = "worldplay"


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
    seed_path: Path
    current_frame_path: Path
    current_frame_b64: str
    pose_history: List[np.ndarray] = field(default_factory=list)
    chunk_index: int = 0
    started_at: float = 0.0
    last_action: Optional[Dict[str, Any]] = None
    last_motion: Optional[Dict[str, float]] = None
    latent_history_cpu: Optional[torch.Tensor] = None
    bootstrap_ctx: Optional[Dict[str, Any]] = None


@dataclass
class Runtime:
    runner: Optional[Any] = None
    session: Optional[SessionState] = None
    loaded: bool = False
    device: str = "cuda:0"
    aux_device: str = "cuda:1"


class WorldPlayRuntimeService:
    def __init__(self) -> None:
        os.environ.setdefault("HF_ENDPOINT", os.getenv("WM_HF_ENDPOINT", "https://hf-mirror.com"))
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = ""
        os.environ["HTTP_PROXY"] = ""
        os.environ["HTTPS_PROXY"] = ""
        os.environ.setdefault("WAN_MODEL_CPU_OFFLOAD", "0")
        os.environ.setdefault("WAN_VAE_USE_TILING", os.getenv("WM_WORLDPLAY_VAE_USE_TILING", "1"))
        os.environ.setdefault(
            "WAN_VAE_TILE_SAMPLE_MIN_HEIGHT",
            os.getenv("WM_WORLDPLAY_VAE_TILE_SAMPLE_MIN_HEIGHT", "128"),
        )
        os.environ.setdefault(
            "WAN_VAE_TILE_SAMPLE_MIN_WIDTH",
            os.getenv("WM_WORLDPLAY_VAE_TILE_SAMPLE_MIN_WIDTH", "128"),
        )
        os.environ.setdefault(
            "WAN_VAE_TILE_SAMPLE_STRIDE_HEIGHT",
            os.getenv("WM_WORLDPLAY_VAE_TILE_SAMPLE_STRIDE_HEIGHT", "96"),
        )
        os.environ.setdefault(
            "WAN_VAE_TILE_SAMPLE_STRIDE_WIDTH",
            os.getenv("WM_WORLDPLAY_VAE_TILE_SAMPLE_STRIDE_WIDTH", "96"),
        )
        os.environ.setdefault(
            "WAN_OFFLOAD_INACTIVE_FOR_DECODE",
            os.getenv("WM_WORLDPLAY_OFFLOAD_INACTIVE_FOR_DECODE", "1"),
        )
        default_aux_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        os.environ.setdefault("WAN_AUX_DEVICE", os.getenv("WM_WORLDPLAY_AUX_DEVICE", default_aux_device))
        default_vae_device = default_aux_device
        default_decode_device = "cuda:0"
        os.environ.setdefault("WAN_VAE_DEVICE", os.getenv("WM_WORLDPLAY_VAE_DEVICE", default_vae_device))
        os.environ.setdefault(
            "WAN_DECODE_VAE_DEVICE",
            os.getenv("WM_WORLDPLAY_DECODE_VAE_DEVICE", default_decode_device),
        )
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(self._pick_free_port()))

        self.runtime = Runtime()
        self._lock = threading.Lock()
        self.session_root = Path(
            os.getenv("WM_WORLDPLAY_SESSION_DIR", str(PROJECT_ROOT / "outputs" / "worldplay" / "sessions"))
        ).resolve()
        self.session_root.mkdir(parents=True, exist_ok=True)
        self.service_log_path = Path(
            os.getenv("WM_WORLDPLAY_SERVICE_LOG", str(SERVICE_ROOT / "worldplay_service.log"))
        ).resolve()
        self.worldplay_snapshot = Path(
            os.getenv(
                "WM_WORLDPLAY_WAN_SNAPSHOT",
                str(WORLDPLAY_CHECKPOINT_ROOT / "HY-WorldPlay"),
            )
        ).resolve()
        self.base_model_dir = Path(
            os.getenv(
                "WM_WORLDPLAY_WAN_BASE_SNAPSHOT",
                str(WORLDPLAY_CHECKPOINT_ROOT / "Wan2.2-TI2V-5B-Diffusers"),
            )
        ).resolve()

        self.ar_model_path = Path(
            os.getenv("WM_WORLDPLAY_AR_MODEL_PATH", str(self.worldplay_snapshot / "wan_transformer"))
        ).resolve()
        self.ckpt_path = Path(
            os.getenv("WM_WORLDPLAY_CKPT_PATH", str(self.worldplay_snapshot / "wan_distilled_model" / "model.pt"))
        ).resolve()
        self.model_id = str(Path(os.getenv("WM_WORLDPLAY_MODEL_ID", str(self.base_model_dir))).resolve())

        self.prompt = os.getenv(
            "WM_WORLDPLAY_PROMPT",
            "Continue this first-person world naturally with coherent motion, stable geometry, and realistic scene persistence.",
        )
        self.negative_prompt = os.getenv(
            "WM_WORLDPLAY_NEGATIVE_PROMPT",
            (
                "色调艳丽,过曝,静态,细节模糊不清,字幕,风格,作品,画作,画面,静止,整体发灰,"
                "最差质量,低质量,JPEG压缩残留,丑陋的,残缺的,多余的手指,画得不好的手部,"
                "画得不好的脸部,畸形的,毁容的,形态畸形的肢体,手指融合,静止不动的画面,"
                "杂乱的背景,三条腿,背景人很多,倒着走"
            ),
        )
        self.height = int(os.getenv("WM_WORLDPLAY_HEIGHT", "704"))
        self.width = int(os.getenv("WM_WORLDPLAY_WIDTH", "1280"))
        self.num_inference_steps = int(os.getenv("WM_WORLDPLAY_NUM_INFERENCE_STEPS", "50"))
        self.context_window_length = int(os.getenv("WM_WORLDPLAY_CONTEXT_WINDOW_LENGTH", "16"))
        self.use_memory = os.getenv("WM_WORLDPLAY_USE_MEMORY", "1") == "1"
        self.seed = int(os.getenv("WM_WORLDPLAY_SEED", "0"))
        self.max_chunks = int(os.getenv("WM_WORLDPLAY_MAX_CHUNKS", "8"))
        self.num_frames = 1 + self.max_chunks * CHUNK_SIZE * 4
        self.move_step = float(os.getenv("WM_WORLDPLAY_MOVE_STEP", "0.08"))
        self.yaw_step = np.deg2rad(float(os.getenv("WM_WORLDPLAY_YAW_DEG", "3.0")))
        self.pitch_step = np.deg2rad(float(os.getenv("WM_WORLDPLAY_PITCH_DEG", "3.0")))
        self.camera_deadzone = float(os.getenv("WM_WORLDPLAY_CAMERA_DEADZONE", "0.08"))
        self.invert_pitch = os.getenv("WM_WORLDPLAY_INVERT_PITCH", "1") == "1"

    def _log(self, message: str) -> None:
        text = f"[service][worldplay] {message}"
        try:
            print(text, flush=True)
        except BrokenPipeError:
            pass
        except OSError:
            pass

    def _pick_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def health(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "model_id": "worldplay",
            "ready": self.runtime.loaded,
            "session_id": None if self.runtime.session is None else self.runtime.session.session_id,
        }

    def load(self) -> Dict[str, Any]:
        with self._lock:
            self._log("load requested")
            if self.runtime.loaded and self.runtime.runner is not None:
                return {
                    "model_id": "worldplay",
                    "status": "already_loaded",
                    "device": self.runtime.device,
                    "aux_device": self.runtime.aux_device,
                    "native_steps": self.num_inference_steps,
                    "chunk_size": CHUNK_SIZE,
                }

            from wan.generate import WanRunner

            runner = WanRunner(
                model_id=self.model_id,
                ckpt_path=str(self.ckpt_path),
                ar_model_path=str(self.ar_model_path),
            )
            self.runtime.runner = runner
            self.runtime.loaded = True
            self.runtime.device = str(runner.device)
            self.runtime.aux_device = str(runner.aux_device)
            self._log(f"load done device={self.runtime.device} aux_device={self.runtime.aux_device}")
            return {
                "model_id": "worldplay",
                "status": "loaded",
                "device": self.runtime.device,
                "aux_device": self.runtime.aux_device,
                "native_steps": self.num_inference_steps,
                "chunk_size": CHUNK_SIZE,
            }

    def start_session(self, init_image_base64: Optional[str]) -> Dict[str, Any]:
        init_image_bytes = self._decode_image(init_image_base64)
        if init_image_bytes is None:
            raise RuntimeError("init_image_base64 is required for WorldPlay start")
        if not self.runtime.loaded:
            self.load()

        with self._lock:
            self._reset_runner_context()
            session_id = str(uuid.uuid4())
            session_dir = self.session_root / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            seed_path = session_dir / "seed.png"
            current_frame_path = session_dir / "current.png"
            frame_b64 = self._write_png(init_image_bytes, seed_path, current_frame_path)
            self.runtime.session = SessionState(
                session_id=session_id,
                session_dir=session_dir,
                seed_path=seed_path,
                current_frame_path=current_frame_path,
                current_frame_b64=frame_b64,
                pose_history=[np.eye(4, dtype=np.float64)],
                chunk_index=0,
                started_at=time.time(),
            )
            self._log(f"start_session done session_id={session_id}")
            return {"session_id": session_id, "frame_base64": frame_b64}

    def reset_session(self, session_id: str, init_image_base64: Optional[str]) -> Dict[str, Any]:
        session = self._require_session(session_id)
        init_image_bytes = self._decode_image(init_image_base64)
        if init_image_bytes is None:
            raise RuntimeError("init_image_base64 is required for WorldPlay reset")

        with self._lock:
            self._reset_runner_context()
            frame_b64 = self._write_png(init_image_bytes, session.seed_path, session.current_frame_path)
            session.current_frame_b64 = frame_b64
            session.pose_history = [np.eye(4, dtype=np.float64)]
            session.chunk_index = 0
            session.started_at = time.time()
            session.last_action = None
            session.last_motion = None
            session.latent_history_cpu = None
            session.bootstrap_ctx = None
            self._log(f"reset_session done session_id={session_id}")
            return {"session_id": session.session_id, "frame_base64": frame_b64}

    def step(self, session_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        session = self._require_session(session_id)
        if not self.runtime.loaded or self.runtime.runner is None:
            self.load()

        with self._lock:
            if session.chunk_index >= self.max_chunks:
                return {
                    "session_id": session.session_id,
                    "frame_base64": session.current_frame_b64,
                    "reward": 0.0,
                    "ended": False,
                    "truncated": True,
                    "extra": {
                        "detail": f"Reached max_chunks={self.max_chunks}",
                        "chunk_index": session.chunk_index,
                    },
                }

            t0 = time.perf_counter()
            motion = self._motion_from_action(action)
            self._append_chunk_poses(session, motion)
            curr_viewmats, curr_Ks, curr_action = self._current_chunk_inputs(session)
            try:
                frame_b64 = self._run_chunk(session, curr_viewmats, curr_Ks, curr_action)
            except Exception:
                self._cleanup_step_state(full=False)
                raise

            session.current_frame_b64 = frame_b64
            session.chunk_index += 1
            session.last_action = dict(action)
            session.last_motion = dict(motion)

            latency_ms = int((time.perf_counter() - t0) * 1000)
            self._log(
                f"step done session_id={session.session_id} chunk={session.chunk_index} "
                f"motion={motion} latency_ms={latency_ms}"
            )
            return {
                "session_id": session.session_id,
                "frame_base64": frame_b64,
                "reward": 0.0,
                "ended": False,
                "truncated": False,
                "extra": {
                    "latency_ms": latency_ms,
                    "chunk_index": session.chunk_index,
                    "motion": motion,
                    "remaining_chunks": self.max_chunks - session.chunk_index,
                },
            }

    def _require_session(self, session_id: str) -> SessionState:
        if self.runtime.session is None:
            raise RuntimeError("Session is not started. Call /sessions/start first.")
        if self.runtime.session.session_id != session_id:
            raise RuntimeError("Unknown or expired session_id")
        return self.runtime.session

    def _reset_runner_context(self) -> None:
        if self.runtime.runner is None:
            return
        self._restore_runner_devices()
        self.runtime.runner.pipe.ctx = None
        self.runtime.runner.pipe._decode_state = {
            "current_latent_idx": 0,
            "total_latents": None,
            "latents_mean": None,
            "latents_std": None,
        }
        self._cleanup_step_state(full=True)

    def _cleanup_step_state(self, full: bool) -> None:
        if self.runtime.runner is None:
            return
        pipe = self.runtime.runner.pipe
        decode_state = getattr(pipe, "_decode_state", None)
        if isinstance(decode_state, dict):
            decode_state["current_latent_idx"] = 0
            decode_state["total_latents"] = None
            decode_state["all_latents"] = None
            decode_state["chunk_i"] = None
            if full:
                decode_state["latents_mean"] = None
                decode_state["latents_std"] = None

        gc.collect()
        try:
            import torch._dynamo as dynamo

            dynamo.reset()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _restore_runner_devices(self) -> None:
        if self.runtime.runner is None:
            return
        runner = self.runtime.runner
        pipe = runner.pipe
        transform_dtype = getattr(pipe.transformer, "dtype", torch.bfloat16)
        if hasattr(pipe, "transformer") and pipe.transformer is not None:
            pipe.transformer.to(device=runner.device, dtype=transform_dtype)
        if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
            pipe.text_encoder.to(runner.aux_device)
        if hasattr(pipe, "vae") and pipe.vae is not None:
            pipe.vae.to(runner.vae_device)
        if hasattr(runner, "vae") and runner.vae is not None:
            runner.vae.to(runner.vae_device)

    def _offload_inactive_modules_for_decode(self) -> None:
        if self.runtime.runner is None:
            return
        if os.getenv("WAN_OFFLOAD_INACTIVE_FOR_DECODE", "1") != "1":
            return
        pipe = self.runtime.runner.pipe
        if hasattr(pipe, "transformer") and pipe.transformer is not None:
            pipe.transformer.to("cpu")
        if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
            pipe.text_encoder.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _extract_bootstrap_ctx(self) -> Dict[str, Any]:
        assert self.runtime.runner is not None
        ctx = self.runtime.runner.pipe.ctx
        if ctx is None:
            raise RuntimeError("WorldPlay pipeline context is unavailable")

        keys = [
            "prompt_embeds",
            "negative_prompt_embeds",
            "guidance_scale",
            "guidance_scale_2",
            "attention_kwargs",
            "transformer_dtype",
            "timesteps",
            "num_inference_steps",
            "chunk_size",
            "first_chunk_size",
            "use_memory",
            "context_window_length",
            "device",
            "batch_size",
            "few_step",
            "sigmas",
            "stabilization_level",
            "sp_world_size",
            "rank_in_sp_group",
            "local_rank_in_sp_group",
        ]
        return {key: ctx[key] for key in keys}

    def _generated_latent_count(self, session: SessionState) -> int:
        return session.chunk_index * CHUNK_SIZE

    def _capture_step_latents(self, session: SessionState, local_history_count: int) -> None:
        assert self.runtime.runner is not None
        ctx = self.runtime.runner.pipe.ctx
        if ctx is None:
            raise RuntimeError("WorldPlay pipeline context is unavailable after generation")

        if session.bootstrap_ctx is None:
            session.bootstrap_ctx = self._extract_bootstrap_ctx()

        start = local_history_count
        end = start + CHUNK_SIZE
        new_latents = ctx["latents"][:, :, start:end].detach().to("cpu")

        if session.latent_history_cpu is None:
            session.latent_history_cpu = new_latents
        else:
            session.latent_history_cpu = torch.cat([session.latent_history_cpu, new_latents], dim=2)

    def _build_windowed_ctx(
        self,
        session: SessionState,
        all_viewmats: torch.Tensor,
        all_Ks: torch.Tensor,
        all_action: torch.Tensor,
    ) -> tuple[int, List[int]]:
        if session.bootstrap_ctx is None or session.latent_history_cpu is None:
            raise RuntimeError("WorldPlay rolling context is not initialized")
        assert self.runtime.runner is not None

        bootstrap = session.bootstrap_ctx
        pipe = self.runtime.runner.pipe
        history_count = session.latent_history_cpu.shape[2]
        current_frame_idx = history_count

        if bootstrap["use_memory"]:
            if current_frame_idx >= bootstrap["context_window_length"]:
                selected_abs = select_mem_frames_wan(
                    all_viewmats.cpu().numpy(),
                    current_frame_idx,
                    memory_frames=16,
                    temporal_context_size=12,
                    pred_latent_size=CHUNK_SIZE,
                    device=self.runtime.runner.device,
                    points_local=pipe.points_local,
                )
            else:
                selected_abs = list(range(0, current_frame_idx))
        else:
            start = max(0, current_frame_idx - bootstrap["context_window_length"])
            selected_abs = list(range(start, current_frame_idx))

        retained_abs = sorted(selected_abs)
        history_window = len(retained_abs)
        if history_window == 0 or history_window % CHUNK_SIZE != 0:
            raise RuntimeError(f"Invalid WorldPlay history window size: {history_window}")

        local_chunk_index = history_window // CHUNK_SIZE
        latent_dtype = session.latent_history_cpu.dtype
        latent_device = bootstrap["device"]
        latent_history = session.latent_history_cpu[:, :, retained_abs].to(device=latent_device, dtype=latent_dtype)

        bsz, channels, _, latent_h, latent_w = latent_history.shape
        generator = torch.Generator(device=str(latent_device))
        generator.manual_seed(self.seed + session.chunk_index)
        latents = torch.randn(
            (bsz, channels, history_window + CHUNK_SIZE, latent_h, latent_w),
            device=latent_device,
            dtype=latent_dtype,
            generator=generator,
        )
        latents[:, :, :history_window] = latent_history

        pipe.init_kv_cache()
        kv_cache = pipe._kv_cache
        kv_cache_neg = pipe._kv_cache_neg

        viewmats = torch.zeros(
            1, history_window + CHUNK_SIZE, 4, 4, device=latent_device, dtype=latents.dtype
        )
        Ks = torch.zeros(
            1, history_window + CHUNK_SIZE, 3, 3, device=latent_device, dtype=latents.dtype
        )
        action = torch.zeros(
            1, history_window + CHUNK_SIZE, device=latent_device, dtype=torch.long
        )
        viewmats[:, :history_window] = all_viewmats[retained_abs].to(device=latent_device, dtype=latents.dtype)
        Ks[:, :history_window] = all_Ks[retained_abs].to(device=latent_device, dtype=latents.dtype)
        action[:, :history_window] = all_action[retained_abs].to(device=latent_device)

        pipe.ctx = {
            "prompt_embeds": bootstrap["prompt_embeds"],
            "negative_prompt_embeds": bootstrap["negative_prompt_embeds"],
            "guidance_scale": bootstrap["guidance_scale"],
            "guidance_scale_2": bootstrap["guidance_scale_2"],
            "attention_kwargs": bootstrap["attention_kwargs"],
            "transformer_dtype": bootstrap["transformer_dtype"],
            "timesteps": bootstrap["timesteps"],
            "num_inference_steps": bootstrap["num_inference_steps"],
            "latents": latents,
            "first_image_condition": None,
            "chunk_size": CHUNK_SIZE,
            "first_chunk_size": CHUNK_SIZE,
            "n_latent": history_window + CHUNK_SIZE,
            "viewmats": viewmats,
            "Ks": Ks,
            "action": action,
            "use_memory": bootstrap["use_memory"],
            "context_window_length": bootstrap["context_window_length"],
            "kv_cache": kv_cache,
            "kv_cache_neg": kv_cache_neg,
            "use_kv_cache": True,
            "sp_world_size": bootstrap["sp_world_size"],
            "rank_in_sp_group": bootstrap["rank_in_sp_group"],
            "local_rank_in_sp_group": bootstrap["local_rank_in_sp_group"],
            "device": bootstrap["device"],
            "batch_size": bootstrap["batch_size"],
            "few_step": bootstrap["few_step"],
            "sigmas": bootstrap["sigmas"],
            "stabilization_level": bootstrap["stabilization_level"],
            "selected_frame_indices_override": list(range(history_window)),
        }
        return local_chunk_index, retained_abs

    def _decode_image(self, payload: Optional[str]) -> Optional[bytes]:
        if not payload:
            return None
        if "," in payload:
            payload = payload.split(",", 1)[1]
        return base64.b64decode(payload)

    def _tail_logs(self, lines: int = 40) -> str:
        try:
            if not self.service_log_path.exists():
                return "service log unavailable"
            content = self.service_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(content[-lines:]) if content else "service log empty"
        except Exception as exc:
            return f"failed to read service log: {exc}"

    def format_exception(self, exc: Exception) -> str:
        tb = traceback.format_exc()
        log_tail = self._tail_logs()
        return f"{exc}\nTraceback:\n{tb}\nRecent worldplay log:\n{log_tail}"

    def _write_png(self, init_image_bytes: bytes, seed_path: Path, current_frame_path: Path) -> str:
        img = Image.open(io.BytesIO(init_image_bytes)).convert("RGB")
        img.save(seed_path, format="PNG")
        img.save(current_frame_path, format="PNG")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _motion_from_action(self, action: Dict[str, Any]) -> Dict[str, float]:
        dx = float(action.get("camera_dx", 0.0) or 0.0)
        dy = float(action.get("camera_dy", 0.0) or 0.0)
        if abs(dx) <= self.camera_deadzone:
            dx = 0.0
        if abs(dy) <= self.camera_deadzone:
            dy = 0.0

        speed_scale = 2.0 if bool(action.get("shift", False)) else 1.0
        forward = ((1.0 if bool(action.get("w", False)) else 0.0) - (1.0 if bool(action.get("s", False)) else 0.0)) * self.move_step * speed_scale
        right = ((1.0 if bool(action.get("d", False)) else 0.0) - (1.0 if bool(action.get("a", False)) else 0.0)) * self.move_step * speed_scale
        yaw = dx * self.yaw_step
        pitch = (-dy if self.invert_pitch else dy) * self.pitch_step

        motion: Dict[str, float] = {}
        if abs(yaw) > 1e-8:
            motion["yaw"] = float(yaw)
        if abs(pitch) > 1e-8:
            motion["pitch"] = float(pitch)
        if abs(forward) > 1e-8:
            motion["forward"] = float(forward)
        if abs(right) > 1e-8:
            motion["right"] = float(right)
        return motion

    def _append_chunk_poses(self, session: SessionState, motion: Dict[str, float]) -> None:
        base_pose = session.pose_history[-1]
        relative_poses = generate_camera_trajectory_local([motion.copy() for _ in range(CHUNK_SIZE)])
        for rel_pose in relative_poses[1:]:
            session.pose_history.append(base_pose @ rel_pose)

    def _current_chunk_inputs(self, session: SessionState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pose_json = {str(i): {"extrinsic": pose.tolist(), "K": INTRINSIC} for i, pose in enumerate(session.pose_history)}
        all_viewmats, all_Ks, all_action = self._pose_to_input(pose_json, len(session.pose_history))
        start_idx = session.chunk_index * CHUNK_SIZE
        end_idx = start_idx + CHUNK_SIZE
        return all_viewmats[start_idx:end_idx], all_Ks[start_idx:end_idx], all_action[start_idx:end_idx]

    def _pose_to_input(self, pose_json: Dict[str, Dict[str, Any]], latent_num: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pose_keys = list(pose_json.keys())
        if len(pose_keys) != latent_num:
            raise RuntimeError(f"pose history mismatch: expected {latent_num}, got {len(pose_keys)}")

        intrinsic_list: List[np.ndarray] = []
        w2c_list: List[np.ndarray] = []
        for i in range(latent_num):
            c2w = np.array(pose_json[str(i)]["extrinsic"], dtype=np.float64)
            w2c = np.linalg.inv(c2w)
            w2c_list.append(w2c)
            intrinsic = np.array(pose_json[str(i)]["K"], dtype=np.float64)
            intrinsic[0, 0] /= intrinsic[0, 2] * 2
            intrinsic[1, 1] /= intrinsic[1, 2] * 2
            intrinsic[0, 2] = 0.5
            intrinsic[1, 2] = 0.5
            intrinsic_list.append(intrinsic)

        w2c_array = np.array(w2c_list, dtype=np.float64)
        c2ws = np.linalg.inv(w2c_array)
        c_inv = np.linalg.inv(c2ws[:-1])
        relative_c2w = np.zeros_like(c2ws)
        relative_c2w[0, ...] = c2ws[0, ...]
        relative_c2w[1:, ...] = c_inv @ c2ws[1:, ...]

        trans_one_hot = np.zeros((relative_c2w.shape[0], 4), dtype=np.int32)
        rotate_one_hot = np.zeros((relative_c2w.shape[0], 4), dtype=np.int32)
        move_norm_valid = 1e-4
        for i in range(1, relative_c2w.shape[0]):
            move_dirs = relative_c2w[i, :3, 3]
            move_norm = np.linalg.norm(move_dirs)
            if move_norm > move_norm_valid:
                move_norm_dirs = move_dirs / move_norm
                angles_rad = np.arccos(move_norm_dirs.clip(-1.0, 1.0))
                trans_angles_deg = angles_rad * (180.0 / np.pi)
            else:
                trans_angles_deg = np.zeros(3, dtype=np.float64)

            rot_angles_deg = R.from_matrix(relative_c2w[i, :3, :3]).as_euler("xyz", degrees=True)
            if move_norm > move_norm_valid:
                if trans_angles_deg[2] < 60:
                    trans_one_hot[i, 0] = 1
                elif trans_angles_deg[2] > 120:
                    trans_one_hot[i, 1] = 1
                if trans_angles_deg[0] < 60:
                    trans_one_hot[i, 2] = 1
                elif trans_angles_deg[0] > 120:
                    trans_one_hot[i, 3] = 1

            if rot_angles_deg[1] > 5e-2:
                rotate_one_hot[i, 0] = 1
            elif rot_angles_deg[1] < -5e-2:
                rotate_one_hot[i, 1] = 1
            if rot_angles_deg[0] > 5e-2:
                rotate_one_hot[i, 2] = 1
            elif rot_angles_deg[0] < -5e-2:
                rotate_one_hot[i, 3] = 1

        trans_one_label = np.array([ACTION_MAPPING[tuple(row.tolist())] for row in trans_one_hot], dtype=np.int64)
        rotate_one_label = np.array([ACTION_MAPPING[tuple(row.tolist())] for row in rotate_one_hot], dtype=np.int64)
        action_one_label = trans_one_label * 9 + rotate_one_label
        return (
            torch.as_tensor(w2c_array, dtype=torch.float32),
            torch.as_tensor(np.array(intrinsic_list), dtype=torch.float32),
            torch.as_tensor(action_one_label, dtype=torch.long),
        )

    def _run_chunk(
        self,
        session: SessionState,
        curr_viewmats: torch.Tensor,
        curr_Ks: torch.Tensor,
        curr_action: torch.Tensor,
    ) -> str:
        assert self.runtime.runner is not None
        runner = self.runtime.runner
        self._restore_runner_devices()

        torch.manual_seed(self.seed)
        local_history_count = 0
        local_chunk_index = session.chunk_index

        if session.chunk_index == 0 or session.latent_history_cpu is None:
            runner.pipe(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                height=self.height,
                width=self.width,
                num_frames=self.num_frames,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=1.0,
                few_step=True,
                first_chunk_size=CHUNK_SIZE,
                return_dict=False,
                image_path=str(session.seed_path),
                use_memory=self.use_memory,
                context_window_length=self.context_window_length,
                chunk_i=session.chunk_index,
                viewmats=curr_viewmats.unsqueeze(0),
                Ks=curr_Ks.unsqueeze(0),
                action=curr_action.unsqueeze(0),
                output_type="latent",
            )
        else:
            pose_json = {str(i): {"extrinsic": pose.tolist(), "K": INTRINSIC} for i, pose in enumerate(session.pose_history)}
            all_viewmats, all_Ks, all_action = self._pose_to_input(pose_json, len(session.pose_history))
            local_chunk_index, retained_abs = self._build_windowed_ctx(session, all_viewmats, all_Ks, all_action)
            local_history_count = len(retained_abs)
            self._log(
                f"rolling window rebuild session_id={session.session_id} "
                f"abs_history={self._generated_latent_count(session)} kept={local_history_count}"
            )
            runner.pipe(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                height=self.height,
                width=self.width,
                num_frames=1 + (local_history_count + CHUNK_SIZE - 1) * 4,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=1.0,
                few_step=True,
                first_chunk_size=CHUNK_SIZE,
                return_dict=False,
                image_path=None,
                use_memory=self.use_memory,
                context_window_length=self.context_window_length,
                chunk_i=local_chunk_index,
                viewmats=curr_viewmats.unsqueeze(0),
                Ks=curr_Ks.unsqueeze(0),
                action=curr_action.unsqueeze(0),
                output_type="latent",
            )

        self._offload_inactive_modules_for_decode()
        final_video = None
        for _ in range(CHUNK_SIZE):
            final_video = runner.pipe.decode_next_latent(output_type="np")
        if final_video is None:
            raise RuntimeError("WorldPlay did not produce a frame")

        self._capture_step_latents(session, local_history_count)
        runner.pipe.ctx = None

        last_frame = self._extract_last_frame(final_video)
        img = Image.fromarray(last_frame)
        img.save(session.current_frame_path, format="PNG")
        with session.current_frame_path.open("rb") as fp:
            frame_b64 = base64.b64encode(fp.read()).decode("utf-8")

        self._cleanup_step_state(full=False)
        return frame_b64

    def _extract_last_frame(self, video: Any) -> np.ndarray:
        arr = np.asarray(video)
        if arr.ndim == 5:
            if arr.shape[-1] in (1, 3, 4):
                frame = arr[0, -1]
            elif arr.shape[2] in (1, 3, 4):
                frame = np.moveaxis(arr[0, -1], 0, -1)
            elif arr.shape[1] in (1, 3, 4):
                frame = np.moveaxis(arr[0, :, -1], 0, -1)
            else:
                raise RuntimeError(f"Unsupported WorldPlay video shape: {arr.shape}")
        elif arr.ndim == 4:
            if arr.shape[-1] in (1, 3, 4):
                frame = arr[-1]
            elif arr.shape[0] in (1, 3, 4):
                frame = np.moveaxis(arr[:, -1], 0, -1)
            else:
                raise RuntimeError(f"Unsupported WorldPlay frame stack shape: {arr.shape}")
        else:
            raise RuntimeError(f"Unsupported WorldPlay output rank: {arr.ndim}")

        if np.issubdtype(frame.dtype, np.floating):
            frame = np.clip(frame, 0.0, 1.0)
            frame = (frame * 255.0).round().astype(np.uint8)
        elif frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(frame)


svc = WorldPlayRuntimeService()
app = FastAPI(title="WMFactory WorldPlay Service")


@app.post("/health")
def health() -> Dict[str, Any]:
    return svc.health()


@app.post("/load")
def load(_: LoadRequest) -> Dict[str, Any]:
    try:
        return svc.load()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=svc.format_exception(exc)) from exc


@app.post("/sessions/start")
def start(req: StartRequest) -> Dict[str, Any]:
    try:
        return svc.start_session(req.init_image_base64)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=svc.format_exception(exc)) from exc


@app.post("/sessions/reset")
def reset(req: ResetRequest) -> Dict[str, Any]:
    try:
        return svc.reset_session(req.session_id, req.init_image_base64)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=svc.format_exception(exc)) from exc


@app.post("/sessions/step")
def step(req: StepRequest) -> Dict[str, Any]:
    try:
        return svc.step(req.session_id, req.action)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=svc.format_exception(exc)) from exc
