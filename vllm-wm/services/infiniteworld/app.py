from __future__ import annotations

import base64
import io
import math
import os
import random
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from fastapi import FastAPI, HTTPException
from omegaconf import OmegaConf
from PIL import Image
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INFWORLD_ROOT = Path(os.getenv("WM_INFINITEWORLD_ROOT", str(PROJECT_ROOT / "vendors" / "infiniteworld"))).resolve()

if str(INFWORLD_ROOT) not in sys.path:
    sys.path.insert(0, str(INFWORLD_ROOT))

from infworld.utils.prepare_dataloader import get_obj_from_str  # type: ignore
from infworld.utils.dataset_utils import is_img  # type: ignore
import infworld.context_parallel.context_parallel_util as cp_util  # type: ignore


class LoadRequest(BaseModel):
    model_id: Optional[str] = "infinite-world"


class StartRequest(BaseModel):
    init_image_base64: Optional[str] = None


class StepRequest(BaseModel):
    session_id: str
    action: Dict[str, Any]


class ResetRequest(BaseModel):
    session_id: str
    init_image_base64: Optional[str] = None


MOVE_ACTION_MAP = {
    "no-op": 0,
    "go forward": 1,
    "go back": 2,
    "go left": 3,
    "go right": 4,
    "go forward and go left": 5,
    "go forward and go right": 6,
    "go back and go left": 7,
    "go back and go right": 8,
    "uncertain": 9,
}

VIEW_ACTION_MAP = {
    "no-op": 0,
    "turn up": 1,
    "turn down": 2,
    "turn left": 3,
    "turn right": 4,
    "turn up and turn left": 5,
    "turn up and turn right": 6,
    "turn down and turn left": 7,
    "turn down and turn right": 8,
    "uncertain": 9,
}

NEGATIVE_PROMPT = (
    "many cars, crowds, Vivid hues, overexposed, static, blurry details, subtitles, style, work, artwork, image, still, "
    "overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn "
    "hands, poorly drawn face, deformed, disfigured, deformed limbs, fused fingers, motionless image, cluttered background, "
    "three legs, crowded background, walking backwards."
)


@dataclass
class SessionState:
    session_id: str
    init_image_bytes: bytes
    seed_frame_b64: str
    video_buffer: torch.Tensor
    latent_history: torch.Tensor
    move_indices: list[int]
    view_indices: list[int]
    step_count: int = 0


@dataclass
class Runtime:
    loaded: bool = False
    gen_device: str = "cuda:0"
    decode_device: str = "cuda:1"
    session: Optional[SessionState] = None


class CachedTextEncoder:
    def __init__(self, encoder: Any, target_device: torch.device) -> None:
        self.encoder = encoder
        self.output_dim = encoder.output_dim
        self.model_max_length = encoder.model_max_length
        self.target_device = target_device
        self._cache: Dict[tuple[str, ...], Dict[str, torch.Tensor]] = {}

    def encode(self, texts: list[str]) -> Dict[str, torch.Tensor]:
        key = tuple(texts)
        cached = self._cache.get(key)
        if cached is not None:
            return {
            "y": cached["y"].to(self.target_device, non_blocking=True),
            "y_mask": cached["y_mask"].to(self.target_device, non_blocking=True),
        }
        if self.encoder is None:
            raise RuntimeError(f"text encoder cache miss for texts={texts!r} after encoder release")
        model_args = self.encoder.encode(texts)
        cached = {
            "y": model_args["y"].cpu(),
            "y_mask": model_args["y_mask"].cpu(),
        }
        self._cache[key] = cached
        return {
            "y": cached["y"].to(self.target_device, non_blocking=True),
            "y_mask": cached["y_mask"].to(self.target_device, non_blocking=True),
        }

    def prime(self, texts: list[str]) -> None:
        self.encode(texts)

    def release_encoder(self) -> None:
        self.encoder = None


class InfiniteWorldRuntimeService:
    def __init__(self) -> None:
        os.environ.setdefault("HF_ENDPOINT", os.getenv("WM_HF_ENDPOINT", "https://hf-mirror.com"))
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = ""
        os.environ["HTTP_PROXY"] = ""
        os.environ["HTTPS_PROXY"] = ""
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        self.runtime = Runtime(
            gen_device=os.getenv("WM_INFINITEWORLD_GEN_DEVICE", "cuda:0"),
            decode_device=os.getenv("WM_INFINITEWORLD_DECODE_DEVICE", "cuda:1"),
        )
        self.config_path = Path(os.getenv("WM_INFINITEWORLD_CONFIG", str(INFWORLD_ROOT / "configs" / "infworld_config.yaml")))
        self.prompt = os.getenv(
            "WM_INFINITEWORLD_PROMPT",
            "A realistic immersive first-person world with coherent geometry, stable texture detail, and smooth motion.",
        )
        self.camera_deadzone = float(os.getenv("WM_INFINITEWORLD_CAMERA_DEADZONE", "0.18"))
        self.text_cfg_scale = float(os.getenv("WM_INFINITEWORLD_TEXT_CFG_SCALE", "5.0"))
        self.num_sampling_steps = int(os.getenv("WM_INFINITEWORLD_NUM_SAMPLING_STEPS", "10"))
        self.shift = float(os.getenv("WM_INFINITEWORLD_SHIFT", "7.0"))
        self.seed = int(os.getenv("WM_INFINITEWORLD_SEED", "42"))
        self.decode_window_latent = int(os.getenv("WM_INFINITEWORLD_DECODE_WINDOW_LATENT", "6"))
        self.decode_stride_latent = int(os.getenv("WM_INFINITEWORLD_DECODE_STRIDE_LATENT", "5"))
        self.max_cond_latent_frames = int(os.getenv("WM_INFINITEWORLD_MAX_COND_LATENT_FRAMES", "41"))
        self.warmup_on_start = os.getenv("WM_INFINITEWORLD_WARMUP_ON_START", "1") == "1"
        self.warmup_steps = int(os.getenv("WM_INFINITEWORLD_WARMUP_STEPS", "1"))
        self.release_text_encoder_after_prime = os.getenv("WM_INFINITEWORLD_RELEASE_TEXT_ENCODER_AFTER_PRIME", "1") == "1"

        self.args = None
        self.vae = None
        self.vae_decoder = None
        self.scheduler = None
        self.dit = None
        self.text_encoder = None
        self.bucket_config = None
        self.latent_size = None
        self._init_single_process_context_parallel()

    def _log(self, message: str) -> None:
        print(f"[service][infinite-world] {message}", flush=True)

    def _init_single_process_context_parallel(self) -> None:
        cp_util.dp_rank = 0
        cp_util.dp_size = 1
        cp_util.cp_rank = 0
        cp_util.cp_size = 1

    def health(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "model_id": "infinite-world",
            "ready": self.runtime.loaded,
            "session_id": None if self.runtime.session is None else self.runtime.session.session_id,
        }

    def load(self) -> Dict[str, Any]:
        if self.runtime.loaded:
            return {
                "model_id": "infinite-world",
                "status": "already_loaded",
                "device": self.runtime.gen_device,
                "decode_device": self.runtime.decode_device,
                "sampling_steps": self.num_sampling_steps,
            }

        args = OmegaConf.load(str(self.config_path))
        args.checkpoint_path = self._resolve_path(args.checkpoint_path)
        args.vae_cfg.vae_pth = self._resolve_path(args.vae_cfg.vae_pth)
        args.text_encoder_cfg.checkpoint_path = self._resolve_path(args.text_encoder_cfg.checkpoint_path)
        args.text_encoder_cfg.tokenizer_path = self._resolve_path(args.text_encoder_cfg.tokenizer_path)
        self.args = args

        gen_device = torch.device(self.runtime.gen_device)
        decode_device = torch.device(self.runtime.decode_device)
        torch.cuda.set_device(gen_device)
        self._setup_seed(self.seed)

        self._log(f"loading VAE on {gen_device}")
        self.vae = get_obj_from_str(args.vae_target)(device=str(gen_device), **args.vae_cfg).to(gen_device)
        self.vae_decoder = self.vae
        if decode_device != gen_device:
            self._log(f"loading decoder VAE on {decode_device}")
            self.vae_decoder = get_obj_from_str(args.vae_target)(device=str(decode_device), **args.vae_cfg).to(decode_device)

        self._log("loading text encoder on CPU")
        raw_text_encoder = get_obj_from_str(args.text_encoder_target)(device="cpu", **args.text_encoder_cfg)
        self.text_encoder = CachedTextEncoder(raw_text_encoder, gen_device)
        self._prime_text_cache()
        if self.release_text_encoder_after_prime:
            self.text_encoder.release_encoder()
            self._log("released raw text encoder after priming fixed prompt cache")

        self._log("loading scheduler")
        self.scheduler = get_obj_from_str(args.scheduler_target)(**args.val_scheduler_cfg)
        self.scheduler.num_sampling_steps = self.num_sampling_steps
        self.scheduler.shift = self.shift

        self._log("loading DiT")
        dtype = getattr(torch, args.amp_dtype)
        self.dit = get_obj_from_str(args.model_target)(
            out_channels=self.vae.out_channels,
            caption_channels=self.text_encoder.output_dim,
            model_max_length=self.text_encoder.model_max_length,
            enable_context_parallel=False,
            **args.model_cfg,
        ).to(dtype)
        state_dict = self._load_dit_state_dict(args.checkpoint_path)
        state_dict.pop("pos_embed_temporal", None)
        state_dict.pop("pos_embed", None)
        missing, unexpected = self.dit.load_state_dict(state_dict, strict=False)
        self._log(f"DiT loaded missing={len(missing)} unexpected={len(unexpected)}")
        self.dit = self.dit.to(gen_device).eval()

        from infworld.configs import bucket_config as bucket_config_module  # type: ignore

        self.bucket_config = getattr(bucket_config_module, "ASPECT_RATIO_627_F64")
        self.runtime.loaded = True
        return {
            "model_id": "infinite-world",
            "status": "loaded",
            "device": self.runtime.gen_device,
            "decode_device": self.runtime.decode_device,
            "sampling_steps": self.num_sampling_steps,
        }

    def start_session(self, init_image_base64: Optional[str]) -> Dict[str, Any]:
        if not self.runtime.loaded:
            self.load()

        init_image_bytes = self._decode_image(init_image_base64)
        if init_image_bytes is None:
            raise RuntimeError("init_image_base64 is required for Infinite-World start")

        seed_video = self._load_condition_image_from_bytes(init_image_bytes).to(torch.device(self.runtime.gen_device))
        with torch.no_grad():
            cond_latent = self.vae.encode(seed_video)
        latent_size = list(cond_latent.shape)
        latent_size[2] = 21
        self.latent_size = torch.Size(latent_size)

        seed_frame_b64 = self._frame_tensor_to_png_base64(seed_video[0, :, 0])
        session_id = str(uuid.uuid4())
        self.runtime.session = SessionState(
            session_id=session_id,
            init_image_bytes=init_image_bytes,
            seed_frame_b64=seed_frame_b64,
            video_buffer=seed_video.cpu(),
            latent_history=cond_latent.cpu(),
            move_indices=[],
            view_indices=[],
        )
        if self.warmup_on_start:
            self._warmup_with_session_latent(cond_latent)
        return {"session_id": session_id, "frame_base64": seed_frame_b64}

    def reset_session(self, session_id: str, init_image_base64: Optional[str]) -> Dict[str, Any]:
        session = self._require_session(session_id)
        init_image_bytes = self._decode_image(init_image_base64) or session.init_image_bytes
        return self.start_session(self._encode_image(init_image_bytes))

    def step(self, session_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        session = self._require_session(session_id)
        gen_device = torch.device(self.runtime.gen_device)
        t0 = time.perf_counter()

        move_label = self._movement_from_action(action)
        view_label = self._view_from_action(action)

        curr_start = session.video_buffer.shape[2] - 1
        curr_end = curr_start + self.args.validation_data.num_frames
        needed = curr_end - len(session.move_indices)
        repeat = max(1, needed)
        session.move_indices.extend([MOVE_ACTION_MAP[move_label]] * repeat)
        session.view_indices.extend([VIEW_ACTION_MAP[view_label]] * repeat)

        if self.max_cond_latent_frames > 0 and session.latent_history.shape[2] > self.max_cond_latent_frames:
            keep_tail = max(1, self.max_cond_latent_frames - 1)
            current_latent = torch.cat(
                [session.latent_history[:, :, :1], session.latent_history[:, :, -keep_tail:]], dim=2
            ).to(gen_device)
        else:
            current_latent = session.latent_history.to(gen_device)

        move = torch.tensor(session.move_indices[curr_start:curr_end], dtype=torch.long, device=gen_device)
        view = torch.tensor(session.view_indices[curr_start:curr_end], dtype=torch.long, device=gen_device)

        additional_args = {
            "image_cond": current_latent,
            "move": move.unsqueeze(0),
            "view": view.unsqueeze(0),
        }

        with torch.no_grad():
            samples = self.scheduler.sample(
                model=self.dit,
                text_encoder=self.text_encoder,
                null_embedder=self.dit.y_embedder,
                z_size=self.latent_size,
                prompts=[self.prompt],
                guidance_scale=self.text_cfg_scale,
                negative_prompts=[NEGATIVE_PROMPT],
                device=gen_device,
                additional_args=additional_args,
                progress=False,
            )

        session.latent_history = torch.cat([session.latent_history, samples[:, :, 1:].cpu()], dim=2)
        decoded_chunk = self._decode_latent_video_streaming(samples)
        session.video_buffer = torch.cat([session.video_buffer, decoded_chunk[:, :, 1:]], dim=2)
        session.step_count += 1

        last_frame_b64 = self._frame_tensor_to_png_base64(session.video_buffer[0, :, -1])
        latency_ms = int((time.perf_counter() - t0) * 1000)
        self._log(
            f"step done session_id={session.session_id} step={session.step_count} move={move_label} "
            f"view={view_label} latency_ms={latency_ms}"
        )
        return {
            "session_id": session.session_id,
            "frame_base64": last_frame_b64,
            "reward": 0.0,
            "ended": False,
            "truncated": False,
            "extra": {
                "latency_ms": latency_ms,
                "step_count": session.step_count,
                "move": move_label,
                "view": view_label,
                "sampling_steps": self.num_sampling_steps,
                "total_frames": int(session.video_buffer.shape[2]),
                "latent_frames": int(session.latent_history.shape[2]),
                "max_cond_latent_frames": self.max_cond_latent_frames,
            },
        }

    def _require_session(self, session_id: str) -> SessionState:
        if self.runtime.session is None:
            raise RuntimeError("Session is not started. Call /sessions/start first.")
        if self.runtime.session.session_id != session_id:
            raise RuntimeError("Unknown or expired session_id")
        return self.runtime.session

    def _prime_text_cache(self) -> None:
        if self.text_encoder is None:
            return
        # scheduler.sample encodes [prompt, negative_prompt] for n=1 under CFG.
        self.text_encoder.prime([self.prompt, NEGATIVE_PROMPT])

    def _warmup_with_session_latent(self, cond_latent: torch.Tensor) -> None:
        if self.scheduler is None or self.dit is None or self.text_encoder is None:
            return
        if self.warmup_steps <= 0:
            return
        gen_device = torch.device(self.runtime.gen_device)
        decode_device = torch.device(self.runtime.decode_device)
        self._log(f"running startup warmup steps={self.warmup_steps}")
        prev_steps = self.scheduler.num_sampling_steps
        try:
            self.scheduler.num_sampling_steps = self.warmup_steps
            current_latent = cond_latent.to(gen_device)
            zero_actions = torch.zeros((1, self.args.validation_data.num_frames), dtype=torch.long, device=gen_device)
            additional_args = {
                "image_cond": current_latent,
                "move": zero_actions,
                "view": zero_actions,
            }
            with torch.no_grad():
                samples = self.scheduler.sample(
                    model=self.dit,
                    text_encoder=self.text_encoder,
                    null_embedder=self.dit.y_embedder,
                    z_size=self.latent_size,
                    prompts=[self.prompt],
                    guidance_scale=self.text_cfg_scale,
                    negative_prompts=[NEGATIVE_PROMPT],
                    device=gen_device,
                    additional_args=additional_args,
                    progress=False,
                )
                # Warm both decode GPU and kernel caches with a minimal temporal slice.
                warm_latent = samples[:, :, : min(samples.shape[2], 2)].to(decode_device)
                _ = self.vae_decoder.decode(warm_latent).cpu()
            torch.cuda.synchronize(gen_device)
            if decode_device != gen_device:
                torch.cuda.synchronize(decode_device)
            self._log("startup warmup done")
        finally:
            self.scheduler.num_sampling_steps = prev_steps

    def _resolve_path(self, path: str) -> str:
        path = str(path)
        if not os.path.isabs(path):
            relative = Path(path)
            project_candidate = (PROJECT_ROOT / "checkpoints" / "infiniteworld" / relative.relative_to("checkpoints")).resolve() if relative.parts and relative.parts[0] == "checkpoints" else None
            if project_candidate is not None and project_candidate.exists():
                return str(project_candidate)
            return str((INFWORLD_ROOT / path).resolve())
        return path

    def _load_dit_state_dict(self, checkpoint_path: str) -> Dict[str, Any]:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        return state_dict

    def _setup_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _decode_image(self, payload: Optional[str]) -> Optional[bytes]:
        if not payload:
            return None
        if "," in payload:
            payload = payload.split(",", 1)[1]
        return base64.b64decode(payload)

    def _encode_image(self, image_bytes: bytes) -> str:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def _resize_and_center_crop(self, image: np.ndarray, target_size: tuple[int, int]) -> torch.Tensor:
        orig_h, orig_w = image.shape[:2]
        target_h, target_w = target_size
        scale = max(target_h / orig_h, target_w / orig_w)
        final_h = math.ceil(scale * orig_h)
        final_w = math.ceil(scale * orig_w)
        resized = cv2.resize(image, (final_w, final_h), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized)[None, ...].permute(0, 3, 1, 2).contiguous()
        cropped = TF.center_crop(tensor, target_size)
        return cropped[:, :, None, :, :]

    def _load_condition_image_from_bytes(self, image_bytes: bytes) -> torch.Tensor:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        frame = np.array(image)
        ratio = frame.shape[0] / frame.shape[1]
        closest_bucket = sorted(self.bucket_config.keys(), key=lambda x: abs(float(x) - ratio))[0]
        target_h, target_w = self.bucket_config[closest_bucket][0]
        tensor = self._resize_and_center_crop(frame, (target_h, target_w))
        return (tensor / 255 - 0.5) * 2

    def _decode_latent_video_streaming(self, latent_video: torch.Tensor) -> torch.Tensor:
        decode_device = torch.device(self.runtime.decode_device)
        total_t = latent_video.shape[2]
        if total_t <= self.decode_window_latent:
            return self.vae_decoder.decode(latent_video.to(decode_device)).cpu()

        decoded_parts = []
        start = 0
        while start < total_t:
            end = min(total_t, start + self.decode_window_latent)
            latent_slice = latent_video[:, :, start:end].to(decode_device)
            decoded_slice = self.vae_decoder.decode(latent_slice).cpu()
            if not decoded_parts:
                decoded_parts.append(decoded_slice)
            else:
                decoded_parts.append(decoded_slice[:, :, 1:])
            if end == total_t:
                break
            start += self.decode_stride_latent
        return torch.cat(decoded_parts, dim=2)

    def _movement_from_action(self, action: Dict[str, Any]) -> str:
        forward = bool(action.get("w")) and not bool(action.get("s"))
        backward = bool(action.get("s")) and not bool(action.get("w"))
        left = bool(action.get("a")) and not bool(action.get("d"))
        right = bool(action.get("d")) and not bool(action.get("a"))
        if forward and left:
            return "go forward and go left"
        if forward and right:
            return "go forward and go right"
        if backward and left:
            return "go back and go left"
        if backward and right:
            return "go back and go right"
        if forward:
            return "go forward"
        if backward:
            return "go back"
        if left:
            return "go left"
        if right:
            return "go right"
        return "no-op"

    def _view_from_action(self, action: Dict[str, Any]) -> str:
        dx = float(action.get("camera_dx", 0.0) or 0.0)
        dy = float(action.get("camera_dy", 0.0) or 0.0)
        left = dx <= -self.camera_deadzone
        right = dx >= self.camera_deadzone
        up = dy <= -self.camera_deadzone
        down = dy >= self.camera_deadzone
        if up and left:
            return "turn up and turn left"
        if up and right:
            return "turn up and turn right"
        if down and left:
            return "turn down and turn left"
        if down and right:
            return "turn down and turn right"
        if up:
            return "turn up"
        if down:
            return "turn down"
        if left:
            return "turn left"
        if right:
            return "turn right"
        return "no-op"

    def _frame_tensor_to_png_base64(self, frame_chw: torch.Tensor) -> str:
        frame = frame_chw.detach().float().clamp(-1, 1)
        frame = frame.add(1.0).div(2.0).mul(255.0).byte().permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


svc = InfiniteWorldRuntimeService()
app = FastAPI(title="WMFactory Infinite-World Service")


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


@app.post("/sessions/reset")
def reset(req: ResetRequest) -> Dict[str, Any]:
    try:
        return svc.reset_session(req.session_id, req.init_image_base64)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/sessions/step")
def step(req: StepRequest) -> Dict[str, Any]:
    try:
        return svc.step(req.session_id, req.action)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
