from __future__ import annotations

import base64
import copy
import io
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from einops import rearrange
from fastapi import FastAPI, HTTPException
from omegaconf import OmegaConf
from PIL import Image
from pydantic import BaseModel
from torchvision.transforms import v2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATRIX_ROOT = PROJECT_ROOT / "vendors" / "matrixgame2"
MATRIX_CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "matrixgame2"

if str(MATRIX_ROOT) not in sys.path:
    sys.path.insert(0, str(MATRIX_ROOT))

from demo_utils.constant import ZERO_VAE_CACHE  # type: ignore
from demo_utils.vae_block3 import VAEDecoderWrapper  # type: ignore
from pipeline import CausalInferenceStreamingPipeline  # type: ignore
from pipeline.causal_inference import cond_current  # type: ignore
from safetensors.torch import load_file  # type: ignore
from utils.wan_wrapper import WanDiffusionWrapper  # type: ignore
from wan.vae.wanx_vae import get_wanx_vae_wrapper  # type: ignore


class LoadRequest(BaseModel):
    model_id: Optional[str] = "matrixgame"


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
    created_at: float
    seed_image_bytes: bytes
    conditional_dict: Dict[str, torch.Tensor]
    noise: torch.Tensor
    current_start_frame: int
    vae_cache: list[Any]


@dataclass
class Runtime:
    device: torch.device
    weight_dtype: torch.dtype
    mode: str
    max_num_output_frames: int
    num_frame_per_block: int
    keyboard_dim: int
    enable_mouse: bool
    camera_scale: float
    pipeline: CausalInferenceStreamingPipeline
    vae: Any
    frame_process: Any
    session: Optional[SessionState] = None


class MatrixGameRuntimeService:
    def __init__(self) -> None:
        os.environ.setdefault("HF_ENDPOINT", os.getenv("WM_HF_ENDPOINT", "https://hf-mirror.com"))
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = ""
        os.environ["HTTP_PROXY"] = ""
        os.environ["HTTPS_PROXY"] = ""

        self.runtime: Optional[Runtime] = None

    def load(self) -> Dict[str, Any]:
        self._log("load requested")
        if self.runtime is not None:
            return {
                "model_id": "matrixgame",
                "status": "already_loaded",
                "device": str(self.runtime.device),
                "mode": self.runtime.mode,
            }

        config_path = Path(
            os.getenv(
                "WM_MATRIXGAME_CONFIG",
                str(MATRIX_ROOT / "configs" / "inference_yaml" / "inference_universal.yaml"),
            )
        )
        pretrained_model_path = Path(
            os.getenv("WM_MATRIXGAME_PRETRAINED_PATH", str(MATRIX_CHECKPOINT_ROOT))
        )
        checkpoint_path = Path(os.getenv("WM_MATRIXGAME_CHECKPOINT_PATH", str(pretrained_model_path / "base_distilled_model" / "base_distill.safetensors")))

        config = OmegaConf.load(str(config_path))
        model_cfg = getattr(getattr(config, "model_kwargs", None), "model_config", None)
        if model_cfg and not Path(str(model_cfg)).is_absolute():
            config.model_kwargs.model_config = str((MATRIX_ROOT / str(model_cfg)).resolve())
        mode = str(getattr(config, "mode", "universal"))
        if mode not in {"universal", "gta_drive", "templerun"}:
            raise RuntimeError(f"Unsupported matrixgame mode '{mode}'")

        if torch.cuda.is_available():
            gpu_index = int(os.getenv("WM_MATRIXGAME_GPU_INDEX", "0"))
            device = torch.device(f"cuda:{gpu_index}")
            torch.cuda.set_device(device)
        else:
            raise RuntimeError("MatrixGame2.0 requires CUDA GPU")

        weight_dtype = torch.bfloat16

        generator = WanDiffusionWrapper(**getattr(config, "model_kwargs", {}), is_causal=True)
        current_vae_decoder = VAEDecoderWrapper()

        vae_state_dict = torch.load(pretrained_model_path / "Wan2.1_VAE.pth", map_location="cpu")
        decoder_state_dict = {k: v for k, v in vae_state_dict.items() if ("decoder." in k or "conv2" in k)}
        current_vae_decoder.load_state_dict(decoder_state_dict)
        current_vae_decoder.to(device, torch.float16)
        current_vae_decoder.requires_grad_(False)
        current_vae_decoder.eval()
        if os.getenv("WM_MATRIXGAME_ENABLE_COMPILE", "1") == "1":
            current_vae_decoder.compile(mode="max-autotune-no-cudagraphs")

        pipeline = CausalInferenceStreamingPipeline(config, generator=generator, vae_decoder=current_vae_decoder)
        self._log("loading distilled checkpoint")
        state_dict = load_file(str(checkpoint_path))
        pipeline.generator.load_state_dict(state_dict)
        pipeline = pipeline.to(device=device, dtype=weight_dtype)
        pipeline.vae_decoder.to(torch.float16)

        vae = get_wanx_vae_wrapper(str(pretrained_model_path), torch.float16)
        vae.requires_grad_(False)
        vae.eval()
        vae = vae.to(device, weight_dtype)

        max_num_output_frames = int(os.getenv("WM_MATRIXGAME_MAX_NUM_OUTPUT_FRAMES", "48"))
        num_frame_per_block = int(getattr(config, "num_frame_per_block", 3))
        if max_num_output_frames % num_frame_per_block != 0:
            raise RuntimeError(
                f"WM_MATRIXGAME_MAX_NUM_OUTPUT_FRAMES ({max_num_output_frames}) must be divisible by num_frame_per_block ({num_frame_per_block})"
            )

        if mode == "universal":
            keyboard_dim, enable_mouse = 4, True
        elif mode == "gta_drive":
            keyboard_dim, enable_mouse = 2, True
        else:
            keyboard_dim, enable_mouse = 7, False

        camera_scale = float(os.getenv("WM_MATRIXGAME_CAMERA_SCALE", "0.1"))

        frame_process = v2.Compose(
            [
                v2.Resize(size=(352, 640), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.runtime = Runtime(
            device=device,
            weight_dtype=weight_dtype,
            mode=mode,
            max_num_output_frames=max_num_output_frames,
            num_frame_per_block=num_frame_per_block,
            keyboard_dim=keyboard_dim,
            enable_mouse=enable_mouse,
            camera_scale=camera_scale,
            pipeline=pipeline,
            vae=vae,
            frame_process=frame_process,
            session=None,
        )
        self._log(f"load done device={device}, mode={mode}, max_latent_frames={max_num_output_frames}")
        return {
            "model_id": "matrixgame",
            "status": "loaded",
            "device": str(device),
            "mode": mode,
            "max_num_output_frames": max_num_output_frames,
            "num_frame_per_block": num_frame_per_block,
        }

    def start_session(self, init_image_base64: Optional[str]) -> Dict[str, Any]:
        runtime = self._require_runtime()
        init_image_bytes = self._decode_image(init_image_base64)
        if init_image_bytes is None:
            raise RuntimeError("init_image_base64 is required for matrixgame start")

        session_id = str(uuid.uuid4())
        runtime.session = self._create_fresh_session(runtime, init_image_bytes, session_id=session_id)

        # Ensure start only returns when the runtime has passed one real step path
        # (diffusion + decoder + KV-cache update), then restore to clean initial state.
        if os.getenv("WM_MATRIXGAME_WARMUP_ON_START", "1") == "1":
            self._log(f"start_session warmup begin session_id={session_id}")
            self.step(session_id, self._neutral_action_payload(runtime.mode))
            runtime.session = self._create_fresh_session(runtime, init_image_bytes, session_id=session_id)
            if runtime.device.type == "cuda":
                torch.cuda.synchronize(runtime.device)
            self._log(f"start_session warmup done session_id={session_id}")

        preview = self._image_bytes_to_frame_base64(init_image_bytes)
        self._log(f"start_session done session_id={session_id}")
        return {"session_id": session_id, "frame_base64": preview}

    def reset_session(self, session_id: str, init_image_base64: Optional[str]) -> Dict[str, Any]:
        runtime = self._require_runtime()
        session = self._require_session(runtime, session_id)
        init_image_bytes = self._decode_image(init_image_base64) or session.seed_image_bytes

        data = self.start_session(self._encode_image(init_image_bytes))
        return {
            "session_id": data["session_id"],
            "frame_base64": data["frame_base64"],
        }

    @torch.inference_mode()
    def step(self, session_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        runtime = self._require_runtime()
        session = self._require_session(runtime, session_id)

        if session.current_start_frame >= runtime.max_num_output_frames:
            return {
                "session_id": session_id,
                "frame_base64": self._image_bytes_to_frame_base64(session.seed_image_bytes),
                "reward": 0.0,
                "ended": True,
                "truncated": False,
                "extra": {"reason": "max_frames_reached"},
            }

        current_num_frames = runtime.num_frame_per_block
        noisy_input = session.noise[:, :, session.current_start_frame : session.current_start_frame + current_num_frames]

        current_actions = self._map_action(runtime, action)
        new_cond, session.conditional_dict = cond_current(
            session.conditional_dict,
            session.current_start_frame,
            runtime.num_frame_per_block,
            replace=current_actions,
            mode=runtime.mode,
        )

        denoised_pred: Optional[torch.Tensor] = None
        for idx, current_timestep in enumerate(runtime.pipeline.denoising_step_list):
            timestep = torch.ones([1, current_num_frames], device=runtime.device, dtype=torch.int64) * current_timestep

            if idx < len(runtime.pipeline.denoising_step_list) - 1:
                _, denoised_pred = runtime.pipeline.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=new_cond,
                    timestep=timestep,
                    kv_cache=runtime.pipeline.kv_cache1,
                    kv_cache_mouse=runtime.pipeline.kv_cache_mouse,
                    kv_cache_keyboard=runtime.pipeline.kv_cache_keyboard,
                    crossattn_cache=runtime.pipeline.crossattn_cache,
                    current_start=session.current_start_frame * runtime.pipeline.frame_seq_length,
                )
                next_timestep = runtime.pipeline.denoising_step_list[idx + 1]
                noisy_input = runtime.pipeline.scheduler.add_noise(
                    rearrange(denoised_pred, "b c f h w -> (b f) c h w"),
                    torch.randn_like(rearrange(denoised_pred, "b c f h w -> (b f) c h w")),
                    next_timestep * torch.ones([1 * current_num_frames], device=runtime.device, dtype=torch.long),
                )
                noisy_input = rearrange(noisy_input, "(b f) c h w -> b c f h w", b=1)
            else:
                _, denoised_pred = runtime.pipeline.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=new_cond,
                    timestep=timestep,
                    kv_cache=runtime.pipeline.kv_cache1,
                    kv_cache_mouse=runtime.pipeline.kv_cache_mouse,
                    kv_cache_keyboard=runtime.pipeline.kv_cache_keyboard,
                    crossattn_cache=runtime.pipeline.crossattn_cache,
                    current_start=session.current_start_frame * runtime.pipeline.frame_seq_length,
                )

        assert denoised_pred is not None
        context_timestep = torch.ones_like(timestep) * runtime.pipeline.args.context_noise
        runtime.pipeline.generator(
            noisy_image_or_video=denoised_pred,
            conditional_dict=new_cond,
            timestep=context_timestep,
            kv_cache=runtime.pipeline.kv_cache1,
            kv_cache_mouse=runtime.pipeline.kv_cache_mouse,
            kv_cache_keyboard=runtime.pipeline.kv_cache_keyboard,
            crossattn_cache=runtime.pipeline.crossattn_cache,
            current_start=session.current_start_frame * runtime.pipeline.frame_seq_length,
        )

        decoded_clip, session.vae_cache = runtime.pipeline.vae_decoder(denoised_pred.transpose(1, 2).half(), *session.vae_cache)
        frame_base64 = self._clip_last_frame_to_base64(decoded_clip)

        session.current_start_frame += current_num_frames
        ended = session.current_start_frame >= runtime.max_num_output_frames
        total_rgb_frames = 1 + 4 * max(0, session.current_start_frame - 1)

        return {
            "session_id": session_id,
            "frame_base64": frame_base64,
            "reward": 0.0,
            "ended": ended,
            "truncated": False,
            "extra": {
                "mode": runtime.mode,
                "latent_frames_done": session.current_start_frame,
                "latent_frames_total": runtime.max_num_output_frames,
                "rgb_frames_done_approx": total_rgb_frames,
            },
        }

    def health(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "model_id": "matrixgame",
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

    def _encode_image(self, image_bytes: bytes) -> str:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def _log(self, message: str) -> None:
        print(f"[service][matrixgame] {message}", flush=True)

    def _resizecrop(self, image: Image.Image, th: int, tw: int) -> Image.Image:
        w, h = image.size
        if h / w > th / tw:
            new_w = int(w)
            new_h = int(new_w * th / tw)
        else:
            new_h = int(h)
            new_w = int(new_h * tw / th)
        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = (w + new_w) / 2
        bottom = (h + new_h) / 2
        return image.crop((left, top, right, bottom))

    def _build_initial_conditions(self, runtime: Runtime, init_image_bytes: bytes) -> Dict[str, torch.Tensor]:
        image = Image.open(io.BytesIO(init_image_bytes)).convert("RGB")
        image = self._resizecrop(image, 352, 640)
        image = runtime.frame_process(image)[None, :, None, :, :].to(dtype=runtime.weight_dtype, device=runtime.device)

        padding_video = torch.zeros_like(image).repeat(1, 1, 4 * (runtime.max_num_output_frames - 1), 1, 1)
        img_cond = torch.concat([image, padding_video], dim=2)
        tiler_kwargs = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
        img_cond = runtime.vae.encode(img_cond, device=runtime.device, **tiler_kwargs).to(runtime.device)

        mask_cond = torch.ones_like(img_cond)
        mask_cond[:, :, 1:] = 0
        cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)
        visual_context = runtime.vae.clip.encode_video(image)

        total_rgb_frames = 1 + 4 * (runtime.max_num_output_frames - 1)
        conditional_dict: Dict[str, torch.Tensor] = {
            "cond_concat": cond_concat.to(device=runtime.device, dtype=runtime.weight_dtype),
            "visual_context": visual_context.to(device=runtime.device, dtype=runtime.weight_dtype),
            "keyboard_cond": torch.zeros((1, total_rgb_frames, runtime.keyboard_dim), device=runtime.device, dtype=runtime.weight_dtype),
        }
        if runtime.enable_mouse:
            conditional_dict["mouse_cond"] = torch.zeros((1, total_rgb_frames, 2), device=runtime.device, dtype=runtime.weight_dtype)
        return conditional_dict

    def _create_fresh_session(self, runtime: Runtime, init_image_bytes: bytes, session_id: Optional[str] = None) -> SessionState:
        conditional_dict = self._build_initial_conditions(runtime, init_image_bytes)
        sampled_noise = torch.randn(
            [1, 16, runtime.max_num_output_frames, 44, 80], device=runtime.device, dtype=runtime.weight_dtype
        )

        runtime.pipeline.kv_cache1 = None
        runtime.pipeline.kv_cache_mouse = None
        runtime.pipeline.kv_cache_keyboard = None
        runtime.pipeline.crossattn_cache = None
        runtime.pipeline._initialize_kv_cache(batch_size=1, dtype=runtime.weight_dtype, device=runtime.device)
        runtime.pipeline._initialize_kv_cache_mouse_and_keyboard(batch_size=1, dtype=runtime.weight_dtype, device=runtime.device)
        runtime.pipeline._initialize_crossattn_cache(batch_size=1, dtype=runtime.weight_dtype, device=runtime.device)

        vae_cache = copy.deepcopy(ZERO_VAE_CACHE)
        for j in range(len(vae_cache)):
            vae_cache[j] = None

        return SessionState(
            session_id=session_id or str(uuid.uuid4()),
            created_at=time.time(),
            seed_image_bytes=init_image_bytes,
            conditional_dict=conditional_dict,
            noise=sampled_noise,
            current_start_frame=0,
            vae_cache=vae_cache,
        )

    def _neutral_action_payload(self, mode: str) -> Dict[str, Any]:
        if mode == "templerun":
            return {}
        return {
            "w": False,
            "a": False,
            "s": False,
            "d": False,
            "camera_dx": 0.0,
            "camera_dy": 0.0,
        }

    def _map_action(self, runtime: Runtime, action: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        mode = runtime.mode
        if mode == "universal":
            keyboard = torch.zeros(4, device=runtime.device, dtype=runtime.weight_dtype)
            if bool(action.get("w")):
                keyboard[0] = 1
            if bool(action.get("s")):
                keyboard[1] = 1
            if bool(action.get("a")):
                keyboard[2] = 1
            if bool(action.get("d")):
                keyboard[3] = 1

            dx = float(action.get("camera_dx", 0.0) or 0.0)
            dy = float(action.get("camera_dy", 0.0) or 0.0)
            dx = max(-1.0, min(1.0, dx))
            dy = max(-1.0, min(1.0, dy))
            mouse = torch.tensor(
                [(-dy) * runtime.camera_scale, dx * runtime.camera_scale],
                device=runtime.device,
                dtype=runtime.weight_dtype,
            )
            return {"keyboard": keyboard, "mouse": mouse}

        if mode == "gta_drive":
            keyboard = torch.zeros(2, device=runtime.device, dtype=runtime.weight_dtype)
            if bool(action.get("w")):
                keyboard[0] = 1
            if bool(action.get("s")):
                keyboard[1] = 1

            steer_from_keys = 0.0
            if bool(action.get("a")):
                steer_from_keys -= 1.0
            if bool(action.get("d")):
                steer_from_keys += 1.0
            steer = steer_from_keys + float(action.get("camera_dx", 0.0) or 0.0)
            steer = max(-1.0, min(1.0, steer))
            mouse = torch.tensor([0.0, steer * runtime.camera_scale], device=runtime.device, dtype=runtime.weight_dtype)
            return {"keyboard": keyboard, "mouse": mouse}

        keyboard = torch.zeros(7, device=runtime.device, dtype=runtime.weight_dtype)
        if bool(action.get("w")):
            keyboard[1] = 1  # jump
        elif bool(action.get("s")):
            keyboard[2] = 1  # slide
        elif bool(action.get("a")):
            keyboard[5] = 1  # leftside
        elif bool(action.get("d")):
            keyboard[6] = 1  # rightside
        else:
            keyboard[0] = 1  # nomove
        return {"keyboard": keyboard}

    def _clip_last_frame_to_base64(self, clip: torch.Tensor) -> str:
        frame = clip[0, -1].detach().float().clamp(-1, 1)
        arr = ((frame + 1.0) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _image_bytes_to_frame_base64(self, image_bytes: bytes) -> str:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = self._resizecrop(image, 352, 640)
        arr = np.asarray(image, dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


svc = MatrixGameRuntimeService()
app = FastAPI(title="WMFactory MatrixGame2.0 Service")


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
