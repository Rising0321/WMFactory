from __future__ import annotations

import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Type

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
WEB_DIR = ROOT / "web"
LOG_DIR = ROOT / "logs"

if str(FRONTEND_ROOT) not in sys.path:
    sys.path.insert(0, str(FRONTEND_ROOT))

import server as shared_gateway
from adapters import (
    DiamondAdapter,
    GameCraftAdapter,
    InfiniteWorldAdapter,
    LingBotWorldAdapter,
    MatrixGameAdapter,
    MineWorldAdapter,
    OpenOasisAdapter,
    Vid2WorldAdapter,
    WhamAdapter,
    WonderWorldAdapter,
    WorldFMAdapter,
    WorldPlayAdapter,
    YumeAdapter,
)
from adapters.runtime_utils import _parse_nvidia_smi


@dataclass(frozen=True)
class ModelConfig:
    adapter_cls: Type[Any]
    env_prefix: str
    default_port: int
    gpu_visible_envs: tuple[str, ...] = ()
    gpu_index_envs: tuple[str, ...] = ()


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "diamond": ModelConfig(DiamondAdapter, "DIAMOND", 9001),
    "gamecraft": ModelConfig(GameCraftAdapter, "GAMECRAFT", 9012, gpu_visible_envs=("WM_GAMECRAFT_CUDA_VISIBLE_DEVICES",)),
    "infinite-world": ModelConfig(
        InfiniteWorldAdapter,
        "INFINITEWORLD",
        9011,
        gpu_visible_envs=("WM_INFINITEWORLD_CUDA_VISIBLE_DEVICES",),
    ),
    "lingbot-world": ModelConfig(LingBotWorldAdapter, "LINGBOTWORLD", 9014),
    "matrixgame": ModelConfig(MatrixGameAdapter, "MATRIXGAME", 9003, gpu_index_envs=("WM_MATRIXGAME_GPU_INDEX",)),
    "mineworld": ModelConfig(MineWorldAdapter, "MINEWORLD", 9005),
    "open-oasis": ModelConfig(OpenOasisAdapter, "OPENOASIS", 9005),
    "vid2world": ModelConfig(Vid2WorldAdapter, "VID2WORLD", 9010),
    "wham": ModelConfig(WhamAdapter, "WHAM", 9007, gpu_index_envs=("WM_WHAM_GPU_INDEX",)),
    "wonderworld": ModelConfig(WonderWorldAdapter, "WONDERWORLD", 9004),
    "worldfm": ModelConfig(WorldFMAdapter, "WORLDFM", 9002, gpu_index_envs=("WM_WORLDFM_GPU_INDEX",)),
    "worldplay": ModelConfig(WorldPlayAdapter, "WORLDPLAY", 9009, gpu_visible_envs=("WM_WORLDPLAY_CUDA_VISIBLE_DEVICES",)),
    "yume": ModelConfig(YumeAdapter, "YUME", 9008),
}

SIDE_PORT_OFFSET = {"left": 1000, "right": 2000}
ENV_LOCK = threading.Lock()


class ArenaLoadRequest(BaseModel):
    left_model_id: str
    right_model_id: str


class ArenaStartRequest(BaseModel):
    init_image_base64: str


class ArenaStepRequest(BaseModel):
    left_action: Dict[str, Any] = Field(default_factory=dict)
    right_action: Dict[str, Any] = Field(default_factory=dict)


class ArenaResetRequest(BaseModel):
    init_image_base64: Optional[str] = None


@dataclass
class SideState:
    side: str
    model_id: Optional[str] = None
    adapter: Optional[Any] = None
    env_overrides: Dict[str, str] = field(default_factory=dict)
    gpu_index: Optional[int] = None
    visible_devices: str = ""
    port: Optional[int] = None
    device: Optional[str] = None
    loaded: bool = False
    session_id: Optional[str] = None
    last_frame_base64: Optional[str] = None


@dataclass
class ArenaState:
    left: SideState = field(default_factory=lambda: SideState("left"))
    right: SideState = field(default_factory=lambda: SideState("right"))
    init_image_base64: Optional[str] = None


state = ArenaState()
app = FastAPI(title="WMArena")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@contextmanager
def _patched_environ(overrides: Dict[str, str]):
    previous: Dict[str, Optional[str]] = {}
    for key, value in overrides.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _require_model(model_id: str) -> ModelConfig:
    cfg = MODEL_CONFIGS.get(model_id)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"Unsupported model '{model_id}'")
    return cfg


def _split_idle_gpu_groups() -> tuple[list[int], list[int]]:
    left_gpus = os.getenv("WMARENA_LEFT_GPUS")
    right_gpus = os.getenv("WMARENA_RIGHT_GPUS")
    if left_gpus is not None and right_gpus is not None:
        left = [int(part.strip()) for part in left_gpus.split(",") if part.strip()]
        right = [int(part.strip()) for part in right_gpus.split(",") if part.strip()]
        if not left or not right:
            raise HTTPException(status_code=400, detail="WMARENA_LEFT_GPUS/WMARENA_RIGHT_GPUS cannot be empty")
        if set(left) & set(right):
            raise HTTPException(status_code=400, detail="WMARENA_LEFT_GPUS and WMARENA_RIGHT_GPUS must not overlap")
        return left, right

    try:
        rows = _parse_nvidia_smi()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to inspect GPUs via nvidia-smi: {exc}") from exc

    if len(rows) < 2:
        raise HTTPException(status_code=400, detail="WMArena requires at least 2 visible GPUs")

    idle_free_mb = int(os.getenv("WMARENA_IDLE_FREE_MB", "20480"))
    idle = sorted(
        [
            int(row["index"])
            for row in rows
            if float(row.get("memory_total", 0.0)) - float(row.get("memory_used", 0.0)) > idle_free_mb
        ]
    )
    if len(idle) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"WMArena needs at least 2 idle GPUs with free memory > {idle_free_mb} MB",
        )

    midpoint = len(idle) // 2
    left = idle[:midpoint]
    right = idle[midpoint:]
    if not left or not right:
        raise HTTPException(status_code=400, detail="Idle GPU split failed; need GPUs on both sides")
    return left, right


def _build_side_env(model_id: str, side: str, visible_devices: list[int]) -> Dict[str, str]:
    cfg = _require_model(model_id)
    port = cfg.default_port + SIDE_PORT_OFFSET[side]
    host = "127.0.0.1"
    visible = ",".join(str(idx) for idx in visible_devices)
    env = {
        "WM_AUTO_CUDA_VISIBLE_DEVICES": "0",
        "CUDA_VISIBLE_DEVICES": visible,
        f"WM_{cfg.env_prefix}_HOST": host,
        f"WM_{cfg.env_prefix}_PORT": str(port),
        f"WM_{cfg.env_prefix}_URL": f"http://{host}:{port}",
        f"WM_{cfg.env_prefix}_LOG": str(LOG_DIR / f"{model_id}_{side}.log"),
        f"WM_{cfg.env_prefix}_CUDA_VISIBLE_DEVICES": visible,
    }
    for extra_key in cfg.gpu_visible_envs:
        env[extra_key] = visible
    for extra_key in cfg.gpu_index_envs:
        env[extra_key] = "0"
    return env


def _call_with_side_env(side_state: SideState, method_name: str, *args: Any) -> Any:
    if side_state.adapter is None:
        raise HTTPException(status_code=400, detail=f"{side_state.side} side is not loaded")
    with ENV_LOCK:
        with _patched_environ(side_state.env_overrides):
            method = getattr(side_state.adapter, method_name)
            return method(*args)


def _make_side_state(side: str, model_id: str, visible_devices: list[int]) -> SideState:
    cfg = _require_model(model_id)
    env_overrides = _build_side_env(model_id, side, visible_devices)
    with ENV_LOCK:
        with _patched_environ(env_overrides):
            adapter = cfg.adapter_cls()
    return SideState(
        side=side,
        model_id=model_id,
        adapter=adapter,
        env_overrides=env_overrides,
        gpu_index=visible_devices[0],
        visible_devices=",".join(str(idx) for idx in visible_devices),
        port=cfg.default_port + SIDE_PORT_OFFSET[side],
    )


def _load_side(side_state: SideState) -> Dict[str, Any]:
    data = _call_with_side_env(side_state, "load")
    side_state.loaded = True
    side_state.device = str(data.get("device", f"cuda:{side_state.gpu_index}"))
    return data


def _start_side(side_state: SideState, init_image_bytes: bytes) -> Dict[str, Any]:
    data = _call_with_side_env(side_state, "start_session", init_image_bytes)
    side_state.session_id = data.get("session_id")
    side_state.last_frame_base64 = data.get("frame_base64")
    return data


def _reset_side(side_state: SideState, init_image_bytes: Optional[bytes]) -> Dict[str, Any]:
    data = _call_with_side_env(side_state, "reset_session", init_image_bytes)
    side_state.session_id = data.get("session_id", side_state.session_id)
    side_state.last_frame_base64 = data.get("frame_base64")
    return data


def _step_side(side_state: SideState, action: Dict[str, Any]) -> Dict[str, Any]:
    result = _call_with_side_env(side_state, "step", action)
    side_state.last_frame_base64 = result.frame_base64
    return {
        "session_id": side_state.session_id,
        "frame_base64": result.frame_base64,
        "reward": result.reward,
        "ended": result.ended,
        "truncated": result.truncated,
        "extra": result.extra,
    }


@app.get("/api/models")
def list_models() -> Dict[str, Any]:
    return shared_gateway.list_models()


@app.get("/api/datasets")
def list_datasets() -> Dict[str, Any]:
    return shared_gateway.list_datasets()


@app.post("/api/datasets/random-image")
def random_dataset_image(req: shared_gateway.RandomDatasetImageRequest) -> Dict[str, Any]:
    return shared_gateway.random_dataset_image(req)


@app.post("/api/arena/load")
def arena_load(req: ArenaLoadRequest) -> Dict[str, Any]:
    if req.left_model_id == req.right_model_id:
        raise HTTPException(status_code=400, detail="Left and right models must be different in phase 1")

    left_gpus, right_gpus = _split_idle_gpu_groups()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    left_state = _make_side_state("left", req.left_model_id, left_gpus)
    right_state = _make_side_state("right", req.right_model_id, right_gpus)

    try:
        left_result = _load_side(left_state)
        right_result = _load_side(right_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    state.left = left_state
    state.right = right_state
    state.init_image_base64 = None
    return {
        "left": {
            "model_id": req.left_model_id,
            "gpu_index": left_state.gpu_index,
            "visible_devices": left_state.visible_devices,
            "port": left_state.port,
            "device": left_result.get("device", "cuda:0"),
            "status": left_result.get("status", "loaded"),
        },
        "right": {
            "model_id": req.right_model_id,
            "gpu_index": right_state.gpu_index,
            "visible_devices": right_state.visible_devices,
            "port": right_state.port,
            "device": right_result.get("device", "cuda:0"),
            "status": right_result.get("status", "loaded"),
        },
    }


@app.post("/api/arena/start")
def arena_start(req: ArenaStartRequest) -> Dict[str, Any]:
    if not state.left.loaded or not state.right.loaded:
        raise HTTPException(status_code=400, detail="Load both models before starting the arena")

    image_bytes = shared_gateway._decode_image(req.init_image_base64)
    if image_bytes is None:
        raise HTTPException(status_code=400, detail="init_image_base64 is required")

    try:
        left_result = _start_side(state.left, image_bytes)
        right_result = _start_side(state.right, image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    state.init_image_base64 = req.init_image_base64
    return {"left": left_result, "right": right_result}


@app.post("/api/arena/reset")
def arena_reset(req: ArenaResetRequest) -> Dict[str, Any]:
    if state.left.session_id is None or state.right.session_id is None:
        raise HTTPException(status_code=400, detail="No active arena session")

    payload = req.init_image_base64 or state.init_image_base64
    image_bytes = shared_gateway._decode_image(payload)

    try:
        left_result = _reset_side(state.left, image_bytes)
        right_result = _reset_side(state.right, image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if payload:
        state.init_image_base64 = payload
    return {"left": left_result, "right": right_result}


@app.post("/api/arena/step")
def arena_step(req: ArenaStepRequest) -> Dict[str, Any]:
    if state.left.session_id is None or state.right.session_id is None:
        raise HTTPException(status_code=400, detail="No active arena session")

    try:
        left_result = _step_side(state.left, dict(req.left_action))
        right_result = _step_side(state.right, dict(req.right_action))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"left": left_result, "right": right_result}


@app.get("/")
def root() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="wmarena-web")
