from __future__ import annotations

import base64
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from adapters import DiamondAdapter, GameCraftAdapter, InfiniteWorldAdapter, LingBotWorldAdapter, MatrixGameAdapter, MatrixGame3Adapter, MineWorldAdapter, OpenOasisAdapter, Vid2WorldAdapter, WhamAdapter, WonderWorldAdapter, WorldFMAdapter, WorldPlayAdapter, YumeAdapter


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
DATA_DIR = ROOT.parent / "data"


class LoadModelRequest(BaseModel):
    model_id: str


class StartSessionRequest(BaseModel):
    model_id: str
    init_image_base64: Optional[str] = None


class StepRequest(BaseModel):
    session_id: str
    action: Dict[str, Any] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    session_id: str
    init_image_base64: Optional[str] = None


class ProgressRequest(BaseModel):
    model_id: Optional[str] = None
    request_id: Optional[str] = None


class RandomDatasetImageRequest(BaseModel):
    dataset_id: str


class AppState:
    def __init__(self) -> None:
        self.adapters = {
            "diamond": DiamondAdapter(),
            "gamecraft": GameCraftAdapter(),
            "infinite-world": InfiniteWorldAdapter(),
            "lingbot-world": LingBotWorldAdapter(),
            "vid2world": Vid2WorldAdapter(),
            "wham": WhamAdapter(),
            "mineworld": MineWorldAdapter(),
            "matrixgame": MatrixGameAdapter(),
            "matrixgame3": MatrixGame3Adapter(),
            "yume": YumeAdapter(),
            "worldplay": WorldPlayAdapter(),
            "open-oasis": OpenOasisAdapter(),
            "worldfm": WorldFMAdapter(),
            "wonderworld": WonderWorldAdapter(),
        }
        self.active_model_id: Optional[str] = None


state = AppState()
app = FastAPI(title="WMFactory Unified Frontend API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _decode_image(payload: Optional[str]) -> Optional[bytes]:
    if not payload:
        return None
    if "," in payload:
        payload = payload.split(",", 1)[1]
    return base64.b64decode(payload)


def _require_adapter(model_id: str):
    adapter = state.adapters.get(model_id)
    if adapter is None:
        raise HTTPException(status_code=404, detail=f"Unsupported model '{model_id}'")
    return adapter


def _dataset_images(dataset_id: str) -> List[Path]:
    path = DATA_DIR / dataset_id
    if not path.exists() and DATA_DIR.exists():
        # Allow case-insensitive dataset id matching (e.g., csgo / CSGO).
        candidates = [x for x in DATA_DIR.iterdir() if x.is_dir() and x.name.lower() == dataset_id.lower()]
        if candidates:
            path = candidates[0]
    if not path.exists() or not path.is_dir():
        return []
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts]


@app.get("/api/models")
def list_models() -> Dict[str, Any]:
    return {
        "models": [
            {
                "id": "diamond",
                "label": "DIAMOND (CSGO)",
                "status": "available",
                "description": "Conditional diffusion world model with CSGO action space.",
            },
            {
                "id": "gamecraft",
                "label": "Hunyuan-GameCraft 1.0",
                "status": "available",
                "description": "Interactive chunked world model with GameCraft native WASD/camera action adaptation and original accelerated sampling settings.",
            },
            {
                "id": "infinite-world",
                "label": "Infinite-World",
                "status": "available",
                "description": "Interactive world model with image-conditioned long-horizon action rollout.",
            },
            {
                "id": "lingbot-world",
                "label": "LingBot-World (Cam NF4)",
                "status": "available",
                "description": "Quantized LingBot-World camera-control model with native 40-step chunk generation and WASD-to-camera-trajectory adaptation.",
            },
            {
                "id": "vid2world",
                "label": "Vid2World (CSGO)",
                "status": "available",
                "description": "Academic interactive world model for CSGO with native 50-step diffusion rollout.",
            },
            {
                "id": "mineworld",
                "label": "MineWorld",
                "status": "available",
                "description": "Real-time interactive Minecraft world model with WASD and camera control.",
            },
            {
                "id": "wham",
                "label": "WHAM (Bleeding Edge)",
                "status": "available",
                "description": "World and Human Action Model with gamepad-like 16D action control.",
            },
            {
                "id": "yume",
                "label": "YUME 1.5",
                "status": "available",
                "description": "Interactive first-person world model with discrete WASD and camera-motion chunk generation.",
            },
            {
                "id": "worldplay",
                "label": "HY-WorldPlay (WAN 5B)",
                "status": "available",
                "description": "Streaming first-person world model with native WorldPlay chunk memory and WASD/camera control.",
            },
            {
                "id": "worldfm",
                "label": "WorldFM",
                "status": "available",
                "description": "Interactive multi-view frame generation (start runs step1-3, step runs action-driven frame update).",
            },
            {
                "id": "open-oasis",
                "label": "Open-Oasis 500M",
                "status": "available",
                "description": "Action-conditional diffusion world model with WASD/camera control (DDIM=10 by default).",
            },
            {
                "id": "matrixgame",
                "label": "Matrix-Game 2.0",
                "status": "available",
                "description": "Streaming interactive world model with WASD and camera control.",
            },
            {
                "id": "matrixgame3",
                "label": "Matrix-Game 3.0",
                "status": "available",
                "description": "Chunked interactive world model with native Matrix-Game-3.0 action rollout and unified WASD/camera adaptation.",
            },
            {
                "id": "wonderworld",
                "label": "WonderWorld",
                "status": "available",
                "description": "Interactive 3D scene generation from a single image with real-time viewpoint control.",
            },
        ]
    }


@app.post("/api/models/load")
def load_model(req: LoadModelRequest) -> Dict[str, Any]:
    adapter = _require_adapter(req.model_id)
    try:
        return adapter.load()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/datasets")
def list_datasets() -> Dict[str, Any]:
    datasets = []
    if DATA_DIR.exists():
        for d in sorted([x for x in DATA_DIR.iterdir() if x.is_dir()]):
            datasets.append({"id": d.name, "label": d.name, "num_images": len(_dataset_images(d.name))})

    if not any(x["id"].lower() == "csgo" for x in datasets):
        datasets.insert(0, {"id": "CSGO", "label": "CSGO", "num_images": len(_dataset_images("CSGO"))})
    if not any(x["id"].lower() in {"minecraft", "mineworld"} for x in datasets):
        datasets.append(
            {
                "id": "minecraft",
                "label": "Minecraft (MineWorld Scene)",
                "num_images": 1,
            }
        )

    return {"datasets": datasets}


@app.post("/api/datasets/random-image")
def random_dataset_image(req: RandomDatasetImageRequest) -> Dict[str, Any]:
    images = _dataset_images(req.dataset_id)
    if not images:
        # Ask DIAMOND service for spawn fallback (CSGO), if available.
        if req.dataset_id.lower() == "csgo":
            fallback_model_ids = []
            if state.active_model_id in {"vid2world", "diamond"}:
                fallback_model_ids.append(state.active_model_id)
            fallback_model_ids.extend([model_id for model_id in ("vid2world", "diamond") if model_id not in fallback_model_ids])
            for model_id in fallback_model_ids:
                adapter = state.adapters.get(model_id)
                if adapter is None or not hasattr(adapter, "random_dataset_image"):
                    continue
                try:
                    return adapter.random_dataset_image(req.dataset_id)
                except Exception:
                    continue
            raise HTTPException(status_code=404, detail=f"Dataset '{req.dataset_id}' has no images and service fallback failed")
        if req.dataset_id.lower() in ("minecraft", "mineworld") and "mineworld" in state.adapters:
            try:
                return state.adapters["mineworld"].random_dataset_image(req.dataset_id)
            except Exception as exc:
                raise HTTPException(status_code=404, detail=f"Dataset '{req.dataset_id}' has no images and service fallback failed: {exc}") from exc
        raise HTTPException(status_code=404, detail=f"Dataset '{req.dataset_id}' has no images")

    path = random.choice(images)
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    suffix = path.suffix.lower().replace(".", "") or "png"
    return {
        "dataset_id": req.dataset_id,
        "file": str(path.relative_to(ROOT.parent)),
        "image_base64": f"data:image/{suffix};base64,{data}",
    }


@app.post("/api/sessions/start")
def start_session(req: StartSessionRequest) -> Dict[str, Any]:
    adapter = _require_adapter(req.model_id)
    image_bytes = _decode_image(req.init_image_base64)
    try:
        result = adapter.start_session(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    state.active_model_id = req.model_id
    return result


@app.post("/api/sessions/step")
def step_session(req: StepRequest) -> Dict[str, Any]:
    if state.active_model_id is None:
        raise HTTPException(status_code=400, detail="No active model session")
    # Single session per model adapter in this first version.
    adapter = state.adapters[state.active_model_id]
    runtime_session_id = getattr(adapter.runtime, "session_id", None) if adapter.runtime is not None else None
    if runtime_session_id != req.session_id:
        raise HTTPException(status_code=400, detail="Unknown or expired session_id")

    try:
        step_result = adapter.step(req.action)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "session_id": req.session_id,
        "frame_base64": step_result.frame_base64,
        "reward": step_result.reward,
        "ended": step_result.ended,
        "truncated": step_result.truncated,
        "extra": step_result.extra,
    }


@app.post("/api/sessions/reset")
def reset_session(req: ResetRequest) -> Dict[str, Any]:
    if state.active_model_id is None:
        raise HTTPException(status_code=400, detail="No active model session")
    adapter = state.adapters[state.active_model_id]
    runtime_session_id = getattr(adapter.runtime, "session_id", None) if adapter.runtime is not None else None
    if runtime_session_id != req.session_id:
        raise HTTPException(status_code=400, detail="Unknown or expired session_id")

    image_bytes = _decode_image(req.init_image_base64)
    try:
        return adapter.reset_session(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/sessions/progress")
def session_progress(req: ProgressRequest) -> Dict[str, Any]:
    model_id = req.model_id or state.active_model_id
    if model_id is None:
        raise HTTPException(status_code=400, detail="No active model session")

    adapter = _require_adapter(model_id)
    fn = getattr(adapter, "progress", None)
    if fn is None:
        raise HTTPException(status_code=400, detail=f"Model '{model_id}' does not expose progress API")
    try:
        return fn(req.request_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/")
def root() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
