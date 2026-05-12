from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from vllm_wm.config import EngineConfig
from vllm_wm.engine.omni_engine import WorldModelEngine


class LoadModelRequest(BaseModel):
    model_id: str


class StartSessionRequest(BaseModel):
    model_id: str
    init_image_base64: str | None = None
    seed_meta: dict[str, Any] | None = None


class StepRequest(BaseModel):
    session_id: str
    action: dict[str, Any] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    session_id: str
    init_image_base64: str | None = None
    seed_meta: dict[str, Any] | None = None


class ProgressRequest(BaseModel):
    session_id: str | None = None
    model_id: str | None = None
    request_id: str | None = None


class RandomDatasetImageRequest(BaseModel):
    dataset_id: str
    model_id: str | None = None


engine = WorldModelEngine(EngineConfig())
app = FastAPI(title="WMBackend Unified World Model Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "WMBackend",
        "ok": True,
        "endpoints": [
            "/models",
            "/models/load",
            "/sessions/start",
            "/sessions/step",
            "/sessions/reset",
            "/sessions/progress",
            "/datasets/random-image",
        ],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return engine.health()


@app.get("/models")
@app.get("/api/models")
def list_models() -> dict[str, Any]:
    return {"models": engine.list_models()}


@app.post("/models/load")
@app.post("/api/models/load")
def load_model(req: LoadModelRequest) -> dict[str, Any]:
    try:
        return engine.load_model(req.model_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/sessions/start")
@app.post("/api/sessions/start")
def start_session(req: StartSessionRequest) -> dict[str, Any]:
    try:
        return engine.start_session(req.model_id, req.init_image_base64, seed_meta=req.seed_meta)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/sessions/step")
@app.post("/api/sessions/step")
def step(req: StepRequest) -> dict[str, Any]:
    try:
        return engine.step(req.session_id, req.action)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/sessions/reset")
@app.post("/api/sessions/reset")
def reset(req: ResetRequest) -> dict[str, Any]:
    try:
        return engine.reset_session(req.session_id, req.init_image_base64, seed_meta=req.seed_meta)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/sessions/progress")
@app.post("/api/sessions/progress")
def progress(req: ProgressRequest) -> dict[str, Any]:
    try:
        return engine.progress(session_id=req.session_id, model_id=req.model_id, request_id=req.request_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/datasets/random-image")
@app.post("/api/datasets/random-image")
def random_dataset_image(req: RandomDatasetImageRequest) -> dict[str, Any]:
    try:
        return engine.random_dataset_image(req.dataset_id, model_id=req.model_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
