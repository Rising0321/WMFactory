from __future__ import annotations

from typing import Any

from vllm_wm.backends.base import BaseWorldModelBackend
from vllm_wm.config import EngineConfig
from vllm_wm.outputs import WorldModelOutput
from vllm_wm.registry import build_backend, normalize_model_id
from vllm_wm.sched.interface import SchedulerOutput
from vllm_wm.worker.utils import RunnerOutput


class ModelRunner:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.backend_cache: dict[str, BaseWorldModelBackend] = {}

    def get_backend(self, model_id: str) -> BaseWorldModelBackend:
        normalized = normalize_model_id(model_id)
        backend = self.backend_cache.get(normalized)
        if backend is None:
            backend = build_backend(normalized, self.config)
            self.backend_cache[normalized] = backend
        return backend

    def load_model(self, model_id: str) -> dict[str, Any]:
        return self.get_backend(model_id).load()

    def start_session(
        self,
        model_id: str,
        init_image_base64: str | None,
        *,
        seed_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        backend = self.get_backend(model_id)
        if self.config.auto_load_on_session_start:
            backend.load()
        return backend.start_session(init_image_base64, seed_meta=seed_meta)

    def reset_session(
        self,
        model_id: str,
        session_id: str,
        init_image_base64: str | None,
        *,
        seed_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.get_backend(model_id).reset_session(session_id, init_image_base64, seed_meta=seed_meta)

    def health(self, model_id: str) -> dict[str, Any]:
        return self.get_backend(model_id).health()

    def progress(self, model_id: str, request_id: str | None = None) -> dict[str, Any]:
        return self.get_backend(model_id).progress(request_id=request_id)

    def random_dataset_image(self, model_id: str, dataset_id: str) -> dict[str, Any]:
        return self.get_backend(model_id).random_dataset_image(dataset_id)

    def execute_request(self, scheduler_output: SchedulerOutput) -> RunnerOutput:
        if scheduler_output.num_scheduled_reqs != 1:
            raise ValueError(
                "vllm-wm currently schedules exactly one active world-model request at a time, "
                f"got {scheduler_output.num_scheduled_reqs}."
            )

        request_data = (
            scheduler_output.scheduled_new_reqs[0]
            if scheduler_output.scheduled_new_reqs
            else None
        )
        if request_data is None:
            sched_req_id = scheduler_output.scheduled_cached_reqs.sched_req_ids[0]
            raise RuntimeError(f"Atomic backend runner received cached request '{sched_req_id}', which is unsupported")

        request = request_data.req
        sched_req_id = request_data.sched_req_id
        try:
            backend = self.get_backend(request.model_id)
            raw = backend.step(request.session_id, request.action)
            result = WorldModelOutput(
                request_id=request.request_id,
                model_id=normalize_model_id(request.model_id),
                session_id=str(raw["session_id"]),
                frame_base64=str(raw["frame_base64"]),
                reward=float(raw.get("reward", 0.0) or 0.0),
                ended=bool(raw.get("ended", False)),
                truncated=bool(raw.get("truncated", False)),
                extra=dict(raw.get("extra", {})),
                raw=dict(raw),
            )
            return RunnerOutput(req_id=sched_req_id, finished=True, result=result)
        except Exception as exc:
            return RunnerOutput(req_id=sched_req_id, finished=True, result=None, error=str(exc))
