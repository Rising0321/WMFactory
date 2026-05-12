from __future__ import annotations

from typing import Any

from vllm_wm.config import EngineConfig
from vllm_wm.engine.model_runner import ModelRunner
from vllm_wm.outputs import WorldModelOutput
from vllm_wm.registry import list_model_cards, normalize_model_id
from vllm_wm.request import WorldModelRequest
from vllm_wm.sched.request_scheduler import RequestScheduler


class WorldModelEngine:
    def __init__(self, config: EngineConfig | None = None):
        self.config = config or EngineConfig()
        self.scheduler = RequestScheduler(max_num_running_reqs=self.config.max_num_running_reqs)
        self.runner = ModelRunner(self.config)
        self.session_to_model: dict[str, str] = {}

    def list_models(self) -> list[dict[str, Any]]:
        return list_model_cards()

    def load_model(self, model_id: str) -> dict[str, Any]:
        return self.runner.load_model(model_id)

    def start_session(
        self,
        model_id: str,
        init_image_base64: str | None,
        *,
        seed_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized = normalize_model_id(model_id)
        payload = self.runner.start_session(normalized, init_image_base64, seed_meta=seed_meta)
        session_id = str(payload["session_id"])
        self.session_to_model[session_id] = normalized
        return payload

    def reset_session(
        self,
        session_id: str,
        init_image_base64: str | None,
        *,
        seed_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        model_id = self._require_model_id(session_id)
        payload = self.runner.reset_session(model_id, session_id, init_image_base64, seed_meta=seed_meta)
        new_session_id = str(payload["session_id"])
        if new_session_id != session_id:
            self.session_to_model.pop(session_id, None)
        self.session_to_model[new_session_id] = model_id
        return payload

    def step(self, session_id: str, action: dict[str, Any]) -> dict[str, Any]:
        model_id = self._require_model_id(session_id)
        request = WorldModelRequest(model_id=model_id, session_id=session_id, action=dict(action))
        self.scheduler.add_request(request)

        result: WorldModelOutput | None = None
        while self.scheduler.has_requests():
            sched_output = self.scheduler.schedule()
            if sched_output.is_empty:
                break
            runner_output = self.runner.execute_request(sched_output)
            finished_req_ids = self.scheduler.update_from_output(sched_output, runner_output)
            for finished_req_id in finished_req_ids:
                self.scheduler.pop_request_state(finished_req_id)
            if runner_output.error:
                raise RuntimeError(runner_output.error)
            if runner_output.result is not None:
                result = runner_output.result
                break

        if result is None:
            raise RuntimeError("Request finished without a world-model result")
        current_session_id = result.raw.get("session_id", session_id)
        if current_session_id != session_id:
            self.session_to_model.pop(session_id, None)
            self.session_to_model[str(current_session_id)] = model_id
        return result.raw

    def progress(
        self,
        *,
        session_id: str | None = None,
        model_id: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        resolved = normalize_model_id(model_id) if model_id else self._require_model_id(session_id or "")
        return self.runner.progress(resolved, request_id=request_id)

    def random_dataset_image(
        self,
        dataset_id: str,
        *,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        if model_id:
            return self.runner.random_dataset_image(model_id, dataset_id)

        fallback_order = []
        if dataset_id.lower() == "csgo":
            fallback_order = ["vid2world", "diamond"]
        for candidate in fallback_order:
            try:
                return self.runner.random_dataset_image(candidate, dataset_id)
            except Exception:
                continue
        raise RuntimeError(f"No backend in WMBackend can sample dataset '{dataset_id}'")

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "engine": "WMBackend",
            "num_registered_models": len(self.list_models()),
            "num_active_sessions": len(self.session_to_model),
            "active_sessions": dict(self.session_to_model),
        }

    def _require_model_id(self, session_id: str) -> str:
        model_id = self.session_to_model.get(session_id)
        if model_id is None:
            raise RuntimeError(f"Unknown or expired session_id '{session_id}'")
        return model_id
