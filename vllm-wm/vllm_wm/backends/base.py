from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    label: str
    description: str
    family: str
    service_dir_name: str
    env_prefix: str
    default_port: int
    aliases: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    supports_progress: bool = False
    supports_seed_meta: bool = False
    supports_random_dataset: bool = False
    request_timeout: float = 120.0
    startup_timeout: float = 300.0
    load_timeout: float | None = None
    start_timeout: float | None = None
    reset_timeout: float | None = None
    step_timeout: float | None = None
    preferred_visible_devices: int = 1
    use_dual_device_hint: bool = False
    priority: int = 100
    default_dataset_ids: tuple[str, ...] = field(default_factory=tuple)


class BaseWorldModelBackend(ABC):
    def __init__(self, spec: ModelSpec):
        self.spec = spec

    @abstractmethod
    def load(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def start_session(
        self,
        init_image_base64: str | None,
        *,
        seed_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def reset_session(
        self,
        session_id: str,
        init_image_base64: str | None,
        *,
        seed_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def step(self, session_id: str, action: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def health(self) -> dict[str, Any]:
        raise NotImplementedError

    def progress(self, request_id: str | None = None) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "phase": "idle",
            "message": f"{self.spec.model_id} backend does not expose progress",
            "active": False,
            "session_id": None,
            "logs": [],
        }

    def random_dataset_image(self, dataset_id: str) -> dict[str, Any]:
        raise RuntimeError(f"{self.spec.model_id} does not expose dataset image sampling")

    def close(self) -> None:
        return None
