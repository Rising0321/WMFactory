from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class WorldModelOutput:
    request_id: str
    model_id: str
    session_id: str
    frame_base64: str
    reward: float = 0.0
    ended: bool = False
    truncated: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
