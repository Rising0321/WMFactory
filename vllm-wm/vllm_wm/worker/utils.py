from __future__ import annotations

from dataclasses import dataclass

from vllm_wm.outputs import WorldModelOutput


@dataclass(slots=True)
class RunnerOutput:
    req_id: str
    finished: bool = False
    result: WorldModelOutput | None = None
    error: str | None = None
