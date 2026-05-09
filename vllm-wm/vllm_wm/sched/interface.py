from __future__ import annotations

import enum
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass

from vllm_wm.request import WorldModelRequest


class RequestStatus(enum.IntEnum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    FINISHED_COMPLETED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_ERROR = enum.auto()

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status >= RequestStatus.FINISHED_COMPLETED


@dataclass(slots=True)
class RequestState:
    sched_req_id: str
    req: WorldModelRequest
    status: RequestStatus = RequestStatus.WAITING
    error: str | None = None

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)


@dataclass(slots=True)
class NewRequestData:
    sched_req_id: str
    req: WorldModelRequest

    @classmethod
    def from_state(cls, state: RequestState) -> "NewRequestData":
        return cls(sched_req_id=state.sched_req_id, req=state.req)


@dataclass(slots=True)
class CachedRequestData:
    sched_req_ids: list[str]

    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        return cls(sched_req_ids=[])


@dataclass(slots=True)
class SchedulerOutput:
    step_id: int
    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: CachedRequestData
    finished_req_ids: set[str]
    num_running_reqs: int
    num_waiting_reqs: int

    @property
    def scheduled_req_ids(self) -> list[str]:
        return [
            *(req.sched_req_id for req in self.scheduled_new_reqs),
            *self.scheduled_cached_reqs.sched_req_ids,
        ]

    @property
    def is_empty(self) -> bool:
        return len(self.scheduled_req_ids) == 0

    @property
    def num_scheduled_reqs(self) -> int:
        return len(self.scheduled_req_ids)


class SchedulerInterface(ABC):
    def _make_sched_req_id(self, request: WorldModelRequest) -> str:
        base = request.request_id or f"req_{uuid.uuid4().hex[:8]}"
        sched_req_id = base
        suffix = 1
        while self.get_request_state(sched_req_id) is not None:
            sched_req_id = f"{base}#{suffix}"
            suffix += 1
        return sched_req_id

    @abstractmethod
    def add_request(self, request: WorldModelRequest) -> str:
        pass

    @abstractmethod
    def schedule(self) -> SchedulerOutput:
        pass

    @abstractmethod
    def update_from_output(self, sched_output: SchedulerOutput, output) -> set[str]:
        pass

    @abstractmethod
    def get_request_state(self, sched_req_id: str) -> RequestState | None:
        pass

    @abstractmethod
    def has_requests(self) -> bool:
        pass

    @abstractmethod
    def get_sched_req_id(self, request_id: str) -> str | None:
        pass

    @abstractmethod
    def pop_request_state(self, sched_req_id: str) -> RequestState | None:
        pass

    @abstractmethod
    def preempt_request(self, sched_req_id: str) -> bool:
        pass

    @abstractmethod
    def finish_requests(self, sched_req_ids: str | list[str], status: RequestStatus) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
