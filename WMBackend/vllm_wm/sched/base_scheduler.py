from __future__ import annotations

from collections import deque

from vllm_wm.request import WorldModelRequest
from vllm_wm.sched.interface import (
    CachedRequestData,
    NewRequestData,
    RequestState,
    RequestStatus,
    SchedulerInterface,
    SchedulerOutput,
)


class BaseScheduler(SchedulerInterface):
    def __init__(self, max_num_running_reqs: int = 1):
        self._request_states: dict[str, RequestState] = {}
        self._request_id_to_sched_req_id: dict[str, str] = {}
        self._step_id = 0
        self._waiting: deque[str] = deque()
        self._running: list[str] = []
        self._finished_req_ids: set[str] = set()
        self.max_num_running_reqs = max_num_running_reqs

    def add_request(self, request: WorldModelRequest) -> str:
        sched_req_id = self._make_sched_req_id(request)
        state = RequestState(sched_req_id=sched_req_id, req=request)
        self._request_states[sched_req_id] = state
        self._request_id_to_sched_req_id[request.request_id] = sched_req_id
        self._waiting.append(sched_req_id)
        return sched_req_id

    def schedule(self) -> SchedulerOutput:
        scheduled_new_reqs: list[NewRequestData] = []
        scheduled_cached_req_ids: list[str] = []

        for sched_req_id in self._running:
            state = self._request_states.get(sched_req_id)
            if state is not None:
                scheduled_cached_req_ids.append(sched_req_id)

        while self._waiting and len(self._running) < self.max_num_running_reqs:
            sched_req_id = self._waiting[0]
            state = self._request_states.get(sched_req_id)
            if state is None:
                self._waiting.popleft()
                continue
            self._waiting.popleft()
            was_new = state.status == RequestStatus.WAITING
            state.status = RequestStatus.RUNNING
            self._running.append(sched_req_id)
            if was_new:
                scheduled_new_reqs.append(NewRequestData.from_state(state))
            else:
                scheduled_cached_req_ids.append(sched_req_id)

        out = SchedulerOutput(
            step_id=self._step_id,
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_cached_reqs=CachedRequestData(sched_req_ids=scheduled_cached_req_ids),
            finished_req_ids=set(self._finished_req_ids),
            num_running_reqs=len(self._running),
            num_waiting_reqs=len(self._waiting),
        )
        self._step_id += 1
        self._finished_req_ids.clear()
        return out

    def has_requests(self) -> bool:
        return bool(self._waiting or self._running)

    def get_request_state(self, sched_req_id: str) -> RequestState | None:
        return self._request_states.get(sched_req_id)

    def get_sched_req_id(self, request_id: str) -> str | None:
        return self._request_id_to_sched_req_id.get(request_id)

    def pop_request_state(self, sched_req_id: str) -> RequestState | None:
        state = self._request_states.pop(sched_req_id, None)
        if state is not None and self._request_id_to_sched_req_id.get(state.req.request_id) == sched_req_id:
            self._request_id_to_sched_req_id.pop(state.req.request_id, None)
        return state

    def preempt_request(self, sched_req_id: str) -> bool:
        if sched_req_id not in self._request_states:
            return False
        if sched_req_id in self._running:
            self._running.remove(sched_req_id)
            self._waiting.appendleft(sched_req_id)
            self._request_states[sched_req_id].status = RequestStatus.PREEMPTED
            return True
        return False

    def finish_requests(self, sched_req_ids: str | list[str], status: RequestStatus) -> None:
        assert RequestStatus.is_finished(status)
        if isinstance(sched_req_ids, str):
            sched_req_ids = [sched_req_ids]
        statuses = {sched_req_id: status for sched_req_id in sched_req_ids}
        self._finish_requests(statuses)

    def close(self) -> None:
        self._request_states.clear()
        self._request_id_to_sched_req_id.clear()
        self._waiting.clear()
        self._running.clear()
        self._finished_req_ids.clear()

    def _finish_requests(
        self,
        statuses: dict[str, RequestStatus],
        errors: dict[str, str | None] | None = None,
    ) -> set[str]:
        if not statuses:
            return set()
        finished_req_ids: set[str] = set()
        running_to_remove: set[str] = set()
        waiting_to_remove: set[str] = set()
        for sched_req_id, status in statuses.items():
            state = self._request_states.get(sched_req_id)
            if state is None or state.is_finished():
                continue
            finished_req_ids.add(sched_req_id)
            if sched_req_id in self._running:
                running_to_remove.add(sched_req_id)
            if sched_req_id in self._waiting:
                waiting_to_remove.add(sched_req_id)
        if running_to_remove:
            self._running = [req_id for req_id in self._running if req_id not in running_to_remove]
        if waiting_to_remove:
            self._waiting = deque(req_id for req_id in self._waiting if req_id not in waiting_to_remove)
        for sched_req_id in finished_req_ids:
            state = self._request_states[sched_req_id]
            state.status = statuses[sched_req_id]
            state.error = None if errors is None else errors.get(sched_req_id)
        self._finished_req_ids |= finished_req_ids
        return finished_req_ids

    def _finalize_update_from_output(
        self,
        sched_output: SchedulerOutput,
        statuses: dict[str, RequestStatus],
        errors: dict[str, str | None] | None = None,
    ) -> set[str]:
        finished_req_ids = {
            sched_req_id for sched_req_id in sched_output.scheduled_req_ids if sched_req_id in self._finished_req_ids
        }
        finished_req_ids |= self._finish_requests(statuses, errors)
        return finished_req_ids
