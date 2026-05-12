from vllm_wm.sched.base_scheduler import BaseScheduler
from vllm_wm.sched.interface import RequestStatus, SchedulerOutput


class RequestScheduler(BaseScheduler):
    def update_from_output(self, sched_output: SchedulerOutput, output) -> set[str]:
        scheduled_req_ids = sched_output.scheduled_req_ids
        if not scheduled_req_ids:
            return set()
        if not output.finished and output.error is None:
            return self._finalize_update_from_output(sched_output, {})
        terminal_statuses: dict[str, RequestStatus] = {}
        terminal_errors: dict[str, str | None] = {}
        for sched_req_id in scheduled_req_ids:
            state = self._request_states.get(sched_req_id)
            if state is None or state.is_finished():
                continue
            if output.result is None:
                terminal_statuses[sched_req_id] = RequestStatus.FINISHED_ERROR
                terminal_errors[sched_req_id] = output.error or "No output result"
            elif output.error:
                terminal_statuses[sched_req_id] = RequestStatus.FINISHED_ERROR
                terminal_errors[sched_req_id] = output.error
            else:
                terminal_statuses[sched_req_id] = RequestStatus.FINISHED_COMPLETED
                terminal_errors[sched_req_id] = None
        return self._finalize_update_from_output(sched_output, terminal_statuses, terminal_errors)
