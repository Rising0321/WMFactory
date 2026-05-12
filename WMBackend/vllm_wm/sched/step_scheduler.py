from vllm_wm.sched.request_scheduler import RequestScheduler


class StepScheduler(RequestScheduler):
    """Reserved for future per-denoise-step execution backends.

    The current WMBackend integration path wraps each world-model ``step()``
    call as one atomic scheduled request. Diffusion backends that expose
    finer-grained denoise scheduling can later specialize this scheduler
    without changing the API surface.
    """
