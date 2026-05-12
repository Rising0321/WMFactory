from __future__ import annotations

import atexit
import os
import signal
import subprocess
import threading
import time
import weakref
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from vllm_wm.backends.base import BaseWorldModelBackend, ModelSpec
from vllm_wm.config import EngineConfig


def _parse_nvidia_smi() -> list[dict[str, float]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    rows: list[dict[str, float]] = []
    for raw_line in proc.stdout.strip().splitlines():
        parts = [part.strip().replace("%", "") for part in raw_line.split(",")]
        if len(parts) != 4:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "memory_used": float(parts[1]),
                "memory_total": float(parts[2]),
                "utilization": float(parts[3]),
            }
        )
    return rows


def _select_visible_devices(num_devices: int) -> str | None:
    if os.getenv("WM_AUTO_CUDA_VISIBLE_DEVICES", "1") != "1":
        return None

    max_mem_fraction = float(os.getenv("WM_AUTO_GPU_MAX_MEMORY_FRACTION", "0.5"))
    max_util_fraction = float(os.getenv("WM_AUTO_GPU_MAX_UTILIZATION_FRACTION", "0.5"))

    try:
        rows = _parse_nvidia_smi()
    except Exception:
        return None
    if not rows:
        return None

    ranked = sorted(
        rows,
        key=lambda row: (
            0.0 if row["memory_total"] <= 0 else row["memory_used"] / row["memory_total"],
            row["utilization"],
            row["index"],
        ),
    )
    eligible = []
    for row in ranked:
        mem_fraction = 0.0 if row["memory_total"] <= 0 else row["memory_used"] / row["memory_total"]
        util_fraction = row["utilization"] / 100.0
        if mem_fraction < max_mem_fraction and util_fraction < max_util_fraction:
            eligible.append(row)

    picked = eligible[:num_devices] if len(eligible) >= num_devices else ranked[:num_devices]
    if not picked:
        return None
    return ",".join(str(int(row["index"])) for row in picked)


class ManagedServiceBackend(BaseWorldModelBackend):
    _instances: weakref.WeakSet["ManagedServiceBackend"] = weakref.WeakSet()
    _atexit_registered = False

    def __init__(self, spec: ModelSpec, config: EngineConfig):
        super().__init__(spec)
        self.config = config
        self.base_url = self._resolve_base_url()
        parsed = urlparse(self.base_url)

        self.service_host = os.getenv(self._key("HOST"), parsed.hostname or "127.0.0.1")
        self.service_port = int(os.getenv(self._key("PORT"), str(parsed.port or spec.default_port)))
        self.autostart = os.getenv(self._key("AUTOSTART"), "1") == "1"
        self.startup_timeout = float(os.getenv(self._key("STARTUP_TIMEOUT"), str(spec.startup_timeout)))
        self.request_timeout = float(
            os.getenv(
                self._key("HTTP_TIMEOUT"),
                os.getenv("WM_MODEL_HTTP_TIMEOUT", str(spec.request_timeout)),
            )
        )
        self.load_timeout = self._resolve_timeout("LOAD_TIMEOUT", spec.load_timeout)
        self.start_timeout = self._resolve_timeout("START_TIMEOUT", spec.start_timeout)
        self.reset_timeout = self._resolve_timeout("RESET_TIMEOUT", spec.reset_timeout or spec.start_timeout)
        self.step_timeout = self._resolve_timeout("STEP_TIMEOUT", spec.step_timeout)
        self.progress_timeout = self._resolve_timeout("PROGRESS_TIMEOUT", None)
        self.dataset_timeout = self._resolve_timeout("DATASET_TIMEOUT", None)
        self.step_log_every = int(os.getenv(self._key("STEP_LOG_EVERY"), "20"))

        default_service_dir = self.config.project_root / "services" / spec.service_dir_name
        self.service_dir = Path(os.getenv(self._key("SERVICE_DIR"), str(default_service_dir))).resolve()
        self.service_log = Path(
            os.getenv(self._key("LOG"), str(self.service_dir / f"{spec.service_dir_name}_service.log"))
        ).resolve()

        default_python = self.config.resolved_unified_python()
        self.service_python = Path(
            os.getenv(self._key("SERVICE_PYTHON"), os.getenv("WM_VLLM_WM_PYTHON", str(default_python)))
        ).resolve()

        self._proc: subprocess.Popen[str] | None = None
        self._log_fp: Any | None = None
        self._step_counter = 0
        self._lock = threading.Lock()

        self._instances.add(self)
        if not self.__class__._atexit_registered:
            atexit.register(self.__class__._shutdown_all)
            self.__class__._atexit_registered = True

    @classmethod
    def _shutdown_all(cls) -> None:
        for backend in list(cls._instances):
            try:
                backend.close()
            except Exception:
                pass

    def _key(self, suffix: str) -> str:
        return f"WM_{self.spec.env_prefix}_{suffix}"

    def _log(self, message: str) -> None:
        print(f"[WMBackend][{self.spec.model_id}] {message}", flush=True)

    def _resolve_timeout(self, suffix: str, spec_value: float | None) -> float:
        default_value = self.request_timeout if spec_value is None else spec_value
        return float(os.getenv(self._key(suffix), str(default_value)))

    def _resolve_base_url(self) -> str:
        configured = os.getenv(self._key("URL"))
        if configured:
            return configured.rstrip("/")
        host = os.getenv(self._key("HOST"), "127.0.0.1")
        port_env = os.getenv(self._key("PORT"))
        if port_env:
            return f"http://{host}:{int(port_env)}"
        return f"http://{host}:{self.spec.default_port}"

    def _tail_service_log(self, lines: int = 80) -> str:
        try:
            if not self.service_log.exists():
                return "service log unavailable"
            content = self.service_log.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(content[-lines:]) if content else "service log empty"
        except Exception as exc:
            return f"failed to read service log: {exc}"

    def _stream_child_logs(self, proc: subprocess.Popen[str]) -> None:
        if proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                if not line:
                    continue
                self._log(line)
                if self._log_fp is not None:
                    self._log_fp.write(f"{line}\n")
                    self._log_fp.flush()
        except Exception as exc:
            self._log(f"log stream reader stopped: {exc}")

    def _healthcheck_live(self) -> bool:
        try:
            with httpx.Client(timeout=2.5) as client:
                resp = client.post(f"{self.base_url}/health", json={})
            if resp.status_code >= 400:
                return False
            return bool(resp.json().get("ok", False))
        except Exception:
            return False

    def _configure_cuda_env(self, env: dict[str, str]) -> None:
        override = os.getenv(self._key("CUDA_VISIBLE_DEVICES"))
        if override is not None:
            env["CUDA_VISIBLE_DEVICES"] = override
            env[self._key("CUDA_VISIBLE_DEVICES")] = override
            self._log(f"using {self._key('CUDA_VISIBLE_DEVICES')}={override}")
            return

        selected = _select_visible_devices(max(1, self.spec.preferred_visible_devices))
        if not selected:
            return

        env["CUDA_VISIBLE_DEVICES"] = selected
        env[self._key("CUDA_VISIBLE_DEVICES")] = selected
        self._log(f"auto-selected CUDA_VISIBLE_DEVICES={selected}")

        if not self.spec.use_dual_device_hint:
            return

        visible = [part.strip() for part in selected.split(",") if part.strip()]
        if os.getenv(self._key("GEN_DEVICE")) is None:
            env[self._key("GEN_DEVICE")] = "cuda:0"
        if os.getenv(self._key("DECODE_DEVICE")) is None:
            env[self._key("DECODE_DEVICE")] = "cuda:1" if len(visible) >= 2 else "cuda:0"
        self._log(
            f"using {self._key('GEN_DEVICE')}={env.get(self._key('GEN_DEVICE'))}, "
            f"{self._key('DECODE_DEVICE')}={env.get(self._key('DECODE_DEVICE'))}"
        )

    def _build_child_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["http_proxy"] = ""
        env["https_proxy"] = ""
        env["HTTP_PROXY"] = ""
        env["HTTPS_PROXY"] = ""
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        env.setdefault("HF_ENDPOINT", os.getenv("WM_HF_ENDPOINT", "https://hf-mirror.com"))
        env.setdefault(self._key("PYTHON"), str(self.service_python))
        env.setdefault(self._key("SERVICE_PYTHON"), str(self.service_python))
        self._configure_cuda_env(env)
        return env

    def _ensure_service_up(self) -> None:
        if self._healthcheck_live():
            return

        if not self.autostart:
            raise RuntimeError(
                f"{self.spec.label} service is not reachable at {self.base_url}. "
                f"Start it manually or set {self._key('AUTOSTART')}=1."
            )

        with self._lock:
            if self._healthcheck_live():
                return

            if self._proc is None or self._proc.poll() is not None:
                self.service_dir.mkdir(parents=True, exist_ok=True)
                self.service_log.parent.mkdir(parents=True, exist_ok=True)
                self._log_fp = self.service_log.open("a", encoding="utf-8")
                cmd = [
                    str(self.service_python),
                    "-m",
                    "uvicorn",
                    "app:app",
                    "--host",
                    self.service_host,
                    "--port",
                    str(self.service_port),
                ]
                env = self._build_child_env()
                self._log(f"service not ready, launching worker at {self.base_url}")
                self._proc = subprocess.Popen(
                    cmd,
                    cwd=str(self.service_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                    env=env,
                )
                threading.Thread(target=self._stream_child_logs, args=(self._proc,), daemon=True).start()
                if self._log_fp is not None:
                    self._log_fp.write(
                        f"[WMBackend][{self.spec.model_id}] spawned worker pid={self._proc.pid} cmd={' '.join(cmd)}\n"
                    )
                    self._log_fp.flush()

        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self._healthcheck_live():
                self._log("service is ready")
                return
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"{self.spec.label} service exited early with code {self._proc.returncode}. "
                    f"Check log: {self.service_log}"
                )
            time.sleep(0.5)

        raise RuntimeError(
            f"Timed out waiting for {self.spec.label} service startup at {self.base_url}. "
            f"Check log: {self.service_log}"
        )

    def _with_meta(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = dict(payload)
        result.setdefault("model_id", self.spec.model_id)
        result.setdefault("label", self.spec.label)
        result.setdefault("family", self.spec.family)
        result.setdefault("worker_url", self.base_url)
        result.setdefault("worker_log", str(self.service_log))
        return result

    def _timeout_for_path(self, path: str) -> float:
        mapping = {
            "/load": self.load_timeout,
            "/sessions/start": self.start_timeout,
            "/sessions/reset": self.reset_timeout,
            "/sessions/step": self.step_timeout,
            "/sessions/progress": self.progress_timeout,
            "/datasets/random-image": self.dataset_timeout,
        }
        return mapping.get(path, self.request_timeout)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._ensure_service_up()
        url = f"{self.base_url}{path}"
        timeout = self._timeout_for_path(path)
        should_log = path != "/sessions/step"
        if path == "/sessions/step":
            self._step_counter += 1
            should_log = self._step_counter % max(1, self.step_log_every) == 0
        if should_log:
            self._log(f"request -> {path}")

        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, json=payload)
        except Exception as exc:
            if path in {"/load", "/sessions/start", "/sessions/reset"}:
                raise RuntimeError(f"Failed to connect to service {url}: {exc}") from exc
            self._ensure_service_up()
            try:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(url, json=payload)
            except Exception as exc2:
                raise RuntimeError(f"Failed to connect to service {url}: {exc2}") from exc

        if resp.status_code >= 400:
            try:
                body = resp.json()
                detail = body.get("detail", body)
            except Exception:
                detail = resp.text
            detail = f"{detail}\nRecent {self.spec.model_id} log:\n{self._tail_service_log()}"
            self._log(f"request <- {path} failed status={resp.status_code}")
            raise RuntimeError(f"{self.spec.label} service error {resp.status_code}: {detail}")

        try:
            data = resp.json()
        except Exception as exc:
            raise RuntimeError(f"Invalid JSON from service {url}: {exc}") from exc
        if should_log:
            self._log(f"request <- {path} ok")
        return data

    def load(self) -> dict[str, Any]:
        return self._with_meta(self._post("/load", {"model_id": self.spec.model_id}))

    def start_session(
        self,
        init_image_base64: str | None,
        *,
        seed_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"init_image_base64": init_image_base64}
        if self.spec.supports_seed_meta and seed_meta is not None:
            payload["seed_meta"] = seed_meta
        return self._with_meta(self._post("/sessions/start", payload))

    def reset_session(
        self,
        session_id: str,
        init_image_base64: str | None,
        *,
        seed_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "session_id": session_id,
            "init_image_base64": init_image_base64,
        }
        if self.spec.supports_seed_meta and seed_meta is not None:
            payload["seed_meta"] = seed_meta
        return self._with_meta(self._post("/sessions/reset", payload))

    def step(self, session_id: str, action: dict[str, Any]) -> dict[str, Any]:
        return self._with_meta(self._post("/sessions/step", {"session_id": session_id, "action": action}))

    def health(self) -> dict[str, Any]:
        return self._with_meta(self._post("/health", {}))

    def progress(self, request_id: str | None = None) -> dict[str, Any]:
        if not self.spec.supports_progress:
            return super().progress(request_id=request_id)
        return self._with_meta(self._post("/sessions/progress", {"request_id": request_id}))

    def random_dataset_image(self, dataset_id: str) -> dict[str, Any]:
        if not self.spec.supports_random_dataset:
            return super().random_dataset_image(dataset_id)
        return self._with_meta(self._post("/datasets/random-image", {"dataset_id": dataset_id}))

    def close(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
        self._proc = None
        if self._log_fp is not None:
            try:
                self._log_fp.close()
            except Exception:
                pass
        self._log_fp = None
