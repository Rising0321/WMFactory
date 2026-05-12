from __future__ import annotations

import base64
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx

from .base import StepResult, WorldModelAdapter
from .runtime_utils import _parse_nvidia_smi, configure_subprocess_cuda, default_service_workdir


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class InfiniteWorldGatewayRuntime:
    session_id: Optional[str] = None
    loaded: bool = False


class InfiniteWorldAdapter(WorldModelAdapter):
    model_id = "infinite-world"

    def __init__(self) -> None:
        configured_url = os.getenv("WM_INFINITEWORLD_URL", "http://127.0.0.1:9011").rstrip("/")
        parsed = urlparse(configured_url)

        self.runtime = InfiniteWorldGatewayRuntime()
        self.base_url = configured_url
        self.timeout = float(os.getenv("WM_INFINITEWORLD_HTTP_TIMEOUT", os.getenv("WM_MODEL_HTTP_TIMEOUT", "1800")))
        self.autostart = os.getenv("WM_INFINITEWORLD_AUTOSTART", "1") == "1"
        self.startup_timeout = float(os.getenv("WM_INFINITEWORLD_STARTUP_TIMEOUT", "300"))

        self.service_host = os.getenv("WM_INFINITEWORLD_HOST", parsed.hostname or "127.0.0.1")
        self.service_port = int(os.getenv("WM_INFINITEWORLD_PORT", str(parsed.port or 9011)))
        self.service_dir = Path(os.getenv("WM_INFINITEWORLD_SERVICE_DIR", str(default_service_workdir("infiniteworld"))))
        self.service_python = os.getenv("WM_INFINITEWORLD_PYTHON", str(ROOT / "venvs" / "Infinite-World" / "bin" / "python"))
        self.service_log = Path(os.getenv("WM_INFINITEWORLD_LOG", str(self.service_dir / "infiniteworld_service.log")))
        self.step_log_every = int(os.getenv("WM_INFINITEWORLD_STEP_LOG_EVERY", "1"))
        self._step_counter = 0

        self._proc: Optional[subprocess.Popen[Any]] = None
        self._log_fp: Optional[Any] = None

    def _log(self, message: str) -> None:
        print(f"[infinite-world] {message}", flush=True)

    def _pick_visible_devices_for_infworld(self) -> Optional[str]:
        override = os.getenv("WM_INFINITEWORLD_CUDA_VISIBLE_DEVICES")
        if override is not None:
            return override
        if os.getenv("WM_AUTO_CUDA_VISIBLE_DEVICES", "1") != "1":
            return None

        try:
            rows = _parse_nvidia_smi()
        except Exception:
            return None
        if not rows:
            return None

        rows = sorted(
            rows,
            key=lambda row: (
                0.0 if row["memory_total"] <= 0 else row["memory_used"] / row["memory_total"],
                row["utilization"],
                row["index"],
            ),
        )
        top = rows[:2]
        if len(top) < 2:
            return str(top[0]["index"])
        return ",".join(str(row["index"]) for row in top)

    def _stream_child_logs(self, proc: subprocess.Popen[Any]) -> None:
        if proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                if line:
                    self._log(line)
        except Exception as exc:
            self._log(f"log stream reader stopped: {exc}")

    def _health(self) -> bool:
        url = f"{self.base_url}/health"
        try:
            with httpx.Client(timeout=2.5) as client:
                resp = client.post(url, json={})
            if resp.status_code >= 400:
                return False
            data = resp.json()
            return bool(data.get("ok", False))
        except Exception:
            return False

    def _ensure_service_up(self) -> None:
        if self._health():
            return

        if not self.autostart:
            raise RuntimeError(
                f"Infinite-World service is not reachable at {self.base_url}. "
                "Start it manually or set WM_INFINITEWORLD_AUTOSTART=1."
            )

        if self._proc is not None and self._proc.poll() is None:
            pass
        else:
            self.service_dir.mkdir(parents=True, exist_ok=True)
            self.service_log.parent.mkdir(parents=True, exist_ok=True)
            self._log_fp = self.service_log.open("a", encoding="utf-8")
            self._log(f"service not ready, launching worker at {self.base_url}")
            cmd = [
                self.service_python,
                "-m",
                "uvicorn",
                "app:app",
                "--host",
                self.service_host,
                "--port",
                str(self.service_port),
            ]
            env = os.environ.copy()
            env["http_proxy"] = ""
            env["https_proxy"] = ""
            env["HTTP_PROXY"] = ""
            env["HTTPS_PROXY"] = ""
            env.setdefault("HF_ENDPOINT", os.getenv("WM_HF_ENDPOINT", "https://hf-mirror.com"))
            selected = self._pick_visible_devices_for_infworld()
            if selected:
                env["CUDA_VISIBLE_DEVICES"] = selected
                visible = [part.strip() for part in selected.split(",") if part.strip()]
                if os.getenv("WM_INFINITEWORLD_GEN_DEVICE") is None:
                    env["WM_INFINITEWORLD_GEN_DEVICE"] = "cuda:0"
                if len(visible) >= 2 and os.getenv("WM_INFINITEWORLD_DECODE_DEVICE") is None:
                    env["WM_INFINITEWORLD_DECODE_DEVICE"] = "cuda:1"
                elif os.getenv("WM_INFINITEWORLD_DECODE_DEVICE") is None:
                    env["WM_INFINITEWORLD_DECODE_DEVICE"] = "cuda:0"
                self._log(f"using CUDA_VISIBLE_DEVICES={selected}, gen={env['WM_INFINITEWORLD_GEN_DEVICE']}, decode={env['WM_INFINITEWORLD_DECODE_DEVICE']}")
            else:
                configure_subprocess_cuda(env, "INFINITEWORLD", self._log)
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
            t = threading.Thread(target=self._stream_child_logs, args=(self._proc,), daemon=True)
            t.start()
            if self._log_fp is not None:
                self._log_fp.write(f"[infinite-world] spawned worker pid={self._proc.pid} cmd={' '.join(cmd)}\n")
                self._log_fp.flush()

        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self._health():
                self._log("service is ready")
                return
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"Infinite-World service exited early with code {self._proc.returncode}. "
                    f"Check log: {self.service_log}"
                )
            time.sleep(0.5)

        raise RuntimeError(
            f"Timed out waiting for Infinite-World service startup at {self.base_url}. "
            f"Check log: {self.service_log}"
        )

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_service_up()
        url = f"{self.base_url}{path}"
        should_log = path != "/sessions/step"
        if path == "/sessions/step":
            self._step_counter += 1
            should_log = (self._step_counter % max(1, self.step_log_every) == 0)
        if should_log:
            self._log(f"request -> {path}")

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=payload)
        except Exception as exc:
            self._ensure_service_up()
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, json=payload)
            except Exception as exc2:
                raise RuntimeError(f"Failed to connect to Infinite-World service {url}: {exc2}") from exc

        if resp.status_code >= 400:
            try:
                body = resp.json()
                detail = body.get("detail", body)
            except Exception:
                detail = resp.text
            self._log(f"request <- {path} failed status={resp.status_code} detail={detail}")
            raise RuntimeError(f"Infinite-World service error {resp.status_code}: {detail}")

        data = resp.json()
        if should_log:
            self._log(f"request <- {path} ok")
        return data

    def _encode_image_bytes(self, data: Optional[bytes]) -> Optional[str]:
        if data is None:
            return None
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def load(self) -> Dict[str, Any]:
        data = self._post("/load", {"model_id": self.model_id})
        self.runtime.loaded = True
        return data

    def start_session(self, init_image_bytes: Optional[bytes]) -> Dict[str, Any]:
        data = self._post(
            "/sessions/start",
            {"init_image_base64": self._encode_image_bytes(init_image_bytes)},
        )
        self.runtime.session_id = data.get("session_id")
        return data

    def reset_session(self, init_image_bytes: Optional[bytes]) -> Dict[str, Any]:
        if self.runtime.session_id is None:
            raise RuntimeError("Session is not started. Call start_session first.")
        return self._post(
            "/sessions/reset",
            {
                "session_id": self.runtime.session_id,
                "init_image_base64": self._encode_image_bytes(init_image_bytes),
            },
        )

    def step(self, action: Dict[str, Any]) -> StepResult:
        if self.runtime.session_id is None:
            raise RuntimeError("Session is not started. Call start_session first.")
        data = self._post(
            "/sessions/step",
            {"session_id": self.runtime.session_id, "action": action},
        )
        return StepResult(
            frame_base64=str(data["frame_base64"]),
            reward=float(data.get("reward", 0.0)),
            ended=bool(data.get("ended", False)),
            truncated=bool(data.get("truncated", False)),
            extra=dict(data.get("extra", {})),
        )
