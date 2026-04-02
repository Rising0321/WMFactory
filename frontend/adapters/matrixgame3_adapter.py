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


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class MatrixGame3GatewayRuntime:
    session_id: Optional[str] = None
    loaded: bool = False


class MatrixGame3Adapter(WorldModelAdapter):
    model_id = "matrixgame3"

    def __init__(self) -> None:
        configured_url = os.getenv("WM_MATRIXGAME3_URL", "http://127.0.0.1:9016").rstrip("/")
        parsed = urlparse(configured_url)

        self.runtime = MatrixGame3GatewayRuntime()
        self.base_url = configured_url
        self.timeout = float(os.getenv("WM_MATRIXGAME3_HTTP_TIMEOUT", os.getenv("WM_MODEL_HTTP_TIMEOUT", "3600")))
        self.autostart = os.getenv("WM_MATRIXGAME3_AUTOSTART", "1") == "1"
        self.startup_timeout = float(os.getenv("WM_MATRIXGAME3_STARTUP_TIMEOUT", "300"))

        self.service_host = os.getenv("WM_MATRIXGAME3_HOST", parsed.hostname or "127.0.0.1")
        self.service_port = int(os.getenv("WM_MATRIXGAME3_PORT", str(parsed.port or 9016)))
        self.service_dir = Path(os.getenv("WM_MATRIXGAME3_SERVICE_DIR", str(ROOT / "services" / "matrixgame3")))
        self.service_python = os.getenv(
            "WM_MATRIXGAME3_SERVICE_PYTHON",
            str(ROOT / "venvs" / "Matrix-Game-3" / "bin" / "python"),
        )
        self.service_log = Path(
            os.getenv("WM_MATRIXGAME3_LOG", str(self.service_dir / "matrixgame3_service.log"))
        )
        self.step_log_every = int(os.getenv("WM_MATRIXGAME3_STEP_LOG_EVERY", "1"))
        self._step_counter = 0

        self._proc: Optional[subprocess.Popen[Any]] = None
        self._log_fp: Optional[Any] = None

    def _log(self, message: str) -> None:
        print(f"[matrixgame3] {message}", flush=True)

    def _tail_service_log(self, lines: int = 80) -> str:
        try:
            if not self.service_log.exists():
                return "service log unavailable"
            content = self.service_log.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(content[-lines:]) if content else "service log empty"
        except Exception as exc:
            return f"failed to read service log: {exc}"

    def _stream_child_logs(self, proc: subprocess.Popen[Any]) -> None:
        if proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                if line:
                    self._log(line)
                    if self._log_fp is not None:
                        self._log_fp.write(f"{line}\n")
                        self._log_fp.flush()
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
                f"Matrix-Game-3.0 service is not reachable at {self.base_url}. "
                "Start it manually or set WM_MATRIXGAME3_AUTOSTART=1."
            )

        if self._proc is None or self._proc.poll() is not None:
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
            env["PYTHONUNBUFFERED"] = "1"
            env.setdefault("HF_ENDPOINT", os.getenv("WM_HF_ENDPOINT", "https://hf-mirror.com"))
            if os.getenv("WM_MATRIXGAME3_CUDA_VISIBLE_DEVICES") is not None:
                env["CUDA_VISIBLE_DEVICES"] = os.getenv("WM_MATRIXGAME3_CUDA_VISIBLE_DEVICES", "")
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
                self._log_fp.write(f"[matrixgame3] spawned worker pid={self._proc.pid} cmd={' '.join(cmd)}\n")
                self._log_fp.flush()

        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self._health():
                self._log("service is ready")
                return
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"Matrix-Game-3.0 service exited early with code {self._proc.returncode}. "
                    f"Check log: {self.service_log}"
                )
            time.sleep(0.5)

        raise RuntimeError(
            f"Timed out waiting for Matrix-Game-3.0 service startup at {self.base_url}. "
            f"Check log: {self.service_log}"
        )

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_service_up()
        url = f"{self.base_url}{path}"
        should_log = path != "/sessions/step"
        if path == "/sessions/step":
            self._step_counter += 1
            should_log = self._step_counter % max(1, self.step_log_every) == 0
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
                raise RuntimeError(f"Failed to connect to Matrix-Game-3.0 service {url}: {exc2}") from exc

        if resp.status_code >= 400:
            try:
                body = resp.json()
                detail = body.get("detail", body)
            except Exception:
                detail = resp.text
            detail = f"{detail}\nRecent matrixgame3 log:\n{self._tail_service_log()}"
            self._log(f"request <- {path} failed status={resp.status_code} detail={detail}")
            raise RuntimeError(f"Matrix-Game-3.0 service error {resp.status_code}: {detail}")

        try:
            data = resp.json()
        except Exception as exc:
            raise RuntimeError(f"Invalid JSON from Matrix-Game-3.0 service {url}: {exc}") from exc
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
        data = self._post(
            "/sessions/reset",
            {
                "session_id": self.runtime.session_id,
                "init_image_base64": self._encode_image_bytes(init_image_bytes),
            },
        )
        self.runtime.session_id = data.get("session_id", self.runtime.session_id)
        return data

    def step(self, action: Dict[str, Any]) -> StepResult:
        if self.runtime.session_id is None:
            raise RuntimeError("Session is not started. Call start_session first.")
        data = self._post(
            "/sessions/step",
            {
                "session_id": self.runtime.session_id,
                "action": action,
            },
        )
        return StepResult(
            frame_base64=str(data["frame_base64"]),
            reward=float(data.get("reward", 0.0)),
            ended=bool(data.get("ended", False)),
            truncated=bool(data.get("truncated", False)),
            extra=dict(data.get("extra", {})),
        )

    def progress(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if request_id is not None:
            payload["request_id"] = request_id
        return self._post("/sessions/progress", payload)
