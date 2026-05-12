from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Callable, List, Optional


def default_service_workdir(service_dir_name: str) -> Path:
    """
    Resolve the directory that contains the per-model FastAPI ``app.py``.

    In this monorepo, service code lives under ``WMFactory/WMBackend/services/<name>/``.
    Older layouts used ``WMFactory/services/<name>/``; we still honor that if it has ``app.py``.
    """
    repo_root = Path(__file__).resolve().parents[2]
    wb = repo_root / "WMBackend" / "services" / service_dir_name
    legacy = repo_root / "services" / service_dir_name
    if (wb / "app.py").is_file():
        return wb
    if (legacy / "app.py").is_file():
        return legacy
    return wb


def _parse_nvidia_smi() -> List[dict]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    rows = []
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


def select_visible_devices() -> Optional[str]:
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

    eligible = []
    for row in rows:
        mem_fraction = 0.0 if row["memory_total"] <= 0 else row["memory_used"] / row["memory_total"]
        util_fraction = row["utilization"] / 100.0
        if mem_fraction < max_mem_fraction and util_fraction < max_util_fraction:
            eligible.append(row)

    if eligible:
        return ",".join(str(row["index"]) for row in eligible)

    best = min(
        rows,
        key=lambda row: (
            0.0 if row["memory_total"] <= 0 else row["memory_used"] / row["memory_total"],
            row["utilization"],
            row["index"],
        ),
    )
    return str(best["index"])


def configure_subprocess_cuda(
    env: dict,
    model_env_prefix: str,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    override_key = f"WM_{model_env_prefix}_CUDA_VISIBLE_DEVICES"
    override = os.getenv(override_key)
    if override is not None:
        env["CUDA_VISIBLE_DEVICES"] = override
        if log_fn is not None:
            log_fn(f"using {override_key}={override}")
        return

    selected = select_visible_devices()
    if not selected:
        return

    env["CUDA_VISIBLE_DEVICES"] = selected
    if log_fn is not None:
        log_fn(f"auto-selected CUDA_VISIBLE_DEVICES={selected}")
