from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


def _detect_project_root() -> Path:
    package_root = Path(__file__).resolve().parent
    candidates = [package_root.parent, package_root]
    for candidate in candidates:
        if (candidate / "services").exists() and (candidate / "vendors").exists():
            return candidate
    return package_root.parent


PROJECT_ROOT = _detect_project_root()
REPO_ROOT = PROJECT_ROOT


def ensure_repo_root_on_path() -> None:
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


@dataclass(slots=True)
class EngineConfig:
    project_root: Path = PROJECT_ROOT
    repo_root: Path = PROJECT_ROOT
    unified_python: Path = PROJECT_ROOT / ".venv" / "bin" / "python"
    max_num_running_reqs: int = 1
    lazy_load: bool = True
    auto_load_on_session_start: bool = True
    host: str = "0.0.0.0"
    port: int = 9100

    def resolved_unified_python(self) -> Path:
        legacy_env_name = "vllm" + "-wm"
        candidates = [
            self.unified_python,
            self.project_root / ".venv" / "Scripts" / "python.exe",
            self.project_root.parent / "venvs" / "WMBackend" / "bin" / "python",
            self.project_root.parent / "venvs" / "WMBackend" / "Scripts" / "python.exe",
            self.project_root / "venvs" / "WMBackend" / "bin" / "python",
            self.project_root / "venvs" / "WMBackend" / "Scripts" / "python.exe",
            # Fall back to the legacy env directory name so existing installs keep working.
            self.project_root.parent / "venvs" / legacy_env_name / "bin" / "python",
            self.project_root.parent / "venvs" / legacy_env_name / "Scripts" / "python.exe",
            self.project_root / "venvs" / legacy_env_name / "bin" / "python",
            self.project_root / "venvs" / legacy_env_name / "Scripts" / "python.exe",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return Path(sys.executable)
