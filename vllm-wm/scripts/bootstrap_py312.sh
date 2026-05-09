#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_DIR="$ROOT_DIR/vllm-wm"
ENV_DIR="${ENV_DIR:-$ROOT_DIR/venvs/vllm-wm}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.12}"
CONDA_BIN="${CONDA_BIN:-/home/anaconda3/condabin/conda}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$ROOT_DIR/venvs/.conda_pkgs}"
PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
FLASH_ATTN_WHL="${FLASH_ATTN_WHL:-$ROOT_DIR/.cache/flashattn/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl}"

export http_proxy=
export https_proxy=
export HTTP_PROXY=
export HTTPS_PROXY=
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONNOUSERSITE=1

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "python3.12 not found at $PYTHON_BIN" >&2
  exit 1
fi

create_env_with_conda() {
  mkdir -p "$CONDA_PKGS_DIRS"
  export CONDA_PKGS_DIRS
  http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= \
    "$CONDA_BIN" create -p "$ENV_DIR" python=3.12 -y
}

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  if [[ -x "$CONDA_BIN" ]]; then
    create_env_with_conda
  else
    "$PYTHON_BIN" -m venv "$ENV_DIR"
  fi
fi

ENV_PYTHON="$ENV_DIR/bin/python"

"$ENV_PYTHON" -m pip install --upgrade pip setuptools wheel \
  -i "$PIP_MIRROR" \
  --trusted-host pypi.tuna.tsinghua.edu.cn

"$ENV_PYTHON" -m pip install -r "$PROJECT_DIR/requirements/py312-cu12.txt" \
  -i "$PIP_MIRROR" \
  --trusted-host pypi.tuna.tsinghua.edu.cn

if [[ -f "$FLASH_ATTN_WHL" ]]; then
  "$ENV_PYTHON" -m pip install "$FLASH_ATTN_WHL"
else
  echo "Local flash-attn wheel not found: $FLASH_ATTN_WHL" >&2
  echo "Attempting PyPI install before any manual GitHub fallback..." >&2
  "$ENV_PYTHON" -m pip install flash-attn==2.8.3 --no-build-isolation \
    -i "$PIP_MIRROR" \
    --trusted-host pypi.tuna.tsinghua.edu.cn
fi

"$ENV_PYTHON" -m pip install -e "$PROJECT_DIR" \
  -i "$PIP_MIRROR" \
  --trusted-host pypi.tuna.tsinghua.edu.cn

echo "vllm-wm environment ready at $ENV_DIR"
echo "Try: PYTHONNOUSERSITE=1 \"$ENV_DIR/bin/python\" -m vllm_wm.cli list-models"
