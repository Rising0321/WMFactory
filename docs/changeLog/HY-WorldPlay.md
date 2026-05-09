# HY-WorldPlay

- Origin: `https://github.com/Tencent-Hunyuan/HY-WorldPlay`
- Local branch: `main`
- Local HEAD: `a6a642130053921a30db458a3219452ac7fd0963`
- Upstream relation: `ahead 0 / behind 2` relative to `origin/main`

## Summary

This repo has local source-code changes and local runtime artifacts.

## Tracked file changes

- `run.sh`
  - Local launch script adjustments.
- `wan/generate.py`
  - Main generation logic changed.
- `wan/inference/pipeline_wan_w_mem_relative_rope.py`
  - Core inference pipeline changed heavily.

`git diff --stat` summary:

- `3 files changed, 200 insertions(+), 84 deletions(-)`

## Untracked files

- `run_wan.sh`
  - New local helper script.
- `outputs_wan_split_small/`
  - Local output directory with generated media and logs.
