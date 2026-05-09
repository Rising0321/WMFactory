# changeLog

This directory records the local differences of each model repository under `models/` relative to its configured `origin` remote and current upstream branch.

Notes:

- `ahead/behind` is relative to the configured upstream branch at the time of inspection.
- Most repos are `ahead 0`; local differences mainly come from uncommitted tracked changes and untracked local artifacts.
- `HY-WorldPlay` is currently `behind 2` relative to `origin/main`.
- Some diffs include runtime cache files such as `__pycache__`; those are called out explicitly where relevant.
