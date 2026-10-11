# Contributing

Contributions to RoboCombine are welcome as long as they preserve the benchmark
protocol and keep generated artifacts out of the source repository.

## Development Setup

Use the separate policy environments described in
[docs/runtime_envs.md](docs/runtime_envs.md): Python 3.10 for Motus and
Python 3.11 for pi05.

Runtime simulator assets, model checkpoints, WAN/VLM directories, and optional
progress-predictor bundles are external dependencies and must remain outside Git.

## Validation

Before proposing changes, run:

```bash
python -m pytest tests
```

If your change touches evaluation routing, also run a one-episode smoke test for:

- `full_inst`
- `oracle_hse`
- `ordered_hse`

If your change touches packaging or release-facing paths, also run:

```bash
uv build --sdist
```

## Pull Request Guidelines

- Keep protocol changes explicit in `handoff_eval/specs/robocombine.json`.
- Update `README.md`, `policy/*.md`, and public `docs/*.md` when release-facing
  behavior changes.
- Do not commit local logs, generated runs, checkpoints, videos, HDF5 files, or
  machine-specific symlinks.
- Prefer small, reviewable changes with focused tests.
