# Contributing

Contributions to RoboCombine are welcome as long as they preserve the benchmark
protocol and keep generated artifacts out of the source repository.

## Development Setup

`pi05` and the shared runner use the local `uv` environment:

```bash
uv venv --python 3.10
uv pip install -e .
```

`Motus` uses a separate runtime. See [docs/runtime_envs.md](docs/runtime_envs.md)
for the validated environment layout.

Runtime simulator assets, model checkpoints, WAN/VLM directories, and optional
progress-predictor bundles are external dependencies and must remain outside Git.

## Validation

Before proposing changes, run:

```bash
uv run pytest
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
- Keep historical timelines, local test notes, and one-off release drafts under
  `docs-private/` or `metadata-private/`; they are local-only and should not be
  committed.
- Do not commit local logs, generated runs, checkpoints, videos, HDF5 files, or
  machine-specific symlinks.
- Prefer small, reviewable changes with focused tests.
