# Policy Runtime Guide

This directory contains only policy adapter code. It does not include models,
checkpoints, tokenizers, or runtime assets.

Each policy subdirectory should answer three questions:

1. How should the environment be prepared?
2. How are checkpoints and assets injected from outside the repository?
3. What path contract does the `RoboCombine` runner expect for this policy?

Currently supported:

- `Motus`
- `pi05`

Entry points:

- [Motus](./Motus/README.md)
- [pi05](./pi05/README.md)

## Recommended Workspace Layout

- `Motus` Python: `/path/to/motus/python`
- `pi05` Python: `/path/to/pi05/python3.11`
- simulator assets: `/path/to/simulator/assets`

Replace the placeholder paths with your own runtime locations instead of
copying machine-specific local paths into the public docs.

## Shared Principles

- no tracked models in the repository
- no tracked symlinks
- no committed local `.venv`
- each policy uses its own Python executable
- all checkpoint, tokenizer, WAN, and VLM paths are injected through CLI flags
  or environment variables
- public documentation uses generic placeholder paths and path contracts, not
  personal local paths
