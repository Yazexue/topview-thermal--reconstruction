# Contributing

## Repository Conventions

- Keep runtime paths configurable. Do not hardcode machine-specific or public file links in source code.
- Preserve the separation between synthetic benchmark evaluation and real-cloud inference.
- Prefer grouped data splits over unrestricted random patch splits when adding new experiments.
- Save machine-readable outputs for any new training or evaluation workflow.

## Code Style

- Use Python 3.10+.
- Keep modules small and task-oriented.
- Add short comments only where the logic is not obvious from the code itself.

## Pull Request Expectations

- Update `README.md` if the workflow changes.
- Update `docs/REPRODUCTION_GUIDE.md` if the execution steps change.
- Update `CHANGELOG.md` for user-visible repository changes.
