# Submission Checklist

This repository is prepared for advisor review and thesis submission support.

## What is explicit in this repository

- Runtime paths are parameterized through configuration and CLI arguments.
- Dataset manifest generation is reproducible.
- Train/validation/test splits are saved to disk.
- The default split strategy matches the thesis `80/10/10` random patch partition.
- Synthetic cloud generation is deterministic per patch and per split.
- Synthetic evaluation and real-cloud inference are separate commands.
- Training outputs, metrics, and checkpoints are saved under versioned run directories.

## Before handing off the repository

- Confirm the actual variable names inside the `.nc` files match the loader assumptions.
- Confirm the thermal bands are stored either in Kelvin brightness temperature or in Celsius values that can be safely converted to Kelvin.
- Run `build_manifest` and inspect the generated metadata CSV.
- Run `prepare_splits` and keep the split CSV under version control or archive it.
- Stage the active split to `/content` during Colab runs instead of streaming every batch from Drive.
- Export one completed training run and one completed evaluation report.
- Add one short note describing the exact Google Drive root used during reproduction.

## Recommended handoff package

- Repository source code
- `requirements.txt`
- `README.md`
- One manifest CSV example
- One split CSV example
- One training run folder with config, history, and best checkpoint
- One evaluation summary JSON
