# TOPVIEW Thermal Reconstruction

Thermal cloud-gap reconstruction pipeline for TOPVIEW using Google Colab, Google
Drive, geostatistical baselines, U-Net, and diffusion models.

## Overview

This repository provides an end-to-end implementation of the thesis workflow for
reconstructing cloud-affected thermal remote sensing patches. The code base is
organized to make the experimental chain explicit, reproducible, and easy to review.

The workflow covers:

1. Dataset inventory generation from patch files
2. Train/validation/test split generation
3. Deterministic synthetic cloud generation for controlled benchmarking
4. Baseline reconstruction with inverse distance weighting and ordinary kriging
5. Learning-based reconstruction with a regression U-Net and a conditional
   diffusion model
6. Cloud-only evaluation, thin/thick cloud robustness analysis, and thesis-style
   table export

## Repository Structure

Core repository files:

- `README.md`
- `CHANGELOG.md`
- `CONTRIBUTING.md`
- `docs/REPRODUCTION_GUIDE.md`
- `docs/REPOSITORY_LAYOUT.md`
- `docs/SUBMISSION_CHECKLIST.md`
- `notebooks/TOPVIEW_Colab_Workflow.ipynb`

Implementation package:

- `src/topview/config.py`
- `src/topview/io.py`
- `src/topview/manifest.py`
- `src/topview/legacy_manifest.py`
- `src/topview/splits.py`
- `src/topview/clouds.py`
- `src/topview/datasets.py`
- `src/topview/baselines.py`
- `src/topview/models/`
- `src/topview/training.py`
- `src/topview/evaluation.py`
- `src/topview/reporting.py`
- `src/topview/cli/`

## Runtime Directory Layout

The repository uses a configurable runtime root for both input data and generated
artifacts. A typical Google Colab + Google Drive setup is:

```python
ROOT = "/content/drive/Shareddrives/HLS - TOPVIEW/patched"
```

Under the runtime root, the workflow uses the following directories:

- `manifests/` for dataset inventories
- `splits/` for saved train/validation/test partitions
- `runs/` for model checkpoints and training logs
- `reports/` for evaluation outputs and summary tables
- `exports/` for derived deliverables and prediction exports

For Colab training, the recommended pattern is:

`Google Drive archive -> local cache in /content -> train/evaluate on local disk -> sync artifacts back to Drive`

Raw data do not need to be committed to the repository. Keep patch files outside
the repository root or under ignored runtime directories such as `manifests/`,
`splits/`, `runs/`, `reports/`, and `exports/`.

## Supported Data Inputs

The repository supports both `.nc` and `.npz` patch files.

### netCDF patches

The preferred raw format is a netCDF patch containing variables equivalent to:

- `B02`-`B07` for reflectance
- `B10` and `B11` for thermal bands
- `Fmask` for cloud/shadow QA

The loader also accepts lower-case variable names and can incorporate sidecar JSON
metadata when present.

For thesis-aligned training and evaluation, thermal inputs are modeled in Kelvin
brightness temperature. The training and evaluation pipeline auto-converts
Celsius-like thermal inputs to Kelvin only when the patch metadata indicates an
HLS/Landsat-style thermal source, and raises an error when the thermal values are
ambiguous or out of range.

### NPZ patches

An `.npz` patch may contain:

- `reflectance`: shape `(6, H, W)`
- `thermal`: shape `(2, H, W)`
- `fmask`: shape `(H, W)`

Optional fields:

- `soft_mask`
- `metadata_json`

If a sidecar JSON file with the same stem exists, it is used to populate metadata
such as `city`, `tile_id`, `acquisition_date`, and `patch_id`.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Option A: build a manifest from patch files

```bash
python -m topview.cli.build_manifest --root "/content/drive/Shareddrives/HLS - TOPVIEW/patched"
python -m topview.cli.prepare_splits --root "/content/drive/Shareddrives/HLS - TOPVIEW/patched" --strategy random_patch
python -m topview.cli.stage_to_local --source-root "/content/drive/Shareddrives/HLS - TOPVIEW/patched" --dest-root "/content/topview_cache" --split all
python -m topview.cli.train_unet --root "/content/topview_cache"
python -m topview.cli.train_diffusion --root "/content/topview_cache"
python -m topview.cli.evaluate_methods --root "/content/topview_cache"
python -m topview.cli.build_thesis_tables --root "/content/topview_cache"
python -m topview.cli.sync_artifacts_to_drive --local-root "/content/topview_cache" --drive-root "/content/drive/Shareddrives/HLS - TOPVIEW/patched"
```

## Colab Workflow

The repository includes a ready-to-run notebook:

- `notebooks/TOPVIEW_Colab_Workflow.ipynb`

This notebook mirrors the recommended Colab A100 workflow:

1. Mount Google Drive
2. Install dependencies
3. Generate the manifest
4. Build random patch splits
5. Stage active data to `/content`
6. Train models
7. Run evaluation
8. Export thesis-style tables
9. Sync outputs back to Google Drive

## Method Components

### Dataset preparation

- Inventory generation from raw patches
- Metadata normalization for city, tile, product, and acquisition date
- Cloud-free patch selection with `p_cloud < 0.1`
- Random 80/10/10 train/validation/test partitioning
- Fmask and soft cloud masks passed as auxiliary inputs to learning-based models
- Random rotations and flips during training
- Thermal inputs validated as Kelvin-compatible before synthetic cloud generation

### Synthetic cloud generation

- Deterministic seed generation per patch and per split
- Multi-scale cloud opacity fields
- Reflectance-space cloud synthesis
- Radiance-space thermal cloud synthesis for B10/B11

### Baselines

- Inverse Distance Weighting
- Ordinary Kriging

The thesis-aligned baseline defaults used by the evaluation workflow are:

- IDW with inverse-distance-squared weighting and `50` nearest valid neighbours
- Ordinary Kriging with an `exponential` variogram model

### Learning-based models

- Regression U-Net
- Conditional diffusion model with RePaint-style inpainting

### Evaluation

- Cloud-only RMSE
- Mean bias error
- Patch-level Pearson correlation
- Thin/thick cloud RMSE
- Global mean thermal RMSE
- PSNR and SSIM for optical reconstruction support
- Thesis-style summary tables

## Reproducibility Choices

The repository follows the thesis logic in the following areas:

- Split generation is explicit and saved to disk
- Cloud-free patch selection uses a `< 10%` cloud threshold
- Random patch splitting is the default `80/10/10` strategy
- Synthetic cloud generation is deterministic per patch and per split
- Training and evaluation outputs are saved as machine-readable artifacts
- Synthetic benchmark evaluation and real-cloud inference are separated

## Main CLI Commands

Manifest and split generation:

- `python -m topview.cli.build_manifest`
- `python -m topview.cli.prepare_splits`

Data staging:

- `python -m topview.cli.stage_to_local`
- `python -m topview.cli.sync_artifacts_to_drive`

Training:

- `python -m topview.cli.train_unet`
- `python -m topview.cli.train_diffusion`

Evaluation and reporting:

- `python -m topview.cli.evaluate_methods`
- `python -m topview.cli.build_thesis_tables`
- `python -m topview.cli.run_real_cloud_inference`
- `python -m topview.cli.audit_thermal_inputs`

## Outputs

Typical generated outputs include:

- `manifests/patch_manifest.csv`
- `splits/patch_splits.csv`
- `runs/.../best.pt`
- `runs/.../history.jsonl`
- `reports/evaluation_summary.json`
- `reports/evaluation_per_patch.csv`
- `reports/thermal_audit/thermal_audit.csv`
- `reports/thermal_audit/thermal_audit_summary.json`
- `reports/tables/table_4_1_cloud_only_primary.csv`
- `reports/tables/table_4_2_thin_thick.csv`
- `reports/tables/table_4_3_global_optical.csv`
- `reports/tables/table_4_4_citywise_diffusion.csv`

## Additional Documentation

- `docs/REPRODUCTION_GUIDE.md`
- `docs/REPOSITORY_LAYOUT.md`
- `docs/SUBMISSION_CHECKLIST.md`

## License

Add a project license here if the repository is to be distributed formally.
