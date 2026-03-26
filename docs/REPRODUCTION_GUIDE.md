# Reproduction Guide

This guide documents the intended Colab A100 workflow for the repository.

## 1. Mount storage and install dependencies

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
pip install -r requirements.txt
pip install -e .
```

## 2. Define the runtime root

Use a path variable rather than hardcoding file links inside the code base.

```bash
ROOT="/content/drive/Shareddrives/HLS - TOPVIEW/patched"
LOCAL_ROOT="/content/topview_cache"
```

## 3. Run the thermal audit on the raw runtime root

```bash
python -m topview.cli.audit_thermal_inputs --root "$ROOT"
```

This step inspects the raw patch directory before train/validation/test filtering
and writes a thermal provenance report under `reports/thermal_audit/`.

## 4. Build the manifest and random patch splits

```bash
python -m topview.cli.build_manifest --root "$ROOT"
python -m topview.cli.prepare_splits --root "$ROOT" --strategy random_patch
```

## 5. Stage the active split to the local Colab disk

```bash
python -m topview.cli.stage_to_local --source-root "$ROOT" --dest-root "$LOCAL_ROOT" --split all
```

This step reduces repeated Drive I/O during model training.

## 6. Train and evaluate

```bash
python -m topview.cli.train_unet --root "$LOCAL_ROOT"
python -m topview.cli.train_diffusion --root "$LOCAL_ROOT"
python -m topview.cli.evaluate_methods --root "$LOCAL_ROOT"
python -m topview.cli.build_thesis_tables --root "$LOCAL_ROOT"
```

## 7. Synchronize outputs back to Drive

```bash
python -m topview.cli.sync_artifacts_to_drive --local-root "$LOCAL_ROOT" --drive-root "$ROOT"
```

## 8. Optional real-cloud inference

```bash
python -m topview.cli.run_real_cloud_inference --root "$ROOT" --manifest "/content/drive/.../real_cloud_manifest.csv" --checkpoint "/content/topview_cache/runs/diffusion_.../best.pt"
```

## Notes

- The repository supports `.nc` and `.npz` patch inputs.
- Thermal bands are modeled in Kelvin brightness temperature. Celsius-like inputs are converted to Kelvin only when the patch metadata supports an HLS/Landsat interpretation, and ambiguous thermal inputs raise an error.
- Cloud-free patch selection follows the thesis threshold `p_cloud < 0.1`.
- The default split strategy is a random patch-level `80/10/10` partition, matching the thesis protocol.
- Learning-based models receive the HLS Fmask channel and a soft cloud mask as auxiliary inputs.
- Training defaults follow the thesis settings of batch size `4`, `50` epochs, and a `40,000` iteration cap with early stopping.
- Synthetic benchmark evaluation and real-cloud inference are kept as separate entry points.
