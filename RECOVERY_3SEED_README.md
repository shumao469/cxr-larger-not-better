# Three-seed recovery and formal rerun

## Recommendation

Use **formal mode** for the NEJM AI manuscript. It retrains all 30 experiments in `results_formal_v2/` with a fixed validation cohort across seeds and strict image-read failures. It does not overwrite the recovered legacy results.

`salvage` mode can reuse complete legacy predictions and loadable checkpoints, but it mixes experiments selected with seed-dependent validation subsets and is therefore exploratory only.

## Why the image datasets are required

The recovered project contains the frozen split/label manifest but not the NIH, CheXpert, or VinDr-CXR image files. Predictions cannot be regenerated and damaged checkpoints cannot be retrained without the images.

Official access pages:

- NIH ChestX-ray14: https://nihcc.app.box.com/v/ChestXray-NIHCC
- CheXpert: https://aimi.stanford.edu/datasets/chexpert-chest-x-rays
- CheXpert canonical page: https://stanfordmlgroup.github.io/competitions/chexpert/
- VinDr-CXR v1.0.0: https://physionet.org/content/vindr-cxr/1.0.0/
- VinDr-CXR Kaggle competition package used by the legacy code: https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data

CheXpert and VinDr-CXR require registration and acceptance of their data-use terms. For exact continuity, use the legacy CheXpert v1.0 images rather than silently substituting a different cohort. The formal manifest builder matches CheXpert images by `patient/study/view`, so extraction-folder names may differ.

## Suggested WSL layout

```text
/mnt/e/data/raw/nih/
  Data_Entry_2017.csv
  train_val_list.txt
  test_list.txt
  images_001/images/*.png
  ...

/mnt/e/data/raw/chexpert/
  .../train/patient*/study*/view*.jpg
  .../valid/patient*/study*/view*.jpg

/mnt/e/data/raw/vindr/
  .../*.dicom

/mnt/e/data/processed/vindr_png/
  .../*.png
```

## 1. Convert VinDr-CXR DICOM images

```bash
cd /mnt/e/cxr_larger_not_better
conda activate cxr_torch_stable
pip install pydicom pylibjpeg pylibjpeg-libjpeg

python src/04_convert_vindr_dicom_to_png_v2.py \
  --dicom-root /mnt/e/data/raw/vindr \
  --png-root /mnt/e/data/processed/vindr_png
```

The v2 converter uses modality/VOI LUTs where available, 0.5th–99.5th percentile clipping, MONOCHROME1 inversion, and deterministic 8-bit PNG output. Because the recovered legacy converter was zero-filled, old seed-1 predictions must not be mixed with a formal v2 rerun.

## 2. Rebuild paths while preserving frozen splits

```bash
python src/13_rebuild_recovered_manifest.py \
  --source-manifest data/processed/harmonized_manifest_all_splits.csv \
  --nih-root /mnt/e/data/raw/nih \
  --chexpert-root /mnt/e/data/raw/chexpert \
  --vindr-png-root /mnt/e/data/processed/vindr_png \
  --out data/processed/model_manifest_formal_v2.csv
```

The script will refuse to write the final manifest unless all 274,439 expected images resolve, dataset/split counts match the recovered study, duplicate image IDs are absent, and patient overlap across splits is zero.

## 3. Review the plan without running GPUs

```bash
python src/14_plan_or_run_three_seed_recovery.py \
  --project-root . \
  --manifest data/processed/model_manifest_formal_v2.csv \
  --mode formal
```

## 4. Run the formal three-seed experiment

```bash
bash scripts/run_stage1_three_seed_formal_v2.sh
```

To include the slower source-validation recalibration stage:

```bash
bash scripts/run_stage1_three_seed_formal_v2.sh --with-calibration
```

Outputs are written below `results_formal_v2/`. The runner is resumable: complete metrics are skipped, complete predictions are evaluated, valid checkpoints are predicted, and only missing stages are rerun.

## 5. Optional exploratory salvage plan

```bash
python src/14_plan_or_run_three_seed_recovery.py \
  --project-root . \
  --manifest data/processed/model_manifest_formal_v2.csv \
  --mode salvage
```

Review `results_salvage_v2/recovery_plan.csv` before adding `--execute`.
