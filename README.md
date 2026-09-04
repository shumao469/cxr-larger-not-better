# CXR Larger Is Not Better

Reproducible research code for studying how chest X-ray training-set size affects internal performance, external generalization, calibration, and diminishing returns.

The formal experiment grid contains two training sources (NIH ChestX-ray14 and CheXpert v1.0), five sample-size conditions (`1k`, `5k`, `10k`, `50k`, and all available training images), and three seeds. VinDr-CXR v1.0.0 is used as an external cohort. Formal mode keeps the validation cohort fixed across seeds and writes to a separate output tree so that exploratory or recovered legacy results cannot be mixed with the formal rerun.

## Repository scope

This repository contains:

- manifest construction, label harmonization, patient-level splitting, and subset generation;
- deterministic VinDr DICOM-to-PNG conversion and integrity auditing;
- GPU training, prediction, calibration, evaluation, and figure-generation code;
- resumable formal-run orchestration and grouped inference tooling;
- lossless resize-cache acceleration with fail-safe fallback to source images;
- unit and integration tests for the main audit and acceleration safeguards.

It intentionally excludes datasets, manifests containing image or patient identifiers, checkpoints, predictions, metrics, figures, logs, manuscript files, credentials, and machine-specific configuration. These remain on the local research workstation and are protected by `.gitignore`.

## Layout

```text
config/       portable configuration examples
scripts/      formal workflow and performance utilities
src/          data preparation, training, inference, audit, and plotting code
tests/        CPU-compatible unit and integration tests
```

`RECOVERY_3SEED_README.md` documents the formal-versus-salvage distinction and the original three-seed recovery route.

## Data access

Obtain each dataset directly from its official source and comply with its access and data-use terms:

- [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)
- [VinDr-CXR v1.0.0](https://physionet.org/content/vindr-cxr/1.0.0/)

The formal continuity workflow expects the original CheXpert v1.0 cohort; it should not be silently replaced with CheXpert Plus or another derivative. No dataset files are redistributed here.

## Environment

The completed workstation run used Ubuntu 22.04 under WSL, Python 3.12, CUDA 12.8-compatible PyTorch, and an NVIDIA GPU. Install the CUDA-specific `torch` and `torchvision` builds using the official PyTorch instructions, then install the remaining packages:

```bash
python -m pip install -r requirements.txt
```

Tested GPU environments used PyTorch 2.10/torchvision 0.25 for conversion and plotting, and PyTorch 2.11/torchvision 0.26 for the final training tests. The code does not require those exact patch versions, but a production rerun should record the resolved environment.

Copy the configuration templates and replace all placeholder paths with paths on the new host:

```bash
cp config/paths.example.yaml config/paths.yaml
cp config/lossless_cache.example.json config/lossless_cache_formal_v2.json
```

The committed templates are safe examples. The populated files are ignored because they may expose local paths and audit metadata.

## Formal workflow

Use explicit paths for all datasets and outputs. A typical WSL run is:

1. Convert VinDr DICOM files with `src/04_convert_vindr_dicom_to_png_v2.py`.
2. Rebuild and audit the frozen manifest with `src/13_rebuild_recovered_manifest.py` and `src/17_finalize_formal_manifest.py`.
3. Review the 30-run plan without executing it:

   ```bash
   python src/14_plan_or_run_three_seed_recovery.py \
     --project-root . \
     --manifest data/processed/model_manifest_formal_v2.csv \
     --mode formal
   ```

4. Start or resume formal training:

   ```bash
   MANIFEST=data/processed/model_manifest_formal_v2.csv \
     bash scripts/run_stage1_three_seed_formal_v2.sh
   ```

5. After all training-completion gates pass, generate grouped predictions, metrics, and figures:

   ```bash
   PROJECT_ROOT="$PWD" PYTHON_BIN=python \
     bash scripts/run_grouped_prediction_formal_v2.sh

   PROJECT_ROOT="$PWD" PYTHON_BIN=python \
     bash scripts/run_formal_metrics_v2.sh

   python src/21_plot_formal_results_v2.py \
     --metrics results_formal_v2/metrics/formal_mean_primary_metrics.csv \
     --out-dir results_formal_v2/figures
   ```

The formal scripts are create-only or overwrite-resistant where publication evidence is involved. Do not bypass completion markers, digest checks, patient-overlap checks, or fixed-validation checks merely to make a run finish.

## Verification

Run the CPU-compatible test suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

At repository preparation time, all 94 tests passed. This verifies the code-level safeguards; it does not replace dataset checksum verification, manifest auditing, GPU-run completion checks, or external validation.

## Research-use boundary

This is research software, not a medical device. It must not be used to diagnose patients or direct clinical care. Any manuscript or deployment claim should remain tied to the audited cohort definitions, prespecified splits, independent validation, calibration, subgroup analysis, and decision-utility evidence.
