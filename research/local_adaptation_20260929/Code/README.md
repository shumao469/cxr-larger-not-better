# CXR local adaptation analysis

This release accompanies the four-figure BMC Medicine manuscript revision dated 29 September 2026. It contains the completed 66-source-model and 576-local-binary-fit analyses. The separate unfinished source-weighted-loss extension is outside the reported result set. The dated protocol retains historical plans for provenance; the completed scope is the model registry in Source_Data.

## Reproduce aggregate checks without original images

Install requirements-analysis.txt, then from this directory run:

    python verify_aggregate_results.py --data ../Source_Data

The check reads released aggregate outputs, verifies the main denominators and paired effects, and writes nothing unless --output is supplied. It neither refits models nor recomputes image-level bootstrap samples from the aggregate tables.

## Reproduce scientific figures

    python figures/reproduce_current_figures.py --source-data ../Source_Data --output ./reproduced

This regenerates Figures 2–4 from frozen aggregate outputs. Figure 1 additionally requires authorized illustrative radiograph assets supplied through --assets: chexpert_example.png, nih_example.png and vindr_example.png. Original images are not redistributed here. The editable PowerPoint is a separately delivered authoring artifact.

## Recorded analysis and source-training routines

- nc_research/train_controlled.py implements ImageNet DenseNet-121/ResNet-34 source training, masked loss, successful-update counting, resumable batches, source stopping and matched-update final checkpoints.
- nejm_ai_recovery contains probability metrics, tied-score paired AUROC resampling, consensus-reference comparisons and empirical/order-statistic threshold routines.
- nc_research_v2 contains local-budget calibration, local-head training, joint calibration/assessment resampling and head operating-point uncertainty.
- nc_research_v3/effect_audit.py calculates the final four-finding paired head effects and direct source contrasts.
- protocols retains the dated settings. Split salts ldh-2026 and nejm-ai-20260926 are unchanged.

For original-data analyses, set CXR_PROJECT_ROOT to an authorized project mirror with the original cohort manifests, 66 checkpoint directories, source-validation predictions, consensus roles/predictions and local frozen features. The original analysis routines expect the documented private intermediate hierarchy below that root. Training registries must contain valid local image and subset paths. Replace the three PRIVATE_* path values in source_training.json with local locations. Scripts write stage-specific results under Code/outputs; required completed-stage records and inputs must be supplied in dependency order. Exact reruns require the retained private artifacts and their compatible runtime. The packaged originals are recorded research routines, not a single-command public-data download pipeline.

Order of analytical dependencies: source cohort manifests -> source training -> source/consensus inference -> local frozen features -> calibration and head fitting -> paired/joint uncertainty -> aggregate summaries and four-finding effect audit. Review each entry point before running; training entry points perform actual model fitting. No training is started by the aggregate or plotting commands.

## Focused numerical tests

    python -m unittest discover -s nejm_ai_recovery -p "test_*.py"
    python nc_research_v2/test_extension.py

Training-specific tests require PyTorch and torchvision. The code_provenance.json file records source and released-script hashes; path changes are isolated from numerical routines. The aggregate release excludes record-level identifiers, predictions and local source paths. Provider-specific access terms govern original images and labels.
