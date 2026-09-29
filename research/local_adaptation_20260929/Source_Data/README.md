# Aggregate source data

Clinical_operating_points.csv is the source of main Table 2: sensitivity and specificity are fractions; fn is missed reference-positive images among 69; fp is false-positive flags among 1,931; flagged is true-positive plus false-positive flags among 2,000. Counts are means over fitted models, not integers from a newly combined classifier. models counts repeated fitted models rather than independent cohorts.

Head_effects_and_source_contrasts.csv supplies Figure 3 and Table S5: effect = local head minus original head; nih_minus_chexpert subtracts those source effects on a common benchmark. Brier values compare both heads after separate local calibration. lower and upper are unadjusted 2.5th and 97.5th percentiles from 2,000 paired assessment resamples, conditional on existing heads and maps. Both local losses are retained.

Joint_uncertainty_summary.csv and Joint_paired_strategy_differences.csv supply Figure 2: sensitivity/specificity are fractions; multiply differences by 100 for percentage points. Joint intervals resample calibration and assessment pools. valid/requested report feasible bootstrap replicates. Local_head_operating_intervals.csv records local-head-minus-original effects and conditions on fixed heads.

Paired_scale_gains.csv supplies Figure 4: gains are smallest-to-full macro-average AUROC differences; bootstrap_97_5_lower/upper are 97.5% intervals. SDs across three source seeds are distinct from the image-bootstrap intervals. Consensus_macro_metrics.csv and Model_registry_and_training.csv retain model-level outcomes and training conditions. The registry's convergence tag denotes source stopping, not demonstrated numerical convergence.

All aggregate CSV headers and row counts are enumerated in Data_dictionary.json. Earlier pleural-effusion-only head interval files are retained as provenance; main head-effect reporting uses the subsequent shared-resample audit above. The incomplete source-loss extension and its interim results are excluded. These files contain model/settings identifiers and aggregate outcomes, not individual patient/image identifiers or record-level predictions.

The revised text adds three main tables and Tables S12–S14 describing settings, comparators and literature context. Original numerical outputs are retained. Main Figure 3 panels now combine discrimination and calibrated Brier effects with budget curves and fitting availability; Figure 4 adds a design matrix and all observed update counts.
