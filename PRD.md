# Sentinel Health CKD Product Requirements

## Objective

Provide a local educational analytics application that demonstrates CKD dataset inspection, preprocessing, model comparison, probability calibration, evaluation, and operational observability using the supplied ARFF files.

## Users and pages

- Overview and cohort label counts
- CKD analytics and EDA plots
- Data quality, missingness, and IQR outlier review
- Benchmark-row prediction demonstration
- 30-candidate model lab with holdout metrics and PR/ROC/calibration plots
- Permutation feature importance
- Offline/live observability
- Local model registry

## Model behavior

Positive label is `ckd`; negative label is `notckd`. Numeric and nominal fields are preprocessed inside model CV folds. Candidate selection uses stratified grouped CV; the holdout remains separate. Validation-only thresholds include maximum F1 and an 80%-recall target.

## Constraints

Only the included 400-row benchmark is used by default. No real patient input, automated care action, clinical recommendation, Docker, or CI/CD pipeline is part of scope. High internal metrics must be presented with small-sample and dataset-source limitations.
