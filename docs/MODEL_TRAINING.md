# CKD Model Training and Evaluation

## Task

Binary classification of the source `class` field: `ckd` is positive and `notckd` is negative. This is current-label classification on a small retrospective data collection, not prediction of future CKD onset. Do not infer clinical utility from internal scores.

## Data and preprocessing

The supplied ARFF has 400 observations, 24 predictors, 11 numeric and 13 nominal variables, and 250/150 labels. `?` and empty values become missing. A malformed extra blank field in the source is normalized after validating row width. Exact duplicate feature rows are grouped to prevent cross-partition leakage; the current file has none.

Numeric fields use median imputation with missing indicators and candidate-selected standard, robust, or no scaling. Categorical fields use most-frequent or `Unknown` imputation and one-hot encoding with unknown/rare-category handling. Derived inputs are urea-to-creatinine ratio with zero guard and count of missing numeric labs. No rows are removed based on IQR; outliers are reported for review only.

## Model screening and split

Thirty fixed configurations compare logistic regression, SVM, KNN, random forest, ExtraTrees, and histogram gradient boosting. They vary regularization, scaling, class weighting, neighbor count, tree depth/leaf size, learning rate, and L2 regularization. Five-fold `StratifiedGroupKFold` uses identical-row hashes as groups. Candidate selection is based on mean CV ROC-AUC, with average precision as a tie-breaker.

A separate stratified patient-record holdout is reserved before model selection; a validation subset chooses operating thresholds. The selected estimator is sigmoid-calibrated by cross-validation on the training partition. The final test set is not used to select candidates or thresholds. With only 400 records, uncertainty remains high even when metrics are strong.

## Metrics and artifacts

The holdout report includes ROC-AUC, average precision, accuracy, balanced accuracy, precision, recall, F1, Brier score, alert rate, and confusion matrix. A second validation threshold targets 80% recall; its test results are reported separately. Artifacts under `artifacts/ckd/` include model comparison, serialized model, report, registry, model card, missingness, IQR outlier review, permutation importance, drift, holdout predictions, PR/ROC/confusion/calibration plots, and latency percentiles.

## Latest run

`random_forest_d4_leaf2_balanced` was selected. On 80 held-out records: ROC-AUC 0.996, AP 0.998, accuracy 0.975, precision 0.962, recall 1.000, F1 0.980, balanced accuracy 0.967, Brier 0.021. The alternative high-recall threshold produced 0.920 test recall and 1.000 precision. These are internal benchmark estimates, not external validation. The target may be strongly encoded by contemporaneous labs and comorbidities; prediction-time availability must be defined before interpreting this as prospective risk modeling.
