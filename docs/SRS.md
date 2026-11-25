# Sentinel Health CKD System Requirements

## Functional
- Read and validate the supplied 400-row CKD ARFF schema and binary class labels.
- Normalize missing markers and the source's malformed blank cell.
- Profile class balance, missingness, outlier flags, numeric and nominal distributions.
- Engineer guarded urea/creatinine ratio and missing-lab count.
- Run 30 model/preprocessing candidates with fold-fitted transformations and duplicate-group isolation.
- Calibrate probabilities, choose operating thresholds on validation, and report holdout ROC-AUC, AP, precision, recall, F1, Brier, and confusion matrix.
- Save plots, comparisons, model card, registry, quality, drift, feature importance, and model artifacts locally.
- Provide local Flask dashboard and FastAPI scoring/readiness/metrics endpoints.
- Persist aggregate scores and service metadata only; do not persist raw patient fields.

## Constraints
- The 400-record historical dataset is educational only and is not clinically validated.
- Never enter real patient information.
- No Docker or CI/CD pipeline is included.
- Features must be available at the intended prediction time before interpreting outputs as prospective risk.

## Acceptance
The loader validates 400 rows, all 30 configurations complete, the serialized model reloads and predicts, the CKD dashboard loads, the API scoring function responds, and project tests pass. These checks do not establish clinical validity.
