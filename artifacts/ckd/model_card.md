# CKD model card

- Task: binary classification of the supplied CKD dataset (`ckd` vs `notckd`).
- Selected by 5-fold duplicate-grouped stratified CV: random_forest_d4_leaf2_balanced.
- Holdout ROC-AUC: 0.9960; average precision: 0.9976.
- Small, single-source retrospective benchmark (n=400); no external validation.
- Educational use only; not for diagnosis, screening, or care decisions.
