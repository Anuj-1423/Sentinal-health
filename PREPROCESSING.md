# CKD Data Preprocessing

- Parse the supplied ARFF schema and normalize the dataset's `?`, blank, and malformed extra-empty cells. Validate 400 rows and 24 feature fields plus binary class.
- Preserve `ckd` / `notckd` labels and map `ckd` to positive class 1.
- Keep all records; report exact duplicate rows and group identical predictor rows during splitting/CV.
- Convert 11 lab/demographic numeric fields to numeric with invalid values becoming missing. Keep 13 nominal fields categorical, including ordinal-coded source fields rather than assuming continuous spacing.
- Numeric pipeline: median imputation, missingness indicators, and variant scaling (standard, robust, or passthrough).
- Categorical pipeline: most-frequent or `Unknown` imputation, one-hot encoding, unknown-category handling, and rare-category grouping variants.
- Add a guarded blood-urea/serum-creatinine ratio and a count of missing numeric labs. Both are engineered strictly from input fields.
- Flag numeric IQR outliers for investigation; do not automatically clip or drop them.
- Fit all imputers, scalers, and encoders within CV/training folds to prevent leakage.

The dataset contains features closely related to diagnosis. Decide which are available at the intended prediction time; a model using post-diagnosis measurements only classifies this retrospective dataset and does not prove early prediction capability.
