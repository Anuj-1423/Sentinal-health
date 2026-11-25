from sentinel_health.ckd_data import FEATURE_COLUMNS, TARGET, load_dataset, make_features, profile_dataset
from sentinel_health.ckd_training import _candidates


def test_ckd_arff_schema_target_and_missing_values():
    df = load_dataset()
    assert df.shape[0] == 400
    assert len(FEATURE_COLUMNS) == 24
    assert df[TARGET].value_counts().to_dict() == {"ckd": 250, "notckd": 150}
    assert df.isna().sum().sum() > 0
    assert profile_dataset(df)["duplicate_rows"] == 0


def test_ckd_feature_contract_and_thirty_candidates():
    df = load_dataset()
    X = make_features(df.head(30))
    assert "target" not in X and TARGET not in X
    assert {"urea_creatinine_ratio", "lab_missing_count"}.issubset(X.columns)
    assert len(_candidates(X)) == 30
