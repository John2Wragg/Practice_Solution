import polars as pl
import pytest

from Model_Building.ModelBuilder import ModelBuilder

FEATURES = ["loss_year", "tp_injured_known", "Vehicle_mobile"]
PARAMS = {"n_estimators": 20, "num_leaves": 4, "min_child_samples": 10}


@pytest.fixture
def fitted_model(model_data):
    mb = ModelBuilder(
        nil_features=FEATURES,
        nil_params=PARAMS,
        sev_features=FEATURES,
        sev_params=PARAMS,
        large_loss_features=FEATURES,
        large_loss_params=PARAMS,
    )
    return mb.fit(model_data)


def test_predict_shape_and_ranges(fitted_model, model_data):
    preds = fitted_model.predict(model_data)
    assert preds.height == model_data.height
    assert preds["p_nonnil"].is_between(0, 1).all()
    assert preds["p_large_loss"].is_between(0, 1).all()
    assert (preds["severity_capped"] > 0).all()


def test_expected_cost_is_capped_plus_excess(fitted_model, model_data):
    preds = fitted_model.predict(model_data)
    rebuilt = preds["expected_capped"] + preds["expected_excess"]
    assert (preds["expected_cost"] - rebuilt).abs().max() < 1e-6


def test_severity_balanced_on_training_data(fitted_model, model_data):
    positives = model_data.filter(pl.col("non_zero_claim_flag") == 1)
    preds = fitted_model.predict(positives)
    assert preds["severity_capped"].sum() == pytest.approx(positives["Capped_Incurred"].sum(), rel=1e-6)


def test_rebase_year_changes_only_the_year(fitted_model, model_data):
    rebased = fitted_model.predict(model_data, rebase_year=2015)
    same_year = fitted_model.predict(model_data.with_columns(pl.lit(2015).cast(pl.Int64).alias("loss_year")))
    assert rebased.equals(same_year)
