import polars as pl
import pytest

from functions.dataloader_helpers import (
    create_large_loss_flag,
    create_non_null_flag,
    drop_negative_claim_amounts,
)
from Model_Building.ModelBuilder import ModelBuilder


@pytest.fixture
def claims() -> pl.DataFrame:
    return pl.DataFrame({"Incurred": [-100.0, 0.0, 250.0, 50000.0, 75000.0]})


def test_drop_negative_claim_amounts_removes_only_negatives(claims):
    result = drop_negative_claim_amounts(claims)

    assert result["Incurred"].to_list() == [0.0, 250.0, 50000.0, 75000.0]


def test_claim_flags_follow_thresholds(claims):
    result = create_large_loss_flag(create_non_null_flag(claims))

    # Non-nil: strictly greater than zero
    assert result["non_zero_claim_flag"].to_list() == [0, 0, 1, 1, 1]
    # Large loss: strictly greater than 50,000
    assert result["Large_Loss_Flag"].to_list() == [0, 0, 0, 0, 1]


def test_interaction_constraints_isolate_year_column():
    builder = ModelBuilder(
        nil_features=[],
        nil_params={},
        sev_features=[],
        sev_params={},
        large_loss_features=[],
        large_loss_params={},
    )

    with_year = builder._interaction_constraints(["age", "loss_year", "region"])
    without_year = builder._interaction_constraints(["age", "region"])

    assert with_year == [[1], [0, 2]]
    assert without_year == []
