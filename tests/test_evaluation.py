import numpy as np
import polars as pl
import pytest

from Model_Evaluation.evaluation import ave_by_group, gini


def test_gini_perfect_and_reversed_ranking():
    actual = np.array([0.0, 100.0, 500.0, 2_000.0, 10_000.0])
    assert gini(actual, actual) == pytest.approx(1.0)
    assert gini(actual, -actual) < 0


def test_ave_by_group_ratio():
    data = pl.DataFrame(
        {
            "loss_year": [2014, 2014, 2015],
            "Incurred": [100.0, 300.0, 50.0],
            "expected_cost": [200.0, 200.0, 100.0],
        }
    )
    result = ave_by_group(data, "loss_year")
    assert result["claims"].to_list() == [2, 1]
    assert result["a_over_e"].to_list() == pytest.approx([1.0, 0.5])
