import numpy as np
import polars as pl
import pytest


@pytest.fixture
def raw_claims() -> pl.DataFrame:
    """A few hand-made claims covering the edge cases the cleaning steps handle."""
    return pl.DataFrame(
        {
            "Claim_Number": [1, 2, 3, 4],
            "date_of_loss": ["2010-03-15", "2012-07-01", "2014-12-31", "2015-06-30"],
            "Notification_period": [5, -3, 0, 400],
            "Incurred": [0.0, -250.0, 1_200.0, 75_000.0],
            "Weather_conditions": ["N/A", "NORMAL", "N/K", "WET"],
            "PH_considered_TP_at_fault": ["#", "N", "Y", "n/k"],
            "TP_type_pass_front": [0, 1, 0, 2],
            "TP_type_pass_back": [0, 0, 0, 1],
            "TP_type_insd_pass_back": [0, 0, 0, 0],
            "TP_type_bike": [0, 0, 1, 0],
            "TP_type_cyclist": [0, 0, 0, 0],
            "TP_type_pedestrian": [0, 0, 0, 1],
            "TP_injury_whiplash": [0, 1, 0, 2],
            "TP_injury_traumatic": [0, 0, 0, 1],
            "TP_injury_fatality": [0, 0, 0, 0],
            "TP_region_london": [0, 1, 0, 2],
            "TP_region_scotland": [0, 0, 0, 1],
            "TP_region_wales": [0, 0, 2, 0],
        }
    )


@pytest.fixture
def model_data() -> pl.DataFrame:
    """Synthetic modelling data: cost rises with injuries, some nil and large claims."""
    rng = np.random.default_rng(0)
    n = 600
    injured = rng.integers(0, 4, n)
    mobile = rng.choice(["Y", "N"], n)
    nonnil = rng.random(n) < np.where(mobile == "N", 0.9, 0.6)
    capped = np.where(nonnil, rng.gamma(2.0, 1_500 * (1 + injured)), 0.0)
    large = nonnil & (rng.random(n) < 0.02 + 0.05 * injured)
    incurred = np.where(large, 50_000 + rng.gamma(1.0, 80_000, n), np.minimum(capped, 49_000))
    return pl.DataFrame(
        {
            "loss_year": rng.integers(2005, 2016, n),
            "tp_injured_known": injured,
            "Vehicle_mobile": pl.Series(mobile).cast(pl.Categorical),
            "Incurred": incurred,
            "Capped_Incurred": np.minimum(incurred, 50_000),
            "non_zero_claim_flag": (incurred > 0).astype(int),
            "Large_Loss_Flag": (incurred > 50_000).astype(int),
        }
    )
