import polars as pl

from functions.dataloader_helpers import (
    add_tp_features,
    correct_notification_period,
    create_large_loss_flag,
    create_non_null_flag,
    drop_negative_claim_amounts,
    parse_loss_date,
    ph_considered_at_fault_correction,
    weather_correction,
)


def test_drop_negative_claim_amounts_keeps_zero_claims(raw_claims):
    result = drop_negative_claim_amounts(raw_claims)
    assert result["Incurred"].min() >= 0
    assert 0.0 in result["Incurred"].to_list()  # nil claims must survive
    assert result.height == raw_claims.height - 1


def test_correct_notification_period_clips_negatives_only(raw_claims):
    result = correct_notification_period(raw_claims)
    assert result["Notification_period"].to_list() == [5, 0, 0, 400]


def test_create_non_null_flag(raw_claims):
    result = create_non_null_flag(drop_negative_claim_amounts(raw_claims))
    assert result["non_zero_claim_flag"].to_list() == [0, 1, 1]


def test_large_loss_flag_is_strictly_above_cap():
    data = pl.DataFrame({"Incurred": [49_999.0, 50_000.0, 50_001.0]})
    assert create_large_loss_flag(data)["Large_Loss_Flag"].to_list() == [0, 0, 1]


def test_parse_loss_date_adds_year_and_month(raw_claims):
    result = parse_loss_date(raw_claims)
    assert result["loss_year"].to_list() == [2010, 2012, 2014, 2015]
    assert result["loss_month"].to_list() == [3, 7, 12, 6]


def test_category_corrections(raw_claims):
    result = ph_considered_at_fault_correction(weather_correction(raw_claims))
    assert "#" not in result["PH_considered_TP_at_fault"].to_list()
    assert "N/A" not in result["Weather_conditions"].to_list()
    assert result["Weather_conditions"].to_list() == ["N/K", "NORMAL", "N/K", "WET"]


def test_add_tp_features(raw_claims):
    result = add_tp_features(raw_claims)

    assert result["tp_region"].cast(pl.String).to_list() == ["none", "london", "wales", "multi"]
    assert result["tp_bike_pedest"].to_list() == [0, 0, 1, 1]
    assert result["tp_injured_known"].to_list() == [0, 1, 0, 3]
    assert result["tp_serious_injury_flag"].to_list() == [False, False, False, True]

    # raw columns that were combined are removed
    assert not any(c.startswith("TP_region_") for c in result.columns)
    assert "TP_type_bike" not in result.columns
