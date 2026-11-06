import polars as pl

from functions import (
    parse_loss_date,
    standardise_columns,
    correct_notification_period,
    drop_negative_claim_amounts,
    create_non_null_flag,
    weather_correction,
    drop_low_cardinality_features,
    add_tp_features,
    cast_strings_to_categorical,
    ph_considered_at_fault_correction,
)


class Dataloader:
    def run_dataloader(self, data):
        return (
            data.pipe(parse_loss_date)
            .pipe(standardise_columns)
            .pipe(correct_notification_period)
            .pipe(drop_negative_claim_amounts)
            .pipe(create_non_null_flag)
            .pipe(weather_correction)
            .pipe(drop_low_cardinality_features)
            .pipe(ph_considered_at_fault_correction)
            .pipe(add_tp_features)
            .pipe(cast_strings_to_categorical)
        )
