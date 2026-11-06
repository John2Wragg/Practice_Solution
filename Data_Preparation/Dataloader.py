from params.generic_params import CAP_VALUE
from params.targets import TARGET_COL
import polars as pl
from typing import Tuple, Optional

from functions.dataloader_helpers import (
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
from sklearn.model_selection import train_test_split
from params.targets import TARGET_COL
from params.generic_params import CAP_VALUE


class Dataloader:
    def __init__(self, path: str):
        self.path = path
        self.target = TARGET_COL

    def load(self) -> pl.DataFrame:
        raw = pl.read_excel(self.path, sheet_name="Data")
        clean = self.prepare(raw)
        # validate with the Pandera schema here
        return clean
    

    def prepare(self, data):
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
    
    def split_data(self, data: pl.DataFrame, test_size: Optional[float] = 0.2, seed: Optional[int] = 42) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Split data for train test split.
        
        Args:
        -----

        data:
            Raw cleaned data
        test_size:
            Default 0.2
        seed:
            Default 42
        
        Returns:
        --------

        Tuple of train, test pl.DataFrames
        
        """
        
        strata = (
            pl.when(pl.col(self.target) == 0).then(pl.lit("nil"))
            .when(pl.col(self.target) == CAP_VALUE).then(pl.lit("large"))
            .otherwise(pl.lit("normal"))
        ) # change to capped value
        labels = data.select(strata.alias("s"))["s"].to_list()
        train_idx, test_idx = train_test_split(
            range(len(data)), test_size=test_size, random_state=seed, stratify=labels
        )
        return data[train_idx], data[test_idx]

