import polars as pl
import polars.selectors as cs


def parse_loss_date(data: pl.DataFrame) -> pl.DataFrame:
    loss_date = pl.col("date_of_loss").str.to_date("%Y-%m-%d")
    return data.with_columns(
        loss_date,
        loss_year=loss_date.dt.year(),
        loss_month=loss_date.dt.month(),
    )


def cast_strings_to_categorical(
    data: pl.DataFrame, exclude: list[str] | None = None
) -> pl.DataFrame:
    exclude = exclude or []
    return data.with_columns(
        (cs.string() - cs.by_name(exclude, require_all=False)).cast(pl.Categorical)
    )


def standardise_columns(data: pl.DataFrame) -> pl.DataFrame:
    data = data.rename({"Vechile_registration_present": "Vehicle_registration_present"})
    # Replace all spaces in column headings with a underscore for cleaner titles
    data.columns = [col.replace(" ", "_") for col in data.columns]
    return data


def correct_notification_period(data: pl.DataFrame) -> pl.DataFrame:
    return data.with_columns(pl.col("Notification_period").clip(lower_bound=0))


def drop_negative_claim_amounts(data: pl.DataFrame) -> pl.DataFrame:
    return data.filter(pl.col("Incurred") >= 0)


def create_non_null_flag(data: pl.DataFrame) -> pl.DataFrame:
    data = data.with_columns(
        pl.when(pl.col("Incurred") > 0)
        .then(1)
        .otherwise(0)
        .alias("non_zero_claim_flag")
    )
    return data


def weather_correction(data: pl.DataFrame) -> pl.DataFrame:
    return data.with_columns(pl.col("Weather_conditions").replace("N/A", "N/K"))


def drop_low_cardinality_features(data: pl.DataFrame) -> pl.DataFrame:
    cols_to_drop = [
        "Loss_code",
        "Loss_description",
        "TP_type_insd_pass_front",
        "TP_type_pass_multi",
        "Vehicle_registration_present",
    ]
    return data.drop(cols_to_drop)


def ph_considered_at_fault_correction(data: pl.DataFrame) -> pl.DataFrame:
    return data.with_columns(pl.col("PH_considered_TP_at_fault").replace("#", "n/k"))


# Look at mean costs by TP type
# Create aggregared columns for TP (passengers, vulnerable, total TP )


def add_tp_features(data: pl.DataFrame) -> pl.DataFrame:

    region_cols = [c for c in data.columns if c.startswith("TP_region_")]
    region_names = {i: c.removeprefix("TP_region_") for i, c in enumerate(region_cols)}

    type_cols = [c for c in data.columns if c.startswith("TP_type_")]
    n_regions = pl.sum_horizontal([pl.col(c) > 0 for c in region_cols])
    return data.with_columns(
        tp_total=pl.sum_horizontal(type_cols),
        tp_passengers=pl.sum_horizontal(
            "TP_type_pass_front", "TP_type_pass_back", "TP_type_insd_pass_back"
        ),
        tp_bike_pedest=pl.sum_horizontal(
            "TP_type_bike", "TP_type_cyclist", "TP_type_pedestrian"
        ),
        tp_injured_known=pl.sum_horizontal(
            "TP_injury_whiplash", "TP_injury_traumatic", "TP_injury_fatality"
        ),
        tp_serious_injury_flag=(
            pl.col("TP_injury_traumatic") + pl.col("TP_injury_fatality")
        )
        > 0,
        tp_region=pl.when(n_regions == 0)
        .then(pl.lit("none"))
        .when(n_regions > 1)
        .then(pl.lit("multi"))
        .otherwise(
            pl.concat_list(region_cols)
            .list.arg_max()
            .replace_strict(region_names, return_dtype=pl.String)
        )
        .cast(pl.Categorical),
        tp_region_total=pl.sum_horizontal(region_cols),
    ).drop(["TP_type_bike", "TP_type_pedestrian", "TP_type_cyclist", *region_cols])

def create_large_loss_flag(data: pl.DataFrame) -> pl.DataFrame:
    return data.with_columns(pl.when(pl.col("Incurred") > 50000).then(1).otherwise(0).alias("Large_Loss_Flag"))