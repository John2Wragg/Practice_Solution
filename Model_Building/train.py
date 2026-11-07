from Model_Building.rfe_cv import run_rfe
from Model_Building.ModelBuilder import ModelBuilder
from params.generic_params import DO_NOT_MODEL
from params.targets import NIL_TARGET, TARGET_COL
import polars as pl

# TODO: Implement HPT in a simple script
MODEL_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=6,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.6,
    reg_lambda=10,
    cat_smooth=30,
    min_data_per_group=50,
    max_cat_to_onehot=8,
)
import os


def main(run_name="test_run"):

    base_model_folder = rf"/Users/johnwragg/Desktop/DS_work/Hastings_Solution/Model_Experiments/{run_name}"
    os.makedirs(base_model_folder, exist_ok=True)

    # Setup model run
    train = pl.read_parquet(
        "/Users/johnwragg/Desktop/DS_work/Hastings_Solution/Prepped_Data/train.parquet"
    )

    features = [col for col in train.columns if col not in DO_NOT_MODEL]

    # run rfe for each model
    feature_selection_dict = {}
    for model in ["nil", "sev"]:
        target = NIL_TARGET if model == "nil" else TARGET_COL
        model_features = run_rfe(train, features, target, model)
        feature_selection_dict[model] = model_features

    # Run modelbuilder
    mb = ModelBuilder(
        nil_features=feature_selection_dict["nil"],
        nil_params=MODEL_PARAMS,
        sev_features=feature_selection_dict["sev"],
        sev_params=MODEL_PARAMS,
    )

    mb.fit(train)

    mb.save(os.path.join(base_model_folder, f"{run_name}.joblib"))


if __name__ == "__main__":
    main()
