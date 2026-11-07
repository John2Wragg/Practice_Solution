from Model_Building.rfe_cv import run_rfe
from Model_Building.ModelBuilder import ModelBuilder
from params.generic_params import DO_NOT_MODEL
from params.targets import NIL_TARGET, TARGET_COL
import polars as pl
import os
import json


def main(run_name="test_run"):

    base_model_folder = rf"/Users/johnwragg/Desktop/DS_work/Hastings_Solution/Model_Experiments/{run_name}"
    os.makedirs(base_model_folder, exist_ok=True)

    # Setup model run
    train = pl.read_parquet(
        "/Users/johnwragg/Desktop/DS_work/Hastings_Solution/Prepped_Data/train.parquet"
    )

    features = [col for col in train.columns if col not in DO_NOT_MODEL]

    # Load hpt trial for model_experiment
    params_path = os.path.join(base_model_folder, "best_params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f"No tuned params found at {params_path}. Run run_tuning.py for '{run_name}' first."
        )
    with open(params_path) as f:
        best_params = json.load(f)
    nil_params = best_params["nil"]
    sev_params = best_params["sev"]

    # run rfe for each model TODO: Put this separately so RFE is ran before tuning and features are loaded
    feature_selection_dict = {}
    for model in ["nil", "sev"]:
        target = NIL_TARGET if model == "nil" else TARGET_COL
        model_features = run_rfe(train, features, target, model)
        feature_selection_dict[model] = model_features

    # Run modelbuilder
    mb = ModelBuilder(
        nil_features=feature_selection_dict["nil"],
        nil_params=nil_params,
        sev_features=feature_selection_dict["sev"],
        sev_params=sev_params,
    )

    mb.fit(train)

    mb.save(os.path.join(base_model_folder, f"{run_name}.joblib"))


if __name__ == "__main__":
    main()
