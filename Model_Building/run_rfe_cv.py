from Model_Building.rfe_cv import run_rfe
import json 
import os 
import polars as pl
from params.targets import NIL_TARGET, TARGET_COL
from params.generic_params import DO_NOT_MODEL

def main(run_name="test_run"):

    base_model_folder = rf"/Users/johnwragg/Desktop/DS_work/Hastings_Solution/Model_Experiments/{run_name}"
    os.makedirs(base_model_folder, exist_ok=True)

    # Setup model run
    train = pl.read_parquet(
        "/Users/johnwragg/Desktop/DS_work/Hastings_Solution/Prepped_Data/train.parquet"
    )

    features = [col for col in train.columns if col not in DO_NOT_MODEL]

    
    feature_selection_dict = {}
    for model in ["nil", "sev"]:
        target = NIL_TARGET if model == "nil" else TARGET_COL
        model_features = run_rfe(train, features, target, model)
        feature_selection_dict[model] = model_features
    
    # save features to model experiment folder
    os.makedirs(base_model_folder, exist_ok=True)
    features_path = os.path.join(base_model_folder, "model_features.json")
    with open(features_path, "w") as f:
        json.dump(feature_selection_dict, f, indent=2)


if __name__ == "__main__":
    main("test_run")