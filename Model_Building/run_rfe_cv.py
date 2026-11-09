from Model_Building.rfe_cv import run_rfe
import json 
import os 
import polars as pl
from params.targets import NIL_TARGET, TARGET_COL, LARGE_LOSS_TARGET
from params.generic_params import DO_NOT_MODEL

def main(run_name="test_run"):

    base_model_folder = rf"/Users/johnwragg/Desktop/DS_work/Hastings_Solution/Model_Experiments/{run_name}"
    os.makedirs(base_model_folder, exist_ok=True)

    # Setup model run
    train = pl.read_parquet(
        "/Users/johnwragg/Desktop/DS_work/Hastings_Solution/Prepped_Data/train_large_loss.parquet"
    )

    features = [col for col in train.columns if col not in DO_NOT_MODEL]

    
    feature_selection_dict = {}
    
    MODEL_TARGETS = {
    "nil": NIL_TARGET,
    "sev": TARGET_COL,
    "large_loss": LARGE_LOSS_TARGET,
}

    for model, target in MODEL_TARGETS.items():
        model_features = run_rfe(train, features, target, model)
        feature_selection_dict[model] = model_features

    
    # save features to model experiment folder
    os.makedirs(base_model_folder, exist_ok=True)
    features_path = os.path.join(base_model_folder, "model_features.json")
    with open(features_path, "w") as f:
        json.dump(feature_selection_dict, f, indent=2)


if __name__ == "__main__":
    main("large_loss")