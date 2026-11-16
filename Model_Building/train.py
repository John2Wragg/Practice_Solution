from Model_Building.rfe_cv import run_rfe
from Model_Building.ModelBuilder import ModelBuilder
from params.generic_params import DO_NOT_MODEL
from params.targets import NIL_TARGET, TARGET_COL
import polars as pl
import os
import json
from functions.mlflow_helpers import start_stage_run, log_params_dict
import mlflow


def main(run_name="test_run"):

    with start_stage_run(run_name, "train"):

        base_model_folder = rf"/Users/johnwragg/Desktop/DS_work/Hastings_Solution/Model_Experiments/{run_name}"
        os.makedirs(base_model_folder, exist_ok=True)

        # Setup model run
        train = pl.read_parquet(
            "/Users/johnwragg/Desktop/DS_work/Hastings_Solution/Prepped_Data/train_large_loss.parquet"
        )

        features = [col for col in train.columns if col not in DO_NOT_MODEL]

        feature_selection_path = os.path.join(base_model_folder, "model_features.json")
        if not os.path.exists(feature_selection_path):
            raise FileNotFoundError(
                f"No feature selection found at {feature_selection_path}. Run run_rfe_cv.py for '{run_name}' first."
            )
        with open(feature_selection_path) as f:
            feature_selection_dict = json.load(f)
            nil_features = feature_selection_dict["nil"]
            sev_features = feature_selection_dict["sev"]
            large_loss_features = feature_selection_dict["large_loss"]

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
        large_loss_params = best_params["large_loss"]

        # Run modelbuilder
        mb = ModelBuilder(
            nil_features=nil_features,
            nil_params=nil_params,
            sev_features=sev_features,
            sev_params=sev_params,
            large_loss_features=large_loss_features,
            large_loss_params=large_loss_params
        )

        log_params_dict(nil_params, prefix="nil.")
        log_params_dict(sev_params, prefix="sev.")
        log_params_dict(large_loss_params, prefix="large_loss.")
        mlflow.log_param("n_train_claims", train.height)

        mb.fit(train)

        # ML Flow saving metrics and model artifact
        model_path = os.path.join(base_model_folder, f"{run_name}.joblib")

        mlflow.log_metrics({
            "sev_balance": mb.sev_balance if mb.sev_balance else -1,
            "mean_excess": mb.mean_excess if mb.mean_excess else -1,
            "large_loss_load": mb.large_loss_load if mb.large_loss_load else -1,
        })

        mb.save(model_path)
        mlflow.log_artifact(model_path)


if __name__ == "__main__":
    main(run_name="mlflow_v1")
