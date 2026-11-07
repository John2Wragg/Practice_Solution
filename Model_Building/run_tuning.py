
import optuna
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier, LGBMRegressor
import polars as pl 
from params.targets import NIL_TARGET, TARGET_COL
from params.generic_params import DO_NOT_MODEL
from Model_Building.rfe_cv import run_rfe
import lightgbm as lgb
import os
import json

#Source: https://machinelearningmastery.com/how-to-perform-scikit-learn-hyperparameter-optimization-with-optuna/

def main(run_name):

    base_model_folder = rf"/Users/johnwragg/Desktop/DS_work/Hastings_Solution/Model_Experiments/{run_name}"
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


    # For nil model
    nil_features = feature_selection_dict["nil"]

    y_train_nil = train[NIL_TARGET].to_numpy()
    X_train_nil = train.select(nil_features).to_pandas()

    year_idx_nil = nil_features.index("loss_year")
    others_nil = [i for i in range(len(nil_features)) if i != year_idx_nil]
    nil_interaction_constraints = [[year_idx_nil], others_nil]  # year in own interaction group


    def objective_nil(trial):
        n_estimators = trial.suggest_int("n_estimators", 10, 200)
        max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
        min_child_samples = trial.suggest_int("min_child_samples", 5, 100)

        clf = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            random_state=42,
            interaction_constraints=nil_interaction_constraints,
            objective="binary",
            verbose=-1,
        )

        score = cross_val_score(clf, X_train_nil, y_train_nil, cv=3, scoring="accuracy").mean()
        return score


    study_nil = optuna.create_study(direction="maximize")
    study_nil.optimize(objective_nil, n_trials=5)

    print("Best hyperparameters:", study_nil.best_params)
    print("Best accuracy:", study_nil.best_value)
    # For sev model

    sev_features = feature_selection_dict["sev"]
    train_sev = train.filter(pl.col(NIL_TARGET) == 1)

    y_train_sev = train_sev[TARGET_COL].to_numpy()
    X_train_sev = train_sev.select(sev_features).to_pandas()

    year_idx = sev_features.index("loss_year")
    others = [i for i in range(len(sev_features)) if i != year_idx]
    sev_interaction_constraints = [[year_idx], others]  # year in own interaction group


    def objective_sev(trial):
        n_estimators = trial.suggest_int("n_estimators", 10, 200)
        max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
        min_child_samples = trial.suggest_int("min_child_samples", 5, 100)

        reg = LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            random_state=42,
            interaction_constraints=sev_interaction_constraints,
            objective="gamma",
            verbose=-1,
        )

        # accuracy is a classification metric; use gamma deviance for a gamma regressor
        # (sklearn returns it negated, so higher is better -> direction="maximize")
        score = cross_val_score(
            reg, X_train_sev, y_train_sev, cv=3, scoring="neg_mean_gamma_deviance"
        ).mean()
        return score


    study_sev = optuna.create_study(direction="maximize")
    study_sev.optimize(objective_sev, n_trials=5)

    print("Best hyperparameters:", study_sev.best_params)
    print("Best neg gamma deviance:", study_sev.best_value)

    # save outputs to model experiment folder
    os.makedirs(base_model_folder, exist_ok=True)
    best_params = {
        "nil": study_nil.best_params,
        "sev": study_sev.best_params,
    }

    params_path = os.path.join(base_model_folder, "best_params.json")
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    
    print(f"Saved best params to {params_path}")
