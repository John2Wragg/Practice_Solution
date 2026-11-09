
import optuna
from lightgbm import LGBMClassifier, LGBMRegressor
import polars as pl 
from params.targets import NIL_TARGET, TARGET_COL, LARGE_LOSS_TARGET
from params.generic_params import DO_NOT_MODEL
from Model_Building.rfe_cv import run_rfe
import lightgbm as lgb
import os
import json
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold


#Source: https://machinelearningmastery.com/how-to-perform-scikit-learn-hyperparameter-optimization-with-optuna/

def main(run_name):

    base_model_folder = rf"/Users/johnwragg/Desktop/DS_work/Hastings_Solution/Model_Experiments/{run_name}"
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

    # for nil model

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
        # nil
        cv_nil = StratifiedKFold(5, shuffle=True, random_state=42)
        score = cross_val_score(clf, X_train_nil, y_train_nil, cv=cv_nil, scoring="roc_auc").mean()
        return score


    study_nil = optuna.create_study(direction="maximize")
    study_nil.optimize(objective_nil, n_trials=5)

    print("Best hyperparameters:", study_nil.best_params)
    print("Best accuracy:", study_nil.best_value)
    
    # For sev model
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


        # sev
        cv_sev = KFold(5, shuffle=True, random_state=42)
        score = cross_val_score(
            reg, X_train_sev, y_train_sev, cv=cv_sev, scoring="neg_mean_gamma_deviance"
        ).mean()
        return score


    study_sev = optuna.create_study(direction="maximize")
    study_sev.optimize(objective_sev, n_trials=5)

    print("Best hyperparameters:", study_sev.best_params)
    print("Best neg gamma deviance:", study_sev.best_value)

    # For large loss
    
    y_train_large_loss = train[LARGE_LOSS_TARGET].to_numpy()
    X_train_large_loss = train.select(large_loss_features).to_pandas()

    year_idx = large_loss_features.index("loss_year")
    others = [i for i in range(len(large_loss_features)) if i != year_idx]
    large_loss_interaction_constraints = [[year_idx], others]  # year in own interaction group

    def objective_large_loss(trial):
        
        n_estimators = trial.suggest_int("n_estimators", 10, 200)
        max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
        min_child_samples = trial.suggest_int("min_child_samples", 5, 100)

        clf = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            random_state=42,
            interaction_constraints=large_loss_interaction_constraints,
            objective="binary",
            verbose=-1,
        )
        # nil
        cv_large_loss = StratifiedKFold(5, shuffle=True, random_state=42)
        score = cross_val_score(clf, X_train_large_loss, y_train_large_loss, cv=cv_large_loss, scoring="roc_auc").mean()
        return score


    study_large_loss = optuna.create_study(direction="maximize")
    study_large_loss.optimize(objective_nil, n_trials=5)

    print("Best hyperparameters:", study_large_loss.best_params)
    print("Best accuracy:", study_large_loss.best_value)


    # save outputs to model experiment folder
    os.makedirs(base_model_folder, exist_ok=True)
    best_params = {
        "nil": study_nil.best_params,
        "sev": study_sev.best_params,
        "large_loss": study_large_loss.best_params
    }

    params_path = os.path.join(base_model_folder, "best_params.json")
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    
    print(f"Saved best params to {params_path}")

if __name__ == "__main__":
    main(run_name="large_loss")
