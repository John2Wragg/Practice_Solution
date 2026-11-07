from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import polars as pl
from sklearn.model_selection import KFold
from typing import List
from params.targets import NIL_TARGET


# Source: https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html
def run_rfe(data: pl.DataFrame, features: List[str], target: str, model_type: str):
    min_features_to_select = 1  # Minimum number of features to consider

    if model_type == "nil":
        cv = StratifiedKFold(5, shuffle=True, random_state=42)
        estimator = LGBMClassifier(verbose=-1, random_state=42, objective="binary")
    else:
        cv = KFold(5, shuffle=True, random_state=42)
        estimator = LGBMRegressor(objective="gamma", verbose=-1, random_state=42)
        data = data.filter(pl.col(NIL_TARGET) == 1)

    X_train = (
        data.select(features)
        .with_columns(pl.col(pl.Categorical).to_physical().cast(pl.Int32))
        .to_pandas()
    )

    y_train = data[target].to_numpy()

    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=cv,
        scoring="roc_auc" if model_type == "nil" else "neg_mean_gamma_deviance",
        min_features_to_select=min_features_to_select,
        n_jobs=2,
    )
    rfecv.fit(X_train, y_train)

    print(f"Optimal number of features: {rfecv.n_features_}")
    selected = list(rfecv.get_feature_names_out())
    return selected
