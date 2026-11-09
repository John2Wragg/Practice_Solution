import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from typing import List, Dict

from params.targets import NIL_TARGET, TARGET_COL, LARGE_LOSS_TARGET


class ModelBuilder:
    """Trains both parts of the hurdle model and combines them.

    Part A (nil model): P(claim > 0), binary objective, fitted on all claims.
    Part B (severity):   E[capped cost | claim > 0], gamma objective, fitted on positive claims.
    Expected cost = P(claim > 0) * E[capped cost | > 0] * large-loss load.
    """

    def __init__(
        self,
        nil_features: list[str],
        nil_params: dict,
        sev_features: list[str],
        sev_params: dict,
        large_loss_features: list[str],
        large_loss_params: dict,
        year_col: str = "loss_year",
        monotone_increasing: list[str] | None = None,
        seed: int = 42,
    ):
        self.nil_features = nil_features
        self.nil_params = nil_params
        self.sev_features = sev_features
        self.sev_params = sev_params
        self.large_loss_features = large_loss_features
        self.large_loss_params = large_loss_params
        self.year_col = year_col
        self.monotone_increasing = set(monotone_increasing or [])
        self.seed = seed

        self.nil_model: lgb.LGBMClassifier | None = None
        self.sev_model: lgb.LGBMRegressor | None = None
        self.large_loss_load: float | None = None
        self.sev_balance: float | None = None

    # ---------- helpers ----------
    def _X(self, data: pl.DataFrame, features: List[str]) -> pd.DataFrame:
        return data.select(features).to_pandas()

    def _interaction_constraints(self, features: List[str]) -> list[list[int]]:
        """Year in its own group to disentangle interactions."""
        if self.year_col not in features:
            return []
        year_idx = features.index(self.year_col)
        others = [i for i in range(len(features)) if i != year_idx]
        return [[year_idx], others]

    def _monotone_constraints(self, features: List[str]) -> list[int]:
        return [1 if f in self.monotone_increasing else 0 for f in features]


    def fit_classifier(self, data: pl.DataFrame, target: str, constraints: List[list[int]], features: List[str], params: Dict[str, int]) -> lgb.LGBMClassifier:

        y_class = data[target].to_numpy()
        model = lgb.LGBMClassifier(
            objective="binary",
            interaction_constraints=constraints,
            random_state=self.seed,
            verbose=-1,
            subsample_freq=1,
            **params,
        ).fit(self._X(data, features), y_class)

        return model

    def fit(self, train: pl.DataFrame) -> "ModelBuilder":
        constraints_nil = self._interaction_constraints(self.nil_features)

        # TODO: Make this a fit_classifier() method so I can recall this to fit and make a larg loss classifier
        # Part A: nil model on all training claims

        self.nil_model = self.fit_classifier(train, target=NIL_TARGET, constraints=constraints_nil, features=self.nil_features, params=self.nil_params)


        # Part B: severity model on positive training claims, capped target
        # TODO: make this a fit regressor() so I can fit a large loss model to this as well. 
        pos = train.filter(pl.col(NIL_TARGET) == 1)
        constraints_sev = self._interaction_constraints(self.sev_features)
        self.sev_model = lgb.LGBMRegressor(
            objective="gamma",
            interaction_constraints=constraints_sev,
            monotone_constraints=self._monotone_constraints(self.sev_features),
            random_state=self.seed,
            verbose=-1,
            subsample_freq=1,
            **self.sev_params,
        ).fit(
            self._X(pos, self.sev_features),
            pos[TARGET_COL].to_numpy(),
        )

        # saving rebasing predictions factor
        X_pos = self._X(pos, self.sev_features)
        self.sev_balance = pos[TARGET_COL].sum() / self.sev_model.predict(X_pos).sum()

        # Large-loss load from training data only
        self.large_loss_load = pos["Incurred"].mean() / pos[TARGET_COL].mean()
        
        # Fit large loss classifier
        constraints_large_loss = self._interaction_constraints(self.large_loss_features)
        self.large_loss_model = self.fit_classifier(train, target=LARGE_LOSS_TARGET, constraints=constraints_large_loss, features=self.large_loss_features, params=self.large_loss_params)
        
        return self

    # ---------- prediction ----------
    def predict(
        self, data: pl.DataFrame, rebase_year: int | None = None
    ) -> pl.DataFrame:
        """Return p_nonnil, severity, expected_capped and expected_cost for each claim.

        rebase_year=None scores each claim at its own year (use for evaluation);
        rebase_year=2015 scores everything at that year's level (use for deployment).
        """
        if self.nil_model is None:
            raise ValueError("self.nil_model is not defined. Require fitting.")

        # Set all loss year rows to rebase year to mimic live implementation / scoring
        if rebase_year is not None:
            data = data.with_columns(
                pl.lit(rebase_year).cast(data[self.year_col].dtype).alias(self.year_col)
            )

        X_nil = self._X(data, self.nil_features)
        X_sev = self._X(data, self.sev_features)
        X_large_loss = self._X(data, self.large_loss_features)
        p = self.nil_model.predict_proba(X_nil)[:, 1]
        sev = self.sev_model.predict(X_sev) * self.sev_balance
        p_large_loss = self.large_loss_model.predict_proba(X_large_loss)[:, 1]

        return pl.DataFrame(
            {
                "p_nonnil": p,
                "severity_capped": sev,
                "expected_capped": p * sev,
                "expected_cost": p * sev * self.large_loss_load,
                "p_large_loss": p_large_loss,
            }
        )

    # ---------- persistence ----------
    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "ModelBuilder":
        return joblib.load(path)
