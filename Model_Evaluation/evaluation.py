import numpy as np
import polars as pl
import pandas as pd
from params.targets import NIL_TARGET, TARGET_COL, LARGE_LOSS_TARGET
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score,
    d2_tweedie_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import os


def shap_plot(mb, data: pl.DataFrame, model_key: str):
    is_sev = model_key == "sev"

    model_selection = {
        "sev": {
            "model" :mb.sev_model, "features": mb.sev_features},
        "nil": {
            "model": mb.nil_model, "features": mb.nil_features},

        "large_loss": {
            "model": mb.large_loss_model, "features": mb.large_loss_features}
    }

    model = model_selection[model_key]["model"]
    features = model_selection[model_key]["features"]
    rows = data.filter(pl.col(NIL_TARGET) == 1) if is_sev else data
    X = rows.select(features).to_pandas()

    explainer = shap.Explainer(model, feature_perturbation="tree_path_dependent")
    shap_values = explainer(X)

    shap.plots.beeswarm(shap_values, show=False)
    fig = plt.gcf()
    fig.suptitle(f"SHAP summary - {'severity' if is_sev else model_key} model")
    return fig


def AvE_by_feature(data, model, target, pred, feature):
    is_sev = model == "sev"
    rows = data.filter(pl.col(NIL_TARGET) == 1) if is_sev else data
    base_plot_data = (
        rows.group_by(feature)
        .agg(
            pl.col(pred).mean(),
            pl.col(target).mean(),
            pl.len().alias("count"),
        )
        .sort(feature)
    )

    return base_plot_data


def plot_ave(res):

    feature = res.columns[0]

    bars = px.histogram(res, x=feature, y="count", color_discrete_sequence=["#c3c2b7"])
    bars.update_traces(name="Claims", showlegend=True)

    lines = px.line(
        res,
        x=feature,
        y=["severity_capped", "Capped_Incurred"],
        markers=True,
        color_discrete_sequence=["#2a78d6", "#eb6834"],
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_traces(bars.data, secondary_ys=[False])
    fig.add_traces(lines.data, secondary_ys=[True, True])

    fig.update_xaxes(type="category", title=feature)
    fig.update_yaxes(title="Claims", showgrid=False, secondary_y=False)
    fig.update_yaxes(title="Mean capped incurred", rangemode="tozero", secondary_y=True)
    fig.update_layout(
        title=f"Actual vs expected by {feature}",
        bargap=0.1,
        legend_title_text="",
    )
    return fig


def gini(actual, pred):
    def area(order):
        return (np.cumsum(actual[order]) / actual.sum()).mean() - 0.5

    return area(np.argsort(-pred, kind="stable")) / area(
        np.argsort(-actual, kind="stable")
    )


def sev_metrics(data):
    rows = data.filter(pl.col(NIL_TARGET) == 1)
    actual = rows[TARGET_COL].to_numpy()
    pred = rows["severity_capped"].to_numpy()
    return {
        "claims": rows.height,
        "gini": gini(actual, pred),
        "gamma_pde": d2_tweedie_score(actual, pred, power=2),
    }


def nil_metrics(data, threshold=0.5):
    actual = data[NIL_TARGET].to_numpy()
    prob = data["p_nonnil"].to_numpy()
    pred = prob >= threshold
    return {
        "claims": data.height,
        "non_nil_rate": actual.mean(),
        "auc": roc_auc_score(actual, prob) if len(set(actual)) == 2 else None,
        "log_loss": log_loss(actual, prob, labels=[0, 1]),
        "accuracy": accuracy_score(actual, pred),
        "precision": precision_score(actual, pred),
        "recall": recall_score(actual, pred),
        "f1": f1_score(actual, pred),
    }


def metrics_table(train, test, metric_fn):
    rows = []
    for name, data in [("train", train), ("test", test)]:
        rows.append({"dataset": name, "loss_year": "all", **metric_fn(data)})
        for year in data["loss_year"].unique().sort():
            metrics = metric_fn(data.filter(pl.col("loss_year") == year))
            rows.append({"dataset": name, "loss_year": str(year), **metrics})
    return pl.DataFrame(rows)


def write_AvE_to_experiments(experiment_dir: str, plots, run_name: str) -> None:

    with open(os.path.join(experiment_dir, "ave_sev.html"), "w") as f:
        f.write(
            f"<html><head><meta charset='utf-8'><title>AvE - severity - {run_name}</title></head>"
            f"<body><h1>Actual vs Expected - severity model ({run_name})</h1>"
            + "".join(f"<div style='margin-bottom:40px'>{div}</div>" for div in plots)
            + "</body></html>"
        )

def youden_threshold(y_true: np.ndarray, p: np.ndarray) -> float:
    """Threshold for confusion matrix which is max distance between roc and diagnoal."""
    fpr, tpr, thr = roc_curve(y_true, p)
    return float(thr[np.argmax(tpr - fpr)])


def ave_summary(data: pl.DataFrame) -> pl.DataFrame:
    """Return table for Actual vs expected totals"""
    pos = data.filter(pl.col(NIL_TARGET) == 1)
    excess = data["Incurred"] - data[TARGET_COL]

    summary = {
        "non-nil claims": (data[NIL_TARGET].sum(), data["p_nonnil"].sum()),
        "severity (non-nil claims)": (pos[TARGET_COL].sum(), pos["severity_capped"].sum()),
        "capped cost": (data[TARGET_COL].sum(), data["expected_capped"].sum()),
        "large claims": (data[LARGE_LOSS_TARGET].sum(), data["p_large_loss"].sum()),
        "excess over cap": (excess.sum(), data["expected_excess"].sum()),
        "total incurred": (data["Incurred"].sum(), data["expected_cost"].sum()),
        "total incurred (flat load)": (data["Incurred"].sum(), data["expected_cost_flat_load"].sum()),
    }

    return pl.DataFrame(
        {
            "measure": list(summary.keys()),
            "actual": [float(a) for a, _ in summary.values()],
            "expected": [float(e) for _, e in summary.values()],
        }
    ).with_columns(a_over_e=pl.col("actual") / pl.col("expected"))


def ave_by_group(data: pl.DataFrame, group: str, target: str = "Incurred", pred: str = "expected_cost") -> pl.DataFrame:
    """Return grouped actual vs expected by each level in a col."""
    return (
        data.group_by(group)
        .agg(
            pl.len().alias("claims"),
            pl.col(target).sum().alias("actual"),
            pl.col(pred).sum().alias("expected"),
        )
        .with_columns(a_over_e=pl.col("actual") / pl.col("expected"))
        .sort(group)
    )

def expected_ultimate_loss(
    mb,
    data: pl.DataFrame,
    rebase_year: int | None = None,
    inflation_factor: float = 0.0,
) -> pl.DataFrame:
    """Expected ultimate loss per claim. Caluclated as expected_capped + expected_excess.

    rebase_year can be set to score every claim at a set level for live scoring.
    """
    preds = mb.predict(data, rebase_year=rebase_year)
    eul_data = pl.concat([data.select("Claim_Number", "loss_year", "Incurred"), preds], how="horizontal")

    return eul_data.with_columns(eul=pl.col("expected_cost") * (1 + inflation_factor))


def eul_by_year(data: pl.DataFrame) -> pl.DataFrame:
    """EUL compared to actual loss by year."""
    by_year = ave_by_group(data, "loss_year", target="Incurred", pred="eul").with_columns(
        pl.col("loss_year").cast(pl.String)
    )

    # Get overall totals for comparison across all data
    total = pl.DataFrame(
        {
            "loss_year": ["total"],
            "claims": [data.height],
            "actual": [data["Incurred"].sum()],
            "expected": [data["eul"].sum()],
        },
        schema=by_year.drop("a_over_e").schema,
    ).with_columns(a_over_e=pl.col("actual") / pl.col("expected"))
    return pl.concat([by_year, total])
