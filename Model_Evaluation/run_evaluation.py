import polars as pl
from Model_Building.ModelBuilder import ModelBuilder
import plotly.express as px
from Model_Evaluation.evaluation import (
    gini,
    shap_plot,
    metrics_table,
    nil_metrics,
    sev_metrics,
    AvE_by_feature,
    plot_ave,
    write_AvE_to_experiments,
    youden_threshold,
    ave_summary,
    ave_by_group,
    expected_ultimate_loss,
    eul_by_year,
    large_loss_approach_gini_metrics,
)
from params.targets import TARGET_COL, LARGE_LOSS_TARGET, NIL_TARGET
from params.generic_params import REBASE_YEAR, CLAIMS_INFLATION, TARGET_YEAR
import os
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def main(run_name: str):

    base_model_folder = rf"/Users/johnwragg/Desktop/DS_work/Hastings_Solution/Model_Experiments/{run_name}"
    shap_folder = os.path.join(base_model_folder, "shap")
    ave_sev_folder = os.path.join(base_model_folder, "AvE", "sev")
    ave_total_folder = os.path.join(base_model_folder, "AvE", "totals")
    for folder in [base_model_folder, shap_folder, ave_sev_folder, ave_total_folder]:
        os.makedirs(folder, exist_ok=True)

    mb = ModelBuilder.load(os.path.join(base_model_folder, f"{run_name}.joblib"))
    # Test predictions
    train = pl.read_parquet(
        r"/Users/johnwragg/Desktop/DS_work/Hastings_Solution/Prepped_Data/train_large_loss.parquet"
    )
    test = pl.read_parquet(
        r"/Users/johnwragg/Desktop/DS_work/Hastings_Solution/Prepped_Data/test_large_loss.parquet"
    )

    # Setting up ave dataframes
    preds_test = mb.predict(test)
    preds_train = mb.predict(train)
    comparison_test = pl.concat([test, preds_test], how="horizontal")
    comparison_train = pl.concat([train, preds_train], how="horizontal")

    # SHAP plots
    for model in ["nil", "sev", "large_loss"]:
        fig = shap_plot(mb, train, model_key=model)
        fig.savefig(
            os.path.join(shap_folder, f"shap_{model}.png"), dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    # AvE by feature (severity model) -> AvE/sev/<feature>.csv + one combined AvE/sev/ave_sev.html
    plot_divs = []
    for i, feature in enumerate(mb.sev_features):
        ave_sev_feature = AvE_by_feature(
            comparison_test,
            "sev",
            target=TARGET_COL,
            pred="severity_capped",
            feature=feature,
        )
        ave_sev_feature.write_csv(os.path.join(ave_sev_folder, f"{feature}.csv"))
        # Embed plotly.js only once (with the first plot) so the file stays small and works offline
        plot_divs.append(
            plot_ave(ave_sev_feature).to_html(
                full_html=False, include_plotlyjs=(i == 0)
            )
        )

    write_AvE_to_experiments(ave_sev_folder, plot_divs, run_name)

    # Confusion matrix
    thresh_nil = youden_threshold(comparison_train[NIL_TARGET], comparison_train["p_nonnil"])

    comparison_test = comparison_test.with_columns(pl.when(pl.col("p_nonnil") > thresh_nil).then(1).otherwise(0).alias("non_zero_claim_label_pred"))
    cm = confusion_matrix(comparison_test[NIL_TARGET], comparison_test["non_zero_claim_label_pred"], labels=mb.nil_model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=mb.nil_model.classes_)
    
    conf_matrix_plot = os.path.join(base_model_folder, "confusion_matrix.png")


    # Confusion matrix for large loss
    thresh_large_loss = youden_threshold(comparison_train[LARGE_LOSS_TARGET], comparison_train["p_large_loss"])
    comparison_test = comparison_test.with_columns(pl.when(pl.col("p_large_loss") > thresh_large_loss).then(1).otherwise(0).alias("large_loss_label_pred"))
    cm_large_loss = confusion_matrix(comparison_test[LARGE_LOSS_TARGET], comparison_test["large_loss_label_pred"], labels=mb.large_loss_model.classes_)
    disp_large_loss = ConfusionMatrixDisplay(confusion_matrix=cm_large_loss,
                                display_labels=mb.large_loss_model.classes_)
    
    conf_matrix_plot_large_loss = os.path.join(base_model_folder, "confusion_matrix_large_loss.png")

    # has to be called before saving to generate plot
    disp.plot()

    disp.figure_.savefig(conf_matrix_plot)

    disp_large_loss.plot()
    disp_large_loss.figure_.savefig(conf_matrix_plot_large_loss)

    # overall metrics
    sev_metrics_dict = sev_metrics(comparison_test)
    nil_metrics_dict = nil_metrics(comparison_test, threshold=thresh_nil)
    large_loss_metrics_dict = large_loss_approach_gini_metrics(comparison_test)
    for name, metrics in [
        ("sev_metrics", sev_metrics_dict),
        ("nil_metrics", nil_metrics_dict),
        ("large_loss_metrics", large_loss_metrics_dict), 
    ]:
        with open(os.path.join(base_model_folder, f"{name}.json"), "w") as f:
            json.dump(metrics, f, indent=2, default=float)

    # Metrics by year sample
    metrics_table_nil = metrics_table(comparison_train, comparison_test, nil_metrics)
    metrics_table_sev = metrics_table(comparison_train, comparison_test, sev_metrics)
    for name, df in zip(
        ["nil_metrics_df", "sev_metrics_df"], [metrics_table_nil, metrics_table_sev]
    ):
        df.write_csv(os.path.join(base_model_folder, f"{name}.csv"))

    # AvE for all data
    for sample, data in [("train", comparison_train), ("test", comparison_test)]:
        ave_summary(data).write_csv(os.path.join(ave_total_folder, f"summary_{sample}.csv"))

    # Generate AvE by loss years
    ave_by_group(comparison_test, "loss_year").write_csv(os.path.join(ave_total_folder, "loss_year_test.csv"))

    # EUL summry 
    eul_test = expected_ultimate_loss(mb, test)
    eul_by_year(eul_test).write_csv(os.path.join(base_model_folder, "eul_test.csv"))

if __name__ == "__main__":
    main("large_loss")  # hardcoded for now but allow argparse
