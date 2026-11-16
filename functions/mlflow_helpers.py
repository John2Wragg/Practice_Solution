import mlflow

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "hastings_eul"


def start_stage_run(run_name: str, stage: str):
    """Start an MLflow run for one pipeline stage, nested under a parent run for run_name."""
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)

    # find the parent run for this run_name, or create it the first time
    parents = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.pipeline_run = '{run_name}' and tags.stage = 'parent'",
        output_format="list",
    )
    if parents:
        parent_id = parents[0].info.run_id
    else:
        with mlflow.start_run(run_name=run_name, tags={"pipeline_run": run_name, "stage": "parent"}) as parent:
            parent_id = parent.info.run_id

    return mlflow.start_run(
        run_name=f"{run_name}_{stage}",
        parent_run_id=parent_id,
        tags={"pipeline_run": run_name, "stage": stage},
    )


def log_metrics_dict(metrics: dict, prefix: str = "") -> None:
    """Log a dict of metrics, skipping anything that isn't a number."""
    clean = {
        f"{prefix}{k}": float(v)
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    mlflow.log_metrics(clean)


def log_params_dict(params: dict, prefix: str = "") -> None:
    mlflow.log_params({f"{prefix}{k}": v for k, v in params.items()})