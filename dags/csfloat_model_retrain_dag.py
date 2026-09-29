"""
Weekly Model C retrain with a champion/challenger gate.

Task flow:
  retrain_and_evaluate (short_circuit) → notify

retrain_and_evaluate pulls the csfloat_sales slice ingested up to data_interval_end,
runs Optuna HPO on CatBoost (every trial logged to MLflow), scores the new model and
the registered champion on the same temporal test slice, and registers the new model
(with the slice's SQL as an artifact) only if it has strictly lower test RMSE(log).
It returns that decision, so notify (Gmail) runs only when a new champion was published.
"""

import logging
import os
from datetime import datetime, timedelta
from html import escape
from typing import Any, Dict

import clickhouse_connect
from airflow.sdk import Variable, dag, task
from airflow.utils.email import send_email

from src.training.data import load_frame
from src.training.train import MODEL_NAME, train_and_gate

log = logging.getLogger(__name__)

# Host-facing MLflow UI (MLFLOW_TRACKING_URI is the in-network address, useless in an email)
MLFLOW_UI_URL = "http://localhost:5000"


def _fmt(value: Any, spec: str) -> str:
    return "&mdash;" if value is None else format(value, spec)


def _render_report_html(r: Dict[str, Any]) -> str:
    rows = [
        ("New model (challenger)", r["challenger_rmse_log"], r["challenger_mdape"]),
        (f"Previous champion (v{r['champion_version']})" if r["champion_version"] else "Previous champion (none)",
         r["champion_rmse_log"], r["champion_mdape"]),
        ("Trust-CSFloat baseline (predict 0)", r["baseline_rmse_log"], r["baseline_mdape"]),
    ]
    body = "".join(
        f"<tr><td>{escape(name)}</td><td>{_fmt(rmse, '.4f')}</td><td>{_fmt(mdape, '.2%')}</td></tr>"
        for name, rmse, mdape in rows
    )
    run_url = f"{MLFLOW_UI_URL}/#/experiments/{r['experiment_id']}/runs/{r['run_id']}"
    return (
        f"<h3>{MODEL_NAME} v{r['new_version']} promoted to champion</h3>"
        f"<p>Test slice: {r['n_test']:,} sales, sold_at {escape(r['test_sold_at_range'])} "
        f"&middot; trained on {r['n_train']:,} rows ingested up to {escape(r['cutoff'])}</p>"
        '<table border="1" cellpadding="4" style="border-collapse:collapse;font-size:13px">'
        "<thead><tr><th>Model</th><th>RMSE (log)</th><th>MDAPE</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
        f"<p>Best params: <code>{escape(str(r['best_params']))}</code></p>"
        f'<p><a href="{run_url}">MLflow run</a></p>'
    )


@dag(
    dag_id="csfloat_model_retrain",
    schedule="@weekly",
    start_date=datetime(2026, 9, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "Aidar"},
    tags=["csfloat", "mlflow", "training"],
)
def csfloat_model_retrain_dag():

    # No retries: a failed HPO run should be looked at, not silently re-run for hours.
    # ~3 min per CatBoost fit on ~900k rows -> 40 trials + refit is ~2h+.
    @task.short_circuit(retries=0, execution_timeout=timedelta(hours=6))
    def retrain_and_evaluate(**context) -> bool:
        cutoff: datetime = context["data_interval_end"]
        n_trials = int(Variable.get("retrain_n_trials", "40"))
        database = Variable.get("CH_DB_KEY")

        client = clickhouse_connect.get_client(
            host=Variable.get("CH_HOST_KEY"),
            port=int(Variable.get("CH_PORT_KEY")),
            database=database,
            username=os.environ.get("CLICKHOUSE_USER"),
            password=os.environ.get("CLICKHOUSE_PASSWORD"),
        )
        sql, df = load_frame(client, database, cutoff)
        log.info("Loaded %d feature rows (ingested <= %s)", len(df), cutoff)

        report = train_and_gate(df, sql, cutoff, n_trials)
        log.info("Gate: challenger %.4f vs champion %s -> promoted=%s",
                 report["challenger_rmse_log"], report["champion_rmse_log"], report["promoted"])
        context["ti"].xcom_push(key="report", value=report)
        return report["promoted"]

    @task
    def notify(**context) -> None:
        report = context["ti"].xcom_pull(task_ids="retrain_and_evaluate", key="report")
        send_email(
            to=[Variable.get("recipients_email")],
            subject=f"{MODEL_NAME} v{report['new_version']} promoted — {report['cutoff'][:10]}",
            html_content=_render_report_html(report),
        )

    retrain_and_evaluate() >> notify()


csfloat_model_retrain_dag()
