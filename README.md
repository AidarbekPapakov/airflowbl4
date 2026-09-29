# Airflow

A single Airflow 3 deployment (LocalExecutor, Docker Compose) that runs the
DAGs for several CS2 market projects.

## DAGs

| DAG                     | Schedule | What it does                                                                                                                  | Origin project    |
|-------------------------|----------|-------------------------------------------------------------------------------------------------------------------------------|-------------------|
| `csfloat_ingest`        | daily    | Pulls CSFloat sales history for a rotating batch of skins (`config/batch_*_skins.yaml`) into ClickHouse `csfloat_sales`       | `item_factor_est` |
| `csfloat_model_retrain` | weekly   | Runs Optuna HPO on CatBoost Model C, logs to MLflow, and promotes the model to `champion` only if it beats the current one. Sends an email on promotion | `item_factor_est` |
| `report_price_changes`  | daily    | Prices a Steam inventory from CSFloat sales and listings, stores the observations, and emails an alert on large price moves   | `price_dynamics`  |

## Layout

```
dags/        DAG files (mounted at /opt/airflow/dags)
src/         shared code, mounted at /opt/airflow/dags/src and ignored by the DAG parser
  ingest/    CSFloat client, pydantic schemas, ClickHouse loader
  features/  feature engineering + temporal split (copied from item_factor_est)
  training/  Model C data slice, training/gating, and eval metrics
  model/     CsItem, the item identity used by report_price_changes
config/      skin batch YAMLs for csfloat_ingest
tests/       pytest
```

## External dependencies

The Compose file joins three external networks, and each one must already exist:

- `item_factor_est_default`: ClickHouse for the CSFloat ingest and retrain DAGs
- `airflow-docker_default`: ClickHouse for `report_price_changes`
- `mlflow_default`: the MLflow server at `http://mlflow:5000` (see `../mlflow`)

## Setup

```bash
cp .env.example .env            # fill in the secrets
docker compose up -d --build
./provision_variables.sh        # push Airflow Variables from .env
```

The UI is at http://localhost:8080. DAGs start paused.

## Tests

```bash
pytest tests/
```
