# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## Environment

- WSL2 on Windows; Windows drives accessible at `/mnt/c/` and `/mnt/d/`
- Python 3.12 venv at `~/study/` (activate with `source ~/study/bin/activate`)
- Airflow home: `/mnt/c/Users/user/Documents/airflow`; DB: PostgreSQL at `localhost/airflow_db`

## Tech. stack

- Programming: Python, C++
- Databases: ClickHouse primarily, a bit of PostgreSQL
- Machine Learning: PyTorch, Scikit-Learn, HuggingFace, and all the adjacent libraries & frameworks (polars, numpy, etc.)
- Opsing: Airflow, MLflow, MinIO  
- OS: Linux (Ubuntu)

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Project: Airflow deployment

This is a shared Airflow 3.0 deployment (`apache/airflow:3.0.2`, LocalExecutor,
Docker Compose). The DAGs come from other projects and run here, not in their
home repos.

### DAGs and where they come from

- `dags/csfloat_ingest_dag.py` (`csfloat_ingest`, daily) comes from
  `/home/aidar/projects/item_factor_est`. That repo keeps an identical copy in
  its own `dags/`, synced by hand, and the copy there is not deployed.
- `dags/csfloat_model_retrain_dag.py` (`csfloat_model_retrain`, weekly)
  retrains Model C, ported from `item_factor_est/notebooks/03_train.ipynb`. The
  logic is in `src/training/train.py:train_and_gate`, which has no Airflow
  imports. The DAG is a `@task.short_circuit` followed by a Gmail `notify` task
  that runs only when a new champion is promoted.
- `dags/cs2_price_tracker.py` (`report_price_changes`, daily) comes from
  `/home/aidar/projects/price_dynamics`.

### Things that silently diverge if you forget

- `src/features/build_features.py` and `src/features/temporal_split.py` are
  **copies** of the files in `item_factor_est/src/features/`. Keep them in sync.
- `src/training/eval.py` mirrors `item_factor_est/src/models/eval.py`.
- `mlflow-skinny` in `requirements.txt` must match `mlflow==` in
  `../mlflow/Dockerfile`.
- `numpy` is pinned to the base image's version so that the extra deps don't
  upgrade it under Airflow.

### Runtime wiring

- `./src` is mounted at `/opt/airflow/dags/src`, `PYTHONPATH=/opt/airflow/dags`,
  and `dags/.airflowignore` excludes `src/` from DAG parsing. Imports look like
  `from src.training.train import ...`.
- Config comes from Airflow Variables (`provision_variables.sh` sets them from
  `.env`), except for ClickHouse credentials (`CLICKHOUSE_USER` /
  `CLICKHOUSE_PASSWORD`) and `MLFLOW_TRACKING_URI`, which are container env vars
  set in `docker-compose.yaml`.
- The retrain DAG also reads the optional Variable `retrain_n_trials` (default 40).
- External networks: `item_factor_est_default`, `airflow-docker_default`, and
  `mlflow_default`.

### Model C retrain facts

- Target: `target_c = log(price / ref_predicted_price)`, CatBoost, temporal split.
- The training slice is bounded by `_ingested_at <= data_interval_end`, not by
  `sold_at`. The SQL is logged as an artifact alongside the row count and a
  content hash.
- The gate promotes a model only on strictly lower test RMSE(log) than the
  current `champion` of `csfloat_model_c`.
- A run takes about 2 hours (40 trials, about 3 minutes per fit on about 900k
  rows). There are no retries, by design.

### Tests

`pytest tests/`. `test_train_gate.py` runs the real gate on synthetic data
against a temporary MLflow store, with no network access.

### Conventions

- Secrets go in `.env` (gitignored), and `.env.example` is committed.
- Match the existing style: TaskFlow API (`airflow.sdk`), polars for data work,
  and `logging` rather than print.
