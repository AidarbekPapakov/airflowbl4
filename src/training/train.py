"""
Model C (CatBoost on target_c = log(price / ref_predicted_price)) retraining with
Optuna HPO, MLflow tracking and a champion/challenger gate.

Ported from item_factor_est/notebooks/03_train.ipynb (features, search space,
early-stopping split). No Airflow imports: the DAG calls `train_and_gate`.
"""

import os
from datetime import datetime
from typing import Any

import catboost as cb
import mlflow
import numpy as np
import optuna
import polars as pl
from matplotlib.figure import Figure
from mlflow import MlflowClient
from mlflow.models import infer_signature

from src.features.temporal_split import temporal_split
from src.training.data import frame_hash
from src.training.eval import mdape_backtransformed, rmse_log

EXPERIMENT_NAME = "csfloat_model_c"
MODEL_NAME = "csfloat_model_c"
CHAMPION_ALIAS = "champion"
TARGET = "target_c"
SEED = 42

NUMERIC_FEATURES = [
    'float_value', 'float_position_in_bucket',
    'n_stickers', 'total_sticker_value', 'sticker_to_base_ratio',
    'fade_pct', 'fade_rank',
    'blue_gem_playside',
    'keychain_ref_price',
    'is_stattrak', 'is_souvenir',
]
CAT_FEATURES = ['def_index', 'wear_name', 'rarity', 'fade_type']
ALL_FEATURES = NUMERIC_FEATURES + CAT_FEATURES

# Leave half the cores to the other DAGs sharing the LocalExecutor
THREAD_COUNT = max(1, (os.cpu_count() or 2) // 2)
BASE_PARAMS = {
    'iterations': 2000,
    'loss_function': 'RMSE',
    'early_stopping_rounds': 50,
    'random_seed': SEED,
    'thread_count': THREAD_COUNT,
    'allow_writing_files': False,
    'verbose': False,
}


def split(df: pl.DataFrame, min_test_start: datetime | None = None) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Temporal train / val / test split.

    Test is the newest 20% by `sold_at`, but never starts before `min_test_start`
    (the previous champion's test start): backfilled old sales can pull the 80%
    quantile earlier, which would put rows the champion trained on into the test set.
    Val is the last 10% of train, used only for early stopping (as in the notebook).
    """
    train, test = temporal_split(df, date_col='sold_at', test_frac=0.2)
    if min_test_start is not None:
        train = pl.concat([train, test.filter(pl.col('sold_at') < min_test_start)])
        test = test.filter(pl.col('sold_at') >= min_test_start)
    if test.is_empty():
        raise ValueError(f"No test rows with sold_at >= {min_test_start}")
    assert test['sold_at'].min() > train['sold_at'].max(), "Temporal leak!"

    val_cutoff = int(len(train) * 0.9)
    return train[:val_cutoff], train[val_cutoff:], test


def _make_pool(df: pl.DataFrame) -> cb.Pool:
    return cb.Pool(df.select(ALL_FEATURES).to_pandas(), label=df[TARGET].to_numpy(), cat_features=CAT_FEATURES)


def tune(pool_train: cb.Pool, pool_val: cb.Pool, n_trials: int) -> dict[str, Any]:
    """Optuna search (notebook search space); each trial is a nested MLflow run."""
    def objective(trial: optuna.Trial) -> float:
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 1e-3, 5.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        }
        model = cb.CatBoostRegressor(**BASE_PARAMS, **params)
        model.fit(pool_train, eval_set=pool_val)
        val_rmse = model.get_best_score()['validation']['RMSE']

        with mlflow.start_run(run_name=f"trial_{trial.number:03d}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metrics({'val_rmse_log': val_rmse, 'best_iteration': model.get_best_iteration()})
        return val_rmse

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _feature_importance_figure(model: cb.CatBoostRegressor) -> Figure:
    fi = model.get_feature_importance(prettified=True)
    fig = Figure(figsize=(8, 6))
    ax = fig.subplots()
    ax.barh(fi['Feature Id'][::-1], fi['Importances'][::-1])
    ax.set_title('Model C — feature importance')
    ax.set_xlabel('Importance')
    fig.tight_layout()
    return fig


def _champion(client: MlflowClient):
    """Current champion ModelVersion, or None on the very first run."""
    models = client.search_registered_models(filter_string=f"name = '{MODEL_NAME}'")
    if not models or CHAMPION_ALIAS not in models[0].aliases:
        return None
    return client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)


def train_and_gate(df: pl.DataFrame, sql: str, cutoff: datetime, n_trials: int) -> dict[str, Any]:
    """
    Tune + train a challenger, score it and the current champion on the same
    temporal test slice, and register the challenger as champion only if its
    test RMSE(log) is strictly lower. Returns a JSON-serialisable report.
    """
    client = MlflowClient()
    mlflow.set_experiment(EXPERIMENT_NAME)

    champion = _champion(client)
    min_test_start = datetime.fromisoformat(champion.tags['test_start']) if champion else None
    df_train, df_val, df_test = split(df, min_test_start)
    test_start = df_test['sold_at'].min()

    pool_train, pool_val, pool_test = _make_pool(df_train), _make_pool(df_val), _make_pool(df_test)
    y_test = df_test[TARGET].to_numpy()

    with mlflow.start_run(run_name=f"retrain_{cutoff:%Y-%m-%d}") as run:
        # Also copied onto the registered version: test_start feeds the next run's split
        lineage = {
            'cutoff': cutoff.isoformat(),
            'data_sha256': frame_hash(df),
            'test_start': test_start.isoformat(),
            'run_id': run.info.run_id,
        }
        mlflow.set_tags({**lineage, 'features': ','.join(ALL_FEATURES)})
        mlflow.log_text(sql, 'training_data.sql')
        mlflow.log_params({
            'n_trials': n_trials,
            'n_rows': len(df), 'n_train': len(df_train), 'n_val': len(df_val), 'n_test': len(df_test),
            'train_sold_at_range': f"{df_train['sold_at'].min()} -> {df_train['sold_at'].max()}",
            'val_sold_at_range': f"{df_val['sold_at'].min()} -> {df_val['sold_at'].max()}",
            'test_sold_at_range': f"{test_start} -> {df_test['sold_at'].max()}",
        })

        best_params = tune(pool_train, pool_val, n_trials)
        mlflow.log_params({**BASE_PARAMS, **best_params})

        model = cb.CatBoostRegressor(**BASE_PARAMS, **best_params)
        model.fit(pool_train, eval_set=pool_val)
        pred = model.predict(pool_test)
        zeros = np.zeros_like(y_test)  # "trust CSFloat" baseline: multiplier exp(0) = 1
        report: dict[str, Any] = {
            'run_id': run.info.run_id,
            'experiment_id': run.info.experiment_id,
            'cutoff': cutoff.isoformat(),
            'n_train': len(df_train),
            'n_test': len(df_test),
            'test_sold_at_range': f"{test_start} -> {df_test['sold_at'].max()}",
            'best_params': best_params,
            'challenger_rmse_log': rmse_log(y_test, pred),
            'challenger_mdape': mdape_backtransformed(y_test, pred),
            'baseline_rmse_log': rmse_log(y_test, zeros),
            'baseline_mdape': mdape_backtransformed(y_test, zeros),
            'champion_version': None,
            'champion_rmse_log': None,
            'champion_mdape': None,
        }
        mlflow.log_metrics({
            'test_rmse_log': report['challenger_rmse_log'],
            'test_mdape': report['challenger_mdape'],
            'baseline_test_rmse_log': report['baseline_rmse_log'],
            'baseline_test_mdape': report['baseline_mdape'],
            'best_iteration': model.get_best_iteration(),
        })
        mlflow.log_figure(_feature_importance_figure(model), 'feature_importance.png')

        X_test = df_test.select(ALL_FEATURES).to_pandas()
        model_info = mlflow.catboost.log_model(
            model, name='model',
            signature=infer_signature(X_test, pred),
            input_example=X_test.head(5),
        )

        if champion is not None:
            champion_uri = f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"
            # Select by the champion's own signature, so a feature-set change fails loudly here
            champion_cols = mlflow.models.get_model_info(champion_uri).signature.inputs.input_names()
            champion_pred = mlflow.catboost.load_model(champion_uri).predict(df_test.select(champion_cols).to_pandas())
            report['champion_version'] = champion.version
            report['champion_rmse_log'] = rmse_log(y_test, champion_pred)
            report['champion_mdape'] = mdape_backtransformed(y_test, champion_pred)
            mlflow.log_metrics({
                'champion_test_rmse_log': report['champion_rmse_log'],
                'champion_test_mdape': report['champion_mdape'],
            })
            mlflow.set_tag('champion_version', champion.version)

        promoted = champion is None or report['challenger_rmse_log'] < report['champion_rmse_log']
        report['promoted'] = promoted
        mlflow.set_tag('promoted', str(promoted).lower())

        if promoted:
            version = mlflow.register_model(model_info.model_uri, MODEL_NAME).version
            for key, value in lineage.items():
                client.set_model_version_tag(MODEL_NAME, version, key, value)
            client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version)
            report['new_version'] = version

    return report
