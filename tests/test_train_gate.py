"""
Champion/challenger gate for the Model C retrain DAG.

Runs the real train_and_gate on a small synthetic frame against a throwaway
MLflow store in tmp_path. No network, no Airflow. Run with `pytest tests/`.
"""

from datetime import datetime, timedelta

import mlflow
import numpy as np
import polars as pl
import pytest
from mlflow import MlflowClient

from src.training.train import ALL_FEATURES, CHAMPION_ALIAS, MODEL_NAME, split, train_and_gate

CUTOFF = datetime(2026, 9, 1)


def _frame(n: int = 600, shuffle_target: bool = False) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    pos = rng.random(n)
    target = 0.5 * (1 - pos) + rng.normal(0, 0.02, n)  # low float within bucket -> overpay
    if shuffle_target:
        target = rng.permutation(target)
    return pl.DataFrame({
        'float_value': pos * 0.07,
        'float_position_in_bucket': pos,
        'n_stickers': rng.integers(0, 5, n),
        'total_sticker_value': rng.integers(0, 100, n),
        'sticker_to_base_ratio': rng.random(n),
        'fade_pct': np.zeros(n),
        'fade_rank': np.zeros(n, dtype=np.int64),
        'blue_gem_playside': np.zeros(n),
        'keychain_ref_price': np.zeros(n, dtype=np.int64),
        'is_stattrak': np.zeros(n, dtype=bool),
        'is_souvenir': np.zeros(n, dtype=bool),
        'def_index': rng.choice([7, 9, 16], n),
        'wear_name': ['Factory New'] * n,
        'rarity': rng.choice([3, 4], n),
        'fade_type': ['none'] * n,
        'target_c': target,
        'sold_at': [datetime(2026, 1, 1) + timedelta(hours=i) for i in range(n)],
    })


@pytest.fixture(autouse=True)
def mlflow_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # sqlite store puts artifacts in ./mlruns
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_registry_uri(f"sqlite:///{tmp_path}/mlflow.db")
    yield
    mlflow.set_tracking_uri(None)


def _champion_version() -> str:
    return MlflowClient().get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS).version


def test_split_is_temporal_and_respects_min_test_start():
    df = _frame()
    train, val, test = split(df)
    assert train['sold_at'].max() < val['sold_at'].min()
    assert val['sold_at'].max() < test['sold_at'].min()
    assert len(train) + len(val) + len(test) == len(df)

    later = test['sold_at'].min() + timedelta(hours=50)
    train2, val2, test2 = split(df, min_test_start=later)
    assert test2['sold_at'].min() == later
    assert len(train2) + len(val2) + len(test2) == len(df)


def test_first_run_registers_champion_with_sql():
    report = train_and_gate(_frame(), "SELECT 1", CUTOFF, n_trials=1)

    assert report['promoted'] and report['champion_version'] is None
    assert _champion_version() == report['new_version']
    assert mlflow.artifacts.load_text(f"runs:/{report['run_id']}/training_data.sql") == "SELECT 1"
    # 1 parent + 1 nested trial run
    assert len(mlflow.search_runs(experiment_names=['csfloat_model_c'])) == 2


def test_identical_retrain_ties_and_is_not_promoted():
    first = train_and_gate(_frame(), "SELECT 1", CUTOFF, n_trials=1)
    second = train_and_gate(_frame(), "SELECT 1", CUTOFF, n_trials=1)

    assert second['champion_rmse_log'] == pytest.approx(second['challenger_rmse_log'])
    assert not second['promoted'] and 'new_version' not in second
    assert _champion_version() == first['new_version']


def test_better_challenger_replaces_weak_champion():
    weak = train_and_gate(_frame(shuffle_target=True), "SELECT 1", CUTOFF, n_trials=1)
    strong = train_and_gate(_frame(), "SELECT 2", CUTOFF, n_trials=1)

    assert strong['champion_version'] == weak['new_version']
    assert strong['challenger_rmse_log'] < strong['champion_rmse_log']
    assert strong['promoted']
    assert _champion_version() == strong['new_version']
    assert list(ALL_FEATURES) == mlflow.models.get_model_info(
        f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}").signature.inputs.input_names()
