"""
Training-data slice for Model C.

The slice is bounded by *ingestion* time, not sale time: the ingest DAG rotates skin
batches and pulls each skin's full sales history, so rows with old `sold_at` keep
arriving. `_ingested_at <= cutoff` pins which rows existed when the model was trained.
(ReplacingMergeTree bumps `_ingested_at` when a sale is re-ingested, so the SQL alone
cannot reproduce the slice forever; the row count + content hash logged next to it
make any such drift detectable.)
"""

import hashlib
from datetime import datetime

import clickhouse_connect
import polars as pl

from src.features.build_features import build_features

# Columns mirror item_factor_est/sql/003_csfloat_feature_eng.sql
_FEATURE_SQL = """\
SELECT
    price,
    ref_base_price,
    ref_predicted_price,
    float_value,
    wear_name,
    def_index,
    paint_index,
    paint_seed,
    rarity,
    is_stattrak,
    is_souvenir,
    stickers,
    keychain_ref_price,
    fade_pct,
    fade_rank,
    fade_type,
    blue_gem,
    market_hash_name,
    sold_at
FROM {database}.csfloat_sales FINAL
WHERE _ingested_at <= toDateTime64('{cutoff}', 3, 'UTC')
ORDER BY id"""


def build_sql(database: str, cutoff: datetime) -> str:
    return _FEATURE_SQL.format(database=database, cutoff=cutoff.strftime("%Y-%m-%d %H:%M:%S"))


def frame_hash(df: pl.DataFrame) -> str:
    """Order-sensitive content hash of the feature frame (the query sorts by id)."""
    return hashlib.sha256(df.hash_rows().to_numpy().tobytes()).hexdigest()


def load_frame(client: clickhouse_connect.driver.Client, database: str, cutoff: datetime) -> tuple[str, pl.DataFrame]:
    """Run the slice query and return (sql, feature frame from build_features)."""
    sql = build_sql(database, cutoff)
    return sql, build_features(pl.from_pandas(client.query_df(sql)))
