import json
import numpy as np
import polars as pl

APEX_THRESHOLD: float = 10.0


def _float_position_in_bucket() -> pl.Expr:
    """Vectorised expression: (float - bucket_min) / (bucket_max - bucket_min)."""
    f = pl.col("float_value")
    w = pl.col("wear_name")
    return (
        pl.when(w == "Factory New")
          .then((f - 0.00) / 0.07)
        .when(w == "Minimal Wear")
          .then((f - 0.07) / 0.08)
        .when(w == "Field-Tested")
          .then((f - 0.15) / 0.23)
        .when(w == "Well-Worn")
          .then((f - 0.38) / 0.07)
        .when(w == "Battle-Scarred")
          .then((f - 0.45) / 0.55)
        .otherwise(0.0)
    )


def _parse_blue_gem_playside(blue_gem_json: str | None) -> float:
    if blue_gem_json is None:
        return 0.0
    try:
        data = json.loads(blue_gem_json)
        return float(data.get("playside_blue", 0.0))
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0.0


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Transform raw ClickHouse pull into a feature-ready DataFrame.

    Input: raw columns from csfloat_sales FINAL query.
    Output: feature columns + target_b, target_c, sold_at, market_hash_name (latter two for
    splitting/debugging, not fed to models).
    """

    # Raw ratios for apex filter
    df = df.with_columns([
        (pl.col("price") / pl.col("ref_base_price")).alias("_ratio_b"),
        (pl.col("price") / pl.col("ref_predicted_price")).alias("_ratio_c"),
    ])
    df = df.filter(
        (pl.col("_ratio_b") <= APEX_THRESHOLD)
        & (pl.col("_ratio_c") <= APEX_THRESHOLD)
    )

    # Log transform targets
    df = df.with_columns([
        pl.col("_ratio_b").log(base=np.e).alias("target_b"),
        pl.col("_ratio_c").log(base=np.e).alias("target_c"),
    ])

    # Float features
    df = df.with_columns(_float_position_in_bucket().alias("float_position_in_bucket"))

    # Sticker features
    df = df.with_columns([
        pl.col("stickers").list.len().alias("n_stickers"),
        pl.col("stickers")
          .list.eval(pl.element().struct.field("ref_price"))
          .list.sum()
          .fill_null(0)
          .alias("total_sticker_value"),
        pl.col("stickers")
          .list.eval(pl.element().struct.field("ref_price"))
          .list.max()
          .fill_null(0)
          .alias("max_sticker_value"),
    ])
    df = df.with_columns(
        (pl.col("total_sticker_value") / pl.col("ref_base_price"))
        .alias("sticker_to_base_ratio")
    )

    # Fade features
    df = df.with_columns([
        pl.col("fade_pct").fill_null(0.0),
        pl.col("fade_rank").fill_null(0),
        pl.col("fade_type").fill_null("none"),
    ])

    # Blue gem
    # map_elements skips nulls by default and returns null for them; fill_null handles that
    df = df.with_columns(
        pl.col("blue_gem").map_elements(
            _parse_blue_gem_playside, return_dtype=pl.Float64
        ).fill_null(0.0).alias("blue_gem_playside")
    )

    # Keychain
    df = df.with_columns(pl.col("keychain_ref_price").fill_null(0))

    # Select output columns
    feature_cols = [
        # Float
        "float_value",
        "float_position_in_bucket",
        # Sticker
        "n_stickers",
        "total_sticker_value",
        "max_sticker_value",
        "sticker_to_base_ratio",
        # Fade / pattern
        "fade_pct",
        "fade_rank",
        "fade_type",
        # Blue gem
        "blue_gem_playside",
        # Keychain
        "keychain_ref_price",
        # Categoricals
        "def_index", # <- basically `def_index` serves as numeric representation of weapon_name string
        "paint_seed",
        "rarity", # <- numeric representation of rarity (Consumer grade, Industrial Grade, 08etc.)

        # For now, we explicitly ignore market_hash_name as it data leakage feature that limits us from
        # generalizing to skins not currently in the dataset (simply ignored by config files or yet unreleased), 
        # and the model will fail on such instances.
        # "paint_index",
        
        "wear_name",
        "is_stattrak",
        "is_souvenir",
        # Targets
        "target_b",
        "target_c",
        # For temporal splitting (not a feature used for modeling)
        "sold_at",

        # We ignore `market_hash_name` for the same reason as we do paint_index for. It gives away an information
        # a model should not know.
        # "market_hash_name",
    ]
    return df.select(feature_cols)
