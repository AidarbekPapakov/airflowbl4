import polars as pl


def temporal_split(
    df: pl.DataFrame,
    date_col: str = "sold_at",
    test_frac: float = 0.2,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Sort by date_col, split at the (1 - test_frac) quantile.
    Returns (df_train, df_test). Test set is strictly after train set.
    """
    df_sorted = df.sort(date_col)
    cutoff_idx = int(len(df_sorted) * (1 - test_frac))
    cutoff_date = df_sorted[date_col][cutoff_idx]

    df_train = df_sorted.filter(pl.col(date_col) < cutoff_date)
    df_test = df_sorted.filter(pl.col(date_col) >= cutoff_date)
    return df_train, df_test
