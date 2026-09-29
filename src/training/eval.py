import numpy as np


def rmse_log(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """RMSE on log-transformed targets."""
    return float(np.sqrt(np.mean((y_true_log - y_pred_log) ** 2)))


def mdape_backtransformed(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """
    Median absolute percentage error on back-transformed ratio predictions.
    Uses median (not mean) from unconstrained log-scale models (Ridge, GAM).
    """
    y_true = np.exp(y_true_log)
    y_pred = np.exp(y_pred_log)
    return float(np.median(np.abs(y_true - y_pred) / y_true))
