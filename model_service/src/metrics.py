from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn import metrics as sk_metrics

if TYPE_CHECKING:
    import pandas as pd


def metrics_report(predictions: pd.Series, labels: pd.Series) -> dict[str, float]:
    return {
        'mse': sk_metrics.mean_squared_error(y_true=labels, y_pred=predictions),
        'mae': sk_metrics.mean_absolute_error(y_true=labels, y_pred=predictions),
        'mape': sk_metrics.mean_absolute_percentage_error(y_true=labels, y_pred=predictions)
    }
