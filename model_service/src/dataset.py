from __future__ import annotations

import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


class RandomRegressionDataset:

    def __init__(self, n_rows: int = 10_000, n_columns: int = 10, target_noise: float = 1.5) -> None:
        if n_rows <= 0 or n_rows >= 100_000:
            raise ValueError('Too much rows in dataset. Usage @n_rows must be in (0, 100_000)')
        if n_columns <= 0 or n_columns >= 30:
            raise ValueError('Too much columns in dataset. Usage @n_columns must be in (0, 30)')
        if target_noise < 0:
            raise ValueError('Negative noise. Usage @target_noise must be >= 0')

        data, label = make_regression(
            n_samples=n_rows,
            n_features=n_columns,
            n_informative=max(1, n_columns // 2),
            noise=1.5,
            random_state=0
        )

        self._data = pd.DataFrame(data, columns=[f'f{i}' for i in range(n_columns)])
        self._label = pd.Series(label)

    def train_test_split(self, train_size: float = 0.75) -> tuple[pd.DataFrame | pd.Series, ...]:
        if train_size >= 1 or train_size <= 0:
            raise ValueError('@train_size must be in (0, 1)')
        x_train, x_test, y_train, y_test = train_test_split(
            self._data,
            self._label,
            train_size=train_size,
            random_state=0
        )
        return x_train, x_test, y_train, y_test
