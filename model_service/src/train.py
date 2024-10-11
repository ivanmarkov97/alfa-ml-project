from __future__ import annotations

import pickle
from typing import Any
from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from dataset import RandomRegressionDataset
from metrics import metrics_report
from utils import get_model_dir


@dataclass
class RandomForestParams:
    n_estimators: int
    max_depth: int
    min_samples_split: int


class RandomForestTrainer:
    """Класс для обучения и оценки качества модели RandomForestRegressor"""

    def __init__(
            self,
            params: RandomForestParams | None = None,
            random_state: int = 0,
            n_jobs: int | None = None
    ) -> None:
        if params is not None:
            self._model = RandomForestRegressor(
                n_estimators=params.n_estimators,
                max_depth=params.max_depth,
                min_samples_split=params.min_samples_split,
                n_jobs=n_jobs,
                random_state=random_state
            )
        else:
            self._model = RandomForestRegressor()
        self._features: list[str] = []
        self._is_fitted: bool = False

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        self._features = x_train.columns.tolist()
        self._model.fit(x_train, y_train)
        self._is_fitted = True

    def save_model(self, model_file: str) -> None:
        with open(model_file, 'wb') as f:
            model_dict: dict[str, Any] = {
                'model': self._model,
                'features': self._features,
                'fitted': self._is_fitted
            }
            pickle.dump(model_dict, f)

    def load_model(self, model_file: str) -> None:
        model_dict: dict[str, Any] = pickle.load(open(model_file, 'rb'))
        try:
            self._model = model_dict['model']
            self._features = model_dict['features']
            self._is_fitted = model_dict['fitted']
        except KeyError:
            print('Cannot load model from file')

    def predict(self, x_test: pd.DataFrame) -> pd.Series:
        if not self._is_fitted:
            raise ValueError('Model not fitted')
        return self._model.predict(x_test[self._features])

    def evaluate(self, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
        predictions: pd.Series = self.predict(x_test)
        return metrics_report(predictions, y_test)


if __name__ == '__main__':
    dataset: RandomRegressionDataset = RandomRegressionDataset(n_rows=1_000, n_columns=10)
    x_train, x_test, y_train, y_test = dataset.train_test_split(train_size=0.75)

    p: RandomForestParams = RandomForestParams(n_estimators=10, max_depth=4, min_samples_split=2)
    trainer: RandomForestTrainer = RandomForestTrainer(p, n_jobs=1, random_state=0)

    trainer.fit(x_train, y_train)
    trainer.save_model(str(get_model_dir() / 'model.pkl'))
    trainer.load_model(str(get_model_dir() / 'model.pkl'))
    metrics: dict[str, float] = trainer.evaluate(x_test, y_test)

    print('### METRICS ###')
    for key, value in metrics.items():
        print(f'{key}:\t{value}')
