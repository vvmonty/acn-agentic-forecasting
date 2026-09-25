"""Fit-at-origin LightGBM conventional baseline for binary tasks.

``LightGBMBaseline`` is the gradient-boosted-tree counterpart of
``LogisticRegressionBaseline``: identical trailing return/volatility feature
construction (reused directly from that module) and identical refit-at-origin
discipline, swapping the scikit-learn logistic regression for
``lightgbm.LGBMClassifier`` so a leaderboard has a non-linear conventional-ML
reference point alongside the linear one.

Usage::

    from aieng.forecasting.methods.baselines import LightGBMBaseline
    from aieng.forecasting.evaluation import backtest, BacktestSpec

    predictor = LightGBMBaseline(
        price_series_id="jet_fuel_proxy_price",
        covariate_series_ids=["wti_crude_oil_price", "usd_index_price"],
    )
    result = backtest(predictor=predictor, spec=spec, data_service=svc)
    print(f"LightGBM mean Brier: {result.mean_score:.4f}")
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import BinaryForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.baselines.logistic_regression import (
    _RETURN_WINDOWS_DAYS,
    _VOL_WINDOW_DAYS,
    _trailing_log_return_vol,
    _trailing_return,
)


class LightGBMBaseline(Predictor):
    """Fit-at-origin LightGBM classifier on trailing price return/volatility features.

    Parameters
    ----------
    price_series_id : str
        Series id of the continuous price series the event is defined on.
        Supplies the return/volatility features every instance uses.
    covariate_series_ids : list[str] or None
        Additional series ids (e.g. related commodity or FX series) to
        include as trailing-return covariate features. ``None`` uses no
        covariates beyond the price series itself.
    n_estimators, max_depth, learning_rate : int, int, float
        Passed through to ``lightgbm.LGBMClassifier``. Deliberately left
        untuned — this is a reference baseline, not a leaderboard entry.
    min_training_examples : int
        Minimum number of resolved past origins with computable features
        required to fit. Below this, falls back to the empirical base rate
        over visible outcomes.
    """

    def __init__(
        self,
        price_series_id: str,
        covariate_series_ids: list[str] | None = None,
        *,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        min_training_examples: int = 30,
    ) -> None:
        self._price_series_id = price_series_id
        self._covariate_series_ids = list(covariate_series_ids) if covariate_series_ids else []
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._min_train = min_training_examples
        self._feature_names = (
            [f"price_return_{w}d" for w in _RETURN_WINDOWS_DAYS]
            + [
                "price_vol_21d",
            ]
            + [f"{series_id}_return_21d" for series_id in self._covariate_series_ids]
        )

    @property
    def predictor_id(self) -> str:
        """Return a stable identifier for this predictor."""
        return "lightgbm_baseline"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Fit on resolved past origins visible now and emit one forecast.

        Raises
        ------
        ValueError
            If the task is not binary or requests more than one horizon.
        """
        if task.payload_type != "binary":
            raise ValueError(f"{type(self).__name__} requires a binary task; got payload_type='{task.payload_type}'.")
        if len(task.horizons) != 1:
            raise ValueError(f"{type(self).__name__} supports exactly one horizon; got {task.horizons}.")
        horizon = task.horizons[0]

        event_df = context.get_series(task.target_series_id)
        price_df = context.get_series(self._price_series_id)
        covariate_dfs = {series_id: context.get_series(series_id) for series_id in self._covariate_series_ids}

        feature_rows: list[list[float]] = []
        outcomes: list[float] = []
        for origin, outcome in zip(event_df["timestamp"], event_df["value"], strict=True):
            features = self._build_features(pd.Timestamp(origin), price_df, covariate_dfs)
            if features is None:
                continue
            feature_rows.append([features[name] for name in self._feature_names])
            outcomes.append(float(outcome))

        current_features = self._build_features(pd.Timestamp(context.as_of), price_df, covariate_dfs)
        payload, model_info = self._fit_and_predict(feature_rows, outcomes, current_features)

        offset = pd.tseries.frequencies.to_offset(task.frequency)
        return [
            Prediction(
                predictor_id=self.predictor_id,
                task_id=task.task_id,
                issued_at=datetime.now(tz=timezone.utc).replace(tzinfo=None),
                as_of=context.as_of,
                forecast_date=(pd.Timestamp(context.as_of) + offset * horizon).to_pydatetime(),
                payload=payload,
                metadata={"n_train": len(outcomes), **model_info},
            )
        ]

    def _build_features(
        self,
        as_of: pd.Timestamp,
        price_df: pd.DataFrame,
        covariate_dfs: dict[str, pd.DataFrame],
    ) -> dict[str, float] | None:
        """Compute the leak-safe feature vector at ``as_of``, or ``None`` if missing."""
        features: dict[str, float] = {}
        for window in _RETURN_WINDOWS_DAYS:
            value = _trailing_return(price_df, as_of, window)
            if value is None:
                return None
            features[f"price_return_{window}d"] = value

        vol = _trailing_log_return_vol(price_df, as_of, _VOL_WINDOW_DAYS)
        if vol is None:
            return None
        features["price_vol_21d"] = vol

        for series_id, df in covariate_dfs.items():
            value = _trailing_return(df, as_of, _VOL_WINDOW_DAYS)
            if value is None:
                return None
            features[f"{series_id}_return_21d"] = value

        return features

    def _fit_and_predict(
        self,
        feature_rows: list[list[float]],
        outcomes: list[float],
        current_features: dict[str, float] | None,
    ) -> tuple[BinaryForecast, dict[str, object]]:
        """Fit the LightGBM classifier and return ``(payload, metadata)``.

        Falls back to the training base rate when the design matrix is too
        small, degenerate (single class), or current features are missing.
        """
        base_rate = float(np.mean(outcomes)) if outcomes else 0.1

        degenerate = (
            current_features is None or len(outcomes) < self._min_train or len(set(outcomes)) < 2  # noqa: PLR2004
        )
        if degenerate:
            return BinaryForecast(probability=base_rate), {"model": "base_rate_fallback"}

        from lightgbm import LGBMClassifier  # noqa: PLC0415

        model = LGBMClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            verbose=-1,
        )
        model.fit(np.asarray(feature_rows), np.asarray(outcomes))

        x_now = np.asarray([[current_features[name] for name in self._feature_names]])
        probability = float(model.predict_proba(x_now)[0, 1])

        importances = dict(zip(self._feature_names, (float(i) for i in model.feature_importances_), strict=True))
        return BinaryForecast(probability=probability), {
            "model": "lightgbm",
            "features": dict(zip(self._feature_names, (float(f) for f in x_now[0]), strict=True)),
            "feature_importances": importances,
        }


__all__ = ["LightGBMBaseline"]
