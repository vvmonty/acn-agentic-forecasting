"""Fit-at-origin logistic-regression conventional baseline for binary tasks.

``LogisticRegressionBaseline`` is a generic conventional-model reference for
binary event tasks driven by a price series: trailing return and volatility
features (optionally extended with covariate series) predict the event
probability via a logistic regression refit at every forecast origin, in the
same "refit inside predict()" style as the Darts numerical predictors and
``boc_rate_decisions.predictors.BoCLogisticPredictor``. Unlike
``BoCLogisticPredictor`` (which hardcodes BoC-specific macro series), this
predictor takes its price/covariate series ids as constructor arguments so it
can be reused across any binary threshold-move task.

Leakage discipline
-------------------
Training examples are built from the task's own target event series (already
cutoff-filtered by ``ForecastContext``, so only resolved outcomes are ever
seen), with each example's features reconstructed using only price/covariate
history visible *as of that example's own origin date* — not the backtest's
current ``as_of``. This mirrors ``BoCLogisticPredictor``'s discipline of
rebuilding features at each past origin rather than reusing today's feature
snapshot for historical training rows.

Usage::

    from aieng.forecasting.methods.baselines import LogisticRegressionBaseline
    from aieng.forecasting.evaluation import backtest, BacktestSpec

    predictor = LogisticRegressionBaseline(
        price_series_id="jet_fuel_proxy_price",
        covariate_series_ids=["wti_crude_oil_price", "usd_index_price"],
    )
    result = backtest(predictor=predictor, spec=spec, data_service=svc)
    print(f"Logistic mean Brier: {result.mean_score:.4f}")
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import BinaryForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask


_RETURN_WINDOWS_DAYS = (21, 63)
_VOL_WINDOW_DAYS = 21


def _trailing_return(df: pd.DataFrame, as_of: pd.Timestamp, window: int) -> float | None:
    """Fractional price change over the trailing ``window`` observations visible at ``as_of``."""
    visible = df[df["timestamp"] <= as_of]
    if len(visible) <= window:
        return None
    values = visible["value"].astype(float)
    return float(values.iloc[-1] / values.iloc[-1 - window] - 1.0)


def _trailing_log_return_vol(df: pd.DataFrame, as_of: pd.Timestamp, window: int) -> float | None:
    """Std. dev. of daily log-returns over the trailing ``window`` observations visible at ``as_of``."""
    visible = df[df["timestamp"] <= as_of]
    if len(visible) <= window:
        return None
    prices = visible["value"].astype(float).iloc[-(window + 1) :]
    log_returns = np.log(prices / prices.shift(1)).dropna()
    if log_returns.empty:
        return None
    return float(log_returns.std())


class LogisticRegressionBaseline(Predictor):
    """Fit-at-origin logistic regression on trailing price return/volatility features.

    Parameters
    ----------
    price_series_id : str
        Series id of the continuous price series the event is defined on.
        Supplies the return/volatility features every instance uses.
    covariate_series_ids : list[str] or None
        Additional series ids (e.g. related commodity or FX series) to
        include as trailing-return covariate features. ``None`` uses no
        covariates beyond the price series itself.
    regularization_c : float
        Inverse regularization strength passed to scikit-learn's
        ``LogisticRegression``. Deliberately left untuned — this is a
        reference baseline, not a leaderboard entry.
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
        regularization_c: float = 1.0,
        min_training_examples: int = 30,
    ) -> None:
        self._price_series_id = price_series_id
        self._covariate_series_ids = list(covariate_series_ids) if covariate_series_ids else []
        self._c = regularization_c
        self._min_train = min_training_examples
        self._feature_names = [
            f"price_return_{w}d" for w in _RETURN_WINDOWS_DAYS
        ] + [
            "price_vol_21d",
        ] + [
            f"{series_id}_return_21d" for series_id in self._covariate_series_ids
        ]

    @property
    def predictor_id(self) -> str:
        """Return a stable identifier for this predictor."""
        return "logistic_regression"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Fit on resolved past origins visible at the current origin and emit one forecast.

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
        """Compute the leak-safe feature vector visible at ``as_of``, or ``None`` if incomplete."""
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
        """Fit the logistic model and return ``(payload, metadata)``.

        Falls back to the training base rate when the design matrix is too
        small, degenerate (single class), or current features are missing.
        """
        base_rate = float(np.mean(outcomes)) if outcomes else 0.1

        degenerate = (
            current_features is None or len(outcomes) < self._min_train or len(set(outcomes)) < 2  # noqa: PLR2004
        )
        if degenerate:
            return BinaryForecast(probability=base_rate), {"model": "base_rate_fallback"}

        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
        from sklearn.pipeline import make_pipeline  # noqa: PLC0415
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

        model = make_pipeline(StandardScaler(), LogisticRegression(C=self._c, max_iter=1000))
        model.fit(np.asarray(feature_rows), np.asarray(outcomes))

        x_now = np.asarray([[current_features[name] for name in self._feature_names]])
        probability = float(model.predict_proba(x_now)[0, 1])

        coefficients = model.named_steps["logisticregression"].coef_[0]
        return BinaryForecast(probability=probability), {
            "model": "logistic_regression",
            "features": dict(zip(self._feature_names, (float(f) for f in x_now[0]), strict=True)),
            "coefficients": dict(zip(self._feature_names, (float(c) for c in coefficients), strict=True)),
        }


__all__ = ["LogisticRegressionBaseline"]
