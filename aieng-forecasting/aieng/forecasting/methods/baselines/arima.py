"""Auto-ARIMA conventional baseline for continuous multi-horizon tasks.

``ARIMABaseline`` fits ``statsforecast.models.AutoARIMA`` to the price series
visible at the forecast origin and emits one quantile forecast per requested
horizon. The point forecast at each horizon is AutoARIMA's own mean forecast;
the quantile spread is symmetric around it, sized from the in-sample one-step
residual standard deviation scaled by ``sqrt(horizon)`` — the same
"variance grows with time" convention ``GARCHPredictor``'s fallback path uses,
applied here instead of AutoARIMA's own multi-step interval API to keep the
quantile construction identical across the statsforecast-based baselines in
this module family.

Refits at every ``predict()`` call (same style as ``LogisticRegressionBaseline``)
— no persisted model state, no leakage risk from a stale fit.

Usage::

    from aieng.forecasting.methods.baselines import ARIMABaseline
    from aieng.forecasting.evaluation import backtest, BacktestSpec

    predictor = ARIMABaseline(price_series_id="jet_fuel_proxy_price")
    result = backtest(predictor=predictor, spec=spec, data_service=svc)
    print(f"ARIMA mean CRPS: {result.mean_score:.4f}")
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, ContinuousForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask


class ARIMABaseline(Predictor):
    """Auto-selected ARIMA(p,d,q) baseline with residual-std quantile bands.

    Parameters
    ----------
    price_series_id : str
        Series id of the continuous price series to forecast. Should equal
        the task's ``target_series_id`` (or a proxy for it).
    season_length : int
        Seasonal period passed to ``AutoARIMA``. Defaults to ``1`` (no
        seasonality) — daily commodity-price series have no reliable
        calendar seasonality at the horizons used here (a handful of
        trading days).
    min_train_observations : int
        Minimum price observations required to attempt a fit. Below this,
        falls back to a last-value-plus-historical-volatility forecast (no
        ARIMA).
    """

    def __init__(
        self,
        price_series_id: str,
        *,
        season_length: int = 1,
        min_train_observations: int = 30,
    ) -> None:
        self._price_series_id = price_series_id
        self._season_length = season_length
        self._min_train = min_train_observations

    @property
    def predictor_id(self) -> str:
        """Return a stable identifier for this predictor."""
        return "arima_baseline"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Fit AutoARIMA on visible prices and emit one forecast per horizon.

        Raises
        ------
        ValueError
            If the task is not continuous.
        """
        if task.payload_type != "continuous":
            raise ValueError(
                f"{type(self).__name__} requires a continuous task; got payload_type='{task.payload_type}'."
            )

        price_df = context.get_series(self._price_series_id)
        prices = price_df["value"].astype(float).to_numpy()
        max_horizon = max(task.horizons)

        point_forecasts, residual_std, model_name = self._fit_and_forecast(prices, max_horizon)

        from scipy.stats import norm  # noqa: PLC0415

        offset = pd.tseries.frequencies.to_offset(task.frequency)
        issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        predictions: list[Prediction] = []
        for horizon in task.horizons:
            point = float(point_forecasts[horizon - 1])
            std_h = residual_std * math.sqrt(horizon)
            quantiles = {q: point + norm.ppf(q) * std_h for q in STANDARD_QUANTILES}
            predictions.append(
                Prediction(
                    predictor_id=self.predictor_id,
                    task_id=task.task_id,
                    issued_at=issued_at,
                    as_of=context.as_of,
                    forecast_date=(pd.Timestamp(context.as_of) + offset * horizon).to_pydatetime(),
                    payload=ContinuousForecast(point_forecast=point, quantiles=quantiles),
                    metadata={
                        "model": model_name,
                        "n_observations": int(len(prices)),
                        "residual_std_1step": residual_std,
                    },
                )
            )
        return predictions

    def _fit_and_forecast(self, prices: np.ndarray, max_horizon: int) -> tuple[np.ndarray, float, str]:
        """Return ``(point_forecasts, one_step_residual_std, model_name)``."""
        if len(prices) < self._min_train:
            last = float(prices[-1]) if len(prices) else 0.0
            return np.full(max_horizon, last), 0.0, "insufficient_history_fallback"
        try:
            from statsforecast.models import AutoARIMA  # noqa: PLC0415

            model = AutoARIMA(season_length=self._season_length)
            model.fit(prices)
            fitted = model.predict_in_sample()["fitted"]
            residuals = prices[len(prices) - len(fitted) :] - fitted
            # Some ARIMA orders leave leading fitted values NaN (no history to
            # difference/smooth from yet); drop them before computing std.
            residuals = residuals[~np.isnan(residuals)]
            residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
            point_forecasts = model.predict(max_horizon)["mean"]
            return point_forecasts, residual_std, "auto_arima"
        except Exception:  # noqa: BLE001 — deliberate fallback on fit failure
            last = float(prices[-1])
            daily_std = float(np.std(np.diff(prices), ddof=1)) if len(prices) > 1 else 0.0
            return np.full(max_horizon, last), daily_std, "last_value_fallback"


__all__ = ["ARIMABaseline"]
