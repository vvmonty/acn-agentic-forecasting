"""GARCH(1,1) conventional baseline for binary "threshold move" tasks.

``GARCHPredictor`` answers questions of the shape *"will series X move more
than P% over the next H steps?"* by fitting a GARCH(1,1) volatility model to
the log-returns of a reference price series, then converting the fitted
drift/volatility into a probability that the cumulative log-return over the
task's horizon exceeds the threshold, under a Gaussian cumulative-return
assumption. This is the standard quantitative-finance baseline for
threshold/shock-style binary event tasks (the doc-referenced alternative to
GARCH is a plain random-walk drift/vol model, which this predictor falls back
to if the GARCH fit itself fails to converge).

The predictor is deliberately generic — it forecasts the probability for
*any* binary task whose event is a threshold move on a configured price
series — so it can be reused across use cases, not just the one that
motivated it (fuel cost-shock). The event series backing the task itself is
only used for warmup/resolution by the harness; this predictor never reads
``task.target_series_id`` and instead always looks at ``price_series_id``.

Usage::

    from aieng.forecasting.methods.baselines import GARCHPredictor
    from aieng.forecasting.evaluation import backtest, BacktestSpec

    predictor = GARCHPredictor(price_series_id="jet_fuel_proxy_price", threshold_pct=0.10)
    result = backtest(predictor=predictor, spec=spec, data_service=svc)
    print(f"GARCH mean Brier: {result.mean_score:.4f}")
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import BinaryForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask


class GARCHPredictor(Predictor):
    """Binary baseline: P(cumulative return over the horizon exceeds a threshold).

    Fits a constant-mean GARCH(1,1) model to the log-returns of
    ``price_series_id`` (as visible at ``context.as_of``), then forecasts the
    mean and variance of the cumulative log-return over the task's single
    horizon. The event probability is read off a Gaussian tail assuming
    independent daily log-returns (variance of the sum = sum of the forecast
    per-step variances).

    Parameters
    ----------
    price_series_id : str
        Series id of the continuous price series to model (not the derived
        0/1 event series declared as the task's target).
    threshold_pct : float
        Fractional price move that defines the event, e.g. ``0.10`` for
        "rises more than 10%". Must match the threshold used to derive the
        task's target event series, or the probability will answer a
        different question than the one being scored.
    min_train_observations : int
        Minimum number of log-returns required to attempt a GARCH fit.
        Below this, falls back to a plain historical mean/variance
        random-walk calculation (no GARCH).
    """

    def __init__(
        self,
        price_series_id: str,
        *,
        threshold_pct: float = 0.10,
        min_train_observations: int = 60,
    ) -> None:
        if threshold_pct <= 0:
            raise ValueError(f"threshold_pct must be positive; got {threshold_pct}")
        self._price_series_id = price_series_id
        self._threshold_pct = threshold_pct
        self._min_train = min_train_observations

    @property
    def predictor_id(self) -> str:
        """Return a stable identifier for this predictor."""
        return "garch_1_1"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Fit GARCH(1,1) on returns visible at the origin and emit one forecast.

        Raises
        ------
        ValueError
            If the task is not binary, requests more than one horizon, or
            the configured price series has too little history to compute
            even one log-return.
        """
        if task.payload_type != "binary":
            raise ValueError(f"{type(self).__name__} requires a binary task; got payload_type='{task.payload_type}'.")
        if len(task.horizons) != 1:
            raise ValueError(f"{type(self).__name__} supports exactly one horizon; got {task.horizons}.")
        horizon = task.horizons[0]

        price_df = context.get_series(self._price_series_id)
        prices = price_df["value"].astype(float).to_numpy()
        if len(prices) < 2:  # noqa: PLR2004
            raise ValueError(
                f"Price series '{self._price_series_id}' has fewer than 2 observations at as_of={context.as_of}."
            )
        log_returns = np.diff(np.log(prices))

        threshold_log_return = math.log(1.0 + self._threshold_pct)
        mean_cum, var_cum, model_name = self._forecast_cumulative_moments(log_returns, horizon)
        std_cum = math.sqrt(max(var_cum, 1e-12))
        z = (threshold_log_return - mean_cum) / std_cum
        probability = float(1.0 - _standard_normal_cdf(z))

        payload = BinaryForecast(probability=probability)
        offset = pd.tseries.frequencies.to_offset(task.frequency)
        issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)

        return [
            Prediction(
                predictor_id=self.predictor_id,
                task_id=task.task_id,
                issued_at=issued_at,
                as_of=context.as_of,
                forecast_date=(pd.Timestamp(context.as_of) + offset * horizon).to_pydatetime(),
                payload=payload,
                metadata={
                    "model": model_name,
                    "n_observations": int(len(log_returns)),
                    "mean_cum_log_return": mean_cum,
                    "std_cum_log_return": std_cum,
                },
            )
        ]

    def _forecast_cumulative_moments(self, log_returns: np.ndarray, horizon: int) -> tuple[float, float, str]:
        """Return ``(mean_cum, var_cum, model_name)`` for the cumulative log-return.

        Tries a GARCH(1,1) fit first; falls back to plain historical
        mean/variance scaled by the horizon if the ``arch`` package is
        unavailable, there is too little history, or the fit fails to
        converge (a known GARCH failure mode on short or unusually flat
        series — silently returning a nonsensical probability would be
        worse than falling back to the simpler, always-defined estimator).
        """
        if len(log_returns) >= self._min_train:
            try:
                return self._fit_garch(log_returns, horizon)
            except Exception:  # noqa: BLE001 — deliberate fallback on any fit failure
                pass
        mu_daily = float(np.mean(log_returns))
        var_daily = float(np.var(log_returns, ddof=1)) if len(log_returns) > 1 else 0.0
        return mu_daily * horizon, var_daily * horizon, "historical_moments_fallback"

    def _fit_garch(self, log_returns: np.ndarray, horizon: int) -> tuple[float, float, str]:
        from arch import arch_model  # noqa: PLC0415

        # arch_model fits more stably on returns scaled to roughly unit variance;
        # rescale by 100 (i.e. percent log-returns) and undo the scaling afterwards.
        scaled_returns = log_returns * 100.0
        model = arch_model(scaled_returns, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
        res = model.fit(disp="off", show_warning=False)

        mu_daily = float(res.params["mu"]) / 100.0
        forecast = res.forecast(horizon=horizon, reindex=False)
        var_cum_scaled = float(forecast.variance.iloc[-1].sum())
        var_cum = var_cum_scaled / (100.0**2)
        return mu_daily * horizon, var_cum, "garch_1_1"


def _standard_normal_cdf(z: float) -> float:
    """Standard normal CDF via ``math.erf`` (avoids a hard ``scipy`` dependency)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


__all__ = ["GARCHPredictor"]
