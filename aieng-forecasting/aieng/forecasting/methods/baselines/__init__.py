"""Baseline predictor implementations.

Baselines provide fast, low-dependency reference points that every more complex
predictor should be compared against.
"""

from .categorical_frequency import CategoricalFrequencyPredictor
from .garch import GARCHPredictor
from .historical_frequency import HistoricalFrequencyPredictor
from .logistic_regression import LogisticRegressionBaseline
from .naive import LastValuePredictor


__all__ = [
    "CategoricalFrequencyPredictor",
    "GARCHPredictor",
    "HistoricalFrequencyPredictor",
    "LastValuePredictor",
    "LogisticRegressionBaseline",
]
