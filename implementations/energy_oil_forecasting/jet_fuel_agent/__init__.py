"""Jet fuel cost-shock agent — adapted from the WTI starter agent template.

Exports the toolbelt-driven AgentConfig factory, the predictor
convenience factory, and the tools module of per-tool factories.
See 99_jet_fuel_shock.ipynb and agent.py.
"""

from energy_oil_forecasting.jet_fuel_agent import tools
from energy_oil_forecasting.jet_fuel_agent.agent import (
    build_starter_agent_config,
    build_starter_agent_predictor,
)


__all__ = [
    "build_starter_agent_config",
    "build_starter_agent_predictor",
    "tools",
]
