"""Jet fuel cost-shock agent — adapted from the WTI starter agent template.

Predicts whether crude oil or jet fuel prices will rise more than 10% 
over the next month — a binary cost-shock event relevant to airline 
fuel hedging, budgeting, and risk decisions.

Ships with the same building blocks as the starter agent:
- **optional news search** — bounded, cutoff-aware Google Search through 
  the Vector proxy; surfaces OPEC decisions, EIA releases, geopolitical signals
- **optional code execution** — an E2B Python sandbox
- **two lightweight skills** — tool-usage playbooks for search and code execution

Everything routes through the Vector proxy — no direct provider keys.

Pair this with ``99_jet_fuel_shock.ipynb``.

Module-level ``__getattr__`` exposes ``root_agent`` lazily so ``adk web`` can
load this module for interactive (schema-free) use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import (
    AgentPredictor,
    ContinuousAgentForecastOutput,
    build_adk_agent,
)
from aieng.forecasting.methods.agentic.agent_factory import (
    AgentConfig,
    CodeExecutionConfig,
    ContextRetrievalConfig,
)
from aieng.forecasting.models import LITE_MODEL

# Reuse the existing WTI prompt builder + history compression — these serialise
# the task/context into the agent's JSON payload and are not worth duplicating.
from energy_oil_forecasting.analyst_agent import WtiPriceForecastPromptBuilder
from energy_oil_forecasting.jet_fuel_agent.tools import ToolSpec, news_search


# Skills live next to this module. The forecasting contract is always loaded;
# each tool loads its own playbook via its ToolSpec (see tools.py).
_SKILLS_ROOT = Path(__file__).parent / "skills"
_FORECASTING_SKILL = _SKILLS_ROOT / "forecasting"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def _build_starter_instruction() -> str:
    """Build the task-agnostic, skill-agnostic starter persona.

    Just the analyst's identity and how to behave — no output schema, no payload
    contract, no skill or tool mechanics. ADK injects the name + description of
    every attached skill (and every tool) into the system prompt, so the agent
    already knows what it can load and call; repeating that here would only
    duplicate dynamically-injected information. The forecasting *contract* lives
    in the loadable ``forecasting`` skill. Edit the persona freely.
    """
    return (
        "## Role\n\n"
        "You are a jet fuel cost-shock analyst for a commercial airline. "
        "You specialize in predicting whether crude oil or jet fuel prices "
        "will rise more than 10% over the next month — a binary cost-shock event. "
        "You are fluent in OPEC+ supply decisions, EIA inventory data, "
        "geopolitical risk, CAD/USD FX dynamics, and energy futures markets.\n\n"
        "## How to respond\n\n"
        "- For open-ended questions, answer directly and concisely.\n"
        "- When asked for a structured forecast, produce a calibrated "
        "probability (0-1) that a cost-shock event occurs, alongside a "
        "point price forecast. Document your reasoning clearly."
    )


_STARTER_INSTRUCTION = _build_starter_instruction()


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------


def build_starter_agent_config(
    model: str = LITE_MODEL,
    *,
    tools: Sequence[ToolSpec] = (),
) -> AgentConfig:
    """Build the WTI starter :class:`AgentConfig` from a toolbelt.

    An agent is a persona plus a list of tools. The persona is fixed here (edit
    ``_build_starter_instruction``); the *toolbelt* is what you compose in the
    notebook — a list of :class:`~energy_oil_forecasting.starter_agent.tools.ToolSpec`
    from the factories in :mod:`energy_oil_forecasting.starter_agent.tools`::

        from energy_oil_forecasting.starter_agent import tools, build_starter_agent_config

        config = build_starter_agent_config(
            model=AGENT_MODEL,
            tools=[tools.news_search(), tools.arima_forecast()],
        )

    Each spec lands in a different ``AgentConfig`` field; this function folds the
    list, routing every fragment to the right place — search sub-agent, code
    sandbox, function tools — and loading each tool's playbook skill and prompt
    supplement. Adding or removing a tool is one line in the notebook.

    Parameters
    ----------
    model : str
        Model for the analyst agent (default: lite). Pass the advanced model
        (``"gemini-3.5-flash"``) for higher-quality runs.
    tools : Sequence[ToolSpec], default=()
        The agent's toolbelt. Build entries with the factories in ``tools.py``
        (``news_search()``, ``code_sandbox()``, ``arima_forecast()``), or write
        your own factory that returns a ``ToolSpec``.

    Returns
    -------
    AgentConfig
    """
    # The forecasting contract is always loaded. Each tool's own playbook is
    # loaded on demand: ADK injects each skill's name + description into the
    # system prompt, and the agent reads the full SKILL.md only when relevant —
    # so adding a tool just adds its skill, no persona edits.
    skills_dirs: list[Path] = [_FORECASTING_SKILL]
    instruction = _STARTER_INSTRUCTION
    context_retrieval = ContextRetrievalConfig()
    code_execution = CodeExecutionConfig()
    function_tools: list[Any] = []
    max_output_tokens: int | None = None

    for spec in tools:
        if spec.skill_dir is not None:
            skills_dirs.append(spec.skill_dir)
        if spec.instruction_supplement:
            instruction += spec.instruction_supplement
        if spec.context_retrieval is not None:
            context_retrieval = spec.context_retrieval
        if spec.code_execution is not None:
            code_execution = spec.code_execution
        if spec.function_tool is not None:
            function_tools.append(spec.function_tool)
        if spec.max_output_tokens is not None:
            max_output_tokens = max(max_output_tokens or 0, spec.max_output_tokens)

    return AgentConfig(
        name="jet_fuel_shock_agent",
        model=model,
        instruction=instruction,
        max_output_tokens=max_output_tokens,
        context_retrieval=context_retrieval,
        code_execution=code_execution,
        function_tools=function_tools,
        skills_dirs=skills_dirs,
    )


# ---------------------------------------------------------------------------
# Predictor convenience factory
# ---------------------------------------------------------------------------


class _StarterForecastPromptBuilder:
    """Add the output schema + a forecast directive to a base builder's payload.

    The exact JSON schema is generated at call time from the output class
    (drift-free) and injected into the user payload — never into the system
    prompt — so the agent stays conversational until it is actually asked to
    forecast. Implements the
    :class:`~aieng.forecasting.methods.agentic.predictor.ForecastPromptBuilder`
    protocol structurally.
    """

    def __init__(self, inner: Callable[..., str], output_schema_json: str) -> None:
        self._inner = inner
        self._schema_json = output_schema_json

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        payload = json.loads(self._inner(task=task, context=context))
        payload["instructions"] = (
            "Produce a calibrated probabilistic forecast for this task and return it by "
            "calling `set_model_response` with a `json_response` string matching "
            "`output_schema` exactly."
        )
        payload["output_schema"] = self._schema_json
        return json.dumps(payload, indent=2)


def build_starter_agent_predictor(config: AgentConfig) -> AgentPredictor:
    """Wrap a starter :class:`AgentConfig` in an :class:`AgentPredictor`.

    Reuses :class:`~energy_oil_forecasting.analyst_agent.WtiPriceForecastPromptBuilder`
    for data serialisation, wrapped so the (drift-free) continuous output schema
    and a forecast directive ride in the payload — keeping the schema out of the
    persona. ``predict(task, context)`` returns one
    :class:`~aieng.forecasting.evaluation.prediction.Prediction` per horizon.
    """
    return AgentPredictor(
        agent_config=config,
        prompt_builder=_StarterForecastPromptBuilder(
            WtiPriceForecastPromptBuilder(),
            ContinuousAgentForecastOutput.prompt_schema_json(),
        ),
        output_schema=ContinuousAgentForecastOutput,
    )


# ---------------------------------------------------------------------------
# Lazy root_agent for `adk web` interactive use
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> Any:
    """Expose ``root_agent`` lazily for schema-free interactive use via ``adk web``."""
    if name == "root_agent":
        # Interactive default: news search on (proxy-only, no extra key).
        return build_adk_agent(build_starter_agent_config(tools=[news_search()]))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
