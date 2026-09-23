# Fuel Cost-Shock Forecaster

Discrete-event forecasting for jet-fuel-proxy price shocks, adapted from the
[`energy_oil_forecasting`](../energy_oil_forecasting/) (WTI) reference
implementation. Full objectives and requirements:
[`../../fuel-cost-shock-forecaster/Fuel_Cost_Shock_Forecaster_Objectives_and_Requirements.md`](../../fuel-cost-shock-forecaster/Fuel_Cost_Shock_Forecaster_Objectives_and_Requirements.md).

**The multi-agent pipeline lives in [`01_fuel_shock_multi_agent.ipynb`](01_fuel_shock_multi_agent.ipynb).**
This is a notebook-first bootcamp project — the Commodity-Data ingestion, the
Geopolitical/News Agent, the Forecaster Agent, and the task specs are all
defined inline in that notebook, not split across importable `.py` modules.
Run it top to bottom.

**The backtest actually runs in [`02_fuel_shock_backtest_eval.ipynb`](02_fuel_shock_backtest_eval.ipynb).**
`01_...` builds and demos the pipeline at a single live origin only; it never
calls the evaluation harness. `02_...` re-declares the setup it needs from
`01_...` (same notebook-only convention — no shared `.py` module between the
two notebooks) and runs the real backtest against `specs/fuel_shock_smoke.yaml`
and `specs/fuel_shock_backtest.yaml`, with the GARCH and logistic-regression
baselines described below.

## Targets

- **Primary (binary):** P(jet-fuel proxy rises > 10% over the next 21 trading
  days) — `fuel_cost_shock_event_21d` derived series, `DiscreteAgentForecastOutput`.
- **Secondary (continuous):** jet-fuel proxy point forecast at 5/10/21 trading
  days — `jet_fuel_proxy_price` series, `ContinuousAgentForecastOutput`.

## Architecture: 3 agents, not 4

The objectives doc originally specified a 4-agent architecture (Commodity-Data,
Geopolitical/News, Forecaster, Adjudicator). **The Adjudicator agent has been
dropped** (explicit scope decision, not an oversight):

- No pattern for multi-run adjudication (calibration averaging, driver
  flagging across runs) exists anywhere in this repo to build on — it would
  be new orchestration logic, not a reuse-and-adapt job.
- The doc's stated Adjudicator responsibility of "flags the dominant market
  drivers" is already covered by `DiscreteAgentForecastOutput.key_signals` /
  `.reasoning` on the Forecaster Agent's shock-task output (the shock task
  spec explicitly asks the agent to name drivers there).
- The self-consistency module (doc's stretch goal: median over N pipeline
  runs) is orthogonal to having a dedicated Adjudicator agent — it can be
  added later as a thin loop around the predictor, no fourth agent required.

| Agent | Notebook section | Role |
|---|---|---|
| Commodity-Data Agent | §1 | `DataService` registration layer, not a separate LLM call — see below |
| Geopolitical/News Agent | §2 | `ContextRetrievalConfig` sub-agent with temporal-leakage verifier, adapted from `energy_oil_forecasting.analyst_agent` |
| Forecaster Agent | §3–4 | Top-level `AgentConfig`; "one agent, two tasks" pattern (shock + trajectory) |

**Why the Commodity-Data Agent isn't an LLM call:** every reference
implementation in this repo (WTI, BoC, S&P 500) treats quantitative
time-series data as a structured prompt payload built by a plain Python
prompt builder, not as something an LLM agent fetches or summarizes. There is
no precedent here for a data-ingestion *agent* as opposed to a data-ingestion
*function*. `FuelMultitaskPromptBuilder` (notebook §3) builds the payload from
`build_fuel_service()`'s `DataService` (notebook §1), fulfilling the
Commodity-Data Agent's role structurally. If a future iteration wants the
Commodity-Data Agent to actively reason (e.g. flag anomalies before handing
off), give it its own `AgentConfig` + structured output schema and slot it in
ahead of the Forecaster call — the pieces here don't preclude that.

## Data sources

| Source (doc section 6) | Status |
|---|---|
| yfinance (jet-fuel proxy `HO=F`, WTI `CL=F`, USD index `DX-Y.NYB`) | **Implemented**, no API key required |
| EIA Open Data (`petroleum/pri/spt`: WTI spot `RWTC`, jet-fuel spot `EER_EPJK_PF4_RGC_DPG`) | **Implemented** — `EIA_API_KEY` set, endpoint/params verified against `eia-api-swagger.yaml` |
| EIA inventories/stocks (`petroleum/stoc/...`) | **Not yet added** — series id not verified against the swagger file yet |
| Bank of Canada Valet API (CAD/USD FX) | **Substituted** — no Valet adapter exists in this repo; FRED's `DEXCAUS` (Canadian Dollars to US Dollar spot rate) covers the same signal and is already wired up now that `FRED_API_KEY` is set |
| FRED (macro/energy indicators beyond FX) | Only `DEXCAUS` wired so far; add more series if a specific macro covariate list is wanted |
| Gemini Grounding | **Implemented** via `ContextRetrievalConfig` (the News Agent) |
| GDELT 2.0 | Not evaluated — Gemini grounded search is used instead; revisit only if search coverage of OPEC+/shipping-lane events proves insufficient |

Checked `felix-dev` and `kiruthika-dev` remote branches for existing EIA/FX
setup — both point at the same commit as `main`, no divergent work there.

## Jet-fuel proxy choice

No liquid, publicly-traded jet-fuel futures contract exists on Yahoo Finance.
`HO=F` (NY Harbor ULSD / Heating Oil) is used as the proxy — the standard
middle-distillate proxy convention (jet fuel prices as a spread off heating
oil / diesel in industry hedging desks), consistent with the doc's own
"crude oil / jet-fuel proxies" framing (section 3).

## What's reused vs. new (vs. `energy_oil_forecasting`)

**Reused near-verbatim:**
- `ContextRetrievalConfig` + temporal-leakage verifier (news agent)
- `WtiMultitaskPromptBuilder` → `FuelMultitaskPromptBuilder` (history compression, task-spec-driven payload)
- `DiscreteAgentForecastOutput` / `ContinuousAgentForecastOutput` schemas
- `energy_oil_forecasting`'s shock task-spec structure (calibration guidance bands), retargeted from fixed `$5/5-day` to `10%/21-day`
- Brier score / `BacktestResult` evaluation plumbing (`aieng.forecasting.evaluation`)
- `FREDAdapter`'s cache-to-parquet convention, mirrored for the new `EIAAdapter`

**New:**
- `EIAAdapter` — no EIA client exists anywhere in `aieng-forecasting`; built against `eia-api-swagger.yaml` (route `/v2/petroleum/{route1}/{route2}/data`, paginated at 5000 rows/request)
- `derive_shock_event_series` / `FuelShockEventAdapter` — no existing repo pattern derives a *rolling* percentage-based event series; `boc_rate_decisions`'s `BoCDecisionEventAdapter` derives per-*meeting* discrete events instead, a different shape
- Percentage-threshold (not fixed-dollar) shock definition
- Two-agent-role split (news vs. forecaster) documented explicitly, vs. WTI's single analyst identity with search as an internal capability

## Baselines (doc section 7)

`GARCHPredictor` and `LogisticRegressionBaseline` have been added to
`aieng.forecasting.methods.baselines` (alongside the existing
`HistoricalFrequencyPredictor`), so the backtest leaderboard in
`02_fuel_shock_backtest_eval.ipynb` compares the Forecaster Agent against all
three:

- `HistoricalFrequencyPredictor` — climatological base rate (unchanged).
- `GARCHPredictor` — constant-mean GARCH(1,1) fit on jet-fuel proxy
  log-returns at every origin; converts the fitted drift/variance into
  P(21-day return > 10%) under a Gaussian cumulative-return assumption. Falls
  back to plain historical mean/variance if the GARCH fit doesn't converge.
  Generic (takes `price_series_id`/`threshold_pct`, not fuel-specific).
- `LogisticRegressionBaseline` — logistic regression refit at every origin
  on leak-safe trailing-return/volatility features (jet-fuel proxy momentum +
  volatility, plus WTI and USD-index trailing returns as covariates). Generic
  (takes `price_series_id`/`covariate_series_ids`), modelled after
  `boc_rate_decisions.predictors.BoCLogisticPredictor`'s fit-at-origin design.

Both are added to `aieng-forecasting`'s `numerical` optional-dependency group
(`arch`, `scikit-learn`) in `pyproject.toml`.

## Backtest window and the LLM knowledge-cutoff constraint

Both configured models (`gemini-3.1-flash-lite-preview`,
`gemini-3.5-flash`) have a **~January 2025** knowledge cutoff. Backtesting
before that date risks the Forecaster Agent's binary shock probability
reflecting *memorized* outcomes rather than genuine forecasting — a
false-positive on accuracy. `specs/fuel_shock_backtest.yaml` and
`specs/fuel_shock_smoke.yaml` are windowed to start **2025-06-02** (a ~5
month safety margin past the cutoff) and (for the full backtest) end
**2026-08-03** (so every origin's 21-trading-day outcome is fully resolved
before "today"). See each spec file's header comment for the full rationale.

On top of the window choice, the News Agent's `search_web` tool has its own
code-level temporal fence — independent of the window and not just a prompt
instruction — that prevents it from grounding on post-origin news during a
backtest (`aieng-forecasting/aieng/forecasting/methods/agentic/agent_factory.py`,
`search_web()`): the harness seeds the ADK session with each backtest
origin's `as_of` date, that value overrides anything the LLM itself passes as
`cutoff_date`, and every search result is checked by an independent
leakage-verifier model before being returned.
`02_fuel_shock_backtest_eval.ipynb` §8 shows how to watch this fire during a
run.

## Layout

```
fuel_cost_shock_forecaster/
├── 01_fuel_shock_multi_agent.ipynb        # data service, news agent, forecaster agent, task specs, live demo
├── 02_fuel_shock_backtest_eval.ipynb      # re-declares the pipeline setup and runs the real backtest + leaderboard
└── specs/                                  # YAML backtest + smoke specs (data/config, not code)
```

## Setup

```bash
uv sync
```

`.env` needs `GEMINI_API_KEY` (agent calls), `FRED_API_KEY` (CAD/USD FX), and
`EIA_API_KEY` (crude/jet-fuel spot prices) — all three are set. Run `make lint`
before pushing changes.

## Next steps

1. Add EIA inventory/stocks series once the correct `petroleum/stoc/...`
   series id is verified against `eia-api-swagger.yaml`.
2. Run `02_fuel_shock_backtest_eval.ipynb` end to end (smoke spec, then the
   full backtest) with real API keys to get an actual leaderboard, not just
   the reference implementation.
3. Optional: move `fuel_shock_backtest.yaml` to an `EvalSpec` with an
   `EvalTracker`-enforced `max_runs`, mirroring
   `boc_rate_decisions/02_boc_rate_direction_experiment.ipynb`'s protected
   post-2025 eval — since the whole window here is already post-cutoff,
   repeated re-runs while iterating on the agent are themselves a mild
   overfitting risk that a run budget would close off.
4. Optional: self-consistency (median over N pipeline runs) per the doc's
   stretch goal — no Adjudicator agent required for this.
