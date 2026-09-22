# **Fuel Cost-Shock Forecaster: Objectives & Requirements**

# **1\. Executive Summary & Overview**

* **Project Name:** Fuel Cost-Shock Forecaster  
* **Domain:** Energy and Commodity Markets (Aviation Sector Focus)  
* **Problem Statement:** Discrete-event forecasting to predict significant upward price shocks in key energy inputs (crude oil / jet fuel proxies).  
* **Core Purpose:** Build an agentic forecasting system capable of predicting the probability of fuel price shocks and estimating point forecasts to inform corporate hedging, budgeting, and financial risk mitigation.

# **2\. Business Motivation & Background**

Commercial airlines operate under thin profit margins heavily exposed to volatile jet-fuel prices, which represent a multi-billion-dollar annual operational expense. With limited hedging capabilities, airlines are highly vulnerable to sudden market shifts driven by geopolitical instability, supply chain disruptions, and macroeconomic factors.

&nbsp;

An agentic cost-shock forecaster addresses this challenge by combining quantitative time-series data with real-time unstructured news and geopolitical signals to deliver:

&nbsp;

1. Calibrated binary cost-shock probabilities for risk assessment.  
2. Contextual explanations identifying key shock drivers to support executive decision-making.

# **3\. Forecasting Targets & Target Variables**

## **Primary Target (Discrete Event / Binary Forecast)**

* **Definition:** Binary prediction on whether a key input commodity (crude oil / jet-fuel proxy) rises by more than a predefined threshold (**\>10%**) over the next month.  
* **Output:** Calibrated probability estimate P(Price Increase \> 10%).

## **Secondary Target (Continuous Point Forecast)**

* **Definition:** Next-month price point forecast.  
* **Frequency & Horizon:** Daily observations aggregated to monthly rolling forecasts.

# **4\. Reference Implementation & Adaptation Strategy**

## **Base Reference Implementation**

* **Source:** WTI Crude Oil Price Forecasting reference implementation from the Vector Agentic Forecasting Bootcamp repository.

## **What Will Be Reused**

* **Evaluation Framework:** Standard Brier score and calibration/reliability diagram logic.  
* **Core Metrics Pipeline:** Baseline metrics calculation infrastructure.

## **What Will Be Adapted & Extended**

* **Threshold Modification:** Adjust target criteria from generic price directional forecasts to a specific binary **\>10% price increase** (cost-shock event).  
* **Multi-Agent Transition:** Upgrade the single-agent / multi-tool reference implementation into a modular **3–4 agent architecture** with task division, reflection, and adjudication.

# **5\. Agentic Solution Architecture**

The system utilizes a 3–4 agent collaborative architecture designed as follows:

&nbsp;

1. **Commodity-Data Agent:**  
   * Ingests quantitative time-series data including spot prices, futures contracts, inventories, interest rates, and foreign exchange rates.  
2. **Geopolitical / News Agent:**  
   * Gathers unstructured textual information, tracking OPEC+ policy decisions, geopolitical supply disruptions, and EIA policy releases using real-time grounding with citations.  
3. **Forecaster Agent:**  
   * Synthesizes quantitative indicators and qualitative news signals to produce binary shock probability estimates and continuous price point forecasts.  
4. **Adjudicator Agent:**  
   * Evaluates outputs across multiple pipeline runs, performs probability calibration, and flags the dominant market drivers influencing the prediction.

## **Optional Extension**

* **Self-Consistency Module:** Takes the median of multiple pipeline execution runs to refine accuracy and reduce forecast variance.

# **6\. Datasets & APIs**

## **Time-Series / Quantitative Data**

* **EIA Open Data:** Crude oil spot/futures, jet fuel prices, and inventory levels.  
* **yfinance:** Energy futures contracts and energy sector ETF proxy data.  
* **Bank of Canada Valet API:** CAD/USD foreign exchange rates (fuel pricing in USD).  
* **FRED (Federal Reserve Economic Data):** Macroeconomic and energy indicators.

## **Unstructured / News Data**

* **Gemini Grounding & Search / GDELT 2.0:** Real-time search for OPEC decisions, geopolitical events, supply chain shocks, and EIA news releases.

# **7\. Baseline Models & Evaluation Framework**

## **Benchmark Baselines**

To validate the value added by the agentic pipeline, performance will be benchmarked against:

&nbsp;

* Naive Base Rate  
* Random-Walk Baseline  
* GARCH Model  
* Logistic Regression Baseline

## **Evaluation Metrics**

* **Binary Shock Prediction:** Brier Score and Calibration / Reliability Diagrams.  
* **Point Forecast Accuracy:** Root Mean Square Error (RMSE), Mean Absolute Percentage Error (MAPE), Mean Absolute Scaled Error (MASE), and Symmetric MAPE (SMAPE).

## **Success Criteria**

* Agentic forecaster outperforms naive and random-walk baselines on Brier Score for cost-shock prediction.  
* Outperforms random-walk and standard ML baselines on point forecast metrics (RMSE, MAPE, MASE).  
* Delivers clear, interpretable explanations of primary shock drivers.

# **8\. Bootcamp Goals & Scope**

## **Minimum Goals (PoC Scope)**

1. Build and validate the 3–4 agent pipeline (Commodity-Data Agent, Geopolitical/News Agent, Forecaster Agent, Adjudicator Agent).  
2. Evaluate binary cost-shock prediction performance (\>10% price increase) using Brier score and calibration diagrams.  
3. Benchmark the agentic system against standard baselines (random-walk, GARCH, logistic regression).

## **Stretch Goals**

1. Implement self-consistency methods using median aggregation across pipeline runs.  
2. Conduct prompt tuning and model selection experiments.  
3. Perform granular performance evaluations of individual agents.

&nbsp;