# Energy Price Forecasting with Physics-Informed ML

Day-ahead electricity price forecasting using physics-informed gradient boosting. The core idea is to compare a purely statistical model against one that incorporates physical constraints of the electricity market — residual demand, ramp rates, merit order effects, duck curve stress — and evaluate whether the physics features improve both point forecast accuracy and drift stability in production.

Live demo: [piml-energy-forecas.streamlit.app](https://piml-energy-forecas.streamlit.app)

---

## Motivation

Standard ML approaches to electricity price forecasting treat the problem as a pure regression task with time features and lags. This misses the underlying physics: prices are set by the intersection of a supply stack (merit order) and demand. Residual demand — what conventional generators must serve after renewables — is a more fundamental driver than raw demand. Similarly, ramp rates, utilization, and evening solar drops (duck curve) all carry causal information that lag features do not.

This project tests the hypothesis that encoding these physical relationships as features improves forecast accuracy, particularly during price spikes and low-price (high-renewable) regimes, and results in slower drift when the generation mix shifts.

---

## Architecture

Two models are trained and compared throughout:

**Statistical model** (`stat`): gradient boosting with time features, price lags, and raw generation volumes.

**Physics-informed model** (`phys`): same base, plus derived features encoding market physics:
- `residual_demand`: demand minus renewable generation — the key driver of conventional dispatch
- `residual_ramp`: rate of change of residual demand
- `renewable_penetration`: fraction of demand met by renewables
- `utilization` and `utilization_squared`: system stress indicator (nonlinear effects near capacity)
- `demand_ramp_1h`, `demand_ramp_3h`: load change rate
- `evening_solar_drop`: solar generation drop in hours 17–20 (duck curve stress)
- `gas_hour_interaction`: gas dispatch interaction with hour of day (merit order proxy)

Both models also produce **quantile predictions** (10th, 25th, 75th, 90th percentiles) for uncertainty quantification, fitted using separate gradient boosting regressors with quantile loss.

Performance is evaluated separately by price regime: normal (10th–90th percentile), spikes (>90th), and low prices (<10th). The hypothesis is that physics features improve most in the tail regimes.

---

## Data

Data is fetched live from the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) via the `entsoe-py` library. Supported markets:

| Code | Market |
|------|--------|
| DE | Germany-Luxembourg |
| ES | Spain |
| NL | Netherlands |
| FR | France |
| DK | Denmark West |
| GB | Great Britain |
| NO | Norway |

Fetched signals per market: day-ahead prices, actual load, load forecast, generation by fuel type (nuclear, coal, gas, wind onshore/offshore, solar, hydro), and installed capacity.

---

## Project Structure

```
energy_forecasting/
├── entsoe_data_fetcher.py       # ENTSO-E API wrapper and physics feature computation
├── model_training_pipeline.py  # EnergyPriceForecaster class: feature engineering,
│                               # time-series CV, training, quantile models, persistence
├── drift_monitor.py            # Automated drift detection: logs predictions vs actuals,
│                               # computes rolling MAE, alerts on degradation
├── streamlit_app_production.py # Dashboard: live forecasts, uncertainty bands,
│                               # generation mix, drift monitoring tab
├── models/                     # Saved model artefacts (pkl + metadata json)
└── requirements.txt
```

---

## Setup

### Requirements

```
pip install entsoe-py pandas numpy scikit-learn streamlit plotly
```

### ENTSO-E API Key

Register at [transparency.entsoe.eu](https://transparency.entsoe.eu/), then email `transparency@entsoe.eu` with subject "Restful API access". Once issued, set:

```bash
export ENTSOE_API_KEY="your-key-here"
```

For the Streamlit app, add the key to `.streamlit/secrets.toml`:

```toml
ENTSOE_API_KEY = "your-key-here"
```

---

## Usage

### Fetch data and train a model

```python
from entsoe_data_fetcher import EntsoeDataFetcher
from model_training_pipeline import EnergyPriceForecaster

fetcher = EntsoeDataFetcher()
df = fetcher.fetch_complete_dataset(country='ES', start_date='2023-01-01', end_date='2024-01-01')
df = fetcher.compute_physics_features(df)

forecaster = EnergyPriceForecaster()
# See EnergyPriceForecaster.train() for full training call with time-series CV
forecaster.save('models/ES_forecaster')
```

### Run drift monitoring

```bash
python drift_monitor.py --country ES --alert-threshold 0.20
```

This fetches yesterday's actuals, logs predictions vs actuals, computes 7-day rolling MAE for both models, compares against training-time baseline, and prints a drift report. If drift exceeds the threshold (default 20% increase in MAE), it recommends retraining.

Can be scheduled as a cron job or GitHub Action for continuous monitoring.

### Run the dashboard

```bash
streamlit run streamlit_app_production.py
```

---

## Dashboard Features

The Streamlit app has four tabs:

**Forecast**: 24-hour ahead price prediction with uncertainty bands (10th–90th percentile) for both statistical and physics models, alongside actuals.

**Model Performance**: MAE and RMSE breakdown by price regime (normal, spike, low). Quantile coverage plots showing whether the uncertainty bands are calibrated.

**Market Analysis**: Stacked generation mix for the past 7 days with demand overlay. Average generation mix pie chart.

**Drift Monitoring**: Rolling 7-day MAE for both models vs training baseline. Status indicators (healthy / minor drift / retrain recommended) and a visualisation of drift trajectory over time.

---

## Known Limitations

- Lag features introduce autocorrelation that is not fully corrected, resulting in visible lag artefacts in the forecast trace. A proper autoregressive correction or using day-ahead demand forecasts as features (rather than actuals) would reduce this.
- Hyperparameter tuning is not yet automated in the live pipeline; the current models use default or manually set parameters.
- The physics feature `utilization` uses an estimated capacity (95th percentile of observed generation * 1.2) rather than the true installed capacity from ENTSO-E. This is a proxy and degrades for markets with large imports.
- Cross-border flows are not included. For highly interconnected markets (NL, FR) this is a meaningful omission.
- The drift monitor requires a pre-trained model and accumulated prediction logs to be meaningful. On a fresh deployment the first 7 days produce no drift signal.

---

## Hypotheses Being Tested

1. Physics features improve MAE on price spikes more than on normal prices, because spikes are driven by physical scarcity (high utilization, low capacity margin) that statistical features do not capture.
2. The physics model drifts more slowly when the renewable generation mix shifts, because `residual_demand` implicitly adjusts to new penetration levels.
3. Quantile coverage is better calibrated for the physics model in high-volatility regimes.

These are not yet rigorously validated — the current implementation is a proof-of-concept.

---

## Author

Suchita Kulkarni
