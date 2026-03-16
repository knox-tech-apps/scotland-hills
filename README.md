# 🏔️ Ben Lomond Summit Weather Predictor

A Python machine-learning pipeline that fetches historical weather data for the
Ben Lomond summit (974 m, Loch Lomond & The Trossachs, Scotland), trains
Random Forest models, and produces a 7-day forecast with a hill-safety summary.

---

## What it does

| Step | Description |
|------|-------------|
| **Fetch** | Downloads 5 years of hourly reanalysis data from Open-Meteo Archive API — no API key required |
| **Engineer** | Creates 40+ features: lag windows, pressure tendency, cyclical time encoding, rolling statistics |
| **Train** | Fits four Random Forest models (temperature, wind, precipitation, cloud cover) using TimeSeriesSplit cross-validation |
| **Evaluate** | Reports MAE and R² per model per fold |
| **Forecast** | Runs the next 7 days of NWP model output through the trained RF models |
| **Summarise** | Prints a plain-English daily safety rating (🟢 GOOD / 🟡 CAUTION / 🔴 AVOID) |

### Output files (written to `ben_lomond_outputs/`)

| File | Contents |
|------|----------|
| `seasonal_climatology.png` | Monthly box plots for all four variables |
| `cv_scores.png` | Cross-validation MAE and R² bar charts |
| `feature_importance.png` | Top-15 features per model |
| `7day_forecast.png` | 7-day forecast vs Open-Meteo NWP |
| `models.pkl` | Serialised models + scaler (reload with `joblib.load`) |

---

## Setup (Mac — first time only)

### 1. Python 3.9 or later
Check your version:
```bash
python3 --version
```
If needed, install via [python.org](https://www.python.org/downloads/) or Homebrew:
```bash
brew install python
```

### 2. Create a virtual environment (recommended)
```bash
cd /path/to/this/folder
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Run

```bash
python ben_lomond_predictor.py
```

First run downloads ~5 years of hourly data (~43,000 rows) and caches it
locally in `.weather_cache/` — subsequent runs are much faster.

Total runtime: ~2–4 minutes on a modern Mac.

---

## Customising the model

All key settings are near the top of `ben_lomond_predictor.py`:

```python
HISTORY_YEARS = 5        # Increase for more training data (up to ~80 years via ERA5)
WIND_DANGER   = 60       # km/h threshold for 🔴 wind warning
PRECIP_HIGH   = 5        # mm/day threshold for heavy rain warning
```

To change the summit location, update the `BEN_LOMOND` dict:
```python
BEN_LOMOND = {
    "name": "Ben Nevis Summit",
    "lat":  56.7969,
    "lon":  -5.0035,
    "elevation": 1345,
}
```

---

## Re-using saved models

```python
import joblib, pandas as pd
from ben_lomond_predictor import fetch_forecast, build_openmeteo_client, engineer_features

bundle = joblib.load("ben_lomond_outputs/models.pkl")
models, scaler, features = bundle["models"], bundle["scaler"], bundle["features"]

client = build_openmeteo_client()
fore = engineer_features(fetch_forecast(client))
X = scaler.transform(fore[features].values)

for target, model in models.items():
    fore[f"pred_{target}"] = model.predict(X)

print(fore[["pred_temperature_2m", "pred_wind_speed_10m"]].head(24))
```

---

## Data source

[Open-Meteo](https://open-meteo.com/) — free, open-source weather API.
- Historical data: ERA5 reanalysis from ECMWF
- Forecast data: ECMWF IFS model
- Elevation correction applied server-side

---

## Safety note

> This tool is for educational and planning purposes only.
> **Always check the [Mountain Weather Information Service (MWIS)](https://www.mwis.org.uk)**
> and [Met Office Mountain Forecast](https://www.metoffice.gov.uk/weather/specialist-forecasts/mountain-forecasts/ben-lomond)
> before any summit attempt.
