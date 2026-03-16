

"""
The Conic Summit Weather Predictor
====================================

Usage:
    python conic_predictor.py

Requirements:
    pip install openmeteo-requests requests-cache retry-requests pandas scikit-learn matplotlib seaborn joblib
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import joblib
import os

# Open-Meteo client libraries
import openmeteo_requests
import requests_cache
from retry_requests import retry

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

CONIC = {
     "name": "Conic Hill Summit",
    "lat":   56.0971,
    "lon":  -4.5252,
    "elevation": 361,

             # metres
}

# How many years of historical data to fetch
HISTORY_YEARS = 6

# Target variables to predict
TARGETS = [
    "temperature_2m",
    "wind_speed_10m",
    "precipitation",
    "cloud_cover",
]

TARGET_LABELS = {
    "temperature_2m": "Temperature (°C)",
    "wind_speed_10m": "Wind Speed (km/h)",
    "precipitation": "Precipitation (mm)",
    "cloud_cover": "Cloud Cover (%)",
}

OUTPUT_DIR = "conic_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. DATA FETCHING
# ─────────────────────────────────────────────

def build_openmeteo_client():
    """Create a cached, retry-enabled Open-Meteo client."""
    cache_session = requests_cache.CachedSession(".weather_cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def fetch_historical(client, years: int = HISTORY_YEARS) -> pd.DataFrame:
    """Fetch hourly historical reanalysis data from Open-Meteo Archive API."""
    end_date = datetime.now().date() - timedelta(days=7)   # archive has ~7-day lag
    start_date = end_date - timedelta(days=365 * years)

    print(f"📡  Fetching historical data: {start_date} → {end_date}")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":  CONIC["lat"],
        "longitude": CONIC["lon"],
        "elevation": CONIC["elevation"],
        "start_date": str(start_date),
        "end_date":   str(end_date),
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "apparent_temperature",
            "precipitation",
            "rain",
            "snowfall",
            "pressure_msl",
            "surface_pressure",
            "cloud_cover",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "shortwave_radiation",
        ],
        "wind_speed_unit": "kmh",
        "timezone": "Europe/London",
    }

    responses = client.weather_api(url, params=params)
    r = responses[0]

    hourly = r.Hourly()
    variables = [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m",
        "apparent_temperature", "precipitation", "rain", "snowfall",
        "pressure_msl", "surface_pressure", "cloud_cover",
        "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
        "shortwave_radiation",
    ]

    data = {"time": pd.date_range(
        start=pd.Timestamp(hourly.Time(), unit="s", tz="Europe/London"),
        end=pd.Timestamp(hourly.TimeEnd(), unit="s", tz="Europe/London"),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )}

    for i, var in enumerate(variables):
        data[var] = hourly.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(data).set_index("time")
    print(f"✅  Historical data loaded: {len(df):,} hourly records")
    return df


def fetch_forecast(client) -> pd.DataFrame:
    """Fetch 7-day hourly forecast from Open-Meteo Forecast API."""
    print("📡  Fetching 7-day forecast...")

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":  CONIC["lat"],
        "longitude": CONIC["lon"],
        "elevation": CONIC["elevation"],
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "apparent_temperature", "precipitation", "rain", "snowfall",
            "pressure_msl", "surface_pressure", "cloud_cover",
            "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
            "shortwave_radiation",
        ],
        "wind_speed_unit": "kmh",
        "timezone": "Europe/London",
        "forecast_days": 14,
    }

    responses = client.weather_api(url, params=params)
    r = responses[0]
    hourly = r.Hourly()

    variables = [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m",
        "apparent_temperature", "precipitation", "rain", "snowfall",
        "pressure_msl", "surface_pressure", "cloud_cover",
        "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
        "shortwave_radiation",
    ]

    data = {"time": pd.date_range(
        start=pd.Timestamp(hourly.Time(), unit="s", tz="Europe/London"),
        end=pd.Timestamp(hourly.TimeEnd(), unit="s", tz="Europe/London"),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )}
    for i, var in enumerate(variables):
        data[var] = hourly.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(data).set_index("time")
    print(f"✅  Forecast data loaded: {len(df)} hourly records")
    return df


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based, lag, rolling, and cyclical features."""
    d = df.copy()

    # ── Temporal features ──────────────────────
    d["hour"]       = d.index.hour
    d["day_of_year"]= d.index.dayofyear
    d["month"]      = d.index.month
    d["week"]       = d.index.isocalendar().week.astype(int)
    d["is_weekend"] = d.index.dayofweek.isin([5, 6]).astype(int)

    # Cyclical encoding (avoid discontinuity at midnight/Jan)
    d["hour_sin"]        = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"]        = np.cos(2 * np.pi * d["hour"] / 24)
    d["doy_sin"]         = np.sin(2 * np.pi * d["day_of_year"] / 365)
    d["doy_cos"]         = np.cos(2 * np.pi * d["day_of_year"] / 365)
    d["wind_dir_sin"]    = np.sin(np.radians(d["wind_direction_10m"]))
    d["wind_dir_cos"]    = np.cos(np.radians(d["wind_direction_10m"]))

    # ── Lag features (past observations as predictors) ──
    for lag in [1, 3, 6, 12, 24]:
        d[f"temp_lag_{lag}h"]     = d["temperature_2m"].shift(lag)
        d[f"pressure_lag_{lag}h"] = d["pressure_msl"].shift(lag)
        d[f"wind_lag_{lag}h"]     = d["wind_speed_10m"].shift(lag)

    # ── Pressure tendency (key mountain weather signal) ──
    d["pressure_change_3h"]  = d["pressure_msl"] - d["pressure_msl"].shift(3)
    d["pressure_change_6h"]  = d["pressure_msl"] - d["pressure_msl"].shift(6)
    d["pressure_change_12h"] = d["pressure_msl"] - d["pressure_msl"].shift(12)

    # ── Rolling statistics (6h and 24h windows) ──
    for window, label in [(6, "6h"), (24, "24h")]:
        d[f"temp_mean_{label}"]   = d["temperature_2m"].rolling(window).mean()
        d[f"temp_std_{label}"]    = d["temperature_2m"].rolling(window).std()
        d[f"precip_sum_{label}"]  = d["precipitation"].rolling(window).sum()
        d[f"wind_max_{label}"]    = d["wind_speed_10m"].rolling(window).max()
        d[f"cloud_mean_{label}"]  = d["cloud_cover"].rolling(window).mean()

    # ── Dew-point spread (proxy for fog / low cloud risk) ──
    d["dewpoint_spread"] = d["temperature_2m"] - d["dew_point_2m"]

    # ── Drop NaN rows introduced by lags/rolling ──
    d.dropna(inplace=True)
    return d


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the list of input feature columns (excludes raw targets)."""
    # These are the raw API variables we keep as features too
    input_vars = [
        "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
        "rain", "snowfall", "pressure_msl", "surface_pressure",
        "wind_direction_10m", "wind_gusts_10m", "shortwave_radiation",
        "dewpoint_spread",
    ]
    engineered = [c for c in df.columns if any(
        c.startswith(p) for p in [
            "hour_", "doy_", "wind_dir_", "pressure_change",
            "temp_lag", "pressure_lag", "wind_lag",
            "temp_mean", "temp_std", "precip_sum", "wind_max", "cloud_mean",
        ]
    )] + ["hour", "day_of_year", "month", "week", "is_weekend"]

    all_features = input_vars + engineered
    return [c for c in all_features if c in df.columns]


# ─────────────────────────────────────────────
# 3. MODEL TRAINING
# ─────────────────────────────────────────────

def train_models(df: pd.DataFrame):
    """
    Train one Random Forest per target using TimeSeriesSplit cross-validation.
    Returns: dict of fitted models, scaler, feature list, cv_scores dict
    """
    features = get_feature_columns(df)
    X = df[features].values
    cv_scores = {}
    models = {}

    # We use a shared scaler (RF is not sensitive to scale, but useful for
    # permutation importance comparisons)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=5)

    for target in TARGETS:
        y = df[target].values
        print(f"\n🌲  Training model for: {TARGET_LABELS[target]}")

        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features=0.4,
            n_jobs=-1,
            random_state=42,
        )

        maes, r2s = [], []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled), 1):
            rf.fit(X_scaled[train_idx], y[train_idx])
            preds = rf.predict(X_scaled[val_idx])
            mae = mean_absolute_error(y[val_idx], preds)
            r2  = r2_score(y[val_idx], preds)
            maes.append(mae)
            r2s.append(r2)
            print(f"   Fold {fold}: MAE={mae:.3f}  R²={r2:.3f}")

        # Final fit on all data
        rf.fit(X_scaled, y)
        models[target] = rf
        cv_scores[target] = {"MAE_mean": np.mean(maes), "MAE_std": np.std(maes),
                             "R2_mean":  np.mean(r2s),  "R2_std":  np.std(r2s)}
        print(f"   ✅ CV MAE: {np.mean(maes):.3f} ± {np.std(maes):.3f}   "
              f"R²: {np.mean(r2s):.3f} ± {np.std(r2s):.3f}")

    return models, scaler, features, cv_scores


# ─────────────────────────────────────────────
# 4. PREDICTION
# ─────────────────────────────────────────────

def predict(models, scaler, features, df: pd.DataFrame) -> pd.DataFrame:
    """Run all models on df and return predictions DataFrame."""
    X = scaler.transform(df[features].values)
    preds = {}
    for target, model in models.items():
        preds[f"pred_{target}"] = model.predict(X)
    return pd.DataFrame(preds, index=df.index)


# ─────────────────────────────────────────────
# 5. VISUALISATION
# ─────────────────────────────────────────────

def plot_cv_scores(cv_scores):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Cross-Validation Performance — Conic Summit Models",
                 fontsize=14, fontweight="bold", y=1.02)

    labels  = [TARGET_LABELS[t] for t in TARGETS]
    mae_vals= [cv_scores[t]["MAE_mean"] for t in TARGETS]
    mae_std = [cv_scores[t]["MAE_std"]  for t in TARGETS]
    r2_vals = [cv_scores[t]["R2_mean"]  for t in TARGETS]
    r2_std  = [cv_scores[t]["R2_std"]   for t in TARGETS]

    colours = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]

    ax = axes[0]
    bars = ax.bar(labels, mae_vals, yerr=mae_std, color=colours, capsize=6,
                  edgecolor="white", linewidth=1.2, alpha=0.85)
    ax.set_title("Mean Absolute Error (lower = better)")
    ax.set_ylabel("MAE")
    ax.tick_params(axis="x", rotation=20)
    for bar, val in zip(bars, mae_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    bars = ax.bar(labels, r2_vals, yerr=r2_std, color=colours, capsize=6,
                  edgecolor="white", linewidth=1.2, alpha=0.85)
    ax.set_title("R² Score (higher = better, max 1.0)")
    ax.set_ylabel("R²")
    ax.set_ylim(0, 1.05)
    ax.axhline(1, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.tick_params(axis="x", rotation=20)
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "cv_scores.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📊  CV scores chart saved → {path}")


def plot_feature_importance(models, features, top_n=15):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Top Feature Importances — Conic Summit Models",
                 fontsize=15, fontweight="bold")
    colours = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]

    for ax, (target, colour) in zip(axes.flat, zip(TARGETS, colours)):
        model = models[target]
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [features[i] for i in indices]
        top_values   = importances[indices]

        ax.barh(top_features[::-1], top_values[::-1], color=colour, alpha=0.8,
                edgecolor="white")
        ax.set_title(TARGET_LABELS[target], fontweight="bold", color=colour)
        ax.set_xlabel("Gini Importance")
        ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊  Feature importance chart saved → {path}")


def plot_forecast(forecast_df: pd.DataFrame, predictions: pd.DataFrame):
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(
        f"14-Day Forecast — {CONIC['name']} ({CONIC['elevation']}m)\n"
        f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')}",
        fontsize=14, fontweight="bold"
    )

    plot_cfg = [
        ("temperature_2m",  "pred_temperature_2m",  "#E53935", "Temperature (°C)",    "°C"),
        ("wind_speed_10m",  "pred_wind_speed_10m",  "#1E88E5", "Wind Speed (km/h)",   "km/h"),
        ("precipitation",   "pred_precipitation",   "#43A047", "Precipitation (mm)",  "mm"),
        ("cloud_cover",     "pred_cloud_cover",      "#757575", "Cloud Cover (%)",     "%"),
    ]

    for ax, (actual_col, pred_col, colour, title, unit) in zip(axes, plot_cfg):
        # Shade behind forecast (predictions = model output)
        ax.fill_between(predictions.index,
                        predictions[pred_col] * 0.85,
                        predictions[pred_col] * 1.15,
                        alpha=0.15, color=colour, label="±15% uncertainty band")
        ax.plot(predictions.index, predictions[pred_col],
                color=colour, linewidth=2, label=f"Model prediction ({unit})")
        if actual_col in forecast_df.columns:
            ax.plot(forecast_df.index, forecast_df[actual_col],
                    color="black", linewidth=1.2, linestyle="--",
                    alpha=0.6, label=f"Open-Meteo NWP ({unit})")

        ax.set_ylabel(unit)
        ax.set_title(title, fontweight="bold", loc="left")
        ax.legend(fontsize=8, loc="upper right")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.grid(True, alpha=0.3)

        # Highlight daytime (6–20) lightly
        for day_offset in range(8):
            day_start = predictions.index[0].normalize() + pd.Timedelta(days=day_offset, hours=6)
            day_end   = day_start + pd.Timedelta(hours=14)
            ax.axvspan(day_start, day_end, alpha=0.05, color="yellow")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "7day_forecast.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊  7-day forecast chart saved → {path}")


def plot_seasonal_climatology(df: pd.DataFrame):
    """Monthly climatology box plots for each target variable."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Monthly Climatology — {CONIC['name']} (last {HISTORY_YEARS} years)",
                 fontsize=14, fontweight="bold")
    colours = ["#E53935", "#1E88E5", "#43A047", "#757575"]
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    for ax, (target, colour) in zip(axes.flat, zip(TARGETS, colours)):
        monthly = [df[df.index.month == m][target].dropna().values
                   for m in range(1, 13)]
        bp = ax.boxplot(monthly, patch_artist=True,
                        medianprops=dict(color="white", linewidth=2),
                        boxprops=dict(facecolor=colour, alpha=0.6),
                        whiskerprops=dict(color=colour),
                        capprops=dict(color=colour),
                        flierprops=dict(marker=".", markersize=2,
                                        color=colour, alpha=0.3))
        ax.set_xticklabels(month_names)
        ax.set_title(TARGET_LABELS[target], fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "seasonal_climatology.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊  Seasonal climatology chart saved → {path}")


# ─────────────────────────────────────────────
# 6. HILL-SAFETY SUMMARY
# ─────────────────────────────────────────────

def safety_summary(predictions: pd.DataFrame):
    """Print a plain-English safety assessment for the next 7 days."""
    print("\n" + "═"*55)
    print("  🏔️   CONIC SUMMIT — 14-DAY SAFETY SUMMARY")
    print("═"*55)

    # Daily aggregates
    daily = predictions.resample("D").agg({
        "pred_temperature_2m":  "mean",
        "pred_wind_speed_10m":  "max",
        "pred_precipitation":   "sum",
        "pred_cloud_cover":     "mean",
    })

    WIND_DANGER = 60    # km/h — strong gale on summit
    PRECIP_HIGH = 5     # mm/day
    CLOUD_HIGH  = 80    # % — likely visibility issues
    TEMP_FREEZE = 2     # °C — risk of ice

    for date, row in daily.iterrows():
        day_str = date.strftime("%a %d %b")
        issues = []

        if row["pred_wind_speed_10m"] >= WIND_DANGER:
            issues.append(f"⚠️  HIGH WINDS ({row['pred_wind_speed_10m']:.0f} km/h)")
        if row["pred_precipitation"] >= PRECIP_HIGH:
            issues.append(f"🌧️  HEAVY RAIN ({row['pred_precipitation']:.1f} mm)")
        if row["pred_cloud_cover"] >= CLOUD_HIGH:
            issues.append(f"☁️  POOR VISIBILITY ({row['pred_cloud_cover']:.0f}% cloud)")
        if row["pred_temperature_2m"] <= TEMP_FREEZE:
            issues.append(f"🧊  FREEZING CONDITIONS ({row['pred_temperature_2m']:.1f}°C)")

        rating = "🟢 GOOD" if not issues else ("🟡 CAUTION" if len(issues) == 1 else "🔴 AVOID")
        print(f"\n  {day_str}  →  {rating}")
        print(f"    Temp: {row['pred_temperature_2m']:.1f}°C  |  "
              f"Wind: {row['pred_wind_speed_10m']:.0f} km/h  |  "
              f"Rain: {row['pred_precipitation']:.1f} mm  |  "
              f"Cloud: {row['pred_cloud_cover']:.0f}%")
        for issue in issues:
            print(f"    {issue}")

    print("\n" + "═"*55)
    print("  ℹ️   Always check the Mountain Weather Information Service")
    print("       (MWIS) before any summit attempt.")
    print("═"*55 + "\n")


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

def main():
    print("="*55)
    print("  CONIC SUMMIT WEATHER PREDICTOR")
    print(f"  Lat: {CONIC['lat']}°N  Lon: {CONIC['lon']}°W  "
          f"Elev: {CONIC['elevation']}m")
    print("="*55 + "\n")

    client = build_openmeteo_client()

    # ── Fetch data ──────────────────────────────
    hist_raw  = fetch_historical(client, years=HISTORY_YEARS)
    fore_raw  = fetch_forecast(client)

    # ── Feature engineering ────────────────────
    print("\n🔧  Engineering features...")
    hist = engineer_features(hist_raw)
    fore = engineer_features(fore_raw)
    print(f"    Features: {len(get_feature_columns(hist))}  |  "
          f"Training rows: {len(hist):,}  |  Forecast rows: {len(fore)}")

    # ── Seasonal climatology (before modelling) ─
    plot_seasonal_climatology(hist_raw)

    # ── Train models ────────────────────────────
    print("\n" + "─"*55)
    models, scaler, features, cv_scores = train_models(hist)
    plot_cv_scores(cv_scores)
    plot_feature_importance(models, features)

    # ── Save models ──────────────────────────────
    model_path = os.path.join(OUTPUT_DIR, "models.pkl")
    joblib.dump({"models": models, "scaler": scaler, "features": features}, model_path)
    print(f"\n💾  Models saved → {model_path}")

    # ── Forecast ─────────────────────────────────
    print("\n🔮  Generating 7-day summit forecast...")
    # Align forecast features to training features (fill any missing with 0)
    for f in features:
        if f not in fore.columns:
            fore[f] = 0.0

    predictions = predict(models, scaler, features, fore)
    # Clip physically impossible values
    predictions["pred_precipitation"] = predictions["pred_precipitation"].clip(lower=0)
    predictions["pred_cloud_cover"]   = predictions["pred_cloud_cover"].clip(0, 100)
    plot_forecast(fore_raw, predictions)

    # ── Safety summary ────────────────────────────
    safety_summary(predictions)

    print(f"✨  All outputs written to '{OUTPUT_DIR}/' directory\n")


if __name__ == "__main__":
    main()
