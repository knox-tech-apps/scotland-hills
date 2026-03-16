"""
Scottish Highlands Weather Dashboard
=====================================
A single-page Streamlit dashboard showing forecast charts and safety summaries
for Ben Lomond, The Cobbler (Ben Arthur), and Conic Hill.

Usage:
    streamlit run dashboard.py

Assumes the three predictor scripts are in the same directory:
    ben_lomond_predictor.py
    cobbler_predictor.py
    conic_predictor.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import importlib, sys, types
from datetime import datetime

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Big Hills Weather Analysis",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Source+Sans+3:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
}

.dashboard-header {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    padding: 2.5rem 2rem 2rem 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.dashboard-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
}
.dashboard-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 900;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.5px;
    position: relative;
}
.dashboard-subtitle {
    color: #a8c8d8;
    font-size: 1rem;
    font-weight: 300;
    margin-top: 0.4rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    position: relative;
}
.dashboard-timestamp {
    color: #6a9ab0;
    font-size: 0.8rem;
    margin-top: 0.6rem;
    position: relative;
}

.mountain-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 2.5rem 0 1rem 0;
    padding-bottom: 0.6rem;
    border-bottom: 2px solid;
}
.mountain-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0;
}
.mountain-meta {
    font-size: 0.85rem;
    color: #888;
    font-weight: 400;
    margin-top: 0.1rem;
}

.safety-grid {
    display: grid;
    grid-template-columns: repeat(7, 1fr);
    gap: 0.5rem;
    margin: 1rem 0 2rem 0;
}
.safety-card {
    border-radius: 10px;
    padding: 0.8rem 0.5rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
}
.safety-card.good    { background: linear-gradient(135deg, #1b4332, #2d6a4f); }
.safety-card.caution { background: linear-gradient(135deg, #7b4f00, #b5770a); }
.safety-card.avoid   { background: linear-gradient(135deg, #7b1d1d, #b91c1c); }
.safety-card.unknown { background: #1e1e2e; }

.safety-day    { font-size: 0.7rem; color: #ccc; text-transform: uppercase; letter-spacing: 1px; }
.safety-date   { font-size: 0.85rem; font-weight: 600; color: #fff; margin: 0.2rem 0; }
.safety-rating { font-size: 1.4rem; margin: 0.3rem 0; }
.safety-label  { font-size: 0.65rem; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; }
.safety-label.good    { color: #6ee7b7; }
.safety-label.caution { color: #fbbf24; }
.safety-label.avoid   { color: #fca5a5; }

.stat-row {
    display: flex;
    gap: 0.4rem;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 0.4rem;
}
.stat-pill {
    font-size: 0.6rem;
    padding: 0.1rem 0.4rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.12);
    color: #e0e0e0;
}

.divider {
    border: none;
    border-top: 1px solid #2a2a3e;
    margin: 2.5rem 0;
}

.info-banner {
    background: #1a1a2e;
    border-left: 3px solid #4a9abb;
    padding: 0.8rem 1.2rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem;
    color: #a0b8c8;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MOUNTAIN CONFIGS
# ─────────────────────────────────────────────

MOUNTAINS = [
    {
        "id":        "ben_lomond",
        "name":      "Ben Lomond",
        "subtitle":  "",
        "module":    "ben_lomond_predictor",
        "lat":       56.1915,
        "lon":       -4.6346,
        "elevation": 974,
        "colour":    "#2196F3",
        "emoji":     "🔵",
    },
    {
        "id":        "cobbler",
        "name":      "The Cobbler",
        "subtitle":  "",
        "module":    "cobbler_predictor",
        "lat":       56.3647,
        "lon":       -4.7618,
        "elevation": 884,
        "colour":    "#E91E63",
        "emoji":     "🔴",
    },
    {
        "id":        "conic_hill",
        "name":      "Conic Hill",
        "subtitle":  "",
        "module":    "conic_predictor",
        "lat":       56.0971,
        "lon":       -4.5252,
        "elevation": 361,
        "colour":    "#4CAF50",
        "emoji":     "🟢",
    },
]

TARGET_LABELS = {
    "temperature_2m": ("Temperature", "°C",    "#E53935"),
    "wind_speed_10m": ("Wind Speed",  "km/h",  "#1E88E5"),
    "precipitation":  ("Precipitation","mm",   "#43A047"),
    "cloud_cover":    ("Cloud Cover", "%",     "#757575"),
}

WIND_DANGER  = 60
PRECIP_HIGH  = 5
CLOUD_HIGH   = 80
TEMP_FREEZE  = 2


# ─────────────────────────────────────────────
# HELPERS — load predictor module dynamically
# ─────────────────────────────────────────────

def load_predictor_module(module_name: str):
    """Import a predictor module by name, returning None if not found."""
    try:
        if module_name in sys.modules:
            return sys.modules[module_name]
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None


def run_pipeline(mountain: dict):
    """
    Run the full pipeline for one mountain.
    Returns (forecast_raw_df, predictions_df) or (None, None) on error.
    """
    mod = load_predictor_module(mountain["module"])
    if mod is None:
        return None, None

    try:
        client     = mod.build_openmeteo_client()
        fore_raw   = mod.fetch_forecast(client)
        fore       = mod.engineer_features(fore_raw)

        # Load or train models
        import os, joblib
        model_path = os.path.join(mod.OUTPUT_DIR, "models.pkl")
        if os.path.exists(model_path):
            bundle  = joblib.load(model_path)
            models  = bundle["models"]
            scaler  = bundle["scaler"]
            features= bundle["features"]
        else:
            hist_raw = mod.fetch_historical(mod.build_openmeteo_client())
            hist     = mod.engineer_features(hist_raw)
            models, scaler, features, _ = mod.train_models(hist)

        for f in features:
            if f not in fore.columns:
                fore[f] = 0.0

        predictions = mod.predict(models, scaler, features, fore)
        predictions["pred_precipitation"] = predictions["pred_precipitation"].clip(lower=0)
        predictions["pred_cloud_cover"]   = predictions["pred_cloud_cover"].clip(0, 100)
        return fore_raw, predictions

    except Exception as e:
        st.error(f"Pipeline error for {mountain['name']}: {e}")
        return None, None


# ─────────────────────────────────────────────
# SAFETY CARD HTML
# ─────────────────────────────────────────────

def daily_safety_html(predictions: pd.DataFrame) -> str:
    daily = predictions.resample("D").agg({
        "pred_temperature_2m": "mean",
        "pred_wind_speed_10m": "max",
        "pred_precipitation":  "sum",
        "pred_cloud_cover":    "mean",
    }).head(14)

    cards = []
    for date, row in daily.iterrows():
        issues = 0
        if row["pred_wind_speed_10m"] >= WIND_DANGER: issues += 1
        if row["pred_precipitation"]  >= PRECIP_HIGH: issues += 1
        if row["pred_cloud_cover"]    >= CLOUD_HIGH:  issues += 1
        if row["pred_temperature_2m"] <= TEMP_FREEZE: issues += 1

        if   issues == 0: css, emoji, label = "good",    "🟢", "GOOD"
        elif issues == 1: css, emoji, label = "caution", "🟡", "CAUTION"
        else:             css, emoji, label = "avoid",   "🔴", "AVOID"

        pills = (
            f'<span class="stat-pill">{row["pred_temperature_2m"]:.0f}°C</span>'
            f'<span class="stat-pill">{row["pred_wind_speed_10m"]:.0f}km/h</span>'
            f'<span class="stat-pill">{row["pred_precipitation"]:.1f}mm</span>'
        )

        cards.append(f"""
        <div class="safety-card {css}">
            <div class="safety-day">{date.strftime('%a')}</div>
            <div class="safety-date">{date.strftime('%d %b')}</div>
            <div class="safety-rating">{emoji}</div>
            <div class="safety-label {css}">{label}</div>
            <div class="stat-row">{pills}</div>
        </div>""")

    return f'<div class="safety-grid">{"".join(cards)}</div>'


# ─────────────────────────────────────────────
# FORECAST CHART (matplotlib → st.pyplot)
# ─────────────────────────────────────────────

def forecast_figure(fore_raw, predictions, mountain):
    colour = mountain["colour"]
    n_days = len(predictions.resample("D").size())
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), facecolor="#0d1117")
    fig.subplots_adjust(hspace=0.45, wspace=0.3)

    targets = [
        ("temperature_2m", "pred_temperature_2m"),
        ("wind_speed_10m", "pred_wind_speed_10m"),
        ("precipitation",  "pred_precipitation"),
        ("cloud_cover",    "pred_cloud_cover"),
    ]

    for ax, (raw_col, pred_col) in zip(axes.flat, targets):
        label, unit, line_colour = TARGET_LABELS[raw_col]
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

        # Uncertainty band
        ax.fill_between(predictions.index,
                        predictions[pred_col] * 0.88,
                        predictions[pred_col] * 1.12,
                        alpha=0.2, color=line_colour)

        # Model prediction
        ax.plot(predictions.index, predictions[pred_col],
                color=line_colour, linewidth=2, label="RF Model", zorder=3)

        # NWP reference
        if raw_col in fore_raw.columns:
            ax.plot(fore_raw.index, fore_raw[raw_col],
                    color="white", linewidth=0.9, linestyle="--",
                    alpha=0.4, label="Open-Meteo NWP", zorder=2)

        ax.set_title(f"{label} ({unit})", color="#e6edf3",
                     fontsize=10, fontweight="bold", pad=6)
        ax.tick_params(colors="#8b949e", labelsize=7.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.grid(True, color="#21262d", linewidth=0.7, zorder=0)
        ax.legend(fontsize=7, loc="upper right",
                  facecolor="#161b22", edgecolor="#30363d",
                  labelcolor="#8b949e")

        # Freeze line for temperature
        if raw_col == "temperature_2m":
            ax.axhline(0, color="#60a5fa", linewidth=0.8,
                       linestyle=":", alpha=0.7, label="0°C")

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# DEMO DATA (used when predictor modules absent)
# ─────────────────────────────────────────────

def make_demo_data(mountain: dict) -> tuple:
    """Generate plausible synthetic forecast data for UI preview."""
    rng   = np.random.default_rng(seed=abs(int(mountain["lat"] * 100)))
    dates = pd.date_range(datetime.now().floor("h"),
                          periods=14 * 24, freq="h", tz="Europe/London")
    elev_factor = mountain["elevation"] / 1000

    fore_raw = pd.DataFrame({
        "temperature_2m":  8  - elev_factor * 6  + rng.normal(0, 2, len(dates)).cumsum() * 0.02 + 3 * np.sin(np.linspace(0, 28*np.pi, len(dates))),
        "wind_speed_10m":  25 + elev_factor * 20 + rng.normal(0, 5, len(dates)),
        "precipitation":   np.clip(rng.exponential(0.3, len(dates)), 0, 15),
        "cloud_cover":     np.clip(60 + rng.normal(0, 20, len(dates)), 0, 100),
    }, index=dates)

    predictions = pd.DataFrame({
        "pred_temperature_2m": fore_raw["temperature_2m"] + rng.normal(0, 0.5, len(dates)),
        "pred_wind_speed_10m": fore_raw["wind_speed_10m"] + rng.normal(0, 2, len(dates)),
        "pred_precipitation":  np.clip(fore_raw["precipitation"] + rng.normal(0, 0.1, len(dates)), 0, None),
        "pred_cloud_cover":    np.clip(fore_raw["cloud_cover"]   + rng.normal(0, 3, len(dates)), 0, 100),
    }, index=dates)

    return fore_raw, predictions


# ─────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────

# ── Header ──────────────────────────────────
st.markdown(f"""
<div class="dashboard-header">
    <p class="dashboard-title">🏴󠁧󠁢󠁳󠁣󠁴󠁿 Big Hills Weather</p>
    <p class="dashboard-subtitle">Forecasts</p>
    <p class="dashboard-timestamp">Generated {datetime.now().strftime('%A %d %B %Y · %H:%M')}</p>
</div>
""", unsafe_allow_html=True)

# ── Per-mountain sections ────────────────────
for i, mountain in enumerate(MOUNTAINS):

    if i > 0:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Mountain header
    c = mountain["colour"]
    st.markdown(f"""
    <div class="mountain-header" style="border-color:{c}40">
        <div>
            <p class="mountain-name" style="color:{c}">{mountain['name']}</p>
            <p class="mountain-meta">
                {mountain['subtitle']} &nbsp;·&nbsp;
                {mountain['elevation']} m &nbsp;·&nbsp;
                {mountain['lat']}°N, {abs(mountain['lon'])}°W
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load data (real pipeline or demo)
    cache_key = f"data_{mountain['id']}"
    if cache_key not in st.session_state:
        mod = load_predictor_module(mountain["module"])
        if mod is not None:
            with st.spinner(f"Running forecast pipeline for {mountain['name']}…"):
                fore_raw, predictions = run_pipeline(mountain)
        else:
            # Show demo data with a notice
            st.info(
                f"ℹ️ `{mountain['module']}.py` not found — showing demo data. "
                f"Add the predictor script to enable live forecasts.",
                icon="📋"
            )
            fore_raw, predictions = make_demo_data(mountain)

        st.session_state[cache_key] = (fore_raw, predictions)
    else:
        fore_raw, predictions = st.session_state[cache_key]

    if predictions is None:
        st.error(f"Could not generate forecast for {mountain['name']}.")
        continue

    # ── Safety summary ───────────────────────
    st.markdown("**14-Day Safety Summary**")
    st.markdown(daily_safety_html(predictions), unsafe_allow_html=True)

    # ── Forecast charts ──────────────────────
    st.markdown("**Forecast Charts**")
    fig = forecast_figure(fore_raw, predictions, mountain)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


