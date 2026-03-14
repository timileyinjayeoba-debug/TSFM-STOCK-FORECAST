import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
from datetime import datetime, timedelta, date
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TSFM Stock Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root {
    --bg: #0a0a0f; --surface: #12121a; --surface2: #1a1a26;
    --border: #2a2a3d; --accent: #00ff88; --accent2: #7c3aed;
    --text: #e2e8f0; --muted: #64748b;
    --danger: #ef4444; --success: #10b981;
}
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }
.metric-card {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px; text-align: center;
}
.metric-card:hover {
    border-color: var(--accent);
    box-shadow: 0 0 20px rgba(0,255,136,0.1);
}
.metric-value {
    font-family: 'Space Mono', monospace; font-size: 1.8rem;
    font-weight: 700; color: var(--accent);
}
.metric-label {
    font-size: 0.75rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px;
}
.metric-delta { font-size: 0.85rem; margin-top: 6px; }
.delta-pos { color: var(--success); }
.delta-neg { color: var(--danger); }
.section-header {
    font-family: 'Space Mono', monospace; font-size: 0.7rem;
    color: var(--accent); text-transform: uppercase;
    letter-spacing: 0.2em; border-bottom: 1px solid var(--border);
    padding-bottom: 8px; margin-bottom: 16px;
}
.stButton > button {
    background: linear-gradient(135deg, var(--accent2), #4f46e5) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important; padding: 12px 24px !important;
    width: 100% !important;
}
.hero-title {
    font-family: 'Space Mono', monospace; font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00ff88, #7c3aed);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { color: var(--muted); font-size: 0.9rem; margin-top: 8px; }
.tag {
    display: inline-block; background: rgba(0,255,136,0.1);
    border: 1px solid rgba(0,255,136,0.3); color: var(--accent);
    border-radius: 4px; padding: 2px 8px; font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# ── Project imports ────────────────────────────────────────────────────────────
try:
    from src.data.fetch_data import fetch_asset
    from src.models.chronos_model import ChronosForecaster
    from src.utils.helpers import load_config
    PROJECT_READY = True
except Exception as e:
    PROJECT_READY = False
    IMPORT_ERROR = str(e)

# ── Asset options ──────────────────────────────────────────────────────────────
ASSET_OPTIONS = {
    "Bitcoin (BTC-USD)":  "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Apple (AAPL)":       "AAPL",
    "Tesla (TSLA)":       "TSLA",
    "NVIDIA (NVDA)":      "NVDA",
    "S&P 500 (^GSPC)":    "^GSPC",
    "Gold (GC=F)":        "GC=F",
}


# ── Chart builder ──────────────────────────────────────────────────────────────
def build_forecast_chart(hist_df, forecast_df, ticker):
    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.75, 0.25],
        shared_xaxes=True, vertical_spacing=0.04
    )

    fig.add_trace(go.Scatter(
        x=hist_df.index, y=hist_df["close"],
        name="Historical",
        line=dict(color="#64748b", width=1.5),
        hovertemplate="%{x|%b %d, %Y}<br>$%{y:,.2f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df["date"], forecast_df["date"].iloc[::-1]]),
        y=pd.concat([forecast_df["upper_price"], forecast_df["lower_price"].iloc[::-1]]),
        fill="toself",
        fillcolor="rgba(124,58,237,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence Band",
        hoverinfo="skip",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["upper_price"],
        name="Upper", line=dict(color="#7c3aed", width=1, dash="dot"),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["lower_price"],
        name="Lower", line=dict(color="#7c3aed", width=1, dash="dot"),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["median_price"],
        name="Forecast",
        line=dict(color="#00ff88", width=2.5),
        mode="lines+markers",
        marker=dict(size=6, color="#00ff88"),
        hovertemplate="%{x|%b %d}<br>Forecast: $%{y:,.2f}<extra></extra>",
    ), row=1, col=1)

    if "volume" in hist_df.columns:
        fig.add_trace(go.Bar(
            x=hist_df.index, y=hist_df["volume"],
            name="Volume",
            marker_color="rgba(100,116,139,0.4)",
        ), row=2, col=1)

    fig.update_layout(
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#0a0a0f",
        font=dict(family="DM Sans", color="#e2e8f0"),
        legend=dict(
            bgcolor="rgba(18,18,26,0.8)",
            bordercolor="#2a2a3d",
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
        yaxis=dict(gridcolor="#1a1a26", tickprefix="$", tickformat=",.0f"),
        yaxis2=dict(gridcolor="#1a1a26"),
        xaxis=dict(gridcolor="#1a1a26"),
        xaxis2=dict(gridcolor="#1a1a26"),
    )
    return fig


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(forecast_df, last_price):
    if forecast_df.empty:
        return {
            "last_price": last_price, "forecast_end": np.nan,
            "pct_change": np.nan, "uncertainty": np.nan,
            "high": np.nan, "low": np.nan,
        }
    median_end  = forecast_df["median_price"].iloc[-1]
    pct_change  = ((median_end - last_price) / last_price) * 100
    high        = forecast_df["upper_price"].max()
    low         = forecast_df["lower_price"].min()
    uncertainty = (
        (forecast_df["upper_price"].iloc[-1] - forecast_df["lower_price"].iloc[-1])
        / median_end
    ) * 100
    return {
        "last_price": last_price,
        "forecast_end": median_end,
        "pct_change": pct_change,
        "uncertainty": uncertainty,
        "high": high,
        "low": low,
    }


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title">TSFM<br>FORECAST</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Powered by Amazon Chronos T5</div>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">⚙ Configuration</div>', unsafe_allow_html=True)

    asset_label = st.selectbox("Asset", list(ASSET_OPTIONS.keys()), index=0)
    ticker      = ASSET_OPTIONS[asset_label]
    horizon     = st.slider("Forecast Horizon (days)", 7, 30, 14, 1)
    lookback    = st.slider("Historical Lookback (days)", 90, 730, 365, 30)
    model_size  = st.selectbox(
        "Chronos Model Size",
        ["chronos-t5-tiny", "chronos-t5-small", "chronos-t5-base"],
        index=1,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🚀 RUN FORECAST")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">ℹ About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.78rem; color:#64748b; line-height:1.6;">
    Uses <strong style="color:#e2e8f0;">Amazon Chronos</strong> — a pretrained
    time-series transformer — for probabilistic price forecasts
    with confidence intervals.
    </div>
    """, unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:24px;">
    <div style="font-family:'Space Mono',monospace; font-size:1.4rem;
                font-weight:700; color:#e2e8f0;">{asset_label}</div>
    <span class="tag">LIVE FORECAST</span>
    <span class="tag">{horizon}D HORIZON</span>
</div>
""", unsafe_allow_html=True)

if not PROJECT_READY:
    st.error(f"Could not load project modules.\n\nError: {IMPORT_ERROR}")
    st.info("Run: `streamlit run app.py` from your TSFM-STOCK-FORECAST directory.")
    st.stop()


# ── Run forecast ───────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner(""):
        progress = st.progress(0, text="📡 Fetching market data...")
        try:
            end_date   = date.today().strftime("%Y-%m-%d")
            start_date = (date.today() - timedelta(days=lookback)).strftime("%Y-%m-%d")

            hist_df = fetch_asset(
                ticker, start=start_date, end=end_date, interval="1d"
            )
            progress.progress(25, text="✅ Data fetched. Loading Chronos model...")

            cfg = load_config("configs/config.yaml")
            cfg["model"]["name"] = f"amazon/{model_size}"
            cfg["features"]["prediction_horizon"] = horizon
            forecaster = ChronosForecaster(cfg)
            progress.progress(55, text="✅ Model loaded. Generating forecast...")

            # ── THE KEY FIX: squeeze + dropna + datetime index ──
            prices_series = hist_df["close"].squeeze()
            prices_series = prices_series.dropna()
            prices_series.index = pd.to_datetime(prices_series.index)

            forecast_df = forecaster.forecast_asset(prices_series, label=ticker)
            forecast_df = forecast_df.dropna().reset_index(drop=True)
            forecast_df["date"] = pd.to_datetime(forecast_df["date"])
            progress.progress(85, text="✅ Forecast complete. Rendering...")

            st.session_state["hist_df"]     = hist_df
            st.session_state["forecast_df"] = forecast_df
            st.session_state["ticker"]      = ticker
            st.session_state["asset_label"] = asset_label
            progress.progress(100, text="🎉 Done!")

        except Exception as e:
            st.error(f"Error during forecast: {e}")
            st.exception(e)
            st.stop()


# ── Results ────────────────────────────────────────────────────────────────────
if "forecast_df" in st.session_state:
    hist_df     = st.session_state["hist_df"]
    forecast_df = st.session_state["forecast_df"]
    ticker      = st.session_state["ticker"]
    asset_label = st.session_state["asset_label"]

    last_price = float(hist_df["close"].squeeze().dropna().iloc[-1])
    metrics    = compute_metrics(forecast_df, last_price)

    cols  = st.columns(5)
    cards = [
        ("Current Price",  f"${metrics['last_price']:,.2f}", None),
        ("Forecast End",   f"${metrics['forecast_end']:,.2f}",
         f"{'▲' if metrics['pct_change'] >= 0 else '▼'} {abs(metrics['pct_change']):.1f}%"),
        ("Projected High", f"${metrics['high']:,.2f}", None),
        ("Projected Low",  f"${metrics['low']:,.2f}", None),
        ("Uncertainty",    f"{metrics['uncertainty']:.1f}%", None),
    ]

    for col, (label, value, delta) in zip(cols, cards):
        delta_html = ""
        if delta:
            cls = "delta-pos" if "▲" in delta else "delta-neg"
            delta_html = f'<div class="metric-delta {cls}">{delta}</div>'
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
            {delta_html}
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📈 Price Forecast Chart</div>',
                unsafe_allow_html=True)
    fig = build_forecast_chart(hist_df.tail(120), forecast_df, ticker)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-header">📋 Forecast Data Table</div>',
                    unsafe_allow_html=True)
        display_df = forecast_df.copy()
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
        for c in ["median_price", "lower_price", "upper_price"]:
            display_df[c] = display_df[c].map("${:,.2f}".format)
        display_df.columns = [
            "Date", "Asset", "Median Return", "Lower Return",
            "Upper Return", "Median Price", "Lower Bound", "Upper Bound"
        ]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown('<div class="section-header">⬇ Export</div>',
                    unsafe_allow_html=True)
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name=f"{ticker}_forecast_{datetime.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown(f"""
        <div class="metric-card" style="text-align:left; margin-top:12px;">
            <div style="font-size:0.7rem; color:#64748b;
                        font-family:'Space Mono',monospace;
                        text-transform:uppercase; letter-spacing:0.1em;
                        margin-bottom:10px;">Run Info</div>
            <div style="font-size:0.8rem; line-height:1.8;">
                <span style="color:#64748b;">Asset:</span>
                <span style="color:#e2e8f0;"> {ticker}</span><br>
                <span style="color:#64748b;">Horizon:</span>
                <span style="color:#e2e8f0;"> {len(forecast_df)} days</span><br>
                <span style="color:#64748b;">Model:</span>
                <span style="color:#e2e8f0;"> {model_size}</span><br>
                <span style="color:#64748b;">Generated:</span>
                <span style="color:#e2e8f0;"> {datetime.now().strftime('%H:%M:%S')}</span>
            </div>
        </div>""", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding:80px 0;">
        <div style="font-size:4rem; margin-bottom:16px;">📊</div>
        <div style="font-family:'Space Mono',monospace; font-size:1.1rem;
                    color:#e2e8f0; margin-bottom:8px;">Ready to Forecast</div>
        <div style="color:#64748b; font-size:0.85rem; max-width:400px;
                    margin:0 auto; line-height:1.6;">
            Select an asset and horizon in the sidebar, then click
            <strong style="color:#7c3aed;">RUN FORECAST</strong> to generate
            AI-powered price predictions using Amazon Chronos.
        </div>
    </div>
    """, unsafe_allow_html=True)