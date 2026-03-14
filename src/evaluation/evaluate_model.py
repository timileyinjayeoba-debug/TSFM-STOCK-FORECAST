import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from src.utils.helpers import (
    load_config, setup_logger, ensure_dirs, directional_accuracy
)

logger = setup_logger(__name__)


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def print_metrics(label, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r   = rmse(y_true, y_pred)
    da  = directional_accuracy(y_true, y_pred)
    logger.info(f"\n{'─'*46}")
    logger.info(f"  {label}")
    logger.info(f"  Test set size      : {len(y_true)}")
    logger.info(f"  MAE                : {mae:.4f}")
    logger.info(f"  RMSE               : {r:.4f}")
    logger.info(f"  Directional Accuracy: {da:.4f}")
    logger.info(f"{'─'*46}")
    return {"MAE": mae, "RMSE": r, "Directional Accuracy": da}


def plot_actual_vs_predicted(y_true, y_pred, label, out_dir):
    """
    The chart that matches your VSCode screenshot —
    blue = actual returns, orange = predicted returns.
    """
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(y_true, color="#1f77b4", lw=0.9, alpha=0.9, label="Actual")
    ax.plot(y_pred, color="#ff7f0e", lw=1.0, alpha=0.85, label="Predicted")
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.set_title(
        f"Actual vs Predicted Returns — {label}",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Test sample index")
    ax.set_ylabel("Log return")
    ax.legend(loc="upper right", framealpha=0.7)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    path = Path(out_dir) / f"{label.replace('-','_')}_backtest.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved → {path}")


def plot_forecast(prices, forecast_df, label, out_dir):
    """
    Two-panel chart:
    Left  — historical price + 14-day forecast with uncertainty band
    Right — bar chart of daily predicted returns
    """
    history = prices.iloc[-60:]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left panel: Price ──────────────────────────────────────
    ax = axes[0]
    ax.plot(history.index, history.values,
            color="#1f77b4", lw=1.5, label="Historical price")
    ax.plot(forecast_df["date"], forecast_df["median_price"],
            color="#ff7f0e", lw=2, marker="o", ms=4, label="Median forecast")
    ax.fill_between(
        forecast_df["date"],
        forecast_df["lower_price"],
        forecast_df["upper_price"],
        color="#ff7f0e", alpha=0.2, label="80% confidence band"
    )
    ax.axvline(history.index[-1], color="grey", lw=1, ls="--")
    ax.set_title(f"{label} — 14-Day Price Forecast", fontweight="bold")
    ax.set_ylabel("Price (USD)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # ── Right panel: Returns ───────────────────────────────────
    ax2 = axes[1]
    colors = ["#2ca02c" if r > 0 else "#d62728"
              for r in forecast_df["median_return"]]
    ax2.bar(range(len(forecast_df)),
            forecast_df["median_return"],
            color=colors, alpha=0.8, label="Median daily return")
    ax2.fill_between(
        range(len(forecast_df)),
        forecast_df["lower_return"],
        forecast_df["upper_return"],
        alpha=0.25, color="#9467bd", label="80% CI"
    )
    ax2.axhline(0, color="grey", lw=0.8, ls="--")
    ax2.set_xticks(range(len(forecast_df)))
    ax2.set_xticklabels(
        [d.strftime("%b %d") for d in forecast_df["date"]],
        rotation=40, ha="right", fontsize=7
    )
    ax2.set_title(f"{label} — Daily Return Forecast", fontweight="bold")
    ax2.set_ylabel("Log return")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)

    plt.suptitle(
        f"Chronos TSFM  •  {label}  •  Next 14 Trading Days",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()

    path = Path(out_dir) / f"{label.replace('-','_')}_forecast_14d.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {path}")


def evaluate(cfg, prices_dict, forecaster):
    plots_dir     = cfg["output"]["plots_dir"]
    forecasts_dir = cfg["output"]["forecasts_dir"]
    ensure_dirs(plots_dir, forecasts_dir)

    all_metrics   = {}
    all_forecasts = []

    for ticker, prices in prices_dict.items():

        # 1. Backtest — test on historical data we already have
        y_true, y_pred = forecaster.backtest(
            prices, label=ticker,
            test_frac=cfg["evaluation"]["test_split"]
        )
        metrics = print_metrics(ticker, y_true, y_pred)
        all_metrics[ticker] = metrics
        plot_actual_vs_predicted(y_true, y_pred, ticker, plots_dir)

        # 2. Forward forecast — predict the REAL next 14 days
        fcast_df = forecaster.forecast_asset(prices, label=ticker)
        plot_forecast(prices, fcast_df, ticker, plots_dir)

        # 3. Save forecast to CSV
        csv_path = Path(forecasts_dir) / \
            f"{ticker.replace('-','_')}_14d_forecast.csv"
        fcast_df.to_csv(csv_path, index=False)
        logger.info(f"Saved forecast CSV → {csv_path}")
        all_forecasts.append(fcast_df)

    # 4. Save summary metrics
    summary = pd.DataFrame(all_metrics).T
    summary.to_csv(Path(forecasts_dir) / "evaluation_summary.csv")
    return summary, all_forecasts