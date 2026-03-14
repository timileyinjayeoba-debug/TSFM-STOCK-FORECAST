import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.helpers import (
    load_config, setup_logger, ensure_dirs, compute_log_returns
)

logger = setup_logger(__name__)


def build_sequences(prices, context_len, horizon=1):
    """
    Slide a window across the return series.
    Each sample:
      X[i] = log-returns for days  i  to  i+context_len
      y[i] = log-return on day  i+context_len+1
    """
    returns = compute_log_returns(prices).values
    X, y = [], []
    for i in range(len(returns) - context_len - horizon + 1):
        X.append(returns[i : i + context_len])
        y.append(returns[i + context_len : i + context_len + horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_all_features(cfg):
    proc_dir = Path(cfg["data"]["processed_dir"])
    ensure_dirs(proc_dir)

    market = pd.read_csv(
        proc_dir / "market_data.csv", parse_dates=["date"]
    )
    ctx     = cfg["features"]["context_length"]
    horizon = cfg["features"]["prediction_horizon"]

    all_X, all_y = [], []

    for ticker in cfg["assets"]:
        sub    = market[market["asset"] == ticker].sort_values("date")
        prices = sub.set_index("date")["close"]

        X, y = build_sequences(prices, ctx, horizon)
        logger.info(f"{ticker}: {X.shape[0]} windows — X{X.shape}  y{y.shape}")

        all_X.append(X)
        all_y.append(y)

    X_all = np.vstack(all_X)
    y_all = np.vstack(all_y)

    np.save(proc_dir / "X.npy", X_all)
    np.save(proc_dir / "y.npy", y_all)

    logger.info(f"Saved X{X_all.shape} and y{y_all.shape} → {proc_dir}")
    return X_all, y_all


if __name__ == "__main__":
    cfg = load_config()
    build_all_features(cfg)