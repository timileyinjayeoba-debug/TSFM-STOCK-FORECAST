import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.helpers import (
    load_config, setup_logger, ensure_dirs, compute_log_returns
)

logger = setup_logger(__name__)


# ── Feature engineering ────────────────────────────────────────────────────────

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI on a 0-100 scale — computed manually, no extra libraries needed."""
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)   # fill early NaNs with neutral value


def compute_ema(close: pd.Series, period: int = 21) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw OHLCV DataFrame (columns: open, high, low, close, volume)
    and returns a DataFrame with 5 normalised features.

    Features
    --------
    log_return      : log(close_t / close_t-1) 
    high_low_range  : (high - low) / close
    volume_change   : log(volume_t / volume_t-1)
    rsi_14          : RSI scaled to [-1, 1]  (original 0-100 → subtract 50, divide 50)
    ema_ratio       : close / EMA(close, 21) - 1
    """
    features = pd.DataFrame(index=df.index)

    # 1. log_return
    features["log_return"]     = compute_log_returns(df["close"])

    # 2. high_low_range
    features["high_low_range"] = (df["high"] - df["low"]) / df["close"]

    # 3. volume_change
    vol_ratio = df["volume"] / df["volume"].shift(1).replace(0, np.nan)
    features["volume_change"] = np.log(vol_ratio.clip(lower=1e-8))

    # 4. rsi_14  → scaled to [-1, 1] so it sits on the same scale as returns
    rsi = compute_rsi(df["close"], period=14)
    features["rsi_14"]         = (rsi - 50) / 50

    # 5. ema_ratio
    ema = compute_ema(df["close"], period=21)
    features["ema_ratio"]      = (df["close"] / ema) - 1

    # Drop the first ~21 rows where indicators are still warming up
    features = features.iloc[21:].copy()

    # Forward-fill then drop any remaining NaNs
    features = features.ffill().dropna()

    return features


def scale_features(features: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Standardise each feature to mean=0, std=1.
    Returns the scaled DataFrame and a dict of (mean, std) per feature
    so we can invert the scaling later if needed.
    """
    stats = {}
    scaled = features.copy()
    for col in features.columns:
        mu          = features[col].mean()
        sigma       = features[col].std() + 1e-8   # avoid divide-by-zero
        scaled[col] = (features[col] - mu) / sigma
        stats[col]  = (mu, sigma)
    return scaled, stats


# ── Sequence builder ───────────────────────────────────────────────────────────

def build_sequences(
    df: pd.DataFrame,
    context_len: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window across the feature matrix.

    Parameters
    ----------
    df          : raw OHLCV DataFrame for one asset
    context_len : number of days in each input window  (e.g. 60)
    horizon     : number of days to predict            (e.g. 14)

    Returns
    -------
    X : (N, context_len, 5)   — input windows
    y : (N, horizon)          — future log-returns (target)
    """
    features, _ = scale_features(build_feature_matrix(df))
    log_returns  = features["log_return"].values
    feat_matrix  = features.values           # shape: (T, 5)

    X, y = [], []
    for i in range(len(feat_matrix) - context_len - horizon + 1):
        X.append(feat_matrix[i : i + context_len])                          # (60, 5)
        y.append(log_returns[i + context_len : i + context_len + horizon])  # (14,)

    X = np.array(X, dtype=np.float32)   # (N, 60, 5)
    y = np.array(y, dtype=np.float32)   # (N, 14)
    return X, y


# ── Pipeline entry point ───────────────────────────────────────────────────────

def build_all_features(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads market_data.csv, builds sequences for every asset in cfg,
    stacks them and saves X.npy / y.npy to the processed dir.
    """
    proc_dir = Path(cfg["data"]["processed_dir"])
    ensure_dirs(proc_dir)

    market  = pd.read_csv(proc_dir / "market_data.csv", parse_dates=["date"])
    ctx     = cfg["features"]["context_length"]
    horizon = cfg["features"]["prediction_horizon"]

    all_X, all_y = [], []

    for ticker in cfg["assets"]:
        sub = (
            market[market["asset"] == ticker]
            .sort_values("date")
            .set_index("date")
        )

        # make sure we have the OHLCV columns we need
        required = {"open", "high", "low", "close", "volume"}
        missing  = required - set(sub.columns)
        if missing:
            logger.warning(f"{ticker}: missing columns {missing} — skipping")
            continue

        X, y = build_sequences(sub, ctx, horizon)
        logger.info(f"{ticker}: {X.shape[0]} windows — X{X.shape}  y{y.shape}")

        all_X.append(X)
        all_y.append(y)

    X_all = np.vstack(all_X)   # (N_total, 60, 5)
    y_all = np.vstack(all_y)   # (N_total, 14)

    np.save(proc_dir / "X.npy", X_all)
    np.save(proc_dir / "y.npy", y_all)

    logger.info(f"Saved X{X_all.shape} and y{y_all.shape} → {proc_dir}")
    return X_all, y_all


if __name__ == "__main__":
    cfg = load_config("configs/config.yaml")
    build_all_features(cfg)