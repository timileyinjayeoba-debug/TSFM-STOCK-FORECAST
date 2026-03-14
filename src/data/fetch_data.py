import yfinance as yf
import pandas as pd
from pathlib import Path
from src.utils.helpers import load_config, setup_logger, ensure_dirs

logger = setup_logger(__name__)


def fetch_asset(ticker, start, end, interval):
    logger.info(f"Downloading {ticker}...")
    df = yf.download(ticker, start=start, end=end,
                     interval=interval, auto_adjust=True)
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]

    df.index.name = "date"
    return df


def build_market_data(cfg):
    raw_dir = Path(cfg["data"]["raw_dir"])
    ensure_dirs(raw_dir, cfg["data"]["processed_dir"])

    frames = []
    for ticker in cfg["assets"]:
        df = fetch_asset(
            ticker,
            start=cfg["data"]["start_date"],
            end=cfg["data"]["end_date"],
            interval=cfg["data"]["interval"],
        )
        # Save each asset as its own CSV
        raw_path = raw_dir / f"{ticker.replace('-', '_')}.csv"
        df.to_csv(raw_path)
        logger.info(f"Saved {raw_path} ({len(df)} rows)")

        df["asset"] = ticker
        frames.append(df[["close", "volume", "asset"]])

    # Combine both into one market_data.csv
    market = pd.concat(frames, axis=0).reset_index()
    out = Path(cfg["data"]["processed_dir"]) / "market_data.csv"
    market.to_csv(out, index=False)
    logger.info(f"Combined market data saved → {out}")
    return market


if __name__ == "__main__":
    cfg = load_config()
    build_market_data(cfg)