import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))

import pandas as pd
from pathlib import Path
from src.utils.helpers        import load_config, setup_logger, ensure_dirs
from src.data.fetch_data      import build_market_data
from src.features.build_features  import build_all_features
from src.models.chronos_model     import ChronosForecaster
from src.evaluation.evaluate_model import evaluate

logger = setup_logger("run_forecast")


def load_prices(cfg):
    raw_dir = Path(cfg["data"]["raw_dir"])
    prices  = {}
    for ticker in cfg["assets"]:
        path = raw_dir / f"{ticker.replace('-', '_')}.csv"
        df   = pd.read_csv(path,  parse_dates=["date"])
        df   = df.set_index("date").sort_index()
        prices[ticker] = df["close"]
    return prices


def main():
    cfg = load_config("configs/config.yaml")

    ensure_dirs(
        cfg["data"]["raw_dir"],
        cfg["data"]["processed_dir"],
        cfg["output"]["plots_dir"],
        cfg["output"]["forecasts_dir"],
    )

    logger.info("=" * 50)
    logger.info("STEP 1/4 — Downloading market data")
    build_market_data(cfg)

    logger.info("=" * 50)
    logger.info("STEP 2/4 — Building features")
    prices_dict = load_prices(cfg)
    build_all_features(cfg)

    logger.info("=" * 50)
    logger.info(f"STEP 3/4 — Loading {cfg['model']['name']}")
    forecaster = ChronosForecaster(cfg)

    logger.info("=" * 50)
    logger.info("STEP 4/4 — Evaluating and forecasting")
    summary, forecasts = evaluate(cfg, prices_dict, forecaster)

    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION SUMMARY")
    print(summary.to_string())

    logger.info("\n14-DAY PRICE FORECASTS")
    for df in forecasts:
        ticker = df["asset"].iloc[0]
        print(f"\n{'─'*50}")
        print(f"  {ticker}")
        print(f"{'─'*50}")
        print(df[[
            "date", "median_price",
            "lower_price", "upper_price"
        ]].to_string(index=False))

    logger.info("\n✓ All outputs saved to outputs/")


if __name__ == "__main__":
    main()