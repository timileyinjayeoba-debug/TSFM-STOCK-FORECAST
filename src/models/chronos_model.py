import torch
import numpy as np
import pandas as pd
from chronos import ChronosPipeline
from src.utils.helpers import (
    load_config, setup_logger, compute_log_returns, inverse_log_return
)

logger = setup_logger(__name__)


class ChronosForecaster:

    def __init__(self, cfg):
        self.cfg = cfg
        model_id = cfg["model"]["name"]
        device   = cfg["model"]["device"]

        logger.info(f"Loading {model_id} on {device} ...")
        self.pipeline = ChronosPipeline.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.float32,
        )
        logger.info("Chronos loaded ✓")

    def predict_returns(self, context, horizon=14, num_samples=100):
        """
        Feed the model a context window, get back
        median + lower/upper uncertainty bands.
        """
        ctx_tensor = torch.tensor(
            context, dtype=torch.float32
        ).unsqueeze(0)                          # shape: (1, context_len)

        forecast = self.pipeline.predict(
        ctx_tensor,
        prediction_length=horizon,
        num_samples=num_samples,
        limit_prediction_length=False,
)
        samples = forecast[0].numpy()           # shape: (num_samples, horizon)

        median = np.median(samples, axis=0)
        lower  = np.percentile(samples, 10, axis=0)
        upper  = np.percentile(samples, 90, axis=0)
        return median, lower, upper

    def forecast_asset(self, prices, label="Asset"):
        """
        Given a full price series, forecast the next 14 days.
        Returns a DataFrame with predicted prices + returns.
        """
        ctx_len    = self.cfg["features"]["context_length"]
        horizon    = self.cfg["features"]["prediction_horizon"]
        n_samples  = self.cfg["model"]["num_samples"]

        # Take the last 60 days of log returns as context
        log_returns = compute_log_returns(prices)
        context     = log_returns.values[-ctx_len:]
        last_price  = prices.iloc[-1]
        last_date   = prices.index[-1]

        logger.info(f"{label}: forecasting {horizon} days from {last_date.date()}")
        median, lower, upper = self.predict_returns(
            context, horizon, n_samples
        )

        # Build future business dates
        future_dates = pd.bdate_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon
        )

        # Convert log returns back to price levels
        median_px = inverse_log_return(last_price, median)
        lower_px  = inverse_log_return(last_price, lower)
        upper_px  = inverse_log_return(last_price, upper)

        return pd.DataFrame({
            "date":          future_dates,
            "asset":         label,
            "median_return": median,
            "lower_return":  lower,
            "upper_return":  upper,
            "median_price":  median_px,
            "lower_price":   lower_px,
            "upper_price":   upper_px,
        })

    def backtest(self, prices, label="Asset", test_frac=0.2):
        """
        Rolling 1-step-ahead backtest over the test set.
        Returns (y_true, y_pred) so we can measure accuracy.
        """
        ctx_len  = self.cfg["features"]["context_length"]
        log_rets = compute_log_returns(prices).values
        split    = int(len(log_rets) * (1 - test_frac))

        y_true, y_pred = [], []
        total = len(log_rets) - split
        for i in range(split, len(log_rets)):
            context = log_rets[max(0, i - ctx_len) : i]
            if len(context) < ctx_len:
                continue
            med, _, _ = self.predict_returns(
                context, horizon=1,
                num_samples=self.cfg["model"]["num_samples"]
            )
            y_true.append(log_rets[i])
            y_pred.append(med[0])

            done = i - split + 1
            if done % 50 == 0:
                logger.info(f"  {label} backtest: {done}/{total} steps")

        logger.info(f"{label}: backtest complete — {len(y_true)} steps")
        return np.array(y_true), np.array(y_pred)