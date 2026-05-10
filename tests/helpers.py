"""Shared test fixtures and synthetic data generators.

Provides parameterized helpers for creating OHLCV, feature, and labeled
DataFrames used across unit and integration tests.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from thesis.shared.config import Config


def _interval_to_minutes(interval: str) -> int:
    """Convert interval string like '1h', '4h', '15m', '1d' to minutes."""
    unit = interval[-1].lower()
    amount = int(interval[:-1])
    multipliers = {"m": 1, "h": 60, "d": 1440}
    if unit not in multipliers:
        raise ValueError(f"Unsupported interval unit: {unit!r}")
    return amount * multipliers[unit]


def create_synthetic_ohlcv(
    n_rows: int = 300,
    seed: int = 42,
    base_price: float = 1800.0,
    start: str = "2023-01-01",
    interval: str = "1h",
) -> pl.DataFrame:
    """Create synthetic OHLCV data for testing.

    Default 300 rows provides enough warmup for regime features (Hurst/FD
    need 100-bar windows) and multi-timeframe resampling.

    Args:
        n_rows: Number of bars to generate.
        seed: Random seed for reproducibility.
        base_price: Starting price for the random walk.
        start: ISO date string for the first timestamp.
        interval: Bar interval (e.g. '1h', '4h', '15m', '1d').

    Returns:
        DataFrame with columns: timestamp (UTC), open, high, low, close, volume.
    """
    np.random.seed(seed)

    parts = start.split("-")
    year, month, day = int(parts[0]), int(parts[1]), int(parts[2])

    total_minutes = _interval_to_minutes(interval) * (n_rows - 1)
    timestamps = pl.datetime_range(
        start=pl.datetime(year, month, day, time_zone="UTC"),
        end=pl.datetime(year, month, day, time_zone="UTC")
        + pl.duration(minutes=total_minutes),
        interval=interval,
        eager=True,
    )

    # Random walk prices
    returns = np.random.normal(0, 0.001, n_rows)
    closes = base_price * np.exp(np.cumsum(returns))

    # OHLC from close
    opens = closes * (1 + np.random.normal(0, 0.0005, n_rows))
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.001, n_rows)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.001, n_rows)))
    volumes = np.random.randint(1000, 10000, n_rows).astype(float)

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


def create_synthetic_features(
    config: Config,
    n_rows: int = 300,
    seed: int = 42,
) -> pl.DataFrame:
    """Create a DataFrame with OHLCV + ATR and basic feature columns.

    Lightweight alternative to running the full feature pipeline. Produces
    valid numeric values suitable for model and backtest tests.

    Args:
        config: Pipeline config (used for ATR period and feature columns).
        n_rows: Number of rows to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with OHLCV, atr_14, and common feature columns filled with
        random valid values.
    """
    np.random.seed(seed)
    df = create_synthetic_ohlcv(n_rows=n_rows, seed=seed)

    atr_period = config.features.atr_period
    atr_col = f"atr_{atr_period}"

    # Compute a realistic ATR from synthetic prices
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )
    atr_vals = np.full(n_rows, np.mean(tr[1:]) if n_rows > 1 else 1.0)

    # Generate basic feature columns with valid ranges
    rng = np.random.default_rng(seed)
    feature_data: dict[str, np.ndarray] = {
        atr_col: atr_vals,
        "return_1h": rng.normal(0, 0.001, n_rows),
        "return_4h": rng.normal(0, 0.002, n_rows),
        "rsi_14": rng.uniform(20, 80, n_rows),
        "adx_14": rng.uniform(15, 50, n_rows),
        "macd_hist_atr": rng.normal(0, 0.5, n_rows),
        "atr_pct_close": atr_vals / df["close"].to_numpy(),
        "atr_ratio": rng.uniform(0.5, 1.5, n_rows),
        "atr_percentile": rng.uniform(0, 1, n_rows),
        "ema_slope_20": rng.normal(0, 0.001, n_rows),
        "regime_strength": rng.uniform(-1, 1, n_rows),
        "ema34_vs_ema89": rng.normal(0, 0.01, n_rows),
        "close_vs_ema_34": rng.normal(0, 0.01, n_rows),
        "price_position_20": rng.uniform(0, 1, n_rows),
        "price_dist_ratio": rng.normal(0, 0.5, n_rows),
        "pivot_position": rng.uniform(0, 1, n_rows),
        "candle_body_ratio": rng.uniform(0, 1, n_rows),
        "upper_wick_ratio": rng.uniform(0, 0.3, n_rows),
        "lower_wick_ratio": rng.uniform(0, 0.3, n_rows),
        "high_low_range_20": rng.uniform(0.005, 0.03, n_rows),
        "volume_zscore_20": rng.normal(0, 1, n_rows),
        "vwap": df["close"].to_numpy() + rng.normal(0, 0.2, n_rows),
        "sess_asia": rng.integers(0, 2, n_rows).astype(float),
        "sess_london": rng.integers(0, 2, n_rows).astype(float),
        "sess_ny_am": rng.integers(0, 2, n_rows).astype(float),
        "sess_ny_pm": rng.integers(0, 2, n_rows).astype(float),
    }

    return df.with_columns([pl.Series(k, v) for k, v in feature_data.items()])


def create_synthetic_labeled_data(
    config: Config,
    n_rows: int = 300,
    seed: int = 42,
) -> pl.DataFrame:
    """Create synthetic features + triple-barrier label columns.

    Combines ``create_synthetic_features`` with label arrays that mimic the
    output of the labeling stage without running the full pipeline.

    Args:
        config: Pipeline config (used for feature generation).
        n_rows: Number of rows to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with all feature columns plus: label, upper_barrier,
        lower_barrier, touched_bar, event_end, sample_weight.
    """
    np.random.seed(seed)
    df = create_synthetic_features(config, n_rows=n_rows, seed=seed)

    rng = np.random.default_rng(seed + 1)

    labels = rng.choice([-1, 0, 1], size=n_rows)
    close = df["close"].to_numpy()
    atr_col = f"atr_{config.features.atr_period}"
    atr = df[atr_col].to_numpy()

    upper_barrier = close + config.labels.barrier_atr_multiplier * atr
    lower_barrier = close - config.labels.barrier_atr_multiplier * atr

    # touched_bar: offset in bars when barrier was hit, -1 for hold
    horizon = config.labels.horizon_bars
    touched_bar = np.where(
        labels != 0,
        rng.integers(1, max(horizon, 2), n_rows),
        -1,
    ).astype(np.int32)

    # event_end: absolute bar index
    event_end = (
        np.arange(n_rows) + np.where(touched_bar < 0, horizon, touched_bar)
    ).astype(np.int32)

    # sample_weight: random positive weights normalized to mean ~1
    sample_weight = np.clip(rng.exponential(1.0, n_rows), 0.1, 5.0)
    sample_weight = sample_weight / sample_weight.mean()

    label_data: dict[str, np.ndarray] = {
        "label": labels.astype(np.int32),
        "upper_barrier": upper_barrier,
        "lower_barrier": lower_barrier,
        "touched_bar": touched_bar,
        "event_end": event_end,
        "sample_weight": sample_weight,
    }

    return df.with_columns([pl.Series(k, v) for k, v in label_data.items()])
