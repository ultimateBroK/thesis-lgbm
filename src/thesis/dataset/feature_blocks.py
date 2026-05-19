"""Feature block composition — groups indicators into semantic categories."""

from __future__ import annotations

import polars as pl

from thesis.dataset.indicators import (
    add_adx,
    add_atr,
    add_atr_percentile,
    add_calendar,
    add_close_vs_vwap,
    add_ema_crossover,
    add_high_low_range,
    add_log_returns,
    add_macd,
    add_price_action,
    add_regime,
    add_rsi,
    add_session_dummies,
    add_tick_count_zscore,
    add_trend_regime,
    add_volatility_regime,
    add_volume_zscore,
    add_vwap,
)
from thesis.shared.config import Config


def add_return_features(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add return and short-horizon momentum features."""
    df = add_log_returns(df, config)
    df = add_rsi(df, config)
    return add_macd(df, config)


def add_trend_features(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add trend direction and trend-strength features."""
    df = add_ema_crossover(df, config)
    return add_adx(df, config)


def add_volatility_features(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add ATR-based volatility and range features."""
    df = add_atr(df, config)
    df = add_atr_percentile(df, config)
    return add_high_low_range(df, config)


def add_position_features(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add price-location features normalized by recent range or ATR."""
    df = add_price_action(df, config)
    df = add_vwap(df)
    return add_close_vs_vwap(df, config)


def add_microstructure_features(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add tick-derived volume and activity features."""
    df = add_volume_zscore(df, config)
    return add_tick_count_zscore(df, config)


def add_time_features(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add session and calendar context features."""
    df = add_session_dummies(df)
    return add_calendar(df, config)


def add_optional_regime_features(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add regime features when enabled by config."""
    if not config.features.enable_regime_features:
        return df
    df = add_volatility_regime(df, config)
    df = add_trend_regime(df, config)
    return add_regime(df, config)


def create_model_features(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Create minimal causal model features from OHLCV+ bars."""
    df = add_volatility_features(df, config)
    df = add_return_features(df, config)
    df = add_trend_features(df, config)
    df = add_position_features(df, config)
    df = add_microstructure_features(df, config)
    df = add_time_features(df, config)
    return add_optional_regime_features(df, config)
