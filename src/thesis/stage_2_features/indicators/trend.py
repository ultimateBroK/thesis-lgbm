"""Price-action, volume, return, and trend/regime indicator functions.

Pure Polars-native computations for candle structure, EMA crossovers,
volume z-scores, OHLCV normalization, log returns, range features,
ADX, EMA slope, and composite regime strength.
"""

from __future__ import annotations

import polars as pl

from thesis.shared.config import Config
from thesis.shared.constants import FEATURE_EPS, STD_EPS

# ── Price-action features ────────────────────────────────────────────


def _add_price_action_features(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add price-action candle and bar structure features.

    Adds candle body, wick, gap, consecutive-bar, and 20-bar price-position
    features derived from OHLCV and ATR values.

    Args:
        df: Input DataFrame with OHLCV and ATR columns.
        config: Configuration with ``features.atr_period``.

    Returns:
        DataFrame with additional price-action feature columns.
    """
    p = config.features.atr_period
    atr_col = pl.col(f"atr_{p}")
    hl_range = pl.col("high") - pl.col("low") + FEATURE_EPS

    return df.with_columns(
        [
            # Candle body strength
            ((pl.col("close") - pl.col("open")).abs() / hl_range).alias(
                "candle_body_ratio"
            ),
            # Upper wick — selling rejection
            (
                (pl.col("high") - pl.max_horizontal([pl.col("open"), pl.col("close")]))
                / hl_range
            ).alias("upper_wick_ratio"),
            # Lower wick — buying support
            (
                (pl.min_horizontal([pl.col("open"), pl.col("close")]) - pl.col("low"))
                / hl_range
            ).alias("lower_wick_ratio"),
            # Gap from previous close, normalized by ATR
            (
                (pl.col("open") - pl.col("close").shift(1)) / (atr_col + FEATURE_EPS)
            ).alias("gap_ratio"),
            # Consecutive direction streak (rolling 5 bars)
            (
                pl.when(pl.col("close") > pl.col("open"))
                .then(1)
                .when(pl.col("close") < pl.col("open"))
                .then(-1)
                .otherwise(0)
                .rolling_sum(window_size=5)
            ).alias("consecutive_bars"),
            # Position within 20-bar range: 0 = at low, 1 = at high
            (
                (pl.col("close") - pl.col("low").rolling_min(window_size=20))
                / (
                    pl.col("high").rolling_max(window_size=20)
                    - pl.col("low").rolling_min(window_size=20)
                    + FEATURE_EPS
                )
            ).alias("price_position_20"),
        ]
    )


def _add_ema_crossover(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add EMA 34/89 crossover features — user's preferred trading stack.

    Adds ``close_vs_ema_34`` and ``ema34_vs_ema89`` as ATR-normalized trend
    distance features.

    Args:
        df: Input DataFrame with OHLCV and ATR columns.
        config: Configuration with ``features.atr_period``.

    Returns:
        DataFrame with additional EMA crossover feature columns.
    """
    p = config.features.atr_period
    atr_col = pl.col(f"atr_{p}")

    ema_34 = pl.col("close").ewm_mean(span=34, adjust=False)
    ema_89 = pl.col("close").ewm_mean(span=89, adjust=False)

    return df.with_columns(
        [
            ((pl.col("close") - ema_34) / (atr_col + FEATURE_EPS)).alias(
                "close_vs_ema_34"
            ),
            ((ema_34 - ema_89) / (atr_col + FEATURE_EPS)).alias("ema34_vs_ema89"),
        ]
    )


# ── Volume features ──────────────────────────────────────────────────


def _add_volume_zscore(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add rolling volume z-score vs ``volume_zscore_period``-bar mean.

    Args:
        df: Input DataFrame with a ``volume`` column.
        config: Configuration with
            ``features.multi_timeframe.volume_zscore_period``.

    Returns:
        DataFrame with an added ``volume_zscore_20`` column.
    """
    n = config.features.multi_timeframe.volume_zscore_period
    vol_mean = pl.col("volume").rolling_mean(window_size=n)
    vol_std = pl.col("volume").rolling_std(window_size=n)
    return df.with_columns(
        ((pl.col("volume") - vol_mean) / (vol_std + FEATURE_EPS)).alias(
            "volume_zscore_20"
        )
    )


# ── Normalized OHLCV for GRU ────────────────────────────────────────


def _add_ohlcv_norm(df: pl.DataFrame) -> pl.DataFrame:
    """Add rolling z-score normalized OHLCV prices for GRU price-level awareness.

    The GRU sequence encoder receives only derived indicators by default.
    Adding normalized raw price columns (open, high, low, close) gives the
    GRU direct access to price levels and ranges, which is critical for
    regime detection — the GRU needs to see actual price dynamics, not just
    derived features.

    Each column is normalized as (value - rolling_mean_20) / (rolling_std_20 + eps)
    so the scale is comparable across features and time periods.

    Args:
        df: Input DataFrame with ``open``, ``high``, ``low``, ``close`` columns.

    Returns:
        DataFrame with added ``open_norm``, ``high_norm``, ``low_norm``,
        ``close_norm`` columns.
    """
    window = 20
    return df.with_columns(
        [
            (
                (pl.col("open") - pl.col("open").rolling_mean(window_size=window))
                / (pl.col("open").rolling_std(window_size=window) + STD_EPS)
            ).alias("open_norm"),
            (
                (pl.col("high") - pl.col("high").rolling_mean(window_size=window))
                / (pl.col("high").rolling_std(window_size=window) + STD_EPS)
            ).alias("high_norm"),
            (
                (pl.col("low") - pl.col("low").rolling_mean(window_size=window))
                / (pl.col("low").rolling_std(window_size=window) + STD_EPS)
            ).alias("low_norm"),
            (
                (pl.col("close") - pl.col("close").rolling_mean(window_size=window))
                / (pl.col("close").rolling_std(window_size=window) + STD_EPS)
            ).alias("close_norm"),
        ]
    )


# ── Log return features ──────────────────────────────────────────────


def _add_log_returns(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add log return features at multiple lookback horizons.

    Produces ``return_1h``, ``return_4h``, ``return_1d``.

    Args:
        df: Input DataFrame with a ``close`` column.
        config: Configuration with
            ``features.multi_timeframe.return_lookbacks``.

    Returns:
        DataFrame with additional log return columns.
    """
    cols: list[pl.Expr] = []
    for lookback in config.features.multi_timeframe.return_lookbacks:
        ret = (pl.col("close") / pl.col("close").shift(lookback)).log()
        name = {1: "return_1h", 4: "return_4h", 24: "return_1d"}.get(
            lookback, f"return_{lookback}b"
        )
        cols.append(ret.alias(name))
    return df.with_columns(cols)


# ── High-low range feature ───────────────────────────────────────────


def _add_high_low_range(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add normalized 20-bar high-low range: (max_high - min_low) / ATR14.

    Args:
        df: Input DataFrame with OHLCV and ATR columns.
        config: Configuration with ``features.atr_period`` and
            ``features.multi_timeframe.range_lookback``.

    Returns:
        DataFrame with an added ``high_low_range_20`` column.
    """
    p = config.features.atr_period
    atr_col = pl.col(f"atr_{p}")
    n = config.features.multi_timeframe.range_lookback
    rolling_high = pl.col("high").rolling_max(window_size=n)
    rolling_low = pl.col("low").rolling_min(window_size=n)
    return df.with_columns(
        ((rolling_high - rolling_low) / (atr_col + FEATURE_EPS)).alias(
            "high_low_range_20"
        )
    )


# ── Trend quality & regime ───────────────────────────────────────────


def _add_adx(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add ADX (Average Directional Index) — config-driven trend strength.

    ADX > 25 typically indicates a trending market; ADX < 20 suggests a
    range-bound market.  Uses Wilder smoothing with period from
    ``config.features.adx_period``.

    Args:
        df: Input OHLCV DataFrame.
        config: Configuration with ``features.adx_period``.

    Returns:
        DataFrame with an added ``adx_{p}`` column.
    """
    period = config.features.adx_period
    alpha = 1.0 / period

    # True Range
    tr = pl.max_horizontal(
        [
            (pl.col("high") - pl.col("low")),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low") - pl.col("close").shift(1)).abs(),
        ]
    )

    # +DM / -DM
    up_move = pl.col("high") - pl.col("high").shift(1)
    down_move = pl.col("low").shift(1) - pl.col("low")
    plus_dm = (
        pl.when((up_move > down_move) & (up_move > 0)).then(up_move).otherwise(0.0)
    )
    minus_dm = (
        pl.when((down_move > up_move) & (down_move > 0)).then(down_move).otherwise(0.0)
    )

    # Wilder smoothing
    atr_smooth = tr.ewm_mean(alpha=alpha, adjust=False)
    plus_dm_smooth = plus_dm.ewm_mean(alpha=alpha, adjust=False)
    minus_dm_smooth = minus_dm.ewm_mean(alpha=alpha, adjust=False)

    # +DI / -DI
    plus_di = 100.0 * plus_dm_smooth / (atr_smooth + FEATURE_EPS)
    minus_di = 100.0 * minus_dm_smooth / (atr_smooth + FEATURE_EPS)

    # DX → ADX
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + FEATURE_EPS)
    adx = dx.ewm_mean(alpha=alpha, adjust=False)

    return df.with_columns(adx.alias(f"adx_{period}"))


def _add_ema_slope(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add EMA slope — short-term rate-of-change of smooth trend line.

    Computes the 5-bar percentage change of an EMA span, yielding a
    directional signal: positive = rising trend, negative = declining
    trend, near-zero = flat.

    Args:
        df: Input OHLCV DataFrame.
        config: Configuration with ``features.ema_slope_period``.

    Returns:
        DataFrame with an added ``ema_slope_{p}`` column.
    """
    p = config.features.ema_slope_period
    ema = pl.col("close").ewm_mean(span=p, adjust=False)
    slope = (ema - ema.shift(5)) / (ema.shift(5).abs() + FEATURE_EPS)
    return df.with_columns(slope.alias(f"ema_slope_{p}"))


def _add_regime(df: pl.DataFrame) -> pl.DataFrame:
    """Add composite regime strength — ADX intensity × EMA slope direction.

    Combines the trend-strength (how strongly trending) with the EMA
    slope sign (which direction) into a single bipolar feature:

    - Positive = strong uptrend
    - Negative = strong downtrend
    - Near 0  = ranging / flat market

    Requires both ``adx_14`` and ``ema_slope_20`` columns already present
    in the DataFrame.

    Args:
        df: DataFrame with ``adx_14`` and ``ema_slope_20`` columns.

    Returns:
        DataFrame with an added ``regime_strength`` column.
    """
    adx = pl.col("adx_14")
    # ADX > 20 → trending signal grows; ADX <= 20 → 0 (ranging)
    adx_signal = ((adx - 20) / 20).clip(0, 3)
    # Direction: +1 for rising EMA, -1 for falling
    slope_sign = pl.col("ema_slope_20").sign()
    regime = adx_signal * slope_sign
    return df.with_columns(regime.alias("regime_strength"))
