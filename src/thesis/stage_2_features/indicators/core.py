"""Core and context indicator functions for feature engineering.

Pure Polars-native indicator computations for ATR, RSI, MACD, session
encoding, and pivot position.  Each function takes a DataFrame and Config,
appends new columns, and returns the enriched DataFrame.
"""

from __future__ import annotations

import polars as pl

from thesis.shared.config import Config
from thesis.shared.constants import FEATURE_EPS

# ── Shared ATR expression ────────────────────────────────────────────


def _compute_atr_expr(period: int) -> pl.Expr:
    """Compute ATR expression (Wilder-smoothed True Range).

    Args:
        period: Lookback period used for ATR smoothing.

    Returns:
        Polars expression that yields the ATR series for the specified period.
    """
    tr = pl.max_horizontal(
        [
            (pl.col("high") - pl.col("low")),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low") - pl.col("close").shift(1)).abs(),
        ]
    )
    return tr.ewm_mean(alpha=1.0 / period, adjust=False)


# ── Core indicators ──────────────────────────────────────────────────


def _add_rsi(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Compute Wilder-style RSI and append it as a column.

    Args:
        df: Input OHLCV dataframe containing a ``close`` column.
        config: Configuration with ``features.rsi_period``.

    Returns:
        DataFrame with an added column ``rsi_{p}``.
    """
    p = config.features.rsi_period
    delta = pl.col("close").diff()
    gain = delta.clip(lower_bound=0.0)
    loss = (-delta).clip(lower_bound=0.0)
    avg_gain = gain.ewm_mean(alpha=1.0 / p, adjust=False)
    avg_loss = loss.ewm_mean(alpha=1.0 / p, adjust=False)
    rs = avg_gain / (avg_loss + FEATURE_EPS)
    return df.with_columns((100.0 - 100.0 / (1.0 + rs)).alias(f"rsi_{p}"))


def _add_atr(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add ATR and close-normalized ATR.

    Args:
        df: Input OHLCV DataFrame.
        config: Configuration with ``features.atr_period``.

    Returns:
        DataFrame with ``atr_{p}`` and ``atr_pct_close``.
    """
    p = config.features.atr_period
    atr = _compute_atr_expr(p)
    return df.with_columns(
        [
            atr.alias(f"atr_{p}"),
            (atr / (pl.col("close").abs() + FEATURE_EPS)).alias("atr_pct_close"),
        ]
    )


def _add_macd(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add raw and ATR-normalized MACD histogram.

    Args:
        df: Input OHLCV DataFrame.
        config: Configuration with ``features.macd_fast``,
            ``features.macd_slow``, ``features.macd_signal``.

    Returns:
        DataFrame with ``macd_hist`` and ``macd_hist_atr`` columns.
    """
    fast = config.features.macd_fast
    slow = config.features.macd_slow
    sig = config.features.macd_signal
    atr = pl.col(f"atr_{config.features.atr_period}")
    ema_fast = pl.col("close").ewm_mean(span=fast, adjust=False)
    ema_slow = pl.col("close").ewm_mean(span=slow, adjust=False)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm_mean(span=sig, adjust=False)
    hist = macd_line - signal_line
    return df.with_columns(
        hist.alias("macd_hist"),
        (hist / (atr + FEATURE_EPS)).alias("macd_hist_atr"),
    )


# ── Context & session features ───────────────────────────────────────


def _add_context_features(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add derived, normalized, and regime/session features.

    Appends: ``atr_ratio``, ``price_dist_ratio``, ``pivot_position``,
    session dummies, and ``atr_percentile``.

    Args:
        df: Input DataFrame with OHLCV and ATR columns.
        config: Configuration with ``features.atr_period``.

    Returns:
        DataFrame with additional context feature columns.
    """
    p = config.features.atr_period

    # ATR Ratio: ATR(5) / ATR(20) — volatility regime
    atr_5 = _compute_atr_expr(5)
    atr_20 = _compute_atr_expr(20)
    df = df.with_columns((atr_5 / (atr_20 + FEATURE_EPS)).alias("atr_ratio"))

    # Price Distance Ratio: (Close - EMA89) / ATR14
    ema_89 = pl.col("close").ewm_mean(span=89, adjust=False)
    df = df.with_columns(
        ((pl.col("close") - ema_89) / (pl.col(f"atr_{p}") + FEATURE_EPS)).alias(
            "price_dist_ratio"
        )
    )

    # Pivot Position: (Close - S1) / (R1 - S1) — bounded [0,1]
    df = _add_pivot_position(df)

    # Session encoding in America/New_York timezone (DST-aware)
    df = _add_ny_session_dummies(df)

    # ATR Percentile: normalized rolling rank of ATR14 over 50 bars → [0, 1]
    df = df.with_columns(
        (
            pl.col(f"atr_{p}").rolling_rank(window_size=50, method="average") / 50.0
        ).alias("atr_percentile")
    )

    return df


def _add_pivot_position(df: pl.DataFrame) -> pl.DataFrame:
    """Compute previous-day pivot levels and add bounded pivot_position column.

    Args:
        df: Input OHLCV DataFrame with timestamp, high, low, close columns.

    Returns:
        DataFrame with an added ``pivot_position`` column.
    """
    trading_day_expr = _to_ny_trading_day(df)
    pivots = _build_pivot_table(df, trading_day_expr)
    df = df.with_columns(trading_day_expr.alias("_trading_day"))
    df = df.join(
        pivots, left_on="_trading_day", right_on="_trading_day", how="left"
    ).drop("_trading_day")
    return _compute_pivot_position(df)


def _to_ny_trading_day(df: pl.DataFrame) -> pl.Expr:
    """Convert timestamp column to NY trading-day expression.

    Args:
        df: Input DataFrame with a ``timestamp`` column (UTC or timezone-aware).

    Returns:
        Polars expression yielding trading-day timestamps in America/New_York
        timezone, shifted so each day starts at 7 PM ET.
    """
    ts = pl.col("timestamp")
    if df["timestamp"].dtype.time_zone is None:
        ts = ts.dt.replace_time_zone("UTC")
    ts_ny = ts.dt.convert_time_zone("America/New_York")
    return (ts_ny + pl.duration(hours=7)).dt.truncate("1d")


def _build_pivot_table(df: pl.DataFrame, trading_day_expr: pl.Expr) -> pl.DataFrame:
    """Build previous-day pivot/R1/S1 lookup table.

    Args:
        df: Input OHLCV DataFrame.
        trading_day_expr: Polars expression yielding trading-day timestamps.

    Returns:
        DataFrame with ``prev_pivot``, ``prev_r1``, ``prev_s1`` columns
        keyed by ``_trading_day``.
    """
    df_with_day = df.with_columns(trading_day_expr.alias("_trading_day"))
    daily = (
        df_with_day.group_by("_trading_day")
        .agg(
            [
                pl.col("high").max().alias("day_high"),
                pl.col("low").min().alias("day_low"),
                pl.col("close").last().alias("day_close"),
            ]
        )
        .sort("_trading_day")
    )
    pivot = (daily["day_high"] + daily["day_low"] + daily["day_close"]) / 3.0
    r1 = 2.0 * pivot - daily["day_low"]
    s1 = 2.0 * pivot - daily["day_high"]
    return (
        daily.with_columns([pivot.alias("pivot"), r1.alias("r1"), s1.alias("s1")])
        .select(["_trading_day", "pivot", "r1", "s1"])
        .with_columns(
            [
                pl.col("pivot").shift(1).alias("prev_pivot"),
                pl.col("r1").shift(1).alias("prev_r1"),
                pl.col("s1").shift(1).alias("prev_s1"),
            ]
        )
        .select(["_trading_day", "prev_pivot", "prev_r1", "prev_s1"])
    )


def _compute_pivot_position(df: pl.DataFrame) -> pl.DataFrame:
    """Compute bounded pivot_position and drop intermediate pivot columns.

    Args:
        df: DataFrame with ``close``, ``prev_s1``, ``prev_r1``,
            ``prev_pivot`` columns.

    Returns:
        DataFrame with ``pivot_position`` column replacing intermediate
        pivot columns.
    """
    return df.with_columns(
        (
            (pl.col("close") - pl.col("prev_s1"))
            / (pl.col("prev_r1") - pl.col("prev_s1") + FEATURE_EPS)
        )
        .clip(0.0, 1.0)
        .alias("pivot_position")
    ).drop(["prev_pivot", "prev_r1", "prev_s1"])


def _add_ny_session_dummies(df: pl.DataFrame) -> pl.DataFrame:
    """Add four NY session indicator columns (DST-aware).

    Adds: ``sess_asia``, ``sess_london``, ``sess_overlap``, ``sess_ny_pm``.

    Args:
        df: Input DataFrame with a ``timestamp`` column (UTC or timezone-aware).

    Returns:
        DataFrame with four session indicator columns.
    """
    ts = pl.col("timestamp")
    if df["timestamp"].dtype.time_zone is None:
        ts = ts.dt.replace_time_zone("UTC")

    ny_hour = ts.dt.convert_time_zone("America/New_York").dt.hour()

    return df.with_columns(
        [
            (ny_hour.is_in(list(range(18, 24))) | ny_hour.is_in(list(range(0, 2))))
            .cast(pl.Int8)
            .alias("sess_asia"),
            ny_hour.is_in(list(range(3, 8))).cast(pl.Int8).alias("sess_london"),
            ny_hour.is_in(list(range(8, 12))).cast(pl.Int8).alias("sess_overlap"),
            ny_hour.is_in(list(range(12, 18))).cast(pl.Int8).alias("sess_ny_pm"),
        ]
    )
