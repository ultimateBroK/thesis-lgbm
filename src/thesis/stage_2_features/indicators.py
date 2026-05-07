"""Polars-native indicator helpers for stage 2 feature engineering."""

from __future__ import annotations

import polars as pl

from thesis.shared.config import Config
from thesis.shared.constants import FEATURE_EPS, STD_EPS


def _compute_atr_expr(period: int) -> pl.Expr:
    """Compute Wilder-smoothed ATR expression."""
    tr = pl.max_horizontal(
        [
            (pl.col("high") - pl.col("low")),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low") - pl.col("close").shift(1)).abs(),
        ]
    )
    return tr.ewm_mean(alpha=1.0 / period, adjust=False)


def _add_rsi(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add Wilder RSI column."""
    p = config.features.rsi_period
    delta = pl.col("close").diff()
    gain = delta.clip(lower_bound=0.0)
    loss = (-delta).clip(lower_bound=0.0)
    avg_gain = gain.ewm_mean(alpha=1.0 / p, adjust=False)
    avg_loss = loss.ewm_mean(alpha=1.0 / p, adjust=False)
    rs = avg_gain / (avg_loss + FEATURE_EPS)
    return df.with_columns((100.0 - 100.0 / (1.0 + rs)).alias(f"rsi_{p}"))


def _add_atr(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add ATR and close-normalized ATR columns."""
    p = config.features.atr_period
    atr_expr = _compute_atr_expr(p)
    return df.with_columns(
        [
            atr_expr.alias(f"atr_{p}"),
            (atr_expr / (pl.col("close").abs() + FEATURE_EPS)).alias("atr_pct_close"),
        ]
    )


def _add_macd(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add MACD histogram and ATR-normalized MACD histogram."""
    fast = config.features.macd_fast
    slow = config.features.macd_slow
    sig = config.features.macd_signal
    ema_fast = pl.col("close").ewm_mean(span=fast, adjust=False)
    ema_slow = pl.col("close").ewm_mean(span=slow, adjust=False)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm_mean(span=sig, adjust=False)
    hist = macd_line - signal_line
    return df.with_columns(hist.alias("macd_hist")).with_columns(
        (
            pl.col("macd_hist")
            / (pl.col(f"atr_{config.features.atr_period}") + FEATURE_EPS)
        ).alias("macd_hist_atr")
    )


def _add_context_features(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add ATR ratio, price distance, pivot/session, and ATR percentile."""
    p = config.features.atr_period
    atr_5 = _compute_atr_expr(5)
    atr_20 = _compute_atr_expr(20)
    df = df.with_columns((atr_5 / (atr_20 + FEATURE_EPS)).alias("atr_ratio"))
    ema_89 = pl.col("close").ewm_mean(span=89, adjust=False)
    df = df.with_columns(
        ((pl.col("close") - ema_89) / (pl.col(f"atr_{p}") + FEATURE_EPS)).alias(
            "price_dist_ratio"
        )
    )
    df = _add_pivot_position(df)
    df = _add_ny_session_dummies(df)
    return df.with_columns(
        (
            pl.col(f"atr_{p}").rolling_rank(window_size=50, method="average") / 50.0
        ).alias("atr_percentile")
    )


def _add_pivot_position(df: pl.DataFrame) -> pl.DataFrame:
    """Add bounded pivot position derived from previous NY trading day."""
    trading_day_expr = _to_ny_trading_day(df)
    pivots = _build_pivot_table(df, trading_day_expr)
    df = df.with_columns(trading_day_expr.alias("_trading_day"))
    df = df.join(
        pivots, left_on="_trading_day", right_on="_trading_day", how="left"
    ).drop("_trading_day")
    return _compute_pivot_position(df)


def _to_ny_trading_day(df: pl.DataFrame) -> pl.Expr:
    """Convert timestamp to 7PM-anchored NY trading-day bucket."""
    ts = pl.col("timestamp")
    if df["timestamp"].dtype.time_zone is None:
        ts = ts.dt.replace_time_zone("UTC")
    ts_ny = ts.dt.convert_time_zone("America/New_York")
    return (ts_ny + pl.duration(hours=7)).dt.truncate("1d")


def _build_pivot_table(df: pl.DataFrame, trading_day_expr: pl.Expr) -> pl.DataFrame:
    """Build previous-day pivot, R1, and S1 lookup table."""
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
    """Compute clipped pivot position from previous S1/R1 bounds."""
    return df.with_columns(
        (
            (pl.col("close") - pl.col("prev_s1"))
            / (pl.col("prev_r1") - pl.col("prev_s1") + FEATURE_EPS)
        )
        .clip(0.0, 1.0)
        .alias("pivot_position")
    ).drop(["prev_pivot", "prev_r1", "prev_s1"])


def _add_ny_session_dummies(df: pl.DataFrame) -> pl.DataFrame:
    """Add NY session dummy columns."""
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


def _add_price_action_features(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add candle structure and short-run price action features."""
    p = config.features.atr_period
    atr_col = pl.col(f"atr_{p}")
    hl_range = pl.col("high") - pl.col("low") + FEATURE_EPS
    return df.with_columns(
        [
            ((pl.col("close") - pl.col("open")).abs() / hl_range).alias(
                "candle_body_ratio"
            ),
            (
                (pl.col("high") - pl.max_horizontal([pl.col("open"), pl.col("close")]))
                / hl_range
            ).alias("upper_wick_ratio"),
            (
                (pl.min_horizontal([pl.col("open"), pl.col("close")]) - pl.col("low"))
                / hl_range
            ).alias("lower_wick_ratio"),
            (
                (pl.col("open") - pl.col("close").shift(1)) / (atr_col + FEATURE_EPS)
            ).alias("gap_ratio"),
            (
                pl.when(pl.col("close") > pl.col("open"))
                .then(1)
                .when(pl.col("close") < pl.col("open"))
                .then(-1)
                .otherwise(0)
                .rolling_sum(window_size=5)
            ).alias("consecutive_bars"),
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
    """Add ATR-normalized EMA34/EMA89 distance features."""
    p = config.features.atr_period
    atr_col = pl.col(f"atr_{p}")
    ema_34_expr = pl.col("close").ewm_mean(span=34, adjust=False)
    ema_89_expr = pl.col("close").ewm_mean(span=89, adjust=False)
    return df.with_columns(
        [
            ((pl.col("close") - ema_34_expr) / (atr_col + FEATURE_EPS)).alias(
                "close_vs_ema_34"
            ),
            ((ema_34_expr - ema_89_expr) / (atr_col + FEATURE_EPS)).alias(
                "ema34_vs_ema89"
            ),
        ]
    )


def _add_volume_zscore(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add rolling volume z-score feature."""
    n = config.features.multi_timeframe.volume_zscore_period
    vol_mean = pl.col("volume").rolling_mean(window_size=n)
    vol_std = pl.col("volume").rolling_std(window_size=n)
    return df.with_columns(
        ((pl.col("volume") - vol_mean) / (vol_std + FEATURE_EPS)).alias(
            "volume_zscore_20"
        )
    )


def _add_ohlcv_norm(df: pl.DataFrame) -> pl.DataFrame:
    """Add rolling z-score normalized OHLC price columns."""
    window = 20
    return df.with_columns(
        [
            (
                (pl.col(col) - pl.col(col).rolling_mean(window_size=window))
                / (pl.col(col).rolling_std(window_size=window) + STD_EPS)
            ).alias(f"{col}_norm")
            for col in ["open", "high", "low", "close"]
        ]
    )


def _add_log_returns(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add multi-horizon log return features."""
    cols: list[pl.Expr] = []
    for lookback in config.features.multi_timeframe.return_lookbacks:
        ret = (pl.col("close") / pl.col("close").shift(lookback)).log()
        name = {1: "return_1h", 4: "return_4h", 24: "return_1d"}.get(
            lookback, f"return_{lookback}b"
        )
        cols.append(ret.alias(name))
    return df.with_columns(cols)


def _add_high_low_range(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add ATR-normalized rolling high-low range feature."""
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


def _add_adx(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add Wilder ADX trend-strength feature."""
    period = config.features.adx_period
    alpha = 1.0 / period
    tr = pl.max_horizontal(
        [
            (pl.col("high") - pl.col("low")),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low") - pl.col("close").shift(1)).abs(),
        ]
    )
    up_move = pl.col("high") - pl.col("high").shift(1)
    down_move = pl.col("low").shift(1) - pl.col("low")
    plus_dm = (
        pl.when((up_move > down_move) & (up_move > 0)).then(up_move).otherwise(0.0)
    )
    minus_dm = (
        pl.when((down_move > up_move) & (down_move > 0)).then(down_move).otherwise(0.0)
    )
    atr_smooth = tr.ewm_mean(alpha=alpha, adjust=False)
    plus_dm_smooth = plus_dm.ewm_mean(alpha=alpha, adjust=False)
    minus_dm_smooth = minus_dm.ewm_mean(alpha=alpha, adjust=False)
    plus_di = 100.0 * plus_dm_smooth / (atr_smooth + FEATURE_EPS)
    minus_di = 100.0 * minus_dm_smooth / (atr_smooth + FEATURE_EPS)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + FEATURE_EPS)
    return df.with_columns(
        dx.ewm_mean(alpha=alpha, adjust=False).alias(f"adx_{period}")
    )


def _add_ema_slope(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add five-bar percent-change slope of EMA."""
    p = config.features.ema_slope_period
    ema_expr = pl.col("close").ewm_mean(span=p, adjust=False)
    slope = (ema_expr - ema_expr.shift(5)) / (ema_expr.shift(5).abs() + FEATURE_EPS)
    return df.with_columns(slope.alias(f"ema_slope_{p}"))


def _add_regime(df: pl.DataFrame) -> pl.DataFrame:
    """Add composite regime strength from ADX and EMA slope."""
    adx_cols = [c for c in df.columns if c.startswith("adx_")]
    slope_cols = [c for c in df.columns if c.startswith("ema_slope_")]
    if not adx_cols or not slope_cols:
        return df.with_columns(pl.lit(0.0).alias("regime_strength"))
    adx_signal = ((pl.col(adx_cols[0]) - 20) / 20).clip(0, 3)
    slope_sign = pl.col(slope_cols[0]).sign()
    return df.with_columns((adx_signal * slope_sign).alias("regime_strength"))
