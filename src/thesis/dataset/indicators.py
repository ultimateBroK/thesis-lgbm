"""Polars-native technical indicators.

Each add_* function appends derived columns to a Polars DataFrame.
ATR MUST be computed first — many indicators normalize by ATR to stay
scale-invariant across price regimes.
"""

from __future__ import annotations

import polars as pl

from thesis.shared.config import Config
from thesis.shared.constants import FEATURE_EPS, STD_EPS


def _true_range() -> pl.Expr:
    return pl.max_horizontal(
        [
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low") - pl.col("close").shift(1)).abs(),
        ]
    )


def _wilder_smooth(expr: pl.Expr, period: int) -> pl.Expr:
    return expr.ewm_mean(alpha=1.0 / period, adjust=False)


def _ensure_utc(ts: pl.Expr, df: pl.DataFrame) -> pl.Expr:
    if df["timestamp"].dtype.time_zone is None:
        ts = ts.dt.replace_time_zone("UTC")
    return ts


def _ny_trading_day(df: pl.DataFrame) -> pl.Expr:
    """XAU/USD session resets at 5PM NY — anchor VWAP accumulation here."""
    ts = _ensure_utc(pl.col("timestamp"), df).dt.convert_time_zone("America/New_York")
    return (ts + pl.duration(hours=7)).dt.truncate("1d")


def add_atr(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Wilder ATR + close-normalized ATR.

    MUST RUN FIRST — many indicators divide by ATR.
    """
    p = config.features.atr_period
    atr = _true_range().ewm_mean(alpha=1.0 / p, adjust=False)
    return df.with_columns(
        [
            atr.alias(f"atr_{p}"),
            (atr / (pl.col("close").abs() + FEATURE_EPS)).alias("atr_pct_close"),
        ]
    )


def add_rsi(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Wilder RSI, 0–100."""
    p = config.features.rsi_period
    delta = pl.col("close").diff()
    gain = delta.clip(lower_bound=0.0)
    loss = (-delta).clip(lower_bound=0.0)
    avg_gain = gain.ewm_mean(alpha=1.0 / p, adjust=False)
    avg_loss = loss.ewm_mean(alpha=1.0 / p, adjust=False)
    rs = avg_gain / (avg_loss + FEATURE_EPS)
    return df.with_columns((100.0 - 100.0 / (1.0 + rs)).alias(f"rsi_{p}"))


def add_adx(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Wilder ADX — trend strength from +DI/-DI convergence."""
    period = config.features.adx_period
    tr = _true_range()
    up = pl.col("high") - pl.col("high").shift(1)
    down = pl.col("low").shift(1) - pl.col("low")
    plus_dm = pl.when((up > down) & (up > 0)).then(up).otherwise(0.0)
    minus_dm = pl.when((down > up) & (down > 0)).then(down).otherwise(0.0)
    atr_s = _wilder_smooth(tr, period)
    plus_dm_s = _wilder_smooth(plus_dm, period)
    minus_dm_s = _wilder_smooth(minus_dm, period)
    plus_di = 100.0 * plus_dm_s / (atr_s + FEATURE_EPS)
    minus_di = 100.0 * minus_dm_s / (atr_s + FEATURE_EPS)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + FEATURE_EPS)
    return df.with_columns(_wilder_smooth(dx, period).alias(f"adx_{period}"))


def add_macd(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """MACD histogram normalized by ATR for cross-asset comparability."""
    fast = config.features.macd_fast
    slow = config.features.macd_slow
    sig = config.features.macd_signal
    ema_fast = pl.col("close").ewm_mean(span=fast, adjust=False)
    ema_slow = pl.col("close").ewm_mean(span=slow, adjust=False)
    macd = ema_fast - ema_slow
    hist = macd - macd.ewm_mean(span=sig, adjust=False)
    atr_col = pl.col(f"atr_{config.features.atr_period}")
    return df.with_columns(
        [
            hist.alias("macd_hist"),
            (hist / (atr_col + FEATURE_EPS)).alias("macd_hist_atr"),
        ]
    )


def add_atr_percentile(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Rolling ATR rank — relative volatility within lookback window."""
    p = config.features.atr_period
    w = config.features.multi_timeframe.atr_percentile_window
    return df.with_columns(
        (pl.col(f"atr_{p}").rolling_rank(window_size=w, method="average") / w).alias(
            "atr_percentile"
        )
    )


def add_ema_slope(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Percent-change slope of EMA over shift window."""
    p = config.features.ema_slope_period
    shift_n = config.features.multi_timeframe.ema_slope_shift
    ema = pl.col("close").ewm_mean(span=p, adjust=False)
    slope = (ema - ema.shift(shift_n)) / (ema.shift(shift_n).abs() + FEATURE_EPS)
    return df.with_columns(slope.alias(f"ema_slope_{p}"))


def add_ema_crossover(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """ATR-normalized EMA distances — scale-invariant across price regimes."""
    p = config.features.atr_period
    atr = pl.col(f"atr_{p}")
    fc = config.features.ema_fast_span
    sc = config.features.ema_slow_span
    ema_f = pl.col("close").ewm_mean(span=fc, adjust=False)
    ema_s = pl.col("close").ewm_mean(span=sc, adjust=False)
    return df.with_columns(
        [
            ((pl.col("close") - ema_f) / (atr + FEATURE_EPS)).alias("close_vs_ema_34"),
            ((ema_f - ema_s) / (atr + FEATURE_EPS)).alias("ema34_vs_ema89"),
        ]
    )


def add_price_action(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Candle structure: body/wick ratios, gap, consecutive bars, price position."""
    p = config.features.atr_period
    mt = config.features.multi_timeframe
    atr = pl.col(f"atr_{p}")
    hl = pl.col("high") - pl.col("low") + FEATURE_EPS
    open_close_max = pl.max_horizontal([pl.col("open"), pl.col("close")])
    open_close_min = pl.min_horizontal([pl.col("open"), pl.col("close")])
    return df.with_columns(
        [
            ((pl.col("close") - pl.col("open")).abs() / hl).alias("candle_body_ratio"),
            ((pl.col("high") - open_close_max) / hl).alias("upper_wick_ratio"),
            ((open_close_min - pl.col("low")) / hl).alias("lower_wick_ratio"),
            ((pl.col("open") - pl.col("close").shift(1)) / (atr + FEATURE_EPS)).alias(
                "gap_ratio"
            ),
            (
                pl.when(pl.col("close") > pl.col("open"))
                .then(1)
                .when(pl.col("close") < pl.col("open"))
                .then(-1)
                .otherwise(0)
                .rolling_sum(window_size=mt.consecutive_bars_window)
            ).alias("consecutive_bars"),
            (
                (
                    pl.col("close")
                    - pl.col("low").rolling_min(window_size=mt.price_position_window)
                )
                / (
                    pl.col("high").rolling_max(window_size=mt.price_position_window)
                    - pl.col("low").rolling_min(window_size=mt.price_position_window)
                    + FEATURE_EPS
                )
            ).alias("price_position_20"),
        ]
    )


def add_vwap(df: pl.DataFrame) -> pl.DataFrame:
    """Session VWAP anchored to 5PM NY open — gold market session boundary."""
    td = _ny_trading_day(df)
    tp = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
    return (
        df.with_columns(td.alias("_td"))
        .with_columns(
            [
                (tp * pl.col("volume")).cum_sum().over("_td").alias("_cum_tpv"),
                pl.col("volume").cum_sum().over("_td").alias("_cum_vol"),
            ]
        )
        .with_columns(
            (pl.col("_cum_tpv") / (pl.col("_cum_vol") + FEATURE_EPS)).alias("vwap")
        )
        .drop(["_td", "_cum_tpv", "_cum_vol"])
    )


def add_close_vs_vwap(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """ATR-normalized close-to-VWAP distance — institutional benchmark deviation."""
    atr = pl.col(f"atr_{config.features.atr_period}")
    return df.with_columns(
        ((pl.col("close") - pl.col("vwap")) / (atr + FEATURE_EPS)).alias(
            "close_vs_vwap_atr"
        )
    )


def add_session_dummies(df: pl.DataFrame) -> pl.DataFrame:
    """NY/London/Asia session flags.

    Liquidity and spread patterns differ by session.
    """
    ny = (
        _ensure_utc(pl.col("timestamp"), df)
        .dt.convert_time_zone("America/New_York")
        .dt.hour()
    )
    return df.with_columns(
        [
            (
                (ny.is_in(list(range(18, 24))) | ny.is_in(list(range(0, 2)))).cast(
                    pl.Int8
                )
            ).alias("sess_asia"),
            ny.is_in(list(range(3, 8))).cast(pl.Int8).alias("sess_london"),
            ny.is_in(list(range(8, 12))).cast(pl.Int8).alias("sess_ny_am"),
            ny.is_in(list(range(12, 18))).cast(pl.Int8).alias("sess_ny_pm"),
        ]
    )


def add_volume_zscore(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Rolling z-score of volume — detects abnormal activity."""
    n = config.features.multi_timeframe.volume_zscore_period
    vol_mean = pl.col("volume").rolling_mean(window_size=n)
    vol_std = pl.col("volume").rolling_std(window_size=n)
    return df.with_columns(
        ((pl.col("volume") - vol_mean) / (vol_std + FEATURE_EPS)).alias(
            "volume_zscore_20"
        )
    )


def add_tick_count_zscore(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Rolling z-score of bar tick counts — detects microstructure regime shifts."""
    n = config.features.multi_timeframe.volume_zscore_period
    tick_mean = pl.col("tick_count").rolling_mean(window_size=n)
    tick_std = pl.col("tick_count").rolling_std(window_size=n)
    return df.with_columns(
        ((pl.col("tick_count") - tick_mean) / (tick_std + STD_EPS)).alias(
            "tick_count_zscore_20"
        )
    )


def add_log_returns(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Multi-horizon log returns — captures momentum at different time scales."""
    cols = []
    name_map = {1: "return_1h", 4: "return_4h", 24: "return_1d"}
    for lb in config.features.multi_timeframe.return_lookbacks:
        name = name_map.get(lb, f"return_{lb}b")
        cols.append((pl.col("close") / pl.col("close").shift(lb)).log().alias(name))
    return df.with_columns(cols)


def add_high_low_range(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """ATR-normalized rolling high-low range — volatility compression/expansion."""
    p = config.features.atr_period
    n = config.features.multi_timeframe.range_lookback
    rh = pl.col("high").rolling_max(window_size=n)
    rl = pl.col("low").rolling_min(window_size=n)
    return df.with_columns(
        ((rh - rl) / (pl.col(f"atr_{p}") + FEATURE_EPS)).alias("high_low_range_20")
    )


def add_volatility_regime(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """ATR percentile bucketed 0/1/2 (low/normal/high) — regime-conditional signals."""
    p33 = config.features.multi_timeframe.vol_regime_p33
    p66 = config.features.multi_timeframe.vol_regime_p66
    atr_pct = pl.col("atr_percentile")
    regime = (
        pl.when(atr_pct >= p66)
        .then(pl.lit(2))
        .when(atr_pct >= p33)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .cast(pl.Float64)
    )
    return df.with_columns(regime.alias("volatility_regime"))


def add_trend_regime(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """EMA slope × ADX level → -2..2 scale — trend direction and conviction."""
    adx_col = f"adx_{config.features.adx_period}"
    slope_col = f"ema_slope_{config.features.ema_slope_period}"
    if adx_col not in df.columns or slope_col not in df.columns:
        return df.with_columns(pl.lit(0.0).alias("trend_regime"))
    threshold = config.features.adx_regime_threshold
    adx = pl.col(adx_col)
    slope = pl.col(slope_col)
    regime = (
        pl.when((slope > 0) & (adx >= threshold))
        .then(pl.lit(2))
        .when((slope > 0) & (adx < threshold))
        .then(pl.lit(1))
        .when((slope < 0) & (adx < threshold))
        .then(pl.lit(-1))
        .when((slope < 0) & (adx >= threshold))
        .then(pl.lit(-2))
        .otherwise(pl.lit(0))
        .cast(pl.Float64)
    )
    return df.with_columns(regime.alias("trend_regime"))


def add_regime(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Composite ADX signal × EMA slope sign — single regime-strength feature."""
    adx_cols = [c for c in df.columns if c.startswith("adx_")]
    slope_cols = [c for c in df.columns if c.startswith("ema_slope_")]
    if not adx_cols or not slope_cols:
        return df.with_columns(pl.lit(0.0).alias("regime_strength"))
    threshold = config.features.adx_regime_threshold
    clip_max = config.features.adx_regime_clip_max
    adx_sig = ((pl.col(adx_cols[0]) - threshold) / threshold).clip(0, clip_max)
    slope_sig = pl.col(slope_cols[0]).sign()
    return df.with_columns((adx_sig * slope_sig).alias("regime_strength"))


def add_calendar(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Day of week — gold exhibits strong day-of-week seasonal patterns."""
    ts = _ensure_utc(pl.col("timestamp"), df).dt.convert_time_zone(
        config.data.market_tz
    )
    return df.with_columns(ts.dt.weekday().alias("day_of_week"))
