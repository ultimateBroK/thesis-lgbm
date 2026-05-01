"""Feature engineering — compact price-action + trend feature set.

The production feature pipeline intentionally stays small and interpretable for
student projects:
    - prioritize price structure and trend distance over stacked indicators
    - avoid strongly redundant transforms of the same signal
    - keep runtime low and behavior stable across runs
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl

from thesis.config import Config
from thesis.constants import EXCLUDE_COLS as _EXCLUDE_COLS

logger = logging.getLogger("thesis.features")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_features(config: Config) -> None:
    """Generate and persist feature-enriched OHLCV bars.

    Loads OHLCV data from ``config.paths.ohlcv``, computes technical
    indicators and normalized/session features, forward-fills remaining
    nulls (then replaces any remaining nulls with 0.0), writes the
    enriched bars to ``config.paths.features``, and saves a sidecar JSON
    file listing the produced feature column names.

    Args:
        config: Application configuration containing input/output paths
            and feature parameters.

    Raises:
        FileNotFoundError: If the OHLCV parquet file does not exist.
    """
    ohlcv_path = Path(config.paths.ohlcv)
    if not ohlcv_path.exists():
        raise FileNotFoundError(f"OHLCV not found: {ohlcv_path}")

    logger.info("Loading OHLCV: %s", ohlcv_path)
    df = pl.read_parquet(ohlcv_path)
    logger.info("Input bars: %d", len(df))
    _validate_ohlcv_input(df, config)

    # --- Core price-volatility anchor ---
    df = _add_atr(df, config)

    # --- Price-action + session context ---
    df = _add_new_features(df, config)

    # --- Price-action structure ---
    df = _add_price_action_features(df, config)
    df = _add_ema_crossover(df, config)

    df = _add_log_returns(df, config)
    df = _add_high_low_range(df, config)

    # --- Trend quality ---
    df = add_trend_strength(df)

    # --- Minimal indicators ---
    df = _add_rsi(df, config)
    df = _add_macd(df, config)
    df = _add_volume_zscore(df, config)

    # Backward compatibility: GRU pipeline may request `log_returns`.
    if "return_1h" in df.columns and "log_returns" not in df.columns:
        df = df.with_columns(pl.col("return_1h").alias("log_returns"))

    # Keep only compact model-facing features to avoid redundant columns.
    keep_features = sorted(
        {
            *config.features.static_feature_cols,
            *config.gru.feature_cols,
        }
    )
    keep_cols = ["timestamp", "open", "high", "low", "close", "volume", *keep_features]
    existing_keep_cols = [c for c in keep_cols if c in df.columns]
    df = df.select(existing_keep_cols)
    df = _drop_warmup_rows(df, keep_features)

    # Persist
    out_path = Path(config.paths.features)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)

    feature_cols = sorted(c for c in df.columns if c not in _EXCLUDE_COLS)
    _save_feature_list(out_path, feature_cols)

    logger.info(
        "Features saved: %s (%d columns, %d rows)", out_path, len(df.columns), len(df)
    )
    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols)


# ---------------------------------------------------------------------------
# Shared ATR expression helper
# ---------------------------------------------------------------------------


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


def _timeframe_to_ms(timeframe: str) -> int:
    """Parse a small timeframe string into milliseconds for validation."""
    tf = timeframe.upper()
    if tf.endswith("H"):
        return int(tf[:-1]) * 3_600_000
    if tf.endswith("MIN"):
        return int(tf[:-3]) * 60_000
    if tf.endswith("M"):
        return int(tf[:-1]) * 60_000
    if tf in ("D", "1D"):
        return 86_400_000
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def _validate_ohlcv_input(df: pl.DataFrame, config: Config) -> None:
    """Log timestamp continuity checks before rolling feature generation."""
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"OHLCV missing required columns: {missing}")
    if df.is_empty():
        raise ValueError("OHLCV is empty; cannot generate features")

    unsorted = int(
        df.select((pl.col("timestamp").diff().dt.total_milliseconds() < 0).sum()).item()
        or 0
    )
    duplicate_count = len(df) - df.get_column("timestamp").n_unique()
    if unsorted > 0:
        raise ValueError(f"OHLCV timestamps are not sorted ({unsorted} reversals)")
    if duplicate_count > 0:
        raise ValueError(
            f"OHLCV timestamps are not unique ({duplicate_count} duplicates)"
        )

    if len(df) < 2:
        return

    expected_ms = _timeframe_to_ms(config.data.timeframe)
    deltas = (
        df.select(
            (pl.col("timestamp").diff().dt.total_milliseconds()).alias("delta_ms")
        )
        .drop_nulls()
        .get_column("delta_ms")
    )
    gaps = deltas.filter(deltas > expected_ms)
    largest = int(deltas.max() or 0)
    logger.info(
        "Feature input gap check: expected_delta=%d ms, gap_count=%d, "
        "largest_gap=%.2f bars",
        expected_ms,
        len(gaps),
        largest / expected_ms if expected_ms else 0.0,
    )


def _drop_warmup_rows(df: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
    """Drop rows whose model-facing features are still null/NaN after warm-up."""
    existing_features = [c for c in feature_cols if c in df.columns]
    n_before = len(df)
    df = df.fill_nan(None).drop_nulls(subset=existing_features)
    dropped = n_before - len(df)
    if dropped > 0:
        logger.info(
            "Dropped %d warm-up rows with incomplete model-facing features",
            dropped,
        )
    if df.is_empty():
        raise ValueError("No feature rows remain after dropping warm-up rows")
    return df


# ---------------------------------------------------------------------------
# Core indicator helpers (all Polars-native)
# ---------------------------------------------------------------------------


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
    rs = avg_gain / (avg_loss + 1e-10)
    return df.with_columns((100.0 - 100.0 / (1.0 + rs)).alias(f"rsi_{p}"))


def _add_atr(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add ATR column to the DataFrame.

    Returns:
        DataFrame with a new column ``atr_{p}``.
    """
    p = config.features.atr_period
    atr = _compute_atr_expr(p)
    return df.with_columns(atr.alias(f"atr_{p}"))


def _add_macd(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add a ``macd_hist`` column using configured MACD spans.

    Returns:
        DataFrame with an added ``macd_hist`` column.
    """
    fast = config.features.macd_fast
    slow = config.features.macd_slow
    sig = config.features.macd_signal
    ema_fast = pl.col("close").ewm_mean(span=fast, adjust=False)
    ema_slow = pl.col("close").ewm_mean(span=slow, adjust=False)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm_mean(span=sig, adjust=False)
    return df.with_columns(
        (macd_line - signal_line).alias("macd_hist"),
    )


# ---------------------------------------------------------------------------
# New normalized features
# ---------------------------------------------------------------------------


def _add_new_features(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add derived, normalized, and regime/session features.

    Appends: ``atr_ratio``, ``price_dist_ratio``, ``pivot_position``,
    session dummies, and ``atr_percentile``.
    """
    p = config.features.atr_period

    # ATR Ratio: ATR(5) / ATR(20) — volatility regime
    atr_5 = _compute_atr_expr(5)
    atr_20 = _compute_atr_expr(20)
    df = df.with_columns((atr_5 / (atr_20 + 1e-10)).alias("atr_ratio"))

    # Price Distance Ratio: (Close - EMA89) / ATR14
    ema_89 = pl.col("close").ewm_mean(span=89, adjust=False)
    df = df.with_columns(
        ((pl.col("close") - ema_89) / (pl.col(f"atr_{p}") + 1e-10)).alias(
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
    """Compute previous-day pivot levels and add bounded pivot_position column."""
    trading_day_expr = _to_ny_trading_day(df)
    pivots = _build_pivot_table(df, trading_day_expr)
    df = df.with_columns(trading_day_expr.alias("_trading_day"))
    df = df.join(
        pivots, left_on="_trading_day", right_on="_trading_day", how="left"
    ).drop("_trading_day")
    return _compute_pivot_position(df)


def _to_ny_trading_day(df: pl.DataFrame) -> pl.Expr:
    """Convert timestamp column to NY trading-day expression."""
    ts = pl.col("timestamp")
    if df["timestamp"].dtype.time_zone is None:
        ts = ts.dt.replace_time_zone("UTC")
    ts_ny = ts.dt.convert_time_zone("America/New_York")
    return (ts_ny + pl.duration(hours=7)).dt.truncate("1d")


def _build_pivot_table(df: pl.DataFrame, trading_day_expr: pl.Expr) -> pl.DataFrame:
    """Build previous-day pivot/R1/S1 lookup table."""
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
    """Compute bounded pivot_position and drop intermediate pivot columns."""
    return df.with_columns(
        (
            (pl.col("close") - pl.col("prev_s1"))
            / (pl.col("prev_r1") - pl.col("prev_s1") + 1e-10)
        )
        .clip(0.0, 1.0)
        .alias("pivot_position")
    ).drop(["prev_pivot", "prev_r1", "prev_s1"])


# ---------------------------------------------------------------------------
# Price-action candle and bar structure features
# ---------------------------------------------------------------------------


def _add_price_action_features(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add price-action candle and bar structure features.

    Features:
        candle_body_ratio: |close - open| / (high - low) — candle strength
        upper_wick_ratio: (high - max(open,close)) / (high - low) — rejection signal
        lower_wick_ratio: (min(open,close) - low) / (high - low) — support signal
        gap_ratio: (open - prev_close) / ATR14 — gap detection
        consecutive_bars: rolling sum of bar direction (±1) over 5 bars — momentum streak
        price_position_20: (close - low_20) / (high_20 - low_20) — position in N-bar range
    """
    p = config.features.atr_period
    atr_col = pl.col(f"atr_{p}")
    hl_range = pl.col("high") - pl.col("low") + 1e-10

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
            ((pl.col("open") - pl.col("close").shift(1)) / (atr_col + 1e-10)).alias(
                "gap_ratio"
            ),
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
                    + 1e-10
                )
            ).alias("price_position_20"),
        ]
    )


def _add_ema_crossover(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add EMA 34/89 crossover features — user's preferred trading stack.

    Features:
        close_vs_ema_34: (close - EMA34) / ATR14 — price distance from EMA 34
        ema34_vs_ema89: (EMA34 - EMA89) / ATR14 — crossover signal / trend direction
    """
    p = config.features.atr_period
    atr_col = pl.col(f"atr_{p}")

    ema_34 = pl.col("close").ewm_mean(span=34, adjust=False)
    ema_89 = pl.col("close").ewm_mean(span=89, adjust=False)

    return df.with_columns(
        [
            ((pl.col("close") - ema_34) / (atr_col + 1e-10)).alias("close_vs_ema_34"),
            ((ema_34 - ema_89) / (atr_col + 1e-10)).alias("ema34_vs_ema89"),
        ]
    )


def _add_ny_session_dummies(df: pl.DataFrame) -> pl.DataFrame:
    """Add four NY session indicator columns (DST-aware).

    Adds: ``sess_asia``, ``sess_london``, ``sess_overlap``, ``sess_ny_pm``.
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


# ---------------------------------------------------------------------------
# Volume z-score
# ---------------------------------------------------------------------------


def _add_volume_zscore(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add rolling volume z-score vs ``volume_zscore_period``-bar mean."""
    n = config.features.multi_timeframe.volume_zscore_period
    vol_mean = pl.col("volume").rolling_mean(window_size=n)
    vol_std = pl.col("volume").rolling_std(window_size=n)
    return df.with_columns(
        ((pl.col("volume") - vol_mean) / (vol_std + 1e-10)).alias("volume_zscore_20")
    )


# ---------------------------------------------------------------------------
# Log return features
# ---------------------------------------------------------------------------


def _add_log_returns(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add log return features at multiple lookback horizons.

    Produces ``return_1h``, ``return_4h``, ``return_1d``.
    """
    cols: list[pl.Expr] = []
    for lookback in config.features.multi_timeframe.return_lookbacks:
        ret = (pl.col("close") / pl.col("close").shift(lookback)).log()
        name = {1: "return_1h", 4: "return_4h", 24: "return_1d"}.get(
            lookback, f"return_{lookback}b"
        )
        cols.append(ret.alias(name))
    return df.with_columns(cols)


# ---------------------------------------------------------------------------
# High-low range feature
# ---------------------------------------------------------------------------


def _add_high_low_range(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Add normalized 20-bar high-low range: (max_high - min_low) / ATR14."""
    p = config.features.atr_period
    atr_col = pl.col(f"atr_{p}")
    n = config.features.multi_timeframe.range_lookback
    rolling_high = pl.col("high").rolling_max(window_size=n)
    rolling_low = pl.col("low").rolling_min(window_size=n)
    return df.with_columns(
        ((rolling_high - rolling_low) / (atr_col + 1e-10)).alias("high_low_range_20")
    )


# ---------------------------------------------------------------------------
# Regime features — trend_strength
# ---------------------------------------------------------------------------


def add_trend_strength(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Add ``trend_strength`` column — Wilder-smoothed ADX.

    ADX > 25 typically indicates a trending market; ADX < 20 suggests a
    range-bound market.
    """
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
    plus_di = 100.0 * plus_dm_smooth / (atr_smooth + 1e-10)
    minus_di = 100.0 * minus_dm_smooth / (atr_smooth + 1e-10)

    # DX → ADX
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm_mean(alpha=alpha, adjust=False)

    return df.with_columns(adx.alias("trend_strength"))


# ---------------------------------------------------------------------------
# Feature list sidecar
# ---------------------------------------------------------------------------


def _save_feature_list(features_path: Path, feature_cols: list[str]) -> None:
    """Write a JSON sidecar listing feature column names."""
    list_path = features_path.with_suffix(".feature_list.json")
    with open(list_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
