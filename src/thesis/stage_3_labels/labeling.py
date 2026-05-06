"""Asymmetric upper/lower barrier direction labeling.

Uses separate ``atr_tp_multiplier`` and ``atr_sl_multiplier`` for take-profit
and stop-loss barriers. No DST detection, no session definitions, no
dead-hour filtering.

Labels are encoded as ``+1`` for long, ``0`` for hold, and ``-1`` for short.
"""

from __future__ import annotations

import logging
from pathlib import Path

from numba import njit
import numpy as np
import polars as pl

from thesis.shared.config import Config
from thesis.shared.constants import (
    ATR_HIGH_QUANTILE,
    ATR_LOW_QUANTILE,
    CENSORED_LABEL,
    LABEL_PROFITABILITY_WARN_PCT,
    ROUNDTRIP_MULT,
    SAMPLE_WEIGHT_MIN,
)
from thesis.shared.ui import console

logger = logging.getLogger("thesis.labels")


# Public API


def generate_labels(config: Config) -> None:
    """Generate direction-barrier labels and write them to parquet.

    Loads engineered features and OHLCV bars, joins them by timestamp, computes
    asymmetric upper/lower barrier labels, appends label metadata, logs the
    class distribution, and writes the configured labels output.

    Args:
        config: Application configuration containing feature, OHLCV, label,
            ATR, and barrier settings.

    Raises:
        FileNotFoundError: If the features or OHLCV input paths do not exist.
        ValueError: If the required ATR column is missing from the features.
    """
    features_path = Path(config.paths.features)
    ohlcv_path = Path(config.paths.ohlcv)
    _validate_paths(features_path, ohlcv_path)

    logger.info("Loading features: %s", features_path)
    with console.status(f"[cyan]Loading features[/] {features_path}"):
        df_feat = pl.read_parquet(features_path)

    logger.info("Loading OHLCV: %s", ohlcv_path)
    with console.status(f"[cyan]Loading OHLCV[/] {ohlcv_path}"):
        df_ohlcv = pl.read_parquet(ohlcv_path).select(
            ["timestamp", "open", "high", "low", "close"]
        )

    _validate_unique_timestamps(df_feat, "features")
    _validate_unique_timestamps(df_ohlcv, "OHLCV")

    df = df_feat.join(df_ohlcv, on="timestamp", how="inner")
    _validate_unique_timestamps(df, "joined feature/OHLCV")
    logger.info("Joined rows: %d", len(df))

    atr_col = f"atr_{config.features.atr_period}"
    if atr_col not in df.columns:
        raise ValueError(f"{atr_col} not in features. Run feature engineering first.")

    _log_atr_stats(df, atr_col, config.labels.min_atr)

    labels_arr, upper_arr, lower_arr, touched_bars_arr, ambiguous_count = (
        _compute_labels(
            close=df["close"].to_numpy(),
            high=df["high"].to_numpy(),
            low=df["low"].to_numpy(),
            atr=df[atr_col].to_numpy(),
            tp_mult=config.labels.atr_tp_multiplier,
            sl_mult=config.labels.atr_sl_multiplier,
            horizon=config.labels.horizon_bars,
            min_atr=config.labels.min_atr,
        )
    )

    logger.info(
        "Direction-barrier params: tp_mult=%.2f, sl_mult=%.2f,"
        " horizon=%d, min_atr=%.6f",
        config.labels.atr_tp_multiplier,
        config.labels.atr_sl_multiplier,
        config.labels.horizon_bars,
        config.labels.min_atr,
    )
    logger.info(
        "Ambiguous same-bar both-hit labels: %d (treated as Hold)",
        ambiguous_count,
    )

    event_end_arr = compute_event_end(touched_bars_arr, config.labels.horizon_bars)
    sample_weight_arr = compute_average_uniqueness(event_end_arr)

    df = _merge_label_columns(
        df,
        labels_arr,
        upper_arr,
        lower_arr,
        touched_bars_arr,
        event_end_arr,
        sample_weight_arr,
    )
    _log_label_profitability(df, config)
    df = _filter_censored(df)
    _log_distribution(df)
    _log_weight_stats(df)

    out_path = Path(config.paths.labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    logger.info("Labels saved: %s (%d rows)", out_path, len(df))


# Core labeling logic


@njit
def _compute_labels(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    tp_mult: float,
    sl_mult: float,
    horizon: int,
    min_atr: float,
) -> tuple:
    """Compute asymmetric direction-barrier outcomes for each bar.

    For each index i this sets upper = close[i] + tp_mult * max(atr[i], min_atr)
    and lower = close[i] - sl_mult * max(atr[i], min_atr), then inspects bars
    i+1 .. i+horizon (bounded by series end) to determine which barrier is
    touched first. If neither barrier is touched within the horizon the label
    remains 0. If both barriers are touched on the same OHLC bar, the sample
    is treated as ambiguous and labeled Hold (0). Rows within `horizon` bars
    of the series end are marked -2 (censored) and are dropped from training.

    Returns:
        Tuple of labels, upper barriers, lower barriers, touched-bar offsets,
        and the same-bar both-hit count.
    """
    n = len(close)
    labels = np.zeros(n, dtype=np.int32)
    upper_barriers = np.zeros(n, dtype=np.float64)
    lower_barriers = np.zeros(n, dtype=np.float64)
    touched_bars = np.full(n, -1, dtype=np.int32)
    ambiguous_count = 0

    for i in range(n):
        a = max(atr[i], min_atr)
        upper = close[i] + tp_mult * a
        lower = close[i] - sl_mult * a
        upper_barriers[i] = upper
        lower_barriers[i] = lower

        # Right-censored: not enough forward bars to evaluate horizon
        if i + horizon >= n:
            labels[i] = CENSORED_LABEL
            touched_bars[i] = CENSORED_LABEL
            continue

        label = 0  # Hold by default
        for j in range(i + 1, min(i + 1 + horizon, n)):
            upper_hit = high[j] >= upper
            lower_hit = low[j] <= lower
            if upper_hit and lower_hit:
                # OHLC bars do not reveal intra-bar path; keep ambiguous
                # samples neutral.
                ambiguous_count += 1
                touched_bars[i] = j - i
                break
            if upper_hit:
                label = 1  # Long
                touched_bars[i] = j - i
                break
            if lower_hit:
                label = -1  # Short
                touched_bars[i] = j - i
                break
        labels[i] = label

    return labels, upper_barriers, lower_barriers, touched_bars, ambiguous_count


@njit
def compute_event_end(touched_bars: np.ndarray, horizon: int) -> np.ndarray:
    """Convert touched-bar offsets to absolute event-end indices.

    Hold/ambiguous labels with ``touched_bar == -1`` are active for the full
    horizon. Censored rows are assigned the full horizon too; they are dropped
    before training, but keeping a finite value makes diagnostics stable.

    Args:
        touched_bars: Array of touched-bar offsets (-1 for hold/censored).
        horizon: Maximum forward horizon in bars.

    Returns:
        Array of absolute event-end indices (0-indexed).
    """
    n = len(touched_bars)
    event_end = np.empty(n, dtype=np.int32)
    for i in range(n):
        offset = touched_bars[i]
        if offset < 0:
            offset = horizon
        event_end[i] = i + offset
    return event_end


@njit
def compute_average_uniqueness(event_end: np.ndarray) -> np.ndarray:
    """Compute López de Prado average-uniqueness sample weights.

    For sample ``i`` active over ``[i, event_end[i]]``, uniqueness is the mean
    of ``1 / concurrency[t]`` across its active bars. Highly overlapping labels
    receive lower weights. Output is clipped to a small positive floor and
    normalized to mean 1 so optimizers keep their usual loss scale.

    Args:
        event_end: Array of absolute event-end indices for each bar.

    Returns:
        Array of sample weights normalized to mean 1.
    """
    n = len(event_end)
    diff = np.zeros(n + 1, dtype=np.float64)
    for i in range(n):
        end = event_end[i]
        if end < i:
            end = i
        if end >= n:
            end = n - 1
        diff[i] += 1.0
        diff[end + 1] -= 1.0

    concurrency = np.empty(n, dtype=np.float64)
    running = 0.0
    for i in range(n):
        running += diff[i]
        concurrency[i] = max(running, 1.0)

    inv_prefix = np.zeros(n + 1, dtype=np.float64)
    for i in range(n):
        inv_prefix[i + 1] = inv_prefix[i] + 1.0 / concurrency[i]

    weights = np.empty(n, dtype=np.float64)
    total = 0.0
    for i in range(n):
        end = event_end[i]
        if end < i:
            end = i
        if end >= n:
            end = n - 1
        span = end - i + 1
        weight = (inv_prefix[end + 1] - inv_prefix[i]) / span
        weight = max(weight, SAMPLE_WEIGHT_MIN)
        weights[i] = weight
        total += weight

    mean = total / n if n > 0 else 1.0
    if mean <= 0.0:
        mean = 1.0
    for i in range(n):
        weights[i] /= mean
    return weights


# Helpers


def _validate_paths(features_path: Path, ohlcv_path: Path) -> None:
    """Validate that required input paths exist.

    Args:
        features_path: Path to the features parquet file.
        ohlcv_path: Path to the OHLCV parquet file.

    Raises:
        FileNotFoundError: If either path does not exist.
    """
    if not features_path.exists():
        raise FileNotFoundError(f"Features not found: {features_path}")
    if not ohlcv_path.exists():
        raise FileNotFoundError(f"OHLCV not found: {ohlcv_path}")


def _validate_unique_timestamps(df: pl.DataFrame, name: str) -> None:
    """Fail fast if a stage boundary contains duplicate timestamps.

    Args:
        df: DataFrame to validate.
        name: Human-readable name for the data source (used in error message).

    Raises:
        ValueError: If duplicate timestamps are found.
    """
    if "timestamp" not in df.columns:
        return
    duplicate_count = len(df) - df["timestamp"].n_unique()
    if duplicate_count > 0:
        raise ValueError(
            f"{name} data contains {duplicate_count} duplicate timestamps; "
            "deduplicate before label generation."
        )


def _merge_label_columns(
    df: pl.DataFrame,
    labels_arr: np.ndarray,
    upper_arr: np.ndarray,
    lower_arr: np.ndarray,
    touched_bars_arr: np.ndarray,
    event_end_arr: np.ndarray,
    sample_weight_arr: np.ndarray,
) -> pl.DataFrame:
    """Build and join label columns into the main dataframe.

    Args:
        df: Main DataFrame to join labels into.
        labels_arr: Label values (-1, 0, 1, or censored).
        upper_arr: Upper barrier prices.
        lower_arr: Lower barrier prices.
        touched_bars_arr: Touched-bar offsets.
        event_end_arr: Event-end indices.
        sample_weight_arr: Sample weights.

    Returns:
        DataFrame with label columns joined on ``timestamp``.
    """
    ts_dtype = df["timestamp"].dtype
    labels_df = pl.DataFrame(
        {
            "timestamp": pl.Series(df["timestamp"].to_list(), dtype=ts_dtype),
            "label": labels_arr,
            "upper_barrier": upper_arr,
            "lower_barrier": lower_arr,
            "touched_bar": touched_bars_arr,
            "event_end": event_end_arr,
            "sample_weight": sample_weight_arr,
        }
    )
    return df.join(labels_df, on="timestamp", how="left")


def _log_atr_stats(df: pl.DataFrame, atr_col: str, min_atr: float) -> None:
    """Log ATR distribution and how often the configured floor applies.

    Args:
        df: DataFrame containing the ATR column.
        atr_col: Name of the ATR column.
        min_atr: Configured minimum ATR value.
    """
    stats = df.select(
        [
            pl.col(atr_col).min().alias("min"),
            pl.col(atr_col).median().alias("median"),
            pl.col(atr_col).quantile(ATR_LOW_QUANTILE).alias("p5"),
            pl.col(atr_col).quantile(ATR_HIGH_QUANTILE).alias("p95"),
            (pl.col(atr_col) < min_atr).mean().alias("floor_rate"),
        ]
    ).row(0, named=True)
    logger.info(
        "ATR stats (%s): min=%.6f, median=%.6f, p5=%.6f, p95=%.6f, "
        "below_min_atr=%.2f%%",
        atr_col,
        stats["min"] or 0.0,
        stats["median"] or 0.0,
        stats["p5"] or 0.0,
        stats["p95"] or 0.0,
        (stats["floor_rate"] or 0.0) * 100,
    )


def _filter_censored(df: pl.DataFrame) -> pl.DataFrame:
    """Remove censored rows where forward horizon is insufficient.

    Censored rows (``label == -2``) lack enough future data to evaluate the
    barrier outcome.  Keeping them as Hold would inject label noise, so they
    are dropped entirely.

    When a ``regression_target`` column is present (regression objective),
    rows with NaN regression target are also dropped.  This provides
    defense-in-depth against zero-target tail leakage when stage-4
    regression-target computation marks rows as censored.

    Args:
        df: DataFrame with a ``label`` column.

    Returns:
        DataFrame with censored rows removed.
    """
    n_before = len(df)
    # Label-based censoring
    n_censored = int((df["label"] == CENSORED_LABEL).sum())
    if n_censored > 0:
        df = df.filter(pl.col("label") != CENSORED_LABEL)

    # NaN regression-target censoring (defense-in-depth for regression mode)
    n_nan = 0
    if "regression_target" in df.columns:
        n_nan = int(df["regression_target"].is_nan().sum())
        if n_nan > 0:
            df = df.filter(pl.col("regression_target").is_not_nan())

    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.info(
            "Dropped %d censored rows (label=%d, regression_nan=%d) — "
            "insufficient forward horizon",
            n_dropped,
            n_censored,
            n_nan,
        )
    return df


def _log_distribution(df: pl.DataFrame) -> None:
    """Log counts and percentages for each value in the dataframe's `label` column.

    If the ``label`` column is not present the function returns without
    logging.  Each logged line reports the label value, its absolute count,
    and its percentage of the dataframe rows.

    Args:
        df (pl.DataFrame): DataFrame expected to contain a `label` column.
    """
    if "label" not in df.columns:
        return
    counts = df["label"].value_counts().sort("label")
    total = len(df)
    for row in counts.iter_rows():
        label, count = row
        logger.info("  Class %s: %d (%.1f%%)", label, count, count / total * 100)


def _log_weight_stats(df: pl.DataFrame) -> None:
    """Log average-uniqueness sample-weight diagnostics.

    Args:
        df: DataFrame with a ``sample_weight`` column.
    """
    if "sample_weight" not in df.columns:
        return
    stats = df.select(
        [
            pl.col("sample_weight").min().alias("min"),
            pl.col("sample_weight").median().alias("median"),
            pl.col("sample_weight").max().alias("max"),
            pl.col("sample_weight").mean().alias("mean"),
        ]
    ).row(0, named=True)
    logger.info(
        "Average-uniqueness sample weights: min=%.4f median=%.4f max=%.4f mean=%.4f",
        stats["min"] or 0.0,
        stats["median"] or 0.0,
        stats["max"] or 0.0,
        stats["mean"] or 0.0,
    )


def _log_label_profitability(df: pl.DataFrame, config: Config) -> None:
    """Log label profitability diagnostics after trading costs.

    Computes the percentage of Long labels (+1) and Short labels (-1) that
    would have been profitable after accounting for spread, slippage, and
    commission. The net return per bar is:

        net_return = (close[i+horizon] - close[i+1]) / close[i] * leverage - costs

    where ``costs`` expressed as a fraction of the notional value using
    ``BacktestConfig`` cost parameters and ``DataConfig.tick_size``.

    Long labels are profitable when ``net_return > 0``; Short labels are
    profitable when ``net_return < 0`` (price moved opposite of the
    long-oriented formula). If both classes fall below 60 % the labels are
    flagged as economically questionable.

    Args:
        df: Full joined feature + OHLCV + label DataFrame (pre-censoring).
        config: Application configuration.
    """
    required = {"close", "label", "timestamp"}
    if not required.issubset(df.columns):
        return

    horizon = config.labels.horizon_bars
    leverage = config.backtest.leverage
    tick_size = config.data.tick_size
    spread_ticks = config.backtest.spread_ticks
    slippage_ticks = config.backtest.slippage_ticks
    commission_per_lot = config.backtest.commission_per_lot
    contract_size = config.data.contract_size

    # Sort by timestamp so positional shifts are strictly chronological.
    df = df.sort("timestamp")

    # Fixed-cost numerator expressed in price terms so cost_frac = num / close[i]
    # is a fraction of notional value that can be subtracted from the leveraged
    # return expression.
    cost_numerator = (spread_ticks + slippage_ticks) * tick_size + (
        commission_per_lot * ROUNDTRIP_MULT
    ) / contract_size

    # net_return = (close[i+horizon] - close[i+1]) / close[i] * leverage - costs
    net_return_expr = (
        pl.col("close").shift(-horizon) - pl.col("close").shift(-1)
    ) / pl.col("close") * pl.lit(leverage) - (pl.lit(cost_numerator) / pl.col("close"))

    result = df.with_columns(net_return_expr.alias("_net_return"))

    # Exclude censored rows (-2) and rows where the shift produced a null
    # (last ``horizon`` rows naturally have no valid close[i+horizon]).
    result = result.filter(
        (pl.col("label") != CENSORED_LABEL) & pl.col("_net_return").is_not_null()
    )

    if result.is_empty():
        logger.warning("Label profitability: no valid samples after filtering.")
        return

    long_pct = 0.0
    short_pct = 0.0

    for label_val, label_name, condition in [
        (1, "Long", pl.col("_net_return") > 0),
        (-1, "Short (net negative)", pl.col("_net_return") < 0),
    ]:
        class_df = result.filter(pl.col("label") == label_val)
        total = len(class_df)
        if total == 0:
            logger.info("  Class %d (%s): no samples", label_val, label_name)
            continue
        profitable = class_df.filter(condition).height
        pct = profitable / total * 100.0
        logger.info(
            "%% of %s labels that are profitable after costs: %.1f%% (%d/%d)",
            label_name.split(" (")[0],
            pct,
            profitable,
            total,
        )
        if label_val == 1:
            long_pct = pct
        elif label_val == -1:
            short_pct = pct

    # Hold class for completeness
    hold_df = result.filter(pl.col("label") == 0)
    hold_total = len(hold_df)
    if hold_total > 0:
        hold_up = hold_df.filter(pl.col("_net_return") > 0).height
        hold_down = hold_df.filter(pl.col("_net_return") < 0).height
        logger.info(
            "  Class 0 (Hold): %d samples (net pos: %d, net neg: %d)",
            hold_total,
            hold_up,
            hold_down,
        )

    if (
        long_pct < LABEL_PROFITABILITY_WARN_PCT
        and short_pct < LABEL_PROFITABILITY_WARN_PCT
    ):
        logger.warning(
            "LABEL PROFITABILITY LOW: Long %.1f%%, Short %.1f%% -- "
            "labels may not be economically useful after trading costs",
            long_pct,
            short_pct,
        )
