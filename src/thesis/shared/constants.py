"""Project-wide constants shared across pipeline stages.

Single source of truth for column exclusion sets and other pipeline-level
constants. Import from here to keep stages in sync.
"""

# Column exclusion sets

#: Columns that are *never* model features — excluded from training,
#: correlation filtering, and feature selection everywhere.
#:
#: Rationale per group:
#:  - timestamp          → index / join key, not a feature
#:  - label              → target variable (look-ahead)
#:  - upper_barrier/lower_barrier/touched_bar/event_end/sample_weight →
#:    label-derived metadata; not predictive features
#:  - tp_price/sl_price → label price metadata, not predictive features
#:  - open_right/high_right/low_right/close_right → label-derived look-ahead
#:  - open/high/low/close/volume → raw OHLCV, excluded to avoid raw price leakage
#:  - avg_spread/tick_count → microstructure columns kept for backtest
#:    but not useful as ML features in their raw form
#:  - log_returns → GRU sequence input; excluded from the *static* LightGBM features
#:    to avoid double-counting the information already encoded in GRU hidden states
#:
#: All 28 engineered features (core indicators, multi-timeframe 4H, trend
#: distances, Bollinger bands, volume z-score, log returns, range, and regime
#: features) are intentionally NOT in this set — they are available as GRU
#: sequence inputs and/or static LightGBM features.
EXCLUDE_COLS: frozenset[str] = frozenset(
    [
        "timestamp",
        "label",
        "upper_barrier",
        "lower_barrier",
        "touched_bar",
        "event_end",
        "sample_weight",
        "tp_price",
        "sl_price",
        "open_right",  # Label-derived — pure look-ahead
        "high_right",  # Label-derived — pure look-ahead
        "low_right",  # Label-derived — pure look-ahead
        "close_right",  # Label-derived — pure look-ahead
        "open",
        "high",
        "low",
        "close",
        "volume",
        "avg_spread",
        "tick_count",
        "atr_14",  # Label-barrier helper; normalized ATR is model-facing
        "log_returns",  # GRU sequence input — not a static feature for LightGBM
    ]
)

# Private alias used by internal modules.
_EXCLUDE_COLS = EXCLUDE_COLS

# Annualization constant for hourly XAU/USD-style markets.
# Uses 24 hours × 5 trading days × 52 weeks; actual bar counts may vary by
# broker holidays, market closures, and missing data.
H1_BARS_PER_YEAR: int = 24 * 5 * 52

# Shared visualization palette (matplotlib + pyecharts)

CHART_COLORS: dict[str, str] = {
    "primary": "#2563EB",
    "secondary": "#7C3AED",
    "success": "#059669",
    "danger": "#DC2626",
    "warning": "#D97706",
    "gray": "#6B7280",
    "long": "#059669",
    "short": "#DC2626",
    "flat": "#6B7280",
}

#: Alias for interactive chart modules (`charts/`) — same set as ``EXCLUDE_COLS``.
EXCLUDED_FEATURE_COLS = EXCLUDE_COLS


def timeframe_to_ms(timeframe: str) -> int:
    """Parse a config timeframe string into milliseconds.

    Supports ``H`` (hours), ``MIN`` / ``M`` (minutes), and ``D`` (day) suffixes.

    Args:
        timeframe: Timeframe string like ``"1H"``, ``"4H"``, ``"5MIN"``, ``"1D"``.

    Returns:
        Timeframe duration in milliseconds.

    Raises:
        ValueError: If the timeframe format is unsupported or the numeric
            component is not positive.
    """
    tf = timeframe.upper()
    if tf.endswith("H"):
        hours = int(tf[:-1])
        if hours <= 0:
            raise ValueError(f"Invalid timeframe '{tf}': hours must be > 0")
        return hours * 3_600_000
    if tf.endswith("MIN"):
        minutes = int(tf[:-3])
        if minutes <= 0:
            raise ValueError(f"Invalid timeframe '{tf}': minutes must be > 0")
        return minutes * 60_000
    if tf.endswith("M"):
        minutes = int(tf[:-1])
        if minutes <= 0:
            raise ValueError(f"Invalid timeframe '{tf}': minutes must be > 0")
        return minutes * 60_000
    if tf in ("D", "1D"):
        return 86_400_000
    raise ValueError(f"Unsupported timeframe: {timeframe}")


# Labeling constants

#: Sample weight minimum floor for average-uniqueness computation.
#: Prevents numerical instability from near-zero weights before
#: normalisation to mean 1. Values below this are clamped up.
SAMPLE_WEIGHT_MIN: float = 0.05

#: ATR quantile p-values for diagnostic logging in ``_log_atr_stats``.
ATR_LOW_QUANTILE: float = 0.05
ATR_HIGH_QUANTILE: float = 0.95

#: Label profitability warning threshold (percentage).
#: If *both* Long and Short profit percentages fall below this value a warning
#: is emitted because the labels may not be economically viable after costs.
LABEL_PROFITABILITY_WARN_PCT: float = 60.0

#: Round-trip multiplier for commission / contract-size → price-unit cost.
ROUNDTRIP_MULT: float = 2.0

#: Special label value marking rows whose forward horizon exceeds available
#: data.  These are dropped before training (see ``_filter_censored``).
CENSORED_LABEL: int = -2

# Distribution-shift weight clipping

#: Minimum and maximum allowed per-class weight ratios when correcting
#: distribution shift between training and validation label frequencies.
DIST_SHIFT_CLIP_MIN: float = 0.5
DIST_SHIFT_CLIP_MAX: float = 3.0

# Feature engineering / numerical stability

#: Small epsilon for division safety in feature expressions (e.g. ATR
#: ratio, pivot position, log-return normalisation).
FEATURE_EPS: float = 1e-10

#: Epsilon used in standard-deviation denominators (e.g. z-score, dataset
#: standardisation) — larger than ``FEATURE_EPS`` to avoid amplifying
#: near-constant series.
STD_EPS: float = 1e-8

# GRU training hyper-parameter constants

#: Gradient clipping max-norm for GRU encoder and classifier parameters.
GRAD_CLIP_NORM: float = 1.0

#: Number of linear warmup epochs before cosine annealing takes over.
WARMUP_EPOCHS: int = 3

#: Number of consecutive epochs without val-loss improvement before emitting
#: a plateau warning (does not affect training — purely diagnostic).
PLATEAU_PATIENCE: int = 5

#: Cosine-annealing-with-warm-restarts base period (``T_0`` in PyTorch
#: ``CosineAnnealingWarmRestarts`` semantics).
COSINE_T0: int = 10

#: Cosine-annealing-with-warm-restarts period multiplier (``T_mult``).
COSINE_TMULT: int = 2

#: Number of confidence bins for Expected Calibration Error (ECE).
ECE_N_BINS: int = 10

#: Learning rate for LBFGS temperature-scaling calibration.
CALIB_LR: float = 0.01

#: Maximum LBFGS iterations for temperature-scaling calibration.
CALIB_MAX_ITER: int = 100

# Default LightGBM tabular features. Keep user config minimal; override only in code.
CORE_STATIC_FEATURES: tuple[str, ...] = (
    # Trend
    "ema34_vs_ema89",
    "close_vs_ema_34",
    "adx_14",
    "ema_slope_20",
    "regime_strength",
    # Momentum
    "return_1h",
    "return_4h",
    "macd_hist_atr",
    "rsi_14",
    # Volatility / Regime
    "atr_pct_close",
    "atr_ratio",
    "atr_percentile",
    "high_low_range_20",
    # Position / Location
    "price_dist_ratio",
    "price_position_20",
    "pivot_position",
    # Candle Structure
    "candle_body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    # Session
    "sess_london",
    "sess_overlap",
    # Volume / Activity
    "volume_zscore_20",
)
