"""Shared modules for the thesis pipeline.

All stable, cross-cutting code lives here so stage implementations can
import from ``thesis.shared`` without coupling to specific pipeline
module locations.
"""

from thesis.shared.config import (
    BacktestConfig,
    Config,
    DataConfig,
    FeaturesConfig,
    GRUConfig,
    LabelsConfig,
    LGBMConfig,
    MultiTimeframeConfig,
    PathsConfig,
    SplittingConfig,
    ValidationConfig,
    WorkflowConfig,
    get_config,
    load_config,
    reload_config,
)
from thesis.shared.constants import (
    ATR_HIGH_QUANTILE,
    ATR_LOW_QUANTILE,
    CALIB_LR,
    CALIB_MAX_ITER,
    CENSORED_LABEL,
    CHART_COLORS,
    CORE_STATIC_FEATURES,
    COSINE_T0,
    COSINE_TMULT,
    DIST_SHIFT_CLIP_MAX,
    DIST_SHIFT_CLIP_MIN,
    ECE_N_BINS,
    EXCLUDE_COLS,
    EXCLUDED_FEATURE_COLS,
    FEATURE_EPS,
    GRAD_CLIP_NORM,
    H1_BARS_PER_YEAR,
    LABEL_PROFITABILITY_WARN_PCT,
    ROUNDTRIP_MULT,
    SAMPLE_WEIGHT_MIN,
    STD_EPS,
    WARMUP_EPOCHS,
)
from thesis.shared.session_paths import (
    configure_session_paths,
    load_config_for_session,
)
from thesis.shared.ui import (
    STAGE_LABELS,
    STAGE_STYLES,
    console,
    stage_header,
    stage_skip,
)
from thesis.shared.zones import _ZONE_COLORS, _get_metric_zone, _is_extreme_value

__all__ = [
    # config
    "Config",
    "load_config",
    "get_config",
    "reload_config",
    "DataConfig",
    "SplittingConfig",
    "ValidationConfig",
    "MultiTimeframeConfig",
    "FeaturesConfig",
    "LabelsConfig",
    "LGBMConfig",
    "GRUConfig",
    "BacktestConfig",
    "WorkflowConfig",
    "PathsConfig",
    # constants
    "EXCLUDE_COLS",
    "EXCLUDED_FEATURE_COLS",
    "H1_BARS_PER_YEAR",
    "CHART_COLORS",
    "SAMPLE_WEIGHT_MIN",
    "ATR_LOW_QUANTILE",
    "ATR_HIGH_QUANTILE",
    "LABEL_PROFITABILITY_WARN_PCT",
    "ROUNDTRIP_MULT",
    "CENSORED_LABEL",
    "DIST_SHIFT_CLIP_MIN",
    "DIST_SHIFT_CLIP_MAX",
    "FEATURE_EPS",
    "STD_EPS",
    "GRAD_CLIP_NORM",
    "WARMUP_EPOCHS",
    "COSINE_T0",
    "COSINE_TMULT",
    "ECE_N_BINS",
    "CALIB_LR",
    "CALIB_MAX_ITER",
    "CORE_STATIC_FEATURES",
    # session_paths
    "configure_session_paths",
    "load_config_for_session",
    # ui
    "console",
    "STAGE_STYLES",
    "STAGE_LABELS",
    "stage_header",
    "stage_skip",
    # zones
    "_get_metric_zone",
    "_is_extreme_value",
    "_ZONE_COLORS",
]
