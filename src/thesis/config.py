"""Centralized configuration for the thesis ML pipeline.

This module is the single entry point for loading and sharing runtime
configuration. TOML files may stay minimal: omitted fields fall back to the
dataclass defaults below.
"""

import logging
import re
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from thesis.constants import CORE_STATIC_FEATURES


logger = logging.getLogger("thesis.config")


# ---------------------------------------------------------------------------
# Dataclasses — one per TOML section
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    """Data loading and OHLCV parameters.

    Attributes:
        symbol: Display symbol used in session names and reports.
        timeframe: Bar timeframe such as ``1H`` or ``4H``.
        market_tz: Time zone used for session-aware feature engineering.
        start_date: Inclusive data start date.
        end_date: Inclusive data end date.
        tick_size: Minimum price movement.
        contract_size: Units per trading lot for the application demo.
    """

    symbol: str = "XAUUSD"
    timeframe: str = "1H"
    market_tz: str = "America/New_York"
    start_date: str = "2013-01-01"
    end_date: str = "2026-03-31"
    tick_size: float = 0.01
    contract_size: int = 100
    symbol_download: str = "XAUUSD"
    asset_class: str = "fx"
    download_concurrency: int = 20
    download_max_retries: int = 7
    download_force: bool = False  # Force re-download even if file exists
    download_skip_current_month: bool = True  # Skip current month (incomplete data)


@dataclass
class SplittingConfig:
    """Train / val / test date ranges for static splits.

    Only used when ``validation.method = "static"``. The default workflow
    uses walk-forward (sliding-window) validation — see :class:`ValidationConfig`.
    """

    train_start: str = "2013-01-01"
    train_end: str = "2022-03-31 23:59:59"
    val_start: str = "2022-04-01"
    val_end: str = "2024-03-31 23:59:59"
    test_start: str = "2024-04-01"
    test_end: str = "2026-03-31 23:59:59"
    purge_bars: int = 25
    embargo_bars: int = 50
    embargo_scale_by_timeframe: bool = True
    embargo_reference_timeframe: str = "1H"


@dataclass
class ValidationConfig:
    """Walk-forward validation parameters — sliding window with purge & embargo."""

    method: str = "sliding"  # "sliding" or "static"
    train_window_bars: int = 26280  # ~3 years of H1 bars
    test_window_bars: int = 4380  # ~6 months of H1 bars
    step_bars: int = 4380  # step between windows (= test_window for non-overlapping)
    purge_bars: int = 25  # bars removed at train/test boundary
    embargo_bars: int = 50  # additional gap after purge
    min_train_bars: int = 10000  # minimum training bars to produce a window
    oof_ensemble: bool = True  # aggregate OOF predictions across windows
    wf_optuna_trials: int = (
        0  # Optuna trials per walk-forward window (0 = fixed params)
    )


@dataclass
class MultiTimeframeConfig:
    """Multi-timeframe and extended feature parameters."""

    sma_periods: list[int] = field(default_factory=lambda: [50])
    ema_long: int = 200
    bb_period: int = 20
    bb_std: float = 2.0
    return_lookbacks: list[int] = field(default_factory=lambda: [1, 4, 24])
    range_lookback: int = 20
    volume_zscore_period: int = 20


@dataclass
class FeaturesConfig:
    """Feature engineering parameters.

    Attributes:
        rsi_period: RSI lookback.
        atr_period: ATR lookback.
        macd_fast: MACD fast EMA span.
        macd_slow: MACD slow EMA span.
        macd_signal: MACD signal EMA span.
        correlation_threshold: Threshold used by compatibility tests and
            optional filtering code.
        static_feature_cols: Compact tabular feature whitelist consumed by
            LightGBM in the simplified hybrid architecture.
        multi_timeframe: Derived-feature settings.
    """

    rsi_period: int = 14
    atr_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    correlation_threshold: float = 0.75
    static_feature_cols: list[str] = field(
        default_factory=lambda: list(CORE_STATIC_FEATURES)
    )
    multi_timeframe: MultiTimeframeConfig = field(default_factory=MultiTimeframeConfig)


@dataclass
class LabelsConfig:
    """Triple-barrier label parameters (single ATR multiplier, no sessions)."""

    atr_multiplier: float = 2.5
    horizon_bars: int = 24
    num_classes: int = 3
    min_atr: float = 0.5


@dataclass
class LGBMConfig:
    """LightGBM parameters."""

    # LightGBM
    architecture: str = "hybrid"  # "hybrid" or "stacking"
    use_optuna: bool = False
    optuna_trials: int = 20
    optuna_timeout: int = 300
    num_leaves: int = 31
    max_depth: int = 6
    learning_rate: float = 0.02
    n_estimators: int = 500
    min_child_samples: int = 50
    subsample: float = 0.80
    subsample_freq: int = 5
    feature_fraction: float = 0.70
    reg_alpha: float = 0.05
    reg_lambda: float = 5.0
    early_stopping_rounds: int = 40


@dataclass
class GRUConfig:
    """GRU feature extractor parameters."""

    input_size: int = 8
    feature_cols: list[str] = field(
        default_factory=lambda: [
            "log_returns",
            "atr_14",
            "close_vs_ema_34",
            "ema34_vs_ema89",
            "candle_body_ratio",
            "return_1h",
            "return_4h",
            "price_position_20",
        ]
    )
    hidden_size: int = 64
    num_layers: int = 2
    sequence_length: int = 48
    dropout: float = 0.3
    learning_rate: float = 0.0005
    batch_size: int = 256
    epochs: int = 50
    patience: int = 15
    min_epochs: int = 5
    focal_loss_gamma: float = 2.0
    warmup_epochs: int = 3


@dataclass
class BacktestConfig:
    """CFD backtest parameters — thin wrapper for backtesting.py."""

    initial_capital: float = 10_000.0
    leverage: int = 10  # margin = 1/leverage
    spread_ticks: float = 35.0  # → spread param (relative)
    slippage_ticks: float = 5.0
    commission_per_lot: float = 10.0  # → callable commission
    atr_stop_multiplier: float = 1.0
    atr_tp_multiplier: float = 2.0  # ATR multiplier for take-profit (0 = disabled)
    lots_per_trade: float = 0.1  # base lot size for position sizing
    min_lots: float = 0.05  # minimum lot size (low-conviction floor)
    max_lots: float = 0.1  # maximum lot size (high-conviction cap)
    confidence_threshold: float = (
        0.55  # min predicted probability to act (0 = disabled)
    )
    max_drawdown_cutoff: float = (
        0.30  # circuit breaker: stop if equity < peak * (1 - cutoff)
    )
    dd_cooldown_bars: int = 12  # bars to pause after drawdown cutoff breach
    max_open_positions: int = 1  # max simultaneous open positions
    daily_loss_limit: float = 0.03  # stop trading for day after -N equity drawdown


@dataclass
class WorkflowConfig:
    """Pipeline execution toggles and seeds."""

    run_data_pipeline: bool = True
    run_feature_engineering: bool = True
    run_label_generation: bool = True
    run_model_training: bool = True
    run_backtest: bool = True
    run_reporting: bool = True
    force_rerun: bool = False
    random_seed: int = 2024
    n_jobs: int = -1
    session_timestamp: str = ""  # Set at runtime


@dataclass
class StackingConfig:
    """True stacking parameters."""

    base_models: list[str] = field(default_factory=lambda: ["gru", "lgbm"])
    meta_model: str = "lightgbm"
    use_probability_features_only: bool = True
    min_meta_train_folds: int = 1
    min_meta_train_rows: int = 500
    final_refit: bool = True


@dataclass
class PathsConfig:
    """Artifact paths with session-based output support."""

    data_raw: str = "data/raw/XAUUSD"
    data_processed: str = "data/processed"
    ohlcv: str = "data/processed/ohlcv.parquet"
    features: str = "data/processed/features.parquet"
    labels: str = "data/processed/labels.parquet"
    train_data: str = "data/processed/train.parquet"  # static split only
    val_data: str = "data/processed/val.parquet"  # static split only
    test_data: str = "data/processed/test.parquet"  # static split only
    model: str = "models/lightgbm_model.pkl"
    gru_model: str = "models/gru_model.pt"
    predictions: str = "data/predictions/final_predictions.parquet"
    stack_bundle: str = "models/stacking_bundle.joblib"
    backtest_results: str = "results/backtest_results.json"
    report: str = "results/thesis_report.md"
    session_dir: str = ""  # Set at runtime by pipeline


# ---------------------------------------------------------------------------
# Top-level Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Main configuration — one attribute per TOML section."""

    data: DataConfig = field(default_factory=DataConfig)
    splitting: SplittingConfig = field(default_factory=SplittingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    labels: LabelsConfig = field(default_factory=LabelsConfig)
    model: LGBMConfig = field(default_factory=LGBMConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    gru: GRUConfig = field(default_factory=GRUConfig)
    stacking: StackingConfig = field(default_factory=StackingConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_SECTION_MAP: dict[str, type] = {
    "data": DataConfig,
    "splitting": SplittingConfig,
    "validation": ValidationConfig,
    "features": FeaturesConfig,
    "labels": LabelsConfig,
    "model": LGBMConfig,
    "backtest": BacktestConfig,
    "gru": GRUConfig,
    "stacking": StackingConfig,
    "workflow": WorkflowConfig,
    "paths": PathsConfig,
}


def _timeframe_to_minutes(timeframe: str) -> int:
    """Convert a timeframe string to minutes.

    Args:
        timeframe: String such as ``15M``, ``1H``, ``4H``, ``1D``, or ``1W``.

    Returns:
        Number of minutes represented by one bar.

    Raises:
        ValueError: If the timeframe format is unsupported.
    """
    match = re.fullmatch(r"\s*(\d+)\s*([mhdwMHDW])\s*", timeframe)
    if not match:
        raise ValueError(
            "Invalid timeframe format: "
            f"{timeframe!r}. Expected forms like 15M, 1H, 4H, 1D, 1W."
        )

    qty = int(match.group(1))
    unit = match.group(2).upper()
    unit_minutes = {
        "M": 1,
        "H": 60,
        "D": 24 * 60,
        "W": 7 * 24 * 60,
    }
    return qty * unit_minutes[unit]


def _scale_bars_by_timeframe(
    base_bars: int,
    base_timeframe: str,
    target_timeframe: str,
) -> int:
    """Scale a bar count to preserve elapsed time across timeframes.

    Args:
        base_bars: Number of bars on the reference timeframe.
        base_timeframe: Reference timeframe.
        target_timeframe: Target timeframe.

    Returns:
        Equivalent number of target-timeframe bars, floored at 1.
    """
    base_minutes = _timeframe_to_minutes(base_timeframe)
    target_minutes = _timeframe_to_minutes(target_timeframe)
    scaled = int(round(base_bars * (base_minutes / target_minutes)))
    return max(1, scaled)


def load_config(config_path: str | Path = "config.toml") -> Config:
    """Load configuration from a flat TOML file.

    Args:
        config_path: Path to TOML configuration file.

    Returns:
        Fully populated ``Config`` object.

    Raises:
        FileNotFoundError: If *config_path* does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        raw: dict[str, Any] = tomllib.load(f)

    cfg = Config()
    for section, cls in _SECTION_MAP.items():
        if section in raw:
            section_data = raw[section]
            # Handle nested subsections — pop them out before **kwargs
            if section == "features" and "multi_timeframe" in section_data:
                mt_data = section_data.pop("multi_timeframe")
                setattr(cfg, section, cls(**section_data))
                cfg.features.multi_timeframe = MultiTimeframeConfig(**mt_data)
            else:
                setattr(cfg, section, cls(**section_data))

    if cfg.splitting.embargo_scale_by_timeframe:
        base_bars = cfg.splitting.embargo_bars
        reference_tf = cfg.splitting.embargo_reference_timeframe
        target_tf = cfg.data.timeframe
        cfg.splitting.embargo_bars = _scale_bars_by_timeframe(
            base_bars,
            reference_tf,
            target_tf,
        )
        logger.info(
            "Scaled embargo bars from %d @ %s to %d @ %s",
            base_bars,
            reference_tf,
            cfg.splitting.embargo_bars,
            target_tf,
        )

    # Ensure base directories exist
    Path(cfg.paths.data_processed).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.data_raw).mkdir(parents=True, exist_ok=True)

    return cfg


@lru_cache(maxsize=8)
def get_config(config_path: str | Path = "config.toml") -> Config:
    """Load and cache a configuration for reuse across modules.

    Prefer dependency injection for pipeline code. This helper is intended for
    UI/reporting modules or scripts that need a shared config without manually
    passing it through every call.

    Args:
        config_path: Path to a TOML configuration file.

    Returns:
        Cached :class:`Config` instance for the given path.
    """
    return load_config(Path(config_path))


def reload_config(config_path: str | Path = "config.toml") -> Config:
    """Clear the config cache and reload a TOML file.

    Args:
        config_path: Path to a TOML configuration file.

    Returns:
        Fresh :class:`Config` instance.
    """
    get_config.cache_clear()
    return get_config(config_path)
