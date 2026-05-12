# Configuration

This project keeps the public experiment surface in `config.toml`. Runtime dataclasses and defaults live in `src/thesis/shared/config.py`. Unknown config keys fail fast with an error message.

---

## Current Default Profile

| Setting | Value |
|---|---|
| Data | XAU/USD, H1 |
| Data range | January 2021 – April 2026 |
| Timezone | America/New_York (exchange timezone) |
| Validation | Sliding walk-forward with purge/embargo |
| Model | Classic Hybrid Stacking |
| Objective | Multiclass Short / Hold / Long |
| Backtest | Application demo only |
| Reproducibility | Seed 2024 |

---

## Config Sections

### `[data]` — Market Data Settings

```toml
[data]
symbol = "XAUUSD"                    # Ticker
timeframe = "1H"                     # Bar granularity (supports M, H, D, W)
market_tz = "America/New_York"       # Exchange timezone
tick_size = 0.01                     # Minimum price increment
contract_size = 100                  # Ounce value per lot
```

Hidden defaults (not in TOML):
- `symbol_download = "XAUUSD"` — download symbol override
- `asset_class = "fx"` — asset class for data source
- `download_concurrency = 20` — parallel download workers
- `download_max_retries = 7`
- `download_skip_current_month = true`

### `[data_range]` — Data Ingestion Boundaries

```toml
[data_range]
start = "2021-01-01"                # Ingestion start
end = "2026-04-30"                   # Ingestion end
```

These define the outer boundary of the dataset. Walk-forward windows slide within this range. The pipeline downloads and processes all data between `start` and `end`.

### `[validation]` — Walk-Forward Validation

```toml
[validation]
method = "sliding"               # "sliding" for walk-forward, other values fall back to static split
train_window_bars = 6240         # ~1 market year (24h * 5d * 52w)
test_window_bars = 1040          # ~2 market months (24h * 5d * ~8.5w)
step_bars = 1040                 # Non-overlapping step
purge_bars = 48                  # Gap between train/test to remove label lookahead
embargo_bars = 50                # Additional gap after test
min_train_bars = 6000            # Minimum bars required for training
```

Hidden defaults:
- `oof_ensemble = true` — concatenate OOF predictions across windows

When `method = "sliding"`, Stage 4 uses walk-forward training. Otherwise it falls back to static train/val/test split with `train_lgbm_fixed`.

### `[features]` — Feature Engineering

```toml
[features]
rsi_period = 14                  # RSI lookback
atr_period = 14                  # ATR lookback
adx_period = 14                  # ADX lookback
ema_slope_period = 20            # EMA slope lookback
macd_fast = 12                   # MACD fast EMA
macd_slow = 26                   # MACD slow EMA
macd_signal = 9                  # MACD signal line
correlation_threshold = 0.75     # Drop features above this pairwise correlation
```

Hidden defaults:
- `static_feature_cols` — defaults to `CORE_STATIC_FEATURES` from `constants.py` (21 features)
- `multi_timeframe.sma_periods = [50]`
- `multi_timeframe.ema_long = 200`
- `multi_timeframe.bb_period = 20`
- `multi_timeframe.bb_std = 2.0`
- `multi_timeframe.return_lookbacks = [1, 4, 24]` — generates `return_1h`, `return_4h`, `return_1d`
- `multi_timeframe.range_lookback = 20`
- `multi_timeframe.volume_zscore_period = 20`

### `[labels]` — Triple-Barrier Labeling

```toml
[labels]
atr_tp_multiplier = 2.0          # Take-profit = multiplier * ATR
atr_sl_multiplier = 2.0          # Stop-loss   = multiplier * ATR
horizon_bars = 24                # Forward-looking window (hours on H1)
```

Hidden defaults:
- `num_classes = 3` — Short / Hold / Long
- `min_atr = 0.5` — ATR floor to prevent near-zero barriers

If TP/SL multipliers change here, the backtest ATR barriers **must** be kept in sync (see `[backtest]`).

Safe rule: tune labels before tuning model complexity.

### `[model]` — Model Training

```toml
[model]
architecture = "stacking"        # "stacking" or "lgbm"
objective = "multiclass"         # Classification target

# LightGBM hyperparameters
lgbm_expanded_features = false   # Use expanded feature set for LightGBM
num_leaves = 15                  # Max leaves per tree
max_depth = 4                    # Max tree depth
learning_rate = 0.03             # Boosting learning rate
n_estimators = 300               # Number of boosting rounds
min_child_samples = 80           # Min samples per leaf
subsample = 0.80                 # Row subsample ratio
subsample_freq = 5               # Subsample every N rounds
feature_fraction = 0.70          # Column subsample per tree
reg_alpha = 0.05                 # L1 regularization
reg_lambda = 10.0                # L2 regularization
early_stopping_rounds = 30       # Stop if no improvement

# Stacking configuration
stacking_base_models = ["logistic_regression", "random_forest", "lightgbm"]
stacking_meta_model = "logistic_regression"
stacking_meta_fraction = 0.20    # Fraction of train window for meta-learner
stacking_passthrough = false     # Pass base features to meta-learner

# Random Forest parameters
random_forest_n_estimators = 300
random_forest_max_depth = 6
random_forest_min_samples_leaf = 80
```

Supported architectures:
- `"stacking"`: Classic Hybrid Stacking (current main path)
- `"lgbm"`: LightGBM-only ablation/baseline

Do not use GRU as the runtime architecture for the current thesis path.

### `[backtest]` — Trading Simulation

```toml
[backtest]
initial_capital = 10000.0         # Starting account equity
leverage = 10                     # Broker leverage
spread_ticks = 35                 # Bid-ask spread in ticks
slippage_ticks = 5                # Execution slippage in ticks
commission_per_lot = 10.0         # Round-turn commission per lot
atr_stop_multiplier = 2.0         # ATR stop-loss (MUST match [labels])
atr_tp_multiplier = 2.0           # ATR take-profit (MUST match [labels])
lots_per_trade = 0.02             # Default position size
min_lots = 0.01                   # Minimum lots
max_lots = 0.5                    # Maximum lots
confidence_threshold = 0.50       # Min model confidence to trade
min_bars_between_trades = 18      # Cooldown bars between entries
max_drawdown_cutoff = 0.30        # Halt if drawdown exceeds 30%
dd_cooldown_bars = 12             # Cooldown after max drawdown
max_open_positions = 1            # Concurrent positions
daily_loss_limit = 0.03           # Daily loss halt at 3%
```

Hidden defaults:
- `oob_start_date = ""` — out-of-backtest start date filter
- `oob_end_date = ""` — out-of-backtest end date filter

### `[workflow]` — Pipeline Controls

```toml
[workflow]
force_rerun = false               # Ignore cached intermediates
random_seed = 2024                # Global reproducibility seed
n_jobs = -1                       # Parallel workers (-1 = all cores)
```

Hidden defaults:
- `run_data_pipeline = true` — Stage 1 toggle
- `run_feature_engineering = true` — Stage 2 toggle
- `run_label_generation = true` — Stage 3 toggle
- `run_model_training = true` — Stage 4 toggle
- `run_backtest = true` — Stage 5 toggle
- `run_reporting = true` — Stage 6 toggle
- `cache_invalidation = "path"` — Cache strategy: `"path"`, `"hash"`, or `"none"`
- `session_timestamp = ""` — Set automatically per run

---

## Feature Set

Default model-facing features are defined in:

```text
src/thesis/shared/constants.py → CORE_STATIC_FEATURES
```

These 21 features are used by LightGBM and as input to the stacking meta-learner:

```text
# Trend
ema34_vs_ema89          EMA34-EMA89 crossover distance (ATR-normalized)
close_vs_ema_34         Close-EMA34 distance (ATR-normalized)
adx_14                  Wilder ADX trend strength
ema_slope_20            5-bar EMA slope (percent change)

# Momentum
return_1h               1-hour log return
return_4h               4-hour log return
macd_hist_atr           MACD histogram (ATR-normalized)
rsi_14                  Wilder RSI

# Volatility / Regime
atr_pct_close           ATR as percentage of close
atr_ratio               ATR(5) / ATR(20) ratio
atr_percentile          Rolling 50-bar ATR rank percentile
high_low_range_20       20-bar high-low range (ATR-normalized)

# Position / Location
price_dist_ratio        Close distance from EMA89 (ATR-normalized)
price_position_20       Close position in 20-bar range [0, 1]
pivot_position          Price position between previous-day S1/R1 [0, 1]
vwap                    Session VWAP (5PM NY-anchored trading day)

# Candle Structure
candle_body_ratio       |close - open| / (high - low)

# Session (24/5 market model, NY timezone)
sess_asia               Asian session dummy (18:00-01:59 ET)
sess_london             London session dummy (03:00-07:59 ET)
sess_ny_am              NY morning session dummy (08:00-11:59 ET)
sess_ny_pm              NY afternoon session dummy (12:00-17:59 ET)
```

Do not add raw OHLCV, timestamp, label, or barrier metadata as model-facing features. The exclusion set is enforced in `EXCLUDE_COLS`.

---

## Safe Tuning Order

1. **Label distribution** — check Short/Hold/Long balance after Stage 3
2. **Feature whitelist** — modify `CORE_STATIC_FEATURES` in `constants.py`
3. **Model regularization/capacity** — tune LightGBM params in `[model]`
4. **Backtest demo settings** — adjust `[backtest]` for presentation
5. **Report/docs wording** — update report text

When changing features, rerun from Stage 2. When changing labels, rerun from Stage 3.

---

## Validation Rules

The config loader (`src/thesis/shared/config.py`) enforces:

- **Unknown keys fail fast**: misspelled config keys raise `ValueError`
- **Unknown sections are warned**: unknown TOML sections are logged and ignored
- **Embargo scaling**: `embargo_bars` is automatically scaled by timeframe ratio
- **Directory creation**: `data/raw/` and `data/processed/` are created on load

---

## Config Caching

Pipeline stages cache their output. The caching behavior depends on `cache_invalidation`:

| Strategy | Behavior |
|---|---|
| `"path"` | Skip if output file exists (default) |
| `"hash"` | Append config fingerprint to filename; different config = different cache |
| `"none"` | Never skip, always rerun |

Use `--force` to override cache and rerun all enabled stages.

Config sections mapped to each stage for hash computation:

| Stage | Config Sections |
|---|---|
| 1 | `[data]` |
| 2 | `[features]` |
| 3 | `[labels]` |
| 4 | `[model]`, `[validation]` |
| 5 | `[backtest]`, `[labels]` |
| 6 | (none) |
