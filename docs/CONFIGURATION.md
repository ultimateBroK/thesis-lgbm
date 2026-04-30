# Configuration Guide

This project exposes only the parameters that matter for a student ML thesis.
Most financial-engineering details stay as code defaults so the report focuses
on reproducible modelling, not overfitting a trading system.

## Scope

The core experiment is:

1. Build OHLCV bars from raw data.
2. Create deterministic technical features.
3. Generate a 3-class target: `Short`, `Hold`, `Long`.
4. Validate with walk-forward time-series splits and purge/embargo gaps.
5. Train a compact hybrid model: GRU temporal embedding + LightGBM classifier,
   or a full stacking ensemble (configurable via `model.architecture`).
6. Report ML metrics first: accuracy, F1, baseline comparison, confusion matrix.
7. Use backtest metrics only as an application demo.

## Model Inputs

GRU sequence input:

```toml
feature_cols = ["log_returns", "rsi_14", "atr_14", "macd_hist", "return_4h", "bb_width"]
sequence_length = 48
hidden_size = 32
```

LightGBM static input is fixed in code for clarity:

```text
rsi_14, atr_14, macd_hist, atr_ratio, return_1h, return_4h,
bb_width, trend_strength, volume_zscore_20, sess_london, sess_overlap
```

This keeps the hybrid feature space compact: `32 GRU embedding features + 11
tabular features`.

---

## Section Reference

All defaults below match `config.toml` and `src/thesis/config.py` dataclasses.

### `[data]`

| Parameter | Default | Description |
| --- | --- | --- |
| `symbol` | `"XAUUSD"` | Display symbol used in session names and reports. |
| `timeframe` | `"1H"` | Bar timeframe. |
| `market_tz` | `"America/New_York"` | Timezone for session-aware feature engineering. |
| `start_date` | `"2013-01-01"` | Inclusive data start date. |
| `end_date` | `"2026-03-31"` | Inclusive data end date. |
| `tick_size` | `0.01` | Minimum price movement. |
| `contract_size` | `100` | Units per trading lot (for backtest demo). |

### `[validation]`

| Parameter | Default | Description |
| --- | --- | --- |
| `method` | `"sliding"` | Validation method: `"sliding"` (walk-forward) or `"static"` (fixed split). |
| `train_window_bars` | `26280` | Training window size (~3 years of H1 bars). |
| `test_window_bars` | `4380` | Test window size (~6 months of H1 bars). |
| `step_bars` | `4380` | Step between consecutive windows. Equals `test_window_bars` for non-overlapping folds. |
| `purge_bars` | `25` | Bars removed at the train/test boundary to prevent label leakage. |
| `embargo_bars` | `50` | Additional gap after purge for extra safety. |
| `min_train_bars` | `10000` | Minimum training bars required to produce a window. Windows below this are skipped. |
| `oof_ensemble` | `true` | Aggregate out-of-fold predictions across all walk-forward windows. |
| `wf_optuna_trials` | `0` | Optuna hyperparameter trials per walk-forward window. `0` means fixed params (faster, more reproducible). |

### `[features]`

| Parameter | Default | Description |
| --- | --- | --- |
| `rsi_period` | `14` | RSI lookback period. |
| `atr_period` | `14` | ATR lookback period. |
| `macd_fast` | `12` | MACD fast EMA span. |
| `macd_slow` | `26` | MACD slow EMA span. |
| `macd_signal` | `9` | MACD signal EMA span. |
| `correlation_threshold` | `0.75` | Threshold for correlation-based feature filtering. |
| `static_feature_cols` | *(see below)* | Whitelist of tabular features consumed by LightGBM. |

Default `static_feature_cols`:

```toml
static_feature_cols = [
  "rsi_14", "atr_14", "macd_hist", "atr_ratio",
  "return_1h", "return_4h", "bb_width", "trend_strength",
  "volume_zscore_20", "sess_london", "sess_overlap",
]
```

### `[labels]`

| Parameter | Default | Description |
| --- | --- | --- |
| `atr_multiplier` | `2.5` | ATR multiplier for triple-barrier width. Defines target classes. |
| `horizon_bars` | `24` | Maximum bars before forced exit if no barrier hit. |

### `[model]`

| Parameter | Default | Description |
| --- | --- | --- |
| `architecture` | `"hybrid"` | Model architecture: `"hybrid"` (GRU embedding + LightGBM) or `"stacking"` (full stacking ensemble). |
| `use_optuna` | `false` | Enable Optuna hyperparameter search for LightGBM. |
| `num_leaves` | `31` | LightGBM leaf count. Controls tree complexity. |
| `max_depth` | `6` | LightGBM max tree depth. |
| `learning_rate` | `0.05` | LightGBM learning rate. |
| `n_estimators` | `200` | Maximum boosting iterations. |
| `min_child_samples` | `150` | Minimum samples per leaf. Higher = more conservative. |
| `subsample` | `0.80` | Row subsample ratio per iteration. |
| `feature_fraction` | `0.70` | Feature subsample ratio per iteration. |
| `reg_lambda` | `5.0` | L2 regularization. |
| `early_stopping_rounds` | `30` | Stop training if validation metric does not improve for this many rounds. |

### `[gru]`

| Parameter | Default | Description |
| --- | --- | --- |
| `feature_cols` | *(see below)* | Input features for the GRU sequence. |
| `hidden_size` | `32` | GRU hidden state dimension. |
| `num_layers` | `2` | Number of stacked GRU layers. |
| `sequence_length` | `48` | Number of bars in each input sequence. |
| `dropout` | `0.2` | Dropout between GRU layers. |
| `learning_rate` | `0.001` | Adam optimizer learning rate. |
| `batch_size` | `256` | Training batch size. |
| `epochs` | `25` | Maximum training epochs. |
| `patience` | `5` | Early-stopping patience (epochs without improvement). |
| `min_epochs` | `5` | Minimum epochs before early-stopping can trigger. |

Default `feature_cols`:

```toml
feature_cols = ["log_returns", "rsi_14", "atr_14", "macd_hist", "return_4h", "bb_width"]
```

### `[stacking]`

| Parameter | Default | Description |
| --- | --- | --- |
| `base_models` | `["gru", "lgbm"]` | List of base models that generate meta-features. |
| `meta_model` | `"lightgbm"` | Meta-learner type for the second stacking stage. |
| `use_probability_features_only` | `true` | If true, meta-features are base-model class probabilities only (no raw features). |
| `min_meta_train_folds` | `1` | Minimum walk-forward folds required to train the meta-model. |
| `min_meta_train_rows` | `500` | Minimum rows required for meta-model training. |
| `final_refit` | `true` | If true, refit all models on full training data after stacking validation. |

### `[backtest]`

| Parameter | Default | Description |
| --- | --- | --- |
| `initial_capital` | `10000.0` | Starting equity in account currency. |
| `leverage` | `10` | Margin leverage (margin = 1/leverage). |
| `spread_ticks` | `35` | Spread in ticks applied on entry/exit. |
| `slippage_ticks` | `5` | Slippage in ticks applied on execution. |
| `commission_per_lot` | `10.0` | Commission per lot per trade. |
| `atr_stop_multiplier` | `1.0` | ATR multiplier for stop-loss distance. |
| `atr_tp_multiplier` | `2.0` | ATR multiplier for take-profit distance (`0` = disabled). |
| `lots_per_trade` | `0.1` | Base lot size for position sizing. |
| `min_lots` | `0.05` | Minimum lot size (low-conviction floor). |
| `max_lots` | `0.1` | Maximum lot size (high-conviction cap). |
| `confidence_threshold` | `0.55` | Minimum predicted probability to open a trade (`0` = disabled). |
| `max_drawdown_cutoff` | `0.30` | Circuit breaker: stop if equity drops below `peak * (1 - cutoff)`. |
| `dd_cooldown_bars` | `12` | Bars to pause trading after a drawdown cutoff breach. |
| `max_open_positions` | `1` | Maximum simultaneous open positions. |
| `daily_loss_limit` | `0.03` | Stop trading for the day after a `-N` equity drawdown (e.g. 3%). |

### `[workflow]`

| Parameter | Default | Description |
| --- | --- | --- |
| `force_rerun` | `false` | Ignore cache and rerun all pipeline stages. |
| `random_seed` | `2024` | Global random seed for reproducibility. |
| `n_jobs` | `-1` | Parallel worker count (`-1` = all CPUs). |

### `[paths]`

| Parameter | Default | Description |
| --- | --- | --- |
| `data_raw` | `"data/raw/XAUUSD"` | Raw tick/data directory. |
| `data_processed` | `"data/processed"` | Processed parquet output directory. |
| `ohlcv` | `"data/processed/ohlcv.parquet"` | OHLCV bars parquet. |
| `features` | `"data/processed/features.parquet"` | Engineered features parquet. |
| `labels` | `"data/processed/labels.parquet"` | Label parquet. |
| `train_data` | `"data/processed/train.parquet"` | Training split. |
| `val_data` | `"data/processed/val.parquet"` | Validation split. |
| `test_data` | `"data/processed/test.parquet"` | Test split. |
| `model` | `"models/lightgbm_model.pkl"` | LightGBM model artifact. |
| `gru_model` | `"models/gru_model.pt"` | GRU model artifact. |
| `predictions` | `"data/predictions/final_predictions.parquet"` | Final predictions. |
| `backtest_results` | `"results/backtest_results.json"` | Backtest output. |
| `report` | `"results/thesis_report.md"` | Generated report. |

---

## Parameters Worth Changing

Use these for experiments:

| Section | Parameter | Why it matters |
| --- | --- | --- |
| `validation` | `train_window_bars`, `test_window_bars` | Controls time-series evaluation stability. |
| `validation` | `wf_optuna_trials` | Per-window hyperparameter tuning (0 = off). |
| `stacking` | `base_models`, `meta_model` | Controls ensemble composition and meta-learner. |
| `labels` | `atr_multiplier`, `horizon_bars` | Defines the target classes. This changes the ML problem. |
| `model` | `architecture` | Switches between `"hybrid"` and `"stacking"`. |
| `model` | `num_leaves`, `max_depth`, `n_estimators` | Controls LightGBM capacity and overfitting. |
| `gru` | `hidden_size`, `sequence_length`, `epochs` | Controls temporal model capacity and runtime. |
| `backtest` | `confidence_threshold`, `lots_per_trade` | Demo-only risk/filter controls. Do not use them to claim model quality. |

## Default Experiment Profile

The default `config.toml` is intentionally conservative:

```toml
[model]
architecture = "hybrid"
use_optuna = false
num_leaves = 31
max_depth = 6
n_estimators = 200

[gru]
hidden_size = 32
epochs = 25
patience = 5
batch_size = 256

[validation]
method = "sliding"
wf_optuna_trials = 0

[stacking]
base_models = ["gru", "lgbm"]
meta_model = "lightgbm"
final_refit = true
```

This gives faster runs and more repeatable comparisons. Use Optuna only after
you have a stable baseline table; otherwise the thesis can look like parameter
search instead of model engineering.

## Evaluation Rules

Treat a result as useful only if it beats simple baselines:

| Metric | Minimum expectation |
| --- | --- |
| Exact accuracy | Higher than majority-class baseline. |
| Macro F1 | Better than predicting only `Hold`. |
| Directional accuracy | Higher than 50% on non-Hold predictions. |
| High-confidence accuracy | Higher than full-sample accuracy. |

Backtest return is secondary. A profitable backtest with weak ML metrics is not
a reliable thesis result; it is likely noise or overfitting.
