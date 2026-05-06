# Configuration

This project keeps `config.toml` intentionally small. Public settings cover the experiment surface; hidden defaults live in `src/thesis/shared/config.py`.

## Current Default Profile

- Data: XAU/USD, 1H, `2018-01-01` to `2026-04-30`.
- Static split ratio: 70/15/15 by calendar months.
- Validation: sliding walk-forward, 2-year train windows, 6-month test windows.
- Model: GRU sequence encoder + LightGBM multiclass classifier.
- Volatility handling: model-facing volatility features use relative scale (`atr_pct_close`, `atr_ratio`, `macd_hist_atr`) instead of raw price-scale ATR/MACD. Raw `atr_14` is still retained in the features file as a label-barrier helper.
- Backtest: fixed small lot size; confidence filters entries only.

## Model Inputs

### GRU sequence input

Default GRU input has 20 features:

```toml
feature_cols = [
  "log_returns",
  "return_1h",
  "return_4h",
  "atr_pct_close",
  "atr_ratio",
  "close_vs_ema_34",
  "ema34_vs_ema89",
  "price_position_20",
  "candle_body_ratio",
  "macd_hist_atr",
  "rsi_14",
  "atr_percentile",
  "adx_14",
  "ema_slope_20",
  "regime_strength",
  "volume_zscore_20",
  "open_norm",
  "high_norm",
  "low_norm",
  "close_norm",
]
input_size = 20
hidden_size = 64
```

### LightGBM static input

Default static input has 22 features:

```text
Trend: ema34_vs_ema89, close_vs_ema_34, adx_14, ema_slope_20, regime_strength
Momentum: return_1h, return_4h, macd_hist_atr, rsi_14
Volatility: atr_pct_close, atr_ratio, atr_percentile, high_low_range_20
Position: price_dist_ratio, price_position_20, pivot_position
Candle: candle_body_ratio, upper_wick_ratio, lower_wick_ratio
Session: sess_london, sess_overlap
Volume: volume_zscore_20
```

With PCA-reduced GRU hidden states, the hybrid matrix is `16 GRU PCA features + 22 static features = 38 features`.

## Public `config.toml` Sections

### `[data]`

| Parameter | Default | Description |
| --- | --- | --- |
| `symbol` | `"XAUUSD"` | Display symbol for sessions and reports. |
| `timeframe` | `"1H"` | Bar timeframe. |
| `market_tz` | `"America/New_York"` | Timezone for session features. |
| `start_date` | `"2018-01-01"` | Inclusive data start. |
| `end_date` | `"2026-04-30"` | Inclusive data end. |
| `tick_size` | `0.01` | Minimum price movement. |
| `contract_size` | `100` | Units per lot for the backtest demo. |

### `[splitting]`

Static 70/15/15 calendar split across Jan 2018 through Apr 2026:

| Parameter | Default |
| --- | --- |
| `train_start` | `"2018-01-01"` |
| `train_end` | `"2023-09-30 23:59:59"` |
| `val_start` | `"2023-10-01"` |
| `val_end` | `"2025-01-31 23:59:59"` |
| `test_start` | `"2025-02-01"` |
| `test_end` | `"2026-04-30 23:59:59"` |

These dates are used by static split mode and report metadata. Default training uses walk-forward windows below.

### `[validation]`

| Parameter | Default | Description |
| --- | --- | --- |
| `method` | `"sliding"` | Sliding walk-forward or static split. |
| `train_window_bars` | `17520` | About 2 years of H1 bars. |
| `test_window_bars` | `4380` | About 6 months of H1 bars. |
| `step_bars` | `4380` | Non-overlapping test windows. |
| `purge_bars` | `48` | Anti-leakage gap; 2x the 24-bar label horizon. |
| `embargo_bars` | `50` | Extra gap after purge. |
| `min_train_bars` | `10000` | Minimum bars required to build a window. |

Sliding walk-forward retrains both GRU and LightGBM at every window. Window sizes are bar counts, so calendar duration can vary when market closures or missing bars exist.

### `[features]`

| Parameter | Default | Description |
| --- | --- | --- |
| `rsi_period` | `14` | RSI lookback. |
| `atr_period` | `14` | ATR lookback. |
| `adx_period` | `14` | ADX trend-strength lookback. |
| `ema_slope_period` | `20` | EMA span for slope/regime strength. |
| `macd_fast` | `12` | MACD fast EMA span. |
| `macd_slow` | `26` | MACD slow EMA span. |
| `macd_signal` | `9` | MACD signal span. |
| `correlation_threshold` | `0.75` | Feature filtering threshold. |

Hidden feature defaults include `static_feature_cols`, GRU `feature_cols`, and multi-timeframe helper parameters. Keep the public file compact unless you intentionally run feature-set experiments.

### `[labels]`

| Parameter | Default | Description |
| --- | --- | --- |
| `atr_tp_multiplier` | `2.0` | Take-profit barrier width. |
| `atr_sl_multiplier` | `2.0` | Stop-loss barrier width. |
| `horizon_bars` | `24` | Max label look-ahead bars. |

Backtest `atr_tp_multiplier` and `atr_stop_multiplier` must match these values.

### `[model]`

| Parameter | Default | Description |
| --- | --- | --- |
| `architecture` | `"hybrid"` | `"static"` LightGBM only or `"hybrid"` GRU + LightGBM. |
| `objective` | `"multiclass"` | 3-class Short/Hold/Long classifier. |
| `static_expanded` | `false` | Use all feature columns for static baseline if true. |
| `num_leaves` | `31` | LightGBM leaf count. |
| `max_depth` | `6` | Tree depth cap. |
| `learning_rate` | `0.02` | Boosting learning rate. |
| `n_estimators` | `500` | Max boosting rounds. |
| `min_child_samples` | `50` | Minimum samples per leaf. |
| `subsample` | `0.80` | Row subsample ratio. |
| `subsample_freq` | `5` | Row subsample frequency. |
| `feature_fraction` | `0.70` | Feature subsample ratio. |
| `reg_alpha` | `0.05` | L1 regularization. |
| `reg_lambda` | `5.0` | L2 regularization. |
| `early_stopping_rounds` | `40` | Stop after no validation improvement. |

### `[gru]`

| Parameter | Default | Description |
| --- | --- | --- |
| `objective` | `"multiclass"` | Stable default: focal loss on Short/Hold/Long labels. Regression is experimental. |
| `hidden_size` | `64` | GRU hidden state dimension. |
| `num_layers` | `2` | Stacked GRU layers. |
| `sequence_length` | `48` | Input sequence length in bars. |
| `dropout` | `0.3` | Dropout between recurrent layers. |
| `learning_rate` | `0.0005` | Adam learning rate. |
| `batch_size` | `256` | Training batch size. |
| `epochs` | `100` | Max epochs. |
| `patience` | `20` | Early-stopping patience. |
| `min_epochs` | `10` | Minimum epochs before stopping. |
| `bidirectional` | `false` | Disabled to avoid look-ahead bias. |
| `gradient_accumulation_steps` | `1` | Effective-batch scaling. |
| `warmup_epochs` | `3` | LR warmup before cosine schedule. |
| `contrastive_pretrain_epochs` | `10` | Triplet pretraining epochs. |
| `temperature_scaling` | `false` | Probability calibration toggle. |
| `pca_components` | `16` | PCA dimensions passed to LightGBM. |

### `[backtest]`

Backtest settings are demo controls, not the thesis proof. The thesis claim should rely on ML metrics first.

| Parameter | Default | Description |
| --- | --- | --- |
| `initial_capital` | `10000.0` | Starting equity. |
| `leverage` | `10` | Margin leverage. |
| `spread_ticks` | `35` | Spread cost. |
| `slippage_ticks` | `5` | Slippage cost. |
| `commission_per_lot` | `10.0` | Commission per standard lot. |
| `atr_stop_multiplier` | `2.0` | Must match label SL barrier. |
| `atr_tp_multiplier` | `2.0` | Must match label TP barrier. |
| `lots_per_trade` | `0.01` | Fixed conservative size. |
| `confidence_threshold` | `0.50` | Entry filter only. |
| `min_bars_between_trades` | `6` | Cooldown after exit. |
| `max_drawdown_cutoff` | `0.30` | Equity circuit breaker. |
| `dd_cooldown_bars` | `12` | Pause after drawdown breach. |
| `max_open_positions` | `1` | Single-position default. |
| `daily_loss_limit` | `0.03` | Daily loss circuit breaker. |

### `[workflow]`

| Parameter | Default | Description |
| --- | --- | --- |
| `force_rerun` | `false` | Ignore cache. |
| `random_seed` | `2024` | Reproducibility seed. |
| `n_jobs` | `-1` | Worker count. |

Stage toggles, cache invalidation, paths, download options, and runtime session paths are hidden defaults in `src/thesis/shared/config.py`.

## Fail-Fast Validation

Unknown keys inside known config sections raise `ValueError`. This prevents silent typos such as `timeframe_typo = "1H"`.

## Parameters Worth Changing

| Section | Parameter | Use |
| --- | --- | --- |
| `validation` | `train_window_bars`, `test_window_bars`, `purge_bars` | Evaluation shape and leakage safety. |
| `labels` | `atr_tp_multiplier`, `atr_sl_multiplier`, `horizon_bars` | Target definition. |
| `model` | `architecture`, tree complexity | Static vs hybrid comparison. |
| `gru` | `hidden_size`, `sequence_length`, `epochs`, `pca_components` | Temporal representation capacity. |
| `backtest` | `confidence_threshold`, `lots_per_trade`, cooldowns | Demo-only trade filtering/risk. |

## Evaluation Rules

Treat a result as useful only if it beats simple baselines:

| Metric | Minimum expectation |
| --- | --- |
| Accuracy | Higher than majority-class baseline. |
| Macro F1 | Better than predicting only Hold. |
| Directional accuracy | Higher than 50% on non-Hold rows. |
| High-confidence accuracy | Higher than full-sample accuracy. |

Backtest return is secondary. A profitable backtest with weak ML metrics is likely noise or overfitting.
