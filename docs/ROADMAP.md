# Roadmap

> What is done and what is still pending.

---

## Completed

### Core Pipeline

- [x] **Data preparation** — Raw tick data to OHLCV bars aggregation (`src/thesis/stage_1_data/`)
- [x] **Feature engineering** — 28 technical indicators including regime features (ADX, EMA slope, regime strength), candle structure, and session dummies (`src/thesis/stage_2_features/`)
- [x] **Data quality analysis** — Missing bars, OHLC consistency, volatility distribution, seasonal patterns, label drift (`src/thesis/stage_6_reporting/`)
- [x] **Label generation** — Triple Barrier method with symmetric 2xATR take-profit and stop-loss (`src/thesis/stage_3_labels/`)
- [x] **Walk-forward validation** — Rolling window cross-validation with purge and embargo; distribution-shift weight correction per window (`src/thesis/stage_4_training/validation.py`)
- [x] **Correlation filtering** — Automatic removal of highly correlated features (>0.75) computed on train set only

### Models

- [x] **GRU feature extractor** — 2-layer GRU, 64 hidden units, 48-bar sequences, 20 input features including raw OHLCV z-scores and relative volatility features (`src/thesis/stage_4_training/gru/`)
- [x] **GRU multiclass objective** — GRU trains on Short/Hold/Long class labels; regression remains experimental
- [x] **Cosine-annealing LR schedule** — Warm restarts (T_0=10, T_mult=2) with warmup for better convergence
- [x] **LightGBM classifier** — Multiclass with distribution-shift weight correction (`src/thesis/stage_4_training/lgbm/training.py`)
- [x] **Hybrid training pipeline** — GRU hidden states (PCA 16-dim) + 22 static features → LightGBM (`src/thesis/pipeline.py`, orchestrated via `stage_4_training/walk_forward/`)


### Evaluation

- [x] **Evaluation-first report restructure** — Report reorganised around ML metrics (classification accuracy, F1, confusion matrix) as primary evidence; backtest demoted to application demo (`src/thesis/stage_6_reporting/generation.py`)
- [x] **Classification metrics as primary** — Accuracy, macro-F1, per-class precision/recall, directional accuracy, high-confidence accuracy computed in `model_metrics.py`; rendered first in report
- [x] **Regression auxiliary metrics** — MAE, RMSE, R² on return magnitude as supplementary evidence alongside classification (`model_metrics.py`)
- [x] **Baseline model comparison** — Naive direction (persistence), majority-class, random, and buy-and-hold baselines computed on same walk-forward windows as hybrid model (`baselines.py`); comparison table in report
- [x] **Probability calibration** — ECE, Brier score, log-loss, confidence-bin reliability (`calibration.py`)
- [x] **Data quality evidence** — OHLCV consistency, missing-bar gaps, outlier returns, label distribution, volatility regime statistics (`data_quality.py`); rendered as evidence section in report
- [x] **Metric zone gauges** — Color-coded metric evaluation (green/yellow/red) with boringedge recommendations (`shared/zones.py`)
- [x] **CFD backtest** — via `backtesting.py` with native margin, spread, commission, ATR stop-loss, circuit breakers, trade cooldown (`src/thesis/stage_5_backtest/`)
- [x] **Fixed-risk sizing after confidence filter** — Confidence filters trades but no longer amplifies lot size
- [x] **Trade cooldown** — min_bars_between_trades=6 prevents overtrading
- [x] **Fractional lot support** — Uses `FractionalBacktest` for precise sizing
- [x] **Comprehensive metrics** — 20+ trading metrics (Sharpe, Sortino, Calmar, SQN, drawdown, etc.)
- [x] **Trade details CSV** — Per-trade export with entry/exit, P&L, duration, confidence
- [x] **Prediction detail CSV** — Per-row predictions with confidence scores and probability columns
- [x] **Equity curve CSV** — Exported equity + drawdown series for external analysis
- [x] **OOF vs OOS comparison** — Out-of-fold vs out-of-sample metric comparison in report
- [x] **Benchmark comparison** — Random strategy baseline and buy-and-hold comparison in report

### Visualization

- [x] **Data charts** — Candlestick, label distribution, feature correlation, feature distributions (`src/thesis/charts/`)
- [x] **Model charts** — Confusion matrix, confidence distribution, feature importance, SHAP summary (`src/thesis/charts/`)
- [x] **Backtest charts** — Equity curve, drawdown, P&L histogram, monthly returns heatmap, rolling Sharpe, duration vs P&L scatter (`src/thesis/charts/`)
- [x] **Metric zone gauges** — Color-coded metric evaluation (green/yellow/red) with boringedge recommendations and extreme value detection (`src/thesis/shared/zones.py`)
- [x] **Auto-generated report** — Markdown report with all metrics, tables, charts, data quality analysis, and verdict (`src/thesis/stage_6_reporting/`)
- [x] **Streamlit dashboard** — Modular ECharts-based visualization on :8501 (`src/thesis/dashboard/` — 10 modules, entry via `main.py`)

### Infrastructure

- [x] **Config management** — Single TOML config with typed dataclasses (`src/thesis/shared/config.py`)
- [x] **Session-based output** — Timestamped results folder (local time) for each run (`src/thesis/shared/session_paths.py`)
- [x] **CLI entry point** — `main.py` with `--force` flag
- [x] **Pixi package management** — Reproducible environment with `pixi.toml`
- [x] **Test suite** — Unit and integration tests with 60% coverage minimum
- [x] **Code quality** — Ruff linting and formatting
- [x] **CI/CD workflows** — GitHub Actions for testing and releases
- [x] **Git conventions** — Conventional commits, branch strategy, PR templates
- [x] **Stage-based subpackages** — Source organized as `stage_1_data/`, `stage_2_features/`, etc. with sub-packages for GRU (`gru/`), walk-forward (`walk_forward/`), report sections (`sections/`), charts, and dashboard. Shared utilities in `shared/`.

### Documentation

- [x] **Architecture doc** — System design and data flow (`docs/ARCHITECTURE.md`)
- [x] **Quickstart guide** — Setup and first run (`docs/QUICKSTART.md`)
- [x] **Evaluation guide** — Metric definitions and interpretation (`docs/EVALUATION.md`)
- [x] **Configuration reference** — All config keys and defaults (`docs/CONFIGURATION.md`)
- [x] **Glossary** — Trading and ML term definitions (`docs/GLOSSARY.md`)
- [x] **Tuning guide** — Hyperparameter tuning tips (`docs/TUNING.md`)
- [x] **API docstrings** — Google-style docstrings on all public functions

---

## Pending

### Model Improvements

- [ ] **Transformer encoder** — Experiment with self-attention as an alternative to GRU
- [ ] **Multi-timeframe features** — Add features from 4H and daily timeframes (config scaffolding exists in `MultiTimeframeConfig`)
- [ ] **Sentiment features** — Incorporate news sentiment or macro indicators
- [ ] **Volume profile analysis** — Add volume-based features (VWAP, volume clusters)

### Backtest Enhancements

- [ ] **Swap/rollover costs** — Model overnight holding costs in the CFD simulator
- [ ] **Slippage model** — Add realistic slippage based on volatility and liquidity
- [ ] **Multi-asset support** — Extend to other currency pairs (EUR/USD, GBP/USD)
- [ ] **Monte Carlo simulation** — Randomize trade order to test robustness of metrics
- [ ] **Parameter sensitivity analysis** — Test how small changes in spread, leverage, etc. affect results

### Operational

- [ ] **Real-time inference** — Deploy model for live signal generation
- [ ] **Model versioning** — Track and compare model versions across experiments
- [ ] **Data pipeline monitoring** — Alert when data quality drops or drifts

### Research

- [ ] **Experiment log** — Track all experiments with parameters and results
- [ ] **Literature comparison** — Compare results with published research benchmarks
- [ ] **Statistical significance tests** — Add Diebold-Mariano or similar tests for model comparison

---

## Progress Summary

| Category          | Completed | Pending | Total |
|-------------------|-----------|---------|-------|
| Core Pipeline     | 6         | 0       | 6     |
| Models            | 5         | 4       | 9     |
| Evaluation        | 17        | 5       | 22    |
| Visualization     | 6         | 0       | 6     |
| Infrastructure    | 9         | 0       | 9     |
| Documentation     | 7         | 0       | 7     |
| Operational       | 0         | 3       | 3     |
| Research          | 0         | 3       | 3     |
| **Total**         | **50**    | **15**  | **65**|

> **Overall: 77% complete** — Core research pipeline is fully functional. Evaluation-first report restructure done. Remaining items are enhancements and production features.
