# Roadmap

> What is done and what is still pending.

---

## Completed

### Core Pipeline

- [x] **Data preparation** — Raw tick data to OHLCV bars aggregation (`src/thesis/data.py`)
- [x] **Feature engineering** — 11 technical indicators: RSI, ATR, MACD, ATR ratio, price distance, pivot position, ATR percentile, 4 session dummies (`src/thesis/features.py`)
- [x] **Label generation** — Triple Barrier method with ATR-based take-profit and stop-loss (`src/thesis/labels.py`)
- [x] **Walk-forward validation** — Rolling window cross-validation with purge and embargo; default training mode (`src/thesis/validation.py`)
- [x] **Correlation filtering** — Automatic removal of highly correlated features (>0.75) computed on train set only

### Models

- [x] **GRU feature extractor** — 2-layer GRU, 32 hidden units, 48-bar sequences (`src/thesis/gru.py`)
- [x] **LightGBM classifier** — Tuned hyperparameters with class weight balancing (`src/thesis/model.py`)
- [x] **Hybrid training pipeline** — GRU hidden states + static features → LightGBM (`src/thesis/pipeline.py`)
- [x] **Stacking ensemble** — Multi-window base models + meta-model for final predictions (`src/thesis/pipeline.py`)
- [x] **SHAP feature importance** — Interpretability analysis of the hybrid model (`src/thesis/model.py`)
- [x] **Optuna hyperparameter search** — Integrated auto-tuning with `optuna_trials` and `optuna_timeout` config options

### Evaluation

- [x] **CFD backtest** — via `backtesting.py` with native margin, spread, commission, ATR stop-loss, circuit breakers (`src/thesis/backtest.py`)
- [x] **Fixed lot position sizing** — Prevents runaway sizing with leverage
- [x] **Fractional lot support** — Uses `FractionalBacktest` for precise sizing
- [x] **Comprehensive metrics** — 20+ trading metrics (Sharpe, Sortino, Calmar, SQN, drawdown, etc.)
- [x] **Trade details CSV** — Per-trade export with entry/exit, P&L, duration, confidence
- [x] **Equity curve CSV** — Exported equity + drawdown series for external analysis
- [x] **Benchmark comparison** — Random strategy baseline and buy-and-hold comparison in report

### Visualization

- [x] **Data charts** — Candlestick, label distribution, feature correlation, feature distributions (`src/thesis/charts.py`)
- [x] **Model charts** — Confusion matrix, confidence distribution, feature importance, SHAP summary (`src/thesis/charts.py`)
- [x] **Backtest charts** — Equity curve, drawdown, P&L histogram, monthly returns heatmap, rolling Sharpe, duration vs P&L scatter (`src/thesis/charts.py`)
- [x] **Metric zones** — Color-coded metric evaluation (green/amber/red) with recommended ranges (`src/thesis/zones.py`)
- [x] **Auto-generated report** — Markdown report with all metrics, tables, charts, and verdict (`src/thesis/report.py`)
- [x] **Streamlit dashboard** — Modular ECharts-based visualization on :8501 (`src/thesis/dashboard.py`)

### Infrastructure

- [x] **Config management** — Single TOML config with typed dataclasses (`src/thesis/config.py`)
- [x] **Session-based output** — Timestamped results folder (local time) for each run (`src/thesis/session_paths.py`)
- [x] **CLI entry point** — `main.py` with `--force` and `--ablation` flags
- [x] **Pixi package management** — Reproducible environment with `pixi.toml`
- [x] **Test suite** — Unit and integration tests with 60% coverage minimum
- [x] **Code quality** — Ruff linting and formatting
- [x] **CI/CD workflows** — GitHub Actions for testing and releases
- [x] **Git conventions** — Conventional commits, branch strategy, PR templates
- [x] **Flat module layout** — All source in `src/thesis/` as flat modules; no nested packages

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
| Core Pipeline     | 5         | 0       | 5     |
| Models            | 6         | 4       | 10    |
| Evaluation        | 7         | 5       | 12    |
| Visualization     | 6         | 0       | 6     |
| Infrastructure    | 9         | 0       | 9     |
| Documentation     | 7         | 0       | 7     |
| Operational       | 0         | 3       | 3     |
| Research          | 0         | 3       | 3     |
| **Total**         | **40**    | **15**  | **55**|

> **Overall: 73% complete** — Core research pipeline is fully functional. Remaining items are enhancements and production features.
