# Roadmap

## Completed

### Core Pipeline

- [x] **Data preparation**: raw tick data → OHLCV H1 bars (`src/thesis/stage_1_data/`)
  - Dukascopy tick download with monthly file caching
  - OHLCV aggregation with deduplication and date filtering
  - Data quality checks: gap classification, candle consistency, outlier detection
  - Quality report JSON sidecar
- [x] **Feature engineering**: causal tabular indicators with static whitelist (`src/thesis/stage_2_features/`)
  - 21 core features across 6 categories (trend, momentum, volatility, position, candle, session)
  - Polars-native computation (no pandas in feature pipeline)
  - Pandera schema validation
  - Warmup row detection and drop
  - Feature list sidecar JSON
  - Leakage guard: no `shift(-n)`, no `center=True`
- [x] **Label generation**: triple-barrier Short/Hold/Long labels (`src/thesis/stage_3_labels/`)
  - Numba-accelerated barrier scanning
  - Asymmetric TP/SL with ATR-based barriers
  - Average-uniqueness sample weights (Lopez de Prado)
  - Censored label filtering
  - Label profitability diagnostics
- [x] **Walk-forward validation**: chronological sliding windows with purge/embargo (`src/thesis/stage_4_training/validation.py`)
  - Non-overlapping test windows
  - Configurable purge and embargo gaps
  - Event-time purge option
  - Window date logging for audit trail
- [x] **Classic Hybrid Stacking runtime**: Logistic Regression + Random Forest + LightGBM → meta Logistic Regression (`src/thesis/stage_4_training/walk_forward/stacking.py`)
  - Chronological base/meta split within each train window
  - Distribution-shift weighting
  - Balanced class weights
  - Validation filtering for unseen classes
  - OOF prediction persistence with manifest
- [x] **LightGBM-only ablation/baseline** (`src/thesis/stage_4_training/walk_forward/lgbm.py`)
  - Same walk-forward framework
  - Fixed hyperparameters with early stopping
  - Feature group interaction constraints
- [x] **Baseline strategies** (`src/thesis/stage_4_training/baselines.py`)
  - Naive direction, majority class, random, always-predict
  - Computed on same walk-forward windows for fair comparison
- [x] **Application-demo backtest** (`src/thesis/stage_5_backtest/`)
  - CFD signal simulation with fractional lots
  - Confidence-threshold filtering
  - Risk gates: max drawdown, daily loss limit, cooldown
  - ATR stop-loss/take-profit aligned with label barriers
  - Barrier alignment guard
  - Per-trade CSV, equity curve CSV, equity curve PNG, feature importance PNG, Bokeh chart
- [x] **Evaluation-first reporting**: classification metrics primary, backtest secondary (`src/thesis/stage_6_reporting/`)
  - Full classification metrics (accuracy, directional accuracy, macro F1, per-class F1, confusion matrix)
  - Calibration metrics (ECE, Brier, log-loss)
  - Model comparison table (CSV, MD, JSON)
  - Feature importance report
  - Metric quality zone assessment
  - Deployment recommendation engine
  - OOF vs OOS generalization check
  - Markdown report with 15 sections
- [x] **Interactive dashboard** (`src/thesis/dashboard/`)
  - Streamlit-based session explorer
  - Metric cards with quality zone indicators
  - Walk-forward training history
  - Chart visualization (pyecharts)
  - Session comparison
- [x] **Session management**: timestamped session directories with config snapshots
  - Config snapshot for reproducibility
  - Session manifest with config hash, timing, metadata
  - Session resume support
- [x] **Pipeline caching**: stage-level cache with path/hash/none strategies
- [x] **Feature pruning pass**: reduced model-facing whitelist from 25 to 21 features
- [x] **Leakage guard tests**: verify no look-ahead in features, no shift(-n), no center=True, unique OOF timestamps
- [x] **Comprehensive test suite**: unit tests, integration tests, leakage guard tests, config contract tests

### Latest Verified Run

```text
Session: results/XAUUSD_1H_20260513_023811/
Pipeline runtime: 75.65 seconds
Hybrid Stacking accuracy: 0.3416
Hybrid Stacking macro F1: 0.3152
LightGBM accuracy: 0.3738
LightGBM macro F1: 0.3265
Backtest demo return: 1.92%
Backtest demo PF: 1.109
Backtest demo Sharpe: 0.384
Backtest demo win rate: 47.17%
Backtest demo trades: 159
```

Interpretation: the pipeline runs end-to-end in ~76 seconds. Hybrid Stacking does not beat LightGBM in this run, which should be reported honestly. All models remain below the majority baseline (0.4850).

### Documentation

- [x] README updated to Classic Hybrid Stacking
- [x] `docs/ARCHITECTURE.md` — full pipeline and module reference
- [x] `docs/CONFIGURATION.md` — all config sections with hidden defaults
- [x] `docs/EVALUATION.md` — metrics, results, interpretation guide
- [x] `docs/QUICKSTART.md` — install, run, inspect, dashboard
- [x] `docs/TUNING.md` — safe tuning order with commands
- [x] `docs/ROADMAP.md` — completed and pending work
- [x] `docs/GLOSSARY.md` — plain-language term definitions
- [x] `bao_cao/` chapter drafts updated away from old runtime wording

---

## Pending

### Model / Label Improvements

- [ ] Improve label design so Hold is not too rare while Long/Short remain balanced
- [ ] Test confidence thresholding/calibration for better signal quality
- [ ] Analyze feature importance by market regime
- [ ] Compare `architecture = "lgbm"` against stacking as a strong simple baseline
- [ ] Test different `stacking_meta_fraction` values (current: 0.20)
- [ ] Experiment with stacking passthrough features (`stacking_passthrough = true`)

### Reporting Improvements

- [ ] Add a dedicated subsection explaining why Hybrid Stacking may underperform LightGBM on noisy financial data
- [ ] Add final thesis tables from the latest verified session
- [ ] Optionally add SHAP plots if runtime and dependencies are stable
- [ ] Add walk-forward window visualization to dashboard

### Backtest Enhancements

- [ ] Model swap/rollover costs
- [ ] Improve slippage assumptions
- [ ] Add parameter sensitivity analysis for spread, leverage, and confidence threshold
- [ ] Add benchmark comparison (buy-and-hold, random strategy, moving average crossover)

### Infrastructure

- [ ] Add Optuna hyperparameter optimization for LightGBM
- [ ] Add experiment tracking (MLflow or similar)
- [ ] Add CI/CD pipeline for automated testing and linting
- [ ] Add data versioning for OHLCV artifacts

---

## Current Recommendation

Do not reintroduce GRU/deep sequence runtime for the current thesis completion path. Finish the thesis narrative around controlled evaluation, transparent comparison, and honest limitations.

The thesis contribution is the controlled evaluation pipeline, not guaranteed market outperformance.
