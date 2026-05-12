# Quickstart

## Prerequisites

- [Pixi](https://pixi.sh) package manager
- Python 3.13 (managed by Pixi)

## Install

```bash
pixi install
```

This installs all dependencies defined in `pixi.toml` including: polars, lightgbm, scikit-learn, numba, backtesting.py, pyecharts, streamlit, pandera, structlog.

## Run the Pipeline

### Full Run

```bash
pixi run workflow
```

This executes all 6 stages from scratch:
1. Data Preparation → 2. Feature Engineering → 3. Label Generation → 4. Model Training → 5. Backtest → 6. Reporting

### Run from a Specific Stage

```bash
pixi run python main.py --stage 2 --force    # Rebuild features + downstream
pixi run python main.py --stage 3 --force    # Rebuild labels + downstream
pixi run python main.py --stage 4 --force    # Retrain model + downstream
pixi run python main.py --stage 5 --force    # Re-run backtest + reporting
pixi run python main.py --stage 6 --force    # Re-generate report only
```

Stage behavior is downstream-oriented: running from Stage 3 continues through Stages 3-6.

### Resume an Existing Session

```bash
pixi run python main.py --session XAUUSD_1H_20260513_023811 --stage 4 --force
```

This loads the config snapshot from the existing session and re-runs from Stage 4.

### CLI Options

| Option | Description |
|---|---|
| `--config PATH` | Config file path (default: `config.toml`) |
| `--session NAME` | Resume from existing session directory name |
| `--stage N` | Start at Stage N, continue through Stage 6 (1-6) |
| `--force` | Force re-run, ignoring cache |

## Inspect Results

Results are written to timestamped session directories:

```text
results/XAUUSD_1H_<timestamp>/
```

### Key Files

```text
results/XAUUSD_1H_<timestamp>/
├── config/
│   ├── config_snapshot.toml          Config used for this run
│   └── session_info.json             Metadata: timing, config hash, validation params
├── predictions/
│   └── final_predictions.parquet     OOF predictions with timestamps
├── reports/
│   ├── thesis_report.md              Full thesis report
│   ├── model_metrics.json            All computed metrics
│   ├── model_comparison.csv          Model comparison table
│   ├── model_comparison.md           Markdown comparison
│   ├── model_comparison.json         Machine-readable comparison data
│   ├── model_evaluation.md           Evaluation summary
│   ├── walk_forward_history.json     Per-window training history
│   └── feature_importance.json       Sorted feature importance
├── backtest/
│   ├── backtest_results.json         Backtest metrics
│   ├── trades_detail.csv             Per-trade records
│   ├── equity_curve.csv              Running equity + drawdown
│   ├── equity_curve.png              Static equity curve image
│   ├── feature_importance.png        Feature importance chart
│   └── backtest_chart.html           Bokeh equity curve
├── models/
│   ├── lightgbm_model.pkl            Trained model
│   └── training_history.json         Training metadata
└── logs/
    └── pipeline.log                  Full pipeline log
```

## Data Artifacts

Intermediate data files are stored in:

```text
data/
├── raw/XAUUSD/                       Downloaded tick data (monthly files)
└── processed/
    ├── ohlcv.parquet                 Stage 1 output
    ├── features.parquet              Stage 2 output
    ├── features.feature_list.json    Feature column list sidecar
    ├── labels.parquet                Stage 3 output
    └── data_quality.json             Data quality report
```

## Validation Commands

```bash
# Lint
pixi run ruff check src

# Format check
pixi run ruff format --check src

# Syntax/import check
pixi run python -m compileall -q src tests

# Fast tests
pixi run test-fast

# Full tests
pixi run test
```

## Interactive Dashboard

```bash
pixi run dashboard
```

Launches a Streamlit dashboard for exploring session results, model metrics, charts, and backtest analysis. Features:
- Session selection and comparison
- Interactive metric cards with quality zone indicators
- Walk-forward training history visualization
- Feature importance charts
- Backtest equity curve and trade analysis
- OOF prediction analysis

## Current Runtime Reminder

The current runtime is Classic Hybrid Stacking:

```text
Logistic Regression + Random Forest + LightGBM -> Logistic Regression meta-model
```

Not GRU or any deep sequence model.
