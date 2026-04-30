# Quickstart

> Step-by-step guide to run the project from scratch.

---

## Prerequisites

Before you begin, make sure you have:

- **Pixi** installed ([install guide](https://pixi.sh/latest/))
- **Git** installed
- **At least 2 GB of free disk space** (for data and models)
- **8 GB RAM minimum** (16 GB recommended for training)

---

## Setup Overview

```mermaid
flowchart LR
    A["Clone"] --> B["Install"]
    B --> C["Get Data"]
    C --> D["Run"]
    D --> E["View Results"]

    style A fill:#2563EB,color:#fff
    style B fill:#2563EB,color:#fff
    style C fill:#2563EB,color:#fff
    style D fill:#7C3AED,color:#fff
    style E fill:#059669,color:#fff
```

---

## Step 1: Clone the Repository

```bash
git clone <your-repo-url> thesis
cd thesis
```

---

## Step 2: Install Dependencies

Pixi will download and install all required packages automatically:

```bash
pixi install
```

This creates an isolated environment with Python 3.13 and all libraries (PyTorch, LightGBM, Polars, etc.).

> **Note:** The first install may take a few minutes because it downloads PyTorch and other large packages.

---

## Step 3: Get the Data

Place your raw XAU/USD tick data in the correct folder:

```bash
data/raw/XAUUSD/
```

Each file should be a **parquet file** containing tick data for one month. The columns expected are:

| Column | Description |
|--------|-------------|
| `timestamp` | Date and time of the tick |
| `bid` | Bid price |
| `ask` | Ask price |

> Alternatively, use the built-in downloader:
> ```bash
> pixi run data
> ```
> This downloads XAU/USD data from 2013 onwards (configured via `start_date` in `config.toml`).

---

## Step 4: Run the Full Pipeline

The simplest command — runs everything from data preparation to report generation:

```bash
pixi run workflow
```

This command:

```mermaid
flowchart TD
    S0["Stage 0: Convert ticks → OHLCV"] --> S1["Stage 1: Generate 11 indicators"]
    S1 --> S2["Stage 2: Triple-barrier labeling"]
    S2 --> S3["Stage 3: Walk-forward training<br/><i>GRU + LightGBM per window</i>"]
    S3 --> S4["Stage 4: CFD backtest<br/><i>on concatenated OOF predictions</i>"]
    S4 --> S5["Stage 5: Report + charts"]

    style S0 fill:#2563EB,color:#fff
    style S1 fill:#2563EB,color:#fff
    style S2 fill:#2563EB,color:#fff
    style S3 fill:#7C3AED,color:#fff
    style S4 fill:#059669,color:#fff
    style S5 fill:#059669,color:#fff
```

> **First run:** This may take 10–30 minutes depending on your hardware.
> **Subsequent runs:** Stages are cached — only changed stages re-run.

### Architecture Choice

The pipeline supports two model architectures, controlled by `model.architecture` in `config.toml`:

| Architecture | Description | Config |
|---|---|---|
| **hybrid** (default) | GRU hidden states concatenated with static features → LightGBM | `model.architecture = "hybrid"` |
| **stacking** | Full stacking ensemble — GRU and LightGBM as base learners, meta-learner on top | `model.architecture = "stacking"` |

Both architectures use walk-forward sliding-window validation when `validation.method = "sliding"`.

---

## Step 5: View the Results

After the pipeline finishes, look in the `results/` folder:

```bash
results/XAUUSD_1H_YYYYMMDD_HHMMSS/
```

### Key Files to Check

| File | What It Contains |
|------|-----------------|
| `reports/thesis_report.md` | Full written report with metrics and charts |
| `reports/walk_forward_history.json` | Window boundaries, OOF prediction counts |
| `backtest/backtest_results.json` | Trading metrics (win rate, return, Sharpe, etc.) |
| `backtest/trades_detail.csv` | Trade-by-trade breakdown (entry/exit, PnL, duration) |
| `backtest/equity_curve.csv` | Equity curve data points over time |
| `backtest/backtest_chart.html` | Interactive Bokeh equity chart |
| `reports/charts/` | All visualization charts |
| `config/config_snapshot.toml` | The exact config used for this run |
| `config/session_info.json` | Session metadata (run ID, timestamps, stage durations) |
| `logs/pipeline.log` | Detailed execution log (ANSI-stripped) |

### Stacking-Only Artifacts

When using `model.architecture = "stacking"`, additional files appear:

| File | What It Contains |
|------|-----------------|
| `models/stacking_bundle.joblib` | Deployment bundle (model paths, class order, feature columns) |
| `models/base_oof_predictions.parquet` | Out-of-fold predictions from base learners |
| `models/lgbm_base_model.pkl` | Standalone LightGBM base model |
| `models/training_history.json` | Per-model training details (iterations, feature cols) |

### Quick Look at Results

```bash
# View the report
cat results/*/reports/thesis_report.md

# View trading metrics
cat results/*/backtest/backtest_results.json

# List all charts
ls results/*/reports/charts/*/
```

---

## Other Useful Commands

| Command | What It Does |
|---------|-------------|
| `pixi run force` | Re-run all stages (ignoring cache) |
| `pixi run test` | Run all tests with coverage |
| `pixi run lint` | Check code for style issues |
| `pixi run format` | Auto-format code |
| `pixi run streamlit` | Interactive Streamlit dashboard (:8501) |
| `pixi run pre-commit` | Lint + format + fast tests |
| `pixi run clean-cache` | Delete processed data files |
| `pixi run clean-all` | Delete processed data + models + results |

---

## Running Individual Stages

If you want to run just one stage, use the `workflow` toggles in `config.toml`:

```toml
[workflow]
run_data_pipeline = false      # Skip data preparation
run_feature_engineering = true  # Run feature engineering
run_label_generation = false    # Skip label generation
run_model_training = true       # Run model training
run_backtest = true             # Run backtest
run_reporting = true            # Generate report
force_rerun = false             # Set true to ignore cache
```

Then run:

```bash
pixi run workflow
```

Only the stages set to `true` will execute.

---

## Changing Settings

All settings live in **`config.toml`**. Edit this file to change:

- Date ranges (which years of data to use)
- Model parameters (learning rate, number of trees, etc.)
- Backtest parameters (capital, leverage, spread, etc.)
- GRU architecture (hidden size, layers, sequence length, etc.)
- Model architecture (`hybrid` vs `stacking`)

See the [Configuration Guide](CONFIGURATION.md) for detailed instructions.

---

## Troubleshooting

### "No module named 'thesis'"

Make sure you're using pixi to run commands:

```bash
pixi run workflow    # Correct
python main.py       # May not find the package
```

### "File not found: data/raw/XAUUSD/*.parquet"

You need raw tick data. Either:
1. Place parquet files manually in `data/raw/XAUUSD/`
2. Run `pixi run data` to download

### "CUDA out of memory"

The GRU training uses very little GPU memory. If you still see this error:

```toml
[gru]
batch_size = 256    # Reduce if needed (e.g., 128 or 64)
```

### "Pipeline says 'skipping' stages"

Stages are cached. To force re-run everything:

```bash
pixi run force
```

Or set in config.toml:

```toml
[workflow]
force_rerun = true
```

---

## Running Tests

```bash
# All tests with coverage
pixi run test

# Unit tests only
pixi run test-unit

# Integration tests only
pixi run test-integration

# Fast tests (skip slow ones)
pixi run test-fast

# Single test file
pixi run pytest tests/unit/test_features.py

# Single test function
pixi run pytest tests/unit/test_features.py::test_rsi_bounds
```

---

## Quick Command Reference

```bash
# === Setup ===
pixi install                    # Install all dependencies

# === Data ===
pixi run data                   # Download XAU/USD data

# === Pipeline ===
pixi run workflow               # Run full pipeline
pixi run force                  # Force re-run everything

# === Visualization ===
pixi run streamlit              # Interactive dashboard on :8501

# === Code Quality ===
pixi run lint                   # Check code style
pixi run format                 # Format code
pixi run pre-commit             # Lint + format + test

# === Testing ===
pixi run test                   # Run tests with coverage
pixi run test-unit              # Unit tests only
pixi run test-fast              # Skip slow tests

# === Cleanup ===
pixi run clean-cache            # Delete processed data
pixi run clean-all              # Delete everything generated
```
