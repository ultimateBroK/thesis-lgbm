# Architecture

> A high-level overview of how this project is built.

---

## What Does This Project Do?

This project builds a reproducible **time-series classification pipeline** on
gold (XAU/USD) 1-hour data. It supports **two architectures**:

1. **Hybrid** (default) — GRU hidden states + static features → LightGBM.
2. **Stacking** — GRU and LightGBM as base learners → LightGBM meta-learner on OOF probabilities.

Both use **walk-forward sliding window validation** to produce out-of-fold predictions,
which are then backtested and reported.

The primary output is an ML evaluation report. The backtest is included as an
application demo for the predicted classes, not as the main thesis claim.

---

## The Big Picture

```mermaid
flowchart LR
    A["Raw Tick Data"] --> B["Prepare<br/>OHLCV"]
    B --> C["Features<br/>21 indicators"]
    C --> D["Labels<br/>3-class target"]
    D --> E["Walk-Forward<br/>Sliding Windows"]

    E --> F["GRU<br/>64 hidden states"]
    E --> G["Static<br/>21 core features"]

    F --> H["LightGBM<br/>85 features"]
    G --> H

    H --> I["Concatenate<br/>OOF Predictions"]
    I --> J["Backtest<br/>Application Demo"]
    I --> K["ML Report<br/>Charts + Markdown"]
```

---

## Pipeline Stages

The pipeline has **6 stages** (0–5) in walk-forward mode (the default).

```mermaid
flowchart TD
    S0["<b>Stage 0</b><br/>Prepare<br/><i>Tick → OHLCV</i>"]
    S1["<b>Stage 1</b><br/>Features<br/><i>21 indicators</i>"]
    S2["<b>Stage 2</b><br/>Labels<br/><i>Triple Barrier</i>"]
    S3["<b>Stage 3</b><br/>Walk-Forward Training<br/><i>GRU + LightGBM per window</i>"]
    S4["<b>Stage 4</b><br/>Backtest<br/><i>CFD Simulation</i>"]
    S5["<b>Stage 5</b><br/>Report<br/><i>Charts + Markdown</i>"]

    S0 --> S1 --> S2 --> S3 --> S4 --> S5

    style S0 fill:#2563EB,color:#fff
    style S1 fill:#2563EB,color:#fff
    style S2 fill:#2563EB,color:#fff
    style S3 fill:#7C3AED,color:#fff
    style S4 fill:#059669,color:#fff
    style S5 fill:#059669,color:#fff
```

| # | Stage | What It Does | Input | Output |
|---|-------|-------------|-------|--------|
| 0 | **Prepare** | Convert raw tick data into 1-hour candle (OHLCV) bars | Raw parquet ticks | `ohlcv.parquet` |
| 1 | **Features** | Calculate 21 technical indicators (RSI, ATR, MACD, etc.) | `ohlcv.parquet` | `features.parquet` |
| 2 | **Labels** | Generate buy/sell/hold labels using the Triple Barrier method | `features.parquet` | `labels.parquet` |
| 3 | **Walk-Forward Training** | For each sliding window: train GRU → extract hidden states → train LightGBM → predict on test slice → collect OOF predictions | `labels.parquet` | `final_predictions.parquet` + model files |
| 4 | **Backtest** | Simulate CFD trading on concatenated OOF predictions | OOF predictions | `backtest_results.json` + `trades_detail.csv` |
| 5 | **Report** | Generate ML metrics, baseline comparison, charts, and application summary | All outputs | Charts + `thesis_report.md` |

> When `validation.method = "static"` in `config.toml`, stage 3 performs a
> traditional train/val/test split and single-pass LightGBM training instead of
> walk-forward. This mode is **not used** by default.

---

## Walk-Forward Validation

The pipeline uses a **sliding window** approach instead of a fixed train/val/test split.
This produces out-of-fold (OOF) predictions across multiple time windows, mimicking
real-world sequential deployment.

```mermaid
flowchart LR
    subgraph W1["Window 1"]
        TR1["Train<br/>sliding"] -->|"purge +<br/>embargo"| TE1["Test"]
    end

    subgraph W2["Window 2"]
        TR2["Train<br/>shifted"] -->|"purge +<br/>embargo"| TE2["Test"]
    end

    subgraph WN["Window N"]
        TRN["Train<br/>shifted"] -->|"purge +<br/>embargo"| TEN["Test"]
    end

    W1 --> W2 --> WN

    TE1 --> OOF["Concatenate<br/>OOF Predictions"]
    TE2 --> OOF
    TEN --> OOF

    OOF --> BT["Backtest"]
    OOF --> RPT["Report"]

    style OOF fill:#D97706,color:#fff
```

Each window:

1. **Slices** the labeled data into a train block and a test block.
2. **Applies purge and embargo** at the boundary (anti-leakage).
3. **Trains GRU** on the train slice (80/20 internal split for early stopping).
4. **Extracts GRU hidden states** for both train and test slices.
5. **Builds the hybrid feature matrix** (GRU hidden states + static indicators).
6. **Trains LightGBM** on the hybrid features.
7. **Predicts on the test slice** and collects as one OOF chunk.

After all windows: OOF chunks are concatenated into a single prediction file
for the backtest and report stages.

Default window parameters (configurable in `config.toml`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train_window_bars` | 26 280 | ~3 years of H1 bars |
| `test_window_bars` | 4 380 | ~6 months of H1 bars |
| `step_bars` | 4 380 | Non-overlapping test windows |
| `purge_bars` | 25 | Bars removed at train/test boundary |
| `embargo_bars` | 50 | Additional gap after purge (~2 days) |
| `min_train_bars` | 10 000 | Minimum training bars per window |

---

## The Hybrid Model (Default Architecture)

This is the core innovation. Here is how it works step by step:

### Step 1: GRU Feature Extractor

The **GRU** (Gated Recurrent Unit) is a neural network that reads sequences of past prices.
Think of it like reading a sentence — it looks at the words one by one and builds an understanding of the whole context.

```mermaid
flowchart LR
    subgraph Input["48-bar sliding window"]
        B1["Bar 1"]
        B2["Bar 2"]
        BD["..."]
        B48["Bar 48"]
    end

    Input --> GRU["GRU<br/>2 layers × 64 hidden"]
    GRU --> HS["64-dim<br/>hidden state"]

    style GRU fill:#7C3AED,color:#fff
    style HS fill:#7C3AED,color:#fff
```

- **Input:** A sliding window of 48 hours using 8 normalized sequence features
  (`log_returns`, `atr_14`, `close_vs_ema_34`, `ema34_vs_ema89`, `candle_body_ratio`, `return_1h`, `return_4h`, `price_position_20`).
- **Output:** A 64-number vector (called "hidden states") that summarizes the temporal pattern.

### Step 2: LightGBM Decision Maker

**LightGBM** is a tree-based model (like a flowchart with many branches).
It takes the GRU's output plus the original 21 technical indicators and makes the final prediction.

```mermaid
flowchart LR
    GRU_OUT["GRU<br/>64 features"] --> COMBINE["Concatenate<br/>85 features"]
    STATIC["Static<br/>21 features"] --> COMBINE
    COMBINE --> LGBM["LightGBM<br/>Classifier"]
    LGBM --> LONG["📈 Long"]
    LGBM --> FLAT["➖ Flat"]
    LGBM --> SHORT["📉 Short"]

    style COMBINE fill:#D97706,color:#fff
    style LGBM fill:#059669,color:#fff
    style LONG fill:#059669,color:#fff
    style FLAT fill:#6B7280,color:#fff
    style SHORT fill:#DC2626,color:#fff
```

- **Input:** 64 GRU hidden states + 21 static features = **85 features total**.
- **Output:** A prediction — **Long** (buy), **Short** (sell), or **Flat** (hold).

### Why Hybrid?

| Approach | Strength | Weakness |
|----------|----------|----------|
| GRU only | Captures time patterns | Misses indicator information |
| LightGBM only | Good with indicators | No sense of time order |
| **Hybrid** | **Captures both time + indicators** | More complex, slower to train |
| **Stacking** | **Learns optimal combination from data** | Needs more folds, longer training |

---

## Stacking Architecture (Alternative — Experimental)

> ⚠️ **Experimental**: Stacking mode is experimental and not the default workflow.
> The primary thesis pipeline uses **hybrid** mode. Do not use stacking for thesis
> defense unless explicitly discussed with your advisor.

When `model.architecture = "stacking"` is set in `config.toml`, the pipeline uses a
**two-level ensemble** instead of the simpler hybrid concatenation:

```mermaid
flowchart TD
    subgraph Base["Level 0 — Base Learners"]
        GRU_B["GRU<br/>sequence features"] --> GRU_P["GRU OOF<br/>probabilities (3)"]
        LGBM_B["LightGBM<br/>static features"] --> LGBM_P["LightGBM OOF<br/>probabilities (3)"]
    end

    GRU_P --> META_FEAT["Meta-Features<br/>6 probability columns"]
    LGBM_P --> META_FEAT

    META_FEAT --> META["Level 1 — Meta-Learner<br/>LightGBM"]
    META --> PRED["Final Prediction<br/>Long / Flat / Short"]

    style META fill:#D97706,color:#fff
    style META_FEAT fill:#D97706,color:#fff
```

### How it works (per walk-forward window)

1. **Train base learners independently:**
   - **GRU base:** Trains on sequence features, produces 3-class probabilities on the test slice.
   - **LightGBM base:** Trains on static features, produces 3-class probabilities on the test slice.
2. **Collect base OOF probabilities** from both learners (`gru_pred_proba_class_*` + `lgbm_pred_proba_class_*`).
3. **Warm-up period:** The meta-learner requires a minimum number of prior folds before it starts training (`stacking.min_meta_train_folds`, default 1).
4. **Train meta-learner** (LightGBM) on the concatenated prior-fold base probabilities.
5. **Generate final predictions** from the meta-learner using the current fold's base probabilities.

After all windows, the pipeline can optionally **final-refit** both base models and the meta-learner
on the full dataset for deployment (`stacking.final_refit = true`).

Stacking-specific artifacts (in addition to the standard session output):

| Artifact | Description |
|----------|-------------|
| `predictions/base_oof_predictions.parquet` | All base-learner OOF probabilities |
| `models/stacking_bundle.joblib` | Deployable bundle with all model paths and config |
| `models/lgbm_base_model.pkl` | Final-refit LightGBM base model |
| `models/training_history.json` | Training details for base and meta models |

---

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| **Walk-forward validation (default)** | Prevents look-ahead bias; mimics real sequential deployment |
| **Flat module layout** | Each pipeline stage is one file — easy to navigate, no nested packages |
| **GRU instead of LSTM** | Fewer parameters (25-30% less), less overfitting on small data |
| **No bidirectional GRU** | Prevents look-ahead bias (seeing future data) |
| **Small attention pooling** | Summarizes the 48-bar GRU output into one fixed-size embedding |
| **LightGBM as the decision maker** | Better interpretability, handles mixed feature types |
| **Stacking as opt-in alternative (experimental)** | Learns optimal base-model weighting from data; needs more folds; not recommended for thesis defense |
| **Polars instead of Pandas** | 10-50x faster for time-series operations |
| **Session-based output folders** | Each run is isolated — easy to compare experiments |
| **Correlation filtering on train only** | Prevents data leakage from test set |
| **Purge and embargo at each window boundary** | Prevents label leakage between train and test slices |
| **Triple Barrier labeling** | Realistic profit targets with a time limit |
| **Backtest as demo only** | Keeps the thesis focused on ML quality instead of trading optimization |
| **Fixed lot position sizing** | Keeps the application demo deterministic and easy to explain |

---

## Project Structure

```text
thesis/
├── config.toml              # All settings in one file
├── main.py                  # Entry point (CLI)
├── pixi.toml                # Package manager config
│
├── src/thesis/              # Source code (flat modules)
│   ├── config.py            # TOML config loader + dataclasses
│   ├── constants.py         # Shared constants and column lists
│   ├── session_paths.py     # Session directory path setup
│   ├── pipeline.py          # Stage orchestration (walk-forward + static)
│   ├── data.py              # Tick → OHLCV aggregation (Stage 0)
│   ├── features.py          # 21 technical indicators (Stage 1)
│   ├── labels.py            # Triple-barrier labeling (Stage 2)
│   ├── validation.py        # Walk-forward window generation + static split
│   ├── gru.py               # GRU feature extractor (train, predict, save)
│   ├── model.py             # LightGBM training (fixed params + Optuna)
│   ├── backtest.py          # CFD trading simulation (Stage 5)
│   ├── report.py            # Report + chart generation (Stage 5)
│   ├── charts.py            # Interactive ECharts (Streamlit)
│   ├── dashboard.py         # Streamlit dashboard
│   ├── zones.py             # Metric zone classification
│   └── ui.py                # Rich console utilities
│
├── scripts/
│   └── data_download.py     # Market data ingestion
│
├── tests/                   # Test suite
│   ├── conftest.py
│   ├── unit/                # Unit tests per module
│   └── integration/         # End-to-end tests
│
├── data/
│   ├── raw/XAUUSD/          # Raw tick data (monthly files)
│   └── processed/           # Generated parquet files
│
├── results/                 # Session-based outputs
│   └── {SYMBOL}_{TF}_{TIMESTAMP}/
│       ├── config/          # Config snapshot
│       ├── models/          # Saved models (LightGBM + GRU)
│       ├── predictions/     # Predictions (parquet + CSV)
│       ├── reports/         # Report + charts + walk_forward_history.json
│       ├── backtest/        # Trading results + trade details CSV
│       └── logs/            # Pipeline log (ANSI-stripped)
│
└── docs/                    # Documentation (you are here)
```

### Core vs Optional Modules

**Core modules** — required to run the main pipeline (`pixi run workflow`):

| Module | Role |
|--------|------|
| `data.py` | Stage 0: Tick → OHLCV |
| `features.py` | Stage 1: 21 technical indicators |
| `labels.py` | Stage 2: Triple-barrier labeling |
| `validation.py` | Walk-forward window generation |
| `gru.py` | GRU feature extractor |
| `model.py` | LightGBM training |
| `backtest.py` | Stage 5: CFD simulation |
| `report.py` | Stage 5: Report + charts |
| `pipeline.py` | Stage orchestration |
| `config.py` | TOML config → dataclasses |

**Optional modules** — not required for the batch pipeline:

| Module | Role |
|--------|------|
| `charts.py` | Interactive ECharts visualizations (Streamlit) |
| `dashboard.py` | Streamlit dashboard UI |
| `zones.py` | Metric zone classification for dashboard |
| `ui.py` | Rich console formatting utilities |

---

## Data Flow

Here is what happens to the data at each step:

```mermaid
flowchart TD
    T0["Raw Ticks<br/><i>millions of rows</i>"] -->|"prepare_data()"| T1["OHLCV<br/><i>~55,000 rows</i>"]
    T1 -->|"generate_features()"| T2["Features<br/><i>+ 21 technical indicators</i>"]
    T2 -->|"generate_labels()"| T3["Labels<br/><i>+ buy/sell/hold + TP/SL prices</i>"]
    T3 -->|"walk-forward<br/>sliding windows"| T4["Per Window:<br/>Train slice → GRU → hidden states<br/>→ LightGBM → OOF predictions"]
    T4 -->|"concatenate<br/>OOF chunks"| T5["Final OOF Predictions<br/><i>timestamp + true_label + pred_label + probabilities</i>"]
    T5 -->|"run_backtest()"| T6["Backtest<br/><i>trades, PnL, metrics</i>"]
    T5 -->|"generate_report()"| T7["Report<br/><i>markdown + charts</i>"]

    style T0 fill:#6B7280,color:#fff
    style T4 fill:#7C3AED,color:#fff
    style T7 fill:#059669,color:#fff
```

---

## Anti-Leakage Protection

Data leakage is when information from the future accidentally "leaks" into the training data.
This project uses **three layers** of protection, applied **dynamically at each walk-forward window boundary**:

```mermaid
flowchart LR
    TR["Train Window<br/>bars [a..b]"] -->|"purge<br/>25 bars"| P1[" "]
    P1 -->|"embargo<br/>50 bars"| TE["Test Window<br/>bars [c..d]"]
    TE -->|"next window<br/>shifts forward"| TR2["Train Window<br/>bars [a+s..b+s]"]

    style P1 fill:#DC2626,color:#fff
    style TR fill:#2563EB,color:#fff
    style TE fill:#059669,color:#fff
    style TR2 fill:#2563EB,color:#fff
```

1. **Purge** — Removes 25 bars at each train/test boundary to prevent overlap
   from the label look-ahead window.
2. **Embargo** — Adds 50 extra bars of gap after each boundary (~2 days,
   covers the 48-bar label horizon).
3. **Correlation filtering on train only** — Feature selection uses only training data.

These gaps apply at **every window boundary**, not just at fixed dates.
The window indices are computed dynamically by `validation.generate_windows()`
based on the total bar count and the configured window sizes.

---

## Session-Based Output

Every time you run the pipeline, a new **session folder** is created:

```mermaid
flowchart TD
    RUN["pixi run workflow"] --> SESSION["results/XAUUSD_1H_20260414_042000/"]

    SESSION --> CFG["config/<br/>config_snapshot.toml"]
    SESSION --> MOD["models/<br/>lightgbm_model.pkl<br/>gru_model.pt"]
    SESSION --> PRED["predictions/<br/>final_predictions.parquet<br/>final_predictions.csv"]
    SESSION --> REP["reports/<br/>thesis_report.md<br/>walk_forward_history.json<br/>charts/"]
    SESSION --> BT["backtest/<br/>backtest_results.json<br/>trades_detail.csv<br/>equity_curve.csv<br/>backtest_chart.html"]
    SESSION --> LOG["logs/<br/>pipeline.log"]

    style SESSION fill:#2563EB,color:#fff
```

**Additional stacking artifacts** (when `model.architecture = "stacking"`):

```mermaid
flowchart TD
    SESSION["Session folder"] --> STACK_PRED["predictions/<br/>base_oof_predictions.parquet"]
    SESSION --> STACK_MOD["models/<br/>stacking_bundle.joblib<br/>lgbm_base_model.pkl<br/>training_history.json"]

    style SESSION fill:#7C3AED,color:#fff
```

This means:
- Old results are never overwritten.
- You can compare different parameter settings.
- Each session has its own log (ANSI-stripped for clean file output), config snapshot, and all outputs.
- `walk_forward_history.json` records the exact window indices and OOF row counts for reproducibility.
