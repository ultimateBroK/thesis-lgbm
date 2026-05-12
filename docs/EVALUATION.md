# Evaluation

The primary thesis evidence is classification performance, not trading profit.

---

## Primary Metrics

Use these first:

| Metric | What It Measures | Range |
|---|---|---|
| **Accuracy** | Overall correct prediction rate | [0, 1] |
| **Balanced Accuracy** | Average recall across classes (robust to imbalance) | [0, 1] |
| **Directional Accuracy** | Accuracy on Short/Long bars only (Hold excluded) | [0, 1] |
| **MDA (no hold)** | Market Directional Accuracy — Short vs Long only | [0, 1] |
| **MDA (binary)** | MDA for Long vs Short, Hold predictions on directional bars count as wrong | [0, 1] |
| **Macro F1** | Average F1 across classes (penalizes ignoring minority) | [0, 1] |
| **Weighted F1** | Support-weighted F1 across classes | [0, 1] |
| **Per-class F1** | Individual F1 for Short, Hold, Long | [0, 1] |
| **Confusion Matrix** | 3x3 true vs predicted label counts | counts |
| **Direction Confusion Matrix** | 2x2 Short vs Long only (Hold excluded) | counts |
| **High-Confidence Accuracy** | Accuracy when max probability > threshold | [0, 1] |
| **Majority Baseline** | Accuracy if always predicting most common class | [0, 1] |

### Calibration Metrics (Secondary)

| Metric | What It Measures | Range |
|---|---|---|
| **ECE** | Expected Calibration Error — reliability of probability estimates | [0, 1] |
| **Brier Score** | Mean squared probability error | [0, 2] |
| **Log Loss** | Cross-entropy of predicted probabilities | [0, inf) |
| **Confidence Bins Accuracy** | Accuracy per probability bin | [0, 1] |

### Auxiliary Regression Metrics (If Available)

| Metric | What It Measures |
|---|---|
| **MAE** | Mean Absolute Error on continuous returns |
| **RMSE** | Root Mean Squared Error on continuous returns |
| **R²** | Coefficient of determination on continuous returns |

Backtest metrics are application-demo metrics only.

---

## Metric Computation

All classification metrics are computed in `src/thesis/stage_6_reporting/model_metrics.py` via `compute_all_classification_metrics()`. This function:

1. Computes confusion matrix (3x3)
2. Computes per-class precision/recall/F1
3. Computes macro F1, weighted F1, accuracy
4. Computes balanced accuracy, directional accuracy, MDA variants
5. Optionally computes calibration metrics (if probabilities available)
6. Optionally computes auxiliary regression metrics (if returns available)

Baseline metrics are computed using four strategies from `src/thesis/stage_4_training/baselines.py`:
- **Naive Direction**: predict previous bar's return direction
- **Majority Class**: always predict most common class
- **Random**: random class assignment
- **Always Predict**: predict a fixed class

---

## Latest Verified Run

Latest verified session:

```text
results/XAUUSD_1H_20260513_023811/
Pipeline runtime: 75.65 seconds
```

### Classification Results

```text
Accuracy              0.3416
Balanced accuracy     0.3675
Directional accuracy  0.4929
Macro F1              0.3152
Weighted F1           0.3674
Majority baseline     0.4850
```

Per-class F1:

```text
Short  0.3640
Hold   0.1780
Long   0.4037
```

High-confidence (>0.7):

```text
Samples    182 (0.76% of total)
Accuracy   0.2418
Directional 0.5588
```

### Model Comparison

```text
Hybrid Stacking     accuracy 0.3416, macro F1 0.3152
Logistic Regression accuracy 0.3568, macro F1 0.3173
Random Forest       accuracy 0.3596, macro F1 0.3280
LightGBM            accuracy 0.3738, macro F1 0.3265
Naive Direction     accuracy 0.4574, macro F1 0.3178
Majority Baseline   accuracy 0.4850, macro F1 0.2177
Random Baseline     accuracy 0.3361, macro F1 0.3056
```

Interpretation: Hybrid Stacking did not outperform LightGBM in this run. This is acceptable thesis evidence if reported honestly: more complex stacking is not guaranteed to beat a strong tabular booster on noisy financial data. All models remain below the majority baseline (0.4850), confirming the difficulty of the prediction task.

### Why Stacking May Underperform

Possible reasons Hybrid Stacking underperforms LightGBM:

1. **Noisy base learner signals**: when base learners disagree frequently, the meta-learner receives conflicting probability inputs
2. **Small meta-training window**: 20% of an already short train window may not be enough for the meta-learner to learn meaningful combinations
3. **Regularization mismatch**: meta Logistic Regression may be too simple or too regularized to capture non-linear probability interactions
4. **Financial time-series noise**: the signal-to-noise ratio in price data may not support additional model complexity

---

## Label Distribution

Current config (`horizon_bars = 24`, TP/SL `2.0/2.0`):

```text
Short 43.6%
Hold   9.0%
Long  47.4%
```

A tested alternative, `horizon_bars = 48`, reduced Hold to about 1.5%, so it was rejected and the config was rolled back to 24.

---

## Backtest Interpretation

Latest application-demo backtest:

```text
Period:        2022-01-27 to 2026-04-29
Total return   1.92%
Max drawdown  -2.72%
Profit factor  1.109
Sharpe ratio   0.384
Sortino ratio  0.637
Calmar ratio   0.138
Win rate      47.17%
Trades        159
```

### Backtest Metric Quality Zones

Backtest metrics are evaluated against quality benchmarks defined in `src/thesis/shared/zones.py`:

| Metric | Poor | Fair | Good |
|---|---|---|---|
| Total Return | < -5% | -5% to 10% | > 10% |
| Max Drawdown | > 20% | 10-20% | < 10% |
| Profit Factor | < 1.0 | 1.0-1.5 | > 1.5 |
| Sharpe Ratio | < 0.0 | 0.0-1.0 | > 1.0 |
| Win Rate | < 40% | 40-55% | > 55% |

Current backtest assessment: Fair zone (positive return, PF > 1.0, Sharpe in fair range).

This does not prove a deployable trading strategy. It only shows how predictions can be consumed by a simple CFD signal simulator.

---

## Reading the Report

The thesis report (`reports/thesis_report.md`) contains these sections in order:

1. **Executive Summary** — key metrics at a glance
2. **Configuration** — config snapshot for reproducibility
3. **Data Quality** — gap analysis, candle consistency, outlier report
4. **Label Design & Methodology** — triple-barrier explanation, distribution stats
5. **Validation Methodology** — walk-forward window layout, purge/embargo
6. **Classification Metrics** — primary metrics table with confusion matrix
7. **Calibration** — ECE, Brier, confidence bins
8. **Auxiliary Regression** — MAE/RMSE/R² on returns
9. **Model Comparison** — all models ranked by macro F1
10. **Backtest Demo** — trading metrics with quality zone indicators
11. **Feature Importance** — top features from LightGBM
12. **OOF vs OOS** — generalization check between out-of-fold and out-of-sample
13. **Issues & Recommendations** — primary issue identification
14. **Metric Zones** — backtest metrics with color-coded quality indicators
15. **Verdict** — synthesized ML quality + trading edge assessment

---

## Deployment Recommendation

The report generates an automatic deployment recommendation based on:

1. **ML Quality**: assessed as POOR / FAIR / GOOD based on macro F1 and directional accuracy
2. **Trading Edge**: assessed as NEGATIVE / MARGINAL / POSITIVE based on backtest profit factor and Sharpe ratio
3. **Primary Issue**: the single most critical issue identified from metrics
4. **Recommendation**: derived from quality + edge assessment

This recommendation is informational only and should be interpreted in the context of the thesis narrative.
