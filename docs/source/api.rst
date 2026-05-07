API Reference
=============

The public documentation currently focuses on workflow guides. The codebase uses
stage-based subpackages under ``src/thesis/``:

- ``thesis.shared`` — configuration, constants (including ``timeframe_to_ms()``), session paths, UI helpers, zones
- ``thesis.stage_1_data`` — OHLCV preparation
- ``thesis.stage_2_features`` — feature engineering (core indicators + trend indicators)
- ``thesis.stage_3_labels`` — triple-barrier labels
- ``thesis.stage_4_training`` — validation, GRU sub-package, LightGBM, walk-forward sub-package, baselines
- ``thesis.stage_4_training.gru`` — GRU architecture, training, losses, calibration, inference, persistence
- ``thesis.stage_4_training.walk_forward`` — walk-forward dispatcher, hybrid/GRU/LGBM training, artifacts, utils
- ``thesis.stage_5_backtest`` — application-demo backtest (strategy, persistence, runners)
- ``thesis.stage_6_reporting`` — report generation with sections sub-package
- ``thesis.stage_6_reporting.sections`` — report section renderers (data, oof, assess, backtest)
- ``thesis.charts`` — interactive ECharts visualizations (Streamlit)
- ``thesis.dashboard`` — Streamlit dashboard (10 modules, entry via ``dashboard/main.py``)

Detailed API pages are intentionally not included until generated Sphinx API
stubs exist. This avoids broken references during documentation builds.
