"""Leakage guard tests: verify no look-ahead data leaks into features or training."""

import ast
import inspect
import textwrap

import numpy as np
import polars as pl
import pytest

from thesis.shared.constants import EXCLUDE_COLS


FORBIDDEN_COLS = {
    "label",
    "upper_barrier",
    "lower_barrier",
    "touched_bar",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


@pytest.mark.unit
def test_forbidden_cols_not_in_features() -> None:
    """EXCLUDE_COLS must contain all raw OHLCV + label-derived columns."""
    for col in FORBIDDEN_COLS:
        assert col in EXCLUDE_COLS, (
            f"Forbidden column {col!r} missing from EXCLUDE_COLS"
        )


@pytest.mark.unit
def test_no_negative_shift_in_features_source() -> None:
    """Feature code must not use shift(-n) (future-looking shift)."""
    import thesis.stage_2_features.engineering as feat_mod

    source = inspect.getsource(feat_mod)
    tree = ast.parse(textwrap.dedent(source))

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "shift" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                    violations.append(f"Line ~{node.lineno}: shift with negative value")

    assert len(violations) == 0, f"Found shift(-n) calls (future-looking): {violations}"


@pytest.mark.unit
def test_no_center_true_in_rolling_features() -> None:
    """Feature code must not use center=True in rolling/ewm operations (uses future)."""
    import thesis.stage_2_features.engineering as feat_mod

    source = inspect.getsource(feat_mod)
    tree = ast.parse(textwrap.dedent(source))

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "center":
                    if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        violations.append(f"Line ~{node.lineno}: center=True detected")

    assert len(violations) == 0, (
        f"Found center=True in rolling ops (future leak): {violations}"
    )


@pytest.mark.unit
def test_oof_predictions_unique_timestamps() -> None:
    """OOF walk-forward predictions must have unique timestamps (no duplicate leakage)."""
    rng = np.random.default_rng(42)
    n = 100
    timestamps = pl.datetime_range(
        start=pl.datetime(2023, 1, 1, 0, time_zone="UTC"),
        end=pl.datetime(2023, 1, 1, 0, time_zone="UTC") + pl.duration(hours=n - 1),
        interval="1h",
        eager=True,
    )

    oof = pl.DataFrame(
        {
            "timestamp": timestamps,
            "pred_long": rng.random(n),
            "pred_short": rng.random(n),
        }
    )

    assert oof["timestamp"].is_unique().all(), (
        "OOF predictions must have unique timestamps"
    )
