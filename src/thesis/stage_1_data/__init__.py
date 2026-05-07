"""Data generation package for aggregating raw ticks to OHLCV bars."""

from .processing import generate_data, prepare_data

__all__ = ["generate_data", "prepare_data"]
