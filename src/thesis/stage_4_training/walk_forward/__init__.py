"""Walk-forward training sub-package.

Re-exports the public dispatcher and architecture-specific entry points
so external callers can import from ``thesis.stage_4_training.walk_forward``.
"""

from thesis.stage_4_training.walk_forward.dispatcher import train_walk_forward
from thesis.stage_4_training.walk_forward.gru import train_gru_walk_forward
from thesis.stage_4_training.walk_forward.hybrid import (
    _compute_regression_target,
    train_hybrid_walk_forward,
)
from thesis.stage_4_training.walk_forward.lgbm import (
    train_lgbm_fixed,
    train_lgbm_walk_forward,
)

__all__ = [
    "_compute_regression_target",
    "train_gru_walk_forward",
    "train_hybrid_walk_forward",
    "train_lgbm_fixed",
    "train_lgbm_walk_forward",
    "train_walk_forward",
]
