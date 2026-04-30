"""Shared Rich UI primitives for the thesis pipeline.

Provides a single Console instance, styled Progress factories,
and helper functions for consistent terminal output across all stages.
"""

import logging

from rich.console import Console
from rich.text import Text

# ---------------------------------------------------------------------------
# Singleton console — every module imports this for consistent rendering
# ---------------------------------------------------------------------------
console = Console()

# ---------------------------------------------------------------------------
# Stage colour map (used by pipeline + training)
# ---------------------------------------------------------------------------
STAGE_STYLES: dict[int, str] = {
    0: "bold blue",
    1: "bold green",
    2: "bold yellow",
    3: "bold cyan",
    4: "bold magenta",
    5: "bold red",
}

STAGE_LABELS: dict[int, str] = {
    0: "Data Preparation",
    1: "Feature Engineering",
    2: "Triple-Barrier Labeling",
    3: "Walk-Forward Training",
    4: "Backtest",
    5: "Report Generation",
}


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def stage_header(stage: int, total: int = 6) -> None:
    """Print a visually distinct stage banner via console (Rich) and logger."""
    _logger = logging.getLogger("thesis")
    style = STAGE_STYLES.get(stage, "bold")
    label = STAGE_LABELS.get(stage, f"Stage {stage}")
    # Rich console output for visual display
    console.print()
    console.rule(
        Text(f"  STAGE {stage}/{total}  ·  {label}  ", style=style),
        style=style,
        characters="─",
    )
    console.print()
    # Logger output for file capture
    _logger.info("")
    _logger.info("STAGE %d/%d | %s", stage, total, label)
    _logger.info("")


def stage_skip(stage: int, reason: str) -> None:
    """Print a dim skip line via console (Rich) and logger."""
    _logger = logging.getLogger("thesis")
    label = STAGE_LABELS.get(stage, f"Stage {stage}")
    # Rich console output
    console.print(Text(f"  ⊘ SKIP {label}: {reason}", style="dim"))
    # Logger output for file capture
    _logger.info("SKIP %s | %s", label, reason)
