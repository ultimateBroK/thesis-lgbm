"""Shared Rich UI primitives for the thesis pipeline.

Provides a single Console instance, styled Progress factories,
and helper functions for consistent terminal output across all stages.
"""

import logging

from rich.console import Console
from rich.text import Text

# Singleton console — every module imports this for consistent rendering
console = Console()

# Stage colour map (used by pipeline + training)
STAGE_STYLES: dict[int, str] = {
    1: "bold blue",
    2: "bold green",
    3: "bold yellow",
    4: "bold cyan",
    5: "bold magenta",
    6: "bold red",
}

STAGE_LABELS: dict[int, str] = {
    1: "Data Preparation",
    2: "Feature Engineering",
    3: "Label Generation",
    4: "Model Training",
    5: "Application Demo / Backtest",
    6: "Report Generation",
}


# UI helpers
def stage_header(stage: int) -> None:
    """Print a visually distinct stage banner via console (Rich) and logger.

    Args:
        stage: Stage number (1-indexed, 1–6).
    """
    _logger = logging.getLogger("thesis")
    style = STAGE_STYLES.get(stage, "bold")
    label = STAGE_LABELS.get(stage, f"Stage {stage}")
    total = 6
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
    """Print a dim skip line via console (Rich) and logger.

    Args:
        stage: Stage number (1-indexed, 1–6).
        reason: Why the stage is being skipped.
    """
    _logger = logging.getLogger("thesis")
    label = STAGE_LABELS.get(stage, f"Stage {stage}")
    console.print(Text(f"  ⊘ SKIP {label}: {reason}", style="dim"))
    # Logger output for file capture
    _logger.info("SKIP %s | %s", label, reason)
