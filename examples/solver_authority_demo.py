"""Importable calculation-first demonstration; no deployment or neural activation."""
from pathlib import Path

from hybrid.solver_experiments import run


def demonstrate(output: Path) -> dict:
    return run(output, iterations=32, updates=64)
