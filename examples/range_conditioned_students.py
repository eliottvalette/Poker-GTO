"""Bounded local solver/student evaluation; never deploys or publishes models."""
from pathlib import Path

from hybrid.range_student_experiments import cpu_comparison, label_stability, range_contrast, run


def demonstrate(output: Path) -> dict:
    """Generate 224 small games, fit optional students and retain comparisons.

    The output must not exist. Measured locally at about four minutes under
    cProfile, one Torch thread, under 400 MiB process RSS for the main study.
    Published policies and production training are not modified.
    """
    study = run(output, train_roots=4, heldout_roots=2, iterations=64, updates=128)
    contrast = range_contrast(output / "contrast", updates=256)
    cpu = cpu_comparison(output)
    return {"study": study, "contrast": contrast, "cpu": cpu, "label_stability": label_stability(output)}
