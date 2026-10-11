"""Explicit bounded training example; importing this module performs no work."""
from pathlib import Path

from training.runner import TrainingRunner
from training.street_experiments import diagnostic_config
from training.vps import export_live


def run_demo(output: Path, player_count: int, *, seed: int = 101) -> dict:
    """Five local cycles, 64 traversals/player and 64 updates/eligible specialist.

    Exports remain inside output. This neither activates nor uploads a policy.
    Existing output is rejected to preserve checkpoint provenance.
    """
    if player_count not in (2, 3):
        raise ValueError("player_count must be 2 or 3")
    if output.exists():
        raise FileExistsError(f"Explicit new experiment directory required: {output}")
    runner = TrainingRunner(diagnostic_config(output, player_count, seed, updates=64, traversals=64))
    runner.enable_bounded_storage()
    for _ in range(5):
        runner.run_iteration()
    checkpoint = output/'checkpoint.pt'
    runner.save_checkpoint(checkpoint)
    resumed = TrainingRunner.load_checkpoint(checkpoint)
    if resumed.iteration != runner.iteration:
        raise RuntimeError("Checkpoint iteration mismatch")
    exported = export_live(resumed)
    return {'checkpoint': str(checkpoint), 'local_export': str(exported),
            'iteration': resumed.iteration, 'status': 'experimental_not_strategically_validated'}
