"""Read-only deployment gate: reject incompatible checkpoints before service activation."""
import json
import os
from pathlib import Path
import torch
from training.runner import TrainingRunner

if __name__ == '__main__':
    torch.set_num_threads(1)
    root = Path(os.environ['GTO_DATA_DIR']).resolve()
    pointer = root/'resume.json'
    selected = json.loads(pointer.read_text()) if pointer.exists() else {'path': 'checkpoint.pt'}
    checkpoint = (root/selected['path']).resolve()
    checkpoint.relative_to(root)
    runner = TrainingRunner.load_checkpoint(checkpoint)
    if set(runner.solvers) != {os.environ['GTO_TRACK']}:
        raise ValueError('Checkpoint track does not match service')
    if runner.runtime_storage is None:
        raise ValueError('Convert the legacy checkpoint locally before VPS deployment')
    if 'iteration' in selected and selected['iteration'] != runner.iteration:
        raise ValueError('Resume pointer iteration mismatch')
    print(f'Checkpoint compatible: iteration {runner.iteration}', flush=True)
