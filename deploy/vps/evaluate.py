"""Hourly solver reference diagnostics, separate from legacy NN win-rate history."""
from pathlib import Path
import torch
from training.publication import SupabaseStorage
from training.solver_service import publish

if __name__ == '__main__':
    torch.set_num_threads(1)
    storage = SupabaseStorage.from_environment()
    for track in ('hu', '3max'):
        directory = Path('/var/lib/gto') / track / 'solver-v1'
        if not all((directory / 'references' / f'{street}.json').exists()
                   for street in ('PREFLOP', 'FLOP', 'TURN', 'RIVER')):
            print(f'{track}: waiting for worker reference tables', flush=True)
            continue
        publish(storage, directory, track)
