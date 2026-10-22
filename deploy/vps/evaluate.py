"""Evaluate exported self-play policies independently against a fixed reference."""
from pathlib import Path
import torch
from training.publication import SupabaseStorage
from training.evaluation_history import evaluate_published

if __name__ == '__main__':
    torch.set_num_threads(1)
    storage = SupabaseStorage.from_environment()
    for track in ('hu', '3max'):
        directory = Path('/var/lib/gto/evaluation/selfplay-v2') / track
        evaluate_published(storage, track, directory)
