"""Validate or initialize the explicit continuous self-play run before activation."""
import os
from pathlib import Path
from training.selfplay import initialize

if __name__ == '__main__':
    root = Path(os.environ['GTO_DATA_DIR']).resolve() / 'selfplay-v2'
    contract = initialize(root, os.environ['GTO_TRACK'])
    print(f"Self-play contract compatible: {contract['engine']} {contract['track']}", flush=True)
