"""Validate or initialize the explicit continuous self-play run before activation."""
import os
from pathlib import Path
from training.selfplay import RUN_DIRECTORY, migrate_v4

if __name__ == '__main__':
    root = Path(os.environ['GTO_DATA_DIR']).resolve() / RUN_DIRECTORY
    contract = migrate_v4(root, os.environ['GTO_TRACK'], source_directory=root.parent / 'selfplay-v3')
    print(f"Self-play contract compatible: {contract['engine']} {contract['track']}", flush=True)
