"""Validate the explicit solver run before service activation; preserve legacy data."""
import os
from pathlib import Path
from training.solver_service import initialize

if __name__ == '__main__':
    root = Path(os.environ['GTO_DATA_DIR']).resolve() / 'solver-v1'
    contract = initialize(root, os.environ['GTO_TRACK'])
    print(f"Solver contract compatible: {contract['engine']} {contract['track']}", flush=True)
