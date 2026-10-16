"""Publish scoped calculated knowledge and current solver progress."""
import os
from pathlib import Path
from training.publication import SupabaseStorage
from training.solver_service import publish_status

if __name__ == '__main__':
    publish_status(SupabaseStorage.from_environment(), Path(os.environ['GTO_DATA_DIR']) / 'solver-v1', os.environ['GTO_TRACK'])
