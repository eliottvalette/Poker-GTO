"""Publish continuous self-play progress and validated policy exports."""
import os
from pathlib import Path
from training.publication import SupabaseStorage, publish_pending
from training.selfplay import RUN_DIRECTORY, publish_status

if __name__ == '__main__':
    storage = SupabaseStorage.from_environment()
    directory = Path(os.environ['GTO_DATA_DIR']) / RUN_DIRECTORY
    track = os.environ['GTO_TRACK']
    publish_status(storage, directory, track)
    publish_pending(storage, track, directory, keep=2, grace_hours=2, require_opening_health=False)
