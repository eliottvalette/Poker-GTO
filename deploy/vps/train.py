"""Service entrypoint; environment selects the explicit dedicated track and data root."""
import os
from pathlib import Path
from training.vps import run_continuous

if __name__ == '__main__':
    track = os.environ['GTO_TRACK']
    directory = Path(os.environ['GTO_DATA_DIR'])
    source = os.environ.get('GTO_SOURCE_CHECKPOINT')
    run_continuous(track, directory, source_checkpoint=Path(source) if source else None,
                   workers=int(os.environ.get('GTO_WORKERS', '1')))
