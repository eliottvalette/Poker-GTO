"""One bounded publication attempt; systemd retries on its next timer tick."""
import os
from pathlib import Path
from training.publication import SupabaseStorage, publish_pending

if __name__ == '__main__':
    storage = SupabaseStorage.from_environment()
    publish_pending(storage, os.environ['GTO_TRACK'], Path(os.environ['GTO_DATA_DIR']))
