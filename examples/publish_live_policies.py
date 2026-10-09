"""Seed Storage from existing validated UI exports; read server credentials from the environment."""
from pathlib import Path
import json
from training.publication import SupabaseStorage, publish_bundle


def publish_existing_exports(root: Path) -> list[dict]:
    storage = SupabaseStorage.from_environment()
    storage.provision()
    catalog = json.loads((root/'index.json').read_text())
    results = []
    for track in ('hu', '3max'):
        entry = catalog['exports'][track]
        result = publish_bundle(storage, track, root/'releases'/entry['bundle_id'])
        results.append(result)
        print(f'{track}: iteration {result["iteration"]}, release {result["release_id"]}', flush=True)
    return results


if __name__ == '__main__':
    publish_existing_exports(Path(__file__).resolve().parents[1]/'ui/public/policy')
