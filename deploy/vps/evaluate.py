"""Sequential, bounded evaluation of both published tracks; no checkpoint loads."""
import os
from pathlib import Path

from training.evaluation_history import evaluate_published
from training.publication import SupabaseStorage


if __name__ == '__main__':
    storage = SupabaseStorage.from_environment()
    failures = []
    for track in ('hu', '3max'):
        try:
            report = evaluate_published(storage, track, Path(os.environ['GTO_EVALUATION_DIR']) / track)
            if report and report['status'] == 'failed':
                failures.append(f'{track}: {report["warnings"][-1]}')
        except Exception as error:
            failures.append(f'{track}: {type(error).__name__}: {error}')
    if failures:
        raise RuntimeError('; '.join(failures))
