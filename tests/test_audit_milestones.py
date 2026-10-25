"""Independent archive survives retention and rejects missing/corrupt milestones."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from deploy.vps.archive_milestones import archive_once


class MilestoneTests(unittest.TestCase):
    def test_archive_survives_source_rotation_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / 'source', root / 'archive'
            (source / 'checkpoints').mkdir(parents=True)
            target.mkdir()
            (source / 'selfplay.json').write_text(json.dumps({'run_id': 'test', 'track': 'hu'}))
            (source / 'resume.json').write_text(json.dumps({'iteration': 1650}))
            checkpoint = source / 'checkpoints/iteration_001650.pt'
            with zipfile.ZipFile(checkpoint, 'w') as archive:
                archive.writestr('payload.pt', b'payload')
                archive.writestr('checkpoint.json', json.dumps({
                    'format': 'streamed-checkpoint-v1', 'sha256': hashlib.sha256(b'payload').hexdigest()}))
            result = archive_once(source, target, (1650, 1750), 'test')
            self.assertEqual(result['milestones'], {'1650': 'saved', '1750': 'pending'})
            checkpoint.unlink()
            self.assertEqual(archive_once(source, target, (1650, 1750), 'test')['milestones'], result['milestones'])
            (source / 'resume.json').write_text(json.dumps({'iteration': 1760}))
            with self.assertRaisesRegex(FileNotFoundError, '1750 missed'):
                archive_once(source, target, (1650, 1750), 'test')
            with self.assertRaisesRegex(ValueError, 'identity'):
                archive_once(source, target, (1650,), 'different')

    def test_corrupt_payload_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'checkpoints').mkdir()
            target = root / 'archive'
            target.mkdir()
            (root / 'selfplay.json').write_text(json.dumps({'run_id': 'test', 'track': 'hu'}))
            (root / 'resume.json').write_text(json.dumps({'iteration': 1650}))
            with zipfile.ZipFile(root / 'checkpoints/iteration_001650.pt', 'w') as archive:
                archive.writestr('payload.pt', b'bad')
                archive.writestr('checkpoint.json', json.dumps({'format': 'streamed-checkpoint-v1', 'sha256': 'wrong'}))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                archive_once(root, target, (1650,), 'test')
            self.assertFalse((target / 'iteration_001650.json').exists())
