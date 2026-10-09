"""Archive source identity must survive Git-free deployment and detect modifications."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from training.runner import source_metadata, source_commit, write_source_manifest


class DeploymentProvenanceTests(unittest.TestCase):
    def test_archive_preserves_identity_and_detects_modified_source(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        expected = source_metadata(repository)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root/'deployment-source.json'
            write_source_manifest(repository, manifest)
            files = json.loads(manifest.read_text())['files']
            for name in files:
                target = root/name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(repository/name, target)
            self.assertEqual(source_metadata(root), expected)
            self.assertEqual(source_commit(root), expected['source_git_commit'])
            (root/files[0]).write_bytes(b'modified source')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                source_metadata(root)

    def test_missing_manifest_is_explicit_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FileNotFoundError, 'deployment provenance'):
                source_metadata(Path(temporary))
