"""Publication ordering, retry, track isolation and bounded retention contracts."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from io import BytesIO
from training.publication import (SupabaseStorage, StorageError, publish_bundle, prune_remote, encoded)


class MemoryStorage:
    def __init__(self):
        self.objects = {}
        self.fail = None
        self.events = []

    def get(self, key, *, public=False):
        self.events.append(('public' if public else 'get', key))
        if key not in self.objects:
            raise StorageError(404, key)
        return self.objects[key]

    def put(self, key, data, *, mutable=False):
        if self.fail and self.fail in key:
            raise StorageError(503, key)
        self.events.append(('put', key))
        self.objects[key] = data

    def ensure(self, key, data):
        if key in self.objects and self.objects[key] != data:
            raise ValueError('Immutable mismatch')
        if key not in self.objects:
            self.put(key, data)

    def list(self, prefix):
        names = {k[len(prefix)+1:].split('/')[0] for k in self.objects if k.startswith(prefix+'/')}
        return [{'name': name, 'created_at': (datetime.now(timezone.utc)-timedelta(days=2)).isoformat()} for name in sorted(names)]

    def remove(self, keys):
        for key in keys:
            del self.objects[key]


def bundle(root, track, iteration):
    directory = root/f'{track}-{iteration}';directory.mkdir()
    model = f'{track}-{iteration}-test-bytes'.encode()
    manifest = {'supported_player_counts': [2 if track == 'hu' else 3], 'iteration': iteration,
                'model_sha256': hashlib.sha256(model).hexdigest()}
    files = {f'average_{track}.onnx': model, f'average_{track}.json': encoded(manifest)}
    for name, data in files.items(): (directory/name).write_bytes(data)
    (directory/'bundle.json').write_bytes(encoded({'version': 1, 'kind': 'live_training_export',
        'tracks': [track], 'iteration': iteration,
        'files': {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}))
    return directory


class PublicationTests(unittest.TestCase):
    def test_failure_preserves_pointer_retry_and_track_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); store = MemoryStorage()
            first = publish_bundle(store,'hu',bundle(root,'hu',1))
            three = publish_bundle(store,'3max',bundle(root,'3max',1))
            second = bundle(root,'hu',2);store.fail = '.onnx'
            with self.assertRaises(StorageError): publish_bundle(store,'hu',second)
            self.assertEqual(json.loads(store.get('hu/current.json')),first)
            store.fail = None
            result = publish_bundle(store,'hu',second)
            self.assertEqual(result['iteration'],2)
            self.assertEqual(json.loads(store.get('3max/current.json')),three)
            writes = [event for event in store.events if event[0]=='put']
            self.assertEqual(writes[-1],('put','hu/current.json'))
            self.assertEqual(publish_bundle(store,'hu',second),result)
            with self.assertRaisesRegex(ValueError,'newer'): publish_bundle(store,'hu',root/'hu-1')

    def test_corruption_rejected_before_upload_and_retention(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); store = MemoryStorage()
            bad = bundle(root,'hu',1);(bad/'average_hu.onnx').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'checksum'): publish_bundle(store,'hu',bad)
            self.assertEqual(store.objects,{})
            for iteration in range(2,7):
                pointer = publish_bundle(store,'hu',bundle(root,'hu',iteration),keep=2)
                prune_remote(store,'hu',pointer)
            self.assertEqual(len(store.objects),5) # two files each + current pointer

    def test_server_credentials_and_supabase_error_codes(self):
        with self.assertRaises(ValueError): SupabaseStorage('https://project.supabase.co','sb_publishable_invalid')
        storage = SupabaseStorage('https://project.supabase.co','sb_secret_test')
        error = HTTPError('https://project.supabase.co',400,'error',{},BytesIO(b'{"statusCode":"404"}'))
        with patch('training.publication.urlopen',side_effect=error):
            with self.assertRaises(StorageError) as result: storage.get('hu/current.json')
        self.assertEqual(result.exception.status,404)
        self.assertNotIn('sb_secret_test',repr(storage))

    def test_real_export_lifecycle_without_checkpoint_reload(self):
        from test_training_runner import tiny_config
        from training.runner import TrainingRunner
        from training.vps import export_live
        import torch
        import onnxruntime as ort
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); config=tiny_config(root)
            config['3max']['enabled']=False
            config['hu']['root_sampling']['tournament_start_players']=2
            runner=TrainingRunner(config);runner.enable_bounded_storage();runner.run_iteration()
            with patch.object(TrainingRunner,'load_checkpoint',side_effect=AssertionError('No reload')):
                directory=export_live(runner)
                store=MemoryStorage();pointer=publish_bundle(store,'hu',directory)
            payload=store.get(f"hu/releases/{pointer['release_id']}/average_hu.onnx",public=True)
            session=ort.InferenceSession(payload,providers=['CPUExecutionProvider'])
            self.assertEqual(session.get_outputs()[0].name,'probabilities')
            self.assertEqual(pointer['iteration'],1)
