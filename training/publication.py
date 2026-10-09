"""Publish validated small policies to Supabase Storage; never load training checkpoints."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.error import HTTPError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

TRACKS = {'hu': 2, '3max': 3}
SHA = re.compile(r'^[a-f0-9]{64}$')
MAX_FILE = 16 * 1024 * 1024


def encoded(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


class StorageError(RuntimeError):
    def __init__(self, status: int, operation: str):
        super().__init__(f'Storage {operation} failed: HTTP {status}')
        self.status = status


@dataclass
class SupabaseStorage:
    url: str
    secret: str = field(repr=False)
    bucket: str = 'poker-policies'
    timeout: float = 30

    def __post_init__(self):
        parsed = urlparse(self.url)
        if parsed.scheme != 'https' or not parsed.netloc or parsed.path not in ('', '/'):
            raise ValueError('SUPABASE_URL must be an HTTPS project origin')
        if not self.secret or self.secret.startswith('sb_publishable_'):
            raise ValueError('A server-only Supabase secret/service_role key is required')
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', self.bucket):
            raise ValueError('Invalid Storage bucket name')
        self.url = self.url.rstrip('/')

    @classmethod
    def from_environment(cls):
        return cls(os.environ['SUPABASE_URL'], os.environ['SUPABASE_SECRET_KEY'],
                   os.environ.get('SUPABASE_POLICY_BUCKET', 'poker-policies'))

    def request(self, method: str, path: str, data: bytes | None = None, *, public=False, headers=None) -> bytes:
        request_headers = {} if public else {'apikey': self.secret}
        # Modern secret keys authenticate via apikey; legacy JWT service keys also use Bearer.
        if not public and not self.secret.startswith('sb_secret_'):
            request_headers['Authorization'] = f'Bearer {self.secret}'
        request_headers.update(headers or {})
        request = Request(f'{self.url}/storage/v1/{path}', data=data, method=method, headers=request_headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = response.read(MAX_FILE+1)
                if len(result) > MAX_FILE:
                    raise ValueError('Storage response exceeds policy size limit')
                return result
        except HTTPError as error:
            # Response bodies can contain sensitive diagnostic context; report the operation/status.
            status = error.code
            try:
                reported = json.loads(error.read(8192)).get('statusCode')
                if str(reported) in ('400', '401', '403', '404', '409', '413', '429'):
                    status = int(reported)
            except (ValueError, AttributeError):
                pass
            raise StorageError(status, f'{method} {path}') from None

    def provision(self) -> None:
        try:
            bucket = json.loads(self.request('GET', f'bucket/{self.bucket}'))
        except StorageError as error:
            if error.status != 404:
                raise
            self.request('POST', 'bucket', encoded({'id': self.bucket, 'name': self.bucket,
                'public': True, 'file_size_limit': MAX_FILE,
                'allowed_mime_types': ['application/json', 'application/octet-stream']}),
                headers={'Content-Type': 'application/json'})
            bucket = json.loads(self.request('GET', f'bucket/{self.bucket}'))
        if bucket.get('public') is not True:
            raise ValueError('Policy bucket exists but is not public; review its configuration explicitly')

    def get(self, key: str, *, public=False) -> bytes:
        route = 'object/public' if public else 'object/authenticated'
        return self.request('GET', f'{route}/{self.bucket}/{quote(key, safe="/")}', public=public)

    def put(self, key: str, data: bytes, *, mutable=False) -> None:
        if len(data) > MAX_FILE:
            raise ValueError('Policy exceeds bucket file limit')
        self.request('POST', f'object/{self.bucket}/{quote(key, safe="/")}', data,
            headers={'Content-Type': 'application/json' if key.endswith('.json') else 'application/octet-stream',
                     'Cache-Control': 'no-cache, max-age=0' if mutable else 'public, max-age=31536000, immutable',
                     'x-upsert': str(mutable).lower()})

    def ensure(self, key: str, data: bytes) -> None:
        try:
            existing = self.get(key)
        except StorageError as error:
            if error.status != 404:
                raise
            try:
                self.put(key, data)
            except StorageError as conflict:
                if conflict.status != 409:
                    raise
            existing = self.get(key)
        if existing != data:
            raise ValueError(f'Immutable remote artifact mismatch: {key}')

    def list(self, prefix: str) -> list[dict]:
        result = []
        while True:
            page = json.loads(self.request('POST', f'object/list/{self.bucket}',
                encoded({'prefix': prefix, 'limit': 100, 'offset': len(result),
                         'sortBy': {'column': 'name', 'order': 'asc'}}),
                headers={'Content-Type': 'application/json'}))
            if not isinstance(page, list):
                raise ValueError('Invalid Storage list response')
            result.extend(page)
            if len(page) < 100:
                return result

    def remove(self, keys: list[str]) -> None:
        if keys:
            self.request('DELETE', f'object/{self.bucket}', encoded({'prefixes': keys}),
                         headers={'Content-Type': 'application/json'})

    def verify_anonymous_write_denied(self, publishable_key: str) -> None:
        """Check that the browser credential cannot publish or overwrite policies."""
        key = '__anonymous_access_probe__.json'
        try:
            self.request('POST', f'object/{self.bucket}/{key}', b'{}', public=True,
                         headers={'apikey': publishable_key, 'Content-Type': 'application/json'})
        except StorageError as error:
            if error.status not in (401, 403):
                raise
        else:
            self.remove([key])
            raise ValueError('Anonymous upload is allowed; review Storage RLS before training publication')


def validate_pointer(raw: dict, track: str) -> dict:
    if (raw.get('version') != 1 or raw.get('track') != track or not SHA.fullmatch(raw.get('release_id', ''))
            or type(raw.get('iteration')) is not int or raw['iteration'] < 1
            or not isinstance(raw.get('history'), list)
            or any(not isinstance(value, str) or not SHA.fullmatch(value) for value in raw['history'])):
        raise ValueError(f'Invalid {track} publication pointer')
    return raw


def publish_bundle(storage: SupabaseStorage, track: str, bundle: Path, *, keep: int = 24) -> dict:
    """One writer per track; pointer commits only after public read-back of both artifacts."""
    if track not in TRACKS or type(keep) is not int or keep < 2:
        raise ValueError('Valid dedicated track and at least two retained releases required')
    seal = json.loads((bundle/'bundle.json').read_text())
    live = seal.get('kind') == 'live_training_export' and seal.get('tracks') == [track]
    migrated = seal.get('kind') == 'onnx_export' and track in seal.get('sources', {})
    if seal.get('version') != 1 or not (live or migrated):
        raise ValueError('Expected a sealed live-training or migrated ONNX export')
    files = {}
    for suffix in ('onnx', 'json'):
        name = f'average_{track}.{suffix}'
        if (bundle/name).stat().st_size > MAX_FILE:
            raise ValueError(f'Export too large: {name}')
        data = (bundle/name).read_bytes()
        if hashlib.sha256(data).hexdigest() != seal['files'].get(name):
            raise ValueError(f'Export checksum mismatch: {name}')
        files[name] = data
    manifest = json.loads(files[f'average_{track}.json'])
    if (manifest.get('supported_player_counts') != [TRACKS[track]]
            or manifest.get('model_sha256') != hashlib.sha256(files[f'average_{track}.onnx']).hexdigest()
            or (live and manifest.get('iteration') != seal.get('iteration'))
            or type(manifest.get('iteration')) is not int or manifest['iteration'] < 1):
        raise ValueError('Export track, model hash or iteration mismatch')
    iteration = manifest['iteration']
    release = hashlib.sha256(encoded({name: hashlib.sha256(data).hexdigest() for name, data in files.items()})).hexdigest()
    key = f'{track}/current.json'
    try:
        previous = validate_pointer(json.loads(storage.get(key)), track)
    except StorageError as error:
        if error.status != 404:
            raise
        previous = None
    if previous and (previous['iteration'] > iteration or
                     (previous['iteration'] == iteration and previous['release_id'] != release)):
        raise ValueError('Refusing to replace a newer or conflicting published iteration')
    if previous and previous['release_id'] == release:
        return previous
    for name, data in files.items():
        object_key = f'{track}/releases/{release}/{name}'
        storage.ensure(object_key, data)
        if storage.get(object_key, public=True) != data:
            raise ValueError(f'Public read-back mismatch: {object_key}')
    if previous and previous['release_id'] == release:
        pointer = previous
    else:
        history = [] if previous is None else [previous['release_id'], *previous['history']]
        pointer = {'version': 1, 'track': track, 'release_id': release, 'iteration': iteration,
                   'published_at': datetime.now(timezone.utc).isoformat(), 'history': history[:keep-1]}
        storage.put(key, encoded(pointer), mutable=True)
        if json.loads(storage.get(key)) != pointer:
            raise ValueError('Publication pointer read-back mismatch')
    return pointer


def prune_remote(storage: SupabaseStorage, track: str, pointer: dict, *, grace_hours: int = 24) -> None:
    """Keep current/history plus a grace period for in-flight clients and partial uploads."""
    validate_pointer(pointer, track)
    retained = {pointer['release_id'], *pointer['history']}
    cutoff = datetime.now(timezone.utc)-timedelta(hours=grace_hours)
    for item in storage.list(f'{track}/releases'):
        release = item.get('name', '')
        if not SHA.fullmatch(release) or release in retained:
            continue
        prefix = f'{track}/releases/{release}'
        files = storage.list(prefix)
        expected = {f'average_{track}.onnx', f'average_{track}.json'}
        if not files or any(row.get('name') not in expected or not row.get('created_at') for row in files):
            raise ValueError(f'Unexpected remote release contents: {prefix}')
        if all(datetime.fromisoformat(row['created_at'].replace('Z', '+00:00')) < cutoff for row in files):
            storage.remove([f"{prefix}/{row['name']}" for row in files])


def publish_pending(storage: SupabaseStorage, track: str, data_dir: Path) -> dict | None:
    """Called by a timer; missing initial export is an explicit waiting state."""
    import fcntl
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir/'.publisher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        selected = data_dir/'exports/current.json'
        if not selected.exists():
            print(f'[{track}] waiting for first validated live export', flush=True)
            return None
        selection = json.loads(selected.read_text())
        directory = selection['directory']
        if not re.fullmatch(r'iteration_[0-9]+', directory):
            raise ValueError('Invalid local export selection')
        pointer = publish_bundle(storage, track, data_dir/'exports'/directory)
        prune_remote(storage, track, pointer)
        print(f'[{track}] published iteration {pointer["iteration"]}, release {pointer["release_id"]}', flush=True)
        return pointer
