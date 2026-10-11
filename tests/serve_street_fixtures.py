"""Local-only fixture server for the actual static UI and experimental model bundles."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit


def serve(static_root: Path, experiment_root: Path, *, port: int = 3137) -> None:
    static_root = static_root.resolve()
    experiment_root = experiment_root.resolve()
    fixture = experiment_root/'canonical-browser-fixtures.json'
    exports = {track: experiment_root/f'canonical-groups-{track}-seed101/exports/iteration_000005'
               for track in ('hu', '3max')}
    if not fixture.is_file() or any(not directory.is_dir() for directory in exports.values()):
        raise FileNotFoundError(f"Canonical street fixture cohort missing at {experiment_root}")

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if urlsplit(self.path).path == '/policy/index.json':
                payload = json.dumps({'version': 1, 'exports': {
                    track: {'bundle_id': ('a' if track == 'hu' else 'b')*64, 'iteration': 5}
                    for track in exports}}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(payload)
                return
            super().do_GET()

        def translate_path(self, path):
            path = unquote(urlsplit(path).path)
            if path == '/__street-fixtures.json':
                return str(fixture)
            if path.startswith('/policy/releases/'):
                name = Path(path).name
                track = 'hu' if name.startswith('average_hu') else '3max'
                return str(exports[track]/name)
            result = (static_root/path.lstrip('/')).resolve()
            result.relative_to(static_root)
            return str(result)

    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()
