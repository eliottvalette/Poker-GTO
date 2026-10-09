"""Exercise coordinated shell deployment with isolated system-command substitutes."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
REVISION = 'a' * 40
PREVIOUS = 'b' * 40


class DeploymentTests(unittest.TestCase):
    def run_deployment(self, *, fail: str = '', selection: str = 'both') -> tuple:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release = root / 'opt/gto/releases' / REVISION
            previous = release.parent / PREVIOUS
            previous.mkdir(parents=True)
            (release / '.venv/bin').mkdir(parents=True)
            (root / 'opt/gto/current').symlink_to(previous)
            (root / 'var/lock').mkdir(parents=True)
            (root / 'etc/gto').mkdir(parents=True)
            (root / 'etc/gto/publication.env').write_text('test')
            for track in ('hu', '3max'):
                data = root / 'var/lib/gto' / track
                data.mkdir(parents=True)
                (data / 'checkpoint.pt').write_text(track)
            commands = root / 'commands'
            commands.mkdir()
            dispatcher = commands / 'dispatch'
            dispatcher.write_text(f'#!{sys.executable}\n' + '''import os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ['COMMAND_LOG'], 'a') as log:
    log.write(name + ' ' + ' '.join(args) + '\\n')
if name == 'id':
    if args == ['-u']: print('0')
elif name == 'readlink':
    print(Path(args[-1]).resolve())
elif name == 'mv':
    os.replace(args[-2], args[-1])
elif name == 'systemctl':
    if args[0] == 'show': print('success')
    if args[:2] == ['enable', '--now'] and args[-1] == os.environ.get('FAIL_COMMAND'):
        sys.exit(1)
elif name == 'runuser':
    if 'GTO_TRACK=3max' in args and os.environ.get('FAIL_COMMAND') == 'preflight':
        sys.exit(1)
''')
            dispatcher.chmod(0o755)
            for name in ('id', 'readlink', 'mv', 'systemctl', 'flock', 'apt-get',
                         'install', 'python3', 'chown', 'runuser', 'sleep'):
                (commands / name).symlink_to(dispatcher)
            for name in ('pip', 'python'):
                (release / '.venv/bin' / name).symlink_to(dispatcher)
            script = (ROOT / 'deploy/vps/install.sh').read_text()
            for prefix in ('/opt/gto', '/var/lib/gto', '/var/lock', '/etc/gto', '/etc/systemd'):
                script = script.replace(prefix, str(root) + prefix)
            script_path = root / 'install.sh'
            script_path.write_text(script)
            log = root / 'commands.log'
            result = subprocess.run(
                [shutil.which('bash'), str(script_path), REVISION, selection],
                env={**os.environ, 'PATH': str(commands) + os.pathsep + os.environ['PATH'],
                     'COMMAND_LOG': str(log), 'FAIL_COMMAND': fail},
                capture_output=True, text=True, timeout=15,
            )
            selected = (root / 'opt/gto/current').resolve().name
            for track in ('hu', '3max'):
                self.assertEqual((root / 'var/lib/gto' / track / 'checkpoint.pt').read_text(), track)
            return result, log.read_text().splitlines(), selected

    def test_both_tracks_stop_before_preflight_and_publish_after_restore(self) -> None:
        result, log, selected = self.run_deployment()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(selected, REVISION)
        preflight = next(i for i, line in enumerate(log) if line.startswith('runuser'))
        for track in ('hu', '3max'):
            self.assertLess(log.index(f'systemctl stop gto-training@{track}.service'), preflight)
        last_start = max(log.index(f'systemctl enable --now gto-training@{t}.service') for t in ('hu', '3max'))
        for track in ('hu', '3max'):
            self.assertGreater(log.index(f'systemctl enable --now gto-publish@{track}.timer'), last_start)

    def test_failure_restores_both_tracks_without_publishing_candidate(self) -> None:
        for failure in ('preflight', 'gto-training@3max.service'):
            with self.subTest(failure=failure):
                result, log, selected = self.run_deployment(fail=failure)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(selected, PREVIOUS)
                for track in ('hu', '3max'):
                    self.assertIn(f'systemctl start gto-training@{track}.service', log)
                    self.assertNotIn(f'systemctl enable --now gto-publish@{track}.timer', log)

    def test_partial_deployment_rejected_when_both_tracks_share_release(self) -> None:
        result, log, selected = self.run_deployment(selection='hu')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('deploy both tracks', result.stderr)
        self.assertEqual(selected, PREVIOUS)
        self.assertFalse(any(line.startswith('systemctl stop') for line in log))
