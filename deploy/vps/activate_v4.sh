#!/usr/bin/env bash
# Activate a checksummed v4 source snapshot using the installed CPU environment.
set -euo pipefail
release=${1:?Explicit immutable staged release directory required}
[[ "$release" =~ ^/opt/gto/releases/selfplay-v4-[a-f0-9]{16}$ ]] || exit 2
[[ "$(id -u)" == 0 && -f "$release/deployment-source.json" && -x "$release/.venv/bin/python" ]] || exit 2
exec 8>/var/lock/ovh-project-deployment.lock
flock -n 8 || { echo 'Another project deployment is active' >&2; exit 1; }
exec 9>/var/lock/gto-deployment.lock
flock -n 9 || { echo 'Another GTO deployment is active' >&2; exit 1; }
previous=$(readlink -f /opt/gto/current)
[[ "$previous" != "$release" ]] || { echo 'v4 is already selected' >&2; exit 1; }
for track in hu 3max; do
  [[ ! -e "/var/lib/gto/$track/selfplay-v4" ]] || { echo "Existing v4 data for $track; inspect before activation" >&2; exit 1; }
  systemctl is-active --quiet "gto-training@$track.service"
done
archive=/var/lib/gto/audit-archives/hu-selfplay-v3/config.json
archive_restore=/var/lib/gto/audit-archives/hu-selfplay-v3/pre-v4-config.json
[[ -f "$archive" && ! -e "$archive_restore" ]] || exit 1
cp -p "$archive" "$archive_restore"
restore() {
  status=$?
  trap - EXIT
  if [[ $status -ne 0 ]]; then
    echo 'v4 activation failed; restoring v3 services and preserving all checkpoint evidence' >&2
    systemctl stop gto-publish@hu.timer gto-publish@3max.timer gto-evaluate.timer gto-hu-audit-archive.service || true
    systemctl stop gto-training@hu.service gto-training@3max.service || true
    ln -sfn "$previous" /opt/gto/current.pending
    mv -Tf /opt/gto/current.pending /opt/gto/current
    cp -p "$archive_restore" "$archive"
    systemctl start gto-training@hu.service gto-training@3max.service || true
    systemctl start gto-publish@hu.timer gto-publish@3max.timer gto-evaluate.timer gto-hu-audit-archive.service || true
  fi
  exit "$status"
}
trap restore EXIT
systemctl stop gto-publish@hu.timer gto-publish@3max.timer gto-evaluate.timer
systemctl stop gto-publish@hu.service gto-publish@3max.service gto-evaluate.service gto-hu-audit-archive.service
systemctl stop gto-training@hu.service gto-training@3max.service
for track in hu 3max; do
  [[ "$(systemctl show "gto-training@$track.service" -p Result --value)" == success ]] || { echo "Unclean stop: $track" >&2; exit 1; }
done
cd "$release"
for track in hu 3max; do
  runuser -u gto -- env PYTHONPATH="$release" TMPDIR="/var/lib/gto/$track/tmp" GTO_DATA_DIR="/var/lib/gto/$track" GTO_TRACK="$track" \
    "$release/.venv/bin/python" deploy/vps/check_resume.py
  runuser -u gto -- env PYTHONPATH="$release" TMPDIR="/var/lib/gto/$track/tmp" GTO_DATA_DIR="/var/lib/gto/$track" GTO_TRACK="$track" \
    "$release/.venv/bin/python" deploy/vps/verify_v4.py
done
# Preserve completed v3 milestones; only pending ones follow the continued run.
"$release/.venv/bin/python" - <<'PY'
import json
from pathlib import Path
from training.checkpoint import atomic_bytes
root = Path('/var/lib/gto/audit-archives/hu-selfplay-v3')
config = json.loads((root / 'config.json').read_text())
pending = [i for i in config['iterations'] if not (root / f'iteration_{i:06d}.json').exists()]
contract = json.loads(Path('/var/lib/gto/hu/selfplay-v4/selfplay.json').read_text())
if any(i <= contract['migration']['source_iteration'] for i in pending):
    raise ValueError('A pending milestone is already past the migration boundary')
destination = Path('/var/lib/gto/audit-archives/hu-selfplay-v4')
destination.mkdir(mode=0o750)
import shutil
shutil.chown(destination, user='gto', group='gto')
config.update(source='/var/lib/gto/hu/selfplay-v4', destination=str(destination),
              iterations=pending, run_id=contract['run_id'])
atomic_bytes(root / 'config.json', (json.dumps(config, indent=2) + '\n').encode())
shutil.chown(root / 'config.json', user='gto', group='gto')
PY
ln -sfn "$release" /opt/gto/current.pending
mv -Tf /opt/gto/current.pending /opt/gto/current
systemctl start gto-training@hu.service gto-training@3max.service
systemctl is-active --quiet gto-training@hu.service gto-training@3max.service
systemctl start gto-publish@hu.timer gto-publish@3max.timer gto-evaluate.timer gto-hu-audit-archive.service
echo "Activated v4 for HU and 3-max: $release"
