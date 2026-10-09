#!/usr/bin/env bash
# Invoked by the SSH deployment job with an immutable Git revision and dedicated track.
set -euo pipefail
revision=${1:?Commit SHA required}
track=${2:?Track required}
[[ "$revision" =~ ^[a-f0-9]{40}$ ]] || { echo 'Invalid revision' >&2; exit 1; }
[[ "$track" == hu || "$track" == 3max ]] || { echo 'Invalid track' >&2; exit 1; }
[[ "$(id -u)" == 0 ]] || { echo 'Installation requires root' >&2; exit 1; }
exec 9>/var/lock/gto-deployment.lock
flock -n 9 || { echo 'Another deployment is running' >&2; exit 1; }
release=/opt/gto/releases/$revision
current=/opt/gto/current
data=/var/lib/gto/$track
previous=$(readlink -f "$current" || true)
service=gto-training@$track.service
was_active=false
if systemctl is-active --quiet "$service"; then was_active=true; fi
restore() {
  status=$?
  trap - EXIT
  if [[ $status -ne 0 ]]; then
    echo 'Deployment failed; preserving checkpoints and restoring prior code' >&2
    systemctl stop "$service" || true
    if [[ -n "$previous" && -d "$previous" ]]; then
      ln -sfn "$previous" /opt/gto/current.pending
      mv -Tf /opt/gto/current.pending "$current"
      if [[ "$was_active" == true ]]; then
        systemctl start "$service"
        systemctl start "gto-publish@$track.timer"
      fi
    fi
  fi
  exit "$status"
}
trap restore EXIT
systemctl stop "gto-publish@$track.timer" 2>/dev/null || true
systemctl stop "gto-publish@$track.service" 2>/dev/null || true
if [[ "$was_active" == true ]]; then systemctl stop "$service"; fi
# A SIGKILL stop is not accepted as a successful checkpoint boundary.
if [[ "$was_active" == true && "$(systemctl show "$service" -p Result --value)" != success ]]; then
  echo 'Training did not stop cleanly; refusing deployment' >&2; exit 1
fi
apt-get update -qq
apt-get install -y python3 python3-venv ca-certificates
id gto >/dev/null 2>&1 || useradd --system --create-home --home-dir /var/lib/gto --shell /usr/sbin/nologin gto
install -d -o gto -g gto "$data" "$data/tmp"
install -d -m 0750 /etc/gto /opt/gto/releases
[[ -f /etc/gto/publication.env ]] || { echo 'Missing /etc/gto/publication.env' >&2; exit 1; }
[[ -f "$data/checkpoint.pt" || -f "$data/resume.json" ]] || { echo 'Upload a compact dedicated checkpoint before deployment' >&2; exit 1; }
if [[ ! -d "$release" ]]; then
  mkdir "$release"
  tar -xzf "/opt/gto/incoming/$revision.tar.gz" -C "$release"
fi
python3 -m venv "$release/.venv"
"$release/.venv/bin/pip" install --disable-pip-version-check 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cpu
"$release/.venv/bin/pip" install --disable-pip-version-check -r "$release/deploy/vps/requirements.txt"
chown -R gto:gto "$data"
cd "$release"
runuser -u gto -- env PYTHONPATH="$release" TMPDIR="$data/tmp" GTO_DATA_DIR="$data" GTO_TRACK="$track" \
  "$release/.venv/bin/python" deploy/vps/check_resume.py
install -m 0644 deploy/vps/gto-training@.service deploy/vps/gto-publish@.service deploy/vps/gto-publish@.timer /etc/systemd/system/
install -m 0644 deploy/vps/journald-gto.conf /etc/systemd/journald@gto.conf
ln -sfn "$release" /opt/gto/current.pending
mv -Tf /opt/gto/current.pending "$current"
ln -sfn /opt/gto/current/.venv /opt/gto/venv
systemctl daemon-reload
systemctl enable --now "$service" "gto-publish@$track.timer"
sleep 5
systemctl is-active --quiet "$service"
# Retain current and previous code environments, independent of training data.
for old in /opt/gto/releases/*; do
  [[ -d "$old" && "${old##*/}" =~ ^[a-f0-9]{40}$ ]] || continue
  if [[ "$old" != "$release" && "$old" != "$previous" ]]; then rm -rf -- "$old"; fi
done
rm -f -- "/opt/gto/incoming/$revision.tar.gz"
echo "Activated $revision for $track; checkpoint and replay retained"
