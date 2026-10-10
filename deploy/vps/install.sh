#!/usr/bin/env bash
# Invoked by the SSH deployment job with an immutable Git revision and explicit track selection.
set -euo pipefail
revision=${1:?Commit SHA required}
track=${2:?Track required}
[[ "$revision" =~ ^[a-f0-9]{40}$ ]] || { echo 'Invalid revision' >&2; exit 1; }
case "$track" in
  hu|3max) tracks=("$track") ;;
  both) tracks=(hu 3max) ;;
  *) echo 'Invalid track selection: expected hu, 3max or both' >&2; exit 1 ;;
esac
[[ "$(id -u)" == 0 ]] || { echo 'Installation requires root' >&2; exit 1; }
exec 9>/var/lock/gto-deployment.lock
flock -n 9 || { echo 'Another deployment is running' >&2; exit 1; }
release=/opt/gto/releases/$revision
current=/opt/gto/current
previous=$(readlink -f "$current" || true)
# Every training sharing this code pointer must participate in the deployment.
for existing in hu 3max; do
  if [[ -d "/var/lib/gto/$existing" && " ${tracks[*]} " != *" $existing "* ]]; then
    echo "Existing $existing data shares the code release; deploy both tracks" >&2
    exit 1
  fi
done
active_training=()
active_timers=()
evaluation_active=false
if systemctl is-active --quiet gto-evaluate.timer; then evaluation_active=true; fi
for selected in "${tracks[@]}"; do
  if systemctl is-active --quiet "gto-training@$selected.service"; then active_training+=("$selected"); fi
  if systemctl is-active --quiet "gto-publish@$selected.timer"; then active_timers+=("$selected"); fi
done
restore() {
  status=$?
  trap - EXIT
  if [[ $status -ne 0 ]]; then
    echo 'Deployment failed; preserving checkpoints and restoring prior code' >&2
    systemctl stop gto-evaluate.timer gto-evaluate.service || true
    for selected in "${tracks[@]}"; do
      systemctl stop "gto-publish@$selected.timer" "gto-publish@$selected.service" || true
      systemctl stop "gto-training@$selected.service" || true
    done
    if [[ -n "$previous" && -d "$previous" ]]; then
      ln -sfn "$previous" /opt/gto/current.pending
      mv -Tf /opt/gto/current.pending "$current"
      for selected in "${active_training[@]}"; do systemctl start "gto-training@$selected.service"; done
      for selected in "${active_timers[@]}"; do systemctl start "gto-publish@$selected.timer"; done
      if [[ "$evaluation_active" == true ]]; then systemctl start gto-evaluate.timer; fi
    fi
  fi
  exit "$status"
}
trap restore EXIT
systemctl stop gto-evaluate.timer gto-evaluate.service 2>/dev/null || true
for selected in "${tracks[@]}"; do
  systemctl stop "gto-publish@$selected.timer" 2>/dev/null || true
  systemctl stop "gto-publish@$selected.service" 2>/dev/null || true
done
for selected in "${active_training[@]}"; do
  service=gto-training@$selected.service
  systemctl stop "$service"
  # A SIGKILL stop is not accepted as a successful checkpoint boundary.
  if [[ "$(systemctl show "$service" -p Result --value)" != success ]]; then
    echo "Training $selected did not stop cleanly; refusing deployment" >&2; exit 1
  fi
done
apt-get update -qq
apt-get install -y python3 python3-venv ca-certificates
id gto >/dev/null 2>&1 || useradd --system --create-home --home-dir /var/lib/gto --shell /usr/sbin/nologin gto
install -d -o gto -g gto /var/lib/gto/evaluation
for selected in "${tracks[@]}"; do
  data=/var/lib/gto/$selected
  install -d -o gto -g gto "$data" "$data/tmp"
  [[ -f "$data/checkpoint.pt" || -f "$data/resume.json" ]] || { echo "Upload a compact $selected checkpoint before deployment" >&2; exit 1; }
done
install -d -m 0750 /etc/gto
install -d -m 0755 /opt/gto/releases
[[ -f /etc/gto/publication.env ]] || { echo 'Missing /etc/gto/publication.env' >&2; exit 1; }
if [[ ! -d "$release" ]]; then
  mkdir "$release"
  tar -xzf "/opt/gto/incoming/$revision.tar.gz" -C "$release"
fi
python3 -m venv "$release/.venv"
"$release/.venv/bin/pip" install --disable-pip-version-check 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cpu
"$release/.venv/bin/pip" install --disable-pip-version-check -r "$release/deploy/vps/requirements.txt"
cd "$release"
# Sequential preflight avoids loading both checkpoints at once.
for selected in "${tracks[@]}"; do
  data=/var/lib/gto/$selected
  chown -R gto:gto "$data"
  runuser -u gto -- env PYTHONPATH="$release" TMPDIR="$data/tmp" GTO_DATA_DIR="$data" GTO_TRACK="$selected" \
    "$release/.venv/bin/python" deploy/vps/check_resume.py
done
install -m 0644 deploy/vps/gto-training@.service deploy/vps/gto-publish@.service deploy/vps/gto-publish@.timer deploy/vps/gto-evaluate.service deploy/vps/gto-evaluate.timer /etc/systemd/system/
install -m 0644 deploy/vps/journald-gto.conf /etc/systemd/journald@gto.conf
ln -sfn "$release" /opt/gto/current.pending
mv -Tf /opt/gto/current.pending "$current"
ln -sfn /opt/gto/current/.venv /opt/gto/venv
systemctl daemon-reload
for selected in "${tracks[@]}"; do
  systemctl enable --now "gto-training@$selected.service"
done
sleep 5
for selected in "${tracks[@]}"; do systemctl is-active --quiet "gto-training@$selected.service"; done
# Publication starts only after every training has restored successfully.
for selected in "${tracks[@]}"; do systemctl enable --now "gto-publish@$selected.timer"; done
systemctl enable --now gto-evaluate.timer
# Retain current and previous code environments, independent of training data.
for old in /opt/gto/releases/*; do
  [[ -d "$old" && "${old##*/}" =~ ^[a-f0-9]{40}$ ]] || continue
  if [[ "$old" != "$release" && "$old" != "$previous" ]]; then rm -rf -- "$old"; fi
done
rm -f -- "/opt/gto/incoming/$revision.tar.gz"
echo "Activated $revision for $track; checkpoint and replay retained"
