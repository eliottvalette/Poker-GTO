"""Service entrypoint; environment selects the explicit dedicated track and data root."""
import os
import socket
from pathlib import Path
from training.selfplay import run


def notify_ready() -> None:
    address = os.environ.get('NOTIFY_SOCKET')
    if address:
        if address.startswith('@'):
            address = '\0' + address[1:]
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as notifier:
            notifier.connect(address)
            notifier.sendall(b'READY=1\nSTATUS=Self-play checkpoint loaded; continuous training ready')

if __name__ == '__main__':
    track = os.environ['GTO_TRACK']
    directory = Path(os.environ['GTO_DATA_DIR'])
    run(directory / 'selfplay-v3', track, notify_ready)
