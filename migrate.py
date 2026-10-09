"""Review and publish trained checkpoint policies through a plain ASCII menu."""
import sys
import time
from training.workflow import candidate_checkpoint, prepare_migration


def main() -> None:
    print("+-----------------------------------------------------+")
    print("| Poker policies                                      |")
    print("| 1  Export and activate 3-max policy for UI           |")
    print("| 2  Export and activate HU policy for UI              |")
    print("| 0  Exit                                             |")
    print("+-----------------------------------------------------+")
    for track in ("3max", "hu"):
        path = candidate_checkpoint(track)
        print(f"{track}: {'available' if path.is_file() else 'missing'} - {path}")
    choice = input("Selection: ").strip()
    if choice == "0":
        return
    options = {"1": ("3max",), "2": ("hu",)}
    if choice not in options:
        raise ValueError(f"Invalid menu selection: {choice!r}")
    prepared = prepare_migration("export", options[choice], progress=lambda message: print(message, flush=True))
    try:
        print("\nPrepared and validated:")
        for row in prepared.summary:
            print(f"  {row['track']}: iteration {row['iteration']}; source {row['source']}")
            print(f"    checkpoint SHA256: {row['checkpoint_sha256']}")
        print(f"  Action: {prepared.action}")
        print("  Publication switches the active catalog atomically.")
        print("Validating and publishing bundles...", flush=True)
        started = time.perf_counter()
        prepared.publish()
        print(f"Publication complete in {time.perf_counter() - started:.2f}s.", flush=True)
    finally:
        prepared.close()


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise ValueError("This script takes no command-line arguments; edit the Python settings or use the interactive menu")
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
