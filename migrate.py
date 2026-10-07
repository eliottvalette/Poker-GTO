"""Review and publish trained checkpoint policies through a plain ASCII menu."""
import sys
from training.workflow import candidate_checkpoint, prepare_migration


def main() -> None:
    print("+-----------------------------------------------------+")
    print("| Poker policies                                      |")
    print("| 1  Activate checkpoint as the reference policy      |")
    print("| 2  Export both checkpoint policies to ONNX for UI   |")
    print("| 3  Activate both policies and export ONNX           |")
    print("| 0  Exit                                             |")
    print("+-----------------------------------------------------+")
    for track in ("3max", "hu"):
        path = candidate_checkpoint(track)
        print(f"{track}: {'available' if path.is_file() else 'missing'} - {path}")
    choice = input("Selection: ").strip()
    if choice == "0":
        return
    actions = {"1": "activate", "2": "export", "3": "both"}
    if choice not in actions:
        raise ValueError(f"Invalid menu selection: {choice!r}")
    tracks = ("3max", "hu")
    if choice == "1":
        selected = input("Activate [3] 3-max, [H] HU, [B] both: ").strip().upper()
        options = {"3": ("3max",), "H": ("hu",), "B": ("3max", "hu")}
        if selected not in options:
            raise ValueError(f"Invalid track selection: {selected!r}")
        tracks = options[selected]
    prepared = prepare_migration(actions[choice], tracks)
    try:
        print("\nPrepared and validated:")
        for row in prepared.summary:
            print(f"  {row['track']}: iteration {row['iteration']}; source {row['source']}")
            print(f"    checkpoint SHA256: {row['checkpoint_sha256']}")
        print(f"  Action: {prepared.action}")
        print("  Publication switches the active catalog atomically.")
        if input("Type APPLY to publish, anything else to cancel: ").strip() != "APPLY":
            print("Cancelled.")
            return
        prepared.publish()
        print("Publication complete.")
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
