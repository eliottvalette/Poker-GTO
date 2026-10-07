"""Resume or initialize 3-max training for the configured session duration."""
import sys
from training.settings import SESSION_SECONDS
from training.workflow import train_track

if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise ValueError("This script takes no command-line arguments; edit the Python settings or use the interactive menu")
    try:
        train_track("3max", SESSION_SECONDS["3max"])
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
