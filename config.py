"""Explicit modest defaults; importing configuration never launches compute."""
from pathlib import Path
from blind_schedule import DEFAULT_SIMULATION_SCHEDULE

ROOT_DIR = Path(__file__).resolve().parent
SEED = 42
STACKS = (25.0, 25.0, 25.0)
SMALL_BLIND = 0.5
BIG_BLIND = 1.0
BLIND_SCHEDULE = DEFAULT_SIMULATION_SCHEDULE
SOLVER_OBJECTIVE = "hand_chip_delta"
PRIMARY_TRAVERSAL = "external_sampling"
TRAVERSAL_WORKERS = 1
MAX_TRAVERSAL_NODES = 10000
MAX_TRAVERSAL_DEPTH = 300
ADVANTAGE_CAPACITY = 10000  # Per player.
STRATEGY_CAPACITY = 10000
MODEL_PATH = ROOT_DIR / "policy" / "hand_cev_average.pt"
TABULAR_POLICY_PATH = ROOT_DIR / "policy" / "tabular_hand_cev_v3.json"
