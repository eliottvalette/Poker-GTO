"""Explicit modest defaults; importing configuration never launches compute."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SEED = 42
STACKS = (25.0, 25.0, 25.0)
SMALL_BLIND = 0.5
BIG_BLIND = 1.0
TRAVERSAL_WORKERS = 1
MAX_TRAVERSAL_NODES = 10000
MAX_TRAVERSAL_DEPTH = 300
ADVANTAGE_CAPACITY = 10000  # Per player.
STRATEGY_CAPACITY = 10000
MODEL_PATH = ROOT_DIR / "policy" / "deep_cfr_average.pt"
TABULAR_POLICY_PATH = ROOT_DIR / "policy" / "tabular_average_v2.json"
