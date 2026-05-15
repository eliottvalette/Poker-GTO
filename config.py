from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent


# =========================
# Shared paths / game setup
# =========================

SEED: int | None = None  # None => time-based seed for CFR, fixed defaults elsewhere.
STACKS: tuple[int, int, int] = (100, 100, 100)

POLICY_PATH = ROOT_DIR / "policy" / "avg_policy.json.gz"
UI_POLICY_PATH = ROOT_DIR / "ui" / "public" / "avg_policy.json.gz"


# =========================
# CFR training
# =========================

CFR_ITERATIONS = 1_000_000
CFR_MODE = "sync"  # "sync", "independent", "sequential"
CFR_WORKERS: int | None = None  # None => 80% of available CPUs.
CFR_ITERATIONS_PER_WORKER = 1_000
CFR_ROUNDS: int | None = None
CFR_WARM_START: Path | str | None = POLICY_PATH
CFR_SAVE_POLICY = POLICY_PATH
CFR_SAVE_UI_COPY = UI_POLICY_PATH
CFR_SKIP_CSV = False


# =========================
# Neural policy training
# =========================

ML_POLICY_PATH = POLICY_PATH
ML_MODEL_PATH = ROOT_DIR / "ml" / "trained_policy_model.pth"
ML_EPOCHS = 10
ML_BATCH_SIZE = 64
ML_LEARNING_RATE = 3e-4
ML_EVAL_SAMPLES = 1_000


# =========================
# Policy benchmark
# =========================

BENCHMARK_HANDS = 1_000
BENCHMARK_SEED = 20_260_515
BENCHMARK_POLICY = POLICY_PATH
BENCHMARK_OLD_POLICY = "git:HEAD:policy/avg_policy.json.gz"
BENCHMARK_OUT_DIR = ROOT_DIR / "benchmarks" / "configured_{timestamp}"
BENCHMARK_POLICY_EPSILON = 0.0
BENCHMARK_MAX_DECISIONS = 250
BENCHMARK_SCENARIOS = (
    "current_vs_2x_random",
    "current_vs_2x_calling_station",
    "current_vs_2x_nit",
    "current_vs_2x_aggro",
    "current_vs_2x_shove_fold",
    "current_vs_old_vs_random",
    "current_vs_old_vs_station",
    "current_vs_2x_old",
)
