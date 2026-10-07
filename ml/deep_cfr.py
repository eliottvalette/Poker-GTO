"""Frozen outer iterations, bounded sample generation and central Deep CFR training."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import asdict, dataclass, field
import math
import resource
import sys
import time
from pathlib import Path
import random
from typing import Callable, TypeVar
import torch
from actions import ACTION_IDS
from poker_game_expresso import HandState
from tournament import TournamentState
from cfr_solver import GameState, Traversal, TraversalBudgetExceeded, hand_root, regret_matching, validate_strategy
from infoset import STATE_VERSION, Observation
from features import FEATURE_SCHEMA_VERSION, SUPPORTED_FEATURE_VERSIONS
from features.neural import NeuralObservation, observe_neural, numeric_names
from features.neural import NEURAL_NUMERIC_NAMES as NUMERIC_NAMES
from ml.memory import DEFAULT_BYTE_BUDGET, TRAVERSAL_MODES, ReservoirMemory, TrainingSample, sample_bytes
from ml.model import MODEL_ARCHITECTURE, AdvantageNetwork, AveragePolicyNetwork, encode_batch, model_architecture
from ml.train import fit

_TraversalResult = TypeVar("_TraversalResult")


def strategy_collector(player_count: int, traversal_mode: str) -> str:
    if player_count not in (2, 3) or traversal_mode not in TRAVERSAL_MODES:
        raise ValueError(f"Invalid collector context: players={player_count}, mode={traversal_mode}")
    if traversal_mode == "outcome_sampling":
        return "outcome_importance"
    return "opponent_nodes" if player_count == 2 else "partial_enumeration"


@dataclass(frozen=True)
class ModelSnapshot:
    version: int
    objective: str
    advantage_weights: dict[str, torch.Tensor] | None
    uniform_initial: bool
    player_count: int

    def validate(self) -> None:
        if self.objective != "hand_chip_delta":
            raise ValueError(f"Only per-hand chip-delta learning is supported, received snapshot objective={self.objective}")
        if self.uniform_initial != (self.version == 0):
            raise ValueError(f"Invalid snapshot initial/version contract: {self.version}")
        if self.player_count not in (2, 3) or self.uniform_initial != (self.advantage_weights is None):
            raise ValueError(f"Invalid shared advantage snapshot: version={self.version}, players={self.player_count}")


@dataclass(frozen=True)
class TraversalTask:
    task_id: int
    player: int
    root: GameState
    seed: int
    max_nodes: int
    max_depth: int
    sample_byte_budget: int = DEFAULT_BYTE_BUDGET
    traversal_mode: str = "external_sampling"
    epsilon: float = 0.6


@dataclass
class GeneratedSamples:
    task_id: int
    version: int
    advantages: list[TrainingSample]
    strategies: list[TrainingSample]
    nodes: int
    value: float
    traversal_mode: str = "external_sampling"
    diagnostics: dict | None = None
    max_depth_seen: int = 0
    worker_seconds: float = field(default=0.0, compare=False)
    worker_cpu_seconds: float = field(default=0.0, compare=False)
    worker_peak_rss_bytes: int = field(default=0, compare=False)
    coverage: dict | None = None


def _diagnostic_metadata(diagnostics) -> dict:
    raw = asdict(diagnostics)
    zeros = {}
    for name, values in raw.items():
        if name.startswith("log_") and isinstance(values, list):
            if any(not math.isfinite(value) and value != -math.inf for value in values):
                raise ValueError(f"Invalid outcome diagnostic {name}: {values}")
            zeros[name] = sum(value == -math.inf for value in values)
            raw[name] = [None if value == -math.inf else value for value in values]
    raw["structural_zero_log_counts"] = zeros
    return raw


class FrozenStrategy:
    """One immutable inference model per frozen worker snapshot."""
    def __init__(self, snapshot: ModelSnapshot) -> None:
        snapshot.validate()
        self.snapshot = snapshot
        self.model = None
        if not snapshot.uniform_initial:
            self.model = AdvantageNetwork()
            self.model.load_state_dict(snapshot.advantage_weights, strict=True)
            self.model.eval()

    def __call__(self, obs: Observation | NeuralObservation) -> tuple[float, ...]:
        if obs.objective != self.snapshot.objective:
            raise ValueError(f"Snapshot objective {self.snapshot.objective} != root {obs.objective}")
        if self.snapshot.uniform_initial:
            return regret_matching([0.0] * len(ACTION_IDS), obs.legal_mask)
        with torch.no_grad():
            values = tuple(self.model(encode_batch([obs]))[0].tolist())
        return regret_matching(values, obs.legal_mask)


def generate_samples(snapshot: ModelSnapshot, task: TraversalTask,
                     frozen_strategy: FrozenStrategy | None = None) -> GeneratedSamples:
    snapshot.validate()
    if not isinstance(task.sample_byte_budget, int) or task.sample_byte_budget < 1:
        raise ValueError(f"Traversal sample byte budget must be positive: {task.sample_byte_budget}")
    if task.traversal_mode not in TRAVERSAL_MODES:
        raise ValueError(f"Invalid task traversal mode: {task.traversal_mode}; expected {TRAVERSAL_MODES}")
    if not math.isfinite(task.epsilon) or not 0 < task.epsilon <= 1:
        raise ValueError(f"Exploration epsilon must be in (0, 1]: {task.epsilon}")
    started = time.perf_counter()
    cpu_started = time.process_time()
    torch.set_num_threads(1)
    if isinstance(task.root, HandState):
        root = task.root
    elif isinstance(task.root, TournamentState):
        root = task.root.hand
    else:
        raise TypeError(f"Traversal requires HandState/TournamentState, received {type(task.root).__name__}")
    if root is None or len(root.players) != snapshot.player_count or task.player not in root.players:
        raise ValueError(f"Snapshot/root track mismatch: expected {snapshot.player_count} players, traverser={task.player}")
    if frozen_strategy is not None and frozen_strategy.snapshot is not snapshot:
        raise ValueError("Frozen inference strategy does not belong to the requested snapshot")
    strategy = FrozenStrategy(snapshot) if frozen_strategy is None else frozen_strategy
    advantages, strategies = [], []
    generated_bytes = 0
    from training.metrics import Coverage
    coverage = Coverage()
    def sink(kind, destination):
        def append(obs, target, weight):
            nonlocal generated_bytes
            sample = TrainingSample(snapshot.version + 1, obs.hero, obs, target,
                                    weight, kind, snapshot.version, task.traversal_mode)
            sample.validate()
            coverage.record(sample.state, kind)
            required = generated_bytes + sample_bytes(sample)
            if required > task.sample_byte_budget:
                raise MemoryError(f"Traversal task={task.task_id} requires {required} accounted sample bytes, "
                                  f"budget={task.sample_byte_budget}; no partial sample batch is returned")
            destination.append(sample)
            generated_bytes = required
        return append
    def finish(result):
        result.worker_seconds = time.perf_counter() - started
        result.worker_cpu_seconds = time.process_time() - cpu_started
        result.worker_peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        result.coverage = coverage.as_dict()
        return result
    rng = random.Random(task.seed)
    def checked_traversal(stage: str, run: Callable[[], _TraversalResult]) -> _TraversalResult:
        try:
            return run()
        except TraversalBudgetExceeded as error:
            raise TraversalBudgetExceeded(
                f"task={task.task_id}, snapshot={snapshot.version}, player={task.player}, "
                f"seed={task.seed}, stage={stage}: {error}; no partial samples returned"
            ) from error
    if task.traversal_mode == "outcome_sampling":
        from outcome_sampling import OutcomeSamplingTraversal
        walk = OutcomeSamplingTraversal(strategy, rng, task.max_nodes, task.max_depth, task.epsilon)
        value = checked_traversal("outcome", lambda: walk.run(task.root, task.player, sink("advantage", advantages), sink("strategy", strategies)))
        return finish(GeneratedSamples(task.task_id, snapshot.version, advantages, strategies, walk.nodes, value,
                                task.traversal_mode, _diagnostic_metadata(walk.diagnostics), walk.max_depth_seen))
    walk = Traversal(strategy, rng, task.max_nodes, task.max_depth, observer=observe_neural)
    if snapshot.player_count == 2:
        value = checked_traversal("external", lambda: walk.regrets(
            task.root, task.player, sink("advantage", advantages), strategy_sink=sink("strategy", strategies)))
        return finish(GeneratedSamples(task.task_id, snapshot.version, advantages, strategies,
                                       walk.nodes, value, task.traversal_mode, None, walk.max_depth_seen))
    value = checked_traversal("advantage", lambda: walk.regrets(task.root, task.player, sink("advantage", advantages)))
    nodes = walk.nodes
    max_depth_seen = walk.max_depth_seen
    strategy_sink = sink("strategy", strategies)
    for opponent in root.players:
        if opponent == task.player:
            continue
        walk = Traversal(strategy, rng, task.max_nodes, task.max_depth, observer=observe_neural)
        checked_traversal(f"strategy_enumerate_{opponent}", lambda: walk.average_partial(
            task.root, task.player, opponent, lambda obs, target, weight: strategy_sink(obs, target, weight / 2)))
        nodes += walk.nodes
        max_depth_seen = max(max_depth_seen, walk.max_depth_seen)
    return finish(GeneratedSamples(task.task_id, snapshot.version, advantages, strategies, nodes, value,
                            task.traversal_mode, None, max_depth_seen))


class DeepCFRSolver:
    def __init__(self, players: tuple[int, ...], objective: str = "hand_chip_delta", seed: int = 0,
                 advantage_capacity: int | None = None, strategy_capacity: int = 10000,
                 memory_byte_budget: int = DEFAULT_BYTE_BUDGET, advantage_byte_budget: int | None = None):
        if len(players) not in (2, 3) or len(set(players)) != len(players):
            raise ValueError(f"Expected distinct 2/3 player IDs: {players}")
        if objective != "hand_chip_delta":
            raise ValueError(f"Only hand_chip_delta learning is supported; tournament-winner/ICM objective={objective} is unsupported")
        self.players, self.objective, self.seed = players, objective, seed
        self.version = 0
        self.rng = random.Random(seed)
        # Capacity and budget are totals for the pooled track, not per seat.
        advantage_capacity = len(players) * 10000 if advantage_capacity is None else advantage_capacity
        advantage_byte_budget = len(players) * memory_byte_budget if advantage_byte_budget is None else advantage_byte_budget
        self.advantage_memory = ReservoirMemory(advantage_capacity, seed, "advantage", objective, advantage_byte_budget)
        self.strategy_memory = ReservoirMemory(strategy_capacity, seed + 10, "strategy", objective, memory_byte_budget)
        self.advantage_model: AdvantageNetwork | None = None
        self.average_model: AveragePolicyNetwork | None = None
        self.metrics: list[dict] = []
        self.traversal_mode: str | None = None

    def fork(self) -> DeepCFRSolver:
        """Stage an iteration without duplicating read-only models or replay."""
        result = copy.copy(self)
        result.rng = random.Random()
        result.rng.setstate(self.rng.getstate())
        result.metrics = self.metrics.copy()
        return result

    def snapshot(self) -> ModelSnapshot:
        weights = None if self.advantage_model is None else {
            k: v.detach().cpu().clone() for k, v in self.advantage_model.state_dict().items()}
        return ModelSnapshot(self.version, self.objective, weights, self.version == 0, len(self.players))

    def traversal_tasks(self, root_factory: Callable[[random.Random], GameState],
                        traversals_per_player: int, max_nodes: int, max_depth: int,
                        sample_byte_budget: int, traversal_mode: str = "external_sampling",
                        epsilon: float = 0.6) -> tuple[list[TraversalTask], random.Random]:
        """Prepare reproducible tasks without advancing the solver RNG or fitting."""
        rng = random.Random()
        rng.setstate(self.rng.getstate())
        tasks = []
        for player in self.players:
            for _ in range(traversals_per_player):
                root = hand_root(root_factory(rng))
                if player not in root.players:
                    raise ValueError(f"Traversal player {player} is not seated in current hand: {tuple(root.players)}")
                tasks.append(TraversalTask(len(tasks), player, root, rng.randrange(2**31),
                                           max_nodes, max_depth, sample_byte_budget, traversal_mode, epsilon))
        return tasks, rng

    def run_iteration(self, root_factory: Callable[[random.Random], GameState], traversals_per_player: int = 1,
                      workers: int = 1, advantage_epochs: int = 1, average_epochs: int = 1, batch_size: int = 32,
                      max_nodes: int = 10000, max_depth: int = 300,
                      sample_byte_budget: int = DEFAULT_BYTE_BUDGET,
                      generation_byte_budget: int = 256 * 1024 * 1024,
                      traversal_mode: str = "external_sampling", epsilon: float = 0.6,
                      learning_rate: float = 3e-4, trainer_threads: int = 1, executor: ProcessPoolExecutor | None = None) -> dict:
        from scripts.parallel_cfr import collect_samples
        if traversals_per_player < 1 or workers < 1:
            raise ValueError(f"Invalid traversal count/workers: {traversals_per_player}, {workers}")
        if traversal_mode not in TRAVERSAL_MODES or (self.traversal_mode is not None and self.traversal_mode != traversal_mode):
            raise ValueError(f"Invalid/mixed traversal mode: requested={traversal_mode}, solver={self.traversal_mode}; expected {TRAVERSAL_MODES}")
        if not math.isfinite(epsilon) or not 0 < epsilon <= 1:
            raise ValueError(f"Exploration epsilon must be in (0, 1]: {epsilon}")
        root_started = time.perf_counter()
        tasks, rng = self.traversal_tasks(root_factory, traversals_per_player, max_nodes, max_depth,
                                          sample_byte_budget, traversal_mode, epsilon)
        root_seconds = time.perf_counter() - root_started
        generation_started = time.perf_counter()
        results = collect_samples(self.snapshot(), tasks, workers, aggregate_byte_budget=generation_byte_budget, executor=executor)
        generation_seconds = time.perf_counter() - generation_started
        torch.set_num_threads(trainer_threads)
        fit_started = time.perf_counter()
        # Build the next complete state before changing any published version.
        advantage_memory = copy.deepcopy(self.advantage_memory)
        strategy_memory = copy.deepcopy(self.strategy_memory)
        for result in results:
            if result.version != self.version:
                raise ValueError(f"Stale samples version={result.version}, expected {self.version}")
            if result.traversal_mode != traversal_mode:
                raise ValueError(f"Worker traversal mode={result.traversal_mode}, expected {traversal_mode}")
            for sample in result.advantages:
                if sample.player not in self.players or round(sample.state.numeric[NUMERIC_NAMES.index("player_count")] * 3) != len(self.players):
                    raise ValueError(f"Advantage sample perspective/player count is not in track {self.players}")
                advantage_memory.add(sample)
            for sample in result.strategies:
                if sample.player not in self.players or round(sample.state.numeric[NUMERIC_NAMES.index("player_count")] * 3) != len(self.players):
                    raise ValueError(f"Strategy sample perspective/player count is not in track {self.players}")
                strategy_memory.add(sample)
        metrics = {"version": self.version + 1, "nodes": sum(r.nodes for r in results),
                   "strategy_collector": strategy_collector(len(self.players), traversal_mode),
                   "advantage_samples": sum(len(r.advantages) for r in results),
                   "strategy_samples": sum(len(r.strategies) for r in results),
                   "values": [r.value for r in results], "traversal_mode": traversal_mode,
                   "epsilon": epsilon if traversal_mode == "outcome_sampling" else None,
                   "traversal_diagnostics": [r.diagnostics for r in results]}
        metrics["max_depth_seen"] = max(r.max_depth_seen for r in results)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + (self.version + 1) * 101)
            advantage = AdvantageNetwork()
            metrics["advantage"] = fit(advantage, advantage_memory.samples, advantage_epochs, batch_size,
                                       self.seed, learning_rate=learning_rate)
            torch.manual_seed(self.seed + (self.version + 1) * 101 + 10)
            average = AveragePolicyNetwork()
            metrics["average_policy"] = fit(average, strategy_memory.samples, average_epochs, batch_size, self.seed + 10, learning_rate=learning_rate)
        metrics["fit_epochs"] = {"advantage": advantage_epochs, "average_policy": average_epochs}
        metrics["fit_seconds"] = time.perf_counter() - fit_started
        metrics["root_seconds"] = root_seconds
        metrics["generation_seconds"] = generation_seconds
        metrics["worker_seconds"] = sum(r.worker_seconds for r in results)
        metrics["worker_cpu_seconds"] = sum(r.worker_cpu_seconds for r in results)
        metrics["worker_peak_rss_bytes"] = max(r.worker_peak_rss_bytes for r in results)
        metrics["nodes_per_traversal"] = [r.nodes for r in results]
        metrics["traversals"] = len(tasks)
        from training.metrics import Coverage
        coverage = Coverage()
        for result in results:
            coverage.merge(result.coverage)
        metrics["sample_coverage"] = coverage.as_dict()
        metrics["replay"] = {"advantage": {"seen": advantage_memory.seen, "retained": len(advantage_memory.samples),
                                             "bytes": advantage_memory.used_bytes},
                             "strategy": {"seen": strategy_memory.seen, "retained": len(strategy_memory.samples),
                                            "bytes": strategy_memory.used_bytes}}
        metrics["advantage_samples_by_player"] = {str(p): sum(s.player == p for r in results for s in r.advantages)
                                                  for p in self.players}
        metrics["advantage_retained_by_player"] = {str(p): sum(s.player == p for s in advantage_memory.samples)
                                                   for p in self.players}
        probes = [s.state for s in strategy_memory.samples[:32]]
        if self.average_model is not None:
            with torch.no_grad():
                batch = encode_batch(probes)
                difference = (average(batch) - self.average_model(batch)).abs().sum(1).mean()
            metrics["policy_change_l1_on_replay_probes"] = float(difference)
        else:
            metrics["policy_change_l1_on_replay_probes"] = None
        self.advantage_memory, self.strategy_memory = advantage_memory, strategy_memory
        self.advantage_model, self.average_model = advantage, average
        self.rng.setstate(rng.getstate())
        self.traversal_mode = traversal_mode
        self.version += 1
        self.metrics.append(metrics)
        return metrics

    def export_average(self, path: str | Path) -> None:
        if self.average_model is None:
            raise ValueError("Average-policy network is unavailable before successful training")
        torch.save({"version": 5, "feature_schema_version": FEATURE_SCHEMA_VERSION, "state_version": STATE_VERSION, "architecture": MODEL_ARCHITECTURE,
                    "actions": list(ACTION_IDS), "numeric_names": list(NUMERIC_NAMES),
                    "objective": self.objective, "iteration": self.version,
                    "training_metadata": {"advantage_layout": "shared_per_player_count",
                                          "strategy_collector": strategy_collector(len(self.players), self.traversal_mode),
                                          "seat_normalization": "hero_then_clockwise_positions",
                                          "traversal_mode": self.traversal_mode,
                                          "utility_units": "initial_big_blind_chips",
                                          "history_scope": "current_hand", "payout_scope": "winner_take_all",
                                          "epsilon_by_iteration": [metric["epsilon"] for metric in self.metrics],
                                          "regret_iteration_weighting": "linear",
                                          "average_policy_iteration_weighting": "linear"},
                    "supported_player_counts": sorted({round(s.state.numeric[NUMERIC_NAMES.index("player_count")] * 3)
                                                        for s in self.strategy_memory.samples}),
                    "weights": self.average_model.state_dict(), "metrics": self.metrics}, path)


class NeuralAveragePolicy:
    def __init__(self, path: str | Path):
        raw = torch.load(path, map_location="cpu", weights_only=True)
        feature_version = raw.get("feature_schema_version")
        if (raw.get("version") != 5 or feature_version not in SUPPORTED_FEATURE_VERSIONS or raw.get("state_version") != STATE_VERSION
                or raw.get("architecture") != model_architecture(feature_version) or raw.get("actions") != list(ACTION_IDS)
                or raw.get("numeric_names") != list(numeric_names(feature_version)) or raw.get("iteration", 0) < 1
                or raw.get("objective") != "hand_chip_delta"
                or raw.get("training_metadata", {}).get("traversal_mode") not in TRAVERSAL_MODES
                or not raw.get("supported_player_counts")
                or any(n not in (2, 3) for n in raw["supported_player_counts"])):
            raise ValueError(f"Incompatible average-policy checkpoint: {path}")
        self.objective = raw["objective"]
        self.iteration = raw["iteration"]
        self.traversal_mode = raw["training_metadata"]["traversal_mode"]
        self.supported_player_counts = tuple(raw["supported_player_counts"])
        self.model = AveragePolicyNetwork(feature_version)
        self.model.load_state_dict(raw["weights"], strict=True)
        self.model.eval()

    def query(self, obs: Observation) -> tuple[float, ...]:
        if obs.objective != self.objective:
            raise ValueError(f"Average model objective={self.objective}, state objective={obs.objective}")
        count = round(obs.numeric[NUMERIC_NAMES.index("player_count")] * 3)
        if count not in self.supported_player_counts:
            raise KeyError(f"Average model has no training coverage for {count} players; supported={self.supported_player_counts}")
        return self.model.probabilities(obs)
