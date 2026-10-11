"""Frozen outer iterations, bounded sample generation and central Deep CFR training."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import asdict, dataclass, field, replace
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
from cfr_solver import GameState, Traversal, TraversalBudgetExceeded, hand_root, regret_matching
from infoset import STATE_VERSION, Observation, observe_fields
from features import FEATURE_SCHEMA_VERSION, SUPPORTED_FEATURE_VERSIONS
from features.neural import NeuralObservation, numeric_names, neural_observation
from features.neural import NEURAL_NUMERIC_NAMES as NUMERIC_NAMES
from ml.memory import DEFAULT_BYTE_BUDGET, TRAVERSAL_MODES, ReservoirMemory, TrainingSample, sample_bytes
from ml.model import MODEL_ARCHITECTURE, AdvantageNetwork, AveragePolicyNetwork, encode_batch, model_architecture
from ml.train import fit
from ml.model import StreetNetworks, STREET_LAYOUT, STREETS

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
    layout: str = "shared_per_player_count"
    collector_enumerate_from_street: int | None = None

    def validate(self) -> None:
        if self.layout not in ("shared_per_player_count", STREET_LAYOUT):
            raise ValueError(f"Unsupported snapshot layout: {self.layout}")
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
            self.model = StreetNetworks("advantage") if snapshot.layout == STREET_LAYOUT else AdvantageNetwork()
            self.model.load_state_dict(snapshot.advantage_weights, strict=True)
            self.model.eval()

    def __call__(self, obs: Observation | NeuralObservation) -> tuple[float, ...]:
        count = round(obs.numeric[NUMERIC_NAMES.index("player_count")] * 3)
        if count != self.snapshot.player_count:
            raise ValueError(f"Frozen strategy track mismatch: expected {self.snapshot.player_count}, received {count}")
        if obs.objective != self.snapshot.objective:
            raise ValueError(f"Snapshot objective {self.snapshot.objective} != root {obs.objective}")
        if self.snapshot.uniform_initial:
            return regret_matching([0.0] * len(ACTION_IDS), obs.legal_mask)
        if obs.street not in range(4):
            raise ValueError(f"Invalid public street: {obs.street}")
        model = self.model.specialists[STREETS[obs.street]] if isinstance(self.model, StreetNetworks) else self.model
        with torch.no_grad():
            values = tuple(model(encode_batch([obs]))[0].tolist())
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
    import hashlib
    import json
    from features.cards import canonical_suits
    seats = list(root.players)
    offset = seats.index(root.button)
    seats = seats[offset:] + seats[:offset]
    identity = {seat: index for index, seat in enumerate(seats)}
    public_root = {"board": canonical_suits(tuple(root.board))[0], "street": root.street,
                   "history": [(identity[e.player_id], e.action, e.amount_to) for e in root.history],
                   "stacks": [root.initial_stacks[p] for p in seats],
                   "blinds": (root.blinds.small, root.blinds.big)}
    root_group = hashlib.sha256(json.dumps(public_root, sort_keys=True).encode()).hexdigest()
    trajectory_id = f"{snapshot.player_count}:{snapshot.version}:{task.task_id}:{task.seed}"
    advantages, strategies = [], []
    generated_bytes = 0
    from training.metrics import Coverage
    coverage = Coverage()
    raw_cards: dict[int, tuple[int, int]] = {}
    def recording_observer(state):
        fields = observe_fields(state)
        obs = neural_observation(fields)
        raw_cards[id(obs)] = tuple(fields.cards[:2])
        return obs
    def sink(kind, destination):
        def append(obs, target, weight):
            nonlocal generated_bytes
            sample = TrainingSample(snapshot.version + 1, obs.hero, obs, target,
                                    weight, kind, snapshot.version, task.traversal_mode, trajectory_id, root_group)
            sample.validate()
            exact_cards = tuple(obs.cards[:2]) if isinstance(obs, Observation) else raw_cards.get(id(obs))
            coverage.record(sample.state, kind, exact_cards=exact_cards, iteration=sample.iteration, weight=sample.weight)
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
    walk = Traversal(strategy, rng, task.max_nodes, task.max_depth, observer=recording_observer)
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
        walk = Traversal(strategy, rng, task.max_nodes, task.max_depth, observer=recording_observer)
        checked_traversal(f"strategy_enumerate_{opponent}", lambda: walk.average_partial(
            task.root, task.player, opponent, lambda obs, target, weight: strategy_sink(obs, target, weight / 2),
            **({"enumerate_from_street": snapshot.collector_enumerate_from_street}
               if snapshot.collector_enumerate_from_street is not None else {})))
        nodes += walk.nodes
        max_depth_seen = max(max_depth_seen, walk.max_depth_seen)
    return finish(GeneratedSamples(task.task_id, snapshot.version, advantages, strategies, nodes, value,
                            task.traversal_mode, None, max_depth_seen))


class DeepCFRSolver:
    def __init__(self, players: tuple[int, ...], objective: str = "hand_chip_delta", seed: int = 0,
                 advantage_capacity: int | None = None, strategy_capacity: int = 10000,
                 memory_byte_budget: int = DEFAULT_BYTE_BUDGET, advantage_byte_budget: int | None = None, replay_opening_fraction: float | None = None, street_networks: dict | None = None):
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
        if replay_opening_fraction is not None:
            from ml.stratified_memory import ProtectedReplay
            if not 0 < replay_opening_fraction < 1 or min(advantage_capacity, strategy_capacity) < 2:
                raise ValueError("Protected replay requires a fraction in (0, 1) and at least two slots")
            def protected(capacity, budget, kind, memory_seed):
                opening = max(1, min(capacity - 1, int(capacity * replay_opening_fraction)))
                return ProtectedReplay(opening, capacity - opening, kind=kind, seed=memory_seed, byte_budget=budget)
            self.advantage_memory = protected(advantage_capacity, advantage_byte_budget, "advantage", seed)
            self.strategy_memory = protected(strategy_capacity, memory_byte_budget, "strategy", seed + 10)
        self.advantage_model: AdvantageNetwork | None = None
        self.average_model: AveragePolicyNetwork | None = None
        self.metrics: list[dict] = []
        self.traversal_mode: str | None = None
        self.street_config = copy.deepcopy(street_networks)
        self.specialist_states = {"advantage": {}, "strategy": {}}
        if street_networks is not None:
            from ml.street_training import validate_street_config
            from ml.street_replay import StreetReplay
            validate_street_config(street_networks)
            if replay_opening_fraction is not None:
                raise ValueError("Street replay requires its explicit allocation, not legacy opening protection")
            for kind, capacity, budget in (("advantage", advantage_capacity, advantage_byte_budget),
                                           ("strategy", strategy_capacity, memory_byte_budget)):
                entries = street_networks[kind]
                if sum(v["capacity"] for v in entries.values()) != capacity or sum(v["byte_budget"] for v in entries.values()) != budget:
                    raise ValueError(f"Street {kind} allocations must match track total capacity and byte budget")
                memory = StreetReplay({k: v["capacity"] for k,v in entries.items()},
                                      {k: v["byte_budget"] for k,v in entries.items()}, kind,
                                      seed + (10 if kind == "strategy" else 0))
                setattr(self, "advantage_memory" if kind == "advantage" else "strategy_memory", memory)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.advantage_model = StreetNetworks("advantage").eval()
                self.average_model = StreetNetworks("strategy").eval()

    def fork(self) -> DeepCFRSolver:
        """Stage an iteration without duplicating read-only models or replay."""
        result = copy.copy(self)
        result.rng = random.Random()
        result.rng.setstate(self.rng.getstate())
        result.metrics = self.metrics.copy()
        return result

    def snapshot(self) -> ModelSnapshot:
        weights = None if self.advantage_model is None or self.version == 0 else {
            k: v.detach().cpu().clone() for k, v in self.advantage_model.state_dict().items()}
        return ModelSnapshot(self.version, self.objective, weights, self.version == 0, len(self.players),
                             STREET_LAYOUT if self.street_config else "shared_per_player_count",
                             self.street_config.get("collector_enumerate_from_street") if self.street_config else None)

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
                      learning_rate: float = 3e-4, trainer_threads: int = 1, executor: ProcessPoolExecutor | None = None,
                      fit_schedule: dict | None = None, generation_batch_size: int | None = None) -> dict:
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
        # Replay updates remain private until the complete generation and fitting succeeds.
        advantage_memory = copy.deepcopy(self.advantage_memory)
        strategy_memory = copy.deepcopy(self.strategy_memory)
        chunk_size = len(tasks) if generation_batch_size is None else generation_batch_size
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("generation_batch_size must be a positive integer")
        snapshot = self.snapshot()
        results = []
        advantage_count = strategy_count = 0
        player_counts = {str(p): 0 for p in self.players}
        generation_started = time.perf_counter()
        for offset in range(0, len(tasks), chunk_size):
            chunk = collect_samples(snapshot, tasks[offset:offset + chunk_size], workers,
                                    aggregate_byte_budget=generation_byte_budget, executor=executor)
            for result in chunk:
                if result.version != self.version:
                    raise ValueError(f"Stale samples version={result.version}, expected {self.version}")
                if result.traversal_mode != traversal_mode:
                    raise ValueError(f"Worker traversal mode={result.traversal_mode}, expected {traversal_mode}")
                advantage_count += len(result.advantages)
                strategy_count += len(result.strategies)
                for memory, samples in ((advantage_memory, result.advantages), (strategy_memory, result.strategies)):
                    for sample in samples:
                        if sample.player not in self.players or round(sample.state.numeric[NUMERIC_NAMES.index("player_count")] * 3) != len(self.players):
                            raise ValueError(f"Sample perspective/player count is not in track {self.players}")
                        memory.add(sample)
                        if sample.kind == 'advantage':
                            player_counts[str(sample.player)] += 1
                results.append(replace(result, advantages=[], strategies=[]))
            del chunk
        generation_seconds = time.perf_counter() - generation_started
        torch.set_num_threads(trainer_threads)
        fit_started = time.perf_counter()
        metrics = {"version": self.version + 1, "nodes": sum(r.nodes for r in results),
                   "strategy_collector": strategy_collector(len(self.players), traversal_mode),
                   "advantage_samples": advantage_count,
                   "strategy_samples": strategy_count,
                   "values": [r.value for r in results], "traversal_mode": traversal_mode,
                   "epsilon": epsilon if traversal_mode == "outcome_sampling" else None,
                   "traversal_diagnostics": [r.diagnostics for r in results]}
        metrics["collector_enumerate_from_street"] = snapshot.collector_enumerate_from_street
        metrics["max_depth_seen"] = max(r.max_depth_seen for r in results)
        advantage_samples = advantage_memory.samples
        strategy_samples = strategy_memory.samples
        schedule = fit_schedule or {}
        fit_average = self.average_model is None or (self.version + 1) % schedule.get('average_every', 1) == 0
        cache_encoding = schedule.get('cache_encoding', False)
        metrics['fit_initialization'] = {}
        metrics['fit_component_seconds'] = {}
        specialist_states = self.specialist_states
        if self.street_config is not None:
            from ml.street_training import fit_specialists
            advantage, a_states, a_metrics = fit_specialists(
                self.advantage_model, advantage_memory, self.specialist_states["advantage"],
                self.street_config["advantage"], version=self.version + 1, seed=self.seed,
                batch_size=batch_size, learning_rate=learning_rate, cache_encoding=cache_encoding)
            average, b_states, b_metrics = fit_specialists(
                self.average_model, strategy_memory, self.specialist_states["strategy"],
                self.street_config["strategy"], version=self.version + 1, seed=self.seed,
                batch_size=batch_size, learning_rate=learning_rate, cache_encoding=cache_encoding)
            specialist_states = {"advantage": a_states, "strategy": b_states}
            metrics["specialists"] = {"advantage": a_metrics, "strategy": b_metrics}
            for kind, rows in (("advantage", a_metrics), ("average_policy", b_metrics)):
                metrics[kind] = {"updates_completed": sum(r.get("updates_completed", 0) for r in rows.values()),
                                 "skipped": all(r.get("skipped", False) for r in rows.values())}
                metrics["fit_component_seconds"][kind] = sum(r.get("seconds", 0) for r in rows.values())
                metrics["fit_initialization"][kind] = "persistent_weights_and_adam"
            metrics["average_model_iteration"] = max(s["model_version"] for s in b_states.values())
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed + (self.version + 1) * 101)
                warm_a = schedule.get('advantage_initialization', 'fresh') == 'warm' and self.advantage_model is not None
                advantage = copy.deepcopy(self.advantage_model) if warm_a else AdvantageNetwork()
                component_started = time.perf_counter()
                metrics["advantage"] = fit(advantage, advantage_samples, advantage_epochs, batch_size,
                                           self.seed, learning_rate=learning_rate,
                                           max_updates=schedule.get('advantage_max_updates'), cache_encoding=cache_encoding)
                metrics['fit_component_seconds']['advantage'] = time.perf_counter() - component_started
                metrics['fit_initialization']['advantage'] = 'warm_weights_fresh_optimizer' if warm_a else 'fresh'
                average = self.average_model
                if fit_average:
                    torch.manual_seed(self.seed + (self.version + 1) * 101 + 10)
                    warm_b = schedule.get('average_initialization', 'fresh') == 'warm' and average is not None
                    average = copy.deepcopy(average) if warm_b else AveragePolicyNetwork()
                    component_started = time.perf_counter()
                    metrics["average_policy"] = fit(average, strategy_samples, average_epochs, batch_size, self.seed + 10,
                        learning_rate=learning_rate, max_updates=schedule.get('average_max_updates'), cache_encoding=cache_encoding)
                    metrics['fit_component_seconds']['average_policy'] = time.perf_counter() - component_started
                    metrics['fit_initialization']['average_policy'] = 'warm_weights_fresh_optimizer' if warm_b else 'fresh'
                else:
                    metrics['average_policy'] = {'skipped': True}
                    metrics['fit_component_seconds']['average_policy'] = 0.0
                    metrics['fit_initialization']['average_policy'] = 'unchanged'
            metrics['average_model_iteration'] = self.version + 1 if fit_average else self.metrics[-1].get('average_model_iteration', self.version)
        if self.street_config is None:
            metrics["fit_epochs"] = {"advantage": advantage_epochs, "average_policy": average_epochs if fit_average else 0}
        else:
            metrics["fit_update_caps"] = {kind: {street: settings["max_updates"] for street, settings in self.street_config[kind].items()}
                                          for kind in ("advantage", "strategy")}
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
        retained = Coverage()
        for samples in (advantage_samples, strategy_samples):
            for sample in samples:
                # Compact replay has canonical cards; label this identity
                # explicitly instead of claiming physical-suit provenance.
                retained.record(sample.state, sample.kind, iteration=sample.iteration, weight=sample.weight)
        metrics["retained_coverage"] = retained.as_dict()
        metrics["replay"] = {"advantage": {"seen": advantage_memory.seen, "retained": len(advantage_samples),
                                             "bytes": advantage_memory.used_bytes},
                             "strategy": {"seen": strategy_memory.seen, "retained": len(strategy_samples),
                                            "bytes": strategy_memory.used_bytes}}
        metrics["advantage_samples_by_player"] = player_counts
        for kind, memory in (("advantage", advantage_memory), ("strategy", strategy_memory)):
            if hasattr(memory, "diagnostics"):
                metrics["replay"][kind]["strata"] = memory.diagnostics()
        metrics["advantage_retained_by_player"] = {str(p): sum(s.player == p for s in advantage_samples)
                                                   for p in self.players}
        probes = [s.state for s in strategy_samples[:32]]
        if self.average_model is not None:
            with torch.no_grad():
                batch = encode_batch(probes)
                difference = (average(batch) - self.average_model(batch)).abs().sum(1).mean()
            metrics["policy_change_l1_on_replay_probes"] = float(difference)
        else:
            metrics["policy_change_l1_on_replay_probes"] = None
        self.specialist_states = specialist_states
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
        if self.street_config is not None:
            if not bool(self.average_model.trained.all()):
                missing = [s for i,s in enumerate(STREETS) if not self.average_model.trained[i]]
                raise ValueError(f"Cannot export cold Average specialists: {missing}")
            torch.save(street_average_payload(self.average_model.state_dict(), self.version,
                       len(self.players), self.traversal_mode, self.specialist_states["strategy"]), path)
            return
        torch.save(average_policy_payload(
            weights=self.average_model.state_dict(), iteration=self.version,
            objective=self.objective, player_count=len(self.players),
            traversal_mode=self.traversal_mode, metrics=self.metrics,
            supported_player_counts=sorted({round(s.state.numeric[NUMERIC_NAMES.index("player_count")] * 3)
                                            for s in self.strategy_memory.samples})), path)


def average_policy_payload(*, weights: dict, iteration: int, objective: str,
                           player_count: int, traversal_mode: str, metrics: list,
                           supported_player_counts: list[int]) -> dict:
    """Shared inference artifact contract for live solvers and checkpoint extraction."""
    return {"version": 5, "feature_schema_version": FEATURE_SCHEMA_VERSION, "state_version": STATE_VERSION, "architecture": MODEL_ARCHITECTURE,
                    "actions": list(ACTION_IDS), "numeric_names": list(NUMERIC_NAMES),
                    "objective": objective, "iteration": iteration,
                    "training_metadata": {"advantage_layout": "shared_per_player_count",
                                          "average_model_iteration": metrics[-1].get('average_model_iteration', iteration),
                                          "strategy_collector": strategy_collector(player_count, traversal_mode),
                                          "seat_normalization": "hero_then_clockwise_positions",
                                          "traversal_mode": traversal_mode,
                                          "utility_units": "initial_big_blind_chips",
                                          "history_scope": "current_hand", "payout_scope": "winner_take_all",
                                          "epsilon_by_iteration": [metric["epsilon"] for metric in metrics],
                                          "epsilon_start_iteration": metrics[0]["version"],
                                          "regret_iteration_weighting": "linear",
                                          "average_policy_iteration_weighting": "linear"},
                    "supported_player_counts": supported_player_counts,
                    "weights": weights}


def street_average_payload(weights: dict, iteration: int, player_count: int,
                           traversal_mode: str, states: dict) -> dict:
    return {"version": 6, "layout": STREET_LAYOUT, "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "state_version": STATE_VERSION, "architecture": MODEL_ARCHITECTURE,
            "actions": list(ACTION_IDS), "numeric_names": list(NUMERIC_NAMES),
            "objective": "hand_chip_delta", "iteration": iteration,
            "training_metadata": {"traversal_mode": traversal_mode},
            "supported_player_counts": [player_count], "weights": weights,
            "specialists": {name: {key: value for key,value in state.items() if key != "optimizer"}
                            for name,state in states.items()}}


class NeuralAveragePolicy:
    def __init__(self, path: str | Path):
        raw = torch.load(path, map_location="cpu", weights_only=True)
        feature_version = raw.get("feature_schema_version")
        if (raw.get("version") not in (5, 6) or feature_version not in SUPPORTED_FEATURE_VERSIONS or raw.get("state_version") != STATE_VERSION
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
        self.specialist_metadata = raw.get("specialists")
        if raw["version"] == 6:
            if raw.get("layout") != STREET_LAYOUT or set(raw.get("specialists", {})) != set(STREETS):
                raise ValueError("Invalid street policy layout/metadata")
            self.model = StreetNetworks("strategy", feature_version)
        else:
            self.model = AveragePolicyNetwork(feature_version)
        self.model.load_state_dict(raw["weights"], strict=True)
        if raw["version"] == 6 and not bool(self.model.trained.all()):
            raise ValueError("Published street policy contains untrained specialists")
        self.model.eval()

    def query(self, obs: Observation) -> tuple[float, ...]:
        if obs.objective != self.objective:
            raise ValueError(f"Average model objective={self.objective}, state objective={obs.objective}")
        count = round(obs.numeric[NUMERIC_NAMES.index("player_count")] * 3)
        if count not in self.supported_player_counts:
            raise KeyError(f"Average model has no training coverage for {count} players; supported={self.supported_player_counts}")
        return self.model.probabilities(obs)
