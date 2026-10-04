"""Frozen outer iterations, bounded sample generation and central Deep CFR training."""
from __future__ import annotations
import copy
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Callable
import torch
from actions import ACTION_IDS
from cfr_solver import GameState, Traversal, regret_matching, validate_strategy
from infoset import NUMERIC_NAMES, STATE_VERSION, Observation
from ml.memory import DEFAULT_BYTE_BUDGET, ReservoirMemory, TrainingSample, sample_bytes
from ml.model import MODEL_ARCHITECTURE, AdvantageNetwork, AveragePolicyNetwork, encode_batch
from ml.train import fit


@dataclass(frozen=True)
class ModelSnapshot:
    version: int
    objective: str
    advantage_weights: dict[int, dict[str, torch.Tensor]]
    uniform_initial: bool


@dataclass(frozen=True)
class TraversalTask:
    task_id: int
    player: int
    root: GameState
    seed: int
    max_nodes: int
    max_depth: int
    sample_byte_budget: int = DEFAULT_BYTE_BUDGET


@dataclass
class GeneratedSamples:
    task_id: int
    version: int
    advantages: list[TrainingSample]
    strategies: list[TrainingSample]
    nodes: int
    value: float


def generate_samples(snapshot: ModelSnapshot, task: TraversalTask) -> GeneratedSamples:
    if not isinstance(task.sample_byte_budget, int) or task.sample_byte_budget < 1:
        raise ValueError(f"Traversal sample byte budget must be positive: {task.sample_byte_budget}")
    torch.set_num_threads(1)
    models = {}
    if snapshot.uniform_initial != (snapshot.version == 0):
        raise ValueError(f"Invalid snapshot initial/version contract: {snapshot.version}")
    if not snapshot.uniform_initial:
        for player, weights in snapshot.advantage_weights.items():
            model = AdvantageNetwork()
            model.load_state_dict(weights, strict=True)
            model.eval()
            models[player] = model
    def strategy(obs):
        if obs.objective != snapshot.objective:
            raise ValueError(f"Snapshot objective {snapshot.objective} != root {obs.objective}")
        if snapshot.uniform_initial:
            return regret_matching([0.0] * len(ACTION_IDS), obs.legal_mask)
        if obs.hero not in models:
            raise ValueError(f"Missing advantage model for player {obs.hero}, version={snapshot.version}")
        with torch.no_grad():
            values = tuple(models[obs.hero](encode_batch([obs]))[0].tolist())
        return regret_matching(values, obs.legal_mask)
    advantages, strategies = [], []
    generated_bytes = 0
    def sink(kind, destination):
        def append(obs, target, weight):
            nonlocal generated_bytes
            sample = TrainingSample(snapshot.version + 1, obs.hero, obs, target,
                                    weight, kind, snapshot.version)
            sample.validate()
            required = generated_bytes + sample_bytes(sample)
            if required > task.sample_byte_budget:
                raise MemoryError(f"Traversal task={task.task_id} requires {required} accounted sample bytes, "
                                  f"budget={task.sample_byte_budget}; no partial sample batch is returned")
            destination.append(sample)
            generated_bytes = required
        return append
    rng = random.Random(task.seed)
    walk = Traversal(strategy, rng, task.max_nodes, task.max_depth)
    value = walk.regrets(task.root.clone(), task.player, sink("advantage", advantages))
    nodes = walk.nodes
    walk = Traversal(strategy, rng, task.max_nodes, task.max_depth)
    walk.average(task.root.clone(), task.player, sink("strategy", strategies))
    return GeneratedSamples(task.task_id, snapshot.version, advantages, strategies, nodes + walk.nodes, value)


class DeepCFRSolver:
    def __init__(self, players: tuple[int, ...], objective: str = "tournament_winner", seed: int = 0,
                 advantage_capacity: int = 10000, strategy_capacity: int = 10000,
                 memory_byte_budget: int = DEFAULT_BYTE_BUDGET):
        if len(players) not in (2, 3) or len(set(players)) != len(players):
            raise ValueError(f"Expected distinct 2/3 player IDs: {players}")
        self.players, self.objective, self.seed = players, objective, seed
        self.version = 0
        self.rng = random.Random(seed)
        self.advantage_memory = {p: ReservoirMemory(advantage_capacity, seed + i, "advantage", objective, memory_byte_budget)
                                 for i, p in enumerate(players)}
        self.strategy_memory = ReservoirMemory(strategy_capacity, seed + 10, "strategy", objective, memory_byte_budget)
        self.advantage_models: dict[int, AdvantageNetwork] = {}
        self.average_model: AveragePolicyNetwork | None = None
        self.metrics: list[dict] = []

    def snapshot(self) -> ModelSnapshot:
        return ModelSnapshot(self.version, self.objective,
                             {p: {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                              for p, m in self.advantage_models.items()}, self.version == 0)

    def run_iteration(self, root_factory: Callable[[random.Random], GameState], traversals_per_player: int = 1,
                      workers: int = 1, epochs: int = 1, batch_size: int = 32,
                      max_nodes: int = 10000, max_depth: int = 300,
                      sample_byte_budget: int = DEFAULT_BYTE_BUDGET,
                      generation_byte_budget: int = 256 * 1024 * 1024) -> dict:
        from scripts.parallel_cfr import collect_samples
        if traversals_per_player < 1 or workers < 1:
            raise ValueError(f"Invalid traversal count/workers: {traversals_per_player}, {workers}")
        rng = random.Random()
        rng.setstate(self.rng.getstate())
        tasks = []
        for player in self.players:
            for _ in range(traversals_per_player):
                tasks.append(TraversalTask(len(tasks), player, root_factory(rng), rng.randrange(2**31),
                                           max_nodes, max_depth, sample_byte_budget))
        results = collect_samples(self.snapshot(), tasks, workers, aggregate_byte_budget=generation_byte_budget)
        # Build the next complete state before changing any published version.
        memories = copy.deepcopy(self.advantage_memory)
        strategy_memory = copy.deepcopy(self.strategy_memory)
        for result in results:
            if result.version != self.version:
                raise ValueError(f"Stale samples version={result.version}, expected {self.version}")
            for sample in result.advantages:
                memories[sample.player].add(sample)
            for sample in result.strategies:
                strategy_memory.add(sample)
        models = {}
        metrics = {"version": self.version + 1, "nodes": sum(r.nodes for r in results),
                   "advantage_samples": sum(len(r.advantages) for r in results),
                   "strategy_samples": sum(len(r.strategies) for r in results),
                   "values": [r.value for r in results]}
        with torch.random.fork_rng(devices=[]):
            for player in self.players:
                torch.manual_seed(self.seed + (self.version + 1) * 101 + player)
                models[player] = AdvantageNetwork()
                metrics[f"advantage_{player}"] = fit(models[player], memories[player].samples, epochs, batch_size,
                                                      self.seed + player)
            torch.manual_seed(self.seed + (self.version + 1) * 101 + 10)
            average = AveragePolicyNetwork()
            metrics["average_policy"] = fit(average, strategy_memory.samples, epochs, batch_size, self.seed + 10)
        probes = [s.state for s in strategy_memory.samples[:32]]
        if self.average_model is not None:
            with torch.no_grad():
                batch = encode_batch(probes)
                difference = (average(batch) - self.average_model(batch)).abs().sum(1).mean()
            metrics["policy_change_l1_on_replay_probes"] = float(difference)
        else:
            metrics["policy_change_l1_on_replay_probes"] = None
        self.advantage_memory, self.strategy_memory = memories, strategy_memory
        self.advantage_models, self.average_model = models, average
        self.rng.setstate(rng.getstate())
        self.version += 1
        self.metrics.append(metrics)
        return metrics

    def export_average(self, path: str | Path) -> None:
        if self.average_model is None:
            raise ValueError("Average-policy network is unavailable before successful training")
        torch.save({"version": 2, "state_version": STATE_VERSION, "architecture": MODEL_ARCHITECTURE,
                    "actions": list(ACTION_IDS), "numeric_names": list(NUMERIC_NAMES),
                    "objective": self.objective, "iteration": self.version,
                    "supported_player_counts": sorted({round(s.state.numeric[NUMERIC_NAMES.index("player_count")] * 3)
                                                        for s in self.strategy_memory.samples}),
                    "weights": self.average_model.state_dict(), "metrics": self.metrics}, path)


class NeuralAveragePolicy:
    def __init__(self, path: str | Path):
        raw = torch.load(path, map_location="cpu", weights_only=True)
        if (raw.get("version") != 2 or raw.get("state_version") != STATE_VERSION
                or raw.get("architecture") != MODEL_ARCHITECTURE or raw.get("actions") != list(ACTION_IDS)
                or raw.get("numeric_names") != list(NUMERIC_NAMES) or raw.get("iteration", 0) < 1
                or raw.get("objective") not in ("tournament_winner", "hand_chip_delta")
                or not raw.get("supported_player_counts")
                or any(n not in (2, 3) for n in raw["supported_player_counts"])):
            raise ValueError(f"Incompatible average-policy checkpoint: {path}")
        self.objective = raw["objective"]
        self.iteration = raw["iteration"]
        self.supported_player_counts = tuple(raw["supported_player_counts"])
        self.model = AveragePolicyNetwork()
        self.model.load_state_dict(raw["weights"], strict=True)
        self.model.eval()

    def query(self, obs: Observation) -> tuple[float, ...]:
        if obs.objective != self.objective:
            raise ValueError(f"Average model objective={self.objective}, state objective={obs.objective}")
        count = round(obs.numeric[NUMERIC_NAMES.index("player_count")] * 3)
        if count not in self.supported_player_counts:
            raise KeyError(f"Average model has no training coverage for {count} players; supported={self.supported_player_counts}")
        with torch.no_grad():
            probabilities = tuple(self.model(encode_batch([obs]))[0].tolist())
        validate_strategy(probabilities, obs.legal_mask)
        return probabilities
