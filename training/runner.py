"""Transactional two-track training, deterministic resume, explicit artifacts and evaluation."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from training.runtime_storage import RuntimeStorage, archive_metric, metric_count, rotate_checkpoints
import copy
import json
import hashlib
from pathlib import Path
import random
import subprocess
import time
import torch
from ml.deep_cfr import DeepCFRSolver, FrozenStrategy
from ml.model import StreetNetworks
from training.checkpoint import checkpoint_contract
from infoset import NUMERIC_NAMES
from ml.model import AdvantageNetwork, AveragePolicyNetwork
from training.checkpoint import (CONTRACT, atomic_bytes, pack_memory, unpack_memory, pack_tournament,
                                 unpack_tournament, write_checkpoint, read_checkpoint)
from training.config import validate_config, config_hash
from training.evaluation import fixed_roots, fixed_drift, evaluate_solver, model_policy
from training.root_sampler import RootSampler


def deployment_metadata(repository: Path) -> dict:
    """Validate source provenance when running a deployment archive without Git."""
    manifest = repository / 'deployment-source.json'
    if not manifest.is_file():
        raise FileNotFoundError(f'Archive requires deployment provenance: {manifest}')
    raw = json.loads(manifest.read_text())
    if raw.get('version') != 1 or not isinstance(raw.get('files'), list) or not isinstance(raw.get('metadata'), dict):
        raise ValueError(f'Invalid deployment provenance: {manifest}')
    metadata = raw['metadata']
    if (set(metadata) != {'source_git_commit', 'source_git_dirty', 'source_code_sha256'}
            or not isinstance(metadata['source_git_commit'], str)
            or len(metadata['source_git_commit']) != 40
            or type(metadata['source_git_dirty']) is not bool):
        raise ValueError(f'Invalid source metadata: {manifest}')
    digest = hashlib.sha256()
    for name in raw['files']:
        if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError(f'Invalid source path in {manifest}')
        path = repository / name
        path.resolve().relative_to(repository.resolve())
        digest.update(name.encode())
        digest.update(path.read_bytes())
    if digest.hexdigest() != metadata['source_code_sha256']:
        raise ValueError(f'Deployment source checksum mismatch: {manifest}')
    return metadata


def write_source_manifest(repository: Path, destination: Path) -> None:
    """Seal the source metadata before packaging a tested checkout for deployment."""
    metadata = source_metadata(repository)
    listing = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=repository)
    files = [name for name in sorted(set(listing.decode().split('\0')) - {''})
             if (Path(name).suffix in ('.py', '.ts', '.tsx') or name.startswith('configs/'))]
    destination.write_text(json.dumps({'version': 1, 'metadata': metadata, 'files': files}, indent=2) + '\n')


def source_commit(repository: Path | None = None) -> str | None:
    repository = repository or Path(__file__).resolve().parents[1]
    if not (repository / '.git').exists():
        return deployment_metadata(repository)['source_git_commit']
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def source_metadata(repository: Path | None = None) -> dict:
    repository = repository or Path(__file__).resolve().parents[1]
    if not (repository / '.git').exists():
        return deployment_metadata(repository)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repository, capture_output=True, text=True, check=True)
    listing = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                             cwd=repository, capture_output=True, check=True)
    digest = hashlib.sha256()
    for name in sorted(set(listing.stdout.decode().split("\0")) - {""}):
        path = repository / name
        if path.suffix not in (".py", ".ts", ".tsx") and not name.startswith("configs/"):
            continue
        digest.update(name.encode())
        digest.update(path.read_bytes() if path.exists() else b"deleted")
    return {"source_git_commit": source_commit(repository), "source_git_dirty": bool(status.stdout),
            "source_code_sha256": digest.hexdigest()}


class TrainingRunner:
    def __init__(self, config: dict) -> None:
        self.config = validate_config(config)
        self.iteration = 0
        self.resume_checkpoint: Path | None = None
        self.resume_budget_change: dict | None = None
        self.resume_execution_change: dict | None = None
        self.solvers = {}
        self.samplers = {}
        self.metrics: list[dict] = []
        self.runtime_storage: RuntimeStorage | None = None
        self.metadata = {**source_metadata(), "network_fit": "fresh_from_reservoir_each_iteration",
                         "device": "cpu", "utility_units": "initial_big_blind_chips"}
        if 'fit_schedule' in self.config:
            self.metadata['network_fit'] = dict(self.config['fit_schedule'])
        if 'street_networks' in self.config:
            self.metadata['network_fit'] = 'independent_streets_persistent_weights_and_adam'
            self.metadata['specialist_schedules'] = self.config['street_networks']
        for name, count in (("3max", 3), ("hu", 2)):
            if config[name]["enabled"]:
                self.solvers[name] = DeepCFRSolver(tuple(range(count)), seed=config["seed"] + count,
                    advantage_capacity=config[name]["advantage_capacity"], strategy_capacity=config["strategy_capacity"],
                    memory_byte_budget=config["memory_byte_budget"], advantage_byte_budget=config[name]["advantage_byte_budget"],
                    replay_opening_fraction=config.get("replay_opening_fraction"), street_networks=config.get("street_networks"))
                self.samplers[name] = RootSampler(count, config["seed"] + count * 1000, config[name]["root_sampling"])
        self.probes = {name: fixed_roots(len(solver.players)) for name, solver in self.solvers.items()}

    def enable_bounded_storage(self, policy: RuntimeStorage = RuntimeStorage()) -> None:
        """Archive imported diagnostics before bounding RAM; no learning-state changes."""
        for row in self.metrics:
            archive_metric(self.output_dir / "diagnostics", row, policy)
        self.runtime_storage = policy
        self.metrics = self.metrics[-policy.metrics_keep:]
        for solver in self.solvers.values():
            solver.metrics = solver.metrics[-policy.metrics_keep:]
        # Root coverage is diagnostic only; journal future per-iteration counters.
        from training.metrics import Coverage
        for sampler in self.samplers.values():
            sampler.coverage = Coverage()
        self.metadata['diagnostics_scope'] = 'bounded_recent_metrics_and_per_iteration_root_coverage'

    @property
    def output_dir(self) -> Path:
        path = Path(self.config["output_dir"])
        return path if path.is_absolute() else Path(__file__).resolve().parents[1] / path

    def run_iteration(self, *, executor: ProcessPoolExecutor | None = None) -> dict:
        # A failed worker, fit, evaluation or publication leaves both tracks and
        # root sampler RNG/coverage at the last complete iteration.
        staged_solvers = {name: solver.fork() for name, solver in self.solvers.items()}
        staged_samplers = copy.deepcopy(self.samplers)
        if self.runtime_storage:
            from training.metrics import Coverage
            for sampler in staged_samplers.values():
                sampler.coverage = Coverage()
        row = {"iteration": self.iteration + 1, "tracks": {}, "checkpoint_version": checkpoint_contract(self.config)["checkpoint"],
               "config_hash": config_hash(self.config),
               "traversal_budget": {key: self.config[key] for key in ("max_nodes", "max_depth")}}
        started = time.perf_counter()
        config = self.config
        rng_state = torch.get_rng_state()
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(config["trainer_threads"])
            root_strategies = ({name: FrozenStrategy(s.snapshot()) for name,s in self.solvers.items()}
                               if config.get('root_policy') == 'current_advantage' else {})
            for name, solver in staged_solvers.items():
                sampler = staged_samplers[name]
                previous = self.solvers[name].average_model
                def rollout_policy(obs):
                    count = round(obs.numeric[NUMERIC_NAMES.index("player_count")] * 3)
                    track_name = "3max" if count == 3 else "hu"
                    if root_strategies:
                        if track_name not in root_strategies:
                            raise KeyError(f"Current-strategy roots require the {count}-player track")
                        return root_strategies[track_name](obs)
                    if track_name not in self.solvers or self.solvers[track_name].average_model is None:
                        raise KeyError(f"On-policy tournament requires the {count}-player average policy at iteration {self.iteration}")
                    return model_policy(self.solvers[track_name].average_model, obs)
                sampler.policy = None if self.iteration == 0 else rollout_policy
                root_factory = sampler.task_factory(solver.players, config[name]["traversals_per_player"])
                metric = solver.run_iteration(root_factory, config[name]["traversals_per_player"],
                    workers=config["workers"], advantage_epochs=config["advantage_epochs"],
                    average_epochs=config["average_epochs"], batch_size=config["batch_size"],
                    max_nodes=config["max_nodes"], max_depth=config["max_depth"],
                    sample_byte_budget=config["sample_byte_budget"], generation_byte_budget=config["generation_byte_budget"],
                    traversal_mode=config["traversal_mode"], learning_rate=config["learning_rate"],
                    trainer_threads=config["trainer_threads"], executor=executor,
                    fit_schedule=config.get('fit_schedule'), generation_batch_size=config.get('generation_batch_size'),
                    **({"epsilon": config["outcome_epsilon"]} if config["traversal_mode"] == "outcome_sampling" else {}))
                sampler.policy = None
                metric["fixed_probe_policy_drift"] = (None if metric['average_policy'].get('skipped') else
                                                      fixed_drift(previous, solver.average_model, self.probes[name]))
                metric['root_policy'] = config.get('root_policy', 'average')
                metric["root_coverage"] = sampler.coverage.as_dict()
                metric["hole_card_coverage"] = {key: dict(values) for key, values in sampler.hole_coverage.items()}
                if row["iteration"] % config["evaluation_every"] == 0:
                    metric["evaluation"] = evaluate_solver(solver, self.probes[name], config["evaluation_max_nodes"], config["batch_size"], self.solvers[name].snapshot())
                row["tracks"][name] = metric
            row["wall_seconds"] = time.perf_counter() - started
            candidate = copy.copy(self)
            candidate.solvers, candidate.samplers = staged_solvers, staged_samplers
            candidate.iteration = row["iteration"]
            candidate.metrics = [*self.metrics, row]
            if self.runtime_storage:
                candidate.metrics = candidate.metrics[-self.runtime_storage.metrics_keep:]
                for solver in candidate.solvers.values():
                    solver.metrics = solver.metrics[-self.runtime_storage.metrics_keep:]
            if candidate.iteration % config["checkpoint_every"] == 0:
                path = self.output_dir / "checkpoints" / f"iteration_{candidate.iteration:06d}.pt"
                row["checkpoint_path"] = str(path)
                candidate.save_checkpoint(path)
            # The checkpoint contains metrics, allowing JSONL repair after a crash.
            candidate._write_metrics()
        except BaseException:
            torch.set_rng_state(rng_state)
            raise
        finally:
            torch.set_num_threads(threads)
        self.solvers, self.samplers = staged_solvers, staged_samplers
        self.iteration = candidate.iteration
        self.metrics = candidate.metrics
        return row

    def _write_metrics(self) -> None:
        if self.runtime_storage and self.metrics:
            archive_metric(self.output_dir / "diagnostics", self.metrics[-1], self.runtime_storage)
        atomic_bytes(self.output_dir / "metrics.jsonl", "".join(json.dumps(row, allow_nan=False) + "\n" for row in self.metrics).encode())
        for name in self.solvers:
            coverage = {"iteration": self.iteration, "roots": self.samplers[name].coverage.as_dict(),
                        "samples_by_iteration": [row["tracks"][name]["sample_coverage"] for row in self.metrics]}
            atomic_bytes(self.output_dir / name / "coverage.json", (json.dumps(coverage, indent=2) + "\n").encode())
            rows = [dict(iteration=row["iteration"], **row["tracks"][name]) for row in self.metrics]
            atomic_bytes(self.output_dir / name / "metrics.jsonl", "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows).encode())

    def save_checkpoint(self, path: str | Path) -> None:
        relative = Path(path).resolve().relative_to(self.output_dir.resolve()) if self.runtime_storage else None
        raw = {"contract": checkpoint_contract(self.config), "config": self.config, "config_hash": config_hash(self.config),
               "iteration": self.iteration, "metadata": self.metadata, "metrics": self.metrics,
               "runtime_storage": asdict(self.runtime_storage) if self.runtime_storage else None,
               "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(), "tracks": {}}
        for name, solver in self.solvers.items():
            sampler = self.samplers[name]
            raw["tracks"][name] = {"version": solver.version, "seed": solver.seed, "rng": solver.rng.getstate(),
                "advantage_weights": solver.snapshot().advantage_weights,
                "average_weights": None if solver.average_model is None else solver.average_model.state_dict(),
                "advantage_memory": pack_memory(solver.advantage_memory),
                "strategy_memory": pack_memory(solver.strategy_memory), "metrics": solver.metrics,
                "traversal_mode": solver.traversal_mode, "sampler_rng": sampler.rng.getstate(),
                "card_sampling_state": sampler.card_state(),
                "stratum_index": sampler.stratum_index, "root_coverage": sampler.coverage.as_dict(),
                "tournament": pack_tournament(sampler.tournament)}
            if solver.street_config is not None:
                raw["tracks"][name]["specialist_states"] = solver.specialist_states
        write_checkpoint(Path(path), raw)
        if self.runtime_storage:
            atomic_bytes(self.output_dir / 'resume.json', (json.dumps({'path': str(relative), 'iteration': self.iteration})+'\n').encode())
            if Path(path).parent == self.output_dir / "checkpoints":
                rotate_checkpoints(Path(path).parent, self.runtime_storage.checkpoints_keep)

    @classmethod
    def load_checkpoint(cls, path: str | Path, config: dict | None = None, *,
                        allow_budget_increase: bool = False,
                        allow_execution_changes: bool = False) -> TrainingRunner:
        """Restore state; explicitly audit guard increases or execution-only changes."""
        raw = read_checkpoint(Path(path))
        if config_hash(raw["config"]) != raw["config_hash"]:
            raise ValueError(f"Checkpoint configuration hash mismatch at {path}")
        target = raw["config"] if config is None else validate_config(config)
        budget_change = None
        execution_change = None
        if config_hash(target) != raw["config_hash"]:
            changed = {key for key in set(target) | set(raw['config']) if target.get(key) != raw['config'].get(key)}
            execution_keys = {'workers', 'trainer_threads', 'generation_batch_size'}
            execution = changed & execution_keys
            guards = changed - execution
            if ((execution and not allow_execution_changes)
                    or (guards and (not allow_budget_increase or not guards <= {'max_nodes','max_depth'}
                                    or any(target[key] < raw['config'][key] for key in guards)))):
                raise ValueError(f"Checkpoint configuration hash mismatch at {path}; changed={sorted(changed)}; "
                                 "only explicitly enabled traversal-guard increases or execution settings may resume")
            change = {"iteration": raw["iteration"], "checkpoint": str(path),
                             "source_config_hash": raw["config_hash"], "config_hash": config_hash(target),
                             "source_metadata": source_metadata(),
                             "before": {key: raw["config"].get(key) for key in sorted(changed)},
                             "after": {key: target.get(key) for key in sorted(changed)}}
            budget_change = change if guards else None
            execution_change = change if execution else None
        runner = cls(raw["config"])
        if set(raw["tracks"]) != set(runner.solvers) or type(raw["iteration"]) is not int or raw["iteration"] < 0 or len(raw["metrics"]) != metric_count(raw):
            raise ValueError(f"Inconsistent checkpoint tracks/iteration at {path}")
        runner.runtime_storage = RuntimeStorage(**raw["runtime_storage"]) if raw.get("runtime_storage") else None
        runner.iteration, runner.metadata, runner.metrics = raw["iteration"], raw["metadata"], raw["metrics"]
        with torch.random.fork_rng(devices=[]):
            for name, track in raw["tracks"].items():
                solver, sampler = runner.solvers[name], runner.samplers[name]
                if (track["traversal_mode"] != (runner.config["traversal_mode"] if runner.iteration else None)
                        or track["seed"] != solver.seed or len(track["metrics"]) != metric_count(raw)):
                    raise ValueError(f"Inconsistent checkpoint mode/seed/metrics at {path}: {name}")
                if track["version"] != runner.iteration:
                    raise ValueError(f"Inconsistent checkpoint model/replay version at {path}: {name}")
                solver.version, solver.metrics, solver.traversal_mode = track["version"], track["metrics"], track["traversal_mode"]
                solver.rng.setstate(track["rng"])
                solver.advantage_memory = unpack_memory(track["advantage_memory"])
                solver.strategy_memory = unpack_memory(track["strategy_memory"])
                if solver.advantage_memory.kind != "advantage" or solver.strategy_memory.kind != "strategy":
                    raise ValueError(f"Checkpoint replay slot kind mismatch at {path}: {name}")
                if not runner.iteration and solver.street_config is None and (track["advantage_weights"] is not None or track["average_weights"] is not None
                                             or solver.advantage_memory.seen or solver.strategy_memory.seen):
                    raise ValueError(f"Uninitialized checkpoint contains trained state at {path}: {name}")
                for memory in (solver.advantage_memory, solver.strategy_memory):
                    capacity = runner.config[name]["advantage_capacity"] if memory.kind == "advantage" else runner.config["strategy_capacity"]
                    budget = runner.config[name]["advantage_byte_budget"] if memory.kind == "advantage" else runner.config["memory_byte_budget"]
                    from ml.stratified_memory import ProtectedReplay
                    fraction = runner.config.get("replay_opening_fraction")
                    if isinstance(memory, ProtectedReplay) != (fraction is not None):
                        raise ValueError(f"Checkpoint replay allocation/config mismatch: {name}")
                    if fraction is not None:
                        expected_opening = max(1, min(capacity - 1, int(capacity * fraction)))
                        if memory.memories["opening"].capacity != expected_opening:
                            raise ValueError(f"Checkpoint opening allocation mismatch: {name}")
                    if solver.street_config is not None:
                        from ml.street_replay import StreetReplay
                        if not isinstance(memory, StreetReplay):
                            raise ValueError("Street checkpoint requires partitioned replay")
                        for street, entry in solver.street_config[memory.kind].items():
                            part = memory.memories[street]
                            if part.capacity != entry["capacity"] or part.byte_budget != entry["byte_budget"]:
                                raise ValueError(f"Street replay allocation mismatch: {name}/{street}")
                    if memory.capacity != capacity or memory.byte_budget != budget:
                        raise ValueError(f"Checkpoint replay/config capacity mismatch: {name}")
                    if any(s.player not in solver.players or s.iteration > runner.iteration
                           or round(s.state.numeric[NUMERIC_NAMES.index("player_count")] * 3) != len(solver.players)
                           for s in memory.samples):
                        raise ValueError(f"Checkpoint contains foreign/future samples: {name}/{memory.kind}")
                    if runner.iteration and memory.traversal_mode != runner.config["traversal_mode"]:
                        raise ValueError(f"Checkpoint replay traversal mismatch: {name}")
                if runner.iteration:
                    if not track["advantage_weights"] or track["average_weights"] is None:
                        raise ValueError(f"Missing shared checkpoint model at {path}: {name}")
                    solver.advantage_model = StreetNetworks("advantage") if solver.street_config else AdvantageNetwork()
                    solver.advantage_model.load_state_dict(track["advantage_weights"], strict=True)
                    solver.advantage_model.eval()
                    solver.average_model = StreetNetworks("strategy") if solver.street_config else AveragePolicyNetwork()
                    solver.average_model.load_state_dict(track["average_weights"], strict=True)
                    solver.average_model.eval()
                if solver.street_config is not None:
                    if set(track.get("specialist_states", {})) != {"advantage", "strategy"}:
                        raise ValueError("Missing street optimizer/schedule states")
                    solver.specialist_states = track["specialist_states"]
                    from ml.street_training import validate_specialist_states
                    validate_specialist_states(solver)
                sampler.rng.setstate(track["sampler_rng"])
                sampler.restore_card_state(track["card_sampling_state"])
                sampler.stratum_index = track["stratum_index"]
                sampler.coverage.merge(track["root_coverage"])
                sampler.tournament = unpack_tournament(track["tournament"])
        torch.set_rng_state(raw["torch_rng"])
        random.setstate(raw["python_rng"])
        runner.resume_checkpoint = Path(path)
        if budget_change is not None or execution_change is not None:
            runner.config = target
            runner.metadata = copy.deepcopy(runner.metadata)
        if budget_change is not None:
            runner.metadata.setdefault("resume_budget_changes", []).append(budget_change)
            runner.resume_budget_change = budget_change
        if execution_change is not None:
            runner.metadata.setdefault('resume_execution_changes', []).append(execution_change)
            runner.resume_execution_change = execution_change
        return runner

    def run(self, iterations: int | None = None) -> None:
        if self.iteration == 0 and self.resume_checkpoint is None and (self.output_dir / "metrics.jsonl").exists():
            raise FileExistsError(f"Training run already exists at {self.output_dir}; load its explicit checkpoint with TrainingRunner.load_checkpoint")
        remaining = max(0, self.config["outer_iterations"] - self.iteration) if iterations is None else iterations
        if type(remaining) is not int or remaining < 0:
            raise ValueError(f"Explicit nonnegative additional iterations required: {remaining}")
        from scripts.parallel_cfr import traversal_workers
        with traversal_workers(self.config["workers"]) as executor:
            for _ in range(remaining):
                row = self.run_iteration(executor=executor)
                print(f"Iteration {self.iteration}: {row['wall_seconds']:.2f}s", flush=True)
        self.save_checkpoint(self.output_dir / "checkpoints" / f"iteration_{self.iteration:06d}.pt")

    def run_for(self, seconds: float, checkpoint_path: str | Path) -> dict:
        """Finish whole iterations until the wall budget, then save complete state."""
        import math
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"Positive finite training duration required: {seconds}")
        if self.iteration == 0 and self.resume_checkpoint is None and (self.output_dir / "metrics.jsonl").exists():
            raise FileExistsError(f"Training history exists at {self.output_dir}; an explicit checkpoint is required")
        started = time.monotonic()
        initial_iteration = self.iteration
        stopped = "duration"
        from scripts.parallel_cfr import traversal_workers
        with traversal_workers(self.config["workers"]) as executor:
            try:
                while time.monotonic() - started < seconds:
                    row = self.run_iteration(executor=executor)
                    detail = '; '.join(f"{name}: {m['traversals']} traversals, {m['nodes']} nodes, "
                        f"generation={m['generation_seconds']:.2f}s, fit={m['fit_seconds']:.2f}s, "
                        f"A updates={m['advantage'].get('updates_completed', '?')}, "
                        f"B updates={m['average_policy'].get('updates_completed', 0)}"
                        for name, m in row['tracks'].items())
                    print(f"Iteration {self.iteration}: {row['wall_seconds']:.2f}s; {detail}; elapsed {time.monotonic() - started:.1f}/{seconds:.1f}s", flush=True)
            except KeyboardInterrupt:
                stopped = "interrupted"
                print("Interrupted; saving the last complete iteration", flush=True)
            except BaseException:
                self.save_checkpoint(checkpoint_path)
                print(f"Training stopped; saved last complete iteration {self.iteration} at {checkpoint_path}", flush=True)
                raise
        self.save_checkpoint(checkpoint_path)
        return {"initial_iteration": initial_iteration, "final_iteration": self.iteration,
                "elapsed_seconds": time.monotonic() - started, "requested_seconds": seconds,
                "stop_reason": stopped, "checkpoint": str(checkpoint_path)}

    def export(self, *, onnx: bool = False) -> dict:
        from ml.export_onnx import export_average_policy
        outputs = {}
        for name, solver in self.solvers.items():
            directory = self.output_dir / "policy"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"average_{name}.pt"
            # Build the complete checkpoint before atomic publication.
            import tempfile
            import os
            descriptor, temporary = tempfile.mkstemp(dir=directory, suffix=".pt")
            os.close(descriptor)
            try:
                solver.export_average(temporary)
                atomic_bytes(path, Path(temporary).read_bytes())
            finally:
                os.unlink(temporary)
            outputs[name] = str(path)
            if onnx:
                export_average_policy(path, path.with_suffix(".onnx"), path.with_suffix(".json"), observe_for_export(self.probes[name]))
        return outputs


def observe_for_export(probes):
    from infoset import observe
    return observe(probes[0][1])
