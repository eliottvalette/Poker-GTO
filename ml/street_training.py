"""Incremental specialist schedules with persisted optimizer and split contracts."""
from __future__ import annotations
import copy
import hashlib
import time
import torch
from ml.model import STREETS
from ml.train import fit


def validate_street_config(config: dict) -> None:
    if set(config) - {"collector_enumerate_from_street"} != {"version", "advantage", "strategy"} or config["version"] != 1:
        raise ValueError("Street configuration requires version=1, advantage, strategy")
    if config.get("collector_enumerate_from_street") is not None and config["collector_enumerate_from_street"] not in range(4):
        raise ValueError("Invalid collector enumeration street")
    for kind in ("advantage", "strategy"):
        if set(config[kind]) != set(STREETS):
            raise ValueError(f"Missing street configuration for {kind}")
        for name, schedule in config[kind].items():
            required = {"capacity", "byte_budget", "every", "min_new_samples", "max_updates"}
            optional = {'context_capacities'} if name == 'PREFLOP' else set()
            if (set(schedule) - optional != required
                    or any(type(schedule[k]) is not int or schedule[k] < 1 for k in required)):
                raise ValueError(f"Invalid {kind}/{name} schedule: {schedule}")
            if 'context_capacities' in schedule:
                from ml.context_replay import expanded_capacities
                if sum(expanded_capacities(schedule['context_capacities']).values()) != schedule['capacity']:
                    raise ValueError('Context capacities must sum to preflop capacity')


def grouped_partition(sample, seed: int) -> str:
    if not sample.root_group:
        raise ValueError("Grouped validation requires root provenance; legacy records need an explicit audit grouping")
    value = int(hashlib.sha256(f"{seed}:{sample.root_group}".encode()).hexdigest()[:8], 16) % 10
    return "test" if value == 0 else "validation" if value == 1 else "train"


def fit_specialists(model, memory, states: dict, schedules: dict, *, version: int,
                    seed: int, batch_size: int, learning_rate: float, cache_encoding: bool,
                    allow_frozen_replay: bool = False):
    model = copy.deepcopy(model)
    states = copy.deepcopy(states)
    metrics = {}
    for index, street in enumerate(STREETS):
        config = schedules[street]
        population = memory.memories[street]
        state = states.setdefault(street, {"optimizer": None, "seen_at_fit": 0,
                                           "fits": 0, "updates": 0, "model_version": 0})
        fresh = population.seen - state["seen_at_fit"]
        partitions = {key: [] for key in ("train", "validation", "test")}
        for sample in population.samples:
            partitions[grouped_partition(sample, seed)].append(sample)
        row = {"new_samples": fresh, "partition_sizes": {k: len(v) for k,v in partitions.items()},
               "cold_start": not bool(model.trained[index]), "model_version": state["model_version"]}
        new_training = sum(s.iteration > state["model_version"] for s in partitions["train"])
        row["new_retained_training_samples"] = new_training
        if not new_training and not allow_frozen_replay:
            row.update(skipped=True, reason="no_new_retained_training_data")
        elif version % config["every"] or fresh < config["min_new_samples"]:
            row.update(skipped=True, reason="cadence_or_new_data_threshold")
        elif not partitions["train"] or not partitions["validation"]:
            row.update(skipped=True, reason="insufficient_independent_train_validation_groups")
        else:
            specialist = model.specialists[street]
            optimizer = torch.optim.Adam(specialist.parameters(), lr=learning_rate)
            if state["optimizer"] is not None:
                optimizer.load_state_dict(state["optimizer"])
            started = time.perf_counter()
            # An update budget determines work, not repeated fixed full-replay epochs.
            batches = max(1, (len(partitions["train"]) + batch_size - 1) // batch_size)
            epochs = (config["max_updates"] + batches - 1) // batches
            row.update(fit(specialist, partitions["train"] + partitions["validation"],
                           epochs, batch_size, seed + version * 101 + index,
                           learning_rate=learning_rate, max_updates=config["max_updates"],
                           cache_encoding=cache_encoding, optimizer=optimizer,
                           explicit_partitions=(partitions["train"], partitions["validation"])))
            row["seconds"] = time.perf_counter() - started
            state.update(optimizer=optimizer.state_dict(), seen_at_fit=population.seen,
                         fits=state["fits"] + 1, updates=state["updates"] + row["updates_completed"],
                         model_version=version)
            model.trained[index] = True
            row.update(cold_start=False, model_version=version)
            state["last_fit_metrics"] = copy.deepcopy(row)
        state["coverage"] = {"generated": population.seen, "retained": len(population.samples),
                             "trajectories": len({s.trajectory_id for s in population.samples}),
                             "public_root_groups": len({s.root_group for s in population.samples}),
                             "partition_sizes": row["partition_sizes"]}
        from ml.fixed_validation import partition_coverage, fixed_metrics
        state['coverage']['splits'] = partition_coverage(partitions)
        if 'updates_completed' in row:
            row['fixed_validation'] = fixed_metrics(model.specialists[street], state, partitions,
                                                    seed+index, version, model.kind)
            state['last_fit_metrics'] = copy.deepcopy(row)
        metrics[street] = row
    return model.eval(), states, metrics


def validate_specialist_states(solver) -> None:
    """Reject mismatched optimizer/model/replay checkpoints before resuming work."""
    for kind, model, replay in (("advantage",solver.advantage_model,solver.advantage_memory),
                                ("strategy",solver.average_model,solver.strategy_memory)):
        states=solver.specialist_states[kind]
        if solver.version == 0 and not states:
            continue
        if set(states) != set(STREETS):
            raise ValueError(f"Missing {kind} specialist state")
        for index,street in enumerate(STREETS):
            state=states[street]
            if (any(type(state.get(k)) is not int or state[k] < 0
                    for k in ("fits","updates","seen_at_fit","model_version"))
                    or state["model_version"] > solver.version
                    or state["seen_at_fit"] > replay.memories[street].seen
                    or bool(model.trained[index]) != (state["fits"] > 0)
                    or (state["optimizer"] is not None) != (state["fits"] > 0)):
                raise ValueError(f"Inconsistent {kind}/{street} model/optimizer/replay state")
            if state["optimizer"] is not None:
                optimizer=torch.optim.Adam(model.specialists[street].parameters())
                optimizer.load_state_dict(state["optimizer"])
                for parameter,values in optimizer.state.items():
                    if any(values[k].shape != parameter.shape or not torch.isfinite(values[k]).all()
                           for k in ("exp_avg","exp_avg_sq")) or int(values["step"]) != state["updates"]:
                        raise ValueError(f"Invalid optimizer moments/step for {kind}/{street}")
            if 'validation_bank' in state:
                from ml.memory import TrainingSample
                from features.neural import NeuralObservation
                bank = state['validation_bank']
                if (bank.get('version') != 1 or not 1 <= bank.get('source_iteration',0) <= solver.version
                        or set(bank.get('splits',{})) != {'validation','test'}):
                    raise ValueError('Invalid frozen validation bank')
                for split, rows in bank['splits'].items():
                    if not 64 <= len(rows) <= 128:
                        raise ValueError('Invalid frozen validation bank size')
                    for packed in rows:
                        sample = TrainingSample(**{**packed,'state':NeuralObservation(**packed['state'])})
                        sample.validate()
                        if (sample.kind != kind or sample.state.street != index
                                or grouped_partition(sample,solver.seed) != split):
                            raise ValueError('Frozen validation bank leaks training roots or foreign samples')
