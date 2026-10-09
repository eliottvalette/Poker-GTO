"""Read-only checkpoint diagnostics with complete opening predictions retained."""
from __future__ import annotations

from collections import defaultdict
import gc
import hashlib
import json
import math
from pathlib import Path
import random

import torch

from actions import ACTION_IDS
from cfr_solver import regret_matching
from features.neural import NEURAL_NUMERIC_NAMES
from ml.model import AdvantageNetwork, encode_batch
from scripts.fit_budget_audit import opening_predictions
from scripts.hu_coverage_experiment import write_json
from training.metrics import opening_hand_class
from training.runner import TrainingRunner

OPENING_LEGAL_INDICES = (0, 2, 3, 4, 5, 6, 12)


def checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def holding(combo: list[int]) -> str:
    high, low = sorted((card // 4 for card in combo), reverse=True)
    return "23456789TJQKA"[high] + "23456789TJQKA"[low] + (
        "" if high == low else "s" if combo[0] % 4 == combo[1] % 4 else "o")


def weighted_mean(rows: list[dict], key: str) -> list[float]:
    total = sum(row["weight"] for row in rows)
    return [sum(row["weight"] * row[key][i] for row in rows) / total
            for i in range(len(ACTION_IDS))]


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"records": 0}
    weights = [row["weight"] for row in rows]
    return {"records": len(rows), "ess": sum(weights) ** 2 / sum(w*w for w in weights),
            "target": weighted_mean(rows, "target"),
            "prediction": weighted_mean(rows, "prediction"),
            "mean_loss": sum(r["weight"] * r["loss"] for r in rows) / sum(weights),
            "nonpositive_predictions": sum(r["nonpositive"] for r in rows)}


def probe_summary(probe: dict, advantage: bool) -> dict:
    groups: dict[str, list[list[float]]] = defaultdict(list)
    nonpositive = []
    for combo, output, probabilities in zip(probe["combos"], probe["outputs"], probe["probabilities"]):
        label = holding(combo)
        groups[label].append(probabilities)
        # Seven legal actions at the fixed opening root.
        if advantage and max(output[i] for i in OPENING_LEGAL_INDICES) <= 0:
            nonpositive.append(label)
    means = {key: [sum(col)/len(values) for col in zip(*values)] for key, values in groups.items()}
    return {"holdings": means, "max_card_order_gap": probe["max_card_order_gap"],
            "nonpositive_combos": len(nonpositive), "nonpositive_classes": sorted(set(nonpositive)),
            "TV_AA_72o": sum(abs(a-b) for a,b in zip(means["AA"], means["72o"]))/2,
            "TV_KK_32o": sum(abs(a-b) for a,b in zip(means["KK"], means["32o"]))/2}


def audit_openings(runner: TrainingRunner, source_models: dict[int, AdvantageNetwork], kind: str) -> dict:
    solver = runner.solvers["hu"]
    memory = solver.advantage_memory if kind == "advantage" else solver.strategy_memory
    model = solver.advantage_model if kind == "advantage" else solver.average_model
    indices = list(range(len(memory.samples)))
    random.Random(solver.seed + (0 if kind == "advantage" else 10)).shuffle(indices)
    cut = max(1, min(len(indices)-1, round(len(indices)*.2)))
    heldout = set(indices[:cut])
    selected = [(i, sample) for i, sample in enumerate(memory.samples) if opening_hand_class(sample.state)]
    records = []
    for offset in range(0, len(selected), 256):
        chunk = selected[offset:offset+256]
        with torch.no_grad():
            predictions = model(encode_batch([s.state for _,s in chunk], model.feature_version)).tolist()
        for (index, sample), output in zip(chunk, predictions):
            mask = sample.state.legal_mask
            target = sample.target
            if kind == "advantage":
                loss = sum((a-b)**2 for a,b,m in zip(output,target,mask) if m)/sum(mask)
            else:
                loss = sum(t * math.log(max(t,1e-12)/max(p,1e-12)) for t,p in zip(target,output))
            values = dict(zip(NEURAL_NUMERIC_NAMES, sample.state.numeric))
            records.append({"replay_index": index, "holding": opening_hand_class(sample.state),
                "split": "heldout" if index in heldout else "train", "iteration": sample.iteration,
                "weight": sample.weight*sample.iteration, "target": list(target), "prediction": output,
                "legal_mask": list(mask), "loss": loss,
                "nonpositive": kind == "advantage" and max(v for v,m in zip(output,mask) if m)<=0,
                "policy": list(regret_matching(output,mask)) if kind == "advantage" else output,
                "hero_stack_bb": values["stack_0"]*25, "effective_stack_bb": values["effective_1"]*25,
                "initial_stack_bb": values["initial_0"]*25})
    provenance_max = 0.0
    if kind == "strategy":
        for iteration in range(1, runner.iteration+1):
            pairs = [(record, sample) for record, (_,sample) in zip(records, selected) if sample.iteration == iteration]
            for offset in range(0,len(pairs),256):
                chunk = pairs[offset:offset+256]
                if iteration == 1:
                    outputs = [[0.0]*len(ACTION_IDS) for _ in chunk]
                else:
                    with torch.no_grad():
                        outputs = source_models[iteration-1](encode_batch([s.state for _,s in chunk])).tolist()
                for (record,sample),output in zip(chunk,outputs):
                    source = regret_matching(output,sample.state.legal_mask)
                    gap = max(abs(a-b) for a,b in zip(source,sample.target))
                    record["source_policy_gap"] = gap
                    provenance_max = max(provenance_max,gap)
    groups = {label: summarize([r for r in records if r["holding"]==label])
              for label in sorted({r["holding"] for r in records})}
    return {"records": records, "summary": summarize(records), "by_holding": groups,
            "by_split": {s: summarize([r for r in records if r["split"]==s]) for s in ("train","heldout")},
            "by_iteration": {str(i): summarize([r for r in records if r["iteration"]==i]) for i in range(1,runner.iteration+1)},
            "source_policy_max_gap": provenance_max if kind == "strategy" else None,
            "scope": "Mixed opening contexts; per-record predictions use exactly the recorded state. Aggregated targets are not a fixed-root reference."}


def run_audit(run_dir: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Audit output already exists: {output}")
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    report = {"run": str(run_dir), "iterations": [], "actions": list(ACTION_IDS)}
    source_models = {}
    for iteration in range(1,7):
        path = run_dir / "checkpoints" / f"iteration_{iteration:06d}.pt"
        before = checkpoint_hash(path)
        runner = TrainingRunner.load_checkpoint(path)
        solver = runner.solvers["hu"]
        source_models[iteration] = solver.advantage_model
        row = {"iteration": iteration, "checkpoint_sha256": before, "probes": {}}
        for kind,model in (("advantage",solver.advantage_model),("average",solver.average_model)):
            probe = opening_predictions(model,2,256)
            write_json(output / f"{kind}-{iteration:02d}.json",probe)
            row["probes"][kind] = probe_summary(probe,kind=="advantage")
        report["iterations"].append(row)
        if iteration == 6:
            for kind in ("advantage","strategy"):
                replay = audit_openings(runner,source_models,kind)
                write_json(output / f"{kind}-opening-replay.json",replay)
                report[kind] = {k:v for k,v in replay.items() if k!="records"}
        after = checkpoint_hash(path)
        if before != after:
            raise RuntimeError(f"Checkpoint changed during audit: {path}")
        write_json(output / "report.json",report)
        print(f"Audited checkpoint {iteration}/6; card-order gaps: "
              f"{[p['max_card_order_gap'] for p in row['probes'].values()]}",flush=True)
        del runner,solver
        gc.collect()
    fixed_root_reference(output)
    return report


def fixed_root_reference(output: Path) -> dict:
    """Average the six generating strategies at a root with own reach equal to one."""
    probes = [json.loads((output / f"advantage-{i:02d}.json").read_text()) for i in range(1,7)]
    average = json.loads((output / "average-06.json").read_text())
    uniform = [float(i in OPENING_LEGAL_INDICES)/len(OPENING_LEGAL_INDICES) for i in range(len(ACTION_IDS))]
    records = []
    for index,combo in enumerate(average["combos"]):
        # Iteration t collects policy from checkpoint t-1, before fitting t.
        sources = [uniform] + [p["probabilities"][index] for p in probes[:5]]
        reference = [sum(t*p[a] for t,p in enumerate(sources,1))/21 for a in range(len(ACTION_IDS))]
        prediction = average["probabilities"][index]
        records.append({"combo": combo,"holding":holding(combo),"weight":1,
                        "target":reference,"prediction":prediction,"source_policies":sources,
                        "loss":sum(t*math.log(max(t,1e-12)/max(p,1e-12)) for t,p in zip(reference,prediction)),
                        "nonpositive":False,"tv":sum(abs(t-p) for t,p in zip(reference,prediction))/2})
    report = {"scope":"Exact arithmetic reference for the six learned generating policies at this opening root, weighted by iteration. Not an equilibrium strategy or an exact-game solution.",
              "records":records,"summary":summarize(records),
              "mean_tv":sum(r["tv"] for r in records)/len(records),
              "by_holding":{label:summarize([r for r in records if r["holding"]==label]) for label in sorted({r["holding"] for r in records})}}
    write_json(output / "fixed-root-reference.json",report)
    return report
