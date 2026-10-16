"""Plan thousands of roots, calculate a bounded pilot, and measure amortization."""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

from hybrid.adaptive_labels import LabelBudget
from hybrid.adaptive_label_experiments import exact_terminal_check
from hybrid.knowledge_benchmark import run as benchmark
from hybrid.knowledge_curriculum import FAMILIES, manifest
from hybrid.label_dataset import build
from training.checkpoint import atomic_bytes


def demonstrate(output: Path) -> dict:
    if output.exists():
        raise ValueError(f"Fresh experiment destination required: {output}")
    output.mkdir(parents=True)
    plan = manifest(output / "planned-4000.json", 100)
    identities = {f"{FAMILIES[(street_index + count_index * 2) % len(FAMILIES)].name}/{count}/{street}/0"
                  for count_index, count in enumerate((2, 3))
                  for street_index, street in enumerate(("PREFLOP", "FLOP", "TURN", "RIVER"))}
    rows = [row for row in plan["roots"] if row["id"] in identities]
    pilot = {**plan, "roots": rows, "count": len(rows), "split_counts": dict(Counter(r["split"] for r in rows))}
    pilot_path = output / "pilot-manifest.json"
    atomic_bytes(pilot_path, json.dumps(pilot).encode())
    budget = LabelBudget(looks=(32, 128, 512), tolerance_bb=.1, max_nodes=100000, seed=141)
    report = {"planned": plan["count"], "pilot_budget": asdict(budget),
              "pilot": build(pilot_path, output / "labels", budget, max_new_labels=8, max_cpu_seconds=120)}
    exact_terminal_check(output / "exact-terminal-check.json")
    report["amortization"] = benchmark(output / "amortization", updates=64, repeats=3)
    atomic_bytes(output / "report.json", json.dumps(report, indent=2, allow_nan=False).encode())
    return report
