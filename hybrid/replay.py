"""Opening-retention experiment using the optional Deep CFR replay allocation."""
from pathlib import Path
from ml.memory import ReservoirMemory, TrainingSample
from ml.stratified_memory import ProtectedReplay
from training.metrics import opening_hand_class


def retention_experiment(
    directory: Path, *, seeds: int = 32, stream_size: int = 1000
) -> dict:
    """Controlled rare-context inclusion test, not a claim about poker supervision."""
    import json
    import random
    from infoset import observe
    from poker_game_expresso import HandState
    from hybrid.experiments import small_game
    from hybrid.quality import mean_interval

    opening = observe(HandState.start({0: 3.0, 1: 3.0}, 0, random.Random(41)))
    river, _ = small_game(2)
    other = observe(river)
    stream = []
    for i in range(stream_size):
        obs = opening if i % 100 == 0 else other
        stream.append(
            TrainingSample(
                1,
                obs.hero,
                obs,
                tuple(float(m) * (i % 17) for m in obs.legal_mask),
                1.0,
                "advantage",
                0,
            )
        )
    exact_total = sum(max(s.target) for s in stream)
    rows = []
    directory.mkdir(parents=True, exist_ok=True)
    for seed in range(seeds):
        ordinary = ReservoirMemory(100, seed, "advantage", "hand_chip_delta")
        protected = ProtectedReplay(10, 90, kind="advantage", seed=seed)
        for sample in stream:
            ordinary.add(sample)
            protected.add(sample)
        rows.append(
            {
                "seed": seed,
                "ordinary_openings": sum(
                    opening_hand_class(s.state) is not None for s in ordinary.samples
                ),
                "protected_openings": protected.diagnostics()["opening"]["retained"],
                "ordinary_total_error": sum(max(s.target) for s in ordinary.samples)
                * stream_size
                / len(ordinary.samples)
                - exact_total,
                "corrected_protected_total_error": sum(
                    s.weight * max(s.target) for s in protected.samples
                )
                - exact_total,
            }
        )
        if seed == 0:
            protected.save(directory / "protected-replay.pt")
    report = {
        "budget": {
            "seeds": seeds,
            "stream_size": stream_size,
            "capacity": 100,
            "opening_capacity": 10,
        },
        "original_exact_total": exact_total,
        "rows": rows,
        "ordinary_zero_opening_fraction": sum(r["ordinary_openings"] == 0 for r in rows)
        / seeds,
        "protected_zero_opening_fraction": sum(
            r["protected_openings"] == 0 for r in rows
        )
        / seeds,
        "corrected_total_error": mean_interval(
            [r["corrected_protected_total_error"] for r in rows]
        ),
        "scope": "Controlled 1% opening stream; inverse-inclusion total estimator. Normalized fitting is a ratio estimator; no collector change or default training migration.",
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
