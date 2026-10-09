"""Recorded-player card strata preserve chance multiplicity for both datasets."""
import random
import unittest
import json
import tempfile
from pathlib import Path
from dataclasses import asdict
from collections import Counter
from unittest.mock import patch

from poker_game_expresso import HandState
from training.root_sampler import RootSampler
from ml.deep_cfr import DeepCFRSolver, TraversalTask, generate_samples
from scripts.strategy_collector_audit import small_root
from training.config import load_config
from training.runner import TrainingRunner


class RecordedCoverageTests(unittest.TestCase):
    def test_both_hu_opening_datasets_receive_full_exact_cycle(self):
        config = {"mixture": {"synthetic": 1, "on_policy": 0, "stratified": 0},
                  "max_rollout_decisions": 64, "max_rollout_hands": 64,
                  "tournament_start_players": 2, "hole_card_sampling": "stratified_recorded_opening"}
        sampler = RootSampler(2, 42, config)
        root = HandState.start({0: 5, 1: 5}, 0, random.Random(1))
        with patch.object(sampler, "_exploration", return_value=root):
            for traverser in (0, 1):
                cards = Counter(tuple(sorted(sampler.sample(traverser=traverser).actor.cards)) for _ in range(1326))
                self.assertEqual(len(cards), 1326)
                self.assertEqual(set(cards.values()), {1})
        self.assertEqual(set(sampler.card_cycles), {"0:SB:advantage", "0:SB:strategy"})

    def test_recorded_coverage_uses_actual_private_cards(self):
        # River records do not count as openings. Existing opening class counters
        # remain unchanged and raw card provenance is obtained at observation time.
        solver = DeepCFRSolver((0, 1))
        root = small_root(2)
        result = generate_samples(solver.snapshot(), TraversalTask(0, 0, root, 3, 1000, 64))
        self.assertNotIn("opening_joint_coverage", result.coverage)
        self.assertEqual(result.coverage["source"]["strategy"], len(result.strategies))

    def test_actual_opening_identity_and_new_mode_checkpoint_resume(self):
        root = HandState.start({0: 1.5, 1: 1.5}, 0, random.Random(39))
        solver = DeepCFRSolver((0, 1))
        generated = generate_samples(solver.snapshot(), TraversalTask(0, 1, root, 3, 1000, 64))
        rows = [json.loads(key) for key in generated.coverage["opening_joint_coverage"]]
        strategy_rows = [r for r in rows if r["dataset"] == "strategy"]
        self.assertEqual(len(strategy_rows), 1)
        self.assertEqual(strategy_rows[0]["exact_combo"], sorted(root.players[0].cards))
        self.assertEqual(strategy_rows[0]["source_iteration"], 1)
        with tempfile.TemporaryDirectory() as directory:
            config = load_config("configs/deep_cfr_pilot.json")
            config["3max"]["enabled"] = False
            config["hu"]["root_sampling"].update(hole_card_sampling="stratified_recorded_opening", tournament_start_players=2)
            config["hu"]["traversals_per_player"] = 2
            config.update(workers=1, advantage_epochs=1, average_epochs=1, output_dir=directory,
                          outer_iterations=1, evaluation_every=2, checkpoint_every=1)
            runner = TrainingRunner(config)
            runner.run_iteration()
            checkpoint = Path(directory) / "checkpoints/iteration_000001.pt"
            restored = TrainingRunner.load_checkpoint(checkpoint)
            for player in (0, 1):
                a = runner.samplers["hu"].sample(traverser=player)
                b = restored.samplers["hu"].sample(traverser=player)
                self.assertEqual(asdict(a), asdict(b))
