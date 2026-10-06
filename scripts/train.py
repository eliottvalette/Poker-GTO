"""Explicit offline training command; import training.runner for the reusable core."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from training.config import load_config
from training.runner import TrainingRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Versioned two-track hand-cEV Deep CFR training")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--iterations", type=int, help="Additional iterations; enables anytime continuation")
    parser.add_argument("--preflight", action="store_true", help="Run bounded measurement, never the pilot")
    parser.add_argument("--export-onnx", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.preflight:
        if args.resume or args.iterations or args.export_onnx:
            parser.error("--preflight must run independently of resume/iterations/export")
        from training.preflight import run_preflight
        report = run_preflight(config)
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    runner = TrainingRunner.load_checkpoint(args.resume, config) if args.resume else TrainingRunner(config)
    if not args.resume and (runner.output_dir / "metrics.jsonl").exists():
        raise FileExistsError(f"Training run already exists: {runner.output_dir}; specify --resume with its checkpoint")
    runner.run(args.iterations)
    runner.export(onnx=args.export_onnx)


if __name__ == "__main__":
    main()
