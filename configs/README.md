# Benchmark policy configs

`scripts/benchmark_policy.py` reads `configs/benchmark_policy.json` by default.

Edit the JSON file instead of passing long command lines. The useful knobs are:

- `hands`: base deals per scenario. With seat rotation and duplicate deals enabled, each scenario runs `hands * 6` table hands.
- `policy`: current policy to test.
- `old_policy`: previous policy path, or `git:HEAD:policy/avg_policy.json.gz`.
- `out_dir`: output folder. `{timestamp}` is replaced at runtime.
- `rotate_seats`: tests each bot in different seats.
- `duplicate_deals`: reuses the same deal across seat permutations.
- `enabled_scenarios`: exact scenario names to run.

Presets:

- `benchmark_policy.quick.json`: small smoke/evaluation run.
- `benchmark_policy.json`: default run.
- `benchmark_policy.long.json`: slower, lower-noise run.
