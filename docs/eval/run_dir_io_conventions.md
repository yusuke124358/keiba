# run_dir I/O conventions (PR-B)

## Rules
- Scripts that read or write evaluation outputs must accept `--run-dir` (or equivalent) explicitly.
- No implicit “latest run” lookup.
- Output defaults to:
  `run_dir/analysis/<script_slug>/<timestamp>/`

## Recommended CLI
- Required: `--run-dir <path>`
- Optional: `--out-dir <path>` (overrides default output location)

## Example
```bash
python py64_analysis/scripts/roi_phase1p5_market_blend_eval.py --run_dir data/holdout_runs/<run_dir>
```

## Related
- `docs/eval/config_conventions.md`
