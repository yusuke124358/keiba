# Config conventions

## Purpose
- Avoid config/config.yaml conflicts
- Preserve exact config used for reproducibility and audits

## Resolution order
1) `KEIBA_CONFIG_PATH` (env)
2) `--config` (CLI)
3) default: `config/config.yaml`

## Recommended usage
Create a full config file and point to it:

```powershell
$env:KEIBA_CONFIG_PATH="config/experiments/<task_id>.yaml"
python py64_analysis/scripts/run_holdout.py ...
```

## Files written to run_dir
- `config_used.yaml` (canonical dump)
- `config_origin.json` (origin, path, git commit, sha256)
- `summary.json` / `metrics.json`

## Do not
- Edit `config/config.yaml` directly for experiments
