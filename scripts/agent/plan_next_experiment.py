#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

try:
    import yaml
except Exception as exc:
    raise RuntimeError(
        "PyYAML is required to run plan_next_experiment. Install with: pip install pyyaml"
    ) from exc


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_codex_bin() -> str:
    if os.name == "nt":
        for name in ("codex.cmd", "codex.exe", "codex"):
            path = shutil.which(name)
            if path:
                return path
    path = shutil.which("codex")
    if not path:
        raise RuntimeError("codex CLI not found in PATH")
    return path


def load_jsons(path: Path) -> list[dict]:
    if not path.exists():
        return []
    items = []
    for p in sorted(path.glob("*.json")):
        try:
            items.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return items


def load_seed_hypotheses(path: Path) -> list[dict]:
    if not path.exists():
        raise RuntimeError(f"Seed hypotheses file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise RuntimeError("seed_hypotheses.yaml must contain a non-empty list.")
    required = {
        "id",
        "title",
        "hypothesis",
        "change_scope",
        "acceptance_criteria",
        "metrics",
        "risk_level",
        "max_diff_size",
    }
    seeds = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise RuntimeError(f"seed_hypotheses entry {idx} is not a mapping.")
        missing = required - set(item.keys())
        if missing:
            seed_name = item.get("id", idx)
            raise RuntimeError(
                f"seed_hypotheses entry {seed_name} missing keys: {sorted(missing)}"
            )
        if not isinstance(item["acceptance_criteria"], list) or not item[
            "acceptance_criteria"
        ]:
            raise RuntimeError(
                f"seed_hypotheses entry {item['id']} has empty acceptance_criteria."
            )
        if not isinstance(item["metrics"], list) or not item["metrics"]:
            raise RuntimeError(f"seed_hypotheses entry {item['id']} has empty metrics.")
        if item["risk_level"] not in {"low", "medium", "high"}:
            raise RuntimeError(
                f"seed_hypotheses entry {item['id']} has invalid risk_level."
            )
        try:
            item["max_diff_size"] = int(item["max_diff_size"])
        except Exception as exc:
            raise RuntimeError(
                f"seed_hypotheses entry {item['id']} has invalid max_diff_size."
            ) from exc
        seeds.append(item)
    return seeds


def collect_used_seed_ids(runs: list[dict]) -> set[str]:
    used = set()
    for run in runs:
        seed_id = run.get("seed_id")
        if seed_id:
            used.add(str(seed_id))
    return used


def select_seed(seeds: list[dict], used_seed_ids: set[str]) -> dict:
    for seed in seeds:
        if str(seed.get("id")) not in used_seed_ids:
            return seed
    return seeds[0]


def collect_existing_run_ids(runs_dir: Path, runs: list[dict]) -> set[str]:
    ids = set()
    if runs_dir.exists():
        for p in runs_dir.glob("*.json"):
            ids.add(p.stem)
    for run in runs:
        run_id = run.get("run_id")
        if run_id:
            ids.add(str(run_id))
    return ids


def is_blank_or_tbd(value) -> bool:
    if not isinstance(value, str):
        return True
    stripped = value.strip()
    if not stripped:
        return True
    return "tbd" in stripped.lower()


def list_has_valid_text(values) -> bool:
    if not isinstance(values, list) or not values:
        return False
    for item in values:
        if is_blank_or_tbd(item):
            return False
    return True


def plan_needs_seed_override(plan: dict, seed_ids: set[str], selected_seed_id: str) -> bool:
    if not isinstance(plan, dict):
        return True
    seed_id = str(plan.get("seed_id") or "").strip()
    if seed_id != selected_seed_id:
        return True
    if seed_id not in seed_ids:
        return True
    if is_blank_or_tbd(plan.get("title")):
        return True
    if is_blank_or_tbd(plan.get("hypothesis")):
        return True
    if is_blank_or_tbd(plan.get("change_scope")):
        return True
    if not list_has_valid_text(plan.get("acceptance_criteria")):
        return True
    if not list_has_valid_text(plan.get("metrics")):
        return True
    if plan.get("risk_level") not in {"low", "medium", "high"}:
        return True
    max_diff = plan.get("max_diff_size")
    if not isinstance(max_diff, int) or max_diff < 1:
        return True
    return False


def append_reason(plan: dict, note: str) -> None:
    reason = str(plan.get("reason", "")).strip()
    plan["reason"] = f"{reason} ({note})" if reason else note


def apply_seed_to_plan(plan: dict, seed: dict, note: str) -> None:
    plan["seed_id"] = seed["id"]
    plan["title"] = seed["title"]
    plan["hypothesis"] = seed["hypothesis"]
    plan["change_scope"] = seed["change_scope"]
    plan["acceptance_criteria"] = seed["acceptance_criteria"]
    plan["metrics"] = seed["metrics"]
    plan["risk_level"] = seed["risk_level"]
    plan["max_diff_size"] = seed["max_diff_size"]
    plan["decision"] = "do"
    append_reason(plan, note)


RUN_ID_PATTERN = re.compile(r"^exp_\d{8}_\d{6}(?:_\d{3})?$")


def run_id_is_valid(run_id) -> bool:
    return isinstance(run_id, str) and bool(RUN_ID_PATTERN.match(run_id))


def build_run_id(now=None) -> str:
    if now is None:
        now = dt.datetime.utcnow()
    return now.strftime("exp_%Y%m%d_%H%M%S")


def ensure_unique_run_id(existing_ids: set[str], base_id: str) -> str:
    if base_id not in existing_ids:
        return base_id
    idx = 1
    while idx < 1000:
        candidate = f"{base_id}_{idx:03d}"
        if candidate not in existing_ids:
            return candidate
        idx += 1
    raise RuntimeError("Unable to generate unique run_id after 999 attempts.")


def run_codex(
    root: Path, prompt: str, schema: Path, output_path: Path, profile: str
) -> None:
    codex_bin = find_codex_bin()
    cmd = [
        codex_bin,
        "exec",
        "--profile",
        profile,
        "--full-auto",
        "--output-schema",
        str(schema),
        "--output-last-message",
        str(output_path),
    ]
    log_path = output_path.with_suffix(".log")
    with open(log_path, "w", encoding="utf-8") as log:
        result = subprocess.run(
            cmd + [prompt], cwd=root, stdout=log, stderr=log, text=True
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"codex exec failed with code {result.returncode}. See {log_path}"
        )


def normalize_schema(node):
    if isinstance(node, dict):
        node = dict(node)
        if node.get("type") == "object":
            props = node.get("properties")
            if isinstance(props, dict):
                node.setdefault("additionalProperties", False)
                req = list(node.get("required", []))
                for key in props.keys():
                    if key not in req:
                        req.append(key)
                node["required"] = req
                node["properties"] = {
                    key: normalize_schema(value) for key, value in props.items()
                }
            else:
                node.setdefault("additionalProperties", False)
        if "items" in node:
            node["items"] = normalize_schema(node["items"])
        return node
    if isinstance(node, list):
        return [normalize_schema(item) for item in node]
    return node


def contains_eval_metrics(eval_command) -> bool:
    if not eval_command:
        return False
    tokens = eval_command if isinstance(eval_command, list) else [eval_command]
    for cmd in tokens:
        lowered = cmd.lower()
        if "run_holdout.py" in lowered or "run_rolling_holdout.py" in lowered:
            return True
    return False


def ensure_eval_plan(plan: dict) -> None:
    run_id = plan.get("run_id") or dt.datetime.utcnow().strftime("RUN_%Y%m%d_%H%M%S")
    plan["run_id"] = run_id
    if contains_eval_metrics(plan.get("eval_command")):
        return
    plan["eval_command"] = [
        "py64_analysis\\.venv\\Scripts\\python.exe py64_analysis/scripts/run_holdout.py "
        "--train-start 2020-01-01 --train-end 2022-12-31 "
        "--valid-start 2023-01-01 --valid-end 2023-12-31 "
        "--test-start 2024-01-01 --test-end 2024-12-31 "
        f"--name {run_id} --out-dir data/holdout_runs/{run_id}"
    ]
    plan["metrics_path"] = f"data/holdout_runs/{run_id}/metrics.json"
    reason = plan.get("reason", "").strip()
    if reason:
        reason = f"{reason} (eval_command overridden to run_holdout)"
    else:
        reason = "eval_command overridden to run_holdout"
    plan["reason"] = reason


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", default="experiments/seed_hypotheses.yaml")
    p.add_argument("--runs", default="experiments/runs")
    p.add_argument("--knowledge", default="docs/knowledge")
    p.add_argument("--schema", default="schemas/agent/experiment_plan.schema.json")
    p.add_argument("--prompt", default="prompts/agent/scientist_plan.md")
    p.add_argument("--profile", default="agent_loop")
    p.add_argument("--out", default="")
    args = p.parse_args()

    root = repo_root()
    seed_path = root / args.seed
    seeds = load_seed_hypotheses(seed_path)
    runs_dir = root / args.runs
    runs = load_jsons(runs_dir)
    recent = runs[-10:] if len(runs) > 10 else runs
    used_seed_ids = collect_used_seed_ids(runs)
    selected_seed = select_seed(seeds, used_seed_ids)
    seed_ids = {str(seed["id"]) for seed in seeds}
    existing_run_ids = collect_existing_run_ids(runs_dir, runs)

    knowledge_dir = root / args.knowledge
    knowledge_files = []
    if knowledge_dir.exists():
        for p in knowledge_dir.rglob("*.md"):
            knowledge_files.append(str(p.relative_to(root)))

    prompt_text = (root / args.prompt).read_text(encoding="utf-8")
    seed_text = yaml.safe_dump([selected_seed], sort_keys=False)
    payload = {
        "seed_hypotheses": seed_text,
        "recent_runs": recent,
        "knowledge_files": knowledge_files,
    }
    prompt = prompt_text + "\n\nINPUT_JSON:\n" + json.dumps(payload, ensure_ascii=True)

    schema_path = root / args.schema
    normalized_schema = normalize_schema(
        json.loads(schema_path.read_text(encoding="utf-8-sig"))
    )
    normalized_schema_path = (
        root / "artifacts" / "agent" / "schema_experiment_plan.normalized.json"
    )
    normalized_schema_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_schema_path.write_text(
        json.dumps(normalized_schema, ensure_ascii=True, indent=2), encoding="utf-8"
    )

    out_path = (
        Path(args.out)
        if args.out
        else root
        / "artifacts"
        / "agent"
        / f"plan_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_codex(root, prompt, normalized_schema_path, out_path, args.profile)
    plan = json.loads(out_path.read_text(encoding="utf-8"))
    if plan.get("decision") != "do":
        reason = plan.get("reason", "").strip()
        if reason:
            reason = f"{reason} (overridden to do)"
        else:
            reason = "overridden to do"
        plan["decision"] = "do"
        plan["reason"] = reason
    if plan_needs_seed_override(plan, seed_ids, str(selected_seed["id"])):
        apply_seed_to_plan(
            plan,
            selected_seed,
            "auto-corrected from seed due to invalid or mismatched plan output",
        )
    run_id = plan.get("run_id")
    if run_id_is_valid(run_id):
        base_id = run_id
    else:
        base_id = build_run_id()
    plan["run_id"] = ensure_unique_run_id(existing_run_ids, base_id)
    ensure_eval_plan(plan)
    out_path.write_text(json.dumps(plan, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Wrote plan: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
