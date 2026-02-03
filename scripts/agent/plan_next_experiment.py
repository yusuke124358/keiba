#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
from pathlib import Path


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
    seed_text = (root / args.seed).read_text(encoding="utf-8")
    runs = load_jsons(root / args.runs)
    recent = runs[-10:] if len(runs) > 10 else runs

    knowledge_dir = root / args.knowledge
    knowledge_files = []
    if knowledge_dir.exists():
        for p in knowledge_dir.rglob("*.md"):
            knowledge_files.append(str(p.relative_to(root)))

    prompt_text = (root / args.prompt).read_text(encoding="utf-8")
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
    if not plan.get("run_id"):
        plan["run_id"] = dt.datetime.utcnow().strftime("RUN_%Y%m%d_%H%M%S")
    out_path.write_text(json.dumps(plan, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Wrote plan: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
