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


def load_runs(runs_dir: Path) -> list[dict]:
    runs = []
    for p in sorted(runs_dir.glob("*.json")):
        try:
            runs.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return runs


def latest_checkpoint(checkpoints_dir: Path) -> dict | None:
    items = sorted(checkpoints_dir.glob("*.json"))
    if not items:
        return None
    try:
        return json.loads(items[-1].read_text(encoding="utf-8"))
    except Exception:
        return None


def run_codex(
    root: Path,
    prompt_path: Path,
    payload: dict,
    schema_path: Path,
    out_path: Path,
    profile: str,
) -> None:
    codex_bin = find_codex_bin()
    prompt_text = prompt_path.read_text(encoding="utf-8")
    prompt = prompt_text + "\n\nINPUT_JSON:\n" + json.dumps(payload, ensure_ascii=True)
    cmd = [
        codex_bin,
        "exec",
        "--profile",
        profile,
        "--full-auto",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(out_path),
    ]
    log_path = out_path.with_suffix(".log")
    with open(log_path, "w", encoding="utf-8") as log:
        result = subprocess.run(
            cmd + [prompt], cwd=root, stdout=log, stderr=log, text=True
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"codex exec failed with code {result.returncode}. See {log_path}"
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="experiments/runs")
    p.add_argument("--checkpoints", default="experiments/checkpoints")
    p.add_argument("--reports", default="reports/checkpoints")
    p.add_argument("--prompt", default="prompts/agent/summarizer_50.md")
    p.add_argument("--schema", default="schemas/agent/checkpoint_summary.schema.json")
    p.add_argument("--profile", default="agent_loop")
    p.add_argument("--threshold", type=int, default=50)
    args = p.parse_args()

    root = repo_root()
    runs_dir = root / args.runs
    checkpoints_dir = root / args.checkpoints
    reports_dir = root / args.reports
    runs = load_runs(runs_dir)

    last_ck = latest_checkpoint(checkpoints_dir)
    start_idx = int(last_ck.get("run_index", 0)) if last_ck else 0
    delta = len(runs) - start_idx
    if delta < args.threshold:
        print(f"Checkpoint not due: delta={delta}, threshold={args.threshold}")
        return 0

    checkpoint_id = dt.datetime.utcnow().strftime("ckpt_%Y%m%d_%H%M%S")
    payload = {
        "checkpoint_id": checkpoint_id,
        "runs": runs[start_idx:],
        "total_runs": len(runs),
        "delta_runs": delta,
    }
    out_path = checkpoints_dir / f"{checkpoint_id}.json"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    run_codex(
        root, root / args.prompt, payload, root / args.schema, out_path, args.profile
    )

    summary = json.loads(out_path.read_text(encoding="utf-8"))
    report_path = reports_dir / f"{checkpoint_id}.md"
    report = [
        f"# Checkpoint {checkpoint_id}",
        "",
        f"- Range start: {summary['range_start']}",
        f"- Range end: {summary['range_end']}",
        f"- Total runs: {summary['total_runs']}",
        f"- Pass: {summary['passes']}",
        f"- Fail: {summary['fails']}",
        f"- Inconclusive: {summary['inconclusive']}",
        "",
        "## Top wins",
        "\n".join(f"- {x}" for x in summary["top_wins"]) or "- (none)",
        "",
        "## Top risks",
        "\n".join(f"- {x}" for x in summary["top_risks"]) or "- (none)",
        "",
        "## Next actions",
        "\n".join(f"- {x}" for x in summary["next_actions"]) or "- (none)",
    ]
    report_path.write_text("\n".join(report), encoding="utf-8")

    meta = {
        "checkpoint_id": checkpoint_id,
        "run_index": len(runs),
        "created_at": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    meta_path = checkpoints_dir / f"{checkpoint_id}.meta.json"
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8"
    )

    mem = root / "memory.md"
    if mem.exists():
        with mem.open("a", encoding="utf-8") as f:
            f.write(
                f"\n\nCheckpoint {checkpoint_id}: {summary['range_start']} to {summary['range_end']}\n"
            )
            for line in summary["top_wins"][:3]:
                f.write(f"- {line}\n")

    subprocess.run(["git", "add", "-A"], cwd=root, check=False)
    subprocess.run(
        ["git", "commit", "-m", f"agent: checkpoint {checkpoint_id}"],
        cwd=root,
        check=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
