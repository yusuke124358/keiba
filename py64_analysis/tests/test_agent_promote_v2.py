from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_module():
    root = Path(__file__).resolve().parents[2]
    agent_dir = root / "scripts" / "agent"
    if str(agent_dir) not in sys.path:
        sys.path.insert(0, str(agent_dir))
    module_path = root / "scripts" / "agent" / "promote_v2.py"
    spec = importlib.util.spec_from_file_location("promote_v2", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def test_main_requires_campaign_arg(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "argv", ["promote_v2.py"])
    with pytest.raises(SystemExit) as exc:
        MODULE.main()
    assert exc.value.code == 2


def test_main_requires_state_repo_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(sys, "argv", ["promote_v2.py", "--campaign", "camp_x"])
    monkeypatch.delenv("KEIBA_STATE_REPO_DIR", raising=False)
    monkeypatch.setattr(MODULE, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        MODULE, "load_campaign_by_id", lambda _root, _cid: SimpleNamespace(campaign_id="camp_x")
    )
    with pytest.raises(RuntimeError, match="state-repo-dir"):
        MODULE.main()


def test_dry_run_prints_would_promote(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    root = tmp_path
    state_repo_dir = tmp_path / "state_repo"
    state_repo_dir.mkdir(parents=True, exist_ok=True)

    # Minimal seed file.
    seeds_path = root / "experiments" / "seed_hypotheses.yaml"
    seeds_path.parent.mkdir(parents=True, exist_ok=True)
    seeds_path.write_text(
        "\n".join(
            [
                "- id: SEED-001",
                "  title: Test Seed",
                "  hypothesis: test",
                "  change_scope: config",
                "  acceptance_criteria: []",
                "  metrics: []",
                "  risk_level: low",
                "  max_diff_size: 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # Patch file in the state worktree.
    patch_rel = "state/campaigns/camp_x/patches/SEED-001/abc.diff"
    patch_path = state_repo_dir / Path(patch_rel)
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_bytes(b"diff --git a/a b/a\n")
    patch_sha256 = hashlib.sha256(patch_path.read_bytes()).hexdigest()

    events = [
        {
            "event_type": "finished",
            "stage": "holdout",
            "result": "success",
            "decision": "accept",
            "seed_id": "SEED-001",
            "run_id": "camp_x-holdout-SEED-001-a1",
            "finished_at": "2026-02-07T00:00:00Z",
            "patch_ref": {"path": patch_rel, "sha256": patch_sha256},
        }
    ]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promote_v2.py",
            "--campaign",
            "camp_x",
            "--state-repo-dir",
            str(state_repo_dir),
            "--dry-run",
        ],
    )
    monkeypatch.setattr(MODULE, "repo_root", lambda: root)
    monkeypatch.setattr(
        MODULE, "load_campaign_by_id", lambda _root, _cid: SimpleNamespace(campaign_id="camp_x")
    )
    monkeypatch.setattr(MODULE, "ensure_state_worktree", lambda **_kwargs: state_repo_dir)
    monkeypatch.setattr(MODULE, "load_events", lambda **_kwargs: events)
    monkeypatch.setattr(MODULE, "_branch_exists_remote", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(MODULE, "_pr_exists_for_branch", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        MODULE,
        "_append_promotion_event",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected promotion event in dry-run")
        ),
    )

    rc = MODULE.main()
    assert rc == 0

    out_lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(out_lines) == 2
    would = json.loads(out_lines[0])
    assert would["action"] == "would_promote"
    assert would["campaign_id"] == "camp_x"
    assert would["seed_id"] == "SEED-001"
    assert would["branch"].startswith("promotion/camp_x/SEED-001/")
    summary = json.loads(out_lines[1])
    assert summary["campaign_id"] == "camp_x"
    assert summary["created_prs"] == 0


def test_hash_mismatch_records_skip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    root = tmp_path
    state_repo_dir = tmp_path / "state_repo"
    state_repo_dir.mkdir(parents=True, exist_ok=True)

    seeds_path = root / "experiments" / "seed_hypotheses.yaml"
    seeds_path.parent.mkdir(parents=True, exist_ok=True)
    seeds_path.write_text("- id: SEED-001\n  title: Test Seed\n", encoding="utf-8")

    patch_rel = "state/campaigns/camp_x/patches/SEED-001/abc.diff"
    patch_path = state_repo_dir / Path(patch_rel)
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_bytes(b"diff --git a/a b/a\n")

    events = [
        {
            "event_type": "finished",
            "stage": "holdout",
            "result": "success",
            "decision": "accept",
            "seed_id": "SEED-001",
            "run_id": "camp_x-holdout-SEED-001-a1",
            "finished_at": "2026-02-07T00:00:00Z",
            "patch_ref": {"path": patch_rel, "sha256": "deadbeef"},
        }
    ]

    calls: list[dict[str, str]] = []

    def _record(**kwargs):
        calls.append({k: str(v) for k, v in kwargs.items()})

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promote_v2.py",
            "--campaign",
            "camp_x",
            "--state-repo-dir",
            str(state_repo_dir),
            "--dry-run",
        ],
    )
    monkeypatch.setattr(MODULE, "repo_root", lambda: root)
    monkeypatch.setattr(
        MODULE, "load_campaign_by_id", lambda _root, _cid: SimpleNamespace(campaign_id="camp_x")
    )
    monkeypatch.setattr(MODULE, "ensure_state_worktree", lambda **_kwargs: state_repo_dir)
    monkeypatch.setattr(MODULE, "load_events", lambda **_kwargs: events)
    monkeypatch.setattr(MODULE, "_append_promotion_event", _record)

    rc = MODULE.main()
    assert rc == 0
    assert any(c.get("event_type") == "promotion_skipped_hash_mismatch" for c in calls)
