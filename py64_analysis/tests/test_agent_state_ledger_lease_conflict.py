from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_module():
    root = Path(__file__).resolve().parents[2]
    agent_dir = root / "scripts" / "agent"
    if str(agent_dir) not in sys.path:
        sys.path.insert(0, str(agent_dir))
    module_path = root / "scripts" / "agent" / "state_ledger.py"
    spec = importlib.util.spec_from_file_location("state_ledger", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def test_existing_lease_idempotent_allows_same_owner_and_run_id(tmp_path: Path):
    out_path = tmp_path / "lease.json"
    existing = {"event_type": "lease_acquired", "owner": "owner-a", "run_id": "run-1"}
    new = {"event_type": "lease_acquired", "owner": "owner-a", "run_id": "run-1"}
    MODULE._ensure_existing_lease_is_idempotent(existing=existing, new_event=new, out_path=out_path)


def test_existing_lease_idempotent_rejects_different_owner(tmp_path: Path):
    out_path = tmp_path / "lease.json"
    existing = {"event_type": "lease_acquired", "owner": "owner-a", "run_id": "run-1"}
    new = {"event_type": "lease_acquired", "owner": "owner-b", "run_id": "run-1"}
    with pytest.raises(MODULE.LeaseAlreadyHeldError):
        MODULE._ensure_existing_lease_is_idempotent(
            existing=existing, new_event=new, out_path=out_path
        )


def test_existing_lease_idempotent_rejects_different_run_id(tmp_path: Path):
    out_path = tmp_path / "lease.json"
    existing = {"event_type": "lease_acquired", "owner": "owner-a", "run_id": "run-1"}
    new = {"event_type": "lease_acquired", "owner": "owner-a", "run_id": "run-2"}
    with pytest.raises(MODULE.LeaseAlreadyHeldError):
        MODULE._ensure_existing_lease_is_idempotent(
            existing=existing, new_event=new, out_path=out_path
        )
