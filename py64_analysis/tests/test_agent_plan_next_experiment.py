from pathlib import Path
import importlib.util


def _load_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "scripts" / "agent" / "plan_next_experiment.py"
    spec = importlib.util.spec_from_file_location(
        "plan_next_experiment", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def _base_plan(seed_id="SEED-001"):
    return {
        "seed_id": seed_id,
        "title": "Valid title",
        "hypothesis": "Valid hypothesis",
        "change_scope": "config/selection thresholds",
        "acceptance_criteria": ["ROI >= baseline"],
        "metrics": ["roi"],
        "risk_level": "low",
        "max_diff_size": 10,
    }


def test_select_seed_prefers_unused():
    seeds = [{"id": "SEED-001"}, {"id": "SEED-002"}]
    selected = MODULE.select_seed(seeds, {"SEED-001"})
    assert selected["id"] == "SEED-002"


def test_plan_needs_seed_override_valid_plan():
    plan = _base_plan()
    assert (
        MODULE.plan_needs_seed_override(plan, {"SEED-001"}, "SEED-001") is False
    )


def test_plan_needs_seed_override_tbd():
    plan = _base_plan()
    plan["title"] = "TBD experiment"
    assert MODULE.plan_needs_seed_override(plan, {"SEED-001"}, "SEED-001") is True


def test_plan_needs_seed_override_seed_mismatch():
    plan = _base_plan(seed_id="SEED-002")
    assert (
        MODULE.plan_needs_seed_override(
            plan, {"SEED-001", "SEED-002"}, "SEED-001"
        )
        is True
    )


def test_ensure_unique_run_id_suffix():
    existing = {"exp_20260204_123456"}
    assert (
        MODULE.ensure_unique_run_id(existing, "exp_20260204_123456")
        == "exp_20260204_123456_001"
    )
