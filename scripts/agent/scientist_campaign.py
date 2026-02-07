#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception as exc:
    raise RuntimeError(
        "PyYAML is required for scientist campaign config. Install with: pip install pyyaml"
    ) from exc


def _parse_date(s: str) -> date:
    return datetime.strptime(str(s).strip(), "%Y-%m-%d").date()


def _fmt(d: date) -> str:
    return d.strftime("%Y-%m-%d")


@dataclass(frozen=True)
class StageRules:
    bootstrap_B: int
    ttl_sec: int
    min_bets: int
    min_days: int
    delta_roi_ci95_lower_gt: float
    p_one_sided_delta_le_0_max: float


@dataclass(frozen=True)
class CampaignPeriods:
    stage1_test_start: date
    stage1_test_end: date
    stage2_test_start: date
    stage2_test_end: date
    holdout_test_start: date
    holdout_test_end: date


@dataclass(frozen=True)
class CampaignConfig:
    campaign_id: str
    campaign_end_date: date
    total_shards: int
    train_days: int
    valid_days: int
    central_heartbeat_bucket_sec: int
    promotion_batch_size: int
    promotion_labels: list[str]
    periods: CampaignPeriods
    stage1: StageRules
    stage2: StageRules
    holdout: StageRules

    def stage_period(self, stage: str) -> tuple[str, str, str]:
        if stage == "stage1":
            return (
                "stage1_test",
                _fmt(self.periods.stage1_test_start),
                _fmt(self.periods.stage1_test_end),
            )
        if stage == "stage2":
            return (
                "stage2_test",
                _fmt(self.periods.stage2_test_start),
                _fmt(self.periods.stage2_test_end),
            )
        if stage == "holdout":
            return (
                "holdout_test",
                _fmt(self.periods.holdout_test_start),
                _fmt(self.periods.holdout_test_end),
            )
        raise ValueError(f"Unknown stage: {stage}")

    def build_splits_for_test_start(self, test_start: date) -> dict[str, str]:
        valid_end = test_start - timedelta(days=1)
        valid_start = valid_end - timedelta(days=self.valid_days - 1)
        train_end = valid_start - timedelta(days=1)
        train_start = train_end - timedelta(days=self.train_days - 1)
        return {
            "train_start": _fmt(train_start),
            "train_end": _fmt(train_end),
            "valid_start": _fmt(valid_start),
            "valid_end": _fmt(valid_end),
        }


def _get(d: dict[str, Any], key: str, default: Any) -> Any:
    v = d.get(key)
    return default if v is None else v


def _get_int(d: dict[str, Any], key: str, default: int) -> int:
    v = d.get(key)
    if v is None:
        return default
    return int(v)


def _get_float(d: dict[str, Any], key: str, default: float) -> float:
    v = d.get(key)
    if v is None:
        return default
    return float(v)


def _relativedelta_months(d: date, months: int) -> date:
    # Avoid extra dependency surprises: dateutil is already a pandas dependency.
    from dateutil.relativedelta import relativedelta  # type: ignore[import-not-found]

    return (d + relativedelta(months=months))


def load_campaign_config(path: Path) -> CampaignConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"Invalid campaign config YAML (expected mapping): {path}")

    campaign_id = str(raw.get("campaign_id") or "").strip()
    if not campaign_id:
        raise RuntimeError("campaign_id is required.")
    end_date = _parse_date(str(raw.get("campaign_end_date") or "").strip())

    sh = raw.get("sharding") if isinstance(raw.get("sharding"), dict) else {}
    total_shards = int(_get(sh, "total_shards", 1))
    if total_shards < 1:
        raise RuntimeError("sharding.total_shards must be >= 1")

    splits = raw.get("splits") if isinstance(raw.get("splits"), dict) else {}
    train_days = _get_int(splits, "train_days", 365 * 3)
    valid_days = _get_int(splits, "valid_days", 365)

    hb = raw.get("heartbeat") if isinstance(raw.get("heartbeat"), dict) else {}
    bucket = _get_int(hb, "central_bucket_sec", 1800)
    if bucket < 60:
        raise RuntimeError("heartbeat.central_bucket_sec must be >= 60")

    promo = raw.get("promotion") if isinstance(raw.get("promotion"), dict) else {}
    batch_size = _get_int(promo, "batch_size", 10)
    labels = promo.get("pr_labels")
    if not isinstance(labels, list) or not labels:
        labels = ["autogen", "promotion"]
    labels = [str(x).strip() for x in labels if str(x).strip()]

    per = raw.get("periods") if isinstance(raw.get("periods"), dict) else {}
    stage1_m = int(_get(per, "stage1_start_months", -24))
    stage2_m = int(_get(per, "stage2_start_months", -9))
    holdout_m = int(_get(per, "holdout_start_months", -3))

    b_stage1 = _relativedelta_months(end_date, stage1_m)
    b_stage2 = _relativedelta_months(end_date, stage2_m)
    b_holdout = _relativedelta_months(end_date, holdout_m)

    periods = CampaignPeriods(
        stage1_test_start=b_stage1,
        stage1_test_end=b_stage2 - timedelta(days=1),
        stage2_test_start=b_stage2,
        stage2_test_end=b_holdout - timedelta(days=1),
        holdout_test_start=b_holdout,
        holdout_test_end=end_date,
    )
    if not (periods.stage1_test_start <= periods.stage1_test_end):
        raise RuntimeError("Invalid stage1 period (start > end).")
    if not (periods.stage2_test_start <= periods.stage2_test_end):
        raise RuntimeError("Invalid stage2 period (start > end).")
    if not (periods.holdout_test_start <= periods.holdout_test_end):
        raise RuntimeError("Invalid holdout period (start > end).")
    if periods.stage1_test_end >= periods.stage2_test_start:
        raise RuntimeError("Stage1/Stage2 periods overlap.")
    if periods.stage2_test_end >= periods.holdout_test_start:
        raise RuntimeError("Stage2/Holdout periods overlap.")

    ev = raw.get("evaluation") if isinstance(raw.get("evaluation"), dict) else {}

    def _stage_rules(name: str, *, defaults: dict[str, Any]) -> StageRules:
        node = ev.get(name) if isinstance(ev.get(name), dict) else {}
        rule = node.get("promote_rule") if name == "stage1" else node.get("accept_rule")
        if not isinstance(rule, dict):
            rule = {}
        return StageRules(
            bootstrap_B=_get_int(node, "bootstrap_B", int(defaults["bootstrap_B"])),
            ttl_sec=_get_int(node, "ttl_sec", int(defaults["ttl_sec"])),
            min_bets=_get_int(node, "min_bets", int(defaults["min_bets"])),
            min_days=_get_int(node, "min_days", int(defaults["min_days"])),
            delta_roi_ci95_lower_gt=_get_float(
                rule, "delta_roi_ci95_lower_gt", float(defaults["delta_roi_ci95_lower_gt"])
            ),
            p_one_sided_delta_le_0_max=_get_float(
                rule,
                "p_one_sided_delta_le_0_max",
                float(defaults["p_one_sided_delta_le_0_max"]),
            ),
        )

    stage1 = _stage_rules(
        "stage1",
        defaults={
            "bootstrap_B": 300,
            "ttl_sec": 7200,
            "min_bets": 60,
            "min_days": 10,
            "delta_roi_ci95_lower_gt": -0.002,
            "p_one_sided_delta_le_0_max": 0.40,
        },
    )
    stage2 = _stage_rules(
        "stage2",
        defaults={
            "bootstrap_B": 2000,
            "ttl_sec": 7200,
            "min_bets": 120,
            "min_days": 20,
            "delta_roi_ci95_lower_gt": 0.0,
            "p_one_sided_delta_le_0_max": 0.20,
        },
    )
    holdout = _stage_rules(
        "holdout",
        defaults={
            "bootstrap_B": 2000,
            "ttl_sec": 7200,
            "min_bets": 120,
            "min_days": 20,
            "delta_roi_ci95_lower_gt": 0.0,
            "p_one_sided_delta_le_0_max": 0.20,
        },
    )

    return CampaignConfig(
        campaign_id=campaign_id,
        campaign_end_date=end_date,
        total_shards=total_shards,
        train_days=train_days,
        valid_days=valid_days,
        central_heartbeat_bucket_sec=bucket,
        promotion_batch_size=batch_size,
        promotion_labels=labels,
        periods=periods,
        stage1=stage1,
        stage2=stage2,
        holdout=holdout,
    )


def load_campaign_by_id(repo_root: Path, campaign_id: str) -> CampaignConfig:
    path = repo_root / "config" / "scientist_campaigns" / f"{campaign_id}.yml"
    if not path.exists():
        path = repo_root / "config" / "scientist_campaigns" / f"{campaign_id}.yaml"
    if not path.exists():
        raise RuntimeError(f"Campaign config not found: {campaign_id}")
    return load_campaign_config(path)


def asdict(cfg: CampaignConfig) -> dict[str, Any]:
    return dataclasses.asdict(cfg)

