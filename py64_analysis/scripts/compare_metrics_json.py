from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _eq(a, b, tol: float = 1e-9) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return str(a) == str(b)


def _get(data: dict, path: str, default=None):
    cur = data
    for key in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def _pick_metric_block(metrics: dict, use_step14: bool) -> dict:
    if use_step14:
        step14 = metrics.get("step14")
        if isinstance(step14, dict) and step14.get("roi") is not None:
            return step14
    return metrics.get("backtest", {})


def _compare_required(baseline: dict, candidate: dict) -> list[str]:
    reasons: list[str] = []
    b_schema = baseline.get("schema_version")
    c_schema = candidate.get("schema_version")
    if b_schema and c_schema and str(b_schema) != str(c_schema):
        reasons.append(f"schema_version mismatch ({b_schema} vs {c_schema})")
    b_kind = baseline.get("run_kind")
    c_kind = candidate.get("run_kind")
    if b_kind and c_kind and str(b_kind) != str(c_kind):
        reasons.append(f"run_kind mismatch ({b_kind} vs {c_kind})")
    for key in ["train", "valid", "test"]:
        for field in ["start", "end"]:
            b = _get(baseline, f"split.{key}.{field}")
            c = _get(candidate, f"split.{key}.{field}")
            if b is not None and c is not None and str(b) != str(c):
                reasons.append(f"split.{key}.{field} mismatch ({b} vs {c})")

    b_buy = _get(baseline, "betting.buy_t_minus_minutes")
    c_buy = _get(candidate, "betting.buy_t_minus_minutes")
    if b_buy is not None and c_buy is not None and int(b_buy) != int(c_buy):
        reasons.append(f"buy_t_minus_minutes mismatch ({b_buy} vs {c_buy})")

    b_prob = baseline.get("prob_variant_used")
    c_prob = candidate.get("prob_variant_used")
    if b_prob and c_prob and b_prob != c_prob:
        reasons.append(f"prob_variant_used mismatch ({b_prob} vs {c_prob})")

    b_method = _get(baseline, "betting.market_prob_method")
    c_method = _get(candidate, "betting.market_prob_method")
    if b_method is not None and c_method is not None and str(b_method) != str(c_method):
        reasons.append(f"market_prob_method mismatch ({b_method} vs {c_method})")

    b_mode = _get(baseline, "betting.market_prob_mode")
    c_mode = _get(candidate, "betting.market_prob_mode")
    if b_mode is not None and c_mode is not None and str(b_mode) != str(c_mode):
        reasons.append(f"market_prob_mode mismatch ({b_mode} vs {c_mode})")

    b_uni = baseline.get("universe") or {}
    c_uni = candidate.get("universe") or {}
    for field in ["track_codes", "require_results", "require_ts_win", "exclude_race_ids_hash"]:
        b = b_uni.get(field)
        c = c_uni.get(field)
        if b is not None and c is not None and b != c:
            reasons.append(f"universe.{field} mismatch")

    b_close = _get(baseline, "betting.closing_odds_multiplier")
    c_close = _get(candidate, "betting.closing_odds_multiplier")
    if b_close is not None and c_close is not None and not _eq(b_close, c_close):
        reasons.append(f"closing_odds_multiplier mismatch ({b_close} vs {c_close})")

    # data_cutoff checks (strict: no mixed presence)
    b_db = _get(baseline, "data_cutoff.db_max_race_date")
    c_db = _get(candidate, "data_cutoff.db_max_race_date")
    if (b_db is None) != (c_db is None):
        reasons.append("data_cutoff.db_max_race_date missing on one side")
    elif b_db and c_db and str(b_db) != str(c_db):
        reasons.append(f"data_cutoff.db_max_race_date mismatch ({b_db} vs {c_db})")
    else:
        b_raw = _get(baseline, "data_cutoff.raw_max_mtime")
        c_raw = _get(candidate, "data_cutoff.raw_max_mtime")
        if (b_raw is None) != (c_raw is None):
            reasons.append("data_cutoff.raw_max_mtime missing on one side")
        elif b_raw and c_raw and str(b_raw) != str(c_raw):
            reasons.append(f"data_cutoff.raw_max_mtime mismatch ({b_raw} vs {c_raw})")
        elif b_raw is None and c_raw is None and b_db is None and c_db is None:
            reasons.append("data_cutoff missing (db and raw)")

    return reasons


def main() -> None:
    p = argparse.ArgumentParser(description="Compare baseline vs candidate metrics.json")
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--gates", type=Path, default=Path("config/gates/default.yaml"))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    baseline = _load_json(args.baseline)
    candidate = _load_json(args.candidate)
    gates = _load_yaml(args.gates) if args.gates else {}

    reasons = _compare_required(baseline, candidate)
    comparable = len(reasons) == 0

    use_step14 = bool(gates.get("use_step14_if_present", True))
    step14_ok = (
        use_step14
        and isinstance(baseline.get("step14"), dict)
        and isinstance(candidate.get("step14"), dict)
        and baseline["step14"].get("roi") is not None
        and candidate["step14"].get("roi") is not None
    )
    base_metrics = _pick_metric_block(baseline, step14_ok)
    cand_metrics = _pick_metric_block(candidate, step14_ok)

    min_n_bets = int(gates.get("min_n_bets", 0))
    roi_tol = float(gates.get("roi_tolerance", 0.0))
    dd_tol = float(gates.get("dd_tolerance", 0.0))

    gate_results = {
        "min_n_bets": {"threshold": min_n_bets, "baseline": None, "candidate": None, "pass": None},
        "roi_non_degradation": {"tolerance": roi_tol, "baseline": None, "candidate": None, "pass": None},
        "max_drawdown_non_worsening": {"tolerance": dd_tol, "baseline": None, "candidate": None, "pass": None},
    }

    decision = "incomparable"
    exit_code = 2
    if comparable:
        b_n = base_metrics.get("n_bets")
        c_n = cand_metrics.get("n_bets")
        gate_results["min_n_bets"]["baseline"] = b_n
        gate_results["min_n_bets"]["candidate"] = c_n
        if c_n is None:
            gate_results["min_n_bets"]["pass"] = False
        else:
            gate_results["min_n_bets"]["pass"] = int(c_n) >= min_n_bets

        b_roi = base_metrics.get("roi")
        c_roi = cand_metrics.get("roi")
        gate_results["roi_non_degradation"]["baseline"] = b_roi
        gate_results["roi_non_degradation"]["candidate"] = c_roi
        if b_roi is None or c_roi is None:
            gate_results["roi_non_degradation"]["pass"] = False
        else:
            gate_results["roi_non_degradation"]["pass"] = float(c_roi) >= (float(b_roi) - roi_tol)

        b_dd = base_metrics.get("max_drawdown")
        c_dd = cand_metrics.get("max_drawdown")
        gate_results["max_drawdown_non_worsening"]["baseline"] = b_dd
        gate_results["max_drawdown_non_worsening"]["candidate"] = c_dd
        if b_dd is None or c_dd is None:
            gate_results["max_drawdown_non_worsening"]["pass"] = False
        else:
            gate_results["max_drawdown_non_worsening"]["pass"] = float(c_dd) <= (float(b_dd) + dd_tol)

        if all(v["pass"] for v in gate_results.values()):
            decision = "pass"
            exit_code = 0
        else:
            decision = "fail"
            exit_code = 1

    comparison = {
        "baseline": {
            "path": str(args.baseline),
            "run_dir": baseline.get("run_dir"),
        },
        "candidate": {
            "path": str(args.candidate),
            "run_dir": candidate.get("run_dir"),
        },
        "comparable": comparable,
        "incomparable_reasons": reasons,
        "use_step14_if_present": use_step14,
        "step14_used": step14_ok,
        "gates": gate_results,
        "decision": decision,
    }

    out_path = args.out
    if out_path is None:
        out_path = args.candidate.parent / "comparison.json"
    out_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"decision: {decision}")
    if reasons:
        print("incomparable_reasons:")
        for r in reasons:
            print(f"- {r}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
