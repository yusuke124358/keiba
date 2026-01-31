from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class MetricsV01:
    schema_version: str = "0.1"
    run_kind: str = ""
    run_dir: str = ""
    name: Optional[str] = None
    generated_at: Optional[str] = None
    git_commit: Optional[str] = None
    config_used_path: Optional[str] = None
    config_hash_sha256: Optional[str] = None
    data_cutoff: Dict[str, Optional[str]] = field(default_factory=dict)
    split: Dict[str, Dict[str, Optional[Any]]] = field(default_factory=dict)
    universe: Dict[str, Any] = field(default_factory=dict)
    betting: Dict[str, Any] = field(default_factory=dict)
    prob_variant_used: str = "unknown"
    backtest: Dict[str, Optional[Any]] = field(default_factory=dict)
    pred_quality: Dict[str, Optional[Any]] = field(default_factory=dict)
    step14: Optional[Dict[str, Optional[Any]]] = None
    incomparable_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_metrics_v01(data: Dict[str, Any]) -> None:
    required = [
        "schema_version",
        "run_kind",
        "run_dir",
        "data_cutoff",
        "split",
        "betting",
        "backtest",
        "pred_quality",
        "prob_variant_used",
        "incomparable_reasons",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")
