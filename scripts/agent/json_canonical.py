#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any


def _assert_canonical(obj: Any, path: str = "$") -> None:
    if obj is None:
        return
    if isinstance(obj, (str, bool, int)):
        return
    if isinstance(obj, float):
        # Floats are allowed in payloads, but should not be used for identity hashes
        # unless they are string-normalized first.
        raise TypeError(f"float is not allowed in canonical identity payload: {path}")
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            _assert_canonical(item, f"{path}[{i}]")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"non-string key is not allowed in canonical payload: {path}")
            _assert_canonical(v, f"{path}.{k}")
        return
    raise TypeError(f"unsupported type in canonical payload: {path} ({type(obj).__name__})")


def canonical_dumps(obj: Any) -> str:
    """
    Deterministic JSON for event_id hashing:
    - sort keys
    - stable separators
    - ASCII output
    """
    _assert_canonical(obj)
    return json.dumps(
        obj,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

