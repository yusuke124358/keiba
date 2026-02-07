#!/usr/bin/env python3
from __future__ import annotations

import hashlib


def parse_shard(text: str) -> tuple[int, int]:
    raw = str(text or "").strip()
    if "/" not in raw:
        raise ValueError("shard must be in i/n form (e.g. 0/2)")
    a, b = raw.split("/", 1)
    i = int(a)
    n = int(b)
    if n < 1:
        raise ValueError("shard total must be >= 1")
    if i < 0 or i >= n:
        raise ValueError("shard index must satisfy 0 <= i < n")
    return i, n


def assigned_shard(seed_id: str, total_shards: int) -> int:
    if total_shards < 1:
        raise ValueError("total_shards must be >= 1")
    sid = str(seed_id).encode("utf-8")
    h = hashlib.sha1(sid).hexdigest()
    return int(h, 16) % int(total_shards)
