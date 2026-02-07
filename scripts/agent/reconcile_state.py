#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

from state_store import default_state_paths, file_lock, load_state, save_state


def _parse_iso(ts: str) -> datetime | None:
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ttl-sec", type=int, required=True)
    p.add_argument("--max-tries", type=int, default=3)
    args = p.parse_args()

    state = load_state()
    items: dict[str, Any] = state.get("items") if isinstance(state.get("items"), dict) else {}

    now = datetime.now(timezone.utc)
    changed = 0
    _, lock_path = default_state_paths()
    with file_lock(lock_path):
        for k, v in list(items.items()):
            if not isinstance(v, dict):
                continue
            status = str(v.get("status") or "").lower()
            if status == "paused":
                continue
            if status != "running":
                continue
            hb = str(v.get("last_heartbeat_at") or "")
            hb_dt = _parse_iso(hb) or _parse_iso(str(v.get("started_at") or "")) or now
            age = (now - hb_dt).total_seconds()
            if age <= float(args.ttl_sec):
                continue
            tries = int(v.get("tries") or 0)
            tries += 1
            v["tries"] = tries
            v["last_error"] = f"stale_running age_sec={int(age)} ttl_sec={args.ttl_sec}"
            if tries >= int(args.max_tries):
                v["status"] = "failed"
            else:
                v["status"] = "todo"
            items[k] = v
            changed += 1

        state["items"] = items
        if changed:
            save_state(state)

    print(f"reconciled={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
