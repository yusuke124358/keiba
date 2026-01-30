from __future__ import annotations

import argparse
from pathlib import Path


def _decode_bytes(data: bytes) -> tuple[str, str]:
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16"), "utf-16"
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig"), "utf-8-sig"
    if b"\x00" in data:
        try:
            return data.decode("utf-16"), "utf-16"
        except Exception:
            pass
    for enc in ("utf-8-sig", "cp932"):
        try:
            return data.decode(enc), enc
        except Exception:
            continue
    return data.decode("utf-8", errors="replace"), "utf-8-replace"


def _iter_targets(root: Path, exts: set[str]) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def main() -> int:
    ap = argparse.ArgumentParser(description="Normalize text files to UTF-8 (no BOM).")
    ap.add_argument("--root", required=True)
    ap.add_argument("--exts", default=".txt,.md")
    ap.add_argument("--log", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    exts = {e.strip().lower() for e in args.exts.split(",") if e.strip()}
    if not root.exists():
        raise SystemExit(f"root not found: {root}")

    log_lines: list[str] = []
    failures = 0
    for path in _iter_targets(root, exts):
        data = path.read_bytes()
        try:
            text, enc = _decode_bytes(data)
        except Exception as exc:
            failures += 1
            log_lines.append(f"failed {path}: {exc}")
            continue

        has_replacement = "\ufffd" in text
        encoded = text.encode("utf-8")
        changed = encoded != data
        if changed and not args.dry_run:
            path.write_bytes(encoded)
        status = "converted" if changed else "ok"
        note = " replacement_char" if has_replacement else ""
        log_lines.append(f"{status} {path} from {enc}{note}")

    if args.log:
        Path(args.log).write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    else:
        for line in log_lines:
            print(line)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
