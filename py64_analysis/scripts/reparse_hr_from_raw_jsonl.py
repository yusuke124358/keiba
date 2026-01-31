"""
旧形式JSONL（HRが _raw hex のみ）から、3連複/3連単の払戻・返還馬番を再パースしてDB投入する。

背景:
  - 以前に収集した RACE_*.jsonl では HR が `_raw`（hex文字列）で保存されていることがある
  - 現行ローダは `_raw` をスキップするため、fact_payout_trio / fact_payout_trifecta が 0件のままになる

使い方（例）:
  cd py64_analysis
  python scripts/reparse_hr_from_raw_jsonl.py --input-dir ../data/raw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable


def _ensure_import_path() -> None:
    project_root = Path(__file__).resolve().parents[2]  # keiba/
    src = project_root / "py64_analysis" / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def _iter_race_jsonl_files(input_dir: Path) -> list[Path]:
    files: list[Path] = []
    # 直下
    files.extend(sorted(input_dir.glob("RACE_*.jsonl")))
    # 既存のサブディレクトリ（例: c2_2020_2023）
    for sub in input_dir.iterdir():
        if not sub.is_dir():
            continue
        files.extend(sorted(sub.glob("RACE_*.jsonl")))
        files.extend(sorted((sub / "old").glob("RACE_*.jsonl")))
    # 重複排除
    uniq: dict[str, Path] = {}
    for p in files:
        uniq[str(p.resolve())] = p
    return list(uniq.values())


def _chunks(it: Iterable[dict], size: int) -> Iterable[list[dict]]:
    buf: list[dict] = []
    for x in it:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def main() -> None:
    _ensure_import_path()
    import json

    from keiba.db.loader import get_session, load_hr_exotics_payout_records
    from py32_fetcher.parsers.race_parser import parse_hr_pay

    p = argparse.ArgumentParser(description="Re-parse HR (_raw hex) from old JSONL and load trio/trifecta payouts")
    p.add_argument("--input-dir", type=Path, required=True, help="data/raw ディレクトリ（RACE_*.jsonl を探す）")
    p.add_argument("--batch-size", type=int, default=5000, help="DB投入のバッチサイズ")
    p.add_argument("--max-files", type=int, default=0, help="デバッグ用: 先頭Nファイルだけ処理（0なら全件）")
    p.add_argument("--lock-timeout-seconds", type=float, default=0)
    p.add_argument("--lock-poll-seconds", type=float, default=0.25)
    p.add_argument("--no-lock", action="store_true")
    args = p.parse_args()

    if not args.no_lock:
        if args.lock_timeout_seconds < 0:
            raise SystemExit("--lock-timeout-seconds must be >= 0")
        if args.lock_poll_seconds <= 0:
            raise SystemExit("--lock-poll-seconds must be > 0")

    def _run() -> None:
        input_dir = args.input_dir
        files = _iter_race_jsonl_files(input_dir)
        if args.max_files and args.max_files > 0:
            files = files[: int(args.max_files)]
    
        print(f"Found RACE jsonl files: {len(files)}")
        for f in files:
            print(" -", f)
    
        total_hr_lines = 0
        total_parsed = 0
        total_inserted = 0
        total_skipped_no_raw = 0
        total_skipped_parse_fail = 0
    
        sess = get_session()
        try:
            for idx, fp in enumerate(files, 1):
                print(f"\n[{idx}/{len(files)}] scanning {fp.name} ...")
                parsed_records: list[dict] = []
    
                with fp.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        if obj.get("record_type") != "HR":
                            continue
                        total_hr_lines += 1
                        if "_raw" not in obj:
                            total_skipped_no_raw += 1
                            continue
    
                        try:
                            raw_hex = obj["_raw"]
                            raw_bytes = bytes.fromhex(raw_hex)
                        except Exception:
                            total_skipped_parse_fail += 1
                            continue
    
                        race_id = obj.get("race_id")
                        if not race_id or len(race_id) < 8:
                            total_skipped_parse_fail += 1
                            continue
                        race_date = race_id[:8]
    
                        parsed = parse_hr_pay(raw_bytes, race_id=race_id, race_date=race_date)
                        if not parsed:
                            total_skipped_parse_fail += 1
                            continue
    
                        # loaderに渡す形に整形（_rawは付けない）
                        rec = {
                            "record_type": "HR",
                            "race_id": race_id,
                            **parsed,
                        }
                        parsed_records.append(rec)
                        total_parsed += 1
    
                        # メモリ節約: ある程度溜まったらDB投入
                        if len(parsed_records) >= int(args.batch_size):
                            inserted = load_hr_exotics_payout_records(sess, parsed_records)
                            sess.commit()
                            total_inserted += int(inserted)
                            parsed_records = []
    
                # file end: flush
                if parsed_records:
                    inserted = load_hr_exotics_payout_records(sess, parsed_records)
                    sess.commit()
                    total_inserted += int(inserted)
    
            print("\n=== summary ===")
            print("hr_lines:", total_hr_lines)
            print("parsed:", total_parsed)
            print("inserted_rows(total refund+payout):", total_inserted)
            print("skipped_no_raw:", total_skipped_no_raw)
            print("skipped_parse_fail:", total_skipped_parse_fail)
        finally:
            sess.close()
    

    if args.no_lock:
        print("DB write lock: DISABLED (--no-lock)")
        _run()
        return

    from keiba.utils.locks import db_write_lock, default_db_lock_path

    lock_path = default_db_lock_path()
    print(
        f"Acquiring DB write lock: {lock_path} "
        f"(timeout={args.lock_timeout_seconds}, poll={args.lock_poll_seconds})"
    )
    with db_write_lock(timeout_seconds=args.lock_timeout_seconds, poll_seconds=args.lock_poll_seconds):
        print("Acquired DB write lock")
        try:
            _run()
        finally:
            print("Released DB write lock")

if __name__ == "__main__":
    main()








