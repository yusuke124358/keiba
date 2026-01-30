"""
速報/レースキー指定で複数レースをまとめて取得するユーティリティ。

背景:
  環境によっては 0B41/0B42 を JVOpen で扱うと -111 になるケースがあるため、
  MVPの検証用途として race_id を指定して JVRTOpen で取得する経路を用意する。

注意:
  - 自動投票はしない（買い目提示まで）
  - JRA-VAN規約に従う
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# プロジェクトルート（keiba/）をsys.pathに追加（直接実行でも import が通るように）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from py32_fetcher.jvlink_client import JVLinkClient
from py32_fetcher.parsers import parse_record


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _read_race_ids(path: Path) -> list[str]:
    race_ids: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            race_ids.append(s)
    return race_ids


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "completed_race_ids": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_state_atomic(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def bulk_fetch_realtime(
    data_spec: str,
    race_ids: list[str],
    output_dir: Path,
    software_id: str,
    service_key: str | None = None,
    sleep_sec: float = 0.0,
    chunk_size: int = 50,
    resume: bool = False,
    state_file: Path | None = None,
) -> dict:
    """
    複数レースを JVRTOpen で取得し、JSONLに保存する。

    - chunk_size > 0 の場合、Nレースごとにファイル分割
    - resume=True の場合、state_file に記録された完了レースをスキップ
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if state_file is None:
        state_file = PROJECT_ROOT / "data" / "state" / f"bulk_fetch_{data_spec}.json"
    state = _load_state(state_file)
    if state.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported state schema_version: {state.get('schema_version')}")
    if state.get("data_spec") and state.get("data_spec") != data_spec:
        raise RuntimeError(f"State data_spec mismatch: {state.get('data_spec')} != {data_spec}")
    state["data_spec"] = data_spec

    completed = set(state.get("completed_race_ids") or [])
    if not resume:
        completed = set()
        state["completed_race_ids"] = []

    client = JVLinkClient(software_id, service_key=service_key)

    race_ok = 0
    race_err = 0
    record_count = 0

    output_files: list[str] = []

    # resume時は「未処理だけ」に絞る（全て完了済みなら即終了）
    pending_race_ids = [rid for rid in race_ids if rid not in completed] if resume else list(race_ids)
    races_skipped = len(race_ids) - len(pending_race_ids)
    if not pending_race_ids:
        return {
            "data_spec": data_spec,
            "races_total": len(race_ids),
            "races_skipped": races_skipped,
            "races_ok": 0,
            "races_err": 0,
            "records": 0,
            "output_files": [],
            "state_file": str(state_file),
        }

    def _open_chunk_file(part_idx: int):
        if chunk_size and chunk_size > 0:
            p = output_dir / f"{data_spec}_bulk_{ts}_part{part_idx:04d}.jsonl"
        else:
            p = output_dir / f"{data_spec}_bulk_{ts}.jsonl"
        output_files.append(str(p))
        return p, open(p, "w", encoding="utf-8")

    part_idx = 1
    current_path, f = _open_chunk_file(part_idx)
    races_in_part = 0
    records_in_part = 0

    def _close_part_file() -> None:
        """partファイルを閉じ、空なら削除して output_files からも除外する。"""
        nonlocal current_path, f, records_in_part, output_files
        try:
            f.close()
        except Exception:
            pass

        # 成功レコードが1件も無いchunkは空ファイルになるので削除する
        if records_in_part == 0:
            try:
                if current_path.exists():
                    current_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete empty output file: {current_path} ({e})")
            # output_files からも除外（末尾にあるはず）
            if output_files and output_files[-1] == str(current_path):
                output_files.pop()

        records_in_part = 0

    try:
        for i, race_id in enumerate(pending_race_ids, start=1):
            try:
                logger.info(f"[{i}/{len(pending_race_ids)}] opening {data_spec} for race_id={race_id}")
                result = client.open_realtime(data_spec, race_id)

                if isinstance(result, tuple):
                    ret_code = result[0]
                else:
                    ret_code = result

                if ret_code != 0 and ret_code < 0:
                    raise RuntimeError(f"JVRTOpen failed: {ret_code}")

                while True:
                    ret, data, filename = client.read()
                    if ret == 0:
                        break
                    if ret == -1:
                        continue
                    if ret < -1:
                        raise RuntimeError(f"JVRead error: {ret}")

                    rec = parse_record(data, data_spec, filename)
                    if rec:
                        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                        record_count += 1
                        records_in_part += 1

                race_ok += 1
                completed.add(race_id)
                state["completed_race_ids"] = sorted(completed)
                state["updated_at"] = datetime.now().isoformat()
                # ★重要: Windows環境ではAV/同期ソフト等で一時的にファイルがロックされ、
                #   tmp -> replace が PermissionError(WinError 5) になることがある。
                #   取得済みJSONLはすでに書けているので、state保存失敗は「レース失敗」にしない。
                try:
                    _save_state_atomic(state_file, state)
                except Exception as e:
                    logger.warning(f"Failed to save state (will continue): {e}")
            except Exception as e:
                race_err += 1
                logger.warning(f"Failed race_id={race_id}: {e}")
            finally:
                try:
                    client.close()
                except Exception:
                    pass

            races_in_part += 1
            if chunk_size and chunk_size > 0 and races_in_part >= chunk_size:
                # ★重要: ちょうど末尾でchunk境界に達した場合、空の次partを作らない
                if i < len(pending_race_ids):
                    _close_part_file()
                    part_idx += 1
                    current_path, f = _open_chunk_file(part_idx)
                    races_in_part = 0

            if sleep_sec > 0:
                time.sleep(sleep_sec)

            if i % 10 == 0:
                logger.info(f"progress: races_ok={race_ok}, races_err={race_err}, records={record_count}")
    finally:
        _close_part_file()

    return {
        "data_spec": data_spec,
        "races_total": len(race_ids),
        "races_skipped": races_skipped,
        "races_ok": race_ok,
        "races_err": race_err,
        "records": record_count,
        "output_files": output_files,
        "state_file": str(state_file),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="JV-Link bulk realtime fetcher (race_key指定)")
    parser.add_argument("data_spec", help="データ種別ID（例: 0B41）")
    parser.add_argument("--race-ids-file", type=Path, required=True, help="race_id一覧ファイル（1行1レース）")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="出力ディレクトリ（デフォルト: data/raw）",
    )
    parser.add_argument(
        "--software-id",
        dest="software_id",
        default=os.environ.get("KEIBA_JV_SID") or os.environ.get("KEIBA_JV_SOFTWARE_ID") or "UNKNOWN",
        help="JVInitに渡すSid（ソフトウェアID）。未指定ならKEIBA_JV_SID/KEIBA_JV_SOFTWARE_ID、なければUNKNOWN",
    )
    parser.add_argument(
        "--use-key",
        dest="use_key",
        default=os.environ.get("KEIBA_JV_USE_KEY") or None,
        help="利用キー（JVSetServiceKey）。推奨: 環境変数KEIBA_JV_USE_KEY",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=0.0,
        help="レースごとの待機秒（過負荷回避用、デフォルト0）",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="NレースごとにJSONLを分割（0以下で1ファイル）。デフォルト50",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="state_file を見て完了レースをスキップして再開する",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="再開用stateファイル（未指定なら data/state/bulk_fetch_{data_spec}.json）",
    )
    args = parser.parse_args()

    race_ids = _read_race_ids(args.race_ids_file)
    if not race_ids:
        raise SystemExit("race_ids_file is empty")

    result = bulk_fetch_realtime(
        data_spec=args.data_spec,
        race_ids=race_ids,
        output_dir=args.output_dir,
        software_id=args.software_id,
        service_key=args.use_key,
        sleep_sec=args.sleep_sec,
        chunk_size=args.chunk_size,
        resume=args.resume,
        state_file=args.state_file,
    )
    logger.info(f"Result: {result}")


if __name__ == "__main__":
    main()


