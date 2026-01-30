"""
三連複/三連単データ品質検証スクリプト

レビュー指摘の「前向き検証に耐える」チェック項目を実行:
  1. 当日運用E2E: buy/closeスナップショットの存在・品質確認
  2. 欠落率の把握: n_rows分布、stale判定で落ちた割合
  3. 決済の一致: HR払戻データとの整合性確認
  4. close vs buy の乖離定量化: ROI差分の可視化
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

# プロジェクトルートをパスに追加（keibaパッケージが py64_analysis/src/keiba にある前提）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "py64_analysis" / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 型ヒント用（実際のimportはmain()内で行う）
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from keiba.exotics.odds_snapshot import TicketType, MissingPolicy


def check_snapshot_coverage(
    session: Session,
    start_date: str,
    end_date: str,
    ticket_type: "TicketType",
    buy_t_minus_minutes: int = 5,
    max_staleness_min: float = 10.0,
) -> pd.DataFrame:
    """
    チェック1: 当日運用E2E - buy/closeスナップショットの存在・品質確認
    """
    from keiba.exotics.odds_snapshot import select_snapshot
    
    rows = []
    race_list = session.execute(
        text(
            """
            SELECT r.race_id, r.date, r.start_time, r.field_size
            FROM fact_race r
            WHERE r.date BETWEEN :s AND :e
              AND r.start_time IS NOT NULL
            ORDER BY r.date, r.race_id
            """
        ),
        {"s": start_date, "e": end_date},
    ).fetchall()

    for race_id, race_date, start_time, field_size in race_list:
        if start_time is None:
            continue
        start_dt = datetime.combine(race_date, start_time)
        buy_time = start_dt - timedelta(minutes=buy_t_minus_minutes)
        close_time = start_dt

        # buy snapshot
        snap_buy = select_snapshot(
            session,
            ticket_type=ticket_type,
            race_id=race_id,
            t=buy_time,
            max_staleness_min=max_staleness_min,
            missing_policy="skip",
        )

        # close snapshot
        snap_close = select_snapshot(
            session,
            ticket_type=ticket_type,
            race_id=race_id,
            t=close_time,
            max_staleness_min=max_staleness_min,
            missing_policy="skip",
        )

        # 期待n_rows計算（trio: nC3, trifecta: nP3）
        expected_rows = None
        if field_size is not None and field_size >= 3:
            if ticket_type == "trio":
                # nC3 = n! / (3! * (n-3)!) = n * (n-1) * (n-2) / 6
                expected_rows = int(field_size * (field_size - 1) * (field_size - 2) / 6)
            else:
                # nP3 = n * (n-1) * (n-2)
                expected_rows = int(field_size * (field_size - 1) * (field_size - 2))

        buy_rows_ratio = None
        if snap_buy.ok and snap_buy.n_rows is not None and expected_rows is not None and expected_rows > 0:
            buy_rows_ratio = float(snap_buy.n_rows) / float(expected_rows)

        close_rows_ratio = None
        if snap_close.ok and snap_close.n_rows is not None and expected_rows is not None and expected_rows > 0:
            close_rows_ratio = float(snap_close.n_rows) / float(expected_rows)

        rows.append(
            {
                "race_id": race_id,
                "race_date": race_date,
                "start_time": start_time,
                "field_size": field_size,
                "expected_rows": expected_rows,
                "buy_time": buy_time,
                "close_time": close_time,
                "buy_ok": snap_buy.ok,
                "buy_snapshot_id": snap_buy.snapshot_id,
                "buy_asof_time": snap_buy.asof_time,
                "buy_age_min": snap_buy.age_min,
                "buy_n_rows": snap_buy.n_rows,
                "buy_rows_ratio": buy_rows_ratio,
                "buy_stale": snap_buy.stale,
                "buy_reason": snap_buy.reason,
                "close_ok": snap_close.ok,
                "close_snapshot_id": snap_close.snapshot_id,
                "close_asof_time": snap_close.asof_time,
                "close_age_min": snap_close.age_min,
                "close_n_rows": snap_close.n_rows,
                "close_rows_ratio": close_rows_ratio,
                "close_stale": snap_close.stale,
                "close_reason": snap_close.reason,
            }
        )

    return pd.DataFrame(rows)


def check_missing_rate_stats(df: pd.DataFrame) -> dict:
    """
    チェック2: 欠落率の把握 - n_rows分布、stale判定で落ちた割合
    """
    n_total = len(df)
    n_buy_ok = int(df["buy_ok"].sum())
    n_close_ok = int(df["close_ok"].sum())
    
    # reason で分離: "missing" vs "stale"
    buy_pure_missing = int((df["buy_reason"] == "missing").sum())
    buy_stale = int((df["buy_reason"] == "stale").sum())
    close_pure_missing = int((df["close_reason"] == "missing").sum())
    close_stale = int((df["close_reason"] == "stale").sum())

    # n_rows統計（to_dict()でpandas Seriesのtruthiness問題を回避）
    buy_stats = df.loc[df["buy_ok"], "buy_n_rows"].describe().to_dict() if n_buy_ok > 0 else {}
    close_stats = df.loc[df["close_ok"], "close_n_rows"].describe().to_dict() if n_close_ok > 0 else {}
    
    # 期待n_rowsとの比較（部分欠損検出）
    buy_ratio_stats = df.loc[df["buy_ok"] & df["buy_rows_ratio"].notna(), "buy_rows_ratio"].describe().to_dict() if n_buy_ok > 0 else {}
    close_ratio_stats = df.loc[df["close_ok"] & df["close_rows_ratio"].notna(), "close_rows_ratio"].describe().to_dict() if n_close_ok > 0 else {}
    buy_low_ratio_count = int((df["buy_ok"] & df["buy_rows_ratio"].notna() & (df["buy_rows_ratio"] < 0.98)).sum()) if n_buy_ok > 0 else 0
    close_low_ratio_count = int((df["close_ok"] & df["close_rows_ratio"].notna() & (df["close_rows_ratio"] < 0.98)).sum()) if n_close_ok > 0 else 0

    return {
        "n_total_races": n_total,
        "buy_coverage": {
            "ok": n_buy_ok,
            "pure_missing": buy_pure_missing,
            "pure_missing_rate": float(buy_pure_missing / n_total) if n_total > 0 else 0.0,
            "stale": buy_stale,
            "stale_rate": float(buy_stale / n_total) if n_total > 0 else 0.0,
            "n_rows_mean": float(buy_stats.get("mean")) if n_buy_ok > 0 else None,
            "n_rows_min": float(buy_stats.get("min")) if n_buy_ok > 0 else None,
            "n_rows_max": float(buy_stats.get("max")) if n_buy_ok > 0 else None,
            "rows_ratio_mean": float(buy_ratio_stats.get("mean")) if buy_ratio_stats else None,
            "rows_ratio_min": float(buy_ratio_stats.get("min")) if buy_ratio_stats else None,
            "low_ratio_count": buy_low_ratio_count,
        },
        "close_coverage": {
            "ok": n_close_ok,
            "pure_missing": close_pure_missing,
            "pure_missing_rate": float(close_pure_missing / n_total) if n_total > 0 else 0.0,
            "stale": close_stale,
            "stale_rate": float(close_stale / n_total) if n_total > 0 else 0.0,
            "n_rows_mean": float(close_stats.get("mean")) if n_close_ok > 0 else None,
            "n_rows_min": float(close_stats.get("min")) if n_close_ok > 0 else None,
            "n_rows_max": float(close_stats.get("max")) if n_close_ok > 0 else None,
            "rows_ratio_mean": float(close_ratio_stats.get("mean")) if close_ratio_stats else None,
            "rows_ratio_min": float(close_ratio_stats.get("min")) if close_ratio_stats else None,
            "low_ratio_count": close_low_ratio_count,
        },
    }


def check_settlement_consistency(
    session: Session,
    start_date: str,
    end_date: str,
    ticket_type: "TicketType",
) -> pd.DataFrame:
    """
    チェック3: 決済の一致 - HR払戻データとの整合性確認
    
    実際のTop3組番がpayoutテーブルに存在するかを直接チェックします。
    """
    if ticket_type == "trio":
        # trio: Top3を馬番昇順にソートして組番を作る
        # 同着レースはスキップ（posごとにCOUNT=1を保証）
        rows = session.execute(
            text(
                """
                WITH top3_raw AS (
                    SELECT 
                        race_id,
                        finish_pos,
                        horse_no
                    FROM fact_result
                    WHERE race_id IN (
                        SELECT race_id FROM fact_race WHERE date BETWEEN :s AND :e
                    )
                      AND finish_pos IN (1,2,3)
                ),
                top3_pos_counts AS (
                    SELECT 
                        race_id,
                        finish_pos,
                        COUNT(*) AS pos_count
                    FROM top3_raw
                    GROUP BY race_id, finish_pos
                ),
                top3_valid AS (
                    SELECT race_id
                    FROM top3_pos_counts
                    GROUP BY race_id
                    HAVING COUNT(*) FILTER (WHERE finish_pos IN (1,2,3) AND pos_count = 1) = 3
                ),
                top3 AS (
                    SELECT 
                        t.race_id,
                        MAX(CASE WHEN t.finish_pos = 1 THEN t.horse_no END) AS h1,
                        MAX(CASE WHEN t.finish_pos = 2 THEN t.horse_no END) AS h2,
                        MAX(CASE WHEN t.finish_pos = 3 THEN t.horse_no END) AS h3
                    FROM top3_raw t
                    INNER JOIN top3_valid v ON v.race_id = t.race_id
                    GROUP BY t.race_id
                ),
                top3_sorted AS (
                    SELECT 
                        race_id,
                        LEAST(h1, h2, h3) AS a,
                        CASE 
                            WHEN h1 != LEAST(h1, h2, h3) AND h1 != GREATEST(h1, h2, h3) THEN h1
                            WHEN h2 != LEAST(h1, h2, h3) AND h2 != GREATEST(h1, h2, h3) THEN h2
                            ELSE h3
                        END AS b,
                        GREATEST(h1, h2, h3) AS c
                    FROM top3
                    WHERE h1 IS NOT NULL AND h2 IS NOT NULL AND h3 IS NOT NULL
                ),
                payouts AS (
                    SELECT 
                        race_id,
                        horse_no_1,
                        horse_no_2,
                        horse_no_3,
                        payout_yen
                    FROM fact_payout_trio
                    WHERE race_id IN (SELECT race_id FROM fact_race WHERE date BETWEEN :s AND :e)
                ),
                refunds AS (
                    SELECT DISTINCT race_id
                    FROM fact_refund_horse
                    WHERE race_id IN (SELECT race_id FROM fact_race WHERE date BETWEEN :s AND :e)
                )
                SELECT 
                    t.race_id,
                    ARRAY[t.a, t.b, t.c] AS top3_horses,
                    COUNT(p.*) AS n_payouts,
                    SUM(CASE WHEN p.payout_yen > 0 THEN 1 ELSE 0 END) AS n_winning_payouts,
                    MAX(CASE WHEN r.race_id IS NOT NULL THEN 1 ELSE 0 END) AS has_refund,
                    MAX(CASE WHEN p.horse_no_1 = t.a AND p.horse_no_2 = t.b AND p.horse_no_3 = t.c THEN 1 ELSE 0 END) AS has_hit_payout,
                    MAX(CASE WHEN p.horse_no_1 = t.a AND p.horse_no_2 = t.b AND p.horse_no_3 = t.c THEN p.payout_yen ELSE NULL END) AS hit_payout_yen
                FROM top3_sorted t
                LEFT JOIN payouts p ON p.race_id = t.race_id
                LEFT JOIN refunds r ON r.race_id = t.race_id
                GROUP BY t.race_id, t.a, t.b, t.c
                ORDER BY t.race_id
                """
            ),
            {"s": start_date, "e": end_date},
        ).fetchall()
    else:
        # trifecta: 順序そのままでチェック
        # 同着レースはスキップ（posごとにCOUNT=1を保証）
        rows = session.execute(
            text(
                """
                WITH top3_raw AS (
                    SELECT 
                        race_id,
                        finish_pos,
                        horse_no
                    FROM fact_result
                    WHERE race_id IN (
                        SELECT race_id FROM fact_race WHERE date BETWEEN :s AND :e
                    )
                      AND finish_pos IN (1,2,3)
                ),
                top3_pos_counts AS (
                    SELECT 
                        race_id,
                        finish_pos,
                        COUNT(*) AS pos_count
                    FROM top3_raw
                    GROUP BY race_id, finish_pos
                ),
                top3_valid AS (
                    SELECT race_id
                    FROM top3_pos_counts
                    GROUP BY race_id
                    HAVING COUNT(*) FILTER (WHERE finish_pos IN (1,2,3) AND pos_count = 1) = 3
                ),
                top3 AS (
                    SELECT 
                        t.race_id,
                        MAX(CASE WHEN t.finish_pos = 1 THEN t.horse_no END) AS first_no,
                        MAX(CASE WHEN t.finish_pos = 2 THEN t.horse_no END) AS second_no,
                        MAX(CASE WHEN t.finish_pos = 3 THEN t.horse_no END) AS third_no
                    FROM top3_raw t
                    INNER JOIN top3_valid v ON v.race_id = t.race_id
                    GROUP BY t.race_id
                ),
                payouts AS (
                    SELECT 
                        race_id,
                        first_no,
                        second_no,
                        third_no,
                        payout_yen
                    FROM fact_payout_trifecta
                    WHERE race_id IN (SELECT race_id FROM fact_race WHERE date BETWEEN :s AND :e)
                ),
                refunds AS (
                    SELECT DISTINCT race_id
                    FROM fact_refund_horse
                    WHERE race_id IN (SELECT race_id FROM fact_race WHERE date BETWEEN :s AND :e)
                )
                SELECT 
                    t.race_id,
                    ARRAY[t.first_no, t.second_no, t.third_no] AS top3_horses,
                    COUNT(p.*) AS n_payouts,
                    SUM(CASE WHEN p.payout_yen > 0 THEN 1 ELSE 0 END) AS n_winning_payouts,
                    MAX(CASE WHEN r.race_id IS NOT NULL THEN 1 ELSE 0 END) AS has_refund,
                    MAX(CASE WHEN p.first_no = t.first_no AND p.second_no = t.second_no AND p.third_no = t.third_no THEN 1 ELSE 0 END) AS has_hit_payout,
                    MAX(CASE WHEN p.first_no = t.first_no AND p.second_no = t.second_no AND p.third_no = t.third_no THEN p.payout_yen ELSE NULL END) AS hit_payout_yen
                FROM top3 t
                LEFT JOIN payouts p ON p.race_id = t.race_id
                LEFT JOIN refunds r ON r.race_id = t.race_id
                GROUP BY t.race_id, t.first_no, t.second_no, t.third_no
                ORDER BY t.race_id
                """
            ),
            {"s": start_date, "e": end_date},
        ).fetchall()

    # 固定列を定義（空の場合でもKeyErrorを避けるため）
    columns = [
        "race_id",
        "top3_horses",
        "n_payouts",
        "n_winning_payouts",
        "has_refund",
        "has_payout_data",
        "has_hit_payout",
        "hit_payout_yen",
    ]
    
    result_rows = []
    for race_id, top3_horses, n_payouts, n_winning, has_refund, has_hit_payout, hit_payout_yen in rows:
        result_rows.append(
            {
                "race_id": race_id,
                "top3_horses": top3_horses,
                "n_payouts": n_payouts,
                "n_winning_payouts": n_winning,
                "has_refund": bool(has_refund),
                "has_payout_data": n_payouts > 0,
                "has_hit_payout": bool(has_hit_payout),
                "hit_payout_yen": int(hit_payout_yen) if hit_payout_yen is not None else None,
            }
        )

    # 空の場合でも列を持つDataFrameを返す
    return pd.DataFrame(result_rows, columns=columns)


def compare_close_vs_buy_roi(
    session: Session,
    start_date: str,
    end_date: str,
    ticket_type: "TicketType",
    max_bets_per_race: int = 10,
    ev_margin: float = 0.10,
    max_staleness_min: float = 10.0,
    buy_t_minus_minutes: int = 5,
) -> dict:
    """
    チェック4: close vs buy の乖離定量化 - ROI差分の可視化
    """
    from keiba.exotics.backtest import ExoticsBacktestEngine

    engine = ExoticsBacktestEngine(session)

    # close mode
    result_close = engine.run(
        start_date=start_date,
        end_date=end_date,
        ticket_type=ticket_type,
        odds_mode="close",
        max_bets_per_race=max_bets_per_race,
        ev_margin=ev_margin,
        max_staleness_min=max_staleness_min,
        missing_policy="skip",
        buy_t_minus_minutes=buy_t_minus_minutes,
    )

    # buy mode
    result_buy = engine.run(
        start_date=start_date,
        end_date=end_date,
        ticket_type=ticket_type,
        odds_mode="buy",
        max_bets_per_race=max_bets_per_race,
        ev_margin=ev_margin,
        max_staleness_min=max_staleness_min,
        missing_policy="skip",
        buy_t_minus_minutes=buy_t_minus_minutes,
    )

    return {
        "close": {
            "roi": result_close.roi,
            "n_bets": result_close.n_bets,
            "n_wins": result_close.n_wins,
            "total_stake": result_close.total_stake,
            "total_profit": result_close.total_profit,
            "n_races_skipped_snapshot": result_close.n_races_skipped_snapshot,
            "n_races_skipped_hr": result_close.n_races_skipped_hr,
        },
        "buy": {
            "roi": result_buy.roi,
            "n_bets": result_buy.n_bets,
            "n_wins": result_buy.n_wins,
            "total_stake": result_buy.total_stake,
            "total_profit": result_buy.total_profit,
            "n_races_skipped_snapshot": result_buy.n_races_skipped_snapshot,
            "n_races_skipped_hr": result_buy.n_races_skipped_hr,
        },
        "diff": {
            "roi_diff": result_buy.roi - result_close.roi,
            "n_bets_diff": result_buy.n_bets - result_close.n_bets,
            "total_profit_diff": result_buy.total_profit - result_close.total_profit,
        },
    }


def generate_report(
    coverage_df: pd.DataFrame,
    missing_stats: dict,
    settlement_df: pd.DataFrame,
    roi_comparison: dict,
    output_path: Path,
) -> None:
    """検証レポートを生成"""
    lines = [
        "# 三連複/三連単データ品質検証レポート",
        "",
        f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. スナップショットカバレッジ（当日運用E2E）",
        "",
        f"- 総レース数: {missing_stats['n_total_races']}",
        "",
        "### Buy スナップショット",
        f"- OK: {missing_stats['buy_coverage']['ok']} ({100*(1-missing_stats['buy_coverage']['pure_missing_rate']-missing_stats['buy_coverage']['stale_rate']):.1f}%)",
        f"- Pure Missing: {missing_stats['buy_coverage']['pure_missing']} ({100*missing_stats['buy_coverage']['pure_missing_rate']:.1f}%)",
        f"- Stale: {missing_stats['buy_coverage']['stale']} ({100*missing_stats['buy_coverage']['stale_rate']:.1f}%)",
        f"- n_rows平均: {missing_stats['buy_coverage']['n_rows_mean']:.1f}" if missing_stats['buy_coverage']['n_rows_mean'] else "- n_rows平均: N/A",
        f"- n_rows範囲: {missing_stats['buy_coverage']['n_rows_min']:.0f} - {missing_stats['buy_coverage']['n_rows_max']:.0f}" if missing_stats['buy_coverage']['n_rows_min'] is not None else "- n_rows範囲: N/A",
        f"- 期待n_rows比平均: {missing_stats['buy_coverage']['rows_ratio_mean']:.3f}" if missing_stats['buy_coverage']['rows_ratio_mean'] else "- 期待n_rows比平均: N/A",
        f"- 期待n_rows比 < 0.98: {missing_stats['buy_coverage']['low_ratio_count']}件" if missing_stats['buy_coverage']['low_ratio_count'] is not None else "",
        "",
        "### Close スナップショット",
        f"- OK: {missing_stats['close_coverage']['ok']} ({100*(1-missing_stats['close_coverage']['pure_missing_rate']-missing_stats['close_coverage']['stale_rate']):.1f}%)",
        f"- Pure Missing: {missing_stats['close_coverage']['pure_missing']} ({100*missing_stats['close_coverage']['pure_missing_rate']:.1f}%)",
        f"- Stale: {missing_stats['close_coverage']['stale']} ({100*missing_stats['close_coverage']['stale_rate']:.1f}%)",
        f"- n_rows平均: {missing_stats['close_coverage']['n_rows_mean']:.1f}" if missing_stats['close_coverage']['n_rows_mean'] else "- n_rows平均: N/A",
        f"- n_rows範囲: {missing_stats['close_coverage']['n_rows_min']:.0f} - {missing_stats['close_coverage']['n_rows_max']:.0f}" if missing_stats['close_coverage']['n_rows_min'] is not None else "- n_rows範囲: N/A",
        f"- 期待n_rows比平均: {missing_stats['close_coverage']['rows_ratio_mean']:.3f}" if missing_stats['close_coverage']['rows_ratio_mean'] else "- 期待n_rows比平均: N/A",
        f"- 期待n_rows比 < 0.98: {missing_stats['close_coverage']['low_ratio_count']}件" if missing_stats['close_coverage']['low_ratio_count'] is not None else "",
        "",
        "## 2. 決済データ整合性",
        "",
    ]
    
    # settlement_df が空の場合の処理
    if settlement_df.empty:
        lines.extend([
            "- ⚠️ 決済データが0件です（fact_result未投入 / 日付範囲に有効レースなし / 全レースが同着でスキップ の可能性）",
            "",
        ])
    else:
        lines.extend([
            f"- レース数: {len(settlement_df)}",
            f"- 払戻データあり: {settlement_df['has_payout_data'].sum()} ({100*settlement_df['has_payout_data'].mean():.1f}%)",
            f"- 的中組番の払戻あり: {settlement_df['has_hit_payout'].sum()} ({100*settlement_df['has_hit_payout'].mean():.1f}%)",
            f"- 的中払戻あり: {settlement_df['n_winning_payouts'].sum()}",
            f"- 返還あり: {settlement_df['has_refund'].sum()}",
            "",
        ])
    
    lines.extend([
        "",
        "## 3. Close vs Buy ROI比較",
        "",
        "### Close Mode",
        f"- ROI: {roi_comparison['close']['roi']:.4f} ({100*roi_comparison['close']['roi']:.2f}%)",
        f"- ベット数: {roi_comparison['close']['n_bets']}",
        f"- 的中数: {roi_comparison['close']['n_wins']}",
        f"- 総投資: {roi_comparison['close']['total_stake']:,}円",
        f"- 総利益: {roi_comparison['close']['total_profit']:,}円",
        f"- スナップショット欠落スキップ: {roi_comparison['close']['n_races_skipped_snapshot']}",
        f"- HR欠落スキップ: {roi_comparison['close']['n_races_skipped_hr']}",
        "",
        "### Buy Mode",
        f"- ROI: {roi_comparison['buy']['roi']:.4f} ({100*roi_comparison['buy']['roi']:.2f}%)",
        f"- ベット数: {roi_comparison['buy']['n_bets']}",
        f"- 的中数: {roi_comparison['buy']['n_wins']}",
        f"- 総投資: {roi_comparison['buy']['total_stake']:,}円",
        f"- 総利益: {roi_comparison['buy']['total_profit']:,}円",
        f"- スナップショット欠落スキップ: {roi_comparison['buy']['n_races_skipped_snapshot']}",
        f"- HR欠落スキップ: {roi_comparison['buy']['n_races_skipped_hr']}",
        "",
        "### 差分",
        f"- ROI差分: {roi_comparison['diff']['roi_diff']:.4f} ({100*roi_comparison['diff']['roi_diff']:.2f}pp)",
        f"- ベット数差分: {roi_comparison['diff']['n_bets_diff']}",
        f"- 総利益差分: {roi_comparison['diff']['total_profit_diff']:,}円",
        "",
        "## 4. 推奨アクション",
        "",
    ])

    # 推奨アクションを自動生成
    if missing_stats['buy_coverage']['pure_missing_rate'] > 0.1:
        lines.append(f"- ⚠️ Buyスナップショット純欠落率が高い ({100*missing_stats['buy_coverage']['pure_missing_rate']:.1f}%) → forward収集の見直し")
    if missing_stats['buy_coverage']['stale_rate'] > 0.05:
        lines.append(f"- ⚠️ BuyスナップショットのStale率が高い ({100*missing_stats['buy_coverage']['stale_rate']:.1f}%) → max_staleness_minの調整検討")
    if missing_stats['buy_coverage'].get('low_ratio_count', 0) > 0:
        lines.append(f"- ⚠️ Buyスナップショットで期待n_rows比 < 0.98 が {missing_stats['buy_coverage']['low_ratio_count']}件 → 組番欠損の可能性")
    if abs(roi_comparison['diff']['roi_diff']) > 0.05:
        lines.append(f"- ⚠️ Close vs Buy ROI差分が大きい ({100*roi_comparison['diff']['roi_diff']:.2f}pp) → オッズ変動耐性の検討")
    if not settlement_df.empty and settlement_df['has_payout_data'].mean() < 0.9:
        lines.append(f"- ⚠️ 払戻データカバレッジが低い ({100*settlement_df['has_payout_data'].mean():.1f}%) → HRローダーの確認")

    if len(lines) == len([l for l in lines if not l.startswith("- ⚠️")]):
        lines.append("- ✅ 主要指標は問題なし")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"レポート生成: {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="三連複/三連単データ品質検証")
    p.add_argument("--start-date", required=True, help="開始日 (YYYYMMDD)")
    p.add_argument("--end-date", required=True, help="終了日 (YYYYMMDD)")
    p.add_argument("--ticket-type", choices=["trio", "trifecta"], default="trio", help="券種")
    p.add_argument("--buy-t-minus-minutes", type=int, default=5, help="購入想定時刻（レース開始前N分）")
    p.add_argument("--max-staleness-min", type=float, default=10.0, help="最大許容stale時間（分）")
    p.add_argument("--max-bets-per-race", type=int, default=10, help="1レースあたり最大ベット数")
    p.add_argument("--ev-margin", type=float, default=0.10, help="EV閾値")
    p.add_argument("--out-dir", type=Path, default=Path("data/validation_exotics"))
    args = p.parse_args()

    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # importはpath設定後に実行
    from keiba.db.loader import get_session
    from keiba.exotics.odds_snapshot import select_snapshot, TicketType, MissingPolicy
    
    session = get_session()

    print("チェック1: スナップショットカバレッジ確認中...")
    coverage_df = check_snapshot_coverage(
        session,
        args.start_date,
        args.end_date,
        args.ticket_type,
        args.buy_t_minus_minutes,
        args.max_staleness_min,
    )
    coverage_df.to_csv(out_dir / f"coverage_{ts}.csv", index=False, encoding="utf-8-sig")

    print("チェック2: 欠落率統計計算中...")
    missing_stats = check_missing_rate_stats(coverage_df)

    print("チェック3: 決済データ整合性確認中...")
    settlement_df = check_settlement_consistency(
        session,
        args.start_date,
        args.end_date,
        args.ticket_type,
    )
    settlement_df.to_csv(out_dir / f"settlement_{ts}.csv", index=False, encoding="utf-8-sig")

    print("チェック4: Close vs Buy ROI比較中...")
    roi_comparison = compare_close_vs_buy_roi(
        session,
        args.start_date,
        args.end_date,
        args.ticket_type,
        args.max_bets_per_race,
        args.ev_margin,
        args.max_staleness_min,
        args.buy_t_minus_minutes,
    )

    print("レポート生成中...")
    report_path = out_dir / f"report_{ts}.md"
    generate_report(coverage_df, missing_stats, settlement_df, roi_comparison, report_path)

    print(f"\n検証完了: {out_dir}")


if __name__ == "__main__":
    main()

