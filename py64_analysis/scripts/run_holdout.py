"""
Ticket3: 30〜90日 holdout を固定パラメータで回すための実行スクリプト（1本で完結）。

目的:
  - 1コマンドで以下をまとめて生成する
    - artifacts（model/calibrator）
    - backtestレポート（ROI + 確率品質 + reliability）
    - summary.json（比較可能な指標）
  - slippage推定（closing_odds_multiplier）は test 期間を参照しない
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def _ensure_import_path() -> None:
    """
    `python py64_analysis/scripts/run_holdout.py ...` で実行しても import が通るようにする。
    """
    project_root = Path(__file__).resolve().parents[2]  # keiba/
    src = project_root / "py64_analysis" / "src"
    if src.exists():
        sys.path.insert(0, str(src))


def _align_features(X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    X2 = X.reindex(columns=feature_names, fill_value=0.0)
    X2 = X2.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    return X2


def main() -> None:
    _ensure_import_path()

    import numpy as np
    from sqlalchemy import text

    from keiba.analysis.odds_slippage import (
        fetch_odds_final_to_buy,
        recommend_closing_odds_multiplier,
        summarize_ratio_final_to_buy,
    )
    from keiba.analysis.slippage_table import (
        fetch_slippage_feature_snapshot,
        fit_slippage_table,
    )
    from keiba.backtest.engine import BacktestEngine, run_backtest
    from keiba.backtest.pred_quality import compute_prediction_quality, plot_reliability
    from keiba.backtest.report import generate_report
    from keiba.betting.odds_band_bias import OddsBandBias
    from keiba.betting.uncertainty_shrink import UncertaintyShrink
    from keiba.config import get_config
    from keiba.db.loader import get_session
    from keiba.features.build_features import FeatureBuilder, build_features
    from keiba.modeling.race_softmax import apply_race_softmax
    from keiba.modeling.train import (
        _surface_segments_from_race_ids,
        prepare_training_data,
        train_model,
    )

    p = argparse.ArgumentParser(description="Run single holdout (train/valid/test) end-to-end")
    p.add_argument("--train-start", required=True)
    p.add_argument("--train-end", required=True)
    p.add_argument("--valid-start", default=None)
    p.add_argument("--valid-end", default=None)
    p.add_argument("--test-start", required=True)
    p.add_argument("--test-end", required=True)
    p.add_argument("--name", default="holdout")
    p.add_argument("--config", default=None, help="config path (optional)")
    p.add_argument(
        "--estimate-closing-mult",
        action="store_true",
        help="train(またはvalidまで)でclosing_odds_multiplierを推定して上書き",
    )
    p.add_argument("--closing-mult-quantile", type=float, default=0.30)
    p.add_argument("--initial-bankroll", type=float, default=None)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="出力先（未指定なら data/holdout_runs/<name>_<ts>）",
    )
    args = p.parse_args()

    from keiba.utils.config_resolver import (
        git_commit,
        rel_path,
        resolve_config_path,
        save_config_origin,
        save_config_used,
    )

    exp_cfg = None
    if args.config is None and not os.environ.get("KEIBA_CONFIG_PATH"):
        cand = Path("config") / "experiments" / f"{args.name}.yaml"
        if cand.exists():
            exp_cfg = cand
        else:
            cand = Path("config") / "experiments" / f"{args.name}.yml"
            if cand.exists():
                exp_cfg = cand

    if exp_cfg is not None:
        resolved_config_path = exp_cfg.resolve()
        config_origin = f"auto:{exp_cfg}"
    else:
        resolved_config_path, config_origin = resolve_config_path(args.config)
    os.environ["KEIBA_CONFIG_PATH"] = str(resolved_config_path)

    cfg = get_config()
    if args.initial_bankroll is None:
        initial_bankroll = float(cfg.betting.bankroll_yen)
    else:
        initial_bankroll = float(args.initial_bankroll)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = Path.cwd()
    if args.out_dir is None:
        run_dir = project_root / "data" / "holdout_runs" / f"{args.name}_{ts}"
    else:
        run_dir = args.out_dir
    artifacts_dir = run_dir / "artifacts"
    report_dir = run_dir / "report"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    config_meta = save_config_used(resolved_config_path, run_dir)
    config_used_path = Path(config_meta["config_used_path"])
    origin_payload = {
        "origin": config_origin,
        "resolved_config_path": rel_path(resolved_config_path, project_root),
        "config_used_path": rel_path(config_used_path, project_root),
        "config_hash_sha256": config_meta.get("config_hash_sha256"),
        "git_commit": git_commit(project_root),
        "generated_at": ts,
    }
    save_config_origin(run_dir, origin_payload)

    session = get_session()

    # 対象race_idを拾う
    def _race_ids_between(s: str, e: str) -> list[str]:
        u = cfg.universe
        track_codes = list(u.track_codes or [])
        exclude_race_ids = list(u.exclude_race_ids or [])
        buy_minutes = int(cfg.backtest.buy_t_minus_minutes)
        rows = session.execute(
            text(
                """
                SELECT r.race_id
                FROM fact_race r
                WHERE r.date BETWEEN :s AND :e
                  AND r.start_time IS NOT NULL
                  AND (:track_codes_len = 0 OR r.track_code = ANY(:track_codes))
                  AND (:require_results = FALSE OR EXISTS (
                        SELECT 1 FROM fact_result res
                        WHERE res.race_id = r.race_id
                          AND res.finish_pos IS NOT NULL
                  ))
                  AND (:require_ts_win = FALSE OR EXISTS (
                        SELECT 1 FROM odds_ts_win o
                        WHERE o.race_id = r.race_id
                          AND o.odds > 0
                          AND o.asof_time <= (
                                (r.date::timestamp + r.start_time)
                                - make_interval(mins => :buy_minutes)
                          )
                  ))
                  AND (:exclude_len = 0 OR NOT (r.race_id = ANY(:exclude_race_ids)))
                ORDER BY r.date, r.race_id
                """
            ),
            {
                "s": s,
                "e": e,
                "track_codes_len": len(track_codes),
                "track_codes": track_codes,
                "require_results": bool(u.require_results),
                "require_ts_win": bool(u.require_ts_win),
                "buy_minutes": buy_minutes,
                "exclude_len": len(exclude_race_ids),
                "exclude_race_ids": exclude_race_ids,
            },
        ).fetchall()
        return [r[0] for r in rows]

    def _fit_race_cost_cap() -> dict | None:
        rcfg = getattr(cfg.betting, "race_cost_filter", None)
        if not rcfg or not bool(getattr(rcfg, "enabled", False)):
            return None

        cap_mode = str(getattr(rcfg, "cap_mode", "fixed") or "fixed").lower()
        if cap_mode != "train_quantile":
            return None

        metric = str(getattr(rcfg, "metric", "takeout_implied") or "takeout_implied").lower()
        if metric not in ("takeout_implied", "overround_sum_inv"):
            metric = "takeout_implied"

        q = getattr(rcfg, "train_quantile", None)
        try:
            q = float(q) if q is not None else None
        except Exception:
            q = None

        min_races = 200
        meta = {
            "enabled": bool(getattr(rcfg, "enabled", False)),
            "cap_mode": cap_mode,
            "metric": metric,
            "train_quantile": q,
            "fit_population": None,
            "fit_population_note": None,
            "n_fit_races_input": None,
            "min_train_races": min_races,
            "n_train_races_used": 0,
            "fit_metric_unique_values": None,
            "cap_value": None,
            "status": None,
        }

        if q is None or not (0.0 < q < 1.0):
            rcfg.enabled = False
            meta["status"] = "disabled_invalid_quantile"
            (artifacts_dir / "race_cost_cap.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"[capfit] window={args.name} metric={metric} q={q} cap_value=None "
                "n_train_races=0 status=invalid_quantile"
            )
            return meta

        fit_scope = str(getattr(rcfg, "cap_fit_scope", "train") or "train").lower()
        if fit_scope not in ("train", "train_valid"):
            fit_scope = "train"
        meta["fit_scope"] = fit_scope

        fit_population = str(
            getattr(rcfg, "cap_fit_population", "all_races") or "all_races"
        ).lower()
        if fit_population not in ("all_races", "precap_candidate_races"):
            fit_population = "all_races"
        meta["fit_population"] = fit_population

        u = cfg.universe
        track_codes = list(u.track_codes or [])
        exclude_race_ids = list(u.exclude_race_ids or [])
        buy_minutes = int(cfg.backtest.buy_t_minus_minutes)

        query = text("""
            WITH buy_times AS (
                SELECT
                    r.race_id,
                    (
                        (r.date::timestamp + r.start_time)
                        - make_interval(mins => :buy_minutes)
                    ) AS buy_time
                FROM fact_race r
                WHERE r.date BETWEEN :min_date AND :max_date
                  AND r.start_time IS NOT NULL
                  AND (:race_ids_len = 0 OR r.race_id = ANY(:race_ids))
                  AND (:track_codes_len = 0 OR r.track_code = ANY(:track_codes))
                  AND (:require_results = FALSE OR EXISTS (
                        SELECT 1 FROM fact_result res
                        WHERE res.race_id = r.race_id
                          AND res.finish_pos IS NOT NULL
                  ))
                  AND (:require_ts_win = FALSE OR EXISTS (
                        SELECT 1 FROM odds_ts_win o
                        WHERE o.race_id = r.race_id
                          AND o.odds > 0
                          AND o.asof_time <= (
                                (r.date::timestamp + r.start_time)
                                - make_interval(mins => :buy_minutes)
                          )
                  ))
                  AND (:exclude_len = 0 OR NOT (r.race_id = ANY(:exclude_race_ids)))
            ),
            latest_features AS (
                SELECT DISTINCT ON (f.race_id, f.horse_id)
                    f.race_id,
                    f.horse_id,
                    f.payload,
                    f.asof_time
                FROM features f
                JOIN buy_times bt ON f.race_id = bt.race_id
                WHERE f.feature_version = :feature_version
                  AND f.asof_time <= bt.buy_time
                ORDER BY f.race_id, f.horse_id, f.asof_time DESC
            )
            SELECT lf.race_id, lf.payload
            FROM latest_features lf
        """)

        def _fetch_metric_values(
            min_date: str, max_date: str, race_ids: list[str] | None = None
        ) -> list[float]:
            race_ids_list = list(race_ids or [])
            rows = session.execute(
                query,
                {
                    "min_date": min_date,
                    "max_date": max_date,
                    "buy_minutes": buy_minutes,
                    "feature_version": FeatureBuilder.VERSION,
                    "race_ids_len": len(race_ids_list),
                    "race_ids": race_ids_list,
                    "track_codes_len": len(track_codes),
                    "track_codes": track_codes,
                    "require_results": bool(u.require_results),
                    "require_ts_win": bool(u.require_ts_win),
                    "exclude_len": len(exclude_race_ids),
                    "exclude_race_ids": exclude_race_ids,
                },
            ).fetchall()

            values_by_race: dict[str, float] = {}
            for r in rows:
                row = dict(r._mapping)
                race_id = row.get("race_id")
                payload = row.get("payload") or {}
                value = payload.get(metric)
                try:
                    value = float(value) if value is not None and np.isfinite(value) else None
                except Exception:
                    value = None
                if value is None:
                    continue
                if race_id and race_id not in values_by_race:
                    values_by_race[str(race_id)] = float(value)
            return list(values_by_race.values())

        def _metric_stats(values: list[float]) -> dict:
            if not values:
                return {
                    "min": None,
                    "p50": None,
                    "p90": None,
                    "max": None,
                }
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) == 0:
                return {
                    "min": None,
                    "p50": None,
                    "p90": None,
                    "max": None,
                }
            return {
                "min": float(np.min(arr)),
                "p50": float(np.quantile(arr, 0.50)),
                "p90": float(np.quantile(arr, 0.90)),
                "max": float(np.max(arr)),
            }

        fit_start = args.train_start
        fit_end = args.train_end
        if fit_scope == "train_valid" and args.valid_end:
            fit_end = args.valid_end

        fit_race_ids = None
        fit_population_note = None
        if fit_population == "precap_candidate_races":
            if model is None:
                rcfg.enabled = False
                meta["status"] = "disabled_model_missing"
                meta["fit_population_note"] = "model_required_for_precap_candidates"
                (artifacts_dir / "race_cost_cap.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(
                    f"[capfit] window={args.name} metric={metric} q={q} cap_value=None "
                    "n_train_races=0 status=model_missing"
                )
                return meta

            orig_state = {
                "enabled": bool(getattr(rcfg, "enabled", False)),
                "cap_mode": str(getattr(rcfg, "cap_mode", "fixed") or "fixed"),
                "max_takeout_implied": getattr(rcfg, "max_takeout_implied", None),
                "max_overround_sum_inv": getattr(rcfg, "max_overround_sum_inv", None),
            }
            stake_damp_cfg = getattr(cfg.betting, "stake_odds_damp", None)
            orig_stake_damp_enabled = None
            if stake_damp_cfg is not None:
                orig_stake_damp_enabled = bool(getattr(stake_damp_cfg, "enabled", False))
            orig_odds_floor_rc = float(getattr(cfg.betting, "odds_floor_min_odds", 0.0) or 0.0)
            cfg.betting.odds_floor_min_odds = 0.0
            rcfg.enabled = False
            rcfg.cap_mode = "fixed"
            rcfg.max_takeout_implied = None
            rcfg.max_overround_sum_inv = None
            if stake_damp_cfg is not None:
                stake_damp_cfg.enabled = False
            try:
                engine_fit = BacktestEngine(
                    session,
                    slippage_table=slippage_table,
                    odds_band_bias=odds_band_bias,
                    ev_upper_cap=ev_upper_cap,
                    uncertainty_shrink=uncertainty_shrink,
                )
                bt_fit = engine_fit.run(
                    start_date=fit_start,
                    end_date=fit_end,
                    model=model,
                    initial_bankroll=initial_bankroll,
                )
            finally:
                rcfg.enabled = orig_state["enabled"]
                rcfg.cap_mode = orig_state["cap_mode"]
                rcfg.max_takeout_implied = orig_state["max_takeout_implied"]
                rcfg.max_overround_sum_inv = orig_state["max_overround_sum_inv"]
                cfg.betting.odds_floor_min_odds = orig_odds_floor_rc
                if stake_damp_cfg is not None and orig_stake_damp_enabled is not None:
                    stake_damp_cfg.enabled = orig_stake_damp_enabled

            fit_race_ids = sorted({br.bet.race_id for br in bt_fit.bets})
            meta["n_fit_races_input"] = int(len(fit_race_ids))
            fit_population_note = "candidates=races_with_bets_under_baseline_gating"
            if not fit_race_ids:
                rcfg.enabled = False
                meta["status"] = "disabled_no_candidate_races"
                meta["fit_population_note"] = fit_population_note
                (artifacts_dir / "race_cost_cap.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(
                    f"[capfit] window={args.name} metric={metric} q={q} cap_value=None "
                    "n_train_races=0 status=no_candidates"
                )
                return meta

        fit_values = _fetch_metric_values(fit_start, fit_end, race_ids=fit_race_ids)
        n_races = len(fit_values)
        meta["n_train_races_used"] = n_races
        meta["fit_population_note"] = fit_population_note
        if fit_values:
            uniq = {f"{float(v):.6f}" for v in fit_values if v is not None and np.isfinite(v)}
            meta["fit_metric_unique_values"] = int(len(uniq))
        fit_stats = _metric_stats(fit_values)
        meta["fit_metric_min"] = fit_stats["min"]
        meta["fit_metric_p50"] = fit_stats["p50"]
        meta["fit_metric_p90"] = fit_stats["p90"]
        meta["fit_metric_max"] = fit_stats["max"]

        if n_races < min_races:
            rcfg.enabled = False
            meta["status"] = "disabled_insufficient_races"
        else:
            cap_value = float(np.quantile(fit_values, q))
            if not np.isfinite(cap_value):
                cap_value = None
            if cap_value is None:
                rcfg.enabled = False
                meta["status"] = "disabled_invalid_cap"
            else:
                if metric == "takeout_implied":
                    rcfg.max_takeout_implied = cap_value
                    rcfg.max_overround_sum_inv = None
                else:
                    rcfg.max_overround_sum_inv = cap_value
                    rcfg.max_takeout_implied = None
                meta["cap_value"] = cap_value
                meta["status"] = "ok"

        test_values = _fetch_metric_values(args.test_start, args.test_end)
        test_stats = _metric_stats(test_values)
        meta["test_metric_p90"] = test_stats["p90"]
        meta["test_metric_max"] = test_stats["max"]
        flag_gt = (
            meta.get("fit_metric_max") is not None
            and meta.get("test_metric_max") is not None
            and float(meta["test_metric_max"]) > float(meta["fit_metric_max"])
        )
        meta["flag_test_max_gt_fit_max"] = bool(flag_gt)

        frac_excluded = None
        cap_value = meta.get("cap_value")
        if cap_value is not None and test_values:
            n_ex = sum(
                1
                for v in test_values
                if v is not None and np.isfinite(v) and float(v) > float(cap_value)
            )
            frac_excluded = float(n_ex) / float(len(test_values)) if test_values else None
        meta["frac_test_races_excluded"] = frac_excluded
        meta["frac_test_bets_excluded"] = None

        (artifacts_dir / "race_cost_cap.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        cap_str = f"{meta['cap_value']:.6f}" if meta.get("cap_value") is not None else "None"
        print(
            f"[capfit] window={args.name} metric={metric} q={q:.3f} cap_value={cap_str} "
            f"n_train_races={n_races}"
        )
        return meta

    def _race_logloss_from_probs(
        race_ids: pd.Series,
        y_true: pd.Series,
        p_used: pd.Series,
        clip_eps: float,
    ) -> float | None:
        if race_ids is None or y_true is None or p_used is None:
            return None
        df = pd.DataFrame(
            {
                "race_id": race_ids.values,
                "y": pd.to_numeric(y_true, errors="coerce"),
                "p": pd.to_numeric(p_used, errors="coerce"),
            }
        ).dropna(subset=["race_id", "y", "p"])
        if df.empty:
            return None
        sum_p = df.groupby("race_id", dropna=False)["p"].sum()
        df = df.join(sum_p, on="race_id", rsuffix="_sum")
        df = df[df["p_sum"] > 0]
        if df.empty:
            return None
        df["p_norm"] = df["p"] / df["p_sum"]
        winners = df[df["y"] == 1]
        if winners.empty:
            return None
        p_win = winners["p_norm"].clip(float(clip_eps), 1.0 - float(clip_eps))
        return float(-np.mean(np.log(p_win)))

    def _run_valid_backtest(model, use_softmax: bool) -> dict:
        rs_cfg = getattr(cfg.model, "race_softmax", None)
        selector_cfg = getattr(rs_cfg, "selector", None) if rs_cfg else None
        orig_enabled = bool(getattr(rs_cfg, "enabled", False)) if rs_cfg else False
        orig_selector_enabled = (
            bool(getattr(selector_cfg, "enabled", False)) if selector_cfg else False
        )
        if rs_cfg is not None:
            rs_cfg.enabled = bool(use_softmax)
            if selector_cfg is not None:
                selector_cfg.enabled = False
        try:
            engine_valid = BacktestEngine(
                session,
                slippage_table=slippage_table,
                odds_band_bias=odds_band_bias,
                ev_upper_cap=ev_upper_cap,
                uncertainty_shrink=uncertainty_shrink,
            )
            bt_valid = engine_valid.run(
                start_date=args.valid_start,
                end_date=args.valid_end,
                model=model,
                initial_bankroll=initial_bankroll,
            )
        finally:
            if rs_cfg is not None:
                rs_cfg.enabled = orig_enabled
            if selector_cfg is not None:
                selector_cfg.enabled = orig_selector_enabled

        valid_roi = (
            (bt_valid.total_profit / bt_valid.total_stake) if bt_valid.total_stake > 0 else None
        )
        return {
            "roi": valid_roi,
            "n_bets": int(bt_valid.n_bets),
            "total_stake": float(bt_valid.total_stake),
            "total_profit": float(bt_valid.total_profit),
            "total_payout": float(bt_valid.total_payout),
            "max_drawdown": float(bt_valid.max_drawdown),
        }

    race_ids_train = _race_ids_between(args.train_start, args.train_end)
    race_ids_valid = (
        _race_ids_between(args.valid_start, args.valid_end)
        if args.valid_start and args.valid_end
        else []
    )
    race_ids_test = _race_ids_between(args.test_start, args.test_end)

    # slippage推定（test期間を参照しない）
    slippage_summary = None
    estimated_mult = None
    slippage_table = None
    slippage_table_path = None
    slippage_table_meta = None
    use_slip_table = bool(
        getattr(cfg.betting, "slippage_table", None) and cfg.betting.slippage_table.enabled
    )

    # slippage推定の対象期間（testは参照しない）
    # - baseline（global multiplier推定）は train〜valid まで使う（従来どおり）
    # - TicketB（slippage_table）は train のみ（リーク防止を強める）
    slip_end_global = args.valid_end if args.valid_end else args.train_end
    slip_end_for_estimation = args.train_end if use_slip_table else slip_end_global
    df_slip = None
    if args.estimate_closing_mult or use_slip_table:
        df_slip = fetch_odds_final_to_buy(
            session,
            start_date=args.train_start,
            end_date=slip_end_for_estimation,
            buy_t_minus_minutes=cfg.backtest.buy_t_minus_minutes,
        )
        slippage_summary = summarize_ratio_final_to_buy(df_slip)
        estimated_mult = recommend_closing_odds_multiplier(
            df_slip, quantile=args.closing_mult_quantile
        )

        # in-memoryで上書き（以降のEV/stake計算に反映）
        if args.estimate_closing_mult and not use_slip_table:
            cfg.betting.closing_odds_multiplier = float(estimated_mult)
        # TicketB: table型が有効なら二重計上を避けるため global multiplier は固定
        if use_slip_table:
            cfg.betting.closing_odds_multiplier = 1.0

    # 再現性向上: 実行時に使った設定と race_id リストを保存
    (run_dir / "race_ids_train.txt").write_text("\n".join(race_ids_train) + "\n", encoding="utf-8")
    (run_dir / "race_ids_valid.txt").write_text("\n".join(race_ids_valid) + "\n", encoding="utf-8")
    (run_dir / "race_ids_test.txt").write_text("\n".join(race_ids_test) + "\n", encoding="utf-8")

    # 特徴量生成（idempotent / 再利用で高速化）
    n_feat_train = build_features(session, race_ids_train, skip_existing=True)
    n_feat_valid = (
        build_features(session, race_ids_valid, skip_existing=True) if race_ids_valid else 0
    )
    n_feat_test = build_features(session, race_ids_test, skip_existing=True)

    # TicketB: slippage table 推定（train〜validまで。testは参照しない）
    if use_slip_table:
        if df_slip is None:
            df_slip = fetch_odds_final_to_buy(
                session,
                start_date=args.train_start,
                end_date=args.train_end,
                buy_t_minus_minutes=cfg.backtest.buy_t_minus_minutes,
            )
        df_feat = fetch_slippage_feature_snapshot(
            session,
            start_date=args.train_start,
            end_date=args.train_end,
            buy_t_minus_minutes=cfg.backtest.buy_t_minus_minutes,
        )
        df_train = df_slip.merge(df_feat, on=["race_id", "horse_no"], how="left")

        st_cfg = cfg.betting.slippage_table
        slippage_table = fit_slippage_table(
            df_train,
            quantile=float(st_cfg.quantile),
            min_count=int(st_cfg.min_count),
            odds_bins=list(st_cfg.odds_bins or []),
            ts_vol_quantiles=list(st_cfg.ts_vol_quantiles or []),
            use_snap_age=bool(st_cfg.use_snap_age),
            snap_age_bins=list(st_cfg.snap_age_bins or []),
        )
        slippage_table_meta = slippage_table.to_dict()
        slippage_table_path = run_dir / "slippage_table.json"
        slippage_table_path.write_text(
            json.dumps(slippage_table_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # 学習（校正器含む）
    model_path = artifacts_dir / "model.pkl"
    calibrator_path = artifacts_dir / "calibrator.pkl"
    model, train_metrics = train_model(
        session,
        train_start=args.train_start,
        train_end=args.train_end,
        valid_start=args.valid_start,
        valid_end=args.valid_end,
        model_path=model_path,
        calibrator_path=calibrator_path,
    )

    # Ticket N2: odds帯バイアス補正（favorite–longshot bias）
    odds_band_bias = None
    odds_band_bias_path = None
    odds_band_bias_meta = None
    use_odds_bias = bool(
        getattr(cfg.betting, "odds_band_bias", None) and cfg.betting.odds_band_bias.enabled
    )
    if use_odds_bias:
        # train期間のみ（リーク防止）
        X_tr, y_tr, p_mkt_tr, race_ids_tr = prepare_training_data(
            session,
            min_date=args.train_start,
            max_date=args.train_end,
            buy_t_minus_minutes=cfg.backtest.buy_t_minus_minutes,
        )
        if X_tr is not None and len(X_tr) > 0:
            X_tr2 = _align_features(X_tr, model.feature_names)
            segments_tr = None
            if getattr(model, "blend_segmented", None) and model.blend_segmented.get("enabled"):
                segments_tr = _surface_segments_from_race_ids(
                    session, race_ids_tr, X_tr.get("is_turf")
                )
            p_cal_tr = model.predict(X_tr2, p_mkt_tr, calibrate=True, segments=segments_tr)
            bias_cfg = cfg.betting.odds_band_bias
            odds_band_bias = OddsBandBias.fit_from_training_data(
                X_tr,
                y_tr,
                pd.Series(p_cal_tr),
                odds_bins=list(bias_cfg.odds_bins or []),
                min_count=int(bias_cfg.min_count),
                lambda_shrink=int(getattr(bias_cfg, "lambda_shrink", 0)),
                enforce_monotone=bool(getattr(bias_cfg, "enforce_monotone", False)),
                k_clip=tuple(bias_cfg.k_clip),
                odds_col="odds",
            )
            odds_band_bias_meta = odds_band_bias.to_dict()
            odds_band_bias_path = artifacts_dir / "odds_band_bias.json"
            odds_band_bias_path.write_text(
                json.dumps(odds_band_bias_meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    # Ticket S2: uncertainty shrink from train p_cal
    uncertainty_shrink = None
    uncertainty_shrink_path = None
    uncertainty_shrink_meta = None
    use_uncertainty = bool(
        getattr(cfg.betting, "uncertainty", None) and cfg.betting.uncertainty.enabled
    )
    if use_uncertainty:
        X_tr, _, p_mkt_tr, race_ids_tr = prepare_training_data(
            session,
            min_date=args.train_start,
            max_date=args.train_end,
            buy_t_minus_minutes=cfg.backtest.buy_t_minus_minutes,
        )
        if X_tr is not None and len(X_tr) > 0:
            X_tr2 = _align_features(X_tr, model.feature_names)
            segments_tr = None
            if getattr(model, "blend_segmented", None) and model.blend_segmented.get("enabled"):
                segments_tr = _surface_segments_from_race_ids(
                    session, race_ids_tr, X_tr.get("is_turf")
                )
            p_cal_tr = model.predict(X_tr2, p_mkt_tr, calibrate=True, segments=segments_tr)
            u_cfg = cfg.betting.uncertainty
            uncertainty_shrink = UncertaintyShrink.fit_from_p_cal(
                p_cal_tr,
                n_bins=int(u_cfg.n_bins),
                n0=float(u_cfg.n0),
                min_mult=float(u_cfg.min_mult),
            )
            uncertainty_shrink_meta = uncertainty_shrink.to_dict()
            uncertainty_shrink_path = artifacts_dir / "uncertainty_shrink.json"
            uncertainty_shrink_path.write_text(
                json.dumps(uncertainty_shrink_meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    # Ticket G2: EV上限計算（train期間のbet候補EV分布から）
    ev_upper_cap = None
    ev_upper_cap_path = None
    ev_upper_cap_meta = None
    ev_upper_cap_cfg = getattr(cfg.betting, "ev_upper_cap", None)
    use_ev_upper_cap = (
        bool(ev_upper_cap_cfg and ev_upper_cap_cfg.get("enabled", False))
        if isinstance(ev_upper_cap_cfg, dict)
        else False
    )
    if use_ev_upper_cap:
        # train期間でbet候補を生成してEV分布を計算
        # 軽量な方法：train期間のデータに対してバックテストを実行してbet候補を取得
        engine_train = BacktestEngine(
            session,
            slippage_table=slippage_table,
            odds_band_bias=odds_band_bias,
        )
        bt_train = engine_train.run(
            start_date=args.train_start,
            end_date=args.train_end,
            model=model,
            initial_bankroll=initial_bankroll,
        )

        # bet候補のEV分布からquantileを計算
        if bt_train.bets and len(bt_train.bets) > 0:
            ev_values = [b.bet.ev for b in bt_train.bets if b.bet.ev is not None]
            if ev_values:
                q_hi = float(ev_upper_cap_cfg.get("quantile", 0.95))
                ev_upper_cap = float(np.quantile(ev_values, q_hi))
                ev_upper_cap_meta = {
                    "enabled": True,
                    "quantile": q_hi,
                    "ev_upper_cap": ev_upper_cap,
                    "n_bets_train": len(ev_values),
                    "ev_min": float(np.min(ev_values)),
                    "ev_max": float(np.max(ev_values)),
                    "ev_median": float(np.median(ev_values)),
                }
                ev_upper_cap_path = artifacts_dir / "ev_upper_cap.json"
                ev_upper_cap_path.write_text(
                    json.dumps(ev_upper_cap_meta, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(
                    "EV upper cap computed: "
                    f"quantile={q_hi}, cap={ev_upper_cap:.4f}, n_bets={len(ev_values)}"
                )

    def _fit_odds_dynamics_filter() -> dict | None:
        cfg_od = getattr(cfg.betting, "odds_dynamics_filter", None)
        if not cfg_od or not bool(getattr(cfg_od, "enabled", False)):
            return None

        threshold_mode = str(getattr(cfg_od, "threshold_mode", "quantile") or "quantile").lower()
        if threshold_mode != "quantile":
            cfg_od.enabled = False
            meta = {
                "enabled": False,
                "status": "disabled_invalid_mode",
                "threshold_mode": threshold_mode,
            }
            (artifacts_dir / "odds_dynamics_filter.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return meta

        q = getattr(cfg_od, "quantile", None)
        try:
            q = float(q) if q is not None else None
        except Exception:
            q = None
        if q is None or not (0.0 < q < 1.0):
            cfg_od.enabled = False
            meta = {
                "enabled": False,
                "status": "disabled_invalid_quantile",
                "quantile": q,
            }
            (artifacts_dir / "odds_dynamics_filter.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return meta

        fit_scope = str(getattr(cfg_od, "fit_scope", "train_valid") or "train_valid").lower()
        if fit_scope not in ("train", "train_valid"):
            fit_scope = "train_valid"

        fit_population = str(
            getattr(cfg_od, "fit_population", "candidates_with_bets_under_baseline_gating")
            or "candidates_with_bets_under_baseline_gating"
        ).lower()
        if fit_population not in ("candidates_with_bets_under_baseline_gating",):
            fit_population = "candidates_with_bets_under_baseline_gating"

        metric = str(getattr(cfg_od, "metric", "odds_delta_log") or "odds_delta_log").lower()
        lookback = int(getattr(cfg_od, "lookback_minutes", 10) or 10)
        direction = str(getattr(cfg_od, "direction", "exclude_high") or "exclude_high").lower()

        fit_start = args.train_start
        fit_end = args.train_end
        if fit_scope == "train_valid" and args.valid_end:
            fit_end = args.valid_end

        meta = {
            "enabled": True,
            "status": None,
            "threshold_mode": threshold_mode,
            "metric": metric,
            "lookback_minutes": lookback,
            "direction": direction,
            "quantile": q,
            "fit_scope": fit_scope,
            "fit_population": fit_population,
            "n_fit_bets": 0,
            "threshold": None,
            "fit_metric_min": None,
            "fit_metric_p50": None,
            "fit_metric_p90": None,
            "fit_metric_max": None,
            "fit_metric_unique_values": None,
        }

        orig_enabled = bool(getattr(cfg_od, "enabled", False))
        orig_threshold = getattr(cfg_od, "threshold", None)
        orig_odds_floor = float(getattr(cfg.betting, "odds_floor_min_odds", 0.0) or 0.0)
        cfg.betting.odds_floor_min_odds = 0.0
        cfg_od.enabled = False
        cfg_od.threshold = None

        try:
            engine_fit = BacktestEngine(
                session,
                slippage_table=slippage_table,
                odds_band_bias=odds_band_bias,
                ev_upper_cap=ev_upper_cap,
                uncertainty_shrink=uncertainty_shrink,
            )
            bt_fit = engine_fit.run(
                start_date=fit_start,
                end_date=fit_end,
                model=model,
                initial_bankroll=initial_bankroll,
            )
        finally:
            cfg_od.enabled = orig_enabled
            cfg_od.threshold = orig_threshold
            cfg.betting.odds_floor_min_odds = orig_odds_floor

        values = []
        for br in bt_fit.bets:
            extra = br.bet.extra or {}
            v = extra.get("odds_dyn_metric_value")
            try:
                v = float(v) if v is not None and np.isfinite(v) else None
            except Exception:
                v = None
            if v is not None:
                values.append(v)

        meta["n_fit_bets"] = int(len(values))
        if not values:
            cfg_od.enabled = False
            meta["status"] = "disabled_no_fit_values"
            (artifacts_dir / "odds_dynamics_filter.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return meta

        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            cfg_od.enabled = False
            meta["status"] = "disabled_no_fit_values"
            (artifacts_dir / "odds_dynamics_filter.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return meta

        meta["fit_metric_min"] = float(np.min(arr))
        meta["fit_metric_p50"] = float(np.quantile(arr, 0.50))
        meta["fit_metric_p90"] = float(np.quantile(arr, 0.90))
        meta["fit_metric_max"] = float(np.max(arr))
        meta["fit_metric_unique_values"] = int(len({f"{v:.6f}" for v in arr}))

        threshold = float(np.quantile(arr, q))
        cfg_od.threshold = threshold
        meta["threshold"] = threshold
        meta["status"] = "ok"

        (artifacts_dir / "odds_dynamics_filter.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return meta

    def _fit_odds_dyn_ev_margin() -> dict | None:
        cfg_odm = getattr(cfg.betting, "odds_dyn_ev_margin", None)
        if not cfg_odm or not bool(getattr(cfg_odm, "enabled", False)):
            return None

        fit_scope = str(getattr(cfg_odm, "fit_scope", "train_valid") or "train_valid").lower()
        if fit_scope not in ("train", "train_valid"):
            fit_scope = "train_valid"

        fit_population = str(
            getattr(cfg_odm, "fit_population", "candidates_with_bets_under_baseline_gating")
            or "candidates_with_bets_under_baseline_gating"
        ).lower()
        if fit_population not in ("candidates_with_bets_under_baseline_gating",):
            fit_population = "candidates_with_bets_under_baseline_gating"

        metric = str(getattr(cfg_odm, "metric", "odds_delta_log") or "odds_delta_log").lower()
        lookback = int(getattr(cfg_odm, "lookback_minutes", 5) or 5)
        direction = str(getattr(cfg_odm, "direction", "high") or "high").lower()

        fit_start = args.train_start
        fit_end = args.train_end
        if fit_scope == "train_valid" and args.valid_end:
            fit_end = args.valid_end

        meta = {
            "enabled": True,
            "status": None,
            "metric": metric,
            "lookback_minutes": lookback,
            "direction": direction,
            "fit_scope": fit_scope,
            "fit_population": fit_population,
            "n_fit_bets": 0,
            "ref": None,
            "ref_source": None,
            "fit_metric_min": None,
            "fit_metric_p50": None,
            "fit_metric_p90": None,
            "fit_metric_max": None,
            "fit_metric_unique_values": None,
            "slope": float(getattr(cfg_odm, "slope", 0.0) or 0.0),
        }

        orig_enabled = bool(getattr(cfg_odm, "enabled", False))
        orig_ref = getattr(cfg_odm, "ref", None)
        orig_odds_floor_ev = float(getattr(cfg.betting, "odds_floor_min_odds", 0.0) or 0.0)
        cfg.betting.odds_floor_min_odds = 0.0
        cfg_odm.enabled = False
        try:
            engine_fit = BacktestEngine(
                session,
                slippage_table=slippage_table,
                odds_band_bias=odds_band_bias,
                ev_upper_cap=ev_upper_cap,
                uncertainty_shrink=uncertainty_shrink,
            )
            bt_fit = engine_fit.run(
                start_date=fit_start,
                end_date=fit_end,
                model=model,
                initial_bankroll=initial_bankroll,
            )
        finally:
            cfg_odm.enabled = orig_enabled
            cfg_odm.ref = orig_ref
            cfg.betting.odds_floor_min_odds = orig_odds_floor_ev

        values = []
        for br in bt_fit.bets:
            extra = br.bet.extra or {}
            v = extra.get("odds_dyn_ev_score")
            try:
                v = float(v) if v is not None and np.isfinite(v) else None
            except Exception:
                v = None
            if v is not None:
                values.append(v)

        meta["n_fit_bets"] = int(len(values))
        if not values:
            cfg_odm.enabled = False
            meta["status"] = "disabled_no_fit_values"
            (artifacts_dir / "odds_dyn_ev_margin.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return meta

        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            cfg_odm.enabled = False
            meta["status"] = "disabled_no_fit_values"
            (artifacts_dir / "odds_dyn_ev_margin.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return meta

        meta["fit_metric_min"] = float(np.min(arr))
        meta["fit_metric_p50"] = float(np.quantile(arr, 0.50))
        meta["fit_metric_p90"] = float(np.quantile(arr, 0.90))
        meta["fit_metric_max"] = float(np.max(arr))
        meta["fit_metric_unique_values"] = int(len({f"{v:.6f}" for v in arr}))

        ref = getattr(cfg_odm, "ref", None)
        try:
            ref_val = float(ref) if ref is not None and np.isfinite(ref) else None
        except Exception:
            ref_val = None
        if ref_val is None:
            ref_val = float(np.quantile(arr, 0.50))
            meta["ref_source"] = "fit_median"
        else:
            meta["ref_source"] = "config"
        cfg_odm.ref = ref_val
        meta["ref"] = ref_val
        meta["status"] = "ok"

        (artifacts_dir / "odds_dyn_ev_margin.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return meta

    _fit_race_cost_cap()
    odds_dyn_filter_meta = _fit_odds_dynamics_filter()
    odds_dyn_ev_margin_meta = _fit_odds_dyn_ev_margin()

    # バックテスト（test）
    # Ticket: race_softmax selector (valid-only)
    rs_cfg = getattr(cfg.model, "race_softmax", None)
    selector_cfg = getattr(rs_cfg, "selector", None) if rs_cfg else None
    selector_enabled = bool(
        rs_cfg
        and getattr(rs_cfg, "enabled", False)
        and selector_cfg
        and getattr(selector_cfg, "enabled", False)
    )
    selector_meta = None
    if selector_enabled:
        fallback = str(getattr(selector_cfg, "fallback", "baseline") or "baseline")
        selector_meta = {
            "chosen": fallback,
            "base": {},
            "softmax": {},
            "thresholds": {
                "min_valid_bets": int(getattr(selector_cfg, "min_valid_bets", 30)),
                "min_valid_bets_ratio": float(getattr(selector_cfg, "min_valid_bets_ratio", 0.50)),
                "min_delta_valid_roi": float(getattr(selector_cfg, "min_delta_valid_roi", 0.02)),
                "min_delta_valid_logloss": float(
                    getattr(selector_cfg, "min_delta_valid_logloss", 0.001)
                ),
            },
        }

        if args.valid_start and args.valid_end:
            try:
                X_valid, y_valid, p_mkt_valid, race_ids_valid = prepare_training_data(
                    session,
                    min_date=args.valid_start,
                    max_date=args.valid_end,
                    buy_t_minus_minutes=cfg.backtest.buy_t_minus_minutes,
                )
            except Exception as e:
                X_valid, y_valid, p_mkt_valid, race_ids_valid = None, None, None, None
                selector_meta["error"] = f"prepare_training_data_failed: {e}"

            if X_valid is not None and len(X_valid) > 0 and race_ids_valid is not None:
                X_valid2 = _align_features(X_valid, model.feature_names)
                p_mkt_val = pd.to_numeric(p_mkt_valid, errors="coerce").fillna(0.0).astype(float)

                segments_valid = None
                if getattr(model, "blend_segmented", None) and model.blend_segmented.get("enabled"):
                    segments_valid = _surface_segments_from_race_ids(
                        session, race_ids_valid, X_valid.get("is_turf")
                    )

                p_base = model.predict(X_valid2, p_mkt_val, calibrate=True, segments=segments_valid)
                base_logloss = _race_logloss_from_probs(
                    race_ids_valid,
                    y_valid,
                    pd.Series(p_base),
                    float(getattr(rs_cfg, "clip_eps", 1e-6)),
                )

                rs_params = getattr(model, "race_softmax_params", None) or {}
                w = rs_params.get("w")
                t = rs_params.get("T")
                softmax_logloss = None
                if w is not None and t is not None:
                    if model.use_market_offset:
                        p_model_val = model.predict(
                            X_valid2, p_mkt_val, calibrate=False, segments=segments_valid
                        )
                    else:
                        p_model_val = (
                            model.lgb_model.predict(X_valid2)
                            if model.lgb_model is not None
                            else np.zeros(len(X_valid2))
                        )
                    df_rs = pd.DataFrame(
                        {
                            "race_id": race_ids_valid.values,
                            "p_model": np.asarray(p_model_val, dtype=float),
                            "p_mkt": p_mkt_val.values,
                        }
                    )
                    p_soft = apply_race_softmax(
                        df_rs,
                        w=float(w),
                        t=float(t),
                        score_space=str(getattr(rs_cfg, "score_space", "logit")),
                        clip_eps=float(getattr(rs_cfg, "clip_eps", 1e-6)),
                    )
                    softmax_logloss = _race_logloss_from_probs(
                        race_ids_valid,
                        y_valid,
                        pd.Series(p_soft),
                        float(getattr(rs_cfg, "clip_eps", 1e-6)),
                    )

                base_bt = _run_valid_backtest(model, use_softmax=False)
                soft_bt = (
                    _run_valid_backtest(model, use_softmax=True)
                    if (w is not None and t is not None)
                    else {
                        "roi": None,
                        "n_bets": 0,
                        "total_stake": 0.0,
                        "total_profit": 0.0,
                    }
                )

                selector_meta["base"] = {
                    "valid_roi": base_bt["roi"],
                    "valid_bets": base_bt["n_bets"],
                    "valid_race_logloss": base_logloss,
                }
                selector_meta["softmax"] = {
                    "valid_roi": soft_bt["roi"],
                    "valid_bets": soft_bt["n_bets"],
                    "valid_race_logloss": softmax_logloss,
                    "w": float(w) if w is not None else None,
                    "T": float(t) if t is not None else None,
                }

                min_bets = selector_meta["thresholds"]["min_valid_bets"]
                min_ratio = selector_meta["thresholds"]["min_valid_bets_ratio"]
                min_d_roi = selector_meta["thresholds"]["min_delta_valid_roi"]
                min_d_ll = selector_meta["thresholds"]["min_delta_valid_logloss"]

                cond_bets = soft_bt["n_bets"] >= min_bets
                cond_ratio = soft_bt["n_bets"] >= (min_ratio * base_bt["n_bets"])
                cond_roi = (
                    base_bt["roi"] is not None
                    and soft_bt["roi"] is not None
                    and (soft_bt["roi"] - base_bt["roi"] >= min_d_roi)
                )
                cond_ll = (
                    base_logloss is not None
                    and softmax_logloss is not None
                    and ((base_logloss - softmax_logloss) >= min_d_ll)
                )

                if cond_bets and cond_ratio and cond_roi and cond_ll:
                    selector_meta["chosen"] = "softmax"
                else:
                    selector_meta["chosen"] = fallback
                    selector_meta["selection_failed"] = {
                        "cond_bets": cond_bets,
                        "cond_ratio": cond_ratio,
                        "cond_roi": cond_roi,
                        "cond_logloss": cond_ll,
                    }
            else:
                selector_meta["error"] = "valid_data_missing"
        else:
            selector_meta["error"] = "valid_range_missing"

        model.race_softmax_selector = selector_meta
        if bool(getattr(selector_cfg, "save_decision", True)):
            sel_path = artifacts_dir / "race_softmax_selector.json"
            sel_path.write_text(
                json.dumps(selector_meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    # Ticket: race_cost cap selector (valid-only)
    rcfg = getattr(cfg.betting, "race_cost_filter", None)
    rc_selector_cfg = getattr(rcfg, "selector", None) if rcfg else None
    rc_selector_enabled = bool(
        rcfg
        and getattr(rcfg, "enabled", False)
        and rc_selector_cfg
        and getattr(rc_selector_cfg, "enabled", False)
    )
    rc_selector_meta = None
    if rc_selector_enabled:
        metric = str(getattr(rcfg, "metric", "takeout_implied") or "takeout_implied").lower()
        if metric not in ("takeout_implied", "overround_sum_inv"):
            metric = "takeout_implied"

        candidate_caps_raw = list(getattr(rc_selector_cfg, "candidate_caps", []) or [])
        candidate_caps = []
        for cap in candidate_caps_raw:
            if cap is None:
                candidate_caps.append(None)
                continue
            try:
                cap_f = float(cap)
            except Exception:
                continue
            if not np.isfinite(cap_f):
                continue
            candidate_caps.append(float(cap_f))
        # dedupe while preserving order
        seen = set()
        dedup_caps = []
        for cap in candidate_caps:
            key = "none" if cap is None else f"{cap:.6f}"
            if key in seen:
                continue
            seen.add(key)
            dedup_caps.append(cap)
        candidate_caps = dedup_caps
        if None not in candidate_caps:
            candidate_caps = [None] + candidate_caps

        min_valid_n_bets = int(getattr(rc_selector_cfg, "min_valid_n_bets", 80))
        min_valid_bets_ratio = float(getattr(rc_selector_cfg, "min_valid_bets_ratio", 0.0))
        min_delta_valid_roi = float(getattr(rc_selector_cfg, "min_delta_valid_roi", 0.02))
        min_delta_valid_roi_shrunk = getattr(rc_selector_cfg, "min_delta_valid_roi_shrunk", None)
        if min_delta_valid_roi_shrunk is None:
            min_delta_valid_roi_shrunk = min_delta_valid_roi
        n0 = float(getattr(rc_selector_cfg, "n0", 0.0))
        if not np.isfinite(n0) or n0 < 0:
            n0 = 0.0
        tie_breaker = str(
            getattr(rc_selector_cfg, "tie_breaker", "prefer_baseline") or "prefer_baseline"
        ).lower()
        cap_labels = [("none" if cap is None else f"{float(cap):.6f}") for cap in candidate_caps]
        candidate_set_id = f"valid_selector:{metric}:caps=" + "|".join(cap_labels)
        rc_selector_meta = {
            "metric": metric,
            "candidate_caps": candidate_caps,
            "candidate_set_id": candidate_set_id,
            "thresholds": {
                "min_valid_n_bets": min_valid_n_bets,
                "min_valid_bets_ratio": min_valid_bets_ratio,
                "min_delta_valid_roi": min_delta_valid_roi,
                "min_delta_valid_roi_shrunk": min_delta_valid_roi_shrunk,
                "n0": n0,
                "tie_breaker": tie_breaker,
            },
            "baseline": None,
            "candidates": [],
            "selected_cap": None,
            "selected_reason": None,
            "selected_from": None,
            "error": None,
        }

        orig_state = {
            "enabled": bool(getattr(rcfg, "enabled", False)),
            "cap_mode": str(getattr(rcfg, "cap_mode", "fixed") or "fixed"),
            "max_takeout_implied": getattr(rcfg, "max_takeout_implied", None),
            "max_overround_sum_inv": getattr(rcfg, "max_overround_sum_inv", None),
        }

        def _apply_race_cost_cap(cap_value):
            rcfg.cap_mode = "fixed"
            if cap_value is None:
                rcfg.enabled = False
                rcfg.max_takeout_implied = None
                rcfg.max_overround_sum_inv = None
                return
            rcfg.enabled = True
            if metric == "takeout_implied":
                rcfg.max_takeout_implied = float(cap_value)
                rcfg.max_overround_sum_inv = None
            else:
                rcfg.max_overround_sum_inv = float(cap_value)
                rcfg.max_takeout_implied = None

        if args.valid_start and args.valid_end:
            use_softmax = bool(
                getattr(getattr(cfg, "model", None), "race_softmax", None)
                and getattr(cfg.model.race_softmax, "enabled", False)
            )
            _apply_race_cost_cap(None)
            base_bt = _run_valid_backtest(model, use_softmax=use_softmax)
            base_valid_bets = int(base_bt.get("n_bets", 0))
            base_valid_roi = base_bt.get("roi")
            rc_selector_meta["baseline"] = {
                "cap": None,
                "valid_roi": base_valid_roi,
                "valid_bets": base_valid_bets,
                "valid_stake": float(base_bt.get("total_stake", 0.0)),
                "valid_profit": float(base_bt.get("total_profit", 0.0)),
            }
            base_roi = base_valid_roi

            for cap in candidate_caps:
                if cap is None:
                    continue
                _apply_race_cost_cap(cap)
                bt = _run_valid_backtest(model, use_softmax=use_softmax)
                cand_bets = int(bt.get("n_bets", 0))
                cand = {
                    "cap": cap,
                    "valid_roi": bt.get("roi"),
                    "valid_bets": cand_bets,
                    "valid_stake": float(bt.get("total_stake", 0.0)),
                    "valid_profit": float(bt.get("total_profit", 0.0)),
                }
                delta_roi = None
                if base_roi is not None and cand["valid_roi"] is not None:
                    delta_roi = float(cand["valid_roi"]) - float(base_roi)
                valid_bets_ratio = None
                if base_valid_bets > 0:
                    valid_bets_ratio = float(cand_bets) / float(base_valid_bets)
                delta_roi_shrunk = None
                if delta_roi is not None:
                    shrink = float(cand_bets) / float(cand_bets + n0) if cand_bets >= 0 else 0.0
                    delta_roi_shrunk = float(delta_roi) * float(shrink)
                cand["delta_valid_roi"] = delta_roi
                cand["valid_bets_ratio_to_base"] = valid_bets_ratio
                cand["delta_valid_roi_shrunk"] = delta_roi_shrunk

                cond_bets = cand_bets >= min_valid_n_bets
                if base_valid_bets <= 0:
                    cond_ratio = False
                else:
                    cond_ratio = (
                        valid_bets_ratio is not None and valid_bets_ratio >= min_valid_bets_ratio
                    )
                cond_delta = delta_roi_shrunk is not None and delta_roi_shrunk >= float(
                    min_delta_valid_roi_shrunk
                )
                eligible = cond_bets and cond_ratio and cond_delta
                reasons = []
                if not cond_bets:
                    reasons.append(f"valid_bets_lt_min({cand_bets}<{min_valid_n_bets})")
                if base_valid_bets <= 0:
                    reasons.append("baseline_bets_zero")
                elif valid_bets_ratio is None:
                    reasons.append("valid_bets_ratio_missing")
                elif not cond_ratio:
                    reasons.append(
                        f"ratio_to_base_lt_min({valid_bets_ratio:.3f}<{min_valid_bets_ratio:.3f})"
                    )
                if delta_roi_shrunk is None:
                    reasons.append("delta_valid_roi_missing")
                elif not cond_delta:
                    reasons.append(
                        f"delta_roi_shrunk_lt_min({delta_roi_shrunk:.6f}<{float(min_delta_valid_roi_shrunk):.6f})"
                    )
                cand["eligible"] = eligible
                cand["eligible_reason"] = "eligible" if eligible else "|".join(reasons)
                rc_selector_meta["candidates"].append(cand)

            eligible = [c for c in rc_selector_meta["candidates"] if c.get("eligible")]
            chosen = None
            if eligible:

                def _score(c):
                    if c.get("delta_valid_roi_shrunk") is not None:
                        return c.get("delta_valid_roi_shrunk")
                    if c.get("delta_valid_roi") is not None:
                        return c.get("delta_valid_roi")
                    return float("-inf")

                if tie_breaker == "prefer_more_bets":
                    eligible.sort(
                        key=lambda c: (
                            _score(c),
                            c.get("valid_bets") if c.get("valid_bets") is not None else 0,
                        ),
                        reverse=True,
                    )
                else:
                    eligible.sort(
                        key=lambda c: (_score(c),),
                        reverse=True,
                    )
                chosen = eligible[0]
                rc_selector_meta["selected_cap"] = chosen.get("cap")
                rc_selector_meta["selected_reason"] = "eligible_best_roi"
                rc_selector_meta["selected_from"] = candidate_set_id
            else:
                rc_selector_meta["selected_cap"] = None
                rc_selector_meta["selected_reason"] = "fallback_baseline"
                rc_selector_meta["selected_from"] = None

            _apply_race_cost_cap(rc_selector_meta["selected_cap"])
            if rc_selector_cfg is not None:
                rc_selector_cfg.selected_cap_value = rc_selector_meta["selected_cap"]
                rc_selector_cfg.selected_from = rc_selector_meta["selected_from"]
        else:
            rc_selector_meta["error"] = "valid_range_missing"
            rcfg.enabled = orig_state["enabled"]
            rcfg.cap_mode = orig_state["cap_mode"]
            rcfg.max_takeout_implied = orig_state["max_takeout_implied"]
            rcfg.max_overround_sum_inv = orig_state["max_overround_sum_inv"]

        if rc_selector_meta is not None:
            sel_path = artifacts_dir / "race_cost_cap_selector.json"
            sel_path.write_text(
                json.dumps(rc_selector_meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    bt = run_backtest(
        session,
        start_date=args.test_start,
        end_date=args.test_end,
        model=model,
        initial_bankroll=initial_bankroll,
        slippage_table=slippage_table,
        odds_band_bias=odds_band_bias,
        ev_upper_cap=ev_upper_cap,
        uncertainty_shrink=uncertainty_shrink,
    )

    # ベット明細をCSV出力（勝ち筋探索用）
    bets_csv = run_dir / "bets.csv"
    fieldnames = [
        "race_id",
        "horse_no",
        "asof_time",
        "stake",
        "stake_raw",
        "stake_clipped",
        "clip_reason",
        "stake_before",
        "stake_mult",
        "stake_after",
        "stake_damp_ref_odds",
        "stake_damp_power",
        "stake_damp_min_mult",
        "stake_damp_odds",
        "stake_damp_ratio",
        "stake_damp_mult_raw",
        "stake_damp_mult_clamped",
        "stake_damp_floor_unit",
        "odds_at_buy",
        "odds_final",
        "ratio_final_to_buy",
        "odds_effective",
        "p_hat",
        "ev",
        "market_blend_enabled",
        "market_prob_method",
        "market_blend_w",
        "p_blend",
        "ev_blend",
        "odds_band_blend",
        "t_ev",
        "odds_cap",
        "exclude_odds_band",
        "p_hat_pre_blend",
        "uncert_mult",
        "n_bin",
        # Ticket N2: odds帯バイアス補正
        "p_hat_raw",
        "p_hat_shrunk",
        "overlay_shrink_alpha",
        "p_adj",
        "ev_adj",
        "k_odds_band",
        "odds_band",
        "finish_pos",
        "is_win",
        "payout",
        "profit",
        # extra（時系列特徴量など）
        "p_mkt",
        "p_mkt_raw",
        "p_mkt_race",
        "overround_sum_inv",
        "takeout_implied",
        "base_min_ev",
        "takeout_ref",
        "takeout_slope",
        "min_ev_eff",
        "passed_takeout_ev_margin",
        "odds_dyn_ev_metric",
        "odds_dyn_ev_lookback_min",
        "odds_dyn_ev_score",
        "odds_dyn_ev_ref",
        "odds_dyn_ev_slope",
        "odds_dyn_ev_margin",
        "min_ev_eff_odds_dyn",
        "passed_odds_dyn_ev_margin",
        "odds_dyn_ev_margin_enabled",
        "odds_floor_min_odds",
        "odds_floor_odds_used",
        "passed_odds_floor",
        "race_cost_cap_mode",
        "race_cost_cap_value",
        "race_cost_cap_selected_from",
        "race_cost_filter_passed",
        "race_cost_filter_reason",
        "segblend_segment",
        "segblend_w_used",
        "segblend_w_global",
        "race_softmax_enabled",
        "race_softmax_w",
        "race_softmax_T",
        "rsx_selected",
        "rsx_w",
        "rsx_T",
        "p_hat_pre_softmax",
        "selection_mode",
        "daily_top_n_n",
        "daily_top_n_metric",
        "daily_top_n_min_value",
        "daily_rank_in_day",
        "daily_candidates_before",
        "daily_candidates_after_min",
        "daily_selected",
        "has_ts_odds",
        "odds_window_passed",
        "min_buy_odds",
        "max_buy_odds",
        "ev_cap_q",
        "ev_cap_thr",
        "ev_cap_passed",
        "ov_cap_q",
        "ov_cap_thr",
        "ov_cap_passed",
        # TicketB: slippage table（effective odds）
        "closing_odds_multiplier",
        "slip_used",
        "slip_r_hat",
        "slip_odds_bin",
        "slip_ts_vol_bin",
        "slip_snap_age_bin",
        "slip_bin_n",
        # Step4: フィルタ追跡
        "ts_vol_cap",
        "ts_vol_cap_passed",
        "odds_chg_5m",
        "odds_chg_10m",
        "odds_chg_30m",
        "odds_chg_60m",
        "p_mkt_chg_5m",
        "p_mkt_chg_10m",
        "p_mkt_chg_30m",
        "p_mkt_chg_60m",
        "log_odds_slope_60m",
        "log_odds_std_60m",
        "n_pts_60m",
        "snap_age_min",
        # Ticket G1: Residual Cap 追跡
        "resid",
        "resid_cap",
        "cap_value",
        "p_hat_capped",
    ]
    with open(bets_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for br in bt.bets:
            b = br.bet
            odds_final = getattr(br, "odds_final", None)
            ratio = None
            try:
                if odds_final is not None and b.odds_at_buy and b.odds_at_buy > 0:
                    ratio = float(odds_final) / float(b.odds_at_buy)
            except Exception:
                ratio = None

            extra = getattr(b, "extra", {}) or {}
            slip_meta = extra.get("slippage_meta") or {}
            bias_meta = extra.get("bias_meta") or {}
            row = {
                "race_id": b.race_id,
                "horse_no": b.horse_no,
                "asof_time": b.asof_time.isoformat(sep=" "),
                "stake": b.stake,
                "stake_raw": extra.get("stake_raw"),
                "stake_clipped": extra.get("stake_clipped"),
                "clip_reason": extra.get("clip_reason"),
                "stake_before": extra.get("stake_before"),
                "stake_mult": extra.get("stake_mult"),
                "stake_after": extra.get("stake_after"),
                "stake_damp_ref_odds": extra.get("stake_damp_ref_odds"),
                "stake_damp_power": extra.get("stake_damp_power"),
                "stake_damp_min_mult": extra.get("stake_damp_min_mult"),
                "stake_damp_odds": extra.get("stake_damp_odds"),
                "stake_damp_ratio": extra.get("stake_damp_ratio"),
                "stake_damp_mult_raw": extra.get("stake_damp_mult_raw"),
                "stake_damp_mult_clamped": extra.get("stake_damp_mult_clamped"),
                "stake_damp_floor_unit": extra.get("stake_damp_floor_unit"),
                "odds_at_buy": b.odds_at_buy,
                "odds_final": odds_final,
                "ratio_final_to_buy": ratio,
                "odds_effective": b.odds_effective,
                "p_hat": b.p_hat,
                "ev": b.ev,
                "market_blend_enabled": extra.get("market_blend_enabled"),
                "market_prob_method": extra.get("market_prob_method"),
                "market_blend_w": extra.get("market_blend_w"),
                "p_blend": extra.get("p_blend"),
                "ev_blend": extra.get("ev_blend"),
                "odds_band_blend": extra.get("odds_band_blend"),
                "t_ev": extra.get("t_ev"),
                "odds_cap": extra.get("odds_cap"),
                "exclude_odds_band": extra.get("exclude_odds_band"),
                "p_hat_pre_blend": extra.get("p_hat_pre_blend"),
                "uncert_mult": extra.get("uncert_mult"),
                "n_bin": extra.get("uncert_n_bin"),
                "p_hat_raw": extra.get("p_hat_raw"),
                "p_hat_shrunk": extra.get("p_hat_shrunk"),
                "overlay_shrink_alpha": extra.get("overlay_shrink_alpha"),
                "p_adj": extra.get("p_hat_adj"),
                "ev_adj": b.ev,
                "k_odds_band": bias_meta.get("k"),
                "odds_band": bias_meta.get("band"),
                "finish_pos": br.finish_pos,
                "is_win": int(br.is_win),
                "payout": br.payout,
                "profit": br.profit,
                # extra
                "p_mkt": extra.get("p_mkt"),
                "p_mkt_raw": extra.get("p_mkt_raw"),
                "p_mkt_race": extra.get("p_mkt_race"),
                "overround_sum_inv": extra.get("overround_sum_inv"),
                "takeout_implied": extra.get("takeout_implied"),
                "base_min_ev": extra.get("base_min_ev"),
                "takeout_ref": extra.get("takeout_ref"),
                "takeout_slope": extra.get("takeout_slope"),
                "min_ev_eff": extra.get("min_ev_eff"),
                "passed_takeout_ev_margin": extra.get("passed_takeout_ev_margin"),
                "odds_dyn_ev_metric": extra.get("odds_dyn_ev_metric"),
                "odds_dyn_ev_lookback_min": extra.get("odds_dyn_ev_lookback_min"),
                "odds_dyn_ev_score": extra.get("odds_dyn_ev_score"),
                "odds_dyn_ev_ref": extra.get("odds_dyn_ev_ref"),
                "odds_dyn_ev_slope": extra.get("odds_dyn_ev_slope"),
                "odds_dyn_ev_margin": extra.get("odds_dyn_ev_margin"),
                "min_ev_eff_odds_dyn": extra.get("min_ev_eff_odds_dyn"),
                "passed_odds_dyn_ev_margin": extra.get("passed_odds_dyn_ev_margin"),
                "odds_dyn_ev_margin_enabled": extra.get("odds_dyn_ev_margin_enabled"),
                "odds_floor_min_odds": extra.get("odds_floor_min_odds"),
                "odds_floor_odds_used": extra.get("odds_floor_odds_used"),
                "passed_odds_floor": extra.get("passed_odds_floor"),
                "race_cost_cap_mode": extra.get("race_cost_cap_mode"),
                "race_cost_cap_value": extra.get("race_cost_cap_value"),
                "race_cost_cap_selected_from": extra.get("race_cost_cap_selected_from"),
                "race_cost_filter_passed": extra.get("race_cost_filter_passed"),
                "race_cost_filter_reason": extra.get("race_cost_filter_reason"),
                "segblend_segment": extra.get("segblend_segment"),
                "segblend_w_used": extra.get("segblend_w_used"),
                "segblend_w_global": extra.get("segblend_w_global"),
                "race_softmax_enabled": extra.get("race_softmax_enabled"),
                "race_softmax_w": extra.get("race_softmax_w"),
                "race_softmax_T": extra.get("race_softmax_T"),
                "rsx_selected": extra.get("rsx_selected"),
                "rsx_w": extra.get("rsx_w"),
                "rsx_T": extra.get("rsx_T"),
                "p_hat_pre_softmax": extra.get("p_hat_pre_softmax"),
                "selection_mode": extra.get("selection_mode"),
                "daily_top_n_n": extra.get("daily_top_n_n"),
                "daily_top_n_metric": extra.get("daily_top_n_metric"),
                "daily_top_n_min_value": extra.get("daily_top_n_min_value"),
                "daily_rank_in_day": extra.get("daily_rank_in_day"),
                "daily_candidates_before": extra.get("daily_candidates_before"),
                "daily_candidates_after_min": extra.get("daily_candidates_after_min"),
                "daily_selected": extra.get("daily_selected"),
                "has_ts_odds": extra.get("has_ts_odds"),
                "odds_window_passed": extra.get("odds_window_passed"),
                "min_buy_odds": extra.get("min_buy_odds"),
                "max_buy_odds": extra.get("max_buy_odds"),
                "ev_cap_q": extra.get("ev_cap_q"),
                "ev_cap_thr": extra.get("ev_cap_thr"),
                "ev_cap_passed": extra.get("ev_cap_passed"),
                "ov_cap_q": extra.get("ov_cap_q"),
                "ov_cap_thr": extra.get("ov_cap_thr"),
                "ov_cap_passed": extra.get("ov_cap_passed"),
                "closing_odds_multiplier": extra.get("closing_odds_multiplier"),
                "slip_used": slip_meta.get("used"),
                "slip_r_hat": slip_meta.get("r_hat"),
                "slip_odds_bin": slip_meta.get("odds_bin"),
                "slip_ts_vol_bin": slip_meta.get("ts_vol_bin"),
                "slip_snap_age_bin": slip_meta.get("snap_age_bin"),
                "slip_bin_n": slip_meta.get("bin_n"),
                "ts_vol_cap": extra.get("ts_vol_cap"),
                "ts_vol_cap_passed": extra.get("ts_vol_cap_passed"),
                "odds_chg_5m": extra.get("odds_chg_5m"),
                "odds_chg_10m": extra.get("odds_chg_10m"),
                "odds_chg_30m": extra.get("odds_chg_30m"),
                "odds_chg_60m": extra.get("odds_chg_60m"),
                "p_mkt_chg_5m": extra.get("p_mkt_chg_5m"),
                "p_mkt_chg_10m": extra.get("p_mkt_chg_10m"),
                "p_mkt_chg_30m": extra.get("p_mkt_chg_30m"),
                "p_mkt_chg_60m": extra.get("p_mkt_chg_60m"),
                "log_odds_slope_60m": extra.get("log_odds_slope_60m"),
                "log_odds_std_60m": extra.get("log_odds_std_60m"),
                "n_pts_60m": extra.get("n_pts_60m"),
                "snap_age_min": extra.get("snap_age_min"),
            }
            w.writerow(row)

    # 確率品質（test）
    pq, _, rel = compute_prediction_quality(session, args.test_start, args.test_end, model)
    rel_img = report_dir / "reliability.png"
    plot_reliability(rel, rel_img, title=f"Reliability {args.test_start}..{args.test_end}")

    # レポート
    report_path = generate_report(
        bt,
        report_dir,
        name="backtest",
        pred_quality=pq,
        pred_quality_period=f"{args.test_start}..{args.test_end}",
        reliability_image=rel_img,
    )

    min_bankroll = getattr(bt, "min_bankroll", bt.final_bankroll)
    min_bankroll_frac = None
    if initial_bankroll > 0:
        min_bankroll_frac = float(min_bankroll) / float(initial_bankroll)
    risk_of_ruin_0_3 = min_bankroll_frac is not None and min_bankroll_frac <= 0.3
    risk_of_ruin_0_5 = min_bankroll_frac is not None and min_bankroll_frac <= 0.5
    risk_of_ruin_0_7 = min_bankroll_frac is not None and min_bankroll_frac <= 0.7

    def _race_cost_cap_summary():
        rcfg = getattr(cfg.betting, "race_cost_filter", None)
        metric = (
            str(getattr(rcfg, "metric", "takeout_implied") or "takeout_implied").lower()
            if rcfg
            else "takeout_implied"
        )
        if metric not in ("takeout_implied", "overround_sum_inv"):
            metric = "takeout_implied"
        cap_mode = str(getattr(rcfg, "cap_mode", "fixed") or "fixed").lower() if rcfg else "fixed"
        if cap_mode not in ("fixed", "train_quantile"):
            cap_mode = "fixed"
        fit_scope = str(getattr(rcfg, "cap_fit_scope", "train") or "train") if rcfg else "train"
        fit_population = (
            str(getattr(rcfg, "cap_fit_population", "all_races") or "all_races")
            if rcfg
            else "all_races"
        )
        cap_value = None
        if rcfg:
            if cap_mode == "train_quantile":
                cap_value = (
                    getattr(rcfg, "max_takeout_implied", None)
                    if metric == "takeout_implied"
                    else getattr(rcfg, "max_overround_sum_inv", None)
                )
            else:
                cap_value = getattr(rcfg, "max_takeout_implied", None)
                if cap_value is None:
                    cap_value = getattr(rcfg, "max_overround_sum_inv", None)
        try:
            cap_value = (
                float(cap_value) if cap_value is not None and np.isfinite(cap_value) else None
            )
        except Exception:
            cap_value = None

        selector_cfg = getattr(rcfg, "selector", None) if rcfg else None
        selected_from = None
        if selector_cfg is not None and bool(getattr(selector_cfg, "enabled", False)):
            selected_from = getattr(selector_cfg, "selected_from", None)
        mode_label = "none"
        if (
            selector_cfg is not None
            and bool(getattr(selector_cfg, "enabled", False))
            and selected_from is not None
        ):
            mode_label = "selected"
        elif rcfg is not None and bool(getattr(rcfg, "enabled", False)) and cap_value is not None:
            mode_label = "fixed"
        return {
            "race_cost_cap_mode": mode_label,
            "race_cost_cap_value": cap_value,
            "race_cost_cap_selected_from": selected_from,
            "race_cost_cap_metric": metric,
            "race_cost_cap_fit_scope": fit_scope,
            "race_cost_cap_fit_population": fit_population,
            "race_cost_cap_candidates": list(getattr(selector_cfg, "candidate_caps", []) or [])
            if selector_cfg
            else None,
        }

    race_cost_cap_summary = _race_cost_cap_summary()

    def _takeout_ev_margin_summary():
        te_cfg = getattr(cfg.betting, "takeout_ev_margin", None)
        if te_cfg is None:
            return {
                "enabled": False,
                "ref_takeout": None,
                "slope": None,
            }
        try:
            ref = float(getattr(te_cfg, "ref_takeout", 0.215))
        except Exception:
            ref = None
        try:
            slope = float(getattr(te_cfg, "slope", 0.0))
        except Exception:
            slope = None
        return {
            "enabled": bool(getattr(te_cfg, "enabled", False)),
            "ref_takeout": ref,
            "slope": slope,
        }

    takeout_ev_margin_summary = _takeout_ev_margin_summary()

    # valid backtest (selector uses valid-only; do not use test)
    valid_backtest = None
    if args.valid_start and args.valid_end:
        use_softmax_valid = bool(
            getattr(getattr(cfg, "model", None), "race_softmax", None)
            and getattr(cfg.model.race_softmax, "enabled", False)
        )
        valid_backtest = _run_valid_backtest(model, use_softmax=use_softmax_valid)

    # summary.json
    summary = {
        "name": args.name,
        "generated_at": ts,
        "config_used_path": rel_path(config_used_path, project_root),
        "train": {
            "start": args.train_start,
            "end": args.train_end,
            "n_races": len(race_ids_train),
            "features_saved": n_feat_train,
        },
        "valid": {
            "start": args.valid_start,
            "end": args.valid_end,
            "n_races": len(race_ids_valid),
            "features_saved": n_feat_valid,
        },
        "test": {
            "start": args.test_start,
            "end": args.test_end,
            "n_races": len(race_ids_test),
            "features_saved": n_feat_test,
        },
        "buy_t_minus_minutes": cfg.backtest.buy_t_minus_minutes,
        "closing_odds_multiplier": float(cfg.betting.closing_odds_multiplier),
        "closing_odds_multiplier_estimated": float(estimated_mult)
        if estimated_mult is not None
        else None,
        "closing_odds_multiplier_quantile": args.closing_mult_quantile
        if (args.estimate_closing_mult or use_slip_table)
        else None,
        "slippage_summary": slippage_summary.__dict__ if slippage_summary is not None else None,
        "slippage_table": {
            "enabled": bool(use_slip_table),
            "path": str(slippage_table_path) if slippage_table_path is not None else None,
            "meta": slippage_table_meta,
            "train_start": args.train_start,
            "train_end": args.train_end,
        },
        "odds_band_bias": {
            "enabled": bool(use_odds_bias),
            "path": str(odds_band_bias_path) if odds_band_bias_path is not None else None,
            "meta": odds_band_bias_meta,
            "train_start": args.train_start,
            "train_end": args.train_end,
        },
        "uncertainty_shrink": {
            "enabled": bool(use_uncertainty),
            "path": str(uncertainty_shrink_path) if uncertainty_shrink_path is not None else None,
            "meta": uncertainty_shrink_meta,
            "train_start": args.train_start,
            "train_end": args.train_end,
        },
        "odds_dynamics_filter": {
            "enabled": bool(getattr(cfg.betting.odds_dynamics_filter, "enabled", False)),
            "path": str(artifacts_dir / "odds_dynamics_filter.json"),
            "meta": odds_dyn_filter_meta,
            "fit_scope": getattr(cfg.betting.odds_dynamics_filter, "fit_scope", None),
            "fit_population": getattr(cfg.betting.odds_dynamics_filter, "fit_population", None),
        },
        "odds_dyn_ev_margin": {
            "enabled": bool(getattr(cfg.betting.odds_dyn_ev_margin, "enabled", False)),
            "path": str(artifacts_dir / "odds_dyn_ev_margin.json"),
            "meta": odds_dyn_ev_margin_meta,
            "fit_scope": getattr(cfg.betting.odds_dyn_ev_margin, "fit_scope", None),
            "fit_population": getattr(cfg.betting.odds_dyn_ev_margin, "fit_population", None),
        },
        "odds_floor": {
            "min_odds": float(getattr(cfg.betting, "odds_floor_min_odds", 0.0) or 0.0),
            "filtered_bets": int(getattr(bt, "odds_floor_filtered_bets", 0)),
            "filtered_stake": float(getattr(bt, "odds_floor_filtered_stake", 0.0)),
        },
        "stake_odds_damp": {
            "enabled": bool(
                getattr(getattr(cfg.betting, "stake_odds_damp", None), "enabled", False)
            ),
            "ref_odds": float(
                getattr(getattr(cfg.betting, "stake_odds_damp", None), "ref_odds", 0.0) or 0.0
            ),
            "power": float(
                getattr(getattr(cfg.betting, "stake_odds_damp", None), "power", 1.0) or 1.0
            ),
            "min_mult": float(
                getattr(getattr(cfg.betting, "stake_odds_damp", None), "min_mult", 0.0) or 0.0
            ),
            "mean_mult": float(getattr(bt, "stake_odds_damp_mean_mult", 1.0)),
            "low_odds_stake_before": float(
                getattr(bt, "stake_odds_damp_low_odds_stake_before", 0.0)
            ),
            "low_odds_stake_after": float(getattr(bt, "stake_odds_damp_low_odds_stake_after", 0.0)),
        },
        "train_metrics": train_metrics,
        "pred_quality": pq.__dict__,
        "valid_backtest": valid_backtest,
        "backtest": {
            "n_bets": bt.n_bets,
            "n_wins": bt.n_wins,
            "total_stake": bt.total_stake,
            "total_payout": bt.total_payout,
            "total_profit": bt.total_profit,
            "roi": bt.roi,
            "max_drawdown": bt.max_drawdown,
            "ending_bankroll": bt.final_bankroll,
            "log_growth": bt.log_growth,
            "max_drawdown_bankroll": bt.max_drawdown_bankroll,
            "min_bankroll": min_bankroll,
            "min_bankroll_frac": min_bankroll_frac,
            "risk_of_ruin_0_3": risk_of_ruin_0_3,
            "risk_of_ruin_0_5": risk_of_ruin_0_5,
            "risk_of_ruin_0_7": risk_of_ruin_0_7,
        },
        "race_cost_cap_mode": race_cost_cap_summary["race_cost_cap_mode"],
        "race_cost_cap_value": race_cost_cap_summary["race_cost_cap_value"],
        "race_cost_cap_selected_from": race_cost_cap_summary["race_cost_cap_selected_from"],
        "race_cost_filter": race_cost_cap_summary,
        "takeout_ev_margin": takeout_ev_margin_summary,
        "paths": {
            "run_dir": str(run_dir),
            "model": str(model_path),
            "calibrator": str(calibrator_path),
            "report": str(report_path),
            "reliability_plot": str(rel_img),
        },
    }

    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    # metrics.json (non-fatal if extraction fails)
    try:
        from keiba.eval.extract_metrics import write_metrics_json

        write_metrics_json(run_dir, run_kind="holdout")
        print("metrics:", run_dir / "metrics.json")
    except Exception as e:
        print(f"[metrics] failed: {e}")

    print("OK")
    print("run_dir:", run_dir)
    print("report:", report_path)
    print("summary:", run_dir / "summary.json")
    print("bets_csv:", bets_csv)


if __name__ == "__main__":
    main()
