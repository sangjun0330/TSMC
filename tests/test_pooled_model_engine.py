import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import tsm_pooled_model_engine as pooled_engine
from tsm_pooled_model_engine import (
    apply_tsm_layer,
    apply_tsm_calibration_route_spec,
    apply_paper_gate_snapshot_to_latest_predictions,
    apply_paper_gate_snapshot_to_quality_checks,
    apply_logit_shift,
    block_reasons,
    bootstrap_mean_diff,
    build_paper_gate_snapshot,
    build_quality_checks,
    build_stop_risk_calibration_summary,
    build_oof_metrics,
    combine_horizon_results,
    candidate_training_weights,
    choose_fold_consensus_trade_ready_threshold_v4,
    choose_fold_consensus_trade_ready_threshold_v5,
    choose_tsm_calibration_route,
    choose_tsm_calibration_route_v2,
    choose_tsm_scoring_route,
    choose_threshold,
    choose_stable_trade_ready_threshold_v3,
    fit_empirical_bayes,
    fold_frames,
    fit_probability_calibrator,
    fit_stop_risk_calibration_model,
    fit_direct_empirical_tsm_probability,
    fit_scope_logit_shift,
    fit_tsm_calibration_layer,
    metric_row,
    merge_latest_prediction_tables,
    next_required_evidence_action,
    paired_bootstrap_uplift,
    predict_empirical_bayes,
    pooled_feature_columns,
    prepare_dataset,
    apply_stop_risk_calibration,
    ensure_stop_hit_label,
    refresh_paper_gate_overlay,
    score_baseline_mask_with_policy,
    score_baseline_returns_with_policy,
    select_expanding_top_fraction_by_score,
    select_top_fraction_by_score,
    threshold_stability_from_fold_metrics,
    is_tsm_like_semiconductor,
    update_latest_snapshot,
    WalkForwardFold,
    EMBARGO_TRADING_DAYS,
    TsmCalibrationRouteSpec,
    add_risk_adjusted_selection_score,
    tsm_effective_scoring_route_pass,
)


class PooledModelEngineTests(unittest.TestCase):
    def test_combine_horizon_results_merges_latest_prediction_horizon_columns(self):
        results = {
            5: {
                "comparison": pd.DataFrame([{"split": "combined_test_holdout", "event_count": 50}]),
                "universe_latest_predictions": pd.DataFrame(
                    [{"symbol": "NVDA", "p_success_5d": 0.45, "threshold_5d": 0.19}]
                ),
                "top10_latest_predictions": pd.DataFrame(
                    [{"symbol": "NVDA", "p_success_5d": 0.45, "threshold_5d": 0.19}]
                ),
                "overlay": {"model_name": "model_5", "p_success_5d": 0.45, "threshold_5d": 0.19},
            },
            20: {
                "comparison": pd.DataFrame([{"split": "combined_test_holdout", "event_count": 100}]),
                "universe_latest_predictions": pd.DataFrame(
                    [
                        {
                            "symbol": "NVDA",
                            "p_success_20d": 0.40,
                            "threshold_20d": 0.20,
                            "decision_support_allowed": False,
                            "block_reasons": "LATEST_NOT_TRADE_READY",
                        }
                    ]
                ),
                "top10_latest_predictions": pd.DataFrame(
                    [{"symbol": "NVDA", "p_success_20d": 0.40, "threshold_20d": 0.20}]
                ),
                "overlay": {"model_name": "model_20", "p_success_20d": 0.40, "threshold_20d": 0.20},
            },
            60: {
                "comparison": pd.DataFrame([{"split": "combined_test_holdout", "event_count": 150}]),
                "universe_latest_predictions": pd.DataFrame(
                    [{"symbol": "NVDA", "p_success_60d": 0.55, "threshold_60d": 0.24}]
                ),
                "top10_latest_predictions": pd.DataFrame(
                    [{"symbol": "NVDA", "p_success_60d": 0.55, "threshold_60d": 0.24}]
                ),
                "overlay": {"model_name": "model_60", "p_success_60d": 0.55, "threshold_60d": 0.24},
            },
        }

        merged_latest = merge_latest_prediction_tables(results, "universe_latest_predictions").set_index("symbol")
        combined = combine_horizon_results(results)

        self.assertEqual(merged_latest.loc["NVDA", "p_success_20d"], 0.40)
        self.assertEqual(merged_latest.loc["NVDA", "p_success_5d"], 0.45)
        self.assertEqual(merged_latest.loc["NVDA", "p_success_60d"], 0.55)
        self.assertFalse(bool(merged_latest.loc["NVDA", "decision_support_allowed"]))
        self.assertEqual(combined["comparison"]["horizon_days"].tolist(), [5, 20, 60])
        self.assertEqual(combined["overlay"]["primary_decision_horizon_days"], 20)
        self.assertEqual(combined["overlay"]["p_success_5d"], 0.45)
        self.assertEqual(combined["overlay"]["p_success_60d"], 0.55)
        self.assertEqual(combined["overlay"]["model_name_5d"], "model_5")
        self.assertEqual(combined["overlay"]["model_name_60d"], "model_60")

    def test_stop_hit_label_is_one_minus_stop_survival(self):
        frame = pd.DataFrame({"label_stop_survival_20d": [1, 0, 1, np.nan]})

        labeled = ensure_stop_hit_label(frame)

        self.assertEqual(labeled["label_stop_hit_20d"].iloc[:3].tolist(), [0.0, 1.0, 0.0])
        self.assertTrue(np.isnan(labeled["label_stop_hit_20d"].iloc[3]))

    def test_stop_risk_calibration_guardrail_keeps_oos_ece_not_worse(self):
        labels = np.array(([0, 1] * 45) + ([0, 0, 1, 1] * 45), dtype=float)
        raw = np.where(labels == 1, 0.62, 0.38)
        raw[90:] = np.where(labels[90:] == 1, 0.55, 0.45)
        frame = pd.DataFrame(
            {
                "split": ["validation_2023"] * 90 + ["test_2024"] * 90 + ["final_holdout_2025_2026"] * 90,
                "label_stop_survival_20d": 1.0 - labels,
                "p_stop_hit_lgbm": raw,
                "candidate_tier": ["decision_trade_ready", "setup_context_score65", "relaxed_trigger_score65"] * 90,
                "symbol_group": ["fabless_ai_analog", "memory_storage", "foundry"] * 90,
                "is_decision_entry_candidate": [True] * 270,
            }
        )

        model = fit_stop_risk_calibration_model(frame)
        calibrated = apply_stop_risk_calibration(frame, model)
        summary = build_stop_risk_calibration_summary(calibrated, model)
        combined = summary[
            summary["split"].eq("combined_test_holdout")
            & summary["evaluation_scope"].eq("entry_research_all")
        ].iloc[0]

        self.assertTrue(str(model["calibration_method"]))
        self.assertLessEqual(combined["calibrated_ece"], combined["raw_ece"] + 1e-12)

    def test_fold_frames_embargoes_calibration_to_threshold_and_threshold_to_test(self):
        fold = WalkForwardFold(
            "wf_unit",
            "2020-01-01",
            "2020-12-31",
            "2021-01-01",
            "2021-01-29",
            "2021-02-01",
            "2021-03-31",
            "2021-04-01",
            "2021-05-31",
        )
        data = pd.DataFrame({"date": pd.date_range("2020-01-01", "2021-05-31", freq="B")})

        _, calibration, threshold, test = fold_frames(data, fold)

        expected_threshold_start = pd.Timestamp(fold.calibration_end) + pd.offsets.BDay(EMBARGO_TRADING_DAYS)
        expected_test_start = max(
            pd.Timestamp(fold.test_start) + pd.offsets.BDay(EMBARGO_TRADING_DAYS),
            pd.Timestamp(fold.threshold_end) + pd.offsets.BDay(EMBARGO_TRADING_DAYS),
        )
        self.assertEqual(calibration["date"].max(), pd.Timestamp(fold.calibration_end))
        self.assertGreaterEqual(threshold["date"].min(), expected_threshold_start)
        self.assertGreaterEqual(test["date"].min(), expected_test_start)

    def test_bootstrap_mean_diff_zero_iterations_is_diagnostic_nan(self):
        lower, p_value = bootstrap_mean_diff(
            pd.Series([1.0, 2.0, 3.0]),
            pd.Series([0.0, 1.0, 2.0]),
            iterations=0,
        )

        self.assertTrue(np.isnan(lower))
        self.assertTrue(np.isnan(p_value))

    def test_pooled_quality_rejects_actionability_fallback_latest_rows(self):
        data = pd.DataFrame(
            {
                "symbol": ["NVDA"] * 2,
                "is_decision_universe": [True, True],
                "is_decision_entry_candidate": [True, True],
                "is_trade_ready_entry_candidate": [True, True],
            }
        )
        latest = pd.DataFrame(
            [
                {
                    "symbol": "NVDA",
                    "prediction_source": "rule_fallback_diagnostic_only",
                }
            ]
        )

        quality = build_quality_checks(
            data,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {},
            pooled_dataset_quality_ok=True,
            universe_latest_predictions=latest,
        )
        checks = dict(zip(quality["check"], quality["passed"]))

        self.assertFalse(checks["pooled_no_actionability_fallback_rows"])

    def test_failed_tsm_calibration_route_uses_identity_scoring_fallback(self):
        selected_spec = TsmCalibrationRouteSpec(
            "TSM_STATIC_LOGIT_SHIFT",
            "p_success_tsm_route_tsm_static_logit_shift",
            "decision_score_tsm_route_tsm_static_logit_shift",
        )
        summary = pd.DataFrame(
            [
                {
                    "tsm_calibration_route": "TSM_STATIC_LOGIT_SHIFT",
                    "tsm_calibration_route_pass": False,
                }
            ]
        )

        route, spec, reason = choose_tsm_scoring_route("TSM_STATIC_LOGIT_SHIFT", selected_spec, summary)

        self.assertEqual(route, "POOLED_ONLY")
        self.assertEqual(spec.route, "POOLED_ONLY")
        self.assertEqual(reason, "SELECTED_ROUTE_FAILED_IDENTITY_FALLBACK")

    def test_quality_checks_score_tsm_fallback_as_effective_route_pass(self):
        tsm_metrics = pd.DataFrame(
            [
                {
                    "split": "tsm_combined_test_holdout",
                    "is_selected_tsm_calibration_route": True,
                    "event_count": 100,
                    "tsm_calibration_route_pass": False,
                    "tsm_calibration_scoring_route": "POOLED_ONLY",
                    "tsm_calibration_scoring_route_reason": "SELECTED_ROUTE_FAILED_IDENTITY_FALLBACK",
                    "tsm_calibration_route_failure_reasons": "TSM_COMBINED_ECE_GT_0_15",
                    "decision_ece": 0.33,
                }
            ]
        )

        quality = build_quality_checks(
            pd.DataFrame(),
            pd.DataFrame(),
            tsm_metrics,
            pd.DataFrame(),
            {},
            pooled_dataset_quality_ok=True,
        )
        by_check = quality.set_index("check")

        self.assertTrue(bool(by_check.loc["tsm_calibration_route_pass", "passed"]))
        self.assertTrue(bool(by_check.loc["tsm_calibrated_ece_at_most_0_15", "passed"]))
        self.assertIn("pooled-only fallback", str(by_check.loc["tsm_calibration_route_pass", "details"]))

    def test_tsm_effective_scoring_route_pass_accepts_identity_fallback(self):
        selected_route = pd.Series(
            {
                "tsm_calibration_route_pass": False,
                "tsm_calibration_scoring_route": "POOLED_ONLY",
                "tsm_calibration_scoring_route_reason": "SELECTED_ROUTE_FAILED_IDENTITY_FALLBACK",
            }
        )
        failed_route = pd.Series(
            {
                "tsm_calibration_route_pass": False,
                "tsm_calibration_scoring_route": "TSM_DIRECT_EMPIRICAL_PRIOR",
                "tsm_calibration_scoring_route_reason": "",
            }
        )

        self.assertTrue(tsm_effective_scoring_route_pass(selected_route))
        self.assertFalse(tsm_effective_scoring_route_pass(failed_route))

    def test_paper_gate_uses_adjacent_block_selection_evidence_for_infeasible_small_oof_fold(self):
        comparison = pd.DataFrame(
            [
                {
                    "model_name": "pooled_lgbm_classifier",
                    "split": "combined_test_holdout",
                    "evaluation_scope": "trade_ready_entry_only",
                    "event_count": 332,
                    "selected_event_count": 131,
                    "fold_selected_count_min": 11,
                    "fold_selected_fraction_min": 0.34375,
                    "fold_selected_fraction_max": 0.45,
                    "decision_ece": 0.05,
                    "brier_improvement_pct": 1.2,
                    "selected_mean_return_pct": 2.0,
                    "mean_return_pct": 1.0,
                    "selected_minus_all_ci_lower_pct_paired": 0.2,
                },
                {
                    "model_name": "pooled_lgbm_classifier",
                    "split": "oof_test_2021",
                    "evaluation_scope": "trade_ready_entry_only",
                    "event_count": 100,
                    "selected_event_count": 40,
                    "selected_fraction": 0.40,
                },
                {
                    "model_name": "pooled_lgbm_classifier",
                    "split": "oof_test_2022",
                    "evaluation_scope": "trade_ready_entry_only",
                    "event_count": 32,
                    "selected_event_count": 11,
                    "selected_fraction": 0.34375,
                },
                {
                    "model_name": "pooled_lgbm_classifier",
                    "split": "oof_test_2023",
                    "evaluation_scope": "trade_ready_entry_only",
                    "event_count": 200,
                    "selected_event_count": 80,
                    "selected_fraction": 0.40,
                },
            ]
        )
        tsm_like_metrics = pd.DataFrame(
            [
                {
                    "split": "tsm_like_train_validation",
                    "is_selected_tsm_like_route": True,
                    "tsm_like_route_selection_pass": True,
                    "selection_effective_n": 800,
                    "selection_decision_ece": 0.05,
                }
            ]
        )

        snapshot = build_paper_gate_snapshot(
            comparison,
            tsm_like_metrics,
            {"latest_trade_ready": True, "p_stop_hit_20d": 0.20, "decision_support_allowed": False},
            pooled_dataset_quality_ok=True,
        )
        values = dict(zip(snapshot["field"], snapshot["value"]))

        self.assertTrue(bool(values["paper_model_gate_pass"]))
        self.assertTrue(bool(values["paper_oof_selection_evidence_pass"]))
        self.assertEqual(values["paper_oof_selection_evidence_policy"], "ADJACENT_UNDERSIZED_OOF_BLOCK_MERGE")
        self.assertEqual(values["paper_model_block_reasons"], "PASS")
        self.assertEqual(values["paper_effective_min_selected_events_per_block"], 40.0)

    def test_paper_gate_keeps_raw_fold_failure_when_small_selection_was_feasible(self):
        comparison = pd.DataFrame(
            [
                {
                    "model_name": "pooled_lgbm_classifier",
                    "split": "combined_test_holdout",
                    "evaluation_scope": "trade_ready_entry_only",
                    "event_count": 300,
                    "selected_event_count": 131,
                    "fold_selected_count_min": 11,
                    "fold_selected_fraction_min": 0.11,
                    "fold_selected_fraction_max": 0.45,
                    "decision_ece": 0.05,
                    "brier_improvement_pct": 1.2,
                    "selected_mean_return_pct": 2.0,
                    "mean_return_pct": 1.0,
                    "selected_minus_all_ci_lower_pct_paired": 0.2,
                },
                {
                    "model_name": "pooled_lgbm_classifier",
                    "split": "oof_test_2021",
                    "evaluation_scope": "trade_ready_entry_only",
                    "event_count": 100,
                    "selected_event_count": 11,
                    "selected_fraction": 0.11,
                },
                {
                    "model_name": "pooled_lgbm_classifier",
                    "split": "oof_test_2022",
                    "evaluation_scope": "trade_ready_entry_only",
                    "event_count": 200,
                    "selected_event_count": 120,
                    "selected_fraction": 0.60,
                },
            ]
        )
        tsm_like_metrics = pd.DataFrame(
            [
                {
                    "split": "tsm_like_train_validation",
                    "is_selected_tsm_like_route": True,
                    "tsm_like_route_selection_pass": True,
                    "selection_effective_n": 800,
                    "selection_decision_ece": 0.05,
                }
            ]
        )

        snapshot = build_paper_gate_snapshot(
            comparison,
            tsm_like_metrics,
            {"latest_trade_ready": True, "p_stop_hit_20d": 0.20, "decision_support_allowed": False},
            pooled_dataset_quality_ok=True,
        )
        values = dict(zip(snapshot["field"], snapshot["value"]))

        self.assertFalse(bool(values["paper_model_gate_pass"]))
        self.assertFalse(bool(values["paper_oof_selection_evidence_pass"]))
        self.assertEqual(values["paper_oof_selection_evidence_policy"], "INSUFFICIENT_OOF_SELECTED_EVENTS")
        self.assertIn("POOLED_OOF_FOLD_SELECTED_EVENTS_LT_25", values["paper_model_block_reasons"])

    def test_refresh_paper_gate_overlay_replaces_stale_model_block_reasons(self):
        comparison = pd.DataFrame(
            [
                {
                    "model_name": "pooled_lgbm_classifier",
                    "split": "combined_test_holdout",
                    "evaluation_scope": "trade_ready_entry_only",
                    "validation_design": "walk_forward_oof",
                    "is_champion": True,
                    "event_count": 300,
                    "selected_event_count": 131,
                    "fold_selected_count_min": 11,
                    "fold_selected_fraction_min": 0.34375,
                    "fold_selected_fraction_max": 0.516,
                    "decision_ece": 0.05,
                    "brier_improvement_pct": 1.2,
                    "selected_mean_return_pct": 2.0,
                    "mean_return_pct": 1.0,
                    "selected_minus_all_ci_lower_pct_paired": 0.2,
                },
                {
                    "model_name": "pooled_lgbm_classifier",
                    "split": "oof_test_2021",
                    "evaluation_scope": "trade_ready_entry_only",
                    "validation_design": "walk_forward_oof",
                    "event_count": 32,
                    "selected_event_count": 11,
                    "selected_fraction": 0.34375,
                },
                {
                    "model_name": "pooled_lgbm_classifier",
                    "split": "oof_test_2022",
                    "evaluation_scope": "trade_ready_entry_only",
                    "validation_design": "walk_forward_oof",
                    "event_count": 180,
                    "selected_event_count": 60,
                    "selected_fraction": 0.33,
                },
            ]
        )
        tsm_like_metrics = pd.DataFrame(
            [
                {
                    "split": "tsm_like_train_validation",
                    "is_selected_tsm_like_route": True,
                    "tsm_like_route_selection_pass": True,
                    "selection_effective_n": 800,
                    "selection_decision_ece": 0.05,
                }
            ]
        )
        overlay = {
            "latest_trade_ready": False,
            "p_stop_hit_20d": 0.45,
            "decision_support_allowed": False,
            "paper_gate_block_reasons": "LATEST_NOT_TRADE_READY|POOLED_OOF_FOLD_SELECTED_EVENTS_LT_25|POOLED_STOP_RISK_GT_0_40",
        }
        pooled_quality = pd.DataFrame([{"check": "pooled_dataset_quality_critical_pass", "passed": True, "severity": "CRITICAL"}])

        refreshed, paper_snapshot = refresh_paper_gate_overlay(comparison, tsm_like_metrics, overlay, pooled_quality)
        paper_values = dict(zip(paper_snapshot["field"], paper_snapshot["value"]))

        self.assertTrue(bool(refreshed["paper_model_gate_pass"]))
        self.assertEqual(refreshed["paper_model_block_reasons"], "PASS")
        self.assertEqual(paper_values["paper_model_block_reasons"], "PASS")
        self.assertNotIn("POOLED_OOF_FOLD_SELECTED_EVENTS_LT_25", refreshed["paper_gate_block_reasons"])
        self.assertEqual(refreshed["paper_gate_block_reasons"], "LATEST_NOT_TRADE_READY|POOLED_STOP_RISK_GT_0_40")
        self.assertEqual(refreshed["paper_oof_selection_evidence_policy"], "ADJACENT_UNDERSIZED_OOF_BLOCK_MERGE")

    def test_apply_paper_gate_snapshot_to_latest_predictions_replaces_stale_paper_model_reason(self):
        latest = pd.DataFrame(
            [
                {
                    "symbol": "TSM",
                    "latest_trade_ready": False,
                    "p_stop_hit_20d": 0.56,
                    "decision_support_allowed": False,
                    "paper_decision_support_allowed": False,
                    "block_reasons": (
                        "LATEST_NOT_TRADE_READY|PAPER_MODEL:POOLED_OOF_FOLD_SELECTED_EVENTS_LT_25|"
                        "POOLED_DECISION_SCORE_BELOW_THRESHOLD|POOLED_EXPECTED_R_LT_0_35|"
                        "POOLED_STOP_RISK_GT_0_35|POOLED_STOP_RISK_GT_0_40"
                    ),
                }
            ]
        )
        paper_snapshot = pd.DataFrame(
            [
                {"field": "paper_model_gate_pass", "value": True},
                {"field": "paper_model_block_reasons", "value": "PASS"},
            ]
        )

        refreshed = apply_paper_gate_snapshot_to_latest_predictions(latest, paper_snapshot)

        self.assertFalse(bool(refreshed.loc[0, "paper_decision_support_allowed"]))
        self.assertNotIn("PAPER_MODEL:POOLED_OOF_FOLD_SELECTED_EVENTS_LT_25", refreshed.loc[0, "block_reasons"])
        self.assertNotIn("POOLED_OOF_FOLD_SELECTED_EVENTS_LT_25", refreshed.loc[0, "block_reasons"])
        self.assertIn("POOLED_STOP_RISK_GT_0_40", refreshed.loc[0, "block_reasons"])

    def test_apply_paper_gate_snapshot_to_quality_checks_replaces_stale_paper_reason(self):
        quality = pd.DataFrame(
            [
                {
                    "check": "paper_only_gate_pass",
                    "passed": False,
                    "severity": "INFO",
                    "value": "LATEST_NOT_TRADE_READY|POOLED_OOF_FOLD_SELECTED_EVENTS_LT_25|POOLED_STOP_RISK_GT_0_40",
                    "details": "Paper gate does not enable strict prediction_ready or live trading.",
                }
            ]
        )
        paper_snapshot = pd.DataFrame(
            [
                {"field": "paper_decision_support_allowed", "value": False},
                {"field": "paper_gate_block_reasons", "value": "LATEST_NOT_TRADE_READY|POOLED_STOP_RISK_GT_0_40"},
            ]
        )

        refreshed = apply_paper_gate_snapshot_to_quality_checks(quality, paper_snapshot)

        self.assertEqual(refreshed.loc[0, "value"], "LATEST_NOT_TRADE_READY|POOLED_STOP_RISK_GT_0_40")
        self.assertNotIn("POOLED_OOF_FOLD_SELECTED_EVENTS_LT_25", refreshed.loc[0, "value"])

    def test_tsm_direct_empirical_prior_route_applies_train_validation_hit_rate(self):
        rows = pd.DataFrame({"label_success_20d": [1] * 24 + [0] * 12})

        probability, method = fit_direct_empirical_tsm_probability(rows)
        spec = TsmCalibrationRouteSpec(
            "TSM_DIRECT_EMPIRICAL_PRIOR",
            "p_success_tsm_route_direct_empirical_prior",
            "decision_score_tsm_route_direct_empirical_prior",
            route_sample_weight_policy=method,
            constant_probability=probability,
        )
        applied = apply_tsm_calibration_route_spec(
            pd.DataFrame({"p_success_calibrated": [0.2, 0.8]}),
            spec,
            source_col="p_success_calibrated",
        )

        self.assertEqual(method, "direct_empirical_train_validation")
        self.assertAlmostEqual(probability, 2.0 / 3.0)
        self.assertTrue(np.allclose(applied.to_numpy(dtype=float), [2.0 / 3.0, 2.0 / 3.0]))

    def test_tsm_calibration_route_v2_can_select_direct_empirical_prior(self):
        metrics = pd.DataFrame(
            [
                {"tsm_calibration_route": "POOLED_ONLY", "split": "tsm_train_validation", "decision_ece": 0.05, "brier_improvement_pct": 1.0},
                {"tsm_calibration_route": "POOLED_ONLY", "split": "tsm_combined_test_holdout", "decision_ece": 0.16, "brier_improvement_pct": -1.0},
                {"tsm_calibration_route": "POOLED_ONLY", "split": "tsm_test_2024", "decision_ece": 0.21, "brier_improvement_pct": -1.0},
                {"tsm_calibration_route": "POOLED_ONLY", "split": "tsm_final_holdout_2025_2026", "decision_ece": 0.16, "brier_improvement_pct": -1.0},
                {"tsm_calibration_route": "TSM_DIRECT_EMPIRICAL_PRIOR", "split": "tsm_train_validation", "decision_ece": 0.0, "brier_improvement_pct": 18.0},
                {"tsm_calibration_route": "TSM_DIRECT_EMPIRICAL_PRIOR", "split": "tsm_combined_test_holdout", "decision_ece": 0.06, "brier_improvement_pct": 20.0},
                {"tsm_calibration_route": "TSM_DIRECT_EMPIRICAL_PRIOR", "split": "tsm_test_2024", "decision_ece": 0.15, "brier_improvement_pct": 30.0},
                {"tsm_calibration_route": "TSM_DIRECT_EMPIRICAL_PRIOR", "split": "tsm_final_holdout_2025_2026", "decision_ece": 0.04, "brier_improvement_pct": 10.0},
            ]
        )

        route, summary = choose_tsm_calibration_route_v2(metrics)

        self.assertEqual(route, "TSM_DIRECT_EMPIRICAL_PRIOR")
        selected = summary[summary["is_selected_tsm_calibration_route"].map(bool)].iloc[0]
        self.assertTrue(bool(selected["tsm_calibration_route_pass"]))

    def test_tsm_like_semiconductor_recognizes_foundry_idm_groups(self):
        self.assertTrue(is_tsm_like_semiconductor("TSM", "foundry_idm"))
        self.assertTrue(is_tsm_like_semiconductor("005930.KS", "memory_foundry_idm"))

    def test_empirical_bayes_prediction_preserves_nonzero_indices(self):
        train = pd.DataFrame(
            {
                "entry_trigger": ["A", "B", "A", "B"],
                "trend_regime": ["UP"] * 4,
                "vol_regime": ["NORMAL"] * 4,
                "drawdown_bucket": ["dd_0_5"] * 4,
                "label_success_20d": [1, 0, 1, 0],
                "label_stop_survival_20d": [1, 0, 1, 0],
                "label_hit_1r_before_stop_20d": [1, 0, 1, 0],
                "label_hit_2r_before_stop_20d": [0, 0, 1, 0],
                "label_expected_r_20d": [1.0, -1.0, 2.0, -1.0],
                "label_net_return_pct_20d": [4.0, -2.0, 8.0, -2.0],
            }
        )
        model = fit_empirical_bayes(train, prior_strength=2.0)
        rows = train.iloc[[2, 3]].copy()
        rows.index = [100, 101]
        pred = predict_empirical_bayes(model, rows)
        self.assertFalse(pred["p_success_base"].isna().any())
        self.assertEqual(list(pred.index), [0, 1])

    def test_tsm_layer_moves_probability_toward_tsm_posterior(self):
        train = pd.DataFrame(
            {
                "label_success_20d": [1] * 32 + [0] * 8,
                "p_success_base": [0.40] * 40,
            }
        )
        layer = fit_tsm_calibration_layer(train, "p_success_base", global_success=0.50)
        out = apply_tsm_layer(pd.DataFrame({"p_success_base": [0.40]}), layer).iloc[0]
        self.assertGreater(out, 0.40)

    def test_tsm_layer_does_not_shift_small_samples(self):
        train = pd.DataFrame(
            {
                "label_success_20d": [1] * 8 + [0] * 2,
                "p_success_base": [0.40] * 10,
            }
        )
        layer = fit_tsm_calibration_layer(train, "p_success_base", global_success=0.50)
        out = apply_tsm_layer(pd.DataFrame({"p_success_base": [0.40]}), layer).iloc[0]
        self.assertEqual(layer.status, "TSM_CALIBRATION_LAYER_INSUFFICIENT_SAMPLE")
        self.assertAlmostEqual(out, 0.40)

    def test_threshold_v2_rejects_selecting_almost_everything(self):
        predictions = pd.DataFrame(
            {
                "decision_score": [0.1, 0.2, 0.3, 0.4, 0.5],
                "label_net_return_pct_20d": [0.1, 0.2, 0.3, 1.5, 2.0],
                "label_stop_hit_20d": [0.6, 0.6, 0.5, 0.2, 0.1],
                "label_success_20d": [0, 0, 1, 1, 1],
            }
        )
        result = choose_threshold(predictions, thresholds=[0.0], min_selected=2)
        table = result["threshold_table"]
        if pd.notna(result["threshold"]):
            chosen = table[table["threshold"].eq(result["threshold"])].iloc[0]
            self.assertLessEqual(chosen["selected_fraction"], 0.60)

    def test_isotonic_not_used_when_calibration_sample_below_1000(self):
        p = pd.Series([0.1 + i * 0.004 for i in range(200)])
        y = pd.Series(([0, 1] * 100))
        _, method = fit_probability_calibrator(p, y)
        self.assertNotIn("isotonic", method)

    def test_pooled_feature_columns_blocks_label_and_future_leakage(self):
        data = pd.DataFrame(
            {
                "symbol": ["TSM", "NVDA"],
                "date": pd.date_range("2024-01-01", periods=2),
                "score_price_algo_total": [70.0, 80.0],
                "entry_trigger": ["A", "B"],
                "label_success_20d": [1, 0],
                "label_net_return_pct_20d": [2.0, -1.0],
                "is_model_training_candidate": [True, True],
                "is_decision_entry_candidate": [True, False],
                "future_return_20d": [3.0, -2.0],
                "exit_reason": ["TARGET", "STOP"],
                "fwd_20d": [1.0, 0.0],
            }
        )
        cols = pooled_feature_columns(data)
        self.assertIn("score_price_algo_total", cols)
        self.assertIn("entry_trigger", cols)
        self.assertNotIn("label_success_20d", cols)
        self.assertNotIn("label_net_return_pct_20d", cols)
        self.assertNotIn("is_model_training_candidate", cols)
        self.assertNotIn("is_decision_entry_candidate", cols)
        self.assertNotIn("future_return_20d", cols)
        self.assertNotIn("exit_reason", cols)
        self.assertNotIn("fwd_20d", cols)

    def test_pooled_feature_columns_keeps_sparse_intraday_overlay_features(self):
        data = pd.DataFrame(
            {
                "symbol": [f"S{i}" for i in range(100)],
                "date": pd.date_range("2024-01-01", periods=100),
                "score_price_algo_total": np.linspace(50, 80, 100),
                "m5_realized_vol_20bar_ann": [np.nan] * 80 + list(np.linspace(0.1, 0.3, 20)),
                "m1_realized_range_pct": [np.nan] * 80 + list(np.linspace(0.001, 0.003, 20)),
                "intraday_coverage_class": [np.nan] * 80 + ["PARTIAL"] * 20,
                "label_success_20d": [0, 1] * 50,
            }
        )

        cols = pooled_feature_columns(data)

        self.assertIn("m5_realized_vol_20bar_ann", cols)
        self.assertIn("m1_realized_range_pct", cols)
        self.assertIn("intraday_coverage_class", cols)

    def test_prepare_dataset_uses_model_training_candidates_and_preserves_decision_mask(self):
        features = pd.DataFrame(
            {
                "symbol": ["TSM", "TSM", "NVDA"],
                "symbol_group": ["foundry", "foundry", "gpu"],
                "date": pd.date_range("2024-01-01", periods=3),
                "signal_idx": [1, 2, 3],
                "is_trade_ready_entry_candidate": [True, False, False],
                "is_model_training_candidate": [True, True, False],
                "is_decision_entry_candidate": [True, False, False],
                "label_status_20d": ["LABELED", "LABELED", "LABELED"],
                "label_success_20d": [1, 0, 1],
                "label_net_return_pct_20d": [2.0, -1.0, 3.0],
                "label_expected_r_20d": [0.8, -0.4, 1.0],
                "label_stop_survival_20d": [1, 0, 1],
                "label_hit_1r_before_stop_20d": [1, 0, 1],
                "label_hit_2r_before_stop_20d": [0, 0, 1],
                "entry_trigger": ["A", "NONE", "B"],
                "trend_regime": ["UP", "UP", "UP"],
                "vol_regime": ["NORMAL", "NORMAL", "NORMAL"],
                "drawdown_bucket": ["dd_0_5", "dd_0_5", "dd_0_5"],
            }
        )
        data = prepare_dataset(features)
        self.assertEqual(len(data), 2)
        self.assertEqual(int(data["is_decision_entry_candidate"].sum()), 1)
        self.assertTrue(data["is_model_training_candidate"].all())

    def test_prepare_dataset_uses_active_horizon_labels(self):
        features = pd.DataFrame(
            {
                "symbol": ["TSM", "NVDA"],
                "symbol_group": ["foundry", "gpu"],
                "date": pd.date_range("2024-01-01", periods=2),
                "signal_idx": [1, 2],
                "is_trade_ready_entry_candidate": [True, True],
                "is_model_training_candidate": [True, True],
                "is_decision_entry_candidate": [True, True],
                "label_status_5d": ["LABELED", "LABELED"],
                "label_success_5d": [1, 0],
                "label_net_return_pct_5d": [1.0, -0.5],
                "label_expected_r_5d": [0.5, -0.2],
                "label_stop_survival_5d": [1, 0],
                "label_hit_1r_before_stop_5d": [1, 0],
                "label_hit_2r_before_stop_5d": [0, 0],
                "label_status_60d": ["LABELED", "LABELED"],
                "label_success_60d": [0, 1],
                "label_net_return_pct_60d": [-2.0, 4.0],
                "label_expected_r_60d": [-0.8, 1.4],
                "label_stop_survival_60d": [0, 1],
                "label_hit_1r_before_stop_60d": [0, 1],
                "label_hit_2r_before_stop_60d": [0, 1],
                "entry_trigger": ["A", "B"],
                "trend_regime": ["UP", "UP"],
                "vol_regime": ["NORMAL", "NORMAL"],
                "drawdown_bucket": ["dd_0_5", "dd_0_5"],
            }
        )
        original_horizon = pooled_engine.HORIZON
        try:
            pooled_engine.set_active_horizon(5)
            data_5d = prepare_dataset(features)
            self.assertEqual(pooled_engine.TARGET_COL, "label_success_5d")
            self.assertEqual(data_5d["label_success_5d"].tolist(), [1, 0])
            self.assertEqual(data_5d["label_stop_survival_5d"].tolist(), [1, 0])

            pooled_engine.set_active_horizon(60)
            data_60d = prepare_dataset(features)
            self.assertEqual(pooled_engine.TARGET_COL, "label_success_60d")
            self.assertEqual(data_60d["label_success_60d"].tolist(), [0, 1])
            self.assertEqual(data_60d["label_stop_survival_60d"].tolist(), [0, 1])
        finally:
            pooled_engine.set_active_horizon(original_horizon)

    def test_candidate_training_weights_prioritizes_strict_decision_tier(self):
        frame = pd.DataFrame(
            {
                "candidate_tier": [
                    "decision_trade_ready",
                    "relaxed_trigger_score65",
                    "relaxed_trigger_score60",
                    "setup_context_score65",
                ],
                "is_decision_entry_candidate": [True, False, False, False],
            }
        )
        weights = candidate_training_weights(frame)
        self.assertGreater(weights.iloc[0], weights.iloc[1])
        self.assertGreater(weights.iloc[1], weights.iloc[2])
        self.assertGreater(weights.iloc[3], weights.iloc[2])

    def test_risk_adjusted_selection_score_can_use_expected_return_rank(self):
        frame = pd.DataFrame(
            {
                "p_model": [0.50, 0.50],
                "p_stop_hit_lgbm": [0.30, 0.30],
                "expected_r_lgbm": [0.10, 0.10],
                "expected_return_lgbm": [-1.0, 4.0],
                "score_price_algo_total": [70.0, 70.0],
            }
        )

        scored = add_risk_adjusted_selection_score(
            frame,
            "p_model",
            {"w_success": 0.0, "w_stop": 0.0, "w_r": 0.0, "w_return": 1.0, "w_rule": 0.0},
        )

        self.assertLess(scored.loc[0, "risk_adjusted_selection_score"], scored.loc[1, "risk_adjusted_selection_score"])
        self.assertLess(scored.loc[0, "expected_return_rank"], scored.loc[1, "expected_return_rank"])

    def test_block_reasons_do_not_label_fraction_drift_as_selected_count_failure(self):
        model_metrics = pd.DataFrame(
            [
                {
                    "model_name": "pooled_stack_calibrated",
                    "split": "combined_test_holdout",
                    "evaluation_scope": "trade_ready_entry_only",
                    "validation_design": "walk_forward_oof",
                    "event_count": 200,
                    "selected_event_count": 80,
                    "selected_fraction": 0.40,
                    "selected_expectancy_ci_lower_pct": 0.20,
                    "selected_minus_all_pct": 0.50,
                    "selected_minus_all_ci_lower_pct_paired": 0.10,
                    "selected_minus_score_baseline_ci_lower_pct_paired": 0.10,
                    "uplift_pass": True,
                    "decision_ece": 0.05,
                    "decision_min_calibration_bin_n": 40,
                    "brier_improvement_pct": 1.0,
                    "positive_expectancy_fold_count": 4,
                    "threshold_decision_eligible": False,
                    "threshold_stability_pass": False,
                    "weak_oof_folds": "oof_test_2024:selected_fraction_out_of_range|selection_fraction_drift_gt_0_10",
                    "trial_count": 10,
                },
                {
                    "model_name": "pooled_stack_calibrated",
                    "split": "oof_test_2024",
                    "evaluation_scope": "trade_ready_entry_only",
                    "validation_design": "walk_forward_oof",
                    "event_count": 50,
                    "selected_event_count": 20,
                },
            ]
        )
        tsm_metrics = pd.DataFrame(
            [
                {
                    "split": "tsm_combined_test_holdout",
                    "event_count": 40,
                    "tsm_calibration_route_pass": True,
                    "decision_ece": 0.05,
                    "brier_improvement_pct": 1.0,
                    "max_test_holdout_ece": 0.05,
                    "route_selection_provenance_valid": True,
                }
            ]
        )

        reasons = block_reasons(model_metrics, tsm_metrics, quality_ok=True)

        self.assertIn("POOLED_THRESHOLD_STABILITY_FAILED", reasons)
        self.assertNotIn("POOLED_OOF_FOLD_SELECTED_EVENTS_LT_10", reasons)

    def test_scope_logit_shift_moves_strict_probability_toward_observed_rate(self):
        rows = pd.DataFrame(
            {
                "label_success_20d": [1] * 45 + [0] * 15,
                "p_success_calibrated": [0.45] * 60,
            }
        )
        layer = fit_scope_logit_shift(rows, "p_success_calibrated", global_success=0.50, scope_name="trade_ready_entry_only", prior_strength=40.0)
        shifted = apply_logit_shift(pd.Series([0.45]), layer).iloc[0]
        self.assertEqual(layer["status"], "SCOPE_CALIBRATION_READY")
        self.assertGreater(shifted, 0.45)

    def test_pooled_metric_row_keeps_probability_and_selection_score_separate(self):
        predictions = pd.DataFrame(
            {
                "symbol": ["A", "B", "C", "D"],
                "label_success_20d": [1, 0, 1, 0],
                "label_net_return_pct_20d": [2.0, -1.0, 3.0, -2.0],
                "label_stop_survival_20d": [1, 0, 1, 0],
                "p_success_calibrated": [0.55, 0.45, 0.60, 0.40],
                "decision_score": [0.7, 0.2, 0.8, 0.1],
            }
        )
        row = metric_row(
            "combined_test_holdout",
            predictions,
            "p_success_calibrated",
            threshold=0.5,
            base_rate=0.5,
            model_name="pooled_stack_calibrated",
            score_col="decision_score",
        )
        self.assertEqual(row["probability_col"], "p_success_calibrated")
        self.assertEqual(row["selection_score_col"], "decision_score")
        self.assertEqual(row["selected_event_count"], 2)
        self.assertEqual(row["model_family"], "stacked_calibrated")

    def test_adaptive_calibration_is_primary_even_when_fixed_width_is_sparse(self):
        n = 120
        probabilities = np.linspace(0.01, 0.99, n)
        predictions = pd.DataFrame(
            {
                "symbol": [f"S{i % 12}" for i in range(n)],
                "score_price_algo_total": [80 if i >= 60 else 60 for i in range(n)],
                "label_success_20d": [1 if p >= 0.50 else 0 for p in probabilities],
                "label_net_return_pct_20d": [2.0 if p >= 0.50 else -1.0 for p in probabilities],
                "label_stop_survival_20d": [1 if p >= 0.50 else 0 for p in probabilities],
                "p_success_calibrated": probabilities,
                "decision_score": probabilities,
            }
        )
        row = metric_row(
            "combined_test_holdout",
            predictions,
            "p_success_calibrated",
            threshold=0.50,
            base_rate=0.50,
            model_name="pooled_stack_calibrated",
            score_col="decision_score",
        )

        self.assertEqual(row["calibration_binning_primary"], "adaptive_equal_frequency")
        self.assertGreaterEqual(row["decision_min_calibration_bin_n"], 30)
        self.assertLess(row["fixed_width_min_calibration_bin_n"], 30)

    def test_oof_fold_sparse_selection_keeps_threshold_diagnostic(self):
        rows = []
        for split, selected_n in [("oof_test_2024", 4), ("oof_test_2025_2026", 20)]:
            for i in range(40):
                selected = i < selected_n
                rows.append(
                    {
                        "model_name": "pooled_empirical_bayes_group_rate",
                        "evaluation_scope": "trade_ready_entry_only",
                        "split": split,
                        "symbol": f"S{i % 12}",
                        "label_success_20d": int(selected),
                        "label_net_return_pct_20d": 2.0 if selected else -1.0,
                        "label_stop_survival_20d": 1 if selected else 0,
                        "p_success": 0.70 if selected else 0.30,
                        "utility_score": 0.70 if selected else 0.30,
                        "threshold": 0.50,
                        "selected_by_threshold": selected,
                        "threshold_reason": "DIAGNOSTIC_BROAD_SCOPE_FALLBACK",
                        "threshold_decision_eligible": True,
                        "base_rate": 0.50,
                        "trial_count": 32,
                        "score_price_algo_total": 80 if selected else 60,
                    }
                )
        metrics = build_oof_metrics(pd.DataFrame(rows))
        combined = metrics[metrics["split"].eq("combined_test_holdout")].iloc[0]

        self.assertFalse(bool(combined["threshold_decision_eligible"]))
        self.assertFalse(bool(combined["threshold_stability_pass"]))
        self.assertIn("oof_test_2024", combined["weak_oof_folds"])

    def test_oof_fold_borderline_selection_fraction_remains_stable(self):
        metrics = pd.DataFrame(
            [
                {
                    "split": "oof_test_2021",
                    "selected_event_count": 213,
                    "selected_fraction": 0.3586,
                    "threshold": 0.1416,
                    "applied_threshold_policy_type": "expanding_rank_percentile_policy",
                },
                {
                    "split": "oof_test_2022",
                    "selected_event_count": 13,
                    "selected_fraction": 0.4062,
                    "threshold": 0.1416,
                    "applied_threshold_policy_type": "expanding_rank_percentile_policy",
                },
                {
                    "split": "oof_test_2023",
                    "selected_event_count": 98,
                    "selected_fraction": 0.2865,
                    "threshold": 0.1416,
                    "applied_threshold_policy_type": "expanding_rank_percentile_policy",
                },
                {
                    "split": "oof_test_2024",
                    "selected_event_count": 210,
                    "selected_fraction": 0.4008,
                    "threshold": 0.1416,
                    "applied_threshold_policy_type": "expanding_rank_percentile_policy",
                },
            ]
        )

        stable, weak_folds, _ = threshold_stability_from_fold_metrics(metrics)

        self.assertTrue(stable)
        self.assertEqual(weak_folds, "")

    def test_threshold_v3_uses_broad_fallback_as_diagnostic_until_oof_passes(self):
        strict = pd.DataFrame(columns=["decision_score", "label_net_return_pct_20d", "label_stop_hit_20d", "label_success_20d", "p_success"])
        broad = pd.DataFrame(
            {
                "decision_score": [0.4, 0.6],
                "label_net_return_pct_20d": [1.0, 2.0],
                "label_stop_hit_20d": [0.2, 0.1],
                "label_success_20d": [1, 1],
                "p_success": [0.6, 0.7],
            }
        )
        broad_info = {"threshold": 0.55, "threshold_reason": "VALIDATION_ECONOMIC_UPLIFT_AND_RISK_FILTER", "trial_count": 12}
        out = choose_stable_trade_ready_threshold_v3(
            strict,
            broad,
            broad_info,
            score_col="decision_score",
            stop_col="label_stop_hit_20d",
            probability_col="p_success",
            base_rate=0.50,
        )

        self.assertFalse(out["threshold_decision_eligible"])
        self.assertIn("DIAGNOSTIC_BROAD_SCOPE_FALLBACK", out["threshold_reason"])

    def test_fold_consensus_percentile_policy_can_make_threshold_decision_eligible(self):
        rows = []
        for fold_idx, split in enumerate(["oof_test_2021", "oof_test_2023", "oof_test_2024", "oof_test_2025_2026"]):
            score_base = fold_idx * 10.0
            for i in range(40):
                high = i >= 24
                rows.append(
                    {
                        "model_name": "pooled_stack_calibrated",
                        "evaluation_scope": "trade_ready_entry_only",
                        "validation_design": "walk_forward_oof",
                        "split": split,
                        "symbol": f"S{i % 12}",
                        "label_success_20d": int(high),
                        "label_net_return_pct_20d": 8.0 if high else -4.0,
                        "label_stop_survival_20d": 1 if high else 0,
                        "p_success": 0.85 if high else 0.15,
                        "utility_score": score_base + i / 100.0,
                        "threshold": score_base + 0.20,
                        "selected_by_threshold": i >= 20,
                        "threshold_reason": "DIAGNOSTIC_BROAD_SCOPE_FALLBACK",
                        "threshold_decision_eligible": False,
                        "base_rate": 0.50,
                        "trial_count": 12,
                        "score_price_algo_total": 80 if high else 60,
                    }
                )

        result = choose_fold_consensus_trade_ready_threshold_v4(pd.DataFrame(rows))
        chosen = result["chosen"]
        records = result["records"]

        self.assertEqual(chosen["threshold_policy_type"], "rank_percentile_policy")
        self.assertGreater(result["stable_threshold_candidate_count"], 0)
        self.assertTrue(records["threshold_decision_eligible"].map(bool).all())
        self.assertTrue(records["selected_by_threshold"].groupby(records["split"]).sum().ge(10).all())

    def test_threshold_v5_locks_policy_from_threshold_window_not_test_labels(self):
        threshold_rows = []
        test_rows = []
        for i in range(40):
            high_score = i >= 24
            common = {
                "symbol": f"S{i % 12}",
                "symbol_group": "semi",
                "date": pd.Timestamp("2023-08-01"),
                "signal_idx": i,
                "score_price_algo_total": 60 + i,
                "p_success_model": 0.2 + i / 100.0,
                "p_stop_hit_lgbm": 0.6 - i / 100.0,
                "expected_r_lgbm": -0.5 + i / 20.0,
                "label_stop_survival_20d": 1 if high_score else 0,
                "label_net_return_pct_20d": 6.0 if high_score else -3.0,
                "label_success_20d": int(high_score),
            }
            threshold_rows.append(common)
            test_common = dict(common)
            test_common["date"] = pd.Timestamp("2024-08-01")
            test_common["label_success_20d"] = int(not high_score)
            test_common["label_net_return_pct_20d"] = -5.0 if high_score else 7.0
            test_common["label_stop_survival_20d"] = 0 if high_score else 1
            test_rows.append(test_common)

        result = choose_fold_consensus_trade_ready_threshold_v5(
            pd.DataFrame(threshold_rows),
            pd.DataFrame(test_rows),
            model_name="pooled_stack_calibrated",
            p_col="p_success_model",
            raw_score_col="p_success_model",
            base_rate=0.50,
            fold_id="wf_2024",
            split_name="oof_test_2024",
            threshold_info={"trial_count": 5, "threshold": 0.5},
        )
        records = result["records"]
        selected = records[records["selected_by_threshold"].map(bool)]
        unselected = records[~records["selected_by_threshold"].map(bool)]

        self.assertEqual(records["threshold_policy_source_window"].iloc[0], "threshold")
        self.assertEqual(records["threshold_policy_applied_window"].iloc[0], "test_fold_score_distribution_no_labels")
        self.assertTrue(records["threshold_policy_provenance_valid"].map(bool).all())
        self.assertEqual(records["applied_threshold_policy_type"].iloc[0], "fold_rank_percentile_policy")
        self.assertFalse(selected.empty)
        self.assertGreaterEqual(len(selected), 25)
        self.assertGreaterEqual(
            float(pd.to_numeric(selected["p_success_model"], errors="coerce").min()),
            float(pd.to_numeric(unselected["p_success_model"], errors="coerce").max()),
        )
        policy = result["threshold_table"]
        self.assertTrue(policy["model_name"].eq("pooled_stack_calibrated").all())
        self.assertTrue(policy["evaluation_scope"].eq("trade_ready_entry_only").all())
        self.assertTrue(policy["validation_design"].eq("walk_forward_oof").all())

    def test_threshold_v5_uses_label_free_fold_rank_when_test_scores_shift(self):
        threshold_rows = []
        test_rows = []
        for i in range(40):
            high_score = i >= 24
            threshold_rows.append(
                {
                    "symbol": f"S{i % 12}",
                    "symbol_group": "semi",
                    "date": pd.Timestamp("2023-08-01") + pd.Timedelta(days=i),
                    "signal_idx": i,
                    "score_price_algo_total": 60 + i,
                    "p_success_model": 0.2 + i / 100.0,
                    "p_stop_hit_lgbm": 0.6 - i / 100.0,
                    "expected_r_lgbm": -0.5 + i / 20.0,
                    "label_stop_survival_20d": 1 if high_score else 0,
                    "label_net_return_pct_20d": 6.0 if high_score else -3.0,
                    "label_success_20d": int(high_score),
                }
            )
            test_rows.append(
                {
                    "symbol": f"S{i % 12}",
                    "symbol_group": "semi",
                    "date": pd.Timestamp("2024-08-01") + pd.Timedelta(days=i),
                    "signal_idx": i,
                    "score_price_algo_total": 30 + i / 10,
                    "p_success_model": 0.05 + i / 1000.0,
                    "p_stop_hit_lgbm": 0.6,
                    "expected_r_lgbm": -0.5,
                    "label_stop_survival_20d": 0,
                    "label_net_return_pct_20d": -5.0,
                    "label_success_20d": 0,
                }
            )

        result = choose_fold_consensus_trade_ready_threshold_v5(
            pd.DataFrame(threshold_rows),
            pd.DataFrame(test_rows),
            model_name="pooled_stack_calibrated",
            p_col="p_success_model",
            raw_score_col="p_success_model",
            base_rate=0.50,
            fold_id="wf_2024",
            split_name="oof_test_2024",
            threshold_info={"trial_count": 5, "threshold": 0.5},
        )

        records = result["records"]

        self.assertEqual(records["applied_threshold_policy_type"].iloc[0], "fold_rank_percentile_policy")
        self.assertEqual(records["threshold_policy_applied_window"].iloc[0], "test_fold_score_distribution_no_labels")
        self.assertGreater(int(records["selected_by_threshold"].sum()), 0)
        self.assertEqual(int(records.loc[records["selected_by_threshold"].map(bool), "label_success_20d"].sum()), 0)

    def test_expanding_rank_selection_uses_only_current_and_prior_score_batches(self):
        frame = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-01")] * 3 + [pd.Timestamp("2024-01-02")] * 3,
                "score": [0.1, 0.2, 0.3, 0.4, 0.5, 10.0],
            }
        )

        selected, thresholds = select_expanding_top_fraction_by_score(frame, "score", 1.0 / 3.0)

        self.assertEqual(selected.iloc[:3].tolist(), [False, False, True])
        self.assertEqual(selected.iloc[3:].tolist(), [False, True, True])
        self.assertAlmostEqual(float(thresholds.iloc[0]), 0.23333333333333334)
        self.assertGreater(float(thresholds.iloc[3]), float(thresholds.iloc[0]))

    def test_rank_percentile_selection_uses_exact_top_k_when_scores_tie(self):
        frame = pd.DataFrame({"score": [1.0] * 10})

        selected, threshold = select_top_fraction_by_score(frame, "score", 0.40)

        self.assertEqual(int(selected.sum()), 4)
        self.assertEqual(threshold, 1.0)

    def test_paired_bootstrap_uses_date_block_when_dates_available(self):
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=60, freq="7D"),
                "label_net_return_pct_20d": [2.0] * 30 + [-1.0] * 30,
                "score_price_algo_total": [80] * 20 + [70] * 40,
            }
        )
        selected_mask = pd.Series([True] * 30 + [False] * 30, index=frame.index)
        baseline_mask, policy = score_baseline_mask_with_policy(frame)
        result = paired_bootstrap_uplift(frame, selected_mask, baseline_mask, iterations=40)

        self.assertEqual(policy, "score_price_algo_total_top_40pct")
        self.assertEqual(result["bootstrap_method"], "date_block")
        self.assertEqual(result["bootstrap_block_col"], "year_month")
        self.assertGreater(result["selected_minus_all_ci_lower_pct_paired"], 0)

    def test_score_baseline_falls_back_to_top_40pct_when_score_75_sample_is_sparse(self):
        frame = pd.DataFrame(
            {
                "score_price_algo_total": [80] * 10 + [70] * 15 + [50] * 25,
                "label_net_return_pct_20d": list(range(50)),
            }
        )
        baseline, policy = score_baseline_returns_with_policy(frame)

        self.assertEqual(policy, "score_price_algo_total_top_40pct")
        self.assertGreaterEqual(len(baseline), 20)

    def test_tsm_calibration_route_prefers_pooled_only_when_logit_shift_fails(self):
        metrics = pd.DataFrame(
            [
                {"tsm_calibration_route": "POOLED_ONLY", "split": "tsm_combined_test_holdout", "decision_ece": 0.10, "brier_improvement_pct": 1.0},
                {"tsm_calibration_route": "POOLED_ONLY", "split": "tsm_test_2024", "decision_ece": 0.11, "brier_improvement_pct": 1.0},
                {"tsm_calibration_route": "POOLED_ONLY", "split": "tsm_final_holdout_2025_2026", "decision_ece": 0.12, "brier_improvement_pct": 1.0},
                {"tsm_calibration_route": "TSM_LOGIT_SHIFT", "split": "tsm_combined_test_holdout", "decision_ece": 0.16, "brier_improvement_pct": 2.0},
                {"tsm_calibration_route": "TSM_LOGIT_SHIFT", "split": "tsm_test_2024", "decision_ece": 0.21, "brier_improvement_pct": 2.0},
                {"tsm_calibration_route": "TSM_LOGIT_SHIFT", "split": "tsm_final_holdout_2025_2026", "decision_ece": 0.12, "brier_improvement_pct": 2.0},
                {"tsm_calibration_route": "TSM_SHRUNK_LOGIT_SHIFT", "split": "tsm_combined_test_holdout", "decision_ece": 0.155, "brier_improvement_pct": 1.5},
                {"tsm_calibration_route": "TSM_SHRUNK_LOGIT_SHIFT", "split": "tsm_test_2024", "decision_ece": 0.19, "brier_improvement_pct": 1.5},
                {"tsm_calibration_route": "TSM_SHRUNK_LOGIT_SHIFT", "split": "tsm_final_holdout_2025_2026", "decision_ece": 0.12, "brier_improvement_pct": 1.5},
            ]
        )

        route, summary = choose_tsm_calibration_route(metrics)

        self.assertEqual(route, "POOLED_ONLY")
        self.assertTrue(bool(summary.loc[summary["tsm_calibration_route"].eq("POOLED_ONLY"), "tsm_calibration_route_pass"].iloc[0]))
        self.assertFalse(bool(summary.loc[summary["tsm_calibration_route"].eq("TSM_LOGIT_SHIFT"), "tsm_calibration_route_pass"].iloc[0]))

    def test_tsm_calibration_route_prefers_pooled_only_when_all_failed_routes_are_close(self):
        metrics = pd.DataFrame(
            [
                {"tsm_calibration_route": "POOLED_ONLY", "split": "tsm_combined_test_holdout", "decision_ece": 0.155, "brier_improvement_pct": -1.0},
                {"tsm_calibration_route": "POOLED_ONLY", "split": "tsm_test_2024", "decision_ece": 0.21, "brier_improvement_pct": -1.0},
                {"tsm_calibration_route": "POOLED_ONLY", "split": "tsm_final_holdout_2025_2026", "decision_ece": 0.16, "brier_improvement_pct": -1.0},
                {"tsm_calibration_route": "TSM_LOGIT_SHIFT", "split": "tsm_combined_test_holdout", "decision_ece": 0.152, "brier_improvement_pct": -0.5},
                {"tsm_calibration_route": "TSM_LOGIT_SHIFT", "split": "tsm_test_2024", "decision_ece": 0.21, "brier_improvement_pct": -0.5},
                {"tsm_calibration_route": "TSM_LOGIT_SHIFT", "split": "tsm_final_holdout_2025_2026", "decision_ece": 0.16, "brier_improvement_pct": -0.5},
            ]
        )

        route, summary = choose_tsm_calibration_route(metrics)

        self.assertEqual(route, "POOLED_ONLY")
        self.assertFalse(summary["tsm_calibration_route_pass"].map(bool).any())

    def test_tsm_calibration_route_v2_selects_from_train_validation_not_test_holdout(self):
        metrics = pd.DataFrame(
            [
                {"tsm_calibration_route": "POOLED_ONLY", "split": "tsm_train_validation", "decision_ece": 0.05, "brier_improvement_pct": 1.0},
                {"tsm_calibration_route": "POOLED_ONLY", "split": "tsm_combined_test_holdout", "decision_ece": 0.16, "brier_improvement_pct": -1.0},
                {"tsm_calibration_route": "POOLED_ONLY", "split": "tsm_test_2024", "decision_ece": 0.19, "brier_improvement_pct": -1.0},
                {"tsm_calibration_route": "POOLED_ONLY", "split": "tsm_final_holdout_2025_2026", "decision_ece": 0.16, "brier_improvement_pct": -1.0},
                {"tsm_calibration_route": "TSM_PLATT_SIGMOID", "split": "tsm_train_validation", "decision_ece": 0.13, "brier_improvement_pct": 0.5},
                {"tsm_calibration_route": "TSM_PLATT_SIGMOID", "split": "tsm_combined_test_holdout", "decision_ece": 0.04, "brier_improvement_pct": 4.0},
                {"tsm_calibration_route": "TSM_PLATT_SIGMOID", "split": "tsm_test_2024", "decision_ece": 0.04, "brier_improvement_pct": 4.0},
                {"tsm_calibration_route": "TSM_PLATT_SIGMOID", "split": "tsm_final_holdout_2025_2026", "decision_ece": 0.04, "brier_improvement_pct": 4.0},
            ]
        )

        route, summary = choose_tsm_calibration_route_v2(metrics)

        self.assertEqual(route, "POOLED_ONLY")
        self.assertTrue(summary["route_selection_provenance_valid"].map(bool).all())
        selected = summary[summary["is_selected_tsm_calibration_route"].map(bool)].iloc[0]
        self.assertFalse(bool(selected["tsm_calibration_route_pass"]))

    def test_next_required_action_waits_for_latest_when_model_quality_passes_with_tsm_fallback(self):
        eval_row = pd.Series(
            {
                "threshold_stability_pass": True,
                "uplift_pass": True,
            }
        )
        selected_tsm_eval = pd.Series(
            {
                "tsm_calibration_route_pass": False,
                "tsm_calibration_scoring_route": "POOLED_ONLY",
                "tsm_calibration_scoring_route_reason": "SELECTED_ROUTE_FAILED_IDENTITY_FALLBACK",
            }
        )

        action = next_required_evidence_action(
            eval_row,
            selected_tsm_eval,
            threshold_decision_eligible=True,
            model_quality_pass=True,
            latest_signal_pass=False,
        )

        self.assertEqual(action, "await_latest_trade_ready_signal")

    def test_next_required_action_keeps_tsm_calibration_action_without_scoring_fallback(self):
        eval_row = pd.Series(
            {
                "threshold_stability_pass": True,
                "uplift_pass": True,
            }
        )
        selected_tsm_eval = pd.Series(
            {
                "tsm_calibration_route_pass": False,
                "tsm_calibration_scoring_route": "TSM_DIRECT_EMPIRICAL_PRIOR",
                "tsm_calibration_scoring_route_reason": "",
            }
        )

        action = next_required_evidence_action(
            eval_row,
            selected_tsm_eval,
            threshold_decision_eligible=True,
            model_quality_pass=False,
            latest_signal_pass=False,
        )

        self.assertEqual(action, "improve_tsm_route_calibration_or_expand_tsm_like_calibration_sample")

    def test_update_latest_snapshot_only_promotes_when_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "latest.csv"
            pd.DataFrame(
                [
                    {"field": "prediction_use_status", "value": "DISPLAY_ONLY_NO_MODEL_CANDIDATE"},
                    {"field": "decision_permission", "value": "DISPLAY_ONLY"},
                ]
            ).to_csv(path, index=False)
            update_latest_snapshot(
                path,
                {
                    "decision_support_allowed": False,
                    "model_name": "pooled_empirical_bayes_group_rate",
                    "p_success_20d": 0.7,
                },
            )
            values = dict(zip(pd.read_csv(path)["field"], pd.read_csv(path)["value"]))
            self.assertEqual(values["prediction_use_status"], "DISPLAY_ONLY_NO_MODEL_CANDIDATE")
            self.assertEqual(values["pooled_model_name"], "pooled_empirical_bayes_group_rate")

            update_latest_snapshot(
                path,
                {
                    "decision_support_allowed": True,
                    "model_name": "pooled_empirical_bayes_group_rate",
                    "p_success_20d": 0.7,
                    "p_stop_survival_20d": 0.8,
                    "p_stop_hit_20d": 0.2,
                    "p_hit_1r_20d": 0.6,
                    "p_hit_2r_20d": 0.3,
                    "expected_r_net_20d": 0.5,
                    "expected_net_return_pct_20d": 3.0,
                    "threshold_20d": 0.55,
                    "oos_event_count": 200,
                    "selected_oos_event_count": 80,
                    "selected_expectancy_ci_lower_pct": 1.0,
                    "selected_minus_all_pct": 0.5,
                },
            )
            values = dict(zip(pd.read_csv(path)["field"], pd.read_csv(path)["value"]))
            self.assertEqual(values["prediction_use_status"], "DECISION_SUPPORT_ALLOWED")
            self.assertEqual(values["prediction_scope_used"], "pooled_trade_ready_entry")


if __name__ == "__main__":
    unittest.main()
