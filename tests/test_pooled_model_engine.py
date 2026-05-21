import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tsm_pooled_model_engine import (
    apply_tsm_layer,
    apply_logit_shift,
    build_oof_metrics,
    candidate_training_weights,
    choose_fold_consensus_trade_ready_threshold_v4,
    choose_fold_consensus_trade_ready_threshold_v5,
    choose_tsm_calibration_route,
    choose_tsm_calibration_route_v2,
    choose_threshold,
    choose_stable_trade_ready_threshold_v3,
    fit_empirical_bayes,
    fit_probability_calibrator,
    fit_scope_logit_shift,
    fit_tsm_calibration_layer,
    metric_row,
    paired_bootstrap_uplift,
    predict_empirical_bayes,
    pooled_feature_columns,
    prepare_dataset,
    score_baseline_mask_with_policy,
    score_baseline_returns_with_policy,
    update_latest_snapshot,
)


class PooledModelEngineTests(unittest.TestCase):
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
            threshold_rows.append(common)
            test_common = dict(common)
            test_common["date"] = pd.Timestamp("2024-08-01") + pd.Timedelta(days=i)
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
        self.assertEqual(records["threshold_policy_applied_window"].iloc[0], "test")
        self.assertTrue(records["threshold_policy_provenance_valid"].map(bool).all())
        self.assertEqual(records["applied_threshold_policy_type"].iloc[0], "rank_percentile_policy")
        self.assertLessEqual(unselected["risk_adjusted_selection_score"].max(), selected["risk_adjusted_selection_score"].min())
        self.assertEqual(int(selected["label_success_20d"].sum()), 0)

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
