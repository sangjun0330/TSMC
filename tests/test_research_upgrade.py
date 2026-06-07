import math
import unittest

import numpy as np
import pandas as pd

from tsm_core.schemas import validate_no_future_features
from tsm_core.splits import FoldSpec, PurgedEventTimeSplit
from tsm_prediction_engine import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    add_news_category_history_features,
    aggregate_model_comparison,
    apply_base_rate_probability_shrinkage,
    build_feature_association_summary,
    build_brier_decomposition_summary,
    build_latest_snapshot,
    build_model_audit,
    build_near_pass_candidates,
    build_performance_gap_summary,
    candidate_tier,
    choose_threshold,
    event_candidate,
    evaluate_prediction_stream,
    fit_validation_base_rate_shrinkage,
    label_event_horizon,
    model_quality_block_reasons,
    score_threshold_candidates,
    select_fold_features,
    top_probability_slice_metrics,
)
from tsm_pooled_dataset_builder import build_quality_checks as build_pooled_quality_checks
from tsm_pooled_dataset_builder import build_rule_threshold_sensitivity, build_rule_threshold_sensitivity_quality_checks, build_rule_threshold_sensitivity_summary
from tsm_pooled_dataset_builder import build_split_manifest, default_config


class ResearchUpgradeTests(unittest.TestCase):
    def test_news_causal_features_are_model_inputs(self):
        self.assertIn("news_primary_cause_type", CATEGORICAL_FEATURES)
        self.assertIn("news_match_confidence", CATEGORICAL_FEATURES)
        self.assertIn("news_penalty_event", NUMERIC_FEATURES)
        self.assertIn("hist_news_category_success_rate_20d", NUMERIC_FEATURES)

    def test_news_category_history_is_causal(self):
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=45, freq="B"),
                "close": [100 + i for i in range(45)],
                "news_primary_cause_type": [
                    "monthly_revenue" if i in {0, 10, 35} else "NO_HIGH_CONFIDENCE_NEWS" if i == 20 else "" for i in range(45)
                ],
            }
        )
        out = add_news_category_history_features(frame)
        self.assertEqual(out.loc[10, "hist_news_category_count_20d"], 0)
        self.assertEqual(out.loc[35, "hist_news_category_count_20d"], 2)
        self.assertGreater(out.loc[35, "hist_news_category_mean_return_20d"], 0)

    def test_schema_blocks_future_feature_names(self):
        blocked = validate_no_future_features(["score_price_algo_total", "label_success_20d", "fwd_return_20d", "exit_reason"])
        self.assertEqual(blocked, ["label_success_20d", "fwd_return_20d", "exit_reason"])

    def test_purged_split_removes_overlap_and_embargo(self):
        frame = pd.DataFrame({"signal_idx": np.arange(260), "value": np.arange(260)})
        splitter = PurgedEventTimeSplit(horizon_days=20, train_days=120, validation_days=40, test_days=40, step_days=40, embargo_days=20)
        spec = FoldSpec(
            fold_id=1,
            train_start_idx=0,
            train_end_idx=190,
            validation_start_idx=191,
            validation_end_idx=200,
            test_start_idx=201,
            test_end_idx=240,
            horizon_days=20,
            embargo_days=20,
        )
        train, validation, test, out_spec = splitter.apply(frame, spec)
        self.assertLessEqual(train["signal_idx"].max(), 170)
        self.assertGreater(out_spec.purged_train_count, 0)
        self.assertEqual(len(validation), 10)
        self.assertEqual(len(test), 40)

    def test_validation_days_config_is_not_overridden_by_test_days(self):
        splitter = PurgedEventTimeSplit(
            horizon_days=20,
            train_days=756,
            validation_days=63,
            test_days=252,
            step_days=63,
            embargo_days=20,
        )
        self.assertEqual(splitter.validation_days, 63)

    def test_feature_selector_uses_train_contract_and_caps_numeric_features(self):
        train = pd.DataFrame({f"feature_{i}": np.arange(50) + i for i in range(30)})
        train["feature_corr"] = train["feature_0"] * 1.0
        train["label_success_20d"] = [0, 1] * 25
        selected_numeric, _, rows = select_fold_features(
            train,
            [*list(train.columns), "missing_col"],
            [],
            horizon=20,
            candidate_scope_name="trigger_all",
            fold_id=1,
            target_name="success",
        )
        self.assertLessEqual(len(selected_numeric), 25)
        self.assertNotIn("label_success_20d", selected_numeric)
        self.assertTrue(any(r["feature"] == "label_success_20d" and r["reason"] == "future_or_label_feature_blocked" for r in rows))
        self.assertTrue(all(r.get("feature_selection_source") == "train_only" for r in rows if "feature_selection_source" in r))
        numeric_rows = [r for r in rows if r.get("feature_type") == "numeric" and r.get("reason") != "future_or_label_feature_blocked"]
        self.assertTrue(all("target_association" in r for r in numeric_rows))

    def test_feature_association_summary_prioritizes_high_association_exclusions(self):
        report = pd.DataFrame(
            [
                {
                    "horizon_days": 20,
                    "candidate_scope": "entry_research",
                    "feature_group": "intraday_context",
                    "feature": "hourly_liquidity",
                    "feature_type": "numeric",
                    "decision": "excluded",
                    "reason": "max_25_numeric_features",
                    "target_association": 0.42,
                    "missing_rate": 0.01,
                },
                {
                    "horizon_days": 20,
                    "candidate_scope": "entry_research",
                    "feature_group": "score_core",
                    "feature": "score_price_algo_total",
                    "feature_type": "numeric",
                    "decision": "selected",
                    "reason": "",
                    "target_association": 0.12,
                    "missing_rate": 0.0,
                },
            ]
        )

        summary = build_feature_association_summary(report)

        assert summary.iloc[0]["feature"] == "hourly_liquidity"
        assert summary.iloc[0]["max_25_cap_excluded_count"] == 1
        assert math.isclose(summary.iloc[0]["median_target_association"], 0.42)
        assert summary.iloc[1]["selected_count"] == 1

    def test_brier_decomposition_summary_flags_reliability_penalty(self):
        audit = pd.DataFrame(
            [
                {
                    "candidate_scope": "entry_research",
                    "horizon_days": 20,
                    "model_name": "elastic_net_logistic",
                    "model_policy": "DIAGNOSTIC_ONLY_NON_DECISION_SCOPE",
                    "prediction_quality_pass": False,
                    "quality_block_reasons": "NO_BRIER_IMPROVEMENT",
                    "brier_score": 0.26,
                    "brier_improvement_pct": -2.0,
                    "brier_reliability": 0.030,
                    "brier_resolution": 0.010,
                    "brier_uncertainty": 0.25,
                }
            ]
        )

        summary = build_brier_decomposition_summary(audit)

        self.assertEqual(summary.iloc[0]["brier_decomposition_regime"], "reliability_penalty_exceeds_resolution")
        self.assertGreater(summary.iloc[0]["brier_reliability_minus_resolution"], 0)
        self.assertEqual(summary.iloc[0]["recommended_brier_action"], "improve_probability_calibration_or_reduce_overfit")

    def test_threshold_requires_enough_validation_selection(self):
        threshold, expectancy, count, source = choose_threshold(
            np.array([0.51, 0.52, 0.53]),
            pd.Series([1.0, -1.0, 2.0]),
            thresholds=[0.50, 0.55],
            min_trades=20,
            default_threshold=0.60,
        )
        self.assertTrue(math.isnan(threshold))
        self.assertTrue(math.isnan(expectancy))
        self.assertEqual(count, 0)
        self.assertEqual(source, "INSUFFICIENT_VALIDATION_SELECTION")

    def test_threshold_adds_validation_quantile_candidates_when_fixed_grid_is_too_high(self):
        probabilities = np.linspace(0.01, 0.40, 40)
        returns = pd.Series([-1.0] * 20 + [2.0] * 20)

        threshold, expectancy, count, source = choose_threshold(
            probabilities,
            returns,
            thresholds=[0.50, 0.55],
            min_trades=20,
            default_threshold=0.60,
        )

        self.assertLess(threshold, 0.50)
        self.assertEqual(count, 20)
        self.assertAlmostEqual(expectancy, 2.0)
        self.assertTrue(source.startswith("TRAIN_VALIDATION_UTILITY_LOWER_BOUND_VALIDATION_TOP_"))

    def test_threshold_compares_validation_quantiles_against_valid_fixed_grid(self):
        probabilities = np.linspace(0.20, 0.90, 40)
        returns = pd.Series([-1.0] * 20 + [3.0] * 20)

        threshold, expectancy, count, source = choose_threshold(
            probabilities,
            returns,
            thresholds=[0.20],
            min_trades=20,
            default_threshold=0.20,
        )

        self.assertGreater(threshold, 0.20)
        self.assertEqual(count, 20)
        self.assertAlmostEqual(expectancy, 3.0)
        self.assertTrue(source.startswith("TRAIN_VALIDATION_UTILITY_LOWER_BOUND_VALIDATION_TOP_"))

    def test_threshold_candidate_scores_explain_selected_utility_lower_bound(self):
        probabilities = np.linspace(0.20, 0.90, 40)
        returns = pd.Series([-1.0] * 20 + [3.0] * 20)

        scores = score_threshold_candidates(
            probabilities,
            returns,
            thresholds=[0.20],
            min_trades=20,
            default_threshold=0.20,
        )
        best = scores.sort_values("validation_utility_lower_bound_pct", ascending=False).iloc[0]

        self.assertIn("validation_absolute_lower_bound_pct", scores.columns)
        self.assertIn("validation_uplift_lower_bound_pct", scores.columns)
        self.assertTrue(bool(best["validation_candidate_eligible"]))
        self.assertGreater(best["threshold"], 0.20)
        self.assertAlmostEqual(best["validation_expectancy_pct"], 3.0)
        self.assertGreater(best["validation_utility_lower_bound_pct"], 0.0)

    def test_model_audit_carries_validation_threshold_utility_context(self):
        comparison = pd.DataFrame(
            [
                {
                    "candidate_scope": "entry_research",
                    "horizon_days": 5,
                    "model_name": "empirical_bayes_group_rate",
                    "model_policy": "DIAGNOSTIC_ONLY_NON_DECISION_SCOPE",
                    "prediction_quality_pass": False,
                    "fold_count": 3,
                    "oos_event_count": 180,
                    "selected_oos_event_count": 90,
                    "brier_score": 0.24,
                    "base_rate_brier_score": 0.25,
                    "brier_improvement_pct": 4.0,
                    "ece": 0.06,
                    "decision_ece": 0.06,
                    "pr_auc": 0.58,
                    "base_rate_pr_auc": 0.50,
                    "expectancy_improvement_pct": 0.10,
                    "selected_signal_expectancy_ci_lower_pct": -0.05,
                    "selected_minus_rule_all_pct": 0.10,
                    "positive_expectancy_folds": 4,
                    "min_selected_events_per_fold": 20,
                    "decision_min_calibration_bin_n": 40,
                    "min_calibration_bin_n": 40,
                    "threshold_iqr": 0.02,
                }
            ]
        )
        threshold_policy = pd.DataFrame(
            [
                {
                    "candidate_scope": "entry_research",
                    "horizon_days": 5,
                    "model_name": "empirical_bayes_group_rate",
                    "fold_id": 1,
                    "threshold": 0.50,
                    "validation_utility_lower_bound_pct": -0.20,
                    "validation_threshold_candidate_count": 9,
                    "validation_threshold_eligible_candidate_count": 4,
                },
                {
                    "candidate_scope": "entry_research",
                    "horizon_days": 5,
                    "model_name": "empirical_bayes_group_rate",
                    "fold_id": 2,
                    "threshold": 0.55,
                    "validation_utility_lower_bound_pct": 0.30,
                    "validation_threshold_candidate_count": 10,
                    "validation_threshold_eligible_candidate_count": 5,
                },
            ]
        )

        audit = build_model_audit(pd.DataFrame(), comparison, pd.DataFrame(), threshold_policy)
        near_pass = build_near_pass_candidates(audit)
        row = near_pass.iloc[0]

        self.assertEqual(row["validation_utility_fold_count"], 2)
        self.assertEqual(row["validation_positive_utility_folds"], 1)
        self.assertAlmostEqual(row["min_validation_utility_lower_bound_pct"], -0.20)
        self.assertAlmostEqual(row["median_validation_utility_lower_bound_pct"], 0.05)
        self.assertEqual(row["min_validation_threshold_candidate_count"], 9)
        self.assertEqual(row["min_validation_threshold_eligible_candidate_count"], 4)

    def test_top_probability_slice_metrics_reports_threshold_free_oos_uplift(self):
        metrics = top_probability_slice_metrics(
            np.array([0.9, 0.8, 0.2, 0.1, 0.05]),
            pd.Series([5.0, 3.0, -1.0, -2.0, 0.0]),
            pd.Series([1, 1, 0, 0, 0]),
            fraction=0.40,
        )

        self.assertEqual(metrics["rank_top_quintile_count"], 2)
        self.assertAlmostEqual(metrics["rank_top_quintile_return_pct"], 4.0)
        self.assertAlmostEqual(metrics["rank_top_quintile_minus_all_pct"], 3.0)
        self.assertAlmostEqual(metrics["rank_top_quintile_success_rate"], 1.0)

    def test_validation_base_rate_shrinkage_uses_validation_brier_only(self):
        y = pd.Series([1] * 20 + [0] * 20)
        overconfident = np.array([0.95] * 20 + [0.60] * 20)

        result = fit_validation_base_rate_shrinkage(overconfident, y, 0.50)
        shrunk = apply_base_rate_probability_shrinkage(overconfident, 0.50, result["weight"])

        self.assertGreater(result["weight"], 0.0)
        self.assertLess(result["shrunk_brier"], result["raw_brier"])
        self.assertTrue(math.isfinite(result["improvement_lower_bound"]))
        self.assertLess(float(((shrunk - y) ** 2).mean()), float(((overconfident - y) ** 2).mean()))

    def test_validation_base_rate_shrinkage_aligns_probabilities_positionally(self):
        y = pd.Series([1] * 20 + [0] * 20, index=range(100, 140))
        overconfident = np.array([0.95] * 20 + [0.60] * 20)

        result = fit_validation_base_rate_shrinkage(overconfident, y, 0.50)

        self.assertTrue(math.isfinite(result["raw_brier"]))
        self.assertGreater(result["weight"], 0.0)
        self.assertLess(result["shrunk_brier"], result["raw_brier"])

    def test_validation_base_rate_shrinkage_can_select_conservative_high_weight(self):
        y = pd.Series([1] * 20 + [0] * 20)
        unskilled_overconfident = np.array([0.95] * 40)

        result = fit_validation_base_rate_shrinkage(unskilled_overconfident, y, 0.50)

        self.assertEqual(result["weight"], 0.95)
        self.assertLess(result["shrunk_brier"], result["raw_brier"])

    def test_aggregate_model_comparison_keeps_rank_uplift_separate_from_threshold_gate(self):
        metrics = pd.DataFrame(
            [
                {
                    "candidate_scope": "trade_ready_entry",
                    "horizon_days": 20,
                    "model_name": "score_logistic",
                    "fold_id": 1,
                    "test_event_count": 120,
                    "selected_signal_count": 0,
                    "brier_score": 0.20,
                    "base_rate_brier_score": 0.25,
                    "log_loss": 0.5,
                    "pr_auc": 0.60,
                    "base_rate_pr_auc": 0.50,
                    "ece": 0.05,
                    "decision_ece": 0.05,
                    "all_signal_expectancy_pct": 1.0,
                    "rank_top_quintile_count": 24,
                    "rank_top_quintile_return_pct": 4.0,
                    "rank_top_quintile_success_rate": 0.70,
                    "probability_shrinkage_weight": 0.25,
                    "validation_brier_raw_before_shrinkage": 0.30,
                    "validation_brier_after_shrinkage": 0.25,
                    "validation_brier_shrinkage_improvement_lower_bound": 0.02,
                    "selected_signal_expectancy_pct": np.nan,
                    "selected_signal_expectancy_ci_lower_pct": np.nan,
                    "calibration_min_bin_n": 40,
                    "decision_min_calibration_bin_n": 40,
                    "threshold": 0.50,
                }
            ]
        )

        comparison = aggregate_model_comparison(metrics)
        row = comparison.iloc[0]

        self.assertFalse(bool(row["prediction_quality_pass"]))
        self.assertAlmostEqual(row["rank_top_quintile_minus_all_pct"], 3.0)
        self.assertAlmostEqual(row["rank_top_quintile_success_rate"], 0.70)
        self.assertEqual(row["rank_top_quintile_fold_count"], 0)
        self.assertFalse(bool(row["rank_policy_diagnostic_pass"]))
        self.assertEqual(row["probability_shrinkage_applied_fold_count"], 1)
        self.assertAlmostEqual(row["validation_brier_shrinkage_improvement"], 0.05)
        self.assertAlmostEqual(row["validation_brier_shrinkage_improvement_lower_bound"], 0.02)

    def test_aggregate_model_comparison_marks_rank_policy_when_fold_evidence_is_stable(self):
        rows = []
        for fold_id, uplift in enumerate([1.2, 2.0, 1.5, 1.8], start=1):
            rows.append(
                {
                    "candidate_scope": "entry_research",
                    "horizon_days": 120,
                    "model_name": "score_logistic",
                    "fold_id": fold_id,
                    "test_event_count": 100,
                    "selected_signal_count": 0,
                    "brier_score": 0.24,
                    "base_rate_brier_score": 0.25,
                    "log_loss": 0.5,
                    "pr_auc": 0.60,
                    "base_rate_pr_auc": 0.50,
                    "ece": 0.05,
                    "decision_ece": 0.05,
                    "all_signal_expectancy_pct": 1.0,
                    "rank_top_quintile_count": 20,
                    "rank_top_quintile_return_pct": 1.0 + uplift,
                    "rank_top_quintile_minus_all_pct": uplift,
                    "rank_top_quintile_success_rate": 0.60,
                    "selected_signal_expectancy_pct": np.nan,
                    "selected_signal_expectancy_ci_lower_pct": np.nan,
                    "calibration_min_bin_n": 40,
                    "decision_min_calibration_bin_n": 40,
                    "threshold": 0.50,
                }
            )

        comparison = aggregate_model_comparison(pd.DataFrame(rows))
        row = comparison.iloc[0]

        self.assertEqual(row["rank_top_quintile_count"], 80)
        self.assertEqual(row["rank_top_quintile_fold_count"], 4)
        self.assertEqual(row["rank_top_quintile_positive_folds"], 4)
        self.assertGreater(row["rank_top_quintile_se_lower_pct"], 0.0)
        self.assertTrue(bool(row["rank_policy_diagnostic_pass"]))

    def test_aggregate_model_comparison_uses_pooled_selected_event_ci_not_worst_fold(self):
        rows = []
        for fold_id, count, mean, std, fold_lower in [
            (1, 10, -5.0, 1.0, -5.32),
            (2, 90, 2.0, 1.0, 1.89),
        ]:
            rows.append(
                {
                    "candidate_scope": "entry_research",
                    "horizon_days": 60,
                    "model_name": "empirical_bayes_group_rate",
                    "fold_id": fold_id,
                    "test_event_count": 100,
                    "selected_signal_count": count,
                    "brier_score": 0.24,
                    "base_rate_brier_score": 0.25,
                    "log_loss": 0.5,
                    "pr_auc": 0.60,
                    "base_rate_pr_auc": 0.50,
                    "ece": 0.05,
                    "decision_ece": 0.05,
                    "all_signal_expectancy_pct": 0.0,
                    "rank_top_quintile_count": 20,
                    "rank_top_quintile_return_pct": 1.0,
                    "rank_top_quintile_minus_all_pct": 1.0,
                    "rank_top_quintile_success_rate": 0.60,
                    "selected_signal_expectancy_pct": mean,
                    "selected_signal_expectancy_ci_lower_pct": fold_lower,
                    "selected_signal_return_std_pct": std,
                    "calibration_min_bin_n": 40,
                    "decision_min_calibration_bin_n": 40,
                    "threshold": 0.50,
                }
            )

        comparison = aggregate_model_comparison(pd.DataFrame(rows))
        row = comparison.iloc[0]

        self.assertGreater(row["selected_signal_expectancy_ci_lower_pct"], 0.0)
        self.assertEqual(row["min_fold_selected_expectancy_ci_lower_pct"], -5.32)
        self.assertEqual(row["selected_expectancy_ci_method"], "pooled_oos_selected_event_one_se")

    def test_performance_quality_pass_ignores_non_decision_scope_blocker(self):
        rows = []
        for fold_id in range(1, 5):
            rows.append(
                {
                    "candidate_scope": "entry_research",
                    "horizon_days": 10,
                    "model_name": "empirical_bayes_group_rate",
                    "fold_id": fold_id,
                    "test_event_count": 50,
                    "selected_signal_count": 15,
                    "brier_score": 0.20,
                    "base_rate_brier_score": 0.25,
                    "log_loss": 0.5,
                    "pr_auc": 0.60,
                    "base_rate_pr_auc": 0.50,
                    "ece": 0.05,
                    "decision_ece": 0.05,
                    "all_signal_expectancy_pct": 0.0,
                    "rank_top_quintile_count": 10,
                    "rank_top_quintile_return_pct": 1.0,
                    "rank_top_quintile_minus_all_pct": 1.0,
                    "rank_top_quintile_success_rate": 0.60,
                    "selected_signal_expectancy_pct": 1.0,
                    "selected_signal_expectancy_ci_lower_pct": 0.80,
                    "selected_signal_return_std_pct": 0.20,
                    "calibration_min_bin_n": 40,
                    "decision_min_calibration_bin_n": 40,
                    "threshold": 0.50,
                }
            )

        comparison = aggregate_model_comparison(pd.DataFrame(rows))
        row = comparison.iloc[0]

        self.assertTrue(bool(row["prediction_quality_pass"]))
        self.assertEqual(row["quality_block_reasons"], "NOT_20D_TRADE_READY_DECISION_SCOPE")
        self.assertTrue(bool(row["performance_quality_pass"]))
        self.assertEqual(row["performance_quality_block_reasons"], "PASS")

    def test_near_pass_candidates_prioritize_few_performance_blockers(self):
        audit = pd.DataFrame(
            [
                {
                    "candidate_scope": "entry_research",
                    "horizon_days": 10,
                    "model_name": "elastic_net_logistic",
                    "model_policy": "DIAGNOSTIC_ONLY_NON_DECISION_SCOPE",
                    "prediction_quality_pass": False,
                    "quality_block_reasons": "NOT_20D_TRADE_READY_DECISION_SCOPE|NO_BRIER_IMPROVEMENT",
                    "rank_score": -0.9,
                    "oos_event_count": 411,
                    "selected_oos_event_count": 372,
                    "brier_improvement_pct": -1.2,
                    "decision_ece": 0.08,
                    "pr_auc": 0.58,
                    "base_rate_pr_auc": 0.51,
                    "selected_minus_rule_all_pct": 0.1,
                    "selected_signal_expectancy_ci_lower_pct": 0.3,
                    "positive_expectancy_folds": 4,
                    "min_selected_events_per_fold": 20,
                    "min_calibration_bin_n": 40,
                    "threshold_iqr": 0.05,
                    "probability_shrinkage_weight": 0.75,
                    "probability_shrinkage_applied_fold_count": 4,
                    "validation_brier_shrinkage_improvement": 0.02,
                    "validation_brier_shrinkage_improvement_lower_bound": 0.01,
                },
                {
                    "candidate_scope": "entry_research",
                    "horizon_days": 5,
                    "model_name": "empirical_bayes_group_rate",
                    "model_policy": "DIAGNOSTIC_ONLY_NON_DECISION_SCOPE",
                    "prediction_quality_pass": False,
                    "quality_block_reasons": "NOT_20D_TRADE_READY_DECISION_SCOPE|SELECTED_EXPECTANCY_CI_LOWER_LE_0|THRESHOLD_IQR_GT_0_10",
                    "rank_score": -0.4,
                    "oos_event_count": 416,
                    "selected_oos_event_count": 314,
                    "brier_improvement_pct": 0.1,
                    "decision_ece": 0.05,
                    "pr_auc": 0.55,
                    "base_rate_pr_auc": 0.50,
                    "selected_minus_rule_all_pct": 0.08,
                    "selected_signal_expectancy_ci_lower_pct": -0.1,
                    "positive_expectancy_folds": 4,
                    "min_selected_events_per_fold": 20,
                    "min_calibration_bin_n": 40,
                    "threshold_iqr": 0.15,
                    "probability_shrinkage_weight": 0.25,
                    "probability_shrinkage_applied_fold_count": 2,
                    "validation_brier_shrinkage_improvement": 0.01,
                    "validation_brier_shrinkage_improvement_lower_bound": -0.002,
                },
            ]
        )

        near_pass = build_near_pass_candidates(audit)

        assert list(near_pass["performance_block_count"]) == [1, 2]
        assert near_pass.iloc[0]["performance_block_reasons"] == "NO_BRIER_IMPROVEMENT"
        assert near_pass.iloc[0]["recommended_action"] == "improve_validation_regularized_probability_scale"
        assert near_pass.iloc[0]["sample_or_scope_block_reasons"] == "NOT_20D_TRADE_READY_DECISION_SCOPE"
        assert near_pass.iloc[0]["brier_gap_pct"] == 1.2
        assert near_pass.iloc[0]["ece_gap"] == 0.0
        assert near_pass.iloc[0]["pr_auc_gap"] == 0.0
        assert near_pass.iloc[0]["selected_ci_gap_pct"] == 0.0
        assert near_pass.iloc[0]["probability_shrinkage_weight"] == 0.75
        assert near_pass.iloc[0]["validation_brier_shrinkage_improvement_lower_bound"] == 0.01
        assert near_pass.iloc[1]["selected_ci_gap_pct"] == 0.1
        assert math.isclose(near_pass.iloc[1]["threshold_iqr_gap"], 0.05)

    def test_performance_gap_summary_groups_reasons_and_top_candidates(self):
        near_pass = pd.DataFrame(
            [
                {
                    "candidate_scope": "entry_research",
                    "horizon_days": 10,
                    "model_name": "elastic_net_logistic",
                    "performance_block_count": 1,
                    "sample_or_scope_block_count": 1,
                    "performance_block_reasons": "NO_BRIER_IMPROVEMENT",
                    "rank_score": -0.9,
                    "brier_gap_pct": 1.2,
                    "ece_gap": 0.0,
                    "pr_auc_gap": 0.0,
                    "selected_ci_gap_pct": 0.0,
                    "threshold_iqr_gap": 0.0,
                    "selected_oos_event_gap": 0.0,
                    "positive_expectancy_fold_gap": 0.0,
                    "calibration_bin_gap": 0.0,
                },
                {
                    "candidate_scope": "entry_research",
                    "horizon_days": 20,
                    "model_name": "coverage_aware_ensemble",
                    "performance_block_count": 2,
                    "sample_or_scope_block_count": 1,
                    "performance_block_reasons": "NO_BRIER_IMPROVEMENT|ECE_GT_0_10",
                    "rank_score": -2.0,
                    "brier_gap_pct": 8.0,
                    "ece_gap": 0.09,
                    "pr_auc_gap": 0.0,
                    "selected_ci_gap_pct": 0.0,
                    "threshold_iqr_gap": 0.0,
                    "selected_oos_event_gap": 0.0,
                    "positive_expectancy_fold_gap": 0.0,
                    "calibration_bin_gap": 0.0,
                },
            ]
        )

        summary = build_performance_gap_summary(near_pass)
        brier = summary.loc[summary["performance_block_reason"].eq("NO_BRIER_IMPROVEMENT")].iloc[0]
        ece = summary.loc[summary["performance_block_reason"].eq("ECE_GT_0_10")].iloc[0]

        assert brier["candidate_count"] == 2
        assert brier["one_block_candidate_count"] == 1
        assert brier["top_candidate_model_name"] == "elastic_net_logistic"
        assert brier["recommended_action"] == "improve_validation_regularized_probability_scale"
        assert math.isclose(brier["max_brier_gap_pct"], 8.0)
        assert ece["candidate_count"] == 1
        assert math.isclose(ece["max_ece_gap"], 0.09)

    def test_sparse_trade_ready_scope_produces_diagnostic_only_metrics(self):
        rows = []
        for i in range(80):
            success = 0 if i % 4 == 0 else 1
            rows.append(
                {
                    "signal_idx": i * 30,
                    "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                    "candidate_scope": "trade_ready_entry",
                    "is_trade_ready_entry_candidate": True,
                    "label_status_20d": "LABELED",
                    "label_success_20d": success,
                    "label_net_return_pct_20d": 2.0 if success else -1.0,
                    "label_expected_r_20d": 0.8 if success else -0.5,
                    "label_stop_survival_20d": success,
                    "label_positive_return_20d": success,
                    "label_hit_1r_before_stop_20d": success,
                    "label_hit_2r_before_stop_20d": 1 if success and i % 3 == 0 else 0,
                    "label_ambiguous_stop_1r_same_day_20d": 0,
                    "label_ambiguous_stop_2r_same_day_20d": 0,
                    "label_gap_through_stop_20d": 0,
                    "label_first_touch_type_20d": "TARGET" if success else "STOP",
                    "label_exit_reason_20d": "HORIZON_20D" if success else "ATR_STOP_2X",
                    "entry_trigger": "A" if i % 2 == 0 else "B",
                    "trade_action": "ENTRY_ALLOWED",
                    "trend_regime": "UP",
                    "vol_regime": "NORMAL" if i % 3 else "HIGH",
                    "drawdown_bucket": "shallow",
                    "entry_gate_status": "READY",
                    "prediction_universe": "trade_ready_entry",
                    "score_price_algo_total": 60 + i % 20,
                }
            )
        frame = pd.DataFrame(rows)
        metrics, thresholds, _, _, oos_predictions, fold_manifest = evaluate_prediction_stream(
            frame,
            20,
            "trade_ready_entry",
            "is_trade_ready_entry_candidate",
            120,
            40,
            40,
            40,
            20,
            [0.50, 0.60],
            20,
            0.50,
            5,
        )
        self.assertFalse(metrics.empty)
        self.assertFalse(oos_predictions.empty)
        self.assertEqual(set(metrics["candidate_scope"]), {"trade_ready_entry"})
        self.assertEqual(set(metrics["model_name"]), {"empirical_bayes_group_rate", "base_rate_by_trigger_regime"})
        self.assertTrue((metrics["test_event_count"] < 100).all())
        self.assertTrue(thresholds["threshold_source"].astype(str).str.startswith("SPARSE_TRADE_READY_").all())
        self.assertEqual(fold_manifest.iloc[0]["status"], "SPARSE_DIAGNOSTIC_USED")
        comparison = aggregate_model_comparison(metrics)
        self.assertTrue((comparison["model_policy"] == "DIAGNOSTIC_ONLY_INSUFFICIENT_SAMPLE").all())
        self.assertFalse(comparison["prediction_quality_pass"].any())

    def test_latest_snapshot_gate_failure_stays_display_only(self):
        feature_matrix = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2024-01-01"),
                    "signal_idx": 1,
                    "is_event_candidate": True,
                    "is_actionable_entry_candidate": True,
                    "is_trade_ready_entry_candidate": True,
                    "entry_gate_status": "TRADE_READY",
                    "candidate_scope": "ACTIONABLE_ENTRY_ALLOWED",
                    "prediction_universe": "trade_ready_entry",
                    "entry_trigger": "20D_BREAKOUT",
                    "trade_action": "ENTRY_ALLOWED",
                }
            ]
        )
        latest = build_latest_snapshot(feature_matrix, pd.DataFrame(), pd.DataFrame())
        values = dict(zip(latest["field"], latest["value"]))
        self.assertNotEqual(values["prediction_use_status"], "DECISION_SUPPORT_ALLOWED")
        self.assertTrue(str(values["prediction_use_status"]).startswith("DISPLAY_ONLY_"))

    def test_label_flags_same_day_stop_target_ambiguity(self):
        frame = pd.DataFrame(
            [
                {"date": pd.Timestamp("2024-01-01"), "open": 100, "high": 101, "low": 99, "close": 100, "atr_14": 2, "entry_trigger": "20D_BREAKOUT", "trade_action": "ENTRY_ALLOWED", "trend_regime": "up", "vol_regime": "normal"},
                {"date": pd.Timestamp("2024-01-02"), "open": 100, "high": 110, "low": 95, "close": 101, "atr_14": 2, "entry_trigger": "NONE", "trade_action": "NO_TRADE", "trend_regime": "up", "vol_regime": "normal"},
                {"date": pd.Timestamp("2024-01-03"), "open": 101, "high": 102, "low": 100, "close": 101, "atr_14": 2, "entry_trigger": "NONE", "trade_action": "NO_TRADE", "trend_regime": "up", "vol_regime": "normal"},
                {"date": pd.Timestamp("2024-01-04"), "open": 101, "high": 102, "low": 100, "close": 101, "atr_14": 2, "entry_trigger": "NONE", "trade_action": "NO_TRADE", "trend_regime": "up", "vol_regime": "normal"},
            ]
        )
        frame["is_event_candidate"] = [True, False, False, False]
        labels = label_event_horizon(frame, 0, 3, commission_bps=0, slippage_bps=0, stop_multiple=2)
        self.assertEqual(labels["label_success_3d"], 0)
        self.assertEqual(labels["label_hit_1r_before_stop_3d"], 0)
        self.assertEqual(labels["label_ambiguous_stop_1r_same_day_3d"], 1)
        self.assertEqual(labels["label_ambiguous_stop_2r_same_day_3d"], 1)
        self.assertEqual(labels["label_first_touch_type_3d"], "AMBIGUOUS_STOP_2R_SAME_DAY")

    def test_label_flags_gap_through_stop_and_entry_gap(self):
        frame = pd.DataFrame(
            [
                {"date": pd.Timestamp("2024-01-01"), "open": 100, "high": 101, "low": 99, "close": 100, "atr_14": 2, "entry_trigger": "20D_BREAKOUT", "trade_action": "ENTRY_ALLOWED", "trend_regime": "up", "vol_regime": "normal"},
                {"date": pd.Timestamp("2024-01-02"), "open": 102, "high": 103, "low": 101, "close": 102, "atr_14": 2, "entry_trigger": "NONE", "trade_action": "NO_TRADE", "trend_regime": "up", "vol_regime": "normal"},
                {"date": pd.Timestamp("2024-01-03"), "open": 97, "high": 98, "low": 96, "close": 97, "atr_14": 2, "entry_trigger": "NONE", "trade_action": "NO_TRADE", "trend_regime": "up", "vol_regime": "normal"},
                {"date": pd.Timestamp("2024-01-04"), "open": 97, "high": 98, "low": 96, "close": 97, "atr_14": 2, "entry_trigger": "NONE", "trade_action": "NO_TRADE", "trend_regime": "up", "vol_regime": "normal"},
            ]
        )
        frame["is_event_candidate"] = [True, False, False, False]
        labels = label_event_horizon(frame, 0, 3, commission_bps=0, slippage_bps=0, stop_multiple=2)
        self.assertEqual(labels["label_gap_through_stop_3d"], 1)
        self.assertAlmostEqual(labels["label_entry_gap_pct_3d"], 2.0)
        self.assertEqual(labels["label_first_touch_type_3d"], "STOP")

    def test_candidate_tiers_expand_training_without_changing_decision_alias(self):
        rows = pd.DataFrame(
            [
                {
                    "entry_trigger": "20D_BREAKOUT",
                    "trade_action": "ENTRY_ALLOWED",
                    "score_price_algo_total": 75,
                    "algo_vol_extreme": False,
                    "algo_overextended_highvol": False,
                    "algo_deep_downtrend_avoid": False,
                    "algo_trend_up_loose": True,
                    "close": 110,
                    "sma_200": 100,
                },
                {
                    "entry_trigger": "PULLBACK_50D",
                    "trade_action": "NO_TRADE",
                    "score_price_algo_total": 65,
                    "algo_vol_extreme": False,
                    "algo_overextended_highvol": False,
                    "algo_deep_downtrend_avoid": False,
                    "algo_trend_up_loose": True,
                    "close": 110,
                    "sma_200": 100,
                },
                {
                    "entry_trigger": "NONE",
                    "trade_action": "HOLD_OR_WAIT_TRIGGER",
                    "score_price_algo_total": 68,
                    "algo_vol_extreme": False,
                    "algo_overextended_highvol": False,
                    "algo_deep_downtrend_avoid": False,
                    "algo_trend_up_loose": True,
                    "close": 90,
                    "sma_200": 100,
                },
                {
                    "entry_trigger": "PULLBACK_50D",
                    "trade_action": "NO_TRADE",
                    "score_price_algo_total": 60,
                    "algo_vol_extreme": False,
                    "algo_overextended_highvol": False,
                    "algo_deep_downtrend_avoid": False,
                    "algo_trend_up_loose": True,
                    "close": 110,
                    "sma_200": 100,
                },
                {
                    "entry_trigger": "20D_BREAKOUT",
                    "trade_action": "NO_TRADE",
                    "score_price_algo_total": 80,
                    "algo_vol_extreme": True,
                    "algo_overextended_highvol": False,
                    "algo_deep_downtrend_avoid": False,
                    "algo_trend_up_loose": True,
                    "close": 110,
                    "sma_200": 100,
                },
                {
                    "entry_trigger": "NONE",
                    "trade_action": "NO_TRADE",
                    "score_price_algo_total": 80,
                    "algo_vol_extreme": False,
                    "algo_overextended_highvol": False,
                    "algo_deep_downtrend_avoid": False,
                    "algo_trend_up_loose": True,
                    "close": 110,
                    "sma_200": 100,
                },
            ]
        )
        tiers = rows.apply(candidate_tier, axis=1).tolist()
        self.assertEqual(
            tiers,
            [
                "decision_trade_ready",
                "relaxed_trigger_score65",
                "setup_context_score65",
                "relaxed_trigger_score60",
                "risk_blocked_research",
                "none",
            ],
        )
        self.assertEqual(tiers.count("decision_trade_ready"), 1)

    def test_label_event_horizon_labels_setup_training_candidates_but_not_none(self):
        frame = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2024-01-01"),
                    "open": 100,
                    "high": 101,
                    "low": 99,
                    "close": 100,
                    "sma_200": 105,
                    "atr_14": 2,
                    "entry_trigger": "NONE",
                    "trade_action": "HOLD_OR_WAIT_TRIGGER",
                    "score_price_algo_total": 70,
                    "algo_vol_extreme": False,
                    "algo_trend_up_loose": True,
                    "trend_regime": "up",
                    "vol_regime": "normal",
                },
                {"date": pd.Timestamp("2024-01-02"), "open": 101, "high": 103, "low": 100, "close": 102, "atr_14": 2, "entry_trigger": "NONE", "trade_action": "NO_TRADE", "trend_regime": "up", "vol_regime": "normal"},
                {"date": pd.Timestamp("2024-01-03"), "open": 102, "high": 104, "low": 101, "close": 103, "atr_14": 2, "entry_trigger": "NONE", "trade_action": "NO_TRADE", "trend_regime": "up", "vol_regime": "normal"},
                {"date": pd.Timestamp("2024-01-04"), "open": 103, "high": 105, "low": 102, "close": 104, "atr_14": 2, "entry_trigger": "NONE", "trade_action": "NO_TRADE", "trend_regime": "up", "vol_regime": "normal"},
                {"date": pd.Timestamp("2024-01-05"), "open": 104, "high": 106, "low": 103, "close": 105, "atr_14": 2, "entry_trigger": "NONE", "trade_action": "NO_TRADE", "trend_regime": "up", "vol_regime": "normal"},
            ]
        )
        frame["candidate_tier"] = frame.apply(candidate_tier, axis=1)
        frame["is_event_candidate"] = frame.apply(event_candidate, axis=1)
        labeled = label_event_horizon(frame, 0, 3, commission_bps=0, slippage_bps=0, stop_multiple=2)
        self.assertEqual(labeled["label_status_3d"], "LABELED")
        none_row = label_event_horizon(frame, 1, 3, commission_bps=0, slippage_bps=0, stop_multiple=2)
        self.assertEqual(none_row["label_status_3d"], "NOT_EVENT")

    def test_model_gate_blocks_selected_count_negative_edge_and_threshold_instability(self):
        metrics = pd.DataFrame(
            [
                {
                    "candidate_scope": "trade_ready_entry",
                    "horizon_days": 20,
                    "model_name": "score_logistic",
                    "fold_id": 1,
                    "test_event_count": 120,
                    "selected_signal_count": 0,
                    "brier_score": 0.20,
                    "base_rate_brier_score": 0.25,
                    "log_loss": 0.5,
                    "pr_auc": 0.60,
                    "base_rate_pr_auc": 0.50,
                    "ece": 0.05,
                    "all_signal_expectancy_pct": 1.0,
                    "selected_signal_expectancy_pct": np.nan,
                    "selected_signal_expectancy_ci_lower_pct": np.nan,
                    "positive_expectancy_folds": 0,
                    "calibration_min_bin_n": 10,
                    "mean_effective_sample_size": 100,
                    "threshold": 0.50,
                },
                {
                    "candidate_scope": "trade_ready_entry",
                    "horizon_days": 20,
                    "model_name": "score_logistic",
                    "fold_id": 2,
                    "test_event_count": 120,
                    "selected_signal_count": 0,
                    "brier_score": 0.20,
                    "base_rate_brier_score": 0.25,
                    "log_loss": 0.5,
                    "pr_auc": 0.60,
                    "base_rate_pr_auc": 0.50,
                    "ece": 0.05,
                    "all_signal_expectancy_pct": 1.0,
                    "selected_signal_expectancy_pct": np.nan,
                    "selected_signal_expectancy_ci_lower_pct": np.nan,
                    "positive_expectancy_folds": 0,
                    "calibration_min_bin_n": 10,
                    "mean_effective_sample_size": 100,
                    "threshold": 0.80,
                },
            ]
        )
        comparison = aggregate_model_comparison(metrics)
        row = comparison.iloc[0]
        self.assertFalse(bool(row["prediction_quality_pass"]))
        reasons = model_quality_block_reasons(row)
        self.assertIn("SELECTED_OOS_EVENT_COUNT_LT_50", reasons)
        self.assertIn("ML_SELECTED_MINUS_RULE_ALL_LE_0", reasons)
        self.assertIn("CALIBRATION_MIN_BIN_N_LT_30", reasons)
        self.assertIn("THRESHOLD_IQR_GT_0_10", reasons)

    def test_pooled_default_universe_and_tsm_only_quality_fails_decision_targets(self):
        config = default_config()
        self.assertGreaterEqual(config["symbol"].nunique(), 10)
        features = pd.DataFrame(
            {
                "symbol": ["TSM", "TSM"],
                "symbol_group": ["foundry", "foundry"],
                "date": [pd.Timestamp("2022-01-03"), pd.Timestamp("2024-01-03")],
                "signal_idx": [1, 2],
            }
        )
        labels = features.copy()
        labels["is_trade_ready_entry_candidate"] = [True, False]
        labels["label_status_20d"] = ["LABELED", "LABELED"]
        labels["label_status_60d"] = ["LABELED", "LABELED"]
        schema = pd.DataFrame({"role": ["numeric_feature"] * 80})
        split_manifest = build_split_manifest(features)
        quality = build_pooled_quality_checks(config.iloc[[0]], labels, features, schema, pd.DataFrame(), split_manifest)
        checks = dict(zip(quality["check"], quality["passed"]))
        self.assertFalse(checks["pooled_loaded_symbol_count_at_least_10"])
        self.assertFalse(checks["pooled_trade_ready_20d_labeled_at_least_500"])
        self.assertFalse(checks["pooled_model_training_20d_labeled_at_least_10000"])
        self.assertTrue(checks["pooled_symbol_date_signal_unique"])

    def test_pooled_decision_coverage_requires_all_12_symbols_when_configured(self):
        symbols = ["NVDA", "TSM", "AVGO", "AMD", "INTC", "MU", "TXN", "LRCX", "AMAT", "QCOM", "005930.KS", "000660.KS"]
        config = pd.DataFrame(
            {
                "symbol": symbols,
                "symbol_group": ["semiconductor"] * len(symbols),
                "is_decision_universe": [True] * len(symbols),
            }
        )
        loaded = symbols[:-2]
        features = pd.DataFrame(
            {
                "symbol": loaded,
                "date": pd.date_range("2024-01-01", periods=len(loaded), freq="B"),
                "signal_idx": range(len(loaded)),
            }
        )
        labels = features.copy()
        labels["is_trade_ready_entry_candidate"] = True
        labels["label_status_20d"] = "LABELED"
        schema = pd.DataFrame({"role": ["numeric_feature"] * 80})
        quality = build_pooled_quality_checks(config, labels, features, schema, pd.DataFrame(), build_split_manifest(features))
        checks = dict(zip(quality["check"], quality["passed"]))

        self.assertTrue(checks["pooled_decision_symbol_count_eq_required"])
        self.assertFalse(checks["pooled_decision_symbols_loaded_all"])

    def test_rule_threshold_sensitivity_grid_is_complete_and_advisory(self):
        features = pd.DataFrame(
            {
                "symbol": ["TSM", "NVDA", "TSM", "NVDA"],
                "date": pd.date_range("2024-01-01", periods=4, freq="B"),
                "signal_idx": [1, 2, 3, 4],
                "score_price_algo_total": [76.0, 72.0, 66.0, 58.0],
                "entry_trigger": ["20D_BREAKOUT", "NONE", "50D_PULLBACK_BOUNCE", "NONE"],
                "algo_trend_up_loose": [True, True, True, False],
                "algo_vol_high": [False, True, False, True],
                "algo_vol_extreme": [False, False, False, False],
                "dist_close_sma_50_pct": [0.04, 0.18, 0.03, 0.30],
            }
        )
        labels = features[["symbol", "date", "signal_idx"]].copy()
        labels["label_net_return_pct_20d"] = [2.0, -1.0, 1.0, -2.0]
        labels["label_net_return_pct_60d"] = [4.0, -2.0, 2.0, -3.0]
        labels["label_expected_r_20d"] = [0.5, -0.2, 0.3, -0.5]
        labels["label_stop_survival_20d"] = [1, 0, 1, 0]

        sensitivity = build_rule_threshold_sensitivity(labels, features)
        summary = build_rule_threshold_sensitivity_summary(sensitivity)
        quality = build_rule_threshold_sensitivity_quality_checks(sensitivity, summary)
        checks = dict(zip(quality["check"], quality["passed"]))

        self.assertEqual(len(sensitivity), 625)
        self.assertTrue(checks["rule_threshold_sensitivity_grid_complete"])
        self.assertTrue(checks["rule_threshold_sensitivity_baseline_present"])
        self.assertFalse(bool(summary.iloc[0]["config_mutation_allowed"]))

    def test_tree_models_are_not_run_for_small_trigger_research_scope(self):
        frame = pd.DataFrame(
            {
                "signal_idx": range(260),
                "date": pd.date_range("2020-01-01", periods=260, freq="B"),
                "candidate_scope": ["trigger_all"] * 260,
                "is_actionable_entry_candidate": [True] * 260,
                "label_status_20d": ["LABELED"] * 260,
                "label_success_20d": [1, 0] * 130,
                "label_net_return_pct_20d": [2.0, -1.0] * 130,
                "label_expected_r_20d": [0.6, -0.4] * 130,
                "label_stop_survival_20d": [1, 0] * 130,
                "label_positive_return_20d": [1, 0] * 130,
                "label_hit_1r_before_stop_20d": [1, 0] * 130,
                "label_hit_2r_before_stop_20d": [0, 0] * 130,
                "label_exit_reason_20d": ["HORIZON_20D", "ATR_STOP_2X"] * 130,
                "entry_trigger": ["A", "B"] * 130,
                "trend_regime": ["UP"] * 260,
                "vol_regime": ["NORMAL"] * 260,
                "drawdown_bucket": ["shallow"] * 260,
                "entry_gate_status": ["READY"] * 260,
                "prediction_universe": ["trigger_all"] * 260,
                "score_price_algo_total": [60 + (i % 30) for i in range(260)],
                "atr_14_pct": [0.03] * 260,
                "risk_pct_2atr": [0.06] * 260,
            }
        )
        metrics, *_ = evaluate_prediction_stream(
            frame,
            20,
            "trigger_all",
            "is_actionable_entry_candidate",
            120,
            40,
            40,
            40,
            20,
            [0.5, 0.6],
            20,
            0.5,
            5,
        )
        models = set(metrics["model_name"]) if not metrics.empty else set()
        self.assertNotIn("random_forest_fixed", models)
        self.assertNotIn("hist_gradient_boosting_fixed", models)


if __name__ == "__main__":
    unittest.main()
