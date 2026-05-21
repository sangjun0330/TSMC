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
    build_latest_snapshot,
    candidate_tier,
    choose_threshold,
    event_candidate,
    evaluate_prediction_stream,
    label_event_horizon,
    model_quality_block_reasons,
    select_fold_features,
)
from tsm_pooled_dataset_builder import build_quality_checks as build_pooled_quality_checks
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
