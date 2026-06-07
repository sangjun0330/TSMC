import unittest

import pandas as pd

from tsm_prediction_engine import add_train_history_features
from tsm_price_rule_engine import add_semiconductor_momentum_v2_layers, add_two_stage_signal_layers
from tsm_shadow_paper_engine import paper_action
from tsm_system_state_engine import build_latest_state, build_scorecard, prediction_block_reason_from_snapshot


class PredictionHistoryFeatureTests(unittest.TestCase):
    def test_train_history_features_use_prior_rows_only(self):
        train = pd.DataFrame(
            [
                {
                    "signal_idx": 1,
                    "entry_trigger": "A",
                    "trend_regime": "UP",
                    "entry_gate_status": "READY",
                    "prediction_universe": "trigger_all",
                    "label_success_20d": 1,
                    "label_net_return_pct_20d": 4.0,
                    "label_exit_reason_20d": "HORIZON_20D",
                },
                {
                    "signal_idx": 2,
                    "entry_trigger": "B",
                    "trend_regime": "UP",
                    "entry_gate_status": "FILTERED",
                    "prediction_universe": "trigger_all",
                    "label_success_20d": 0,
                    "label_net_return_pct_20d": -2.0,
                    "label_exit_reason_20d": "ATR_STOP_2X",
                },
                {
                    "signal_idx": 3,
                    "entry_trigger": "A",
                    "trend_regime": "UP",
                    "entry_gate_status": "READY",
                    "prediction_universe": "trigger_all",
                    "label_success_20d": 0,
                    "label_net_return_pct_20d": -1.0,
                    "label_exit_reason_20d": "ATR_STOP_2X",
                },
            ]
        )
        train_aug, pred_aug = add_train_history_features(
            train,
            train.iloc[[-1]].copy(),
            "label_success_20d",
            "label_net_return_pct_20d",
            "label_exit_reason_20d",
        )

        self.assertAlmostEqual(train_aug["hist_success_rate_trigger_trend"].iloc[0], 0.5)
        self.assertGreater(train_aug["hist_success_rate_trigger_trend"].iloc[2], 0.5)
        self.assertLess(pred_aug["hist_success_rate_trigger_trend"].iloc[0], train_aug["hist_success_rate_trigger_trend"].iloc[2])
        self.assertAlmostEqual(train_aug["hist_mean_return_gate"].iloc[0], 0.0)


class TwoStageSignalTests(unittest.TestCase):
    def test_research_signal_layer_does_not_change_strict_trade_action(self):
        frame = pd.DataFrame(
            [
                {
                    "close": 401.0,
                    "sma_20": 400.0,
                    "sma_50": 370.0,
                    "sma_200": 316.0,
                    "return_20d": 0.03,
                    "drawdown_from_ath": -0.04,
                    "score_price_algo_total": 60.4,
                    "entry_trigger": "NONE",
                    "trade_action": "NO_TRADE",
                    "position_weight_if_0_5pct_account_risk": 0.06,
                    "algo_trend_up_loose": True,
                    "algo_deep_downtrend_avoid": False,
                    "algo_overextended_highvol": False,
                    "algo_event_shock_day": False,
                    "algo_vol_extreme": False,
                },
                {
                    "close": 106.0,
                    "sma_20": 104.0,
                    "sma_50": 100.0,
                    "sma_200": 90.0,
                    "return_20d": 0.04,
                    "drawdown_from_ath": -0.03,
                    "score_price_algo_total": 66.0,
                    "entry_trigger": "20D_BREAKOUT",
                    "trade_action": "NO_TRADE",
                    "position_weight_if_0_5pct_account_risk": 0.06,
                    "algo_trend_up_loose": True,
                    "algo_deep_downtrend_avoid": False,
                    "algo_overextended_highvol": False,
                    "algo_event_shock_day": False,
                    "algo_vol_extreme": False,
                },
                {
                    "close": 120.0,
                    "sma_20": 115.0,
                    "sma_50": 110.0,
                    "sma_200": 100.0,
                    "return_20d": 0.05,
                    "drawdown_from_ath": -0.02,
                    "score_price_algo_total": 80.0,
                    "entry_trigger": "60D_BREAKOUT",
                    "trade_action": "ENTRY_ALLOWED",
                    "position_weight_if_0_5pct_account_risk": 0.05,
                    "algo_trend_up_loose": True,
                    "algo_deep_downtrend_avoid": False,
                    "algo_overextended_highvol": False,
                    "algo_event_shock_day": False,
                    "algo_vol_extreme": False,
                },
            ]
        )

        out = add_two_stage_signal_layers(frame)

        self.assertEqual(out.loc[0, "trade_action"], "NO_TRADE")
        self.assertEqual(out.loc[0, "research_signal_stage"], "EARLY_BULLISH_WATCH")
        self.assertEqual(out.loc[1, "research_signal_stage"], "PAPER_BUY_SETUP")
        self.assertAlmostEqual(out.loc[1, "paper_tracking_weight"], 0.02)
        self.assertEqual(out.loc[2, "strict_signal_stage"], "STRICT_LIVE_ENTRY")
        self.assertEqual(out.loc[2, "research_signal_action"], "USE_STRICT_LIVE_SIGNAL")


class SemiconductorMomentumV2Tests(unittest.TestCase):
    def _base_row(self, **overrides):
        row = {
            "close": 120.0,
            "high": 121.0,
            "low": 116.0,
            "sma_10": 112.0,
            "sma_20": 108.0,
            "sma_50": 100.0,
            "sma_200": 90.0,
            "prev_20d_high": 110.0,
            "prev_60d_high": 115.0,
            "prev_252d_high": 125.0,
            "return_20d": 0.12,
            "return_60d": 0.25,
            "close_change_pct": 0.06,
            "open_gap_pct": 0.03,
            "open_to_close_pct": 0.03,
            "score_price_algo_total": 62.0,
            "score_momentum": 82.0,
            "score_relative_strength": 70.0,
            "dist_close_sma_50_pct": 0.20,
            "relative_return_vs_smh_20d": 0.04,
            "relative_return_vs_qqq_20d": 0.07,
            "symbol_group": "memory_storage",
            "trade_action": "NO_TRADE",
            "entry_trigger": "NONE",
            "position_weight_if_0_5pct_account_risk": 0.05,
            "atr_14": 5.0,
            "algo_trend_up_loose": True,
            "algo_trend_down_clean": False,
            "algo_vol_extreme": False,
            "algo_overextended_highvol": False,
            "algo_event_shock_day": False,
            "algo_deep_downtrend_avoid": False,
            "algo_relative_strength": True,
            "algo_pullback_to_50_bounce": False,
        }
        row.update(overrides)
        return row

    def test_high_vol_positive_raw_breakout_becomes_tiny_extension(self):
        frame = pd.DataFrame(
            [
                self._base_row(close=100.0, high=101.0, prev_20d_high=105.0, prev_60d_high=110.0, close_change_pct=0.0),
                self._base_row(algo_vol_extreme=True, algo_overextended_highvol=True, algo_event_shock_day=True),
            ]
        )
        out = add_semiconductor_momentum_v2_layers(frame)
        self.assertEqual(out.loc[1, "decision_tier"], "BREAKOUT_EXTENSION_TINY")
        self.assertEqual(out.loc[1, "suggested_action"], "HIGH_VOL_TINY_EXTENSION")
        self.assertGreater(out.loc[1, "suggested_weight"], 0.0)
        self.assertLessEqual(out.loc[1, "suggested_weight"], 0.03)

    def test_negative_shock_or_below_200d_stays_avoid(self):
        frame = pd.DataFrame(
            [
                self._base_row(close=100.0, high=101.0, close_change_pct=0.0),
                self._base_row(close=80.0, sma_200=90.0, close_change_pct=-0.06, open_to_close_pct=-0.05),
            ]
        )
        out = add_semiconductor_momentum_v2_layers(frame)
        self.assertEqual(out.loc[1, "decision_tier"], "AVOID_OR_WAIT")
        self.assertEqual(out.loc[1, "suggested_weight"], 0.0)

    def test_score_60_to_75_strong_semi_breakout_gets_aggressive_entry(self):
        frame = pd.DataFrame([self._base_row(score_price_algo_total=64.0, dist_close_sma_50_pct=0.10)])
        out = add_semiconductor_momentum_v2_layers(frame)
        self.assertEqual(out.loc[0, "decision_tier"], "AGGRESSIVE_TREND_ENTRY")
        self.assertEqual(out.loc[0, "suggested_action"], "SEMI_MOMENTUM_ENTRY")


class SystemReadinessTests(unittest.TestCase):
    def test_scorecard_prefers_live_like_backtest_when_available(self):
        ok_quality = pd.DataFrame([{"passed": True}])
        integrity = pd.DataFrame([{"passed": True, "severity": "CRITICAL"}])
        validation = pd.DataFrame([{"passed": True}])
        prediction = pd.DataFrame([{"passed": True, "severity": "CRITICAL"}])
        prediction_snapshot = {"prediction_use_status": "DECISION_SUPPORT_ALLOWED"}
        risk_snapshot = {"risk_state": "WAIT_FOR_TRIGGER"}
        stress_snapshot = {"stress_status": "LOW_STRESS_FOR_CURRENT_SIZE"}
        backtest = pd.DataFrame(
            [
                {"strategy_id": "D_DIAGNOSTIC", "strategy_group": "diagnostic", "cagr_pct": 25.0, "max_drawdown_pct": -30.0},
                {"strategy_id": "L_LIVE", "strategy_group": "live_like", "cagr_pct": 0.4, "max_drawdown_pct": -2.0},
            ]
        )
        walk_forward = pd.DataFrame(
            [
                {"strategy_id": "D_DIAGNOSTIC", "strategy_group": "diagnostic", "test_positive": True, "test_cagr_pct": 20.0},
                {"strategy_id": "L_LIVE", "strategy_group": "live_like", "test_positive": True, "test_cagr_pct": 0.3},
            ]
        )

        scorecard = build_scorecard(
            ok_quality,
            integrity,
            validation,
            prediction,
            prediction_snapshot,
            risk_snapshot,
            stress_snapshot,
            backtest,
            walk_forward,
        )

        full_period_note = scorecard.loc[scorecard["domain"].eq("full_period_backtest"), "note"].iloc[0]
        walk_forward_note = scorecard.loc[scorecard["domain"].eq("walk_forward_evidence"), "note"].iloc[0]
        self.assertIn("Best live-like strategy L_LIVE", full_period_note)
        self.assertIn("Best live-like positive test rate", walk_forward_note)

    def test_scorecard_prefers_alpha_research_when_available(self):
        ok_quality = pd.DataFrame([{"passed": True}])
        integrity = pd.DataFrame([{"passed": True, "severity": "CRITICAL"}])
        validation = pd.DataFrame([{"passed": True}])
        prediction = pd.DataFrame([{"passed": True, "severity": "CRITICAL"}])
        prediction_snapshot = {"prediction_use_status": "DECISION_SUPPORT_ALLOWED"}
        risk_snapshot = {"risk_state": "WAIT_FOR_TRIGGER"}
        stress_snapshot = {"stress_status": "LOW_STRESS_FOR_CURRENT_SIZE"}
        backtest = pd.DataFrame(
            [
                {"strategy_id": "L_LIVE", "strategy_group": "live_like", "cagr_pct": 0.4, "max_drawdown_pct": -2.0},
                {"strategy_id": "AR_ALPHA", "strategy_group": "alpha_research", "cagr_pct": 5.0, "max_drawdown_pct": -12.0},
            ]
        )
        walk_forward = pd.DataFrame(
            [
                {"strategy_id": "L_LIVE", "strategy_group": "live_like", "test_positive": True, "test_cagr_pct": 0.3},
                {"strategy_id": "AR_ALPHA", "strategy_group": "alpha_research", "test_positive": True, "test_cagr_pct": 3.2},
            ]
        )

        scorecard = build_scorecard(
            ok_quality,
            integrity,
            validation,
            prediction,
            prediction_snapshot,
            risk_snapshot,
            stress_snapshot,
            backtest,
            walk_forward,
        )

        full_period_note = scorecard.loc[scorecard["domain"].eq("full_period_backtest"), "note"].iloc[0]
        walk_forward_note = scorecard.loc[scorecard["domain"].eq("walk_forward_evidence"), "note"].iloc[0]
        self.assertIn("Best alpha-research strategy AR_ALPHA", full_period_note)
        self.assertIn("Best alpha-research positive test rate", walk_forward_note)

    def test_prediction_failure_blocks_alpha_prediction_and_live_readiness(self):
        ok_quality = pd.DataFrame([{"passed": True}])
        integrity = pd.DataFrame([{"passed": True, "severity": "CRITICAL"}])
        validation = pd.DataFrame([{"passed": True}])
        prediction = pd.DataFrame([{"passed": True, "severity": "CRITICAL"}])
        prediction_snapshot = {
            "prediction_use_status": "DISPLAY_ONLY_QUALITY_NOT_PASSED",
            "model_quality_block_reasons": "SELECTED_OOS_EVENT_COUNT_LT_50",
            "pooled_model_quality_block_reasons": "POOLED_NO_BRIER_IMPROVEMENT",
        }
        risk_snapshot = {"risk_state": "WAIT_FOR_TRIGGER"}
        stress_snapshot = {"stress_status": "LOW_STRESS_FOR_CURRENT_SIZE"}
        backtest = pd.DataFrame(
            [{"strategy_id": "L_LIVE", "strategy_group": "live_like", "cagr_pct": 0.4, "max_drawdown_pct": -2.0}]
        )
        walk_forward = pd.DataFrame(
            [{"strategy_id": "L_LIVE", "strategy_group": "live_like", "test_positive": True, "test_cagr_pct": 0.3}]
        )
        scorecard = build_scorecard(
            ok_quality,
            integrity,
            validation,
            prediction,
            prediction_snapshot,
            risk_snapshot,
            stress_snapshot,
            backtest,
            walk_forward,
        )
        latest = build_latest_state(scorecard, risk_snapshot, stress_snapshot, prediction_snapshot)
        values = dict(zip(latest["field"], latest["value"]))

        self.assertEqual(float(values["system_readiness_score"]), 100.0)
        self.assertLess(float(values["composite_gate_score"]), 100.0)
        self.assertEqual(values["research_ready"], True)
        self.assertEqual(values["alpha_ready"], False)
        self.assertEqual(values["prediction_ready"], False)
        self.assertEqual(values["live_ready"], False)
        self.assertIn("PREDICTION_NOT_DECISION_SUPPORT", str(values["alpha_block_reasons"]))
        self.assertIn("POOLED:POOLED_NO_BRIER_IMPROVEMENT", str(values["prediction_block_reasons"]))


class SystemStatePredictionBlockReasonTests(unittest.TestCase):
    def test_latest_state_exposes_performance_only_gate_fields(self):
        scorecard = pd.DataFrame(
            [
                {"domain": "operational_quality", "passed": True, "weight": 10.0, "score": 10.0, "note": "ok"},
                {"domain": "daily_integrity", "passed": True, "weight": 15.0, "score": 15.0, "note": "ok"},
                {"domain": "validation_quality", "passed": True, "weight": 15.0, "score": 15.0, "note": "ok"},
                {"domain": "prediction_decision_support", "passed": False, "weight": 15.0, "score": 0.0, "note": "latest blocked"},
                {"domain": "risk_policy", "passed": True, "weight": 10.0, "score": 10.0, "note": "ok"},
                {"domain": "stress_tolerance", "passed": True, "weight": 10.0, "score": 10.0, "note": "ok"},
                {"domain": "full_period_backtest", "passed": True, "weight": 15.0, "score": 15.0, "note": "ok"},
                {"domain": "walk_forward_evidence", "passed": True, "weight": 10.0, "score": 10.0, "note": "ok"},
            ]
        )
        latest = build_latest_state(
            scorecard,
            {"risk_state": "WAIT_FOR_TRIGGER"},
            {"stress_status": "LOW_STRESS_FOR_CURRENT_SIZE"},
            {"prediction_use_status": "DISPLAY_ONLY_RULE_FILTERED"},
            {
                "model_gate_status": "PASS_MODEL_QUALITY_SIGNAL_STANDBY",
                "performance_gate_status": "PASS_ACTIVE_MODEL_PERFORMANCE_DIAGNOSTIC_WARNINGS_ONLY",
                "active_performance_failed_gate_count": 0,
                "active_performance_blocking_failed_gate_count": 0,
                "active_metric_performance_failed_gate_count": 0,
                "metric_performance_failed_gate_count": 362,
                "performance_evidence_gap_count": 179,
                "aggregate_performance_quality_flag_count": 78,
                "diagnostic_performance_warning_gate_count": 619,
                "diagnostic_metric_performance_warning_gate_count": 362,
                "rank_policy_supported_performance_warning_count": 159,
                "rank_policy_supported_threshold_warning_count": 86,
                "rank_policy_supported_threshold_metric_warning_count": 62,
                "unresolved_diagnostic_metric_performance_warning_gate_count": 300,
                "classified_diagnostic_performance_warning_count": 319,
                "unresolved_diagnostic_performance_warning_count": 300,
                "performance_warning_resolution_status": "WARN_UNRESOLVED_DIAGNOSTIC_WARNINGS",
                "active_performance_failed_gate_groups": "PASS",
                "metric_performance_failed_families": "backtest_overfit_control|discrimination|economic_uplift|probabilistic_skill|probability_calibration|threshold_stability",
                "diagnostic_performance_failed_gate_groups": "local_prediction_diagnostic|strategy_validation_diagnostic",
                "next_required_performance_action": "PASS_ACTIVE_MODEL_PERFORMANCE",
                "performance_gate_interpretation": "ACTIVE_PERFORMANCE_PASS_DIAGNOSTIC_WARNINGS_NONBLOCKING",
                "rank_uplift_diagnostic_count": 78,
                "rank_uplift_positive_diagnostic_count": 51,
                "rank_policy_diagnostic_pass_count": 20,
                "rank_policy_best_se_lower_pct": 3.096,
            },
        )
        values = dict(zip(latest["field"], latest["value"]))

        self.assertEqual(values["model_gate_status"], "PASS_MODEL_QUALITY_SIGNAL_STANDBY")
        self.assertEqual(values["performance_gate_status"], "PASS_ACTIVE_MODEL_PERFORMANCE_DIAGNOSTIC_WARNINGS_ONLY")
        self.assertEqual(values["active_performance_failed_gate_count"], 0)
        self.assertEqual(values["active_performance_blocking_failed_gate_count"], 0)
        self.assertEqual(values["active_metric_performance_failed_gate_count"], 0)
        self.assertEqual(values["prediction_block_reasons"], "LATEST_SIGNAL:DISPLAY_ONLY_RULE_FILTERED")
        self.assertEqual(values["prediction_performance_block_reasons"], "PASS")
        self.assertEqual(values["metric_performance_failed_gate_count"], 362)
        self.assertEqual(values["performance_evidence_gap_count"], 179)
        self.assertEqual(values["aggregate_performance_quality_flag_count"], 78)
        self.assertEqual(values["diagnostic_performance_warning_gate_count"], 619)
        self.assertEqual(values["diagnostic_metric_performance_warning_gate_count"], 362)
        self.assertEqual(values["rank_policy_supported_performance_warning_count"], 159)
        self.assertEqual(values["rank_policy_supported_threshold_warning_count"], 86)
        self.assertEqual(values["rank_policy_supported_threshold_metric_warning_count"], 62)
        self.assertEqual(values["unresolved_diagnostic_metric_performance_warning_gate_count"], 300)
        self.assertEqual(values["classified_diagnostic_performance_warning_count"], 319)
        self.assertEqual(values["unresolved_diagnostic_performance_warning_count"], 300)
        self.assertEqual(values["performance_warning_resolution_status"], "WARN_UNRESOLVED_DIAGNOSTIC_WARNINGS")
        self.assertEqual(values["active_performance_failed_gate_groups"], "PASS")
        self.assertEqual(
            values["metric_performance_failed_families"],
            "backtest_overfit_control|discrimination|economic_uplift|probabilistic_skill|probability_calibration|threshold_stability",
        )
        self.assertEqual(values["diagnostic_performance_failed_gate_groups"], "local_prediction_diagnostic|strategy_validation_diagnostic")
        self.assertEqual(values["next_required_performance_action"], "PASS_ACTIVE_MODEL_PERFORMANCE")
        self.assertEqual(values["performance_gate_interpretation"], "ACTIVE_PERFORMANCE_PASS_DIAGNOSTIC_WARNINGS_NONBLOCKING")
        self.assertEqual(values["rank_uplift_diagnostic_count"], 78)
        self.assertEqual(values["rank_uplift_positive_diagnostic_count"], 51)
        self.assertEqual(values["rank_policy_diagnostic_pass_count"], 20)
        self.assertEqual(values["rank_policy_best_se_lower_pct"], 3.096)

    def test_prediction_block_reason_prefers_pooled_latest_when_pooled_quality_passes(self):
        reason = prediction_block_reason_from_snapshot(
            {
                "model_quality_block_reasons": "LOCAL_SMALL_SAMPLE_DIAGNOSTIC",
                "pooled_model_quality_pass": True,
                "pooled_model_quality_block_reasons": "PASS",
                "pooled_decision_block_reasons": "LATEST_NOT_TRADE_READY|POOLED_EXPECTED_R_LT_0_35",
            },
            prediction_ready=False,
        )

        self.assertEqual(reason, "POOLED_LATEST:LATEST_NOT_TRADE_READY|POOLED_EXPECTED_R_LT_0_35")

    def test_prediction_block_reason_uses_latest_signal_when_active_performance_passes(self):
        reason = prediction_block_reason_from_snapshot(
            {
                "prediction_use_status": "DISPLAY_ONLY_NO_ENTRY_TRIGGER",
                "prediction_signal_status": "NO_ENTRY_TRIGGER_CONTEXT_ONLY",
                "latest_signal_block_reasons": "LATEST_CONTEXT_ONLY_NO_ENTRY_TRIGGER",
                "model_quality_block_reasons": "NO_BRIER_IMPROVEMENT|ECE_GT_0_10",
            },
            prediction_ready=False,
            model_gate_snapshot={
                "performance_gate_status": "PASS_ACTIVE_MODEL_PERFORMANCE_DIAGNOSTIC_WARNINGS_ONLY",
                "active_performance_failed_gate_count": 0,
                "active_performance_blocking_failed_gate_count": 0,
            },
        )

        self.assertEqual(reason, "LATEST_SIGNAL:LATEST_CONTEXT_ONLY_NO_ENTRY_TRIGGER")

    def test_prediction_block_reason_prefers_model_gate_latest_when_active_performance_passes(self):
        reason = prediction_block_reason_from_snapshot(
            {
                "prediction_use_status": "DISPLAY_ONLY_NO_ENTRY_TRIGGER",
                "prediction_signal_status": "NO_ENTRY_TRIGGER_CONTEXT_ONLY",
                "latest_signal_block_reasons": "LATEST_CONTEXT_ONLY_NO_ENTRY_TRIGGER",
                "model_quality_block_reasons": "NO_BRIER_IMPROVEMENT",
            },
            prediction_ready=False,
            model_gate_snapshot={
                "performance_gate_status": "PASS_ACTIVE_MODEL_PERFORMANCE_DIAGNOSTIC_WARNINGS_ONLY",
                "active_performance_failed_gate_count": 0,
                "active_performance_blocking_failed_gate_count": 0,
                "pooled_latest_block_reasons": "LATEST_NOT_TRADE_READY|POOLED_STOP_RISK_GT_0_35",
            },
        )

        self.assertEqual(reason, "POOLED_LATEST:LATEST_NOT_TRADE_READY|POOLED_STOP_RISK_GT_0_35")


class ShadowPaperDecisionSchemaTests(unittest.TestCase):
    def test_shadow_accepts_current_decision_support_schema(self):
        row = {
            "latest_is_event_candidate": True,
            "decision_permission": "DECISION_SUPPORT_ONLY",
            "prediction_quality_pass": True,
            "p_success": 0.72,
            "threshold": 0.60,
            "final_trade_decision": "ALPHA_RESEARCH_LONG_ALLOWED",
        }

        self.assertEqual(paper_action(row), "PAPER_LONG_CONFIRMED")

    def test_shadow_blocks_display_only_even_when_probability_passes(self):
        row = {
            "latest_is_event_candidate": True,
            "decision_permission": "DISPLAY_ONLY",
            "prediction_quality_pass": True,
            "p_success": 0.72,
            "threshold": 0.60,
            "final_trade_decision": "ALPHA_RESEARCH_LONG_ALLOWED",
        }

        self.assertEqual(paper_action(row), "DISPLAY_ONLY_NO_PAPER_TRADE")


if __name__ == "__main__":
    unittest.main()
