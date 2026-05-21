import unittest

import pandas as pd

from tsm_prediction_engine import add_train_history_features
from tsm_price_rule_engine import add_two_stage_signal_layers
from tsm_shadow_paper_engine import paper_action
from tsm_system_state_engine import build_latest_state, build_scorecard


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
