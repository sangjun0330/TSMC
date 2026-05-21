import pandas as pd

from tsm_model_gate_engine import build_audit, build_root_causes, build_snapshot, evaluate_local_model_row


def _passing_local_row():
    return pd.Series(
        {
            "candidate_scope": "trade_ready_entry",
            "horizon_days": 20,
            "model_name": "score_logistic",
            "oos_event_count": 120,
            "selected_oos_event_count": 60,
            "brier_improvement_pct": 2.0,
            "pr_auc": 0.60,
            "base_rate_pr_auc": 0.50,
            "ece": 0.05,
            "selected_minus_rule_all_pct": 1.0,
            "selected_signal_expectancy_ci_lower_pct": 0.2,
            "positive_expectancy_folds": 4,
            "min_selected_events_per_fold": 12,
            "min_calibration_bin_n": 30,
            "threshold_iqr": 0.05,
            "min_decision_oos_events": 100,
            "min_selected_oos_events": 50,
            "decision_scope_eligible": True,
            "prediction_quality_pass": True,
        }
    )


def test_local_model_gate_passes_when_all_thresholds_pass():
    audit = pd.DataFrame(evaluate_local_model_row(_passing_local_row()))

    assert audit["passed"].all()


def test_local_model_gate_reports_failed_calibration():
    row = _passing_local_row()
    row["ece"] = 0.20

    audit = pd.DataFrame(evaluate_local_model_row(row))
    failed = audit.loc[~audit["passed"]]

    assert "decision_ece_within_limit" in set(failed["gate"])
    assert "ECE_GT_LIMIT" in set(failed["block_reason"])


def test_local_tsm_28_oos_row_stays_display_only():
    row = _passing_local_row()
    row["oos_event_count"] = 28
    row["selected_oos_event_count"] = 28
    row["prediction_quality_pass"] = False

    audit = pd.DataFrame(evaluate_local_model_row(row))
    failed_reasons = set(audit.loc[~audit["passed"], "block_reason"])

    assert "OOS_EVENT_COUNT_LT_MIN" in failed_reasons
    assert "SELECTED_OOS_EVENT_COUNT_LT_MIN" in failed_reasons
    assert "PREDICTION_QUALITY_FALSE" in failed_reasons


def test_model_gate_snapshot_passes_when_latest_local_support_is_allowed():
    comparison = pd.DataFrame([_passing_local_row()])
    audit = build_audit(
        comparison,
        {"prediction_use_status": "DECISION_SUPPORT_ALLOWED"},
        pd.DataFrame(),
        {},
        pd.DataFrame(),
    )
    snapshot = build_snapshot(audit, {"prediction_use_status": "DECISION_SUPPORT_ALLOWED"}, {})
    values = dict(zip(snapshot["field"], snapshot["value"]))

    assert values["model_gate_status"] == "PASS"
    assert bool(values["local_prediction_decision_support"]) is True


def test_model_gate_root_causes_group_failed_reasons_by_priority():
    row = _passing_local_row()
    row["ece"] = 0.20
    row["min_calibration_bin_n"] = 3
    audit = pd.DataFrame(evaluate_local_model_row(row))

    root_causes = build_root_causes(audit)
    reasons = set(root_causes["root_cause"])

    assert "ECE_GT_LIMIT" in reasons
    assert "CALIBRATION_MIN_BIN_N_LT_MIN" in reasons
    assert int(root_causes.loc[root_causes["root_cause"].eq("CALIBRATION_MIN_BIN_N_LT_MIN"), "priority"].iloc[0]) == 1
    assert root_causes.loc[root_causes["root_cause"].eq("CALIBRATION_MIN_BIN_N_LT_MIN"), "category"].iloc[0] == "system_calibration"


def test_model_quality_pass_can_coexist_with_latest_blocked():
    pooled_comparison = pd.DataFrame(
        [
            {
                "model_name": "pooled_stack_calibrated",
                "split": "combined_test_holdout",
                "event_count": 200,
                "selected_event_count": 80,
                "selected_fraction": 0.40,
                "brier_improvement_pct": 1.0,
                "ece": 0.05,
                "decision_ece": 0.05,
                "decision_min_calibration_bin_n": 40,
                "selected_minus_all_pct": 0.5,
                "selected_minus_all_ci_lower_pct": 0.2,
                "selected_minus_score_baseline_ci_lower_pct": 0.1,
                "selected_expectancy_ci_lower_pct": 0.2,
                "threshold_decision_eligible": True,
                "threshold_stability_pass": True,
                "min_selected_events_per_fold": 12,
                "positive_expectancy_fold_count": 4,
                "trial_count": 64,
                "validation_design": "walk_forward_oof",
                "is_champion": True,
            }
        ]
    )
    tsm_calibration = pd.DataFrame(
        [
            {
                "model_name": "tsm_specific_calibrated_layer",
                "split": "tsm_combined_test_holdout",
                "event_count": 40,
                "ece": 0.10,
            }
        ]
    )
    audit = build_audit(
        pd.DataFrame(),
        {"prediction_use_status": "DISPLAY_ONLY_NO_MODEL_CANDIDATE"},
        pooled_comparison,
        {
            "latest_signal_pass": False,
            "decision_support_allowed": False,
            "latest_block_reasons": "LATEST_NOT_TRADE_READY|POOLED_STOP_RISK_GT_0_35",
            "decision_block_reasons": "LATEST_NOT_TRADE_READY|POOLED_STOP_RISK_GT_0_35",
            "p_stop_hit_20d": 0.50,
            "expected_r_net_20d": 0.20,
        },
        tsm_calibration,
    )
    snapshot = build_snapshot(audit, {"prediction_use_status": "DISPLAY_ONLY_NO_MODEL_CANDIDATE"}, {"latest_signal_pass": False})
    values = dict(zip(snapshot["field"], snapshot["value"]))

    assert values["pooled_model_quality_pass"] is True
    assert values["pooled_latest_signal_pass"] is False
    assert values["pooled_prediction_decision_support"] is False
    assert values["model_gate_status"] == "MODEL_QUALITY_PASS_LATEST_BLOCKED"


def test_pooled_model_quality_fails_when_strict_threshold_is_diagnostic_only():
    pooled_comparison = pd.DataFrame(
        [
            {
                "model_name": "pooled_empirical_bayes_group_rate",
                "split": "combined_test_holdout",
                "evaluation_scope": "trade_ready_entry_only",
                "validation_design": "walk_forward_oof",
                "event_count": 200,
                "selected_event_count": 80,
                "selected_fraction": 0.40,
                "brier_improvement_pct": 1.0,
                "ece": 0.05,
                "decision_ece": 0.05,
                "decision_min_calibration_bin_n": 40,
                "selected_minus_all_pct": 0.5,
                "selected_minus_all_ci_lower_pct": 0.2,
                "selected_minus_score_baseline_ci_lower_pct": 0.1,
                "selected_expectancy_ci_lower_pct": 0.2,
                "threshold_decision_eligible": False,
                "threshold_stability_pass": True,
                "min_selected_events_per_fold": 12,
                "positive_expectancy_fold_count": 4,
                "trial_count": 64,
                "is_champion": True,
            }
        ]
    )
    tsm_calibration = pd.DataFrame(
        [
            {
                "model_name": "tsm_specific_calibrated_layer",
                "split": "tsm_combined_test_holdout",
                "event_count": 40,
                "ece": 0.10,
            }
        ]
    )
    audit = build_audit(
        pd.DataFrame(),
        {"prediction_use_status": "DISPLAY_ONLY_NO_MODEL_CANDIDATE"},
        pooled_comparison,
        {"latest_signal_pass": False, "decision_support_allowed": False},
        tsm_calibration,
    )
    snapshot = build_snapshot(audit, {"prediction_use_status": "DISPLAY_ONLY_NO_MODEL_CANDIDATE"}, {"latest_signal_pass": False})
    values = dict(zip(snapshot["field"], snapshot["value"]))

    assert values["pooled_model_quality_pass"] is False
    assert "POOLED_THRESHOLD_NOT_DECISION_ELIGIBLE" in values["pooled_model_quality_block_reasons"]


def test_pooled_model_quality_fails_when_oof_fold_selection_is_sparse():
    pooled_comparison = pd.DataFrame(
        [
            {
                "model_name": "pooled_empirical_bayes_group_rate",
                "split": "combined_test_holdout",
                "evaluation_scope": "trade_ready_entry_only",
                "validation_design": "walk_forward_oof",
                "event_count": 200,
                "selected_event_count": 80,
                "selected_fraction": 0.40,
                "brier_improvement_pct": 1.0,
                "ece": 0.05,
                "decision_ece": 0.05,
                "decision_min_calibration_bin_n": 40,
                "selected_minus_all_pct": 0.5,
                "selected_minus_all_ci_lower_pct": 0.2,
                "selected_minus_score_baseline_ci_lower_pct": 0.1,
                "selected_expectancy_ci_lower_pct": 0.2,
                "threshold_decision_eligible": True,
                "threshold_stability_pass": False,
                "weak_oof_folds": "oof_test_2024:selected_lt_10",
                "positive_expectancy_fold_count": 3,
                "trial_count": 64,
                "is_champion": True,
            },
            {
                "model_name": "pooled_empirical_bayes_group_rate",
                "split": "oof_test_2024",
                "evaluation_scope": "trade_ready_entry_only",
                "validation_design": "walk_forward_oof",
                "event_count": 50,
                "selected_event_count": 4,
                "threshold_decision_eligible": True,
            },
        ]
    )
    tsm_calibration = pd.DataFrame(
        [
            {
                "model_name": "tsm_specific_calibrated_layer",
                "split": "tsm_combined_test_holdout",
                "event_count": 40,
                "ece": 0.10,
            }
        ]
    )
    audit = build_audit(
        pd.DataFrame(),
        {"prediction_use_status": "DISPLAY_ONLY_NO_MODEL_CANDIDATE"},
        pooled_comparison,
        {"latest_signal_pass": False, "decision_support_allowed": False},
        tsm_calibration,
    )
    snapshot = build_snapshot(audit, {"prediction_use_status": "DISPLAY_ONLY_NO_MODEL_CANDIDATE"}, {"latest_signal_pass": False})
    values = dict(zip(snapshot["field"], snapshot["value"]))

    assert values["pooled_model_quality_pass"] is False
    assert "POOLED_OOF_FOLD_SELECTED_EVENTS_LT_10" in values["pooled_model_quality_block_reasons"]


def test_pooled_quality_fails_when_uplift_primary_gate_fails():
    pooled_comparison = pd.DataFrame(
        [
            {
                "model_name": "pooled_stack_calibrated",
                "split": "combined_test_holdout",
                "evaluation_scope": "trade_ready_entry_only",
                "validation_design": "walk_forward_oof",
                "event_count": 220,
                "selected_event_count": 90,
                "selected_fraction": 0.41,
                "brier_improvement_pct": 1.2,
                "ece": 0.05,
                "decision_ece": 0.05,
                "decision_min_calibration_bin_n": 40,
                "selected_minus_all_pct": 0.8,
                "selected_minus_all_ci_lower_pct": 0.2,
                "selected_minus_score_baseline_ci_lower_pct": 0.2,
                "selected_expectancy_ci_lower_pct": 0.3,
                "threshold_decision_eligible": True,
                "threshold_stability_pass": True,
                "min_selected_events_per_fold": 12,
                "positive_expectancy_fold_count": 4,
                "trial_count": 64,
                "uplift_pass": False,
                "uplift_failure_reasons": "FOLD_LOWER_BOUND_NOT_REPEATABLE",
                "is_champion": True,
            }
        ]
    )
    tsm_calibration = pd.DataFrame(
        [
            {
                "model_name": "tsm_specific_pooled_only",
                "split": "tsm_combined_test_holdout",
                "event_count": 40,
                "ece": 0.10,
                "decision_ece": 0.10,
                "tsm_calibration_route_pass": True,
                "is_selected_tsm_calibration_route": True,
            }
        ]
    )

    audit = build_audit(
        pd.DataFrame(),
        {"prediction_use_status": "DISPLAY_ONLY_NO_MODEL_CANDIDATE"},
        pooled_comparison,
        {"latest_signal_pass": False, "decision_support_allowed": False},
        tsm_calibration,
    )
    snapshot = build_snapshot(audit, {"prediction_use_status": "DISPLAY_ONLY_NO_MODEL_CANDIDATE"}, {"latest_signal_pass": False})
    values = dict(zip(snapshot["field"], snapshot["value"]))

    assert values["pooled_model_quality_pass"] is False
    assert values["pooled_economic_uplift_pass"] is False
    assert "POOLED_UPLIFT_NOT_PASSED" in values["pooled_model_quality_block_reasons"]


def test_tsm_calibration_route_failure_is_pooled_system_quality_not_latest_timing():
    pooled_comparison = pd.DataFrame(
        [
            {
                "model_name": "pooled_stack_calibrated",
                "split": "combined_test_holdout",
                "evaluation_scope": "trade_ready_entry_only",
                "validation_design": "walk_forward_oof",
                "event_count": 220,
                "selected_event_count": 90,
                "selected_fraction": 0.41,
                "brier_improvement_pct": 1.2,
                "ece": 0.05,
                "decision_ece": 0.05,
                "decision_min_calibration_bin_n": 40,
                "selected_minus_all_pct": 0.8,
                "selected_minus_all_ci_lower_pct": 0.2,
                "selected_minus_score_baseline_ci_lower_pct": 0.2,
                "selected_expectancy_ci_lower_pct": 0.3,
                "threshold_decision_eligible": True,
                "threshold_stability_pass": True,
                "min_selected_events_per_fold": 12,
                "positive_expectancy_fold_count": 4,
                "trial_count": 64,
                "uplift_pass": True,
                "is_champion": True,
            }
        ]
    )
    tsm_calibration = pd.DataFrame(
        [
            {
                "model_name": "tsm_specific_logit_shift",
                "split": "tsm_combined_test_holdout",
                "event_count": 40,
                "ece": 0.14,
                "decision_ece": 0.14,
                "tsm_calibration_route_pass": False,
                "tsm_calibration_route_failure_reasons": "TSM_TEST_OR_HOLDOUT_ECE_GT_0_20",
                "is_selected_tsm_calibration_route": True,
            }
        ]
    )

    audit = build_audit(
        pd.DataFrame(),
        {"prediction_use_status": "DISPLAY_ONLY_NO_MODEL_CANDIDATE"},
        pooled_comparison,
        {
            "latest_signal_pass": False,
            "decision_support_allowed": False,
            "latest_block_reasons": "LATEST_NOT_TRADE_READY",
        },
        tsm_calibration,
    )
    snapshot = build_snapshot(audit, {"prediction_use_status": "DISPLAY_ONLY_NO_MODEL_CANDIDATE"}, {"latest_signal_pass": False})
    values = dict(zip(snapshot["field"], snapshot["value"]))

    assert values["pooled_tsm_calibration_pass"] is False
    assert "TSM_CALIBRATION_ROUTE_NOT_PASSED" in values["pooled_model_quality_block_reasons"]
    assert "LATEST_NOT_TRADE_READY" in values["pooled_latest_block_reasons"]
    assert "TSM_CALIBRATION_ROUTE_NOT_PASSED" not in values["pooled_latest_block_reasons"]
