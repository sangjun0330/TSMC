import pandas as pd

from tsm_model_gate_engine import (
    build_audit,
    build_latest_prediction_blocker_audit,
    build_model_candidate_disposition_summary,
    build_performance_gate_audit,
    build_performance_warning_resolution_summary,
    build_rank_policy_promotion_watchlist,
    build_root_causes,
    build_snapshot,
    build_unresolved_performance_priorities,
    evaluate_local_model_row,
    evaluate_local_models,
    evaluate_research_validation,
    write_report,
)


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


def test_local_model_gate_adds_rank_uplift_info_without_failures():
    row = _passing_local_row()
    row["rank_top_quintile_count"] = 24
    row["rank_top_quintile_minus_all_pct"] = 3.5
    row["rank_top_quintile_return_pct"] = 5.0
    row["rank_top_quintile_success_rate"] = 0.62

    audit = pd.DataFrame(evaluate_local_model_row(row))
    rank_rows = audit[audit["gate"].eq("rank_top_quintile_uplift_diagnostic")]

    assert audit["passed"].all()
    assert len(rank_rows) == 1
    rank_row = rank_rows.iloc[0]
    assert rank_row["gate_group"] == "local_prediction_rank_diagnostic"
    assert rank_row["severity"] == "INFO"
    assert rank_row["block_reason"] == "RANK_TOP_QUINTILE_UPLIFT_POSITIVE"
    assert rank_row["rank_top_quintile_count"] == 24


def test_local_model_gate_marks_rank_policy_pass_as_info_evidence():
    row = _passing_local_row()
    row["rank_top_quintile_count"] = 80
    row["rank_top_quintile_minus_all_pct"] = 2.4
    row["rank_top_quintile_return_pct"] = 4.0
    row["rank_top_quintile_success_rate"] = 0.62
    row["rank_top_quintile_fold_count"] = 4
    row["rank_top_quintile_positive_folds"] = 4
    row["rank_top_quintile_se_lower_pct"] = 1.1
    row["rank_policy_diagnostic_pass"] = True

    audit = pd.DataFrame(evaluate_local_model_row(row))
    rank_row = audit[audit["gate"].eq("rank_top_quintile_uplift_diagnostic")].iloc[0]

    assert audit["passed"].all()
    assert rank_row["severity"] == "INFO"
    assert rank_row["block_reason"] == "RANK_TOP_QUINTILE_POLICY_PASS"
    assert bool(rank_row["rank_policy_diagnostic_pass"]) is True


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
    assert set(audit.loc[~audit["passed"], "severity"]) == {"CRITICAL"}


def test_non_decision_local_rows_are_diagnostic_warnings():
    decision_row = _passing_local_row()
    diagnostic_row = _passing_local_row()
    diagnostic_row["candidate_scope"] = "entry_research"
    diagnostic_row["decision_scope_eligible"] = False
    diagnostic_row["prediction_quality_pass"] = False
    diagnostic_row["brier_improvement_pct"] = -1.0

    audit = pd.DataFrame(
        evaluate_local_models(
            pd.DataFrame([decision_row, diagnostic_row]),
            {"prediction_use_status": "DECISION_SUPPORT_ALLOWED"},
        )
    )
    diagnostic_failures = audit[
        audit["gate_group"].eq("local_prediction_diagnostic")
        & ~audit["passed"]
    ]

    assert not diagnostic_failures.empty
    assert set(diagnostic_failures["severity"]) == {"WARN"}
    assert audit.loc[audit["gate_group"].eq("local_prediction"), "passed"].all()


def test_display_only_local_decision_rows_are_diagnostic_warnings():
    row = _passing_local_row()
    row["oos_event_count"] = 28
    row["selected_oos_event_count"] = 28
    row["prediction_quality_pass"] = False

    audit = pd.DataFrame(
        evaluate_local_models(
            pd.DataFrame([row]),
            {
                "prediction_use_status": "DISPLAY_ONLY_RULE_FILTERED",
                "model_quality_block_reasons": "LATEST_RULE_FILTERED_NOT_TRADE_READY",
            },
        )
    )
    failed = audit.loc[~audit["passed"]]

    assert "local_prediction" not in set(failed["gate_group"])
    assert "local_prediction_diagnostic" in set(failed["gate_group"])
    assert "local_latest" in set(failed["gate_group"])
    assert set(failed["severity"]) == {"WARN"}


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
    assert values["model_gate_status"] == "PASS_MODEL_QUALITY_SIGNAL_STANDBY"


def test_pooled_model_quality_uses_20d_decision_horizon_when_other_horizons_fail():
    pooled_comparison = pd.DataFrame(
        [
            {
                "horizon_days": 5,
                "model_name": "pooled_lgbm_classifier",
                "split": "combined_test_holdout",
                "evaluation_scope": "trade_ready_entry_only",
                "validation_design": "walk_forward_oof",
                "event_count": 2385,
                "selected_event_count": 383,
                "selected_fraction": 0.16,
                "brier_improvement_pct": -1.7,
                "decision_ece": 0.073,
                "decision_min_calibration_bin_n": 40,
                "selected_minus_all_pct": 1.1,
                "selected_minus_all_ci_lower_pct": 0.2,
                "selected_minus_score_baseline_ci_lower_pct": 0.2,
                "selected_expectancy_ci_lower_pct": 0.3,
                "threshold_decision_eligible": False,
                "threshold_stability_pass": False,
                "min_selected_events_per_fold": 25,
                "positive_expectancy_fold_count": 4,
                "trial_count": 64,
                "is_champion": True,
            },
            {
                "horizon_days": 20,
                "model_name": "pooled_xgb_classifier",
                "split": "combined_test_holdout",
                "evaluation_scope": "trade_ready_entry_only",
                "validation_design": "walk_forward_oof",
                "event_count": 2150,
                "selected_event_count": 768,
                "selected_fraction": 0.357,
                "brier_improvement_pct": 5.28,
                "decision_ece": 0.055,
                "decision_min_calibration_bin_n": 40,
                "selected_minus_all_pct": 2.75,
                "selected_minus_all_ci_lower_pct_paired": 0.72,
                "selected_minus_score_baseline_ci_lower_pct_paired": 0.72,
                "selected_expectancy_ci_lower_pct": 1.0,
                "threshold_decision_eligible": True,
                "threshold_stability_pass": True,
                "min_selected_events_per_fold": 25,
                "positive_expectancy_fold_count": 4,
                "trial_count": 64,
                "is_champion": True,
                "uplift_pass": True,
            },
        ]
    )
    tsm_calibration = pd.DataFrame(
        [
            {
                "model_name": "tsm_specific_direct",
                "split": "tsm_combined_test_holdout",
                "event_count": 41,
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

    assert values["pooled_model_quality_pass"] is True
    assert values["pooled_threshold_stability_pass"] is True
    assert values["pooled_model_quality_block_reasons"] == "PASS"
    assert values["model_gate_status"] == "PASS_MODEL_QUALITY_SIGNAL_STANDBY"


def test_latest_prediction_blocker_audit_reports_threshold_gaps():
    audit = build_latest_prediction_blocker_audit(
        {
            "prediction_use_status": "DISPLAY_ONLY_NO_ENTRY_TRIGGER",
            "prediction_signal_status": "NO_ENTRY_TRIGGER_CONTEXT_ONLY",
            "latest_signal_block_reasons": "LATEST_CONTEXT_ONLY_NO_ENTRY_TRIGGER",
        },
        {
            "decision_support_allowed": False,
            "latest_signal_pass": False,
            "latest_trade_ready": False,
            "decision_score_20d": -0.05,
            "threshold_20d": 0.12,
            "p_stop_hit_20d": 0.565,
            "expected_r_net_20d": -0.25,
            "paper_decision_support_allowed": False,
            "paper_model_gate_pass": True,
            "paper_oof_selection_evidence_pass": True,
            "paper_gate_block_reasons": "LATEST_NOT_TRADE_READY|POOLED_STOP_RISK_GT_0_40",
            "latest_block_reasons": "LATEST_NOT_TRADE_READY|POOLED_DECISION_SCORE_BELOW_THRESHOLD|POOLED_STOP_RISK_GT_0_35|POOLED_EXPECTED_R_LT_0_35",
        },
    )
    rows = {row["check"]: row for _, row in audit.iterrows()}

    assert bool(rows["latest_trade_ready"]["passed"]) is False
    assert abs(rows["decision_score_at_or_above_threshold"]["gap_to_pass"] - 0.17) < 1e-12
    assert abs(rows["stop_risk_within_strict_limit"]["gap_to_pass"] - 0.215) < 1e-12
    assert abs(rows["expected_r_at_or_above_min"]["gap_to_pass"] - 0.6) < 1e-12
    assert bool(rows["paper_model_gate_pass"]["passed"]) is True
    assert bool(rows["paper_oof_selection_evidence_pass"]["passed"]) is True


def test_latest_no_trade_is_warning_when_pooled_model_quality_passes():
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
    pooled_latest = {
        "model_quality_pass": True,
        "latest_signal_pass": False,
        "decision_support_allowed": False,
        "latest_block_reasons": "LATEST_NOT_TRADE_READY|POOLED_STOP_RISK_GT_0_35",
        "p_stop_hit_20d": 0.50,
        "expected_r_net_20d": 0.20,
    }

    audit = build_audit(
        pd.DataFrame(),
        {"prediction_use_status": "DISPLAY_ONLY_NO_MODEL_CANDIDATE"},
        pooled_comparison,
        pooled_latest,
        tsm_calibration,
    )
    snapshot = build_snapshot(audit, {"prediction_use_status": "DISPLAY_ONLY_NO_MODEL_CANDIDATE"}, pooled_latest)
    failed = audit.loc[~audit["passed"]]
    values = dict(zip(snapshot["field"], snapshot["value"]))

    assert set(failed.loc[failed["gate_group"].eq("pooled_latest"), "severity"]) == {"WARN"}
    assert values["critical_failed_gate_count"] == 0
    assert values["blocking_failed_gate_count"] == 0
    assert values["critical_failed_gate_groups"] == "PASS"
    assert values["blocking_failed_gate_groups"] == "PASS"
    assert values["warning_failed_gate_count"] > 0
    assert "pooled_latest" in values["warning_failed_gate_groups"]
    assert values["model_quality_failed_gate_count"] == 0
    assert values["model_quality_warning_gate_count"] == 0
    assert values["performance_gate_status"] == "PASS_ACTIVE_MODEL_PERFORMANCE_DIAGNOSTIC_WARNINGS_ONLY"
    assert values["performance_blocking_failed_gate_count"] == 0
    assert values["active_performance_failed_gate_count"] == 0
    assert values["next_required_performance_action"] == "PASS_ACTIVE_MODEL_PERFORMANCE"
    assert values["performance_gate_interpretation"] == "ACTIVE_PERFORMANCE_PASS_DIAGNOSTIC_WARNINGS_NONBLOCKING"
    assert values["model_gate_status"] == "PASS_MODEL_QUALITY_SIGNAL_STANDBY"


def test_latest_no_trade_stays_critical_when_pooled_model_quality_fails():
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
                "threshold_decision_eligible": False,
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
        {
            "model_quality_pass": False,
            "latest_signal_pass": False,
            "decision_support_allowed": False,
            "latest_block_reasons": "LATEST_NOT_TRADE_READY",
            "p_stop_hit_20d": 0.50,
            "expected_r_net_20d": 0.20,
        },
        tsm_calibration,
    )
    failed = audit.loc[~audit["passed"]]

    assert set(failed.loc[failed["gate_group"].eq("pooled_latest"), "severity"]) == {"CRITICAL"}


def test_report_summarizes_diagnostic_warnings_separately_from_blockers(tmp_path):
    audit = pd.DataFrame(
        [
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "context_all",
                "horizon_days": 20,
                "model_name": "base_rate",
                "split": "",
                "gate": "prediction_quality_pass",
                "passed": False,
                "severity": "WARN",
                "value": False,
                "threshold": "True",
                "block_reason": "PREDICTION_QUALITY_FALSE",
            },
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "context_all",
                "horizon_days": 20,
                "model_name": "base_rate",
                "split": "",
                "gate": "prediction_quality_pass",
                "passed": False,
                "severity": "WARN",
                "value": False,
                "threshold": "True",
                "block_reason": "PREDICTION_QUALITY_FALSE",
            },
        ]
    )
    snapshot = pd.DataFrame(
        [
            {"field": "model_gate_status", "value": "PASS_MODEL_QUALITY_SIGNAL_STANDBY"},
            {"field": "blocking_failed_gate_count", "value": 0},
            {"field": "warning_failed_gate_count", "value": 2},
            {"field": "blocking_failed_gate_groups", "value": "PASS"},
            {"field": "warning_failed_gate_groups", "value": "local_prediction_diagnostic"},
        ]
    )
    root_causes = build_root_causes(audit)

    write_report(tmp_path, audit, snapshot, root_causes)
    report = (tmp_path / "tsm_model_gate_report.md").read_text(encoding="utf-8")
    blocking_section = report.split("## Diagnostic Warning Summary", 1)[0]

    assert "## Blocking Failed Gates" in report
    assert "| PASS | NA | NA | NA | NA | NA | NA | NA | NA | PASS |" in blocking_section
    assert "## Performance-Only Gate Summary" in report
    assert "| local_prediction_diagnostic | WARN | prediction_quality_pass | 2 | PREDICTION_QUALITY_FALSE |" in report
    assert "## Diagnostic Warning Summary" in report
    assert "| local_prediction_diagnostic | prediction_quality_pass | 2 | PREDICTION_QUALITY_FALSE |" in report


def test_report_includes_prediction_performance_gap_summary(tmp_path):
    audit = pd.DataFrame(
        [
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 20,
                "model_name": "elastic_net_logistic",
                "split": "",
                "gate": "brier_improvement_positive",
                "passed": False,
                "severity": "WARN",
                "value": -2.4,
                "threshold": ">0",
                "block_reason": "NO_BRIER_IMPROVEMENT",
            }
        ]
    )
    snapshot = pd.DataFrame(
        [
            {"field": "model_gate_status", "value": "PASS_MODEL_QUALITY_SIGNAL_STANDBY"},
            {"field": "blocking_failed_gate_count", "value": 0},
            {"field": "warning_failed_gate_count", "value": 1},
            {"field": "blocking_failed_gate_groups", "value": "PASS"},
            {"field": "warning_failed_gate_groups", "value": "local_prediction_diagnostic"},
        ]
    )
    root_causes = build_root_causes(audit)
    gap_summary = pd.DataFrame(
        [
            {
                "performance_block_reason": "NO_BRIER_IMPROVEMENT",
                "candidate_count": 2,
                "one_block_candidate_count": 1,
                "recommended_action": "improve_validation_regularized_probability_scale",
                "top_candidate_scope": "entry_research",
                "top_candidate_horizon_days": 20,
                "top_candidate_model_name": "elastic_net_logistic",
                "median_brier_gap_pct": 2.4,
                "max_ece_gap": 0.0,
                "max_selected_ci_gap_pct": 0.0,
            }
        ]
    )

    write_report(tmp_path, audit, snapshot, root_causes, gap_summary)
    report = (tmp_path / "tsm_model_gate_report.md").read_text(encoding="utf-8")

    assert "### Prediction Performance Gap Summary" in report
    assert "| NO_BRIER_IMPROVEMENT | 2 | 1 | improve_validation_regularized_probability_scale | entry_research/20d/elastic_net_logistic | 2.40% | 0.0000 | 0.00% |" in report


def test_report_summarizes_rank_uplift_info_rows(tmp_path):
    audit = pd.DataFrame(
        [
            {
                "gate_group": "local_prediction_rank_diagnostic",
                "candidate_scope": "trigger_all",
                "horizon_days": 120,
                "model_name": "empirical_bayes_group_rate",
                "split": "",
                "gate": "rank_top_quintile_uplift_diagnostic",
                "passed": True,
                "severity": "INFO",
                "value": 17.8139,
                "threshold": "diagnostic_only_top_20pct_minus_all_pct",
                "block_reason": "RANK_TOP_QUINTILE_POLICY_PASS",
                "rank_top_quintile_count": 44,
                "rank_top_quintile_return_pct": 24.0,
                "rank_top_quintile_success_rate": 0.57,
                "rank_top_quintile_fold_count": 4,
                "rank_top_quintile_positive_folds": 4,
                "rank_top_quintile_se_lower_pct": 2.1,
                "rank_policy_diagnostic_pass": True,
            }
        ]
    )
    snapshot = pd.DataFrame(
        [
            {"field": "model_gate_status", "value": "PASS_MODEL_QUALITY_SIGNAL_STANDBY"},
            {"field": "blocking_failed_gate_count", "value": 0},
            {"field": "warning_failed_gate_count", "value": 0},
            {"field": "blocking_failed_gate_groups", "value": "PASS"},
            {"field": "warning_failed_gate_groups", "value": "PASS"},
        ]
    )
    root_causes = build_root_causes(audit)

    write_report(tmp_path, audit, snapshot, root_causes)
    report = (tmp_path / "tsm_model_gate_report.md").read_text(encoding="utf-8")

    assert "## Ranking Uplift Diagnostics" in report
    assert "- Positive top-20% minus all: 1" in report
    assert "- Robust rank-policy pass rows: 1" in report
    assert "| trigger_all | 120 | empirical_bayes_group_rate | 44 | 17.81% | 2.10% | 4 | True | RANK_TOP_QUINTILE_POLICY_PASS |" in report


def test_model_gate_snapshot_counts_rank_policy_diagnostics():
    audit = pd.DataFrame(
        [
            {
                "gate_group": "local_prediction_rank_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 120,
                "model_name": "score_logistic",
                "split": "",
                "gate": "rank_top_quintile_uplift_diagnostic",
                "passed": True,
                "severity": "INFO",
                "value": 6.2,
                "threshold": "diagnostic_only_top_20pct_minus_all_pct",
                "block_reason": "RANK_TOP_QUINTILE_POLICY_PASS",
                "rank_top_quintile_count": 86,
                "rank_top_quintile_se_lower_pct": 2.1,
                "rank_policy_diagnostic_pass": True,
            },
            {
                "gate_group": "local_prediction_rank_diagnostic",
                "candidate_scope": "trigger_all",
                "horizon_days": 120,
                "model_name": "base_rate",
                "split": "",
                "gate": "rank_top_quintile_uplift_diagnostic",
                "passed": True,
                "severity": "INFO",
                "value": -0.2,
                "threshold": "diagnostic_only_top_20pct_minus_all_pct",
                "block_reason": "RANK_TOP_QUINTILE_UPLIFT_NEGATIVE",
                "rank_top_quintile_count": 44,
                "rank_top_quintile_se_lower_pct": -0.2,
                "rank_policy_diagnostic_pass": False,
            },
        ]
    )

    snapshot = build_snapshot(audit, {}, {})
    values = dict(zip(snapshot["field"], snapshot["value"]))

    assert values["rank_uplift_diagnostic_count"] == 2
    assert values["rank_uplift_positive_diagnostic_count"] == 1
    assert values["rank_policy_diagnostic_pass_count"] == 1
    assert values["rank_policy_best_se_lower_pct"] == 2.1


def test_performance_gate_audit_excludes_latest_and_marks_active_scope():
    audit = pd.DataFrame(
        [
            {
                "gate_group": "pooled_latest",
                "candidate_scope": "",
                "horizon_days": "",
                "model_name": "",
                "split": "",
                "gate": "latest_signal_pass",
                "passed": False,
                "severity": "WARN",
                "value": False,
                "threshold": "True",
                "block_reason": "LATEST_NOT_TRADE_READY",
            },
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 20,
                "model_name": "score_logistic",
                "split": "",
                "gate": "brier_improvement_positive",
                "passed": False,
                "severity": "WARN",
                "value": -1.0,
                "threshold": ">0",
                "block_reason": "NO_BRIER_IMPROVEMENT",
            },
            {
                "gate_group": "pooled_model",
                "candidate_scope": "pooled_trade_ready_entry",
                "horizon_days": 20,
                "model_name": "pooled_candidate",
                "split": "combined_test_holdout",
                "gate": "brier_improvement_positive",
                "passed": False,
                "severity": "CRITICAL",
                "value": -0.1,
                "threshold": ">0",
                "block_reason": "NO_BRIER_IMPROVEMENT",
            },
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "context_all",
                "horizon_days": 20,
                "model_name": "base_rate",
                "split": "",
                "gate": "decision_scope_eligible",
                "passed": False,
                "severity": "WARN",
                "value": False,
                "threshold": "True",
                "block_reason": "NOT_20D_TRADE_READY_DECISION_SCOPE",
            },
        ]
    )

    performance = build_performance_gate_audit(audit)

    assert set(performance["gate_group"]) == {"local_prediction_diagnostic", "pooled_model"}
    active = performance[performance["gate_group"].eq("pooled_model")].iloc[0]
    diagnostic = performance[performance["gate_group"].eq("local_prediction_diagnostic")].iloc[0]
    assert active["performance_gate_scope"] == "active"
    assert active["performance_scope_detail"] == "active_pooled_model_quality_gate"
    assert active["performance_failure_family"] == "probabilistic_skill"
    assert active["performance_failure_kind"] == "metric_shortfall"
    assert bool(active["active_performance_gate"]) is True
    assert bool(active["performance_blocks_model_gate"]) is True
    assert active["performance_diagnostic_reason"] == "ACTIVE_MODEL_PERFORMANCE_GATE"
    assert diagnostic["performance_gate_scope"] == "diagnostic"
    assert diagnostic["performance_scope_detail"] == "inactive_local_prediction_diagnostic"
    assert diagnostic["performance_failure_family"] == "probabilistic_skill"
    assert diagnostic["performance_failure_kind"] == "metric_shortfall"
    assert bool(diagnostic["excluded_from_active_performance_gate"]) is True
    assert bool(diagnostic["performance_blocks_model_gate"]) is False
    assert diagnostic["performance_diagnostic_reason"] == "INACTIVE_OR_DISPLAY_ONLY_LOCAL_MODEL_CANDIDATE"


def test_performance_gate_audit_marks_rank_policy_supported_threshold_warnings():
    audit = pd.DataFrame(
        [
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 120,
                "model_name": "score_logistic",
                "split": "",
                "gate": "selected_expectancy_ci_lower_positive",
                "passed": False,
                "severity": "WARN",
                "value": -0.1,
                "threshold": ">0",
                "block_reason": "SELECTED_EXPECTANCY_CI_LOWER_LE_0",
            },
            {
                "gate_group": "local_prediction_rank_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 120,
                "model_name": "score_logistic",
                "split": "",
                "gate": "rank_top_quintile_uplift_diagnostic",
                "passed": True,
                "severity": "INFO",
                "value": 6.2,
                "threshold": "diagnostic_only_top_20pct_minus_all_pct",
                "block_reason": "RANK_TOP_QUINTILE_POLICY_PASS",
                "rank_top_quintile_se_lower_pct": 2.1,
                "rank_policy_diagnostic_pass": True,
            },
        ]
    )

    performance = build_performance_gate_audit(audit)
    row = performance.iloc[0]

    assert bool(row["rank_policy_supported_candidate"]) is True
    assert bool(row["rank_policy_supported_threshold_warning"]) is True
    assert bool(row["rank_policy_supported_metric_warning"]) is True
    assert row["rank_policy_context_reason"] == "RANK_POLICY_PASS_THRESHOLD_SELECTED_WARNING"


def test_performance_gate_audit_marks_evidence_limited_metric_warnings():
    audit = pd.DataFrame(
        [
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 20,
                "model_name": "score_logistic",
                "split": "",
                "gate": "decision_ece_within_limit",
                "passed": False,
                "severity": "WARN",
                "value": 0.2,
                "threshold": "<=0.1",
                "block_reason": "ECE_GT_LIMIT",
            },
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 20,
                "model_name": "score_logistic",
                "split": "",
                "gate": "selected_oos_event_count",
                "passed": False,
                "severity": "WARN",
                "value": 20,
                "threshold": ">=50",
                "block_reason": "SELECTED_OOS_EVENT_COUNT_LT_MIN",
            },
        ]
    )

    performance = build_performance_gate_audit(audit)
    metric = performance[performance["gate"].eq("decision_ece_within_limit")].iloc[0]

    assert bool(metric["same_candidate_evidence_gap"]) is True
    assert bool(metric["evidence_limited_metric_warning"]) is True
    assert metric["evidence_context_reason"] == "SELECTED_OOS_EVENT_COUNT_LT_MIN"


def test_fold_count_gap_marks_same_candidate_metrics_evidence_limited():
    audit = pd.DataFrame(evaluate_local_model_row(pd.Series({
        **_passing_local_row().to_dict(),
        "candidate_scope": "trigger_all",
        "horizon_days": 10,
        "model_name": "empirical_bayes_group_rate",
        "fold_count": 1,
        "brier_improvement_pct": -0.1,
        "decision_ece": 0.13,
        "positive_expectancy_folds": 1,
        "prediction_quality_pass": False,
    }), diagnostic=True))

    performance = build_performance_gate_audit(audit)
    brier = performance[performance["gate"].eq("brier_improvement_positive")].iloc[0]
    fold_count = performance[performance["gate"].eq("oos_fold_count")].iloc[0]

    assert fold_count["performance_failure_family"] == "sample_evidence"
    assert bool(brier["evidence_limited_metric_warning"]) is True
    assert brier["evidence_context_reason"] == "OOS_FOLD_COUNT_LT_MIN"


def test_selection_contrast_gap_only_marks_selected_economic_metrics():
    row = pd.Series({
        **_passing_local_row().to_dict(),
        "candidate_scope": "entry_research",
        "horizon_days": 5,
        "model_name": "score_logistic",
        "selected_oos_event_count": 120,
        "brier_improvement_pct": -0.1,
        "selected_minus_rule_all_pct": 0.0,
        "selected_signal_expectancy_ci_lower_pct": -0.1,
        "prediction_quality_pass": False,
    })
    audit = pd.DataFrame(evaluate_local_model_row(row, diagnostic=True))

    performance = build_performance_gate_audit(audit)
    brier = performance[performance["gate"].eq("brier_improvement_positive")].iloc[0]
    selected_minus = performance[performance["gate"].eq("selected_minus_rule_all_positive")].iloc[0]

    assert bool(brier["evidence_limited_metric_warning"]) is False
    assert bool(selected_minus["evidence_limited_metric_warning"]) is True
    assert selected_minus["evidence_context_reason"] == "SELECTION_CONTRAST_MISSING"


def test_no_skill_diagnostic_model_metrics_are_marked_rejected():
    audit = pd.DataFrame(
        [
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 5,
                "model_name": "score_logistic",
                "split": "",
                "gate": "brier_improvement_positive",
                "passed": False,
                "severity": "WARN",
                "value": -0.5,
                "threshold": ">0",
                "block_reason": "NO_BRIER_IMPROVEMENT",
            },
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 5,
                "model_name": "score_logistic",
                "split": "",
                "gate": "pr_auc_above_base",
                "passed": False,
                "severity": "WARN",
                "value": 0.45,
                "threshold": ">0.50",
                "block_reason": "PR_AUC_NOT_ABOVE_BASE",
            },
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 5,
                "model_name": "score_logistic",
                "split": "",
                "gate": "decision_ece_within_limit",
                "passed": False,
                "severity": "WARN",
                "value": 0.12,
                "threshold": "<=0.10",
                "block_reason": "ECE_GT_LIMIT",
            },
        ]
    )

    performance = build_performance_gate_audit(audit)
    priorities = build_unresolved_performance_priorities(performance)
    rejected = performance[performance["gate"].eq("decision_ece_within_limit")].iloc[0]

    assert bool(rejected["rejected_model_metric_warning"]) is True
    assert rejected["model_rejection_reason"] == "REJECTED_NO_BRIER_SKILL_AND_PR_AUC_BELOW_BASE"
    assert priorities.empty


def test_no_brier_skill_and_no_economic_uplift_diagnostic_model_is_marked_rejected():
    audit = pd.DataFrame(
        [
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 40,
                "model_name": "elastic_net_logistic",
                "split": "",
                "gate": "brier_improvement_positive",
                "passed": False,
                "severity": "WARN",
                "value": -0.06,
                "threshold": ">0",
                "block_reason": "NO_BRIER_IMPROVEMENT",
            },
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 40,
                "model_name": "elastic_net_logistic",
                "split": "",
                "gate": "pr_auc_above_base",
                "passed": True,
                "severity": "WARN",
                "value": 0.451,
                "threshold": ">0.440",
                "block_reason": "PASS",
            },
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 40,
                "model_name": "elastic_net_logistic",
                "split": "",
                "gate": "selected_minus_rule_all_positive",
                "passed": False,
                "severity": "WARN",
                "value": -0.03,
                "threshold": ">0",
                "block_reason": "ML_SELECTED_MINUS_RULE_ALL_LE_0",
            },
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 40,
                "model_name": "elastic_net_logistic",
                "split": "",
                "gate": "decision_ece_within_limit",
                "passed": False,
                "severity": "WARN",
                "value": 0.117,
                "threshold": "<=0.10",
                "block_reason": "ECE_GT_LIMIT",
            },
        ]
    )

    performance = build_performance_gate_audit(audit)
    priorities = build_unresolved_performance_priorities(performance)
    rejected = performance[performance["gate"].eq("decision_ece_within_limit")].iloc[0]

    assert bool(rejected["rejected_model_metric_warning"]) is True
    assert rejected["model_rejection_reason"] == "REJECTED_NO_BRIER_SKILL_AND_NO_ACTIONABLE_ECONOMIC_UPLIFT"
    assert priorities.empty


def test_no_discrimination_diagnostic_model_metrics_are_marked_rejected():
    audit = pd.DataFrame(
        [
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 5,
                "model_name": "score_logistic",
                "split": "",
                "gate": "brier_improvement_positive",
                "passed": True,
                "severity": "WARN",
                "value": 0.2,
                "threshold": ">0",
                "block_reason": "BRIER_IMPROVEMENT_POSITIVE",
            },
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 5,
                "model_name": "score_logistic",
                "split": "",
                "gate": "pr_auc_above_base",
                "passed": False,
                "severity": "WARN",
                "value": 0.45,
                "threshold": ">0.50",
                "block_reason": "PR_AUC_NOT_ABOVE_BASE",
            },
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 5,
                "model_name": "score_logistic",
                "split": "",
                "gate": "selection_contrast_available",
                "passed": False,
                "severity": "WARN",
                "value": "selected=120;oos=120",
                "threshold": "0<selected<oos",
                "block_reason": "SELECTION_CONTRAST_MISSING",
            },
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 5,
                "model_name": "score_logistic",
                "split": "",
                "gate": "decision_ece_within_limit",
                "passed": False,
                "severity": "WARN",
                "value": 0.105,
                "threshold": "<=0.10",
                "block_reason": "ECE_GT_LIMIT",
            },
        ]
    )

    performance = build_performance_gate_audit(audit)
    priorities = build_unresolved_performance_priorities(performance)
    rejected = performance[performance["gate"].eq("decision_ece_within_limit")].iloc[0]

    assert bool(rejected["rejected_model_metric_warning"]) is True
    assert rejected["model_rejection_reason"] == "REJECTED_NO_DISCRIMINATION_OR_ACTIONABLE_SELECTION_SKILL"
    assert priorities.empty


def test_model_gate_snapshot_splits_metric_shortfalls_from_evidence_gaps():
    audit = pd.DataFrame(
        [
            {
                "gate_group": "pooled_model",
                "candidate_scope": "pooled_trade_ready_entry",
                "horizon_days": 20,
                "model_name": "pooled_candidate",
                "split": "combined_test_holdout",
                "gate": "brier_improvement_positive",
                "passed": False,
                "severity": "CRITICAL",
                "value": -0.1,
                "threshold": ">0",
                "block_reason": "POOLED_NO_BRIER_IMPROVEMENT",
            },
            {
                "gate_group": "pooled_model",
                "candidate_scope": "pooled_trade_ready_entry",
                "horizon_days": 20,
                "model_name": "pooled_candidate",
                "split": "combined_test_holdout",
                "gate": "selected_event_count",
                "passed": False,
                "severity": "CRITICAL",
                "value": 12,
                "threshold": ">=50",
                "block_reason": "POOLED_SELECTED_EVENTS_LT_50",
            },
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 20,
                "model_name": "score_logistic",
                "split": "",
                "gate": "decision_ece_within_limit",
                "passed": False,
                "severity": "WARN",
                "value": 0.2,
                "threshold": "<=0.1",
                "block_reason": "ECE_GT_LIMIT",
            },
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 20,
                "model_name": "score_logistic",
                "split": "",
                "gate": "selected_oos_event_count",
                "passed": False,
                "severity": "WARN",
                "value": 20,
                "threshold": ">=50",
                "block_reason": "SELECTED_OOS_EVENT_COUNT_LT_MIN",
            },
        ]
    )

    snapshot = build_snapshot(audit, {}, {})
    values = dict(zip(snapshot["field"], snapshot["value"]))

    assert values["metric_performance_failed_gate_count"] == 2
    assert values["performance_evidence_gap_count"] == 2
    assert values["actionable_performance_gate_status"] == "ACTION_REQUIRED_PERFORMANCE"
    assert values["actionable_performance_failed_gate_count"] == 2
    assert values["actionable_metric_performance_failed_gate_count"] == 1
    assert values["active_metric_performance_failed_gate_count"] == 1
    assert values["active_performance_evidence_gap_count"] == 1
    assert values["diagnostic_metric_performance_warning_gate_count"] == 1
    assert values["diagnostic_performance_evidence_gap_count"] == 1
    assert values["evidence_limited_metric_warning_count"] == 1
    assert values["unresolved_diagnostic_metric_performance_warning_gate_count"] == 0
    assert values["resolved_diagnostic_performance_warning_count"] == 2
    assert values["classified_diagnostic_performance_warning_count"] == 2
    assert values["unresolved_diagnostic_performance_warning_count"] == 0
    assert values["performance_warning_resolution_status"] == "ACTION_REQUIRED_ACTIVE_PERFORMANCE_FAILURES"
    assert values["metric_performance_failed_families"] == "probabilistic_skill|probability_calibration"


def test_model_gate_snapshot_counts_rank_policy_supported_warnings():
    audit = pd.DataFrame(
        [
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 120,
                "model_name": "score_logistic",
                "split": "",
                "gate": "selected_expectancy_ci_lower_positive",
                "passed": False,
                "severity": "WARN",
                "value": -0.1,
                "threshold": ">0",
                "block_reason": "SELECTED_EXPECTANCY_CI_LOWER_LE_0",
            },
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 120,
                "model_name": "score_logistic",
                "split": "",
                "gate": "decision_ece_within_limit",
                "passed": False,
                "severity": "WARN",
                "value": 0.2,
                "threshold": "<=0.1",
                "block_reason": "ECE_GT_LIMIT",
            },
            {
                "gate_group": "local_prediction_rank_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 120,
                "model_name": "score_logistic",
                "split": "",
                "gate": "rank_top_quintile_uplift_diagnostic",
                "passed": True,
                "severity": "INFO",
                "value": 6.2,
                "threshold": "diagnostic_only_top_20pct_minus_all_pct",
                "block_reason": "RANK_TOP_QUINTILE_POLICY_PASS",
                "rank_top_quintile_se_lower_pct": 2.1,
                "rank_policy_diagnostic_pass": True,
            },
        ]
    )

    snapshot = build_snapshot(audit, {}, {})
    values = dict(zip(snapshot["field"], snapshot["value"]))

    assert values["diagnostic_metric_performance_warning_gate_count"] == 2
    assert values["actionable_performance_gate_status"] == "PASS_ACTIONABLE_PERFORMANCE"
    assert values["actionable_performance_failed_gate_count"] == 0
    assert values["actionable_metric_performance_failed_gate_count"] == 0
    assert values["rank_policy_supported_performance_warning_count"] == 2
    assert values["rank_policy_supported_threshold_warning_count"] == 1
    assert values["rank_policy_supported_metric_warning_count"] == 2
    assert values["rank_policy_supported_threshold_metric_warning_count"] == 1
    assert values["unresolved_diagnostic_metric_performance_warning_gate_count"] == 0
    assert values["resolved_diagnostic_performance_warning_count"] == 2
    assert values["classified_diagnostic_performance_warning_count"] == 2
    assert values["unresolved_diagnostic_performance_warning_count"] == 0
    assert values["performance_warning_resolution_status"] == "PASS_ALL_NONACTIVE_WARNINGS_CLASSIFIED"


def test_performance_warning_resolution_summary_classifies_all_warning_types():
    performance = pd.DataFrame(
        [
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 20,
                "model_name": "weak_model",
                "gate": "decision_ece_within_limit",
                "block_reason": "ECE_GT_LIMIT",
                "performance_failure_family": "probability_calibration",
                "performance_failure_kind": "metric_shortfall",
                "active_performance_gate": False,
                "rank_policy_supported_metric_warning": False,
                "evidence_limited_metric_warning": False,
                "rejected_model_metric_warning": True,
            },
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 20,
                "model_name": "rank_model",
                "gate": "selected_minus_rule_all_positive",
                "block_reason": "ML_SELECTED_MINUS_RULE_ALL_LE_0",
                "performance_failure_family": "economic_uplift",
                "performance_failure_kind": "metric_shortfall",
                "active_performance_gate": False,
                "rank_policy_supported_candidate": True,
                "rank_policy_supported_metric_warning": True,
            },
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 20,
                "model_name": "small_sample_model",
                "gate": "brier_improvement_positive",
                "block_reason": "NO_BRIER_IMPROVEMENT",
                "performance_failure_family": "probabilistic_skill",
                "performance_failure_kind": "metric_shortfall",
                "active_performance_gate": False,
                "evidence_limited_metric_warning": True,
            },
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 20,
                "model_name": "small_sample_model",
                "gate": "selected_oos_event_count",
                "block_reason": "SELECTED_OOS_EVENT_COUNT_LT_MIN",
                "performance_failure_family": "sample_evidence",
                "performance_failure_kind": "evidence_gap",
                "active_performance_gate": False,
            },
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 20,
                "model_name": "small_sample_model",
                "gate": "prediction_quality_pass",
                "block_reason": "PREDICTION_QUALITY_FALSE",
                "performance_failure_family": "aggregate_quality_flag",
                "performance_failure_kind": "aggregate_flag",
                "active_performance_gate": False,
            },
            {
                "gate_group": "strategy_validation_diagnostic",
                "candidate_scope": "strategy",
                "horizon_days": "",
                "model_name": "",
                "gate": "dsr_pass",
                "block_reason": "DSR_NOT_PASSED",
                "performance_failure_family": "backtest_overfit_control",
                "performance_failure_kind": "metric_shortfall",
                "active_performance_gate": False,
                "strategy_diagnostic_metric_warning": True,
            },
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 60,
                "model_name": "unresolved_model",
                "gate": "decision_ece_within_limit",
                "block_reason": "ECE_GT_LIMIT",
                "performance_failure_family": "probability_calibration",
                "performance_failure_kind": "metric_shortfall",
                "active_performance_gate": False,
            },
        ]
    )

    summary = build_performance_warning_resolution_summary(performance)
    counts = dict(zip(summary["resolution_bucket"], summary["warning_count"]))

    assert counts["rejected_model_metric_warning"] == 1
    assert counts["rank_policy_supported_metric_warning"] == 1
    assert counts["evidence_limited_metric_warning"] == 1
    assert counts["diagnostic_evidence_gap"] == 1
    assert counts["diagnostic_aggregate_quality_flag"] == 1
    assert counts["strategy_diagnostic_nonblocking"] == 1
    assert counts["unresolved_diagnostic_warning"] == 1


def test_candidate_disposition_summary_separates_rejected_rank_and_scope_blocked_models():
    comparison = pd.DataFrame(
        [
            {
                "candidate_scope": "entry_research",
                "horizon_days": 20,
                "model_name": "scope_blocked",
                "prediction_quality_pass": True,
                "performance_quality_pass": True,
                "decision_scope_eligible": False,
                "rank_policy_diagnostic_pass": False,
                "oos_event_count": 100,
                "selected_oos_event_count": 80,
            },
            {
                "candidate_scope": "entry_research",
                "horizon_days": 20,
                "model_name": "rank_model",
                "prediction_quality_pass": False,
                "performance_quality_pass": False,
                "decision_scope_eligible": False,
                "rank_policy_diagnostic_pass": True,
            },
            {
                "candidate_scope": "entry_research",
                "horizon_days": 20,
                "model_name": "weak_model",
                "prediction_quality_pass": False,
                "performance_quality_pass": False,
                "decision_scope_eligible": False,
                "rank_policy_diagnostic_pass": False,
            },
        ]
    )
    next_day_comparison = pd.DataFrame(
        [
            {
                "candidate_scope": "next_day_up_all",
                "horizon_days": 1,
                "model_name": "next_day_pass",
                "prediction_quality_pass": True,
                "performance_quality_pass": True,
                "decision_scope_eligible": False,
                "directional_diagnostic_quality_pass": True,
                "performance_quality_block_reasons": "PASS",
                "brier_improvement_pct": -0.04,
                "decision_ece": 0.06,
            },
            {
                "candidate_scope": "next_day_up_all",
                "horizon_days": 1,
                "model_name": "next_day_weak",
                "prediction_quality_pass": False,
                "performance_quality_pass": False,
                "decision_scope_eligible": False,
                "directional_diagnostic_quality_pass": False,
                "performance_quality_block_reasons": "NO_BRIER_IMPROVEMENT|PR_AUC_NOT_ABOVE_BASE",
                "brier_improvement_pct": -0.3,
                "decision_ece": 0.08,
            },
        ]
    )
    audit = pd.DataFrame(
        [
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 20,
                "model_name": "scope_blocked",
                "split": "",
                "gate": "decision_scope_eligible",
                "passed": False,
                "severity": "WARN",
                "value": False,
                "threshold": True,
                "block_reason": "NOT_20D_TRADE_READY_DECISION_SCOPE",
            },
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 20,
                "model_name": "rank_model",
                "split": "",
                "gate": "selected_minus_rule_all_positive",
                "passed": False,
                "severity": "WARN",
                "value": -0.1,
                "threshold": ">0",
                "block_reason": "ML_SELECTED_MINUS_RULE_ALL_LE_0",
            },
            {
                "gate_group": "local_prediction_rank_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 20,
                "model_name": "rank_model",
                "split": "",
                "gate": "rank_top_quintile_uplift_diagnostic",
                "passed": True,
                "severity": "INFO",
                "value": 1.0,
                "threshold": "diagnostic_only_top_20pct_minus_all_pct",
                "block_reason": "PASS",
                "rank_policy_diagnostic_pass": True,
            },
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 20,
                "model_name": "weak_model",
                "split": "",
                "gate": "brier_improvement_positive",
                "passed": False,
                "severity": "WARN",
                "value": -1.0,
                "threshold": ">0",
                "block_reason": "NO_BRIER_IMPROVEMENT",
            },
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 20,
                "model_name": "weak_model",
                "split": "",
                "gate": "pr_auc_above_base",
                "passed": False,
                "severity": "WARN",
                "value": 0.4,
                "threshold": ">0.5",
                "block_reason": "PR_AUC_NOT_ABOVE_BASE",
            },
        ]
    )
    performance = build_performance_gate_audit(audit)
    summary = build_model_candidate_disposition_summary(audit, performance, comparison, next_day_comparison)
    dispositions = dict(zip(summary["model_name"], summary["disposition"]))
    roles = dict(zip(summary["model_name"], summary["model_role"]))

    assert dispositions["scope_blocked"] == "local_quality_pass_scope_blocked"
    assert dispositions["rank_model"] == "rank_policy_supported_diagnostic"
    assert dispositions["weak_model"] == "rejected_model_diagnostic"
    assert dispositions["next_day_pass"] == "next_day_directional_display_pass"
    assert dispositions["next_day_weak"] == "next_day_rejected_challenger"
    assert roles["next_day_pass"] == "next_day_directional_candidate"


def test_rank_policy_promotion_watchlist_quantifies_promotion_gaps():
    summary = pd.DataFrame(
        [
            {
                "candidate_scope": "entry_research",
                "horizon_days": 40,
                "model_name": "rank_model",
                "disposition": "rank_policy_supported_diagnostic",
                "warning_count": 4,
                "oos_event_count": 300,
                "selected_oos_event_count": 120,
                "rank_top_quintile_minus_all_pct": 5.0,
                "rank_top_quintile_se_lower_pct": 2.0,
                "brier_improvement_pct": -1.5,
                "decision_ece": 0.14,
                "selected_minus_rule_all_pct": -0.2,
                "selected_signal_expectancy_ci_lower_pct": -0.1,
                "positive_expectancy_folds": 3,
                "min_selected_events_per_fold": 8,
                "threshold_iqr": 0.12,
                "min_calibration_bin_n": 24,
                "fixed_width_min_calibration_bin_n": 10,
                "decision_scope_eligible": False,
            },
            {
                "candidate_scope": "entry_research",
                "horizon_days": 20,
                "model_name": "rejected",
                "disposition": "rejected_model_diagnostic",
                "warning_count": 8,
            },
        ]
    )

    watchlist = build_rank_policy_promotion_watchlist(summary)
    row = watchlist.iloc[0]

    assert len(watchlist) == 1
    assert row["model_name"] == "rank_model"
    assert row["brier_gap_to_zero_pct"] == 1.5
    assert round(row["decision_ece_gap_to_limit"], 6) == 0.04
    assert row["selected_minus_all_gap_pct"] == 0.2
    assert row["selected_ci_gap_pct"] == 0.1
    assert row["positive_expectancy_fold_gap"] == 1.0
    assert row["selected_events_per_fold_gap"] == 2.0
    assert row["calibration_bin_gap"] == 6.0
    assert row["promotion_blocker_count"] == 9
    assert "DECISION_ECE" in row["promotion_blockers"]
    assert "SELECTED_CI" in row["promotion_blockers"]
    assert row["recommended_action"] == "improve_probability_skill_before_rank_signal_promotion"


def test_rank_policy_promotion_watchlist_does_not_hide_selected_ci_blocker():
    summary = pd.DataFrame(
        [
            {
                "candidate_scope": "entry_research",
                "horizon_days": 5,
                "model_name": "rank_model",
                "disposition": "rank_policy_supported_diagnostic",
                "warning_count": 2,
                "oos_event_count": 400,
                "selected_oos_event_count": 220,
                "rank_top_quintile_minus_all_pct": 1.0,
                "rank_top_quintile_se_lower_pct": 0.2,
                "brier_improvement_pct": 0.1,
                "decision_ece": 0.05,
                "selected_minus_rule_all_pct": 0.1,
                "selected_signal_expectancy_ci_lower_pct": -0.2,
                "positive_expectancy_folds": 4,
                "min_selected_events_per_fold": 20,
                "threshold_iqr": 0.01,
                "min_calibration_bin_n": 31,
                "fixed_width_min_calibration_bin_n": 31,
                "decision_scope_eligible": False,
            }
        ]
    )

    watchlist = build_rank_policy_promotion_watchlist(summary)
    row = watchlist.iloc[0]

    assert row["selected_ci_gap_pct"] == 0.2
    assert row["promotion_blocker_count"] == 2
    assert row["promotion_blockers"] == "SELECTED_CI|DECISION_SCOPE"
    assert row["recommended_action"] == "increase_selected_expectancy_lower_bound_before_promotion"


def test_rank_policy_promotion_watchlist_prioritizes_fewer_remaining_blockers():
    summary = pd.DataFrame(
        [
            {
                "candidate_scope": "entry_research",
                "horizon_days": 40,
                "model_name": "strong_rank_many_blockers",
                "disposition": "rank_policy_supported_diagnostic",
                "warning_count": 6,
                "oos_event_count": 400,
                "selected_oos_event_count": 200,
                "rank_top_quintile_minus_all_pct": 8.0,
                "rank_top_quintile_se_lower_pct": 4.0,
                "brier_improvement_pct": -2.0,
                "decision_ece": 0.16,
                "selected_minus_rule_all_pct": -1.0,
                "selected_signal_expectancy_ci_lower_pct": -0.5,
                "positive_expectancy_folds": 2,
                "min_selected_events_per_fold": 3,
                "threshold_iqr": 0.2,
                "min_calibration_bin_n": 20,
                "fixed_width_min_calibration_bin_n": 10,
                "decision_scope_eligible": False,
            },
            {
                "candidate_scope": "entry_research",
                "horizon_days": 5,
                "model_name": "weaker_rank_fewer_blockers",
                "disposition": "rank_policy_supported_diagnostic",
                "warning_count": 2,
                "oos_event_count": 400,
                "selected_oos_event_count": 220,
                "rank_top_quintile_minus_all_pct": 1.0,
                "rank_top_quintile_se_lower_pct": 0.2,
                "brier_improvement_pct": 0.1,
                "decision_ece": 0.05,
                "selected_minus_rule_all_pct": 0.1,
                "selected_signal_expectancy_ci_lower_pct": -0.2,
                "positive_expectancy_folds": 4,
                "min_selected_events_per_fold": 20,
                "threshold_iqr": 0.01,
                "min_calibration_bin_n": 31,
                "fixed_width_min_calibration_bin_n": 31,
                "decision_scope_eligible": False,
            },
        ]
    )

    watchlist = build_rank_policy_promotion_watchlist(summary)

    assert watchlist.iloc[0]["model_name"] == "weaker_rank_fewer_blockers"
    assert watchlist.iloc[0]["promotion_blocker_count"] < watchlist.iloc[1]["promotion_blocker_count"]


def test_unresolved_performance_priorities_exclude_rank_supported_threshold_warnings():
    performance = pd.DataFrame(
        [
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 120,
                "model_name": "score_logistic",
                "split": "",
                "gate": "selected_expectancy_ci_lower_positive",
                "passed": False,
                "severity": "WARN",
                "value": -0.1,
                "threshold": ">0",
                "block_reason": "SELECTED_EXPECTANCY_CI_LOWER_LE_0",
                "performance_failure_family": "economic_uplift",
                "performance_failure_kind": "metric_shortfall",
                "active_performance_gate": False,
                "rank_policy_supported_candidate": True,
                "rank_policy_supported_threshold_warning": True,
                "rank_policy_supported_metric_warning": True,
            },
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 120,
                "model_name": "score_logistic",
                "split": "",
                "gate": "decision_ece_within_limit",
                "passed": False,
                "severity": "WARN",
                "value": 0.2,
                "threshold": "<=0.1",
                "block_reason": "ECE_GT_LIMIT",
                "performance_failure_family": "probability_calibration",
                "performance_failure_kind": "metric_shortfall",
                "active_performance_gate": False,
                "rank_policy_supported_candidate": True,
                "rank_policy_supported_threshold_warning": False,
                "rank_policy_supported_metric_warning": True,
            },
        ]
    )

    priorities = build_unresolved_performance_priorities(performance)

    assert priorities.empty


def test_unresolved_performance_priorities_exclude_evidence_limited_metric_warnings():
    performance = pd.DataFrame(
        [
            {
                "gate_group": "local_prediction_diagnostic",
                "candidate_scope": "entry_research",
                "horizon_days": 20,
                "model_name": "score_logistic",
                "split": "",
                "gate": "decision_ece_within_limit",
                "passed": False,
                "severity": "WARN",
                "value": 0.2,
                "threshold": "<=0.1",
                "block_reason": "ECE_GT_LIMIT",
                "performance_failure_family": "probability_calibration",
                "performance_failure_kind": "metric_shortfall",
                "active_performance_gate": False,
                "rank_policy_supported_candidate": False,
                "rank_policy_supported_threshold_warning": False,
                "rank_policy_supported_metric_warning": False,
                "evidence_limited_metric_warning": True,
            }
        ]
    )

    priorities = build_unresolved_performance_priorities(performance)

    assert priorities.empty


def test_model_cpcv_pass_keeps_strategy_dsr_as_warning_not_research_blocker():
    comparison = pd.DataFrame([_passing_local_row().to_dict()])
    latest = {"prediction_use_status": "DISPLAY_ONLY_NO_MODEL_CANDIDATE"}
    pooled_comparison = pd.DataFrame(
        [
            {
                "model_name": "pooled_candidate",
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
                "selected_minus_all_pct": 0.6,
                "selected_minus_all_ci_lower_pct": 0.2,
                "selected_minus_score_baseline_ci_lower_pct": 0.1,
                "selected_expectancy_ci_lower_pct": 0.2,
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
                "model_name": "tsm_specific_calibrated_layer",
                "split": "tsm_combined_test_holdout",
                "event_count": 40,
                "ece": 0.10,
                "tsm_calibration_route_pass": True,
            }
        ]
    )
    cpcv_model = pd.DataFrame(
        [
            {
                "model_name": "pooled_candidate",
                "candidate_scope": "trade_ready_entry_only",
                "selected_minus_all_pct_median": 1.0,
                "selected_minus_all_pct_q25": 0.3,
            }
        ]
    )
    pbo = pd.DataFrame([{"pbo_proxy": 0.20}])
    dsr = pd.DataFrame([{"dsr_pass": False, "deflated_sharpe_ratio": 0.4, "observed_sharpe": 0.6}])

    audit = build_audit(
        comparison,
        latest,
        pooled_comparison,
        {
            "latest_signal_pass": False,
            "decision_support_allowed": False,
            "latest_block_reasons": "LATEST_NOT_TRADE_READY",
            "tsm_like_effective_train_validation_n": 600,
        },
        tsm_calibration,
        pd.DataFrame(),
        cpcv_model,
        pbo,
        pd.DataFrame(),
        dsr,
    )
    dsr_row = audit[audit["gate"].eq("dsr_pass")].iloc[0]
    performance = build_performance_gate_audit(audit)
    dsr_performance = performance[performance["gate"].eq("dsr_pass")].iloc[0]
    priorities = build_unresolved_performance_priorities(performance)
    snapshot = build_snapshot(audit, latest, {"latest_signal_pass": False})
    values = dict(zip(snapshot["field"], snapshot["value"]))

    assert bool(dsr_row["passed"]) is False
    assert dsr_row["severity"] == "WARN"
    assert dsr_row["gate_group"] == "strategy_validation_diagnostic"
    assert dsr_row["candidate_scope"] == "strategy_dsr_diagnostic"
    assert values["research_validation_pass"] is True
    assert values["model_quality_warning_gate_count"] == 0
    assert values["strategy_warning_gate_count"] == 1
    assert bool(dsr_performance["strategy_diagnostic_metric_warning"]) is True
    assert values["strategy_diagnostic_metric_warning_count"] == 1
    assert values["unresolved_diagnostic_metric_performance_warning_gate_count"] == 0
    assert values["actionable_performance_gate_status"] == "PASS_ACTIONABLE_PERFORMANCE"
    assert values["actionable_performance_failed_gate_count"] == 0
    assert priorities.empty
    assert values["model_gate_status"] == "PASS_MODEL_QUALITY_SIGNAL_STANDBY"


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


def test_failed_tsm_route_does_not_block_when_scoring_falls_back_to_pooled_route():
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
                "model_name": "tsm_specific_direct",
                "split": "tsm_combined_test_holdout",
                "event_count": 41,
                "ece": 0.33,
                "decision_ece": 0.33,
                "tsm_calibration_route_pass": False,
                "tsm_calibration_route_failure_reasons": "TSM_COMBINED_ECE_GT_0_15",
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
            "tsm_calibration_scoring_route": "POOLED_ONLY",
            "tsm_calibration_scoring_route_reason": "SELECTED_ROUTE_FAILED_IDENTITY_FALLBACK",
        },
        tsm_calibration,
    )
    snapshot = build_snapshot(audit, {"prediction_use_status": "DISPLAY_ONLY_NO_MODEL_CANDIDATE"}, {"latest_signal_pass": False})
    values = dict(zip(snapshot["field"], snapshot["value"]))

    assert values["pooled_tsm_calibration_pass"] is True
    assert values["effective_tsm_scoring_route_pass"] is True
    assert values["pooled_model_quality_pass"] is True
    assert values["model_gate_status"] == "PASS_MODEL_QUALITY_SIGNAL_STANDBY"


def test_research_validation_reports_present_dsr_failure_counts():
    rows = pd.DataFrame(
        evaluate_research_validation(
            pd.DataFrame(),
            pd.DataFrame(
                [
                    {
                        "model_name": "pooled_stack_calibrated",
                        "candidate_scope": "all",
                        "selected_minus_all_pct_median": 0.5,
                        "selected_minus_all_pct_q25": 0.1,
                    }
                ]
            ),
            pd.DataFrame([{"pbo_proxy": 0.1}]),
            pd.DataFrame([{"pbo_cscv": 0.1}]),
            pd.DataFrame(
                [
                    {"strategy_name": "A", "deflated_sharpe_ratio": 0.25, "observed_sharpe": 0.5, "dsr_pass": False},
                    {"strategy_name": "B", "deflated_sharpe_ratio": 0.75, "observed_sharpe": 1.0, "dsr_pass": False},
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "model_name": "tsm_specific_direct",
                        "split": "tsm_combined_test_holdout",
                        "event_count": 800,
                        "is_selected_tsm_calibration_route": True,
                    }
                ]
            ),
        )
    )

    dsr = rows[rows["gate"].eq("dsr_pass")].iloc[0]

    assert bool(dsr["passed"]) is False
    assert "passing_strategy_count=0/2" in dsr["value"]
    assert "best_dsr=0.750000" in dsr["value"]
    assert "best_strategy=B" in dsr["value"]


def test_research_validation_uses_pooled_tsm_like_effective_n_before_direct_tsm_count():
    rows = pd.DataFrame(
        evaluate_research_validation(
            pd.DataFrame(),
            pd.DataFrame(
                [
                    {
                        "model_name": "pooled_stack_calibrated",
                        "candidate_scope": "all",
                        "selected_minus_all_pct_median": 0.5,
                        "selected_minus_all_pct_q25": 0.1,
                    }
                ]
            ),
            pd.DataFrame([{"pbo_proxy": 0.1}]),
            pd.DataFrame([{"pbo_cscv": 0.1}]),
            pd.DataFrame([{"dsr_pass": True}]),
            pd.DataFrame(
                [
                    {
                        "model_name": "tsm_specific_direct",
                        "split": "tsm_combined_test_holdout",
                        "event_count": 41,
                        "is_selected_tsm_calibration_route": True,
                    }
                ]
            ),
            {"tsm_like_calibration_route": "TSM_LIKE_WEIGHTED_ISOTONIC", "tsm_like_effective_train_validation_n": 800},
        )
    )

    effective = rows[rows["gate"].eq("tsm_like_effective_calibration_n")].iloc[0]

    assert bool(effective["passed"]) is True
    assert float(effective["value"]) == 800.0
    assert effective["model_name"] == "TSM_LIKE_WEIGHTED_ISOTONIC"
