from __future__ import annotations

import pandas as pd

from tsm_dashboard import command_for_mode


def _value_after(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def test_dashboard_daily_command_wires_universe_mode_and_skip_flags():
    mode, cmd = command_for_mode(
        {
            "mode": "downstream",
            "universe_mode": "universe",
            "skip_symbol_build": True,
            "skip_symbol_diagnostics": True,
        }
    )

    assert mode == "downstream"
    assert _value_after(cmd, "--universe-mode") == "universe"
    assert _value_after(cmd, "--decision-universe-config").endswith("semiconductor_universe_top10.csv")
    assert _value_after(cmd, "--research-universe-config").endswith("semiconductor_universe_expanded.csv")
    assert "--skip-data-refresh" in cmd
    assert "--skip-universe-symbol-build" in cmd
    assert "--skip-universe-symbol-diagnostics" in cmd


def test_dashboard_paper_oms_commands_use_universe_inputs():
    _, intent_cmd = command_for_mode({"mode": "order_intent_engine"})
    assert "--universe-config" in intent_cmd
    assert _value_after(intent_cmd, "--latest-predictions").endswith("tsm_universe_latest_predictions.csv")
    assert _value_after(intent_cmd, "--signals-root") == "tsm_price_rule_output"

    _, risk_cmd = command_for_mode({"mode": "portfolio_risk_engine"})
    assert _value_after(risk_cmd, "--latest-signals").endswith("tsm_universe_latest_signals.csv")

    _, execution_cmd = command_for_mode({"mode": "paper_execution_engine"})
    assert _value_after(execution_cmd, "--signals-root") == "tsm_price_rule_output"
    assert "--universe-config" in execution_cmd


def test_dashboard_pooled_universe_command_can_skip_symbol_diagnostics():
    _, cmd = command_for_mode({"mode": "pooled_universe_update", "skip_symbol_diagnostics": True})

    assert "--skip-symbol-diagnostics" in cmd
    assert _value_after(cmd, "--decision-universe-config").endswith("semiconductor_universe_top10.csv")
    assert _value_after(cmd, "--research-universe-config").endswith("semiconductor_universe_expanded.csv")
    assert _value_after(cmd, "--intraday-features").endswith("tsm_intraday_daily_features.csv")


def test_dashboard_next_day_command_updates_latest_prediction_snapshot():
    _, cmd = command_for_mode({"mode": "next_day_up_model_engine"})

    assert cmd[1] == "tsm_next_day_up_model_engine.py"
    assert "--aggregate-universe" in cmd
    assert _value_after(cmd, "--universe-config").endswith("semiconductor_universe_top10.csv")
    assert _value_after(cmd, "--latest-prediction").endswith("tsm_latest_prediction_snapshot.csv")
    assert _value_after(cmd, "--outdir") == "tsm_price_rule_output"
    assert _value_after(cmd, "--symbol") == "TSM"


def test_dashboard_next_close_command_uses_pooled_feature_matrix():
    _, cmd = command_for_mode({"mode": "next_close_forecast_engine"})

    assert cmd[1] == "tsm_next_close_forecast_engine.py"
    assert _value_after(cmd, "--pooled-feature-matrix").endswith("tsm_prediction_pooled_feature_matrix.csv")
    assert _value_after(cmd, "--decision-universe-config").endswith("semiconductor_universe_top10.csv")
    assert _value_after(cmd, "--latest-prediction").endswith("tsm_latest_prediction_snapshot.csv")
    assert _value_after(cmd, "--outdir") == "tsm_price_rule_output"


def test_dashboard_universe_market_data_command_updates_intraday_layers():
    _, cmd = command_for_mode({"mode": "universe_market_data_update", "interval": "5m"})

    assert cmd[1] == "run_universe_market_data_update.py"
    assert "--universe-config" in cmd
    assert _value_after(cmd, "--bar-scope") == "both"
    assert _value_after(cmd, "--model-minute-interval") == "5m"
    assert _value_after(cmd, "--execution-minute-interval") == "1m"
    assert "--skip-charts" in cmd


def test_dashboard_hourly_and_intraday_commands_are_top10_universe_updates():
    _, hourly_cmd = command_for_mode({"mode": "hourly"})
    assert hourly_cmd[1] == "run_universe_market_data_update.py"
    assert _value_after(hourly_cmd, "--bar-scope") == "hourly"
    assert _value_after(hourly_cmd, "--universe-config").endswith("semiconductor_universe_top10.csv")

    _, intraday_cmd = command_for_mode({"mode": "intraday", "model_minute_interval": "5m", "execution_minute_interval": "1m"})
    assert intraday_cmd[1] == "run_universe_market_data_update.py"
    assert _value_after(intraday_cmd, "--bar-scope") == "minute"
    assert _value_after(intraday_cmd, "--model-minute-interval") == "5m"
    assert _value_after(intraday_cmd, "--execution-minute-interval") == "1m"


def test_dashboard_summary_exposes_model_monitoring_tables(tmp_path, monkeypatch):
    import tsm_dashboard

    output_dir = tmp_path / "output"
    rule_dir = tmp_path / "rules"
    output_dir.mkdir()
    rule_dir.mkdir()

    pd.DataFrame(
        [
            {
                "candidate_scope": "entry_research",
                "horizon_days": 5,
                "model_name": "empirical_bayes_group_rate",
                "validation_utility_fold_count": 6,
                "validation_positive_utility_folds": 1,
                "min_validation_utility_lower_bound_pct": -0.7,
            }
        ]
    ).to_csv(rule_dir / "tsm_prediction_near_pass_candidates.csv", index=False)
    pd.DataFrame(
        [
            {"field": "next_day_p_up_1d", "value": 0.52},
            {"field": "next_day_threshold_1d", "value": 0.49},
            {"field": "next_day_best_model_1d", "value": "empirical_bayes_group_rate"},
        ]
    ).to_csv(rule_dir / "tsm_next_day_up_latest_snapshot.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_scope": "next_day_up_all",
                "horizon_days": 1,
                "model_name": "empirical_bayes_group_rate",
                "oos_event_count": 1000,
            }
        ]
    ).to_csv(rule_dir / "tsm_next_day_up_model_comparison.csv", index=False)
    pd.DataFrame([{"check": "next_day_up_labels_non_empty", "passed": True, "severity": "CRITICAL"}]).to_csv(
        rule_dir / "tsm_next_day_up_quality_checks.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "candidate_scope": "entry_research",
                "horizon_days": 5,
                "model_name": "empirical_bayes_group_rate",
                "promotion_blocker_count": 2,
                "promotion_blockers": "SELECTED_CI|DECISION_SCOPE",
            }
        ]
    ).to_csv(rule_dir / "tsm_model_rank_policy_promotion_watchlist.csv", index=False)
    pd.DataFrame([{"candidate_scope": "entry_research", "disposition": "rank_policy_supported_diagnostic"}]).to_csv(
        rule_dir / "tsm_model_candidate_disposition_summary.csv",
        index=False,
    )
    pd.DataFrame([{"resolution_bucket": "rejected_model_metric_warning", "warning_count": 82}]).to_csv(
        rule_dir / "tsm_model_performance_warning_resolution_summary.csv",
        index=False,
    )
    pd.DataFrame(
        columns=[
            "priority",
            "performance_failure_family",
            "performance_failure_kind",
            "gate",
            "block_reason",
            "candidate_scope",
            "horizon_days",
            "model_name",
            "warning_count",
            "active_warning_count",
            "rank_policy_supported_count",
            "value_min",
            "value_max",
            "recommended_action",
        ]
    ).to_csv(rule_dir / "tsm_model_unresolved_performance_priorities.csv", index=False)

    monkeypatch.setattr(tsm_dashboard, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(tsm_dashboard, "RULE_DIR", rule_dir)
    monkeypatch.setattr(tsm_dashboard, "ALLOWED_FILE_ROOTS", [output_dir.resolve(), rule_dir.resolve()])

    summary = tsm_dashboard.build_summary()

    assert summary["tables"]["prediction_near_pass_candidates"][0]["validation_utility_fold_count"] == 6
    assert float(summary["snapshots"]["next_day_prediction"]["p_up"]) == 0.52
    assert summary["tables"]["next_day_model_comparison"][0]["horizon_days"] == 1
    assert summary["quality"]["next_day_prediction"]["failed"] == 0
    assert summary["tables"]["model_rank_policy_promotion_watchlist"][0]["promotion_blockers"] == "SELECTED_CI|DECISION_SCOPE"
    assert summary["tables"]["model_candidate_disposition_summary"][0]["disposition"] == "rank_policy_supported_diagnostic"
    assert summary["tables"]["model_performance_warning_resolution_summary"][0]["warning_count"] == 82
    assert summary["tables"]["model_unresolved_performance_priorities"] == []
