import pandas as pd

from tsm_pooled_dataset_builder import build_sample_audit
from tsm_pooled_model_engine import (
    build_paper_gate_snapshot,
    effective_sample_size,
    fit_tsm_like_weight_table,
    choose_tsm_like_route,
    choose_fold_consensus_trade_ready_threshold_v5,
    threshold_stability_from_fold_metrics,
)
from tsm_universe_validator import validate_symbol_frame


def test_expanded_universe_config_has_at_least_50_candidates():
    config = pd.read_csv("config/semiconductor_universe_expanded.csv")

    assert config["symbol"].nunique() >= 50
    assert {"TSM", "NVDA", "ASML", "SMH"}.issubset(set(config["symbol"]))


def test_universe_validator_excludes_short_history_from_strict_training():
    short_history = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=250, freq="B"),
            "open": 10.0,
            "high": 10.5,
            "low": 9.5,
            "close": 10.0,
            "volume": 1_000_000,
            "return_20d": 0.01,
            "dollar_volume": 10_000_000,
        }
    )
    row = pd.Series({"symbol": "ARM", "symbol_group": "fabless_ai_analog"})

    result = validate_symbol_frame(row, short_history)

    assert result["strict_eligible"] is False
    assert result["eligibility_status"] == "SHORT_HISTORY_RESEARCH_ONLY"


def test_sample_audit_reports_missing_symbols_without_stopping_build():
    config = pd.DataFrame(
        [
            {"symbol": "TSM", "symbol_group": "foundry_idm"},
            {"symbol": "ARM", "symbol_group": "fabless_ai_analog"},
        ]
    )
    labels = pd.DataFrame(
        {
            "symbol": ["TSM"],
            "symbol_group": ["foundry_idm"],
            "date": [pd.Timestamp("2024-01-02")],
            "is_trade_ready_entry_candidate": [True],
            "is_model_training_candidate": [True],
            "label_status_20d": ["LABELED"],
        }
    )
    features = labels.copy()

    audit = build_sample_audit(config, labels, features)
    values = audit.set_index("symbol")

    assert bool(values.loc["TSM", "loaded"]) is True
    assert bool(values.loc["ARM", "loaded"]) is False
    assert int(values.loc["TSM", "trade_ready_20d_labeled"]) == 1


def test_tsm_like_weights_are_clipped_and_effective_n_matches_formula():
    rows = []
    dates = pd.date_range("2020-01-01", periods=80, freq="B")
    for symbol, group, beta, vol in [("TSM", "foundry_idm", 1.2, 0.30), ("ASML", "semicap_osat", 1.1, 0.32), ("XYZ", "other", 2.5, 0.90)]:
        for i, date in enumerate(dates):
            rows.append(
                {
                    "symbol": symbol,
                    "symbol_group": group,
                    "date": date,
                    "return_20d": i / 1000 if symbol != "XYZ" else -i / 1000,
                    "beta_vs_smh_126d": beta,
                    "vol_20d_ann": vol,
                }
            )

    table = fit_tsm_like_weight_table(pd.DataFrame(rows))

    assert table["tsm_like_weight"].between(0.10, 1.00).all()
    assert table.loc[table["symbol"].eq("TSM"), "tsm_like_weight"].iloc[0] == 1.0
    assert effective_sample_size([1.0, 1.0, 0.5, 0.5]) == (3.0**2) / (1.0 + 1.0 + 0.25 + 0.25)


def test_tsm_like_route_selection_uses_train_validation_metrics():
    metrics = pd.DataFrame(
        [
            {"tsm_like_calibration_route": "TSM_DIRECT_ONLY", "split": "tsm_like_train_validation", "decision_ece": 0.05, "brier_improvement_pct": 1.0, "effective_n": 600},
            {"tsm_like_calibration_route": "TSM_DIRECT_ONLY", "split": "tsm_like_combined_test_holdout", "decision_ece": 0.20, "brier_improvement_pct": -1.0, "effective_n": 100},
            {"tsm_like_calibration_route": "TSM_LIKE_WEIGHTED_PLATT", "split": "tsm_like_train_validation", "decision_ece": 0.14, "brier_improvement_pct": 0.5, "effective_n": 600},
            {"tsm_like_calibration_route": "TSM_LIKE_WEIGHTED_PLATT", "split": "tsm_like_combined_test_holdout", "decision_ece": 0.02, "brier_improvement_pct": 4.0, "effective_n": 100},
        ]
    )

    route, summary = choose_tsm_like_route(metrics)

    assert route == "TSM_DIRECT_ONLY"
    assert summary["route_selection_provenance_valid"].map(bool).all()


def test_rank_percentile_policy_keeps_fold_selection_fraction_stable():
    threshold_rows = []
    test_rows = []
    for i in range(100):
        high = i >= 70
        common = {
            "symbol": f"S{i % 20}",
            "symbol_group": "semi",
            "date": pd.Timestamp("2023-07-01"),
            "signal_idx": i,
            "score_price_algo_total": i,
            "p_success_model": 0.2 + i / 200.0,
            "p_stop_hit_lgbm": 0.5 - i / 400.0,
            "expected_r_lgbm": -0.5 + i / 80.0,
            "label_stop_survival_20d": 1 if high else 0,
            "label_net_return_pct_20d": 4.0 if high else -2.0,
            "label_success_20d": int(high),
        }
        threshold_rows.append(common)
        test_common = dict(common)
        test_common["date"] = pd.Timestamp("2024-07-01")
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

    assert records["applied_threshold_policy_type"].iloc[0] == "fold_rank_percentile_policy"
    assert records["risk_adjusted_selection_score_col"].iloc[0] == "p_success_model"
    assert 0.30 <= records["selected_by_threshold"].mean() <= 0.60


def test_threshold_stability_merges_undersized_oof_fold_diagnostic():
    metrics = pd.DataFrame(
        [
            {"split": "oof_test_2021", "event_count": 590, "selected_event_count": 207, "selected_fraction": 0.350, "threshold": 0.80, "applied_threshold_policy_type": "fold_rank_percentile_policy", "target_selected_fraction": 0.35},
            {"split": "oof_test_2022", "event_count": 33, "selected_event_count": 25, "selected_fraction": 0.758, "threshold": 0.20, "applied_threshold_policy_type": "fold_rank_percentile_policy", "target_selected_fraction": 0.35},
            {"split": "oof_test_2023", "event_count": 346, "selected_event_count": 122, "selected_fraction": 0.353, "threshold": 0.02, "applied_threshold_policy_type": "fold_rank_percentile_policy", "target_selected_fraction": 0.35},
            {"split": "oof_test_2024", "event_count": 525, "selected_event_count": 184, "selected_fraction": 0.350, "threshold": 0.49, "applied_threshold_policy_type": "fold_rank_percentile_policy", "target_selected_fraction": 0.35},
            {"split": "oof_test_2025_2026", "event_count": 656, "selected_event_count": 230, "selected_fraction": 0.351, "threshold": 0.24, "applied_threshold_policy_type": "fold_rank_percentile_policy", "target_selected_fraction": 0.35},
        ]
    )

    stable, weak_folds, threshold_iqr = threshold_stability_from_fold_metrics(metrics)

    assert stable is True
    assert "oof_test_2022:undersized_fold_events_33_merged_for_stability" in weak_folds
    assert "selected_fraction_out_of_range" not in weak_folds
    assert threshold_iqr == 0.0


def test_paper_gate_never_enables_live_and_can_pass_without_strict_gate():
    comparison = pd.DataFrame(
        [
            {
                "model_name": "pooled_stack_calibrated",
                "split": "combined_test_holdout",
                "evaluation_scope": "trade_ready_entry_only",
                "event_count": 300,
                "selected_event_count": 120,
                "selected_mean_return_pct": 1.5,
                "mean_return_pct": 1.0,
                "selected_minus_all_ci_lower_pct_paired": -0.10,
                "decision_ece": 0.10,
                "brier_improvement_pct": 1.0,
                "fold_selected_count_min": 30,
            }
        ]
    )
    tsm_like_metrics = pd.DataFrame(
        [
            {
                "tsm_like_calibration_route": "TSM_LIKE_WEIGHTED_LOGIT_SHIFT",
                "split": "tsm_like_train_validation",
                "selection_effective_n": 700,
                "selection_decision_ece": 0.12,
                "tsm_like_route_selection_pass": True,
                "is_selected_tsm_like_route": True,
            }
        ]
    )
    overlay = {
        "decision_support_allowed": False,
        "latest_trade_ready": True,
        "p_stop_hit_20d": 0.30,
    }

    snapshot = build_paper_gate_snapshot(comparison, tsm_like_metrics, overlay, pooled_dataset_quality_ok=True)
    values = dict(zip(snapshot["field"], snapshot["value"]))

    assert values["paper_decision_support_allowed"] is True
    assert values["prediction_ready"] is False
    assert values["live_ready"] is False
    assert values["live_trading_status"] == "DISABLED_BY_DESIGN"
