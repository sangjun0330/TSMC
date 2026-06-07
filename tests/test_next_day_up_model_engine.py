import pandas as pd

from tsm_next_day_up_model_engine import (
    NEXT_DAY_CANDIDATE_COL,
    build_next_day_up_label_dataset,
    merge_latest_prediction_snapshot,
)
from tsm_prediction_engine import (
    NEXT_DAY_UP_SCOPE,
    evaluate_prediction_stream,
    model_quality_block_reasons,
    performance_quality_pass_from_reasons,
)


def test_next_day_up_labels_use_close_to_close_direction():
    signals = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="B"),
            "signal_idx": [0, 1, 2],
            "symbol": ["TSM", "TSM", "TSM"],
            "close": [100.0, 102.0, 101.0],
            "high": [101.0, 103.0, 102.0],
            "low": [99.0, 101.0, 100.0],
            "atr_14_pct": [0.02, 0.02, 0.02],
            "entry_trigger": ["NONE", "NONE", "NONE"],
            "trade_action": ["NO_TRADE", "NO_TRADE", "NO_TRADE"],
        }
    )

    labels = build_next_day_up_label_dataset(signals)

    assert labels.loc[0, "label_status_1d"] == "LABELED"
    assert labels.loc[0, "label_success_1d"] == 1
    assert labels.loc[1, "label_success_1d"] == 0
    assert labels.loc[2, "label_status_1d"] == "UNAVAILABLE_FUTURE_WINDOW"
    assert labels.loc[0, "label_entry_price_1d"] == 100.0
    assert labels.loc[0, "label_exit_price_1d"] == 102.0


def test_next_day_up_latest_merge_preserves_existing_prediction_fields(tmp_path):
    latest_path = tmp_path / "latest.csv"
    pd.DataFrame(
        [
            {"field": "p_success_20d", "value": 0.61},
            {"field": "prediction_use_status", "value": "DISPLAY_ONLY_RULE_FILTERED"},
        ]
    ).to_csv(latest_path, index=False)
    next_day = pd.DataFrame(
        [
            {"field": "next_day_p_up_1d", "value": 0.54},
            {"field": "next_day_prediction_signal_status", "value": "UP_BIAS"},
        ]
    )

    assert merge_latest_prediction_snapshot(latest_path, next_day)

    merged = pd.read_csv(latest_path)
    values = dict(zip(merged["field"], merged["value"]))
    assert values["p_success_20d"] == "0.61" or float(values["p_success_20d"]) == 0.61
    assert values["prediction_use_status"] == "DISPLAY_ONLY_RULE_FILTERED"
    assert values["next_day_p_up_1d"] == "0.54" or float(values["next_day_p_up_1d"]) == 0.54
    assert values["next_day_prediction_signal_status"] == "UP_BIAS"


def test_next_day_up_scope_runs_ml_candidate_models():
    rows = []
    for i, date in enumerate(pd.date_range("2020-01-01", periods=260, freq="B")):
        high_score = i % 10 >= 5
        success = int(high_score)
        rows.append(
            {
                "date": date,
                "signal_idx": i,
                NEXT_DAY_CANDIDATE_COL: True,
                "entry_trigger": "NONE",
                "trade_action": "NO_TRADE",
                "trend_regime": "UP" if high_score else "DOWN",
                "vol_regime": "NORMAL",
                "drawdown_bucket": "dd_0_5",
                "entry_gate_status": "NO_ENTRY_TRIGGER",
                "prediction_universe": NEXT_DAY_UP_SCOPE,
                "score_price_algo_total": 80.0 if high_score else 40.0,
                "atr_14_pct": 0.03,
                "risk_pct_2atr": 0.06,
                "label_status_1d": "LABELED",
                "label_success_1d": success,
                "label_stop_survival_1d": 1,
                "label_positive_return_1d": success,
                "label_net_return_pct_1d": 1.0 if success else -1.0,
                "label_expected_r_1d": 0.5 if success else -0.5,
                "label_hit_1r_before_stop_1d": 0,
                "label_hit_2r_before_stop_1d": 0,
                "label_ambiguous_stop_1r_same_day_1d": 0,
                "label_ambiguous_stop_2r_same_day_1d": 0,
                "label_gap_through_stop_1d": 0,
                "label_first_touch_type_1d": "NEXT_CLOSE_UP" if success else "NEXT_CLOSE_DOWN_OR_FLAT",
                "label_exit_reason_1d": "NEXT_CLOSE_UP" if success else "NEXT_CLOSE_DOWN_OR_FLAT",
                "label_entry_date_1d": date,
                "label_exit_date_1d": date + pd.Timedelta(days=1),
                "label_entry_price_1d": 100.0,
                "label_exit_price_1d": 101.0 if success else 99.0,
                "label_stop_price_1d": pd.NA,
            }
        )
    frame = pd.DataFrame(rows)

    metrics, *_ = evaluate_prediction_stream(
        frame,
        1,
        NEXT_DAY_UP_SCOPE,
        NEXT_DAY_CANDIDATE_COL,
        120,
        40,
        40,
        40,
        1,
        [0.50, 0.55],
        20,
        0.50,
        5,
    )

    models = set(metrics["model_name"]) if not metrics.empty else set()
    assert "elastic_net_logistic" in models
    assert "hist_gradient_boosting_fixed" in models


def test_next_day_directional_diagnostic_passes_near_neutral_brier_with_rank_edge():
    row = pd.Series(
        {
            "candidate_scope": NEXT_DAY_UP_SCOPE,
            "horizon_days": 1,
            "model_name": "empirical_bayes_group_rate",
            "decision_scope_eligible": False,
            "oos_event_count": 1728,
            "selected_oos_event_count": 1654,
            "min_selected_events_per_fold": 182,
            "brier_score": 0.250203,
            "base_rate_brier_score": 0.250099,
            "brier_improvement_pct": -0.041617,
            "decision_ece": 0.063102,
            "pr_auc": 0.527287,
            "base_rate_pr_auc": 0.517361,
            "expectancy_improvement_pct": 0.001657,
            "selected_signal_expectancy_ci_lower_pct": 0.105222,
            "positive_expectancy_folds": 6,
            "decision_min_calibration_bin_n": 30,
            "threshold_iqr": 0.019973,
        }
    )

    reasons = model_quality_block_reasons(row)

    assert "NO_BRIER_IMPROVEMENT" not in reasons
    assert performance_quality_pass_from_reasons(reasons)


def test_next_day_directional_diagnostic_rejects_material_brier_loss():
    row = pd.Series(
        {
            "candidate_scope": NEXT_DAY_UP_SCOPE,
            "horizon_days": 1,
            "model_name": "empirical_bayes_group_rate",
            "decision_scope_eligible": False,
            "oos_event_count": 1728,
            "selected_oos_event_count": 1654,
            "min_selected_events_per_fold": 182,
            "brier_score": 0.2510,
            "base_rate_brier_score": 0.250099,
            "brier_improvement_pct": -0.360257,
            "decision_ece": 0.063102,
            "pr_auc": 0.527287,
            "base_rate_pr_auc": 0.517361,
            "expectancy_improvement_pct": 0.001657,
            "selected_signal_expectancy_ci_lower_pct": 0.105222,
            "positive_expectancy_folds": 6,
            "decision_min_calibration_bin_n": 30,
            "threshold_iqr": 0.019973,
        }
    )

    reasons = model_quality_block_reasons(row)

    assert "NO_BRIER_IMPROVEMENT" in reasons
    assert not performance_quality_pass_from_reasons(reasons)
