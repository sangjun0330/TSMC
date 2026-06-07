from pathlib import Path

import pandas as pd

from tsm_prediction_experiment_guardrail import build_guardrail_report


def write_prediction_outputs(
    root: Path,
    comparison: pd.DataFrame,
    near_pass: pd.DataFrame,
    gap: pd.DataFrame,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(root / "tsm_prediction_model_comparison.csv", index=False)
    near_pass.to_csv(root / "tsm_prediction_near_pass_candidates.csv", index=False)
    gap.to_csv(root / "tsm_prediction_performance_gap_summary.csv", index=False)


def comparison_row(
    *,
    model_name: str = "elastic_net_logistic",
    prediction_quality_pass: bool = False,
    brier_improvement_pct: float = -1.0,
    decision_ece: float = 0.09,
    expectancy_improvement_pct: float = 0.2,
    selected_signal_expectancy_ci_lower_pct: float = 0.4,
    rank_score: float = -0.5,
    threshold_iqr: float = 0.05,
) -> dict:
    return {
        "candidate_scope": "entry_research",
        "horizon_days": 20,
        "model_name": model_name,
        "prediction_quality_pass": prediction_quality_pass,
        "brier_improvement_pct": brier_improvement_pct,
        "decision_ece": decision_ece,
        "pr_auc": 0.55,
        "expectancy_improvement_pct": expectancy_improvement_pct,
        "selected_signal_expectancy_ci_lower_pct": selected_signal_expectancy_ci_lower_pct,
        "rank_score": rank_score,
        "threshold_iqr": threshold_iqr,
        "selected_oos_event_count": 80,
        "min_selected_events_per_fold": 20,
    }


def near_pass_frame(one_block_count: int = 1) -> pd.DataFrame:
    rows = [
        {
            "candidate_scope": "entry_research",
            "horizon_days": 20,
            "model_name": f"model_{idx}",
            "performance_block_count": 1,
        }
        for idx in range(one_block_count)
    ]
    return pd.DataFrame(rows, columns=["candidate_scope", "horizon_days", "model_name", "performance_block_count"])


def gap_frame(brier_candidates: int = 2, brier_gap: float = 3.0, ece_candidates: int = 1, ece_gap: float = 0.05) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "performance_block_reason": "NO_BRIER_IMPROVEMENT",
                "candidate_count": brier_candidates,
                "max_brier_gap_pct": brier_gap,
                "max_ece_gap": 0.0,
            },
            {
                "performance_block_reason": "ECE_GT_0_10",
                "candidate_count": ece_candidates,
                "max_brier_gap_pct": 0.0,
                "max_ece_gap": ece_gap,
            },
        ]
    )


def test_guardrail_rejects_target_oos_metric_regression(tmp_path):
    baseline = tmp_path / "baseline"
    variant = tmp_path / "variant"
    write_prediction_outputs(
        baseline,
        pd.DataFrame([comparison_row(), comparison_row(model_name="empirical_bayes_group_rate", prediction_quality_pass=True, brier_improvement_pct=0.5, rank_score=999.0)]),
        near_pass_frame(one_block_count=2),
        gap_frame(brier_candidates=2, brier_gap=3.0, ece_candidates=1, ece_gap=0.05),
    )
    write_prediction_outputs(
        variant,
        pd.DataFrame([comparison_row(brier_improvement_pct=-2.0, decision_ece=0.12, expectancy_improvement_pct=0.1), comparison_row(model_name="empirical_bayes_group_rate", prediction_quality_pass=True, brier_improvement_pct=0.5, rank_score=999.0)]),
        near_pass_frame(one_block_count=1),
        gap_frame(brier_candidates=3, brier_gap=5.0, ece_candidates=2, ece_gap=0.08),
    )

    report = build_guardrail_report(baseline, variant, target_scope="entry_research", target_horizon=20, target_model="elastic_net_logistic")
    values = dict(zip(report["check"], report["passed"]))
    verdict = dict(zip(report["check"], report["variant_value"]))["guardrail_verdict"]

    assert not values["target_brier_improvement_pct_not_worse"]
    assert not values["target_decision_ece_not_worse"]
    assert not values["one_block_near_pass_count_not_lower_without_new_pass"]
    assert verdict == "REJECT"


def test_guardrail_accepts_target_improvement_without_gap_expansion(tmp_path):
    baseline = tmp_path / "baseline"
    variant = tmp_path / "variant"
    write_prediction_outputs(
        baseline,
        pd.DataFrame([comparison_row(), comparison_row(model_name="empirical_bayes_group_rate", prediction_quality_pass=True, brier_improvement_pct=0.5, rank_score=999.0)]),
        near_pass_frame(one_block_count=1),
        gap_frame(brier_candidates=2, brier_gap=3.0, ece_candidates=1, ece_gap=0.05),
    )
    write_prediction_outputs(
        variant,
        pd.DataFrame([comparison_row(brier_improvement_pct=-0.2, decision_ece=0.08, expectancy_improvement_pct=0.3, selected_signal_expectancy_ci_lower_pct=0.6, rank_score=-0.1), comparison_row(model_name="empirical_bayes_group_rate", prediction_quality_pass=True, brier_improvement_pct=0.5, rank_score=999.0)]),
        near_pass_frame(one_block_count=1),
        gap_frame(brier_candidates=1, brier_gap=2.0, ece_candidates=1, ece_gap=0.04),
    )

    report = build_guardrail_report(baseline, variant, target_scope="entry_research", target_horizon=20, target_model="elastic_net_logistic")
    verdict = dict(zip(report["check"], report["variant_value"]))["guardrail_verdict"]

    assert verdict == "ACCEPT"


def test_guardrail_allows_threshold_iqr_increase_when_gate_pass_count_improves(tmp_path):
    baseline = tmp_path / "baseline"
    variant = tmp_path / "variant"
    write_prediction_outputs(
        baseline,
        pd.DataFrame([comparison_row(threshold_iqr=0.0), comparison_row(model_name="empirical_bayes_group_rate", prediction_quality_pass=True, brier_improvement_pct=0.5, rank_score=999.0)]),
        near_pass_frame(one_block_count=2),
        gap_frame(brier_candidates=2, brier_gap=3.0, ece_candidates=1, ece_gap=0.05),
    )
    write_prediction_outputs(
        variant,
        pd.DataFrame([comparison_row(prediction_quality_pass=True, brier_improvement_pct=0.1, decision_ece=0.08, expectancy_improvement_pct=0.2, selected_signal_expectancy_ci_lower_pct=0.4, rank_score=1000.0, threshold_iqr=0.09), comparison_row(model_name="empirical_bayes_group_rate", prediction_quality_pass=True, brier_improvement_pct=0.5, rank_score=999.0)]),
        near_pass_frame(one_block_count=1),
        gap_frame(brier_candidates=1, brier_gap=2.0, ece_candidates=1, ece_gap=0.04),
    )

    report = build_guardrail_report(baseline, variant, target_scope="entry_research", target_horizon=20, target_model="elastic_net_logistic")
    values = dict(zip(report["check"], report["passed"]))
    verdict = dict(zip(report["check"], report["variant_value"]))["guardrail_verdict"]

    assert values["target_threshold_iqr_not_worse"]
    assert verdict == "ACCEPT"


def test_guardrail_allows_unchanged_threshold_iqr_even_when_above_gate(tmp_path):
    baseline = tmp_path / "baseline"
    variant = tmp_path / "variant"
    write_prediction_outputs(
        baseline,
        pd.DataFrame([comparison_row(threshold_iqr=0.12), comparison_row(model_name="empirical_bayes_group_rate", prediction_quality_pass=True, brier_improvement_pct=0.5, rank_score=999.0)]),
        near_pass_frame(one_block_count=1),
        gap_frame(brier_candidates=2, brier_gap=3.0, ece_candidates=1, ece_gap=0.05),
    )
    write_prediction_outputs(
        variant,
        pd.DataFrame([comparison_row(threshold_iqr=0.12), comparison_row(model_name="empirical_bayes_group_rate", prediction_quality_pass=True, brier_improvement_pct=0.5, rank_score=999.0)]),
        near_pass_frame(one_block_count=1),
        gap_frame(brier_candidates=2, brier_gap=3.0, ece_candidates=1, ece_gap=0.05),
    )

    report = build_guardrail_report(baseline, variant, target_scope="entry_research", target_horizon=20, target_model="elastic_net_logistic")
    values = dict(zip(report["check"], report["passed"]))
    verdict = dict(zip(report["check"], report["variant_value"]))["guardrail_verdict"]

    assert values["target_threshold_iqr_not_worse"]
    assert verdict == "ACCEPT"
