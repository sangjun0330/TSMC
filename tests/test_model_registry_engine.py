import pandas as pd

from tsm_model_registry_engine import build_pooled_registry, registry_id


def test_blocked_pooled_champion_is_diagnostic_not_promotable():
    comparison = pd.DataFrame(
        [
            {
                "model_name": "pooled_candidate",
                "split": "combined_test_holdout",
                "evaluation_scope": "trade_ready_entry_only",
                "validation_design": "walk_forward_oof",
                "is_champion": True,
                "event_count": 100,
                "selected_event_count": 40,
                "selected_minus_all_pct": -1.0,
                "brier_improvement_pct": 1.0,
                "ece": 0.05,
            }
        ]
    )
    latest = pd.DataFrame(
        [
            {"field": "model_quality_pass", "value": "False"},
            {"field": "latest_signal_pass", "value": "False"},
            {"field": "decision_support_allowed", "value": "False"},
            {"field": "model_quality_block_reasons", "value": "POOLED_UPLIFT_NOT_PASSED"},
        ]
    )
    gate = pd.DataFrame(
        [
            {"field": "pooled_system_quality_pass", "value": "False"},
            {"field": "pooled_prediction_decision_support", "value": "False"},
        ]
    )

    registry = build_pooled_registry(comparison, latest, pd.DataFrame(), pd.DataFrame(), gate, {})
    row = registry.iloc[0]

    assert row["model_policy"] == "POOLED_DIAGNOSTIC_CHAMPION_BLOCKED"
    assert bool(row["diagnostic_champion"]) is True
    assert bool(row["promotable_model"]) is False
    assert bool(row["prediction_quality_pass"]) is False
    assert row["promotion_status"] == "BLOCKED"


def test_pooled_registry_id_includes_split_scope_and_validation_design():
    base = pd.Series(
        {
            "candidate_scope": "pooled_trade_ready_entry",
            "horizon_days": 20,
            "model_name": "pooled_candidate",
            "split": "combined_test_holdout",
            "evaluation_scope": "trade_ready_entry_only",
            "validation_design": "walk_forward_oof",
            "champion_scope": "pooled_trade_ready_20d",
            "generated_at_utc": "2026-05-31T00:00:00+00:00",
        }
    )
    other_split = base.copy()
    other_split["split"] = "test_2024"

    assert registry_id(base) != registry_id(other_split)
