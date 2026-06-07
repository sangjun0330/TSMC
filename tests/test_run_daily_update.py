import pandas as pd
from pathlib import Path

from run_daily_update import build_data_quality_report


def test_operational_quality_accepts_model_quality_pass_latest_blocked(tmp_path):
    output_dir = tmp_path / "output"
    rule_outdir = tmp_path / "tsm_price_rule_output"
    output_dir.mkdir()
    rule_outdir.mkdir()
    pd.DataFrame(
        [
            {"field": "model_gate_status", "value": "PASS_MODEL_QUALITY_SIGNAL_STANDBY"},
            {"field": "blocking_failed_gate_groups", "value": "PASS"},
            {"field": "blocking_failed_gate_count", "value": 0},
        ]
    ).to_csv(rule_outdir / "tsm_model_gate_snapshot.csv", index=False)

    quality = build_data_quality_report(output_dir, rule_outdir, "2026-05-31", include_system_outputs=False)
    row = quality[quality["check"].eq("model_gate_audit_ran")].iloc[0]

    assert bool(row["passed"]) is True
    assert row["value"] == "PASS_MODEL_QUALITY_SIGNAL_STANDBY"
    assert row["details"] == "PASS"


def test_daily_update_runs_next_close_after_pooled_dataset_before_pooled_model():
    source = (Path(__file__).resolve().parents[1] / "run_daily_update.py").read_text(encoding="utf-8")
    pooled_dataset_idx = source.index('"pooled_dataset_builder"')
    next_close_idx = source.index('"next_close_forecast_engine"')
    pooled_model_idx = source.index('"pooled_model_engine"')

    assert pooled_dataset_idx < next_close_idx < pooled_model_idx
