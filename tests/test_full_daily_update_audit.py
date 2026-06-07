import pandas as pd

import tsm_full_daily_update_audit as audit
from tsm_full_daily_update_audit import inventory_paths, metadata_row


def test_model_gate_audit_inventory_uses_critical_failures_only(tmp_path):
    root = tmp_path
    outdir = root / "tsm_price_rule_output"
    outdir.mkdir()
    audit_path = outdir / "tsm_model_gate_audit.csv"
    pd.DataFrame(
        [
            {
                "gate_group": "local_prediction_diagnostic",
                "gate": "prediction_quality_pass",
                "passed": False,
                "severity": "WARN",
                "block_reason": "PREDICTION_QUALITY_FALSE",
            }
        ]
    ).to_csv(audit_path, index=False)

    row = metadata_row(root, "tsm_model_gate_audit", "tsm_price_rule_output/tsm_model_gate_audit.csv", "inventory", False)

    assert row["status"] == "PASS"
    assert row["passed"] is True


def test_optional_large_inventory_csv_uses_metadata_without_parsing(tmp_path, monkeypatch):
    root = tmp_path
    outdir = root / "tsm_price_rule_output"
    outdir.mkdir()
    path = outdir / "large_optional.csv"
    path.write_text("not,a,valid\nunterminated", encoding="utf-8")
    monkeypatch.setattr(audit, "OPTIONAL_CSV_READ_LIMIT_BYTES", 1)

    row = metadata_row(root, "large_optional", "tsm_price_rule_output/large_optional.csv", "inventory", False)

    assert row["status"] == "PRESENT"
    assert row["passed"] is True
    assert row["rows"] == ""
    assert row["details"] == "csv_read_skipped_size_bytes_gt_1"


def test_performance_gate_audit_required_row_passes_diagnostic_only_failures(tmp_path):
    root = tmp_path
    outdir = root / "tsm_price_rule_output"
    outdir.mkdir()
    path = outdir / "tsm_model_performance_gate_audit.csv"
    pd.DataFrame(
        [
            {
                "gate_group": "local_prediction_diagnostic",
                "gate": "brier_improvement_positive",
                "passed": False,
                "severity": "WARN",
                "block_reason": "NO_BRIER_IMPROVEMENT",
                "performance_gate_scope": "diagnostic",
                "active_performance_gate": False,
            }
        ]
    ).to_csv(path, index=False)

    row = metadata_row(root, "tsm_model_performance_gate_audit", "tsm_price_rule_output/tsm_model_performance_gate_audit.csv", "system", True)

    assert row["status"] == "PASS"
    assert row["passed"] is True
    assert row["details"] == "active_performance_failures=0;diagnostic_rows=1"


def test_performance_gate_audit_required_row_fails_active_performance_failures(tmp_path):
    root = tmp_path
    outdir = root / "tsm_price_rule_output"
    outdir.mkdir()
    path = outdir / "tsm_model_performance_gate_audit.csv"
    pd.DataFrame(
        [
            {
                "gate_group": "pooled_model",
                "gate": "brier_improvement_positive",
                "passed": False,
                "severity": "CRITICAL",
                "block_reason": "NO_BRIER_IMPROVEMENT",
                "performance_gate_scope": "active",
                "active_performance_gate": True,
            }
        ]
    ).to_csv(path, index=False)

    row = metadata_row(root, "tsm_model_performance_gate_audit", "tsm_price_rule_output/tsm_model_performance_gate_audit.csv", "system", True)

    assert row["status"] == "FAIL"
    assert row["passed"] is False
    assert "pooled_model:brier_improvement_positive:NO_BRIER_IMPROVEMENT" in row["details"]


def test_performance_gate_audit_is_required_inventory(tmp_path):
    rows = inventory_paths(tmp_path, tmp_path / "output", tmp_path / "tsm_price_rule_output")
    match = [row for row in rows if row[0] == "tsm_model_performance_gate_audit"]

    assert match == [("tsm_model_performance_gate_audit", "tsm_price_rule_output/tsm_model_performance_gate_audit.csv", "system", True)]
