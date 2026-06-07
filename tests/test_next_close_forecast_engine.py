from __future__ import annotations

import numpy as np
import pandas as pd

import tsm_next_close_forecast_engine as engine
from tsm_model_gate_engine import evaluate_next_close_forecast


SYMBOLS = ["NVDA", "TSM", "AVGO", "AMD", "INTC", "MU", "TXN", "LRCX", "AMAT", "QCOM", "005930.KS", "000660.KS"]


def test_next_close_labels_use_log_close_return_and_exclude_unmatured_rows():
    pooled = pd.DataFrame(
        {
            "symbol": ["TSM", "TSM", "TSM"],
            "date": pd.date_range("2024-01-01", periods=3, freq="B"),
            "signal_idx": [0, 1, 2],
            "close": [100.0, 105.0, 103.0],
            "close_usd": [100.0, 105.0, 103.0],
            "close_native": [100.0, 105.0, 103.0],
        }
    )

    labels = engine.build_next_close_label_dataset(pooled, [1, 2])

    assert labels.loc[0, "label_status_1d"] == "LABELED"
    assert np.isclose(labels.loc[0, "label_close_return_log_1d"], np.log(105.0 / 100.0))
    assert np.isclose(labels.loc[0, "label_close_return_pct_1d"], 5.0)
    assert labels.loc[2, "label_status_1d"] == "UNAVAILABLE_FUTURE_WINDOW"
    assert labels.loc[1, "label_status_2d"] == "UNAVAILABLE_FUTURE_WINDOW"
    assert np.isclose(labels.loc[0, "label_target_close_engine_2d"], 103.0)


def test_next_close_krw_native_usd_engine_conversion_is_consistent():
    pooled = pd.DataFrame(
        {
            "symbol": ["005930.KS", "005930.KS"],
            "date": pd.date_range("2024-01-01", periods=2, freq="B"),
            "close": [55.0, 57.2],
            "close_native": [75000.0, 78000.0],
            "fx_rate_to_usd": [55.0 / 75000.0, 57.2 / 78000.0],
            "display_currency": ["KRW", "KRW"],
        }
    )

    labels = engine.build_next_close_label_dataset(pooled, [1])

    assert np.isclose(labels.loc[0, "label_close_return_log_1d"], np.log(57.2 / 55.0))
    assert np.isclose(labels.loc[0, "label_target_close_native_1d"], 78000.0)
    assert np.isclose(labels.loc[0, "label_target_close_usd_1d"], 57.2)


def test_next_close_feature_selection_blocks_future_and_label_columns():
    train = pd.DataFrame(
        {
            "symbol": ["TSM"] * 40,
            "date": pd.date_range("2024-01-01", periods=40, freq="B"),
            "score_price_algo_total": np.arange(40),
            "label_close_return_log_1d": np.linspace(-0.01, 0.01, 40),
            "future_magic": np.arange(40) * 2,
            "next_close_predicted_return_pct_1d": np.arange(40),
            "actual_return_hint": np.arange(40),
        }
    )

    selected, report = engine.select_fold_features(train, "label_close_return_log_1d", max_numeric_features=20)

    assert "score_price_algo_total" in selected
    assert "future_magic" not in selected
    assert "next_close_predicted_return_pct_1d" not in selected
    assert "actual_return_hint" not in selected
    assert report.loc[report["feature"].eq("future_magic"), "reason"].iloc[0] == "future_or_label_feature_blocked"


def test_next_close_intraday_sparse_features_are_fold_selected_by_train_coverage():
    train = pd.DataFrame(
        {
            "symbol": ["TSM"] * 100,
            "date": pd.date_range("2024-01-01", periods=100, freq="B"),
            "label_close_return_log_1d": np.linspace(-0.01, 0.01, 100),
            "hourly_enough": [np.nan] * 80 + list(np.linspace(0.0, 1.0, 20)),
            "hourly_too_sparse": [np.nan] * 99 + [1.0],
            "daily_too_sparse": [np.nan] * 80 + list(np.linspace(0.0, 1.0, 20)),
        }
    )

    selected, report = engine.select_fold_features(train, "label_close_return_log_1d", max_numeric_features=20)

    assert "hourly_enough" in selected
    assert "hourly_too_sparse" not in selected
    assert "daily_too_sparse" not in selected
    assert report.loc[report["feature"].eq("hourly_too_sparse"), "reason"].iloc[0] == "too_sparse"


def _synthetic_top12_frame() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2020-01-01", periods=150, freq="B")
    for sidx, symbol in enumerate(SYMBOLS):
        is_krw = symbol.endswith(".KS")
        close = 100.0 + sidx * 5.0
        for idx, date in enumerate(dates):
            alpha = np.sin((idx + sidx) / 9.0)
            drift = 0.0015 * alpha + 0.0002 * (sidx % 3)
            close *= float(np.exp(drift))
            native = close / 0.00075 if is_krw else close
            rows.append(
                {
                    "symbol": symbol,
                    "symbol_group": "memory_storage" if symbol in {"MU", "005930.KS", "000660.KS"} else "fabless_ai_analog",
                    "date": date,
                    "signal_idx": idx,
                    "close": close,
                    "close_usd": close,
                    "close_native": native,
                    "display_currency": "KRW" if is_krw else "USD",
                    "fx_rate_to_usd": 0.00075 if is_krw else 1.0,
                    "atr_14_pct": 0.02,
                    "score_price_algo_total": 50.0 + alpha * 20.0,
                    "trend_regime": "UP" if alpha > 0 else "DOWN",
                    "vol_regime": "NORMAL",
                    "hourly_return_20bar": alpha,
                    "model_minute_return_20bar": alpha * 0.5,
                    "execution_minute_minutes_since_bar": idx % 20,
                }
            )
    return pd.DataFrame(rows)


def test_next_close_synthetic_top12_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "LGBMRegressor", None)
    monkeypatch.setattr(engine, "XGBRegressor", None)
    pooled = _synthetic_top12_frame()
    pooled_path = tmp_path / "pooled.csv"
    config_path = tmp_path / "universe.csv"
    latest_path = tmp_path / "latest.csv"
    outdir = tmp_path / "out"
    pooled.to_csv(pooled_path, index=False)
    pd.DataFrame({"symbol": SYMBOLS, "enabled": [True] * len(SYMBOLS)}).to_csv(config_path, index=False)
    pd.DataFrame([{"field": "prediction_use_status", "value": "DISPLAY_ONLY"}]).to_csv(latest_path, index=False)

    artifacts = engine.build_next_close_artifacts(
        pooled_feature_matrix_path=pooled_path,
        decision_universe_config_path=config_path,
        latest_prediction_path=latest_path,
        outdir=outdir,
        horizons=[1, 5, 20],
        train_days=45,
        validation_days=10,
        test_days=10,
        step_days=20,
        max_numeric_features=20,
        max_categorical_features=5,
        min_oos_events=20,
        update_latest=True,
        symbol="TSM",
    )
    engine.write_next_close_artifacts(outdir, artifacts)

    universe_latest = artifacts["universe_latest"]
    comparison = artifacts["comparison"]
    quality = artifacts["quality"]
    assert isinstance(universe_latest, pd.DataFrame)
    assert isinstance(comparison, pd.DataFrame)
    assert isinstance(quality, pd.DataFrame)
    assert set(universe_latest["symbol"]) == set(SYMBOLS)
    assert {"next_close_predicted_close_1d", "next_close_predicted_return_pct_5d", "next_close_lower_80_20d"}.issubset(universe_latest.columns)
    assert set(comparison["horizon_days"]) == {1, 5, 20}
    assert (outdir / "tsm_next_close_universe_latest_predictions.csv").exists()
    assert bool(quality[quality["check"].eq("next_close_selected_features_have_no_forbidden_columns")]["passed"].iloc[0])


def test_latest_next_close_champion_skips_market_wide_constant_baselines():
    comparison = pd.DataFrame(
        [
            {
                "horizon_days": 1,
                "model_name": "market_recent_1d_mean_sign_mag_0018",
                "rank_score": 10.0,
                "mae_log_return": 0.01,
            },
            {
                "horizon_days": 1,
                "model_name": "validation_weighted_blend",
                "rank_score": 9.0,
                "mae_log_return": 0.011,
            },
            {
                "horizon_days": 1,
                "model_name": "symbol_all_median_shrink_085",
                "rank_score": 2.0,
                "mae_log_return": 0.012,
            },
            {
                "horizon_days": 5,
                "model_name": "market_recent_1d_mean_shrink_010",
                "rank_score": 9.0,
                "mae_log_return": 0.02,
            },
            {
                "horizon_days": 5,
                "model_name": "recent_symbol_mean_return",
                "rank_score": 1.0,
                "mae_log_return": 0.021,
            },
        ]
    )

    champions = engine.latest_champion_rows(comparison)
    by_horizon = {int(row["horizon_days"]): row["model_name"] for _, row in champions.iterrows()}

    assert by_horizon[1] == "symbol_all_median_shrink_085"
    assert by_horizon[5] == "recent_symbol_mean_return"


def test_model_gate_adds_nonblocking_next_close_forecast_rows():
    comparison = pd.DataFrame(
        [
            {
                "candidate_scope": "next_close_forecast_top12",
                "horizon_days": 1,
                "model_name": "validation_weighted_blend",
                "rank_score": 1.0,
                "mae_log_return": 0.01,
                "oos_event_count": 3000,
                "fold_count": 3,
                "mae_improvement_pct": 0.5,
                "rmse_delta_vs_naive": 0.0,
                "interval_coverage_80": 0.79,
                "direction_accuracy": 0.51,
                "direction_threshold": 0.52,
                "performance_quality_pass": False,
                "quality_block_reasons": "MAE_IMPROVEMENT_LT_1PCT",
            }
        ]
    )
    quality = pd.DataFrame([{"check": "next_close_model_comparison_non_empty", "passed": True, "severity": "CRITICAL"}])
    latest = {"next_close_model_quality_status": "DISPLAY_ONLY_FORECAST_MODEL_QUALITY_NOT_PASSED"}

    rows = pd.DataFrame(evaluate_next_close_forecast(comparison, quality, latest))

    assert "next_close_forecast" in set(rows["gate_group"])
    assert set(rows["severity"]) == {"WARN"}
    assert not rows.loc[rows["gate"].eq("all_horizons_promoted_diagnostic"), "passed"].iloc[0]
