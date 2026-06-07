import argparse

from tsm_core.config import apply_config_defaults, load_run_config


def test_prediction_config_defaults_include_optional_feature_paths():
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-features", default="")
    parser.add_argument("--intraday-features", default="")
    args = parser.parse_args([])

    configured = apply_config_defaults(args, parser, load_run_config("config/tsm_research.toml"))

    assert configured.external_features == "tsm_price_rule_output/tsm_external_daily_features.csv"
    assert configured.intraday_features == "tsm_price_rule_output/tsm_intraday_daily_features.csv"

