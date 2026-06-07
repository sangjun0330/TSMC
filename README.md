# Top10+2 반도체 멀티 타임프레임 Meta-Labeling + Risk Sizing 패키지

이 패키지는 `config/semiconductor_universe_top10.csv`의 Top10+2 운영 종목을 기준으로 두고, 일봉/시간봉/5분봉/1분봉 데이터에서 가격/변동성/추세/유동성/상대강도 지표를 만듭니다. 현재 운영 대상은 기존 Top10에 삼성전자(`005930.KS`)와 SK hynix(`000660.KS`)를 추가한 12종목입니다. Universal research pool은 학습·보정·cross-sectional 진단에만 사용하고, risk/intent/paper execution/dashboard 판단 후보는 운영 유니버스로 필터링합니다. 목표는 주가 자체를 직접 예측하는 것이 아니라 “룰상 후보가 실거래 기대값을 갖는가”를 보수적으로 판단하는 것입니다.

## 1. 설치

```bash
python3 -m pip install -r requirements.txt
```

## 2. 실행

데스크톱 프로그램 창으로 실행하려면 다음 명령을 사용합니다.

```bash
./launch_dashboard.sh
```

대시보드는 브라우저 주소를 직접 열지 않고 별도 앱 창으로 뜹니다. 내부적으로만 loopback 서버를 쓰며, 대시보드에서 최신 판단, 매매 계획, 가격/백테스트/리스크/예측/검증 그래프, Markdown 리포트, CSV 산출물 미리보기, 전체/부분 파이프라인 재실행을 모두 사용할 수 있습니다.

macOS `.app` 번들을 다시 만들고 실행하려면 다음 명령을 사용합니다.

```bash
./script/build_and_run.sh
```

이 명령은 `dist/Top10 Dashboard.app`을 만들고 실행합니다. 이후에는 Finder에서 해당 앱을 더블클릭하면 됩니다.

로컬 웹 서버를 직접 열어 디버깅해야 할 때만 다음 명령을 사용합니다.

```bash
.venv/bin/python tsm_dashboard.py --host 127.0.0.1 --port 8765
```

```bash
python3 tsm_daily_quant_pipeline.py --start 2016-05-12 --end 2026-05-12 --outdir output
```

기본값은 Stooq CSV 데이터를 먼저 사용하고, 실패하면 Yahoo Chart JSON으로 대체합니다.

```bash
python3 tsm_daily_quant_pipeline.py --preferred-source yahoo --start 2016-05-12 --end 2026-05-12 --outdir output
```

벤치마크 상대강도/베타 계산이 필요 없으면 다음처럼 실행합니다.

```bash
python3 tsm_daily_quant_pipeline.py --skip-benchmarks --start 2016-05-12 --end 2026-05-12 --outdir output
```

룰 엔진, 백테스트, 최신 매매 계획까지 이어서 생성하려면 다음 순서로 실행합니다.

```bash
python3 run_daily_update.py \
  --start 2016-05-12 \
  --end 2026-05-12 \
  --output-dir output \
  --rule-outdir tsm_price_rule_output
```

이미 `output/*.csv`가 있고 후속 판단/검증 산출물만 다시 만들려면 다음처럼 실행합니다.

```bash
python3 run_daily_update.py \
  --skip-data-refresh \
  --end 2026-05-12 \
  --output-dir output \
  --rule-outdir tsm_price_rule_output
```

Top10+2 전체의 시간봉 소스 점검과 공개로 확보 가능한 시간봉 CSV를 만들려면 다음처럼 실행합니다.

```bash
.venv/bin/python run_universe_market_data_update.py \
  --universe-config config/semiconductor_universe_top10.csv \
  --bar-scope hourly \
  --start 2016-05-12 \
  --end 2026-05-12 \
  --outdir output \
  --provider auto \
  --skip-charts \
  --continue-on-error
```

주의: Yahoo 공개 chart API는 1시간봉 요청이 최근 730일 안에 있어야 하므로, 인증 없이 2016년부터의 완전한 10년 시간봉은 내려받을 수 없습니다. 스크립트는 `output/tsm_universe_intraday_update_manifest.csv`와 `output/tsm_universe_market_data_latest.csv`에 이 제한과 종목별 커버리지를 기록하고, 공개로 확보 가능한 최대 범위를 `output/universe/<SYMBOL>/tsm_hourly_available_raw.csv`와 `output/universe/<SYMBOL>/tsm_hourly_available_enriched.csv`로 저장합니다. `POLYGON_API_KEY`가 환경변수에 있으면 `--provider auto`가 Polygon 1시간봉을 먼저 시도합니다.

Top10+2 전체의 5분봉 모델 피처와 1분봉 실행/슬리피지 확인 CSV를 만들려면 다음처럼 실행합니다.

```bash
.venv/bin/python run_universe_market_data_update.py \
  --universe-config config/semiconductor_universe_top10.csv \
  --bar-scope minute \
  --model-minute-interval 5m \
  --execution-minute-interval 1m \
  --start 2016-05-12 \
  --end 2026-05-12 \
  --outdir output \
  --provider auto \
  --skip-charts \
  --continue-on-error
```

주의: Yahoo 공개 chart API는 `1m`은 약 8일, `2m/5m/15m/30m`은 최근 60일, `60m/1h`는 최근 730일 제한을 반환합니다. 이 스크립트는 Top10 각 종목의 `output/universe/<SYMBOL>/tsm_5min_available_*.csv`와 `output/universe/<SYMBOL>/tsm_minute_available_*.csv`를 만들고, coverage 부족은 품질/게이트 진단에 반영합니다.

각 단계를 수동으로 실행하려면 다음 순서를 사용합니다.

```bash
python3 tsm_price_rule_engine.py \
  --enriched output/tsm_daily_10y_enriched.csv \
  --raw output/tsm_daily_10y_raw.csv \
  --summary output/tsm_daily_10y_summary.csv \
  --events output/tsm_event_impact_10y.csv \
  --outdir tsm_price_rule_output

python3 tsm_backtest_engine.py \
  --signals tsm_price_rule_output/tsm_daily_algorithmic_signals.csv \
  --enriched output/tsm_daily_10y_enriched.csv \
  --outdir tsm_price_rule_output

python3 tsm_risk_engine.py \
  --signals tsm_price_rule_output/tsm_daily_algorithmic_signals.csv \
  --outdir tsm_price_rule_output

python3 tsm_validation_engine.py \
  --signals tsm_price_rule_output/tsm_daily_algorithmic_signals.csv \
  --enriched output/tsm_daily_10y_enriched.csv \
  --outdir tsm_price_rule_output

python3 tsm_daily_stress_engine.py \
  --signals tsm_price_rule_output/tsm_daily_algorithmic_signals.csv \
  --risk-policy tsm_price_rule_output/tsm_risk_policy_daily.csv \
  --drawdowns tsm_price_rule_output/tsm_drawdown_episodes.csv \
  --equity-curves tsm_price_rule_output/tsm_backtest_equity_curves.csv \
  --outdir tsm_price_rule_output

python3 tsm_daily_integrity_engine.py \
  --raw output/tsm_daily_10y_raw.csv \
  --enriched output/tsm_daily_10y_enriched.csv \
  --signals tsm_price_rule_output/tsm_daily_algorithmic_signals.csv \
  --risk-policy tsm_price_rule_output/tsm_risk_policy_daily.csv \
  --trade-log tsm_price_rule_output/tsm_backtest_trade_log.csv \
  --equity-curves tsm_price_rule_output/tsm_backtest_equity_curves.csv \
  --outdir tsm_price_rule_output

python3 tsm_prediction_engine.py \
  --signals tsm_price_rule_output/tsm_daily_algorithmic_signals.csv \
  --trade-log tsm_price_rule_output/tsm_backtest_trade_log.csv \
  --risk-policy tsm_price_rule_output/tsm_risk_policy_daily.csv \
  --enriched output/tsm_daily_10y_enriched.csv \
  --outdir tsm_price_rule_output

python3 tsm_next_day_up_model_engine.py \
  --signals tsm_price_rule_output/tsm_daily_algorithmic_signals.csv \
  --trade-log tsm_price_rule_output/tsm_backtest_trade_log.csv \
  --risk-policy tsm_price_rule_output/tsm_risk_policy_daily.csv \
  --enriched output/tsm_daily_10y_enriched.csv \
  --external-features tsm_price_rule_output/tsm_external_daily_features.csv \
  --intraday-features tsm_price_rule_output/tsm_intraday_daily_features.csv \
  --outdir tsm_price_rule_output

python3 tsm_ml_overlay_backtest.py \
  --oos-predictions tsm_price_rule_output/tsm_prediction_oos_predictions.csv \
  --outdir tsm_price_rule_output

python3 tsm_pooled_dataset_builder.py \
  --outdir tsm_price_rule_output

python3 tsm_next_close_forecast_engine.py \
  --pooled-feature-matrix tsm_price_rule_output/tsm_prediction_pooled_feature_matrix.csv \
  --decision-universe-config config/semiconductor_universe_top10.csv \
  --latest-prediction tsm_price_rule_output/tsm_latest_prediction_snapshot.csv \
  --outdir tsm_price_rule_output

python3 tsm_model_registry_engine.py \
  --outdir tsm_price_rule_output

python3 tsm_shadow_paper_engine.py \
  --prediction tsm_price_rule_output/tsm_latest_prediction_snapshot.csv \
  --signals tsm_price_rule_output/tsm_daily_algorithmic_signals.csv \
  --ledger tsm_price_rule_output/tsm_shadow_paper_predictions.csv \
  --outdir tsm_price_rule_output

python3 tsm_system_state_engine.py \
  --operational-quality tsm_price_rule_output/tsm_operational_quality_checks.csv \
  --integrity-quality tsm_price_rule_output/tsm_daily_integrity_checks.csv \
  --validation-quality tsm_price_rule_output/tsm_validation_quality_checks.csv \
  --prediction-quality tsm_price_rule_output/tsm_prediction_quality_checks.csv \
  --prediction-snapshot tsm_price_rule_output/tsm_latest_prediction_snapshot.csv \
  --risk-snapshot tsm_price_rule_output/tsm_latest_risk_snapshot.csv \
  --stress-snapshot tsm_price_rule_output/tsm_latest_stress_snapshot.csv \
  --backtest-summary tsm_price_rule_output/tsm_backtest_strategy_summary.csv \
  --walk-forward tsm_price_rule_output/tsm_validation_walk_forward_summary.csv \
  --outdir tsm_price_rule_output

python3 tsm_daily_trading_report.py \
  --latest tsm_price_rule_output/tsm_latest_decision_snapshot.csv \
  --signals tsm_price_rule_output/tsm_daily_algorithmic_signals.csv \
  --prediction tsm_price_rule_output/tsm_latest_prediction_snapshot.csv \
  --outdir tsm_price_rule_output
```

## 3. 생성 파일

| 파일 | 설명 |
|---|---|
| `output/tsm_daily_10y_raw.csv` | 원시 일봉 OHLCV 데이터 |
| `output/tsm_daily_10y_enriched.csv` | 수익률, 상승/하락폭, 변동성, ATR, 이동평균, Drawdown, 유동성, 상대강도까지 포함한 핵심 분석 CSV |
| `output/tsm_daily_10y_summary.csv` | 10년 요약 성과: 총수익률, CAGR, 연율화 변동성, MDD, Sharpe, Sortino 등 |
| `output/tsm_event_impact_10y.csv` | 이벤트 전후 1/2/3/5/10/20/60일 수익률 분석 |
| `output/tsm_daily_columns_dictionary.csv` | 주요 컬럼 설명서 |
| `output/charts/tsm_price_ma_drawdown.png` | 종가 + 20/50/200일선 + 고점 대비 낙폭 |
| `output/charts/tsm_rolling_vol_atr.png` | 20/63/252일 연율화 변동성 + ATR14 |
| `output/charts/tsm_daily_return_distribution.png` | 일별 수익률 분포 |
| `output/charts/tsm_daily_return_volume.png` | 일별 수익률 막대 + 20일 평균 거래량 |
| `output/charts/tsm_daily_move_heatmap.png` | 연/월별 평균 절대 일간 변동폭 히트맵 |
| `output/charts/tsm_event_impact.png` | 이벤트 당일 및 이벤트 후 수익률 그래프 |

Top10+2 시간봉 추가 생성 파일:

| 파일 | 설명 |
|---|---|
| `output/universe/<SYMBOL>/tsm_hourly_available_raw.csv` | Top10+2 종목별 일봉 raw와 같은 컬럼 순서의 시간봉 OHLCV 데이터 |
| `output/universe/<SYMBOL>/tsm_hourly_available_enriched.csv` | Top10+2 종목별 일봉 enriched와 같은 지표 스키마의 시간봉 분석 CSV |
| `output/universe/<SYMBOL>/tsm_hourly_available_summary.csv` | Top10+2 종목별 시간봉 확보 범위, 수익률, CAGR, 변동성, MDD 등 요약 |
| `output/universe/<SYMBOL>/tsm_hourly_10y_source_audit.csv` | Top10+2 종목별 Yahoo/Stooq/Polygon/Alpha Vantage/EODHD 기준 10년 시간봉 확보 가능성 점검 |
| `output/tsm_universe_intraday_update_manifest.csv` | Top10+2 종목별 시간/분봉 갱신 실행 기록 |
| `output/tsm_universe_market_data_latest.csv` | Top10+2 종목별 일/시간/분봉 최신성·커버리지 요약 |

Top10+2 분봉 추가 생성 파일:

| 파일 | 설명 |
|---|---|
| `output/universe/<SYMBOL>/tsm_5min_available_raw.csv` | Top10+2 종목별 5분봉 모델 입력 OHLCV 데이터 |
| `output/universe/<SYMBOL>/tsm_5min_available_enriched.csv` | Top10+2 종목별 5분봉 모델 피처용 지표 CSV |
| `output/universe/<SYMBOL>/tsm_minute_available_raw.csv` | Top10+2 종목별 1분봉 paper execution/slippage 확인용 OHLCV 데이터 |
| `output/universe/<SYMBOL>/tsm_minute_available_enriched.csv` | Top10+2 종목별 1분봉 실행 확인용 지표 CSV |
| `output/universe/<SYMBOL>/tsm_minute_10y_source_audit.csv` | Top10+2 종목별 Yahoo/Stooq/Polygon/Alpha Vantage/EODHD 기준 10년 분봉 확보 가능성 점검 |

룰 엔진/백테스트 추가 생성 파일:

| 파일 | 설명 |
|---|---|
| `tsm_price_rule_output/tsm_daily_algorithmic_signals.csv` | 모든 거래일별 알고리즘 점수, 진입 트리거, 행동, 손절가, 목표가 |
| `tsm_price_rule_output/tsm_latest_decision_snapshot.csv` | 최신 거래일 기준 매수/대기/축소 판단용 핵심 값 |
| `tsm_price_rule_output/tsm_backtest_strategy_summary.csv` | diagnostic/full sizing 전략과 live-like/risk sizing 전략별 총수익률, CAGR, MDD, Sharpe, Sortino, Calmar, 승률, Profit Factor |
| `tsm_price_rule_output/tsm_backtest_trade_log.csv` | 전략별 실제 진입/청산 거래 로그. live-like 전략은 partial exit와 aggregate exit를 함께 기록 |
| `tsm_price_rule_output/tsm_backtest_equity_curves.csv` | 전략별 날짜별 equity curve, drawdown, 포지션 비중 |
| `tsm_price_rule_output/tsm_backtest_yearly_returns.csv` | 전략별 연도별 수익률 |
| `tsm_price_rule_output/tsm_backtest_report.md` | 백테스트 사람이 읽는 요약 리포트 |
| `tsm_price_rule_output/tsm_risk_policy_daily.csv` | 날짜별 변동성/추세/낙폭/점수/계좌위험 기반 최종 권장 최대 비중 |
| `tsm_price_rule_output/tsm_latest_risk_snapshot.csv` | 최신 거래일 기준 리스크 엔진 스냅샷 |
| `tsm_price_rule_output/tsm_risk_policy_report.md` | 리스크 정책 요약 리포트 |
| `tsm_price_rule_output/tsm_validation_segment_summary.csv` | 고정 장세 구간별 전략 성과 |
| `tsm_price_rule_output/tsm_validation_walk_forward_summary.csv` | 롤링 train/test 워크포워드식 검증 |
| `tsm_price_rule_output/tsm_validation_causal_walk_forward_summary.csv` | fold별로 룰 신호를 재계산하는 causal walk-forward 검증 |
| `tsm_price_rule_output/tsm_validation_parameter_sensitivity.csv` | ATR 손절 배수와 점수 기준 민감도 분석 |
| `tsm_price_rule_output/tsm_model_trials_log.csv` | 테스트한 전략/파라미터 trial 로그 |
| `tsm_price_rule_output/tsm_pbo_report.csv` | trial 수와 성과 분산을 기록하는 PBO proxy 리포트 |
| `tsm_price_rule_output/tsm_deflated_sharpe_report.csv` | multiple testing을 감안한 PSR/DSR 리포트 |
| `tsm_price_rule_output/tsm_validation_quality_checks.csv` | 룩어헤드, 마지막 행 체결, 비용 반영, 포지션 중복 검증 |
| `tsm_price_rule_output/tsm_validation_report.md` | 전략 신뢰도 검증 요약 리포트 |
| `tsm_price_rule_output/tsm_daily_stress_scenarios.csv` | 현재 비중 기준 일봉 충격/낙폭/ATR 스트레스 시나리오 |
| `tsm_price_rule_output/tsm_strategy_stress_summary.csv` | 전략별 최악 1일/20일 수익률과 MDD |
| `tsm_price_rule_output/tsm_latest_stress_snapshot.csv` | 최신 스트레스 상태 요약 |
| `tsm_price_rule_output/tsm_daily_stress_report.md` | 일봉 스트레스 리포트 |
| `tsm_price_rule_output/tsm_daily_integrity_checks.csv` | OHLC, 날짜 정렬, 수식, 리스크 비중, 거래 로그, equity curve 계약 검증 |
| `tsm_price_rule_output/tsm_latest_integrity_snapshot.csv` | 최신 무결성 검증 통과/실패 요약 |
| `tsm_price_rule_output/tsm_daily_integrity_report.md` | 무결성 검증 Markdown 리포트 |
| `tsm_price_rule_output/tsm_prediction_label_dataset.csv` | 기존 룰 신호 후보의 20/60거래일 success, stop-hit, 1R/2R, expected-R, MFE/MAE 메타 라벨 |
| `tsm_price_rule_output/tsm_prediction_feature_matrix.csv` | 예측 모델용 allowlist 피처와 라벨 결합 데이터 |
| `tsm_price_rule_output/tsm_prediction_candidate_scope_stats.csv` | 예측 후보 범위/진입 게이트/트리거/행동별 성공률, 순수익률, 손절률 기준 통계 |
| `tsm_price_rule_output/tsm_prediction_label_diagnostics.csv` | 20/60일 triple-barrier 라벨별 손절 회피, 1R/2R 도달, 양수 수익 진단 |
| `tsm_price_rule_output/tsm_prediction_feature_selection_report.csv` | fold별 결측/분산/상관 필터와 train-only target association으로 선택·제외된 피처 감사 로그 |
| `tsm_price_rule_output/tsm_prediction_feature_association_summary.csv` | 선택·제외된 피처의 train-only association 요약. 제외된 고연관 피처는 다음 검증 실험 후보로만 사용 |
| `tsm_price_rule_output/tsm_prediction_walk_forward_metrics.csv` | 예측 모델별 walk-forward OOS Brier, log loss, PR AUC, 기대값 |
| `tsm_price_rule_output/tsm_prediction_oos_predictions.csv` | fold별 OOS 예측 원장. 날짜, 확률, threshold 선택 여부, 실제 라벨을 보존 |
| `tsm_price_rule_output/tsm_prediction_model_comparison.csv` | `trade_ready_entry`, `trigger_all`, `context_all`을 분리한 예측 모델별 OOS 품질/경제성 비교 |
| `tsm_price_rule_output/tsm_prediction_model_audit.csv` | 모델별 Brier decomposition, threshold 안정성, 품질 차단 사유 |
| `tsm_price_rule_output/tsm_prediction_brier_decomposition_summary.csv` | 모델별 Brier reliability/resolution/uncertainty 분해와 다음 calibration/action 진단 |
| `tsm_price_rule_output/tsm_prediction_calibration_bins.csv` | 확률 구간별 예측확률과 실제 성공률 calibration 표 |
| `tsm_price_rule_output/tsm_prediction_calibration_summary.csv` | fixed-width/equal-frequency calibration별 ECE와 bin 커버리지 요약 |
| `tsm_price_rule_output/tsm_prediction_threshold_policy.csv` | fold별 validation 구간에서 선택된 확률 임계값 |
| `tsm_price_rule_output/tsm_latest_prediction_snapshot.csv` | 최신 거래일 기준 20/60일 성공확률, 임계값, 사용 가능 상태 |
| `tsm_price_rule_output/tsm_prediction_quality_checks.csv` | 예측 라벨/피처/룩어헤드/threshold/품질 체크 |
| `tsm_price_rule_output/tsm_prediction_report.md` | 예측 정확도와 메타-라벨 품질 요약 리포트 |
| `tsm_price_rule_output/tsm_prediction_reliability_report.md` | 최신 2단계 확률, 80% 확률 구간, reliability bin 리포트 |
| `tsm_price_rule_output/tsm_next_day_up_label_dataset.csv` | 오늘 종가 대비 다음 거래일 종가 상승 여부(`label_success_1d`) close-to-close 라벨 |
| `tsm_price_rule_output/tsm_next_day_up_feature_matrix.csv` | 1일 상승 예측용 기존 allowlist 피처와 close-to-close 라벨 결합 데이터 |
| `tsm_price_rule_output/tsm_next_day_up_model_comparison.csv` | 1일 상승 모델별 walk-forward OOS Brier, PR AUC, calibration, threshold 성과 |
| `tsm_price_rule_output/tsm_next_day_up_oos_predictions.csv` | 1일 상승 모델 fold별 OOS 확률 원장 |
| `tsm_price_rule_output/tsm_next_day_up_latest_snapshot.csv` | 최신 거래일 기준 `next_day_p_up_1d`, threshold, 상태 요약. 기존 최신 예측 스냅샷에도 `next_day_*` 필드로 병합 |
| `tsm_price_rule_output/tsm_next_day_up_quality_checks.csv` | 1일 상승 라벨/피처/워크포워드/threshold 품질 체크 |
| `tsm_price_rule_output/tsm_next_day_up_report.md` | 1일 상승 모델 설계와 최신 확률 요약 리포트 |
| `tsm_price_rule_output/tsm_next_close_label_dataset.csv` | Top12 전체의 1D/5D/20D 종가 direct forecast 라벨. 목표값은 `log(close_engine[t+h] / close_engine[t])` |
| `tsm_price_rule_output/tsm_next_close_feature_matrix.csv` | pooled feature matrix에 종가 forecast 라벨을 붙인 학습 입력. 일봉, 외부, hourly, m5/m1 daily-aligned intraday 피처를 포함 |
| `tsm_price_rule_output/tsm_next_close_feature_selection_report.csv` | fold별 train-only 피처 선택 로그. 결측이 과도한 intraday 피처는 제외하고 충분한 coverage 피처는 유지 |
| `tsm_price_rule_output/tsm_next_close_model_comparison.csv` | horizon별 baseline/ElasticNet/HGBR/LightGBM/XGBoost/validation blend의 OOS MAE, RMSE, 방향 적중, 구간 coverage, 성공 gate |
| `tsm_price_rule_output/tsm_next_close_oos_predictions.csv` | fold별 OOS 예상 log return, 예상 종가(engine/USD/native), 80% 구간, 실제 종가 비교 원장 |
| `tsm_price_rule_output/tsm_next_close_interval_calibration.csv` | 모델별 80% 구간 coverage와 평균 구간 폭 요약 |
| `tsm_price_rule_output/tsm_next_close_universe_latest_predictions.csv` | Top12 최신 1D/5D/20D 예상 종가, 예상 수익률, 80% 구간, horizon별 품질 상태 |
| `tsm_price_rule_output/tsm_next_close_latest_snapshot.csv` | TSM 최신 forecast snapshot. 기존 최신 예측 스냅샷에도 `next_close_*` 필드로 병합 |
| `tsm_price_rule_output/tsm_next_close_quality_checks.csv` | forecast 라벨, 누수 방지, OOS 순서, Top12 최신 row, success gate 품질 체크 |
| `tsm_price_rule_output/tsm_next_close_report.md` | 1D/5D/20D 예상 종가 모델 품질과 최신 TSM forecast 요약 |
| `tsm_price_rule_output/tsm_ml_overlay_summary.csv` | 룰 전체 OOS 이벤트와 ML threshold 필터링 이벤트의 수익률/성공률 비교 |
| `tsm_price_rule_output/tsm_ml_overlay_equity_curves.csv` | ML overlay별 이벤트 가중 equity curve와 drawdown |
| `tsm_price_rule_output/tsm_ml_overlay_quality_checks.csv` | overlay 백테스트 산출물 계약 검증 |
| `tsm_price_rule_output/tsm_prediction_pooled_label_dataset.csv` | Universal research pool 기반 symbol 포함 라벨 데이터셋. 최종 판단 후보는 Top10으로 제한 |
| `tsm_price_rule_output/tsm_prediction_pooled_feature_matrix.csv` | 멀티 심볼 확장을 위한 symbol 포함 피처 행렬 |
| `tsm_price_rule_output/tsm_prediction_pooled_schema.csv` | pooled dataset 컬럼 role, dtype, 결측률 스키마 |
| `tsm_price_rule_output/tsm_prediction_model_registry.csv` | 모델/스코프/호라이즌별 품질, calibration, overlay, source hash를 묶은 registry |

예측 실험 variant를 baseline 산출물과 비교해 OOS 악화 여부를 자동 판정하려면 다음처럼 실행합니다.

```bash
.venv/bin/python tsm_prediction_experiment_guardrail.py \
  --baseline-dir tsm_price_rule_output \
  --variant-dir /tmp/tsm_variant_candidate \
  --target-scope entry_research \
  --target-model elastic_net_logistic \
  --out-csv /tmp/tsm_prediction_experiment_guardrail.csv
```
| `tsm_price_rule_output/tsm_prediction_experiment_log.csv` | registry 기반 실험 상태와 차단 사유 로그 |
| `tsm_price_rule_output/tsm_shadow_paper_predictions.csv` | 최신 예측을 날짜/호라이즌별로 기록하고 미래 창이 생기면 realized label을 채우는 섀도우 원장 |
| `tsm_price_rule_output/tsm_shadow_paper_quality_checks.csv` | 섀도우 원장 중복/상태/날짜 계약 검증 |
| `tsm_price_rule_output/tsm_system_readiness_scorecard.csv` | 데이터/검증/리스크/스트레스/워크포워드 기반 준비도 점수표 |
| `tsm_price_rule_output/tsm_latest_system_state.csv` | 최신 시스템 상태와 페이퍼/라이브 상태 |
| `tsm_price_rule_output/tsm_system_readiness_report.md` | 시스템 준비도 리포트 |
| `tsm_price_rule_output/tsm_daily_trading_plan.csv` | 최신 실전 매매표 |
| `tsm_price_rule_output/tsm_daily_trading_plan.md` | 최신 실전 매매 계획 Markdown |
| `tsm_price_rule_output/tsm_daily_update_manifest.csv` | 자동 업데이트 단계별 실행 명령, 상태, 시간, stdout/stderr tail |
| `tsm_price_rule_output/tsm_operational_quality_checks.csv` | 파일 존재, 날짜, 중복, raw/enriched 검증, 백테스트 품질 체크 |
| `tsm_price_rule_output/tsm_daily_update_operational_report.md` | 운영 상태와 품질 체크 요약 |

예측 엔진은 최신 행을 세 단계로 분리합니다.

- `trade_ready_entry`: `entry_trigger != NONE`이고 기존 룰 엔진이 `ENTRY_ALLOWED`로 실제 진입을 허용한 후보
- `trigger_all`: 진입 트리거는 발생했지만 점수, 변동성, 과이격 등으로 필터링된 후보까지 포함
- `context_all`: 트리거가 없는 `HOLD_OR_WAIT_TRIGGER`, `WATCHLIST_PULLBACK_ONLY` 관찰 상태까지 포함

최신 스냅샷의 `prediction_signal_status`와 `prediction_use_status`는 이 구분을 반영합니다. 예측값은 기존 `trade_action`을 뒤집지 않으며, `DECISION_SUPPORT_ALLOWED`가 아니면 표시 전용입니다.

예측 v2는 `p_stop_survival * p_positive_given_survival = p_success`의 2단계 확률을 유지하면서 `p_stop_hit`, `p_hit_1r`, `p_hit_2r`, `expected_r`, `expected_net_return`을 함께 표시합니다. `effective_oos_event_count`가 100 미만이거나 Brier/ECE/PR AUC/기대값 기준을 통과하지 못하면 최신 예측은 자동으로 표시 전용입니다.

별도 1일 방향성 모델인 `tsm_next_day_up_model_engine.py`는 오늘 종가 대비 다음 거래일 종가가 상승할 확률(`next_day_p_up_1d`)을 계산합니다. 이 모델은 close-to-close 방향성 진단용이며 기존 20D trade-ready 의사결정 게이트나 live trading 상태를 켜지 않습니다.

현재 ML 개선 단계는 다음 흐름으로 고정되어 있습니다.

- Phase 0: 라벨/피처 누수 차단, calibration 선택 보수화, 최신 no-signal 표시 분리
- Phase 1: OOS 예측 원장과 fixed-width/equal-frequency calibration 요약 생성
- Phase 2: ML threshold overlay가 룰 전체 이벤트보다 실제 수익률을 개선하는지 백테스트
- Phase 3: 심볼 컬럼이 있는 pooled dataset 계약을 만들어 멀티 심볼 학습 확장 준비
- Phase 4: 모델 registry와 experiment log로 모델 품질, source hash, 차단 사유 추적
- Phase 5: 섀도우 페이퍼 원장으로 최신 예측을 누적하고 미래 20/60거래일 라벨을 사후 확정

## 4. 핵심 컬럼 예시

- `close_change_usd`: 전일 종가 대비 달러 상승/하락폭
- `close_change_pct`: 전일 종가 대비 상승률/하락률
- `open_gap_pct`: 전일 종가 대비 당일 시가 갭
- `open_to_close_pct`: 당일 시가 대비 종가 변화율
- `intraday_range_pct_prev_close`: 고가-저가 일중 변동폭 / 전일 종가
- `vol_20d_ann`, `vol_63d_ann`, `vol_252d_ann`: 연율화 변동성
- `atr_14`, `atr_14_pct`: 14일 ATR과 종가 대비 ATR 비율
- `drawdown_from_ath`: 누적 고점 대비 현재 낙폭
- `max_drawdown_252d`: 최근 252거래일 최대낙폭
- `breakout_20d_high`, `breakout_60d_high`: 이전 고점 돌파 여부
- `beta_vs_spy_252d`, `relative_return_vs_smh_63d`: 벤치마크 대비 베타/상대수익률
- `atr14_stop_long_k_2_0`: ATR x 2 기준 롱 포지션 손절가

## 5. 백테스트 가정

- 신호는 당일 종가 기준으로 생성합니다.
- 진입과 이동평균 청산은 다음 거래일 시가에 체결합니다.
- ATR 손절은 장중 저가가 손절가를 터치하면 당일 손절가에 체결합니다. 단, 시가가 손절가 아래로 갭 하락하면 시가 체결로 처리합니다.
- 기본 거래비용은 진입/청산 각각 수수료 1bp + 슬리피지 5bp입니다. `tsm_backtest_engine.py`의 `--commission-bps`, `--slippage-bps`로 조정할 수 있습니다.
- 배당, 세금, 환율, ADR 수수료는 1차 백테스트에서 제외합니다.

## 6. 신뢰도 검증 원칙

- 전체 10년 누적수익률 하나만으로 전략을 판단하지 않습니다.
- `tsm_validation_engine.py`는 고정 장세 구간, 워크포워드식 train/test 구간, 손절 배수/점수 기준 민감도를 함께 출력합니다.
- `tsm_validation_engine.py`는 전체 기간 신호를 잘라 쓰는 검증과 별도로 fold마다 causal rule engine을 재계산하는 검증, prefix stability, PBO proxy, DSR도 출력합니다.
- 파라미터 민감도 분석은 최고값을 고르기 위한 최적화가 아니라, 규칙이 작은 변경에도 무너지지 않는지 확인하기 위한 안정성 점검입니다.
- 거래비용을 뺀 결과만 보지 않고, 기본 비용 포함 결과를 기준으로 판단합니다.
- 리스크 엔진은 신호 엔진과 분리되어 있으며, 변동성/200일선/고점 대비 낙폭/점수/계좌위험 제한 중 가장 보수적인 비중을 최종 권장 최대 비중으로 사용합니다.
- 스트레스 엔진은 현재 최종 권장 비중 기준으로 과거 최악 일봉, 과거 낙폭 구간, ATR 충격이 포트폴리오에 주는 영향을 추정합니다.
- 무결성 엔진은 백테스트/리포트와 독립적으로 핵심 수식, 날짜 정렬, 다음날 체결 계약, 비용 반영, 포지션 중복, 리스크 비중 범위를 다시 검사합니다.
- 예측 엔진은 가격 자체가 아니라 기존 룰 신호 후보가 20/60거래일 안에 2ATR 손절을 피하고 비용 차감 후 성공할 확률을 추정합니다.
- 예측 엔진은 `entry_trigger != NONE`인 실제 진입 후보와 `WATCHLIST/HOLD` 같은 관찰 컨텍스트를 분리합니다.
- 예측 피처는 신호일에 이미 존재하는 trailing 일봉 지표만 allowlist로 사용하며, enriched 파일의 변동성/밴드/거래량/낙폭/상대강도 지표를 추가로 반영합니다.
- 예측 품질은 승률 하나가 아니라 Brier score, log loss, calibration error, PR AUC, OOS 기대값 개선으로 판단합니다.
- 예측 결과는 `trade_action`을 자동 변경하지 않는 보조 계층이며, 품질 기준 미달 시 표시 전용으로 둡니다.
- 시스템 준비도 엔진은 운영 품질, 무결성, 검증 품질, 예측 품질, 리스크 상태, 스트레스 상태, 전체기간 성과, 워크포워드 근거를 100점으로 점수화합니다.

## 7. 운영 원칙

- `run_daily_update.py`는 실제 주문을 넣지 않습니다. CSV/Markdown 리포트만 갱신합니다.
- 장 마감 후 실행을 전제로 설계되어 있으며, 최신 거래일이 실행일보다 7일 이상 오래되면 품질 체크에서 경고합니다.
- 모든 실행 단계는 `tsm_daily_update_manifest.csv`에 기록됩니다.
- 품질 체크가 실패하면 `tsm_daily_update_operational_report.md`에서 어떤 검사가 실패했는지 먼저 확인하세요.
- LaunchAgent/Codex 일일 자동화의 최종 전체 점검은 `tsm_full_daily_update_audit.csv`와 `tsm_full_daily_update_audit.md`에 기록됩니다. 이 파일은 일봉, 시간봉, 분봉, 뉴스, pooled universe, 시스템 상태, paper OMS 산출물을 한 번에 확인합니다.
- 네트워크 또는 데이터 제공처 장애가 있을 때는 `--skip-data-refresh`로 기존 CSV 기준 후속 리포트만 재생성할 수 있습니다.
- 라이브 주문은 명시적으로 비활성화되어 있습니다. 시스템 상태 파일의 `live_trading_status`도 `DISABLED_BY_DESIGN`으로 고정됩니다.

## 8. 이벤트 파일 수정

`tsm_events_seed.csv`를 수정하면 이벤트 분석 CSV와 이벤트 그래프에 반영됩니다. 컬럼은 아래 형식을 유지하세요.

```csv
event_date,event_name,event_type,source_url,notes
2024-04-03,Taiwan earthquake,operational,https://...,notes
```

## 9. 뉴스 원인 자동 수집

API 키 없이 공개 웹/RSS와 반도체 관련 공개 보도자료를 직접 수집해 가격 변동일과 매칭하려면 다음을 실행합니다.

```bash
python3 tsm_news_causal_engine.py \
  --mode backfill \
  --start 2016-05-12 \
  --end 2026-05-17 \
  --outdir output \
  --rule-outdir tsm_price_rule_output
```

일일 업데이트에는 기본으로 뉴스 수집 단계가 포함됩니다. 매일 07:30 로컬 시간 실행용 LaunchAgent는 다음 명령으로 설치합니다.

```bash
./script/install_daily_launch_agent.sh
```

LaunchAgent 실행 스크립트는 `config/semiconductor_universe_top10.csv`의 enabled 종목 전체에 대해 일봉 연구 산출물과 `output/universe/<SYMBOL>/tsm_hourly_available_*.csv`, `output/universe/<SYMBOL>/tsm_minute_available_*.csv`를 함께 갱신합니다. 확장 후보 전체는 `config/semiconductor_universe_expanded.csv`에 보존되어 있습니다. 유니버스 시간/분봉만 수동 갱신하려면 다음을 실행합니다.

```bash
.venv/bin/python run_universe_market_data_update.py \
  --universe-config config/semiconductor_universe_top10.csv \
  --start 2016-05-12 \
  --end "$(date '+%Y-%m-%d')" \
  --bar-scope both \
  --skip-charts
```

## 10. 주의

- Stooq 데이터는 명시적 `Adj Close` 컬럼이 없는 경우 `adj_close = close`로 처리합니다.
- 배당/액면분할을 엄밀히 반영한 총수익률 백테스트가 필요하면 브로커/유료 데이터의 corporate actions 자료로 검증하세요.
- 이 패키지는 교육 및 리서치용입니다. 매수/매도 추천이 아닙니다.
