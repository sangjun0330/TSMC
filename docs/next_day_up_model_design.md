# Next-Day Up Prediction Model Design

## 목적

기존 `tsm_prediction_engine.py`는 가격 자체가 아니라 룰 기반 후보의 5/10/20/40/60/120거래일 메타라벨 성공확률을 추정한다. 사용자가 요청한 모델은 이 계약과 다르게, 오늘 장 마감까지 관측 가능한 데이터로 다음 거래일 종가가 오늘 종가보다 상승할 확률을 추정하는 1거래일 방향성 모델이다.

따라서 새 모델은 기존 20D/60D 의사결정 게이트를 대체하지 않고, 별도 진단/보조 예측으로 운영한다.

## 웹 리서치 요약

- Krauss, Do, Huck(2017)은 S&P 500 종목의 일별 one-day-ahead 확률 예측을 만들고, 가장 높은 확률/낮은 확률 구간만 거래하는 방식으로 신호를 평가했다. 설계 시 확률 랭킹, OOS 평가, 거래비용 전후 해석을 분리해야 한다. Source: https://ideas.repec.org/a/eee/ejores/v259y2017i2p689-702.html
- Ghosh, Neufeld, Sahoo(2021)는 다음 방향성 예측에서 종가 수익률뿐 아니라 시가/장중 수익률 피처를 함께 쓰는 multi-feature 설정을 제시했다. 현재 저장소의 `open_gap_pct`, `open_to_close_pct`, intraday feature 세트와 호환된다. Source: https://arxiv.org/abs/2004.10178
- Malla et al.(2026)은 one-step-ahead daily log-return 예측을 walk-forward validation으로 평가하고, R-squared보다 방향 정확도와 OOS 오차를 중시했다. 1D 모델도 고정 holdout이 아니라 기존 purged walk-forward 흐름을 재사용한다. Source: https://arxiv.org/abs/2601.08896
- Lopez de Prado의 금융 ML 목차는 fixed horizon labeling, triple barrier, sample uniqueness, purged K-Fold CV를 핵심 검증 항목으로 둔다. 기존 엔진도 이 철학을 이미 반영하므로 1D 모델도 train-only feature selection, embargo, leakage check를 유지한다. Source: https://www.oreilly.com/library/view/advances-in-financial/9781119482086/ftoc.xhtml
- scikit-learn 문서는 확률 예측 품질을 calibration curve, Brier score, log loss 관점에서 점검하고, calibration 데이터가 모델 학습 데이터와 분리되어야 한다고 설명한다. 따라서 1D 모델은 `p_up_1d`를 hard class가 아니라 calibration된 확률로 출력한다. Source: https://scikit-learn.org/stable/modules/calibration.html

## 현재 시스템 분석

- `tsm_prediction_engine.py`
  - 입력: `tsm_daily_algorithmic_signals.csv`, `tsm_risk_policy_daily.csv`, `tsm_backtest_trade_log.csv`, 선택적 enriched/external/intraday feature.
  - 라벨: 이벤트 후보에 대해 다음 세션 open 진입, horizon close/ATR stop 기반 성공 여부.
  - 모델: empirical/group base rate, score logistic, elastic-net logistic, multi-timeframe overlay 등.
  - 검증: `PurgedEventTimeSplit`, validation threshold, calibration bins, Brier/log loss/PR AUC, 품질 차단 사유.
  - 최신: `tsm_latest_prediction_snapshot.csv`에 `p_success_20d`, `p_success_60d` 등 field/value 형태로 기록.
- `tsm_pooled_model_engine.py`
  - pooled universe 기반 20D trade-ready 모델 전용.
  - 최신 스냅샷에 `pooled_*` 필드를 병합하고, strict 조건이 통과할 때만 20D 의사결정 필드를 덮어쓴다.
- `tsm_shadow_paper_engine.py`
  - 기존 `HORIZONS`의 메타라벨 예측만 원장화한다.
  - 1D 방향성은 진입/손절 메타라벨이 아니므로 같은 원장에 섞지 않는다.
- `run_daily_update.py`
  - prediction, ML overlay, pooled dataset/model, model gate, registry, shadow paper 순으로 실행한다.
  - 새 1D 엔진은 `prediction_engine` 직후 실행하면 pooled update가 최신 스냅샷의 추가 field를 보존한다.

## 라벨 설계

- 모델명: `next_day_up_1d`.
- 스코프: `next_day_up_all`.
- 학습 대상: 모든 일봉 행 중 다음 거래일 종가가 존재하는 행.
- 라벨:
  - `label_success_1d = 1` if `close[t+1] > close[t]`, else `0`.
  - `label_positive_return_1d`는 동일 값.
  - `label_stop_survival_1d = 1`로 고정해 기존 2-stage 예측 함수를 재사용하되, 실제 의미는 "stop model 없음"으로 문서화한다.
  - `label_net_return_pct_1d = close[t+1] / close[t] - 1`.
  - 마지막 행은 `UNAVAILABLE_FUTURE_WINDOW`.
- 최신 예측:
  - `next_day_p_up_1d`
  - `next_day_p_down_1d`
  - `next_day_threshold_1d`
  - `next_day_prediction_signal_status`
  - `next_day_model_quality_status`

## 피처 설계

기존 allowlist를 그대로 쓴다.

- 가격/추세: `close_change_pct`, `return_20d`, `return_60d`, `momentum_12m_ex_1m`, SMA 거리, RSI, MACD, Bollinger 위치.
- 변동성/리스크: ATR, rolling vol, downside vol, range z-score, risk weight.
- 상대강도/시장: SPY/SMH/QQQ 상대수익률, beta, external market/peer features.
- 뉴스: `news_*`, release 후에만 채워지는 TSMC revenue features.
- intraday: `hourly_*`, `model_minute_*`, `m5_*`, coverage/freshness.

금지:

- `next_`, `future`, `label_`, `fwd_`, `return[t+1]` 같은 미래/라벨 컬럼.
- feature selection은 fold별 train 데이터만 사용한다.

## 검증 및 품질 기준

- Split: 기존 `PurgedEventTimeSplit(horizon_days=1)` 사용. 기본 embargo는 최소 1일이며 CLI `--gap-days`로 보수적으로 확대 가능.
- Metrics:
  - Brier score / base-rate Brier improvement.
  - Log loss.
  - PR AUC와 base-rate PR AUC.
  - Directional accuracy.
  - Calibration ECE와 bin 최소 표본 수.
  - Validation threshold로 선택된 구간의 평균 다음날 수익률.
- 운영 판단:
  - 이 모델은 `DECISION_SUPPORT_ALLOWED`를 켜지 않는다.
  - 품질 통과 시에도 "next-day directional diagnostic"으로만 표시한다.
  - threshold 이상이면 `UP_BIAS`, 미만이면 `NO_UP_EDGE`로 표현한다.

## 구현 계획

1. `tsm_prediction_engine.py`의 scope별 모델 선택에 `next_day_up_all`을 추가한다.
2. 새 파일 `tsm_next_day_up_model_engine.py`를 만든다.
   - 기존 `load_inputs`, feature allowlist, `evaluate_prediction_stream`, `latest_prediction_for_horizon` 재사용.
   - 1D close-to-close 라벨을 별도로 생성.
   - OOS/metrics/calibration/threshold/latest/quality/report 산출.
   - `tsm_latest_prediction_snapshot.csv`가 있으면 `next_day_*` 필드를 병합.
3. `run_daily_update.py`에 `next_day_up_model_engine` 단계를 `prediction_engine` 직후 추가한다.
4. `README.md` 생성 파일 표와 실행 예시에 새 엔진을 추가한다.
5. 테스트를 추가한다.
   - 1D 라벨이 close-to-close 상승을 정확히 계산한다.
   - 마지막 행은 future unavailable이다.
   - 최신 스냅샷 병합이 기존 20D 필드를 보존한다.
   - `evaluate_prediction_stream`이 `next_day_up_all`에서 ML 후보 모델을 생성한다.
