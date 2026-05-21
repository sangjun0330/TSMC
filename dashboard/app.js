const state = {
  data: null,
  activeView: "overview",
  activeFile: null,
  lastRunStatus: "IDLE",
  lastRun: null,
};

const THEME_KEY = "tsm-dashboard-theme-v3";

const colors = {
  ink: "#2962ff",
  blue: "#2962ff",
  teal: "#00a3a3",
  green: "#089981",
  amber: "#f59e0b",
  red: "#f23645",
  violet: "#7b61ff",
  gray: "#787b86",
};

const pageTitles = {
  overview: "홈",
  decision: "매매 판단",
  market: "차트",
  data: "데이터·이벤트",
  news: "뉴스 원인",
  prediction: "예측 요약",
  models: "모델·성과",
  diagnostics: "원본 진단",
  backtest: "백테스트",
  risk: "위험",
  quality: "검증",
  system: "시스템",
  outputs: "자료실",
  run: "실행",
};

const runStatusLabels = {
  IDLE: "대기",
  RUNNING: "실행 중",
  PASS: "성공",
  FAIL: "실패",
};

const valueLabels = {
  True: "예",
  False: "아니오",
  true: "예",
  false: "아니오",
  TSM: "TSMC",
  PASS: "문제 없음",
  FAIL: "문제 있음",
  WARN: "주의 필요",
  BLOCKED: "기준 미달",
  NOT_APPLICABLE: "해당 없음",
  PASS_WITH_WARNINGS: "대체로 정상",
  RUNNING: "실행 중",
  IDLE: "대기",
  NONE: "없음",
  NA: "없음",
  "N/A": "없음",
  NO_MATCH: "매칭 없음",
  NO_HIGH_CONFIDENCE_NEWS: "고신뢰 원인 없음",
  NO_CLUSTER: "묶음 없음",
  DIRECT_WEB_OK: "직접 수집 정상",
  DIRECT_WEB_PARTIAL: "직접 수집 일부",
  DIRECT_WEB_FAILED: "직접 수집 실패",
  DIRECT_WEB_EMPTY: "수집 결과 없음",
  CASH: "현금",
  LONG: "보유",
  NO_TRADE: "거래 없음",
  HOLD_OR_WAIT_TRIGGER: "아직 기다림",
  WATCHLIST_PULLBACK_ONLY: "눌림목 관찰",
  ENTRY_ALLOWED: "진입 가능",
  STRICT_LIVE_ENTRY: "엄격 기준 실전 후보",
  STRICT_TRIGGER_WAITING_FOR_SCORE_OR_RISK: "엄격 기준 보류",
  STRICT_NO_ENTRY: "엄격 기준 신규 진입 없음",
  LIVE_ENTRY_ALREADY_ALLOWED: "실전 신호 이미 발생",
  PAPER_BUY_SETUP: "공격형 Paper 후보",
  EARLY_BULLISH_WATCH: "조기 상승 관찰 강화",
  WATCHLIST_ONLY: "관찰 전용",
  RESEARCH_BLOCKED_RISK: "연구 신호 위험 차단",
  NO_RESEARCH_SIGNAL: "연구 신호 없음",
  USE_STRICT_LIVE_SIGNAL: "실전 신호 사용",
  PAPER_TRACK_LONG_SETUP: "Paper 전용 추적",
  WATCH_ONLY_EARLY_BULLISH: "조기 관찰만",
  WATCH_ONLY: "관찰만",
  NO_ACTION: "행동 없음",
  REDUCE_OR_AVOID: "축소/회피",
  PAPER_READY_WAITING_FOR_SIGNAL: "가상 기록 준비됨",
  PAPER_READY_ENTRY_SIGNAL: "가상 기록 후보",
  PAPER_READY_ENTRY_ALLOWED: "가상 기록 가능",
  RESEARCH_BLOCKED_DATA_QUALITY: "데이터 문제로 분석 보류",
  RESEARCH_BLOCKED_INTEGRITY: "데이터 일관성 문제로 분석 보류",
  RESEARCH_BLOCKED_VALIDATION: "검증 부족으로 분석 보류",
  RESEARCH_READY_PREDICTION_BLOCKED: "분석 가능, 예측은 참고용",
  MODEL_QUALITY_PASS_LATEST_BLOCKED: "모델은 통과, 최신 신호는 차단",
  DECISION_SUPPORT_ALLOWED: "판단에 참고 가능",
  STRICT_DECISION_SUPPORT_ALLOWED: "엄격 기준 판단 가능",
  PAPER_DECISION_SUPPORT_ALLOWED: "가상 판단 가능",
  DISPLAY_ONLY_MODEL_BLOCKED: "모델 기준 미달",
  DISPLAY_ONLY_NO_LATEST_TRADE_READY: "최신 trade-ready 아님",
  POOLED_SEMI_DECISION_SUPPORT: "여러 종목 반도체 모델 판단 참고",
  LOCAL_TSM_DIAGNOSTIC: "TSMC 단독 진단값",
  DISPLAY_ONLY: "참고용",
  RESEARCH_READY_ALPHA_NOT_READY: "분석 가능, 매매 우위 부족",
  RESEARCH_READY_NOT_PAPER_READY: "분석 가능, 가상 기록 미준비",
  RESEARCH_ONLY_REVIEW_REQUIRED: "분석용으로만 확인 필요",
  RULE_BASED_READY_PREDICTION_DISPLAY_ONLY: "기본 규칙은 가능, 예측은 참고용",
  PREDICTION_PAPER_ALPHA_READY: "예측 기반 가상 alpha 준비",
  NO_LIVE_BROKER_BY_DESIGN: "실거래 브로커 미사용 설계",
  PREDICTION_CONFIRMED: "예측 확인됨",
  PREDICTION_NEUTRAL: "예측 중립",
  PREDICTION_FILTERED: "예측 기준 미달",
  INSUFFICIENT_DATA: "데이터 부족",
  INSUFFICIENT_OOS_EVIDENCE: "테스트 근거 부족",
  ENTRY_TRIGGER_FILTERED_BY_RULES: "룰 기준에서 매수 후보 제외",
  READY: "준비됨",
  NOT_READY: "준비 안 됨",
  DISABLED_BY_DESIGN: "설계상 미사용",
  WAIT_FOR_TRIGGER: "매수 신호 대기",
  ENTRY_RISK_ALLOWED: "위험 한도 안",
  OBSERVATION_OR_TINY_SIZE_ONLY: "관찰/소액",
  NO_NEW_RISK: "신규 위험 없음",
  NORMAL_RISK_ALLOWED: "위험 한도 안",
  LOW_STRESS_FOR_CURRENT_SIZE: "현재 비중에서 하락 부담 낮음",
  MODERATE_STRESS_FOR_CURRENT_SIZE: "현재 비중에서 하락 부담 중간",
  HIGH_STRESS_FOR_CURRENT_SIZE: "현재 비중에서 하락 부담 높음",
  DISPLAY_ONLY_NO_ENTRY_TRIGGER: "참고용 표시",
  DISPLAY_ONLY_NO_MODEL_CANDIDATE: "예측 대상 아님",
  DISPLAY_ONLY_QUALITY_BLOCKED: "품질 미달이라 참고용",
  DISPLAY_ONLY_QUALITY_NOT_PASSED: "품질 미달이라 참고용",
  DISPLAY_ONLY_RULE_FILTERED: "룰 기준 미달이라 참고용",
  DISPLAY_ONLY_INSUFFICIENT_DATA: "데이터 부족이라 참고용",
  DISPLAY_ONLY_INSUFFICIENT_OOS_EVIDENCE: "테스트 근거 부족이라 참고용",
  NO_TRADE_NO_TRIGGER: "매수 신호 없음",
  NO_TRADE_MODEL_NOT_TRUSTED: "모델 신뢰 부족으로 거래 없음",
  NO_TRADE_STOP_RISK: "손절 위험 과다로 거래 없음",
  NO_TRADE_LOW_EXPECTANCY: "기대값 부족으로 거래 없음",
  NO_TRADE_BELOW_PROBABILITY_THRESHOLD: "확률 기준 미달로 거래 없음",
  NO_SIGNAL: "신호 없음",
  SCORE60_SMALL_OBSERVATION_ONLY: "점수 60대라 소액 관찰만",
  PAPER_RECORD_ONLY: "가상 기록 전용",
  PAPER_ONLY_RULE_BASED: "룰 기반 가상 기록 전용",
  ALPHA_RESEARCH_LONG_ALLOWED: "분석용 매수 허용",
  ALPHA_RESEARCH_SMALL_LONG_ALLOWED: "분석용 소액 매수 허용",
  POOLED_ALPHA_RESEARCH_LONG_ALLOWED: "여러 종목 모델 기준 분석용 매수 허용",
  RULE_BASED_SMALL_OR_PAPER_ONLY: "룰 기반 소액/가상 기록만",
  DECISION_SUPPORT_ONLY: "판단 참고 전용",
  PAPER_LONG_CONFIRMED: "가상 기록 매수 확인",
  PAPER_WATCH_CONFIRMED: "가상 기록 관찰 확인",
  DISPLAY_ONLY_NO_PAPER_TRADE: "가상 기록 대상 아님",
  NOT_EVENT: "이벤트 아님",
  NO_LATEST_EVENT_CANDIDATE: "최신 이벤트 후보 없음",
  NO_MODEL_CANDIDATE: "예측 대상 아님",
  NO_MODEL: "사용할 예측 모델 없음",
  NO_ENTRY_TRIGGER_CONTEXT_ONLY: "매수 신호 없음",
  NO_ENTRY_TRIGGER: "매수 신호 없음",
  HIGH: "높음",
  LOW_NEGATIVE: "낮음/부정",
  LOW_POSITIVE: "낮음/긍정",
  MEDIUM: "중간",
  strong_uptrend: "강한 상승 추세",
  uptrend: "상승 추세",
  downtrend: "하락 추세",
  insufficient_data: "데이터 부족",
  positive_multi_horizon: "다중 기간 긍정",
  mixed: "혼조",
  normal_vol: "일반 변동성",
  high_vol: "높은 변동성",
  extreme_vol: "극단 변동성",
  context_all: "관찰만 하는 상태",
  entry_research: "연구용 확장 후보",
  trigger_all: "매수 신호가 나온 상태",
  trade_ready_entry: "매수 판단 후보",
  entry_research_all: "연구용 확장 전체",
  trade_ready_entry_only: "엄격 매수 후보만",
  decision_trade_ready: "엄격 매수 후보",
  relaxed_trigger_score65: "완화 트리거 후보",
  relaxed_trigger_score60: "확장 트리거 연구 후보",
  setup_context_score65: "설정 관찰 후보",
  risk_blocked_research: "위험 차단 연구 후보",
  CONTEXT_WATCHLIST: "관찰 후보",
  CONTEXT_HIGH_SCORE_WAIT: "고점수 대기 후보",
  ACTIONABLE_ENTRY_FILTERED: "조건을 통과한 매수 후보",
  ACTIONABLE_ENTRY_ALLOWED: "매수 가능 후보",
  NOT_CANDIDATE: "후보 아님",
  none: "없음",
  base_rate_by_trigger_regime: "비슷한 상황의 기본 성공률",
  empirical_bayes_group_rate: "그룹별 과거 성공률 보정",
  pooled_empirical_bayes_group_rate: "Pooled 경험 베이즈",
  pooled_elastic_net_logistic: "Pooled 로지스틱",
  pooled_hist_gradient_boosting: "Pooled 부스팅",
  pooled_lgbm_classifier: "Pooled LightGBM",
  pooled_xgb_classifier: "Pooled XGBoost",
  pooled_stack_calibrated: "Pooled 스택",
  tsm_specific_calibrated_layer: "TSMC 전용 확률 조정",
  rank_percentile_policy: "fold별 상위 분위 선택",
  locked_validation_threshold: "검증 기준값 고정",
  diagnostic_raw_threshold: "원점수 진단 기준",
  TSM_CALIBRATION_LAYER_READY: "TSMC 전용 확률 조정 준비됨",
  TSM_CALIBRATION_LAYER_INSUFFICIENT_SAMPLE: "TSMC 표본 부족으로 조정 보류",
  TSM_DIRECT: "TSM 직접 표본",
  TSM_DIRECT_ONLY: "TSM 직접 보정",
  TSM_LIKE_FOUNDRY_IDM: "TSM 유사 Foundry/IDM",
  TSM_SUPPLY_CHAIN: "TSM 공급망",
  SEMI_BREADTH_REGIME: "반도체 ETF/레짐",
  OTHER_SEMI: "기타 반도체",
  TSM_LIKE_WEIGHTED_PLATT: "TSM-like weighted Platt",
  TSM_LIKE_WEIGHTED_LOGIT_SHIFT: "TSM-like weighted logit shift",
  TSM_LIKE_WEIGHTED_ISOTONIC: "TSM-like weighted isotonic",
  TSM_DIRECT_PLUS_LIKE_SHRINKAGE: "TSM 직접+유사 표본 shrinkage",
  TSM_LIKE_WEIGHTED_ISOTONIC_READY: "TSM-like isotonic 준비됨",
  TSM_LIKE_WEIGHTED_PLATT_READY: "TSM-like Platt 준비됨",
  TSM_LIKE_WEIGHTED_LOGIT_SHIFT_READY: "TSM-like logit shift 준비됨",
  TSM_DIRECT_PLUS_LIKE_SHRINKAGE_ADVISORY: "직접+유사 shrinkage 진단",
  STRICT_TRAINING_ELIGIBLE: "엄격 학습 가능",
  SHORT_HISTORY_RESEARCH_ONLY: "상장 이력 짧아 연구용",
  MISSING_ENRICHED: "가공 데이터 없음",
  SKIPPED_STRICT_INELIGIBLE: "엄격 학습 제외",
  sigmoid_platt_shrunk: "수축 Platt 확률 보정",
  symbol_group_logit_shift: "종목 그룹 확률 조정",
  global_probability_calibration: "전체 확률 보정",
  stacked_calibrated: "스택 보정",
  empirical_bayes: "경험 베이즈",
  linear_logistic: "선형 로지스틱",
  lightgbm: "LightGBM",
  xgboost: "XGBoost",
  hist_gradient_boosting: "히스토그램 부스팅",
  date_block: "월별 블록 bootstrap",
  paired_event: "이벤트 paired bootstrap",
  skipped_diagnostic: "진단 생략",
  score_logistic: "점수 기반 예측",
  logistic_balanced: "균형형 예측",
  elastic_net_logistic: "규제형 예측",
  random_forest_fixed: "나무 묶음 예측",
  hist_gradient_boosting_fixed: "부스팅 예측",
  daily_integrity: "일별 데이터 일관성",
  validation_quality: "검증 결과",
  prediction_quality: "예측 상태",
  prediction_decision_support: "예측을 판단에 참고할 수 있는지",
  alpha_research: "수익 가능성 분석",
  schema: "데이터 형식",
  ml_overlay: "예측 적용 비교",
  pooled_dataset: "여러 종목을 합친 데이터",
  pooled_model: "여러 종목으로 만든 모델",
  shadow_paper: "가상 기록",
  risk_stress_state: "위험 점검",
  walk_forward_backtest: "시간 순서대로 한 과거 테스트",
  paper_trading_gate: "가상 기록 통과 기준",
  live_trading_gate: "실거래 통과 기준",
  data_pipeline: "일봉 데이터/차트",
  news_causal_engine: "뉴스 원인 수집",
  rule_engine: "기본 규칙 계산",
  backtest_engine: "과거 성과 계산",
  risk_engine: "위험 기준 계산",
  validation_engine: "검증 계산",
  stress_engine: "큰 하락 가정 계산",
  integrity_engine: "데이터 일관성 확인",
  prediction_engine: "예측 계산",
  external_feature_engine: "외부 피처 계산",
  ml_overlay_backtest: "예측을 붙였을 때 성과 계산",
  pooled_dataset_builder: "여러 종목 데이터 만들기",
  pooled_model_engine: "여러 종목 예측 모델",
  model_registry_engine: "모델 목록 관리",
  shadow_paper_engine: "가상 기록",
  pooled_universe_update: "반도체 비교 종목 업데이트",
  research_expansion_update: "연구 확장 업데이트",
  system_state_engine: "시스템 준비 상태",
  daily_trading_report: "매매 계획",
  data_quality_engine: "데이터 품질",
  model_gate_engine: "모델 통과 기준 점검",
  backtest_event_ledger: "백테스트 이벤트 원장",
  backtest_feedback_feature_engine: "백테스트 피드백 피처",
  auto_research_engine: "자동 연구 Trial",
  daily_health_report: "오늘 시스템 점검",
  news_causal_context: "뉴스 원인 피처",
  daily_return_shock: "일별 수익률 충격",
  historical_drawdown_replay: "과거 낙폭 재현",
  atr_shock: "평균 변동폭 기준 충격",
  tsm_daily_algorithmic_signals: "일별 매수 신호",
  tsm_drawdown_episodes: "낙폭 구간",
  latest_atr_14: "최신 14일 ATR",
  worst_1d_close_to_close: "최악 1일 종가 수익률",
  p01_1d_close_to_close: "하위 1% 일별 종가 수익률",
  p05_1d_close_to_close: "하위 5% 일별 종가 수익률",
  minus_10pct_manual_shock: "-10% 수동 충격",
  minus_20pct_manual_shock: "-20% 수동 충격",
  minus_1atr: "-1ATR 충격",
  minus_2atr_stop: "-2ATR 손절 거리",
  minus_3atr_tail: "-3ATR 꼬리 위험",
  selected: "선택",
  blocked: "기준 미달",
  passed: "문제 없음",
  core: "핵심",
  universe: "유니버스",
  RULE_ALL_OOS_EVENTS: "기본 규칙 전체 결과",
  ML_THRESHOLD_SELECTED: "예측 기준으로 고른 결과",
  ML_SELECTED_MINUS_RULE_ALL: "예측 선택 결과 - 기본 규칙 결과",
  conservative: "보수적",
  research: "분석용",
  train_2016_2022: "학습 2016~2022",
  validation_2023: "검증 2023",
  test_2024: "테스트 2024",
  final_holdout_2025_2026: "마지막 확인 구간 2025~2026",
  tsm_train_validation: "TSMC 학습+검증",
  tsm_test_2024: "TSMC 테스트 2024",
  tsm_final_holdout_2025_2026: "TSMC 마지막 확인 구간 2025~2026",
  combined_test_holdout: "테스트+마지막 확인 구간",
  tsm_specific_logit_shift: "TSMC 전용 확률 조정",
  tsm_specific_logit_shift_latest_fit: "최신 TSMC 확률 조정",
  train_2016_2022_plus_validation_2023: "2016~2023 학습·검증",
  all_labeled_history_for_latest_snapshot: "최신 값을 위한 전체 과거 결과",
  excluded: "제외",
  numeric: "숫자형",
  categorical: "범주형",
  bool: "불리언",
  success: "성공",
  diagnostic: "확인용",
  live_like: "실전과 비슷하게",
  baseline: "기본 기준",
  decision_scope_missing: "판단 대상 범위 부족",
  insufficient_oos_evidence: "테스트 사례 부족",
  insufficient_selected_evidence: "선택 사례 부족",
  unstable_fold_selection: "검증 묶음별 선택 불안정",
  sparse_calibration_bins: "확률 확인 표본 부족",
  poor_probability_calibration: "확률 보정 필요",
  quality_gate_failed: "품질 기준 미달",
  no_base_rate_improvement: "기본 확률보다 개선 없음",
  weak_ranking_power: "좋은 신호 구분력 부족",
  threshold_not_adding_expectancy: "선택 기준 기대수익 부족",
  expectancy_not_statistically_positive: "보수 기대수익 부족",
  expectancy_not_repeatable: "기대수익 반복성 부족",
  threshold_instability: "선택 기준 불안정",
  tsm_specific_calibration_unstable: "TSMC 전용 보정 불안정",
  latest_local_model_missing: "최신 단독 모델 없음",
  pooled_quality_gate_failed: "여러 종목 품질 기준 미달",
  latest_signal_not_trade_ready: "최신 신호가 매수 준비 아님",
  pooled_latest_stop_risk_high: "여러 종목 손절 위험 높음",
  pooled_latest_expectancy_low: "여러 종목 기대값 낮음",
  stop_multiple: "손절 배수",
  score_threshold: "매수 후보 점수 기준",
  largest_up_close_to_close: "종가 기준 최대 상승",
  largest_down_close_to_close: "종가 기준 최대 하락",
  largest_intraday_range: "장중 변동폭 최대",
  largest_gap_up: "상승 갭 최대",
  largest_gap_down: "하락 갭 최대",
  POSITIVE_PERSISTENT: "긍정 지속",
  NEGATIVE_PERSISTENT: "부정 지속",
  MIXED_OR_LOW_IMPACT: "혼조/저영향",
  NOT_20D_TRADE_READY_DECISION_SCOPE: "20일 매수 판단 대상이 아님",
  SELECTED_OOS_EVENT_COUNT_LT_50: "테스트 사례가 50개 미만",
  SELECTED_EVENTS_PER_FOLD_LT_10: "검증 묶음별 사례가 10개 미만",
  NO_BRIER_IMPROVEMENT: "확률 예측이 기본값보다 낫지 않음",
  ECE_GT_0_10: "예측 확률 오차가 큼",
  PR_AUC_NOT_ABOVE_BASE: "좋은 신호를 구분하는 힘이 부족",
  ML_SELECTED_MINUS_RULE_ALL_LE_0: "예측으로 고른 결과가 기본 규칙보다 낫지 않음",
  SELECTED_EXPECTANCY_CI_LOWER_LE_0: "선택한 신호의 기대값 하단이 0 이하",
  POSITIVE_EXPECTANCY_FOLDS_LT_4: "좋은 결과가 나온 검증 묶음이 부족",
  CALIBRATION_MIN_BIN_N_LT_30: "확률 확인 구간의 사례가 부족",
  PREDICTION_NOT_DECISION_SUPPORT: "예측을 매매 판단에 쓰기에는 부족",
  BROKER_ORDER_RECONCILE_KILL_SWITCH_NOT_IMPLEMENTED: "브로커 주문 대조 중단장치 미구현",
  POOLED_SELECTED_MINUS_ALL_LE_0: "여러 종목 모델이 고른 결과가 전체보다 낫지 않음",
  POOLED_SELECTED_MINUS_ALL_CI_LOWER_LE_0: "보수적으로 보면 선택 결과가 전체보다 낫다고 보기 어려움",
  POOLED_SELECTED_MINUS_SCORE_BASELINE_CI_LOWER_LE_0: "점수 기준 대비 선택 결과의 보수 하단이 부족",
  POOLED_SELECTED_FRACTION_OUT_OF_RANGE: "여러 종목 선택 비율이 기준 범위 밖",
  POOLED_ECE_GT_0_10: "여러 종목 모델의 확률 오차가 큼",
  POOLED_NO_BRIER_IMPROVEMENT: "여러 종목 모델의 확률 예측이 개선되지 않음",
  POOLED_THRESHOLD_NOT_DECISION_ELIGIBLE: "여러 종목 선택 기준이 판단용 기준 미달",
  POOLED_OOF_FOLD_SELECTED_EVENTS_LT_10: "워크포워드 묶음별 선택 사례가 10개 미만",
  POOLED_OOF_FOLD_SELECTED_EVENTS_LT_25: "워크포워드 묶음별 선택 사례가 25개 미만",
  TSM_CALIBRATION_ECE_GT_0_15: "TSMC 전용 확률 조정 오차가 큼",
  TSM_CALIBRATION_ROUTE_NOT_PASSED: "TSMC 확률 조정 경로가 기준 미달",
  TSM_CALIBRATION_ROUTE_FAILED: "TSMC 확률 조정 경로 실패",
  POOLED_SELECTED_MINUS_RULE_PAIRED_CI_LOWER_LT_MINUS_0_50: "룰 대비 보수 하단이 -0.5% 미만",
  POOLED_STOP_RISK_GT_0_40: "손절 가능성이 40%를 넘음",
  POOLED_MODEL_QUALITY_NOT_PASSED: "여러 종목 모델이 기준을 못 넘음",
  LATEST_NOT_TRADE_READY: "최신 신호가 매수 판단 대상이 아님",
  POOLED_DECISION_SCORE_BELOW_THRESHOLD: "최신 결정 점수가 선택 기준보다 낮음",
  POOLED_STOP_RISK_GT_0_35: "손절 가능성이 35%를 넘음",
  POOLED_EXPECTED_R_LT_0_35: "기대 수익 대비 위험이 낮음",
  POOLED_EXPECTED_R_LT_MIN: "기대 수익 대비 위험이 기준보다 낮음",
  POOLED_STOP_RISK_GT_LIMIT: "손절 가능성이 기준보다 큼",
  POOLED_THRESHOLD_STABILITY_FAILED: "선택 기준이 검증 구간마다 불안정",
  POOLED_UPLIFT_NOT_PASSED: "선택 신호의 개선 근거가 부족",
  POOLED_EVAL_EVENTS_LT_150: "여러 종목 평가 사례가 150개 미만",
  POOLED_SELECTED_EVENTS_LT_50: "여러 종목 선택 사례가 50개 미만",
  POOLED_POSITIVE_EXPECTANCY_FOLDS_LT_4: "긍정 기대값 fold 수가 부족",
  POOLED_DECISION_CALIBRATION_BIN_N_LT_30: "판단 구간 확률 보정 표본 부족",
  POOLED_DATASET_QUALITY_FAILED: "여러 종목 데이터셋 품질 미달",
  LATEST_NOT_EVENT_CANDIDATE: "최신 행이 이벤트 후보가 아님",
  LATEST_RULE_FILTERED_NOT_TRADE_READY: "최신 신호가 룰 기준에서 매수 준비 상태가 아님",
  LATEST_CONTEXT_ONLY_NO_ENTRY_TRIGGER: "최신 행은 관찰용이며 매수 신호가 없음",
  LOCAL_LATEST_NOT_DECISION_SUPPORT: "최신 예측을 판단에 쓰기에는 부족",
  LOCAL_MODEL_COMPARISON_MISSING: "모델 비교 파일 없음",
  MODEL_GATE_AUDIT_MISSING: "모델 통과 기준 점검 없음",
  MISSING_MODEL_GATE: "모델 통과 기준 없음",
  OOS_EVENT_COUNT_LT_100: "테스트 사례가 100개 미만",
  OOS_EVENT_COUNT_LT_MIN: "테스트 사례 수 부족",
  SELECTED_OOS_EVENT_COUNT_LT_MIN: "선택된 테스트 사례 수 부족",
  SELECTED_EVENTS_PER_FOLD_LT_MIN: "검증 묶음별 선택 사례 수 부족",
  ECE_GT_LIMIT: "예측 확률 오차가 기준보다 큼",
  THRESHOLD_IQR_GT_LIMIT: "선택 기준값이 너무 흔들림",
  CALIBRATION_MIN_BIN_N_LT_MIN: "확률 확인 구간의 사례 수 부족",
  POSITIVE_EXPECTANCY_FOLDS_LT_MIN: "좋은 결과가 나온 검증 묶음 수 부족",
  PREDICTION_QUALITY_FALSE: "예측 품질 미달",
  POOLED_EVENT_COUNT_LT_MIN: "여러 종목 모델의 테스트 사례 수 부족",
  POOLED_SELECTED_EVENT_COUNT_LT_MIN: "여러 종목 모델이 선택한 사례 수 부족",
  POOLED_SELECTED_EXPECTANCY_CI_LOWER_LE_0: "여러 종목 선택 신호의 보수 기대값이 0 이하",
  POOLED_ECE_GT_LIMIT: "여러 종목 모델의 확률 오차가 기준보다 큼",
  TSM_CALIBRATION_ECE_GT_LIMIT: "TSMC 전용 확률 조정 오차가 기준보다 큼",
  THRESHOLD_IQR_GT_0_10: "선택 기준값이 너무 흔들림",
  data_quality: "데이터 품질",
  model_gate: "모델 통과 기준",
  research_readiness: "분석 준비도",
  operational_quality: "운영 품질",
  paper_shadow: "가상 기록",
  live_trading: "실거래",
  local_latest: "최신 단독 예측",
  local_prediction: "단독 예측 모델",
  pooled_latest: "최신 여러 종목 예측",
  pooled_tsm_calibration: "TSMC 확률 조정",
  VALIDATION_MAX_CI_LOWER: "검증 결과의 안전 기준",
  SKIPPED_INSUFFICIENT_FOLD_EVENTS: "검증 사례 부족으로 생략",
  INSUFFICIENT_VALIDATION_CLASS_BALANCE: "검증 데이터 균형 부족",
  DEEP_DD_RECOVERY: "깊은 낙폭 회복",
  RESEARCH_ONLY_DEEP_DD: "분석용 깊은 하락",
  "60D_BREAKOUT": "60일 돌파",
  STOP: "손절",
  TARGET_1R: "1차 목표가",
  TARGET_2R: "2차 목표가",
  ATR_STOP_2X: "평균 변동폭 기준 손절",
  risk_policy: "위험 기준",
  stress_tolerance: "하락 견딤 정도",
  full_period_backtest: "전체 기간 과거 테스트",
  walk_forward_evidence: "시간 순서 검증 근거",
  LIVE_TRADING_DISABLED: "실거래 비활성화",
  STALE_DATA: "오래된 데이터",
  MODEL_NOT_TRUSTED: "신뢰 기준 미달 모델",
  LOW_EXPECTANCY: "기대값 부족",
  STOP_RISK_TOO_HIGH: "손절 위험 과다",
  RISK_LIMIT_EXCEEDED: "위험 한도 초과",
  MISSING_BROKER: "브로커 미연결",
  RECONCILIATION_REQUIRED: "주문 대조 필요",
  KILL_SWITCH_ACTIVE: "중단장치 작동",
  earnings_results: "실적 발표",
  guidance: "가이던스",
  monthly_revenue: "월간 매출",
  capex_fab_expansion: "설비·공장 확장",
  ai_hpc_demand: "AI/HPC 수요",
  customer_supply_chain: "고객·공급망",
  regulation_export_controls: "규제·수출통제",
  geopolitics_taiwan: "지정학 리스크",
  operational_disruption: "운영 차질",
  analyst_rating_target: "애널리스트 평가",
  dividend_capital_return: "배당·자본환원",
  peer_sector_move: "동종업계 움직임",
  macro_rates_fx: "매크로·금리·환율",
  technical_market_move: "기술적·시장성 기사",
  other: "기타",
};

const strategyLabels = {
  A_20D_BREAKOUT_2ATR: "전략 A: 20일 돌파",
  B_50D_PULLBACK_2ATR: "전략 B: 50일선 눌림목",
  C_SCORE75_TRIGGER_2ATR: "전략 C: 점수 75+트리거",
  D_200D_TREND_2ATR: "전략 D: 200일선 추세",
  "Strategy A - 20D breakout + 2ATR stop": "전략 A: 20일 돌파",
  "Strategy B - 50D pullback bounce + 2ATR stop": "전략 B: 50일선 눌림목",
  "Strategy C - score >= 75 + trigger + 2ATR stop": "전략 C: 점수 75+트리거",
  "Strategy D - 200D trend hold + 2ATR stop": "전략 D: 200일선 추세",
};

const columnLabels = {
  section: "구분",
  item: "항목",
  value: "값",
  raw_field: "원본 항목",
  raw_value: "원본 값",
  failed_count: "실패 수",
  passed_count: "통과 수",
  total_count: "전체 수",
  unit: "단위",
  notes: "메모",
  strategy_id: "전략",
  strategy_name: "전략명",
  description: "설명",
  start_date: "시작일",
  end_date: "종료일",
  total_return_pct: "총수익률",
  cagr_pct: "연복리",
  annualized_volatility_pct: "연간 변동폭",
  max_drawdown_pct: "가장 크게 빠진 폭",
  sharpe_zero_rf: "수익 대비 변동성",
  sortino_zero_rf: "하락 위험 대비 수익",
  calmar_ratio: "낙폭 대비 수익",
  trade_count: "거래 수",
  win_rate_pct: "승률",
  profit_factor: "손실 대비 수익",
  exposure_days_pct: "투자 중이던 기간",
  year: "연도",
  year_return_pct: "연간 수익률",
  year_max_drawdown_pct: "연간 최대 낙폭",
  candidate_scope: "어떤 신호인지",
  candidate_tier: "후보 계층",
  evaluation_scope: "평가 범위",
  is_entry_research_candidate: "연구용 후보",
  is_model_training_candidate: "학습 후보",
  is_decision_entry_candidate: "엄격 판단 후보",
  is_trade_ready_entry_candidate: "매수 준비 후보",
  is_actionable_entry_candidate: "조건 통과 매수 후보",
  is_risk_research_candidate: "위험 연구 후보",
  champion_scope: "대표 모델 범위",
  horizon_days: "며칠 뒤를 볼지",
  model_name: "사용한 예측 방식",
  model_family: "모델 계열",
  fold_count: "검증 묶음 수",
  fold_id: "검증 묶음",
  oos_event_count: "테스트 사례 수",
  selected_oos_event_count: "선택된 테스트 사례 수",
  effective_oos_event_count: "실제 사용 사례 수",
  brier_score: "확률 예측 오차",
  brier_reliability: "확률 신뢰 오차",
  brier_resolution: "확률 구분력",
  brier_uncertainty: "결과 불확실성",
  brier_improvement_pct: "확률 오차 개선",
  average_precision: "정밀도-재현 요약",
  base_rate_average_precision: "기본 성공률 기준 AP",
  log_loss: "예측 손실",
  pr_auc: "좋은 신호 구분력",
  base_rate_pr_auc: "기본 성공률 기준",
  ece: "확률 실제 오차",
  all_signal_expectancy_pct: "전체 신호 평균 기대수익",
  selected_signal_expectancy_pct: "선택 신호 평균 기대수익",
  expectancy_improvement_pct: "기대수익 개선",
  positive_expectancy_folds: "좋은 결과 검증 묶음 수",
  mean_effective_sample_size: "평균 사용 사례 수",
  min_decision_oos_events: "최소 테스트 사례 수",
  prediction_quality_pass: "예측 사용 가능",
  model_quality_pass: "모델 사용 가능",
  latest_signal_pass: "최신 신호 통과",
  pooled_decision_support_allowed: "여러 종목 판단 참고 가능",
  decision_support_allowed: "판단에 참고 가능",
  decision_block_reasons: "판단에 못 쓰는 이유",
  model_quality_block_reasons: "모델이 막힌 이유",
  quality_block_reasons: "품질 문제 이유",
  block_reasons: "막힌 이유",
  prediction_scope_label: "예측 종류",
  prediction_status: "예측 상태",
  model_policy: "모델 선택 기준",
  promotion_status: "승격 상태",
  dependency_versions: "의존성 버전",
  feature_count: "사용 가능 항목 수",
  root_cause: "차단 원인",
  priority: "우선순위",
  failed_gate_count: "막힌 기준 수",
  category: "원인 분류",
  gate_groups: "점검 묶음",
  gates: "기준",
  candidate_scopes: "신호 종류",
  horizons: "기간",
  models: "예측 방식",
  splits: "검증 구간",
  value_min: "최솟값",
  value_max: "최댓값",
  recommended_action: "우선 조치",
  selected_minus_rule_all_pct: "기본 규칙 대비 차이",
  selected_expectancy_ci_lower_pct: "보수적으로 본 기대수익",
  selected_mean_return_pct: "선택 평균 수익률",
  selected_success_rate: "선택 성공률",
  selected_stop_rate: "선택 손절률",
  selected_event_count: "선택된 사례 수",
  paper_selected_oos_event_count: "Paper 선택 테스트 사례",
  paper_min_selected_events_per_oof_fold: "Paper fold 최소 선택",
  paper_selected_minus_rule_ci_lower_pct: "Paper uplift 보수 하단",
  selected_fraction: "선택 비율",
  event_count: "이벤트 수",
  symbol_count: "종목 수",
  symbol: "종목",
  symbol_group: "종목 그룹",
  loaded: "로드됨",
  strict_eligible: "엄격 학습 가능",
  eligibility_status: "유니버스 상태",
  failure_reasons: "제외 사유",
  trade_ready_20d_labeled: "20D trade-ready 표본",
  model_training_20d_labeled: "20D 학습 표본",
  test_holdout_trade_ready_20d: "test/holdout trade-ready",
  tsm_like_group: "TSM-like 그룹",
  tsm_like_weight_mean: "평균 TSM-like 가중치",
  tsm_like_effective_n_contribution: "effective N 기여",
  base_group_weight: "기본 그룹 가중치",
  dynamic_weight: "동적 유사도 가중치",
  tsm_like_weight: "TSM-like 최종 가중치",
  p_success_tsm_like_calibrated: "TSM-like 보정 성공 가능성",
  decision_score_tsm_like_calibrated: "TSM-like 결정 점수",
  selected_tsm_like_calibration_route: "선택된 TSM-like route",
  candidate_symbols: "후보 종목",
  loaded_symbols: "로드된 종목",
  strict_eligible_symbols: "엄격 학습 가능 종목",
  short_history_research_only: "짧은 이력 연구용",
  missing_enriched_symbols: "가공 데이터 누락",
  split: "구간",
  success_rate: "성공률",
  mean_return_pct: "평균 수익률",
  selected_minus_all_pct: "전체 대비 선택 결과",
  selected_minus_all_ci_lower_pct: "전체 대비 보수 하단",
  selected_minus_all_ci_lower_pct_independent: "전체 대비 독립 bootstrap 하단",
  selected_minus_all_ci_lower_pct_paired: "전체 대비 paired bootstrap 하단",
  score_baseline_mean_return_pct: "점수 기준 평균 수익률",
  selected_minus_score_baseline_pct: "점수 기준 대비 선택 결과",
  selected_minus_score_baseline_ci_lower_pct: "점수 기준 대비 보수 하단",
  selected_minus_score_baseline_ci_lower_pct_independent: "점수 기준 대비 독립 bootstrap 하단",
  selected_minus_score_baseline_ci_lower_pct_paired: "점수 기준 대비 paired bootstrap 하단",
  uplift_bootstrap_p_value: "Uplift p-value",
  uplift_bootstrap_p_value_independent: "독립 bootstrap p-value",
  uplift_bootstrap_p_value_paired: "paired bootstrap p-value",
  bootstrap_method: "Bootstrap 방식",
  bootstrap_block_col: "Bootstrap 블록",
  bootstrap_iterations: "Bootstrap 반복 수",
  positive_expectancy_fold_count: "긍정 기대값 fold 수",
  fold_positive_uplift_count: "전체 대비 긍정 fold 수",
  fold_positive_score_baseline_uplift_count: "점수 기준 대비 긍정 fold 수",
  fold_uplift_ci_lower_min: "전체 대비 fold 하단 최솟값",
  fold_score_baseline_ci_lower_min: "점수 기준 대비 fold 하단 최솟값",
  uplift_pass: "Uplift 통과",
  uplift_failure_reasons: "Uplift 실패 이유",
  uplift_failure_reasons_detail: "Uplift 실패 상세",
  strict_gate_status: "Strict gate",
  paper_gate_status: "Paper gate",
  strict_decision_support_allowed: "Strict 판단 가능",
  paper_decision_support_allowed: "Paper 판단 가능",
  paper_model_gate_pass: "Paper 모델 통과",
  paper_latest_signal_pass: "Paper 최신 신호 통과",
  paper_model_block_reasons: "Paper 모델 차단 이유",
  paper_latest_block_reasons: "Paper 최신 신호 차단 이유",
  paper_gate_block_reasons: "Paper gate 차단 이유",
  tsm_like_calibration_route: "TSM-like 보정 route",
  tsm_like_effective_train_validation_n: "TSM-like train/validation effective N",
  tsm_like_calibration_ece: "TSM-like ECE",
  tsm_like_route_selection_pass: "TSM-like route 통과",
  pooled_ece_for_paper: "Paper용 pooled ECE",
  pooled_brier_improvement_pct_for_paper: "Paper용 Brier 개선",
  latest_trade_ready: "최신 trade-ready",
  latest_stop_hit_20d: "최신 20D 손절 가능성",
  score_baseline_policy: "점수 기준 정책",
  selected_ci_lower_pct: "보수적으로 본 선택 결과",
  selected_count: "선택 수",
  policy: "적용 기준",
  event_weight: "사례 비중",
  selection_rate_pct: "선택률",
  cumulative_weighted_return_pct: "누적 가중 수익률",
  max_event_curve_drawdown_pct: "사례별 최대 하락폭",
  threshold_median: "선택 기준 중앙값",
  threshold_std: "선택 기준 흔들림",
  calibration_methods: "확률 조정 방식",
  rank_score: "종합 순위 점수",
  score_col: "점수 항목",
  effective_min_selected: "실제 최소 선택 수",
  probability_quality_pass: "확률 품질 통과",
  economics_pass: "경제성 통과",
  eligible: "판단 기준 충족",
  selected_stop_hit: "선택 손절률",
  all_stop_hit: "전체 손절률",
  utility_weight_label: "가중치 설명",
  utility_weight_p_success: "성공 확률 가중치",
  utility_weight_p_stop_hit: "손절 확률 가중치",
  utility_weight_expected_r: "기대 R 가중치",
  utility_weight_score_price_algo_total: "알고리즘 점수 가중치",
  utility_score: "위험조정 선택 점수",
  raw_utility_score: "원 위험조정 점수",
  threshold_decision_eligible: "판단용 선택 기준",
  bin_id: "구간",
  n: "수",
  mean_predicted_probability: "평균 예상 확률",
  observed_success_rate: "실제 성공률",
  abs_calibration_error: "확률 실제 오차",
  max_abs_calibration_error: "가장 큰 확률 오차",
  binning: "확률 구간 나누기",
  bin_count: "구간 수",
  min_bin_n: "구간 최소 표본",
  layer: "조정 단계",
  tsm_like_calibration_route: "TSM-like route",
  weighted_event_count: "가중 이벤트 수",
  effective_n: "effective N",
  decision_ece: "결정 확률 오차",
  decision_min_calibration_bin_n: "보정 bin 최소 N",
  route_status: "route 상태",
  route_shrinkage: "route shrinkage",
  route_logit_shift: "route logit shift",
  route_selection_provenance_valid: "route 선택 근거 유효",
  tsm_like_route_selection_pass: "route 선택 통과",
  tsm_like_route_selection_failure_reasons: "route 실패 이유",
  selection_decision_ece: "선택구간 ECE",
  selection_brier_improvement_pct: "선택구간 Brier 개선",
  selection_effective_n: "선택구간 effective N",
  combined_decision_ece: "test+holdout ECE",
  combined_brier_improvement_pct: "test+holdout Brier 개선",
  route_priority: "route 우선순위",
  route_selection_window: "route 선택 구간",
  route_evaluation_window: "route 평가 구간",
  is_selected_tsm_like_route: "선택된 route",
  actual_success_rate: "실제 성공률",
  predicted_success_rate: "예측 성공률",
  posterior_success_rate: "조정 후 성공률",
  logit_shift: "확률 조정값",
  shrinkage: "과신 줄인 정도",
  fit_source: "계산 기준",
  prediction_universe: "예측 대상",
  labeled_count: "결과가 확인된 사례 수",
  success_rate_pct: "성공률",
  stop_survival_rate_pct: "손절 회피율",
  positive_return_rate_pct: "양수 수익률",
  hit_1r_before_stop_rate_pct: "1차 목표 먼저 도달률",
  hit_2r_before_stop_rate_pct: "2차 목표 먼저 도달률",
  atr_stop_rate_pct: "손절률",
  time_barrier_rate_pct: "시간 만료율",
  mean_holding_days: "평균 보유일",
  mean_net_return_pct: "평균 순수익률",
  group_type: "그룹 유형",
  group_value: "그룹 값",
  unavailable_count: "사용 불가 수",
  median_net_return_pct: "중앙 순수익률",
  threshold: "선택 기준값",
  threshold_20d: "20일 선택 기준값",
  threshold_reason: "선택 기준 이유",
  threshold_score_col: "선택 점수 항목",
  threshold_source: "선택 기준 출처",
  selection_score_col: "선택 점수 항목",
  probability_col: "확률 품질 항목",
  raw_probability_col: "원확률 항목",
  is_champion: "대표 후보",
  validation_brier_improvement_pct: "검증 확률 오차 개선",
  validation_ece: "검증 확률 실제 오차",
  validation_expectancy_pct: "검증 기대수익",
  validation_selected_count: "검증 선택 수",
  min_validation_trades: "최소 검증 거래",
  train_start_date: "학습 시작",
  train_end_date: "학습 종료",
  test_start_date: "검증 시작",
  test_end_date: "검증 종료",
  train_event_count: "학습 이벤트",
  test_event_count: "검증 이벤트",
  test_positive_rate: "검증 양성률",
  calibration_method: "확률 조정 방식",
  fallback_reason: "대체 사유",
  model_feature_count: "전체 데이터 항목 수",
  selected_model_feature_count: "선택한 데이터 항목 수",
  mean_p_stop_survival: "평균 손절 회피",
  mean_p_positive_given_survival: "평균 생존 후 양수",
  selected_signal_count: "선택 신호 수",
  selected_signal_win_rate_pct: "선택 신호 승률",
  selected_signal_profit_factor: "선택 신호 손실 대비 수익",
  p_success: "오를 가능성",
  p_success_20d: "20일 안에 오를 가능성",
  p_success_base: "기본 성공 가능성",
  p_success_eb: "경험 베이즈 성공 가능성",
  p_success_logistic: "로지스틱 성공 가능성",
  p_success_lgbm: "LightGBM 성공 가능성",
  p_success_xgb: "XGBoost 성공 가능성",
  p_success_stack_raw: "스택 원확률",
  p_success_calibrated: "보정 성공 가능성",
  p_success_tsm_calibrated: "TSMC 맞춤 조정 가능성",
  p_success_lower_80: "낮게 봤을 때 가능성",
  p_success_upper_80: "높게 봤을 때 가능성",
  p_stop_survival: "손절 안 날 가능성",
  p_stop_survival_20d: "20일 안에 손절 안 날 가능성",
  p_stop_hit: "손절 날 가능성",
  p_stop_hit_lgbm: "LightGBM 손절 가능성",
  p_hit_1r: "1차 목표 도달 가능성",
  p_hit_2r: "2차 목표 도달 가능성",
  p_positive_given_survival: "손절 없이 버틴 뒤 수익 가능성",
  expected_r: "위험 대비 기대수익",
  expected_r_20d: "20일 위험 대비 기대수익",
  expected_r_net: "비용 반영 위험 대비 기대수익",
  expected_r_lgbm: "LightGBM 기대 R",
  expected_net_return: "기대 순수익률",
  expected_net_return_20d: "20일 기대 순수익률",
  expected_net_return_pct: "기대 순수익률",
  expected_net_return_pct_20d: "20일 기대 순수익률",
  decision_score: "결정 점수",
  decision_score_20d: "20일 결정 점수",
  decision_score_tsm_calibrated: "TSMC 조정 결정 점수",
  effective_group_n: "비슷한 사례 수",
  label_success: "실제 성공",
  label_success_20d: "20일 실제 성공",
  label_stop_survival: "실제 손절 회피",
  label_positive_return: "실제 양수 수익",
  label_net_return_pct: "실제 순수익률",
  label_net_return_pct_20d: "20일 실제 순수익률",
  label_expected_r: "실제 위험 대비 수익",
  label_expected_r_20d: "20일 실제 위험 대비 수익",
  label_exit_reason: "실제 종료 사유",
  selected_by_threshold: "선택 기준 통과",
  entry_gate_status: "매수 기준 상태",
  latest_entry_gate_status: "최신 매수 기준 상태",
  prediction_scope_used: "사용한 예측 종류",
  prediction_signal_status: "예측 신호 상태",
  prediction_use_status: "예측 사용 상태",
  decision_permission: "판단 권한",
  final_trade_decision: "최종 매매 판단",
  paper_action: "가상 기록 행동",
  signal_entry_trigger: "매수 신호",
  signal_trade_action: "신호 기준 매매 행동",
  realized_status: "실현 상태",
  realized_success: "실현 성공",
  realized_net_return_pct: "실현 순수익률",
  realized_exit_reason: "실현 종료 사유",
  prediction_asof_date: "예측 기준일",
  recorded_at_utc: "기록 시각",
  feature: "사용한 데이터 항목",
  column: "열",
  role: "역할",
  available_at: "사용 가능 시점",
  leakage_policy: "미래 정보 섞임 방지",
  feature_group: "데이터 항목 그룹",
  dtype: "데이터 종류",
  non_null_count: "비결측 수",
  feature_type: "데이터 항목 유형",
  decision: "결정",
  reason: "사유",
  missing_rate: "결측률",
  check: "체크",
  passed: "문제 없음",
  severity: "중요도",
  tolerance: "허용 범위",
  details: "상세",
  scenario_type: "시나리오 유형",
  scenario_name: "시나리오",
  source: "출처",
  shock_return_pct: "충격 수익률",
  latest_close: "최신 종가",
  implied_price: "암시 가격",
  final_recommended_max_weight_pct: "최종 권장 비중",
  estimated_portfolio_impact_pct: "예상 영향",
  step: "단계",
  status: "상태",
  component: "구성요소",
  gate_group: "점검 묶음",
  gate: "통과 기준",
  block_reason: "막힌 이유",
  block_reason_explanation: "쉬운 설명",
  data_quality_status: "데이터 상태",
  decision_support_data_gate: "판단용 데이터 상태",
  model_gate_status: "모델 기준 상태",
  daily_health_status: "오늘 점검 상태",
  duration_sec: "소요 초",
  returncode: "종료 코드",
  ended_at_utc: "종료 시각",
  domain: "점검 영역",
  weight: "가중치",
  score: "점수",
  note: "메모",
  dataset: "데이터셋",
  file: "파일",
  rows_compared: "비교 행",
  mismatch_count: "불일치 수",
  max_abs_diff: "최대 차이",
  provider: "공급자",
  interval: "간격",
  auth_status: "인증",
  request_status: "요청 상태",
  supports_complete_requested_10y: "10년 완성",
  actual_start: "실제 시작",
  actual_end: "실제 종료",
  rows: "행",
  evidence: "근거",
  source_url: "출처 주소",
  source_dir: "산출 폴더",
  scope: "범위",
  period: "기간",
  trading_days: "거래일",
  start_close_usd: "시작 종가",
  end_close_usd: "종료 종가",
  price_total_return_pct: "가격 총수익률",
  adj_total_return_pct: "수정 총수익률",
  price_cagr_pct: "가격 연평균 수익률",
  adj_cagr_pct: "수정 가격 연평균 수익률",
  extreme_type: "극단 유형",
  open: "시가",
  high: "고가",
  low: "저가",
  close: "종가",
  volume: "거래량",
  close_change_pct: "종가 변화",
  open_gap_pct: "시가 갭",
  open_to_close_pct: "시가-종가",
  intraday_range_pct_prev_close: "장중 변동폭",
  volume_ratio_20: "20일 거래량 배율",
  atr_14_pct: "ATR14",
  vol_20d_ann: "20일 변동성",
  drawdown_from_ath: "고점 대비 낙폭",
  trend_regime: "추세 상태",
  event_date: "이벤트일",
  trading_date_used: "사용 거래일",
  event_name: "이벤트",
  event_type: "유형",
  event_impact_score: "뉴스/이벤트 영향 점수",
  event_direction_class: "방향성",
  news_event_count_1d: "당일 뉴스 수",
  news_event_count_3d: "3일 뉴스 수",
  news_sentiment_score_1d: "뉴스 감성",
  news_primary_cause_type: "대표 원인",
  news_primary_cluster_id: "원인 묶음",
  news_match_confidence: "매칭 신뢰도",
  news_match_confidence_score: "매칭 점수",
  news_coverage_status: "수집 상태",
  news_source_count: "출처 수",
  news_primary_source_url: "대표 URL",
  news_cause_summary: "원인 요약",
  news_penalty_event: "뉴스 위험 감점",
  penalty_days: "감점 발생일",
  recent_120_penalty_days: "최근 120일 감점일",
  total_penalty: "누적 감점",
  max_penalty: "최대 감점",
  latest_penalty_date: "최근 감점일",
  latest_penalty_cause: "최근 감점 원인",
  latest_penalty_summary: "최근 감점 요약",
  candidate_event_titles: "후보 뉴스",
  cluster_id: "묶음 ID",
  representative_title: "대표 사건명",
  cause_type: "원인 카테고리",
  article_count: "기사 수",
  source_count: "출처 수",
  source_urls: "출처 URL",
  coverage_status: "수집 상태",
  hist_news_category_count_20d: "같은 원인 20일 표본",
  hist_news_category_success_rate_20d: "같은 원인 20일 성공률",
  hist_news_category_mean_return_20d: "같은 원인 20일 평균",
  hist_news_category_count_60d: "같은 원인 60일 표본",
  hist_news_category_success_rate_60d: "같은 원인 60일 성공률",
  hist_news_category_mean_return_60d: "같은 원인 60일 평균",
  post_20d_return_pct: "이후 20일",
  post_60d_return_pct: "이후 60일",
  pre_20d_return_pct: "이전 20일",
  return_on_event_day_pct: "이벤트 당일",
  month_return_pct: "월간 수익률",
  mean_daily_pct: "평균 일수익률",
  avg_abs_daily_pct: "평균 절대 일변동",
  ann_vol_pct: "연율 변동성",
  avg_intraday_range_pct: "평균 장중 변동",
  avg_atr14_pct: "평균 ATR14",
  max_up_day_pct: "최대 상승일",
  max_down_day_pct: "최대 하락일",
  up_days: "상승일",
  down_days: "하락일",
  high_vol_days: "고변동일",
  event_shock_days: "충격일",
  rule: "조건",
  n_signals: "신호 수",
  mean_fwd_return_pct: "신호 후 평균 수익률",
  median_fwd_return_pct: "신호 후 중앙 수익률",
  p10_fwd_return_pct: "나쁜 쪽 10%",
  p90_fwd_return_pct: "좋은 쪽 10%",
  worst_fwd_return_pct: "신호 후 최악 수익률",
  best_fwd_return_pct: "신호 후 최고 수익률",
  strategy_group: "전략 그룹",
  window_id: "창",
  test_trading_days: "검증 거래일",
  train_total_return_pct: "학습 총수익률",
  test_total_return_pct: "검증 총수익률",
  train_cagr_pct: "학습 CAGR",
  test_cagr_pct: "검증 CAGR",
  train_max_drawdown_pct: "학습 MDD",
  test_max_drawdown_pct: "검증 MDD",
  train_sharpe_zero_rf: "학습 샤프",
  test_sharpe_zero_rf: "검증 샤프",
  test_exposure_days_pct: "검증 노출",
  test_positive: "검증 양수",
  trial_count: "실험 수",
  best_trial_source: "최고 실험 출처",
  best_trial_parameters: "최고 실험 파라미터",
  best_sharpe: "최고 샤프",
  median_sharpe: "중앙 샤프",
  best_cagr_pct: "최고 CAGR",
  pbo_proxy: "과최적화 확률 대용값",
  overfit_warning: "과최적화 경고",
  observations: "관측 수",
  observed_sharpe: "관측 샤프",
  skew: "왜도",
  kurtosis: "첨도",
  probabilistic_sharpe_ratio: "PSR",
  deflated_sharpe_ratio: "DSR",
  deflated_benchmark_sharpe: "DSR 기준 샤프",
  dsr_pass: "DSR 통과",
  trade_event: "거래 이벤트",
  entry_date: "진입일",
  entry_price: "진입가",
  target_weight_pct: "목표 비중",
  exit_date: "청산일",
  exit_price: "청산가",
  exit_reason: "청산 사유",
  holding_trading_days: "보유 거래일",
  net_return_pct: "순수익률",
  portfolio_return_pct: "포트폴리오 수익률",
  r_multiple: "R 배수",
  sensitivity_type: "민감도 유형",
  parameter_value: "파라미터",
  row_count: "행 수",
  universe_size_target: "유니버스 크기",
  selected_minus_rule_mean: "선택-룰 평균",
  selected_minus_rule_paired_ci_lower: "선택-룰 paired 하단",
  tsm_like_calibration_ece: "TSM-like ECE",
  selected_fraction_drift: "선택률 흔들림",
  fold_positive_uplift_count: "긍정 uplift fold",
  external_feature_status: "외부 피처 상태",
  stdout_tail: "표준 출력",
  stderr_tail: "오류 출력",
};

const fileKindLabels = { csv: "CSV", report: "리포트", image: "이미지", file: "파일" };
const fileScopeLabels = { core: "핵심", universe: "유니버스" };
const fileLabels = {
  "tsm_daily_10y_raw.csv": "일봉 원본 데이터",
  "tsm_daily_10y_enriched.csv": "일봉 분석 데이터",
  "tsm_daily_10y_summary.csv": "일봉 요약",
  "tsm_daily_columns_dictionary.csv": "일봉 컬럼 설명",
  "tsm_event_impact_10y.csv": "이벤트 영향 원본",
  "tsm_news_raw_articles.csv": "뉴스 수집 원본",
  "tsm_news_events_normalized.csv": "뉴스 사건 구조화",
  "tsm_news_event_clusters.csv": "뉴스 사건 묶음",
  "tsm_price_news_matches.csv": "가격-뉴스 원인 매칭",
  "tsm_hourly_available_raw.csv": "시간봉 원본 데이터",
  "tsm_hourly_available_enriched.csv": "시간봉 분석 데이터",
  "tsm_hourly_available_summary.csv": "시간봉 요약",
  "tsm_hourly_10y_source_audit.csv": "시간봉 소스 점검",
  "tsm_hourly_data_report.md": "시간봉 데이터 리포트",
  "tsm_minute_available_raw.csv": "분봉 원본 데이터",
  "tsm_minute_available_enriched.csv": "분봉 분석 데이터",
  "tsm_minute_available_summary.csv": "분봉 요약",
  "tsm_minute_10y_source_audit.csv": "분봉 소스 점검",
  "tsm_minute_data_report.md": "분봉 데이터 리포트",
  "tsm_algorithmic_rulebook.md": "알고리즘 룰북",
  "tsm_backtest_report.md": "백테스트 리포트",
  "tsm_backtest_equity_curves.csv": "백테스트 자산 흐름",
  "tsm_backtest_yearly_returns.csv": "백테스트 연도별 수익률",
  "tsm_latest_decision_snapshot.csv": "최신 매매 판단",
  "tsm_latest_risk_snapshot.csv": "최신 위험 상태",
  "tsm_risk_policy_daily.csv": "일별 위험 정책",
  "tsm_risk_policy_report.md": "위험 정책 리포트",
  "tsm_daily_algorithmic_signals.csv": "일별 알고리즘 신호",
  "tsm_daily_trading_plan.md": "일일 매매 계획",
  "tsm_daily_trading_plan.csv": "일일 매매 계획 표",
  "tsm_daily_update_manifest.csv": "일일 업데이트 실행 기록",
  "tsm_daily_update_operational_report.md": "일일 운영 리포트",
  "tsm_daily_integrity_checks.csv": "일별 데이터 일관성 체크",
  "tsm_daily_integrity_report.md": "일별 데이터 일관성 리포트",
  "tsm_daily_stress_scenarios.csv": "일별 스트레스 시나리오",
  "tsm_daily_stress_report.md": "일별 스트레스 리포트",
  "tsm_latest_stress_snapshot.csv": "최신 스트레스 상태",
  "tsm_latest_integrity_snapshot.csv": "최신 일관성 상태",
  "tsm_prediction_reliability_report.md": "예측 신뢰도 리포트",
  "tsm_prediction_report.md": "예측 리포트",
  "tsm_prediction_label_dataset.csv": "예측 라벨 데이터",
  "tsm_prediction_feature_matrix.csv": "예측 입력 행렬",
  "tsm_prediction_model_audit.csv": "예측 모델 점검",
  "tsm_prediction_feature_selection_report.csv": "예측에 쓴 데이터 항목 점검",
  "tsm_prediction_label_diagnostics.csv": "예측 과거 결과 확인",
  "tsm_prediction_model_comparison.csv": "예측 모델 비교",
  "tsm_prediction_calibration_bins.csv": "예측 확률 실제 오차 구간",
  "tsm_prediction_threshold_policy.csv": "예측 선택 기준",
  "tsm_prediction_oos_predictions.csv": "예측 테스트 기록",
  "tsm_prediction_calibration_summary.csv": "예측 확률 오차 요약",
  "tsm_prediction_policy_audit.csv": "예측 사용 기준 점검",
  "tsm_prediction_feature_contract.csv": "예측 데이터 사용 규칙",
  "tsm_prediction_fold_manifest.csv": "예측 검증 구간 구성",
  "tsm_prediction_model_registry.csv": "예측 모델 목록",
  "tsm_prediction_experiment_log.csv": "예측 실험 로그",
  "tsm_schema_quality_checks.csv": "데이터 형식 점검",
  "tsm_ml_overlay_summary.csv": "예측 적용 요약",
  "tsm_ml_overlay_quality_checks.csv": "예측 적용 점검",
  "tsm_ml_overlay_equity_curves.csv": "예측 적용 자산 흐름",
  "tsm_pooled_latest_prediction_overlay.csv": "여러 종목 최신 예측",
  "tsm_pooled_model_comparison.csv": "여러 종목 모델 비교",
  "tsm_pooled_model_quality_checks.csv": "여러 종목 모델 점검",
  "tsm_pooled_tsm_calibration.csv": "TSMC 확률 조정",
  "tsm_pooled_tsm_calibration_metrics.csv": "TSMC 확률 조정 결과",
  "tsm_pooled_model_threshold_policy.csv": "여러 종목 모델 선택 기준",
  "tsm_pooled_uplift_bootstrap_report.csv": "여러 종목 Bootstrap uplift 검증",
  "tsm_pooled_model_oos_predictions.csv": "여러 종목 모델 테스트 기록",
  "tsm_pooled_model_oof_predictions.csv": "여러 종목 워크포워드 테스트 기록",
  "tsm_pooled_model_slice_diagnostics.csv": "여러 종목 모델 구간별 진단",
  "tsm_prediction_pooled_quality_checks.csv": "여러 종목 데이터 점검",
  "tsm_prediction_pooled_feature_matrix.csv": "여러 종목 입력 행렬",
  "tsm_prediction_pooled_label_dataset.csv": "여러 종목 라벨 데이터",
  "tsm_prediction_pooled_split_manifest.csv": "여러 종목 데이터 분할",
  "tsm_prediction_pooled_scope_stats.csv": "여러 종목 신호 종류 통계",
  "tsm_prediction_pooled_schema.csv": "여러 종목 데이터 형식",
  "tsm_prediction_pooled_input_failures.csv": "여러 종목 입력 실패",
  "tsm_prediction_pooled_universe_config.csv": "여러 종목 비교 설정",
  "tsm_pooled_universe_update_manifest.csv": "여러 종목 업데이트 기록",
  "tsm_universe_validation_report.csv": "반도체 유니버스 검증",
  "tsm_universe_validation_report.md": "반도체 유니버스 검증 리포트",
  "tsm_prediction_pooled_sample_audit.csv": "여러 종목 표본 감사",
  "tsm_tsm_like_calibration_pool.csv": "TSM-like 보정 표본",
  "tsm_tsm_like_calibration_metrics.csv": "TSM-like 보정 결과",
  "tsm_paper_gate_snapshot.csv": "Paper gate 요약",
  "tsm_pooled_learning_curve_report.csv": "Pooled learning curve",
  "tsm_external_daily_features.csv": "외부 피처 일별 데이터",
  "tsm_external_feature_schema.csv": "외부 피처 스키마",
  "tsm_event_portfolio_backtest.csv": "Non-overlap event portfolio",
  "tsm_prediction_benchmark_comparison.csv": "예측 벤치마크 비교",
  "tsm_shadow_paper_predictions.csv": "가상 매매 기록",
  "tsm_shadow_paper_quality_checks.csv": "가상 기록 점검",
  "tsm_shadow_paper_report.md": "가상 기록 리포트",
  "tsm_ml_overlay_report.md": "예측 적용 리포트",
  "tsm_pooled_model_report.md": "여러 종목 모델 리포트",
  "tsm_prediction_model_registry_report.md": "모델 목록 리포트",
  "tsm_prediction_pooled_dataset_report.md": "여러 종목 데이터 리포트",
  "tsm_cscv_pbo_report.csv": "조합형 과최적화 확률 점검",
  "tsm_backtest_strategy_summary.csv": "백테스트 전략 요약",
  "tsm_backtest_trade_log.csv": "백테스트 거래 로그",
  "tsm_validation_causal_walk_forward_summary.csv": "엄격한 시간 순서 검증",
  "tsm_pbo_report.csv": "과최적화 확률 점검",
  "tsm_deflated_sharpe_report.csv": "수익 대비 위험 점검",
  "tsm_integrated_price_summary.csv": "통합 가격 요약",
  "tsm_extreme_daily_moves.csv": "극단 변동일",
  "tsm_event_integrated_analysis.csv": "이벤트 통합 분석",
  "tsm_news_integrated_daily.csv": "일별 뉴스 원인 피처",
  "tsm_news_causal_event_report.md": "뉴스 원인 리포트",
  "tsm_news_cause_forward_return_stats.csv": "뉴스 원인별 이후 수익률",
  "tsm_yearly_price_volatility_stats.csv": "연간 가격·변동성",
  "tsm_monthly_price_volatility_stats.csv": "월간 가격·변동성",
  "tsm_rule_forward_return_stats.csv": "조건별 이후 수익률",
  "tsm_regime_forward_return_stats.csv": "시장 상태별 이후 수익률",
  "tsm_data_validation_raw_vs_enriched.csv": "원본-가공 데이터 비교",
  "tsm_data_quality_checks.csv": "데이터 품질 체크",
  "tsm_data_quality_issues.csv": "데이터 품질 이슈",
  "tsm_data_quality_report.md": "데이터 품질 리포트",
  "tsm_latest_data_quality_snapshot.csv": "최신 데이터 품질",
  "tsm_model_gate_audit.csv": "모델 통과 기준 점검",
  "tsm_model_gate_root_causes.csv": "모델 차단 원인 요약",
  "tsm_model_gate_snapshot.csv": "모델 기준 요약",
  "tsm_model_gate_report.md": "모델 기준 리포트",
  "tsm_backtest_event_ledger.csv": "백테스트 이벤트 원장",
  "tsm_backtest_feedback_features.csv": "백테스트 피드백 피처",
  "tsm_backtest_feedback_quality_checks.csv": "백테스트 피드백 점검",
  "tsm_cpcv_path_summary.csv": "CPCV 경로 요약",
  "tsm_cpcv_strategy_distribution.csv": "CPCV 전략 분포",
  "tsm_cpcv_model_distribution.csv": "CPCV 모델 분포",
  "tsm_auto_research_trial_ledger.csv": "자동 연구 Trial 원장",
  "tsm_auto_research_best_candidates.csv": "자동 연구 우수 후보",
  "tsm_auto_research_report.md": "자동 연구 리포트",
  "tsm_research_expansion_manifest.csv": "연구 확장 실행 기록",
  "tsm_daily_health_snapshot.csv": "오늘 시스템 점검 요약",
  "tsm_system_block_reasons.csv": "시스템이 막힌 이유",
  "tsm_daily_health_report.md": "오늘 시스템 점검 리포트",
  "tsm_latest_system_state.csv": "최신 시스템 상태",
  "tsm_system_readiness_report.md": "시스템 준비도 리포트",
  "tsm_system_readiness_scorecard.csv": "시스템 준비도 점수표",
  "tsm_price_ma_drawdown.png": "가격·이동평균·낙폭 차트",
  "tsm_rolling_vol_atr.png": "롤링 변동성·ATR 차트",
  "tsm_daily_return_distribution.png": "일별 수익률 분포",
  "tsm_daily_return_volume.png": "일별 수익률·거래량",
  "tsm_daily_move_heatmap.png": "월별 변동폭 히트맵",
  "tsm_event_impact.png": "이벤트 영향 차트",
};

const blockReasonDescriptions = {
  NO_MODEL: "현재 상황에 바로 쓸 수 있는 예측 모델이 없습니다. 그래도 참고용 확률은 계속 보여줍니다.",
  NO_MODEL_CANDIDATE: "현재 신호가 예측 모델이 판단하도록 정한 대상에 들어오지 않았습니다. 관찰용 값은 계속 보여줍니다.",
  PREDICTION_NOT_DECISION_SUPPORT: "예측을 실제 판단에 참고하기에는 기준이 부족합니다. 기본 규칙, 위험, 가상 기록 정보는 계속 보여줍니다.",
  NO_LATEST_EVENT_CANDIDATE: "최신 신호가 예측 이벤트 후보가 아니라서 매매 판단용 예측으로 쓰지 않습니다.",
  NOT_20D_TRADE_READY_DECISION_SCOPE: "20일 안의 매수 판단 후보가 아니라서 매매 판단용 예측으로 쓰지 않습니다.",
  OOS_EVENT_COUNT_LT_100: "학습에 쓰지 않은 테스트 사례가 100개보다 적어 믿을 근거가 부족합니다.",
  OOS_EVENT_COUNT_LT_MIN: "학습에 쓰지 않은 테스트 사례 수가 최소 기준보다 적습니다.",
  SELECTED_OOS_EVENT_COUNT_LT_50: "예측이 고른 테스트 사례가 50개보다 적어 신뢰하기 어렵습니다.",
  SELECTED_OOS_EVENT_COUNT_LT_MIN: "예측이 고른 테스트 사례 수가 최소 기준보다 적습니다.",
  SELECTED_EVENTS_PER_FOLD_LT_10: "검증을 나눈 각 기간마다 선택 사례가 10개보다 적어 안정성이 부족합니다.",
  SELECTED_EVENTS_PER_FOLD_LT_MIN: "검증 기간별 선택 사례 수가 최소 기준보다 적습니다.",
  NO_BRIER_IMPROVEMENT: "예측 확률이 단순 기준값보다 나아지지 않았습니다.",
  ECE_GT_0_10: "예상 확률과 실제 성공률 차이가 큽니다.",
  ECE_GT_LIMIT: "예상 확률과 실제 성공률 차이가 허용 범위를 넘었습니다.",
  PR_AUC_NOT_ABOVE_BASE: "좋은 신호와 나쁜 신호를 구분하는 힘이 기본값보다 낫지 않습니다.",
  ML_SELECTED_MINUS_RULE_ALL_LE_0: "예측으로 고른 신호가 기본 규칙 전체 결과보다 낫지 않습니다.",
  SELECTED_EXPECTANCY_CI_LOWER_LE_0: "보수적으로 보면 선택한 신호의 기대수익이 0 이하일 수 있습니다.",
  POSITIVE_EXPECTANCY_FOLDS_LT_4: "좋은 결과가 나온 검증 기간 수가 부족합니다.",
  POSITIVE_EXPECTANCY_FOLDS_LT_MIN: "좋은 결과가 나온 검증 기간 수가 최소 기준보다 적습니다.",
  CALIBRATION_MIN_BIN_N_LT_30: "확률이 맞는지 확인할 사례 수가 구간별로 부족합니다.",
  CALIBRATION_MIN_BIN_N_LT_MIN: "확률 확인 구간의 사례 수가 최소 기준보다 적습니다.",
  THRESHOLD_IQR_GT_0_10: "좋은 신호를 고르는 기준값이 기간마다 너무 흔들립니다.",
  THRESHOLD_IQR_GT_LIMIT: "좋은 신호를 고르는 기준값의 흔들림이 허용 범위를 넘었습니다.",
  PREDICTION_QUALITY_FALSE: "이 예측 결과는 내부 품질 기준을 통과하지 못했습니다.",
  POOLED_SELECTED_MINUS_ALL_LE_0: "여러 종목으로 만든 모델이 고른 신호가 전체 후보보다 낫지 않습니다.",
  POOLED_SELECTED_MINUS_ALL_CI_LOWER_LE_0: "bootstrap으로 보수적으로 보면 선택 신호가 전체 후보보다 확실히 낫다고 보기 어렵습니다.",
  POOLED_SELECTED_MINUS_SCORE_BASELINE_CI_LOWER_LE_0: "점수만으로 고른 기준과 비교했을 때 모델 선택의 보수적 개선 폭이 부족합니다.",
  POOLED_SELECTED_MINUS_RULE_PAIRED_CI_LOWER_LT_MINUS_0_50: "Paper 기준에서도 선택 신호가 기본 규칙보다 충분히 낫다는 paired bootstrap 하단 근거가 부족합니다.",
  POOLED_OOF_FOLD_SELECTED_EVENTS_LT_25: "Paper 기준에서 fold별 선택 사례가 25개보다 적은 구간이 있어 선택 안정성이 부족합니다.",
  LATEST_NOT_TRADE_READY: "최신 TSM 신호가 trade-ready 후보가 아니라서 판단 지원을 열지 않습니다.",
  POOLED_STOP_RISK_GT_0_40: "최신 손절 가능성이 paper 기준 40%를 넘었습니다.",
  POOLED_STOP_RISK_GT_0_35: "최신 손절 가능성이 strict 기준 35%를 넘었습니다.",
  POOLED_EXPECTED_R_LT_0_35: "최신 기대 R이 strict 기준 0.35보다 낮습니다.",
  TSM_LIKE_EFFECTIVE_TRAIN_VALIDATION_N_LT_500: "TSM-like weighted calibration effective N이 500보다 낮습니다.",
  TSM_LIKE_CALIBRATION_FAILED: "TSM-like calibration route가 paper 기준을 통과하지 못했습니다.",
  TSM_CALIBRATION_ROUTE_FAILED: "TSM 직접 또는 선택 route의 보정 성능이 strict 기준을 넘지 못했습니다.",
  POOLED_ECE_GT_0_10: "여러 종목 모델의 예상 확률과 실제 성공률 차이가 큽니다.",
  POOLED_ECE_GT_LIMIT: "여러 종목 모델의 확률 오차가 허용 범위를 넘었습니다.",
  POOLED_NO_BRIER_IMPROVEMENT: "여러 종목 모델의 확률 예측이 단순 기준값보다 나아지지 않았습니다.",
  POOLED_THRESHOLD_NOT_DECISION_ELIGIBLE: "여러 종목 모델의 선택 기준은 참고용으로만 남기고 실제 판단에는 쓰지 않습니다.",
  POOLED_OOF_FOLD_SELECTED_EVENTS_LT_10: "워크포워드 검증 묶음마다 선택된 사례가 10개보다 적어 반복성이 부족합니다.",
  POOLED_MODEL_QUALITY_NOT_PASSED: "여러 종목 모델이 종합 기준을 통과하지 못했습니다.",
  LATEST_NOT_TRADE_READY: "최신 신호가 매수 판단 후보가 아닙니다.",
  POOLED_STOP_RISK_GT_0_35: "여러 종목 모델 기준 손절 가능성이 35%보다 높습니다.",
  POOLED_STOP_RISK_GT_LIMIT: "여러 종목 모델 기준 손절 가능성이 허용 범위를 넘었습니다.",
  POOLED_EXPECTED_R_LT_0_35: "예상되는 수익이 감수하는 위험에 비해 낮습니다.",
  POOLED_EXPECTED_R_LT_MIN: "위험 대비 기대수익이 최소 기준보다 낮습니다.",
  TSM_CALIBRATION_ECE_GT_0_15: "TSMC 맞춤 확률 조정 뒤에도 실제와 차이가 큽니다.",
  TSM_CALIBRATION_ECE_GT_LIMIT: "TSMC 맞춤 확률 조정 오차가 허용 범위를 넘었습니다.",
  POOLED_EVENT_COUNT_LT_MIN: "여러 종목 모델의 테스트 사례 수가 최소 기준보다 적습니다.",
  POOLED_SELECTED_EVENT_COUNT_LT_MIN: "여러 종목 모델이 선택한 사례 수가 최소 기준보다 적습니다.",
  POOLED_SELECTED_EXPECTANCY_CI_LOWER_LE_0: "보수적으로 보면 여러 종목 모델의 선택 신호 기대수익이 0 이하일 수 있습니다.",
  POOLED_THRESHOLD_STABILITY_FAILED: "좋은 신호를 고르는 기준이 기간마다 너무 흔들려 판단용 기준으로 쓰기 어렵습니다.",
  POOLED_UPLIFT_NOT_PASSED: "선택 신호가 전체 후보나 점수 기준보다 낫다는 bootstrap 근거가 부족합니다.",
  POOLED_EVAL_EVENTS_LT_150: "여러 종목 모델을 판단용으로 검증하기 위한 평가 사례가 부족합니다.",
  POOLED_SELECTED_EVENTS_LT_50: "모델이 실제로 선택한 사례가 부족해 반복성을 확인하기 어렵습니다.",
  POOLED_POSITIVE_EXPECTANCY_FOLDS_LT_4: "좋은 결과가 반복된 검증 기간 수가 부족합니다.",
  POOLED_DECISION_CALIBRATION_BIN_N_LT_30: "판단 구간의 확률 보정을 확인할 표본이 부족합니다.",
  POOLED_DATASET_QUALITY_FAILED: "여러 종목 모델 입력 데이터셋의 품질 점검을 통과하지 못했습니다.",
  LATEST_NOT_EVENT_CANDIDATE: "최신 행이 예측 이벤트 후보가 아니어서 모델 판단에 쓰지 않습니다.",
  LATEST_RULE_FILTERED_NOT_TRADE_READY: "최신 신호가 룰 엔진의 엄격한 매수 준비 기준을 통과하지 못했습니다.",
  LATEST_CONTEXT_ONLY_NO_ENTRY_TRIGGER: "최신 행은 관찰용이며 아직 매수 신호가 없습니다.",
  LIVE_TRADING_DISABLED: "현재 시스템은 연구와 가상 기록만 지원하므로 실거래 주문을 만들지 않습니다.",
  STALE_DATA: "최신 데이터가 오래되어 주문 판단으로 넘기지 않습니다.",
  MODEL_NOT_TRUSTED: "모델 기준이 부족해 주문 판단으로 넘기지 않습니다.",
  LOW_EXPECTANCY: "기대수익이 최소 기준보다 낮아 주문 판단으로 넘기지 않습니다.",
  STOP_RISK_TOO_HIGH: "손절 가능성이나 손실 폭이 허용 범위를 넘었습니다.",
  RISK_LIMIT_EXCEEDED: "요청 비중이 위험 정책의 최대 허용 비중을 넘었습니다.",
  MISSING_BROKER: "브로커 연결이 없어서 주문 실행은 설계상 막혀 있습니다.",
  RECONCILIATION_REQUIRED: "주문·체결 대조 장치가 준비되지 않아 실거래를 허용하지 않습니다.",
  KILL_SWITCH_ACTIVE: "중단장치가 작동 중이거나 준비되지 않아 실거래를 허용하지 않습니다.",
};

function $(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function toNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(String(value).replace(/,/g, "").replace(/%/g, ""));
  return Number.isFinite(parsed) ? parsed : null;
}

function isBlankSnapshotValue(value) {
  if (value === null || value === undefined) return true;
  if (value === "") return true;
  if (typeof value === "string") {
    const text = value.trim().toLowerCase();
    return text === "" || text === "na" || text === "n/a" || text === "none" || text === "null";
  }
  return false;
}

function snapshotValue(snapshot, keys) {
  for (const key of keys) {
    const value = snapshot?.[key];
    if (!isBlankSnapshotValue(value)) return value;
  }
  return "";
}

function snapshotWeight(value) {
  const n = toNumber(value);
  if (n === null) return null;
  return Math.abs(n) > 1 ? n / 100 : n;
}

function snapshotWeightValue(snapshot, keys) {
  return snapshotWeight(snapshotValue(snapshot, keys));
}

function isTruthy(value) {
  return value === true || String(value).toLowerCase() === "true" || String(value) === "1";
}

function statusByBool(value, yesText, noText) {
  if (value === null || value === undefined || value === "") return "없음";
  return isTruthy(value) ? yesText : noText;
}

function statusTone(value) {
  const text = String(value ?? "").toUpperCase();
  if (["PASS", "READY", "DECISION_SUPPORT_ALLOWED"].includes(text) || text.includes("ENTRY_ALLOWED")) return "good";
  if (["WARN", "BLOCKED"].includes(text) || text.includes("BLOCKED") || text.includes("DISPLAY_ONLY")) return "warn";
  if (["FAIL", "MISSING"].includes(text) || text.includes("FAILED")) return "bad";
  return "neutral";
}

function gateTone(value) {
  if (value === true || String(value).toLowerCase() === "true") return "good";
  if (value === false || String(value).toLowerCase() === "false") return "warn";
  const text = String(value ?? "").toUpperCase();
  if (["PASS", "READY"].includes(text)) return "good";
  if (["WARN", "BLOCKED", "NOT_APPLICABLE", "DISPLAY_ONLY"].some((token) => text.includes(token))) return "warn";
  if (["FAIL", "MISSING"].some((token) => text.includes(token))) return "bad";
  return "neutral";
}

function gateStatusText(value) {
  if (value === true || String(value).toLowerCase() === "true") return "통과";
  if (value === false || String(value).toLowerCase() === "false") return "차단";
  return labelValue(value);
}

function fmtNumber(value, digits = 2) {
  const n = toNumber(value);
  if (n === null) return "없음";
  return n.toLocaleString("ko-KR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function fmtCurrency(value) {
  const n = toNumber(value);
  if (n === null) return "없음";
  return `$${n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function fmtPct(value, digits = 1, fraction = false) {
  const n = toNumber(value);
  if (n === null) return "없음";
  const pct = fraction ? n * 100 : n;
  return `${pct.toLocaleString("ko-KR", { minimumFractionDigits: digits, maximumFractionDigits: digits })}%`;
}

function fmtMaybePct(value, digits = 1) {
  const n = toNumber(value);
  if (n === null) return "없음";
  return Math.abs(n) <= 1 ? fmtPct(n, digits, true) : fmtPct(n, digits, false);
}

function shortDate(value) {
  if (!value) return "없음";
  return String(value).slice(0, 10);
}

function fmtKstDateTime(value) {
  if (!value) return "없음";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return shortDate(value);
  return date.toLocaleString("ko-KR", {
    timeZone: "Asia/Seoul",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function todayKstInputValue() {
  const parts = new Intl.DateTimeFormat("ko-KR", {
    timeZone: "Asia/Seoul",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(new Date());
  const part = (type) => parts.find((item) => item.type === type)?.value || "01";
  return `${part("year")}-${part("month")}-${part("day")}`;
}

function compactText(value, max = 44) {
  const text = String(value ?? "없음");
  return text.length > max ? `${text.slice(0, max - 1)}...` : text;
}

function labelValue(value) {
  const text = String(value ?? "");
  const drawdownMatch = text.match(/^peak_(\d{4}-\d{2}-\d{2})_to_trough_(\d{4}-\d{2}-\d{2})$/);
  if (drawdownMatch) return `${drawdownMatch[1]} 고점 -> ${drawdownMatch[2]} 저점`;
  if (strategyLabels[text]) return strategyLabels[text];
  if (valueLabels[text]) return valueLabels[text];
  if (text.includes("|")) return text.split("|").map((part) => labelValue(part.trim())).join(" / ");
  if (text.includes(":")) return text.split(":").map((part) => labelValue(part.trim())).join(": ");
  return fallbackLabel(text);
}

function reasonCodes(value) {
  return String(value ?? "")
    .split("|")
    .flatMap((part) => {
      const trimmed = part.trim();
      if (!trimmed || trimmed.toUpperCase() === "PASS") return [];
      if (trimmed.includes(":")) {
        const parts = trimmed.split(":").map((item) => item.trim()).filter(Boolean);
        const code = parts.at(-1) || trimmed;
        return [{ raw: trimmed, scope: parts.slice(0, -1).join(":"), code }];
      }
      return [{ raw: trimmed, scope: "", code: trimmed }];
    });
}

function explainReason(code) {
  return blockReasonDescriptions[code] || "이 항목은 현재 설정한 최소 기준을 통과하지 못했습니다. 실제 수치와 기준값은 상세 표에서 확인할 수 있습니다.";
}

function explainReasonList(value, max = 2) {
  const reasons = reasonCodes(value);
  if (!reasons.length) return labelValue(value || "PASS");
  return reasons.slice(0, max).map((item) => explainReason(item.code)).join(" / ");
}

function fallbackLabel(text) {
  if (!text) return "없음";
  if (!/^[A-Z0-9_./-]+$/.test(text)) return text;
  const wordLabels = {
    ACTIONABLE: "매매 가능",
    ALL: "전체",
    ALLOWED: "허용",
    ATR: "ATR",
    BASE: "기본",
    BIN: "구간",
    BRIER: "확률 예측 오차",
    CALIBRATION: "확률 실제 오차",
    CANDIDATE: "후보",
    CLASS: "클래스",
    CONTEXT: "관찰",
    COUNT: "수",
    DATA: "데이터",
    DECISION: "판단",
    DISPLAY: "표시",
    ENTRY: "진입",
    EVENT: "이벤트",
    EXPECTANCY: "기대수익",
    FAILURES: "실패",
    FOLD: "검증 묶음",
    GATE: "통과 기준",
    HIGH: "높음",
    INSUFFICIENT: "부족",
    LATEST: "최신",
    LIVE: "실전",
    LOW: "낮음",
    MODEL: "모델",
    NONE: "없음",
    NOT: "아님",
    OOS: "테스트",
    PASS: "통과",
    POLICY: "기준",
    QUALITY: "품질",
    READY: "준비",
    RESEARCH: "분석",
    RULE: "기본 규칙",
    SELECTED: "선택",
    SIGNAL: "신호",
    STATUS: "상태",
    STOP: "손절",
    SUPPORT: "참고",
    THRESHOLD: "선택 기준값",
    TRADE: "매매",
    TRIGGER: "매수 신호",
    VALIDATION: "검증",
    WATCHLIST: "관찰 목록",
    POOLED: "여러 종목 모델",
  };
  return text
    .split("_")
    .filter(Boolean)
    .map((part) => wordLabels[part] || part)
    .join(" ");
}

function fileDisplayName(file) {
  const label = fileLabels[file.name] || file.name;
  return file.symbol ? `${label} · ${file.symbol}` : label;
}

function kindLabel(kind) {
  return fileKindLabels[kind] || kind || "파일";
}

function scopeLabel(scope) {
  return fileScopeLabels[scope] || scope || "전체";
}

function translateReportText(text) {
  const replacements = [
    ["TSM Prediction Reliability Report", "TSMC 예측 신뢰도 리포트"],
    ["TSM Prediction Accuracy Report", "TSMC 예측 정확도 리포트"],
    ["TSM ML Overlay Report", "TSMC 예측 적용 리포트"],
    ["TSM Pooled Model Report", "TSMC 통합 표본 모델 리포트"],
    ["TSM Model Registry Report", "TSMC 모델 목록 리포트"],
    ["TSM Shadow Paper Report", "TSMC 가상 기록 리포트"],
    ["TSM Daily Trading Plan", "TSMC 일일 매매 계획"],
    ["Key Levels", "주요 가격대"],
    ["Buy Restrictions", "매수 금지 조건"],
    ["Risk Policy", "리스크 정책"],
    ["Daily Stress", "일별 스트레스"],
    ["Prediction Overlay", "예측 참고 정보"],
    ["Validation Evidence", "검증 근거"],
    ["System State", "시스템 상태"],
    ["Latest scope", "최신 예측 종류"],
    ["Latest signal status", "최신 신호 상태"],
    ["Latest use status", "최신 사용 상태"],
    ["Model Audit", "모델 점검"],
    ["Calibration Detail", "확률 실제 오차 상세"],
    ["Quality Checks", "상태 점검"],
    ["Model Registry", "모델 목록"],
    ["Shadow Paper", "가상 기록"],
    ["Pooled Model", "여러 종목 모델"],
    ["ML Overlay", "예측 적용 비교"],
    ["Blocked", "기준 미달"],
    ["Passed", "문제 없음"],
    ["Interpretation", "해석"],
    ["This plan is research tooling, not investment advice.", "이 계획은 분석 도구이며 투자 조언이 아닙니다."],
    ["Trade log is an input audit dependency; labels are rebuilt independently from OHLC.", "거래 기록은 확인용으로만 사용합니다. 실제 결과 라벨은 가격 데이터에서 따로 다시 계산합니다."],
    ["out-of-sample", "학습에 쓰지 않은 테스트"],
    ["OOS", "테스트"],
    ["threshold", "선택 기준값"],
    ["calibration", "확률 실제 오차 확인"],
    ["Brier", "확률 예측 오차"],
    ["gate", "통과 기준"],
  ];
  return replacements.reduce((acc, [from, to]) => acc.replaceAll(from, to), String(text ?? "")).replace(/\bTSM\b/g, "TSMC");
}

async function fetchJson(url, options) {
  const res = await fetch(url, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

function setTextIfPresent(id, value) {
  const element = $(id);
  if (element) element.textContent = value;
}

function renderWorkspaceHeader(data) {
  const snapshots = data.snapshots || {};
  const decision = snapshots.decision || {};
  const latestPrice = snapshots.latest_price || {};
  const summary = snapshots.integrated_price || snapshots.summary || {};
  const risk = snapshots.risk || {};
  const prediction = snapshots.pooled_prediction || snapshots.prediction || {};
  const close = snapshotValue(decision, ["close", "latest_close", "latest_close_usd"])
    || snapshotValue(latestPrice, ["close", "adj_close"])
    || snapshotValue(summary, ["latest_close_usd", "end_adj_close", "latest_close"]);
  const change = snapshotValue(latestPrice, ["close_change_pct"])
    || snapshotValue(decision, ["close_change_pct"])
    || snapshotValue(summary, ["latest_close_change_pct"]);
  const score = snapshotValue(decision, ["score_price_algo_total", "research_signal_score"]);
  const signal = snapshotValue(decision, ["entry_trigger", "trade_action", "latest_entry_gate_status"]) || "NO_SIGNAL";
  const predictionValue = snapshotValue(prediction, [
    "p_success_tsm_calibrated",
    "p_success_calibrated",
    "p_success_20d",
    "p_success",
    "decision_score_tsm_like_calibrated",
  ]);
  const riskText = snapshotValue(risk, ["risk_state", "latest_entry_gate_status", "prediction_entry_gate_status"]) || "NO_NEW_RISK";
  const updated = snapshotValue(decision, ["date", "asof_date", "signal_date"])
    || snapshotValue(latestPrice, ["date"])
    || snapshotValue(summary, ["end_date", "latest_date"]);

  setTextIfPresent("tickerPrice", fmtCurrency(close));
  setTextIfPresent("tickerScore", isBlankSnapshotValue(score) ? "없음" : `${fmtNumber(score, 1)}점`);
  setTextIfPresent("tickerSignal", labelValue(signal));
  setTextIfPresent(
    "tickerPrediction",
    isBlankSnapshotValue(predictionValue) ? labelValue(prediction.prediction_use_status || prediction.prediction_status || "DISPLAY_ONLY") : fmtMaybePct(predictionValue, 1)
  );
  setTextIfPresent("tickerRisk", labelValue(riskText));
  setTextIfPresent("tickerUpdated", shortDate(updated));

  const changeElement = $("tickerChange");
  if (changeElement) {
    const n = toNumber(change);
    changeElement.textContent = n === null ? "-" : fmtMaybePct(n, 2);
    changeElement.className = n > 0 ? "positive" : n < 0 ? "negative" : "";
  }
}

async function loadData() {
  $("asOfText").textContent = "업데이트 중";
  const data = await fetchJson("/api/summary");
  state.data = data;
  renderAll();
  updateRunState(data.run || {});
  const rootName = String(data.paths?.root || "").split("/").filter(Boolean).at(-1) || "작업 폴더";
  $("asOfText").textContent = `생성 ${fmtKstDateTime(data.generated_at)} · ${rootName}`;
  $("asOfText").title = data.paths?.root || "";
}

function renderAll() {
  const data = state.data;
  if (!data) return;
  renderWorkspaceHeader(data);
  renderOverview(data);
  renderDecision(data);
  renderMarket(data);
  renderDataView(data);
  renderNews(data);
  renderPrediction(data);
  renderModels(data);
  renderDiagnostics(data);
  renderBacktest(data);
  renderRisk(data);
  renderQuality(data);
  renderSystem(data);
  renderFiles(data);
  renderRunPage(data);
}

function renderOverview(data) {
  renderOntologyCommand("ontologyCommand", data);
  renderOverviewSystemMap("overviewSystemMap", data);
  renderOntologyMap("ontologyMap", data);
  renderOntologyLinkedCharts("ontologyLinkedCharts", data);
  renderOntologyRelationMatrix("ontologyRelationMatrix", data);
  renderOntologyPriorityStack("ontologyPriorityStack", data);
  renderOntologyRiskReward("ontologyRiskReward", data);
  renderOntologyEvidenceRail("ontologyEvidenceRail", data);
  renderOntologyBlockGraph("ontologyBlockGraph", data);
  renderOntologyNewsCausal("ontologyNewsCausal", data);
}

function renderPageSynthesis(id, synthesis) {
  const container = $(id);
  if (!container || !synthesis) return;
  const score = clampNumber(toNumber(synthesis.score) ?? scoreFromTone(synthesis.tone), 0, 100);
  const tone = synthesis.tone || toneFromScore(score);
  const drivers = (synthesis.drivers || []).slice(0, 6);
  const blockers = (synthesis.blockers || []).slice(0, 5);
  const conflicts = (synthesis.conflicts || []).slice(0, 3);
  const actions = (synthesis.nextActions || []).slice(0, 5);
  const evidence = (synthesis.evidenceRefs || []).slice(0, 5);
  const blockerItems = [
    ...blockers.map((item) => normalizeSynthesisItem(item, "병목")),
    ...conflicts.map((item) => normalizeSynthesisItem(item, "모순")),
  ];
  container.innerHTML = `
    <section class="page-synthesis-board ${tone}" aria-label="${escapeHtml(synthesis.title || "합성 판단")}">
      <article class="page-verdict ${tone}" style="--score:${score}; --tone-color:${toneColor(tone)}">
        <span>합성 결론 · ${escapeHtml(synthesis.objectType || "Ontology")}</span>
        <strong>${escapeHtml(synthesis.verdict || "상태 확인")}</strong>
        <p>${escapeHtml(synthesis.meaning || "가격, 데이터, 모델, 리스크 객체를 다시 합성합니다.")}</p>
        <div class="page-verdict-meter">
          <div><i></i></div>
          <b>${escapeHtml(fmtNumber(score, 0))}</b>
        </div>
      </article>
      <section class="page-synthesis-main">
        <div class="page-synthesis-head">
          <div>
            <h2>주요 동인</h2>
            <p>서로 다른 객체를 같은 판단 축으로 환산해서 영향이 큰 순서로 표시</p>
          </div>
          <strong>${escapeHtml(synthesis.title || "객체 합성")}</strong>
        </div>
        <div class="page-driver-grid">
          ${drivers.map(pageDriverHtml).join("") || `<div class="preview-empty">동인 없음</div>`}
        </div>
      </section>
      <section class="page-synthesis-side">
        <article class="page-synthesis-block">
          <div>
            <h2>병목/모순</h2>
            <p>판단을 막거나 근거 사이에 충돌이 생기는 지점</p>
          </div>
          <div class="page-item-list">
            ${blockerItems.map(pageSynthesisItemHtml).join("") || `<div class="page-empty-state">현재 첫 화면에서 막는 핵심 병목 없음</div>`}
          </div>
        </article>
        <article class="page-synthesis-block action">
          <div>
            <h2>다음 액션</h2>
            <p>오늘 먼저 확인하거나 다음 실행에서 바뀌어야 할 조건</p>
          </div>
          <div class="page-item-list">
            ${actions.map((item) => pageSynthesisItemHtml(normalizeSynthesisItem(item, "액션"))).join("") || `<div class="page-empty-state">추가 액션 없음</div>`}
          </div>
        </article>
        <article class="page-synthesis-block evidence">
          <div>
            <h2>원본 증거</h2>
            <p>아래 상세 표와 산출물에서 추적할 객체</p>
          </div>
          <div class="page-evidence-list">
            ${evidence.map(pageEvidenceHtml).join("") || `<div class="page-empty-state">증거 링크 없음</div>`}
          </div>
        </article>
      </section>
    </section>
  `;
}

function renderOverviewSystemMap(id, data) {
  const container = $(id);
  if (!container) return;
  const pages = [
    ["decision", "매매", deriveDecisionSynthesis(data)],
    ["market", "가격", deriveMarketSynthesis(data)],
    ["data", "데이터", deriveDataSynthesis(data)],
    ["news", "뉴스", deriveNewsSynthesis(data)],
    ["prediction", "예측", derivePredictionSynthesis(data)],
    ["models", "모델", deriveModelSynthesis(data)],
    ["backtest", "백테스트", deriveBacktestSynthesis(data)],
    ["risk", "위험", deriveRiskSynthesis(data)],
    ["quality", "검증", deriveQualitySynthesis(data)],
    ["system", "운영", deriveSystemSynthesis(data)],
  ];
  container.innerHTML = `
    <section class="panel ontology-panel overview-system-panel">
      <div class="panel-head">
        <div><h2>전체 시스템 맵</h2><p>각 페이지의 결론을 하나의 판단 네트워크로 묶어 어디가 기회이고 어디가 병목인지 표시</p></div>
      </div>
      <div class="system-map-grid">
        ${pages.map(([view, label, synthesis]) => overviewSystemCardHtml(view, label, synthesis)).join("")}
      </div>
    </section>
  `;
  container.querySelectorAll("[data-jump-view]").forEach((button) => button.addEventListener("click", () => activateView(button.dataset.jumpView)));
}

function overviewSystemCardHtml(view, label, synthesis) {
  const score = clampNumber(toNumber(synthesis.score) ?? scoreFromTone(synthesis.tone), 0, 100);
  const tone = synthesis.tone || toneFromScore(score);
  const primaryAction = normalizeSynthesisItem((synthesis.nextActions || [])[0], "액션");
  return `
    <button class="system-map-card ${tone}" type="button" data-jump-view="${escapeHtml(view)}" style="--score:${score}; --tone-color:${toneColor(tone)}">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(synthesis.verdict)}</strong>
      <p>${escapeHtml(primaryAction.text || synthesis.meaning)}</p>
      <i><b></b></i>
    </button>
  `;
}

function pageDriverHtml(driver) {
  const score = clampNumber(toNumber(driver.score) ?? scoreFromTone(driver.tone), 0, 100);
  const tone = driver.tone || toneFromScore(score);
  return `
    <article class="page-driver-card ${tone}" style="--score:${score}; --tone-color:${toneColor(tone)}">
      <div>
        <span>${escapeHtml(driver.label || "동인")}</span>
        <strong>${escapeHtml(driver.value || "없음")}</strong>
      </div>
      <p>${escapeHtml(driver.explanation || "")}</p>
      <div class="driver-meter"><i></i></div>
      <em>${escapeHtml(driver.source || "계산값")}</em>
    </article>
  `;
}

function pageSynthesisItemHtml(item) {
  return `
    <div class="page-synthesis-item ${item.tone || "neutral"}">
      <span>${escapeHtml(item.label || "항목")}</span>
      <strong>${escapeHtml(item.text || "없음")}</strong>
    </div>
  `;
}

function pageEvidenceHtml(ref) {
  return `
    <div class="page-evidence-ref">
      <span>${escapeHtml(ref.objectType || "객체")}</span>
      <strong>${escapeHtml(ref.tableKey || ref.fileName || "원본")}</strong>
      <p>${escapeHtml(ref.rowHint || ref.fileName || "상세 영역에서 확인")}</p>
    </div>
  `;
}

function normalizeSynthesisItem(item, fallbackLabel) {
  if (typeof item === "string") return { label: fallbackLabel, text: item, tone: "neutral" };
  return {
    label: item?.label || fallbackLabel,
    text: item?.text || item?.value || item?.title || "없음",
    tone: item?.tone || "neutral",
  };
}

function synthesis(title, objectType, verdict, meaning, tone, score, drivers, blockers, conflicts, nextActions, evidenceRefs) {
  return {
    title,
    objectType,
    verdict,
    meaning,
    tone: tone || toneFromScore(score),
    score: clampNumber(toNumber(score) ?? scoreFromTone(tone), 0, 100),
    drivers: drivers || [],
    blockers: blockers || [],
    conflicts: conflicts || [],
    nextActions: nextActions || [],
    evidenceRefs: evidenceRefs || [],
  };
}

function driver(label, value, score, explanation, source, tone = null) {
  return { label, value, score: clampNumber(toNumber(score) ?? scoreFromTone(tone), 0, 100), explanation, source, tone: tone || toneFromScore(score) };
}

function item(label, text, tone = "neutral") {
  return { label, text, tone };
}

function evidenceRef(objectType, tableKey, fileName = "", rowHint = "") {
  return { objectType, tableKey, fileName, rowHint };
}

function toneColor(tone) {
  if (tone === "good") return colors.green;
  if (tone === "warn") return colors.amber;
  if (tone === "bad") return colors.red;
  return colors.blue;
}

function toneFromScore(score) {
  const n = toNumber(score);
  if (n === null) return "neutral";
  if (n >= 72) return "good";
  if (n < 45) return "warn";
  return "neutral";
}

function scoreFromTone(tone) {
  if (tone === "good") return 78;
  if (tone === "warn") return 46;
  if (tone === "bad") return 24;
  return 58;
}

function boolScore(value, trueScore = 82, falseScore = 36) {
  if (value === null || value === undefined || value === "") return 50;
  return isTruthy(value) ? trueScore : falseScore;
}

function statusScore(value) {
  const tone = statusTone(value);
  if (tone === "good") return 84;
  if (tone === "warn") return 44;
  if (tone === "bad") return 22;
  return 58;
}

function averageScore(values) {
  const nums = values.map((value) => toNumber(value)).filter((value) => value !== null && Number.isFinite(value));
  if (!nums.length) return 50;
  return nums.reduce((sum, value) => sum + value, 0) / nums.length;
}

function latestSeriesValue(rows, key) {
  const row = (rows || []).slice().reverse().find((candidate) => toNumber(candidate[key]) !== null);
  return row ? toNumber(row[key]) : null;
}

function avgNumber(rows, key) {
  const nums = (rows || []).map((row) => toNumber(row[key])).filter((value) => value !== null && Number.isFinite(value));
  if (!nums.length) return null;
  return nums.reduce((sum, value) => sum + value, 0) / nums.length;
}

function passRate(rows, key = "passed") {
  if (!(rows || []).length) return null;
  return (rows.filter((row) => isTruthy(row[key]) || String(row.status || "").toUpperCase() === "PASS").length / rows.length) * 100;
}

function worstByAbs(rows, key) {
  return (rows || [])
    .filter((row) => toNumber(row[key]) !== null)
    .slice()
    .sort((a, b) => Math.abs(toNumber(b[key])) - Math.abs(toNumber(a[key])))[0] || {};
}

function hasEntryTrigger(decision) {
  const trigger = String(decision.entry_trigger || "").toUpperCase();
  return Boolean(trigger && !["NONE", "NO_ENTRY_TRIGGER", "NO_SIGNAL"].includes(trigger));
}

function probabilityScore(value, threshold = 50) {
  const p = probabilityPercent(value);
  if (p === null) return 50;
  return clampNumber(50 + (p - threshold) * 1.8, 0, 100);
}

function deriveDecisionSynthesis(data) {
  const decision = data.snapshots.decision || {};
  const risk = data.snapshots.risk || {};
  const prediction = data.snapshots.prediction || {};
  const pooled = data.snapshots.pooled_prediction || {};
  const system = data.snapshots.system || {};
  const news = data.snapshots.latest_news || {};
  const dataQuality = data.snapshots.data_quality || {};
  const state = overviewDecisionState(decision, prediction, system);
  const score = toNumber(decision.score_price_algo_total) || 0;
  const trigger = hasEntryTrigger(decision);
  const researchStage = String(decision.research_signal_stage || "NO_RESEARCH_SIGNAL").toUpperCase();
  const researchSignalScore = toNumber(decision.research_signal_score);
  const researchScore = researchSignalScore === null ? (researchStage === "EARLY_BULLISH_WATCH" ? 66 : researchStage === "PAPER_BUY_SETUP" ? 74 : 38) : researchSignalScore;
  const maxWeight = snapshotWeightValue(risk, ["final_recommended_max_weight", "final_recommended_max_weight_pct"]);
  const pSuccess = snapshotValue(pooled, ["p_success_20d", "p_success_tsm_like_20d"]) || snapshotValue(prediction, ["trade_ready_p_success_20d", "trigger_p_success_20d", "context_p_success_20d"]);
  const pStop = snapshotValue(pooled, ["p_stop_hit_20d"]) || snapshotValue(prediction, ["trade_ready_p_stop_hit_20d", "trigger_p_stop_hit_20d", "context_p_stop_hit_20d"]);
  const riskScore = maxWeight === null ? 42 : clampNumber(maxWeight * 100 * 14, 20, 92);
  const finalScore = averageScore([score, trigger ? 82 : 42, researchScore, riskScore, probabilityScore(pSuccess, 52), boolScore(system.prediction_decision_support), statusScore(dataQuality.data_quality_status)]);
  const blockers = [];
  if (!trigger) blockers.push(item("가격", "진입 트리거가 아직 발생하지 않았습니다.", "warn"));
  if (!isTruthy(system.prediction_decision_support)) blockers.push(item("예측", "모델은 판단 근거가 아니라 참고용 상태입니다.", "warn"));
  if (maxWeight !== null && maxWeight < 0.03) blockers.push(item("리스크", `허용 비중이 ${fmtMaybePct(maxWeight, 2)}로 낮아 실행 크기가 제한됩니다.`, "warn"));
  const block = primaryBlockReason(data);
  if (block) blockers.push(item("게이트", `${labelValue(block.code)} 기준이 ${fmtNumber(block.count, 0)}번 반복 차단됩니다.`, "warn"));
  const conflicts = [];
  if (score >= 60 && !trigger) conflicts.push(item("룰/행동", "점수는 관찰권이지만 행동 트리거는 아직 없습니다.", "warn"));
  if (["EARLY_BULLISH_WATCH", "PAPER_BUY_SETUP"].includes(researchStage) && !trigger) conflicts.push(item("2단계 신호", "실전 신호는 아니지만 연구/Paper 단계는 강화됐습니다.", "neutral"));
  if (probabilityPercent(pSuccess) !== null && probabilityPercent(pStop) !== null && probabilityPercent(pSuccess) - probabilityPercent(pStop) < 8) conflicts.push(item("예측/손절", "성공 확률과 손절 위험의 간격이 좁습니다.", "warn"));
  return synthesis(
    "매매 의사결정 파이프라인",
    "Decision Ontology",
    state.title,
    `가격 트리거, 모델 허용, 리스크 비중, 뉴스 원인, Paper 상태를 합치면 현재 결론은 ${state.title}입니다.`,
    state.tone,
    finalScore,
    [
      driver("가격 트리거", labelValue(decision.entry_trigger || "NO_ENTRY_TRIGGER"), trigger ? 82 : 38, `${fmtNumber(score, 1)}점 · 종가 ${fmtCurrency(decision.close)}`, "decision snapshot", trigger ? "good" : "warn"),
      driver("연구/Paper 신호", labelValue(decision.research_signal_stage || "NO_RESEARCH_SIGNAL"), researchScore, labelValue(decision.research_signal_reason || decision.research_signal_action), "decision snapshot", ["EARLY_BULLISH_WATCH", "PAPER_BUY_SETUP"].includes(researchStage) ? "good" : "warn"),
      driver("리스크 비중", fmtMaybePct(maxWeight, 2), riskScore, `${labelValue(risk.risk_state)} · ${labelValue(risk.limiting_reason)}`, "risk snapshot", gateTone(riskStatusFromSnapshot(data))),
      driver("20D 예측", fmtMaybePct(pSuccess, 1), probabilityScore(pSuccess, 52), `손절 ${fmtMaybePct(pStop, 1)} · ${labelValue(pooled.model_name || prediction.best_model_20d)}`, "prediction snapshots"),
      driver("뉴스 원인", labelValue(news.news_primary_cause_type || "NO_HIGH_CONFIDENCE_NEWS"), newsTone(news) === "warn" ? 42 : 62, `${labelValue(news.news_match_confidence || "NO_MATCH")} · 감점 ${fmtNumber(news.news_penalty_event, 1)}`, "latest news", newsTone(news)),
      driver("Paper/Live", labelValue(system.paper_trading_status || "PAPER_RECORD_ONLY"), boolScore(system.paper_ready, 76, 44), `실거래 ${labelValue(system.live_trading_status || "DISABLED_BY_DESIGN")}`, "system snapshot", isTruthy(system.paper_ready) ? "good" : "warn"),
    ],
    blockers,
    conflicts,
    [
      item("1순위", trigger ? "Paper 기록과 리스크 비중을 확인합니다." : buildSynthesisNextAction({ hasTrigger: false, predictionAllowed: isTruthy(system.prediction_decision_support), riskCapacity: riskScore, dataIntegrity: statusScore(dataQuality.data_quality_status), block, planRows: data.tables.trading_plan || [] }), "neutral"),
      item("2순위", "모델 차단 원인을 원본 진단 페이지에서 실제 값/기준값으로 확인합니다.", block ? "warn" : "neutral"),
      item("3순위", "실제 브로커 주문은 계속 제외하고 연구·Paper 기록만 유지합니다.", "neutral"),
    ],
    [
      evidenceRef("판단", "trading_plan", "tsm_daily_trading_plan.csv", "가격 트리거와 행동 계획"),
      evidenceRef("리스크", "risk snapshot", "tsm_risk_policy.csv", "허용 비중과 손절가"),
      evidenceRef("예측", "pooled_prediction", "tsm_latest_pooled_prediction.csv", "20D 확률과 모델 게이트"),
    ]
  );
}

function deriveMarketSynthesis(data) {
  const decision = data.snapshots.decision || {};
  const summary = data.snapshots.integrated_price || data.snapshots.summary || {};
  const close = toNumber(decision.close ?? summary.latest_close_usd);
  const dist20 = toNumber(decision.dist_close_sma_20_pct);
  const dist50 = toNumber(decision.dist_close_sma_50_pct);
  const dist200 = toNumber(decision.dist_close_sma_200_pct);
  const dd = toNumber(decision.drawdown_from_ath ?? summary.latest_drawdown_from_ath_pct);
  const vol20 = toNumber(decision.vol_20d_ann ?? summary.latest_vol20_ann_pct ?? summary.latest_20d_ann_vol_pct);
  const relSmh = toNumber(decision.relative_return_vs_smh_60d);
  const trendScore = clampNumber(50 + (dist20 || 0) * 2 + (dist50 || 0) * 1.2 + (dist200 || 0) * 0.6, 0, 100);
  const drawdownScore = clampNumber(84 + (dd || 0) * 2.8, 0, 100);
  const volatilityScore = clampNumber(92 - Math.abs(vol20 || 0) * 1.2, 10, 92);
  const relativeScore = clampNumber(50 + (relSmh || 0) * 2.5, 0, 100);
  const score = averageScore([trendScore, drawdownScore, volatilityScore, relativeScore, toNumber(decision.score_momentum) || 50]);
  const blockers = [];
  if (dd !== null && dd < -10) blockers.push(item("낙폭", `고점 대비 ${fmtMaybePct(dd, 1)}로 회복 확인이 필요합니다.`, "warn"));
  if (vol20 !== null && vol20 > 45) blockers.push(item("변동성", `20일 변동성이 ${fmtMaybePct(vol20, 1)}로 비중을 제한합니다.`, "warn"));
  if (dist50 !== null && dist50 < 0) blockers.push(item("추세", `50일선 대비 ${fmtMaybePct(dist50, 1)}로 중기 추세가 약합니다.`, "warn"));
  return synthesis(
    "시장 구조 판정",
    "Market Ontology",
    score >= 70 ? "상승 구조 우위" : score < 45 ? "방어 구조" : "혼합 구조 관찰",
    `종가, 평균선 거리, 낙폭, 변동성, SMH/SPY 상대강도를 합성한 시장 구조 점수는 ${fmtNumber(score, 0)}입니다.`,
    toneFromScore(score),
    score,
    [
      driver("추세 체제", labelValue(decision.trend_regime || summary.latest_trend_regime), trendScore, `20D ${fmtMaybePct(dist20, 1)} · 50D ${fmtMaybePct(dist50, 1)} · 200D ${fmtMaybePct(dist200, 1)}`, "price series"),
      driver("돌파 거리", fmtCurrency(close), trendScore, `20일 고점 돌파와 60일 고점 돌파 기준을 매매 계획에서 연결`, "trading_plan"),
      driver("낙폭", fmtMaybePct(dd, 1), drawdownScore, `고점 대비 하락폭이 리스크 허용 비중과 직접 연결됩니다.`, "price series", drawdownScore < 45 ? "warn" : "neutral"),
      driver("변동성", fmtMaybePct(vol20, 1), volatilityScore, `ATR ${fmtMaybePct(decision.atr_14_pct, 2)} · 변동성 체제 ${labelValue(decision.vol_regime)}`, "price/risk series", volatilityScore < 45 ? "warn" : "neutral"),
      driver("SMH 상대강도", fmtMaybePct(relSmh, 1), relativeScore, `SPY 대비 ${fmtMaybePct(decision.relative_return_vs_spy_60d, 1)} · 베타 ${fmtNumber(decision.beta_vs_spy_252d, 2)}`, "relative strength"),
    ],
    blockers,
    [
      dist20 !== null && dist20 > 0 && !hasEntryTrigger(decision) ? item("가격/룰", "평균선 위에 있지만 룰 트리거는 아직 진입으로 번역하지 않았습니다.", "warn") : null,
    ].filter(Boolean),
    [
      item("가격 확인", "가격 페이지의 평균선 차트에서 현재가와 20D/60D 돌파 기준의 거리를 확인합니다."),
      item("상대강도 확인", "SMH와 SPY 대비 60일 상대강도가 동시에 개선되는지 봅니다."),
      item("변동성 확인", "변동성이 높으면 리스크 페이지의 권장 비중이 먼저 제한됩니다.", volatilityScore < 50 ? "warn" : "neutral"),
    ],
    [
      evidenceRef("가격", "price series", "tsm_daily_10y_enriched.csv", "종가/평균선/변동성"),
      evidenceRef("가격", "yearly_price", "tsm_yearly_price.csv", "연간 구조"),
      evidenceRef("이벤트", "event_analysis", "tsm_event_analysis.csv", "이벤트 이후 가격"),
    ]
  );
}

function deriveDataSynthesis(data) {
  const dq = data.snapshots.data_quality || {};
  const daily = data.snapshots.summary || {};
  const hourly = data.snapshots.hourly_summary || {};
  const minute = data.snapshots.minute_summary || {};
  const failedChecks = toNumber(dq.failed_checks) || 0;
  const critical = toNumber(dq.critical_failed_checks) || 0;
  const sourceRows = [...(data.tables.hourly_source_audit || []), ...(data.tables.minute_source_audit || [])];
  const sourcePass = passRate(sourceRows, "supports_complete_requested_10y");
  const dataScore = clampNumber(statusScore(dq.data_quality_status) - failedChecks * 6 - critical * 14, 0, 100);
  const hourlyScore = isTruthy(hourly.complete_requested_coverage) ? 82 : 52;
  const minuteScore = isTruthy(minute.complete_requested_coverage) ? 82 : 52;
  const score = averageScore([dataScore, hourlyScore, minuteScore, sourcePass ?? 58]);
  const blockers = [];
  if (failedChecks) blockers.push(item("품질 점검", `${fmtNumber(failedChecks, 0)}개 데이터 점검이 실패했습니다.`, failedChecks > 3 ? "warn" : "neutral"));
  if (!isTruthy(hourly.complete_requested_coverage)) blockers.push(item("시간봉", "공개 소스 한계로 요청한 10년 전체 시간봉을 완전히 채우지 못했습니다.", "warn"));
  if (!isTruthy(minute.complete_requested_coverage)) blockers.push(item("분봉", "분봉은 공개 소스 부분 범위로 보는 보조 증거입니다.", "warn"));
  return synthesis(
    "데이터 라인리지 보드",
    "Data Ontology",
    dataScore >= 70 ? "판단 입력 사용 가능" : "입력 검증 우선",
    `일봉, 시간봉, 분봉 최신성, 출처 신뢰, 이벤트 충격, 이상치 점검을 하나의 데이터 계보로 묶었습니다.`,
    dataScore >= 70 ? "good" : "warn",
    score,
    [
      driver("일봉 라인리지", `${shortDate(daily.start_date)}~${shortDate(daily.end_date)}`, statusScore(dq.data_quality_status), `${fmtNumber(daily.trading_days, 0)}거래일 · 소스 ${labelValue(dq.data_source || daily.data_source)}`, "summary/data_quality", statusTone(dq.data_quality_status)),
      driver("시간봉 최신성", shortDate(hourly.end_timestamp), hourlyScore, `${fmtNumber(hourly.bars, 0)}개 바 · ${labelValue(hourly.data_source)}`, "hourly summary", hourlyScore > 70 ? "good" : "warn"),
      driver("분봉 최신성", shortDate(minute.end_timestamp), minuteScore, `${fmtNumber(minute.bars, 0)}개 바 · ${labelValue(minute.data_source)}`, "minute summary", minuteScore > 70 ? "good" : "warn"),
      driver("출처 감사", sourcePass === null ? "없음" : `${fmtNumber(sourcePass, 0)}%`, sourcePass ?? 50, `${fmtNumber(sourceRows.length, 0)}개 provider/interval 감사`, "source audit"),
      driver("이벤트 충격", `${fmtNumber((data.tables.event_analysis || []).length, 0)}개`, 62, `급등락 원인 ${fmtNumber((data.tables.extreme_moves || []).length, 0)}건과 연결`, "event/extreme tables"),
      driver("이상치 점검", labelValue(dq.data_quality_status), dataScore, `critical ${fmtNumber(critical, 0)} · warn ${fmtNumber(dq.warn_failed_checks, 0)}`, "data_quality_checks", dataScore < 50 ? "warn" : "good"),
    ],
    blockers,
    [
      item("해석 주의", "일봉은 핵심 판단 입력이고 시간봉/분봉은 공개 소스 범위 차이 때문에 보조 계층입니다.", "neutral"),
    ],
    [
      item("품질 점검", "실패한 data_quality_checks를 먼저 확인합니다.", failedChecks ? "warn" : "neutral"),
      item("라인리지", "sourceAuditTable에서 provider별 요청 범위와 실제 범위를 비교합니다."),
      item("이벤트", "가격 급등락 원인 테이블에서 데이터 이상치와 실제 뉴스 원인을 분리합니다."),
    ],
    [
      evidenceRef("데이터", "data_quality_checks", "tsm_data_quality_checks.csv", "품질 점검"),
      evidenceRef("라인리지", "hourly_source_audit/minute_source_audit", "", "provider 범위"),
      evidenceRef("이벤트", "event_analysis/extreme_moves", "", "가격 충격"),
    ]
  );
}

function deriveNewsSynthesis(data) {
  const news = data.snapshots.latest_news || {};
  const features = data.snapshots.latest_prediction_features || {};
  const penaltySummary = data.snapshots.news_penalty_summary || {};
  const highRows = data.tables.news_high_matches || [];
  const clusterRows = data.tables.news_clusters || [];
  const forwardRows = data.tables.news_cause_forward || [];
  const cause = news.news_primary_cause_type || "NO_HIGH_CONFIDENCE_NEWS";
  const confidence = String(news.news_match_confidence || "NO_MATCH").toUpperCase();
  const confidenceScore = confidence === "HIGH" ? 82 : confidence === "MEDIUM" ? 62 : 42;
  const penalty = Math.abs(toNumber(features.news_penalty_event ?? news.news_penalty_event) || 0);
  const penaltyScore = clampNumber(76 - penalty * 18, 18, 82);
  const sameCauseScore = probabilityScore(features.hist_news_category_success_rate_20d, 50);
  const highScore = clampNumber(Math.log10(highRows.length + 1) * 30, 30, 86);
  const score = averageScore([confidenceScore, penaltyScore, sameCauseScore, highScore]);
  const matchedForward = forwardRows.find((row) => String(row.rule || "").includes(cause) && String(row.horizon_days) === "20") || forwardRows.find((row) => String(row.horizon_days) === "20") || {};
  const blockers = [];
  if (confidence === "NO_MATCH") blockers.push(item("대표 원인", "최신 뉴스에 고신뢰 대표 원인이 없습니다.", "neutral"));
  if (penalty > 0) blockers.push(item("뉴스 감점", `고신뢰 부정 이벤트 감점 ${fmtNumber(penalty, 1)}점이 룰 점수에 반영됩니다.`, "warn"));
  if (!(data.tables.news_feature_contract || []).length) blockers.push(item("피처 계약", "예측 피처 계약 표가 비어 있습니다.", "warn"));
  return synthesis(
    "뉴스 원인 그래프",
    "News Ontology",
    confidence === "HIGH" ? "고신뢰 원인 연결" : "대표 원인 미확정",
    `뉴스 클러스터, 신뢰도, 감점, 과거 20D/60D 결과, 예측 피처 반영 여부를 원인 그래프로 합성했습니다.`,
    score >= 70 ? "good" : "neutral",
    score,
    [
      driver("대표 원인", labelValue(cause), confidenceScore, `${labelValue(confidence)} · 점수 ${fmtNumber(news.news_match_confidence_score, 1)}`, "latest news", confidenceScore >= 75 ? "good" : "neutral"),
      driver("뉴스 감점", penalty > 0 ? `-${fmtNumber(penalty, 1)}점` : "감점 없음", penaltyScore, `전체 감점일 ${fmtNumber(penaltySummary.penalty_days, 0)} · 최근 120일 ${fmtNumber(penaltySummary.recent_120_penalty_days, 0)}`, "news penalty", penalty > 0 ? "warn" : "good"),
      driver("원인 클러스터", `${fmtNumber(clusterRows.length, 0)}개`, clampNumber(Math.log10(clusterRows.length + 1) * 32, 32, 88), `HIGH 매칭 ${fmtNumber(highRows.length, 0)}건`, "news clusters"),
      driver("과거 20D 결과", fmtMaybePct(matchedForward.mean_fwd_return_pct, 1), probabilityScore(matchedForward.win_rate_pct, 50), `승률 ${fmtMaybePct(matchedForward.win_rate_pct, 1)} · 표본 ${fmtNumber(matchedForward.n_signals, 0)}`, "news_cause_forward"),
      driver("예측 피처 반영", `${fmtNumber(features.hist_news_category_count_20d, 0)}건`, sameCauseScore, `성공률 ${fmtMaybePct(features.hist_news_category_success_rate_20d, 1)} · 평균 ${fmtMaybePct(features.hist_news_category_mean_return_20d, 1)}`, "latest prediction features"),
    ],
    blockers,
    [
      confidence !== "HIGH" && highRows.length ? item("최신/과거", "과거에는 HIGH 원인이 있지만 최신일에는 대표 원인이 확정되지 않았습니다.", "neutral") : null,
    ].filter(Boolean),
    [
      item("최신 원인", "newsFeatureFacts에서 최신 원인이 룰 점수와 예측 피처에 어떻게 들어갔는지 확인합니다."),
      item("감점 이력", "감점이 있는 날은 뉴스 위험 감점 이력에서 원문과 가격 반응을 함께 봅니다.", penalty > 0 ? "warn" : "neutral"),
      item("원인 학습", "같은 원인 20D/60D 결과가 충분하지 않으면 예측보다 원문 증거를 우선합니다."),
    ],
    [
      evidenceRef("뉴스", "news_clusters", "tsm_news_clusters.csv", "기사 묶음"),
      evidenceRef("뉴스", "news_cause_forward", "tsm_news_cause_forward.csv", "원인별 이후 결과"),
      evidenceRef("피처", "news_feature_matrix", "tsm_news_feature_matrix.csv", "예측 입력"),
    ]
  );
}

function derivePredictionSynthesis(data) {
  const p = data.snapshots.prediction || {};
  const pooled = data.snapshots.pooled_prediction || {};
  const source = predictionDisplaySource(p, pooled);
  const pSuccess = source.pSuccess20;
  const threshold = source.threshold ?? pooled.threshold_20d ?? p.threshold_20d;
  const pStop = pooled.p_stop_hit_20d ?? p.trade_ready_p_stop_hit_20d ?? p.trigger_p_stop_hit_20d ?? p.context_p_stop_hit_20d;
  const expectedR = pooled.expected_r_net_20d ?? p.trade_ready_expected_r_20d ?? p.expected_r_20d;
  const ece = pooled.decision_ece ?? pooled.ece ?? p.trade_ready_ece_20d;
  const pScore = threshold ? clampNumber(52 + (probabilityPercent(pSuccess) - probabilityPercent(threshold)) * 2.1, 0, 100) : probabilityScore(pSuccess, 50);
  const calibrationScore = toNumber(ece) === null ? 50 : clampNumber(90 - toNumber(ece) * 360, 15, 90);
  const stopScore = probabilityPercent(pStop) === null ? 50 : clampNumber(90 - probabilityPercent(pStop) * 1.15, 10, 90);
  const score = averageScore([pScore, calibrationScore, stopScore, boolScore(pooled.decision_support_allowed), boolScore(pooled.model_quality_pass)]);
  const blockers = [];
  if (!isTruthy(pooled.decision_support_allowed)) blockers.push(item("판단 허용", labelValue(pooled.decision_block_reasons || "모델 기준 미달"), "warn"));
  if (!isTruthy(pooled.model_quality_pass)) blockers.push(item("모델 품질", labelValue(pooled.model_quality_block_reasons || p.model_quality_block_reasons_20d), "warn"));
  if (probabilityPercent(pSuccess) !== null && probabilityPercent(threshold) !== null && probabilityPercent(pSuccess) < probabilityPercent(threshold)) blockers.push(item("선택 기준", "성공 확률이 현재 선택 기준보다 낮습니다.", "warn"));
  return synthesis(
    "예측 사용 가능성",
    "Prediction Ontology",
    isTruthy(pooled.decision_support_allowed) ? "판단 참고 가능" : "참고용 표시",
    `단독 예측, pooled 예측, threshold, calibration, stop risk, expected R을 합쳐 예측을 실제 판단에 연결할 수 있는지 판정합니다.`,
    isTruthy(pooled.decision_support_allowed) ? "good" : "warn",
    score,
    [
      driver("대표 예측", fmtMaybePct(pSuccess, 1), pScore, `${source.scopeLabel} · 기준 ${fmtMaybePct(threshold, 1)}`, "prediction snapshots", pScore >= 70 ? "good" : "warn"),
      driver("Pooled 예측", fmtMaybePct(pooled.p_success_20d, 1), probabilityScore(pooled.p_success_20d, 52), `${labelValue(pooled.model_name)} · 표본 ${fmtNumber(pooled.oos_event_count, 0)}`, "pooled_prediction"),
      driver("Calibration", toNumber(ece) === null ? "없음" : `ECE ${fmtNumber(ece, 3)}`, calibrationScore, `Brier 개선 ${fmtMaybePct(pooled.brier_improvement_pct, 2)} · route ${labelValue(pooled.tsm_calibration_route || pooled.tsm_like_calibration_route)}`, "calibration metrics", calibrationScore < 45 ? "warn" : "good"),
      driver("Stop Risk", fmtMaybePct(pStop, 1), stopScore, `손절 회피 ${fmtMaybePct(pooled.p_stop_survival_20d ?? source.pStopSurvival20, 1)} · 기대 R ${fmtNumber(expectedR, 2)}`, "prediction risk", stopScore < 45 ? "warn" : "neutral"),
      driver("Policy Audit", `${fmtNumber((data.tables.prediction_policy_audit || []).length, 0)}개`, passRate(data.tables.prediction_policy_audit || [], "status") ?? 62, "정책 감사와 피처 계약은 상세 접힘 영역에서 확인", "policy audit"),
    ],
    blockers,
    [
      !isTruthy(pooled.decision_support_allowed) && pScore >= 65 ? item("확률/게이트", "확률 자체는 읽을 만하지만 게이트가 판단 연결을 막습니다.", "warn") : null,
    ].filter(Boolean),
    [
      item("게이트 원인", "원본 진단에서 모델 게이트의 실제 값과 기준값을 확인합니다.", blockers.length ? "warn" : "neutral"),
      item("확률 검증", "calibration chart에서 예상 확률과 실제 성공률의 벌어짐을 확인합니다."),
      item("운영 원칙", "기준 미달 예측은 숨기지 않고 표시하지만 매매 판단에는 참고용으로 제한합니다."),
    ],
    [
      evidenceRef("예측", "prediction_comparison", "tsm_prediction_comparison.csv", "모델별 OOS"),
      evidenceRef("예측", "pooled_threshold_policy", "tsm_pooled_threshold_policy.csv", "선택 기준"),
      evidenceRef("예측", "prediction_calibration_summary", "tsm_prediction_calibration_summary.csv", "확률 오차"),
    ]
  );
}

function deriveModelSynthesis(data) {
  const pooled = data.snapshots.pooled_prediction || {};
  const universe = data.snapshots.universe_validation || {};
  const sample = data.snapshots.pooled_sample_audit || {};
  const paper = data.snapshots.paper_gate || {};
  const research = data.snapshots.research_expansion || {};
  const champion = (data.tables.pooled_model_comparison || []).find((row) => isTruthy(row.is_champion)) || {};
  const root = primaryBlockReason(data);
  const sampleScore = clampNumber(((toNumber(sample.trade_ready_20d_labeled) || 0) / 10000) * 100, 0, 92);
  const universeScore = boolScore(universe.strict_eligible_target_pass, 82, 42);
  const upliftScore = boolScore(pooled.uplift_pass, 82, 38);
  const researchScore = boolScore(research.research_validation_pass, 82, 36);
  const score = averageScore([boolScore(pooled.model_quality_pass), boolScore(pooled.decision_support_allowed), sampleScore, universeScore, upliftScore, boolScore(paper.paper_decision_support_allowed), researchScore]);
  const blockers = [];
  if (root) blockers.push(item("Root cause", `${labelValue(root.code)} · ${fmtNumber(root.count, 0)}개 실패 게이트`, "warn"));
  if (!isTruthy(pooled.model_quality_pass)) blockers.push(item("모델 품질", labelValue(pooled.model_quality_block_reasons), "warn"));
  if (!isTruthy(paper.paper_decision_support_allowed)) blockers.push(item("Paper gate", labelValue(paper.paper_gate_block_reasons), "warn"));
  return synthesis(
    "모델 운영 공장",
    "Model Ontology",
    isTruthy(pooled.decision_support_allowed) ? "모델 판단 연결 가능" : "모델 운영 보강 필요",
    `Champion/Challenger, 표본 충분성, TSM-like calibration, uplift, registry, shadow paper를 모델 공장으로 연결했습니다.`,
    isTruthy(pooled.decision_support_allowed) ? "good" : "warn",
    score,
    [
      driver("Champion", labelValue(champion.model_name || pooled.model_name), boolScore(champion.is_champion || pooled.model_quality_pass), `${labelValue(champion.model_family || pooled.model_family)} · split ${labelValue(champion.split)}`, "pooled_model_comparison"),
      driver("표본 충분성", `${fmtNumber(sample.trade_ready_20d_labeled, 0)}건`, sampleScore, `model-training ${fmtNumber(sample.model_training_20d_labeled, 0)} · target 10,000/100,000`, "pooled_sample_audit", sampleScore >= 70 ? "good" : "warn"),
      driver("유니버스", `${fmtNumber(universe.strict_eligible_symbols, 0)}개 strict`, universeScore, `loaded ${fmtNumber(universe.loaded_symbols, 0)} / 후보 ${fmtNumber(universe.candidate_symbols, 0)}`, "universe_validation", universeScore >= 70 ? "good" : "warn"),
      driver("연구 검증", statusByBool(research.research_validation_pass, "통과", "보류"), researchScore, `CPCV ${fmtNumber(research.best_cpcv_model_median_uplift_pct ?? research.best_cpcv_strategy_median_uplift_pct, 2)} · trials ${fmtNumber(research.auto_research_trial_count, 0)}`, "research_expansion", researchScore >= 70 ? "good" : "warn"),
      driver("이벤트 원장", `${fmtNumber(research.event_ledger_rows, 0)}행`, clampNumber(((toNumber(research.event_ledger_rows) || 0) / 100000) * 100, 0, 88), `near-miss ${fmtNumber(research.event_ledger_near_miss_rows, 0)} · variants ${fmtNumber(research.event_ledger_variant_count, 0)}`, "backtest_event_ledger"),
      driver("TSM-like 보정", labelValue(paper.tsm_like_calibration_route), boolScore(paper.tsm_like_route_selection_pass), `effective N ${fmtNumber(paper.tsm_like_effective_train_validation_n, 0)} · ECE ${fmtNumber(paper.tsm_like_calibration_ece, 3)}`, "tsm_like_calibration_metrics"),
      driver("Uplift", statusByBool(pooled.uplift_pass, "통과", "보류"), upliftScore, `paired p ${fmtNumber(pooled.uplift_bootstrap_p_value_paired, 3)} · CI ${fmtMaybePct(pooled.selected_minus_all_ci_lower_pct_paired, 2)}`, "pooled_uplift_bootstrap", upliftScore >= 70 ? "good" : "warn"),
      driver("Shadow Paper", statusByBool(data.quality?.shadow_paper?.failed === 0, "정상", "확인 필요"), data.quality?.shadow_paper ? boolScore(data.quality.shadow_paper.failed === 0) : 50, `${fmtNumber((data.tables.shadow_predictions || []).length, 0)}개 기록`, "shadow_predictions"),
    ],
    blockers,
    [
      isTruthy(pooled.model_quality_pass) && !isTruthy(pooled.latest_signal_pass) ? item("품질/최신 신호", "모델 품질은 통과했지만 최신 신호가 판단 후보가 아닙니다.", "warn") : null,
    ].filter(Boolean),
    [
      item("RCA", "차단 원인 우선순위 표에서 가장 반복되는 게이트부터 수정합니다.", root ? "warn" : "neutral"),
      item("Calibration", "TSM-like route가 선택되었는지와 holdout ECE가 안정적인지 확인합니다."),
      item("Shadow", "가상 기록은 live를 열지 않고 모델 운영 품질만 추적합니다."),
    ],
    [
      evidenceRef("모델", "pooled_model_comparison", "tsm_pooled_model_comparison.csv", "champion/challenger"),
      evidenceRef("모델", "model_gate_root_causes", "tsm_model_gate_root_causes.csv", "RCA"),
      evidenceRef("Paper", "shadow_predictions", "tsm_shadow_predictions.csv", "가상 기록"),
    ]
  );
}

function deriveDiagnosticsSynthesis(data) {
  const modelGate = data.snapshots.model_gate || {};
  const gateRows = data.tables.model_gate_audit || [];
  const failed = gateRows.filter((row) => !isTruthy(row.passed));
  const rootRows = data.tables.model_gate_root_causes || [];
  const root = primaryBlockReason(data);
  const score = clampNumber(88 - failed.length * 0.9 - (toNumber(modelGate.critical_failed_gate_count) || 0) * 8, 12, 88);
  return synthesis(
    "Root Cause Analysis 보드",
    "Diagnostics Ontology",
    failed.length ? "우선순위 RCA 필요" : "진단 통과",
    `raw snapshot 나열보다 root cause, failed gate, 실제 값/기준값/권장 조치를 우선순위로 보여줍니다.`,
    failed.length ? "warn" : "good",
    score,
    [
      driver("최상위 원인", root ? labelValue(root.code) : "차단 없음", root ? 38 : 82, root ? `${fmtNumber(root.count, 0)}개 게이트 · ${labelValue(root.category)}` : "PASS", "model_gate_root_causes", root ? "warn" : "good"),
      driver("실패 게이트", `${fmtNumber(failed.length, 0)}개`, clampNumber(86 - failed.length * 1.2, 10, 86), `전체 감사 ${fmtNumber(gateRows.length, 0)}행`, "model_gate_audit", failed.length ? "warn" : "good"),
      driver("모델 기준 상태", labelValue(modelGate.model_gate_status), statusScore(modelGate.model_gate_status), `critical ${fmtNumber(modelGate.critical_failed_gate_count, 0)} · failed ${fmtNumber(modelGate.failed_gate_count, 0)}`, "model_gate snapshot", statusTone(modelGate.model_gate_status)),
      driver("RCA 항목", `${fmtNumber(rootRows.length, 0)}개`, rootRows.length ? 58 : 76, "권장 조치와 실제 값 범위를 함께 표시", "root causes"),
      driver("원본 스냅샷", `${fmtNumber(Object.keys(data.snapshots.prediction || {}).length + Object.keys(data.snapshots.pooled_prediction || {}).length, 0)}필드`, 62, "예측 원본값은 상세 표에서 접힘 없이 추적", "snapshots"),
    ],
    [
      root ? item("최우선 RCA", `${labelValue(root.code)}: ${explainReason(root.code)}`, "warn") : null,
      failed.length ? item("실패 게이트", "막힌 기준의 실제 값/기준값 테이블을 먼저 확인해야 합니다.", "warn") : null,
    ].filter(Boolean),
    [
      item("표시 정책", "기준 미달 예측도 숨기지 않지만 판단 사용과 표시 사용을 분리합니다.", "neutral"),
    ],
    [
      item("1순위", "diagnosticFailedGateTable에서 value와 threshold 차이를 확인합니다.", failed.length ? "warn" : "neutral"),
      item("2순위", "modelRootCauseTable의 recommended_action을 모델·성과 페이지와 연결합니다."),
      item("3순위", "snapshot 원본은 마지막 검산용으로만 사용합니다."),
    ],
    [
      evidenceRef("진단", "model_gate_audit", "tsm_model_gate_audit.csv", "실제 값/기준값"),
      evidenceRef("진단", "model_gate_root_causes", "tsm_model_gate_root_causes.csv", "권장 조치"),
      evidenceRef("예측", "prediction snapshots", "", "원본 필드"),
    ]
  );
}

function deriveBacktestSynthesis(data) {
  const best = bestBacktestStrategy(data.tables.backtest_summary || []);
  const walk = walkForwardSummary(data.tables.validation_walk_forward || [], best.strategy_id);
  const pbo = (data.tables.pbo_report || []).find((row) => String(row.strategy_id) === String(best.strategy_id)) || {};
  const dsr = (data.tables.deflated_sharpe || []).find((row) => String(row.strategy_id) === String(best.strategy_id)) || {};
  const yearly = (data.tables.yearly_returns || []).filter((row) => String(row.strategy_id) === String(best.strategy_id));
  const positiveYears = yearly.length ? yearly.filter((row) => (toNumber(row.year_return_pct) || 0) > 0).length / yearly.length * 100 : null;
  const drawdownScore = clampNumber(92 + (toNumber(best.max_drawdown_pct) || 0) * 2.2, 10, 92);
  const returnScore = clampNumber(50 + (toNumber(best.cagr_pct) || 0) * 4, 0, 92);
  const stabilityScore = averageScore([walk.positive_rate_pct ?? 50, positiveYears ?? 50, boolScore(!isTruthy(pbo.overfit_warning)), boolScore(dsr.dsr_pass)]);
  const score = averageScore([returnScore, drawdownScore, stabilityScore, clampNumber((toNumber(best.profit_factor) || 1) * 34, 20, 90)]);
  const blockers = [];
  if (toNumber(best.trade_count) !== null && toNumber(best.trade_count) < 20) blockers.push(item("거래 수", "거래 수가 작아 전략 증거가 약합니다.", "warn"));
  if (isTruthy(pbo.overfit_warning)) blockers.push(item("PBO", "과최적화 경고가 있습니다.", "warn"));
  if (dsr.dsr_pass !== undefined && !isTruthy(dsr.dsr_pass)) blockers.push(item("DSR", "다중 실험 보정 후 Sharpe 유의성이 부족합니다.", "warn"));
  return synthesis(
    "전략 증거 보드",
    "Backtest Ontology",
    score >= 68 ? "전략 증거 양호" : "증거 보강 필요",
    `equity curve, drawdown, trade lifecycle, yearly stability, overfit flags를 전략 증거로 묶었습니다.`,
    toneFromScore(score),
    score,
    [
      driver("대표 전략", labelValue(best.strategy_id || "없음"), returnScore, `CAGR ${fmtMaybePct(best.cagr_pct, 1)} · 총수익 ${fmtMaybePct(best.total_return_pct, 1)}`, "backtest_summary", returnScore >= 70 ? "good" : "neutral"),
      driver("Drawdown", fmtMaybePct(best.max_drawdown_pct, 1), drawdownScore, `Sharpe ${fmtNumber(best.sharpe_zero_rf, 2)} · Calmar ${fmtNumber(best.calmar_ratio, 2)}`, "equity/drawdown", drawdownScore < 45 ? "warn" : "neutral"),
      driver("Trade lifecycle", `${fmtNumber(best.trade_count, 0)}회`, clampNumber((toNumber(best.trade_count) || 0) * 3, 30, 86), `승률 ${fmtMaybePct(best.win_rate_pct, 1)} · PF ${fmtNumber(best.profit_factor, 2)}`, "trade_log"),
      driver("연도 안정성", positiveYears === null ? "없음" : `${fmtNumber(positiveYears, 0)}%`, positiveYears ?? 50, `${fmtNumber(yearly.length, 0)}개 연도 · WF 양수 ${fmtMaybePct(walk.positive_rate_pct, 1)}`, "yearly_returns/walk_forward"),
      driver("과최적화 방어", isTruthy(pbo.overfit_warning) ? "경고" : "경고 없음", stabilityScore, `PBO ${fmtNumber(pbo.pbo_proxy, 3)} · DSR ${fmtNumber(dsr.deflated_sharpe_ratio, 3)}`, "pbo/dsr", stabilityScore < 50 ? "warn" : "good"),
    ],
    blockers,
    [
      returnScore >= 70 && stabilityScore < 55 ? item("성과/검증", "성과는 좋지만 검증 안정성이 아직 결론을 제한합니다.", "warn") : null,
    ].filter(Boolean),
    [
      item("자산곡선", "대표 전략의 equity curve와 drawdown을 먼저 확인합니다."),
      item("거래 기록", "최근 거래 기록에서 진입/청산 사이클과 손절 이유를 봅니다."),
      item("검증 연결", "검증 페이지에서 PBO/DSR와 walk-forward를 함께 확인합니다.", blockers.length ? "warn" : "neutral"),
    ],
    [
      evidenceRef("백테스트", "backtest_summary", "tsm_backtest_summary.csv", "전략별 성과"),
      evidenceRef("백테스트", "trade_log", "tsm_backtest_trade_log.csv", "거래 생애주기"),
      evidenceRef("검증", "pbo_report/deflated_sharpe", "", "과최적화 방어"),
    ]
  );
}

function deriveRiskSynthesis(data) {
  const risk = data.snapshots.risk || {};
  const stress = data.snapshots.stress || {};
  const maxWeight = snapshotWeightValue(risk, ["final_recommended_max_weight", "final_recommended_max_weight_pct"]);
  const maxWeightScore = maxWeight === null ? 40 : clampNumber(maxWeight * 100 * 16, 18, 90);
  const stopScore = clampNumber(90 - Math.abs(toNumber(risk.risk_pct_2atr) || 0) * 3.8, 18, 90);
  const impact = toNumber(stress.worst_current_weight_portfolio_impact_pct);
  const stressScore = impact === null ? 56 : clampNumber(88 + impact * 35, 12, 90);
  const strategyStress = worstByAbs(data.tables.strategy_stress || [], "max_drawdown_pct");
  const strategyScore = clampNumber(92 + (toNumber(strategyStress.max_drawdown_pct) || 0) * 2.2, 12, 92);
  const score = averageScore([maxWeightScore, stopScore, stressScore, strategyScore, statusScore(risk.risk_state)]);
  const blockers = [];
  if (maxWeight !== null && maxWeight < 0.03) blockers.push(item("비중", `권장 최대 비중이 ${fmtMaybePct(maxWeight, 2)}로 낮습니다.`, "warn"));
  if (toNumber(risk.risk_pct_2atr) !== null && Math.abs(toNumber(risk.risk_pct_2atr)) > 8) blockers.push(item("손절 거리", "2ATR 손절폭이 커서 계좌 위험 기준이 비중을 제한합니다.", "warn"));
  if (stressScore < 45) blockers.push(item("스트레스", "현재 비중 기준 하락 시나리오의 계좌 영향이 큽니다.", "warn"));
  return synthesis(
    "리스크 통제 타워",
    "Risk Ontology",
    score >= 65 ? "위험 한도 안" : "비중 제한 우선",
    `max weight decomposition, 2ATR stop, stress scenarios, account impact, strategy stress를 리스크 통제 타워로 합성했습니다.`,
    score >= 65 ? "good" : "warn",
    score,
    [
      driver("권장 최대 비중", fmtMaybePct(maxWeight, 2), maxWeightScore, `${labelValue(risk.limiting_reason)} · ${labelValue(risk.risk_state)}`, "risk snapshot", maxWeightScore < 45 ? "warn" : "good"),
      driver("2ATR 손절", fmtCurrency(risk.stop_price_2atr), stopScore, `손절까지 ${fmtMaybePct(risk.risk_pct_2atr, 2)} · 주당 위험 ${fmtCurrency(risk.risk_per_share_2atr)}`, "risk snapshot"),
      driver("하락 시나리오", fmtMaybePct(impact, 2), stressScore, `${labelValue(stress.worst_current_weight_scenario)} · 가정가 ${fmtCurrency(stress.worst_current_weight_implied_price)}`, "stress snapshot", stressScore < 45 ? "warn" : "neutral"),
      driver("전략 스트레스", labelValue(strategyStress.strategy_id || stress.worst_strategy_by_mdd), strategyScore, `MDD ${fmtMaybePct(strategyStress.max_drawdown_pct ?? stress.worst_strategy_mdd_pct, 2)} · 최신 낙폭 ${fmtMaybePct(strategyStress.latest_drawdown_pct, 2)}`, "strategy_stress"),
      driver("분해 한도", labelValue(risk.limiting_reason), maxWeightScore, `변동성 ${fmtMaybePct(risk.vol_limit_weight, 2)} · 추세 ${fmtMaybePct(risk.trend_limit_weight, 2)} · 점수 ${fmtMaybePct(risk.score_limit_weight, 2)}`, "risk series"),
    ],
    blockers,
    [
      hasEntryTrigger(data.snapshots.decision || {}) && maxWeightScore < 50 ? item("신호/비중", "매수 신호가 생겨도 비중 한도가 먼저 실행을 막습니다.", "warn") : null,
    ].filter(Boolean),
    [
      item("비중", "최종 권장 비중이 어떤 한도에서 막혔는지 riskWeightChart에서 확인합니다."),
      item("손절", "2ATR 손절가와 목표가의 R 배수를 매매 판단과 함께 봅니다."),
      item("스트레스", "하락 시나리오의 계좌 영향을 기준으로 신규 위험을 제한합니다.", blockers.length ? "warn" : "neutral"),
    ],
    [
      evidenceRef("리스크", "risk series", "tsm_risk_policy.csv", "비중 분해"),
      evidenceRef("리스크", "stress_scenarios", "tsm_stress_scenarios.csv", "계좌 영향"),
      evidenceRef("리스크", "strategy_stress", "tsm_strategy_stress.csv", "전략별 스트레스"),
    ]
  );
}

function deriveQualitySynthesis(data) {
  const quality = data.quality || {};
  const totals = Object.values(quality).filter(Boolean).reduce((acc, section) => {
    acc.total += toNumber(section.total) || 0;
    acc.failed += toNumber(section.failed) || 0;
    acc.passed += toNumber(section.passed) || 0;
    return acc;
  }, { total: 0, failed: 0, passed: 0 });
  const manifestRate = passRate(data.tables.manifest || [], "status");
  const best = bestBacktestStrategy(data.tables.backtest_summary || []);
  const walk = walkForwardSummary(data.tables.validation_walk_forward || [], best.strategy_id);
  const causalWalk = walkForwardSummary(data.tables.validation_causal_walk_forward || [], best.strategy_id);
  const pboWorst = worstByAbs(data.tables.pbo_report || [], "pbo_proxy");
  const dsrPass = passRate(data.tables.deflated_sharpe || [], "dsr_pass");
  const qualityScore = totals.total ? ((totals.total - totals.failed) / totals.total) * 100 : 60;
  const score = averageScore([qualityScore, manifestRate ?? 60, walk.positive_rate_pct ?? 50, causalWalk.positive_rate_pct ?? 50, dsrPass ?? 50, boolScore(!isTruthy(pboWorst.overfit_warning))]);
  const blockers = [];
  if (totals.failed) blockers.push(item("품질 점검", `${fmtNumber(totals.failed, 0)}개 점검 실패`, "warn"));
  if (manifestRate !== null && manifestRate < 100) blockers.push(item("실행 기록", `manifest 통과율 ${fmtNumber(manifestRate, 0)}%`, "warn"));
  if (isTruthy(pboWorst.overfit_warning)) blockers.push(item("과최적화", "PBO 과최적화 경고가 있습니다.", "warn"));
  return synthesis(
    "신뢰도/과최적화 방어 보드",
    "Validation Ontology",
    score >= 72 ? "검증 방어 양호" : "검증 보강 필요",
    `walk-forward, causal walk-forward, sensitivity, PBO, DSR, segment validation을 신뢰도와 과최적화 방어로 합성했습니다.`,
    toneFromScore(score),
    score,
    [
      driver("품질 점검", `${fmtNumber(totals.passed, 0)} / ${fmtNumber(totals.total, 0)}`, qualityScore, `실패 ${fmtNumber(totals.failed, 0)}개`, "quality object", totals.failed ? "warn" : "good"),
      driver("실행 Manifest", manifestRate === null ? "없음" : `${fmtNumber(manifestRate, 0)}%`, manifestRate ?? 50, `${fmtNumber((data.tables.manifest || []).length, 0)}개 단계`, "manifest", manifestRate === 100 ? "good" : "warn"),
      driver("Walk-forward", fmtMaybePct(walk.positive_rate_pct, 1), walk.positive_rate_pct ?? 50, `중앙 CAGR ${fmtMaybePct(walk.median_cagr_pct, 1)}`, "validation_walk_forward"),
      driver("Causal WF", fmtMaybePct(causalWalk.positive_rate_pct, 1), causalWalk.positive_rate_pct ?? 50, `중앙 CAGR ${fmtMaybePct(causalWalk.median_cagr_pct, 1)}`, "validation_causal_walk_forward"),
      driver("PBO", fmtNumber(pboWorst.pbo_proxy, 3), boolScore(!isTruthy(pboWorst.overfit_warning), 82, 35), `best source ${labelValue(pboWorst.best_trial_source)}`, "pbo_report", isTruthy(pboWorst.overfit_warning) ? "warn" : "good"),
      driver("DSR", dsrPass === null ? "없음" : `${fmtNumber(dsrPass, 0)}%`, dsrPass ?? 50, "다중 신호 실험 수를 감안한 Sharpe 검증", "deflated_sharpe"),
    ],
    blockers,
    [
      qualityScore >= 80 && (walk.positive_rate_pct || 0) < 50 ? item("품질/성과", "파일 품질은 괜찮지만 walk-forward 성과 안정성이 약합니다.", "warn") : null,
    ].filter(Boolean),
    [
      item("WF", "시간 순서 검증 차트에서 테스트 구간별 양수/음수 분포를 확인합니다."),
      item("PBO/DSR", "과최적화 방어 표에서 경고가 있는 전략을 제외합니다.", blockers.length ? "warn" : "neutral"),
      item("Sensitivity", "손절폭과 점수 기준 변경에도 성과가 유지되는지 봅니다."),
    ],
    [
      evidenceRef("검증", "validation_walk_forward", "tsm_validation_walk_forward.csv", "시간 순서"),
      evidenceRef("검증", "pbo_report", "tsm_pbo_report.csv", "과최적화"),
      evidenceRef("검증", "deflated_sharpe", "tsm_deflated_sharpe.csv", "다중 실험 보정"),
    ]
  );
}

function deriveSystemSynthesis(data) {
  const system = data.snapshots.system || {};
  const health = data.snapshots.daily_health || {};
  const dq = data.snapshots.data_quality || {};
  const modelGate = data.snapshots.model_gate || {};
  const readinessRows = data.tables.readiness_scorecard || [];
  const blocks = (data.tables.system_block_reasons || []).filter((row) => String(row.block_reasons || "").toUpperCase() !== "PASS");
  const readiness = toNumber(system.system_readiness_score) ?? avgNumber(readinessRows, "score") ?? 50;
  const blockPenalty = blocks.length * 6 + (toNumber(dq.failed_checks) || 0) * 4 + (toNumber(modelGate.failed_gate_count) || 0) * 0.6;
  const score = clampNumber(readiness - blockPenalty, 0, 100);
  const blockers = blocks.slice(0, 4).map((row) => item(labelValue(row.component), labelValue(row.block_reasons), "warn"));
  if (String(system.live_trading_status || "").toUpperCase().includes("DISABLED")) blockers.push(item("Live", "실거래는 설계상 비활성입니다.", "neutral"));
  return synthesis(
    "운영 관제 보드",
    "System Ontology",
    score >= 70 ? "운영 준비 양호" : "운영 병목 확인",
    `readiness, block reasons, data quality, manifest, health를 운영 관제 보드로 합성했습니다.`,
    score >= 70 ? "good" : "warn",
    score,
    [
      driver("시스템 준비도", `${fmtNumber(system.system_readiness_score, 0)} / 100`, readiness, `${labelValue(system.system_state)} · composite ${fmtNumber(system.composite_gate_score, 0)}`, "system snapshot", readiness >= 70 ? "good" : "warn"),
      driver("일일 헬스", labelValue(health.daily_health_status), statusScore(health.daily_health_status), labelValue(health.failed_or_blocked_components || "PASS"), "daily_health", statusTone(health.daily_health_status)),
      driver("데이터 품질", labelValue(dq.data_quality_status), statusScore(dq.data_quality_status), `failed ${fmtNumber(dq.failed_checks, 0)} · critical ${fmtNumber(dq.critical_failed_checks, 0)}`, "data_quality", statusTone(dq.data_quality_status)),
      driver("모델 게이트", labelValue(modelGate.model_gate_status), statusScore(modelGate.model_gate_status), `failed ${fmtNumber(modelGate.failed_gate_count, 0)} · next ${labelValue(modelGate.next_required_evidence_action)}`, "model_gate", statusTone(modelGate.model_gate_status)),
      driver("Paper/Live", labelValue(system.paper_trading_status), boolScore(system.paper_ready), `live ${labelValue(system.live_trading_status || "DISABLED_BY_DESIGN")}`, "system state"),
      driver("실행 Manifest", `${fmtNumber((data.tables.manifest || []).length, 0)}단계`, passRate(data.tables.manifest || [], "status") ?? 60, "마지막 실행 단계별 성공/실패", "manifest"),
    ],
    blockers,
    [
      isTruthy(system.research_ready) && !isTruthy(system.alpha_ready) ? item("연구/alpha", "분석 준비는 되었지만 alpha 기준은 아직 준비되지 않았습니다.", "warn") : null,
    ].filter(Boolean),
    [
      item("상태 점검", "systemBlockTable에서 component별 block reason을 먼저 봅니다.", blocks.length ? "warn" : "neutral"),
      item("일일 업데이트", "manifest와 daily_health가 PASS인지 확인합니다."),
      item("안전 원칙", "live trading은 설계상 비활성으로 유지하고 Paper/연구 산출물만 갱신합니다."),
    ],
    [
      evidenceRef("시스템", "readiness_scorecard", "tsm_system_readiness_scorecard.csv", "준비도"),
      evidenceRef("시스템", "system_block_reasons", "tsm_system_block_reasons.csv", "차단"),
      evidenceRef("운영", "manifest", "tsm_daily_update_manifest.csv", "실행 이력"),
    ]
  );
}

function deriveOutputsSynthesis(data) {
  const files = data.files || [];
  const byKind = files.reduce((acc, file) => {
    acc[file.kind] = (acc[file.kind] || 0) + 1;
    return acc;
  }, {});
  const categories = [
    ["가격", ["price", "daily", "chart"]],
    ["뉴스", ["news", "cause", "cluster"]],
    ["예측", ["prediction", "pooled", "model"]],
    ["리스크", ["risk", "stress"]],
    ["검증", ["validation", "pbo", "sharpe", "quality"]],
    ["운영", ["manifest", "health", "system"]],
  ].map(([label, tokens]) => {
    const count = files.filter((file) => tokens.some((token) => String(file.path).toLowerCase().includes(token))).length;
    return { label, count };
  });
  const active = files.find((file) => file.path === state.activeFile) || files[0] || {};
  const score = files.length ? 76 : 30;
  return synthesis(
    "객체별 증거 라이브러리",
    "Evidence Ontology",
    files.length ? "증거 추적 가능" : "산출물 없음",
    `파일 목록을 가격, 뉴스, 예측, 모델, 리스크, 검증, 운영 객체별 증거 라이브러리로 재분류했습니다.`,
    files.length ? "good" : "warn",
    score,
    [
      driver("전체 산출물", `${fmtNumber(files.length, 0)}개`, score, `CSV ${fmtNumber(byKind.csv, 0)} · 리포트 ${fmtNumber(byKind.report, 0)} · 이미지 ${fmtNumber(byKind.image, 0)}`, "files"),
      ...categories.slice(0, 5).map((cat) => driver(cat.label, `${fmtNumber(cat.count, 0)}개`, clampNumber(Math.log10(cat.count + 1) * 34, 30, 88), `${cat.label} 객체와 직접 연결된 산출물`, "file paths")),
    ],
    [
      !files.length ? item("파일", "필터 조건 또는 산출물 생성 상태를 확인해야 합니다.", "warn") : null,
    ].filter(Boolean),
    [
      active.path ? item("미리보기", `${fileDisplayName(active)}를 현재 증거로 열었습니다.`, "neutral") : null,
    ].filter(Boolean),
    [
      item("필터", "객체별로 검색어와 kind/scope 필터를 조합해 원본 증거를 좁힙니다."),
      item("미리보기", "CSV는 표, 리포트는 문서, 이미지는 차트 증거로 바로 확인합니다."),
      item("추적", "각 페이지 합성 보드의 원본 증거 이름을 자료실에서 검색합니다."),
    ],
    categories.slice(0, 5).map((cat) => evidenceRef(cat.label, `${cat.count} files`, "", `${cat.label} 관련 산출물`))
  );
}

function deriveRunSynthesis(data, run = {}) {
  const form = $("runForm");
  const modeSelect = $("runMode");
  const modeLabel = modeSelect?.selectedOptions?.[0]?.textContent || run.mode_label || run.mode || "대기";
  const mode = modeSelect?.value || run.mode || "downstream";
  const status = run.status || state.lastRunStatus || "IDLE";
  const strict = Boolean(form?.elements?.schema_strict?.checked);
  const continueOnError = Boolean(form?.elements?.continue_on_error?.checked);
  const skipCharts = Boolean(form?.elements?.skip_charts?.checked);
  const skipNews = Boolean(form?.elements?.skip_news_refresh?.checked);
  const dependencyScore = mode === "full" || mode === "downstream" ? 82 : 64;
  const safetyScore = continueOnError ? 42 : strict ? 84 : 68;
  const statusScoreValue = status === "PASS" ? 84 : status === "RUNNING" ? 58 : status === "FAIL" ? 24 : 62;
  const score = averageScore([dependencyScore, safetyScore, statusScoreValue, statusScore(data.snapshots.system?.system_state)]);
  const expected = runModeOutputs(mode);
  return synthesis(
    "실행 의존성 그래프",
    "Run Ontology",
    status === "RUNNING" ? "실행 중" : status === "FAIL" ? "실행 실패 확인" : "안전 실행 대기",
    `실행 모드, 의존 엔진, 예상 산출물, 안전 상태, 마지막 실행 결과를 연결했습니다. 실제 브로커 주문은 포함하지 않습니다.`,
    status === "FAIL" ? "bad" : status === "RUNNING" ? "neutral" : "good",
    score,
    [
      driver("선택 작업", modeLabel, dependencyScore, `${expected.dependencies.join(" → ")}`, "run form"),
      driver("예상 산출물", `${fmtNumber(expected.outputs.length, 0)}개`, 72, expected.outputs.join(" · "), "run mode map"),
      driver("안전 상태", continueOnError ? "오류 후 계속" : "중단 우선", safetyScore, `${strict ? "schema strict" : "schema 일반"} · ${skipCharts ? "차트 생략" : "차트 생성"} · ${skipNews ? "뉴스 갱신 생략" : "뉴스 포함"}`, "run options", continueOnError ? "warn" : "good"),
      driver("마지막 실행", labelValue(status), statusScoreValue, `${run.duration_sec ? `${run.duration_sec}초` : "시간 없음"} · ${run.mode_label || run.mode || "대기"}`, "run status", status === "FAIL" ? "bad" : status === "PASS" ? "good" : "neutral"),
      driver("거래 안전장치", "실거래 제외", 88, "Run 페이지는 연구/가상 기록/데이터 갱신만 수행합니다.", "system design", "good"),
    ],
    [
      status === "RUNNING" ? item("실행 중", "현재 작업이 끝날 때까지 새 실행을 피합니다.", "warn") : null,
      continueOnError ? item("안전 옵션", "오류 후 계속 옵션은 실패 원인을 숨길 수 있습니다.", "warn") : null,
    ].filter(Boolean),
    [
      skipNews && mode === "full" ? item("전체 업데이트/뉴스", "전체 업데이트에서 뉴스 새 수집을 생략하면 원인 그래프가 오래된 상태일 수 있습니다.", "warn") : null,
    ].filter(Boolean),
    [
      item("실행 전", "의존성 그래프의 시작일/종료일과 데이터 소스를 확인합니다."),
      item("실행 후", "manifest, data_quality, system 상태가 PASS로 돌아왔는지 확인합니다."),
      item("안전", "live broker 주문은 이 UI에서 실행하지 않습니다.", "good"),
    ],
    [
      evidenceRef("실행", "manifest", "tsm_daily_update_manifest.csv", "단계별 결과"),
      evidenceRef("시스템", "daily_health", "tsm_daily_health_report.md", "일일 점검"),
      evidenceRef("데이터", "data_quality_checks", "tsm_data_quality_checks.csv", "데이터 검증"),
    ]
  );
}

function runModeOutputs(mode) {
  const map = {
    full: { dependencies: ["데이터 수집", "뉴스", "룰", "모델", "검증", "시스템"], outputs: ["가격", "뉴스", "예측", "리스크", "검증", "헬스"] },
    downstream: { dependencies: ["기존 데이터", "룰", "백테스트", "리스크", "예측", "시스템"], outputs: ["매매 계획", "예측", "리스크", "시스템"] },
    hourly: { dependencies: ["시간봉 provider", "검증", "차트"], outputs: ["시간봉", "라인리지"] },
    intraday: { dependencies: ["분봉 provider", "검증", "차트"], outputs: ["분봉", "라인리지"] },
    news_causal_engine: { dependencies: ["뉴스 수집", "클러스터", "가격 매칭", "피처"], outputs: ["뉴스 원인", "뉴스 피처"] },
    prediction_engine: { dependencies: ["피처 계약", "워크포워드", "확률 보정"], outputs: ["예측", "calibration"] },
    pooled_model_engine: { dependencies: ["유니버스", "학습", "게이트", "보정"], outputs: ["pooled 예측", "모델 비교"] },
    system_state_engine: { dependencies: ["품질", "모델 게이트", "리스크", "Paper"], outputs: ["readiness", "block reasons"] },
  };
  return map[mode] || { dependencies: ["선택 엔진", "입력 검증", "산출물 갱신"], outputs: ["선택 산출물", "manifest"] };
}

function renderOntologyCommand(id, data) {
  const container = $(id);
  if (!container) return;
  const system = data.snapshots.system || {};
  const health = data.snapshots.daily_health || {};
  const dataQuality = data.snapshots.data_quality || {};
  const modelGate = data.snapshots.model_gate || {};
  const paperGate = data.snapshots.paper_gate || {};
  const decision = data.snapshots.decision || {};
  const risk = data.snapshots.risk || {};
  const prediction = data.snapshots.prediction || {};
  const pooled = data.snapshots.pooled_prediction || {};
  const latestNews = data.snapshots.latest_news || {};
  const decisionState = overviewDecisionState(decision, prediction, system);
  const maxWeight = snapshotWeightValue(risk, ["final_recommended_max_weight", "final_recommended_max_weight_pct"]);
  const pSuccess = snapshotValue(pooled, ["p_success_20d", "p_success_tsm_like_20d"]) || snapshotValue(prediction, ["trade_ready_p_success_20d", "trigger_p_success_20d", "context_p_success_20d"]);
  const stopRisk = snapshotValue(pooled, ["p_stop_hit_20d"]) || snapshotValue(prediction, ["trade_ready_p_stop_hit_20d", "trigger_p_stop_hit_20d", "context_p_stop_hit_20d"]);
  const asOf = shortDate(decision.date || prediction.prediction_asof_date || pooled.asof_date);
  const cards = [
    {
      label: "가격 신호",
      value: `${fmtCurrency(decision.close)} · ${fmtNumber(decision.score_price_algo_total, 1)}점`,
      note: labelValue(decision.entry_trigger || "NO_ENTRY_TRIGGER"),
      tone: scoreTone(decision.score_price_algo_total, decision.entry_trigger),
    },
    {
      label: "허용 비중",
      value: fmtMaybePct(maxWeight, 2),
      note: labelValue(risk.limiting_reason || risk.risk_state),
      tone: gateTone(riskStatusFromSnapshot(data)),
    },
    {
      label: "20D 예측",
      value: fmtMaybePct(pSuccess, 1),
      note: `손절 ${fmtMaybePct(stopRisk, 1)} · ${statusByBool(system.prediction_decision_support, "판단 가능", "참고용")}`,
      tone: isTruthy(system.prediction_decision_support) ? "good" : "warn",
    },
    {
      label: "운영 상태",
      value: labelValue(health.daily_health_status || system.system_state),
      note: `${labelValue(dataQuality.data_quality_status)} · ${labelValue(modelGate.model_gate_status || paperGate.paper_gate_status)}`,
      tone: statusTone(health.daily_health_status || system.system_state),
    },
  ];

  container.innerHTML = `
    <section class="ontology-command-primary ${decisionState.tone}">
      <span>온톨로지 메인 판단 · ${escapeHtml(asOf)}</span>
      <strong>${escapeHtml(decisionState.title)}</strong>
      <p>${escapeHtml(`${decisionState.action} · ${decisionState.trigger} · ${decisionState.note}`)}</p>
      <div class="command-kpis">
        <div><span>시스템 준비도</span><strong>${escapeHtml(fmtNumber(system.system_readiness_score, 0))}</strong></div>
        <div><span>Paper 상태</span><strong>${escapeHtml(labelValue(system.paper_trading_status || paperGate.paper_gate_status))}</strong></div>
        <div><span>뉴스 원인</span><strong>${escapeHtml(labelValue(latestNews.news_primary_cause_type || "NO_HIGH_CONFIDENCE_NEWS"))}</strong></div>
      </div>
    </section>
    <section class="ontology-command-metrics">
      ${cards.map(ontologyMetricHtml).join("")}
    </section>
  `;
}

function ontologyMetricHtml(card) {
  return `
    <article class="ontology-metric ${card.tone || "neutral"}">
      <span>${escapeHtml(card.label)}</span>
      <strong>${escapeHtml(card.value)}</strong>
      <p title="${escapeHtml(card.note)}">${escapeHtml(compactText(card.note, 96))}</p>
    </article>
  `;
}

function renderOntologyMap(id, data) {
  const container = $(id);
  if (!container) return;
  const synthesis = deriveOntologySynthesis(data);

  container.innerHTML = `
    <section class="ontology-map-panel synthesis-panel ${synthesis.tone}">
      <div class="ontology-map-head synthesis-head">
        <div>
          <h2>판단 합성 보드</h2>
          <p>가격, 뉴스, 예측, 검증, 리스크를 섞어 오늘 새로 읽어야 할 의미를 계산</p>
        </div>
        <strong>${escapeHtml(synthesis.asOf)} · 합성 준비도 ${escapeHtml(fmtNumber(synthesis.readiness, 0))}/100</strong>
      </div>
      <div class="synthesis-board">
        <article class="synthesis-verdict ${synthesis.tone}" style="--score:${synthesis.readiness}; --ring-color:${synthesis.color}">
          <div>
            <span>데이터를 합성한 결론</span>
            <strong>${escapeHtml(synthesis.verdict)}</strong>
            <p>${escapeHtml(synthesis.meaning)}</p>
          </div>
          <div class="synthesis-ring"><b>${escapeHtml(fmtNumber(synthesis.readiness, 0))}</b><span>합성</span></div>
          <div class="synthesis-kpis">
            <div><span>기회 압력</span><strong>${escapeHtml(fmtNumber(synthesis.opportunityPressure, 0))}</strong></div>
            <div><span>차단 압력</span><strong>${escapeHtml(fmtNumber(synthesis.blockPressure, 0))}</strong></div>
            <div><span>다음 판단</span><strong>${escapeHtml(synthesis.nextAction)}</strong></div>
          </div>
        </article>
        <section class="fusion-panel">
          <div class="fusion-axis">
            <span>차단</span>
            <div><i style="left:${synthesis.readiness}%"></i></div>
            <span>실행</span>
          </div>
          <div class="fusion-factor-grid">
            ${synthesis.factors.map(synthesisFactorHtml).join("")}
          </div>
        </section>
        <section class="synthesis-insights">
          ${synthesis.insights.map(synthesisInsightHtml).join("")}
        </section>
        <section class="synthesis-routes">
          ${synthesis.routes.map(synthesisRouteHtml).join("")}
        </section>
      </div>
    </section>
  `;
}

function deriveOntologySynthesis(data) {
  const system = data.snapshots.system || {};
  const dataQuality = data.snapshots.data_quality || {};
  const modelGate = data.snapshots.model_gate || {};
  const paperGate = data.snapshots.paper_gate || {};
  const decision = data.snapshots.decision || {};
  const risk = data.snapshots.risk || {};
  const prediction = data.snapshots.prediction || {};
  const pooled = data.snapshots.pooled_prediction || {};
  const latestNews = data.snapshots.latest_news || {};
  const sample = data.snapshots.pooled_sample_audit || {};
  const planRows = data.tables.trading_plan || [];
  const state = overviewDecisionState(decision, prediction, system);
  const score = toNumber(decision.score_price_algo_total);
  const triggerText = String(decision.entry_trigger || "").toUpperCase();
  const hasTrigger = triggerText && !["NONE", "NO_ENTRY_TRIGGER", "NO_SIGNAL"].includes(triggerText);
  const maxWeight = snapshotWeightValue(risk, ["final_recommended_max_weight", "final_recommended_max_weight_pct"]);
  const maxWeightPct = maxWeight === null ? null : maxWeight * 100;
  const pSuccess = probabilityPercent(snapshotValue(pooled, ["p_success_20d", "p_success_tsm_like_20d"]) || snapshotValue(prediction, ["trade_ready_p_success_20d", "trigger_p_success_20d", "context_p_success_20d"]));
  const pStop = probabilityPercent(snapshotValue(pooled, ["p_stop_hit_20d"]) || snapshotValue(prediction, ["trade_ready_p_stop_hit_20d", "trigger_p_stop_hit_20d", "context_p_stop_hit_20d"]));
  const threshold = probabilityPercent(snapshotValue(pooled, ["threshold_20d", "threshold"]) || snapshotValue(prediction, ["trade_ready_threshold_20d", "trigger_threshold_20d", "context_threshold_20d"]));
  const expectedR = toNumber(snapshotValue(pooled, ["expected_r_net_20d"]) || snapshotValue(prediction, ["trade_ready_expected_r_20d", "trigger_expected_r_20d", "context_expected_r_20d"]));
  const eventCount = toNumber(pooled.decision_event_count || sample.trade_ready_20d_labeled) || 0;
  const selectedCount = toNumber(pooled.selected_oos_event_count) || 0;
  const failedChecks = toNumber(dataQuality.failed_checks) || 0;
  const newsPenalty = Math.abs(toNumber(latestNews.news_penalty_event) || 0);
  const newsEvents = toNumber(latestNews.news_event_count_3d || latestNews.news_source_count) || 0;
  const newsConfidence = String(latestNews.news_match_confidence || "").toUpperCase();
  const dataTone = statusTone(dataQuality.data_quality_status);
  const riskTone = gateTone(riskStatusFromSnapshot(data));
  const systemReadyRaw = toNumber(system.system_readiness_score);
  const predictionAllowed = isTruthy(system.prediction_decision_support) || isTruthy(pooled.decision_support_allowed);

  const priceForce = clampNumber((score ?? 0) + (hasTrigger ? 12 : 0), 0, 100);
  let riskCapacity = maxWeightPct === null ? 34 : clampNumber((maxWeightPct / 6) * 100, 0, 100);
  if (riskTone === "good") riskCapacity = Math.max(riskCapacity, 62);
  if (riskTone === "warn") riskCapacity = Math.min(riskCapacity, 52);
  if (riskTone === "bad") riskCapacity = Math.min(riskCapacity, 28);

  let predictionEdge = predictionAllowed ? 68 : 36;
  if (pSuccess !== null && threshold !== null) predictionEdge = clampNumber(50 + (pSuccess - threshold) * 2.2 + (expectedR || 0) * 8, 0, 100);
  else if (pSuccess !== null) predictionEdge = clampNumber(pSuccess + (expectedR || 0) * 8, 0, 100);
  if (!predictionAllowed) predictionEdge = Math.min(predictionEdge, 55);

  const eventScore = clampNumber(Math.log10(eventCount + 1) * 28, 0, 100);
  const selectedScore = clampNumber(Math.log10(selectedCount + 1) * 34, 0, 100);
  let validationTrust = eventCount ? eventScore * 0.65 + selectedScore * 0.35 : 38;
  if (isTruthy(paperGate.tsm_like_route_selection_pass)) validationTrust = Math.max(validationTrust, 72);
  if (!predictionAllowed) validationTrust = Math.min(validationTrust, 58);

  const dataIntegrity = clampNumber((dataTone === "good" ? 92 : dataTone === "warn" ? 62 : dataTone === "bad" ? 28 : 50) - failedChecks * 8, 0, 100);
  const newsDrag = clampNumber(newsPenalty * 18 + Math.min(newsEvents * 3, 18) + (newsConfidence === "HIGH" ? 14 : newsConfidence === "MEDIUM" ? 7 : 0), 0, 100);
  const newsClarity = clampNumber(100 - newsDrag, 0, 100);
  const systemReadiness = clampNumber(systemReadyRaw ?? (statusTone(system.system_state) === "good" ? 82 : 50), 0, 100);

  const factors = [
    {
      key: "price",
      label: "가격·룰",
      score: priceForce,
      value: `${fmtNumber(score, 1)}점`,
      raw: `${labelValue(decision.entry_trigger || "NO_ENTRY_TRIGGER")} · ${labelValue(decision.trend_regime)}`,
      meaning: hasTrigger ? "행동 신호가 가격에서 발생" : "점수는 누적됐지만 행동 트리거는 아직 없음",
      color: colors.blue,
    },
    {
      key: "prediction",
      label: "예측 우위",
      score: predictionEdge,
      value: pSuccess === null ? "없음" : `${fmtNumber(pSuccess, 1)}%`,
      raw: `기준 ${threshold === null ? "없음" : `${fmtNumber(threshold, 1)}%`} · 손절 ${pStop === null ? "없음" : `${fmtNumber(pStop, 1)}%`}`,
      meaning: predictionAllowed ? "모델을 판단 근거로 연결 가능" : "모델은 아직 표시·참고용",
      color: colors.violet,
    },
    {
      key: "risk",
      label: "리스크 여유",
      score: riskCapacity,
      value: fmtMaybePct(maxWeight, 2),
      raw: `${labelValue(risk.risk_state)} · 손절 ${fmtCurrency(risk.stop_price_2atr)}`,
      meaning: riskCapacity >= 60 ? "신호가 와도 비중 산정 가능" : "신호가 와도 실행 크기 제한",
      color: colors.green,
    },
    {
      key: "validation",
      label: "검증 표본",
      score: validationTrust,
      value: `${fmtNumber(eventCount, 0)}건`,
      raw: `선택 ${fmtNumber(selectedCount, 0)}건 · ${labelValue(modelGate.model_gate_status || pooled.model_support_route)}`,
      meaning: predictionAllowed ? "표본과 게이트가 판단을 지지" : "표본은 있으나 게이트가 판단 연결을 제한",
      color: colors.teal,
    },
    {
      key: "data",
      label: "데이터 신뢰",
      score: dataIntegrity,
      value: labelValue(dataQuality.data_quality_status || "MISSING"),
      raw: `${shortDate(dataQuality.latest_signal_date)} · 실패 ${fmtNumber(failedChecks, 0)}개`,
      meaning: dataIntegrity >= 70 ? "오늘 판단의 입력값은 사용 가능" : "먼저 데이터 최신성과 실패 항목 확인 필요",
      color: colors.ink,
    },
    {
      key: "news",
      label: "뉴스 부담",
      score: newsClarity,
      value: labelValue(latestNews.news_primary_cause_type || "NO_HIGH_CONFIDENCE_NEWS"),
      raw: `${labelValue(latestNews.news_match_confidence || "NO_MATCH")} · 감점 ${fmtNumber(latestNews.news_penalty_event, 1)}`,
      meaning: newsDrag >= 35 ? "뉴스가 점수와 예측 해석을 흔드는 중" : "뉴스는 핵심 차단 원인이 아님",
      color: colors.amber,
    },
    {
      key: "system",
      label: "운영 준비",
      score: systemReadiness,
      value: fmtNumber(systemReadiness, 0),
      raw: `${labelValue(system.paper_trading_status || paperGate.paper_gate_status)} · live ${labelValue(system.live_trading_status)}`,
      meaning: "자동 업데이트와 Paper 기록 상태를 함께 반영",
      color: colors.gray,
    },
  ].map((factor) => ({ ...factor, tone: synthesisScoreTone(factor.score), influence: Math.round(factor.score - 50) }));

  const readiness = Math.round(clampNumber(
    priceForce * 0.22 +
      riskCapacity * 0.17 +
      predictionEdge * 0.18 +
      validationTrust * 0.16 +
      dataIntegrity * 0.12 +
      newsClarity * 0.07 +
      systemReadiness * 0.08,
    0,
    100
  ));
  const block = primaryBlockReason(data);
  const blockPressure = Math.round(clampNumber(
    (100 - riskCapacity) * 0.24 +
      (100 - predictionEdge) * 0.24 +
      (100 - validationTrust) * 0.17 +
      (100 - dataIntegrity) * 0.16 +
      newsDrag * 0.12 +
      (block ? 9 : 0),
    0,
    100
  ));
  const opportunityPressure = Math.round(clampNumber((priceForce + predictionEdge + riskCapacity + validationTrust) / 4, 0, 100));
  const driverFactors = factors.filter((factor) => ["price", "prediction", "risk", "validation"].includes(factor.key));
  const constraintFactors = factors.filter((factor) => factor.key !== "system");
  const strongest = driverFactors.slice().sort((a, b) => b.score - a.score)[0] || factors[0];
  const weakest = constraintFactors.slice().sort((a, b) => a.score - b.score)[0] || factors[0];
  const tone = readiness >= 72 && blockPressure < 45 ? "good" : readiness < 45 || blockPressure > 62 ? "warn" : state.tone || "neutral";
  const verdict = buildSynthesisVerdict({ state, readiness, blockPressure, dataIntegrity, hasTrigger, predictionAllowed, riskCapacity });
  const nextAction = buildSynthesisNextAction({ hasTrigger, predictionAllowed, riskCapacity, dataIntegrity, block, planRows });
  const meaning = `${koreanSubject(strongest.label)} 가장 강한 매매 근거입니다(${strongest.value}). ${koreanTopic(weakest.label)} ${weakest.raw} 상태라 결론을 제한합니다. 그래서 현재 판단은 ${verdict}입니다.`;

  return {
    asOf: shortDate(decision.date || prediction.prediction_asof_date || pooled.asof_date),
    readiness,
    blockPressure,
    opportunityPressure,
    tone,
    color: tone === "good" ? colors.green : tone === "warn" ? colors.amber : tone === "bad" ? colors.red : colors.blue,
    verdict,
    meaning,
    nextAction,
    factors,
    insights: buildSynthesisInsights({
      score,
      hasTrigger,
      predictionAllowed,
      predictionEdge,
      riskCapacity,
      maxWeight,
      pSuccess,
      pStop,
      threshold,
      dataIntegrity,
      newsDrag,
      block,
      strongest,
      weakest,
      decision,
      risk,
      latestNews,
    }),
    routes: buildSynthesisRoutes({ strongest, weakest, block, nextAction, readiness, blockPressure, opportunityPressure }),
  };
}

function buildSynthesisVerdict(ctx) {
  if (ctx.dataIntegrity < 40) return "데이터 확인 먼저";
  if (ctx.blockPressure > ctx.readiness + 14) return "근거 보강 대기";
  if (ctx.hasTrigger && ctx.predictionAllowed && ctx.riskCapacity >= 60 && ctx.readiness >= 72) return "실행 후보";
  if (ctx.readiness >= 60) return "조건부 관찰 우위";
  return ctx.state.title || "신규 진입 대기";
}

function buildSynthesisNextAction(ctx) {
  if (ctx.dataIntegrity < 40) return "데이터 갱신";
  if (!ctx.hasTrigger) {
    const breakout = planValue(ctx.planRows, "진입", "20일 고점 돌파 기준가") || planValue(ctx.planRows, "진입", "60일 고점 돌파 기준가");
    return breakout ? `${breakout} 돌파 확인` : "가격 트리거 확인";
  }
  if (!ctx.predictionAllowed) return ctx.block ? labelValue(ctx.block.code) : "모델 게이트 확인";
  if (ctx.riskCapacity < 55) return "손절·비중 재계산";
  return "Paper 기록 확인";
}

function buildSynthesisInsights(ctx) {
  const insights = [];
  if ((ctx.score || 0) >= 60 && !ctx.hasTrigger) {
    insights.push({
      label: "가격 의미",
      title: "점수는 쌓였지만 행동 신호는 없다",
      body: `${fmtNumber(ctx.score, 1)}점은 관찰권이지만 트리거가 없어 매수 판단으로 번역되지 않습니다.`,
      tone: "warn",
    });
  }
  if (!ctx.predictionAllowed && (ctx.score || 0) >= 55) {
    insights.push({
      label: "모순 발견",
      title: "룰 신호와 모델 신뢰가 분리됨",
      body: `가격·룰은 읽을 만하지만 예측 객체는 아직 판단 근거가 아니라서 결론이 보수적으로 내려갑니다.`,
      tone: "warn",
    });
  }
  if (ctx.riskCapacity < 55) {
    insights.push({
      label: "실행 제약",
      title: "신호보다 비중 한도가 먼저 병목",
      body: `허용 비중 ${fmtMaybePct(ctx.maxWeight, 2)} 상태라 상승 신호가 와도 포지션 크기는 제한됩니다.`,
      tone: "warn",
    });
  }
  if (ctx.pSuccess !== null && ctx.pStop !== null) {
    const spread = ctx.pSuccess - ctx.pStop;
    insights.push({
      label: "확률 합성",
      title: spread >= 10 ? "성공 확률이 손절 위험보다 충분히 높음" : "성공 확률과 손절 위험 차이가 작음",
      body: `20D 성공 ${fmtNumber(ctx.pSuccess, 1)}%, 손절 ${fmtNumber(ctx.pStop, 1)}%, 기준 ${ctx.threshold === null ? "없음" : `${fmtNumber(ctx.threshold, 1)}%`}를 함께 본 값입니다.`,
      tone: spread >= 10 ? "good" : "warn",
    });
  }
  if (ctx.newsDrag < 25) {
    insights.push({
      label: "원인 해석",
      title: "뉴스는 현재 핵심 차단 원인이 아님",
      body: `${labelValue(ctx.latestNews.news_primary_cause_type || "NO_HIGH_CONFIDENCE_NEWS")} 상태라 가격·모델 쪽 병목을 우선 봐야 합니다.`,
      tone: "neutral",
    });
  } else {
    insights.push({
      label: "원인 해석",
      title: "뉴스가 판단을 흔드는 압력으로 반영됨",
      body: `${labelValue(ctx.latestNews.news_match_confidence || "NO_MATCH")} 뉴스와 감점 ${fmtNumber(ctx.latestNews.news_penalty_event, 1)}가 합성 점수를 낮춥니다.`,
      tone: "warn",
    });
  }
  if (ctx.dataIntegrity >= 75 && ctx.block) {
    insights.push({
      label: "병목 위치",
      title: "데이터가 아니라 게이트가 막고 있음",
      body: `입력 데이터보다 ${labelValue(ctx.block.code)} 조건이 오늘 판단 연결을 제한합니다.`,
      tone: "warn",
    });
  }
  if (!insights.length) {
    insights.push({
      label: "핵심 의미",
      title: `${ctx.strongest.label}이 오늘 판단의 주된 설명 변수`,
      body: `${ctx.weakest.label} 보강 전까지는 결론을 확대하지 않는 편이 맞습니다.`,
      tone: ctx.strongest.tone,
    });
  }
  return insights.slice(0, 5);
}

function buildSynthesisRoutes(ctx) {
  return [
    {
      label: "기회 경로",
      value: fmtNumber(ctx.opportunityPressure, 0),
      formula: `${ctx.strongest.label} → 판단 우위`,
      body: `${ctx.strongest.meaning} · ${ctx.strongest.raw}`,
      tone: ctx.strongest.tone,
    },
    {
      label: "차단 경로",
      value: fmtNumber(ctx.blockPressure, 0),
      formula: `${ctx.weakest.label} → 보수 결론`,
      body: ctx.block ? `${labelValue(ctx.block.code)} · ${fmtNumber(ctx.block.count, 0)}개 기준` : ctx.weakest.meaning,
      tone: ctx.blockPressure > 55 ? "warn" : "neutral",
    },
    {
      label: "다음 액션",
      value: fmtNumber(ctx.readiness, 0),
      formula: ctx.nextAction,
      body: "다음 업데이트에서 이 조건이 바뀌면 결론 카드와 그래프가 함께 재계산됩니다.",
      tone: ctx.readiness >= 65 ? "good" : "warn",
    },
  ];
}

function synthesisScoreTone(score) {
  if (score >= 70) return "good";
  if (score < 45) return "warn";
  return "neutral";
}

function koreanSubject(text) {
  const value = String(text || "");
  const chars = Array.from(value.trim());
  const last = chars.at(-1);
  if (!last) return value;
  const code = last.charCodeAt(0);
  if (code < 0xac00 || code > 0xd7a3) return `${value}가`;
  return `${value}${(code - 0xac00) % 28 ? "이" : "가"}`;
}

function koreanTopic(text) {
  const value = String(text || "");
  const chars = Array.from(value.trim());
  const last = chars.at(-1);
  if (!last) return value;
  const code = last.charCodeAt(0);
  if (code < 0xac00 || code > 0xd7a3) return `${value}는`;
  return `${value}${(code - 0xac00) % 28 ? "은" : "는"}`;
}

function synthesisFactorHtml(factor) {
  const offset = factor.influence >= 0 ? "positive" : "negative";
  const width = clampNumber(Math.abs(factor.influence) * 2, 4, 100);
  return `
    <article class="fusion-factor ${factor.tone || "neutral"} ${offset}" style="--force:${width}%; --factor-color:${factor.color}">
      <div>
        <span>${escapeHtml(factor.label)}</span>
        <strong>${escapeHtml(factor.value)}</strong>
      </div>
      <p title="${escapeHtml(`${factor.raw} · ${factor.meaning}`)}">${escapeHtml(compactText(`${factor.raw} · ${factor.meaning}`, 116))}</p>
      <div class="factor-meter"><i></i></div>
    </article>
  `;
}

function synthesisInsightHtml(item) {
  return `
    <article class="synthesis-insight ${item.tone || "neutral"}">
      <span>${escapeHtml(item.label)}</span>
      <strong>${escapeHtml(item.title)}</strong>
      <p title="${escapeHtml(item.body)}">${escapeHtml(compactText(item.body, 126))}</p>
    </article>
  `;
}

function synthesisRouteHtml(item) {
  return `
    <article class="synthesis-route ${item.tone || "neutral"}">
      <span>${escapeHtml(item.label)}</span>
      <div><strong>${escapeHtml(item.value)}</strong><b>${escapeHtml(item.formula)}</b></div>
      <p title="${escapeHtml(item.body)}">${escapeHtml(compactText(item.body, 120))}</p>
    </article>
  `;
}

function ontologyNodeHtml(node) {
  return `
    <article class="ontology-node ${node.tone || "neutral"}${node.primary ? " primary" : ""}" style="left:${node.x}%; top:${node.y}%;">
      <span>${escapeHtml(node.label)}</span>
      <strong>${escapeHtml(node.title)}</strong>
      <p title="${escapeHtml(node.note)}">${escapeHtml(compactText(node.note, 76))}</p>
    </article>
  `;
}

function renderOntologyLinkedCharts(id, data) {
  const container = $(id);
  if (!container) return;
  const decision = data.snapshots.decision || {};
  const risk = data.snapshots.risk || {};
  const prediction = data.snapshots.prediction || {};
  const pooled = data.snapshots.pooled_prediction || {};
  const latestNews = data.snapshots.latest_news || {};
  const planRows = data.tables.trading_plan || [];
  const priceRows = (data.series.price || []).slice(-180);
  const signalRows = (data.series.signals || []).slice(-180);
  const riskRows = (data.series.risk || []).slice(-180).map((row) => ({
    ...row,
    final_weight_pct_view: (snapshotWeight(row.final_recommended_max_weight) ?? 0) * 100,
    vol_limit_pct_view: (snapshotWeight(row.vol_limit_weight) ?? 0) * 100,
    score_limit_pct_view: (snapshotWeight(row.score_limit_weight) ?? 0) * 100,
  }));
  const newsRows = latestFirst(data.tables.news_daily || []).slice(0, 120).reverse().map((row) => ({
    ...row,
    news_event_count_3d_scaled: clampNumber((toNumber(row.news_event_count_3d) || 0) * 1.5, 0, 100),
    news_penalty_scaled: clampNumber(Math.abs(toNumber(row.news_penalty_event) || 0) * 20, 0, 100),
  }));
  const blockRows = blockReasonRows(data).slice(0, 6);
  const best = bestBacktestStrategy(data.tables.backtest_summary || []);
  const equityRows = (data.series.equity || []).filter((row) => !best.strategy_id || String(row.strategy_id) === String(best.strategy_id)).slice(-260);

  const breakout20 = moneyNumber(planValue(planRows, "진입", "20일 고점 돌파 기준가"));
  const breakout60 = moneyNumber(planValue(planRows, "진입", "60일 고점 돌파 기준가"));
  const stopPrice = toNumber(risk.stop_price_2atr);
  const pSuccess = probabilityPercent(pooled.p_success_20d || prediction.trade_ready_p_success_20d || prediction.context_p_success_20d);
  const pStop = probabilityPercent(pooled.p_stop_hit_20d || prediction.trade_ready_p_stop_hit_20d || prediction.context_p_stop_hit_20d);

  const cards = [
    {
      label: "가격 그래프 객체",
      title: `${fmtCurrency(decision.close)} · ${labelValue(decision.trend_regime)}`,
      note: `20D ${fmtCurrency(breakout20)} · 60D ${fmtCurrency(breakout60)} · 손절 ${fmtCurrency(stopPrice)}`,
      tone: scoreTone(decision.score_price_algo_total, decision.entry_trigger),
      links: ["데이터", "가격", "룰", "판단"],
      chart: miniLineSvg(
        priceRows,
        [
          { key: "close", label: "종가", color: colors.ink },
          { key: "sma_20", label: "20일선", color: colors.blue },
          { key: "sma_50", label: "50일선", color: colors.teal },
        ],
        {
          yFormat: (v) => `$${Math.round(v)}`,
          thresholds: [
            { value: breakout20, label: "20D", color: colors.green },
            { value: breakout60, label: "60D", color: colors.violet },
            { value: stopPrice, label: "손절", color: colors.red },
          ],
        }
      ),
    },
    {
      label: "룰 점수 객체",
      title: `${fmtNumber(decision.score_price_algo_total, 1)}점 · ${labelValue(decision.entry_trigger)}`,
      note: `75점 이상과 트리거가 함께 나와야 매수 후보로 이동`,
      tone: scoreTone(decision.score_price_algo_total, decision.entry_trigger),
      links: ["가격", "룰", "리스크", "판단"],
      chart: miniLineSvg(
        signalRows,
        [{ key: "score_price_algo_total", label: "점수", color: colors.blue }],
        { minY: 0, maxY: 100, yFormat: (v) => fmtNumber(v, 0), thresholds: [{ value: 75, label: "후보", color: colors.red }] }
      ),
    },
    {
      label: "리스크 한도 객체",
      title: `${fmtMaybePct(snapshotWeightValue(risk, ["final_recommended_max_weight", "final_recommended_max_weight_pct"]), 2)} · ${labelValue(risk.risk_state)}`,
      note: `${labelValue(risk.limiting_reason)} · 손절까지 ${fmtMaybePct(risk.risk_pct_2atr, 2)}`,
      tone: gateTone(riskStatusFromSnapshot(data)),
      links: ["가격", "리스크", "판단", "Paper"],
      chart: miniLineSvg(
        riskRows,
        [
          { key: "final_weight_pct_view", label: "최대 비중", color: colors.green },
          { key: "vol_limit_pct_view", label: "변동성 한도", color: colors.amber },
          { key: "score_limit_pct_view", label: "점수 한도", color: colors.violet },
        ],
        { minY: 0, yFormat: (v) => `${fmtNumber(v, 0)}%` }
      ),
    },
    {
      label: "예측 확률 객체",
      title: `${fmtMaybePct(pooled.p_success_20d, 1)} · ${labelValue(pooled.model_name || prediction.model_health_best_model_20d)}`,
      note: `손절 ${fmtMaybePct(pooled.p_stop_hit_20d, 1)} · 기대 R ${fmtNumber(pooled.expected_r_net_20d, 2)}`,
      tone: isTruthy(pooled.decision_support_allowed) ? "good" : "warn",
      links: ["표본", "예측", "리스크", "판단"],
      chart: miniBarsSvg([
        { label: "성공", value: pSuccess, color: colors.green },
        { label: "손절위험", value: pStop, color: colors.red },
        { label: "결정점수", value: probabilityPercent(pooled.decision_score_20d), color: colors.blue },
        { label: "기준값", value: probabilityPercent(pooled.threshold_20d), color: colors.violet },
      ]),
    },
    {
      label: "뉴스 원인 객체",
      title: `${labelValue(latestNews.news_match_confidence || "NO_MATCH")} · ${labelValue(latestNews.news_primary_cause_type || "NO_HIGH_CONFIDENCE_NEWS")}`,
      note: `3일 뉴스 ${fmtNumber(latestNews.news_event_count_3d, 0)}건 · 감점 ${fmtNumber(latestNews.news_penalty_event, 1)}`,
      tone: newsTone(latestNews),
      links: ["뉴스", "룰", "예측", "판단"],
      chart: miniLineSvg(
        newsRows,
        [
          { key: "news_match_confidence_score", label: "매칭 점수", color: colors.teal },
          { key: "news_event_count_3d_scaled", label: "3일 뉴스", color: colors.blue },
          { key: "news_penalty_scaled", label: "감점 압력", color: colors.red },
        ],
        { minY: 0, maxY: 100, yFormat: (v) => fmtNumber(v, 0) }
      ),
    },
    {
      label: "검증·성과 객체",
      title: best.strategy_id ? `${labelValue(best.strategy_id)} · CAGR ${fmtMaybePct(best.cagr_pct, 1)}` : "성과 데이터 없음",
      note: `MDD ${fmtMaybePct(best.max_drawdown_pct, 1)} · PF ${fmtNumber(best.profit_factor, 2)} · 차단 ${fmtNumber(blockRows.length, 0)}개`,
      tone: best.strategy_id ? "neutral" : "warn",
      links: ["검증", "예측", "차단", "판단"],
      chart: equityRows.length
        ? miniLineSvg(equityRows, [{ key: "equity", label: "자산곡선", color: colors.green }], { yFormat: (v) => fmtNumber(v, 1) })
        : miniReasonBarsSvg(blockRows),
    },
  ];

  container.innerHTML = cards.map(ontologyChartCardHtml).join("");
}

function ontologyChartCardHtml(card) {
  const links = (card.links || []).map((link, idx) => `${idx ? "<i>→</i>" : ""}<span>${escapeHtml(link)}</span>`).join("");
  return `
    <article class="ontology-chart-card ${card.tone || "neutral"}">
      <header>
        <div>
          <span>${escapeHtml(card.label)}</span>
          <strong>${escapeHtml(card.title)}</strong>
          <p title="${escapeHtml(card.note)}">${escapeHtml(compactText(card.note, 140))}</p>
        </div>
      </header>
      <div class="ontology-mini-chart">${card.chart}</div>
      <div class="object-link-row">${links}</div>
    </article>
  `;
}

function renderOntologyRelationMatrix(id, data) {
  const container = $(id);
  if (!container) return;
  const system = data.snapshots.system || {};
  const dataQuality = data.snapshots.data_quality || {};
  const decision = data.snapshots.decision || {};
  const risk = data.snapshots.risk || {};
  const prediction = data.snapshots.prediction || {};
  const pooled = data.snapshots.pooled_prediction || {};
  const latestNews = data.snapshots.latest_news || {};
  const modelGate = data.snapshots.model_gate || {};
  const paper = data.snapshots.paper_gate || {};
  const block = primaryBlockReason(data);
  const rows = [
    {
      from: "데이터",
      to: "가격 그래프",
      signal: labelValue(dataQuality.data_quality_status || "MISSING"),
      impact: `${shortDate(dataQuality.latest_signal_date)} · 실패 ${fmtNumber(dataQuality.failed_checks, 0)}개`,
      tone: statusTone(dataQuality.data_quality_status),
    },
    {
      from: "가격 그래프",
      to: "룰 점수",
      signal: `${fmtCurrency(decision.close)} · ${fmtNumber(decision.score_price_algo_total, 1)}점`,
      impact: `${labelValue(decision.trend_regime)} · ${labelValue(decision.vol_regime)}`,
      tone: scoreTone(decision.score_price_algo_total, decision.entry_trigger),
    },
    {
      from: "뉴스 원인",
      to: "룰/예측",
      signal: labelValue(latestNews.news_match_confidence || "NO_MATCH"),
      impact: `${labelValue(latestNews.news_primary_cause_type || "NO_HIGH_CONFIDENCE_NEWS")} · 감점 ${fmtNumber(latestNews.news_penalty_event, 1)}`,
      tone: newsTone(latestNews),
    },
    {
      from: "검증·표본",
      to: "예측",
      signal: labelValue(modelGate.model_gate_status || pooled.model_support_route),
      impact: block ? `${labelValue(block.code)} · ${fmtNumber(block.count, 0)}개 기준` : "차단 없음",
      tone: isTruthy(system.prediction_decision_support) ? "good" : "warn",
    },
    {
      from: "예측",
      to: "판단",
      signal: `${fmtMaybePct(pooled.p_success_20d || prediction.trade_ready_p_success_20d, 1)} 성공`,
      impact: `${statusByBool(system.prediction_decision_support, "판단 가능", "참고용")} · 손절 ${fmtMaybePct(pooled.p_stop_hit_20d, 1)}`,
      tone: isTruthy(system.prediction_decision_support) ? "good" : "warn",
    },
    {
      from: "리스크",
      to: "판단",
      signal: fmtMaybePct(snapshotWeightValue(risk, ["final_recommended_max_weight", "final_recommended_max_weight_pct"]), 2),
      impact: `${labelValue(risk.risk_state)} · ${labelValue(risk.limiting_reason)}`,
      tone: gateTone(riskStatusFromSnapshot(data)),
    },
    {
      from: "판단",
      to: "Paper 기록",
      signal: labelValue(decision.trade_action || "NO_TRADE"),
      impact: `${labelValue(paper.paper_gate_status || system.paper_trading_status)} · live ${labelValue(system.live_trading_status)}`,
      tone: isTruthy(system.paper_ready) ? "good" : "warn",
    },
  ];

  container.innerHTML = rows
    .map((row) => `
      <article class="relation-card ${row.tone || "neutral"}">
        <div class="relation-route"><span>${escapeHtml(row.from)}</span><i>→</i><span>${escapeHtml(row.to)}</span></div>
        <strong>${escapeHtml(row.signal)}</strong>
        <p title="${escapeHtml(row.impact)}">${escapeHtml(compactText(row.impact, 112))}</p>
      </article>
    `)
    .join("");
}

function miniLineSvg(rows, series, options = {}) {
  const usableRows = (rows || []).filter((row) => series.some((s) => miniValue(row, s) !== null));
  if (!usableRows.length) return `<div class="preview-empty mini-empty">데이터 없음</div>`;
  const w = 720;
  const h = 230;
  const p = { l: 48, r: 18, t: 26, b: 30 };
  const thresholds = (options.thresholds || []).filter((item) => toNumber(item.value) !== null);
  const values = [];
  for (const row of usableRows) for (const s of series) {
    const n = miniValue(row, s);
    if (n !== null) values.push(n);
  }
  for (const item of thresholds) values.push(toNumber(item.value));
  let yMin = options.minY ?? Math.min(...values);
  let yMax = options.maxY ?? Math.max(...values);
  if (yMin === yMax) {
    yMin -= 1;
    yMax += 1;
  }
  const pad = (yMax - yMin) * 0.08;
  yMin = options.minY ?? yMin - pad;
  yMax = options.maxY ?? yMax + pad;
  const x = (i) => p.l + (i * (w - p.l - p.r)) / Math.max(1, usableRows.length - 1);
  const y = (v) => h - p.b - ((v - yMin) / (yMax - yMin)) * (h - p.t - p.b);
  const yFormat = options.yFormat || fmtNumber;
  const grid = [];
  for (let i = 0; i <= 3; i += 1) {
    const value = yMin + ((yMax - yMin) * i) / 3;
    const yy = y(value);
    grid.push(`<line x1="${p.l}" y1="${yy}" x2="${w - p.r}" y2="${yy}" stroke="currentColor" opacity="0.11" />`);
    grid.push(`<text x="6" y="${yy + 4}" class="axis-label">${escapeHtml(yFormat(value))}</text>`);
  }
  const thresholdLines = thresholds
    .map((item) => {
      const yy = y(toNumber(item.value));
      return `<line x1="${p.l}" y1="${yy}" x2="${w - p.r}" y2="${yy}" stroke="${item.color || colors.amber}" stroke-width="2" stroke-dasharray="7 7" /><text x="${w - p.r - 42}" y="${yy - 5}" class="axis-label">${escapeHtml(item.label)}</text>`;
    })
    .join("");
  const paths = series
    .map((s) => {
      const pts = usableRows
        .map((row, i) => {
          const n = miniValue(row, s);
          return n === null ? null : `${x(i).toFixed(1)},${y(n).toFixed(1)}`;
        })
        .filter(Boolean)
        .join(" ");
      return pts ? `<polyline points="${pts}" fill="none" stroke="${s.color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />` : "";
    })
    .join("");
  const lastDots = series
    .map((s) => {
      const lastIndex = usableRows.findLastIndex ? usableRows.findLastIndex((row) => miniValue(row, s) !== null) : lastValueIndex(usableRows, s);
      if (lastIndex < 0) return "";
      const value = miniValue(usableRows[lastIndex], s);
      return `<circle cx="${x(lastIndex)}" cy="${y(value)}" r="4.5" fill="${s.color}"></circle>`;
    })
    .join("");
  const axis = `<text x="${p.l}" y="${h - 8}" class="axis-label">${escapeHtml(shortDate(usableRows[0]?.date))}</text><text x="${w - p.r - 78}" y="${h - 8}" class="axis-label">${escapeHtml(shortDate(usableRows.at(-1)?.date))}</text>`;
  return `${miniLegend(series)}${chartSvg(w, h, grid.join("") + thresholdLines + paths + lastDots + axis)}`;
}

function miniValue(row, seriesItem) {
  const raw = typeof seriesItem.value === "function" ? seriesItem.value(row) : row[seriesItem.key];
  return toNumber(raw);
}

function lastValueIndex(rows, seriesItem) {
  for (let i = rows.length - 1; i >= 0; i -= 1) {
    if (miniValue(rows[i], seriesItem) !== null) return i;
  }
  return -1;
}

function miniLegend(series) {
  return `<div class="mini-legend">${series.map((s) => `<span><i style="background:${s.color}"></i>${escapeHtml(s.label)}</span>`).join("")}</div>`;
}

function miniBarsSvg(items) {
  const prepared = (items || []).filter((item) => toNumber(item.value) !== null);
  if (!prepared.length) return `<div class="preview-empty mini-empty">데이터 없음</div>`;
  return `
    <div class="mini-bars">
      ${prepared
        .map((item) => {
          const width = clampNumber(toNumber(item.value), 0, 100);
          return `<div class="mini-bar-row"><span>${escapeHtml(item.label)}</span><div><i style="width:${width}%; background:${item.color || colors.blue}"></i></div><strong>${escapeHtml(fmtNumber(width, 1))}%</strong></div>`;
        })
        .join("")}
    </div>
  `;
}

function miniReasonBarsSvg(rows) {
  const top = (rows || []).slice(0, 5);
  if (!top.length) return `<div class="preview-empty mini-empty">차단 원인 없음</div>`;
  const maxCount = Math.max(...top.map((row) => row.count || 1), 1);
  return `
    <div class="mini-bars">
      ${top
        .map((row) => {
          const width = clampNumber(((row.count || 1) / maxCount) * 100, 8, 100);
          return `<div class="mini-bar-row"><span>${escapeHtml(labelValue(row.code))}</span><div><i style="width:${width}%; background:${colors.amber}"></i></div><strong>${escapeHtml(fmtNumber(row.count, 0))}</strong></div>`;
        })
        .join("")}
    </div>
  `;
}

function moneyNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(String(value).replace(/[$,%\s,]/g, ""));
  return Number.isFinite(parsed) ? parsed : null;
}

function renderOntologyPriorityStack(id, data) {
  const container = $(id);
  if (!container) return;
  const system = data.snapshots.system || {};
  const dataQuality = data.snapshots.data_quality || {};
  const decision = data.snapshots.decision || {};
  const risk = data.snapshots.risk || {};
  const prediction = data.snapshots.prediction || {};
  const pooled = data.snapshots.pooled_prediction || {};
  const latestNews = data.snapshots.latest_news || {};
  const state = overviewDecisionState(decision, prediction, system);
  const block = primaryBlockReason(data);
  const planRows = data.tables.trading_plan || [];
  const items = [
    {
      rank: "P1",
      title: state.title,
      meta: "오늘 행동 객체",
      body: `${state.action} · ${state.trigger}`,
      tone: state.tone,
    },
    {
      rank: "P2",
      title: planValue(planRows, "진입", "20일 고점 돌파 기준가") || "트리거 대기",
      meta: "다음 가격 객체",
      body: `현재 ${fmtCurrency(decision.close)} · 60D ${planValue(planRows, "진입", "60일 고점 돌파 기준가") || "없음"}`,
      tone: scoreTone(decision.score_price_algo_total, decision.entry_trigger),
    },
    {
      rank: "P3",
      title: fmtMaybePct(snapshotWeightValue(risk, ["final_recommended_max_weight", "final_recommended_max_weight_pct"]), 2),
      meta: "리스크 한도 객체",
      body: `${labelValue(risk.risk_state)} · ${labelValue(risk.limiting_reason)}`,
      tone: gateTone(riskStatusFromSnapshot(data)),
    },
    {
      rank: "P4",
      title: statusByBool(system.prediction_decision_support, "예측 판단 가능", "예측 참고용"),
      meta: "모델 게이트 객체",
      body: block ? `${labelValue(block.code)} · ${fmtNumber(block.count, 0)}개 기준` : labelValue(pooled.model_support_route || prediction.prediction_use_status),
      tone: isTruthy(system.prediction_decision_support) ? "good" : "warn",
    },
    {
      rank: "P5",
      title: labelValue(latestNews.news_primary_cause_type || "NO_HIGH_CONFIDENCE_NEWS"),
      meta: "뉴스 원인 객체",
      body: `${labelValue(latestNews.news_match_confidence || "NO_MATCH")} · 감점 ${fmtNumber(latestNews.news_penalty_event, 1)}`,
      tone: newsTone(latestNews),
    },
    {
      rank: "P6",
      title: labelValue(dataQuality.data_quality_status || system.system_state),
      meta: "데이터 품질 객체",
      body: `${shortDate(dataQuality.latest_signal_date)} · 실패 ${fmtNumber(dataQuality.failed_checks, 0)}개`,
      tone: statusTone(dataQuality.data_quality_status),
    },
  ];
  container.innerHTML = items
    .map((item) => `
      <article class="priority-item ${item.tone || "neutral"}">
        <b>${escapeHtml(item.rank)}</b>
        <div>
          <span>${escapeHtml(item.meta)}</span>
          <strong>${escapeHtml(item.title)}</strong>
          <p title="${escapeHtml(item.body)}">${escapeHtml(compactText(item.body, 110))}</p>
        </div>
      </article>
    `)
    .join("");
}

function renderOntologyRiskReward(id, data) {
  const container = $(id);
  if (!container) return;
  const decision = data.snapshots.decision || {};
  const risk = data.snapshots.risk || {};
  const prediction = data.snapshots.prediction || {};
  const pooled = data.snapshots.pooled_prediction || {};
  const score = toNumber(decision.score_price_algo_total);
  const maxWeight = snapshotWeightValue(risk, ["final_recommended_max_weight", "final_recommended_max_weight_pct"]);
  const maxWeightPct = maxWeight === null ? null : maxWeight * 100;
  const pSuccess = probabilityPercent(snapshotValue(pooled, ["p_success_20d", "p_success_tsm_like_20d"]) || snapshotValue(prediction, ["trade_ready_p_success_20d", "trigger_p_success_20d", "context_p_success_20d"]));
  const stopRisk = probabilityPercent(snapshotValue(pooled, ["p_stop_hit_20d"]) || snapshotValue(prediction, ["trade_ready_p_stop_hit_20d", "trigger_p_stop_hit_20d", "context_p_stop_hit_20d"]));
  const expectedR = toNumber(snapshotValue(pooled, ["expected_r_net_20d"]) || snapshotValue(prediction, ["trade_ready_expected_r_20d", "trigger_expected_r_20d", "context_expected_r_20d"]));
  const meters = [
    {
      label: "알고리즘 점수",
      value: `${fmtNumber(score, 1)} / 75`,
      note: labelValue(decision.entry_trigger || "NO_ENTRY_TRIGGER"),
      fill: clampNumber(score || 0, 0, 100),
      tone: scoreTone(score, decision.entry_trigger),
    },
    {
      label: "허용 비중",
      value: fmtMaybePct(maxWeight, 2),
      note: labelValue(risk.limiting_reason || risk.risk_state),
      fill: clampNumber(((maxWeightPct || 0) / 20) * 100, 0, 100),
      tone: gateTone(riskStatusFromSnapshot(data)),
    },
    {
      label: "20D 성공 가능성",
      value: pSuccess === null ? "없음" : `${fmtNumber(pSuccess, 1)}%`,
      note: labelValue(pooled.model_name || prediction.model_health_best_model_20d || "모델 없음"),
      fill: clampNumber(pSuccess || 0, 0, 100),
      tone: pSuccess !== null && pSuccess >= 55 ? "good" : pSuccess !== null && pSuccess >= 40 ? "neutral" : "warn",
    },
    {
      label: "20D 손절 위험",
      value: stopRisk === null ? "없음" : `${fmtNumber(stopRisk, 1)}%`,
      note: `2ATR 손절 ${fmtCurrency(risk.stop_price_2atr)} · 거리 ${fmtMaybePct(risk.risk_pct_2atr, 2)}`,
      fill: clampNumber(stopRisk || 0, 0, 100),
      tone: stopRisk !== null && stopRisk <= 35 ? "good" : stopRisk !== null && stopRisk <= 50 ? "warn" : "bad",
    },
    {
      label: "기대 R",
      value: expectedR === null ? "없음" : fmtNumber(expectedR, 2),
      note: `기대 순수익 ${fmtPct(pooled.expected_net_return_pct_20d, 2, false)}`,
      fill: clampNumber(((expectedR || 0) / 1.5) * 100, 0, 100),
      tone: expectedR !== null && expectedR >= 0.5 ? "good" : "warn",
    },
  ];
  container.innerHTML = meters.map(riskMeterHtml).join("");
}

function riskMeterHtml(item) {
  return `
    <article class="risk-meter ${item.tone || "neutral"}" style="--meter:${item.fill}%">
      <div><span>${escapeHtml(item.label)}</span><strong>${escapeHtml(item.value)}</strong></div>
      <div class="meter-track"><i></i></div>
      <p title="${escapeHtml(item.note)}">${escapeHtml(compactText(item.note, 92))}</p>
    </article>
  `;
}

function renderOntologyEvidenceRail(id, data) {
  const container = $(id);
  if (!container) return;
  const system = data.snapshots.system || {};
  const dataQuality = data.snapshots.data_quality || {};
  const modelGate = data.snapshots.model_gate || {};
  const paperGate = data.snapshots.paper_gate || {};
  const decision = data.snapshots.decision || {};
  const risk = data.snapshots.risk || {};
  const latestNews = data.snapshots.latest_news || {};
  const steps = [
    {
      label: "데이터",
      value: labelValue(dataQuality.data_quality_status || "MISSING"),
      note: `${shortDate(dataQuality.latest_signal_date)} · ${fmtNumber(dataQuality.checks, 0)}개 점검`,
      tone: statusTone(dataQuality.data_quality_status),
    },
    {
      label: "가격·룰",
      value: labelValue(decision.entry_trigger || "NO_ENTRY_TRIGGER"),
      note: `${fmtNumber(decision.score_price_algo_total, 1)}점 · ${labelValue(decision.trade_action)}`,
      tone: scoreTone(decision.score_price_algo_total, decision.entry_trigger),
    },
    {
      label: "뉴스 원인",
      value: labelValue(latestNews.news_match_confidence || "NO_MATCH"),
      note: `${labelValue(latestNews.news_primary_cause_type || "NO_HIGH_CONFIDENCE_NEWS")} · ${fmtNumber(latestNews.news_penalty_event, 1)}점`,
      tone: newsTone(latestNews),
    },
    {
      label: "예측",
      value: statusByBool(system.prediction_decision_support, "통과", "참고용"),
      note: labelValue(modelGate.model_gate_status || system.prediction_use_status),
      tone: isTruthy(system.prediction_decision_support) ? "good" : "warn",
    },
    {
      label: "리스크",
      value: labelValue(risk.risk_state),
      note: `최대 ${fmtMaybePct(snapshotWeightValue(risk, ["final_recommended_max_weight", "final_recommended_max_weight_pct"]), 2)}`,
      tone: gateTone(riskStatusFromSnapshot(data)),
    },
    {
      label: "Paper/Live",
      value: labelValue(paperGate.paper_gate_status || system.paper_trading_status),
      note: labelValue(system.live_trading_status || "DISABLED_BY_DESIGN"),
      tone: isTruthy(paperGate.paper_decision_support_allowed) ? "good" : "warn",
    },
  ];
  container.innerHTML = steps
    .map((step, idx) => `
      <article class="evidence-step ${step.tone || "neutral"}">
        <em>${idx + 1}</em>
        <span>${escapeHtml(step.label)}</span>
        <strong>${escapeHtml(step.value)}</strong>
        <p title="${escapeHtml(step.note)}">${escapeHtml(compactText(step.note, 86))}</p>
      </article>
    `)
    .join("");
}

function renderOntologyBlockGraph(id, data) {
  const container = $(id);
  if (!container) return;
  const rows = blockReasonRows(data).slice(0, 7);
  if (!rows.length) {
    container.innerHTML = `<div class="preview-empty">차단 원인 없음</div>`;
    return;
  }
  const maxCount = Math.max(...rows.map((row) => row.count || 1), 1);
  container.innerHTML = `
    <article class="block-graph-center">
      <span>핵심 차단 객체</span>
      <strong>${escapeHtml(labelValue(rows[0].code))}</strong>
      <p>${escapeHtml(explainReason(rows[0].code))}</p>
    </article>
    <div class="block-node-list">
      ${rows
        .map((row, idx) => {
          const width = clampNumber(((row.count || 1) / maxCount) * 100, 12, 100);
          return `
            <article class="block-node" style="--weight:${width}%">
              <b>${escapeHtml(String(idx + 1).padStart(2, "0"))}</b>
              <div>
                <span>${escapeHtml(labelValue(row.category || row.source || "차단 기준"))}</span>
                <strong>${escapeHtml(labelValue(row.code))}</strong>
                <i></i>
              </div>
              <em>${escapeHtml(fmtNumber(row.count, 0))}</em>
            </article>
          `;
        })
        .join("")}
    </div>
  `;
}

function renderOntologyNewsCausal(id, data) {
  const container = $(id);
  if (!container) return;
  const latestNews = data.snapshots.latest_news || {};
  const latestFeatures = data.snapshots.latest_prediction_features || {};
  const penaltySummary = data.snapshots.news_penalty_summary || {};
  const highRows = data.tables.news_high_matches || [];
  const contractRows = data.tables.news_feature_contract || [];
  const penalty = toNumber(latestFeatures.news_penalty_event ?? latestNews.news_penalty_event) || 0;
  const cause = labelValue(latestNews.news_primary_cause_type || "NO_HIGH_CONFIDENCE_NEWS");
  const flow = [
    ["수집", labelValue(latestNews.news_coverage_status || "DIRECT_WEB_EMPTY"), `${fmtNumber(latestNews.news_event_count_1d, 0)} / ${fmtNumber(latestNews.news_event_count_3d, 0)}건`],
    ["매칭", labelValue(latestNews.news_match_confidence || "NO_MATCH"), `점수 ${fmtNumber(latestNews.news_match_confidence_score, 1)}`],
    ["룰", penalty > 0 ? `-${fmtNumber(penalty, 1)}점` : "감점 없음", `전체 ${fmtNumber(penaltySummary.penalty_days, 0)}일`],
    ["예측", `${fmtNumber(contractRows.length, 0)}개 피처`, `HIGH ${fmtNumber(highRows.length, 0)}건`],
  ];
  container.innerHTML = `
    <article class="cause-core ${newsTone(latestNews)}">
      <span>${escapeHtml(labelValue(latestNews.news_match_confidence || "NO_MATCH"))}</span>
      <strong>${escapeHtml(cause)}</strong>
      <p title="${escapeHtml(latestNews.news_cause_summary || "")}">${escapeHtml(compactText(latestNews.news_cause_summary || "고신뢰 대표 원인 없음", 120))}</p>
    </article>
    <div class="cause-flow">
      ${flow
        .map(([label, value, note]) => `
          <article>
            <span>${escapeHtml(label)}</span>
            <strong>${escapeHtml(value)}</strong>
            <p>${escapeHtml(note)}</p>
          </article>
        `)
        .join("")}
    </div>
  `;
}

function scoreTone(scoreValue, triggerValue) {
  const score = toNumber(scoreValue);
  const trigger = String(triggerValue || "").toUpperCase();
  if (trigger && !["NONE", "NO_ENTRY_TRIGGER"].includes(trigger)) return "good";
  if (score !== null && score >= 75) return "good";
  if (score !== null && score >= 60) return "warn";
  return "neutral";
}

function clampNumber(value, min, max) {
  const n = Number(value);
  if (!Number.isFinite(n)) return min;
  return Math.max(min, Math.min(max, n));
}

function probabilityPercent(value) {
  const n = toNumber(value);
  if (n === null) return null;
  return Math.abs(n) <= 1 ? n * 100 : n;
}

function primaryBlockReason(data) {
  return blockReasonRows(data)[0] || null;
}

function blockReasonRows(data) {
  const rootCauses = data.tables.model_gate_root_causes || [];
  const rows = rootCauses
    .filter((row) => String(row.root_cause || "").toUpperCase() !== "PASS")
    .map((row) => ({
      code: row.root_cause,
      count: toNumber(row.failed_gate_count) || 1,
      priority: toNumber(row.priority) || 99,
      category: row.category,
      source: row.gate_groups,
    }));
  if (rows.length) return rows.sort((a, b) => (toNumber(a.priority) || 99) - (toNumber(b.priority) || 99) || b.count - a.count);

  const system = data.snapshots.system || {};
  const modelGate = data.snapshots.model_gate || {};
  const grouped = new Map();
  const add = (source, rawReason) => {
    for (const reason of reasonCodes(rawReason)) {
      const current = grouped.get(reason.code) || { code: reason.code, count: 0, source };
      current.count += 1;
      grouped.set(reason.code, current);
    }
  };
  add("예측", system.prediction_block_reasons);
  add("모델", modelGate.pooled_decision_block_reasons || modelGate.local_diagnostic_block_reasons);
  return Array.from(grouped.values()).sort((a, b) => b.count - a.count);
}

function overviewDecisionState(decision, prediction, system) {
  const tradeAction = String(decision.trade_action || "").toUpperCase();
  const entryTrigger = String(decision.entry_trigger || snapshotValue(prediction, ["latest_entry_gate_status", "prediction_entry_gate_status"]) || "").toUpperCase();
  const researchStage = String(decision.research_signal_stage || "NO_RESEARCH_SIGNAL").toUpperCase();
  if (tradeAction.includes("BUY") || tradeAction.includes("ENTER")) {
    return {
      title: "진입 후보 발생",
      action: labelValue(decision.trade_action),
      trigger: labelValue(decision.entry_trigger),
      note: "룰 신호와 리스크 비중을 함께 확인",
      tone: "good",
    };
  }
  if (researchStage === "PAPER_BUY_SETUP") {
    return {
      title: "공격형 Paper 후보",
      action: labelValue(decision.research_signal_action),
      trigger: labelValue(decision.entry_trigger || decision.research_signal_stage),
      note: "실전 주문이 아니라 가상 기록으로 먼저 검증",
      tone: "neutral",
    };
  }
  if (researchStage === "EARLY_BULLISH_WATCH") {
    return {
      title: "관찰 강화",
      action: labelValue(decision.research_signal_action),
      trigger: labelValue(decision.research_signal_stage),
      note: "실전 매수 전 단계의 조기 상승 구조",
      tone: "neutral",
    };
  }
  if (tradeAction.includes("NO_TRADE") || entryTrigger === "NONE" || entryTrigger.includes("NO_ENTRY")) {
    return {
      title: "신규 진입 대기",
      action: labelValue(decision.trade_action || "NO_TRADE"),
      trigger: "트리거 없음",
      note: "돌파 또는 눌림목 반등 확인 전까지 대기",
      tone: "warn",
    };
  }
  if (!isTruthy(system.prediction_decision_support)) {
    return {
      title: "판단 보류",
      action: labelValue(decision.trade_action),
      trigger: labelValue(decision.entry_trigger),
      note: "예측 게이트가 막혀 있어 룰과 리스크만 참고",
      tone: "warn",
    };
  }
  return {
    title: labelValue(decision.trade_action || "상태 확인"),
    action: labelValue(decision.trade_action),
    trigger: labelValue(decision.entry_trigger),
    note: "세부 탭에서 근거 확인",
    tone: "neutral",
  };
}

function renderMainHero(id, data) {
  const system = data.snapshots.system || {};
  const health = data.snapshots.daily_health || {};
  const dataQuality = data.snapshots.data_quality || {};
  const modelGate = data.snapshots.model_gate || {};
  const paperGate = data.snapshots.paper_gate || {};
  const decision = data.snapshots.decision || {};
  const risk = data.snapshots.risk || {};
  const prediction = data.snapshots.prediction || {};
  const state = overviewDecisionState(decision, prediction, system);
  const subtitle = [
    `${shortDate(decision.date || prediction.prediction_asof_date)} 기준`,
    `현재가 ${fmtCurrency(decision.close)}`,
    `점수 ${fmtNumber(decision.score_price_algo_total, 1)}`,
    state.note,
  ].join(" · ");
  const cards = [
    ["오늘 행동", state.action, state.trigger],
    ["최대 비중", fmtMaybePct(snapshotWeightValue(risk, ["final_recommended_max_weight", "final_recommended_max_weight_pct"]), 2), labelValue(risk.limiting_reason)],
    ["Strict/Paper", `${gateStatusText(paperGate.strict_decision_support_allowed)} / ${gateStatusText(paperGate.paper_decision_support_allowed)}`, labelValue(paperGate.paper_gate_status || modelGate.model_gate_status)],
    ["데이터", labelValue(dataQuality.data_quality_status), `${shortDate(dataQuality.latest_signal_date)} 기준`],
  ];
  $(id).innerHTML = `
    <section class="main-hero-status ${state.tone}">
      <span>현재 결론</span>
      <strong>${escapeHtml(state.title)}</strong>
      <p>${escapeHtml(subtitle)}</p>
    </section>
    <section class="main-hero-cards">
      ${cards.map(([label, value, note]) => `<article><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><p>${escapeHtml(note)}</p></article>`).join("")}
    </section>
  `;
}

function renderGateStrip(id, data) {
  const system = data.snapshots.system || {};
  const dataQuality = data.snapshots.data_quality || {};
  const modelGate = data.snapshots.model_gate || {};
  const paperGate = data.snapshots.paper_gate || {};
  const decision = data.snapshots.decision || {};
  const prediction = data.snapshots.prediction || {};
  const risk = data.snapshots.risk || {};
  const latestNews = data.snapshots.latest_news || {};
  const gates = [
    ["데이터", dataQuality.data_quality_status || "MISSING", `${shortDate(dataQuality.latest_signal_date)} 기준`],
    ["룰 신호", decision.entry_trigger && String(decision.entry_trigger).toUpperCase() !== "NONE" ? "PASS" : "BLOCKED", decision.trade_action || "NO_TRADE"],
    ["연구 신호", decision.research_signal_stage || "NO_RESEARCH_SIGNAL", decision.research_signal_action || decision.research_signal_reason || "NO_ACTION"],
    ["뉴스", newsGateStatus(latestNews), latestNews.news_primary_cause_type || latestNews.news_coverage_status || "NO_MATCH"],
    ["위험", riskStatusFromSnapshot(data), `${labelValue(risk.risk_state)} · 최대 ${fmtMaybePct(snapshotWeightValue(risk, ["final_recommended_max_weight", "final_recommended_max_weight_pct"]), 2)}`],
    ["Strict", paperGate.strict_gate_status || modelGate.model_gate_status || "MISSING", modelGate.pooled_decision_block_reasons || system.prediction_block_reasons],
    ["Paper", paperGate.paper_gate_status || "MISSING", paperGate.paper_gate_block_reasons || "PASS"],
    ["실거래", system.live_trading_status || "DISABLED_BY_DESIGN", system.live_block_reasons || "NO_LIVE_BROKER_BY_DESIGN"],
  ];
  $(id).innerHTML = gates
    .map(([label, status, reason]) => {
      const tone = gateTone(status);
      const reasonText = labelValue(reason);
      return `<article class="gate-chip ${tone}"><span>${escapeHtml(label)}</span><strong>${escapeHtml(gateStatusText(status))}</strong><p title="${escapeHtml(reasonText)}">${escapeHtml(compactText(reasonText, 80))}</p></article>`;
    })
    .join("");
}

function renderOverviewSampleBoard(id, data) {
  const container = $(id);
  if (!container) return;
  const universe = data.snapshots.universe_validation || {};
  const sample = data.snapshots.pooled_sample_audit || {};
  const paper = data.snapshots.paper_gate || {};
  const cards = [
    {
      label: "유니버스",
      value: `${fmtNumber(universe.loaded_symbols, 0)} / ${fmtNumber(universe.candidate_symbols, 0)}`,
      note: `strict eligible ${fmtNumber(universe.strict_eligible_symbols, 0)} · short history ${fmtNumber(universe.short_history_research_only, 0)}`,
      tone: isTruthy(universe.loaded_target_pass) && isTruthy(universe.strict_eligible_target_pass) ? "good" : "warn",
    },
    {
      label: "20D 표본",
      value: fmtNumber(sample.trade_ready_20d_labeled, 0),
      note: `학습 ${fmtNumber(sample.model_training_20d_labeled, 0)} · test/holdout ${fmtNumber(sample.test_holdout_trade_ready_20d, 0)}`,
      tone: isTruthy(sample.trade_ready_target_pass) && isTruthy(sample.model_training_target_pass) ? "good" : "warn",
    },
    {
      label: "TSM-like 보정",
      value: fmtNumber(paper.tsm_like_effective_train_validation_n, 0),
      note: `${labelValue(paper.tsm_like_calibration_route)} · ECE ${fmtNumber(paper.tsm_like_calibration_ece, 4)}`,
      tone: isTruthy(paper.tsm_like_route_selection_pass) ? "good" : "warn",
    },
    {
      label: "Paper gate",
      value: gateStatusText(paper.paper_decision_support_allowed),
      note: explainReasonList(paper.paper_gate_block_reasons || paper.paper_gate_status, 2),
      tone: isTruthy(paper.paper_decision_support_allowed) ? "good" : "warn",
    },
    {
      label: "Live",
      value: labelValue(paper.live_trading_status || "DISABLED_BY_DESIGN"),
      note: "paper gate가 통과해도 실거래는 열지 않음",
      tone: "neutral",
    },
  ];
  container.innerHTML = cards.map(sampleCardHtml).join("");
}

function sampleCardHtml(card) {
  return `
    <article class="sample-card ${card.tone || "neutral"}">
      <span>${escapeHtml(card.label)}</span>
      <strong>${escapeHtml(card.value)}</strong>
      <p title="${escapeHtml(card.note)}">${escapeHtml(compactText(card.note, 120))}</p>
    </article>
  `;
}

function renderPaperGateBoard(id, data) {
  const paper = data.snapshots.paper_gate || {};
  const pooled = data.snapshots.pooled_prediction || {};
  const cards = [
    {
      label: "Strict decision",
      value: gateStatusText(paper.strict_decision_support_allowed),
      note: labelValue(paper.strict_gate_status || pooled.decision_block_reasons),
      tone: isTruthy(paper.strict_decision_support_allowed) ? "good" : "warn",
    },
    {
      label: "Paper decision",
      value: gateStatusText(paper.paper_decision_support_allowed),
      note: explainReasonList(paper.paper_gate_block_reasons || paper.paper_gate_status, 2),
      tone: isTruthy(paper.paper_decision_support_allowed) ? "good" : "warn",
    },
    {
      label: "Latest signal",
      value: gateStatusText(paper.latest_trade_ready),
      note: `${explainReasonList(paper.paper_latest_block_reasons || "PASS", 1)} · stop ${fmtMaybePct(paper.latest_stop_hit_20d, 1)}`,
      tone: isTruthy(paper.latest_trade_ready) ? "good" : "warn",
    },
    {
      label: "Pooled calibration",
      value: `ECE ${fmtNumber(paper.pooled_ece_for_paper, 3)}`,
      note: `Brier ${fmtMaybePct(paper.pooled_brier_improvement_pct_for_paper, 2)} · selected ${fmtNumber(paper.paper_selected_oos_event_count, 0)}`,
      tone: toNumber(paper.pooled_ece_for_paper) !== null && toNumber(paper.pooled_ece_for_paper) <= 0.12 ? "good" : "warn",
    },
    {
      label: "Economic uplift",
      value: fmtPct(paper.paper_selected_minus_rule_ci_lower_pct, 2, false),
      note: `fold 최소 선택 ${fmtNumber(paper.paper_min_selected_events_per_oof_fold, 0)}`,
      tone: toNumber(paper.paper_selected_minus_rule_ci_lower_pct) !== null && toNumber(paper.paper_selected_minus_rule_ci_lower_pct) > -0.5 ? "good" : "warn",
    },
    {
      label: "Live",
      value: labelValue(paper.live_trading_status || "DISABLED_BY_DESIGN"),
      note: "실거래 상태는 게이트와 무관하게 잠금",
      tone: "neutral",
    },
  ];
  const container = $(id);
  if (container) container.innerHTML = cards.map(sampleCardHtml).join("");
}

function riskStatusFromSnapshot(data) {
  const risk = data.snapshots.risk || {};
  const state = String(risk.risk_state || "").toUpperCase();
  if (state === "NO_NEW_RISK" || state === "UNKNOWN") return "BLOCKED";
  return "PASS";
}

function newsGateStatus(latestNews = {}) {
  const coverage = String(latestNews.news_coverage_status || "").toUpperCase();
  if (!coverage) return "MISSING";
  if (coverage.includes("FAILED") || coverage.includes("EMPTY")) return "WARN";
  return "PASS";
}

function renderDecisionBoard(id, data) {
  const decision = data.snapshots.decision || {};
  const risk = data.snapshots.risk || {};
  const prediction = data.snapshots.prediction || {};
  const pooled = data.snapshots.pooled_prediction || {};
  const system = data.snapshots.system || {};
  const modelGate = data.snapshots.model_gate || {};
  const health = data.snapshots.daily_health || {};
  const dataQuality = data.snapshots.data_quality || {};
  const latestNews = data.snapshots.latest_news || {};
  const best = bestBacktestStrategy(data.tables.backtest_summary || []);
  const walk = walkForwardSummary(data.tables.validation_walk_forward || [], best.strategy_id);
  const state = overviewDecisionState(decision, prediction, system);
  const cards = [
    {
      label: "1. 매매 판단",
      value: state.title,
      note: `${state.action} · ${state.trigger}`,
      tone: state.tone,
      items: [
        ["기준일", shortDate(decision.date || prediction.prediction_asof_date)],
        ["알고리즘 점수", `${fmtNumber(decision.score_price_algo_total, 1)} / 75`],
        ["엄격 신호", labelValue(decision.strict_signal_stage || decision.entry_trigger)],
        ["연구 신호", labelValue(decision.research_signal_stage || "NO_RESEARCH_SIGNAL")],
      ],
    },
    {
      label: "2. 리스크",
      value: fmtMaybePct(snapshotWeightValue(risk, ["final_recommended_max_weight", "final_recommended_max_weight_pct"]), 2),
      note: "현재 살 수 있는 최대 비중",
      tone: gateTone(riskStatusFromSnapshot(data)),
      items: [
        ["위험 상태", labelValue(risk.risk_state)],
        ["손절가", fmtCurrency(risk.stop_price_2atr)],
        ["손절까지", fmtMaybePct(risk.risk_pct_2atr, 2)],
      ],
    },
    {
      label: "3. 예측",
      value: statusByBool(system.prediction_decision_support, "판단 가능", "참고만"),
      note: labelValue(prediction.prediction_use_status),
      tone: isTruthy(system.prediction_decision_support) ? "good" : "warn",
      items: [
        ["모델 게이트", labelValue(modelGate.model_gate_status)],
        ["통합 확률", fmtMaybePct(pooled.p_success_20d, 1)],
        ["손절 가능성", fmtMaybePct(pooled.p_stop_hit_20d, 1)],
      ],
    },
    {
      label: "4. 뉴스 원인",
      value: labelValue(latestNews.news_match_confidence || "NO_MATCH"),
      note: labelValue(latestNews.news_primary_cause_type || "NO_HIGH_CONFIDENCE_NEWS"),
      tone: newsTone(latestNews),
      items: [
        ["3일 뉴스", fmtNumber(latestNews.news_event_count_3d, 0)],
        ["매칭 점수", fmtNumber(latestNews.news_match_confidence_score, 1)],
        ["뉴스 감점", fmtNumber(latestNews.news_penalty_event, 1)],
      ],
    },
    {
      label: "5. 시스템",
      value: labelValue(health.daily_health_status || system.system_state),
      note: labelValue(health.failed_or_blocked_components || system.system_state),
      tone: statusTone(health.daily_health_status || system.system_state),
      items: [
        ["데이터", labelValue(dataQuality.data_quality_status)],
        ["준비도", `${fmtNumber(system.system_readiness_score, 0)} / 100`],
        ["실거래", labelValue(system.live_trading_status)],
      ],
    },
    {
      label: "6. 검증",
      value: labelValue(best.strategy_id || "없음"),
      note: `CAGR ${fmtMaybePct(best.cagr_pct, 1)} · MDD ${fmtMaybePct(best.max_drawdown_pct, 1)}`,
      tone: best.strategy_id ? "neutral" : "warn",
      items: [
        ["손실 대비 수익", fmtNumber(best.profit_factor, 2)],
        ["WF 중앙 CAGR", fmtMaybePct(walk.median_cagr_pct, 1)],
        ["WF 양수 비율", fmtMaybePct(walk.positive_rate_pct, 1)],
      ],
    },
    {
      label: "7. 시장 상태",
      value: labelValue(decision.trend_regime),
      note: `${labelValue(decision.vol_regime)} · 고점 대비 ${fmtMaybePct(decision.drawdown_from_ath, 1)}`,
      tone: String(decision.vol_regime || "").toLowerCase().includes("high") ? "warn" : "neutral",
      items: [
        ["SMH 대비 60일", fmtMaybePct(decision.relative_return_vs_smh_60d, 1)],
        ["SPY 베타", fmtNumber(decision.beta_vs_spy_252d, 2)],
        ["RSI14", fmtNumber(decision.rsi_14, 1)],
      ],
    },
  ];
  $(id).innerHTML = cards.map(overviewCardHtml).join("");
}

function renderDisplayPolicy(id, data) {
  const decision = data.snapshots.decision || {};
  const risk = data.snapshots.risk || {};
  const planRows = data.tables.trading_plan || [];
  const cards = [
    ["현재 종가", planValue(planRows, "가격", "현재 종가") || fmtCurrency(decision.close), `일간 ${fmtMaybePct(decision.close_change_pct, 1)}`],
    ["20일 돌파", planValue(planRows, "진입", "20일 고점 돌파 기준가"), "종가 마감 기준"],
    ["60일 돌파", planValue(planRows, "진입", "60일 고점 돌파 기준가"), "강한 추세 확인"],
    ["50일선 눌림", planValue(planRows, "진입", "50일선 눌림목 구간"), "반등 확인 필요"],
    ["2ATR 손절", planValue(planRows, "리스크", "2ATR 손절가") || fmtCurrency(risk.stop_price_2atr), `손절폭 ${fmtMaybePct(risk.risk_pct_2atr, 2)}`],
    ["목표가", `${planValue(planRows, "목표", "1차 목표가 2R")} / ${planValue(planRows, "목표", "2차 목표가 3R")}`, "현재가 기준 R 배수"],
  ];
  $(id).innerHTML = cards
    .map(([label, value, note]) => `<article class="display-item"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><p>${escapeHtml(note)}</p></article>`)
    .join("");
}

function overviewCardHtml(card) {
  return `
    <article class="decision-item overview-card ${card.tone || "neutral"}">
      <header>
        <span>${escapeHtml(card.label)}</span>
        <strong>${escapeHtml(card.value)}</strong>
        <p>${escapeHtml(card.note)}</p>
      </header>
      <dl>
        ${(card.items || [])
          .map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`)
          .join("")}
      </dl>
    </article>
  `;
}

function bestBacktestStrategy(rows) {
  return (rows || [])
    .filter((row) => toNumber(row.cagr_pct) !== null)
    .slice()
    .sort((a, b) => toNumber(b.cagr_pct) - toNumber(a.cagr_pct))[0] || {};
}

function walkForwardSummary(rows, strategyId = "") {
  const sourceRows = strategyId ? (rows || []).filter((row) => String(row.strategy_id) === String(strategyId)) : rows || [];
  const cagrValues = sourceRows.map((row) => toNumber(row.test_cagr_pct)).filter((value) => value !== null).sort((a, b) => a - b);
  const positiveValues = sourceRows.map((row) => isTruthy(row.test_positive));
  const mid = Math.floor(cagrValues.length / 2);
  const median = !cagrValues.length ? null : cagrValues.length % 2 ? cagrValues[mid] : (cagrValues[mid - 1] + cagrValues[mid]) / 2;
  const positiveRate = positiveValues.length ? (positiveValues.filter(Boolean).length / positiveValues.length) * 100 : null;
  return {
    median_cagr_pct: median,
    positive_rate_pct: positiveRate,
  };
}

function planValue(rows, section, item) {
  const row = (rows || []).find((candidate) => String(candidate.section) === section && String(candidate.item) === item);
  return row?.value ? labelValue(row.value) : "";
}

function renderBlockReasons(id, data) {
  const rootCauses = data.tables.model_gate_root_causes || [];
  if (rootCauses.length) {
    const topRows = rootCauses
      .filter((row) => String(row.root_cause || "").toUpperCase() !== "PASS")
      .slice(0, 6);
    if (!topRows.length) {
      $(id).innerHTML = `<div class="preview-empty">차단 사유 없음</div>`;
      return;
    }
    $(id).innerHTML = topRows
      .map((row) => {
        const label = labelValue(row.root_cause);
        const action = row.recommended_action ? translateReportText(row.recommended_action) : explainReason(row.root_cause);
        const scope = [labelValue(row.category), `${fmtNumber(row.failed_gate_count, 0)}개 기준`, `우선순위 ${fmtNumber(row.priority, 0)}`].filter(Boolean).join(" · ");
        const examples = [labelValue(row.gate_groups), labelValue(row.candidate_scopes)].filter((value) => value && value !== "없음").join(" / ");
        return `<article class="block-item"><div><span>${escapeHtml(scope)}</span><strong>${escapeHtml(label)}</strong></div><p>${escapeHtml(action)}</p>${examples ? `<small>${escapeHtml(compactText(examples, 140))}</small>` : ""}</article>`;
      })
      .join("");
    return;
  }
  const systemBlocks = data.tables.system_block_reasons || [];
  const modelGateRows = (data.tables.model_gate_audit || []).filter((row) => !isTruthy(row.passed));
  const reasonMap = new Map();
  const addReason = (source, rawReason, details = "") => {
    for (const item of reasonCodes(rawReason)) {
      const key = item.code;
      if (!reasonMap.has(key)) {
        reasonMap.set(key, { code: key, sources: new Set(), details: new Set(), raw: item.raw });
      }
      const target = reasonMap.get(key);
      target.sources.add(source);
      if (details) target.details.add(details);
    }
  };
  systemBlocks
    .filter((row) => !["live_trading"].includes(String(row.component)))
    .forEach((row) => addReason(labelValue(row.component), row.block_reasons, row.details));
  modelGateRows.forEach((row) => addReason(labelValue(row.gate_group), row.block_reason, `${labelValue(row.gate)} · ${labelValue(row.candidate_scope)} ${row.horizon_days || ""}`));
  const rows = Array.from(reasonMap.values()).sort((a, b) => String(a.code).localeCompare(String(b.code), "ko"));
  if (!rows.length) {
    $(id).innerHTML = `<div class="preview-empty">차단 사유 없음</div>`;
    return;
  }
  $(id).innerHTML = rows
    .map((row) => {
      const label = labelValue(row.code);
      const explanation = explainReason(row.code);
      const sources = Array.from(row.sources).join(" / ");
      const examples = Array.from(row.details).slice(0, 2).join(" · ");
      return `<article class="block-item"><div><span>${escapeHtml(sources)}</span><strong>${escapeHtml(label)}</strong></div><p>${escapeHtml(explanation)}</p>${examples ? `<small>${escapeHtml(examples)}</small>` : ""}</article>`;
    })
    .join("");
}

function renderDecision(data) {
  renderPageSynthesis("decisionSynthesis", deriveDecisionSynthesis(data));
  const d = data.snapshots.decision || {};
  renderFacts("decisionFacts", [
    ["기준일", shortDate(d.date)],
    ["행동", labelValue(d.trade_action)],
    ["엄격 신호", labelValue(d.strict_signal_stage)],
    ["연구/Paper 신호", labelValue(d.research_signal_stage)],
    ["연구/Paper 행동", labelValue(d.research_signal_action)],
    ["연구 신호 점수", fmtNumber(d.research_signal_score, 1)],
    ["매수 신호", labelValue(d.entry_trigger)],
    ["매수 점수", fmtNumber(d.score_price_algo_total, 1)],
    ["종가", fmtCurrency(d.close)],
    ["손절 기준가", fmtCurrency(d.atr_stop_2x)],
    ["2차 목표가", fmtCurrency(d.take_profit_2R)],
    ["계좌 위험 0.5% 기준 매수 비중", fmtMaybePct(d.position_weight_if_0_5pct_account_risk, 2)],
  ]);
  renderTable("tradingPlanTable", decisionPlanRows(data), ["section", "item", "value", "unit", "notes"], 220);
  renderSignalChart("signalChart", data.series.signals || []);
}

function decisionPlanRows(data) {
  const rows = data.tables.trading_plan || [];
  const baseRows = rows.filter((row) => String(row.section) !== "예측");
  const predictionRowsForPlan = decisionPredictionPlanRows(data);
  return [...predictionRowsForPlan, ...baseRows];
}

function decisionPredictionPlanRows(data) {
  const p = data.snapshots.prediction || {};
  const pooled = data.snapshots.pooled_prediction || {};
  const latestNews = data.snapshots.latest_news || {};
  const latestFeatures = data.snapshots.latest_prediction_features || {};
  const source = predictionDisplaySource(p, pooled);
  const rows = [
    planRow("예측", "표시 기준", "막혀도 모두 표시", "", "매매에 바로 못 쓰는 예측도 참고용으로 숨기지 않음"),
    planRow("예측", "대표 예측", source.scopeLabel, "", source.successNote),
    planRow("예측", "최신 신호 종류", labelValue(p.latest_candidate_scope), "", "현재 신호가 매수 후보인지 관찰용인지 표시"),
    planRow("예측", "매수 기준 상태", labelValue(snapshotValue(p, ["latest_entry_gate_status", "prediction_entry_gate_status"])), "", "기본 규칙이 매수를 허용했는지 표시"),
    planRow("예측", "예측 사용 상태", labelValue(p.prediction_use_status), "", "예측이 매매 판단용인지 참고용인지 표시"),
    planRow("예측", "실제 판단에 쓴 예측", labelValue(p.prediction_scope_used), "", "최종 판단에 직접 반영된 예측 종류"),
    planRow("예측", "대표 20일 성공 확률", fmtMaybePct(source.pSuccess20, 1), "%", source.modelLabel),
    planRow("예측", "대표 20일 손절 회피", fmtMaybePct(source.pStopSurvival20, 1), "%", source.sampleNote),
    planRow("뉴스 원인", "대표 원인", labelValue(latestNews.news_primary_cause_type || "NO_HIGH_CONFIDENCE_NEWS"), "", latestNews.news_cause_summary || "HIGH 신뢰도 원인만 대표 원인으로 사용"),
    planRow("뉴스 원인", "매칭 신뢰도", labelValue(latestNews.news_match_confidence || "NO_MATCH"), "", `점수 ${fmtNumber(latestNews.news_match_confidence_score, 1)} · 출처 ${fmtNumber(latestNews.news_source_count, 0)}개`),
    planRow("뉴스 원인", "위험 감점", fmtNumber(latestFeatures.news_penalty_event ?? latestNews.news_penalty_event, 1), "점", "고신뢰 부정 이벤트만 룰 점수에서 보수적으로 차감"),
    planRow("뉴스 원인", "같은 원인 20일 표본", fmtNumber(latestFeatures.hist_news_category_count_20d, 0), "건", `성공률 ${fmtMaybePct(latestFeatures.hist_news_category_success_rate_20d, 1)} · 평균 ${fmtMaybePct(latestFeatures.hist_news_category_mean_return_20d, 1)}`),
  ];

  rows.push(
    planRow("예측", "여러 종목 예측 참고 가능", statusByBool(pooled.decision_support_allowed, "가능", "기준 미달"), "", labelValue(pooled.decision_block_reasons)),
    planRow("예측", "여러 종목 모델 상태", statusByBool(pooled.model_quality_pass, "사용 가능", "기준 미달"), "", labelValue(pooled.model_quality_block_reasons)),
    planRow("예측", "TSMC 맞춤 확률 조정", labelValue(pooled.tsm_calibration_status), "", labelValue(pooled.tsm_calibration_block_reasons))
  );
  return rows;
}

function predictionPlanMetricRows(row) {
  const prefix = `${row.prediction_scope_label} ${row.horizon_days}일`;
  return [
    planRow("예측", `${prefix} 상태`, row.prediction_status, "", labelValue(row.quality_block_reasons)),
    planRow("예측", `${prefix} 예측 방식`, labelValue(row.model_name), "", "이 값 계산에 사용한 예측 방식"),
    planRow("예측", `${prefix} 오를 가능성`, fmtMaybePct(row.p_success, 1), "%", "손절을 피하고 비용을 빼도 수익으로 끝날 가능성"),
    planRow("예측", `${prefix} 손절 안 날 가능성`, fmtMaybePct(row.p_stop_survival, 1), "%", "손절가에 먼저 닿지 않을 가능성"),
    planRow("예측", `${prefix} 손절 확률`, fmtMaybePct(row.p_stop_hit, 1), "%", "1 - 손절 회피확률"),
    planRow("예측", `${prefix} 수익으로 끝날 가능성`, fmtMaybePct(row.p_positive_given_survival, 1), "%", "손절을 피한 뒤 플러스로 끝날 가능성"),
    planRow("예측", `${prefix} 1차 목표 도달 가능성`, fmtMaybePct(row.p_hit_1r, 1), "%", "손절보다 1차 목표가에 먼저 닿을 가능성"),
    planRow("예측", `${prefix} 2차 목표 도달 가능성`, fmtMaybePct(row.p_hit_2r, 1), "%", "손절보다 2차 목표가에 먼저 닿을 가능성"),
    planRow("예측", `${prefix} 위험 대비 기대수익`, fmtPlanNumber(row.expected_r, 3), "R", "손절폭을 1로 봤을 때 기대값"),
    planRow("예측", `${prefix} 기대 순수익률`, fmtPlanPctPoint(row.expected_net_return_pct, 1), "%", "비용 차감 후 기대 수익률"),
    planRow("예측", `${prefix} 선택 기준값`, fmtMaybePct(row.threshold, 1), "%", "좋은 신호로 고르는 기준"),
    planRow("예측", `${prefix} 사례 수`, `${fmtPlanNumber(row.oos_event_count, 0)} / ${fmtPlanNumber(row.effective_oos_event_count, 0)} / ${fmtPlanNumber(row.selected_oos_event_count, 0)}`, "건", "전체 테스트 / 실제 사용 / 선택된 사례"),
  ];
}

function planRow(section, item, value, unit = "", notes = "") {
  return { section, item, value, unit, notes };
}

function fmtPlanNumber(value, digits = 2) {
  const n = toNumber(value);
  return n === null ? "없음" : fmtNumber(n, digits);
}

function fmtPlanPctPoint(value, digits = 1) {
  const n = toNumber(value);
  return n === null ? "없음" : fmtPct(n, digits, false);
}

function renderMarket(data) {
  renderPageSynthesis("marketSynthesis", deriveMarketSynthesis(data));
  renderPriceLine("priceChart", data.series.price || []);
  lineChart($("drawdownChart"), data.series.price || [], [{ key: "drawdown_from_ath", label: "고점 대비 낙폭", color: colors.red }], {
    yFormat: (v) => fmtPct(v, 0, true),
    fillZero: true,
  });
  lineChart(
    $("volChart"),
    data.series.price || [],
    [
      { key: "vol_20d_ann", label: "최근 20일 흔들림", color: colors.blue },
      { key: "vol_63d_ann", label: "최근 63일 흔들림", color: colors.teal },
      { key: "atr_14_pct", label: "평균 하루 변동폭", color: colors.amber },
    ],
    { yFormat: (v) => fmtMaybePct(v, 0) }
  );
  renderImageGallery(data.images || []);
}

function renderDataView(data) {
  renderPageSynthesis("dataSynthesis", deriveDataSynthesis(data));
  renderTimeframeCards("timeframeCards", data);
  renderPriceSummary("priceSummaryFacts", data);
  renderPriceLine("hourlyPriceChart", data.series.hourly || []);
  renderPriceLine("minutePriceChart", data.series.minute || []);
  const audits = [
    ...(data.tables.hourly_source_audit || []).map((row) => ({ ...row, dataset: "시간봉" })),
    ...(data.tables.minute_source_audit || []).map((row) => ({ ...row, dataset: "분봉" })),
  ];
  renderTable("sourceAuditTable", audits, ["dataset", "provider", "interval", "auth_status", "request_status", "supports_complete_requested_10y", "actual_start", "actual_end", "rows", "evidence"], 80);
  horizontalBarChart(
    $("yearlyPriceBars"),
    data.tables.yearly_price || [],
    (row) => row.year,
    (row) => toNumber(row.year_return_pct),
    { valueFormat: (v) => fmtPct(v, 1), color: colors.green, limit: 20 }
  );
  renderTable("yearlyPriceTable", data.tables.yearly_price || [], ["year", "year_return_pct", "ann_vol_pct", "max_drawdown_pct", "avg_intraday_range_pct", "avg_atr14_pct", "max_up_day_pct", "max_down_day_pct", "up_days", "down_days"], 40);
  renderTable("monthlyPriceTable", data.tables.monthly_price || [], ["year", "month", "month_return_pct", "ann_vol_pct", "avg_abs_daily_pct", "avg_intraday_range_pct", "high_vol_days", "event_shock_days"], 80);
  horizontalBarChart(
    $("eventImpactBars"),
    data.tables.event_analysis || [],
    (row) => row.event_name,
    (row) => toNumber(row.event_impact_score),
    { valueFormat: (v) => fmtNumber(v, 2), color: colors.teal, limit: 20 }
  );
  renderTable("eventAnalysisTable", data.tables.event_analysis || [], ["event_date", "event_name", "event_type", "return_on_event_day_pct", "post_20d_return_pct", "post_60d_return_pct", "event_impact_score", "event_direction_class"], 50);
  renderTable("extremeMovesTable", data.tables.extreme_moves || [], ["extreme_type", "date", "close_change_pct", "open_gap_pct", "intraday_range_pct_prev_close", "volume_ratio_20", "atr_14_pct", "vol_20d_ann", "trend_regime", "entry_trigger", "trade_action", "score_price_algo_total"], 120);
  renderTable("newsCauseTable", data.tables.news_matches || [], ["date", "close_change_pct", "open_gap_pct", "news_match_confidence", "news_primary_cause_type", "news_cause_summary", "news_source_count", "news_primary_source_url"], 120);
  renderTable("newsCauseForwardTable", data.tables.news_cause_forward || [], ["rule", "horizon_days", "n_signals", "mean_fwd_return_pct", "median_fwd_return_pct", "win_rate_pct", "worst_fwd_return_pct", "best_fwd_return_pct"], 120);
  renderTable("ruleForwardTable", data.tables.rule_forward || [], ["rule", "horizon_days", "n_signals", "mean_fwd_return_pct", "median_fwd_return_pct", "win_rate_pct", "p10_fwd_return_pct", "p90_fwd_return_pct"], 140);
  renderTable("regimeForwardTable", data.tables.regime_forward || [], ["rule", "horizon_days", "n_signals", "mean_fwd_return_pct", "median_fwd_return_pct", "win_rate_pct", "worst_fwd_return_pct", "best_fwd_return_pct"], 100);
}

function renderNews(data) {
  renderPageSynthesis("newsSynthesis", deriveNewsSynthesis(data));
  const latestNews = data.snapshots.latest_news || {};
  const latestFeatures = data.snapshots.latest_prediction_features || {};
  const penaltySummary = data.snapshots.news_penalty_summary || {};
  const focusSourceRows = (data.tables.news_matches_focus || []).length ? data.tables.news_matches_focus : data.tables.news_matches || [];
  const focusRows = latestFirst(focusSourceRows);
  const highRows = latestFirst(data.tables.news_high_matches || []);
  const penaltyRows = latestFirst(data.tables.news_penalty_events || []);
  const featureRows = latestFirst(data.tables.news_feature_matrix || []);
  const clusterRows = latestFirst(data.tables.news_clusters || []);
  const dailyRows = latestFirst(data.tables.news_daily || []);
  const contractRows = data.tables.news_feature_contract || [];

  renderNewsHero(data);
  renderFacts("newsFeatureFacts", [
    ["기준일", shortDate(latestNews.date || latestFeatures.date)],
    ["대표 원인", labelValue(latestNews.news_primary_cause_type || "NO_HIGH_CONFIDENCE_NEWS")],
    ["매칭 신뢰도", labelValue(latestNews.news_match_confidence || "NO_MATCH")],
    ["매칭 점수", fmtNumber(latestNews.news_match_confidence_score, 1)],
    ["뉴스 위험 감점", fmtNumber(latestFeatures.news_penalty_event ?? latestNews.news_penalty_event, 1)],
    ["같은 원인 20일 성공률", fmtMaybePct(latestFeatures.hist_news_category_success_rate_20d, 1)],
    ["같은 원인 20일 평균", fmtMaybePct(latestFeatures.hist_news_category_mean_return_20d, 1)],
    ["수집 상태", labelValue(latestNews.news_coverage_status || "DIRECT_WEB_EMPTY")],
  ]);
  renderNewsPipeline("newsPipeline", data);
  renderFacts("newsPenaltyFacts", [
    ["전체 감점일", `${fmtNumber(penaltySummary.penalty_days, 0)} / ${fmtNumber(penaltySummary.all_days, 0)}`],
    ["최근 120일 감점일", fmtNumber(penaltySummary.recent_120_penalty_days, 0)],
    ["누적 감점", fmtNumber(penaltySummary.total_penalty, 1)],
    ["최대 단일 감점", fmtNumber(penaltySummary.max_penalty, 1)],
    ["최근 감점일", shortDate(penaltySummary.latest_penalty_date)],
    ["최근 감점 원인", labelValue(penaltySummary.latest_penalty_cause || "없음")],
  ]);
  renderTable("newsPenaltyTable", penaltyRows, ["date", "news_primary_cause_type", "news_match_confidence", "news_match_confidence_score", "news_penalty_event", "news_cause_summary", "news_primary_source_url"], 100);
  renderTable("newsFocusTable", focusRows, ["date", "close_change_pct", "open_gap_pct", "volume_ratio_20", "algo_event_shock_day", "news_match_confidence", "news_primary_cause_type", "news_cause_summary", "news_match_confidence_score", "news_source_count", "news_penalty_event", "news_primary_source_url"], 220);
  renderTable("newsForwardTable", data.tables.news_cause_forward || [], ["rule", "horizon_days", "n_signals", "mean_fwd_return_pct", "median_fwd_return_pct", "win_rate_pct", "worst_fwd_return_pct", "best_fwd_return_pct"], 160);
  horizontalBarChart(
    $("newsCauseBars"),
    newsCauseDistribution(highRows),
    (row) => row.cause_type,
    (row) => row.high_count,
    { valueFormat: (v) => fmtNumber(v, 0), color: colors.teal, limit: 16 }
  );
  renderTable("newsFeatureMatrixTable", featureRows, ["date", "news_event_count_1d", "news_event_count_3d", "news_sentiment_score_1d", "news_primary_cause_type", "news_primary_cluster_id", "news_match_confidence", "news_match_confidence_score", "news_coverage_status", "news_source_count", "news_penalty_event", "hist_news_category_count_20d", "hist_news_category_success_rate_20d", "hist_news_category_mean_return_20d", "score_price_algo_total", "trade_action", "entry_trigger"], 160);
  renderTable("newsClusterTable", clusterRows, ["cluster_id", "representative_title", "cause_type", "start_date", "end_date", "article_count", "source_count", "coverage_status", "source_urls"], 180);
  renderTable("newsFeatureContractTable", contractRows, ["column", "role", "available_at", "leakage_policy", "feature_group"], 120);
  renderTable("newsDailyTable", dailyRows, ["date", "news_event_count_1d", "news_event_count_3d", "news_sentiment_score_1d", "news_primary_cause_type", "news_match_confidence", "news_match_confidence_score", "news_coverage_status", "news_source_count", "news_penalty_event", "news_cause_summary", "news_primary_source_url"], 180);
}

function renderNewsHero(data) {
  const latestNews = data.snapshots.latest_news || {};
  const latestFeatures = data.snapshots.latest_prediction_features || {};
  const penaltySummary = data.snapshots.news_penalty_summary || {};
  const highRows = data.tables.news_high_matches || [];
  const focusRows = data.tables.news_matches_focus || [];
  const contractRows = data.tables.news_feature_contract || [];
  const sourceCount = toNumber(latestNews.news_source_count);
  const penalty = toNumber(latestFeatures.news_penalty_event ?? latestNews.news_penalty_event) || 0;
  const cards = [
    ["수집 상태", labelValue(latestNews.news_coverage_status || "DIRECT_WEB_EMPTY"), `${shortDate(latestNews.date || latestFeatures.date)} 기준 · 출처 ${sourceCount === null ? "없음" : fmtNumber(sourceCount, 0)}개`, true],
    ["최신 원인", labelValue(latestNews.news_primary_cause_type || "NO_HIGH_CONFIDENCE_NEWS"), `${labelValue(latestNews.news_match_confidence || "NO_MATCH")} · 점수 ${fmtNumber(latestNews.news_match_confidence_score, 1)}`, false],
    ["카테고리 학습", `${fmtNumber(latestFeatures.hist_news_category_count_20d, 0)}건`, `20일 성공률 ${fmtMaybePct(latestFeatures.hist_news_category_success_rate_20d, 1)}`, false],
    ["룰·예측 반영", penalty > 0 ? `-${fmtNumber(penalty, 1)}점` : "최신 감점 없음", `전체 감점 ${fmtNumber(penaltySummary.penalty_days, 0)}일 · 최근 120일 ${fmtNumber(penaltySummary.recent_120_penalty_days, 0)}일`, false],
  ];
  $("newsHero").innerHTML = cards
    .map(([label, value, note, primary]) => `<article class="prediction-card${primary ? " primary" : ""}"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><p>${escapeHtml(note)}</p></article>`)
    .join("");
}

function renderNewsPipeline(id, data) {
  const latestNews = data.snapshots.latest_news || {};
  const latestFeatures = data.snapshots.latest_prediction_features || {};
  const highRows = data.tables.news_high_matches || [];
  const focusRows = data.tables.news_matches_focus || [];
  const clusterRows = data.tables.news_clusters || [];
  const contractRows = data.tables.news_feature_contract || [];
  const penalty = toNumber(latestFeatures.news_penalty_event ?? latestNews.news_penalty_event) || 0;
  const steps = [
    ["1", "직접 수집", labelValue(latestNews.news_coverage_status || "DIRECT_WEB_EMPTY"), `당일 ${fmtNumber(latestNews.news_event_count_1d, 0)}건 · 3일 ${fmtNumber(latestNews.news_event_count_3d, 0)}건`],
    ["2", "사건 구조화", `${fmtNumber(clusterRows.length, 0)}개 묶음 표시`, "카테고리별로 기사 제목과 출처를 묶음"],
    ["3", "가격 매칭", `HIGH ${fmtNumber(highRows.length, 0)}건`, `${fmtNumber(focusRows.length, 0)}개 급등락·원인 후보 표시`],
    ["4", "룰·예측 반영", penalty > 0 ? `-${fmtNumber(penalty, 1)}점` : "감점 없음", `${fmtNumber(contractRows.length, 0)}개 피처가 신호 시점 기준으로 사용`],
  ];
  $(id).innerHTML = steps
    .map(([step, title, value, note]) => `<article class="pipeline-step"><span>${escapeHtml(step)}</span><div><strong>${escapeHtml(title)}</strong><em>${escapeHtml(value)}</em><p>${escapeHtml(note)}</p></div></article>`)
    .join("");
}

function renderNewsConnectionBoard(id, data) {
  const container = $(id);
  if (!container) return;
  const latestNews = data.snapshots.latest_news || {};
  const latestFeatures = data.snapshots.latest_prediction_features || {};
  const penaltySummary = data.snapshots.news_penalty_summary || {};
  const highRows = data.tables.news_high_matches || [];
  const focusRows = data.tables.news_matches_focus || [];
  const contractRows = data.tables.news_feature_contract || [];
  const penalty = toNumber(latestFeatures.news_penalty_event ?? latestNews.news_penalty_event) || 0;
  const cards = [
    {
      label: "최신 매칭",
      value: labelValue(latestNews.news_match_confidence || "NO_MATCH"),
      note: `${labelValue(latestNews.news_primary_cause_type || "NO_HIGH_CONFIDENCE_NEWS")} · ${shortDate(latestNews.date || latestFeatures.date)}`,
      tone: newsTone(latestNews),
    },
    {
      label: "원인 카테고리",
      value: labelValue(latestNews.news_primary_cause_type || "NO_HIGH_CONFIDENCE_NEWS"),
      note: latestNews.news_cause_summary || "HIGH 신뢰도만 대표 원인으로 확정",
      tone: String(latestNews.news_match_confidence || "").toUpperCase() === "HIGH" ? "good" : "neutral",
    },
    {
      label: "같은 카테고리 계산",
      value: `${fmtNumber(latestFeatures.hist_news_category_count_20d, 0)}건`,
      note: `20일 성공률 ${fmtMaybePct(latestFeatures.hist_news_category_success_rate_20d, 1)} · 평균 ${fmtMaybePct(latestFeatures.hist_news_category_mean_return_20d, 1)}`,
      tone: toNumber(latestFeatures.hist_news_category_count_20d) > 0 ? "good" : "warn",
    },
    {
      label: "룰 점수 영향",
      value: penalty > 0 ? `-${fmtNumber(penalty, 1)}점` : "감점 없음",
      note: `고신뢰 부정 이벤트만 차감 · 전체 ${fmtNumber(penaltySummary.penalty_days, 0)}일 / 최근 120일 ${fmtNumber(penaltySummary.recent_120_penalty_days, 0)}일`,
      tone: penalty > 0 ? "warn" : "neutral",
    },
    {
      label: "예측 피처 계약",
      value: `${fmtNumber(contractRows.length, 0)}개`,
      note: "신호 시점 이전에 알 수 있는 값만 사용",
      tone: contractRows.length ? "good" : "warn",
    },
    {
      label: "과거 매칭 표본",
      value: `HIGH ${fmtNumber(highRows.length, 0)}건`,
      note: `${fmtNumber(focusRows.length, 0)}개 급등락·가능 원인 화면 표시`,
      tone: highRows.length ? "neutral" : "warn",
    },
  ];
  container.innerHTML = cards
    .map((card) => `<article class="news-connection-card ${card.tone}"><span>${escapeHtml(card.label)}</span><strong>${escapeHtml(card.value)}</strong><p title="${escapeHtml(card.note)}">${escapeHtml(compactText(card.note, 110))}</p></article>`)
    .join("");
}

function latestFirst(rows) {
  return (rows || []).slice().reverse();
}

function newsCauseDistribution(rows) {
  const grouped = new Map();
  for (const row of rows || []) {
    const cause = row.news_primary_cause_type || row.cause_type || "other";
    if (!cause || String(cause).toUpperCase() === "NO_HIGH_CONFIDENCE_NEWS") continue;
    const key = labelValue(cause);
    const item = grouped.get(key) || { cause_type: key, high_count: 0, source_count: 0 };
    item.high_count += 1;
    item.source_count += toNumber(row.news_source_count || row.source_count) || 0;
    grouped.set(key, item);
  }
  return Array.from(grouped.values()).sort((a, b) => b.high_count - a.high_count);
}

function newsTone(row = {}) {
  const confidence = String(row.news_match_confidence || "").toUpperCase();
  const penalty = toNumber(row.news_penalty_event) || 0;
  if (penalty > 0) return "warn";
  if (confidence === "HIGH") return "good";
  if (confidence === "MEDIUM") return "neutral";
  return "neutral";
}

function renderPrediction(data) {
  renderPageSynthesis("predictionSynthesis", derivePredictionSynthesis(data));
  const p = data.snapshots.prediction || {};
  const pooled = data.snapshots.pooled_prediction || {};
  renderPredictionHero(p, pooled);
  renderPredictionNotice(p, pooled);
  renderPaperGateBoard("paperGateBoard", data);
  renderNewsConnectionBoard("predictionNewsBoard", data);
  renderAllPredictionTable("predictionAllTable", p, pooled);
  renderStageProbChart("stageProbChart", p, pooled);
  const audit = data.tables.prediction_audit || [];
  horizontalBarChart(
    $("auditBars"),
    audit.slice(0, 14),
    (row) => `${labelValue(row.candidate_scope)} / ${row.horizon_days}일 / ${labelValue(row.model_name)}`,
    (row) => toNumber(row.brier_improvement_pct),
    { valueFormat: (v) => fmtPct(v, 1), color: colors.violet, limit: 14 }
  );
  renderTable(
    "predictionComparisonTable",
    data.tables.prediction_comparison || [],
    ["candidate_scope", "horizon_days", "model_name", "prediction_quality_pass", "oos_event_count", "mean_effective_sample_size", "brier_improvement_pct", "ece", "pr_auc", "expectancy_improvement_pct", "rank_score"],
    160
  );
  renderTable(
    "labelDiagnosticsTable",
    data.tables.prediction_label_diagnostics || [],
    ["horizon_days", "prediction_universe", "labeled_count", "success_rate_pct", "stop_survival_rate_pct", "hit_1r_before_stop_rate_pct", "hit_2r_before_stop_rate_pct", "atr_stop_rate_pct", "mean_net_return_pct"],
    120
  );
  renderCalibrationChart("calibrationChart", data.tables.prediction_calibration || [], p);
  renderTable(
    "featureSelectionTable",
    data.tables.prediction_feature_selection || [],
    ["horizon_days", "candidate_scope", "fold_id", "target_name", "feature", "feature_type", "decision", "reason", "missing_rate"],
    220
  );
  renderTable(
    "candidateScopeTable",
    data.tables.prediction_candidate_scope || [],
    ["horizon_days", "group_type", "group_value", "total_count", "labeled_count", "success_rate_pct", "mean_net_return_pct", "mean_expected_r", "hit_1r_rate_pct", "hit_2r_rate_pct", "profit_factor", "atr_stop_rate_pct"],
    160
  );
  renderTable(
    "thresholdPolicyTable",
    data.tables.prediction_threshold_policy || [],
    ["horizon_days", "candidate_scope", "fold_id", "model_name", "threshold", "threshold_source", "validation_expectancy_pct", "validation_selected_count", "min_validation_trades"],
    160
  );
  renderTable(
    "predictionWalkMetricsTable",
    data.tables.prediction_walk_metrics || [],
    ["horizon_days", "candidate_scope", "fold_id", "model_name", "test_start_date", "test_end_date", "test_event_count", "brier_improvement_pct", "ece", "selected_signal_count", "selected_signal_expectancy_pct", "expectancy_improvement_pct"],
    220
  );
  renderTable(
    "predictionOosTable",
    data.tables.prediction_oos_predictions || [],
    ["horizon_days", "candidate_scope", "fold_id", "model_name", "date", "entry_trigger", "trade_action", "p_success", "p_stop_survival", "threshold", "selected_by_threshold", "label_success", "label_net_return_pct", "label_expected_r", "label_exit_reason"],
    250
  );
  renderTable(
    "calibrationSummaryTable",
    data.tables.prediction_calibration_summary || [],
    ["candidate_scope", "horizon_days", "model_name", "binning", "bin_count", "oos_event_count", "ece", "mean_predicted_probability", "observed_success_rate", "max_abs_calibration_error", "min_bin_n"],
    120
  );
  renderTable("policyAuditTable", data.tables.prediction_policy_audit || [], ["audit_item", "status", "details"], 80);
  renderTable("featureContractTable", data.tables.prediction_feature_contract || [], ["column", "role", "available_at", "leakage_policy", "feature_group"], 260);
  renderTable(
    "foldManifestTable",
    data.tables.prediction_fold_manifest || [],
    ["candidate_scope", "horizon_days", "fold_id", "train_event_count", "validation_event_count", "test_event_count", "embargo_days", "purged_train_count", "status"],
    80
  );
}

function renderModels(data) {
  renderPageSynthesis("modelsSynthesis", deriveModelSynthesis(data));
  const pooled = data.snapshots.pooled_prediction || {};
  const system = data.snapshots.system || {};
  const modelGateAudit = (data.tables.model_gate_audit || []).map((row) => {
    const reasons = reasonCodes(row.block_reason);
    return {
      ...row,
      block_reason_explanation: reasons.length ? reasons.map((item) => explainReason(item.code)).join(" / ") : "통과",
    };
  });
  renderModelOpsHero(pooled, system);
  renderSampleExpansionHero(data);
  renderResearchExpansionHero(data);
  renderSampleGroupChart("sampleGroupChart", data.tables.pooled_sample_audit || []);
  renderTsmLikeFacts("tsmLikeFacts", data);
  renderPaperGateFacts("paperGateFacts", data);
  renderTable(
    "pooledLearningCurveTable",
    data.tables.pooled_learning_curve || [],
    ["universe_size_target", "symbol_count", "model_training_20d_labeled", "trade_ready_20d_labeled", "brier_improvement_pct", "ece", "selected_minus_rule_mean", "selected_minus_rule_paired_ci_lower", "tsm_like_calibration_ece", "selected_fraction_drift", "fold_positive_uplift_count", "status"],
    20
  );
  renderTable(
    "sampleAuditTable",
    groupedSampleAudit(data.tables.pooled_sample_audit || []),
    ["symbol_group", "loaded", "strict_eligible", "trade_ready_20d_labeled", "model_training_20d_labeled", "test_holdout_trade_ready_20d", "tsm_like_effective_n_contribution"],
    20
  );
  renderTable(
    "modelRootCauseTable",
    data.tables.model_gate_root_causes || [],
    ["priority", "root_cause", "failed_gate_count", "category", "gate_groups", "candidate_scopes", "value_min", "value_max", "recommended_action"],
    80
  );
  renderTable(
    "autoResearchBestTable",
    data.tables.auto_research_best_candidates || [],
    ["trial_id", "trial_source", "objective_score", "model_family", "model_name", "candidate_scope", "split", "pbo_penalty", "dsr_penalty", "cpcv_penalty", "feature_set_hash", "label_config_hash"],
    25
  );
  renderTable(
    "cpcvModelDistributionTable",
    data.tables.cpcv_model_distribution || [],
    ["model_name", "candidate_scope", "fold_count", "selected_event_count", "min_selected_events_per_fold", "selected_minus_all_pct_median", "selected_minus_all_pct_q25", "positive_fold_rate_pct", "cpcv_model_median_uplift_pass", "cpcv_model_worst_quartile_pass"],
    80
  );
  renderTable(
    "cpcvStrategyDistributionTable",
    data.tables.cpcv_strategy_distribution || [],
    ["strategy_id", "cpcv_path_count", "test_uplift_pct_median", "test_uplift_pct_q25", "positive_path_rate_pct", "worst_test_drawdown_pct", "cpcv_median_uplift_pass", "cpcv_worst_quartile_pass"],
    40
  );
  renderQualityGrid("feedbackQualityGrid", { backtest_feedback: data.quality?.backtest_feedback });
  renderTable(
    "modelGateAuditTable",
    modelGateAudit,
    ["gate_group", "candidate_scope", "horizon_days", "model_name", "split", "gate", "passed", "value", "threshold", "block_reason", "block_reason_explanation"],
    1000
  );
  renderTable(
    "backtestEventLedgerTable",
    data.tables.backtest_event_ledger || [],
    ["symbol", "symbol_group", "date", "strategy_id", "variant_id", "horizon_days", "stop_multiple", "candidate_tier", "entry_trigger", "trade_action", "strict_signal_stage", "research_signal_stage", "research_signal_action", "research_signal_score", "entry_signal_pass", "entry_failure_reason", "next_open_available", "target_weight_pct", "position_overlap_flag", "matched_trade_flag", "exit_reason", "realized_r_multiple", "net_return_pct", "stop_hit", "hit_1r", "hit_2r", "label_overlap_count", "sample_uniqueness_weight"],
    800
  );
  renderTable(
    "backtestFeedbackFeatureTable",
    data.tables.backtest_feedback_features || [],
    ["symbol", "date", "strategy_id", "variant_id", "candidate_tier", "entry_trigger", "fb_trigger_success_rate_ewm_60", "fb_trigger_expected_r_ewm_60", "fb_candidate_tier_success_rate_ewm_120", "fb_symbol_stop_rate_252", "fb_symbol_expected_r_20", "fb_symbol_expected_r_60", "fb_symbol_expected_r_120", "fb_model_calibration_residual_ewm_120", "fb_threshold_selected_uplift_ewm_120", "fb_strategy_drawdown_sensitivity_252"],
    800
  );
  renderTable(
    "cpcvPathSummaryTable",
    data.tables.cpcv_path_summary || [],
    ["path_id", "strategy_id", "test_group_ids", "horizon_days", "embargo_days", "purged_train_count", "train_event_count", "test_event_count", "test_start_date", "test_end_date", "test_uplift_pct", "test_cagr_pct", "test_max_drawdown_pct", "test_positive"],
    500
  );
  renderTable(
    "autoResearchTrialTable",
    data.tables.auto_research_trial_ledger || [],
    ["trial_id", "trial_source", "objective_score", "model_family", "model_name", "candidate_scope", "split", "pbo_penalty", "dsr_penalty", "cpcv_penalty", "feature_set_hash", "label_config_hash", "split_config_hash", "data_snapshot_hash", "tested_at_utc"],
    500
  );
  renderTable(
    "researchExpansionManifestTable",
    data.tables.research_expansion_manifest || [],
    ["step", "status", "returncode", "duration_sec", "command"],
    80
  );
  renderTable(
    "pooledModelTable",
    data.tables.pooled_model_comparison || [],
    ["model_name", "model_family", "is_champion", "validation_design", "split", "evaluation_scope", "event_count", "symbol_count", "success_rate", "mean_return_pct", "brier_improvement_pct", "average_precision", "ece", "threshold", "threshold_decision_eligible", "selection_score_col", "probability_col", "selected_event_count", "selected_fraction", "selected_success_rate", "selected_mean_return_pct", "selected_minus_all_pct", "selected_minus_all_ci_lower_pct_paired", "selected_minus_score_baseline_ci_lower_pct_paired", "uplift_bootstrap_p_value_paired", "selected_expectancy_ci_lower_pct", "selected_stop_rate", "utility_weight_label", "calibration_method"],
    80
  );
  renderTable(
    "pooledUpliftBootstrapTable",
    data.tables.pooled_uplift_bootstrap || [],
    ["model_name", "validation_design", "split", "evaluation_scope", "event_count", "selected_event_count", "selected_fraction", "selected_mean_return_pct", "mean_return_pct", "selected_minus_all_pct", "selected_minus_all_ci_lower_pct", "selected_minus_all_ci_lower_pct_paired", "score_baseline_mean_return_pct", "selected_minus_score_baseline_pct", "selected_minus_score_baseline_ci_lower_pct", "selected_minus_score_baseline_ci_lower_pct_paired", "uplift_bootstrap_p_value", "uplift_bootstrap_p_value_paired", "bootstrap_method", "bootstrap_block_col", "fold_positive_uplift_count", "fold_positive_score_baseline_uplift_count", "uplift_pass", "uplift_failure_reasons", "score_baseline_policy", "bootstrap_iterations", "is_champion"],
    200
  );
  renderTable(
    "pooledSliceDiagnosticsTable",
    data.tables.pooled_slice_diagnostics || [],
    ["model_name", "is_champion", "evaluation_scope", "split", "slice_dimension", "slice_value", "event_count", "success_rate", "mean_return_pct", "brier_improvement_pct", "ece", "threshold", "selected_event_count", "selected_fraction", "selected_success_rate", "selected_mean_return_pct", "selected_minus_all_pct", "selected_ci_lower_pct", "selected_stop_rate", "threshold_reason"],
    300
  );
  renderTable(
    "pooledOofPredictionsTable",
    data.tables.pooled_oof_predictions || [],
    ["fold_id", "split", "validation_design", "model_name", "evaluation_scope", "symbol", "symbol_group", "date", "candidate_tier", "entry_trigger", "trend_regime", "vol_regime", "label_success_20d", "label_net_return_pct_20d", "p_success", "utility_score", "threshold", "selected_by_threshold", "threshold_reason", "threshold_decision_eligible"],
    300
  );
  renderTable("pooledCalibrationTable", data.tables.pooled_tsm_calibration || [], ["layer", "model_name", "evaluation_scope", "candidate_tier", "symbol_group", "event_count", "actual_success_rate", "predicted_success_rate", "posterior_success_rate", "logit_shift", "shrinkage", "prior_strength", "status", "fit_source"], 100);
  renderQualityGrid("pooledQualityGrid", {
    schema: data.quality?.schema,
    pooled_dataset: data.quality?.pooled_dataset,
    pooled_model: data.quality?.pooled_model,
  });
  renderTable("pooledUniverseManifestTable", data.tables.pooled_universe_manifest || [], ["step", "status", "returncode", "stdout_tail", "stderr_tail"], 120);
  renderTable("pooledUniverseConfigTable", data.tables.pooled_universe_config || [], ["symbol", "symbol_group", "signals", "risk_policy", "trade_log", "enriched"], 60);
  renderTable("universeValidationTable", data.tables.universe_validation || [], ["symbol", "symbol_group", "loaded", "strict_eligible", "eligibility_status", "row_count", "start_date", "end_date", "failure_reasons"], 120);
  renderTable(
    "tsmLikeMetricsTable",
    data.tables.tsm_like_calibration_metrics || [],
    ["tsm_like_calibration_route", "split", "event_count", "weighted_event_count", "effective_n", "decision_ece", "brier_improvement_pct", "route_status", "tsm_like_route_selection_pass", "tsm_like_route_selection_failure_reasons", "selection_decision_ece", "selection_brier_improvement_pct", "selection_effective_n", "combined_decision_ece", "combined_brier_improvement_pct", "is_selected_tsm_like_route"],
    80
  );
  renderTable(
    "tsmLikePoolTable",
    data.tables.tsm_like_calibration_pool || [],
    ["symbol", "symbol_group", "date", "split", "tsm_like_group", "base_group_weight", "dynamic_weight", "tsm_like_weight", "label_success_20d", "label_net_return_pct_20d", "p_success_calibrated", "p_success_tsm_like_calibrated", "decision_score_tsm_like_calibrated", "selected_tsm_like_calibration_route"],
    300
  );
  renderTable("pooledSplitManifestTable", data.tables.pooled_split_manifest || [], ["split", "symbol_count", "row_count", "start_date", "end_date"], 40);
  renderTable(
    "pooledScopeStatsTable",
    data.tables.pooled_scope_stats || [],
    ["symbol", "symbol_group", "horizon_days", "group_type", "group_value", "total_count", "labeled_count", "unavailable_count", "success_rate_pct", "mean_net_return_pct", "median_net_return_pct", "mean_expected_r", "hit_1r_rate_pct", "hit_2r_rate_pct", "profit_factor", "atr_stop_rate_pct"],
    300
  );
  renderTable("pooledInputFailuresTable", data.tables.pooled_input_failures || [], null, 120);
  renderTable("pooledSchemaTable", data.tables.pooled_schema || [], ["column", "role", "dtype", "missing_rate", "non_null_count"], 300);
  renderTable("pooledThresholdTable", data.tables.pooled_threshold_policy || [], ["model_name", "evaluation_scope", "threshold", "score_col", "probability_col", "selected_count", "selected_fraction", "selected_success_rate", "selected_mean_return_pct", "selected_minus_all_pct", "selected_ci_lower_pct", "selected_stop_hit", "all_stop_hit", "validation_brier_improvement_pct", "validation_ece", "threshold_decision_eligible", "eligible", "threshold_reason", "utility_weight_label"], 160);
  renderMlOverlayEquity("mlOverlayEquityChart", data.series.ml_overlay_equity || []);
  renderTable(
    "mlOverlayTable",
    data.tables.ml_overlay_summary || [],
    ["candidate_scope", "horizon_days", "model_name", "policy", "selection_rate_pct", "event_count", "success_rate_pct", "mean_net_return_pct", "profit_factor", "stop_rate_pct", "cumulative_weighted_return_pct", "max_event_curve_drawdown_pct"],
    100
  );
  renderQualityGrid("mlOverlayQualityGrid", { ml_overlay: data.quality?.ml_overlay });
  renderQualityGrid("shadowQualityGrid", { shadow_paper: data.quality?.shadow_paper });
  renderTable(
    "registryTable",
    data.tables.prediction_model_registry || [],
    ["candidate_scope", "champion_scope", "horizon_days", "model_name", "model_family", "model_policy", "promotion_status", "prediction_quality_pass", "model_quality_pass", "latest_signal_pass", "pooled_decision_support_allowed", "oos_event_count", "selected_oos_event_count", "selected_fraction", "brier_improvement_pct", "ece", "pr_auc", "average_precision", "expectancy_improvement_pct", "rank_score", "feature_count", "dependency_versions", "quality_block_reasons"],
    80
  );
  renderTable("experimentLogTable", data.tables.prediction_experiment_log || [], ["candidate_scope", "horizon_days", "model_name", "status", "block_reasons", "oos_event_count", "brier_improvement_pct", "ece", "pr_auc", "expectancy_improvement_pct", "overlay_mean_return_improvement_pct"], 100);
  renderTable("shadowPaperTable", data.tables.shadow_predictions || [], ["symbol", "prediction_asof_date", "horizon_days", "prediction_scope_used", "best_model", "p_success", "threshold", "prediction_quality_pass", "prediction_signal_status", "prediction_use_status", "decision_permission", "final_trade_decision", "paper_action", "realized_status"], 40);
  renderTable(
    "pooledOosTable",
    data.tables.pooled_oos_predictions || [],
    ["symbol", "symbol_group", "date", "split", "candidate_tier", "is_model_training_candidate", "is_decision_entry_candidate", "is_trade_ready_entry_candidate", "label_success_20d", "label_net_return_pct_20d", "p_success_eb", "p_success_logistic", "p_success_lgbm", "p_success_xgb", "p_success_stack_raw", "p_success_calibrated", "p_success_tsm_calibrated", "p_stop_hit_lgbm", "expected_r_lgbm", "expected_net_return_pct", "decision_score", "decision_score_tsm_calibrated", "utility_score", "threshold", "selected_by_threshold", "threshold_decision_eligible", "entry_trigger", "trend_regime", "vol_regime", "drawdown_bucket"],
    250
  );
}

function groupedSampleAudit(rows) {
  const grouped = new Map();
  for (const row of rows || []) {
    const key = row.symbol_group || "unknown";
    const item = grouped.get(key) || {
      symbol_group: key,
      loaded: 0,
      strict_eligible: 0,
      trade_ready_20d_labeled: 0,
      model_training_20d_labeled: 0,
      test_holdout_trade_ready_20d: 0,
      tsm_like_effective_n_contribution: 0,
    };
    item.loaded += isTruthy(row.loaded) ? 1 : 0;
    item.strict_eligible += isTruthy(row.strict_eligible) ? 1 : 0;
    item.trade_ready_20d_labeled += toNumber(row.trade_ready_20d_labeled) || 0;
    item.model_training_20d_labeled += toNumber(row.model_training_20d_labeled) || 0;
    item.test_holdout_trade_ready_20d += toNumber(row.test_holdout_trade_ready_20d) || 0;
    item.tsm_like_effective_n_contribution += toNumber(row.tsm_like_effective_n_contribution) || 0;
    grouped.set(key, item);
  }
  return Array.from(grouped.values()).sort((a, b) => b.trade_ready_20d_labeled - a.trade_ready_20d_labeled);
}

function renderSampleExpansionHero(data) {
  const universe = data.snapshots.universe_validation || {};
  const sample = data.snapshots.pooled_sample_audit || {};
  const paper = data.snapshots.paper_gate || {};
  const cards = [
    ["후보/로드", `${fmtNumber(universe.loaded_symbols, 0)} / ${fmtNumber(universe.candidate_symbols, 0)}`, `target >=45 · ${statusByBool(universe.loaded_target_pass, "통과", "미달")}`, true],
    ["Strict eligible", fmtNumber(universe.strict_eligible_symbols, 0), `target >=35 · short history ${fmtNumber(universe.short_history_research_only, 0)}`, false],
    ["20D trade-ready", fmtNumber(sample.trade_ready_20d_labeled, 0), "target >=10,000", false],
    ["20D model-training", fmtNumber(sample.model_training_20d_labeled, 0), "target >=100,000", false],
    ["TSM-like effective N", fmtNumber(paper.tsm_like_effective_train_validation_n, 0), `target >=500 · ${labelValue(paper.tsm_like_calibration_route)}`, false],
    ["Paper gate", labelValue(paper.paper_gate_status), explainReasonList(paper.paper_gate_block_reasons, 2), false],
  ];
  const container = $("sampleExpansionHero");
  if (container) {
    container.innerHTML = cards
      .map(([label, value, note, primary]) => `<article class="sample-card ${primary ? "good" : "neutral"}"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><p title="${escapeHtml(note)}">${escapeHtml(compactText(note, 120))}</p></article>`)
      .join("");
  }
}

function renderResearchExpansionHero(data) {
  const research = data.snapshots.research_expansion || {};
  const modelGate = data.snapshots.model_gate || {};
  const feedbackCoverage = clampNumber(((toNumber(research.feedback_feature_rows) || 0) / Math.max(toNumber(research.event_ledger_rows) || 1, 1)) * 100, 0, 92);
  const cards = [
    ["연구 검증", statusByBool(research.research_validation_pass, "통과", "보류"), `system quality ${statusByBool(research.pooled_system_quality_pass, "통과", "보류")}`, true],
    ["이벤트 원장", fmtNumber(research.event_ledger_rows, 0), `symbols ${fmtNumber(research.event_ledger_symbol_count, 0)} · variants ${fmtNumber(research.event_ledger_variant_count, 0)}`, false],
    ["near-miss", fmtNumber(research.event_ledger_near_miss_rows, 0), `research rows ${fmtNumber(research.event_ledger_research_candidate_rows, 0)}`, false],
    ["피드백 피처", fmtNumber(research.feedback_feature_rows, 0), `features ${fmtNumber(research.feedback_feature_count, 0)} · coverage ${fmtNumber(feedbackCoverage, 0)}`, false],
    ["자동 Trial", fmtNumber(research.auto_research_trial_count, 0), `${labelValue(research.best_research_model)} · score ${fmtNumber(research.best_research_objective_score, 2)}`, false],
    ["CPCV 모델", labelValue(research.best_cpcv_model), `median ${fmtMaybePct(research.best_cpcv_model_median_uplift_pct, 2)} · q25 ${fmtMaybePct(research.best_cpcv_model_q25_uplift_pct, 2)}`, false],
    ["CPCV 전략", labelValue(research.best_cpcv_strategy), `median ${fmtMaybePct(research.best_cpcv_strategy_median_uplift_pct, 2)} · q25 ${fmtMaybePct(research.best_cpcv_strategy_q25_uplift_pct, 2)}`, false],
    ["Gate", labelValue(modelGate.model_gate_status), `${fmtNumber(modelGate.failed_gate_count, 0)} failed · ${labelValue(modelGate.failed_gate_groups)}`, false],
  ];
  const container = $("researchExpansionHero");
  if (container) {
    container.innerHTML = cards
      .map(([label, value, note, primary]) => `<article class="sample-card ${primary ? (isTruthy(research.research_validation_pass) ? "good" : "warn") : "neutral"}"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><p title="${escapeHtml(note)}">${escapeHtml(compactText(note, 120))}</p></article>`)
      .join("");
  }
}

function renderSampleGroupChart(id, rows) {
  const grouped = groupedSampleAudit(rows);
  horizontalBarChart(
    $(id),
    grouped,
    (row) => labelValue(row.symbol_group),
    (row) => toNumber(row.trade_ready_20d_labeled),
    { valueFormat: (v) => fmtNumber(v, 0), color: colors.teal, limit: 12 }
  );
}

function renderTsmLikeFacts(id, data) {
  const paper = data.snapshots.paper_gate || {};
  const selected = (data.tables.tsm_like_calibration_metrics || []).find((row) => isTruthy(row.is_selected_tsm_like_route) && String(row.split) === "tsm_like_train_validation") || {};
  renderFacts(id, [
    ["선택 route", labelValue(paper.tsm_like_calibration_route || selected.tsm_like_calibration_route)],
    ["Route 통과", statusByBool(paper.tsm_like_route_selection_pass, "통과", "미통과")],
    ["Effective N", fmtNumber(paper.tsm_like_effective_train_validation_n || selected.effective_n, 0)],
    ["Train/validation ECE", fmtNumber(paper.tsm_like_calibration_ece || selected.decision_ece, 4)],
    ["Brier 개선", fmtMaybePct(selected.brier_improvement_pct, 2)],
    ["Test+holdout ECE", fmtNumber(selected.combined_decision_ece, 4)],
  ]);
}

function renderPaperGateFacts(id, data) {
  const paper = data.snapshots.paper_gate || {};
  renderFacts(id, [
    ["Strict", labelValue(paper.strict_gate_status)],
    ["Paper", labelValue(paper.paper_gate_status)],
    ["Paper 판단 가능", statusByBool(paper.paper_decision_support_allowed, "가능", "차단")],
    ["최신 trade-ready", statusByBool(paper.latest_trade_ready, "예", "아니오")],
    ["최신 손절 가능성", fmtMaybePct(paper.latest_stop_hit_20d, 1)],
    ["Live", labelValue(paper.live_trading_status || "DISABLED_BY_DESIGN")],
  ]);
}

function renderModelOpsHero(pooled, system) {
  const cards = [
    ["판단에 참고 가능", statusByBool(pooled.decision_support_allowed, "가능", "기준 미달"), explainReasonList(pooled.decision_block_reasons, 2), true],
    ["모델 품질", statusByBool(pooled.model_quality_pass, "통과", "기준 미달"), explainReasonList(pooled.model_quality_block_reasons, 2), false],
    ["최신 신호", statusByBool(pooled.latest_signal_pass, "통과", "차단"), explainReasonList(pooled.latest_block_reasons, 2), false],
    ["여러 종목 모델", labelValue(pooled.model_family || pooled.model_name), `${compactText(labelValue(pooled.model_name), 28)} · 선택비율 ${fmtMaybePct(pooled.selected_fraction, 1)}`, false],
    ["결정 점수", fmtNumber(pooled.decision_score_20d, 3), `기준 ${fmtNumber(pooled.threshold_20d, 3)}`, false],
    ["20일 오를 가능성", fmtMaybePct(pooled.p_success_20d, 1), `손절 가능성 ${fmtMaybePct(pooled.p_stop_hit_20d, 1)}`, false],
    ["Uplift 검증", statusByBool(pooled.uplift_pass, "통과", "보류"), `p ${fmtNumber(pooled.uplift_bootstrap_p_value_paired ?? pooled.uplift_bootstrap_p_value, 3)} · ${labelValue(pooled.bootstrap_method)}`, false],
    ["가상 기록", statusByBool(system.paper_ready, "준비됨", "미준비"), labelValue(system.paper_trading_status), false],
  ];
  $("modelOpsHero").innerHTML = cards
    .map(([label, value, note, primary]) => `<article class="prediction-card${primary ? " primary" : ""}"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><p>${escapeHtml(note)}</p></article>`)
    .join("");
}

function renderDiagnostics(data) {
  renderPageSynthesis("diagnosticsSynthesis", deriveDiagnosticsSynthesis(data));
  const prediction = data.snapshots.prediction || {};
  const pooled = data.snapshots.pooled_prediction || {};
  const modelGate = data.snapshots.model_gate || {};
  const system = data.snapshots.system || {};
  const gateAudit = data.tables.model_gate_audit || [];
  const gateAuditWithExplanation = gateAudit.map((row) => enrichGateRow(row));
  const failedGateRows = gateAuditWithExplanation.filter((row) => !isTruthy(row.passed));

  renderDiagnosticHero(data);
  renderDiagnosticNotice(prediction, pooled, modelGate, system);
  renderDiagnosticPredictionCards("diagnosticPredictionCards", predictionRows(prediction, pooled));
  renderDiagnosticGateSummary("diagnosticGateSummary", gateAudit, modelGate, system);
  renderTable(
    "diagnosticRootCauseTable",
    data.tables.model_gate_root_causes || [],
    ["priority", "root_cause", "failed_gate_count", "category", "gate_groups", "gates", "candidate_scopes", "horizons", "models", "splits", "value_min", "value_max", "recommended_action"],
    200
  );
  renderTable(
    "diagnosticFailedGateTable",
    failedGateRows,
    ["gate_group", "candidate_scope", "horizon_days", "model_name", "split", "gate", "passed", "value", "threshold", "block_reason", "block_reason_explanation"],
    1000
  );
  renderTable("diagnosticPredictionSnapshotTable", snapshotRows(prediction), ["raw_field", "raw_value"], 1000);
  renderTable("diagnosticPooledSnapshotTable", snapshotRows(pooled), ["raw_field", "raw_value"], 1000);
  renderTable("diagnosticModelGateSnapshotTable", snapshotRows(modelGate), ["raw_field", "raw_value"], 120);
  renderTable("diagnosticSystemSnapshotTable", snapshotRows(system), ["raw_field", "raw_value"], 160);
  renderTable(
    "diagnosticLocalModelTable",
    data.tables.prediction_comparison || [],
    [
      "candidate_scope",
      "horizon_days",
      "model_name",
      "decision_scope_eligible",
      "prediction_quality_pass",
      "oos_event_count",
      "selected_oos_event_count",
      "min_selected_events_per_fold",
      "brier_improvement_pct",
      "ece",
      "pr_auc",
      "base_rate_pr_auc",
      "selected_minus_rule_all_pct",
      "selected_signal_expectancy_ci_lower_pct",
      "positive_expectancy_folds",
      "min_calibration_bin_n",
      "threshold_iqr",
      "rank_score",
    ],
    500
  );
  renderTable(
    "diagnosticPooledModelTable",
    data.tables.pooled_model_comparison || [],
    [
      "model_name",
      "validation_design",
      "split",
      "evaluation_scope",
      "event_count",
      "symbol_count",
      "success_rate",
      "mean_return_pct",
      "brier_improvement_pct",
      "ece",
      "threshold",
      "selected_event_count",
      "selected_success_rate",
      "selected_mean_return_pct",
      "selected_minus_all_pct",
      "selected_minus_all_ci_lower_pct_paired",
      "selected_minus_score_baseline_ci_lower_pct_paired",
      "uplift_bootstrap_p_value_paired",
      "selected_expectancy_ci_lower_pct",
      "selected_stop_rate",
      "utility_weight_label",
    ],
    200
  );
  renderTable(
    "diagnosticPooledSliceDiagnosticsTable",
    data.tables.pooled_slice_diagnostics || [],
    ["model_name", "is_champion", "evaluation_scope", "split", "slice_dimension", "slice_value", "event_count", "success_rate", "mean_return_pct", "brier_improvement_pct", "ece", "selected_fraction", "selected_ci_lower_pct", "selected_stop_rate"],
    500
  );
  renderTable(
    "diagnosticCalibrationPolicyTable",
    diagnosticCalibrationPolicyRows(data),
    ["source", "candidate_scope", "horizon_days", "model_name", "binning", "bin_count", "oos_event_count", "ece", "mean_predicted_probability", "observed_success_rate", "max_abs_calibration_error", "min_bin_n", "threshold", "threshold_source", "validation_expectancy_pct", "validation_selected_count", "selected_count", "selected_success_rate", "selected_mean_return_pct", "selected_minus_all_pct", "selected_ci_lower_pct"],
    500
  );
}

function enrichGateRow(row) {
  const reasons = reasonCodes(row.block_reason);
  return {
    ...row,
    block_reason_explanation: reasons.length ? reasons.map((item) => explainReason(item.code)).join(" / ") : "통과",
  };
}

function renderDiagnosticHero(data) {
  const prediction = data.snapshots.prediction || {};
  const pooled = data.snapshots.pooled_prediction || {};
  const modelGate = data.snapshots.model_gate || {};
  const system = data.snapshots.system || {};
  const cards = [
    ["표시 방식", "기준은 유지", "막힌 값도 확인용으로 모두 표시", true],
    ["모델 기준", labelValue(modelGate.model_gate_status), `${fmtNumber(modelGate.failed_gate_count, 0)}개 미통과`, false],
    ["예측 사용 상태", labelValue(prediction.prediction_use_status), labelValue(prediction.prediction_signal_status), false],
    ["라이브 상태", labelValue(system.live_trading_status), labelValue(system.live_block_reasons), false],
  ];
  $("diagnosticHero").innerHTML = cards
    .map(([label, value, note, primary]) => `<article class="prediction-card${primary ? " primary" : ""}"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><p>${escapeHtml(note)}</p></article>`)
    .join("");
}

function renderDiagnosticNotice(prediction, pooled, modelGate, system) {
  const notice = $("diagnosticNotice");
  if (!notice) return;
  notice.classList.add("visible");
  const parts = [
    `모델 기준은 ${labelValue(modelGate.model_gate_status)} 상태입니다.`,
    `예측은 ${labelValue(prediction.prediction_use_status)}이며, 판단 가능 여부는 ${labelValue(prediction.decision_permission)}입니다.`,
    `여러 종목 예측은 ${statusByBool(pooled.decision_support_allowed, "판단에 참고 가능", "기준 미달")}입니다.`,
    `라이브 트레이딩은 ${labelValue(system.live_trading_status)}입니다.`,
  ];
  notice.innerHTML = `전체 확인 보기<span>${escapeHtml(parts.join(" "))}</span>`;
}

function renderDiagnosticPredictionCards(id, rows) {
  const target = $(id);
  if (!rows.length) {
    target.innerHTML = `<div class="preview-empty">예측 수치 없음</div>`;
    return;
  }
  target.innerHTML = rows
    .map((row) => {
      const status = row.prediction_status || "참고용";
      const tone = status.includes("통과") ? "good" : status.includes("미달") || status.includes("이하") ? "warn" : "neutral";
      const reasonText = row.quality_block_reasons ? labelValue(row.quality_block_reasons) : "막힌 이유 없음";
      const metrics = [
        ["오를 가능성", fmtMaybePct(row.p_success, 1)],
        ["손절 안 날 가능성", fmtMaybePct(row.p_stop_survival, 1)],
        ["손절 확률", fmtMaybePct(row.p_stop_hit, 1)],
        ["수익으로 끝날 가능성", fmtMaybePct(row.p_positive_given_survival, 1)],
        ["1차 목표 도달", fmtMaybePct(row.p_hit_1r, 1)],
        ["2차 목표 도달", fmtMaybePct(row.p_hit_2r, 1)],
        ["위험 대비 기대수익", fmtNumber(row.expected_r, 3)],
        ["기대 순수익", fmtPlanPctPoint(row.expected_net_return_pct, 1)],
        ["선택 기준값", fmtMaybePct(row.threshold, 1)],
        ["사례 수", `${fmtNumber(row.oos_event_count, 0)} / ${fmtNumber(row.effective_oos_event_count, 0)} / ${fmtNumber(row.selected_oos_event_count, 0)}`],
      ];
      return `
        <article class="diagnostic-card ${tone}">
          <div class="diagnostic-card-head">
            <div>
              <span>${escapeHtml(row.prediction_scope_label)}</span>
              <strong>${escapeHtml(row.horizon_days)}일 · ${escapeHtml(labelValue(row.model_name))}</strong>
            </div>
            <em>${escapeHtml(status)}</em>
          </div>
          <dl>${metrics.map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("")}</dl>
          <p title="${escapeHtml(reasonText)}">${escapeHtml(compactText(reasonText, 120))}</p>
        </article>
      `;
    })
    .join("");
}

function renderDiagnosticGateSummary(id, audit, modelGate, system) {
  const failed = (audit || []).filter((row) => !isTruthy(row.passed));
  const passed = (audit || []).filter((row) => isTruthy(row.passed));
  const byGroup = countRows(failed, (row) => labelValue(row.gate_group));
  const byGate = countRows(failed, (row) => labelValue(row.gate));
  const groupCards = Object.entries(byGroup)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([label, count]) => [label, `${count}개`, "기준 미달"]);
  const topGate = Object.entries(byGate).sort((a, b) => b[1] - a[1])[0];
  const cards = [
    ["전체 기준", `${audit.length}개`, `${passed.length}개 통과 / ${failed.length}개 미통과`],
    ["중요 기준 미통과", fmtNumber(modelGate.critical_failed_gate_count, 0), labelValue(modelGate.failed_gate_groups)],
    ["예측 참고 가능", statusByBool(system.prediction_decision_support, "가능", "아직 불가"), labelValue(system.prediction_block_reasons)],
    ["가장 많이 막힌 기준", topGate ? topGate[0] : "없음", topGate ? `${topGate[1]}개` : "문제 없음"],
    ...groupCards,
  ];
  $(id).innerHTML = cards
    .map(([label, value, note]) => `<article class="display-item"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><p>${escapeHtml(note)}</p></article>`)
    .join("");
}

function countRows(rows, keyFn) {
  return (rows || []).reduce((acc, row) => {
    const key = keyFn(row) || "없음";
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});
}

function snapshotRows(snapshot) {
  return Object.entries(snapshot || {}).map(([key, value]) => ({
    raw_field: key,
    raw_value: value,
  }));
}

function diagnosticCalibrationPolicyRows(data) {
  const calibration = (data.tables.prediction_calibration_summary || []).map((row) => ({ source: "단독 예측 확률 오차", ...row }));
  const threshold = (data.tables.prediction_threshold_policy || []).map((row) => ({ source: "단독 예측 선택 기준", ...row }));
  const pooledThreshold = (data.tables.pooled_threshold_policy || []).map((row) => ({ source: "여러 종목 선택 기준", ...row }));
  const pooledCalibration = (data.tables.pooled_tsm_calibration_metrics || []).map((row) => ({ source: "TSMC 확률 조정", ...row }));
  return [...calibration, ...threshold, ...pooledThreshold, ...pooledCalibration];
}

function renderMlOverlayEquity(id, rows) {
  const preferred = (rows || []).filter((row) => String(row.horizon_days) === "20" && ["RULE_ALL_OOS_EVENTS", "ML_THRESHOLD_SELECTED"].includes(String(row.policy)));
  const source = preferred.length ? preferred : rows || [];
  const grouped = groupBy(source, (row) => `${row.candidate_scope}|${row.horizon_days}|${row.model_name}|${row.policy}`);
  const palette = [colors.blue, colors.green, colors.amber, colors.red, colors.violet, colors.teal];
  const series = Object.keys(grouped)
    .slice(0, 8)
    .map((key, idx) => {
      const [scope, horizon, model, policy] = key.split("|");
      return {
        key,
        label: `${labelValue(scope)} ${horizon}일 ${labelValue(model)} ${labelValue(policy)}`,
        color: palette[idx % palette.length],
        rows: grouped[key],
      };
    });
  groupedLineChart($(id), series, "equity", { yFormat: (v) => fmtNumber(v, 2) });
}

function renderPredictionHero(p, pooled = {}) {
  const source = predictionDisplaySource(p, pooled);
  const scopeNote = source.prefix
    ? `${labelValue(snapshotValue(p, ["latest_entry_gate_status", "prediction_entry_gate_status"]))} / 참고값 표시`
    : `${labelValue(p.prediction_signal_status)} / ${labelValue(p.prediction_use_status)}`;
  const cards = [
    ["현재 예측 종류", source.scopeLabel, scopeNote, true],
    ["20일 오를 가능성", fmtMaybePct(source.pSuccess20, 1), source.successNote, false],
    ["20일 손절 안 날 가능성", fmtMaybePct(source.pStopSurvival20, 1), source.sampleNote, false],
    ["20일 수익 가능성", fmtMaybePct(source.pPositive20, 1), `예측 방식 ${source.modelLabel}`, false],
  ];
  $("predictionHero").innerHTML = cards
    .map(([label, value, note, primary]) => `<article class="prediction-card${primary ? " primary" : ""}"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><p>${escapeHtml(note)}</p></article>`)
    .join("");
}

function renderAllPredictionTable(id, p, pooled = {}) {
  const rows = predictionRows(p, pooled);
  renderTable(
    id,
    rows,
    [
      "prediction_scope_label",
      "horizon_days",
      "model_name",
      "prediction_status",
      "p_success",
      "p_stop_survival",
      "p_stop_hit",
      "p_positive_given_survival",
      "p_hit_1r",
      "p_hit_2r",
      "expected_r",
      "expected_net_return_pct",
      "decision_score",
      "threshold",
      "oos_event_count",
      "effective_oos_event_count",
      "selected_oos_event_count",
      "selected_fraction",
      "model_quality_pass",
      "latest_signal_pass",
      "decision_support_allowed",
      "quality_block_reasons",
    ],
    20
  );
}

function predictionRows(p, pooled = {}) {
  const rows = [];
  const valueGroups = [
    ["매수 신호 후보", "trigger"],
    ["관찰 후보", "context"],
  ];
  for (const [label, prefix] of valueGroups) {
    for (const horizon of [20, 60]) {
      const row = predictionRowFromSnapshot(p, label, prefix, horizon);
      if (row) rows.push(row);
    }
  }
  const pooledRow = predictionRowFromPooled(pooled, p);
  if (pooledRow) rows.push(pooledRow);
  const decisionGroups = [
    ["최종 판단", ""],
    ["매수 준비", "trade_ready"],
  ];
  for (const [label, prefix] of decisionGroups) {
    for (const horizon of [20, 60]) {
      const row = predictionRowFromSnapshot(p, label, prefix, horizon);
      if (row) rows.push(row);
    }
  }
  return rows;
}

function predictionRowFromSnapshot(p, label, prefix, horizon) {
  const key = (name) => (prefix ? `${prefix}_${name}_${horizon}d` : `${name}_${horizon}d`);
  const model = p[key("best_model")];
  const pSuccess = p[key("p_success")];
  const hasAny =
    model ||
    toNumber(pSuccess) !== null ||
    toNumber(p[key("p_stop_survival")]) !== null ||
    toNumber(p[key("p_hit_1r")]) !== null ||
    toNumber(p[key("oos_event_count")]) !== null ||
    p[key("model_quality_block_reasons")];
  if (!hasAny) return null;
  return {
    prediction_scope_label: label,
    horizon_days: horizon,
    model_name: model,
    prediction_status: predictionStatusLabel(p[key("prediction_quality_pass")], p[key("model_quality_block_reasons")]),
    p_success: pSuccess,
    p_stop_survival: p[key("p_stop_survival")],
    p_stop_hit: p[key("p_stop_hit")],
    p_positive_given_survival: p[key("p_positive_given_survival")],
    p_hit_1r: p[key("p_hit_1r")],
    p_hit_2r: p[key("p_hit_2r")],
    expected_r: p[key("expected_r")],
    expected_net_return_pct: p[key("expected_net_return")],
    threshold: p[key("threshold")],
    oos_event_count: p[key("oos_event_count")],
    effective_oos_event_count: p[key("effective_oos_event_count")],
    selected_oos_event_count: p[key("selected_oos_event_count")],
    quality_block_reasons: p[key("model_quality_block_reasons")],
  };
}

function predictionRowFromPooled(pooled, p) {
  const pSuccess = pooled.p_success_20d ?? p.pooled_p_success_20d;
  if (toNumber(pSuccess) === null && !(pooled.model_name ?? p.pooled_model_name)) return null;
  const quality = pooled.model_quality_pass ?? p.pooled_model_quality_pass;
  return {
    prediction_scope_label: "통합 표본",
    horizon_days: 20,
    model_name: pooled.model_name ?? p.pooled_model_name,
    prediction_status: predictionStatusLabel(quality, pooled.model_quality_block_reasons ?? p.pooled_model_quality_block_reasons),
    p_success: pSuccess,
    p_stop_survival: pooled.p_stop_survival_20d ?? p.pooled_p_stop_survival_20d,
    p_stop_hit: pooled.p_stop_hit_20d ?? p.pooled_p_stop_hit_20d,
    p_positive_given_survival: null,
    p_hit_1r: pooled.p_hit_1r_20d ?? p.pooled_p_hit_1r_20d,
    p_hit_2r: pooled.p_hit_2r_20d ?? p.pooled_p_hit_2r_20d,
    expected_r: pooled.expected_r_net_20d ?? pooled.expected_r_20d ?? p.pooled_expected_r_net_20d ?? p.pooled_expected_r_20d,
    expected_net_return_pct: pooled.expected_net_return_pct_20d ?? pooled.expected_net_return_20d ?? p.pooled_expected_net_return_pct_20d ?? p.pooled_expected_net_return_20d,
    decision_score: pooled.decision_score_20d ?? p.pooled_decision_score_20d,
    threshold: pooled.threshold_20d ?? p.pooled_threshold_20d,
    oos_event_count: pooled.oos_event_count ?? p.pooled_oos_event_count,
    effective_oos_event_count: pooled.effective_group_n ?? p.pooled_effective_group_n,
    selected_oos_event_count: pooled.selected_oos_event_count ?? p.pooled_selected_oos_event_count,
    selected_fraction: pooled.selected_fraction ?? p.pooled_selected_fraction,
    model_quality_pass: pooled.model_quality_pass ?? p.pooled_model_quality_pass,
    latest_signal_pass: pooled.latest_signal_pass ?? p.pooled_latest_signal_pass,
    decision_support_allowed: pooled.decision_support_allowed ?? p.pooled_decision_support_allowed,
    quality_block_reasons: pooled.model_quality_block_reasons ?? p.pooled_model_quality_block_reasons,
  };
}

function predictionStatusLabel(passValue, reasons) {
  if (isTruthy(passValue)) return "기준 통과";
  if (reasons && String(reasons).toUpperCase() !== "PASS") return "기준 미달";
  return "참고용";
}

function renderPredictionNotice(p, pooled = {}) {
  const notice = $("predictionNotice");
  if (!notice) return;
  const latestStatus = `${labelValue(snapshotValue(p, ["latest_entry_gate_status", "prediction_entry_gate_status"]))} / ${labelValue(p.latest_candidate_scope)}`;
  const pooledReason = pooled.decision_block_reasons ? `여러 종목 예측이 막힌 이유: ${labelValue(pooled.decision_block_reasons)}` : "";
  const splitStatus = `모델 품질 ${statusByBool(pooled.model_quality_pass, "통과", "미통과")}, 최신 신호 ${statusByBool(pooled.latest_signal_pass, "통과", "차단")}`;
  notice.classList.add("visible");
  notice.innerHTML = `매매에 바로 못 쓰는 예측도 숨기지 않고 보여줍니다.<span>최신 신호 상태는 ${escapeHtml(latestStatus)}입니다. 여러 종목 기준은 ${escapeHtml(splitStatus)}로 분리해 표시합니다.${pooledReason ? ` ${escapeHtml(pooledReason)}` : ""}</span>`;
}

function predictionDisplaySource(p, pooled = {}) {
  const candidates = [
    {
      prefix: "",
      scopeLabel: labelValue(p.prediction_scope_used),
      typeLabel: "최종 판단값",
      pSuccess20: p.p_success_20d,
      pStopSurvival20: p.p_stop_survival_20d,
      pPositive20: p.p_positive_given_survival_20d,
      pLower20: p.p_success_lower_80_20d,
      pUpper20: p.p_success_upper_80_20d,
      sample: p.effective_oos_event_count_20d,
      model: p.best_model_20d,
    },
    {
      prefix: "pooled",
      scopeLabel: "여러 종목 참고값",
      typeLabel: "여러 종목",
      pSuccess20: pooled.p_success_20d ?? p.pooled_p_success_20d,
      pStopSurvival20: pooled.p_stop_survival_20d ?? p.pooled_p_stop_survival_20d,
      pPositive20: null,
      pLower20: null,
      pUpper20: null,
      sample: pooled.oos_event_count ?? p.pooled_oos_event_count,
      model: pooled.model_name ?? p.pooled_model_name,
      decisionScore: pooled.decision_score_20d ?? p.pooled_decision_score_20d,
      threshold: pooled.threshold_20d ?? p.pooled_threshold_20d,
    },
    {
      prefix: "trigger",
      scopeLabel: "매수 신호 후보 참고값",
      typeLabel: "매수 신호 후보",
      pSuccess20: p.trigger_p_success_20d,
      pStopSurvival20: p.trigger_p_stop_survival_20d,
      pPositive20: p.trigger_p_positive_given_survival_20d,
      pLower20: p.trigger_p_success_lower_80_20d,
      pUpper20: p.trigger_p_success_upper_80_20d,
      sample: p.trigger_effective_oos_event_count_20d,
      model: p.trigger_best_model_20d,
    },
    {
      prefix: "context",
      scopeLabel: "관찰 후보 참고값",
      typeLabel: "관찰 후보",
      pSuccess20: p.context_p_success_20d,
      pStopSurvival20: p.context_p_stop_survival_20d,
      pPositive20: p.context_p_positive_given_survival_20d,
      pLower20: p.context_p_success_lower_80_20d,
      pUpper20: p.context_p_success_upper_80_20d,
      sample: p.context_effective_oos_event_count_20d,
      model: p.context_best_model_20d,
    },
  ];
  const selected = candidates.find((item) => toNumber(item.pSuccess20) !== null) || candidates[0];
  return {
    ...selected,
    scopeLabel: selected.prefix ? selected.scopeLabel : selected.scopeLabel || "최종 판단 없음",
    successNote: selected.decisionScore !== undefined ? `결정 점수 ${fmtNumber(selected.decisionScore, 3)} / 기준 ${fmtNumber(selected.threshold, 3)}` : selected.pLower20 || selected.pUpper20 ? `${fmtMaybePct(selected.pLower20, 1)} ~ ${fmtMaybePct(selected.pUpper20, 1)}` : selected.typeLabel,
    sampleNote: `사용 사례 ${fmtNumber(selected.sample, 0)}건`,
    modelLabel: labelValue(selected.model),
  };
}

function renderStageProbChart(id, p, pooled = {}) {
  const probRow = (label, value) => {
    const n = toNumber(value);
    return { label, value: n === null ? null : n * 100 };
  };
  const rows = [
    probRow("최종 판단 20일 오를 가능성", p.p_success_20d),
    probRow("최종 판단 20일 손절 안 날 가능성", p.p_stop_survival_20d),
    probRow("매수 신호 20일 오를 가능성", p.trigger_p_success_20d),
    probRow("매수 신호 20일 손절 안 날 가능성", p.trigger_p_stop_survival_20d),
    probRow("관찰 후보 20일 오를 가능성", p.context_p_success_20d),
    probRow("관찰 후보 20일 손절 안 날 가능성", p.context_p_stop_survival_20d),
    probRow("여러 종목 20일 오를 가능성", pooled.p_success_20d ?? p.pooled_p_success_20d),
    probRow("여러 종목 20일 손절 안 날 가능성", pooled.p_stop_survival_20d ?? p.pooled_p_stop_survival_20d),
    probRow("매수 신호 60일 오를 가능성", p.trigger_p_success_60d),
    probRow("관찰 후보 60일 오를 가능성", p.context_p_success_60d),
  ].filter((row) => row.value !== null && Number.isFinite(row.value));
  horizontalBarChart($(id), rows, (row) => row.label, (row) => row.value, { valueFormat: (v) => fmtPct(v, 1), color: colors.blue, maxAbs: 100 });
}

function renderBacktest(data) {
  renderPageSynthesis("backtestSynthesis", deriveBacktestSynthesis(data));
  renderEquityChart("equityChart", data.series.equity || []);
  horizontalBarChart(
    $("backtestBars"),
    data.tables.backtest_summary || [],
    (row) => labelValue(row.strategy_id),
    (row) => toNumber(row.total_return_pct),
    { valueFormat: (v) => fmtPct(v, 1), color: colors.green }
  );
  renderTable("backtestTable", data.tables.backtest_summary || [], ["strategy_id", "total_return_pct", "cagr_pct", "max_drawdown_pct", "sharpe_zero_rf", "trade_count", "win_rate_pct", "profit_factor"], 30);
  renderTable("yearlyTable", data.tables.yearly_returns || [], ["strategy_id", "year", "year_return_pct", "year_max_drawdown_pct", "exposure_days_pct"], 160);
  renderTable("tradeLogTable", data.tables.trade_log || [], ["strategy_id", "strategy_group", "trade_event", "entry_date", "entry_price", "target_weight_pct", "exit_date", "exit_price", "exit_reason", "holding_trading_days", "net_return_pct", "portfolio_return_pct", "r_multiple"], 250);
}

function renderRisk(data) {
  renderPageSynthesis("riskSynthesis", deriveRiskSynthesis(data));
  const r = data.snapshots.risk || {};
  const s = data.snapshots.stress || {};
  renderFacts("riskFacts", [
    ["기준일", shortDate(r.date)],
    ["위험 상태", labelValue(r.risk_state)],
    ["살 수 있는 최대 비중", fmtMaybePct(snapshotWeightValue(r, ["final_recommended_max_weight", "final_recommended_max_weight_pct"]), 2)],
    ["제한 사유", labelValue(r.limiting_reason)],
    ["손절가", fmtCurrency(r.stop_price_2atr)],
    ["손절까지 거리", fmtMaybePct(r.risk_pct_2atr, 2)],
  ]);
  renderFacts("stressFacts", [
    ["가장 나쁜 가정", labelValue(s.worst_current_weight_scenario)],
    ["계좌 영향", fmtMaybePct(s.worst_current_weight_portfolio_impact_pct, 2)],
    ["가정 가격", fmtCurrency(s.worst_current_weight_implied_price)],
    ["최악 전략", labelValue(s.worst_strategy_by_mdd)],
    ["하락 점검 상태", labelValue(s.stress_status)],
    ["가장 크게 빠진 폭", fmtMaybePct(s.worst_strategy_mdd_pct, 2)],
  ]);
  lineChart(
    $("riskWeightChart"),
    data.series.risk || [],
    [
      { key: "final_recommended_max_weight", label: "최종 권장", color: colors.green },
      { key: "vol_limit_weight", label: "가격 흔들림 기준", color: colors.blue },
      { key: "trend_limit_weight", label: "추세", color: colors.teal },
      { key: "score_limit_weight", label: "점수", color: colors.amber },
    ],
    { yFormat: (v) => fmtMaybePct(v, 0), minY: 0 }
  );
  renderTable("stressTable", data.tables.stress_scenarios || [], ["scenario_type", "scenario_name", "shock_return_pct", "implied_price", "final_recommended_max_weight_pct", "estimated_portfolio_impact_pct", "notes"], 120);
  renderTable("strategyStressTable", data.tables.strategy_stress || [], ["strategy_id", "worst_daily_return_pct", "worst_20d_return_pct", "max_drawdown_pct", "latest_drawdown_pct", "latest_position_weight_pct", "exposure_days_pct"], 40);
}

function renderQuality(data) {
  renderPageSynthesis("qualitySynthesis", deriveQualitySynthesis(data));
  horizontalBarChart(
    $("walkForwardChart"),
    (data.tables.validation_walk_forward || []).slice(-60),
    (row) => `${labelValue(row.strategy_id)} ${row.window_id}번`,
    (row) => toNumber(row.test_total_return_pct),
    { valueFormat: (v) => fmtPct(v, 1), color: colors.teal }
  );
  renderQualityGrid("qualityDetailGrid", data.quality || {});
  renderTable("manifestTable", data.tables.manifest || [], ["step", "status", "duration_sec", "returncode", "ended_at_utc"], 60);
  renderTable("sensitivityTable", data.tables.validation_sensitivity || [], ["sensitivity_type", "parameter_value", "strategy_id", "total_return_pct", "cagr_pct", "max_drawdown_pct", "trade_count", "win_rate_pct", "profit_factor"], 220);
  horizontalBarChart(
    $("causalWalkForwardChart"),
    (data.tables.validation_causal_walk_forward || []).slice(-80),
    (row) => `${labelValue(row.strategy_id)} ${row.window_id}번`,
    (row) => toNumber(row.test_total_return_pct),
    { valueFormat: (v) => fmtPct(v, 1), color: colors.violet }
  );
  renderTable("causalWalkForwardTable", data.tables.validation_causal_walk_forward || [], ["strategy_id", "strategy_group", "window_id", "test_start_date", "test_end_date", "test_total_return_pct", "test_cagr_pct", "test_max_drawdown_pct", "test_trade_count", "test_exposure_days_pct", "test_positive"], 120);
  renderTable("validationSegmentsTable", data.tables.validation_segments || [], ["strategy_id", "period_type", "period_name", "start_date", "end_date", "total_return_pct", "cagr_pct", "max_drawdown_pct", "sharpe_zero_rf", "exposure_days_pct"], 120);
  renderTable("pboTable", data.tables.pbo_report || [], ["strategy_id", "trial_count", "best_trial_source", "best_sharpe", "median_sharpe", "best_cagr_pct", "pbo_proxy", "overfit_warning"], 40);
  renderTable("deflatedSharpeTable", data.tables.deflated_sharpe || [], ["strategy_id", "trial_count", "observations", "observed_sharpe", "probabilistic_sharpe_ratio", "deflated_sharpe_ratio", "deflated_benchmark_sharpe", "dsr_pass"], 40);
  renderTable("modelTrialsTable", data.tables.model_trials || [], ["strategy_id", "strategy_group", "trial_source", "sharpe", "cagr", "mdd", "tested_at"], 140);
}

function renderSystem(data) {
  renderPageSynthesis("systemSynthesis", deriveSystemSynthesis(data));
  const systemBlocks = (data.tables.system_block_reasons || []).map((row) => {
    const reasons = reasonCodes(row.block_reasons);
    return {
      ...row,
      block_reason_explanation: reasons.length ? reasons.map((item) => explainReason(item.code)).join(" / ") : "통과",
    };
  });
  renderSystemFacts("systemFacts", data);
  renderReadinessBars("systemReadinessBars", data.tables.readiness_scorecard || []);
  renderTable("systemScorecardTable", data.tables.readiness_scorecard || [], ["domain", "passed", "weight", "score", "note"], 40);
  renderTable("systemBlockTable", systemBlocks, ["component", "status", "block_reasons", "block_reason_explanation", "details"], 200);
  renderTable("dataQualityTable", data.tables.data_quality_checks || [], ["check", "passed", "severity", "value", "tolerance", "details"], 120);
  renderQualityGrid("systemQualityGrid", data.quality || {});
  renderTable("dataValidationTable", data.tables.data_validation || [], ["check", "rows_compared", "mismatch_count", "max_abs_diff", "note"], 40);
  renderFailedChecks("failedChecks", data.quality || {});
}

function renderSystemFacts(id, data) {
  const system = data.snapshots.system || {};
  const health = data.snapshots.daily_health || {};
  const dataQuality = data.snapshots.data_quality || {};
  const modelGate = data.snapshots.model_gate || {};
  const integrated = data.snapshots.integrated_price || data.snapshots.summary || {};
  const hourly = data.snapshots.hourly_summary || {};
  const minute = data.snapshots.minute_summary || {};
  renderFacts(id, [
    ["오늘 점검 상태", labelValue(health.daily_health_status)],
    ["막힌 부분", labelValue(health.failed_or_blocked_components)],
    ["데이터 상태", labelValue(dataQuality.data_quality_status)],
    ["모델 기준 상태", labelValue(modelGate.model_gate_status)],
    ["시스템 상태", labelValue(system.system_state)],
    ["전체 준비 점수", `${fmtNumber(system.system_readiness_score, 0)} / 100`],
    ["전체 통과 점수", `${fmtNumber(system.composite_gate_score, 0)} / 100`],
    ["분석 준비 점수", `${fmtNumber(system.research_readiness_score, 0)} / 100`],
    ["수익 가능성 준비 점수", `${fmtNumber(system.alpha_readiness_score, 0)} / 100`],
    ["예측 준비 점수", `${fmtNumber(system.prediction_readiness_score, 0)} / 100`],
    ["가상 기록 준비 점수", `${fmtNumber(system.paper_readiness_score, 0)} / 100`],
    ["가상 기록 상태", labelValue(system.paper_trading_status)],
    ["라이브 상태", labelValue(system.live_trading_status)],
    ["수익 가능성 기준 미달 이유", labelValue(system.alpha_block_reasons)],
    ["예측 기준 미달 이유", labelValue(system.prediction_block_reasons)],
    ["10년 거래일", fmtNumber(integrated.trading_days, 0)],
    ["수정 가격 총수익률", fmtPct(integrated.adj_total_return_pct ?? integrated.total_return_pct, 1)],
    ["시간봉 범위", `${shortDate(hourly.start_timestamp)} ~ ${shortDate(hourly.end_timestamp)}`],
    ["분봉 범위", `${shortDate(minute.start_timestamp)} ~ ${shortDate(minute.end_timestamp)}`],
  ]);
}

function renderTimeframeCards(id, data) {
  const cards = [
    ["일봉", data.snapshots.summary || {}, data.snapshots.latest_price || {}, "trading_days", "start_date", "end_date"],
    ["시간봉", data.snapshots.hourly_summary || {}, data.snapshots.latest_hourly || {}, "hourly_bars", "start_timestamp", "end_timestamp"],
    ["분봉", data.snapshots.minute_summary || {}, data.snapshots.latest_minute || {}, "bars", "start_timestamp", "end_timestamp"],
  ];
  $(id).innerHTML = cards
    .map(([label, summary, latest, rowsKey, startKey, endKey]) => {
      const complete = summary.complete_requested_coverage;
      const coverage = complete === undefined ? "기준 데이터" : complete === true || String(complete).toLowerCase() === "true" ? "요청 범위 완성" : "공개 소스 부분 범위";
      return `<article class="timeframe-card"><div><span>${escapeHtml(label)}</span><strong>${escapeHtml(fmtCurrency(latest.close ?? summary.end_adj_close))}</strong></div><dl><div><dt>범위</dt><dd>${escapeHtml(shortDate(summary[startKey]))} ~ ${escapeHtml(shortDate(summary[endKey]))}</dd></div><div><dt>행 수</dt><dd>${escapeHtml(fmtNumber(summary[rowsKey], 0))}</dd></div><div><dt>수익률</dt><dd>${escapeHtml(fmtPct(summary.total_return_pct, 1))}</dd></div><div><dt>변동성</dt><dd>${escapeHtml(fmtPct(summary.annualized_volatility_pct, 1))}</dd></div><div><dt>상태</dt><dd>${escapeHtml(coverage)}</dd></div></dl></article>`;
    })
    .join("");
}

function renderPriceSummary(id, data) {
  const summary = data.snapshots.integrated_price || data.snapshots.summary || {};
  renderFacts(id, [
    ["기간", summary.period || `${summary.start_date || "없음"} ~ ${summary.end_date || "없음"}`],
    ["거래일", fmtNumber(summary.trading_days, 0)],
    ["시작 종가", fmtCurrency(summary.start_close_usd ?? summary.start_adj_close)],
    ["최신 종가", fmtCurrency(summary.end_close_usd ?? summary.end_adj_close)],
    ["가격 CAGR", fmtPct(summary.price_cagr_pct ?? summary.cagr_pct, 1)],
    ["최대 낙폭", fmtPct(summary.max_drawdown_pct, 1)],
    ["최신 추세", labelValue(summary.latest_trend_regime)],
    ["최신 변동성", labelValue(summary.latest_vol_regime)],
  ]);
}

function renderFailedChecks(id, quality) {
  const sections = Object.entries(quality || {});
  const rows = sections.flatMap(([key, item]) => (item.failed_rows || []).map((row) => ({ source: key, ...row })));
  renderTable(id, rows, ["source", "check", "passed", "severity", "value", "tolerance", "details"], 80);
}

function renderFacts(id, facts) {
  $(id).innerHTML = facts.map(([label, value]) => `<div class="fact"><span>${escapeHtml(label)}</span><strong>${escapeHtml(translateReportText(labelValue(value)))}</strong></div>`).join("");
}

function renderQualityGrid(id, quality) {
  const labels = {
    operational: "운영",
    integrity: "데이터 일관성",
    validation: "검증",
    prediction: "예측",
    data_quality: "데이터 상태",
    data_contract: "원본-가공 비교",
    model_gate: "모델 기준",
    schema: "데이터 형식",
    ml_overlay: "예측 적용",
    pooled_dataset: "여러 종목 데이터",
    pooled_model: "여러 종목 모델",
    backtest_feedback: "피드백 피처",
    shadow_paper: "가상 기록",
  };
  const rows = Object.entries(quality || {}).filter(([, item]) => item).map(([key, item]) => {
    const failed = item.failed ?? 0;
    return `<div class="quality-item"><div><strong>${escapeHtml(labels[key] || key)}</strong><div class="muted">${escapeHtml(item.passed ?? 0)} / ${escapeHtml(item.total ?? 0)}</div></div><strong class="${failed > 0 ? "fail" : "pass"}">${failed > 0 ? `${failed}개 실패` : "통과"}</strong></div>`;
  });
  $(id).innerHTML = rows.join("") || `<div class="preview-empty">데이터 없음</div>`;
}

function renderPriceLine(id, rows) {
  lineChart(
    $(id),
    rows || [],
    [
      { key: "close", label: "종가", color: colors.ink },
      { key: "sma_20", label: "20일선", color: colors.blue },
      { key: "sma_50", label: "50일선", color: colors.teal },
      { key: "sma_200", label: "200일선", color: colors.amber },
    ],
    { yFormat: (v) => `$${Math.round(v)}` }
  );
}

function renderSignalChart(id, rows) {
  lineChart($(id), rows || [], [{ key: "score_price_algo_total", label: "점수", color: colors.blue }], { yFormat: (v) => fmtNumber(v, 0), minY: 0, maxY: 100, threshold: 75 });
}

function renderEquityChart(id, rows) {
  const grouped = groupBy(rows, "strategy_id");
  const palette = [colors.blue, colors.green, colors.amber, colors.red, colors.violet, colors.teal];
  const series = Object.keys(grouped).map((key, idx) => ({ key, label: labelValue(key), color: palette[idx % palette.length], rows: grouped[key] }));
  groupedLineChart($(id), series, "equity", { yFormat: (v) => fmtNumber(v, 1) });
}

function renderReadinessBars(id, rows) {
  horizontalBarChart($(id), rows || [], (row) => labelValue(row.domain), (row) => toNumber(row.score), { valueFormat: (v) => fmtNumber(v, 1), color: colors.blue, maxAbs: 20 });
}

function renderCalibrationChart(id, rows, prediction) {
  const scope = prediction.prediction_scope_used || "context_all";
  const horizon = "20";
  const model = prediction.best_model_20d;
  let selected = (rows || []).filter((row) => String(row.candidate_scope) === String(scope) && String(row.horizon_days) === horizon && String(row.model_name) === String(model));
  if (!selected.length) selected = (rows || []).filter((row) => String(row.horizon_days) === horizon).slice(0, 80);
  scatterCalibration($(id), selected.slice(0, 100));
}

function groupBy(rows, key) {
  return (rows || []).reduce((acc, row) => {
    const value = typeof key === "function" ? key(row) : row[key] || "없음";
    if (!acc[value]) acc[value] = [];
    acc[value].push(row);
    return acc;
  }, {});
}

function setEmpty(container, text = "데이터 없음") {
  container.innerHTML = `<div class="chart-empty">${escapeHtml(text)}</div>`;
}

function chartSvg(width, height, inner) {
  return `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img">${inner}</svg>`;
}

function lineChart(container, rows, series, options = {}) {
  const usableRows = (rows || []).filter((row) => series.some((s) => toNumber(row[s.key]) !== null));
  if (!usableRows.length) return setEmpty(container);
  const w = 1000;
  const h = container.classList.contains("chart-large") ? 440 : 320;
  const p = { l: 64, r: 24, t: 42, b: 38 };
  const values = [];
  for (const row of usableRows) for (const s of series) {
    const n = toNumber(row[s.key]);
    if (n !== null) values.push(n);
  }
  if (options.threshold !== undefined) values.push(options.threshold);
  let yMin = options.minY ?? Math.min(...values);
  let yMax = options.maxY ?? Math.max(...values);
  if (options.fillZero) yMax = Math.max(0, yMax);
  if (yMin === yMax) {
    yMin -= 1;
    yMax += 1;
  }
  const pad = (yMax - yMin) * 0.08;
  yMin = options.minY ?? yMin - pad;
  yMax = options.maxY ?? yMax + pad;
  const x = (i) => p.l + (i * (w - p.l - p.r)) / Math.max(1, usableRows.length - 1);
  const y = (v) => h - p.b - ((v - yMin) / (yMax - yMin)) * (h - p.t - p.b);
  const grid = [];
  for (let i = 0; i <= 4; i += 1) {
    const value = yMin + ((yMax - yMin) * i) / 4;
    const yy = y(value);
    grid.push(`<line x1="${p.l}" y1="${yy}" x2="${w - p.r}" y2="${yy}" stroke="currentColor" opacity="0.12" />`);
    grid.push(`<text x="10" y="${yy + 4}" class="axis-label">${escapeHtml((options.yFormat || fmtNumber)(value))}</text>`);
  }
  const paths = series.map((s) => {
    const pts = usableRows.map((row, i) => {
      const n = toNumber(row[s.key]);
      return n === null ? null : `${x(i).toFixed(1)},${y(n).toFixed(1)}`;
    }).filter(Boolean).join(" ");
    return pts ? `<polyline points="${pts}" fill="none" stroke="${s.color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />` : "";
  }).join("");
  const threshold = options.threshold !== undefined ? `<line x1="${p.l}" y1="${y(options.threshold)}" x2="${w - p.r}" y2="${y(options.threshold)}" stroke="${colors.red}" stroke-dasharray="8 8" stroke-width="2" />` : "";
  const axis = `<text x="${p.l}" y="${h - 10}" class="axis-label">${escapeHtml(shortDate(usableRows[0]?.date))}</text><text x="${w - p.r - 86}" y="${h - 10}" class="axis-label">${escapeHtml(shortDate(usableRows.at(-1)?.date))}</text>`;
  container.innerHTML = `${legend(series)}${chartSvg(w, h, grid.join("") + threshold + paths + axis)}`;
}

function groupedLineChart(container, series, key, options = {}) {
  const usable = series.filter((s) => s.rows?.some((row) => toNumber(row[key]) !== null));
  if (!usable.length) return setEmpty(container);
  const w = 1000;
  const h = container.classList.contains("chart-large") ? 440 : 320;
  const p = { l: 64, r: 24, t: 42, b: 38 };
  const vals = usable.flatMap((s) => s.rows.map((row) => toNumber(row[key])).filter((v) => v !== null));
  let yMin = Math.min(...vals);
  let yMax = Math.max(...vals);
  if (yMin === yMax) {
    yMin -= 1;
    yMax += 1;
  }
  const pad = (yMax - yMin) * 0.08;
  yMin -= pad;
  yMax += pad;
  const y = (v) => h - p.b - ((v - yMin) / (yMax - yMin)) * (h - p.t - p.b);
  const grid = [];
  for (let i = 0; i <= 4; i += 1) {
    const value = yMin + ((yMax - yMin) * i) / 4;
    const yy = y(value);
    grid.push(`<line x1="${p.l}" y1="${yy}" x2="${w - p.r}" y2="${yy}" stroke="currentColor" opacity="0.12" />`);
    grid.push(`<text x="10" y="${yy + 4}" class="axis-label">${escapeHtml((options.yFormat || fmtNumber)(value))}</text>`);
  }
  const paths = usable.map((s) => {
    const rows = s.rows.filter((row) => toNumber(row[key]) !== null);
    const x = (i) => p.l + (i * (w - p.l - p.r)) / Math.max(1, rows.length - 1);
    const pts = rows.map((row, i) => `${x(i).toFixed(1)},${y(toNumber(row[key])).toFixed(1)}`).join(" ");
    return `<polyline points="${pts}" fill="none" stroke="${s.color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />`;
  }).join("");
  const baseRows = usable[0].rows;
  const axis = `<text x="${p.l}" y="${h - 10}" class="axis-label">${escapeHtml(shortDate(baseRows[0]?.date))}</text><text x="${w - p.r - 86}" y="${h - 10}" class="axis-label">${escapeHtml(shortDate(baseRows.at(-1)?.date))}</text>`;
  container.innerHTML = `${legend(usable)}${chartSvg(w, h, grid.join("") + paths + axis)}`;
}

function horizontalBarChart(container, rows, labelFn, valueFn, options = {}) {
  const prepared = (rows || []).map((row) => ({ label: labelFn(row), value: valueFn(row) })).filter((x) => x.value !== null && Number.isFinite(x.value));
  if (!prepared.length) return setEmpty(container);
  const shown = prepared.slice(0, options.limit || 60);
  const w = 1000;
  const rowHeight = 34;
  const h = Math.max(320, shown.length * rowHeight + 58);
  const p = { l: 330, r: 118, t: 24, b: 26 };
  const maxAbs = options.maxAbs || Math.max(...shown.map((item) => Math.abs(item.value)), 1);
  const zeroX = p.l + (w - p.l - p.r) / 2;
  const scale = (w - p.l - p.r) / 2 / maxAbs;
  const bars = shown.map((item, i) => {
    const yy = p.t + i * rowHeight;
    const x0 = item.value >= 0 ? zeroX : zeroX + item.value * scale;
    const width = Math.max(1, Math.abs(item.value * scale));
    const color = item.value >= 0 ? options.color || colors.green : colors.red;
    const rawValueX = item.value >= 0 ? x0 + width + 8 : x0 - 8;
    const valueX = Math.max(p.l + 10, Math.min(w - 10, rawValueX));
    const anchor = rawValueX > w - 10 ? "end" : rawValueX < p.l + 10 ? "start" : item.value >= 0 ? "start" : "end";
    return `<text x="14" y="${yy + 22}" class="bar-label"><title>${escapeHtml(item.label)}</title>${escapeHtml(compactText(item.label, 34))}</text><rect x="${x0}" y="${yy + 7}" width="${width}" height="20" rx="4" fill="${color}" opacity="0.92"></rect><text x="${valueX}" y="${yy + 22}" text-anchor="${anchor}" class="axis-label">${escapeHtml((options.valueFormat || fmtNumber)(item.value))}</text>`;
  }).join("");
  container.innerHTML = chartSvg(w, h, `<line x1="${zeroX}" y1="${p.t - 8}" x2="${zeroX}" y2="${h - p.b}" stroke="currentColor" opacity="0.2" />${bars}`);
}

function scatterCalibration(container, rows) {
  const points = (rows || []).map((row) => ({
    x: toNumber(row.mean_predicted_probability),
    y: toNumber(row.observed_success_rate),
    n: toNumber(row.n) || 1,
  })).filter((p) => p.x !== null && p.y !== null);
  if (!points.length) return setEmpty(container);
  const w = 720;
  const h = 320;
  const p = { l: 58, r: 28, t: 30, b: 44 };
  const x = (v) => p.l + v * (w - p.l - p.r);
  const y = (v) => h - p.b - v * (h - p.t - p.b);
  const grid = [];
  for (let i = 0; i <= 4; i += 1) {
    const v = i / 4;
    grid.push(`<line x1="${p.l}" y1="${y(v)}" x2="${w - p.r}" y2="${y(v)}" stroke="currentColor" opacity="0.12" />`);
    grid.push(`<line x1="${x(v)}" y1="${p.t}" x2="${x(v)}" y2="${h - p.b}" stroke="currentColor" opacity="0.08" />`);
    grid.push(`<text x="8" y="${y(v) + 4}" class="axis-label">${Math.round(v * 100)}%</text>`);
    grid.push(`<text x="${x(v) - 12}" y="${h - 14}" class="axis-label">${Math.round(v * 100)}%</text>`);
  }
  const dots = points.map((pt) => `<circle cx="${x(pt.x)}" cy="${y(pt.y)}" r="${Math.min(12, 4 + Math.sqrt(pt.n))}" fill="${colors.blue}" opacity="0.75"></circle>`).join("");
  container.innerHTML = chartSvg(w, h, `${grid.join("")}<line x1="${p.l}" y1="${h - p.b}" x2="${w - p.r}" y2="${p.t}" stroke="${colors.green}" stroke-width="2" stroke-dasharray="7 7" />${dots}<text x="${w / 2 - 60}" y="${h - 4}" class="axis-label">예측 확률</text>`);
}

function legend(series) {
  return `<div class="legend">${series.map((s) => `<span class="legend-item"><span class="legend-swatch" style="background:${s.color}"></span>${escapeHtml(s.label)}</span>`).join("")}</div>`;
}

function renderTable(target, rows, columns = null, limit = 100) {
  const container = typeof target === "string" ? $(target) : target;
  if (!container) return;
  const data = (rows || []).slice(0, limit);
  if (!data.length) {
    container.innerHTML = `<div class="preview-empty">데이터 없음</div>`;
    return;
  }
  const cols = (columns || Object.keys(data[0])).filter((col) => data.some((row) => row[col] !== undefined));
  const dense = cols.length >= 9 ? " class=\"dense\"" : "";
  container.innerHTML = `<table${dense}><thead><tr>${cols.map((c) => `<th>${escapeHtml(columnLabels[c] || fallbackLabel(c))}</th>`).join("")}</tr></thead><tbody>${data.map((row) => `<tr>${cols.map((c) => `<td>${formatCell(c, row[c])}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}

function formatCell(key, value) {
  if (value === null || value === undefined || value === "") return "";
  const lower = String(key).toLowerCase();
  if (lower === "raw_field" || lower === "raw_value") {
    const text = String(value);
    return `<span class="truncate" title="${escapeHtml(text)}">${escapeHtml(text)}</span>`;
  }
  if (lower.endsWith("url") || lower.endsWith("urls") || lower.includes("_url")) {
    const text = String(value);
    const firstUrl = text.split(/[|,\s]+/).find((part) => /^https?:\/\//i.test(part));
    if (firstUrl) {
      return `<a class="table-link" href="${escapeHtml(firstUrl)}" target="_blank" rel="noopener noreferrer">열기</a>`;
    }
    return `<span class="truncate" title="${escapeHtml(text)}">${escapeHtml(text)}</span>`;
  }
  if (lower === "date" || lower.endsWith("_date") || lower.endsWith("_at") || lower.includes("_date_")) return escapeHtml(shortDate(value));
  if (["open", "high", "low", "close", "entry_price", "exit_price", "implied_price", "latest_close", "start_close_usd", "end_close_usd"].includes(lower)) {
    const n = toNumber(value);
    if (n !== null) return escapeHtml(fmtCurrency(n));
  }
  if (lower === "horizon_days") {
    const n = toNumber(value);
    if (n !== null) return `${escapeHtml(fmtNumber(n, 0))}일`;
  }
  if (lower.startsWith("label_") && !lower.includes("return") && !lower.includes("expected")) {
    const n = toNumber(value);
    if (n !== null && (n === 0 || n === 1)) return n === 1 ? "예" : "아니오";
  }
  if (lower.startsWith("p_") || lower.includes("_p_") || lower.startsWith("trade_ready_p") || lower.startsWith("trigger_p") || lower.startsWith("context_p")) {
    const n = toNumber(value);
    if (n !== null) return escapeHtml(fmtPct(n, 1, Math.abs(n) <= 1));
  }
  if (lower === "fold_id" || lower === "fold_count" || lower.endsWith("_count") || lower === "trade_count" || lower === "year" || lower === "rows" || lower === "days" || lower === "observations" || lower === "trial_count" || lower === "trading_days" || lower === "window_id") {
    const n = toNumber(value);
    if (n !== null) return escapeHtml(fmtNumber(n, 0));
  }
  if (lower.includes("probability") || lower.includes("success_rate") || lower.includes("observed_success_rate") || lower.includes("positive_rate") || lower.startsWith("mean_p_")) {
    const n = toNumber(value);
    if (n !== null) return escapeHtml(fmtPct(n, 1, Math.abs(n) <= 1));
  }
  if (lower.startsWith("hist_news_category_mean_return")) {
    const n = toNumber(value);
    if (n !== null) return escapeHtml(fmtPct(n, 1, true));
  }
  if (
    lower.includes("expected_r") ||
    lower.includes("expectancy_r") ||
    lower === "r_multiple" ||
    lower === "threshold" ||
    lower === "score" ||
    lower === "rank_score"
  ) {
    const n = toNumber(value);
    if (n !== null) return escapeHtml(fmtNumber(n, 3));
  }
  if (
    lower.includes("pct") ||
    lower.includes("return") ||
    lower.includes("drawdown") ||
    lower.includes("rate") ||
    lower.includes("max_weight") ||
    lower.includes("limit_weight") ||
    lower.includes("position_weight") ||
    lower === "target_weight"
  ) {
    const n = toNumber(value);
    if (n !== null) return escapeHtml(fmtPct(n, 1));
  }
  if (["brier_score", "log_loss", "pr_auc", "ece", "profit_factor", "sharpe", "cagr", "mdd", "weight", "pbo_proxy", "observed_sharpe", "probabilistic_sharpe_ratio", "deflated_sharpe_ratio", "deflated_benchmark_sharpe", "skew", "kurtosis"].includes(lower)) {
    const n = toNumber(value);
    if (n !== null) return escapeHtml(fmtNumber(n, 3));
  }
  if (typeof value === "number") return escapeHtml(fmtNumber(value, 2));
  const text = translateReportText(labelValue(value));
  return `<span class="truncate" title="${escapeHtml(text)}">${escapeHtml(text)}</span>`;
}

function renderImageGallery(images) {
  const selected = images.filter((img) => img.path.includes("/charts/")).slice(0, 12);
  const gallery = $("imageGallery");
  if (!selected.length) {
    gallery.innerHTML = `<div class="preview-empty">이미지 없음</div>`;
    return;
  }
  gallery.innerHTML = selected.map((img) => `<article class="image-tile"><button type="button" data-image="${escapeHtml(img.path)}"><img src="${img.url}" alt="${escapeHtml(fileDisplayName(img))}" loading="lazy" /><span>${escapeHtml(fileDisplayName(img))}</span></button></article>`).join("");
  gallery.querySelectorAll("button[data-image]").forEach((button) => {
    button.addEventListener("click", () => {
      activateView("outputs");
      selectFile(button.dataset.image);
    });
  });
}

function renderFiles(data) {
  renderPageSynthesis("outputsSynthesis", deriveOutputsSynthesis(data));
  const search = $("fileSearch").value.trim().toLowerCase();
  const scope = $("fileScope").value;
  const kind = $("fileKind").value;
  const files = (data.files || []).filter((file) => (!search || `${file.path} ${fileDisplayName(file)}`.toLowerCase().includes(search)) && (!scope || file.scope === scope) && (!kind || file.kind === kind));
  const activeVisible = files.some((file) => file.path === state.activeFile);
  $("fileCount").textContent = `${files.length}`;
  $("fileList").innerHTML = files
    .map((file) => {
      const meta = [
        file.path,
        scopeLabel(file.scope),
        file.source_dir,
        formatBytes(file.size_bytes),
        file.rows !== null ? `${file.rows}행` : "",
      ].filter(Boolean).join(" · ");
      return `<button class="file-item ${state.activeFile === file.path ? "active" : ""}" type="button" data-file="${escapeHtml(file.path)}"><span><strong>${escapeHtml(fileDisplayName(file))}</strong><span>${escapeHtml(meta)}</span></span><span class="kind">${escapeHtml(kindLabel(file.kind))}</span></button>`;
    })
    .join("");
  $("fileList").querySelectorAll("button[data-file]").forEach((button) => button.addEventListener("click", () => selectFile(button.dataset.file)));
  if (!state.activeFile || !activeVisible) {
    const defaultFile = files.find((file) => file.path.endsWith("tsm_prediction_reliability_report.md")) || files.find((file) => file.path.endsWith("tsm_daily_trading_plan.md")) || files[0] || null;
    if (defaultFile) {
      selectFile(defaultFile.path);
    } else {
      state.activeFile = null;
      $("previewTitle").textContent = "미리보기";
      $("downloadLink").href = "#";
      $("filePreview").innerHTML = `<div class="preview-empty">파일 없음</div>`;
    }
  }
}

function renderRunPage(data) {
  renderPageSynthesis("runSynthesis", deriveRunSynthesis(data || state.data || {}, state.lastRun || state.data?.run || {}));
}

function formatBytes(bytes) {
  const n = toNumber(bytes);
  if (n === null) return "없음";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

async function selectFile(path) {
  state.activeFile = path;
  const file = (state.data?.files || []).find((item) => item.path === path);
  if (!file) return;
  $("previewTitle").textContent = fileDisplayName(file);
  $("downloadLink").href = file.url;
  $("fileList").querySelectorAll(".file-item").forEach((item) => item.classList.toggle("active", item.dataset.file === path));
  const preview = $("filePreview");
  preview.innerHTML = `<div class="preview-empty">로딩 중</div>`;
  try {
    if (file.kind === "csv") {
      const table = await fetchJson(`/api/table?file=${encodeURIComponent(path)}&limit=500&tail=true`);
      renderTable(preview, table.rows, table.columns, 500);
    } else if (file.kind === "report") {
      const report = await fetchJson(`/api/report?file=${encodeURIComponent(path)}`);
      preview.innerHTML = `<div class="markdown">${renderMarkdown(report.content)}</div>`;
    } else if (file.kind === "image") {
      preview.innerHTML = `<img src="${file.url}" alt="${escapeHtml(fileDisplayName(file))}" />`;
    } else {
      preview.innerHTML = `<div class="preview-empty">미리보기 없음</div>`;
    }
  } catch (error) {
    preview.innerHTML = `<div class="preview-empty">${escapeHtml(error.message)}</div>`;
  }
}

function renderMarkdown(markdown) {
  const lines = String(markdown || "").split(/\r?\n/);
  const html = [];
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    if (!line.trim()) continue;
    if (line.startsWith("|") && lines[i + 1]?.includes("---")) {
      const tableLines = [];
      while (lines[i]?.startsWith("|")) {
        tableLines.push(lines[i]);
        i += 1;
      }
      i -= 1;
      html.push(markdownTable(tableLines));
    } else if (line.startsWith("### ")) {
      html.push(`<h3>${escapeHtml(translateReportText(line.slice(4)))}</h3>`);
    } else if (line.startsWith("## ")) {
      html.push(`<h2>${escapeHtml(translateReportText(line.slice(3)))}</h2>`);
    } else if (line.startsWith("# ")) {
      html.push(`<h1>${escapeHtml(translateReportText(line.slice(2)))}</h1>`);
    } else if (line.startsWith("- ")) {
      html.push(`<p>${escapeHtml(translateReportText(line.slice(2)))}</p>`);
    } else {
      html.push(`<p>${escapeHtml(translateReportText(line))}</p>`);
    }
  }
  return html.join("");
}

function markdownTable(lines) {
  const rows = lines.filter((line) => !/^\|\s*-+/.test(line)).map((line) => line.split("|").slice(1, -1).map((cell) => translateReportText(labelValue(cell.trim()))));
  if (!rows.length) return "";
  const [head, ...body] = rows;
  return `<table><thead><tr>${head.map((cell) => `<th>${escapeHtml(cell)}</th>`).join("")}</tr></thead><tbody>${body.map((row) => `<tr>${row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}

function activateView(name) {
  state.activeView = name;
  $("pageTitle").textContent = pageTitles[name] || "대시보드";
  document.querySelectorAll(".nav-item, .view-tab").forEach((item) => item.classList.toggle("active", item.dataset.view === name));
  document.querySelectorAll(".view").forEach((view) => view.classList.toggle("active", view.id === `view-${name}`));
  window.scrollTo(0, 0);
}

function formPayload(form) {
  const payload = Object.fromEntries(new FormData(form).entries());
  payload.skip_charts = form.elements.skip_charts.checked;
  payload.skip_benchmarks = form.elements.skip_benchmarks.checked;
  payload.schema_strict = form.elements.schema_strict.checked;
  payload.continue_on_error = form.elements.continue_on_error.checked;
  payload.skip_symbol_build = Boolean(form.elements.skip_symbol_build?.checked);
  payload.skip_news_refresh = Boolean(form.elements.skip_news_refresh?.checked);
  return payload;
}

async function startRun(event) {
  event.preventDefault();
  $("runBtn").disabled = true;
  try {
    const run = await fetchJson("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formPayload(event.currentTarget)),
    });
    updateRunState(run);
  } catch (error) {
    alert(error.message);
  } finally {
    $("runBtn").disabled = false;
  }
}

async function stopRun() {
  try {
    updateRunState(await fetchJson("/api/stop", { method: "POST" }));
  } catch (error) {
    alert(error.message);
  }
}

async function pollRun() {
  try {
    const run = await fetchJson("/api/run-status");
    const wasRunning = state.lastRunStatus === "RUNNING";
    updateRunState(run);
    if (wasRunning && run.status !== "RUNNING") await loadData();
  } catch {
    $("runStatusText").textContent = "실행 상태를 읽지 못했습니다";
  }
}

function updateRunState(run) {
  const status = run.status || "IDLE";
  state.lastRun = run;
  state.lastRunStatus = status;
  const pill = $("runPill");
  pill.textContent = runStatusLabels[status] || status;
  pill.className = "status-pill";
  if (status === "RUNNING") pill.classList.add("running");
  if (status === "PASS") pill.classList.add("pass");
  if (status === "FAIL") pill.classList.add("fail");
  $("runStatusText").textContent = `${run.mode_label || run.mode || "대기"} · ${runStatusLabels[status] || status}${run.duration_sec ? ` · ${run.duration_sec}초` : ""}`;
  $("runLog").textContent = (run.log || []).join("\n");
  $("stopBtn").disabled = status !== "RUNNING";
  if (state.data) renderRunPage(state.data);
}

function setTheme(theme) {
  document.body.classList.toggle("dark", theme === "dark");
  localStorage.setItem(THEME_KEY, theme);
  $("themeToggle").querySelector("span").textContent = theme === "dark" ? "라이트모드" : "다크모드";
}

function wireEvents() {
  document.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => activateView(button.dataset.view)));
  $("refreshBtn").addEventListener("click", loadData);
  $("themeToggle").addEventListener("click", () => setTheme(document.body.classList.contains("dark") ? "light" : "dark"));
  $("fileSearch").addEventListener("input", () => renderFiles(state.data || { files: [] }));
  $("fileScope").addEventListener("change", () => renderFiles(state.data || { files: [] }));
  $("fileKind").addEventListener("change", () => renderFiles(state.data || { files: [] }));
  $("runForm").addEventListener("submit", startRun);
  $("stopBtn").addEventListener("click", stopRun);
  $("endInput").value = todayKstInputValue();
  setTheme(localStorage.getItem(THEME_KEY) || "dark");
  setInterval(pollRun, 2000);
}

document.addEventListener("DOMContentLoaded", async () => {
  wireEvents();
  try {
    await loadData();
  } catch (error) {
    $("asOfText").textContent = error.message;
    console.error(error);
  }
});
