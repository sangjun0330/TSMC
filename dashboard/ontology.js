(function () {
  "use strict";

  const STATUS_LABELS = {
    PASS: "문제 없음",
    WARN: "주의 필요",
    FAIL: "문제 있음",
    BLOCKED: "차단",
    READY: "준비됨",
    NOT_READY: "준비 안 됨",
    DISPLAY_ONLY: "참고용",
    DISPLAY_ONLY_MODEL_BLOCKED: "모델 기준 미달",
    DISPLAY_ONLY_NO_LATEST_TRADE_READY: "최신 trade-ready 아님",
    DECISION_SUPPORT_ALLOWED: "판단 참고 가능",
    PAPER_DECISION_SUPPORT_ALLOWED: "Paper 판단 가능",
    LIVE_DISABLED_BY_DESIGN: "설계상 비활성",
    NO_LIVE_BROKER_BY_DESIGN: "실거래 브로커 미사용 설계",
    NO_TRADE: "거래 없음",
    WATCH_ONLY: "관찰만",
    WATCHLIST_ONLY: "관찰 전용",
    STRICT_NO_ENTRY: "엄격 기준 신규 진입 없음",
    EARLY_BULLISH_WATCH: "조기 상승 관찰 강화",
    PAPER_BUY_SETUP: "공격형 Paper 후보",
    PAPER_TRACK_LONG_SETUP: "Paper 전용 추적",
    NO_RESEARCH_SIGNAL: "연구 신호 없음",
    NONE: "없음",
    NO_ENTRY_TRIGGER: "트리거 없음",
    NO_SIGNAL: "신호 없음",
    NO_HIGH_CONFIDENCE_NEWS: "고신뢰 원인 없음",
    NO_MATCH: "매칭 없음",
    DIRECT_WEB_OK: "직접 수집 정상",
    DIRECT_WEB_PARTIAL: "직접 수집 일부",
    DIRECT_WEB_EMPTY: "수집 결과 없음",
    DIRECT_WEB_FAILED: "직접 수집 실패",
    INSUFFICIENT_OOS_EVIDENCE: "OOS 근거 부족",
    NOT_20D_TRADE_READY_DECISION_SCOPE: "20D 판단 후보 아님",
    SELECTED_EVENTS_PER_FOLD_LT_MIN: "fold별 선택 사례 부족",
    SELECTED_OOS_EVENT_COUNT_LT_MIN: "선택 OOS 사례 부족",
    PREDICTION_QUALITY_FALSE: "예측 품질 미통과",
  };

  function buildModel(data) {
    const ctx = buildContext(data || {});
    const metrics = computeMetrics(ctx);
    const permission = derivePermission(ctx, metrics);
    const blockers = rankBlockers(ctx, metrics);
    const contradictions = detectContradictions(ctx, metrics);
    metrics.ontologyConflictCount = contradictions.length;
    const objects = buildObjects(ctx, metrics, permission, blockers);
    const edges = buildEdges(ctx, metrics);
    const actionPlan = deriveActionPlan(ctx, metrics, permission, blockers, contradictions);
    const insights = buildInsights(ctx, metrics, permission, blockers, contradictions, actionPlan);
    const lineage = buildLineage(ctx, metrics);
    const executionLadder = buildExecutionLadder(ctx, metrics, permission);

    return {
      asOf: shortDate(ctx.asOfDate),
      symbol: (data.meta && (data.meta.symbol_display_name || data.meta.symbol_name || data.meta.symbol)) || "",
      groupLabel: (data.meta && data.meta.group_label) || "",
      generatedAt: data.generated_at || "",
      objects,
      edges,
      metrics,
      permission,
      insights,
      contradictions,
      blockers,
      actionPlan,
      lineage,
      executionLadder,
      graph: buildGraph(objects, edges, actionPlan, metrics),
    };
  }

  function buildContext(data) {
    const snapshots = data.snapshots || {};
    const tables = data.tables || {};
    const series = data.series || {};
    const quality = data.quality || {};
    const decision = snapshots.decision || {};
    const risk = snapshots.risk || {};
    const prediction = snapshots.prediction || {};
    const pooled = snapshots.pooled_prediction || {};
    const system = snapshots.system || {};
    const dataQuality = snapshots.data_quality || {};
    const dailyHealth = snapshots.daily_health || {};
    const modelGate = snapshots.model_gate || {};
    const paperGate = snapshots.paper_gate || {};
    const orderIntent = snapshots.order_intent || {};
    const portfolioRisk = snapshots.portfolio_risk || {};
    const paperPosition = snapshots.paper_position || {};
    const paperReconciliation = snapshots.paper_reconciliation || {};
    const orderLifecycle = snapshots.order_lifecycle || {};
    const executionFeedback = snapshots.execution_feedback || {};
    const fillModelCalibration = snapshots.fill_model_calibration || {};
    const automation = snapshots.automation || {};
    const latestNews = snapshots.latest_news || {};
    const stress = snapshots.stress || {};
    const integrated = snapshots.integrated_price || snapshots.summary || {};
    const latestPrice = snapshots.latest_price || {};
    const universe = snapshots.universe_validation || {};
    const sample = snapshots.pooled_sample_audit || {};
    const research = snapshots.research_expansion || {};
    const planRows = tables.trading_plan || [];
    const pSuccess = probabilityPercent(firstValue(pooled, ["p_success_20d", "p_success_tsm_like_20d", "decision_score_20d"], prediction, ["trade_ready_p_success_20d", "trigger_p_success_20d", "context_p_success_20d", "p_success_20d"]));
    const pStop = probabilityPercent(firstValue(pooled, ["p_stop_hit_calibrated_20d", "p_stop_hit_20d"], prediction, ["trade_ready_p_stop_hit_20d", "trigger_p_stop_hit_20d", "context_p_stop_hit_20d", "p_stop_hit_20d"]));
    const threshold = probabilityPercent(firstValue(pooled, ["threshold_20d", "threshold"], prediction, ["trade_ready_threshold_20d", "trigger_threshold_20d", "context_threshold_20d", "threshold_20d"]));
    const expectedR = number(firstValue(pooled, ["expected_r_net_20d"], prediction, ["trade_ready_expected_r_20d", "trigger_expected_r_20d", "context_expected_r_20d", "expected_r_20d"]));
    const expectedNet = number(firstValue(pooled, ["expected_net_return_pct_20d"], prediction, ["trade_ready_expected_net_return_20d", "trigger_expected_net_return_20d", "context_expected_net_return_20d", "expected_net_return_20d"]));
    const maxWeight = weight(firstValue(risk, ["final_recommended_max_weight", "final_recommended_max_weight_pct"], system, ["final_recommended_max_weight", "final_recommended_max_weight_pct"]));
    const maxWeightPct = maxWeight === null ? null : maxWeight * 100;
    const entryTrigger = text(decision.entry_trigger || prediction.latest_entry_gate_status || "NONE").toUpperCase();
    const hasTrigger = Boolean(entryTrigger && !["NONE", "NO_ENTRY_TRIGGER", "NO_SIGNAL", "FALSE"].includes(entryTrigger));
    const score = number(decision.score_price_algo_total ?? risk.score_price_algo_total);
    const asOfDate = decision.date || prediction.prediction_asof_date || pooled.asof_date || dataQuality.latest_signal_date || latestPrice.date || integrated.end_date;
    const displayCurrency = inferDisplayCurrency(data, latestPrice, decision, risk);
    const fxRateToUsd = inferFxRateToUsd(latestPrice, decision, risk, integrated);

    return {
      data,
      snapshots,
      tables,
      series,
      quality,
      decision,
      risk,
      prediction,
      pooled,
      system,
      dataQuality,
      dailyHealth,
      modelGate,
      paperGate,
      orderIntent,
      portfolioRisk,
      paperPosition,
      paperReconciliation,
      orderLifecycle,
      executionFeedback,
      fillModelCalibration,
      automation,
      latestNews,
      stress,
      integrated,
      latestPrice,
      universe,
      sample,
      research,
      planRows,
      pSuccess,
      pStop,
      threshold,
      expectedR,
      expectedNet,
      maxWeight,
      maxWeightPct,
      entryTrigger,
      hasTrigger,
      score,
      asOfDate,
      dataAgeDays: dateAgeDays(dataQuality.latest_signal_date || asOfDate),
      predictionAllowed: bool(system.prediction_decision_support) || bool(pooled.decision_support_allowed) || text(prediction.prediction_use_status).includes("DECISION_SUPPORT_ALLOWED"),
      paperAllowed: bool(paperGate.paper_decision_support_allowed)
        || text(paperGate.paper_gate_status).toUpperCase().includes("PAPER_DECISION_SUPPORT_ALLOWED")
        || text(paperGate.paper_gate_status).toUpperCase().includes("PAPER_READY_ENTRY"),
      strictAllowed: bool(paperGate.strict_decision_support_allowed),
      liveStatus: system.live_trading_status || "LIVE_DISABLED_BY_DESIGN",
      displayCurrency,
      fxRateToUsd,
    };
  }

  function computeMetrics(ctx) {
    const dataTrust = computeDataTrust(ctx);
    const ruleSignalStrength = computeRuleSignalStrength(ctx);
    const riskCapacityScore = computeRiskCapacity(ctx);
    const predictionPermissionEdge = computePredictionPermissionEdge(ctx);
    const validationTrustScore = computeValidationTrust(ctx);
    const stressTolerance = computeStressTolerance(ctx);
    const newsDragScore = computeNewsDrag(ctx);
    const newsClarity = clamp(100 - newsDragScore, 0, 100);
    const operationalLineage = computeOperationalLineage(ctx);
    const intradayConfirmationScore = computeIntradayConfirmation(ctx);
    const crossSectionalSupport = computeCrossSectionalSupport(ctx);
    const tsmLikeSampleStrength = computeTsmLikeSampleStrength(ctx);
    const regimeFitScore = computeRegimeFit(ctx);
    const overfitRiskLevel = computeOverfitRisk(ctx);
    const paperToLiveGap = computePaperToLiveGap(ctx);
    const shadowRealizationScore = computeShadowRealization(ctx);
    const pipelineFreshnessScore = computePipelineFreshness(ctx);
    const paperOmsScore = average([
      statusScoreValue(ctx.orderIntent.status || "MISSING"),
      statusScoreValue(ctx.portfolioRisk.portfolio_risk_status || "MISSING"),
      statusScoreValue(ctx.paperReconciliation.status || "MISSING"),
      statusScoreValue(ctx.paperPosition.position_state || "CASH"),
      statusScoreValue(ctx.orderLifecycle.lifecycle_state || "MISSING"),
      statusScoreValue(ctx.fillModelCalibration.calibration_status || "MISSING"),
    ]) || 50;
    const predictionEdge = ctx.pSuccess === null || ctx.threshold === null ? null : ctx.pSuccess - ctx.threshold;
    const stopAdjustedEdge = predictionEdge === null ? null : predictionEdge - (ctx.pStop || 0) * 0.18;
    const expectedTradeQuality = computeExpectedTradeQuality(ctx, predictionPermissionEdge);
    const opportunityPressure = computeOpportunityPressure(ctx, {
      predictionPermissionEdge,
      newsClarity,
      intradayConfirmationScore,
      crossSectionalSupport,
    });
    const blockPressure = computeBlockPressure(ctx, {
      riskCapacityScore,
      dataTrust,
      validationTrustScore,
      stressTolerance,
      newsDragScore,
      pipelineFreshnessScore,
    });
    const todayActionabilityScore = Math.round(clamp(
      dataTrust * 0.18 +
        ruleSignalStrength * 0.18 +
        riskCapacityScore * 0.16 +
        predictionPermissionEdge * 0.16 +
        validationTrustScore * 0.12 +
        stressTolerance * 0.08 +
        newsClarity * 0.06 +
        operationalLineage * 0.06,
      0,
      100
    ));

    return {
      todayActionabilityScore,
      opportunityPressure: Math.round(opportunityPressure),
      blockPressure: Math.round(blockPressure),
      dataTrust: Math.round(dataTrust),
      ruleSignalStrength: Math.round(ruleSignalStrength),
      riskCapacityScore: Math.round(riskCapacityScore),
      predictionPermissionEdge: Math.round(predictionPermissionEdge),
      validationTrustScore: Math.round(validationTrustScore),
      stressTolerance: Math.round(stressTolerance),
      operationalLineage: Math.round(operationalLineage),
      predictionEdge,
      stopAdjustedEdge,
      expectedTradeQuality: Math.round(expectedTradeQuality),
      overfitRiskLevel,
      regimeFitScore: Math.round(regimeFitScore),
      newsDragScore: Math.round(newsDragScore),
      causeClarityScore: Math.round(newsClarity),
      intradayConfirmationScore: Math.round(intradayConfirmationScore),
      crossSectionalSupport: Math.round(crossSectionalSupport),
      tsmLikeSampleStrength: Math.round(tsmLikeSampleStrength),
      paperToLiveGap: Math.round(paperToLiveGap),
      shadowRealizationScore: Math.round(shadowRealizationScore),
      pipelineFreshnessScore: Math.round(pipelineFreshnessScore),
      paperOmsScore: Math.round(paperOmsScore),
      ontologyConflictCount: 0,
    };
  }

  function computeDataTrust(ctx) {
    const statusScore = statusScoreValue(ctx.dataQuality.data_quality_status || ctx.quality.data_quality?.status || "MISSING");
    const qualityScores = [
      qualityPassRate(ctx.quality.data_quality),
      qualityPassRate(ctx.quality.integrity),
      qualityPassRate(ctx.quality.schema),
      qualityPassRate(ctx.quality.data_contract),
      qualityPassRate(ctx.quality.operational),
    ].filter((value) => value !== null);
    const passRate = qualityScores.length ? average(qualityScores) : statusScore;
    const failedChecks = number(ctx.dataQuality.failed_checks) || 0;
    const agePenalty = ctx.dataAgeDays === null ? 12 : ctx.dataAgeDays > 7 ? 24 : ctx.dataAgeDays > 3 ? 10 : 0;
    return clamp(statusScore * 0.45 + passRate * 0.55 - failedChecks * 6 - agePenalty, 0, 100);
  }

  function computeRuleSignalStrength(ctx) {
    const score = clamp(ctx.score ?? 0, 0, 100);
    const strict = text(ctx.decision.strict_signal_stage).toUpperCase();
    const research = text(ctx.decision.research_signal_stage).toUpperCase();
    let bonus = 0;
    if (ctx.hasTrigger) bonus += 18;
    if (strict.includes("ENTRY") && !strict.includes("NO_ENTRY")) bonus += 12;
    if (research.includes("PAPER_BUY")) bonus += 9;
    if (research.includes("WATCH")) bonus += 5;
    return clamp(score + bonus, 0, 100);
  }

  function computeRiskCapacity(ctx) {
    if (ctx.maxWeightPct === null) return 38;
    const base = clamp((ctx.maxWeightPct / 12) * 100, 0, 100);
    const state = text(ctx.risk.risk_state || ctx.system.risk_state).toUpperCase();
    const reason = text(ctx.risk.limiting_reason).toUpperCase();
    let penalty = 0;
    if (state.includes("NO_NEW_RISK") || state.includes("UNKNOWN")) penalty += 32;
    if (state.includes("WAIT")) penalty += 14;
    if (reason.includes("SCORE") || reason.includes("TRIGGER")) penalty += 14;
    if (reason.includes("DRAWDOWN") || reason.includes("VOL")) penalty += 10;
    return clamp(base - penalty, 0, 100);
  }

  function computePredictionPermissionEdge(ctx) {
    let score = ctx.predictionAllowed ? 68 : 36;
    if (ctx.pSuccess !== null && ctx.threshold !== null) {
      score = 50 + (ctx.pSuccess - ctx.threshold) * 2.2;
    } else if (ctx.pSuccess !== null) {
      score = ctx.pSuccess;
    }
    if (ctx.expectedR !== null) score += ctx.expectedR * 8;
    if (ctx.pStop !== null) score -= Math.max(0, ctx.pStop - 35) * 0.75;
    if (!ctx.predictionAllowed) score = Math.min(score, 54);
    return clamp(score, 0, 100);
  }

  function computeValidationTrust(ctx) {
    const qualityScore = qualityPassRate(ctx.quality.validation) ?? statusScoreValue("WARN");
    const modelGateScore = qualityPassRate(ctx.quality.model_gate);
    const walk = positiveRate(ctx.tables.validation_walk_forward || [], "test_positive");
    const causal = positiveRate(ctx.tables.validation_causal_walk_forward || [], "test_positive");
    const cpcvStrategy = positiveRate(ctx.tables.cpcv_strategy_distribution || [], "cpcv_median_uplift_pass");
    const cpcvModel = positiveRate(ctx.tables.cpcv_model_distribution || [], "cpcv_model_median_uplift_pass");
    const dsr = positiveRate(ctx.tables.deflated_sharpe || [], "dsr_pass");
    const pieces = [qualityScore, walk, causal, cpcvStrategy, cpcvModel, dsr].filter((value) => value !== null);
    let score = pieces.length ? average(pieces) : 45;
    if (modelGateScore !== null) score = score * 0.78 + modelGateScore * 0.22;
    if (!ctx.predictionAllowed) score = Math.min(score, 62);
    return clamp(score, 0, 100);
  }

  function computeStressTolerance(ctx) {
    const impact = Math.abs(number(ctx.stress.worst_current_weight_portfolio_impact_pct) ?? number(ctx.stress.worst_current_weight_impact_pct) ?? 0);
    const stressStatus = text(ctx.system.stress_status || ctx.stress.stress_status).toUpperCase();
    let score = clamp(100 - impact * 8, 0, 100);
    if (stressStatus.includes("FAIL") || stressStatus.includes("HIGH")) score -= 25;
    if (stressStatus.includes("WARN")) score -= 12;
    return clamp(score, 0, 100);
  }

  function computeNewsDrag(ctx) {
    const penalty = Math.abs(number(ctx.latestNews.news_penalty_event) || 0);
    const confidence = text(ctx.latestNews.news_match_confidence).toUpperCase();
    const events = number(ctx.latestNews.news_event_count_3d ?? ctx.latestNews.news_source_count) || 0;
    let drag = penalty * 18 + Math.min(events * 3, 18);
    if (confidence === "HIGH") drag += 18;
    if (confidence === "MEDIUM") drag += 8;
    const cause = text(ctx.latestNews.news_primary_cause_type).toUpperCase();
    if (cause.includes("NEGATIVE") || cause.includes("RISK") || cause.includes("EXPORT")) drag += 12;
    return clamp(drag, 0, 100);
  }

  function computeOperationalLineage(ctx) {
    const manifest = ctx.tables.manifest || [];
    if (!manifest.length) return statusScoreValue(ctx.dailyHealth.daily_health_status || ctx.system.system_state || "WARN");
    const passRate = positiveRate(manifest, "status", "PASS") ?? 50;
    const failed = manifest.filter((row) => text(row.status).toUpperCase() === "FAIL").length;
    return clamp(passRate - failed * 8, 0, 100);
  }

  function computeIntradayConfirmation(ctx) {
    const dailyClose = number(ctx.decision.close || ctx.latestPrice.close);
    const hourlyClose = number(ctx.snapshots.latest_hourly?.close);
    const minuteClose = number(ctx.snapshots.latest_minute?.close);
    const hourlyAge = dateAgeDays(ctx.snapshots.latest_hourly?.date || ctx.snapshots.latest_hourly?.timestamp);
    const minuteAge = dateAgeDays(ctx.snapshots.latest_minute?.date || ctx.snapshots.latest_minute?.timestamp);
    let score = 48;
    if (dailyClose && hourlyClose) score += Math.abs(hourlyClose - dailyClose) / dailyClose < 0.02 ? 18 : 4;
    if (dailyClose && minuteClose) score += Math.abs(minuteClose - dailyClose) / dailyClose < 0.02 ? 14 : 3;
    if (hourlyAge !== null && hourlyAge <= 5) score += 8;
    if (minuteAge !== null && minuteAge <= 5) score += 6;
    return clamp(score, 0, 100);
  }

  function computeCrossSectionalSupport(ctx) {
    const allowed = bool(ctx.pooled.decision_support_allowed);
    const eventCount = number(ctx.pooled.decision_event_count || ctx.pooled.oos_event_count || ctx.sample.trade_ready_20d_labeled) || 0;
    const selected = number(ctx.pooled.selected_oos_event_count || ctx.sample.selected_oos_event_count) || 0;
    const modelPass = bool(ctx.pooled.model_quality_pass);
    let score = clamp(Math.log10(eventCount + 1) * 30 + Math.log10(selected + 1) * 18, 0, 85);
    if (allowed) score += 12;
    if (modelPass) score += 8;
    return clamp(score, 0, 100);
  }

  function computeTsmLikeSampleStrength(ctx) {
    const groupN = number(ctx.pooled.effective_group_n || ctx.pooled.tsm_like_effective_n || ctx.sample.tsm_like_effective_n) || 0;
    const routePass = bool(ctx.pooled.tsm_calibration_route_pass) || bool(ctx.pooled.route_selection_pass) || bool(ctx.paperGate.tsm_like_route_selection_pass);
    let score = clamp(Math.log10(groupN + 1) * 34, 0, 85);
    if (routePass) score += 15;
    return clamp(score, 0, 100);
  }

  function computeRegimeFit(ctx) {
    const trend = text(ctx.decision.trend_regime || ctx.risk.trend_regime).toUpperCase();
    const vol = text(ctx.decision.vol_regime || ctx.risk.vol_regime).toUpperCase();
    const drawdown = Math.abs(number(ctx.decision.drawdown_from_ath ?? ctx.risk.drawdown_from_ath) || 0);
    const ruleRows = ctx.tables.rule_forward || [];
    const regimeRows = ctx.tables.regime_forward || [];
    const avgForward = average(
      [...ruleRows, ...regimeRows]
        .map((row) => number(row.mean_forward_return_20d_pct ?? row.mean_return_20d_pct ?? row.forward_return_20d_mean_pct))
        .filter((value) => value !== null)
    );
    let score = 52;
    if (trend.includes("UP") || trend.includes("BULL") || trend.includes("ABOVE")) score += 14;
    if (vol.includes("NORMAL") || vol.includes("LOW")) score += 8;
    if (vol.includes("HIGH") || vol.includes("EXTREME")) score -= 10;
    if (drawdown > 0.25 || drawdown > 25) score -= 12;
    if (avgForward !== null) score += clamp(avgForward * 3, -16, 18);
    return clamp(score, 0, 100);
  }

  function computeOverfitRisk(ctx) {
    const pboRows = ctx.tables.pbo_report || [];
    const cscvRows = ctx.tables.cscv_pbo_report || [];
    const dsrRows = ctx.tables.deflated_sharpe || [];
    const pboWarn = pboRows.some((row) => bool(row.overfit_warning) || (number(row.pbo_proxy) || 0) > 0.2);
    const cscvWarn = cscvRows.some((row) => (number(row.pbo_proxy) || number(row.pbo) || 0) > 0.2);
    const dsrPassRate = positiveRate(dsrRows, "dsr_pass");
    if (pboWarn || cscvWarn || (dsrPassRate !== null && dsrPassRate < 35)) return "HIGH";
    if (dsrPassRate !== null && dsrPassRate < 65) return "MEDIUM";
    return "LOW";
  }

  function computePaperToLiveGap(ctx) {
    let gap = 100;
    if (ctx.paperAllowed) gap -= 36;
    if (ctx.strictAllowed) gap -= 28;
    if (ctx.predictionAllowed) gap -= 14;
    if (ctx.hasTrigger) gap -= 12;
    if (ctx.maxWeightPct !== null && ctx.maxWeightPct >= 5) gap -= 10;
    return clamp(gap, 0, 100);
  }

  function computeShadowRealization(ctx) {
    const rows = ctx.tables.shadow_predictions || [];
    const realized = rows.filter((row) => text(row.realized_status).toUpperCase().includes("REALIZED") || hasValue(row.realized_success));
    if (!realized.length) return 50;
    return positiveRate(realized, "realized_success") ?? 50;
  }

  function computePipelineFreshness(ctx) {
    const age = ctx.dataAgeDays;
    if (age === null) return 45;
    if (age <= 2) return 96;
    if (age <= 4) return 78;
    if (age <= 7) return 58;
    return 32;
  }

  function computeExpectedTradeQuality(ctx, predictionPermissionEdge) {
    const probabilityEdge = ctx.pSuccess !== null && ctx.threshold !== null ? clamp(50 + (ctx.pSuccess - ctx.threshold) * 2.4, 0, 100) : predictionPermissionEdge;
    const expectedRScore = ctx.expectedR === null ? 45 : clamp(50 + ctx.expectedR * 24, 0, 100);
    const stopPenalty = ctx.pStop === null ? 0 : clamp((ctx.pStop - 32) * 1.1, 0, 35);
    const overlay = bestOverlay(ctx.tables.ml_overlay_summary || []);
    const overlayUplift = overlay ? clamp(50 + (number(overlay.selected_minus_all_pct ?? overlay.mean_net_return_pct) || 0) * 5, 0, 100) : 50;
    const calibration = ctx.pooled.tsm_calibration_status || ctx.pooled.tsm_calibration_route_pass;
    const calibrationScore = bool(calibration) || text(calibration).toUpperCase().includes("READY") || text(calibration).toUpperCase().includes("PASS") ? 72 : 44;
    const gatePenalty = ctx.predictionAllowed ? 0 : 18;
    return clamp(probabilityEdge * 0.32 + expectedRScore * 0.24 + overlayUplift * 0.18 + calibrationScore * 0.16 + predictionPermissionEdge * 0.1 - stopPenalty - gatePenalty, 0, 100);
  }

  function computeOpportunityPressure(ctx, scores) {
    const triggerBonus = ctx.hasTrigger ? 100 : ctx.score !== null && ctx.score >= 60 ? 54 : 22;
    const trendRegime = trendScore(ctx.decision.trend_regime || ctx.risk.trend_regime);
    const rs = clamp(50 + (number(ctx.decision.relative_return_vs_spy_60d ?? ctx.latestPrice.relative_return_vs_spy_60d) || 0) * 2.2, 0, 100);
    return clamp(
      (ctx.score || 0) * 0.35 +
        triggerBonus * 0.2 +
        trendRegime * 0.15 +
        rs * 0.1 +
        scores.predictionPermissionEdge * 0.1 +
        scores.newsClarity * 0.05 +
        (scores.intradayConfirmationScore * 0.03 + scores.crossSectionalSupport * 0.02),
      0,
      100
    );
  }

  function computeBlockPressure(ctx, scores) {
    const modelGateBlockScore = modelGateBlocked(ctx) ? 86 : ctx.predictionAllowed ? 18 : 58;
    const riskBottleneckScore = 100 - scores.riskCapacityScore;
    const dataQualityFailScore = 100 - scores.dataTrust;
    const validationFailScore = 100 - scores.validationTrustScore;
    const stressScore = 100 - scores.stressTolerance;
    const stalePipelineScore = 100 - scores.pipelineFreshnessScore;
    return clamp(
      modelGateBlockScore * 0.22 +
        riskBottleneckScore * 0.2 +
        dataQualityFailScore * 0.18 +
        validationFailScore * 0.15 +
        stressScore * 0.1 +
        scores.newsDragScore * 0.07 +
        stalePipelineScore * 0.08,
      0,
      100
    );
  }

  function derivePermission(ctx, metrics) {
    const strictStage = text(ctx.decision.strict_signal_stage).toUpperCase();
    const researchStage = text(ctx.decision.research_signal_stage).toUpperCase();
    let state = "NO_SIGNAL";
    let title = "신호 없음";
    let tone = "neutral";

    if (metrics.dataTrust < 45 || statusTone(ctx.dataQuality.data_quality_status) === "bad") {
      state = "DATA_NOT_TRUSTED";
      title = "데이터 확인 먼저";
      tone = "bad";
    } else if (ctx.hasTrigger && ctx.strictAllowed && metrics.riskCapacityScore >= 60 && ctx.predictionAllowed) {
      state = "STRICT_LIVE_ENTRY_CANDIDATE";
      title = "엄격 기준 진입 후보";
      tone = "good";
    } else if (ctx.paperAllowed || researchStage.includes("PAPER_BUY")) {
      state = "PAPER_TRACKING_ALLOWED";
      title = "Paper 후보";
      tone = "neutral";
    } else if ((ctx.score || 0) >= 60 || researchStage.includes("WATCH")) {
      state = metrics.blockPressure > 62 ? "DECISION_SUPPORT_BLOCKED" : "WATCH_ONLY";
      title = metrics.blockPressure > 62 ? "근거 보강 대기" : "조건부 관찰 우위";
      tone = "warn";
    } else if (ctx.hasTrigger) {
      state = "WATCH_ONLY";
      title = "트리거 관찰";
      tone = "warn";
    }

    return {
      state,
      title,
      tone,
      live: "LIVE_DISABLED_BY_DESIGN",
      strict: strictStage || "STRICT_NO_ENTRY",
      paper: ctx.paperAllowed ? "PAPER_TRACKING_ALLOWED" : ctx.paperGate.paper_gate_status || "NOT_READY",
      prediction: ctx.predictionAllowed ? "DECISION_SUPPORT_ALLOWED" : ctx.prediction.prediction_use_status || ctx.modelGate.model_gate_status || "DISPLAY_ONLY",
      risk: ctx.risk.risk_state || ctx.system.risk_state || "UNKNOWN",
      data: ctx.dataQuality.data_quality_status || "MISSING",
      confidence: confidenceLevel(metrics, ctx),
    };
  }

  function buildObjects(ctx, metrics, permission, blockers) {
    const bestStrategy = bestBacktest(ctx.tables.backtest_summary || []);
    const bestOverlayRow = bestOverlay(ctx.tables.ml_overlay_summary || []);
    const primaryBlocker = blockers[0];
    const riskBottleneck = ctx.risk.limiting_reason || "NO_LIMITING_REASON";
    const pEdgeText = metrics.predictionEdge === null ? "없음" : `${fmtSigned(metrics.predictionEdge, 1)}%`;

    return [
      object("price_rule_today", "PriceRule", "가격·룰 객체", permission.state.includes("WATCH") ? "관찰" : label(ctx.decision.trade_action || "NO_TRADE"), scoreTone(ctx.score, ctx.hasTrigger), [
        ["Close", fmtDisplayCurrency(ctx, ctx.decision.close || ctx.latestPrice.close)],
        ["Score", `${fmtNumber(ctx.score, 1)} / 100`],
        ["Trigger", label(ctx.decision.entry_trigger || "NO_ENTRY_TRIGGER")],
        ["Strict", label(ctx.decision.strict_signal_stage || "STRICT_NO_ENTRY")],
        ["Research", label(ctx.decision.research_signal_stage || "NO_RESEARCH_SIGNAL")],
        ["Action", label(ctx.decision.research_signal_action || ctx.decision.trade_action || "NO_ACTION")],
        ["Trend", label(ctx.decision.trend_regime || "없음")],
        ["Vol", label(ctx.decision.vol_regime || "없음")],
        ["20D Breakout", planValue(ctx.planRows, "진입", "20일 고점 돌파 기준가") || "없음"],
        ["60D Breakout", planValue(ctx.planRows, "진입", "60일 고점 돌파 기준가") || "없음"],
      ], ctx.hasTrigger ? "가격 조건이 신호 후보를 만들었습니다." : "점수는 관찰권일 수 있지만 아직 행동 트리거가 없습니다.", { actionability: Math.round((ctx.score || 0) * 0.18), blockPressure: ctx.hasTrigger ? -6 : 10 }, "tsm_latest_decision_snapshot.csv"),

      object("risk_policy_today", "RiskPolicy", "리스크 객체", label(ctx.risk.risk_state || "UNKNOWN"), scoreTone(metrics.riskCapacityScore, metrics.riskCapacityScore >= 60), [
        ["Max Weight", fmtPct(ctx.maxWeightPct, 2)],
        ["Bottleneck", label(riskBottleneck)],
        ["Vol Limit", fmtWeight(ctx.risk.vol_limit_weight)],
        ["Trend Limit", fmtWeight(ctx.risk.trend_limit_weight)],
        ["Drawdown Limit", fmtWeight(ctx.risk.drawdown_limit_weight)],
        ["Score Limit", fmtWeight(ctx.risk.score_limit_weight)],
        ["Account Risk", fmtWeight(ctx.risk.account_risk_limit_weight)],
        ["2ATR Stop", fmtDisplayCurrency(ctx, ctx.risk.stop_price_2atr)],
        ["Stop Distance", fmtMaybePct(ctx.risk.risk_pct_2atr, 2)],
      ], "신호의 alpha가 아니라 오늘 감당 가능한 위험량과 병목을 설명합니다.", { actionability: Math.round((metrics.riskCapacityScore - 50) * 0.24), blockPressure: Math.round(100 - metrics.riskCapacityScore) }, "tsm_latest_risk_snapshot.csv"),

      object("paper_oms_today", "PaperOMS", "Paper OMS 객체", label(ctx.orderIntent.status || ctx.portfolioRisk.portfolio_risk_status || "MISSING"), metrics.paperOmsScore >= 70 ? "good" : metrics.paperOmsScore < 45 ? "bad" : "warn", [
        ["Intent", label(ctx.orderIntent.status || "MISSING")],
        ["Risk Gate", label(ctx.portfolioRisk.portfolio_risk_status || "MISSING")],
        ["Approved Weight", fmtWeight(ctx.portfolioRisk.portfolio_approved_weight)],
        ["Paper Position", label(ctx.paperPosition.position_state || "CASH")],
        ["Position Weight", fmtWeight(ctx.paperPosition.weight)],
        ["Reconciliation", label(ctx.paperReconciliation.status || "MISSING")],
        ["Mismatch Count", fmtNumber(ctx.paperReconciliation.mismatch_count, 0)],
        ["Lifecycle", label(ctx.orderLifecycle.lifecycle_state || "MISSING")],
        ["Feedback Matured", label(ctx.executionFeedback.label_matured || "TRACKING")],
        ["Fill Calibration", label(ctx.fillModelCalibration.calibration_status || "MISSING")],
        ["Automation", label(ctx.automation.automation_status || "MISSING")],
        ["Live", label(ctx.orderIntent.live_trading_status || ctx.paperReconciliation.live_trading_status || "DISABLED_BY_DESIGN")],
      ], "실제 브로커 없이 주문 의도, Paper 체결, 포지션 원장, 정합성, 실행 피드백, 체결모델 보정까지 닫힌 루프로 기록합니다.", { actionability: Math.round((metrics.paperOmsScore - 50) * 0.18), blockPressure: Math.round(100 - metrics.paperOmsScore) }, "tsm_order_intents.csv"),

      object("prediction_model_today", "PredictionSnapshot", "예측·모델 객체", label(permission.prediction), ctx.predictionAllowed ? "good" : "warn", [
        ["20D Success", fmtPct(ctx.pSuccess, 1)],
        ["Threshold", fmtPct(ctx.threshold, 1)],
        ["Prediction Edge", pEdgeText],
        ["Stop Hit", fmtPct(ctx.pStop, 1)],
        ["Expected R", fmtNumber(ctx.expectedR, 2)],
        ["Expected Net", fmtPct(ctx.expectedNet, 2)],
        ["Use Status", label(ctx.prediction.prediction_use_status || ctx.modelGate.model_gate_status || "DISPLAY_ONLY")],
        ["Best Model", label(ctx.pooled.model_name || ctx.prediction.trade_ready_best_model_20d || ctx.prediction.best_model_20d || "없음")],
      ], ctx.predictionAllowed ? "게이트 통과 예측만 판단 보조로 연결합니다." : "예측값은 있지만 룰 판단을 뒤집는 근거로 쓰지 않습니다.", { actionability: Math.round((metrics.predictionPermissionEdge - 50) * 0.22), blockPressure: ctx.predictionAllowed ? -8 : 24 }, "tsm_latest_prediction_snapshot.csv"),

      object("validation_evidence_today", "ValidationEvidence", "검증 객체", metrics.overfitRiskLevel === "HIGH" ? "과최적화 주의" : "검증 확인", metrics.validationTrustScore >= 70 ? "good" : "warn", [
        ["Validation Trust", `${fmtNumber(metrics.validationTrustScore, 0)} / 100`],
        ["Causal WF", passRateText(ctx.tables.validation_causal_walk_forward, "test_positive")],
        ["Walk-forward", passRateText(ctx.tables.validation_walk_forward, "test_positive")],
        ["CPCV Strategy", passRateText(ctx.tables.cpcv_strategy_distribution, "cpcv_median_uplift_pass")],
        ["CPCV Model", passRateText(ctx.tables.cpcv_model_distribution, "cpcv_model_median_uplift_pass")],
        ["DSR", passRateText(ctx.tables.deflated_sharpe, "dsr_pass")],
        ["Overfit Risk", metrics.overfitRiskLevel],
      ], "전체 백테스트보다 시간 순서 검증과 과최적화 점검을 우선 반영합니다.", { actionability: Math.round((metrics.validationTrustScore - 50) * 0.16), blockPressure: Math.round(100 - metrics.validationTrustScore) }, "tsm_validation_causal_walk_forward_summary.csv"),

      object("backtest_strategy_today", "BacktestStrategy", "성과 객체", bestStrategy?.strategy_id ? "대표 전략 있음" : "성과 없음", bestStrategy ? "neutral" : "warn", [
        ["Best Strategy", label(bestStrategy?.strategy_id || "없음")],
        ["CAGR", fmtMaybePct(bestStrategy?.cagr_pct, 1)],
        ["MDD", fmtMaybePct(bestStrategy?.max_drawdown_pct, 1)],
        ["Sharpe", fmtNumber(bestStrategy?.sharpe_zero_rf, 2)],
        ["Profit Factor", fmtNumber(bestStrategy?.profit_factor, 2)],
        ["Trades", fmtNumber(bestStrategy?.trade_count, 0)],
        ["ML Overlay", label(bestOverlayRow?.policy || "없음")],
      ], "오늘 신호 유형이 과거 어떤 전략군과 닮았는지 연결하는 근거입니다.", { actionability: bestStrategy ? 8 : -6, blockPressure: bestStrategy ? -4 : 12 }, "tsm_backtest_strategy_summary.csv"),

      object("news_cause_today", "NewsCause", "뉴스 원인 객체", label(ctx.latestNews.news_primary_cause_type || "NO_HIGH_CONFIDENCE_NEWS"), newsTone(ctx.latestNews, metrics.newsDragScore), [
        ["Coverage", label(ctx.latestNews.news_coverage_status || "없음")],
        ["Confidence", label(ctx.latestNews.news_match_confidence || "NO_MATCH")],
        ["Penalty", fmtNumber(ctx.latestNews.news_penalty_event, 1)],
        ["3D Events", fmtNumber(ctx.latestNews.news_event_count_3d, 0)],
        ["Source Count", fmtNumber(ctx.latestNews.news_source_count, 0)],
        ["Recent Penalty Days", fmtNumber(ctx.snapshots.news_penalty_summary?.recent_120_penalty_days, 0)],
      ], metrics.newsDragScore >= 35 ? "뉴스가 점수와 예측 해석을 끌어내립니다." : "오늘은 뉴스가 핵심 차단 원인이 아닙니다.", { actionability: Math.round((metrics.causeClarityScore - 50) * 0.08), blockPressure: metrics.newsDragScore }, "tsm_news_integrated_daily.csv"),

      object("pooled_universe_today", "PooledUniverse", "반도체 유니버스 객체", label(ctx.pooled.model_support_route || ctx.pooled.tsm_calibration_status || "DISPLAY_ONLY"), metrics.crossSectionalSupport >= 70 ? "good" : "warn", [
        ["Loaded Symbols", `${fmtNumber(ctx.universe.loaded_symbols, 0)} / ${fmtNumber(ctx.universe.candidate_symbols, 0)}`],
        ["Strict Eligible", fmtNumber(ctx.universe.strict_eligible_symbols, 0)],
        ["Decision Events", fmtNumber(ctx.pooled.decision_event_count || ctx.pooled.oos_event_count, 0)],
        ["Selected OOS", fmtNumber(ctx.pooled.selected_oos_event_count, 0)],
        ["Similarity Strength", `${fmtNumber(metrics.tsmLikeSampleStrength, 0)} / 100`],
        ["Cross Support", `${fmtNumber(metrics.crossSectionalSupport, 0)} / 100`],
      ], "Top10 판단 표본을 Universal research pool과 similarity calibration으로 보완합니다.", { actionability: Math.round((metrics.crossSectionalSupport - 50) * 0.12), blockPressure: ctx.predictionAllowed ? -8 : 12 }, "tsm_pooled_latest_prediction_overlay.csv"),

      object("operations_quality_today", "DataQualityState", "운영·품질 객체", label(ctx.dataQuality.data_quality_status || "MISSING"), metrics.dataTrust >= 75 ? "good" : metrics.dataTrust < 45 ? "bad" : "warn", [
        ["Pipeline", ctx.lineageStatus || label(lineageStatus(ctx.tables.manifest))],
        ["Data Quality", label(ctx.dataQuality.data_quality_status || "MISSING")],
        ["Integrity", summaryStatus(ctx.quality.integrity)],
        ["Prediction Quality", summaryStatus(ctx.quality.prediction)],
        ["Schema", summaryStatus(ctx.quality.schema)],
        ["Latest Signal Age", ctx.dataAgeDays === null ? "없음" : `${ctx.dataAgeDays}일`],
        ["Missing Required Files", fmtNumber(ctx.dataQuality.missing_required_files, 0)],
      ], primaryBlocker ? `${label(primaryBlocker.code)} 차단을 우선 확인합니다.` : "판단 입력과 운영 계보를 신뢰할 수 있는지 확인합니다.", { actionability: Math.round((metrics.dataTrust - 50) * 0.22), blockPressure: Math.round(100 - metrics.dataTrust) }, "tsm_latest_data_quality_snapshot.csv"),
    ];
  }

  function buildEdges(ctx, metrics) {
    return [
      edge("operations_quality_today", "price_rule_today", "controls", "데이터 품질이 가격·룰 입력을 통제", metrics.dataTrust >= 70 ? "good" : "warn"),
      edge("news_cause_today", "price_rule_today", "penalizes", "뉴스 원인이 룰 점수 감점으로 연결", metrics.newsDragScore >= 35 ? "warn" : "neutral"),
      edge("price_rule_today", "risk_policy_today", "sized_by", "신호 후보가 리스크 정책으로 비중 제한", metrics.riskCapacityScore >= 60 ? "good" : "warn"),
      edge("risk_policy_today", "paper_oms_today", "approves", "리스크 승인 후에만 Paper 주문 의도가 체결 시뮬레이션으로 이동", metrics.paperOmsScore >= 70 ? "good" : "warn"),
      edge("price_rule_today", "paper_oms_today", "creates_intent", "가격 신호가 주문 의도의 원천 signal_id가 됨", ctx.orderIntent.intent_id ? "good" : "warn"),
      edge("price_rule_today", "prediction_model_today", "evaluated_by", "룰 후보만 예측 품질 평가 대상", ctx.predictionAllowed ? "good" : "warn"),
      edge("validation_evidence_today", "prediction_model_today", "gated_by", "검증 근거가 모델 사용 권한을 결정", ctx.predictionAllowed ? "good" : "warn"),
      edge("backtest_strategy_today", "validation_evidence_today", "evidences", "백테스트 성과를 시간순 검증으로 재평가", metrics.validationTrustScore >= 70 ? "good" : "warn"),
      edge("pooled_universe_today", "prediction_model_today", "supports", "Universal research pool이 Top10 예측을 보조", metrics.crossSectionalSupport >= 70 ? "good" : "warn"),
      edge("prediction_model_today", "risk_policy_today", "informs", "게이트 통과 시 신호 품질을 리스크 해석에 반영", ctx.predictionAllowed ? "good" : "warn"),
    ];
  }

  function rankBlockers(ctx, metrics) {
    const blockers = [];
    for (const row of ctx.tables.model_gate_root_causes || []) {
      const code = text(row.root_cause || row.block_reason);
      if (!code || code.toUpperCase() === "PASS") continue;
      const count = number(row.failed_gate_count) || 1;
      const priority = number(row.priority) || 5;
      blockers.push({
        code,
        title: label(code),
        source: "ModelGate",
        detail: row.recommended_action || row.gates || row.gate_groups || "",
        affectedObjects: ["PredictionSnapshot", "ModelGate", "TradingPlan"],
        severity: priority <= 1 ? "CRITICAL" : "WARN",
        score: 42 + count * 3 + (6 - Math.min(priority, 5)) * 8 + (ctx.predictionAllowed ? 0 : 18),
      });
    }
    for (const row of ctx.tables.system_block_reasons || []) {
      const raw = text(row.block_reasons);
      if (!raw || raw.toUpperCase() === "PASS") continue;
      for (const code of splitReasons(raw)) {
        blockers.push({
          code,
          title: label(code),
          source: row.component || "SystemState",
          detail: row.details || "",
          affectedObjects: ["Operations", "TradingPlan"],
          severity: text(row.status).toUpperCase().includes("FAIL") ? "CRITICAL" : "WARN",
          score: 48 + (text(row.status).toUpperCase().includes("FAIL") ? 24 : 8),
        });
      }
    }
    for (const [name, summary] of Object.entries(ctx.quality || {})) {
      for (const failed of summary?.failed_rows || []) {
        const code = failed.block_reason || failed.check_name || failed.gate || `${name}_FAILED`;
        blockers.push({
          code,
          title: label(code),
          source: `quality.${name}`,
          detail: failed.details || failed.threshold || "",
          affectedObjects: ["DataQualityState"],
          severity: failed.severity || "WARN",
          score: text(failed.severity).toUpperCase() === "CRITICAL" ? 76 : 58,
        });
      }
    }
    if (ctx.risk.limiting_reason) {
      blockers.push({
        code: ctx.risk.limiting_reason,
        title: label(ctx.risk.limiting_reason),
        source: "RiskPolicy",
        detail: `최대 권장 비중 ${fmtPct(ctx.maxWeightPct, 2)}`,
        affectedObjects: ["SignalCandidate", "RiskPolicy", "TradingPlan"],
        severity: metrics.riskCapacityScore < 45 ? "CRITICAL" : "WARN",
        score: 45 + (100 - metrics.riskCapacityScore) * 0.55,
      });
    }
    if (!ctx.hasTrigger) {
      blockers.push({
        code: "NO_ENTRY_TRIGGER",
        title: "가격 트리거 대기",
        source: "PriceRule",
        detail: "점수가 있어도 entry trigger 없이는 행동 후보로 승격하지 않습니다.",
        affectedObjects: ["PriceRule", "TradingPlan"],
        severity: "INFO",
        score: (ctx.score || 0) >= 60 ? 62 : 42,
      });
    }
    return mergeBlockers(blockers).sort((a, b) => b.score - a.score).slice(0, 8);
  }

  function detectContradictions(ctx, metrics) {
    const contradictions = [];
    const pEdgePositive = metrics.predictionEdge !== null && metrics.predictionEdge > 0;
    const modelBlocked = modelGateBlocked(ctx);
    const backtest = bestBacktest(ctx.tables.backtest_summary || []);
    const causal = positiveRate(ctx.tables.validation_causal_walk_forward || [], "test_positive");
    const stressImpact = Math.abs(number(ctx.stress.worst_current_weight_portfolio_impact_pct) || 0);
    const newsConfidence = text(ctx.latestNews.news_match_confidence).toUpperCase();
    const newsCause = text(ctx.latestNews.news_primary_cause_type).toUpperCase();

    if ((ctx.score || 0) >= 60 && !ctx.hasTrigger) {
      contradictions.push(conflict("PRICE_SCORE_WITHOUT_TRIGGER", "가격 점수 높음 + trigger 없음", "강하지만 아직 행동 조건이 없습니다.", "20D/60D 돌파 가격을 확인합니다.", "warn"));
    }
    if (ctx.hasTrigger && metrics.riskCapacityScore < 45) {
      contradictions.push(conflict("TRIGGER_WITH_LOW_RISK_CAPACITY", "trigger 있음 + risk capacity 낮음", "신호는 있으나 포지션 크기가 제한됩니다.", "손절과 최종 비중을 다시 확인합니다.", "warn"));
    }
    if ((ctx.pSuccess !== null && ctx.pSuccess >= 55 || pEdgePositive) && modelBlocked) {
      contradictions.push(conflict("HIGH_PROBABILITY_MODEL_BLOCKED", "모델 확률 높음 + model gate blocked", "예측값은 좋아도 판단 근거로 쓸 수 없습니다.", "model gate root cause를 확인합니다.", "warn"));
    }
    if (bool(ctx.pooled.decision_support_allowed) && !ctx.predictionAllowed) {
      contradictions.push(conflict("POOLED_SUPPORT_LOCAL_WEAK", "pooled model 좋음 + Top10 direct 약함", "Research pool은 지지하지만 Top10 직접 판단 권한이 약합니다.", "Paper만 허용하고 live 판단은 금지합니다.", "warn"));
    }
    if (backtest && (number(backtest.cagr_pct) || 0) > 5 && causal !== null && causal < 45) {
      contradictions.push(conflict("BACKTEST_GOOD_CAUSAL_WEAK", "backtest summary 좋음 + causal WF 약함", "전체 성과보다 시간순 검증이 취약합니다.", "causal WF와 CPCV 결과를 우선 확인합니다.", "warn"));
    }
    if (metrics.dataTrust < 45 && hasValue(ctx.decision.trade_action)) {
      contradictions.push(conflict("DECISION_WITH_LOW_DATA_TRUST", "data quality fail + decision exists", "판단값은 있지만 입력 신뢰가 부족합니다.", "데이터 갱신을 먼저 실행합니다.", "bad"));
    }
    if (newsConfidence === "HIGH" && (newsCause.includes("NEGATIVE") || metrics.newsDragScore > 45) && ctx.hasTrigger) {
      contradictions.push(conflict("NEGATIVE_NEWS_WITH_RULE_ENTRY", "negative news HIGH + rule entry allowed", "가격 신호와 뉴스 원인이 충돌합니다.", "뉴스 감점과 보류 여부를 확인합니다.", "warn"));
    }
    if (stressImpact > 3 && ctx.maxWeightPct !== null && ctx.maxWeightPct > 8) {
      contradictions.push(conflict("HIGH_STRESS_HIGH_WEIGHT", "stress high + max weight high", "스트레스와 리스크 정책이 충돌할 수 있습니다.", "risk cap을 재검토합니다.", "warn"));
    }
    return contradictions;
  }

  function deriveActionPlan(ctx, metrics, permission, blockers, contradictions) {
    const breakout20 = planValue(ctx.planRows, "진입", "20일 고점 돌파 기준가");
    const breakout60 = planValue(ctx.planRows, "진입", "60일 고점 돌파 기준가");
    let next = "Paper 기록 가능 여부 확인";
    let reason = "모든 게이트가 바뀌면 결론을 다시 합성합니다.";
    if (metrics.dataTrust < 45) {
      next = "데이터 갱신 및 품질 실패 행 확인";
      reason = "데이터 신뢰가 낮아 어떤 판단도 권한을 얻지 못합니다.";
    } else if (!ctx.hasTrigger) {
      next = breakout20 || breakout60 ? `${breakout20 || breakout60} 돌파 확인` : "20D / 60D 돌파 기준가 확인";
      reason = "가격 점수가 있어도 entry trigger가 없으면 행동 후보가 아닙니다.";
    } else if (modelGateBlocked(ctx)) {
      next = blockers[0] ? `${label(blockers[0].code)} 확인` : "모델 게이트 root cause 확인";
      reason = "예측은 표시 전용이며 trade action을 뒤집을 수 없습니다.";
    } else if (metrics.riskCapacityScore < 55) {
      next = "손절가와 최종 비중 재계산";
      reason = "신호가 와도 오늘 감당 가능한 위험량이 낮습니다.";
    } else if (ctx.paperAllowed) {
      next = "Paper Gate와 Shadow Paper 기록 확인";
      reason = "실거래는 설계상 비활성이므로 Paper 기록 중심으로 검증합니다.";
    }
    return {
      next,
      reason,
      mainDriver: mainDriver(ctx, metrics),
      mainBlocker: blockers[0] || null,
      targetPrice20d: breakout20,
      targetPrice60d: breakout60,
      stopPrice: ctx.risk.stop_price_2atr,
      stopPriceDisplay: fmtDisplayCurrency(ctx, ctx.risk.stop_price_2atr),
      maxWeight: ctx.maxWeightPct,
      contradictionCount: contradictions.length,
    };
  }

  function buildInsights(ctx, metrics, permission, blockers, contradictions, actionPlan) {
    const insights = [];
    if ((ctx.score || 0) >= 60 && !ctx.hasTrigger) {
      insights.push(insight("가격은 관찰 우위지만 트리거가 없다", `${fmtNumber(ctx.score, 1)}점은 관찰권이지만 entry_trigger가 ${label(ctx.entryTrigger)}입니다.`, "다음 확인: 20D / 60D 돌파 기준가", "warn"));
    }
    if (metrics.riskCapacityScore < 55) {
      insights.push(insight("리스크 한도가 오늘의 실행 병목이다", `최대 권장 비중은 ${fmtPct(ctx.maxWeightPct, 2)}이고 병목은 ${label(ctx.risk.limiting_reason)}입니다.`, "신호가 와도 소액/관찰 이상으로 확대하지 않습니다.", "warn"));
    }
    if (!ctx.predictionAllowed) {
      insights.push(insight("예측은 판단 근거가 아니라 참고용이다", `20D 성공 확률 ${fmtPct(ctx.pSuccess, 1)}, threshold ${fmtPct(ctx.threshold, 1)} 상태입니다.`, "model gate root cause를 먼저 확인합니다.", "warn"));
    }
    if (metrics.newsDragScore < 25) {
      insights.push(insight("뉴스는 현재 핵심 차단 요인이 아니다", `${label(ctx.latestNews.news_primary_cause_type || "NO_HIGH_CONFIDENCE_NEWS")} · 감점 ${fmtNumber(ctx.latestNews.news_penalty_event, 1)}입니다.`, "가격/리스크/모델 병목을 우선 확인합니다.", "neutral"));
    } else {
      insights.push(insight("뉴스가 판단을 끌어내리는 압력이다", `${label(ctx.latestNews.news_match_confidence)} 원인과 감점이 합성 점수를 낮춥니다.`, "뉴스 원인과 룰 감점 연결을 확인합니다.", "warn"));
    }
    if (metrics.dataTrust >= 75 && blockers.length) {
      insights.push(insight("시스템 자체는 동작 가능하지만 게이트가 막고 있다", `데이터 신뢰 ${fmtNumber(metrics.dataTrust, 0)}/100, 첫 차단 ${label(blockers[0].code)}입니다.`, "데이터보다 모델/리스크 병목을 먼저 봅니다.", "warn"));
    }
    if (contradictions.length) {
      insights.push(insight(`오늘 발견된 모순 ${contradictions.length}개`, contradictions[0].meaning, contradictions[0].action, contradictions[0].tone));
    }
    insights.push(insight("다음 액션", actionPlan.next, actionPlan.reason, permission.tone));
    return insights.slice(0, 7);
  }

  function buildLineage(ctx, metrics) {
    const manifest = ctx.tables.manifest || [];
    const failed = manifest.filter((row) => text(row.status).toUpperCase() === "FAIL");
    const last = manifest[manifest.length - 1] || {};
    const passCount = manifest.filter((row) => text(row.status).toUpperCase() === "PASS").length;
    return {
      lastRunStatus: failed.length ? "FAIL" : manifest.length ? "PASS" : ctx.data.run?.status || "IDLE",
      startedAt: manifest[0]?.started_at_utc || ctx.data.run?.started_at || "",
      endedAt: last.ended_at_utc || ctx.data.run?.ended_at || "",
      durationSec: sum(manifest.map((row) => number(row.duration_sec) || 0)),
      failedSteps: failed.length,
      failedStepNames: failed.map((row) => row.step).filter(Boolean),
      passSteps: passCount,
      totalSteps: manifest.length,
      latestSignalDate: shortDate(ctx.dataQuality.latest_signal_date || ctx.asOfDate),
      dataAgeDays: ctx.dataAgeDays,
      requiredFilesStatus: summaryStatus(ctx.quality.operational),
      reports: (ctx.data.reports || []).length,
      images: (ctx.data.images || []).length,
      newsCollection: label(ctx.latestNews.news_coverage_status || "없음"),
      pipelineFreshnessScore: metrics.pipelineFreshnessScore,
    };
  }

  function buildExecutionLadder(ctx, metrics, permission) {
    return [
      ladder("Data Loaded", summaryStatus(ctx.quality.operational), metrics.dataTrust >= 45 ? "PASS" : "BLOCKED", "필수 파일과 운영 품질 점검"),
      ladder("Raw/Enriched Match", summaryStatus(ctx.quality.data_contract), (ctx.quality.data_contract?.failed || 0) > 0 ? "BLOCKED" : "PASS", "raw/enriched row/date 계약"),
      ladder("Price Rule Built", label(ctx.decision.entry_trigger || "NO_ENTRY_TRIGGER"), hasValue(ctx.decision.trade_action) ? "PASS" : "WARN", "daily algorithmic signals"),
      ladder("Signal Trigger", ctx.hasTrigger ? label(ctx.entryTrigger) : "WAIT", ctx.hasTrigger ? "PASS" : "WARN", "entry trigger"),
      ladder("Risk Capacity", fmtPct(ctx.maxWeightPct, 2), metrics.riskCapacityScore >= 55 ? "PASS" : "LIMITED", label(ctx.risk.limiting_reason || ctx.risk.risk_state)),
      ladder("Prediction Quality", label(permission.prediction), ctx.predictionAllowed ? "PASS" : "DISPLAY_ONLY", "latest prediction / model gate"),
      ladder("Model Gate", label(ctx.modelGate.model_gate_status || permission.prediction), modelGateBlocked(ctx) ? "BLOCKED" : "PASS", "model gate snapshot/audit/root causes"),
      ladder("Stress Check", `${fmtNumber(metrics.stressTolerance, 0)} / 100`, metrics.stressTolerance >= 65 ? "PASS" : "WARN", "latest stress snapshot"),
      ladder("Paper Gate", label(permission.paper), ctx.paperAllowed ? "READY" : "NOT_READY", "paper gate / shadow paper"),
      ladder("Paper OMS", label(ctx.orderLifecycle.lifecycle_state || ctx.orderIntent.status || "MISSING"), metrics.paperOmsScore >= 70 ? "READY" : "WARN", "order intent / execution / feedback / calibration"),
      ladder("Trading Plan", permission.title, permission.state.includes("STRICT") ? "READY" : permission.state.includes("DATA") ? "BLOCKED" : "WAIT", "daily trading plan"),
    ];
  }

  function buildGraph(objects, edges, actionPlan, metrics) {
    const graphNodes = [
      { id: "operations_quality_today", x: 10, y: 18 },
      { id: "news_cause_today", x: 10, y: 62 },
      { id: "price_rule_today", x: 30, y: 38 },
      { id: "risk_policy_today", x: 51, y: 38 },
      { id: "validation_evidence_today", x: 51, y: 72 },
      { id: "pooled_universe_today", x: 70, y: 72 },
      { id: "prediction_model_today", x: 70, y: 38 },
      { id: "paper_oms_today", x: 84, y: 58 },
      { id: "trading_plan_today", x: 90, y: 38, virtual: true },
    ];
    const objectById = new Map(objects.map((objectItem) => [objectItem.id, objectItem]));
    const nodes = graphNodes.map((node) => {
      const objectItem = objectById.get(node.id) || {
        id: node.id,
        type: "TradingPlan",
        label: "매매 계획",
        title: actionPlan.next,
        status: "Next Action",
        tone: metrics.blockPressure >= 62 ? "warn" : "neutral",
        fields: [
          ["20D", actionPlan.targetPrice20d || "없음"],
          ["60D", actionPlan.targetPrice60d || "없음"],
          ["Stop", actionPlan.stopPriceDisplay || fmtCurrency(actionPlan.stopPrice)],
          ["Max Weight", fmtPct(actionPlan.maxWeight, 2)],
        ],
        meaning: actionPlan.reason,
        impacts: { actionability: 0, blockPressure: 0 },
      };
      return { ...objectItem, ...node };
    });
    const graphEdges = [
      ...edges,
      edge("risk_policy_today", "trading_plan_today", "converted_to", "최종 비중과 손절을 매매 계획으로 전달", metrics.riskCapacityScore >= 55 ? "good" : "warn"),
      edge("prediction_model_today", "trading_plan_today", "gated_by", "예측 권한은 판단 보조 여부만 결정", metrics.predictionPermissionEdge >= 60 ? "good" : "warn"),
    ];
    return { nodes, edges: graphEdges };
  }

  function object(id, type, labelText, status, tone, fields, meaning, impacts, sourceFile) {
    return { id, type, label: labelText, status, tone, fields, meaning, impacts, source: { file: sourceFile } };
  }

  function edge(from, to, relation, labelText, tone) {
    return { from, to, relation, label: labelText, tone };
  }

  function conflict(code, title, meaning, action, tone) {
    return { code, title, meaning, action, tone };
  }

  function insight(title, body, action, tone) {
    return { title, body, action, tone };
  }

  function ladder(labelText, value, status, note) {
    return { label: labelText, value, status, tone: ladderTone(status), note };
  }

  function mainDriver(ctx, metrics) {
    const candidates = [
      { objectId: "price_rule_today", label: "가격·룰", score: metrics.ruleSignalStrength },
      { objectId: "risk_policy_today", label: "리스크", score: metrics.riskCapacityScore },
      { objectId: "prediction_model_today", label: "예측", score: metrics.predictionPermissionEdge },
      { objectId: "paper_oms_today", label: "Paper OMS", score: metrics.paperOmsScore },
      { objectId: "validation_evidence_today", label: "검증", score: metrics.validationTrustScore },
      { objectId: "news_cause_today", label: "뉴스 명확성", score: metrics.causeClarityScore },
      { objectId: "operations_quality_today", label: "운영·품질", score: metrics.dataTrust },
    ];
    return candidates.sort((a, b) => b.score - a.score)[0];
  }

  function mergeBlockers(blockers) {
    const map = new Map();
    for (const blocker of blockers) {
      const key = text(blocker.code).toUpperCase();
      const current = map.get(key);
      if (!current) {
        map.set(key, { ...blocker, count: 1 });
      } else {
        current.count += 1;
        current.score = Math.max(current.score, blocker.score) + 2;
        current.affectedObjects = Array.from(new Set([...current.affectedObjects, ...blocker.affectedObjects]));
        if (!current.detail && blocker.detail) current.detail = blocker.detail;
      }
    }
    return Array.from(map.values());
  }

  function modelGateBlocked(ctx) {
    const gate = text(ctx.modelGate.model_gate_status || ctx.paperGate.paper_gate_status || ctx.prediction.prediction_use_status || ctx.system.prediction_use_status).toUpperCase();
    return gate.includes("BLOCK") || gate.includes("DISPLAY_ONLY") || !ctx.predictionAllowed;
  }

  function confidenceLevel(metrics, ctx) {
    if (metrics.dataTrust >= 80 && metrics.validationTrustScore >= 70 && metrics.pipelineFreshnessScore >= 75 && !modelGateBlocked(ctx)) return "High";
    if (metrics.dataTrust >= 65 && metrics.pipelineFreshnessScore >= 55) return "Medium";
    return "Low";
  }

  function bestBacktest(rows) {
    return (rows || [])
      .filter((row) => number(row.cagr_pct) !== null)
      .slice()
      .sort((a, b) => (number(b.cagr_pct) || 0) - (number(a.cagr_pct) || 0))[0] || null;
  }

  function bestOverlay(rows) {
    return (rows || [])
      .filter((row) => number(row.cumulative_weighted_return_pct ?? row.mean_net_return_pct) !== null)
      .slice()
      .sort((a, b) => (number(b.cumulative_weighted_return_pct ?? b.mean_net_return_pct) || 0) - (number(a.cumulative_weighted_return_pct ?? a.mean_net_return_pct) || 0))[0] || null;
  }

  function planValue(rows, section, item) {
    const row = (rows || []).find((candidate) => text(candidate.section) === section && text(candidate.item).includes(item));
    return row?.value || "";
  }

  function firstValue(a, keysA, b, keysB) {
    for (const key of keysA || []) if (hasValue(a?.[key])) return a[key];
    for (const key of keysB || []) if (hasValue(b?.[key])) return b[key];
    return "";
  }

  function hasValue(value) {
    if (value === null || value === undefined) return false;
    const raw = String(value).trim();
    return raw !== "" && !["na", "n/a", "none", "null", "nan"].includes(raw.toLowerCase());
  }

  function text(value) {
    return hasValue(value) ? String(value).trim() : "";
  }

  function number(value) {
    if (!hasValue(value)) return null;
    const parsed = Number(String(value).replace(/[$,%\s,]/g, ""));
    return Number.isFinite(parsed) ? parsed : null;
  }

  function weight(value) {
    const parsed = number(value);
    if (parsed === null) return null;
    return Math.abs(parsed) > 1 ? parsed / 100 : parsed;
  }

  function probabilityPercent(value) {
    const parsed = number(value);
    if (parsed === null) return null;
    return Math.abs(parsed) <= 1 ? parsed * 100 : parsed;
  }

  function bool(value) {
    if (value === true) return true;
    if (value === false) return false;
    const raw = text(value).toLowerCase();
    return ["true", "1", "yes", "y", "pass", "ready", "decision_support_allowed"].includes(raw);
  }

  function clamp(value, min, max) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return min;
    return Math.max(min, Math.min(max, parsed));
  }

  function average(values) {
    const nums = values.filter((value) => value !== null && Number.isFinite(value));
    if (!nums.length) return null;
    return nums.reduce((acc, value) => acc + value, 0) / nums.length;
  }

  function sum(values) {
    return values.reduce((acc, value) => acc + (Number.isFinite(value) ? value : 0), 0);
  }

  function statusScoreValue(value) {
    const raw = text(value).toUpperCase();
    if (raw.includes("PASS") || raw.includes("READY") || raw.includes("ALLOWED") || raw.includes("APPROVED") || raw.includes("FILLED")) return 92;
    if (raw.includes("WARN") || raw.includes("DISPLAY_ONLY") || raw.includes("PAPER_SUBMITTED") || raw.includes("EXPIRED")) return 58;
    if (raw.includes("FAIL") || raw.includes("MISSING") || raw.includes("REJECTED") || raw.includes("BLOCKED") || raw.includes("MISMATCH")) return 25;
    return 50;
  }

  function statusTone(value) {
    const raw = text(value).toUpperCase();
    if (raw.includes("PASS") || raw.includes("READY") || raw.includes("ALLOWED") || raw.includes("APPROVED") || raw.includes("FILLED")) return "good";
    if (raw.includes("FAIL") || raw.includes("MISSING") || raw.includes("REJECTED") || raw.includes("BLOCKED") || raw.includes("MISMATCH")) return "bad";
    if (raw.includes("WARN") || raw.includes("DISPLAY_ONLY") || raw.includes("PAPER_SUBMITTED") || raw.includes("EXPIRED")) return "warn";
    return "neutral";
  }

  function scoreTone(score, positive) {
    if (positive === true || (number(score) || 0) >= 72) return "good";
    if ((number(score) || 0) >= 45) return "warn";
    return "neutral";
  }

  function newsTone(news, drag) {
    const confidence = text(news.news_match_confidence).toUpperCase();
    if (drag >= 55 || confidence === "HIGH") return "warn";
    if (drag <= 15) return "neutral";
    return "warn";
  }

  function ladderTone(status) {
    const raw = text(status).toUpperCase();
    if (raw.includes("PASS") || raw.includes("READY")) return "good";
    if (raw.includes("BLOCK") || raw.includes("FAIL")) return "bad";
    if (raw.includes("WARN") || raw.includes("WAIT") || raw.includes("LIMIT") || raw.includes("DISPLAY")) return "warn";
    return "neutral";
  }

  function trendScore(value) {
    const raw = text(value).toUpperCase();
    if (raw.includes("UP") || raw.includes("BULL") || raw.includes("ABOVE")) return 76;
    if (raw.includes("DOWN") || raw.includes("BEAR") || raw.includes("BELOW")) return 34;
    return 52;
  }

  function qualityPassRate(summary) {
    if (!summary) return null;
    const passRate = number(summary.pass_rate_pct);
    if (passRate !== null) return clamp(passRate, 0, 100);
    const total = number(summary.total);
    const passed = number(summary.passed);
    if (total && passed !== null) return clamp((passed / total) * 100, 0, 100);
    return null;
  }

  function summaryStatus(summary) {
    if (!summary) return "없음";
    if ((number(summary.failed) || 0) > 0) return "주의 필요";
    if ((number(summary.total) || 0) > 0) return "문제 없음";
    return "없음";
  }

  function positiveRate(rows, key, passValue) {
    if (!rows || !rows.length) return null;
    let count = 0;
    let total = 0;
    for (const row of rows) {
      const value = row[key];
      if (!hasValue(value)) continue;
      total += 1;
      if (passValue !== undefined ? text(value).toUpperCase() === passValue : bool(value)) count += 1;
    }
    if (!total) return null;
    return (count / total) * 100;
  }

  function passRateText(rows, key) {
    const rate = positiveRate(rows || [], key);
    return rate === null ? "없음" : `${fmtNumber(rate, 0)}%`;
  }

  function dateAgeDays(value) {
    if (!hasValue(value)) return null;
    const raw = String(value).slice(0, 10);
    const parsed = new Date(`${raw}T00:00:00`);
    if (Number.isNaN(parsed.getTime())) return null;
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    return Math.max(0, Math.round((today.getTime() - parsed.getTime()) / 86400000));
  }

  function shortDate(value) {
    return hasValue(value) ? String(value).slice(0, 10) : "없음";
  }

  function label(value) {
    const raw = text(value);
    if (!raw) return "없음";
    return STATUS_LABELS[raw] || STATUS_LABELS[raw.toUpperCase()] || raw.replaceAll("_", " ");
  }

  function splitReasons(value) {
    return text(value)
      .split(/[|,;]/)
      .map((part) => part.trim())
      .filter(Boolean)
      .filter((part) => part.toUpperCase() !== "PASS");
  }

  function fmtNumber(value, digits = 2) {
    const parsed = number(value);
    if (parsed === null) return "없음";
    return parsed.toLocaleString("ko-KR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
  }

  function fmtSigned(value, digits = 1) {
    const parsed = number(value);
    if (parsed === null) return "없음";
    const prefix = parsed > 0 ? "+" : "";
    return `${prefix}${fmtNumber(parsed, digits)}`;
  }

  function inferDisplayCurrency(data, ...rows) {
    const sources = [data && data.meta, ...rows].filter(Boolean);
    for (const source of sources) {
      const raw = source.display_currency || source.listing_currency;
      if (hasValue(raw)) return text(raw).toUpperCase();
    }
    const symbol = text(data && data.meta && data.meta.symbol).toUpperCase();
    return symbol.endsWith(".KS") || symbol.endsWith(".KQ") ? "KRW" : "USD";
  }

  function inferFxRateToUsd(...rows) {
    for (const row of rows.filter(Boolean)) {
      const fx = number(row.fx_rate_to_usd);
      if (fx !== null && fx > 0) return fx;
      const usdkrw = number(row.usdkrw);
      if (usdkrw !== null && usdkrw > 0) return 1 / usdkrw;
    }
    return null;
  }

  function fmtDisplayCurrency(ctx, value) {
    const parsed = number(value);
    if (parsed === null) return "없음";
    const currency = text(ctx && ctx.displayCurrency).toUpperCase();
    if (currency === "KRW") {
      const fx = number(ctx && ctx.fxRateToUsd);
      const native = fx !== null && fx > 0 ? parsed / fx : parsed;
      return `₩${native.toLocaleString("ko-KR", { maximumFractionDigits: 0 })}`;
    }
    return fmtCurrency(parsed);
  }

  function fmtCurrency(value) {
    const parsed = number(value);
    if (parsed === null) return "없음";
    return `$${parsed.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }

  function fmtPct(value, digits = 1) {
    const parsed = number(value);
    if (parsed === null) return "없음";
    return `${parsed.toLocaleString("ko-KR", { minimumFractionDigits: digits, maximumFractionDigits: digits })}%`;
  }

  function fmtMaybePct(value, digits = 1) {
    const parsed = number(value);
    if (parsed === null) return "없음";
    const pct = Math.abs(parsed) <= 1 ? parsed * 100 : parsed;
    return fmtPct(pct, digits);
  }

  function fmtWeight(value) {
    const parsed = weight(value);
    return parsed === null ? "없음" : fmtPct(parsed * 100, 2);
  }

  function lineageStatus(manifest) {
    if (!manifest || !manifest.length) return "IDLE";
    return manifest.some((row) => text(row.status).toUpperCase() === "FAIL") ? "FAIL" : "PASS";
  }

  window.TsmOntology = {
    buildModel,
    label,
    format: {
      number: fmtNumber,
      currency: fmtCurrency,
      pct: fmtPct,
      maybePct: fmtMaybePct,
    },
  };
})();
