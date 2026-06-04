const state = {
  data: null,
  activeTab: "all",
  selectedSymbol: null,
  runTimer: null,
  portfolio: null,
  portfolioCurrency: "KRW",
  portfolioLoaded: false,
  pfShowNewBuys: false,
};

const $ = (id) => document.getElementById(id);

const actionLabels = {
  execute_candidate: "실행 후보",
  watch: "관찰",
  reduce: "추격 금지",
  avoid: "매수 금지",
  blocked: "차단",
};

const statusTone = {
  정상: "good",
  주의: "warn",
  오래됨: "bad",
  "판단 가능": "good",
  "표시 전용": "warn",
  차단: "bad",
  PASS: "good",
  FAIL: "bad",
  DISABLED_BY_DESIGN: "neutral",
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function fetchJson(url, options) {
  const res = await fetch(url, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

function toNumber(value) {
  if (value === null || value === undefined) return null;
  if (typeof value === "string" && value.trim() === "") return null;
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function hasNumber(...values) {
  return values.some((value) => toNumber(value) !== null);
}

function fmtPct(value, digits = 1) {
  const num = toNumber(value);
  if (num === null) return "없음";
  const pct = Math.abs(num) <= 1 ? num * 100 : num;
  return `${pct.toFixed(digits)}%`;
}

function fmtPctPoints(value, digits = 1) {
  const num = toNumber(value);
  if (num === null) return "없음";
  return `${num.toFixed(digits)}%`;
}

function stopRiskNote(pred) {
  if (!pred) return "";
  const parts = [];
  if (toNumber(pred.down_risk_raw) !== null) parts.push(`raw ${fmtPct(pred.down_risk_raw)}`);
  if (toNumber(pred.down_risk_oos_percentile) !== null) parts.push(`OOS ${fmtPct(pred.down_risk_oos_percentile)}`);
  const warning = pred.stop_risk_calibration_warning && pred.stop_risk_calibration_warning !== "PASS"
    ? " · 최근 보수 과대평가 가능성"
    : "";
  if (!parts.length) return warning.trim().replace(/^·\s*/, "");
  return `${parts.join(" · ")}${warning}`;
}

function fmtWeight(value) {
  const num = toNumber(value);
  if (num === null || num <= 0) return "0%";
  return fmtPct(num, num < 0.01 ? 2 : 1);
}

function fmtNumber(value, digits = 2) {
  const num = toNumber(value);
  if (num === null) return "없음";
  return num.toLocaleString("ko-KR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function fmtPrice(value, currency) {
  const num = toNumber(value);
  if (num === null) return "없음";
  if (currency === "KRW") {
    return `₩${Math.round(num).toLocaleString("ko-KR")}`;
  }
  return `$${num.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function fmtUsdKrw(value) {
  const num = toNumber(value);
  if (num === null) return "환율 없음";
  return `1달러 = ${Math.round(num).toLocaleString("ko-KR")}원`;
}

function fmtExchangeResult(amount, direction, usdkrw) {
  const amt = toNumber(amount);
  const rate = toNumber(usdkrw);
  if (amt === null || rate === null || amt < 0 || rate <= 0) return "계산 불가";
  if (direction === "krw_to_usd") {
    return `$${(amt / rate).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }
  return `₩${Math.round(amt * rate).toLocaleString("ko-KR")}`;
}

function compactDate(value) {
  return String(value || "").slice(0, 10) || "날짜 없음";
}

function statusPill(label) {
  const tone = statusTone[label] || "neutral";
  return `<span class="status-pill ${tone}">${escapeHtml(label || "확인 필요")}</span>`;
}

function actionPill(level) {
  return `<span class="action-pill ${escapeHtml(level || "neutral")}">${escapeHtml(actionLabels[level] || "확인")}</span>`;
}

function coverageText(coverage) {
  if (!coverage) return "확인 필요";
  const present = coverage.present ?? 0;
  const expected = coverage.expected ?? 0;
  return `${present}/${expected}`;
}

function failedText(section) {
  if (!section) return "확인 필요";
  const critical = section.critical_failed ?? 0;
  const required = section.required_failed ?? 0;
  const failed = section.failed ?? 0;
  return `CRITICAL ${critical} · REQUIRED ${required} · FAIL ${failed}`;
}

function statusCountsText(counts) {
  const entries = Object.entries(counts || {});
  if (!entries.length) return "없음";
  return entries.map(([key, value]) => `${key} ${value}`).join(" · ");
}

function probabilityLabel(pred, fallback = "상승 확률") {
  return pred?.probability_label || fallback;
}

function setSystem(system) {
  const daily = system?.daily_update || {};
  const auditLabel = daily.full_audit_status || "확인 필요";
  $("dataStatus").textContent = system?.data_status || "확인 필요";
  $("modelStatus").textContent = system?.model_status || "확인 필요";
  $("auditStatus").textContent = auditLabel;
  $("liveStatus").textContent = system?.live_trading_status === "DISABLED_BY_DESIGN" ? "비활성" : system?.live_trading_status || "확인 필요";
  $("dataStatus").dataset.tone = statusTone[system?.data_status] || "neutral";
  $("modelStatus").dataset.tone = statusTone[system?.model_status] || "neutral";
  $("auditStatus").dataset.tone = auditLabel === "PASS" ? "good" : auditLabel === "FAIL" ? "bad" : "neutral";
  $("liveStatus").dataset.tone = "neutral";
}

function setFx(fx) {
  const usdkrw = fx?.usdkrw ?? fx?.fx_close;
  $("fxRate").textContent = fmtUsdKrw(usdkrw);
  $("fxRate").dataset.tone = toNumber(usdkrw) === null ? "neutral" : "good";
  const source = fx?.data_source ? ` · ${fx.data_source}` : "";
  $("fxRate").parentElement.title = fx?.date ? `기준일 ${compactDate(fx.date)}${source}` : "환율 정보 없음";
}

function renderActionRow() {
  const data = state.data;
  const queue = data?.action_queue || [];
  const symbols = data?.symbols || [];
  const topCandidate = queue.find((item) => item.action_level === "execute_candidate") || queue[0];
  const hardAvoid = symbols.find((item) => ["avoid", "reduce", "blocked"].includes(item.action_level));
  const system = data?.system || {};
  const daily = system.daily_update || {};
  const top12Coverage = coverageText(daily.coverage?.latest_prediction_top12);
  const foreignCoverage = coverageText(daily.coverage?.latest_prediction_overseas);
  const fallbackRows = daily.latest_predictions?.fallback_rows ?? "확인 필요";
  const cards = [
    {
      label: "1순위",
      title: topCandidate ? topCandidate.action : "실행 후보 없음",
      meta: topCandidate ? `${topCandidate.name} · ${topCandidate.symbol}` : "새 매수는 대기",
      body: topCandidate ? `${topCandidate.main_reason} · 최대 ${fmtWeight(topCandidate.max_weight)}` : "조건이 맞는 종목이 나올 때까지 관찰",
      tone: topCandidate?.action_level || "neutral",
    },
    {
      label: "금지 행동",
      title: hardAvoid ? hardAvoid.action_text : "추격매수 금지",
      meta: hardAvoid ? `${hardAvoid.name} · ${hardAvoid.symbol}` : "공통 원칙",
      body: hardAvoid ? hardAvoid.warning : "손절 기준 없는 진입은 하지 않음",
      tone: hardAvoid?.action_level || "avoid",
    },
    {
      label: "신뢰 상태",
      title: `${system.data_status || "확인 필요"} · ${daily.full_audit_status || system.model_status || "확인 필요"}`,
      meta: `Top12 ${top12Coverage} · 해외 ${foreignCoverage}`,
      body: system.live_trading_status === "DISABLED_BY_DESIGN" ? `${system.main_model || "내일 상승 예측"} 메인 · fallback ${fallbackRows} · 실거래 주문 비활성` : "실거래 상태 확인 필요",
      tone: system.data_status === "정상" && system.model_status === "판단 가능" ? "execute_candidate" : "watch",
    },
  ];
  $("actionRow").innerHTML = cards
    .map(
      (card) => `
        <article class="action-card ${escapeHtml(card.tone)}">
          <span>${escapeHtml(card.label)}</span>
          <h2>${escapeHtml(card.title)}</h2>
          <strong>${escapeHtml(card.meta)}</strong>
          <p>${escapeHtml(card.body)}</p>
        </article>
      `,
    )
    .join("");
}

function filteredSymbols() {
  const symbols = state.data?.symbols || [];
  if (state.activeTab === "all") return symbols;
  if (state.activeTab === "risk") return symbols.filter((item) => ["avoid", "reduce", "blocked"].includes(item.action_level));
  return symbols.filter((item) => item.action_level === state.activeTab);
}

function renderSymbolGrid() {
  const symbols = filteredSymbols();
  const all = state.data?.symbols || [];
  const counts = all.reduce((acc, item) => {
    acc[item.action_level] = (acc[item.action_level] || 0) + 1;
    return acc;
  }, {});
  $("boardSummary").textContent = `실행 ${counts.execute_candidate || 0} · 관찰 ${counts.watch || 0} · 금지/위험 ${(counts.avoid || 0) + (counts.reduce || 0) + (counts.blocked || 0)}`;
  $("symbolGrid").innerHTML = symbols
    .map((item) => {
      const selected = item.symbol === state.selectedSymbol ? "selected" : "";
      const prob = fmtPct(item.prediction?.up_probability);
      return `
        <button class="symbol-chip ${escapeHtml(item.action_level)} ${selected}" type="button" role="tab" aria-selected="${selected ? "true" : "false"}" data-symbol="${escapeHtml(item.symbol)}" title="${escapeHtml(item.name)} · ${escapeHtml(actionLabels[item.action_level] || "")} · ${fmtPrice(item.price, item.currency)}">
          <span class="chip-dot" aria-hidden="true"></span>
          <b class="chip-ticker">${escapeHtml(item.symbol)}</b>
          <span class="chip-prob">${prob}</span>
        </button>
      `;
    })
    .join("");
  document.querySelectorAll(".symbol-chip").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedSymbol = button.dataset.symbol;
      render();
    });
  });
}

function chartSvg(item) {
  const series = (item?.price_series || []).filter((row) => toNumber(row.price) !== null).slice(-90);
  if (series.length < 2) {
    return `<div class="empty-chart">가격 데이터 없음</div>`;
  }
  const width = 680;
  const height = 220;
  const values = series.map((row) => Number(row.price));
  const stop = toNumber(item.risk?.stop_price);
  const target = toNumber(item.risk?.target_price);
  const extra = [stop, target].filter((value) => value !== null);
  const min = Math.min(...values, ...extra);
  const max = Math.max(...values, ...extra);
  const pad = Math.max((max - min) * 0.08, max * 0.01);
  const yMin = min - pad;
  const yMax = max + pad;
  const xFor = (idx) => (idx / Math.max(series.length - 1, 1)) * width;
  const yFor = (value) => height - ((value - yMin) / Math.max(yMax - yMin, 1)) * height;
  const path = series.map((row, idx) => `${idx === 0 ? "M" : "L"} ${xFor(idx).toFixed(2)} ${yFor(Number(row.price)).toFixed(2)}`).join(" ");
  const line = (value, cssClass, label) => {
    if (value === null) return "";
    const y = yFor(value).toFixed(2);
    return `<g class="${cssClass}"><line x1="0" y1="${y}" x2="${width}" y2="${y}"></line><text x="${width - 96}" y="${Number(y) - 6}">${escapeHtml(label)}</text></g>`;
  };
  return `
    <svg class="price-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(item.symbol)} 가격 차트">
      <g class="grid-lines">
        <line x1="0" y1="${height * 0.25}" x2="${width}" y2="${height * 0.25}"></line>
        <line x1="0" y1="${height * 0.5}" x2="${width}" y2="${height * 0.5}"></line>
        <line x1="0" y1="${height * 0.75}" x2="${width}" y2="${height * 0.75}"></line>
      </g>
      ${line(target, "target-line", "목표")}
      ${line(stop, "stop-line", "손절")}
      <path class="price-path" d="${path}"></path>
    </svg>
  `;
}

function predictionBlock(item, horizonKey) {
  const pred = item.predictions?.[horizonKey] || {};
  const horizonLabel = pred.horizon || (horizonKey === "1d" ? "내일" : horizonKey === "5d" ? "5거래일" : horizonKey === "20d" ? "20거래일" : horizonKey === "60d" ? "60거래일" : horizonKey);
  if (toNumber(pred.up_probability) === null && toNumber(pred.down_risk) === null && !pred.model_name) {
    return `
      <div class="prediction-block empty">
        <span>${escapeHtml(horizonLabel)}</span>
        <strong>예측 없음</strong>
      </div>
    `;
  }
  const mainClass = pred.is_main_model ? " main" : "";
  const downRiskLabel = pred.is_main_model ? "하락 가능성" : "보정 손절";
  const primaryLabel = probabilityLabel(pred);
  const metric = (label, value, formatter = fmtPct) => {
    if (toNumber(value) === null) return "";
    return `<div><dt>${escapeHtml(label)}</dt><dd>${formatter(value)}</dd></div>`;
  };
  const note = stopRiskNote(pred);
  return `
    <div class="prediction-block${mainClass}">
      <span>${escapeHtml(horizonLabel)}</span>
      <strong>${fmtPct(pred.up_probability)}</strong>
      <dl>
        <div><dt>확률 기준</dt><dd>${escapeHtml(primaryLabel)}</dd></div>
        ${metric(downRiskLabel, pred.down_risk_calibrated ?? pred.down_risk)}
        ${metric("원시 손절", pred.down_risk_raw)}
        ${metric("OOS 백분위", pred.down_risk_oos_percentile)}
        ${metric("엄격 초과폭", pred.strict_stop_risk_gap)}
        ${metric("기대값", pred.expected_r, (value) => fmtNumber(value, 2))}
        ${metric("통과 기준", pred.threshold)}
        <div><dt>신뢰도</dt><dd>${escapeHtml(pred.confidence || "낮음")}</dd></div>
      </dl>
      ${note ? `<p>${escapeHtml(note)}</p>` : ""}
    </div>
  `;
}

function hasPrediction(pred) {
  return Boolean(pred) && (toNumber(pred.up_probability) !== null || toNumber(pred.down_risk) !== null || pred.model_name);
}

function hasCloseForecast(forecast) {
  return Boolean(forecast) && (toNumber(forecast.predicted_close) !== null || toNumber(forecast.predicted_return_pct) !== null || forecast.model_name);
}

function closeForecastBlock(item, horizonKey) {
  const forecast = item.close_forecasts?.[horizonKey] || {};
  const horizonLabel = forecast.horizon || horizonKey.toUpperCase();
  if (toNumber(forecast.predicted_close) === null && toNumber(forecast.predicted_return_pct) === null && !forecast.model_name) {
    return `
      <div class="prediction-block empty">
        <span>${escapeHtml(horizonLabel)} 종가</span>
        <strong>예측 없음</strong>
      </div>
    `;
  }
  const interval =
    toNumber(forecast.lower_80) !== null || toNumber(forecast.upper_80) !== null
      ? `${fmtPrice(forecast.lower_80, item.currency)} ~ ${fmtPrice(forecast.upper_80, item.currency)}`
      : "구간 없음";
  return `
    <div class="prediction-block">
      <span>${escapeHtml(horizonLabel)} 종가</span>
      <strong>${fmtPrice(forecast.predicted_close, item.currency)}</strong>
      <dl>
        <div><dt>예상 수익률</dt><dd>${fmtPctPoints(forecast.predicted_return_pct)}</dd></div>
        <div><dt>80% 구간</dt><dd>${escapeHtml(interval)}</dd></div>
        <div><dt>품질</dt><dd>${forecast.quality_pass ? "통과" : "표시 전용"}</dd></div>
        <div><dt>OOS</dt><dd>${fmtNumber(forecast.oos_event_count, 0)}</dd></div>
      </dl>
    </div>
  `;
}

function combinedForecastBlock(item, horizonKey) {
  const pred = item.predictions?.[horizonKey] || {};
  const forecast = item.close_forecasts?.[horizonKey] || {};
  const horizonLabel =
    pred.horizon ||
    forecast.horizon ||
    (horizonKey === "1d" ? "내일" : horizonKey === "5d" ? "5거래일" : horizonKey === "20d" ? "20거래일" : horizonKey);
  const hasUp = hasPrediction(pred);
  const hasClose = hasCloseForecast(forecast);
  if (!hasUp && !hasClose) {
    return `
      <div class="forecast-card empty">
        <span>${escapeHtml(horizonLabel)}</span>
        <strong>예측 없음</strong>
      </div>
    `;
  }
  const downRiskLabel = pred.is_main_model ? "하락 가능성" : "보정 손절";
  const primaryLabel = probabilityLabel(pred);
  const upMetric = (label, value, formatter = fmtPct) => {
    if (toNumber(value) === null) return "";
    return `<div><dt>${escapeHtml(label)}</dt><dd>${formatter(value)}</dd></div>`;
  };
  const interval =
    toNumber(forecast.lower_80) !== null || toNumber(forecast.upper_80) !== null
      ? `${fmtPrice(forecast.lower_80, item.currency)} ~ ${fmtPrice(forecast.upper_80, item.currency)}`
      : "구간 없음";
  return `
    <article class="forecast-card${pred.is_main_model ? " main" : ""}">
      <div class="forecast-card-head">
        <span>${escapeHtml(horizonLabel)}</span>
        <small>${escapeHtml(forecast.asof_date || pred.asof_date || item.main_prediction_as_of || "")}</small>
      </div>
      <div class="forecast-columns">
        <section title="${escapeHtml(pred.probability_definition || "")}">
          <span>${escapeHtml(primaryLabel)}</span>
          <strong>${hasUp ? fmtPct(pred.up_probability) : "예측 없음"}</strong>
          <dl>
            ${upMetric(downRiskLabel, pred.down_risk_calibrated ?? pred.down_risk)}
            ${upMetric("기대값", pred.expected_r, (value) => fmtNumber(value, 2))}
            ${upMetric("통과 기준", pred.threshold)}
            ${hasUp ? `<div><dt>신뢰도</dt><dd>${escapeHtml(pred.confidence || "낮음")}</dd></div>` : ""}
          </dl>
        </section>
        <section>
          <span>종가예측</span>
          <strong>${hasClose ? fmtPrice(forecast.predicted_close, item.currency) : "예측 없음"}</strong>
          <dl>
            ${hasClose ? `<div><dt>예상 수익률</dt><dd>${fmtPctPoints(forecast.predicted_return_pct)}</dd></div>` : ""}
            ${hasClose ? `<div><dt>80% 구간</dt><dd>${escapeHtml(interval)}</dd></div>` : ""}
            ${hasClose ? `<div><dt>품질</dt><dd>${forecast.quality_pass ? "통과" : "표시 전용"}</dd></div>` : ""}
            ${hasClose ? `<div><dt>OOS</dt><dd>${fmtNumber(forecast.oos_event_count, 0)}</dd></div>` : ""}
          </dl>
        </section>
      </div>
    </article>
  `;
}

function renderDetail() {
  const all = state.data?.symbols || [];
  const filtered = filteredSymbols();
  const pool = filtered.length ? filtered : all;
  const selected = pool.find((item) => item.symbol === state.selectedSymbol) || pool[0];
  if (!selected) {
    $("detailPanel").innerHTML = `<div class="empty-state">표시할 종목이 없습니다.</div>`;
    return;
  }
  state.selectedSymbol = selected.symbol;
  $("detailPanel").innerHTML = `
    <div class="focus-head">
      <div class="focus-id">
        <span>${escapeHtml(selected.symbol)}</span>
        <h2>${escapeHtml(selected.name)}</h2>
      </div>
      ${actionPill(selected.action_level)}
      <div class="focus-price">
        <strong>${fmtPrice(selected.price, selected.currency)}</strong>
        <span>${escapeHtml(selected.currency)} · ${escapeHtml(compactDate(selected.as_of))}</span>
      </div>
    </div>
    <p class="decision-text">${escapeHtml(selected.action_text)}</p>
    <div class="focus-section">
      <h3 class="focus-section-title">예측</h3>
      <div class="forecast-grid">
        ${combinedForecastBlock(selected, "1d")}
        ${combinedForecastBlock(selected, "5d")}
        ${combinedForecastBlock(selected, "20d")}
      </div>
    </div>
    <div class="focus-section">
      <h3 class="focus-section-title">가격 추이</h3>
      ${chartSvg(selected)}
    </div>
    <div class="focus-grid">
      <div class="scenario-list">
        <div><span>상승 조건</span><strong>${escapeHtml(selected.scenario?.rise_if)}</strong></div>
        <div><span>하락 조건</span><strong>${escapeHtml(selected.scenario?.fall_if)}</strong></div>
        <div><span>다음 확인</span><strong>${escapeHtml(selected.scenario?.next_check)}</strong></div>
      </div>
      <div class="risk-strip">
        <div><span>최대 비중</span><strong>${fmtWeight(selected.risk?.approved_weight)}</strong></div>
        <div><span>손절가</span><strong>${fmtPrice(selected.risk?.stop_price, selected.currency)}</strong></div>
        <div><span>목표가</span><strong>${fmtPrice(selected.risk?.target_price, selected.currency)}</strong></div>
        <div><span>손절 거리</span><strong>${fmtPct(selected.risk?.risk_pct)}</strong></div>
      </div>
    </div>
  `;
}

function renderSystemPanel() {
  const system = state.data?.system || {};
  const daily = system.daily_update || {};
  const symbols = state.data?.symbols || [];
  const auditFailures = daily.full_audit_required_failures || daily.audit_required_failures || [];
  const smhModes = daily.portfolio_risk?.smh_beta_limit_modes || [];
  const rows = [
    ["데이터 상태", system.data_status],
    ["모델 상태", system.model_status],
    ["Full Audit", daily.full_audit_status || "확인 필요"],
    ["감사 실패", auditFailures.length ? auditFailures.map((row) => row.component || row.details || row.status).join(" · ") : "없음"],
    ["최근 데이터", compactDate(system.latest_data_date)],
    ["가장 오래된 데이터", system.max_data_age_days === null || system.max_data_age_days === undefined ? "없음" : `${system.max_data_age_days}일 전`],
    ["데이터 문제 수", system.daily_data_issue_count ?? 0],
    ["모델 품질 실패", system.model_quality_failed ?? 0],
    ["실거래", system.live_trading_status === "DISABLED_BY_DESIGN" ? "비활성" : system.live_trading_status],
  ];
  const cards = [
    ["해외 10 예측", coverageText(daily.coverage?.latest_prediction_overseas), `rows ${daily.latest_predictions?.foreign_rows ?? 0} · fallback ${daily.latest_predictions?.fallback_rows ?? 0}`],
    ["Top12 커버리지", coverageText(daily.coverage?.latest_prediction_top12), `market data ${daily.market_data?.top12_rows ?? 0} · ${statusCountsText(daily.market_data?.status_counts)}`],
    ["Pooled 데이터", coverageText(daily.coverage?.pooled_feature_overseas), `label ${coverageText(daily.coverage?.pooled_label_overseas)} · ${failedText(daily.quality?.pooled_dataset)}`],
    ["Threshold", `${daily.threshold_sensitivity?.grid_rows ?? 0} grid`, `${daily.threshold_sensitivity?.recommendation || "확인 필요"} · ${failedText(daily.quality?.threshold_sensitivity)}`],
    ["Risk/OMS", `SMH ${smhModes.join("|") || "확인 필요"}`, `approved ${daily.portfolio_risk?.approved_overseas ?? 0} · rejected ${daily.portfolio_risk?.rejected_overseas ?? 0}`],
    ["Paper OMS", `${daily.paper_oms?.order_rows ?? 0} orders`, `${statusCountsText(daily.paper_oms?.order_status_counts)} · fills ${daily.paper_oms?.fill_rows ?? 0}`],
  ];
  $("systemPanel").innerHTML = `
    <div class="section-head">
      <div>
        <h2>시스템 상태</h2>
        <p>${symbols.length}개 운영 종목 · 최신 로그 ${escapeHtml(daily.latest_log || "확인 필요")}</p>
      </div>
      ${statusPill(daily.full_audit_status || system.data_status)}
    </div>
    <div class="system-grid">
      ${rows.map(([label, value]) => `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("")}
    </div>
    <div class="system-summary-grid">
      ${cards.map(([label, value, detail]) => `
        <article class="system-summary-card ${value === "FAIL" ? "bad" : value === "PASS" ? "good" : ""}">
          <span>${escapeHtml(label)}</span>
          <strong>${escapeHtml(value)}</strong>
          <p>${escapeHtml(detail)}</p>
        </article>
      `).join("")}
    </div>
  `;
}

function showCurrentView() {
  const isSystem = state.activeTab === "system";
  const isRun = state.activeTab === "run";
  const isPortfolio = state.activeTab === "portfolio";
  $("dashboardView").classList.toggle("hidden", isSystem || isRun || isPortfolio);
  $("systemPanel").classList.toggle("hidden", !isSystem);
  $("runPanel").classList.toggle("hidden", !isRun);
  $("portfolioPanel").classList.toggle("hidden", !isPortfolio);
  $("actionRow").classList.toggle("hidden", isPortfolio);
}

function render() {
  if (!state.data) return;
  setFx(state.data.fx || {});
  setSystem(state.data.system || {});
  renderActionRow();
  renderSystemPanel();
  showCurrentView();
  if (state.activeTab === "portfolio") {
    renderPortfolioPanel();
  } else if (state.activeTab !== "system" && state.activeTab !== "run") {
    const pool = filteredSymbols();
    if (pool.length && !pool.some((item) => item.symbol === state.selectedSymbol)) {
      state.selectedSymbol = pool[0].symbol;
    }
    renderSymbolGrid();
    renderDetail();
  }
}

// ---------------------------------------------------------------------------
// Portfolio cockpit (manual transactions -> holdings, weights, returns,
// rebalance execution tickets). Shares the /api/portfolio backend with the
// operations dashboard. Values arrive in engine USD; the client converts to
// KRW (KRW = usd * usdkrw). No live broker is ever called.
// ---------------------------------------------------------------------------

const PF_NAMES = {
  NVDA: "엔비디아", TSM: "TSMC(ADR)", AVGO: "브로드컴", AMD: "AMD", INTC: "인텔",
  MU: "마이크론 테크놀로지", TXN: "텍사스 인스트루먼트", LRCX: "램 리서치",
  AMAT: "어플라이드 머티리얼즈", QCOM: "퀄컴", "005930.KS": "삼성전자", "000660.KS": "SK하이닉스",
};
const PF_ACTION_LABEL = { ADD: "추가매수", BUY_NEW: "신규매수", TRIM: "비중축소", EXIT: "청산", HOLD: "보유" };
const PF_ACTION_TONE = { ADD: "good", BUY_NEW: "good", TRIM: "warn", EXIT: "bad", HOLD: "neutral" };

function pfName(symbol) {
  return PF_NAMES[String(symbol || "").toUpperCase()] || symbol;
}

function pfFx() {
  return toNumber(state.portfolio?.fx?.usdkrw) || null;
}

function pfDisp(usd) {
  const n = toNumber(usd);
  if (n === null) return null;
  if (state.portfolioCurrency === "KRW") {
    const fx = pfFx();
    return fx ? n * fx : null;
  }
  return n;
}

function pfMoney(usd) {
  const v = pfDisp(usd);
  if (v === null) return "없음";
  if (state.portfolioCurrency === "KRW") return "₩" + Math.round(v).toLocaleString("ko-KR");
  return "$" + v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function pfSigned(usd) {
  const v = pfDisp(usd);
  if (v === null) return "없음";
  return (v > 0 ? "+" : "") + pfMoney(usd);
}

function pfShares(qty) {
  const n = toNumber(qty);
  if (n === null) return "없음";
  return n.toLocaleString("ko-KR", { minimumFractionDigits: 0, maximumFractionDigits: 6 });
}

function pfPnlTone(value) {
  const n = toNumber(value);
  if (n === null || n === 0) return "flat";
  return n > 0 ? "up" : "down";
}

async function loadPortfolio() {
  try {
    state.portfolio = await fetchJson("/api/portfolio");
    state.portfolioLoaded = true;
    renderPortfolioPanel();
  } catch (error) {
    state.portfolioLoaded = true;
    $("pfHero").innerHTML = `<div class="pf-empty bad">포트폴리오를 불러오지 못했습니다: ${escapeHtml(error.message)}</div>`;
  }
}

function renderPortfolioPanel() {
  const p = state.portfolio;
  if (!p) {
    $("pfHero").innerHTML = `<div class="pf-empty">불러오는 중…</div>`;
    return;
  }
  document.querySelectorAll("#pfCurrency button").forEach((b) =>
    b.classList.toggle("active", b.dataset.cur === state.portfolioCurrency)
  );
  renderPfHero(p);
  renderPfTickets(p);
  renderPfHoldings(p);
  renderPfAllocation(p);
  renderPfTransactions(p);
}

function renderPfHero(p) {
  const t = p.totals || {};
  const mv = toNumber(t.market_value_usd) || 0;
  const cost = toNumber(t.cost_basis_usd) || 0;
  const totalPnl = mv - cost;
  const totalPct = cost > 0 ? (totalPnl / cost) * 100 : 0;
  const daily = toNumber(t.daily_pnl_usd) || 0;
  const prevMv = mv - daily;
  const dailyPct = prevMv > 0 ? (daily / prevMv) * 100 : 0;
  const fx = pfFx();
  const recon = p.reconciliation_status || "PASS";
  $("pfHero").innerHTML = `
    <div class="pf-total">
      <span>내 투자 · 평가금</span>
      <strong>${pfMoney(mv)}</strong>
      <em class="pnl ${pfPnlTone(totalPnl)}">${pfSigned(totalPnl)} (${fmtPctPoints(totalPct)})</em>
    </div>
    <div class="pf-metrics">
      <div><span>원금</span><b>${pfMoney(cost)}</b></div>
      <div><span>총 수익</span><b class="pnl ${pfPnlTone(totalPnl)}">${pfSigned(totalPnl)} (${fmtPctPoints(totalPct)})</b></div>
      <div><span>일간 수익</span><b class="pnl ${pfPnlTone(daily)}">${pfSigned(daily)} (${fmtPctPoints(dailyPct)})</b></div>
      <div><span>보유 종목</span><b>${t.n_holdings ?? 0}</b></div>
      <div><span>환율</span><b>${fx ? "₩" + Math.round(fx).toLocaleString("ko-KR") : "없음"}</b></div>
      <div><span>정합성</span><b class="${recon === "PASS" ? "ok" : "bad"}">${escapeHtml(recon)}</b></div>
    </div>`;
}

const PF_WARN_LABEL = {
  NO_ACTIONABLE_SIGNAL: "유효 신호 없음",
  REQUESTED_WEIGHT_ZERO: "권장 비중 0",
  BETA_TO_SPY_WARN: "시장 베타 높음",
  BETA_TO_SMH_WARN: "반도체 베타 높음",
  BETA_TO_SPY_LIMIT: "시장 베타 한도 초과",
  BETA_TO_SMH_LIMIT: "반도체 베타 한도 초과",
  ORDER_CAPACITY_EXCEEDED: "주문 규모 초과",
  INTRADAY_COVERAGE_LOW: "분봉 데이터 부족",
  INTRADAY_VOL_HIGH: "분봉 변동성 높음",
  INTENT_NOT_APPROVED: "주문 미승인",
};

function pfWarnText(warning) {
  if (!warning) return "";
  return String(warning)
    .split("|")
    .map((w) => w.trim())
    .filter((w) => w && w !== "PASS")
    .map((w) => PF_WARN_LABEL[w] || w)
    .join(" · ");
}

function pfTicketCard(t) {
  const action = String(t.action || "HOLD");
  const tone = PF_ACTION_TONE[action] || "neutral";
  const label = PF_ACTION_LABEL[action] || action;
  const delta = toNumber(t.delta_shares) || 0;
  const verb = delta > 0 ? "매수" : "매도";
  const instruction = action === "HOLD"
    ? "현재 비중 유지"
    : `${pfShares(Math.abs(delta))}주 ${verb} <span class="muted">@ ${pfMoney(t.ref_price_usd)}</span>`;
  const pSucc = toNumber(t.p_success_20d);
  const warn = pfWarnText(t.warning);
  return `
    <article class="pf-ticket ${tone}">
      <div class="pf-ticket-top">
        <span class="pf-badge ${tone}">${escapeHtml(label)}</span>
        <span class="pf-ticket-name"><b>${escapeHtml(pfName(t.symbol))}</b><i>${escapeHtml(t.symbol)}</i></span>
      </div>
      <div class="pf-ticket-instruction">${instruction}</div>
      <div class="pf-ticket-facts">
        <div><span>현재 → 목표 비중</span><b>${fmtPct(t.current_weight)} → ${fmtPct(t.target_weight)}</b></div>
        <div><span>20일 성공확률</span><b>${pSucc === null ? "없음" : fmtPctPoints(pSucc * 100, 0)}</b></div>
        <div><span>손절가</span><b>${pfMoney(t.stop_price_usd)}</b></div>
        <div><span>목표가</span><b>${pfMoney(t.target_price_usd)}</b></div>
      </div>
      ${warn ? `<p class="pf-ticket-warn">⚠ ${escapeHtml(warn)}</p>` : ""}
    </article>`;
}

function renderPfTickets(p) {
  const all = p.rebalance_queue || [];
  const target = $("pfTickets");
  if (!all.length) {
    target.innerHTML = `<div class="pf-empty">표시할 실행 티켓이 없습니다. 거래를 입력하면 보유 비중 대비 모델 목표 비중 차이가 계산됩니다.</div>`;
    return;
  }
  const held = all.filter((t) => String(t.action) !== "BUY_NEW");
  const newBuys = all.filter((t) => String(t.action) === "BUY_NEW");
  const heldHtml = held.length
    ? `<div class="pf-tickets">${held.map(pfTicketCard).join("")}</div>`
    : `<div class="pf-empty">보유 종목에 대한 조정 티켓이 없습니다.</div>`;
  let newHtml = "";
  if (newBuys.length) {
    newHtml = `
      <details class="pf-newbuys"${state.pfShowNewBuys ? " open" : ""}>
        <summary>
          <span class="pf-newbuys-title">신규 매수 추천 ${newBuys.length}개</span>
          <span class="pf-newbuys-hint">미보유 유니버스 종목에 대한 모델 제안 · 펼치기</span>
        </summary>
        <div class="pf-tickets pf-newbuys-grid">${newBuys.map(pfTicketCard).join("")}</div>
      </details>`;
  }
  target.innerHTML = heldHtml + newHtml;
  const details = target.querySelector(".pf-newbuys");
  if (details) details.addEventListener("toggle", () => { state.pfShowNewBuys = details.open; });
}

function renderPfHoldings(p) {
  const holdings = p.holdings || [];
  const target = $("pfHoldings");
  if (!holdings.length) {
    target.innerHTML = `<div class="pf-empty">보유 종목이 없습니다. "+ 거래 입력"으로 매수 기록을 추가하세요.</div>`;
    return;
  }
  const rows = holdings
    .map((h) => {
      const tone = PF_ACTION_TONE[String(h.recommendation || "HOLD")] || "neutral";
      const label = PF_ACTION_LABEL[String(h.recommendation || "HOLD")] || (h.recommendation || "-");
      return `<tr>
        <td class="pf-sym"><b>${escapeHtml(pfName(h.symbol))}</b><i>${escapeHtml(h.symbol)}</i></td>
        <td class="num">${pfShares(h.shares)}</td>
        <td class="num">${pfMoney(h.avg_price_usd)}</td>
        <td class="num">${pfMoney(h.current_price_usd)}</td>
        <td class="num">${pfMoney(h.market_value_usd)}</td>
        <td class="num">${fmtPctPoints(h.weight_pct)}</td>
        <td class="num pnl ${pfPnlTone(h.return_pct)}">${h.return_pct === null ? "없음" : fmtPctPoints(h.return_pct)}</td>
        <td><span class="pf-pill ${tone}">${escapeHtml(label)}</span></td>
      </tr>`;
    })
    .join("");
  target.innerHTML = `<table class="pf-table"><thead><tr>
      <th>종목</th><th class="num">보유 주수</th><th class="num">평단</th><th class="num">현재가</th>
      <th class="num">평가금</th><th class="num">비중</th><th class="num">수익률</th><th>권고</th>
    </tr></thead><tbody>${rows}</tbody></table>`;
}

function renderPfAllocation(p) {
  const holdings = (p.holdings || []).filter((h) => (toNumber(h.market_value_usd) || 0) > 0);
  const target = $("pfAllocation");
  if (!holdings.length) {
    target.innerHTML = `<div class="pf-empty">데이터 없음</div>`;
    return;
  }
  const total = holdings.reduce((s, h) => s + (toNumber(h.market_value_usd) || 0), 0) || 1;
  const palette = ["#2b62e8", "#0a8f5b", "#b5740a", "#7b61ff", "#cc3b30", "#1aa3a3", "#8e9aaf", "#c44dff"];
  const bars = holdings
    .map((h, i) => {
      const w = ((toNumber(h.market_value_usd) || 0) / total) * 100;
      const c = palette[i % palette.length];
      return `<li>
        <div class="pf-alloc-label"><span class="pf-dot" style="background:${c}"></span>${escapeHtml(pfName(h.symbol))}<b>${fmtPctPoints(w)}</b></div>
        <div class="pf-alloc-track"><div class="pf-alloc-fill" style="width:${Math.max(w, 1)}%;background:${c}"></div></div>
      </li>`;
    })
    .join("");
  target.innerHTML = `<ul class="pf-alloc-list">${bars}</ul>`;
}

function renderPfTransactions(p) {
  const txs = p.transactions || [];
  const target = $("pfTransactions");
  if (!txs.length) {
    target.innerHTML = `<div class="pf-empty">입력한 거래가 없습니다.</div>`;
    return;
  }
  const rows = txs
    .map((t) => `<tr class="pf-tx-row" data-txid="${escapeHtml(t.transaction_id)}">
      <td>${escapeHtml(compactDate(t.trade_date))}</td>
      <td class="pf-sym"><b>${escapeHtml(pfName(t.symbol))}</b><i>${escapeHtml(t.symbol)}</i></td>
      <td><span class="pf-pill ${String(t.side).toUpperCase() === "BUY" ? "good" : "warn"}">${String(t.side).toUpperCase() === "BUY" ? "매수" : "매도"}</span></td>
      <td class="num">${pfShares(t.quantity)}</td>
      <td class="num">${fmtNumber(t.price)} ${escapeHtml(t.price_currency || "")}</td>
      <td>${escapeHtml(t.note || "")}</td>
    </tr>`)
    .join("");
  target.innerHTML = `<table class="pf-table"><thead><tr>
      <th>거래일</th><th>종목</th><th>구분</th><th class="num">주수</th><th class="num">체결가</th><th>메모</th>
    </tr></thead><tbody>${rows}</tbody></table>`;
  target.querySelectorAll(".pf-tx-row").forEach((row) => {
    row.addEventListener("click", () => {
      const tx = (state.portfolio?.transactions || []).find((x) => String(x.transaction_id) === row.dataset.txid);
      if (tx) openPfModal(tx);
    });
  });
}

function populatePfSymbols() {
  const select = $("pfSymbol");
  if (!select) return;
  const universe = state.portfolio?.universe || Object.keys(PF_NAMES);
  select.innerHTML = universe.map((s) => `<option value="${escapeHtml(s)}">${escapeHtml(pfName(s))} (${escapeHtml(s)})</option>`).join("");
}

function openPfModal(tx) {
  const form = $("pfForm");
  form.reset();
  populatePfSymbols();
  $("pfFormError").classList.add("hidden");
  const editing = Boolean(tx && tx.transaction_id);
  $("pfModalTitle").textContent = editing ? "거래 수정" : "거래 입력";
  $("pfDeleteBtn").classList.toggle("hidden", !editing);
  form.elements.transaction_id.value = editing ? tx.transaction_id : "";
  if (tx) {
    if (tx.symbol) form.elements.symbol.value = String(tx.symbol).toUpperCase();
    form.elements.side.value = String(tx.side || "BUY").toUpperCase();
    form.elements.trade_date.value = compactDate(tx.trade_date) !== "날짜 없음" ? compactDate(tx.trade_date) : "";
    form.elements.quantity.value = tx.quantity ?? "";
    form.elements.price.value = tx.price ?? "";
    form.elements.price_currency.value = String(tx.price_currency || "USD").toUpperCase();
    form.elements.fees.value = tx.fees ?? 0;
    form.elements.note.value = tx.note ?? "";
  } else {
    form.elements.trade_date.value = new Date().toISOString().slice(0, 10);
    form.elements.price_currency.value = "USD";
  }
  $("pfModalBackdrop").classList.remove("hidden");
}

function closePfModal() {
  $("pfModalBackdrop").classList.add("hidden");
}

async function submitPfForm(event) {
  event.preventDefault();
  const entries = Object.fromEntries(new FormData(event.currentTarget).entries());
  const id = String(entries.transaction_id || "").trim();
  const payload = {
    action: id ? "update" : "add",
    transaction: {
      transaction_id: id,
      trade_date: entries.trade_date,
      symbol: entries.symbol,
      side: entries.side,
      quantity: entries.quantity,
      price: entries.price,
      price_currency: entries.price_currency,
      fees: entries.fees,
      note: entries.note,
    },
  };
  $("pfSubmitBtn").disabled = true;
  try {
    state.portfolio = await fetchJson("/api/portfolio/transaction", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderPortfolioPanel();
    closePfModal();
  } catch (error) {
    const err = $("pfFormError");
    err.textContent = error.message;
    err.classList.remove("hidden");
  } finally {
    $("pfSubmitBtn").disabled = false;
  }
}

async function deletePfTx() {
  const id = $("pfForm").elements.transaction_id.value;
  if (!id || !confirm("이 거래를 삭제할까요?")) return;
  try {
    state.portfolio = await fetchJson("/api/portfolio/transaction", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "delete", transaction_id: id }),
    });
    renderPortfolioPanel();
    closePfModal();
  } catch (error) {
    const err = $("pfFormError");
    err.textContent = error.message;
    err.classList.remove("hidden");
  }
}

function wirePortfolioEvents() {
  $("pfAddBtn").addEventListener("click", () => openPfModal(null));
  $("pfModalClose").addEventListener("click", closePfModal);
  $("pfCancelBtn").addEventListener("click", closePfModal);
  $("pfDeleteBtn").addEventListener("click", deletePfTx);
  $("pfForm").addEventListener("submit", submitPfForm);
  $("pfModalBackdrop").addEventListener("click", (event) => {
    if (event.target === $("pfModalBackdrop")) closePfModal();
  });
  $("pfCurrency").addEventListener("click", (event) => {
    const btn = event.target.closest("button[data-cur]");
    if (!btn) return;
    state.portfolioCurrency = btn.dataset.cur;
    renderPortfolioPanel();
  });
}

async function loadDashboard() {
  const data = await fetchJson("/api/investor-dashboard");
  state.data = data;
  if (!state.selectedSymbol && data.symbols?.length) state.selectedSymbol = data.symbols[0].symbol;
  updateRunState(data.run || {});
  render();
}

async function fetchLiveFx() {
  const result = await fetchJson("/api/fx/refresh", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  setFx(result.fx || {});
  return result.fx || {};
}

async function refreshFx() {
  const button = $("fxRefreshBtn");
  const previousText = button.textContent;
  button.disabled = true;
  button.textContent = "조회 중";
  try {
    await fetchLiveFx();
    await loadDashboard();
    button.textContent = "완료";
    setTimeout(() => {
      button.textContent = previousText;
    }, 1200);
  } catch (error) {
    $("fxRate").textContent = "조회 실패";
    $("fxRate").dataset.tone = "bad";
    $("fxRate").parentElement.title = error.message;
    button.textContent = previousText;
  } finally {
    button.disabled = false;
  }
}

async function calculateFx() {
  const button = $("fxCalcBtn");
  const previousText = button.textContent;
  button.disabled = true;
  button.textContent = "조회";
  $("fxCalcResult").textContent = "계산 중";
  try {
    const fx = await fetchLiveFx();
    const usdkrw = fx.usdkrw ?? fx.fx_close;
    $("fxCalcResult").textContent = fmtExchangeResult($("fxAmount").value, $("fxDirection").value, usdkrw);
    await loadDashboard();
  } catch (error) {
    $("fxCalcResult").textContent = "실패";
    $("fxCalcResult").title = error.message;
  } finally {
    button.textContent = previousText;
    button.disabled = false;
  }
}

function setActiveTab(tab) {
  state.activeTab = tab;
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tab);
  });
  showCurrentView();
  render();
  if (tab === "portfolio") {
    if (!state.portfolioLoaded) loadPortfolio();
    else renderPortfolioPanel();
  }
}

function runPayloadFromForm() {
  const payload = {
    mode: $("runMode").value,
    start: $("runStart").value || "2016-05-12",
    end: $("runEnd").value || new Date().toISOString().slice(0, 10),
    skip_charts: $("skipCharts").checked,
    continue_on_error: true,
  };
  if (payload.mode === "downstream") payload.skip_data_refresh = true;
  return payload;
}

async function startRun(event) {
  event.preventDefault();
  $("runBtn").disabled = true;
  try {
    const run = await fetchJson("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(runPayloadFromForm()),
    });
    updateRunState(run);
    scheduleRunPoll();
  } catch (error) {
    $("runStatusText").textContent = error.message;
  } finally {
    $("runBtn").disabled = false;
  }
}

async function stopRun() {
  try {
    updateRunState(await fetchJson("/api/stop", { method: "POST" }));
  } catch (error) {
    $("runStatusText").textContent = error.message;
  }
}

function updateRunState(run) {
  const status = run.status || "IDLE";
  $("runStatusText").textContent = `${run.mode_label || run.mode || "대기"} · ${status}`;
  $("runLog").textContent = (run.log || []).join("\n");
  $("runBtn").disabled = status === "RUNNING";
  $("stopRunBtn").disabled = status !== "RUNNING";
}

function scheduleRunPoll() {
  if (state.runTimer) clearInterval(state.runTimer);
  state.runTimer = setInterval(async () => {
    try {
      const run = await fetchJson("/api/run-status");
      updateRunState(run);
      if (run.status !== "RUNNING") {
        clearInterval(state.runTimer);
        state.runTimer = null;
        await loadDashboard();
      }
    } catch {
      clearInterval(state.runTimer);
      state.runTimer = null;
    }
  }, 4000);
}

function init() {
  $("runEnd").value = new Date().toISOString().slice(0, 10);
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.addEventListener("click", () => setActiveTab(button.dataset.tab));
  });
  $("refreshBtn").addEventListener("click", loadDashboard);
  $("fxRefreshBtn").addEventListener("click", refreshFx);
  $("fxCalcBtn").addEventListener("click", calculateFx);
  $("runForm").addEventListener("submit", startRun);
  $("stopRunBtn").addEventListener("click", stopRun);
  wirePortfolioEvents();
  loadDashboard().catch((error) => {
    $("actionRow").innerHTML = `<article class="action-card blocked"><span>오류</span><h2>대시보드 로드 실패</h2><p>${escapeHtml(error.message)}</p></article>`;
  });
}

document.addEventListener("DOMContentLoaded", init);
