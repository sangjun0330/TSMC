const state = {
  data: null,
  activeTab: "all",
  selectedSymbol: null,
  runTimer: null,
  portfolio: null,
  portfolioCurrency: "KRW",
  portfolioLoaded: false,
  pfShowNewBuys: false,
  signalFilter: "all",
  live: null,
  liveTimer: null,
  liveHistory: {},
};

const $ = (id) => document.getElementById(id);

const LIVE_POLL_MS = 30000;
// Max live-price points kept per symbol while the dashboard stays open
// (240 × 30s ≈ 2 hours of intraday session history).
const LIVE_HISTORY_MAX = 240;

const marketStateLabel = {
  REGULAR: "장중",
  PRE: "장전",
  POST: "장후",
  CLOSED: "장마감",
  UNKNOWN: "확인 중",
};

function liveQuote(symbol) {
  return state.live?.quotes?.[symbol] || null;
}

function isLiveMarket(quote) {
  return Boolean(quote) && ["REGULAR", "PRE", "POST"].includes(String(quote.market_state || ""));
}

function fmtSignedPct(value, digits = 2) {
  const num = toNumber(value);
  if (num === null) return "";
  return `${num >= 0 ? "+" : ""}${num.toFixed(digits)}%`;
}

function moveTone(value) {
  const num = toNumber(value);
  if (num === null || num === 0) return "flat";
  return num > 0 ? "up" : "down";
}

// Distance from the live price to a fixed level (stop/target/forecast), in %.
function pctTo(level, from) {
  const a = toNumber(level);
  const b = toNumber(from);
  if (a === null || b === null || b === 0) return null;
  return (a / b - 1) * 100;
}

const actionLabels = {
  execute_candidate: "실행후보",
  watch: "관찰",
  reduce: "매수금지",
  avoid: "매수금지",
  blocked: "매수금지",
};

const signalFilterLabels = {
  all: "전체",
  positive: "실행후보",
  neutral: "관찰",
  defensive: "매수금지",
};

const signalFilterOrder = ["all", "positive", "neutral", "defensive"];

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

function fmtChipPrice(value, currency) {
  const num = toNumber(value);
  if (num === null) return "현재가 대기";
  if (currency === "KRW") return `₩${Math.round(num).toLocaleString("ko-KR")}`;
  return `$${num.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function fmtChipChange(value) {
  const num = toNumber(value);
  if (num === null) return "전일대비 대기";
  return `${num >= 0 ? "+" : ""}${num.toFixed(2)}%`;
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

function signalProfile(item) {
  return item?.signal_profile || {};
}

function signalLabel(item) {
  const profile = signalProfile(item);
  return profile.tier_label || actionLabels[item?.action_level] || "관찰";
}

function signalToneClass(item) {
  const tier = signalProfile(item).tier;
  if (tier === "positive") return "execute_candidate";
  if (tier === "neutral") return "watch";
  return item?.action_level || "avoid";
}

function signalMetricText(item) {
  const metrics = signalProfile(item).metrics || {};
  const parts = [];
  if (toNumber(metrics.p_success_20d) !== null) parts.push(`성공 ${fmtPct(metrics.p_success_20d)}`);
  if (toNumber(metrics.p_stop_hit_20d) !== null) parts.push(`손절 ${fmtPct(metrics.p_stop_hit_20d)}`);
  if (toNumber(metrics.expected_r_20d) !== null) parts.push(`기대 ${fmtNumber(metrics.expected_r_20d, 2)}R`);
  if (toNumber(metrics.forecast_return_20d) !== null) parts.push(`종가 ${fmtSignedPct(metrics.forecast_return_20d)}`);
  return parts.join(" · ") || "20일 예측 지표 없음";
}

function primaryBlockReason(item) {
  return (item?.new_buy_block_reasons || [])[0] || null;
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
  const topCandidate = queue.find((item) => item.signal_profile?.new_buy_allowed || item.signal_profile?.tier === "positive" || item.action_level === "execute_candidate");
  const fallbackFocus = topCandidate || queue.find((item) => item.signal_profile?.tier === "neutral") || queue[0];
  const hardAvoid = symbols.find((item) => (item.new_buy_block_reasons || []).length);
  const system = data?.system || {};
  const daily = system.daily_update || {};
  const top12Coverage = coverageText(daily.coverage?.latest_prediction_top12);
  const foreignCoverage = coverageText(daily.coverage?.latest_prediction_overseas);
  const fallbackRows = daily.latest_predictions?.fallback_rows ?? "확인 필요";
  const cards = [
    {
      label: "실행후보",
      title: topCandidate ? (topCandidate.signal_profile?.headline || topCandidate.action) : "우위 신호 없음",
      meta: fallbackFocus ? `${fallbackFocus.name} · ${fallbackFocus.symbol}` : "새 매수는 대기",
      body: topCandidate
        ? `${topCandidate.main_reason} · ${signalMetricText(topCandidate)} · 최대 ${fmtWeight(topCandidate.max_weight)}`
        : fallbackFocus
          ? `${signalLabel(fallbackFocus)} · ${signalMetricText(fallbackFocus)}`
          : "20일 예측 우위가 생길 때까지 대기",
      tone: topCandidate?.action_level || fallbackFocus?.action_level || "neutral",
    },
    {
      label: "매수금지",
      title: hardAvoid ? (primaryBlockReason(hardAvoid)?.label || hardAvoid.action_text) : "공통 리스크 원칙",
      meta: hardAvoid ? `${hardAvoid.name} · ${hardAvoid.symbol}` : "공통 원칙",
      body: hardAvoid ? (primaryBlockReason(hardAvoid)?.detail || hardAvoid.warning) : "손절 기준 없는 진입은 하지 않음",
      tone: hardAvoid?.action_level || "avoid",
    },
    {
      label: "신뢰 상태",
      title: `${system.data_status || "확인 필요"} · ${daily.full_audit_status || system.model_status || "확인 필요"}`,
      meta: `Top12 ${top12Coverage} · 해외 ${foreignCoverage}`,
      body: system.live_trading_status === "DISABLED_BY_DESIGN" ? `${system.main_model || "20일 예측"} 메인 · fallback ${fallbackRows} · 실거래 주문 비활성` : "실거래 상태 확인 필요",
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
  if (state.signalFilter === "all") return symbols;
  return symbols.filter((item) => (signalProfile(item).tier || "unknown") === state.signalFilter);
}

function renderSignalFilter(counts) {
  const target = $("signalFilter");
  if (!target) return;
  const total = Object.values(counts || {}).reduce((sum, value) => sum + Number(value || 0), 0);
  target.innerHTML = signalFilterOrder
    .map((key) => {
      const count = key === "all" ? total : counts?.[key] || 0;
      const active = state.signalFilter === key ? "active" : "";
      return `
        <button class="signal-filter-button ${active}" type="button" data-filter="${escapeHtml(key)}" aria-pressed="${active ? "true" : "false"}">
          <span>${escapeHtml(signalFilterLabels[key])}</span>
          <b>${count}</b>
        </button>
      `;
    })
    .join("");
  target.querySelectorAll(".signal-filter-button").forEach((button) => {
    button.addEventListener("click", () => {
      state.signalFilter = button.dataset.filter || "all";
      state.selectedSymbol = null;
      render();
    });
  });
}

function renderSymbolGrid() {
  const all = state.data?.symbols || [];
  const counts = all.reduce((acc, item) => {
    const tier = signalProfile(item).tier || "unknown";
    acc[tier] = (acc[tier] || 0) + 1;
    return acc;
  }, {});
  renderSignalFilter(counts);
  const symbols = filteredSymbols();
  $("boardSummary").textContent = `실행후보 ${counts.positive || 0} · 관찰 ${counts.neutral || 0} · 매수금지 ${counts.defensive || 0} · 칩 = 실시간 현재가 / 전일 종가 대비`;
  if (!symbols.length) {
    $("symbolGrid").innerHTML = `<div class="empty-state compact">${escapeHtml(signalFilterLabels[state.signalFilter] || "선택한 분류")} 종목이 없습니다.</div>`;
    return;
  }
  $("symbolGrid").innerHTML = symbols
    .map((item) => {
      const selected = item.symbol === state.selectedSymbol ? "selected" : "";
      const q = liveQuote(item.symbol);
      const livePrice = q?.price ?? item.price;
      const liveCurrency = q?.currency || item.currency;
      const changePct = q ? q.change_pct : null;
      const priceText = fmtChipPrice(livePrice, liveCurrency);
      const changeText = fmtChipChange(changePct);
      const changeTone = moveTone(changePct);
      const liveTitle = q
        ? ` · 실시간 현재가 ${fmtPrice(q.price, liveCurrency)} · 전일 종가 ${fmtPrice(q.prev_close, liveCurrency)} 대비 ${fmtSignedPct(q.change_pct)}`
        : ` · 실시간 현재가 대기 · 기준일 종가 ${fmtPrice(item.price, item.currency)}`;
      return `
        <button class="symbol-chip ${escapeHtml(signalToneClass(item))} ${selected}" type="button" role="tab" aria-selected="${selected ? "true" : "false"}" data-symbol="${escapeHtml(item.symbol)}" title="${escapeHtml(item.name)}${liveTitle}">
          <span class="chip-dot" aria-hidden="true"></span>
          <b class="chip-ticker">${escapeHtml(item.symbol)}</b>
          <span class="chip-signal">${escapeHtml(priceText)}</span>
          <span class="chip-prob ${escapeHtml(changeTone)}">${escapeHtml(changeText)}</span>
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

// Append one live price point per symbol from the latest quote poll.
function recordLiveHistory(quotes) {
  const now = Date.now();
  const store = state.liveHistory || (state.liveHistory = {});
  for (const [symbol, quote] of Object.entries(quotes || {})) {
    const price = toNumber(quote?.price);
    if (price === null) continue;
    const arr = store[symbol] || (store[symbol] = []);
    arr.push({ ts: now, price, currency: quote.currency || "USD" });
    if (arr.length > LIVE_HISTORY_MAX) arr.splice(0, arr.length - LIVE_HISTORY_MAX);
  }
}

function fmtClock(ts) {
  const d = new Date(ts);
  const p = (n) => String(n).padStart(2, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

// Generic responsive line chart with hover crosshair support. `points` is a list
// of { value, label }. Reference lines (target/stop) render as dashed lines with
// HTML labels. Returns chart-wrap markup, or an empty-state when too few points.
function lineChartMarkup(points, opts = {}) {
  const { currency = "USD", lines = [], ariaLabel = "가격 차트", emptyText = "데이터 없음" } = opts;
  const clean = (points || []).filter((p) => toNumber(p.value) !== null);
  if (clean.length < 2) {
    return `<div class="empty-chart">${escapeHtml(emptyText)}</div>`;
  }
  const width = 680;
  const height = 220;
  const values = clean.map((p) => Number(p.value));
  const refs = lines.map((l) => toNumber(l.value)).filter((v) => v !== null);
  const min = Math.min(...values, ...refs);
  const max = Math.max(...values, ...refs);
  const pad = Math.max((max - min) * 0.08, Math.abs(max) * 0.01) || 1;
  const yMin = min - pad;
  const yMax = max + pad;
  const xFor = (idx) => (idx / Math.max(clean.length - 1, 1)) * width;
  const yFor = (value) => height - ((value - yMin) / Math.max(yMax - yMin, 1e-9)) * height;
  const path = clean.map((p, idx) => `${idx === 0 ? "M" : "L"} ${xFor(idx).toFixed(2)} ${yFor(Number(p.value)).toFixed(2)}`).join(" ");
  const pts = clean.map((p, idx) => ({ x: Number(xFor(idx).toFixed(2)), y: Number(yFor(Number(p.value)).toFixed(2)), v: Number(p.value), label: p.label || "" }));
  const refLines = lines
    .filter((l) => toNumber(l.value) !== null)
    .map((l) => `<line class="ref-line ${l.cssClass}" x1="0" y1="${yFor(Number(l.value)).toFixed(2)}" x2="${width}" y2="${yFor(Number(l.value)).toFixed(2)}"></line>`)
    .join("");
  const refLabels = lines
    .filter((l) => toNumber(l.value) !== null)
    .map((l) => `<span class="ref-label ${l.cssClass}" style="top:${(yFor(Number(l.value)) / height * 100).toFixed(2)}%">${escapeHtml(l.label)}</span>`)
    .join("");
  const dataAttr = escapeHtml(JSON.stringify({ pts, currency, w: width, h: height }));
  return `
    <div class="chart-wrap">
      <svg class="price-chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="${escapeHtml(ariaLabel)}" data-chart="${dataAttr}">
        <g class="grid-lines">
          <line x1="0" y1="${height * 0.25}" x2="${width}" y2="${height * 0.25}"></line>
          <line x1="0" y1="${height * 0.5}" x2="${width}" y2="${height * 0.5}"></line>
          <line x1="0" y1="${height * 0.75}" x2="${width}" y2="${height * 0.75}"></line>
        </g>
        ${refLines}
        <path class="price-path" d="${path}"></path>
        <g class="chart-cursor" opacity="0">
          <line class="cursor-line" x1="0" y1="0" x2="0" y2="${height}"></line>
          <circle class="cursor-dot" r="4" cx="0" cy="0"></circle>
        </g>
      </svg>
      ${refLabels}
      <div class="chart-tooltip" hidden></div>
    </div>
  `;
}

// Daily close history (engine price_series) with target/stop reference lines.
function chartSvg(item) {
  const series = (item?.price_series || []).filter((row) => toNumber(row.price) !== null).slice(-90);
  const points = series.map((row) => ({ value: Number(row.price), label: compactDate(row.date) }));
  return lineChartMarkup(points, {
    currency: item.currency,
    lines: [
      { value: toNumber(item.risk?.target_price), cssClass: "target-line", label: "목표" },
      { value: toNumber(item.risk?.stop_price), cssClass: "stop-line", label: "손절" },
    ],
    ariaLabel: `${item.symbol} 일일 가격 차트`,
    emptyText: "가격 데이터 없음",
  });
}

// Intraday session history accumulated from the 30s live-quote polling.
function liveChartSvg(item) {
  const history = state.liveHistory?.[item.symbol] || [];
  const points = history.map((h) => ({ value: Number(h.price), label: fmtClock(h.ts) }));
  const currency = history.length ? history[history.length - 1].currency : (liveQuote(item.symbol)?.currency || item.currency);
  return lineChartMarkup(points, {
    currency,
    ariaLabel: `${item.symbol} 실시간 세션 차트`,
    emptyText: "실시간 데이터 수집 중 · 30초마다 갱신 (점 2개부터 표시)",
  });
}

// Wire hover crosshair + price tooltip onto every chart inside `root`.
function attachChartInteractions(root) {
  if (!root) return;
  root.querySelectorAll(".chart-wrap").forEach((wrap) => {
    const svg = wrap.querySelector(".price-chart[data-chart]");
    const tip = wrap.querySelector(".chart-tooltip");
    if (!svg || !tip) return;
    let cfg;
    try {
      cfg = JSON.parse(svg.dataset.chart);
    } catch (err) {
      return;
    }
    const pts = cfg.pts || [];
    if (pts.length < 2) return;
    const cursor = svg.querySelector(".chart-cursor");
    const line = svg.querySelector(".cursor-line");
    const dot = svg.querySelector(".cursor-dot");
    const onMove = (event) => {
      const rect = svg.getBoundingClientRect();
      if (!rect.width) return;
      const vbX = ((event.clientX - rect.left) / rect.width) * cfg.w;
      let best = pts[0];
      let bestDist = Infinity;
      for (const p of pts) {
        const dist = Math.abs(p.x - vbX);
        if (dist < bestDist) {
          bestDist = dist;
          best = p;
        }
      }
      cursor.setAttribute("opacity", "1");
      line.setAttribute("x1", best.x);
      line.setAttribute("x2", best.x);
      dot.setAttribute("cx", best.x);
      dot.setAttribute("cy", best.y);
      tip.hidden = false;
      tip.innerHTML = `<b>${fmtPrice(best.v, cfg.currency)}</b>${best.label ? `<span>${escapeHtml(best.label)}</span>` : ""}`;
      tip.style.left = `${(best.x / cfg.w) * rect.width}px`;
      tip.style.top = `${(best.y / cfg.h) * rect.height}px`;
    };
    const onLeave = () => {
      cursor.setAttribute("opacity", "0");
      tip.hidden = true;
    };
    svg.addEventListener("pointermove", onMove);
    svg.addEventListener("pointerleave", onLeave);
  });
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

function liveRiskSection(selected, q) {
  if (!q || toNumber(q.price) === null) return "";
  const cur = selected.currency;
  const live = toNumber(q.price);
  const stop = toNumber(selected.risk?.stop_price);
  const target = toNumber(selected.risk?.target_price);
  const fc20 = toNumber(selected.close_forecasts?.["20d"]?.predicted_close ?? selected.predictions?.["20d"]?.predicted_close);
  const stopCushion = stop ? (live / stop - 1) * 100 : null; // + = above stop (safe)
  const targetGap = target ? (target / live - 1) * 100 : null; // + = room left to target
  const upside = fc20 ? (fc20 / live - 1) * 100 : null; // + = model expects higher than now
  const alerts = [];
  if (stop && live <= stop) alerts.push(`<div class="live-alert danger">손절가 이탈 · 현재가 ${fmtPrice(live, cur)} ≤ 손절 ${fmtPrice(stop, cur)}</div>`);
  if (target && live >= target) alerts.push(`<div class="live-alert good">목표가 도달 · 현재가 ${fmtPrice(live, cur)} ≥ 목표 ${fmtPrice(target, cur)}</div>`);
  const card = (label, value, note, tone) =>
    `<div class="live-card"><span>${label}</span><strong>${value}</strong>${note ? `<small class="${tone || ""}">${escapeHtml(note)}</small>` : ""}</div>`;
  return `
    <div class="focus-section live-section">
      <h3 class="focus-section-title">실시간 가격 · 리스크 <small class="live-tag">● LIVE</small></h3>
      ${alerts.join("")}
      <div class="live-grid">
        ${card("현재가", fmtPrice(live, cur), `${fmtSignedPct(q.change_pct)} 장중`, moveTone(q.change_pct))}
        ${card("손절가", fmtPrice(stop, cur), stopCushion === null ? "" : `${fmtSignedPct(stopCushion)} 여유`, stopCushion !== null && stopCushion <= 0 ? "down" : "up")}
        ${card("목표가", fmtPrice(target, cur), targetGap === null ? "" : (targetGap <= 0 ? "도달" : `${fmtSignedPct(targetGap)} 남음`), targetGap !== null && targetGap <= 0 ? "up" : "")}
        ${card("20일 예측종가", fmtPrice(fc20, cur), upside === null ? "" : `현재가 대비 ${fmtSignedPct(upside)}`, moveTone(upside))}
      </div>
    </div>`;
}

function renderDetail() {
  const all = state.data?.symbols || [];
  const filtered = filteredSymbols();
  const pool = state.signalFilter === "all" ? all : filtered;
  const selected = pool.find((item) => item.symbol === state.selectedSymbol) || pool[0];
  if (!selected) {
    $("detailPanel").innerHTML = `<div class="empty-state">선택한 분류에 표시할 종목이 없습니다.</div>`;
    return;
  }
  state.selectedSymbol = selected.symbol;
  const q = liveQuote(selected.symbol);
  const cur = selected.currency;
  const headPrice = q
    ? `
      <div class="focus-price live">
        <div class="focus-live-row">
          <strong>${fmtPrice(q.price, q.currency || cur)}</strong>
          <span class="move ${moveTone(q.change_pct)}">${fmtSignedPct(q.change_pct)}</span>
          <em class="market-badge state-${escapeHtml(q.market_state || "UNKNOWN")}">${isLiveMarket(q) ? "● " : ""}${escapeHtml(marketStateLabel[q.market_state] || q.market_state || "")}</em>
        </div>
        <span class="focus-close-note">종가 ${fmtPrice(selected.price, cur)} · ${escapeHtml(compactDate(selected.as_of))} 기준</span>
      </div>`
    : `
      <div class="focus-price">
        <strong>${fmtPrice(selected.price, cur)}</strong>
        <span>${escapeHtml(cur)} · ${escapeHtml(compactDate(selected.as_of))}</span>
      </div>`;
  $("detailPanel").innerHTML = `
    <div class="focus-head">
      <div class="focus-id">
        <span>${escapeHtml(selected.symbol)}</span>
        <h2>${escapeHtml(selected.name)}</h2>
      </div>
      ${actionPill(selected.action_level)}
      ${headPrice}
    </div>
    ${liveRiskSection(selected, q)}
    <div class="focus-section">
      <h3 class="focus-section-title">예측 <small class="muted-note">기준일 종가 기준 · 실시간 아님</small></h3>
      <div class="forecast-grid">
        ${combinedForecastBlock(selected, "1d")}
        ${combinedForecastBlock(selected, "5d")}
        ${combinedForecastBlock(selected, "20d")}
      </div>
    </div>
    <div class="focus-section">
      <div class="chart-grid">
        <div class="chart-col">
          <h3 class="focus-section-title">가격 추이 <small class="muted-note">일일 종가 · 목표/손절</small></h3>
          ${chartSvg(selected)}
        </div>
        <div class="chart-col">
          <h3 class="focus-section-title">실시간 추이 <small class="live-tag">● LIVE · 30초</small></h3>
          ${liveChartSvg(selected)}
        </div>
      </div>
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
  attachChartInteractions($("detailPanel"));
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

function renderLiveBar() {
  const bar = $("liveBar");
  if (!bar) return;
  const live = state.live;
  if (!live || !live.quotes || !Object.keys(live.quotes).length) {
    bar.classList.add("hidden");
    return;
  }
  bar.classList.remove("hidden");
  const us = marketStateLabel[live.us_market_state] || live.us_market_state || "-";
  const kr = marketStateLabel[live.kr_market_state] || live.kr_market_state || "-";
  const usLive = ["REGULAR", "PRE", "POST"].includes(live.us_market_state);
  const krLive = live.kr_market_state === "REGULAR";
  const t = live.generated_at ? String(live.generated_at).slice(11, 19) : "";
  bar.innerHTML = `
    <span class="live-pulse" aria-hidden="true"></span>
    <strong>실시간 시세</strong>
    <span class="live-market ${usLive ? "on" : ""}">미국 ${escapeHtml(us)}</span>
    <span class="live-market ${krLive ? "on" : ""}">한국 ${escapeHtml(kr)}</span>
    <span class="live-updated">갱신 ${escapeHtml(t)}</span>
  `;
}

function render() {
  if (!state.data) return;
  // Prefer the live polled FX over the daily reference rate when available.
  setFx(state.live?.fx ? { ...state.live.fx, date: state.live.generated_at } : state.data.fx || {});
  setSystem(state.data.system || {});
  renderLiveBar();
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
  const liveTag = p.live_as_of
    ? `<span class="pf-live-tag">● 실시간 ${escapeHtml(String(p.live_as_of).slice(11, 19))}</span>`
    : "";
  $("pfHero").innerHTML = `
    <div class="pf-total">
      <span>내 투자 · 평가금 ${liveTag}</span>
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
      const intr = toNumber(h.intraday_pct);
      return `<tr>
        <td class="pf-sym"><b>${escapeHtml(pfName(h.symbol))}</b><i>${escapeHtml(h.symbol)}</i></td>
        <td class="num">${pfShares(h.shares)}</td>
        <td class="num">${pfMoney(h.avg_price_usd)}</td>
        <td class="num">${pfMoney(h.current_price_usd)}</td>
        <td class="num ${intr === null ? "" : moveTone(intr)}">${intr === null ? "–" : fmtSignedPct(intr)}</td>
        <td class="num">${pfMoney(h.market_value_usd)}</td>
        <td class="num">${fmtPctPoints(h.weight_pct)}</td>
        <td class="num pnl ${pfPnlTone(h.return_pct)}">${h.return_pct === null ? "없음" : fmtPctPoints(h.return_pct)}</td>
        <td><span class="pf-pill ${tone}">${escapeHtml(label)}</span></td>
      </tr>`;
    })
    .join("");
  target.innerHTML = `<table class="pf-table"><thead><tr>
      <th>종목</th><th class="num">보유 주수</th><th class="num">평단</th><th class="num">현재가</th><th class="num">장중</th>
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

async function pollLiveQuotes() {
  try {
    const live = await fetchJson("/api/live-quotes");
    if (live && live.quotes) {
      state.live = live;
      // Accumulate a live intraday price point per symbol on every poll, so the
      // session chart fills in for as long as the dashboard stays open.
      recordLiveHistory(live.quotes);
      // Live FX: update the rate display every poll (no manual button needed).
      if (live.fx && toNumber(live.fx.usdkrw) !== null) {
        setFx({ ...live.fx, date: live.generated_at });
      }
      // Re-render only the price-bearing views; never re-fetch the heavy payload.
      if (state.data) {
        renderLiveBar();
        if (state.activeTab === "portfolio") {
          // Re-price the portfolio from live quotes (engine re-run is cheap).
          if (state.portfolioLoaded) loadPortfolio();
        } else if (state.activeTab !== "system" && state.activeTab !== "run") {
          renderSymbolGrid();
          renderDetail();
        }
      }
    }
  } catch (error) {
    /* transient network error — keep last known quotes */
  }
}

function startLiveQuotePolling() {
  if (state.liveTimer) clearInterval(state.liveTimer);
  pollLiveQuotes();
  state.liveTimer = setInterval(pollLiveQuotes, LIVE_POLL_MS);
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
  $("fxCalcResult").textContent = "계산 중";
  try {
    // Prefer the live polled rate; fall back to an on-demand fetch only if the
    // poll has not landed yet. Never trigger the heavy dashboard rebuild here.
    let usdkrw = toNumber(state.live?.fx?.usdkrw);
    if (usdkrw === null) {
      const fx = await fetchLiveFx();
      usdkrw = fx.usdkrw ?? fx.fx_close;
    }
    $("fxCalcResult").textContent = fmtExchangeResult($("fxAmount").value, $("fxDirection").value, usdkrw);
  } catch (error) {
    $("fxCalcResult").textContent = "실패";
    $("fxCalcResult").title = error.message;
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
  $("fxCalcBtn").addEventListener("click", calculateFx);
  $("runForm").addEventListener("submit", startRun);
  $("stopRunBtn").addEventListener("click", stopRun);
  wirePortfolioEvents();
  loadDashboard().catch((error) => {
    $("actionRow").innerHTML = `<article class="action-card blocked"><span>오류</span><h2>대시보드 로드 실패</h2><p>${escapeHtml(error.message)}</p></article>`;
  });
  startLiveQuotePolling();
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") pollLiveQuotes();
  });
}

document.addEventListener("DOMContentLoaded", init);
