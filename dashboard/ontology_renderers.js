(function () {
  "use strict";

  function renderOntologyDecisionHero(id, ontology) {
    const container = byId(id);
    if (!container || !ontology) return;
    const metrics = ontology.metrics || {};
    const permission = ontology.permission || {};
    const action = ontology.actionPlan || {};
    const blocker = action.mainBlocker;
    const driver = action.mainDriver;
    const tone = permission.tone || toneFromScore(metrics.todayActionabilityScore);
    container.innerHTML = `
      <section class="ontology-command-hero ${escapeAttr(tone)}">
        <div class="ontology-hero-copy">
          <span>${escapeHtml(ontology.symbol ? `${ontology.symbol} Object${ontology.groupLabel ? ` · ${ontology.groupLabel}` : ""}` : "Top10 Ontology Command Center")} · ${escapeHtml(ontology.asOf || "없음")}</span>
          <strong>${escapeHtml(permission.title || "상태 확인")}</strong>
          <p>${escapeHtml(heroSentence(ontology))}</p>
        </div>
        <div class="ontology-hero-score" style="--score:${num(metrics.todayActionabilityScore)}">
          <span>Actionability</span>
          <strong>${escapeHtml(fmtNumber(metrics.todayActionabilityScore, 0))}</strong>
          <p>오늘 행동 가능성</p>
        </div>
        <div class="ontology-hero-next">
          <span>Next Action</span>
          <strong>${escapeHtml(action.next || "확인 필요")}</strong>
          <p>${escapeHtml(action.reason || "다음 업데이트에서 다시 합성합니다.")}</p>
        </div>
      </section>
      <section class="ontology-pressure-grid">
        ${pressureCard("Opportunity", metrics.opportunityPressure, "기회 압력", "가격·트리거·예측·뉴스 명확성", toneFromScore(metrics.opportunityPressure)).outerHTML}
        ${pressureCard("Block", metrics.blockPressure, "차단 압력", "모델 게이트·리스크·품질·스트레스", metrics.blockPressure >= 65 ? "bad" : metrics.blockPressure >= 45 ? "warn" : "good").outerHTML}
        ${pressureCard("Confidence", confidenceScore(permission.confidence), permission.confidence || "Low", "데이터·검증·계보 신뢰도", confidenceTone(permission.confidence)).outerHTML}
        <article class="ontology-pressure-card neutral">
          <span>Main Driver</span>
          <strong>${escapeHtml(driver?.label || "없음")}</strong>
          <p>${escapeHtml(driver ? `${fmtNumber(driver.score, 0)} / 100` : "합성 동인 없음")}</p>
        </article>
        <article class="ontology-pressure-card ${escapeAttr(blocker ? "warn" : "good")}">
          <span>Main Blocker</span>
          <strong>${escapeHtml(blocker?.title || "차단 없음")}</strong>
          <p>${escapeHtml(blocker?.source || "현재 우선 차단 객체 없음")}</p>
        </article>
      </section>
      <section class="ontology-permission-strip">
        ${permissionChip("Live", permission.live || "LIVE_DISABLED_BY_DESIGN", "neutral")}
        ${permissionChip("Strict", permission.strict || "STRICT_NO_ENTRY", permission.state?.includes("STRICT") ? "good" : "warn")}
        ${permissionChip("Paper", permission.paper || "NOT_READY", permission.paper?.includes("ALLOWED") ? "good" : "warn")}
        ${permissionChip("Prediction", permission.prediction || "DISPLAY_ONLY", permission.prediction?.includes("ALLOWED") ? "good" : "warn")}
        ${permissionChip("Risk", permission.risk || "UNKNOWN", permission.risk?.includes("NO_NEW") ? "warn" : "neutral")}
      </section>
    `;
  }

  function renderOntologyObjectCards(id, ontology) {
    const container = byId(id);
    if (!container || !ontology) return;
    container.innerHTML = (ontology.objects || [])
      .map((objectItem) => `
        <article class="ontology-object-card ${escapeAttr(objectItem.tone || "neutral")}" data-object-card="${escapeAttr(objectItem.id)}">
          <header>
            <span>${escapeHtml(objectItem.type)}</span>
            <strong>${escapeHtml(objectItem.label)}</strong>
            <em>${escapeHtml(objectItem.status || "상태 없음")}</em>
          </header>
          <p>${escapeHtml(objectItem.meaning || "")}</p>
          <dl>
            ${(objectItem.fields || []).slice(0, 7).map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("")}
          </dl>
          <footer>
            <span>Actionability ${impactText(objectItem.impacts?.actionability)}</span>
            <span>Block ${impactText(objectItem.impacts?.blockPressure)}</span>
          </footer>
        </article>
      `)
      .join("");
  }

  function renderOntologyObjectGraph(id, ontology) {
    const container = byId(id);
    if (!container || !ontology) return;
    const graph = ontology.graph || { nodes: [], edges: [] };
    const nodeMap = new Map((graph.nodes || []).map((node) => [node.id, node]));
    const lines = (graph.edges || [])
      .map((edge) => {
        const from = nodeMap.get(edge.from);
        const to = nodeMap.get(edge.to);
        if (!from || !to) return "";
        const mx = (from.x + to.x) / 2;
        const my = (from.y + to.y) / 2;
        return `
          <line class="ontology-graph-edge ${escapeAttr(edge.tone || "neutral")}" x1="${from.x}" y1="${from.y}" x2="${to.x}" y2="${to.y}" />
          <text x="${mx}" y="${my - 1.6}" class="ontology-graph-label">${escapeHtml(edge.relation)}</text>
        `;
      })
      .join("");
    const firstNode = graph.nodes?.[0] || {};
    container.innerHTML = `
      <div class="ontology-graph-shell">
        <div class="ontology-graph-canvas" aria-label="Ontology object graph">
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">${lines}</svg>
          ${(graph.nodes || []).map(graphNodeHtml).join("")}
        </div>
        <aside class="ontology-graph-drawer" id="ontologyGraphDrawer">
          ${drawerHtml(firstNode, graph, ontology)}
        </aside>
      </div>
    `;
    container.querySelectorAll("[data-ontology-node]").forEach((button) => {
      button.addEventListener("click", () => {
        const node = nodeMap.get(button.dataset.ontologyNode);
        const drawer = byId("ontologyGraphDrawer");
        if (!node || !drawer) return;
        container.querySelectorAll("[data-ontology-node]").forEach((item) => item.classList.toggle("active", item === button));
        drawer.innerHTML = drawerHtml(node, graph, ontology);
      });
    });
    const firstButton = container.querySelector("[data-ontology-node]");
    if (firstButton) firstButton.classList.add("active");
  }

  function renderOntologyInsightPanel(id, ontology) {
    const container = byId(id);
    if (!container || !ontology) return;
    container.innerHTML = (ontology.insights || [])
      .map((item, idx) => `
        <article class="ontology-insight-item ${escapeAttr(item.tone || "neutral")}">
          <b>${idx + 1}</b>
          <div>
            <strong>${escapeHtml(item.title)}</strong>
            <p>${escapeHtml(item.body)}</p>
            <span>${escapeHtml(item.action || "")}</span>
          </div>
        </article>
      `)
      .join("") || `<div class="preview-empty">생성된 insight 없음</div>`;
  }

  function renderOntologyRootCausePanel(id, ontology) {
    const container = byId(id);
    if (!container || !ontology) return;
    const blockers = ontology.blockers || [];
    const contradictions = ontology.contradictions || [];
    container.innerHTML = `
      <div class="ontology-root-list">
        ${blockers.slice(0, 5).map((blocker, idx) => `
          <article class="ontology-root-item ${escapeAttr(blocker.severity === "CRITICAL" ? "bad" : "warn")}">
            <b>P${idx + 1}</b>
            <div>
              <span>${escapeHtml(blocker.source || "Blocker")}</span>
              <strong>${escapeHtml(blocker.title || blocker.code)}</strong>
              <p>${escapeHtml(blocker.detail || "세부 기준 확인 필요")}</p>
              <em>${escapeHtml((blocker.affectedObjects || []).join(" → "))}</em>
            </div>
          </article>
        `).join("") || `<div class="preview-empty">차단 원인 없음</div>`}
      </div>
      <div class="ontology-contradiction-list">
        <h3>Contradiction Detector</h3>
        ${contradictions.slice(0, 5).map((item) => `
          <article class="ontology-contradiction ${escapeAttr(item.tone || "warn")}">
            <strong>${escapeHtml(item.title)}</strong>
            <p>${escapeHtml(item.meaning)}</p>
            <span>${escapeHtml(item.action)}</span>
          </article>
        `).join("") || `<p class="ontology-muted">오늘 감지된 핵심 모순 없음</p>`}
      </div>
    `;
  }

  function renderOntologyExecutionLadder(id, ontology) {
    const container = byId(id);
    if (!container || !ontology) return;
    container.innerHTML = (ontology.executionLadder || [])
      .map((step, idx) => `
        <article class="ontology-ladder-step ${escapeAttr(step.tone || "neutral")}">
          <em>${idx + 1}</em>
          <span>${escapeHtml(step.label)}</span>
          <strong>${escapeHtml(step.status)}</strong>
          <p>${escapeHtml(step.value)}</p>
          <small>${escapeHtml(step.note || "")}</small>
        </article>
      `)
      .join("");
  }

  function renderOntologyTradingPlanCompact(id, ontology) {
    const container = byId(id);
    if (!container || !ontology) return;
    const action = ontology.actionPlan || {};
    const permission = ontology.permission || {};
    const rows = [
      ["현재 결론", permission.title || "상태 확인"],
      ["20D 돌파 기준", action.targetPrice20d || "없음"],
      ["60D 돌파 기준", action.targetPrice60d || "없음"],
      ["2ATR 손절가", action.stopPriceDisplay || fmtCurrency(action.stopPrice)],
      ["최대 비중", fmtPct(action.maxWeight, 2)],
      ["Paper Action", permission.paper || "NOT_READY"],
      ["Live Action", permission.live || "LIVE_DISABLED_BY_DESIGN"],
      ["금지/제약 사유", action.mainBlocker?.title || "없음"],
    ];
    container.innerHTML = `
      <article class="ontology-plan-card ${escapeAttr(permission.tone || "neutral")}">
        <span>오늘 매매 계획</span>
        <strong>${escapeHtml(action.next || "확인 필요")}</strong>
        <p>${escapeHtml(action.reason || "")}</p>
      </article>
      <dl class="ontology-plan-facts">
        ${rows.map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(formatStatus(value))}</dd></div>`).join("")}
      </dl>
    `;
  }

  function renderOntologyPipelineLineage(id, ontology) {
    const container = byId(id);
    if (!container || !ontology) return;
    const lineage = ontology.lineage || {};
    const rows = [
      ["Last Run", lineage.lastRunStatus || "IDLE"],
      ["Started", shortDateTime(lineage.startedAt)],
      ["Ended", shortDateTime(lineage.endedAt)],
      ["Duration", lineage.durationSec ? `${fmtNumber(lineage.durationSec, 0)}초` : "없음"],
      ["Failed Steps", fmtNumber(lineage.failedSteps, 0)],
      ["Step Pass", `${fmtNumber(lineage.passSteps, 0)} / ${fmtNumber(lineage.totalSteps, 0)}`],
      ["Latest Signal", lineage.latestSignalDate || "없음"],
      ["Data Age", lineage.dataAgeDays === null || lineage.dataAgeDays === undefined ? "없음" : `${lineage.dataAgeDays}일`],
      ["Required Files", lineage.requiredFilesStatus || "없음"],
      ["News", lineage.newsCollection || "없음"],
      ["Reports", fmtNumber(lineage.reports, 0)],
      ["Images", fmtNumber(lineage.images, 0)],
    ];
    container.innerHTML = `
      <article class="ontology-lineage-status ${escapeAttr(lineage.lastRunStatus === "PASS" ? "good" : lineage.lastRunStatus === "FAIL" ? "bad" : "neutral")}">
        <span>판단 계보</span>
        <strong>${escapeHtml(lineage.lastRunStatus || "IDLE")}</strong>
        <p>Freshness ${escapeHtml(fmtNumber(lineage.pipelineFreshnessScore, 0))} / 100</p>
      </article>
      <dl class="ontology-plan-facts">
        ${rows.map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(formatStatus(value))}</dd></div>`).join("")}
      </dl>
      ${lineage.failedStepNames?.length ? `<p class="ontology-muted">실패 단계: ${escapeHtml(lineage.failedStepNames.join(", "))}</p>` : ""}
    `;
  }

  function graphNodeHtml(node) {
    return `
      <button class="ontology-graph-node ${escapeAttr(node.tone || "neutral")}" type="button" data-ontology-node="${escapeAttr(node.id)}" style="left:${node.x}%; top:${node.y}%;">
        <span>${escapeHtml(node.type)}</span>
        <strong>${escapeHtml(node.label)}</strong>
        <p>${escapeHtml(node.status || "")}</p>
      </button>
    `;
  }

  function drawerHtml(node, graph, ontology) {
    const incoming = (graph.edges || []).filter((edge) => edge.to === node.id);
    const outgoing = (graph.edges || []).filter((edge) => edge.from === node.id);
    return `
      <div class="ontology-drawer-head">
        <span>${escapeHtml(node.type || "Object")}</span>
        <strong>${escapeHtml(node.label || "객체")}</strong>
        <p>${escapeHtml(node.meaning || "")}</p>
      </div>
      <dl class="ontology-drawer-fields">
        ${(node.fields || []).map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(formatStatus(value))}</dd></div>`).join("")}
      </dl>
      <div class="ontology-drawer-impact">
        <article><span>Actionability</span><strong>${escapeHtml(impactText(node.impacts?.actionability))}</strong></article>
        <article><span>Block Pressure</span><strong>${escapeHtml(impactText(node.impacts?.blockPressure))}</strong></article>
      </div>
      <div class="ontology-drawer-relations">
        <h3>연결</h3>
        ${[...incoming, ...outgoing].map((edge) => relationHtml(edge, node.id)).join("") || `<p class="ontology-muted">연결 없음</p>`}
      </div>
      <p class="ontology-muted">Source: ${escapeHtml(node.source?.file || "derived ontology")}</p>
      <p class="ontology-muted">Ontology conflicts: ${escapeHtml(fmtNumber(ontology.metrics?.ontologyConflictCount, 0))}</p>
    `;
  }

  function relationHtml(edge, currentId) {
    const direction = edge.from === currentId ? "out" : "in";
    const other = direction === "out" ? edge.to : edge.from;
    return `
      <article class="ontology-drawer-relation ${escapeAttr(edge.tone || "neutral")}">
        <span>${direction === "out" ? "to" : "from"} ${escapeHtml(other)}</span>
        <strong>${escapeHtml(edge.relation)}</strong>
        <p>${escapeHtml(edge.label || "")}</p>
      </article>
    `;
  }

  function pressureCard(label, value, title, note, tone) {
    const article = document.createElement("article");
    article.className = `ontology-pressure-card ${tone || "neutral"}`;
    article.style.setProperty("--score", num(value));
    article.innerHTML = `
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(fmtNumber(value, 0))}</strong>
      <p><b>${escapeHtml(title)}</b> · ${escapeHtml(note)}</p>
      <div><i></i></div>
    `;
    return article;
  }

  function permissionChip(label, value, tone) {
    return `
      <article class="ontology-permission-chip ${escapeAttr(tone || "neutral")}">
        <span>${escapeHtml(label)}</span>
        <strong>${escapeHtml(formatStatus(value))}</strong>
      </article>
    `;
  }

  function heroSentence(ontology) {
    const permission = ontology.permission || {};
    const metrics = ontology.metrics || {};
    const blocker = ontology.actionPlan?.mainBlocker;
    const driver = ontology.actionPlan?.mainDriver;
    return `${permission.title || "상태 확인"}입니다. 기회 압력 ${fmtNumber(metrics.opportunityPressure, 0)}, 차단 압력 ${fmtNumber(metrics.blockPressure, 0)}이며, 가장 강한 근거는 ${driver?.label || "없음"}이고 가장 큰 제약은 ${blocker?.title || "없음"}입니다.`;
  }

  function formatStatus(value) {
    const raw = String(value ?? "");
    const map = {
      LIVE_DISABLED_BY_DESIGN: "설계상 비활성",
      NO_LIVE_BROKER_BY_DESIGN: "실거래 브로커 미사용 설계",
      DISPLAY_ONLY_NO_LATEST_TRADE_READY: "최신 trade-ready 아님",
      DISPLAY_ONLY_MODEL_BLOCKED: "모델 기준 미달",
      DECISION_SUPPORT_ALLOWED: "판단 참고 가능",
      PAPER_TRACKING_ALLOWED: "Paper 추적 가능",
      STRICT_NO_ENTRY: "엄격 기준 신규 진입 없음",
      NO_NEW_RISK: "신규 위험 없음",
      NO_TRADE: "거래 없음",
      NOT_READY: "준비 안 됨",
      READY: "준비됨",
      PASS: "문제 없음",
      FAIL: "문제 있음",
      WARN: "주의 필요",
      BLOCKED: "차단",
      UNKNOWN: "알 수 없음",
    };
    return map[raw] || map[raw.toUpperCase?.()] || raw.replaceAll("_", " ");
  }

  function confidenceScore(value) {
    const raw = String(value || "").toUpperCase();
    if (raw === "HIGH") return 86;
    if (raw === "MEDIUM") return 62;
    return 34;
  }

  function confidenceTone(value) {
    const raw = String(value || "").toUpperCase();
    if (raw === "HIGH") return "good";
    if (raw === "MEDIUM") return "warn";
    return "bad";
  }

  function toneFromScore(value) {
    const parsed = num(value);
    if (parsed >= 72) return "good";
    if (parsed < 45) return "bad";
    return "warn";
  }

  function impactText(value) {
    const parsed = Number(value || 0);
    if (!Number.isFinite(parsed) || parsed === 0) return "0";
    return `${parsed > 0 ? "+" : ""}${Math.round(parsed)}`;
  }

  function byId(id) {
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

  function escapeAttr(value) {
    return escapeHtml(String(value || "").replace(/[^a-zA-Z0-9_-]/g, ""));
  }

  function num(value) {
    const parsed = Number(String(value ?? "").replace(/[$,%\s,]/g, ""));
    return Number.isFinite(parsed) ? parsed : 0;
  }

  function fmtNumber(value, digits = 1) {
    const parsed = num(value);
    return parsed.toLocaleString("ko-KR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
  }

  function fmtPct(value, digits = 1) {
    if (value === null || value === undefined || value === "") return "없음";
    return `${fmtNumber(value, digits)}%`;
  }

  function fmtCurrency(value) {
    if (value === null || value === undefined || value === "") return "없음";
    const parsed = num(value);
    return `$${parsed.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }

  function shortDateTime(value) {
    if (!value) return "없음";
    return String(value).replace("T", " ").slice(0, 19);
  }

  window.renderOntologyDecisionHero = renderOntologyDecisionHero;
  window.renderOntologyObjectCards = renderOntologyObjectCards;
  window.renderOntologyObjectGraph = renderOntologyObjectGraph;
  window.renderOntologyInsightPanel = renderOntologyInsightPanel;
  window.renderOntologyRootCausePanel = renderOntologyRootCausePanel;
  window.renderOntologyExecutionLadder = renderOntologyExecutionLadder;
  window.renderOntologyTradingPlanCompact = renderOntologyTradingPlanCompact;
  window.renderOntologyPipelineLineage = renderOntologyPipelineLineage;
})();
