const ui = {
  endpoint: document.getElementById('endpoint-input'),
  labMode: document.getElementById('lab-mode-select'),
  prompt: document.getElementById('prompt-input'),
  seed: document.getElementById('seed-input'),
  temperature: document.getElementById('temperature-input'),
  topP: document.getElementById('top-p-input'),
  maxTokens: document.getElementById('max-tokens-input'),
  vectorMode: document.getElementById('vector-mode-select'),
  riskThreshold: document.getElementById('risk-threshold-input'),
  riskPersistence: document.getElementById('risk-persistence-input'),
  rollbackBuffer: document.getElementById('rollback-buffer-input'),
  branchCandidates: document.getElementById('branch-candidates-input'),
  branchLookahead: document.getElementById('branch-lookahead-input'),
  maxInterventions: document.getElementById('max-interventions-input'),
  runBtn: document.getElementById('run-btn'),
  stopBtn: document.getElementById('stop-btn'),
  probeBackendBtn: document.getElementById('probe-backend-btn'),
  status: document.getElementById('status-badge'),

  experimentMeta: document.getElementById('experiment-meta'),
  experimentSummary: document.getElementById('experiment-summary'),
  iterationTraceSummary: document.getElementById('iteration-trace-summary'),
  iterationTrace: document.getElementById('iteration-trace'),
  openMainLink: document.getElementById('open-main-link'),
  modeInfoNote: document.getElementById('mode-info-note'),
  retractionPanel: document.getElementById('retraction-panel'),
  retractionTraceSummary: document.getElementById('retraction-trace-summary'),
  retractionDraft: document.getElementById('retraction-draft'),
  retractionCommitments: document.getElementById('retraction-commitments'),
  retractionProbes: document.getElementById('retraction-probes'),
  retractionScoring: document.getElementById('retraction-scoring'),
  retractionReconciliation: document.getElementById('retraction-reconciliation'),
  retractionFinalAnswer: document.getElementById('retraction-final-answer'),
  retractionVerifier: document.getElementById('retraction-verifier'),
  retractionProvenance: document.getElementById('retraction-provenance'),
  retractionBudget: document.getElementById('retraction-budget'),

  baselineRunId: document.getElementById('baseline-run-id'),
  baselineRisk: document.getElementById('baseline-risk'),
  baselineAlerts: document.getElementById('baseline-alerts'),
  baselineTokens: document.getElementById('baseline-tokens'),
  baselineText: document.getElementById('baseline-text'),

  correctedTitle: document.getElementById('corrected-title'),
  correctedRunId: document.getElementById('corrected-run-id'),
  correctedRisk: document.getElementById('corrected-risk'),
  correctedAlerts: document.getElementById('corrected-alerts'),
  correctedTokens: document.getElementById('corrected-tokens'),
  correctedPassNote: document.getElementById('corrected-pass-note'),
  correctedPassLog: document.getElementById('corrected-pass-log'),
  correctedText: document.getElementById('corrected-text'),
  tokenTooltip: document.getElementById('live-token-tooltip'),

  backendReachable: document.getElementById('backend-reachable'),
  backendModel: document.getElementById('backend-model'),
  backendProbs: document.getElementById('backend-probs'),
  backendEmbeddings: document.getElementById('backend-embeddings'),
  backendModelRequired: document.getElementById('backend-model-required'),
  backendStopSemantics: document.getElementById('backend-stop-semantics'),
  backendTokenForcing: document.getElementById('backend-token-forcing'),
  backendMode: document.getElementById('backend-mode'),

  interventionCount: document.getElementById('intervention-count'),
  interventionList: document.getElementById('intervention-list'),

  modeCurrent: document.getElementById('mode-current'),
  modeFocusObject: document.getElementById('mode-focus-object'),
  baselineObjectRisk: document.getElementById('baseline-object-risk'),
  baselineObjectLabel: document.getElementById('baseline-object-label'),
  correctedObjectRisk: document.getElementById('corrected-object-risk'),
  correctedObjectLabel: document.getElementById('corrected-object-label'),
  objectEvidence: document.getElementById('object-evidence'),

  harnessSummary: document.getElementById('harness-summary'),
  baselineClaimRisk: document.getElementById('baseline-claim-risk'),
  baselineClaimLabel: document.getElementById('baseline-claim-label'),
  baselineClaimText: document.getElementById('baseline-claim-text'),
  baselineClaimEvidence: document.getElementById('baseline-claim-evidence'),
  correctedClaimRisk: document.getElementById('corrected-claim-risk'),
  correctedClaimLabel: document.getElementById('corrected-claim-label'),
  correctedClaimText: document.getElementById('corrected-claim-text'),
  correctedClaimEvidence: document.getElementById('corrected-claim-evidence'),
};

const state = {
  backendInfo: null,
  lastResult: null,
  currentExperimentId: null,
  partialInterventions: [],
  partialRuns: {
    baseline: null,
    corrected: null,
    rewritePreview: null,
  },
  passLog: [],
  iterationTrace: {
    passes: [],
    finalEvidence: null,
  },
  retractionTrace: {
    draft: null,
    commitments: [],
    probePackets: [],
    probeEvents: [],
    reconciliationPasses: [],
    verifierTrace: [],
    provenance: null,
    metrics: null,
    status: null,
  },
  experimentRunning: false,
  loopBudget: 0,
  hoverToken: null,
};

function asNumber(value, fallback = null) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function formatNum(value, digits = 3) {
  if (value == null || !Number.isFinite(value)) return '-';
  return Number(value).toFixed(digits);
}

function formatPercent(value, digits = 0) {
  if (value == null || !Number.isFinite(value)) return '-';
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function capabilityText(value, fallback = 'unknown') {
  const raw = String(value ?? '').trim();
  if (!raw) return fallback;
  return raw.replace(/_/g, ' ');
}

function normalizeBaseUrl(rawValue) {
  let raw = String(rawValue ?? '').trim();
  if (!raw) return '';
  if (!/^https?:\/\//i.test(raw)) raw = `http://${raw}`;
  try {
    const u = new URL(raw);
    const path = u.pathname && u.pathname !== '/' ? u.pathname : '';
    return `${u.protocol}//${u.host}${path}`.replace(/\/+$/, '');
  } catch {
    return '';
  }
}

function setStatus(text, tone = 'neutral') {
  ui.status.textContent = text;
  if (tone === 'error') {
    ui.status.style.color = '#7e130b';
    ui.status.style.background = '#ffe8e6';
    ui.status.style.borderColor = '#efb4ae';
    return;
  }
  if (tone === 'success') {
    ui.status.style.color = '#0e5563';
    ui.status.style.background = '#e3f5f8';
    ui.status.style.borderColor = '#acd9e1';
    return;
  }
  ui.status.style.color = '#415468';
  ui.status.style.background = '#eef4fb';
  ui.status.style.borderColor = '#bfd0de';
}

function setExperimentRunning(running, { stopping = false } = {}) {
  state.experimentRunning = Boolean(running);
  ui.runBtn.disabled = Boolean(running);
  ui.stopBtn.disabled = !running || stopping;
  ui.stopBtn.textContent = stopping ? 'Stopping...' : 'Stop';
}

function currentLabMode() {
  return ui.labMode?.value === 'retraction' ? 'retraction' : 'hybrid';
}

function setLabModeVisibility() {
  const retraction = currentLabMode() === 'retraction';
  document.querySelectorAll('.hybrid-only').forEach((node) => node.classList.toggle('hidden', retraction));
  ui.retractionPanel?.classList.toggle('hidden', !retraction);
  if (ui.openMainLink) ui.openMainLink.classList.toggle('hidden', retraction);
  if (ui.modeInfoNote) {
    ui.modeInfoNote.innerHTML = retraction
      ? 'Internal consistency retraction mode does <strong>not</strong> prove factual truth. It shows whether the model can obey a deterministic behavioral contract built from internal probe consistency.'
      : 'This page does not claim to make the model truthful. In the current implementation, intervention uses <strong>prefix replay</strong>, not exact KV-state rewind. Because llama.cpp streaming mode does not expose token-probability telemetry token-by-token, the lab runs one-token decode steps (`n_predict=1`) to observe risk and decide whether to branch.';
  }
  if (ui.backendMode) {
    ui.backendMode.textContent = retraction ? 'internal consistency retraction' : 'hybrid replay / rewrite';
  }
}

function resetRetractionTrace() {
  state.retractionTrace = {
    draft: null,
    commitments: [],
    probePackets: [],
    probeEvents: [],
    reconciliationPasses: [],
    verifierTrace: [],
    provenance: null,
    metrics: null,
    status: null,
  };
}

function clearNode(node, emptyText = '') {
  if (!node) return;
  node.innerHTML = '';
  if (emptyText) node.textContent = emptyText;
}

function appendTraceCard(parent, { title = '', pill = '', body = '', detail = '', tone = '' } = {}) {
  if (!parent) return null;
  const card = document.createElement('div');
  card.className = `trace-card ${tone || ''}`.trim();
  const head = document.createElement('div');
  head.className = 'trace-card-head';
  const titleNode = document.createElement('span');
  titleNode.className = 'trace-card-title';
  titleNode.textContent = title || '-';
  const pillNode = document.createElement('span');
  pillNode.className = 'trace-card-pill';
  pillNode.textContent = pill || '-';
  head.append(titleNode, pillNode);
  card.appendChild(head);
  if (body) {
    const bodyNode = document.createElement('div');
    bodyNode.className = 'trace-card-body';
    bodyNode.textContent = body;
    card.appendChild(bodyNode);
  }
  if (detail) {
    const detailNode = document.createElement('div');
    detailNode.className = 'trace-card-detail';
    detailNode.textContent = detail;
    card.appendChild(detailNode);
  }
  parent.appendChild(card);
  return card;
}

function shortJson(value) {
  if (value == null) return '';
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function renderRetractionBudget(metrics = state.retractionTrace.metrics) {
  if (!ui.retractionBudget) return;
  ui.retractionBudget.innerHTML = '';
  if (!metrics) {
    ui.retractionBudget.textContent = 'No budget state yet.';
    return;
  }
  const phases = metrics.phase_durations_ms || {};
  appendTraceCard(ui.retractionBudget, {
    title: 'Budget',
    pill: metrics.budget_exhausted ? 'exhausted' : 'active',
    tone: metrics.budget_exhausted ? 'fail' : 'pass',
    body: `model calls ${metrics.model_calls ?? 0} | elapsed ${metrics.duration_ms ?? 0}ms`,
    detail: [
      metrics.budget_exhausted_phase ? `budget_exhausted_phase: ${metrics.budget_exhausted_phase}` : '',
      metrics.stopped_phase ? `stopped_phase: ${metrics.stopped_phase}` : '',
      Object.keys(phases).length ? `phase_durations_ms:\n${shortJson(phases)}` : '',
    ].filter(Boolean).join('\n'),
  });
}

function renderRetractionProvenance(provenance = state.retractionTrace.provenance) {
  if (!ui.retractionProvenance) return;
  ui.retractionProvenance.innerHTML = '';
  if (!provenance) {
    ui.retractionProvenance.textContent = 'No provenance yet.';
    return;
  }
  appendTraceCard(ui.retractionProvenance, {
    title: 'Models and versions',
    pill: provenance.policy_version || 'policy',
    body: `draft=${provenance.draft_model || '-'} | probe=${provenance.probe_model || '-'} | reconciliation=${provenance.reconciliation_model || '-'}`,
    detail: `scorer=${provenance.scorer_version || '-'}\nverifier=${provenance.verifier_version || '-'}\nbackend_hash=${provenance.backend_base_url_hash || '-'}`,
  });
}

function renderRetractionCommitments(commitments = state.retractionTrace.commitments) {
  if (!ui.retractionCommitments) return;
  ui.retractionCommitments.innerHTML = '';
  if (!commitments?.length) {
    ui.retractionCommitments.textContent = 'No commitments yet.';
    return;
  }
  commitments.forEach((item) => {
    const specifics = (item.protected_specifics || []).map((spec) => `${spec.kind}:${spec.text}`).join(', ') || 'none';
    appendTraceCard(ui.retractionCommitments, {
      title: item.id || 'commitment',
      pill: item.origin || '-',
      tone: item.selected_for_probe ? 'warn' : '',
      body: item.text || '',
      detail: [
        `kind: ${item.kind || '-'}`,
        `span: ${item.source_span?.start ?? '-'}..${item.source_span?.end ?? '-'}`,
        `centrality: ${formatPercent(item.centrality, 0)}`,
        `in_scope: ${item.in_retraction_scope ? 'yes' : 'no'}`,
        `selected_for_probe: ${item.selected_for_probe ? 'yes' : 'no'}`,
        `protected_specifics: ${specifics}`,
        `risk_reasons: ${(item.risk_reasons || []).join(', ') || 'none'}`,
      ].join('\n'),
    });
  });
}

function renderRetractionProbePackets(packets = state.retractionTrace.probePackets) {
  if (!ui.retractionProbes || !ui.retractionScoring) return;
  ui.retractionProbes.innerHTML = '';
  ui.retractionScoring.innerHTML = '';
  if (!packets?.length) {
    ui.retractionProbes.textContent = 'No probe packets yet.';
    ui.retractionScoring.textContent = 'No scoring output yet.';
    return;
  }
  packets.forEach((packet) => {
    appendTraceCard(ui.retractionScoring, {
      title: packet.commitment_id || 'packet',
      pill: packet.deterministic_recommendation || '-',
      tone: packet.deterministic_recommendation === 'retract' ? 'fail' : packet.deterministic_recommendation === 'keep_with_caveat' ? 'pass' : 'warn',
      body: `label=${packet.consistency_label || '-'} | agreement=${formatPercent(packet.agreement_score, 0)} | contradiction=${formatPercent(packet.contradiction_score, 0)} | empty=${formatPercent(packet.emptiness_score, 0)}`,
      detail: shortJson(packet.scoring_debug || {}),
    });
    (packet.probe_observations || []).forEach((obs) => {
      appendTraceCard(ui.retractionProbes, {
        title: `${packet.commitment_id || 'packet'} sample ${obs.sample_index ?? '-'}`,
        pill: obs.normalized_label || '-',
        tone: obs.normalized_label === 'agreement' ? 'pass' : obs.normalized_label === 'contradiction' ? 'fail' : 'warn',
        body: obs.normalized_answer || '',
        detail: [
          obs.parse_error ? 'parse_error: yes' : 'parse_error: no',
          obs.raw_answer ? `raw:\n${obs.raw_answer}` : '',
        ].filter(Boolean).join('\n'),
      });
    });
  });
}

function renderRetractionReconciliation(passes = state.retractionTrace.reconciliationPasses) {
  if (!ui.retractionReconciliation) return;
  ui.retractionReconciliation.innerHTML = '';
  if (!passes?.length) {
    ui.retractionReconciliation.textContent = 'No reconciliation pass yet.';
    return;
  }
  passes.forEach((pass) => {
    appendTraceCard(ui.retractionReconciliation, {
      title: `Pass ${pass.pass_index ?? '-'}`,
      pill: pass.parse_error ? 'parse error' : 'parsed',
      tone: pass.parse_error ? 'fail' : 'pass',
      body: `model=${pass.model || '-'}`,
      detail: [
        `allowed_decisions:\n${shortJson(pass.allowed_decisions || {})}`,
        `returned_decisions:\n${shortJson(pass.decisions || [])}`,
      ].join('\n\n'),
    });
  });
}

function renderRetractionVerifier(trace = state.retractionTrace.verifierTrace) {
  if (!ui.retractionVerifier) return;
  ui.retractionVerifier.innerHTML = '';
  if (!trace?.length) {
    ui.retractionVerifier.textContent = 'No verifier result yet.';
    return;
  }
  trace.forEach((item) => {
    appendTraceCard(ui.retractionVerifier, {
      title: item.check || 'check',
      pill: item.passed ? 'pass' : 'fail',
      tone: item.passed ? 'pass' : 'fail',
      body: item.message || '',
      detail: shortJson(Object.fromEntries(Object.entries(item).filter(([key]) => !['check', 'passed', 'message'].includes(key)))),
    });
  });
}

function renderRetractionTraceSummary(result = null) {
  if (!ui.retractionTraceSummary) return;
  const trace = state.retractionTrace;
  const status = result?.status || trace.status || 'running';
  ui.retractionTraceSummary.textContent =
    `${status.replace(/_/g, ' ')} | commitments ${trace.commitments.length} | packets ${trace.probePackets.length} | verifier checks ${trace.verifierTrace.length}`;
}

function renderRetractionTrace() {
  renderRetractionTraceSummary();
  renderRetractionCommitments();
  renderRetractionProbePackets();
  renderRetractionReconciliation();
  renderRetractionVerifier();
  renderRetractionProvenance();
  renderRetractionBudget();
}

async function apiPost(path, payload) {
  const resp = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload ?? {}),
  });
  const text = await resp.text();
  let json = null;
  try {
    json = text ? JSON.parse(text) : {};
  } catch {
    json = null;
  }
  if (!resp.ok) {
    throw new Error(json?.error || text || `HTTP ${resp.status}`);
  }
  return json || {};
}

async function apiGet(path) {
  const resp = await fetch(path);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || `HTTP ${resp.status}`);
  }
  return resp.json();
}

function textFromRun(run) {
  const tokens = Array.isArray(run?.tokens) ? run.tokens : [];
  return tokens.map((token) => token?.text || '').join('') || '-';
}

function tokenRiskValue(token) {
  return asNumber(token?.decoder_risk);
}

function peakRiskIndex(run) {
  const tokens = Array.isArray(run?.tokens) ? run.tokens : [];
  let bestIndex = -1;
  let bestRisk = null;
  for (let i = 0; i < tokens.length; i += 1) {
    const risk = tokenRiskValue(tokens[i]);
    if (risk == null) continue;
    if (bestRisk == null || risk > bestRisk) {
      bestRisk = risk;
      bestIndex = i;
    }
  }
  return bestIndex;
}

function emptyPartialRun(runId = '') {
  return {
    run_id: runId || null,
    tokens: [],
    summary: {
      token_count: 0,
      decoder_alert_count: 0,
      decoder_risk_max: null,
    },
  };
}

function syncPartialRunSummary(run, overrides = {}) {
  if (!run || typeof run !== 'object') return null;
  const tokens = Array.isArray(run.tokens) ? run.tokens.filter(Boolean) : [];
  let maxRisk = null;
  let alertCount = 0;
  tokens.forEach((token) => {
    const risk = tokenRiskValue(token);
    if (risk == null) return;
    if (maxRisk == null || risk > maxRisk) maxRisk = risk;
    if (risk >= 0.64) alertCount += 1;
  });
  run.tokens = tokens;
  run.summary = {
    ...(run.summary || {}),
    token_count: overrides.token_count ?? tokens.length,
    decoder_alert_count: overrides.alert_count ?? alertCount,
    decoder_risk_max: overrides.max_decoder_risk ?? maxRisk,
  };
  return run;
}

function ensurePartialRun(phase, runId) {
  const existing = state.partialRuns[phase];
  if (!existing || (runId && existing.run_id && existing.run_id !== runId)) {
    state.partialRuns[phase] = emptyPartialRun(runId);
  }
  if (runId) state.partialRuns[phase].run_id = runId;
  return state.partialRuns[phase];
}

function applyProgressToken(phase, event) {
  const run = ensurePartialRun(phase, event?.run_id);
  const token = event?.token && typeof event.token === 'object'
    ? { ...event.token }
    : {
        index: asNumber(event?.token_index, run.tokens.length),
        text: event?.token_text || '',
        decoder_risk: asNumber(event?.decoder_risk),
      };
  const tokenIndex = Number.isInteger(token?.index) ? token.index : asNumber(event?.token_index, run.tokens.length);
  const replaceIndex = Number.isInteger(tokenIndex) && tokenIndex >= 0 ? tokenIndex : run.tokens.length;
  if (replaceIndex < run.tokens.length) {
    run.tokens[replaceIndex] = token;
  } else {
    run.tokens.push(token);
  }
  syncPartialRunSummary(run, {
    token_count: event?.token_count,
    alert_count: event?.alert_count,
    max_decoder_risk: event?.max_decoder_risk,
  });
  return run;
}

function applyRunSnapshot(phase, snapshot) {
  if (!snapshot || typeof snapshot !== 'object') return null;
  const run = {
    ...snapshot,
    tokens: Array.isArray(snapshot.tokens) ? snapshot.tokens.filter(Boolean).map((token) => ({ ...token })) : [],
    summary: { ...(snapshot.summary || {}) },
  };
  syncPartialRunSummary(run);
  state.partialRuns[phase] = run;
  return run;
}

function displayTokenText(text) {
  const raw = String(text ?? '');
  if (!raw) return '[empty]';
  return raw
    .replace(/ /g, '␠')
    .replace(/\n/g, '↵')
    .replace(/\t/g, '⇥');
}

function scalarTelemetry(token) {
  return [
    ['Index', token?.index],
    ['Token', displayTokenText(token?.text)],
    ['Token ID', token?.chosen_token_id ?? '-'],
    ['Time', token?.t != null ? `${token.t} ms` : '-'],
    ['Prob', formatPercent(token?.prob, 2)],
    ['Logprob', formatNum(token?.logprob, 3)],
    ['Entropy', formatNum(token?.entropy, 3)],
    ['Margin', formatNum(token?.margin, 3)],
    ['Uncertainty', formatPercent(token?.uncertainty, 1)],
    ['Choice Gap', formatPercent(token?.prob_gap, 1)],
    ['Jump', formatPercent(token?.entropy_delta, 1)],
    ['Repeat', formatPercent(token?.repetition_pressure, 1)],
    ['Risk', formatPercent(token?.decoder_risk, 1)],
    ['Velocity', formatNum(token?.velocity, 3)],
    ['Curvature', formatNum(token?.curvature, 3)],
  ];
}

function ensureTooltipPosition(clientX, clientY) {
  const tooltip = ui.tokenTooltip;
  const pad = 14;
  const rect = tooltip.getBoundingClientRect();
  let left = clientX + 18;
  let top = clientY + 18;
  if (left + rect.width + pad > window.innerWidth) {
    left = clientX - rect.width - 18;
  }
  if (top + rect.height + pad > window.innerHeight) {
    top = clientY - rect.height - 18;
  }
  tooltip.style.left = `${Math.max(pad, left)}px`;
  tooltip.style.top = `${Math.max(pad, top)}px`;
}

function hideTokenTooltip() {
  ui.tokenTooltip.classList.remove('visible');
  ui.tokenTooltip.setAttribute('aria-hidden', 'true');
  state.hoverToken = null;
}

function showTokenTooltip(token, event) {
  if (!token) return;
  state.hoverToken = token;
  const tooltip = ui.tokenTooltip;
  tooltip.innerHTML = '';

  const head = document.createElement('div');
  head.className = 'live-tooltip-head';
  const title = document.createElement('strong');
  title.textContent = `Token #${token.index ?? '-'}`;
  const text = document.createElement('span');
  text.textContent = displayTokenText(token.text);
  head.appendChild(title);
  head.appendChild(text);
  tooltip.appendChild(head);

  const grid = document.createElement('div');
  grid.className = 'live-tooltip-grid';
  for (const [label, value] of scalarTelemetry(token)) {
    const item = document.createElement('div');
    item.className = 'live-tooltip-item';
    const key = document.createElement('span');
    key.textContent = label;
    const val = document.createElement('strong');
    val.textContent = String(value ?? '-');
    item.appendChild(key);
    item.appendChild(val);
    grid.appendChild(item);
  }
  tooltip.appendChild(grid);

  const topn = Array.isArray(token?.topN) ? token.topN : [];
  if (topn.length) {
    const section = document.createElement('div');
    section.className = 'live-tooltip-topn';
    const heading = document.createElement('div');
    heading.className = 'live-tooltip-subhead';
    heading.textContent = 'Top Alternatives';
    section.appendChild(heading);
    topn.slice(0, 5).forEach((item, idx) => {
      const row = document.createElement('div');
      row.className = 'live-tooltip-alt';
      row.textContent = `#${idx + 1} ${displayTokenText(item?.token_text)} | p ${formatPercent(item?.prob, 1)} | lp ${formatNum(item?.logprob, 3)}`;
      section.appendChild(row);
    });
    tooltip.appendChild(section);
  }

  tooltip.classList.add('visible');
  tooltip.setAttribute('aria-hidden', 'false');
  ensureTooltipPosition(event.clientX, event.clientY);
}

function moveTokenTooltip(event) {
  if (!state.hoverToken || !ui.tokenTooltip.classList.contains('visible')) return;
  ensureTooltipPosition(event.clientX, event.clientY);
}

function renderRunTokens(container, run, fallbackText = '-') {
  const tokens = Array.isArray(run?.tokens) ? run.tokens : [];
  if (!tokens.length) {
    container.textContent = fallbackText;
    container.classList.remove('tokenized-output');
    return;
  }

  container.innerHTML = '';
  container.classList.add('tokenized-output');
  const peakIndex = peakRiskIndex(run);

  tokens.forEach((token, idx) => {
    const span = document.createElement('span');
    span.className = 'live-token';
    const risk = tokenRiskValue(token);
    if (risk != null) span.classList.add('has-telemetry');
    if (idx === peakIndex) span.classList.add('peak-risk');
    if (risk != null && risk >= 0.64) span.classList.add('alert-risk');
    span.textContent = token?.text || '';
    span.dataset.index = String(token?.index ?? idx);
    span.addEventListener('mouseenter', (event) => showTokenTooltip(token, event));
    span.addEventListener('mousemove', moveTokenTooltip);
    span.addEventListener('mouseleave', hideTokenTooltip);
    container.appendChild(span);
  });
}

function renderPassLog() {
  if (!ui.correctedPassLog) return;
  if (!state.passLog.length) {
    ui.correctedPassLog.classList.add('empty-list');
    ui.correctedPassLog.textContent = 'No rewrite pass has been accepted yet.';
    return;
  }
  ui.correctedPassLog.classList.remove('empty-list');
  ui.correctedPassLog.textContent = state.passLog.join('\n');
}

function appendPassLog(line) {
  const text = String(line || '').trim();
  if (!text) return;
  if (state.passLog[state.passLog.length - 1] === text) return;
  state.passLog.push(text);
  renderPassLog();
}

function traceLabel(value, fallback = 'pending') {
  const raw = String(value ?? '').trim();
  return raw ? raw.replace(/_/g, ' ').replace(/-/g, ' ') : fallback;
}

function tracePassNumber(value, fallback = 1) {
  const n = asNumber(value, fallback);
  return Math.max(1, Math.floor(n || fallback || 1));
}

function resetIterationTrace(loopBudget = state.loopBudget) {
  state.iterationTrace = {
    passes: [],
    finalEvidence: null,
  };
  state.loopBudget = asNumber(loopBudget, state.loopBudget) ?? 0;
  ensureTracePass(1, {
    status: 'running',
    currentStage: 'Replay path is decoding from the original prompt.',
    acceptedRewrites: 0,
    acceptedBefore: 0,
  });
  renderIterationTrace();
}

function ensureTracePass(passNumber, seed = {}) {
  const n = tracePassNumber(passNumber);
  let pass = state.iterationTrace.passes.find((item) => item.passNumber === n);
  if (!pass) {
    pass = {
      passNumber: n,
      status: 'running',
      currentStage: n === 1 ? 'Replay path is decoding from the original prompt.' : 'Continuing from the latest accepted trajectory.',
      acceptedBefore: Math.max(0, n - 1),
      acceptedRewrites: Math.max(0, n - 1),
      loopBudget: state.loopBudget,
      triggerIndex: null,
      rollbackIndex: null,
      tokenCount: null,
      maxRisk: null,
      harness: null,
      object: null,
      candidates: [],
      events: [],
      outcome: null,
      hasLiveActivity: false,
    };
    state.iterationTrace.passes.push(pass);
    state.iterationTrace.passes.sort((a, b) => a.passNumber - b.passNumber);
  }
  Object.assign(pass, Object.fromEntries(Object.entries(seed).filter(([, value]) => value !== undefined)));
  return pass;
}

function appendTraceEvent(pass, line) {
  const text = String(line || '').trim();
  if (!pass || !text) return;
  pass.hasLiveActivity = true;
  const last = pass.events[pass.events.length - 1];
  if (last === text) return;
  pass.events.push(text);
  if (pass.events.length > 7) pass.events = pass.events.slice(-7);
}

function traceCandidateKey(strategy, rank) {
  const safeStrategy = String(strategy || 'candidate');
  const safeRank = Number.isFinite(Number(rank)) ? Number(rank) : 'na';
  return `${safeStrategy}:${safeRank}`;
}

function candidateRankFromPayload(payload) {
  const raw = payload?.alt_rank ?? payload?.candidate_alt_rank ?? payload?.candidate_rank;
  return Number.isFinite(Number(raw)) ? Number(raw) : null;
}

function upsertTraceCandidate(pass, payload = {}) {
  const strategy = payload.strategy || payload.candidate_strategy || 'candidate';
  const rank = candidateRankFromPayload(payload);
  const key = traceCandidateKey(strategy, rank);
  let candidate = pass.candidates.find((item) => item.key === key);
  if (!candidate) {
    candidate = {
      key,
      strategy,
      rank,
      tokenText: payload.alt_token_text || (rank === -1 ? '[policy rewrite]' : rank === -2 ? '[global rewrite]' : ''),
      status: 'running',
      avgRisk: null,
      maxRisk: null,
      claimRisk: null,
      claimLabel: null,
      objectRisk: null,
      objectLabel: null,
    };
    pass.candidates.push(candidate);
  }
  if (payload.alt_token_text) candidate.tokenText = payload.alt_token_text;
  if (payload.avg_risk != null) candidate.avgRisk = payload.avg_risk;
  if (payload.max_risk != null) candidate.maxRisk = payload.max_risk;
  if (payload.claim_risk != null) candidate.claimRisk = payload.claim_risk;
  if (payload.claim_label) candidate.claimLabel = payload.claim_label;
  if (payload.object_risk != null) candidate.objectRisk = payload.object_risk;
  if (payload.object_label) candidate.objectLabel = payload.object_label;
  if (payload.stage === 'candidate_completed') candidate.status = 'completed';
  if (payload.stage === 'candidate_result') candidate.status = payload.accepted ? 'accepted' : 'rejected';
  if (payload.accepted === true) candidate.status = 'accepted';
  if (payload.accepted === false && candidate.status !== 'accepted') candidate.status = 'rejected';
  return candidate;
}

function describeTraceHarnessEvent(payload = {}) {
  const stage = payload.stage;
  if (stage === 'claim_selected') return 'Selected a check-worthy claim near the risk trigger.';
  if (stage === 'plan_ready') return `Prepared ${payload.facts_count ?? 0} factual probes.`;
  if (stage === 'plan_fallback') return 'Used heuristic fact extraction for this claim.';
  if (stage === 'probe_sample_completed') {
    return `Probe ${Number(payload.probe_index ?? 0) + 1}, sample ${payload.sample_index ?? '-'} answered ${payload.answer_text || 'unknown'}.`;
  }
  if (stage === 'judge_completed') {
    return `Judged detail ${Number(payload.fact_index ?? 0) + 1}: ${traceLabel(payload.verdict, 'uncertain')} ${formatPercent(payload.fact_risk, 0)}.`;
  }
  if (stage === 'harness_completed' || payload.claim_risk != null || payload.label) {
    return `Claim evidence: ${traceLabel(payload.label || payload.status)} ${formatPercent(payload.claim_risk, 0)}.`;
  }
  return payload.message || 'Evaluating claim evidence.';
}

function describeTraceObjectEvent(payload = {}) {
  const stage = payload.stage;
  if (stage === 'object_selected') return `Selected object: ${payload.object_summary || 'referenced source'}.`;
  if (stage === 'object_probe_completed') {
    return `Object sample ${Number(payload.sample_index ?? 0) + 1}: ${traceLabel(payload.verdict, 'uncertain')} ${formatPercent(payload.object_risk, 0)}.`;
  }
  if (stage === 'object_harness_completed' || payload.object_risk != null || payload.label) {
    return `Object evidence: ${traceLabel(payload.label || payload.status)} ${formatPercent(payload.object_risk, 0)}.`;
  }
  return payload.reason || 'Evaluating object-level evidence.';
}

function updateTraceFromStatus(payload = {}) {
  if (payload.phase === 'baseline' || payload.phase === 'setup') return;
  if (payload.phase === 'finalize') {
    const final = ensureFinalEvidenceTrace();
    final.status = 'running';
    final.currentStage = payload.message || 'Running final evidence passes.';
    appendFinalTraceEvent(final, payload.message || 'Running final evidence passes.');
    renderIterationTrace();
    return;
  }
  if (!payload.rewrite_pass && !payload.phase) return;
  const pass = ensureTracePass(payload.rewrite_pass || 1, {
    status: 'running',
    acceptedRewrites: asNumber(payload.accepted_rewrites, undefined),
    loopBudget: asNumber(payload.loop_budget, state.loopBudget),
    rollbackIndex: payload.rollback_index ?? undefined,
    triggerIndex: payload.trigger_index ?? undefined,
  });
  if (payload.message) {
    pass.currentStage = payload.message;
    appendTraceEvent(pass, payload.message);
    if (payload.phase === 'stopped') {
      pass.status = 'stopped';
      pass.outcome = payload.message;
    }
    if (payload.message.includes('No candidate cleared')) {
      pass.outcome = 'No candidate cleared the acceptance gates; replay continued.';
      pass.status = 'running';
    }
  }
  renderIterationTrace();
}

function updateTraceDecodeProgress(event = {}) {
  if (event.phase !== 'corrected') return;
  const pass = ensureTracePass(event.rewrite_pass || 1, {
    status: 'running',
    tokenCount: event.token_count ?? undefined,
    maxRisk: event.max_decoder_risk ?? undefined,
    acceptedRewrites: asNumber(event.accepted_rewrites, undefined),
    loopBudget: asNumber(event.loop_budget, state.loopBudget),
  });
  pass.currentStage = `Decoding token ${Number(event.token_index ?? 0) + 1} on this trajectory.`;
  renderIterationTrace();
}

function updateTraceFromHarness(payload = {}) {
  if (payload.phase === 'finalize') {
    const final = ensureFinalEvidenceTrace();
    final.status = 'running';
    final.currentStage = describeTraceHarnessEvent(payload);
    appendFinalTraceEvent(final, `${traceLabel(payload.target, 'target')} claim: ${describeTraceHarnessEvent(payload)}`);
    renderIterationTrace();
    return;
  }
  const pass = ensureTracePass(payload.rewrite_pass || 1, {
    rollbackIndex: payload.rollback_index ?? undefined,
    triggerIndex: payload.trigger_index ?? undefined,
  });
  const eventText = describeTraceHarnessEvent(payload);
  pass.currentStage = eventText;
  pass.harness = {
    stage: payload.stage || pass.harness?.stage || null,
    label: payload.label || pass.harness?.label || payload.status || null,
    risk: payload.claim_risk ?? pass.harness?.risk ?? null,
    claimText: payload.claim_text || pass.harness?.claimText || '',
  };
  const isCandidate = payload.phase === 'candidate' || payload.candidate_strategy;
  if (isCandidate) {
    const candidate = upsertTraceCandidate(pass, {
      strategy: payload.candidate_strategy,
      alt_rank: payload.candidate_alt_rank,
      claim_risk: payload.claim_risk,
      claim_label: payload.label,
    });
    candidate.status = candidate.status === 'running' ? 'completed' : candidate.status;
  }
  appendTraceEvent(pass, eventText);
  renderIterationTrace();
}

function updateTraceFromObject(payload = {}) {
  if (payload.phase === 'finalize') {
    const final = ensureFinalEvidenceTrace();
    final.status = 'running';
    final.currentStage = describeTraceObjectEvent(payload);
    appendFinalTraceEvent(final, `${traceLabel(payload.target, 'target')} object: ${describeTraceObjectEvent(payload)}`);
    renderIterationTrace();
    return;
  }
  const pass = ensureTracePass(payload.rewrite_pass || 1, {
    rollbackIndex: payload.rollback_index ?? undefined,
    triggerIndex: payload.trigger_index ?? undefined,
  });
  const eventText = describeTraceObjectEvent(payload);
  pass.currentStage = eventText;
  pass.object = {
    stage: payload.stage || pass.object?.stage || null,
    label: payload.label || pass.object?.label || payload.status || null,
    risk: payload.object_risk ?? pass.object?.risk ?? null,
    summary: payload.object_summary || pass.object?.summary || '',
    modeRecommendation: payload.mode_recommendation || pass.object?.modeRecommendation || null,
  };
  const isCandidate = payload.phase === 'candidate' || payload.candidate_strategy;
  if (isCandidate) {
    const candidate = upsertTraceCandidate(pass, {
      strategy: payload.candidate_strategy,
      alt_rank: payload.candidate_alt_rank,
      object_risk: payload.object_risk,
      object_label: payload.label,
    });
    candidate.status = candidate.status === 'running' ? 'completed' : candidate.status;
  }
  appendTraceEvent(pass, eventText);
  renderIterationTrace();
}

function updateTraceFromCandidate(payload = {}) {
  const pass = ensureTracePass(payload.rewrite_pass || 1, {
    rollbackIndex: payload.rollback_index ?? undefined,
    triggerIndex: payload.trigger_index ?? undefined,
  });
  const candidate = upsertTraceCandidate(pass, payload);
  const strategy = traceLabel(candidate.strategy, 'candidate');
  if (payload.stage === 'candidate_started') {
    pass.currentStage = `Evaluating ${strategy} candidate.`;
    appendTraceEvent(pass, `Started ${strategy} candidate ${displayTokenText(candidate.tokenText)}.`);
  } else if (payload.stage === 'candidate_completed') {
    pass.currentStage = `Candidate decoded; waiting for evidence checks.`;
    appendTraceEvent(pass, `Decoded ${strategy}: avg ${formatPercent(candidate.avgRisk, 0)}, max ${formatPercent(candidate.maxRisk, 0)}.`);
  } else if (payload.stage === 'candidate_result') {
    appendTraceEvent(pass, `${strategy} ${payload.accepted ? 'accepted' : 'rejected'} by acceptance gates.`);
  }
  renderIterationTrace();
}

function updateTraceFromIntervention(payload = {}) {
  const nextPassNumber = tracePassNumber(payload.rewrite_pass || 2);
  const currentPassNumber = Math.max(1, nextPassNumber - 1);
  const pass = ensureTracePass(currentPassNumber, {
    status: 'accepted',
    rollbackIndex: payload.rollback_index ?? undefined,
    triggerIndex: payload.trigger_index ?? undefined,
  });
  pass.outcome = `Accepted ${traceLabel(payload.chosen_strategy, 'candidate')} at trigger token ${payload.trigger_index ?? '-'}.`;
  pass.currentStage = pass.outcome;
  pass.acceptedRewrites = asNumber(payload.accepted_rewrites, pass.acceptedRewrites);
  pass.outcomeMetrics = {
    decoderBefore: payload.baseline_avg_risk,
    decoderAfter: payload.chosen_avg_risk,
    claimBefore: payload.baseline_claim_risk,
    claimAfter: payload.chosen_claim_risk,
    objectBefore: payload.baseline_object_risk,
    objectAfter: payload.chosen_object_risk,
  };
  (payload.candidates || []).forEach((candidatePayload) => {
    upsertTraceCandidate(pass, {
      ...candidatePayload,
      stage: 'candidate_result',
      strategy: candidatePayload.strategy,
      alt_rank: candidatePayload.alt_rank,
      alt_token_text: candidatePayload.alt_token_text,
      accepted: Boolean(candidatePayload.accepted),
    });
  });
  appendTraceEvent(pass, pass.outcome);
  ensureTracePass(nextPassNumber, {
    status: 'running',
    currentStage: `Continuing from accepted ${traceLabel(payload.chosen_strategy, 'candidate')} trajectory.`,
    acceptedRewrites: asNumber(payload.accepted_rewrites, nextPassNumber - 1),
    acceptedBefore: asNumber(payload.accepted_rewrites, nextPassNumber - 1),
  });
  renderIterationTrace();
}

function ensureFinalEvidenceTrace() {
  if (!state.iterationTrace.finalEvidence) {
    state.iterationTrace.finalEvidence = {
      status: 'running',
      currentStage: 'Waiting for final baseline and corrected evidence passes.',
      events: [],
    };
  }
  return state.iterationTrace.finalEvidence;
}

function appendFinalTraceEvent(final, line) {
  const text = String(line || '').trim();
  if (!text) return;
  if (final.events[final.events.length - 1] === text) return;
  final.events.push(text);
  if (final.events.length > 8) final.events = final.events.slice(-8);
}

function finalizeIterationTrace(result = {}) {
  const stopped = result?.status === 'stopped' || result?.corrected?.meta?.status === 'stopped';
  const interventionCount = Number(result?.metrics?.intervention_count ?? 0);
  const rewritePasses = Math.max(1, Number(result?.metrics?.rewrite_passes ?? interventionCount + 1));
  state.loopBudget = asNumber(result?.metrics?.correction_loops, state.loopBudget) ?? state.loopBudget;
  if (!state.iterationTrace.passes.length) {
    ensureTracePass(1, { status: 'running' });
  }
  (result.interventions || []).forEach((item, idx) => {
    const pass = ensureTracePass(idx + 1, {
      status: 'accepted',
      rollbackIndex: item.rollback_index,
      triggerIndex: item.trigger_index,
    });
    pass.outcome = `Accepted ${traceLabel(item.chosen_strategy, 'candidate')} at trigger token ${item.trigger_index ?? '-'}.`;
    pass.currentStage = pass.outcome;
    pass.outcomeMetrics = {
      decoderBefore: item.baseline_avg_risk,
      decoderAfter: item.chosen_avg_risk,
      claimBefore: item.baseline_claim_risk,
      claimAfter: item.chosen_claim_risk,
      objectBefore: item.baseline_object_risk,
      objectAfter: item.chosen_object_risk,
    };
    pass.harness = {
      ...(pass.harness || {}),
      label: item.baseline_claim_label || pass.harness?.label || null,
      risk: item.baseline_claim_risk ?? pass.harness?.risk ?? null,
      claimText: item.baseline_claim || pass.harness?.claimText || '',
    };
    pass.object = {
      ...(pass.object || {}),
      label: item.baseline_object_label || pass.object?.label || null,
      risk: item.baseline_object_risk ?? pass.object?.risk ?? null,
      summary: item.baseline_object_summary || pass.object?.summary || '',
    };
    (item.candidates || []).forEach((candidate) => {
      upsertTraceCandidate(pass, {
        ...candidate,
        stage: 'candidate_result',
        strategy: candidate.strategy,
        alt_rank: candidate.alt_rank,
        alt_token_text: candidate.alt_token_text,
        accepted: Boolean(candidate.accepted),
      });
    });
    appendTraceEvent(pass, pass.outcome);
  });
  for (let passNumber = 1; passNumber <= rewritePasses; passNumber += 1) {
    const pass = ensureTracePass(passNumber);
    if (passNumber <= interventionCount) {
      pass.status = 'accepted';
    } else if (stopped && passNumber === rewritePasses) {
      pass.status = 'stopped';
      pass.outcome = pass.outcome || 'Experiment stopped before this trajectory completed.';
      pass.currentStage = pass.outcome;
    } else {
      pass.status = 'complete';
      if (!pass.outcome) {
        pass.outcome = interventionCount > 0
          ? 'Final trajectory completed without another accepted rewrite.'
          : 'Replay completed without an accepted rewrite.';
      }
      pass.currentStage = pass.outcome;
    }
  }
  const final = ensureFinalEvidenceTrace();
  if (stopped) {
    final.status = 'skipped';
    final.currentStage = 'Final evidence passes were skipped because the experiment was stopped.';
    appendFinalTraceEvent(final, 'Stopped before final claim/object evidence could be collected.');
  } else {
    final.status = 'complete';
    final.currentStage = 'Final baseline/corrected evidence passes completed.';
    appendFinalTraceEvent(final, 'Final claim and object evidence are available in the evidence panels.');
  }
  renderIterationTrace();
}

function addTraceText(parent, className, text) {
  if (!text) return null;
  const node = document.createElement('div');
  node.className = className;
  node.textContent = text;
  parent.appendChild(node);
  return node;
}

function renderTraceMetric(parent, label, value) {
  if (value == null || !Number.isFinite(Number(value))) return;
  const node = document.createElement('span');
  node.className = 'trace-metric';
  node.textContent = `${label} ${formatPercent(value, 0)}`;
  parent.appendChild(node);
}

function renderIterationTrace() {
  if (!ui.iterationTrace) return;
  const passes = [...state.iterationTrace.passes].sort((a, b) => a.passNumber - b.passNumber);
  const final = state.iterationTrace.finalEvidence;
  const acceptedCount = passes.filter((pass) => pass.status === 'accepted').length;
  const activePass = [...passes].reverse().find((pass) => pass.status === 'running');
  const summaryBits = [
    `${passes.length || 0} pass${passes.length === 1 ? '' : 'es'}`,
    `accepted ${acceptedCount}/${Math.max(0, Number(state.loopBudget || 0))}`,
  ];
  if (final?.status === 'running') {
    summaryBits.push('final evidence running');
  } else if (activePass) {
    summaryBits.push(`current pass ${activePass.passNumber}`);
  }
  ui.iterationTraceSummary.textContent = summaryBits.join(' | ');

  ui.iterationTrace.innerHTML = '';
  if (!passes.length && !final) {
    const empty = document.createElement('div');
    empty.className = 'empty-list';
    empty.textContent = 'Run an experiment to see each replay, evidence, candidate, and rewrite decision.';
    ui.iterationTrace.appendChild(empty);
    return;
  }

  passes.forEach((pass) => {
    const block = document.createElement('article');
    block.className = `trace-pass ${pass.status || 'running'}`;

    const head = document.createElement('div');
    head.className = 'trace-head';
    const title = document.createElement('div');
    title.className = 'trace-title';
    title.textContent = `Pass ${pass.passNumber}`;
    const pill = document.createElement('span');
    pill.className = 'trace-pill';
    pill.textContent = traceLabel(pass.status, 'running');
    head.append(title, pill);
    block.appendChild(head);

    const meta = document.createElement('div');
    meta.className = 'trace-meta';
    const metaItems = [
      pass.passNumber === 1 ? 'starts from raw replay' : 'starts from accepted trajectory',
      `accepted before ${Math.max(0, pass.acceptedBefore ?? pass.passNumber - 1)}/${Math.max(0, Number(state.loopBudget || 0))}`,
    ];
    if (pass.tokenCount != null) metaItems.push(`tokens ${pass.tokenCount}`);
    if (pass.rollbackIndex != null) metaItems.push(`rollback ${pass.rollbackIndex}`);
    if (pass.triggerIndex != null) metaItems.push(`trigger ${pass.triggerIndex}`);
    meta.textContent = metaItems.join(' | ');
    block.appendChild(meta);

    addTraceText(block, 'trace-step', pass.currentStage || pass.outcome || 'Waiting for activity on this pass.');
    if (pass.harness?.claimText) {
      addTraceText(block, 'trace-claim', `Claim: ${pass.harness.claimText}`);
    }
    if (pass.object?.summary) {
      addTraceText(block, 'trace-claim', `Object: ${pass.object.summary}`);
    }

    const metrics = document.createElement('div');
    metrics.className = 'trace-metrics';
    renderTraceMetric(metrics, 'max', pass.maxRisk);
    renderTraceMetric(metrics, 'claim', pass.harness?.risk);
    renderTraceMetric(metrics, 'object', pass.object?.risk);
    if (pass.outcomeMetrics) {
      renderTraceMetric(metrics, 'decoder before', pass.outcomeMetrics.decoderBefore);
      renderTraceMetric(metrics, 'decoder after', pass.outcomeMetrics.decoderAfter);
      renderTraceMetric(metrics, 'claim after', pass.outcomeMetrics.claimAfter);
      renderTraceMetric(metrics, 'object after', pass.outcomeMetrics.objectAfter);
    }
    if (metrics.childElementCount) block.appendChild(metrics);

    if (pass.candidates.length) {
      const candidates = document.createElement('div');
      candidates.className = 'trace-candidates';
      pass.candidates.forEach((candidate) => {
        const row = document.createElement('div');
        row.className = `trace-candidate-row ${candidate.status || 'running'}`;
        const kind = document.createElement('span');
        kind.textContent = traceLabel(candidate.strategy, 'candidate');
        const token = document.createElement('span');
        token.className = 'trace-token';
        token.textContent = displayTokenText(candidate.tokenText || (candidate.rank >= 0 ? `alt ${candidate.rank + 1}` : '[rewrite]'));
        const avg = document.createElement('span');
        avg.textContent = `avg ${formatPercent(candidate.avgRisk, 0)}`;
        const claim = document.createElement('span');
        claim.textContent = `claim ${formatPercent(candidate.claimRisk, 0)}`;
        const object = document.createElement('span');
        object.textContent = `object ${formatPercent(candidate.objectRisk, 0)}`;
        const status = document.createElement('span');
        status.textContent = traceLabel(candidate.status, 'running');
        row.append(kind, token, avg, claim, object, status);
        candidates.appendChild(row);
      });
      block.appendChild(candidates);
    }

    if (pass.events.length) {
      const events = document.createElement('div');
      events.className = 'trace-events';
      pass.events.slice(-4).forEach((line) => {
        addTraceText(events, 'trace-event', line);
      });
      block.appendChild(events);
    }

    ui.iterationTrace.appendChild(block);
  });

  if (final) {
    const block = document.createElement('article');
    block.className = `trace-final ${final.status || 'running'}`;
    const head = document.createElement('div');
    head.className = 'trace-head';
    const title = document.createElement('div');
    title.className = 'trace-title';
    title.textContent = 'Final Evidence';
    const pill = document.createElement('span');
    pill.className = 'trace-pill';
    pill.textContent = traceLabel(final.status, 'running');
    head.append(title, pill);
    block.appendChild(head);
    addTraceText(block, 'trace-step', final.currentStage);
    if (final.events.length) {
      const events = document.createElement('div');
      events.className = 'trace-events';
      final.events.slice(-5).forEach((line) => addTraceText(events, 'trace-event', line));
      block.appendChild(events);
    }
    ui.iterationTrace.appendChild(block);
  }
}

function setCorrectedPassState({ rewritePass = 1, acceptedRewrites = 0, loopBudget = state.loopBudget, reason = '' } = {}) {
  state.loopBudget = asNumber(loopBudget, state.loopBudget) ?? 0;
  const maxPasses = 1 + Math.max(0, Number(state.loopBudget || 0));
  const suffix = reason ? ` ${reason}` : '';
  ui.correctedPassNote.textContent =
    `Current pass ${rewritePass} of ${maxPasses}. Accepted rewrites ${acceptedRewrites}/${Math.max(0, Number(state.loopBudget || 0))}.${suffix}`;
}

function renderCorrectedPanelFromRun(run, { title = null } = {}) {
  if (!run) return;
  if (title) ui.correctedTitle.textContent = title;
  ui.correctedRunId.textContent = run?.run_id || ui.correctedRunId.textContent;
  ui.correctedTokens.textContent = String(run?.summary?.token_count ?? ui.correctedTokens.textContent);
  ui.correctedRisk.textContent = formatPercent(run?.summary?.decoder_risk_max, 0);
  ui.correctedAlerts.textContent = String(run?.summary?.decoder_alert_count ?? ui.correctedAlerts.textContent);
  renderRunTokens(ui.correctedText, run, '-');
}

function renderBackend() {
  const info = state.backendInfo;
  ui.backendReachable.textContent = info ? (info.reachable ? 'yes' : 'no') : 'not probed';
  ui.backendModel.textContent = info?.model || '-';
  ui.backendProbs.textContent = info ? (info.token_probs ? 'yes' : info.completion ? 'no / weak' : '-') : '-';
  ui.backendEmbeddings.textContent = info ? (info.embedding ? 'yes' : 'no') : '-';
  ui.backendModelRequired.textContent = info ? capabilityText(info.model_required_explicit) : '-';
  ui.backendStopSemantics.textContent = info ? capabilityText(info.stop_semantics) : '-';
  ui.backendTokenForcing.textContent = info ? capabilityText(info.strict_token_forcing) : '-';
  ui.backendMode.textContent = currentLabMode() === 'retraction' ? 'internal consistency retraction' : 'hybrid replay / rewrite';
}

function applyBackendInfo(info, { updateStatus = true } = {}) {
  state.backendInfo = info || null;
  renderBackend();
  if (!updateStatus || !info) return;
  const baseUrl = normalizeBaseUrl(ui.endpoint.value) || info.base_url || '';
  if (info.probe_state === 'warming') {
    setStatus(`Backend warming: ${info.model || baseUrl || 'default backend'}`, 'neutral');
    return;
  }
  if (info.reachable) {
    setStatus(`${info.cached ? 'Backend warmed' : 'Backend ready'}: ${info.model || baseUrl}`, 'success');
    return;
  }
  if (info.error) {
    setStatus(`Backend probe failed: ${info.error}`, 'error');
    return;
  }
  setStatus('Backend probe unavailable.', 'neutral');
}

function renderInterventions(result) {
  const interventions = Array.isArray(result?.interventions) ? result.interventions : [];
  ui.interventionCount.textContent = String(interventions.length);
  ui.interventionList.innerHTML = '';

  if (!interventions.length) {
    const empty = document.createElement('div');
    empty.className = 'empty-list';
    empty.textContent = 'No intervention was triggered for this run.';
    ui.interventionList.appendChild(empty);
    return;
  }

  interventions.forEach((item, idx) => {
    const block = document.createElement('div');
    block.className = 'mutation-item';

    const head = document.createElement('div');
    head.className = 'mutation-head';
    head.innerHTML = `<span class="mutation-index">I${idx + 1} @ rollback ${item.rollback_index}</span><span class="mutation-index">trigger ${item.trigger_index}</span>`;
    block.appendChild(head);

    const summary = document.createElement('div');
    summary.className = 'mutation-parent';
    summary.textContent =
      `decoder avg ${formatPercent(item.baseline_avg_risk, 0)} -> ${formatPercent(item.chosen_avg_risk, 0)} | `
      + `claim ${formatPercent(item.baseline_claim_risk, 0)} -> ${formatPercent(item.chosen_claim_risk, 0)} `
      + `| object ${formatPercent(item.baseline_object_risk, 0)} -> ${formatPercent(item.chosen_object_risk, 0)} `
      + `via ${String(item.chosen_strategy || 'candidate').replace(/_/g, ' ')} `
      + `${item.chosen_alt_rank >= 0 ? `alt ${item.chosen_alt_rank + 1}` : ''} (${displayTokenText(item.chosen_alt_text)})`;
    block.appendChild(summary);

    const candidates = document.createElement('div');
    candidates.className = 'text-block';
    candidates.textContent = (item.candidates || [])
      .map((candidate) => {
        const accepted = candidate.accepted ? 'accepted' : 'candidate';
        const rank = candidate.alt_rank >= 0 ? `#${candidate.alt_rank + 1}` : '[rewrite]';
        return `${rank} ${accepted} | ${String(candidate.strategy || 'candidate').replace(/_/g, ' ')} | ${displayTokenText(candidate.alt_token_text)} | avg risk ${formatPercent(candidate.avg_risk, 0)} | claim ${formatPercent(candidate.claim_risk, 0)} | object ${formatPercent(candidate.object_risk, 0)} | ${candidate.claim_label || 'no harness'}`;
      })
      .join('\n');
    block.appendChild(candidates);

    ui.interventionList.appendChild(block);
  });
}

function formatObjectEvidence(harness) {
  if (!harness || typeof harness !== 'object') return '-';
  if (harness.status && harness.status !== 'ok') {
    return `Status: ${String(harness.status).replace(/_/g, ' ')}`;
  }
  const profile = harness.profile || {};
  const samples = Array.isArray(harness.samples) ? harness.samples : [];
  const lines = [
    `Focus: ${profile.summary || '-'}`,
    `Mode recommendation: ${String(harness.mode_recommendation || 'continue').replace(/_/g, ' ')}`,
    `Abstain recommended: ${harness.abstain_recommended ? 'yes' : 'no'}`,
    harness.reason_summary ? `Reason: ${harness.reason_summary}` : '',
  ].filter(Boolean);
  samples.forEach((sample, idx) => {
    lines.push(
      `Sample ${idx + 1}: ${(sample.verdict || 'uncertain').replace(/_/g, ' ')} | confidence ${(sample.confidence || 'mixed').replace(/_/g, ' ')} | `
      + `risk ${formatPercent(sample.risk, 0)} | abstain ${sample.abstain_recommended ? 'yes' : 'no'}`
    );
    if (sample.reason) lines.push(`Reason ${idx + 1}: ${sample.reason}`);
  });
  return lines.join('\n');
}

function renderObjectHarness(result) {
  const baselineObject = result?.object_harness?.baseline || null;
  const correctedObject = result?.object_harness?.corrected || null;
  const comparedLabel = Number(result?.metrics?.intervention_count ?? 0) > 0 ? 'Corrected' : 'Replay';

  ui.baselineObjectRisk.textContent = formatPercent(baselineObject?.object_risk, 0);
  ui.baselineObjectLabel.textContent = baselineObject?.label ? baselineObject.label.replace(/-/g, ' ') : (baselineObject?.status || '-').replace(/_/g, ' ');
  ui.correctedObjectRisk.textContent = formatPercent(correctedObject?.object_risk, 0);
  ui.correctedObjectLabel.textContent = correctedObject?.label ? correctedObject.label.replace(/-/g, ' ') : (correctedObject?.status || '-').replace(/_/g, ' ');

  const focusSummary = correctedObject?.profile?.summary || baselineObject?.profile?.summary || '-';
  ui.modeFocusObject.textContent = focusSummary;
  const lastIntervention = Array.isArray(result?.interventions) && result.interventions.length ? result.interventions[result.interventions.length - 1] : null;
  ui.modeCurrent.textContent = String(
    lastIntervention?.chosen_strategy
    || correctedObject?.mode_recommendation
    || (Number(result?.metrics?.intervention_count ?? 0) > 0 ? 'corrected' : 'replay')
  ).replace(/_/g, ' ');

  if (baselineObject?.object_risk != null && correctedObject?.object_risk != null) {
    ui.objectEvidence.textContent =
      `Baseline ${formatPercent(baselineObject.object_risk, 0)} (${baselineObject.label || baselineObject.status || '-'}) vs `
      + `${comparedLabel.toLowerCase()} ${formatPercent(correctedObject.object_risk, 0)} (${correctedObject.label || correctedObject.status || '-'}).\n\n`
      + formatObjectEvidence(correctedObject);
    return;
  }
  ui.objectEvidence.textContent = formatObjectEvidence(correctedObject || baselineObject);
}

function harnessClaimText(harness) {
  if (!harness || typeof harness !== 'object') return '-';
  return harness?.standalone_claim || harness?.claim?.text || '-';
}

function formatHarnessEvidence(harness) {
  if (!harness || typeof harness !== 'object') return '-';
  if (harness.status && harness.status !== 'ok') {
    return `Status: ${harness.status.replace(/_/g, ' ')}`;
  }
  const facts = Array.isArray(harness?.facts) ? harness.facts : [];
  if (!facts.length) return 'No targeted probes were captured for this claim.';
  return facts.map((item, idx) => {
    const answers = Array.isArray(item?.answers) ? item.answers : [];
    const judge = item?.judgement || {};
    return [
      `Fact ${idx + 1}: ${item?.fact || '-'}`,
      `Question: ${item?.question || '-'}`,
      `Answer 1: ${answers[0]?.text || '-'}`,
      `Answer 2: ${answers[1]?.text || '-'}`,
      `Judge: ${(judge?.verdict || '-').replace(/_/g, ' ')} | agreement ${(judge?.agreement || '-').replace(/_/g, ' ')} | risk ${formatPercent(judge?.fact_risk, 0)}`,
      judge?.reason ? `Reason: ${judge.reason}` : '',
    ].filter(Boolean).join('\n');
  }).join('\n\n');
}

function renderHarness(result) {
  const baselineHarness = result?.harness?.baseline || null;
  const correctedHarness = result?.harness?.corrected || null;
  const comparedLabel = Number(result?.metrics?.intervention_count ?? 0) > 0 ? 'Corrected' : 'Replay';

  ui.baselineClaimRisk.textContent = formatPercent(baselineHarness?.claim_risk, 0);
  ui.baselineClaimLabel.textContent = baselineHarness?.label ? baselineHarness.label.replace(/-/g, ' ') : (baselineHarness?.status || '-').replace(/_/g, ' ');
  ui.baselineClaimText.textContent = harnessClaimText(baselineHarness);
  ui.baselineClaimEvidence.textContent = formatHarnessEvidence(baselineHarness);

  ui.correctedClaimRisk.textContent = formatPercent(correctedHarness?.claim_risk, 0);
  ui.correctedClaimLabel.textContent = correctedHarness?.label ? correctedHarness.label.replace(/-/g, ' ') : (correctedHarness?.status || '-').replace(/_/g, ' ');
  ui.correctedClaimText.textContent = harnessClaimText(correctedHarness);
  ui.correctedClaimEvidence.textContent = formatHarnessEvidence(correctedHarness);

  if (baselineHarness?.claim_risk != null && correctedHarness?.claim_risk != null) {
    ui.harnessSummary.textContent =
      `Baseline focus-claim risk ${formatPercent(baselineHarness.claim_risk, 0)} vs ${comparedLabel.toLowerCase()} ${formatPercent(correctedHarness.claim_risk, 0)}. `
      + `These scores come from targeted factual probes, not from decoder telemetry alone.`;
    return;
  }

  if (correctedHarness?.status) {
    ui.harnessSummary.textContent = `Harness status: ${(correctedHarness.status || '-').replace(/_/g, ' ')}.`;
    return;
  }
  ui.harnessSummary.textContent = 'No claim-level harness evidence was collected for this run.';
}

function resetExperimentView() {
  state.lastResult = null;
  state.partialInterventions = [];
  state.partialRuns = { baseline: null, corrected: null, rewritePreview: null };
  state.passLog = [];
  state.iterationTrace = { passes: [], finalEvidence: null };
  resetRetractionTrace();
  state.loopBudget = 0;
  hideTokenTooltip();
  ui.experimentMeta.textContent = 'Experiment is running...';
  ui.experimentSummary.textContent = currentLabMode() === 'retraction'
    ? 'Waiting for draft, commitment extraction, probes, reconciliation, and verifier checks.'
    : 'Waiting for the first token-level updates from the live experiment loop.';
  ui.iterationTraceSummary.textContent = 'Waiting for correction loop activity.';
  ui.iterationTrace.innerHTML = '<div class=\"empty-list\">Waiting for replay, evidence, candidate, and rewrite events.</div>';
  ui.retractionTraceSummary.textContent = 'Waiting for retraction loop activity.';
  ui.retractionDraft.textContent = 'No draft yet.';
  ui.retractionCommitments.textContent = 'No commitments yet.';
  ui.retractionProbes.textContent = 'No probe packets yet.';
  ui.retractionScoring.textContent = 'No scoring output yet.';
  ui.retractionReconciliation.textContent = 'No reconciliation pass yet.';
  ui.retractionFinalAnswer.textContent = 'No final answer yet.';
  ui.retractionVerifier.textContent = 'No verifier result yet.';
  ui.retractionProvenance.textContent = 'No provenance yet.';
  ui.retractionBudget.textContent = 'No budget state yet.';
  ui.baselineRunId.textContent = '-';
  ui.baselineRisk.textContent = '-';
  ui.baselineAlerts.textContent = '-';
  ui.baselineTokens.textContent = '-';
  ui.baselineText.classList.remove('tokenized-output');
  ui.baselineText.textContent = '-';
  ui.correctedRunId.textContent = '-';
  ui.correctedRisk.textContent = '-';
  ui.correctedAlerts.textContent = '-';
  ui.correctedTokens.textContent = '-';
  ui.correctedPassNote.textContent = 'Current replay/rewrite pass will appear here.';
  ui.correctedPassLog.textContent = 'No rewrite pass has been accepted yet.';
  ui.correctedPassLog.classList.add('empty-list');
  ui.correctedText.classList.remove('tokenized-output');
  ui.correctedText.textContent = '-';
  ui.interventionCount.textContent = '0';
  ui.interventionList.innerHTML = '<div class=\"empty-list\">No intervention has been triggered yet.</div>';
  ui.modeCurrent.textContent = 'replay';
  ui.modeFocusObject.textContent = '-';
  ui.baselineObjectRisk.textContent = '-';
  ui.baselineObjectLabel.textContent = '-';
  ui.correctedObjectRisk.textContent = '-';
  ui.correctedObjectLabel.textContent = '-';
  ui.objectEvidence.textContent = 'No object-level evidence yet.';
  ui.harnessSummary.textContent = 'Waiting for a completed check-worthy claim before the harness can evaluate anything.';
  ui.baselineClaimRisk.textContent = '-';
  ui.baselineClaimLabel.textContent = '-';
  ui.baselineClaimText.textContent = '-';
  ui.baselineClaimEvidence.textContent = '-';
  ui.correctedClaimRisk.textContent = '-';
  ui.correctedClaimLabel.textContent = '-';
  ui.correctedClaimText.textContent = '-';
  ui.correctedClaimEvidence.textContent = '-';
  setLabModeVisibility();
}

function renderPartialInterventions() {
  renderInterventions({ interventions: state.partialInterventions });
}

function updateProgress(event) {
  const phase = event?.phase;
  if (phase === 'baseline') {
    const run = applyProgressToken('baseline', event);
    ui.baselineRunId.textContent = event.run_id || ui.baselineRunId.textContent;
    ui.baselineTokens.textContent = String(run?.summary?.token_count ?? event.token_count ?? ui.baselineTokens.textContent);
    ui.baselineRisk.textContent = formatPercent(event.max_decoder_risk ?? run?.summary?.decoder_risk_max, 0);
    ui.baselineAlerts.textContent = String(event.alert_count ?? run?.summary?.decoder_alert_count ?? ui.baselineAlerts.textContent);
    renderRunTokens(ui.baselineText, run, event.text || ui.baselineText.textContent);
    setStatus(`Baseline token ${event.token_index + 1}`, 'neutral');
  } else if (phase === 'corrected') {
    const run = applyProgressToken('corrected', event);
    updateTraceDecodeProgress(event);
    renderCorrectedPanelFromRun(run, {
      title: Number(event.accepted_rewrites ?? 0) > 0 ? 'Corrected Candidate' : 'Replay Sample',
    });
    setCorrectedPassState({
      rewritePass: asNumber(event.rewrite_pass, 1),
      acceptedRewrites: asNumber(event.accepted_rewrites, 0),
      loopBudget: asNumber(event.loop_budget, state.loopBudget),
      reason: Number(event.accepted_rewrites ?? 0) > 0
        ? 'Current text reflects the latest accepted rewrite pass.'
        : 'Current text reflects the live replay path.',
    });
    setStatus(`Corrected token ${event.token_index + 1} on pass ${event.rewrite_pass ?? 1}`, 'neutral');
  } else if (phase === 'rewrite_candidate') {
    const run = applyProgressToken('rewritePreview', event);
    const title = event?.strategy === 'global_rewrite' ? 'Global Rewrite Preview' : 'Rewrite Preview';
    renderCorrectedPanelFromRun(run, { title });
    setCorrectedPassState({
      rewritePass: asNumber(event.rewrite_pass, 1),
      acceptedRewrites: asNumber(event.accepted_rewrites, 0),
      loopBudget: asNumber(event.loop_budget, state.loopBudget),
      reason: event?.strategy === 'global_rewrite'
        ? 'Evaluating a whole-answer conservative rewrite candidate before acceptance.'
        : 'Evaluating a conservative rewrite candidate before acceptance.',
    });
    setStatus(`${title} token ${event.token_index + 1} on pass ${event.rewrite_pass ?? 1}`, 'neutral');
  }
}

function upsertProbePacket(packet) {
  if (!packet?.commitment_id) return;
  const idx = state.retractionTrace.probePackets.findIndex((item) => item.commitment_id === packet.commitment_id);
  if (idx >= 0) {
    state.retractionTrace.probePackets[idx] = packet;
  } else {
    state.retractionTrace.probePackets.push(packet);
  }
}

function handleRetractionEvent(payload) {
  state.retractionTrace.metrics = payload.metrics || state.retractionTrace.metrics;
  if (payload.provenance) state.retractionTrace.provenance = payload.provenance;

  if (payload.type === 'retraction_experiment_started') {
    state.retractionTrace.status = 'running';
    ui.experimentMeta.textContent = `Mode=${payload.mode || 'internal_consistency_retraction_lab'} | experiment=${payload.experiment_id || '-'}`;
    ui.experimentSummary.textContent = 'Draft generation started. The trace below will show each verifier loop step as it completes.';
    renderRetractionTrace();
    setStatus('Retraction experiment started.', 'neutral');
    return;
  }

  if (payload.type === 'retraction_experiment_status') {
    const waiting = asNumber(payload.waiting_ms, 0);
    const seconds = waiting >= 1000 ? ` (${(waiting / 1000).toFixed(waiting >= 10000 ? 0 : 1)}s)` : '';
    if (payload.message) setStatus(`${payload.message}${seconds}`, 'neutral');
    state.retractionTrace.status = payload.phase || state.retractionTrace.status;
    renderRetractionBudget(payload.metrics || state.retractionTrace.metrics);
    renderRetractionTraceSummary();
    return;
  }

  if (payload.type === 'retraction_experiment_draft') {
    if (payload.stage === 'draft_completed') {
      state.retractionTrace.draft = payload;
      ui.retractionDraft.textContent =
        `Model: ${payload.model || '-'}\nTokens: ${payload.token_budget_used ?? '-'}\nStop reason: ${payload.stop_reason || '-'}\n\n${payload.draft_answer || '-'}`;
      ui.experimentSummary.textContent = 'Draft completed. Extracting commitments and separating model claims from user-provided context.';
    } else if (payload.text) {
      ui.retractionDraft.textContent = payload.text;
    }
    renderRetractionBudget(payload.metrics || state.retractionTrace.metrics);
    renderRetractionTraceSummary();
    return;
  }

  if (payload.type === 'retraction_experiment_commitments') {
    state.retractionTrace.commitments = payload.commitments || [];
    renderRetractionCommitments();
    renderRetractionBudget(payload.metrics || state.retractionTrace.metrics);
    renderRetractionTraceSummary();
    ui.experimentSummary.textContent = `Extracted ${state.retractionTrace.commitments.length} commitments. Running probes for selected model-draft commitments.`;
    return;
  }

  if (payload.type === 'retraction_experiment_probe') {
    state.retractionTrace.probeEvents.push(payload);
    if (payload.packet) upsertProbePacket(payload.packet);
    if (payload.stage === 'probe_sample_completed' && payload.observation) {
      const tempPacket = state.retractionTrace.probePackets.find((item) => item.commitment_id === payload.commitment_id);
      if (!tempPacket) {
        upsertProbePacket({
          commitment_id: payload.commitment_id,
          evidence_basis: 'internal_probe_consistency',
          probe_observations: [payload.observation],
          agreement_score: null,
          contradiction_score: null,
          emptiness_score: null,
          consistency_label: 'pending',
          deterministic_recommendation: 'pending',
          scoring_debug: {},
        });
      } else if (!payload.packet) {
        const observations = tempPacket.probe_observations || [];
        if (!observations.some((item) => item.sample_index === payload.observation.sample_index)) {
          tempPacket.probe_observations = [...observations, payload.observation];
        }
      }
    }
    renderRetractionProbePackets();
    renderRetractionBudget(payload.metrics || state.retractionTrace.metrics);
    renderRetractionTraceSummary();
    return;
  }

  if (payload.type === 'retraction_experiment_reconciliation') {
    if (payload.reconciliation_pass) {
      state.retractionTrace.reconciliationPasses = [payload.reconciliation_pass];
    }
    if (payload.final_answer) ui.retractionFinalAnswer.textContent = payload.final_answer;
    renderRetractionReconciliation();
    renderRetractionProvenance(payload.provenance || state.retractionTrace.provenance);
    renderRetractionBudget(payload.metrics || state.retractionTrace.metrics);
    renderRetractionTraceSummary();
    ui.experimentSummary.textContent = 'Reconciliation completed. Running deterministic behavioral verifier.';
    return;
  }

  if (payload.type === 'retraction_experiment_completed') {
    renderRetractionResult(payload);
    setExperimentRunning(false);
    setStatus(
      payload.status === 'contract_satisfied' ? 'Retraction contract satisfied.' : `Retraction result: ${(payload.status || 'needs_review').replace(/_/g, ' ')}.`,
      payload.status === 'contract_satisfied' ? 'success' : 'neutral',
    );
  }
}

function connectStream() {
  const source = new EventSource('/stream');
  source.onmessage = (message) => {
    let payload;
    try {
      payload = JSON.parse(message.data);
    } catch {
      return;
    }
    if (!payload?.type) return;

    if (payload.type === 'backend_probe_updated' && payload.backend) {
      const endpoint = normalizeBaseUrl(ui.endpoint.value);
      if (!endpoint || payload.backend.base_url === endpoint) {
        applyBackendInfo(payload.backend, { updateStatus: false });
      }
      return;
    }

    if (!payload.experiment_id || payload.experiment_id !== state.currentExperimentId) return;

    if (String(payload.type).startsWith('retraction_experiment_')) {
      handleRetractionEvent(payload);
      return;
    }

    if (payload.type === 'live_experiment_started') {
      state.loopBudget = asNumber(payload.correction_loops, 0) ?? 0;
      state.passLog = ['Pass 1 started from the raw replay path.'];
      resetIterationTrace(state.loopBudget);
      renderPassLog();
      setCorrectedPassState({ rewritePass: 1, acceptedRewrites: 0, loopBudget: state.loopBudget });
      ui.correctedTitle.textContent = 'Replay Sample';
      ui.modeCurrent.textContent = 'replay';
      ui.baselineRunId.textContent = payload.baseline_run_id || '-';
      ui.correctedRunId.textContent = payload.corrected_run_id || '-';
      ui.experimentMeta.textContent =
        `Mode=${payload.mode || 'prefix_replay'} | correction_loops=${payload.correction_loops ?? 0} | `
        + `baseline=${payload.baseline_run_id || '-'} | corrected=${payload.corrected_run_id || '-'}`;
      return;
    }

    if (payload.type === 'live_experiment_progress') {
      updateProgress(payload);
      return;
    }

    if (payload.type === 'live_experiment_status') {
      const waiting = asNumber(payload.waiting_ms, 0);
      const seconds = waiting >= 1000 ? ` (${(waiting / 1000).toFixed(waiting >= 10000 ? 0 : 1)}s)` : '';
      if (payload.message) {
        setStatus(`${payload.message}${seconds}`, 'neutral');
      }
      updateTraceFromStatus(payload);
      if (payload.phase === 'rewrite_candidate') {
        setCorrectedPassState({
          rewritePass: asNumber(payload.rewrite_pass, 1),
          acceptedRewrites: asNumber(payload.accepted_rewrites, 0),
          loopBudget: asNumber(payload.loop_budget, state.loopBudget),
          reason: payload.message || 'Evaluating a rewrite candidate.',
        });
      }
      return;
    }

    if (payload.type === 'live_experiment_harness') {
      updateTraceFromHarness(payload);
      if (payload.stage) {
        const stage = String(payload.stage);
        if (stage === 'probe_sample_completed') {
          ui.harnessSummary.textContent =
            `Probe ${Number(payload.probe_index ?? 0) + 1} sample ${payload.sample_index ?? '-'}: ${payload.answer_text || 'unknown'}`;
        } else if (stage === 'judge_completed') {
          ui.harnessSummary.textContent =
            `Judged detail ${Number(payload.fact_index ?? 0) + 1}: ${(payload.verdict || 'uncertain').replace(/_/g, ' ')}`
            + (payload.fact_risk != null ? ` (${formatPercent(payload.fact_risk, 0)})` : '');
        } else if (stage === 'harness_completed') {
          ui.harnessSummary.textContent =
            `Harness checked claim near token ${payload.rollback_index ?? '-'}: ${(payload.label || 'pending').replace(/-/g, ' ')}`
            + (payload.claim_risk != null ? ` (${formatPercent(payload.claim_risk, 0)})` : '')
            + `. ${payload.claim_text || 'pending claim extraction'}`;
        } else {
          ui.harnessSummary.textContent = payload.claim_text || payload.message || 'Evaluating risky claim.';
        }
      } else {
        const claim = payload.claim_text || 'pending claim extraction';
        const label = payload.label ? payload.label.replace(/-/g, ' ') : (payload.status || 'pending').replace(/_/g, ' ');
        ui.harnessSummary.textContent =
          `Harness checked claim near token ${payload.rollback_index ?? '-'}: ${label}`
          + (payload.claim_risk != null ? ` (${formatPercent(payload.claim_risk, 0)})` : '')
          + `. ${claim}`;
      }
      return;
    }

    if (payload.type === 'live_experiment_object') {
      updateTraceFromObject(payload);
      if (payload.object_summary) {
        ui.modeFocusObject.textContent = payload.object_summary;
      }
      if (payload.mode_recommendation) {
        ui.modeCurrent.textContent = String(payload.mode_recommendation).replace(/_/g, ' ');
      }
      if (payload.phase === 'corrected') {
        ui.correctedObjectRisk.textContent = formatPercent(payload.object_risk, 0);
        ui.correctedObjectLabel.textContent = payload.label ? payload.label.replace(/-/g, ' ') : (payload.status || '-').replace(/_/g, ' ');
      }
      if (payload.target === 'baseline') {
        ui.baselineObjectRisk.textContent = formatPercent(payload.object_risk, 0);
        ui.baselineObjectLabel.textContent = payload.label ? payload.label.replace(/-/g, ' ') : (payload.status || '-').replace(/_/g, ' ');
      }
      if (payload.target === 'corrected') {
        ui.correctedObjectRisk.textContent = formatPercent(payload.object_risk, 0);
        ui.correctedObjectLabel.textContent = payload.label ? payload.label.replace(/-/g, ' ') : (payload.status || '-').replace(/_/g, ' ');
      }
      if (payload.stage === 'object_harness_completed') {
        ui.objectEvidence.textContent =
          `${payload.object_summary || 'Focused object'} | ${(payload.label || 'pending').replace(/-/g, ' ')} `
          + `${payload.object_risk != null ? `(${formatPercent(payload.object_risk, 0)})` : ''}\n`
          + `Mode recommendation: ${String(payload.mode_recommendation || 'continue').replace(/_/g, ' ')}\n`
          + `Abstain recommended: ${payload.abstain_recommended ? 'yes' : 'no'}`
          + (payload.reason ? `\nReason: ${payload.reason}` : '');
      } else if (payload.stage === 'object_probe_completed') {
        ui.objectEvidence.textContent =
          `Object audit sample ${Number(payload.sample_index ?? 0) + 1}: ${(payload.verdict || 'uncertain').replace(/_/g, ' ')} `
          + `${payload.object_risk != null ? `(${formatPercent(payload.object_risk, 0)})` : ''}`
          + (payload.reason ? `\nReason: ${payload.reason}` : '');
      } else if (payload.object_summary || payload.reason) {
        ui.objectEvidence.textContent = [payload.object_summary, payload.reason].filter(Boolean).join('\n');
      }
      return;
    }

    if (payload.type === 'live_experiment_candidate') {
      updateTraceFromCandidate(payload);
      if (payload.strategy === 'policy_rewrite' || payload.strategy === 'global_rewrite') {
        const isGlobal = payload.strategy === 'global_rewrite';
        const label = isGlobal ? 'whole-answer rewrite' : 'conservative rewrite candidate';
        if (payload.stage === 'candidate_started') {
          appendPassLog(`Pass ${payload.rewrite_pass ?? 1}: started ${label}.`);
        } else if (payload.stage === 'candidate_completed') {
          appendPassLog(
            `Pass ${payload.rewrite_pass ?? 1}: ${label} decoded with avg risk ${formatPercent(payload.avg_risk, 0)}.`
          );
        } else if (payload.stage === 'candidate_result' && payload.accepted === false) {
          const restored = applyRunSnapshot('corrected', payload.current_corrected_run);
          if (restored) {
            renderCorrectedPanelFromRun(restored, {
              title: Number(payload.accepted_rewrites ?? 0) > 0 ? 'Corrected Candidate' : 'Replay Sample',
            });
          }
          setCorrectedPassState({
            rewritePass: asNumber(payload.rewrite_pass, 1),
            acceptedRewrites: asNumber(payload.accepted_rewrites, 0),
            loopBudget: asNumber(payload.loop_budget, state.loopBudget),
            reason: `Rejected ${label}; returned to the current accepted path.`,
          });
          appendPassLog(`Pass ${payload.rewrite_pass ?? 1}: ${label} rejected; resumed current path.`);
        }
      }
      return;
    }

    if (payload.type === 'live_experiment_intervention') {
      updateTraceFromIntervention(payload);
      state.partialInterventions.push(payload);
      renderPartialInterventions();
      const acceptedPass = Math.max(1, asNumber(payload.rewrite_pass, 2) - 1);
      const nextPass = acceptedPass + 1;
      if (payload.corrected_run) {
        const run = applyRunSnapshot('corrected', payload.corrected_run);
        renderCorrectedPanelFromRun(run, { title: 'Corrected Candidate' });
      } else if (payload.corrected_text) {
        ui.correctedText.textContent = payload.corrected_text;
      }
      ui.correctedTitle.textContent = 'Corrected Candidate';
      setCorrectedPassState({
        rewritePass: asNumber(payload.rewrite_pass, 2),
        acceptedRewrites: asNumber(payload.accepted_rewrites, 1),
        loopBudget: asNumber(payload.loop_budget, state.loopBudget),
        reason: `Accepted ${String(payload.chosen_strategy || 'candidate').replace(/_/g, ' ')} at token ${payload.trigger_index}.`,
      });
      state.passLog.push(
        `Pass ${acceptedPass}: accepted ${String(payload.chosen_strategy || 'candidate').replace(/_/g, ' ')} at token ${payload.trigger_index}; pass ${nextPass} starts from that trajectory. `
        + `Decoder ${formatPercent(payload.baseline_avg_risk, 0)} -> ${formatPercent(payload.chosen_avg_risk, 0)} | `
        + `claim ${formatPercent(payload.baseline_claim_risk, 0)} -> ${formatPercent(payload.chosen_claim_risk, 0)}.`
      );
      renderPassLog();
      if (payload.baseline_harness || payload.chosen_harness) {
        renderHarness({ harness: { baseline: payload.baseline_harness, corrected: payload.chosen_harness }, metrics: { intervention_count: 1 } });
      }
      if (payload.baseline_object_harness || payload.chosen_object_harness) {
        renderObjectHarness({ object_harness: { baseline: payload.baseline_object_harness, corrected: payload.chosen_object_harness }, metrics: { intervention_count: 1 } });
      }
      setStatus(`Accepted rewrite after pass ${acceptedPass} at token ${payload.trigger_index}`, 'neutral');
      return;
    }

    if (payload.type === 'live_experiment_completed') {
      renderResult(payload);
      finalizeIterationTrace(payload);
      setExperimentRunning(false);
      setStatus(payload.status === 'stopped' ? 'Experiment stopped.' : 'Experiment completed.', payload.status === 'stopped' ? 'neutral' : 'success');
    }
  };

  source.onerror = () => {
    if (state.currentExperimentId) {
      setStatus('Stream reconnecting...', 'neutral');
    }
  };
}

function renderRetractionResult(result) {
  state.lastResult = result;
  state.retractionTrace.status = result?.status || 'needs_review';
  state.retractionTrace.commitments = result?.commitments || [];
  state.retractionTrace.probePackets = result?.probe_packets || [];
  state.retractionTrace.reconciliationPasses = result?.reconciliation_passes || [];
  state.retractionTrace.verifierTrace = result?.verifier_trace || [];
  state.retractionTrace.provenance = result?.provenance || null;
  state.retractionTrace.metrics = result?.metrics || null;

  ui.experimentMeta.textContent =
    `Mode=${result?.mode || '-'} | status=${(result?.status || '-').replace(/_/g, ' ')} | model_calls=${result?.metrics?.model_calls ?? 0} | duration=${result?.metrics?.duration_ms ?? '-'}ms`;
  ui.experimentSummary.textContent =
    result?.status === 'contract_satisfied'
      ? 'The final answer passed deterministic behavioral verification for the internal-consistency retraction contract. This is not factual verification.'
      : `The run needs review: ${(result?.status || 'unknown').replace(/_/g, ' ')}. Inspect verifier failures and probe packets below.`;

  ui.retractionDraft.textContent =
    `Model: ${result?.provenance?.draft_model || '-'}\nTokens: ${result?.draft_run?.summary?.token_count ?? '-'}\nStop reason: ${result?.draft_run?.meta?.stop?.reason || '-'}\n\n${result?.draft_answer || '-'}`;
  ui.retractionFinalAnswer.textContent = result?.final_answer || '-';

  renderRetractionTraceSummary(result);
  renderRetractionCommitments();
  renderRetractionProbePackets();
  renderRetractionReconciliation();
  renderRetractionVerifier();
  renderRetractionProvenance();
  renderRetractionBudget();

  const draftRun = result?.draft_run;
  if (draftRun?.run_id) {
    const link = new URL('./', window.location.href);
    link.searchParams.set('runA', draftRun.run_id);
    link.searchParams.set('focus', 'max-risk');
    ui.openMainLink.href = link.toString();
  }
}

function renderResult(result) {
  state.lastResult = result;
  const baseline = result?.baseline;
  const corrected = result?.corrected;
  const interventionCount = Number(result?.metrics?.intervention_count ?? 0);
  const stopped = result?.status === 'stopped' || corrected?.meta?.status === 'stopped';
  const correctedLabel = stopped ? 'Stopped Sample' : interventionCount > 0 ? 'Corrected Candidate' : 'Replay Sample';
  ui.correctedTitle.textContent = correctedLabel;

  if (!state.passLog.length) {
    state.passLog = ['Pass 1 started from the raw replay path.'];
  }
  if (stopped) {
    state.passLog = state.passLog.length
      ? state.passLog
      : ['Experiment stopped before the live correction loop completed.'];
  } else if (interventionCount > 0) {
    const fromResult = (result?.interventions || []).map((item, idx) =>
      `Pass ${idx + 1}: accepted ${String(item.chosen_strategy || 'candidate').replace(/_/g, ' ')} at token ${item.trigger_index}; pass ${idx + 2} starts from that trajectory. `
      + `Decoder ${formatPercent(item.baseline_avg_risk, 0)} -> ${formatPercent(item.chosen_avg_risk, 0)} | `
      + `claim ${formatPercent(item.baseline_claim_risk, 0)} -> ${formatPercent(item.chosen_claim_risk, 0)}.`
    );
    state.passLog = ['Pass 1 started from the raw replay path.', ...fromResult];
  } else {
    state.passLog = ['Pass 1 completed as a matched replay sample. No rewrite was accepted.'];
  }
  renderPassLog();
  setCorrectedPassState({
    rewritePass: asNumber(result?.metrics?.rewrite_passes, 1),
    acceptedRewrites: interventionCount,
    loopBudget: asNumber(result?.metrics?.correction_loops, state.loopBudget),
    reason: interventionCount > 0
      ? 'Final panel shows the latest accepted rewrite candidate.'
      : 'No rewrite cleared the acceptance gates, so the final panel remains a replay sample.',
  });

  ui.experimentMeta.textContent =
    `Mode=${result?.mode || '-'} | correction_loops=${result?.metrics?.correction_loops ?? 0} | `
    + `accepted_rewrites=${result?.metrics?.intervention_count ?? 0} | baseline=${result?.baseline_run_id || '-'} | corrected=${result?.corrected_run_id || '-'}`;
  if (stopped) {
    ui.experimentSummary.textContent =
      `Experiment stopped by user. Baseline tokens ${baseline?.summary?.token_count ?? '-'}; compared sample tokens ${corrected?.summary?.token_count ?? '-'}. `
      + `Final claim/object evidence passes were skipped, so this result is a partial trace, not a completed comparison.`;
  } else if (interventionCount > 0) {
    ui.experimentSummary.textContent =
      `Baseline max risk ${formatPercent(result?.metrics?.baseline_max_risk, 0)} vs corrected ${formatPercent(result?.metrics?.corrected_max_risk, 0)}. `
      + `Claim risk: ${formatPercent(result?.metrics?.baseline_claim_risk, 0)} -> ${formatPercent(result?.metrics?.corrected_claim_risk, 0)}. `
      + `Object risk: ${formatPercent(result?.metrics?.baseline_object_risk, 0)} -> ${formatPercent(result?.metrics?.corrected_object_risk, 0)}. `
      + `Alerts: ${result?.metrics?.baseline_alerts ?? '-'} -> ${result?.metrics?.corrected_alerts ?? '-'}. `
      + `Accepted rewrites: ${result?.metrics?.intervention_count ?? 0}/${result?.metrics?.correction_loops ?? 0}. `
      + `Use "Open In Snake Scope" to inspect both trajectories in the main visualizer.`;
  } else {
    ui.experimentSummary.textContent =
      `No intervention fired. Baseline max risk ${formatPercent(result?.metrics?.baseline_max_risk, 0)} and replay max risk ${formatPercent(result?.metrics?.corrected_max_risk, 0)} are two matched decode samples, not evidence that a correction policy helped or hurt. `
      + `Claim risk is ${formatPercent(result?.metrics?.baseline_claim_risk, 0)} vs ${formatPercent(result?.metrics?.corrected_claim_risk, 0)}. `
      + `Object risk is ${formatPercent(result?.metrics?.baseline_object_risk, 0)} vs ${formatPercent(result?.metrics?.corrected_object_risk, 0)}. `
      + `Accepted rewrites: 0/${result?.metrics?.correction_loops ?? 0}. `
      + `Use "Open In Snake Scope" to inspect both trajectories in the main visualizer.`;
  }

  applyRunSnapshot('baseline', baseline);
  ui.baselineRunId.textContent = baseline?.run_id || '-';
  ui.baselineRisk.textContent = formatPercent(baseline?.summary?.decoder_risk_max, 0);
  ui.baselineAlerts.textContent = String(baseline?.summary?.decoder_alert_count ?? '-');
  ui.baselineTokens.textContent = String(baseline?.summary?.token_count ?? '-');
  renderRunTokens(ui.baselineText, baseline, '-');

  applyRunSnapshot('corrected', corrected);
  ui.correctedRunId.textContent = corrected?.run_id || '-';
  ui.correctedRisk.textContent = formatPercent(corrected?.summary?.decoder_risk_max, 0);
  ui.correctedAlerts.textContent = String(corrected?.summary?.decoder_alert_count ?? '-');
  ui.correctedTokens.textContent = String(corrected?.summary?.token_count ?? '-');
  renderRunTokens(ui.correctedText, corrected, '-');

  const link = new URL('./', window.location.href);
  if (interventionCount > 0) {
    if (baseline?.run_id) link.searchParams.set('runA', corrected?.run_id || baseline.run_id);
    if (corrected?.run_id) link.searchParams.set('runB', baseline?.run_id || corrected.run_id);
  } else {
    if (baseline?.run_id) link.searchParams.set('runA', baseline.run_id);
    if (corrected?.run_id) link.searchParams.set('runB', corrected.run_id);
  }
  link.searchParams.set('diff', '1');
  link.searchParams.set('focus', 'max-risk');
  ui.openMainLink.href = link.toString();

  renderInterventions(result);
  renderHarness(result);
  renderObjectHarness(result);
}

async function probeBackend() {
  const baseUrl = normalizeBaseUrl(ui.endpoint.value);
  if (!baseUrl) {
    setStatus('Endpoint is invalid.', 'error');
    return;
  }
  ui.endpoint.value = baseUrl;
  try {
    setStatus('Probing backend...', 'neutral');
    const response = await apiPost('/api/backend-info', { base_url: baseUrl, refresh: true });
    applyBackendInfo(response?.backend || null);
  } catch (err) {
    setStatus(`Probe failed: ${err.message}`, 'error');
  }
}

async function stopExperiment() {
  if (!state.experimentRunning && !state.currentExperimentId) return;
  try {
    setExperimentRunning(true, { stopping: true });
    setStatus('Stop requested...', 'neutral');
    if (currentLabMode() === 'retraction') {
      await apiPost(`/api/retraction-experiment/${encodeURIComponent(state.currentExperimentId)}/stop`, {});
    } else {
      await apiPost('/api/stop', { experiment_id: state.currentExperimentId });
    }
    setStatus('Stopping after the current backend call returns...', 'neutral');
  } catch (err) {
    setExperimentRunning(state.experimentRunning);
    setStatus(`Stop failed: ${err.message}`, 'error');
  }
}

async function runExperiment() {
  const prompt = ui.prompt.value.trim();
  const baseUrl = normalizeBaseUrl(ui.endpoint.value);
  const mode = currentLabMode();
  if (!prompt) {
    setStatus('Prompt is required.', 'error');
    return;
  }
  if (!baseUrl) {
    setStatus('Endpoint is invalid.', 'error');
    return;
  }

  const settings = {
    seed: asNumber(ui.seed.value, 1234),
    temperature: asNumber(ui.temperature.value, 0.7),
    top_p: asNumber(ui.topP.value, 0.95),
    max_tokens: asNumber(ui.maxTokens.value, 64),
    top_n: 5,
    n_probs: 20,
    vector_mode: ui.vectorMode.value || 'placeholder',
    vector_dim: 24,
    chunk_size: mode === 'retraction' ? Math.max(8, Math.min(asNumber(ui.maxTokens.value, 64), 64)) : 1,
  };

  const policy = {
    risk_threshold: asNumber(ui.riskThreshold.value, 0.64),
    risk_persistence: asNumber(ui.riskPersistence.value, 2),
    rollback_buffer: asNumber(ui.rollbackBuffer.value, 1),
    branch_candidates: asNumber(ui.branchCandidates.value, 3),
    branch_lookahead: asNumber(ui.branchLookahead.value, 6),
    correction_loops: asNumber(ui.maxInterventions.value, 2),
    max_interventions: asNumber(ui.maxInterventions.value, 2),
  };

  try {
    state.currentExperimentId = `exp_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`;
    resetExperimentView();
    setExperimentRunning(true);
    if (mode === 'retraction') {
      setStatus('Running internal consistency retraction loop...', 'neutral');
      const retractionPolicy = {
        commitment_limit: 3,
        probe_samples_per_commitment: 3,
        probe_answer_temperature: 0.8,
        judge_temperature: 0.1,
        max_total_model_calls: 18,
        max_total_duration_ms: 180000,
        max_probe_duration_ms: 60000,
        max_draft_tokens: asNumber(ui.maxTokens.value, 64),
        max_commitment_extraction_tokens: 0,
        max_reconciliation_tokens: Math.max(220, asNumber(ui.maxTokens.value, 64) * 4),
        on_budget_exceeded: 'return_needs_review',
      };
      const result = await apiPost('/api/retraction-experiment', {
        experiment_id: state.currentExperimentId,
        prompt,
        base_url: baseUrl,
        settings,
        policy: retractionPolicy,
      });
      renderRetractionResult(result);
      setStatus(
        result?.status === 'contract_satisfied' ? 'Retraction contract satisfied.' : `Retraction result: ${(result?.status || 'needs_review').replace(/_/g, ' ')}.`,
        result?.status === 'contract_satisfied' ? 'success' : 'neutral',
      );
    } else {
      setStatus('Running baseline and replay/rewrite loops...', 'neutral');
      const result = await apiPost('/api/live-experiment', {
        experiment_id: state.currentExperimentId,
        prompt,
        base_url: baseUrl,
        settings,
        policy,
      });
      renderResult(result);
      finalizeIterationTrace(result);
      setStatus(result?.status === 'stopped' ? 'Experiment stopped.' : 'Experiment completed.', result?.status === 'stopped' ? 'neutral' : 'success');
    }
  } catch (err) {
    setStatus(`Experiment failed: ${err.message}`, 'error');
  } finally {
    setExperimentRunning(false);
  }
}

async function bootstrap() {
  const status = await apiGet('/api/status').catch(() => null);
  if (status?.defaults?.base_url) ui.endpoint.value = String(status.defaults.base_url);
  if (status?.defaults?.vector_mode) ui.vectorMode.value = String(status.defaults.vector_mode);

  ui.runBtn.addEventListener('click', runExperiment);
  ui.stopBtn.addEventListener('click', stopExperiment);
  ui.probeBackendBtn.addEventListener('click', probeBackend);
  ui.labMode.addEventListener('change', () => {
    resetExperimentView();
    ui.experimentMeta.textContent = 'No experiment has been run yet.';
    ui.experimentSummary.textContent = currentLabMode() === 'retraction'
      ? 'Run internal consistency retraction to inspect draft, commitments, probe packets, deterministic scoring, reconciliation, and verifier checks.'
      : 'Run a baseline vs corrected decode to compare decoder-risk trajectory and intervention count.';
    setLabModeVisibility();
    renderBackend();
  });
  ui.endpoint.addEventListener('blur', () => {
    const normalized = normalizeBaseUrl(ui.endpoint.value);
    if (normalized) ui.endpoint.value = normalized;
  });
  window.addEventListener('scroll', hideTokenTooltip, true);
  window.addEventListener('resize', hideTokenTooltip);

  connectStream();
  setLabModeVisibility();
  if (status?.backend_probe?.base_url) {
    applyBackendInfo(status.backend_probe);
  }
  if (!status?.backend_probe?.base_url || status?.backend_probe?.probe_state === 'idle') {
    await probeBackend().catch(() => null);
  }
}

bootstrap();
