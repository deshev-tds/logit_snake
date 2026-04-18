const ui = {
  endpoint: document.getElementById('endpoint-input'),
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
  probeBackendBtn: document.getElementById('probe-backend-btn'),
  status: document.getElementById('status-badge'),

  experimentMeta: document.getElementById('experiment-meta'),
  experimentSummary: document.getElementById('experiment-summary'),
  openMainLink: document.getElementById('open-main-link'),

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
  ui.backendMode.textContent = 'hybrid replay / rewrite';
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
  state.loopBudget = 0;
  hideTokenTooltip();
  ui.experimentMeta.textContent = 'Experiment is running...';
  ui.experimentSummary.textContent = 'Waiting for the first token-level updates from the live experiment loop.';
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

    if (payload.type === 'live_experiment_started') {
      state.loopBudget = asNumber(payload.correction_loops, 0) ?? 0;
      state.passLog = ['Pass 1 started from the raw replay path.'];
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
      state.partialInterventions.push(payload);
      renderPartialInterventions();
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
        `Pass ${payload.rewrite_pass ?? state.partialInterventions.length + 1}: accepted ${String(payload.chosen_strategy || 'candidate').replace(/_/g, ' ')} at token ${payload.trigger_index}. `
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
      setStatus(`Accepted rewrite pass ${payload.rewrite_pass ?? 2} at token ${payload.trigger_index}`, 'neutral');
      return;
    }

    if (payload.type === 'live_experiment_completed') {
      renderResult(payload);
      setStatus('Experiment completed.', 'success');
    }
  };

  source.onerror = () => {
    if (state.currentExperimentId) {
      setStatus('Stream reconnecting...', 'neutral');
    }
  };
}

function renderResult(result) {
  state.lastResult = result;
  const baseline = result?.baseline;
  const corrected = result?.corrected;
  const interventionCount = Number(result?.metrics?.intervention_count ?? 0);
  const correctedLabel = interventionCount > 0 ? 'Corrected Candidate' : 'Replay Sample';
  ui.correctedTitle.textContent = correctedLabel;

  if (!state.passLog.length) {
    state.passLog = ['Pass 1 started from the raw replay path.'];
  }
  if (interventionCount > 0) {
    const fromResult = (result?.interventions || []).map((item, idx) =>
      `Pass ${idx + 2}: accepted ${String(item.chosen_strategy || 'candidate').replace(/_/g, ' ')} at token ${item.trigger_index}. `
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
  if (interventionCount > 0) {
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

async function runExperiment() {
  const prompt = ui.prompt.value.trim();
  const baseUrl = normalizeBaseUrl(ui.endpoint.value);
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
    chunk_size: 1,
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
    setStatus('Running baseline and replay/rewrite loops...', 'neutral');
    const result = await apiPost('/api/live-experiment', {
      experiment_id: state.currentExperimentId,
      prompt,
      base_url: baseUrl,
      settings,
      policy,
    });
    renderResult(result);
    setStatus('Experiment completed.', 'success');
  } catch (err) {
    setStatus(`Experiment failed: ${err.message}`, 'error');
  }
}

async function bootstrap() {
  const status = await apiGet('/api/status').catch(() => null);
  if (status?.defaults?.base_url) ui.endpoint.value = String(status.defaults.base_url);
  if (status?.defaults?.vector_mode) ui.vectorMode.value = String(status.defaults.vector_mode);

  ui.runBtn.addEventListener('click', runExperiment);
  ui.probeBackendBtn.addEventListener('click', probeBackend);
  ui.endpoint.addEventListener('blur', () => {
    const normalized = normalizeBaseUrl(ui.endpoint.value);
    if (normalized) ui.endpoint.value = normalized;
  });
  window.addEventListener('scroll', hideTokenTooltip, true);
  window.addEventListener('resize', hideTokenTooltip);

  connectStream();
  if (status?.backend_probe?.base_url) {
    applyBackendInfo(status.backend_probe);
  }
  if (!status?.backend_probe?.base_url || status?.backend_probe?.probe_state === 'idle') {
    await probeBackend().catch(() => null);
  }
}

bootstrap();
