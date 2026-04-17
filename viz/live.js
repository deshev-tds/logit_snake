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
  correctedText: document.getElementById('corrected-text'),
  tokenTooltip: document.getElementById('live-token-tooltip'),

  backendReachable: document.getElementById('backend-reachable'),
  backendModel: document.getElementById('backend-model'),
  backendProbs: document.getElementById('backend-probs'),
  backendEmbeddings: document.getElementById('backend-embeddings'),
  backendMode: document.getElementById('backend-mode'),

  interventionCount: document.getElementById('intervention-count'),
  interventionList: document.getElementById('intervention-list'),
};

const state = {
  backendInfo: null,
  lastResult: null,
  currentExperimentId: null,
  partialInterventions: [],
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

function renderBackend() {
  const info = state.backendInfo;
  ui.backendReachable.textContent = info ? (info.reachable ? 'yes' : 'no') : 'not probed';
  ui.backendModel.textContent = info?.model || '-';
  ui.backendProbs.textContent = info ? (info.token_probs ? 'yes' : info.completion ? 'no / weak' : '-') : '-';
  ui.backendEmbeddings.textContent = info ? (info.embedding ? 'yes' : 'no') : '-';
  ui.backendMode.textContent = 'prefix_replay';
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
    summary.textContent = `baseline avg risk ${formatPercent(item.baseline_avg_risk, 0)} -> chosen ${formatPercent(item.chosen_avg_risk, 0)} via alt ${item.chosen_alt_rank + 1} (${displayTokenText(item.chosen_alt_text)})`;
    block.appendChild(summary);

    const candidates = document.createElement('div');
    candidates.className = 'text-block';
    candidates.textContent = (item.candidates || [])
      .map((candidate) => {
        const accepted = candidate.accepted ? 'accepted' : 'candidate';
        return `#${candidate.alt_rank + 1} ${accepted} | ${displayTokenText(candidate.alt_token_text)} | avg risk ${formatPercent(candidate.avg_risk, 0)} | max ${formatPercent(candidate.max_risk, 0)}`;
      })
      .join('\n');
    block.appendChild(candidates);

    ui.interventionList.appendChild(block);
  });
}

function resetExperimentView() {
  state.lastResult = null;
  state.partialInterventions = [];
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
  ui.correctedText.classList.remove('tokenized-output');
  ui.correctedText.textContent = '-';
  ui.interventionCount.textContent = '0';
  ui.interventionList.innerHTML = '<div class=\"empty-list\">No intervention has been triggered yet.</div>';
}

function renderPartialInterventions() {
  renderInterventions({ interventions: state.partialInterventions });
}

function updateProgress(event) {
  const phase = event?.phase;
  if (phase === 'baseline') {
    ui.baselineRunId.textContent = event.run_id || ui.baselineRunId.textContent;
    ui.baselineTokens.textContent = String(event.token_count ?? ui.baselineTokens.textContent);
    ui.baselineRisk.textContent = formatPercent(event.decoder_risk, 0);
    ui.baselineText.textContent = event.text || ui.baselineText.textContent;
    setStatus(`Baseline token ${event.token_index + 1}`, 'neutral');
  } else if (phase === 'corrected') {
    ui.correctedRunId.textContent = event.run_id || ui.correctedRunId.textContent;
    ui.correctedTokens.textContent = String(event.token_count ?? ui.correctedTokens.textContent);
    ui.correctedRisk.textContent = formatPercent(event.decoder_risk, 0);
    ui.correctedText.textContent = event.text || ui.correctedText.textContent;
    setStatus(`Corrected token ${event.token_index + 1}`, 'neutral');
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
    if (!payload.experiment_id || payload.experiment_id !== state.currentExperimentId) return;

    if (payload.type === 'live_experiment_started') {
      ui.baselineRunId.textContent = payload.baseline_run_id || '-';
      ui.correctedRunId.textContent = payload.corrected_run_id || '-';
      ui.experimentMeta.textContent = `Mode=${payload.mode || 'prefix_replay'} | baseline=${payload.baseline_run_id || '-'} | corrected=${payload.corrected_run_id || '-'}`;
      return;
    }

    if (payload.type === 'live_experiment_progress') {
      updateProgress(payload);
      return;
    }

    if (payload.type === 'live_experiment_intervention') {
      state.partialInterventions.push(payload);
      renderPartialInterventions();
      if (payload.corrected_text) {
        ui.correctedText.textContent = payload.corrected_text;
      }
      setStatus(`Intervention at token ${payload.trigger_index}`, 'neutral');
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
  const correctedLabel = interventionCount > 0 ? 'Corrected' : 'Replay Sample';
  ui.correctedTitle.textContent = correctedLabel;

  ui.experimentMeta.textContent = `Mode=${result?.mode || '-'} | interventions=${result?.metrics?.intervention_count ?? 0} | baseline=${result?.baseline_run_id || '-'} | corrected=${result?.corrected_run_id || '-'}`;
  if (interventionCount > 0) {
    ui.experimentSummary.textContent =
      `Baseline max risk ${formatPercent(result?.metrics?.baseline_max_risk, 0)} vs corrected ${formatPercent(result?.metrics?.corrected_max_risk, 0)}. `
      + `Alerts: ${result?.metrics?.baseline_alerts ?? '-'} -> ${result?.metrics?.corrected_alerts ?? '-'}. `
      + `Use "Open In Snake Scope" to inspect both trajectories in the main visualizer.`;
  } else {
    ui.experimentSummary.textContent =
      `No intervention fired. Baseline max risk ${formatPercent(result?.metrics?.baseline_max_risk, 0)} and replay max risk ${formatPercent(result?.metrics?.corrected_max_risk, 0)} are two matched decode samples, not evidence that a correction policy helped or hurt. `
      + `Use "Open In Snake Scope" to inspect both trajectories in the main visualizer.`;
  }

  ui.baselineRunId.textContent = baseline?.run_id || '-';
  ui.baselineRisk.textContent = formatPercent(baseline?.summary?.decoder_risk_max, 0);
  ui.baselineAlerts.textContent = String(baseline?.summary?.decoder_alert_count ?? '-');
  ui.baselineTokens.textContent = String(baseline?.summary?.token_count ?? '-');
  renderRunTokens(ui.baselineText, baseline, '-');

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
    const response = await apiPost('/api/backend-info', { base_url: baseUrl });
    state.backendInfo = response?.backend || null;
    renderBackend();
    setStatus(state.backendInfo?.reachable ? `Backend ready: ${state.backendInfo.model || baseUrl}` : 'Backend probe failed.', state.backendInfo?.reachable ? 'success' : 'error');
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
    max_interventions: asNumber(ui.maxInterventions.value, 2),
  };

  try {
    state.currentExperimentId = `exp_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`;
    resetExperimentView();
    setStatus('Running baseline and corrected decode...', 'neutral');
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
  await probeBackend().catch(() => null);
}

bootstrap();
