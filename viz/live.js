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

  correctedRunId: document.getElementById('corrected-run-id'),
  correctedRisk: document.getElementById('corrected-risk'),
  correctedAlerts: document.getElementById('corrected-alerts'),
  correctedTokens: document.getElementById('corrected-tokens'),
  correctedText: document.getElementById('corrected-text'),

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

function displayTokenText(text) {
  const raw = String(text ?? '');
  if (!raw) return '[empty]';
  return raw
    .replace(/ /g, '␠')
    .replace(/\n/g, '↵')
    .replace(/\t/g, '⇥');
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

function renderResult(result) {
  state.lastResult = result;
  const baseline = result?.baseline;
  const corrected = result?.corrected;

  ui.experimentMeta.textContent = `Mode=${result?.mode || '-'} | interventions=${result?.metrics?.intervention_count ?? 0} | baseline=${result?.baseline_run_id || '-'} | corrected=${result?.corrected_run_id || '-'}`;
  ui.experimentSummary.textContent =
    `Baseline max risk ${formatPercent(result?.metrics?.baseline_max_risk, 0)} vs corrected ${formatPercent(result?.metrics?.corrected_max_risk, 0)}. `
    + `Alerts: ${result?.metrics?.baseline_alerts ?? '-'} -> ${result?.metrics?.corrected_alerts ?? '-'}. `
    + `Use "Open In Snake Scope" to inspect both trajectories in the main visualizer.`;

  ui.baselineRunId.textContent = baseline?.run_id || '-';
  ui.baselineRisk.textContent = formatPercent(baseline?.summary?.decoder_risk_max, 0);
  ui.baselineAlerts.textContent = String(baseline?.summary?.decoder_alert_count ?? '-');
  ui.baselineTokens.textContent = String(baseline?.summary?.token_count ?? '-');
  ui.baselineText.textContent = textFromRun(baseline);

  ui.correctedRunId.textContent = corrected?.run_id || '-';
  ui.correctedRisk.textContent = formatPercent(corrected?.summary?.decoder_risk_max, 0);
  ui.correctedAlerts.textContent = String(corrected?.summary?.decoder_alert_count ?? '-');
  ui.correctedTokens.textContent = String(corrected?.summary?.token_count ?? '-');
  ui.correctedText.textContent = textFromRun(corrected);

  const link = new URL('./', window.location.href);
  if (baseline?.run_id) link.searchParams.set('runA', corrected?.run_id || baseline.run_id);
  if (corrected?.run_id) link.searchParams.set('runB', baseline?.run_id || corrected.run_id);
  link.searchParams.set('diff', '1');
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
    setStatus('Running baseline and corrected decode...', 'neutral');
    const result = await apiPost('/api/live-experiment', {
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

  await probeBackend().catch(() => null);
}

bootstrap();
