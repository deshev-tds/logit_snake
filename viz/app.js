const ui = {
  endpoint: document.getElementById('endpoint-input'),
  prompt: document.getElementById('prompt-input'),
  seed: document.getElementById('seed-input'),
  temperature: document.getElementById('temperature-input'),
  topP: document.getElementById('top-p-input'),
  maxTokens: document.getElementById('max-tokens-input'),
  vectorMode: document.getElementById('vector-mode-select'),
  generateBtn: document.getElementById('generate-btn'),
  probeBackendBtn: document.getElementById('probe-backend-btn'),
  stopBtn: document.getElementById('stop-btn'),
  loadFileBtn: document.getElementById('load-file-btn'),
  exportBtn: document.getElementById('export-btn'),
  fileInput: document.getElementById('run-file-input'),
  status: document.getElementById('status-badge'),

  runASelect: document.getElementById('run-a-select'),
  runBSelect: document.getElementById('run-b-select'),
  projectionMode: document.getElementById('projection-mode-select'),
  diffToggle: document.getElementById('diff-toggle'),
  shiftThreshold: document.getElementById('shift-threshold-input'),
  runMeta: document.getElementById('run-meta'),

  snakeCanvas: document.getElementById('snake-canvas'),
  playBtn: document.getElementById('play-btn'),
  stepBackBtn: document.getElementById('step-back-btn'),
  stepForwardBtn: document.getElementById('step-forward-btn'),
  speedSelect: document.getElementById('speed-select'),
  freezeToggle: document.getElementById('freeze-toggle'),
  timeline: document.getElementById('timeline-input'),
  timelineLabel: document.getElementById('timeline-label'),

  diffAvg: document.getElementById('diff-avg'),
  diffMax: document.getElementById('diff-max'),
  diffFirst: document.getElementById('diff-first'),
  diffAlignment: document.getElementById('diff-alignment'),

  storyRisk: document.getElementById('story-risk'),
  storyRunRisk: document.getElementById('story-run-risk'),
  storyReasons: document.getElementById('story-reasons'),
  storyMeaning: document.getElementById('story-meaning'),

  backendReachable: document.getElementById('backend-reachable'),
  backendModel: document.getElementById('backend-model'),
  backendProbs: document.getElementById('backend-probs'),
  backendEmbeddings: document.getElementById('backend-embeddings'),
  backendModelRequired: document.getElementById('backend-model-required'),
  backendStopSemantics: document.getElementById('backend-stop-semantics'),
  backendTokenForcing: document.getElementById('backend-token-forcing'),
  backendGeometry: document.getElementById('backend-geometry'),

  bookmarkBtn: document.getElementById('bookmark-btn'),
  bookmarkList: document.getElementById('bookmark-list'),
  mutationCount: document.getElementById('mutation-count'),
  mutationList: document.getElementById('mutation-list'),

  tokenStream: document.getElementById('token-stream'),

  metricUncertainty: document.getElementById('metric-uncertainty'),
  metricGap: document.getElementById('metric-gap'),
  metricJump: document.getElementById('metric-jump'),
  metricRepeat: document.getElementById('metric-repeat'),
  metricRisk: document.getElementById('metric-risk'),
  metricVelocity: document.getElementById('metric-velocity'),

  popup: document.getElementById('token-popup'),
  popupTitle: document.getElementById('popup-title'),
  popupMeta: document.getElementById('popup-meta'),
  popupAltList: document.getElementById('popup-alt-list'),
  popupClose: document.getElementById('popup-close'),
  popupConfirm: document.getElementById('popup-confirm'),
};

const RUN_COLORS = {
  a: '#187a90',
  b: '#d9782d',
  marker: '#c2463b',
  axis: '#a8bccf',
  grid: '#dce6f1',
};

const ALT_TOKEN_POOL = [
  ' the',
  ' a',
  ' and',
  ' to',
  ' of',
  ' in',
  ',',
  '.',
  ' that',
  ' is',
  ' with',
  ' for',
  ' on',
  ' as',
  ' by',
  ' it',
  ' this',
  ' from',
  '\n',
];

const state = {
  runs: new Map(),
  runOrder: [],
  runAId: null,
  runBId: null,
  diffEnabled: false,
  currentIndex: 0,
  playing: false,
  freeze: false,
  speed: 1,
  baseRate: 6,
  playCursor: 0,
  lastFrameTs: performance.now(),
  projectionCache: new Map(),
  needsRender: true,
  lastIndex: -1,
  tokenNodes: [],
  mutationIndices: new Set(),
  latestMutationIndex: null,
  popup: null,
  diffContext: null,
  serverRunning: false,
  backendInfo: null,
  projectionMode: '2d',
  view3d: {
    yaw: 0.72,
    pitch: -0.44,
    dragging: false,
    lastX: 0,
    lastY: 0,
  },
};

function clamp(value, minValue, maxValue) {
  return Math.max(minValue, Math.min(maxValue, value));
}

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

function hashString(input) {
  let h = 2166136261 >>> 0;
  const text = String(input ?? '');
  for (let i = 0; i < text.length; i += 1) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function normalizeBaseUrl(rawValue) {
  let raw = String(rawValue ?? '').trim();
  if (!raw) return '';

  // Guard against accidental concatenation while editing the endpoint field.
  const starts = [];
  for (const prefix of ['http://', 'https://']) {
    let idx = raw.indexOf(prefix);
    while (idx !== -1) {
      starts.push(idx);
      idx = raw.indexOf(prefix, idx + 1);
    }
  }
  if (starts.length > 1) {
    raw = raw.slice(Math.max(...starts));
  }

  if (!/^https?:\/\//i.test(raw)) {
    raw = `http://${raw}`;
  }

  try {
    const u = new URL(raw);
    const path = u.pathname && u.pathname !== '/' ? u.pathname : '';
    return `${u.protocol}//${u.host}${path}`.replace(/\/+$/, '');
  } catch {
    return '';
  }
}

function mulberry32(seed) {
  let t = seed >>> 0;
  return function rand() {
    t = (t + 0x6d2b79f5) >>> 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function getRun(runId) {
  if (!runId) return null;
  return state.runs.get(runId) || null;
}

function currentRunA() {
  return getRun(state.runAId);
}

function currentRunB() {
  if (!state.diffEnabled) return null;
  return getRun(state.runBId);
}

function hasTextFocus() {
  const active = document.activeElement;
  if (!active) return false;
  const tag = active.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || active.isContentEditable;
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

async function apiGet(path) {
  const resp = await fetch(path);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || `HTTP ${resp.status}`);
  }
  return resp.json();
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
    const msg = json?.error || text || `HTTP ${resp.status}`;
    throw new Error(msg);
  }
  return json || {};
}

function deterministicTopN(tokenText, tokenId, runSeed, index, n = 5) {
  const count = Math.max(1, n);
  const rng = mulberry32(hashString(`${runSeed}|${index}|${tokenText}|${tokenId ?? 'na'}`));
  const probsTemplate = [0.62, 0.17, 0.1, 0.07, 0.04];
  const probs = probsTemplate.slice(0, count);
  const total = probs.reduce((acc, x) => acc + x, 0) || 1;
  const norm = probs.map((p) => p / total);

  const topn = [
    {
      token_id: tokenId ?? null,
      token_text: tokenText,
      prob: norm[0],
      logprob: Math.log(norm[0]),
    },
  ];

  const used = new Set([tokenText]);
  for (let i = 1; i < count; i += 1) {
    let candidate = ALT_TOKEN_POOL[Math.floor(rng() * ALT_TOKEN_POOL.length)];
    if (used.has(candidate)) {
      candidate = `${tokenText}${i}`;
    }
    used.add(candidate);
    topn.push({
      token_id: null,
      token_text: candidate,
      prob: norm[i],
      logprob: Math.log(norm[i]),
    });
  }
  return topn;
}

function normalizeTopN(rawTopN, token, runSeed, index, topNCount = 5) {
  let topN = [];
  if (Array.isArray(rawTopN)) {
    for (const alt of rawTopN) {
      if (!alt || typeof alt !== 'object') continue;
      const tokenText = alt.token_text ?? alt.token ?? '';
      const tokenId = alt.token_id ?? alt.id ?? null;
      let prob = asNumber(alt.prob);
      let logprob = asNumber(alt.logprob);
      if (logprob == null && prob != null && prob > 0) {
        logprob = Math.log(prob);
      }
      if (prob == null && logprob != null && Number.isFinite(logprob)) {
        prob = Math.exp(logprob);
      }
      if (prob == null) prob = 0;
      topN.push({ token_id: tokenId, token_text: tokenText, prob, logprob });
    }
  }

  if (!topN.length) {
    topN = deterministicTopN(
      token.chosen_token_text ?? token.text ?? '',
      token.chosen_token_id ?? token.token_id ?? null,
      runSeed,
      index,
      topNCount
    );
  }

  topN.sort((a, b) => (b.prob ?? 0) - (a.prob ?? 0));
  return topN.slice(0, topNCount);
}

function entropyFromTopN(topN) {
  if (!Array.isArray(topN) || !topN.length) return null;
  let mass = 0;
  let entropy = 0;
  for (const item of topN) {
    const p = Math.max(0, asNumber(item.prob, 0));
    mass += p;
    if (p > 0) entropy -= p * Math.log(p);
  }
  const rest = Math.max(0, 1 - mass);
  if (rest > 0) entropy -= rest * Math.log(rest);
  return entropy;
}

function marginFromTopN(topN) {
  if (!Array.isArray(topN) || topN.length < 2) return null;
  const lp1 = asNumber(topN[0].logprob);
  const lp2 = asNumber(topN[1].logprob);
  if (lp1 == null || lp2 == null) return null;
  return lp1 - lp2;
}

function observedMassFromTopN(topN) {
  if (!Array.isArray(topN) || !topN.length) return null;
  let mass = 0;
  for (const item of topN) {
    mass += Math.max(0, asNumber(item.prob, 0));
  }
  return clamp(mass, 0, 1);
}

function probGapFromTopN(topN) {
  if (!Array.isArray(topN) || !topN.length) {
    return { top1: null, top2: null, gap: null };
  }
  const top1 = Math.max(0, asNumber(topN[0]?.prob, 0));
  const top2 = Math.max(0, asNumber(topN[1]?.prob, 0));
  return {
    top1,
    top2,
    gap: clamp(top1 - top2, 0, 1),
  };
}

function normalizeRepeatToken(text) {
  const raw = String(text ?? '');
  const compact = raw.replace(/\s+/g, ' ').trim().toLowerCase();
  return compact || raw.trim().toLowerCase() || raw.toLowerCase();
}

function angleBetween(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return null;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na <= 1e-12 || nb <= 1e-12) return null;
  const cos = clamp(dot / (Math.sqrt(na) * Math.sqrt(nb)), -1, 1);
  return Math.acos(cos);
}

function vectorNorm(v) {
  if (!Array.isArray(v) || !v.length) return 0;
  let acc = 0;
  for (let i = 0; i < v.length; i += 1) acc += v[i] * v[i];
  return Math.sqrt(acc);
}

function vectorDistance(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return null;
  let acc = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    acc += d * d;
  }
  return Math.sqrt(acc);
}

function pointDistance(a, b) {
  if (!a || !b) return null;
  const dims = ('z' in a || 'z' in b) ? ['x', 'y', 'z'] : ['x', 'y'];
  let acc = 0;
  for (const key of dims) {
    const da = asNumber(a[key], 0);
    const db = asNumber(b[key], 0);
    acc += (da - db) ** 2;
  }
  return Math.sqrt(acc);
}

function placeholderVector(run, token, index, prevVector = null) {
  const dim = clamp(
    asNumber(run?.meta?.generation_settings?.vector_dim, 24),
    4,
    256
  );
  const runSeed = run?.meta?.prompt_hash || run?.run_id || 'run';
  const tokenSeed = hashString(`${runSeed}|${index}|${token.chosen_token_id ?? 'na'}|${token.text}`);
  const rand = mulberry32(tokenSeed);

  const vec = new Array(dim);
  for (let i = 0; i < dim; i += 1) {
    vec[i] = rand() * 2 - 1;
  }

  vec[0] += asNumber(token.entropy, 0) * 0.04;
  vec[1] += asNumber(token.margin, 0) * 0.04;
  vec[2] += asNumber(token.logprob, 0) * 0.025;

  if (Array.isArray(prevVector) && prevVector.length === dim) {
    for (let i = 0; i < dim; i += 1) {
      vec[i] = 0.72 * prevVector[i] + 0.28 * vec[i];
    }
  }

  const norm = vectorNorm(vec) || 1;
  return vec.map((v) => v / norm);
}

function normalizeToken(raw, run, index, prevVector = null) {
  const token = {
    index,
    t: asNumber(raw?.t, index),
    text: String(raw?.text ?? raw?.chosen_token_text ?? raw?.token_text ?? ''),
    chosen_token_id: raw?.chosen_token_id ?? raw?.token_id ?? null,
    chosen_token_text: String(raw?.chosen_token_text ?? raw?.text ?? raw?.token_text ?? ''),
    logprob: asNumber(raw?.logprob),
    prob: asNumber(raw?.prob),
    entropy: asNumber(raw?.entropy),
    margin: asNumber(raw?.margin),
    topN: [],
    embedding: null,
    velocity: asNumber(raw?.velocity),
    curvature: asNumber(raw?.curvature),
    observed_mass: asNumber(raw?.observed_mass),
    top1_prob: asNumber(raw?.top1_prob),
    top2_prob: asNumber(raw?.top2_prob),
    prob_gap: asNumber(raw?.prob_gap),
    uncertainty: asNumber(raw?.uncertainty),
    entropy_delta: asNumber(raw?.entropy_delta),
    repeat_unigram: asNumber(raw?.repeat_unigram),
    repeat_bigram: asNumber(raw?.repeat_bigram),
    repetition_pressure: asNumber(raw?.repetition_pressure),
    decoder_risk: asNumber(raw?.decoder_risk),
    topn_provider: raw?.topn_provider || 'backend_logprobs',
  };

  const runSeed = run?.meta?.prompt_hash || run?.run_id || 'run';
  token.topN = normalizeTopN(raw?.topN, token, runSeed, index, 5);

  if (token.logprob == null) {
    const chosen = token.topN.find(
      (alt) =>
        (token.chosen_token_id != null && alt.token_id === token.chosen_token_id) ||
        alt.token_text === token.chosen_token_text
    );
    if (chosen) token.logprob = asNumber(chosen.logprob);
  }
  if (token.prob == null) {
    const chosen = token.topN.find(
      (alt) =>
        (token.chosen_token_id != null && alt.token_id === token.chosen_token_id) ||
        alt.token_text === token.chosen_token_text
    );
    if (chosen) token.prob = asNumber(chosen.prob);
  }

  if (token.entropy == null) token.entropy = entropyFromTopN(token.topN);
  if (token.margin == null) token.margin = marginFromTopN(token.topN);

  if (Array.isArray(raw?.embedding) && raw.embedding.length) {
    token.embedding = raw.embedding.map((v) => asNumber(v, 0));
  } else {
    token.embedding = placeholderVector(run, token, index, prevVector);
  }

  return token;
}

function recomputeKinematics(run) {
  const tokens = run.tokens;
  let prevVec = null;
  let prevDelta = null;
  for (let i = 0; i < tokens.length; i += 1) {
    const t = tokens[i];
    if (!Array.isArray(t.embedding) || !t.embedding.length) {
      t.embedding = placeholderVector(run, t, i, prevVec);
    }

    let delta = null;
    if (prevVec) {
      delta = t.embedding.map((x, k) => x - prevVec[k]);
      t.velocity = vectorNorm(delta);
      t.curvature = prevDelta ? angleBetween(prevDelta, delta) : null;
    } else {
      t.velocity = 0;
      t.curvature = null;
    }

    t.index = i;
    prevVec = t.embedding;
    prevDelta = delta;
  }
}

function enrichDecoderDiagnostics(run) {
  const tokens = run.tokens || [];
  let previousEntropy = null;
  const seenNormTokens = [];
  const seenBigrams = [];

  for (let i = 0; i < tokens.length; i += 1) {
    const token = tokens[i];
    const probStats = probGapFromTopN(token.topN);
    if (token.observed_mass == null) token.observed_mass = observedMassFromTopN(token.topN);
    if (token.top1_prob == null) token.top1_prob = probStats.top1;
    if (token.top2_prob == null) token.top2_prob = probStats.top2;
    if (token.prob_gap == null) token.prob_gap = probStats.gap;

    if (token.uncertainty == null && token.entropy != null) {
      const denom = Math.log(Math.max(2, (token.topN?.length || 0) + 1));
      token.uncertainty = denom > 0 ? clamp(token.entropy / denom, 0, 1) : null;
    }

    if (token.entropy_delta == null && token.entropy != null && previousEntropy != null) {
      token.entropy_delta = Math.abs(token.entropy - previousEntropy);
    }
    if (token.entropy != null) previousEntropy = token.entropy;

    const currNorm = normalizeRepeatToken(token.text || token.chosen_token_text || '');
    const lookback = seenNormTokens.slice(-12);
    const unigramHits = lookback.filter((item) => item && item === currNorm).length;
    if (token.repeat_unigram == null) {
      token.repeat_unigram = lookback.length ? unigramHits / lookback.length : 0;
    }

    const prevNorm = seenNormTokens.length ? seenNormTokens[seenNormTokens.length - 1] : '';
    const bigram = (prevNorm || currNorm) ? `${prevNorm}|${currNorm}` : '';
    const bigramHits = seenBigrams.slice(-12).filter((item) => item && item === bigram).length;
    if (token.repeat_bigram == null) {
      token.repeat_bigram = bigramHits > 0 ? 1 : 0;
    }

    if (token.repetition_pressure == null) {
      token.repetition_pressure = clamp(0.6 * (token.repeat_unigram || 0) + 0.4 * (token.repeat_bigram || 0), 0, 1);
    }

    if (token.decoder_risk == null) {
      const uncertainty = asNumber(token.uncertainty);
      const gapRisk = token.prob_gap == null ? null : clamp(1 - token.prob_gap, 0, 1);
      const volatility = token.entropy_delta == null ? null : clamp(token.entropy_delta / 0.75, 0, 1);
      if (uncertainty != null || gapRisk != null || volatility != null) {
        token.decoder_risk = clamp(
          0.38 * (uncertainty || 0)
          + 0.24 * (gapRisk || 0)
          + 0.22 * (volatility || 0)
          + 0.16 * (token.repetition_pressure || 0),
          0,
          1
        );
      }
    }

    seenNormTokens.push(currNorm);
    seenBigrams.push(bigram);
  }
}

function detectRegimeMarkers(run) {
  const tokens = run.tokens;
  if (!tokens || tokens.length < 4) return [];

  const velocities = tokens.map((t) => asNumber(t.velocity, 0));
  const entropySlope = [0];
  for (let i = 1; i < tokens.length; i += 1) {
    const e0 = asNumber(tokens[i - 1].entropy, 0);
    const e1 = asNumber(tokens[i].entropy, 0);
    entropySlope.push(Math.abs(e1 - e0));
  }

  const stats = (arr) => {
    const mean = arr.reduce((acc, x) => acc + x, 0) / arr.length;
    const variance = arr.reduce((acc, x) => acc + (x - mean) ** 2, 0) / Math.max(1, arr.length - 1);
    return { mean, std: Math.sqrt(variance) };
  };

  const vStats = stats(velocities);
  const eStats = stats(entropySlope);
  const vThreshold = vStats.mean + 2 * vStats.std;
  const eThreshold = eStats.mean + 2 * eStats.std;

  const markers = [];
  let last = -10;
  for (let i = 1; i < tokens.length; i += 1) {
    const reasons = [];
    if (velocities[i] > vThreshold) reasons.push('velocity_spike');
    if (entropySlope[i] > eThreshold) reasons.push('entropy_slope_spike');
    if (reasons.length && i - last >= 3) {
      markers.push({ index: i, reasons });
      last = i;
    }
  }
  return markers;
}

function detectDecoderAlerts(run) {
  const tokens = run.tokens || [];
  const alerts = [];
  let last = -10;
  for (let i = 0; i < tokens.length; i += 1) {
    const token = tokens[i];
    const reasons = [];
    if (asNumber(token.uncertainty) != null && token.uncertainty >= 0.72) reasons.push('high_uncertainty');
    if (asNumber(token.prob_gap) != null && token.prob_gap <= 0.12) reasons.push('weak_choice_gap');
    if (asNumber(token.entropy_delta) != null && token.entropy_delta >= 0.38) reasons.push('uncertainty_jump');
    if (asNumber(token.repetition_pressure) != null && token.repetition_pressure >= 0.45) reasons.push('repetition_pressure');
    if (asNumber(token.decoder_risk) != null && token.decoder_risk >= 0.64 && i - last >= 3) {
      alerts.push({ index: i, risk: token.decoder_risk, reasons: reasons.length ? reasons : ['decoder_risk'] });
      last = i;
    }
  }
  return alerts;
}

function finalizeRun(rawRun, source = 'server') {
  const rawMeta = (rawRun?.meta && typeof rawRun.meta === 'object') ? rawRun.meta : {};
  const run = {
    schema_version: String(rawRun?.schema_version ?? '2.0'),
    run_id: String(rawRun?.run_id || `local_${Date.now()}_${Math.floor(Math.random() * 1e6)}`),
    tokens: [],
    bookmarks: Array.isArray(rawRun?.bookmarks) ? rawRun.bookmarks.slice() : [],
    meta: {
      ...rawMeta,
      label: rawMeta.label || 'Run',
      timestamp: rawMeta.timestamp || new Date().toISOString(),
      status: rawMeta.status || 'complete',
      prompt_hash:
        rawMeta.prompt_hash ||
        `${hashString(rawMeta.prompt || rawRun?.run_id || 'run').toString(16)}`,
    },
    analysis: (rawRun?.analysis && typeof rawRun.analysis === 'object') ? rawRun.analysis : {},
    summary: (rawRun?.summary && typeof rawRun.summary === 'object') ? rawRun.summary : {},
    source,
  };
  run.meta.branch_history = normalizeBranchHistory(run.meta);

  const rawTokens = Array.isArray(rawRun?.tokens) ? rawRun.tokens : [];
  let prevVector = null;
  for (let i = 0; i < rawTokens.length; i += 1) {
    const token = normalizeToken(rawTokens[i], run, i, prevVector);
    run.tokens.push(token);
    prevVector = token.embedding;
  }

  recomputeKinematics(run);
  enrichDecoderDiagnostics(run);
  run.analysis.regime_markers = detectRegimeMarkers(run);
  run.analysis.decoder_alerts = detectDecoderAlerts(run);
  run.summary.token_count = run.tokens.length;

  return run;
}

function convertLegacyRecordsToRun(records, fileLabel = 'Imported') {
  const telemetry = records.filter((r) => r && r.type === 'telemetry');
  const sessionStart = records.find((r) => r && r.type === 'session_start');

  telemetry.sort((a, b) => asNumber(a.token_index, 0) - asNumber(b.token_index, 0));

  const tokens = telemetry.map((rec, idx) => {
    const topN = Array.isArray(rec.topN) ? rec.topN : [];
    const embedding = Array.isArray(rec.embedding)
      ? rec.embedding
      : (rec.position && Number.isFinite(rec.position.x) && Number.isFinite(rec.position.y) && Number.isFinite(rec.position.z)
          ? [rec.position.x, rec.position.y, rec.position.z]
          : null);

    return {
      index: idx,
      t: asNumber(rec.t_ms, idx),
      text: rec.token_text || '',
      chosen_token_id: rec.token_id ?? null,
      chosen_token_text: rec.token_text || '',
      logprob: asNumber(rec.logprob),
      prob: asNumber(rec.prob),
      entropy: asNumber(rec.metrics?.entropy),
      margin: asNumber(rec.metrics?.logit_margin),
      topN,
      embedding,
    };
  });

  return {
    schema_version: '2.0',
    run_id: `import_${Date.now()}_${Math.floor(Math.random() * 1e6)}`,
    tokens,
    meta: {
      label: fileLabel,
      prompt: sessionStart?.config?.prompt || '',
      timestamp: new Date().toISOString(),
      status: 'complete',
      generation_settings: {
        max_tokens: tokens.length,
        temperature: sessionStart?.config?.temperature,
        top_p: sessionStart?.config?.top_p,
        seed: sessionStart?.config?.seed,
      },
      model: sessionStart?.model?.model_path || null,
      prompt_hash: sessionStart?.config?.prompt ? `${hashString(sessionStart.config.prompt).toString(16)}` : `${hashString(fileLabel).toString(16)}`,
    },
    analysis: {},
    summary: {},
  };
}

function parseRunContent(text, fileName = 'Imported') {
  const trimmed = text.trim();
  if (!trimmed) return [];

  const parsedRuns = [];

  const tryParseJson = () => {
    try {
      return JSON.parse(trimmed);
    } catch {
      return null;
    }
  };

  const jsonData = tryParseJson();
  if (jsonData != null) {
    if (Array.isArray(jsonData) && jsonData.length && jsonData[0]?.type) {
      parsedRuns.push(convertLegacyRecordsToRun(jsonData, fileName));
      return parsedRuns;
    }
    if (Array.isArray(jsonData) && jsonData.length && jsonData[0]?.tokens) {
      for (const run of jsonData) parsedRuns.push(run);
      return parsedRuns;
    }
    if (jsonData.run) {
      parsedRuns.push(jsonData.run);
      return parsedRuns;
    }
    if (jsonData.tokens) {
      parsedRuns.push(jsonData);
      return parsedRuns;
    }
  }

  const lines = trimmed.split(/\r?\n/).filter((line) => line.trim().length > 0);
  const records = [];
  for (const line of lines) {
    try {
      records.push(JSON.parse(line));
    } catch {
      return [];
    }
  }
  if (records.length) {
    parsedRuns.push(convertLegacyRecordsToRun(records, fileName));
  }
  return parsedRuns;
}

function normalizeBranchHistory(meta) {
  const out = [];
  const raw = meta?.branch_history;
  if (Array.isArray(raw)) {
    for (const item of raw) {
      if (!item || typeof item !== 'object') continue;
      const forkIndex = asNumber(item.fork_index);
      if (!Number.isFinite(forkIndex)) continue;
      out.push({
        parent_run_id: item.parent_run_id ?? null,
        fork_index: Math.max(0, Math.floor(forkIndex)),
        alt_rank: asNumber(item.alt_rank),
        from_token_id: item.from_token_id ?? null,
        from_text: String(item.from_text ?? ''),
        to_token_id: item.to_token_id ?? null,
        to_text: String(item.to_text ?? ''),
        logprob: asNumber(item.logprob),
        prob: asNumber(item.prob),
        timestamp: item.timestamp || null,
      });
    }
  }

  if (!out.length && meta?.branch && typeof meta.branch === 'object') {
    const branch = meta.branch;
    const forkIndex = asNumber(branch.fork_index);
    if (Number.isFinite(forkIndex)) {
      const chosen = (branch.chosen_alt_token && typeof branch.chosen_alt_token === 'object')
        ? branch.chosen_alt_token
        : {};
      out.push({
        parent_run_id: branch.parent_run_id ?? null,
        fork_index: Math.max(0, Math.floor(forkIndex)),
        alt_rank: asNumber(branch.alt_rank),
        from_token_id: null,
        from_text: '',
        to_token_id: chosen.token_id ?? null,
        to_text: String(chosen.token_text ?? ''),
        logprob: asNumber(chosen.logprob),
        prob: asNumber(chosen.prob),
        timestamp: branch.timestamp || null,
      });
    }
  }

  return out;
}

function branchHistoryForRun(run) {
  const history = run?.meta?.branch_history;
  return Array.isArray(history) ? history : [];
}

function mutationIndexSet(run) {
  const set = new Set();
  for (const item of branchHistoryForRun(run)) {
    const forkIndex = asNumber(item.fork_index);
    if (!Number.isFinite(forkIndex)) continue;
    set.add(Math.max(0, Math.floor(forkIndex)));
  }
  return set;
}

function compactToken(text, maxLen = 22) {
  const shown = displayTokenText(String(text ?? ''));
  if (shown.length <= maxLen) return shown;
  return `${shown.slice(0, maxLen - 1)}…`;
}

function compactRunId(runId) {
  const raw = String(runId ?? '');
  if (!raw) return '-';
  if (raw.length <= 16) return raw;
  return `${raw.slice(0, 12)}…`;
}

function updateRunMeta() {
  const run = currentRunA();
  if (!run) {
    ui.runMeta.textContent = 'No run selected.';
    return;
  }

  const meta = run.meta || {};
  const settings = meta.generation_settings || {};
  const history = branchHistoryForRun(run);
  const latest = history.length ? history[history.length - 1] : null;
  const mutationText = history.length ? `mutations=${history.length}` : '';
  const latestText = latest
    ? `latest @${latest.fork_index}: ${compactToken(latest.from_text)} -> ${compactToken(latest.to_text)}`
    : '';
  const providers = meta.providers || {};
  const geometryText = providers.embedding === 'real' ? 'geometry=real' : 'geometry=placeholder';
  const topNText = providers.topN ? `topN=${providers.topN}` : '';

  ui.runMeta.textContent = [
    `${meta.label || 'Run'} (${run.run_id})`,
    `tokens=${run.tokens.length}`,
    geometryText,
    topNText,
    `temp=${formatNum(asNumber(settings.temperature), 2)}`,
    `top_p=${formatNum(asNumber(settings.top_p), 2)}`,
    `seed=${settings.seed ?? '-'}`,
    mutationText,
    latestText,
  ].filter(Boolean).join(' | ');
}

function explainRisk(token) {
  const risk = asNumber(token?.decoder_risk);
  if (risk == null) {
    return {
      label: 'No estimate',
      meaning: 'This position has no decoder-side risk estimate yet.',
    };
  }
  if (risk >= 0.7) {
    return {
      label: 'High',
      meaning: 'The decoder is unstable here. This is a strong place to inspect whether the answer becomes unsupported.',
    };
  }
  if (risk >= 0.45) {
    return {
      label: 'Medium',
      meaning: 'Several decoder-side warning signs are active. Watch for drift, hedging, or abrupt topic shifts.',
    };
  }
  return {
    label: 'Low',
    meaning: 'The decoder looks comparatively settled here. That still does not prove the claim is true.',
  };
}

function reasonLabels(token) {
  if (!token) return [];
  const reasons = [];
  if (asNumber(token.uncertainty) != null && token.uncertainty >= 0.72) reasons.push('many plausible next tokens');
  if (asNumber(token.prob_gap) != null && token.prob_gap <= 0.12) reasons.push('top choices are too close');
  if (asNumber(token.entropy_delta) != null && token.entropy_delta >= 0.38) reasons.push('uncertainty jumped sharply');
  if (asNumber(token.repetition_pressure) != null && token.repetition_pressure >= 0.45) reasons.push('local repetition pressure is high');
  return reasons;
}

function renderAudienceSummary() {
  const run = currentRunA();
  const token = run?.tokens?.[state.currentIndex];
  if (!run || !token) {
    ui.storyRisk.textContent = '-';
    ui.storyRunRisk.textContent = '-';
    ui.storyReasons.textContent = '-';
    ui.storyMeaning.textContent = 'Load or generate a run to get a plain-language explanation.';
    return;
  }

  const risk = explainRisk(token);
  const reasons = reasonLabels(token);
  ui.storyRisk.textContent = `${risk.label} (${formatPercent(token.decoder_risk, 0)})`;
  ui.storyRunRisk.textContent = formatPercent(maxRiskValueForRun(run), 0);
  ui.storyReasons.textContent = reasons.length ? reasons.join(', ') : 'no strong decoder-side warning signs';
  ui.storyMeaning.textContent = risk.meaning;
}

function maxRiskValueForRun(run) {
  const summaryValue = asNumber(run?.summary?.decoder_risk_max);
  if (summaryValue != null) return summaryValue;
  const tokens = Array.isArray(run?.tokens) ? run.tokens : [];
  let best = null;
  for (const token of tokens) {
    const risk = asNumber(token?.decoder_risk);
    if (risk == null) continue;
    if (best == null || risk > best) best = risk;
  }
  return best;
}

function maxRiskIndexForRun(run) {
  const tokens = Array.isArray(run?.tokens) ? run.tokens : [];
  let bestIndex = 0;
  let bestRisk = null;
  for (let i = 0; i < tokens.length; i += 1) {
    const risk = asNumber(tokens[i]?.decoder_risk);
    if (risk == null) continue;
    if (bestRisk == null || risk > bestRisk) {
      bestRisk = risk;
      bestIndex = i;
    }
  }
  return bestIndex;
}

function renderBackendPanel() {
  const info = state.backendInfo;
  const run = currentRunA();
  const providers = run?.meta?.providers || {};
  const selectedGeometry = ui.vectorMode?.value || 'placeholder';
  ui.backendReachable.textContent = info ? (info.reachable ? 'yes' : 'no') : 'not probed';
  ui.backendModel.textContent = info?.model || run?.meta?.model || '-';
  ui.backendProbs.textContent = info ? (info.token_probs ? 'yes' : info.completion ? 'no / weak' : '-') : (providers.topN || '-');
  ui.backendEmbeddings.textContent = info ? (info.embedding ? 'yes' : 'no') : '-';
  ui.backendModelRequired.textContent = info ? capabilityText(info.model_required_explicit) : '-';
  ui.backendStopSemantics.textContent = info ? capabilityText(info.stop_semantics) : '-';
  ui.backendTokenForcing.textContent = info ? capabilityText(info.strict_token_forcing) : '-';
  ui.backendGeometry.textContent = providers.embedding === 'real'
    ? 'real embeddings'
    : providers.embedding === 'placeholder'
      ? 'placeholder vectors'
      : selectedGeometry === 'real'
        ? 'requested real embeddings'
        : 'placeholder vectors';
}

function applyBackendInfo(info, { updateStatus = true } = {}) {
  state.backendInfo = info || null;
  renderBackendPanel();
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

function addOrUpdateRun(run, source = 'server') {
  const normalized = finalizeRun(run, source);
  const existing = state.runs.get(normalized.run_id);
  if (existing && Array.isArray(existing.bookmarks) && existing.bookmarks.length) {
    normalized.bookmarks = existing.bookmarks;
  }

  state.runs.set(normalized.run_id, normalized);
  if (!state.runOrder.includes(normalized.run_id)) {
    state.runOrder.push(normalized.run_id);
  }

  if (!state.runAId) {
    state.runAId = normalized.run_id;
  }

  renderRunSelectors();
  syncTimelineBounds();
  updateRunMeta();
  renderBookmarks();
  renderMutations();
  renderTokens(true);
  state.needsRender = true;
}

function updateRunFromTokenEvent(runId, rawToken) {
  let run = state.runs.get(runId);
  if (!run) {
    run = finalizeRun(
      {
        run_id: runId,
        tokens: [],
        meta: {
          label: 'Live Run',
          status: 'running',
          timestamp: new Date().toISOString(),
          prompt_hash: `${hashString(runId).toString(16)}`,
          generation_settings: { vector_dim: 24 },
        },
      },
      'server'
    );
    state.runs.set(runId, run);
    if (!state.runOrder.includes(runId)) state.runOrder.push(runId);
    if (!state.runAId) state.runAId = runId;
    renderRunSelectors();
  }

  const index = run.tokens.length;
  const prevVector = index > 0 ? run.tokens[index - 1].embedding : null;
  const token = normalizeToken(rawToken, run, index, prevVector);
  run.tokens.push(token);
  recomputeKinematics(run);
  enrichDecoderDiagnostics(run);
  run.analysis.regime_markers = detectRegimeMarkers(run);
  run.analysis.decoder_alerts = detectDecoderAlerts(run);
  run.summary.token_count = run.tokens.length;

  syncTimelineBounds();
  updateRunMeta();
  renderBookmarks();
  renderTokens(true);
  state.needsRender = true;
}

function renderRunSelectors() {
  const runIds = state.runOrder.slice();

  const populate = (selectEl, selectedId, includeBlank = false) => {
    const previous = selectedId;
    selectEl.innerHTML = '';
    if (includeBlank) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = '(none)';
      selectEl.appendChild(opt);
    }
    for (const runId of runIds) {
      const run = state.runs.get(runId);
      if (!run) continue;
      const opt = document.createElement('option');
      opt.value = runId;
      const label = run.meta?.label || 'Run';
      opt.textContent = `${label} • ${runId} (${run.tokens.length})`;
      selectEl.appendChild(opt);
    }
    if (previous && runIds.includes(previous)) {
      selectEl.value = previous;
    } else if (includeBlank) {
      selectEl.value = '';
    } else if (runIds.length > 0) {
      selectEl.value = runIds[0];
    }
  };

  populate(ui.runASelect, state.runAId, false);
  populate(ui.runBSelect, state.runBId, true);

  state.runAId = ui.runASelect.value || null;
  state.runBId = ui.runBSelect.value || null;

  ui.diffToggle.checked = state.diffEnabled;
}

function syncTimelineBounds() {
  const run = currentRunA();
  const max = run ? Math.max(0, run.tokens.length - 1) : 0;
  ui.timeline.max = String(max);
  state.currentIndex = clamp(state.currentIndex, 0, max);
  ui.timeline.value = String(state.currentIndex);
  ui.timelineLabel.textContent = `${state.currentIndex} / ${max}`;
  if (state.lastIndex !== state.currentIndex) {
    highlightActiveToken();
  }
}

function setCurrentIndex(index) {
  const run = currentRunA();
  const max = run ? Math.max(0, run.tokens.length - 1) : 0;
  const next = clamp(index, 0, max);
  if (next === state.currentIndex && state.lastIndex === next) return;

  state.currentIndex = next;
  state.playCursor = next;
  ui.timeline.value = String(next);
  ui.timelineLabel.textContent = `${next} / ${max}`;
  highlightActiveToken();
  state.needsRender = true;
}

function displayTokenText(text) {
  if (!text) return '[empty]';
  if (text === ' ') return '␠';
  return text
    .replace(/\n/g, '↵')
    .replace(/\t/g, '⇥');
}

function buildTokenTitle(token) {
  return [
    `index: ${token.index}`,
    `risk: ${formatPercent(token.decoder_risk, 0)}`,
    `uncertainty: ${formatPercent(token.uncertainty, 0)}`,
    `gap: ${formatPercent(token.prob_gap, 0)}`,
    `jump: ${formatNum(token.entropy_delta, 2)}`,
    `repeat: ${formatPercent(token.repetition_pressure, 0)}`,
  ].join(' | ');
}

function renderTokens(force = false) {
  const run = currentRunA();
  if (!run) {
    if (force) {
      ui.tokenStream.innerHTML = '';
      state.tokenNodes = [];
    }
    state.mutationIndices = new Set();
    state.latestMutationIndex = null;
    return;
  }

  if (!force && state.tokenNodes.length === run.tokens.length) {
    highlightActiveToken();
    return;
  }

  ui.tokenStream.innerHTML = '';
  state.tokenNodes = [];

  const history = branchHistoryForRun(run);
  const mutationIndices = mutationIndexSet(run);
  const latestMutation = history.length ? asNumber(history[history.length - 1]?.fork_index) : null;
  state.mutationIndices = mutationIndices;
  state.latestMutationIndex = Number.isFinite(latestMutation) ? Math.floor(latestMutation) : null;

  for (const token of run.tokens) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'token-chip';
    if (mutationIndices.has(token.index)) {
      btn.classList.add('mutated');
    }
    if (state.latestMutationIndex != null && token.index === state.latestMutationIndex) {
      btn.classList.add('branch');
    }
    btn.dataset.index = String(token.index);
    btn.textContent = displayTokenText(token.text);
    btn.title = buildTokenTitle(token);

    btn.addEventListener('click', () => {
      openTokenPopup(run.run_id, token.index);
      setCurrentIndex(token.index);
    });

    ui.tokenStream.appendChild(btn);
    state.tokenNodes.push(btn);
  }

  highlightActiveToken();
}

function highlightActiveToken() {
  if (!state.tokenNodes.length) return;
  const idx = state.currentIndex;
  if (state.lastIndex >= 0 && state.lastIndex < state.tokenNodes.length) {
    state.tokenNodes[state.lastIndex].classList.remove('active');
  }
  if (idx >= 0 && idx < state.tokenNodes.length) {
    state.tokenNodes[idx].classList.add('active');
    const node = state.tokenNodes[idx];
    const rect = node.getBoundingClientRect();
    const parentRect = ui.tokenStream.getBoundingClientRect();
    if (rect.top < parentRect.top || rect.bottom > parentRect.bottom) {
      node.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    }
  }
  state.lastIndex = idx;
}

function renderBookmarks() {
  const run = currentRunA();
  ui.bookmarkList.innerHTML = '';
  if (!run || !Array.isArray(run.bookmarks) || !run.bookmarks.length) {
    return;
  }

  for (const bm of run.bookmarks) {
    const row = document.createElement('div');
    row.className = 'bookmark-item';

    const label = document.createElement('span');
    label.textContent = `#${bm.index} ${bm.label || 'bookmark'}`;
    row.appendChild(label);

    const jump = document.createElement('button');
    jump.type = 'button';
    jump.textContent = 'Jump';
    jump.addEventListener('click', () => setCurrentIndex(bm.index));
    row.appendChild(jump);

    ui.bookmarkList.appendChild(row);
  }
}

function renderMutations() {
  const run = currentRunA();
  ui.mutationList.innerHTML = '';
  ui.mutationCount.textContent = '0';
  state.mutationIndices = new Set();
  state.latestMutationIndex = null;

  if (!run) return;

  const history = branchHistoryForRun(run);
  ui.mutationCount.textContent = String(history.length);
  state.mutationIndices = mutationIndexSet(run);

  if (!history.length) {
    const empty = document.createElement('div');
    empty.className = 'empty-list';
    empty.textContent = 'No branch mutations on this run.';
    ui.mutationList.appendChild(empty);
    return;
  }

  const latest = history[history.length - 1];
  state.latestMutationIndex = Math.max(0, Math.floor(asNumber(latest?.fork_index, 0)));

  history.forEach((mutation, idx) => {
    const forkIndex = Math.max(0, Math.floor(asNumber(mutation.fork_index, 0)));
    const row = document.createElement('div');
    row.className = 'mutation-item';

    const head = document.createElement('div');
    head.className = 'mutation-head';

    const index = document.createElement('span');
    index.className = 'mutation-index';
    const rankText = Number.isFinite(asNumber(mutation.alt_rank)) ? ` • alt ${Math.floor(asNumber(mutation.alt_rank, 0)) + 1}` : '';
    index.textContent = `M${idx + 1} @${forkIndex}${rankText}`;
    head.appendChild(index);

    const jump = document.createElement('button');
    jump.type = 'button';
    jump.textContent = 'Jump';
    jump.addEventListener('click', () => setCurrentIndex(forkIndex));
    head.appendChild(jump);
    row.appendChild(head);

    const change = document.createElement('div');
    change.className = 'mutation-change';
    const from = document.createElement('span');
    from.className = 'mutation-token from';
    from.textContent = compactToken(mutation.from_text || '(unknown)');
    const arrow = document.createElement('span');
    arrow.className = 'mutation-arrow';
    arrow.textContent = '->';
    const to = document.createElement('span');
    to.className = 'mutation-token to';
    to.textContent = compactToken(mutation.to_text || '(empty)');
    change.appendChild(from);
    change.appendChild(arrow);
    change.appendChild(to);
    row.appendChild(change);

    const parent = document.createElement('div');
    parent.className = 'mutation-parent';
    parent.textContent = `from ${compactRunId(mutation.parent_run_id)}${mutation.timestamp ? ` • ${mutation.timestamp}` : ''}`;
    row.appendChild(parent);

    row.addEventListener('click', (event) => {
      if (event.target instanceof HTMLButtonElement) return;
      setCurrentIndex(forkIndex);
    });

    ui.mutationList.appendChild(row);
  });
}

function addBookmark() {
  const run = currentRunA();
  if (!run) return;
  const label = window.prompt('Bookmark label', `Marker @${state.currentIndex}`);
  if (label == null) return;
  if (!Array.isArray(run.bookmarks)) run.bookmarks = [];
  run.bookmarks.push({
    index: state.currentIndex,
    label: label.trim() || `Marker @${state.currentIndex}`,
    timestamp: new Date().toISOString(),
  });
  renderBookmarks();
  state.needsRender = true;
}

function dot(a, b) {
  let acc = 0;
  for (let i = 0; i < a.length; i += 1) acc += a[i] * b[i];
  return acc;
}

function matVec(mat, vec) {
  const out = new Array(mat.length).fill(0);
  for (let i = 0; i < mat.length; i += 1) {
    let acc = 0;
    for (let j = 0; j < vec.length; j += 1) {
      acc += mat[i][j] * vec[j];
    }
    out[i] = acc;
  }
  return out;
}

function normalizeVec(vec) {
  const n = vectorNorm(vec);
  if (n <= 1e-12) return null;
  return vec.map((x) => x / n);
}

function canonicalizeSign(vec) {
  let idx = 0;
  let maxAbs = 0;
  for (let i = 0; i < vec.length; i += 1) {
    const abs = Math.abs(vec[i]);
    if (abs > maxAbs) {
      maxAbs = abs;
      idx = i;
    }
  }
  if (vec[idx] < 0) return vec.map((x) => -x);
  return vec;
}

function powerIteration(mat, initVec, iterations = 64, orthogonals = []) {
  let v = normalizeVec(initVec.slice());
  if (!v) return null;

  for (let step = 0; step < iterations; step += 1) {
    let next = matVec(mat, v);
    for (const basis of orthogonals) {
      const proj = dot(next, basis);
      for (let i = 0; i < next.length; i += 1) {
        next[i] -= proj * basis[i];
      }
    }
    const normed = normalizeVec(next);
    if (!normed) break;
    v = normed;
  }
  return canonicalizeSign(v);
}

function pca2(vectors) {
  if (!vectors.length) return null;

  const d = vectors[0].length;
  const mean = new Array(d).fill(0);

  for (const v of vectors) {
    for (let i = 0; i < d; i += 1) {
      mean[i] += v[i];
    }
  }
  for (let i = 0; i < d; i += 1) {
    mean[i] /= vectors.length;
  }

  const cov = Array.from({ length: d }, () => new Array(d).fill(0));
  for (const v of vectors) {
    const centered = new Array(d);
    for (let i = 0; i < d; i += 1) centered[i] = v[i] - mean[i];

    for (let i = 0; i < d; i += 1) {
      for (let j = i; j < d; j += 1) {
        cov[i][j] += centered[i] * centered[j];
      }
    }
  }

  const scale = 1 / Math.max(1, vectors.length - 1);
  for (let i = 0; i < d; i += 1) {
    for (let j = i; j < d; j += 1) {
      cov[i][j] *= scale;
      cov[j][i] = cov[i][j];
    }
  }

  const init1 = Array.from({ length: d }, (_, i) => (i % 2 === 0 ? 1 : -1));
  const v1 = powerIteration(cov, init1, 72, []);
  if (!v1) return null;

  const lambda1 = dot(v1, matVec(cov, v1));
  const deflated = Array.from({ length: d }, (_, i) =>
    Array.from({ length: d }, (_, j) => cov[i][j] - lambda1 * v1[i] * v1[j])
  );

  const init2 = Array.from({ length: d }, (_, i) => (i % 3 === 0 ? 1 : 0.4));
  let v2 = powerIteration(deflated, init2, 72, [v1]);
  if (!v2) {
    v2 = new Array(d).fill(0);
    v2[Math.min(1, d - 1)] = 1;
  }

  return { mean, v1, v2 };
}

function pca3(vectors) {
  if (!vectors.length) return null;

  const d = vectors[0].length;
  const mean = new Array(d).fill(0);
  for (const v of vectors) {
    for (let i = 0; i < d; i += 1) mean[i] += v[i];
  }
  for (let i = 0; i < d; i += 1) mean[i] /= vectors.length;

  const cov = Array.from({ length: d }, () => new Array(d).fill(0));
  for (const v of vectors) {
    const centered = new Array(d);
    for (let i = 0; i < d; i += 1) centered[i] = v[i] - mean[i];
    for (let i = 0; i < d; i += 1) {
      for (let j = i; j < d; j += 1) {
        cov[i][j] += centered[i] * centered[j];
      }
    }
  }

  const scale = 1 / Math.max(1, vectors.length - 1);
  for (let i = 0; i < d; i += 1) {
    for (let j = i; j < d; j += 1) {
      cov[i][j] *= scale;
      cov[j][i] = cov[i][j];
    }
  }

  const v1 = powerIteration(cov, Array.from({ length: d }, (_, i) => (i % 2 === 0 ? 1 : -1)), 72, []);
  if (!v1) return null;
  const lambda1 = dot(v1, matVec(cov, v1));
  const deflated1 = Array.from({ length: d }, (_, i) =>
    Array.from({ length: d }, (_, j) => cov[i][j] - lambda1 * v1[i] * v1[j])
  );

  const v2 = powerIteration(deflated1, Array.from({ length: d }, (_, i) => (i % 3 === 0 ? 1 : 0.4)), 72, [v1]);
  if (!v2) return null;
  const lambda2 = dot(v2, matVec(cov, v2));
  const deflated2 = Array.from({ length: d }, (_, i) =>
    Array.from({ length: d }, (_, j) => deflated1[i][j] - lambda2 * v2[i] * v2[j])
  );

  let v3 = powerIteration(deflated2, Array.from({ length: d }, (_, i) => (i % 4 === 0 ? -1 : 0.6)), 72, [v1, v2]);
  if (!v3) {
    v3 = new Array(d).fill(0);
    v3[Math.min(2, d - 1)] = 1;
  }

  return { mean, v1, v2, v3 };
}

function projectPoints(vectors, basis) {
  const points = [];
  for (const v of vectors) {
    const centered = v.map((x, i) => x - basis.mean[i]);
    points.push({
      x: dot(centered, basis.v1),
      y: dot(centered, basis.v2),
    });
  }
  return points;
}

function projectPoints3(vectors, basis) {
  const points = [];
  for (const v of vectors) {
    const centered = v.map((x, i) => x - basis.mean[i]);
    points.push({
      x: dot(centered, basis.v1),
      y: dot(centered, basis.v2),
      z: dot(centered, basis.v3),
    });
  }
  return points;
}

function boundsFromPoints(points) {
  if (!points.length) {
    return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
  }
  let minX = points[0].x;
  let maxX = points[0].x;
  let minY = points[0].y;
  let maxY = points[0].y;

  for (const p of points) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }

  if (Math.abs(maxX - minX) < 1e-9) {
    minX -= 1;
    maxX += 1;
  }
  if (Math.abs(maxY - minY) < 1e-9) {
    minY -= 1;
    maxY += 1;
  }

  return { minX, maxX, minY, maxY };
}

function normalizePoints(points, bounds) {
  return points.map((p) => ({
    x: (p.x - bounds.minX) / (bounds.maxX - bounds.minX),
    y: (p.y - bounds.minY) / (bounds.maxY - bounds.minY),
  }));
}

function boundsFromPoints3(points) {
  if (!points.length) {
    return {
      minX: 0, maxX: 1,
      minY: 0, maxY: 1,
      minZ: 0, maxZ: 1,
    };
  }
  let minX = points[0].x;
  let maxX = points[0].x;
  let minY = points[0].y;
  let maxY = points[0].y;
  let minZ = points[0].z;
  let maxZ = points[0].z;
  for (const p of points) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
    if (p.z < minZ) minZ = p.z;
    if (p.z > maxZ) maxZ = p.z;
  }
  if (Math.abs(maxX - minX) < 1e-9) { minX -= 1; maxX += 1; }
  if (Math.abs(maxY - minY) < 1e-9) { minY -= 1; maxY += 1; }
  if (Math.abs(maxZ - minZ) < 1e-9) { minZ -= 1; maxZ += 1; }
  return { minX, maxX, minY, maxY, minZ, maxZ };
}

function normalizePoints3(points, bounds) {
  return points.map((p) => ({
    x: (p.x - bounds.minX) / (bounds.maxX - bounds.minX),
    y: (p.y - bounds.minY) / (bounds.maxY - bounds.minY),
    z: (p.z - bounds.minZ) / (bounds.maxZ - bounds.minZ),
  }));
}

function projectionKey(runA, runB) {
  if (!runA) return 'empty';
  if (!runB || !state.diffEnabled) {
    return `${state.projectionMode}:single:${runA.run_id}:${runA.tokens.length}`;
  }
  return `${state.projectionMode}:diff:${runA.run_id}:${runA.tokens.length}:${runB.run_id}:${runB.tokens.length}`;
}

function getProjection(runA, runB) {
  const key = projectionKey(runA, runB);
  if (state.projectionCache.has(key)) {
    return state.projectionCache.get(key);
  }

  const vectorsA = runA?.tokens?.map((t) => t.embedding) || [];
  const vectorsB = runB?.tokens?.map((t) => t.embedding) || [];

  const combined = runB && state.diffEnabled ? vectorsA.concat(vectorsB) : vectorsA;
  if (!combined.length) {
    const empty = { pointsA: [], pointsB: [] };
    state.projectionCache.set(key, empty);
    return empty;
  }

  if (state.projectionMode === '3d') {
    const basis3 = pca3(combined);
    if (!basis3) {
      const empty = { pointsA: [], pointsB: [] };
      state.projectionCache.set(key, empty);
      return empty;
    }
    const rawA3 = projectPoints3(vectorsA, basis3);
    const rawB3 = runB && state.diffEnabled ? projectPoints3(vectorsB, basis3) : [];
    const bounds3 = boundsFromPoints3(rawA3.concat(rawB3));
    const result3 = {
      dims: 3,
      pointsA: normalizePoints3(rawA3, bounds3),
      pointsB: normalizePoints3(rawB3, bounds3),
      bounds: bounds3,
      basis: basis3,
    };
    state.projectionCache.set(key, result3);
    return result3;
  }

  const basis = pca2(combined);
  if (!basis) {
    const empty = { pointsA: [], pointsB: [] };
    state.projectionCache.set(key, empty);
    return empty;
  }

  const rawA = projectPoints(vectorsA, basis);
  const rawB = runB && state.diffEnabled ? projectPoints(vectorsB, basis) : [];
  const bounds = boundsFromPoints(rawA.concat(rawB));

  const result = {
    dims: 2,
    pointsA: normalizePoints(rawA, bounds),
    pointsB: normalizePoints(rawB, bounds),
    bounds,
    basis,
  };

  state.projectionCache.set(key, result);
  return result;
}

function alignRuns(runA, runB, pointsA, pointsB) {
  if (!runA || !runB || !pointsA.length || !pointsB.length) {
    return null;
  }

  const lenA = pointsA.length;
  const lenB = pointsB.length;
  const pairs = [];
  const mapAtoB = new Array(lenA).fill(0);

  if (lenA === lenB) {
    for (let i = 0; i < lenA; i += 1) {
      const d = pointDistance(pointsA[i], pointsB[i]) || 0;
      pairs.push({ a: i, b: i, distance: d });
      mapAtoB[i] = i;
    }
    return {
      strategy: 'index',
      pairs,
      mapAtoB,
      ...summarizeAlignment(pairs),
    };
  }

  const windowSize = 6;
  let lastB = 0;
  for (let i = 0; i < lenA; i += 1) {
    const ratio = lenA > 1 ? i / (lenA - 1) : 0;
    const center = Math.round(ratio * Math.max(0, lenB - 1));
    const start = Math.max(lastB, center - windowSize);
    const end = Math.min(lenB - 1, center + windowSize);

    let bestJ = start;
    let bestDist = Infinity;

    for (let j = start; j <= end; j += 1) {
      const dist = pointDistance(pointsA[i], pointsB[j]);
      if (dist != null && dist < bestDist) {
        bestDist = dist;
        bestJ = j;
      }
    }

    lastB = bestJ;
    mapAtoB[i] = bestJ;
    pairs.push({ a: i, b: bestJ, distance: Number.isFinite(bestDist) ? bestDist : 0 });
  }

  return {
    strategy: 'sliding_window_nearest',
    pairs,
    mapAtoB,
    ...summarizeAlignment(pairs),
  };
}

function summarizeAlignment(pairs) {
  if (!pairs.length) {
    return { avgDistance: null, maxDistance: null, maxPair: null, firstShift: null };
  }
  const avgDistance = pairs.reduce((acc, p) => acc + p.distance, 0) / pairs.length;

  let maxPair = pairs[0];
  for (const pair of pairs) {
    if (pair.distance > maxPair.distance) maxPair = pair;
  }

  const threshold = asNumber(ui.shiftThreshold.value, 0.18);
  const firstShift = pairs.find((p) => p.distance > threshold) || null;

  return {
    avgDistance,
    maxDistance: maxPair.distance,
    maxPair,
    firstShift,
  };
}

function computeDiffContext() {
  const runA = currentRunA();
  const runB = currentRunB();
  if (!runA || !runB || !state.diffEnabled) {
    state.diffContext = null;
    return null;
  }

  const proj = getProjection(runA, runB);
  const alignment = alignRuns(runA, runB, proj.pointsA, proj.pointsB);
  state.diffContext = { ...alignment, projection: proj };
  return state.diffContext;
}

function resizeCanvasToDisplaySize(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.floor(canvas.clientWidth * dpr);
  const h = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    return true;
  }
  return false;
}

function toCanvas(point, width, height, pad = 36) {
  const innerW = width - pad * 2;
  const innerH = height - pad * 2;
  return {
    x: pad + point.x * innerW,
    y: height - pad - point.y * innerH,
  };
}

function drawGrid(ctx, width, height, pad = 36) {
  ctx.save();
  ctx.strokeStyle = RUN_COLORS.grid;
  ctx.lineWidth = 1;
  const steps = 6;
  for (let i = 0; i <= steps; i += 1) {
    const x = pad + ((width - pad * 2) * i) / steps;
    ctx.beginPath();
    ctx.moveTo(x, pad);
    ctx.lineTo(x, height - pad);
    ctx.stroke();

    const y = pad + ((height - pad * 2) * i) / steps;
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(width - pad, y);
    ctx.stroke();
  }

  ctx.strokeStyle = RUN_COLORS.axis;
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.rect(pad, pad, width - pad * 2, height - pad * 2);
  ctx.stroke();

  ctx.fillStyle = '#6a7786';
  ctx.font = `${Math.max(11, Math.round(width * 0.012))}px "IBM Plex Mono"`;
  ctx.fillText('PC1', width - pad - 36, height - pad + 20);
  ctx.save();
  ctx.translate(pad - 24, pad + 20);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('PC2', 0, 0);
  ctx.restore();

  ctx.restore();
}

function rotatePoint3(point, yaw, pitch) {
  const cx = point.x - 0.5;
  const cy = point.y - 0.5;
  const cz = point.z - 0.5;

  const cosY = Math.cos(yaw);
  const sinY = Math.sin(yaw);
  const x1 = cx * cosY - cz * sinY;
  const z1 = cx * sinY + cz * cosY;

  const cosX = Math.cos(pitch);
  const sinX = Math.sin(pitch);
  const y2 = cy * cosX - z1 * sinX;
  const z2 = cy * sinX + z1 * cosX;

  return { x: x1, y: y2, z: z2 };
}

function toCanvas3D(point, width, height) {
  const rotated = rotatePoint3(point, state.view3d.yaw, state.view3d.pitch);
  const dist = 1.95;
  const scale = Math.min(width, height) * 0.78;
  const perspective = 1 / (dist - rotated.z);
  return {
    x: width / 2 + rotated.x * scale * perspective,
    y: height / 2 - rotated.y * scale * perspective,
    depth: rotated.z,
  };
}

function drawGrid3D(ctx, width, height) {
  const corners = [
    { x: 0, y: 0, z: 0 }, { x: 1, y: 0, z: 0 }, { x: 1, y: 1, z: 0 }, { x: 0, y: 1, z: 0 },
    { x: 0, y: 0, z: 1 }, { x: 1, y: 0, z: 1 }, { x: 1, y: 1, z: 1 }, { x: 0, y: 1, z: 1 },
  ].map((p) => toCanvas3D(p, width, height));
  const edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];

  ctx.save();
  ctx.strokeStyle = RUN_COLORS.grid;
  ctx.lineWidth = 1;
  for (const [a, b] of edges) {
    ctx.beginPath();
    ctx.moveTo(corners[a].x, corners[a].y);
    ctx.lineTo(corners[b].x, corners[b].y);
    ctx.stroke();
  }

  ctx.fillStyle = '#6a7786';
  ctx.font = `${Math.max(11, Math.round(width * 0.012))}px "IBM Plex Mono"`;
  const axisLabels = [
    { from: { x: 0, y: 0, z: 0 }, to: { x: 1.1, y: 0, z: 0 }, label: 'PC1' },
    { from: { x: 0, y: 0, z: 0 }, to: { x: 0, y: 1.1, z: 0 }, label: 'PC2' },
    { from: { x: 0, y: 0, z: 0 }, to: { x: 0, y: 0, z: 1.1 }, label: 'PC3' },
  ];
  for (const axis of axisLabels) {
    const a = toCanvas3D(axis.from, width, height);
    const b = toCanvas3D(axis.to, width, height);
    ctx.strokeStyle = RUN_COLORS.axis;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    ctx.fillText(axis.label, b.x + 4, b.y + 4);
  }
  ctx.restore();
}

function drawTrajectory(ctx, points, currentIndex, colorHex, markers = []) {
  if (!points.length) return;

  const width = ctx.canvas.width;
  const height = ctx.canvas.height;
  const pad = 36;

  const projected = points.map((p) => toCanvas(p, width, height, pad));

  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  ctx.lineWidth = 1.2;
  ctx.strokeStyle = `${colorHex}44`;
  ctx.beginPath();
  ctx.moveTo(projected[0].x, projected[0].y);
  for (let i = 1; i < projected.length; i += 1) {
    ctx.lineTo(projected[i].x, projected[i].y);
  }
  ctx.stroke();

  const end = clamp(currentIndex, 0, projected.length - 1);
  ctx.lineWidth = 2.8;
  ctx.strokeStyle = colorHex;
  ctx.beginPath();
  ctx.moveTo(projected[0].x, projected[0].y);
  for (let i = 1; i <= end; i += 1) {
    ctx.lineTo(projected[i].x, projected[i].y);
  }
  ctx.stroke();

  for (const marker of markers) {
    if (!marker || marker.index == null || marker.index < 0 || marker.index >= projected.length) continue;
    const p = projected[marker.index];
    ctx.strokeStyle = RUN_COLORS.marker;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(p.x - 5, p.y - 5);
    ctx.lineTo(p.x + 5, p.y + 5);
    ctx.moveTo(p.x + 5, p.y - 5);
    ctx.lineTo(p.x - 5, p.y + 5);
    ctx.stroke();
  }

  const active = projected[end];
  ctx.fillStyle = colorHex;
  ctx.beginPath();
  ctx.arc(active.x, active.y, 5, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.arc(active.x, active.y, 2, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();
}

function drawTrajectory3D(ctx, points, currentIndex, colorHex, markers = []) {
  if (!points.length) return;

  const width = ctx.canvas.width;
  const height = ctx.canvas.height;
  const projected = points.map((p) => toCanvas3D(p, width, height));

  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.lineWidth = 1.2;
  ctx.strokeStyle = `${colorHex}44`;
  ctx.beginPath();
  ctx.moveTo(projected[0].x, projected[0].y);
  for (let i = 1; i < projected.length; i += 1) {
    ctx.lineTo(projected[i].x, projected[i].y);
  }
  ctx.stroke();

  const end = clamp(currentIndex, 0, projected.length - 1);
  ctx.lineWidth = 2.8;
  ctx.strokeStyle = colorHex;
  ctx.beginPath();
  ctx.moveTo(projected[0].x, projected[0].y);
  for (let i = 1; i <= end; i += 1) {
    ctx.lineTo(projected[i].x, projected[i].y);
  }
  ctx.stroke();

  for (const marker of markers) {
    if (!marker || marker.index == null || marker.index < 0 || marker.index >= projected.length) continue;
    const p = projected[marker.index];
    ctx.strokeStyle = RUN_COLORS.marker;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(p.x - 5, p.y - 5);
    ctx.lineTo(p.x + 5, p.y + 5);
    ctx.moveTo(p.x + 5, p.y - 5);
    ctx.lineTo(p.x - 5, p.y + 5);
    ctx.stroke();
  }

  const active = projected[end];
  ctx.fillStyle = colorHex;
  ctx.beginPath();
  ctx.arc(active.x, active.y, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.arc(active.x, active.y, 2, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawDiffConnectors(ctx, pointsA, pointsB, alignment, currentIndex) {
  if (!alignment || !pointsA.length || !pointsB.length) return;
  const width = ctx.canvas.width;
  const height = ctx.canvas.height;
  const pad = 36;

  ctx.save();
  ctx.strokeStyle = 'rgba(112, 126, 144, 0.25)';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 4]);

  for (let i = 0; i < alignment.pairs.length; i += 6) {
    const pair = alignment.pairs[i];
    const a = toCanvas(pointsA[pair.a], width, height, pad);
    const b = toCanvas(pointsB[pair.b], width, height, pad);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  const idxA = clamp(currentIndex, 0, alignment.pairs.length - 1);
  const pair = alignment.pairs[idxA];
  if (pair) {
    const a = toCanvas(pointsA[pair.a], width, height, pad);
    const b = toCanvas(pointsB[pair.b], width, height, pad);
    ctx.setLineDash([]);
    ctx.strokeStyle = 'rgba(194, 70, 59, 0.75)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  ctx.restore();
}

function drawDiffConnectors3D(ctx, pointsA, pointsB, alignment, currentIndex) {
  if (!alignment || !pointsA.length || !pointsB.length) return;
  const width = ctx.canvas.width;
  const height = ctx.canvas.height;

  ctx.save();
  ctx.strokeStyle = 'rgba(112, 126, 144, 0.25)';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 4]);

  for (let i = 0; i < alignment.pairs.length; i += 6) {
    const pair = alignment.pairs[i];
    const a = toCanvas3D(pointsA[pair.a], width, height);
    const b = toCanvas3D(pointsB[pair.b], width, height);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  const idxA = clamp(currentIndex, 0, alignment.pairs.length - 1);
  const pair = alignment.pairs[idxA];
  if (pair) {
    const a = toCanvas3D(pointsA[pair.a], width, height);
    const b = toCanvas3D(pointsB[pair.b], width, height);
    ctx.setLineDash([]);
    ctx.strokeStyle = 'rgba(194, 70, 59, 0.75)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  ctx.restore();
}

function drawSnakeCanvas() {
  const runA = currentRunA();
  const runB = currentRunB();

  resizeCanvasToDisplaySize(ui.snakeCanvas);
  const ctx = ui.snakeCanvas.getContext('2d');
  const width = ui.snakeCanvas.width;
  const height = ui.snakeCanvas.height;

  ctx.clearRect(0, 0, width, height);
  if (state.projectionMode === '3d') {
    drawGrid3D(ctx, width, height);
  } else {
    drawGrid(ctx, width, height);
  }

  if (!runA || !runA.tokens.length) {
    ctx.fillStyle = '#5d7084';
    ctx.font = '14px "IBM Plex Mono"';
    const message =
      runA?.meta?.status === 'error'
        ? `Run failed: ${runA?.meta?.error || runA?.summary?.error || 'unknown error'}`
        : 'No run data loaded.';
    ctx.fillText(message, 48, 54);
    return;
  }

  const projection = getProjection(runA, runB);
  const markersA = runA.analysis?.decoder_alerts || [];

  if (state.projectionMode === '3d') {
    drawTrajectory3D(ctx, projection.pointsA, state.currentIndex, RUN_COLORS.a, markersA);
  } else {
    drawTrajectory(ctx, projection.pointsA, state.currentIndex, RUN_COLORS.a, markersA);
  }

  if (runB && state.diffEnabled) {
    const diff = computeDiffContext();
    const runBMarkers = runB.analysis?.decoder_alerts || [];
    if (state.projectionMode === '3d') {
      drawTrajectory3D(ctx, projection.pointsB, diff?.mapAtoB?.[state.currentIndex] ?? state.currentIndex, RUN_COLORS.b, runBMarkers);
      drawDiffConnectors3D(ctx, projection.pointsA, projection.pointsB, diff, state.currentIndex);
    } else {
      drawTrajectory(ctx, projection.pointsB, diff?.mapAtoB?.[state.currentIndex] ?? state.currentIndex, RUN_COLORS.b, runBMarkers);
      drawDiffConnectors(ctx, projection.pointsA, projection.pointsB, diff, state.currentIndex);
    }
  }
}

function drawMetricGraph(canvas, key, runA, runB, mapAtoB) {
  resizeCanvasToDisplaySize(canvas);
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;

  ctx.clearRect(0, 0, width, height);

  if (!runA || !runA.tokens.length) {
    ctx.fillStyle = '#6a7786';
    ctx.font = '12px "IBM Plex Mono"';
    ctx.fillText('No data', 12, 22);
    return;
  }

  const valuesA = runA.tokens.map((t) => asNumber(t[key], 0));
  const valuesB = runB ? runB.tokens.map((t) => asNumber(t[key], 0)) : [];

  const all = valuesA.concat(valuesB);
  let min = Math.min(...all);
  let max = Math.max(...all);
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    min = 0;
    max = 1;
  }
  if (Math.abs(max - min) < 1e-9) {
    min -= 1;
    max += 1;
  }

  const padX = 28;
  const padY = 16;
  const innerW = width - padX * 2;
  const innerH = height - padY * 2;

  const xAt = (i, len) => padX + (len <= 1 ? 0 : (i / (len - 1)) * innerW);
  const yAt = (value) => padY + (1 - (value - min) / (max - min)) * innerH;

  ctx.strokeStyle = '#dfe8f1';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i += 1) {
    const y = padY + (innerH * i) / 4;
    ctx.moveTo(padX, y);
    ctx.lineTo(width - padX, y);
  }
  ctx.stroke();

  const drawSeries = (values, color) => {
    if (!values.length) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.9;
    ctx.beginPath();
    ctx.moveTo(xAt(0, values.length), yAt(values[0]));
    for (let i = 1; i < values.length; i += 1) {
      ctx.lineTo(xAt(i, values.length), yAt(values[i]));
    }
    ctx.stroke();
  };

  drawSeries(valuesA, RUN_COLORS.a);
  drawSeries(valuesB, RUN_COLORS.b);

  const markers = runA.analysis?.decoder_alerts || [];
  ctx.strokeStyle = 'rgba(194, 70, 59, 0.55)';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  for (const marker of markers) {
    if (marker.index < 0 || marker.index >= valuesA.length) continue;
    const x = xAt(marker.index, valuesA.length);
    ctx.beginPath();
    ctx.moveTo(x, padY);
    ctx.lineTo(x, height - padY);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  const activeA = clamp(state.currentIndex, 0, valuesA.length - 1);
  const activeX = xAt(activeA, valuesA.length);
  ctx.strokeStyle = '#5c6f83';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(activeX, padY);
  ctx.lineTo(activeX, height - padY);
  ctx.stroke();

  if (runB && mapAtoB && mapAtoB.length) {
    const mapped = clamp(mapAtoB[activeA] ?? activeA, 0, valuesB.length - 1);
    const mappedX = xAt(mapped, valuesB.length);
    ctx.strokeStyle = 'rgba(217, 120, 45, 0.6)';
    ctx.beginPath();
    ctx.moveTo(mappedX, padY);
    ctx.lineTo(mappedX, height - padY);
    ctx.stroke();
  }

  ctx.fillStyle = '#6a7786';
  ctx.font = '11px "IBM Plex Mono"';
  ctx.fillText(`min ${formatNum(min, 2)}`, 6, height - 4);
  ctx.fillText(`max ${formatNum(max, 2)}`, 6, 12);
}

function drawMetrics() {
  const runA = currentRunA();
  const runB = currentRunB();
  const diff = runA && runB && state.diffEnabled ? computeDiffContext() : null;

  drawMetricGraph(ui.metricUncertainty, 'uncertainty', runA, runB, diff?.mapAtoB);
  drawMetricGraph(ui.metricGap, 'prob_gap', runA, runB, diff?.mapAtoB);
  drawMetricGraph(ui.metricJump, 'entropy_delta', runA, runB, diff?.mapAtoB);
  drawMetricGraph(ui.metricRepeat, 'repetition_pressure', runA, runB, diff?.mapAtoB);
  drawMetricGraph(ui.metricRisk, 'decoder_risk', runA, runB, diff?.mapAtoB);
  drawMetricGraph(ui.metricVelocity, 'velocity', runA, runB, diff?.mapAtoB);
}

function renderDiffSummary() {
  const runA = currentRunA();
  const runB = currentRunB();
  if (!runA || !runB || !state.diffEnabled) {
    ui.diffAvg.textContent = '-';
    ui.diffMax.textContent = '-';
    ui.diffFirst.textContent = '-';
    ui.diffAlignment.textContent = 'index';
    return;
  }

  const diff = computeDiffContext();
  if (!diff) {
    ui.diffAvg.textContent = '-';
    ui.diffMax.textContent = '-';
    ui.diffFirst.textContent = '-';
    ui.diffAlignment.textContent = '-';
    return;
  }

  ui.diffAvg.textContent = formatNum(diff.avgDistance, 4);
  ui.diffMax.textContent = `${formatNum(diff.maxDistance, 4)} @A${diff.maxPair?.a ?? '-'} / B${diff.maxPair?.b ?? '-'}`;
  ui.diffFirst.textContent = diff.firstShift
    ? `A${diff.firstShift.a} / B${diff.firstShift.b} (d=${formatNum(diff.firstShift.distance, 4)})`
    : 'none';
  ui.diffAlignment.textContent = diff.strategy;
}

function openTokenPopup(runId, tokenIndex) {
  const run = getRun(runId);
  if (!run) return;
  const token = run.tokens[tokenIndex];
  if (!token) return;

  const topN = Array.isArray(token.topN) ? token.topN.slice(0, 5) : [];
  if (!topN.length) return;

  state.popup = {
    runId,
    tokenIndex,
    selectedRank: 0,
  };

  ui.popup.classList.remove('hidden');
  ui.popupTitle.textContent = `Token #${tokenIndex}: ${displayTokenText(token.text)}`;
  ui.popupMeta.textContent = `risk ${formatPercent(token.decoder_risk, 0)} | uncertainty ${formatPercent(token.uncertainty, 0)} | gap ${formatPercent(token.prob_gap, 0)} | repeat ${formatPercent(token.repetition_pressure, 0)}`;

  renderPopupAlternatives();
}

function closeTokenPopup() {
  state.popup = null;
  ui.popup.classList.add('hidden');
}

function renderPopupAlternatives() {
  if (!state.popup) return;
  const run = getRun(state.popup.runId);
  const token = run?.tokens?.[state.popup.tokenIndex];
  if (!run || !token) return;

  const topN = token.topN.slice(0, 5);
  ui.popupAltList.innerHTML = '';

  topN.forEach((alt, rank) => {
    const row = document.createElement('div');
    row.className = 'alt-item';
    if (rank === state.popup.selectedRank) row.classList.add('selected');

    const rankEl = document.createElement('span');
    rankEl.className = 'alt-rank';
    rankEl.textContent = String(rank + 1);

    const tokenEl = document.createElement('span');
    tokenEl.className = 'alt-token';
    tokenEl.textContent = displayTokenText(alt.token_text ?? '');

    const probEl = document.createElement('span');
    probEl.className = 'alt-prob';
    probEl.textContent = `p=${formatNum(asNumber(alt.prob), 4)}`;

    const lpEl = document.createElement('span');
    lpEl.className = 'alt-logprob';
    lpEl.textContent = `lp=${formatNum(asNumber(alt.logprob), 3)}`;

    row.appendChild(rankEl);
    row.appendChild(tokenEl);
    row.appendChild(probEl);
    row.appendChild(lpEl);

    row.addEventListener('click', () => {
      if (!state.popup) return;
      state.popup.selectedRank = rank;
      renderPopupAlternatives();
    });

    ui.popupAltList.appendChild(row);
  });
}

async function createBranchFromPopup() {
  if (!state.popup) return;
  const run = getRun(state.popup.runId);
  if (!run) return;

  const token = run.tokens[state.popup.tokenIndex];
  if (!token || !Array.isArray(token.topN) || !token.topN.length) return;

  const selectedRank = clamp(state.popup.selectedRank, 0, token.topN.length - 1);
  const parentRunId = run.run_id;
  const forkIndex = state.popup.tokenIndex;

  try {
    setStatus('Creating branch...', 'neutral');
    const response = await apiPost('/api/branch', {
      run_id: parentRunId,
      fork_index: forkIndex,
      alt_rank: selectedRank,
      label: `Branch@${forkIndex}`,
    });

    const branchRunId = response.run_id;
    if (branchRunId) {
      await refreshRunsFromServer();

      state.runAId = branchRunId;
      state.runBId = parentRunId;
      state.diffEnabled = true;
      ui.diffToggle.checked = true;

      renderRunSelectors();
      setCurrentIndex(forkIndex);
      renderBookmarks();
      renderMutations();
      renderTokens(true);
      state.needsRender = true;

      setStatus(`Branch started: ${branchRunId}`, 'success');
    }
  } catch (err) {
    setStatus(`Branch failed: ${err.message}`, 'error');
  } finally {
    closeTokenPopup();
  }
}

async function refreshRunsFromServer() {
  try {
    const status = await apiGet('/api/status');
    state.serverRunning = Boolean(status.running);
    const defaults = status?.defaults || {};
    if (defaults.base_url && !ui.endpoint.dataset.hydrated) {
      ui.endpoint.value = String(defaults.base_url);
      ui.endpoint.dataset.hydrated = '1';
    }
    if (defaults.vector_mode && !ui.vectorMode.dataset.hydrated) {
      ui.vectorMode.value = String(defaults.vector_mode);
      ui.vectorMode.dataset.hydrated = '1';
    }

    const data = await apiGet('/api/runs');
    const summaries = Array.isArray(data.runs) ? data.runs : [];

    for (const summary of summaries) {
      if (!summary.run_id) continue;
      if (!state.runOrder.includes(summary.run_id)) state.runOrder.push(summary.run_id);
      if (!state.runs.has(summary.run_id)) {
        try {
          const runResp = await apiGet(`/api/run/${summary.run_id}`);
          if (runResp?.run) addOrUpdateRun(runResp.run, 'server');
        } catch {
          // Ignore transient read errors.
        }
      }
    }

    renderRunSelectors();
    syncTimelineBounds();
    updateRunMeta();
    renderBackendPanel();
    state.needsRender = true;
  } catch {
    // Server can be unavailable while booting; keep UI responsive.
  }
}

function applyUrlSelection() {
  const params = new URLSearchParams(window.location.search);
  const runA = params.get('runA');
  const runB = params.get('runB');
  const diff = params.get('diff');
  const indexParam = params.get('index');
  const focus = params.get('focus');

  let changed = false;
  if (runA && state.runs.has(runA)) {
    state.runAId = runA;
    changed = true;
  }
  if (runB && state.runs.has(runB)) {
    state.runBId = runB;
    changed = true;
  }
  if (diff != null) {
    state.diffEnabled = diff === '1' || diff === 'true';
    changed = true;
  }
  if (focus === 'max-risk') {
    const run = runA && state.runs.has(runA) ? state.runs.get(runA) : currentRunA();
    state.currentIndex = clamp(maxRiskIndexForRun(run), 0, Math.max(0, (run?.tokens?.length || 1) - 1));
    changed = true;
  } else if (indexParam != null) {
    state.currentIndex = asNumber(indexParam, state.currentIndex) ?? state.currentIndex;
    changed = true;
  }

  if (changed) {
    renderRunSelectors();
    renderBookmarks();
    renderMutations();
    renderTokens(true);
    syncTimelineBounds();
    state.needsRender = true;
  }
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
    applyBackendInfo({
      base_url: baseUrl,
      reachable: false,
      completion: false,
      token_probs: false,
      embedding: false,
      strict_token_forcing: 'unknown',
      model_required_explicit: 'unknown',
      stop_semantics: 'unknown',
      probe_state: 'ready',
      model: null,
      error: err.message,
    });
  }
}

function wireSSE() {
  const source = new EventSource('/stream');

  source.onmessage = (event) => {
    let payload;
    try {
      payload = JSON.parse(event.data);
    } catch {
      return;
    }
    if (!payload || !payload.type) return;

    if (payload.type === 'backend_probe_updated' && payload.backend) {
      const endpoint = normalizeBaseUrl(ui.endpoint.value);
      if (!endpoint || payload.backend.base_url === endpoint) {
        applyBackendInfo(payload.backend, { updateStatus: false });
        state.needsRender = true;
      }
      return;
    }

    if (payload.type === 'run_started' && payload.run) {
      addOrUpdateRun(payload.run, 'server');
      if (!state.runAId) state.runAId = payload.run.run_id;
      renderRunSelectors();
      setStatus(`Run started: ${payload.run.run_id}`, 'success');
      state.needsRender = true;
      return;
    }

    if (payload.type === 'token' && payload.run_id && payload.token) {
      updateRunFromTokenEvent(payload.run_id, payload.token);
      if (!state.runAId) {
        state.runAId = payload.run_id;
      }
      state.needsRender = true;
      return;
    }

    if (payload.type === 'run_completed' && payload.run) {
      addOrUpdateRun(payload.run, 'server');
      if (state.runAId === payload.run.run_id) {
        renderTokens(true);
      }
      const runStatus = payload.run?.meta?.status;
      const runError = payload.run?.meta?.error || payload.run?.summary?.error;
      if (runStatus === 'error') {
        setStatus(`Run failed: ${runError || payload.run.run_id}`, 'error');
      } else {
        setStatus(`Run completed: ${payload.run.run_id}`, 'success');
      }
      state.needsRender = true;
      return;
    }

    if (payload.type === 'run_error') {
      setStatus(`Run error: ${payload.error || 'unknown'}`, 'error');
      state.needsRender = true;
      return;
    }

    if (payload.type === 'run_imported' && payload.run) {
      addOrUpdateRun(payload.run, 'server');
      setStatus(`Imported run: ${payload.run.run_id}`, 'success');
      state.needsRender = true;
    }
  };

  source.onerror = () => {
    setStatus('Stream reconnecting...', 'neutral');
  };
}

async function startGeneration() {
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
  ui.endpoint.value = baseUrl;

  const settings = {
    seed: asNumber(ui.seed.value, 1234),
    temperature: asNumber(ui.temperature.value, 0.7),
    top_p: asNumber(ui.topP.value, 0.95),
    max_tokens: asNumber(ui.maxTokens.value, 256),
    top_n: 5,
    n_probs: 20,
    vector_mode: ui.vectorMode.value || 'placeholder',
    vector_dim: 24,
  };

  try {
    state.playing = false;
    ui.playBtn.textContent = 'Play';
    setStatus('Starting generation...', 'neutral');
    const response = await apiPost('/api/generate', {
      prompt,
      base_url: baseUrl,
      settings,
      label: 'Run',
    });

    if (response?.run_id) {
      if (!state.runOrder.includes(response.run_id)) {
        state.runOrder.push(response.run_id);
      }
      if (!state.runs.has(response.run_id)) {
        addOrUpdateRun(
          {
            run_id: response.run_id,
            tokens: [],
            meta: {
              label: 'Run',
              prompt,
              base_url: baseUrl,
              status: 'running',
              timestamp: new Date().toISOString(),
              generation_settings: settings,
              prompt_hash: `${hashString(prompt).toString(16)}`,
            },
          },
          'server'
        );
      }
      state.runAId = response.run_id;
      renderRunSelectors();
    }

    setStatus(`Run started: ${response.run_id}`, 'success');
  } catch (err) {
    setStatus(`Start failed: ${err.message}`, 'error');
  }
}

async function stopGeneration() {
  try {
    await apiPost('/api/stop', {});
    setStatus('Generation stopped.', 'neutral');
  } catch (err) {
    setStatus(`Stop failed: ${err.message}`, 'error');
  }
}

function exportRunA() {
  const run = currentRunA();
  if (!run) {
    setStatus('No Run A to export.', 'error');
    return;
  }

  const payload = JSON.stringify(run, null, 2);
  const blob = new Blob([payload], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${run.run_id}.json`;
  a.click();
  URL.revokeObjectURL(url);
  setStatus(`Exported ${run.run_id}`, 'success');
}

async function tryImportRunToServer(run) {
  try {
    const response = await apiPost('/api/import-run', { run });
    const runResp = await apiGet(`/api/run/${response.run_id}`);
    if (runResp?.run) {
      addOrUpdateRun(runResp.run, 'server');
      return true;
    }
  } catch {
    return false;
  }
  return false;
}

async function loadRunsFromFiles(files) {
  for (const file of files) {
    try {
      const text = await file.text();
      const parsedRuns = parseRunContent(text, file.name);
      if (!parsedRuns.length) {
        setStatus(`No run parsed from ${file.name}`, 'error');
        continue;
      }

      for (const rawRun of parsedRuns) {
        const run = finalizeRun(rawRun, 'file');
        run.meta.label = run.meta.label || file.name;

        const imported = await tryImportRunToServer(run);
        if (!imported) {
          addOrUpdateRun(run, 'file');
        }
      }

      setStatus(`Loaded ${parsedRuns.length} run(s) from ${file.name}`, 'success');
    } catch (err) {
      setStatus(`Failed to load ${file.name}: ${err.message}`, 'error');
    }
  }
}

function playPause() {
  state.playing = !state.playing;
  ui.playBtn.textContent = state.playing ? 'Pause' : 'Play';
  state.needsRender = true;
}

function stepBy(delta) {
  state.playing = false;
  ui.playBtn.textContent = 'Play';
  setCurrentIndex(state.currentIndex + delta);
}

function updatePlayback(ts) {
  const run = currentRunA();
  if (!run || !run.tokens.length) {
    state.lastFrameTs = ts;
    return;
  }

  const elapsedSec = (ts - state.lastFrameTs) / 1000;
  state.lastFrameTs = ts;

  if (!state.playing || state.freeze) return;

  const deltaTokens = elapsedSec * state.baseRate * state.speed;
  if (deltaTokens <= 0) return;

  state.playCursor += deltaTokens;
  setCurrentIndex(Math.floor(state.playCursor));

  const max = run.tokens.length - 1;
  if (state.currentIndex >= max) {
    state.playing = false;
    ui.playBtn.textContent = 'Play';
    state.playCursor = max;
  }
}

function renderAll() {
  updateRunMeta();
  renderAudienceSummary();
  renderBackendPanel();
  renderDiffSummary();
  drawSnakeCanvas();
  drawMetrics();
  syncTimelineBounds();
}

function onFrame(ts) {
  updatePlayback(ts);
  if (state.needsRender || state.playing) {
    renderAll();
    state.needsRender = false;
  }
  requestAnimationFrame(onFrame);
}

function bindUI() {
  ui.generateBtn.addEventListener('click', startGeneration);
  ui.probeBackendBtn.addEventListener('click', probeBackend);
  ui.stopBtn.addEventListener('click', stopGeneration);
  ui.exportBtn.addEventListener('click', exportRunA);

  ui.loadFileBtn.addEventListener('click', () => ui.fileInput.click());
  ui.fileInput.addEventListener('change', async () => {
    if (!ui.fileInput.files?.length) return;
    const files = Array.from(ui.fileInput.files);
    await loadRunsFromFiles(files);
    ui.fileInput.value = '';
  });

  ui.endpoint.addEventListener('blur', () => {
    const normalized = normalizeBaseUrl(ui.endpoint.value);
    if (normalized) {
      ui.endpoint.value = normalized;
    }
  });

  ui.projectionMode.addEventListener('change', () => {
    state.projectionMode = ui.projectionMode.value || '2d';
    state.needsRender = true;
  });

  ui.vectorMode.addEventListener('change', () => {
    renderBackendPanel();
  });

  ui.runASelect.addEventListener('change', async () => {
    state.runAId = ui.runASelect.value || null;
    if (state.runAId && !state.runs.has(state.runAId)) {
      try {
        const resp = await apiGet(`/api/run/${state.runAId}`);
        if (resp?.run) addOrUpdateRun(resp.run, 'server');
      } catch {
        // keep stale selection
      }
    }
    setCurrentIndex(0);
    syncTimelineBounds();
    renderTokens(true);
    renderBookmarks();
    renderMutations();
    closeTokenPopup();
    state.needsRender = true;
  });

  ui.runBSelect.addEventListener('change', async () => {
    state.runBId = ui.runBSelect.value || null;
    if (state.runBId && !state.runs.has(state.runBId)) {
      try {
        const resp = await apiGet(`/api/run/${state.runBId}`);
        if (resp?.run) addOrUpdateRun(resp.run, 'server');
      } catch {
        // keep stale selection
      }
    }
    state.needsRender = true;
  });

  ui.diffToggle.addEventListener('change', () => {
    state.diffEnabled = ui.diffToggle.checked;
    state.needsRender = true;
  });

  ui.shiftThreshold.addEventListener('input', () => {
    state.needsRender = true;
  });

  ui.playBtn.addEventListener('click', playPause);
  ui.stepBackBtn.addEventListener('click', () => stepBy(-1));
  ui.stepForwardBtn.addEventListener('click', () => stepBy(1));

  ui.speedSelect.addEventListener('change', () => {
    state.speed = asNumber(ui.speedSelect.value, 1);
  });

  ui.freezeToggle.addEventListener('change', () => {
    state.freeze = ui.freezeToggle.checked;
    state.needsRender = true;
  });

  ui.timeline.addEventListener('input', () => {
    setCurrentIndex(asNumber(ui.timeline.value, 0));
  });

  ui.snakeCanvas.addEventListener('pointerdown', (event) => {
    if (state.projectionMode !== '3d') return;
    state.view3d.dragging = true;
    state.view3d.lastX = event.clientX;
    state.view3d.lastY = event.clientY;
    ui.snakeCanvas.setPointerCapture(event.pointerId);
  });

  ui.snakeCanvas.addEventListener('pointermove', (event) => {
    if (state.projectionMode !== '3d' || !state.view3d.dragging) return;
    const dx = event.clientX - state.view3d.lastX;
    const dy = event.clientY - state.view3d.lastY;
    state.view3d.lastX = event.clientX;
    state.view3d.lastY = event.clientY;
    state.view3d.yaw += dx * 0.01;
    state.view3d.pitch = clamp(state.view3d.pitch - dy * 0.01, -1.25, 1.25);
    state.needsRender = true;
  });

  const release3dDrag = (event) => {
    if (state.projectionMode !== '3d') return;
    state.view3d.dragging = false;
    if (event?.pointerId != null && ui.snakeCanvas.hasPointerCapture?.(event.pointerId)) {
      ui.snakeCanvas.releasePointerCapture(event.pointerId);
    }
  };
  ui.snakeCanvas.addEventListener('pointerup', release3dDrag);
  ui.snakeCanvas.addEventListener('pointercancel', release3dDrag);

  ui.bookmarkBtn.addEventListener('click', addBookmark);

  ui.popupClose.addEventListener('click', closeTokenPopup);
  ui.popupConfirm.addEventListener('click', createBranchFromPopup);

  window.addEventListener('resize', () => {
    state.needsRender = true;
  });

  document.addEventListener('keydown', (event) => {
    if (state.popup) {
      if (event.key >= '1' && event.key <= '5') {
        const rank = Number(event.key) - 1;
        state.popup.selectedRank = rank;
        renderPopupAlternatives();
        event.preventDefault();
        return;
      }
      if (event.key === 'Enter') {
        createBranchFromPopup();
        event.preventDefault();
        return;
      }
      if (event.key === 'Escape') {
        closeTokenPopup();
        event.preventDefault();
        return;
      }
    }

    if (hasTextFocus()) return;

    if (event.code === 'Space') {
      playPause();
      event.preventDefault();
      return;
    }

    if (event.key === 'ArrowLeft') {
      stepBy(-1);
      event.preventDefault();
      return;
    }

    if (event.key === 'ArrowRight') {
      stepBy(1);
      event.preventDefault();
      return;
    }

    if (event.key.toLowerCase() === 'b') {
      addBookmark();
      event.preventDefault();
      return;
    }

    if (event.key.toLowerCase() === 'd') {
      state.diffEnabled = !state.diffEnabled;
      ui.diffToggle.checked = state.diffEnabled;
      state.needsRender = true;
      event.preventDefault();
    }
  });
}

async function bootstrap() {
  bindUI();
  wireSSE();
  await refreshRunsFromServer();
  applyUrlSelection();

  const status = await apiGet('/api/status').catch(() => null);
  if (status?.defaults?.base_url) ui.endpoint.value = String(status.defaults.base_url);
  if (status?.defaults?.vector_mode) ui.vectorMode.value = String(status.defaults.vector_mode);
  if (status?.backend_probe?.base_url) {
    applyBackendInfo(status.backend_probe);
  } else if (status?.running) {
    setStatus('Server running.', 'success');
  } else {
    setStatus('Idle', 'neutral');
  }
  if (!status?.backend_probe?.base_url || status?.backend_probe?.probe_state === 'idle') {
    await probeBackend().catch(() => null);
  }

  state.lastFrameTs = performance.now();
  requestAnimationFrame(onFrame);
}

bootstrap();
