import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

const params = new URLSearchParams(window.location.search);
const TOKEN_RATE = Number(params.get('rate')) || 6;
const SPACE_SCALE = Number(params.get('scale')) || 10;
const PATH_SCALE = Number(params.get('pathScale')) || 2.2;
const AXIS_SCALE = Number(params.get('axisScale')) || 0.6;
const MAX_POINTS = Number(params.get('maxPoints')) || 2000;
const BLOOM_STRENGTH = Number(params.get('bloomStrength')) || 2.2;
const BLOOM_RADIUS = Number(params.get('bloomRadius')) || 0.65;
const BLOOM_THRESHOLD = Number(params.get('bloomThreshold')) || 0.55;
const SHOW_SPRITES = params.get('labels') === '1';
const SHOW_HEAD_LABEL = params.get('headLabel') !== '0';
const THINK_FILTER = params.get('think') !== '0';
const SHOW_ZONES = params.get('zones') !== '0';
const TOKEN_PERIOD_MS = 1000 / TOKEN_RATE;

const HUD = {
  token: document.getElementById('hud-token'),
  entropy: document.getElementById('hud-entropy'),
  margin: document.getElementById('hud-margin'),
  drift: document.getElementById('hud-drift'),
  topk: document.getElementById('hud-topk'),
  rate: document.getElementById('hud-rate'),
  status: document.getElementById('hud-status'),
  endpoint: document.getElementById('endpoint-input'),
  prompt: document.getElementById('prompt-input'),
  start: document.getElementById('start-btn'),
  stop: document.getElementById('stop-btn'),
};
HUD.rate.textContent = `${TOKEN_RATE.toFixed(1)} t/s`;

const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
renderer.autoClear = false;

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x05060a, 6, 38);

const worldGroup = new THREE.Group();
scene.add(worldGroup);

const axisGroup = new THREE.Group();
worldGroup.add(axisGroup);
const zoneGroup = new THREE.Group();
worldGroup.add(zoneGroup);

const overlayScene = new THREE.Scene();
const overlayGroup = new THREE.Group();
overlayScene.add(overlayGroup);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.01, 500);
camera.position.set(4, 4, 7);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.rotateSpeed = 0.7;
controls.zoomSpeed = 0.8;
controls.panSpeed = 0.6;
controls.enableRotate = true;
controls.enableZoom = true;
controls.enablePan = true;
controls.minPolarAngle = 0;
controls.maxPolarAngle = Math.PI;
controls.minAzimuthAngle = -Infinity;
controls.maxAzimuthAngle = Infinity;
controls.minDistance = 0.2;
controls.maxDistance = 250;
controls.mouseButtons = {
  LEFT: THREE.MOUSE.ROTATE,
  MIDDLE: THREE.MOUSE.DOLLY,
  RIGHT: THREE.MOUSE.PAN,
};
controls.touches = {
  ONE: THREE.TOUCH.ROTATE,
  TWO: THREE.TOUCH.DOLLY_PAN,
};

const renderScene = new RenderPass(scene, camera);
const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(window.innerWidth, window.innerHeight),
  BLOOM_STRENGTH,
  BLOOM_RADIUS,
  BLOOM_THRESHOLD
);
const composer = new EffectComposer(renderer);
composer.addPass(renderScene);
composer.addPass(bloomPass);

const geometry = new THREE.BufferGeometry();
geometry.setAttribute('position', new THREE.Float32BufferAttribute([], 3));
geometry.setAttribute('color', new THREE.Float32BufferAttribute([], 3));

const lineMaterial = new THREE.LineBasicMaterial({
  vertexColors: true,
  transparent: true,
  opacity: 0.95,
  blending: THREE.AdditiveBlending,
  depthWrite: false,
});

const line = new THREE.Line(geometry, lineMaterial);
worldGroup.add(line);

const pointMaterial = new THREE.PointsMaterial({
  size: 0.12,
  vertexColors: true,
  transparent: true,
  opacity: 0.85,
  blending: THREE.AdditiveBlending,
  depthWrite: false,
});

const pointsMesh = new THREE.Points(geometry, pointMaterial);
worldGroup.add(pointsMesh);

const points = [];
const colors = [];
const sprites = [];
const axisSprites = [];
let headSprite = null;

buildAxes();
buildZones();
controls.target.set(0, 0, 0);
controls.update();

const pending = [];
let lastScheduledTime = performance.now();
let lastTokenIndex = null;

let entropyMax = 8.0;
let marginMax = 10.0;
let driftMax = 2.0;
const transcriptEl = document.getElementById('transcript');
const MAX_TRANSCRIPT_CHARS = 12000;
const THINK_GUARD = 20;
const thinkStateTokens = { active: false, buffer: '' };
const thinkStateTranscript = { active: false, buffer: '' };


function setStatus(text, tone = 'info') {
  if (!HUD.status) return;
  HUD.status.textContent = text;
  if (tone === 'error') {
    HUD.status.style.color = '#ff6b6b';
  } else if (tone === 'success') {
    HUD.status.style.color = '#57f0d5';
  } else {
    HUD.status.style.color = '#9fb2c7';
  }
}

function disposeSprite(sprite) {
  if (!sprite) return;
  if (sprite.material && sprite.material.map) {
    sprite.material.map.dispose();
  }
  if (sprite.material) {
    sprite.material.dispose();
  }
}

function resetScene() {
  for (const sprite of sprites) {
    overlayGroup.remove(sprite);
    disposeSprite(sprite);
  }
  sprites.length = 0;
  points.length = 0;
  colors.length = 0;
  pending.length = 0;
  geometry.setFromPoints(points);
  geometry.setAttribute('color', new THREE.Float32BufferAttribute([], 3));
  geometry.attributes.position.needsUpdate = true;
  geometry.attributes.color.needsUpdate = true;
  lastScheduledTime = performance.now();
  lastTokenIndex = null;
  thinkStateTokens.active = false;
  thinkStateTokens.buffer = '';
  thinkStateTranscript.active = false;
  thinkStateTranscript.buffer = '';
  if (headSprite) {
    overlayGroup.remove(headSprite);
    disposeSprite(headSprite);
    headSprite = null;
  }
  if (transcriptEl) {
    transcriptEl.textContent = '';
  }
  setStatus('Idle');
}

function processThink(rawText, state) {
  if (!THINK_FILTER || rawText == null || rawText === '') {
    return rawText || '';
  }
  state.buffer += rawText;
  let output = '';

  const openTags = [
    '<think>',
    '<analysis>',
    '<reasoning>',
    '<|think|>',
    '<|analysis|>',
    '<|reasoning|>',
  ];
  const closeTags = [
    '</think>',
    '</analysis>',
    '</reasoning>',
    '<|end|>',
    '<|end_think|>',
    '<|end_of_thought|>',
    '<|end_of_reasoning|>',
    '<|end_of_analysis|>',
  ];

  const findNextTag = (tags) => {
    let best = -1;
    let tag = null;
    for (const t of tags) {
      const idx = state.buffer.toLowerCase().indexOf(t);
      if (idx !== -1 && (best === -1 || idx < best)) {
        best = idx;
        tag = t;
      }
    }
    return { idx: best, tag };
  };

  while (state.buffer.length) {
    if (state.active) {
      const { idx, tag } = findNextTag(closeTags);
      if (idx === -1) {
        if (state.buffer.length > THINK_GUARD) {
          state.buffer = state.buffer.slice(-THINK_GUARD);
        }
        return output;
      }
      state.active = false;
      state.buffer = state.buffer.slice(idx + tag.length);
      continue;
    }

    const { idx, tag } = findNextTag(openTags);
    if (idx === -1) {
      if (state.buffer.length > THINK_GUARD) {
        output += state.buffer.slice(0, -THINK_GUARD);
        state.buffer = state.buffer.slice(-THINK_GUARD);
      }
      return output;
    }
    output += state.buffer.slice(0, idx);
    state.active = true;
    state.buffer = state.buffer.slice(idx + tag.length);
  }

  return output;
}

function appendToken(tokenText) {
  if (!transcriptEl || tokenText == null) return;
  transcriptEl.textContent += tokenText;
  if (transcriptEl.textContent.length > MAX_TRANSCRIPT_CHARS) {
    transcriptEl.textContent = transcriptEl.textContent.slice(-MAX_TRANSCRIPT_CHARS);
  }
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function clamp01(value) {
  if (!Number.isFinite(value)) return 0;
  if (value <= 0) return 0;
  if (value >= 1) return 1;
  return value;
}

function normalize(value, maxValue) {
  if (!Number.isFinite(value) || !Number.isFinite(maxValue) || maxValue <= 0) return 0;
  return clamp01(value / maxValue);
}

function colorFromMetrics(metrics) {
  const entropy = normalize(metrics?.entropy ?? 0, entropyMax);
  const margin = normalize(metrics?.logit_margin ?? 0, marginMax);
  const hue = 0.62 - entropy * 0.55;
  const sat = 0.5 + margin * 0.4;
  const light = 0.45 + 0.15 * (1 - entropy);
  const color = new THREE.Color();
  color.setHSL(hue, sat, light);
  return color;
}

function makeTextSprite(text, options = {}) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const fontSize = options.fontSize || 32;
  ctx.font = `600 ${fontSize}px "Space Grotesk", sans-serif`;
  const paddingX = options.paddingX || 20;
  const paddingY = options.paddingY || 12;
  const metrics = ctx.measureText(text);
  const width = Math.ceil(metrics.width + paddingX * 2);
  const height = Math.ceil(fontSize + paddingY * 2);
  canvas.width = width;
  canvas.height = height;

  ctx.font = `600 ${fontSize}px "Space Grotesk", sans-serif`;
  ctx.fillStyle = 'rgba(6, 10, 16, 0.75)';
  ctx.strokeStyle = 'rgba(87, 240, 213, 0.6)';
  ctx.lineWidth = 3;
  const radius = options.radius || 14;
  const x = 6;
  const y = 6;
  const w = width - 12;
  const h = height - 12;
  ctx.beginPath();
  if (ctx.roundRect) {
    ctx.roundRect(x, y, w, h, radius);
  } else {
    const r = Math.min(radius, w / 2, h / 2);
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
  }
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = '#e8f1ff';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, paddingX, height / 2 + 2);

  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;

  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: false,
  });

  const sprite = new THREE.Sprite(material);
  sprite.renderOrder = 999;
  sprite.userData.aspect = width / height;
  sprite.userData.baseHeight = options.baseHeight || 0.12;
  sprite.userData.maxHeight = options.maxHeight || 0.6;
  return sprite;
}

function makeAxisLabel(text, color = '#d7e9ff') {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const fontSize = 20;
  ctx.font = `600 ${fontSize}px "Space Grotesk", sans-serif`;
  const paddingX = 18;
  const paddingY = 10;
  const metrics = ctx.measureText(text);
  const width = Math.ceil(metrics.width + paddingX * 2);
  const height = Math.ceil(fontSize + paddingY * 2);
  canvas.width = width;
  canvas.height = height;

  ctx.font = `600 ${fontSize}px "Space Grotesk", sans-serif`;
  ctx.fillStyle = 'rgba(6, 10, 16, 0.55)';
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.rect(4, 4, width - 8, height - 8);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = color;
  ctx.textBaseline = 'middle';
  ctx.fillText(text, paddingX, height / 2 + 2);

  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: false,
  });
  const sprite = new THREE.Sprite(material);
  sprite.renderOrder = 998;
  sprite.userData.aspect = width / height;
  sprite.userData.baseHeight = 0.1;
  sprite.userData.maxHeight = 0.45;
  return sprite;
}

function buildAxes() {
  axisGroup.clear();
  for (const sprite of axisSprites) {
    overlayGroup.remove(sprite);
    disposeSprite(sprite);
  }
  axisSprites.length = 0;
  const axisLength = (SPACE_SCALE * PATH_SCALE * AXIS_SCALE) / 2;
  const axisOpacity = 0.65;
  const xColor = 0x57f0d5;
  const yColor = 0xffb347;
  const zColor = 0x7ad1ff;

  const makeAxisLine = (dir, color) => {
    const geom = new THREE.BufferGeometry().setFromPoints([
      dir.clone().multiplyScalar(-axisLength),
      dir.clone().multiplyScalar(axisLength),
    ]);
    const mat = new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity: axisOpacity,
    });
    return new THREE.Line(geom, mat);
  };

  axisGroup.add(makeAxisLine(new THREE.Vector3(1, 0, 0), xColor));
  axisGroup.add(makeAxisLine(new THREE.Vector3(0, 1, 0), yColor));
  axisGroup.add(makeAxisLine(new THREE.Vector3(0, 0, 1), zColor));

  const xLabel = makeAxisLabel('Entropy ↑', '#9ff5e6');
  xLabel.position.set(axisLength + 0.3, 0.0, 0.0);
  axisSprites.push(xLabel);
  overlayGroup.add(xLabel);

  const yLabel = makeAxisLabel('Margin ↑', '#ffd19a');
  yLabel.position.set(0.0, axisLength + 0.3, 0.0);
  axisSprites.push(yLabel);
  overlayGroup.add(yLabel);

  const zLabel = makeAxisLabel('Drift ↑', '#bfe6ff');
  zLabel.position.set(0.0, 0.0, axisLength + 0.3);
  axisSprites.push(zLabel);
  overlayGroup.add(zLabel);

  const grid = new THREE.GridHelper(SPACE_SCALE * PATH_SCALE * AXIS_SCALE, 10, 0x1a2b3f, 0x0f1724);
  grid.material.opacity = 0.35;
  grid.material.transparent = true;
  axisGroup.add(grid);

  const boxSize = SPACE_SCALE * PATH_SCALE * AXIS_SCALE;
  const boxGeom = new THREE.BoxGeometry(boxSize, boxSize, boxSize);
  const boxMat = new THREE.LineBasicMaterial({ color: 0x1f2a3b, transparent: true, opacity: 0.35 });
  const box = new THREE.LineSegments(new THREE.EdgesGeometry(boxGeom), boxMat);
  axisGroup.add(box);
}

function buildZones() {
  zoneGroup.clear();
  if (!SHOW_ZONES) return;

  const zoneSize = SPACE_SCALE * PATH_SCALE * AXIS_SCALE;
  const cell = zoneSize / 3;
  const geom = new THREE.BoxGeometry(cell, cell, cell);
  const mat = new THREE.MeshBasicMaterial({
    transparent: true,
    opacity: 0.14,
    depthWrite: false,
    depthTest: false,
    vertexColors: true,
  });
  const mesh = new THREE.InstancedMesh(geom, mat, 27);

  const entropyColor = new THREE.Color(0x57f0d5);
  const marginColor = new THREE.Color(0xffb347);
  const driftColor = new THREE.Color(0x7ad1ff);
  const weights = [0.12, 0.35, 0.6];

  let idx = 0;
  for (let xi = 0; xi < 3; xi += 1) {
    for (let yi = 0; yi < 3; yi += 1) {
      for (let zi = 0; zi < 3; zi += 1) {
        const x = (xi - 1) * cell;
        const y = (yi - 1) * cell;
        const z = (zi - 1) * cell;
        const matrix = new THREE.Matrix4().setPosition(x, y, z);
        mesh.setMatrixAt(idx, matrix);

        const color = new THREE.Color(0x000000);
        const wE = weights[xi];
        const wM = weights[yi];
        const wD = weights[zi];
        color.r = entropyColor.r * wE + marginColor.r * wM + driftColor.r * wD;
        color.g = entropyColor.g * wE + marginColor.g * wM + driftColor.g * wD;
        color.b = entropyColor.b * wE + marginColor.b * wM + driftColor.b * wD;
        mesh.setColorAt(idx, color);
        idx += 1;
      }
    }
  }

  mesh.instanceColor.needsUpdate = true;
  mesh.instanceMatrix.needsUpdate = true;
  zoneGroup.add(mesh);
}

function formatToken(text) {
  if (text == null) return '';
  if (text.includes('\n')) {
    text = text.replace(/\n/g, '\\n');
  }
  if (text.trim() === '') {
    return '[ws]';
  }
  if (text.length > 28) {
    return text.slice(0, 25) + '...';
  }
  return text;
}

function positionFromRecord(record) {
  const pos = record.position;
  let x;
  let y;
  let z;
  if (pos && Number.isFinite(pos.x) && Number.isFinite(pos.y) && Number.isFinite(pos.z)) {
    x = pos.x;
    y = pos.y;
    z = pos.z;
  } else {
    const metrics = record.metrics || {};
    x = normalize(metrics.entropy ?? 0, entropyMax);
    y = normalize(metrics.logit_margin ?? 0, marginMax);
    z = normalize(metrics.drift ?? 0, driftMax);
  }
  return new THREE.Vector3(
    (x - 0.5) * SPACE_SCALE * PATH_SCALE,
    (y - 0.5) * SPACE_SCALE * PATH_SCALE,
    (z - 0.5) * SPACE_SCALE * PATH_SCALE
  );
}

function addPoint(record) {
  const visibleText = processThink(record.token_text || '', thinkStateTokens);
  if (THINK_FILTER && visibleText === '' && thinkStateTokens.active) {
    return;
  }

  const pos = positionFromRecord(record);
  const color = colorFromMetrics(record.metrics || {});

  points.push(pos);
  colors.push(color.r, color.g, color.b);

  if (points.length > MAX_POINTS) {
    points.shift();
    colors.splice(0, 3);
    const oldSprite = sprites.shift();
    if (oldSprite) {
      overlayGroup.remove(oldSprite);
      disposeSprite(oldSprite);
    }
  }

  geometry.setFromPoints(points);
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.attributes.position.needsUpdate = true;
  geometry.attributes.color.needsUpdate = true;

  const tokenText = formatToken(visibleText || '');
  if (SHOW_SPRITES && tokenText) {
    const sprite = makeTextSprite(tokenText);
    sprite.position.copy(pos);
    sprites.push(sprite);
    overlayGroup.add(sprite);
  }

  if (visibleText) {
    if (SHOW_HEAD_LABEL) {
      if (headSprite) {
        overlayGroup.remove(headSprite);
        disposeSprite(headSprite);
      }
      headSprite = makeTextSprite(tokenText, { fontSize: 28, baseHeight: 0.1, maxHeight: 0.5 });
      headSprite.position.copy(pos.clone().add(new THREE.Vector3(0.2, 0.2, 0.2)));
      overlayGroup.add(headSprite);
    }
  }

  HUD.token.textContent = record.token_index ?? '-';
  if (record.metrics) {
    HUD.entropy.textContent = (record.metrics.entropy ?? 0).toFixed(3);
    HUD.margin.textContent = (record.metrics.logit_margin ?? 0).toFixed(3);
    HUD.drift.textContent = (record.metrics.drift ?? 0).toFixed(3);
    HUD.topk.textContent = (record.metrics.topk_mass ?? 0).toFixed(3);
  }
}

function scheduleRecord(record) {
  const now = performance.now();
  let deltaTokens = 1;
  if (Number.isFinite(record.token_index) && lastTokenIndex != null) {
    deltaTokens = Math.max(1, record.token_index - lastTokenIndex);
  }
  const scheduled = Math.max(now, lastScheduledTime + deltaTokens * TOKEN_PERIOD_MS);
  pending.push({ record, time: scheduled });
  lastScheduledTime = scheduled;
  if (Number.isFinite(record.token_index)) {
    lastTokenIndex = record.token_index;
  }
}

function handleRecord(record) {
  if (!record || !record.type) return;
  if (record.type === 'session_start') {
    const config = record.config || {};
    if (Number.isFinite(config.xyz_entropy_max)) entropyMax = config.xyz_entropy_max;
    if (Number.isFinite(config.xyz_margin_max)) marginMax = config.xyz_margin_max;
    if (Number.isFinite(config.xyz_drift_max)) driftMax = config.xyz_drift_max;
    if (HUD.endpoint && record.config?.base_url) {
      HUD.endpoint.value = record.config.base_url;
    }
    setStatus('Streaming', 'success');
    return;
  }
  if (record.type === 'text') {
    const visibleText = processThink(record.text || '', thinkStateTranscript);
    if (visibleText) {
      appendToken(visibleText);
    }
    return;
  }
  if (record.type === 'telemetry') {
    scheduleRecord(record);
  }
  if (record.type === 'session_end') {
    setStatus('Complete', 'info');
  }
}

const source = new EventSource('/stream');
source.onmessage = (event) => {
  try {
    const record = JSON.parse(event.data);
    handleRecord(record);
  } catch (err) {
    console.warn('Failed to parse telemetry', err);
  }
};

source.onerror = () => {
  console.warn('Stream disconnected. Retrying...');
};

async function startSession() {
  const prompt = HUD.prompt?.value?.trim();
  const baseUrl = HUD.endpoint?.value?.trim();
  if (!prompt) {
    setStatus('Enter a prompt', 'error');
    return;
  }
  if (!baseUrl) {
    setStatus('Enter endpoint URL', 'error');
    return;
  }
  setStatus('Starting...', 'info');
  try {
    const resp = await fetch('/api/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, base_url: baseUrl }),
    });
    if (!resp.ok) {
      const data = await resp.json();
      throw new Error(data.error || 'Failed to start');
    }
    resetScene();
    setStatus('Streaming', 'success');
  } catch (err) {
    setStatus(err.message, 'error');
  }
}

async function stopSession() {
  setStatus('Stopping...', 'info');
  try {
    const resp = await fetch('/api/stop', { method: 'POST' });
    if (!resp.ok) throw new Error('Failed to stop');
    setStatus('Stopped', 'info');
  } catch (err) {
    setStatus(err.message, 'error');
  }
}

if (HUD.start) {
  HUD.start.addEventListener('click', startSession);
}
if (HUD.stop) {
  HUD.stop.addEventListener('click', stopSession);
}

if (HUD.prompt) {
  HUD.prompt.addEventListener('keydown', (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
      startSession();
    }
  });
}

function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  while (pending.length && pending[0].time <= now) {
    const item = pending.shift();
    addPoint(item.record);
  }

  for (const sprite of sprites) {
    const distance = camera.position.distanceTo(sprite.position);
    const height = Math.min(sprite.userData.baseHeight * distance, sprite.userData.maxHeight || Infinity);
    sprite.scale.set(height * sprite.userData.aspect, height, 1);
  }

  for (const sprite of axisSprites) {
    const distance = camera.position.distanceTo(sprite.position);
    const height = Math.min(sprite.userData.baseHeight * distance, sprite.userData.maxHeight || Infinity);
    sprite.scale.set(height * sprite.userData.aspect, height, 1);
  }

  if (headSprite) {
    const distance = camera.position.distanceTo(headSprite.position);
    const height = Math.min(headSprite.userData.baseHeight * distance, headSprite.userData.maxHeight || Infinity);
    headSprite.scale.set(height * headSprite.userData.aspect, height, 1);
  }

  controls.update();
  renderer.clear();
  composer.render();
  renderer.clearDepth();
  renderer.render(overlayScene, camera);
}

function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
  bloomPass.setSize(window.innerWidth, window.innerHeight);
}

window.addEventListener('resize', onResize);

animate();
