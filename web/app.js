/* Echo Edge browser client.
 *
 * Polar H10 -> Web Bluetooth (standard Heart Rate Service) -> RR intervals
 * -> WASM inference core -> readiness state. No network calls.
 *
 * The H10 also exposes raw 130 Hz ECG over Polar's proprietary PMD service.
 * That path is not implemented here; see README "Raw ECG" for the plan.
 */

const HR_SERVICE = 'heart_rate';
const HR_CHAR = 'heart_rate_measurement';
const STATES = ['Green', 'Amber', 'Red'];
const ENROLL_MS = 180000;              // three minutes of quiet
const BASELINE_KEY = 'echo.baseline';  // per-browser, never transmitted
const FEATURES = ['mean_rr', 'mean_hr', 'sdnn', 'rmssd', 'pnn50', 'rr_slope', 'hr_cv', 'coverage'];

const $ = (id) => document.getElementById(id);
const el = {
  connect: $('connect'), sim: $('sim'), status: $('status'), device: $('device'),
  beats: $('beats'), model: $('model'), state: $('state'), detail: $('detail'),
  stack: $('stack'), trace: $('trace'), enroll: $('enroll'), bar: $('bar'),
  basestat: $('basestat'), forget: $('forget'),
};

let echo = null;      // wasm bindings
let beatCount = 0;
let history = [];     // recent HR for the trace
let simTimer = null;

/* ---------- wasm ---------- */

async function loadCore() {
  if (typeof EchoModule !== 'function') {
    fail('echo.js missing — run wasm/build.sh first');
    return false;
  }
  const m = await EchoModule();
  echo = {
    init: m.cwrap('echo_wasm_init', null, ['number']),
    push: m.cwrap('echo_wasm_push', 'number', ['number']),
    state: m.cwrap('echo_wasm_state', 'number', []),
    conf: m.cwrap('echo_wasm_confidence', 'number', []),
    feat: m.cwrap('echo_wasm_feature', 'number', ['number']),
    modelId: m.cwrap('echo_wasm_model_id', 'string', []),
    enrolling: m.cwrap('echo_wasm_enrolling', 'number', []),
    ready: m.cwrap('echo_wasm_ready', 'number', []),
    baselined: m.cwrap('echo_wasm_baselined', 'number', []),
    progress: m.cwrap('echo_wasm_enroll_progress', 'number', []),
    baseline: m.cwrap('echo_wasm_baseline', 'number', ['number']),
    stage: m.cwrap('echo_wasm_stage_baseline', null, ['number', 'number']),
    commit: m.cwrap('echo_wasm_commit_baseline', null, []),
  };
  echo.init(60000, ENROLL_MS);
  el.model.textContent = echo.modelId();

  if (!echo.baselined()) {
    el.basestat.textContent = 'not required';
    el.forget.disabled = true;
    return true;
  }
  restoreBaseline();
  return true;
}

/* ---------- baseline persistence ----------
 * The baseline is eight numbers describing someone's resting physiology.
 * It stays in this browser and is never sent anywhere. Clearing it forces
 * re-enrollment, which is the right move if the model is behaving oddly or
 * the device is shared. */

function restoreBaseline() {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(BASELINE_KEY) || 'null'); }
  catch { saved = null; }

  if (Array.isArray(saved) && saved.length === FEATURES.length &&
      saved.every(Number.isFinite)) {
    saved.forEach((v, i) => echo.stage(i, v));
    echo.commit();
    el.basestat.textContent = 'restored';
    el.detail.textContent = 'Baseline restored — ready to classify';
  } else {
    el.basestat.textContent = 'not enrolled';
  }
}

function saveBaseline() {
  const b = FEATURES.map((_, i) => echo.baseline(i));
  try { localStorage.setItem(BASELINE_KEY, JSON.stringify(b)); } catch { /* private mode */ }
  el.basestat.textContent = 'enrolled';
}

function forgetBaseline() {
  try { localStorage.removeItem(BASELINE_KEY); } catch { /* ignore */ }
  echo.init(60000, ENROLL_MS);
  beatCount = 0; history = []; lastState = -1; enrolledOnce = false;
  el.basestat.textContent = 'not enrolled';
  el.state.dataset.s = '-';
  el.state.textContent = 'Standby';
  el.detail.textContent = 'Baseline cleared — enrollment will restart';
  el.trace.innerHTML = '';
}

/* ---------- ingest ---------- */

function parseHR(dv) {
  const flags = dv.getUint8(0);
  let i = 1;
  i += (flags & 0x01) ? 2 : 1;              // HR value, 8- or 16-bit
  if (flags & 0x08) i += 2;                 // energy expended
  const rr = [];
  if (flags & 0x10) {                       // RR intervals present
    for (; i + 1 < dv.byteLength; i += 2) {
      rr.push(dv.getUint16(i, true) / 1024 * 1000); // 1/1024 s -> ms
    }
  }
  return rr;
}

async function connect() {
  if (!navigator.bluetooth) {
    fail('Web Bluetooth unavailable. Use Chrome or Edge over HTTPS.');
    return;
  }
  try {
    el.status.textContent = 'pairing';
    const dev = await navigator.bluetooth.requestDevice({
      filters: [{ services: [HR_SERVICE] }],
      optionalServices: [HR_SERVICE],
    });
    dev.addEventListener('gattserverdisconnected', () => {
      el.status.textContent = 'disconnected';
      el.connect.classList.remove('armed');
    });

    const server = await dev.gatt.connect();
    const char = await (await server.getPrimaryService(HR_SERVICE))
      .getCharacteristic(HR_CHAR);
    await char.startNotifications();
    char.addEventListener('characteristicvaluechanged', (e) => {
      parseHR(e.target.value).forEach(ingest);
    });

    stopSim();
    el.device.textContent = dev.name || 'heart rate strap';
    el.status.textContent = 'streaming';
    el.connect.classList.add('armed');
    el.detail.textContent = 'Filling the 60-second window';
  } catch (err) {
    el.status.textContent = 'standby';
    if (err && err.name !== 'NotFoundError') fail(err.message);
  }
}

/* ---------- simulated feed ---------- */

function startSim() {
  if (simTimer) { stopSim(); return; }
  stopSimNoLabel();
  el.device.textContent = 'simulated';
  el.status.textContent = 'streaming';
  el.sim.classList.add('armed');
  el.detail.textContent = 'Filling the 60-second window';

  let hr = 62, t = 0;
  simTimer = setInterval(() => {
    t += 1;
    hr = 62 + 46 * Math.min(t / 220, 1) + Math.sin(t / 9) * 2;   // gradual load
    const hrv = Math.max(52 - 40 * Math.min(t / 220, 1), 6);
    ingest(60000 / hr + (Math.random() - 0.5) * hrv * 2);
  }, 120);
}

function stopSimNoLabel() { if (simTimer) { clearInterval(simTimer); simTimer = null; } }
function stopSim() {
  stopSimNoLabel();
  el.sim.classList.remove('armed');
  if (el.device.textContent === 'simulated') el.status.textContent = 'standby';
}

/* ---------- inference + render ---------- */

let enrolledOnce = false;

function ingest(rrMs) {
  if (!echo) return;
  beatCount++;
  el.beats.textContent = beatCount;
  const valid = echo.push(rrMs);

  if (echo.enrolling()) {
    showEnrolling(echo.progress());
    return;
  }
  if (!enrolledOnce && echo.ready() && echo.baselined()) {
    enrolledOnce = true;
    el.enroll.hidden = true;
    saveBaseline();
  }
  if (!valid) return;

  const f = {};
  FEATURES.forEach((n, i) => { f[n] = echo.feat(i); });
  render({ state: echo.state(), conf: echo.conf(), f });
}

function showEnrolling(p) {
  el.enroll.hidden = false;
  el.bar.style.width = `${Math.round(p * 100)}%`;
  el.state.dataset.s = '-';
  el.state.textContent = 'Enrolling';
  const left = Math.max(0, Math.ceil((1 - p) * ENROLL_MS / 1000));
  el.detail.textContent = `Learning your baseline — about ${left}s remaining`;
  el.basestat.textContent = 'enrolling';
}

let lastState = -1;

function render({ state, conf, f }) {
  el.state.dataset.s = String(state);
  el.state.textContent = STATES[state];
  el.detail.textContent =
    `${f.mean_hr.toFixed(0)} bpm · rmssd ${f.rmssd.toFixed(0)} ms · ` +
    `confidence ${(conf * 100).toFixed(0)}%`;

  history.push(f.mean_hr);
  if (history.length > 220) history.shift();
  drawTrace(state);

  // One strip per state transition. A strip per beat would be noise.
  if (state !== lastState) {
    lastState = state;
    addStrip(state, conf, f);
  }
}

function addStrip(state, conf, f) {
  const empty = el.stack.querySelector('.empty');
  if (empty) empty.remove();

  const cell = (val, label) =>
    `<div class="cell"><b>${val}</b><i>${label}</i></div>`;

  const strip = document.createElement('div');
  strip.className = 'strip enter';
  strip.innerHTML =
    `<div class="block" data-s="${state}">${STATES[state].toUpperCase()}</div>` +
    cell(f.mean_hr.toFixed(0), 'bpm') +
    cell(f.sdnn.toFixed(0), 'sdnn ms') +
    cell(f.rmssd.toFixed(0), 'rmssd ms') +
    cell((f.pnn50 * 100).toFixed(0) + '%', 'pnn50') +
    cell(f.rr_slope.toFixed(2), 'drift ms/beat') +
    cell(new Date().toLocaleTimeString([], { hour12: false }), 'logged');

  el.stack.prepend(strip);
  while (el.stack.children.length > 60) el.stack.lastElementChild.remove();
}

function drawTrace(state) {
  const w = 1000, h = 100, n = history.length;
  if (n < 2) return;
  const lo = Math.min(...history) - 3, hi = Math.max(...history) + 3;
  const pts = history.map((v, i) =>
    `${(i / (n - 1)) * w},${h - ((v - lo) / (hi - lo || 1)) * h}`).join(' ');
  const stroke = ['--green', '--amber', '--red'][state] || '--dim';
  el.trace.setAttribute('viewBox', `0 0 ${w} ${h}`);
  el.trace.innerHTML =
    `<polyline points="${pts}" fill="none" stroke-width="2"
       stroke="var(${stroke})" vector-effect="non-scaling-stroke"/>`;
}

function fail(msg) {
  el.status.textContent = 'error';
  el.detail.textContent = msg;
}

/* ---------- boot ---------- */

el.connect.addEventListener('click', connect);
el.sim.addEventListener('click', startSim);
el.forget.addEventListener('click', forgetBaseline);

loadCore().then((ok) => {
  if (!ok) { el.connect.disabled = true; el.sim.disabled = true; el.forget.disabled = true; }
});
