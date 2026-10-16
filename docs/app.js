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
const TREND_LEVELS = ['NONE', 'WATCH', 'ELEVATED', 'HIGH'];  // echo_wasm_trend_level order
const ENROLL_MS = 180000;              // three minutes of quiet
const BASELINE_KEY = 'echo.baseline';  // suffixed per participant code
const DEFAULT_PID = 'P01';
const FEATURES = ['mean_rr', 'mean_hr', 'sdnn', 'rmssd', 'pnn50', 'rr_slope', 'hr_cv', 'coverage'];

/* Quick-mark labels. Edit for a different setting. */
const MARK_TAGS = ['waiting', 'moving', 'take', 'talking', 'stress', 'food'];

/* Random context sampling. Marking only when something spikes yields a record
   of flagged moments with nothing to compare them against, which leaves the
   flagged moments uninterpretable. Random prompts supply the base rate. */
const PROMPT_MIN_MS = 600000;
const PROMPT_MAX_MS = 900000;
const PROMPT_TIMEOUT_MS = 90000;

const $ = (id) => document.getElementById(id);
const el = {
  connect: $('connect'), sim: $('sim'), status: $('status'), device: $('device'),
  beats: $('beats'), model: $('model'), state: $('state'), detail: $('detail'),
  stack: $('stack'), trace: $('trace'), enroll: $('enroll'), bar: $('bar'),
  basestat: $('basestat'), forget: $('forget'), pid: $('pid'),
  logged: $('logged'), export: $('export'), note: $('note'),
  savebase: $('savebase'), loadbase: $('loadbase'), basefile: $('basefile'),
  temp: $('temp'), rh: $('rh'), wbgt: $('wbgt'), setting: $('setting'),
  quality: $('quality'), discard: $('discard'),
  mark: $('mark'), marklabel: $('marklabel'), marktags: $('marktags'),
  marklast: $('marklast'), prompt: $('prompt'), ptags: $('ptags'),
  pskip: $('pskip'), sampling: $('sampling'),
  visit: $('visit'), phasebtns: $('phasebtns'), beats_logged: $('beats_logged'),
  exportbeats: $('exportbeats'),
};

/* Environment is read at log time, not at session start, so a value entered
 * mid-session applies from that point forward rather than being lost. */
function environment() {
  const num = (elem) => {
    const v = parseFloat(elem.value);
    return Number.isFinite(v) ? String(v) : '';
  };
  return [num(el.temp), num(el.rh), num(el.wbgt), el.setting.value];
}

/* Every classified window, kept for export. Strips show transitions only;
 * research needs the whole series. */
let sessionLog = [];

/* Every accepted or rejected interval as it arrived, before any processing.
 *
 * The window log above records features, and a feature is a derived product
 * of one version of the extractor. When the extractor changes, old feature
 * rows stop being comparable with new ones and nothing can be recomputed,
 * because the signal they came from is gone. In a reliability study that
 * failure does not look like a software change, it looks like between-session
 * variance, which is the quantity being measured.
 *
 * The interval series has no such problem. Artifact rejection, epoch length
 * and respiration correction are all pure functions of it, so any of them can
 * be varied or re-run later over sessions already collected. This array is
 * what makes a dataset outlive the code that produced it. */
let beatLog = [];
let packetSeq = 0;

/* The participant code the in-memory logs belong to, which is not always the
 * one in the input field.
 *
 * switchParticipant fires on the field's change event, so by the time it runs
 * the field already holds the incoming code. Keying the outgoing rows off the
 * field therefore filed them under the new participant, and a reload then
 * recovered one person's session as another person's data. The logs are keyed
 * off this instead, and it moves only after the outgoing rows are written. */
let activePid = DEFAULT_PID;

let sessionStart = null;
let wakeLock = null;
let mode = 'rest';
let pendingMark = null;
let markCount = 0;
let promptTimer = null;
let promptTimeout = null;
let samplingOn = false;

/* Mobile browsers evict background tabs under memory pressure. A glance at
 * another app can therefore destroy an entire session, and the loss is silent
 * — the page simply reloads empty. The log is mirrored to local storage as it
 * grows so a reload recovers it. */
const SESSION_PREFIX = 'echo.session.';
const BEATS_PREFIX = 'echo.beats.';
const PERSIST_EVERY = 10;    // windows between writes
const PERSIST_BEATS_EVERY = 100;  // intervals between writes

function sessionKey() { return SESSION_PREFIX + activePid; }
function beatsKey() { return BEATS_PREFIX + activePid; }

function persistSession() {
  try { localStorage.setItem(sessionKey(), JSON.stringify(sessionLog)); }
  catch { /* quota or private mode — the session simply is not recoverable */ }
}

/* Beats are mirrored on the same principle as windows, but they arrive about
 * sixty times as often, so the write interval is correspondingly longer. A
 * two-hour visit at 90 bpm is roughly 11,000 rows, which fits a typical 5 MB
 * origin quota with room to spare. */
function persistBeats() {
  try { localStorage.setItem(beatsKey(), JSON.stringify(beatLog)); }
  catch { /* quota or private mode, as in persistSession */ }
}

/* Recover whatever survived, each log independently.
 *
 * The two mirrors have independent lifetimes: exporting windows clears the
 * window mirror and leaves the beat mirror alone, so after a window export
 * and a reload the beats are present while the windows are gone. An earlier
 * version returned early when the window mirror was missing and never looked
 * at the beat mirror, which left the interval series sitting in storage,
 * unreachable through the interface, with the export control disabled. The
 * next clear or participant switch then destroyed it silently. Since the
 * interval series is the measurement and the window log is a convenience,
 * that was exactly the wrong one to lose. Read both, report both. */
function recoverSession() {
  activePid = participant();

  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(sessionKey()) || 'null'); }
  catch { saved = null; }
  const haveWindows = Array.isArray(saved) && saved.length > 0;

  let savedBeats = null;
  try { savedBeats = JSON.parse(localStorage.getItem(beatsKey()) || 'null'); }
  catch { savedBeats = null; }
  const haveBeats = Array.isArray(savedBeats) && savedBeats.length > 0;

  if (!haveWindows && !haveBeats) return;

  if (haveWindows) {
    sessionLog = saved;
    pendingMark = null;
    el.logged.textContent = sessionLog.length;
    el.export.disabled = false;
    if (sessionLog.length % PERSIST_EVERY === 0) persistSession();
  }

  if (haveBeats) {
    beatLog = savedBeats;
    packetSeq = beatLog[beatLog.length - 1][2] || 0;
    el.beats_logged.textContent = beatLog.length;
    el.exportbeats.disabled = false;
  }

  el.discard.disabled = false;

  const parts = [];
  if (haveWindows) parts.push(`${sessionLog.length} window rows`);
  if (haveBeats) parts.push(`${beatLog.length} beats`);
  el.detail.textContent =
    `Recovered ${parts.join(' and ')} for ${activePid}` +
    (haveBeats && !haveWindows
      ? ' — the windows were already exported, the intervals were not. Export the beats before clearing.'
      : ' from an interrupted session — export or clear before starting a new one');
}

function clearSession() {
  sessionLog = [];
  beatLog = [];
  packetSeq = 0;
  sessionStart = null;
  try { localStorage.removeItem(sessionKey()); } catch { /* ignore */ }
  try { localStorage.removeItem(beatsKey()); } catch { /* ignore */ }
  el.logged.textContent = '0';
  el.beats_logged.textContent = '0';
  el.export.disabled = true;
  el.exportbeats.disabled = true;
  el.discard.disabled = true;
}

/* Keep the screen awake while streaming. Screen-off is the most common route
 * to tab eviction, and a dark screen also stops the operator noticing a
 * dropped connection. */
async function holdWakeLock() {
  if (!('wakeLock' in navigator) || wakeLock) return;
  try {
    wakeLock = await navigator.wakeLock.request('screen');
    wakeLock.addEventListener('release', () => { wakeLock = null; });
  } catch { /* unsupported or denied — not fatal */ }
}

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') {
    if (el.status.textContent === 'streaming') holdWakeLock();
  } else {
    persistSession();   // last chance before the tab may be discarded
  }
});

function participant() {
  const v = (el.pid.value || '').trim().toUpperCase();
  return v || DEFAULT_PID;
}

function baselineKey() { return `${BASELINE_KEY}.${participant()}`; }

/* Visit number is entered rather than derived from the date. A protocol that
 * handles missing sessions through a mixed model expects visits to be missed
 * and rescheduled, and the moment one is, date arithmetic puts a visit 5 in
 * the visit 4 slot without saying so. */
function visitNo() {
  const v = parseInt(el.visit.value, 10);
  return Number.isFinite(v) && v > 0 ? String(v) : '';
}

/* Identity has to be complete before a single beat is recorded.
 *
 * A reliability estimate on a crossed design needs to know which occasion
 * each measurement belongs to. Position in a file is not an occasion label,
 * and a visit number cannot be reconstructed afterwards from a timestamp once
 * a session has been missed or rescheduled. Both sessions exported on 9
 * October carry a blank visit field and `vx` in their filenames, which makes
 * them unusable for a two-way model no matter what else is right about them.
 *
 * The participant check reads the input directly rather than through
 * participant(), because that helper substitutes DEFAULT_PID for an empty
 * box. A silent default is how a session ends up filed under someone else. */
function identityComplete() {
  const missing = [];
  if (!(el.pid.value || '').trim()) missing.push('participant code');
  if (!visitNo()) missing.push('visit number');
  if (!phaseNow()) missing.push('protocol phase');
  if (!missing.length) return true;
  fail(`Cannot start: enter the ${missing.join(', ')} first. `
       + 'These are stamped on every row and cannot be added later.');
  return false;
}

/* Protocol phase, stamped on every row of both logs.
 *
 * This used to live in the free-text session note. A phase label typed by
 * hand ninety minutes into a treadmill test is not a label, and an analysis
 * that reports reliability separately by phase has no outcome to attach to a
 * row whose phase is unknown or misspelled. */
const PHASES = ['supine', 'seated', 'paced', 'bruce', 'recovery', 'other'];
let phase = '';
function phaseNow() { return phase; }

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
    readiness: m.cwrap('echo_wasm_readiness', 'number', []),
    trend: m.cwrap('echo_wasm_trend_level', 'number', []),
    feat: m.cwrap('echo_wasm_feature', 'number', ['number']),
    modelId: m.cwrap('echo_wasm_model_id', 'string', []),
    enrolling: m.cwrap('echo_wasm_enrolling', 'number', []),
    ready: m.cwrap('echo_wasm_ready', 'number', []),
    baselined: m.cwrap('echo_wasm_baselined', 'number', []),
    progress: m.cwrap('echo_wasm_enroll_progress', 'number', []),
    baseline: m.cwrap('echo_wasm_baseline', 'number', ['number']),
    stage: m.cwrap('echo_wasm_stage_baseline', null, ['number', 'number']),
    vote: m.cwrap('echo_wasm_vote', 'number', ['number']),
    quality: m.cwrap('echo_wasm_quality', 'number', []),
    rawState: m.cwrap('echo_wasm_raw_state', 'number', []),
    commit: m.cwrap('echo_wasm_commit_baseline', null, []),
  };
  echo.init(60000, ENROLL_MS);
  el.model.textContent = echo.modelId();

  if (!echo.baselined()) {
    el.basestat.textContent = 'not required';
    el.forget.disabled = true;
    recoverSession();
    return true;
  }
  restoreBaseline();
  recoverSession();
  return true;
}

/* ---------- baseline persistence ----------
 * The baseline is eight numbers describing someone's resting physiology.
 * It stays in this browser and is never sent anywhere. Clearing it forces
 * re-enrollment, which is the right move if the model is behaving oddly or
 * the device is shared. */

function restoreBaseline() {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(baselineKey()) || 'null'); }
  catch { saved = null; }

  const stale = (why) => {
    // Refused rather than silently discarded: an operator who thinks a
    // participant is enrolled and is actually re-enrolling should be told.
    try { localStorage.removeItem(baselineKey()); } catch { /* ignore */ }
    el.basestat.textContent = 'stale, re-enroll';
    el.savebase.disabled = true;
    el.detail.textContent = `Stored baseline rejected (${why}). Re-enroll before collecting.`;
  };

  if (Array.isArray(saved)) return stale('saved by an older build');
  if (!saved || typeof saved !== 'object') {
    el.basestat.textContent = 'not enrolled';
    el.savebase.disabled = true;
    return;
  }
  if (saved.v !== BASELINE_FORMAT) return stale('unrecognised format');
  if (!Array.isArray(saved.features) ||
      saved.features.join(',') !== FEATURES.join(',')) return stale('different feature set');
  if (saved.model !== echo.modelId()) return stale('different build');
  if (!Array.isArray(saved.baseline) || saved.baseline.length !== FEATURES.length ||
      !saved.baseline.every(Number.isFinite)) return stale('malformed');

  saved.baseline.forEach((v, i) => echo.stage(i, v));
  echo.commit();
  el.basestat.textContent = 'restored';
  el.savebase.disabled = false;
  el.detail.textContent = 'Baseline restored, ready to classify';
}

/* A stored baseline is only valid for the build that produced it.
 *
 * The baseline is the reference every feature is expressed as deviation from,
 * so one enrolled under a different extractor biases every subsequent reading
 * by the difference between the two extractors, silently and for as long as
 * it persists. This used to be stored as a bare array of eight numbers and
 * restored on the strength of being an array of eight numbers, which cannot
 * detect that at all.
 *
 * The model id is the available stamp and it is an imperfect one: it hashes
 * the training configuration and the feature names, so two builds with the
 * same hyperparameters and a changed extractor share an id. It catches the
 * case that occurs in practice, where a change to the extractor is followed
 * by a retrain at different settings. A fingerprint computed from the
 * extractor's own output would catch all of them and is the right next
 * version of this guard. */
const BASELINE_FORMAT = 2;

function saveBaseline() {
  const b = FEATURES.map((_, i) => echo.baseline(i));
  const payload = {
    v: BASELINE_FORMAT,
    model: echo.modelId(),
    features: FEATURES,
    created: new Date().toISOString(),
    baseline: b,
  };
  try { localStorage.setItem(baselineKey(), JSON.stringify(payload)); }
  catch { /* private mode */ }
  el.basestat.textContent = 'enrolled';
  el.savebase.disabled = false;
}

/* ---------- baseline as a portable file ----------
 * Local storage is per-browser and per-device. Without a file, moving a
 * participant to another laptop silently re-enrolls them against a different
 * reference, which looks like a physiological step change and is not one. */

function downloadBaseline() {
  const b = FEATURES.map((_, i) => echo.baseline(i));
  if (!b.every(Number.isFinite)) return;

  const payload = {
    participant: participant(),
    created: new Date().toISOString(),
    model: echo.modelId(),
    features: FEATURES,
    baseline: b,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)],
                        { type: 'application/json' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `baseline_${participant()}.json`;
  link.click();
  URL.revokeObjectURL(link.href);
}

async function uploadBaseline(file) {
  let data;
  try { data = JSON.parse(await file.text()); }
  catch { fail('Baseline file is not readable JSON'); return; }

  if (!Array.isArray(data.baseline) || data.baseline.length !== FEATURES.length
      || !data.baseline.every(Number.isFinite)) {
    fail('Baseline file is malformed'); return;
  }
  // Feature order is part of the contract; a mismatch means a different build.
  if (Array.isArray(data.features) &&
      data.features.join(',') !== FEATURES.join(',')) {
    fail('Baseline was made with a different feature set. Re-enroll instead.');
    return;
  }
  // Same reasoning as restoreBaseline: a baseline from another build is not
  // a baseline for this one, whatever the feature names say.
  if (data.model && data.model !== echo.modelId()) {
    fail(`Baseline was made by build ${data.model}, this is ${echo.modelId()}. Re-enroll instead.`);
    return;
  }
  if (data.participant && data.participant !== participant()) {
    el.pid.value = data.participant;
    switchParticipant();
  }

  data.baseline.forEach((v, i) => echo.stage(i, v));
  echo.commit();
  enrolledOnce = true;
  el.enroll.hidden = true;
  el.basestat.textContent = 'loaded from file';
  el.savebase.disabled = false;
  const when = (data.created || '').slice(0, 10);
  el.detail.textContent =
    `Baseline loaded for ${participant()}${when ? ` (enrolled ${when})` : ''}`;
  try { localStorage.setItem(baselineKey(), JSON.stringify(data.baseline)); }
  catch { /* private mode */ }
}

function forgetBaseline() {
  try { localStorage.removeItem(baselineKey()); } catch { /* ignore */ }
  echo.init(60000, ENROLL_MS);
  beatCount = 0; history = []; lastState = -1; enrolledOnce = false;
  el.basestat.textContent = 'not enrolled';
  el.savebase.disabled = true;
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
  if (!identityComplete()) return;
  if (!navigator.bluetooth) {
    fail('Web Bluetooth unavailable. Use Chrome or Edge over HTTPS.');
    return;
  }
  try {
    el.status.textContent = 'pairing';
    const dev = await navigator.bluetooth.requestDevice({
      filters: [{ services: [HR_SERVICE] }],
      optionalServices: [HR_SERVICE, PMD_SERVICE],
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
      const rr = parseHR(e.target.value);
      if (!rr.length) return;
      // One sequence number per notification. Interval values reconstruct the
      // fine time axis on their own, but a dropped notification is invisible
      // to a cumulative sum: the series simply continues, shorter. Grouping
      // intervals by the packet they arrived in makes the gap detectable
      // against wall clock.
      packetSeq++;
      rr.forEach((v) => ingest(v));
    });

    stopSim();
    el.device.textContent = dev.name || 'heart rate strap';
    el.status.textContent = 'streaming';
    el.connect.classList.add('armed');
    el.detail.textContent = 'Filling the 60-second window';
    holdWakeLock();
    // Accelerometer rides alongside; a refusal leaves heart rate untouched.
    startMotion(server);
  } catch (err) {
    el.status.textContent = 'standby';
    if (err && err.name !== 'NotFoundError') fail(err.message);
  }
}

/* ---------- simulated feed ---------- */

function startSim() {
  if (simTimer) { stopSim(); return; }
  if (!identityComplete()) return;
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
    packetSeq++;
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

/* ---------- activity mode ----------
 * The model was trained on seated adults under mental stress. Under exertion
 * its input is out of distribution and the state saturates, so a strap worn
 * through a long active stretch reads RED throughout. That is the signal
 * running out of range, not a finding. While Active the light is withheld and
 * the raw physiology shown instead. Everything is still logged. */

function setMode(next) {
  mode = next;
  document.querySelectorAll('.mode').forEach((b) =>
    b.classList.toggle('on', b.dataset.mode === next));
  lastState = -1;
  if (next === 'active') {
    el.state.dataset.s = '-';
    el.state.textContent = 'Active';
  }
  markEvent('mode:' + next, true);
}

/* ---------- marks ----------
 * A mark stamps the moment. The label can be vague or missing and still be
 * useful; the timestamp is the part that cannot be reconstructed afterwards.
 * It attaches to the next logged window, under a second away at rest. */

function markEvent(label, quiet) {
  const clean = String(label || 'mark')
    .replace(/[",\n]/g, ' ').trim().slice(0, 40) || 'mark';
  pendingMark = clean;
  markCount++;
  const now = new Date().toLocaleTimeString([], { hour12: false });
  el.marklast.textContent = markCount + ' marks \u00b7 last "' + clean + '" at ' + now;
  if (!quiet && navigator.vibrate) navigator.vibrate(35);
  dismissPrompt(true);
}

function markFromInput() {
  markEvent(el.marklabel.value);
  el.marklabel.value = '';
  el.marklabel.blur();
}

/* ---------- random context prompts ---------- */

function nextPromptDelay() {
  return PROMPT_MIN_MS + Math.floor(Math.random() * (PROMPT_MAX_MS - PROMPT_MIN_MS));
}

function queuePrompt() {
  clearTimeout(promptTimer);
  if (!samplingOn) return;
  promptTimer = setTimeout(firePrompt, nextPromptDelay());
}

function firePrompt() {
  el.prompt.hidden = false;
  if (navigator.vibrate) navigator.vibrate([120, 80, 120]);
  clearTimeout(promptTimeout);
  // An ignored prompt is still data: it records that the schedule fired and
  // went unanswered, which keeps the sampling density honest.
  promptTimeout = setTimeout(() => {
    markEvent('prompt:missed', true);
    el.prompt.hidden = true;
    queuePrompt();
  }, PROMPT_TIMEOUT_MS);
}

function dismissPrompt(fromMark) {
  const wasOpen = !el.prompt.hidden;
  el.prompt.hidden = true;
  clearTimeout(promptTimeout);
  promptTimeout = null;
  if (wasOpen && fromMark) queuePrompt();
}

function startSampling() {
  samplingOn = true;
  el.sampling.textContent = 'Stop random prompts';
  el.sampling.classList.add('armed');
  clearTimeout(promptTimer);
  promptTimer = setTimeout(firePrompt, 30000);
}

function stopSampling() {
  samplingOn = false;
  clearTimeout(promptTimer);
  clearTimeout(promptTimeout);
  el.prompt.hidden = true;
  el.sampling.textContent = 'Start random prompts';
  el.sampling.classList.remove('armed');
}

function buildTagButtons() {
  MARK_TAGS.forEach((tag) => {
    const a = document.createElement('button');
    a.textContent = tag;
    a.addEventListener('click', () => markEvent(tag));
    el.marktags.appendChild(a);

    const b = document.createElement('button');
    b.textContent = tag;
    b.addEventListener('click', () => markEvent('prompt:' + tag));
    el.ptags.appendChild(b);
  });
}

let enrolledOnce = false;

function ingest(rrMs) {
  if (!echo) return;
  beatCount++;
  el.beats.textContent = beatCount;

  // The session clock starts at the first beat, not at the first classified
  // window, so the two exports share one origin and can be joined on
  // elapsed_s. A window log that starts sixty seconds after the beat log
  // cannot be aligned with it without guessing.
  if (sessionStart === null) sessionStart = Date.now();
  const elapsed = ((Date.now() - sessionStart) / 1000).toFixed(2);

  // Written before the filter sees it. Whether this interval is an artifact
  // is a judgement the current extractor makes and a future one may make
  // differently, so the judgement is not what gets stored: the interval is.
  beatLog.push([new Date().toISOString(), elapsed, packetSeq, rrMs.toFixed(1),
                participant(), visitNo(), phaseNow(), mode, motionNow()]);
  el.beats_logged.textContent = beatLog.length;
  el.exportbeats.disabled = false;
  if (beatLog.length % PERSIST_BEATS_EVERY === 0) persistBeats();

  const valid = echo.push(rrMs);

  // Signal quality is the share of intervals surviving the artifact filters.
  // Poor electrode contact shows up here long before it looks like physiology.
  const q = echo.quality();
  el.quality.textContent = `${(q * 100).toFixed(0)}%`;
  el.quality.style.color = q < 0.9 ? 'var(--red)'
                         : q < 0.97 ? 'var(--amber)' : 'var(--dim)';

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
  if (mode === 'active') {
    el.state.dataset.s = '-';
    el.state.textContent = 'Active';
    el.detail.textContent =
      `${f.mean_hr.toFixed(0)} bpm · rmssd ${f.rmssd.toFixed(0)} ms · ` +
      `state withheld during exertion`;
  } else {
    el.state.dataset.s = String(state);
    el.state.textContent = STATES[state];
    const tl = TREND_LEVELS[echo.trend()] || 'NONE';
    el.detail.textContent =
      `${f.mean_hr.toFixed(0)} bpm · rmssd ${f.rmssd.toFixed(0)} ms · ` +
      `readiness ${echo.readiness().toFixed(0)} · confidence ${(conf * 100).toFixed(0)}%` +
      (tl !== 'NONE' ? ` · trend ${tl}` : '');
  }

  // Log every window, regardless of whether it changes the displayed state.
  // sessionStart is set on the first beat, in ingest.
  sessionLog.push({
    t: new Date().toISOString(),
    elapsed: ((Date.now() - sessionStart) / 1000).toFixed(1),
    pid: participant(),
    visit: visitNo(),
    phase: phaseNow(),
    note: (el.note.value || '').replace(/[",\n]/g, ' ').trim(),
    mode: mode,
    mark: pendingMark || '',
    env: environment(),
    quality: echo.quality().toFixed(3),
    motion: motionNow(),
    raw: STATES[echo.rawState()].toUpperCase(),
    state: STATES[state].toUpperCase(),
    conf: conf.toFixed(4),
    readiness: echo.readiness().toFixed(2),
    trend: TREND_LEVELS[echo.trend()] || 'NONE',
    votes: [0, 1, 2].map((i) => echo.vote(i).toFixed(4)),
    f: FEATURES.map((n) => f[n].toFixed(4)),
  });
  pendingMark = null;
  el.logged.textContent = sessionLog.length;
  el.export.disabled = false;
  el.discard.disabled = false;
  if (sessionLog.length % PERSIST_EVERY === 0) persistSession();

  history.push(f.mean_hr);
  if (history.length > 220) history.shift();
  drawTrace(state);

  // One strip per state transition. A strip per beat would be noise.
  if (mode !== 'active' && state !== lastState) {
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

/* The interval series, one row per beat, unprocessed.
 *
 * This is the file that has to survive. The window export is a convenience
 * and a record of what the device decided at the time; this one is the
 * measurement. Rows are written in arrival order and carry the packet they
 * arrived in, so a dropped notification can be found by comparing the jump
 * in elapsed_s against the intervals that bridge it. */
function exportBeats() {
  if (!beatLog.length) return false;
  const header = ['timestamp', 'elapsed_s', 'packet', 'rr_ms',
                  'participant', 'visit', 'phase', 'activity', 'motion_mg'].join(',');
  // Rows recovered from a session saved before the motion column existed
  // carry one field fewer; pad them blank rather than abort the export.
  const rows = beatLog.map((r) => (r.length === 8 ? [...r, ''] : r).join(','));

  const nHead = header.split(',').length;
  const bad = rows.findIndex((r) => r.split(',').length !== nHead);
  if (bad !== -1) {
    fail(`Export aborted: beat row has ${rows[bad].split(',').length} fields, header has ${nHead}`);
    return false;
  }

  const blob = new Blob([[header, ...rows].join('\n')], { type: 'text/csv' });
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '');
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `echo_beats_${participant()}_v${visitNo() || 'x'}_${stamp}.csv`;
  link.click();
  URL.revokeObjectURL(link.href);

  // The rows are on disk, so the recovery mirror has done its job. Mirrors
  // the window export rather than holding the beats back, which would leave
  // a stale copy to be recovered after the real file already exists.
  try { localStorage.removeItem(beatsKey()); } catch { /* ignore */ }
  return true;
}

/* One action writes both files.
 *
 * The interval series and the window log are not alternatives. The windows
 * record what the device decided at the time; the intervals are the
 * measurement, and they are the only thing a corrected pipeline can be rerun
 * against. A separate button for the one that matters means the session where
 * someone forgets is the session that cannot be reanalysed, and that already
 * happened: the July and August recordings exist only as processed output
 * because the client of that era never wrote intervals at all, so they cannot
 * be reprocessed by anything, ever.
 *
 * Beats go first, so that if the browser blocks the second of two
 * programmatic downloads it is the replaceable file that is lost. Chrome may
 * ask once to allow multiple downloads from this origin; allow it. */
function exportCSV() {
  const wroteBeats = exportBeats();
  exportMotion();
  if (!sessionLog.length) {
    if (!wroteBeats) fail('Nothing to export.');
    return;
  }
  const header = ['timestamp', 'elapsed_s', 'participant', 'visit', 'phase',
                  'note', 'activity', 'mark',
                  'temp_f', 'humidity_pct', 'wbgt_f', 'setting', 'signal_quality', 'motion_mg', 'state', 'raw_state',
                  'confidence', 'readiness', 'trend', 'p_green', 'p_amber', 'p_red',
                  ...FEATURES, 'model'].join(',');
  const model = echo.modelId();
  const rows = sessionLog.map((r) =>
    [r.t, r.elapsed, r.pid, r.visit || '', r.phase || '',
     r.note, r.mode || '', r.mark || '',
     ...r.env, r.quality, r.motion ?? '', r.state, r.raw, r.conf, r.readiness, r.trend,
     ...r.votes, ...r.f, model].join(','));

  // Header and rows are built separately, so they can silently drift apart —
  // a missing field shifts every later column and the file still looks valid.
  const nHead = header.split(',').length;
  const bad = rows.findIndex((r) => r.split(',').length !== nHead);
  if (bad !== -1) {
    fail(`Export aborted: row has ${rows[bad].split(',').length} fields, header has ${nHead}`);
    return;
  }

  const blob = new Blob([[header, ...rows].join('\n')], { type: 'text/csv' });
  const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 16);
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `echo_windows_${participant()}_v${visitNo() || 'x'}_${stamp}.csv`;
  link.click();
  URL.revokeObjectURL(link.href);

  // Exported rows are safe on disk; the recovery copy has done its job.
  try { localStorage.removeItem(sessionKey()); } catch { /* ignore */ }
}

function switchParticipant() {
  // A new code is a new person: drop the session and re-enroll.
  stopSim();
  // Both logs are written while activePid still names the outgoing
  // participant. Only then does it move.
  persistSession();
  persistBeats();
  activePid = participant();

  sessionLog = [];
  beatLog = [];
  packetSeq = 0;
  sessionStart = null;
  beatCount = 0;
  history = [];
  lastState = -1;
  enrolledOnce = false;
  el.logged.textContent = '0';
  el.export.disabled = true;
  el.beats_logged.textContent = '0';
  el.exportbeats.disabled = true;
  el.beats.textContent = '0';
  el.trace.innerHTML = '';
  el.state.dataset.s = '-';
  el.state.textContent = 'Standby';
  el.stack.innerHTML =
    '<div class="empty">No classifications yet for this participant.</div>';
  echo.init(60000, ENROLL_MS);
  if (echo.baselined()) restoreBaseline();
  el.detail.textContent = `Participant ${participant()}`;
}

function fail(msg) {
  el.status.textContent = 'error';
  el.detail.textContent = msg;
}

/* ---------- boot ---------- */

document.querySelectorAll('.mode').forEach((b) =>
  b.addEventListener('click', () => setMode(b.dataset.mode)));

el.mark.addEventListener('click', markFromInput);
el.marklabel.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') markFromInput();
});
el.sampling.addEventListener('click', () =>
  (samplingOn ? stopSampling() : startSampling()));
el.pskip.addEventListener('click', () => {
  markEvent('prompt:skipped', true);
  dismissPrompt(true);
});
buildTagButtons();

/* Selecting a phase also sets the activity mode, because they are not
 * independent and asking an operator to keep two controls in agreement
 * during a maximal treadmill test is asking for them to disagree. */
const PHASE_MODE = { bruce: 'active', recovery: 'recovery' };
PHASES.forEach((name) => {
  const b = document.createElement('button');
  b.className = 'pbtn';
  b.dataset.phase = name;
  b.textContent = name === 'paced' ? 'paced breathing'
                : name === 'bruce' ? 'Bruce' : name;
  b.addEventListener('click', () => {
    phase = name;
    el.phasebtns.querySelectorAll('button')
      .forEach((x) => x.classList.toggle('on', x.dataset.phase === name));
    setMode(PHASE_MODE[name] || 'rest');
  });
  el.phasebtns.appendChild(b);
});

el.connect.addEventListener('click', connect);
el.sim.addEventListener('click', startSim);
el.forget.addEventListener('click', forgetBaseline);
el.export.addEventListener('click', exportCSV);
el.exportbeats.addEventListener('click', exportBeats);
el.discard.addEventListener('click', () => {
  if ((sessionLog.length || beatLog.length) &&
      !confirm(`Discard ${sessionLog.length} window rows and ${beatLog.length} beats?`)) return;
  clearSession();
  el.detail.textContent = 'Session cleared';
});
el.pid.addEventListener('change', switchParticipant);
el.savebase.addEventListener('click', downloadBaseline);
el.loadbase.addEventListener('click', () => el.basefile.click());
el.basefile.addEventListener('change', (e) => {
  if (e.target.files[0]) uploadBaseline(e.target.files[0]);
  e.target.value = '';
});

loadCore().then((ok) => {
  if (!ok) { el.connect.disabled = true; el.sim.disabled = true; el.forget.disabled = true; }
});
