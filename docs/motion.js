/* Echo Edge motion channel: Polar H10 accelerometer over the PMD service.
 *
 * Why this exists: the readiness model cannot tell exertion from autonomic
 * stress, because both raise heart rate and lower HRV (first live session,
 * Aug 3 2026: walking into the living room read RED). Movement that begins
 * before heart rate rises points to exertion; a rise with no movement ahead
 * of it does not. This file records the movement.
 *
 * Two outputs, kept separate on purpose:
 *   1. Raw samples (x, y, z in milli-g), exported as their own file. As with
 *      the beat log, the samples are the measurement and anything derived
 *      from them can be recomputed later.
 *   2. motion_mg: a one-second summary written beside every beat and window,
 *      so the existing exports can be read against movement without a join.
 *
 * Protocol reference: Polar BLE SDK, PMD (Polar Measurement Data) service.
 * The heart-rate path does not depend on anything here. If the strap refuses
 * the accelerometer stream, Echo keeps running and motion_mg stays blank.
 */

const PMD_SERVICE = 'fb005c80-02e7-f387-1cad-8acd2d8df0c8';
const PMD_CONTROL = 'fb005c81-02e7-f387-1cad-8acd2d8df0c8';
const PMD_DATA    = 'fb005c82-02e7-f387-1cad-8acd2d8df0c8';
const PMD_ACC = 0x02;
const ACC_HZ = 50;   // H10 offers 25, 50, 100, 200. 50 Hz is ample for gait and posture.

/* Start-measurement request: opcode 0x02, type ACC, then settings as
 * [setting id, value count, uint16 little-endian]:
 * sample rate 50, resolution 16 bit, range 8 g. */
const ACC_START = new Uint8Array([
  0x02, PMD_ACC,
  0x00, 0x01, ACC_HZ & 0xff, ACC_HZ >> 8,
  0x01, 0x01, 0x10, 0x00,
  0x02, 0x01, 0x08, 0x00,
]);

const motion = {
  active: false,
  samples: [],        // [elapsed_s, sensor_ts_ns, x, y, z]
  recent: [],         // vector magnitudes, last second only
  lastMg: null,       // current one-second summary
  frames: 0,
  unsupported: 0,     // frames in a format this parser does not decode
};

/* Decode one data notification. Pure function so it can be tested without a
 * strap. Returns { ts, xyz: [[x,y,z], ...] } or null if not decodable. */
function parseAccFrame(dv) {
  if (dv.byteLength < 10 || dv.getUint8(0) !== PMD_ACC) return null;
  const ts = dv.getBigUint64(1, true);
  const frameType = dv.getUint8(9);
  if (frameType & 0x80) return { ts, xyz: null };   // delta-compressed: not handled
  const width = { 0x00: 1, 0x01: 2, 0x02: 3 }[frameType];
  if (!width) return { ts, xyz: null };
  const read = (off) => {
    if (width === 1) return dv.getInt8(off);
    if (width === 2) return dv.getInt16(off, true);
    const v = dv.getUint8(off) | (dv.getUint8(off + 1) << 8) | (dv.getUint8(off + 2) << 16);
    return v & 0x800000 ? v - 0x1000000 : v;
  };
  const xyz = [];
  for (let off = 10; off + 3 * width <= dv.byteLength; off += 3 * width) {
    xyz.push([read(off), read(off + width), read(off + 2 * width)]);
  }
  return { ts, xyz };
}

/* Mean absolute deviation of vector magnitude over the last second.
 * Magnitude is used so the figure does not depend on how the strap is
 * rotated on the chest; subtracting the window mean removes gravity and any
 * calibration offset. Seated still reads a few mg; walking reads hundreds. */
function summarise(mags) {
  if (!mags.length) return null;
  const mean = mags.reduce((a, b) => a + b, 0) / mags.length;
  return mags.reduce((a, m) => a + Math.abs(m - mean), 0) / mags.length;
}

function onAccData(e) {
  const frame = parseAccFrame(e.target.value);
  if (!frame) return;
  motion.frames++;
  if (!frame.xyz) { motion.unsupported++; showMotion(); return; }

  const elapsed = sessionStart === null ? '' : ((Date.now() - sessionStart) / 1000).toFixed(2);
  for (const [x, y, z] of frame.xyz) {
    motion.samples.push([elapsed, frame.ts.toString(), x, y, z]);
    motion.recent.push(Math.hypot(x, y, z));
  }
  if (motion.recent.length > ACC_HZ) motion.recent.splice(0, motion.recent.length - ACC_HZ);
  motion.lastMg = summarise(motion.recent);
  showMotion();
}

function showMotion() {
  const out = document.getElementById('motion');
  if (!out) return;
  if (!motion.active) { out.textContent = '—'; return; }
  if (motion.unsupported && motion.unsupported === motion.frames) {
    out.textContent = 'format unsupported';
    return;
  }
  out.textContent = motion.lastMg === null ? 'waiting' : `${motion.lastMg.toFixed(0)} mg`;
}

/* Value written into the beat and window logs. Blank, not zero, when there
 * is no accelerometer: zero would claim the wearer was still. */
function motionNow() {
  return motion.lastMg === null ? '' : motion.lastMg.toFixed(1);
}

async function startMotion(server) {
  try {
    const svc = await server.getPrimaryService(PMD_SERVICE);
    const ctrl = await svc.getCharacteristic(PMD_CONTROL);
    const data = await svc.getCharacteristic(PMD_DATA);

    // The strap answers on the control point; listen before asking.
    let reply = null;
    const replied = new Promise((resolve) => {
      ctrl.addEventListener('characteristicvaluechanged', (e) => {
        reply = e.target.value;
        resolve();
      }, { once: true });
    });
    await ctrl.startNotifications();
    await data.startNotifications();
    data.addEventListener('characteristicvaluechanged', onAccData);

    await ctrl.writeValueWithResponse(ACC_START);
    await Promise.race([replied, new Promise((r) => setTimeout(r, 3000))]);

    // Reply: 0xF0, opcode, type, error code. Zero is success.
    if (reply && reply.byteLength >= 4 && reply.getUint8(3) !== 0) {
      throw new Error(`strap refused accelerometer (code ${reply.getUint8(3)})`);
    }
    motion.active = true;
    showMotion();
    return true;
  } catch (err) {
    motion.active = false;
    const out = document.getElementById('motion');
    if (out) out.textContent = 'unavailable';
    console.warn('[motion]', err && err.message ? err.message : err);
    return false;
  }
}

function exportMotion() {
  if (!motion.samples.length) return false;
  const header = 'elapsed_s,sensor_ts_ns,x_mg,y_mg,z_mg';
  const rows = motion.samples.map((r) => r.join(','));
  const blob = new Blob([[header, ...rows].join('\n')], { type: 'text/csv' });
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '');
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `echo_motion_${participant()}_v${visitNo() || 'x'}_${stamp}.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
  return true;
}

if (typeof module !== 'undefined') module.exports = { parseAccFrame, summarise, ACC_START };
