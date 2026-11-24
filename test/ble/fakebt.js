// Fake Web Bluetooth: one H10-like strap, scriptable failures.
(() => {
  const bt = window.__bt = { mode: 'ok', connectCalls: 0, inFlight: 0, maxInFlight: 0, packets: 0, log: [] };
  let notifTimer = null;
  class Char extends EventTarget {
    async startNotifications() {
      if (!dev.gatt.connected) throw new DOMException('GATT Server is disconnected.', 'NetworkError');
      clearInterval(notifTimer);
      notifTimer = setInterval(() => {
        if (bt.zombie) return;
        const b = new DataView(new ArrayBuffer(4));
        b.setUint8(0, 0x10); b.setUint8(1, 70);
        b.setUint16(2, Math.round((850 + Math.random() * 40) / 1000 * 1024), true);
        this.value = b; bt.packets++;
        this.dispatchEvent(new Event('characteristicvaluechanged'));
      }, 50);
      return this;
    }
  }
  const ch = new Char();
  const server = { async getPrimaryService() {
    if (bt.mode === 'failService') { bt.mode = 'ok'; throw new DOMException('GATT operation failed for unknown reason.', 'NetworkError'); }
    return { async getCharacteristic() { return ch; } };
  } };
  const gatt = {
    connected: false,
    connect() {
      bt.connectCalls++; bt.inFlight++; bt.maxInFlight = Math.max(bt.maxInFlight, bt.inFlight);
      bt.log.push('connect:' + bt.mode);
      if (bt.mode === 'hang') { bt.mode = 'ok'; return new Promise((res, rej) => { bt.abortPending = () => { bt.inFlight--; rej(new DOMException('aborted', 'AbortError')); }; }); }
      return new Promise((res) => setTimeout(() => { bt.inFlight--; gatt.connected = true; res(server); }, 30));
    },
    disconnect() {
      bt.log.push('disconnect');
      if (bt.abortPending) { const a = bt.abortPending; bt.abortPending = null; a(); }
      if (!gatt.connected) return;
      gatt.connected = false; clearInterval(notifTimer);
      setTimeout(() => dev.dispatchEvent(new Event('gattserverdisconnected')), 0);
    },
  };
  const dev = new EventTarget();
  Object.assign(dev, { id: 'fake-h10', name: 'Polar H10 FAKE', gatt });
  bt.drop = () => { gatt.connected = false; clearInterval(notifTimer); dev.dispatchEvent(new Event('gattserverdisconnected')); };
  bt.dev = dev;
  Object.defineProperty(navigator, 'bluetooth', { value: { requestDevice: async () => (bt.cancelChooser ? (bt.cancelChooser = false, Promise.reject(new DOMException('cancelled', 'NotFoundError'))) : dev) } });
})();
