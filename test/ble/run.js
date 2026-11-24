const { chromium } = require('playwright');
const path = require('path');
const wait = (ms) => new Promise(r => setTimeout(r, ms));
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  page.on('pageerror', e => console.log('PAGEERROR', e.message));
  await page.addInitScript({ path: path.join(__dirname, 'fakebt.js') });
  await page.goto(process.env.ECHO_URL || 'http://localhost:8765/index.html');
  await page.waitForFunction(() => document.getElementById('model').textContent !== '—', null, { timeout: 15000 }).catch(() => {});
  await page.fill('#pid', 'T01');
  await page.evaluate(() => { const v = document.getElementById('visit'); if (v) { v.value = '1'; v.dispatchEvent(new Event('change')); v.dispatchEvent(new Event('input')); } });
  await page.click('.pbtn[data-phase="seated"]');
  const st = () => page.evaluate(() => ({ status: document.getElementById('status').textContent, detail: document.getElementById('detail').textContent, packets: __bt.packets, calls: __bt.connectCalls, maxInFlight: __bt.maxInFlight, connected: __bt.dev.gatt.connected }));
  const scenario = process.argv[2];
  await page.click('#connect'); await wait(600);
  console.log('after connect', await st());
  if (scenario === 'hang') { await page.evaluate(() => { __bt.mode = 'hang'; __bt.drop(); }); }
  if (scenario === 'zombie') { await page.evaluate(() => { __bt.zombie = true; }); }
  if (scenario === 'failService') { await page.evaluate(() => { __bt.mode = 'failService'; __bt.drop(); }); }
  if (scenario === 'repair') { await page.evaluate(() => { __bt.drop(); }); await wait(200); await page.click('#connect'); }
  if (scenario === 'cancel') { await page.evaluate(() => { __bt.drop(); __bt.cancelChooser = true; }); await wait(200); await page.click('#connect'); }
  for (let i = 0; i < 6; i++) { await wait(5000); if (scenario==='zombie' && i===2) await page.evaluate(() => { __bt.zombie = false; }); console.log(`t+${(i+1)*5}s`, await st()); }
  console.log(await page.evaluate(() => __bt.log.join(' ')));
  await browser.close();
})();
