/* ── tools/verify.mjs ────────────────────────────────────────────────
   Zero-dependency visual acceptance harness.

   Serves the project over HTTP, drives a real headless Chrome through
   the DevTools Protocol using Node's own global WebSocket and fetch,
   loads every scene, captures a PNG of each, and reports any console
   error or uncaught exception that occurred along the way.

     node tools/verify.mjs                 # all scenes
     node tools/verify.mjs --scene fluid   # one scene
     node tools/verify.mjs --serve         # just the static server
     node tools/verify.mjs --head          # visible browser window

   No puppeteer, no playwright, no test runner. The protocol is JSON
   over a socket; that is the entire dependency surface.
   ------------------------------------------------------------------ */

import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { readFile, writeFile, mkdir, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { extname, join, normalize, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = normalize(join(dirname(fileURLToPath(import.meta.url)), '..'));
const SHOTS = join(ROOT, 'tools', 'shots');

const args = process.argv.slice(2);
const flag = (name) => args.includes(`--${name}`);
const value = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : fallback;
};

const VIEWPORT = { width: 1600, height: 900 };

/**
 * Each shot is a hash route plus a settle time. Parameters go in the
 * URL because the app already treats the URL as its state — the same
 * mechanism that makes a scene shareable makes it testable.
 */
const SHOTS_PLAN = [
  { name: '01-intro',    hash: '#/flow',                        settle: 1200, intro: false },
  { name: '02-flow',     hash: '#/flow?count=512',              settle: 4500 },
  { name: '03-fluid',    hash: '#/fluid',                       settle: 5000 },
  { name: '04-march',    hash: '#/march',                       settle: 4000 },
  { name: '05-reaction', hash: '#/reaction',                    settle: 6000 },
  { name: '06-flow-alt', hash: '#/flow?palette=spectra&trail=0.96&count=512', settle: 4500 },
];

/* ═══ static server ═══════════════════════════════════════════════ */

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js':   'text/javascript; charset=utf-8',
  '.mjs':  'text/javascript; charset=utf-8',
  '.css':  'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg':  'image/svg+xml',
  '.png':  'image/png',
  '.ico':  'image/x-icon',
};

function serve() {
  const server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url, 'http://localhost');
      let path = decodeURIComponent(url.pathname);
      if (path.endsWith('/')) path += 'index.html';

      // Contain every request inside ROOT — this server is only ever
      // pointed at a local checkout, but path traversal is cheap to close.
      const file = normalize(join(ROOT, path));
      if (!file.startsWith(ROOT) || !existsSync(file)) {
        res.writeHead(404, { 'content-type': 'text/plain' });
        res.end('404');
        return;
      }

      const body = await readFile(file);
      res.writeHead(200, {
        'content-type': MIME[extname(file).toLowerCase()] ?? 'application/octet-stream',
        'cache-control': 'no-store',
      });
      res.end(body);
    } catch (err) {
      res.writeHead(500, { 'content-type': 'text/plain' });
      res.end(String(err));
    }
  });

  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port }));
  });
}

/* ═══ chrome ══════════════════════════════════════════════════════ */

function findChrome() {
  const candidates = [
    process.env.CHROME_PATH,
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
    join(process.env.LOCALAPPDATA ?? '', 'Google/Chrome/Application/chrome.exe'),
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
  ].filter(Boolean);
  return candidates.find((p) => existsSync(p));
}

async function launchChrome({ headless }) {
  const bin = findChrome();
  if (!bin) throw new Error('No Chrome or Edge found. Set CHROME_PATH.');

  const port = 9000 + Math.floor(Math.random() * 900);
  const profile = join(SHOTS, `.profile-${port}`);

  const flags = [
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${profile}`,
    `--window-size=${VIEWPORT.width},${VIEWPORT.height}`,
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-extensions',
    '--disable-background-timer-throttling',
    '--disable-backgrounding-occluded-windows',
    '--disable-renderer-backgrounding',
    // Headless Chrome will happily fall back to a software rasteriser,
    // which cannot run WebGL2 at all unless these are set.
    '--ignore-gpu-blocklist',
    '--enable-unsafe-swiftshader',
    'about:blank',
  ];
  if (headless) flags.unshift('--headless=new', '--hide-scrollbars');

  const child = spawn(bin, flags, { stdio: 'ignore', detached: false });

  // Poll the DevTools endpoint rather than guessing a startup delay.
  const deadline = Date.now() + 20000;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`http://127.0.0.1:${port}/json/version`);
      if (res.ok) return { child, port, profile, bin };
    } catch { /* not up yet */ }
    await sleep(150);
  }
  child.kill();
  throw new Error('Chrome did not expose a DevTools endpoint in time');
}

/* ═══ minimal CDP client ══════════════════════════════════════════ */

class CDP {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.pending = new Map();
    this.listeners = new Map();

    ws.addEventListener('message', (e) => {
      const msg = JSON.parse(e.data);
      if (msg.id !== undefined) {
        const p = this.pending.get(msg.id);
        if (!p) return;
        this.pending.delete(msg.id);
        msg.error ? p.reject(new Error(`${msg.error.message} (${msg.error.code})`)) : p.resolve(msg.result);
      } else {
        for (const fn of this.listeners.get(msg.method) ?? []) fn(msg.params);
      }
    });
  }

  static async attach(port) {
    const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    const page = targets.find((t) => t.type === 'page');
    if (!page) throw new Error('No page target');
    const ws = new WebSocket(page.webSocketDebuggerUrl);
    await new Promise((resolve, reject) => {
      ws.addEventListener('open', resolve, { once: true });
      ws.addEventListener('error', reject, { once: true });
    });
    return new CDP(ws);
  }

  send(method, params = {}) {
    const id = ++this.id;
    this.ws.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      setTimeout(() => {
        if (this.pending.delete(id)) reject(new Error(`CDP timeout: ${method}`));
      }, 60000);
    });
  }

  on(event, fn) {
    if (!this.listeners.has(event)) this.listeners.set(event, []);
    this.listeners.get(event).push(fn);
  }

  /** Evaluate in the page and return the value, unwrapping promises. */
  async eval(expression) {
    const { result, exceptionDetails } = await this.send('Runtime.evaluate', {
      expression,
      awaitPromise: true,
      returnByValue: true,
    });
    if (exceptionDetails) throw new Error(exceptionDetails.exception?.description ?? exceptionDetails.text);
    return result.value;
  }

  close() { this.ws.close(); }
}

/* ═══ run ═════════════════════════════════════════════════════════ */

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  const { server, port } = await serve();
  const base = `http://127.0.0.1:${port}/index.html`;
  console.log(`▸ serving ${ROOT}\n  ${base}`);

  if (flag('serve')) {
    console.log('  (--serve: press Ctrl+C to stop)');
    return;
  }

  // Only wipe the gallery on a full scene run. Every other mode is an
  // iteration step, and blowing away the other shots each time makes
  // comparing them impossible.
  const isFullRun = !value('scene', null) && !value('eval', null)
    && !flag('interact') && !flag('responsive');
  if (isFullRun) {
    await rm(SHOTS, { recursive: true, force: true }).catch(() => {});
  }
  await mkdir(SHOTS, { recursive: true });

  const chrome = await launchChrome({ headless: !flag('head') });
  console.log(`▸ chrome ${chrome.bin.split(/[\\/]/).pop()} on :${chrome.port}`);

  const cdp = await CDP.attach(chrome.port);
  const problems = [];

  cdp.on('Runtime.consoleAPICalled', ({ type, args }) => {
    if (type !== 'error' && type !== 'warning') return;
    const text = args.map((a) => a.value ?? a.description ?? a.type).join(' ');
    problems.push({ kind: type, text });
  });
  cdp.on('Runtime.exceptionThrown', ({ exceptionDetails }) => {
    problems.push({
      kind: 'exception',
      text: exceptionDetails.exception?.description ?? exceptionDetails.text,
    });
  });
  cdp.on('Log.entryAdded', ({ entry }) => {
    if (entry.level === 'error') problems.push({ kind: `log/${entry.source}`, text: entry.text });
  });

  await cdp.send('Page.enable');
  await cdp.send('Runtime.enable');
  await cdp.send('Log.enable');
  await cdp.send('Emulation.setDeviceMetricsOverride', {
    ...VIEWPORT, deviceScaleFactor: 1, mobile: false,
  });

  // --eval: load one route, settle, then run an expression in the page.
  // The quickest way to interrogate live GPU state from here.
  let probe = value('eval', null);
  if (probe && existsSync(probe)) probe = await readFile(probe, 'utf8');
  if (probe) {
    const hash = value('at', '#/flow');
    await cdp.send('Page.navigate', { url: base + hash });
    await waitFor(cdp, `document.documentElement.dataset.boot`, 'ready', 20000);
    await cdp.eval(`document.getElementById('intro-enter').click(); true`);
    await sleep(Number(value('settle', 3000)));
    console.dir(await cdp.eval(`(() => { ${probe} })()`), { depth: 8 });
    for (const p of problems) console.log(`  ${p.kind}: ${truncate(p.text, 600)}`);
    cdp.close(); chrome.child.kill(); server.close();
    await sleep(200);
    process.exit(0);
  }

  if (flag('responsive')) {
    // The layout has to survive a phone as well as a desktop, and the
    // pixel budget has to survive a 3× device ratio.
    const viewports = [
      { name: 'r1-phone',   width: 390,  height: 844,  dpr: 3, mobile: true },
      { name: 'r2-tablet',  width: 820,  height: 1180, dpr: 2, mobile: true },
      { name: 'r3-laptop',  width: 1280, height: 720,  dpr: 2, mobile: false },
      { name: 'r4-wide',    width: 2560, height: 1080, dpr: 1, mobile: false },
    ];
    console.log('▸ responsive');
    await cdp.send('Page.navigate', { url: `${base}#/reaction` });
    await waitFor(cdp, 'document.documentElement.dataset.boot', 'ready', 20000);
    await cdp.eval(`document.getElementById('intro-enter').click(); true`);

    for (const v of viewports) {
      await cdp.send('Emulation.setDeviceMetricsOverride', {
        width: v.width, height: v.height, deviceScaleFactor: v.dpr, mobile: v.mobile,
      });
      await sleep(2600);
      const info = await cdp.eval(`(() => {
        const c = document.getElementById('stage');
        return {
          buffer: c.width + '×' + c.height,
          pixels: c.width * c.height,
          bodyScrollX: document.documentElement.scrollWidth > document.documentElement.clientWidth,
          panelVisible: getComputedStyle(document.getElementById('panel')).opacity,
          fps: document.getElementById('hud-fps').textContent,
        };
      })()`);
      await shot(cdp, v.name);
      // The app clamps DPR down to 1 but never below, so a very wide
      // 1× display legitimately exceeds the nominal pixel budget.
      const budgetOk = info.pixels <= Math.max(2.7e6, v.width * v.height * 1.02);
      console.log(`  ${budgetOk && !info.bodyScrollX ? '✓' : '✗'} ${v.name.padEnd(10)}` +
        ` css ${v.width}×${v.height}@${v.dpr}x → buffer ${info.buffer}` +
        ` (${(info.pixels / 1e6).toFixed(2)} Mpx)` +
        (info.bodyScrollX ? '  ⚠ horizontal scroll' : '') +
        (budgetOk ? '' : '  ⚠ over pixel budget'));
    }
    cdp.close(); chrome.child.kill(); server.close();
    await sleep(200);
    process.exit(0);
  }

  if (flag('interact')) {
    await interact(cdp, base, problems);
    cdp.close(); chrome.child.kill(); server.close();
    await sleep(200);
    process.exit(0);
  }

  const only = value('scene', null);
  const plan = only ? SHOTS_PLAN.filter((s) => s.hash.includes(only)) : SHOTS_PLAN;
  const report = [];

  for (const shot of plan) {
    const url = base + shot.hash;
    process.stdout.write(`▸ ${shot.name.padEnd(12)} `);

    const before = problems.length;
    await cdp.send('Page.navigate', { url });
    try {
      await waitFor(cdp, `document.documentElement.dataset.boot`, 'ready', 20000);
    } catch (err) {
      // A boot that never completes is almost always a module or shader
      // error the page already logged — surface it instead of the timeout.
      console.log('✗ boot failed');
      for (const p of problems) console.log(`     ${p.kind}: ${truncate(p.text, 1400)}`);
      throw err;
    }

    if (shot.intro !== false) {
      await cdp.eval(`document.getElementById('intro-enter').click(); true`);
    }
    await sleep(shot.settle);

    const diag = await cdp.eval(`(() => {
      const c = document.getElementById('stage');
      const gl = c.getContext('webgl2');
      return {
        boot: document.documentElement.dataset.boot,
        scene: document.documentElement.dataset.scene,
        canvas: c.width + '×' + c.height,
        fps: document.getElementById('hud-fps').textContent,
        caption: document.getElementById('caption-title').textContent,
        renderer: (() => { const d = gl.getExtension('WEBGL_debug_renderer_info');
          return d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER); })(),
        glError: gl.getError(),
      };
    })()`);

    const { data } = await cdp.send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false });
    const file = join(SHOTS, `${shot.name}.png`);
    await writeFile(file, Buffer.from(data, 'base64'));

    // A canvas that is entirely one colour has not rendered anything.
    const stats = await cdp.eval(`(() => {
      const c = document.getElementById('stage');
      const s = document.createElement('canvas');
      s.width = 64; s.height = 36;
      const x = s.getContext('2d');
      x.drawImage(c, 0, 0, 64, 36);
      const d = x.getImageData(0, 0, 64, 36).data;
      let min = 255, max = 0, sum = 0, n = 0;
      for (let i = 0; i < d.length; i += 4) {
        const l = (d[i] * 0.299 + d[i+1] * 0.587 + d[i+2] * 0.114);
        min = Math.min(min, l); max = Math.max(max, l); sum += l; n++;
      }
      return { min: Math.round(min), max: Math.round(max), mean: +(sum / n).toFixed(1) };
    })()`);

    const newProblems = problems.slice(before);
    const blank = stats.max - stats.min < 4;
    const ok = !blank && newProblems.length === 0 && diag.glError === 0;

    report.push({ ...shot, diag, stats, problems: newProblems, ok, file });
    console.log(
      `${ok ? '✓' : '✗'}  ${diag.canvas}  fps ${String(diag.fps).padStart(3)}  ` +
      `luma ${String(stats.min).padStart(3)}–${String(stats.max).padEnd(3)} (μ${stats.mean})` +
      (blank ? '  ⚠ BLANK' : '') +
      (diag.glError ? `  ⚠ glError ${diag.glError}` : '') +
      (newProblems.length ? `  ⚠ ${newProblems.length} problem(s)` : '')
    );
    for (const p of newProblems) console.log(`     ${p.kind}: ${truncate(p.text, 260)}`);
  }

  const renderer = report[0]?.diag.renderer ?? 'unknown';
  console.log(`\n▸ renderer: ${renderer}`);
  console.log(`▸ shots:    ${SHOTS}`);
  const failed = report.filter((r) => !r.ok);
  console.log(failed.length ? `▸ FAILED:   ${failed.map((r) => r.name).join(', ')}` : '▸ all scenes rendered clean');

  cdp.close();
  chrome.child.kill();
  server.close();
  await sleep(300);
  process.exit(failed.length ? 1 : 0);
}

/* ═══ interaction acceptance ══════════════════════════════════════
   Screenshots prove a scene renders. They prove nothing about whether
   it can be *used*. This drives real input events through the DevTools
   protocol — the browser cannot tell them from a hand on the mouse —
   and asserts on the state that comes out the other side.             */

async function interact(cdp, base, problems) {
  const results = [];
  const check = (name, pass, detail = '') => {
    results.push({ name, pass, detail });
    console.log(`  ${pass ? '✓' : '✗'} ${name}${detail ? `  — ${detail}` : ''}`);
  };

  const key = async (k, { code, vk, mods = 0, text } = {}) => {
    const common = {
      key: k, code: code ?? `Key${k.toUpperCase()}`,
      windowsVirtualKeyCode: vk ?? k.toUpperCase().charCodeAt(0),
      modifiers: mods,
    };
    await cdp.send('Input.dispatchKeyEvent', { type: text ? 'keyDown' : 'rawKeyDown', text, ...common });
    await cdp.send('Input.dispatchKeyEvent', { type: 'keyUp', ...common });
    await sleep(180);
  };

  const mouse = (type, x, y, extra = {}) =>
    cdp.send('Input.dispatchMouseEvent', { type, x, y, button: 'left', clickCount: 1, ...extra });

  /** Returns a mid-drag sample of the app's own pointer state. */
  const drag = async (from, to, steps = 26) => {
    // Move to the start first: a real mouse is always somewhere before
    // it is pressed, and dispatching a press at a fresh coordinate skips
    // the hover state the page may be relying on.
    await mouse('mouseMoved', from[0], from[1], { buttons: 0 });
    await sleep(40);
    await mouse('mousePressed', from[0], from[1], { buttons: 1 });
    await sleep(40);
    const pressed = await cdp.eval(`__aether.pointer.down`);
    let mid = { pressed };
    for (let i = 1; i <= steps; i++) {
      const t = i / steps;
      await mouse('mouseMoved',
        from[0] + (to[0] - from[0]) * t,
        from[1] + (to[1] - from[1]) * t, { buttons: 1 });
      if (i === Math.floor(steps / 2)) {
        Object.assign(mid, await cdp.eval(`({ down: __aether.pointer.down,
                                 points: __aether.pointer._points.size,
                                 x: +__aether.pointer.x.toFixed(3),
                                 y: +__aether.pointer.y.toFixed(3) })`));
      }
      await sleep(16);
    }
    await mouse('mouseReleased', to[0], to[1], { buttons: 0 });
    return mid;
  };

  console.log('▸ interaction');

  await cdp.send('Page.navigate', { url: `${base}#/fluid` });
  await waitFor(cdp, 'document.documentElement.dataset.boot', 'ready', 20000);

  /* intro */
  await cdp.eval(`document.getElementById('intro-enter').click(); true`);
  await sleep(600);
  check('intro dismisses', await cdp.eval(`document.body.classList.contains('intro-done')`));

  /* pointer painting into the fluid */
  await cdp.eval(`__aether.panel.setValues({ auto: false }, { notify: true }); true`);
  await sleep(300);
  const before = await cdp.eval(`__aether.scene.splatCount`);
  const mid = await drag([420, 700], [1050, 260]);
  await sleep(400);
  const after = await cdp.eval(`__aether.scene.splatCount`);
  check('canvas receives pointer events', mid?.down === true, JSON.stringify(mid));
  // Auto-injection is off, so every splat here came from the drag.
  check('pointer drag injects dye', after > before + 10, `${before} → ${after} splats`);
  await cdp.eval(`__aether.panel.setValues({ auto: true }, { notify: true }); true`);
  await shot(cdp, 'i1-fluid-drag');

  /* command palette: open, fuzzy-filter, run */
  await key('k', { mods: 2 /* Ctrl */, vk: 75 });
  check('Ctrl+K opens palette', await cdp.eval(`document.getElementById('cmdk').open`));
  for (const ch of 'sdf') await key(ch, { text: ch });
  const matches = await cdp.eval(`document.querySelectorAll('#cmdk-list li').length`);
  const top = await cdp.eval(`document.querySelector('#cmdk-list li .cmdk__label')?.textContent ?? ''`);
  check('fuzzy search filters', matches > 0 && matches < 12, `${matches} hits, top = "${top}"`);
  await shot(cdp, 'i2-cmdk');

  await key('Enter', { code: 'Enter', vk: 13 });
  await sleep(900);
  check('palette navigates', await cdp.eval(`__aether.router.current`) === 'march',
    `now at ${await cdp.eval(`__aether.router.current`)}`);

  /* the swimmer follows the pointer, and survives a scene change */
  await cdp.eval(`__aether.panel.setValues({ agentMode: 'follow' }, { notify: true }); true`);

  // Alignment, not distance. The pointer un-projects to a point that can
  // sit well outside the creature's own bounds, so "did it get closer"
  // has a floor it can never cross — but "is it on that side" is exactly
  // the thing a viewer judges.
  const alignment = () => cdp.eval(`(() => {
    const a = __aether.agent;
    const hl = Math.hypot(a.nodes[0], a.nodes[1]) || 1;
    const al = Math.hypot(a._aim[0], a._aim[1]) || 1;
    return (a.nodes[0] * a._aim[0] + a.nodes[1] * a._aim[1]) / (hl * al);
  })()`);

  const corner = [250, 740];
  await mouse('mouseMoved', corner[0], corner[1], { buttons: 0 });
  await sleep(300);
  const before1 = await alignment();
  for (let i = 0; i < 16; i++) {
    await mouse('mouseMoved', corner[0], corner[1], { buttons: 0 });
    await sleep(130);
  }
  const after1 = await alignment();
  check('swimmer turns toward the pointer', after1 > 0.55,
    `alignment ${before1.toFixed(2)} → ${after1.toFixed(2)}`);
  await shot(cdp, 'i6-swimmer-follow');

  const before2 = await cdp.eval(`[...__aether.agent.nodes.slice(0, 3)]`);
  await cdp.eval(`document.querySelector('.tab[data-id="flow"]').click(); true`);
  await sleep(500);
  const after2 = await cdp.eval(`[...__aether.agent.nodes.slice(0, 3)]`);
  const drift = Math.hypot(after2[0] - before2[0], after2[1] - before2[1], after2[2] - before2[2]);
  check('swimmer persists across scenes', drift < 0.6,
    `moved ${drift.toFixed(3)} in canonical units while the scene swapped`);

  await cdp.eval(`document.querySelector('.tab[data-id="march"]').click(); true`);
  await sleep(900);

  /* camera orbit by dragging */
  const yaw0 = await cdp.eval(`__aether.scene.yaw`);
  const midOrbit = await drag([700, 450], [980, 380]);
  const yaw1 = await cdp.eval(`__aether.scene.yaw`);
  check('drag orbits the camera', Math.abs(yaw1 - yaw0) > 0.2,
    `yaw ${yaw0.toFixed(2)} → ${yaw1.toFixed(2)}, mid ${JSON.stringify(midOrbit)}`);

  /* keyboard scene switching + URL is state */
  await key('1', { code: 'Digit1', vk: 49 });
  await sleep(700);
  check('number keys switch scene', await cdp.eval(`__aether.router.current`) === 'flow');

  await cdp.eval(`__aether.panel.setValues({ palette: 'ember' }, { notify: true }); true`);
  await sleep(300);
  const url = await cdp.eval(`location.hash`);
  check('params round-trip into the URL', url.includes('palette=ember'), url);

  /* pause / step */
  await key(' ', { code: 'Space', vk: 32 });
  check('space pauses', await cdp.eval(`__aether.clock.paused`));
  const f0 = await cdp.eval(`__aether.clock.time`);
  await sleep(400);
  const f1 = await cdp.eval(`__aether.clock.time`);
  check('paused clock does not advance', f0 === f1, `t = ${f1.toFixed(3)}`);
  await key('.', { code: 'Period', vk: 190 });
  check('period steps one frame', (await cdp.eval(`__aether.clock.time`)) > f1);
  await key(' ', { code: 'Space', vk: 32 });

  /* zen mode + help */
  await key('z', { vk: 90 });
  check('zen mode hides chrome', await cdp.eval(`document.body.classList.contains('zen')`));
  await shot(cdp, 'i3-zen');
  await key('z', { vk: 90 });

  await key('?', { code: 'Slash', vk: 191, mods: 8 });
  check('? opens shortcuts', await cdp.eval(`document.getElementById('help').open`));
  await shot(cdp, 'i4-help');
  await key('Escape', { code: 'Escape', vk: 27 });

  /* tabs */
  await cdp.eval(`document.querySelector('.tab[data-id="reaction"]').click(); true`);
  await sleep(900);
  check('tab click switches scene', await cdp.eval(`__aether.router.current`) === 'reaction');
  const inkW = await cdp.eval(`document.getElementById('tabs-ink').style.width`);
  check('tab indicator tracks selection', parseFloat(inkW) > 20, `ink = ${inkW}`);

  /* every scene has been created and torn down at least once by now */
  check('no console errors during interaction', problems.length === 0,
    problems.map((p) => p.text.slice(0, 120)).join(' | '));
  await shot(cdp, 'i5-reaction-after-switching');

  const failed = results.filter((r) => !r.pass);
  console.log(failed.length ? `\n▸ ${failed.length}/${results.length} FAILED`
                            : `\n▸ all ${results.length} interaction checks passed`);
  if (failed.length) process.exitCode = 1;
}

async function shot(cdp, name) {
  const { data } = await cdp.send('Page.captureScreenshot', { format: 'png' });
  await writeFile(join(SHOTS, `${name}.png`), Buffer.from(data, 'base64'));
}

async function waitFor(cdp, expression, expected, timeout) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    try {
      const v = await cdp.eval(expression);
      if (v === expected) return v;
      if (v === 'failed') throw new Error('App reported boot failure');
    } catch (err) {
      if (String(err).includes('boot failure')) throw err;
    }
    await sleep(200);
  }
  throw new Error(`Timed out waiting for ${expression} === ${expected}`);
}

function truncate(s, n) { return s.length > n ? `${s.slice(0, n)}…` : s; }

main().catch((err) => { console.error('\n✗', err); process.exit(1); });
