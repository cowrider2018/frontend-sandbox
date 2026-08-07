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
import { readFile, writeFile, mkdir, rm, readdir } from 'node:fs/promises';
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
  { name: '01-intro',  hash: '#/march',                                   settle: 1400, intro: false },
  { name: '02-scene',  hash: '#/march',                                   settle: 8000 },
  { name: '03-full',   hash: '#/march?scale=1&steps=160&spin=0&taa=0.9',  settle: 9000 },
  // Deterministic reference: frozen clock, and every quality knob pinned
  // to what the pre-optimisation build had hard-coded. Two builds
  // rendering this must agree pixel for pixel.
  { name: '00-ref',    hash: '#/march?scale=1&spin=0&taa=0.9', settle: 4000, freeze: 8.0 },
  // `click` fires a burst at a sphere's centre and waits, so the ring has
  // travelled far enough across the surface to be visible in a still.
  { name: '04-impact', hash: '#/march?spin=0&scale=1&rippleAmp=0.05&rippleFreq=13',
    settle: 5000, click: true, after: 260 },
  { name: '05-ripple', hash: '#/march?spin=0&scale=1&rippleAmp=0.052&rippleFreq=12&rippleSpeed=0.95',
    settle: 5000, click: true, after: 700 },
  // Enough dissipation to open a gap in the ring and eat into a sphere.
  { name: '07-erode',  hash: '#/march?spin=0&scale=1&erode=0.4&rippleAmp=0.09&rippleLife=3',
    settle: 5000, click: true, after: 700 },

  // One impact, photographed at four points in its life. Clock frozen
  // and the age pinned, so these four frames differ by the age alone.
  ...[0.15, 0.35, 0.55, 0.78].map((age, i) => ({
    name: `08-heal-${'abcd'[i]}`,
    hash: '#/march?spin=0&scale=1&erode=0.42&rippleAmp=0.02&taa=0.9',
    settle: 3500,
    freeze: 8.0,
    click: true,
    after: 200,
    poke: `const s = __aether.scene; s._rippleAge[0] = ${age}; s.flash = 0; return true;`,
  })),
  { name: '06-lit',    hash: '#/march?light=0.18,0.62&tint=rose&ao=1&shadow=1', settle: 8000 },
];

/**
 * CSS-pixel position of a sphere's centre. Aiming a synthetic click by
 * eye at "the middle of the screen" lands on empty sky often enough to
 * make a test lie about the feature it is checking.
 */
const BALL_ON_SCREEN = `(() => {
  const s = __aether.scene, b = s.basis, focal = 1.5;
  const rel = [s.ballPos[0] - b.pos[0], s.ballPos[1] - b.pos[1], s.ballPos[2] - b.pos[2]];
  const dot = (v) => rel[0] * v[0] + rel[1] * v[1] + rel[2] * v[2];
  const vx = dot(b.right), vy = dot(b.up), vz = dot(b.fwd);
  const aspect = s.width / s.height;
  return [
    ((vx * focal / aspect / vz) * 0.5 + 0.5) * innerWidth,
    (0.5 - (vy * focal / vz) * 0.5) * innerHeight,
  ];
})()`;

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

async function launchChrome({ headless, gpu }) {
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
  ];

  // On a laptop with two GPUs, Chrome picks one for the whole browser
  // process. `powerPreference: 'high-performance'` on the context is a
  // hint the page makes *after* that choice has already been made, so it
  // cannot move you to the discrete card on its own.
  if (gpu === 'high') flags.push('--force-high-performance-gpu');
  if (gpu === 'low') flags.push('--force-low-power-gpu');

  flags.push('about:blank');
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

/**
 * Shaders live inside JS template literals, so a back-tick anywhere in
 * GLSL — including in a comment — silently ends the literal and turns
 * the rest of the shader into JavaScript. The failure surfaces as a
 * parse error on some unrelated identifier fifty lines later, which is
 * a genuinely bad way to spend ten minutes. This catches it in a
 * millisecond, before a browser is even launched.
 */
async function lintShaders() {
  const files = [];
  const walk = async (dir) => {
    for (const e of await readdir(dir, { withFileTypes: true })) {
      const p = join(dir, e.name);
      if (e.isDirectory()) await walk(p);
      else if (e.name.endsWith('.js')) files.push(p);
    }
  };
  await walk(join(ROOT, 'src'));

  const bad = [];
  for (const file of files) {
    const src = await readFile(file, 'utf8');
    const lines = src.split('\n');
    let inGlsl = false;
    lines.forEach((line, i) => {
      if (!inGlsl) {
        if (/\/\* glsl \*\/`/.test(line)) inGlsl = true;
        return;
      }
      // The closing line is a lone back-tick; anything else carrying one
      // is inside the shader and will end the literal early.
      if (/^\s*`\s*;?\s*$/.test(line)) { inGlsl = false; return; }
      if (line.includes('`')) {
        bad.push(`${file.slice(ROOT.length + 1)}:${i + 1}  ${line.trim()}`);
      }
    });
  }

  if (bad.length) {
    console.error('✗ back-tick inside a GLSL template literal:');
    for (const b of bad) console.error(`    ${b}`);
    throw new Error('shader lint failed');
  }
}

async function main() {
  await lintShaders();
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

  const chrome = await launchChrome({ headless: !flag('head'), gpu: value('gpu', null) });
  console.log(`▸ chrome ${chrome.bin.split(/[\\/]/).pop()} on :${chrome.port}`);

  const cdp = await CDP.attach(chrome.port);
  cdp.__port = port;          // so bench() can build its own URLs
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

  if (flag('diff')) {
    await pixelDiff(cdp);
    cdp.close(); chrome.child.kill(); server.close();
    await sleep(200);
    process.exit(0);
  }

  if (flag('bench')) {
    await bench(cdp);
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
    // Freeze simulation time at a fixed value and let the temporal
    // filter converge. Rendering keeps going while the clock is paused,
    // so this produces a deterministic image that two builds can be
    // compared pixel for pixel — a settle-time-only shot drifts by a few
    // frames of animation and makes any comparison guesswork.
    if (shot.freeze !== undefined) {
      await cdp.eval(`(() => {
        const c = __aether.clock;
        c.paused = true;
        c.time = ${shot.freeze};
        return true;
      })()`);
      await sleep(1600);
    } else {
      await sleep(shot.settle);
    }

    if (shot.click) {
      const at = await cdp.eval(BALL_ON_SCREEN);
      const ev = (type, buttons) => cdp.send('Input.dispatchMouseEvent', {
        type, x: at[0], y: at[1], button: 'left', clickCount: 1, buttons,
      });
      await ev('mouseMoved', 0);
      await sleep(80);
      await ev('mousePressed', 1);
      await sleep(60);
      await ev('mouseReleased', 0);
      await sleep(shot.after ?? 600);
    }

    // Reach into the scene and pin state that time would otherwise carry
    // past. With the clock frozen an impact stays at whatever age you
    // set, which is the only way to photograph one stage of an animation
    // and compare it against another.
    if (shot.poke) {
      await cdp.eval(`(() => { ${shot.poke} })()`);
      await sleep(1400);
    }

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

/* ═══ pixel diff ══════════════════════════════════════════════════
   "You will not see the difference" is an opinion until someone
   measures it. This freezes simulation time, renders the same frame
   with one setting flipped, and reports what actually changed.        */

const DIFF_CASES = [
  {
    name: 'shadowNoise',
    label: '陰影含表面擾動',
    base: { shadowNoise: false },
    alt: { shadowNoise: true },
  },
  {
    name: 'reflectLit',
    label: '反射含陰影遮蔽',
    base: { reflectLit: false },
    alt: { reflectLit: true },
  },
  // Where does each budget stop buying anything? Every case is measured
  // against a reference well past the point of diminishing returns.
  { name: 'shadow-12', label: '陰影步數 12 對 48', base: { shadowSteps: 12 }, alt: { shadowSteps: 48 } },
  { name: 'shadow-20', label: '陰影步數 20 對 48', base: { shadowSteps: 20 }, alt: { shadowSteps: 48 } },
  { name: 'shadow-32', label: '陰影步數 32 對 48', base: { shadowSteps: 32 }, alt: { shadowSteps: 48 } },
  { name: 'ao-3',      label: '遮蔽取樣 3 對 8',   base: { aoTaps: 3 },       alt: { aoTaps: 8 } },
  { name: 'ao-5',      label: '遮蔽取樣 5 對 8',   base: { aoTaps: 5 },       alt: { aoTaps: 8 } },
  { name: 'reflect-24', label: '反射步數 24 對 90', base: { reflectSteps: 24 }, alt: { reflectSteps: 90 } },
  { name: 'reflect-48', label: '反射步數 48 對 90', base: { reflectSteps: 48 }, alt: { reflectSteps: 90 } },
  { name: 'steps-70',  label: '行進步數 70 對 190', base: { steps: 70 },      alt: { steps: 190 } },
  { name: 'steps-100', label: '行進步數 100 對 190', base: { steps: 100 },    alt: { steps: 190 } },
];

/**
 * Read the canvas back at a reduced size. Sampling the real drawing
 * buffer rather than a PNG keeps this dependency-free and avoids
 * decoding anything.
 */
const GRAB = `(() => {
  const c = document.getElementById('stage');
  const s = document.createElement('canvas');
  s.width = 320; s.height = 180;
  const x = s.getContext('2d');
  x.drawImage(c, 0, 0, 320, 180);
  return [...x.getImageData(0, 0, 320, 180).data];
})()`;

async function pixelDiff(cdp) {
  const base = `http://127.0.0.1:${cdp.__port}/index.html`;
  const url = `${base}#/march?scale=1&steps=140&spin=0&taa=0.9`;

  console.log('▸ pixel diff  (frozen clock, 320×180 sample, 8-bit channels)');

  await cdp.send('Page.navigate', { url });
  await waitFor(cdp, 'document.documentElement.dataset.boot', 'ready', 20000);
  await cdp.eval(`document.getElementById('intro-enter').click(); true`);
  await sleep(2500);

  // Freeze: rendering continues, animation does not, so two settings
  // differ by the setting alone.
  await cdp.eval(`(() => { const c = __aether.clock; c.paused = true; c.time = 8.0; return true; })()`);
  await sleep(1500);

  const capture = async (values) => {
    await cdp.eval(`__aether.panel.setValues(${JSON.stringify(values)}); true`);
    await sleep(1400);   // let the temporal filter re-converge
    return cdp.eval(GRAB);
  };

  for (const c of DIFF_CASES) {
    const a = await capture(c.base);
    const b = await capture(c.alt);

    let maxD = 0, sum = 0, n = 0, over2 = 0;
    for (let i = 0; i < a.length; i += 4) {
      const d = Math.max(
        Math.abs(a[i] - b[i]),
        Math.abs(a[i + 1] - b[i + 1]),
        Math.abs(a[i + 2] - b[i + 2]));
      maxD = Math.max(maxD, d);
      sum += d;
      if (d > 2) over2++;
      n++;
    }
    const mean = sum / n;
    const pct = (over2 / n) * 100;

    // A max of 1–2 on an 8-bit channel is dithering noise; anything that
    // moves more than a couple of levels across a visible share of the
    // frame is a real difference.
    const verdict = maxD <= 3 ? '看不出來'
      : pct < 0.5 ? '極小區域有差'
      : '看得出來';
    console.log(
      `  ${c.name.padEnd(13)} ${c.label.padEnd(16)} ` +
      `max ${String(maxD).padStart(3)}   mean ${mean.toFixed(2).padStart(5)}   ` +
      `>2 的像素 ${pct.toFixed(2).padStart(5)}%   → ${verdict}`);
  }
}

/* ═══ benchmark ═══════════════════════════════════════════════════
   A fixed workload, measured the same way every time, so "is the other
   GPU faster?" has a number attached instead of an impression.        */

/**
 * Each case pins everything that could drift: no camera spin, no
 * temporal accumulation (which would hide cost behind a smeared frame),
 * a stated render scale and step count. The name says what it isolates.
 */
const BENCH_CASES = [
  { name: 'baseline',    hash: '#/march?spin=0&taa=0&scale=0.75' },
  { name: 'full-res',    hash: '#/march?spin=0&taa=0&scale=1' },
  // Both switches are on by default now, so the informative cases turn
  // them off.
  { name: '−shadowNoise',hash: '#/march?spin=0&taa=0&scale=1&shadowNoise=0' },
  { name: '−reflectLit', hash: '#/march?spin=0&taa=0&scale=1&reflectLit=0' },
  { name: 'both off',    hash: '#/march?spin=0&taa=0&scale=1&shadowNoise=0&reflectLit=0' },
  { name: 'no-noise',    hash: '#/march?spin=0&taa=0&scale=1&displace=0' },
  { name: 'no-reflect',  hash: '#/march?spin=0&taa=0&scale=1&reflect=0' },
  { name: 'bare',        hash: '#/march?spin=0&taa=0&scale=1&displace=0&reflect=0&shadow=0&ao=0' },
];

async function bench(cdp) {
  const base = `http://127.0.0.1:${cdp.__port}/index.html`;
  const seconds = Number(value('seconds', 6));

  console.log(`▸ benchmark  (${seconds}s per case, ${VIEWPORT.width}×${VIEWPORT.height})`);

  let renderer = null;
  const rows = [];

  for (const c of BENCH_CASES) {
    await cdp.send('Page.navigate', { url: base + c.hash });
    await waitFor(cdp, 'document.documentElement.dataset.boot', 'ready', 20000);
    await cdp.eval(`document.getElementById('intro-enter').click(); true`);

    // Warm up before measuring: shader compilation, the first few frames
    // of texture allocation and the driver's own ramp-up are not what we
    // are trying to time.
    await sleep(2500);
    renderer ??= await cdp.eval(`(() => {
      const gl = document.getElementById('stage').getContext('webgl2');
      const d = gl.getExtension('WEBGL_debug_renderer_info');
      return d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
    })()`);

    await sleep(seconds * 1000);

    // The ring buffer holds the last 90 frames; percentiles off that are
    // far more honest than a smoothed fps counter.
    const r = await cdp.eval(`(() => {
      const p = __aether.perf;
      const h = [...p.ordered()].filter((v) => v > 0).sort((a, b) => a - b);
      const at = (q) => h[Math.min(h.length - 1, Math.floor(h.length * q))];
      return {
        n: h.length,
        median: +at(0.5).toFixed(2),
        p95: +at(0.95).toFixed(2),
        gpuMs: +p.gpuMs.toFixed(2),
        buffer: __aether.scene.rt.width + '×' + __aether.scene.rt.height,
      };
    })()`);

    rows.push({ ...c, ...r });
    console.log(
      `  ${c.name.padEnd(11)} ${String(r.buffer).padStart(9)}  ` +
      `${(1000 / r.median).toFixed(1).padStart(5)} fps   ` +
      `frame ${r.median.toFixed(1).padStart(5)} ms (p95 ${r.p95.toFixed(1)})   ` +
      `gpu ${r.gpuMs.toFixed(1).padStart(5)} ms`);
  }

  console.log(`\n▸ renderer: ${renderer}`);

  // Compared on GPU time, not frame time. Once the shader is fast
  // enough, frame time pins itself to the compositor's cadence and every
  // case reads the same — which says nothing about the shader.
  const fullRes = rows.find((r) => r.name === 'full-res');
  if (fullRes) {
    console.log('▸ GPU time against the full-res case:');
    for (const r of rows) {
      if (r.name === 'full-res' || r.name === 'baseline') continue;
      const d = r.gpuMs - fullRes.gpuMs;
      const pct = (d / fullRes.gpuMs) * 100;
      console.log(`    ${r.name.padEnd(13)} ${d >= 0 ? '+' : '−'}${Math.abs(d).toFixed(1)} ms` +
        `  (${d >= 0 ? '+' : '−'}${Math.abs(pct).toFixed(0)}%)`);
    }
  }
  return rows;
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

  await cdp.send('Page.navigate', { url: `${base}#/march` });
  await waitFor(cdp, 'document.documentElement.dataset.boot', 'ready', 20000);

  /* intro */
  await cdp.eval(`document.getElementById('intro-enter').click(); true`);
  await sleep(600);
  check('intro dismisses', await cdp.eval(`document.body.classList.contains('intro-done')`));

  /* clicking the surface bursts it */
  await cdp.eval(`__aether.panel.setValues({ spin: false }, { notify: true }); true`);
  await sleep(500);
  const bursts0 = await cdp.eval(`__aether.scene.bursts`);

  const target = await cdp.eval(BALL_ON_SCREEN);
  await mouse('mouseMoved', target[0], target[1], { buttons: 0 });
  await sleep(120);
  await mouse('mousePressed', target[0], target[1], { buttons: 1 });
  await sleep(80);
  await mouse('mouseReleased', target[0], target[1], { buttons: 0 });
  await sleep(250);

  const burst = await cdp.eval(`({
    bursts: __aether.scene.bursts,
    ripples: [...__aether.scene.ripples].filter((_, i) => i % 4 === 3).filter((v) => v > 0).length,
    flash: +__aether.scene.flash.toFixed(2),
  })`);
  check('a click strikes the surface', burst.bursts > bursts0,
    `${bursts0} → ${burst.bursts} impacts, flash ${burst.flash}`);
  check('the impact leaves a ripple', burst.ripples > 0, `${burst.ripples} active`);

  // The ripple has to ride its host: everything here orbits, and a centre
  // frozen in world space slides off the surface it belongs to.
  const rode = await cdp.eval(`(() => {
    const s = __aether.scene, i = [...s.ripples].findIndex((v, k) => k % 4 === 3 && v > 0);
    return i < 0 ? null : [s.ripples[i - 3], s.ripples[i - 2], s.ripples[i - 1]];
  })()`);
  await sleep(600);
  const rode2 = await cdp.eval(`(() => {
    const s = __aether.scene, i = [...s.ripples].findIndex((v, k) => k % 4 === 3 && v > 0);
    return i < 0 ? null : [s.ripples[i - 3], s.ripples[i - 2], s.ripples[i - 1]];
  })()`);
  const carried = rode && rode2
    ? Math.hypot(rode2[0] - rode[0], rode2[1] - rode[1], rode2[2] - rode[2]) : -1;
  check('the ripple rides its host', carried > 0.0005 && carried < 0.6,
    `centre carried ${carried.toFixed(4)} world units`);
  await shot(cdp, 'i0-impact');

  /* a click on empty sky must not burst anything */
  const bursts1 = await cdp.eval(`__aether.scene.bursts`);
  await mouse('mouseMoved', 180, 140, { buttons: 0 });
  await sleep(120);
  await mouse('mousePressed', 180, 140, { buttons: 1 });
  await sleep(80);
  await mouse('mouseReleased', 180, 140, { buttons: 0 });
  await sleep(250);
  check('a click that misses does nothing',
    (await cdp.eval(`__aether.scene.bursts`)) === bursts1,
    'the CPU pick agrees with what is on screen');

  /* the canvas is actually receiving pointer input, and a drag orbits */
  const yaw0 = await cdp.eval(`__aether.scene.yaw`);
  const midOrbit = await drag([700, 450], [980, 380]);
  const yaw1 = await cdp.eval(`__aether.scene.yaw`);
  check('canvas receives pointer events', midOrbit?.down === true, JSON.stringify(midOrbit));
  check('drag orbits the camera', Math.abs(yaw1 - yaw0) > 0.2,
    `yaw ${yaw0.toFixed(2)} → ${yaw1.toFixed(2)}`);
  await shot(cdp, 'i1-orbited');

  /* the wheel zooms */
  const dist0 = await cdp.eval(`__aether.scene.targetDist`);
  await cdp.send('Input.dispatchMouseEvent', {
    type: 'mouseWheel', x: 700, y: 450, deltaX: 0, deltaY: -400,
  });
  await sleep(400);
  const dist1 = await cdp.eval(`__aether.scene.targetDist`);
  check('wheel zooms the camera', dist1 < dist0,
    `${dist0.toFixed(2)} → ${dist1.toFixed(2)}`);

  /* the XY pad drives the light direction */
  const light0 = await cdp.eval(`[...__aether.scene.lightDir]`);
  await cdp.eval(`__aether.panel.setValues({ light: [0.15, 0.8] }, { notify: true }); true`);
  await sleep(400);
  const light1 = await cdp.eval(`[...__aether.scene.lightDir]`);
  const swung = Math.hypot(light1[0] - light0[0], light1[1] - light0[1], light1[2] - light0[2]);
  check('the XY pad steers the light', swung > 0.3,
    `direction moved ${swung.toFixed(2)}`);

  /* command palette: open, fuzzy-filter, run */
  await key('k', { mods: 2 /* Ctrl */, vk: 75 });
  check('Ctrl+K opens palette', await cdp.eval(`document.getElementById('cmdk').open`));
  for (const ch of 'sdf') await key(ch, { text: ch });
  const matches = await cdp.eval(`document.querySelectorAll('#cmdk-list li').length`);
  const top = await cdp.eval(`document.querySelector('#cmdk-list li .cmdk__label')?.textContent ?? ''`);
  check('fuzzy search filters', matches > 0 && matches < 12, `${matches} hits, top = "${top}"`);
  await shot(cdp, 'i2-cmdk');
  await key('Escape', { code: 'Escape', vk: 27 });
  await sleep(300);

  /* the master quality control drives the seven below it, and they
     remain individually editable afterwards */
  const q = () => cdp.eval(`(() => {
    const s = __aether.state;
    return { steps: s.steps, shadowSteps: s.shadowSteps, aoTaps: s.aoTaps,
             reflectSteps: s.reflectSteps, scale: s.scale,
             shadowNoise: s.shadowNoise, reflectLit: s.reflectLit };
  })()`);

  const qHigh = await q();
  await cdp.eval(`__aether.panel.setValues({ quality: 0.02 }, { notify: true }); true`);
  await sleep(500);
  const qLow = await q();
  check('the master quality slider moves all seven',
    qLow.steps < qHigh.steps && qLow.shadowSteps < qHigh.shadowSteps
    && qLow.aoTaps < qHigh.aoTaps && qLow.reflectSteps < qHigh.reflectSteps
    && qLow.scale !== qHigh.scale
    && qLow.shadowNoise === false && qLow.reflectLit === false,
    `steps ${qHigh.steps}→${qLow.steps}, shadow ${qHigh.shadowSteps}→${qLow.shadowSteps}, ` +
    `ao ${qHigh.aoTaps}→${qLow.aoTaps}, scale ${qHigh.scale}→${qLow.scale}`);

  await cdp.eval(`__aether.panel.setValues({ steps: 150 }, { notify: true }); true`);
  await sleep(400);
  check('an individual slider still overrides the master',
    (await cdp.eval(`__aether.state.steps`)) === 150,
    'the master is an action, not a binding');

  await cdp.eval(`__aether.panel.setValues({ quality: 0.5 }, { notify: true }); true`);
  await sleep(400);

  /* the render-scale control actually reallocates the target */
  const rt0 = await cdp.eval(`__aether.scene.rt.width`);
  await cdp.eval(`__aether.panel.setValues({ scale: '0.5' }, { notify: true }); true`);
  await sleep(500);
  const rt1 = await cdp.eval(`__aether.scene.rt.width`);
  check('render scale resizes the target', rt1 < rt0, `${rt0}px → ${rt1}px wide`);
  await cdp.eval(`__aether.panel.setValues({ scale: '0.75' }, { notify: true }); true`);
  await sleep(400);

  /* URL is state */
  await cdp.eval(`__aether.panel.setValues({ tint: 'jade' }, { notify: true }); true`);
  await sleep(300);
  const url = await cdp.eval(`location.hash`);
  check('params round-trip into the URL', url.includes('tint=jade'), url);

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
  await shot(cdp, 'i5-zen');
  await key('z', { vk: 90 });

  await key('?', { code: 'Slash', vk: 191, mods: 8 });
  check('? opens shortcuts', await cdp.eval(`document.getElementById('help').open`));
  await shot(cdp, 'i6-help');
  await key('Escape', { code: 'Escape', vk: 27 });

  /* a lab with one scene must not show a tab bar that cannot do anything */
  check('single-scene lab hides the tab bar',
    await cdp.eval(`getComputedStyle(document.getElementById('tabs')).display === 'none'`));

  /* reset puts back everything: parameters, camera and the URL. It used
     to do only half of that, with the other half hidden in a separate
     command. */
  await cdp.eval(`__aether.panel.setValues(
    { tint: 'rose', steps: 180, erode: 0.4 }, { notify: true }); true`);
  await drag([700, 450], [1050, 340]);
  await sleep(400);
  const messy = await cdp.eval(`({ tint: __aether.state.tint, steps: __aether.state.steps,
    erode: __aether.state.erode, yaw: +__aether.scene.yaw.toFixed(2) })`);

  await key('r', { vk: 82 });
  await sleep(600);
  const clean = await cdp.eval(`({ tint: __aether.state.tint, steps: __aether.state.steps,
    erode: __aether.state.erode, yaw: +__aether.scene.yaw.toFixed(2),
    hash: location.hash })`);

  check('reset restores every parameter',
    clean.tint === 'amber' && clean.steps === 100 && clean.erode === 0.1,
    `${JSON.stringify(messy)} → ${JSON.stringify(clean)}`);
  check('reset restores the camera and the URL',
    Math.abs(clean.yaw - 0.85) < 0.2 && clean.hash === '#/march',
    `yaw ${messy.yaw} → ${clean.yaw}, hash ${clean.hash}`);

  check('no console errors during interaction', problems.length === 0,
    problems.map((p) => p.text.slice(0, 120)).join(' | '));
  await shot(cdp, 'i7-final');

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
