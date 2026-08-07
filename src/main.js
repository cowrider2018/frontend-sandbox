/* ── main.js ─────────────────────────────────────────────────────────
   AETHER — application shell.

   Owns exactly one of everything: one WebGL2 context, one rAF loop, one
   pointer, one router. Scenes are swapped in and out of that fixed
   machinery; they never create a context, start a loop, or touch the
   chrome. Everything below is wiring.
   ------------------------------------------------------------------ */

import { createContext, createFullscreenTriangle, createEmptyVAO } from './core/gl.js';
import { ShaderError } from './core/program.js';
import { Clock } from './core/clock.js';
import { Pointer } from './core/pointer.js';
import { Perf } from './core/perf.js';
import { Router } from './core/router.js';
import { signal, effect } from './core/signal.js';
import { Panel } from './ui/panel.js';
import { Tabs, Hud, Toaster } from './ui/chrome.js';
import { CommandPalette } from './ui/cmdk.js';
import { SCENES, SCENE_BY_ID, DEFAULT_SCENE } from './scenes/index.js';

const $ = (id) => document.getElementById(id);

/** Never allocate more than this many pixels of drawing buffer. */
const MAX_PIXELS = 2.6e6;

const SHORTCUTS = [
  ['1…4', '切換場景'],
  ['Ctrl K', '指令面板'],
  ['W A S D', '驅動貓走動（顯示貓時；此時 S 不是截圖）'],
  ['Space', '暫停 / 繼續'],
  ['.', '暫停時前進一幀'],
  ['R', '重設場景與所有參數'],
  ['S', '儲存截圖'],
  ['P', '顯示 / 隱藏參數面板'],
  ['Z', '禪模式（隱藏所有介面）'],
  ['F', '全螢幕'],
  ['?', '這張表'],
];

async function boot() {
  const canvas = $('stage');
  const log = $('boot-log');
  const say = (msg) => { log.textContent = msg; };

  say('probing webgl2…');

  let ctx;
  try {
    ctx = createContext(canvas);
  } catch (err) {
    fail(err);
    return;
  }

  const { gl, ext, limits } = ctx;
  say(limits.renderer.toLowerCase().slice(0, 48));

  const app = new App(canvas, ctx);
  try {
    await app.start(say);
  } catch (err) {
    fail(err);
    return;
  }

  // Debug handle. Everything the app owns hangs off this, which is what
  // makes the headless harness in tools/ able to inspect real GPU state
  // instead of guessing from pixels.
  globalThis.__aether = app;
  globalThis.__aether.__restore = () => app._restoreDefaults();

  document.documentElement.dataset.boot = 'ready';
}

function fail(err) {
  console.error(err);
  const box = $('gl-fallback');
  box.hidden = false;
  document.documentElement.dataset.boot = 'failed';
  $('boot').style.display = 'none';
  if (err?.code === 'FLOAT_TARGETS_UNAVAILABLE') {
    box.querySelector('p').textContent =
      '你的瀏覽器有 WebGL2，但缺少浮點材質渲染（EXT_color_buffer_float）。本站的每一個場景都把模擬狀態存在浮點貼圖裡，沒有它無法運作。';
  } else if (err instanceof ShaderError) {
    box.querySelector('h1').textContent = '著色器編譯失敗';
    box.querySelector('p').textContent = '完整的錯誤與行號已輸出到開發者主控台。';
  }
}

/* ═══════════════════════════════════════════════════════════════════ */

class App {
  constructor(canvas, ctx) {
    this.canvas = canvas;
    this.gl = ctx.gl;
    this.ext = ctx.ext;
    this.limits = ctx.limits;

    this.scene = null;
    this.sceneDef = null;
    this.state = {};
    this.dpr = 1;
    this.width = 1;
    this.height = 1;
    this.readoutTick = 0;

    /**
     * Chrome state as signals. Every toggle below has three or four
     * consequences — a class, an ARIA attribute, a label, the clock —
     * and keeping those together in one effect is the difference between
     * a button that works and a button that works *and* announces
     * itself correctly. Callers just write `ui.zen.set(true)`.
     */
    this.ui = {
      paused: signal(false),
      panelOpen: signal(true),
      zen: signal(false),
      scene: signal(null),
    };

    this.tri = createFullscreenTriangle(this.gl);
    this.empty = createEmptyVAO(this.gl);
    this.perf = new Perf(this.gl, this.ext.timer);
    this.clock = new Clock((c) => this.frame(c));
    // Press and release are forwarded as events rather than sampled from
    // `pointer.down` in the render loop: a quick click on a scene running
    // at 25 fps can begin and end between two frames, and a click that
    // does nothing because the frame rate was low is not acceptable.
    this.pointer = new Pointer(canvas, {
      onDown: (p) => this.scene?.onPointerDown?.(p),
      onUp: (p) => this.scene?.onPointerUp?.(p),
    });
    this.toaster = new Toaster($('toasts'));

    /** The object every scene receives. */
    this.sceneCtx = {
      gl: this.gl,
      ext: this.ext,
      limits: this.limits,
      canvas,
      tri: this.tri,
      empty: this.empty,
      setParams: (values) => this.panel.setValues(values),
      toast: (msg) => this.toaster.show(msg),
    };
  }

  async start(say) {
    say('compiling shaders…');
    this._buildChrome();
    this._bindKeys();
    this._bindResize();
    this._bindContextLoss();

    this.router = new Router({
      routes: SCENES.map((s) => s.id),
      fallback: DEFAULT_SCENE,
      onChange: ({ id, params, changed }) => this._activate(id, params, changed),
    });

    // Yield once so the boot splash actually paints before the first
    // (expensive) shader compile blocks the main thread.
    await new Promise((r) => requestAnimationFrame(r));

    this.router.start();
    this._resize();
    this.clock.start();

    say('ready');
    this._bindIntro();
  }

  /* ── chrome ───────────────────────────────────────────────────── */

  _buildChrome() {
    this.tabs = new Tabs($('tabs'), $('tabs-ink'), SCENES, (id) => this.router.go(id));
    // A one-tab tab bar is a control that cannot do anything.
    document.body.classList.toggle('single-scene', SCENES.length < 2);

    this.hud = new Hud({
      canvas: $('hud-graph'),
      fpsEl: $('hud-fps'),
      gpuEl: $('hud-gpu'),
      root: $('hud'),
    });

    this.panel = new Panel($('panel-body'), (id, value, commit) => {
      this.state[id] = value;
      if (commit) this._syncUrl();
    });

    this.cmdk = new CommandPalette({
      dialog: $('cmdk'),
      input: $('cmdk-input'),
      list: $('cmdk-list'),
      empty: $('cmdk-empty'),
      provider: () => this._commands(),
    });

    $('help-keys').innerHTML = SHORTCUTS
      .map(([k, d]) => `<dt>${k.split(' ').map((x) => `<kbd>${x}</kbd>`).join('')}</dt><dd>${d}</dd>`)
      .join('');

    $('btn-palette').addEventListener('click', () => this.cmdk.open());
    $('btn-panel').addEventListener('click', () => this._togglePanel());
    $('panel-close').addEventListener('click', () => this._togglePanel());
    $('act-reset').addEventListener('click', () => this._reset());
    $('act-pause').addEventListener('click', () => this._togglePause());
    $('act-shot').addEventListener('click', () => this._screenshot());

    this._bindState();
  }

  /** One effect per piece of chrome state; each owns all of its DOM. */
  _bindState() {
    const { paused, panelOpen, zen, scene } = this.ui;

    effect(() => {
      const on = paused();
      this.clock.setPaused(on);
      const btn = $('act-pause');
      btn.setAttribute('aria-pressed', String(on));
      btn.firstChild.textContent = on ? '繼續 ' : '暫停 ';
    });

    effect(() => {
      const closed = !panelOpen();
      document.body.classList.toggle('panel-closed', closed);
      $('btn-panel').setAttribute('aria-expanded', String(!closed));
    });

    effect(() => {
      document.body.classList.toggle('zen', zen());
    });

    effect(() => {
      const id = scene();
      if (!id) return;
      // Re-tints the entire chrome: every accent in the design system is
      // derived from --accent-h, which this attribute selects.
      document.documentElement.dataset.scene = id;
      this.tabs.select(id);
    });
  }

  _bindIntro() {
    const dismiss = () => {
      if (document.body.classList.contains('intro-done')) return;
      document.body.classList.add('intro-done');
      this.tabs.reposition();
      this.canvas.focus?.();
    };
    $('intro-enter').addEventListener('click', dismiss);
    $('intro').addEventListener('pointerdown', dismiss);
    this._dismissIntro = dismiss;
  }

  /* ── scene lifecycle ──────────────────────────────────────────── */

  _activate(id, params, changed) {
    const def = SCENE_BY_ID.get(id);
    if (!def) return;

    if (changed && this.scene) {
      this.scene.dispose();
      this.scene = null;
    }

    this.ui.scene.set(id);

    if (!this.scene) {
      try {
        this.sceneDef = def;
        this.scene = def.init(this.sceneCtx);
      } catch (err) {
        console.error(err);
        this.toaster.show(`<b>${def.title}</b> 初始化失敗，詳見主控台`, { ms: 5000 });
        if (err instanceof ShaderError) throw err;
        return;
      }
    }

    this.state = this.panel.build(def.params, params);
    this._paintCaption(def);
    // Forced: a freshly created scene still has 2×2 render targets, and
    // switching scenes does not change the canvas size, so the ordinary
    // "did anything change?" path would skip it entirely.
    this._resize(true);
  }

  _paintCaption(def) {
    const apply = () => {
      $('caption-index').textContent = def.index;
      $('caption-title').textContent = def.title;
      $('caption-tech').textContent = def.tech;
      $('panel-title').textContent = def.title;
      $('panel-desc').textContent = def.desc;
      document.title = `${def.index} ${def.title} — AETHER`;
    };
    // Cross-fade the caption if the browser can; plain swap if not.
    if (document.startViewTransition && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
      const vt = document.startViewTransition(apply);
      // Switching scenes faster than a transition can finish aborts it,
      // and an aborted transition rejects. That is expected here — the
      // DOM update still ran — so swallow it rather than letting it
      // surface as an unhandled rejection.
      vt.ready.catch(() => {});
      vt.finished.catch(() => {});
    } else {
      apply();
    }
  }

  _syncUrl() {
    if (!this.sceneDef) return;
    this.router.replaceParams(this.panel.serialise(this.sceneDef.params));
  }

  /* ── sizing ───────────────────────────────────────────────────── */

  _bindResize() {
    const ro = new ResizeObserver(() => this._resize());
    ro.observe(this.canvas);
    window.addEventListener('resize', () => this._resize());
    matchMedia(`(resolution: ${devicePixelRatio}dppx)`).addEventListener?.('change', () => this._resize());
  }

  _resize(force = false) {
    const cssW = this.canvas.clientWidth || window.innerWidth;
    const cssH = this.canvas.clientHeight || window.innerHeight;

    // Clamp total pixels rather than DPR: a 4K display and a phone at
    // 3× both end up with a drawing buffer the GPU can actually feed.
    let dpr = Math.min(devicePixelRatio || 1, 2);
    const budget = Math.sqrt(MAX_PIXELS / (cssW * cssH));
    if (budget < dpr) dpr = Math.max(1, budget);

    const w = Math.max(2, Math.round(cssW * dpr));
    const h = Math.max(2, Math.round(cssH * dpr));
    if (!force && w === this.width && h === this.height) return;

    this.canvas.width = w;
    this.canvas.height = h;
    this.width = w;
    this.height = h;
    this.dpr = dpr;
    this.scene?.resize(w, h);
  }

  /* ── frame ────────────────────────────────────────────────────── */

  frame(clock) {
    this.pointer.update(clock.wallDt);
    this.perf.sample(clock.wallDt);

    if (this.scene) {
      this.perf.begin();
      try {
        this.scene.frame({ state: this.state, clock, pointer: this.pointer });
      } catch (err) {
        console.error(err);
        this.toaster.show('場景執行時發生錯誤，已暫停', { ms: 5000 });
        this.scene.dispose();
        this.scene = null;
      }
      this.perf.end();
    }

    this.hud.update(this.perf);

    // Telemetry does not need 60 Hz, and re-laying-out text every frame
    // is the one thing here that would actually cost frames.
    if (++this.readoutTick % 8 === 0 && this.scene?.readout) {
      this.panel.updateReadout(this.scene.readout(this.state));
    }
  }

  /* ── actions ──────────────────────────────────────────────────── */

  /**
   * Reset means reset. It used to put the camera back but leave every
   * slider where it was, with a second, separate command for the
   * parameters — two half-resets, and no way to guess which one the
   * button did.
   */
  _reset() {
    this._restoreDefaults();
    this.scene?.reset?.();
    this.toaster.show('場景與參數已重設');
  }

  _togglePause() { this.ui.paused.set((v) => !v); }

  _togglePanel() { this.ui.panelOpen.set((v) => !v); }

  _toggleZen() {
    const on = this.ui.zen.set((v) => !v);
    if (on) this.toaster.show('禪模式 — 再按一次 <kbd>Z</kbd> 還原');
  }

  async _toggleFullscreen() {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await document.documentElement.requestFullscreen();
    } catch { this.toaster.show('這個瀏覽器拒絕了全螢幕請求'); }
  }

  _screenshot() {
    // preserveDrawingBuffer keeps the frame readable after the compositor
    // has taken it, which is what makes this a one-liner.
    this.canvas.toBlob((blob) => {
      if (!blob) { this.toaster.show('截圖失敗'); return; }
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
      a.href = url;
      a.download = `aether-${this.sceneDef?.id ?? 'scene'}-${stamp}.png`;
      a.click();
      URL.revokeObjectURL(url);
      this.toaster.show(`已儲存 <b>${a.download}</b>`);
    }, 'image/png');
  }

  async _copyLink() {
    this._syncUrl();
    try {
      await navigator.clipboard.writeText(location.href);
      this.toaster.show('連結已複製 — 裡面包含現在的每一個參數');
    } catch {
      this.toaster.show('無法存取剪貼簿，請直接複製網址列');
    }
  }

  _step(delta) {
    const i = SCENES.findIndex((s) => s.id === this.router.current);
    this.router.go(SCENES[(i + delta + SCENES.length) % SCENES.length].id);
  }

  /* ── commands ─────────────────────────────────────────────────── */

  _commands() {
    const list = SCENES.map((s) => ({
      label: s.title,
      meta: s.index,
      glyph: s.glyph,
      keywords: `${s.id} ${s.tech}`,
      run: () => this.router.go(s.id),
    }));

    return list.concat([
      { label: '重設場景與所有參數', glyph: '↺', meta: 'R',
        keywords: 'reset defaults restore params camera', run: () => this._reset() },
      { label: '暫停 / 繼續', glyph: '⏸', meta: 'Space', keywords: 'pause play', run: () => this._togglePause() },
      { label: '儲存 PNG 截圖', glyph: '⤓', meta: 'S', keywords: 'screenshot save png', run: () => this._screenshot() },
      { label: '複製含參數的連結', glyph: '⧉', meta: '', keywords: 'copy link share url', run: () => this._copyLink() },
      { label: '顯示 / 隱藏參數面板', glyph: '☰', meta: 'P', keywords: 'panel controls', run: () => this._togglePanel() },
      { label: '禪模式', glyph: '☯', meta: 'Z', keywords: 'zen hide ui clean', run: () => this._toggleZen() },
      { label: '全螢幕', glyph: '⛶', meta: 'F', keywords: 'fullscreen', run: () => this._toggleFullscreen() },
      { label: '快捷鍵一覽', glyph: '?', meta: '?', keywords: 'help keys shortcuts', run: () => $('help').showModal() },
    ]);
  }

  /**
   * Every control the panel actually registered, back to the value its
   * schema declares. Static blocks are skipped: they carry an id for the
   * author's convenience but hold nothing.
   */
  _restoreDefaults() {
    if (!this.sceneDef) return;
    const defaults = {};
    for (const p of this.sceneDef.params) {
      if (p.id !== undefined && this.panel.widgets.has(p.id)) defaults[p.id] = p.value;
    }
    this.panel.setValues(defaults);
    this.router.replaceParams({});
  }

  /* ── keyboard ─────────────────────────────────────────────────── */

  _bindKeys() {
    window.addEventListener('keydown', (e) => {
      // Never steal keys from a text field or an open dialog's input.
      const t = e.target;
      if (t instanceof HTMLInputElement || t instanceof HTMLTextAreaElement || t?.isContentEditable) {
        if (!(e.key === 'k' && (e.ctrlKey || e.metaKey))) return;
      }

      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        this.cmdk.toggle();
        return;
      }
      if (e.ctrlKey || e.metaKey || e.altKey) return;

      // While the intro is up, the only keys that mean anything are the
      // ones that dismiss it — otherwise Enter would also toggle pause.
      if (!document.body.classList.contains('intro-done')) {
        if (['Enter', ' ', 'Escape'].includes(e.key)) {
          e.preventDefault();
          this._dismissIntro?.();
          return;
        }
      }

      // The scene gets first refusal. A scene that is driving something
      // needs keys the shortcuts already claim — WASD overlaps S for
      // screenshot — and only the scene knows whether it is currently in
      // a state where that key means something. Returning true is how it
      // says "mine", and nothing below runs.
      if (this.scene?.onKey?.(e, true)) {
        e.preventDefault();
        return;
      }

      const index = '1234'.indexOf(e.key);
      if (index >= 0 && index < SCENES.length) {
        this._dismissIntro?.();
        this.router.go(SCENES[index].id);
        return;
      }

      switch (e.key) {
        case ' ': e.preventDefault(); this._togglePause(); break;
        case '.': this.clock.step(); break;
        case '[': this._step(-1); break;
        case ']': this._step(1); break;
        case '?': $('help').showModal(); break;
        default:
          switch (e.key.toLowerCase()) {
            case 'r': this._reset(); break;
            case 's': this._screenshot(); break;
            case 'p': this._togglePanel(); break;
            case 'z': this._toggleZen(); break;
            case 'f': this._toggleFullscreen(); break;
          }
      }
    });

    // Key release only ever concerns the scene — nothing in the shell is
    // held down.
    window.addEventListener('keyup', (e) => { this.scene?.onKey?.(e, false); });

    // Alt-tabbing away with a key held would otherwise leave the scene
    // believing it is still held, and whatever it drives running away.
    window.addEventListener('blur', () => { this.scene?.releaseKeys?.(); });
  }

  /* ── context loss ─────────────────────────────────────────────── */

  _bindContextLoss() {
    this.canvas.addEventListener('webglcontextlost', (e) => {
      // Without preventDefault the context is gone for good.
      e.preventDefault();
      this.clock.stop();
      this.scene = null;
      this.toaster.show('GPU 內容遺失 — 正在重建…', { ms: 4000 });
    });

    this.canvas.addEventListener('webglcontextrestored', () => {
      this.tri = createFullscreenTriangle(this.gl);
      this.empty = createEmptyVAO(this.gl);
      this.sceneCtx.tri = this.tri;
      this.sceneCtx.empty = this.empty;
      this.perf = new Perf(this.gl, this.ext.timer);
      this.width = this.height = 0;   // force a real resize
      this.router.current = null;     // force a full scene rebuild
      this.router.start();
      this._resize();
      this.clock.start();
      this.toaster.show('已重建完成');
    });
  }
}

// Class declarations are hoisted but not initialised, so boot() has to
// run after App exists — hence the call at the very bottom of the file.
boot();
