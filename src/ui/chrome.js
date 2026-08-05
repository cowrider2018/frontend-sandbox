/* ── ui/chrome.js ────────────────────────────────────────────────────
   Tabs, the performance HUD, and toasts.
   ------------------------------------------------------------------ */

/* ═══ tabs ════════════════════════════════════════════════════════ */

export class Tabs {
  constructor(nav, ink, scenes, onSelect) {
    this.nav = nav;
    this.ink = ink;
    this.buttons = new Map();

    for (const scene of scenes) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'tab';
      btn.setAttribute('role', 'tab');
      btn.setAttribute('aria-selected', 'false');
      btn.dataset.id = scene.id;
      btn.innerHTML = `<span class="tab__num">${scene.index}</span><span class="tab__label">${scene.title}</span>`;
      btn.addEventListener('click', () => onSelect(scene.id));
      nav.append(btn);
      this.buttons.set(scene.id, btn);
    }
    nav.setAttribute('role', 'tablist');

    // The ink pill is measured, not calculated: it tracks whatever the
    // font and container actually produced, including after a reflow.
    this._ro = new ResizeObserver(() => this.reposition());
    this._ro.observe(nav);
  }

  select(id) {
    for (const [key, btn] of this.buttons) {
      btn.setAttribute('aria-selected', String(key === id));
    }
    this.active = id;
    this.reposition();
  }

  reposition() {
    const btn = this.buttons.get(this.active);
    if (!btn) return;
    this.ink.style.width = `${btn.offsetWidth}px`;
    this.ink.style.transform = `translateX(${btn.offsetLeft - 3}px)`;
  }
}

/* ═══ performance HUD ═════════════════════════════════════════════ */

export class Hud {
  constructor({ canvas, fpsEl, gpuEl, root }) {
    this.canvas = canvas;
    this.fpsEl = fpsEl;
    this.gpuEl = gpuEl;
    this.root = root;
    this.ctx = canvas.getContext('2d');
    this._buf = null;
    this._resize();
    new ResizeObserver(() => this._resize()).observe(canvas);
  }

  _resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const w = canvasWidth(this.canvas), h = canvasHeight(this.canvas);
    this.canvas.width = Math.max(1, Math.round(w * dpr));
    this.canvas.height = Math.max(1, Math.round(h * dpr));
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = w;
    this.h = h;
  }

  update(perf) {
    const fps = perf.fps;
    this.fpsEl.textContent = fps > 0 ? String(Math.round(fps)) : '—';
    this.gpuEl.textContent = perf.supported && perf.gpuMs > 0 ? perf.gpuMs.toFixed(1) : '—';
    this.root.classList.toggle('is-low', fps > 0 && fps < 45);
    this._draw(perf);
  }

  _draw(perf) {
    const { ctx, w, h } = this;
    if (!w || !h) return;
    this._buf = perf.ordered(this._buf ?? new Float32Array(perf.length));
    const data = this._buf;
    const n = data.length;

    ctx.clearRect(0, 0, w, h);

    // Fixed 33 ms ceiling: the graph must not silently rescale, or a
    // performance cliff looks identical to a smooth run.
    const ceiling = 33.4;
    const y = (ms) => h - Math.min(ms, ceiling) / ceiling * (h - 2) - 1;

    const style = getComputedStyle(document.documentElement);
    const accent = style.getPropertyValue('--accent-bright').trim() || '#7de3ff';

    // 16.7 ms reference line = 60 fps
    ctx.strokeStyle = 'rgba(255,255,255,0.13)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, Math.round(y(16.7)) + 0.5);
    ctx.lineTo(w, Math.round(y(16.7)) + 0.5);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, h);
    for (let i = 0; i < n; i++) ctx.lineTo((i / (n - 1)) * w, y(data[i]));
    ctx.lineTo(w, h);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, withAlpha(accent, 0.35));
    grad.addColorStop(1, withAlpha(accent, 0.02));
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const px = (i / (n - 1)) * w;
      const py = y(data[i]);
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    }
    ctx.strokeStyle = accent;
    ctx.lineWidth = 1.25;
    ctx.stroke();
  }
}

function canvasWidth(c) { return c.clientWidth || Number(c.getAttribute('width')) || 60; }
function canvasHeight(c) { return c.clientHeight || Number(c.getAttribute('height')) || 20; }

/** oklch()/hex/rgb → a colour with the given alpha, via the 2D context. */
function withAlpha(color, alpha) {
  const probe = withAlpha._c ??= document.createElement('canvas').getContext('2d');
  probe.fillStyle = '#000';
  probe.fillStyle = color;
  const resolved = probe.fillStyle;
  if (resolved.startsWith('#')) {
    const n = parseInt(resolved.slice(1), 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
  }
  return resolved.replace(/^rgba?\(([^)]+)\)$/, (_, body) => {
    const parts = body.split(',').slice(0, 3).map((s) => s.trim());
    return `rgba(${parts.join(', ')}, ${alpha})`;
  });
}

/* ═══ toasts ══════════════════════════════════════════════════════ */

export class Toaster {
  constructor(host) { this.host = host; }

  show(message, { ms = 2200 } = {}) {
    const el = document.createElement('div');
    el.className = 'toast';
    el.innerHTML = message;
    this.host.append(el);

    const remove = () => {
      el.classList.add('is-out');
      el.addEventListener('animationend', () => el.remove(), { once: true });
      // Belt and braces: if the animation never fires, still clean up.
      setTimeout(() => el.remove(), 600);
    };
    setTimeout(remove, ms);
    return el;
  }
}
