/* ── ui/widgets.js ───────────────────────────────────────────────────
   Custom elements for the control panel.

   Each widget wraps a real, focusable native control (<input type=range>,
   <button role=switch>, <button role=radio>) and paints a skin over it.
   That is the whole trick: keyboard interaction, screen-reader semantics
   and IME behaviour come from the platform, and only the pixels are ours.
   Nothing here re-implements a control from scratch.
   ------------------------------------------------------------------ */

const html = String.raw;

/** Sensible decimal places from the step size, unless told otherwise. */
function digitsFor(spec) {
  if (spec.digits !== undefined) return spec.digits;
  const step = spec.step ?? 1;
  if (step >= 1) return 0;
  return Math.min(4, String(step).split('.')[1]?.length ?? 2);
}

function format(spec, value) {
  return value.toFixed(digitsFor(spec)) + (spec.unit ? ` ${spec.unit}` : '');
}

let uid = 0;
const nextId = () => `ctl-${++uid}`;

/* ═══ base ════════════════════════════════════════════════════════ */

class Widget extends HTMLElement {
  connectedCallback() {
    if (this._built) return;
    this._built = true;
    this.render();
  }

  /** Notify the panel. `commit` marks the end of a drag, for URL writes. */
  emit(value, commit = false) {
    this._value = value;
    this.dispatchEvent(new CustomEvent('ctl-input', {
      bubbles: true,
      detail: { id: this.spec.id, value, commit },
    }));
  }

  get value() { return this._value; }
}

/* ═══ slider ══════════════════════════════════════════════════════ */

export class AeSlider extends Widget {
  render() {
    const s = this.spec;
    const id = nextId();
    this.innerHTML = html`
      <div class="ctl">
        <div class="ctl__row">
          <label class="ctl__label" for="${id}">${s.label}</label>
          <output class="ctl__value" for="${id}"></output>
        </div>
        <div class="ctl__track">
          <div class="ctl__rail"><div class="ctl__fill"></div></div>
          <input class="ctl__range" type="range" id="${id}"
                 min="${s.min}" max="${s.max}" step="${s.step}"
                 aria-label="${s.label}">
          <div class="ctl__thumb"></div>
        </div>
      </div>`;

    this.input = this.querySelector('input');
    this.out = this.querySelector('output');
    this.fill = this.querySelector('.ctl__fill');
    this.thumb = this.querySelector('.ctl__thumb');

    this.input.addEventListener('input', () => {
      const v = Number(this.input.value);
      this.paint(v);
      this.emit(v);
    });
    this.input.addEventListener('change', () => this.emit(Number(this.input.value), true));

    this.set(this._value ?? s.value);
  }

  set(value) {
    this._value = value;
    if (!this.input) return;
    this.input.value = String(value);
    this.paint(value);
  }

  paint(v) {
    const s = this.spec;
    const p = (v - s.min) / (s.max - s.min);
    this.fill.style.setProperty('--p', p.toFixed(4));
    this.thumb.style.setProperty('--pp', `${(p * 100).toFixed(2)}%`);
    this.out.textContent = format(s, v);
  }
}

/* ═══ switch ══════════════════════════════════════════════════════ */

export class AeSwitch extends Widget {
  render() {
    const s = this.spec;
    this.innerHTML = html`
      <div class="ctl ctl--switch">
        <div class="ctl__row">
          <span class="ctl__label">${s.label}</span>
          <button class="switch" type="button" role="switch" aria-checked="false" aria-label="${s.label}"></button>
        </div>
      </div>`;

    this.btn = this.querySelector('.switch');
    const toggle = () => this.set(!this._value, true);
    this.btn.addEventListener('click', toggle);
    this.querySelector('.ctl__row').addEventListener('click', (e) => {
      if (e.target !== this.btn) toggle();
    });

    this.set(this._value ?? s.value);
  }

  set(value, notify = false) {
    this._value = Boolean(value);
    this.btn?.setAttribute('aria-checked', String(this._value));
    if (notify) this.emit(this._value, true);
  }
}

/* ═══ segmented select ════════════════════════════════════════════ */

export class AeSelect extends Widget {
  render() {
    const s = this.spec;
    this.innerHTML = html`
      <div class="ctl">
        <div class="ctl__row"><span class="ctl__label">${s.label}</span></div>
        <div class="seg" role="radiogroup" aria-label="${s.label}">
          ${s.options.map((o) => html`
            <button class="seg__opt" type="button" role="radio" aria-checked="false"
                    data-value="${o.value}" tabindex="-1">${o.label}</button>`).join('')}
        </div>
      </div>`;

    this.opts = [...this.querySelectorAll('.seg__opt')];
    this.opts.forEach((btn, i) => {
      btn.addEventListener('click', () => this.set(btn.dataset.value, true));
      btn.addEventListener('keydown', (e) => {
        const dir = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
        if (!dir) return;
        e.preventDefault();
        const next = this.opts[(i + dir + this.opts.length) % this.opts.length];
        this.set(next.dataset.value, true);
        next.focus();
      });
    });

    this.set(this._value ?? s.value);
  }

  set(value, notify = false) {
    this._value = String(value);
    for (const btn of this.opts ?? []) {
      const on = btn.dataset.value === this._value;
      btn.setAttribute('aria-checked', String(on));
      // Roving tabindex: one stop for the whole group, arrows move within.
      btn.tabIndex = on ? 0 : -1;
    }
    if (notify) this.emit(this._value, true);
  }
}

/* ═══ xy pad ══════════════════════════════════════════════════════ */

export class AeXY extends Widget {
  render() {
    const s = this.spec;
    this.innerHTML = html`
      <div class="ctl">
        <div class="ctl__row">
          <span class="ctl__label">${s.label}</span>
          <output class="ctl__value"></output>
        </div>
        <div class="xy" tabindex="0" role="slider" aria-label="${s.label}"
             aria-valuemin="0" aria-valuemax="1" aria-valuenow="0.5">
          <div class="xy__cross"></div>
          <div class="xy__dot"></div>
        </div>
      </div>`;

    this.pad = this.querySelector('.xy');
    this.out = this.querySelector('output');

    const at = (e) => {
      const r = this.pad.getBoundingClientRect();
      return [
        clamp01((e.clientX - r.left) / r.width),
        clamp01((e.clientY - r.top) / r.height),
      ];
    };

    let dragging = false;
    this.pad.addEventListener('pointerdown', (e) => {
      dragging = true;
      this.pad.setPointerCapture(e.pointerId);
      this.set(at(e), true);
      e.preventDefault();
    });
    this.pad.addEventListener('pointermove', (e) => { if (dragging) this.set(at(e), true); });
    this.pad.addEventListener('pointerup', (e) => {
      dragging = false;
      this.pad.releasePointerCapture(e.pointerId);
      this.emit(this._value, true);
    });

    this.pad.addEventListener('keydown', (e) => {
      const step = e.shiftKey ? 0.1 : 0.02;
      const [x, y] = this._value;
      const moves = {
        ArrowLeft:  [x - step, y], ArrowRight: [x + step, y],
        ArrowUp:    [x, y - step], ArrowDown:  [x, y + step],
      };
      const next = moves[e.key];
      if (!next) return;
      e.preventDefault();
      this.set([clamp01(next[0]), clamp01(next[1])], true);
    });

    this.set(this._value ?? s.value);
  }

  set(value, notify = false) {
    this._value = value;
    if (this.pad) {
      this.pad.style.setProperty('--x', `${(value[0] * 100).toFixed(2)}%`);
      this.pad.style.setProperty('--y', `${(value[1] * 100).toFixed(2)}%`);
      this.pad.setAttribute('aria-valuenow', value[0].toFixed(2));
      this.pad.setAttribute('aria-valuetext', `x ${value[0].toFixed(2)}, y ${value[1].toFixed(2)}`);
      this.out.textContent = `${value[0].toFixed(2)} · ${value[1].toFixed(2)}`;
    }
    if (notify) this.emit(value);
  }
}

/* ═══ static blocks ═══════════════════════════════════════════════ */

export class AeHint extends Widget {
  render() { this.innerHTML = html`<p class="ctl__hint">${this.spec.text}</p>`; }
}

export class AeGroup extends Widget {
  render() { this.innerHTML = html`<h3 class="group__title">${this.spec.group}</h3>`; }
}

/** Live numbers the scene reports each frame. */
export class AeReadout extends Widget {
  render() { this.innerHTML = html`<dl class="readout"></dl>`; this.dl = this.querySelector('dl'); }

  update(entries) {
    if (!this.dl) return;
    const keys = Object.keys(entries);
    if (this._keys?.join() !== keys.join()) {
      this._keys = keys;
      this.dl.innerHTML = keys.map((k) => html`<dt>${k}</dt><dd data-k="${k}"></dd>`).join('');
      this._cells = new Map(keys.map((k) => [k, this.dl.querySelector(`dd[data-k="${CSS.escape(k)}"]`)]));
    }
    for (const k of keys) {
      const cell = this._cells.get(k);
      const v = String(entries[k]);
      if (cell && cell.textContent !== v) cell.textContent = v;
    }
  }
}

function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }

/* ═══ registration ════════════════════════════════════════════════ */

export const WIDGETS = {
  slider: 'ae-slider',
  switch: 'ae-switch',
  select: 'ae-select',
  xy: 'ae-xy',
  hint: 'ae-hint',
  group: 'ae-group',
  readout: 'ae-readout',
};

customElements.define('ae-slider', AeSlider);
customElements.define('ae-switch', AeSwitch);
customElements.define('ae-select', AeSelect);
customElements.define('ae-xy', AeXY);
customElements.define('ae-hint', AeHint);
customElements.define('ae-group', AeGroup);
customElements.define('ae-readout', AeReadout);
