/* ── ui/panel.js ─────────────────────────────────────────────────────
   Schema in, control panel out.

   A scene declares what it can be told to do as plain data; the panel
   turns that into widgets, owns the resulting state object, and pushes
   every change back into the URL. A new scene never touches DOM code —
   which is the point of describing UI as data in the first place.
   ------------------------------------------------------------------ */

import { WIDGETS } from './widgets.js';

export class Panel {
  /**
   * @param {HTMLElement} host  container to render into
   * @param {(id:string, value:any, commit:boolean)=>void} onChange
   */
  constructor(host, onChange) {
    this.host = host;
    this.onChange = onChange;
    this.widgets = new Map();
    this.state = {};
    this.readout = null;

    // One delegated listener for the whole panel, whatever it contains.
    host.addEventListener('ctl-input', (e) => {
      const { id, value, commit } = e.detail;
      this.state[id] = value;
      this.onChange(id, value, commit);
    });
  }

  /** Rebuild for a new scene. Returns the freshly defaulted state. */
  build(spec, overrides = {}) {
    this.host.replaceChildren();
    this.widgets.clear();
    this.state = {};

    const frag = document.createDocumentFragment();
    let group = null;

    for (const item of spec) {
      const type = item.group ? 'group' : item.type;
      const tag = WIDGETS[type];
      if (!tag) { console.warn('Panel: unknown control type', type); continue; }

      const el = document.createElement(tag);
      el.spec = item;

      if (item.group) {
        group = document.createElement('div');
        group.className = 'group';
        group.append(el);
        frag.append(group);
        continue;
      }

      if (item.id !== undefined) {
        const value = coerce(item, overrides[item.id], item.value);
        el._value = value;
        this.state[item.id] = value;
        this.widgets.set(item.id, el);
      }

      (group ?? frag).append(el);
    }

    // Live scene telemetry always lands at the bottom.
    const stats = document.createElement('div');
    stats.className = 'group';
    this.readout = document.createElement(WIDGETS.readout);
    this.readout.spec = {};
    stats.append(Object.assign(document.createElement('h3'), {
      className: 'group__title', textContent: '即時狀態',
    }), this.readout);
    frag.append(stats);

    this.host.append(frag);
    return this.state;
  }

  /** Programmatic update — used by presets and by shared URLs. */
  setValues(values, { notify = false } = {}) {
    for (const [id, value] of Object.entries(values)) {
      const el = this.widgets.get(id);
      if (!el) continue;
      const spec = el.spec;
      const v = coerce(spec, value, this.state[id]);
      this.state[id] = v;
      el.set(v);
      if (notify) this.onChange(id, v, true);
    }
  }

  updateReadout(entries) { this.readout?.update(entries); }

  /** Only what differs from the defaults, so shared URLs stay short. */
  serialise(spec) {
    const out = {};
    for (const item of spec) {
      if (item.id === undefined) continue;
      const v = this.state[item.id];
      const d = item.value;
      if (Array.isArray(v)) {
        if (v.some((n, i) => Math.abs(n - d[i]) > 1e-6)) out[item.id] = v.map((n) => round(n)).join(',');
      } else if (typeof v === 'number') {
        if (Math.abs(v - d) > 1e-9) out[item.id] = String(round(v));
      } else if (typeof v === 'boolean') {
        if (v !== d) out[item.id] = v ? '1' : '0';
      } else if (v !== d) {
        out[item.id] = String(v);
      }
    }
    return out;
  }
}

/** URL params are strings; widget specs say what they should have been. */
function coerce(spec, raw, fallback) {
  if (raw === undefined || raw === null) return fallback;
  switch (spec.type) {
    case 'slider': {
      const n = Number(raw);
      if (!Number.isFinite(n)) return fallback;
      return Math.min(spec.max, Math.max(spec.min, n));
    }
    case 'switch':
      return typeof raw === 'boolean' ? raw : raw === '1' || raw === 'true';
    case 'select': {
      const s = String(raw);
      return spec.options.some((o) => String(o.value) === s) ? s : fallback;
    }
    case 'xy': {
      const parts = Array.isArray(raw) ? raw : String(raw).split(',').map(Number);
      if (parts.length !== 2 || parts.some((n) => !Number.isFinite(n))) return fallback;
      return parts.map((n) => Math.min(1, Math.max(0, n)));
    }
    default:
      return raw;
  }
}

function round(n) { return Math.round(n * 10000) / 10000; }
