/* ── core/router.js ──────────────────────────────────────────────────
   Hash routing with the scene's parameters encoded in the URL.

   `#/fluid?curl=28&dye=0.97` is a complete description of what you are
   looking at, so any state worth showing someone is state you can send
   them. Values are written back with `replaceState` while you drag a
   slider — the address bar stays live, the back button stays useful.
   ------------------------------------------------------------------ */

export class Router {
  constructor({ routes, fallback, onChange }) {
    this.routes = routes;      // Set/array of valid ids
    this.fallback = fallback;
    this.onChange = onChange;
    this.current = null;
    this._suppress = false;

    window.addEventListener('hashchange', () => {
      if (this._suppress) { this._suppress = false; return; }
      this._resolve();
    });
  }

  start() { this._resolve(); return this; }

  /** Parse `#/id?a=1&b=2` into `{ id, params }`. */
  parse(hash = location.hash) {
    const raw = hash.replace(/^#\/?/, '');
    const [id, query = ''] = raw.split('?');
    const params = {};
    for (const [k, v] of new URLSearchParams(query)) params[k] = v;
    return { id: decodeURIComponent(id || ''), params };
  }

  _resolve() {
    const { id, params } = this.parse();
    const valid = [...this.routes].includes(id) ? id : this.fallback;
    const changed = valid !== this.current;
    this.current = valid;
    this.onChange({ id: valid, params, changed });
  }

  /** Navigate, pushing a history entry. */
  go(id, params) {
    if (id === this.current && !params) return;
    location.hash = buildHash(id, params);
  }

  /**
   * Update the query string for the current scene without adding a
   * history entry — the right behaviour for a slider being dragged.
   */
  replaceParams(params) {
    const hash = buildHash(this.current, params);
    if (hash === location.hash.replace(/^#/, '')) return;
    this._suppress = true;
    history.replaceState(null, '', `#${hash}`);
    // replaceState does not fire hashchange, so clear the guard ourselves.
    this._suppress = false;
  }
}

function buildHash(id, params) {
  const query = params ? new URLSearchParams(params).toString() : '';
  return `/${id}${query ? `?${query}` : ''}`;
}
