/* ── core/pointer.js ─────────────────────────────────────────────────
   Unified mouse / pen / touch input, normalised for shaders.

   Pointer Events give us one code path for all three devices, so there
   is no touch-vs-mouse branching anywhere else in the app. Movement is
   accumulated per-frame rather than per-event: a 1000 Hz mouse fires ~16
   events between frames, and a simulation wants their *sum*, not the
   last one. Velocity is smoothed so releasing the pointer glides instead
   of stopping dead.
   ------------------------------------------------------------------ */

export class Pointer {
  constructor(element, { onDown, onUp } = {}) {
    this.el = element;
    this.onDown = onDown;
    this.onUp = onUp;

    /** normalised 0..1, origin top-left */
    this.x = 0.5;
    this.y = 0.5;
    /** movement accumulated since the last frame, in normalised units */
    this.dx = 0;
    this.dy = 0;
    /** low-pass filtered velocity, survives pointerup */
    this.vx = 0;
    this.vy = 0;
    this.down = false;
    this.active = false;      // has the user ever interacted?
    this.buttons = 0;
    /** second finger, for pinch-driven parameters */
    this.pinch = 0;

    this._pending = { dx: 0, dy: 0 };
    this._points = new Map();
    this._pinchStart = 0;
    this._bind();
  }

  _bind() {
    const el = this.el;
    const opts = { passive: false };

    el.addEventListener('pointerdown', (e) => {
      el.setPointerCapture?.(e.pointerId);
      this._points.set(e.pointerId, this._norm(e));
      if (this._points.size === 1) {
        const p = this._norm(e);
        this.x = p.x; this.y = p.y;
        this.down = true;
        this.active = true;
        this.buttons = e.buttons;
        el.classList.add('is-grabbing');
        this.onDown?.(this);
      } else if (this._points.size === 2) {
        this._pinchStart = this._spread();
      }
      e.preventDefault();
    }, opts);

    el.addEventListener('pointermove', (e) => {
      // Self-heal a missed press. A pointerdown can legitimately never
      // reach us — it lands on an overlay that closes on the same event,
      // or on a modal that was just dismissed — and then the button is
      // held with no drag in progress. Adopting it here means a drag can
      // never get stuck in a half-started state.
      if ((e.buttons & 1) && !this.down && this._points.size === 0) {
        this._points.set(e.pointerId, this._norm(e));
        const p = this._norm(e);
        this.x = p.x; this.y = p.y;
        this.down = true;
        this.active = true;
        this.buttons = e.buttons;
        el.classList.add('is-grabbing');
        this.onDown?.(this);
      }

      // coalesced events recover the full high-frequency path that the
      // browser batched into this single frame's dispatch.
      const events = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
      for (const ev of events) {
        const p = this._norm(ev);
        if (this._points.has(ev.pointerId)) this._points.set(ev.pointerId, p);
        if (this._points.size > 1 && ev.pointerId !== this._firstId()) continue;
        this._pending.dx += p.x - this.x;
        this._pending.dy += p.y - this.y;
        this.x = p.x;
        this.y = p.y;
      }
      this.active = true;
      if (this._points.size === 2 && this._pinchStart > 0) {
        this.pinch = this._spread() / this._pinchStart - 1;
      }
      e.preventDefault();
    }, opts);

    const release = (e) => {
      this._points.delete(e.pointerId);
      if (this._points.size === 0) {
        this.down = false;
        this.buttons = 0;
        this.pinch = 0;
        this._pinchStart = 0;
        el.classList.remove('is-grabbing');
        this.onUp?.(this);
      }
    };

    // Deliberately NOT pointerleave. Calling setPointerCapture moves the
    // event target, and the browser announces that with an out/leave pair
    // on the element we just captured to — so treating leave as "the drag
    // ended" cancels the drag on the very frame it began. With capture in
    // place, pointerup is guaranteed to arrive even outside the element,
    // which is exactly what leave was there to cover.
    el.addEventListener('pointerup', release);
    el.addEventListener('pointercancel', release);
    el.addEventListener('lostpointercapture', release);
    el.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  _firstId() { return this._points.keys().next().value; }

  _norm(e) {
    const r = this.el.getBoundingClientRect();
    return {
      x: (e.clientX - r.left) / r.width,
      y: (e.clientY - r.top) / r.height,
    };
  }

  _spread() {
    const [a, b] = [...this._points.values()];
    return Math.hypot(a.x - b.x, a.y - b.y);
  }

  /**
   * Fold this frame's accumulated movement in. Call once per frame,
   * before the scene reads `dx`/`dy`.
   */
  update(dt) {
    this.dx = this._pending.dx;
    this.dy = this._pending.dy;
    this._pending.dx = 0;
    this._pending.dy = 0;

    // Critically-damped-ish smoothing, frame-rate independent.
    const k = 1 - Math.exp(-dt * 14);
    const instX = dt > 0 ? this.dx / dt : 0;
    const instY = dt > 0 ? this.dy / dt : 0;
    this.vx += (instX - this.vx) * k;
    this.vy += (instY - this.vy) * k;
    return this;
  }

  get moved() { return this.dx !== 0 || this.dy !== 0; }
  get speed() { return Math.hypot(this.vx, this.vy); }
}
