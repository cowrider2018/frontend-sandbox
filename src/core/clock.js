/* ── core/clock.js ───────────────────────────────────────────────────
   The single requestAnimationFrame loop for the whole app.

   Wall time and simulation time are deliberately separate: pausing or
   slowing a scene must not make its UI animations stutter, and a long
   frame (tab restore, shader recompile) must never be handed to a
   simulation that would explode when integrated with dt = 3 seconds.
   ------------------------------------------------------------------ */

const MAX_DT = 1 / 20; // clamp: better a slow-motion frame than a blow-up

export class Clock {
  constructor(onFrame) {
    this.onFrame = onFrame;
    this.running = false;
    this.paused = false;
    this.speed = 1;

    this.time = 0;        // simulation seconds (scaled, pausable)
    this.wall = 0;        // seconds since start (unscaled)
    this.dt = 1 / 60;     // simulation delta — zero while paused
    this.wallDt = 1 / 60; // real delta — never zero, drives UI easing
    this.frame = 0;
    this.stepOnce = false;

    this._last = 0;
    this._raf = 0;
    this._tick = this._tick.bind(this);
  }

  start() {
    if (this.running) return this;
    this.running = true;
    this._last = performance.now();
    this._raf = requestAnimationFrame(this._tick);
    return this;
  }

  stop() {
    this.running = false;
    cancelAnimationFrame(this._raf);
    return this;
  }

  /** Advance exactly one frame while paused — invaluable for debugging. */
  step() { this.stepOnce = true; return this; }

  setPaused(paused) { this.paused = paused; return this; }
  togglePaused() { this.paused = !this.paused; return this; }

  _tick(now) {
    if (!this.running) return;
    this._raf = requestAnimationFrame(this._tick);

    const raw = (now - this._last) / 1000;
    this._last = now;

    const wallDt = Math.min(raw, MAX_DT);
    this.wallDt = wallDt;
    this.wall += wallDt;

    const advancing = !this.paused || this.stepOnce;
    this.dt = advancing ? wallDt * this.speed : 0;
    this.stepOnce = false;
    this.time += this.dt;
    this.frame++;

    this.onFrame(this);
  }
}
