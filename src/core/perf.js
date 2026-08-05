/* ── core/perf.js ────────────────────────────────────────────────────
   Honest performance numbers.

   CPU frame time is easy and mostly useless here: every scene in this
   lab is GPU-bound, and the CPU has usually finished submitting work
   long before the GPU starts it. So we also issue a real GPU timer
   query per frame via EXT_disjoint_timer_query_webgl2. Results arrive
   a few frames late and must be polled, never blocked on — blocking
   would stall the pipeline and change the number you are measuring.
   ------------------------------------------------------------------ */

const HISTORY = 90;

export class Perf {
  constructor(gl, ext) {
    this.gl = gl;
    this.ext = ext;
    this.supported = Boolean(ext);

    this.fps = 0;
    this.frameMs = 0;
    this.gpuMs = 0;
    /** ring buffer of recent frame times, for the HUD sparkline */
    this.history = new Float32Array(HISTORY);
    this.cursor = 0;

    this._pool = [];      // free query objects, reused forever
    this._inflight = [];  // { query } awaiting results
    this._active = null;
    this._acc = 0;
    this._count = 0;
  }

  /** Wrap the frame's GL work. Safe to call when the extension is absent. */
  begin() {
    if (!this.supported || this._active) return;
    const gl = this.gl;
    const query = this._pool.pop() || gl.createQuery();
    gl.beginQuery(this.ext.TIME_ELAPSED_EXT, query);
    this._active = query;
  }

  end() {
    if (!this._active) return;
    this.gl.endQuery(this.ext.TIME_ELAPSED_EXT);
    this._inflight.push(this._active);
    this._active = null;
    this._poll();
  }

  _poll() {
    const gl = this.gl;
    const disjoint = gl.getParameter(this.ext.GPU_DISJOINT_EXT);

    if (disjoint) {
      // The GPU was interrupted (power state change, context switch);
      // every outstanding timing is now meaningless.
      for (const q of this._inflight) this._pool.push(q);
      this._inflight.length = 0;
      return;
    }

    while (this._inflight.length) {
      const query = this._inflight[0];
      if (!gl.getQueryParameter(query, gl.QUERY_RESULT_AVAILABLE)) break;
      const ns = gl.getQueryParameter(query, gl.QUERY_RESULT);
      // Smooth: a single frame's number jitters far too much to read.
      this.gpuMs += (ns / 1e6 - this.gpuMs) * 0.12;
      this._inflight.shift();
      this._pool.push(query);
    }
  }

  /** Feed the CPU-side frame time. Call once per frame. */
  sample(dtSeconds) {
    const ms = dtSeconds * 1000;
    this.frameMs += (ms - this.frameMs) * 0.1;
    this.history[this.cursor] = ms;
    this.cursor = (this.cursor + 1) % HISTORY;

    this._acc += dtSeconds;
    this._count++;
    if (this._acc >= 0.25) {
      this.fps = this._count / this._acc;
      this._acc = 0;
      this._count = 0;
    }
  }

  /** History in chronological order, oldest first. */
  ordered(out = new Float32Array(HISTORY)) {
    for (let i = 0; i < HISTORY; i++) out[i] = this.history[(this.cursor + i) % HISTORY];
    return out;
  }

  get length() { return HISTORY; }
}
