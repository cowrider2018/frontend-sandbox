/* ── core/agent.js ───────────────────────────────────────────────────
   The inhabitant.

   One creature, shared by every scene. It lives in a canonical space —
   a ball of radius `bounds` around the origin — and each scene maps
   that space into its own: world units for the 3D scenes, UV for the
   2D ones. Because it belongs to the app rather than to any scene, it
   keeps its momentum when you switch scenes; the fluid you were just
   stirring hands the same swimmer, mid-turn, to the raymarcher.

   Two systems, kept apart on purpose:

   1. STEERING decides where the head wants to go. Craig Reynolds'
      formulation: compute a desired velocity, and accelerate toward it.
      Speed is clamped from *below* as well as above, because a creature
      that can stop dead reads as a particle, not an animal.

   2. The SPINE follows. Each node is pulled to sit exactly one segment
      behind the node in front of it — follow-the-leader inverse
      kinematics. Nothing simulates the body; it is a pure consequence
      of where the head has been, which is why it never tangles and
      never needs a solver.

   Undulation is added on top as display-only offset. Folding it into
   the chain would feed the wiggle back into the constraint and the body
   would slowly saw itself apart.
   ------------------------------------------------------------------ */

const UP = [0, 1, 0];

export class Agent {
  constructor({
    nodes = 34,
    bounds = 1.9,
    segment = 0.082,
    thickness = 0.082,
  } = {}) {
    this.count = nodes;
    this.bounds = bounds;
    this.segment = segment;
    this.thickness = thickness;

    /** physics chain — head first */
    this.chain = new Float32Array(nodes * 3);
    /** display nodes = chain + undulation; this is what scenes read */
    this.nodes = new Float32Array(nodes * 3);
    this.radii = new Float32Array(nodes);

    this.head = this.nodes.subarray(0, 3);
    this.vel = new Float32Array([0.4, 0.1, 0.25]);
    this.speed = 0;
    this.time = 0;
    this.phase = 0;

    this._aim = new Float32Array(3);
    this._aimAt = -1e9;
    this._target = new Float32Array(3);
    this._tmp = new Float32Array(3);

    this._profile();
    this.reset();
  }

  /** Fish taper: slim nose, shoulders at ~25%, vanishing tail. */
  _profile() {
    const n = this.count;
    for (let i = 0; i < n; i++) {
      const u = i / (n - 1);
      this.radii[i] = this.thickness * Math.sin(Math.PI * Math.pow(u * 0.85 + 0.06, 0.6));
    }
  }

  reset() {
    const { chain, nodes, count, segment } = this;
    for (let i = 0; i < count; i++) {
      chain[i * 3 + 0] = -segment * i * 0.6;
      chain[i * 3 + 1] = 0;
      chain[i * 3 + 2] = -segment * i * 0.8;
    }
    nodes.set(chain);
    this.vel.set([0.4, 0.1, 0.25]);
    this.time = 0;
    this.phase = 0;
  }

  /**
   * Point the creature at somewhere in canonical space. Scenes call this
   * during their frame; the value is consumed on the *next* update, so
   * a scene can derive it from its own camera without a circular
   * dependency. One frame of lag is invisible at 60 Hz.
   */
  aim(x, y, z) {
    this._aim[0] = x;
    this._aim[1] = y;
    this._aim[2] = z;
    this._aimAt = this.time;
  }

  get hasAim() { return this.time - this._aimAt < 0.4; }

  /* ── update ───────────────────────────────────────────────────── */

  /**
   * `flatten` (0–1) pulls the creature toward the z = 0 plane. The 2D
   * scenes project the body onto their canvas, and without this a good
   * fraction of its motion happens along the axis they discard — it
   * reads as the animal repeatedly stalling.
   */
  update(dt, { mode = 'wander', speed = 1, agility = 3.2, flatten = 0 } = {}) {
    if (dt <= 0) return this;
    this.time += dt;

    const maxSpeed = speed;
    const minSpeed = speed * 0.42;

    this._pickTarget(mode);
    if (flatten > 0) this._target[2] *= 1 - flatten;
    this._steer(dt, maxSpeed, minSpeed, agility);
    this._advance(dt, flatten);
    this._solveSpine();
    this._undulate(dt);

    return this;
  }

  _pickTarget(mode) {
    const t = this.time;
    const R = this.bounds * 0.78;

    // Three mutually irrational frequencies: the path never repeats, and
    // it never settles into a plane the way two frequencies would.
    const wx = R * Math.sin(t * 0.37) * Math.cos(t * 0.231);
    const wy = R * 0.62 * Math.sin(t * 0.293 + 1.7);
    const wz = R * Math.cos(t * 0.411 + 2.3) * Math.cos(t * 0.187);

    const T = this._target;
    if (mode === 'wander' || !this.hasAim) {
      T[0] = wx; T[1] = wy; T[2] = wz;
      return;
    }

    const a = this._aim;
    if (mode === 'flee') {
      // Run for the far side of the volume, but keep some wander in so
      // the escape curves instead of being a straight line.
      T[0] = -a[0] * 0.9 + wx * 0.45;
      T[1] = -a[1] * 0.9 + wy * 0.45;
      T[2] = -a[2] * 0.9 + wz * 0.45;
      return;
    }

    // follow: mostly the pointer, with a little wander so it circles
    // its target rather than sitting on top of it.
    T[0] = a[0] * 0.86 + wx * 0.20;
    T[1] = a[1] * 0.86 + wy * 0.20;
    T[2] = a[2] * 0.86 + wz * 0.20;
  }

  _steer(dt, maxSpeed, minSpeed, agility) {
    const h = this.chain;
    const v = this.vel;
    const T = this._target;

    let dx = T[0] - h[0], dy = T[1] - h[1], dz = T[2] - h[2];
    const dist = Math.hypot(dx, dy, dz) || 1;
    dx /= dist; dy /= dist; dz /= dist;

    // Accelerate toward the desired velocity rather than snapping to it:
    // the lag is what produces banking on the turns.
    const k = agility * dt;
    v[0] += (dx * maxSpeed - v[0]) * k;
    v[1] += (dy * maxSpeed - v[1]) * k;
    v[2] += (dz * maxSpeed - v[2]) * k;

    // Soft wall: only engages outside the boundary, so the interior
    // motion stays unconstrained.
    const r = Math.hypot(h[0], h[1], h[2]);
    if (r > this.bounds) {
      const push = (r - this.bounds) * 5.0 * dt / r;
      v[0] -= h[0] * push;
      v[1] -= h[1] * push;
      v[2] -= h[2] * push;
    }

    let s = Math.hypot(v[0], v[1], v[2]);
    if (s < 1e-5) {
      v[0] = minSpeed; v[1] = 0; v[2] = 0;
      s = minSpeed;
    }
    const clamped = s > maxSpeed ? maxSpeed : s < minSpeed ? minSpeed : s;
    if (clamped !== s) {
      const f = clamped / s;
      v[0] *= f; v[1] *= f; v[2] *= f;
    }
    this.speed = clamped;
  }

  _advance(dt, flatten = 0) {
    const h = this.chain;
    const v = this.vel;
    h[0] += v[0] * dt;
    h[1] += v[1] * dt;
    h[2] += v[2] * dt;

    if (flatten > 0) {
      // Ease the head onto the plane rather than clamping it: a hard
      // clamp would leave the spine trailing off-plane behind a head
      // that is pinned to it, and the body would look kinked.
      const k = Math.min(1, flatten * 5 * dt);
      h[2] -= h[2] * k;
      v[2] -= v[2] * k;
    }
  }

  /** Follow-the-leader: each node sits exactly one segment behind. */
  _solveSpine() {
    const c = this.chain;
    const seg = this.segment;
    for (let i = 1; i < this.count; i++) {
      const a = (i - 1) * 3;
      const b = i * 3;
      let dx = c[b] - c[a], dy = c[b + 1] - c[a + 1], dz = c[b + 2] - c[a + 2];
      let d = Math.hypot(dx, dy, dz);
      if (d < 1e-6) {
        // Degenerate: fall back to trailing straight behind the head.
        dx = -this.vel[0]; dy = -this.vel[1]; dz = -this.vel[2];
        d = Math.hypot(dx, dy, dz) || 1;
      }
      const k = seg / d;
      c[b] = c[a] + dx * k;
      c[b + 1] = c[a + 1] + dy * k;
      c[b + 2] = c[a + 2] + dz * k;
    }
  }

  /**
   * Lateral travelling wave, applied to the display copy only. Tail
   * beat frequency rises with speed — the single cue that reads most
   * strongly as "swimming" rather than "being dragged".
   */
  _undulate(dt) {
    this.phase += dt * (3.4 + this.speed * 4.2);

    const c = this.chain;
    const n = this.nodes;
    const count = this.count;
    const amp = this.segment * 1.35;

    for (let i = 0; i < count; i++) {
      const o = i * 3;

      // Body direction at this node.
      const p = i === 0 ? 0 : (i - 1) * 3;
      const q = i === 0 ? 3 : o;
      let bx = c[p] - c[q], by = c[p + 1] - c[q + 1], bz = c[p + 2] - c[q + 2];
      let bl = Math.hypot(bx, by, bz) || 1;
      bx /= bl; by /= bl; bz /= bl;

      // side = normalize(cross(body, worldUp))
      let sx = by * UP[2] - bz * UP[1];
      let sy = bz * UP[0] - bx * UP[2];
      let sz = bx * UP[1] - by * UP[0];
      const sl = Math.hypot(sx, sy, sz);
      if (sl < 1e-4) { sx = 1; sy = 0; sz = 0; } else { sx /= sl; sy /= sl; sz /= sl; }

      // vertical = cross(side, body)
      const vx = sy * bz - sz * by;
      const vy = sz * bx - sx * bz;
      const vz = sx * by - sy * bx;

      const u = i / (count - 1);
      // The head barely moves; amplitude grows toward the tail.
      const taper = u * u * (3 - 2 * u);
      const ph = this.phase - i * 0.42;
      const lat = Math.sin(ph) * amp * taper;
      const vert = Math.cos(ph * 0.7) * amp * taper * 0.3;

      n[o] = c[o] + sx * lat + vx * vert;
      n[o + 1] = c[o + 1] + sy * lat + vy * vert;
      n[o + 2] = c[o + 2] + sz * lat + vz * vert;
    }
  }

  /* ── consumers ────────────────────────────────────────────────── */

  /**
   * Resample the body down to `k` nodes as vec4(x, y, z, radius),
   * scaled and offset into a scene's own space. Used by the raymarcher,
   * where every extra capsule is paid for once per march step.
   */
  resample(k, out, scale = 1, offsetY = 0, girth = 1) {
    const last = this.count - 1;
    for (let j = 0; j < k; j++) {
      const i = Math.round((j / (k - 1)) * last);
      const o = i * 3;
      out[j * 4 + 0] = this.nodes[o] * scale;
      out[j * 4 + 1] = this.nodes[o + 1] * scale + offsetY;
      out[j * 4 + 2] = this.nodes[o + 2] * scale;
      // `girth` is separate from `scale` because a resampled spine has
      // longer segments, and a body drawn at its true radius over those
      // gaps looks like a string of beads rather than one animal.
      out[j * 4 + 3] = this.radii[i] * scale * girth;
    }
    return out;
  }

  /** Bounding sphere of the whole body, in a scene's own space. */
  boundingSphere(out, scale = 1, offsetY = 0) {
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < this.count; i++) {
      cx += this.nodes[i * 3];
      cy += this.nodes[i * 3 + 1];
      cz += this.nodes[i * 3 + 2];
    }
    cx /= this.count; cy /= this.count; cz /= this.count;

    let r = 0;
    for (let i = 0; i < this.count; i++) {
      const d = Math.hypot(
        this.nodes[i * 3] - cx,
        this.nodes[i * 3 + 1] - cy,
        this.nodes[i * 3 + 2] - cz,
      ) + this.radii[i];
      if (d > r) r = d;
    }

    out[0] = cx * scale;
    out[1] = cy * scale + offsetY;
    out[2] = cz * scale;
    out[3] = r * scale;
    return out;
  }
}
