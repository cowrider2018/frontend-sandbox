/* ── core/agent.js ───────────────────────────────────────────────────
   The inhabitant: a sphere that sheds droplets.

   One orb, shared by every scene. It lives in a canonical space — a
   ball of radius `bounds` around the origin — and each scene maps that
   space into its own: world units for the 3D scenes, UV for the 2D
   ones. Because it belongs to the app rather than to any scene, it
   keeps its momentum when you switch scenes.

   The body is a single sphere. What follows it is not a tail: it is a
   handful of *droplets*, thrown off when the orb accelerates and then
   left to coast on their own inertia. Each one is a sphere with its own
   velocity, its own drag, and its own short life. Nothing is a strand,
   nothing is a chain, and there is no rig — the trail is a consequence
   of the body being accelerated, exactly as it is for a blob of liquid.

   The scenes that can afford it blend the droplets back into the orb
   with a smooth minimum, so a droplet stays connected for the first
   moments after it is shed and then pinches off. That single operator
   is what makes a set of independent spheres read as one viscous body.

   Motion is Craig Reynolds' steering: compute a desired velocity, then
   accelerate toward it. Speed is clamped from *below* as well as above,
   because something that can stop dead reads as a particle rather than
   as a body with mass. On top of that sit two impulse primitives —
   `reflect` and `displace` — which is all a collision response needs,
   and all a jump between two bodies would need either.
   ------------------------------------------------------------------ */

export class Agent {
  constructor({
    drops = 10,
    bounds = 1.9,
    radius = 0.20,
    dropLife = 1.35,
    dropDrag = 1.9,
  } = {}) {
    this.bounds = bounds;
    this.radius = radius;
    this.dropCapacity = drops;
    this.dropLifespan = dropLife;
    this.dropDrag = dropDrag;

    this.head = new Float32Array(3);
    this.vel = new Float32Array([0.4, 0.1, 0.25]);
    this.accel = new Float32Array(3);
    this.speed = 0;
    this.time = 0;
    /** rises to 1 on impact, decays back — scenes flash the body with it */
    this.impact = 0;

    // Droplet pool. Slots are reused oldest-first; nothing is allocated
    // after construction.
    this._dropPos = new Float32Array(drops * 3);
    this._dropVel = new Float32Array(drops * 3);
    this._dropSize = new Float32Array(drops);
    this._dropLife = new Float32Array(drops);
    this._next = 0;
    this._budget = 0;

    /**
     * Flat view for renderers: node 0 is the orb, then every live
     * droplet, newest first. `live` is how many of them are real.
     */
    this.count = drops + 1;
    this.nodes = new Float32Array(this.count * 3);
    this.radii = new Float32Array(this.count);
    this.live = 1;

    this._aim = new Float32Array(3);
    this._aimAt = -1e9;
    this._target = new Float32Array(3);
    this._prevVel = new Float32Array(3);

    this.reset();
  }

  reset() {
    this.head.set([0, 0, 0]);
    this.vel.set([0.4, 0.1, 0.25]);
    this._prevVel.set(this.vel);
    this._dropLife.fill(0);
    this._dropSize.fill(0);
    this._next = 0;
    this._budget = 0;
    this.time = 0;
    this.impact = 0;
    this._pack();
  }

  /**
   * Point the orb at somewhere in canonical space. Scenes call this
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

  /* ── impulses ─────────────────────────────────────────────────── */

  /**
   * Bounce off a surface. `bounce` below 1 loses energy on impact; the
   * tangential component is kept intact, so a glancing hit skims and a
   * head-on hit rebounds — which is the whole feel of a bouncing game.
   */
  reflect(nx, ny, nz, bounce = 0.85) {
    const v = this.vel;
    const dot = v[0] * nx + v[1] * ny + v[2] * nz;
    if (dot >= 0) return false;          // already moving away
    const k = (1 + bounce) * dot;
    v[0] -= k * nx;
    v[1] -= k * ny;
    v[2] -= k * nz;
    this.impact = 1;
    // A hard bounce throws off a spray, the same way a struck droplet does.
    this._budget += 3.5;
    return true;
  }

  /** Move the orb bodily, without touching its velocity. */
  displace(dx, dy, dz) {
    this.head[0] += dx;
    this.head[1] += dy;
    this.head[2] += dz;
  }

  /* ── update ───────────────────────────────────────────────────── */

  /**
   * `flatten` (0–1) pulls the orb toward the z = 0 plane. The 2D scenes
   * project it onto their canvas, and without this a good fraction of
   * its motion happens along the axis they discard — it reads as the
   * thing repeatedly stalling.
   */
  update(dt, { mode = 'wander', speed = 1, agility = 3.2, flatten = 0, shed = 1 } = {}) {
    if (dt <= 0) return this;
    this.time += dt;
    this.impact *= Math.exp(-dt * 5.5);

    this._prevVel.set(this.vel);
    this._pickTarget(mode);
    if (flatten > 0) this._target[2] *= 1 - flatten;
    this._steer(dt, speed, speed * 0.42, agility);
    this._measureAccel(dt);
    this._advance(dt, flatten);
    this._shed(dt, shed, flatten);
    this._stepDrops(dt, flatten);
    this._pack();

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
    const h = this.head;
    const v = this.vel;
    const T = this._target;

    let dx = T[0] - h[0], dy = T[1] - h[1], dz = T[2] - h[2];
    const dist = Math.hypot(dx, dy, dz) || 1;
    dx /= dist; dy /= dist; dz /= dist;

    // Accelerate toward the desired velocity rather than snapping to it.
    // Steering is eased off right after an impact so a bounce actually
    // reads as a bounce instead of being corrected away within a frame.
    const k = agility * dt * (1 - this.impact * 0.85);
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
    // A bounce is allowed to briefly exceed the cruise speed.
    const ceiling = maxSpeed * (1 + this.impact * 0.9);
    const clamped = s > ceiling ? ceiling : s < minSpeed ? minSpeed : s;
    if (clamped !== s) {
      const f = clamped / s;
      v[0] *= f; v[1] *= f; v[2] *= f;
    }
    this.speed = clamped;
  }

  _measureAccel(dt) {
    const a = this.accel;
    a[0] = (this.vel[0] - this._prevVel[0]) / dt;
    a[1] = (this.vel[1] - this._prevVel[1]) / dt;
    a[2] = (this.vel[2] - this._prevVel[2]) / dt;
  }

  _advance(dt, flatten = 0) {
    const h = this.head;
    const v = this.vel;
    h[0] += v[0] * dt;
    h[1] += v[1] * dt;
    h[2] += v[2] * dt;

    if (flatten > 0) {
      const k = Math.min(1, flatten * 5 * dt);
      h[2] -= h[2] * k;
      v[2] -= v[2] * k;
    }
  }

  /* ── droplets ─────────────────────────────────────────────────── */

  /**
   * Shedding rate is driven by acceleration, not speed. A body moving
   * steadily through nothing has no reason to come apart; a body being
   * yanked into a turn does. Cruising still sheds a little, or the orb
   * looks inert while it drifts.
   */
  _shed(dt, gain, flatten) {
    if (gain <= 0) return;
    const a = Math.hypot(this.accel[0], this.accel[1], this.accel[2]);
    this._budget += (a * 0.55 + this.speed * 2.4) * gain * dt;

    // Cap the per-frame spawn count: a single very long frame must not
    // empty the whole pool at one point in space.
    let guard = 3;
    while (this._budget >= 1 && guard-- > 0) {
      this._budget -= 1;
      this._spawn(flatten);
    }
    if (this._budget > 3) this._budget = 3;
  }

  _spawn(flatten) {
    const i = this._next;
    this._next = (this._next + 1) % this.dropCapacity;

    const s = Math.max(this.speed, 1e-4);
    const hx = this.vel[0] / s, hy = this.vel[1] / s, hz = this.vel[2] / s;

    // Behind the orb, on its trailing surface.
    const back = this.radius * 0.85;
    const jitter = this.radius * 0.35;
    const jx = (Math.random() * 2 - 1) * jitter;
    const jy = (Math.random() * 2 - 1) * jitter;
    const jz = (Math.random() * 2 - 1) * jitter * (1 - flatten);

    const o = i * 3;
    this._dropPos[o + 0] = this.head[0] - hx * back + jx;
    this._dropPos[o + 1] = this.head[1] - hy * back + jy;
    this._dropPos[o + 2] = this.head[2] - hz * back + jz;

    // It keeps some of the parent's momentum, which is why it lags
    // behind rather than being dropped dead in place.
    const keep = 0.42 + Math.random() * 0.28;
    this._dropVel[o + 0] = this.vel[0] * keep + jx * 2.2;
    this._dropVel[o + 1] = this.vel[1] * keep + jy * 2.2;
    this._dropVel[o + 2] = this.vel[2] * keep + jz * 2.2;

    this._dropSize[i] = this.radius * (0.34 + Math.random() * 0.26);
    this._dropLife[i] = 1;
  }

  _stepDrops(dt, flatten) {
    const drag = Math.exp(-dt * this.dropDrag);
    const decay = dt / this.dropLifespan;

    for (let i = 0; i < this.dropCapacity; i++) {
      if (this._dropLife[i] <= 0) continue;
      const o = i * 3;

      this._dropVel[o] *= drag;
      this._dropVel[o + 1] *= drag;
      this._dropVel[o + 2] *= drag;

      this._dropPos[o] += this._dropVel[o] * dt;
      this._dropPos[o + 1] += this._dropVel[o + 1] * dt;
      this._dropPos[o + 2] += this._dropVel[o + 2] * dt;

      if (flatten > 0) {
        const k = Math.min(1, flatten * 5 * dt);
        this._dropPos[o + 2] -= this._dropPos[o + 2] * k;
      }

      this._dropLife[i] -= decay;
      if (this._dropLife[i] < 0) this._dropLife[i] = 0;
    }
  }

  /** Compact the orb + live droplets into the flat render view. */
  _pack() {
    this.nodes[0] = this.head[0];
    this.nodes[1] = this.head[1];
    this.nodes[2] = this.head[2];
    this.radii[0] = this.radius;

    let n = 1;
    // Walk the ring backwards from the most recently filled slot, so
    // the packed order is newest droplet first.
    for (let k = 0; k < this.dropCapacity; k++) {
      const i = (this._next - 1 - k + this.dropCapacity * 2) % this.dropCapacity;
      const life = this._dropLife[i];
      if (life <= 0) continue;
      const o = i * 3;
      this.nodes[n * 3 + 0] = this._dropPos[o];
      this.nodes[n * 3 + 1] = this._dropPos[o + 1];
      this.nodes[n * 3 + 2] = this._dropPos[o + 2];
      // Droplets shrink as they age, which is what sells them as
      // evaporating rather than merely fading out.
      this.radii[n] = this._dropSize[i] * Math.pow(life, 0.45);
      n++;
    }

    // Unused slots are zeroed so a renderer that ignores `live` still
    // draws nothing rather than stale geometry.
    for (let j = n; j < this.count; j++) {
      this.radii[j] = 0;
      this.nodes[j * 3] = this.nodes[0];
      this.nodes[j * 3 + 1] = this.nodes[1];
      this.nodes[j * 3 + 2] = this.nodes[2];
    }
    this.live = n;
  }

  /* ── consumers ────────────────────────────────────────────────── */

  /** The orb itself as vec4(x, y, z, radius) in a scene's own space. */
  orb(out, scale = 1, offsetY = 0, girth = 1) {
    out[0] = this.head[0] * scale;
    out[1] = this.head[1] * scale + offsetY;
    out[2] = this.head[2] * scale;
    out[3] = this.radius * scale * girth;
    return out;
  }

  /**
   * Orb + droplets as vec4(x, y, z, radius), scaled into a scene's own
   * space. Writes `capacity` entries; entries past the live count carry
   * radius 0 and sit on the orb, so a shader can either test the radius
   * or simply let them contribute nothing.
   */
  bodies(out, capacity, scale = 1, offsetY = 0, girth = 1) {
    const n = Math.min(capacity, this.count);
    for (let i = 0; i < n; i++) {
      out[i * 4 + 0] = this.nodes[i * 3] * scale;
      out[i * 4 + 1] = this.nodes[i * 3 + 1] * scale + offsetY;
      out[i * 4 + 2] = this.nodes[i * 3 + 2] * scale;
      out[i * 4 + 3] = this.radii[i] * scale * girth;
    }
    return Math.min(this.live, capacity);
  }

  /** Bounding sphere of orb + droplets, in a scene's own space. */
  boundingSphere(out, scale = 1, offsetY = 0, girth = 1) {
    let r = 0;
    for (let i = 0; i < this.live; i++) {
      const d = Math.hypot(
        this.nodes[i * 3] - this.head[0],
        this.nodes[i * 3 + 1] - this.head[1],
        this.nodes[i * 3 + 2] - this.head[2],
      ) + this.radii[i] * girth;
      if (d > r) r = d;
    }
    out[0] = this.head[0] * scale;
    out[1] = this.head[1] * scale + offsetY;
    out[2] = this.head[2] * scale;
    out[3] = Math.max(r, this.radius * girth) * scale;
    return out;
  }
}
