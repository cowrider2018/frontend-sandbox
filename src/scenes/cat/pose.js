/* ── scenes/cat/pose.js ──────────────────────────────────────────────
   What the cat is doing, and how that reaches the bones.

   Two halves, deliberately separate:

     Driver     decides the pose — a walk cycle when it is moving, idle
                breathing and blinking when it is not
     applyPose  writes that pose onto the rig

   The split is the one the original model used, and it is worth keeping:
   `applyPose` is the contract with the skeleton, so a different driver
   (a mood brain, say, or a path follower) can be swapped in later
   without touching a single bone name.
   ------------------------------------------------------------------ */

/* Rest offsets the model bakes in and the pose has to build on top of.
   Taken from cat-model.js, where they are the same constants. */
const HEAD_LEAN = 0.12;   // the head sits slightly forward at rest
const TAIL_REST = -0.15;  // tail's resting roll

/* ── the tail's spring chain ──
   A tail is not a stick. It is carried by the body, and every point
   along it is dragged by the point nearer the base — so the tip is the
   last thing to start moving and the last thing to stop.

   Modelled as a chain of nodes, each pulled toward its inboard
   neighbour by a spring. The pull is one-way: a node never tugs back on
   the one ahead of it, so the body leads and the tail follows instead
   of the two negotiating. The lag accumulates down the chain, which is
   where the softness comes from — nobody authored a delay curve.

   Constants are the model's own, so the tail moves the way it always
   did. K: bigger is stiffer and snappier. C: bigger settles sooner. */
/* Nodes along a soft part. Sixteen segments and not eight, and the
   number is load-bearing rather than a comfort: the turn a segment
   carries is what folds a tube. Reversing this tail's own curve put 30
   to 37 degrees on each of eight segments, and at that rate the rotation
   applied across a segment outruns the segment — the inside of the bend
   turns further than it travels, and the surface passes through itself.
   Counted on the tail's own triangles, eight segments turned 355 of 3456
   of them inside out. Halving the angle takes the margin back. */
const SEG = 16;
const CH_K = 240;
const CH_C = 15;

/* ── the tail's own line ──
   Where the rest pose puts the centre of the tail at each of the nine
   nodes, measured off the mesh rather than assumed: the bake curves this
   tail hard — nearly four tenths of its length off the chord — so a bend
   applied about a straight axis would be bending the wrong thing.

   It exists because of *folding*. The deformation used to be "turn every
   vertex about the bone's origin by the angle at its own outerness",
   which is not a bend at all: a point at distance y sweeps an arc of
   radius y, so where y·dθ/dy climbs past one the outer rings overtake
   the inner ones and the surface creases into itself. At the walk's two
   or three tenths of a radian that is invisible. At the whirl's two and
   a half it is the first thing anyone sees — and it cannot be tuned
   away, because the gradient that folds the tail *is* the bend.

   So each ring is now moved rather than swung: rotated about its own
   place on this line, and then set down on the line's bent copy, which
   the CPU integrates one segment at a time. Rings can no longer pass
   through each other, because nothing is being swung past anything. */
const TAIL_AXIS = [
  [0.0143, 0.0418, -0.0334],
  [0.0143, 0.1674, -0.1336],
  [0.0143, 0.3348, -0.2673],
  [0.0143, 0.4601, -0.3679],
  [0.0143, 0.6222, -0.5078],
  [0.0140, 0.8239, -0.6916],
  [0.0140, 0.9426, -0.7806],
  [0.0141, 1.0988, -0.8711],
  [0.0143, 1.2850, -0.9439],
  [0.0140, 1.5485, -0.9984],
  [0.0140, 1.7333, -0.9876],
  [0.0140, 1.8596, -0.9510],
  [0.0141, 2.0868, -0.8062],
  [0.0139, 2.2143, -0.6736],
  [0.0140, 2.2832, -0.5405],
  [0.0142, 2.3377, -0.3679],
  [0.0014, 2.3985, -0.1488],
];

/** How thick the tail is, measured off the mesh: the mean distance of
    its vertices from its own centreline, which is nearly the same all
    the way along. It is what decides how far the thing can be bent —
    see maxTurn. */
const TAIL_RADIUS = 0.215;

/* The same line, chewed once at load: which way each segment runs, how
   long it is, and the turn that would put it back in line with the
   first — which is how the hook is taken out. */
const TAIL_SEG = (() => {
  const out = [null];
  for (let i = 1; i < TAIL_AXIS.length; i++) {
    const d = [
      TAIL_AXIS[i][0] - TAIL_AXIS[i - 1][0],
      TAIL_AXIS[i][1] - TAIL_AXIS[i - 1][1],
      TAIL_AXIS[i][2] - TAIL_AXIS[i - 1][2],
    ];
    const len = Math.hypot(d[0], d[1], d[2]);
    out.push({ dir: d.map((v) => v / len), len });
  }
  /* And how much each segment turns off the one before it. Off the one
     before it, and not off the first one, which is the difference
     between a well-posed number and a useless one: by the far end this
     tail has turned most of the way round, so the turn onto the *base*
     segment is close to half a circle and the axis it is taken about is
     the cross product of two nearly opposite vectors — which is a
     direction made of rounding error. Neighbouring nodes then disagree
     about which way to go and the surface tears. From one segment to the
     next it is twenty degrees about a well-conditioned axis. */
  for (let i = 2; i < out.length; i++) {
    const a = out[i - 1].dir, b = out[i].dir;
    const ax = [a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0]];
    const sin = Math.hypot(ax[0], ax[1], ax[2]);
    const cos = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    out[i].turn = sin < 1e-9
      ? { axis: [1, 0, 0], angle: 0 }
      : { axis: ax.map((v) => v / sin), angle: Math.atan2(sin, cos) };
  }
  out[1].turn = { axis: [1, 0, 0], angle: 0 };

  /* And how far each segment may be turned before the tube folds — not
     as a promise, because it does not hold: see the note by TAIL_FLIP.
     A ring of radius r carried along a line that turns dθ over a length
     ds sweeps its inside edge back by r·dθ while the line goes forward
     by ds, so past r·dθ/ds = 1 the surface passes through itself. This
     tail is a fifth of a unit thick against segments of about the same,
     so the margin is thin everywhere and gone in the middle. */
  for (let i = 1; i < out.length; i++) {
    out[i].maxTurn = 0.9 * out[i].len / TAIL_RADIUS;
  }
  return out;
})();


/** Where the tail is taken to leave the body, as a share of its length.
    Nothing inside that moves: everything grows from here outward. */
const TAIL_PIN = 0.125;

/* ── what the water does to the shape of it ──
   Three things, applied along the chain rather than at the root, and the
   order they are described in is the order they are built:

   *Straightened.* The bake gives this tail a deep hook — four tenths of
   its own length off the chord — which is a cat's tail at rest on land
   and nothing like one in water. Taking it out is free here and only
   here: a frame chain can be told to undo the rest shape segment by
   segment, which nothing built out of two angles about a fixed origin
   can do.

   *Laid back.* From straight up to straight behind, so it lies along the
   surface the animal is floating in.

   That pair alone is the tail of a cat that has stopped: straight, back,
   and afloat. The third is what happens when it starts working.

   Both accumulate from the pin outward, so the base itself never turns —
   which is the difference between a tail and a stick being waggled. */
/* How far the curve is carried the other way, in units of the turn the
   bake put there. One takes the hook out and leaves a straight tail; two
   carries it the same distance the other way, which is the same curve
   reversed — and that is what the water does to this tail.

   It is applied to each segment's turn *off the one before it*, walked
   along the chain, so the shape is rebuilt rather than rotated. The
   first version reflected each segment about the *base* segment, which
   is ill-conditioned by the far end — the turn onto the base is nearly
   half a circle there and its axis is the cross product of two nearly
   opposite vectors.

   ── the surface still folds, and this is what is known ──
   Counted on the tail's own triangles, through the deformation the
   shader applies (tools/verify.mjs has the survey; on land it is clean):

     in water   353 of 3456 inside-out, in a band at three fifths along

   Three things have been ruled out by measurement rather than argument.
   It is not the conditioning above: rebuilding from each segment's own
   turn changed the count by two. It is not the segment count: sixteen
   segments instead of eight changed it by twelve. It is not the tip's
   cap: freezing the last stretch changed nothing.

   What it tracks is the turn a segment carries against its own length
   and the tail's thickness. At rest this tail already sits near that
   limit — it is a fat tube curled tightly — and every one of the water's
   terms adds to it, the flip most of all, since reversing a curve turns
   the rings through twice the shape's own curvature. Straightening alone
   (TAIL_FLIP = 1) folds more, not less, which says the pivot is part of
   it too: a vertex at the far end of a segment turns about a lever
   longer than the ring's radius.

   The fix is a change of deformation and not of any number here. */
const TAIL_FLIP = 2.0;
/* How far back it is laid, from straight up. A taste knob, and
   deliberately not aimed at the waterline: a floating cat's rump sits
   under the surface and the tail drifts up and down as it lies there, so
   "along the water" is not a height anything here can be held to. What
   this is for is how far back the tail reads, and it is turned by eye. */
const TAIL_LAY = -1.10;
/* And the float: a tail lying on water is never quite still. Two slow
   sines beating against each other so it does not repeat visibly — the
   same trick as the breeze the land tail has, and for the same reason,
   but slower and only up and down. Water moves a floating thing on its
   own schedule, and a tail that held one line while the surface under it
   did not would read as a prop. */
const TAIL_FLOAT = 0.55;

/* Gains from body motion to tail deflection. Bigger swings wider.

   Doubled when the bend became an arc. The old deformation threw the tip
   out by the whole radius times the sine of the angle; an arc integrates
   the sine along the way instead, which for a gently growing angle is
   about half as far. Same numbers on the way in, same swish on the way
   out. */
const TAIL_GY = 0.40;  // to yaw — the sideways swish when it turns
const TAIL_GP = 0.40;  // to pitch — the fore-and-aft float

/* Whiskers follow the head rather than the body, and barely move — they
   are stiff, short and light. Well under the tail's gain, or they read
   as antennae rather than as whiskers.

   Above the 0.05 the original used, though: that was authored for a
   locked close-up where the cat fills the frame, and this scene watches
   a much smaller animal from much further away. At 0.05 the sweep here
   is under half a degree of arc on screen — correct, and invisible. */
const WHISKER_GY = 0.12;
const WHISKER_GP = 0.12;

/* ── quaternions ──
   Four of them and a rotate, written out rather than pulled in: the tail
   is a chain of frames now and Euler angles cannot carry a twist along
   one. Same convention as everywhere else here — (x, y, z, w). */
const qMul = (a, b, out) => {
  const ax = a[0], ay = a[1], az = a[2], aw = a[3];
  const bx = b[0], by = b[1], bz = b[2], bw = b[3];
  out[0] = aw * bx + ax * bw + ay * bz - az * by;
  out[1] = aw * by - ax * bz + ay * bw + az * bx;
  out[2] = aw * bz + ax * by - ay * bx + az * bw;
  out[3] = aw * bw - ax * bx - ay * by - az * bz;
  return out;
};
/** A turn about one of the three axes: 0 = x, 1 = y, 2 = z. */
const qAxis = (axis, angle, out) => {
  out[0] = out[1] = out[2] = 0;
  out[axis] = Math.sin(angle * 0.5);
  out[3] = Math.cos(angle * 0.5);
  return out;
};
const qRot = (q, v, out) => {
  const x = q[0], y = q[1], z = q[2], w = q[3];
  const tx = 2 * (y * v[2] - z * v[1]);
  const ty = 2 * (z * v[0] - x * v[2]);
  const tz = 2 * (x * v[1] - y * v[0]);
  out[0] = v[0] + w * tx + y * tz - z * ty;
  out[1] = v[1] + w * ty + z * tx - x * tz;
  out[2] = v[2] + w * tz + x * ty - y * tx;
  return out;
};

class Chain {
  constructor() {
    this.a = new Float64Array(SEG + 1);
    this.v = new Float64Array(SEG + 1);
  }

  /** Advance one step, with the base pinned to whatever drives it. */
  step(drive, d) {
    const { a, v } = this;
    a[0] = drive;
    v[0] = 0;
    for (let i = 1; i <= SEG; i++) {
      const acc = CH_K * (a[i - 1] - a[i]) - CH_C * v[i];
      v[i] += acc * d;
      a[i] += v[i] * d;
    }
  }

  /** How far the point at `o` (0 base → 1 tip) trails the base. */
  lag(o) {
    const x = o * SEG;
    let i = Math.floor(x);
    if (i >= SEG) i = SEG - 1;
    const f = x - i;
    return this.a[i] * (1 - f) + this.a[i + 1] * f - this.a[0];
  }
}

/* ── breeze ──
   Independent of the lag: a tail is never quite still. Two slow sines
   beating against each other so it never repeats visibly, scaled by
   outerness so the base stays put and only the tip drifts. */
const windAz = (o, t) => (0.05 * Math.sin(t * 1.3 + o * 2.0) + 0.03 * Math.sin(t * 0.7 + 0.5)) * o;
const windAx = (o, t) => 0.035 * Math.sin(t * 1.0 + o * 1.6 + 1.0) * o;

/**
 * The tail's softness, as nine pairs of angles the vertex shader can
 * interpolate between.
 *
 * The original deformed the tail's vertices on the CPU every frame and
 * re-uploaded them. That is a lot of bandwidth to spend on a rotation
 * the GPU could do per-vertex — so what crosses the boundary here is the
 * *chain*, nine nodes of it, and the shader looks up its own vertex by
 * the outerness baked into the mesh. The breeze is sampled at the nodes
 * too; it is smooth enough along the tail that nine samples and a lerp
 * are indistinguishable from evaluating it per-vertex.
 */
export class Sway {
  constructor() {
    this.yaw = new Chain();
    this.pitch = new Chain();
    /** The head's own chains. Whiskers hang off the head, not the body,
        so they trail a turn of the neck the body never made. */
    this.headYaw = new Chain();
    this.headPitch = new Chain();
    /** Per node: how its ring is turned, and where it has been carried
        to from where the bake left it. Two halves of one arc — either one
        alone is a crease. */
    this.qs = new Float32Array((SEG + 1) * 4);
    this.bend = new Float32Array((SEG + 1) * 3);
    // Scratch for the walk along the chain, so a frame allocates nothing.
    this._q = new Float64Array(4);
    this._tmp = new Float64Array(4);
    this._acc = new Float64Array(4);
    this._carry = new Float64Array(4);
    this._v = new Float64Array(3);
    /** (ay, az) per node for the whiskers, for one side of the face. */
    this.whiskers = new Float32Array((SEG + 1) * 2);
    this.count = SEG + 1;
  }

  /**
   * @param {number} d      seconds, already clamped
   * @param {number} t      wall time, for the breeze
   * @param {number} yaw    the body's heading — in this scene that is the
   *                        cat's world facing, not a bone, because the
   *                        turn lives in the model matrix
   * @param {number} pitch  the body's pitch
   * @param {object} pose   the frame's pose: the head's own turn, and
   *                        both of the things the tail is doing
   */
  step(d, t, yaw, pitch, pose) {
    /* The swish drives the same chain as the body's turn, so a deliberate
       flick also arrives at the tip late rather than moving the whole
       tail as one rigid piece.

       The swim's circle is deliberately *not* in here, and that was the
       second thing tried rather than the first. Feeding both axes of it
       into the two chains is the obvious way to make a whirling tail
       curve, and it does not work: the chain is a spring, the stroke
       drives it at over a hertz with a radian of throw, and what comes
       back is not a lag but a resonance — measured at three times the
       drive, which leaves the curve seventy degrees off the path instead
       of trailing it. A trailing rope is a *kinematic* fact, not a
       dynamic one, so it is written as one below. */
    this.yaw.step(yaw + pose.tailSwish, d);
    this.pitch.step(pitch, d);

    // The head's absolute aim: where the body points, plus where the neck
    // is turned on top of it. Same multipliers `applyPose` uses, or the
    // whiskers would trail a head that is not the one being drawn.
    this.headYaw.step(yaw + pose.headYaw * 0.85 + pose.aimYaw * pose.aimWeight, d);
    this.headPitch.step(pitch + pose.headPitch * 0.8 + pose.aimPitch * pose.aimWeight, d);

    /* ── the tail, as a chain of frames ──
       Each node carries a rotation and the place its ring has been moved
       to, and the ring is set down there rather than swung about the
       bone's origin. That distinction is the whole of why this is not
       two angles any more: swinging a ring about a point a long way off
       drags it along an arc of that radius, and where the angle climbs
       faster than one radian per unit of radius the outer rings overtake
       the inner ones and the surface creases through itself. On this
       mesh that is unavoidable — the last three nodes sit at 2.23, 2.35
       and 2.40 from the origin, so there is almost no radius between
       them to spread an angle over, and the fold shows up even at the
       angles a walk uses.

       Positions here are integrated from tangents instead. Nothing is
       swung past anything, so nothing can fold, at any angle.

       What the water does, in the order it is built:
         flipped  the bake's deep hook is turned through itself and
                  comes out curving the other way
         laid     from straight up to straight behind, along the surface

       Both grow from the pin outward, so the tail leaves the rump
       pointing exactly where it always did. */
    const water = pose.tailWater;

    const q = this._q, tmp = this._tmp, acc = this._acc, v = this._v;
    // What the chain has turned through so far, before this node's own.
    const carry = this._carry;
    carry[0] = carry[1] = carry[2] = 0; carry[3] = 1;
    q[0] = q[1] = q[2] = 0; q[3] = 1;
    this.qs[0] = 0; this.qs[1] = 0; this.qs[2] = 0; this.qs[3] = 1;
    this.bend[0] = this.bend[1] = this.bend[2] = 0;
    this.whiskers[0] = this.headYaw.lag(0) * WHISKER_GY;
    this.whiskers[1] = this.headPitch.lag(0) * WHISKER_GP;

    let cx = 0, cy = 0, cz = 0;
    let laid = 0, lastGrow = 0;
    for (let i = 1; i <= SEG; i++) {
      const o = i / SEG;
      const seg = TAIL_SEG[i];

      const u = Math.max(0.0, (o - TAIL_PIN) / (1.0 - TAIL_PIN));
      const grow = u * u * (3.0 - 2.0 * u);

      /* Carried on from the segment before it: each one's own turn, run
         backwards. Written into the frame *before* whatever the previous
         node accumulated, because the axis it is taken about is a
         direction in the rest tail — the shape is being rebuilt out of
         its own turns rather than rotated as a whole. */
      if (water > 0.0) {
        const tn = seg.turn;
        /* Held to what a segment of this thickness can carry. It is not
           enough on its own — the pivot is the node, so a vertex out at
           the far end of a segment turns about a lever longer than the
           radius this limit is built from — but it is the half of the
           problem that has a closed form. */
        const want = -tn.angle * water * TAIL_FLIP;
        const a = Math.max(-seg.maxTurn, Math.min(seg.maxTurn, want)) * 0.5;
        const sn = Math.sin(a);
        tmp[0] = tn.axis[0] * sn; tmp[1] = tn.axis[1] * sn;
        tmp[2] = tn.axis[2] * sn; tmp[3] = Math.cos(a);
        qMul(carry, tmp, acc);
        carry[0] = acc[0]; carry[1] = acc[1]; carry[2] = acc[2]; carry[3] = acc[3];
      }
      q[0] = carry[0]; q[1] = carry[1]; q[2] = carry[2]; q[3] = carry[3];

      if (water > 0.0) {
        /* And laid back, from the pin outward — with the water lifting
           and dropping the end of it as it lies there. */
        const float = (Math.sin(t * 0.62) * 0.7 + Math.sin(t * 0.41 + 1.3) * 0.3)
                    * TAIL_FLOAT;
        /* The lay-back is spread over the whole tail rather than applied
           at one joint, so its own share of the turn per segment is small
           — but it is added to the flip's, and the pair of them have to
           clear the same limit. */
        const lay = (TAIL_LAY + float) * water;
        const step = lay * (grow - lastGrow);
        const room = seg.maxTurn - Math.abs(-seg.turn.angle * water * TAIL_FLIP);
        const held = Math.max(-Math.max(room, 0), Math.min(Math.max(room, 0), step));
        laid += held;
        qMul(qAxis(0, laid, tmp), q, acc);
        q[0] = acc[0]; q[1] = acc[1]; q[2] = acc[2]; q[3] = acc[3];
      }

      /* And the walk's own chain on top, outermost, so that on dry land
         this is exactly the swish it always was — with the difference
         that it is now integrated rather than swung, which is what takes
         the crease out of that too. */
      qMul(qAxis(2, this.yaw.lag(o) * TAIL_GY + windAz(o, t), tmp), q, acc);
      qMul(qAxis(0, this.pitch.lag(o) * TAIL_GP + windAx(o, t), tmp), acc, q);

      this.qs[i * 4] = q[0];
      this.qs[i * 4 + 1] = q[1];
      this.qs[i * 4 + 2] = q[2];
      this.qs[i * 4 + 3] = q[3];

      // Where that leaves this segment, and where its ring ends up.
      v[0] = seg.dir[0] * seg.len;
      v[1] = seg.dir[1] * seg.len;
      v[2] = seg.dir[2] * seg.len;
      qRot(q, v, acc);
      cx += acc[0] - v[0];
      cy += acc[1] - v[1];
      cz += acc[2] - v[2];
      this.bend[i * 3] = cx;
      this.bend[i * 3 + 1] = cy;
      this.bend[i * 3 + 2] = cz;

      // No breeze on these: a whisker is far too stiff for it, and the
      // drift that reads as life on a tail reads as a twitch on a face.
      this.whiskers[i * 2] = this.headYaw.lag(o) * WHISKER_GY;
      this.whiskers[i * 2 + 1] = this.headPitch.lag(o) * WHISKER_GP;
      lastGrow = grow;
    }
    return this.qs;
  }

  /** How much the chains are still doing, for the temporal filter. */
  get activity() {
    let m = 0;
    for (const c of [this.yaw, this.pitch, this.headYaw, this.headPitch]) {
      for (let i = 1; i <= SEG; i++) m += Math.abs(c.a[i] - c.a[0]) + Math.abs(c.v[i]);
    }
    return m;
  }
}

/* ── gait ──
   A cat walks diagonally: front-left with hind-right. One phase drives
   everything, with the two diagonals half a cycle apart. */
const STRIDE_HZ = 1.45;   // cycles per second per unit of speed
const SWING_HIND = 0.55;  // hind leg swing amplitude, radians
const SWING_KNEE = 0.34;  // knee follows the hip, lagging a quarter turn
const SWING_FRONT = 0.62; // front leg swing
const BOB_AMP = 0.055;    // vertical travel of the whole body per step

/* ── the paddle ──
   Swimming is the walk with three numbers moved, and that is not a
   shortcut — it is what a cat actually does. It keeps the same diagonal
   gait in the water; what changes is that the legs hang *under* the body
   instead of reaching out in front of it, that they travel a fraction as
   far, and that they never stop, because a cat that stops paddling
   sinks.

   The alternative was a second animation, which would have been a second
   description of how this animal moves — and the two would have had to
   agree at the waterline, where the blend between them happens. */
/** How far back the legs sit once they are under the animal. */
const PADDLE_BACK_HIND = 0.50;
const PADDLE_BACK_FRONT = 0.34;
/* What is left of the swing when it becomes a paddle — which by now is
   very nearly all of it. What separates the two is no longer the size of
   the travel: it is which legs go together, where they sit while they do
   it, and how fast. A short stroke read as an animal treading water very
   politely; this is one working. */
const PADDLE_SWING = 0.96;
/* The stroke a floating cat keeps up while going nowhere. Small, but
   never zero: stillness in deep water reads as a stuffed animal.

   One number for two things, and that is the point rather than a
   shortcut: the drive scales the swing *and* advances the phase, so a
   cat treading water paddles both shorter and slower than one going
   somewhere. Those are the same fact about an animal doing less work,
   and splitting them into two knobs would make it possible to set them
   against each other. */
const PADDLE_IDLE = 0.22;
/** Strokes per second per unit of drive. Faster than a walk and shorter,
    which is the difference between pushing on ground and on water. */
const PADDLE_HZ = 2.60;
export class Driver {
  constructor() {
    this.phase = 0;
    this.time = 0;
    this.blink = 0;
    this.nextBlink = 2.4;
    this.speed = 0;      // smoothed, so the gait does not snap on keydown
    this.turn = 0;
    /** How much of the animal is being carried by water rather than by
        its feet, smoothed for the same reason the speed is: a shelving
        bank hands this over gradually and the gait has to arrive with
        it. */
    this.swim = 0;

    this.pose = {
      headPitch: 0, headYaw: 0, headTilt: 0,
      earL: 0, earR: 0,
      tailYaw: 0, tailWater: 0, lean: 0, bob: 0,
      /* The tail's two jobs, kept apart because they are driven
         differently: the walk's sideways flick goes through the spring
         chain, the swim's lean does not — a chain driven at the stroke
         rate resonates rather than trails. */
      tailSwish: 0,
      bodyYaw: 0, bodyPitch: 0,
      eyeOpen: 1,
      // Named by diagonal, not by side: A is one hind leg plus the front
      // leg across from it, B is the other pair. Which physical legs
      // those are is settled in `buildCache`.
      hipA: 0, kneeA: 0, shoulderA: 0,
      hipB: 0, kneeB: 0, shoulderB: 0,
      /* Where the head is being aimed, over and above whatever the gait
         is doing with it, and how much of that to apply. Written from
         outside — the driver has no idea anything is being aimed at. */
      aimYaw: 0, aimPitch: 0, aimWeight: 0,
    };
  }

  /**
   * @param {number} dt      seconds
   * @param {number} speed   0…1, how hard it is being driven forward
   * @param {number} turn    -1…1, steering, for the lean into a corner
   * @param {number} [swim]  0…1, how much of it the water is carrying
   */
  step(dt, speed, turn, swim = 0) {
    const d = Math.min(0.05, Math.max(0, dt));
    this.time += d;

    // Ease into and out of the gait. Stepping straight from 0 to full
    // stride on a keypress reads as a glitch, not as a cat.
    this.speed += (speed - this.speed) * (1 - Math.exp(-d * 9));
    this.turn += (turn - this.turn) * (1 - Math.exp(-d * 6));
    this.swim += (swim - this.swim) * (1 - Math.exp(-d * 5));

    const p = this.pose;
    const s = this.speed;
    const w = this.swim;
    const t = this.time;

    /* What the legs are doing, which on the ground is what the animal is
       doing and in the water is not. A walk's stride advances with speed
       so the feet keep pace with the ground instead of scrubbing along
       it; there is no ground to scrub on out here, and a cat treading
       water is working hardest of all. */
    const drive = Math.max(s, w * PADDLE_IDLE);
    const advance = d * (STRIDE_HZ + (PADDLE_HZ - STRIDE_HZ) * w)
                  * Math.PI * 2 * Math.max(drive, 0.0001);
    this.phase += advance;
    if (this.phase > Math.PI * 2) this.phase -= Math.PI * 2;

    const a = this.phase;
    /* The same swing, a quarter of the size, about a rest angle that has
       moved back under the body. Both halves matter: the small travel on
       its own reads as a cat walking on the bottom, and the rest angle on
       its own reads as one frozen mid-stride. */
    const legs = drive * (1.0 - w * (1.0 - PADDLE_SWING));
    const backH = -PADDLE_BACK_HIND * w;
    const backF = -PADDLE_BACK_FRONT * w;

    /* Which legs go together — and it is not the same answer in the
       water.

       On the ground one diagonal swings while the other is planted, half
       a cycle apart: a hind leg and the front leg *across* from it move
       together. That is what a walk is, and pairing same-side legs
       instead gives a rocking horse. It is an arrangement for holding an
       animal up.

       Nothing is holding this one up. What a swimming cat does instead
       is paddle: the two sides go together and the two ends go opposite,
       front against hind, which is an arrangement for pulling itself
       along. The diagonal survives it in name only — A and B stop being
       diagonals and become the front pair and the hind pair.

       Crossfaded on the values rather than switched on the phase. Two of
       the four channels have to reverse between the arrangements, and
       mixing the values carries them through zero on the way — so the
       gait *reorganises* as the ground lets go, instead of flipping into
       a new one on whichever frame crossed the threshold. */
    const swingA = Math.sin(a), swingB = -swingA;
    const hipB = swingB + (swingA - swingB) * w;      // joins the other hip
    const frontA = swingA + (swingB - swingA) * w;    // and moves off to meet
    // ...the other front leg, which was already there: swingB is what a
    // paddling front leg does, so this one channel never moves at all.
    const frontB = swingB;

    p.hipA = backH + swingA * SWING_HIND * legs;
    p.hipB = backH + hipB * SWING_HIND * legs;
    p.shoulderA = backF + frontA * SWING_FRONT * legs;
    p.shoulderB = backF + frontB * SWING_FRONT * legs;

    // The knee trails the hip — the lower leg is still catching up when
    // the thigh has already reversed, which is what makes a walk read as
    // jointed rather than as a pendulum. It follows its own hip across
    // the change of arrangement, or the leg would fold the wrong way.
    const kneeA = Math.max(0, Math.sin(a - Math.PI / 2));
    const kneeB = Math.max(0, Math.sin(a + Math.PI / 2));
    p.kneeA = kneeA * SWING_KNEE * legs;
    p.kneeB = (kneeB + (kneeA - kneeB) * w) * SWING_KNEE * legs;

    /* Two bobs per stride — the body rises on each diagonal, not once per
       cycle. Idle breathing takes over as the gait fades out, and the
       water takes over from both: a floating cat does not bob to its own
       footfalls, it rides a surface that has its own slow period. */
    const step = Math.sin(a * 2) * BOB_AMP * s + Math.sin(t * 1.6) * 0.012 * (1 - s);
    p.bob = step * (1 - w) + Math.sin(t * 1.15) * 0.020 * w;
    /* And it swims nose-up. Negative is the lift here, the same sense the
       gait already uses to raise the head at speed — see setAim, where
       the sign is written down. */
    p.bodyPitch = (-s * 0.10 + Math.sin(a * 2 + 1.0) * 0.02 * s) * (1 - w)
                - 0.09 * w;

    // Lean into the turn, and let the tail counterweight it.
    p.lean = -this.turn * 0.16 * Math.max(s, 0.35);
    /* And in the water it goes round: sine on one axis, cosine on the
       other, which is a circle — swept clockwise seen from behind the
       animal, where behind is where the follow camera is.

       The walk's three terms fade out as it does. They are a swish, a
       drift and a lean into a corner, all of them side to side, and left
       running under the circle they flatten it into an egg and add a
       second beat at the stroke rate. The tail is doing one thing out
       here, not four. */
    const swish = Math.sin(t * 1.1) * 0.10 * (1 - s)     // idle drift
      + Math.sin(a + 0.6) * 0.20 * s                     // sway with the gait
      + this.turn * 0.28;                                // and swing wide on a corner

    p.tailSwish = swish * (1 - w);
    /* How much of the tail the water has. One number: it flips the bake's
       hook and lays the tail back along the surface, and there is nothing
       else the water does to it. */
    p.tailWater = w;
    // The bone carries the walk's flick and nothing else.
    p.tailYaw = p.tailSwish;

    // The head leads the turn and lifts a little at speed — and a lot
    // more in the water, which is the one thing a swimming cat is
    // unambiguously doing.
    p.headYaw = this.turn * 0.30;
    /* And it nods with the stroke — upward, always: the oscillation is
       taken off a sine lifted into 0..1 rather than one centred on zero,
       so the head never dips *below* where a swimming cat holds it. It
       is what it is doing with its neck to keep its nose clear. */
    p.headPitch = -s * 0.10 - (0.14 + 0.05 * (0.5 + 0.5 * Math.sin(a))) * w;
    p.headTilt = Math.sin(t * 0.7) * 0.03 * (1 - s);

    // Ears flick back as it picks up speed, and twitch at rest.
    const twitch = Math.max(0, Math.sin(t * 0.9) - 0.96) * 12;
    p.earL = -s * 0.22 + twitch * (1 - s);
    p.earR = -s * 0.22 - twitch * (1 - s) * 0.6;

    p.eyeOpen = this._blink(d);
    return p;
  }

  /** Closed for a tenth of a second, at irregular intervals. */
  _blink(d) {
    this.nextBlink -= d;
    if (this.nextBlink <= 0) {
      this.blink = 1;
      this.nextBlink = 1.8 + Math.random() * 4.0;
    }
    if (this.blink > 0) {
      this.blink -= d * 9;
      if (this.blink < 0) this.blink = 0;
      // Up and down again: a triangle, not a step.
      return Math.max(0.08, 1 - Math.sin(Math.min(1, this.blink) * Math.PI) * 1.6);
    }
    return 1;
  }
}

/* ═══ pose → bones ════════════════════════════════════════════════ */

/**
 * Resolve the bones once, and resolve the legs by *where they are*
 * rather than by what they are called.
 *
 * The upstream model names its two pairs from opposite ends —
 * `hindHips[s < 0 ? 'R' : 'L']` against `frontPaws[s < 0 ? 'L' : 'R']` —
 * so `hipHL` sits at x = +0.92 while `pawFL` sits at x = −0.35. Trusting
 * those names pairs each hind leg with the front leg on its *own* side,
 * and the cat walks like a rocking horse: both legs on one flank swing
 * forward together.
 *
 * A cat's walk is diagonal. The only thing that reliably says which
 * flank a leg is on is its rest position, so that is what decides here.
 * If the names are ever made consistent upstream, this keeps working.
 */
function buildCache(rig) {
  const xOf = (name) => rig.rest.position[rig.bone(name) * 3];
  const side = (a, b) => (xOf(a) > xOf(b) ? [a, b] : [b, a]);

  const [hindPlus, hindMinus] = side('hipHL', 'hipHR');
  const [frontPlus, frontMinus] = side('pawFL', 'pawFR');
  // The knee hangs off its own hip, so it follows whichever hip it is
  // parented to rather than being resolved separately.
  const kneeOf = { hipHL: 'pawHL', hipHR: 'pawHR' };

  return {
    root: rig.bone('root'), torso: rig.bone('torso'),
    bodyPivot: rig.bone('bodyPivot'), head: rig.bone('head'),
    earL: rig.bone('earL'), earR: rig.bone('earR'), tail: rig.bone('tail'),

    // Diagonal pairs: each hind leg swings with the front leg on the
    // *opposite* flank.
    hindA: rig.bone(hindPlus), kneeA: rig.bone(kneeOf[hindPlus]), frontA: rig.bone(frontMinus),
    hindB: rig.bone(hindMinus), kneeB: rig.bone(kneeOf[hindMinus]), frontB: rig.bone(frontPlus),

    eyes: rig.names.map((n, i) => (n.startsWith('eye') ? i : -1)).filter((i) => i >= 0),
  };
}

/**
 * Write a pose onto the rig. Every channel is absolute — set, never
 * accumulated — so a dropped frame cannot leave the cat bent.
 *
 * This is a port of `applyPose` from cat-model.js, with the scratching
 * and pupil behaviour left out (this skin has no two-layer eyes, and
 * nothing here scratches yet) and a walk cycle added in their place.
 */
export function applyPose(rig, p) {
  const B = rig._cache ??= buildCache(rig);

  /* The aim rides on top of the gait's own head motion rather than
     replacing it, so a cat tracking something still breathes and still
     leans into its corners. */
  rig.setRotation(B.head,
    HEAD_LEAN + p.headPitch * 0.8 + p.aimPitch * p.aimWeight,
    p.headYaw * 0.85 + p.aimYaw * p.aimWeight,
    p.headTilt);

  // The ears carry a rest roll that splays them outward; the pose adds
  // to it rather than replacing it, and mirrored so both flick the same
  // way in world space.
  rig.rotation[B.earL * 3 + 2] = rig.userData[B.earL].base + p.earL * 0.6;
  rig.rotation[B.earR * 3 + 2] = rig.userData[B.earR].base - p.earR * 0.6;

  /* The bone carries the walk's flick and nothing else: everything the
     water does to this tail is done along the chain, where it can be
     grown from the first node outward instead of turned at the root. */
  /* The bone itself only ever carries the walk's swish. The swim's whirl
     is not here at all — see the note by TAIL_FLIP: a tail that turns
     from its root swings the part of itself that is inside the animal,
     and what has to stay still is exactly that part. */
  rig.rotation[B.tail * 3 + 2] = TAIL_REST + p.tailYaw;
  rig.rotation[B.bodyPivot * 3 + 2] = p.lean;

  // Two diagonals, half a cycle apart. Hind legs swing at the hip and
  // again at the knee; front legs swing at the shoulder. All negative-
  // forward, matching how the joints were authored.
  rig.rotation[B.hindA * 3] = -p.hipA;
  rig.rotation[B.kneeA * 3] = -p.kneeA;
  rig.rotation[B.frontA * 3] = -p.shoulderA;
  rig.rotation[B.hindB * 3] = -p.hipB;
  rig.rotation[B.kneeB * 3] = -p.kneeB;
  rig.rotation[B.frontB * 3] = -p.shoulderB;

  // Blinking is a squash of the eyeball, which is how the model has
  // always done it — there are no lids to close.
  const open = Math.max(0.08, p.eyeOpen);
  for (const e of B.eyes) rig.scale[e * 3 + 1] = open;

  rig.position[B.root * 3 + 1] = rig.rest.position[B.root * 3 + 1] + p.bob;
  rig.rotation[B.root * 3 + 2] = p.headTilt * 0.04;
  rig.rotation[B.root * 3 + 1] = p.bodyYaw;
  rig.rotation[B.torso * 3] = p.bodyPitch;
}
