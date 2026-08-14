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
const SEG = 8;
const CH_K = 240;
const CH_C = 15;

/** How far behind the base the tip is, in radians of the whirl's own
    phase. It is the whole of the curve while the tail is turning: at
    zero the tail is a straight rod being waved in a cone. */
const TAIL_TRAIL = 0.70;

/** Gains from body motion to tail deflection. Bigger swings wider. */
const TAIL_GY = 0.20;  // to yaw — the sideways swish when it turns
const TAIL_GP = 0.20;  // to pitch — the fore-and-aft float

/* Whiskers follow the head rather than the body, and barely move — they
   are stiff, short and light. Well under the tail's gain, or they read
   as antennae rather than as whiskers.

   Above the 0.05 the original used, though: that was authored for a
   locked close-up where the cat fills the frame, and this scene watches
   a much smaller animal from much further away. At 0.05 the sweep here
   is under half a degree of arc on screen — correct, and invisible. */
const WHISKER_GY = 0.12;
const WHISKER_GP = 0.12;

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
    /** (ax, az) per node for the tail, packed for a `vec2[]` uniform. */
    this.nodes = new Float32Array((SEG + 1) * 2);
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

    /* The whirl: each node points where the base was pointing a moment
       ago, and the moment grows along the tail. That is what a rope being
       swung round actually does, and it is why the curve lies *along* the
       circle — the difference between two nearby points on a circle is
       its tangent, so the deflection is tangential by construction and
       not by tuning. */
    const spin = pose.tailSpin;
    const ph = pose.tailPhase;

    for (let i = 0; i <= SEG; i++) {
      const o = i / SEG;
      let ax = this.pitch.lag(o) * TAIL_GP + windAx(o, t);
      let az = this.yaw.lag(o) * TAIL_GY + windAz(o, t);
      if (spin > 0.0) {
        const back = ph - TAIL_TRAIL * o;
        az += (Math.sin(back) - Math.sin(ph)) * spin;
        ax += (Math.cos(back) - Math.cos(ph)) * spin;
      }
      this.nodes[i * 2] = ax;
      this.nodes[i * 2 + 1] = az;

      // No breeze on these: a whisker is far too stiff for it, and the
      // drift that reads as life on a tail reads as a twitch on a face.
      this.whiskers[i * 2] = this.headYaw.lag(o) * WHISKER_GY;
      this.whiskers[i * 2 + 1] = this.headPitch.lag(o) * WHISKER_GP;
    }

    return this.nodes;
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
/** The stroke a floating cat keeps up while going nowhere. Small, but
    never zero: stillness in deep water reads as a stuffed animal. */
const PADDLE_IDLE = 0.45;
/** Strokes per second per unit of drive. Faster than a walk and shorter,
    which is the difference between pushing on ground and on water. */
const PADDLE_HZ = 2.60;
/* And the tail, which stops being a rudder and becomes a propeller: it
   sweeps a cone rather than swishing across, one turn for every two
   strokes. Two channels ninety degrees apart is the whole of what a
   circle is — the bone already had the axis for the second one, because
   the sway chain was already floating it fore-and-aft.

   Its own phase, and not half of the stroke's. The stroke phase is
   wrapped at a full turn to keep it in range, and half of a wrapped
   number jumps by half a turn every time it wraps — which would put a
   flick in the tail once per stroke pair, at whatever moment the clock
   happened to cross.

   A full sixty degrees of cone. This is a rotor and not a rudder, and
   the sweep is most of what says so. */
const PADDLE_TAIL = 1.00;

export class Driver {
  constructor() {
    this.phase = 0;
    /** The tail's own, at half the stroke rate — see PADDLE_TAIL. */
    this.tailPhase = 0;
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
      tailYaw: 0, tailPitch: 0, lean: 0, bob: 0,
      /* The tail's two jobs, kept apart because they are driven
         differently: the walk's sideways flick goes through the spring
         chain, the swim's whirl is trailed kinematically. */
      tailSwish: 0, tailSpin: 0, tailPhase: 0,
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
    this.tailPhase += advance * 0.5;
    if (this.tailPhase > Math.PI * 2) this.tailPhase -= Math.PI * 2;

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

    const tc = Math.cos(this.tailPhase), ts = Math.sin(this.tailPhase);
    p.tailSwish = swish * (1 - w);
    p.tailSpin = PADDLE_TAIL * w;
    p.tailPhase = this.tailPhase;
    p.tailYaw = p.tailSwish + ts * p.tailSpin;
    p.tailPitch = tc * p.tailSpin;

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

  rig.rotation[B.tail * 3 + 2] = TAIL_REST + p.tailYaw;
  // The second axis of the tail's circle, added to the rest pose rather
  // than replacing it: nothing wrote this channel before, so what is in
  // it is what the bake put there.
  rig.rotation[B.tail * 3] = rig.rest.rotation[B.tail * 3] + p.tailPitch;
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
