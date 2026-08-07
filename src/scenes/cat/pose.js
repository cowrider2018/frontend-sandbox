/* ── scenes/cat/pose.js ──────────────────────────────────────────────
   What the cat is doing, and how that reaches the bones.

   Two halves, deliberately separate:

     Driver     decides the pose — a walk cycle when it is moving, idle
                breathing and blinking when it is not
     applyPose  writes that pose onto the rig

   The split is the one the original model used, and it is worth keeping:
   `applyPose` is the contract with the skeleton, so a different driver
   (relpet's mood brain, say, or a path follower) can be swapped in later
   without touching a single bone name.
   ------------------------------------------------------------------ */

/* Rest offsets the model bakes in and the pose has to build on top of.
   Taken from cat-model.js, where they are the same constants. */
const HEAD_LEAN = 0.12;   // the head sits slightly forward at rest
const TAIL_REST = -0.15;  // tail's resting roll

/* ── gait ──
   A cat walks diagonally: front-left with hind-right. One phase drives
   everything, with the two diagonals half a cycle apart. */
const STRIDE_HZ = 1.45;   // cycles per second per unit of speed
const SWING_HIND = 0.55;  // hind leg swing amplitude, radians
const SWING_KNEE = 0.34;  // knee follows the hip, lagging a quarter turn
const SWING_FRONT = 0.62; // front leg swing
const BOB_AMP = 0.055;    // vertical travel of the whole body per step

export class Driver {
  constructor() {
    this.phase = 0;
    this.time = 0;
    this.blink = 0;
    this.nextBlink = 2.4;
    this.speed = 0;      // smoothed, so the gait does not snap on keydown
    this.turn = 0;

    this.pose = {
      headPitch: 0, headYaw: 0, headTilt: 0,
      earL: 0, earR: 0,
      tailYaw: 0, lean: 0, bob: 0,
      bodyYaw: 0, bodyPitch: 0,
      eyeOpen: 1,
      hipL: 0, hipR: 0, kneeL: 0, kneeR: 0, shoulderL: 0, shoulderR: 0,
    };
  }

  /**
   * @param {number} dt      seconds
   * @param {number} speed   0…1, how hard it is being driven forward
   * @param {number} turn    -1…1, steering, for the lean into a corner
   */
  step(dt, speed, turn) {
    const d = Math.min(0.05, Math.max(0, dt));
    this.time += d;

    // Ease into and out of the gait. Stepping straight from 0 to full
    // stride on a keypress reads as a glitch, not as a cat.
    this.speed += (speed - this.speed) * (1 - Math.exp(-d * 9));
    this.turn += (turn - this.turn) * (1 - Math.exp(-d * 6));

    // The stride advances with speed, so the feet keep pace with the
    // ground instead of scrubbing along it.
    this.phase += d * STRIDE_HZ * Math.PI * 2 * Math.max(this.speed, 0.0001);
    if (this.phase > Math.PI * 2) this.phase -= Math.PI * 2;

    const p = this.pose;
    const s = this.speed;
    const a = this.phase;
    const t = this.time;

    // Diagonal gait: left-hind pairs with right-front.
    const swing = Math.sin(a), counter = Math.sin(a + Math.PI);
    p.hipL = swing * SWING_HIND * s;
    p.hipR = counter * SWING_HIND * s;
    // The knee trails the hip — the lower leg is still catching up when
    // the thigh has already reversed, which is what makes a walk read as
    // jointed rather than as a pendulum.
    p.kneeL = Math.max(0, Math.sin(a - Math.PI / 2)) * SWING_KNEE * s;
    p.kneeR = Math.max(0, Math.sin(a + Math.PI / 2)) * SWING_KNEE * s;
    p.shoulderL = counter * SWING_FRONT * s;
    p.shoulderR = swing * SWING_FRONT * s;

    // Two bobs per stride — the body rises on each diagonal, not once
    // per cycle. Idle breathing takes over as the gait fades out.
    p.bob = Math.sin(a * 2) * BOB_AMP * s + Math.sin(t * 1.6) * 0.012 * (1 - s);
    p.bodyPitch = -s * 0.10 + Math.sin(a * 2 + 1.0) * 0.02 * s;

    // Lean into the turn, and let the tail counterweight it.
    p.lean = -this.turn * 0.16 * Math.max(s, 0.35);
    p.tailYaw = Math.sin(t * 1.1) * 0.10 * (1 - s)      // idle drift
      + Math.sin(a + 0.6) * 0.20 * s                     // sway with the gait
      + this.turn * 0.28;                                // and swing wide on a corner

    // The head leads the turn and lifts a little at speed.
    p.headYaw = this.turn * 0.30;
    p.headPitch = -s * 0.10;
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
 * Write a pose onto the rig. Every channel is absolute — set, never
 * accumulated — so a dropped frame cannot leave the cat bent.
 *
 * This is a port of `applyPose` from cat-model.js, with the scratching
 * and pupil behaviour left out (this skin has no two-layer eyes, and
 * nothing here scratches yet) and a walk cycle added in their place.
 */
export function applyPose(rig, p) {
  const B = rig._cache ??= {
    root: rig.bone('root'), torso: rig.bone('torso'),
    bodyPivot: rig.bone('bodyPivot'), head: rig.bone('head'),
    earL: rig.bone('earL'), earR: rig.bone('earR'), tail: rig.bone('tail'),
    hipHL: rig.bone('hipHL'), hipHR: rig.bone('hipHR'),
    pawHL: rig.bone('pawHL'), pawHR: rig.bone('pawHR'),
    pawFL: rig.bone('pawFL'), pawFR: rig.bone('pawFR'),
    eyes: rig.names.map((n, i) => (n.startsWith('eye') ? i : -1)).filter((i) => i >= 0),
  };

  rig.setRotation(B.head, HEAD_LEAN + p.headPitch * 0.8, p.headYaw * 0.85, p.headTilt);

  // The ears carry a rest roll that splays them outward; the pose adds
  // to it rather than replacing it, and mirrored so both flick the same
  // way in world space.
  rig.rotation[B.earL * 3 + 2] = rig.userData[B.earL].base + p.earL * 0.6;
  rig.rotation[B.earR * 3 + 2] = rig.userData[B.earR].base - p.earR * 0.6;

  rig.rotation[B.tail * 3 + 2] = TAIL_REST + p.tailYaw;
  rig.rotation[B.bodyPivot * 3 + 2] = p.lean;

  // Hind legs swing at the hip; the paw swings again at the knee. Both
  // are negative-forward, matching how the joints were authored.
  rig.rotation[B.hipHL * 3] = -p.hipL;
  rig.rotation[B.hipHR * 3] = -p.hipR;
  rig.rotation[B.pawHL * 3] = -p.kneeL;
  rig.rotation[B.pawHR * 3] = -p.kneeR;
  rig.rotation[B.pawFL * 3] = -p.shoulderL;
  rig.rotation[B.pawFR * 3] = -p.shoulderR;

  // Blinking is a squash of the eyeball, which is how the model has
  // always done it — there are no lids to close.
  const open = Math.max(0.08, p.eyeOpen);
  for (const e of B.eyes) rig.scale[e * 3 + 1] = open;

  rig.position[B.root * 3 + 1] = rig.rest.position[B.root * 3 + 1] + p.bob;
  rig.rotation[B.root * 3 + 2] = p.headTilt * 0.04;
  rig.rotation[B.root * 3 + 1] = p.bodyYaw;
  rig.rotation[B.torso * 3] = p.bodyPitch;
}
