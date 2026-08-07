/* ── scenes/cat/index.js ─────────────────────────────────────────────
   A rasterised cat inside a raymarched scene.

   The two do not share a pipeline and never will — one walks rays
   through a distance field, the other pushes triangles — so the question
   is only where they meet. It is depth, and the scene was already
   publishing it: `march.js` writes the ray's travel distance into the
   colour target's alpha so the impact flares can hide behind geometry.

   The cat writes the same quantity, `length(world - camera)`, in the same
   units, into the alpha of its own target. Compositing is then one
   comparison per pixel, and neither side had to learn anything about the
   other. There is no depth buffer on the canvas and none is needed; the
   one attached here is private, and exists only so the cat's own
   triangles sort against each other.

   What is deliberately NOT here yet: the cat is lit by its own toon ramp
   rather than the scene's light integrator, it casts no shadow, and the
   spheres cannot see it. That is the next stage — this one is about
   getting a solid, correctly occluded cat standing on the floor.
   ------------------------------------------------------------------ */

import { Program } from '../../core/program.js';
import { PRECISION } from '../../shaders/common.js';
import { parseCat, Rig, modelMatrix } from './rig.js';
import { Driver, Sway, applyPose } from './pose.js';

const NEAR = 0.05;
const FAR = 200.0;

const VERT_CAT = /* glsl */`
${PRECISION}

in vec3 aPosition;
in vec4 aNormal;     // xyz normal, w = outerness along a soft part
in vec4 aColor;      // rgb sRGB, a = bone index / 255

uniform mat4 uBones[BONE_N];
uniform mat4 uModel;
uniform vec3 uCamPos, uRight, uUp, uFwd;
uniform float uFocal, uAspect;
uniform vec2 uJitter;

/** (pitch, yaw) deflection at each node of the tail's chain. */
uniform vec2 uSway[SWAY_N];

out vec3 vNormal;
out vec3 vColor;
out vec3 vWorld;

/**
 * How far this vertex trails the base, read out of the chain the CPU
 * uploaded. Outerness is baked per vertex — 0 all along the rigid parts
 * of the cat, rising to 1 at the tip of the tail — so this costs one
 * texture-free lookup and no branch anywhere else in the model.
 */
vec2 swayAt(float o) {
  float x = o * float(SWAY_N - 1);
  float i = floor(x);
  int lo = int(i);
  int hi = min(lo + 1, SWAY_N - 1);
  return mix(uSway[lo], uSway[hi], x - i);
}

/** Z then X, the order the model's own deformation used. */
vec3 swayRotate(vec3 p, vec2 a) {
  float c = cos(a.y), s = sin(a.y);
  p = vec3(p.x * c - p.y * s, p.x * s + p.y * c, p.z);
  c = cos(a.x); s = sin(a.x);
  return vec3(p.x, p.y * c - p.z * s, p.y * s + p.z * c);
}

void main() {
  // The bone index rides in the colour's alpha byte. Rounding rather
  // than truncating matters: 8/255 does not survive the trip to float
  // exactly, and floor() would land one bone short.
  int b = int(aColor.a * 255.0 + 0.5);
  mat4 bone = uBones[b];

  vec3 local = aPosition;
  vec3 normal = aNormal.xyz;

  // Soft parts bend in their own bone's space, around its origin —
  // which for the tail is where it meets the body. Everything else has
  // an outerness of zero and skips it.
  float o = aNormal.w;
  if (o > 0.0) {
    vec2 a = swayAt(o);
    local = swayRotate(local, a);
    normal = swayRotate(normal, a);
  }

  vec4 world = uModel * (bone * vec4(local, 1.0));
  vWorld = world.xyz;
  vColor = aColor.rgb;
  vNormal = mat3(uModel) * (mat3(bone) * normal);

  // The camera basis is the march's, used the march's way: a ray for
  // NDC (x,y) has view-space direction (x·aspect/focal, y/focal, 1), so
  // inverting that is the projection. Deriving it from the same three
  // vectors instead of building a second view matrix is what guarantees
  // the two images line up — there is no second copy to drift.
  vec3 rel = world.xyz - uCamPos;
  vec3 view = vec3(dot(rel, uRight), dot(rel, uUp), dot(rel, uFwd));

  float z = view.z * (FAR + NEAR) / (FAR - NEAR) - 2.0 * FAR * NEAR / (FAR - NEAR);
  gl_Position = vec4(
    view.x * uFocal / uAspect + uJitter.x * view.z,
    view.y * uFocal + uJitter.y * view.z,
    z, view.z);
}
`;

const FRAG_CAT = /* glsl */`
${PRECISION}

in vec3 vNormal;
in vec3 vColor;
in vec3 vWorld;
out vec4 outColor;

uniform vec3 uCamPos, uLightDir, uTint;
uniform float uUnlit;

/* The bake stored colours as sRGB bytes because that is the space they
   were authored in and eight bits go furthest there. Everything past
   this point is linear, like the rest of the pipeline. */
vec3 srgbToLinear(vec3 c) {
  return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}

void main() {
  vec3 albedo = srgbToLinear(vColor);
  vec3 col = albedo;

  if (uUnlit < 0.5) {
    vec3 n = normalize(vNormal);
    float ndl = max(dot(n, uLightDir), 0.0);

    // The model's three-step gradient map, inlined: 176 / 220 / 255 out
    // of 255, sampled with a nearest filter. Quantising the diffuse term
    // like this is the entire toon look — everything else is ordinary.
    float ramp = ndl < 0.3333 ? 0.690 : ndl < 0.6667 ? 0.863 : 1.0;

    // Floor plus key, normalised so a fully-lit surface lands near 1.0.
    // The scene tonemaps afterwards, and a toon ramp that runs hot just
    // gets crushed flat by the shoulder of the curve.
    col = albedo * (0.42 + 0.58 * ramp) * mix(vec3(1.0), uTint, 0.22);
  }

  // Alpha is the scene's depth channel: distance travelled from the eye,
  // in world units, exactly as the march reports it.
  outColor = vec4(col, length(vWorld - uCamPos));
}
`;

/* ═══ locomotion ══════════════════════════════════════════════════ */

const ACCEL = 7.0;       // how hard it gets moving, units/s²
const DRAG = 6.0;        // and how hard it stops
const TOP_SPEED = 2.6;   // units/s at full scale
const TURN_RATE = 3.2;   // radians/s
const STRAFE_SCALE = 0.85;  // sidling is slower than walking, as it should be

/**
 * Two ways to drive a cat.
 *
 *   turn   A and D steer the body, W and S drive it along its nose.
 *          Everything is relative to the animal; the mouse is free for
 *          orbiting and for striking the spheres.
 *
 *   look   the mouse turns it — no button held — and WASD becomes the
 *          usual four-way: W/S along the nose, A/D sidestepping. The
 *          body can now move in a direction it is not facing.
 */
export const CONTROL_MODES = ['turn', 'look'];

export class Cat {
  /** Fetch and upload. Returns null if the asset is missing, never throws. */
  static async load(gl, url) {
    const res = await fetch(url, { cache: 'force-cache' });
    if (!res.ok) throw new Error(`cat.bin: ${res.status}`);
    return new Cat(gl, parseCat(await res.arrayBuffer()));
  }

  constructor(gl, data) {
    this.gl = gl;
    this.header = data.header;
    this.rig = new Rig(data.header);
    this.driver = new Driver();
    this.sway = new Sway();
    this.time = 0;

    this.program = new Program(gl, VERT_CAT, FRAG_CAT, {
      name: 'cat/mesh',
      defines: {
        BONE_N: this.rig.count,
        SWAY_N: this.sway.count,
        NEAR: NEAR.toFixed(4),
        FAR: FAR.toFixed(1),
      },
    });

    /* ── buffers ── */
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    const buf = (target, src, usage = gl.STATIC_DRAW) => {
      const b = gl.createBuffer();
      gl.bindBuffer(target, b);
      gl.bufferData(target, src, usage);
      return b;
    };

    // Slot 0 is bound to aPosition by Program, by convention.
    this.vboPosition = buf(gl.ARRAY_BUFFER, data.position);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

    const locNormal = gl.getAttribLocation(this.program.program, 'aNormal');
    this.vboNormal = buf(gl.ARRAY_BUFFER, data.normal);
    gl.enableVertexAttribArray(locNormal);
    gl.vertexAttribPointer(locNormal, 4, gl.SHORT, true, 0, 0);

    const locColor = gl.getAttribLocation(this.program.program, 'aColor');
    this.vboColor = buf(gl.ARRAY_BUFFER, data.color);
    gl.enableVertexAttribArray(locColor);
    // Not normalised for .a alone is not an option, so the bone index is
    // un-normalised in the shader instead.
    gl.vertexAttribPointer(locColor, 4, gl.UNSIGNED_BYTE, true, 0, 0);

    this.ibo = buf(gl.ELEMENT_ARRAY_BUFFER, data.index);
    gl.bindVertexArray(null);

    this.vao = vao;
    this.indexType = data.header.indexBits === 32 ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT;
    this.indexBytes = data.header.indexBits === 32 ? 4 : 2;
    this.groups = data.header.groups;
    this.triangles = data.header.indexCount / 3;

    /* ── placement ──
       Scaled so the cat reads as an animal beside a two-unit cluster
       rather than as another piece of scenery, and lifted so the lowest
       vertex in the rest pose sits exactly on the floor plane. */
    this.scale = 0.34;
    this.footOffset = -this.header.bounds.min[1] * this.scale;

    this.x = 1.6;
    this.z = 1.6;
    this.yaw = -0.6;
    /** Signed speed along the nose. Kept as the scalar it always was —
        the camera, the gait and the readout all still mean this by it. */
    this.velocity = 0;
    /** Signed speed along the cat's own right. Only `look` mode has it. */
    this.strafeVel = 0;
    this.floorY = 0;
    this.animating = true;
    this.mode = 'turn';
    /** Radians of mouse-look banked since the last update. */
    this._lookYaw = 0;
    this._lookPitch = 0;
    /** Signed turn rate, −1…1, for the lean and the tail. Derived from
        the yaw that actually happened, so both modes feed it alike. */
    this.turnRate = 0;

    this.keys = { w: false, a: false, s: false, d: false };
    this._model = new Float32Array(16);
    this._jitter = new Float32Array(2);
  }

  /** Total ground speed, whichever direction it is going. */
  get speed() { return Math.hypot(this.velocity, this.strafeVel); }

  onKey(e, down) {
    const k = e.key.toLowerCase();
    if (k in this.keys) { this.keys[k] = down; return true; }
    return false;
  }

  releaseKeys() {
    this.keys.w = this.keys.a = this.keys.s = this.keys.d = false;
  }

  setMode(mode) {
    if (!CONTROL_MODES.includes(mode) || mode === this.mode) return false;
    this.mode = mode;
    // Sideways momentum has nowhere to go once A and D go back to
    // steering, and leaving it in would slide the cat for half a second
    // after the switch.
    this.strafeVel = 0;
    this._lookYaw = this._lookPitch = 0;
    return true;
  }

  /**
   * Bank mouse-look, in radians, to be spent on the next update.
   *
   * Accumulated rather than applied here because the mouse fires far
   * more often than the frame does — a 1000 Hz mouse would otherwise
   * turn the cat sixteen times between two pictures of it.
   */
  look(dYaw, dPitch) {
    this._lookYaw += dYaw;
    this._lookPitch += dPitch;
  }

  /** Take the banked pitch, in radians. The camera owns pitch, not the cat. */
  takeLookPitch() {
    const p = this._lookPitch;
    this._lookPitch = 0;
    return p;
  }

  /**
   * Advance locomotion and the pose.
   *
   * Movement is always in the floor plane and always relative to the
   * cat, never to the camera — what changes between the two modes is
   * only what A and D mean, and where the heading comes from.
   */
  update(dt, floorY) {
    const d = Math.min(0.05, Math.max(0, dt));
    this.floorY = floorY;

    const forward = (this.keys.w ? 1 : 0) - (this.keys.s ? 1 : 0);
    const ad = (this.keys.a ? 1 : 0) - (this.keys.d ? 1 : 0);
    const look = this.mode === 'look';
    const steer = look ? 0 : ad;
    const strafe = look ? -ad : 0;   // A is left, and left is −right

    const yawBefore = this.yaw;

    if (look) {
      // Turning right means *decreasing* yaw: the heading (sin, cos)
      // differentiates to (cos, −sin), which points along the cat's left.
      this.yaw -= this._lookYaw;
      this._lookYaw = 0;
    } else {
      // Turning is scaled by how fast it is going, with a floor so the
      // cat can still pivot on the spot — a body that spins at full rate
      // while stationary looks like a turret.
      this.yaw += steer * TURN_RATE * d
        * (0.35 + 0.65 * Math.min(1, Math.abs(this.velocity) / TOP_SPEED));
    }

    const ease = (v, want, input) => v + (want - v) * (1 - Math.exp(-d * (input ? ACCEL : DRAG)));
    this.velocity = ease(this.velocity, forward * TOP_SPEED, forward);
    this.strafeVel = ease(this.strafeVel, strafe * TOP_SPEED * STRAFE_SCALE, strafe);
    if (Math.abs(this.velocity) < 1e-4) this.velocity = 0;
    if (Math.abs(this.strafeVel) < 1e-4) this.strafeVel = 0;

    /* Nose and right-hand side. `right` is cross(facing, up), which for
       a right-handed frame with Y up puts the cat's right at (−cos, sin)
       — the same basis the camera builds, so "right" means one thing
       everywhere in this scene. */
    const fx = Math.sin(this.yaw), fz = Math.cos(this.yaw);
    const rx = -fz, rz = fx;

    this.x += (fx * this.velocity + rx * this.strafeVel) * d;
    this.z += (fz * this.velocity + rz * this.strafeVel) * d;

    // One turn signal for both modes, taken from the yaw that actually
    // happened rather than from whichever input caused it. The lean and
    // the tail then behave the same however the cat is being driven.
    const turned = d > 0 ? (this.yaw - yawBefore) / (TURN_RATE * d) : 0;
    this.turnRate = Math.max(-1, Math.min(1, turned));

    const speed = Math.min(1, this.speed / TOP_SPEED);
    const pose = this.driver.step(d, speed, this.turnRate);
    applyPose(this.rig, pose);
    this.rig.update();

    /* The tail trails the *world* heading, not a bone. In the model this
       came from, the cat only ever turned by rotating its root; here the
       turn lives in the model matrix, so that is what the chain has to
       be driven by — otherwise the tail would hang dead through every
       corner, which is exactly when a real one swings widest. */
    this.time += d;
    this.sway.step(d, this.time, this.yaw, pose.bodyPitch, pose.tailYaw);

    // Anything the scene's temporal filter must not hold on to: walking,
    // but also breathing, a flicking ear, a blink — which lasts about six
    // frames and would otherwise resolve as a smear — or a tail still
    // settling after the body has stopped.
    this.animating = this.speed !== 0 || this.turnRate !== 0
      || this.rig.changed || this.sway.activity > 2e-3;

    modelMatrix(this._model, this.x, floorY + this.footOffset, this.z, this.yaw, this.scale);
    return this;
  }

  /** Put it back where it started, standing still. */
  reset() {
    this.x = 1.6;
    this.z = 1.6;
    this.yaw = -0.6;
    this.velocity = 0;
    this.strafeVel = 0;
    this.turnRate = 0;
    this._lookYaw = this._lookPitch = 0;
    this.mode = 'turn';
    this.releaseKeys();
    this.rig.reset();
    this.driver = new Driver();
    this.sway = new Sway();
    this.time = 0;
  }

  /**
   * Draw into the currently bound target.
   *
   * The target must have a depth attachment and must be cleared with an
   * alpha the scene reads as "nothing here" — 1e4 is what the march
   * writes for sky, and it is well inside half-float range.
   */
  draw(camera, light, frame = 0) {
    const gl = this.gl;

    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
    gl.depthMask(true);
    gl.enable(gl.CULL_FACE);

    // A subpixel offset per frame, so the scene's temporal accumulation
    // resolves the cat's edges the same way it resolves the march's
    // jittered rays. Without it the cat is the one hard-aliased thing in
    // an otherwise anti-aliased image.
    //
    // Two incommensurable frequencies rather than a random pair: the
    // offsets never repeat and never clump, which is what a temporal
    // filter needs to converge instead of settling into a pattern.
    this._jitter[0] = Math.sin(frame * 2.39996) / camera.width;
    this._jitter[1] = Math.sin(frame * 4.10000 + 1.7) / camera.height;

    this.program.use({
      uBones: this.rig.matrices,
      uModel: this._model,
      uCamPos: camera.pos,
      uRight: camera.right,
      uUp: camera.up,
      uFwd: camera.fwd,
      uFocal: camera.focal,
      uAspect: camera.aspect,
      uJitter: this._jitter,
      uSway: this.sway.nodes,
      uLightDir: light.dir,
      uTint: light.tint,
      uUnlit: 0,
    });

    gl.bindVertexArray(this.vao);

    for (const g of this.groups) {
      if (!g.count) continue;

      // The outlines are inverted hulls: the same shells, wound the other
      // way, so what shows is their far side peeking past the silhouette.
      const outline = g.name === 'outline';
      gl.cullFace(outline ? gl.FRONT : gl.BACK);
      this.program.use({ uUnlit: g.name === 'lit' ? 0 : 1 });

      gl.drawElements(gl.TRIANGLES, g.count, this.indexType, g.start * this.indexBytes);
    }

    gl.bindVertexArray(null);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
  }

  dispose() {
    const gl = this.gl;
    this.program.dispose();
    gl.deleteVertexArray(this.vao);
    gl.deleteBuffer(this.vboPosition);
    gl.deleteBuffer(this.vboNormal);
    gl.deleteBuffer(this.vboColor);
    gl.deleteBuffer(this.ibo);
  }
}
