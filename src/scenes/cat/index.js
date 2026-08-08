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

   Light crosses the same boundary, in both directions. The cat fires a
   shadow ray at the sun through the cluster's own field, so the spheres
   darken it; and it hands the marcher a handful of capsules hung off its
   skeleton, so it darkens the floor. Neither side has to know what the
   other is made of.
   ------------------------------------------------------------------ */

import { Program } from '../../core/program.js';
import { PRECISION, CONSTANTS, ROTATE, SIMPLEX3 } from '../../shaders/common.js';
import {
  CLUSTER_UNIFORMS, CLUSTER_FIELD, CLUSTER_LAYERS, CLUSTER_SHADOW, SKY,
} from '../cluster.js';
import { parseCat, Rig, modelMatrix } from './rig.js';
import { Driver, Sway, applyPose } from './pose.js';

const NEAR = 0.05;
const FAR = 200.0;

/** Capsules in the shadow proxy. Seven are used; the slot count is fixed. */
export const CAT_CAPS = 8;

/**
 * The cat as the distance field sees it.
 *
 * Nothing marches the real mesh — forty thousand triangles is the wrong
 * shape of problem for a sphere tracer, and the marcher would have to
 * learn about bones to do it. What goes into the field instead is seven
 * capsules hung off the same skeleton: a body, a head, four legs and a
 * tail. They are wrong in every detail and right in outline, which is
 * all a shadow is.
 *
 * Deliberately *not* in the primary field. Rays that draw the picture
 * still see only the cluster, or you would get a blobby second cat
 * standing inside the real one. The proxy exists for the queries that
 * ask "is anything in the way" — shadows and occlusion — and those
 * cannot tell a capsule from a cat.
 */
export const CAT_PROXY_GLSL = /* glsl */`
#define CAT_CAPS ${CAT_CAPS}

uniform vec4 uCatCapA[CAT_CAPS];   // xyz = one end, w = radius
uniform vec4 uCatCapB[CAT_CAPS];   // xyz = the other end
uniform vec4 uCatBound;            // xyz = centre, w = radius
uniform float uCatCaps;            // how many slots are live, 0 = no cat

float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
  vec3 pa = p - a, ba = b - a;
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - r;
}

float catProxy(vec3 p) {
  if (uCatCaps < 0.5) return 1e9;
  float d = 1e9;
  for (int i = 0; i < CAT_CAPS; i++) {
    if (float(i) >= uCatCaps) break;
    d = min(d, sdCapsule(p, uCatCapA[i].xyz, uCatCapB[i].xyz, uCatCapA[i].w));
  }
  return d;
}

/**
 * How much of the sun a point loses to the cat — solved, not marched.
 *
 * A soft shadow is the closest the ray ever comes to the occluder,
 * divided by how far it had gone when it got there. Against a union of
 * capsules that closest approach has a closed form: the shortest
 * distance between a ray and a line segment. So there is no loop over t
 * at all, and the answer is exact rather than sampled.
 *
 * It had to become this. Marched, the loop landed inside softShadow,
 * which the compiler inlines once for the primary ray and again for the
 * reflection bounce, and the program stopped linking — reported as an
 * empty log, which is worth knowing. This version is a handful of dot
 * products per capsule and is strictly the better answer besides: a
 * sampled march can step straight past the thinnest part of a leg.
 */
float catShadow(vec3 ro, vec3 rd, float k) {
  if (uCatCaps < 0.5) return 1.0;

  float res = 1.0;
  for (int i = 0; i < CAT_CAPS; i++) {
    if (float(i) >= uCatCaps) break;

    vec3 a = uCatCapA[i].xyz;
    vec3 ba = uCatCapB[i].xyz - a;
    vec3 ao = ro - a;
    float bb = max(dot(ba, ba), 1e-6);
    float rb = dot(rd, ba);
    float ab = dot(ao, ba);
    float ar = dot(ao, rd);

    // Where along the ray the two lines come closest; rd is unit, so the
    // denominator is what is left of the segment across the ray.
    float den = bb - rb * rb;
    float t = abs(den) > 1e-5 ? (ab * rb - ar * bb) / den : -ar;

    // Clamp to the segment, then re-solve the ray for that point: the
    // nearest point on an infinite line is often off the end of a leg.
    float s = clamp((ab + max(t, 0.0) * rb) / bb, 0.0, 1.0);
    vec3 onSeg = a + ba * s;
    t = max(dot(onSeg - ro, rd), 0.0);

    float d = length(ro + rd * t - onSeg) - uCatCapA[i].w;
    if (t > 0.02) res = min(res, k * d / t);
  }
  return clamp(res, 0.0, 1.0);
}
`;

const VERT_CAT = /* glsl */`
${PRECISION}

in vec3 aPosition;
in vec4 aNormal;     // xyz normal, w = outerness along a soft part
in vec4 aColor;      // rgb sRGB, a = bone index | sway group << 5

uniform mat4 uBones[BONE_N];
uniform mat4 uModel;
uniform vec3 uCamPos, uRight, uUp, uFwd;
uniform float uFocal, uAspect;
uniform vec2 uJitter;

/** Deflection at each node: the tail's chain, and the whiskers' own. */
uniform vec2 uSway[SWAY_N];
uniform vec2 uWhisker[SWAY_N];

out vec3 vNormal;
out vec3 vColor;
out vec3 vWorld;

/**
 * How far this vertex trails the base, read out of the chain the CPU
 * uploaded. Outerness is baked per vertex — 0 all along the rigid parts
 * of the cat, rising to 1 at the tip of the tail — so this costs one
 * texture-free lookup and no branch anywhere else in the model.
 */
vec2 sampleChain(vec2 chain[SWAY_N], float o) {
  float x = o * float(SWAY_N - 1);
  float i = floor(x);
  int lo = int(i);
  int hi = min(lo + 1, SWAY_N - 1);
  return mix(chain[lo], chain[hi], x - i);
}

vec3 rotZ(vec3 p, float a) {
  float c = cos(a), s = sin(a);
  return vec3(p.x * c - p.y * s, p.x * s + p.y * c, p.z);
}
vec3 rotX(vec3 p, float a) {
  float c = cos(a), s = sin(a);
  return vec3(p.x, p.y * c - p.z * s, p.y * s + p.z * c);
}
vec3 rotY(vec3 p, float a) {
  float c = cos(a), s = sin(a);
  return vec3(p.x * c + p.z * s, p.y, -p.x * s + p.z * c);
}

/**
 * Bend a vertex by whichever chain owns it, in its own bone's space.
 *
 * The axes differ because the parts do. A tail is carried behind the
 * body and swings across it and floats fore-and-aft — Z then X. A
 * whisker sticks out sideways from a face, so the same head movements
 * sweep it back and tilt it — Z then Y. Both orders are the model's own.
 *
 * The two sides of the face take the same angles with the sign flipped:
 * mirrored in their local frames is what makes them trail the *same*
 * way in the world.
 */
vec3 swayRotate(vec3 p, float o, int group) {
  if (group == SWAY_TAIL) {
    vec2 a = sampleChain(uSway, o);
    return rotX(rotZ(p, a.y), a.x);
  }
  vec2 a = sampleChain(uWhisker, o);
  if (group == SWAY_WHISKER_L) a = -a;
  return rotY(rotZ(p, a.y), a.x);
}

void main() {
  /* The colour's alpha byte carries the bone in its low five bits and
     the sway group in its top three. Rounding rather than truncating
     matters: 8/255 does not survive the trip to float exactly, and
     floor() would land one bone short. */
  int packed = int(aColor.a * 255.0 + 0.5);
  int b = packed & 31;
  int group = packed >> 5;
  mat4 bone = uBones[b];

  vec3 local = aPosition;
  vec3 normal = aNormal.xyz;

  /* Soft parts bend in their own bone's space, around its origin. For
     the tail that is where it meets the body; the whiskers were each
     given a bone of their own at the cheek so the same thing is true of
     them. Everything else is group 0 and skips it. */
  float o = aNormal.w;
  if (group != SWAY_NONE) {
    local = swayRotate(local, o, group);
    normal = swayRotate(normal, o, group);
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
${CONSTANTS}
${ROTATE}
${SIMPLEX3}
${CLUSTER_UNIFORMS}
${CLUSTER_FIELD}
${CLUSTER_LAYERS}
${CLUSTER_SHADOW}
${SKY}

in vec3 vNormal;
in vec3 vColor;
in vec3 vWorld;
out vec4 outColor;

uniform vec3 uCamPos;
uniform float uUnlit, uShadowSoft, uFog;

/* Fur is not a polished surface, so it does not get a polished
   surface's highlight. What it has is a sheen: light catching the pile
   at grazing angles, broad and soft, brightest around the silhouette.

   Broad is the operative word. A Blinn lobe tight enough to read as a
   highlight raises a dot product to about the 28th power, and that is
   an error amplifier — it turns a small disagreement between
   neighbouring normals into a visible seam. Sheen is a low power of a
   grazing term and stays quiet over the same normals. */
const float FUR_SHEEN = 0.55;
const float FUR_SHEEN_POWER = 3.0;
/** How much sky the coat catches. Far less than the metal next door. */
const float FUR_SKY = 0.30;

/* The bake stored colours as sRGB bytes because that is the space they
   were authored in and eight bits go furthest there. Everything past
   this point is linear, like the rest of the pipeline. */
vec3 srgbToLinear(vec3 c) {
  return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}

void main() {
  vec3 albedo = srgbToLinear(vColor);
  vec3 col = albedo;

  vec3 toEye = uCamPos - vWorld;
  float dist = length(toEye);
  vec3 rd = -toEye / dist;

  if (uUnlit < 0.5) {
    /* The scene's own shading, term for term, rather than the model's
       three-step gradient map.

       The model arrives with a three-step gradient map, which suits a
       cat that is the whole picture. Here it is one object among
       several lit by one sun, and quantising only its diffuse made it
       the single thing in frame that did not respond smoothly to the
       light — which reads as a cut-out however good its shadow is. The
       outlines stay: they are ink, not shading, and they are what keeps
       the drawn look after the lighting stops being drawn. */
    vec3 n = normalize(vNormal);
    vec3 l = uLightDir;
    vec3 v = -rd;
    float ndl = max(dot(n, l), 0.0);

    /* Is the sun actually reaching this point? The cat marches the same
       layered field the marcher shades against, through the same
       function, so a sphere drifting between the light and the animal
       darkens it with the identical penumbra.

       A surface facing away from the light is unlit whatever the ray
       finds, so no ray is fired for it. Nor does the cat shadow itself:
       its capsules are a coarse fit that pokes through the real surface
       in places, and self-shadowing against them is all acne. */
    float sh = 1.0;
    if (ndl > 0.0 && uShadowSoft > 0.0) {
      sh = clusterShadow(vWorld + n * 0.01, l, mix(6.0, 26.0, uShadowSoft));
    }

    // Grazing sheen instead of a specular lobe, gated on the lit side so
    // it rims the coat rather than glowing all the way round it.
    float sheen = pow(1.0 - max(dot(n, v), 0.0), FUR_SHEEN_POWER) * ndl;
    float fresnel = pow(1.0 - max(dot(n, v), 0.0), 5.0);

    col = albedo * (uTint * 2.3 * ndl * sh + vec3(0.10, 0.12, 0.16));
    col += uTint * sheen * sh * FUR_SHEEN;
    col += sky(reflect(rd, n)) * (0.04 + fresnel * FUR_SKY);
  }

  /* The same distance fog the marcher applies, toward the same horizon.
     Without it the cat stays perfectly crisp while everything around it
     softens with depth, which reads as a sticker however well it is lit.
     Applied outside the branch on purpose: the outlines have to fade
     with the body they wrap, or the cat dissolves and keeps its edges. */
  col = mix(col, sky(rd), 1.0 - exp(-dist * uFog * 0.045));

  // Alpha is the scene's depth channel: distance travelled from the eye,
  // in world units, exactly as the march reports it.
  outColor = vec4(col, dist);
}
`;

/* ═══ locomotion ══════════════════════════════════════════════════ */

const ACCEL = 7.0;       // how hard it gets moving, units/s²
const DRAG = 6.0;        // and how hard it stops
const TOP_SPEED = 2.6;   // units/s at full scale
const TURN_RATE = 3.2;   // radians/s, steering into a corner
/** Turning to a heading you pointed at. Faster, because it is a reply. */
const TURN_RATE_COURSE = 6.0;
const STRAFE_SCALE = 0.85;  // sidling is slower than walking, as it should be

/**
 * Two ways to drive a cat.
 *
 *   camera  WASD names a direction *on screen* — W away from you, S
 *           toward you — and the cat turns to face it and walks. It only
 *           ever walks forward; there is no reversing and no sidling,
 *           because an animal turns round and then goes.
 *
 *   look    the mouse turns it, no button held, and WASD becomes the
 *           usual four-way relative to the body: W/S along the nose,
 *           A/D sidestepping. Here it *can* move in a direction it is
 *           not facing.
 *
 * The names say which frame WASD is expressed in, which is the whole of
 * the difference.
 */
export const CONTROL_MODES = ['camera', 'look'];

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
        // Must match the bake's numbering; it packs these into a byte.
        SWAY_NONE: 0,
        SWAY_TAIL: 1,
        SWAY_WHISKER_L: 3,
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

    /* The colour buffer is the only thing a skin owns, so it is sized
       once and refilled on a change rather than rebuilt. */
    this.colors = data.colors;
    this.skins = data.header.skins;
    this.skin = this.skins[0];

    const locColor = gl.getAttribLocation(this.program.program, 'aColor');
    this.vboColor = buf(gl.ARRAY_BUFFER, this.colors.get(this.skin), gl.DYNAMIC_DRAW);
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
    this.mode = 'camera';
    /** Radians of mouse-look banked since the last update. */
    this._lookYaw = 0;
    this._lookPitch = 0;
    /** Signed turn rate, −1…1, for the lean and the tail. Derived from
        the yaw that actually happened, so both modes feed it alike. */
    this.turnRate = 0;

    this.keys = { w: false, a: false, s: false, d: false };
    this._model = new Float32Array(16);
    this._jitter = new Float32Array(2);

    /* The shadow proxy, refilled every frame from the live skeleton.
       Two ends and a radius per capsule, plus one sphere that contains
       the lot so a shadow ray can miss the whole animal at once. */
    this.capA = new Float32Array(CAT_CAPS * 4);
    this.capB = new Float32Array(CAT_CAPS * 4);
    this.capBound = new Float32Array(4);
    this.capCount = 0;
  }

  /** Total ground speed, whichever direction it is going. */
  get speed() { return Math.hypot(this.velocity, this.strafeVel); }

  /**
   * Where an eye is, in the world, right now.
   *
   * Taken from the posed skeleton rather than from the cat's position
   * and a guess: the head turns, the body leans and the whole animal
   * bobs through its stride, and a beam that leaves from where the eye
   * used to be reads as a mis-aim rather than as motion.
   *
   * @param {number} which 0 or 1
   */
  eyeWorld(which, out) {
    const B = this._eyeBones ??= this.rig.names
      .map((n, i) => (n.startsWith('eye') ? i : -1))
      .filter((i) => i >= 0);
    // No eye bones is possible for a colourway without them; the muzzle
    // is a serviceable place to fire from.
    const bone = B[which] ?? this.rig.bone('head');

    const m = this.rig.matrices, M = this._model, o = bone * 16;
    // The eyeball's own centre, pushed out through its bone and then the
    // cat's placement.
    const bx = m[o + 12], by = m[o + 13], bz = m[o + 14];
    out[0] = M[0] * bx + M[4] * by + M[8] * bz + M[12];
    out[1] = M[1] * bx + M[5] * by + M[9] * bz + M[13];
    out[2] = M[2] * bx + M[6] * by + M[10] * bz + M[14];
    return out;
  }

  onKey(e, down) {
    const k = e.key.toLowerCase();
    if (k in this.keys) { this.keys[k] = down; return true; }
    return false;
  }

  releaseKeys() {
    this.keys.w = this.keys.a = this.keys.s = this.keys.d = false;
  }

  /**
   * Turn to face a direction on the ground, at once.
   *
   * Snapped rather than eased on purpose: this is used when the cat
   * whips round to fire, and the beam leaves in the same instant. Easing
   * the body would have the shot come from somewhere the animal is not
   * yet looking, which reads as a mis-aim rather than as weight.
   */
  faceTowards(dx, dz) {
    if (Math.abs(dx) < 1e-6 && Math.abs(dz) < 1e-6) return;
    this.yaw = Math.atan2(dx, dz);
    // Rebuild the placement now rather than next frame: callers turn the
    // cat in order to read where its eyes ended up, and a stale matrix
    // would hand them the eyes it had before it turned.
    modelMatrix(this._model, this.x, this.floorY + this.footOffset, this.z, this.yaw, this.scale);
  }

  /** Swap the colourway. One buffer upload; the geometry is shared. */
  setSkin(name) {
    if (name === this.skin || !this.colors.has(name)) return false;
    this.skin = name;
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vboColor);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.colors.get(name));
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    return true;
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
   * Turn toward the direction the keys are pointing at *on screen*, and
   * report how fast to walk along the nose.
   *
   * The keys name a direction in the camera's frame, which is flattened
   * to the ground first — a camera looking down at the cat still has to
   * mean "away from me" by W, and its unflattened forward is mostly
   * downward. The camera's own right vector is already horizontal, so it
   * is used as it comes: "right" then means the same thing here as it
   * does in the shader that drew the picture you are reacting to.
   *
   * Speed is scaled by how well the body already points where it is
   * going, with a floor. Without the scale the cat power-slides through
   * every reversal; without the floor it stops dead at every corner.
   *
   * @returns {number} 0…1, the forward throttle
   */
  _steerToCourse(kx, kz, d, camera) {
    if (!camera || (kx === 0 && kz === 0)) return 0;

    let cfx = camera.fwd[0], cfz = camera.fwd[2];
    const cl = Math.hypot(cfx, cfz);
    if (cl < 1e-4) return 0;    // camera straight down: no usable heading
    cfx /= cl; cfz /= cl;

    let dx = cfx * kz + camera.right[0] * kx;
    let dz = cfz * kz + camera.right[2] * kx;
    const dl = Math.hypot(dx, dz);
    if (dl < 1e-6) return 0;
    dx /= dl; dz /= dl;

    // Shortest way round, then clamped to what one frame is allowed to
    // turn — so a reversal is a pivot the eye can follow, not a snap.
    const want = Math.atan2(dx, dz);
    const delta = (want - this.yaw + Math.PI * 3) % (Math.PI * 2) - Math.PI;
    const step = TURN_RATE_COURSE * d;
    this.yaw += Math.max(-step, Math.min(step, delta));

    const align = Math.sin(this.yaw) * dx + Math.cos(this.yaw) * dz;
    return 0.35 + 0.65 * Math.max(0, align);
  }

  /**
   * Advance locomotion and the pose.
   *
   * Movement is always in the floor plane, and the cat only ever travels
   * along its own nose unless it is sidling. What changes between the
   * modes is where the heading comes from — the keys, or the mouse.
   */
  update(dt, floorY, camera) {
    const d = Math.min(0.05, Math.max(0, dt));
    this.floorY = floorY;

    const look = this.mode === 'look';
    const kx = (this.keys.d ? 1 : 0) - (this.keys.a ? 1 : 0);
    const kz = (this.keys.w ? 1 : 0) - (this.keys.s ? 1 : 0);

    const yawBefore = this.yaw;
    let forward = 0, strafe = 0;

    if (look) {
      // Turning right means *decreasing* yaw: the heading (sin, cos)
      // differentiates to (cos, −sin), which points along the cat's left.
      this.yaw -= this._lookYaw;
      this._lookYaw = 0;
      forward = kz;
      strafe = kx;
    } else {
      forward = this._steerToCourse(kx, kz, d, camera);
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
    this.sway.step(d, this.time, this.yaw, pose.bodyPitch, pose.tailYaw, pose);

    // Anything the scene's temporal filter must not hold on to: walking,
    // but also breathing, a flicking ear, a blink — which lasts about six
    // frames and would otherwise resolve as a smear — or a tail still
    // settling after the body has stopped.
    this.animating = this.speed !== 0 || this.turnRate !== 0
      || this.rig.changed || this.sway.activity > 2e-3;

    modelMatrix(this._model, this.x, floorY + this.footOffset, this.z, this.yaw, this.scale);
    this._fitProxy();
    return this;
  }

  /**
   * Hang the shadow capsules off the posed skeleton.
   *
   * Each end is a point in some bone's own space, pushed out through
   * that bone's world matrix and then the cat's — so the proxy walks,
   * turns and leans because the skeleton does, with nothing here having
   * to know what a gait is. The radii are in model units and pick up the
   * cat's scale on the way through.
   */
  _fitProxy() {
    const B = this._proxyBones ??= {
      body: this.rig.bone('bodyPivot'), head: this.rig.bone('head'),
      tail: this.rig.bone('tail'),
      hipL: this.rig.bone('hipHL'), hipR: this.rig.bone('hipHR'),
      pawL: this.rig.bone('pawHL'), pawR: this.rig.bone('pawHR'),
      frontL: this.rig.bone('pawFL'), frontR: this.rig.bone('pawFR'),
    };
    const m = this.rig.matrices, M = this._model;
    let n = 0;

    /** A point in a bone's space, in the world. */
    const put = (into, bone, lx, ly, lz, w) => {
      const o = bone * 16;
      const bx = m[o] * lx + m[o + 4] * ly + m[o + 8] * lz + m[o + 12];
      const by = m[o + 1] * lx + m[o + 5] * ly + m[o + 9] * lz + m[o + 13];
      const bz = m[o + 2] * lx + m[o + 6] * ly + m[o + 10] * lz + m[o + 14];
      const i = n * 4;
      into[i] = M[0] * bx + M[4] * by + M[8] * bz + M[12];
      into[i + 1] = M[1] * bx + M[5] * by + M[9] * bz + M[13];
      into[i + 2] = M[2] * bx + M[6] * by + M[10] * bz + M[14];
      into[i + 3] = w;
    };

    const capsule = (bone, a, b, radius) => {
      put(this.capA, bone, a[0], a[1], a[2], radius * this.scale);
      put(this.capB, bone, b[0], b[1], b[2], 0);
      n++;
    };

    // Torso, from the hips up to the base of the neck.
    capsule(B.body, [0, 0.1, -0.5], [0, 1.0, 0.5], 0.80);
    // Head. Its bone sits at the base of the neck, so the ball of the
    // skull is a radius up its own Y.
    capsule(B.head, [0, 0.5, 0.1], [0, 1.3, 0.3], 0.95);
    // Tail, along the bone's own Y — which is where the arc runs after
    // the bake turns it to sweep out behind.
    capsule(B.tail, [0, 0, 0], [0, 2.2, -0.6], 0.28);
    // Legs. The hind pair reaches hip to paw; the front pair hangs from
    // the shoulder, which is the only joint they have.
    capsule(B.hipL, [0, -0.2, 0], [0, -0.9, 0.2], 0.34);
    capsule(B.hipR, [0, -0.2, 0], [0, -0.9, 0.2], 0.34);
    capsule(B.frontL, [0, -0.1, 0.1], [0, -0.6, 0.3], 0.30);
    capsule(B.frontR, [0, -0.1, 0.1], [0, -0.6, 0.3], 0.30);

    this.capCount = n;

    /* One sphere around all of it. A shadow ray that misses this never
       looks at a capsule, which is what keeps the cat almost free for
       every pixel that is not near it. */
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < n; i++) {
      cx += this.capA[i * 4] + this.capB[i * 4];
      cy += this.capA[i * 4 + 1] + this.capB[i * 4 + 1];
      cz += this.capA[i * 4 + 2] + this.capB[i * 4 + 2];
    }
    cx /= n * 2; cy /= n * 2; cz /= n * 2;

    let r = 0;
    for (let i = 0; i < n; i++) {
      const rad = this.capA[i * 4 + 3];
      for (const arr of [this.capA, this.capB]) {
        const d = Math.hypot(arr[i * 4] - cx, arr[i * 4 + 1] - cy, arr[i * 4 + 2] - cz) + rad;
        if (d > r) r = d;
      }
    }

    this.capBound[0] = cx;
    this.capBound[1] = cy;
    this.capBound[2] = cz;
    this.capBound[3] = r;
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
    this.mode = 'camera';
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
  draw(camera, env, frame = 0) {
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
      uWhisker: this.sway.whiskers,

      // The scene's light, and the field it has to cast through.
      uLightDir: env.dir,
      uTint: env.tint,
      uFog: env.fog,
      uShadowSoft: env.shadowSoft,
      uShadowSteps: env.shadowSteps,
      uShadowNoise: env.shadowNoise,
      uTime: env.time,
      uBlend: env.blend,
      uBallPos: env.ballPos,
      uBalls: env.balls,
      uBound: env.bound,
      uRipples: env.ripples,
      uRippleTo: env.rippleTo,
      uRippleOn: env.rippleOn,
      uRippleAmp: env.rippleAmp,
      uRippleSpeed: env.rippleSpeed,
      uRippleFreq: env.rippleFreq,
      uRippleTight: 5.0,
      uErode: env.erode,
      uDisplace: env.displace,

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
