/* ── scenes/march.js ─────────────────────────────────────────────────
   03 · Signed distance fields — a whole 3D scene in one fragment shader.

   There is no geometry here. Not one vertex, not one triangle: the
   entire image is produced by a single fullscreen triangle whose
   fragment shader walks a ray forward through an implicit surface
   defined by distance functions. Shadows come from marching toward the
   light, ambient occlusion from sampling the field around a point, and
   reflections from marching a second time along the mirror direction.

   Clicking dents the surface where you touched it and sends a ring
   travelling outward across it. That needs one thing the shader cannot
   provide: an answer, in JavaScript, to "what is under the cursor". So
   the ray is marched a second time on the CPU, against the same
   primitives, and the sphere positions are computed in JS and uploaded
   rather than derived from uTime in the shader — one source of truth
   for a scene that now has two readers.
   ------------------------------------------------------------------ */

import { Program } from '../core/program.js';
import { Target, DoubleTarget, bindScreen, BLEND } from '../core/gl.js';
import { PRECISION, CONSTANTS, HASH, COLOR, ROTATE, SIMPLEX3, VERT_FULLSCREEN } from '../shaders/common.js';

const BALL_N = 9;      // spheres in the cluster
const RIPPLE_N = 4;    // concurrent surface rings

const RING_MAJOR = 1.28;
const RING_MINOR = 0.075;

const FRAG_MARCH = /* glsl */`
${PRECISION}
${CONSTANTS}
${HASH}
${COLOR}
${ROTATE}
${SIMPLEX3}

#define BALL_N ${BALL_N}
#define RIPPLE_N ${RIPPLE_N}

in vec2 vUv;
out vec4 outColor;

uniform vec3  uCamPos, uRight, uUp, uFwd;
uniform vec2  uResolution;
uniform float uFocal, uTime;
uniform int   uSteps;
uniform float uBlend, uDisplace, uRough, uFloorMix;
uniform vec3  uLightDir, uTint;
uniform float uReflect, uFog, uAO, uShadowSoft;

uniform vec4  uBallPos[BALL_N];   // xyz = centre, w = radius
uniform float uBalls;

uniform vec4  uRipples[RIPPLE_N]; // xyz = impact point, w = normalised age
uniform float uRippleOn, uRippleAmp, uRippleSpeed, uRippleFreq, uRippleTight, uRippleGlow;

/* ═══ distance functions ══════════════════════════════════════════ */

float sdSphere(vec3 p, float r) { return length(p) - r; }

float sdTorus(vec3 p, vec2 t) {
  vec2 q = vec2(length(p.xz) - t.x, p.y);
  return length(q) - t.y;
}

/** Polynomial smooth minimum — the operator that makes SDFs feel alive. */
float smin(float a, float b, float k) {
  float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}

/**
 * Expanding rings from each recorded impact.
 *
 * Displacing a distance field is not the same as displacing a mesh:
 * there is no surface to move, so the ripple is authored as a term
 * added to the distance itself. Keep the amplitude small — a large one
 * breaks the field's Lipschitz bound and the sphere tracer starts
 * overshooting straight through the surface.
 */
float ripples(vec3 p) {
  float sum = 0.0;
  for (int i = 0; i < RIPPLE_N; i++) {
    float age = uRipples[i].w;
    if (age <= 0.0) continue;
    float d = length(p - uRipples[i].xyz);
    float front = age * uRippleSpeed;
    // Tight in space around the travelling front, fading in time.
    float env = exp(-abs(d - front) * uRippleTight) * (1.0 - age) * (1.0 - age);
    sum += sin((d - front) * uRippleFreq) * env;
  }
  return sum * uRippleAmp;
}

/* ═══ the scene ═══════════════════════════════════════════════════ */

// x = distance, y = material id
vec2 map(vec3 p) {
  float t = uTime * 0.42;

  // A cluster of orbiting spheres, welded together by smin. Each one
  // takes a different, mutually irrational orbit so the cluster never
  // repeats a pose. Positions arrive as uniforms so the CPU and the GPU
  // agree on where they are, exactly.
  float d = 1e9;
  for (int i = 0; i < BALL_N; i++) {
    if (float(i) >= uBalls) break;
    d = smin(d, sdSphere(p - uBallPos[i].xyz, uBallPos[i].w), uBlend);
  }

  // A ring threading the cluster, on its own slow tumble.
  vec3 q = p;
  q.yz = rot2(t * 0.31) * q.yz;
  q.xz = rot2(t * 0.19) * q.xz;
  d = smin(d, sdTorus(q, vec2(${RING_MAJOR}, ${RING_MINOR})), uBlend * 0.6);

  // Surface displacement: perturb the distance, not the geometry.
  if (uDisplace > 0.0) {
    d += uDisplace * 0.12 * snoise(p * 3.1 + vec3(0.0, 0.0, uTime * 0.3));
  }
  if (uRippleOn > 0.5) d += ripples(p);

  vec2 res = vec2(d, 1.0);

  float floorD = p.y + 1.35;
  if (floorD < res.x) res = vec2(floorD, 2.0);

  return res;
}

vec3 calcNormal(vec3 p) {
  // Tetrahedral sampling: four taps instead of six, and no bias.
  const vec2 k = vec2(1.0, -1.0);
  const float h = 0.0012;
  return normalize(
    k.xyy * map(p + k.xyy * h).x +
    k.yyx * map(p + k.yyx * h).x +
    k.yxy * map(p + k.yxy * h).x +
    k.xxx * map(p + k.xxx * h).x
  );
}

vec2 march(vec3 ro, vec3 rd, int steps, float maxDist) {
  float t = 0.04;
  float mat = 0.0;
  for (int i = 0; i < 256; i++) {
    if (i >= steps) break;
    vec3 p = ro + rd * t;
    vec2 h = map(p);
    // Relax the hit threshold with distance: far pixels are subpixel
    // anyway, and it buys back a lot of steps.
    if (h.x < 0.0008 * t + 0.0006) { mat = h.y; break; }
    // Under-relaxed: the ripple term perturbs the field, and a full step
    // would overshoot through a rippling surface.
    t += h.x * 0.88;
    if (t > maxDist) { mat = 0.0; break; }
  }
  return vec2(t, mat);
}

/** IQ's soft shadow: the closest approach along the ray *is* the penumbra. */
float softShadow(vec3 ro, vec3 rd, float k) {
  float res = 1.0;
  float t = 0.06;
  for (int i = 0; i < 48; i++) {
    float h = map(ro + rd * t).x;
    res = min(res, k * h / t);
    t += clamp(h, 0.02, 0.35);
    if (res < 0.004 || t > 8.0) break;
  }
  return clamp(res, 0.0, 1.0);
}

float ambientOcclusion(vec3 p, vec3 n) {
  float occ = 0.0, sca = 1.0;
  for (int i = 0; i < 5; i++) {
    float h = 0.02 + 0.14 * float(i) / 4.0;
    // If the field is closer than h, something is nearby: that is occlusion.
    occ += (h - map(p + n * h).x) * sca;
    sca *= 0.72;
  }
  return clamp(1.0 - 2.2 * occ, 0.0, 1.0);
}

/* ═══ shading ═════════════════════════════════════════════════════ */

vec3 sky(vec3 rd) {
  float h = rd.y * 0.5 + 0.5;
  vec3 top = vec3(0.045, 0.062, 0.10);
  vec3 hor = vec3(0.10, 0.11, 0.135) * 1.1;
  vec3 bot = vec3(0.015, 0.016, 0.022);
  vec3 c = mix(bot, hor, smoothstep(0.35, 0.5, h));
  c = mix(c, top, smoothstep(0.5, 1.0, h));
  // A soft sun disc so reflections have something to catch.
  float sun = pow(max(dot(rd, uLightDir), 0.0), 220.0);
  c += uTint * sun * 4.0;
  c += uTint * 0.14 * pow(max(dot(rd, uLightDir), 0.0), 5.0);
  return c;
}

vec3 material(vec3 p, vec3 n, float mat, out float rough, out float metal) {
  if (mat > 1.5) {
    // floor: a faint grid that fades out with distance
    vec2 g = abs(fract(p.xz * 0.5) - 0.5);
    float line = 1.0 - smoothstep(0.0, 0.03, min(g.x, g.y));
    float fade = exp(-length(p.xz) * 0.09);
    rough = mix(0.42, 0.12, uFloorMix);
    metal = 0.0;
    return mix(vec3(0.024, 0.026, 0.032), uTint * 0.6, line * fade * 0.5);
  }

  // body: iridescent, driven by the angle between normal and view
  rough = uRough;
  metal = 1.0;
  float f = dot(n, normalize(uCamPos - p)) * 0.5 + 0.5;
  vec3 base = cosPalette(f * 0.8 + 0.1,
    vec3(0.5, 0.5, 0.55), vec3(0.45, 0.42, 0.42),
    vec3(1.0, 0.98, 0.94), vec3(0.0, 0.22, 0.46));
  vec3 col = mix(base, uTint, 0.28);

  // A hot ring riding the wavefront. The geometric displacement alone is
  // a couple of centimetres on a smooth cream surface — technically
  // correct and almost invisible. This is emission, not geometry, so it
  // costs nothing in field stability and reads instantly.
  if (uRippleOn > 0.5 && uRippleGlow > 0.0) {
    float ring = 0.0;
    for (int i = 0; i < RIPPLE_N; i++) {
      float age = uRipples[i].w;
      if (age <= 0.0) continue;
      float d = length(p - uRipples[i].xyz);
      float front = age * uRippleSpeed;
      ring += exp(-abs(d - front) * uRippleTight * 2.4) * (1.0 - age) * (1.0 - age);
    }
    col += vec3(1.0, 0.86, 0.62) * ring * uRippleGlow * 1.6;
  }
  return col;
}

/**
 * Direct lighting only. GLSL has no recursion, so the reflection bounce
 * cannot call the full shader again — it calls *this*, and the mirror
 * ray therefore sees a correctly lit but non-reflective world. One
 * bounce is all the eye asks for, and it costs a fixed budget instead
 * of an unbounded one.
 */
vec3 shadeDirect(vec3 ro, vec3 rd, vec2 hit, out vec3 pOut, out vec3 nOut, out float roughOut) {
  pOut = ro + rd * hit.x;
  nOut = vec3(0.0, 1.0, 0.0);
  roughOut = 1.0;
  if (hit.y < 0.5) return sky(rd);

  vec3 p = pOut;
  vec3 n = calcNormal(p);
  nOut = n;

  float rough, metal;
  vec3 albedo = material(p, n, hit.y, rough, metal);
  roughOut = rough;

  vec3 l = uLightDir;
  vec3 v = -rd;
  vec3 h = normalize(l + v);

  float ndl = max(dot(n, l), 0.0);
  float sh = uShadowSoft > 0.0 ? softShadow(p + n * 0.004, l, mix(6.0, 26.0, uShadowSoft)) : 1.0;
  float occ = mix(1.0, ambientOcclusion(p, n), uAO);

  // Blinn-Phong with a roughness-derived exponent: not physically based,
  // but stable, cheap, and it reads correctly next to the SDF shadows.
  float spec = pow(max(dot(n, h), 0.0), mix(400.0, 9.0, rough)) * mix(0.35, 1.6, metal);
  float fresnel = pow(1.0 - max(dot(n, v), 0.0), 5.0);

  vec3 col = albedo * (uTint * 2.3 * ndl * sh + vec3(0.10, 0.12, 0.16) * occ);
  col += uTint * spec * sh * 2.0;
  col += sky(reflect(rd, n)) * (0.06 + fresnel * 0.9) * occ;

  // Distance fog toward the sky colour keeps the horizon from ending abruptly.
  float fog = 1.0 - exp(-hit.x * uFog * 0.045);
  return mix(col, sky(rd), fog);
}

vec3 shade(vec3 ro, vec3 rd, vec2 hit, int steps) {
  vec3 p, n; float rough;
  vec3 col = shadeDirect(ro, rd, hit, p, n, rough);
  if (hit.y < 0.5 || uReflect <= 0.0) return col;

  vec3 rd2 = reflect(rd, n);
  vec3 ro2 = p + n * 0.02;
  vec2 hit2 = march(ro2, rd2, steps / 2, 26.0);

  vec3 p2, n2; float rough2;
  vec3 refl = shadeDirect(ro2, rd2, hit2, p2, n2, rough2);

  float fresnel = pow(1.0 - max(dot(n, -rd), 0.0), 5.0);
  float amount = uReflect * mix(0.12, 0.72, 1.0 - rough) * (0.25 + fresnel * 0.75);
  return mix(col, refl, clamp(amount, 0.0, 0.9));
}

/* ═══ entry ═══════════════════════════════════════════════════════ */

void main() {
  vec2 ndc = vUv * 2.0 - 1.0;
  float aspect = uResolution.x / uResolution.y;

  // Jittered per-pixel per-frame: with the temporal blend in the
  // accumulation pass this becomes free anti-aliasing over ~4 frames.
  vec2 jitter = (hash22(gl_FragCoord.xy + uTime * 60.0) - 0.5) / uResolution;
  ndc += jitter * 2.0;

  vec3 rd = normalize(uFwd + uRight * ndc.x * aspect / uFocal + uUp * ndc.y / uFocal);
  vec2 hit = march(uCamPos, rd, uSteps, 34.0);
  vec3 col = shade(uCamPos, rd, hit, uSteps);

  // Alpha carries scene depth, so the additive flare pass can hide
  // itself behind geometry without a depth buffer ever existing.
  outColor = vec4(col, hit.y > 0.5 ? hit.x : 1e4);
}
`;

/**
 * Temporal accumulation. Written to a ping-pong pair, never in place:
 * sampling a texture that is also the current colour attachment is
 * undefined behaviour, and the artefacts it produces look plausible
 * enough to waste an evening on.
 */
const FRAG_ACCUM = /* glsl */`
${PRECISION}
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uSrc;
uniform sampler2D uHistory;
uniform float uBlend;
void main() {
  outColor = vec4(mix(texture(uSrc, vUv).rgb, texture(uHistory, vUv).rgb, uBlend), 1.0);
}
`;

const FRAG_RESOLVE = /* glsl */`
${PRECISION}
${CONSTANTS}
${COLOR}
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uSrc;
uniform float uExposure;

void main() {
  vec3 col = acesFilm(texture(uSrc, vUv).rgb * uExposure);
  vec2 q = vUv - 0.5;
  col *= 1.0 - dot(q, q) * 0.42;
  outColor = vec4(dither(col, gl_FragCoord.xy), 1.0);
}
`;

/**
 * The flare at each impact, added over the resolved image. It samples
 * the marched depth out of the render target's alpha and discards
 * anything behind the surface, so the flare is occluded by the scene
 * without a depth buffer.
 */
const VERT_FLARE = /* glsl */`
${PRECISION}
${CONSTANTS}
#define RIPPLE_N ${RIPPLE_N}

uniform vec4  uFlare[RIPPLE_N];    // xyz = centre, w = world radius
uniform float uFlareAmt[RIPPLE_N]; // 0 = unused
uniform vec3  uCamPos, uRight, uUp, uFwd;
uniform float uFocal, uAspect, uViewportH;

out float vAmt;
out float vFront;

void main() {
  int i = gl_VertexID;
  float amt = uFlareAmt[i];
  if (amt <= 0.0) {
    // Degenerate slot: park it outside the clip volume.
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    gl_PointSize = 1.0;
    vAmt = 0.0;
    vFront = 0.0;
    return;
  }

  vec4 g = uFlare[i];
  vec3 rel = g.xyz - uCamPos;
  vec3 view = vec3(dot(rel, uRight), dot(rel, uUp), dot(rel, uFwd));

  vAmt = amt;
  // Compare the *front* of the sphere against the scene, not its centre,
  // or the flare hides behind the surface it is sitting on.
  vFront = view.z - g.w;

  gl_Position = vec4(view.x * uFocal / uAspect, view.y * uFocal, 0.0, view.z);
  gl_PointSize = clamp(g.w * uViewportH * 3.0 / max(view.z, 0.05), 2.0, 500.0);
}
`;

const FRAG_FLARE = /* glsl */`
${PRECISION}
${CONSTANTS}
in float vAmt;
in float vFront;
out vec4 outColor;

uniform sampler2D uScene;
uniform vec2 uResolution;
uniform vec3 uColor;
uniform float uIntensity;

void main() {
  if (vAmt <= 0.0) discard;

  float sceneDepth = texture(uScene, gl_FragCoord.xy / uResolution).a;
  if (vFront > sceneDepth + 0.05) discard;

  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float d2 = dot(c, c);
  if (d2 > 1.0) discard;

  float halo = exp(-d2 * 3.0) * (1.0 - d2);
  outColor = vec4(uColor * halo * vAmt * uIntensity, 1.0);
}
`;

/* ═══ scene ═══════════════════════════════════════════════════════ */

const TINTS = {
  amber:  [1.0, 0.78, 0.45],
  ice:    [0.55, 0.78, 1.0],
  jade:   [0.5, 1.0, 0.72],
  rose:   [1.0, 0.55, 0.72],
};

export default {
  id: 'march',
  index: '03',
  title: 'SDF 光線行進',
  tech: 'sphere tracing · soft shadows · SSAO · impact ripples',
  desc: '整個 3D 場景沒有任何一個頂點：一道 fragment shader 沿著射線走進隱式曲面。點擊星體或星環，被碰到的那一點會盪出漣漪。',
  glyph: '◈',
  hue: 62,

  params: [
    { group: '幾何' },
    { id: 'balls', type: 'slider', label: '球體數', min: 1, max: 9, step: 1, value: 6 },
    { id: 'blend', type: 'slider', label: '融合半徑', min: 0.02, max: 0.75, step: 0.005, value: 0.32 },
    { id: 'displace', type: 'slider', label: '表面擾動', min: 0, max: 1, step: 0.01, value: 0.14 },

    { group: '撞擊' },
    // Capped: past about 0.06 the added term overwhelms the field's
    // Lipschitz bound and the tracer cuts through surfaces instead of
    // rippling them.
    { id: 'rippleAmp', type: 'slider', label: '漣漪深度', min: 0, max: 0.055, step: 0.001, value: 0.042, digits: 3 },
    { id: 'rippleSpeed', type: 'slider', label: '傳播速度', min: 0.2, max: 4, step: 0.01, value: 1.0 },
    { id: 'rippleFreq', type: 'slider', label: '波數', min: 4, max: 60, step: 0.5, value: 16 },
    { id: 'rippleLife', type: 'slider', label: '持續時間', min: 0.4, max: 6, step: 0.05, value: 1.9, unit: 's' },
    { id: 'flash', type: 'slider', label: '閃光', min: 0, max: 3, step: 0.01, value: 1.0 },

    { group: '光線' },
    { id: 'light', type: 'xy', label: '光源方向', value: [0.68, 0.24] },
    { id: 'tint', type: 'select', label: '光色', value: 'amber',
      options: [
        { value: 'amber', label: '琥珀' },
        { value: 'ice', label: '冷冽' },
        { value: 'jade', label: '翡翠' },
        { value: 'rose', label: '玫瑰' },
      ] },
    { id: 'shadow', type: 'slider', label: '陰影銳利度', min: 0, max: 1, step: 0.01, value: 0.55 },
    { id: 'ao', type: 'slider', label: '環境遮蔽', min: 0, max: 1, step: 0.01, value: 0.85 },
    { id: 'reflect', type: 'slider', label: '反射', min: 0, max: 1, step: 0.01, value: 0.55 },
    { id: 'rough', type: 'slider', label: '粗糙度', min: 0.02, max: 1, step: 0.01, value: 0.22 },

    { group: '品質' },
    { id: 'steps', type: 'slider', label: '行進步數', min: 32, max: 220, step: 1, value: 100 },
    { id: 'scale', type: 'select', label: '渲染縮放', value: '0.75',
      options: [
        { value: '0.5', label: '50%' },
        { value: '0.75', label: '75%' },
        { value: '1', label: '100%' },
      ] },
    { id: 'taa', type: 'slider', label: '時間累積', min: 0, max: 0.94, step: 0.01, value: 0.78 },
    { id: 'exposure', type: 'slider', label: '曝光', min: 0.2, max: 3, step: 0.01, value: 1.25 },
    { id: 'spin', type: 'switch', label: '自動繞行', value: true },
    { id: 'hint', type: 'hint', text: '點擊星體或星環會在該點盪出漣漪；拖曳畫布繞行鏡頭，滾輪縮放。' },
  ],

  init(ctx) { return new MarchScene(ctx); },
};

class MarchScene {
  constructor(ctx) {
    this.ctx = ctx;
    const { gl } = ctx;

    this.march = new Program(gl, VERT_FULLSCREEN, FRAG_MARCH, { name: 'march/scene' });
    this.accum = new Program(gl, VERT_FULLSCREEN, FRAG_ACCUM, { name: 'march/accum' });
    this.resolve = new Program(gl, VERT_FULLSCREEN, FRAG_RESOLVE, { name: 'march/resolve' });
    this.flare = new Program(gl, VERT_FLARE, FRAG_FLARE, { name: 'march/flare' });

    this.rt = new Target(gl, { width: 2, height: 2, format: 'rgba16f', filter: gl.LINEAR });
    this.history = new DoubleTarget(gl, { width: 2, height: 2, format: 'rgba16f', filter: gl.LINEAR });

    this.yaw = 0.85;
    this.pitch = 0.22;
    this.dist = 4.6;
    this.targetDist = 4.6;
    this.scale = 0;
    this.width = 2;
    this.height = 2;
    this.moving = 1;
    this.time = 0;

    this.basis = {
      pos: new Float32Array(3),
      right: new Float32Array(3),
      up: new Float32Array(3),
      fwd: new Float32Array(3),
    };
    this.lightDir = new Float32Array([0.5, 0.6, 0.4]);

    /* ── the cluster, computed here and uploaded ── */
    this.ballPos = new Float32Array(BALL_N * 4);
    this.ballCount = 6;
    this.blend = 0.32;

    /* ── impacts ── */
    this.ripples = new Float32Array(RIPPLE_N * 4);         // world xyz + age
    this._rippleHost = new Int32Array(RIPPLE_N).fill(-2);  // ball index, -1 = ring
    this._rippleLocal = new Float32Array(RIPPLE_N * 3);    // unit dir, or ring-local point
    this._rippleAge = new Float32Array(RIPPLE_N);
    this._rippleNext = 0;

    this.flarePts = new Float32Array(RIPPLE_N * 4);
    this.flareAmt = new Float32Array(RIPPLE_N);
    this.flash = 0;
    this.bursts = 0;

    /* ── click vs drag ── */
    this._pressed = false;
    this._dragDist = 0;
    this._pendingBurst = false;
    this._ray = new Float32Array(3);
    this._hit = new Float32Array(3);

    this._onWheel = (e) => {
      e.preventDefault();
      this.targetDist = clamp(this.targetDist * Math.exp(e.deltaY * 0.0011), 2.0, 12);
    };
    ctx.canvas.addEventListener('wheel', this._onWheel, { passive: false });
  }

  /* ── pointer ──────────────────────────────────────────────────── */

  /**
   * Click versus drag: a drag orbits the camera, a click strikes the
   * surface. They are told apart by how far the pointer travelled
   * between press and release — and that has to be decided on the
   * *events*, not on whatever the render loop happened to observe: at
   * 25 fps a quick click can begin and end between two frames.
   */
  onPointerDown() {
    this._pressed = true;
    this._dragDist = 0;
  }

  onPointerUp() {
    if (this._pressed && this._dragDist < 0.015) this._pendingBurst = true;
    this._pressed = false;
  }

  resize(width, height) {
    this.width = width;
    this.height = height;
    this._applyScale(this.scale || 0.75, true);
  }

  _applyScale(scale, force = false) {
    if (!force && scale === this.scale) return;
    this.scale = scale;
    const w = Math.max(2, Math.round(this.width * scale));
    const h = Math.max(2, Math.round(this.height * scale));
    this.rt.resize(w, h);
    this.history.resize(w, h);
    this.history.clear(0, 0, 0, 1);
  }

  reset() {
    this.yaw = 0.85;
    this.pitch = 0.22;
    this.targetDist = 4.6;
    this.history.clear(0, 0, 0, 1);
    this._rippleAge.fill(0);
    this._rippleHost.fill(-2);
    this.flash = 0;
    this.bursts = 0;
  }

  /* ── camera ───────────────────────────────────────────────────── */

  _updateCamera(state, clock, pointer) {
    let moved = false;
    if (pointer.down && pointer.moved) {
      this.yaw -= pointer.dx * 3.6;
      this.pitch = clamp(this.pitch + pointer.dy * 2.4, -0.35, 1.25);
      moved = true;
    } else if (state.spin) {
      this.yaw += clock.dt * 0.075;
      moved = clock.dt > 0;
    }
    const prevDist = this.dist;
    this.dist += (this.targetDist - this.dist) * (1 - Math.exp(-clock.wallDt * 8));
    if (Math.abs(prevDist - this.dist) > 1e-4) moved = true;
    this.moving = moved ? 1 : 0;

    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    const { pos, right, up, fwd } = this.basis;

    pos[0] = this.dist * cp * sy;
    pos[1] = this.dist * sp + 0.35;
    pos[2] = this.dist * cp * cy;

    // Aim slightly above the origin so the cluster sits on the upper
    // third of the frame rather than dead centre.
    const tx = 0, ty = 0.1, tz = 0;
    let fx = tx - pos[0], fy = ty - pos[1], fz = tz - pos[2];
    const fl = Math.hypot(fx, fy, fz) || 1;
    fx /= fl; fy /= fl; fz /= fl;
    fwd[0] = fx; fwd[1] = fy; fwd[2] = fz;

    // right = normalize(cross(fwd, worldUp)). Note the signs: the
    // opposite order gives cross(worldUp, fwd), which negates `right`,
    // which flips `up` through the cross product below — and renders
    // the whole scene upside down with the floor grid in the sky.
    let rx = -fz, ry = 0, rz = fx;
    const rl = Math.hypot(rx, ry, rz) || 1;
    rx /= rl; rz /= rl;
    right[0] = rx; right[1] = ry; right[2] = rz;

    // up = cross(right, fwd)
    up[0] = ry * fz - rz * fy;
    up[1] = rz * fx - rx * fz;
    up[2] = rx * fy - ry * fx;
  }

  /** The ray under the pointer, in world space. Matches the shader. */
  _pointerRay(pointer, out) {
    const focal = 1.5;
    const aspect = this.width / Math.max(this.height, 1);
    const ndcX = pointer.x * 2 - 1;
    const ndcY = 1 - pointer.y * 2;
    const { right, up, fwd } = this.basis;

    let x = fwd[0] + right[0] * ndcX * aspect / focal + up[0] * ndcY / focal;
    let y = fwd[1] + right[1] * ndcX * aspect / focal + up[1] * ndcY / focal;
    let z = fwd[2] + right[2] * ndcX * aspect / focal + up[2] * ndcY / focal;
    const l = Math.hypot(x, y, z) || 1;
    out[0] = x / l; out[1] = y / l; out[2] = z / l;
    return out;
  }

  /* ── the cluster, on the CPU ──────────────────────────────────── */

  /**
   * The orbits used to be derived from uTime inside the shader. They are
   * computed here now and uploaded, because picking needs the CPU to
   * agree with the GPU about where every sphere is — and two copies of
   * the same formula is exactly the kind of thing that silently drifts.
   */
  _updateBalls(state, time) {
    const t = time * 0.42;
    const n = Math.round(state.balls);
    this.ballCount = n;
    this.blend = state.blend;
    this.time = time;

    for (let i = 0; i < n; i++) {
      const a = i * 2.399963;                       // golden angle
      const o = i * 4;
      this.ballPos[o + 0] = Math.sin(t * (0.7 + i * 0.11) + a) * (0.62 + 0.1 * Math.sin(i));
      this.ballPos[o + 1] = Math.cos(t * (0.5 + i * 0.09) + a * 1.7) * 0.5;
      this.ballPos[o + 2] = Math.cos(t * (0.62 + i * 0.13) + a * 0.6) * (0.62 + 0.1 * Math.cos(i));
      this.ballPos[o + 3] = 0.30 + 0.10 * Math.sin(t * 0.9 + i * 2.1);
    }
    for (let i = n; i < BALL_N; i++) this.ballPos[i * 4 + 3] = 0;
  }

  /* The ring tumbles; these move a point in and out of its frame.
     `q.yz = rot2(a) * q.yz` with GLSL's column-major mat2(c,-s,s,c)
     expands to (c·y + s·z, −s·y + c·z), which is what these mirror. */

  _worldToRing(x, y, z, out) {
    const t = this.time * 0.42;
    const ca = Math.cos(t * 0.31), sa = Math.sin(t * 0.31);
    const y1 = ca * y + sa * z;
    const z1 = -sa * y + ca * z;
    const cb = Math.cos(t * 0.19), sb = Math.sin(t * 0.19);
    out[0] = cb * x + sb * z1;
    out[1] = y1;
    out[2] = -sb * x + cb * z1;
    return out;
  }

  _ringToWorld(x, y, z, out) {
    const t = this.time * 0.42;
    const cb = Math.cos(t * 0.19), sb = Math.sin(t * 0.19);
    const x1 = cb * x - sb * z;
    const z1 = sb * x + cb * z;
    const ca = Math.cos(t * 0.31), sa = Math.sin(t * 0.31);
    out[0] = x1;
    out[1] = ca * y - sa * z1;
    out[2] = sa * y + ca * z1;
    return out;
  }

  _ringDistance(x, y, z) {
    const q = this._worldToRing(x, y, z, this._tmpA ??= new Float32Array(3));
    const radial = Math.hypot(q[0], q[2]) - RING_MAJOR;
    return Math.hypot(radial, q[1]) - RING_MINOR;
  }

  /** The same field the shader marches, minus the terms picking can ignore. */
  _mapCPU(x, y, z) {
    let d = 1e9;
    for (let i = 0; i < this.ballCount; i++) {
      const o = i * 4;
      const s = Math.hypot(x - this.ballPos[o], y - this.ballPos[o + 1], z - this.ballPos[o + 2])
        - this.ballPos[o + 3];
      d = smin(d, s, this.blend);
    }
    return smin(d, this._ringDistance(x, y, z), this.blend * 0.6);
  }

  /**
   * March the pointer's ray on the CPU. The surface displacement and the
   * ripples are left out: they move the surface by at most a couple of
   * centimetres, which is far below the accuracy a click needs, and
   * including them would make every pick pay for a noise function.
   */
  _pick(pointer) {
    const ro = this.basis.pos;
    const rd = this._pointerRay(pointer, this._ray);

    let t = 0.05;
    let hit = false;
    for (let i = 0; i < 128; i++) {
      const d = this._mapCPU(ro[0] + rd[0] * t, ro[1] + rd[1] * t, ro[2] + rd[2] * t);
      if (d < 0.0015) { hit = true; break; }
      t += d * 0.9;
      if (t > 30) break;
    }
    if (!hit) return null;

    this._hit[0] = ro[0] + rd[0] * t;
    this._hit[1] = ro[1] + rd[1] * t;
    this._hit[2] = ro[2] + rd[2] * t;
    return this._hit;
  }

  /* ── impacts ──────────────────────────────────────────────────── */

  /**
   * A ripple has to be anchored to whatever it landed on, because
   * everything here is moving. For a sphere that means storing a
   * *direction* and re-placing the centre on the surface every frame —
   * the spheres pulse, so a fixed offset would sink into one or float
   * off another. For the ring it means storing the point in ring-local
   * coordinates and rotating it back.
   */
  _anchorRipple(px, py, pz) {
    let best = this._ringDistance(px, py, pz);
    let host = -1;

    for (let i = 0; i < this.ballCount; i++) {
      const o = i * 4;
      const d = Math.abs(
        Math.hypot(px - this.ballPos[o], py - this.ballPos[o + 1], pz - this.ballPos[o + 2])
        - this.ballPos[o + 3]);
      if (d < Math.abs(best)) { best = d; host = i; }
    }

    const i = this._rippleNext;
    this._rippleNext = (this._rippleNext + 1) % RIPPLE_N;
    this._rippleHost[i] = host;
    this._rippleAge[i] = 1e-4;

    const l = i * 3;
    if (host >= 0) {
      const o = host * 4;
      let dx = px - this.ballPos[o];
      let dy = py - this.ballPos[o + 1];
      let dz = pz - this.ballPos[o + 2];
      const dl = Math.hypot(dx, dy, dz) || 1;
      this._rippleLocal[l + 0] = dx / dl;
      this._rippleLocal[l + 1] = dy / dl;
      this._rippleLocal[l + 2] = dz / dl;
    } else {
      this._worldToRing(px, py, pz, this._tmpB ??= new Float32Array(3));
      this._rippleLocal[l + 0] = this._tmpB[0];
      this._rippleLocal[l + 1] = this._tmpB[1];
      this._rippleLocal[l + 2] = this._tmpB[2];
    }
  }

  _burst(state, pointer) {
    const p = this._pick(pointer);
    if (!p) return false;
    this._anchorRipple(p[0], p[1], p[2]);
    this.flash = 1;
    this.bursts++;
    return true;
  }

  _updateRipples(state, dt) {
    let active = 0;
    const out = this._tmpC ??= new Float32Array(3);

    for (let i = 0; i < RIPPLE_N; i++) {
      const age = this._rippleAge[i];
      if (age <= 0) {
        this.ripples[i * 4 + 3] = 0;
        this.flareAmt[i] = 0;
        this.flarePts[i * 4 + 3] = 0;
        continue;
      }

      const next = age + dt / state.rippleLife;
      const host = this._rippleHost[i];
      if (next >= 1 || (host >= 0 && host >= this.ballCount)) {
        this._rippleAge[i] = 0;
        this.ripples[i * 4 + 3] = 0;
        this.flareAmt[i] = 0;
        this.flarePts[i * 4 + 3] = 0;
        continue;
      }
      this._rippleAge[i] = next;

      const l = i * 3;
      if (host >= 0) {
        const o = host * 4;
        const r = this.ballPos[o + 3];
        this.ripples[i * 4 + 0] = this.ballPos[o + 0] + this._rippleLocal[l + 0] * r;
        this.ripples[i * 4 + 1] = this.ballPos[o + 1] + this._rippleLocal[l + 1] * r;
        this.ripples[i * 4 + 2] = this.ballPos[o + 2] + this._rippleLocal[l + 2] * r;
      } else {
        this._ringToWorld(this._rippleLocal[l], this._rippleLocal[l + 1], this._rippleLocal[l + 2], out);
        this.ripples[i * 4 + 0] = out[0];
        this.ripples[i * 4 + 1] = out[1];
        this.ripples[i * 4 + 2] = out[2];
      }
      this.ripples[i * 4 + 3] = next;

      // The flare grows briefly, then goes out well before the ripple does.
      const g = i * 4;
      this.flarePts[g + 0] = this.ripples[i * 4 + 0];
      this.flarePts[g + 1] = this.ripples[i * 4 + 1];
      this.flarePts[g + 2] = this.ripples[i * 4 + 2];
      this.flarePts[g + 3] = 0.10 + next * 0.35;
      this.flareAmt[i] = Math.max(0, 1 - next * 3.2) * state.flash;

      active++;
    }
    return active;
  }

  /* ── frame ────────────────────────────────────────────────────── */

  frame({ state, clock, pointer }) {
    const { gl, tri, empty } = this.ctx;
    this._applyScale(Number(state.scale));
    this._updateCamera(state, clock, pointer);

    // XY pad → hemisphere direction.
    const az = (state.light[0] - 0.5) * Math.PI * 2.2;
    const el = (1 - state.light[1]) * 1.35 + 0.05;
    this.lightDir[0] = Math.cos(el) * Math.sin(az);
    this.lightDir[1] = Math.sin(el);
    this.lightDir[2] = Math.cos(el) * Math.cos(az);

    const tint = TINTS[state.tint] || TINTS.amber;
    const dt = Math.min(clock.dt, 1 / 30);

    this._updateBalls(state, clock.time);

    if (pointer.down) {
      this._dragDist += Math.abs(pointer.dx) + Math.abs(pointer.dy);
    }
    if (this._pendingBurst) {
      this._pendingBurst = false;
      this._burst(state, pointer);
    }

    const rippleActive = this._updateRipples(state, dt);
    this.flash *= Math.exp(-dt * 4.5);

    // A travelling ripple invalidates the accumulated history, so the
    // temporal filter has to be told to let go of it.
    if (rippleActive) this.moving = 1;

    BLEND.none(gl);
    this.rt.bind();
    this.march.use({
      uCamPos: this.basis.pos,
      uRight: this.basis.right,
      uUp: this.basis.up,
      uFwd: this.basis.fwd,
      uResolution: [this.rt.width, this.rt.height],
      uFocal: 1.5,
      uTime: clock.time,
      uSteps: Math.round(state.steps),
      uBlend: state.blend,
      uDisplace: state.displace,
      uRough: state.rough,
      uFloorMix: 0.6,
      uLightDir: this.lightDir,
      uTint: tint,
      uReflect: state.reflect,
      uFog: 1.0,
      uAO: state.ao,
      uShadowSoft: state.shadow,

      uBallPos: this.ballPos,
      uBalls: this.ballCount,

      uRipples: this.ripples,
      uRippleOn: rippleActive > 0 ? 1 : 0,
      uRippleAmp: state.rippleAmp,
      uRippleSpeed: state.rippleSpeed,
      uRippleFreq: state.rippleFreq,
      uRippleTight: 5.0,
      uRippleGlow: state.flash,
    });
    tri.draw();

    // Temporal blend is dialled back while anything moves, or the jitter
    // turns into a smear.
    const blend = this.moving ? Math.min(state.taa, 0.55) : state.taa;

    this.history.write.bind();
    this.accum.use({
      uSrc: this.rt.texture,
      uHistory: this.history.read.texture,
      uBlend: blend,
    });
    tri.draw();
    this.history.swap();

    bindScreen(gl, this.width, this.height);
    this.resolve.use({ uSrc: this.history.read.texture, uExposure: state.exposure });
    tri.draw();

    if (rippleActive && state.flash > 0) {
      BLEND.additive(gl);
      this.flare.use({
        uFlare: this.flarePts,
        uFlareAmt: this.flareAmt,
        uCamPos: this.basis.pos,
        uRight: this.basis.right,
        uUp: this.basis.up,
        uFwd: this.basis.fwd,
        uFocal: 1.5,
        uAspect: this.width / Math.max(this.height, 1),
        uViewportH: this.height,
        uScene: this.rt.texture,
        uResolution: [this.width, this.height],
        uColor: [0.9, 0.94, 1.0],
        uIntensity: 0.55,
      });
      empty.drawPoints(RIPPLE_N);
      BLEND.none(gl);
    }
  }

  readout() {
    return {
      '渲染尺寸': `${this.rt.width}×${this.rt.height}`,
      '幾何': '0 頂點 · 0 三角形',
      '進行中的漣漪': String(this._rippleAge.reduce((n, a) => n + (a > 0 ? 1 : 0), 0)),
      '撞擊次數': String(this.bursts),
      '鏡頭距離': this.dist.toFixed(2),
    };
  }

  dispose() {
    this.ctx.canvas.removeEventListener('wheel', this._onWheel);
    this.march.dispose();
    this.accum.dispose();
    this.resolve.dispose();
    this.flare.dispose();
    this.rt.dispose();
    this.history.dispose();
  }
}

function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

/** The same polynomial smooth minimum the shader uses. */
function smin(a, b, k) {
  const h = Math.min(Math.max(0.5 + 0.5 * (b - a) / k, 0), 1);
  return b + (a - b) * h - k * h * (1 - h);
}
