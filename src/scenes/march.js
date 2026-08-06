/* ── scenes/march.js ─────────────────────────────────────────────────
   03 · Signed distance fields — a whole 3D scene in one fragment shader.

   There is no geometry here. Not one vertex, not one triangle: the
   entire image is produced by a single fullscreen triangle whose
   fragment shader walks a ray forward through an implicit surface
   defined by distance functions. Shadows come from marching toward the
   light, ambient occlusion from sampling the field around a point, and
   reflections from marching a second time along the mirror direction.

   The orb that swims through it is real geometry in that same sense —
   it goes into map(), so it casts real soft shadows, occludes, and is
   reflected, none of which is drawn or faked. When it strikes one of
   the bodies, the impact point is recorded and a ring displaces the
   surface outward from it: the star ripples, because the ripple *is*
   the surface.

   The bodies' positions are computed on the CPU and uploaded, rather
   than derived from uTime in the shader. That costs nine uniforms and
   buys the one thing the shader cannot give back: the ability to ask,
   in JavaScript, where everything is — which is what collision, and
   any game built on top of it, needs.
   ------------------------------------------------------------------ */

import { Program } from '../core/program.js';
import { Target, DoubleTarget, bindScreen, BLEND } from '../core/gl.js';
import { PRECISION, CONSTANTS, HASH, COLOR, ROTATE, SIMPLEX3, VERT_FULLSCREEN } from '../shaders/common.js';

// Every one of these is a term inside map(), which runs ~90 times per
// pixel for the primary ray alone, plus shadow and AO taps. They are
// sized to what reads on screen, not to what would be nice to have.
const BALL_N = 9;      // orbiting bodies
const BODY_N = 11;     // orb + droplets
const RIPPLE_N = 4;    // concurrent impacts

/** Canonical agent space → world, plus a lift so it clears the floor. */
const AGENT_SCALE = 0.82;
const AGENT_LIFT = 0.28;
const AGENT_GIRTH = 1.55;

const FRAG_MARCH = /* glsl */`
${PRECISION}
${CONSTANTS}
${HASH}
${COLOR}
${ROTATE}
${SIMPLEX3}

#define BALL_N ${BALL_N}
#define BODY_N ${BODY_N}
#define RIPPLE_N ${RIPPLE_N}

in vec2 vUv;
layout(location = 0) out vec4 outColor;

uniform vec3  uCamPos, uRight, uUp, uFwd;
uniform vec2  uResolution;
uniform float uFocal, uTime;
uniform int   uSteps;
uniform float uBlend, uDisplace, uRough, uFloorMix;
uniform vec3  uLightDir, uTint;
uniform float uReflect, uFog, uAO, uShadowSoft;

uniform vec4  uBalls[BALL_N];    // xyz = centre, w = radius
uniform float uBallCount;

uniform vec4  uBody[BODY_N];     // xyz = centre, w = radius (0 = unused)
uniform float uBodyCount, uBodyBlend, uBodyOn, uImpact;
uniform vec4  uBodyBound;

uniform vec4  uRipples[RIPPLE_N];  // xyz = impact point, w = normalised age
uniform float uRippleOn, uRippleAmp, uRippleSpeed, uRippleFreq, uRippleTight;

/* ═══ distance functions ══════════════════════════════════════════ */

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
 * Displacing the distance field is not the same as displacing a mesh:
 * there is no surface to move, so the ripple is authored as a term
 * added to the distance itself. Keep the amplitude small — a large one
 * breaks the field's Lipschitz bound and the sphere tracer starts
 * overshooting through the surface.
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

/**
 * The swimmer: one orb plus whatever droplets it has thrown off,
 * welded with a smooth minimum. A droplet shed a moment ago is still
 * within the blend radius and stays connected; as it drifts clear the
 * neck thins and pinches. That is the whole "viscous body" effect —
 * one operator, no simulation.
 */
float swimmer(vec3 p) {
  float bound = length(p - uBodyBound.xyz) - uBodyBound.w;
  if (bound > 0.22) return bound;

  float d = length(p - uBody[0].xyz) - uBody[0].w;
  for (int i = 1; i < BODY_N; i++) {
    if (float(i) >= uBodyCount) break;
    d = smin(d, length(p - uBody[i].xyz) - uBody[i].w, uBodyBlend);
  }
  return d;
}

/* ═══ the scene ═══════════════════════════════════════════════════ */

// x = distance, y = material id
vec2 map(vec3 p) {
  float t = uTime * 0.42;

  float d = 1e9;
  for (int i = 0; i < BALL_N; i++) {
    if (float(i) >= uBallCount) break;
    d = smin(d, length(p - uBalls[i].xyz) - uBalls[i].w, uBlend);
  }

  // A ring threading the cluster, on its own slow tumble. Kept thin and
  // barely blended so it reads as a separate object the orb can bounce
  // off rather than as part of the stars.
  vec3 q = p;
  q.yz = rot2(t * 0.31) * q.yz;
  q.xz = rot2(t * 0.19) * q.xz;
  d = smin(d, sdTorus(q, vec2(1.28, 0.045)), uBlend * 0.35);

  // Surface displacement: perturb the distance, not the geometry.
  if (uDisplace > 0.0) {
    d += uDisplace * 0.12 * snoise(p * 3.1 + vec3(0.0, 0.0, uTime * 0.3));
  }
  if (uRippleOn > 0.5) d += ripples(p);

  vec2 res = vec2(d, 1.0);

  float floorD = p.y + 1.35;
  if (floorD < res.x) res = vec2(floorD, 2.0);

  if (uBodyOn > 0.5) {
    float c = swimmer(p);
    if (c < res.x) res = vec2(c, 3.0);
  }

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
    // Under-relaxed because the ripple term perturbs the field: a full
    // step would overshoot straight through a rippling surface.
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
  if (mat > 2.5) {
    // The swimmer: wet, near-mirror, lit from inside so it stays legible
    // against whatever it is passing through, and flashing on impact.
    rough = 0.06;
    metal = 1.0;
    float away = clamp(length(p - uBody[0].xyz) / max(uBody[0].w * 4.0, 1e-3), 0.0, 1.0);
    vec3 base = mix(vec3(1.20, 1.10, 0.92), vec3(0.20, 0.52, 1.05), away);
    return base + vec3(1.0, 0.75, 0.45) * uImpact * 1.6;
  }
  if (mat > 1.5) {
    // floor: a faint grid that fades out with distance
    vec2 g = abs(fract(p.xz * 0.5) - 0.5);
    float line = 1.0 - smoothstep(0.0, 0.03, min(g.x, g.y));
    float fade = exp(-length(p.xz) * 0.09);
    rough = mix(0.42, 0.12, uFloorMix);
    metal = 0.0;
    return mix(vec3(0.024, 0.026, 0.032), uTint * 0.6, line * fade * 0.5);
  }
  // A star. Each one takes its hue from its own index rather than from
  // its position, so a body keeps its identity as it orbits — which is
  // the whole point once you are meant to aim at a particular one.
  // Cheap despite the loop: material() runs once per hit, not once per
  // march step.
  int nearest = 0;
  float best = 1e9;
  for (int i = 0; i < BALL_N; i++) {
    if (float(i) >= uBallCount) break;
    float d = length(p - uBalls[i].xyz) - uBalls[i].w;
    if (d < best) { best = d; nearest = i; }
  }

  rough = uRough;
  metal = 1.0;
  float hue = fract(float(nearest) * 0.6180339887 + 0.06);
  float f = dot(n, normalize(uCamPos - p)) * 0.5 + 0.5;
  vec3 base = cosPalette(hue,
    vec3(0.42, 0.40, 0.44), vec3(0.40, 0.38, 0.40),
    vec3(1.0, 1.0, 1.0), vec3(0.0, 0.33, 0.67));
  // Pulled a quarter of the way toward its own luminance: enough hue to
  // tell the bodies apart, not so much that the scene turns into sweets.
  base = mix(base, vec3(dot(base, vec3(0.299, 0.587, 0.114))), 0.28);
  // Rim brightening keeps the spheres reading as spheres once they stop
  // being welded into one blob.
  return base * (0.55 + 0.75 * f) + uTint * 0.10;
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

  // The swimmer carries its own light, so it never vanishes into a
  // shadowed pocket of the cluster.
  if (hit.y > 2.5) col += albedo * 0.45;

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

  // Alpha carries scene depth so the additive droplet-glow pass can
  // hide itself behind geometry without a depth buffer.
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
 * The droplet glow: a soft additive halo around the orb and each shed
 * droplet, drawn over the resolved image. The bodies themselves are
 * already in the SDF — this is only the light they give off, which a
 * single reflection bounce cannot express.
 *
 * It samples the marched depth out of the render target's alpha and
 * discards anything behind the surface, so the glow is occluded by the
 * scene without a depth buffer ever existing.
 */
const VERT_GLOW = /* glsl */`
${PRECISION}
${CONSTANTS}
#define BODY_N ${BODY_N}

uniform vec4  uBody[BODY_N];
uniform vec3  uCamPos, uRight, uUp, uFwd;
uniform float uFocal, uAspect, uViewportH, uSpread;

out float vFade;
out float vDepth;

void main() {
  int i = gl_VertexID;
  vec4 b = uBody[i];
  if (b.w <= 0.0) {
    // Degenerate slot: park it outside the clip volume.
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    gl_PointSize = 1.0;
    vFade = 0.0;
    vDepth = 0.0;
    return;
  }

  vec3 rel = b.xyz - uCamPos;
  vec3 view = vec3(dot(rel, uRight), dot(rel, uUp), dot(rel, uFwd));

  vFade = i == 0 ? 1.0 : 0.55;
  vDepth = view.z;

  gl_Position = vec4(view.x * uFocal / uAspect, view.y * uFocal, 0.0, view.z);
  gl_PointSize = clamp(b.w * uSpread * uViewportH / max(view.z, 0.05), 2.0, 400.0);
}
`;

const FRAG_GLOW = /* glsl */`
${PRECISION}
${CONSTANTS}
in float vFade;
in float vDepth;
out vec4 outColor;

uniform sampler2D uScene;
uniform vec2 uResolution;
uniform vec3 uColor;
uniform float uIntensity;

void main() {
  if (vFade <= 0.0) discard;

  // Occlusion: alpha of the march target holds the distance to whatever
  // the camera actually hit at this pixel.
  float sceneDepth = texture(uScene, gl_FragCoord.xy / uResolution).a;
  if (vDepth > sceneDepth + 0.06) discard;

  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float d2 = dot(c, c);
  if (d2 > 1.0) discard;

  float halo = exp(-d2 * 3.0) * (1.0 - d2);
  outColor = vec4(uColor * halo * vFade * uIntensity, 1.0);
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
  desc: '整個 3D 場景沒有任何一個頂點。游者撞上星體時，被推開的是距離場本身。',
  glyph: '◈',
  hue: 62,

  params: [
    { group: '星體' },
    { id: 'balls', type: 'slider', label: '星體數', min: 1, max: 9, step: 1, value: 6 },
    // Low by default: the stars are targets now, and targets have to be
    // individually identifiable. Turn it up to weld them into one mass.
    { id: 'blend', type: 'slider', label: '融合半徑', min: 0.02, max: 0.75, step: 0.005, value: 0.06 },
    { id: 'displace', type: 'slider', label: '表面擾動', min: 0, max: 1, step: 0.01, value: 0.06 },

    { group: '游者' },
    { id: 'agent', type: 'switch', label: '放入游者', value: true },
    { id: 'agentMode', type: 'select', label: '行為', value: 'follow',
      options: [
        { value: 'wander', label: '漫遊' },
        { value: 'follow', label: '跟隨' },
        { value: 'flee', label: '迴避' },
      ] },
    { id: 'agentSpeed', type: 'slider', label: '泳速', min: 0.2, max: 3, step: 0.01, value: 1.0 },
    { id: 'shed', type: 'slider', label: '剝離量', min: 0, max: 3, step: 0.01, value: 1.0 },
    { id: 'cohesion', type: 'slider', label: '黏滯（融合）', min: 0.01, max: 0.35, step: 0.005, value: 0.13 },
    { id: 'glow', type: 'slider', label: '輝光', min: 0, max: 2, step: 0.01, value: 0.7 },

    { group: '碰撞' },
    { id: 'bounce', type: 'slider', label: '彈性', min: 0, max: 1.2, step: 0.01, value: 0.82 },
    // Capped: past about 0.06 the added term overwhelms the field's
    // Lipschitz bound and the tracer starts cutting through surfaces
    // instead of rippling them.
    { id: 'rippleAmp', type: 'slider', label: '漣漪深度', min: 0, max: 0.055, step: 0.001, value: 0.040, digits: 3 },
    { id: 'rippleSpeed', type: 'slider', label: '傳播速度', min: 0.2, max: 4, step: 0.01, value: 1.1 },
    { id: 'rippleFreq', type: 'slider', label: '波數', min: 4, max: 60, step: 0.5, value: 17 },

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
    { id: 'steps', type: 'slider', label: '行進步數', min: 32, max: 220, step: 1, value: 92 },
    { id: 'scale', type: 'select', label: '渲染縮放', value: '0.75',
      options: [
        { value: '0.5', label: '50%' },
        { value: '0.75', label: '75%' },
        { value: '1', label: '100%' },
      ] },
    { id: 'taa', type: 'slider', label: '時間累積', min: 0, max: 0.94, step: 0.01, value: 0.62 },
    { id: 'exposure', type: 'slider', label: '曝光', min: 0.2, max: 3, step: 0.01, value: 1.25 },
    { id: 'spin', type: 'switch', label: '自動繞行', value: true },
    { id: 'hint', type: 'hint', text: '移動指標帶著游者撞向星體；拖曳畫布繞行鏡頭，滾輪縮放。' },
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
    this.glow = new Program(gl, VERT_GLOW, FRAG_GLOW, { name: 'march/glow' });

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

    this.basis = {
      pos: new Float32Array(3),
      right: new Float32Array(3),
      up: new Float32Array(3),
      fwd: new Float32Array(3),
    };
    this.lightDir = new Float32Array([0.5, 0.6, 0.4]);

    /* ── bodies ── */
    this.balls = new Float32Array(BALL_N * 4);
    this.ballCount = 6;
    this.body = new Float32Array(BODY_N * 4);
    this.bodyCount = 1;
    this.bodyBound = new Float32Array(4);

    /* ── impacts ── */
    this.ripples = new Float32Array(RIPPLE_N * 4);   // xyz + normalised age
    this._rippleBall = new Int32Array(RIPPLE_N).fill(-1);
    this._rippleNormal = new Float32Array(RIPPLE_N * 3);
    this._rippleAge = new Float32Array(RIPPLE_N);
    this._rippleNext = 0;
    this._cooldown = new Float32Array(BALL_N);
    this.hits = 0;
    this.rippleLife = 1.9;

    this._onWheel = (e) => {
      e.preventDefault();
      this.targetDist = clamp(this.targetDist * Math.exp(e.deltaY * 0.0011), 2.0, 12);
    };
    ctx.canvas.addEventListener('wheel', this._onWheel, { passive: false });
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
    this._rippleBall.fill(-1);
    this._cooldown.fill(0);
    this.hits = 0;
    this.ctx.agent.reset();
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

  /**
   * Un-project the pointer onto the plane through the scene's centre
   * that faces the camera, then divide out the scene scale to get a
   * target in the agent's own space. Going through the camera basis
   * rather than mapping screen x/y directly is what keeps "the orb
   * follows my cursor" true after the camera has orbited.
   */
  _aimAgent(pointer, agent) {
    const focal = 1.5;
    const aspect = this.width / Math.max(this.height, 1);
    const ndcX = pointer.x * 2 - 1;
    const ndcY = 1 - pointer.y * 2;

    const sx = (ndcX * aspect / focal) * this.dist;
    const sy = (ndcY / focal) * this.dist;
    const { pos, right, up, fwd } = this.basis;

    const wx = pos[0] + fwd[0] * this.dist + right[0] * sx + up[0] * sy;
    const wy = pos[1] + fwd[1] * this.dist + right[1] * sx + up[1] * sy;
    const wz = pos[2] + fwd[2] * this.dist + right[2] * sx + up[2] * sy;

    agent.aim(wx / AGENT_SCALE, (wy - AGENT_LIFT) / AGENT_SCALE, wz / AGENT_SCALE);
  }

  /* ── bodies ───────────────────────────────────────────────────── */

  /**
   * The orbits used to live in the shader. Moving them to the CPU costs
   * nine uniforms and buys the ability to ask, in JavaScript, where
   * every body is — which is what collision needs, and what anything
   * built on top of collision will need.
   */
  _updateBalls(state, time) {
    const t = time * 0.42;
    const n = Math.round(state.balls);
    this.ballCount = n;

    for (let i = 0; i < n; i++) {
      const a = i * 2.399963;                       // golden angle
      const o = i * 4;
      this.balls[o + 0] = Math.sin(t * (0.7 + i * 0.11) + a) * (0.62 + 0.1 * Math.sin(i));
      this.balls[o + 1] = Math.cos(t * (0.5 + i * 0.09) + a * 1.7) * 0.5;
      this.balls[o + 2] = Math.cos(t * (0.62 + i * 0.13) + a * 0.6) * (0.62 + 0.1 * Math.cos(i));
      this.balls[o + 3] = 0.30 + 0.10 * Math.sin(t * 0.9 + i * 2.1);
    }
    for (let i = n; i < BALL_N; i++) this.balls[i * 4 + 3] = 0;
  }

  /**
   * Orb versus bodies. Positions are resolved before the impulse so the
   * orb can never end a frame inside a body — otherwise it re-collides
   * on the next frame, and a single touch turns into a buzz.
   */
  _collide(state, agent, dt) {
    const R = agent.radius * AGENT_SCALE * AGENT_GIRTH;
    const px = agent.head[0] * AGENT_SCALE;
    const py = agent.head[1] * AGENT_SCALE + AGENT_LIFT;
    const pz = agent.head[2] * AGENT_SCALE;

    for (let i = 0; i < this.ballCount; i++) {
      this._cooldown[i] = Math.max(0, this._cooldown[i] - dt);

      const o = i * 4;
      const r = this.balls[o + 3];
      let nx = px - this.balls[o];
      let ny = py - this.balls[o + 1];
      let nz = pz - this.balls[o + 2];
      const d = Math.hypot(nx, ny, nz);
      const reach = R + r;
      if (d >= reach || d < 1e-5) continue;

      nx /= d; ny /= d; nz /= d;

      // Push clear of the surface, in canonical units.
      const overlap = (reach - d) / AGENT_SCALE;
      agent.displace(nx * overlap, ny * overlap, nz * overlap);

      const bounced = agent.reflect(nx, ny, nz, state.bounce);
      if (bounced && this._cooldown[i] <= 0) {
        this._cooldown[i] = 0.22;
        this._spawnRipple(i, nx, ny, nz);
        this.hits++;
      }
    }
  }

  _spawnRipple(ball, nx, ny, nz) {
    const i = this._rippleNext;
    this._rippleNext = (this._rippleNext + 1) % RIPPLE_N;
    this._rippleBall[i] = ball;
    this._rippleNormal[i * 3 + 0] = nx;
    this._rippleNormal[i * 3 + 1] = ny;
    this._rippleNormal[i * 3 + 2] = nz;
    this._rippleAge[i] = 1e-4;
  }

  /**
   * Ripple centres are recomputed from their host body every frame, not
   * baked at impact: the bodies are orbiting, and a centre frozen in
   * world space would slide off the surface it is supposed to be on.
   */
  _updateRipples(dt) {
    let active = 0;
    for (let i = 0; i < RIPPLE_N; i++) {
      const age = this._rippleAge[i];
      if (age <= 0) { this.ripples[i * 4 + 3] = 0; continue; }

      const next = age + dt / this.rippleLife;
      if (next >= 1) {
        this._rippleAge[i] = 0;
        this._rippleBall[i] = -1;
        this.ripples[i * 4 + 3] = 0;
        continue;
      }
      this._rippleAge[i] = next;

      const b = this._rippleBall[i];
      const o = b * 4;
      const r = this.balls[o + 3];
      this.ripples[i * 4 + 0] = this.balls[o + 0] + this._rippleNormal[i * 3 + 0] * r;
      this.ripples[i * 4 + 1] = this.balls[o + 1] + this._rippleNormal[i * 3 + 1] * r;
      this.ripples[i * 4 + 2] = this.balls[o + 2] + this._rippleNormal[i * 3 + 2] * r;
      this.ripples[i * 4 + 3] = next;
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
    const agent = this.ctx.agent;
    const bodyOn = state.agent !== false;
    const dt = Math.min(clock.dt, 1 / 30);

    this._updateBalls(state, clock.time);

    if (bodyOn) {
      if (pointer.active) this._aimAgent(pointer, agent);
      if (dt > 0) this._collide(state, agent, dt);
      this.bodyCount = agent.bodies(this.body, BODY_N, AGENT_SCALE, AGENT_LIFT, AGENT_GIRTH);
      agent.boundingSphere(this.bodyBound, AGENT_SCALE, AGENT_LIFT, AGENT_GIRTH);
      this.bodyBound[3] += agent.radius * AGENT_SCALE * AGENT_GIRTH * 0.5;
      // A moving body invalidates the accumulated history everywhere it
      // has been, so the temporal filter has to be told to let go.
      this.moving = 1;
    }
    const rippleActive = dt > 0 ? this._updateRipples(dt) : this._updateRipples(0);

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

      uBalls: this.balls,
      uBallCount: this.ballCount,

      uBody: this.body,
      uBodyCount: bodyOn ? this.bodyCount : 0,
      uBodyBlend: state.cohesion,
      uBodyOn: bodyOn ? 1 : 0,
      uBodyBound: this.bodyBound,
      uImpact: agent.impact,

      uRipples: this.ripples,
      uRippleOn: rippleActive > 0 ? 1 : 0,
      uRippleAmp: state.rippleAmp,
      uRippleSpeed: state.rippleSpeed,
      uRippleFreq: state.rippleFreq,
      uRippleTight: 3.6,
    });
    tri.draw();

    // Temporal blend is dialled back while anything moves, or the
    // jitter turns into a smear.
    const blend = this.moving ? Math.min(state.taa, 0.5) : state.taa;

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

    if (bodyOn && state.glow > 0) {
      BLEND.additive(gl);
      this.glow.use({
        uBody: this.body,
        uCamPos: this.basis.pos,
        uRight: this.basis.right,
        uUp: this.basis.up,
        uFwd: this.basis.fwd,
        uFocal: 1.5,
        uAspect: this.width / Math.max(this.height, 1),
        uViewportH: this.height,
        uSpread: 5.5,
        uScene: this.rt.texture,
        uResolution: [this.width, this.height],
        uColor: [0.55, 0.78, 1.0],
        uIntensity: state.glow * 0.5,
      });
      empty.drawPoints(this.bodyCount);
      BLEND.none(gl);
    }
  }

  readout() {
    return {
      '渲染尺寸': `${this.rt.width}×${this.rt.height}`,
      '幾何': '0 頂點 · 0 三角形',
      '星體': String(this.ballCount),
      '游者球體': `1 + ${Math.max(0, this.bodyCount - 1)} 滴`,
      '撞擊次數': String(this.hits),
      '鏡頭距離': this.dist.toFixed(2),
    };
  }

  dispose() {
    this.ctx.canvas.removeEventListener('wheel', this._onWheel);
    this.march.dispose();
    this.accum.dispose();
    this.resolve.dispose();
    this.glow.dispose();
    this.rt.dispose();
    this.history.dispose();
  }
}

function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
