/* ── scenes/march.js ─────────────────────────────────────────────────
   03 · Signed distance fields — a whole world in one fragment shader.

   There is no geometry here. Not one vertex, not one triangle: the
   entire image is produced by a single fullscreen triangle whose
   fragment shader walks a ray forward through an implicit surface
   defined by distance functions. Shadows come from marching toward the
   light, ambient occlusion from sampling the field around a point, and
   reflections from marching a second time along the mirror direction.

   The world is a handful of *composite* stars. Each one is several
   spheres welded with a smooth minimum into a single irregular mass
   that breathes as its lobes drift — the operator that makes a set of
   spheres read as one body. Stars are combined with a plain minimum,
   not a smooth one: they are separate objects you can aim at, and the
   moment two of them fuse they stop being two places to go.

   The orb that lives here is real geometry in the same sense. It goes
   into map(), so it casts soft shadows, occludes, and is reflected. It
   collides, it bounces, it can settle onto a star and ride it, and a
   click throws it off toward wherever you are pointing.
   ------------------------------------------------------------------ */

import { Program } from '../core/program.js';
import { Target, DoubleTarget, bindScreen, BLEND } from '../core/gl.js';
import { PRECISION, CONSTANTS, HASH, COLOR, ROTATE, SIMPLEX3, VERT_FULLSCREEN } from '../shaders/common.js';

// Every one of these is a term inside map(), which runs ~90 times per
// pixel for the primary ray alone, plus shadow and AO taps. They are
// sized to what reads on screen, not to what would be nice to have.
const STAR_N = 7;      // composite bodies
const LOBE_N = 5;      // spheres per body
const BODY_N = 11;     // orb + droplets
const RIPPLE_N = 4;    // concurrent impacts

/** Canonical agent space → world, plus a lift so it clears the floor. */
const AGENT_SCALE = 0.86;
const AGENT_LIFT = 0.25;
const AGENT_GIRTH = 1.5;

const FRAG_MARCH = /* glsl */`
${PRECISION}
${CONSTANTS}
${HASH}
${COLOR}
${ROTATE}
${SIMPLEX3}

#define STAR_N ${STAR_N}
#define LOBE_N ${LOBE_N}
#define BODY_N ${BODY_N}
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

uniform vec4  uLobes[STAR_N * LOBE_N];  // xyz = centre, w = radius (0 = unused)
uniform vec4  uStarBound[STAR_N];       // xyz = centre, w = bounding radius
uniform float uStarMass[STAR_N];        // 0..1, drives the colour temperature
uniform float uStarCount;

uniform vec4  uBody[BODY_N];     // xyz = centre, w = radius (0 = unused)
uniform float uBodyCount, uBodyBlend, uBodyOn, uImpact;
uniform vec4  uBodyBound;

uniform vec4  uRipples[RIPPLE_N];  // xyz = impact point, w = normalised age
uniform float uRippleOn, uRippleAmp, uRippleSpeed, uRippleFreq, uRippleTight;

/* ═══ distance functions ══════════════════════════════════════════ */

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
 * The stars.
 *
 * Thirty-five spheres would be unaffordable evaluated blindly, so each
 * body carries a bounding sphere. For any point comfortably outside it,
 * the bound's own distance is already a valid conservative step and the
 * whole lobe loop is skipped — which is almost every sample, because
 * almost every sample is somewhere else.
 *
 * The out-parameter reports which body was nearest, so the shading
 * pass can colour it without a second search. (No back-ticks in here:
 * the shader lives inside a JS template literal, and one would end it.)
 */
float starField(vec3 p, out int which) {
  float d = 1e9;
  which = 0;

  for (int s = 0; s < STAR_N; s++) {
    if (float(s) >= uStarCount) break;

    vec4 b = uStarBound[s];
    float bd = length(p - b.xyz) - b.w;
    if (bd > 0.22) {
      if (bd < d) { d = bd; which = s; }
      continue;
    }

    float sd = 1e9;
    for (int l = 0; l < LOBE_N; l++) {
      vec4 lo = uLobes[s * LOBE_N + l];
      if (lo.w <= 0.0) break;
      float ld = length(p - lo.xyz) - lo.w;
      // Lobes weld into one mass; bodies do not weld to each other.
      sd = (l == 0) ? ld : smin(sd, ld, uBlend);
    }
    if (sd < d) { d = sd; which = s; }
  }
  return d;
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
  int which;
  float d = starField(p, which);

  // Surface displacement: perturb the distance, not the geometry.
  if (uDisplace > 0.0) {
    d += uDisplace * 0.10 * snoise(p * 3.1 + vec3(0.0, 0.0, uTime * 0.3));
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
    // The orb: wet, near-mirror, lit from inside so it stays legible
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

  // A star. The iridescence is view-dependent, so a lumpy composite body
  // shades differently over each lobe and reads as one solid mass rather
  // than as flat-tinted spheres.
  rough = uRough;
  metal = 1.0;
  float f = dot(n, normalize(uCamPos - p)) * 0.5 + 0.5;
  vec3 base = cosPalette(f * 0.8 + 0.1,
    vec3(0.5, 0.5, 0.55), vec3(0.45, 0.42, 0.42),
    vec3(1.0, 0.98, 0.94), vec3(0.0, 0.22, 0.46));
  vec3 col = mix(base, uTint, 0.28);

  // Mass → colour temperature, the way it goes for real stars: the
  // bigger ones run hotter, so they sit slightly bluer and brighter.
  // Deliberately small — a few percent, not a repaint.
  int which;
  starField(p, which);
  float mass = uStarMass[which];
  col *= mix(vec3(1.05, 0.98, 0.91), vec3(0.93, 0.98, 1.09), mass);
  col *= 0.92 + 0.22 * mass;
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

  // The orb carries its own light, so it never vanishes into a shadowed
  // pocket between the stars.
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
  title: 'SDF 星群',
  tech: 'sphere tracing · composite bodies · impact ripples · perch & launch',
  desc: '整個世界沒有任何一個頂點。星體是幾顆球融成的團塊；游者撞上去會彈，也可以停在上面。點一下畫布就把牠彈射出去。',
  glyph: '◈',
  hue: 62,

  params: [
    { group: '星群' },
    { id: 'stars', type: 'slider', label: '星體數', min: 1, max: 7, step: 1, value: 5 },
    { id: 'lobes', type: 'slider', label: '每顆的塊數', min: 1, max: 5, step: 1, value: 4 },
    { id: 'blend', type: 'slider', label: '塊間融合', min: 0.02, max: 0.4, step: 0.005, value: 0.16 },
    { id: 'spread', type: 'slider', label: '星群範圍', min: 0.4, max: 2.4, step: 0.01, value: 1.45 },
    { id: 'churn', type: 'slider', label: '團塊蠕動', min: 0, max: 2, step: 0.01, value: 0.7 },
    { id: 'displace', type: 'slider', label: '表面擾動', min: 0, max: 1, step: 0.01, value: 0.08 },

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

    { group: '碰撞與跳躍' },
    { id: 'bounce', type: 'slider', label: '彈性', min: 0, max: 1.2, step: 0.01, value: 0.82 },
    { id: 'perch', type: 'switch', label: '可停棲', value: true },
    { id: 'dwell', type: 'slider', label: '停留時間', min: 0.3, max: 6, step: 0.1, value: 2.2, unit: 's' },
    { id: 'leap', type: 'slider', label: '彈射力道', min: 0.5, max: 8, step: 0.05, value: 3.6 },
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
    { id: 'hint', type: 'hint', text: '移動指標帶著游者跑，點一下畫布把牠彈射出去。拖曳繞行鏡頭，滾輪縮放。' },
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
    this.dist = 4.9;
    this.targetDist = 4.9;
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

    /* ── stars ── */
    this.lobes = new Float32Array(STAR_N * LOBE_N * 4);
    this.starBound = new Float32Array(STAR_N * 4);
    this.starMass = new Float32Array(STAR_N);
    this.starCount = 5;
    this.lobeCount = 4;
    /**
     * Fixed star positions on a golden-angle spiral, so no two bodies
     * ever end up in the same place and the arrangement is identical
     * from run to run. They drift around those anchors rather than
     * orbiting the origin: destinations you can aim at have to stay
     * where you aimed.
     *
     * The spiral is laid out over the full capacity, not over the live
     * count, so raising the star count adds a body without rearranging
     * the ones already there.
     */
    this.seeds = new Float32Array(STAR_N * 8);
    for (let i = 0; i < STAR_N; i++) {
      const gi = i + 0.5;
      const inc = Math.acos(1 - 2 * gi / STAR_N);
      const az = Math.PI * (1 + Math.sqrt(5)) * gi;
      const s = i * 4;
      const b = STAR_N * 4 + i * 3;

      this.seeds[s + 0] = 0.16 + 0.15 * ((i * 7) % 5) / 4;   // drift rate
      this.seeds[s + 1] = i * 2.399963;                       // phase
      this.seeds[s + 2] = 0.52 + 0.48 * ((i * 3) % 5) / 4;    // anchor radius
      this.seeds[s + 3] = 0.30 + 0.70 * ((i * 5) % 7) / 6;    // size

      // Unit anchor direction, squashed vertically so the group reads as
      // a field rather than as a ball of bodies.
      this.seeds[b + 0] = Math.sin(inc) * Math.cos(az);
      this.seeds[b + 1] = Math.cos(inc) * 0.52;
      this.seeds[b + 2] = Math.sin(inc) * Math.sin(az);
    }

    /* ── orb ── */
    this.body = new Float32Array(BODY_N * 4);
    this.bodyCount = 1;
    this.bodyBound = new Float32Array(4);

    /* ── impacts ── */
    this.ripples = new Float32Array(RIPPLE_N * 4);   // xyz + normalised age
    this._rippleStar = new Int32Array(RIPPLE_N).fill(-1);
    this._rippleOffset = new Float32Array(RIPPLE_N * 3);
    this._rippleAge = new Float32Array(RIPPLE_N);
    this._rippleNext = 0;
    this._cooldown = new Float32Array(STAR_N);
    this.hits = 0;
    this.leaps = 0;
    this.rippleLife = 1.9;

    /* ── perch ── */
    this.perchStar = -1;
    this.perchOffset = new Float32Array(3);
    this.perchTime = 0;

    /* ── click vs drag ── */
    this._pressed = false;
    this._dragDist = 0;
    this._pendingLaunch = false;

    this._onWheel = (e) => {
      e.preventDefault();
      this.targetDist = clamp(this.targetDist * Math.exp(e.deltaY * 0.0011), 2.2, 13);
    };
    ctx.canvas.addEventListener('wheel', this._onWheel, { passive: false });
  }

  /* ── pointer events ───────────────────────────────────────────── */

  /**
   * Click versus drag: a drag orbits the camera, a click launches the
   * orb. They are told apart by how far the pointer travelled between
   * press and release — which is what every canvas app has to do, and
   * which has to be decided on the *events*, not on whatever the render
   * loop happened to observe.
   */
  onPointerDown() {
    this._pressed = true;
    this._dragDist = 0;
  }

  onPointerUp() {
    if (this._pressed && this._dragDist < 0.015) this._pendingLaunch = true;
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
    this.targetDist = 4.9;
    this.history.clear(0, 0, 0, 1);
    this._rippleAge.fill(0);
    this._rippleStar.fill(-1);
    this._cooldown.fill(0);
    this.hits = 0;
    this.leaps = 0;
    this._unperch();
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
   * that faces the camera. Going through the camera basis rather than
   * mapping screen x/y directly is what keeps "it follows my cursor"
   * true after the camera has orbited.
   */
  _pointerWorld(pointer, out) {
    const focal = 1.5;
    const aspect = this.width / Math.max(this.height, 1);
    const sx = ((pointer.x * 2 - 1) * aspect / focal) * this.dist;
    const sy = ((1 - pointer.y * 2) / focal) * this.dist;
    const { pos, right, up, fwd } = this.basis;

    out[0] = pos[0] + fwd[0] * this.dist + right[0] * sx + up[0] * sy;
    out[1] = pos[1] + fwd[1] * this.dist + right[1] * sx + up[1] * sy;
    out[2] = pos[2] + fwd[2] * this.dist + right[2] * sx + up[2] * sy;
    return out;
  }

  /* ── stars ───────────────────────────────────────────────────── */

  /**
   * Build the composite bodies on the CPU.
   *
   * This used to live in the shader, derived from uTime. Moving it out
   * costs a few dozen uniforms and buys the one thing the shader cannot
   * give back: the ability to ask, in JavaScript, where everything is —
   * which is what collision, perching, and any game built on them need.
   */
  _updateStars(state, time) {
    const n = Math.round(state.stars);
    const k = Math.round(state.lobes);
    this.starCount = n;
    this.lobeCount = k;

    const spread = state.spread;
    const churn = state.churn;

    for (let s = 0; s < n; s++) {
      const rate = this.seeds[s * 4 + 0];
      const phase = this.seeds[s * 4 + 1];
      const anchor = this.seeds[s * 4 + 2] * spread;
      const size = this.seeds[s * 4 + 3];
      const a = STAR_N * 4 + s * 3;

      // A fixed anchor plus a small drift, not an orbit: a body that
      // sweeps across the whole group is impossible to aim at.
      const t = time * rate;
      const wobble = spread * 0.14;
      const cx = this.seeds[a + 0] * anchor + Math.sin(t + phase) * wobble;
      const cy = this.seeds[a + 1] * anchor + Math.cos(t * 0.83 + phase * 1.7) * wobble * 0.7 + 0.10;
      const cz = this.seeds[a + 2] * anchor + Math.cos(t * 0.61 + phase * 0.6) * wobble;

      const core = 0.13 + 0.22 * size;
      let bound = core;

      for (let l = 0; l < k; l++) {
        const o = (s * LOBE_N + l) * 4;
        if (l === 0) {
          this.lobes[o + 0] = cx;
          this.lobes[o + 1] = cy;
          this.lobes[o + 2] = cz;
          this.lobes[o + 3] = core;
          continue;
        }
        // Lobes drift around their own body on unrelated periods, so the
        // mass churns without any lobe ever escaping it.
        const a = phase + l * 2.399963;
        const w = time * churn * (0.45 + 0.17 * l) + a;
        const arm = core * (0.62 + 0.16 * ((l * 3) % 4));
        const lx = cx + Math.cos(w) * arm;
        const ly = cy + Math.sin(w * 1.31 + a) * arm * 0.8;
        const lz = cz + Math.sin(w * 0.79 + a * 1.4) * arm;
        const lr = core * (0.50 + 0.22 * ((l * 5) % 3));

        this.lobes[o + 0] = lx;
        this.lobes[o + 1] = ly;
        this.lobes[o + 2] = lz;
        this.lobes[o + 3] = lr;

        bound = Math.max(bound, Math.hypot(lx - cx, ly - cy, lz - cz) + lr);
      }
      for (let l = k; l < LOBE_N; l++) this.lobes[(s * LOBE_N + l) * 4 + 3] = 0;

      const b = s * 4;
      this.starBound[b + 0] = cx;
      this.starBound[b + 1] = cy;
      this.starBound[b + 2] = cz;
      // A little slack, because the smooth minimum bulges the surface
      // slightly outside the union of the raw spheres.
      this.starBound[b + 3] = bound + state.blend * 0.9;

      // Normalised mass drives the colour temperature. Using the seeded
      // size rather than the live bound keeps a body's identity steady
      // while its lobes churn.
      this.starMass[s] = clamp((size - 0.30) / 0.70, 0, 1);
    }
    for (let s = n; s < STAR_N; s++) this.starBound[s * 4 + 3] = 0;
  }

  /* ── collision, perch, launch ─────────────────────────────────── */

  _worldOrb(agent, out) {
    out[0] = agent.head[0] * AGENT_SCALE;
    out[1] = agent.head[1] * AGENT_SCALE + AGENT_LIFT;
    out[2] = agent.head[2] * AGENT_SCALE;
    return out;
  }

  _unperch() {
    this.perchStar = -1;
    this.perchTime = 0;
    this.ctx.agent.frozen = false;
  }

  /**
   * Orb versus stars. Positions are resolved before the impulse so the
   * orb can never end a frame inside a body — otherwise it re-collides
   * on the next frame, and a single touch turns into a buzz.
   *
   * An arrival slow enough to settle becomes a perch instead of a
   * bounce: the orb rides the body it landed on until it is thrown off.
   */
  _collide(state, agent, dt) {
    const R = agent.radius * AGENT_SCALE * AGENT_GIRTH;
    const p = this._worldOrb(agent, this._tmp ??= new Float32Array(3));

    for (let s = 0; s < this.starCount; s++) {
      this._cooldown[s] = Math.max(0, this._cooldown[s] - dt);

      for (let l = 0; l < this.lobeCount; l++) {
        const o = (s * LOBE_N + l) * 4;
        const r = this.lobes[o + 3];
        if (r <= 0) continue;

        let nx = p[0] - this.lobes[o];
        let ny = p[1] - this.lobes[o + 1];
        let nz = p[2] - this.lobes[o + 2];
        const d = Math.hypot(nx, ny, nz);
        const reach = R + r;
        if (d >= reach || d < 1e-5) continue;

        nx /= d; ny /= d; nz /= d;

        // Push clear of the surface, in canonical units.
        const overlap = (reach - d) / AGENT_SCALE;
        agent.displace(nx * overlap, ny * overlap, nz * overlap);
        this._worldOrb(agent, p);

        const closing = -(agent.vel[0] * nx + agent.vel[1] * ny + agent.vel[2] * nz);

        if (state.perch && this.perchStar < 0 && closing > 0 && closing < 0.5) {
          this._perchOn(s, agent);
        } else if (agent.reflect(nx, ny, nz, state.bounce) && this._cooldown[s] <= 0) {
          this._cooldown[s] = 0.2;
          this._spawnRipple(s, p[0], p[1], p[2]);
          this.hits++;
        }
      }
    }
  }

  _perchOn(star, agent) {
    const p = this._worldOrb(agent, this._tmp);
    const b = star * 4;
    this.perchStar = star;
    this.perchOffset[0] = p[0] - this.starBound[b + 0];
    this.perchOffset[1] = p[1] - this.starBound[b + 1];
    this.perchOffset[2] = p[2] - this.starBound[b + 2];
    this.perchTime = 0;
    agent.vel.fill(0);
    agent.frozen = true;
    this.hits++;
    this._spawnRipple(star, p[0], p[1], p[2]);
  }

  /** Ride the body. The offset is fixed; the body's centre is not. */
  _ridePerch(agent, dt) {
    const b = this.perchStar * 4;
    const wx = this.starBound[b + 0] + this.perchOffset[0];
    const wy = this.starBound[b + 1] + this.perchOffset[1];
    const wz = this.starBound[b + 2] + this.perchOffset[2];
    agent.head[0] = wx / AGENT_SCALE;
    agent.head[1] = (wy - AGENT_LIFT) / AGENT_SCALE;
    agent.head[2] = wz / AGENT_SCALE;
    agent.frozen = true;
    this.perchTime += dt;
  }

  /** Throw the orb toward a world point. This is the jump. */
  _launchToward(agent, target, power) {
    const p = this._worldOrb(agent, this._tmp ??= new Float32Array(3));
    let dx = target[0] - p[0];
    let dy = target[1] - p[1];
    let dz = target[2] - p[2];
    if (Math.hypot(dx, dy, dz) < 1e-4) { dx = 0; dy = 1; dz = 0; }
    this._unperch();
    agent.launch(dx, dy, dz, power);
    this.leaps++;
  }

  _spawnRipple(star, wx, wy, wz) {
    const i = this._rippleNext;
    this._rippleNext = (this._rippleNext + 1) % RIPPLE_N;
    const b = star * 4;
    this._rippleStar[i] = star;
    this._rippleOffset[i * 3 + 0] = wx - this.starBound[b + 0];
    this._rippleOffset[i * 3 + 1] = wy - this.starBound[b + 1];
    this._rippleOffset[i * 3 + 2] = wz - this.starBound[b + 2];
    this._rippleAge[i] = 1e-4;
  }

  /**
   * Ripple centres are recomputed from their host body every frame, not
   * baked at impact: the bodies are moving, and a centre frozen in
   * world space would slide off the surface it is supposed to be on.
   */
  _updateRipples(dt) {
    let active = 0;
    for (let i = 0; i < RIPPLE_N; i++) {
      const age = this._rippleAge[i];
      if (age <= 0) { this.ripples[i * 4 + 3] = 0; continue; }

      const next = age + dt / this.rippleLife;
      const star = this._rippleStar[i];
      if (next >= 1 || star < 0 || star >= this.starCount) {
        this._rippleAge[i] = 0;
        this._rippleStar[i] = -1;
        this.ripples[i * 4 + 3] = 0;
        continue;
      }
      this._rippleAge[i] = next;

      const b = star * 4;
      this.ripples[i * 4 + 0] = this.starBound[b + 0] + this._rippleOffset[i * 3 + 0];
      this.ripples[i * 4 + 1] = this.starBound[b + 1] + this._rippleOffset[i * 3 + 1];
      this.ripples[i * 4 + 2] = this.starBound[b + 2] + this._rippleOffset[i * 3 + 2];
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

    this._updateStars(state, clock.time);

    const aim = this._aimWorld ??= new Float32Array(3);
    if (pointer.active) {
      this._pointerWorld(pointer, aim);
      agent.aim(aim[0] / AGENT_SCALE, (aim[1] - AGENT_LIFT) / AGENT_SCALE, aim[2] / AGENT_SCALE);
    }

    if (pointer.down) {
      this._dragDist += Math.abs(pointer.dx) + Math.abs(pointer.dy);
    }

    if (bodyOn && dt > 0) {
      if (this._pendingLaunch) {
        this._pendingLaunch = false;
        if (pointer.active) this._launchToward(agent, aim, state.leap);
      }

      if (this.perchStar >= 0) {
        this._ridePerch(agent, dt);
        // A perch is a pause, not a parking space: after its dwell it
        // pushes off on its own so the scene never goes still.
        if (this.perchTime > state.dwell) {
          this._launchToward(agent, aim, state.leap * 0.55);
        }
      } else {
        this._collide(state, agent, dt);
      }
      this.moving = 1;
    }

    if (bodyOn) {
      this.bodyCount = agent.bodies(this.body, BODY_N, AGENT_SCALE, AGENT_LIFT, AGENT_GIRTH);
      agent.boundingSphere(this.bodyBound, AGENT_SCALE, AGENT_LIFT, AGENT_GIRTH);
      this.bodyBound[3] += agent.radius * AGENT_SCALE * AGENT_GIRTH * 0.5;
    } else if (agent.frozen) {
      this._unperch();
    }

    const rippleActive = this._updateRipples(dt);

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

      uLobes: this.lobes,
      uStarBound: this.starBound,
      uStarMass: this.starMass,
      uStarCount: this.starCount,

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
      '星群': `${this.starCount} 顆 × ${this.lobeCount} 塊`,
      '游者': `1 + ${Math.max(0, this.bodyCount - 1)} 滴`,
      '狀態': this.perchStar >= 0 ? `停棲於 #${this.perchStar + 1}` : '飛行中',
      '撞擊 / 彈射': `${this.hits} / ${this.leaps}`,
    };
  }

  dispose() {
    this.ctx.canvas.removeEventListener('wheel', this._onWheel);
    this.ctx.agent.frozen = false;
    this.march.dispose();
    this.accum.dispose();
    this.resolve.dispose();
    this.glow.dispose();
    this.rt.dispose();
    this.history.dispose();
  }
}

function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
