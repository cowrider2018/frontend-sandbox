/* ── scenes/march.js ─────────────────────────────────────────────────
   03 · Signed distance fields — a whole 3D scene in one fragment shader.

   There is no geometry here. Not one vertex, not one triangle: the
   entire image is produced by a single fullscreen triangle whose
   fragment shader walks a ray forward through an implicit surface
   defined by distance functions. Shadows come from marching toward the
   light, ambient occlusion from sampling the field around a point, and
   reflections from marching a second time along the mirror direction.
   ------------------------------------------------------------------ */

import { Program } from '../core/program.js';
import { Target, DoubleTarget, bindScreen, BLEND } from '../core/gl.js';
import { PRECISION, CONSTANTS, HASH, COLOR, ROTATE, SIMPLEX3, VERT_FULLSCREEN } from '../shaders/common.js';

const FRAG_MARCH = /* glsl */`
${PRECISION}
${CONSTANTS}
${HASH}
${COLOR}
${ROTATE}
${SIMPLEX3}

in vec2 vUv;
out vec4 outColor;

uniform vec3  uCamPos, uRight, uUp, uFwd;
uniform vec2  uResolution;
uniform float uFocal, uTime;
uniform int   uSteps;
uniform float uBlend, uBalls, uDisplace, uRough, uFloorMix;
uniform vec3  uLightDir, uTint;
uniform float uReflect, uFog, uAO, uShadowSoft;

/* ═══ distance functions ══════════════════════════════════════════ */

float sdSphere(vec3 p, float r) { return length(p) - r; }

float sdTorus(vec3 p, vec2 t) {
  vec2 q = vec2(length(p.xz) - t.x, p.y);
  return length(q) - t.y;
}

float sdBox(vec3 p, vec3 b) {
  vec3 q = abs(p) - b;
  return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

/** Polynomial smooth minimum — the operator that makes SDFs feel alive. */
float smin(float a, float b, float k) {
  float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}

/* ═══ the scene ═══════════════════════════════════════════════════ */

// x = distance, y = material id
vec2 map(vec3 p) {
  float t = uTime * 0.42;

  // A cluster of orbiting spheres, welded together by smin. Each one
  // takes a different, mutually irrational orbit so the cluster never
  // repeats a pose.
  float d = 1e9;
  for (int i = 0; i < 9; i++) {
    if (float(i) >= uBalls) break;
    float fi = float(i);
    float a = fi * 2.399963;                    // golden angle
    vec3 c = vec3(
      sin(t * (0.7 + fi * 0.11) + a) * (0.62 + 0.1 * sin(fi)),
      cos(t * (0.5 + fi * 0.09) + a * 1.7) * 0.5,
      cos(t * (0.62 + fi * 0.13) + a * 0.6) * (0.62 + 0.1 * cos(fi))
    );
    float r = 0.30 + 0.10 * sin(t * 0.9 + fi * 2.1);
    d = smin(d, sdSphere(p - c, r), uBlend);
  }

  // A ring threading the cluster, on its own slow tumble.
  vec3 q = p;
  q.yz = rot2(t * 0.31) * q.yz;
  q.xz = rot2(t * 0.19) * q.xz;
  d = smin(d, sdTorus(q, vec2(1.28, 0.075)), uBlend * 0.6);

  // Surface displacement: perturb the distance, not the geometry.
  if (uDisplace > 0.0) {
    d += uDisplace * 0.12 * snoise(p * 3.1 + vec3(0.0, 0.0, uTime * 0.3));
  }

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
    t += h.x * 0.92;
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
  return mix(base, uTint, 0.28);
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
  vec2 uv = vUv;
  vec2 ndc = uv * 2.0 - 1.0;
  float aspect = uResolution.x / uResolution.y;

  // Jittered per-pixel per-frame: with the temporal blend in the
  // composite pass this becomes free anti-aliasing over ~4 frames.
  vec2 jitter = (hash22(gl_FragCoord.xy + uTime * 60.0) - 0.5) / uResolution;
  ndc += jitter * 2.0;

  vec3 rd = normalize(uFwd + uRight * ndc.x * aspect / uFocal + uUp * ndc.y / uFocal);
  vec2 hit = march(uCamPos, rd, uSteps, 34.0);
  vec3 col = shade(uCamPos, rd, hit, uSteps);

  outColor = vec4(col, 1.0);
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
  tech: 'sphere tracing · soft shadows · SSAO · temporal AA',
  desc: '整個 3D 場景沒有任何一個頂點：一道 fragment shader 沿著射線走進隱式曲面。',
  glyph: '◈',
  hue: 62,

  params: [
    { group: '幾何' },
    { id: 'balls', type: 'slider', label: '球體數', min: 1, max: 9, step: 1, value: 6 },
    { id: 'blend', type: 'slider', label: '融合半徑', min: 0.02, max: 0.75, step: 0.005, value: 0.32 },
    { id: 'displace', type: 'slider', label: '表面擾動', min: 0, max: 1, step: 0.01, value: 0.14 },

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
    { id: 'steps', type: 'slider', label: '行進步數', min: 32, max: 220, step: 1, value: 110 },
    { id: 'scale', type: 'select', label: '渲染縮放', value: '0.75',
      options: [
        { value: '0.5', label: '50%' },
        { value: '0.75', label: '75%' },
        { value: '1', label: '100%' },
      ] },
    { id: 'taa', type: 'slider', label: '時間累積', min: 0, max: 0.94, step: 0.01, value: 0.78 },
    { id: 'exposure', type: 'slider', label: '曝光', min: 0.2, max: 3, step: 0.01, value: 1.25 },
    { id: 'spin', type: 'switch', label: '自動繞行', value: true },
    { id: 'hint', type: 'hint', text: '拖曳畫布繞行鏡頭；滾輪縮放；拖曳上方 XY 盤改變光源。' },
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
  }

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

  frame({ state, clock, pointer }) {
    const { gl, tri } = this.ctx;
    this._applyScale(Number(state.scale));
    this._updateCamera(state, clock, pointer);

    // XY pad → hemisphere direction.
    const az = (state.light[0] - 0.5) * Math.PI * 2.2;
    const el = (1 - state.light[1]) * 1.35 + 0.05;
    this.lightDir[0] = Math.cos(el) * Math.sin(az);
    this.lightDir[1] = Math.sin(el);
    this.lightDir[2] = Math.cos(el) * Math.cos(az);

    const tint = TINTS[state.tint] || TINTS.amber;

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
      uBalls: state.balls,
      uDisplace: state.displace,
      uRough: state.rough,
      uFloorMix: 0.6,
      uLightDir: this.lightDir,
      uTint: tint,
      uReflect: state.reflect,
      uFog: 1.0,
      uAO: state.ao,
      uShadowSoft: state.shadow,
    });
    tri.draw();

    // Temporal blend is dialled back while the camera moves, or the
    // jitter turns into a smear.
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
  }

  readout() {
    return {
      '渲染尺寸': `${this.rt.width}×${this.rt.height}`,
      '幾何': '0 頂點 · 0 三角形',
      '鏡頭距離': this.dist.toFixed(2),
      '鏡頭狀態': this.moving ? '移動中' : '收斂中',
    };
  }

  dispose() {
    this.ctx.canvas.removeEventListener('wheel', this._onWheel);
    this.march.dispose();
    this.accum.dispose();
    this.resolve.dispose();
    this.rt.dispose();
    this.history.dispose();
  }
}

function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
