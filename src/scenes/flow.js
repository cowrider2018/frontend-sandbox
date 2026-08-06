/* ── scenes/flow.js ──────────────────────────────────────────────────
   01 · Curl-noise flow field — up to 1,048,576 particles.

   Every particle's position and velocity lives in a floating-point
   texture. Each frame two fullscreen passes integrate the whole system
   on the GPU, then one attribute-less draw call renders it: the vertex
   shader turns gl_VertexID into a texel coordinate and pulls the
   position it needs. Nothing crosses the PCIe bus — the CPU's entire
   contribution is four uniforms and a draw call.
   ------------------------------------------------------------------ */

import { Program } from '../core/program.js';
import { Target, DoubleTarget, bindScreen, BLEND } from '../core/gl.js';
import { PRECISION, CONSTANTS, HASH, COLOR, SIMPLEX3, CURL, VERT_FULLSCREEN } from '../shaders/common.js';

const PRELUDE = PRECISION + CONSTANTS + HASH;

/* ═══ palettes ════════════════════════════════════════════════════
   IQ cosine palettes: (a, b, c, d) → a + b·cos(2π(c·t + d))        */

/** Canonical agent space → this scene's world units. */
const AGENT_SCALE = 1.18;
const SWIM_HEAD = [1.0, 0.92, 0.78];
const SWIM_TAIL = [0.16, 0.55, 1.0];

const PALETTES = {
  ice:    [[0.5, 0.55, 0.62], [0.42, 0.38, 0.35], [1.0, 1.0, 1.0], [0.62, 0.52, 0.35]],
  ember:  [[0.52, 0.32, 0.22], [0.48, 0.34, 0.24], [1.0, 0.95, 0.9], [0.02, 0.16, 0.32]],
  spectra:[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [0.0, 0.33, 0.67]],
  mono:   [[0.62, 0.64, 0.68], [0.36, 0.35, 0.34], [1.0, 1.0, 1.0], [0.0, 0.05, 0.1]],
};

/* ═══ shaders ═════════════════════════════════════════════════════ */

const FRAG_VELOCITY = /* glsl */`
${PRELUDE}
${SIMPLEX3}
${CURL}
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uPos;
uniform sampler2D uVel;
uniform float uDt, uTime;
uniform float uNoiseScale, uSpeed, uEvolve, uInertia, uDrag;
uniform float uContain, uSoftRadius;
uniform vec3  uAttract;
uniform float uAttractStrength;
uniform vec3  uSwimmer, uSwimmerVel;
uniform float uWake, uWakeRadius;

void main() {
  vec3 pos = texture(uPos, vUv).xyz;
  vec3 vel = texture(uVel, vUv).xyz;

  // The 4th dimension is time: sliding the sample point through z makes
  // the whole field evolve smoothly instead of snapping between states.
  vec3 field = curlNoise(pos * uNoiseScale + vec3(0.0, 0.0, uTime * uEvolve)) * uSpeed;

  // Exponential blend → frame-rate independent inertia.
  vel = mix(vel, field, 1.0 - exp(-uDt * uInertia));

  vec3 toA = uAttract - pos;
  float d2 = dot(toA, toA) + 0.3;
  vel += (uAttractStrength * toA / d2) * uDt;

  // Curl noise is divergence-free, but it is not *bounded*: over a few
  // seconds the field's large-scale drift carries the whole cloud out of
  // frame. A soft radial spring that only engages past uSoftRadius keeps
  // the swarm on screen without flattening the motion inside it.
  float r = length(pos);
  vel -= pos * (max(r - uSoftRadius, 0.0) * uContain / max(r, 1e-3)) * uDt;

  // The swimmer's wake. A body moving through a fluid does two things
  // to it: shoulders the medium aside along its heading, and sheds a
  // vortex ring around that heading. The cross product is the second
  // one, and it is what makes the disturbance read as a wake rather
  // than as a bulldozer.
  if (uWake > 0.0) {
    vec3 rel = pos - uSwimmer;
    float g = exp(-dot(rel, rel) / (uWakeRadius * uWakeRadius));
    float sp = length(uSwimmerVel);
    if (sp > 1e-4) {
      vec3 heading = uSwimmerVel / sp;
      vel += (heading * 1.1 + cross(heading, rel) * 3.4) * g * uWake * sp * uDt;
    }
  }

  vel *= exp(-uDt * uDrag);

  outColor = vec4(vel, 0.0);
}
`;

const FRAG_POSITION = /* glsl */`
${PRELUDE}
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uPos;
uniform sampler2D uVel;
uniform float uDt, uTime, uLifespan, uBounds, uSpawnRadius;

void main() {
  vec4 P = texture(uPos, vUv);
  vec3 vel = texture(uVel, vUv).xyz;

  vec3 pos = P.xyz + vel * uDt;

  // Per-particle lifespan jitter keeps respawns from pulsing in unison.
  float jitter = 0.55 + 0.9 * hash11(vUv.x * 7919.0 + vUv.y * 104729.0);
  float life = P.w - uDt / (uLifespan * jitter);

  if (life <= 0.0 || dot(pos, pos) > uBounds * uBounds) {
    vec3 r = seedFromIndex(vUv, floor(uTime * 13.0));
    // Uniform point in a ball: cube-root the radius, or everything
    // clusters at the centre.
    float theta = r.x * TAU;
    float z = r.y * 2.0 - 1.0;
    float ring = sqrt(max(0.0, 1.0 - z * z));
    float rad = uSpawnRadius * pow(max(r.z, 1e-4), 1.0 / 3.0);
    pos = vec3(ring * cos(theta), ring * sin(theta), z) * rad;
    life = 1.0;
  }

  outColor = vec4(pos, life);
}
`;

const VERT_POINTS = /* glsl */`
${PRECISION}
${CONSTANTS}
uniform sampler2D uPos;
uniform sampler2D uVel;
uniform float uSide;
uniform vec3 uCamPos, uRight, uUp, uFwd;
uniform float uFocal, uAspect, uViewportH, uPointSize;

out float vSpeed;
out float vLife;
out float vDepth;
out float vDir;

void main() {
  int n = int(uSide);
  int id = gl_VertexID;
  vec2 uv = (vec2(float(id % n), float(id / n)) + 0.5) / uSide;

  vec4 P = texture(uPos, uv);
  vec3 vel = texture(uVel, uv).xyz;

  vec3 rel = P.xyz - uCamPos;
  vec3 view = vec3(dot(rel, uRight), dot(rel, uUp), dot(rel, uFwd));

  vSpeed = length(vel);
  vLife = P.w;
  vDepth = view.z;

  // Colour by direction of travel, the way optical flow is visualised:
  // it turns an undifferentiated haze of particles into a readable map
  // of the field, because neighbours moving together share a hue.
  vec3 dir = vSpeed > 1e-5 ? vel / vSpeed : vec3(0.0, 1.0, 0.0);
  vDir = fract(atan(dir.y, dir.x) / TAU + 0.5 + dir.z * 0.12);

  // z = 0 in clip space: depth testing is off, additive blending is
  // order-independent, so there is nothing to sort and nothing to test.
  gl_Position = vec4(view.x * uFocal / uAspect, view.y * uFocal, 0.0, view.z);
  // ALIASED_POINT_SIZE_RANGE starts at 1 on this class of hardware, and a
  // gl_PointSize below that is not clamped — the point is dropped. Never
  // let the perspective divide take it under 1.
  gl_PointSize = clamp(uPointSize * uViewportH * 0.0085 / max(view.z, 0.05), 1.0, 48.0);
}
`;

const FRAG_POINTS = /* glsl */`
${PRECISION}
${CONSTANTS}
${COLOR}
in float vSpeed;
in float vLife;
in float vDepth;
in float vDir;
out vec4 outColor;

uniform vec3 uPalA, uPalB, uPalC, uPalD;
uniform float uIntensity, uSpeedScale;

void main() {
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float d2 = dot(c, c);
  if (d2 > 1.0) discard;

  // Gaussian-ish falloff: sprites overlap into a continuous field
  // rather than reading as a spray of discrete discs.
  float sprite = exp(-d2 * 3.2) * (1.0 - d2);

  // Fade in and out over the particle's life so nothing pops.
  float fade = smoothstep(0.0, 0.12, vLife) * smoothstep(1.0, 0.82, vLife);

  vec3 col = cosPalette(vDir, uPalA, uPalB, uPalC, uPalD);

  // Fast particles burn toward white; slow ones keep their hue. Without
  // this the image is evenly bright everywhere and reads as fog.
  float energy = clamp(vSpeed * uSpeedScale, 0.0, 1.6);
  col = mix(col * 0.35, col, energy);
  col += vec3(1.0) * pow(sprite, 4.0) * energy * 0.5;

  // Cheap aerial perspective — distant particles cool and dim.
  float fog = exp(-max(vDepth - 2.0, 0.0) * 0.16);

  outColor = vec4(col * sprite * fade * fog * uIntensity * (0.35 + energy), 1.0);
}
`;

/**
 * The swimmer, drawn as a chain of overlapping glowing sprites.
 *
 * The spine arrives as a uniform array and the vertex shader samples
 * *between* nodes — `SWIM_SUB` points per segment — so the body is a
 * continuous ribbon rather than a string of beads, without uploading a
 * single byte of geometry. It draws additively into the same buffer as
 * the particles, which means the bloom pass gets it for free.
 */
const VERT_SWIMMER = /* glsl */`
${PRECISION}
${CONSTANTS}
#define SWIM_MAX 40

uniform vec3  uNodes[SWIM_MAX];
uniform float uRadii[SWIM_MAX];
uniform float uSub, uCount, uScale, uSize;
uniform vec3  uCamPos, uRight, uUp, uFwd;
uniform float uFocal, uAspect, uViewportH;

out float vU;

void main() {
  int sub = int(uSub);
  int i = gl_VertexID / sub;
  float f = float(gl_VertexID % sub) / uSub;

  vec3 p = mix(uNodes[i], uNodes[i + 1], f) * uScale;
  float r = mix(uRadii[i], uRadii[i + 1], f) * uScale;
  vU = (float(i) + f) / (uCount - 1.0);

  vec3 rel = p - uCamPos;
  vec3 view = vec3(dot(rel, uRight), dot(rel, uUp), dot(rel, uFwd));

  gl_Position = vec4(view.x * uFocal / uAspect, view.y * uFocal, 0.0, view.z);
  gl_PointSize = clamp(r * uSize * uViewportH / max(view.z, 0.05), 2.0, 260.0);
}
`;

const FRAG_SWIMMER = /* glsl */`
${PRECISION}
${CONSTANTS}
in float vU;
out vec4 outColor;
uniform vec3 uHead, uTail;
uniform float uIntensity;

void main() {
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float d2 = dot(c, c);
  if (d2 > 1.0) discard;

  float body = exp(-d2 * 2.4) * (1.0 - d2);
  vec3 col = mix(uHead, uTail, pow(vU, 0.7));
  // White-hot core, strongest at the head — the eye reads this as the
  // front of the animal without any explicit anatomy.
  col += vec3(1.0) * pow(body, 5.0) * (1.0 - vU) * 2.2;

  outColor = vec4(col * body * (1.0 - vU * 0.55) * uIntensity, 1.0);
}
`;

const FRAG_FADE = /* glsl */`
${PRECISION}
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uSrc;
uniform float uDecay;
void main() {
  vec4 c = texture(uSrc, vUv) * uDecay;
  // Snap the tail to zero, or half-float denormals keep a ghost forever.
  outColor = max(c - 2e-4, vec4(0.0));
}
`;

/**
 * Bloom, in three cheap passes at quarter resolution.
 *
 * 1. prefilter — keep only what is brighter than the knee
 * 2. blur ×2   — separable Gaussian, horizontal then vertical
 * 3. composite — add back
 *
 * Doing it at ¼ resolution is not a compromise: the blur radius needed
 * for a convincing glow is large in screen space, so the low-frequency
 * result is identical and costs a sixteenth of the bandwidth.
 */
const FRAG_PREFILTER = /* glsl */`
${PRECISION}
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uSrc;
uniform float uThreshold, uKnee;

void main() {
  vec3 c = texture(uSrc, vUv).rgb;
  float lum = max(c.r, max(c.g, c.b));
  // Soft knee: a hard cutoff makes the bloom pop on and off as things
  // cross the threshold.
  float soft = clamp(lum - uThreshold + uKnee, 0.0, 2.0 * uKnee);
  soft = soft * soft / (4.0 * uKnee + 1e-4);
  float contribution = max(soft, lum - uThreshold) / max(lum, 1e-4);
  outColor = vec4(c * contribution, 1.0);
}
`;

const FRAG_BLUR = /* glsl */`
${PRECISION}
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uSrc;
uniform vec2 uDirection;   // texel-sized step, one axis at a time

void main() {
  // 9-tap Gaussian folded into 5 bilinear fetches.
  const float o[3] = float[3](0.0, 1.3846153846, 3.2307692308);
  const float w[3] = float[3](0.2270270270, 0.3162162162, 0.0702702703);
  vec3 sum = texture(uSrc, vUv).rgb * w[0];
  for (int i = 1; i < 3; i++) {
    sum += texture(uSrc, vUv + uDirection * o[i]).rgb * w[i];
    sum += texture(uSrc, vUv - uDirection * o[i]).rgb * w[i];
  }
  outColor = vec4(sum, 1.0);
}
`;

const FRAG_COMPOSITE = /* glsl */`
${PRECISION}
${CONSTANTS}
${COLOR}
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uSrc;
uniform sampler2D uBloom;
uniform float uExposure, uBloomStrength;

void main() {
  vec3 hdr = texture(uSrc, vUv).rgb;
  hdr += texture(uBloom, vUv).rgb * uBloomStrength;
  vec3 col = acesFilm(hdr * uExposure);

  vec2 q = vUv - 0.5;
  col *= 1.0 - dot(q, q) * 0.55;

  outColor = vec4(dither(col, gl_FragCoord.xy), 1.0);
}
`;

/* ═══ scene ═══════════════════════════════════════════════════════ */

export default {
  id: 'flow',
  index: '01',
  title: '旋度噪聲流場',
  tech: 'GPGPU · vertex pulling · divergence-free advection',
  desc: '位置與速度存在浮點貼圖裡，兩道全螢幕 pass 在 GPU 上積分整個系統。',
  glyph: '✳',
  hue: 205,

  params: [
    { group: '模擬' },
    { id: 'count', type: 'select', label: '粒子數', value: '512',
      options: [
        { value: '256', label: '65k' },
        { value: '512', label: '262k' },
        { value: '1024', label: '1.05M' },
      ] },
    { id: 'speed', type: 'slider', label: '流速', min: 0, max: 3, step: 0.01, value: 1.15 },
    { id: 'scale', type: 'slider', label: '噪聲尺度', min: 0.05, max: 1.2, step: 0.005, value: 0.30 },
    { id: 'evolve', type: 'slider', label: '場演化', min: 0, max: 1.5, step: 0.01, value: 0.26 },
    { id: 'inertia', type: 'slider', label: '慣性', min: 0.5, max: 20, step: 0.1, value: 7.5 },
    { id: 'life', type: 'slider', label: '壽命', min: 0.5, max: 12, step: 0.1, value: 4.5, unit: 's' },

    { group: '渲染' },
    { id: 'palette', type: 'select', label: '配色', value: 'ice',
      options: [
        { value: 'ice', label: '冰' },
        { value: 'ember', label: '燼' },
        { value: 'spectra', label: '光譜' },
        { value: 'mono', label: '素' },
      ] },
    { id: 'size', type: 'slider', label: '粒徑', min: 0.2, max: 4, step: 0.05, value: 1.15 },
    { id: 'trail', type: 'slider', label: '拖尾', min: 0, max: 0.99, step: 0.01, value: 0.94 },
    { id: 'bloom', type: 'slider', label: '輝光', min: 0, max: 2.5, step: 0.01, value: 0.85 },
    { id: 'exposure', type: 'slider', label: '曝光', min: 0.1, max: 4, step: 0.01, value: 1.4 },

    { group: '游者' },
    { id: 'agent', type: 'switch', label: '顯示游者', value: true },
    { id: 'agentMode', type: 'select', label: '行為', value: 'follow',
      options: [
        { value: 'wander', label: '漫遊' },
        { value: 'follow', label: '跟隨' },
        { value: 'flee', label: '迴避' },
      ] },
    { id: 'agentSpeed', type: 'slider', label: '泳速', min: 0.2, max: 3, step: 0.01, value: 1.05 },
    { id: 'wake', type: 'slider', label: '尾流強度', min: 0, max: 3, step: 0.01, value: 1.15 },
    { id: 'swimSize', type: 'slider', label: '體型', min: 0.3, max: 3, step: 0.01, value: 1.35 },
    { id: 'swimGlow', type: 'slider', label: '發光', min: 0, max: 2, step: 0.01, value: 0.55 },

    { group: '鏡頭' },
    { id: 'spin', type: 'switch', label: '自動繞行', value: true },
    { id: 'attract', type: 'switch', label: '指標吸引', value: false },
    { id: 'hint', type: 'hint', text: '移動指標帶著游者跑；拖曳畫布繞行鏡頭；滾輪縮放。' },
  ],

  init(ctx) { return new FlowScene(ctx); },
};

class FlowScene {
  constructor(ctx) {
    this.ctx = ctx;
    const { gl } = ctx;

    this.velProg  = new Program(gl, VERT_FULLSCREEN, FRAG_VELOCITY,  { name: 'flow/velocity' });
    this.posProg  = new Program(gl, VERT_FULLSCREEN, FRAG_POSITION,  { name: 'flow/position' });
    this.drawProg = new Program(gl, VERT_POINTS,     FRAG_POINTS,    { name: 'flow/points' });
    this.fadeProg = new Program(gl, VERT_FULLSCREEN, FRAG_FADE,      { name: 'flow/fade' });
    this.swimProg = new Program(gl, VERT_SWIMMER,    FRAG_SWIMMER,   { name: 'flow/swimmer' });
    this.preProg  = new Program(gl, VERT_FULLSCREEN, FRAG_PREFILTER, { name: 'flow/prefilter' });
    this.blurProg = new Program(gl, VERT_FULLSCREEN, FRAG_BLUR,      { name: 'flow/blur' });
    this.compProg = new Program(gl, VERT_FULLSCREEN, FRAG_COMPOSITE, { name: 'flow/composite' });

    this.side = 0;
    this.pos = null;
    this.vel = null;

    this.accum = new DoubleTarget(gl, { width: 2, height: 2, format: 'rgba16f', filter: gl.LINEAR });
    this.bloomA = new Target(gl, { width: 2, height: 2, format: 'rgba16f', filter: gl.LINEAR });
    this.bloomB = new Target(gl, { width: 2, height: 2, format: 'rgba16f', filter: gl.LINEAR });
    this.width = 2;
    this.height = 2;

    // camera
    this.yaw = 0.6;
    this.pitch = 0.28;
    this.dist = 6.2;
    this.targetDist = 6.2;
    this.dragging = false;
    this.attractor = new Float32Array([0, 0, 0]);
    this.swimPos = new Float32Array(3);
    this.swimVel = new Float32Array(3);
    this.basis = {
      pos: new Float32Array(3),
      right: new Float32Array(3),
      up: new Float32Array(3),
      fwd: new Float32Array(3),
    };

    this._onWheel = (e) => {
      e.preventDefault();
      this.targetDist = clamp(this.targetDist * Math.exp(e.deltaY * 0.0012), 1.6, 22);
    };
    ctx.canvas.addEventListener('wheel', this._onWheel, { passive: false });

    this._allocate(512);
  }

  /* ── particle state ───────────────────────────────────────────── */

  _allocate(side) {
    if (side === this.side) return;
    const gl = this.ctx.gl;

    this.pos?.dispose();
    this.vel?.dispose();

    this.side = side;
    this.count = side * side;

    this.pos = new DoubleTarget(gl, { width: side, height: side, format: 'rgba32f', filter: gl.NEAREST });
    this.vel = new DoubleTarget(gl, { width: side, height: side, format: 'rgba16f', filter: gl.NEAREST });

    this._seed();
  }

  /** Seed positions on the CPU once; from then on the GPU owns them. */
  _seed() {
    const gl = this.ctx.gl;
    const n = this.count;
    const data = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
      // uniform in a ball
      let x, y, z, d;
      do {
        x = Math.random() * 2 - 1;
        y = Math.random() * 2 - 1;
        z = Math.random() * 2 - 1;
        d = x * x + y * y + z * z;
      } while (d > 1 || d < 1e-6);
      const r = 2.1;
      data[i * 4 + 0] = x * r;
      data[i * 4 + 1] = y * r;
      data[i * 4 + 2] = z * r;
      data[i * 4 + 3] = Math.random(); // staggered initial life
    }

    for (const target of [this.pos.a, this.pos.b]) {
      gl.bindTexture(gl.TEXTURE_2D, target.texture);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.side, this.side, gl.RGBA, gl.FLOAT, data);
    }
    gl.bindTexture(gl.TEXTURE_2D, null);
    this.vel.clear(0, 0, 0, 0);
    this.accum.clear(0, 0, 0, 1);
  }

  reset() { this._seed(); this.yaw = 0.6; this.pitch = 0.28; this.targetDist = 6.2; }

  resize(width, height) {
    this.width = width;
    this.height = height;
    this.accum.resize(width, height);
    this.accum.clear(0, 0, 0, 1);

    const bw = Math.max(2, Math.round(width / 4));
    const bh = Math.max(2, Math.round(height / 4));
    this.bloomA.resize(bw, bh);
    this.bloomB.resize(bw, bh);
    this.bloomA.clear(0, 0, 0, 1);
    this.bloomB.clear(0, 0, 0, 1);
  }

  /* ── camera ───────────────────────────────────────────────────── */

  _updateCamera(state, clock, pointer) {
    if (pointer.down && pointer.moved) {
      this.yaw -= pointer.dx * 4.2;
      this.pitch = clamp(this.pitch + pointer.dy * 3.0, -1.35, 1.35);
      this.dragging = true;
    } else if (state.spin) {
      this.yaw += clock.dt * 0.06;
    }

    this.dist += (this.targetDist - this.dist) * (1 - Math.exp(-clock.wallDt * 8));

    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);

    const { pos, right, up, fwd } = this.basis;
    pos[0] = this.dist * cp * sy;
    pos[1] = this.dist * sp;
    pos[2] = this.dist * cp * cy;

    // forward = -normalize(pos), i.e. always looking at the origin
    fwd[0] = -cp * sy; fwd[1] = -sp; fwd[2] = -cp * cy;
    // right = normalize(cross(fwd, worldUp))
    right[0] = cy; right[1] = 0; right[2] = -sy;
    // up = cross(right, fwd) — the signs matter: get them backwards and
    // the image is mirrored, which is almost invisible on a symmetric
    // scene and completely wrong on any other.
    up[0] = -sp * sy; up[1] = cp; up[2] = -sp * cy;
  }

  /** Project the pointer onto the plane through the origin facing the camera. */
  _updateAttractor(state, pointer) {
    if (!state.attract || !pointer.active) {
      this.attractor[0] = this.attractor[1] = this.attractor[2] = 0;
      return 0;
    }
    const focal = 1 / Math.tan(0.5 * 0.95);
    const aspect = this.width / Math.max(this.height, 1);
    const ndcX = pointer.x * 2 - 1;
    const ndcY = 1 - pointer.y * 2;

    const sx = (ndcX * aspect / focal) * this.dist;
    const sy = (ndcY / focal) * this.dist;

    const { pos, right, up, fwd } = this.basis;
    for (let i = 0; i < 3; i++) {
      this.attractor[i] = pos[i] + fwd[i] * this.dist + right[i] * sx + up[i] * sy;
    }
    return pointer.down ? 3.4 : 0.85;
  }

  /* ── frame ────────────────────────────────────────────────────── */

  frame({ state, clock, pointer }) {
    const { gl, tri, empty } = this.ctx;

    const side = Number(state.count);
    if (side !== this.side) this._allocate(side);

    this._updateCamera(state, clock, pointer);
    const attractStrength = this._updateAttractor(state, pointer);

    // The attractor is already the pointer un-projected onto the plane
    // through the origin, so dividing by the scene scale hands the agent
    // a target in its own canonical space — correct under any camera
    // orbit, which a naive screen-space mapping would not be.
    const agent = this.ctx.agent;
    if (pointer.active) {
      agent.aim(
        this.attractor[0] / AGENT_SCALE,
        this.attractor[1] / AGENT_SCALE,
        this.attractor[2] / AGENT_SCALE,
      );
    }
    for (let i = 0; i < 3; i++) {
      this.swimPos[i] = agent.nodes[i] * AGENT_SCALE;
      this.swimVel[i] = agent.vel[i] * AGENT_SCALE;
    }

    const dt = Math.min(clock.dt, 1 / 30);

    /* 1 ── velocity integration */
    if (dt > 0) {
      BLEND.none(gl);
      this.vel.write.bind();
      this.velProg.use({
        uPos: this.pos.read.texture,
        uVel: this.vel.read.texture,
        uDt: dt,
        uTime: clock.time,
        uNoiseScale: state.scale,
        uSpeed: state.speed * 1.6,
        uEvolve: state.evolve,
        uInertia: state.inertia,
        uDrag: 0.25,
        uContain: 2.6,
        uSoftRadius: 2.4,
        uAttract: this.attractor,
        uAttractStrength: attractStrength,
        uSwimmer: this.swimPos,
        uSwimmerVel: this.swimVel,
        uWake: state.agent === false ? 0 : state.wake,
        uWakeRadius: 0.85,
      });
      tri.draw();
      this.vel.swap();

      /* 2 ── position integration + respawn */
      this.pos.write.bind();
      this.posProg.use({
        uPos: this.pos.read.texture,
        uVel: this.vel.read.texture,
        uDt: dt,
        uTime: clock.time,
        uLifespan: state.life,
        uBounds: 6.5,
        uSpawnRadius: 1.7,
      });
      tri.draw();
      this.pos.swap();
    }

    /* 3 ── fade the accumulation buffer (this is the motion trail) */
    const decay = state.trail <= 0 ? 0 : Math.pow(state.trail, Math.max(clock.wallDt, 1e-3) * 60);
    BLEND.none(gl);
    this.accum.write.bind();
    this.fadeProg.use({ uSrc: this.accum.read.texture, uDecay: decay });
    tri.draw();

    /* 4 ── additive splat of every particle, then the swimmer */
    BLEND.additive(gl);
    const pal = PALETTES[state.palette] || PALETTES.ice;
    this.drawProg.use({
      uPos: this.pos.read.texture,
      uVel: this.vel.read.texture,
      uSide: this.side,
      uCamPos: this.basis.pos,
      uRight: this.basis.right,
      uUp: this.basis.up,
      uFwd: this.basis.fwd,
      uFocal: 1 / Math.tan(0.5 * 0.95),
      uAspect: this.width / Math.max(this.height, 1),
      uViewportH: this.height,
      uPointSize: state.size,
      uPalA: pal[0], uPalB: pal[1], uPalC: pal[2], uPalD: pal[3],
      // Fewer particles must each carry more light, or changing the count
      // would double as a brightness slider.
      uIntensity: 0.075 * (262144 / this.count) ** 0.4,
      uSpeedScale: 0.62 / Math.max(state.speed, 0.05),
    });
    empty.drawPoints(this.count);

    if (state.agent !== false) {
      const agent = this.ctx.agent;
      const sub = 9;
      this.swimProg.use({
        uNodes: agent.nodes,
        uRadii: agent.radii,
        uSub: sub,
        uCount: agent.count,
        uScale: AGENT_SCALE,
        uSize: state.swimSize,
        uCamPos: this.basis.pos,
        uRight: this.basis.right,
        uUp: this.basis.up,
        uFwd: this.basis.fwd,
        uFocal: 1 / Math.tan(0.5 * 0.95),
        uAspect: this.width / Math.max(this.height, 1),
        uViewportH: this.height,
        uHead: SWIM_HEAD,
        uTail: SWIM_TAIL,
        // Each pixel of the body is covered by ~`sub` overlapping
        // sprites, so the per-sprite value has to be divided by that or
        // the body clips to flat white. The trail buffer does *not* add
        // another factor of 1/(1-decay): the body is moving, so any one
        // pixel is only under it for a couple of frames.
        uIntensity: state.swimGlow * 0.30 / sub,
      });
      empty.drawPoints((agent.count - 1) * sub);
    }

    this.accum.swap();

    /* 5 ── bloom: prefilter → blur H → blur V */
    BLEND.none(gl);
    this.bloomA.bind();
    this.preProg.use({ uSrc: this.accum.read.texture, uThreshold: 0.55, uKnee: 0.4 });
    tri.draw();

    const bt = this.bloomA.texelSize;
    this.bloomB.bind();
    this.blurProg.use({ uSrc: this.bloomA.texture, uDirection: [bt[0], 0] });
    tri.draw();

    this.bloomA.bind();
    this.blurProg.use({ uSrc: this.bloomB.texture, uDirection: [0, bt[1]] });
    tri.draw();

    /* 6 ── tone map to the screen */
    bindScreen(gl, this.width, this.height);
    this.compProg.use({
      uSrc: this.accum.read.texture,
      uBloom: this.bloomA.texture,
      uExposure: state.exposure,
      uBloomStrength: state.bloom,
    });
    tri.draw();
  }

  readout(state) {
    return {
      '粒子': this.count.toLocaleString('en-US'),
      '狀態貼圖': `${this.side}² RGBA32F`,
      '每幀 pass': '8',
      '游者泳速': this.ctx.agent.speed.toFixed(2),
      '鏡頭距離': this.dist.toFixed(2),
    };
  }

  dispose() {
    this.ctx.canvas.removeEventListener('wheel', this._onWheel);
    this.pos?.dispose();
    this.vel?.dispose();
    this.accum.dispose();
    this.bloomA.dispose();
    this.bloomB.dispose();
    for (const p of [this.velProg, this.posProg, this.drawProg, this.swimProg,
                     this.fadeProg, this.preProg, this.blurProg, this.compProg]) p.dispose();
  }
}

function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
