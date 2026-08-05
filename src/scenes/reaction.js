/* ── scenes/reaction.js ──────────────────────────────────────────────
   04 · Gray–Scott reaction–diffusion.

   Two chemicals, one page of arithmetic:

     ∂u/∂t = Dᵤ∇²u − uv² + F(1 − u)
     ∂v/∂t = D_v∇²v + uv² − (F + k)v

   Every pattern in this scene — coral, mitosis, solitons, labyrinths —
   comes out of those two lines. Which one you get depends entirely on
   where (F, k) lands, and the difference between a living pattern and a
   dead grey field is often the third decimal place. The grid wraps, so
   the field is a torus with no boundary at all.
   ------------------------------------------------------------------ */

import { Program } from '../core/program.js';
import { DoubleTarget, bindScreen, BLEND } from '../core/gl.js';
import { PRECISION, CONSTANTS, HASH, COLOR, VERT_FULLSCREEN } from '../shaders/common.js';

const P = PRECISION + CONSTANTS + HASH + COLOR;

const FRAG_STEP = /* glsl */`
${P}
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uState;
uniform vec2 uTexel;
uniform float uFeed, uKill, uDu, uDv, uDt, uVariation, uTime;

/** Nine-point Laplacian — isotropic enough that patterns do not grow
    along the texel axes, which the five-point stencil visibly does. */
vec2 laplacian(vec2 uv) {
  vec2 t = uTexel;
  vec2 sum = vec2(0.0);
  sum += texture(uState, uv + vec2(-t.x, -t.y)).xy * 0.05;
  sum += texture(uState, uv + vec2( 0.0, -t.y)).xy * 0.20;
  sum += texture(uState, uv + vec2( t.x, -t.y)).xy * 0.05;
  sum += texture(uState, uv + vec2(-t.x,  0.0)).xy * 0.20;
  sum += texture(uState, uv).xy                    * -1.00;
  sum += texture(uState, uv + vec2( t.x,  0.0)).xy * 0.20;
  sum += texture(uState, uv + vec2(-t.x,  t.y)).xy * 0.05;
  sum += texture(uState, uv + vec2( 0.0,  t.y)).xy * 0.20;
  sum += texture(uState, uv + vec2( t.x,  t.y)).xy * 0.05;
  return sum;
}

void main() {
  vec2 s = texture(uState, vUv).xy;
  float u = s.x, v = s.y;
  vec2 lap = laplacian(vUv);

  // Letting F and k drift across space turns one uniform pattern into a
  // continuous atlas of every regime the model can produce.
  float f = uFeed + (vUv.x - 0.5) * 0.020 * uVariation;
  float k = uKill + (vUv.y - 0.5) * 0.010 * uVariation;

  float reaction = u * v * v;
  float du = uDu * lap.x - reaction + f * (1.0 - u);
  float dv = uDv * lap.y + reaction - (f + k) * v;

  outColor = vec4(clamp(u + du * uDt, 0.0, 1.0), clamp(v + dv * uDt, 0.0, 1.0), 0.0, 1.0);
}
`;

const FRAG_SEED = /* glsl */`
${P}
in vec2 vUv;
out vec4 outColor;
uniform vec2 uPoint;
uniform float uRadius, uAspect, uAmount;
uniform sampler2D uState;
uniform int uMode;   // 0 = paint, 1 = full reseed

void main() {
  if (uMode == 1) {
    // Clean substrate plus scattered inoculation points.
    vec2 g = floor(vUv * 22.0);
    float spot = step(0.86, hash11(g.x * 71.0 + g.y * 131.0));
    vec2 local = fract(vUv * 22.0) - 0.5;
    float blob = spot * smoothstep(0.34, 0.05, length(local));
    outColor = vec4(1.0 - blob * 0.5, blob, 0.0, 1.0);
    return;
  }

  vec2 d = vUv - uPoint;
  d.x *= uAspect;
  float amt = exp(-dot(d, d) / uRadius) * uAmount;
  vec2 s = texture(uState, vUv).xy;
  outColor = vec4(clamp(s.x - amt * 0.6, 0.0, 1.0), clamp(s.y + amt, 0.0, 1.0), 0.0, 1.0);
}
`;

const FRAG_RENDER = /* glsl */`
${P}
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uState;
uniform vec2 uTexel;
uniform vec3 uC0, uC1, uC2, uC3;
uniform float uContrast, uRelief, uExposure, uGlow;

/**
 * Four explicit stops beat a cosine palette here. A cosine ramp is
 * elegant but it cannot be dark at one end and hot at the other without
 * a fight; the substrate ends up as bright as the pattern, and the
 * structure vanishes into its own background.
 */
vec3 ramp(float t) {
  t = clamp(t, 0.0, 1.0) * 3.0;
  if (t < 1.0) return mix(uC0, uC1, t);
  if (t < 2.0) return mix(uC1, uC2, t - 1.0);
  return mix(uC2, uC3, t - 2.0);
}

void main() {
  vec2 s = texture(uState, vUv).xy;
  float v = s.y;

  // Slope of the v field, used both as a surface normal and as an
  // edge-detector — the boundary between phases is where the structure is.
  float l = texture(uState, vUv - vec2(uTexel.x, 0.0)).y;
  float r = texture(uState, vUv + vec2(uTexel.x, 0.0)).y;
  float b = texture(uState, vUv - vec2(0.0, uTexel.y)).y;
  float t = texture(uState, vUv + vec2(0.0, uTexel.y)).y;

  vec3 n = normalize(vec3((l - r) * uRelief, (b - t) * uRelief, 0.06));
  vec3 lightDir = normalize(vec3(-0.45, 0.55, 0.72));
  float diffuse = clamp(dot(n, lightDir) * 0.5 + 0.62, 0.0, 1.4);
  float spec = pow(clamp(dot(reflect(-lightDir, n), vec3(0.0, 0.0, 1.0)), 0.0, 1.0), 26.0);

  float shade = smoothstep(0.04, 0.04 + 0.36 / uContrast, v);
  vec3 col = ramp(shade);

  col *= diffuse;
  col += spec * uGlow * shade * 1.1;
  // Rim light along the phase boundary — where all the structure lives.
  float edge = clamp(length(vec2(r - l, t - b)) * 7.0, 0.0, 1.6);
  col += ramp(0.85) * uGlow * 0.5 * pow(edge, 1.4);

  col = acesFilm(col * uExposure);
  vec2 q = vUv - 0.5;
  col *= 1.0 - dot(q, q) * 0.35;
  outColor = vec4(dither(col, gl_FragCoord.xy), 1.0);
}
`;

/* ═══ presets ═════════════════════════════════════════════════════
   These (F, k) pairs are the classic named regions of the Gray–Scott
   parameter space; nudging either by 0.002 lands somewhere else.     */

const PRESETS = {
  coral:     { feed: 0.0545, kill: 0.0620 },
  mitosis:   { feed: 0.0367, kill: 0.0649 },
  solitons:  { feed: 0.0300, kill: 0.0620 },
  worms:     { feed: 0.0580, kill: 0.0650 },
  maze:      { feed: 0.0290, kill: 0.0570 },
  spirals:   { feed: 0.0180, kill: 0.0510 },
};

/** Four stops each, dark → hot. Top stops go above 1.0 on purpose:
    the tone mapper needs headroom to roll the highlights off. */
const PALETTES = {
  magma: [[0.020, 0.006, 0.030], [0.380, 0.040, 0.120], [1.050, 0.280, 0.060], [1.700, 1.250, 0.600]],
  oil:   [[0.008, 0.014, 0.040], [0.090, 0.330, 0.560], [0.760, 0.320, 0.880], [1.250, 1.350, 1.500]],
  jade:  [[0.006, 0.022, 0.016], [0.030, 0.300, 0.210], [0.360, 0.980, 0.560], [1.250, 1.550, 0.950]],
  ink:   [[0.010, 0.010, 0.013], [0.130, 0.140, 0.165], [0.560, 0.590, 0.640], [1.350, 1.370, 1.400]],
};

export default {
  id: 'reaction',
  index: '04',
  title: 'Gray–Scott 反應擴散',
  tech: '9-point laplacian · 16 sub-steps/frame · toroidal domain',
  desc: '兩行方程式長出珊瑚、迷宮、細胞分裂與孤立子；差別只在小數點後第三位。',
  glyph: '❋',
  hue: 155,

  params: [
    { group: '化學' },
    { id: 'preset', type: 'select', label: '型態', value: 'coral',
      options: [
        { value: 'coral', label: '珊瑚' },
        { value: 'mitosis', label: '分裂' },
        { value: 'solitons', label: '孤子' },
        { value: 'worms', label: '蠕蟲' },
        { value: 'maze', label: '迷宮' },
        { value: 'spirals', label: '螺旋' },
      ] },
    { id: 'feed', type: 'slider', label: '進料率 F', min: 0.01, max: 0.09, step: 0.0002, value: 0.0545, digits: 4 },
    { id: 'kill', type: 'slider', label: '移除率 k', min: 0.03, max: 0.075, step: 0.0002, value: 0.0620, digits: 4 },
    { id: 'variation', type: 'slider', label: '空間漸變', min: 0, max: 1, step: 0.01, value: 0.35 },

    { group: '數值' },
    { id: 'res', type: 'select', label: '網格', value: '512',
      options: [
        { value: '256', label: '256' },
        { value: '512', label: '512' },
        { value: '768', label: '768' },
        { value: '1024', label: '1024' },
      ] },
    { id: 'steps', type: 'slider', label: '每幀子步', min: 1, max: 40, step: 1, value: 16 },
    { id: 'dt', type: 'slider', label: '步長', min: 0.2, max: 1.4, step: 0.01, value: 1.0 },

    { group: '渲染' },
    { id: 'palette', type: 'select', label: '配色', value: 'magma',
      options: [
        { value: 'magma', label: '岩漿' },
        { value: 'oil', label: '油膜' },
        { value: 'jade', label: '翡翠' },
        { value: 'ink', label: '墨' },
      ] },
    { id: 'contrast', type: 'slider', label: '對比', min: 0.4, max: 4, step: 0.01, value: 0.9 },
    { id: 'relief', type: 'slider', label: '浮雕', min: 0, max: 3, step: 0.01, value: 1.1 },
    { id: 'glow', type: 'slider', label: '輝光', min: 0, max: 1, step: 0.01, value: 0.32 },
    { id: 'exposure', type: 'slider', label: '曝光', min: 0.2, max: 3, step: 0.01, value: 1.15 },

    { group: '筆刷' },
    { id: 'brush', type: 'slider', label: '半徑', min: 0.02, max: 0.4, step: 0.005, value: 0.09 },
    { id: 'hint', type: 'hint', text: '在畫布上拖曳即可播種；不同的 F/k 會長出完全不同的東西。' },
  ],

  init(ctx) { return new ReactionScene(ctx); },
};

class ReactionScene {
  constructor(ctx) {
    this.ctx = ctx;
    const { gl } = ctx;

    this.stepProg   = new Program(gl, VERT_FULLSCREEN, FRAG_STEP,   { name: 'reaction/step' });
    this.seedProg   = new Program(gl, VERT_FULLSCREEN, FRAG_SEED,   { name: 'reaction/seed' });
    this.renderProg = new Program(gl, VERT_FULLSCREEN, FRAG_RENDER, { name: 'reaction/render' });

    // REPEAT wrapping makes the domain a torus: no walls, no artefacts
    // creeping in from the edges, and the pattern tiles seamlessly.
    this.state = new DoubleTarget(gl, {
      width: 2, height: 2, format: 'rg16f', filter: gl.LINEAR, wrap: gl.REPEAT,
    });

    this.res = 0;
    this.width = 2;
    this.height = 2;
    this.aspect = 1;
    this.lastPreset = 'coral';
    this.elapsedSteps = 0;
  }

  resize(width, height) {
    this.width = width;
    this.height = height;
    this.aspect = width / Math.max(height, 1);
    this._resizeGrid(this.res || 512, true);
  }

  _resizeGrid(res, force = false) {
    if (!force && res === this.res) return;
    this.res = res;
    const w = this.aspect >= 1 ? Math.round(res * this.aspect) : res;
    const h = this.aspect >= 1 ? res : Math.round(res / this.aspect);
    this.state.resize(w, h);
    this.reset();
  }

  reset() {
    const { gl, tri } = this.ctx;
    BLEND.none(gl);
    this.state.write.bind();
    this.seedProg.use({ uMode: 1, uState: this.state.read.texture, uPoint: [0, 0], uRadius: 1, uAspect: 1, uAmount: 0 });
    tri.draw();
    this.state.swap();
    this.elapsedSteps = 0;
  }

  _paint(x, y, radius, amount) {
    const { gl, tri } = this.ctx;
    BLEND.none(gl);
    this.state.write.bind();
    this.seedProg.use({
      uMode: 0,
      uState: this.state.read.texture,
      uPoint: [x, 1 - y],
      uRadius: radius * radius * 0.04,
      uAspect: this.aspect,
      uAmount: amount,
    });
    tri.draw();
    this.state.swap();
  }

  frame({ state, clock, pointer }) {
    const { gl, tri } = this.ctx;
    this._resizeGrid(Number(state.res));

    // A preset changes F and k together; the panel writes them back so
    // the sliders stay in sync and remain editable afterwards.
    if (state.preset !== this.lastPreset) {
      this.lastPreset = state.preset;
      const p = PRESETS[state.preset];
      if (p) this.ctx.setParams({ feed: p.feed, kill: p.kill });
    }

    if (pointer.down) this._paint(pointer.x, pointer.y, state.brush, 0.85);

    if (clock.dt > 0) {
      BLEND.none(gl);
      const substeps = Math.round(state.steps);
      for (let i = 0; i < substeps; i++) {
        this.state.write.bind();
        this.stepProg.use({
          uState: this.state.read.texture,
          uTexel: this.state.texelSize,
          uFeed: state.feed,
          uKill: state.kill,
          uDu: 0.21,
          uDv: 0.105,
          uDt: state.dt,
          uVariation: state.variation,
          uTime: clock.time,
        });
        tri.draw();
        this.state.swap();
      }
      this.elapsedSteps += substeps;
    }

    const pal = PALETTES[state.palette] || PALETTES.magma;
    bindScreen(gl, this.width, this.height);
    this.renderProg.use({
      uState: this.state.read.texture,
      uTexel: this.state.texelSize,
      uC0: pal[0], uC1: pal[1], uC2: pal[2], uC3: pal[3],
      uContrast: state.contrast,
      uRelief: state.relief,
      uGlow: state.glow,
      uExposure: state.exposure,
    });
    tri.draw();
  }

  readout(state) {
    return {
      '網格': `${this.state.width}×${this.state.height}`,
      '每幀子步': String(Math.round(state.steps)),
      '累計步數': this.elapsedSteps.toLocaleString('en-US'),
      '邊界': '環面 (REPEAT)',
    };
  }

  dispose() {
    this.stepProg.dispose();
    this.seedProg.dispose();
    this.renderProg.dispose();
    this.state.dispose();
  }
}
