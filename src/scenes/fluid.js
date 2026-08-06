/* ── scenes/fluid.js ─────────────────────────────────────────────────
   02 · Navier–Stokes — Stam's stable fluids, on the GPU.

   The incompressible Navier–Stokes equations, solved the way Jos Stam
   showed in 1999: advect the velocity field along itself with a
   semi-Lagrangian backward trace (unconditionally stable at any dt),
   add vorticity confinement to put back the small eddies that the
   trace smears away, then project the field divergence-free by solving
   a Poisson equation for pressure with Jacobi iterations.

   That is eight to forty render passes per frame, every one of them a
   fullscreen triangle over a float texture.
   ------------------------------------------------------------------ */

import { Program } from '../core/program.js';
import { Target, DoubleTarget, bindScreen, BLEND } from '../core/gl.js';
import { PRECISION, CONSTANTS, COLOR, VERT_FULLSCREEN } from '../shaders/common.js';

const P = PRECISION + CONSTANTS;

/* ═══ shaders ═════════════════════════════════════════════════════ */

const FRAG_ADVECT = /* glsl */`
${P}
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uVelocity;
uniform sampler2D uSource;
uniform vec2 uTexel;
uniform float uDt, uDissipation;

void main() {
  // Backward trace: where was the stuff that is here now, one step ago?
  // Because we always sample the *previous* field, this can never blow
  // up, no matter how large dt or how fast the flow.
  vec2 coord = vUv - uDt * texture(uVelocity, vUv).xy * uTexel;
  vec4 result = texture(uSource, coord);
  outColor = result / (1.0 + uDissipation * uDt);
}
`;

const FRAG_DIVERGENCE = /* glsl */`
${P}
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uVelocity;
uniform vec2 uTexel;

void main() {
  float L = texture(uVelocity, vUv - vec2(uTexel.x, 0.0)).x;
  float R = texture(uVelocity, vUv + vec2(uTexel.x, 0.0)).x;
  float B = texture(uVelocity, vUv - vec2(0.0, uTexel.y)).y;
  float T = texture(uVelocity, vUv + vec2(0.0, uTexel.y)).y;

  // No-slip walls: mirror the normal component across the boundary.
  vec2 C = texture(uVelocity, vUv).xy;
  if (vUv.x - uTexel.x < 0.0) L = -C.x;
  if (vUv.x + uTexel.x > 1.0) R = -C.x;
  if (vUv.y - uTexel.y < 0.0) B = -C.y;
  if (vUv.y + uTexel.y > 1.0) T = -C.y;

  outColor = vec4(0.5 * (R - L + T - B), 0.0, 0.0, 1.0);
}
`;

const FRAG_CURL = /* glsl */`
${P}
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uVelocity;
uniform vec2 uTexel;

void main() {
  float L = texture(uVelocity, vUv - vec2(uTexel.x, 0.0)).y;
  float R = texture(uVelocity, vUv + vec2(uTexel.x, 0.0)).y;
  float B = texture(uVelocity, vUv - vec2(0.0, uTexel.y)).x;
  float T = texture(uVelocity, vUv + vec2(0.0, uTexel.y)).x;
  outColor = vec4(0.5 * ((R - L) - (T - B)), 0.0, 0.0, 1.0);
}
`;

const FRAG_VORTICITY = /* glsl */`
${P}
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uVelocity;
uniform sampler2D uCurl;
uniform vec2 uTexel;
uniform float uCurlStrength, uDt;

void main() {
  float L = texture(uCurl, vUv - vec2(uTexel.x, 0.0)).x;
  float R = texture(uCurl, vUv + vec2(uTexel.x, 0.0)).x;
  float B = texture(uCurl, vUv - vec2(0.0, uTexel.y)).x;
  float T = texture(uCurl, vUv + vec2(0.0, uTexel.y)).x;
  float C = texture(uCurl, vUv).x;

  // Push energy back *into* each vortex, along the gradient of |curl|.
  // Advection is diffusive; this is the counterweight that keeps small
  // eddies alive instead of dissolving into a smooth blur.
  vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
  force /= length(force) + 1e-4;
  force *= uCurlStrength * C;
  force.y *= -1.0;

  vec2 vel = texture(uVelocity, vUv).xy + force * uDt;
  outColor = vec4(clamp(vel, -1500.0, 1500.0), 0.0, 1.0);
}
`;

const FRAG_PRESSURE = /* glsl */`
${P}
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uPressure;
uniform sampler2D uDivergence;
uniform vec2 uTexel;

void main() {
  float L = texture(uPressure, vUv - vec2(uTexel.x, 0.0)).x;
  float R = texture(uPressure, vUv + vec2(uTexel.x, 0.0)).x;
  float B = texture(uPressure, vUv - vec2(0.0, uTexel.y)).x;
  float T = texture(uPressure, vUv + vec2(0.0, uTexel.y)).x;
  float div = texture(uDivergence, vUv).x;
  // One Jacobi relaxation of ∇²p = ∇·u
  outColor = vec4((L + R + B + T - div) * 0.25, 0.0, 0.0, 1.0);
}
`;

const FRAG_GRADIENT = /* glsl */`
${P}
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uPressure;
uniform sampler2D uVelocity;
uniform vec2 uTexel;

void main() {
  float L = texture(uPressure, vUv - vec2(uTexel.x, 0.0)).x;
  float R = texture(uPressure, vUv + vec2(uTexel.x, 0.0)).x;
  float B = texture(uPressure, vUv - vec2(0.0, uTexel.y)).x;
  float T = texture(uPressure, vUv + vec2(0.0, uTexel.y)).x;

  // u ← u − ∇p leaves exactly the divergence-free part of the field.
  vec2 vel = texture(uVelocity, vUv).xy - vec2(R - L, T - B);
  outColor = vec4(vel, 0.0, 1.0);
}
`;

const FRAG_DECAY = /* glsl */`
${P}
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uSrc;
uniform float uValue;
void main() { outColor = texture(uSrc, vUv) * uValue; }
`;

const FRAG_SPLAT = /* glsl */`
${P}
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uTarget;
uniform vec2 uPoint;
uniform vec3 uValue;
uniform float uRadius, uAspect;

void main() {
  vec2 d = vUv - uPoint;
  d.x *= uAspect;
  vec3 splat = exp(-dot(d, d) / uRadius) * uValue;
  outColor = vec4(texture(uTarget, vUv).xyz + splat, 1.0);
}
`;

const FRAG_DISPLAY = /* glsl */`
${P}
${COLOR}
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uDye;
uniform sampler2D uVelocity;
uniform vec2 uTexel;
uniform float uExposure, uShading, uVelocityMix;

void main() {
  vec3 c = texture(uDye, vUv).rgb;

  if (uShading > 0.5) {
    // Fake a normal from the dye field's own gradient: dense regions
    // read as thick, lit volume rather than flat colour.
    vec3 lc = texture(uDye, vUv - vec2(uTexel.x, 0.0)).rgb;
    vec3 rc = texture(uDye, vUv + vec2(uTexel.x, 0.0)).rgb;
    vec3 bc = texture(uDye, vUv - vec2(0.0, uTexel.y)).rgb;
    vec3 tc = texture(uDye, vUv + vec2(0.0, uTexel.y)).rgb;

    float dx = length(rc) - length(lc);
    float dy = length(tc) - length(bc);
    vec3 n = normalize(vec3(dx, dy, length(uTexel) * 4.0));
    float diffuse = clamp(dot(n, normalize(vec3(-0.4, 0.5, 0.75))) + 0.72, 0.66, 1.25);
    c *= diffuse;
  }

  if (uVelocityMix > 0.0) {
    vec2 v = texture(uVelocity, vUv).xy;
    float s = clamp(length(v) * 0.0035, 0.0, 1.0);
    c += cosPalette(s, vec3(0.1), vec3(0.35), vec3(1.0), vec3(0.0, 0.33, 0.67)) * s * uVelocityMix;
  }

  vec3 col = acesFilm(c * uExposure);
  vec2 q = vUv - 0.5;
  col *= 1.0 - dot(q, q) * 0.4;
  outColor = vec4(dither(col, gl_FragCoord.xy), 1.0);
}
`;

/* ═══ scene ═══════════════════════════════════════════════════════ */

/**
 * Canonical agent space → this scene's UV. Asymmetric on purpose: the
 * agent's wander volume is wider than it is tall, and the canvas is
 * wider than it is tall by a different ratio, so one factor per axis
 * is what makes the creature use the whole frame.
 */
const AGENT_U = 0.24;
const AGENT_V = 0.36;

export default {
  id: 'fluid',
  index: '02',
  agentFlatten: 1,
  title: 'Navier–Stokes 流體',
  tech: 'semi-Lagrangian advection · Jacobi pressure projection',
  desc: '每幀 8–40 道 pass：對流、渦度強化、求解壓力泊松方程、投影成無散度場。',
  glyph: '≈',
  hue: 285,

  params: [
    { group: '解算器' },
    { id: 'simRes', type: 'select', label: '模擬解析度', value: '192',
      options: [
        { value: '128', label: '128' },
        { value: '192', label: '192' },
        { value: '256', label: '256' },
        { value: '384', label: '384' },
      ] },
    { id: 'dyeRes', type: 'select', label: '染料解析度', value: '768',
      options: [
        { value: '384', label: '384' },
        { value: '768', label: '768' },
        { value: '1024', label: '1024' },
      ] },
    { id: 'iterations', type: 'slider', label: '壓力迭代', min: 4, max: 40, step: 1, value: 22 },
    { id: 'curl', type: 'slider', label: '渦度強化', min: 0, max: 50, step: 0.5, value: 26 },
    { id: 'velDiss', type: 'slider', label: '速度衰減', min: 0, max: 4, step: 0.01, value: 0.20 },
    { id: 'dyeDiss', type: 'slider', label: '染料衰減', min: 0, max: 4, step: 0.01, value: 0.28 },

    { group: '注入' },
    { id: 'radius', type: 'slider', label: '筆刷半徑', min: 0.05, max: 1.2, step: 0.01, value: 0.32 },
    { id: 'force', type: 'slider', label: '力道', min: 500, max: 12000, step: 50, value: 5200 },
    { id: 'auto', type: 'switch', label: '無人時自動攪動', value: true },

    { group: '游者' },
    { id: 'agent', type: 'switch', label: '放入游者', value: true },
    { id: 'agentMode', type: 'select', label: '行為', value: 'follow',
      options: [
        { value: 'wander', label: '漫遊' },
        { value: 'follow', label: '跟隨' },
        { value: 'flee', label: '迴避' },
      ] },
    { id: 'agentSpeed', type: 'slider', label: '泳速', min: 0.2, max: 3, step: 0.01, value: 1.15 },
    { id: 'wake', type: 'slider', label: '尾流強度', min: 0, max: 3, step: 0.01, value: 1.0 },

    { group: '渲染' },
    { id: 'shading', type: 'switch', label: '體積打光', value: true },
    { id: 'velocityMix', type: 'slider', label: '速度場疊加', min: 0, max: 1, step: 0.01, value: 0.18 },
    { id: 'exposure', type: 'slider', label: '曝光', min: 0.2, max: 3, step: 0.01, value: 1.05 },
    { id: 'hint', type: 'hint', text: '在畫布上拖曳即可攪動流體。' },
  ],

  init(ctx) { return new FluidScene(ctx); },
};

class FluidScene {
  constructor(ctx) {
    this.ctx = ctx;
    const { gl } = ctx;
    const N = gl.NEAREST, L = gl.LINEAR;

    this.programs = {
      advect:    new Program(gl, VERT_FULLSCREEN, FRAG_ADVECT,     { name: 'fluid/advect' }),
      divergence:new Program(gl, VERT_FULLSCREEN, FRAG_DIVERGENCE, { name: 'fluid/divergence' }),
      curl:      new Program(gl, VERT_FULLSCREEN, FRAG_CURL,       { name: 'fluid/curl' }),
      vorticity: new Program(gl, VERT_FULLSCREEN, FRAG_VORTICITY,  { name: 'fluid/vorticity' }),
      pressure:  new Program(gl, VERT_FULLSCREEN, FRAG_PRESSURE,   { name: 'fluid/pressure' }),
      gradient:  new Program(gl, VERT_FULLSCREEN, FRAG_GRADIENT,   { name: 'fluid/gradient' }),
      decay:     new Program(gl, VERT_FULLSCREEN, FRAG_DECAY,      { name: 'fluid/decay' }),
      splat:     new Program(gl, VERT_FULLSCREEN, FRAG_SPLAT,      { name: 'fluid/splat' }),
      display:   new Program(gl, VERT_FULLSCREEN, FRAG_DISPLAY,    { name: 'fluid/display' }),
    };

    // Velocity needs bilinear filtering — the semi-Lagrangian trace
    // lands between texels on essentially every pixel.
    this.velocity   = new DoubleTarget(gl, { width: 2, height: 2, format: 'rg16f',   filter: L });
    this.dye        = new DoubleTarget(gl, { width: 2, height: 2, format: 'rgba16f', filter: L });
    this.pressure   = new DoubleTarget(gl, { width: 2, height: 2, format: 'r16f',    filter: N });
    this.divergence = new Target(gl,       { width: 2, height: 2, format: 'r16f',    filter: N });
    this.curl       = new Target(gl,       { width: 2, height: 2, format: 'r16f',    filter: N });

    this.simRes = 0;
    this.dyeRes = 0;
    this.width = 2;
    this.height = 2;
    this.aspect = 1;
    this.hue = Math.random();
    this.autoPhase = Math.random() * 100;
    this.lastUserInput = -99;
    this.splatCount = 0;
  }

  resize(width, height) {
    this.width = width;
    this.height = height;
    this.aspect = width / Math.max(height, 1);
    this._resizeSim(this.simRes || 192, this.dyeRes || 768, true);
  }

  /** Grids are sized to the viewport's aspect so cells stay square. */
  _resizeSim(sim, dye, force = false) {
    if (!force && sim === this.simRes && dye === this.dyeRes) return;
    this.simRes = sim;
    this.dyeRes = dye;

    const dims = (base) => {
      const a = this.aspect;
      return a >= 1
        ? { w: Math.round(base * a), h: base }
        : { w: base, h: Math.round(base / a) };
    };

    const s = dims(sim);
    const d = dims(dye);

    this.velocity.resize(s.w, s.h);
    this.pressure.resize(s.w, s.h);
    this.divergence.resize(s.w, s.h);
    this.curl.resize(s.w, s.h);
    this.dye.resize(d.w, d.h);

    this.velocity.clear(0, 0, 0, 1);
    this.pressure.clear(0, 0, 0, 1);
    this.dye.clear(0, 0, 0, 1);
    this._seedSplats();
  }

  reset() {
    this.velocity.clear(0, 0, 0, 1);
    this.pressure.clear(0, 0, 0, 1);
    this.dye.clear(0, 0, 0, 1);
    this.hue = Math.random();
    this._seedSplats();
  }

  /** A few splats at startup so there is something to look at instantly. */
  _seedSplats() {
    for (let i = 0; i < 7; i++) {
      const a = (i / 7) * Math.PI * 2;
      this._splat(
        0.5 + Math.cos(a) * 0.22,
        0.5 + Math.sin(a) * 0.22,
        -Math.sin(a) * 620,
        Math.cos(a) * 620,
        this._burstColor(1.6),
        0.36,
      );
    }
  }

  /**
   * A continuous stream must drift its hue *slowly*. Advancing by the
   * golden ratio every frame — the right choice for discrete splats —
   * dumps every hue into the same place, and the dye averages to white.
   */
  _streamColor(scale = 1, dt = 1 / 60) {
    this.hue = (this.hue + dt * 0.055) % 1;
    const c = hsv(this.hue, 0.85, 1.0);
    return [c[0] * scale, c[1] * scale, c[2] * scale];
  }

  /** Maximally-separated hues, for splats far apart in space or time. */
  _burstColor(scale = 1) {
    this.hue = (this.hue + 0.618033988749895) % 1;
    const c = hsv(this.hue, 0.82, 1.0);
    return [c[0] * scale, c[1] * scale, c[2] * scale];
  }

  /** One velocity splat + one dye splat, both Gaussian. */
  _splat(x, y, dx, dy, color, radius) {
    const { gl, tri } = this.ctx;
    const prog = this.programs.splat;
    // GL's texture origin is bottom-left; the pointer's is top-left.
    const point = [x, 1 - y];
    const r = radius * radius * 0.01;

    BLEND.none(gl);
    this.velocity.write.bind();
    prog.use({
      uTarget: this.velocity.read.texture,
      uPoint: point,
      uValue: [dx, -dy, 0],
      uRadius: r,
      uAspect: this.aspect,
    });
    tri.draw();
    this.velocity.swap();

    this.dye.write.bind();
    prog.use({
      uTarget: this.dye.read.texture,
      uPoint: point,
      uValue: color,
      uRadius: r,
      uAspect: this.aspect,
    });
    tri.draw();
    this.dye.swap();

    this.splatCount++;
  }

  /**
   * One automatic stirrer. Velocity is taken from the path's own finite
   * difference, so the injected force always points where the stirrer is
   * actually going — exactly what a dragging finger would produce.
   */
  _stir(t, phase, state, dt) {
    const path = (u) => [
      0.5 + 0.34 * Math.sin(u * 0.62 + phase) * Math.cos(u * 0.21 + phase),
      0.5 + 0.33 * Math.sin(u * 0.47 + phase * 1.7 + 1.3),
    ];
    const [x, y] = path(t);
    const [px, py] = path(t - 0.06);
    this._splat(
      x, y,
      (x - px) * state.force * 1.1,
      (y - py) * state.force * 1.1,
      this._streamColor(0.20, dt),
      state.radius * 0.8,
    );
  }

  /**
   * The swimmer as a stirring body. The orb drives the wake; its shed
   * droplets each stir a little on their own, which is what widens the
   * disturbance into something the vorticity term can curl into eddies
   * rather than a single thin thread.
   */
  _swim(state, dt) {
    const agent = this.ctx.agent;
    const sp = Math.max(agent.speed, 1e-3);
    const mag = state.wake * 760 * (sp / 1.15);

    // Body-space velocity → UV-space velocity, then into force units.
    const dx = (agent.vel[0] / sp) * mag * AGENT_U * 2.6;
    const dy = -(agent.vel[1] / sp) * mag * AGENT_V * 2.6;

    // The orb, then at most three droplets — enough to broaden the wake
    // without spending a full splat pair on every speck.
    const taps = Math.min(agent.live, 4);
    for (let k = 0; k < taps; k++) {
      const i = k * 3;
      const u = 0.5 + agent.nodes[i] * AGENT_U;
      const v = 0.5 + agent.nodes[i + 1] * AGENT_V;
      const falloff = k === 0 ? 1 : 0.45 * (1 - (k - 1) / 3);
      this._splat(
        u, 1 - v,
        dx * falloff, dy * falloff,
        this._streamColor(0.16 * falloff + 0.04, dt),
        state.radius * (0.55 + 0.2 * falloff),
      );
    }
  }

  /* ── frame ────────────────────────────────────────────────────── */

  frame({ state, clock, pointer }) {
    const { gl, tri } = this.ctx;
    this._resizeSim(Number(state.simRes), Number(state.dyeRes));

    const dt = Math.min(clock.dt, 1 / 30);

    /* input ─────────────────────────────────────────────────────── */
    if (pointer.down && pointer.moved) {
      this.lastUserInput = clock.wall;
      this._splat(
        pointer.x, pointer.y,
        pointer.dx * state.force,
        pointer.dy * state.force,
        this._streamColor(0.16 + Math.min(pointer.speed * 0.22, 0.30), dt),
        state.radius,
      );
    }

    // The pointer aims the swimmer; the swimmer stirs the fluid. The
    // user never touches the fluid directly in this mode — they lead an
    // animal around and the wake is the drawing.
    if (state.agent !== false && dt > 0) {
      if (pointer.active) {
        this.ctx.agent.aim(
          (pointer.x - 0.5) / AGENT_U,
          (0.5 - pointer.y) / AGENT_V,
          0,
        );
      }
      this._swim(state, dt);
    }

    // Idle attractors: two lissajous stirrers on mutually irrational
    // periods, so an unattended canvas is never static. Redundant once
    // the swimmer is in, hence the extra guard.
    if (state.auto && state.agent === false
        && clock.wall - this.lastUserInput > 1.6 && dt > 0) {
      this.autoPhase += dt;
      this._stir(this.autoPhase, 0.0, state, dt);
      this._stir(this.autoPhase * 0.71 + 17.3, 2.4, state, dt);
    }

    if (dt <= 0) { this._display(state); return; }

    const texel = this.velocity.texelSize;
    BLEND.none(gl);

    /* 1 ── vorticity confinement */
    if (state.curl > 0) {
      this.curl.bind();
      this.programs.curl.use({ uVelocity: this.velocity.read.texture, uTexel: texel });
      tri.draw();

      this.velocity.write.bind();
      this.programs.vorticity.use({
        uVelocity: this.velocity.read.texture,
        uCurl: this.curl.texture,
        uTexel: texel,
        uCurlStrength: state.curl,
        uDt: dt,
      });
      tri.draw();
      this.velocity.swap();
    }

    /* 2 ── projection: ∇·u → p → u − ∇p */
    this.divergence.bind();
    this.programs.divergence.use({ uVelocity: this.velocity.read.texture, uTexel: texel });
    tri.draw();

    // Warm-starting from the previous frame's pressure, decayed, means
    // far fewer Jacobi iterations are needed for the same convergence.
    this.pressure.write.bind();
    this.programs.decay.use({ uSrc: this.pressure.read.texture, uValue: 0.8 });
    tri.draw();
    this.pressure.swap();

    const iterations = Math.round(state.iterations);
    for (let i = 0; i < iterations; i++) {
      this.pressure.write.bind();
      this.programs.pressure.use({
        uPressure: this.pressure.read.texture,
        uDivergence: this.divergence.texture,
        uTexel: texel,
      });
      tri.draw();
      this.pressure.swap();
    }

    this.velocity.write.bind();
    this.programs.gradient.use({
      uPressure: this.pressure.read.texture,
      uVelocity: this.velocity.read.texture,
      uTexel: texel,
    });
    tri.draw();
    this.velocity.swap();

    /* 3 ── advect velocity by itself, then dye by velocity */
    this.velocity.write.bind();
    this.programs.advect.use({
      uVelocity: this.velocity.read.texture,
      uSource: this.velocity.read.texture,
      uTexel: texel,
      uDt: dt,
      uDissipation: state.velDiss,
    });
    tri.draw();
    this.velocity.swap();

    this.dye.write.bind();
    this.programs.advect.use({
      uVelocity: this.velocity.read.texture,
      uSource: this.dye.read.texture,
      // The dye grid is finer than the velocity grid; the trace must be
      // expressed in *dye* texels or the field shears.
      uTexel: this.dye.texelSize,
      uDt: dt,
      uDissipation: state.dyeDiss,
    });
    tri.draw();
    this.dye.swap();

    this._display(state);
  }

  _display(state) {
    const { gl, tri } = this.ctx;
    BLEND.none(gl);
    bindScreen(gl, this.width, this.height);
    this.programs.display.use({
      uDye: this.dye.read.texture,
      uVelocity: this.velocity.read.texture,
      uTexel: this.dye.texelSize,
      uExposure: state.exposure,
      uShading: state.shading ? 1 : 0,
      uVelocityMix: state.velocityMix,
    });
    tri.draw();
  }

  readout(state) {
    return {
      '速度網格': `${this.velocity.width}×${this.velocity.height}`,
      '染料網格': `${this.dye.width}×${this.dye.height}`,
      '每幀 pass': `${8 + Math.round(state.iterations)}`,
      '游者泳速': this.ctx.agent.speed.toFixed(2),
      '累計注入': this.splatCount.toLocaleString('en-US'),
    };
  }

  dispose() {
    for (const p of Object.values(this.programs)) p.dispose();
    this.velocity.dispose();
    this.dye.dispose();
    this.pressure.dispose();
    this.divergence.dispose();
    this.curl.dispose();
  }
}

/** HSV → RGB, kept in JS so splat colours can be picked on the CPU. */
function hsv(h, s, v) {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  switch (i % 6) {
    case 0: return [v, t, p];
    case 1: return [q, v, p];
    case 2: return [p, v, t];
    case 3: return [p, q, v];
    case 4: return [t, p, v];
    default: return [v, p, q];
  }
}
