/* ── scenes/laser.js ─────────────────────────────────────────────────
   Two beams from the cat's eyes.

   Drawn, not marched. A beam is a millimetre wide and perfectly
   straight, which is the worst possible shape for a sphere tracer —
   every ray that misses it still pays for the misses. As a pair of
   camera-facing quads it is twelve vertices.

   Two beams are drawn because there are two eyes, but everything
   physical uses one axis down the middle: they are a few centimetres
   apart on a metre-tall animal, and no impulse worth computing can tell
   them apart.

   Occlusion comes free from the scene's depth channel — the same alpha
   the impact flares read — so a beam is cut off exactly where it meets
   a sphere rather than being drawn over it.
   ------------------------------------------------------------------ */

import { Program } from '../core/program.js';
import { PRECISION } from '../shaders/common.js';

/** How long a shot lives, in seconds. A zap, not a sustained ray. */
const LIFE = 0.22;
/** Fallback length, if the caller does not say where the beam stops. */
const REACH = 40.0;
/** Half-width in NDC, so the beam holds its thickness at any range. */
const WIDTH = 0.0075;

const VERT_LASER = /* glsl */`
${PRECISION}

uniform vec3 uEyeA, uEyeB, uDir;
uniform vec3 uCamPos, uRight, uUp, uFwd;
uniform float uFocal, uAspect;
uniform float uReach, uWidth;

out float vSide;    // -1..1 across the beam
out float vAlong;   // 0 at the eye, 1 at the far end
out float vDepth;   // distance from the eye, for the depth test

/** The scene's projection, as the marcher builds its rays. */
vec4 project(vec3 p) {
  vec3 rel = p - uCamPos;
  vec3 view = vec3(dot(rel, uRight), dot(rel, uUp), dot(rel, uFwd));
  return vec4(view.x * uFocal / uAspect, view.y * uFocal, 0.0, view.z);
}

void main() {
  int beam = gl_VertexID / 6;
  int corner = gl_VertexID % 6;

  // Two triangles: (0,0) (1,0) (0,1) / (1,0) (1,1) (0,1)
  float along = (corner == 1 || corner == 3 || corner == 4) ? 1.0 : 0.0;
  float side = (corner == 2 || corner == 4 || corner == 5) ? 1.0 : -1.0;

  vec3 origin = beam == 0 ? uEyeA : uEyeB;

  /* Stop the far end short of the eye plane. The beam is aimed by the
     same mouse that aims the camera, so it usually recedes — but the cat
     can be turned to face the lens, and an endpoint behind the eye
     projects to garbage that smears across the screen. */
  float reach = uReach;
  float z0 = dot(origin - uCamPos, uFwd);
  float dz = dot(uDir, uFwd);
  if (z0 + dz * reach < 0.05) reach = dz < -1e-4 ? max(0.0, (0.05 - z0) / dz) : reach;

  vec4 ca = project(origin);
  vec4 cb = project(origin + uDir * reach);

  /* Widen in *screen* space, not world space.

     The obvious billboard — cross(beamDir, toEye) — is exactly wrong
     here. This beam is fired down the crosshair, so it points very
     nearly along the view axis, and that cross product collapses to
     zero: the quad ends up with no width at all and the shot is
     invisible. Expanding perpendicular to the beam's *projected*
     direction has no such degenerate case, and gives a constant pixel
     width, which is what a laser looks like anyway. */
  vec2 na = ca.xy / max(ca.w, 1e-3);
  vec2 nb = cb.xy / max(cb.w, 1e-3);

  vec2 run = (nb - na) * vec2(uAspect, 1.0);   // into square pixels
  float len = length(run);
  vec2 perp = len > 1e-5 ? vec2(-run.y, run.x) / len : vec2(1.0, 0.0);
  perp /= vec2(uAspect, 1.0);                  // and back

  vec4 c = along < 0.5 ? ca : cb;
  vec2 n = c.xy / max(c.w, 1e-3) + perp * side * uWidth;

  vSide = side;
  vAlong = along;
  vDepth = along < 0.5 ? length(origin - uCamPos) : length(origin + uDir * reach - uCamPos);

  gl_Position = vec4(n * c.w, 0.0, c.w);
}
`;

const FRAG_LASER = /* glsl */`
${PRECISION}

in float vSide;
in float vAlong;
in float vDepth;
out vec4 outColor;

uniform sampler2D uScene;
uniform vec2 uResolution;
uniform float uFade;      // 1 at the instant of firing, 0 when spent

void main() {
  // The scene publishes ray distance in alpha; anything past the surface
  // in front of it is not there. This is what stops the beam at a sphere
  // instead of painting it across one.
  float sceneDepth = texture(uScene, gl_FragCoord.xy / uResolution).a;
  if (vDepth > sceneDepth) discard;

  // A hot white core inside a red bloom.
  float across = 1.0 - abs(vSide);
  float core = pow(across, 14.0);
  float glow = pow(across, 2.0);

  // Fades along its length, and dies from the muzzle end last.
  float travel = 1.0 - vAlong * 0.55;

  vec3 red = vec3(1.0, 0.10, 0.06);
  vec3 col = red * glow * 2.2 + vec3(1.0, 0.72, 0.62) * core * 3.0;
  outColor = vec4(col * travel * uFade, 1.0);
}
`;

/**
 * A shot. Fired once, lives briefly, then is gone.
 *
 * The state is deliberately flat and single-slot: a second shot while
 * one is in flight replaces it. Two overlapping beams from the same
 * pair of eyes would look like one brighter beam anyway.
 */
export class Laser {
  constructor(gl) {
    this.gl = gl;
    this.program = new Program(gl, VERT_LASER, FRAG_LASER, { name: 'march/laser' });

    this.age = LIFE;                       // starts spent
    this.eyeA = new Float32Array(3);
    this.eyeB = new Float32Array(3);
    /** Midpoint of the two eyes: the one axis everything physical uses. */
    this.origin = new Float32Array(3);
    this.dir = new Float32Array([0, 0, 1]);
    /** Where this shot stops. Measured by the caller against the field,
        because a depth test hides a beam without ending it. */
    this.reach = REACH;
  }

  get active() { return this.age < LIFE; }
  /** 1 at the instant of firing, 0 when spent. */
  get fade() { return Math.max(0, 1 - this.age / LIFE); }

  fire(eyeA, eyeB, dir, reach = REACH) {
    this.eyeA.set(eyeA);
    this.eyeB.set(eyeB);
    for (let i = 0; i < 3; i++) this.origin[i] = (eyeA[i] + eyeB[i]) * 0.5;
    this.dir.set(dir);
    this.reach = reach;
    this.age = 0;
  }

  update(dt) {
    if (this.age < LIFE) this.age = Math.min(LIFE, this.age + dt);
  }

  draw(camera, sceneTexture, resolution) {
    if (!this.active) return;

    this.program.use({
      uEyeA: this.eyeA,
      uEyeB: this.eyeB,
      uDir: this.dir,
      uCamPos: camera.pos,
      uRight: camera.right,
      uUp: camera.up,
      uFwd: camera.fwd,
      uFocal: camera.focal,
      uAspect: camera.aspect,
      uReach: this.reach,
      uWidth: WIDTH,
      uScene: sceneTexture,
      uResolution: resolution,
      uFade: this.fade,
    });
  }

  dispose() { this.program.dispose(); }
}
