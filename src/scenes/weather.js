/* ── scenes/weather.js ───────────────────────────────────────────────
   Three weathers, one of which is the scene as it already was.

   Clear, rain, snow. A three-way choice rather than three sliders,
   because they are not three amounts of the same thing: rain wets the
   ground and snow covers it, and a control that could give you half of
   each would be a control nobody could answer a question with.

   What is here is the *field* — how much snow is lying at a point, how
   wet the ground is — and the precipitation itself. Everything that
   consumes them lives where it always did: the soil colour in the
   marcher, the blades in the grass, the sowing in the flowers. That
   split is the whole design. A weather that owned its own copy of the
   ground would be a second answer to where the ground is, which is the
   one thing this scene has spent its whole life refusing to have.

   ── what snow deliberately does not do ───────────────────────────
   The first version raised the terrain: snow is a layer, a layer has a
   thickness, and the height field is right there. It came out costing
   the one property the ground is built around. `terrainAt` is marched
   against a slope bound that is a closed-form sum over the wave table,
   and snow that varies with slope adds a gradient the bound knows
   nothing about — so either the bound is loosened for everyone (long
   shadows get slower everywhere, in clear weather too) or the march
   overshoots and the hills get holes in them.

   And it bought nothing. Snow lies a few centimetres deep on a field
   whose hills are metres tall and which is looked at from fifteen
   metres up; the lift is invisible at every distance it would be paid
   for. What actually reads as snow is that the ground is white, the
   grass is buried, and there is something falling. So snow is a
   *coverage*, not a height, and the slope bound is untouched.

   The shoreline argument survives that change intact, and is the reason
   `snowCover` still asks where the water is: snow that ignored the lake
   would paint a white rim floating on the first half-metre of it. It
   fades out as the ground approaches the surface instead, which is also
   what drifted snow does at a water's edge.
   ------------------------------------------------------------------ */

import { Program } from '../core/program.js';
import { PRECISION, HASH } from '../shaders/common.js';
import { WIND_GLSL } from './wind.js';

/* ── the control ──────────────────────────────────────────────────
   Each mode is the pair of numbers every consumer actually reads. The
   names are the only place the modes exist as names; past this table
   the rest of the scene knows nothing but "how wet" and "how white". */

/* `overcast` is the third number and it is the sky's, not the ground's.
   The other two say what has landed; this one says what is in the way of
   the light — and it has to be its own figure rather than max(rain,
   snow), because snow falls out of a lower, more solid deck than rain
   does and neither of them is a hundred per cent of anything. It is
   read by one function, `sky()`, and a shader that never receives it
   gets zero, which is the fair-weather sky this scene has always had. */
/** @type {Record<string, {rain: number, snow: number, drops: number,
    overcast: number}>} */
export const WEATHER = {
  clear: { rain: 0.0, snow: 0.0, drops: 0,    overcast: 0.00 },
  rain:  { rain: 1.0, snow: 0.0, drops: 4400, overcast: 0.88 },
  snow:  { rain: 0.0, snow: 1.0, drops: 2000, overcast: 0.95 },
};

/** What the mode control offers, in the order it offers it. */
export const WEATHER_MODES = Object.keys(WEATHER);

/** The mode's numbers, falling back to fair weather for an unknown name
    — which is what a stale URL carrying a mode that no longer exists
    should get, rather than a crash or a blank sky. */
export function weatherOf(mode) {
  return WEATHER[mode] || WEATHER.clear;
}

/* ── how snow lies ────────────────────────────────────────────────
   Mottling, as a handful of sines, generated into both languages from
   one table — the same arrangement, for the same reason, as the hills.

   Sines rather than noise here for a smaller reason than the terrain's:
   nothing marches this field, so its gradient is nobody's business. What
   is still true is that two hand-written noise functions agreeing across
   JS and GLSL is real work, and three sines agreeing is arithmetic —
   and the two sides *do* have to agree, because the shader decides
   whether a patch of ground looks snowed under and the flower sowing
   decides whether a flower there was buried. Disagreement puts blooms
   on top of a snowdrift. */
const PATCH = [
  // kx, kz, phase, weight
  [ 0.310,  0.130, 1.70, 0.55],
  [-0.190,  0.440, -0.90, 0.30],
  [ 0.630, -0.370, 3.10, 0.15],
];

/** Where the ground stops holding snow, as a slope. Roughly 25° to 50°:
    below the first everything sticks, above the second nothing does. */
const HOLD_LO = 0.45;
const HOLD_HI = 1.15;

/** How far above the water snow starts lying, in metres. */
const SHORE = 0.50;

const glslPatch = PATCH.map(([kx, kz, ph, w]) =>
  `\n  mottle += ${w.toFixed(4)} * sin(dot(p, vec2(${kx.toFixed(4)}, ${kz.toFixed(4)})) + ${ph.toFixed(4)});`
).join('');

/**
 * How wet the ground is and how much snow is lying on it.
 *
 * Include *after* the terrain block: `snowCover` asks the ground for its
 * slope and asks the water where it is, and both of those are the
 * terrain's to answer.
 *
 * Two uniforms, both zero in fair weather, which is what makes clear
 * weather cost nothing anywhere rather than cost a branch everywhere.
 *
 * (No back-ticks below: this lives inside a JS template literal and one
 * would end it.)
 */
export const WEATHER_GLSL = /* glsl */`
#define SNOW_HOLD_LO ${HOLD_LO.toFixed(4)}
#define SNOW_HOLD_HI ${HOLD_HI.toFixed(4)}
#define SNOW_SHORE ${SHORE.toFixed(4)}

/** How hard it is raining, 0..1. Darkens and polishes what it lands on. */
uniform float uRain;
/** How much snow has settled, 0..1, before the ground has its say. */
uniform float uSnow;

/**
 * How much snow is lying at a world xz, 0..1.
 *
 * Three factors, and each is there because leaving it out looks wrong in
 * a specific way. Without the slope term the hills are iced rather than
 * snowed on, every face the same white regardless of which way it
 * leans. Without the shore term a white rim floats out over the first
 * half metre of the lake, because the ground under it is still ground.
 * Without the mottle it is a coat of paint: real fallen snow is thinner
 * where the wind scoured it and banked where it did not.
 */
float snowCover(vec2 p) {
  if (uSnow <= 0.0) return 0.0;

  vec2 g;
  terrainAt(p, g);
  float hold = 1.0 - smoothstep(SNOW_HOLD_LO, SNOW_HOLD_HI, length(g));
  if (hold <= 0.0) return 0.0;

  // Height above the water, so the cover thins out as it reaches it.
  float dry = smoothstep(0.0, SNOW_SHORE, -waterDepth(p));

  float mottle = 0.0;${glslPatch}
  return clamp(uSnow * hold * dry * (0.72 + 0.42 * mottle), 0.0, 1.0);
}

/**
 * What weather does to a surface it has fallen on.
 *
 * One function so that the soil, the blades and the leaves cannot each
 * invent their own idea of wet and white. Rain darkens and polishes;
 * snow whitens and roughens, and wins where it is lying, since a surface
 * cannot be both.
 */
vec3 weatherSurface(vec3 albedo, float snow, inout float rough) {
  if (uRain > 0.0) {
    // Wet is darker and smoother — the two halves of why a wet path
    // reads as wet, and neither works without the other.
    albedo *= mix(1.0, 0.58, uRain);
    rough = mix(rough, 0.34, uRain * 0.8);
  }
  if (snow > 0.0) {
    /* Cooler and darker than white, and both matter. The sun in this
       scene is amber and the tonemap is generous, so snow authored at
       the brightness it has in life comes out of the pipe as cream and
       reads as sand — the first version photographed as a dune field
       with trees on it. Snow is bright because it is *lit*, not because
       its albedo is near one; giving it a blue-grey albedo and letting
       the sun do the rest puts it back.

       Low enough not to clip, which is the other half. The direct term
       carries a gain of 2.3 before the tonemap sees it, so snow authored
       much above a third saturates every lit face to the same flat
       value — and a snowfield with no shading gradient across it has
       nothing left to say it is snow rather than paper. Kept under that,
       the lit faces stay warm from the amber sun and the shaded ones
       fall to the blue ambient, which is the contrast the eye is
       actually reading when it calls something snow. */
    albedo = mix(albedo, vec3(0.34, 0.39, 0.52), snow);
    rough = mix(rough, 0.82, snow);
  }
  return albedo;
}
`;

/* ── the same thing, in JS ────────────────────────────────────────
   Read by the flower sowing, which has to know whether a bloom is under
   a drift before it spends an instance on it. Generated from the same
   table as the GLSL above.                                            */

/**
 * How much snow is lying at a world xz — the JS side of `snowCover`.
 *
 * @param {number} x
 * @param {number} z
 * @param {number} hills      the hill amplitude
 * @param {number} surfaceY   the water level, from `waterSurfaceY`
 * @param {number} snow       the weather's snow amount, 0..1
 * @param {(x:number,z:number,hills:number,grad?:number[])=>number} height
 *        the terrain height function, passed in rather than imported so
 *        this file does not have to know that terrain.js exists
 * @param {number[]} grad     scratch array, to keep this allocation-free
 *        inside a sowing loop
 */
export function snowCoverAt(x, z, hills, surfaceY, snow, height, grad) {
  if (!(snow > 0)) return 0;

  const h = height(x, z, hills, grad);
  const slope = Math.hypot(grad[0], grad[1]);
  const u = Math.min(1, Math.max(0, (slope - HOLD_LO) / (HOLD_HI - HOLD_LO)));
  const hold = 1 - u * u * (3 - 2 * u);
  if (hold <= 0) return 0;

  const above = h - surfaceY;
  const d = Math.min(1, Math.max(0, above / SHORE));
  const dry = d * d * (3 - 2 * d);

  let mottle = 0;
  for (const [kx, kz, ph, w] of PATCH) mottle += w * Math.sin(kx * x + kz * z + ph);

  return Math.min(1, Math.max(0, snow * hold * dry * (0.72 + 0.42 * mottle)));
}

/* ── what is falling ──────────────────────────────────────────────
   A rain drop and a snow flake are the same six vertices.

   Both are a quad built around a point that is falling; what separates
   them is how fast it falls, how far it is smeared along its own
   velocity, and what shape is drawn inside. That is three numbers, so
   they are three numbers and not two programs.

   There are no attributes and no buffer. Every drop's position is a hash
   of its index, advanced by the clock and wrapped into a box that
   follows the eye — so the field is infinite, nothing is ever spawned or
   retired, and the CPU's entire contribution is one draw call with a
   vertex count in it.                                                */

/** The box the drops live in, in metres. Wide enough that its edge is
    past where a drop is still legible, short enough that the drops
    inside it are not spread too thin to read as weather. */
const BOX = 30.0;
const BOX_H = 22.0;

const VERT_PRECIP = /* glsl */`
${PRECISION}
${HASH}

uniform vec3 uCamPos, uRight, uUp, uFwd;
uniform float uFocal, uAspect;
uniform float uTime, uWind;
/** 0 draws rain, 1 draws snow. Everything below reads it as a mix. */
uniform float uSnowy;

${WIND_GLSL}

out float vSide;     // -1..1 across the drop
out float vAlong;    //  0..1 down its length
out float vDepth;    // distance from the eye, for the depth test
out float vFade;

const vec3 BOX = vec3(${BOX.toFixed(1)}, ${BOX_H.toFixed(1)}, ${BOX.toFixed(1)});

void main() {
  int id = gl_VertexID / 6;
  int corner = gl_VertexID % 6;
  float along = (corner == 1 || corner == 3 || corner == 4) ? 1.0 : 0.0;
  float side  = (corner == 2 || corner == 4 || corner == 5) ? 1.0 : -1.0;

  vec3 h = hash33(vec3(float(id) * 0.7139, 11.3, 4.7));
  vec3 h2 = hash33(vec3(float(id) * 0.3311 + 7.1, 2.9, 19.3));

  /* Rain falls an order of magnitude faster than snow, and that single
     ratio is most of what tells them apart before either is close
     enough to see the shape of. */
  float fall = mix(11.0, 0.95, uSnowy) * (0.72 + h.z * 0.55);

  vec3 p = h * BOX;
  p.y -= uTime * fall;

  /* Blown along the same direction the grass is leaning, out of the same
     field, so the weather crossing the meadow and the weather falling
     through it are one weather. Rain is carried much harder than snow
     for its speed: a drop is in the air for a second and arrives
     slanted, a flake wanders for half a minute and arrives anywhere. */
  vec2 drift = WIND_DIR * uWind * mix(3.4, 1.1, uSnowy);
  p.xz += drift * uTime;

  /* And the flake's own wander, which rain does not get. A drop that
     dithered sideways would read as a bug; a flake that fell straight
     reads as ash. */
  if (uSnowy > 0.5) {
    float w = 0.35 + 0.65 * uWind;
    p.x += sin(uTime * (0.55 + h2.x * 0.9) + h2.y * 6.28) * 0.62 * w;
    p.z += cos(uTime * (0.50 + h2.y * 0.9) + h2.x * 6.28) * 0.62 * w;
    // And the gust the rest of the meadow is feeling, so a squall
    // crosses the flakes and the grass together.
    p.xz += WIND_DIR * slowGust(p.xz) * uWind * 0.9;
  }

  /* Wrapped into a box centred a little above the eye, which is where
     the drops that can be seen are: below the camera the ground is in
     the way within a metre or two, above it there is sky to fill. */
  vec3 centre = uCamPos + vec3(0.0, BOX.y * 0.18, 0.0);
  p = centre - BOX * 0.5 + mod(p - centre + BOX * 0.5, BOX);

  vec3 toEye = p - uCamPos;
  float dist = length(toEye);

  /* Smeared along its own velocity — which is what a photograph of rain
     is, and why rain is legible at all at these speeds. Snow gets the
     same construction with the smear turned almost off, so the quad
     comes out very nearly square and the fragment shader can round it
     into a flake. */
  vec3 vel = normalize(vec3(drift.x, -fall, drift.y));
  vec3 view = dist > 1e-4 ? toEye / dist : vec3(0.0, 0.0, 1.0);
  vec3 across = cross(vel, view);
  float k = length(across);
  // Looking straight down the drop's own path. Rare, and any consistent
  // perpendicular will do, because at that angle it is a dot anyway.
  across = k > 1e-4 ? across / k : normalize(cross(vel, vec3(1.0, 0.0, 0.0)));

  /* Shorter than the drop's actual travel over a frame. A streak drawn
     at its true length reads as a scratch on the lens rather than as
     rain — the eye wants many short marks, not few long ones, and the
     count is where the sense of downpour comes from. */
  float len  = mix(0.42, 0.055, uSnowy) * (0.7 + h2.z * 0.6);
  float wide = mix(0.009, 0.042, uSnowy) * (0.7 + h2.z * 0.6);

  vec3 wp = p + vel * (along - 0.5) * len + across * side * wide;

  vec3 rel = wp - uCamPos;
  vec3 v = vec3(dot(rel, uRight), dot(rel, uUp), dot(rel, uFwd));

  /* Faded at the box's edge and again right in front of the lens.
     Without the first, drops wink in and out along a square boundary
     that is very easy to see once you have seen it; without the second,
     the nearest few are metres across and read as smears on the glass. */
  vec2 e = abs(p.xz - uCamPos.xz) / (BOX.xz * 0.5);
  float edge = 1.0 - smoothstep(0.60, 1.0, max(e.x, e.y));
  vFade = edge * smoothstep(0.30, 1.30, dist);

  vSide = side;
  vAlong = along;
  vDepth = dist;
  gl_Position = vec4(v.x * uFocal / uAspect, v.y * uFocal, 0.0, v.z);
}
`;

const FRAG_PRECIP = /* glsl */`
${PRECISION}

in float vSide;
in float vAlong;
in float vDepth;
in float vFade;
out vec4 outColor;

uniform sampler2D uScene;
uniform sampler2D uMesh;
uniform float uMeshOn;
uniform vec2 uResolution;
uniform float uSnowy;
uniform float uIntensity;

void main() {
  if (vFade <= 0.002) discard;

  /* The same depth channel the flares and the beams read, and for the
     same reason — but taking the nearer of the two halves of the scene,
     because unlike a flare at the cluster, weather falls in front of and
     behind a wood. A drop tested only against the marched distance
     would fall straight through every trunk in the frame. */
  vec2 uv = gl_FragCoord.xy / uResolution;
  float sceneT = texture(uScene, uv).a;
  if (uMeshOn > 0.5) sceneT = min(sceneT, texture(uMesh, uv).a);
  if (vDepth > sceneT) discard;

  float across = 1.0 - abs(vSide);
  float down = vAlong * 2.0 - 1.0;

  // A streak: bright down the middle, tapered at both ends.
  float streak = pow(across, 1.7) * (1.0 - abs(down) * 0.30);
  // A flake: round, soft, and dimmer at the rim so it has no edge.
  float flake = pow(max(0.0, 1.0 - length(vec2(vSide, down))), 1.5);

  float shape = mix(streak, flake, uSnowy);
  // Rain is the colour of what it is reflecting, which here is the sky.
  vec3 col = mix(vec3(0.62, 0.72, 0.92), vec3(1.0, 1.0, 1.0), uSnowy);

  outColor = vec4(col * shape * vFade * uIntensity, 1.0);
}
`;

/**
 * Rain or snow, drawn over the resolved image.
 *
 * Additive and after the tonemap, exactly like the impact flares and the
 * beams. That is not laziness about ordering: what falls is small, fast
 * and bright, and putting it through the temporal filter would smear
 * every drop into a grey wash — the filter's whole job is to converge on
 * what is *not* moving.
 */
export class Precipitation {
  constructor(gl) {
    this.gl = gl;
    this.program = new Program(gl, VERT_PRECIP, FRAG_PRECIP,
                               { name: 'march/precip' });
    /** Drops drawn last frame, for the readout. */
    this.drops = 0;
  }

  /**
   * @param {object} camera  pos/right/up/fwd/focal/aspect, as the laser takes
   * @param {object} opts    mode numbers, clock, wind, and the two depth
   *                         channels to hide behind
   * @returns {number} vertices to draw, or 0 if there is nothing falling
   */
  draw(camera, opts) {
    const { rain, snow, drops } = opts.weather;
    if (!drops) { this.drops = 0; return 0; }

    this.drops = drops;
    this.program.use({
      uCamPos: camera.pos,
      uRight: camera.right,
      uUp: camera.up,
      uFwd: camera.fwd,
      uFocal: camera.focal,
      uAspect: camera.aspect,
      uTime: opts.time,
      uWind: opts.wind,
      uSnowy: snow > rain ? 1 : 0,
      uScene: opts.scene,
      uMesh: opts.mesh,
      uMeshOn: opts.meshOn ? 1 : 0,
      uResolution: opts.resolution,
      /* Rain is dimmer per drop and there is more of it; snow is the
         other way round. Tuned against the additive blend, where the
         number that matters is the sum over a pixel and not the value
         of any one drop. */
      uIntensity: snow > rain ? 0.55 : 0.30,
    });
    return drops * 6;
  }

  dispose() { this.program.dispose(); }
}
