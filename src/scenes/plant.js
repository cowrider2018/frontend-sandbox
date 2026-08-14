/* ── scenes/plant.js ─────────────────────────────────────────────────
   What everything growing on this floor has in common.

   Grass, flowers and trees are three different problems — one is
   re-derived from a hash every frame, one is sown into a buffer, one is
   grown by recursion into chunks — but they are all lit by the same sun,
   through the same shadow function, faded into the same horizon, and
   resolved against the marched scene by the same depth convention. Those
   are the parts that must not fork. A tree lit by its own copy of the
   lighting is a tree standing in a different photograph.

   Requires the includer to have brought in PI, hash33, rot2, snoise, the
   cluster field with its shadow, SKY, and the wind — the same set the
   cat's fragment shader pulls in, and for the same reason.
   ------------------------------------------------------------------ */

import { PRECISION } from '../shaders/common.js';
import { CANOPY_SHADE_GLSL } from './canopy.js';
import { RASTER_NEAR, RASTER_FAR } from './raster.js';
import { TERRAIN_GLSL } from './terrain.js';
import { WEATHER_GLSL } from './weather.js';
import { WIND_GLSL } from './wind.js';

export const PLANT_COMMON = /* glsl */`
#define NEAR ${RASTER_NEAR.toFixed(4)}
#define FAR ${RASTER_FAR.toFixed(1)}

uniform vec3  uCamPos, uRight, uUp, uFwd;
uniform float uFocal, uAspect;
uniform vec2  uJitter;

/** Where the cover is centred. Each consumer snaps this to its own cell
    size for itself, which is what keeps its hashes still. */
uniform vec2  uViewer;
/** Where this pass fades out. Every kind of plant is given its own. */
uniform float uRadius;
uniform float uWind;
uniform float uFog;

out vec3 vColor;
out vec3 vRound;
out float vDist;

/** The cat's projection, term for term: everything here shares its
    depth buffer, so everything here has to agree with its depth. */
vec4 rasterise(vec3 world) {
  vec3 rel = world - uCamPos;
  vec3 view = vec3(dot(rel, uRight), dot(rel, uUp), dot(rel, uFwd));
  float z = view.z * (FAR + NEAR) / (FAR - NEAR) - 2.0 * FAR * NEAR / (FAR - NEAR);
  return vec4(view.x * uFocal / uAspect + uJitter.x * view.z,
              view.y * uFocal + uJitter.y * view.z,
              z, view.z);
}

${TERRAIN_GLSL}
/* After the terrain, and not by preference: snow asks the ground for its
   slope and asks the water where the shore is, so the block that answers
   both has to already be here. */
${WEATHER_GLSL}
${WIND_GLSL}
${CANOPY_SHADE_GLSL}

/**
 * How much of the sun reaches a point, through the cluster, the cat, the
 * leaves overhead, and the hill in the way.
 *
 * Cheap almost everywhere despite the marches: the cluster ray leaves
 * its bounding sphere immediately unless it is actually headed into it,
 * and the ground ray leaves its slab the moment it clears the ridge
 * line, so the overwhelming majority of plants pay a handful of compares
 * and stop. Only the ones standing in a shadow do the walk.
 *
 * The hill term is the one that has to be here rather than anywhere
 * else. A blade on a north face and the soil it grows out of are shaded
 * by two different passes — this vertex shader and the marcher — and the
 * only reason they agree about where the shade falls is that both call
 * one shared terrainShadow over one shared height field.
 */
float sunlight(vec3 p, float soft) {
  float sh = canopyShade(p);
  if (soft <= 0.0) return sh;
  float k = mix(6.0, 26.0, soft);
  sh = min(sh, terrainShadow(p, uLightDir, k));
  return min(sh, min(clusterShadow(p, uLightDir, k), catShadow(p, uLightDir, k)));
}

/* Light through a leaf. A blade of grass is one cell thick and glows
   when the sun is behind it, and that backlight is most of what
   separates a meadow from a carpet of green spikes. A tree's leaves are
   thinner still. */
const float TRANSMIT = 0.85;

/**
 * How far a leaf's normal is turned back toward straight up.
 *
 * Not a cheat for its own sake. A blade standing vertically has a
 * horizontal normal, so with the sun anywhere overhead half the meadow
 * faces away from it and goes black — which is exactly what the first
 * version looked like, a field of charred spikes. The eye does not read
 * a lawn as a million vertical planes; it reads it as a *surface* with
 * texture on it, and that surface faces up. Biasing toward the ground
 * normal is what puts the sward back and leaves the blade shape as
 * variation across it rather than as the whole signal.
 *
 * A canopy is the same argument stood on its head and it holds: what the
 * eye reads is a mass, lit from above.
 */
const float SWARD_BIAS = 1.15;

/**
 * The scene's shading, minus the specular lobe.
 *
 * Occlusion multiplies the direct term as well as the ambient one, which
 * is not what a single leaf in free air would do. It is what a leaf in a
 * *mass* does: the bottom of a sward is buried in its neighbours and the
 * inside of a canopy is buried in itself, and lighting each one as
 * though it stood alone is what makes cheap foliage read as a pile of
 * green needles.
 */
vec3 shadeBlade(vec3 n, vec3 albedo, float sun, float occ, float transmit) {
  vec3 l = uLightDir;
  float ndl = max(dot(n, l), 0.0);
  float back = max(-dot(n, l), 0.0);
  back = back * back * transmit;
  return albedo * (uTint * 2.4 * (ndl + back) * sun * occ
                 + uAmbient * 1.5 * occ);
}

/**
 * Finish a plant vertex: fog it, and publish its depth.
 *
 * Fogged here rather than per fragment, which is not the usual place for
 * it. The cover is drawn several plants deep over much of the screen, so
 * a horizon term with a 220th power in it was being evaluated a dozen
 * times per pixel to shade something a few pixels tall. Across one leaf
 * the answer does not measurably change.
 */
void emit(vec3 world, vec3 col) {
  vec3 toEye = uCamPos - world;
  float dist = length(toEye);
  vColor = mix(col, sky(-toEye / dist), 1.0 - exp(-dist * uFog * 0.045));
  vDist = dist;
  gl_Position = rasterise(world);
}
`;

/**
 * The JS side of the same contract: everything `PLANT_COMMON` declares,
 * filled from the scene's one description of the light and the field.
 *
 * It exists for exactly the reason the GLSL block does. Three passes now
 * include that header — the grass, the wood, and whatever is flying over
 * both — and a header shared by three shaders whose *values* are typed
 * out three times is a header that is only shared on paper. The way that
 * fails is not a crash: it is one pass quietly keeping last week's
 * ripple radius, and a meadow where the shadow under the cluster has two
 * different edges depending on what is standing in it.
 *
 * Callers add their own on top. Extra keys cost nothing — a uniform a
 * shader does not declare is skipped on upload — so a pass that needs a
 * grid or a reach simply spreads this and appends.
 *
 * @param {object} camera pos/right/up/fwd/focal/aspect
 * @param {object} env    the scene's light, ground, weather and cluster
 * @param {object} opts   `wind`, and the subpixel `jitter` this target's
 *                        temporal filter is expecting
 */
export function plantUniforms(camera, env, opts) {
  return {
    uCamPos: camera.pos,
    uRight: camera.right,
    uUp: camera.up,
    uFwd: camera.fwd,
    uFocal: camera.focal,
    uAspect: camera.aspect,
    uJitter: opts.jitter,

    uWind: opts.wind,
    uFog: env.fog,

    /* The ground everything here is standing on, the water on it, and
       the weather over both. One number each, read by every pass, which
       is the only reason they can all be in the same place at once. */
    uHills: env.hills,
    uWaterY: env.waterY,
    uRain: env.weather.rain,
    uSnow: env.weather.snow,
    /* And the cloud, which is none of their business except that every
       one of them draws sky somewhere — into its fog, into a leaf's rim
       — and a meadow fogging out into a clear horizon under an overcast
       one is the same failure as fogging into a daytime horizon at
       night. */
    uOvercast: env.weather.overcast,

    uLightDir: env.dir,
    uTint: env.tint,
    uDay: env.day,
    uAmbient: env.ambient,
    uShadowSoft: env.shadowSoft,
    uShadowSteps: env.shadowSteps,
    uShadowNoise: env.shadowNoise,

    // The cluster, exactly as the marcher sees it this frame, so the
    // shadow it throws across a leaf is the shadow it throws.
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

    // And the cat, so it casts one too.
    uCatCapA: env.catCapA,
    uCatCapB: env.catCapB,
    uCatBound: env.catBound,
    uCatCaps: env.catCaps,

    // And the wood, so anything under a tree is in its shade.
    ...env.canopy,
  };
}

/**
 * The fragment shader, shared by every plant pass.
 *
 * Deliberately almost empty. The cover is drawn several plants deep over
 * much of the screen, so anything in here is paid for many times per
 * pixel to shade something a few pixels tall — the lighting and the fog
 * are both settled per vertex instead.
 */
export const FRAG_PLANT = /* glsl */`
${PRECISION}

in vec3 vColor;
in vec3 vRound;
in float vDist;
out vec4 outColor;

void main() {
  // Square geometry, round shape. One varying and one compare beats
  // spending a triangle fan on something four pixels across.
  if (vRound.z > 0.5 && dot(vRound.xy, vRound.xy) > 1.0) discard;

  // Alpha is the scene's depth channel, in world units from the eye.
  outColor = vec4(vColor, vDist);
}
`;
