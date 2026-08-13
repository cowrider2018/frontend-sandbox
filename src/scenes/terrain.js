/* ── scenes/terrain.js ───────────────────────────────────────────────
   How high the ground is, in one place, in two languages.

   Everything in this scene used to agree on where the floor was by
   sharing a constant. That worked because the answer was the same
   everywhere. It is not any more, so the constant becomes a function —
   and the moment it does, the important property is no longer *what* the
   function returns but that there is exactly one of it. Grass roots,
   flower roots, tree trunks, the cat's feet, the camera's floor, the
   beam's stopping point and the marcher's primary ray all have to get
   the identical number for the identical spot, or the meadow floats, the
   animal sinks, and nothing in the frame is standing on the same world.

   Two implementations is one more than one, and that is the tax this
   file exists to keep small: the shape lives in `WAVES` and both the
   GLSL and the JS are generated from it. Change a wavelength and both
   sides change together, because there is nothing to keep in sync.

   ── why sines ────────────────────────────────────────────────────
   A sum of a few sines is not the richest terrain available. It was
   chosen for three properties that noise does not have:

   Its gradient is exact and free. The derivative of a sine is a cosine,
   so the slope comes out of the same evaluation that produced the
   height — no second sample, no finite difference, no epsilon to tune.
   The cat's footing and the ground's normal are the same arithmetic.

   Its gradient is *bounded*, and the bound is known in closed form.
   That is what lets a ray march the ground conservatively (see
   `traceTerrain`) instead of creeping along it, and it is the whole
   reason the hills can cast real shadows at a price worth paying.

   And it ports without risk. Two hand-written noise implementations
   agreeing to the last bit across JS and GLSL is a real piece of work;
   four sines agreeing is arithmetic. That bill comes due when the
   terrain wants detail this cannot give it — and when it does, it is
   paid inside `terrainField` alone.

   ── the shape it is meant to grow into ───────────────────────────
   The height is authored as `low + detail`, and `detail` is currently
   empty. `low` is the part that must stay marchable: it is what casts
   shadows across the field and what a ray is traced against, so its
   slope bound is load-bearing and it should stay a handful of sines
   however elaborate the ground gets. Anything sharper — crests, banks,
   cut channels — belongs in `detail`, which only ever has to be
   evaluated at a vertex and a footfall, and can therefore be as rough as
   it likes. Splitting them now costs nothing and is what keeps the cost
   of long-range shadow independent of how complicated the ground looks.
   ------------------------------------------------------------------ */

import { FLOOR_Y } from './cluster.js';

/** What the ground averages to — the height the world had before it had
    any. Kept equal to the old floor so that nothing that measures
    against it has to be re-tuned. */
export const TERRAIN_BASE = FLOOR_Y;

/** Where the undulation control starts and stops. */
export const HILLS_DEFAULT = 2.0;
export const HILLS_MAX = 8.0;

/* ── the shape ────────────────────────────────────────────────────
   Wavelength, heading, phase, weight.

   The headings are deliberately not axis-aligned and not evenly spaced;
   four waves on a regular fan read as a woven basket from the air. The
   wavelengths are mutually non-harmonic for the same reason — any ratio
   near a simple fraction gives the field a repeat you can walk to.

   The weights fall off roughly as fast as the wavelengths do, which is
   what makes every term contribute about the same amount of *slope*
   (weight times frequency is near-constant across the four). A set
   weighted any other way is either a single hill with ripples on it or a
   flat plain with corrugations. */
const WAVES = [
  // λ,    heading, phase, weight
  [74.0, 0.00, 0.00, 1.00],
  [41.0, 1.12, 1.70, 0.55],
  [23.0, 2.37, 3.90, 0.30],
  [13.0, 3.71, 0.60, 0.17],
];

/** The waves with their weights normalised and their wave vectors
    resolved, so the unit field lands in [-1, 1] and the amplitude
    control means what it says. */
const TERMS = (() => {
  const total = WAVES.reduce((s, w) => s + w[3], 0);
  return WAVES.map(([lambda, heading, phase, weight]) => {
    const f = (Math.PI * 2) / lambda;
    return {
      kx: f * Math.cos(heading),
      kz: f * Math.sin(heading),
      phase,
      w: weight / total,
      f,
    };
  });
})();

/* ── the clearing ─────────────────────────────────────────────────
   The hills fade out at the origin.

   Not a cosmetic choice. The cluster hangs at a fixed height above the
   old floor and the cat plays underneath it; ground that can rise two
   metres would swallow both, and every constant that was measured
   against a flat floor — chest height, orbit centre, the ring's
   clearance — would need re-deriving against a surface that moves. A
   clearing costs one smoothstep and makes all of that continue to be
   true where it is looked at.

   It reads as a reason for the scene to be where it is, too: the flat
   ground is the reason anything is standing there.

   Sized against what actually needs protecting, which is the cluster and
   nothing else — there is barely a metre of clearance under the ring, so
   the ground beneath it cannot be allowed to move, and everything past
   about three metres is free. The first version held a flat disc
   twenty-six metres across and it photographed as a flat world: the
   meadow is only visible for fifteen to sixty metres, so a clearing that
   size *is* the view. What the hills are for is the part you can see. */
export const CLEAR_R0 = 2.5;
export const CLEAR_R1 = 18.0;

/**
 * The steepest the unit field can get: the sum of weight times
 * frequency, which is the gradient with every cosine at one and every
 * wave pulling the same way. Unreachable in practice — the headings
 * differ, so they cannot align — but it is an honest upper bound, and an
 * upper bound is what the ray march needs to stay conservative.
 *
 * The clearing's own ramp adds to it: at its steepest a smoothstep rises
 * 1.5 over its width, and it is carrying a field that reaches 1.
 */
const SLOPE_UNIT = TERMS.reduce((s, t) => s + t.w * t.f, 0)
                 + 1.5 / (CLEAR_R1 - CLEAR_R0);

/** Multiply by the amplitude for the real bound. Published so the tests
    and the JS side can state the same guarantee the shader relies on. */
export const TERRAIN_SLOPE_PER_AMP = SLOPE_UNIT;

/* How far the ground is marched before it is solved as a plane instead,
   and how many steps it is given to get there.

   Both are ceilings on a cost, not on a look. A ray that runs out of
   either is a grazing one heading for the horizon, where the fog has
   already taken the ground's colour and where flat and undulating are
   the same handful of pixels. See `traceGround`. */
const TERRAIN_REACH = 220.0;
const TERRAIN_STEPS = 96;

/* And the shadow ray's, which are much smaller. A shadow ray can afford
   error that a silhouette cannot — the penumbra blurs whatever is left —
   and beyond seventy-odd metres a hill's shadow is behind the fog. */
const SHADOW_REACH = 72.0;
const SHADOW_STEPS = 40;

/* ── generated GLSL ───────────────────────────────────────────────── */

const glslWaves = TERMS.map(t => `
  {
    float a = ${t.kx.toFixed(9)} * p.x + ${t.kz.toFixed(9)} * p.y + ${t.phase.toFixed(6)};
    n += ${t.w.toFixed(7)} * sin(a);
    g += vec2(${(t.w * t.kx).toFixed(9)}, ${(t.w * t.kz).toFixed(9)}) * cos(a);
  }`).join('');

/**
 * The ground, for every shader that has to know where it is.
 *
 * Declares `uHills` and `uWaterY` and nothing else. Keeping that list
 * short is a goal rather than an accident: this block is included by the
 * marcher, by the grass, by the flowers and by the trees, and every
 * uniform in it is one more thing four call sites have to remember to
 * upload identically. The step budgets and the reach are compiled in,
 * because they are costs and not controls — nobody looking at the
 * picture can tell you what they should be.
 *
 * The second one earned its place by absorbing three controls into a
 * number: whether there is a lake, how high it stands and how the hills
 * scale it are all resolved once in JS (see `waterSurfaceY`), and what
 * arrives here is the only thing any shader actually asks — the height
 * of the water.
 */
export const TERRAIN_GLSL = /* glsl */`
#define TERRAIN_BASE ${TERRAIN_BASE.toFixed(4)}
#define CLEAR_R0 ${CLEAR_R0.toFixed(2)}
#define CLEAR_R1 ${CLEAR_R1.toFixed(2)}
#define SLOPE_UNIT ${SLOPE_UNIT.toFixed(7)}
#define TERRAIN_REACH ${TERRAIN_REACH.toFixed(1)}
#define TERRAIN_STEPS ${TERRAIN_STEPS}
#define SHADOW_REACH ${SHADOW_REACH.toFixed(1)}
#define SHADOW_STEPS ${SHADOW_STEPS}

/** How high the hills stand. Zero is the flat floor this scene had
    before, exactly — every path below returns early on it. */
uniform float uHills;

/**
 * The unit field and its gradient, in one pass.
 *
 * Unrolled from the wave table in JS rather than looped: four iterations
 * of a loop carrying a per-iteration uniform fetch is strictly worse
 * than four inlined constants, and this function is called from inside
 * two marches.
 *
 * The argument is the world's xz, so the gradient is with respect to
 * those: its components are dh/dx and dh/dz. The y is not missing from
 * it — there is no y. (No back-ticks anywhere in here: this shader lives
 * inside a JS template literal and one would end it.)
 */
float terrainField(vec2 p, out vec2 g) {
  float n = 0.0;
  g = vec2(0.0);
  ${glslWaves}
  return n;
}

/**
 * How high the ground is at a point, and which way it slopes.
 *
 * The clearing is applied as a product, so the gradient needs the
 * product rule: the waves scaled down by the clearing, plus the
 * clearing's own ramp carrying the height the waves would have had.
 * Dropping the second term is the obvious simplification and it puts a
 * visible crease around the clearing's rim, because the surface would
 * then be lit by a slope it does not have.
 */
float terrainAt(vec2 p, out vec2 grad) {
  if (uHills <= 0.0) { grad = vec2(0.0); return TERRAIN_BASE; }

  vec2 gn;
  float n = terrainField(p, gn);

  float r = length(p);
  float u = clamp((r - CLEAR_R0) / (CLEAR_R1 - CLEAR_R0), 0.0, 1.0);
  float s = u * u * (3.0 - 2.0 * u);
  float ds = 6.0 * u * (1.0 - u) / (CLEAR_R1 - CLEAR_R0);
  vec2 outward = r > 1e-4 ? p / r : vec2(0.0);

  grad = uHills * (s * gn + n * ds * outward);
  return TERRAIN_BASE + uHills * s * n;
}

/** When only the height is wanted. */
float terrainH(vec2 p) { vec2 g; return terrainAt(p, g); }

/** The surface normal, from the slope rather than from four more taps. */
vec3 terrainNormal(vec2 p) {
  vec2 g;
  terrainAt(p, g);
  return normalize(vec3(-g.x, 1.0, -g.y));
}

/**
 * A height difference, converted to a lower bound on the true distance
 * to the surface.
 *
 * This one factor is what makes the ground marchable. For a height field
 * whose slope is bounded by g, a point sitting dy above the surface
 * cannot be closer to it than dy / sqrt(1 + g*g) in any direction — so
 * that is a step the march can take without ever overshooting. The bound
 * is a constant for the whole frame, which reduces the entire argument
 * to one reciprocal square root hoisted out of the loop.
 */
float terrainStepScale() {
  float g = uHills * SLOPE_UNIT;
  return inversesqrt(1.0 + g * g);
}

/**
 * The span of a ray that could possibly be underground.
 *
 * The ground is trapped between base ± amplitude, so everything above
 * the highest ridge and everything below the deepest valley is skipped
 * in a divide instead of walked. On a ray shot at the sky this returns
 * an empty span and the whole march never starts, which is most of why
 * marching the ground is affordable at all.
 */
vec2 terrainSlab(vec3 ro, vec3 rd) {
  float top = TERRAIN_BASE + uHills;
  float bot = TERRAIN_BASE - uHills;
  if (abs(rd.y) < 1e-6) {
    return (ro.y < bot || ro.y > top) ? vec2(1.0, -1.0) : vec2(0.0, 1e9);
  }
  float ta = (top - ro.y) / rd.y;
  float tb = (bot - ro.y) / rd.y;
  return vec2(min(ta, tb), max(ta, tb));
}

/** Where a ray meets the hills, or -1 if it runs out of slab or budget. */
float traceTerrain(vec3 ro, vec3 rd, float tMax) {
  vec2 slab = terrainSlab(ro, rd);
  float tEnd = min(slab.y, tMax);
  float t = max(slab.x, 0.0);
  if (tEnd <= 0.0 || t > tEnd) return -1.0;

  float k = terrainStepScale();
  for (int i = 0; i < TERRAIN_STEPS; i++) {
    vec3 p = ro + rd * t;
    float dy = p.y - terrainH(p.xz);
    // Relaxed with distance, on the same reasoning as the cluster's
    // march: a far pixel is subpixel, and the tolerance buys back steps
    // exactly where the steps are being spent.
    if (dy < 0.0012 * t + 0.0008) return t;
    t += max(dy * k, 0.008);
    if (t > tEnd) return -1.0;
  }
  return -1.0;
}

/**
 * The ground: marched where it can be seen, solved where it cannot.
 *
 * A descending ray always meets a height field eventually, so a failed
 * march means the budget ran out, not that there was nothing there —
 * and returning nothing would open a hole in the world at the horizon,
 * which is precisely where grazing rays exhaust their budget. The
 * fallback is the plane the terrain averages to. It is wrong by up to
 * the amplitude, at a distance where the fog is already most of the
 * colour and the ground is a few pixels tall.
 *
 * That fallback is also the shape of the future here: when the far field
 * is eventually drawn as a mesh rather than marched, this is the
 * function that changes, and nothing that calls it has to know.
 */
float traceGround(vec3 ro, vec3 rd) {
  float t = traceTerrain(ro, rd, TERRAIN_REACH);
  if (t > 0.0) return t;
  if (rd.y >= -1e-5) return -1.0;
  float tp = (TERRAIN_BASE - ro.y) / rd.y;
  return tp > 0.0 ? tp : -1.0;
}

/* ── the water ────────────────────────────────────────────────────
   A lake is one number: the height the water stands at.

   It is a plane, and it is a plane on purpose. Every richer surface — a
   second height field, a flow map, something simulated — buys detail
   that is invisible from the fifteen metres this scene is ever viewed
   at, and spends the one property that makes water nearly free here: a
   ray meets a horizontal plane in a divide, not a march. What makes it
   read as a lake rather than a sheet of glass laid over the world is
   that the hit only counts where the ground beneath it is lower, and
   that costs exactly one height evaluation, at the one point the ray
   landed.

   So the shoreline is never authored. It is wherever the hills happen to
   cross uWaterY; it moves when either of them moves; and no two
   consumers can disagree about where it is, because every one of them
   asks the same question of the same field. The same property is what
   lets the planting keep seeds out of the lake without knowing what a
   lake is — a seed is drowned if waterDepth is positive, and that is the
   whole rule.

   Off is not a flag. Water standing below the deepest the ground can
   reach is water no ray can meet and no seed can drown in, so hasWater
   is an honest test rather than a sentinel check — and it is the same
   test that makes a flat world (uHills == 0, where the ground and the
   surface would be coplanar and the lake would be everywhere and
   nowhere) dry for free.

   (No back-ticks in here: this block lives inside a JS template literal
   and one would end it. Same rule as terrainField above.) */
uniform float uWaterY;

/** Whether there is any water in this world at all. */
bool hasWater() { return uWaterY > TERRAIN_BASE - uHills; }

/** Positive under water, negative on dry land, zero on the shoreline. */
float waterDepth(vec2 p) { return uWaterY - terrainH(p); }

/**
 * Where a ray meets the surface of the water, or -1.
 *
 * Deliberately *not* occluded against the ground on the way in: the
 * caller marches the ground anyway and takes whichever came first, which
 * is what stops a ray that crests a ridge from seeing the lake through
 * it. Doing that test here would mean marching the terrain twice for an
 * answer the caller already has.
 */
float traceWater(vec3 ro, vec3 rd) {
  if (!hasWater()) return -1.0;
  if (abs(rd.y) < 1e-6) return -1.0;
  float t = (uWaterY - ro.y) / rd.y;
  if (t <= 0.0) return -1.0;
  vec3 p = ro + rd * t;
  return terrainH(p.xz) < uWaterY ? t : -1.0;
}

/**
 * How much of the sun the ground itself takes away.
 *
 * The same conservative step as the primary march, walked coarser and
 * cut off sooner. This is the term that rakes long shadows out of a low
 * sun across the whole field — and the reason the hills are marched
 * rather than mapped: there is no resolution here to run out of, no
 * cascade to split, no depth bias to tune, and the answer is the same
 * function the grass and the trees and the cat are already standing on,
 * so four passes drawn separately cannot disagree about where the shade
 * is.
 */
float terrainShadow(vec3 ro, vec3 rd, float k) {
  if (uHills <= 0.0) return 1.0;
  // Above the highest ridge and climbing: nothing left to be behind.
  if (rd.y > 0.0 && ro.y > TERRAIN_BASE + uHills) return 1.0;

  vec2 slab = terrainSlab(ro, rd);
  float tEnd = min(slab.y, SHADOW_REACH);
  float t = max(slab.x, 0.05);
  if (tEnd <= t) return 1.0;

  float sc = terrainStepScale();
  float res = 1.0;
  for (int i = 0; i < SHADOW_STEPS; i++) {
    vec3 p = ro + rd * t;
    float d = (p.y - terrainH(p.xz)) * sc;
    if (d < 0.0) return 0.0;
    res = min(res, k * d / t);
    if (res < 0.004) break;
    t += max(d, 0.06);
    if (t > tEnd) break;
  }
  return clamp(res, 0.0, 1.0);
}
`;

/* ── the same thing, in JS ────────────────────────────────────────
   Read by the cat's footing, the camera's floor, the beam's stop and the
   trees' planting. Generated from the same table as the GLSL above, so
   the only way the two can disagree is a difference between float and
   double — a part in ten million of a metre, which is not a difference
   anything in this scene can stand on.                               */

/**
 * How high the ground is at a world xz, and optionally which way it
 * slopes there.
 *
 * @param {number} x
 * @param {number} z
 * @param {number} hills   amplitude; 0 gives the flat floor exactly
 * @param {number[]} [grad] filled with [dh/dx, dh/dz] when supplied
 */
export function terrainHeight(x, z, hills, grad) {
  if (grad) { grad[0] = 0; grad[1] = 0; }
  if (!(hills > 0)) return TERRAIN_BASE;

  let n = 0, gx = 0, gz = 0;
  for (const t of TERMS) {
    const a = t.kx * x + t.kz * z + t.phase;
    n += t.w * Math.sin(a);
    const c = Math.cos(a);
    gx += t.w * t.kx * c;
    gz += t.w * t.kz * c;
  }

  const r = Math.hypot(x, z);
  const u = Math.min(1, Math.max(0, (r - CLEAR_R0) / (CLEAR_R1 - CLEAR_R0)));
  const s = u * u * (3 - 2 * u);

  if (grad) {
    const ds = 6 * u * (1 - u) / (CLEAR_R1 - CLEAR_R0);
    const ox = r > 1e-4 ? x / r : 0;
    const oz = r > 1e-4 ? z / r : 0;
    grad[0] = hills * (s * gx + n * ds * ox);
    grad[1] = hills * (s * gz + n * ds * oz);
  }
  return TERRAIN_BASE + hills * s * n;
}

/**
 * Where a ray first meets the ground, or -1.
 *
 * The JS side of `traceGround`, marched with the identical conservative
 * step, and wanted by exactly one caller: the beam, which has to stop
 * somewhere and used to stop at a plane. Kept here rather than in the
 * laser so that "how a ray meets this ground" has one answer in each
 * language and not one per user.
 */
export function traceTerrainJS(ox, oy, oz, dx, dy, dz, tMax, hills) {
  if (!(hills > 0)) {
    if (dy >= -1e-5) return -1;
    const t = (TERRAIN_BASE - oy) / dy;
    return t > 0 && t <= tMax ? t : -1;
  }

  const top = TERRAIN_BASE + hills;
  const bot = TERRAIN_BASE - hills;
  let t0, t1;
  if (Math.abs(dy) < 1e-6) {
    if (oy < bot || oy > top) return -1;
    t0 = 0; t1 = tMax;
  } else {
    const ta = (top - oy) / dy;
    const tb = (bot - oy) / dy;
    t0 = Math.min(ta, tb);
    t1 = Math.max(ta, tb);
  }

  const tEnd = Math.min(t1, tMax);
  let t = Math.max(t0, 0);
  if (tEnd <= 0 || t > tEnd) return -1;

  const g = hills * SLOPE_UNIT;
  const k = 1 / Math.sqrt(1 + g * g);
  for (let i = 0; i < 96; i++) {
    const h = terrainHeight(ox + dx * t, oz + dz * t, hills);
    const gap = oy + dy * t - h;
    if (gap < 0.0012 * t + 0.0008) return t;
    t += Math.max(gap * k, 0.008);
    if (t > tEnd) return -1;
  }
  return -1;
}

/* ── the water level ──────────────────────────────────────────────
   Three controls in, one number out.

   The shader half above takes `uWaterY` and asks nothing else; this is
   where that number is made, and it is made once per frame in one place
   so that the marcher, the grass, the flowers and the trees cannot be
   looking at four different lakes.                                    */

/**
 * How much of the open meadow stands under water at either end of the
 * control.
 *
 * The control is authored in *area*, not in depth, and the difference is
 * the whole usability of the slider. Depth was the obvious choice and it
 * was wrong: the ground's heights are a sum of sines, so they pile up
 * near the middle and thin out at the extremes, and a slider linear in
 * depth spends its bottom half moving the surface through ground that
 * hardly any of the meadow reaches. Measured on the real field, a depth
 * control put every lake anyone would want in its top fifth — 0.45 was a
 * puddle and 0.95 was the first setting that read as water.
 *
 * Area is what the eye is actually reading, so that is what the number
 * means: 0 is a few ponds in the deepest hollows, 1 is a little under
 * half the open ground. Depth is then whatever produces that, which is
 * a question about the shape of the field, and the field is right here.
 *
 * The top end deliberately stops short of half. Past that the lakes join
 * up and the meadow becomes an archipelago — and, more to the point, the
 * surface approaches the base, which is the height the clearing holds
 * the ground at. Water at the base is water lapping at the cluster and
 * the cat, and every constant in this scene that was measured against a
 * flat dry floor would need re-deriving. Same argument the clearing
 * itself is made of; this is the number that keeps it true.
 */
const WATER_AREA_MIN = 0.02;
const WATER_AREA_MAX = 0.42;

/**
 * Depth as a function of flooded area — the field's own height
 * distribution, sampled once at load.
 *
 * Derived rather than authored, which is the point: a table of numbers
 * typed in here would be a second description of the terrain, correct
 * until the first time somebody changed a wavelength in `WAVES` and
 * silently wrong from then on. Computing it from `TERMS` costs a few
 * milliseconds once and cannot go stale.
 *
 * Sampled on a lattice whose spacing is non-harmonic with all four
 * wavelengths, so the grid cannot land on the same phase repeatedly and
 * report a distribution the ground does not have — the same reason the
 * wavelengths themselves are mutually non-harmonic. The clearing is not
 * applied: this is the open field, which is where all the water is.
 */
const WATER_TABLE_TO = 0.6;      // the table spans flooded fractions 0..0.6
const WATER_TABLE_N = 65;

const WATER_TABLE = (() => {
  const SIDE = 128, STRIDE = 1.9;
  const h = new Float64Array(SIDE * SIDE);
  for (let i = 0; i < SIDE; i++) {
    const x = (i - SIDE / 2) * STRIDE;
    for (let j = 0; j < SIDE; j++) {
      const z = (j - SIDE / 2) * STRIDE;
      let n = 0;
      for (const t of TERMS) n += t.w * Math.sin(t.kx * x + t.kz * z + t.phase);
      h[i * SIDE + j] = n;
    }
  }
  h.sort();

  const out = new Float64Array(WATER_TABLE_N);
  for (let i = 0; i < WATER_TABLE_N; i++) {
    const p = (i / (WATER_TABLE_N - 1)) * WATER_TABLE_TO;
    // The depth that puts exactly this fraction of the field under it —
    // negated, because depth is measured downward from the base.
    out[i] = -h[Math.round(p * (h.length - 1))];
  }
  return out;
})();

/** How deep the surface has to sit for that much of the field to drown. */
function waterDepthForArea(area) {
  const u = Math.min(WATER_TABLE_TO, Math.max(0, area))
          / WATER_TABLE_TO * (WATER_TABLE_N - 1);
  const i = Math.min(WATER_TABLE_N - 2, Math.floor(u));
  const f = u - i;
  return WATER_TABLE[i] * (1 - f) + WATER_TABLE[i + 1] * f;
}

/**
 * How much of the open meadow a given level puts under water.
 *
 * The control's own definition, exported so the acceptance tool can hold
 * the slider to what it claims rather than to a screenshot.
 */
export function waterArea(level) {
  const u = Math.min(1, Math.max(0, level));
  return WATER_AREA_MIN + (WATER_AREA_MAX - WATER_AREA_MIN) * u;
}

/** Where the level control starts. Enough lake to be worth switching on. */
export const WATER_DEFAULT = 0.45;

/** A surface no ray can reach and no seed can drown in. */
export const WATER_OFF = -1e4;

/**
 * The world height of the water, from the switch, the level and the
 * hills — the sole producer of `uWaterY`.
 *
 * A flat world has no water. Not a restriction so much as an
 * observation: the ground and a horizontal surface would be coplanar,
 * and "which of these two is in front" has no answer that is stable
 * frame to frame. There is nowhere for a lake to be in a world with no
 * low ground, and the honest thing is to say so here rather than let
 * every consumer discover it as z-fighting.
 *
 * @param {boolean} on     the lake switch
 * @param {number} level   0..1, the level control
 * @param {number} hills   the hill amplitude
 * @returns {number} world y of the surface, or `WATER_OFF`
 */
export function waterSurfaceY(on, level, hills) {
  if (!on || !(hills > 0)) return WATER_OFF;
  /* Scaled by the amplitude rather than fixed in metres, which is what
     keeps the control meaning the same thing as the hills are dragged: a
     given level floods the same fraction of the meadow whether the hills
     are one metre or eight. */
  return TERRAIN_BASE - hills * waterDepthForArea(waterArea(level));
}

/**
 * Positive under water, negative on dry land — the JS side of
 * `waterDepth`, and the whole of what the planting needs to know.
 *
 * Takes the surface height rather than the controls that made it, so
 * that a caller seeding ten thousand blades resolves the level once and
 * then pays for nothing here but the height field it was going to
 * evaluate anyway.
 */
export function waterDepthAt(x, z, hills, surfaceY) {
  return surfaceY - terrainHeight(x, z, hills);
}
