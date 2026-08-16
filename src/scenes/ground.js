/* ── scenes/ground.js ────────────────────────────────────────────────
   Grass and flowers on the marched floor.

   The floor is one plane solved in closed form, which is why it costs
   nothing and also why nothing can grow out of it: putting a hundred
   thousand blades into a distance field means a sphere tracer walking
   past every one of them to reach the ground. So the cover is not in
   the field at all. It is triangles, drawn into the same target the cat
   draws into, resolved against the march by the same one-line depth
   comparison — cat and grass share a depth buffer, so the animal stands
   *in* the field rather than on a picture of one.

   There is no mesh. A blade is seven vertices derived from gl_VertexID
   and a flower is forty-eight; nothing is loaded, and the grass is not
   even stored — it is re-derived from a hash of its cell every frame,
   which for two hash calls across seven vertices is cheaper than
   fetching it would be. The flowers are stored, because for them it is
   not: reaching forty-eight vertices meant running the whole placement
   chain forty-eight times for an answer that never changes.

   Three ideas carry the file:

   The patch follows the viewer. A fixed lawn is either too small to
   reach the horizon or too large to draw, so the grid is centred on the
   camera and snapped to its own cell size, and every plant is seeded
   from the *world* coordinate of its cell. The grid slides; the meadow
   does not. It fades at the rim by shrinking into the ground, which is
   the one edge treatment that has nothing to pop.

   How far it reaches is a control, and the cell size grows with the
   square root of it, so the cost grows linearly rather than
   quadratically and what thins out is the far field.

   The wind is a field, not an animation. One function of position and
   time, read by the blades and by the flower stems alike, so a gust
   crosses the whole meadow as a single event and the flowers lean with
   the grass around them instead of keeping their own private time.
   ------------------------------------------------------------------ */

import { Program } from '../core/program.js';
import { PRECISION, CONSTANTS, HASH, ROTATE, SIMPLEX3 } from '../shaders/common.js';
import {
  CLUSTER_UNIFORMS, CLUSTER_FIELD, CLUSTER_LAYERS, CLUSTER_SHADOW, SKY,
} from './cluster.js';
import { CAT_PROXY_GLSL } from './cat/index.js';
import { PLANT_COMMON, FRAG_PLANT, plantUniforms } from './plant.js';
import { terrainHeight, waterDepthAt, TERRAIN_BASE } from './terrain.js';
import { snowCoverAt } from './weather.js';
import { CLUMP_CELL, CLUMP_R_MIN, CLUMP_R_MAX, seed } from './clumps.js';

/* ── the grid ─────────────────────────────────────────────────────
   How far the cover reaches is a control, because the two things people
   want from it pull opposite ways: an unbroken plain out to the fog, or
   the frame rate back.

   It is a control over *how much ground is covered*, and emphatically
   not over how big the plants are. One grid whose cells grow with the
   reach was the obvious way to keep the cost down and it is the wrong
   thing entirely: scaling the cells scales the whole pattern, so
   winding the reach out thins the grass at your feet. What you asked
   for was more field, and what you got was the same field enlarged.

   So the near cells never change size. Reach is bought by adding
   *rings* around them: concentric square bands, each one coarser than
   the one inside it, each with a hole cut out where the finer ring has
   already covered the ground. Ring 0 is the same lawn at every setting.
   The cost then grows with the number of rings — roughly the log of the
   reach — rather than with its square, and what thins out is the far
   field, which is where blades are smallest on screen anyway. */
const CELL = 0.5;
/** Cells across one ring. Ring 0 therefore covers GRID*CELL square. */
const GRID = 64;
/** Half-extent of the innermost ring: the reach one ring alone can give. */
const HALF0 = GRID * CELL * 0.5;
const MAX_RINGS = 5;

/** Reach, in world units, at each end of the control. */
export const RADIUS_MIN = 8.0;
export const RADIUS_MAX = 200.0;

/** Blades per cell at full density. One instance each. */
const MAX_LAYERS = 6;
/** Vertices in a blade: three quads narrowing to a single tip vertex. */
const BLADE_VERTS = 7;
/** And out past the first rings, one triangle. */
const BLADE_VERTS_FAR = 3;

/* How many rings still ask what the sun can see.

   Ring 2 begins twenty-odd metres out, where the fog has already taken
   most of a blade's colour and where its shadow is a fraction of a
   pixel. Every blade out there was paying a bounding-sphere test against
   the cluster and five texture fetches against the canopy, several
   hundred thousand times a frame, for something nobody can resolve. */
const SHADOW_RINGS = 2;

/**
 * How many rings a reach needs, and how much coarser each is than the
 * one inside it.
 *
 * The step is solved rather than fixed at two, so the outermost ring
 * lands on the requested reach instead of overshooting to the next
 * power of two — which keeps the control continuous. The square is
 * sized a little past the fade circle so the corners cannot come up
 * short of it once the origin is snapped.
 */
function coverPlan(reach) {
  const r = Math.max(RADIUS_MIN, Math.min(RADIUS_MAX, reach));
  // One ring can serve any reach that fits inside it, and that is the
  // cheapest the cover ever gets.
  if (r <= HALF0 - CELL) return { rings: 1, step: 1, radius: r };

  const want = r * 1.10;
  const rings = Math.min(MAX_RINGS, Math.max(2, Math.ceil(Math.log2(want / HALF0)) + 1));
  const step = Math.pow(want / HALF0, 1 / (rings - 1));
  return { rings, step, radius: r };
}

/* ── flowers ──────────────────────────────────────────────────────
   Flowers are placed by clump, not by blade. A clump cell either grows
   one or it does not, and the flowers inside it thin out toward its rim,
   so what comes out is a dense middle with singles scattered round it
   rather than a disc with an edge.

   The grid itself lives in clumps.js now, because the butterflies read
   it too. A flight aimed at a second, private scatter would circle bare
   grass a few metres from the flowers, and a near miss is far more
   visible than no aim at all — the eye notices the gap, and never
   notices an absence. */

/* Flowers stop well short of where the grass does. A blade at the rim
   still counts, because what it contributes there is the *colour* of
   the far field; a flower head at that range is under a pixel and
   contributes a speck.

   The ratio is the unified part — one reach control moves the grass, the
   flowers and the wood together — and the ceiling is where the ratio
   stops being worth honouring. Clump cells are a fixed four metres, so
   flower count grows with the square of the reach, and past this the
   sowing buffer fills before the circle does. */
const FLOWER_REACH = 0.62;
const FLOWER_MAX = 60.0;
/* How many clump cells grow one, and how many flowers a clump gets.
   Two controls rather than one, because they are two different pictures:
   a few dense clusters in bare grass, or flowers scattered thinly right
   across the field. One slider can reach either but never says which of
   the two it is doing, and the halfway settings of the pair are the ones
   worth having. */
const CLUMP_CHANCE = 0.62;
/** Flower slots per clump at full density; the thinning eats about 30%. */
const MAX_PER_CLUMP = 30;
/* How wide a clump is before the spread control scales it — in clumps.js
   with the rest of the grid. Wound right up, clumps overlap into a
   continuous scatter, which is a legitimate thing to want and the reason
   the control's range runs past 1. */
/** Ceiling on the placement buffer. Comfortably above the widest reach. */
const MAX_FLOWERS = 12288;
/** Floats per flower in that buffer; see the attribute layout below. */
const FLOWER_STRIDE = 11;

const PETALS = 5;
/** Stem 2 quads, one quad per petal, one quad for the eye. */
const FLOWER_VERTS = 12 + PETALS * 6 + 6;

/** Half-width at the root, and how tall an average blade stands. Shared,
    because a flower's stem is measured against the sward it grows out of. */
const BLADE_W = 0.013;
const BLADE_H = 0.32;

/* ── reeds ────────────────────────────────────────────────────────
   The first thing growing on this floor that wants the water rather
   than avoids it.

   Everything else asks waterDepth one question — am I drowned — and
   answers it three ways, but all three answers are "not here": the
   grass tapers out over the last twelve centimetres, the flowers are
   left unsown, the trees are never grown. So the shoreline ends up a
   line that nothing lives on, and a lake with a mown edge is a lake
   somebody drew.

   A reed asks the identical question of the identical field and accepts
   a *band* of answers instead of a half-plane: from a little way up the
   bank to a little way out into the shallows. That is the whole of the
   difference, and it is why this needs no second description of where
   the water is.

   The band is walked onto, not searched for. A cell picks the depth it
   wants to stand in and takes a Newton step along the height field's own
   gradient to get there — the gradient is exact and comes back from the
   same evaluation that gave the height, so finding the waterline costs
   two terrain samples rather than a scan. It also puts more of them
   where the bank is shallow, because that is where a step reaches, and
   a shallow bank is where reeds are.

   Nothing clips the part that is under water. The lake is drawn by the
   marcher and the reeds by this pass, and the composite keeps whichever
   is nearer — so a submerged stem is behind the surface and gone, and
   what stands above it is in front and drawn. A reed standing in a lake
   is the depth buffer doing its job and no code at all. */

/* Further than the flowers, nearer than the trees. The count grows with
   the *length* of shoreline in view rather than with the area, so the
   reach is cheap here in a way it is not for anything sown across the
   open field — what it costs is the scan, and that is paid once when the
   patch moves. */
const REED_REACH = 0.75;
const REED_MAX = 90.0;
/** The shoreline scan's cell. At most one reed comes out of each. */
const REED_CELL = 0.95;
/* How far the patch has to move before the shore is scanned again.
   Deliberately not the cell: the scan is the one expensive thing here —
   tens of thousands of cells, each an evaluation of the height field —
   and tying it to the cell would pay that every metre walked. Snapping
   it coarsely and sowing a margin past the reach costs a ring of reeds
   nobody sees and buys back seven scans in eight. */
const REED_SOW_STEP = 4.0;
/* How wide a stand of them is. Reeds grow in beds, not as a picket
   fence: a coarse hash over this many cells thickens some stretches of
   shore and empties others, which is the same argument the flower clumps
   are made of and the same one extra hash. */
const REED_STAND = 8;
/** The band, in metres of water depth: up the bank, and out into it. */
const REED_ABOVE = 0.26;
const REED_BELOW = 0.42;
const MAX_REEDS = 6144;
/** Floats per reed; see the attribute layout in the shader. */
const REED_STRIDE = 8;
/* Where the rows up a stalk sit. A table rather than an even division
   because five sixths of a reed is a straight line: three of the six are
   spent between 0.70 and 0.92, which is the only part with a shape. */
const REED_ROWS = [0.0, 0.32, 0.58, 0.70, 0.82, 0.92];
/** Six rows of two, and a point on top. */
const REED_VERTS = REED_ROWS.length * 2 + 1;
const REED_H_MIN = 0.85;
const REED_H_MAX = 1.55;
/** What the wind's push is measured against, so a tall one leans further
    than a short one and both still answer the one slider. */
const REED_H_REF = (REED_H_MIN + REED_H_MAX) * 0.5;
/** How many carry a seed head rather than running out to a point. */
const HEAD_CHANCE = 0.55;

/* ── shared shader ────────────────────────────────────────────────── */

/**
 * What the blades and the flowers need on top of what every plant needs.
 *
 * The lighting, the fog, the projection, the wind and the depth
 * convention all live in plant.js, because the trees read exactly the
 * same ones and two copies of "how this scene lights a leaf" is how a
 * meadow and a wood end up looking like two photographs.
 */
const GROUND_COMMON = /* glsl */`
${PLANT_COMMON}

#define PETALS ${PETALS}
#define BLADE_W ${BLADE_W.toFixed(4)}
#define BLADE_H ${BLADE_H.toFixed(4)}

/**
 * A point on a bending stalk, and the direction the stalk is heading
 * there.
 *
 * The bend is a rotation about the base whose angle grows as the square
 * of the height, which is what a stalk anchored at one end and pushed
 * along its length actually does: stiff at the root, most of the give
 * near the tip. Rotating rather than shearing matters — a sheared blade
 * gets longer as it leans, and a meadow of blades all getting longer
 * together is the single most obvious way to make wind look wrong.
 */
vec3 stalk(vec3 base, float len, float f, vec2 dir, float bend, out vec3 tangent) {
  float a = bend * f * f;
  float s = sin(a), c = cos(a);
  // The derivative of the curve, not of the straight blade: the extra
  // term is the angle still turning as f grows.
  float g = 2.0 * a;
  tangent = normalize(vec3(dir.x * (s + c * g), c - s * g, dir.y * (s + c * g)));
  return base + vec3(dir.x * s, c, dir.y * s) * (len * f);
}
`;

/* ── grass ────────────────────────────────────────────────────────── */

const VERT_GRASS = /* glsl */`
${PRECISION}
${CONSTANTS}
${HASH}
${ROTATE}
${SIMPLEX3}
${CLUSTER_UNIFORMS}
${CLUSTER_FIELD}
${CLUSTER_LAYERS}
${CLUSTER_SHADOW}
${CAT_PROXY_GLSL}
${SKY}
${GROUND_COMMON}

uniform float uShadowSoft;
/* The rings. uCell is ring 0's cell and never changes with the reach —
   that is the whole point of the scheme. */
uniform int   uGrid, uLayers, uRings;
uniform float uCell, uRingStep;
/* Which ring this draw is, and how many vertices it is spending on a
   blade. Both were derived per instance when every ring went out in one
   call; they are per-draw now, which is what lets the far rings be
   cheaper rather than merely smaller. */
uniform int   uRing, uBladeVerts;

/* Root, tip, and the colour of a blade that has given up. Authored
   linear, like everything downstream of here. */
const vec3 GRASS_ROOT = vec3(0.038, 0.072, 0.022);
const vec3 GRASS_TIP  = vec3(0.155, 0.290, 0.072);
const vec3 GRASS_DRY  = vec3(0.290, 0.230, 0.062);

/** How far the normal splays toward the edges. A flat ribbon without it. */
const float BLADE_CURL = 0.85;

void main() {
  vColor = vec3(0.0);
  vRound = vec3(0.0);
  vDist = 0.0;

  int cells = uGrid * uGrid;
  int ring = uRing;
  int cell = gl_InstanceID % cells;
  int layer = gl_InstanceID / cells;

  vec2 ci = vec2(float(cell % uGrid), float(cell / uGrid)) - float(uGrid) * 0.5;

  /* Each ring is coarser than the one inside it and snapped to its own
     size, so the fine cells stay fine no matter how far the cover
     reaches. Ring 0 is the same lawn at every setting of the control. */
  float cellR = uCell * pow(uRingStep, float(ring));
  vec2 origin = floor(uViewer / cellR + 0.5) * cellR;

  /* Cut out the middle: a finer ring has already covered it. Biased a
     cell inward so the rings overlap slightly rather than risk a gap —
     each is snapped to its own size, so their edges do not line up, and
     an overlapping band is invisible where a bare one would be a ring of
     naked soil around the viewer. */
  if (ring > 0) {
    float hole = float(uGrid) * 0.5 / uRingStep - 1.0;
    if (max(abs(ci.x + 0.5), abs(ci.y + 0.5)) < hole) {
      gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
      return;
    }
  }

  vec2 corner = origin + ci * cellR;

  /* Hashed off the world coordinate of the cell, never off the instance
     index. The grid slides under the camera; if the seed slid with it
     the entire meadow would crawl along the ground as you walked.

     Quantised before hashing, and not for tidiness: the corner is a
     multiple of the cell size, but it is a multiple computed in floats,
     and two paths to the same cell can land an ulp apart. One ulp is
     enough for hash33 to return a different blade.

     The ring goes into the salt as well. Without it, ring 1's cell at
     the origin and ring 0's cell at the origin are the same key, and
     two blades grow out of the same spot in the overlap band. */
  vec2 key = floor(corner / cellR + 0.5);
  float salt = float(layer) * 13.3 + float(ring) * 57.1;
  vec3 h1 = hash33(vec3(key * 31.7, salt + 0.7));
  vec3 h2 = hash33(vec3(key * 17.1 + 91.3, salt * 0.44 + 2.1));

  vec2 base = corner + (h1.xy * 0.94 + 0.03) * cellR;

  /* The rim fade is a height fade. Blades do not become transparent and
     they do not wink out; they shorten into the ground, and the last
     thing to go is the thing that was already the least visible. */
  float rim = 1.0 - smoothstep(uRadius * 0.72, uRadius, length(base - uViewer));

  /* And the shore. Grass drowns, and it drowns the same way it fades at
     the rim: by shortening into the ground rather than winking out.
     A hard cull at the waterline would be a hard cull at a line nobody
     drew — the shore is wherever the hills happen to cross the surface,
     so it wanders across the cells at whatever angle it likes, and a
     binary test would show every one of those cells as a stair. Twelve
     centimetres of taper turns the same information into a fringe of
     short grass standing in the shallows, which is what a lake edge
     looks like anyway.

     Multiplied into the same factor rather than tested separately, so a
     blade that is both far and wet gets the smaller of the two and the
     cull below keeps working unchanged. */
  rim *= 1.0 - smoothstep(0.0, 0.12, waterDepth(base));

  /* And the snow. A drift does not shorten a blade, it hides the bottom
     of one — but a blade is drawn upward from its root, so on this
     geometry those are the same edit, and the third factor goes into the
     same product as the other two.

     Not all the way to zero at full cover. What reads as a snowed-over
     meadow is not a blank white plain; it is stubble showing through,
     thinning as the drifts deepen, which is also what the mottle in
     snowCover is for. */
  float snow = snowCover(base);
  rim *= 1.0 - snow * 0.78;
  if (rim <= 0.002) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }

  float height = BLADE_H * mix(0.55, 1.20, h1.z) * rim;

  /* Near, a blade is three quads and a point: rows 0..2 carry a left and
     a right vertex and the seventh sits on the centreline at the tip.

     Far, it is one triangle — two roots and a tip. That loses the bend
     along its length, which is the whole of what a blade's silhouette
     is, and past twenty metres a blade is two pixels tall and has no
     silhouette to lose. */
  int vid = gl_VertexID;
  float row, side;
  if (uBladeVerts <= 3) {
    row  = vid == 2 ? 3.0 : 0.0;
    side = vid == 2 ? 0.0 : float(vid) * 2.0 - 1.0;
  } else {
    bool tip = vid >= 6;
    row  = tip ? 3.0 : float(vid / 2);
    side = tip ? 0.0 : float(vid & 1) * 2.0 - 1.0;
  }
  float f = row / 3.0;

  // Which way the blade leans of its own accord, and how far.
  float yaw = h2.x * PI * 2.0;
  vec2 own = vec2(cos(yaw), sin(yaw));
  float lean = mix(0.18, 0.52, h2.y);

  /* The gust is sampled with a per-blade offset so that neighbours are
     not in phase. Without it a gust arrives as a single hard line
     sweeping across, which is a wave, not weather. */
  float g = gust(base + own * (h2.z * 2.4));
  float push = uWind * (0.12 + 0.88 * g) * mix(0.75, 1.3, h1.x);

  vec2 bendDir = normalize(own * lean + WIND_DIR * push);
  float bend = min(lean + push * 1.15, 1.35);

  /* The ground came up to meet it. A blade still grows straight up —
     grass grows against gravity, not out of the slope — but where it
     starts and which way the sward it belongs to is facing both come
     from the height field. */
  vec2 slope;
  vec3 root = vec3(base.x, terrainAt(base, slope), base.y);
  vec3 sward = normalize(vec3(-slope.x, 1.0, -slope.y));

  vec3 tangent;
  vec3 p = stalk(root, height, f, bendDir, bend, tangent);

  vec2 perp = vec2(-bendDir.y, bendDir.x);
  vec3 across = vec3(perp.x, 0.0, perp.y);
  float halfW = BLADE_W * mix(0.75, 1.30, h1.y) * (1.0 - f * 0.55) * rim;
  p += across * (halfW * side);

  /* Which face the eye is on, decided once for the whole blade from its
     root. Per-vertex it would be decided six times and can disagree
     across the width, and a blade lit from both sides at once flickers.
     The curl is applied after and keeps its own sense either way — it
     splays outward from the centreline, which is not a handedness. */
  /* Biased toward the *hill's* normal, not toward straight up. On flat
     ground the two are the same and nothing changes; on a slope they are
     not, and a sward that keeps facing the sky while the soil it grows
     out of turns away from the sun is how a hillside ends up with lit
     grass standing on dark earth. The soil is shaded by the marcher and
     the grass by this shader — they only agree because both read the
     same gradient. */
  vec3 face = normalize(cross(across, tangent));
  float sgn = dot(face, uCamPos - root) < 0.0 ? -1.0 : 1.0;
  vec3 n = normalize(face * sgn + across * (side * BLADE_CURL)
                   + sward * SWARD_BIAS);

  /* Shaded from one point half-way up, not from this vertex. The shadow
     is the expensive term and it cannot resolve a blade anyway, so
     asking seven times per blade would buy seven identical answers —
     and asking at slightly different points buys a blade lit unevenly
     along its own length, which is worse than either. */
  float sun = sunlight(root + vec3(0.0, height * 0.5, 0.0), uShadowSoft);

  float dry = smoothstep(0.70, 1.0, h2.z) * 0.6;
  vec3 albedo = mix(GRASS_ROOT, GRASS_TIP, f);
  albedo = mix(albedo, GRASS_DRY, dry * f);
  albedo *= mix(0.80, 1.20, h1.z);

  /* And the shore, which is where this actually shows.

     The soil goes dark and smooth for the last half metre before the
     water — but the soil is not what anyone is looking at down there.
     The sward is: a blade is one of a hundred thousand, and what reads
     from thirty metres away is their colour, not the dirt between them.
     Wetting the ground and leaving the grass bright over it puts the
     whole band under a lawn and hides it, which is exactly what the
     first version did.

     Same field and the same question the taper up at the top of this
     shader already asks, over a range four times as wide: the taper is
     about the blades standing *in* the water and this is about the
     ground they are walking out of, which reaches further inland than
     the lake does. */
  albedo *= mix(1.0, 0.58, smoothstep(-0.55, 0.0, waterDepth(base)));

  /* The blade takes the weather the soil under it took, out of the same
     function — a rained-on meadow whose grass stayed bright over dark
     wet ground is the specific wrongness this prevents. Snow is fed in
     at a fraction of the ground's: what is left standing above a drift
     is the part the snow did not reach. */
  float bladeRough = 0.9;
  albedo = weatherSurface(albedo, snow * 0.35, bladeRough);

  emit(p, shadeBlade(n, albedo, sun, mix(0.50, 1.05, f), TRANSMIT));
}
`;

/* ── flowers ──────────────────────────────────────────────────────── */

const VERT_FLOWER = /* glsl */`
${PRECISION}
${CONSTANTS}
${HASH}
${ROTATE}
${SIMPLEX3}
${CLUSTER_UNIFORMS}
${CLUSTER_FIELD}
${CLUSTER_LAYERS}
${CLUSTER_SHADOW}
${CAT_PROXY_GLSL}
${SKY}
${GROUND_COMMON}

uniform float uShadowSoft;

/* Where this flower stands, decided once and kept.
   The grass can afford to re-derive itself from a hash every frame:
   two hash33 calls, and there are only seven vertices to pay them
   across. A flower is forty-eight vertices, and reaching one of them
   meant running the whole chain — does this cell grow a clump, where is
   its centre, how wide is it, is this one of the ones thinned away —
   forty-eight times over, for an answer that is the same every frame
   and the same for every vertex. It is sown on the CPU instead, when
   the patch moves, and read from here. */
layout(location = 0) in vec4 aPlant;  // xy = where it stands, z = head, w = stem
layout(location = 1) in vec4 aForm;   // x = lean heading, y = lean, z = petal spin
layout(location = 2) in vec3 aTint;   // petal colour — one species per clump

const vec3 STEM_COL = vec3(0.045, 0.115, 0.030);
const vec3 EYE_COL  = vec3(0.640, 0.360, 0.030);

void main() {
  vColor = vec3(0.0);
  vRound = vec3(0.0);
  vDist = 0.0;

  vec2 pos = aPlant.xy;

  /* The one thing that cannot be sown ahead: the fade is measured from
     wherever the eye is *now*, and the eye moves continuously while the
     sowing only happens when the patch shifts a whole cell. */
  float rim = 1.0 - smoothstep(uRadius * 0.72, uRadius, length(pos - uViewer));
  if (rim <= 0.002) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }

  float head = aPlant.z * rim;      // petal reach
  float stemH = aPlant.w * rim;

  vec2 own = vec2(cos(aForm.x), sin(aForm.x));
  float lean = aForm.y;
  float spin = aForm.z;

  /* Stiffer than a blade of grass and pushed by the same gust: a stem
     carrying a head has more to hold up and less area to catch. Reading
     the identical field is the point — the flowers have to lean with the
     grass around them or they look like they are in a different room. */
  float push = uWind * (0.12 + 0.88 * gust(pos)) * 0.62;
  vec2 bendDir = normalize(own * lean + WIND_DIR * push);
  float bend = min(lean + push, 1.05);

  /* Sown on the CPU, but only in xz — the height is looked up here.
     Deliberately: the sowing runs once every few metres walked and the
     ground under a flower never changes, so storing it would be one more
     float per record and one more thing that can disagree with the
     shader that draws the soil. */
  vec2 slope;
  vec3 root = vec3(pos.x, terrainAt(pos, slope), pos.y);
  vec3 sward = normalize(vec3(-slope.x, 1.0, -slope.y));

  vec2 perp = vec2(-bendDir.y, bendDir.x);
  vec3 across = vec3(perp.x, 0.0, perp.y);

  vec3 tipT;
  vec3 crown = stalk(root, stemH, 1.0, bendDir, bend, tipT);

  int vid = gl_VertexID;
  vec3 p, n, albedo;
  float occ = 1.0, transmit = TRANSMIT;

  if (vid < 12) {
    // Stem: two quads up the curve.
    int q = vid / 6, c = vid % 6;
    float up = (c == 2 || c == 4 || c == 5) ? 1.0 : 0.0;
    float side = (c == 1 || c == 2 || c == 4) ? 1.0 : -1.0;
    float f = (float(q) + up) * 0.5;

    vec3 tangent;
    p = stalk(root, stemH, f, bendDir, bend, tangent);
    p += across * (0.0080 * rim * (1.0 - f * 0.3) * side);

    vec3 face = normalize(cross(across, tangent));
    float sgn = dot(face, uCamPos - root) < 0.0 ? -1.0 : 1.0;
    n = normalize(face * sgn + across * (side * 0.9)
               + sward * (SWARD_BIAS * 0.5));
    albedo = STEM_COL;
    occ = mix(0.55, 1.05, f);
  } else {
    /* The head sits on the end of the stem and tilts with it, so the
       petals turn to face wherever the wind has left the stalk pointing.
       A head welded upright is the tell that a flower is a billboard. */
    vec3 T = tipT;
    vec3 ref = abs(T.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 A = normalize(cross(ref, T));
    vec3 B = cross(T, A);

    int rest = vid - 12;
    if (rest < PETALS * 6) {
      int pi = rest / 6;
      int c = rest % 6;
      float out1 = (c == 2 || c == 4 || c == 5) ? 1.0 : 0.0;
      float side = (c == 1 || c == 2 || c == 4) ? 1.0 : -1.0;

      float th = (float(pi) + 0.5) / float(PETALS) * PI * 2.0 + spin;
      /* Broad for nearly the whole span between neighbours, narrowing
         only a little toward the end. Tapered petals are the prettier
         drawing and the wrong one: five of them leave five wedges of
         gap, and what a flower reads as at twenty pixels is a disc of
         colour with notches in it, not a star. */
      float hw = PI / float(PETALS) * mix(1.0, 0.78, out1);
      float a2 = th + side * hw;
      float r = mix(head * 0.26, head, out1);
      float cup = head * 0.34 * out1 * out1;

      vec3 radial = A * cos(a2) + B * sin(a2);
      p = crown + radial * r + T * cup;

      // Perpendicular to the petal's own outward slope, in the plane
      // that slope lies in. Constant across the petal, which is what
      // keeps five of them from looking like five separate objects.
      vec3 mid = A * cos(th) + B * sin(th);
      n = normalize(T * (head - head * 0.26) - mid * (head * 0.34));
      albedo = aTint;
      // Petals are thinner than grass and glow harder from behind.
      transmit = 0.95;
    } else {
      int c = rest - PETALS * 6;
      float ux = (c == 1 || c == 2 || c == 4) ? 1.0 : -1.0;
      float uy = (c == 2 || c == 4 || c == 5) ? 1.0 : -1.0;
      float r = head * 0.30;
      p = crown + (A * ux + B * uy) * r + T * (head * 0.10);
      n = T;
      albedo = EYE_COL;
      transmit = 0.0;
      // The eye is a square in geometry and a disc on screen; the corners
      // are thrown away in the fragment shader.
      vRound = vec3(ux, uy, 1.0);
    }
  }

  float sun = sunlight(root + vec3(0.0, stemH * 0.6, 0.0), uShadowSoft);
  if (dot(n, uCamPos - p) < 0.0 && vRound.z < 0.5) n = -n;

  emit(p, shadeBlade(n, albedo, sun, occ, transmit));
}
`;

/* ── reeds ────────────────────────────────────────────────────────── */

const VERT_REED = /* glsl */`
${PRECISION}
${CONSTANTS}
${HASH}
${ROTATE}
${SIMPLEX3}
${CLUSTER_UNIFORMS}
${CLUSTER_FIELD}
${CLUSTER_LAYERS}
${CLUSTER_SHADOW}
${CAT_PROXY_GLSL}
${SKY}
${GROUND_COMMON}

#define REED_ROWS ${REED_ROWS.length}
#define REED_H_REF ${REED_H_REF.toFixed(4)}

uniform float uShadowSoft;

/* Sown on the CPU, exactly as the flowers are and for the same reason:
   where the waterline runs, which way to step to reach it and how deep
   this one ended up are one answer for all thirteen vertices and the
   same answer every frame. */
layout(location = 0) in vec4 aReed;   // xy = where it stands, z = height, w = lean heading
layout(location = 1) in vec4 aForm;   // x = lean, y = half-width, z = head, w = how dry

const float ROW_F[REED_ROWS] = float[REED_ROWS](${REED_ROWS.map(f => f.toFixed(3)).join(', ')});

/* Greyer and paler than the meadow behind it. A reed bed authored in the
   grass's own green vanishes into it — and what says "the water is here"
   from thirty metres away, before any single stalk can be resolved, is
   that the band standing along that line is a different colour from the
   field it interrupts. */
const vec3 REED_COL = vec3(0.105, 0.140, 0.052);
const vec3 REED_DRY = vec3(0.300, 0.245, 0.095);
const vec3 HEAD_COL = vec3(0.150, 0.078, 0.030);

void main() {
  vColor = vec3(0.0);
  vRound = vec3(0.0);
  vDist = 0.0;

  vec2 pos = aReed.xy;

  /* Measured from where the eye is now, not from where the sowing was
     centred — the one thing about a reed that cannot be settled on the
     CPU, for the same reason it cannot be for a flower. */
  float rim = 1.0 - smoothstep(uRadius * 0.72, uRadius, length(pos - uViewer));
  if (rim <= 0.002) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }

  float height = aReed.z * rim;

  /* Stiffer than a blade and far lighter than a trunk, and a tall one
     takes more of the gust than a short one — which is most of why a bed
     reads as one surface travelling across rather than as a row of
     sticks. The field is the one the grass in front of it is reading.

     The travel is deliberately small. A reed is a cane: it *sways*, and
     the first version let it fold to sixty degrees, which photographed
     as a bank of fishing rods being cast. What says "reed" is a nearly
     vertical stroke that is not quite vertical and not quite still. */
  float push = uWind * (0.10 + 0.90 * gust(pos)) * 0.42 * (aReed.z / REED_H_REF);
  vec2 own = vec2(cos(aReed.w), sin(aReed.w));
  vec2 bendDir = normalize(own * aForm.x + WIND_DIR * push);
  float bend = min(aForm.x + push * 0.85, 0.80);

  vec2 slope;
  vec3 root = vec3(pos.x, terrainAt(pos, slope), pos.y);
  vec3 sward = normalize(vec3(-slope.x, 1.0, -slope.y));

  int vid = gl_VertexID;
  bool tip = vid >= REED_ROWS * 2;
  int row = min(vid / 2, REED_ROWS - 1);
  float f = tip ? 1.0 : ROW_F[row];
  float side = tip ? 0.0 : float(vid & 1) * 2.0 - 1.0;

  vec3 tangent;
  vec3 p = stalk(root, height, f, bendDir, bend, tangent);

  /* The head is a width, not a second object. A cattail is a stalk that
     swells for twenty centimetres near the top and thins again above it,
     so a profile is the honest description of one — and being part of
     the stalk it bends with it for free, which a head welded to the tip
     does not. */
  float spike = smoothstep(0.66, 0.745, f) * (1.0 - smoothstep(0.875, 0.955, f));
  float halfW = (aForm.y * (1.0 - f * 0.35) + aForm.z * spike) * rim;

  vec2 perp = vec2(-bendDir.y, bendDir.x);
  vec3 across = vec3(perp.x, 0.0, perp.y);
  p += across * (halfW * side);

  /* Biased toward the ground almost as hard as the grass is, and the
     first version was not. Standing a reed's normal up on its own edge
     is defensible on paper — a bed really is vertical, unlike a lawn —
     and it photographed as a row of charred wires along the shore, which
     is the identical failure the grass had before SWARD_BIAS existed:
     with the sun anywhere but straight down the sunlit side of a
     vertical ribbon is not facing anyone. What the eye reads at this
     range is a *mass* standing in the water, and a mass is lit from
     above. Held slightly under the grass's, so the bed keeps a little
     more of its own shading than the sward around it. */
  vec3 face = normalize(cross(across, tangent));
  float sgn = dot(face, uCamPos - root) < 0.0 ? -1.0 : 1.0;
  vec3 n = normalize(face * sgn + across * (side * 0.55)
                   + sward * (SWARD_BIAS * 0.85));

  /* Winter, not snowfall. These stand in the one place snowCover is held
     to zero — the shore term, without which the lake wears a white rim
     floating on its first half metre — so a reed asking how much snow
     was lying on it would be told "none" in the middle of a blizzard and
     stay summer green. What it should be reading is the season, and
     uSnow is the number that says there is one. */
  vec3 albedo = mix(REED_COL, REED_DRY, max(aForm.w, uSnow * 0.85));
  albedo = mix(albedo, HEAD_COL, smoothstep(0.12, 0.55, spike));
  float rough = 0.85;
  albedo = weatherSurface(albedo, 0.0, rough);

  /* Asked once, half way up, exactly as a blade asks: the shadow is the
     expensive term and it cannot resolve a stalk, so thirteen queries
     would buy thirteen identical answers — or, worse, a reed lit
     unevenly along its own length. */
  float sun = sunlight(root + vec3(0.0, height * 0.55, 0.0), uShadowSoft);
  // Less buried in its neighbours than a blade is: the bed is open at
  // the bottom and standing in water, not packed into a sward.
  emit(p, shadeBlade(n, albedo, sun, mix(0.55, 1.10, f), TRANSMIT));
}
`;

/* ── the object ───────────────────────────────────────────────────── */

/** Floor styles, in the order the picker offers them. */
export const GROUND_STYLES = ['grid', 'grass'];

/** True when this style wants soil under it instead of the grid. */
export function isCovered(style) {
  return style === 'grass';
}

/* ── sowing ───────────────────────────────────────────────────────
   The flowers' placement, on the CPU, once.

   Deterministic in the world coordinate of the cell, exactly as the
   grass's hash is, and for the same reason: the patch slides under the
   camera and the flowers must not slide with it. What changed is only
   *when* it is evaluated — the grid moves a whole cell at a time, so
   this runs on the order of once every few metres walked instead of
   forty-eight times per flower per frame.

   Written out in plain loops rather than as a hash chain, which is the
   other half of the win: "does this cell grow a clump, and how many
   flowers does it get" is a nested loop in any language that has them,
   and it was only ever an arithmetic puzzle because a vertex shader has
   no way to say it. */

/** One deterministic value in [0,1) from a cell and a salt. */
/** Cell offsets out to a span, ordered by distance. Cached: the span
    changes only when the reach does. */
const ORDERS = new Map();
function cellOrder(span) {
  let o = ORDERS.get(span);
  if (o) return o;
  const cells = [];
  for (let dz = -span; dz <= span; dz++) {
    for (let dx = -span; dx <= span; dx++) cells.push([dx, dz]);
  }
  cells.sort((a, b) => (a[0] * a[0] + a[1] * a[1]) - (b[0] * b[0] + b[1] * b[1]));
  o = new Int32Array(cells.length * 2);
  cells.forEach(([dx, dz], i) => { o[i * 2] = dx; o[i * 2 + 1] = dz; });
  ORDERS.set(span, o);
  return o;
}

/** A clump is one species, so the colour is drawn once per clump. */
const PETAL_COLOURS = [
  [0.780, 0.760, 0.700],   // white
  [0.880, 0.480, 0.020],   // yellow
  [0.820, 0.175, 0.290],   // pink
  [0.330, 0.170, 0.700],   // lilac
  [0.880, 0.105, 0.025],   // orange-red
];

export class GroundCover {
  constructor(gl) {
    this.gl = gl;
    this.grass = new Program(gl, VERT_GRASS, FRAG_PLANT, { name: 'ground/grass' });
    this.flower = new Program(gl, VERT_FLOWER, FRAG_PLANT, { name: 'ground/flower' });
    this.reed = new Program(gl, VERT_REED, FRAG_PLANT, { name: 'ground/reed' });

    /* Nothing is fetched per vertex for the grass, but something has to
       be bound: a leftover attribute array from another draw would be
       read for seven vertices that do not want it. An empty one says so. */
    this.grassVao = gl.createVertexArray();

    // The flowers do fetch, one record per instance.
    this.flowerVao = gl.createVertexArray();
    this.flowerVbo = gl.createBuffer();
    this._sown = new Float32Array(MAX_FLOWERS * FLOWER_STRIDE);

    gl.bindVertexArray(this.flowerVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.flowerVbo);
    gl.bufferData(gl.ARRAY_BUFFER, this._sown.byteLength, gl.DYNAMIC_DRAW);
    const stride = FLOWER_STRIDE * 4;
    for (const [loc, size, offset] of [[0, 4, 0], [1, 4, 16], [2, 3, 32]]) {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, offset);
      gl.vertexAttribDivisor(loc, 1);          // one record per flower
    }
    // And the reeds, on the same scheme with a shorter record.
    this.reedVao = gl.createVertexArray();
    this.reedVbo = gl.createBuffer();
    this._reedSown = new Float32Array(MAX_REEDS * REED_STRIDE);

    gl.bindVertexArray(this.reedVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.reedVbo);
    gl.bufferData(gl.ARRAY_BUFFER, this._reedSown.byteLength, gl.DYNAMIC_DRAW);
    for (const [loc, size, offset] of [[0, 4, 0], [1, 4, 16]]) {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, REED_STRIDE * 4, offset);
      gl.vertexAttribDivisor(loc, 1);
    }

    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    /** The lattice the patch origin is snapped to. Published because it
        is the whole reason the grass stays put, and a claim that cannot
        be checked from outside is a claim nobody will notice breaking. */
    this.cell = CELL;
    this._patch = new Float32Array(2);
    this._viewer = new Float32Array(2);
    this._jitter = new Float32Array(2);
    /** Where the sown buffer was centred, and what it was sown for. */
    this._sowKey = null;
    /** The same, for the shoreline — a different grid and a different
        set of answers, so a different key. */
    this._reedKey = null;
    /** Reused, so a per-ring draw allocates nothing. */
    this._ring = { uRing: 0, uBladeVerts: BLADE_VERTS, uShadowSoft: 0, uCanopyOn: 0 };
    /** How many times the flowers have been re-sown. Watched by the tests:
        a number that climbs while the camera stands still is the bug this
        replaced. */
    this.sowings = 0;
    this.blades = 0;
    this.flowers = 0;
    this.reeds = 0;
    /** How far each kind actually reaches. One control sets all of them;
        each keeps its own ratio and its own ceiling. */
    this.flowerRadius = 0;
    this.reedRadius = 0;
    this.triangles = 0;
  }

  /**
   * Fill the instance buffer with every flower within reach of a snapped
   * origin, and upload it. Called only when that origin, the clump size
   * or the density changes.
   *
   * The drowned ones are dropped here rather than shortened away in the
   * shader as the grass is, and the asymmetry is the difference between
   * where the two are placed. A blade's position is derived in the vertex
   * shader from the cell it stands in, so there is no earlier moment at
   * which it could be left out; a flower is sown on the CPU, so a flower
   * in the lake is an instance record, a draw and a stem's worth of
   * triangles that can simply never be written. The taper the grass gets
   * is also the wrong look here — a flower is one object, and half a
   * flower standing in the water is a flower cut in half.
   */
  _sow(originX, originZ, clumpCell, perClump, reach, chance, spread,
       hills, surfaceY, snow) {
    /* Scratch for the slope the snow cover needs, hoisted out of a loop
       that runs tens of thousands of times. */
    const grad = this._grad || (this._grad = [0, 0]);
    const buf = this._sown;
    /* One cell of margin past the reach: the fade is measured from where
       the eye is now, which can be half a cell beyond the origin this was
       sown around, and a flower missing from the buffer cannot fade in.

       Plus however far a clump can throw a flower. A wide clump seeded
       just outside the visited cells still drops flowers inside them, and
       leaving those out thins a ring at the rim — invisible at spread 1,
       obvious once the clumps are metres across. */
    const bleed = clumpCell + CLUMP_R_MAX * spread;
    const span = Math.ceil((reach + bleed) / clumpCell);
    const ix0 = Math.round(originX / clumpCell);
    const iz0 = Math.round(originZ / clumpCell);
    const limit = (reach + clumpCell) * (reach + clumpCell);

    /* Nearest cell first. The buffer has a ceiling, and if it fills, what
       should be lost is the far side of the field in every direction —
       not, as a row-major scan would give, everything past one line. */
    const order = cellOrder(span);

    let n = 0;
    for (let o = 0; o < order.length && n < MAX_FLOWERS; o += 2) {
      {
        const ix = ix0 + order[o], iz = iz0 + order[o + 1];
        if (seed(ix, iz, 1) > chance) continue;         // no clump here

        const cx = (ix + seed(ix, iz, 2) * 0.72 + 0.14) * clumpCell;
        const cz = (iz + seed(ix, iz, 3) * 0.72 + 0.14) * clumpCell;
        const clumpR = (CLUMP_R_MIN
          + seed(ix, iz, 4) * (CLUMP_R_MAX - CLUMP_R_MIN)) * spread;
        const tint = PETAL_COLOURS[Math.min(4, (seed(ix, iz, 5) * 5) | 0)];

        for (let k = 0; k < perClump && n < MAX_FLOWERS; k++) {
          const s = k * 8 + 16;
          /* Radius as the raw value rather than its square root. Uniform
             area density would give a disc; this piles them into the
             middle, which is what anything self-seeding looks like. */
          const u = seed(ix, iz, s);
          /* And thinned toward the rim, so the clump has no edge — a few
             stragglers standing on their own in the grass around it,
             which is what stops it reading as a planted bed. */
          if (seed(ix, iz, s + 1) < u * u * 0.92) continue;

          const ang = seed(ix, iz, s + 2) * Math.PI * 2;
          const x = cx + Math.cos(ang) * clumpR * u;
          const z = cz + Math.sin(ang) * clumpR * u;

          const rx = x - originX, rz = z - originZ;
          if (rx * rx + rz * rz > limit) continue;

          // Drowned. Tested per flower and not per clump: a clump that
          // straddles the shore should lose the half that is in the
          // water, not all of itself or none of it.
          if (waterDepthAt(x, z, hills, surfaceY) > 0) continue;

          /* And buried. A threshold rather than the taper the grass
             gets, for the same reason the drowning is: a flower is one
             object, and the top half of one sticking out of a drift is
             a flower with its stem cut off. The line sits well short of
             full cover, so the blooms go first from the deep drifts and
             last from the scoured patches — which is where the mottle
             in snowCover finally shows up as something other than a
             shade of white. */
          if (snowCoverAt(x, z, hills, surfaceY, snow, terrainHeight, grad) > 0.45) {
            continue;
          }

          const o = n * FLOWER_STRIDE;
          buf[o] = x;
          buf[o + 1] = z;
          buf[o + 2] = 0.055 + seed(ix, iz, s + 3) * 0.040;   // petal reach
          /* Taller than the grass around it, deliberately and by a clear
             margin. A flower whose head sits at blade height is one
             nobody will ever see: the sward closes over it from every
             angle but straight down. */
          buf[o + 3] = BLADE_H * (1.15 + seed(ix, iz, s + 4) * 0.60);
          buf[o + 4] = seed(ix, iz, s + 5) * Math.PI * 2;     // lean heading
          buf[o + 5] = 0.04 + seed(ix, iz, s + 6) * 0.18;     // lean
          buf[o + 6] = seed(ix, iz, s + 7) * Math.PI * 2;     // petal spin
          buf[o + 7] = 0;
          buf[o + 8] = tint[0];
          buf[o + 9] = tint[1];
          buf[o + 10] = tint[2];
          n++;
        }
      }
    }

    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.flowerVbo);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, buf, 0, n * FLOWER_STRIDE);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    this.sowings++;
    return n;
  }

  /**
   * Fill the reed buffer: every cell near enough the shore to be walked
   * onto it, kept if the walk landed inside the band.
   *
   * The other two plantings are a placement followed by a rejection —
   * scatter a seed, drop it if it drowned. This one is the reverse: the
   * cell is *aimed* at a depth and the rejection is what happens when
   * there is no shoreline within reach of a step. That difference is the
   * entire reason a band a few tens of centimetres wide can be populated
   * at all; scattering into it and throwing away the misses would spend
   * a thousand rejections per reed and still leave gaps wherever the
   * bank ran diagonally across a cell.
   *
   * @param {number} originX  the snapped patch centre
   * @param {number} originZ
   * @param {number} reach    how far out to look
   * @param {number} chance   how many cells grow one
   * @param {number} hills    the hill amplitude
   * @param {number} surfaceY where the water stands, from waterSurfaceY
   */
  _sowReeds(originX, originZ, reach, chance, hills, surfaceY) {
    const buf = this._reedSown;
    const grad = this._reedGrad || (this._reedGrad = [0, 0]);

    /* Off is not a flag here either. Water standing below the lowest
       ground the hills can reach is water no cell can be in the band of
       — the same honest test hasWater() makes in the shader — and it is
       here only to skip a scan that could not have found anything. */
    if (!(surfaceY > TERRAIN_BASE - hills)) return 0;

    /* A margin of one snap step past the reach, because the eye is
       already up to that far from the origin this was sown around and a
       reed missing from the buffer cannot fade in. */
    const span = Math.ceil((reach + REED_SOW_STEP) / REED_CELL);
    const ix0 = Math.round(originX / REED_CELL);
    const iz0 = Math.round(originZ / REED_CELL);
    const edge = reach + REED_SOW_STEP;
    const limit = edge * edge;
    /* How far a cell is allowed to walk. A shore further away than this
       belongs to a cell nearer to it, and letting them all chase it
       would pile every reed on the same waterline and leave the cells
       behind them bare. */
    const cap = REED_CELL * 1.6;
    const order = cellOrder(span);

    let n = 0;
    for (let o = 0; o < order.length && n < MAX_REEDS; o += 2) {
      const ix = ix0 + order[o], iz = iz0 + order[o + 1];
      /* Thicker in some stretches of shore than others. A fringe at one
         even spacing all the way round a lake is a fence; what a bed
         does is fill a bay and leave the next one open. */
      const stand = 0.25 + 1.35 * seed(
        Math.floor(ix / REED_STAND), Math.floor(iz / REED_STAND), 22);
      if (seed(ix, iz, 11) > chance * stand) continue;

      let x = (ix + seed(ix, iz, 12) * 0.90 + 0.05) * REED_CELL;
      let z = (iz + seed(ix, iz, 13) * 0.90 + 0.05) * REED_CELL;
      // The depth this one wants to stand in, anywhere across the band.
      const want = -REED_ABOVE + (REED_ABOVE + REED_BELOW) * seed(ix, iz, 14);

      /* Two Newton steps onto that depth. Depth is surfaceY - h, so its
         gradient is the height field's negated — and the field hands
         that back from the same evaluation that gave the height, which
         is the whole reason the ground is a sum of sines.

         The step runs along the gradient, which is across the shoreline;
         nothing moves it along the shore. So neighbouring cells keep
         their spacing where it matters and cannot collapse onto one
         point. A hollow with no slope in it divides by nothing and comes
         out infinite, which fails the same test that rejects a shore too
         far to reach — no branch for the flat case. */
      let depth = surfaceY - terrainHeight(x, z, hills, grad);
      for (let it = 0; it < 2 && Math.abs(want - depth) > 0.02; it++) {
        const miss = want - depth;
        const g2 = grad[0] * grad[0] + grad[1] * grad[1];
        const sx = -miss * grad[0] / g2, sz = -miss * grad[1] / g2;
        if (!(sx * sx + sz * sz < cap * cap)) { depth = -1e9; break; }
        x += sx; z += sz;
        depth = surfaceY - terrainHeight(x, z, hills, grad);
      }
      // The band, tested where it actually ended up rather than where it
      // was aimed: two steps do not always land on a curved field.
      if (!(depth > -REED_ABOVE && depth < REED_BELOW)) continue;

      const rx = x - originX, rz = z - originZ;
      if (rx * rx + rz * rz > limit) continue;

      /* The ones standing in the water are the tall ones. Not decoration:
         it is the order a real bed grows in, and it is what keeps the
         band from having a flat top cut across it. */
      const wet = (depth + REED_ABOVE) / (REED_ABOVE + REED_BELOW);

      const rec = n * REED_STRIDE;
      buf[rec] = x;
      buf[rec + 1] = z;
      buf[rec + 2] = (REED_H_MIN + seed(ix, iz, 15) * (REED_H_MAX - REED_H_MIN))
                   * (0.78 + 0.44 * wet);
      buf[rec + 3] = seed(ix, iz, 16) * Math.PI * 2;     // lean heading
      buf[rec + 4] = 0.05 + seed(ix, iz, 17) * 0.15;     // lean
      buf[rec + 5] = 0.0055 + seed(ix, iz, 18) * 0.0035; // half-width
      /* A head or none, and never a small one: a cattail that has not
         swelled is a rush, and both grow here. Interpolating between
         them would give a stalk with a bulge in it, which is neither. */
      buf[rec + 6] = seed(ix, iz, 19) < HEAD_CHANCE
        ? 0.0090 + seed(ix, iz, 20) * 0.0045 : 0.0;
      buf[rec + 7] = seed(ix, iz, 21) * 0.28;            // how dry
      n++;
    }

    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.reedVbo);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, buf, 0, n * REED_STRIDE);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    return n;
  }

  /**
   * Draw into the currently bound target, which must be the one the cat
   * draws into: same depth attachment, same camera, same alpha
   * convention.
   *
   * @param {object} camera  pos/right/up/fwd/focal/aspect/width/height
   * @param {object} env     the scene's light and its cluster field
   * @param {object} opts    style, density, flowers, reeds, wind, radius,
   *                          frame — see draw() in march.js for the full set
   */
  draw(camera, env, opts) {
    if (!isCovered(opts.style)) {
      this.blades = 0; this.flowers = 0; this.reeds = 0; this.triangles = 0;
      return;
    }
    const gl = this.gl;

    const { rings, step, radius } = coverPlan(opts.radius);
    /* Ring 0's cell, which is a constant. Published, because "the reach
       control does not resize the plants" is the claim this whole scheme
       exists to make good on, and it should be checkable. */
    this.cell = CELL;
    this.radius = radius;
    this.rings = rings;

    /* Centred a little ahead of the camera rather than under it. Half a
       patch centred on the lens is spent behind the viewer; pushing it
       down the view axis spends the same budget on ground that is
       actually in frame. */
    const ahead = Math.min(radius, HALF0) * 0.35;
    const cx = camera.pos[0] + camera.fwd[0] * ahead;
    const cz = camera.pos[2] + camera.fwd[2] * ahead;
    this._viewer[0] = cx;
    this._viewer[1] = cz;

    /* Ring 0's origin, snapped, so a cell keeps its world coordinate and
       therefore its hash while the grid slides underneath. The shader
       derives this and every coarser ring's origin the same way from
       uViewer; this copy is what the flower sowing is keyed on. */
    this._patch[0] = Math.round(cx / CELL) * CELL;
    this._patch[1] = Math.round(cz / CELL) * CELL;

    // The cat's jitter, for the same reason: this is the same target and
    // the same temporal filter resolves both.
    this._jitter[0] = Math.sin(opts.frame * 2.39996) / camera.width;
    this._jitter[1] = Math.sin(opts.frame * 4.10000 + 1.7) / camera.height;

    const density = Math.max(0, Math.min(1, opts.density));
    const layers = Math.max(1, Math.round(density * MAX_LAYERS));

    /* Everything every plant needs, from the one place that knows what
       that is, plus the rings — which are this pass's alone. */
    const common = {
      ...plantUniforms(camera, env, { wind: opts.wind, jitter: this._jitter }),

      uViewer: this._viewer,
      uRadius: radius,
      uGrid: GRID,
      uCell: CELL,
      uLayers: layers,
      uRings: rings,
      uRingStep: step,
    };

    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
    gl.depthMask(true);
    // Both faces of a blade are the blade. The normal is turned toward
    // the eye in the vertex shader instead.
    gl.disable(gl.CULL_FACE);

    gl.bindVertexArray(this.grassVao);

    /* A draw per ring rather than one for all of them. Rings differ in
       what they can afford to be — how many vertices a blade is worth
       out there, and whether it is worth asking about shadow at all —
       and none of that can vary inside a single draw. Five calls buys
       the whole of it.

       Every ring still gets a full square of instances; the ones over
       the hole park themselves in four instructions. */
    const perRing = GRID * GRID * layers;
    this.blades = perRing * rings;
    this.triangles = 0;

    /* Everything that does not vary between rings goes up once. Handing
       the whole set to each draw is the obvious way to write this loop
       and it measured *slower* than the single draw it replaced: the
       common block carries the cluster's spheres, four impacts, eight cat
       capsules and the camera, and re-uploading all of it five times a
       frame cost more than the vertices the rings were saving. */
    this.grass.use(common);
    const canopyOn = common.uCanopyOn || 0;
    for (let r = 0; r < rings; r++) {
      const verts = r === 0 ? BLADE_VERTS : BLADE_VERTS_FAR;
      const lit = r < SHADOW_RINGS;
      this._ring.uRing = r;
      this._ring.uBladeVerts = verts;
      // Both shadow terms read a zero as "ask nothing" and return in one
      // compare, so switching them off needs no branch in the shader and
      // no second program.
      this._ring.uShadowSoft = lit ? env.shadowSoft : 0;
      this._ring.uCanopyOn = lit ? canopyOn : 0;
      this.grass.use(this._ring);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, verts, perRing);
      this.triangles += perRing * (verts - 2);
    }

    if (opts.flowers) {
      /* Fixed, like ring 0's cell and for the same reason: winding the
         reach out must not thin the flowers underfoot.

         What is capped instead is how far they go. A blade at the rim
         still contributes the far field's colour; a flower head out
         there is a fraction of a pixel, so past FLOWER_MAX there is
         nothing to buy. */
      const clumpCell = CLUMP_CELL;
      const reach = Math.min(radius * FLOWER_REACH, FLOWER_MAX);
      this.flowerRadius = reach;
      const perClump = Math.max(3, Math.round(
        Math.max(0, Math.min(1, opts.flowerDensity)) * MAX_PER_CLUMP));
      const chance = Math.max(0, Math.min(1, opts.flowerClumps));
      const spread = Math.max(0.05, opts.flowerSpread);

      /* Sown when — and only when — the answer would differ. The origin
         moves a whole clump cell at a time, so walking a straight line
         re-sows every few metres and standing still never does. */
      const ox = Math.round(this._patch[0] / clumpCell) * clumpCell;
      const oz = Math.round(this._patch[1] / clumpCell) * clumpCell;
      const key = `${ox}|${oz}|${clumpCell.toFixed(4)}|${perClump}`
        + `|${chance.toFixed(3)}|${spread.toFixed(3)}|${reach.toFixed(3)}`
        /* The ground and the water on it are part of the answer now:
           raise the hills or the level and a different set of flowers
           has drowned, which is a different buffer and not merely a
           different way of drawing the same one. */
        + `|${env.hills.toFixed(3)}|${env.waterY.toFixed(3)}`
        // And the snow, which buries a different set of them.
        + `|${env.weather.snow.toFixed(3)}`;
      if (key !== this._sowKey) {
        this.flowers = this._sow(ox, oz, clumpCell, perClump, reach, chance,
                                 spread, env.hills, env.waterY,
                                 env.weather.snow);
        this._sowKey = key;
      }

      if (this.flowers) {
        gl.bindVertexArray(this.flowerVao);
        /* Its own, shorter reach — see FLOWER_REACH — and its own shadow
         terms, because the loop above left the last ring's stripped ones
           bound, and every flower is well inside where shadow still shows. */
        this.flower.use({ ...common, uRadius: reach, uShadowSoft: env.shadowSoft });
        gl.drawArraysInstanced(gl.TRIANGLES, 0, FLOWER_VERTS, this.flowers);
        this.triangles += this.flowers * (FLOWER_VERTS / 3);
      }
    } else {
      this.flowers = 0;
      // Forget where they were sown, or coming back to the meadow after
      // walking away shows the ones that were around the old spot.
      this._sowKey = null;
    }

    if (opts.reeds) {
      /* Its own reach off the same control, and its own grid. The
         shoreline is not where the flowers are, so there is nothing to
         share with them but the height field. */
      const reach = Math.min(radius * REED_REACH, REED_MAX);
      this.reedRadius = reach;
      const chance = Math.max(0, Math.min(1, opts.reedDensity));

      const ox = Math.round(this._patch[0] / REED_SOW_STEP) * REED_SOW_STEP;
      const oz = Math.round(this._patch[1] / REED_SOW_STEP) * REED_SOW_STEP;
      /* The ground and the level are in the key for a stronger reason
         than they are in the flowers'. Move either and the shoreline
         itself has moved, so it is not that a different set of reeds
         drowned — it is that the band they stand in is somewhere else.
         The weather is not in it: nothing buries a stalk this tall, and
         what winter does to one is a colour the shader reads. */
      const key = `${ox}|${oz}|${chance.toFixed(3)}|${reach.toFixed(3)}`
        + `|${env.hills.toFixed(3)}|${env.waterY.toFixed(3)}`;
      if (key !== this._reedKey) {
        this.reeds = this._sowReeds(ox, oz, reach, chance, env.hills, env.waterY);
        this._reedKey = key;
      }

      if (this.reeds) {
        gl.bindVertexArray(this.reedVao);
        this.reed.use({ ...common, uRadius: reach, uShadowSoft: env.shadowSoft });
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, REED_VERTS, this.reeds);
        this.triangles += this.reeds * (REED_VERTS - 2);
      }
    } else {
      this.reeds = 0;
      this._reedKey = null;
    }

    gl.bindVertexArray(null);
    gl.disable(gl.DEPTH_TEST);
  }

  dispose() {
    this.grass.dispose();
    this.flower.dispose();
    this.reed.dispose();
    this.gl.deleteVertexArray(this.grassVao);
    this.gl.deleteVertexArray(this.flowerVao);
    this.gl.deleteBuffer(this.flowerVbo);
    this.gl.deleteVertexArray(this.reedVao);
    this.gl.deleteBuffer(this.reedVbo);
  }
}
