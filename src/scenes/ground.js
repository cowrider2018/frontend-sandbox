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
import { RASTER_NEAR, RASTER_FAR } from './raster.js';

/* ── the grid ─────────────────────────────────────────────────────
   How far the cover reaches is a control, because the two things people
   want from it pull opposite ways: an unbroken plain out to the fog, or
   the frame rate back.

   The cell size is not held constant across that range. Fixed cells make
   the instance count grow with the square of the reach, and doubling the
   view distance for four times the cost is not a trade anyone would take
   twice. Growing the cell as the square root of the reach makes the cost
   grow *linearly* instead, and it thins the far field rather than the
   near one — which is the right place to lose blades, since that is
   where they are smallest on screen. */
const RADIUS_REF = 16.0;
const CELL_REF = 0.5;

/** Reach, in world units, at each end of the control. */
export const RADIUS_MIN = 7.0;
export const RADIUS_MAX = 44.0;

/** Blades per cell at full density. One instance each. */
const MAX_LAYERS = 6;
/** Vertices in a blade: three quads narrowing to a single tip vertex. */
const BLADE_VERTS = 7;

/** Cell size, and how many cells across, for a given reach. */
function grassGrid(radius) {
  const cell = CELL_REF * Math.sqrt(radius / RADIUS_REF);
  // Even, so the grid is symmetric about its centre.
  const grid = Math.max(8, Math.round((radius * 2) / cell / 2) * 2);
  return { cell, grid };
}

/* ── flowers ──────────────────────────────────────────────────────
   Flowers are placed by clump, not by blade. A clump cell either grows
   one or it does not, and the flowers inside it thin out toward its rim,
   so what comes out is a dense middle with singles scattered round it
   rather than a disc with an edge. */
const CLUMP_CELL_REF = 4.0;

/* Flowers stop well short of where the grass does. A blade at the rim
   still counts, because what it contributes there is the *colour* of
   the far field; a flower head at that range is under a pixel and
   contributes a speck. */
const FLOWER_REACH = 0.62;
/** How many clump cells in the patch actually grow one. */
const CLUMP_CHANCE = 0.62;
/** Flower slots per clump at full density; the thinning eats about 30%. */
const MAX_PER_CLUMP = 30;
/** Ceiling on the placement buffer. Comfortably above the widest reach. */
const MAX_FLOWERS = 6144;
/** Floats per flower in that buffer; see the attribute layout below. */
const FLOWER_STRIDE = 11;

const PETALS = 5;
/** Stem 2 quads, one quad per petal, one quad for the eye. */
const FLOWER_VERTS = 12 + PETALS * 6 + 6;

/** Half-width at the root, and how tall an average blade stands. Shared,
    because a flower's stem is measured against the sward it grows out of. */
const BLADE_W = 0.013;
const BLADE_H = 0.32;

/* ── shared shader ────────────────────────────────────────────────── */

/**
 * Everything both the blades and the flowers need.
 *
 * Requires PI, rot2, snoise, hash33, the cluster field and SKY — the
 * same set the cat's fragment shader pulls in, and for the same reason:
 * whatever grows here is lit by the scene's sun, through the scene's
 * own shadow function, or it will not look like it is standing in the
 * same place as everything else.
 */
const GROUND_COMMON = /* glsl */`
#define NEAR ${RASTER_NEAR.toFixed(4)}
#define FAR ${RASTER_FAR.toFixed(1)}
#define PETALS ${PETALS}
#define BLADE_W ${BLADE_W.toFixed(4)}
#define BLADE_H ${BLADE_H.toFixed(4)}

uniform vec3  uCamPos, uRight, uUp, uFwd;
uniform float uFocal, uAspect;
uniform vec2  uJitter;

/** Grid centre, snapped to the cell size so the hashes stay put. */
uniform vec2  uPatch;
/** Where the fade is measured from — the unsnapped centre. */
uniform vec2  uViewer;
/** Where the cover fades out. The flowers are handed a shorter one. */
uniform float uRadius;
uniform float uWind;
uniform float uFog;

out vec3 vColor;
out vec3 vRound;
out float vDist;

/** The cat's projection, term for term: the two share a depth buffer. */
vec4 rasterise(vec3 world) {
  vec3 rel = world - uCamPos;
  vec3 view = vec3(dot(rel, uRight), dot(rel, uUp), dot(rel, uFwd));
  float z = view.z * (FAR + NEAR) / (FAR - NEAR) - 2.0 * FAR * NEAR / (FAR - NEAR);
  return vec4(view.x * uFocal / uAspect + uJitter.x * view.z,
              view.y * uFocal + uJitter.y * view.z,
              z, view.z);
}

/* Which way the weather is going. Fixed, because a wind that wanders is
   a wind nobody can read: what makes a gust legible is seeing it arrive
   from the same side every time. */
const vec2 WIND_DIR = vec2(0.8829, 0.4696);

/**
 * The wind, as a scalar field over the ground.
 *
 * Two waves travelling along the same heading at different rates: a
 * slow, long one that is the gust crossing the meadow, and a short fast
 * one that is the chop inside it. Sampled at the plant's *base*, so a
 * whole clump leans together and the far side of the field is still
 * standing up when the near side has already been flattened.
 */
float gust(vec2 p) {
  float d = dot(p, WIND_DIR);
  float slow = sin(d * 0.32 - uTime * 1.10);
  float chop = sin(d * 1.55 - uTime * 2.85 + 1.7);
  return 0.5 + 0.5 * (slow * 0.66 + chop * 0.34);
}

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

/**
 * How much of the sun reaches a point, through the cluster and the cat.
 *
 * Cheap almost everywhere despite the march: the ray leaves the
 * cluster's bounding sphere immediately unless it is actually headed
 * into it, so the overwhelming majority of blades pay three dot products
 * and stop. Only the ones standing in the shadow do the walk.
 */
float sunlight(vec3 p, float soft) {
  if (soft <= 0.0) return 1.0;
  float k = mix(6.0, 26.0, soft);
  return min(clusterShadow(p, uLightDir, k), catShadow(p, uLightDir, k));
}

/* Light through a leaf. Grass is one cell thick and glows when the sun
   is behind it, and that backlight is most of what separates a meadow
   from a carpet of green spikes. */
const float TRANSMIT = 0.85;

/**
 * How far a blade's normal is turned back toward straight up.
 *
 * Not a cheat for its own sake. A blade standing vertically has a
 * horizontal normal, so with the sun anywhere overhead half the meadow
 * faces away from it and goes black — which is exactly what the first
 * version looked like, a field of charred spikes. The eye does not read
 * a lawn as a million vertical planes; it reads it as a *surface* with
 * texture on it, and that surface faces up. Biasing toward the ground
 * normal is what puts the sward back and leaves the blade shape as
 * variation across it rather than as the whole signal.
 */
const float SWARD_BIAS = 1.15;

/**
 * Finish a plant vertex: fog it, and publish its depth.
 *
 * Fogged here rather than per fragment, which is not the usual place for
 * it. The grass covers the bottom half of the screen several blades
 * deep, so a horizon term with a 220th power in it was being evaluated a
 * dozen times per pixel to shade something a few pixels tall. Across a
 * blade the answer does not measurably change.
 */
void emit(vec3 world, vec3 col) {
  vec3 toEye = uCamPos - world;
  float dist = length(toEye);
  vColor = mix(col, sky(-toEye / dist), 1.0 - exp(-dist * uFog * 0.045));
  vDist = dist;
  gl_Position = rasterise(world);
}

/**
 * The scene's shading, minus the specular lobe.
 *
 * Occlusion multiplies the direct term as well as the ambient one,
 * which is not what a single blade in free air would do. It is what a
 * blade in a *sward* does: the bottom third of it is buried in the
 * neighbours, and lighting each stalk as though it stood alone is what
 * makes cheap grass read as a field of green needles.
 */
vec3 shadeBlade(vec3 n, vec3 albedo, float sun, float occ, float transmit) {
  vec3 l = uLightDir;
  float ndl = max(dot(n, l), 0.0);
  float back = max(-dot(n, l), 0.0);
  back = back * back * transmit;
  return albedo * (uTint * 2.4 * (ndl + back) * sun * occ
                 + vec3(0.16, 0.19, 0.24) * occ);
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
/* The grid, as numbers rather than as defines: how far the cover
   reaches is a control now, and both of these are derived from it. */
uniform int   uGrid;
uniform float uCell;

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
  int cell = gl_InstanceID % cells;
  int layer = gl_InstanceID / cells;

  vec2 ci = vec2(float(cell % uGrid), float(cell / uGrid)) - float(uGrid) * 0.5;
  vec2 corner = uPatch + ci * uCell;

  /* Hashed off the world coordinate of the cell, never off the instance
     index. The grid slides under the camera; if the seed slid with it
     the entire meadow would crawl along the ground as you walked.

     Quantised before hashing, and not for tidiness: the corner is a
     multiple of the cell size, but it is a multiple computed in floats,
     and two paths to the same cell can land an ulp apart. One ulp is
     enough for hash33 to return a different blade. */
  vec2 key = floor(corner / uCell + 0.5);
  vec3 h1 = hash33(vec3(key * 31.7, float(layer) * 13.3 + 0.7));
  vec3 h2 = hash33(vec3(key * 17.1 + 91.3, float(layer) * 5.9 + 2.1));

  vec2 base = corner + (h1.xy * 0.94 + 0.03) * uCell;

  /* The rim fade is a height fade. Blades do not become transparent and
     they do not wink out; they shorten into the ground, and the last
     thing to go is the thing that was already the least visible. */
  float rim = 1.0 - smoothstep(uRadius * 0.72, uRadius, length(base - uViewer));
  if (rim <= 0.002) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }

  float height = BLADE_H * mix(0.55, 1.20, h1.z) * rim;

  // Three quads and a point: rows 0..2 have a left and a right vertex,
  // the seventh vertex is the tip and sits on the centreline.
  int vid = gl_VertexID;
  bool tip = vid >= 6;
  float row = tip ? 3.0 : float(vid / 2);
  float side = tip ? 0.0 : (float(vid & 1) * 2.0 - 1.0);
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

  vec3 root = vec3(base.x, FLOOR_Y, base.y);
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
  vec3 face = normalize(cross(across, tangent));
  float sgn = dot(face, uCamPos - root) < 0.0 ? -1.0 : 1.0;
  vec3 n = normalize(face * sgn + across * (side * BLADE_CURL)
                   + vec3(0.0, SWARD_BIAS, 0.0));

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

  vec3 root = vec3(pos.x, FLOOR_Y, pos.y);
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
               + vec3(0.0, SWARD_BIAS * 0.5, 0.0));
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

/* ── one fragment shader for both ─────────────────────────────────── */

const FRAG_COVER = /* glsl */`
${PRECISION}

in vec3 vColor;
in vec3 vRound;
in float vDist;
out vec4 outColor;

/* Deliberately almost empty. The meadow is drawn several blades deep
   over the bottom half of the screen, so anything in here is paid for
   many times per pixel to shade something a few pixels tall — the
   lighting and the fog are both settled per vertex instead. */
void main() {
  // Square geometry, round flower. One varying and one compare beats
  // spending a triangle fan on something four pixels across.
  if (vRound.z > 0.5 && dot(vRound.xy, vRound.xy) > 1.0) discard;

  // Alpha is the scene's depth channel, in world units from the eye.
  outColor = vec4(vColor, vDist);
}
`;

/* ── the object ───────────────────────────────────────────────────── */

/** Floor styles, in the order the picker offers them. */
export const GROUND_STYLES = ['grid', 'grass', 'meadow'];

/** True when this style wants soil under it instead of the grid. */
export function isCovered(style) {
  return style === 'grass' || style === 'meadow';
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
function seed(ix, iz, k) {
  let h = Math.imul(ix | 0, 0x27d4eb2d) ^ Math.imul(iz | 0, 0x165667b1)
        ^ Math.imul(k | 0, 0x9e3779b1);
  h ^= h >>> 15; h = Math.imul(h, 0x2c1b3c6d);
  h ^= h >>> 12; h = Math.imul(h, 0x297a2d39);
  h ^= h >>> 15;
  return (h >>> 0) / 4294967296;
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
    this.grass = new Program(gl, VERT_GRASS, FRAG_COVER, { name: 'ground/grass' });
    this.flower = new Program(gl, VERT_FLOWER, FRAG_COVER, { name: 'ground/flower' });

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
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    /** The lattice the patch origin is snapped to. Published because it
        is the whole reason the grass stays put, and a claim that cannot
        be checked from outside is a claim nobody will notice breaking. */
    this.cell = CELL_REF;
    this._patch = new Float32Array(2);
    this._viewer = new Float32Array(2);
    this._jitter = new Float32Array(2);
    /** Where the sown buffer was centred, and what it was sown for. */
    this._sowKey = null;
    /** How many times the flowers have been re-sown. Watched by the tests:
        a number that climbs while the camera stands still is the bug this
        replaced. */
    this.sowings = 0;
    this.blades = 0;
    this.flowers = 0;
    this.triangles = 0;
  }

  /**
   * Fill the instance buffer with every flower within reach of a snapped
   * origin, and upload it. Called only when that origin, the clump size
   * or the density changes.
   */
  _sow(originX, originZ, clumpCell, perClump, reach) {
    const buf = this._sown;
    /* One cell of margin past the reach: the fade is measured from where
       the eye is now, which can be half a cell beyond the origin this was
       sown around, and a flower missing from the buffer cannot fade in. */
    const span = Math.ceil((reach + clumpCell) / clumpCell);
    const ix0 = Math.round(originX / clumpCell);
    const iz0 = Math.round(originZ / clumpCell);
    const limit = (reach + clumpCell) * (reach + clumpCell);

    let n = 0;
    for (let dz = -span; dz <= span && n < MAX_FLOWERS; dz++) {
      for (let dx = -span; dx <= span && n < MAX_FLOWERS; dx++) {
        const ix = ix0 + dx, iz = iz0 + dz;
        if (seed(ix, iz, 1) > CLUMP_CHANCE) continue;   // no clump here

        const cx = (ix + seed(ix, iz, 2) * 0.72 + 0.14) * clumpCell;
        const cz = (iz + seed(ix, iz, 3) * 0.72 + 0.14) * clumpCell;
        const clumpR = 0.40 + seed(ix, iz, 4) * 1.10;
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
   * Draw into the currently bound target, which must be the one the cat
   * draws into: same depth attachment, same camera, same alpha
   * convention.
   *
   * @param {object} camera  pos/right/up/fwd/focal/aspect/width/height
   * @param {object} env     the scene's light and its cluster field
   * @param {object} opts    style, density, wind, radius, frame
   */
  draw(camera, env, opts) {
    if (!isCovered(opts.style)) { this.blades = 0; this.flowers = 0; this.triangles = 0; return; }
    const gl = this.gl;

    const radius = Math.max(RADIUS_MIN, Math.min(RADIUS_MAX, opts.radius));
    const { cell, grid } = grassGrid(radius);
    this.cell = cell;
    this.radius = radius;

    /* Centred a little ahead of the camera rather than under it. Half a
       patch centred on the lens is spent behind the viewer; pushing it
       down the view axis spends the same budget on ground that is
       actually in frame. */
    const ahead = radius * 0.35;
    const cx = camera.pos[0] + camera.fwd[0] * ahead;
    const cz = camera.pos[2] + camera.fwd[2] * ahead;
    this._viewer[0] = cx;
    this._viewer[1] = cz;

    // Snapped, so a cell keeps its world coordinate and therefore its
    // hash while the grid slides underneath.
    this._patch[0] = Math.round(cx / cell) * cell;
    this._patch[1] = Math.round(cz / cell) * cell;

    // The cat's jitter, for the same reason: this is the same target and
    // the same temporal filter resolves both.
    this._jitter[0] = Math.sin(opts.frame * 2.39996) / camera.width;
    this._jitter[1] = Math.sin(opts.frame * 4.10000 + 1.7) / camera.height;

    const density = Math.max(0, Math.min(1, opts.density));
    const layers = Math.max(1, Math.round(density * MAX_LAYERS));

    const common = {
      uCamPos: camera.pos,
      uRight: camera.right,
      uUp: camera.up,
      uFwd: camera.fwd,
      uFocal: camera.focal,
      uAspect: camera.aspect,
      uJitter: this._jitter,

      uPatch: this._patch,
      uViewer: this._viewer,
      uRadius: radius,
      uGrid: grid,
      uCell: cell,
      uWind: opts.wind,
      uFog: env.fog,

      uLightDir: env.dir,
      uTint: env.tint,
      uShadowSoft: env.shadowSoft,
      uShadowSteps: env.shadowSteps,
      uShadowNoise: env.shadowNoise,

      // The cluster, exactly as the marcher sees it this frame, so the
      // shadow it throws across the grass is the shadow it throws.
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
    };

    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
    gl.depthMask(true);
    // Both faces of a blade are the blade. The normal is turned toward
    // the eye in the vertex shader instead.
    gl.disable(gl.CULL_FACE);

    gl.bindVertexArray(this.grassVao);
    this.blades = grid * grid * layers;
    this.grass.use(common);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, BLADE_VERTS, this.blades);
    this.triangles = this.blades * (BLADE_VERTS - 2);

    if (opts.style === 'meadow') {
      /* The clump grid grows with the reach the same way the grass grid
         does, so a wider view spreads the clumps out rather than sowing
         quadratically more of them. */
      const clumpCell = CLUMP_CELL_REF * Math.sqrt(radius / RADIUS_REF);
      const reach = radius * FLOWER_REACH;
      const perClump = Math.max(4, Math.round(density * MAX_PER_CLUMP));

      /* Sown when — and only when — the answer would differ. The origin
         moves a whole clump cell at a time, so walking a straight line
         re-sows every few metres and standing still never does. */
      const ox = Math.round(this._patch[0] / clumpCell) * clumpCell;
      const oz = Math.round(this._patch[1] / clumpCell) * clumpCell;
      const key = `${ox}|${oz}|${clumpCell.toFixed(4)}|${perClump}|${reach.toFixed(3)}`;
      if (key !== this._sowKey) {
        this.flowers = this._sow(ox, oz, clumpCell, perClump, reach);
        this._sowKey = key;
      }

      if (this.flowers) {
        gl.bindVertexArray(this.flowerVao);
        // Its own, shorter reach — see FLOWER_REACH.
        this.flower.use({ ...common, uRadius: reach });
        gl.drawArraysInstanced(gl.TRIANGLES, 0, FLOWER_VERTS, this.flowers);
        this.triangles += this.flowers * (FLOWER_VERTS / 3);
      }
    } else {
      this.flowers = 0;
      // Forget where they were sown, or coming back to the meadow after
      // walking away shows the ones that were around the old spot.
      this._sowKey = null;
    }

    gl.bindVertexArray(null);
    gl.disable(gl.DEPTH_TEST);
  }

  dispose() {
    this.grass.dispose();
    this.flower.dispose();
    this.gl.deleteVertexArray(this.grassVao);
    this.gl.deleteVertexArray(this.flowerVao);
    this.gl.deleteBuffer(this.flowerVbo);
  }
}
