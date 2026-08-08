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

   There is still no vertex buffer anywhere. A blade is seven vertices
   derived from gl_VertexID and a flower is forty-eight, and where each
   one stands comes from hashing its cell. Nothing is uploaded per frame
   but a camera and a clock.

   Two ideas carry the whole file:

   The patch follows the viewer. A fixed lawn is either too small to
   reach the horizon or too large to draw, so the grid is centred on the
   camera and snapped to its own cell size, and every blade is hashed
   from the *world* coordinate of its cell. The grid slides; the grass
   does not. Blades fade out at the rim by shrinking into the ground,
   which is the one edge treatment that has nothing to pop.

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
   Cells across, and how wide one is. The product is the patch, and the
   patch has to reach far enough that its rim is lost in fog rather than
   read as the edge of a rug. */
const GRID = 64;
const CELL = 0.5;

/* Where the cover fades to nothing. A cell short of the grid's inscribed
   circle, because the grid is snapped and can sit up to half a cell off
   the centre the fade is measured from — without the margin a corner of
   the meadow would occasionally come up short. */
const RADIUS = GRID * CELL * 0.5 - CELL;

/** Blades per cell at full density. One instance each. */
const MAX_LAYERS = 6;
/** Vertices in a blade: three quads narrowing to a single tip vertex. */
const BLADE_VERTS = 7;

/* ── flowers ──────────────────────────────────────────────────────
   Flowers are placed by clump, not by blade. A clump cell either grows
   one or it does not, and the flowers inside it thin out toward its rim,
   so what comes out is a dense middle with singles scattered round it
   rather than a disc with an edge. */
const CLUMP_CELL = 4.0;

/* Flowers stop well short of where the grass does. A blade at the rim
   still counts, because what it contributes there is the *colour* of
   the far field; a flower head at that range is under a pixel and
   contributes a speck. Sizing the clump grid to a shorter reach is the
   cheapest saving in the file — it is a square law — and there is
   nothing to see for it. */
const FLOWER_RADIUS = RADIUS * 0.62;
const CLUMP_GRID = Math.ceil((FLOWER_RADIUS * 2) / CLUMP_CELL) + 1;
/** How many clump cells in the patch actually grow one. */
const CLUMP_CHANCE = 0.62;
/** Flower slots per clump at full density; the thinning eats about 30%. */
const MAX_PER_CLUMP = 30;

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
#define GRID ${GRID}
#define CELL ${CELL.toFixed(4)}
#define CLUMP_GRID ${CLUMP_GRID}
#define CLUMP_CELL ${CLUMP_CELL.toFixed(4)}
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

  int cells = GRID * GRID;
  int cell = gl_InstanceID % cells;
  int layer = gl_InstanceID / cells;

  vec2 ci = vec2(float(cell % GRID), float(cell / GRID)) - float(GRID) * 0.5;
  vec2 corner = uPatch + ci * CELL;

  /* Hashed off the world coordinate of the cell, never off the instance
     index. The grid slides under the camera; if the seed slid with it
     the entire meadow would crawl along the ground as you walked. */
  vec3 h1 = hash33(vec3(corner * 31.7, float(layer) * 13.3 + 0.7));
  vec3 h2 = hash33(vec3(corner * 17.1 + 91.3, float(layer) * 5.9 + 2.1));

  vec2 base = corner + (h1.xy * 0.94 + 0.03) * CELL;

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
uniform float uClumpChance;

const vec3 STEM_COL = vec3(0.045, 0.115, 0.030);
const vec3 EYE_COL  = vec3(0.640, 0.360, 0.030);

/** A clump is one species, so the colour is hashed per clump. */
vec3 petalColour(float r) {
  int k = int(r * 5.0);
  if (k <= 0) return vec3(0.780, 0.760, 0.700);   // white
  if (k == 1) return vec3(0.880, 0.480, 0.020);   // yellow
  if (k == 2) return vec3(0.820, 0.175, 0.290);   // pink
  if (k == 3) return vec3(0.330, 0.170, 0.700);   // lilac
  return vec3(0.880, 0.105, 0.025);               // orange-red
}

void main() {
  vColor = vec3(0.0);
  vRound = vec3(0.0);
  vDist = 0.0;

  int cells = CLUMP_GRID * CLUMP_GRID;
  int cell = gl_InstanceID % cells;
  int slot = gl_InstanceID / cells;

  vec2 ci = vec2(float(cell % CLUMP_GRID), float(cell / CLUMP_GRID))
          - float(CLUMP_GRID) * 0.5;
  vec2 ccorner = uPatch + ci * CLUMP_CELL;

  // Does this cell grow a clump at all, and where inside it.
  vec3 c1 = hash33(vec3(ccorner * 23.9, 5.0));
  if (c1.x > uClumpChance) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }

  vec2 centre = ccorner + (c1.yz * 0.72 + 0.14) * CLUMP_CELL;
  vec3 c2 = hash33(vec3(centre * 11.3, 13.0));
  float clumpR = mix(0.40, 1.50, c2.x);

  vec3 fh = hash33(vec3(centre * 29.1, float(slot) + 0.5));

  /* Radius as the hash itself rather than its square root. Uniform area
     density would give a disc; this piles them into the middle, which is
     what a clump of anything self-seeding looks like. */
  float u = fh.x;
  float ang = fh.y * PI * 2.0;
  vec2 pos = centre + vec2(cos(ang), sin(ang)) * (clumpR * u);

  /* And thinned toward the rim, so the clump has no edge — a few
     stragglers standing on their own in the grass around it, which is
     the part that stops it reading as a planted bed. */
  if (fh.z < u * u * 0.92) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }

  float rim = 1.0 - smoothstep(uRadius * 0.72, uRadius, length(pos - uViewer));
  if (rim <= 0.002) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }

  vec3 fh2 = hash33(vec3(pos * 19.7 + 3.3, 7.0));
  float head = mix(0.055, 0.095, fh2.x) * rim;   // petal reach
  /* Taller than the grass around it, deliberately and by a clear
     margin. A flower whose head sits at blade height is a flower nobody
     will ever see: the sward closes over it from every angle but
     straight down. */
  float stemH = mix(BLADE_H * 1.15, BLADE_H * 1.75, fh2.y) * rim;

  float yaw = fh2.z * PI * 2.0;
  vec2 own = vec2(cos(yaw), sin(yaw));
  float lean = mix(0.04, 0.22, fh.y);

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

      float th = (float(pi) + 0.5) / float(PETALS) * PI * 2.0 + fh.y * PI * 2.0;
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
      albedo = petalColour(c2.y);
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

export class GroundCover {
  constructor(gl) {
    this.gl = gl;
    this.grass = new Program(gl, VERT_GRASS, FRAG_COVER, { name: 'ground/grass' });
    this.flower = new Program(gl, VERT_FLOWER, FRAG_COVER, { name: 'ground/flower' });

    /* Nothing is fetched per vertex, but something has to be bound: a
       leftover attribute array from another draw would be read for
       seven vertices that do not want it. An empty one says so. */
    this.vao = gl.createVertexArray();

    /** The lattice the patch origin is snapped to. Published because it
        is the whole reason the grass stays put, and a claim that cannot
        be checked from outside is a claim nobody will notice breaking. */
    this.cell = CELL;
    this._patch = new Float32Array(2);
    this._viewer = new Float32Array(2);
    this._jitter = new Float32Array(2);
    this.blades = 0;
    this.flowers = 0;
    this.triangles = 0;
  }

  /**
   * Draw into the currently bound target, which must be the one the cat
   * draws into: same depth attachment, same camera, same alpha
   * convention.
   *
   * @param {object} camera  pos/right/up/fwd/focal/aspect/width/height
   * @param {object} env     the scene's light and its cluster field
   * @param {object} opts    style, density, wind, frame
   */
  draw(camera, env, opts) {
    if (!isCovered(opts.style)) { this.blades = 0; this.flowers = 0; this.triangles = 0; return; }
    const gl = this.gl;

    /* Centred a little ahead of the camera rather than under it. Half a
       patch centred on the lens is spent behind the viewer; pushing it
       down the view axis spends the same budget on ground that is
       actually in frame. */
    const ahead = RADIUS * 0.35;
    const cx = camera.pos[0] + camera.fwd[0] * ahead;
    const cz = camera.pos[2] + camera.fwd[2] * ahead;
    this._viewer[0] = cx;
    this._viewer[1] = cz;

    // Snapped, so a cell keeps its world coordinate and therefore its
    // hash while the grid slides underneath.
    this._patch[0] = Math.round(cx / CELL) * CELL;
    this._patch[1] = Math.round(cz / CELL) * CELL;

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
      uRadius: RADIUS,
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
    gl.bindVertexArray(this.vao);

    this.blades = GRID * GRID * layers;
    this.grass.use(common);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, BLADE_VERTS, this.blades);
    this.triangles = this.blades * (BLADE_VERTS - 2);

    if (opts.style === 'meadow') {
      const perClump = Math.max(4, Math.round(density * MAX_PER_CLUMP));
      this.flowers = CLUMP_GRID * CLUMP_GRID * perClump;
      // Its own, shorter reach — see FLOWER_RADIUS.
      this.flower.use({ ...common, uRadius: FLOWER_RADIUS, uClumpChance: CLUMP_CHANCE });
      gl.drawArraysInstanced(gl.TRIANGLES, 0, FLOWER_VERTS, this.flowers);
      this.triangles += this.flowers * (FLOWER_VERTS / 3);
    } else {
      this.flowers = 0;
    }

    gl.bindVertexArray(null);
    gl.disable(gl.DEPTH_TEST);
  }

  dispose() {
    this.grass.dispose();
    this.flower.dispose();
    this.gl.deleteVertexArray(this.vao);
  }
}
