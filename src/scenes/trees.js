/* ── scenes/trees.js ─────────────────────────────────────────────────
   Trees, grown on the CPU and expanded on the GPU.

   The division of labour is the whole design. Recursive branching is a
   nested loop with a stack — trivial in JS, an arithmetic puzzle in a
   vertex shader, and we have already learned that lesson once on the
   flowers. But a branch expanded into a tapered tube, and a leaf
   expanded into a shaped card, are pure functions of a vertex index, and
   those belong on the GPU. So the CPU writes *lines and attachment
   points*; nothing here ever builds a triangle.

   Chunks, because generating is expensive and storing is not.

   A tree is about a hundred branch segments and three hundred leaves,
   and growing a hundred of them takes a couple of milliseconds — enough
   to be seen as a hitch if it happened whenever the view shifted. So the
   world is cut into fixed sixteen-metre chunks, each generated once and
   kept, and moving costs only the chunks newly in range. The chunk size
   is *fixed in world space* and deliberately not scaled by the reach:
   scale it and changing the reach invalidates everything, where a fixed
   size means a reach change only loads or drops the outer ring and every
   chunk you already had stays.

   One chunk of margin is always generated past the reach, so the edge of
   the world is never the edge of what has been grown.

   Placement is a jittered sub-grid rather than dart-throwing. Each chunk
   holds at most SUB×SUB trees, one per sub-cell, jittered inside it —
   which caps the count per chunk (so a chunk's slot in the buffer has a
   fixed bound) and guarantees a minimum separation *across chunk
   boundaries* without any chunk ever having to look at its neighbours.
   ------------------------------------------------------------------ */

import { Program } from '../core/program.js';
import { Target } from '../core/gl.js';
import { PRECISION, CONSTANTS, HASH, ROTATE, SIMPLEX3 } from '../shaders/common.js';
import {
  CLUSTER_UNIFORMS, CLUSTER_FIELD, CLUSTER_LAYERS, CLUSTER_SHADOW, SKY, FLOOR_Y,
} from './cluster.js';
import { CAT_PROXY_GLSL } from './cat/index.js';
import { PLANT_COMMON, FRAG_PLANT } from './plant.js';
import { TERRAIN_GLSL, waterDepthAt } from './terrain.js';
import { WIND_GLSL } from './wind.js';
import { CANOPY_SHADE_GLSL, CANOPY_EXTENT, CANOPY_SIZE } from './canopy.js';

/* ── the world grid ───────────────────────────────────────────────── */

/** Chunk edge, in world units. Fixed, on purpose — see the header. */
const CHUNK = 16.0;
/** Sub-cells across a chunk. SUB*SUB is the hard cap on trees per chunk. */
const SUB = 2;
const SUB_CELL = CHUNK / SUB;
/** How far from its sub-cell's centre a tree may wander, as a fraction.
    Whatever is left over is the guaranteed separation: at 0.28 of an
    eight-metre cell, no two trees are ever closer than 3.5 metres. */
const TREE_JITTER = 0.28;
export const MIN_SEPARATION = SUB_CELL * (1 - 2 * TREE_JITTER);
/** Fraction of sub-cells that actually grow one, at full density. */
const TREE_CHANCE = 0.34;

/* Nothing grows within this of the origin. The cluster sits there, a
   tree would stand inside the spheres and through the ring, and the
   scene's subject would spend half its time behind a trunk. A clearing
   also happens to be the most natural-looking thing a wood can do. */
const CLEAR_R = 6.0;

/* How far trees are grown, as a share of the cover's reach and a
   ceiling of its own — the same bargain the flowers make, so that one
   control moves all three together.

   Trees earn a longer reach than flowers because they are the one thing
   here tall enough to break the horizon: a treeline is what says how far
   away the far side is. The ceiling is where branch segments, which have
   no level of detail, start to fill their buffer. */
const TREE_REACH = 0.75;
const TREE_REACH_MIN = 16.0;
const TREE_REACH_MAX = 90.0;

/* Ceilings on the two instance buffers. Sized for the reach above with
   room to spare; growth stops when they are full rather than reallocating
   mid-frame. */
const MAX_SEGMENTS = 26000;
const MAX_LEAVES = 72000;

/** Floats per record. See the attribute layouts in the shaders. */
const SEG_STRIDE = 20;
const LEAF_STRIDE = 16;

/* Leaf level of detail.

   A tree fifty metres away was carrying the same four hundred leaves as
   one at arm's length, and at that range the whole canopy is thirty
   pixels. What is dropped is the *count*; what compensates is the size
   of the survivors, so the canopy keeps its coverage and only loses a
   detail nobody can resolve. Fewer and larger is invisible past twenty
   metres and it is most of the leaf budget.

   Both halves have to agree, so both read these: the CPU packs a prefix
   of each chunk's leaves, and the vertex shader scales what is left by
   the inverse square root of the same fraction — which is exactly what
   holds the covered area constant. */
const LEAF_LOD_NEAR = 14.0;
const LEAF_LOD_SPAN = 42.0;
const LEAF_LOD_MIN = 0.22;
/** Ceiling on the compensation. 1/sqrt(0.22) is 2.13; this is kinder. */
const LEAF_LOD_GROW = 1.9;

/** How far the viewer moves before the levels are recomputed. */
const LOD_STEP = 4.0;

/** The fraction of a chunk's leaves worth drawing at this range. */
function leafLod(d) {
  return Math.min(1, Math.max(LEAF_LOD_MIN, 1 - (d - LEAF_LOD_NEAR) / LEAF_LOD_SPAN));
}

/** Sides around a branch tube, and the strip that wraps it. */
const SIDES = 5;
const SEG_VERTS = 2 * (SIDES + 1);
/** Four rows of two, plus a tip on the centreline. */
const LEAF_VERTS = 9;

/* ── shape ────────────────────────────────────────────────────────── */

const TAU = Math.PI * 2;
/** Phyllotaxis. Successive branches step by the golden angle. */
const GOLDEN = 2.399963;

const TREE_MIN_H = 3.0;
const TREE_MAX_H = 5.0;
/** Recursion depth. Twigs are grown at the last one and carry the leaves. */
const MAX_DEPTH = 3;

/* Bark and leaf palettes, authored linear like everything downstream. */
const BARK = [
  [0.052, 0.040, 0.031],
  [0.068, 0.055, 0.042],
  [0.040, 0.036, 0.034],
];
const FOLIAGE = [
  [0.055, 0.135, 0.038],
  [0.085, 0.160, 0.042],
  [0.045, 0.105, 0.048],
  [0.110, 0.145, 0.036],
];

/* ── deterministic randomness ─────────────────────────────────────── */

/** One value in [0,1) from a cell and a salt. The flowers' hash. */
function seed(ix, iz, k) {
  let h = Math.imul(ix | 0, 0x27d4eb2d) ^ Math.imul(iz | 0, 0x165667b1)
        ^ Math.imul(k | 0, 0x9e3779b1);
  h ^= h >>> 15; h = Math.imul(h, 0x2c1b3c6d);
  h ^= h >>> 12; h = Math.imul(h, 0x297a2d39);
  h ^= h >>> 15;
  return (h >>> 0) / 4294967296;
}

/**
 * A stream of values for one tree.
 *
 * Seeded from the tree's own sub-cell, so the same tree comes out of the
 * same ground however many times its chunk is generated — which is what
 * lets a chunk be dropped and regrown without the wood rearranging
 * itself behind you.
 */
function rngFor(ix, iz, k) {
  let s = (Math.imul(ix, 0x9e3779b1) ^ Math.imul(iz, 0x85ebca6b)
        ^ Math.imul(k + 1, 0xc2b2ae35)) | 0;
  return () => {
    s = Math.imul(s ^ (s >>> 15), 0x2c1b3c6d);
    s ^= s >>> 12;
    s = Math.imul(s, 0x297a2d39);
    s ^= s >>> 15;
    return (s >>> 0) / 4294967296;
  };
}

/** IEEE half back to a JS number. Only the readback path needs it. */
function halfToFloat(h) {
  const s = (h & 0x8000) ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return s * Math.pow(2, -14) * (m / 1024);
  if (e === 31) return m ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + m / 1024);
}

/* ── small vector helpers ─────────────────────────────────────────── */

const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
function unit(v) {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}
/** Re-perpendicularise f against d, which is parallel transport enough
    for a chain of short segments and never degenerates the way rebuilding
    a frame from a world axis does on a vertical trunk. */
function transport(f, d) {
  const k = dot3(f, d);
  return unit([f[0] - d[0] * k, f[1] - d[1] * k, f[2] - d[2] * k]);
}

/** Fisher-Yates over whole leaf records. */
function shuffleLeaves(buf, n, rnd) {
  const tmp = new Float32Array(LEAF_STRIDE);
  for (let i = n - 1; i > 0; i--) {
    const j = (rnd() * (i + 1)) | 0;
    if (i === j) continue;
    tmp.set(buf.subarray(i * LEAF_STRIDE, (i + 1) * LEAF_STRIDE));
    buf.copyWithin(i * LEAF_STRIDE, j * LEAF_STRIDE, (j + 1) * LEAF_STRIDE);
    buf.set(tmp, j * LEAF_STRIDE);
  }
}

/* ── growing one tree ─────────────────────────────────────────────── */

/**
 * Emit a tree's branch segments and leaves into the two arrays.
 *
 * Everything that makes one tree unlike another is drawn once here and
 * then held constant down the recursion: height, taper, how far the
 * branches spread, how hard they reach for the light, how lobed the bark
 * is, what colour the leaves are.
 */
function growTree(out, rnd, x, z) {
  const height = TREE_MIN_H + rnd() * (TREE_MAX_H - TREE_MIN_H);

  const P = {
    base: [x, z],
    /* Trunk radius as a fraction of height. Real trees are stockier than
       anyone draws them from memory. */
    lenRatio: 0.58 + rnd() * 0.16,
    spread: 0.38 + rnd() * 0.40,          // radians a child leaves its parent by
    /* How hard a branch turns back toward the sky as it grows. This one
       number is most of the difference between a poplar and an oak. */
    up: 0.06 + rnd() * 0.22,
    wobble: 0.05 + rnd() * 0.09,
    lateral: 0.30 + rnd() * 0.35,
    lobeAmp: 0.05 + rnd() * 0.11,
    lobePhase: rnd() * TAU,
    bark: BARK[(rnd() * BARK.length) | 0],
    barkVary: 0.8 + rnd() * 0.5,
    foliage: FOLIAGE[(rnd() * FOLIAGE.length) | 0],
    /* A "leaf" here is really a sprig — a hand-sized clump of foliage,
       not one botanical leaf. At true scale a leaf on a four-metre tree
       is ten centimetres, and three hundred of those cover about two
       percent of a canopy: the first version grew a wood of bare winter
       sticks. The instance budget buys coverage or it buys botany, and
       at any distance you will ever see this from, coverage is what
       reads as a tree. */
    leafSize: 0.20 + rnd() * 0.15,
    perTwig: 13 + ((rnd() * 8) | 0),
    canopy: { y: 0, r: 0, n: 0 },          // accumulated for the shadow proxy
  };

  const lean = rnd() * TAU;
  const tilt = 0.03 + rnd() * 0.10;
  const dir = unit([Math.cos(lean) * tilt, 1, Math.sin(lean) * tilt]);
  const frame = unit(cross3(dir, [Math.cos(lean + 1.7), 0.2, Math.sin(lean + 1.7)]));

  grow(out, rnd, P,
    [x, FLOOR_Y, z], dir, frame,
    height * 0.46, height * (0.030 + rnd() * 0.016),
    0, rnd() * TAU);

  return P.canopy;
}

/** One branch: a chain of segments, its children, and its leaves. */
function grow(out, rnd, P, start, dir0, frame0, len, rad, depth, phase) {
  if (out.segN >= MAX_SEGMENTS) return;

  const segs = depth === 0 ? 5 : depth >= MAX_DEPTH ? 2 : 3;
  const segLen = len / segs;
  const endRad = rad * (depth === 0 ? 0.40 : 0.56);

  let p = start, d = dir0, f = frame0;
  const kids = [];

  for (let i = 0; i < segs; i++) {
    /* Gravitropism plus a wobble. The wobble has to be applied to the
       *direction*, not to the vertices: nudging the points gives a
       knobbly straight branch, nudging the heading gives one that
       wanders, which is what a branch actually does. */
    d = unit([
      d[0] + (rnd() - 0.5) * P.wobble,
      d[1] + P.up * (1 - d[1] * d[1]) + (rnd() - 0.5) * P.wobble * 0.5,
      d[2] + (rnd() - 0.5) * P.wobble,
    ]);
    f = transport(f, d);

    const q = [p[0] + d[0] * segLen, p[1] + d[1] * segLen, p[2] + d[2] * segLen];
    const t0 = i / segs, t1 = (i + 1) / segs;
    let ra = rad + (endRad - rad) * t0;
    let rb = rad + (endRad - rad) * t1;

    /* Root flare. The bottom tenth of a trunk swells into the ground,
       and it is the single detail that separates a tree that grew from a
       pole that was pushed in. */
    if (depth === 0) {
      ra *= 1 + 0.95 * Math.exp(-t0 * len / 0.22);
      rb *= 1 + 0.95 * Math.exp(-t1 * len / 0.22);
    }

    pushSegment(out, P, p, q, ra, rb, f, phase,
      flexAt(depth, t0), flexAt(depth, t1));

    p = q;

    // A lateral, taken off an interior joint rather than the tip, which
    // is what keeps a tree from looking like a firework.
    if (depth < MAX_DEPTH && i >= 1 && i < segs - 1 && rnd() < P.lateral) {
      kids.push([p, d, f, len * P.lenRatio * 0.72, ra * 0.58, kids.length]);
    }
  }

  if (depth < MAX_DEPTH) {
    /* Two at the tip. Da Vinci's rule — the children's cross-sections sum
       to the parent's — gives the radius, and it is worth more to the eye
       than any amount of noise: pick the ratio freely and the tree reads
       as plastic. */
    const r = endRad / Math.SQRT2;
    kids.push([p, d, f, len * P.lenRatio, r, kids.length]);
    kids.push([p, d, f, len * P.lenRatio * 0.86, r, kids.length]);

    for (const [cp, cd, cf, clen, crad, idx] of kids) {
      const az = GOLDEN * idx + rnd() * 0.7 + phase;
      const sp = P.spread * (0.72 + rnd() * 0.56);
      const side = cross3(cd, cf);
      const outward = unit([
        cf[0] * Math.cos(az) + side[0] * Math.sin(az),
        cf[1] * Math.cos(az) + side[1] * Math.sin(az),
        cf[2] * Math.cos(az) + side[2] * Math.sin(az),
      ]);
      const nd = unit([
        cd[0] * Math.cos(sp) + outward[0] * Math.sin(sp),
        cd[1] * Math.cos(sp) + outward[1] * Math.sin(sp),
        cd[2] * Math.cos(sp) + outward[2] * Math.sin(sp),
      ]);
      grow(out, rnd, P, cp, nd, transport(outward, nd),
        clen * (0.82 + rnd() * 0.36), crad, depth + 1,
        // Children lag their parent, so a branch moves as one thing and
        // its twigs trail it instead of beating in unison.
        phase + 0.8 + rnd() * 1.2);
    }

    /* A thinner scatter on the second-to-last branches as well.
       Foliage grown only on the final twigs comes out as a row of
       distinct cabbages on bare sticks — real canopies carry leaves for
       most of the length that is thin enough to hold them, and the few
       here are what fill the gaps between the clumps. */
    if (depth === MAX_DEPTH - 1) {
      leaves(out, rnd, P, start, p, d, f, phase, 0.42);
    }
  } else {
    leaves(out, rnd, P, start, p, d, f, phase, 1.0);
  }
}

/** How much the wind moves a point: nothing at the base, everything at a twig. */
function flexAt(depth, t) {
  return (depth + t) / (MAX_DEPTH + 1);
}

function pushSegment(out, P, a, b, ra, rb, f, phase, flex0, flex1) {
  if (out.segN >= MAX_SEGMENTS) return;
  const o = out.segN * SEG_STRIDE;
  const s = out.segs;
  s[o] = a[0]; s[o + 1] = a[1]; s[o + 2] = a[2]; s[o + 3] = ra;
  s[o + 4] = b[0]; s[o + 5] = b[1]; s[o + 6] = b[2]; s[o + 7] = rb;
  s[o + 8] = f[0]; s[o + 9] = f[1]; s[o + 10] = f[2]; s[o + 11] = phase;
  s[o + 12] = flex0; s[o + 13] = flex1; s[o + 14] = P.base[0]; s[o + 15] = P.base[1];
  s[o + 16] = P.lobeAmp; s[o + 17] = P.lobePhase;
  s[o + 18] = P.barkVary; s[o + 19] = 0;
  out.segN++;
}

/** Leaves scattered along a twig, each pointing out and up from it. */
function leaves(out, rnd, P, from, to, d, f, phase, share) {
  const side = cross3(d, f);
  const n = Math.max(1, Math.round(P.perTwig * share));
  /* Scattered off the twig, not threaded onto it. Leaves placed exactly
     on the centreline read as beads on a string; a little spread
     perpendicular to it is what turns a row of them into a mass. */
  const spread = P.leafSize * 0.85;
  for (let k = 0; k < n; k++) {
    if (out.leafN >= MAX_LEAVES) return;
    const t = 0.06 + 0.94 * rnd();
    const px = from[0] + (to[0] - from[0]) * t + (rnd() - 0.5) * spread;
    const py = from[1] + (to[1] - from[1]) * t + (rnd() - 0.5) * spread;
    const pz = from[2] + (to[2] - from[2]) * t + (rnd() - 0.5) * spread;

    const az = GOLDEN * k + rnd() * 0.9;
    const ca = Math.cos(az), sa = Math.sin(az);
    // Out from the twig, forward along it, and reaching for the light.
    const ld = unit([
      f[0] * ca + side[0] * sa + d[0] * 0.55,
      f[1] * ca + side[1] * sa + d[1] * 0.55 + 0.30,
      f[2] * ca + side[2] * sa + d[2] * 0.55,
    ]);

    const o = out.leafN * LEAF_STRIDE;
    const s = out.leaves;
    s[o] = px; s[o + 1] = py; s[o + 2] = pz;
    s[o + 3] = P.leafSize * (0.7 + rnd() * 0.7);
    s[o + 4] = ld[0]; s[o + 5] = ld[1]; s[o + 6] = ld[2];
    s[o + 7] = rnd() * TAU;                          // roll about its own axis
    s[o + 8] = flexAt(MAX_DEPTH, t); s[o + 9] = phase;
    s[o + 10] = P.base[0]; s[o + 11] = P.base[1];
    s[o + 12] = P.foliage[0] * (0.8 + rnd() * 0.45);
    s[o + 13] = P.foliage[1] * (0.8 + rnd() * 0.45);
    s[o + 14] = P.foliage[2] * (0.8 + rnd() * 0.45);
    s[o + 15] = 0;
    out.leafN++;

    // The canopy proxy is measured from the leaves themselves rather than
    // guessed from the height, so it fits whatever actually grew.
    const c = P.canopy;
    c.n++;
    c.y += py;
    c.r = Math.max(c.r, Math.hypot(px - P.base[0], pz - P.base[1]));
  }
}

/* ── shaders ──────────────────────────────────────────────────────── */

/* The sway, and the two geometry expansions, kept apart from any one
   program on purpose: each is compiled twice, once for the camera and
   once for the sun. A shadow cast by a tree standing still while the
   tree itself leans is worse than no shadow at all, so the light pass is
   not allowed its own copy of any of this. */

const TREE_SWAY = /* glsl */`
#define FLOOR_Y ${FLOOR_Y.toFixed(4)}

/* How far the top of a trunk travels in a full gust, and how far a twig
   swings on its own. The trunk figure is small on purpose: a tree is a
   tonne of timber and the give is nearly all in the branches. */
const float TRUNK_BEND = 0.016;
const float BRANCH_SWAY = 0.085;

/**
 * Where the wind has put a point that rests at "rest".
 *
 * A pure function of the rest position and of two numbers carried by
 * whichever branch owns the point — which is the entire reason the
 * leaves stay attached to the twigs without a skeleton, a matrix or a
 * skinning weight anywhere. A leaf sitting on a twig's tip passes the
 * twig's flex and the twig's phase, so it is displaced by exactly what
 * displaced the tip. Nothing has to be kept in sync because nothing was
 * ever split in two.
 *
 * The trunk reads only the slow half of the wind. A tree does not answer
 * a half-second flutter, and filtering by mass is most of what makes the
 * scale of a thing legible.
 */
vec3 sway(vec3 rest, vec2 base, float flex, float phase) {
  float h = max(rest.y - FLOOR_Y, 0.0);
  vec2 perp = vec2(-WIND_DIR.y, WIND_DIR.x);

  /* A tree is grown against a flat floor on the CPU and lifted onto the
     ground here, by the height under its own trunk.
     Deliberately not baked into the vertices. Chunks are generated once
     and cached, so baking would mean regrowing the entire wood every
     time the undulation control moves — and the trunk's xz is already an
     attribute, because the wind needed it, so the lookup is four sines
     and no new data. It also keeps a tree vertical on a slope, which is
     what a tree does: they grow toward the light, not perpendicular to
     the hill. */
  float lift = terrainH(base) - FLOOR_Y;

  // The whole tree leaning, growing as the square of the height.
  vec2 d = WIND_DIR * (slowGust(base) * uWind * TRUNK_BEND * h * h);

  // And the branch, on its own beat. Two frequencies at right angles, so
  // a twig traces a figure of eight rather than sliding on a rail.
  float s = flex * flex * uWind * BRANCH_SWAY;
  d += WIND_DIR * (sin(uTime * 2.10 + phase) * s);
  d += perp * (cos(uTime * 1.63 + phase * 1.3) * s * 0.45);

  return rest + vec3(d.x, lift, d.y);
}
`;

const BRANCH_GEOM = /* glsl */`
#define SIDES ${SIDES}

layout(location = 0) in vec4 aA;     // xyz = start of the segment, w = radius there
layout(location = 1) in vec4 aB;     // xyz = end,                  w = radius there
layout(location = 2) in vec4 aFrame; // xyz = transported frame,    w = branch phase
layout(location = 3) in vec4 aFlex;  // xy = flex at each end,      zw = tree base
layout(location = 4) in vec4 aBark;  // x = lobe amount, y = lobe phase, z = tint

/** One vertex of a branch's tube: where it is, which way it faces, and
    the point on the centreline it hangs off. */
void branchVertex(out vec3 p, out vec3 n, out vec3 anchor) {
  int side = gl_VertexID / 2;
  float row = float(gl_VertexID % 2);

  vec3 axis = normalize(aB.xyz - aA.xyz);
  /* The ring's frame comes down the branch from the CPU, parallel
     transported segment by segment. Rebuilding it here from a world axis
     is the obvious thing and it is wrong twice: it degenerates on a
     vertical trunk, and it rotates from one segment to the next, so the
     lobes of the bark do not line up and every joint shows a seam. */
  vec3 U = normalize(aFrame.xyz - axis * dot(aFrame.xyz, axis));
  vec3 V = cross(axis, U);

  float a = float(side) / float(SIDES) * PI * 2.0;
  /* A trunk is not a cylinder. Two low harmonics around the ring give it
     lobes and hollows, and because they are a function of the angle they
     cost one sine each and nothing at all in geometry. */
  float lobe = 1.0 + aBark.x * (sin(a * 3.0 + aBark.y) * 0.62
                              + sin(a * 7.0 - aBark.y * 2.1) * 0.38);

  float flex = mix(aFlex.x, aFlex.y, row);
  /* Swayed on the centreline, then offset. Swaying the surface point
     instead would stretch a thick trunk, because the displacement varies
     across its width. */
  anchor = sway(mix(aA.xyz, aB.xyz, row), aFlex.zw, flex, aFrame.w);
  n = U * cos(a) + V * sin(a);
  p = anchor + n * (mix(aA.w, aB.w, row) * lobe);
}
`;

const LEAF_GEOM = /* glsl */`
#define LEAF_LOD_NEAR ${LEAF_LOD_NEAR.toFixed(2)}
#define LEAF_LOD_SPAN ${LEAF_LOD_SPAN.toFixed(2)}
#define LEAF_LOD_MIN ${LEAF_LOD_MIN.toFixed(3)}
#define LEAF_LOD_GROW ${LEAF_LOD_GROW.toFixed(2)}

layout(location = 0) in vec4 aLeaf;  // xyz = where it joins the twig, w = size
layout(location = 1) in vec4 aDir;   // xyz = the leaf's own axis, w = roll
layout(location = 2) in vec4 aFlex;  // x = flex, y = phase, zw = tree base
layout(location = 3) in vec4 aTint;  // rgb = colour

/**
 * Half-width along a leaf, as one continuous family of silhouettes.
 *
 * Two exponents rather than a table of shapes. The first moves the
 * widest point — low is a leaf broadest near its stalk, high is one
 * broadest past the middle — and the second says how sharply it comes to
 * a point. Between them they cover ovate, lanceolate and obovate and
 * everything in between, so no two leaves on a tree need be the same
 * shape, and none of it costs a branch.
 */
float leafWidth(float t, float skew, float sharp) {
  return pow(sin(PI * pow(clamp(t, 0.0, 1.0), skew)), sharp);
}

/** One vertex of a leaf: its position, its normal, and how far along the
    blade it sits — 0 at the stalk, 1 at the tip. */
void leafVertex(out vec3 p, out vec3 n, out float along) {
  /* Shape drawn from the leaf's own position rather than stored. It is
     unique and it never moves, so the hash is stable, free, and four
     bytes lighter per leaf than keeping it. */
  vec3 h = hash33(vec3(aLeaf.xy * 61.3, aLeaf.z * 37.1));
  float skew = mix(0.42, 0.95, h.x);
  float sharp = mix(0.85, 1.90, h.y);
  float aspect = mix(0.26, 0.44, h.z);
  float fold = mix(0.10, 0.28, h.x);

  bool tip = gl_VertexID >= 8;
  float row = tip ? 4.0 : float(gl_VertexID / 2);
  float lat = tip ? 0.0 : (float(gl_VertexID & 1) * 2.0 - 1.0);
  along = row / 4.0;

  /* Grown to cover for the ones that were dropped. The CPU packed a
     fraction of this chunk's leaves by exactly this curve, and covered
     area goes as count times size squared — so scaling by the inverse
     square root of the fraction holds the canopy's density constant
     while the leaf count falls away with distance. */
  float lod = clamp(1.0 - (distance(uCamPos, aLeaf.xyz) - LEAF_LOD_NEAR) / LEAF_LOD_SPAN,
                    LEAF_LOD_MIN, 1.0);
  float len = aLeaf.w * clamp(inversesqrt(lod), 1.0, LEAF_LOD_GROW);
  vec3 F = normalize(aDir.xyz);

  // A frame for the blade, rolled about its own axis so leaves on one
  // twig do not all present the same face.
  vec3 ref = abs(F.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
  vec3 S0 = normalize(cross(ref, F));
  vec3 N0 = cross(F, S0);
  /* Plus a flutter. A leaf has almost no mass, so unlike the branch it
     answers the fast part of the wind — and it is the highest frequency
     motion in the scene, which is what stops a canopy reading as a
     single carved object. */
  float roll = aDir.w + sin(uTime * 5.3 + aFlex.y * 2.7) * uWind * 0.22;
  float cr = cos(roll), sr = sin(roll);
  vec3 S = S0 * cr + N0 * sr;
  vec3 N = -S0 * sr + N0 * cr;

  float w = leafWidth(along, skew, sharp) * len * aspect;
  /* Folded along the midrib. Flat leaves read as paper at any distance;
     a shallow V catches the light down one half and shades the other,
     and that is the whole of what makes a canopy look deep. */
  vec3 rest = aLeaf.xyz + F * (along * len) + S * (w * lat) - N * (fold * w * abs(lat));

  p = sway(rest, aFlex.zw, aFlex.x, aFlex.y);
  n = normalize(N + S * (lat * fold * 2.0));
}
`;

/* ── the camera pass ──────────────────────────────────────────────── */

const CAMERA_HEAD = /* glsl */`
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
${PLANT_COMMON}
${TREE_SWAY}

uniform float uShadowSoft;
`;

const VERT_BRANCH = CAMERA_HEAD + BRANCH_GEOM + /* glsl */`
void main() {
  vColor = vec3(0.0);
  vRound = vec3(0.0);
  vDist = 0.0;

  vec3 p, n, anchor;
  branchVertex(p, n, anchor);

  float sun = sunlight(anchor, uShadowSoft) * canopyShade(anchor);
  vec3 albedo = vec3(0.052, 0.040, 0.031) * aBark.z;
  // Bark is rough and sits inside the canopy's own shade; no transmission.
  emit(p, shadeBlade(n, albedo, sun, 0.85, 0.0));
}
`;

const VERT_LEAF = CAMERA_HEAD + LEAF_GEOM + /* glsl */`
void main() {
  vColor = vec3(0.0);
  vRound = vec3(0.0);
  vDist = 0.0;

  vec3 p, n; float along;
  leafVertex(p, n, along);

  /* Shaded from where the leaf joins its twig, not from this vertex. The
     canopy lookup is the expensive term and it cannot resolve one leaf
     anyway, so asking nine times would buy nine identical answers — and
     asking at slightly different points buys a leaf lit unevenly across
     its own width, which is worse than either. */
  float sun = sunlight(aLeaf.xyz, uShadowSoft) * canopyShade(aLeaf.xyz);

  /* Darker toward the stalk, where the leaf is buried in the canopy, and
     the tips catch the sky. The same argument as a blade of grass, made
     upside down. */
  float occ = mix(0.55, 1.05, along);
  vec3 lit = normalize(n + vec3(0.0, SWARD_BIAS * 0.55, 0.0));

  /* The weather, from the cover lying on the ground the tree is standing
     on. A proxy, and an honest one: snow settles on a canopy roughly
     where it settles under it, and the alternative — asking each of four
     hundred leaves per tree which way it happens to be facing — buys a
     difference nobody can see from under the branches. Fed in at a
     fraction, because a canopy sheds most of what lands on it, and only
     the leaves facing up hold any of it.

     Without this the wood stays in full summer green in the middle of a
     snowfield, which is the one thing in the frame that says the weather
     is a coat of paint on the ground rather than a thing that happened
     to the meadow. */
  float leafRough = 0.85;
  vec3 albedo = weatherSurface(aTint.rgb, snowCover(aLeaf.xz) * 0.55, leafRough);
  emit(p, shadeBlade(lit, albedo, sun, occ, TRANSMIT));
}
`;

/* ── the light pass ───────────────────────────────────────────────── */

/**
 * The same geometry, projected from the sun instead of from the eye.
 *
 * Orthographic, because the sun is a direction and not a place. The
 * depth written is distance along the light's own axis, so the depth
 * test picks whichever leaf is nearest the sun — exactly the reduction a
 * shadow map needs, done by hardware that was going to run anyway.
 */
const LIGHT_HEAD = /* glsl */`
${PRECISION}
${CONSTANTS}
${HASH}
uniform float uTime, uWind;
${WIND_GLSL}
/* The sun's pass reads the ground for the same reason the camera's does:
   the trees are lifted onto it in the vertex shader, and a shadow map
   drawn from trees still standing on the old flat floor would put every
   shadow in the wrong place on a slope. */
${TERRAIN_GLSL}
${TREE_SWAY}

uniform vec3  uCanopyX, uCanopyY, uCanopyC, uLightDir;
uniform float uCanopyExtent, uCanopyDepth;
/* The *camera's* position, in the sun's pass. Not a mistake: the leaf
   levels are chosen by distance from the eye, and the shadow has to be
   cast by the geometry the eye is actually being shown. A map drawn
   from full-detail leaves would shade a canopy that is not there. */
uniform vec3  uCamPos;

out float vAxis;

void project(vec3 p) {
  vec3 rel = p - uCanopyC;
  vAxis = dot(rel, uLightDir);
  gl_Position = vec4(
    dot(rel, uCanopyX) / uCanopyExtent,
    dot(rel, uCanopyY) / uCanopyExtent,
    clamp(-vAxis / uCanopyDepth, -1.0, 1.0),
    1.0);
}
`;

const VERT_BRANCH_LIGHT = LIGHT_HEAD + BRANCH_GEOM + /* glsl */`
void main() {
  vec3 p, n, anchor;
  branchVertex(p, n, anchor);
  project(p);
}
`;

const VERT_LEAF_LIGHT = LIGHT_HEAD + LEAF_GEOM + /* glsl */`
void main() {
  vec3 p, n; float along;
  leafVertex(p, n, along);
  project(p);
}
`;

const FRAG_LIGHT = /* glsl */`
${PRECISION}
in float vAxis;
out vec4 outColor;

/** Coverage in red, and how far along the light this occluder sits in
    green. The depth test has already decided which occluder that is. */
void main() { outColor = vec4(1.0, vAxis, 0.0, 1.0); }
`;

/* ── the object ───────────────────────────────────────────────────── */

export class Trees {
  constructor(gl) {
    this.gl = gl;
    this.branch = new Program(gl, VERT_BRANCH, FRAG_PLANT, { name: 'trees/branch' });
    this.leaf = new Program(gl, VERT_LEAF, FRAG_PLANT, { name: 'trees/leaf' });
    this.branchLight = new Program(gl, VERT_BRANCH_LIGHT, FRAG_LIGHT, { name: 'trees/branch-sun' });
    this.leafLight = new Program(gl, VERT_LEAF_LIGHT, FRAG_LIGHT, { name: 'trees/leaf-sun' });

    /* The canopy's own shadow map: coverage in red, distance along the
       light in green, and a depth attachment doing the reduction. */
    this.map = new Target(gl, {
      width: CANOPY_SIZE, height: CANOPY_SIZE,
      format: 'rgba16f', filter: gl.LINEAR, depth: true,
    });

    this.segVao = gl.createVertexArray();
    this.segVbo = gl.createBuffer();
    this.leafVao = gl.createVertexArray();
    this.leafVbo = gl.createBuffer();

    this._segs = new Float32Array(MAX_SEGMENTS * SEG_STRIDE);
    this._leaves = new Float32Array(MAX_LEAVES * LEAF_STRIDE);

    this._bind(this.segVao, this.segVbo, this._segs.byteLength, SEG_STRIDE, 5);
    this._bind(this.leafVao, this.leafVbo, this._leaves.byteLength, LEAF_STRIDE, 4);

    /** Generated chunks, by world cell. Kept well past the live set: a
        chunk costs a few kilobytes to hold and milliseconds to regrow. */
    this.cache = new Map();
    this._key = null;
    /** Chunk generations, and buffer packs. Both watched by the tests. */
    this.grown = 0;
    this.packs = 0;
    this.trees = 0;
    this.segments = 0;
    this.leaves = 0;
    this.triangles = 0;
    this.chunk = CHUNK;
    this.reach = TREE_REACH_MIN;
    /** What the cached chunks were grown against: density, hills and
        water level. Any of the three moving means every chunk is wrong,
        not merely repacked. */
    this._shape = null;
    this._viewer = new Float32Array(2);
    /** Canopies of the live trees, for whatever wants to cast their
        shadow. Filled by the pack. */
    this.canopies = [];

    /* The light-space frame, and the last state it was rendered for. */
    this._sunX = new Float32Array(3);
    this._sunY = new Float32Array(3);
    this._sunC = new Float32Array(3);
    this._mapKey = null;
    this.mapRenders = 0;
    this._uniforms = {
      uCanopy: null,
      uCanopyX: this._sunX,
      uCanopyY: this._sunY,
      uCanopyC: this._sunC,
      uCanopyExtent: CANOPY_EXTENT,
      uCanopyOn: 0,
    };
  }

  /** Everything a shader needs to read the canopy's shadow. */
  uniforms() { return this._uniforms; }

  /**
   * What fraction of a block at the middle of the shadow map has a
   * canopy in it, and how high the highest of them sits.
   *
   * Only the tests call this — a shadow you cannot see in a screenshot
   * is exactly the kind of thing that quietly stops working, and reading
   * the map back is the one way to ask it directly rather than inferring
   * it from a few dark pixels in a dim frame.
   */
  mapCoverage(side = 768) {
    /* Wide on purpose. The middle of the map is wherever the camera is
       standing, and there is a six-metre clearing around the cluster —
       sample only the centre and the honest answer is "no trees", from a
       map that is perfectly fine. */
    const gl = this.gl;
    const o = ((CANOPY_SIZE - side) / 2) | 0;
    this.map.bind();

    /* Ask what this driver will actually hand back. RGBA/FLOAT is the
       obvious guess and it is not guaranteed for a half-float target;
       when it is refused readPixels leaves the buffer as it found it,
       which is all zeroes — indistinguishable from a map with nothing
       in it, and exactly the sort of false negative a test must not
       report as a real one. */
    const fmt = gl.getParameter(gl.IMPLEMENTATION_COLOR_READ_FORMAT);
    const type = gl.getParameter(gl.IMPLEMENTATION_COLOR_READ_TYPE);
    const half = type === gl.HALF_FLOAT;
    const buf = half ? new Uint16Array(side * side * 4) : new Float32Array(side * side * 4);

    while (gl.getError() !== gl.NO_ERROR) { /* drain */ }
    gl.readPixels(o, o, side, side, fmt, type, buf);
    const err = gl.getError();

    const value = half ? (i) => halfToFloat(buf[i]) : (i) => buf[i];
    let covered = 0, top = -1e9;
    for (let i = 0; i < side * side; i++) {
      if (value(i * 4) > 0.5) { covered++; top = Math.max(top, value(i * 4 + 1)); }
    }
    return { covered: covered / (side * side), top, err, half };
  }

  /**
   * Redraw the shadow map, if anything it depends on has moved.
   *
   * Which is rarely. It is anchored in the world and snapped to its own
   * texels, so walking a straight line redraws it every few metres and
   * standing still never does — the same bargain the chunks make, and
   * the reason a real projected shadow costs about nothing here. Turning
   * the sun does force a redraw every frame while the pad is dragged;
   * one pass of the trees is a millisecond and that is the right place
   * to spend it.
   *
   * Snapping matters more than it looks. Unsnapped, the map slides by a
   * fraction of a texel each frame and every shadow edge in the scene
   * crawls.
   */
  _renderMap(gl, light, camPos, time, wind, hills, waterY) {
    const cx = camPos[0], cz = camPos[2];
    // A frame perpendicular to the light. The reference axis is swapped
    // near the poles, where the obvious one is parallel to the light and
    // the cross product collapses.
    const ref = Math.abs(light[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    const X = unit(cross3(ref, [light[0], light[1], light[2]]));
    const Y = cross3([light[0], light[1], light[2]], X);

    /* Snapped on the light's own lattice, not the world's: the texels
       are laid out in light space, so that is the only space in which
       snapping stops them crawling. */
    const texel = (2 * CANOPY_EXTENT) / CANOPY_SIZE;
    const raw = [cx, FLOOR_Y, cz];
    const sx = Math.round(dot3(raw, X) / texel) * texel;
    const sy = Math.round(dot3(raw, Y) / texel) * texel;
    const sz = dot3(raw, [light[0], light[1], light[2]]);

    this._sunX.set(X);
    this._sunY.set(Y);
    for (let i = 0; i < 3; i++) {
      this._sunC[i] = X[i] * sx + Y[i] * sy + light[i] * sz;
    }

    const key = `${sx.toFixed(3)}|${sy.toFixed(3)}|${sz.toFixed(3)}`
      + `|${light[0].toFixed(4)},${light[1].toFixed(4)},${light[2].toFixed(4)}`
      /* The wood moves, so the map has to be redrawn while it does —
         a shadow of a still tree under a leaning one is worse than none.
         With the wind off it goes back to being redrawn only when the
         ground under it moves, which is the cheap case and the common
         one. */
      /* And the ground, because the trees are lifted onto it in the
         vertex shader: raise the hills and every trunk in the map moves
         without a single instance record changing. */
      + `|${this.packs}|${hills.toFixed(3)}`
      + `|${wind > 0.001 ? time.toFixed(2) : 'calm'}`;
    if (key === this._mapKey) return;
    this._mapKey = key;

    const common = {
      uCanopyX: this._sunX,
      uCanopyY: this._sunY,
      uCanopyC: this._sunC,
      uCanopyExtent: CANOPY_EXTENT,
      uCanopyDepth: CANOPY_EXTENT * 2,
      uLightDir: light,
      uCamPos: camPos,
      uTime: time,
      uWind: wind,
      uHills: hills,
      uWaterY: waterY,
    };

    this.map.bind();
    // Green clears far below anything a tree can reach, so an untouched
    // texel can never be mistaken for an occluder.
    gl.clearColor(0, -1e5, 0, 1);
    gl.clearDepth(1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
    gl.depthMask(true);
    gl.disable(gl.CULL_FACE);

    gl.bindVertexArray(this.segVao);
    this.branchLight.use(common);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, SEG_VERTS, this.segments);

    gl.bindVertexArray(this.leafVao);
    this.leafLight.use(common);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, LEAF_VERTS, this.leaves);

    gl.bindVertexArray(null);
    gl.disable(gl.DEPTH_TEST);
    this.mapRenders++;
  }

  _bind(vao, vbo, bytes, stride, attribs) {
    const gl = this.gl;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, bytes, gl.DYNAMIC_DRAW);
    for (let i = 0; i < attribs; i++) {
      gl.enableVertexAttribArray(i);
      gl.vertexAttribPointer(i, 4, gl.FLOAT, false, stride * 4, i * 16);
      gl.vertexAttribDivisor(i, 1);
    }
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /**
   * Grow one chunk: at most SUB*SUB trees, one per sub-cell, jittered
   * inside it. The sub-grid is what caps the count and what guarantees
   * the separation, and it does both without a chunk ever consulting its
   * neighbours — which is the only reason chunks can be generated in any
   * order and dropped independently.
   */
  _growChunk(ix, iz, density, hills, waterY) {
    /* Sized for the chunk's own hard cap — SUB*SUB trees, each bounded by
       the recursion — so growing can never outrun the array and never has
       to reallocate half way down a branch. */
    const out = {
      segs: new Float32Array(SUB * SUB * 160 * SEG_STRIDE),
      leaves: new Float32Array(SUB * SUB * 460 * LEAF_STRIDE),
      segN: 0,
      leafN: 0,
    };

    const canopies = [];
    for (let sz = 0; sz < SUB; sz++) {
      for (let sx = 0; sx < SUB; sx++) {
        const k = sz * SUB + sx;
        if (seed(ix * SUB + sx, iz * SUB + sz, 11) > TREE_CHANCE * density) continue;

        const cx = (ix * CHUNK) + (sx + 0.5) * SUB_CELL;
        const cz = (iz * CHUNK) + (sz + 0.5) * SUB_CELL;
        const jx = (seed(ix * SUB + sx, iz * SUB + sz, 12) - 0.5) * 2 * TREE_JITTER * SUB_CELL;
        const jz = (seed(ix * SUB + sx, iz * SUB + sz, 13) - 0.5) * 2 * TREE_JITTER * SUB_CELL;
        const x = cx + jx, z = cz + jz;

        // The clearing. Measured before anything is grown, so a rejected
        // tree costs one hypot.
        if (x * x + z * z < CLEAR_R * CLEAR_R) continue;

        /* And the lake, in the same place and for the same reason: a
           drowned tree is rejected for four sines instead of a whole
           recursion, a hundred and sixty segments and four hundred
           leaves. The test is the bare one — no margin, no shoreline
           band — because a trunk standing at the waterline is what the
           edge of a wood by a lake actually looks like, and a margin
           would be a second, invisible shoreline to keep in step with
           the real one. */
        if (waterDepthAt(x, z, hills, waterY) > 0) continue;

        const before = { segs: out.segN, leaves: out.leafN };
        const canopy = growTree(out, rngFor(ix * SUB + sx, iz * SUB + sz, k), x, z);
        if (out.segN === before.segs) continue;
        if (canopy.n) {
          canopies.push([x, canopy.y / canopy.n, z, Math.max(canopy.r, 0.6)]);
        }
      }
    }

    /* Shuffled, so that taking the first half of a chunk's leaves takes
       half of every tree in it rather than all of the first two and none
       of the rest. Deterministic, from the chunk's own cell, so a chunk
       dropped and regrown comes back identical. */
    shuffleLeaves(out.leaves, out.leafN, rngFor(ix, iz, 977));

    this.grown++;
    return {
      segs: out.segs.subarray(0, out.segN * SEG_STRIDE),
      leaves: out.leaves.subarray(0, out.leafN * LEAF_STRIDE),
      segN: out.segN,
      leafN: out.leafN,
      cx: (ix + 0.5) * CHUNK,
      cz: (iz + 0.5) * CHUNK,
      canopies,
    };
  }

  /** Make sure every chunk within reach exists, then pack the live ones. */
  _ensure(cx, cz, density, reach, hills, waterY) {
    /* One chunk of margin past the reach, so the far edge of what has
       been grown is always beyond the far edge of what can be seen. */
    const span = Math.ceil((reach + CHUNK) / CHUNK);
    const ix0 = Math.floor(cx / CHUNK);
    const iz0 = Math.floor(cz / CHUNK);
    /* Keyed on the levels as well as the chunks. The chunk set changes
       every sixteen metres; how much of each chunk is worth drawing
       changes rather sooner than that, so the viewer goes into the key
       quantised to a few metres. A repack is a memcpy — cheap enough to
       do four times as often as growing, and far cheaper than a draw
       call per chunk, which was the alternative. */
    const key = `${ix0}|${iz0}|${density.toFixed(3)}|${reach.toFixed(1)}`
      + `|${Math.round(cx / LOD_STEP)}|${Math.round(cz / LOD_STEP)}`
      // The ground and the water on it decide which trees exist at all.
      + `|${hills.toFixed(3)}|${waterY.toFixed(3)}`;
    if (key === this._key) return;
    this._key = key;

    /* A density change invalidates every chunk; a step does not. So does
       moving the water or the hills, because both change which sub-cells
       came up drowned — the trees are lifted onto the terrain in the
       vertex shader, so the hills alone never used to reach this far
       back into the pipeline, and now they do. */
    const shape = `${density.toFixed(4)}|${hills.toFixed(4)}|${waterY.toFixed(4)}`;
    if (this._shape !== shape) { this.cache.clear(); this._shape = shape; }

    const live = [];
    const limit = (reach + CHUNK) * (reach + CHUNK);
    for (let dz = -span; dz <= span; dz++) {
      for (let dx = -span; dx <= span; dx++) {
        const ix = ix0 + dx, iz = iz0 + dz;
        // Chunk centre against the reach, with the chunk's own half
        // diagonal as slack so a corner never gets cut off.
        const mx = (ix + 0.5) * CHUNK - cx;
        const mz = (iz + 0.5) * CHUNK - cz;
        if (mx * mx + mz * mz > limit) continue;

        const ck = `${ix},${iz}`;
        let chunk = this.cache.get(ck);
        if (!chunk) {
          chunk = this._growChunk(ix, iz, density, hills, waterY);
          this.cache.set(ck, chunk);
        }
        live.push(chunk);
      }
    }

    // Drop what has fallen well out of range, but keep a generous margin:
    // walking a few metres back and forth must not regrow anything.
    if (this.cache.size > live.length * 3) {
      for (const [k] of this.cache) {
        if (this.cache.size <= live.length * 2) break;
        const [ix, iz] = k.split(',').map(Number);
        const mx = (ix + 0.5) * CHUNK - cx, mz = (iz + 0.5) * CHUNK - cz;
        if (mx * mx + mz * mz > limit * 4) this.cache.delete(k);
      }
    }

    this._pack(live, cx, cz);
  }

  /** Concatenate the live chunks into the two instance buffers. */
  _pack(live, vx, vz) {
    const gl = this.gl;
    let segN = 0, leafN = 0, trees = 0, grown = 0;
    const canopies = [];

    for (const c of live) {
      if (segN + c.segN <= MAX_SEGMENTS) {
        this._segs.set(c.segs, segN * SEG_STRIDE);
        segN += c.segN;
      }
      /* Only as much of this chunk's leaves as its distance earns. They
         were shuffled when the chunk was grown, so a prefix is a fair
         sample across every tree in it. */
      const d = Math.hypot(c.cx - vx, c.cz - vz);
      const want = Math.min(c.leafN, Math.ceil(c.leafN * leafLod(d)));
      grown += c.leafN;
      if (leafN + want <= MAX_LEAVES) {
        this._leaves.set(c.leaves.subarray(0, want * LEAF_STRIDE), leafN * LEAF_STRIDE);
        leafN += want;
      }
      trees += c.canopies.length;
      for (const k of c.canopies) canopies.push(k);
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, this.segVbo);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this._segs, 0, segN * SEG_STRIDE);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.leafVbo);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this._leaves, 0, leafN * LEAF_STRIDE);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    this.segments = segN;
    this.leaves = leafN;
    /** What the wood would have cost without the levels. Reported so the
        saving is visible rather than merely believed. */
    this.leavesGrown = grown;
    this.trees = trees;
    this.canopies = canopies;
    this.packs++;
  }

  /**
   * The nearest canopies to a point, packed as xyz-centre and radius.
   *
   * Published for the birds, which need somewhere to sit. It is the same
   * list the shadow map is drawn from — the real trees, at their real
   * heights, out of the recursion that grew them — so a sparrow perched
   * by this is in the tree rather than beside it. Deriving the positions
   * a second time from the placement hash was the alternative and would
   * have got the trunk right and the canopy height wrong, because a
   * tree's height comes out of its own growth stream and not out of its
   * cell.
   *
   * @param {number} x
   * @param {number} z
   * @param {Float32Array} out  4 floats per slot; the length sets how many
   * @returns {number} slots filled
   */
  perches(x, z, out) {
    const slots = Math.floor(out.length / 4);
    if (!slots || !this.canopies.length) return 0;

    /* A partial selection rather than a sort. The list runs to a couple
       of hundred trees and only the first handful are wanted, so this is
       n*k compares against n log n and no allocation at all — and it runs
       every frame the wood is repacked. */
    let filled = 0;
    const best = this._perchD || (this._perchD = new Float64Array(slots));
    for (const c of this.canopies) {
      const dx = c[0] - x, dz = c[2] - z;
      const d = dx * dx + dz * dz;
      if (filled === slots && d >= best[filled - 1]) continue;

      let i = Math.min(filled, slots - 1);
      while (i > 0 && best[i - 1] > d) {
        best[i] = best[i - 1];
        out.copyWithin(i * 4, (i - 1) * 4, i * 4);
        i--;
      }
      best[i] = d;
      out[i * 4] = c[0];
      out[i * 4 + 1] = c[1];
      out[i * 4 + 2] = c[2];
      out[i * 4 + 3] = c[3];
      if (filled < slots) filled++;
    }
    return filled;
  }

  /**
   * Draw into the currently bound target — the same one the cat and the
   * grass draw into, sharing its depth buffer.
   */
  /**
   * Grow whatever is in range and draw the sun's view of it.
   *
   * Separate from the drawing, and first, because four other shaders
   * read the map this produces — the marched floor, the grass, the cat,
   * and the leaves themselves. Any of them running before it would be
   * shading against the previous frame's wood.
   */
  prepare(camPos, lightDir, time, opts) {
    if (!opts.on) {
      this.trees = 0;
      this.triangles = 0;
      this._uniforms.uCanopyOn = 0;
      return;
    }
    this.reach = Math.max(TREE_REACH_MIN,
      Math.min(TREE_REACH_MAX, opts.radius * TREE_REACH));
    this._ensure(camPos[0], camPos[2],
      Math.max(0.05, Math.min(1, opts.density)), this.reach,
      opts.hills, opts.waterY);
    if (!this.segments) { this.triangles = 0; this._uniforms.uCanopyOn = 0; return; }

    this._renderMap(this.gl, lightDir, camPos, time,
                    opts.wind, opts.hills, opts.waterY);
    this._uniforms.uCanopy = this.map.texture;
    this._uniforms.uCanopyOn = 1;
  }

  draw(camera, env, opts) {
    if (!opts.on || !this.segments) { this.triangles = 0; return; }
    const gl = this.gl;

    this._viewer[0] = camera.pos[0];
    this._viewer[1] = camera.pos[2];

    const common = {
      uCamPos: camera.pos,
      uRight: camera.right,
      uUp: camera.up,
      uFwd: camera.fwd,
      uFocal: camera.focal,
      uAspect: camera.aspect,
      uJitter: opts.jitter,
      uViewer: this._viewer,
      uRadius: this.reach,
      uWind: opts.wind,
      uFog: env.fog,
      uHills: env.hills,
      uWaterY: env.waterY,
      uRain: env.weather.rain,
      uSnow: env.weather.snow,
      // A leaf's rim is lit by the sky, so the wood has to be told what
      // is in it — see plantUniforms, which says this once for everything
      // that does not hand-roll its own list like this one.
      uOvercast: env.weather.overcast,

      uLightDir: env.dir,
      uTint: env.tint,
      uDay: env.day,
      uAmbient: env.ambient,
      uShadowSoft: env.shadowSoft,
      uShadowSteps: env.shadowSteps,
      uShadowNoise: env.shadowNoise,

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

      uCatCapA: env.catCapA,
      uCatCapB: env.catCapB,
      uCatBound: env.catBound,
      uCatCaps: env.catCaps,

      ...this._uniforms,
    };

    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
    gl.depthMask(true);
    // Bark is a closed tube and could be culled; leaves are two-sided and
    // cannot. One state for both is cheaper than the state change.
    gl.disable(gl.CULL_FACE);

    gl.bindVertexArray(this.segVao);
    this.branch.use(common);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, SEG_VERTS, this.segments);

    gl.bindVertexArray(this.leafVao);
    this.leaf.use(common);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, LEAF_VERTS, this.leaves);

    gl.bindVertexArray(null);
    gl.disable(gl.DEPTH_TEST);

    this.triangles = this.segments * (SEG_VERTS - 2) + this.leaves * (LEAF_VERTS - 2);
  }

  dispose() {
    const gl = this.gl;
    this.branch.dispose();
    this.leaf.dispose();
    gl.deleteVertexArray(this.segVao);
    gl.deleteVertexArray(this.leafVao);
    gl.deleteBuffer(this.segVbo);
    gl.deleteBuffer(this.leafVbo);
    this.branchLight.dispose();
    this.leafLight.dispose();
    this.map.dispose();
  }
}
