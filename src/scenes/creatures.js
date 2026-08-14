/* ── scenes/creatures.js ─────────────────────────────────────────────
   The things in the meadow that are alive and are not the cat.

   Two so far, and they are built at opposite ends of the pipeline on
   purpose, because they are opposite kinds of thing:

   A butterfly is an *object*. It is opaque, it is lit, the grass stands
   in front of it and the trees shade it — so it is geometry, drawn into
   the same target and the same depth buffer as the grass, out of the
   same shared plant header. It costs four triangles.

   A firefly is a *light*. It has no body worth drawing at this scale;
   what you see is the glow, and a glow is something added to the image
   rather than a surface that occludes. So it is drawn after the tonemap
   like the impact flares and the rain, additively, hiding behind the
   scene's published depth instead of writing any.

   Neither has a buffer, an attribute or a particle record. Every one of
   them is a hash of its own index, advanced by the clock and wrapped
   into a patch that follows the eye — so the CPU's whole contribution is
   a vertex count, and turning the density up is a bigger number in a
   draw call and nothing else.

   ── where they are allowed to be ─────────────────────────────────
   Both ask the ground the same questions the grass does. Nothing hovers
   over the lake, because a butterfly a metre above open water is the
   kind of detail that is invisible until it is the only thing you can
   see; and both thin out under snow, because that is what the meadow
   they belong to is doing. Getting that for free is the reason the
   height field and the water level were made single answers rather than
   numbers each pass keeps its own copy of.

   ── what they deliberately are not ───────────────────────────────
   A butterfly is two wings. No body, no antennae, no legs. At the size
   one occupies on screen — a few dozen pixels at the very closest, four
   or five at any honest distance — a body is a dark smudge between two
   wings, and the wings are doing all the work: what reads as a butterfly
   is the flicker of two surfaces alternately catching and losing the
   sun, and that is exactly what two hinged quads give you.

   A firefly is not even that. It is a point.
   ------------------------------------------------------------------ */

import { Program } from '../core/program.js';
import {
  PRECISION, CONSTANTS, HASH, ROTATE, SIMPLEX3,
} from '../shaders/common.js';
import {
  CLUSTER_UNIFORMS, CLUSTER_FIELD, CLUSTER_LAYERS, CLUSTER_SHADOW, SKY,
} from './cluster.js';
import { CAT_PROXY_GLSL } from './cat/index.js';
import { PLANT_COMMON, FRAG_PLANT, plantUniforms } from './plant.js';
import { CLUMP_GLSL, seed } from './clumps.js';
import { terrainHeight, waterDepthAt } from './terrain.js';
import { snowCoverAt } from './weather.js';
import { TERRAIN_GLSL } from './terrain.js';

/* ── how many, and how far ────────────────────────────────────────
   The counts are ceilings; the sliders scale them. Each kind gets its
   own control because they are not two amounts of one thing — one is a
   thing you watch in daylight and the other is a thing you turn the
   exposure down to find, and a single "wildlife" slider would be a
   control that could never be set correctly for either. */

/* ── and when ─────────────────────────────────────────────────────
   The hour decides how many of each there are, and the three answers
   are not three settings of one thing.

   A butterfly is out in the sun. A firefly is a thing you can only find
   once the sun is not. A sparrow is neither: it does not glow and it
   does not disappear, it is asleep in a hedge, and what that looks like
   from fifteen metres away is an empty field.

   All three read the same `day` the sky is drawn from, and that number
   is deliberately wider than the sun's own crossing — the light is still
   in the sky after the sun is under it, and so is the flying.

   The roost is a *fade* and not an errand. The tempting version sends
   every flock to a perch at dusk, and it cannot be done from here: a
   flock's position is derived from where it is inside a pass whose
   period is its flight plus its stay, so lengthening the stay at dusk
   changes the period, which changes which pass it is on, which puts the
   whole flock somewhere else in one frame. A bird that is not visible at
   night is the true statement anyway.

   Only the firefly asks whether there is a clock at all. With the pad in
   charge `day` is exactly 1 — which is what keeps every frame taken
   before this file heard of an hour still true — and 1 means noon, so a
   nocturnal population read straight off it would leave the control
   switched on and nothing in the picture. A scene with no time of day
   has no night in it, and nothing in it can be nocturnal. */
const smooth = (a, b, x) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
/** Out in the sun: up with the morning, down at dusk. */
const dayPart = (day) => smooth(0.18, 0.55, day);
/** Awake in the same light, and simply not visible outside it. */
const roostPart = (day) => smooth(0.10, 0.45, day);
/** And the other way round, where there is an hour to read. */
const nightPart = (day, timed) => (timed ? 1 - smooth(0.10, 0.45, day) : 1);

/** Butterflies at full density. Four triangles each. */
const BUTTERFLIES = 180;
/** Fireflies at full density. Two triangles each, and no shading. */
const FIREFLIES = 260;
/** Birds in a flock at full density. The control moves this and nothing
    else about the population — see the flock block in VERT_BIRD. */
const PER_FLOCK = 15;

/* Flocks handed to the shader at once, nearest first.

   Twenty-four, and the slider still scales it, so what this really sets
   is how full the sky can get at the top of the control rather than how
   full it is.

   It was six, on the argument that past half a dozen groups the eye
   stops counting groups and goes back to seeing a scatter. That holds
   for a *small* patch of meadow, and stopped holding once the reach
   control could open the view to a hundred and twenty metres: over that
   much ground six flocks is not a legible few, it is an empty field with
   a handful of birds in it. The count that reads correctly is a function
   of how much ground is in shot, and the reach is what decides that.

   Cheap to raise. The whole population is three uniform arrays and a
   vertex count — at this ceiling, seventy-five thousand vertices and
   three hundred and thirty-six floats — and most slots draw nothing
   most of the time, because two in five cells hold a flock at all and
   the far ones are faded out before they are shaded. */
const MAX_FLOCKS = 24;

/** How high a perched bird sits above the ground it is standing on.
    Clear of a 32 cm blade: a sparrow at its own leg height is a sparrow
    buried in the sward, which reads as one that has disappeared. */
const GROUND_PERCH = 0.40;

/** How far apart a flock settles on open ground. Wider than a crown,
    because nothing is holding them together down there. */
const GROUND_GATHER = 1.6;

/* Flock slots: the five by five of world cells around the eye.

   Five and not three, and the two extra rings are not generosity. The
   window shifts by a whole cell whenever the eye crosses a boundary, so
   any flock still visible at the window's edge blinks when it does — the
   very bug the world grid was introduced to fix, moved outward by one
   step. What makes it go away is the fade finishing *inside* the
   guaranteed radius, and the guarantee for a (2k+1) window is k cells,
   because the eye can sit anywhere in its own. Three cells guarantees
   one; five guarantees two, and the fade is clamped to that.

   Most of these slots draw nothing: two in five hold a flock at all, and
   the far ones are faded out before they are shaded. They cost a handful
   of culled vertices each. */
const FLOCK_SLOTS = 25;
const FLOCK_WINDOW = 5;

/** Tree crowns handed to the birds to sit in. Six is a couple of trees'
    worth of choice at the range a flock can reach, and the array costs
    twenty-four floats. */
const PERCH_N = 6;


/* ── how far each kind is drawn ───────────────────────────────────
   One control, three ratios and three ceilings — the same bargain the
   cover reach makes with the grass, the flowers and the wood, and made
   here for the same reason: they are three ranges of one quantity, not
   three independent questions. A butterfly is seven centimetres and
   stops being anything at twenty metres; a firefly is a light and
   carries further than its body ever could; a sparrow is the biggest of
   the three and flies highest, so it earns the longest reach and the
   ceiling that matters.

   The ceilings are where each stops being worth buying rather than where
   it stops being affordable. Past a butterfly's, the instances are being
   spent on something under a pixel. */
const REACH_RATIO = { fly: 0.50, glow: 0.80, bird: 1.60 };
const REACH_CAP = { fly: 30.0, glow: 55.0, bird: 130.0 };
const REACH_MIN = 8.0;

/** Where the control starts and stops. */
export const CREATURE_REACH_MIN = 10;
export const CREATURE_REACH_MAX = 120;
export const CREATURE_REACH_DEFAULT = 45;

/** The patch one kind lives in, from the shared control. */
function reachFor(kind, reach) {
  const r = Math.max(CREATURE_REACH_MIN, Math.min(CREATURE_REACH_MAX, reach));
  return Math.max(REACH_MIN, Math.min(REACH_CAP[kind], r * REACH_RATIO[kind]));
}

/* Density is a density, so the count follows the area. Winding the reach
   out at a fixed count would thin the population to nothing, which is
   what the control would then really be doing — and the whole argument
   for a separate density slider is that the two are different questions.

   These are the counts at the reference patch each kind was tuned at,
   divided by its area, so the setting that used to be right still is. */
const FLY_PER_M2 = 180 / (20 * 20);
const GLOW_PER_M2 = 260 / (34 * 34);
/** Hard ceilings, so a wide reach cannot outrun the vertex budget. */
const FLY_MAX = 900;
const GLOW_MAX = 1400;

/* ── placement ────────────────────────────────────────────────────
   Shared by both, because "scatter N of these around the eye, on the
   ground, out of the water" is one problem and having two answers to it
   is how the fireflies end up in a lake the butterflies are avoiding. */

const SCATTER_GLSL = /* glsl */`
/**
 * Where creature number id is standing, and how much of it there is.
 *
 * The hashed point is wrapped into a box that follows the eye, so the
 * population is infinite and nothing is ever spawned or retired. The
 * wrap is a teleport, which is why the fade at the box edge is not
 * decoration: without it, a creature crosses the boundary and reappears
 * on the far side of the field at full brightness.
 *
 * Returns 0 in the fade if there is nothing here worth drawing — over
 * the water, under a drift, or past the rim.
 */
float scatter(int id, float box, float salt, out vec3 pos, out vec3 h) {
  h = hash33(vec3(float(id) * 0.6131 + salt, 7.7, 3.1));

  vec2 c = uCamPos.xz;
  vec2 p0 = h.xy * box;
  vec2 home = c - box * 0.5 + mod(p0 - c + box * 0.5, vec2(box));

  pos = vec3(home.x, terrainH(home), home.y);

  /* Nothing over the lake, and nothing standing in a drift. Both are the
     same test the flowers get, asked of the same two fields — which is
     the entire reason a butterfly knows where the shore is without
     anybody having told it. */
  if (waterDepth(home) > -0.05) return 0.0;
  float snowed = 1.0 - snowCover(home) * 0.85;

  vec2 e = abs(home - c) / (box * 0.5);
  return snowed * (1.0 - smoothstep(0.62, 1.0, max(e.x, e.y)));
}
`;

/* ── butterflies ──────────────────────────────────────────────────── */

/** Wing size, in metres. A cabbage white is about this across. */
const WING_L = 0.070;
const WING_W = 0.058;

/** How long a kick takes to fade, in kick-periods — see the block in
    VERT_FLY, which is generated from this and derives its own loop
    bound from it. Here rather than in the shader so that the two cannot
    be edited apart. */
const KICK_LIFE = 3.0;

/* The same stack the grass and the wood are built on, in the same
   order. A butterfly is shaded by the cluster, the cat, the canopy and
   the hills exactly as a blade of grass is, and it is shaded by them
   *identically* because it is calling the same functions with the same
   uniforms — not because two shaders were written to match. */
const FLY_HEAD = /* glsl */`
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

uniform float uShadowSoft;
`;

const VERT_FLY = FLY_HEAD + CLUMP_GLSL + SCATTER_GLSL + /* glsl */`
#define WING_L ${WING_L.toFixed(4)}
#define WING_W ${WING_W.toFixed(4)}

/* How long a kick takes to fade, in kick-periods.

   The knob to reach for if the flight wants to be twitchier or looser.
   Measured across the whole population, it moves the *excursion* and
   almost nothing else — which is not what it looks like it should do,
   so the numbers are here rather than left to be rediscovered:

     KICK_LIFE   speed p90 / p99      excursion p90
        2          1.37 / 1.97           0.11 m
        3          1.41 / 1.99           0.22 m     <- here
        4          1.44 / 2.03           0.36 m
        5          1.46 / 2.07           0.52 m

   Speed barely moves because a kick's peak velocity is its amplitude
   times the kick rate — s'(0) is 1 whatever the life — and the top of
   the speed distribution belongs to the anchor slide anyway, which knows
   nothing about any of this. Normalising the amplitude by the shape's
   peak (4L/27) to hold the excursion still was tried and does not work:
   more life also means more kicks overlapping, so the excursion still
   grows, and the correction only drags the median speed the other way.
   Two knobs that each do one thing cleanly is not available here; one
   knob that does one thing is, and this is it.

   KICKS is derived, not typed. Getting it wrong truncates a kick before
   it has faded and puts a periodic pop in every animal at once, which is
   the kind of coupling that should be impossible rather than documented
   — the ceiling plus one, because at any instant one kick is partway
   into its first period and the oldest is partway through its last. */
#define KICK_LIFE ${KICK_LIFE.toFixed(1)}
#define KICKS ${Math.ceil(KICK_LIFE) + 1}

/** How far this kind is drawn, in metres — the shared reach control
    after its own ratio and ceiling have been applied. */
uniform float uBox;

/** The flower controls, as the flight sees them: how many cells grow a
    clump, and how wide one is. Zero for the first means the flowers are
    switched off and there is nothing to visit. */
uniform float uClumpChance;
uniform float uClumpSpread;

/* Five species, which is four more than one and far fewer than anyone
   would notice a sixth of. Read by a hash, so a given butterfly keeps
   its colours for as long as it is on screen. */
const vec3 WINGS[5] = vec3[5](
  vec3(0.92, 0.86, 0.66),   // white
  vec3(0.86, 0.42, 0.09),   // fritillary
  vec3(0.24, 0.34, 0.72),   // blue
  vec3(0.72, 0.68, 0.20),   // brimstone
  vec3(0.58, 0.16, 0.22)    // red admiral
);

void main() {
  vColor = vec3(0.0);
  vRound = vec3(0.0);
  vDist = 0.0;

  int id = gl_VertexID / 12;
  int v = gl_VertexID % 12;
  float side = v < 6 ? -1.0 : 1.0;      // which wing
  int corner = v % 6;

  // Two triangles over the unit square: along the body, across the wing.
  float along = (corner == 1 || corner == 3 || corner == 4) ? 1.0 : 0.0;
  float out_ = (corner == 2 || corner == 4 || corner == 5) ? 1.0 : 0.0;

  vec3 ground, h;
  float live = scatter(id, uBox, 0.0, ground, h);
  if (live <= 0.004) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }

  vec3 h2 = hash33(vec3(float(id) * 0.2711 + 19.3, 5.3, 11.9));

  /* ── where it is going, and when it leaves ──
     A butterfly's day is: work a flower for a few seconds, then dash to
     another one. Both halves matter and the second is the one that was
     missing. An orbit whose speed and radius vary is still an orbit —
     the eye locks onto a closed path within about half a minute however
     unevenly it is walked, and a whole field of them locked to one clump
     each reads as a set of fixed rings. What breaks a circle is leaving
     it.

     So time is cut into legs, and each leg picks its own clump out of
     the nine cells around home. The pick varies with the leg number as
     well as with the animal, so no two neighbours sit on the same flower
     and none of them sits on one forever. */
  /* The leg's total length is fixed per animal; how much of it is spent
     crossing is not. That order matters and is not an implementation
     detail: the crossing has to be long enough for the distance, and the
     distance is not known until the leg number has picked the two
     clumps — so the span cannot depend on it or the two definitions
     chase each other. Fixed span, derived split. */
  float legSpan = 9.0 + h.y * 6.0;
  float legT = (uTime + h.x * 40.0) / legSpan;
  float leg = floor(legT);
  float legS = (legT - leg) * legSpan;     // seconds into this leg

  /* Two clumps, this leg's and the next. With the flowers off, or on a
     bare patch, both fall back to the home point and the animal simply
     wanders there — the honest answer rather than a fallback, because
     there is nothing to visit. */
  vec2 fromC = ground.xz, toC = ground.xz;
  float fromR = 0.75, toR = 0.75;
  if (uClumpChance > 0.0) {
    vec3 pk = hash33(vec3(float(id) * 0.919, leg, 31.7));
    vec2 c; float cr;
    if (pickClump(ground.xz, uClumpChance, uClumpSpread, pk.x, c, cr)) {
      fromC = c;
      // Circled a little wider than the clump, so the orbit passes over
      // the flowers rather than around the outside of them.
      fromR = cr * 1.15 + 0.22;
    }
    if (pickClump(ground.xz, uClumpChance, uClumpSpread, pk.y, c, cr)) {
      toC = c;
      toR = cr * 1.15 + 0.22;
    }
  }

  /* ── how long the crossing takes ──
     As long as the crossing needs at a butterfly's pace, not a fixed
     number of seconds. A constant dash time means the speed is whatever
     the gap happens to be divided by it — and the gaps here run to eight
     metres, which at the second and a half the first version allowed was
     twelve metres per second. That is a bird. Deriving the duration from
     the distance instead bounds the speed no matter how far apart two
     clumps land, which is the only version that cannot be wrong for some
     arrangement of the grid.

     CRUISE is what a butterfly actually does between flowers. The peak
     of a smoothstep is 1.5 times its average, so the fastest instant of
     any crossing is about 1.5 * CRUISE whatever the distance. */
  const float CRUISE = 1.05;
  float crossing = length(toC - fromC);
  float legDash = clamp(crossing / CRUISE, 0.8, legSpan * 0.7);
  float legDwell = legSpan - legDash;

  /* Where it is between the two. Zero for the whole dwell, then a
     smoothstep across the dash — so the departure and the arrival are
     both gentle and the middle is quick, which is what the animal
     actually does and also what keeps the path differentiable. */
  float e = clamp((legS - legDwell) / legDash, 0.0, 1.0);
  float ease = e * e * (3.0 - 2.0 * e);
  float dEase = 6.0 * e * (1.0 - e) / legDash;

  /* The orbit's centre slides from one to the other. Sliding the centre
     rather than blending two finished paths is what makes the transit
     come out for free: the animal keeps turning while it travels, its
     velocity is the orbit's plus the centre's, and the heading below —
     which is just the derivative — leans into the crossing and back out
     of it without anything having to be written for the dash itself. */
  vec2 target = mix(fromC, toC, ease);
  vec2 dTarget = (toC - fromC) * dEase;

  /* Wound in while it is travelling. A butterfly crossing open grass is
     not circling anything, and leaving the full radius on turns a dash
     into a wide curve rather than a crossing. */
  float transit = 4.0 * ease * (1.0 - ease);        // 0 at both ends, 1 mid-dash
  float reach = mix(fromR, toR, ease) * (1.0 - 0.62 * transit);

  /* ── how it gets there: a train of kicks ──
     There is no orbit here and that is the entire point of this version.

     Everything before it was built out of sines, and sines have neither
     of the two features that make a butterfly's path recognisable: they
     have no corners, and they are symmetric in time. Modulating the
     phase changes how fast the curve is *walked*; it cannot change that
     it is a curve. A closed path is something the eye finds inside half
     a minute however unevenly it is travelled, and no amount of wobble
     laid over it helps — the wobble is smooth too.

     What an animal actually does is kick, coast, and kick again. So the
     displacement is written as the sum of the last few kicks, each one a
     sharp onset that decays away:

       s(a)  = a * (1 - a/L)^2        a = kick-periods since it landed
       s'(a) = (1 - a/L)(1 - 3a/L)

     s(0) = 0 with s'(0) = 1, so the velocity jumps the instant a kick
     arrives — that discontinuity *is* the corner, and it costs nothing
     because it is what the arithmetic already does. s(L) = s'(L) = 0, so
     a kick leaves the window having already faded to nothing and there
     is no pop when it drops off the end.

     Finite memory is what keeps it a closed form. A real impulse
     response never quite reaches zero and its sum would need every kick
     the animal has ever made; forcing it to zero at L bounds the loop at
     four iterations, which is what lets a butterfly still be a function
     of its own index and the clock, with no buffer anywhere. */
  float kickRate = 2.0 + h2.x * 1.2;              // kicks per second
  float tau = uTime * kickRate + h2.y * 64.0;
  float kNow = floor(tau);

  vec2 travel = crossing > 1e-3 ? (toC - fromC) / crossing : vec2(1.0, 0.0);

  vec3 off = vec3(0.0);      // displacement from the anchor
  vec3 dOff = vec3(0.0);     // and its derivative, in metres per second
  vec3 newest = vec3(0.0);   // the kick that has just landed, for the roll

  for (int i = 0; i < KICKS; i++) {
    float kIdx = kNow - float(i);
    float age = tau - kIdx;
    if (age >= KICK_LIFE) continue;

    vec3 kh = hash33(vec3(float(id) * 0.4111, kIdx, 5.9));

    /* Scattered while it is working a flower, aimed while it is crossing
       to the next one. The same train therefore produces both the milling
       about and the surges of a long move, which is what makes a crossing
       read as a chain of dashes rather than as a slide — there is no
       second mechanism for travel. */
    float ang = kh.x * 6.2831;
    vec2 lat = normalize(mix(vec2(cos(ang), sin(ang)), travel, 0.62 * transit)
                         + vec2(1e-5, 0.0));

    /* Bigger kicks when crossing. A butterfly beats harder to travel and
       barely at all to hold station over a flower, and the amplitude is
       the only place that difference can live once both are the same
       train.

       Capped, and the cap is not cosmetic. A kick's peak *speed* is its
       amplitude times the kick rate, so tying the amplitude to the clump
       radius alone means a wide clump throws the animal about at metres
       per second — measured, the uncapped version peaked at 6.4 m/s,
       which is a bird. Capped and calmed it runs at a median of 0.45 and
       a 99th of 2.0, while the turn rate is barely touched: 101 rad/s at
       the 99th against 112 before. Which is the whole point of tuning
       these two separately — the sharpness lives in the *onset* and the
       speed lives in the amplitude, and only one of them was wrong. */
    float amp = min(reach, 0.85) * (0.20 + 0.24 * kh.y) * (1.0 + 0.8 * transit);
    // Some of every kick goes upward, most of it does not.
    float up = (kh.z - 0.42) * 0.50;

    float x = age / KICK_LIFE;
    float s = age * (1.0 - x) * (1.0 - x);
    float ds = (1.0 - x) * (1.0 - 3.0 * x);

    vec3 d = vec3(lat.x, up, lat.y);
    off += d * (amp * s);
    dOff += d * (amp * ds * kickRate);
    if (i == 0) newest = d * amp;
  }

  /* ── the heading ──
     Straight off the derivative of the path, so the animal always faces
     where it is actually going. Free, and the alternative — remembering
     last frame's position — is the per-creature buffer this whole file
     exists in order not to have.

     The anchor's own velocity is added here in the same units. dOff was
     converted by kickRate on the way out of the loop because its shape
     is a function of kick-periods, while dTarget is per second, since the
     leg timing is in real seconds. Summing the two without that
     conversion is not a scale error that comes out in the wash: it is two
     vectors in different units being added, so the *direction* is wrong
     wherever both are live — which is precisely during a crossing, which
     is precisely where the heading matters. */
  vec2 vel = dOff.xz + dTarget;
  float speed = length(vel);
  vec2 fwd2 = vel / max(speed, 1e-4);

  vec2 world = target + off.xz;

  /* Height above the ground it is over, not above its home: a butterfly
     that kept a fixed altitude would fly into the hillside on the way up
     one and hang in mid-air on the way down. */
  float rate = 9.0 + h2.z * 5.0;
  float beat = sin(uTime * rate + h2.y * 6.2831);
  /* The rise and fall comes out of the same kick train — every kick has
     a vertical component, so the animal is thrown up and settles back on
     exactly the rhythm it is being thrown sideways on, which is the one
     thing a separate height wave could never get right.

     It climbs to cross on top of that. A dash at working height is a
     dash through the grass; lifting through the middle of one is both
     what the animal does and what makes the crossing legible from any
     angle, since the silhouette comes clear of the sward for exactly as
     long as it is travelling.

     Clear of the blooms even at its lowest, which cost a version to
     learn: aiming the flight at the flowers put the butterflies at
     exactly flower height, among things that are flower-sized and
     flower-coloured, and they stopped reading as butterflies at all.
     What separates an animal from the plant it is visiting, at this
     distance, is that it is above it. */
  float hover = max(0.16, 0.34 + off.y + 0.022 * beat + 0.30 * transit);
  vec3 centre = vec3(world.x, terrainH(world) + hover, world.y);

  vec3 fwd = vec3(fwd2.x, 0.0, fwd2.y);
  vec3 sideDir = vec3(-fwd2.y, 0.0, fwd2.x);

  /* The flap. Fast, and never flat: a wing that reaches horizontal
     catches the light across its whole face for one frame and strobes.
     Held between about twenty and eighty degrees instead, which is also
     the range a real one spends nearly all its time in. */
  float flap = mix(0.35, 1.40, 0.5 + 0.5 * beat);

  /* Rolled into the kick that has just landed, rather than into some
     measure of curvature. With the path made of impulses there is no
     smooth curvature to read — and the kick is the better quantity
     anyway: a butterfly is nearly all wing and almost no mass, so what
     it visibly does is flick over as it throws itself sideways, not lean
     gradually through a corner. Sideways component only, because a kick
     straight down the heading is an acceleration and not a turn. */
  float bank = clamp(dot(sideDir, newest) * 1.60, -0.62, 0.62);
  sideDir = normalize(sideDir + vec3(0.0, 1.0, 0.0) * bank);

  /* The wing, hinged on the body line. Local x runs outward from the
     hinge and local z runs along the body; the flap rotates the outward
     axis up out of the horizontal, which is the whole animation. */
  vec3 hinge = fwd * ((along - 0.35) * WING_L);
  vec3 outward = sideDir * (side * cos(flap)) + vec3(0.0, sin(flap), 0.0);
  vec3 p = centre + hinge + outward * (out_ * WING_W);

  /* The wing's own normal, which is what makes the flap visible at all:
     the two wings face different ways at every instant except the top
     and bottom of the stroke, so one is lit while the other is not. */
  vec3 n = normalize(cross(fwd, outward)) * side;

  vec3 albedo = WINGS[int(h.z * 4.999)];
  // Paler toward the outer edge, darker at the hinge, as almost every
  // wing in nature is — and it stops the quad reading as a flat chip.
  albedo *= mix(0.55, 1.15, out_);

  float sun = sunlight(centre, uShadowSoft);
  // A wing is one membrane thick and glows when the sun is behind it,
  // for the same reason and by the same term as a blade of grass.
  vec3 col = shadeBlade(n, albedo, sun, 0.95, TRANSMIT);

  // Round, from two triangles, by the same trick the flower heads use.
  vRound = vec3(out_ * 2.0 - 1.0, along * 2.0 - 1.0, 1.0);

  emit(p, col * live);
}
`;

/* ── sparrows ─────────────────────────────────────────────────────
   The first thing here that could not be got away with as a billboard.

   A butterfly is two wings because two wings is what a butterfly *is* at
   any distance this scene shows one. A bird is not: what identifies a
   small passerine, before you can see a feather, is its outline — a
   rounded chest a third of the way back, a head that is smaller than the
   chest and joined to it without a neck, and a long tail behind. Get any
   of those three wrong and you have a cartoon, and the specific way it
   goes wrong is always the same one: the head comes out too big.

   ── the proportions are measured, not drawn ──────────────────────
   So the body is not authored as a shape. It is a table of rings taken
   off a real tree sparrow — where along the body, how wide, and how far
   the centreline is lifted — and the tube is built through them. The
   number that matters most is the ratio of the crown's radius to the
   chest's: 0.64 here, about 0.65 on the bird, and anything at or above
   1.0 is a children's drawing. There is no profile function to tune,
   because there is nothing to tune: the vertices sit on the
   measurements.

   Sized against a 9 cm body, which is what this table is in units of:
   head 2.5 cm across, chest 4.0, wingspan 21, tail 5. Those are the
   dimensions of the animal.

   ── what it costs, and why that is allowed ───────────────────────
   Seventy triangles against a butterfly's four. It is affordable for
   exactly one reason: there are far fewer of them. A meadow holds
   hundreds of butterflies and a handful of sparrows, so the ceiling is
   twenty-six, and the whole population is under two thousand triangles —
   less than the grass spends on a square metre. */

/** Rings through the body: distance along it, radius, and how far the
    centreline lifts. All in body-lengths, all measured. */
const BIRD_RINGS = [
  // u,    r,      lift
  [0.00, 0.004,  0.010],   // beak tip
  [0.10, 0.045,  0.020],   // beak base, into the forehead
  [0.20, 0.140,  0.040],   // crown — the widest the head gets
  [0.30, 0.125,  0.028],   // nape. The notch, and the reason there is a head
  [0.46, 0.220,  0.000],   // chest — the widest point, a third back
  [0.72, 0.150, -0.012],   // belly
  [1.00, 0.070,  0.000],   // tail base
];

/** A bird is taller than it is wide. One number, applied to every ring —
    a circular cross-section reads as a fish. */
const BIRD_FLAT = 0.84;

/**
 * Body length in metres, beak tip to tail base.
 *
 * A tree sparrow is 0.09. This is one and a half times that, and it is
 * the one measurement here that is deliberately not the animal's.
 *
 * The reason is the frame, not the bird. At life size and a commuting
 * sparrow's 6 to 9.5 metres a second, the thing moves 10 to 16 cm
 * between frames — *more than its own body length every frame* — and
 * nothing that displaces itself by more than its own length in a frame
 * can be resolved, before the temporal filter has even had its turn at
 * smearing it. Enlarging and slowing together (see the cruise speed
 * below) brings that to about half a body length, which is the number
 * that decides whether there is a bird on screen or a streak.
 *
 * The proportions are untouched: this scales the ring table, it does not
 * reshape it, so the head is still 0.64 of the chest and the animal is
 * still a sparrow rather than a sparrow-shaped pigeon.
 */
const BIRD_LEN = 0.34425;

/**
 * A multiplier on every ring's radius, and the second place this bird
 * departs from its own measurements.
 *
 * It does not touch the proportions the table exists to protect — one
 * number applied to all seven rings leaves the crown at 0.64 of the
 * chest, which is the ratio that separates a sparrow from a cartoon.
 * What it changes is how heavy the animal reads: a bird scaled up by
 * length alone comes out long and thin, because the silhouette that
 * survives at twenty pixels is dominated by the widest part and there
 * was not enough of it.
 */
const BIRD_GIRTH = 1.20;

/**
 * Tail length, as a fraction of the body.
 *
 * A tree sparrow's is about 0.55 and this is 0.33. Shortened on the
 * same grounds the body was enlarged: the tail is the part most likely
 * to be mistaken for something else at range — a long one on a small
 * body reads as a wagtail, or as a smear behind the bird — and the
 * silhouette is stronger for the animal being more compact than life.
 */
const TAIL_LEN = 0.33;

/* What a cat does to a flock on the ground.

   NEAR and FAR are where the panic starts and ends, in metres of the
   cat's approach — two and a half metres of walking, which is under a
   second and reads as immediate without ever being a step function.

   FAN is how far apart they burst at the worst of it and RUN is how far
   they get away by the end. The fan is deliberately larger than the
   formation they close into: a flock that leaves in formation is a
   flight of geese, and what a scattered sparrow flock does is come
   apart and gather again on the wing. */
/* Where the panic starts and where it is complete, in metres of the
   cat's approach.

   Narrow, and further out than it was. A wide band is a slow reaction —
   the flock commits over however long the cat takes to cross it — and
   2.6 to 5.2 metres was two and a half metres of walking before they
   were properly going. One and a half metres, starting sooner, is
   something under a second and reads as the animal noticing rather than
   the animal being caught up with. */
const SCARE_NEAR = 4.0;
const SCARE_FAR = 5.5;
const SCARE_FAN = 2.4;

/** How far a startled flock is shoved along its own schedule, as a
    fraction of one leg. Enough to clear the take-off and be properly
    under way — the point is that the escape becomes an ordinary flight
    rather than a special case with its own rules. */
/* How long the drop onto a perch takes, and the climb off it, in
   seconds. Neither is eased at both ends — see the block in VERT_BIRD —
   and they are deliberately different lengths from each other, because
   that asymmetry is most of what tells the two apart at a glance. */
const DIVE = 4.5;
const CLIMB = 2.5;

/* Where in the drop, and in the climb, the easing starts — as a fraction
   of each. Before it the bird is gaining vertical speed; after it, it is
   giving it back, and it arrives with no vertical rate at all.

   Both ends need one. Without the first, the bird met the branch at its
   fastest, which is a bird hitting a branch; without the second, it
   reached cruise at its fastest and was level in the next frame, which
   is a corner in the path. */
const DIVE_BRAKE = 0.80;
const CLIMB_BRAKE = 0.80;

/* The ground speed, and how long a stop lasts. These two now *define* a
   pass; its length is whatever they add up to.

   That is the right way round and it was the wrong way round before. A
   pass length was chosen and the speeds fell out of it, so they came out
   different at each end and changed whenever the line length did — which
   it does, because the line is tied to the view. Naming the speed makes
   it the same everywhere and makes the reach control stop affecting how
   fast a bird flies. */
const SPEED_MIN = 3.6;
const SPEED_MAX = 6.0;
const DWELL_MIN = 12;
const DWELL_MAX = 26;

/** Sides around the body tube. Five is enough at a range where the whole
    animal is twenty pixels, and the silhouette is doing the work. */
/** How long a flock is, in metres. Strung out along its heading; the
    across-track spread is a third of this, in the shader. */
const FLOCK_SPREAD = 7.0;

/* The world grid the flocks live on, and how many of its cells hold one.

   World-anchored, and that is the whole point: a flock placed relative
   to the camera moves when the camera turns, and with only a handful of
   them a single one blinking out is the most visible thing in the frame.
   Cells are the same idea the grass, the clumps and the tree chunks all
   use, and they are large because a flock is.

   Nine cells are considered — the three by three around the eye — and
   two in five hold a flock, so what is in the sky is three to four
   groups most of the time and never more than nine. */
const FLOCK_CHANCE = 0.40;

/* How far apart flocks live, in metres of world.

   A fixed distance, and that is the whole point. It used to be a
   fraction of the reach, which sounds reasonable and quietly made the
   count a *constant*: the cell grew with the view, the two cancelled in
   πr²/cell², and the sky held the same six groups whether you could see
   twenty-five metres or sixty. The control moved the horizon and nothing
   arrived at it.

   Fixed, the count goes with the area you can see — which is what the
   eye expects of a bigger view, and what makes the ceiling above mean
   something rather than being unreachable. Roughly ten groups at the
   default reach, hitting the ceiling of twenty-four somewhere around
   seventy.

   The one thing the reach still does is stop the grid being coarser than
   the view. A cell wider than what you can see is a cell that can hold
   nothing you would ever reach, and the sky at the short end of the
   control comes out not thin but empty — so below about thirty-five the
   grid tightens to keep something in it. That is a floor, not the
   behaviour. */
const FLOCK_CELL = 24.0;
function flockCell(box) {
  return Math.min(FLOCK_CELL, box * 0.95 * 0.45);
}

const BIRD_SIDES = 5;
const BIRD_SEGS = BIRD_RINGS.length - 1;

/* Vertices, by part. The body is a tube of quads; each wing is two
   panels so it can break at the wrist, which is most of what separates a
   bird from a paper dart; the tail is one fan. */
const BODY_VERTS = BIRD_SEGS * BIRD_SIDES * 6;
const WING_VERTS = 2 * 2 * 6;
const TAIL_VERTS = 6;
const BIRD_VERTS = BODY_VERTS + WING_VERTS + TAIL_VERTS;

const glslRings = BIRD_RINGS
  .map(([u, r, y]) => `vec3(${u.toFixed(4)}, ${r.toFixed(4)}, ${y.toFixed(4)})`)
  .join(',\n  ');

/* No clump grid and no scatter here. Both were included while the birds
   derived their own homes and perches in the shader; the pass model moved
   all of that to the CPU and left the two chunks dead, still being
   compiled into a programme that never called them. */
const VERT_BIRD = FLY_HEAD + /* glsl */`
#define BIRD_SIDES ${BIRD_SIDES}
#define BIRD_SEGS ${BIRD_SEGS}
#define BODY_VERTS ${BODY_VERTS}
#define WING_VERTS ${WING_VERTS}
#define BIRD_LEN ${BIRD_LEN.toFixed(4)}
#define BIRD_FLAT ${BIRD_FLAT.toFixed(3)}
#define BIRD_GIRTH ${BIRD_GIRTH.toFixed(3)}
#define TAIL_LEN ${TAIL_LEN.toFixed(3)}
#define SCARE_NEAR ${SCARE_NEAR.toFixed(2)}
#define SCARE_FAR ${SCARE_FAR.toFixed(2)}
#define SCARE_FAN ${SCARE_FAN.toFixed(2)}
#define DIVE ${DIVE.toFixed(2)}
#define CLIMB ${CLIMB.toFixed(2)}
#define DIVE_BRAKE ${DIVE_BRAKE.toFixed(3)}
#define CLIMB_BRAKE ${CLIMB_BRAKE.toFixed(3)}
#define PERCH_N ${PERCH_N}
#define FLOCK_SPREAD ${FLOCK_SPREAD.toFixed(1)}
#define FLOCK_CHANCE ${FLOCK_CHANCE.toFixed(3)}
#define FLOCK_WINDOW ${FLOCK_WINDOW}

/** The measured rings: x = along the body, y = radius, z = lift. */
const vec3 RINGS[${BIRD_RINGS.length}] = vec3[${BIRD_RINGS.length}](
  ${glslRings}
);

/* Plumage. A tree sparrow is a brown bird and reads as one from the
   colour of its back; the chestnut cap and the black cheek spot are
   below the resolution of anything this scene draws, so they are not
   here. What is here is the one contrast that survives at twenty pixels:
   a dark back over a pale underside, which is countershading and is why
   a bird against the sky and a bird against the ground are both
   legible. */
/** How many birds are in a flock this frame — the control, and the only
    thing about the population that is one. */
uniform int uPerFlock;

/* ── a flock is a pass, not an orbit ───────────────────────────────
   The whole of a flock's behaviour is one straight line across the
   meadow, flown once, with at most one stop on the way.

   Everything before this was a shuttle: two perches and a flock going
   back and forth between them for ever. It was wrong in a way that no
   amount of tuning reached, because the shape of the motion was a closed
   loop and a closed loop is the one thing that reads as machinery. Birds
   do not commute in circles across an empty field; they cross it, and
   sometimes they stop, and then they are gone.

   So a pass is: come in on a heading, either drop onto something or do
   not, and carry on out the far side. Half the passes are aimed at a
   tree, because a tree is where a flock is going; the other half either
   put down on open ground or never stop at all.

   What this buys, beyond looking right, is that there is nothing left to
   phase against anything else. The descent is the approach — the same
   smoothstep drives both — so there is exactly one acceleration into the
   stop and one out of it, and no way for the height and the distance to
   disagree about when the bird arrived. The three-curve balancing act
   the escape needed is gone with the loop that made it necessary. */
#define MAX_FLOCKS ${MAX_FLOCKS}

/** xy = where the line starts, z = its heading in radians, w = how wide
    the flock gathers at its stop — positive in a crown, negative on open
    ground, so one number carries both the size and the kind. The heading
    is an angle rather than a vector so that the fourth slot is free: two
    trigonometric calls per vertex against a whole extra uniform array. */
uniform vec4 uFlockLine[MAX_FLOCKS];
/** xyz = the stop, w = how far along the line it is. Negative w is a
    pass that never stops — it crosses and goes. */
uniform vec4 uFlockStop[MAX_FLOCKS];
/** x = birds in it, y = seconds per pass, z = phase, w = fade. */
uniform vec4 uFlockInfo[MAX_FLOCKS];
uniform float uFlockN;
/** x = when the stay begins, y = when it ends, in seconds into the pass.
    Both derived on the CPU from one ground speed, and the second of them
    is where the cat's whole effect arrives — resolved and latched up
    there, so nothing in here knows a cat exists. */
uniform vec2 uFlockTimes[MAX_FLOCKS];
/** How long a line is, in metres. One number for all of them. */
uniform float uFlockRun;

const vec3 BIRD_BACK = vec3(0.105, 0.072, 0.040);
const vec3 BIRD_BELLY = vec3(0.230, 0.205, 0.170);
const vec3 BIRD_BEAK = vec3(0.055, 0.048, 0.040);

void main() {
  vColor = vec3(0.0);
  vRound = vec3(0.0);
  vDist = 0.0;

  int id = gl_VertexID / ${BIRD_VERTS};
  int v = gl_VertexID % ${BIRD_VERTS};

  /* ── a few flocks, not many birds ──
     What reads as a flock is a small number of dense groups, and the
     first version had it backwards: many small ones, spread thin, each
     bird beating to its own clock. That is not an understaffed flock,
     it is a swarm of flies — and the cause was not the count. Birds read
     as birds because *the whole group rises and falls together*, and
     every member here was undulating on a rate and a phase hashed from
     its own index, so seven of them bobbed independently.

     So the rate, the heading and the schedule all moved up to the flock,
     and only a small phase offset stayed with the bird. The control now
     sets how many are *in* a flock; how many flocks there are is three
     to six and is not a control, because past half a dozen groups the
     eye stops counting groups and goes back to seeing a scatter. */
  int flock = id / uPerFlock;
  int member = id - flock * uPerFlock;

  if (float(flock) >= uFlockN) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  vec4 info = uFlockInfo[flock];
  if (float(member) >= info.x) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }

  vec4 line = uFlockLine[flock];
  vec4 stop = uFlockStop[flock];
  vec2 dir = vec2(cos(line.z), sin(line.z));
  vec2 across = vec2(-dir.y, dir.x);
  float live = info.w;

  /* The individual, hashed from where its flock's line begins and which
     member it is — never from the vertex id. The list is ranked by
     distance, so two flocks trade slots when the eye moves past the
     point their ranks cross; everything else about a flock rides in its
     record and comes through that unharmed, and this has to as well or a
     swap would deal the individuals into the other group. */
  vec3 h2 = hash33(vec3(line.x * 0.317 + float(member) * 1.73,
                        line.y * 0.291 - float(member) * 2.11, 4.2));
  vec2 across2 = across;

  float T = info.y;
  float tt = uTime + info.z;
  float pt = (tt / T - floor(tt / T)) * T;      // seconds into this pass

  bool stops = stop.w >= 0.0;

  /* One ground speed for the whole pass, and the times derived from it
     rather than the other way round.

     It used to be the reverse — a pass length was chosen and the speeds
     fell out of it — and the two halves came out different, five point
     seven metres a second going in and four point nine coming out, with
     a step across the stop. Naming the speed and solving for the times
     makes every part of every pass run at the same pace, and the only
     thing a stop now costs is the stop. */
  vec2 times = uFlockTimes[flock];
  float tArrive = times.x;
  float speed = stops ? stop.w / max(tArrive, 1e-3) : uFlockRun / T;
  /* When the stay ends. Normally 0.62 of the pass; earlier if a cat
     walked up to this flock while it was down, which is decided on the
     CPU and latched there for the rest of the pass. */
  float tLeave = times.y;

  /* The cat used to be read here, and that was the third time in this
     file that a world quantity was made a function of something that
     moves. It shifted the clock by the cat's distance — so the flock
     covered thirty metres while the cat covered one and a half, which is
     a bird being fired out of a cannon, and then slid all the way back
     when the cat wandered off again. Position was a function of the cat,
     so it ran at the cat's speed and it was reversible.

     A flush is an event: it happens once, at a moment, and nothing
     afterwards undoes it. That is state, and state is exactly what a
     vertex shader cannot have — but the CPU can, and it is the CPU that
     builds these records. So the scare is latched up there and arrives
     here as an ordinary end-of-stay. Nothing in this shader knows a cat
     exists, and a flock already on the wing is immune for free, because
     the only thing the cat can change is when a *stay* ends.

     A scared flock therefore leaves at exactly the speed a calm one
     does. It just leaves sooner. */
  /* Where it is along the line, and how far down it has come.

     These were one smoothstep, on the reasoning that the descent *is*
     the approach and so cannot arrive early or late. It was true and it
     was unwatchable: spread over the eight to twelve seconds of the run
     in, the drop from cruise height is a slope of about one in ten, and
     at the range this scene is viewed from that is a straight line. The
     most characteristic thing the animal does was there and could not be
     seen.

     So the height gets its own window, and inside it the vertical motion
     is *accelerated in one direction only* — not eased at both ends and
     not at a constant rate.

     A drop starts from level flight and gains speed all the way down, so
     it is fastest at the moment of arrival, which is the instant the
     braking pose exists for. A climb is the reverse: everything goes in
     at the bottom and bleeds off on the way up, so it leaves the ground
     hard and arrives at cruise with nothing left. The smoothstep that
     was here first was symmetric, which made both events look the same
     and spent most of each window nearly level; a constant rate was the
     correction to that and was also wrong, because it has the bird
     leaving the branch and touching down at the same speed.

     Down and up are not the same length. A drop is a drop — a second and
     a half, three or four metres a second — and a climb out is work, so
     it takes five. That asymmetry is most of what separates the two
     events from each other at a glance.

     The ground track is linear too, and for a separate reason. It was a
     smoothstep, so the flock accelerated out of a stop over the whole
     departure — which meant the climb, five seconds of it, was flown at
     a horizontal speed still ramping up from nothing, and lengthening
     the climb quietly slowed the whole exit down. Height and ground
     speed are different questions: DIVE and CLIMB say *when the animal
     changes height*, and the ground track just runs at its own steady
     pace underneath. Changing one no longer changes the other.

     A pass with no stop is a straight run at a steady pace, because
     there is nothing to slow down for. */
  float along;
  float down = 0.0;
  float leaving = 1.0;

  /* Nose-down into the dive and nose-up out of the climb. The vertical
     rate, as the derivative of the smoothstep that is driving it. */
  float vRate = 0.0;

  if (stops) {
    if (pt < tArrive) {
      along = speed * pt;
      /* The drop is a fall with a stop on the end of it: four fifths of
         the window gaining speed, then a hard cushion in the last fifth
         that brings the vertical rate back to nothing exactly as the
         feet arrive.

         The two halves are joined so that the *speed* matches across the
         seam, not just the height — the coefficient is 1.25 for no
         better reason than that it is what makes the junction smooth,
         and it puts the animal four fifths of the way down when the
         braking starts. Without the cushion the bird met the branch at
         eight metres a second, which is a bird hitting a branch. */
      float u = clamp((pt - (tArrive - DIVE)) / DIVE, 0.0, 1.0);
      if (u < DIVE_BRAKE) {
        down = 1.25 * u * u;
        vRate = -2.5 * u;
      } else {
        float w = (u - DIVE_BRAKE) / (1.0 - DIVE_BRAKE);
        down = 0.8 + 0.2 * (1.0 - (1.0 - w) * (1.0 - w));
        vRate = -2.0 * (1.0 - w);
      }
      leaving = 0.0;
    } else if (pt < tLeave) {
      along = stop.w;
      down = 1.0;
      leaving = 0.0;
    } else {
      along = stop.w + speed * (pt - tLeave);
      float u = clamp((along - stop.w) / max(uFlockRun - stop.w, 1e-3), 0.0, 1.0);
      /* The climb is the drop read backwards, cushion and all: pushing
         off slowly, gaining upward speed for four fifths of it, then
         easing off through the last fifth so that it arrives at cruise
         already level. The same curve at both ends, once each way.

         Levelling out matters as much here as the braking does there. An
         accelerating climb that ran all the way to cruise would arrive
         at its fastest and then be flat in the next frame, which is a
         corner in the path and reads as the animal hitting a ceiling. */
      float v = clamp((pt - tLeave) / CLIMB, 0.0, 1.0);
      float lift;
      if (v < CLIMB_BRAKE) {
        lift = 1.25 * v * v;
        vRate = 2.5 * v;
      } else {
        float w = (v - CLIMB_BRAKE) / (1.0 - CLIMB_BRAKE);
        lift = 0.8 + 0.2 * (1.0 - (1.0 - w) * (1.0 - w));
        vRate = 2.0 * (1.0 - w);
      }
      down = 1.0 - lift;
      leaving = u;
    }
  } else {
    along = speed * pt;
  }

  float land = down;
  /* The braking pose, aligned with the dive rather than with a fraction
     of the approach: it is the dive it is braking out of. */
  float flare = stops
    ? smoothstep(tArrive - DIVE * (1.0 - DIVE_BRAKE) * 1.8, tArrive, pt)
      * (1.0 - smoothstep(tArrive, tArrive + 0.9, pt))
    : 0.0;

  float boundRate = 1.25 + fract(info.z * 0.37) * 0.55;
  float bt = uTime * boundRate + info.z + (h2.z - 0.5) * 1.2;
  float climb = sin(bt);
  float flapping = smoothstep(-0.15, 0.35, climb);

  /* The flock, strung out along its heading and narrow across it — and
     opened up for a moment as it leaves a stop, which is the one thing a
     straight line cannot say. Keyed to the departure, so it flowers at
     the take-off and closes on the wing. */
  float burst = 6.75 * leaving * (1.0 - leaving) * (1.0 - leaving) * 0.9;
  vec2 flying = dir * ((h2.x - 0.5) * FLOCK_SPREAD)
              + across * ((h2.y - 0.5) * FLOCK_SPREAD * 0.34)
              + across * ((h2.y - 0.5) * burst * SCARE_FAN * 2.0);

  /* The formation has to *close* as they land, and it did not.
     A flock in the air is seven metres long, which is what makes it read
     as going somewhere; carried into the stop unchanged it scattered the
     birds over seven metres of ground around the tree, so what you saw
     was a group pausing near a crown rather than sitting in it. Gathered
     into the crown's own width instead — a metre or two — and the same
     factor that brings them down brings them together. */
  float gather = abs(line.w);
  vec2 perched = (h2.xy - 0.5) * gather * 1.7;
  vec2 world = line.xy + dir * along + mix(flying, perched, down);

  /* No fade along the line. There used to be one at each end, and it was
     the bug: it made *where a flock was in its pass* a reason to be
     invisible, on top of the real one, which is how far away it is. With
     the line now longer than the view, the ends are out in the fog
     already and the distance fade — computed on the CPU from where the
     birds actually are — is the only thing deciding this. */
  if (live <= 0.004) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }

  float cruise = 5.0 + h2.z * 4.0;
  /* Spread through the crown's depth as well as its width, so a perched
     flock occupies the tree rather than forming a layer across it. */
  float perchY = stop.y + (line.w > 0.0 ? (h2.z - 0.45) * gather * 0.7 : 0.0);
  float y = mix(terrainH(world) + cruise + climb * 0.22, perchY, down);
  vec3 centre = vec3(world.x, y, world.y);


  /* Perched birds hold still and hold their wings shut — except while
     they are braking, which is the one moment the wings are working
     hardest and the body is barely moving. */
  flapping = max(flapping * (1.0 - land), flare);
  climb = mix(climb * (1.0 - land), 0.55, flare);

  /* The body frame. Pitched with the climb, because a bird that stayed
     level while rising and falling is a bird on a wire — and levelled
     again as it settles, because a perched one sits upright. */
  /* Pitched by the bounding flight and by the dive together. Without
     the second term the animal drops three metres a second while
     staying dead level, which reads as a lift rather than a bird. */
  vec3 fwd = normalize(vec3(dir.x, climb * 0.30 + vRate * 0.55, dir.y));
  vec3 side = normalize(cross(fwd, vec3(0.0, 1.0, 0.0)));
  vec3 up = cross(side, fwd);

  vec3 p = vec3(0.0);
  vec3 n = vec3(0.0, 1.0, 0.0);
  vec3 albedo = BIRD_BACK;

  if (v < BODY_VERTS) {
    /* The tube through the measured rings. Two rings and two sides make
       a quad; the ring table supplies both ends of it, so nothing here
       decides what shape a sparrow is. */
    int seg = v / (BIRD_SIDES * 6);
    int rest = v - seg * (BIRD_SIDES * 6);
    int sideI = rest / 6;
    int corner = rest - sideI * 6;

    float alongEnd = (corner == 1 || corner == 3 || corner == 4) ? 1.0 : 0.0;
    float aroundEnd = (corner == 2 || corner == 4 || corner == 5) ? 1.0 : 0.0;

    vec3 ra = RINGS[seg];
    vec3 rb = RINGS[seg + 1];
    vec3 ring = mix(ra, rb, alongEnd);

    float ang = (float(sideI) + aroundEnd) / float(BIRD_SIDES) * 6.2831;
    vec2 rim = vec2(cos(ang) * BIRD_FLAT, sin(ang));

    // Nose at the front, tail at the back, and the lift on the y axis.
    vec3 local = fwd * ((0.5 - ring.x) * BIRD_LEN)
               + up * (ring.z * BIRD_LEN + rim.y * ring.y * BIRD_LEN * BIRD_GIRTH)
               + side * (rim.x * ring.y * BIRD_LEN * BIRD_GIRTH);
    p = centre + local;

    /* The normal from the rim, tilted by how fast the radius is changing
       along the body — without it the beak is lit as though it were a
       cylinder and reads as a peg. */
    // Scaled with the girth, or a fatter bird would be lit as a thinner one.
    float slope = (rb.y - ra.y) * BIRD_GIRTH / max(rb.x - ra.x, 1e-3);
    vec3 radial = normalize(up * rim.y + side * rim.x);
    n = normalize(radial + fwd * slope);

    // Countershaded, and a dark beak in front of it.
    albedo = mix(BIRD_BACK, BIRD_BELLY, clamp(0.5 - rim.y * 0.75, 0.0, 1.0));
    albedo = mix(BIRD_BEAK, albedo, smoothstep(0.06, 0.14, ring.x));
  } else if (v < BODY_VERTS + WING_VERTS) {
    int w = v - BODY_VERTS;
    float wingSide = w < 12 ? -1.0 : 1.0;
    int panel = (w % 12) / 6;                       // 0 inner, 1 outer
    int corner = w % 6;
    float outEnd = (corner == 1 || corner == 3 || corner == 4) ? 1.0 : 0.0;
    float backEnd = (corner == 2 || corner == 4 || corner == 5) ? 1.0 : 0.0;

    /* Two panels hinged at the wrist. The inner one carries the beat and
       the outer one trails it — the outer sweeps back and folds later,
       which is the shape a bird's wing makes and a single flat quad
       cannot. */
    /* Faster and deeper while braking. A bird flaring beats at something
       like twice its cruising rate through a much bigger arc, and the
       wings come *forward* — the sweep goes negative — because what it is
       doing is presenting them to the air it is still moving through. */
    float beat = sin(uTime * (14.0 + 13.0 * flare) + h2.y * 6.2831) * flapping;
    float dihedral = beat * (0.95 + 0.55 * flare) - (1.0 - flapping) * 1.25;
    float wristFold = beat * 0.55 - (1.0 - flapping) * 1.45 + flare * 0.55;
    float sweep = 0.35 + (1.0 - flapping) * 0.70 - flare * 1.05;

    float shoulderU = 0.40;
    vec3 root = centre
      + fwd * ((0.5 - shoulderU) * BIRD_LEN)
      + up * (0.05 * BIRD_LEN)
      + side * (wingSide * 0.16 * BIRD_LEN);

    // Inner span, then the outer hinged off its tip.
    vec3 inDir = normalize(side * (wingSide * cos(dihedral))
                         + up * sin(dihedral)
                         - fwd * sweep * 0.35);
    vec3 wrist = root + inDir * (0.45 * BIRD_LEN);
    vec3 outDir = normalize(inDir * cos(wristFold)
                          - fwd * sin(wristFold) * 1.0
                          - fwd * sweep * 0.55);

    vec3 a = panel == 0 ? root : wrist;
    vec3 d = panel == 0 ? inDir : outDir;
    float span = panel == 0 ? 0.45 : 0.55;
    float chordFront = panel == 0 ? 0.30 : 0.22;
    float chordBack = panel == 0 ? 0.26 : 0.06;

    vec3 lead = a + d * (span * BIRD_LEN * outEnd);
    float chord = mix(chordFront, chordBack, outEnd) * BIRD_LEN;
    p = lead - fwd * (chord * backEnd);

    n = normalize(cross(d, fwd)) * wingSide;
    // Flight feathers are darker than the coverts nearer the body.
    albedo = mix(BIRD_BACK * 1.15, BIRD_BACK * 0.55, outEnd);
  } else {
    int corner = v - BODY_VERTS - WING_VERTS;
    float backEnd = (corner == 1 || corner == 3 || corner == 4) ? 1.0 : 0.0;
    float acrossEnd = (corner == 2 || corner == 4 || corner == 5) ? 1.0 : 0.0;

    /* One flat fan, spread while gliding and closed while beating —
       which is what a tail is for, and it is visible from behind at
       ranges where nothing else about the bird is. */
    /* Fanned wide and dropped while braking: the tail is the airbrake,
       and it is the part of the pose visible from behind. */
    float spread = mix(0.13, 0.30, 1.0 - flapping) + flare * 0.28;
    vec3 base = centre - fwd * (0.5 * BIRD_LEN);
    p = base
      - fwd * (TAIL_LEN * BIRD_LEN * backEnd)
      + side * ((acrossEnd * 2.0 - 1.0) * spread * BIRD_LEN * mix(0.35, 1.0, backEnd))
      - up * ((0.04 + flare * 0.42) * BIRD_LEN * backEnd);

    n = up;
    albedo = BIRD_BACK * mix(1.0, 0.7, backEnd);
  }

  float sun = sunlight(centre, uShadowSoft);
  emit(p, shadeBlade(n, albedo, sun, 0.95, 0.0) * live);
}
`;

/* ── fireflies ────────────────────────────────────────────────────── */

const VERT_GLOW = /* glsl */`
${PRECISION}
${HASH}
${TERRAIN_GLSL}

uniform vec3 uCamPos, uRight, uUp, uFwd;
uniform float uFocal, uAspect, uTime;
/** How far the glow is drawn, in metres. */
uniform float uBox;

out vec2 vQuad;
out float vDepth;
out float vLive;

/* Deliberately not sharing the scatter above. That one needs snowCover,
   which needs the weather block, which needs the whole plant header —
   and this pass is a point sprite drawn after the tonemap that has no
   business knowing how a leaf is lit. It asks the two questions it can
   answer with the terrain block alone, and the third — snow — it does
   not ask, because a firefly is not standing on the ground. */

void main() {
  int id = gl_VertexID / 6;
  int corner = gl_VertexID % 6;
  float u = (corner == 1 || corner == 3 || corner == 4) ? 1.0 : 0.0;
  float v = (corner == 2 || corner == 4 || corner == 5) ? 1.0 : 0.0;

  vec3 h = hash33(vec3(float(id) * 0.4211 + 3.3, 13.1, 2.7));
  vec3 h2 = hash33(vec3(float(id) * 0.1913 + 27.7, 8.9, 5.1));

  vec2 c = uCamPos.xz;
  vec2 p0 = h.xy * uBox;
  vec2 home = c - uBox * 0.5 + mod(p0 - c + uBox * 0.5, vec2(uBox));

  /* A slow drift, an order of magnitude lazier than the butterflies'.
     What separates the two at a glance, before either is close enough to
     have a shape, is entirely how fast they move. */
  float t = uTime * (0.16 + h2.x * 0.16) + h2.y * 6.2831;
  vec2 world = home + vec2(cos(t), sin(t * 0.81 + 1.3)) * (0.7 + h2.z * 1.1);

  /* Down among the stems, not up in the air. A firefly at head height is
     a star; a firefly at knee height is an insect, because the grass in
     front of it keeps cutting it off. */
  float hover = 0.10 + 0.55 * (0.5 + 0.5 * sin(t * 0.9 + h2.z * 5.0));
  vec3 pos = vec3(world.x, terrainH(world) + hover, world.y);

  /* The blink. A sharp power of a sine, so it is dark most of the time
     and briefly very bright — the duty cycle is the signal, and a
     firefly that merely pulsed would read as a bad LED.

     The floor used to be a tenth, because this scene had no night to
     hide the meadow in and a population that spent nine tenths of its
     life fully dark read as a handful of insects rather than a field of
     them — so the dark ones were kept present as embers. There is a
     night now, and the population is only out in it, so the crutch can
     go: what a firefly is against a dark field is the flash, and an
     ember that never goes out is a tail light. What is left of the floor
     is the little that keeps the field's *extent* legible between
     flashes. */
  float pulse = pow(max(0.0, sin(t * (2.3 + h2.x * 1.7) + h.z * 6.2831)), 6.0);
  float lit = 0.03 + 0.97 * pulse;

  vec2 e = abs(home - c) / (uBox * 0.5);
  float edge = 1.0 - smoothstep(0.62, 1.0, max(e.x, e.y));
  // Nothing over the water, on the same test everything else uses.
  float dry = waterDepth(home) < -0.05 ? 1.0 : 0.0;

  vec3 rel = pos - uCamPos;
  float dist = length(rel);
  vLive = lit * edge * dry * smoothstep(0.25, 0.9, dist);
  vDepth = dist;

  /* Held at a constant *angular* size below a couple of metres, so the
     nearest ones do not become dinner plates, and at a constant world
     size beyond it so the far ones do not vanish. A glow is a glow at
     any range; what changes is how much of one there is. */
  float r = 0.030 * clamp(dist / 2.0, 0.45, 3.0);

  vec2 q = vec2(u, v) * 2.0 - 1.0;
  vec3 wp = pos + uRight * (q.x * r) + uUp * (q.y * r);

  vec3 rel2 = wp - uCamPos;
  vec3 view = vec3(dot(rel2, uRight), dot(rel2, uUp), dot(rel2, uFwd));

  vQuad = q;
  gl_Position = vec4(view.x * uFocal / uAspect, view.y * uFocal, 0.0, view.z);
}
`;

const FRAG_GLOW = /* glsl */`
${PRECISION}

in vec2 vQuad;
in float vDepth;
in float vLive;
out vec4 outColor;

uniform sampler2D uScene;
uniform sampler2D uMesh;
uniform float uMeshOn;
uniform vec2 uResolution;
uniform float uIntensity;

void main() {
  if (vLive <= 0.003) discard;

  /* Hidden behind whichever half of the scene is nearer — the marched
     hills or the rasterised grass. A firefly tested against only one of
     them shines straight through the other, and since these live down
     among the stems, the one it would shine through is the grass. */
  vec2 uv = gl_FragCoord.xy / uResolution;
  float sceneT = texture(uScene, uv).a;
  if (uMeshOn > 0.5) sceneT = min(sceneT, texture(uMesh, uv).a);
  if (vDepth > sceneT) discard;

  /* A hot core inside a wide soft halo, which is what a small bright
     thing looks like through any real lens and what the eye expects
     from one. Two powers of the same falloff; no texture. */
  float d = length(vQuad);
  float halo = pow(max(0.0, 1.0 - d), 2.2);
  float core = pow(max(0.0, 1.0 - d), 9.0);

  vec3 col = vec3(0.72, 1.00, 0.42) * halo + vec3(1.00, 1.00, 0.86) * core * 1.6;
  outColor = vec4(col * vLive * uIntensity, 1.0);
}
`;

/**
 * The meadow's population.
 *
 * Two programs, two draws, one class — because they share their whole
 * reason for existing (a density control, a patch that follows the eye,
 * and the rule about the water) and share none of their pipeline.
 */
export class Creatures {
  constructor(gl) {
    this.gl = gl;
    this.fly = new Program(gl, VERT_FLY, FRAG_PLANT, { name: 'march/butterfly' });
    this.bird = new Program(gl, VERT_BIRD, FRAG_PLANT, { name: 'march/sparrow' });
    this.glow = new Program(gl, VERT_GLOW, FRAG_GLOW, { name: 'march/firefly' });
    /** Drawn last frame, for the readout. */
    this.butterflies = 0;
    this.sparrows = 0;
    this.fireflies = 0;
    /** The flock list, rebuilt each frame in world coordinates. */
    this._flockLine = new Float32Array(MAX_FLOCKS * 4);
    this._flockStop = new Float32Array(MAX_FLOCKS * 4);
    this._flockInfo = new Float32Array(MAX_FLOCKS * 4);
    this._flockTimes = new Float32Array(MAX_FLOCKS * 2);
    /** Which flocks have been flushed, and on which pass. The one piece
        of state in this file, and it is here rather than in the shader
        because a shader cannot have any. */
    this._flushed = new Map();
    this._grad = [0, 0];
  }

  /**
   * Butterflies, into the raster target, alongside the grass — same
   * projection, same depth buffer, same jitter, same light.
   *
   * @returns {number} vertices to draw, or 0
   */
  /** Triangles drawn last frame, across every kind that has any. */
  get triangles() {
    return this.butterflies * 4 + this.sparrows * (BIRD_VERTS / 3);
  }

  drawFlies(camera, env, opts) {
    const box = reachFor('fly', opts.reach);
    /* The hour thins the population rather than dimming it, which is the
       same thing the density and the reach controls already do and works
       for the same reason: every butterfly is a function of its own
       index, so dropping the count takes the last ones off the end and
       leaves the rest exactly where they were. Fading them out instead
       would put a field of half-transparent insects over the meadow at
       dusk, which is not what dusk does to a butterfly. */
    const n = Math.min(FLY_MAX, Math.round(
      Math.max(0, Math.min(1, opts.density)) * dayPart(env.day)
      * FLY_PER_M2 * box * box));
    this.butterflies = n;
    if (!n) return 0;

    this.fly.use({
      ...plantUniforms(camera, env, { wind: opts.wind, jitter: opts.jitter }),
      /* The flower controls, exactly as the sowing was given them this
         frame. Zero when the flowers are off, which is what turns the
         flight back into a wander rather than aiming it at clumps that
         are not there. */
      uClumpChance: opts.flowers ? opts.clumpChance : 0,
      uClumpSpread: opts.clumpSpread,
      uBox: box,
    });
    return n * 12;
  }

  /**
   * Sparrows, into the raster target beside the butterflies.
   *
   * Same programme shape and the same shared light, but a bird is not a
   * scaled-up insect: it is ten times the size, flies three metres up,
   * commutes in a straight line instead of milling about a flower, and
   * is drawn out of a table of measurements rather than out of a rule of
   * thumb. What the two genuinely share is the placement and the shading,
   * which is why those live in one function each and not in this one.
   *
   * @returns {number} vertices to draw, or 0
   */
  drawBirds(camera, env, opts) {
    /* Two controls, the same pair the flowers have and for the same
       reason: "a few big flocks" and "many small ones" are two different
       skies, and one slider can reach both ends without ever saying
       which of them it is doing. */
    const perFlock = Math.round(Math.max(0, Math.min(1, opts.density)) * PER_FLOCK);
    const flockPart = Math.max(0, Math.min(1, opts.flocks ?? 1));
    if (!perFlock || !flockPart) { this.sparrows = 0; return 0; }

    /* Which flock slots are live, and how big each one is, decided here
       by the same rule the shader uses so that the readout is the truth
       and not an upper bound. Every slot is drawn either way — the dead
       ones are culled in the first four lines of the vertex shader, which
       is cheaper than telling the CPU about them — but what gets counted
       is what survives. */
    /* ── one pass per flock, decided here ──
       A pass is a straight line across the flock's territory with at
       most one stop on it. Deciding it on the CPU keeps every world
       quantity out of the shader, which is what ended three successive
       bugs of the same shape: a flock's home, its route and its perch
       were each, at one time or another, derived from a set the camera
       defined, and each of them made flocks jump or vanish when the view
       merely turned. Nothing in the vertex shader knows where the camera
       is any more. */
    const box = reachFor('bird', opts.reach);
    const cell = flockCell(box);
    const far = box * 0.95;
    /* A line has to be longer than the view, and it was not: at seventy
       metres against a sixty-eight metre reach, its ends fell *inside*
       the visible area, so a flock reached the end of its pass and
       evaporated in mid-air fifty-eight metres away — plainly visible —
       then reappeared at the other end. Measured, that was the whole of
       the vanishing report.

       Tied to the reach instead, both ends are well outside anything the
       fog is showing, and the only thing that decides whether a bird can
       be seen is how far away it is. */
    const run = far * 2.4;                       // how long a line is
    const ex = camera.pos[0], ez = camera.pos[2];

    const span = Math.ceil(far / cell) + 1;
    const cx = Math.floor(ex / cell), cz = Math.floor(ez / cell);
    const near = [];

    for (let dz = -span; dz <= span; dz++) {
      for (let dx = -span; dx <= span; dx++) {
        const ix = cx + dx, iz = cz + dz;
        if (seed(ix, iz, 41) > FLOCK_CHANCE) continue;

        const hx = (ix + seed(ix, iz, 42)) * cell;
        const hz = (iz + seed(ix, iz, 43)) * cell;
        const d = Math.hypot(hx - ex, hz - ez);
        if (d >= far) continue;
        if (waterDepthAt(hx, hz, opts.hills, opts.waterY) > -0.05) continue;

        const u = Math.min(1, Math.max(0, (d - far * 0.62) / (far * 0.38)));
        const fade = 1 - u * u * (3 - 2 * u);
        if (fade <= 0.01) continue;

        near.push({ ix, iz, hx, hz, d, fade });
      }
    }
    /* Thinned to a *share* of what the ground actually offers, not to a
       fixed number. A ceiling alone stopped biting the moment it was
       raised above what a view of this size supplies — at the default
       reach the world hands over about six flocks, so any setting of the
       control above a quarter was cutting nothing and the slider did
       nothing for most of its travel. A proportion always bites, and at
       the top of the control you still get everything up to the
       ceiling. */
    near.sort((a, b) => a.d - b.d);
    near.length = Math.min(near.length, MAX_FLOCKS);
    /* The floor of one is inside the min, not outside it. Written the
       other way round it lengthens an *empty* list to one — which in JS
       is a hole, not an element, and the loop below then reads the
       properties of undefined. Nothing in range is a real state: it is
       what a lake or a snowfield leaves behind. */
    near.length = Math.min(near.length,
                           Math.max(1, Math.round(near.length * flockPart)));

    const roost = roostPart(env.day);
    const L = this._flockLine, S = this._flockStop, I = this._flockInfo;
    const TM = this._flockTimes;
    let alive = 0;

    /* Forget flushes for flocks that are no longer anywhere near, or the
       map grows for as long as the scene runs. */
    if (this._flushed.size > MAX_FLOCKS * 8) this._flushed.clear();

    for (let i = 0; i < near.length; i++) {
      const f = near[i];
      const o = i * 4;

      const speed = SPEED_MIN + seed(f.ix, f.iz, 64) * (SPEED_MAX - SPEED_MIN);
      const dwell = DWELL_MIN + seed(f.ix, f.iz, 46) * (DWELL_MAX - DWELL_MIN);
      const phase = seed(f.ix, f.iz, 47) * 137;
      // Provisional, so the pass index can be found; refined once it is
      // known whether this pass stops at all.
      const T0 = run / speed + dwell;
      const pass = Math.floor((opts.time + phase) / T0);

      /* Half the passes are aimed at a tree, because a tree is where a
         flock is going. The rest either put down on open ground or cross
         without stopping at all — and those two are the same behaviour
         with a different landing height, which is why there is one code
         path for the stop and only the y differs. */
      const wantTree = seed(f.ix, f.iz, 60 + pass * 2) < 0.5;

      let crown = null;
      if (wantTree && opts.trees) {
        let bestD = cell * cell;
        for (const c of opts.trees.canopies) {
          const dx = c[0] - f.hx, dz = c[2] - f.hz;
          const dd = dx * dx + dz * dz;
          if (dd < bestD) { bestD = dd; crown = c; }
        }
      }

      const ang = seed(f.ix, f.iz, 61 + pass * 2) * Math.PI * 2;
      const dx = Math.cos(ang), dz = Math.sin(ang);

      /* The stop sits a little past the middle of the line, so a flock
         is seen coming in for longer than it is seen leaving — which is
         the way round the eye wants it, because the approach is the part
         with the descent and the braking in it. */
      const at = run * (0.42 + seed(f.ix, f.iz, 62 + pass * 2) * 0.12);

      let sx, sz, sy, gather = -GROUND_GATHER;
      let stopping = true;

      if (crown) {
        /* The line is laid through the crown rather than the crown found
           along the line: aiming at the tree is the point of the pass.

           Up in the top of it, not at its middle. trees.js reports the
           mean height of the leaf mass, which is where the canopy is
           densest and where a bird sitting would be inside the foliage
           and out of sight — and a flock that cannot be seen in a tree
           is a flock that looks like it stopped beside one. */
        sx = crown[0]; sz = crown[2];
        sy = crown[1] + crown[3] * 0.45;
        gather = crown[3];
      } else if (seed(f.ix, f.iz, 63 + pass * 2) < 0.5) {
        sx = f.hx + dx * (at - run * 0.5);
        sz = f.hz + dz * (at - run * 0.5);
        sy = terrainHeight(sx, sz, opts.hills) + GROUND_PERCH;
      } else {
        stopping = false;
        sx = f.hx; sz = f.hz; sy = 0;
      }

      // The line starts far enough back that the stop lands where it should.
      L[o] = sx - dx * at;
      L[o + 1] = sz - dz * at;
      L[o + 2] = ang;
      L[o + 3] = gather;

      S[o] = sx;
      S[o + 1] = sy;
      S[o + 2] = sz;
      S[o + 3] = stopping ? at : -1;

      const size = Math.round(perFlock * (0.55 + 0.55 * seed(f.ix, f.iz, 44)));
      const snow = snowCoverAt(f.hx, f.hz, opts.hills, opts.waterY,
                               opts.snow, terrainHeight, this._grad);

      /* Faded on where the flock *is*, not on where its cell is.
         A pass runs seventy metres, so a flock spends most of it a long
         way from the point that admitted it — and keyed to the cell, one
         whose home sits near the rim is dim for the whole crossing even
         as it passes overhead, while one with a near home stays bright
         all the way out to the fog. The cell decides which flocks exist;
         where the birds are decides whether you can see them. */
      const T = stopping ? run / speed + dwell : run / speed;
      const pt = ((opts.time + phase) / T % 1) * T;
      const tA = stopping ? at / speed : 0;
      const tL = tA + dwell;
      const along = !stopping ? speed * pt
        : pt < tA ? speed * pt
        : pt < tL ? at
        : at + speed * (pt - tL);

      const wx = L[o] + dx * along, wz = L[o + 1] + dz * along;
      const wd = Math.hypot(wx - ex, wz - ez);
      const wu = Math.min(1, Math.max(0, (wd - far * 0.62) / (far * 0.38)));
      const hereFade = 1 - wu * wu * (3 - 2 * wu);

      /* ── the cat, latched ──
         A flush happens once. It is an event, and an event needs
         somewhere to be remembered — so it is remembered here, keyed to
         the flock's cell and the pass it happened on, and the shader is
         told nothing but a time.

         Read as a distance every frame instead, which is what this used
         to be, it stops being an event: the birds move at the cat's pace
         because their position is a function of where the cat is, and
         they slide back when it leaves. Latching costs one map entry per
         startled flock and makes the escape irreversible, which is the
         only property that mattered.

         Only a flock that is down can be startled — in a tree or on the
         ground, both. One in the air is immune without a test for it,
         because the only thing this can change is when a stay ends, and
         a flock in the air is not in one. */
      let dwellEnd = tL;
      if (stopping) {
        const key = f.ix + ',' + f.iz;
        const seen = this._flushed.get(key);
        if (seen && seen.pass === pass) {
          dwellEnd = seen.at;
        } else if (opts.cat && pt > tA && pt < tL) {
          const dcx = opts.cat[0] - sx, dcz = opts.cat[1] - sz;
          if (dcx * dcx + dcz * dcz < SCARE_FAR * SCARE_FAR) {
            dwellEnd = pt;
            this._flushed.set(key, { pass, at: pt });
          }
        }
      }
      TM[i * 2] = tA;
      TM[i * 2 + 1] = dwellEnd;

      I[o] = size;
      I[o + 1] = T;
      I[o + 2] = phase;
      /* And the hour, into the same fade the distance and the snow use.
         A flock is one object as far as this term is concerned, so the
         whole of it goes together — half a flock roosting while the
         other half commutes is the one arrangement that is wrong at
         both ends. */
      I[o + 3] = hereFade * (1 - snow * 0.85) * roost;

      alive += Math.round(size * I[o + 3]);
    }
    this._lastRun = run;
    this.sparrows = alive;
    if (!alive) return 0;

    this.bird.use({
      ...plantUniforms(camera, env, { wind: opts.wind, jitter: opts.jitter }),
      uPerFlock: perFlock,
      uFlockLine: L,
      uFlockStop: S,
      uFlockInfo: I,
      uFlockN: near.length,
      uFlockRun: run,
      uFlockTimes: TM,
    });
    return MAX_FLOCKS * perFlock * BIRD_VERTS;
  }

  /**
   * Fireflies, over the resolved image, additive.
   *
   * @returns {number} vertices to draw, or 0
   */
  drawGlow(camera, opts) {
    const box = reachFor('glow', opts.reach);
    // The only one of the three that has to ask whether there is a clock
    // in the scene at all — see the note by nightPart.
    const n = Math.min(GLOW_MAX, Math.round(
      Math.max(0, Math.min(1, opts.density)) * nightPart(opts.day, opts.timed)
      * GLOW_PER_M2 * box * box));
    this.fireflies = n;
    if (!n) return 0;

    this.glow.use({
      uCamPos: camera.pos,
      uRight: camera.right,
      uUp: camera.up,
      uFwd: camera.fwd,
      uFocal: camera.focal,
      uAspect: camera.aspect,
      uTime: opts.time,
      uHills: opts.hills,
      uWaterY: opts.waterY,
      uScene: opts.scene,
      uMesh: opts.mesh,
      uMeshOn: opts.meshOn ? 1 : 0,
      uResolution: opts.resolution,
      uIntensity: 0.9,
      uBox: box,
    });
    return n * 6;
  }

  dispose() {
    this.fly.dispose();
    this.bird.dispose();
    this.glow.dispose();
  }
}
