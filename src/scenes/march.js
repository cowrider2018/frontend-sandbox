/* ── scenes/march.js ─────────────────────────────────────────────────
   03 · Signed distance fields — a whole 3D scene in one fragment shader.

   There is no geometry here. Not one vertex, not one triangle: the
   entire image is produced by a single fullscreen triangle whose
   fragment shader walks a ray forward through an implicit surface
   defined by distance functions.

   Two structural decisions carry most of the performance:

   1. THE GROUND IS MARCHED, BUT NEVER BLINDLY. It was a plane once,
      solved in closed form; it is a height field now, and the closed
      form survives as the far-field fallback. What keeps it affordable
      is that the field is trapped in a slab and its slope is bounded, so
      a ray skips everything above the ridge line in a divide and takes
      conservative steps through the rest. A grazing ray that exhausts
      its budget lands on the plane the hills average to, at a distance
      where the fog owns the colour anyway. See scenes/terrain.js.

   2. THE CLUSTER HAS A BOUNDING SPHERE, computed in JS every frame from
      the same numbers that place the spheres. Every distance the shader
      needs — where to start marching, where to give up, how far a
      shadow ray can matter — is derived from it. There is not one
      hard-coded extent left in the shader, so changing the scene's size
      does not mean re-tuning six constants.

   The field itself comes in three layers, so each consumer pays only
   for the detail it can actually show. See `clusterFull`.
   ------------------------------------------------------------------ */

import { Program } from '../core/program.js';
import { Target, DoubleTarget, bindScreen, BLEND } from '../core/gl.js';
import { PRECISION, CONSTANTS, HASH, COLOR, ROTATE, SIMPLEX3, VERT_FULLSCREEN } from '../shaders/common.js';
import {
  BALL_N, RIPPLE_N, RING_MAJOR, RING_MINOR, DISPLACE_AMP,
  CLUSTER_UNIFORMS, CLUSTER_FIELD, CLUSTER_LAYERS, CLUSTER_SHADOW, SKY,
} from './cluster.js';
import {
  TERRAIN_GLSL, HILLS_DEFAULT, HILLS_MAX, terrainHeight, traceTerrainJS,
  WATER_DEFAULT, waterSurfaceY,
} from './terrain.js';
import {
  WEATHER_GLSL, WEATHER_MODES, weatherOf, Precipitation,
} from './weather.js';
import {
  HOUR_DEFAULT, AMBIENT_FIXED, hourDrives, skyAt, advanceHour,
} from './daycycle.js';
import {
  Creatures, CREATURE_REACH_MIN, CREATURE_REACH_MAX, CREATURE_REACH_DEFAULT,
} from './creatures.js';
import { Cat, CAT_PROXY_GLSL, CAT_CAPS, HEAD_YAW_MAX } from './cat/index.js';
import { Laser } from './laser.js';
import { GroundCover, isCovered } from './ground.js';
import { Trees } from './trees.js';
import { CANOPY_SHADE_GLSL } from './canopy.js';
import { WIND_GLSL } from './wind.js';

/* Mouse-look sensitivity. The locked figure is per device pixel and is
   the usual first-person number; the unlocked one is per canvas width,
   so one sweep across the screen is a little over half a turn. */
const LOOK_PER_PIXEL = 0.0026;
const LOOK_PER_SWEEP = 3.4;
/** Vertical look is the camera's elevation, and rides much shallower. */
const PITCH_FROM_LOOK = 0.75;

/** Seconds between shots while the trigger is held. */
const FIRE_INTERVAL = 0.13;
/** How far a beam carries if it never hits anything. */
const BEAM_REACH = 40.0;
/** A ripple that is a line through space, owned by no body. */
const HOST_BORE = -3;
/* How long the cat takes to bring its head round before an aimed shot.
   Not latency to be minimised: a beam that leaves before the animal has
   looked reads as a mis-aim rather than as a reflex. Only the free-cursor
   path pays it — with the pointer captured the cat is already facing
   wherever the crosshair is. */
const AIM_TURN = 0.15;
/* How much more material a beam dissipates than a click. A click dents;
   a beam has to go through, and a hole must be deeper than the wall. */
const BORE_ERODE = 4.0;
/* How much the body comes round with the head even when the neck could
   have covered the whole angle on its own. A cat turning to look does
   move its shoulders a little; none at all reads as an owl. */
const BODY_FOLLOW = 0.25;

/* Uploaded in place of the cat's capsules when there is no cat. The
   shader is gated on the count, but a sampler-free uniform array still
   has to be given something. */
const ZERO_CAPS = new Float32Array(CAT_CAPS * 4);
const ZERO_BOUND = new Float32Array(4);

/** AO weight sum for the reference 5-tap schedule; see ambientOcclusion. */
const AO_REFERENCE = 0.1959;

/** Where the wind control starts. Named because the water is tuned
    against it — see CHOP_NORM. */
const WIND_DEFAULT = 0.55;
/* How fast chop comes up with the wind. Saturating and not linear: the
   first breath of wind puts most of the slope on a lake and a gale adds
   little to it, which is also what keeps the top of the slider from
   turning the surface into noise the temporal filter has to argue with. */
const CHOP_RATE = 2.1;
/* And the scale that puts the default breeze at exactly the surface the
   four wave amplitudes below were tuned on. Derived rather than typed,
   so moving the slider's default moves this with it instead of quietly
   re-tuning the lake. */
const CHOP_NORM = 1 / (1 - Math.exp(-CHOP_RATE * WIND_DEFAULT));

const FRAG_MARCH = /* glsl */`
${PRECISION}
${CONSTANTS}
${HASH}
${COLOR}
${ROTATE}
${SIMPLEX3}
${CLUSTER_UNIFORMS}
${TERRAIN_GLSL}
${WEATHER_GLSL}

#define AO_REFERENCE ${AO_REFERENCE.toFixed(6)}
#define CHOP_RATE ${CHOP_RATE.toFixed(4)}
#define CHOP_NORM ${CHOP_NORM.toFixed(6)}
/* The rings a cat makes: how many per metre, how fast they travel out,
   how far they carry, and how deep. Short, quick and local — this is a
   four-kilo animal, not a boat, and rings that outlived it or crossed the
   whole lake would read as one.

   The depth was found by photographing it rather than by reasoning, and
   the first two guesses were both wrong by a lot. This lake shows its
   normal almost entirely at grazing angles — that is where the fresnel
   term hands the surface over to the reflection — and at a grazing angle
   a ring a metre across is foreshortened into a line. A slope that
   plainly beat the wind's chop on paper was invisible in the frame. */
#define WAKE_K 9.0
#define WAKE_SPEED 5.5
#define WAKE_FALL 0.50
#define WAKE_AMP 0.080

/* Air into water, as the ratio refract() wants: 1 / 1.33.

   The one number that makes the bed behave. It bounds itself, which is
   the property worth knowing here — a ray arriving dead along the
   surface still leaves the interface at 48.8 degrees off vertical, so
   the refracted ray can never travel further sideways than 1.14 times
   the depth, and the horizontal offset below needs no clamp to stay on
   the lake. Snell does the clamping. */
#define WATER_IOR 0.751880

/* How fast each channel goes out, per metre of water. Was one number
   for all three, which is a lake made of grey glass: the thing that
   says water rather than "dark" is that red dies first and the last
   colour left standing before the bed vanishes is blue-green.

   Weighted against luminance the three come to 2.13, which is the 2.1
   they replace — so the depth at which the bed disappears is unchanged
   and only its colour on the way out is new. */
#define WATER_EXTINCT vec3(3.1, 1.9, 1.6)

/* How fast the bed loses its detail, per metre of water travelled.

   Slower than the extinction above, and the first version had it faster
   for an argument that turned out to be self-defeating. The reasoning
   was that detail has to go while the bed is still visible or there is
   no blurred stage at all — true as far as it goes, but it ignored what
   the detail is *for*. The refraction offset is a fraction of a metre;
   the only thing in the ground with a wavelength short enough to show a
   shift that size is the same fine grain this term takes away. Blur it
   at 4.0 and the shallows do go soft, and they take the refraction with
   them: the offset is still computed, still correct, and lands entirely
   inside a colour that no longer varies over the distance it moved.

   At 1.2 the wading depth in 50-lake-shallows keeps about three
   quarters of its grain, which is enough for the chop to visibly drag
   the bed around, and the softening arrives over the half metre after
   it — still ahead of the extinction, which is all the original
   argument actually required. */
#define WATER_BLUR_RATE 1.2

in vec2 vUv;
out vec4 outColor;

uniform vec3  uCamPos, uRight, uUp, uFwd;
uniform vec2  uResolution;
uniform float uFocal;
uniform float uRough, uFloorMix;
/** 0 = the reference grid, 1 = soil, because something is growing on it. */
uniform float uGround;
uniform float uReflect, uFog, uAO, uShadowSoft;

// Only the marcher draws the wavefront's glow; the field itself, and
// every uniform that shapes it, is declared in cluster.js.
uniform float uRippleGlow;
/* The strongest dissipation any live impact carries: 1 for clicks, more
   for a bore. The tracer's step budget is derived from it, so charging
   the worst case unconditionally would slow every ordinary click down to
   a beam's pace for nothing. */
uniform float uErodeMax;

// quality
uniform int   uSteps, uAoTaps, uReflectSteps;
uniform float uReflectLit;

/* The wind, which the marcher has exactly one consumer for.

   Nothing grows in here — the plants are all in the raster half — but
   the lake is a surface the weather works on, and the field that works
   it is the field the grass is already reading. A lake with its own
   private chop is a lake in different weather from the meadow it is
   sitting in, and that is visible the moment both are in frame. */
uniform float uWind;
${WIND_GLSL}

/* How hard the cat is working the surface it is standing in: 0 with dry
   paws, a little while it floats, more while it swims. Where it is doing
   it comes from uCatBound, which this shader already has because the
   shadow needs it — so an animal in the water costs one float. */
uniform float uCatWake;

/* ═══ primitives and the shape ════════════════════════════════════ */
${CLUSTER_FIELD}

${CLUSTER_LAYERS}

${CAT_PROXY_GLSL}
${CLUSTER_SHADOW}

/* ═══ tracing ═════════════════════════════════════════════════════ */

vec3 calcNormal(vec3 p) {
  // Tetrahedral sampling: four taps instead of six, and no bias.
  const vec2 k = vec2(1.0, -1.0);
  const float h = 0.0012;
  return normalize(
    k.xyy * clusterFull(p + k.xyy * h) +
    k.yyx * clusterFull(p + k.yyx * h) +
    k.yxy * clusterFull(p + k.yxy * h) +
    k.xxx * clusterFull(p + k.xxx * h)
  );
}

/**
 * March the cluster only, and only across the span where its bounding
 * sphere says it can exist. A ray that misses the bound returns
 * immediately; a ray that hits starts at the entry point instead of
 * creeping through the empty space in front of it.
 */
float traceCluster(vec3 ro, vec3 rd, int steps, float tMin) {
  vec2 span = sphereSpan(ro, rd, uBound.xyz, uBound.w);
  if (span.y <= tMin) return -1.0;

  float t = max(span.x, tMin);
  float tEnd = span.y;

  // A perturbed field is not a true distance function, so the step has
  // to be under-relaxed — and by how much is not a constant to guess at.
  // The wave's gradient scales with its own amplitude and frequency, so
  // the bound is derived from them: crank the depth and the tracer slows
  // down instead of cutting through the surface.
  //
  // The crater is deliberately absent from this: a boolean subtraction
  // of two 1-Lipschitz fields is still 1-Lipschitz, however deep it
  // goes, so it costs nothing here.
  float lip = 1.0;
  if (uDisplace > 0.0) lip += uDisplace * DISPLACE_AMP * 6.0;
  if (uRippleOn > 0.5) {
    lip += uRippleAmp * uRippleFreq * 0.9;
    // A smoothstep's steepest slope is 1.5 over its width, and that
    // width is fixed, so this term does not grow as the damage opens.
    // Both the dissipation itself and the wave it silences ride the same
    // edge, so both are charged against it.
    lip += (uErode * uErodeMax + uRippleAmp) * 1.5 / ERODE_EDGE;
  }
  float relax = 0.97 / lip;

  for (int i = 0; i < 256; i++) {
    if (i >= steps) break;
    float h = clusterFull(ro + rd * t);
    // Relax the hit threshold with distance: far pixels are subpixel
    // anyway, and it buys back a lot of steps.
    if (h < 0.0008 * t + 0.0006) return t;
    t += h * relax;
    if (t > tEnd) break;
  }
  return -1.0;
}

/**
 * The cluster is marched, the cat is solved, the ground is marched
 * against its own slope bound. Three different answers to the same
 * question, each the cheap one for its own shape — and every one of them
 * is the shared copy, so the cat is lit by exactly the penumbra the
 * spheres are and stands in exactly the hillshade the grass does.
 */
float softShadow(vec3 ro, vec3 rd, float k) {
  return min(min(clusterShadow(ro, rd, k), catShadow(ro, rd, k)),
             terrainShadow(ro, rd, k));
}

/**
 * Five-tap occlusion, generalised to any tap count. The accumulated
 * weight is normalised against the reference schedule so that changing
 * the tap count changes the cost and not the look.
 */
float ambientOcclusion(vec3 p, vec3 n) {
  float occ = 0.0, sca = 1.0, wsum = 0.0;
  float span = 1.0 / max(float(uAoTaps) - 1.0, 1.0);
  for (int i = 0; i < 8; i++) {
    if (i >= uAoTaps) break;
    float h = 0.02 + 0.14 * float(i) * span;
    // If the field is closer than h, something is nearby: that is
    // occlusion — and the cat counts, so it darkens the floor it stands
    // on and the crease where it meets a sphere.
    occ += (h - min(clusterLit(p + n * h), catProxy(p + n * h))) * sca;
    wsum += h * sca;
    sca *= 0.72;
  }
  return clamp(1.0 - 2.2 * occ * AO_REFERENCE / max(wsum, 1e-4), 0.0, 1.0);
}

/* ═══ shading ═════════════════════════════════════════════════════ */

${SKY}
${CANOPY_SHADE_GLSL}

/**
 * The ground's own colour, wherever it is asked for.
 *
 * Split out of material() the moment the water arrived, because the
 * shallows have to show what they are standing on: a lake whose edge
 * does not read the bed is a lake with a painted-on shoreline. It only
 * ever looks at xz, which is why the surface point can be handed to it
 * in place of the bed point underneath.
 *
 * The blur argument is how much of the fine grain to drop, 0 on dry
 * land — no back-ticks in here, this is inside a template literal. Water is
 * the only caller that ever asks for any, and what it is asking for is
 * a low-pass and not a softening: losing the high frequencies *is* what
 * blur is, so the cheap honest implementation is to fade the fine term
 * toward its own mean and leave the broad one alone. It costs one mix.
 *
 * The alternative — sampling this function several times around the
 * point and averaging — is a genuine blur and costs several times two
 * 3D simplex evaluations plus a snow cover, per water pixel. The
 * stochastic version of that (one jittered tap, resolved by the
 * temporal filter) is cheaper still and is not available here: a lake
 * in any wind pins the moving flag, which caps the accumulation blend at 0.55,
 * and a single jittered tap under that cap is visible noise.
 */
vec3 groundAlbedo(vec3 p, float blur, out float rough) {
  /* Soil, once there is a meadow standing on it.
     The grid is a reference surface — a ruled, faintly polished plane
     that says where the floor is. Grass growing out of a ruled plane
     reads as grass growing out of a diagram, and the polish is worse
     than the ruling: a mirror finish between the blades throws the
     cluster back up through the sward. So the ground under the cover
     is rough, dark and mottled, and its job is to be the shadow
     between the blades rather than a surface anyone looks at. */
  if (uGround > 0.5) {
    float broad = snoise(vec3(p.x * 0.42, 3.7, p.z * 0.42)) * 0.5 + 0.5;
    float fine = snoise(vec3(p.x * 3.10, 8.1, p.z * 3.10)) * 0.5 + 0.5;
    /* A third of a metre per cycle, which is the only frequency here
       small enough for water to take away. The broad term is nearly two
       and a half metres across and survives to the depth the extinction
       has already finished it off at, so blurring it would be spending
       an instruction under a colour nobody can see through. */
    fine = mix(fine, 0.5, blur);
    /* Green, not brown. There is no instance budget that puts a blade
       over every square centimetre, so what shows between them has to
       be the same colour as what is standing in it — a dark sward seen
       from above, not the dirt underneath. Soil-coloured ground turns
       a thin meadow into stubble on a ploughed field. */
    vec3 soil = mix(vec3(0.016, 0.030, 0.011), vec3(0.046, 0.086, 0.026), broad);
    rough = 0.94;

    /* And the shore, which until now went straight from dry meadow to
       water with nothing in between — a waterline with no width.

       Ground within a few centimetres of the surface is wet ground, and
       wet ground is the two things this scene already knows how to say:
       darker, and less rough. It costs one height sample the shading has
       already taken, and it lands exactly where it can be seen — the
       grass tapers into the last twelve centimetres before the water, so
       the soil is what is showing there.

       It carries on under the water rather than stopping at the line.
       That is not a special case for the bed: the bed *is* wet, and the
       clarity term is already fading it out of sight by the time the
       difference could be argued with. */
    float damp = smoothstep(-0.45, -0.01, waterDepth(p.xz));
    soil *= mix(1.0, 0.55, damp);
    rough = mix(rough, 0.46, damp * 0.85);
    /* And whatever fell on it. Applied here rather than at every call
       site because there are three of them now — the soil under the eye,
       the bed under the water, and the bounce — and a lake whose floor
       had not heard about the snow would be a lake with a green bottom
       in a white field. */
    return weatherSurface(soil * (0.82 + 0.36 * fine), snowCover(p.xz), rough);
  }

  // floor: a faint grid that fades out with distance
  vec2 g = abs(fract(p.xz * 0.5) - 0.5);
  float line = 1.0 - smoothstep(0.0, 0.03, min(g.x, g.y));
  float fade = exp(-length(p.xz) * 0.09);
  rough = mix(0.42, 0.12, uFloorMix);
  /* The grid is a diagram, not ground: it is the reference surface you
     switch to in order to see where the floor is, and snowing on a
     diagram would only make it harder to read. */
  return mix(vec3(0.024, 0.026, 0.032), uTint * 0.6, line * fade * 0.5);
}

/** A wave vector of a given length, turned off the wind by an angle.
    Both arguments are compile-time constants at every call site, so the
    sine and cosine are folded and this costs a multiply. */
vec2 fetchDir(float len, float spread) {
  float c = cos(spread), s = sin(spread);
  return len * vec2(WIND_DIR.x * c - WIND_DIR.y * s,
                    WIND_DIR.x * s + WIND_DIR.y * c);
}

/**
 * The surface of the water: flat, with four wave trains laid over its
 * normal by whatever wind is blowing across it.
 *
 * Nothing is displaced. A lake seen from eye height is almost entirely
 * reflection, and reflection is a function of the normal alone — so
 * perturbing the normal buys the whole look and the plane stays a plane,
 * which is what keeps traceWater a divide instead of a march.
 *
 * The trains fade out with distance for the reason every other detail in
 * this scene does: ripple finer than a pixel is not ripple, it is noise
 * that the temporal filter then has to spend frames arguing with.
 */
vec3 waterNormal(vec2 q, float dist) {
  float fade = exp(-dist * 0.030);
  if (fade < 0.02) return vec3(0.0, 1.0, 0.0);

  vec2 g = vec2(0.0);

  /* How hard the wind is working the surface, and where in the gust this
     patch of it is.

     The first factor saturates, because that is what chop does: the
     first breath of wind puts most of the slope on a lake and the rest
     of the slider hardly adds to it. The second is the gust itself,
     sampled at the point on the water — the same field, arriving on the
     same beat, as the one crossing the grass a few metres away. That is
     the whole reason to do this: the cat's paw sliding over the lake and
     the ripple going through the meadow are one event, and a surface
     that chops on its own private schedule is the tell that they are
     not.

     Centred on 1 rather than added, so the gust redistributes the chop
     instead of raising it — the lulls go glassy and the gust front takes
     what they gave up. */
  float breeze = (1.0 - exp(-uWind * CHOP_RATE)) * CHOP_NORM;
  breeze *= 1.0 + 0.55 * slowGust(q);
  /* Still air is a mirror, and that is the honest answer rather than a
     special case: water is only rough because something is roughing it.
     It is also eight sines cheaper than the alternative — but it is no
     longer the end of the function, because the wind is not the only
     thing that can be roughing it. */
  if (breeze >= 0.01) {
    /* The trains, fanned about the wind instead of crossing at fixed
       angles. A wave is made by the wind blowing along it, so it travels
       downwind — but not all of them exactly, because four trains dead in
       line is corduroy and the spread either side is what gives a real
       fetch its interference.

       Wave vector carries the direction and the frequency together, so
       the gradient of each term is just that vector times the cosine —
       the same trick, and the same reason, as the hills. The four lengths
       and the four speeds are unchanged; what moved is where they
       point. */
    vec2 gw = vec2(0.0), k;
    k = fetchDir(1.80, -0.38); gw += k * (0.0300 * cos(dot(q, k) - uTime * 1.30));
    k = fetchDir(2.84,  0.52); gw += k * (0.0140 * cos(dot(q, k) - uTime * 1.05));
    k = fetchDir(5.02, -0.86); gw += k * (0.0045 * cos(dot(q, k) - uTime * 1.70));
    k = fetchDir(8.75,  1.02); gw += k * (0.0012 * cos(dot(q, k) - uTime * 2.20));
    g += gw * breeze;
  }

  /* And whatever is standing in it.

     Rings out from the animal, and they are added *after* the wind's
     term rather than scaled by it — a cat in a dead calm is the case
     these exist for, and one that only disturbed water the wind had
     already disturbed would be a cat that cannot be seen to be in the
     lake at all.

     The centre costs nothing: the marcher is already given the cat's
     bounding sphere, because the shadow needs it. What is new is one
     float saying how hard the animal is working the surface — standing
     in the shallows, or swimming — and it is zero the moment its paws
     are dry, which is what keeps a cat walking *past* a lake from
     stirring it.

     The gradient is the derivative of the ring along the radius times
     the radial direction. The wavenumber dominates the falloff by an
     order of magnitude, so the falloff's own derivative is dropped: it
     would be a correction of a few per cent to the slope of something
     already made up. */
  if (uCatWake > 0.0) {
    vec2 rel = q - uCatBound.xz;
    float r = length(rel);
    float fall = exp(-r * WAKE_FALL);
    if (fall > 0.02) {
      g += (rel / max(r, 1e-4))
         * (uCatWake * WAKE_AMP * WAKE_K * cos(r * WAKE_K - uTime * WAKE_SPEED) * fall);
    }
  }

  g *= fade;
  return normalize(vec3(-g.x, 1.0, -g.y));
}

/**
 * The ray direction and the lit flag are here for the water and nothing
 * else: the bed is read along the refracted ray, which needs the
 * direction the eye came in on, and the flag is the same one shadeDirect
 * already carries to say whether this is the primary ray or the
 * reflection bounce.
 *
 * The bounce does not refract. That is not an approximation anyone can
 * catch — it is the lake seen in the lake, at a fraction of a weight,
 * through a second surface that is itself rippling — and it matters
 * because this whole function is inlined twice. Everything added below
 * would otherwise be paid for twice against an instruction ceiling this
 * shader has already hit once.
 */
vec3 material(vec3 p, vec3 n, vec3 rd, float mat, bool lit,
              out float rough, out float metal, out float snowOut) {
  /* Reported back rather than recomputed by the caller. The shading needs
     it too — snow's shaded side is lit almost entirely by the sky, which
     is a change to the *fill* and not to the albedo — and snowCover is a
     terrain evaluation plus three sines, which is not a thing to do twice
     per pixel for an answer that was already on the stack. */
  snowOut = 0.0;
  /* Water.
     Not simulated. Three things sell a lake and this is all of them:
     the shallows show the bed and show it *bent*, the bed goes soft and
     then goes out over the first half metre, and the surface is smooth
     enough that the reflection bounce main() was already firing lands
     on it. The last one is why there is no reflection code here — the
     water is simply the first surface in this scene polished enough to
     use the one that already existed. */
  if (mat > 2.5) {
    metal = 0.0;
    rough = 0.045;

    float d = max(waterDepth(p.xz), 0.0);

    /* Where the bed actually is, as opposed to straight down.

       The surface is a plane and the bed is a height field, so the
       refracted ray does not have to be marched: it has to be crossed
       with a horizontal plane one depth below, and that is a divide.
       The offset it produces is the whole of what reads as refraction —
       not the displacement itself, which at these depths is a few
       centimetres, but the fact that it *moves*, because the normal it
       is computed from is the same one the four wave trains are already
       tilting. The chop wobbles the bed for free.

       The shoreline needs no protection from this. The offset scales
       with the depth and the depth is zero at the water's edge, so the
       sample walks back to the surface point exactly where sampling dry
       ground would start to show.

       Skipped when the ray is not descending, which is the only way
       refract() can be handed a geometry it has no answer for: a camera
       below the surface looking up. The lake is drawn from above and
       this costs a compare. */
    vec2 bedXZ = p.xz;
    float path = d;
    if (lit && rd.y < -1e-3) {
      vec3 rr = refract(rd, n, WATER_IOR);
      float k = d / max(-rr.y, 0.5);
      bedXZ += rr.xz * k;
      path = k;
    }

    /* How far down you can see, per channel now. The one number between
       a puddle and a lake, and the reason the shoreline needs no
       geometry: the bed fades out of view over the first half metre by
       itself.

       Driven by the vertical depth and not the path length, which is
       the shallower of the two claims and deliberately so — it is the
       curve the lake was tuned on, and the blur below is the new thing
       here, not the visibility. */
    vec3 clarity = 1.0 - exp(-d * WATER_EXTINCT);

    /* The blur *is* driven by the path, because that is the distance
       the light actually travelled through water, and at a grazing
       angle it is half again the depth. Bounded by the same Snell
       argument the offset is: it can never exceed 1.52 depths, so this
       needs no clamp either. */
    float blur = 1.0 - exp(-path * WATER_BLUR_RATE);

    float bedRough;
    // Wet, so darker than the same ground in air.
    vec3 bed = groundAlbedo(vec3(bedXZ.x, p.y, bedXZ.y), blur, bedRough) * 0.55;
    vec3 deep = vec3(0.014, 0.043, 0.055);
    return mix(bed, deep, clarity);
  }

  if (mat > 1.5) {
    metal = 0.0;
    snowOut = snowCover(p.xz);
    return groundAlbedo(p, 0.0, rough);
  }

  // body: iridescent, driven by the angle between normal and view
  rough = uRough;
  metal = 1.0;
  float f = dot(n, normalize(uCamPos - p)) * 0.5 + 0.5;
  vec3 base = cosPalette(f * 0.8 + 0.1,
    vec3(0.5, 0.5, 0.55), vec3(0.45, 0.42, 0.42),
    vec3(1.0, 0.98, 0.94), vec3(0.0, 0.22, 0.46));
  vec3 col = mix(base, uTint, 0.28);

  // A hot ring riding the wavefront. The geometric displacement alone is
  // a couple of centimetres on a smooth cream surface — technically
  // correct and almost invisible. This is emission, not geometry, so it
  // costs nothing in field stability and reads instantly.
  if (uRippleOn > 0.5 && uRippleGlow > 0.0) {
    float ring = 0.0;
    for (int i = 0; i < RIPPLE_N; i++) {
      float age = uRipples[i].w;
      if (age <= 0.0) continue;
      float d = length(p - uRipples[i].xyz);
      float front = age * uRippleSpeed;
      ring += exp(-abs(d - front) * uRippleTight * 2.4) * (1.0 - age) * (1.0 - age);
    }
    col += vec3(1.0, 0.86, 0.62) * ring * uRippleGlow * 1.6;
  }
  return col;
}

/**
 * Direct lighting only. GLSL has no recursion, so the reflection bounce
 * cannot call the full shader again — it calls *this*, and the mirror
 * ray therefore sees a correctly lit but non-reflective world. One
 * bounce is all the eye asks for, and it costs a fixed budget instead
 * of an unbounded one.
 *
 * The lit flag switches off the shadow and occlusion queries, which is
 * what the reflection bounce does by default: it is a blurred secondary
 * weighted at a fraction, and its own 50-odd extra field evaluations
 * buy almost nothing.
 */
vec3 shadeDirect(vec3 ro, vec3 rd, float t, float mat, bool lit,
                 out vec3 pOut, out vec3 nOut, out float roughOut) {
  pOut = ro + rd * max(t, 0.0);
  nOut = vec3(0.0, 1.0, 0.0);
  roughOut = 1.0;
  if (mat < 0.5) return sky(rd);

  vec3 p = pOut;
  /* The ground's normal comes out of the same sum of sines that gave its
     height — the derivative of a sine is a cosine, so it is exact and it
     is free. The water's comes out of four more of them, for the same
     reason. Only the cluster needs four field evaluations to find one. */
  vec3 n = mat > 2.5 ? waterNormal(p.xz, t)
         : mat > 1.5 ? terrainNormal(p.xz)
         : calcNormal(p);
  nOut = n;

  float rough, metal, snow;
  vec3 albedo = material(p, n, rd, mat, lit, rough, metal, snow);
  roughOut = rough;

  vec3 l = uLightDir;
  vec3 v = -rd;
  vec3 h = normalize(l + v);

  float ndl = max(dot(n, l), 0.0);

  // A surface facing away from the light is unlit whatever the shadow
  // ray finds, so do not fire one.
  float sh = 1.0;
  if (lit && uShadowSoft > 0.0 && ndl > 0.0) {
    sh = softShadow(p + n * 0.004, l, mix(6.0, 26.0, uShadowSoft));
  }
  /* And the wood. A texture fetch rather than another primitive in
     softShadow, which is inlined twice — once for the primary ray and
     once for the reflection bounce — and is the function that blew the
     instruction limit the last time something was added to it. */
  if (lit) sh = min(sh, canopyShade(p));
  float occ = (lit && uAO > 0.0) ? mix(1.0, ambientOcclusion(p, n), uAO) : 1.0;

  // Blinn-Phong with a roughness-derived exponent: not physically based,
  // but stable, cheap, and it reads correctly next to the SDF shadows.
  float spec = pow(max(dot(n, h), 0.0), mix(400.0, 9.0, rough)) * mix(0.35, 1.6, metal);
  spec *= step(0.0001, ndl);
  float fresnel = pow(1.0 - max(dot(n, v), 0.0), 5.0);

  /* The fill, and the one place snow changes the *lighting* rather than
     the surface.

     Snow is bright because of what falls on it, and on its shaded side
     almost all of that is the sky. Left on the meadow's own fill it comes
     out as a flat pale surface with no blue anywhere in it — which is
     sand, and was exactly what the first version photographed as. Lifting
     the sky term where the snow is lying puts the lit faces warm from the
     sun and the shaded ones blue from above, and that contrast is what
     the eye is actually reading when it calls something snow. */
  vec3 fill = uAmbient * mix(1.0, 3.4, snow);

  vec3 col = albedo * (uTint * 2.3 * ndl * sh + fill * occ);
  col += uTint * spec * sh * 2.0;
  col += sky(reflect(rd, n)) * (0.06 + fresnel * 0.9) * occ;

  // Distance fog toward the sky colour keeps the horizon from ending abruptly.
  float fog = 1.0 - exp(-max(t, 0.0) * uFog * 0.045);
  return mix(col, sky(rd), fog);
}

/* ═══ entry ═══════════════════════════════════════════════════════ */

/**
 * Nearest of the marched cluster, the marched ground and the water.
 *
 * The water is last and cheapest — one divide and one height sample —
 * and it is resolved here rather than inside traceGround so that the
 * ground march stays a function of the ground alone. A ray that crests a
 * ridge before reaching the lake is handled by nothing more than this
 * comparison: the hill came first, so the hill is what it hit.
 */
void trace(vec3 ro, vec3 rd, int steps, out float t, out float mat) {
  float tc = traceCluster(ro, rd, steps, 0.02);
  float tf = traceGround(ro, rd);
  float tw = traceWater(ro, rd);

  t = 0.0; mat = 0.0;
  if (tc > 0.0)                          { t = tc; mat = 1.0; }
  if (tf > 0.0 && (mat < 0.5 || tf < t)) { t = tf; mat = 2.0; }
  if (tw > 0.0 && (mat < 0.5 || tw < t)) { t = tw; mat = 3.0; }
}

void main() {
  vec2 ndc = vUv * 2.0 - 1.0;
  float aspect = uResolution.x / uResolution.y;

  // Jittered per-pixel per-frame: with the temporal blend in the
  // accumulation pass this becomes free anti-aliasing over ~4 frames.
  vec2 jitter = (hash22(gl_FragCoord.xy + uTime * 60.0) - 0.5) / uResolution;
  ndc += jitter * 2.0;

  vec3 rd = normalize(uFwd + uRight * ndc.x * aspect / uFocal + uUp * ndc.y / uFocal);

  float t, mat;
  trace(uCamPos, rd, uSteps, t, mat);

  vec3 p, n; float rough;
  vec3 col = shadeDirect(uCamPos, rd, t, mat, true, p, n, rough);

  if (mat > 0.5 && uReflect > 0.0) {
    float fresnel = pow(1.0 - max(dot(n, -rd), 0.0), 5.0);
    /* Water reflects on its own curve, and it is the real one: Schlick
       from an F0 of 0.02, which is what water is.

       The shared expression was written for the spheres and the ground,
       where a roughness-weighted floor with a mild fresnel on top is a
       fair approximation and nothing looks wrong. Handed to a lake it
       says two things that are visibly false at once. Looking straight
       down it mixes about a tenth of the sky into the water, so the
       deeps never quite go dark and the bed is always competing with a
       haze that is not there. Looking along the surface it stops at
       0.38, when the honest answer at that angle is nearly all of it —
       and a lake is *mostly* seen at that angle, which is why this is
       the single largest change to how the water reads.

       It costs nothing. It is a different expression, not more work,
       and the trace it guards fires slightly less often than before:
       overhead water now lands under the 0.02 skip threshold, which is
       exactly where the bounce was buying the least. */
    float amount = mat > 2.5
      ? uReflect * (0.02 + 0.98 * fresnel)
      : uReflect * mix(0.12, 0.72, 1.0 - rough) * (0.25 + fresnel * 0.75);
    // Below this the bounce cannot move an 8-bit channel; skip the
    // entire second trace.
    if (amount > 0.02) {
      vec3 rd2 = reflect(rd, n);
      vec3 ro2 = p + n * 0.02;
      // Deliberately not named mat2 — that is a built-in type name, and
      // shadowing it is a syntax error rather than a warning.
      float tHit, matHit;
      trace(ro2, rd2, uReflectSteps, tHit, matHit);

      vec3 p2, n2; float rough2;
      vec3 refl = shadeDirect(ro2, rd2, tHit, matHit, uReflectLit > 0.5, p2, n2, rough2);
      col = mix(col, refl, min(amount, 0.9));
    }
  }

  // Alpha carries scene depth, so the additive flare pass can hide
  // itself behind geometry without a depth buffer ever existing.
  outColor = vec4(col, mat > 0.5 ? t : 1e4);
}
`;

/**
 * Temporal accumulation, and the one place the rasterised half of the
 * scene meets the marched half.
 *
 * Both write the same quantity into alpha — distance travelled from the
 * eye, in world units — so resolving them is a comparison and nothing
 * more. No depth buffer is shared, no matrices are reconciled, and the
 * march did not have to learn that there is a cat standing in a meadow
 * somewhere in front of it. Sky is 1e4 on both
 * sides, so an empty pixel resolves to the scene either way.
 *
 * Written to a ping-pong pair, never in place: sampling a texture that
 * is also the current colour attachment is undefined behaviour, and the
 * artefacts it produces look plausible enough to waste an evening on.
 */
const FRAG_ACCUM = /* glsl */`
${PRECISION}
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uSrc;
uniform sampler2D uHistory;
uniform sampler2D uMesh;
/** How much history to keep: the first for the marched half, the second
    for whatever the rasterised half put in front of it. */
uniform float uBlend, uBlendMesh;
uniform float uMeshOn;

void main() {
  vec4 scene = texture(uSrc, vUv);
  vec3 col = scene.rgb;
  float keep = uBlend;

  if (uMeshOn > 0.5) {
    vec4 mesh = texture(uMesh, vUv);
    /* The depth comparison that composites the two halves also says
       which of them a pixel is, and that is worth more than the
       composite: a meadow in wind is a reason to distrust the history
       *of the meadow*, not of the sky above it or the lake beside it.
       One blend for the whole frame threw the accumulation away
       everywhere the grass happened to be moving — which, with the
       default wind, is every frame the cover is switched on. */
    if (mesh.a < scene.a) { col = mesh.rgb; keep = uBlendMesh; }
  }

  outColor = vec4(mix(col, texture(uHistory, vUv).rgb, keep), 1.0);
}
`;

const FRAG_RESOLVE = /* glsl */`
${PRECISION}
${CONSTANTS}
${COLOR}
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uSrc;
uniform float uExposure;

void main() {
  vec3 col = acesFilm(texture(uSrc, vUv).rgb * uExposure);
  vec2 q = vUv - 0.5;
  col *= 1.0 - dot(q, q) * 0.42;
  outColor = vec4(dither(col, gl_FragCoord.xy), 1.0);
}
`;

/**
 * The flare at each impact, added over the resolved image. It samples
 * the marched depth out of the render target's alpha and discards
 * anything behind the surface, so the flare is occluded by the scene
 * without a depth buffer.
 */
const VERT_FLARE = /* glsl */`
${PRECISION}
${CONSTANTS}
#define RIPPLE_N ${RIPPLE_N}

uniform vec4  uFlare[RIPPLE_N];    // xyz = centre, w = world radius
uniform float uFlareAmt[RIPPLE_N]; // 0 = unused
uniform vec3  uCamPos, uRight, uUp, uFwd;
uniform float uFocal, uAspect, uViewportH;

out float vAmt;
out float vFront;

void main() {
  int i = gl_VertexID;
  float amt = uFlareAmt[i];
  if (amt <= 0.0) {
    // Degenerate slot: park it outside the clip volume.
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    gl_PointSize = 1.0;
    vAmt = 0.0;
    vFront = 0.0;
    return;
  }

  vec4 g = uFlare[i];
  vec3 rel = g.xyz - uCamPos;
  vec3 view = vec3(dot(rel, uRight), dot(rel, uUp), dot(rel, uFwd));

  vAmt = amt;
  // Compare the *front* of the sphere against the scene, not its centre,
  // or the flare hides behind the surface it is sitting on.
  vFront = view.z - g.w;

  gl_Position = vec4(view.x * uFocal / uAspect, view.y * uFocal, 0.0, view.z);
  gl_PointSize = clamp(g.w * uViewportH * 3.0 / max(view.z, 0.05), 2.0, 500.0);
}
`;

const FRAG_FLARE = /* glsl */`
${PRECISION}
${CONSTANTS}
in float vAmt;
in float vFront;
out vec4 outColor;

uniform sampler2D uScene;
uniform vec2 uResolution;
uniform vec3 uColor;
uniform float uIntensity;

void main() {
  if (vAmt <= 0.0) discard;

  float sceneDepth = texture(uScene, gl_FragCoord.xy / uResolution).a;
  if (vFront > sceneDepth + 0.05) discard;

  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float d2 = dot(c, c);
  if (d2 > 1.0) discard;

  float halo = exp(-d2 * 3.0) * (1.0 - d2);
  outColor = vec4(uColor * halo * vAmt * uIntensity, 1.0);
}
`;

/* ═══ scene ═══════════════════════════════════════════════════════ */

const TINTS = {
  amber:  [1.0, 0.78, 0.45],
  ice:    [0.55, 0.78, 1.0],
  jade:   [0.5, 1.0, 0.72],
  rose:   [1.0, 0.55, 0.72],
};

const SCALES = ['0.5', '0.75', '1'];

/**
 * The master quality control. Moving it writes every one of these into
 * its own slider — it is an action, not a binding, so the individual
 * controls stay yours afterwards and the master simply goes stale.
 */
function qualityPreset(q) {
  const lerp = (a, b) => a + (b - a) * q;
  // The ranges are placed so that the midpoint lands on each budget's
  // measured knee — the point past which `verify.mjs --diff` reports a
  // maximum channel difference of zero. Above the midpoint you are
  // buying margin for camera angles and blend settings that were not in
  // the measurement; below it you are trading something visible.
  return {
    scale: SCALES[q < 0.34 ? 0 : q < 0.78 ? 1 : 2],
    steps: Math.round(lerp(40, 160)),          // knee ≈ 70
    shadowSteps: Math.round(lerp(8, 56)),      // knee ≈ 32
    aoTaps: Math.round(lerp(2, 8)),            // knee ≈ 5
    reflectSteps: Math.round(lerp(16, 96)),    // still moving at 48
    // Both of these are visible, if only across a fraction of a percent
    // of the frame, and both now cost about a millisecond. They come on
    // well before the midpoint so the default matches what the scene
    // looked like before any of this — the speed is the change, not the
    // picture.
    shadowNoise: q >= 0.35,
    reflectLit: q >= 0.45,
  };
}

export default {
  id: 'march',
  index: '03',
  title: 'SDF 光線行進',
  tech: 'sphere tracing · bounded field · height-field ground · impact ripples',
  desc: '整個 3D 場景沒有任何一個頂點：一道 fragment shader 沿著射線走進隱式曲面。點擊星體或星環，被碰到的那一點會盪出漣漪。',
  glyph: '◈',
  hue: 62,

  params: [
    { group: '幾何' },
    { id: 'balls', type: 'slider', label: '球體數', min: 1, max: 9, step: 1, value: 6 },
    { id: 'blend', type: 'slider', label: '融合半徑', min: 0.02, max: 0.75, step: 0.005, value: 0.32 },
    { id: 'displace', type: 'slider', label: '表面擾動', min: 0, max: 1, step: 0.01, value: 0.14 },

    { group: '撞擊' },
    // The old 0.055 cap existed because the tracer's step was a fixed
    // 0.88 and a deeper wave cut straight through the surface. The step
    // is derived from this value now, so the ceiling can go up — it just
    // costs frames as you climb.
    { id: 'rippleAmp', type: 'slider', label: '漣漪深度', min: 0, max: 0.16, step: 0.001, value: 0.042, digits: 3 },
    // How much material the blow takes away. The ring is 0.075 thick, so
    // anything past about 0.10 opens a gap in it; a sphere's 0.30 radius
    // needs roughly 0.38 before the far side shows through.
    { id: 'erode', type: 'slider', label: '質量消散', min: 0, max: 0.45, step: 0.005, value: 0.10, digits: 2 },
    { id: 'rippleSpeed', type: 'slider', label: '傳播速度', min: 0.2, max: 4, step: 0.01, value: 1.0 },
    { id: 'rippleFreq', type: 'slider', label: '波數', min: 4, max: 60, step: 0.5, value: 16 },
    { id: 'rippleLife', type: 'slider', label: '持續時間', min: 0.4, max: 6, step: 0.05, value: 1.9, unit: 's' },
    { id: 'flash', type: 'slider', label: '閃光', min: 0, max: 3, step: 0.01, value: 1.0 },

    { group: '光線' },
    { id: 'daylight', type: 'select', label: '天光', value: 'fixed',
      options: [
        { value: 'fixed', label: '固定' },
        { value: 'hour', label: '指定時刻' },
        { value: 'cycle', label: '自動循環' },
      ] },
    { id: 'hour', type: 'slider', label: '時刻', min: 0, max: 24, step: 0.05,
      value: HOUR_DEFAULT, unit: 'h' },
    { id: 'light', type: 'xy', label: '光源方向', value: [0.68, 0.24] },
    { id: 'hintDay', type: 'hint',
      text: '**天光**說的是誰在管光源，因為「時刻」和「光源方向」是同一個問題的兩個答案，'
        + '與其讓它們互相搶，不如直接講明白：'
        + '**固定**＝方向盤說了算，和這個檔案存在之前一模一樣（舊的網址和參考截圖仍然成立）；'
        + '**指定時刻**＝時刻滑桿說了算，方向盤變成過時的讀數；**自動循環**＝同上，而且時刻自己會走。'
        + '三個狀態不是兩個開關，因為「方向盤在管」和「時鐘在管」不可能同時成立。'
        + '一天 130 秒。太陽落下時月亮從對面升起——**兩者是同一道弧讀正讀反**，'
        + '所以沒有第二顆天體要跟第一顆對齊；黃昏時換的只是那一個 `uLightDir` 指著誰，'
        + '而換的瞬間兩者都在地平線上、方向項已經降到零，所以看不出來。'
        + '低角度的太陽不只是變紅**也變暗**：只改色相的夕陽讀起來像一盞有顏色的燈。' },
    { id: 'tint', type: 'select', label: '光色', value: 'amber',
      options: [
        { value: 'amber', label: '琥珀' },
        { value: 'ice', label: '冷冽' },
        { value: 'jade', label: '翡翠' },
        { value: 'rose', label: '玫瑰' },
      ] },
    { id: 'shadow', type: 'slider', label: '陰影銳利度', min: 0, max: 1, step: 0.01, value: 0.55 },
    { id: 'ao', type: 'slider', label: '環境遮蔽', min: 0, max: 1, step: 0.01, value: 0.85 },
    { id: 'reflect', type: 'slider', label: '反射', min: 0, max: 1, step: 0.01, value: 0.55 },
    { id: 'rough', type: 'slider', label: '粗糙度', min: 0.02, max: 1, step: 0.01, value: 0.22 },

    { group: '品質' },
    { id: 'quality', type: 'slider', label: '總體品質', min: 0, max: 1, step: 0.01, value: 0.5 },
    { id: 'hintQ', type: 'hint',
      text: '動「總體品質」會一次寫入下面七項；之後仍可單獨微調任何一項。' },
    { id: 'scale', type: 'select', label: '渲染縮放', value: '0.75',
      options: [
        { value: '0.5', label: '50%' },
        { value: '0.75', label: '75%' },
        { value: '1', label: '100%' },
      ] },
    { id: 'steps', type: 'slider', label: '行進步數', min: 24, max: 220, step: 1, value: 100 },
    { id: 'shadowSteps', type: 'slider', label: '陰影步數', min: 4, max: 64, step: 1, value: 32 },
    { id: 'aoTaps', type: 'slider', label: '遮蔽取樣', min: 2, max: 8, step: 1, value: 5 },
    { id: 'reflectSteps', type: 'slider', label: '反射步數', min: 6, max: 110, step: 1, value: 56 },
    { id: 'shadowNoise', type: 'switch', label: '陰影含表面擾動', value: true },
    { id: 'reflectLit', type: 'switch', label: '反射含陰影遮蔽', value: true },

    { group: '地面' },
    { id: 'visibility', type: 'slider', label: '能見度', min: 40, max: 220, step: 1, value: 66, unit: 'u' },
    { id: 'hintVis', type: 'hint',
      text: '**能見度**是主控：它同時定霧的濃度**並寫入**下面的「植被視距」，'
        + '之後那支滑桿仍然是你的（跟「總體品質」與它下面七項的關係一樣）。'
        + '兩者刻意不鎖死——同一個霧濃度需要的視距會隨鏡頭高度與俯角改變，'
        + '而且霧是氛圍、視距是預算，鎖在一起就禁掉了「濃霧＋遠視距」這個又便宜又好看的組合。'
        + '真正的約束只是單向的：**視距要 ≥ 霧關閉的距離**，否則會看見世界的邊緣。' },
    { id: 'ground', type: 'select', label: '地面造型', value: 'grid',
      options: [
        { value: 'grid', label: '網格' },
        { value: 'grass', label: '草地' },
      ] },
    { id: 'hills', type: 'slider', label: '地形起伏', min: 0, max: HILLS_MAX, step: 0.05, value: HILLS_DEFAULT, unit: 'u' },
    { id: 'hintHills', type: 'hint',
      text: '地面是一個高度場，不是一張貼圖：它有真的輪廓、真的背光面，'
        + '而且**自己會投影**——把光源壓低，影子會被拉長掃過整片地。'
        + '高度是幾個正弦的疊加，所以斜率有上界，'
        + '射線可以用保守步長走過去而不是沿著地面爬——這就是起伏付得起真陰影的原因。'
        + '原點附近是**平的**：星團懸在那裡、貓在那裡玩，兩者都是對著平地量出來的。'
        + '拉到 0 就完全還原成原本那片平地。' },
    { id: 'weather', type: 'select', label: '天氣', value: 'clear',
      options: [
        { value: 'clear', label: '晴' },
        { value: 'rain', label: '雨' },
        { value: 'snow', label: '雪' },
      ] },
    { id: 'hintWeather', type: 'hint',
      text: '三選一，不是三支滑桿：雨把地弄濕、雪把地蓋白，'
        + '一個能給你「一半雨加一半雪」的控制是沒有人能拿來問問題的控制。'
        + '落下來的東西吃的是**草和花正在讀的同一個風場**——'
        + '所以掃過草地的那陣風，和穿過它落下的那陣雨，是同一陣風。'
        + '雪是**覆蓋率不是高度**：它按坡度衰減（陡坡留不住），'
        + '並在靠近水面時收到零，否則岸邊會浮出一圈站在水上的白。'
        + '刻意**不**去抬高地形——雪只有幾公分厚，在這個尺度下看不見，'
        + '卻會把地面賴以被行進的斜率上界弄壞。'
        + '天氣原本只到**天空底下**為止，所以曾經有一格沒人會特地去找的畫面：'
        + '**下著雪的午夜，滿天星斗。** 現在模式表有第三個數 `overcast` 進到天空裡：'
        + '星星沒了、日月的**盤面整個消失**（陰天的定義就是找不到那個盤）、'
        + '光暈反而變大變糊，而最要緊的是**梯度被壓平**——'
        + '晴空是頂上深、地平線亮，陰天是一片到處差不多亮的發光板，'
        + '那比任何顏色都更早說出「有雲」。壓平是往**地平線自己的顏色**壓，'
        + '所以每個時刻的色盤自動保留：陰天的黃昏還是橘的，陰天的午夜還是接近全黑。' },
    { id: 'water', type: 'switch', label: '湖', value: false },
    { id: 'waterLevel', type: 'slider', label: '水位', min: 0, max: 1, step: 0.01, value: WATER_DEFAULT },
    { id: 'hintWater', type: 'hint',
      text: '水是**一個數字**：水面的高度。它是一個水平面，射線用一次除法就碰到它，'
        + '不需要行進——而讓它讀起來像湖而不是蓋在世界上的一片玻璃的，'
        + '是「只有當那一點底下的地更低才算命中」這一條，代價是一次高度取樣。'
        + '所以**岸線不是畫出來的**：它就是地形跟水面相交的地方，'
        + '拉「地形起伏」或拉「水位」它都會跟著動，而草、花、樹也用同一條規則決定自己有沒有被淹到。'
        + '水位是**相對於起伏**的：0.5 在一公尺的丘陵和八公尺的丘陵都是「谷底積水一半」。'
        + '地形起伏為 0 時沒有水——平地跟水面共面，誰在前面沒有穩定的答案。'
        + '水面本身沒有模擬也沒有折射：湖幾乎全是反射，而反射只是法線的函數，'
        + '所以擾動法線就買到整個外觀，平面仍然是平面。'
        + '**擾動它的是「風」那支滑桿**——四道波列順著風向散開行進，'
        + '振幅隨風飽和，並被草和樹讀的**同一陣陣風**調製：'
        + '所以掃過草原的那陣風，你會看見它掃過湖面（真實湖面上那片粗糙的暗紋）。'
        + '把風關到 0，湖就是一面鏡子——那不是特例，是誠實的答案：**水會粗糙只因為有東西在弄粗它**。'
        + '**貓也算一個弄粗它的東西**：牠站在水裡會盪出漣漪，移動時更大、不動時仍有；'
        + '那一圈的中心不用花任何代價，因為行進器本來就拿著牠的包圍球（陰影要用）。'
        + '而水深過牠身高一半，牠就從走路換成**浮在水面上**——'
        + '游泳不是第二套動畫，是同一套步態把腿的靜止角度往後移到身體底下、'
        + '擺動縮到四分之一、而且**永遠不停**（停止划水的貓會沉）。'
        + '唯一真正換掉的是**哪些腿一起動**：走路是對角（那是用來把動物撐住的），'
        + '水裡沒有東西在撐它，所以變成**左右同向、前後反向**——那是用來把自己拉著走的。'
        + '尾巴在水裡只做三件事：**彎曲方向與陸地相反**（把烘焙進去的鉤子穿過直線彎到另一邊）、'
        + '**放平貼著水面**、以及**上下輕輕飄動**——三者都從釘住的第一個節點往外長，根部不動。'
        + '為此變形本身也換掉了：從「把每個頂點繞骨頭原點轉」換成**逐節點的座標框鏈**，'
        + '位置由切線積分而來。舊做法會讓外圈的環追過內圈，那就是那個一直都在的**摺疊**——'
        + '這條尾巴最外面三個節點離原點 2.23／2.35／2.40，中間根本沒有半徑可以攤開角度。'
        + '沒入水下的部分不需要裁切：湖是行進器畫的、貓是光柵畫的，合成取近的那個。' },
    { id: 'cover', type: 'slider', label: '草的密度', min: 0.1, max: 1, step: 0.01, value: 0.7 },
    { id: 'flowers', type: 'switch', label: '花', value: false },
    { id: 'flowerClumps', type: 'slider', label: '花叢密度', min: 0.05, max: 1, step: 0.01, value: 0.62 },
    { id: 'flowerDensity', type: 'slider', label: '叢內花密度', min: 0.1, max: 1, step: 0.01, value: 0.7 },
    { id: 'flowerSpread', type: 'slider', label: '花叢範圍', min: 0.2, max: 3, step: 0.01, value: 1 },
    { id: 'reeds', type: 'switch', label: '蘆葦', value: false },
    { id: 'reedDensity', type: 'slider', label: '蘆葦密度', min: 0.05, max: 1, step: 0.01, value: 0.75 },
    { id: 'hintReeds', type: 'hint',
      text: '**要先有湖**——蘆葦是這裡第一個「要水」而不是「躲水」的東西。'
        + '草、花、樹問水深都只問一件事（我淹到了沒有），三種回答方式不同但答案都是「不在這裡」，'
        + '所以岸線變成一條**沒有東西住在上面的線**，讀起來像被割草機修過。'
        + '蘆葦問的是**同一個場的同一個問題**，只是接受一個**區間**而不是半平面：'
        + '從岸上 26 公分到水裡 42 公分。這就是全部的差別，'
        + '所以它不需要第二份「水在哪裡」的描述——水位或起伏一動，這條帶自己就跟著走。'
        + '它也不是撒下去再丟掉淹死的：一個格子**挑一個自己想站的水深**，'
        + '然後沿著高度場的梯度走一步過去（梯度是正弦的餘弦，本來就在手上）——'
        + '所以找岸線是兩次取樣而不是一次搜尋，而且淺灘自然比陡岸長得多，那也正是蘆葦真正的分布。'
        + '**水面下那一截不需要任何裁切**：湖是行進器畫的、蘆葦是光柵畫的，合成取近的那個，'
        + '所以沒入水下的莖自然被水面擋掉——一根站在湖裡的蘆葦，是深度緩衝在做它的工作。'
        + '穗不是第二個物件，是**莖的一段寬度**：香蒲就是一根在接近頂端處鼓起二十公分再收掉的莖，'
        + '所以它跟著莖一起彎，而焊在頂端的頭不會。' },
    { id: 'coverRadius', type: 'slider', label: '植被視距', min: 8, max: 200, step: 1, value: 15 },
    { id: 'trees', type: 'switch', label: '樹', value: false },
    { id: 'treeDensity', type: 'slider', label: '樹的密度', min: 0.1, max: 1, step: 0.01, value: 0.6 },
    { id: 'wind', type: 'slider', label: '風', min: 0, max: 1.4, step: 0.01, value: WIND_DEFAULT },
    { id: 'butterflies', type: 'slider', label: '蝴蝶', min: 0, max: 1, step: 0.01, value: 0 },
    { id: 'fireflies', type: 'slider', label: '螢火蟲', min: 0, max: 1, step: 0.01, value: 0 },
    { id: 'sparrowFlocks', type: 'slider', label: '麻雀群數', min: 0, max: 1, step: 0.01, value: 0 },
    { id: 'sparrows', type: 'slider', label: '群體規模', min: 0, max: 1, step: 0.01, value: 0.7 },
    { id: 'lifeRadius', type: 'slider', label: '生物視距',
      min: CREATURE_REACH_MIN, max: CREATURE_REACH_MAX, step: 1,
      value: CREATURE_REACH_DEFAULT, unit: 'u' },
    { id: 'hintLife', type: 'hint',
      text: '一種一支滑桿，不是一支「生物」總量——'
        + '蝴蝶是白天看的，螢火蟲是把曝光壓下去才找得到的，'
        + '一支滑桿沒辦法同時為兩者調對。'
        + '**而「什麼時候」是時刻在管**（天光＝指定時刻／自動循環時）：'
        + '蝴蝶隨天亮起飛、黃昏收工；螢火蟲只在有夜的時候出現；'
        + '麻雀不發光也不消失，牠是睡在看不進去的樹籬裡——'
        + '所以牠淡出，而不是被派去停棲：一群的位置是從「牠在這趟行程的哪裡」推出來的，'
        + '在黃昏把停留時間拉長就是把週期換掉，整群會在一幀之內跳到別的地方。'
        + '三者用的是同一個 `day`，但只有螢火蟲要問「**這個場景裡有沒有時鐘**」——'
        + '天光＝固定時 `day` 恰好是 1（那正是舊網址與舊截圖仍然成立的原因），'
        + '而 1 是正午：對白天飛的東西已經是對的答案，對夜行的則正好相反。'
        + '沒有時刻的場景裡沒有夜，所以裡面沒有東西是夜行的。'
        + '兩者都問地面同樣的問題：**湖上不會有**，雪地上會變少——'
        + '牠們知道岸線在哪，是因為高度場和水位只有一份答案，不是因為誰去告訴牠們。'
        + '**蝴蝶就是兩片翅膀**，沒有身體：在牠佔到的像素數下，'
        + '身體只是兩翅之間的一團暗色，而讀起來像蝴蝶的是兩個面交替接住又失去陽光的那個閃爍。'
        + '牠是幾何，和草畫進同一個深度緩衝，所以草會擋在牠前面。'
        + '**螢火蟲連翅膀都沒有**，牠是一個點：你看到的是光，而光是加上去的東西，不是會擋住東西的表面。'
        + '**麻雀是第一個矇混不過去的**：認出一隻小雀的是輪廓——'
        + '三分之一處最寬的胸、比胸小而且沒有脖子接上去的頭、後面一條長尾。'
        + '所以牠的身體不是畫出來的，是一張**量出來的環表**（頭胸半徑比 0.64，真麻雀約 0.65，'
        + '**1.0 以上就是童書插圖**）。九公分的身體、二十一公分翼展、五公分尾。'
        + '牠 70 個三角形而蝴蝶只有 4 個——付得起只因為**牠們數量少**：'
        + '一片草原有幾百隻蝴蝶和幾隻麻雀，整群加起來還不到草在一平方公尺上花的。' },
    { id: 'hintGround', type: 'hint',
      text: '草與花是三角形，不在距離場裡——它們和貓畫進同一張深度緩衝，'
        + '所以貓是站在草裡而不是站在草的圖片上。'
        + '草皮跟著鏡頭走並對齊格線，每一株的位置由它所在格子的**世界座標**決定，'
        + '所以格線在滑動、草沒有。風是一個吹過整片地的場，草和花讀的是同一份。'
        + '花的兩支滑桿是兩種不同的畫面：**花叢密度**是「有幾叢」，'
        + '**叢內花密度**是「一叢有多密」——'
        + '疏落的幾叢濃花，和撒滿整片地的稀花，一支滑桿到得了兩端卻說不出自己在做哪一種。'
        + '**花叢範圍**是第三件事：同樣數量的花攤在多大一片地上——'
        + '拉小是緊實的一球，拉大到叢與叢重疊就成了連續的散花。'
        + '「視距」拉遠是一望無際的草原，拉近把幀數還回來。它加的是**面積不是尺寸**：'
        + '近處那一圈的格寬永遠不變，遠的距離是往外加一圈比一圈粗的環買來的，'
        + '所以腳邊的草在任何設定下都長得一樣，成本則隨環數（約是視距的對數）成長。'
        + '**草、花、樹共用這一個視距**，各自帶自己的比例與上限：'
        + '花是 0.62 倍（封頂 60），樹是 0.75 倍（封頂 90）——'
        + '樹看得比花遠，因為樹是這裡唯一高到能構成天際線的東西，遠處那道樹線就是「多遠」本身。' },

    { group: '貓' },
    { id: 'cat', type: 'switch', label: '顯示貓', value: true },
    { id: 'skin', type: 'select', label: '花色', value: 'orangin',
      options: [
        { value: 'orangin', label: '橘白' },
        { value: 'tabby', label: '虎斑' },
        { value: 'calico', label: '三花' },
      ] },
    { id: 'camera', type: 'select', label: '鏡頭', value: 'orbit',
      options: [
        { value: 'orbit', label: '鎖定星體' },
        { value: 'follow', label: '跟隨貓（第三人稱）' },
      ] },
    { id: 'hintCat', type: 'hint',
      text: 'WASD 驅動貓在地面上走動。預設「依鏡頭方向」：WASD 是畫面上的前後左右，'
        + '貓會轉向該方向再走過去——按 S 牠就轉過身朝你走來。'
        + '按 Y 切換成「滑鼠轉向」：滑鼠滑動轉身（不用按鍵）、A／D 改成左右平移，'
        + '**按住滑鼠左鍵從眼睛連發雷射**（準心方向），一路打穿星體；'
        + '指標未鎖定時改成點擊瞄準——貓會轉身面向游標指到的點再射。'
        + '這個模式會鎖定指標，Esc 放開，點畫布重新鎖定。' },

    { group: '呈現' },
    { id: 'taa', type: 'slider', label: '時間累積', min: 0, max: 0.94, step: 0.01, value: 0.78 },
    { id: 'exposure', type: 'slider', label: '曝光', min: 0.2, max: 3, step: 0.01, value: 1.25 },
    { id: 'spin', type: 'switch', label: '自動繞行', value: true },
    { id: 'hint', type: 'hint', text: '點擊星體或星環會在該點盪出漣漪；拖曳畫布繞行鏡頭，滾輪縮放。' },
  ],

  init(ctx) { return new MarchScene(ctx); },
};

class MarchScene {
  constructor(ctx) {
    this.ctx = ctx;
    const { gl } = ctx;

    this.march = new Program(gl, VERT_FULLSCREEN, FRAG_MARCH, { name: 'march/scene' });
    this.accum = new Program(gl, VERT_FULLSCREEN, FRAG_ACCUM, { name: 'march/accum' });
    this.resolve = new Program(gl, VERT_FULLSCREEN, FRAG_RESOLVE, { name: 'march/resolve' });
    this.flare = new Program(gl, VERT_FLARE, FRAG_FLARE, { name: 'march/flare' });

    this.rt = new Target(gl, { width: 2, height: 2, format: 'rgba16f', filter: gl.LINEAR });
    this.history = new DoubleTarget(gl, { width: 2, height: 2, format: 'rgba16f', filter: gl.LINEAR });

    /* Where everything made of triangles goes — the cat, and the grass
       it stands in. One target and one depth attachment between them,
       so they sort against each other for free and resolve against the
       march as a single layer. The canvas still has no depth buffer. */
    this.meshRT = new Target(gl, { width: 2, height: 2, format: 'rgba16f', filter: gl.LINEAR, depth: true });
    this.ground = new GroundCover(gl);
    this.trees = new Trees(gl);
    this.precip = new Precipitation(gl);
    this.creatures = new Creatures(gl);

    /* Loaded off the critical path: the scene renders from the first
       frame and the cat joins when its geometry arrives. A failure here
       is not fatal — you lose the cat, not the scene. */
    this.cat = null;
    this.catError = null;
    this._showCat = false;
    this._catView = false;
    /** The ground's amplitude, refreshed every frame. Held because the
        input handlers fire beams and have no state to read. */
    this.hills = HILLS_DEFAULT;
    Cat.load(gl, new URL('./cat/cat.bin', import.meta.url).href)
      .then((cat) => { this.cat = cat; })
      .catch((err) => { this.catError = err; console.error('cat failed to load', err); });

    this.yaw = 0.85;
    this.pitch = 0.22;
    this.dist = 4.6;
    this.targetDist = 4.6;
    /* What the camera orbits, and what it looks at. They are not the
       same point — the locked view circles the origin but aims a little
       above it — and in follow mode both chase the cat. */
    this._treeJitter = new Float32Array(2);
    this._catXZ = new Float32Array(2);
    this.center = new Float32Array([0, 0, 0]);
    /** The orbit centre in xz, for anything that must not move when
        the view is merely turned. */
    this._orbitXZ = new Float32Array(2);
    this.target = new Float32Array([0, 0.1, 0]);
    this.camMode = 'orbit';
    this.scale = 0;
    this.width = 2;
    this.height = 2;
    this.moving = 1;
    this.movingMesh = 1;
    this.time = 0;
    this.frameCount = 0;

    this.basis = {
      pos: new Float32Array(3),
      right: new Float32Array(3),
      up: new Float32Array(3),
      fwd: new Float32Array(3),
    };
    this.lightDir = new Float32Array([0.5, 0.6, 0.4]);
    /** The scene's own hour, advanced by the simulation clock rather than
        read from it — so pausing stops the sun and `.` steps it one frame
        like everything else. */
    this.hour = HOUR_DEFAULT;
    this._hourShown = -1;

    /* ── the cluster, computed here and uploaded ── */
    this.ballPos = new Float32Array(BALL_N * 4);
    this.ballCount = 6;
    this.blend = 0.32;
    /** xyz = centre, w = radius. The one number every distance derives from. */
    this.bound = new Float32Array([0, 0, 0, 2]);

    /* ── impacts ── */
    this.ripples = new Float32Array(RIPPLE_N * 4);         // world xyz + age
    /** The impact's other end. Equal to the first for a click; the far
        side of the crossing for a beam. */
    this.rippleTo = new Float32Array(RIPPLE_N * 4);
    /** ball index · -1 ring · -2 unused · HOST_BORE a line through space */
    this._rippleHost = new Int32Array(RIPPLE_N).fill(-2);
    /** Length of the impact. Zero for a click; the crossing for a beam. */
    this._rippleLen = new Float32Array(RIPPLE_N);
    /** The bore's far end, in world space, since nothing carries it. */
    this._boreEnd = new Float32Array(RIPPLE_N * 3);
    this._rippleLocal = new Float32Array(RIPPLE_N * 3);    // unit dir, or ring-local point
    this._rippleAge = new Float32Array(RIPPLE_N);
    this._rippleNext = 0;

    this.flarePts = new Float32Array(RIPPLE_N * 4);
    this.flareAmt = new Float32Array(RIPPLE_N);
    this.flash = 0;
    this.bursts = 0;
    this.erodeMax = 0;

    /* ── eye beams ── */
    this.laser = new Laser(gl);
    this.shots = 0;

    this._aim = new Float32Array(3);
    this._fireCooldown = 0;
    /** A shot waiting on the cat's head to come round. */
    this._aimShot = null;
    this._eyeA = new Float32Array(3);
    this._eyeB = new Float32Array(3);

    /** The masters are actions; these remember where each last fired. */
    this._lastQuality = null;
    this._lastVis = null;

    /* ── click vs drag ── */
    this._pressed = false;
    this._dragDist = 0;
    this._pendingBurst = false;
    this._ray = new Float32Array(3);
    this._hit = new Float32Array(3);

    this._onWheel = (e) => {
      e.preventDefault();
      this.targetDist = clamp(this.targetDist * Math.exp(e.deltaY * 0.0011), 2.0, 12);
    };
    ctx.canvas.addEventListener('wheel', this._onWheel, { passive: false });

    /* ── mouse-look ──
       Pointer lock is what makes turning unbounded: without it the
       cursor reaches the edge of the screen and the cat stops turning
       halfway through a corner. It is requested on the keypress that
       enters the mode, which is the user gesture browsers require.

       It is not required, though. If the lock is refused, or the user
       drops it with Escape, the mode still works off the ordinary
       pointer deltas the app already accumulates — it just runs out of
       desk. Degrading to something usable beats refusing the mode. */
    this._lookDx = 0;
    this._lookDy = 0;

    this._onMouseMove = (e) => {
      if (document.pointerLockElement !== ctx.canvas) return;
      this._lookDx += e.movementX;
      this._lookDy += e.movementY;
    };
    this._onLockChange = () => {
      this._lookDx = this._lookDy = 0;
    };
    document.addEventListener('mousemove', this._onMouseMove);
    document.addEventListener('pointerlockchange', this._onLockChange);
  }

  /** True while the cursor is captured, which changes what the mouse means. */
  get locked() {
    return document.pointerLockElement === this.ctx.canvas;
  }

  /* ── pointer ──────────────────────────────────────────────────── */

  /**
   * Click versus drag: a drag orbits the camera, a click strikes the
   * surface. They are told apart by how far the pointer travelled
   * between press and release — and that has to be decided on the
   * *events*, not on whatever the render loop happened to observe: at
   * 25 fps a quick click can begin and end between two frames.
   */
  onPointerDown() {
    const alreadyDown = this._pressed;
    this._pressed = true;
    this._dragDist = 0;
    if (alreadyDown) return;   // the same press, reported twice

    /* Escape drops the lock, and clicking the canvas is the convention
       for getting it back. The click still fires: the lock can be
       refused outright — an embedded or automated document may never be
       granted it — and a trigger that silently did nothing there would
       contradict the mode being built to work unlocked. */
    if (this._catView && this.cat.mode === 'look' && !this.locked) this._requestLock();
  }

  onPointerUp() {
    /* What a click means on release depends on what the mouse is for.
     *
     *   mouse-look   the mouse is the weapon; the trigger is handled by
     *                the held state, and nothing is owed here
     *   with a cat   the mouse also orbits, so only a click that did not
     *                drag counts, and it fires one shot
     *   no cat       the original behaviour: it strikes the surface
     */
    const clicked = this._pressed && this._dragDist < 0.015;
    const look = this._catView && this.cat?.mode === 'look';

    if (clicked && !look) {
      if (this._catView) this._pendingShot = true;
      else this._pendingBurst = true;
    }
    this._pressed = false;
    this._fireCooldown = 0;   // the next press fires at once
  }

  /**
   * WASD drives the cat. The app's own shortcuts include S for
   * screenshot and R for reset, so this claims a key only when there is
   * actually a cat to steer — and says so by returning true, which is
   * what stops the app from acting on it as well. With the cat hidden,
   * every key means what it always meant.
   */
  onKey(e, down) {
    if (!this.cat || !this._showCat) return false;

    // Y swaps how the cat is driven. Guarded against auto-repeat, or
    // holding the key flips the mode sixty times a second.
    if (down && !e.repeat && e.key.toLowerCase() === 'y') {
      this._setControlMode(this.cat.mode === 'look' ? 'camera' : 'look');
      return true;
    }
    return this.cat.onKey(e, down);
  }

  /**
   * Switching modes also decides who owns the cursor. Entering mouse-look
   * asks for it; leaving gives it back, because a captured cursor cannot
   * reach the parameter panel and there is no way to guess that Escape
   * is what releases it.
   */
  _setControlMode(mode) {
    if (!this.cat?.setMode(mode)) return;
    this._lookDx = this._lookDy = 0;

    if (mode === 'look') this._requestLock();
    else if (this.locked) document.exitPointerLock?.();
  }

  /**
   * Ask for the cursor, and do not care if the answer is no.
   *
   * Refusal is routine rather than exceptional: an embedded or
   * automated document may not be allowed the lock at all, and a user
   * can simply have denied it. The mode is built to work either way, so
   * the rejection is swallowed — left alone it surfaces as an
   * unhandled promise, which reads like a fault and is not one. Older
   * browsers return nothing here, hence the optional chaining.
   */
  _requestLock() {
    try {
      this.ctx.canvas.requestPointerLock?.()?.catch?.(() => {});
    } catch { /* synchronous refusal, same non-event */ }
  }

  releaseKeys() { this.cat?.releaseKeys(); }

  resize(width, height) {
    this.width = width;
    this.height = height;
    this._applyScale(this.scale || 0.75, true);
  }

  _applyScale(scale, force = false) {
    if (!force && scale === this.scale) return;
    this.scale = scale;
    const w = Math.max(2, Math.round(this.width * scale));
    const h = Math.max(2, Math.round(this.height * scale));
    this.rt.resize(w, h);
    // The triangles are composited per-texel against the march, so they
    // have to be rendered at exactly the march's resolution — not the
    // canvas's.
    this.meshRT.resize(w, h);
    this.history.resize(w, h);
    this.history.clear(0, 0, 0, 1);
  }

  reset() {
    this.yaw = 0.85;
    this.pitch = 0.22;
    this.targetDist = 4.6;
    this.center.set([0, 0, 0]);
    this.target.set([0, 0.1, 0]);
    this.camMode = 'orbit';
    this.cat?.reset();
    this.laser.silence();
    this.history.clear(0, 0, 0, 1);
    this._rippleAge.fill(0);
    this._rippleHost.fill(-2);
    this.flash = 0;
    this.bursts = 0;
    this.erodeMax = 0;
    this.shots = 0;
    this._pendingShot = false;
    this._aimShot = null;
    this._fireCooldown = 0;
    // The app has just put every slider back, including the masters, so
    // forget where they last fired or they will stamp over them next frame.
    this._lastQuality = null;
    this._lastVis = null;
  }

  /* ── camera ───────────────────────────────────────────────────── */

  _updateCamera(state, clock, pointer) {
    // Following something that is not being drawn is just a camera stuck
    // in a corner, so the mode needs the cat to actually be there.
    const follow = state.camera === 'follow' && Boolean(this.cat) && Boolean(state.cat);
    // In mouse-look the pointer already has a job. Letting it orbit as
    // well would fight the steering for the same movement.
    const looking = this._showCat && this.cat?.mode === 'look';
    let moved = false;

    if (looking) {
      moved = true;
    } else if (pointer.down && pointer.moved) {
      this.yaw -= pointer.dx * 3.6;
      this.pitch = clamp(this.pitch + pointer.dy * 2.4, -0.35, 1.25);
      moved = true;
    } else if (state.spin && !follow) {
      // Orbiting a cat you are trying to steer is motion sickness, not a
      // feature; the auto-spin only belongs to the locked mode.
      this.yaw += clock.dt * 0.075;
      moved = clock.dt > 0;
    }

    /* ── where to orbit, and where to look ──
       Both modes are the same spherical rig; all that differs is the
       point at its centre and whether the yaw has a mind of its own.
       Centre and aim are not the same point: the locked view orbits the
       origin but aims slightly above it, which is what puts the cluster
       on the upper third of the frame instead of dead centre. */
    let cx = 0, cy0 = 0, cz = 0;
    let ax = 0, ay = 0.1, az = 0;

    if (follow) {
      const cat = this.cat;
      /* Chest height, so the camera looks at the animal and not at the
         floor it is standing on.
         Measured from wherever the animal actually is, not from a
         constant and not from the ground: on a slope the two differ by
         metres and the camera would aim at the hillside beside it, and
         out in the lake the ground is not what the cat is standing on
         at all. */
      const chest = cat.rideY + cat.header.bounds.max[1] * cat.scale * 0.55;
      cx = ax = cat.x;
      cy0 = ay = chest;
      cz = az = cat.z;

      /* Whether the camera chases the cat's heading depends entirely on
         who is steering, and getting this wrong breaks the other mode.

         Under mouse-look the mouse *is* the camera, so it is pinned to
         directly behind: any lag there reads as the view sticking.

         Under camera-relative keys it must NOT chase. WASD names a
         direction in this camera's frame, so a camera that swings to
         follow the cat also swings the meaning of the keys — hold S and
         the cat turns to face you, the camera comes round behind it,
         "toward you" now points somewhere else, and it turns again. It
         spins on the spot forever. The camera tracks position only, and
         its heading stays the user's to set by dragging. */
      if (looking) {
        const want = cat.yaw + Math.PI;
        // Shortest way round, or the camera takes the long path through
        // a full turn every time the yaw crosses ±π.
        this.yaw += (want - this.yaw + Math.PI * 3) % (Math.PI * 2) - Math.PI;
        moved = true;
      }
      if (cat.speed > 1e-3) moved = true;
    }

    // Ease both rather than jumping: at speed the cat covers several
    // units a second, and a rigid target turns every change of direction
    // into a snap. On the first follow frame there is nothing to ease
    // from, so the rig is placed outright instead of flying across the
    // scene from wherever the locked view left it.
    const snap = follow !== (this.camMode === 'follow');
    this.camMode = follow ? 'follow' : 'orbit';
    const k = snap ? 1 : 1 - Math.exp(-clock.wallDt * (follow ? 6 : 4));

    this.center[0] += (cx - this.center[0]) * k;
    this.center[1] += (cy0 - this.center[1]) * k;
    this.center[2] += (cz - this.center[2]) * k;
    this._orbitXZ[0] = this.center[0];
    this._orbitXZ[1] = this.center[2];
    this.target[0] += (ax - this.target[0]) * k;
    this.target[1] += (ay - this.target[1]) * k;
    this.target[2] += (az - this.target[2]) * k;

    const prevDist = this.dist;
    this.dist += (this.targetDist - this.dist) * (1 - Math.exp(-clock.wallDt * 8));
    if (Math.abs(prevDist - this.dist) > 1e-4) moved = true;
    /* Both are settled here and nowhere else: this runs once a frame and
       everything that adds to them runs after it. The camera's own
       motion belongs to both halves — it moves every pixel there is. */
    this.moving = moved ? 1 : 0;
    this.movingMesh = 0;

    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const cyw = Math.cos(this.yaw), syw = Math.sin(this.yaw);
    const { pos, right, up, fwd } = this.basis;

    pos[0] = this.center[0] + this.dist * cp * syw;
    pos[1] = this.center[1] + this.dist * sp + 0.35;
    pos[2] = this.center[2] + this.dist * cp * cyw;

    /* Never let the camera drop through the ground while chasing a cat
       that is, by definition, standing on it. Against the hill under the
       *camera*, not under the cat: swinging round a rising slope is
       exactly when the two are furthest apart, and it is the ground the
       lens is about to enter that matters. */
    if (follow) {
      const floor = terrainHeight(pos[0], pos[2], state.hills) + 0.3;
      if (pos[1] < floor) pos[1] = floor;
    }

    let fx = this.target[0] - pos[0], fy = this.target[1] - pos[1], fz = this.target[2] - pos[2];
    const fl = Math.hypot(fx, fy, fz) || 1;
    fx /= fl; fy /= fl; fz /= fl;
    fwd[0] = fx; fwd[1] = fy; fwd[2] = fz;

    // right = normalize(cross(fwd, worldUp)). Note the signs: the
    // opposite order gives cross(worldUp, fwd), which negates `right`,
    // which flips `up` through the cross product below — and renders
    // the whole scene upside down with the floor grid in the sky.
    let rx = -fz, ry = 0, rz = fx;
    const rl = Math.hypot(rx, ry, rz) || 1;
    rx /= rl; rz /= rl;
    right[0] = rx; right[1] = ry; right[2] = rz;

    // up = cross(right, fwd)
    up[0] = ry * fz - rz * fy;
    up[1] = rz * fx - rx * fz;
    up[2] = rx * fy - ry * fx;
  }

  /**
   * Hand this frame's mouse movement to the cat, in radians.
   *
   * Two sources for the same gesture. Locked, the browser reports raw
   * device movement in pixels and the cursor never runs out of room.
   * Unlocked, the app's own pointer deltas are already accumulated per
   * frame — they are in fractions of the canvas, so a sweep across it is
   * about half a turn, and that is as far as one sweep can go.
   */
  _feedLook(pointer) {
    if (this.cat.mode !== 'look') return;

    if (this.locked) {
      this.cat.look(this._lookDx * LOOK_PER_PIXEL, this._lookDy * LOOK_PER_PIXEL);
      this._lookDx = this._lookDy = 0;
    } else {
      this.cat.look(pointer.dx * LOOK_PER_SWEEP, pointer.dy * LOOK_PER_SWEEP);
    }

    // Pitch belongs to the camera, not the animal. Same sign as the
    // existing orbit drag, so pushing the mouse down raises the camera
    // whichever way you are steering.
    this.pitch = clamp(this.pitch + this.cat.takeLookPitch() * PITCH_FROM_LOOK, -0.35, 1.25);
  }

  /** The ray under the pointer, in world space. Matches the shader. */
  _pointerRay(pointer, out) {
    const focal = 1.5;
    const aspect = this.width / Math.max(this.height, 1);
    // With the cursor captured there is no cursor to aim with: it stops
    // reporting a position the moment the lock takes hold. Centre of
    // frame is the only honest answer, and it is what a crosshair would
    // do anyway — so clicking still strikes the surface in mouse-look.
    const locked = this.locked;
    const ndcX = locked ? 0 : pointer.x * 2 - 1;
    const ndcY = locked ? 0 : 1 - pointer.y * 2;
    const { right, up, fwd } = this.basis;

    let x = fwd[0] + right[0] * ndcX * aspect / focal + up[0] * ndcY / focal;
    let y = fwd[1] + right[1] * ndcX * aspect / focal + up[1] * ndcY / focal;
    let z = fwd[2] + right[2] * ndcX * aspect / focal + up[2] * ndcY / focal;
    const l = Math.hypot(x, y, z) || 1;
    out[0] = x / l; out[1] = y / l; out[2] = z / l;
    return out;
  }

  /* ── the cluster, on the CPU ──────────────────────────────────── */

  /**
   * The orbits used to be derived from uTime inside the shader. They are
   * computed here now and uploaded, because picking needs the CPU to
   * agree with the GPU about where every sphere is — and two copies of
   * the same formula is exactly the kind of thing that silently drifts.
   *
   * The bounding sphere falls out of the same loop, and every distance
   * limit in the shader is derived from it: where to start marching,
   * where to give up, how far a shadow ray can still matter. Change the
   * scene's scale and nothing needs re-tuning.
   */
  _updateBalls(state, time) {
    const t = time * 0.42;
    const n = Math.round(state.balls);
    this.ballCount = n;
    this.blend = state.blend;
    this.time = time;

    // The ring is fixed in extent; the spheres are not.
    let reach = RING_MAJOR + RING_MINOR;

    for (let i = 0; i < n; i++) {
      const a = i * 2.399963;                       // golden angle
      const o = i * 4;
      const x = Math.sin(t * (0.7 + i * 0.11) + a) * (0.62 + 0.1 * Math.sin(i));
      const y = Math.cos(t * (0.5 + i * 0.09) + a * 1.7) * 0.5;
      const z = Math.cos(t * (0.62 + i * 0.13) + a * 0.6) * (0.62 + 0.1 * Math.cos(i));
      const r = 0.30 + 0.10 * Math.sin(t * 0.9 + i * 2.1);
      this.ballPos[o + 0] = x;
      this.ballPos[o + 1] = y;
      this.ballPos[o + 2] = z;
      this.ballPos[o + 3] = r;
      reach = Math.max(reach, Math.hypot(x, y, z) + r);
    }
    for (let i = n; i < BALL_N; i++) this.ballPos[i * 4 + 3] = 0;

    // Slack for everything that pushes the surface outward past the raw
    // union: the smooth minimum's bulge, the displacement noise, and any
    // ripple currently travelling.
    this.bound[0] = 0;
    this.bound[1] = 0;
    this.bound[2] = 0;
    this.bound[3] = reach
      + state.blend
      + state.displace * DISPLACE_AMP
      + state.rippleAmp
      + 0.02;
  }

  /* The ring tumbles; these move a point in and out of its frame.
     `q.yz = rot2(a) * q.yz` with GLSL's column-major mat2(c,-s,s,c)
     expands to (c·y + s·z, −s·y + c·z), which is what these mirror. */

  _worldToRing(x, y, z, out) {
    const t = this.time * 0.42;
    const ca = Math.cos(t * 0.31), sa = Math.sin(t * 0.31);
    const y1 = ca * y + sa * z;
    const z1 = -sa * y + ca * z;
    const cb = Math.cos(t * 0.19), sb = Math.sin(t * 0.19);
    out[0] = cb * x + sb * z1;
    out[1] = y1;
    out[2] = -sb * x + cb * z1;
    return out;
  }

  _ringToWorld(x, y, z, out) {
    const t = this.time * 0.42;
    const cb = Math.cos(t * 0.19), sb = Math.sin(t * 0.19);
    const x1 = cb * x - sb * z;
    const z1 = sb * x + cb * z;
    const ca = Math.cos(t * 0.31), sa = Math.sin(t * 0.31);
    out[0] = x1;
    out[1] = ca * y - sa * z1;
    out[2] = sa * y + ca * z1;
    return out;
  }

  _ringDistance(x, y, z) {
    const q = this._worldToRing(x, y, z, this._tmpA ??= new Float32Array(3));
    const radial = Math.hypot(q[0], q[2]) - RING_MAJOR;
    return Math.hypot(radial, q[1]) - RING_MINOR;
  }

  /** The same field the shader marches, minus the terms picking can ignore. */
  _mapCPU(x, y, z) {
    let d = 1e9;
    for (let i = 0; i < this.ballCount; i++) {
      const o = i * 4;
      const s = Math.hypot(x - this.ballPos[o], y - this.ballPos[o + 1], z - this.ballPos[o + 2])
        - this.ballPos[o + 3];
      d = smin(d, s, this.blend);
    }
    return smin(d, this._ringDistance(x, y, z), this.blend * 0.6);
  }

  /**
   * March the pointer's ray on the CPU. The surface displacement and the
   * ripples are left out: they move the surface by at most a couple of
   * centimetres, which is far below the accuracy a click needs, and
   * including them would make every pick pay for a noise function.
   */
  _pick(pointer) {
    const ro = this.basis.pos;
    const rd = this._pointerRay(pointer, this._ray);

    let t = 0.05;
    let hit = false;
    for (let i = 0; i < 128; i++) {
      const d = this._mapCPU(ro[0] + rd[0] * t, ro[1] + rd[1] * t, ro[2] + rd[2] * t);
      if (d < 0.0015) { hit = true; break; }
      t += d * 0.9;
      if (t > 30) break;
    }
    if (!hit) return null;

    this._hit[0] = ro[0] + rd[0] * t;
    this._hit[1] = ro[1] + rd[1] * t;
    this._hit[2] = ro[2] + rd[2] * t;
    return this._hit;
  }

  /* ── impacts ──────────────────────────────────────────────────── */

  /**
   * A ripple has to be anchored to whatever it landed on, because
   * everything here is moving. For a sphere that means storing a
   * *direction* and re-placing the centre on the surface every frame —
   * the spheres pulse, so a fixed offset would sink into one or float
   * off another. For the ring it means storing the point in ring-local
   * coordinates and rotating it back.
   */
  _anchorRipple(px, py, pz) {
    let best = this._ringDistance(px, py, pz);
    let host = -1;

    for (let i = 0; i < this.ballCount; i++) {
      const o = i * 4;
      const d = Math.abs(
        Math.hypot(px - this.ballPos[o], py - this.ballPos[o + 1], pz - this.ballPos[o + 2])
        - this.ballPos[o + 3]);
      if (d < Math.abs(best)) { best = d; host = i; }
    }

    const i = this._rippleNext;
    this._rippleNext = (this._rippleNext + 1) % RIPPLE_N;
    this._rippleHost[i] = host;
    this._rippleAge[i] = 1e-4;
    // A click is a segment of zero length, which the shader handles with
    // the same code a beam uses.
    this._rippleLen[i] = 0;

    const l = i * 3;
    if (host >= 0) {
      const o = host * 4;
      let dx = px - this.ballPos[o];
      let dy = py - this.ballPos[o + 1];
      let dz = pz - this.ballPos[o + 2];
      const dl = Math.hypot(dx, dy, dz) || 1;
      this._rippleLocal[l + 0] = dx / dl;
      this._rippleLocal[l + 1] = dy / dl;
      this._rippleLocal[l + 2] = dz / dl;
    } else {
      this._worldToRing(px, py, pz, this._tmpB ??= new Float32Array(3));
      this._rippleLocal[l + 0] = this._tmpB[0];
      this._rippleLocal[l + 1] = this._tmpB[1];
      this._rippleLocal[l + 2] = this._tmpB[2];
    }
  }

  /**
   * Record a bore: an impact that is a line rather than a point.
   *
   * Pinned in world space, unlike a click's. A click belongs to the
   * sphere it landed on and has to ride it as it orbits; a bore runs
   * through whatever happened to be in the way, so no one body owns it
   * and there is nothing to ride.
   */
  _anchorBore(ax, ay, az, bx, by, bz) {
    const i = this._rippleNext;
    this._rippleNext = (this._rippleNext + 1) % RIPPLE_N;

    this._rippleHost[i] = HOST_BORE;
    this._rippleAge[i] = 1e-4;
    this._rippleLen[i] = Math.hypot(bx - ax, by - ay, bz - az);

    const l = i * 3;
    this._rippleLocal[l] = ax; this._rippleLocal[l + 1] = ay; this._rippleLocal[l + 2] = az;
    this._boreEnd[l] = bx; this._boreEnd[l + 1] = by; this._boreEnd[l + 2] = bz;
  }

  _burst(state, pointer) {
    const p = this._pick(pointer);
    if (!p) return false;
    this._anchorRipple(p[0], p[1], p[2]);
    this.flash = 1;
    this.bursts++;
    return true;
  }

  /* ── eye beams ──────────────────────────────────────────────────── */

  /**
   * How far a beam carries.
   *
   * It is not stopped by the cluster: this beam bores, so it goes
   * through and out the far side, and the channel it opens is what lets
   * it be seen doing so. The only thing that ends it is the ground —
   * which used to be a plane and needed no searching, and is now a height
   * field marched by the JS half of terrain.js. The same conservative
   * step the shader takes, so the beam stops on the hillside the eye can
   * see it striking.
   */
  _beamReach(ox, oy, oz, dir) {
    const t = traceTerrainJS(ox, oy, oz, dir[0], dir[1], dir[2],
                             BEAM_REACH, this.hills);
    return t > 0 ? t : BEAM_REACH;
  }

  /**
   * Fire, from the eyes, down the crosshair.
   *
   * The aim is the camera's forward vector because in mouse-look that
   * *is* where the mouse points: the pointer is captured, there is no
   * cursor to read, and the mode pins the camera behind the animal so
   * the two agree. The same reasoning already decides where a click
   * strikes the surface.
   */
  /**
   * Pull the trigger.
   *
   * Where the shot goes, and whether it goes at once, both depend on
   * whether the pointer is captured.
   *
   * Captured, the crosshair is the aim and the camera's forward vector
   * is the crosshair — nothing to look up, nothing to wait for, and
   * always somewhere to shoot. Free, the aim is whatever the cursor is
   * over: the cat has to bring its head round to it first, and a click
   * on nothing does not fire at all.
   *
   * That delay is not a cost to be minimised — a beam that leaves before
   * the animal has looked reads as a mis-aim.
   */
  _trigger(pointer) {
    if (!this._catView) return false;

    // No cursor to read: fire down the crosshair, now.
    if (this.locked || !pointer) return this._shootAlong(this.basis.fwd);

    /* Nothing under the cursor is not a shot. With a free cursor the
       click *is* the aim, so a click on empty sky has not aimed at
       anything — firing into it would spend a shot on a slip of the
       hand. The captured-pointer path has no such case: there the
       crosshair always points somewhere. */
    const hit = this._pick(pointer);
    if (!hit) return false;

    /* The body's share is settled here, once, not renegotiated every
       frame. Recomputing it while easing by "the time that is left"
       makes the body chase a quarter of the *remaining* angle over and
       over, and it arrives having turned far more than its share. */
    const wrap = (a) => (a + Math.PI * 3) % (Math.PI * 2) - Math.PI;
    const cat = this.cat;
    const need = wrap(Math.atan2(hit[0] - cat.x, hit[2] - cat.z) - cat.yaw);
    const beyondNeck = need - Math.max(-HEAD_YAW_MAX, Math.min(HEAD_YAW_MAX, need));

    /* The wait is the turn, not a toll. A cat already looking at what
       was clicked has nothing to bring round, and charging it the full
       delay anyway would make held fire stutter at a target it is
       staring straight at. */
    const turn = Math.min(1, Math.abs(need) / HEAD_YAW_MAX);

    this._aimShot = {
      x: hit[0], y: hit[1], z: hit[2], t: AIM_TURN * turn,
      // Where the body will have got to when the head arrives.
      bodyYaw: cat.yaw + beyondNeck + (need - beyondNeck) * BODY_FOLLOW,
    };
    return true;
  }

  /**
   * Bring the head round toward a scheduled shot, and take it on arrival.
   *
   * The turn is spread over exactly the time left rather than run at a
   * fixed rate, so it lands on the target as the delay expires however
   * far it had to come.
   *
   * Runs after the cat's own update so that, if the player is steering
   * at the same time, the aim owns the moment it was given.
   */
  _aimTick(dt) {
    const shot = this._aimShot;
    if (!shot) return;
    if (!this._catView) { this._aimShot = null; return; }

    const cat = this.cat;
    const wrap = (a) => (a + Math.PI * 3) % (Math.PI * 2) - Math.PI;

    /* Settle the body's share if whoever booked the shot did not. */
    if (shot.bodyYaw === undefined) {
      const need = wrap(Math.atan2(shot.x - cat.x, shot.z - cat.z) - cat.yaw);
      const beyond = need - Math.max(-HEAD_YAW_MAX, Math.min(HEAD_YAW_MAX, need));
      shot.bodyYaw = cat.yaw + beyond + (need - beyond) * BODY_FOLLOW;
    }

    /* The head leads and the body follows, which is how an animal looks
       at something: the neck covers what it can, and the shoulders come
       round only for the part it cannot — plus a little besides, because
       a cat that turned nothing but its head would read as an owl.

       The body eases toward the share fixed when the shot was booked, so
       the fraction of the time left is a fraction of a fixed distance
       and it arrives exactly as the delay runs out. */
    const step = shot.t > dt ? dt / shot.t : 1;
    const yaw = cat.yaw + wrap(shot.bodyYaw - cat.yaw) * step;
    cat.faceTowards(Math.sin(yaw), Math.cos(yaw));

    // Whatever the body did not take, the neck does.
    this._aimHead(shot.x, shot.y, shot.z);

    shot.t -= dt;
    if (shot.t > 0) return;
    this._aimShot = null;

    // Aim from where the eyes ended up, not from where they started.
    this.cat.eyeWorld(0, this._eyeA);
    this.cat.eyeWorld(1, this._eyeB);
    const aim = this._aim;
    aim[0] = shot.x - (this._eyeA[0] + this._eyeB[0]) * 0.5;
    aim[1] = shot.y - (this._eyeA[1] + this._eyeB[1]) * 0.5;
    aim[2] = shot.z - (this._eyeA[2] + this._eyeB[2]) * 0.5;
    const l = Math.hypot(aim[0], aim[1], aim[2]) || 1;
    aim[0] /= l; aim[1] /= l; aim[2] /= l;
    this._shootAlong(aim);
  }

  /**
   * Turn the head onto a world point, leaving the body where it is.
   *
   * Pitch is taken from the line the beam will actually travel rather
   * than from the target's height alone, so a shot fired from a crouched
   * cat at something just above it still points the nose up the beam.
   */
  _aimHead(tx, ty, tz) {
    const cat = this.cat;
    cat.eyeWorld(0, this._eyeA);
    cat.eyeWorld(1, this._eyeB);

    const dx = tx - (this._eyeA[0] + this._eyeB[0]) * 0.5;
    const dy = ty - (this._eyeA[1] + this._eyeB[1]) * 0.5;
    const dz = tz - (this._eyeA[2] + this._eyeB[2]) * 0.5;
    this._aimAlong(dx, dy, dz);
  }

  /** The same, from a direction rather than a point. */
  _aimAlong(dx, dy, dz) {
    const cat = this.cat;
    const l = Math.hypot(dx, dy, dz) || 1;
    const wrap = (a) => (a + Math.PI * 3) % (Math.PI * 2) - Math.PI;

    const yaw = wrap(Math.atan2(dx, dz) - cat.yaw);
    // The head bone's positive X is nose-down, so looking up is negative.
    const pitch = -Math.asin(Math.max(-1, Math.min(1, dy / l)));
    cat.setAim(yaw, pitch);
  }

  /** Fire, from wherever the eyes are now, along a unit direction. */
  _shootAlong(dir) {
    if (!this._catView) return false;

    this.cat.eyeWorld(0, this._eyeA);
    this.cat.eyeWorld(1, this._eyeB);

    const aim = this._aim;
    if (aim !== dir) aim.set(dir);

    // Look down the barrel: the head turns to whatever was just fired
    // along, in either mode, and lets go of it a moment later.
    this._aimAlong(aim[0], aim[1], aim[2]);

    const mid = [(this._eyeA[0] + this._eyeB[0]) * 0.5,
                 (this._eyeA[1] + this._eyeB[1]) * 0.5,
                 (this._eyeA[2] + this._eyeB[2]) * 0.5];
    this.laser.fire(this._eyeA, this._eyeB, aim,
      this._beamReach(mid[0], mid[1], mid[2], aim));
    this.shots++;
    this.flash = 1;
    this._bore(mid, aim);
    return true;
  }

  /**
   * What the shot does to the cluster: it bores through it.
   *
   * A click leaves a point impact and the wave comes off that point as a
   * sphere. A beam leaves a *segment* — everything it passed through —
   * and the same wave comes off it as a cylinder, so what opens is a
   * channel the length of the crossing rather than a crater where it
   * happened to land first.
   *
   * The segment is the beam clipped to the cluster's bounding sphere,
   * which is the only stretch of it that can be inside anything. It is
   * pinned in world space rather than anchored to a sphere the way a
   * click is: the bore runs through whatever was in the way, and no one
   * body owns it.
   */
  _bore(origin, dir) {
    const cx = origin[0] - this.bound[0];
    const cy = origin[1] - this.bound[1];
    const cz = origin[2] - this.bound[2];

    const b = cx * dir[0] + cy * dir[1] + cz * dir[2];
    const c = cx * cx + cy * cy + cz * cz - this.bound[3] * this.bound[3];
    const h = b * b - c;
    if (h < 0) return false;                 // the shot missed entirely

    const root = Math.sqrt(h);
    const enter = Math.max(-b - root, 0);    // never behind the eyes
    const leave = -b + root;
    if (leave <= enter) return false;

    this._anchorBore(
      origin[0] + dir[0] * enter, origin[1] + dir[1] * enter, origin[2] + dir[2] * enter,
      origin[0] + dir[0] * leave, origin[1] + dir[1] * leave, origin[2] + dir[2] * leave,
    );
    return true;
  }

  _updateRipples(state, dt) {
    let active = 0;
    this.erodeMax = 0;
    const out = this._tmpC ??= new Float32Array(3);

    for (let i = 0; i < RIPPLE_N; i++) {
      const age = this._rippleAge[i];
      if (age <= 0) {
        this.ripples[i * 4 + 3] = 0;
        this.flareAmt[i] = 0;
        this.flarePts[i * 4 + 3] = 0;
        continue;
      }

      const next = age + dt / state.rippleLife;
      const host = this._rippleHost[i];
      if (next >= 1 || (host >= 0 && host >= this.ballCount)) {
        this._rippleAge[i] = 0;
        this.ripples[i * 4 + 3] = 0;
        this.flareAmt[i] = 0;
        this.flarePts[i * 4 + 3] = 0;
        continue;
      }
      this._rippleAge[i] = next;

      const l = i * 3, q = i * 4;
      if (host === HOST_BORE) {
        // Both ends already in world space, and nothing to ride.
        this.ripples[q + 0] = this._rippleLocal[l + 0];
        this.ripples[q + 1] = this._rippleLocal[l + 1];
        this.ripples[q + 2] = this._rippleLocal[l + 2];
        this.rippleTo[q + 0] = this._boreEnd[l + 0];
        this.rippleTo[q + 1] = this._boreEnd[l + 1];
        this.rippleTo[q + 2] = this._boreEnd[l + 2];
        this.rippleTo[q + 3] = BORE_ERODE;
        this.erodeMax = Math.max(this.erodeMax, BORE_ERODE);
      } else {
        if (host >= 0) {
          const o = host * 4;
          const r = this.ballPos[o + 3];
          this.ripples[q + 0] = this.ballPos[o + 0] + this._rippleLocal[l + 0] * r;
          this.ripples[q + 1] = this.ballPos[o + 1] + this._rippleLocal[l + 1] * r;
          this.ripples[q + 2] = this.ballPos[o + 2] + this._rippleLocal[l + 2] * r;
        } else {
          this._ringToWorld(this._rippleLocal[l], this._rippleLocal[l + 1], this._rippleLocal[l + 2], out);
          this.ripples[q + 0] = out[0];
          this.ripples[q + 1] = out[1];
          this.ripples[q + 2] = out[2];
        }
        // A point impact is a segment with both ends in the same place.
        this.rippleTo[q + 0] = this.ripples[q + 0];
        this.rippleTo[q + 1] = this.ripples[q + 1];
        this.rippleTo[q + 2] = this.ripples[q + 2];
        this.rippleTo[q + 3] = 1;
        this.erodeMax = Math.max(this.erodeMax, 1);
      }
      this.ripples[q + 3] = next;

      /* The flare grows briefly, then goes out well before the ripple
         does. A bore has no single point to flare at, so it lights the
         middle of what it crossed. */
      this.flarePts[q + 0] = (this.ripples[q + 0] + this.rippleTo[q + 0]) * 0.5;
      this.flarePts[q + 1] = (this.ripples[q + 1] + this.rippleTo[q + 1]) * 0.5;
      this.flarePts[q + 2] = (this.ripples[q + 2] + this.rippleTo[q + 2]) * 0.5;
      this.flarePts[q + 3] = 0.10 + next * 0.35 + this._rippleLen[i] * 0.25;
      this.flareAmt[i] = Math.max(0, 1 - next * 3.2) * state.flash;

      active++;
    }
    return active;
  }

  /* ── quality ──────────────────────────────────────────────────── */

  /**
   * The master is a one-shot action, not a binding. It fires only when
   * its own value changes, and never on the first frame — so a URL that
   * carries both a quality value and an individual override keeps the
   * override instead of having it stamped over at load.
   *
   * `notify: false` keeps the drag from writing history sixty times a
   * second; the master's own commit, on release, syncs the URL once —
   * and by then the derived values are already in the panel's state, so
   * they land in the URL with it.
   */
  /**
   * The visibility master.
   *
   * Fog was a constant, and the constant put the horizon at 66 units,
   * which is what made the world feel small. It is a control now, and it
   * carries the cover's reach along with it — the two are not the same
   * quantity, but they have a one-way constraint: reach shorter than the
   * fog's own distance means you can see the edge of the grass.
   *
   * Written into the reach slider rather than read from here, exactly as
   * the quality master writes into its seven. The reach stays yours
   * afterwards, and this simply goes stale, which is the point: a
   * deliberately hazy scene with a long reach is a legitimate thing to
   * want and a locked pair would forbid it.
   */
  _applyVisibility(state) {
    if (this._lastVis === null) { this._lastVis = state.visibility; return; }
    if (state.visibility === this._lastVis) return;
    this._lastVis = state.visibility;
    this.ctx.setParams({
      coverRadius: Math.round(clamp(state.visibility * 0.9, 8, 200)),
    });
  }

  _applyQuality(state) {
    if (this._lastQuality === null) { this._lastQuality = state.quality; return; }
    if (state.quality === this._lastQuality) return;
    this._lastQuality = state.quality;
    this.ctx.setParams(qualityPreset(state.quality));
  }

  /* ── frame ────────────────────────────────────────────────────── */

  frame({ state, clock, pointer }) {
    const { gl, tri, empty } = this.ctx;
    this._applyQuality(state);
    this._applyVisibility(state);
    this._applyScale(Number(state.scale));

    /* Fog thick enough that the requested distance is where it closes.
       The term is 1 - exp(-t·uFog·0.045); putting 95% of it at t = V
       means uFog·0.045·V = 3. */
    const fog = 3 / (0.045 * Math.max(state.visibility, 1));

    const dt = Math.min(clock.dt, 1 / 30);

    /* How high the ground stands, kept on the scene rather than passed
       down. The pointer and key handlers fire a beam that has to stop
       somewhere, and they have no state to read — the same reason
       `_catView` and `_showCat` are stashed here. */
    this.hills = state.hills;

    /* ── the cat ──
       Stepped before the camera, not after: a follow shot that reads
       last frame's position lags the animal it is following by exactly
       the amount that makes it feel loose. */
    const showCat = Boolean(this.cat && state.cat);
    this._showCat = showCat;                    // read by onKey, which has no state

    /* Grass is triangles, not field, so the marcher only has to know
       that something is growing on its floor — which changes what the
       floor is made of and nothing else about the march. */
    const covered = isCovered(state.ground);
    const wooded = Boolean(state.trees);
    /* Butterflies are geometry, so they are one more reason the raster
       target has to be drawn and composited at all — turn every other
       thing in it off and a meadow with butterflies still needs it. */
    const flying = state.butterflies > 0
      || (state.sparrows > 0 && state.sparrowFlocks > 0);
    const raster = showCat || covered || wooded || flying;
    /* The lake, resolved once. Three controls collapse into the one
       number every pass is handed, for the same reason `hills` is: four
       passes that each worked out their own water level would each be
       standing on a different shore. */
    const waterY = waterSurfaceY(state.water, state.waterLevel, state.hills);
    /* And the weather, resolved to the two numbers anything downstream
       actually reads. Past this line the name of the mode is gone. */
    const weather = weatherOf(state.weather);
    /* The weapon belongs to the cat's own view. With the camera locked
       on the cluster you are watching it, not being it, and a click
       there means what it always meant: strike the surface. Read by the
       pointer handlers, which have no state either. */
    this._catView = showCat && state.camera === 'follow';
    let reskinned = false;
    if (showCat) {
      /* Changing the colourway invalidates the accumulated history the
         same way moving does — the pixels change without the cat having.

         Held until after the camera has been updated rather than set
         here, because that is where both of these are settled for the
         frame: written now, it was overwritten a dozen lines later and
         the swap smeared for a few frames every time. */
      reskinned = this.cat.setSkin(state.skin);
      this._feedLook(pointer);
      // Last frame's basis, deliberately: the camera has not been moved
      // yet this frame, and steering off the picture the user is
      // actually looking at is both correct and one less feedback path.
      this.cat.update(dt, state.hills, this.basis, waterY);
    } else if (this.cat) {
      this.cat.releaseKeys();                   // or it resumes mid-stride
    }

    this._updateCamera(state, clock, pointer);

    /* A cat that moved invalidates the accumulated history, exactly like
       a travelling ripple does — and "moved" includes breathing and
       blinking, not just walking.

       Which half it invalidates is the question, and the animal answers
       it twice. Where it *is* changes the marched half too, because the
       shadow it throws lands there; what it is doing with its ribs and
       its tail changes only the triangles it is made of. So a walking cat
       stops the whole frame accumulating and a breathing one stops only
       itself — the capsules do drift a hair as it breathes, and that
       hair is a soft shadow's worth of ghosting against a whole frame of
       aliasing. */
    if (reskinned) this.movingMesh = 1;
    if (showCat && this.cat.animating) this.movingMesh = 1;
    if (showCat && (this.cat.speed !== 0 || this.cat.turnRate !== 0
                    || this.cat.swim > 0.01)) this.moving = 1;

    /* The light, from whichever control the mode put in charge.

       The hour is advanced here rather than read off the clock, because
       the clock can be paused and stepped a frame at a time and the sun
       has to do the same — a scene that keeps turning while it is frozen
       is a scene whose reference shot is not reproducible. */
    const timed = hourDrives(state.daylight);
    if (state.daylight === 'cycle') {
      this.hour = advanceHour(this.hour, clock.dt);
      /* Written back into the slider, so the control reads the time it
         is producing rather than sitting at whatever it was left at.
         Throttled: at sixty frames a second a slider redrawn every frame
         is a slider nobody can grab, and the hour moves 0.18 of an hour
         in the quarter second between writes. */
      if (clock.time - this._hourShown > 0.25) {
        this._hourShown = clock.time;
        this.ctx.setParams({ hour: Math.round(this.hour * 20) / 20 });
      }
    } else if (timed) {
      this.hour = state.hour;
    }

    let tint = TINTS[state.tint] || TINTS.amber;
    let day = 1;
    let ambient = AMBIENT_FIXED;

    if (timed) {
      const sky = skyAt(this.hour);
      this.lightDir.set(sky.dir);
      day = sky.day;
      ambient = sky.ambient;
      /* The tint control still says which hue the light leans; the hour
         says how strong it is and how far toward the horizon its colour
         has been pushed. Multiplied rather than replaced, so "amber at
         dawn" and "ice at dawn" stay two different scenes — and so that
         every surface already multiplying by uTint gets a time of day
         without a single shader learning that there is one. */
      tint = [
        tint[0] * sky.tint[0],
        tint[1] * sky.tint[1],
        tint[2] * sky.tint[2],
      ];
    } else {
      // XY pad → hemisphere direction.
      const az = (state.light[0] - 0.5) * Math.PI * 2.2;
      const el = (1 - state.light[1]) * 1.35 + 0.05;
      this.lightDir[0] = Math.cos(el) * Math.sin(az);
      this.lightDir[1] = Math.sin(el);
      this.lightDir[2] = Math.cos(el) * Math.cos(az);
    }

    this._updateBalls(state, clock.time);

    if (pointer.down) {
      this._dragDist += Math.abs(pointer.dx) + Math.abs(pointer.dy);
    }
    if (this._pendingBurst) {
      this._pendingBurst = false;
      this._burst(state, pointer);
    }
    /* Fired after the spheres are placed for this frame, so the blast is
       measured against where they actually are rather than where they
       were when the button went down.

       In mouse-look the mouse has no other job, so holding the button
       keeps firing on a cooldown — a beam weapon that needs one click
       per shot is a beam weapon nobody uses twice. Everywhere else the
       mouse still orbits, so a shot is one per click and is armed on
       release, where a drag can already be told from a click. */
    this._fireCooldown -= dt;
    const holding = this._pressed && this._catView && this.cat.mode === 'look';
    if (holding && this._fireCooldown <= 0) {
      this._trigger(pointer);
      this._fireCooldown = FIRE_INTERVAL;
    }
    if (this._pendingShot) {
      this._pendingShot = false;
      this._trigger(pointer);
    }
    this._aimTick(dt);
    this.laser.update(dt);
    if (this.laser.active) this.moving = 1;

    const rippleActive = this._updateRipples(state, dt);
    this.flash *= Math.exp(-dt * 4.5);

    // A travelling ripple invalidates the accumulated history, so the
    // temporal filter has to be told to let go of it.
    if (rippleActive) this.moving = 1;

    /* Grass in wind never holds still, so the accumulation buffer must
       not be allowed to believe it does — but only about the grass. The
       blades are triangles and they cast nothing into the marched half,
       so the sky, the cluster, the hills and the lake can go on
       converging while the meadow blows.

       A wood is not the same case and is kept whole: its shadow map is
       read by the floor the marcher draws, so leaves moving in the wind
       move pixels on the marched side of the composite. */
    if (covered && state.wind > 0.001) this.movingMesh = 1;
    if (wooded && state.wind > 0.001) this.moving = 1;
    /* And the lake, which is marched: its normal is a function of the
       wind and the clock, so a windy surface is never twice the same. */
    if (state.water && state.wind > 0.001) this.moving = 1;

    /* The wood is grown and its shadow map drawn before anything is
       shaded, because everything that follows reads it: the floor here,
       the grass and the cat in the next pass, and the leaves themselves.
       It binds its own target, so the march has to bind the scene's
       again afterwards. */
    this.trees.prepare(this.basis.pos, this.lightDir, clock.time, {
      on: wooded,
      density: state.treeDensity,
      radius: state.coverRadius,
      wind: state.wind,
      hills: state.hills,
      waterY,
    });


    /* ── the rasterised half ──
       Its own target, its own depth buffer, the same camera basis. The
       alpha it clears to is the same 1e4 the march writes for sky, so an
       uncovered pixel loses the depth comparison to anything at all.

       The cat and the ground cover go into it together and in either
       order: they are both opaque and they share the depth buffer, which
       is the whole reason the animal stands in the grass instead of on
       a picture of it. */
    if (raster) {
      this.meshRT.bind();
      /* State first, and the depth mask is not optional: a depth clear
         is masked by it exactly as a depth write is, so leaving it false
         from the march below silently keeps last frame's depth buffer
         and this pass starts culling against a frame that is gone. That
         reads as things going missing, not as anything depth-shaped. */
      BLEND.none(gl);
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LESS);
      gl.depthMask(true);
      gl.clearColor(0, 0, 0, 1e4);
      gl.clearDepth(1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      const camera = {
        pos: this.basis.pos,
        right: this.basis.right,
        up: this.basis.up,
        fwd: this.basis.fwd,
        focal: 1.5,
        aspect: this.meshRT.width / Math.max(this.meshRT.height, 1),
        width: this.meshRT.width,
        height: this.meshRT.height,
      };

      /* One description of the light and the field, read by both. Two
         copies is how the cat ends up standing in a shadow the grass
         around it never got. */
      const env = {
        dir: this.lightDir,
        tint,
        /* How much daylight is in the sky. Every pass here draws sky
           somewhere — into its fog, or into a leaf's rim — so every one
           of them has to be told what time it is, or the meadow fogs out
           into a daytime horizon under a night sky. */
        day,
        ambient,
        fog,
        /* The ground itself. Every pass drawn into this target puts
           something on it — the cat's feet, the grass's roots, the
           trees' trunks — and they are only on the same ground because
           they are handed the same number. */
        hills: state.hills,
        /* And where the water on it stands — for the same reason and
           with the same consequence: the grass, the flowers and the
           trees decide whether a seed drowned by asking this. */
        waterY,
        /* And the weather on both, for the same reason a third time. */
        weather,
        shadowSoft: state.shadow,
        // The same budget the marcher spends. A shadow that converges
        // differently is a differently shaped shadow.
        shadowSteps: Math.round(state.shadowSteps),
        shadowNoise: state.shadowNoise ? 1 : 0,
        // The cluster, exactly as the marcher will see it this frame —
        // ripples, dissipation and surface noise included, because both
        // of these march the same layers it does.
        time: clock.time,
        blend: state.blend,
        ballPos: this.ballPos,
        balls: this.ballCount,
        bound: this.bound,
        ripples: this.ripples,
        rippleTo: this.rippleTo,
        rippleOn: rippleActive > 0 ? 1 : 0,
        rippleAmp: state.rippleAmp,
        rippleSpeed: state.rippleSpeed,
        rippleFreq: state.rippleFreq,
        erode: state.erode,
        displace: state.displace,
        // And the cat's own outline, so it throws a shadow on the grass
        // exactly as it throws one on the floor.
        catCapA: showCat ? this.cat.capA : ZERO_CAPS,
        catCapB: showCat ? this.cat.capB : ZERO_CAPS,
        catBound: showCat ? this.cat.capBound : ZERO_BOUND,
        catCaps: showCat ? this.cat.capCount : 0,
        // The wood's shadow, for everything drawn into this target.
        canopy: this.trees.uniforms(),
      };

      const frame = this.frameCount++;
      if (showCat) {
        this._catXZ[0] = this.cat.x;
        this._catXZ[1] = this.cat.z;
        this.cat.draw(camera, env, frame);
      }
      this.ground.draw(camera, env, {
        style: state.ground,
        density: state.cover,
        flowers: Boolean(state.flowers),
        flowerClumps: state.flowerClumps,
        flowerDensity: state.flowerDensity,
        flowerSpread: state.flowerSpread,
        /* The lake's edge. Nothing here has to tell it where that is —
           it reads the same waterY every other planting reads, and asks
           for a band of it instead of the dry half. */
        reeds: Boolean(state.reeds),
        reedDensity: state.reedDensity,
        radius: state.coverRadius,
        wind: state.wind,
        frame,
      });
      /* The same subpixel offset the cat and the grass use. Everything in
         this target is resolved by one temporal filter, so everything in
         it has to be jittered by the same amount or the filter converges
         on one and smears the others. */
      this._treeJitter[0] = Math.sin(frame * 2.39996) / camera.width;
      this._treeJitter[1] = Math.sin(frame * 4.10000 + 1.7) / camera.height;
      this.trees.draw(camera, env, {
        on: wooded,
        wind: state.wind,
        jitter: this._treeJitter,
      });

      /* And whatever is flying over it — last into this target, because
         a butterfly is the smallest thing in it and the depth buffer is
         already carrying everything it has to get behind.

         The trees' jitter, not a third one. Everything drawn here is
         resolved by one temporal filter, and a pass jittered by a
         different amount is a pass the filter never converges on. */
      const flyVerts = this.creatures.drawFlies(camera, env, {
        density: state.butterflies,
        reach: state.lifeRadius,
        wind: state.wind,
        jitter: this._treeJitter,
        /* The same two numbers the sowing was handed, so the flight
           circles the clumps that were actually grown. Passing the
           control rather than a copy is the whole point: move the flower
           density and the butterflies move with it. */
        flowers: Boolean(state.flowers),
        clumpChance: Math.max(0, Math.min(1, state.flowerClumps)),
        clumpSpread: Math.max(0.05, state.flowerSpread),
      });
      /* Each draw follows its own bind immediately. Both creature passes
         end with the programme they are about to use already current, so
         calling them both and then drawing twice would draw the
         butterflies with the sparrow's shader. */
      if (flyVerts) {
        gl.enable(gl.DEPTH_TEST);
        gl.depthFunc(gl.LESS);
        gl.depthMask(true);
        // Both faces of a wing are the wing, exactly as with a blade.
        gl.disable(gl.CULL_FACE);
        empty.drawTriangles(flyVerts);
      }

      const birdVerts = this.creatures.drawBirds(camera, env, {
        density: state.sparrows,
        flocks: state.sparrowFlocks,
        reach: state.lifeRadius,
        /* The pass index is a function of the clock, and without it every
           flock flies one frozen line for ever. */
        time: clock.time,
        wind: state.wind,
        jitter: this._treeJitter,
        /* The wood, so the birds can sit in it. Handed the object rather
           than a list, because what they want is the nearest few crowns
           to wherever the eye is now and only the wood knows that. */
        trees: wooded ? this.trees : null,
        hills: state.hills,
        waterY,
        snow: weather.snow,
        /* Where the cat is, for the one thing it does to a flock. Null
           when there is no cat, which is also "nothing to run from". */
        cat: showCat ? this._catXZ : null,
      });
      if (birdVerts) {
        gl.enable(gl.DEPTH_TEST);
        gl.depthFunc(gl.LESS);
        gl.depthMask(true);
        /* A bird's body is a closed tube, so its far side is behind its
           near side and culling would save nothing — and the wings and
           the tail are single-sided sheets that have to be visible from
           underneath, which is most of where a bird is seen from. */
        gl.disable(gl.CULL_FACE);
        empty.drawTriangles(birdVerts);
      }
    }

    /* The march runs after the raster half, not before it, so that the
       water can read the target the raster half just wrote. Nothing in
       that half reads this one, which is what makes the order free to
       choose; see the water in material().

       The depth state is put back explicitly because the pass above
       leaves it on, and this is a fullscreen triangle that must not be
       depth-tested against whatever the last mesh wrote. */
    gl.disable(gl.DEPTH_TEST);
    gl.depthMask(false);
    BLEND.none(gl);
    this.rt.bind();
    this.march.use({
      uCamPos: this.basis.pos,
      uRight: this.basis.right,
      uUp: this.basis.up,
      uFwd: this.basis.fwd,
      uResolution: [this.rt.width, this.rt.height],
      uFocal: 1.5,
      uTime: clock.time,
      uBlend: state.blend,
      uDisplace: state.displace,
      uRough: state.rough,
      uFloorMix: 0.6,
      uGround: covered ? 1 : 0,
      uHills: state.hills,
      uWaterY: waterY,
      /* The same number the grass, the trees and the falling rain are
         handed. The lake is this shader's only reader of it. */
      uWind: state.wind,
      uRain: weather.rain,
      uSnow: weather.snow,
      // And what the weather does above the scene rather than to it.
      uOvercast: weather.overcast,
      uLightDir: this.lightDir,
      uTint: tint,
      uDay: day,
      uAmbient: ambient,
      uReflect: state.reflect,
      uFog: fog,
      uAO: state.ao,
      uShadowSoft: state.shadow,

      uBallPos: this.ballPos,
      uBalls: this.ballCount,
      uBound: this.bound,

      // The cat, as the shadow and occlusion queries see it.
      uCatCapA: showCat ? this.cat.capA : ZERO_CAPS,
      uCatCapB: showCat ? this.cat.capB : ZERO_CAPS,
      uCatBound: showCat ? this.cat.capBound : ZERO_BOUND,
      uCatCaps: showCat ? this.cat.capCount : 0,
      // And what it is doing to the water, if it is in any.
      uCatWake: showCat ? this.cat.wake : 0,

      ...this.trees.uniforms(),

      uRipples: this.ripples,
      uRippleTo: this.rippleTo,
      uRippleOn: rippleActive > 0 ? 1 : 0,
      uRippleAmp: state.rippleAmp,
      uRippleSpeed: state.rippleSpeed,
      uRippleFreq: state.rippleFreq,
      uRippleTight: 5.0,
      uRippleGlow: state.flash,
      uErodeMax: this.erodeMax,
      uErode: state.erode,

      uSteps: Math.round(state.steps),
      uShadowSteps: Math.round(state.shadowSteps),
      uAoTaps: Math.round(state.aoTaps),
      uReflectSteps: Math.round(state.reflectSteps),
      uShadowNoise: state.shadowNoise ? 1 : 0,
      uReflectLit: state.reflectLit ? 1 : 0,
    });
    tri.draw();

    // Temporal blend is dialled back while anything moves, or the jitter
    // turns into a smear.
    /* Anything that moves the marched half has moved the rasterised one
       as well — the camera at least — so the mesh's own limit is the
       lower of the two. */
    const blend = this.moving ? Math.min(state.taa, 0.55) : state.taa;
    const blendMesh = (this.moving || this.movingMesh)
      ? Math.min(state.taa, 0.55) : state.taa;

    this.history.write.bind();
    this.accum.use({
      uSrc: this.rt.texture,
      uHistory: this.history.read.texture,
      uMesh: this.meshRT.texture,
      uMeshOn: raster ? 1 : 0,
      uBlend: blend,
      uBlendMesh: blendMesh,
    });
    tri.draw();
    this.history.swap();

    bindScreen(gl, this.width, this.height);
    this.resolve.use({ uSrc: this.history.read.texture, uExposure: state.exposure });
    tri.draw();

    if (rippleActive && state.flash > 0) {
      BLEND.additive(gl);
      this.flare.use({
        uFlare: this.flarePts,
        uFlareAmt: this.flareAmt,
        uCamPos: this.basis.pos,
        uRight: this.basis.right,
        uUp: this.basis.up,
        uFwd: this.basis.fwd,
        uFocal: 1.5,
        uAspect: this.width / Math.max(this.height, 1),
        uViewportH: this.height,
        uScene: this.rt.texture,
        uResolution: [this.width, this.height],
        uColor: [0.9, 0.94, 1.0],
        uIntensity: 0.55,
      });
      empty.drawPoints(RIPPLE_N);
      BLEND.none(gl);
    }

    /* Whatever is falling, over the resolved image and additive like the
       flares — and before the beams, because a shot fired through rain
       should be the brighter of the two.

       Deliberately outside the temporal filter. Everything in this
       target that is worth accumulating is worth accumulating because it
       is standing still; a raindrop crosses the frame in a fifth of a
       second, and the filter's answer to that is a grey smear where the
       weather was. */
    {
      const camera = {
        pos: this.basis.pos,
        right: this.basis.right,
        up: this.basis.up,
        fwd: this.basis.fwd,
        focal: 1.5,
        aspect: this.width / Math.max(this.height, 1),
      };
      const verts = this.precip.draw(camera, {
        weather,
        time: clock.time,
        wind: state.wind,
        scene: this.rt.texture,
        mesh: this.meshRT.texture,
        meshOn: raster,
        resolution: [this.width, this.height],
      });
      if (verts) {
        BLEND.additive(gl);
        empty.drawTriangles(verts);
        BLEND.none(gl);
      }

      /* And the fireflies, which are here for the same reason the rain
         is and not for the same reason the butterflies are: a glow is
         something added to the image, not a surface that occludes one.
         They still hide behind the scene's published depth, so a lit one
         behind a trunk stays behind it. */
      const glowVerts = this.creatures.drawGlow(camera, {
        density: state.fireflies,
        reach: state.lifeRadius,
        time: clock.time,
        /* The one population that needs both halves of the answer: how
           much daylight there is, and whether anything is in charge of
           saying so. The other two are handed `day` inside env and it is
           1 when nothing is — which is already the right answer for
           something that flies by day, and the wrong one for this. */
        day,
        timed,
        hills: state.hills,
        waterY,
        scene: this.rt.texture,
        mesh: this.meshRT.texture,
        meshOn: raster,
        resolution: [this.width, this.height],
      });
      if (glowVerts) {
        BLEND.additive(gl);
        empty.drawTriangles(glowVerts);
        BLEND.none(gl);
      }
    }

    /* The beams, over the resolved image and additive like the flares.
       They read the same depth channel, so a beam ends at the surface it
       strikes instead of being painted across it. */
    if (this.laser.active) {
      BLEND.additive(gl);
      this.laser.draw(
        {
          pos: this.basis.pos,
          right: this.basis.right,
          up: this.basis.up,
          fwd: this.basis.fwd,
          focal: 1.5,
          aspect: this.width / Math.max(this.height, 1),
        },
        this.rt.texture,
        [this.width, this.height],
      );
      empty.drawTriangles(12);   // two beams, two triangles each
      BLEND.none(gl);
    }
  }

  readout() {
    /* The scene's own boast is that it has no geometry. That is still
       true of the *field* — the triangles belong to the cat and to the
       meadow — and naming them separately is more honest than either
       claiming zero or lumping the three together. */
    const parts = [];
    if (this._showCat) parts.push(`貓：${this.cat.triangles.toLocaleString()} 三角形`);
    if (this.ground.triangles) parts.push(`植被：${this.ground.triangles.toLocaleString()} 三角形`);
    if (this.trees.triangles) {
      parts.push(`樹：${this.trees.trees} 棵 / ${this.trees.triangles.toLocaleString()} 三角形`);
    }
    /* Counted, and counted separately, because the whole claim about the
       butterflies is that one costs four triangles — a claim nobody
       should have to take on trust. The fireflies have none: that is the
       claim about them. */
    if (this.creatures.butterflies) {
      parts.push(`蝴蝶：${this.creatures.butterflies} 隻 / `
        + `${this.creatures.triangles.toLocaleString()} 三角形`);
    }
    if (this.creatures.sparrows) {
      parts.push(`麻雀：${this.creatures.sparrows} 隻`);
    }
    if (this.creatures.fireflies) {
      parts.push(`螢火蟲：${this.creatures.fireflies} 點`);
    }
    if (this.precip.drops) parts.push(`天氣：${this.precip.drops.toLocaleString()} 顆`);
    const geometry = parts.length ? `場：0 頂點 · ${parts.join(' · ')}` : '0 頂點 · 0 三角形';

    const out = {
      '渲染尺寸': `${this.rt.width}×${this.rt.height}`,
      '幾何': geometry,
      '場景半徑': this.bound[3].toFixed(2),
      '進行中的漣漪': String(this._rippleAge.reduce((n, a) => n + (a > 0 ? 1 : 0), 0)),
      '撞擊次數': String(this.bursts),
      '鏡頭距離': this.dist.toFixed(2),
    };

    if (this.catError) out['貓'] = '載入失敗';
    else if (this._showCat) {
      out['貓的位置'] = `${this.cat.x.toFixed(2)}, ${this.cat.z.toFixed(2)}`;
      out['貓的速度'] = `${this.cat.speed.toFixed(2)} u/s`;
      // Whether the cursor is captured is not cosmetic — it decides what
      // the mouse does and how far the cat can turn — so it is stated.
      if (this.shots) out['雷射次數'] = String(this.shots);
      out['操作模式'] = this.cat.mode === 'look'
        ? (this.locked ? '滑鼠轉向 · 已鎖定指標' : '滑鼠轉向 · 未鎖定（點畫布可重新鎖定）')
        : 'WASD 依鏡頭方向';
    }
    return out;
  }

  dispose() {
    this.ctx.canvas.removeEventListener('wheel', this._onWheel);
    document.removeEventListener('mousemove', this._onMouseMove);
    document.removeEventListener('pointerlockchange', this._onLockChange);
    // Leaving the scene must never leave the cursor captured by it.
    if (this.locked) document.exitPointerLock?.();
    this.march.dispose();
    this.accum.dispose();
    this.resolve.dispose();
    this.flare.dispose();
    this.laser.dispose();
    this.rt.dispose();
    this.meshRT.dispose();
    this.ground.dispose();
    this.trees.dispose();
    this.precip.dispose();
    this.creatures.dispose();
    this.history.dispose();
    this.cat?.dispose();
  }
}

function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

/** The same polynomial smooth minimum the shader uses. */
function smin(a, b, k) {
  const h = Math.min(Math.max(0.5 + 0.5 * (b - a) / k, 0), 1);
  return b + (a - b) * h - k * h * (1 - h);
}
