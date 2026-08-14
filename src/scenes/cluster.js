/* ── scenes/cluster.js ───────────────────────────────────────────────
   The cluster's shape, as GLSL, in one place.

   It used to live entirely inside `march.js`, which was correct while
   the marcher was its only reader. It is not any more: the cat has to
   ask the same field whether the sun is blocked, and a second copy of
   these functions is exactly the kind of thing that agrees for a week
   and then quietly stops.

   Only the *shape* is here — spheres and ring. The ripples, the surface
   noise and the dissipation stay in `march.js`, because they move the
   surface by a couple of centimetres and nothing outside the marcher can
   see a difference that small.
   ------------------------------------------------------------------ */

export const BALL_N = 9;        // spheres in the cluster
export const RIPPLE_N = 4;      // concurrent surface rings
export const RING_MAJOR = 1.28;
export const RING_MINOR = 0.075;
export const FLOOR_Y = -1.35;

/** Peak displacement per unit of the `displace` slider. */
export const DISPLACE_AMP = 0.12;

/**
 * Everything the shape needs, declared once. A shader that includes
 * `CLUSTER_FIELD` must include this too and must not redeclare any of
 * it — which is why the marcher's own uniform block no longer mentions
 * time or blend radius.
 */
export const CLUSTER_UNIFORMS = /* glsl */`
#define BALL_N ${BALL_N}
#define RIPPLE_N ${RIPPLE_N}
#define RING_MAJOR ${RING_MAJOR.toFixed(4)}
#define RING_MINOR ${RING_MINOR.toFixed(4)}
#define FLOOR_Y ${FLOOR_Y.toFixed(4)}
#define DISPLACE_AMP ${DISPLACE_AMP.toFixed(4)}

uniform float uTime;
uniform float uBlend;
uniform vec4  uBallPos[BALL_N];   // xyz = centre, w = radius
uniform float uBalls;
/** xyz = centre, w = radius. Everything the cluster can reach. */
uniform vec4  uBound;

uniform vec4  uRipples[RIPPLE_N]; // xyz = where it started, w = normalised age
/* An impact is a *segment*, not a point. A click sets both ends the same
   and behaves exactly as it always did; a beam sets them to where it
   entered and left what it passed through. */
uniform vec4  uRippleTo[RIPPLE_N];   // xyz = far end, w = dissipation strength
uniform float uRippleOn, uRippleAmp, uRippleSpeed, uRippleFreq, uRippleTight;
uniform float uErode, uDisplace;

// Shadow quality, shared so both readers soften identically.
uniform float uShadowNoise;
uniform int   uShadowSteps;
`;

/** Primitives, the smooth minimum, and the ray/sphere span. */
export const CLUSTER_FIELD = /* glsl */`
float sdSphere(vec3 p, float r) { return length(p) - r; }

float sdTorus(vec3 p, vec2 t) {
  vec2 q = vec2(length(p.xz) - t.x, p.y);
  return length(q) - t.y;
}

/** Polynomial smooth minimum — the operator that makes SDFs feel alive. */
float smin(float a, float b, float k) {
  float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}

/**
 * Entry and exit distance along a ray for a sphere, or (1, -1) if it
 * misses. The direction must be unit length, which makes the
 * quadratic's leading coefficient 1 and the whole thing three dot
 * products. (No back-ticks in here: the shader lives inside a JS
 * template literal and one would end it.)
 */
vec2 sphereSpan(vec3 ro, vec3 rd, vec3 c, float r) {
  vec3 oc = ro - c;
  float b = dot(oc, rd);
  float k = dot(oc, oc) - r * r;
  float h = b * b - k;
  if (h < 0.0) return vec2(1.0, -1.0);
  h = sqrt(h);
  return vec2(-b - h, -b + h);
}

/** Spheres and ring: the shape, before anything is done to its surface. */
float clusterBase(vec3 p) {
  float d = 1e9;
  for (int i = 0; i < BALL_N; i++) {
    if (float(i) >= uBalls) break;
    d = smin(d, sdSphere(p - uBallPos[i].xyz, uBallPos[i].w), uBlend);
  }

  // A ring threading the cluster, on its own slow tumble.
  float t = uTime * 0.42;
  vec3 q = p;
  q.yz = rot2(t * 0.31) * q.yz;
  q.xz = rot2(t * 0.19) * q.xz;
  return smin(d, sdTorus(q, vec2(RING_MAJOR, RING_MINOR)), uBlend * 0.6);
}
`;

/**
 * The rest of the field, in the layers each consumer can afford.
 *
 *   clusterBase   spheres and ring          — the shape
 *   clusterShape  + impact ripples          — shadows and occlusion
 *   clusterFull   + surface displacement    — primary rays and normals
 *
 * These moved out of the marcher when the cat needed to cast its shadow
 * ray through the same thing the marcher shades against. A shadow that
 * walks a simpler field than the surface it lands on is a shadow of a
 * slightly different object, and the two drift apart exactly when the
 * scene is at its most active.
 *
 * Requires `snoise` and `PI`, so include SIMPLEX3 and CONSTANTS first.
 */
export const CLUSTER_LAYERS = /* glsl */`
/**
 * Distance from a point to an impact.
 *
 * The impact is a segment. A zero-length one is a point, and the clamp
 * makes that fall out for free rather than needing its own branch — so
 * a click and a beam run the identical code.
 */
float rippleDist(int i, vec3 p) {
  vec3 a = uRipples[i].xyz;
  vec3 ba = uRippleTo[i].xyz - a;
  vec3 pa = p - a;
  float bb = dot(ba, ba);
  float h = bb > 1e-8 ? clamp(dot(pa, ba) / bb, 0.0, 1.0) : 0.0;
  return length(pa - ba * h);
}

/**
 * Expanding rings from each recorded impact.
 *
 * Displacing a distance field is not the same as displacing a mesh:
 * there is no surface to move, so the ripple is authored as a term
 * added to the distance itself. Keep the amplitude small — a large one
 * breaks the field's Lipschitz bound and the sphere tracer starts
 * overshooting straight through the surface.
 *
 * A point throws a sphere. A beam does not throw anything sideways, so
 * its wave travels along its own axis instead — see below.
 */
float ripples(vec3 p) {
  float sum = 0.0;
  for (int i = 0; i < RIPPLE_N; i++) {
    float age = uRipples[i].w;
    if (age <= 0.0) continue;

    vec3 a = uRipples[i].xyz;
    vec3 ba = uRippleTo[i].xyz - a;
    float len2 = dot(ba, ba);

    float d, front, confine = 1.0;
    if (len2 > 1e-6) {
      /* A bore. The wave runs *down* the channel, not out of its sides.
         Taking the distance to the segment instead would throw a
         cylindrical wave outward from the beam, which is how a point
         impact behaves and is wrong here: nothing about a beam pushes
         sideways. So the front is measured along the axis and swept
         from the entry to the exit over the impact's life, and a radial
         falloff keeps it inside the hole it is travelling through. */
      float len = sqrt(len2);
      vec3 dir = ba / len;
      vec3 rel = p - a;
      d = dot(rel, dir);
      front = age * len * uRippleSpeed;
      confine = exp(-length(rel - dir * d) * 2.5);
    } else {
      // A point impact, expanding as a sphere the way it always did.
      d = length(p - a);
      front = age * uRippleSpeed;
    }

    // Tight in space around the travelling front, fading in time.
    float env = exp(-abs(d - front) * uRippleTight) * (1.0 - age) * (1.0 - age);
    sum += sin((d - front) * uRippleFreq) * env * confine;
  }
  return sum * uRippleAmp;
}

/** How far the damage can reach at its peak. Harder blows spread further. */
float erodeReach() { return 0.26 + uErode * 1.15; }

/**
 * Width of the damaged region's soft edge. Held constant on purpose:
 * it is what bounds the term's gradient, so the tracer's step budget
 * does not depend on where in its life an impact happens to be.
 */
const float ERODE_EDGE = 0.20;

/**
 * Mass dissipating under the shock.
 *
 * Subtracting a sphere would be the obvious way to open a hole, and it
 * is exact and cheap — but it leaves a machined circular rim, an edge
 * no amount of energy would actually produce. This is an *additive*
 * term instead: it pushes the surface inward wherever the impact's
 * energy is high, and because the falloff is a smooth blob the surface
 * tapers away to nothing rather than being cut. Where the dissipation
 * exceeds the local thickness the surface simply stops existing, which
 * is how the ring — 0.075 thick — opens a gap and then closes it again
 * as the energy drains.
 *
 * Being additive also means its gradient is amplitude over width rather
 * than amplitude times frequency, so it feeds the same derived step
 * bound as everything else instead of needing a special case.
 */
float erodeMask(vec3 p) {
  if (uErode <= 0.0) return 0.0;
  float reach = erodeReach();
  float sum = 0.0;
  for (int i = 0; i < RIPPLE_N; i++) {
    float age = uRipples[i].w;
    if (age <= 0.0) continue;

    // What closes is the damaged region's *radius*, not its depth.
    //
    // Fading the depth uniformly is the obvious move and it is wrong:
    // anything small enough to sit entirely inside the damage just
    // shrinks and then grows back out of its own middle. Contracting the
    // boundary instead means the outermost material returns first and
    // the centre is the last thing to close — which is how a hole in
    // anything actually heals.
    //
    // Opens fast, closes slowly: the exponent skews the sine early.
    float r = reach * sin(PI * pow(age, 0.62));
    if (r <= 0.0) continue;

    float d = rippleDist(i, p);
    // Strength rides in the far end's spare component. A click is 1; a
    // beam is several, because boring is the whole of what it does, and
    // a hole has to be deeper than the wall it is going through.
    sum += uRippleTo[i].w * (1.0 - smoothstep(r - ERODE_EDGE, r, d));
  }
  return sum;
}

float clusterShape(vec3 p) {
  float d = clusterBase(p);
  if (uRippleOn > 0.5) {
    /* Unclamped for depth, clamped for masking. A beam dissipates more
       material than any one click can, and that is what perforates a
       sphere rather than dimpling it — but the wave can only be silenced
       once, so the mask saturates while the depth does not. */
    float gone = erodeMask(p);
    float mask = min(gone, 1.0);

    // The wave is silenced inside the damage. A travelling sine has
    // negative phases, and a negative phase pushes the surface *outward*
    // — so without this the gap grows its own little lump of material at
    // the centre and carries it out to the rim. Nothing that has been
    // dissipated is left to carry a wave.
    d += ripples(p) * (1.0 - mask);

    // In the shape layer, not the full one: a gap has to be a gap for
    // the shadow and occlusion rays too, or light refuses to come
    // through something you can see straight out of.
    d += gone * uErode;
  }
  return d;
}

/**
 * The full field. The noise is the single most expensive thing here and
 * it was being evaluated at every march step, every shadow tap and every
 * AO tap — including at samples nowhere near a surface, where a
 * displacement of a couple of centimetres cannot possibly change the
 * answer.
 *
 * Outside a band a few amplitudes wide, subtracting the peak amplitude
 * is a valid lower bound on the true distance: the tracer stays
 * conservative, never overshoots, and skips the noise entirely.
 */
float clusterFull(vec3 p) {
  float d = clusterShape(p);
  if (uDisplace <= 0.0) return d;

  float amp = uDisplace * DISPLACE_AMP;
  if (d > amp * 4.0 + 0.02) return d - amp;

  return d + amp * snoise(p * 3.1 + vec3(0.0, 0.0, uTime * 0.3));
}

/** The layer shadows and ambient occlusion march. */
float clusterLit(vec3 p) {
  return uShadowNoise > 0.5 ? clusterFull(p) : clusterShape(p);
}
`;

/**
 * How much of the sun reaches a point, marching the cluster only.
 *
 * IQ's soft shadow: the closest approach along the ray *is* the
 * penumbra. The ray stops the moment it leaves the cluster's bounding
 * sphere, which is what makes shadows on the open floor almost free —
 * and is why anything outside the cluster can afford to ask.
 *
 * One copy, marching the same layered field, with the same step
 * schedule, for everything that asks. The spheres and the cat get the
 * identical penumbra out of it because it is identically computed —
 * which is the only way two objects lit by one sun can look like they
 * are standing in the same room.
 */
export const CLUSTER_SHADOW = /* glsl */`
float clusterShadow(vec3 ro, vec3 rd, float k) {
  vec2 span = sphereSpan(ro, rd, uBound.xyz, uBound.w);
  if (span.y <= 0.02) return 1.0;

  float res = 1.0;
  float t = max(span.x, 0.06);
  for (int i = 0; i < 64; i++) {
    if (i >= uShadowSteps) break;
    float h = clusterLit(ro + rd * t);
    res = min(res, k * h / t);
    t += clamp(h, 0.02, 0.35);
    if (res < 0.004 || t > span.y) break;
  }
  return clamp(res, 0.0, 1.0);
}
`;

/**
 * The sky, which is also the fog colour. Anything drawn into this scene
 * has to fade toward the same horizon or it will read as a cut-out no
 * matter how well it is lit.
 */
export const SKY = /* glsl */`
uniform vec3 uLightDir, uTint;

/* The stars' own hash, rather than the shared one.

   This block is included by four shaders and only two of them carry the
   common hash header — the cat's fragment stage is one that does not,
   and reaching for hash33 here is what broke it. A sky that drags a
   dependency in behind it is a sky that can only be included where
   somebody already thought to include something else, and the whole
   point of a shared chunk is that it goes anywhere. Six lines is a
   cheaper answer than a rule everybody has to remember. */
vec3 starHash(vec3 p) {
  p = fract(p * vec3(0.1031, 0.1030, 0.0973));
  p += dot(p, p.yxz + 33.33);
  return fract((p.xxy + p.yxx) * p.zyx);
}
/** How much daylight is in the sky, 0..1. One is the sky this scene has
    always had; the fixed-light mode uploads exactly that, so a frame
    taken before there was a time of day still renders byte for byte. */
uniform float uDay;

/* How much cloud is between here and it, 0..1.

   The weather already reaches everything under the sky and nothing in
   it: the ground goes dark and wet, the drifts lie, something falls —
   and above all of that, a midnight downpour under a clear starfield.
   That is the specific frame this number exists to stop.

   It arrives resolved rather than as the rain and the snow themselves,
   and not for tidiness. This block is included by five shaders and two
   of them pull it in *before* the weather's own uniforms; a sky that
   reached for uRain would be a sky that compiles only where somebody
   already thought to include something else, which is the one thing a
   shared chunk must never be. One number, produced in JS from the mode
   table, and an upload site that forgets it reads zero — which is fair
   weather, and exactly the sky this scene had before. */
uniform float uOvercast;

/**
 * The light that arrives from everywhere rather than from the sun.
 *
 * Was three hardcoded numbers, repeated in the marcher, the cat and the
 * plants. It became a uniform the moment there was a night, because it
 * is the *only* term still delivering light once the sun is down — the
 * ground here has an albedo of about four percent and a directional term
 * scaled by moonlight, so with the fill left at its daytime value the
 * whole meadow resolves to black and the picture is a starfield with a
 * cluster in it.
 *
 * It is also what snow needs. Snow is bright because it is lit, and most
 * of what lights the shaded side of a drift is the sky; a snowfield with
 * a daytime-meadow fill has no blue in its shadows, and blue shadows are
 * most of what the eye is reading when it calls something snow rather
 * than sand.
 */
uniform vec3 uAmbient;

vec3 sky(vec3 rd) {
  float h = rd.y * 0.5 + 0.5;

  /* Two palettes and a lerp. The day one is the original set, unchanged
     and unchangeable — every reference shot in the project is measured
     against it. The night one is the same sky with the air taken out of
     it: a horizon barely brighter than the zenith, because at night what
     lights the low sky is the ground, and this ground is dark. */
  vec3 top = mix(vec3(0.006, 0.009, 0.022), vec3(0.045, 0.062, 0.10), uDay);
  vec3 hor = mix(vec3(0.012, 0.016, 0.030), vec3(0.110, 0.121, 0.149), uDay);
  vec3 bot = mix(vec3(0.004, 0.005, 0.010), vec3(0.015, 0.016, 0.022), uDay);

  vec3 c = mix(bot, hor, smoothstep(0.35, 0.5, h));
  c = mix(c, top, smoothstep(0.5, 1.0, h));

  /* Overcast, which is mostly the *loss of a gradient*. A clear sky is
     deep overhead and pale at the horizon because that is where the air
     is; a covered one is a lit sheet a kilometre up and reads almost the
     same brightness everywhere, which is why a flattened gradient says
     "cloud" before any colour does.

     Toward the horizon's own colour rather than toward a grey of its
     own: the horizon band is already the mixture of daylight and air
     this sky is made of, so flattening onto it keeps every hour's
     palette — an overcast dusk stays orange, an overcast midnight stays
     nearly black — without a second set of constants to keep in step
     with the first. Lifted a little, because cloud lit from above is
     brighter than the air it replaced. */
  if (uOvercast > 0.0) {
    c = mix(c, hor * mix(1.0, 1.35, uDay), uOvercast * 0.85);
  }

  /* Stars, and only the ones the sky is dark enough to show. Placed by
     hashing a cell of the direction itself rather than a texture: the
     sphere is being sampled by the same rays that sample everything
     else, so there is no projection to distort and no seam to hide.

     The threshold is what keeps them sparse. Every cell has a candidate
     in it; roughly one in eighty is allowed to be a star, and which one
     is decided by the same hash that placed it, so the field is fixed to
     the world and does not swim as the camera turns. */
  // And no stars through cloud, which is the whole of why the weather is
  // in this function: a snowstorm at midnight had a full sky of them.
  float night = (1.0 - smoothstep(0.0, 0.62, uDay)) * (1.0 - uOvercast);
  if (night > 0.002 && rd.y > -0.02) {
    vec3 d = rd * 46.0;
    vec3 g = starHash(floor(d));
    float star = pow(max(0.0, 1.0 - length(fract(d) - 0.15 - g * 0.7) * 2.6), 22.0);
    star *= step(0.9875, g.z);
    // Faded into the horizon haze, where there is more air to look
    // through and where the ground is about to be in the way anyway.
    c += vec3(0.86, 0.90, 1.0) * star * night * smoothstep(-0.02, 0.30, rd.y) * 1.5;
  }

  /* A soft disc so reflections have something to catch. It is the sun by
     day and the moon by night without being told which: uTint already
     carries the body's colour and its strength, so the same two lines
     draw a white-hot sun, a red one at dusk, and a small pale moon. */
  float toLight = max(dot(rd, uLightDir), 0.0);
  /* The disc goes first and goes completely. A body still visible
     through the cloud that is dimming everything else is the tell, and
     it is a sharper one than the light level: overcast is defined by
     there being no disc to find. */
  c += uTint * pow(toLight, 220.0) * 4.0 * (1.0 - uOvercast);

  /* The halo around it, which has to tighten at night or the moon has no
     disc left. By day the wide glow is the sun's own scatter and reads
     correctly at the horizon; at night the same width spread over a sky
     that is a hundredth as bright swallows the body inside it and leaves
     a pale smear where the moon should be. Both ends interpolate from
     uDay, so full daylight is the original two constants exactly.

     Under cloud it goes the other way from the disc: it survives, and it
     spreads. The sun behind an overcast sky is a bright region you can
     point at without being able to say where its edge is, and that is
     one exponent and one weight rather than a second glow. */
  float halo = mix(0.030, 0.14, uDay) * mix(1.0, 1.8, uOvercast);
  c += uTint * halo * pow(toLight, mix(26.0, 5.0, uDay) * mix(1.0, 0.22, uOvercast));
  return c;
}
`;
