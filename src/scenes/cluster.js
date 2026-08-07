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
export const RING_MAJOR = 1.28;
export const RING_MINOR = 0.075;
export const FLOOR_Y = -1.35;

/**
 * Everything the shape needs, declared once. A shader that includes
 * `CLUSTER_FIELD` must include this too and must not redeclare any of
 * it — which is why the marcher's own uniform block no longer mentions
 * time or blend radius.
 */
export const CLUSTER_UNIFORMS = /* glsl */`
#define BALL_N ${BALL_N}
#define RING_MAJOR ${RING_MAJOR.toFixed(4)}
#define RING_MINOR ${RING_MINOR.toFixed(4)}
#define FLOOR_Y ${FLOOR_Y.toFixed(4)}

uniform float uTime;
uniform float uBlend;
uniform vec4  uBallPos[BALL_N];   // xyz = centre, w = radius
uniform float uBalls;
/** xyz = centre, w = radius. Everything the cluster can reach. */
uniform vec4  uBound;
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
 * How much of the sun reaches a point, marching the cluster only.
 *
 * IQ's soft shadow: the closest approach along the ray *is* the
 * penumbra. The ray stops the moment it leaves the cluster's bounding
 * sphere, which is what makes shadows on the open floor almost free —
 * and is why anything outside the cluster can afford to ask.
 *
 * The marcher has its own copy of this that walks the full perturbed
 * field. This one is the cheap version, over the shape alone: a
 * consumer that is not itself the surface cannot see a ripple in its
 * own shadow.
 */
export const CLUSTER_SHADOW = /* glsl */`
float clusterShadow(vec3 ro, vec3 rd, float k, int steps) {
  vec2 span = sphereSpan(ro, rd, uBound.xyz, uBound.w);
  if (span.y <= 0.02) return 1.0;

  float res = 1.0;
  float t = max(span.x, 0.06);
  for (int i = 0; i < 48; i++) {
    if (i >= steps) break;
    float h = clusterBase(ro + rd * t);
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

vec3 sky(vec3 rd) {
  float h = rd.y * 0.5 + 0.5;
  vec3 top = vec3(0.045, 0.062, 0.10);
  vec3 hor = vec3(0.10, 0.11, 0.135) * 1.1;
  vec3 bot = vec3(0.015, 0.016, 0.022);
  vec3 c = mix(bot, hor, smoothstep(0.35, 0.5, h));
  c = mix(c, top, smoothstep(0.5, 1.0, h));
  // A soft sun disc so reflections have something to catch.
  float sun = pow(max(dot(rd, uLightDir), 0.0), 220.0);
  c += uTint * sun * 4.0;
  c += uTint * 0.14 * pow(max(dot(rd, uLightDir), 0.0), 5.0);
  return c;
}
`;
