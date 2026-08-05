/* ── shaders/common.js ───────────────────────────────────────────────
   GLSL chunks as tagged template strings.

   Without a build step there is no #include, so shared GLSL lives in JS
   string constants and is composed with `${}`. That is not a workaround
   — it means shader code is a first-class module, tree-shaken by the
   browser's own module graph, with no preprocessor to debug.
   ------------------------------------------------------------------ */

/** Every fragment shader in this project starts here. */
export const PRECISION = /* glsl */`
precision highp float;
precision highp int;
precision highp sampler2D;
`;

/** Vertex shader for the oversized fullscreen triangle in core/gl.js. */
export const VERT_FULLSCREEN = /* glsl */`
in vec2 aPosition;
out vec2 vUv;
void main() {
  vUv = aPosition * 0.5 + 0.5;
  gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

export const CONSTANTS = /* glsl */`
#define PI    3.14159265359
#define TAU   6.28318530718
#define EPS   1e-4
`;

/* ═══ hashing ═════════════════════════════════════════════════════ */

export const HASH = /* glsl */`
float hash11(float p) {
  p = fract(p * 0.1031);
  p *= p + 33.33;
  p *= p + p;
  return fract(p);
}

vec2 hash22(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.xx + p3.yz) * p3.zy);
}

vec3 hash33(vec3 p3) {
  p3 = fract(p3 * vec3(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yxz + 33.33);
  return fract((p3.xxy + p3.yxx) * p3.zyx);
}

/** Deterministic per-particle randomness from a texel index. */
vec3 seedFromIndex(vec2 uv, float salt) {
  return hash33(vec3(uv * 512.0, salt));
}
`;

/* ═══ simplex noise ═══════════════════════════════════════════════
   Ashima Arts / Stefan Gustavson's classic 3D simplex, MIT licensed.
   Simplex over Perlin because the gradient is well-behaved in every
   direction — which matters enormously when you differentiate it to
   build a curl field.                                                */

export const SIMPLEX3 = /* glsl */`
vec3 mod289v3(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 mod289v4(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289v4(((x * 34.0) + 1.0) * x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v) {
  const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

  vec3 i  = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);

  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);

  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;

  i = mod289v3(i);
  vec4 p = permute(permute(permute(
             i.z + vec4(0.0, i1.z, i2.z, 1.0))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0))
           + i.x + vec4(0.0, i1.x, i2.x, 1.0));

  float n_ = 0.142857142857;
  vec3 ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);

  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);

  vec4 s0 = floor(b0) * 2.0 + 1.0;
  vec4 s1 = floor(b1) * 2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);

  vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;

  vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

float fbm(vec3 p, int octaves, float lacunarity, float gain) {
  float sum = 0.0, amp = 0.5, norm = 0.0;
  for (int i = 0; i < 8; i++) {
    if (i >= octaves) break;
    sum  += amp * snoise(p);
    norm += amp;
    p    *= lacunarity;
    amp  *= gain;
  }
  return sum / max(norm, EPS);
}
`;

/* ═══ curl noise ══════════════════════════════════════════════════ */

export const CURL = /* glsl */`
/**
 * Curl of a noise vector potential. The result is divergence-free by
 * construction, which is exactly what makes the particle motion read as
 * "fluid" rather than "drifting": nothing ever piles up or thins out.
 */
vec3 potential(vec3 p) {
  return vec3(
    snoise(p),
    snoise(p + vec3(31.416, 0.0, 47.853)),
    snoise(p - vec3(17.239, 91.472, 0.0))
  );
}

vec3 curlNoise(vec3 p) {
  const float e = 0.09;
  vec3 dx = vec3(e, 0.0, 0.0);
  vec3 dy = vec3(0.0, e, 0.0);
  vec3 dz = vec3(0.0, 0.0, e);

  vec3 px0 = potential(p - dx), px1 = potential(p + dx);
  vec3 py0 = potential(p - dy), py1 = potential(p + dy);
  vec3 pz0 = potential(p - dz), pz1 = potential(p + dz);

  float x = (py1.z - py0.z) - (pz1.y - pz0.y);
  float y = (pz1.x - pz0.x) - (px1.z - px0.z);
  float z = (px1.y - px0.y) - (py1.x - py0.x);

  return vec3(x, y, z) / (2.0 * e);
}
`;

/* ═══ colour ══════════════════════════════════════════════════════ */

export const COLOR = /* glsl */`
/** Iñigo Quílez's cosine palette: four vec3s describe a whole ramp. */
vec3 cosPalette(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
  return a + b * cos(TAU * (c * t + d));
}

vec3 hsv2rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + vec3(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
  return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

/** ACES filmic curve (Narkowicz fit) — keeps highlights from clipping flat. */
vec3 acesFilm(vec3 x) {
  const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 linearToSrgb(vec3 c) {
  return mix(c * 12.92, 1.055 * pow(max(c, 1e-5), vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}

/** Ordered dithering. 8-bit output on a dark gradient bands badly;
    a fraction of a bit of structured noise removes it entirely. */
float bayer4(vec2 p) {
  vec2 c = floor(mod(p, 4.0));
  float i = c.x + c.y * 4.0;
  const float m[16] = float[16](
     0.0,  8.0,  2.0, 10.0,
    12.0,  4.0, 14.0,  6.0,
     3.0, 11.0,  1.0,  9.0,
    15.0,  7.0, 13.0,  5.0);
  return m[int(i)] / 16.0 - 0.5;
}

vec3 dither(vec3 c, vec2 fragCoord) {
  return c + bayer4(fragCoord) / 255.0;
}
`;

/* ═══ transforms ══════════════════════════════════════════════════ */

export const ROTATE = /* glsl */`
mat2 rot2(float a) { float s = sin(a), c = cos(a); return mat2(c, -s, s, c); }

mat3 orbit(float yaw, float pitch) {
  float cy = cos(yaw),   sy = sin(yaw);
  float cp = cos(pitch), sp = sin(pitch);
  return mat3(cy, 0.0, -sy, 0.0, 1.0, 0.0, sy, 0.0, cy)
       * mat3(1.0, 0.0, 0.0, 0.0, cp, -sp, 0.0, sp, cp);
}
`;

/** Convenience bundle: everything a typical fragment shader wants. */
export const PRELUDE = PRECISION + CONSTANTS + HASH + COLOR;
