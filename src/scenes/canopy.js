/* ── scenes/canopy.js ────────────────────────────────────────────────
   Reading the trees' shadow.

   Only the lookup lives here. The map itself is rendered by trees.js,
   because it is drawn from the trees' own instance buffers with the
   trees' own wind applied — a shadow of a tree standing still while the
   tree it belongs to leans is worse than no shadow. What is here is the
   half that four different shaders need: the marched floor, the grass,
   the cat, and the leaves themselves.

   It is a real shadow map, with depth. The cheaper thing was a coverage
   map — one channel saying whether a canopy is overhead — and it fails
   in a way that is easy to overlook until you see it: with no depth
   there is no way to tell an occluder between you and the sun from one
   behind you, so the inside of every canopy comes out uniformly black
   instead of dappled. The second channel costs one attachment and buys
   both the self-shadowing and a penumbra that widens with the gap, which
   is what lets a tree's shadow behave like the cluster's marched one
   instead of announcing that it came from a texture.
   ------------------------------------------------------------------ */

/** Half-width of the ground the map covers, in world units. */
export const CANOPY_EXTENT = 40.0;
/** Side of the map. 1024 over 80 units is about 8 cm per texel. */
export const CANOPY_SIZE = 1024;

/**
 * How much of the sun a point loses to the leaves above it.
 *
 * Requires uLightDir. Returns 1 where nothing is overhead, including
 * everywhere outside the map — a border that shadowed would be a black
 * square edge across the meadow.
 */
export const CANOPY_SHADE_GLSL = /* glsl */`
uniform sampler2D uCanopy;
uniform vec3  uCanopyX, uCanopyY, uCanopyC;
uniform float uCanopyExtent, uCanopyOn;

float canopyShade(vec3 p) {
  if (uCanopyOn < 0.5) return 1.0;

  vec3 rel = p - uCanopyC;
  vec2 uv = vec2(dot(rel, uCanopyX), dot(rel, uCanopyY)) / uCanopyExtent * 0.5 + 0.5;
  if (any(lessThan(uv, vec2(0.004))) || any(greaterThan(uv, vec2(0.996)))) return 1.0;

  // Where this point sits along the light's own axis. Anything with a
  // larger value is nearer the sun, and therefore in the way.
  float mine = dot(rel, uLightDir);

  vec2 texel = vec2(1.0) / vec2(textureSize(uCanopy, 0));
  vec4 c = texture(uCanopy, uv);

  /* The gap between receiver and occluder sets how wide the penumbra is.
     This is the whole reason the map carries a height: a fixed blur
     makes every tree's shadow equally soft, which reads as a sticker
     next to the cluster's marched shadow — that one is sharp where it
     touches the ground and spreads with distance, and so should this. */
  float gap = max(c.g - mine, 0.0) * step(0.5, c.r);
  float radius = clamp(gap * 0.9, 1.6, 11.0);

  /* Four taps on a rotated cross plus the centre. Enough at this texel
     size, and it has to be cheap: the marcher inlines its caller twice,
     once for the primary ray and once for the reflection bounce, and
     that is the function that already blew the instruction limit once. */
  float sum = 0.0;
  for (int i = 0; i < 4; i++) {
    float a = float(i) * 1.5708 + 0.3927;
    vec2 o = vec2(cos(a), sin(a)) * radius * texel;
    vec4 s = texture(uCanopy, uv + o);
    sum += (s.r > 0.5 && s.g > mine + 0.06) ? 0.0 : 1.0;
  }
  sum += (c.r > 0.5 && c.g > mine + 0.06) ? 0.0 : 1.0;

  /* Never all the way to black. A canopy leaks, and the fraction that
     gets through the gaps is the difference between shade and a hole cut
     in the world. */
  return mix(0.22, 1.0, sum * 0.2);
}
`;
