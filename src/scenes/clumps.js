/* ── scenes/clumps.js ────────────────────────────────────────────────
   Where the flowers are, in both languages.

   The flowers are sown by clump: the world is cut into four-metre cells,
   a hash decides which cells grow one, and a second hash puts its centre
   somewhere inside. That was ground.js's private business right up until
   something else wanted to know the answer — and what wanted to know was
   the butterflies, because a butterfly that is not going anywhere in
   particular is a butterfly that reads as a leaf in the wind.

   So the grid moved here, and now there is one description of where a
   clump is: the sowing reads it to place blooms, and the flight reads it
   to pick something to visit. The alternative — a second scatter for the
   butterflies to circle — would have them orbiting patches of bare grass
   next to the flowers, which is worse than not aiming them at all,
   because the eye notices the *near miss* and never notices the absence.

   ── the hash across two languages ────────────────────────────────
   `seed` is a 32-bit integer mix. JS does it in doubles through
   Math.imul; GLSL does it in uint, where the arithmetic is defined
   modulo 2^32 and the bit patterns come out identical. The only place
   they part company is the final divide: JS has 53 bits of mantissa and
   GLSL ES has 24, so the two answers can differ in the seventh decimal.

   That matters for exactly one thing and it is worth naming. A cell
   whose seed sits within one part in ten million of the density
   threshold could be judged to hold a clump by one side and not the
   other, and the symptom would be a single butterfly circling nothing.
   One cell in ten million, and the cost of fixing it is carrying an
   integer comparison through both sides for the rest of the project's
   life. It is written down instead.
   ------------------------------------------------------------------ */

/** The clump grid, in metres. Fixed, and deliberately not scaled by the
    flower reach — see ground.js: winding the view out must not thin the
    flowers underfoot. */
export const CLUMP_CELL = 4.0;

/** How wide a clump is before the spread control scales it. Read here by
    the flight, which wants to circle a clump at about its own radius. */
export const CLUMP_R_MIN = 0.40;
export const CLUMP_R_MAX = 1.50;

/**
 * One value in [0,1) from a cell and a salt.
 *
 * @param {number} ix
 * @param {number} iz
 * @param {number} k  which question is being asked of this cell
 */
export function seed(ix, iz, k) {
  let h = Math.imul(ix | 0, 0x27d4eb2d) ^ Math.imul(iz | 0, 0x165667b1)
        ^ Math.imul(k | 0, 0x9e3779b1);
  h ^= h >>> 15; h = Math.imul(h, 0x2c1b3c6d);
  h ^= h >>> 12; h = Math.imul(h, 0x297a2d39);
  h ^= h >>> 15;
  return (h >>> 0) / 4294967296;
}

/**
 * The same grid, for a shader.
 *
 * Declares nothing and takes everything as an argument, so it can be
 * included by a pass that has never heard of the flower controls — the
 * caller passes the density it happens to be drawing with.
 *
 * (No back-ticks below: this lives inside a JS template literal.)
 */
export const CLUMP_GLSL = /* glsl */`
#define CLUMP_CELL ${CLUMP_CELL.toFixed(3)}
#define CLUMP_R_MIN ${CLUMP_R_MIN.toFixed(3)}
#define CLUMP_R_MAX ${CLUMP_R_MAX.toFixed(3)}

/** ground.js's sowing hash, in uint. Same bits, same answers. */
float clumpSeed(int ix, int iz, int k) {
  uint h = uint(ix) * 0x27d4eb2du
         ^ uint(iz) * 0x165667b1u
         ^ uint(k)  * 0x9e3779b1u;
  h ^= h >> 15u; h *= 0x2c1b3c6du;
  h ^= h >> 12u; h *= 0x297a2d39u;
  h ^= h >> 15u;
  return float(h) * (1.0 / 4294967296.0);
}

/**
 * The clump in one cell, if there is one.
 *
 * Returns its radius, or 0 for a cell that grows nothing. The centre
 * comes out in the out parameter, and it is the identical arithmetic the
 * sowing uses — same salts, same 0.72 and 0.14 — so the point returned
 * here is the point the blooms were scattered around, not merely near
 * it. (No back-ticks in this block: it is a JS template literal.)
 */
float clumpAt(int ix, int iz, float chance, float spread, out vec2 centre) {
  centre = vec2(0.0);
  if (clumpSeed(ix, iz, 1) > chance) return 0.0;
  centre = vec2(
    (float(ix) + clumpSeed(ix, iz, 2) * 0.72 + 0.14) * CLUMP_CELL,
    (float(iz) + clumpSeed(ix, iz, 3) * 0.72 + 0.14) * CLUMP_CELL);
  return (CLUMP_R_MIN
    + clumpSeed(ix, iz, 4) * (CLUMP_R_MAX - CLUMP_R_MIN)) * spread;
}

/**
 * One of the clumps around a point, chosen by the caller's own number.
 *
 * Deliberately not "the nearest". The nearest is a function of position
 * alone, so everything standing at that position picks the same clump
 * and keeps picking it forever — which is how a field of butterflies
 * ends up as a set of fixed rings. Handing in a value from 0 to 1 lets
 * the caller vary the answer by whatever it likes: by which animal is
 * asking, and by which leg of its journey it is on.
 *
 * Nine cells and not more. A grid four metres across means anything
 * worth flying to is in the ring immediately around you, and the case
 * the search genuinely fails — a bare patch tens of metres wide at low
 * density — is a case with nothing to fly to. Returning false rather
 * than widening the search is what keeps this a fixed cost per vertex.
 *
 * Two passes over the same nine cells: count, then take the wanted one.
 * The alternative is an array of nine candidates in a vertex shader,
 * which is real register pressure to avoid recomputing four sines.
 */
bool pickClump(vec2 p, float chance, float spread, float pick,
               out vec2 centre, out float radius) {
  int cx = int(floor(p.x / CLUMP_CELL));
  int cz = int(floor(p.y / CLUMP_CELL));

  centre = vec2(0.0);
  radius = 0.0;

  int n = 0;
  for (int dz = -1; dz <= 1; dz++) {
    for (int dx = -1; dx <= 1; dx++) {
      vec2 c;
      if (clumpAt(cx + dx, cz + dz, chance, spread, c) > 0.0) n++;
    }
  }
  if (n == 0) return false;

  int want = min(int(pick * float(n)), n - 1);
  int i = 0;
  for (int dz = -1; dz <= 1; dz++) {
    for (int dx = -1; dx <= 1; dx++) {
      vec2 c;
      float r = clumpAt(cx + dx, cz + dz, chance, spread, c);
      if (r <= 0.0) continue;
      if (i == want) { centre = c; radius = r; return true; }
      i++;
    }
  }
  return false;
}
`;
