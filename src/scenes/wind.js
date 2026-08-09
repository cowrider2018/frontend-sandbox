/* ── scenes/wind.js ──────────────────────────────────────────────────
   The weather, as one function of position and time.

   It lives on its own because three things read it now — the grass, the
   flowers, and the trees — and a second copy of a wind field is the kind
   of thing that agrees for a week and then quietly stops. When it does,
   the symptom is not a bug anyone can point at: the meadow simply stops
   looking like one place, because a gust arrives at the flowers on a
   different beat from the grass they are standing in.

   Two terms, deliberately separable. The slow one is the gust itself
   crossing the field; the fast one is the chop inside it. Grass is light
   enough to feel both. A tree is not — a trunk with a tonne of timber in
   it does not answer a half-second flutter — so the trunks read only the
   slow term, and that difference in what each plant *filters out* is
   most of what makes a scale read correctly.

   Requires uTime and uWind to be declared by whoever includes it.
   ------------------------------------------------------------------ */

export const WIND_GLSL = /* glsl */`
/* Which way the weather is going. Fixed, because a wind that wanders is
   a wind nobody can read: what makes a gust legible is seeing it arrive
   from the same side every time. */
const vec2 WIND_DIR = vec2(0.8829, 0.4696);

/**
 * The long, slow wave: one gust crossing the field.
 *
 * Sampled at the plant's *base*, always, so a whole clump leans together
 * and the far side of the meadow is still standing up when the near side
 * has already been flattened.
 */
float slowGust(vec2 p) {
  return sin(dot(p, WIND_DIR) * 0.32 - uTime * 1.10);
}

/** The gust plus the chop inside it, normalised to 0..1. */
float gust(vec2 p) {
  float chop = sin(dot(p, WIND_DIR) * 1.55 - uTime * 2.85 + 1.7);
  return 0.5 + 0.5 * (slowGust(p) * 0.66 + chop * 0.34);
}
`;
