/* ── scenes/raster.js ────────────────────────────────────────────────
   The depth range shared by everything in this scene made of triangles.

   The marched image has no depth buffer and does not want one: it
   publishes the ray's travel distance in alpha and everything
   composites against that number. Triangles do need one, to sort
   against each other — and the cat and the ground cover draw into the
   same target, so they sort in the *same* buffer.

   Which is why this is one constant and not two copies. Two near/far
   pairs map the same world point to two different depths, and the
   symptom is not an obvious one: the cat's feet sink into the grass, or
   float a centimetre above it, depending on which projection happened
   to be steeper at that range.
   ------------------------------------------------------------------ */

export const RASTER_NEAR = 0.05;
export const RASTER_FAR = 200.0;
