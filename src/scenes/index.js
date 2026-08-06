/* ── scenes/index.js ─────────────────────────────────────────────────
   The scene registry. Order here is the order in the tab bar and the
   1–n keyboard shortcuts.

   `index` is a catalogue number, not a position: 01 is archived in
   ./archive/, and the remaining scenes keep the numbers they were
   built under so that "the 03 scene" keeps meaning what it has always
   meant. See ./archive/README.md to bring one back.
   ------------------------------------------------------------------ */

import fluid from './fluid.js';
import march from './march.js';
import reaction from './reaction.js';

export const SCENES = [march, fluid, reaction];

export const SCENE_BY_ID = new Map(SCENES.map((s) => [s.id, s]));

export const DEFAULT_SCENE = march.id;
