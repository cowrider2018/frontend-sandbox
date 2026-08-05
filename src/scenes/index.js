/* ── scenes/index.js ─────────────────────────────────────────────────
   The scene registry. Order here is the order in the tab bar, the
   numbering in the captions, and the 1–4 keyboard shortcuts.
   ------------------------------------------------------------------ */

import flow from './flow.js';
import fluid from './fluid.js';
import march from './march.js';
import reaction from './reaction.js';

export const SCENES = [flow, fluid, march, reaction];

export const SCENE_BY_ID = new Map(SCENES.map((s) => [s.id, s]));

export const DEFAULT_SCENE = SCENES[0].id;
