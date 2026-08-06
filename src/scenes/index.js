/* ── scenes/index.js ─────────────────────────────────────────────────
   The scene registry. Order here is the order in the tab bar and the
   1–n keyboard shortcuts; with a single scene the tab bar hides itself.

   `index` is a catalogue number, not a position. See /archive for the
   scenes and the agent that are no longer wired in, and for how to
   bring each of them back.
   ------------------------------------------------------------------ */

import march from './march.js';

export const SCENES = [march];

export const SCENE_BY_ID = new Map(SCENES.map((s) => [s.id, s]));

export const DEFAULT_SCENE = march.id;
