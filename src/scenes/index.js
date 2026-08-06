/* ── scenes/index.js ─────────────────────────────────────────────────
   The scene registry. Order here is the order in the tab bar and the
   1–n keyboard shortcuts; with a single scene the tab bar hides itself.

   `index` is a catalogue number, not a position. 01 lives in
   ./archive/ (see its README to bring it back); 02 and 04 were removed
   and are recoverable from git:

     git show 979cc95:src/scenes/fluid.js    > src/scenes/fluid.js
     git show 979cc95:src/scenes/reaction.js > src/scenes/reaction.js
   ------------------------------------------------------------------ */

import march from './march.js';

export const SCENES = [march];

export const SCENE_BY_ID = new Map(SCENES.map((s) => [s.id, s]));

export const DEFAULT_SCENE = march.id;
