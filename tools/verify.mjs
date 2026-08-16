/* ── tools/verify.mjs ────────────────────────────────────────────────
   Zero-dependency visual acceptance harness.

   Serves the project over HTTP, drives a real headless Chrome through
   the DevTools Protocol using Node's own global WebSocket and fetch,
   loads every scene, captures a PNG of each, and reports any console
   error or uncaught exception that occurred along the way.

     node tools/verify.mjs                 # all scenes
     node tools/verify.mjs --scene fluid   # one scene
     node tools/verify.mjs --serve         # just the static server
     node tools/verify.mjs --head          # visible browser window

   No puppeteer, no playwright, no test runner. The protocol is JSON
   over a socket; that is the entire dependency surface.
   ------------------------------------------------------------------ */

import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { readFile, writeFile, mkdir, rm, readdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { extname, join, normalize, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = normalize(join(dirname(fileURLToPath(import.meta.url)), '..'));
const SHOTS = join(ROOT, 'tools', 'shots');

const args = process.argv.slice(2);
const flag = (name) => args.includes(`--${name}`);
const value = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : fallback;
};

const VIEWPORT = { width: 1600, height: 900 };

/**
 * Each shot is a hash route plus a settle time. Parameters go in the
 * URL because the app already treats the URL as its state — the same
 * mechanism that makes a scene shareable makes it testable.
 */
const SHOTS_PLAN = [
  { name: '01-intro',  hash: '#/march',                                   settle: 1400, intro: false },
  { name: '02-scene',  hash: '#/march',                                   settle: 8000 },
  { name: '03-full',   hash: '#/march?scale=1&steps=160&spin=0&taa=0.9',  settle: 9000 },
  // Deterministic reference: frozen clock, and every quality knob pinned
  // to what the pre-optimisation build had hard-coded. Two builds
  // rendering this must agree pixel for pixel.
  { name: '00-ref',    hash: '#/march?scale=1&spin=0&taa=0.9', settle: 4000, freeze: 8.0 },
  // `click` fires a burst at a sphere's centre and waits, so the ring has
  // travelled far enough across the surface to be visible in a still.
  { name: '04-impact', hash: '#/march?spin=0&scale=1&rippleAmp=0.05&rippleFreq=13',
    settle: 5000, click: true, after: 260 },
  { name: '05-ripple', hash: '#/march?spin=0&scale=1&rippleAmp=0.052&rippleFreq=12&rippleSpeed=0.95',
    settle: 5000, click: true, after: 700 },
  // Enough dissipation to open a gap in the ring and eat into a sphere.
  { name: '07-erode',  hash: '#/march?spin=0&scale=1&erode=0.4&rippleAmp=0.09&rippleLife=3',
    settle: 5000, click: true, after: 700 },

  // One impact, photographed at four points in its life. Clock frozen
  // and the age pinned, so these four frames differ by the age alone.
  ...[0.15, 0.35, 0.55, 0.78].map((age, i) => ({
    name: `08-heal-${'abcd'[i]}`,
    hash: '#/march?spin=0&scale=1&erode=0.42&rippleAmp=0.02&taa=0.9',
    settle: 3500,
    freeze: 8.0,
    click: true,
    after: 200,
    poke: `const s = __aether.scene; s._rippleAge[0] = ${age}; s.flash = 0; return true;`,
  })),
  { name: '06-lit',    hash: '#/march?light=0.18,0.62&tint=rose&ao=1&shadow=1', settle: 8000 },

  /* ── the cat ──
     Placed by hand with the clock frozen, so the gait, the blink and the
     idle tail drift cannot vary between runs. */
  { name: '09-cat',    hash: '#/march?spin=0&scale=1&taa=0.9', settle: 3000, freeze: 8.0,
    poke: 'const c = __aether.scene.cat; c.x = 1.7; c.z = 1.9; c.yaw = -2.35; return true;' },

  // The proof that the composite is a per-pixel depth comparison and not
  // a layer stacked on top: the cat stands *inside* the cluster, so the
  // spheres in front of it must cut into its silhouette while it hides
  // the ones behind. A layered composite would show a whole cat.
  { name: '10-cat-occlude', hash: '#/march?spin=0&scale=1&taa=0.9', settle: 3000, freeze: 8.0,
    poke: 'const c = __aether.scene.cat; c.x = 0.15; c.z = 0.1; c.yaw = -2.35; return true;' },

  /* Driven for real, through the app's own key handling — the clock has
     to run for these, so they are moving frames rather than fixed
     references. They prove the key routing, the gait and the follow
     camera in one go.

     `pre` aims the cat before it is driven. Left on its default heading
     it walks straight behind the cluster, which photographs as a cat
     that never moved — the staging has to keep the subject in shot. */
  { name: '11-cat-walk',   hash: '#/march?spin=0&scale=1&taa=0.9', settle: 2500,
    pre: 'const c = __aether.scene.cat; c.x = 0.4; c.z = 2.6; c.yaw = 1.9; return true;',
    hold: { key: 'w', ms: 1700 } },
  // Third person: the camera swings to `catYaw + π` and rides behind.
  { name: '12-cat-follow', hash: '#/march?spin=0&scale=1&taa=0.9&camera=follow', settle: 2500,
    pre: 'const c = __aether.scene.cat; c.x = -0.6; c.z = 3.0; c.yaw = 1.57; return true;',
    hold: { key: 'w', ms: 2200 } },
  // Close enough to read the shading: the smooth falloff across the
  // head, the broad sheen on the brow, and the ink — outline, eyes,
  // whiskers — still flat on top of it.
  { name: '13-cat-close', hash: '#/march?spin=0&scale=1&taa=0.9&camera=follow', settle: 2500,
    pre: 'const c = __aether.scene.cat; c.x = -0.6; c.z = 3.0; c.yaw = 1.57; return true;',
    hold: { key: 'w', ms: 2200 },
    poke: '__aether.scene.targetDist = 1.9; return true;' },

  /* The cat in the cluster's shadow, and the same cat on the sunlit side
     of it. The pair differs by the cat's position and nothing else.

     The sun is put near the horizon for this. With it high, a cat
     standing on the floor cannot get behind the cluster at all without
     standing inside it — there is barely a metre of clearance under the
     ring — so the only way to have the spheres between the animal and
     the light is to throw the shadow sideways. */
  ...[
    { name: '17-cat-nosun', shadow: 0 },
    { name: '17-cat-shadowed', shadow: 1 },
  ].map((s) => ({
    name: s.name,
    hash: `#/march?spin=0&scale=1&taa=0.9&shadow=${s.shadow}&light=0.68,0.95&camera=follow`,
    settle: 2200, freeze: 8.0,
    // Same cat, same camera, same light. The pair differs by whether the
    // shadow ray is fired at all, so the difference between the two
    // frames is the cluster blocking the sun and nothing else.
    poke: `const sc = __aether.scene, c = sc.cat;
           c.x = -2.25; c.z = -1.5; c.yaw = 1.9;
           sc.targetDist = 2.6; sc.yaw = 1.5; sc.pitch = 0.12;
           return true;`,
  })),

  /* The cat's own shadow, thrown long across the floor by a low sun and
     seen from above so it is the subject rather than a smudge under the
     feet. Nothing marches the mesh — what casts this is seven capsules
     hung off the same skeleton. */
  /* A sun halfway up, for the same reason the check uses one: high and
     the shadow hides under the cat, near the horizon and the floor has
     no light on it to lose. */
  { name: '18-cat-casts',
    hash: '#/march?spin=0&scale=1&taa=0.9&shadow=1&ao=1&light=0.68,0.667&camera=follow',
    settle: 2200, freeze: 8.0,
    poke: `const sc = __aether.scene, c = sc.cat;
           c.x = 4.2; c.z = 3.4; c.yaw = 1.2;
           sc.targetDist = 3.6; sc.yaw = 2.4; sc.pitch = 0.66;
           return true;` },

  /* Eye beams, mid-shot. The cat is aimed at the cluster and the
     trigger pulled through the scene's own path, so what is drawn is a
     real shot: two beams cut off where they meet a sphere, and the
     cluster already coming apart along the axis. */
  /* Clock frozen, or the shot is over before the shutter opens: a beam
     lives 0.22 s and the runner waits 1.4 s after poking. Frozen, the
     beam holds at full brightness and the blast is stepped by hand. */
  { name: '20-cat-laser', hash: '#/march?spin=0&scale=1&taa=0.35&camera=follow&shadow=1&erode=0.3',
    settle: 2200, freeze: 8.0,
    pre: `const s = __aether.scene, c = s.cat;
          c.x = 0; c.z = 4.2; c.yaw = Math.PI;      // stood off, facing the cluster
          s._setControlMode('look');
          document.exitPointerLock?.();
          /* Pitch chosen so the beam leaves level. The follow camera sits
             0.35 above what it aims at, so it looks *down* at the chest,
             and a shot down that axis passes under the cluster. */
          s.yaw = 0; s.pitch = -0.15; s.targetDist = 2.4;
          return true;`,
    /* Fire down the crosshair, then step aside to photograph it.
       A beam aimed along the view axis is end-on from the shooter's own
       eye — a dot behind the cat's head — so the shot is taken from
       across the line instead. The beam keeps the direction it was fired
       with; only the camera moves. */
    poke: `const s = __aether.scene;
           s._shootAlong(s.basis.fwd);
           s._setControlMode('camera');            // release the pinned yaw
           __aether.panel.setValues({ camera: 'orbit' });
           s.center.set([0, 0, 0]); s.target.set([0, 0.1, 0]);
           s.yaw = 1.65; s.pitch = 0.30; s.targetDist = 7.5;
           return true;` },

  /* The same weapon with the pointer free. The aim comes from the
     cursor, so the cat turns to face what was clicked and the beam runs
     from its eyes to that point — which, unlike a shot down the
     crosshair, is not along the view axis and so needs no camera trick
     to photograph. */
  { name: '21-cat-aimed', hash: '#/march?spin=0&scale=1&taa=0.35&shadow=1&camera=follow&erode=0.3',
    settle: 2200, freeze: 8.0,
    /* Staged first, fired second. Aiming reads the camera basis, and
       that is only rebuilt on the next frame — setting the camera and
       firing in one breath aims down the camera that was there before. */
    pre: `const s = __aether.scene, c = s.cat;
          c.x = 3.4; c.z = 3.4; c.yaw = 2.6;       // stood aside, facing away
          s._setControlMode('camera');
          document.exitPointerLock?.();
          s.yaw = 0.55; s.pitch = 0.34; s.targetDist = 8;
          return true;`,
    hold: { key: [], ms: 900 },                    // let the camera arrive
    /* Aim by projecting the cluster's centre back to a cursor position,
       rather than guessing a pixel: a guess that lands beside it falls
       through to the cursor's own ray and fires off into the sky. */
    poke: `const s = __aether.scene, b = s.basis, focal = 1.5;
           const rel = [-b.pos[0], -b.pos[1], -b.pos[2]];
           const vx = rel[0]*b.right[0] + rel[1]*b.right[1] + rel[2]*b.right[2];
           const vy = rel[0]*b.up[0]    + rel[1]*b.up[1]    + rel[2]*b.up[2];
           const vz = rel[0]*b.fwd[0]   + rel[1]*b.fwd[1]   + rel[2]*b.fwd[2];
           const aspect = innerWidth / innerHeight;
           s._trigger({
             x: (vx * focal / aspect / vz) * 0.5 + 0.5,
             y: 0.5 - (vy * focal / vz) * 0.5,
           });
           // The clock is frozen, so the turn is stepped by hand.
           s._aimTick(0.2);
           return true;` },

  /* ── the floor's three styles ──
     The grid is the reference surface and stays the default; these are
     what the other two look like. Frozen clock, because the wind is a
     function of time and a shot of a meadow taken at an arbitrary
     instant is not comparable with the last one. */
  { name: '22-grass', hash: '#/march?spin=0&scale=1&taa=0.9&ground=grass&cat=0',
    settle: 2500, freeze: 8.0,
    /* The previous shot fires a beam with the clock stopped, and a beam
       dies of nothing but time. Frozen, it hangs in the air across every
       frame that follows it. */
    pre: '__aether.scene.laser.silence(); return true;',
    poke: `const s = __aether.scene;
           s.yaw = 0.85; s.pitch = 0.06; s.targetDist = 6.5;
           return true;` },

  /* Down at eye level in the flowers. Low and close is the only angle
     that shows what the cover actually is: separate blades with their
     own silhouettes, leaning together, and clumps of flowers with
     stragglers scattered out of them. */
  { name: '23-meadow', hash: '#/march?spin=0&scale=1&taa=0.9&ground=grass&flowers=1&cat=0&cover=1',
    settle: 2500, freeze: 8.0,
    pre: '__aether.scene.laser.silence(); return true;',
    poke: `const s = __aether.scene;
           s.yaw = 0.85; s.pitch = -0.02; s.targetDist = 3.2;
           return true;` },

  /* A wood. Grown on the CPU into world chunks, expanded on the GPU:
     the trunk's lobed cross-section, branches thinning by Da Vinci's
     rule, and leaves whose silhouettes are drawn from a two-parameter
     family so no two are the same shape. */
  { name: '26-trees',
    hash: '#/march?spin=0&scale=1&taa=0.9&ground=grass&flowers=1&trees=1&cat=0&coverRadius=30',
    settle: 3000, freeze: 8.0,
    pre: '__aether.scene.laser.silence(); return true;',
    poke: `const s = __aether.scene;
           s.yaw = 0.85; s.pitch = 0.10; s.targetDist = 12;
           return true;` },

  /* Close in under a canopy, where the leaf shapes and the bark are
     actually resolvable — and with the cat standing in it for scale.
     The locked camera always orbits the origin, so the only way to frame
     something else is to put the cat there and follow it. */
  { name: '27-tree-close',
    hash: '#/march?spin=0&scale=1&taa=0.9&ground=grass&flowers=1&trees=1&camera=follow&shadow=1',
    settle: 3000, freeze: 8.0,
    pre: '__aether.scene.laser.silence(); return true;',
    poke: `const s = __aether.scene, c = s.cat;
           let best = null, bd = 1e9;
           for (const k of s.trees.canopies) {
             const d = Math.hypot(k[0], k[2]);
             if (d < bd) { bd = d; best = k; }
           }
           if (best) {
             c.x = best[0] + 1.6; c.z = best[2] + 1.2;
             c.yaw = Math.atan2(best[0] - c.x, best[2] - c.z);
           }
           s.yaw = 0.8; s.pitch = 0.30; s.targetDist = 4.2;
           return true;` },

  /* The wood's own shadow, with the sun low enough to throw it a long
     way across the meadow. This is a real projected shadow — the trees
     are drawn a second time from the sun into a depth map — so what
     should be on the ground is the canopy's actual outline, gappy, and
     not a blob under each trunk. */
  { name: '28-tree-shadow',
    hash: '#/march?spin=0&scale=1&taa=0.9&ground=grass&trees=1&shadow=1&camera=follow&light=0.68,0.34',
    settle: 3000, freeze: 8.0,
    pre: '__aether.scene.laser.silence(); return true;',
    poke: `const s = __aether.scene, c = s.cat;
           // Stand the cat a few metres downsun of the nearest tree, which
           // is where its shadow lands.
           let best = null, bd = 1e9;
           for (const k of s.trees.canopies) {
             const d = Math.hypot(k[0], k[2]);
             if (d < bd) { bd = d; best = k; }
           }
           if (best) {
             const l = s.lightDir;
             const drop = best[1] / Math.max(l[1], 0.2);
             c.x = best[0] - l[0] * drop; c.z = best[2] - l[2] * drop;
             c.yaw = Math.atan2(best[0] - c.x, best[2] - c.z);
           }
           s.yaw = 1.3; s.pitch = 0.42; s.targetDist = 7;
           return true;` },

  /* The horizon pushed right out. Fog used to be a constant that closed
     at 66 units, which is what made the world feel small; the master
     opens it and drags the cover's reach along behind it. */
  { name: '29-open-horizon',
    hash: '#/march?spin=0&scale=1&taa=0.9&ground=grass&flowers=1&trees=1&cat=0&visibility=200',
    settle: 3000, freeze: 8.0,
    pre: '__aether.scene.laser.silence(); return true;',
    poke: `const s = __aether.scene;
           s.yaw = 0.85; s.pitch = 0.09; s.targetDist = 14;
           return true;` },

  /* The ground with a shape to it, and the same ground without.
     The pair differs by the undulation control alone, so between them
     they say two things a single frame cannot. That the hills are
     geometry and not a texture: a real horizon line, real back slopes
     going dark, grass and trees carried up and over them. And that
     turning the control off restores exactly the floor this scene had
     before there was any — anything that shifts in `31` is a leak.

     The sun is put low on purpose. A hill under a high sun is a shape;
     a hill under a raking one throws its shadow the length of the field,
     and that shadow is the whole reason the ground is marched against
     its own slope bound rather than drawn into a map. If it ever stops
     working, it stops here first. */
  ...[
    { name: '30-hills', hills: 4.5 },
    { name: '31-hills-flat', hills: 0 },
  ].map((s) => ({
    name: s.name,
    hash: '#/march?spin=0&scale=1&taa=0.9&ground=grass&flowers=1&trees=1&cat=0'
        + `&visibility=200&light=0.68,0.10&hills=${s.hills}`,
    settle: 3200, freeze: 8.0,
    pre: '__aether.scene.laser.silence(); return true;',
    poke: `const s = __aether.scene;
           s.yaw = 0.85; s.pitch = 0.16; s.targetDist = 16;
           return true;` }),
  ),

  /* One meadow, four times of day, everything else held.
     The hour is the only thing that differs between these four, which is
     the claim: a time of day is one number in, and out come the light's
     direction, its colour, its strength and the sky's — not four
     controls that have to be set consistently with each other.

     What each is watching:
       06:30 sunrise. The light is red and *dim*, and the shadows are
         raked the length of the field. A dusk that only changed hue
         would fail here by being as bright as noon.
       12:00 the sun overhead. Shadows short, colour near white, and the
         hills lit on every face — which is the frame that shows the
         terrain still has a slope bound, since the snow shots and this
         one are marched by the same conservative step.
       19:00 after sunset. The sun is below the horizon and the moon has
         taken over from the opposite side; the sky has not gone out with
         it, because it is lit by air rather than by line of sight.
       00:00 the moon at its highest, and the stars. The one frame where
         the fireflies are what the picture is about.

     Every population is switched on in all four, at the same setting,
     and the hour is still the only difference — which is the second
     claim these have to carry now: what is *alive* changes with the
     light and not all of it the same way. Butterflies and birds in the
     middle two, neither at midnight, and the fireflies only where there
     is a night to find them in. Setting each shot's own creature amounts
     would hide exactly that. */
  ...[
    { name: '42-dawn',  hour: 6.5 },
    { name: '43-noon',  hour: 12.0 },
    { name: '44-dusk',  hour: 18.3 },
    { name: '45-night', hour: 0.0 },
  ].map((s) => ({
    name: s.name,
    hash: '#/march?spin=0&scale=1&taa=0.9&ground=grass&flowers=1&trees=1&cat=0'
        + '&visibility=200&hills=4.5&water=1&waterLevel=0.9&coverRadius=70'
        + '&butterflies=0.7&fireflies=1&sparrowFlocks=0.7'
        + `&daylight=hour&hour=${s.hour}`,
    settle: 3600, freeze: 8.0,
    pre: '__aether.scene.laser.silence(); return true;',
    poke: `const s = __aether.scene;
           s.yaw = 0.85; s.pitch = 0.06; s.targetDist = 22;
           return true;` }),
  ),

  /* The cat, out of its depth.
     Driven for real: `pre` puts it in the lake and W swims it further in,
     so what is photographed is the animal arriving in deep water rather
     than being posed there. The clock runs — the swim blend and the
     paddle are both time, and a frozen frame would catch a cat that had
     not noticed the water yet.

     Three things in one frame, and none of them is drawn by anything
     that knows about the other two: the legs are gone because the lake
     is nearer to the eye than they are, the body is riding the surface
     rather than the bed under it, and the rings are coming off the same
     bounding sphere the shadow is thrown from. */
  { name: '49-cat-swims',
    hash: '#/march?spin=0&scale=1&taa=0.9&ground=grass&hills=4.5&water=1'
        + '&waterLevel=1&coverRadius=50&visibility=140&reeds=1&wind=0.35'
        + '&camera=follow&daylight=hour&hour=11',
    settle: 2800,
    pre: `const c = __aether.scene.cat;
          c.x = -10.5; c.z = -7.5; c.yaw = 3.9; c.velocity = 0;
          __aether.scene.laser.silence(); return true;`,
    hold: { key: 'w', ms: 1500, after: 400 },
    poke: `const s = __aether.scene;
           s.targetDist = 4.0; s.pitch = 0.15;
           return true;` },

  /* The same midnight as 45, with weather in it — and the pair is the
     whole point, because the two frames differ by one control and what
     has to change is the *sky*. Before this, a snowstorm at midnight
     came with a full field of stars and a moon with a hard edge on it,
     which is the frame nobody would have gone looking for: everything
     under the sky had the weather and nothing in it did.

     What to look for: no stars, no disc, and the gradient gone flat —
     and the meadow below it still lit the way 45 lit it, because this
     changes what the sky *is* and not how much light comes out of it. */
  { name: '48-night-snow',
    hash: '#/march?spin=0&scale=1&taa=0.9&ground=grass&flowers=1&trees=1&cat=0'
        + '&visibility=200&hills=4.5&water=1&waterLevel=0.9&coverRadius=70'
        + '&butterflies=0.7&fireflies=1&sparrowFlocks=0.7'
        + '&daylight=hour&hour=0&weather=snow',
    settle: 3600, freeze: 8.0,
    pre: '__aether.scene.laser.silence(); return true;',
    poke: `const s = __aether.scene;
           s.yaw = 0.85; s.pitch = 0.06; s.targetDist = 22;
           return true;` },

  /* The meadow's population, close enough to see what each one is.
     40 is butterflies at full density from three metres away — two
     wings, no body, and every one of them at a different point in its
     stroke, which is the thing that has to be true for a field of them
     to read as alive rather than as a flock of identical toys.
     41 is fireflies, with the exposure wound down to where they are
     what is left in the frame. */
  /* The cluster is wound down to one small sphere in both. It hangs at
     the origin and the camera orbits it, so at the range these need it
     fills the frame and there is nowhere else to stand — and what is
     being photographed here is two wings, not the star. */
  /* Looking down over the meadow rather than across it, which is the
     only angle that shows the thing this is testing: the butterflies are
     gathered over the flower clumps and not scattered evenly, because
     they are reading the grid the flowers were sown from. Spread wound
     in and clump density down, so the clumps are separated enough for
     "gathered over" to be a visible claim rather than an assertion. */
  /* Two frozen instants, twelve seconds apart — longer than any one
     animal's leg, which runs five to ten seconds. Held at the same
     camera and the same everything, so what differs between them is the
     population having moved on.

     This is the pair that catches the failure the still above cannot.
     A butterfly locked to one clump makes a closed path, and a closed
     path is a thing the eye finds inside half a minute however unevenly
     it is walked — so the frames would come out with the same animals
     over the same flowers in slightly different places, and everything
     would look correct and be wrong. What should differ is *which*
     flowers are occupied, with a few caught out over the open grass
     between them and riding higher for it. */
  ...[
    { name: '40-butterflies', freeze: 8.0 },
    { name: '40b-butterflies-later', freeze: 20.0 },
  ].map((s) => ({
    name: s.name,
    hash: '#/march?spin=0&scale=1&taa=0.9&ground=grass&flowers=1&cat=0'
        + '&hills=2&butterflies=1&wind=0.4&balls=1&blend=0.02&displace=0'
        + '&flowerClumps=0.35&flowerSpread=0.7&coverRadius=26',
    settle: 3000, freeze: s.freeze,
    pre: '__aether.scene.laser.silence(); return true;',
    poke: `const s = __aether.scene;
           s.yaw = 0.85; s.pitch = 0.46; s.targetDist = 9.0;
           return true;` }),
  ),

  /* The sparrow, close enough to check the one thing that separates a
     bird from a cartoon of one: the head has to be smaller than the
     chest, and joined to it without a neck. The ring table says 0.64
     against a real bird's 0.65, but a table is not a silhouette — this
     is where that claim is actually looked at.

     Also the bounding flight. A sparrow climbs on every burst of
     wingbeats and falls with its wings shut between them, so a frozen
     frame should catch some of the population with wings spread and
     rising and the rest folded and dropping. All of them in the same
     pose means the fold and the climb have come apart. */
  { name: '46-sparrows',
    hash: '#/march?spin=0&scale=1&taa=0.9&ground=grass&cat=0'
        + '&hills=2&sparrows=1&sparrowFlocks=1&balls=1&blend=0.02&displace=0&visibility=140'
        + '&daylight=hour&hour=10.5&trees=1&coverRadius=40',
    settle: 3200, freeze: 8.0,
    pre: '__aether.scene.laser.silence(); return true;',
    poke: `const s = __aether.scene;
           s.yaw = 0.85; s.pitch = 0.62; s.targetDist = 13.0;
           return true;` },

  { name: '41-fireflies',
    hash: '#/march?spin=0&scale=1&taa=0.9&ground=grass&cat=0'
        + '&hills=2&fireflies=1&exposure=0.5&balls=1&blend=0.02&displace=0',
    settle: 3000, freeze: 8.0,
    pre: '__aether.scene.laser.silence(); return true;',
    poke: `const s = __aether.scene;
           s.yaw = 0.85; s.pitch = 0.09; s.targetDist = 5.5;
           return true;` },

  /* The three weathers, over one meadow with one lake in it.
     Held to the same camera, the same clock and the same everything
     else, because what is being compared is a mode and not a picture:
     37 is the scene as it has always been, and anything that shows up
     in it that was not there before belongs to a weather that is
     supposed to be switched off.

     What each of the other two is watching:
       rain — the ground goes darker and shinier, and there are streaks
         slanting the same way the grass is leaning. Slanting the wrong
         way means the drops and the blades have stopped reading the
         same wind field, which is the failure this arrangement exists
         to make visible.
       snow — white on the flats, bare on the steep faces, and stopping
         short of the water rather than out over it. The blades thin to
         stubble and the flowers are gone from the drifts but not from
         the scoured patches. */
  ...[
    { name: '37-clear', mode: 'clear' },
    { name: '38-rain',  mode: 'rain' },
    { name: '39-snow',  mode: 'snow' },
  ].map((s) => ({
    name: s.name,
    hash: '#/march?spin=0&scale=1&taa=0.9&ground=grass&flowers=1&trees=1&cat=0'
        + '&visibility=200&light=0.68,0.10&hills=4.5&water=1&waterLevel=0.9'
        + `&coverRadius=70&wind=0.7&weather=${s.mode}`,
    settle: 4000, freeze: 8.0,
    pre: '__aether.scene.laser.silence(); return true;',
    poke: `const s = __aether.scene;
           s.yaw = 0.85; s.pitch = 0.06; s.targetDist = 22;
           return true;` }),
  ),

  /* The shore, with the cover wound out far enough to reach it.
     The three shots above look at the lake from beyond the grass, which
     is exactly where the drowning cannot be seen. This is the one that
     watches it: blades taper out over the last few centimetres of dry
     ground and stop, flowers are simply absent, and no trunk stands in
     the water. What would show here and nowhere else is a shoreline that
     the planting and the marcher disagree about — grass standing on top
     of the surface, or a bare metre of soil around a lake that has not
     reached it yet. */
  { name: '36-lake-shore',
    hash: '#/march?spin=0&scale=1&taa=0.9&ground=grass&flowers=1&trees=1&cat=0'
        + '&visibility=200&light=0.68,0.10&hills=4.5&water=1&waterLevel=0.95'
        + '&coverRadius=70',
    settle: 4000, freeze: 8.0,
    pre: '__aether.scene.laser.silence(); return true;',
    poke: `const s = __aether.scene;
           s.yaw = 0.85; s.pitch = 0.06; s.targetDist = 22;
           return true;` },

  /* The shallows, which is the one thing every lake shot above is at the
     wrong angle to see.

     33 through 36 all look across the water from roughly eye height, and
     at that angle a lake is its reflection — which is the honest picture
     of a lake and exactly why none of them can photograph what is
     happening *under* the surface. This one leans over the edge: close,
     pitched down, and at a level that leaves a wide shelf of shallow
     water rather than a shoreline that drops away.

     Three things show here and nowhere else. The bed is displaced along
     the refracted ray, so it does not sit directly under the surface
     point — visible as an offset between where the shore reads on the
     bed and where the waterline actually is. The bed loses its fine
     grain before it loses its light, so there is a band of soft bed
     between the crisp shallows and the dark. And what is left of the bed
     at the far edge of visibility is blue-green, because red went first.

     Wind on, because the displacement is computed from the same normal
     the wave trains tilt: a still lake refracts, but it refracts by a
     constant, and a constant offset is not something the eye reads as
     water. The motion is the tell.

     The camera gets there by standing the cat there, which is the only
     way it can: the orbit rig eases its centre back to the origin every
     frame, so a poke that moves it is undone before the shot is taken.
     Follow mode aims at the animal, and the animal can be put anywhere.

     Where it is put was solved rather than eyeballed — (-2.5, -22.5) is
     0.21 m under at this level, and every point within a couple of
     metres of it is between 0.02 and 0.6, so the frame is a shelf and
     not a ledge. It is a wading depth, not a swimming one, which is
     what keeps the animal small in frame and the water the subject.

     Exposed up rather than lit differently, and the distinction matters
     if this shot is ever re-tuned. The other lake shots look *across*
     the water, where the low side-light is what puts a reflection on it.
     This one looks into it, and what it needs is not a different sun but
     more of the frame above the noise floor: the bed is the darkest
     surface in the scene, seen through a medium that is subtracting from
     it. Moving the light instead was tried and made the whole frame
     darker, the meadow included, which is the tell that the exposure and
     not the sun was the thing in the way.

     The clock runs, for 49's reason: a frozen frame catches a cat that
     has not noticed the water yet. */
  { name: '50-lake-shallows',
    hash: '#/march?spin=0&scale=1&taa=0.9&ground=grass&hills=4.5&water=1'
        + '&waterLevel=0.72&coverRadius=50&visibility=140&wind=0.5'
        + '&camera=follow&daylight=hour&hour=11&exposure=2.4',
    settle: 3200,
    pre: `const c = __aether.scene.cat;
          c.x = -2.5; c.z = -22.5; c.yaw = 2.2; c.velocity = 0;
          __aether.scene.laser.silence(); return true;`,
    poke: `const s = __aether.scene;
           s.pitch = 0.55; s.targetDist = 7;
           return true;` },

  /* The reflection, which is the other half of the surface and which no
     shot in this file could see either.

     50 looks straight down and is the right picture of what is under the
     water, which is exactly why it has no reflection in it at all: at
     that angle water's fresnel is 0.025, the weight lands under the skip
     threshold, and the bounce never fires. That is not a gap in the
     shot, it is what a lake looks like from above.

     So this is the same lake from the other extreme — eye almost on the
     surface, where the fresnel runs to nearly one and the water hands
     itself over to the mirror. The wind is off, which is not a special
     case but the honest limit: water is only rough because something is
     roughing it, and a still lake is a mirror. The trees are on because
     a reflection needs something to be a reflection *of*; sky alone
     reflected in water is just more sky.

     The animal is put at the deepest point rather than in the shallows,
     and for the reflection that is the whole difference. From the shore
     the only water low enough in frame to see is water with sky directly
     above it, and sky reflected in water is indistinguishable from
     water. Out here there is 12 to 16 metres of open surface in every
     direction, so whichever way the camera turns the far bank — and the
     trees standing on it — is what sits above the water and therefore
     what lands in it. Depth helps twice: past a metre or so the bed is
     gone entirely, and a dark surface is what a reflection shows on. */
  { name: '51-lake-mirror',
    hash: '#/march?spin=0&scale=1&taa=0.9&ground=grass&trees=1&hills=4.5'
        + '&water=1&waterLevel=0.72&coverRadius=50&visibility=140&wind=0'
        + '&camera=follow&daylight=hour&hour=11&exposure=1.8',
    settle: 3200,
    pre: `const c = __aether.scene.cat;
          c.x = -17.5; c.z = -14; c.yaw = 2.2; c.velocity = 0;
          __aether.scene.laser.silence(); return true;`,
    poke: `const s = __aether.scene;
           s.pitch = 0.03; s.targetDist = 6;
           return true;` },

  /* The reeds, which is the same shore as 36 with something living on
     it. Two things to look at, and both are about the band rather than
     about the stalks: it has to follow the waterline round every bay
     instead of ringing the lake at a fixed distance from it, and it has
     to stand *through* the surface — stems cut off by the water, not
     floating on it and not sunk under it. The second is drawn by
     nothing: the marcher owns the lake, this pass owns the reeds, and
     the composite keeps whichever is nearer.

     Wound out further than 36's camera so the far side of the lake is in
     frame too, because a band that only reads correctly on the near
     shore is a band that is following the camera. */
  { name: '47-reeds',
    hash: '#/march?spin=0&scale=1&taa=0.9&ground=grass&flowers=1&trees=1&cat=0'
        + '&visibility=200&light=0.68,0.10&hills=4.5&water=1&waterLevel=0.95'
        + '&coverRadius=70&reeds=1&wind=0.7',
    settle: 4000, freeze: 8.0,
    pre: '__aether.scene.laser.silence(); return true;',
    poke: `const s = __aether.scene;
           s.yaw = 0.85; s.pitch = 0.10; s.targetDist = 20;
           return true;` },

  /* The lake, at three levels and once without it.
     What these are watching is the shoreline, because the shoreline is
     the only part of the water nobody authored: it is wherever the hills
     cross the surface, so it has to move when the level moves and it has
     to stay put when nothing does. The pair to compare are 33 and 35 —
     same world, same camera, more water — and what should differ between
     them is the outline of the lake and nothing else about the ground.

     `32-lake-off` is the regression that matters most. Water switched
     off has to give back the frame the hills gave before there was any
     water at all, and the way that could quietly stop being true is a
     consumer left reading an unset uWaterY, which defaults to zero and
     reads as a lake standing above every ridge in the world. */
  ...[
    { name: '32-lake-off',  water: 0, level: 0.45 },
    { name: '33-lake',      water: 1, level: 0.45 },
    { name: '34-lake-low',  water: 1, level: 0.05 },
    { name: '35-lake-high', water: 1, level: 0.95 },
  ].map((s) => ({
    name: s.name,
    hash: '#/march?spin=0&scale=1&taa=0.9&ground=grass&flowers=1&trees=1&cat=0'
        + '&visibility=200&light=0.68,0.10&hills=4.5'
        + `&water=${s.water}&waterLevel=${s.level}`,
    settle: 3200, freeze: 8.0,
    pre: '__aether.scene.laser.silence(); return true;',
    poke: `const s = __aether.scene;
           s.yaw = 0.85; s.pitch = 0.16; s.targetDist = 16;
           return true;` }),
  ),

  /* Reach wound out to the end of its travel: grass to the fog, and the
     rim nowhere to be seen. The near cells are the same size they are at
     every other setting — what bought the distance is rings, not bigger
     plants, so the grass underfoot here should be indistinguishable from
     the default shot's. */
  { name: '25-grass-far',
    hash: '#/march?spin=0&scale=1&taa=0.9&ground=grass&cat=0&coverRadius=120',
    settle: 2500, freeze: 8.0,
    pre: '__aether.scene.laser.silence(); return true;',
    poke: `const s = __aether.scene;
           s.yaw = 0.85; s.pitch = 0.05; s.targetDist = 9;
           return true;` },

  /* And the point of sharing a depth buffer: the cat is in the grass,
     not on it — blades in front of its paws, blades behind them, and
     its own shadow lying across the sward. */
  { name: '24-meadow-cat',
    hash: '#/march?spin=0&scale=1&taa=0.9&ground=grass&flowers=1&camera=follow&shadow=1&ao=1&light=0.68,0.667',
    settle: 2500, freeze: 8.0,
    pre: '__aether.scene.laser.silence(); return true;',
    poke: `const sc = __aether.scene, c = sc.cat;
           c.x = 3.6; c.z = 3.2; c.yaw = 1.2;
           sc.targetDist = 2.2; sc.yaw = 2.4; sc.pitch = 0.20;
           return true;` },

  /* The three colourways. They share one set of vertices, one set of
     normals and one element array — only the palette differs — so these
     three frames are the same cat with a different buffer bound. */
  ...['orangin', 'tabby', 'calico'].map((skin) => ({
    name: `19-skin-${skin}`,
    hash: `#/march?spin=0&scale=1&taa=0.9&camera=follow&skin=${skin}`,
    settle: 2000, freeze: 8.0,
    poke: `const sc = __aether.scene, c = sc.cat;
           c.x = 3.0; c.z = 3.0; c.yaw = 0.35;
           sc.targetDist = 2.0; sc.yaw = 0.0; sc.pitch = 0.16;
           return true;`,
  })),

  /* Face on, with the head shaking. Camera-relative mode leaves the
     follow camera's heading free, so it can be parked on the nose while
     the cat turns underneath it — which keeps the whiskers' chain loaded
     and the face toward the lens at the same time. */
  { name: '16-cat-face', hash: '#/march?spin=0&scale=1&taa=0.3&camera=follow', settle: 2000,
    pre: `const s = __aether.scene, c = s.cat;
          c.x = 3.0; c.z = 3.0; s.targetDist = 1.7; s.pitch = 0.10;
          let stop = false;
          window.__stopShotHook = () => { stop = true; };
          const shake = () => {
            if (stop) return;
            c.yaw = 0.45 * Math.sin(performance.now() / 1000 * 7);
            s.yaw = 0;                  // camera parked on the nose
            requestAnimationFrame(shake);
          };
          requestAnimationFrame(shake);
          return true;`,
    poke: 'return true;' },

  // Mouse-look, sidling. The body is facing where it started and the
  // whole cat has moved across its own nose — the one thing A and D
  // cannot do in the default mode.
  { name: '15-cat-strafe', hash: '#/march?spin=0&scale=1&taa=0.9&camera=follow', settle: 2500,
    pre: `const c = __aether.scene.cat;
          c.x = 0.4; c.z = 3.0; c.yaw = 1.57;
          __aether.scene._setControlMode('look');
          document.exitPointerLock?.();
          return true;`,
    hold: { key: ['d'], ms: 1400 },
    poke: '__aether.scene.targetDist = 2.6; return true;' },

  /* Walking a continuous circle, so the tail's deflection is a steady
     state rather than a transient.

     Photographing the swing right after a turn does not work: the chain
     settles in about a third of a second and the temporal filter
     averages most of that away, so the shutter has to be luckier than a
     test should need to be. Turning without stopping holds the tail out
     at a fixed angle for as long as you like. The mouse is driven
     through the same `look()` the pointer would call. */
  // Low accumulation on purpose: a camera pinned behind a turning cat
  // sweeps the whole background, and at 0.9 the history smears it into
  // streaks that say nothing about the tail.
  { name: '14-cat-tail', hash: '#/march?spin=0&scale=1&taa=0.35&camera=follow', settle: 2500,
    pre: `const s = __aether.scene, c = s.cat;
          c.x = 0.2; c.z = 3.2; s.targetDist = 2.6;
          s._setControlMode('look');
          document.exitPointerLock?.();
          let stop = false;
          window.__stopShotHook = () => { stop = true; };
          const spin = () => {
            if (stop) return;                        // never outlives the shot
            c.look(0.030, 0);
            requestAnimationFrame(spin);
          };
          requestAnimationFrame(spin);
          return true;`,
    hold: { key: ['w'], ms: 1800, after: 0 } },
];

/**
 * CSS-pixel position of a sphere's centre. Aiming a synthetic click by
 * eye at "the middle of the screen" lands on empty sky often enough to
 * make a test lie about the feature it is checking.
 */
const BALL_ON_SCREEN = `(() => {
  const s = __aether.scene, b = s.basis, focal = 1.5;
  const rel = [s.ballPos[0] - b.pos[0], s.ballPos[1] - b.pos[1], s.ballPos[2] - b.pos[2]];
  const dot = (v) => rel[0] * v[0] + rel[1] * v[1] + rel[2] * v[2];
  const vx = dot(b.right), vy = dot(b.up), vz = dot(b.fwd);
  const aspect = s.width / s.height;
  return [
    ((vx * focal / aspect / vz) * 0.5 + 0.5) * innerWidth,
    (0.5 - (vy * focal / vz) * 0.5) * innerHeight,
  ];
})()`;

/* ═══ static server ═══════════════════════════════════════════════ */

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js':   'text/javascript; charset=utf-8',
  '.mjs':  'text/javascript; charset=utf-8',
  '.css':  'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg':  'image/svg+xml',
  '.png':  'image/png',
  '.ico':  'image/x-icon',
};

function serve() {
  const server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url, 'http://localhost');
      let path = decodeURIComponent(url.pathname);
      if (path.endsWith('/')) path += 'index.html';

      // Contain every request inside ROOT — this server is only ever
      // pointed at a local checkout, but path traversal is cheap to close.
      const file = normalize(join(ROOT, path));
      if (!file.startsWith(ROOT) || !existsSync(file)) {
        res.writeHead(404, { 'content-type': 'text/plain' });
        res.end('404');
        return;
      }

      const body = await readFile(file);
      res.writeHead(200, {
        'content-type': MIME[extname(file).toLowerCase()] ?? 'application/octet-stream',
        'cache-control': 'no-store',
      });
      res.end(body);
    } catch (err) {
      res.writeHead(500, { 'content-type': 'text/plain' });
      res.end(String(err));
    }
  });

  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port }));
  });
}

/* ═══ chrome ══════════════════════════════════════════════════════ */

function findChrome() {
  const candidates = [
    process.env.CHROME_PATH,
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
    join(process.env.LOCALAPPDATA ?? '', 'Google/Chrome/Application/chrome.exe'),
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
  ].filter(Boolean);
  return candidates.find((p) => existsSync(p));
}

async function launchChrome({ headless, gpu }) {
  const bin = findChrome();
  if (!bin) throw new Error('No Chrome or Edge found. Set CHROME_PATH.');

  const port = 9000 + Math.floor(Math.random() * 900);
  const profile = join(SHOTS, `.profile-${port}`);

  const flags = [
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${profile}`,
    `--window-size=${VIEWPORT.width},${VIEWPORT.height}`,
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-extensions',
    '--disable-background-timer-throttling',
    '--disable-backgrounding-occluded-windows',
    '--disable-renderer-backgrounding',
    // Headless Chrome will happily fall back to a software rasteriser,
    // which cannot run WebGL2 at all unless these are set.
    '--ignore-gpu-blocklist',
    '--enable-unsafe-swiftshader',
  ];

  // On a laptop with two GPUs, Chrome picks one for the whole browser
  // process. `powerPreference: 'high-performance'` on the context is a
  // hint the page makes *after* that choice has already been made, so it
  // cannot move you to the discrete card on its own.
  if (gpu === 'high') flags.push('--force-high-performance-gpu');
  if (gpu === 'low') flags.push('--force-low-power-gpu');

  flags.push('about:blank');
  if (headless) flags.unshift('--headless=new', '--hide-scrollbars');

  const child = spawn(bin, flags, { stdio: 'ignore', detached: false });

  // Poll the DevTools endpoint rather than guessing a startup delay.
  const deadline = Date.now() + 20000;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`http://127.0.0.1:${port}/json/version`);
      if (res.ok) return { child, port, profile, bin };
    } catch { /* not up yet */ }
    await sleep(150);
  }
  child.kill();
  throw new Error('Chrome did not expose a DevTools endpoint in time');
}

/* ═══ minimal CDP client ══════════════════════════════════════════ */

class CDP {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.pending = new Map();
    this.listeners = new Map();

    ws.addEventListener('message', (e) => {
      const msg = JSON.parse(e.data);
      if (msg.id !== undefined) {
        const p = this.pending.get(msg.id);
        if (!p) return;
        this.pending.delete(msg.id);
        msg.error ? p.reject(new Error(`${msg.error.message} (${msg.error.code})`)) : p.resolve(msg.result);
      } else {
        for (const fn of this.listeners.get(msg.method) ?? []) fn(msg.params);
      }
    });
  }

  static async attach(port) {
    const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    const page = targets.find((t) => t.type === 'page');
    if (!page) throw new Error('No page target');
    const ws = new WebSocket(page.webSocketDebuggerUrl);
    await new Promise((resolve, reject) => {
      ws.addEventListener('open', resolve, { once: true });
      ws.addEventListener('error', reject, { once: true });
    });
    return new CDP(ws);
  }

  send(method, params = {}) {
    const id = ++this.id;
    this.ws.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      setTimeout(() => {
        if (this.pending.delete(id)) reject(new Error(`CDP timeout: ${method}`));
      }, 60000);
    });
  }

  on(event, fn) {
    if (!this.listeners.has(event)) this.listeners.set(event, []);
    this.listeners.get(event).push(fn);
  }

  /** Evaluate in the page and return the value, unwrapping promises. */
  async eval(expression) {
    const { result, exceptionDetails } = await this.send('Runtime.evaluate', {
      expression,
      awaitPromise: true,
      returnByValue: true,
    });
    if (exceptionDetails) throw new Error(exceptionDetails.exception?.description ?? exceptionDetails.text);
    return result.value;
  }

  close() { this.ws.close(); }
}

/* ═══ run ═════════════════════════════════════════════════════════ */

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * Shaders live inside JS template literals, so a back-tick anywhere in
 * GLSL — including in a comment — silently ends the literal and turns
 * the rest of the shader into JavaScript. The failure surfaces as a
 * parse error on some unrelated identifier fifty lines later, which is
 * a genuinely bad way to spend ten minutes. This catches it in a
 * millisecond, before a browser is even launched.
 */
async function lintShaders() {
  const files = [];
  const walk = async (dir) => {
    for (const e of await readdir(dir, { withFileTypes: true })) {
      const p = join(dir, e.name);
      if (e.isDirectory()) await walk(p);
      else if (e.name.endsWith('.js')) files.push(p);
    }
  };
  await walk(join(ROOT, 'src'));

  const bad = [];
  for (const file of files) {
    const src = await readFile(file, 'utf8');
    const lines = src.split('\n');
    let inGlsl = false;
    lines.forEach((line, i) => {
      if (!inGlsl) {
        if (/\/\* glsl \*\/`/.test(line)) inGlsl = true;
        return;
      }
      // The closing line is a lone back-tick; anything else carrying one
      // is inside the shader and will end the literal early.
      if (/^\s*`\s*;?\s*$/.test(line)) { inGlsl = false; return; }
      if (line.includes('`')) {
        bad.push(`${file.slice(ROOT.length + 1)}:${i + 1}  ${line.trim()}`);
      }
    });
  }

  if (bad.length) {
    console.error('✗ back-tick inside a GLSL template literal:');
    for (const b of bad) console.error(`    ${b}`);
    throw new Error('shader lint failed');
  }
}

/**
 * The water level control, held to what it claims.
 *
 * A screenshot cannot check this one. The level is defined as *how much
 * of the open meadow stands under water*, and the depth that produces
 * that is read out of a distribution sampled from the wave table — so
 * the thing that can break is not the picture but the agreement between
 * the control's meaning and the ground's actual shape, which changes
 * the moment anyone edits a wavelength or a weight in `WAVES`. Nothing
 * about that would look wrong in a still; the lake would simply be a
 * different size than the number said.
 *
 * Measured independently of the table: sample the real height field on
 * a different lattice and count. Two samplings of the same ground, one
 * of which never sees the other's answer.
 */
async function checkWater() {
  const t = await import(new URL('../src/scenes/terrain.js', import.meta.url));

  const flooded = (y, hills) => {
    let wet = 0, total = 0;
    // Well outside the clearing, and on a stride sharing no factor with
    // the one the table itself was built on.
    for (let i = 0; i < 260; i++) {
      for (let j = 0; j < 260; j++) {
        total++;
        if (t.terrainHeight(70 + i * 1.13, -150 + j * 1.13, hills) < y) wet++;
      }
    }
    return wet / total;
  };

  const bad = [];
  for (const hills of [1.5, 4.5]) {
    for (const level of [0, 0.25, 0.5, 0.75, 1]) {
      const claimed = t.waterArea(level);
      const actual = flooded(t.waterSurfaceY(true, level, hills), hills);
      if (Math.abs(actual - claimed) > 0.03) {
        bad.push(`hills ${hills} level ${level}: claims `
          + `${(claimed * 100).toFixed(1)}% flooded, measures `
          + `${(actual * 100).toFixed(1)}%`);
      }
    }
  }

  /* And the two ways there is no water at all. The second matters more
     than it looks: a flat world has no low ground for a lake to sit in,
     and a surface that did not retreat out of reach would be coplanar
     with the floor everywhere. */
  if (t.waterSurfaceY(false, 1, 4.5) > t.TERRAIN_BASE - 4.5) {
    bad.push('water switched off still reaches the ground');
  }
  if (t.waterSurfaceY(true, 1, 0) > t.TERRAIN_BASE) {
    bad.push('water on a flat world still reaches the floor');
  }

  if (bad.length) {
    console.error('✗ water level does not mean what it says:');
    for (const b of bad) console.error(`    ${b}`);
    throw new Error('water check failed');
  }
}

/**
 * The two gaits, and which legs go together in each.
 *
 * A screenshot is the wrong instrument for this twice over: the legs of
 * a swimming cat are under the water and the water is nearer to the eye
 * than they are, so the one arrangement that matters most is the one
 * nothing can photograph. And the claim is about *phase* — a pair moving
 * together or against — which is a correlation over a cycle rather than
 * anything visible in a single frame.
 *
 * On the ground the pairs are diagonal, because that is what holds an
 * animal up. In the water the sides go together and the ends go
 * opposite, because that is what pulls one along. The driver is plain
 * arithmetic with no GL in it, so both can simply be run.
 */
async function checkGaits() {
  const { Driver, Sway, applyPose } = await import(new URL('../src/scenes/cat/pose.js', import.meta.url));
  const { parseCat, Rig } = await import(new URL('../src/scenes/cat/rig.js', import.meta.url));

  /* Correlation of two channels over a whole number of cycles, with each
     one's own mean taken out — the paddle carries a large rest offset
     and the question is only which way the *swing* goes. +1 is together,
     -1 is opposite. */
  const phasing = (swim) => {
    const d = new Driver();
    // Long enough for the smoothed speed and the swim blend to arrive.
    for (let i = 0; i < 400; i++) d.step(1 / 60, 1, 0, swim);
    const ch = { hipA: [], hipB: [], shoulderA: [], shoulderB: [],
                 tailWater: [], headPitch: [] };
    for (let i = 0; i < 600; i++) {
      const p = d.step(1 / 60, 1, 0, swim);
      for (const k of Object.keys(ch)) ch[k].push(p[k]);
    }
    const centre = (v) => {
      const m = v.reduce((s, x) => s + x, 0) / v.length;
      return v.map((x) => x - m);
    };
    const corr = (a, b) => {
      const x = centre(ch[a]), y = centre(ch[b]);
      const dot = x.reduce((s, v, i) => s + v * y[i], 0);
      const nx = Math.hypot(...x), ny = Math.hypot(...y);
      return dot / Math.max(nx * ny, 1e-9);
    };
    const swing = (k) => (Math.max(...ch[k]) - Math.min(...ch[k])) / 2;
    const rest = (k) => ch[k].reduce((s, x) => s + x, 0) / ch[k].length;
    /* Rising zero crossings of a centred channel — how many times round
       it went over the window. This is how the tail's turn is compared
       against the stroke's without either one's phase being known. */
    const cycles = (k) => {
      const v = centre(ch[k]);
      let n = 0;
      for (let i = 1; i < v.length; i++) if (v[i - 1] <= 0 && v[i] > 0) n++;
      return n;
    };
    return {
      hinds: corr('hipA', 'hipB'),
      ends: corr('hipA', 'shoulderA'),
      fronts: corr('shoulderA', 'shoulderB'),
      swing: swing('hipA'),
      rest: rest('hipA'),
      water: Math.abs(rest('tailWater')),
      headLift: [Math.min(...ch.headPitch), Math.max(...ch.headPitch)],
    };
  };

  const walk = phasing(0);
  const swim = phasing(1);
  const bad = [];

  // Diagonal: the two hind legs oppose each other, and each hind leg
  // moves with the front leg across from it.
  if (walk.hinds > -0.98) bad.push(`walking hind legs are not opposed (${walk.hinds.toFixed(3)})`);
  if (walk.ends < 0.98) bad.push(`walking diagonal is broken (${walk.ends.toFixed(3)})`);

  // Paddle: sides together, ends opposed.
  if (swim.hinds < 0.98) bad.push(`paddling hind legs are not together (${swim.hinds.toFixed(3)})`);
  if (swim.fronts < 0.98) bad.push(`paddling front legs are not together (${swim.fronts.toFixed(3)})`);
  if (swim.ends > -0.98) bad.push(`paddling ends are not opposed (${swim.ends.toFixed(3)})`);

  // And the two numbers that make it a paddle rather than a walk in
  // water: a much smaller travel, about a rest angle that has moved back.
  /* The travel is no longer what separates the two — a paddling cat works
     nearly as hard as a walking one — so this is a sanity band rather
     than the claim. What makes it a paddle is the three checks above and
     the rest angle below. */
  if (!(swim.swing > walk.swing * 0.5 && swim.swing < walk.swing * 1.2)) {
    bad.push(`the paddle's travel is off (${swim.swing.toFixed(3)} vs a stride of ${walk.swing.toFixed(3)})`);
  }

  /* The water reaches the tail, and only the water: on dry land every
     term it adds is skipped outright. */
  if (!(swim.water > 0.98)) bad.push(`the tail has not gone into the water (${swim.water.toFixed(3)})`);
  if (walk.water > 1e-6) bad.push(`a walking tail is in the water (${walk.water.toFixed(3)})`);

  /* And the head nods, upward. Both halves: it has to move, and it must
     never drop below where a swimming cat holds it — the oscillation is
     taken off a sine lifted into 0..1 for exactly that reason. */
  const [headLow, headHigh] = swim.headLift;
  if (!(headLow < headHigh - 0.03)) {
    bad.push(`the head is not nodding (${headLow.toFixed(3)}..${headHigh.toFixed(3)})`);
  }
  if (!(headHigh < walk.headLift[1] - 0.1)) {
    bad.push(`the head is not held up in the water `
      + `(${headHigh.toFixed(3)} against ${walk.headLift[1].toFixed(3)} ashore)`);
  }
  if (!(swim.rest < -0.3)) bad.push(`the paddling legs are not back under the body (${swim.rest.toFixed(3)})`);
  // A cat that stops paddling sinks, so the stroke has to survive
  // standing still — which a walk's does not.
  const idle = new Driver();
  for (let i = 0; i < 400; i++) idle.step(1 / 60, 0, 0, 1);
  let lo = 1e9, hi = -1e9;
  for (let i = 0; i < 300; i++) {
    const v = idle.step(1 / 60, 0, 0, 1).hipA;
    lo = Math.min(lo, v); hi = Math.max(hi, v);
  }
  if (hi - lo < 0.02) bad.push(`a floating cat has stopped paddling (${(hi - lo).toFixed(4)})`);

  /* ── and which way the tail curves ──
     The bake gives this tail a deep hook — nearly four tenths of its own
     length off the chord — and in the water it comes out the other way.
     Not straightened and not swung: reflected, which is the same turn
     that would straighten it, taken twice.

     What is compared is how the centreline *turns*, summed along it —
     the cross product of each pair of segments, which is the direction
     the line is bending in and does not care where the line as a whole
     has been pointed. The first attempt at this compared the bow against
     the chord from root to tip instead, and read 78°: the chord moves
     too, so the number was measuring the lay-back as much as the flip.
     This one reads 178.

     It is measured on the chain rather than in the numbers going in,
     because what has to be true is about the shape that comes out. And
     it cannot be photographed: the tail of a swimming cat is behind
     water that is nearer to the eye than it is. */
  /* The tail's rest centreline at its nodes, as pose.js measures it off
     the mesh. Kept in step with the table there: reading a shorter one
     against a longer chain silently compares the wrong nodes. */
  const bin = await readFile(join(ROOT, 'src', 'scenes', 'cat', 'cat.bin'));
  const cat = parseCat(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
  const meshRig = new Rig(cat.header);
  const tailBone = meshRig.bone('tail');
  const skin = cat.colors.get(cat.header.skins[0]);

  const AXIS = [
    [0.0143, 0.0418, -0.0334], [0.0143, 0.1674, -0.1336], [0.0143, 0.3348, -0.2673],
    [0.0143, 0.4601, -0.3679], [0.0143, 0.6222, -0.5078], [0.0140, 0.8239, -0.6916],
    [0.0140, 0.9426, -0.7806], [0.0141, 1.0988, -0.8711], [0.0143, 1.2850, -0.9439],
    [0.0140, 1.5485, -0.9984], [0.0140, 1.7333, -0.9876], [0.0140, 1.8596, -0.9510],
    [0.0141, 2.0868, -0.8062], [0.0139, 2.2143, -0.6736], [0.0140, 2.2832, -0.5405],
    [0.0142, 2.3377, -0.3679], [0.0014, 2.3985, -0.1488],
  ];

  const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const len3 = (a) => Math.hypot(a[0], a[1], a[2]);
  const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1],
                            a[2] * b[0] - a[0] * b[2],
                            a[0] * b[1] - a[1] * b[0]];

  /** Settle the tail in or out of the water, and report the shape of it:
      where its nodes are, which way it turns, and how deep the bow is. */
  const tailShape = (wet, seconds = 22) => {
    const drv = new Driver();
    const chain = new Sway();
    let clock = 0;
    const lift = [];
    for (let i = 0; i < Math.round(seconds * 60); i++) {
      const pose = drv.step(1 / 60, 0, 0, wet);
      clock += 1 / 60;
      chain.step(1 / 60, clock, 0, pose.bodyPitch, pose);
      // Past the first couple of seconds only: the tail is still going
      // into the water before that, and the range would be the ramp
      // rather than the drift.
      if (clock > 2) lift.push(chain.bend[(AXIS.length - 1) * 3 + 1]);
    }
    const P = AXIS.map((a, i) => [
      a[0] + chain.bend[i * 3], a[1] + chain.bend[i * 3 + 1], a[2] + chain.bend[i * 3 + 2],
    ]);
    let turn = [0, 0, 0];
    for (let i = 1; i < P.length - 1; i++) {
      const c = cross3(sub3(P[i], P[i - 1]), sub3(P[i + 1], P[i]));
      turn = [turn[0] + c[0], turn[1] + c[1], turn[2] + c[2]];
    }
    const chord = sub3(P[P.length - 1], P[0]);
    const u = chord.map((v) => v / len3(chord));
    const rel = sub3(P[Math.floor(P.length / 2)], P[0]);
    const along = dot3(rel, u);
    const bow = [rel[0] - u[0] * along, rel[1] - u[1] * along, rel[2] - u[2] * along];
    return { base: P[1], turn, bow: len3(bow),
             float: Math.max(...lift) - Math.min(...lift) };
  };

  const dry = tailShape(0);
  const wet = tailShape(1);
  const flipped = Math.acos(Math.max(-1, Math.min(1,
    dot3(dry.turn, wet.turn) / Math.max(len3(dry.turn) * len3(wet.turn), 1e-9)))) * 180 / Math.PI;
  const moved = len3(sub3(wet.base, dry.base));

  /* How much of the curve the water takes out, as a signed amount along
     the direction the dry tail bends in: the same number covers the
     whole travel of the knob — none of it, all of it (a straight tail),
     or past it (the curve carried over the other way) — so the check
     does not have to be rewritten every time the constant is turned. */
  const dryTurn = len3(dry.turn);
  const signed = dot3(wet.turn, dry.turn) / Math.max(dryTurn, 1e-9);
  if (!(signed < dryTurn * 0.5)) {
    bad.push(`the water does not take the curve out of the tail `
      + `(${signed.toFixed(2)} left of ${dryTurn.toFixed(2)})`);
  }
  /* The pin: the water's terms grow from the first node outward, so the
     point where the tail leaves the rump does not move when it goes in.
     A tail turned at its root moves it by a third of a unit and drags
     the buried part of itself through the animal. */
  if (!(moved < 0.01)) {
    bad.push(`the tail's base moves when it goes in (${moved.toFixed(3)} units)`);
  }
  /* ── and none of it turns the surface inside out ──
     The one thing a bend must never do. A ring of radius r carried along
     a line that turns dθ over a length ds sweeps its inside edge back by
     r·dθ while the line goes forward by ds, so past r·dθ/ds = 1 the
     inside of the bend travels further back than the segment travels
     forward and the mesh passes through itself. pose.js holds every
     segment under that ceiling; this is the check that the ceiling is
     doing its job, taken on the tail's own triangles rather than on the
     numbers going in.

     A triangle is compared against its own rest normal *carried by the
     same frame*, which is the part that took three tries to get right.
     Compared against the rest normal where it lay, every triangle in the
     stretch where the tail has turned through more than a right angle
     reports itself inside-out — 353 of them, in a band that moved with
     nothing and stayed put through four different deformations, because
     it was a property of the measurement. */
  const foldSurvey = (wet) => {
    const drv = new Driver();
    const chain = new Sway();
    let clock = 0;
    for (let i = 0; i < 400; i++) {
      const pose = drv.step(1 / 60, 0, 0, wet);
      clock += 1 / 60;
      chain.step(1 / 60, clock, 0, pose.bodyPitch, pose);
    }
    const qAt = (o) => {
      const k = Math.min(Math.round(Math.min(o, 1) * (AXIS.length - 1)), AXIS.length - 1);
      return [chain.qs[k * 4], chain.qs[k * 4 + 1], chain.qs[k * 4 + 2], chain.qs[k * 4 + 3]];
    };
    const spin = (q, v) => {
      const t = [2 * (q[1] * v[2] - q[2] * v[1]),
                 2 * (q[2] * v[0] - q[0] * v[2]),
                 2 * (q[0] * v[1] - q[1] * v[0])];
      return [v[0] + q[3] * t[0] + q[1] * t[2] - q[2] * t[1],
              v[1] + q[3] * t[1] + q[2] * t[0] - q[0] * t[2],
              v[2] + q[3] * t[2] + q[0] * t[1] - q[1] * t[0]];
    };
    // The shader's own skinning: placed by each of its two nodes, blended.
    const move = (p, o) => {
      const x = Math.min(o, 1) * (AXIS.length - 1);
      const i = Math.floor(x), t = x - i;
      const at = (k) => {
        const kk = Math.min(k, AXIS.length - 1);
        const a = AXIS[kk];
        const r = spin([chain.qs[kk * 4], chain.qs[kk * 4 + 1],
                        chain.qs[kk * 4 + 2], chain.qs[kk * 4 + 3]], sub3(p, a));
        return [r[0] + a[0] + chain.bend[kk * 3],
                r[1] + a[1] + chain.bend[kk * 3 + 1],
                r[2] + a[2] + chain.bend[kk * 3 + 2]];
      };
      const A = at(i), B = at(i + 1);
      return [A[0] + (B[0] - A[0]) * t, A[1] + (B[1] - A[1]) * t, A[2] + (B[2] - A[2]) * t];
    };

    let inside = 0, seen = 0;
    for (let k = 0; k < cat.index.length; k += 3) {
      const idx = [cat.index[k], cat.index[k + 1], cat.index[k + 2]];
      if ((skin[idx[0] * 4 + 3] & 31) !== tailBone) continue;
      const P = idx.map((v) => [cat.position[v * 3], cat.position[v * 3 + 1], cat.position[v * 3 + 2]]);
      const O = idx.map((v) => cat.normal[v * 4 + 3] / 32767);
      const n0 = cross3(sub3(P[1], P[0]), sub3(P[2], P[0]));
      if (len3(n0) < 1e-9) continue;
      const M = P.map((v, j) => move(v, O[j]));
      const n1 = cross3(sub3(M[1], M[0]), sub3(M[2], M[0]));
      if (len3(n1) < 1e-12) continue;
      const carried = spin(qAt((O[0] + O[1] + O[2]) / 3), n0);
      seen++;
      if (dot3(carried, n1) < 0) inside++;
    }
    return { inside, seen };
  };

  const foldDry = foldSurvey(0);
  const foldWet = foldSurvey(1);
  if (foldDry.inside || foldWet.inside) {
    bad.push(`the tail folds through itself: ${foldDry.inside} triangles on land, `
      + `${foldWet.inside} in the water, of ${foldWet.seen}`);
  }

  /* And it drifts while it lies there. A floating tail that held one
     line would read as a prop rather than as something in water. */
  if (!(wet.float > 0.10)) {
    bad.push(`the tail lies dead still on the water (${wet.float.toFixed(3)})`);
  }

  if (bad.length) {
    console.error('✗ the gait does not pair the legs as claimed:');
    for (const b of bad) console.error(`    ${b}`);
    throw new Error('gait check failed');
  }
  console.log(`▸ gaits:    walk diagonal ${walk.ends.toFixed(2)} · `
    + `paddle sides ${swim.hinds.toFixed(2)}, ends ${swim.ends.toFixed(2)}, `
    + `travel x${(swim.swing / walk.swing).toFixed(2)} · `
    + `tail curve ${dryTurn.toFixed(2)} on land to ${signed.toFixed(2)} in the water, `
    + `bow ${dry.bow.toFixed(2)} to ${wet.bow.toFixed(2)}, base fixed to ${moved.toFixed(3)}, `
    + `float ${wet.float.toFixed(2)}, nothing inside-out of ${foldWet.seen}`);
}

async function main() {
  await lintShaders();
  await checkWater();
  await checkGaits();
  const { server, port } = await serve();
  const base = `http://127.0.0.1:${port}/index.html`;
  console.log(`▸ serving ${ROOT}\n  ${base}`);

  if (flag('serve')) {
    console.log('  (--serve: press Ctrl+C to stop)');
    return;
  }

  // Only wipe the gallery on a full scene run. Every other mode is an
  // iteration step, and blowing away the other shots each time makes
  // comparing them impossible.
  const isFullRun = !value('scene', null) && !value('eval', null)
    && !flag('interact') && !flag('responsive');
  if (isFullRun) {
    await rm(SHOTS, { recursive: true, force: true }).catch(() => {});
  }
  await mkdir(SHOTS, { recursive: true });

  const chrome = await launchChrome({ headless: !flag('head'), gpu: value('gpu', null) });
  console.log(`▸ chrome ${chrome.bin.split(/[\\/]/).pop()} on :${chrome.port}`);

  const cdp = await CDP.attach(chrome.port);
  cdp.__port = port;          // so bench() can build its own URLs
  const problems = [];

  cdp.on('Runtime.consoleAPICalled', ({ type, args }) => {
    if (type !== 'error' && type !== 'warning') return;
    const text = args.map((a) => a.value ?? a.description ?? a.type).join(' ');
    problems.push({ kind: type, text });
  });
  cdp.on('Runtime.exceptionThrown', ({ exceptionDetails }) => {
    problems.push({
      kind: 'exception',
      text: exceptionDetails.exception?.description ?? exceptionDetails.text,
    });
  });
  cdp.on('Log.entryAdded', ({ entry }) => {
    if (entry.level === 'error') problems.push({ kind: `log/${entry.source}`, text: entry.text });
  });

  await cdp.send('Page.enable');
  await cdp.send('Runtime.enable');
  await cdp.send('Log.enable');
  await cdp.send('Emulation.setDeviceMetricsOverride', {
    ...VIEWPORT, deviceScaleFactor: 1, mobile: false,
  });

  // --eval: load one route, settle, then run an expression in the page.
  // The quickest way to interrogate live GPU state from here.
  let probe = value('eval', null);
  if (probe && existsSync(probe)) probe = await readFile(probe, 'utf8');
  if (probe) {
    const hash = value('at', '#/flow');
    await cdp.send('Page.navigate', { url: base + hash });
    await waitFor(cdp, `document.documentElement.dataset.boot`, 'ready', 90000);
    await cdp.eval(`document.getElementById('intro-enter').click(); true`);
    await sleep(Number(value('settle', 3000)));
    console.dir(await cdp.eval(`(() => { ${probe} })()`), { depth: 8 });
    for (const p of problems) console.log(`  ${p.kind}: ${truncate(p.text, 600)}`);
    cdp.close(); chrome.child.kill(); server.close();
    await sleep(200);
    process.exit(0);
  }

  if (flag('diff')) {
    await pixelDiff(cdp);
    cdp.close(); chrome.child.kill(); server.close();
    await sleep(200);
    process.exit(0);
  }

  if (flag('bench')) {
    await bench(cdp);
    cdp.close(); chrome.child.kill(); server.close();
    await sleep(200);
    process.exit(0);
  }

  if (flag('responsive')) {
    // The layout has to survive a phone as well as a desktop, and the
    // pixel budget has to survive a 3× device ratio.
    const viewports = [
      { name: 'r1-phone',   width: 390,  height: 844,  dpr: 3, mobile: true },
      { name: 'r2-tablet',  width: 820,  height: 1180, dpr: 2, mobile: true },
      { name: 'r3-laptop',  width: 1280, height: 720,  dpr: 2, mobile: false },
      { name: 'r4-wide',    width: 2560, height: 1080, dpr: 1, mobile: false },
    ];
    console.log('▸ responsive');
    await cdp.send('Page.navigate', { url: `${base}#/reaction` });
    await waitFor(cdp, 'document.documentElement.dataset.boot', 'ready', 90000);
    await cdp.eval(`document.getElementById('intro-enter').click(); true`);

    for (const v of viewports) {
      await cdp.send('Emulation.setDeviceMetricsOverride', {
        width: v.width, height: v.height, deviceScaleFactor: v.dpr, mobile: v.mobile,
      });
      await sleep(2600);
      const info = await cdp.eval(`(() => {
        const c = document.getElementById('stage');
        return {
          buffer: c.width + '×' + c.height,
          pixels: c.width * c.height,
          bodyScrollX: document.documentElement.scrollWidth > document.documentElement.clientWidth,
          panelVisible: getComputedStyle(document.getElementById('panel')).opacity,
          fps: document.getElementById('hud-fps').textContent,
        };
      })()`);
      await shot(cdp, v.name);
      // The app clamps DPR down to 1 but never below, so a very wide
      // 1× display legitimately exceeds the nominal pixel budget.
      const budgetOk = info.pixels <= Math.max(2.7e6, v.width * v.height * 1.02);
      console.log(`  ${budgetOk && !info.bodyScrollX ? '✓' : '✗'} ${v.name.padEnd(10)}` +
        ` css ${v.width}×${v.height}@${v.dpr}x → buffer ${info.buffer}` +
        ` (${(info.pixels / 1e6).toFixed(2)} Mpx)` +
        (info.bodyScrollX ? '  ⚠ horizontal scroll' : '') +
        (budgetOk ? '' : '  ⚠ over pixel budget'));
    }
    cdp.close(); chrome.child.kill(); server.close();
    await sleep(200);
    process.exit(0);
  }

  if (flag('interact')) {
    await interact(cdp, base, problems);
    cdp.close(); chrome.child.kill(); server.close();
    await sleep(200);
    process.exit(0);
  }

  const only = value('scene', null);
  const plan = only ? SHOTS_PLAN.filter((s) => s.hash.includes(only)) : SHOTS_PLAN;
  const report = [];

  for (const shot of plan) {
    const url = base + shot.hash;
    process.stdout.write(`▸ ${shot.name.padEnd(12)} `);

    const before = problems.length;
    await cdp.send('Page.navigate', { url });
    try {
      await waitFor(cdp, `document.documentElement.dataset.boot`, 'ready', 90000);
    } catch (err) {
      // A boot that never completes is almost always a module or shader
      // error the page already logged — surface it instead of the timeout.
      console.log('✗ boot failed');
      for (const p of problems) console.log(`     ${p.kind}: ${truncate(p.text, 1400)}`);
      throw err;
    }

    if (shot.intro !== false) {
      await cdp.eval(`document.getElementById('intro-enter').click(); true`);
    }
    // Freeze simulation time at a fixed value and let the temporal
    // filter converge. Rendering keeps going while the clock is paused,
    // so this produces a deterministic image that two builds can be
    // compared pixel for pixel — a settle-time-only shot drifts by a few
    // frames of animation and makes any comparison guesswork.
    if (shot.freeze !== undefined) {
      await cdp.eval(`(() => {
        const c = __aether.clock;
        c.paused = true;
        c.time = ${shot.freeze};
        return true;
      })()`);
      await sleep(1600);
    } else {
      await sleep(shot.settle);
    }

    if (shot.click) {
      const at = await cdp.eval(BALL_ON_SCREEN);
      const ev = (type, buttons) => cdp.send('Input.dispatchMouseEvent', {
        type, x: at[0], y: at[1], button: 'left', clickCount: 1, buttons,
      });
      await ev('mouseMoved', 0);
      await sleep(80);
      await ev('mousePressed', 1);
      await sleep(60);
      await ev('mouseReleased', 0);
      await sleep(shot.after ?? 600);
    }

    /* Kill any per-frame hook a previous shot installed. Shots share one
       page — only the hash changes — so a `requestAnimationFrame` loop
       set up to drive one of them keeps running into the next and
       quietly steers its camera. That is a scaffolding bug that looks
       exactly like a rendering bug. */
    await cdp.eval(`window.__stopShotHook?.(); window.__stopShotHook = null; true`);

    // Stage the scene before it is driven — where the subject starts
    // decides whether it is still in frame when the shutter opens.
    if (shot.pre) {
      await cdp.eval(`(() => { ${shot.pre} })()`);
      await sleep(120);
    }

    // Hold a key down, let the scene act on it, release. Synthesising
    // the key rather than poking the scene's state is the point: it is
    // the only way to prove the app actually routes the key to the scene
    // instead of eating it as a shortcut.
    if (shot.hold) {
      const { key, ms, after = 400 } = shot.hold;
      // Several at once, so a shot can walk and steer in the same frame
      // — which is the only way to photograph anything that depends on
      // the body turning.
      const keys = Array.isArray(key) ? key : [key];
      const ev = (type, k) => cdp.send('Input.dispatchKeyEvent', {
        type,
        key: k,
        code: `Key${k.toUpperCase()}`,
        windowsVirtualKeyCode: k.toUpperCase().charCodeAt(0),
        nativeVirtualKeyCode: k.toUpperCase().charCodeAt(0),
      });
      for (const k of keys) await ev('keyDown', k);
      await sleep(ms);
      for (const k of keys) await ev('keyUp', k);
      await sleep(after);
    }

    // Reach into the scene and pin state that time would otherwise carry
    // past. With the clock frozen an impact stays at whatever age you
    // set, which is the only way to photograph one stage of an animation
    // and compare it against another.
    if (shot.poke) {
      await cdp.eval(`(() => { ${shot.poke} })()`);
      await sleep(1400);
    }

    const diag = await cdp.eval(`(() => {
      const c = document.getElementById('stage');
      const gl = c.getContext('webgl2');
      return {
        boot: document.documentElement.dataset.boot,
        scene: document.documentElement.dataset.scene,
        canvas: c.width + '×' + c.height,
        fps: document.getElementById('hud-fps').textContent,
        caption: document.getElementById('caption-title').textContent,
        renderer: (() => { const d = gl.getExtension('WEBGL_debug_renderer_info');
          return d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER); })(),
        glError: gl.getError(),
      };
    })()`);

    const { data } = await cdp.send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false });
    const file = join(SHOTS, `${shot.name}.png`);
    await writeFile(file, Buffer.from(data, 'base64'));

    // A canvas that is entirely one colour has not rendered anything.
    const stats = await cdp.eval(`(() => {
      const c = document.getElementById('stage');
      const s = document.createElement('canvas');
      s.width = 64; s.height = 36;
      const x = s.getContext('2d');
      x.drawImage(c, 0, 0, 64, 36);
      const d = x.getImageData(0, 0, 64, 36).data;
      let min = 255, max = 0, sum = 0, n = 0;
      for (let i = 0; i < d.length; i += 4) {
        const l = (d[i] * 0.299 + d[i+1] * 0.587 + d[i+2] * 0.114);
        min = Math.min(min, l); max = Math.max(max, l); sum += l; n++;
      }
      return { min: Math.round(min), max: Math.round(max), mean: +(sum / n).toFixed(1) };
    })()`);

    const newProblems = problems.slice(before);
    const blank = stats.max - stats.min < 4;
    const ok = !blank && newProblems.length === 0 && diag.glError === 0;

    report.push({ ...shot, diag, stats, problems: newProblems, ok, file });
    console.log(
      `${ok ? '✓' : '✗'}  ${diag.canvas}  fps ${String(diag.fps).padStart(3)}  ` +
      `luma ${String(stats.min).padStart(3)}–${String(stats.max).padEnd(3)} (μ${stats.mean})` +
      (blank ? '  ⚠ BLANK' : '') +
      (diag.glError ? `  ⚠ glError ${diag.glError}` : '') +
      (newProblems.length ? `  ⚠ ${newProblems.length} problem(s)` : '')
    );
    for (const p of newProblems) console.log(`     ${p.kind}: ${truncate(p.text, 260)}`);
  }

  const renderer = report[0]?.diag.renderer ?? 'unknown';
  console.log(`\n▸ renderer: ${renderer}`);
  console.log(`▸ shots:    ${SHOTS}`);
  const failed = report.filter((r) => !r.ok);
  console.log(failed.length ? `▸ FAILED:   ${failed.map((r) => r.name).join(', ')}` : '▸ all scenes rendered clean');

  cdp.close();
  chrome.child.kill();
  server.close();
  await sleep(300);
  process.exit(failed.length ? 1 : 0);
}

/* ═══ pixel diff ══════════════════════════════════════════════════
   "You will not see the difference" is an opinion until someone
   measures it. This freezes simulation time, renders the same frame
   with one setting flipped, and reports what actually changed.        */

const DIFF_CASES = [
  {
    name: 'shadowNoise',
    label: '陰影含表面擾動',
    base: { shadowNoise: false },
    alt: { shadowNoise: true },
  },
  {
    name: 'reflectLit',
    label: '反射含陰影遮蔽',
    base: { reflectLit: false },
    alt: { reflectLit: true },
  },
  // Where does each budget stop buying anything? Every case is measured
  // against a reference well past the point of diminishing returns.
  { name: 'shadow-12', label: '陰影步數 12 對 48', base: { shadowSteps: 12 }, alt: { shadowSteps: 48 } },
  { name: 'shadow-20', label: '陰影步數 20 對 48', base: { shadowSteps: 20 }, alt: { shadowSteps: 48 } },
  { name: 'shadow-32', label: '陰影步數 32 對 48', base: { shadowSteps: 32 }, alt: { shadowSteps: 48 } },
  { name: 'ao-3',      label: '遮蔽取樣 3 對 8',   base: { aoTaps: 3 },       alt: { aoTaps: 8 } },
  { name: 'ao-5',      label: '遮蔽取樣 5 對 8',   base: { aoTaps: 5 },       alt: { aoTaps: 8 } },
  { name: 'reflect-24', label: '反射步數 24 對 90', base: { reflectSteps: 24 }, alt: { reflectSteps: 90 } },
  { name: 'reflect-48', label: '反射步數 48 對 90', base: { reflectSteps: 48 }, alt: { reflectSteps: 90 } },
  { name: 'steps-70',  label: '行進步數 70 對 190', base: { steps: 70 },      alt: { steps: 190 } },
  { name: 'steps-100', label: '行進步數 100 對 190', base: { steps: 100 },    alt: { steps: 190 } },
];

/**
 * Read the canvas back at a reduced size. Sampling the real drawing
 * buffer rather than a PNG keeps this dependency-free and avoids
 * decoding anything.
 */
const GRAB = `(() => {
  const c = document.getElementById('stage');
  const s = document.createElement('canvas');
  s.width = 320; s.height = 180;
  const x = s.getContext('2d');
  x.drawImage(c, 0, 0, 320, 180);
  return [...x.getImageData(0, 0, 320, 180).data];
})()`;

async function pixelDiff(cdp) {
  const base = `http://127.0.0.1:${cdp.__port}/index.html`;
  const url = `${base}#/march?scale=1&steps=140&spin=0&taa=0.9`;

  console.log('▸ pixel diff  (frozen clock, 320×180 sample, 8-bit channels)');

  await cdp.send('Page.navigate', { url });
  await waitFor(cdp, 'document.documentElement.dataset.boot', 'ready', 90000);
  await cdp.eval(`document.getElementById('intro-enter').click(); true`);
  await sleep(2500);

  // Freeze: rendering continues, animation does not, so two settings
  // differ by the setting alone.
  await cdp.eval(`(() => { const c = __aether.clock; c.paused = true; c.time = 8.0; return true; })()`);
  await sleep(1500);

  const capture = async (values) => {
    await cdp.eval(`__aether.panel.setValues(${JSON.stringify(values)}); true`);
    await sleep(1400);   // let the temporal filter re-converge
    return cdp.eval(GRAB);
  };

  for (const c of DIFF_CASES) {
    const a = await capture(c.base);
    const b = await capture(c.alt);

    let maxD = 0, sum = 0, n = 0, over2 = 0;
    for (let i = 0; i < a.length; i += 4) {
      const d = Math.max(
        Math.abs(a[i] - b[i]),
        Math.abs(a[i + 1] - b[i + 1]),
        Math.abs(a[i + 2] - b[i + 2]));
      maxD = Math.max(maxD, d);
      sum += d;
      if (d > 2) over2++;
      n++;
    }
    const mean = sum / n;
    const pct = (over2 / n) * 100;

    // A max of 1–2 on an 8-bit channel is dithering noise; anything that
    // moves more than a couple of levels across a visible share of the
    // frame is a real difference.
    const verdict = maxD <= 3 ? '看不出來'
      : pct < 0.5 ? '極小區域有差'
      : '看得出來';
    console.log(
      `  ${c.name.padEnd(13)} ${c.label.padEnd(16)} ` +
      `max ${String(maxD).padStart(3)}   mean ${mean.toFixed(2).padStart(5)}   ` +
      `>2 的像素 ${pct.toFixed(2).padStart(5)}%   → ${verdict}`);
  }
}

/* ═══ benchmark ═══════════════════════════════════════════════════
   A fixed workload, measured the same way every time, so "is the other
   GPU faster?" has a number attached instead of an impression.        */

/**
 * Each case pins everything that could drift: no camera spin, no
 * temporal accumulation (which would hide cost behind a smeared frame),
 * a stated render scale and step count. The name says what it isolates.
 */
const BENCH_CASES = [
  { name: 'baseline',    hash: '#/march?spin=0&taa=0&scale=0.75' },
  { name: 'full-res',    hash: '#/march?spin=0&taa=0&scale=1' },
  // Both switches are on by default now, so the informative cases turn
  // them off.
  { name: '−shadowNoise',hash: '#/march?spin=0&taa=0&scale=1&shadowNoise=0' },
  { name: '−reflectLit', hash: '#/march?spin=0&taa=0&scale=1&reflectLit=0' },
  { name: 'both off',    hash: '#/march?spin=0&taa=0&scale=1&shadowNoise=0&reflectLit=0' },
  { name: 'no-noise',    hash: '#/march?spin=0&taa=0&scale=1&displace=0' },
  { name: 'no-reflect',  hash: '#/march?spin=0&taa=0&scale=1&reflect=0' },
  { name: 'bare',        hash: '#/march?spin=0&taa=0&scale=1&displace=0&reflect=0&shadow=0&ao=0' },

  /* What is growing on the floor, which is now where most of the frame
     goes. Each of these is the one above it plus one thing, so the
     differences are the costs. */
  { name: 'grass',       hash: '#/march?spin=0&taa=0&scale=1&ground=grass' },
  { name: 'grass far',   hash: '#/march?spin=0&taa=0&scale=1&ground=grass&coverRadius=120' },
  { name: 'flowers',     hash: '#/march?spin=0&taa=0&scale=1&ground=grass&flowers=1&cover=1' },
  { name: 'wood',        hash: '#/march?spin=0&taa=0&scale=1&ground=grass&trees=1' },
  { name: 'everything',  hash: '#/march?spin=0&taa=0&scale=1&ground=grass&flowers=1&cover=1&trees=1' },
];

async function bench(cdp) {
  const base = `http://127.0.0.1:${cdp.__port}/index.html`;
  const seconds = Number(value('seconds', 6));

  console.log(`▸ benchmark  (${seconds}s per case, ${VIEWPORT.width}×${VIEWPORT.height})`);

  let renderer = null;
  const rows = [];

  for (const c of BENCH_CASES) {
    await cdp.send('Page.navigate', { url: base + c.hash });
    await waitFor(cdp, 'document.documentElement.dataset.boot', 'ready', 90000);
    await cdp.eval(`document.getElementById('intro-enter').click(); true`);

    // Warm up before measuring: shader compilation, the first few frames
    // of texture allocation and the driver's own ramp-up are not what we
    // are trying to time.
    await sleep(2500);
    renderer ??= await cdp.eval(`(() => {
      const gl = document.getElementById('stage').getContext('webgl2');
      const d = gl.getExtension('WEBGL_debug_renderer_info');
      return d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
    })()`);

    await sleep(seconds * 1000);

    // The ring buffer holds the last 90 frames; percentiles off that are
    // far more honest than a smoothed fps counter.
    const r = await cdp.eval(`(() => {
      const p = __aether.perf;
      const h = [...p.ordered()].filter((v) => v > 0).sort((a, b) => a - b);
      const at = (q) => h[Math.min(h.length - 1, Math.floor(h.length * q))];
      return {
        n: h.length,
        median: +at(0.5).toFixed(2),
        p95: +at(0.95).toFixed(2),
        gpuMs: +p.gpuMs.toFixed(2),
        buffer: __aether.scene.rt.width + '×' + __aether.scene.rt.height,
      };
    })()`);

    rows.push({ ...c, ...r });
    console.log(
      `  ${c.name.padEnd(11)} ${String(r.buffer).padStart(9)}  ` +
      `${(1000 / r.median).toFixed(1).padStart(5)} fps   ` +
      `frame ${r.median.toFixed(1).padStart(5)} ms (p95 ${r.p95.toFixed(1)})   ` +
      `gpu ${r.gpuMs.toFixed(1).padStart(5)} ms`);
  }

  console.log(`\n▸ renderer: ${renderer}`);

  // Compared on GPU time, not frame time. Once the shader is fast
  // enough, frame time pins itself to the compositor's cadence and every
  // case reads the same — which says nothing about the shader.
  const fullRes = rows.find((r) => r.name === 'full-res');
  if (fullRes) {
    console.log('▸ GPU time against the full-res case:');
    for (const r of rows) {
      if (r.name === 'full-res' || r.name === 'baseline') continue;
      const d = r.gpuMs - fullRes.gpuMs;
      const pct = (d / fullRes.gpuMs) * 100;
      console.log(`    ${r.name.padEnd(13)} ${d >= 0 ? '+' : '−'}${Math.abs(d).toFixed(1)} ms` +
        `  (${d >= 0 ? '+' : '−'}${Math.abs(pct).toFixed(0)}%)`);
    }
  }
  return rows;
}

/* ═══ interaction acceptance ══════════════════════════════════════
   Screenshots prove a scene renders. They prove nothing about whether
   it can be *used*. This drives real input events through the DevTools
   protocol — the browser cannot tell them from a hand on the mouse —
   and asserts on the state that comes out the other side.             */

async function interact(cdp, base, problems) {
  const results = [];
  const check = (name, pass, detail = '') => {
    results.push({ name, pass, detail });
    console.log(`  ${pass ? '✓' : '✗'} ${name}${detail ? `  — ${detail}` : ''}`);
  };

  const key = async (k, { code, vk, mods = 0, text } = {}) => {
    const common = {
      key: k, code: code ?? `Key${k.toUpperCase()}`,
      windowsVirtualKeyCode: vk ?? k.toUpperCase().charCodeAt(0),
      modifiers: mods,
    };
    await cdp.send('Input.dispatchKeyEvent', { type: text ? 'keyDown' : 'rawKeyDown', text, ...common });
    await cdp.send('Input.dispatchKeyEvent', { type: 'keyUp', ...common });
    await sleep(180);
  };

  const mouse = (type, x, y, extra = {}) =>
    cdp.send('Input.dispatchMouseEvent', { type, x, y, button: 'left', clickCount: 1, ...extra });

  /** Returns a mid-drag sample of the app's own pointer state. */
  const drag = async (from, to, steps = 26) => {
    // Move to the start first: a real mouse is always somewhere before
    // it is pressed, and dispatching a press at a fresh coordinate skips
    // the hover state the page may be relying on.
    await mouse('mouseMoved', from[0], from[1], { buttons: 0 });
    await sleep(40);
    await mouse('mousePressed', from[0], from[1], { buttons: 1 });
    await sleep(40);
    const pressed = await cdp.eval(`__aether.pointer.down`);
    let mid = { pressed };
    for (let i = 1; i <= steps; i++) {
      const t = i / steps;
      await mouse('mouseMoved',
        from[0] + (to[0] - from[0]) * t,
        from[1] + (to[1] - from[1]) * t, { buttons: 1 });
      if (i === Math.floor(steps / 2)) {
        Object.assign(mid, await cdp.eval(`({ down: __aether.pointer.down,
                                 points: __aether.pointer._points.size,
                                 x: +__aether.pointer.x.toFixed(3),
                                 y: +__aether.pointer.y.toFixed(3) })`));
      }
      await sleep(16);
    }
    await mouse('mouseReleased', to[0], to[1], { buttons: 0 });
    return mid;
  };

  console.log('▸ interaction');

  await cdp.send('Page.navigate', { url: `${base}#/march` });
  await waitFor(cdp, 'document.documentElement.dataset.boot', 'ready', 90000);

  /* intro */
  await cdp.eval(`document.getElementById('intro-enter').click(); true`);
  await sleep(600);
  check('intro dismisses', await cdp.eval(`document.body.classList.contains('intro-done')`));

  /* Clicking the surface bursts it. The locked-on-the-cluster camera is
     not the cat's view, so a click there still means what it always
     meant, cat on the floor or not. */
  await cdp.eval(`__aether.panel.setValues({ spin: false }, { notify: true }); true`);
  await sleep(500);
  const bursts0 = await cdp.eval(`__aether.scene.bursts`);

  const target = await cdp.eval(BALL_ON_SCREEN);
  await mouse('mouseMoved', target[0], target[1], { buttons: 0 });
  await sleep(120);
  await mouse('mousePressed', target[0], target[1], { buttons: 1 });
  await sleep(80);
  await mouse('mouseReleased', target[0], target[1], { buttons: 0 });
  await sleep(250);

  const burst = await cdp.eval(`({
    bursts: __aether.scene.bursts,
    ripples: [...__aether.scene.ripples].filter((_, i) => i % 4 === 3).filter((v) => v > 0).length,
    flash: +__aether.scene.flash.toFixed(2),
  })`);
  check('a click strikes the surface', burst.bursts > bursts0,
    `${bursts0} → ${burst.bursts} impacts, flash ${burst.flash}`);
  check('the impact leaves a ripple', burst.ripples > 0, `${burst.ripples} active`);

  // The ripple has to ride its host: everything here orbits, and a centre
  // frozen in world space slides off the surface it belongs to.
  const rode = await cdp.eval(`(() => {
    const s = __aether.scene, i = [...s.ripples].findIndex((v, k) => k % 4 === 3 && v > 0);
    return i < 0 ? null : [s.ripples[i - 3], s.ripples[i - 2], s.ripples[i - 1]];
  })()`);
  await sleep(600);
  const rode2 = await cdp.eval(`(() => {
    const s = __aether.scene, i = [...s.ripples].findIndex((v, k) => k % 4 === 3 && v > 0);
    return i < 0 ? null : [s.ripples[i - 3], s.ripples[i - 2], s.ripples[i - 1]];
  })()`);
  const carried = rode && rode2
    ? Math.hypot(rode2[0] - rode[0], rode2[1] - rode[1], rode2[2] - rode[2]) : -1;
  check('the ripple rides its host', carried > 0.0005 && carried < 0.6,
    `centre carried ${carried.toFixed(4)} world units`);
  await shot(cdp, 'i0-impact');

  /* a click on empty sky must not burst anything */
  const bursts1 = await cdp.eval(`__aether.scene.bursts`);
  await mouse('mouseMoved', 180, 140, { buttons: 0 });
  await sleep(120);
  await mouse('mousePressed', 180, 140, { buttons: 1 });
  await sleep(80);
  await mouse('mouseReleased', 180, 140, { buttons: 0 });
  await sleep(250);
  check('a click that misses does nothing',
    (await cdp.eval(`__aether.scene.bursts`)) === bursts1,
    'the CPU pick agrees with what is on screen');

  /* the canvas is actually receiving pointer input, and a drag orbits */
  const yaw0 = await cdp.eval(`__aether.scene.yaw`);
  const midOrbit = await drag([700, 450], [980, 380]);
  const yaw1 = await cdp.eval(`__aether.scene.yaw`);
  check('canvas receives pointer events', midOrbit?.down === true, JSON.stringify(midOrbit));
  check('drag orbits the camera', Math.abs(yaw1 - yaw0) > 0.2,
    `yaw ${yaw0.toFixed(2)} → ${yaw1.toFixed(2)}`);
  await shot(cdp, 'i1-orbited');

  /* the wheel zooms */
  const dist0 = await cdp.eval(`__aether.scene.targetDist`);
  await cdp.send('Input.dispatchMouseEvent', {
    type: 'mouseWheel', x: 700, y: 450, deltaX: 0, deltaY: -400,
  });
  await sleep(400);
  const dist1 = await cdp.eval(`__aether.scene.targetDist`);
  check('wheel zooms the camera', dist1 < dist0,
    `${dist0.toFixed(2)} → ${dist1.toFixed(2)}`);

  /* the XY pad drives the light direction */
  const light0 = await cdp.eval(`[...__aether.scene.lightDir]`);
  await cdp.eval(`__aether.panel.setValues({ light: [0.15, 0.8] }, { notify: true }); true`);
  await sleep(400);
  const light1 = await cdp.eval(`[...__aether.scene.lightDir]`);
  const swung = Math.hypot(light1[0] - light0[0], light1[1] - light0[1], light1[2] - light0[2]);
  check('the XY pad steers the light', swung > 0.3,
    `direction moved ${swung.toFixed(2)}`);

  /* command palette: open, fuzzy-filter, run */
  await key('k', { mods: 2 /* Ctrl */, vk: 75 });
  check('Ctrl+K opens palette', await cdp.eval(`document.getElementById('cmdk').open`));
  for (const ch of 'sdf') await key(ch, { text: ch });
  const matches = await cdp.eval(`document.querySelectorAll('#cmdk-list li').length`);
  const top = await cdp.eval(`document.querySelector('#cmdk-list li .cmdk__label')?.textContent ?? ''`);
  check('fuzzy search filters', matches > 0 && matches < 12, `${matches} hits, top = "${top}"`);
  await shot(cdp, 'i2-cmdk');
  await key('Escape', { code: 'Escape', vk: 27 });
  await sleep(300);

  /* the master quality control drives the seven below it, and they
     remain individually editable afterwards */
  const q = () => cdp.eval(`(() => {
    const s = __aether.state;
    return { steps: s.steps, shadowSteps: s.shadowSteps, aoTaps: s.aoTaps,
             reflectSteps: s.reflectSteps, scale: s.scale,
             shadowNoise: s.shadowNoise, reflectLit: s.reflectLit };
  })()`);

  const qHigh = await q();
  await cdp.eval(`__aether.panel.setValues({ quality: 0.02 }, { notify: true }); true`);
  await sleep(500);
  const qLow = await q();
  check('the master quality slider moves all seven',
    qLow.steps < qHigh.steps && qLow.shadowSteps < qHigh.shadowSteps
    && qLow.aoTaps < qHigh.aoTaps && qLow.reflectSteps < qHigh.reflectSteps
    && qLow.scale !== qHigh.scale
    && qLow.shadowNoise === false && qLow.reflectLit === false,
    `steps ${qHigh.steps}→${qLow.steps}, shadow ${qHigh.shadowSteps}→${qLow.shadowSteps}, ` +
    `ao ${qHigh.aoTaps}→${qLow.aoTaps}, scale ${qHigh.scale}→${qLow.scale}`);

  await cdp.eval(`__aether.panel.setValues({ steps: 150 }, { notify: true }); true`);
  await sleep(400);
  check('an individual slider still overrides the master',
    (await cdp.eval(`__aether.state.steps`)) === 150,
    'the master is an action, not a binding');

  await cdp.eval(`__aether.panel.setValues({ quality: 0.5 }, { notify: true }); true`);
  await sleep(400);

  /* the render-scale control actually reallocates the target */
  const rt0 = await cdp.eval(`__aether.scene.rt.width`);
  await cdp.eval(`__aether.panel.setValues({ scale: '0.5' }, { notify: true }); true`);
  await sleep(500);
  const rt1 = await cdp.eval(`__aether.scene.rt.width`);
  check('render scale resizes the target', rt1 < rt0, `${rt0}px → ${rt1}px wide`);
  await cdp.eval(`__aether.panel.setValues({ scale: '0.75' }, { notify: true }); true`);
  await sleep(400);

  /* URL is state */
  await cdp.eval(`__aether.panel.setValues({ tint: 'jade' }, { notify: true }); true`);
  await sleep(300);
  const url = await cdp.eval(`location.hash`);
  check('params round-trip into the URL', url.includes('tint=jade'), url);

  /* pause / step */
  await key(' ', { code: 'Space', vk: 32 });
  check('space pauses', await cdp.eval(`__aether.clock.paused`));
  const f0 = await cdp.eval(`__aether.clock.time`);
  await sleep(400);
  const f1 = await cdp.eval(`__aether.clock.time`);
  check('paused clock does not advance', f0 === f1, `t = ${f1.toFixed(3)}`);
  await key('.', { code: 'Period', vk: 190 });
  check('period steps one frame', (await cdp.eval(`__aether.clock.time`)) > f1);
  await key(' ', { code: 'Space', vk: 32 });

  /* zen mode + help */
  await key('z', { vk: 90 });
  check('zen mode hides chrome', await cdp.eval(`document.body.classList.contains('zen')`));
  await shot(cdp, 'i5-zen');
  await key('z', { vk: 90 });

  await key('?', { code: 'Slash', vk: 191, mods: 8 });
  check('? opens shortcuts', await cdp.eval(`document.getElementById('help').open`));
  await shot(cdp, 'i6-help');
  await key('Escape', { code: 'Escape', vk: 27 });

  /* a lab with one scene must not show a tab bar that cannot do anything */
  check('single-scene lab hides the tab bar',
    await cdp.eval(`getComputedStyle(document.getElementById('tabs')).display === 'none'`));

  /* ── the cat ──
     It is the one thing here made of triangles, and the one thing the
     app has to hand keys to. */
  await cdp.eval(`location.hash = '#/march?spin=0&scale=0.5'; true`);
  await sleep(900);

  const hold = async (keys, ms) => {
    const ev = (type, k) => cdp.send('Input.dispatchKeyEvent', {
      type, key: k, code: `Key${k.toUpperCase()}`,
      windowsVirtualKeyCode: k.toUpperCase().charCodeAt(0),
    });
    for (const k of keys) await ev('rawKeyDown', k);
    await sleep(ms);
    for (const k of keys) await ev('keyUp', k);
  };

  /* Camera-relative by default: W means "away from the camera" on
     screen, whichever way the cat happened to be facing. Asserted
     against the camera's own flattened forward, because that is the
     direction the user was looking at when they pressed the key. */
  await cdp.eval(`(() => { const c = __aether.scene.cat;
    c.x = 0; c.z = 3; c.yaw = 1.57; c.velocity = 0; return true; })()`);
  await sleep(200);
  const away = await cdp.eval(`(() => {
    const s = __aether.scene, f = s.basis.fwd;
    const l = Math.hypot(f[0], f[2]);
    return { x: s.cat.x, z: s.cat.z, fx: f[0] / l, fz: f[2] / l };
  })()`);
  await hold(['w'], 900);
  const wentTo = await cdp.eval(`(() => {
    const c = __aether.scene.cat; return { x: c.x, z: c.z, yaw: c.yaw }; })()`);
  const dxw = wentTo.x - away.x, dzw = wentTo.z - away.z;
  const alongCam = dxw * away.fx + dzw * away.fz;
  check('W walks the cat away from the camera, not along its old heading',
    alongCam > 0.4 && alongCam > Math.hypot(dxw, dzw) * 0.85,
    `moved ${Math.hypot(dxw, dzw).toFixed(2)}, of which ${alongCam.toFixed(2)} away from camera`);

  /* The point of camera-relative steering: S turns the animal round to
     face the viewer rather than reversing it. */
  await cdp.eval(`(() => { const c = __aether.scene.cat;
    c.velocity = 0; return true; })()`);
  await hold(['s'], 1200);
  const faced = await cdp.eval(`(() => {
    const s = __aether.scene, f = s.basis.fwd;
    const l = Math.hypot(f[0], f[2]);
    // +1 when the cat faces straight back at the camera.
    return -(Math.sin(s.cat.yaw) * f[0] / l + Math.cos(s.cat.yaw) * f[2] / l);
  })()`);
  check('S turns the cat to face the camera and walk toward it', faced > 0.9,
    `facing·(toward camera) = ${faced.toFixed(3)}`);

  /* A cat walks diagonally: each hind leg swings with the front leg on
     the *opposite* flank. The model names its two pairs from opposite
     ends, so this is asserted on where the legs actually are — and it is
     sampled at the extreme of the swing, because every gait passes
     through zero twice a cycle and would pass a sloppier test there. */
  const gait = await cdp.eval(`(() => {
    const r = __aether.scene.cat.rig;
    const xOf = (n) => r.rest.position[r.bone(n) * 3];
    const rotOf = (n) => r.rotation[r.bone(n) * 3];
    const hindPlus = xOf('hipHL') > xOf('hipHR') ? 'hipHL' : 'hipHR';
    const frontPlus = xOf('pawFL') > xOf('pawFR') ? 'pawFL' : 'pawFR';
    const frontMinus = frontPlus === 'pawFL' ? 'pawFR' : 'pawFL';
    return { hip: rotOf(hindPlus), same: rotOf(frontPlus), opposite: rotOf(frontMinus) };
  })()`);
  await hold(['w'], 500);
  const swing = await cdp.eval(`(() => {
    const r = __aether.scene.cat.rig;
    const xOf = (n) => r.rest.position[r.bone(n) * 3];
    const rotOf = (n) => r.rotation[r.bone(n) * 3];
    const hindPlus = xOf('hipHL') > xOf('hipHR') ? 'hipHL' : 'hipHR';
    const frontPlus = xOf('pawFL') > xOf('pawFR') ? 'pawFL' : 'pawFR';
    const frontMinus = frontPlus === 'pawFL' ? 'pawFR' : 'pawFL';
    let best = null;
    for (let i = 0; i < 40; i++) {
      const h = rotOf(hindPlus);
      if (!best || Math.abs(h) > Math.abs(best.hip)) {
        best = { hip: h, same: rotOf(frontPlus), opposite: rotOf(frontMinus) };
      }
      __aether.scene.cat.update(1 / 60, -1.35);
    }
    return best;
  })()`);
  check('the gait is diagonal, not same-side',
    Math.abs(swing.hip) > 0.05
      && Math.sign(swing.hip) === Math.sign(swing.opposite)
      && Math.sign(swing.hip) !== Math.sign(swing.same),
    `hind ${swing.hip.toFixed(3)} · opposite front ${swing.opposite.toFixed(3)} · same-side front ${swing.same.toFixed(3)}`);

  /* The tail is a spring chain: under a sustained turn the lag has to
     grow monotonically from base to tip.

     Driven directly rather than through the keys, and measured during a
     *steady* turn. Camera-relative steering turns once and then goes
     straight, so holding a key settles the chain and leaves it
     overshooting — where the tip can legitimately read less than the
     middle on its way back through. That is the spring behaving, not
     failing, and a test that caught it there would be testing the
     moment it sampled rather than the thing it claims. */
  const tail = await cdp.eval(`(() => {
    const c = __aether.scene.cat, s = __aether.scene;
    const was = c.mode;
    c.setMode('look');
    let best = null;
    for (let i = 0; i < 90; i++) {
      c.look(0.05, 0);                       // a steady 3 rad/s turn
      c.update(1 / 60, -1.35, s.basis);
      const tip = c.sway.yaw.a[8] - c.sway.yaw.a[0];
      const mid = c.sway.yaw.a[4] - c.sway.yaw.a[0];
      if (!best || Math.abs(tip) > Math.abs(best.tip)) best = { tip, mid };
    }
    c.setMode(was);
    return best;
  })()`);
  check('the tail trails the body, tip furthest',
    Math.abs(tail.tip) > 1e-3 && Math.abs(tail.tip) > Math.abs(tail.mid),
    `tip lags ${tail.tip.toFixed(3)} rad, middle ${tail.mid.toFixed(3)}`);

  /* The whiskers hang off the head and run on their own chain, so they
     have to lag too — and by much less, because a whisker is stiff and
     the gain is a quarter of the tail's. Both facts are asserted: a
     whisker that swept like a tail would read as an antenna. */
  const whisk = await cdp.eval(`(() => {
    const c = __aether.scene.cat, s = __aether.scene;
    const was = c.mode;
    c.setMode('look');
    let best = null;
    for (let i = 0; i < 90; i++) {
      c.look(0.05, 0);
      c.update(1 / 60, -1.35, s.basis);
      const w = c.sway;
      const tip = w.headYaw.a[8] - w.headYaw.a[0];
      const mid = w.headYaw.a[4] - w.headYaw.a[0];
      if (!best || Math.abs(tip) > Math.abs(best.tip)) {
        /* The tail's tip is a frame now rather than a pair of angles, so
           what is compared against the whisker's deflection is the angle
           that frame stands for: 2·acos of a unit quaternion's w. */
        const q = w.qs[(w.count - 1) * 4 + 3];
        best = { tip, mid, deflect: Math.abs(w.whiskers[16]),
                 tailDeflect: 2 * Math.acos(Math.min(1, Math.abs(q))) };
      }
    }
    c.setMode(was);
    return best;
  })()`);
  check('the whiskers trail the head, and far less than the tail',
    Math.abs(whisk.tip) > 1e-3 && Math.abs(whisk.tip) > Math.abs(whisk.mid)
      && whisk.deflect > 1e-3 && whisk.deflect < whisk.tailDeflect,
    `tip lags ${whisk.tip.toFixed(3)} rad; tip deflects ${whisk.deflect.toFixed(3)} vs tail ${whisk.tailDeflect.toFixed(3)}`);

  /* ── the cat in the scene's light ──
     Isolated by holding everything still and moving one slider. The cat,
     the camera, the light direction and the clock are all fixed; only
     the shadow term is switched on. Anything that changes on screen is
     the cluster blocking the sun, because nothing else can be.

     The cat is parked where the sun, put near the horizon, has the whole
     cluster between it and the animal. */
  await cdp.eval(`location.hash = '#/march?spin=0&scale=0.5&taa=0&light=0.68,0.95&shadow=0'; true`);
  await sleep(1200);
  await cdp.eval(`(() => {
    const s = __aether.scene, c = s.cat;
    __aether.clock.paused = true; __aether.clock.time = 8;
    c.x = -2.82; c.z = -0.95; c.yaw = 1.9;
    s.targetDist = 2.6; s.yaw = 1.5; s.pitch = 0.12;
    return true;
  })()`);
  await sleep(900);

  /** Mean luma over a box of the canvas, in CSS pixels. */
  const patch = () => cdp.eval(`(() => {
    const c = document.getElementById('stage');
    const s = document.createElement('canvas');
    s.width = 96; s.height = 96;
    const g = s.getContext('2d');
    // The cat sits mid-frame under this camera; sample only it.
    g.drawImage(c, c.width * 0.42, c.height * 0.38, c.width * 0.16, c.height * 0.28, 0, 0, 96, 96);
    const d = g.getImageData(0, 0, 96, 96).data;
    let sum = 0;
    for (let i = 0; i < d.length; i += 4) sum += 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2];
    return sum / (d.length / 4);
  })()`);

  const unshadowed = await patch();
  await cdp.eval(`__aether.panel.setValues({ shadow: 1 }); true`);
  await sleep(900);
  const shadowed = await patch();

  check('the cluster casts a shadow onto the cat',
    unshadowed > 1 && shadowed < unshadowed * 0.85,
    `mean luma ${unshadowed.toFixed(1)} → ${shadowed.toFixed(1)}`);

  /* ── the cat's own shadow ──
     The other direction: the floor losing light to the animal standing
     on it. Isolated by hiding the cat, which takes its capsules out of
     the field with it, and sampling a patch of bare floor beside where
     it stood — so the cat itself is never in the box either way, and
     what changes there is only what it was blocking. */
  /* A sun halfway up. High and the shadow lands under the cat, where
     the cat hides it; near the horizon and the floor's own ndl drops to
     nothing, so there is no light there for a shadow to remove. Around
     thirty degrees the shadow clears the animal and the floor is still
     lit enough to show it. */
  await cdp.eval(`location.hash = '#/march?spin=0&scale=0.5&taa=0&shadow=1&ao=1&light=0.68,0.667'; true`);
  await sleep(1200);
  /* The locked camera, not the follow one. Hiding the cat also switches
     follow mode off — there is nothing left to follow — so the camera
     would jump between the two samples and the frames would not be
     comparable at all. Orbit stays put whatever the cat is doing. */
  await cdp.eval(`(() => {
    const s = __aether.scene, c = s.cat;
    __aether.clock.paused = true; __aether.clock.time = 8;
    c.x = 1.9; c.z = 1.9; c.yaw = 1.2;
    // High enough to see the floor beside the animal: the shadow is
    // darkest directly beneath it, which is exactly where the cat itself
    // hides it from a low camera.
    s.yaw = 0.85; s.pitch = 1.02; s.targetDist = 4.4;
    return true;
  })()`);
  await sleep(900);

  /* Sample where the shadow *must* be rather than hunting for it.
     A body centre h above the floor throws its shadow h/L.y back along
     the light, so the landing point is known in world space and can be
     projected with the same basis the shader rays are built from. A
     search over the frame keeps finding the cat itself, or the cluster's
     own much larger shadow; this cannot. */
  const shadowPatch = () => cdp.eval(`(() => {
    const sc = __aether.scene, cat = sc.cat, b = sc.basis, L = sc.lightDir, focal = 1.5;
    const h = 0.55;                       // body centre above the floor
    const wx = cat.x - L[0] * (h / L[1]);
    const wz = cat.z - L[2] * (h / L[1]);
    const rel = [wx - b.pos[0], -1.35 - b.pos[1], wz - b.pos[2]];
    const vx = rel[0]*b.right[0] + rel[1]*b.right[1] + rel[2]*b.right[2];
    const vy = rel[0]*b.up[0]    + rel[1]*b.up[1]    + rel[2]*b.up[2];
    const vz = rel[0]*b.fwd[0]   + rel[1]*b.fwd[1]   + rel[2]*b.fwd[2];
    const aspect = innerWidth / innerHeight;
    const px = ((vx * focal / aspect / vz) * 0.5 + 0.5) * innerWidth;
    const py = (0.5 - (vy * focal / vz) * 0.5) * innerHeight;

    const c = document.getElementById('stage');
    const sx = px / innerWidth * c.width, sy = py / innerHeight * c.height;
    const box = c.width * 0.035;
    const s = document.createElement('canvas');
    s.width = 32; s.height = 32;
    const g = s.getContext('2d');
    g.drawImage(c, sx - box / 2, sy - box / 2, box, box, 0, 0, 32, 32);
    const d = g.getImageData(0, 0, 32, 32).data;
    let sum = 0;
    for (let i = 0; i < d.length; i += 4) sum += 0.2126*d[i] + 0.7152*d[i+1] + 0.0722*d[i+2];
    return sum / (d.length / 4);
  })()`);

  const litByCat = await shadowPatch();
  await cdp.eval(`__aether.panel.setValues({ cat: false }); true`);
  await sleep(900);
  const noCat = await shadowPatch();
  await cdp.eval(`__aether.panel.setValues({ cat: true }); true`);
  await sleep(400);

  // The floor's own albedo is nearly black — it is a dark plane with a
  // grid drawn on it — so the gate is only there to prove something was
  // lit at all. The drop is what carries the claim.
  check('the cat casts a shadow onto the floor',
    noCat > 8 && litByCat < noCat * 0.7,
    `floor where the shadow lands: ${noCat.toFixed(1)} without the cat → ${litByCat.toFixed(1)} with it`);

  // Hand the clock back. Everything below drives the cat with keys, and
  // a paused clock advances nothing.
  await cdp.eval(`__aether.clock.paused = false; true`);
  await sleep(300);

  /* WASD overlaps the app's own shortcuts, so the scene may only claim
     them while there is something to steer. */
  await cdp.eval(`(() => { __aether.scene.cat.velocity = 0; return true; })()`);
  await hold(['s'], 400);
  const claimed = await cdp.eval(`__aether.scene.cat.speed`);
  check('S reaches the cat rather than the screenshot shortcut', claimed > 0.05,
    `speed ${claimed.toFixed(2)}`);

  await cdp.eval(`__aether.panel.setValues({ cat: false }); true`);
  await sleep(300);
  const released = await cdp.eval(`(() => {
    const e = { key: 's' };
    return __aether.scene.onKey(e, true) === false;
  })()`);
  check('and hands S back when the cat is hidden', released);
  await cdp.eval(`__aether.panel.setValues({ cat: true }); true`);
  await sleep(200);

  /* ── mouse-look ──
     Y swaps what the mouse and A/D mean. The lock is released straight
     after switching so these exercise the unlocked fallback: synthetic
     events do not carry the movementX a captured pointer reports, and a
     test that silently depended on the lock being granted would pass or
     fail on the browser's mood. */
  await key('y', { vk: 89 });
  const mode = await cdp.eval(`(() => {
    document.exitPointerLock?.();
    return __aether.scene.cat.mode;
  })()`);
  check('Y switches to mouse-look', mode === 'look', `mode = ${mode}`);
  await sleep(200);

  const catYaw0 = await cdp.eval(`__aether.scene.cat.yaw`);
  await mouse('mouseMoved', 440, 450, { buttons: 0 });
  await sleep(60);
  await mouse('mouseMoved', 960, 450, { buttons: 0 });
  await sleep(220);
  const catYaw1 = await cdp.eval(`__aether.scene.cat.yaw`);
  check('the mouse turns the cat with no button held',
    Math.abs(catYaw1 - catYaw0) > 0.2,
    `yaw ${catYaw0.toFixed(2)} → ${catYaw1.toFixed(2)}`);

  /* Strafing is movement the cat is not facing. Measured as the
     component across its own nose, which is exactly what "平移" means
     and what steering would leave at zero. */
  const strafed = await cdp.eval(`(() => {
    const c = __aether.scene.cat;
    c.x = 0; c.z = 0; c.velocity = 0; c.strafeVel = 0;
    return { yaw: c.yaw };
  })()`);
  await hold(['d'], 700);
  const moved = await cdp.eval(`(() => {
    const c = __aether.scene.cat;
    const fx = Math.sin(${strafed.yaw}), fz = Math.cos(${strafed.yaw});
    return {
      along: c.x * fx + c.z * fz,          // forward component
      across: c.x * -fz + c.z * fx,        // along its own right
      yawDrift: c.yaw - ${strafed.yaw},
    };
  })()`);
  check('A and D strafe instead of steering in mouse-look',
    Math.abs(moved.across) > 0.3 && Math.abs(moved.across) > Math.abs(moved.along) * 2
      && Math.abs(moved.yawDrift) < 0.01,
    `across ${moved.across.toFixed(2)} · along ${moved.along.toFixed(2)} · yaw drift ${moved.yawDrift.toFixed(3)}`);

  await key('y', { vk: 89 });
  const back = await cdp.eval(`(() => {
    const c = __aether.scene.cat;
    c.velocity = 0; c.strafeVel = 0;
    return { mode: c.mode, yaw: c.yaw };
  })()`);
  await hold(['d'], 700);
  const steered = await cdp.eval(`(() => {
    const s = __aether.scene, r = s.basis.right;
    // Back in camera-relative: D should have turned the cat to face
    // screen-right, not sidled it while facing where it was.
    return { yaw: s.cat.yaw, align: Math.sin(s.cat.yaw) * r[0] + Math.cos(s.cat.yaw) * r[2] };
  })()`);
  check('Y switches back, and D turns the cat toward screen-right',
    back.mode === 'camera' && steered.align > 0.9,
    `mode = ${back.mode}, facing·screen-right = ${steered.align.toFixed(3)}`);

  /* ── eye beams ──
     A left click in mouse-look fires, and the shot drives the cluster
     apart along its axis. Both halves are asserted: that the trigger
     reaches the scene at all, and that what it delivers is radial —
     spheres thrown clear of the line rather than shoved down it. */
  await cdp.eval(`location.hash = '#/march?spin=0&scale=0.5&taa=0&balls=6&camera=follow'; true`);
  await sleep(1200);
  const aimed = await cdp.eval(`(() => {
    const s = __aether.scene, c = s.cat;
    __aether.clock.paused = true; __aether.clock.time = 8;
    s._setControlMode('look');
    document.exitPointerLock?.();
    c.x = 0; c.z = 4.2; c.yaw = Math.PI;
    s.yaw = 0; s.pitch = -0.15; s.targetDist = 2.4;
    return s.shots;
  })()`);
  await sleep(600);

  /* Hold the trigger. Mouse-look repeats on a cooldown, so a single
     sustained press has to produce several shots — a beam weapon that
     needs one click per shot is one nobody uses twice. */
  await cdp.eval(`__aether.clock.paused = false; true`);
  /* Held on the cluster itself. Pointer lock is refused in a driven
     browser, so this exercises the free-cursor path, where a click that
     hits nothing is not a shot — a fixed pixel would mostly miss. */
  const held = await cdp.eval(BALL_ON_SCREEN);
  await mouse('mouseMoved', held[0], held[1], { buttons: 0 });
  await sleep(60);
  await mouse('mousePressed', held[0], held[1], { buttons: 1 });
  await sleep(700);
  await mouse('mouseReleased', held[0], held[1], { buttons: 0 });
  await sleep(200);

  /* What a shot leaves behind is a *line*, not a point: the stretch of
     beam that was inside the cluster. Measured as the longest live
     impact segment, and where its middle sits relative to the cluster. */
  const blast = await cdp.eval(`(() => {
    const s = __aether.scene;
    let span = 0, midFromCentre = 1e9;
    for (let i = 0; i < 4; i++) {
      if (s.ripples[i * 4 + 3] <= 0) continue;
      const q = i * 4;
      const len = Math.hypot(
        s.rippleTo[q] - s.ripples[q],
        s.rippleTo[q + 1] - s.ripples[q + 1],
        s.rippleTo[q + 2] - s.ripples[q + 2]);
      if (len <= span) continue;
      span = len;
      midFromCentre = Math.hypot(
        (s.ripples[q] + s.rippleTo[q]) / 2,
        (s.ripples[q + 1] + s.rippleTo[q + 1]) / 2,
        (s.ripples[q + 2] + s.rippleTo[q + 2]) / 2);
    }
    return { shots: s.shots, span, midFromCentre, bound: s.bound[3] };
  })()`);

  check('holding the trigger in mouse-look fires repeatedly',
    blast.shots >= aimed + 3, `shots ${aimed} → ${blast.shots} while held for 0.7 s`);
  check('the shot bores a line through the cluster, not a crater on it',
    blast.span > 1.0 && blast.midFromCentre < blast.bound,
    `impact spans ${blast.span.toFixed(2)} units, `
    + `its middle ${blast.midFromCentre.toFixed(2)} from the centre of a ${blast.bound.toFixed(2)} cluster`);

  /* With the pointer free the aim comes from the cursor, not the
     crosshair: the cat turns to face what was clicked and the beam runs
     from its eyes to that point. Asserted against the cluster's actual
     position rather than against the camera, since from anywhere but
     directly behind the animal the two are different directions. */
  await cdp.eval(`(() => {
    const s = __aether.scene, c = s.cat;
    s._setControlMode('camera');
    __aether.panel.setValues({ camera: 'follow' });
    // The clock has to run: an aimed shot waits on the cat's head, and
    // a frozen clock is a head that never arrives.
    __aether.clock.paused = false;
    c.x = 3.4; c.z = 3.4; c.yaw = 2.6;        // stood aside, facing away
    s.yaw = 0.85; s.pitch = 0.30; s.targetDist = 9;
    return true;
  })()`);
  await sleep(700);

  const before = await cdp.eval(`__aether.scene.cat.yaw`);
  const onBall = await cdp.eval(BALL_ON_SCREEN);
  await mouse('mouseMoved', onBall[0], onBall[1], { buttons: 0 });
  await sleep(120);
  await mouse('mousePressed', onBall[0], onBall[1], { buttons: 1 });
  await sleep(80);
  await mouse('mouseReleased', onBall[0], onBall[1], { buttons: 0 });
  await sleep(400);

  const aimedAt = await cdp.eval(`(() => {
    const s = __aether.scene, c = s.cat, d = s.laser.dir;
    // Where the beam left from, and the line from there to the cluster.
    const e = s.laser.origin;
    let tx = -e[0], ty = -e[1], tz = -e[2];      // cluster sits at the origin
    const l = Math.hypot(tx, ty, tz) || 1;
    /* The head against the *beam*, not against the cluster's centre: the
       shot aimed at the surface point the cursor picked, which from this
       range sits a good ten degrees off centre. And the head, not the
       body — the neck covers most of the turn, so the body deliberately
       stops short. */
    const headYaw = c.yaw + c.rig.rotation[c.rig.bone('head') * 3 + 1];
    const flat = Math.hypot(d[0], d[2]) || 1;
    return {
      yaw: c.yaw,
      onTarget: (tx / l) * d[0] + (ty / l) * d[1] + (tz / l) * d[2],
      facing: Math.sin(headYaw) * (d[0] / flat) + Math.cos(headYaw) * (d[2] / flat),
    };
  })()`);

  check('an unlocked click turns the cat onto the target and fires along that line',
    Math.abs(aimedAt.yaw - before) > 0.2 && aimedAt.onTarget > 0.9 && aimedAt.facing > 0.9,
    `body yaw ${before.toFixed(2)} → ${aimedAt.yaw.toFixed(2)}, `
    + `beam·target ${aimedAt.onTarget.toFixed(3)}, head·beam ${aimedAt.facing.toFixed(3)}`);

  /* A click on empty sky is not a shot. With a free cursor the click is
     the aim, so aiming at nothing has to cost nothing — the corner of
     the floor, well clear of the cluster, stands in for a slip. */
  const beforeMiss = await cdp.eval(`__aether.scene.shots`);
  await mouse('mouseMoved', 180, 140, { buttons: 0 });
  await sleep(120);
  await mouse('mousePressed', 180, 140, { buttons: 1 });
  await sleep(80);
  await mouse('mouseReleased', 180, 140, { buttons: 0 });
  await sleep(400);
  const afterMiss = await cdp.eval(`__aether.scene.shots`);
  check('a click that hits nothing does not fire', afterMiss === beforeMiss,
    `shots ${beforeMiss} → ${afterMiss}`);

  /* The head points down the beam, and it is the head that does most of
     the turning. Both are driven directly rather than through a click,
     so the angles are exact rather than whatever a cursor happened to
     pick. */
  const aimed2 = await cdp.eval(`(() => {
    const s = __aether.scene, c = s.cat, r = c.rig;
    const head = r.bone('head');
    // Hold an aim until the weight has settled, then read the bone.
    const pitchFor = (dx, dy, dz) => {
      for (let i = 0; i < 40; i++) { s._aimAlong(dx, dy, dz); c.update(1 / 60, -1.35, s.basis); }
      return r.rotation[head * 3];
    };
    const up = pitchFor(0, 1, 0.3);
    const down = pitchFor(0, -1, 0.3);

    // A modest turn, well inside what the neck can cover on its own.
    c.x = 3.4; c.z = 3.4;
    const want = Math.atan2(-c.x, -c.z);
    c.yaw = want - 0.55;
    const before = c.yaw;
    s._aimShot = { x: 0, y: 0, z: 0, t: 0.15 };
    for (let i = 0; i < 20; i++) { s._aimTick(1 / 60); c.update(1 / 60, -1.35, s.basis); }

    const wrap = (a) => Math.abs((a + Math.PI * 3) % (Math.PI * 2) - Math.PI);
    return { up, down, body: wrap(c.yaw - before), head: Math.abs(c._aimYaw) };
  })()`);

  // Positive rotation on the head bone is nose-down, so up must be less.
  check('the head pitches to follow the beam',
    aimed2.up < aimed2.down - 0.6,
    `head pitch ${aimed2.up.toFixed(2)} aiming up vs ${aimed2.down.toFixed(2)} aiming down`);
  check('the head leads the turn and the body only follows',
    aimed2.head > aimed2.body * 2,
    `head turned ${aimed2.head.toFixed(2)} rad, body ${aimed2.body.toFixed(2)}`);

  await cdp.eval(`__aether.clock.paused = false; true`);
  await sleep(300);

  /* ── the floor's three styles ──
     The grid is the default and grows nothing; the other two are the
     only things in the scene that put triangles on the ground. */
  await cdp.eval(`__aether.panel.setValues({ ground: 'grid' }, { notify: true }); true`);
  await sleep(300);
  const onGrid = await cdp.eval(`({ tri: __aether.scene.ground.triangles,
    flowers: __aether.scene.ground.flowers })`);
  check('the grid floor grows nothing',
    onGrid.tri === 0 && onGrid.flowers === 0,
    `${onGrid.tri} triangles, ${onGrid.flowers} flowers`);

  await cdp.eval(`__aether.panel.setValues(
    { ground: 'grass', flowers: false }, { notify: true }); true`);
  await sleep(300);
  const onGrass = await cdp.eval(`({ tri: __aether.scene.ground.triangles,
    blades: __aether.scene.ground.blades, flowers: __aether.scene.ground.flowers })`);
  check('grass covers the floor and grows no flowers',
    onGrass.blades > 1000 && onGrass.tri > 1000 && onGrass.flowers === 0,
    `${onGrass.blades} blades, ${onGrass.tri} triangles, ${onGrass.flowers} flowers`);

  /* Flowers are their own switch now rather than a third floor style,
     so turning them on must add flowers and change nothing about the
     grass they are standing in. */
  await cdp.eval(`__aether.panel.setValues({ flowers: true }, { notify: true }); true`);
  await sleep(300);
  const onMeadow = await cdp.eval(`({ blades: __aether.scene.ground.blades,
    flowers: __aether.scene.ground.flowers })`);
  check('the flower switch adds flowers and leaves the grass alone',
    onMeadow.blades === onGrass.blades && onMeadow.flowers > 100,
    `${onMeadow.blades} blades either way, ${onMeadow.flowers} flowers`);

  /* Two flower controls, two different pictures. Halving how many clumps
     there are and halving how full each one is both take flowers away,
     but they are not the same field, and a single slider could never say
     which of the two it was doing. */
  const flowerAt = async (clumps, density) => {
    await cdp.eval(`__aether.panel.setValues(
      { flowerClumps: ${clumps}, flowerDensity: ${density} }, { notify: true }); true`);
    await sleep(350);
    return cdp.eval(`__aether.scene.ground.flowers`);
  };
  const bothFull = await flowerAt(1.0, 1.0);
  const fewClumps = await flowerAt(0.3, 1.0);
  const thinClumps = await flowerAt(1.0, 0.2);
  check('the two flower controls pull on different things',
    fewClumps < bothFull * 0.55 && thinClumps < bothFull * 0.55
    && Math.abs(fewClumps - thinClumps) > bothFull * 0.05,
    `all ${bothFull} · few clumps ${fewClumps} · thin clumps ${thinClumps}`);
  await flowerAt(0.62, 0.7);

  /* Spread is the third of them, and the odd one out: the other two take
     flowers away, this one spends the same flowers over more ground.
     Measured as how many square metres the flowers land in — tight
     clumps pile many into one, a wide scatter puts one in each. */
  const spreadAt = async (v) => {
    await cdp.eval(`__aether.panel.setValues(
      { flowerSpread: ${v} }, { notify: true }); true`);
    await sleep(350);
    return cdp.eval(`(() => {
      const g = __aether.scene.ground, cells = new Set();
      for (let i = 0; i < g.flowers; i++) {
        cells.add(Math.round(g._sown[i * 11]) + ',' + Math.round(g._sown[i * 11 + 1]));
      }
      return { n: g.flowers, cells: cells.size };
    })()`);
  };
  const tight = await spreadAt(0.25);
  const loose = await spreadAt(2.5);
  check('the spread control moves the ground covered, not the count',
    Math.abs(loose.n - tight.n) < tight.n * 0.35
    && loose.cells > tight.cells * 1.5,
    `tight ${tight.n} flowers over ${tight.cells} m², `
    + `loose ${loose.n} over ${loose.cells} m²`);
  await cdp.eval(`__aether.panel.setValues({ flowerSpread: 1 }, { notify: true }); true`);
  await sleep(250);

  /* The patch follows the camera but is snapped to its own lattice, and
     that snap is the entire reason a blade keeps its place: the hash is
     taken from the world coordinate of its cell, so an origin that
     landed anywhere off the lattice would reseed the whole meadow every
     frame and the ground would crawl as you walked over it. */
  /* Sown once and kept. The flowers used to be re-derived in the vertex
     shader, off a 4 m clump grid whose corners were snapped to the
     0.5 m *grass* lattice — so every half-metre of camera travel
     re-rolled every clump, and the flowers changed places continuously
     while you walked. They are placed on the CPU now, when the patch
     shifts a whole clump cell. */
  const WHERE = `(() => {
    const g = __aether.scene.ground, out = [];
    for (let i = 0; i < g.flowers; i++) {
      out.push(g._sown[i * 11].toFixed(4) + ',' + g._sown[i * 11 + 1].toFixed(4));
    }
    const off = (v) => Math.abs(v / g.cell - Math.round(v / g.cell));
    return { at: out, sowings: g.sowings, patch: [...g._patch],
             off: Math.max(off(g._patch[0]), off(g._patch[1])) };
  })()`;

  const parked = await cdp.eval(WHERE);
  await sleep(900);                                  // a good many frames
  const stillParked = await cdp.eval(WHERE);
  check('standing still never re-sows the flowers',
    stillParked.sowings === parked.sowings,
    `${parked.sowings} sowings → ${stillParked.sowings} over ~0.9 s`);

  await drag([700, 450], [1180, 470]);
  await sleep(400);
  const walked = await cdp.eval(WHERE);

  /* And when it does re-sow, a flower the two buffers have in common is
     at the identical coordinate — the placement is deterministic in the
     world cell, so what the shift changes is which flowers are in the
     buffer, never where any of them stands. */
  const shared = new Set(parked.at);
  const kept = walked.at.filter((p) => shared.has(p)).length;
  check('flowers keep their exact places when the patch moves under them',
    kept > walked.at.length * 0.4,
    `${kept} of ${walked.at.length} unchanged to 0.1 mm, `
    + `${walked.sowings - parked.sowings} re-sowing(s) for the whole drag`);

  /* The patch is snapped to its own lattice, which is what lets the
     grass be re-derived from a hash of its cell every frame without the
     ground crawling as you walk over it. */
  check('the patch follows the camera but stays on its own lattice',
    walked.off < 1e-4
    && (walked.patch[0] !== parked.patch[0] || walked.patch[1] !== parked.patch[1]),
    `origin ${parked.patch.map((v) => v.toFixed(2))} → ${walked.patch.map((v) => v.toFixed(2))}, `
    + `off-lattice by ${walked.off.toExponential(1)}`);

  /* Reach adds ground, not size. The near cells are one constant at
     every setting of the control — an earlier version grew them with the
     reach, which kept the cost down and scaled the whole pattern, so
     winding the view out thinned the grass at your feet. */
  const READ = `({ r: __aether.scene.ground.radius, rings: __aether.scene.ground.rings,
    blades: __aether.scene.ground.blades, cell: __aether.scene.ground.cell })`;
  await cdp.eval(`__aether.panel.setValues({ coverRadius: 8 }, { notify: true }); true`);
  await sleep(300);
  const near = await cdp.eval(READ);
  await cdp.eval(`__aether.panel.setValues({ coverRadius: 120 }, { notify: true }); true`);
  await sleep(300);
  const far = await cdp.eval(READ);

  check('the reach control never resizes the plants',
    near.cell === far.cell && near.r === 8 && far.r === 120,
    `${near.r}u and ${far.r}u both on a ${near.cell} cell`);

  /* And what it costs is the rings it had to add, which is the log of
     the reach — not its square. */
  check('reach is bought in rings, and costs exactly the rings it buys',
    far.rings > near.rings
    && Math.abs(far.blades / near.blades - far.rings / near.rings) < 1e-6,
    `${near.r}u → ${near.rings} ring / ${near.blades} blades, `
    + `${far.r}u → ${far.rings} rings / ${far.blades}; ×15 reach for `
    + `×${(far.blades / near.blades).toFixed(1)} blades`);

  await cdp.eval(`__aether.panel.setValues({ coverRadius: 15 }, { notify: true }); true`);
  await sleep(200);

  /* ── the wood ──
     Grown into world chunks, and its shadow drawn from the sun into a
     depth map. Both are things a screenshot is bad at confirming, so
     both are asked directly. */
  await cdp.eval(`__aether.panel.setValues(
    { trees: true, ground: 'grass', flowers: false, wind: 0, spin: false, cat: false },
    { notify: true }); true`);
  await sleep(700);
  const wood = await cdp.eval(`(() => {
    const t = __aether.scene.trees;
    const c = t.mapCoverage();
    return { trees: t.trees, segs: t.segments, leaves: t.leaves,
             leavesGrown: t.leavesGrown,
             grown: t.grown, packs: t.packs, renders: t.mapRenders,
             covered: c.covered, top: c.top, err: c.err, half: c.half };
  })()`);

  check('a wood grows, in chunks, out of nothing but a hash',
    wood.trees > 8 && wood.segs > 500 && wood.leaves > 2000 && wood.grown > 8,
    `${wood.trees} trees from ${wood.grown} chunks — `
    + `${wood.segs} branch segments, ${wood.leaves} of ${wood.leavesGrown} leaves drawn `
    + `(${(100 - wood.leaves / wood.leavesGrown * 100).toFixed(0)}% dropped by distance)`);

  /* The map has to actually have canopy in it. A shadow term that
     silently reads an empty texture returns "lit" everywhere and looks
     exactly like a shadow that is merely subtle. */
  check('the sun sees the canopy',
    wood.covered > 0.01 && wood.top > 0,
    `${(wood.covered * 100).toFixed(1)}% of the map is canopy, `
    + `highest ${wood.top.toFixed(2)} along the light`
    + ` (readback ${wood.half ? 'half' : 'float'}, gl error ${wood.err})`);

  const rendersBefore = wood.renders;
  await sleep(900);
  const rendersAfter = await cdp.eval(`__aether.scene.trees.mapRenders`);
  check('a still wood in calm air redraws its shadow map no more',
    rendersAfter === rendersBefore,
    `${rendersBefore} renders → ${rendersAfter} over ~0.9 s`);

  /* A chunk is grown on a cache miss and never again, so this counter
     only climbs for ground that has genuinely come into range. Swinging
     the camera right round may bring a little in; regrowing what is
     already there would send it up by dozens. */
  const grownBefore = await cdp.eval(`__aether.scene.trees.grown`);
  await drag([700, 450], [1150, 470]);
  await sleep(500);
  const afterWalk = await cdp.eval(`({ grown: __aether.scene.trees.grown,
    cached: __aether.scene.trees.cache.size })`);
  check('turning the view grows only ground that was not there before',
    afterWalk.grown - grownBefore <= 8 && afterWalk.grown <= afterWalk.cached,
    `${grownBefore} chunks grown → ${afterWalk.grown} after a full drag, `
    + `${afterWalk.cached} held`);

  await cdp.eval(`__aether.panel.setValues({ trees: false }, { notify: true }); true`);
  await sleep(300);
  const felled = await cdp.eval(`({ tri: __aether.scene.trees.triangles,
    on: __aether.scene.trees.uniforms().uCanopyOn })`);
  check('switching the wood off takes its shadow with it',
    felled.tri === 0 && felled.on === 0,
    `${felled.tri} triangles, canopy term ${felled.on}`);

  /* ── the lake ──
     Every one of these is really the same assertion asked three ways:
     that there is one answer to where the water is. The screenshots
     already show the shoreline; what a still cannot show is that the
     planting arrived at the same line by itself, from the same field,
     without anything having been told where to stop. */
  const sown = async (set) => {
    await cdp.eval(`__aether.panel.setValues(${JSON.stringify(set)},
      { notify: true }); true`);
    await sleep(450);
    return cdp.eval(`({ flowers: __aether.scene.ground.flowers,
      reeds: __aether.scene.ground.reeds,
      trees: __aether.scene.trees.trees })`);
  };

  /* Everything this section reads, set explicitly rather than inherited.
     It runs after the wood, which leaves its own switches where it likes,
     and a section that silently depended on the previous one's leftovers
     is a section that breaks the day somebody reorders them. */
  const dry = await sown({ ground: 'grass', flowers: true, flowerClumps: 1,
                           hills: 4.5, coverRadius: 60,
                           trees: true, water: false, weather: 'clear' });
  const flooded = await sown({ water: true, waterLevel: 1 });

  check('flooding the valleys drowns what was growing in them',
    flooded.flowers < dry.flowers * 0.85 && flooded.trees < dry.trees * 0.9,
    `dry ${dry.flowers} flowers / ${dry.trees} trees → `
    + `flooded ${flooded.flowers} / ${flooded.trees}`);

  /* And puts them back. This is the regression that would otherwise be
     invisible: every shader including the terrain block declares
     uWaterY, so an upload site left unfilled reads zero — which is a
     lake standing above every ridge in the world, and a meadow with
     nothing left growing in it. */
  const drained = await sown({ water: false });
  check('switching the lake off gives the whole meadow back',
    drained.flowers === dry.flowers && drained.trees === dry.trees,
    `${drained.flowers} flowers / ${drained.trees} trees, exactly as before`);

  /* The level control claims to be linear in flooded *area*, not depth.
     checkWater() holds it to that against the height field itself; this
     holds the planting to the same curve, from the other end of the
     pipeline — half the water should take away far less than all of it,
     and a control that had stayed linear in depth would fail here by
     taking away almost nothing. */
  const half = await sown({ water: true, waterLevel: 0.5 });
  check('the water level is linear in how much it floods, not how deep it is',
    half.flowers < dry.flowers * 0.95 && half.flowers > flooded.flowers * 1.1,
    `dry ${dry.flowers} · half ${half.flowers} · full ${flooded.flowers} flowers`);

  /* ── the reeds ──
     The other three plantings avoid the water; this one wants it. So the
     assertion is the mirror image of the one above: a lake that takes
     flowers away has to *give* reeds, and no lake has to leave none —
     not because a switch said so, but because there is no band for them
     to be in. That is the property worth guarding, since the way it
     would break is a fallback: a walk that gave up and dropped the reed
     where the cell happened to be would put a fringe of them through the
     middle of a dry meadow, and nothing else in the picture would say
     so. */
  const reeded = await sown({ reeds: true, water: true, waterLevel: 0.5 });
  const reedsDry = await sown({ water: false });
  check('the reeds want the water the rest of the planting avoids',
    reeded.reeds > 40 && reedsDry.reeds === 0,
    `${reeded.reeds} at half a lake → ${reedsDry.reeds} with no lake`);

  /* And they follow the shoreline rather than ring the lake at a fixed
     distance from wherever it was: raising the level moves the band,
     which is a different set of reeds and not the same set slid along.
     A cached sowing that had forgotten to key on the level would come
     back with the identical count. */
  const reedsHigh = await sown({ water: true, waterLevel: 1 });
  check('moving the water moves the band they stand in',
    reedsHigh.reeds > 0 && reedsHigh.reeds !== reeded.reeds,
    `${reeded.reeds} at level 0.5 → ${reedsHigh.reeds} at level 1`);
  await sown({ reeds: false, water: false });

  /* ── the weather ──
     Rain and snow are not two amounts of one thing, and the flowers are
     where that is checkable: snow buries them and rain does not touch
     them. A weather that had been built as one intensity would move both
     numbers together. */
  await sown({ water: false });
  const rained = await sown({ weather: 'rain' });
  const snowed = await sown({ weather: 'snow' });
  check('snow buries the flowers and rain leaves them where they are',
    rained.flowers === dry.flowers && snowed.flowers < dry.flowers * 0.7,
    `clear ${dry.flowers} · rain ${rained.flowers} · snow ${snowed.flowers}`);

  /* Not all of them. The mottle in snowCover is the whole reason snow
     reads as fallen rather than painted, and the way to see that it is
     doing anything is that the blooms go from the drifts and stay in the
     scoured patches. All-or-nothing here means the mottle is flat. */
  /* ── the cat in it ──
     Everything else in this scene answers the water by staying out of
     it. The cat is the one thing that can be driven into it, so it is
     the one thing that needs two answers: it wades while the ground is
     still holding it up and it swims when the ground has let go — and
     the threshold is a share of the animal's own height rather than a
     depth in metres, which is what makes it mean the same thing at any
     scale.

     What could break without showing: a cat walking along the bottom of
     the lake. It would be *invisible* while it happened, because the
     water is nearer to the eye than the animal and the composite would
     hide the whole thing — right up to the moment it walked back out of
     a lake it should have had to swim across. */
  const catNow = () => cdp.eval(`(() => {
    const c = __aether.scene.cat;
    return { swim: c.swim, wake: c.wake, ride: c.rideY,
             bed: c.floorY + c.footOffset, h: c.standH };
  })()`);
  const putCat = async (x, z) => {
    await cdp.eval(`(() => { const c = __aether.scene.cat;
      c.x = ${x}; c.z = ${z}; c.velocity = 0; c.strafeVel = 0;
      return true; })()`);
    await sleep(900);
    return catNow();
  };
  await cdp.eval(`__aether.panel.setValues({ cat: true, hills: 4.5,
    water: true, waterLevel: 1 }, { notify: true }); true`);
  await sleep(400);
  const onLand = await putCat(1.6, 1.6);
  const afloat = await putCat(-10.5, -7.5);

  check('deep water carries the cat rather than letting it walk the bottom',
    onLand.swim === 0 && afloat.swim > 0.9
    && Math.abs(onLand.ride - onLand.bed) < 1e-6
    && afloat.ride > afloat.bed + afloat.h * 0.15,
    `swim ${onLand.swim.toFixed(2)} ashore → ${afloat.swim.toFixed(2)} afloat, `
    + `held ${(afloat.ride - afloat.bed).toFixed(2)}u off the bed `
    + `(it stands ${afloat.h.toFixed(2)}u)`);

  /* And that the surface is told, harder while it is going somewhere.
     Zero on dry land is the half that matters: the rings are centred on
     the cat's bounding sphere, which exists whether or not it is in any
     water, so an animal walking past a lake could stir it from the
     bank. */
  const wakeIdle = (await catNow()).wake;
  await hold(['w'], 700);
  const wakeMoving = (await catNow()).wake;

  check('a cat in the water works the surface, and works it harder moving',
    onLand.wake === 0 && wakeIdle > 0 && wakeMoving > wakeIdle * 1.4,
    `dry ${onLand.wake.toFixed(2)} · floating ${wakeIdle.toFixed(2)} `
    + `· swimming ${wakeMoving.toFixed(2)}`);

  /* Put the animal back on dry land *and* take the lake away again. The
     sections after this one count what is flying over the meadow, and a
     flooded meadow has fewer flocks on it — which is correct behaviour
     and a wrong measurement, since what those checks are asking about is
     the reach control. Left on, it reads as the reach having stopped
     working. */
  await cdp.eval(`(() => { const c = __aether.scene.cat;
    c.x = 1.6; c.z = 1.6; c.velocity = 0;
    __aether.panel.setValues({ water: false }, { notify: true });
    return true; })()`);
  await sleep(300);

  /* ── and the sky over it ──
     The weather used to reach everything under the sky and nothing in
     it, so a midnight snowstorm came with a clear field of stars. What
     an overcast sky mostly is, is the *loss of the gradient*: clear air
     is deep overhead and pale at the horizon, and a cloud deck is a lit
     sheet that reads nearly the same brightness everywhere. So that is
     what is measured — two bands of sky, and the ratio between them
     collapsing toward one — rather than a colour, which would be a
     restatement of the constants.

     Sampled down the left quarter of the frame. This is an orbit camera
     and it always looks at the middle, so the middle is where the
     cluster is. */
  const skyBand = (a, b) => cdp.eval(`(() => {
    const c = document.getElementById('stage');
    const s = document.createElement('canvas');
    s.width = 32; s.height = 12;
    const g = s.getContext('2d');
    g.drawImage(c, 0, c.height * ${a}, c.width * 0.25, c.height * (${b} - ${a}),
                0, 0, 32, 12);
    const d = g.getImageData(0, 0, 32, 12).data;
    let sum = 0;
    for (let i = 0; i < d.length; i += 4) {
      sum += 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2];
    }
    return sum / (d.length / 4);
  })()`);
  const skyTilt = async (mode) => {
    /* Wrapped, and not for style: these run as top-level script in one
       long-lived context, so a bare `const s` here collides with the one
       another check declared an hour of test time ago. */
    await cdp.eval(`(() => {
      __aether.panel.setValues({ weather: '${mode}',
        daylight: 'hour', hour: 12 }, { notify: true });
      const sc = __aether.scene;
      sc.pitch = -0.5; sc.targetDist = 20;
      return true;
    })()`);
    await sleep(700);
    const high = await skyBand(0.03, 0.13);
    const low = await skyBand(0.42, 0.52);
    return { high, low, tilt: high / Math.max(low, 0.001) };
  };
  const openSky = await skyTilt('clear');
  const shutSky = await skyTilt('snow');

  check('cloud flattens the sky it is covering',
    openSky.tilt < 0.9
    && Math.abs(1 - shutSky.tilt) < Math.abs(1 - openSky.tilt) * 0.5
    && shutSky.low > openSky.low,
    `zenith/horizon ${openSky.tilt.toFixed(2)} clear → ${shutSky.tilt.toFixed(2)} `
    + `under cloud (luma ${openSky.high.toFixed(0)}/${openSky.low.toFixed(0)} `
    + `→ ${shutSky.high.toFixed(0)}/${shutSky.low.toFixed(0)})`);

  /* And the stars, which are the other half of the same claim and the
     one a picture is worst at judging: they are single pixels, so any
     downscale averages them into the sky before anyone can count them.

     This is a regression check with a real failure behind it. The first
     star field hashed a point inside a *cubic* cell and measured the
     distance from the ray to it — but the rays only ever sample the unit
     shell that passes through those cells, so a centre a tenth of a cell
     off the shell was already down to a thousandth of its brightness.
     The sky was not faint, it was empty, and nothing in the frame said
     so: an empty night sky looks exactly like a night sky. Counting
     bright pixels at full resolution is the only thing that does. */
  const starCount = () => cdp.eval(`(() => {
    const c = document.getElementById('stage');
    const s = document.createElement('canvas');
    // The middle of the frame, above the horizon: no logo, no panel.
    const x0 = Math.round(c.width * 0.25), y0 = Math.round(c.height * 0.05);
    s.width = Math.round(c.width * 0.45);
    s.height = Math.round(c.height * 0.22);
    const g = s.getContext('2d');
    g.drawImage(c, x0, y0, s.width, s.height, 0, 0, s.width, s.height);
    const d = g.getImageData(0, 0, s.width, s.height).data;
    let n = 0;
    for (let i = 0; i < d.length; i += 4) {
      if (0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2] > 20) n++;
    }
    return n;
  })()`);
  const skyAtNight = async (mode) => {
    await cdp.eval(`(() => {
      __aether.panel.setValues({ weather: '${mode}', daylight: 'hour', hour: 0,
        fireflies: 0, butterflies: 0, sparrowFlocks: 0 }, { notify: true });
      const sc = __aether.scene;
      sc.pitch = 0.06; sc.targetDist = 22;
      return true;
    })()`);
    await sleep(1600);
    return starCount();
  };
  const starry = await skyAtNight('clear');

  /* Only the clear half is counted. The other one cannot be measured
     this way and it would be a lie to pretend otherwise: a covered sky
     comes with something falling out of it, and a snowflake in front of
     the camera is a bright pixel in exactly this box. What the cloud
     does to the stars is held by the flattening check above, by the
     construction — the star term is multiplied by (1 - overcast) — and
     by 48-night-snow, where it is the whole subject of the frame. */
  check('a clear night has stars in it',
    starry > 30,
    `${starry} lit pixels in the clear midnight sky`);

  await cdp.eval(`__aether.panel.setValues({ weather: 'snow', daylight: 'fixed',
    light: [0.68, 0.24] }, { notify: true });
    __aether.scene.pitch = 0.16; true`);
  await sleep(400);

  check('snow takes the drifts and leaves the scoured patches',
    snowed.flowers > 0,
    `${snowed.flowers} of ${dry.flowers} flowers still standing in the open`);

  await sown({ weather: 'clear' });

  /* ── the time of day ──
     The light direction is a Float32Array the scene publishes, so where
     the sun is can simply be read. What is being checked is not that the
     arithmetic runs but that the *authority* is where the mode says: a
     time of day and an XY pad are two answers to one question, and the
     failure that matters is both of them being half in charge. */
  const lit = async (set) => {
    await cdp.eval(`__aether.panel.setValues(${JSON.stringify(set)},
      { notify: true }); true`);
    await sleep(260);
    return cdp.eval(`({ dir: Array.from(__aether.scene.lightDir),
      hour: __aether.scene.hour })`);
  };

  const noon = await lit({ daylight: 'hour', hour: 12 });
  const morning = await lit({ hour: 7.5 });
  const evening = await lit({ hour: 16.5 });
  const midnight = await lit({ hour: 0 });

  /* Morning and evening are taken well inside the day rather than at
     06:00 and 18:00. Those two are the tie: the sun and the moon are
     both exactly on the horizon and which one is chosen is arbitrary, so
     testing there tests the tie-break and not the arc. Equidistant from
     noon instead — the same height, opposite sides — which is the claim
     that there is one arc and not two hand-placed bodies. */
  check('the hour puts the sun on an arc, and the moon on the same one',
    noon.dir[1] > 0.85
    && Math.abs(morning.dir[1] - evening.dir[1]) < 1e-3
    && morning.dir[0] * evening.dir[0] < 0
    /* And midnight is the moon, at the noon sun's height and — this is
       the part worth having a test for — at the noon sun's *bearing*.
       It looks like a bug and is the definition: a moon opposite the sun
       is full, and a full moon at midnight is exactly where the sun is
       at noon. Falling out of one arc read backwards rather than being
       placed is what makes it come out right without being aimed. */
    && midnight.dir[1] > 0.85
    && Math.abs(midnight.dir[2] - noon.dir[2]) < 1e-3,
    `noon y ${noon.dir[1].toFixed(2)} · 07:30 and 16:30 both y `
    + `${morning.dir[1].toFixed(3)}, x ${morning.dir[0].toFixed(2)}/`
    + `${evening.dir[0].toFixed(2)} · midnight moon y `
    + `${midnight.dir[1].toFixed(2)} on the noon bearing (full)`);

  /* And the reason the swap at dusk cannot be seen. At the moment the
     two bodies trade places both are on the horizon, so whichever one
     wins is delivering a grazing light that the whole meadow is nearly
     edge-on to — the direction flips, and there is no directional term
     left for the flip to show up in. */
  const terminator = await lit({ hour: 18 });
  check('the light has nothing left to give at the moment it swaps bodies',
    Math.abs(terminator.dir[1]) < 0.02,
    `at 18:00 the light sits ${terminator.dir[1].toFixed(4)} above the horizon`);

  /* The pad, and whether it is listened to. Under the hour it must not
     be — a control that still moves the light after another one has been
     put in charge of it is the worst of the three possible designs,
     because nothing on screen says which of them you are looking at. */
  const padBase = await lit({ hour: 9 });
  const padUnderHour = await lit({ light: [0.1, 0.9] });
  const padUnderFixed = await lit({ daylight: 'fixed' });

  check('the hour takes the pad off the light, and giving it back works',
    Math.abs(padUnderHour.dir[1] - padBase.dir[1]) < 1e-6
    && Math.abs(padUnderFixed.dir[1] - padBase.dir[1]) > 0.2,
    `pad dragged under the hour: y ${padBase.dir[1].toFixed(3)} → `
    + `${padUnderHour.dir[1].toFixed(3)} (unmoved); back on fixed: y `
    + `${padUnderFixed.dir[1].toFixed(3)}`);

  /* And that the clock is what advances it, not the wall — so pausing
     stops the sun, and the reference shots stay reproducible. */
  await lit({ daylight: 'cycle' });
  await cdp.eval(`__aether.clock.setPaused(true); true`);
  await sleep(120);
  const frozen = await cdp.eval(`__aether.scene.hour`);
  await sleep(700);
  const stillFrozen = await cdp.eval(`__aether.scene.hour`);
  await cdp.eval(`__aether.clock.setPaused(false); true`);
  await sleep(600);
  const sunMoved = await cdp.eval(`__aether.scene.hour`);

  check('a paused scene stops the sun, and running it starts it again',
    stillFrozen === frozen && sunMoved !== frozen,
    `${frozen.toFixed(3)}h held for 0.7 s paused, then ${sunMoved.toFixed(3)}h`);

  await lit({ daylight: 'fixed', light: [0.68, 0.24] });

  /* ── the population ──
     One slider each, and the claim about what each one costs. A
     butterfly is four triangles; a firefly is none at all, because it is
     a light and not a surface. */
  const life = async (b, f) => {
    await cdp.eval(`__aether.panel.setValues({ butterflies: ${b}, fireflies: ${f} },
      { notify: true }); true`);
    await sleep(300);
    return cdp.eval(`({ b: __aether.scene.creatures.butterflies,
      f: __aether.scene.creatures.fireflies,
      tri: __aether.scene.creatures.triangles })`);
  };
  const none = await life(0, 0);
  const flies = await life(1, 0);
  const glows = await life(0, 1);

  check('each kind of creature has its own control',
    none.b === 0 && none.f === 0
    && flies.b > 0 && flies.f === 0
    && glows.b === 0 && glows.f > 0,
    `off ${none.b}/${none.f} · butterflies ${flies.b}/${flies.f} `
    + `· fireflies ${glows.b}/${glows.f}`);

  check('a butterfly is four triangles and a firefly is none',
    flies.tri === flies.b * 4 && glows.tri === 0,
    `${flies.b} butterflies → ${flies.tri} triangles, `
    + `${glows.f} fireflies → ${glows.tri}`);

  /* And the bird, which is the one that had to be paid for. Seventy
     triangles against four is the trade this whole file argues about, so
     the number is held to rather than merely described — and the count
     being a whole number of flocks is what says the flocking survived,
     since a population that had quietly gone back to scattering would
     still report the right total. */
  await cdp.eval(`__aether.panel.setValues({ butterflies: 0, fireflies: 0,
    sparrows: 1, sparrowFlocks: 1 }, { notify: true }); true`);
  await sleep(320);
  const birds = await cdp.eval(`({ n: __aether.scene.creatures.sparrows,
    tri: __aether.scene.creatures.triangles })`);

  check('a sparrow costs seventy triangles',
    birds.n > 0 && birds.tri === birds.n * 70,
    `${birds.n} sparrows → ${birds.tri} triangles`);

  /* The flock structure, which is the whole of the second redesign. What
     it is guarding is that the *control* moves the size of a flock and
     not the number of them: a slider that added groups instead of
     members is how a sky of birds turns back into a scatter of flies,
     which is the specific thing that was wrong. So the count of live
     flocks has to sit between three and six at every setting, and the
     total has to move anyway. */
  const flockAt = async (d) => {
    await cdp.eval(`__aether.panel.setValues({ sparrows: ${d}, sparrowFlocks: 1 },
      { notify: true }); true`);
    await sleep(300);
    return cdp.eval(`__aether.scene.creatures.sparrows`);
  };
  const few = await flockAt(0.35);
  const many = await flockAt(1);

  check('the control sets how big a flock is, not how many there are',
    few > 0 && many > few * 1.8,
    `density 0.35 → ${few} birds, density 1 → ${many}`);

  /* And the other one, which sets how many. The pair is the flowers'
     bargain — a few big flocks and many small ones are two different
     skies, and one slider reaches both ends without ever saying which of
     them it is doing — so what has to hold is that they move different
     things: halving the groups roughly halves the birds while leaving
     each group the size it was. */
  await cdp.eval(`__aether.panel.setValues({ sparrows: 1, sparrowFlocks: 1 },
    { notify: true }); true`);
  await sleep(320);
  const allFlocks = await cdp.eval(`__aether.scene.creatures.sparrows`);
  await cdp.eval(`__aether.panel.setValues({ sparrowFlocks: 0.4 },
    { notify: true }); true`);
  await sleep(320);
  const someFlocks = await cdp.eval(`__aether.scene.creatures.sparrows`);

  check('the second control sets how many flocks there are',
    someFlocks > 0 && someFlocks < allFlocks * 0.75,
    `all flocks → ${allFlocks} birds, 40% of them → ${someFlocks}`);

  /* And that a wood offers somewhere to sit. The perches are the real
     canopies out of the list the shadow map is drawn from, so this is
     also the check that the two halves of the wood — the one that is
     drawn and the one that is sat in — are the same wood. */
  await cdp.eval(`__aether.panel.setValues({ trees: true }, { notify: true }); true`);
  await sleep(500);
  /* Every pass is one of three things and the wood decides which: aimed
     at a crown, put down on open ground, or straight across without
     stopping. The line's w flags a tree and the stop's w flags whether
     there is a stop at all, so both can be counted — and with a wood on
     the map some of them must be trees, or the crowns are not reaching
     the flight at all. */
  const kinds = await cdp.eval(`(() => {
    const c = __aether.scene.creatures, L = c._flockLine, S = c._flockStop;
    const I = c._flockInfo;
    let tree = 0, ground = 0, through = 0;
    for (let i = 0; i < L.length / 4; i++) {
      if (I[i * 4] <= 0) continue;
      if (S[i * 4 + 3] < 0) through++;
      else if (L[i * 4 + 3] > 0) tree++;
      else ground++;
    }
    return { tree, ground, through };
  })()`);

  check('a wood gives the birds somewhere to perch, at the real canopies',
    kinds.tree > 0,
    `${kinds.tree} passes aimed at a crown, ${kinds.ground} onto open `
    + `ground, ${kinds.through} straight across`);

  await cdp.eval(`__aether.panel.setValues({ trees: false }, { notify: true }); true`);
  await sleep(200);

  /* The reach, which is one control over three ranges — the same bargain
     the cover reach makes with the grass, the flowers and the wood.
     What has to hold is that it buys *area* and not thinning: the
     populations grow with the patch, because a density that fell as the
     view widened would make the two sliders the same slider.

     The sparrows are the exception and are checked for being one. A
     flock is the unit there, so a wider patch spreads the flocks apart
     and does not add birds — the count must stay put. */
  const reachAt = async (r) => {
    await cdp.eval(`__aether.panel.setValues({ butterflies: 1, fireflies: 1,
      sparrows: 1, sparrowFlocks: 1, lifeRadius: ${r} }, { notify: true }); true`);
    await sleep(400);
    return cdp.eval(`({ b: __aether.scene.creatures.butterflies,
      f: __aether.scene.creatures.fireflies,
      s: __aether.scene.creatures.sparrows })`);
  };
  const lifeNear = await reachAt(15);
  const lifeFar = await reachAt(120);

  check('the creature reach buys ground, and does not thin what is on it',
    lifeFar.b > lifeNear.b * 2 && lifeFar.f > lifeNear.f * 2,
    `15u → ${lifeNear.b} butterflies / ${lifeNear.f} fireflies, `
    + `120u → ${lifeFar.b} / ${lifeFar.f}`);

  /* Seeing further has to put more in the sky — for both kinds — but it
     does it to them differently, and that difference is the claim.

     Insects are a scatter on a patch, so their count follows the area
     outright: fourteen times over this range. Flocks live on a grid of
     fixed world spacing, so their count follows the area too, up to the
     ceiling on how many can be handed over at once, and then it stops.
     The birds grow, and grow by less.

     This check used to assert the opposite — that the bird count stayed
     put — and it was right about the code and wrong about what the code
     should do. The grid was a fraction of the reach; the two cancelled
     in the area, and the control moved the horizon with nothing new
     arriving at it. */
  check('seeing further puts more in the sky, and most of all the small things',
    lifeFar.s > lifeNear.s * 1.5
    && lifeFar.b > lifeNear.b * 8
    && lifeFar.b / lifeNear.b > lifeFar.s / lifeNear.s,
    `sparrows ${lifeNear.s} → ${lifeFar.s} across the range, `
    + `where butterflies go ${lifeNear.b} → ${lifeFar.b}`);

  /* ── and what time it is ──
     Three populations, one hour, and the point is that they do not move
     together. Noon has butterflies and birds and no fireflies; midnight
     has the reverse and nothing else. A "wildlife" amount that had crept
     back in — one number scaling all three — would fail here by moving
     them the same way, and so would a day cycle that only changed the
     light. */
  const hourLife = async (hour) => {
    await cdp.eval(`__aether.panel.setValues({ daylight: 'hour', hour: ${hour},
      butterflies: 1, fireflies: 1, sparrows: 1, sparrowFlocks: 1,
      lifeRadius: 60 }, { notify: true }); true`);
    await sleep(360);
    return cdp.eval(`({ b: __aether.scene.creatures.butterflies,
      f: __aether.scene.creatures.fireflies,
      s: __aether.scene.creatures.sparrows })`);
  };
  const upByDay = await hourLife(12);
  const upByNight = await hourLife(0);

  check('the hour decides what is alive, and not all of it the same way',
    upByDay.b > 0 && upByDay.s > 0 && upByDay.f === 0
    && upByNight.b === 0 && upByNight.s === 0 && upByNight.f > 0,
    `noon ${upByDay.b} butterflies / ${upByDay.s} sparrows / ${upByDay.f} fireflies · `
    + `midnight ${upByNight.b} / ${upByNight.s} / ${upByNight.f}`);

  /* And the mode that has no hour in it keeps every population it always
     had. `day` is exactly 1 with the pad in charge, which is already the
     right answer for anything that flies by day and exactly the wrong
     one for something nocturnal — so the firefly is the only one that
     has to ask whether a clock exists, and this is the check that says
     it still does. Every reference frame taken before there was a time
     of day depends on it. */
  await cdp.eval(`__aether.panel.setValues({ daylight: 'fixed',
    light: [0.68, 0.24] }, { notify: true }); true`);
  await sleep(360);
  const untimed = await cdp.eval(`({ b: __aether.scene.creatures.butterflies,
    f: __aether.scene.creatures.fireflies,
    s: __aether.scene.creatures.sparrows })`);

  check('a scene with no time of day has nothing nocturnal in it',
    untimed.b > 0 && untimed.f > 0 && untimed.s > 0,
    `fixed light: ${untimed.b} butterflies / ${untimed.f} fireflies `
    + `/ ${untimed.s} sparrows, all of them out at once`);

  /* Turning the camera must not change what is in the sky.
     This is a regression, and the bug it is guarding was reported from
     the running scene rather than found here: the flocks used to be
     wrapped into a box centred on the eye, and this is an orbit camera,
     so a yaw drag translates the eye and re-wrapped them — whole groups
     faded out at one edge and reappeared at the other. With a few
     hundred insects that is invisible; with four flocks it is the most
     obvious thing in the frame.

     Swept on a tight orbit so the eye stays inside one flock cell, which
     is what isolates "the camera turned" from "the camera travelled".

     And at a *short* reach, deliberately. The count has to be small
     enough that one group is a large share of it, or the test loses the
     only thing it can see: with a hundred and ten birds in two dozen
     flocks, a whole group blinking is six per cent of the total and
     hides inside the ordinary breathing of everything at the rim. With a
     handful of flocks it is a fifth, and unmissable. */
  await cdp.eval(`const s = __aether.scene; s.targetDist = 2.5; s.yaw = 0;
    __aether.panel.setValues({ sparrows: 1, sparrowFlocks: 1, lifeRadius: 28 },
      { notify: true }); true`);
  await sleep(400);
  const spun = [];
  for (const yaw of [0, 1.6, 3.1, 4.7]) {
    await cdp.eval(`__aether.scene.yaw = ${yaw}; true`);
    await sleep(220);
    spun.push(await cdp.eval(`__aether.scene.creatures.sparrows`));
  }

  /* Not "unchanging" — that was tried and it was the wrong target. An
     orbit camera turning is an orbit camera *travelling*: nine metres at
     this distance, forty at twenty. The observer really is somewhere
     else, and a population that ignored that would be pinned to the
     view. Anchoring on the orbit centre did make the number constant,
     and it did so by nailing the birds to the point the camera always
     looks at — the picture was worse and the test was greener, which is
     the whole hazard of measuring the wrong invariant.

     What must not happen is a flock appearing or disappearing *whole*.
     The count is fade-weighted, so a group popping in at full brightness
     moves it by its own size — seven to fifteen birds — while a group
     fading across the rim moves it by two or three. Watching the largest
     single step separates those two, and it is the difference that was
     being reported from the running scene. */
  let jump = 0;
  for (let i = 1; i < spun.length; i++) {
    jump = Math.max(jump, Math.abs(spun[i] - spun[i - 1]));
  }
  check('no flock appears or disappears whole as the camera turns',
    spun[0] > 0 && jump < 6,
    `${spun.join(' → ')} sparrows through a full turn `
    + `(largest step ${jump}, a whole flock is 7–15)`);

  await cdp.eval(`__aether.panel.setValues({ butterflies: 0, fireflies: 0,
    sparrows: 0, sparrowFlocks: 0, lifeRadius: 45 }, { notify: true }); true`);
  await sleep(200);

  await cdp.eval(`__aether.panel.setValues({ sparrows: 0, sparrowFlocks: 0 }, { notify: true }); true`);
  await sleep(200);

  await life(0, 0);
  await cdp.eval(`__aether.panel.setValues({ hills: 2, coverRadius: 15,
    flowerClumps: 0.62, trees: false }, { notify: true }); true`);
  await sleep(300);


  /* Visibility is a master, in the same sense the quality slider is: it
     writes into the reach and then goes stale, so the reach stays yours
     afterwards. Locking the two together would forbid thick fog with a
     long reach, which is both cheap and worth having. */
  await cdp.eval(`__aether.panel.setValues({ visibility: 180 }, { notify: true }); true`);
  await sleep(400);
  const wide = await cdp.eval(`({ reach: __aether.state.coverRadius,
    radius: __aether.scene.ground.radius, rings: __aether.scene.ground.rings })`);
  check('visibility drives the cover reach with it',
    wide.reach === 162 && Math.abs(wide.radius - 162) < 1,
    `visibility 180 → reach ${wide.reach}u, ${wide.rings} rings`);

  /* One reach, three consumers. Each keeps its own ratio and its own
     ceiling — flowers stop at 60, because their clump grid is a fixed
     four metres and their count grows with the square of the reach;
     trees stop at 90, where branch segments, which have no level of
     detail, would start to fill their buffer. */
  await cdp.eval(`__aether.panel.setValues(
    { flowers: true, trees: true }, { notify: true }); true`);
  await sleep(700);
  const reaches = await cdp.eval(`({ grass: __aether.scene.ground.radius,
    flowers: __aether.scene.ground.flowerRadius, trees: __aether.scene.trees.reach })`);
  check('one reach carries the grass, the flowers and the wood',
    Math.abs(reaches.grass - 162) < 1
    && Math.abs(reaches.flowers - 60) < 0.5
    && Math.abs(reaches.trees - 90) < 0.5,
    `grass ${reaches.grass.toFixed(0)}u, flowers ${reaches.flowers.toFixed(0)}u, `
    + `trees ${reaches.trees.toFixed(0)}u`);
  await cdp.eval(`__aether.panel.setValues(
    { flowers: false, trees: false }, { notify: true }); true`);
  await sleep(200);

  await cdp.eval(`__aether.panel.setValues({ coverRadius: 30 }, { notify: true }); true`);
  await sleep(400);
  const overridden = await cdp.eval(`({ reach: __aether.state.coverRadius,
    vis: __aether.state.visibility })`);
  check('and then lets go of it',
    overridden.reach === 30 && overridden.vis === 180,
    `reach pulled back to ${overridden.reach}u with visibility still ${overridden.vis}u`);

  await cdp.eval(`__aether.panel.setValues(
    { visibility: 66, coverRadius: 15 }, { notify: true }); true`);
  await sleep(300);

  /* Grass in wind never comes to rest, so the temporal filter must not
     be allowed to believe the frame has settled. Tested with everything
     else in the scene held still, or the flag would be true anyway. */
  await cdp.eval(`__aether.panel.setValues(
    { spin: false, cat: false, wind: 0 }, { notify: true }); true`);
  /* Polled rather than slept on. The check just above drags the camera,
     and the camera eases to a stop over an interval nobody has promised
     is shorter than any particular sleep — a fixed wait here failed
     about one run in five, which is worse than no check at all. */
  let still = 1;
  for (let i = 0; i < 25 && still !== 0; i++) {
    await sleep(120);
    still = await cdp.eval(`__aether.scene.moving`);
  }
  await cdp.eval(`__aether.panel.setValues({ wind: 0.8, ground: 'grass',
    trees: false, water: false }, { notify: true }); true`);
  await sleep(400);
  const blowing = await cdp.eval(`({ scene: __aether.scene.moving,
    mesh: __aether.scene.movingMesh })`);

  /* Wind stops the *meadow* settling, and leaves the rest of the frame
     alone. The blades are triangles that cast nothing into the marched
     half, so the sky, the cluster, the hills and the lake go on
     converging while the grass blows — which they did not when one blend
     covered the whole frame, and with the default wind that was every
     frame the cover was on. */
  check('wind stops the meadow settling and lets the rest of the frame converge',
    still === 0 && blowing.mesh === 1 && blowing.scene === 0,
    `with wind: marched ${blowing.scene}, rasterised ${blowing.mesh}`);

  /* A wood is the exception and is kept whole: the floor the marcher
     draws reads the canopy's shadow map, so leaves moving in the wind
     move pixels on the marched side of the composite. */
  await cdp.eval(`__aether.panel.setValues({ trees: true }, { notify: true }); true`);
  await sleep(500);
  const wooded = await cdp.eval(`__aether.scene.moving`);
  check('a wood in wind moves the marched half too, through its shadow',
    wooded === 1, `moving ${wooded} with the wood on`);
  await cdp.eval(`__aether.panel.setValues({ trees: false }, { notify: true }); true`);

  /* reset puts back everything: parameters, camera and the URL. It used
     to do only half of that, with the other half hidden in a separate
     command. */
  await cdp.eval(`__aether.panel.setValues(
    { tint: 'rose', steps: 180, erode: 0.4 }, { notify: true }); true`);
  await drag([700, 450], [1050, 340]);
  await sleep(400);
  const messy = await cdp.eval(`({ tint: __aether.state.tint, steps: __aether.state.steps,
    erode: __aether.state.erode, yaw: +__aether.scene.yaw.toFixed(2) })`);

  await key('r', { vk: 82 });
  await sleep(600);
  const clean = await cdp.eval(`({ tint: __aether.state.tint, steps: __aether.state.steps,
    erode: __aether.state.erode, yaw: +__aether.scene.yaw.toFixed(2),
    hash: location.hash })`);

  check('reset restores every parameter',
    clean.tint === 'amber' && clean.steps === 100 && clean.erode === 0.1,
    `${JSON.stringify(messy)} → ${JSON.stringify(clean)}`);
  check('reset restores the camera and the URL',
    Math.abs(clean.yaw - 0.85) < 0.2 && clean.hash === '#/march',
    `yaw ${messy.yaw} → ${clean.yaw}, hash ${clean.hash}`);

  check('no console errors during interaction', problems.length === 0,
    problems.map((p) => p.text.slice(0, 120)).join(' | '));
  await shot(cdp, 'i7-final');

  const failed = results.filter((r) => !r.pass);
  console.log(failed.length ? `\n▸ ${failed.length}/${results.length} FAILED`
                            : `\n▸ all ${results.length} interaction checks passed`);
  if (failed.length) process.exitCode = 1;
}

async function shot(cdp, name) {
  const { data } = await cdp.send('Page.captureScreenshot', { format: 'png' });
  await writeFile(join(SHOTS, `${name}.png`), Buffer.from(data, 'base64'));
}

/**
 * Poll the page until an expression reaches a value.
 *
 * The boot budget is generous — a minute and a half — and it is worth
 * saying why, because a long timeout usually hides something.
 *
 * Booting this app compiles a dozen large programmes, and on integrated
 * graphics under a loaded machine that is genuinely slow: measured here
 * at a hundred seconds on a laptop with the user's own browser running,
 * against a handful of seconds on an idle one. The old twenty-second
 * budget turned that into a bare "boot failed" with no exception and no
 * console error, which is the least useful failure a harness can
 * produce — it looks exactly like a hang.
 *
 * So the budget is large and the *time taken* is reported instead. A
 * compile-time regression then shows up as a number climbing rather than
 * as an intermittent mystery, and a busy machine is merely slow.
 */
async function waitFor(cdp, expression, expected, timeout) {
  const started = Date.now();
  const deadline = started + timeout;
  while (Date.now() < deadline) {
    try {
      const v = await cdp.eval(expression);
      if (v === expected) {
        const took = Date.now() - started;
        if (took > 8000) {
          console.log(`  (boot took ${(took / 1000).toFixed(1)} s — `
            + 'shader compilation on a loaded machine)');
        }
        return v;
      }
      if (v === 'failed') throw new Error('App reported boot failure');
    } catch (err) {
      if (String(err).includes('boot failure')) throw err;
    }
    await sleep(200);
  }
  throw new Error(`Timed out waiting for ${expression} === ${expected} `
    + `after ${(timeout / 1000).toFixed(0)} s`);
}

function truncate(s, n) { return s.length > n ? `${s.slice(0, n)}…` : s; }

main().catch((err) => { console.error('\n✗', err); process.exit(1); });
