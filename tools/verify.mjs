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
  { name: '23-meadow', hash: '#/march?spin=0&scale=1&taa=0.9&ground=meadow&cat=0&cover=1',
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
    hash: '#/march?spin=0&scale=1&taa=0.9&ground=meadow&trees=1&cat=0&coverRadius=30',
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
    hash: '#/march?spin=0&scale=1&taa=0.9&ground=meadow&trees=1&camera=follow&shadow=1',
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
    hash: '#/march?spin=0&scale=1&taa=0.9&ground=meadow&trees=1&cat=0&visibility=200',
    settle: 3000, freeze: 8.0,
    pre: '__aether.scene.laser.silence(); return true;',
    poke: `const s = __aether.scene;
           s.yaw = 0.85; s.pitch = 0.09; s.targetDist = 14;
           return true;` },

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
    hash: '#/march?spin=0&scale=1&taa=0.9&ground=meadow&camera=follow&shadow=1&ao=1&light=0.68,0.667',
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

async function main() {
  await lintShaders();
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
    await waitFor(cdp, `document.documentElement.dataset.boot`, 'ready', 20000);
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
    await waitFor(cdp, 'document.documentElement.dataset.boot', 'ready', 20000);
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
      await waitFor(cdp, `document.documentElement.dataset.boot`, 'ready', 20000);
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
  await waitFor(cdp, 'document.documentElement.dataset.boot', 'ready', 20000);
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
  { name: 'meadow',      hash: '#/march?spin=0&taa=0&scale=1&ground=meadow&cover=1' },
  { name: 'wood',        hash: '#/march?spin=0&taa=0&scale=1&ground=grass&trees=1' },
  { name: 'wood+meadow', hash: '#/march?spin=0&taa=0&scale=1&ground=meadow&cover=1&trees=1' },
];

async function bench(cdp) {
  const base = `http://127.0.0.1:${cdp.__port}/index.html`;
  const seconds = Number(value('seconds', 6));

  console.log(`▸ benchmark  (${seconds}s per case, ${VIEWPORT.width}×${VIEWPORT.height})`);

  let renderer = null;
  const rows = [];

  for (const c of BENCH_CASES) {
    await cdp.send('Page.navigate', { url: base + c.hash });
    await waitFor(cdp, 'document.documentElement.dataset.boot', 'ready', 20000);
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
  await waitFor(cdp, 'document.documentElement.dataset.boot', 'ready', 20000);

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
        best = { tip, mid, deflect: Math.abs(w.whiskers[16]), tailDeflect: Math.abs(w.nodes[17]) };
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

  await cdp.eval(`__aether.panel.setValues({ ground: 'grass' }, { notify: true }); true`);
  await sleep(300);
  const onGrass = await cdp.eval(`({ tri: __aether.scene.ground.triangles,
    blades: __aether.scene.ground.blades, flowers: __aether.scene.ground.flowers })`);
  check('grass covers the floor and grows no flowers',
    onGrass.blades > 1000 && onGrass.tri > 1000 && onGrass.flowers === 0,
    `${onGrass.blades} blades, ${onGrass.tri} triangles, ${onGrass.flowers} flowers`);

  await cdp.eval(`__aether.panel.setValues({ ground: 'meadow' }, { notify: true }); true`);
  await sleep(300);
  const onMeadow = await cdp.eval(`({ blades: __aether.scene.ground.blades,
    flowers: __aether.scene.ground.flowers })`);
  check('the meadow keeps the grass and adds flowers to it',
    onMeadow.blades === onGrass.blades && onMeadow.flowers > 100,
    `${onMeadow.blades} blades, ${onMeadow.flowers} flower slots`);

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
    { trees: true, ground: 'grass', wind: 0, spin: false, cat: false },
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
  await cdp.eval(`__aether.panel.setValues({ wind: 0.8 }, { notify: true }); true`);
  await sleep(400);
  const blowing = await cdp.eval(`__aether.scene.moving`);
  check('wind stops the accumulation buffer from settling',
    still === 0 && blowing === 1,
    `moving ${still} with no wind, ${blowing} with wind`);

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

async function waitFor(cdp, expression, expected, timeout) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    try {
      const v = await cdp.eval(expression);
      if (v === expected) return v;
      if (v === 'failed') throw new Error('App reported boot failure');
    } catch (err) {
      if (String(err).includes('boot failure')) throw err;
    }
    await sleep(200);
  }
  throw new Error(`Timed out waiting for ${expression} === ${expected}`);
}

function truncate(s, n) { return s.length > n ? `${s.slice(0, n)}…` : s; }

main().catch((err) => { console.error('\n✗', err); process.exit(1); });
