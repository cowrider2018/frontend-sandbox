/* ── tools/bake-cat.mjs ──────────────────────────────────────────────
   Bake the procedural cat in tools/cat-model.js into a flat binary the
   app can draw without three.js.

     node tools/bake-cat.mjs [--skins a,b,c] [--out src/scenes/cat/cat.bin]

   WHY THIS EXISTS
   ---------------
   `tools/cat-model.js` builds the cat out of ~40 deformed spheres, tubes
   and boxes, and it takes THREE as an argument rather than importing it,
   which is what lets this run headless. We run it once, here, in Node,
   and throw the library away: what ships is vertices.

   WHAT COMES OUT
   --------------
   The cat animates as a *rigid* hierarchy — every part is a whole mesh
   swinging on a joint, and there is not one skinning weight anywhere in
   the model. So the bake does not need a skeleton in the glTF sense. It
   needs two things:

     bones   the handful of nodes `applyPose` actually writes to, each
             with its rest transform and its parent
     meshes  every vertex, pre-transformed into the space of its nearest
             bone ancestor

   At runtime the cat is then one uniform array of bone matrices and
   three draw calls. The rest transform between a bone and a mesh hanging
   off it never changes, so it is folded into the vertices here and never
   computed again.
   ------------------------------------------------------------------ */

import { writeFile, mkdir } from 'node:fs/promises';
import { dirname, join, normalize, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = normalize(join(dirname(fileURLToPath(import.meta.url)), '..'));

const args = process.argv.slice(2);
const value = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};

/* Both the model and the library it is written against live here, in
   tools/. Nothing outside this repository is read, so the cat can still
   be rebuilt long after wherever it came from is gone — which is the
   only reason to keep a bake step at all rather than shipping the
   binary and forgetting how it was made.

   tools/ is cold path: none of this is served, bundled or imported by
   the app, which still has no runtime dependencies whatsoever. */
const THREE_PATH = resolve(ROOT, 'tools/vendor/three.module.min.js');
const MODEL_PATH = resolve(ROOT, 'tools/cat-model.js');

const OUT = resolve(ROOT, value('out', 'src/scenes/cat/cat.bin'));

/* Every colourway goes into one file. They differ only in vertex
   colour — same vertices, same normals, same indices, same bones —
   so paying for the geometry once and the palettes three times is
   most of a megabyte cheaper than three whole cats, and switching
   between them at runtime is one buffer upload instead of a reload. */
const SKINS_WANTED = value('skins', 'orangin,tabby,calico').split(',');

const THREE = await import(pathToFileURL(THREE_PATH).href);
const { buildCat, SKINS } = await import(pathToFileURL(MODEL_PATH).href);

for (const name of SKINS_WANTED) {
  if (SKINS[name]) continue;
  console.error(`unknown skin "${name}" — have: ${Object.keys(SKINS).join(', ')}`);
  process.exit(1);
}

/* ═══ build ═══════════════════════════════════════════════════════ */

/**
 * Walk one colourway of the model into flat arrays.
 *
 * Called once per skin. Everything but the vertex colours comes out
 * identical every time — the palette cannot move a vertex — and the
 * caller asserts exactly that rather than assuming it.
 */
function collect(SKIN) {
const parts = buildCat(THREE, { skin: SKIN, gradientMap: null });

/**
 * The nodes `applyPose` writes to, in the order the runtime will hold
 * them. Everything else in the graph is static and gets folded away.
 *
 * `legs` and `torso` are siblings under `root` on purpose — that is what
 * lets the model tilt the body without the legs following, so the cat
 * leans while its feet stay planted. Preserving that split matters more
 * than the handful of matrices it costs.
 */
// Rest pose is whatever buildCat left behind: matrixWorld for every node.
parts.root.updateMatrixWorld(true);

/**
 * The model's own registry of bendable geometry. Each entry carries a
 * per-vertex "outerness" — 0 at the base, 1 at the tip — which is the
 * only thing needed to make a part trail behind the body it hangs off.
 * Keyed by geometry, because that is what a mesh can be matched on.
 */
const flexByGeo = new Map((parts.flex || []).map((f) => [f.geo, f]));

/**
 * The whiskers, which bend around a pivot out at the cheek rather than
 * around the origin of the bone they hang off. That is the one thing
 * that stopped them being done with the tail: a shader that rotates
 * about a bone's origin cannot bend them where they actually hinge.
 *
 * So each one is given a bone of its own, placed at its pivot. Nothing
 * poses these — they exist purely to put an origin in the right place,
 * and after that the whiskers use the same machinery as the tail.
 */
const whiskers = [];
parts.root.traverse((o) => {
  if (o.isMesh && flexByGeo.get(o.geometry)?.src === 'head') whiskers.push(o);
});

const BONES = [
  ['root',    parts.root],
  ['torso',   parts.torso],
  ['legs',    parts.legs],
  ['tail',    parts.tail],
  ['bodyPivot', parts.bodyPivot],
  ['body',    parts.body],
  ['head',    parts.head],
  ['earL',    parts.earL],
  ['earR',    parts.earR],
  ['hipHL',   parts.hipHL],
  ['hipHR',   parts.hipHR],
  ['pawHL',   parts.pawHL],
  ['pawHR',   parts.pawHR],
  ['pawFL',   parts.pawFL],
  ['pawFR',   parts.pawFR],
  ...parts.eyes.map((e, i) => [`eye${i}`, e]),
  ...parts.pupils.map((p, i) => [`pupil${i}`, p]),
  // Appended last, so every bone still follows its parent in the array
  // and the runtime's single forward pass stays valid.
  ...whiskers.map((w, i) => [`whisker${i}`, w, flexByGeo.get(w.geometry).pivot]),
].filter(([, obj]) => obj);

/* The bone index shares a byte with the sway group, five bits to three.
   Nothing here is close to the limit, but a silent wrap would show up as
   a stray triangle welded to the wrong joint. */
if (BONES.length > 32) throw new Error(`too many bones for a 5-bit index: ${BONES.length}`);

const boneIndex = new Map(BONES.map(([, obj], i) => [obj, i]));

/**
 * Where each bone actually sits at rest. Usually its object's own place
 * in the world; for the whiskers, shifted out to the pivot they hinge
 * around. Held separately because a synthetic bone has no object whose
 * matrix could answer for it.
 */
const boneWorld = BONES.map(([, obj, pivot]) => {
  const m = obj.matrixWorld.clone();
  if (pivot) m.multiply(new THREE.Matrix4().makeTranslation(pivot.x, pivot.y, pivot.z));
  return m;
});

/**
 * Fixed rotations applied to a bone's vertices, in that bone's own space.
 *
 * The tail's curve lies in the XY plane, so it arcs out to one side of
 * the body. That reads under a camera locked to a three-quarter view;
 * under this scene's free one it is a tail sticking out of a flank.
 * Rotating it a quarter turn about Y swings the arc into the YZ plane,
 * so it sweeps up and *behind* the animal and reads from anywhere.
 *
 * Baked into the vertices rather than added to the bone's rest rotation
 * on purpose: it has to apply *after* the pose, or it would take the
 * wag axis with it and the tail would swish forwards and backwards.
 */
const REORIENT = {
  tail: new THREE.Matrix4().makeRotationY(Math.PI / 2),
};

/**
 * Average the normals across a mesh, without moving a vertex.
 *
 * The head is a sphere with a muzzle pushed out of its front, and the
 * push starts and stops at a hard boundary — inside a test on y and z,
 * with nothing feathering the edge. The surface is therefore continuous
 * but its *derivative* is not, and `computeVertexNormals` faithfully
 * reports the kink: the faces either side of the boundary disagree, and
 * the shading shows a crisp arc across the cheek.
 *
 * A quantised three-step ramp hid this, because both sides of the kink
 * fell in the same step. Continuous shading does not, and a specular
 * lobe positively advertises it.
 *
 * So the normals are smoothed and the vertices are left alone. The
 * silhouette, the outline shells and the shadow proxy are all unchanged
 * — only the direction each point claims to face is relaxed, which is
 * exactly the quantity that was wrong.
 *
 * Two passes, both over the index buffer:
 *
 *   weld    vertices at the same position get one shared normal. Sphere
 *           geometry duplicates its seam and poles for UVs, and those
 *           duplicates otherwise carry different normals — a seam of its
 *           own, running up the back of the head.
 *   relax   each vertex takes the mean of the vertices it shares an edge
 *           with. This is what actually softens the crease.
 */
function smoothNormals(geo, passes) {
  const pos = geo.attributes.position;
  const nor = geo.attributes.normal;
  const idx = geo.index.array;
  const n = pos.count;

  // Weld by quantised position. The grid is far finer than any feature
  // and far coarser than the float noise two equal vertices pick up.
  const key = new Map();
  const rep = new Int32Array(n);
  const q = (v) => Math.round(v * 1e5);
  for (let i = 0; i < n; i++) {
    const k = `${q(pos.getX(i))},${q(pos.getY(i))},${q(pos.getZ(i))}`;
    if (!key.has(k)) key.set(k, i);
    rep[i] = key.get(k);
  }

  const nx = new Float64Array(n), ny = new Float64Array(n), nz = new Float64Array(n);
  for (let i = 0; i < n; i++) { nx[i] = nor.getX(i); ny[i] = nor.getY(i); nz[i] = nor.getZ(i); }

  const share = (src) => {
    const ax = new Float64Array(n), ay = new Float64Array(n), az = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const r = rep[i];
      ax[r] += src.x[i]; ay[r] += src.y[i]; az[r] += src.z[i];
    }
    for (let i = 0; i < n; i++) {
      const r = rep[i];
      src.x[i] = ax[r]; src.y[i] = ay[r]; src.z[i] = az[r];
    }
  };
  share({ x: nx, y: ny, z: nz });

  for (let p = 0; p < passes; p++) {
    const ax = new Float64Array(n), ay = new Float64Array(n), az = new Float64Array(n);
    // Every triangle contributes each of its corners to the other two.
    for (let t = 0; t < idx.length; t += 3) {
      for (let a = 0; a < 3; a++) {
        for (let b = 0; b < 3; b++) {
          if (a === b) continue;
          const i = rep[idx[t + a]], j = rep[idx[t + b]];
          ax[i] += nx[j]; ay[i] += ny[j]; az[i] += nz[j];
        }
      }
    }
    // Keep half of what the vertex already claimed, so the surface
    // relaxes toward its neighbours instead of dissolving into them.
    for (let i = 0; i < n; i++) {
      const r = rep[i];
      let x = nx[r] + ax[r] * 0.5, y = ny[r] + ay[r] * 0.5, z = nz[r] + az[r] * 0.5;
      const l = Math.hypot(x, y, z) || 1;
      nx[i] = x / l; ny[i] = y / l; nz[i] = z / l;
    }
    share({ x: nx, y: ny, z: nz });
  }

  for (let i = 0; i < n; i++) {
    const l = Math.hypot(nx[i], ny[i], nz[i]) || 1;
    nor.setXYZ(i, nx[i] / l, ny[i] / l, nz[i] / l);
  }
}

/** Nearest ancestor that is a bone, plus the object itself if it is one. */
function nearestBone(obj) {
  for (let o = obj; o; o = o.parent) {
    const i = boneIndex.get(o);
    if (i !== undefined) return i;
  }
  return 0;
}

/* ═══ bone table ══════════════════════════════════════════════════ */

const _m = new THREE.Matrix4();

/**
 * A bone's parent in the *bone* hierarchy is usually not its parent in
 * the scene graph: the ears and eyes hang off `headMesh`, an anonymous
 * group that only exists to shift the head's contents back up after the
 * pivot was moved to the base of the neck. Those in-between nodes are
 * static, so each bone carries the accumulated matrix across them —
 *
 *     boneWorld[i] = boneWorld[parent] · offset[i] · localTRS[i]
 *
 * — and the runtime only ever touches localTRS, exactly as applyPose does.
 */
const _pivot = new THREE.Vector3();

const bones = BONES.map(([name, obj, pivot], i) => {
  const parent = i === 0 ? -1 : nearestBone(obj.parent);

  // Everything strictly between the parent bone and this bone.
  _m.identity();
  for (let o = obj.parent; o && boneIndex.get(o) !== parent; o = o.parent) {
    _m.premultiply(o.matrix);
  }

  /* A pivoted bone sits at `objectMatrix · translate(pivot)`, and that
     is still a plain TRS: the translation is simply rotated into the
     object's own frame first. Keeping it in TRS form means the runtime
     needs no special case for these at all. */
  _pivot.set(0, 0, 0);
  if (pivot) _pivot.set(pivot.x, pivot.y, pivot.z).applyQuaternion(obj.quaternion).multiply(obj.scale);

  return {
    name,
    parent,
    offset: [..._m.elements],
    position: [obj.position.x + _pivot.x, obj.position.y + _pivot.y, obj.position.z + _pivot.z],
    rotation: [obj.rotation.x, obj.rotation.y, obj.rotation.z],
    order: obj.rotation.order,
    scale: [obj.scale.x, obj.scale.y, obj.scale.z],
    // `applyPose` reads these back off the model (ear rest roll, pupil
    // rest offset and the eyeball's tilt) — they are part of the pose
    // contract, so they travel with the bone rather than being re-derived.
    userData: obj.userData?.base !== undefined || obj.userData?.tilt !== undefined
      ? { base: obj.userData.base, tilt: obj.userData.tilt }
      : null,
  };
});

/* ═══ mesh collection ═════════════════════════════════════════════ */

/**
 * Three groups, because each needs different GL state and nothing else
 * about them differs:
 *
 *   0  lit      MeshToonMaterial — the fur, shaded
 *   1  unlit    MeshBasicMaterial front faces — eyes, nose leather, mouth
 *   2  outline  the inverted hulls, drawn with the winding reversed
 *
 * Sorting into groups here means the runtime is three draw calls with no
 * per-vertex branching and no state changes inside a group.
 */
const GROUP_LIT = 0, GROUP_UNLIT = 1, GROUP_OUTLINE = 2;

/** Relaxation passes over the creased shapes' normals. See smoothNormals. */
const SMOOTH_PASSES = 2;

/** Which spring chain, if any, bends a vertex. Rides in the bone byte. */
const SWAY_NONE = 0, SWAY_TAIL = 1, SWAY_WHISKER_R = 2, SWAY_WHISKER_L = 3;

const items = [[], [], []];
let vertexTotal = 0, indexTotal = 0;

/* Rest-pose extent in model space. The scene needs it to stand the cat
   on the floor instead of guessing an offset, and to frame a camera on
   it — both are questions only the geometry can answer. */
const lo = [Infinity, Infinity, Infinity];
const hi = [-Infinity, -Infinity, -Infinity];

const _nm = new THREE.Matrix3();
const _v = new THREE.Vector3();
const _boneInv = new THREE.Matrix4();

parts.root.traverse((obj) => {
  if (!obj.isMesh) return;

  const geo = obj.geometry;
  const pos = geo.attributes.position;
  const nor = geo.attributes.normal;
  const col = geo.attributes.color || null;
  if (!pos || !nor) throw new Error(`mesh without position/normal: ${obj.name || obj.type}`);

  const bone = nearestBone(obj);
  const renderGroup = obj.material.side === THREE.BackSide ? GROUP_OUTLINE
    : obj.material.isMeshToonMaterial ? GROUP_LIT
    : GROUP_UNLIT;

  // Rest transform from the bone's space down to this mesh, with any
  // fixed reorientation on top. Baked into the vertices below, then
  // forgotten.
  _boneInv.copy(boneWorld[bone]).invert();
  const rest = _boneInv.multiply(obj.matrixWorld);
  const fix = REORIENT[BONES[bone][0]];
  if (fix) rest.premultiply(fix);
  _nm.getNormalMatrix(rest);

  /* Outerness, if this geometry is one of the model's bendable ones.
     Everything else gets 0, which the shader reads as "rigid" — and
     reads correctly without a branch, because zero lag is no rotation.

     The group says which chain drives it. The whiskers follow the head
     and the tail follows the body, and the two sides of the face bend
     in opposite directions in their own local frames — mirrored so both
     trail the same way in the world. That sign is the only difference
     between the two whisker groups. */
  /* The two shapes with a hand-cut crease in them: the head, where the
     muzzle is pushed out of the sphere's front, and the body, where the
     underside is sliced flat. Both are painted with vertex colours,
     which is what tells them apart from the plain toon parts hanging off
     the same bones — the nose is on the head bone too. */
  const boneName = BONES[bone][0];
  if (renderGroup === GROUP_LIT && col && (boneName === 'head' || boneName === 'body')) {
    smoothNormals(geo, SMOOTH_PASSES);
  }

  const flex = flexByGeo.get(geo);
  const outer = flex ? flex.o : null;
  const sway = !flex ? SWAY_NONE
    : flex.src === 'head' ? (obj.position.x > 0 ? SWAY_WHISKER_R : SWAY_WHISKER_L)
    : SWAY_TAIL;

  const n = pos.count;
  const P = new Float32Array(n * 3);
  const N = new Float32Array(n * 3);
  const C = new Float32Array(n * 3);
  const O = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    // Root space, for the extent only — the outline hulls are included
    // because they are part of what you see.
    _v.fromBufferAttribute(pos, i).applyMatrix4(obj.matrixWorld);
    for (let k = 0; k < 3; k++) {
      const c = _v.getComponent(k);
      if (c < lo[k]) lo[k] = c;
      if (c > hi[k]) hi[k] = c;
    }

    _v.fromBufferAttribute(pos, i).applyMatrix4(rest);
    P[i * 3] = _v.x; P[i * 3 + 1] = _v.y; P[i * 3 + 2] = _v.z;

    _v.fromBufferAttribute(nor, i).applyMatrix3(_nm).normalize();
    N[i * 3] = _v.x; N[i * 3 + 1] = _v.y; N[i * 3 + 2] = _v.z;

    // Vertex colours where the model painted them, the flat material
    // colour everywhere else. Both are already in linear space — three's
    // colour management converted them on the way in.
    if (col) {
      C[i * 3] = col.getX(i); C[i * 3 + 1] = col.getY(i); C[i * 3 + 2] = col.getZ(i);
    } else {
      C[i * 3] = obj.material.color.r; C[i * 3 + 1] = obj.material.color.g; C[i * 3 + 2] = obj.material.color.b;
    }

    O[i] = outer ? outer[i] : 0;
  }

  // Un-indexed geometry would be a bug in the model, not something to
  // paper over — every primitive three builds here is indexed.
  const idx = geo.index;
  if (!idx) throw new Error(`un-indexed geometry on ${obj.material.type}`);

  items[renderGroup].push({ P, N, C, O, bone, sway, index: idx.array, count: n });
  vertexTotal += n;
  indexTotal += idx.count;
});

return { bones, items, vertexTotal, indexTotal, bounds: { min: lo, max: hi }, whiskers: whiskers.length };
}

/* ═══ collect every colourway ═════════════════════════════════════ */

const takes = SKINS_WANTED.map(collect);
const base = takes[0];

/* The palette cannot move a vertex, so every take must agree on
   geometry. Asserted rather than assumed: a skin that quietly changed
   the head's subdivision would otherwise pack its colours against
   somebody else's vertices, and the cat would come out tie-dyed. */
for (let i = 1; i < takes.length; i++) {
  const t = takes[i];
  if (t.vertexTotal !== base.vertexTotal || t.indexTotal !== base.indexTotal) {
    console.error(`skin "${SKINS_WANTED[i]}" does not share the geometry of "${SKINS_WANTED[0]}"`);
    process.exit(1);
  }
}

const { bones, items, vertexTotal, indexTotal } = base;

/* ═══ pack ════════════════════════════════════════════════════════ */

/**
 * Interleaving would save a little bandwidth and cost a lot of clarity;
 * these are separate arrays and the VAO points at each in turn.
 *
 * Positions stay float32 — they are the one thing a wrong bit shows up
 * in. Normals drop to snorm16, which is far finer than the toon ramp can
 * resolve, and colours to sRGB bytes, which is the encoding they were
 * authored in before three linearised them. The bone index rides in the
 * colour's fourth byte: it is an integer under 32 and the slot was
 * already being padded.
 *
 * The normal's fourth component carries outerness. A three-component
 * snorm16 attribute is padded to four by the driver anyway, so the
 * channel that makes the tail soft is, in bytes, free.
 */
const P = new Float32Array(vertexTotal * 3);
const N = new Int16Array(vertexTotal * 4);
/** One colour block per skin, laid end to end in the order requested. */
const C = takes.map(() => new Uint8Array(vertexTotal * 4));

// The whole cat is well under 65536 vertices, so the element array is
// half the size it would otherwise be — and indices are the largest
// single thing in the file. The 32-bit path stays for the day someone
// raises the head's subdivision to smooth a colour band.
const wide = vertexTotal > 65536;
const I = wide ? new Uint32Array(indexTotal) : new Uint16Array(indexTotal);

const linearToSrgb = (c) => (c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055);
const u8 = (c) => Math.max(0, Math.min(255, Math.round(linearToSrgb(c) * 255)));
const s16 = (c) => Math.max(-32767, Math.min(32767, Math.round(c * 32767)));

const groups = [];
let vOff = 0, iOff = 0;

for (let g = 0; g < items.length; g++) {
  const start = iOff;

  for (let itemIndex = 0; itemIndex < items[g].length; itemIndex++) {
    const it = items[g][itemIndex];
    for (let i = 0; i < it.count; i++) {
      const s = (vOff + i) * 3;
      P[s] = it.P[i * 3]; P[s + 1] = it.P[i * 3 + 1]; P[s + 2] = it.P[i * 3 + 2];

      const n4 = (vOff + i) * 4;
      N[n4] = s16(it.N[i * 3]); N[n4 + 1] = s16(it.N[i * 3 + 1]); N[n4 + 2] = s16(it.N[i * 3 + 2]);
      N[n4 + 3] = s16(it.O[i]);

      /* Every skin's colours, written at the same vertex. The takes walk
         the same graph in the same order, so `items[g][k]` is the same
         mesh in each of them and the indices below line up for all. */
      const c = (vOff + i) * 4;
      for (let t = 0; t < takes.length; t++) {
        const src = takes[t].items[g][itemIndex].C;
        C[t][c] = u8(src[i * 3]); C[t][c + 1] = u8(src[i * 3 + 1]); C[t][c + 2] = u8(src[i * 3 + 2]);
        // Bone in the low five bits, sway group in the top three. Same
        // for every skin, so any take could have supplied it.
        C[t][c + 3] = it.bone | (it.sway << 5);
      }
    }
    // Indices are absolute into the merged buffer, so the whole cat is
    // one element array and a group is a range inside it.
    for (let i = 0; i < it.index.length; i++) I[iOff + i] = it.index[i] + vOff;

    vOff += it.count;
    iOff += it.index.length;
  }

  groups.push({ name: ['lit', 'unlit', 'outline'][g], start, count: iOff - start });
}

/* ═══ write ═══════════════════════════════════════════════════════ */

const header = {
  format: 'cat2',
  /* Every colourway in the file, in the order their blocks appear. The
     first is what loads unless the scene asks otherwise. */
  skins: SKINS_WANTED,
  vertexCount: vertexTotal,
  indexCount: indexTotal,
  indexBits: wide ? 32 : 16,
  bounds: base.bounds,
  // The chains the runtime has to drive, and what each one bends. The
  // angles it uploads are in the bending bone's own space.
  sway: { tail: 'tail', whiskers: base.whiskers },
  bones,
  groups,
};

const headerBytes = new TextEncoder().encode(JSON.stringify(header));
const pad = (n) => (4 - (n % 4)) % 4;
const headerPadded = headerBytes.length + pad(headerBytes.length);

const colourBytes = C.reduce((n, c) => n + c.byteLength, 0);
const total = 8 + headerPadded + P.byteLength + N.byteLength + pad(N.byteLength)
            + colourBytes + I.byteLength;
const out = new Uint8Array(total);
const view = new DataView(out.buffer);

view.setUint32(0, 0x43415432, false);        // 'CAT2'
view.setUint32(4, headerBytes.length, true);
out.set(headerBytes, 8);

let o = 8 + headerPadded;
out.set(new Uint8Array(P.buffer), o); o += P.byteLength;
out.set(new Uint8Array(N.buffer), o); o += N.byteLength + pad(N.byteLength);
for (const c of C) { out.set(c, o); o += c.byteLength; }
out.set(new Uint8Array(I.buffer), o);

await mkdir(dirname(OUT), { recursive: true });
await writeFile(OUT, out);

const kb = (n) => `${(n / 1024).toFixed(1)} kB`;
console.log(`baked ${SKINS_WANTED.length} skins (${SKINS_WANTED.join(', ')}) → ${OUT}`);
console.log(`  ${vertexTotal} vertices · ${indexTotal / 3} triangles · ${bones.length} bones`);
console.log(`  groups: ${groups.map((g) => `${g.name} ${g.count / 3}△`).join(' · ')}`);
console.log(`  ${kb(total)} total (header ${kb(headerBytes.length)}, pos ${kb(P.byteLength)}, nrm ${kb(N.byteLength)}, col ${kb(colourBytes)} for ${C.length}, idx ${kb(I.byteLength)})`);
