/* ── tools/bake-cat.mjs ──────────────────────────────────────────────
   Bake relpet's procedural cat into a flat binary this project can draw
   without three.js.

     node tools/bake-cat.mjs [--skin orangin] [--out src/scenes/cat/cat.bin]

   WHY THIS EXISTS
   ---------------
   `cat-model.js` builds the cat out of ~40 deformed spheres, tubes and
   boxes, and it takes THREE as an argument rather than importing it —
   which is the whole reason this is possible. We run it once, here, in
   Node, and throw the library away: what ships is vertices.

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

/* The model and the library both live in the sibling relpet checkout.
   This is a build step run by hand, not part of the app, so pointing at
   a sibling directory is honest rather than fragile — and both paths are
   overridable when it moves. */
const RELPET = resolve(ROOT, value('relpet', '../relpet'));
const THREE_PATH = resolve(RELPET, 'node_modules/three/build/three.module.js');
const MODEL_PATH = resolve(RELPET, 'public/cat-model.js');

const OUT = resolve(ROOT, value('out', 'src/scenes/cat/cat.bin'));
const SKIN = value('skin', 'orangin');

const THREE = await import(pathToFileURL(THREE_PATH).href);
const { buildCat, SKINS } = await import(pathToFileURL(MODEL_PATH).href);

if (!SKINS[SKIN]) {
  console.error(`unknown skin "${SKIN}" — have: ${Object.keys(SKINS).join(', ')}`);
  process.exit(1);
}

/* ═══ build ═══════════════════════════════════════════════════════ */

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
].filter(([, obj]) => obj);

const boneIndex = new Map(BONES.map(([, obj], i) => [obj, i]));

// Rest pose is whatever buildCat left behind: matrixWorld for every node.
parts.root.updateMatrixWorld(true);

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
const bones = BONES.map(([name, obj], i) => {
  const parent = i === 0 ? -1 : nearestBone(obj.parent);

  // Everything strictly between the parent bone and this bone.
  _m.identity();
  for (let o = obj.parent; o && boneIndex.get(o) !== parent; o = o.parent) {
    _m.premultiply(o.matrix);
  }

  return {
    name,
    parent,
    offset: [..._m.elements],
    position: [obj.position.x, obj.position.y, obj.position.z],
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
  const group = obj.material.side === THREE.BackSide ? GROUP_OUTLINE
    : obj.material.isMeshToonMaterial ? GROUP_LIT
    : GROUP_UNLIT;

  // Rest transform from the bone's space down to this mesh. Baked into
  // the vertices below, then forgotten.
  _boneInv.copy(BONES[bone][1].matrixWorld).invert();
  const rest = _boneInv.multiply(obj.matrixWorld);
  _nm.getNormalMatrix(rest);

  const n = pos.count;
  const P = new Float32Array(n * 3);
  const N = new Float32Array(n * 3);
  const C = new Float32Array(n * 3);

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
  }

  // Un-indexed geometry would be a bug in the model, not something to
  // paper over — every primitive three builds here is indexed.
  const idx = geo.index;
  if (!idx) throw new Error(`un-indexed geometry on ${obj.material.type}`);

  items[group].push({ P, N, C, bone, index: idx.array, count: n });
  vertexTotal += n;
  indexTotal += idx.count;
});

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
 */
const P = new Float32Array(vertexTotal * 3);
const N = new Int16Array(vertexTotal * 3);
const C = new Uint8Array(vertexTotal * 4);

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

  for (const it of items[g]) {
    for (let i = 0; i < it.count; i++) {
      const s = (vOff + i) * 3;
      P[s] = it.P[i * 3]; P[s + 1] = it.P[i * 3 + 1]; P[s + 2] = it.P[i * 3 + 2];
      N[s] = s16(it.N[i * 3]); N[s + 1] = s16(it.N[i * 3 + 1]); N[s + 2] = s16(it.N[i * 3 + 2]);

      const c = (vOff + i) * 4;
      C[c] = u8(it.C[i * 3]); C[c + 1] = u8(it.C[i * 3 + 1]); C[c + 2] = u8(it.C[i * 3 + 2]);
      C[c + 3] = it.bone;
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
  format: 'cat1',
  skin: SKIN,
  vertexCount: vertexTotal,
  indexCount: indexTotal,
  indexBits: wide ? 32 : 16,
  bounds: { min: lo, max: hi },
  bones,
  groups,
};

const headerBytes = new TextEncoder().encode(JSON.stringify(header));
const pad = (n) => (4 - (n % 4)) % 4;
const headerPadded = headerBytes.length + pad(headerBytes.length);

const total = 8 + headerPadded + P.byteLength + N.byteLength + pad(N.byteLength) + C.byteLength + I.byteLength;
const out = new Uint8Array(total);
const view = new DataView(out.buffer);

view.setUint32(0, 0x43415431, false);        // 'CAT1'
view.setUint32(4, headerBytes.length, true);
out.set(headerBytes, 8);

let o = 8 + headerPadded;
out.set(new Uint8Array(P.buffer), o); o += P.byteLength;
out.set(new Uint8Array(N.buffer), o); o += N.byteLength + pad(N.byteLength);
out.set(C, o); o += C.byteLength;
out.set(new Uint8Array(I.buffer), o);

await mkdir(dirname(OUT), { recursive: true });
await writeFile(OUT, out);

const kb = (n) => `${(n / 1024).toFixed(1)} kB`;
console.log(`baked "${SKIN}" → ${OUT}`);
console.log(`  ${vertexTotal} vertices · ${indexTotal / 3} triangles · ${bones.length} bones`);
console.log(`  groups: ${groups.map((g) => `${g.name} ${g.count / 3}△`).join(' · ')}`);
console.log(`  ${kb(total)} total (header ${kb(headerBytes.length)}, pos ${kb(P.byteLength)}, nrm ${kb(N.byteLength)}, col ${kb(C.byteLength)}, idx ${kb(I.byteLength)})`);
