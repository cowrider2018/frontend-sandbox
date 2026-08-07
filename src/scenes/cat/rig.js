/* ── scenes/cat/rig.js ───────────────────────────────────────────────
   The cat's skeleton, and the twenty lines of matrix maths that are all
   it needs.

   The model this came from is animated as a rigid hierarchy: every part
   is a whole mesh swinging on a joint, and there is not one skinning
   weight in it. That is the entire reason a scene graph is unnecessary
   here — a bone is a position, a rotation and a scale, and a world
   matrix is the product down the chain. Seventeen of them, once a frame.

   `tools/bake-cat.mjs` folded every static transform between the bones
   into the vertices, so what survives is only what the pose actually
   moves.
   ------------------------------------------------------------------ */

/**
 * Parse the container written by the bake. Layout, after an eight-byte
 * prologue of magic and header length:
 *
 *   header   JSON, padded to four bytes
 *   position float32 × 3          — the one thing a wrong bit shows in
 *   normal   snorm16 × 4          — .xyz normal, .w outerness for sway
 *   colour   uint8 × 4, once per skin — sRGB, bone index in .a
 *   index    uint16 (or uint32)
 *
 * The colourways share everything but their palette, so the geometry is
 * stored once and only the colour block repeats. Switching skin is then
 * one buffer upload rather than another cat.
 */
export function parseCat(buffer) {
  const view = new DataView(buffer);
  if (view.getUint32(0, false) !== 0x43415432) throw new Error('cat.bin: bad magic');

  const headerLength = view.getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, headerLength)));

  const pad = (n) => (4 - (n % 4)) % 4;
  const { vertexCount: nv, indexCount: ni } = header;

  let o = 8 + headerLength + pad(headerLength);
  const position = new Float32Array(buffer, o, nv * 3); o += nv * 12;
  const normal = new Int16Array(buffer, o, nv * 4); o += nv * 8;

  const colors = new Map();
  for (const name of header.skins) {
    colors.set(name, new Uint8Array(buffer, o, nv * 4));
    o += nv * 4;
  }

  const index = header.indexBits === 32
    ? new Uint32Array(buffer, o, ni)
    : new Uint16Array(buffer, o, ni);

  return { header, position, normal, colors, index };
}

/* ═══ the rig ═════════════════════════════════════════════════════ */

export class Rig {
  constructor(header) {
    this.names = header.bones.map((b) => b.name);
    this.index = new Map(this.names.map((n, i) => [n, i]));
    this.parent = Int8Array.from(header.bones.map((b) => b.parent));
    this.bounds = header.bounds;

    const n = header.bones.length;
    this.count = n;

    // Live pose, seeded from the rest pose the bake captured. These are
    // exactly the channels `pose.js` writes, and nothing else exists.
    this.position = new Float32Array(n * 3);
    this.rotation = new Float32Array(n * 3);
    this.scale = new Float32Array(n * 3);
    this.order = header.bones.map((b) => b.order);
    this.userData = header.bones.map((b) => b.userData);

    /** Static transform across the anonymous nodes between two bones. */
    this.offset = new Float32Array(n * 16);
    /** What `reset()` and every idle frame return to. */
    this.rest = { position: new Float32Array(n * 3), rotation: new Float32Array(n * 3), scale: new Float32Array(n * 3) };

    header.bones.forEach((b, i) => {
      for (let k = 0; k < 3; k++) {
        this.rest.position[i * 3 + k] = b.position[k];
        this.rest.rotation[i * 3 + k] = b.rotation[k];
        this.rest.scale[i * 3 + k] = b.scale[k];
      }
      this.offset.set(b.offset, i * 16);
    });
    this.reset();

    /** What the vertex shader reads: one world matrix per bone. */
    this.matrices = new Float32Array(n * 16);
    this._local = new Float32Array(16);
    this._composed = new Float32Array(16);

    /* Did the pose actually change this frame? The scene's temporal
       filter holds several frames of history, so anything that moves has
       to say so or it smears — and a blink is fast enough to smear
       visibly. Comparing the pose channels is cheaper and more direct
       than comparing the matrices they produce. */
    this._prev = new Float32Array(n * 9);
    this.changed = true;
  }

  reset() {
    this.position.set(this.rest.position);
    this.rotation.set(this.rest.rotation);
    this.scale.set(this.rest.scale);
  }

  bone(name) {
    const i = this.index.get(name);
    if (i === undefined) throw new Error(`cat rig: no bone "${name}"`);
    return i;
  }

  /** Set a bone's Euler rotation, matching `Object3D.rotation.set`. */
  setRotation(i, x, y, z) {
    this.rotation[i * 3] = x;
    this.rotation[i * 3 + 1] = y;
    this.rotation[i * 3 + 2] = z;
  }

  /**
   * Resolve every bone to a world matrix.
   *
   * The bones arrive parent-before-child — the bake emits them in that
   * order and the hierarchy is authored, not discovered — so one forward
   * pass is enough and there is no traversal here at all.
   */
  update() {
    const { matrices, offset, parent, _local, _composed, _prev } = this;

    // A tenth of a milliradian is far below anything a pixel can show,
    // and well above the noise a float round-trip introduces.
    const EPS = 1e-4;
    let changed = false;
    for (let i = 0; i < this.count; i++) {
      for (let k = 0; k < 3; k++) {
        const o = i * 9 + k * 3;
        const a = this.position[i * 3 + k], b = this.rotation[i * 3 + k], c = this.scale[i * 3 + k];
        if (Math.abs(a - _prev[o]) > EPS || Math.abs(b - _prev[o + 1]) > EPS || Math.abs(c - _prev[o + 2]) > EPS) {
          changed = true;
        }
        _prev[o] = a; _prev[o + 1] = b; _prev[o + 2] = c;
      }
    }
    this.changed = changed;

    for (let i = 0; i < this.count; i++) {
      compose(_composed, this.position, this.rotation, this.scale, i, this.order[i]);
      multiply(_local, offset, i * 16, _composed, 0);

      const p = parent[i];
      if (p < 0) matrices.set(_local, i * 16);
      else {
        multiplyInto(matrices, i * 16, matrices, p * 16, _local, 0);
      }
    }
    return matrices;
  }
}

/* ═══ matrices ════════════════════════════════════════════════════
   Column-major, the same convention GL and three both use: element
   [c*4+r]. Written out longhand rather than pulled from a library —
   there are three operations here and a dependency would be larger than
   the code.                                                          */

/**
 * Euler angles to a quaternion, then TRS into a matrix.
 *
 * The order matters and is not always XYZ: the ears are authored YXZ so
 * that yaw aims them outward first and the lean then tips them along
 * that new direction. Composing those in the wrong order swings the ear
 * through a visibly different arc.
 */
function compose(out, positions, rotations, scales, i, order) {
  const o = i * 3;
  const x = rotations[o], y = rotations[o + 1], z = rotations[o + 2];

  const c1 = Math.cos(x / 2), c2 = Math.cos(y / 2), c3 = Math.cos(z / 2);
  const s1 = Math.sin(x / 2), s2 = Math.sin(y / 2), s3 = Math.sin(z / 2);

  let qx, qy, qz, qw;
  switch (order) {
    case 'YXZ':
      qx = s1 * c2 * c3 + c1 * s2 * s3;
      qy = c1 * s2 * c3 - s1 * c2 * s3;
      qz = c1 * c2 * s3 - s1 * s2 * c3;
      qw = c1 * c2 * c3 + s1 * s2 * s3;
      break;
    case 'ZYX':
      qx = s1 * c2 * c3 - c1 * s2 * s3;
      qy = c1 * s2 * c3 + s1 * c2 * s3;
      qz = c1 * c2 * s3 - s1 * s2 * c3;
      qw = c1 * c2 * c3 + s1 * s2 * s3;
      break;
    default: // XYZ
      qx = s1 * c2 * c3 + c1 * s2 * s3;
      qy = c1 * s2 * c3 - s1 * c2 * s3;
      qz = c1 * c2 * s3 + s1 * s2 * c3;
      qw = c1 * c2 * c3 - s1 * s2 * s3;
  }

  const x2 = qx + qx, y2 = qy + qy, z2 = qz + qz;
  const xx = qx * x2, xy = qx * y2, xz = qx * z2;
  const yy = qy * y2, yz = qy * z2, zz = qz * z2;
  const wx = qw * x2, wy = qw * y2, wz = qw * z2;

  const sx = scales[o], sy = scales[o + 1], sz = scales[o + 2];

  out[0] = (1 - (yy + zz)) * sx; out[1] = (xy + wz) * sx; out[2] = (xz - wy) * sx; out[3] = 0;
  out[4] = (xy - wz) * sy; out[5] = (1 - (xx + zz)) * sy; out[6] = (yz + wx) * sy; out[7] = 0;
  out[8] = (xz + wy) * sz; out[9] = (yz - wx) * sz; out[10] = (1 - (xx + yy)) * sz; out[11] = 0;
  out[12] = positions[o]; out[13] = positions[o + 1]; out[14] = positions[o + 2]; out[15] = 1;
}

/** out = a·b, reading a and b at the given offsets. */
function multiply(out, a, ao, b, bo) {
  multiplyInto(out, 0, a, ao, b, bo);
}

function multiplyInto(out, oo, a, ao, b, bo) {
  for (let c = 0; c < 4; c++) {
    const b0 = b[bo + c * 4], b1 = b[bo + c * 4 + 1], b2 = b[bo + c * 4 + 2], b3 = b[bo + c * 4 + 3];
    for (let r = 0; r < 4; r++) {
      out[oo + c * 4 + r] =
        a[ao + r] * b0 + a[ao + 4 + r] * b1 + a[ao + 8 + r] * b2 + a[ao + 12 + r] * b3;
    }
  }
}

/**
 * The cat's placement in the world: where it stands, which way it faces,
 * how big it is. Kept out of the rig because it is the scene's business,
 * not the skeleton's.
 */
export function modelMatrix(out, x, y, z, yaw, scale) {
  const c = Math.cos(yaw) * scale, s = Math.sin(yaw) * scale;
  out[0] = c; out[1] = 0; out[2] = -s; out[3] = 0;
  out[4] = 0; out[5] = scale; out[6] = 0; out[7] = 0;
  out[8] = s; out[9] = 0; out[10] = c; out[11] = 0;
  out[12] = x; out[13] = y; out[14] = z; out[15] = 1;
  return out;
}
