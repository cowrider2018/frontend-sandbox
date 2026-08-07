/* ── core/gl.js ──────────────────────────────────────────────────────
   A thin, honest WebGL2 layer. No scene graph, no material system —
   just the four things every GPU sketch actually needs: a context,
   textures, render targets, and a fullscreen triangle.
   ------------------------------------------------------------------ */

/**
 * Create a WebGL2 context and probe the extensions this lab depends on.
 * Float render targets are non-negotiable (every simulation stores state
 * in a texture), so their absence is reported rather than swallowed.
 */
export function createContext(canvas, opts = {}) {
  const gl = canvas.getContext('webgl2', {
    alpha: false,
    antialias: false,          // we resolve our own AA in shader where needed
    depth: false,
    stencil: false,
    powerPreference: 'high-performance',
    // Required for canvas.toBlob(): without it the drawing buffer is
    // already gone by the time the callback runs. (Deliberately paired
    // with desynchronized:false — the two do not co-operate.)
    preserveDrawingBuffer: true,
    ...opts,
  });

  if (!gl) {
    const err = new Error('WEBGL2_UNAVAILABLE');
    err.code = 'WEBGL2_UNAVAILABLE';
    throw err;
  }

  const ext = {
    colorBufferFloat: gl.getExtension('EXT_color_buffer_float'),
    floatLinear:      gl.getExtension('OES_texture_float_linear'),
    floatBlend:       gl.getExtension('EXT_float_blend'),
    timer:            gl.getExtension('EXT_disjoint_timer_query_webgl2'),
  };

  if (!ext.colorBufferFloat) {
    const err = new Error('FLOAT_TARGETS_UNAVAILABLE');
    err.code = 'FLOAT_TARGETS_UNAVAILABLE';
    throw err;
  }

  const limits = {
    maxTextureSize:  gl.getParameter(gl.MAX_TEXTURE_SIZE),
    maxRenderbuffer: gl.getParameter(gl.MAX_RENDERBUFFER_SIZE),
    maxTextureUnits: gl.getParameter(gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS),
    maxDrawBuffers:  gl.getParameter(gl.MAX_DRAW_BUFFERS),
    renderer: describeRenderer(gl),
  };

  return { gl, ext, limits };
}

function describeRenderer(gl) {
  const dbg = gl.getExtension('WEBGL_debug_renderer_info');
  const raw = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  return String(raw || 'unknown')
    .replace(/ANGLE \(|\)$/g, '')
    .replace(/Direct3D11 vs_\d_\d ps_\d_\d/, '')
    .replace(/\s*,\s*$/, '')
    .trim();
}

/* ═══ formats ═════════════════════════════════════════════════════ */

/**
 * Named format triples. Half-float is the default working precision:
 * it is filterable in core WebGL2, half the bandwidth of 32F, and more
 * than enough for velocity/dye/chemistry fields. Positions that must
 * not drift use `rgba32f`.
 */
export const FORMATS = {
  rgba8:   (gl) => ({ internal: gl.RGBA8,   format: gl.RGBA, type: gl.UNSIGNED_BYTE }),
  rgba16f: (gl) => ({ internal: gl.RGBA16F, format: gl.RGBA, type: gl.HALF_FLOAT }),
  rgba32f: (gl) => ({ internal: gl.RGBA32F, format: gl.RGBA, type: gl.FLOAT }),
  rg16f:   (gl) => ({ internal: gl.RG16F,   format: gl.RG,   type: gl.HALF_FLOAT }),
  r16f:    (gl) => ({ internal: gl.R16F,    format: gl.RED,  type: gl.HALF_FLOAT }),
};

/* ═══ texture ═════════════════════════════════════════════════════ */

export function createTexture(gl, {
  width, height,
  format = 'rgba16f',
  filter = gl.LINEAR,
  wrap = gl.CLAMP_TO_EDGE,
  data = null,
} = {}) {
  const f = FORMATS[format](gl);
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, f.internal, width, height, 0, f.format, f.type, data);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrap);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrap);
  gl.bindTexture(gl.TEXTURE_2D, null);

  tex.width = width;
  tex.height = height;
  tex.texelSize = new Float32Array([1 / width, 1 / height]);
  return tex;
}

/* ═══ render target ═══════════════════════════════════════════════ */

/**
 * A colour texture plus the framebuffer that draws into it.
 *
 * `depth: true` adds a depth renderbuffer. Nothing that marches a field
 * needs one — the ray already knows how far it went — but anything that
 * *rasterises* does, or its own triangles sort by draw order. It is a
 * renderbuffer rather than a texture because so far nobody samples it:
 * the scene publishes depth in the colour target's alpha instead.
 */
export class Target {
  constructor(gl, opts) {
    this.gl = gl;
    this.opts = opts;
    this.fbo = gl.createFramebuffer();
    this._alloc(opts.width, opts.height);
  }

  _alloc(width, height) {
    const gl = this.gl;
    this.texture = createTexture(gl, { ...this.opts, width, height });
    this.width = width;
    this.height = height;
    this.texelSize = this.texture.texelSize;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.texture, 0);

    if (this.opts.depth) {
      this.depthBuffer = gl.createRenderbuffer();
      gl.bindRenderbuffer(gl.RENDERBUFFER, this.depthBuffer);
      gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, width, height);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, this.depthBuffer);
      gl.bindRenderbuffer(gl.RENDERBUFFER, null);
    }

    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`Framebuffer incomplete (0x${status.toString(16)}) for ${this.opts.format}`);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /** Bind as the draw target and set the viewport to match. */
  bind() {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.viewport(0, 0, this.width, this.height);
    return this;
  }

  clear(r = 0, g = 0, b = 0, a = 1) {
    const gl = this.gl;
    this.bind();
    gl.clearColor(r, g, b, a);
    gl.clear(gl.COLOR_BUFFER_BIT);
    return this;
  }

  resize(width, height) {
    if (width === this.width && height === this.height) return this;
    this.gl.deleteTexture(this.texture);
    if (this.depthBuffer) this.gl.deleteRenderbuffer(this.depthBuffer);
    this._alloc(width, height);
    return this;
  }

  dispose() {
    this.gl.deleteTexture(this.texture);
    if (this.depthBuffer) this.gl.deleteRenderbuffer(this.depthBuffer);
    this.gl.deleteFramebuffer(this.fbo);
  }
}

/**
 * Ping-pong pair. Simulations read `.read` and draw into `.write`, then
 * `.swap()`. Keeping this in one object removes the single most common
 * source of GPGPU bugs: forgetting which half you are on.
 */
export class DoubleTarget {
  constructor(gl, opts) {
    this.gl = gl;
    this.a = new Target(gl, opts);
    this.b = new Target(gl, opts);
    this.width = this.a.width;
    this.height = this.a.height;
    this.texelSize = this.a.texelSize;
  }

  get read()  { return this.a; }
  get write() { return this.b; }

  swap() { const t = this.a; this.a = this.b; this.b = t; return this; }

  clear(r, g, b, a) { this.a.clear(r, g, b, a); this.b.clear(r, g, b, a); return this; }

  resize(width, height) {
    this.a.resize(width, height);
    this.b.resize(width, height);
    this.width = this.a.width;
    this.height = this.a.height;
    this.texelSize = this.a.texelSize;
    return this;
  }

  dispose() { this.a.dispose(); this.b.dispose(); }
}

/* ═══ geometry ════════════════════════════════════════════════════ */

/**
 * One oversized triangle instead of two quad triangles: no diagonal
 * seam, one fewer vertex, and the GPU never rasterises the same pixel
 * twice at the hypotenuse.
 */
export function createFullscreenTriangle(gl) {
  const vao = gl.createVertexArray();
  const buf = gl.createBuffer();
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);

  return {
    vao,
    draw() {
      gl.bindVertexArray(vao);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    },
    dispose() { gl.deleteVertexArray(vao); gl.deleteBuffer(buf); },
  };
}

/**
 * An attribute-less VAO for vertex-pulling draws: the shader derives
 * everything from gl_VertexID and a data texture, so a million points
 * cost zero bytes of vertex bandwidth.
 */
export function createEmptyVAO(gl) {
  const vao = gl.createVertexArray();
  return {
    vao,
    drawPoints(count) {
      gl.bindVertexArray(vao);
      gl.drawArrays(gl.POINTS, 0, count);
    },
    dispose() { gl.deleteVertexArray(vao); },
  };
}

/* ═══ misc ════════════════════════════════════════════════════════ */

export function bindScreen(gl, width, height) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, width, height);
}

export const BLEND = {
  none(gl)     { gl.disable(gl.BLEND); },
  additive(gl) { gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE); },
  alpha(gl)    { gl.enable(gl.BLEND); gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA); },
  premul(gl)   { gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA); },
};
