/* ── core/program.js ─────────────────────────────────────────────────
   Shader programs with introspected uniforms.

   The GPU already knows the name, type and location of every uniform in
   a linked program — so asking the author to repeat that in JS is pure
   duplication. `Program` reads the reflection data once at link time and
   exposes a single `use({ ... })` call that dispatches each value to the
   right gl.uniform* by type, and binds textures to auto-assigned units.
   ------------------------------------------------------------------ */

const GLSL_VERSION = '#version 300 es\n';

export class ShaderError extends Error {
  constructor(message, { source, log, stage, name }) {
    super(message);
    this.name = 'ShaderError';
    this.source = source;
    this.log = log;
    this.stage = stage;
    this.programName = name;
  }
}

export class Program {
  /**
   * @param {WebGL2RenderingContext} gl
   * @param {string} vert  vertex source, *without* the #version line
   * @param {string} frag  fragment source, *without* the #version line
   * @param {{name?:string, defines?:Record<string,string|number|boolean>}} [opts]
   */
  constructor(gl, vert, frag, opts = {}) {
    this.gl = gl;
    this.name = opts.name || 'program';

    const defines = buildDefines(opts.defines);
    const vs = compile(gl, gl.VERTEX_SHADER, GLSL_VERSION + defines + vert, this.name, 'vertex');
    const fs = compile(gl, gl.FRAGMENT_SHADER, GLSL_VERSION + defines + frag, this.name, 'fragment');

    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    // Slot 0 is the fullscreen triangle's only attribute, by convention.
    gl.bindAttribLocation(prog, 0, 'aPosition');
    gl.linkProgram(prog);

    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(prog);
      gl.deleteProgram(prog);
      throw new ShaderError(`[${this.name}] link failed: ${log}`, { log, stage: 'link', name: this.name });
    }

    // Shader objects are reference-counted by the program; drop ours.
    gl.detachShader(prog, vs); gl.deleteShader(vs);
    gl.detachShader(prog, fs); gl.deleteShader(fs);

    this.program = prog;
    this.uniforms = reflect(gl, prog);
    this.textureUnits = new Map();

    // Sampler → texture-unit assignment is fixed for the program's life,
    // so it can be uploaded once here instead of every frame.
    let unit = 0;
    gl.useProgram(prog);
    for (const [name, info] of this.uniforms) {
      if (isSampler(gl, info.type)) {
        this.textureUnits.set(name, unit);
        gl.uniform1i(info.location, unit);
        unit++;
      }
    }
    gl.useProgram(null);
  }

  /**
   * Bind the program and upload every provided uniform.
   * Unknown keys are ignored — a uniform the compiler optimised away is
   * not an error, and treating it as one makes shaders painful to edit.
   */
  use(values) {
    const gl = this.gl;
    gl.useProgram(this.program);
    if (!values) return this;

    for (const key in values) {
      const info = this.uniforms.get(key);
      if (info === undefined) continue;
      setUniform(gl, info, values[key], this.textureUnits.get(key));
    }
    return this;
  }

  dispose() { this.gl.deleteProgram(this.program); }
}

/* ═══ internals ═══════════════════════════════════════════════════ */

function buildDefines(defines) {
  if (!defines) return '';
  return Object.entries(defines)
    .map(([k, v]) => `#define ${k} ${v === true ? 1 : v === false ? 0 : v}\n`)
    .join('');
}

function compile(gl, type, source, name, stage) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (gl.getShaderParameter(shader, gl.COMPILE_STATUS)) return shader;

  const log = gl.getShaderInfoLog(shader) || '';
  gl.deleteShader(shader);
  const message = `[${name}] ${stage} shader failed to compile\n${log}\n${excerpt(source, log)}`;
  throw new ShaderError(message, { source, log, stage, name });
}

/**
 * GLSL errors come back as `ERROR: 0:47: '...'`. Printing the offending
 * region with line numbers turns a 30-second hunt into a glance.
 */
function excerpt(source, log, radius = 4) {
  const match = /ERROR:\s*\d+:(\d+)/.exec(log);
  const lines = source.split('\n');
  if (!match) return lines.map((l, i) => `${String(i + 1).padStart(4)} │ ${l}`).slice(0, 40).join('\n');

  const line = Number(match[1]);
  const from = Math.max(0, line - radius - 1);
  const to = Math.min(lines.length, line + radius);
  return lines.slice(from, to)
    .map((l, i) => {
      const n = from + i + 1;
      return `${n === line ? '▶' : ' '} ${String(n).padStart(4)} │ ${l}`;
    })
    .join('\n');
}

function reflect(gl, program) {
  const map = new Map();
  const count = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < count; i++) {
    const info = gl.getActiveUniform(program, i);
    if (!info) continue;
    // Array uniforms are reported as `name[0]`; callers write `name`.
    const name = info.name.replace(/\[0\]$/, '');
    const location = gl.getUniformLocation(program, info.name);
    if (location === null) continue;
    map.set(name, { location, type: info.type, size: info.size, name });
  }
  return map;
}

function isSampler(gl, type) {
  return type === gl.SAMPLER_2D || type === gl.SAMPLER_CUBE ||
         type === gl.SAMPLER_3D || type === gl.SAMPLER_2D_ARRAY ||
         type === gl.INT_SAMPLER_2D || type === gl.UNSIGNED_INT_SAMPLER_2D ||
         type === gl.SAMPLER_2D_SHADOW;
}

function setUniform(gl, info, value, unit) {
  const { location, type, size } = info;

  if (unit !== undefined) {
    // Sampler: the location already holds `unit`, so all that remains is
    // to put the caller's texture on that unit.
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, value || null);
    return;
  }

  switch (type) {
    case gl.FLOAT:
      size > 1 ? gl.uniform1fv(location, value) : gl.uniform1f(location, value); break;
    case gl.FLOAT_VEC2: gl.uniform2fv(location, value); break;
    case gl.FLOAT_VEC3: gl.uniform3fv(location, value); break;
    case gl.FLOAT_VEC4: gl.uniform4fv(location, value); break;
    case gl.INT:
    case gl.BOOL:
      size > 1 ? gl.uniform1iv(location, value) : gl.uniform1i(location, value === true ? 1 : value === false ? 0 : value); break;
    case gl.INT_VEC2:
    case gl.BOOL_VEC2: gl.uniform2iv(location, value); break;
    case gl.INT_VEC3:
    case gl.BOOL_VEC3: gl.uniform3iv(location, value); break;
    case gl.INT_VEC4:
    case gl.BOOL_VEC4: gl.uniform4iv(location, value); break;
    case gl.UNSIGNED_INT: gl.uniform1ui(location, value); break;
    case gl.FLOAT_MAT2: gl.uniformMatrix2fv(location, false, value); break;
    case gl.FLOAT_MAT3: gl.uniformMatrix3fv(location, false, value); break;
    case gl.FLOAT_MAT4: gl.uniformMatrix4fv(location, false, value); break;
    default:
      console.warn(`Program: unhandled uniform type 0x${type.toString(16)} for "${info.name}"`);
  }
}
