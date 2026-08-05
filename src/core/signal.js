/* ── core/signal.js ──────────────────────────────────────────────────
   ~70 lines of fine-grained reactivity, no framework.

   A signal is a value that remembers who read it. An effect runs, and
   every signal it touches during that run subscribes it. Writing to a
   signal re-runs exactly the effects that depend on it — nothing more.
   That is the whole idea behind every modern signals implementation;
   it fits in one file when you do not need SSR, batching heuristics or
   a component tree.
   ------------------------------------------------------------------ */

/** The effect currently executing, if any. Signals read this to subscribe. */
let ACTIVE = null;
/** Effects queued during a batch, flushed once at the end. */
let QUEUE = null;

export function signal(initial, { equals = Object.is } = {}) {
  const subscribers = new Set();
  let value = initial;

  const read = () => {
    if (ACTIVE) {
      subscribers.add(ACTIVE);
      ACTIVE.deps.add(subscribers);
    }
    return value;
  };

  read.set = (next) => {
    const resolved = typeof next === 'function' ? next(value) : next;
    if (equals(resolved, value)) return value;
    value = resolved;
    notify(subscribers);
    return value;
  };

  /** Read without subscribing — for render loops that poll every frame. */
  read.peek = () => value;

  read.subscribe = (fn) => effect(() => fn(read()));

  return read;
}

function notify(subscribers) {
  const run = [...subscribers];
  if (QUEUE) { for (const e of run) QUEUE.add(e); return; }
  for (const e of run) e.execute();
}

export function effect(fn) {
  const runner = {
    deps: new Set(),
    disposed: false,
    execute() {
      if (runner.disposed) return;
      cleanup(runner);
      const prev = ACTIVE;
      ACTIVE = runner;
      try { fn(); } finally { ACTIVE = prev; }
    },
  };
  runner.execute();
  return () => { runner.disposed = true; cleanup(runner); };
}

function cleanup(runner) {
  for (const set of runner.deps) set.delete(runner);
  runner.deps.clear();
}

/** A signal derived from others; recomputed only when an input changes. */
export function computed(fn) {
  const out = signal(undefined);
  effect(() => out.set(fn()));
  return out;
}

/** Coalesce many writes into one round of effect runs. */
export function batch(fn) {
  if (QUEUE) return fn();
  QUEUE = new Set();
  try { fn(); } finally {
    const queued = QUEUE;
    QUEUE = null;
    for (const e of queued) e.execute();
  }
}

/** Read inside an effect without creating a dependency. */
export function untrack(fn) {
  const prev = ACTIVE;
  ACTIVE = null;
  try { return fn(); } finally { ACTIVE = prev; }
}
