/* ── ui/cmdk.js ──────────────────────────────────────────────────────
   Command palette.

   Built on <dialog>, so the modal semantics — focus trapping, inert
   background, Escape to dismiss, the top layer — are the platform's
   job, not ours. All that is left is a fuzzy matcher.
   ------------------------------------------------------------------ */

export class CommandPalette {
  constructor({ dialog, input, list, empty, provider }) {
    this.dialog = dialog;
    this.input = input;
    this.list = list;
    this.empty = empty;
    this.provider = provider;
    this.index = 0;
    this.matches = [];

    input.addEventListener('input', () => { this.index = 0; this.render(); });

    input.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowDown' || (e.key === 'n' && e.ctrlKey)) { this.move(1); e.preventDefault(); }
      else if (e.key === 'ArrowUp' || (e.key === 'p' && e.ctrlKey)) { this.move(-1); e.preventDefault(); }
      else if (e.key === 'Enter') { e.preventDefault(); this.run(this.matches[this.index]); }
    });

    list.addEventListener('click', (e) => {
      const li = e.target.closest('.cmdk__item');
      if (li) this.run(this.matches[Number(li.dataset.i)]);
    });
    list.addEventListener('pointermove', (e) => {
      const li = e.target.closest('.cmdk__item');
      if (li && Number(li.dataset.i) !== this.index) {
        this.index = Number(li.dataset.i);
        this.paintSelection();
      }
    });

    // Clicking the backdrop (i.e. the dialog element itself) closes.
    dialog.addEventListener('click', (e) => { if (e.target === dialog) dialog.close(); });
    dialog.addEventListener('close', () => { this.input.value = ''; });
  }

  open() {
    this.index = 0;
    this.input.value = '';
    this.render();
    if (!this.dialog.open) this.dialog.showModal();
    this.input.focus();
  }

  toggle() { this.dialog.open ? this.dialog.close() : this.open(); }

  move(delta) {
    if (!this.matches.length) return;
    this.index = (this.index + delta + this.matches.length) % this.matches.length;
    this.paintSelection();
  }

  run(command) {
    if (!command) return;
    this.dialog.close();
    // Let the dialog finish closing before the action steals focus.
    requestAnimationFrame(() => command.run());
  }

  render() {
    const query = this.input.value.trim();
    const commands = this.provider();

    this.matches = query
      ? commands
          .map((c) => ({ c, m: fuzzy(query, c.label + ' ' + (c.keywords ?? '')) }))
          .filter((r) => r.m)
          .sort((a, b) => b.m.score - a.m.score)
          .map((r) => Object.assign({}, r.c, { highlight: r.m.ranges }))
      : commands;

    this.empty.hidden = this.matches.length > 0;
    this.list.innerHTML = this.matches.map((c, i) => `
      <li class="cmdk__item" role="option" aria-selected="false" data-i="${i}">
        <span class="cmdk__glyph">${c.glyph ?? '›'}</span>
        <span class="cmdk__label">${highlight(c.label, c.highlight)}</span>
        <span class="cmdk__meta">${c.meta ?? ''}</span>
      </li>`).join('');
    this.paintSelection();
  }

  paintSelection() {
    const items = this.list.children;
    for (let i = 0; i < items.length; i++) {
      items[i].setAttribute('aria-selected', String(i === this.index));
    }
    items[this.index]?.scrollIntoView({ block: 'nearest' });
  }
}

/**
 * Subsequence match with a bonus for consecutive hits and for matches at
 * word boundaries — the same heuristic every good fuzzy finder uses, and
 * the reason "fl" ranks "Flow field" above "Reaction–diffusion".
 */
function fuzzy(query, text) {
  const q = query.toLowerCase();
  const t = text.toLowerCase();
  const ranges = [];
  let score = 0;
  let ti = 0;
  let streak = 0;

  for (const ch of q) {
    if (ch === ' ') { streak = 0; continue; }
    const found = t.indexOf(ch, ti);
    if (found === -1) return null;
    const boundary = found === 0 || /[\s\-–_/·]/.test(t[found - 1]);
    score += 1 + streak * 3 + (boundary ? 4 : 0) - Math.min(found - ti, 6) * 0.15;
    streak = found === ti ? streak + 1 : 0;
    ranges.push(found);
    ti = found + 1;
  }
  return { score, ranges };
}

function highlight(label, ranges) {
  if (!ranges?.length) return escapeHtml(label);
  // Match indices are into `label + ' ' + keywords`; anything past the
  // label simply never matches a position below.
  const marked = new Set(ranges);
  let out = '';
  for (let i = 0; i < label.length; i++) {
    const ch = escapeHtml(label[i]);
    out += marked.has(i) ? `<mark>${ch}</mark>` : ch;
  }
  return out;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
