/* ============================================================
   widgets/la.js — interactive figures for Linear Algebra
   ============================================================ */
import {
  Plot, Dragger, C, el, slider, toggle, segmented, button,
  matrixInput, matrixView, readout, status, defineWidget,
  canvasHost, trackPlot, clamp, fmt, round, mat,
} from '../viz.js';
import * as LA from '../linalg.js';

/* ---------- shared scaffolding ---------- */
function split(node, { aspect = 1.5, hint, wide = false, even = false } = {}) {
  const left = el('div');
  const right = el('div', { class: 'pg-controls' });
  const cls = 'pg-split' + (wide ? ' pg-split--wide-ctrl' : '') + (even ? ' pg-split--even' : '');
  node.appendChild(el('div', { class: cls }, left, right));
  const { canvas } = canvasHost(left, { hint });
  return { left, right, canvas, aspect };
}

const note = html => el('div', { class: 'pg-note', html });

/* ============================================================
   1. Vector addition — drag two vectors, see the parallelogram
   ============================================================ */
defineWidget('vector-add', node => {
  const { right, canvas } = split(node, { hint: 'Drag the arrow tips' });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -6, xmax: 6, ymin: -4.2, ymax: 4.2, aspect: 1.45, pad: 8,
  }));

  const S = { a: [3, 1], b: [1, 2.4], mode: 'add' };
  const snap = v => (opts.snap ? Math.round(v * 2) / 2 : v);
  const opts = { snap: true, para: true };

  const modeCtl = segmented(
    [{ label: 'a + b', value: 'add' }, { label: 'a − b', value: 'sub' }],
    { value: 'add', label: 'Operation', onChange: v => { S.mode = v; refresh(); } }
  );
  const snapCtl = toggle('Snap to half-units', {
    value: true, onChange: v => { opts.snap = v; refresh(); },
  });
  const paraCtl = toggle('Show parallelogram', {
    value: true, onChange: v => { opts.para = v; plot.render(); },
  });
  const out = readout([['a', 0], ['b', 0], ['result', 0], ['‖a‖₂', 0], ['‖b‖₂', 0], ['‖result‖₂', 0]]);
  right.append(modeCtl.root, snapCtl.root, paraCtl.root, out.root);

  const result = () => S.mode === 'add'
    ? [S.a[0] + S.b[0], S.a[1] + S.b[1]]
    : [S.a[0] - S.b[0], S.a[1] - S.b[1]];

  const drag = new Dragger(plot);
  drag.add(() => S.a, p => { S.a = [clamp(snap(p[0]), -5.5, 5.5), clamp(snap(p[1]), -4, 4)]; });
  drag.add(() => S.b, p => { S.b = [clamp(snap(p[0]), -5.5, 5.5), clamp(snap(p[1]), -4, 4)]; });
  drag.onchange = refresh;

  function refresh() { plot.render(); sync(); }
  function sync() {
    const r = result();
    const v = p => `(${fmt(p[0], 1)}, ${fmt(p[1], 1)})`;
    out.set([v(S.a), v(S.b), v(r),
      fmt(Math.hypot(...S.a), 2), fmt(Math.hypot(...S.b), 2), fmt(Math.hypot(...r), 2)]);
  }

  plot.onDraw(p => {
    p.grid(1); p.axes({ ticks: 1 });
    const r = result();
    const bDraw = S.mode === 'add' ? S.b : [-S.b[0], -S.b[1]];

    if (opts.para) {
      p.polygon([[0, 0], S.a, r, bDraw], { fill: C.fill, stroke: false });
    }
    // tip-to-tail copies
    p.arrow(S.a, r, { color: C.c3, lw: 2, dash: [5, 4], alpha: .85 });
    p.arrow(bDraw, r, { color: C.c1, lw: 2, dash: [5, 4], alpha: .5 });

    p.arrow([0, 0], S.a, { color: C.c1, lw: 3 });
    p.arrow([0, 0], bDraw, { color: C.c3, lw: 3 });
    p.arrow([0, 0], r, { color: C.c2, lw: 3.4 });

    p.badge(S.a, 'a', { color: C.c1, align: 'center', dy: -16 });
    p.badge(bDraw, S.mode === 'add' ? 'b' : '−b', { color: C.c3, align: 'center', dy: -16 });
    p.badge(r, S.mode === 'add' ? 'a + b' : 'a − b', { color: C.c2, align: 'center', dy: -16 });

    p.handle(S.a, { color: C.c1 });
    p.handle(S.b, { color: C.c3 });
  });
  sync();

  node.appendChild(note(
    `Addition is <strong>tip-to-tail</strong>: slide <em>b</em> so its tail sits on the tip of <em>a</em>, ` +
    `and the sum reaches from the origin to where you land. The dashed copies show that you get the ` +
    `same point whichever vector you walk first — that is commutativity, drawn.`
  ));
});

/* ============================================================
   2. Matrix multiplication — click an entry of C, see where it comes from
   ============================================================ */
defineWidget('matmul', node => {
  const wrap = el('div');
  node.appendChild(wrap);

  const A0 = [[1, 0, -1], [2, 3, 1]];
  const B0 = [[1, 2], [0, -1], [4, 0]];
  let sel = [0, 0];

  const Ain = matrixInput(2, 3, A0, { label: 'A &nbsp;(2×3)', step: 1, onInput: update });
  const Bin = matrixInput(3, 2, B0, { label: 'B &nbsp;(3×2)', step: 1, onInput: update });
  const Cview = matrixView([[0, 0], [0, 0]], { label: 'C = AB &nbsp;(2×2)', fmt: v => fmt(v, 2) });

  Cview.spans.flat().forEach((s, k) => {
    s.style.cursor = 'pointer';
    s.tabIndex = 0;
    s.setAttribute('role', 'button');
    const i = Math.floor(k / 2), j = k % 2;
    s.setAttribute('aria-label', `Show how entry c${i + 1}${j + 1} is computed`);
    const pick = () => { sel = [i, j]; update(); };
    s.addEventListener('click', pick);
    s.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
  });

  const row = el('div', {
    style: 'display:flex;flex-wrap:wrap;align-items:center;gap:1rem 1.1rem;justify-content:center',
  },
    Ain.root,
    el('span', { style: 'font-size:1.3rem;color:var(--ink-faint);align-self:center;margin-top:1rem', text: '·' }),
    Bin.root,
    el('span', { style: 'font-size:1.3rem;color:var(--ink-faint);align-self:center;margin-top:1rem', text: '=' }),
    Cview.root
  );
  const work = el('div', { class: 'readout', style: 'margin-top:1rem' });
  const st = status('', 'info');
  wrap.append(row, st.root, work);

  function update() {
    const A = Ain.get(), B = Bin.get();
    const Cm = LA.matmul(A, B);
    Cview.set(Cm);

    const [i, j] = sel;
    Ain.highlight(A[i].map((_, k) => [i, k, 'is-hl']));
    Bin.highlight(B.map((_, k) => [k, j, 'is-hl2']));
    Cview.highlight([[i, j, 'is-hl']]);

    const terms = A[i].map((a, k) => `${fmt(a, 2)}·${fmt(B[k][j], 2)}`);
    const prods = A[i].map((a, k) => a * B[k][j]);
    work.innerHTML =
      `<div style="color:var(--ink-faint);margin-bottom:.3em">Entry c<sub>${i + 1}${j + 1}</sub> ` +
      `= row ${i + 1} of A · column ${j + 1} of B</div>` +
      `<div>${terms.join(' &nbsp;+&nbsp; ')}</div>` +
      `<div>= ${prods.map(v => fmt(v, 2)).join(' + ')} = <em>${fmt(Cm[i][j], 2)}</em></div>`;
    st.set(
      `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16.5v.01" stroke-linecap="round"/></svg>` +
      `Click any entry of <strong style="margin:0 .25em">C</strong> to trace exactly which numbers produced it.`,
      'info');
  }
  update();

  node.appendChild(note(
    `The inner dimensions must agree: A is 2×<strong>3</strong> and B is <strong>3</strong>×2, so the ` +
    `shared 3 is consumed by the sum and the result is 2×2. Every entry of C is one dot product — ` +
    `<em>one row meeting one column</em>. Try editing A so a whole row is zero and watch that row of C vanish.`
  ));
});

/* ============================================================
   3. Linear systems — two lines, three outcomes
   ============================================================ */
defineWidget('linsys', node => {
  const { right, canvas } = split(node, { aspect: 1.35, hint: 'Drag either line' });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -6, xmax: 6, ymin: -5, ymax: 5, aspect: 1.35, pad: 8,
  }));

  const S = { a1: 1, b1: 1, c1: 3, a2: 2, b2: -1, c2: 0 };

  const M = matrixInput(2, 3, [[S.a1, S.b1, S.c1], [S.a2, S.b2, S.c2]], {
    label: 'Augmented matrix &nbsp;[A | b]', step: .5,
    onInput: m => {
      S.a1 = m[0][0]; S.b1 = m[0][1]; S.c1 = m[0][2];
      S.a2 = m[1][0]; S.b2 = m[1][1]; S.c2 = m[1][2];
      refresh();
    },
  });
  // visually separate the b column
  M.cells.forEach(r => { r[2].style.marginLeft = '7px'; r[2].style.borderLeftWidth = '2px'; });

  const presets = el('div', { class: 'pg-actions' },
    button('Unique', () => set([1, 1, 3, 2, -1, 0])),
    button('No solution', () => set([1, 1, 2, 2, 2, 5])),
    button('Infinitely many', () => set([1, 2, 4, 2, 4, 8])),
  );
  const st = status('');
  const rrefView = el('div', { class: 'readout' });
  right.append(M.root, presets, st.root, rrefView);

  function set(v) {
    [S.a1, S.b1, S.c1, S.a2, S.b2, S.c2] = v;
    M.set([[S.a1, S.b1, S.c1], [S.a2, S.b2, S.c2]]);
    refresh();
  }

  function classify() {
    const d = S.a1 * S.b2 - S.b1 * S.a2;
    if (Math.abs(d) > 1e-9) {
      return { kind: 'unique', x: (S.c1 * S.b2 - S.b1 * S.c2) / d, y: (S.a1 * S.c2 - S.c1 * S.a2) / d };
    }
    // parallel or coincident — compare the augmented ranks
    const r1 = LA.rref([[S.a1, S.b1], [S.a2, S.b2]]).rank;
    const r2 = LA.rref([[S.a1, S.b1, S.c1], [S.a2, S.b2, S.c2]]).rank;
    return { kind: r1 === r2 ? 'infinite' : 'none' };
  }

  function lineOf(a, b, c) {
    // returns two far-apart points on ax + by = c, or null when the row is 0=0 / 0=k
    const big = 40;
    if (Math.abs(b) > 1e-9) return [[-big, (c + a * big) / b], [big, (c - a * big) / b]];
    if (Math.abs(a) > 1e-9) return [[c / a, -big], [c / a, big]];
    return null;
  }

  function refresh() { plot.render(); sync(); }

  function sync() {
    const r = classify();
    const icons = {
      unique: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>`,
      none: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M18 6 6 18M6 6l12 12"/></svg>`,
      infinite: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 12c0-2 1.5-3.5 3-3.5s2.5 1 3 3.5 1.5 3.5 3 3.5S18 14 18 12s-1.5-3.5-3-3.5-2.5 1-3 3.5-1.5 3.5-3 3.5S6 14 6 12Z"/></svg>`,
    };
    const msg = {
      unique: `<strong>Unique solution</strong> — the lines cross at one point: (${fmt(r.x, 2)}, ${fmt(r.y, 2)})`,
      none: `<strong>No solution</strong> — the lines are parallel and never meet. The system is inconsistent.`,
      infinite: `<strong>Infinitely many solutions</strong> — the two equations describe the same line.`,
    };
    const kindCls = { unique: 'ok', none: 'warn', infinite: 'info' };
    st.set(icons[r.kind] + msg[r.kind], kindCls[r.kind]);

    const { R } = LA.rref([[S.a1, S.b1, S.c1], [S.a2, S.b2, S.c2]], {});
    const fmtRow = row => row.map(v => fmt(v, 2).padStart(6)).join(' ');
    rrefView.innerHTML =
      `<div style="color:var(--ink-faint);margin-bottom:.25em">Reduced row echelon form</div>` +
      `<div>[ ${fmtRow(R[0].slice(0, 2))} | ${fmt(R[0][2], 2)} ]</div>` +
      `<div>[ ${fmtRow(R[1].slice(0, 2))} | ${fmt(R[1][2], 2)} ]</div>`;
  }

  // drag a line by moving its constant term (translation) — intuitive and stable
  const drag = new Dragger(plot);
  const anchor = (a, b, c) => {
    const n2 = a * a + b * b;
    return n2 < 1e-9 ? [0, 0] : [a * c / n2, b * c / n2];
  };
  drag.add(() => anchor(S.a1, S.b1, S.c1), p => {
    S.c1 = round(S.a1 * p[0] + S.b1 * p[1], 2); M.set([[S.a1, S.b1, S.c1], [S.a2, S.b2, S.c2]]);
  }, { r: 16 });
  drag.add(() => anchor(S.a2, S.b2, S.c2), p => {
    S.c2 = round(S.a2 * p[0] + S.b2 * p[1], 2); M.set([[S.a1, S.b1, S.c1], [S.a2, S.b2, S.c2]]);
  }, { r: 16 });
  drag.onchange = refresh;

  plot.onDraw(p => {
    p.grid(1); p.axes({ ticks: 1 });
    const l1 = lineOf(S.a1, S.b1, S.c1);
    const l2 = lineOf(S.a2, S.b2, S.c2);
    if (l1) p.line(l1[0], l1[1], { color: C.c1, lw: 2.8 });
    if (l2) p.line(l2[0], l2[1], { color: C.c3, lw: 2.8 });

    const r = classify();
    if (r.kind === 'unique' && Math.abs(r.x) < 20 && Math.abs(r.y) < 20) {
      p.dot([r.x, r.y], { r: 7, color: C.c2, ring: true });
      p.badge([r.x, r.y], `(${fmt(r.x, 2)}, ${fmt(r.y, 2)})`, { color: C.c2, align: 'center', dy: -20 });
    }
    if (l1) p.handle(anchor(S.a1, S.b1, S.c1), { color: C.c1, r: 6 });
    if (l2) p.handle(anchor(S.a2, S.b2, S.c2), { color: C.c3, r: 6 });
  });
  sync();

  node.appendChild(note(
    `Each equation is a <strong>line</strong>; a solution is a point on both. That is why there are exactly ` +
    `three possibilities — two lines can cross once, never, or lie on top of each other. Drag one line onto ` +
    `the other and watch the RREF collapse to a row of zeros.`
  ));
});

/* ============================================================
   4. Linear transformation sandbox — the flagship
      det, rank, eigenvectors, invertibility in one picture
   ============================================================ */
defineWidget('transform2d', node => {
  const { right, canvas } = split(node, { aspect: 1.15, wide: true, hint: 'Drag the coloured basis arrows' });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -4.2, xmax: 4.2, ymin: -4.2, ymax: 4.2, aspect: 1.15, pad: 8,
  }));

  // A = [[a,b],[c,d]]; columns are the images of e1 and e2
  let A = [1.4, -0.6, 0.5, 1.1];
  const show = { eig: true, circle: false, square: true, grid: true };

  const Min = matrixInput(2, 2, [[A[0], A[1]], [A[2], A[3]]], {
    label: 'A', step: .1,
    onInput: m => { A = [m[0][0], m[0][1], m[1][0], m[1][1]]; refresh(false); },
  });

  const presets = el('div', { class: 'pg-actions' },
    button('Identity', () => set([1, 0, 0, 1])),
    button('Rotate 30°', () => { const t = Math.PI / 6; set([Math.cos(t), -Math.sin(t), Math.sin(t), Math.cos(t)]); }),
    button('Scale', () => set([1.8, 0, 0, .6])),
    button('Shear', () => set([1, 1, 0, 1])),
    button('Reflect', () => set([1, 0, 0, -1])),
    button('Singular', () => set([1, 2, .5, 1])),
  );

  const out = readout([
    ['det(A)', 0], ['rank(A)', 0], ['trace(A)', 0],
    ['λ₁, λ₂', 0], ['invertible', 0],
  ]);
  const st = status('');
  const toggles = el('div', { style: 'display:flex;flex-direction:column;gap:.55rem' },
    toggle('Transformed grid', { value: true, onChange: v => { show.grid = v; plot.render(); } }).root,
    toggle('Unit square → image', { value: true, onChange: v => { show.square = v; plot.render(); } }).root,
    toggle('Eigenvector directions', { value: true, onChange: v => { show.eig = v; plot.render(); } }).root,
    toggle('Unit circle → ellipse', { value: false, onChange: v => { show.circle = v; plot.render(); } }).root,
  );
  right.append(Min.root, presets, toggles, out.root, st.root);

  function set(m) { A = m.slice(); Min.set([[A[0], A[1]], [A[2], A[3]]]); refresh(false); }
  function refresh(fromDrag = true) {
    if (fromDrag) Min.set([[A[0], A[1]], [A[2], A[3]]]);
    plot.render(); sync();
  }

  const drag = new Dragger(plot);
  drag.add(() => [A[0], A[2]], p => {          // image of e1 = first column
    A[0] = clamp(round(p[0], 2), -3.5, 3.5); A[2] = clamp(round(p[1], 2), -3.5, 3.5);
  });
  drag.add(() => [A[1], A[3]], p => {          // image of e2 = second column
    A[1] = clamp(round(p[0], 2), -3.5, 3.5); A[3] = clamp(round(p[1], 2), -3.5, 3.5);
  });
  drag.onchange = () => refresh(true);

  function sync() {
    const d = mat.det(A);
    const tr = A[0] + A[3];
    const rk = Math.abs(d) > 1e-6 ? 2 : (A.some(v => Math.abs(v) > 1e-9) ? 1 : 0);
    const e = mat.eig(A);
    const eigTxt = e
      ? `${fmt(e.l1, 2)}, ${fmt(e.l2, 2)}`
      : 'complex pair';
    out.set([
      { html: fmt(d, 3), cls: Math.abs(d) < 1e-6 ? 'is-warn' : '' },
      { html: String(rk), cls: rk < 2 ? 'is-warn' : 'is-ok' },
      fmt(tr, 3),
      eigTxt,
      { html: Math.abs(d) > 1e-6 ? 'yes' : 'no', cls: Math.abs(d) > 1e-6 ? 'is-ok' : 'is-warn' },
    ]);

    if (Math.abs(d) < 1e-6) {
      st.set(`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 9v4M12 17v.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/></svg>` +
        `<span><strong>Singular.</strong> The plane has been squashed onto a line — area is destroyed, so no inverse exists.</span>`, 'warn');
    } else if (d < 0) {
      st.set(`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 2.1 21 6l-4 3.9"/><path d="M3 6h18"/><path d="M7 21.9 3 18l4-3.9"/><path d="M21 18H3"/></svg>` +
        `<span><strong>Orientation flipped.</strong> A negative determinant means the plane was turned over; |det| = ${fmt(Math.abs(d), 2)} is the area scale factor.</span>`, 'info');
    } else {
      st.set(`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>` +
        `<span><strong>Invertible.</strong> Areas are scaled by ${fmt(d, 2)}× and orientation is preserved.</span>`, 'ok');
    }
  }

  plot.onDraw(p => {
    // faint original grid
    p.grid(1, { color: C.grid });
    p.axes();

    if (show.grid) p.transformedGrid(A, { range: 5, color: C.c1, alpha: .3 });

    if (show.square) {
      const sq = [[0, 0], [1, 0], [1, 1], [0, 1]];
      p.polygon(sq, { color: C.muted, lw: 1.4, dash: [4, 4], alpha: .7 });
      p.polygon(sq.map(v => mat.apply(A, v)), { fill: C.fill2, color: C.c2, lw: 2 });
      const cen = mat.apply(A, [.5, .5]);
      const d = mat.det(A);
      if (Math.abs(d) > .06) {
        p.badge(cen, `area ${fmt(Math.abs(d), 2)}`, { color: C.c2, align: 'center', bg: C.raised });
      }
    }

    if (show.circle) {
      const circ = [], img = [];
      for (let i = 0; i <= 72; i++) {
        const t = i / 72 * Math.PI * 2;
        const v = [Math.cos(t), Math.sin(t)];
        circ.push(v); img.push(mat.apply(A, v));
      }
      p.path(circ, { color: C.muted, lw: 1.2, dash: [3, 3], alpha: .6, close: true });
      p.path(img, { color: C.c4, lw: 2.2, close: true });
    }

    if (show.eig) {
      const e = mat.eig(A);
      if (e) {
        [[e.v1, e.l1], [e.v2, e.l2]].forEach(([v, l], i) => {
          const col = i === 0 ? C.c3 : C.c5;
          p.ray([0, 0], v, { color: col, lw: 1.6, dash: [7, 5], alpha: .75 });
          const tip = [v[0] * 1.9, v[1] * 1.9];
          p.badge(tip, `λ = ${fmt(l, 2)}`, { color: col, align: 'center', bg: C.raised });
        });
      }
    }

    // basis vectors, original (faint) and images (bold, draggable)
    p.arrow([0, 0], [1, 0], { color: C.muted, lw: 1.6, alpha: .5 });
    p.arrow([0, 0], [0, 1], { color: C.muted, lw: 1.6, alpha: .5 });
    p.arrow([0, 0], [A[0], A[2]], { color: C.c1, lw: 3.2 });
    p.arrow([0, 0], [A[1], A[3]], { color: C.c2, lw: 3.2 });
    p.badge([A[0], A[2]], 'A e₁', { color: C.c1, align: 'center', dy: -17 });
    p.badge([A[1], A[3]], 'A e₂', { color: C.c2, align: 'center', dy: -17 });
    p.handle([A[0], A[2]], { color: C.c1, r: 6 });
    p.handle([A[1], A[3]], { color: C.c2, r: 6 });
  });
  sync();

  node.appendChild(note(
    `<strong>The two arrows you drag are literally the columns of A</strong> — a matrix is nothing more than ` +
    `a record of where the basis vectors land. Everything else follows: the shaded area is |det A|, the ` +
    `dashed rays are directions that only get stretched (eigenvectors), and when you drag the two arrows onto ` +
    `the same line the determinant hits zero and the matrix stops being invertible.`
  ));
});

/* ============================================================
   5. Eigenvector hunt — drag v until Av lines up with it
   ============================================================ */
defineWidget('eigen-explore', node => {
  const { right, canvas } = split(node, { aspect: 1.25, hint: 'Drag the blue arrow around' });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -4, xmax: 4, ymin: -3.4, ymax: 3.4, aspect: 1.25, pad: 8,
  }));

  let A = [4, 1, 2, 3];               // the notes' example: λ = 5, 2
  let theta = 0.9;
  let found = null;

  const Min = matrixInput(2, 2, [[A[0], A[1]], [A[2], A[3]]], {
    label: 'A', step: .5,
    onInput: m => { A = [m[0][0], m[0][1], m[1][0], m[1][1]]; refresh(); },
  });
  const ang = slider('Direction of <strong>v</strong>', {
    min: 0, max: 6.2832, step: .002, value: theta,
    format: v => `${Math.round(v * 180 / Math.PI)}°`,
    onInput: v => { theta = v; refresh(); },
  });
  const snapBtn = el('div', { class: 'pg-actions' },
    button('Snap to λ₁', () => snapTo(0)),
    button('Snap to λ₂', () => snapTo(1)),
  );
  const out = readout([['v', 0], ['Av', 0], ['angle between', 0], ['‖Av‖ / ‖v‖', 0]]);
  const st = status('');
  right.append(Min.root, ang.root, snapBtn, out.root, st.root);

  function snapTo(k) {
    const e = mat.eig(A);
    if (!e) { st.set('This matrix has no real eigenvectors — rotation leaves no direction fixed.', 'warn'); return; }
    const v = k === 0 ? e.v1 : e.v2;
    theta = Math.atan2(v[1], v[0]);
    ang.set(theta < 0 ? theta + Math.PI * 2 : theta);
    refresh();
  }

  const V = () => [Math.cos(theta) * 1.7, Math.sin(theta) * 1.7];

  const drag = new Dragger(plot);
  drag.add(V, p => { theta = Math.atan2(p[1], p[0]); ang.set(theta < 0 ? theta + Math.PI * 2 : theta); });
  drag.onchange = refresh;

  function refresh() { plot.render(); sync(); }

  function sync() {
    const v = V(), av = mat.apply(A, v);
    const nv = Math.hypot(...v), na = Math.hypot(...av);
    const cosang = na < 1e-9 ? 1 : (v[0] * av[0] + v[1] * av[1]) / (nv * na);
    const angle = Math.acos(clamp(cosang, -1, 1)) * 180 / Math.PI;
    const aligned = angle < 2.2 || angle > 177.8;
    found = aligned;

    out.set([
      `(${fmt(v[0], 2)}, ${fmt(v[1], 2)})`,
      `(${fmt(av[0], 2)}, ${fmt(av[1], 2)})`,
      { html: `${fmt(angle, 1)}°`, cls: aligned ? 'is-ok' : '' },
      fmt(na / nv, 3),
    ]);

    if (aligned) {
      const lam = (v[0] * av[0] + v[1] * av[1]) / (nv * nv);
      st.set(`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>` +
        `<span><strong>Eigenvector found.</strong> Av points along v, so Av = λv with λ ≈ ${fmt(lam, 3)}.</span>`, 'ok');
    } else {
      st.set(`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16.5v.01" stroke-linecap="round"/></svg>` +
        `<span>Av has been <strong>rotated</strong> away from v by ${fmt(angle, 1)}°. Keep turning until the two arrows line up.</span>`, '');
    }
  }

  plot.onDraw(p => {
    p.grid(1); p.axes({ ticks: 1 });

    // faint trace of where the whole unit circle goes
    const img = [];
    for (let i = 0; i <= 96; i++) {
      const t = i / 96 * Math.PI * 2;
      img.push(mat.apply(A, [Math.cos(t) * 1.7, Math.sin(t) * 1.7]));
    }
    p.path(img, { color: C.muted, lw: 1.1, dash: [3, 4], alpha: .45, close: true });

    const e = mat.eig(A);
    if (e) {
      p.ray([0, 0], e.v1, { color: C.c3, lw: 1.3, dash: [6, 5], alpha: .5 });
      p.ray([0, 0], e.v2, { color: C.c5, lw: 1.3, dash: [6, 5], alpha: .5 });
    }

    const v = V(), av = mat.apply(A, v);
    if (found) p.ray([0, 0], v, { color: C.c3, lw: 2, alpha: .35 });
    p.arrow([0, 0], av, { color: C.c4, lw: 3.2 });
    p.arrow([0, 0], v, { color: C.c1, lw: 3, dash: found ? null : [6, 4] });
    p.badge(v, 'v', { color: C.c1, align: 'center', dy: -16 });
    p.badge(av, 'Av', { color: C.c4, align: 'center', dy: -16 });
    p.handle(v, { color: C.c1, r: 6, glow: found });
  });
  sync();

  node.appendChild(note(
    `Almost every direction gets <em>rotated</em> by A. The rare directions that come back pointing the same ` +
    `way (or exactly backwards) are the <strong>eigenvectors</strong>, and the factor they are stretched by is ` +
    `the <strong>eigenvalue</strong>. Try the rotation matrix <span class="u-mono">[0 −1; 1 0]</span> — every ` +
    `direction turns, so there is no real eigenvector at all.`
  ));
});

/* ============================================================
   6. Rank in 3D — watch a cube collapse
   ============================================================ */
defineWidget('rank3d', node => {
  const { right, canvas } = split(node, { aspect: 1.4, hint: 'Drag to orbit' });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -2.6, xmax: 2.6, ymin: -2.1, ymax: 2.1, aspect: 1.4, pad: 6,
  }));

  let yaw = 0.72, pitch = 0.42;
  let M = [[1, .5, 0], [0, 1, .5], [.3, 0, 1]];

  const Min = matrixInput(3, 3, M, {
    label: 'A', step: .1, onInput: m => { M = m; refresh(); },
  });
  const presets = el('div', { class: 'pg-actions' },
    button('Full rank (3)', () => set([[1, .5, 0], [0, 1, .5], [.3, 0, 1]])),
    button('Rank 2', () => set([[1, 0, 1], [0, 1, 1], [1, 1, 2]])),
    button('Rank 1', () => set([[1, 1, 1], [1, 1, 1], [1, 1, 1]])),
  );
  const out = readout([['rank(A)', 0], ['det(A)', 0], ['image is a', 0]]);
  const st = status('');
  right.append(Min.root, presets, out.root, st.root);

  function set(m) { M = m.map(r => r.slice()); Min.set(M); refresh(); }
  function refresh() { plot.render(); sync(); }

  // orbit with a plain pointer drag
  let dragging = false, last = null;
  canvas.addEventListener('pointerdown', e => {
    dragging = true; last = [e.clientX, e.clientY];
    canvas.setPointerCapture?.(e.pointerId); canvas.classList.add('is-grabbing');
  });
  canvas.addEventListener('pointermove', e => {
    if (!dragging) return;
    yaw += (e.clientX - last[0]) * .01;
    pitch = clamp(pitch + (e.clientY - last[1]) * .01, -1.4, 1.4);
    last = [e.clientX, e.clientY];
    plot.render();
  });
  const stop = e => {
    dragging = false; canvas.classList.remove('is-grabbing');
    canvas.releasePointerCapture?.(e.pointerId);
  };
  canvas.addEventListener('pointerup', stop);
  canvas.addEventListener('pointercancel', stop);
  canvas.classList.add('is-grabbable');

  const project = ([x, y, z]) => {
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    const cp = Math.cos(pitch), sp = Math.sin(pitch);
    const X = x * cy - z * sy;
    const Z = x * sy + z * cy;
    const Y = y * cp - Z * sp;
    return [X, Y];
  };

  const CUBE = [];
  for (let i = 0; i < 8; i++) {
    CUBE.push([(i & 1) ? 1 : -1, (i & 2) ? 1 : -1, (i & 4) ? 1 : -1].map(v => v * .8));
  }
  const EDGES = [];
  for (let i = 0; i < 8; i++) for (let j = i + 1; j < 8; j++) {
    let diff = 0;
    for (let b = 0; b < 3; b++) if (((i >> b) & 1) !== ((j >> b) & 1)) diff++;
    if (diff === 1) EDGES.push([i, j]);
  }

  function sync() {
    const rk = LA.rank(M);
    const d = LA.det(M);
    const desc = ['a single point', 'a line', 'a plane', 'a solid (all of ℝ³)'][rk];
    out.set([
      { html: String(rk), cls: rk === 3 ? 'is-ok' : 'is-warn' },
      { html: fmt(d, 4), cls: Math.abs(d) < 1e-9 ? 'is-warn' : '' },
      desc,
    ]);
    if (rk === 3) {
      st.set(`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>` +
        `<span><strong>Full rank.</strong> The cube stays a solid — no dimension is lost, and A is invertible.</span>`, 'ok');
    } else {
      st.set(`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 9v4M12 17v.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/></svg>` +
        `<span><strong>Rank ${rk}.</strong> The cube has been flattened onto ${desc} — information is gone for good, and A has no inverse.</span>`, 'warn');
    }
  }

  plot.onDraw(p => {
    const pts = CUBE.map(v => LA.matvec(M, v));
    // scale to fit
    const maxr = Math.max(1e-6, ...pts.map(v => Math.hypot(...v)));
    const k = Math.min(1, 1.75 / maxr);

    // reference (untransformed) cube, faint
    EDGES.forEach(([i, j]) => {
      p.line(project(CUBE[i].map(v => v * .55)), project(CUBE[j].map(v => v * .55)),
        { color: C.muted, lw: 1, alpha: .3, dash: [3, 3] });
    });

    EDGES.forEach(([i, j]) => {
      p.line(project(pts[i].map(v => v * k)), project(pts[j].map(v => v * k)),
        { color: C.c1, lw: 2, alpha: .9 });
    });
    pts.forEach(v => p.dot(project(v.map(c => c * k)), { r: 3.2, color: C.c2 }));

    p.text({ px: 12, py: 11 }, 'original', { color: C.muted, size: 11 });
    p.text({ px: 12, py: 32 }, 'image under A', { color: C.c1, size: 11 });
  });
  sync();

  node.appendChild(note(
    `Rank is <strong>how many dimensions survive</strong>. Set every row equal and the whole cube is crushed ` +
    `onto a single line — that is rank 1. Once dimensions are lost there is no way to undo the map, which is ` +
    `exactly what "singular" means. Yet a rank-1 3×3 matrix needs only 6 numbers to store instead of 9: the ` +
    `same collapse that destroys invertibility is what makes compression possible.`
  ));
});

/* ============================================================
   7. SVD geometry — rotate, stretch, rotate
   ============================================================ */
defineWidget('svd-geometry', node => {
  const { right, canvas } = split(node, { aspect: 1.3 });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -3.4, xmax: 3.4, ymin: -2.6, ymax: 2.6, aspect: 1.3, pad: 8,
  }));

  let A = [1.6, 1.1, 0.4, 1.3];
  let t = 3;                       // 0..3 stage scrubber

  const Min = matrixInput(2, 2, [[A[0], A[1]], [A[2], A[3]]], {
    label: 'A', step: .1,
    onInput: m => { A = [m[0][0], m[0][1], m[1][0], m[1][1]]; refresh(); },
  });
  const stage = slider('Stage', {
    min: 0, max: 3, step: .01, value: 3,
    format: v => ['start', 'Vᵀ rotate', 'Σ stretch', 'U rotate'][Math.min(3, Math.round(v))],
    onInput: v => { t = v; plot.render(); },
  });
  const out = readout([['σ₁', 0], ['σ₂', 0], ['σ₁/σ₂', 0], ['rank', 0]]);
  const st = status(
    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16.5v.01" stroke-linecap="round"/></svg>` +
    `Drag the stage slider to pull the transformation apart into its three steps.`, 'info');
  right.append(Min.root, stage.root, out.root, st.root);

  function refresh() { plot.render(); sync(); }

  function sync() {
    const s = mat.svd(A);
    const rk = (s.s1 > 1e-7 ? 1 : 0) + (s.s2 > 1e-7 ? 1 : 0);
    out.set([
      fmt(s.s1, 3), fmt(s.s2, 3),
      s.s2 > 1e-9 ? fmt(s.s1 / s.s2, 2) : '∞',
      { html: String(rk), cls: rk === 2 ? 'is-ok' : 'is-warn' },
    ]);
  }

  plot.onDraw(p => {
    p.grid(1); p.axes();
    const s = mat.svd(A);
    // stage matrices
    const Vt = [s.v1[0], s.v1[1], s.v2[0], s.v2[1]];         // rows are v1ᵀ, v2ᵀ
    const Sg = [s.s1, 0, 0, s.s2];
    const U  = [s.u1[0], s.u2[0], s.u1[1], s.u2[1]];         // columns are u1, u2

    // interpolate matrix from identity through the three stages
    const lerpM = (M0, M1, k) => M0.map((v, i) => v + (M1[i] - v) * k);
    const I = [1, 0, 0, 1];
    let Mcur;
    if (t <= 1)      Mcur = lerpM(I, Vt, t);
    else if (t <= 2) Mcur = lerpM(Vt, mat.mul(Sg, Vt), t - 1);
    else             Mcur = lerpM(mat.mul(Sg, Vt), A, t - 2);

    // unit circle and its current image, with two marker directions
    const circ = [], img = [];
    for (let i = 0; i <= 100; i++) {
      const th = i / 100 * Math.PI * 2;
      const v = [Math.cos(th), Math.sin(th)];
      circ.push(v); img.push(mat.apply(Mcur, v));
    }
    p.path(circ, { color: C.muted, lw: 1.2, dash: [4, 4], alpha: .55, close: true });
    p.path(img, { color: C.c4, lw: 2.4, close: true, fill: C.fill2 });

    // track where v1 and v2 go
    const a1 = mat.apply(Mcur, s.v1), a2 = mat.apply(Mcur, s.v2);
    p.arrow([0, 0], s.v1, { color: C.c3, lw: 1.6, alpha: .4, dash: [4, 3] });
    p.arrow([0, 0], s.v2, { color: C.c5, lw: 1.6, alpha: .4, dash: [4, 3] });
    p.arrow([0, 0], a1, { color: C.c3, lw: 3 });
    p.arrow([0, 0], a2, { color: C.c5, lw: 3 });

    const stageName = ['input circle', 'after Vᵀ (rotation)', 'after ΣVᵀ (stretch)', 'after UΣVᵀ = A'][Math.min(3, Math.round(t))];
    p.text({ px: 12, py: 18 }, stageName, { color: C.ink, size: 12.5, weight: 700 });
    if (t > 2.9) {
      p.badge(a1, `σ₁ = ${fmt(s.s1, 2)}`, { color: C.c3, align: 'center', dy: -16 });
      p.badge(a2, `σ₂ = ${fmt(s.s2, 2)}`, { color: C.c5, align: 'center', dy: -16 });
    }
  });
  sync();

  node.appendChild(note(
    `Every matrix — square or not, invertible or not — does exactly three things in sequence: ` +
    `<strong>rotate, stretch along axes, rotate again</strong>. That is all SVD says. The circle always ` +
    `becomes an ellipse, and the singular values σ₁ ≥ σ₂ are its semi-axis lengths. Push σ₂ to zero ` +
    `(try a singular matrix) and the ellipse flattens into a segment.`
  ));
});

/* ============================================================
   8. SVD image compression — the rank-k slider
   ============================================================ */
defineWidget('svd-compress', node => {
  const N = 72;
  const imgs = {
    gradient: buildScene(N),
    stripes: buildStripes(N),
    noise: buildNoisy(N),
  };
  let key = 'gradient';
  let A = imgs[key];
  let dec = LA.svd(A);
  let k = 8;

  const wrap = el('div');
  node.appendChild(wrap);

  const cvOrig = el('canvas', { width: N, height: N, style: 'width:100%;image-rendering:pixelated;border-radius:8px' });
  const cvRec  = el('canvas', { width: N, height: N, style: 'width:100%;image-rendering:pixelated;border-radius:8px' });
  const cvSpec = el('canvas');

  const panel = (title, cv, sub) => el('div', {},
    el('div', { class: 'matrix-label', html: title }),
    el('div', { class: 'pg-canvas-wrap' }, cv),
    el('div', { style: 'font-size:.78rem;color:var(--ink-faint);margin-top:.35rem;text-align:center', html: sub || '' })
  );
  const subOrig = el('span'), subRec = el('span');

  const grid = el('div', { style: 'display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1rem;align-items:start' },
    panel('Original', cvOrig, ''),
    panel('Rank-k reconstruction', cvRec, ''),
    el('div', {},
      el('div', { class: 'matrix-label', html: 'Singular value spectrum' }),
      el('div', { class: 'pg-canvas-wrap' }, cvSpec)
    )
  );
  grid.children[0].lastChild.appendChild(subOrig);
  grid.children[1].lastChild.appendChild(subRec);

  const specPlot = trackPlot(new Plot(cvSpec, {
    xmin: 0, xmax: N, ymin: 0, ymax: 1.06, aspect: 1.05, equal: false, pad: 0,
  }));

  const kCtl = slider('Rank k (components kept)', {
    min: 1, max: N, step: 1, value: k,
    format: v => `${v} / ${N}`,
    onInput: v => { k = v; redraw(); },
  });
  const pick = segmented(
    [{ label: 'Smooth', value: 'gradient' }, { label: 'Structured', value: 'stripes' }, { label: 'Noisy', value: 'noise' }],
    { value: 'gradient', label: 'Image', onChange: v => { key = v; A = imgs[v]; dec = LA.svd(A); redraw(); } }
  );
  const out = readout([['stored numbers', 0], ['vs. full image', 0], ['compression', 0], ['energy kept', 0], ['error (Frobenius)', 0]]);
  const st = status('');

  const controls = el('div', { class: 'pg-controls', style: 'margin-top:1.1rem' },
    pick.root, kCtl.root, out.root, st.root);
  wrap.append(grid, controls);

  function paint(cv, M) {
    const ctx = cv.getContext('2d');
    const im = ctx.createImageData(N, N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const v = clamp(Math.round(M[i][j]), 0, 255);
      const o = (i * N + j) * 4;
      im.data[o] = im.data[o + 1] = im.data[o + 2] = v;
      im.data[o + 3] = 255;
    }
    ctx.putImageData(im, 0, 0);
  }

  function redraw() {
    const R = LA.reconstruct(dec, k);
    paint(cvOrig, A); paint(cvRec, R);

    const total = dec.s.reduce((s, v) => s + v * v, 0);
    const kept = dec.s.slice(0, k).reduce((s, v) => s + v * v, 0);
    const stored = k * (2 * N + 1);
    const full = N * N;
    let err = 0;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) err += (A[i][j] - R[i][j]) ** 2;

    subOrig.textContent = `${N}×${N} = ${full} numbers`;
    subRec.textContent = `k = ${k}`;
    out.set([
      String(stored),
      String(full),
      { html: stored < full ? `${fmt(full / stored, 2)}× smaller` : `${fmt(stored / full, 2)}× larger`,
        cls: stored < full ? 'is-ok' : 'is-warn' },
      `${fmt(100 * kept / total, 2)}%`,
      fmt(Math.sqrt(err), 1),
    ]);

    const pct = 100 * kept / total;
    if (stored >= full) {
      st.set(`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 9v4M12 17v.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/></svg>` +
        `<span>At this k you are storing <strong>more</strong> numbers than the original image. Low-rank only pays off when k is small.</span>`, 'warn');
    } else if (pct > 99) {
      st.set(`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>` +
        `<span>Keeping just <strong>${k}</strong> of ${N} components already captures <strong>${fmt(pct, 1)}%</strong> of the energy at ${fmt(full / stored, 1)}× compression.</span>`, 'ok');
    } else {
      st.set(`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16.5v.01" stroke-linecap="round"/></svg>` +
        `<span>${fmt(pct, 1)}% of the energy retained. Slide k up until the difference stops being visible.</span>`, 'info');
    }
    specPlot.render();
  }

  specPlot.onDraw(p => {
    const total = dec.s.reduce((s, v) => s + v * v, 0) || 1;
    const mx = dec.s[0] || 1;
    // normalised singular values as bars
    const bw = Math.max(1, p.px(1) * .8);
    dec.s.forEach((v, i) => {
      const h = v / mx;
      const x = p.X(i + .5), y0 = p.Y(0), y1 = p.Y(h);
      p.ctx.fillStyle = i < k ? C.c1 : C.grid;
      p.ctx.fillRect(x - bw / 2, y1, bw, y0 - y1);
    });
    // cumulative energy curve
    let acc = 0;
    const cum = dec.s.map((v, i) => { acc += v * v; return [i + 1, acc / total]; });
    p.path([[0, 0], ...cum], { color: C.c2, lw: 2.2 });
    p.line([k, 0], [k, 1.06], { color: C.c4, lw: 1.6, dash: [5, 4] });
    p.text({ px: 8, py: 11 }, 'σᵢ (bars) · cumulative energy (line)', { color: C.muted, size: 10.5 });
  });

  redraw();

  node.appendChild(note(
    `Storing a rank-k approximation costs <span class="u-mono">k(m + n + 1)</span> numbers instead of ` +
    `<span class="u-mono">m·n</span>. Because singular values decay fast for real images, a handful of ` +
    `components carries almost all the structure and the rest is mostly noise — which is why the same trick ` +
    `is used for <strong>denoising</strong>, <strong>PCA</strong> and <strong>compressing network weights</strong>. ` +
    `Compare the three images: the smooth one compresses beautifully, the noisy one barely at all.`
  ));

  /* --- synthetic test images --- */
  function buildScene(n) {
    const M = LA.zeros(n, n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const x = j / n, y = i / n;
      let v = 150 + 90 * Math.sin(3.1 * x + .6) * Math.cos(2.2 * y);
      const dx = x - .34, dy = y - .38;
      if (dx * dx + dy * dy < .026) v = 232;
      if (Math.abs(x - .74) < .11 && y > .3) v = 58;
      M[i][j] = clamp(v, 0, 255);
    }
    return M;
  }
  function buildStripes(n) {
    const M = LA.zeros(n, n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const band = Math.floor(j / (n / 8)) % 2;
      const ramp = 60 + 150 * (i / n);
      M[i][j] = clamp(band ? ramp : 255 - ramp, 0, 255);
    }
    return M;
  }
  function buildNoisy(n) {
    // deterministic pseudo-noise so the figure is stable across reloads
    let seed = 12345;
    const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
    const M = LA.zeros(n, n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) M[i][j] = clamp(40 + 190 * rnd(), 0, 255);
    return M;
  }
});

/* ============================================================
   9. Norms — unit balls for varying p
   ============================================================ */
defineWidget('norms', node => {
  const { right, canvas } = split(node, { aspect: 1.1, hint: 'Drag the point' });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -2.1, xmax: 2.1, ymin: -2.1, ymax: 2.1, aspect: 1.1, pad: 8,
  }));

  let pval = 2;
  let pt = [1.2, .7];
  const showAll = { all: true };

  const pCtl = slider('p (norm exponent)', {
    min: 1, max: 6, step: .05, value: 2,
    format: v => v >= 5.95 ? '∞' : fmt(v, 2),
    onInput: v => { pval = v; refresh(); },
  });
  const quick = el('div', { class: 'pg-actions' },
    button('p = 1', () => { pval = 1; pCtl.set(1); refresh(); }),
    button('p = 2', () => { pval = 2; pCtl.set(2); refresh(); }),
    button('p = ∞', () => { pval = 6; pCtl.set(6); refresh(); }),
  );
  const allCtl = toggle('Show ℓ₁, ℓ₂, ℓ∞ together', {
    value: true, onChange: v => { showAll.all = v; plot.render(); },
  });
  const out = readout([['x', 0], ['‖x‖₁', 0], ['‖x‖₂', 0], ['‖x‖∞', 0], ['‖x‖ₚ', 0]]);
  right.append(pCtl.root, quick, allCtl.root, out.root);

  const pnorm = (v, p) => p >= 5.95
    ? Math.max(Math.abs(v[0]), Math.abs(v[1]))
    : (Math.abs(v[0]) ** p + Math.abs(v[1]) ** p) ** (1 / p);

  const ball = p => {
    const pts = [];
    for (let i = 0; i <= 240; i++) {
      const t = i / 240 * Math.PI * 2;
      const c = Math.cos(t), s = Math.sin(t);
      const r = 1 / Math.max(1e-9, pnorm([c, s], p));
      pts.push([c * r, s * r]);
    }
    return pts;
  };

  const drag = new Dragger(plot);
  drag.add(() => pt, p => { pt = [clamp(round(p[0], 2), -2, 2), clamp(round(p[1], 2), -2, 2)]; });
  drag.onchange = refresh;

  function refresh() { plot.render(); sync(); }
  function sync() {
    out.set([
      `(${fmt(pt[0], 2)}, ${fmt(pt[1], 2)})`,
      fmt(Math.abs(pt[0]) + Math.abs(pt[1]), 3),
      fmt(Math.hypot(...pt), 3),
      fmt(Math.max(Math.abs(pt[0]), Math.abs(pt[1])), 3),
      fmt(pnorm(pt, pval), 3),
    ]);
  }

  plot.onDraw(p => {
    p.grid(.5, { color: C.grid }); p.axes({ ticks: 1 });
    if (showAll.all) {
      p.path(ball(1), { color: C.c3, lw: 1.5, alpha: .55, close: true, dash: [5, 4] });
      p.path(ball(2), { color: C.c5, lw: 1.5, alpha: .55, close: true, dash: [5, 4] });
      p.path(ball(6), { color: C.c2, lw: 1.5, alpha: .55, close: true, dash: [5, 4] });
    }
    p.path(ball(pval), { color: C.c1, lw: 2.8, close: true, fill: C.fill });
    p.arrow([0, 0], pt, { color: C.c4, lw: 2.6 });
    p.handle(pt, { color: C.c4, r: 6 });
    p.text({ px: 12, py: 18 },
      pval >= 5.95 ? '‖x‖∞ = 1  (square)' : `‖x‖${sub(pval)} = 1`,
      { color: C.c1, size: 12, weight: 700 });
  });
  function sub(v) { return v === 1 ? '₁' : v === 2 ? '₂' : `_${fmt(v, 2)}`; }
  sync();

  node.appendChild(note(
    `A norm answers "how big is this vector?", and there is more than one honest answer. The shaded region is ` +
    `every vector of length exactly 1 under the current p. At <strong>p = 1</strong> it is a diamond — its ` +
    `corners sit on the axes, which is precisely why ℓ₁ penalties (LASSO) drive coefficients to exactly zero. ` +
    `At <strong>p = 2</strong> it is the familiar circle, and as <strong>p → ∞</strong> it inflates to a square.`
  ));
});

/* ============================================================
   10. Orthogonal projection
   ============================================================ */
defineWidget('projection', node => {
  const { right, canvas } = split(node, { aspect: 1.35, hint: 'Drag either arrow' });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -1.5, xmax: 6, ymin: -1.5, ymax: 5, aspect: 1.35, pad: 8,
  }));

  let a = [1, 2], b = [3, 4];
  const out = readout([['a', 0], ['b', 0], ['⟨a, b⟩', 0], ['proj_a(b)', 0], ['residual', 0], ['⟨a, residual⟩', 0]]);
  const st = status('');
  right.append(out.root, st.root);

  const proj = () => {
    const d = a[0] * a[0] + a[1] * a[1];
    if (d < 1e-9) return [0, 0];
    const s = (a[0] * b[0] + a[1] * b[1]) / d;
    return [a[0] * s, a[1] * s];
  };

  const drag = new Dragger(plot);
  drag.add(() => a, p => { a = [round(clamp(p[0], -1.2, 5.6), 2), round(clamp(p[1], -1.2, 4.6), 2)]; });
  drag.add(() => b, p => { b = [round(clamp(p[0], -1.2, 5.6), 2), round(clamp(p[1], -1.2, 4.6), 2)]; });
  drag.onchange = () => { plot.render(); sync(); };

  function sync() {
    const pr = proj();
    const res = [b[0] - pr[0], b[1] - pr[1]];
    const ip = a[0] * res[0] + a[1] * res[1];
    out.set([
      `(${fmt(a[0], 2)}, ${fmt(a[1], 2)})`,
      `(${fmt(b[0], 2)}, ${fmt(b[1], 2)})`,
      fmt(a[0] * b[0] + a[1] * b[1], 2),
      `(${fmt(pr[0], 2)}, ${fmt(pr[1], 2)})`,
      `(${fmt(res[0], 2)}, ${fmt(res[1], 2)})`,
      { html: fmt(Math.abs(ip) < 1e-9 ? 0 : ip, 4), cls: 'is-ok' },
    ]);
    st.set(`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>` +
      `<span>The residual is <strong>always</strong> perpendicular to a — that is what makes this the closest point.</span>`, 'ok');
  }

  plot.onDraw(p => {
    p.grid(1); p.axes({ ticks: 1 });
    const pr = proj();
    p.ray([0, 0], a, { color: C.c1, lw: 1.3, dash: [6, 5], alpha: .5 });
    p.line(pr, b, { color: C.c4, lw: 2, dash: [5, 4] });
    p.arrow([0, 0], a, { color: C.c1, lw: 3 });
    p.arrow([0, 0], b, { color: C.c3, lw: 3 });
    p.arrow([0, 0], pr, { color: C.c2, lw: 3.4 });
    if (Math.hypot(...a) > .2) p.rightAngle([0, 0], pr, b, { size: 12, color: C.c4 });
    p.badge(a, 'a', { color: C.c1, align: 'center', dy: -16 });
    p.badge(b, 'b', { color: C.c3, align: 'center', dy: -16 });
    p.badge(pr, 'proj', { color: C.c2, align: 'center', dy: 18 });
    p.handle(a, { color: C.c1, r: 6 });
    p.handle(b, { color: C.c3, r: 6 });
  });
  sync();

  node.appendChild(note(
    `The projection is the <strong>shadow</strong> b casts on the line through a — the single closest point on ` +
    `that line. The dashed error vector meets a at a right angle no matter where you drag, and that ` +
    `perpendicularity is the whole of least squares: regression finds the projection of your labels onto the ` +
    `space your features can reach.`
  ));
});

/* ============================================================
   11. Pixels ↔ matrix — draw a digit, read the numbers
   ============================================================ */
defineWidget('pixel-matrix', node => {
  const N = 16;
  let G = LA.zeros(N, N);
  seedDigit();

  const wrap = el('div');
  node.appendChild(wrap);

  const cvImg = el('canvas', { style: 'width:100%;image-rendering:pixelated;cursor:crosshair' });
  const cvNum = el('canvas', { style: 'width:100%' });
  let hover = null;

  const grid = el('div', { style: 'display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem' },
    el('div', {},
      el('div', { class: 'matrix-label', html: 'The image &nbsp;<span style="font-weight:400;color:var(--ink-faint)">— draw on it</span>' }),
      el('div', { class: 'pg-canvas-wrap' }, cvImg)),
    el('div', {},
      el('div', { class: 'matrix-label', html: 'The matrix <b>I</b> ∈ ℝ<sup>16×16</sup>' }),
      el('div', { class: 'pg-canvas-wrap' }, cvNum))
  );

  const acts = el('div', { class: 'pg-actions', style: 'margin-top:1rem' },
    button('Clear', () => { G = LA.zeros(N, N); draw(); }),
    button('Reset digit', () => { seedDigit(); draw(); }),
    button('Invert', () => { G = G.map(r => r.map(v => 255 - v)); draw(); }),
    button('Brighten ×1.2', () => { G = G.map(r => r.map(v => clamp(v * 1.2, 0, 255))); draw(); }),
    button('Blur 3×3', () => { G = blur(G); draw(); }),
  );
  const st = status(
    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16.5v.01" stroke-linecap="round"/></svg>` +
    `Paint with the mouse — every stroke you make is just a number changing in the grid on the right.`, 'info');
  wrap.append(grid, acts, st.root);

  const pImg = trackPlot(new Plot(cvImg, { xmin: 0, xmax: N, ymin: 0, ymax: N, aspect: 1, pad: 0 }));
  const pNum = trackPlot(new Plot(cvNum, { xmin: 0, xmax: N, ymin: 0, ymax: N, aspect: 1, pad: 0 }));

  function cellAt(ev) {
    const r = cvImg.getBoundingClientRect();
    const j = Math.floor((ev.clientX - r.left) / r.width * N);
    const i = Math.floor((ev.clientY - r.top) / r.height * N);
    return (i >= 0 && i < N && j >= 0 && j < N) ? [i, j] : null;
  }
  let painting = false;
  cvImg.addEventListener('pointerdown', e => {
    painting = true; cvImg.setPointerCapture?.(e.pointerId); paint(e);
  });
  cvImg.addEventListener('pointermove', e => {
    const c = cellAt(e);
    if (`${hover}` !== `${c}`) { hover = c; draw(); }
    if (painting) paint(e);
  });
  const stopPaint = () => { painting = false; };
  cvImg.addEventListener('pointerup', stopPaint);
  cvImg.addEventListener('pointercancel', stopPaint);
  cvImg.addEventListener('pointerleave', () => { hover = null; painting = false; draw(); });

  function paint(ev) {
    const c = cellAt(ev); if (!c) return;
    const [i, j] = c;
    for (let di = -1; di <= 1; di++) for (let dj = -1; dj <= 1; dj++) {
      const y = i + di, x = j + dj;
      if (y < 0 || y >= N || x < 0 || x >= N) continue;
      const w = di === 0 && dj === 0 ? 255 : 120;
      G[y][x] = Math.max(G[y][x], w);
    }
    draw();
  }

  function draw() { pImg.render(); pNum.render(); }

  pImg.onDraw(p => {
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const v = Math.round(G[i][j]);
      p.ctx.fillStyle = `rgb(${v},${v},${v})`;
      p.ctx.fillRect(p.X(j), p.Y(N - i), p.px(1) + 1, p.px(1) + 1);
    }
    if (hover) {
      p.ctx.strokeStyle = C.c2; p.ctx.lineWidth = 2.4;
      p.ctx.strokeRect(p.X(hover[1]), p.Y(N - hover[0]), p.px(1), p.px(1));
    }
  });

  pNum.onDraw(p => {
    const cell = p.px(1);
    const fs = Math.max(6, Math.min(11, cell * .42));
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const v = Math.round(G[i][j]);
      const t = v / 255;
      p.ctx.fillStyle = `rgba(${hexToRgb(C.c1)},${.06 + t * .55})`;
      p.ctx.fillRect(p.X(j), p.Y(N - i), cell + 1, cell + 1);
      if (cell > 13) {
        p.ctx.fillStyle = t > .55 ? C.raised : C.muted;
        p.ctx.font = `600 ${fs}px ${getComputedStyle(document.documentElement).getPropertyValue('--font-mono')}`;
        p.ctx.textAlign = 'center'; p.ctx.textBaseline = 'middle';
        p.ctx.fillText(String(v), p.X(j) + cell / 2, p.Y(N - i) + cell / 2);
      }
    }
    if (hover) {
      p.ctx.strokeStyle = C.c2; p.ctx.lineWidth = 2.4;
      p.ctx.strokeRect(p.X(hover[1]), p.Y(N - hover[0]), cell, cell);
    }
  });

  function hexToRgb(h) {
    h = h.trim();
    if (h.startsWith('#')) {
      const n = h.length === 4
        ? h.slice(1).split('').map(c => parseInt(c + c, 16))
        : [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
      return n.join(',');
    }
    return '90,55,212';
  }

  function blur(M) {
    const O = LA.zeros(N, N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      let s = 0, c = 0;
      for (let di = -1; di <= 1; di++) for (let dj = -1; dj <= 1; dj++) {
        const y = i + di, x = j + dj;
        if (y < 0 || y >= N || x < 0 || x >= N) continue;
        s += M[y][x]; c++;
      }
      O[i][j] = s / c;
    }
    return O;
  }

  function seedDigit() {
    G = LA.zeros(N, N);
    // a hand-drawn-looking "0"
    for (let t = 0; t < 260; t++) {
      const th = t / 260 * Math.PI * 2;
      const rx = 4.1 + .35 * Math.sin(th * 2.1);
      const ry = 5.6 + .3 * Math.cos(th * 1.7);
      const x = 7.5 + rx * Math.cos(th);
      const y = 7.5 + ry * Math.sin(th);
      for (let di = -1; di <= 1; di++) for (let dj = -1; dj <= 1; dj++) {
        const i = Math.round(y) + di, j = Math.round(x) + dj;
        if (i < 0 || i >= N || j < 0 || j >= N) continue;
        const w = (di === 0 && dj === 0) ? 255 : 140;
        G[i][j] = Math.max(G[i][j], w);
      }
    }
  }

  draw();

  node.appendChild(note(
    `A grayscale image <em>is</em> a matrix — nothing is lost in the translation. Each entry is one pixel ` +
    `intensity, 0 for black and 255 for white. "Brighten" is scalar multiplication, "invert" is ` +
    `255 − <b>I</b>, and "blur" is a convolution with a 3×3 averaging kernel. Every image operation you know ` +
    `is linear algebra wearing a friendlier name.`
  ));
});

/* ============================================================
   12. Null space — which vectors get sent to zero
   ============================================================ */
defineWidget('nullspace', node => {
  const { right, canvas } = split(node, { aspect: 1.3, hint: 'Drag x' });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -4, xmax: 4, ymin: -3.2, ymax: 3.2, aspect: 1.3, pad: 8,
  }));

  let A = [[1, 2], [2, 4]];
  let x = [2, -1];

  const Min = matrixInput(2, 2, A, {
    label: 'A', step: .5, onInput: m => { A = m; refresh(); },
  });
  const presets = el('div', { class: 'pg-actions' },
    button('Singular', () => set([[1, 2], [2, 4]])),
    button('Identity', () => set([[1, 0], [0, 1]])),
    button('Zero matrix', () => set([[0, 0], [0, 0]])),
  );
  const out = readout([['x', 0], ['Ax', 0], ['rank(A)', 0], ['nullity', 0], ['rank + nullity', 0]]);
  const st = status('');
  right.append(Min.root, presets, out.root, st.root);

  function set(m) { A = m.map(r => r.slice()); Min.set(A); refresh(); }
  function refresh() { plot.render(); sync(); }

  const drag = new Dragger(plot);
  drag.add(() => x, p => { x = [round(clamp(p[0], -3.5, 3.5), 2), round(clamp(p[1], -3, 3), 2)]; });
  drag.onchange = refresh;

  function sync() {
    const ax = LA.matvec(A, x);
    const rk = LA.rank(A);
    const nul = 2 - rk;
    const isNull = Math.hypot(...ax) < 1e-6;
    out.set([
      `(${fmt(x[0], 2)}, ${fmt(x[1], 2)})`,
      { html: `(${fmt(ax[0], 2)}, ${fmt(ax[1], 2)})`, cls: isNull ? 'is-ok' : '' },
      String(rk), String(nul), `${rk} + ${nul} = 2 ✓`,
    ]);

    if (nul === 0) {
      st.set(`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>` +
        `<span><strong>Trivial null space.</strong> Only x = 0 maps to zero, so nothing is lost and A is invertible.</span>`, 'ok');
    } else if (isNull) {
      st.set(`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>` +
        `<span><strong>x is in the null space.</strong> A crushes it to the origin — and so it would every scalar multiple of x.</span>`, 'ok');
    } else {
      st.set(`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16.5v.01" stroke-linecap="round"/></svg>` +
        `<span>Drag x onto the dashed line — that is the whole null space, a ${nul === 2 ? 'plane' : 'line'} of vectors A sends to zero.</span>`, '');
    }
  }

  plot.onDraw(p => {
    p.grid(1); p.axes({ ticks: 1 });
    const basis = LA.nullBasis(A);
    basis.forEach(v => {
      if (Math.hypot(...v) > 1e-9) {
        p.ray([0, 0], v, { color: C.c3, lw: 2.4, dash: [8, 5], alpha: .8 });
      }
    });
    if (basis.length) {
      const v = basis[0];
      const n = Math.hypot(...v) || 1;
      p.badge([v[0] / n * 2.6, v[1] / n * 2.6], 'Null(A)', { color: C.c3, align: 'center' });
    }
    const ax = LA.matvec(A, x);
    p.arrow([0, 0], x, { color: C.c1, lw: 3 });
    p.arrow([0, 0], ax, { color: C.c4, lw: 3 });
    p.badge(x, 'x', { color: C.c1, align: 'center', dy: -16 });
    if (Math.hypot(...ax) > .18) p.badge(ax, 'Ax', { color: C.c4, align: 'center', dy: -16 });
    else p.badge([0, 0], 'Ax = 0', { color: C.c3, align: 'center', dy: 20 });
    p.handle(x, { color: C.c1, r: 6 });
  });
  sync();

  node.appendChild(note(
    `The null space collects every input that A annihilates. It is never empty — <strong>x = 0 is always ` +
    `there</strong> — so the right question is how <em>big</em> it is. Rank–nullity says the dimensions ` +
    `always add up to n: whatever the map does not preserve, it destroys.`
  ));
});

/* ============================================================
   13. Row reduction, one step at a time
   ============================================================ */
defineWidget('rref-stepper', node => {
  const wrap = el('div');
  node.appendChild(wrap);

  let M = [[1, 1, 3], [2, -1, 0]];
  let steps = [], idx = 0;

  const Min = matrixInput(2, 3, M, {
    label: 'Augmented matrix &nbsp;[A | b]', step: 1,
    onInput: m => { M = m; rebuild(); },
  });
  Min.cells.forEach(r => { r[2].style.marginLeft = '7px'; r[2].style.borderLeftWidth = '2px'; });

  const view = el('div', { class: 'readout', style: 'min-height:5.4em' });
  const label = el('div', { style: 'font-size:.88rem;color:var(--ink-muted);margin:.65rem 0 .5rem;min-height:1.4em' });
  const nav = el('div', { class: 'pg-actions' },
    button('◀ Back', () => { idx = Math.max(0, idx - 1); paint(); }),
    button('Step ▶', () => { idx = Math.min(steps.length - 1, idx + 1); paint(); }),
    button('Run to RREF', () => { idx = steps.length - 1; paint(); }),
    button('Restart', () => { idx = 0; paint(); }),
  );
  wrap.append(Min.root, nav, label, view);

  function rebuild() { steps = LA.rref(M, { trace: true }).steps; idx = 0; paint(); }

  function paint() {
    const s = steps[idx];
    if (!s) return;
    label.innerHTML = `<strong>Step ${idx} / ${steps.length - 1}:</strong> ${s.desc}`;
    view.innerHTML = s.M.map((row, i) => {
      const hl = s.hi?.rows?.includes(i);
      const cells = row.map((v, j) => {
        const c = s.hi?.cols?.includes(j);
        const txt = fmt(v, 2).padStart(6);
        return (c || hl)
          ? `<span style="color:var(--accent);font-weight:700">${txt}</span>`
          : txt;
      });
      return `<div>[ ${cells.slice(0, 2).join(' ')} | ${cells[2]} ]</div>`;
    }).join('');
  }
  rebuild();

  node.appendChild(note(
    `Each row operation is <strong>reversible</strong> and leaves the solution set untouched — that is the ` +
    `whole justification for the method. Watch for the tell-tale endings: a row reading ` +
    `<span class="u-mono">[0 0 | k]</span> with k ≠ 0 means <em>no solution</em>, and a row of all zeros ` +
    `means a free variable and <em>infinitely many</em>.`
  ));
});

/* ============================================================
   14. Matrix zoo — special structures at a glance
   ============================================================ */
defineWidget('matrix-zoo', node => {
  const kinds = {
    identity:  { M: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], name: 'Identity', blurb: 'Leaves every vector untouched. The "do nothing" matrix.' },
    diagonal:  { M: [[3, 0, 0], [0, 7, 0], [0, 0, 2]], name: 'Diagonal', blurb: 'Scales each axis independently. Trivial to invert: reciprocate the diagonal.' },
    symmetric: { M: [[2, -1, 0], [-1, 3, 4], [0, 4, 5]], name: 'Symmetric', blurb: 'A = Aᵀ — mirror-image across the diagonal. Always has real eigenvalues and an orthonormal eigenbasis.' },
    skew:      { M: [[0, 2, -1], [-2, 0, -4], [1, 4, 0]], name: 'Skew-symmetric', blurb: 'Aᵀ = −A, so the diagonal must be zero. Generates rotations; eigenvalues are imaginary or zero.' },
    upper:     { M: [[1, 2, 3], [0, 5, 6], [0, 0, 9]], name: 'Upper triangular', blurb: 'Everything below the diagonal is zero. Solve by back-substitution — no full inversion needed.' },
    lower:     { M: [[4, 0, 0], [7, 2, 0], [5, 8, 1]], name: 'Lower triangular', blurb: 'The mirror case; forward substitution. Together these give the LU decomposition.' },
    orthogonal:{ M: [[0.7071, 0.7071, 0], [-0.7071, 0.7071, 0], [0, 0, 1]], name: 'Orthogonal', blurb: 'QᵀQ = I. Rotations and reflections — lengths and angles come through unchanged.' },
    sparse:    { M: [[0, 0, 3], [0, 0, 0], [0, 1, 0]], name: 'Sparse', blurb: 'Mostly zeros. Store only what is non-zero and skip the rest of the arithmetic entirely.' },
    rank1:     { M: [[3, 4, 5], [6, 8, 10], [9, 12, 15]], name: 'Rank one', blurb: 'An outer product uvᵀ. Every row is a multiple of every other — 6 numbers instead of 9.' },
  };
  let cur = 'identity';

  const wrap = el('div');
  node.appendChild(wrap);

  const pick = segmented(
    Object.entries(kinds).map(([k, v]) => ({ label: v.name, value: k })),
    { value: cur, onChange: v => { cur = v; refresh(); } }
  );

  const cvHeat = el('canvas');
  const view = matrixView(kinds[cur].M, { fmt: v => fmt(v, 2) });
  const blurb = el('p', { style: 'font-size:.92rem;color:var(--ink-muted);margin:.9rem 0 0' });
  const out = readout([['symmetric', 0], ['rank', 0], ['det', 0], ['trace', 0], ['non-zeros', 0]]);

  const grid = el('div', { style: 'display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:1.1rem;align-items:start;margin-top:1rem' },
    el('div', {}, el('div', { class: 'matrix-label', html: 'Entries' }), view.root),
    el('div', {}, el('div', { class: 'matrix-label', html: 'Sparsity pattern' }), el('div', { class: 'pg-canvas-wrap' }, cvHeat)),
    out.root
  );
  wrap.append(pick.root, grid, blurb);

  const heat = trackPlot(new Plot(cvHeat, { xmin: 0, xmax: 3, ymin: 0, ymax: 3, aspect: 1, pad: 0 }));

  function refresh() {
    const { M, blurb: b } = kinds[cur];
    view.set(M);
    blurb.innerHTML = b;
    const T = LA.transpose(M);
    const sym = M.every((r, i) => r.every((v, j) => Math.abs(v - T[i][j]) < 1e-9));
    const nz = M.flat().filter(v => Math.abs(v) > 1e-9).length;
    out.set([
      { html: sym ? 'yes' : 'no', cls: sym ? 'is-ok' : '' },
      String(LA.rank(M)),
      fmt(LA.det(M), 3),
      fmt(M[0][0] + M[1][1] + M[2][2], 2),
      `${nz} / 9`,
    ]);
    heat.render();
  }

  heat.onDraw(p => {
    const M = kinds[cur].M;
    const mx = Math.max(...M.flat().map(Math.abs), 1);
    const cell = p.px(1);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
      const v = M[i][j];
      const t = Math.abs(v) / mx;
      p.ctx.fillStyle = Math.abs(v) < 1e-9
        ? C.grid
        : (v > 0 ? withAlpha(C.c1, .18 + t * .75) : withAlpha(C.c4, .18 + t * .75));
      p.ctx.fillRect(p.X(j) + 1.5, p.Y(3 - i) + 1.5, cell - 3, cell - 3);
    }
    p.text({ px: 6, py: 12 }, 'zero', { color: C.muted, size: 10 });
  });

  function withAlpha(hex, a) {
    hex = hex.trim();
    if (!hex.startsWith('#')) return hex;
    const n = hex.length === 4
      ? hex.slice(1).split('').map(c => parseInt(c + c, 16))
      : [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));
    return `rgba(${n.join(',')},${a})`;
  }

  refresh();

  node.appendChild(note(
    `These are not arbitrary labels — each structure buys you something concrete. Triangular means you can ` +
    `solve by substitution instead of inverting; orthogonal means the inverse is just the transpose; sparse ` +
    `means you can skip most of the arithmetic; symmetric guarantees real eigenvalues. ` +
    `<strong>Recognising structure is how you make large problems tractable.</strong>`
  ));
});
