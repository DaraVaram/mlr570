/* ============================================================
   widgets/sim.js — interactive figures for Similarity Measures
   ============================================================ */
import {
  Plot, Dragger, C, el, slider, toggle, segmented, button,
  readout, status, defineWidget, canvasHost, trackPlot,
  clamp, fmt, round,
} from '../viz.js';
import * as ML from '../ml.js';

function split(node, { hint, wide = false } = {}) {
  const left = el('div');
  const right = el('div', { class: 'pg-controls' });
  node.appendChild(el('div', { class: 'pg-split' + (wide ? ' pg-split--wide-ctrl' : '') }, left, right));
  const { canvas } = canvasHost(left, { hint });
  return { left, right, canvas };
}
const note = html => el('div', { class: 'pg-note', html });
const ICON_OK = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>`;
const ICON_INFO = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16.5v.01" stroke-linecap="round"/></svg>`;
const ICON_WARN = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 9v4M12 17v.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/></svg>`;

/* ============================================================
   1. Pearson correlation — and the traps it falls into
   ============================================================ */
defineWidget('pearson-explore', node => {
  const { right, canvas } = split(node, { hint: 'Drag any point', wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -6, xmax: 6, ymin: -6, ymax: 6, aspect: 1.2, pad: 8,
  }));

  const PRESETS = {
    linear:    { name: 'Linear', make: () => Array.from({ length: 11 }, (_, i) => [i - 5, (i - 5) * 0.9]) },
    parabola:  { name: 'Parabola', make: () => Array.from({ length: 11 }, (_, i) => [i - 5, ((i - 5) ** 2) / 4.2 - 3]) },
    noisy:     { name: 'Noisy trend', make: () => { const r = ML.rng(8); return Array.from({ length: 24 }, (_, i) => { const x = -5 + i * 10 / 23; return [x, x * .7 + ML.gauss(r) * 1.6]; }); } },
    outlier:   { name: 'One outlier', make: () => { const base = Array.from({ length: 12 }, (_, i) => [-5 + i * .55, -2.4 + i * .12]); base.push([5, 5.4]); return base; } },
    none:      { name: 'No relation', make: () => { const r = ML.rng(3); return Array.from({ length: 24 }, () => [ML.gauss(r) * 2.4, ML.gauss(r) * 2.4]); } },
  };
  let key = 'linear';
  let pts = PRESETS[key].make();
  let showSpearman = true;

  const pick = segmented(
    Object.entries(PRESETS).map(([k, v]) => ({ label: v.name, value: k })),
    { value: key, label: 'Pattern', onChange: v => { key = v; pts = PRESETS[v].make(); rebind(); refresh(); } }
  );
  const spCtl = toggle('Also show Spearman (rank) ρ', { value: true, onChange: v => { showSpearman = v; sync(); } });
  const out = readout([['n points', 0], ['Pearson r', 0], ['Spearman ρ', 0], ['best-fit slope', 0]]);
  const st = status('');
  right.append(pick.root, spCtl.root, out.root, st.root);

  const drag = new Dragger(plot);
  function rebind() {
    drag.clear();
    pts.forEach((_, i) => drag.add(() => pts[i], p => {
      pts[i] = [clamp(p[0], -5.7, 5.7), clamp(p[1], -5.7, 5.7)];
    }, { r: 12 }));
  }
  drag.onchange = refresh;
  rebind();

  function refresh() { plot.render(); sync(); }

  function sync() {
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
    const r = ML.pearson(xs, ys);
    const sp = ML.spearman(xs, ys);
    const mx = ML.mean(xs), my = ML.mean(ys);
    let num = 0, den = 0;
    pts.forEach(([x, yv]) => { num += (x - mx) * (yv - my); den += (x - mx) ** 2; });
    const slope = den < 1e-12 ? 0 : num / den;

    out.set([
      String(pts.length),
      { html: fmt(r, 4), cls: Math.abs(r) > .8 ? 'is-ok' : Math.abs(r) < .2 ? 'is-warn' : '' },
      { html: showSpearman ? fmt(sp, 4) : '—', cls: showSpearman && Math.abs(sp) > .8 ? 'is-ok' : '' },
      fmt(slope, 3),
    ]);

    let msg, kind;
    if (key === 'parabola' || (Math.abs(r) < .2 && Math.abs(sp) < .3 && key !== 'none')) {
      msg = `${ICON_WARN}<span><strong>r ≈ ${fmt(r, 3)}, yet the relationship is perfect.</strong> Pearson measures <em>linear</em> association only. Every point here is exactly determined by x — Pearson simply cannot see it.</span>`;
      kind = 'warn';
    } else if (key === 'outlier') {
      msg = `${ICON_WARN}<span><strong>One point is doing the work.</strong> Drag the top-right outlier back down to the trend and watch r collapse — a single observation can manufacture a correlation.</span>`;
      kind = 'warn';
    } else if (Math.abs(r) > .9) {
      msg = `${ICON_OK}<span>Strong linear association (r = ${fmt(r, 3)}). Pearson is the right tool for exactly this shape.</span>`;
      kind = 'ok';
    } else {
      msg = `${ICON_INFO}<span>r = ${fmt(r, 3)}. Remember r measures <em>tightness about a straight line</em>, not the steepness of it — the slope is a separate number.</span>`;
      kind = 'info';
    }
    st.set(msg, kind);
  }

  plot.onDraw(p => {
    p.grid(1); p.axes({ ticks: 2 });
    const xs = pts.map(q => q[0]), ys = pts.map(q => q[1]);
    const mx = ML.mean(xs), my = ML.mean(ys);
    let num = 0, den = 0;
    pts.forEach(([x, yv]) => { num += (x - mx) * (yv - my); den += (x - mx) ** 2; });
    const slope = den < 1e-12 ? 0 : num / den;

    // quadrant shading about the means: the sign of each product term
    p.line([p.o.xmin, my], [p.o.xmax, my], { color: C.muted, lw: 1, dash: [4, 4], alpha: .5 });
    p.line([mx, p.o.ymin], [mx, p.o.ymax], { color: C.muted, lw: 1, dash: [4, 4], alpha: .5 });
    p.ray([mx, my], [1, slope], { color: C.c2, lw: 2.2, alpha: .9 });

    pts.forEach(q => {
      const sign = (q[0] - mx) * (q[1] - my);
      p.dot(q, { r: 5, color: sign >= 0 ? C.c1 : C.c4, ring: true, ringLw: 2 });
    });
    p.dot([mx, my], { r: 5, color: C.c3, ring: true });
    p.badge([mx, my], 'mean', { color: C.c3, align: 'center', dy: 18 });
    p.text({ px: 12, py: 11 }, 'violet: (x−x̄)(y−ȳ) > 0   ·   red: < 0', { color: C.muted, size: 10.5 });
  });

  refresh();

  node.appendChild(note(
    `Pearson's r is the average of the products <span class="u-mono">(x−x̄)(y−ȳ)</span>, normalised by the ` +
    `two standard deviations. Points coloured violet push r up; red points push it down. The <strong>parabola</strong> ` +
    `preset is the important one: the two sides contribute equal and opposite products, they cancel exactly, ` +
    `and r = 0 despite y being a perfect function of x. Spearman's rank correlation survives some of these ` +
    `traps because it only cares about ordering — but it too returns 0 on the symmetric parabola.`
  ));
});

/* ============================================================
   2. Euclidean vs Manhattan vs Chebyshev
   ============================================================ */
defineWidget('distance-metrics', node => {
  const { right, canvas } = split(node, { hint: 'Drag either point', wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -.6, xmax: 8.6, ymin: -.6, ymax: 6.6, aspect: 1.35, pad: 8,
  }));

  let A = [1, 1], B = [7, 5];
  let snap = true, showBalls = false;

  const snapCtl = toggle('Snap to the grid', { value: true, onChange: v => { snap = v; refresh(); } });
  const ballCtl = toggle('Show equal-distance sets around A', { value: false, onChange: v => { showBalls = v; plot.render(); } });
  const out = readout([
    ['Δx, Δy', 0], ['Euclidean (ℓ₂)', 0], ['Manhattan (ℓ₁)', 0],
    ['Chebyshev (ℓ∞)', 0], ['ℓ₁ / ℓ₂', 0],
  ]);
  const st = status('');
  right.append(snapCtl.root, ballCtl.root, out.root, st.root);

  const fix = v => snap ? Math.round(v) : round(v, 2);
  const drag = new Dragger(plot);
  drag.add(() => A, p => { A = [clamp(fix(p[0]), 0, 8), clamp(fix(p[1]), 0, 6)]; });
  drag.add(() => B, p => { B = [clamp(fix(p[0]), 0, 8), clamp(fix(p[1]), 0, 6)]; });
  drag.onchange = refresh;

  function refresh() { plot.render(); sync(); }
  function sync() {
    const dx = B[0] - A[0], dy = B[1] - A[1];
    const l2 = Math.hypot(dx, dy);
    const l1 = Math.abs(dx) + Math.abs(dy);
    const li = Math.max(Math.abs(dx), Math.abs(dy));
    out.set([
      `${fmt(dx, 2)}, ${fmt(dy, 2)}`,
      fmt(l2, 4), fmt(l1, 4), fmt(li, 4),
      l2 < 1e-9 ? '—' : fmt(l1 / l2, 4),
    ]);
    const aligned = Math.abs(dx) < 1e-9 || Math.abs(dy) < 1e-9;
    st.set(
      aligned
        ? `${ICON_INFO}<span>Along a single axis all three metrics <strong>agree</strong> — they only differ once movement is diagonal.</span>`
        : `${ICON_INFO}<span>ℓ∞ ≤ ℓ₂ ≤ ℓ₁ always holds. Here the ratio ℓ₁/ℓ₂ is <strong>${fmt(l1 / l2, 3)}</strong>; in 2-D it can never exceed √2 ≈ 1.414.</span>`,
      'info');
  }

  plot.onDraw(p => {
    p.grid(1); p.axes({ ticks: 1 });
    // Manhattan staircase path
    const stair = [[A[0], A[1]], [B[0], A[1]], [B[0], B[1]]];
    p.path(stair, { color: C.c2, lw: 3.4, alpha: .9 });
    // Euclidean straight line
    p.line(A, B, { color: C.c1, lw: 3 });
    // Chebyshev: the limiting single-axis move
    const dx = B[0] - A[0], dy = B[1] - A[1];
    const chebEnd = Math.abs(dx) >= Math.abs(dy) ? [B[0], A[1]] : [A[0], B[1]];
    p.line(A, chebEnd, { color: C.c3, lw: 2, dash: [6, 4] });

    if (showBalls) {
      const rad = Math.hypot(dx, dy);
      const ball = (pnorm, col) => {
        const pts = [];
        for (let i = 0; i <= 200; i++) {
          const t = i / 200 * Math.PI * 2;
          const c = Math.cos(t), s = Math.sin(t);
          const nrm = pnorm === Infinity ? Math.max(Math.abs(c), Math.abs(s))
                     : (Math.abs(c) ** pnorm + Math.abs(s) ** pnorm) ** (1 / pnorm);
          pts.push([A[0] + c / nrm * rad, A[1] + s / nrm * rad]);
        }
        p.path(pts, { color: col, lw: 1.5, dash: [5, 4], alpha: .6, close: true });
      };
      ball(1, C.c2); ball(2, C.c1); ball(Infinity, C.c3);
    }

    p.handle(A, { color: C.c5, r: 7 });
    p.handle(B, { color: C.c4, r: 7 });
    p.badge(A, 'A', { color: C.c5, align: 'center', dy: -18 });
    p.badge(B, 'B', { color: C.c4, align: 'center', dy: -18 });
    p.badge([(A[0] + B[0]) / 2, (A[1] + B[1]) / 2], `ℓ₂ = ${fmt(Math.hypot(dx, dy), 2)}`,
      { color: C.c1, align: 'center', dy: -14 });
    p.badge([B[0], A[1]], `ℓ₁ = ${fmt(Math.abs(dx) + Math.abs(dy), 2)}`,
      { color: C.c2, align: 'center', dy: 18 });
  });

  refresh();

  node.appendChild(note(
    `The violet straight line is Euclidean distance — the flying-crow route. The orange staircase is Manhattan ` +
    `distance, which is what you actually travel if diagonal movement is forbidden: city blocks, grid ` +
    `pathfinding, pixel neighbourhoods. Chebyshev takes only the largest single-axis gap, which is how a king ` +
    `moves on a chessboard. Switch on the equal-distance sets and you get the <a href="01-mathematical-background.html#norms">unit balls</a> ` +
    `from Chapter 1 again — diamond, circle, square — now centred on A.`
  ));
});

/* ============================================================
   3. Cosine similarity — angle, not length
   ============================================================ */
defineWidget('cosine-sim', node => {
  const { right, canvas } = split(node, { hint: 'Drag either arrow', wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -5.5, xmax: 5.5, ymin: -4, ymax: 4, aspect: 1.3, pad: 8,
  }));

  let A = [3, 1], B = [1.4, 2.8];
  let scaleB = 1;

  const scaleCtl = slider('Scale B by', {
    min: .2, max: 3, step: .05, value: 1, format: v => `${fmt(v, 2)}×`,
    onInput: v => { scaleB = v; refresh(); },
  });
  const out = readout([
    ['A', 0], ['B (scaled)', 0], ['A · B', 0], ['‖A‖, ‖B‖', 0],
    ['cosine similarity', 0], ['angle', 0], ['Euclidean distance', 0],
  ]);
  const st = status('');
  right.append(scaleCtl.root, out.root, st.root);

  const Bs = () => [B[0] * scaleB, B[1] * scaleB];

  const drag = new Dragger(plot);
  drag.add(() => A, p => { A = [clamp(round(p[0], 2), -5, 5), clamp(round(p[1], 2), -3.6, 3.6)]; });
  drag.add(Bs, p => {
    B = [clamp(round(p[0], 2), -5, 5) / scaleB, clamp(round(p[1], 2), -3.6, 3.6) / scaleB];
  });
  drag.onchange = refresh;

  function refresh() { plot.render(); sync(); }
  function sync() {
    const b = Bs();
    const dot = A[0] * b[0] + A[1] * b[1];
    const na = Math.hypot(...A), nb = Math.hypot(...b);
    const cos = na * nb < 1e-12 ? 0 : dot / (na * nb);
    const ang = Math.acos(clamp(cos, -1, 1)) * 180 / Math.PI;
    const eu = Math.hypot(A[0] - b[0], A[1] - b[1]);
    out.set([
      `(${fmt(A[0], 2)}, ${fmt(A[1], 2)})`,
      `(${fmt(b[0], 2)}, ${fmt(b[1], 2)})`,
      fmt(dot, 3),
      `${fmt(na, 3)}, ${fmt(nb, 3)}`,
      { html: fmt(cos, 4), cls: cos > .95 ? 'is-ok' : cos < 0 ? 'is-warn' : '' },
      `${fmt(ang, 2)}°`,
      fmt(eu, 3),
    ]);
    st.set(
      Math.abs(cos - 1) < 1e-3
        ? `${ICON_OK}<span><strong>Perfectly aligned.</strong> B is a positive multiple of A, so cosine similarity is exactly 1 no matter how long either vector is.</span>`
        : cos < 0
          ? `${ICON_WARN}<span>Angle exceeds 90°, so cosine similarity is <strong>negative</strong> — the vectors point in broadly opposing directions.</span>`
          : `${ICON_INFO}<span>Move the <strong>scale</strong> slider: the Euclidean distance changes a great deal while cosine similarity does not move at all.</span>`,
      Math.abs(cos - 1) < 1e-3 ? 'ok' : cos < 0 ? 'warn' : 'info');
  }

  plot.onDraw(p => {
    p.grid(1); p.axes({ ticks: 1 });
    const b = Bs();
    const na = Math.hypot(...A), nb = Math.hypot(...b);

    // the angle wedge
    if (na > .1 && nb > .1) {
      const a0 = Math.atan2(A[1], A[0]), a1 = Math.atan2(b[1], b[0]);
      const rad = Math.min(na, nb) * .42;
      const pts = [[0, 0]];
      let d = a1 - a0;
      while (d > Math.PI) d -= 2 * Math.PI;
      while (d < -Math.PI) d += 2 * Math.PI;
      for (let i = 0; i <= 40; i++) {
        const t = a0 + d * i / 40;
        pts.push([Math.cos(t) * rad, Math.sin(t) * rad]);
      }
      p.polygon(pts, { fill: C.fill2, stroke: false });
      const mid = a0 + d / 2;
      const cos = na * nb < 1e-12 ? 0 : (A[0] * b[0] + A[1] * b[1]) / (na * nb);
      p.badge([Math.cos(mid) * rad * 1.5, Math.sin(mid) * rad * 1.5],
        `${fmt(Math.acos(clamp(cos, -1, 1)) * 180 / Math.PI, 1)}°`,
        { color: C.c2, align: 'center' });
    }

    // the ray B lives on — every point on it has the same cosine similarity
    p.ray([0, 0], B, { color: C.c3, lw: 1.2, dash: [6, 5], alpha: .45 });
    p.line([0, 0], [A[0] - b[0] + b[0], A[1] - b[1] + b[1]], { color: 'transparent' });
    p.line(A, b, { color: C.c4, lw: 1.6, dash: [4, 4], alpha: .7 });

    p.arrow([0, 0], A, { color: C.c1, lw: 3.2 });
    p.arrow([0, 0], b, { color: C.c3, lw: 3.2 });
    p.badge(A, 'A', { color: C.c1, align: 'center', dy: -17 });
    p.badge(b, 'B', { color: C.c3, align: 'center', dy: -17 });
    p.handle(A, { color: C.c1, r: 6 });
    p.handle(b, { color: C.c3, r: 6 });
    p.text({ px: 12, py: 11 }, 'dashed red: Euclidean gap · dashed teal: B’s direction',
      { color: C.muted, size: 10.5, halo: true, haloWidth: 4 });
  });

  refresh();

  node.appendChild(note(
    `Cosine similarity throws away magnitude and keeps only direction. Slide B's scale from 0.2× to 3×: it ` +
    `slides along its dashed ray, the Euclidean distance to A changes enormously, and the cosine similarity ` +
    `does not budge. That invariance is why it dominates <strong>text and embeddings</strong> — a long ` +
    `document and a short one about the same topic point the same way even though their word-count vectors ` +
    `have very different lengths. The flip side: if magnitude carries real information, cosine similarity ` +
    `discards it.`
  ));
});

/* ============================================================
   4. Jaccard similarity — overlap of sets
   ============================================================ */
defineWidget('jaccard', node => {
  const wrap = el('div');
  node.appendChild(wrap);

  const UNIVERSE = ['machine', 'learning', 'is', 'amazing', 'deep', 'neural', 'data'];
  let inA = new Set(['machine', 'learning', 'is', 'amazing']);
  let inB = new Set(['deep', 'learning', 'is', 'amazing']);

  const cv = el('canvas');
  const left = el('div', {}, el('div', { class: 'pg-canvas-wrap' }, cv));
  const right = el('div', { class: 'pg-controls' });
  wrap.appendChild(el('div', { class: 'pg-split pg-split--wide-ctrl' }, left, right));

  const chipRow = el('div', { style: 'display:flex;flex-direction:column;gap:.45rem' });
  UNIVERSE.forEach(word => {
    const row = el('div', { style: 'display:flex;align-items:center;gap:.5rem;font-size:.86rem' },
      el('span', { style: 'flex:1;font-family:var(--font-mono);font-size:.82rem', text: word }),
      mkBtn(word, 'A'), mkBtn(word, 'B'));
    chipRow.appendChild(row);
  });
  function mkBtn(word, which) {
    const set = which === 'A' ? inA : inB;
    const b = el('button', {
      type: 'button',
      class: 'btn btn--ghost btn--sm',
      style: 'min-width:2.6em;justify-content:center',
      text: which,
    });
    const paint = () => {
      const on = (which === 'A' ? inA : inB).has(word);
      b.style.background = on ? (which === 'A' ? 'var(--brand-soft)' : 'var(--ok-soft)') : '';
      b.style.borderColor = on ? (which === 'A' ? 'var(--brand)' : 'var(--ok)') : '';
      b.style.color = on ? (which === 'A' ? 'var(--brand)' : 'var(--ok)') : 'var(--ink-faint)';
      b.style.fontWeight = on ? '700' : '500';
    };
    b.addEventListener('click', () => {
      const s = which === 'A' ? inA : inB;
      if (s.has(word)) s.delete(word); else s.add(word);
      paint(); refresh();
    });
    paint();
    return b;
  }

  const presets = el('div', { class: 'pg-actions' },
    button('Documents', () => {
      inA = new Set(['machine', 'learning', 'is', 'amazing']);
      inB = new Set(['deep', 'learning', 'is', 'amazing']);
      rebuild();
    }),
    button('Identical', () => { inB = new Set(inA); rebuild(); }),
    button('Disjoint', () => {
      inA = new Set(['machine', 'learning', 'is']);
      inB = new Set(['deep', 'neural', 'data']);
      rebuild();
    }),
  );
  const out = readout([['|A|', 0], ['|B|', 0], ['|A ∩ B|', 0], ['|A ∪ B|', 0], ['Jaccard J(A,B)', 0], ['Jaccard distance', 0]]);
  const st = status('');
  right.append(chipRow, presets, out.root, st.root);

  const plot = trackPlot(new Plot(cv, { xmin: 0, xmax: 10, ymin: 0, ymax: 6.4, aspect: 1.5, pad: 0 }));

  function rebuild() {
    chipRow.innerHTML = '';
    UNIVERSE.forEach(word => {
      chipRow.appendChild(el('div', { style: 'display:flex;align-items:center;gap:.5rem;font-size:.86rem' },
        el('span', { style: 'flex:1;font-family:var(--font-mono);font-size:.82rem', text: word }),
        mkBtn(word, 'A'), mkBtn(word, 'B')));
    });
    refresh();
  }

  function refresh() {
    const inter = [...inA].filter(w => inB.has(w));
    const union = [...new Set([...inA, ...inB])];
    const J = union.length ? inter.length / union.length : 0;
    out.set([
      String(inA.size), String(inB.size), String(inter.length), String(union.length),
      { html: `${inter.length}/${union.length} = ${fmt(J, 4)}`, cls: J > .8 ? 'is-ok' : J < .2 ? 'is-warn' : '' },
      fmt(1 - J, 4),
    ]);
    st.set(
      J === 1 ? `${ICON_OK}<span>Identical sets — J = 1.</span>`
      : J === 0 ? `${ICON_WARN}<span>No shared elements at all — J = 0.</span>`
      : `${ICON_INFO}<span><strong>${inter.length}</strong> shared out of <strong>${union.length}</strong> distinct elements between them.</span>`,
      J === 1 ? 'ok' : J === 0 ? 'warn' : 'info');
    plot.render();
  }

  plot.onDraw(p => {
    const cA = [3.7, 3.3], cB = [6.3, 3.3], R = 2.35;
    const inter = [...inA].filter(w => inB.has(w));
    const onlyA = [...inA].filter(w => !inB.has(w));
    const onlyB = [...inB].filter(w => !inA.has(w));

    const circle = (c, col) => {
      const pts = [];
      for (let i = 0; i <= 120; i++) {
        const t = i / 120 * Math.PI * 2;
        pts.push([c[0] + Math.cos(t) * R, c[1] + Math.sin(t) * R]);
      }
      p.path(pts, { color: col, lw: 2.4, close: true, fill: col === C.c1 ? C.fill : C.fill2 });
    };
    circle(cA, C.c1); circle(cB, C.c3);

    const place = (words, cx, startY, col) => {
      words.forEach((w, i) => {
        p.text([cx, startY - i * .48], w, {
          align: 'center', size: 11.5, weight: 600, color: col, halo: true,
        });
      });
    };
    place(onlyA, cA[0] - .95, 3.3 + (onlyA.length - 1) * .24, C.c1);
    place(onlyB, cB[0] + .95, 3.3 + (onlyB.length - 1) * .24, C.c3);
    place(inter, 5, 3.3 + (inter.length - 1) * .24, C.c2);

    p.text([cA[0] - 1.1, 5.95], 'A', { align: 'center', size: 15, weight: 750, color: C.c1 });
    p.text([cB[0] + 1.1, 5.95], 'B', { align: 'center', size: 15, weight: 750, color: C.c3 });
    p.text([5, .45], `J = |A ∩ B| / |A ∪ B| = ${inter.length} / ${[...new Set([...inA, ...inB])].length}`,
      { align: 'center', size: 13, weight: 700, color: C.ink });
  });

  rebuild();

  node.appendChild(note(
    `Jaccard cares only about <strong>membership</strong>, never about counts or order. Two documents sharing ` +
    `three words out of five distinct words score 3/5 = 0.6 regardless of how many times each word appeared. ` +
    `That is a feature when comparing sets of tags, users, or shingles — and a limitation when frequency ` +
    `matters, in which case cosine similarity over count vectors is the better instrument. Note ` +
    `<strong>1 − J</strong> (the Jaccard distance) <em>is</em> a true metric, which is not true of cosine.`
  ));
});

/* ============================================================
   5. The curse of dimensionality
   (restores a section that was commented out of the notes)
   ============================================================ */
defineWidget('curse-dimensionality', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: 0, xmax: 3.1, ymin: 0, ymax: 1.05, aspect: 1.6, equal: false, pad: 0,
  }));

  let nPoints = 500, showL1 = true, showL2 = true;
  const DIMS = [2, 3, 5, 10, 20, 50, 100, 200, 500, 1000];

  const nCtl = slider('Points per dimension', {
    min: 100, max: 1500, step: 50, value: 500, format: v => String(v),
    onInput: v => { nPoints = v; compute(); },
  });
  const toggles = el('div', { style: 'display:flex;flex-direction:column;gap:.5rem' },
    toggle('Euclidean (ℓ₂)', { value: true, onChange: v => { showL2 = v; plot.render(); } }).root,
    toggle('Manhattan (ℓ₁)', { value: true, onChange: v => { showL1 = v; plot.render(); } }).root,
  );
  const out = readout([['at d = 2', 0], ['at d = 10', 0], ['at d = 100', 0], ['at d = 1000', 0]]);
  const st = status('');
  right.append(nCtl.root, toggles, out.root, st.root);

  let series = { l1: [], l2: [] };
  function compute() {
    const r = ML.rng(2024);
    series = { l1: [], l2: [] };
    for (const d of DIMS) {
      const q = Array.from({ length: d }, () => r());
      let minL2 = Infinity, maxL2 = -Infinity, minL1 = Infinity, maxL1 = -Infinity;
      for (let i = 0; i < nPoints; i++) {
        let s2 = 0, s1 = 0;
        for (let j = 0; j < d; j++) {
          const diff = r() - q[j];
          s2 += diff * diff; s1 += Math.abs(diff);
        }
        const e = Math.sqrt(s2);
        if (e < minL2) minL2 = e; if (e > maxL2) maxL2 = e;
        if (s1 < minL1) minL1 = s1; if (s1 > maxL1) maxL1 = s1;
      }
      series.l2.push([Math.log10(d), minL2 / maxL2]);
      series.l1.push([Math.log10(d), minL1 / maxL1]);
    }
    const at = d => {
      const i = DIMS.indexOf(d);
      return `${fmt(series.l2[i][1], 3)}  /  ${fmt(series.l1[i][1], 3)}`;
    };
    out.set([at(2), at(10), at(100), at(1000)]);
    const last = series.l2[series.l2.length - 1][1];
    st.set(
      `${ICON_WARN}<span>At d = 1000 the nearest point is <strong>${fmt(last * 100, 1)}%</strong> as far as ` +
      `the farthest one. "Nearest neighbour" has stopped meaning anything.</span>`, 'warn');
    plot.render();
  }

  plot.onDraw(p => {
    p.grid(.25, { color: C.grid });
    p.line([0, 1], [3.1, 1], { color: C.c4, lw: 1.4, dash: [5, 4], alpha: .8 });
    // sits on the ratio = 1 rule, at the left where the curves are low
    p.text([0.06, 1], 'everything equidistant',
      { align: 'left', size: 10.5, color: C.c4, dy: -9, halo: true, haloWidth: 4 });

    if (showL2) {
      p.path(series.l2, { color: C.c1, lw: 2.8 });
      series.l2.forEach(q => p.dot(q, { r: 3.4, color: C.c1 }));
    }
    if (showL1) {
      p.path(series.l1, { color: C.c2, lw: 2.8, dash: [6, 4] });
      series.l1.forEach(q => p.dot(q, { r: 3.4, color: C.c2 }));
    }

    // log-scale ticks
    p.ctx.strokeStyle = C.axis; p.ctx.lineWidth = 1.3;
    p.ctx.beginPath();
    p.ctx.moveTo(p.X(0), p.Y(0)); p.ctx.lineTo(p.X(3.1), p.Y(0));
    p.ctx.moveTo(p.X(0), p.Y(0)); p.ctx.lineTo(p.X(0), p.Y(1.05));
    p.ctx.stroke();
    [1, 10, 100, 1000].forEach(d => {
      const x = Math.log10(d);
      p.text({ px: p.X(x), py: p.h - (p.reserveBottom || 16) - 3 }, String(d),
        { align: 'center', baseline: 'bottom', size: 11, color: C.muted, weight: 500 });
    });
    [0, .25, .5, .75, 1].forEach(v => {
      p.text({ px: 30, py: p.Y(v) }, fmt(v, 2),
        { align: 'right', size: 11, color: C.muted, halo: true, haloWidth: 3.5 });
    });
    p.xlabel('dimension d (log scale)', { size: 11 });
    // the curves climb to the top-right, so the key goes bottom-right
    p.legend([[C.c1, 'Euclidean'], [C.c2, 'Manhattan', [6, 4]]],
      { corner: 'br', margin: 30, title: 'nearest / farthest distance ratio' });
  });

  compute();

  node.appendChild(note(
    `Draw uniform random points, pick a query point, and take the ratio of the <em>nearest</em> distance to ` +
    `the <em>farthest</em>. In 2-D that ratio is small — the closest point is genuinely much closer than the ` +
    `farthest. By d = 1000 the ratio is close to 1: everything is about equally far away, so the very idea of ` +
    `a nearest neighbour dissolves. This is the <strong>curse of dimensionality</strong>, and it is why k-NN, ` +
    `k-means and RBF kernels degrade in high dimensions unless you reduce dimensionality first — which is ` +
    `exactly what <a href="#pca">PCA</a> and feature selection are for. Manhattan concentrates slightly more ` +
    `slowly than Euclidean, which is the practical argument for preferring lower-order norms when d is large.`
  ));
});
