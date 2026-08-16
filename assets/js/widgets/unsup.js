/* ============================================================
   widgets/unsup.js — Unsupervised learning: clustering & autoencoders
   ============================================================ */
import {
  Plot, C, el, slider, toggle, segmented, button,
  readout, status, defineWidget, canvasHost, trackPlot, clamp, fmt,
} from '../viz.js';
import * as ML from '../ml.js';
import * as CL from '../cluster.js';
import { pca } from '../ml.js';

function split(node, { hint, wide = false } = {}) {
  const left = el('div');
  const right = el('div', { class: 'pg-controls' });
  node.appendChild(el('div', { class: 'pg-split' + (wide ? ' pg-split--wide-ctrl' : '') }, left, right));
  const { canvas } = canvasHost(left, { hint });
  return { left, right, canvas };
}
const note = html => el('div', { class: 'pg-note', html });
const OK = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>`;
const INFO = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16.5v.01" stroke-linecap="round"/></svg>`;
const WARN = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 9v4M12 17v.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/></svg>`;

function withA(hex, a) {
  hex = (hex || '').trim();
  if (!hex.startsWith('#')) return hex;
  const n = hex.length === 4
    ? hex.slice(1).split('').map(c => parseInt(c + c, 16))
    : [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));
  return `rgba(${n.join(',')},${a})`;
}
const PAL = () => [C.c1, C.c2, C.c3, C.c4, C.c5];
/** Fit plot bounds around a point cloud with a margin. */
function fitBounds(p, pts, pad = .12) {
  const xs = pts.map(q => q[0]), ys = pts.map(q => q[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  const mx = (x1 - x0) * pad + 1e-6, my = (y1 - y0) * pad + 1e-6;
  p.o.xmin = x0 - mx; p.o.xmax = x1 + mx;
  p.o.ymin = y0 - my; p.o.ymax = y1 + my;
  p._computeScale();
}
/** Draw a cross marker (used for centroids). */
function cross(p, q, col, r = 8, lw = 3) {
  const [x, y] = p.toScreen(q);
  p.ctx.strokeStyle = col; p.ctx.lineWidth = lw; p.ctx.lineCap = 'round';
  p.ctx.beginPath();
  p.ctx.moveTo(x - r, y - r); p.ctx.lineTo(x + r, y + r);
  p.ctx.moveTo(x + r, y - r); p.ctx.lineTo(x - r, y + r);
  p.ctx.stroke();
}

/* ============================================================
   1. Lloyd's algorithm, step by step, on the notes' dataset
   ============================================================ */
defineWidget('kmeans-lab', node => {
  const wrap = el('div');
  node.appendChild(wrap);

  const X = [[1, 1], [1.5, 2], [3, 4], [5, 7], [3.5, 5], [4.5, 5], [3.5, 4.5], [2, 2], [8, 8], [9, 11]];
  const SEEDS = {
    notes: { label: "The notes' seeds", z: [[1, 1], [5, 7]] },
    bad: { label: 'A poor pair', z: [[8, 8], [9, 11]] },
    far: { label: 'Two far corners', z: [[1, 1], [9, 11]] },
  };
  let seedKey = 'notes', step = 0, frames = CL.lloydTrace(X, SEEDS.notes.z);

  const seedCtl = segmented(Object.entries(SEEDS).map(([k, v]) => ({ label: v.label, value: k })),
    { value: 'notes', label: 'Initial centroids', onChange: v => { seedKey = v; rebuild(); } });
  const stepCtl = slider('Iteration', {
    min: 0, max: 1, step: 1, value: 0, format: v => String(v + 1),
    onInput: v => { step = v; sync(); },
  });
  const showTable = toggle('Show the squared-distance table', { value: true, onChange: () => sync() });

  const cv = el('canvas');
  const left = el('div', {}, el('div', { class: 'pg-canvas-wrap' }, cv));
  const right = el('div', { class: 'pg-controls' }, seedCtl.root, stepCtl.root, showTable.root);
  const out = readout([['assignment', 0], ['WCSS before the update', 0], ['centroids after', 0], ['points that changed', 0]]);
  const st = status('');
  const tableHost = el('div', { class: 'table-scroll', style: 'margin-top:1rem' });
  right.append(out.root, st.root);
  wrap.appendChild(el('div', { class: 'pg-split pg-split--wide-ctrl' }, left, right));
  wrap.appendChild(tableHost);

  const plot = trackPlot(new Plot(cv, { xmin: 0, xmax: 10, ymin: 0, ymax: 12, aspect: 1.25, equal: true, pad: 0 }));

  function rebuild() {
    frames = CL.lloydTrace(X, SEEDS[seedKey].z);
    stepCtl.input.max = String(frames.length - 1);
    step = Math.min(step, frames.length - 1);
    stepCtl.set(step);
    sync();
  }

  function sync() {
    const f = frames[step];
    const C1 = f.labels.map((l, i) => (l === 0 ? i + 1 : 0)).filter(Boolean);
    const C2 = f.labels.map((l, i) => (l === 1 ? i + 1 : 0)).filter(Boolean);
    out.set([
      { html: `C₁ = {x${C1.join(', x')}}<br>C₂ = {x${C2.join(', x')}}` },
      { html: fmt(f.wcssBefore, 6), cls: 'is-ok' },
      { html: f.centers.map(c => `(${c[0].toFixed(4)}, ${c[1].toFixed(4)})`).join('<br>') },
      step === 0 ? '— (first assignment)' : String(f.moved),
    ]);
    const isLast = step === frames.length - 1;
    if (step === 0) {
      st.set(`${INFO}<span>Iteration 1 reproduces the notes exactly: WCSS = <strong>77.25</strong>, and x₃ sits at distance 13 from <em>both</em> centroids — a tie, broken toward the lower index.</span>`, 'info');
    } else if (step === 1 && seedKey === 'notes') {
      st.set(`${OK}<span>WCSS falls to <strong>63.821181</strong> and exactly one point — x₇ = (3.5, 4.5) — switches sides. That matches the notes.</span>`, 'ok');
    } else if (isLast) {
      st.set(`${OK}<span>Converged after ${frames.length} iterations at WCSS = <strong>${fmt(f.wcssAfter, 4)}</strong>. The notes stop at iteration 2; the algorithm keeps going.</span>`, 'ok');
    } else {
      st.set(`${INFO}<span>${f.moved} point${f.moved === 1 ? '' : 's'} changed cluster. WCSS is non-increasing at every half-step, which is why Lloyd's algorithm must terminate.</span>`, 'info');
    }

    tableHost.innerHTML = '';
    if (showTable.get()) {
      const rows = X.map((x, i) => {
        const d1 = CL.sqd(x, f.before[0]), d2 = CL.sqd(x, f.before[1]);
        const tie = Math.abs(d1 - d2) < 1e-12;
        return `<tr><td>x<sub>${i + 1}</sub> = (${x[0]}, ${x[1]})</td>` +
          `<td${f.labels[i] === 0 ? ' style="font-weight:700"' : ''}>${fmt(d1, 4)}</td>` +
          `<td${f.labels[i] === 1 ? ' style="font-weight:700"' : ''}>${fmt(d2, 4)}</td>` +
          `<td>C<sub>${f.labels[i] + 1}</sub>${tie ? ' <em>(tie → lower index)</em>' : ''}</td></tr>`;
      }).join('');
      tableHost.innerHTML =
        `<table><thead><tr><th>Point</th><th>d²(xᵢ, z₁)</th><th>d²(xᵢ, z₂)</th><th>Assigned</th></tr></thead>` +
        `<tbody>${rows}</tbody></table>`;
    }
    plot.render();
  }

  plot.onDraw(p => {
    const f = frames[step];
    const cols = PAL();
    p.grid(1, { color: C.grid });
    p.axes(); p.ticks(2);
    // assignment spokes
    X.forEach((x, i) => {
      const c = f.before[f.labels[i]];
      p.line(x, c, { color: withA(cols[f.labels[i]], .35), lw: 1.4 });
    });
    X.forEach((x, i) => {
      p.dot(x, { r: 6, color: cols[f.labels[i]] });
      p.text([x[0], x[1]], `${i + 1}`, { align: 'center', dy: -13, size: 9.5, color: C.muted });
    });
    f.before.forEach((c, j) => cross(p, c, cols[j], 8, 3));
    f.centers.forEach((c, j) => {
      cross(p, c, cols[j], 6, 2);
      p.ctx.setLineDash([4, 3]);
      p.line(f.before[j], c, { color: cols[j], lw: 1.8, dash: [4, 3] });
      p.ctx.setLineDash([]);
    });
    p.legend([[cols[0], 'cluster 1'], [cols[1], 'cluster 2']],
      { corner: 'tl', title: 'big × = centroid used, small × = updated' });
  });

  rebuild();

  node.appendChild(note(
    `This is the worked example from the notes, recomputed live. Step through the iterations and watch the ` +
    `two halves alternate: <strong>assign</strong> every point to its nearest centroid, then <strong>move</strong> ` +
    `each centroid to the mean of its members. Neither half can ever increase WCSS, and there are finitely ` +
    `many possible assignments, so the loop must stop. Note that the notes stop at iteration 2 — keep going ` +
    `and WCSS falls further to 43.12, which is in fact the <em>globally</em> optimal 2-clustering of these ten ` +
    `points. Try the other seedings to see that this is luck, not a guarantee.`
  ));
});

/* ============================================================
   2. TSS = WCSS + BCSS, and the elbow
   ============================================================ */
defineWidget('kmeans-anova', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, { xmin: 0, xmax: 1, ymin: 0, ymax: 1, aspect: 1.6, equal: false, pad: 0 }));

  const DATA = {
    notes: { label: "Notes' 10 points", pts: [[1, 1], [1.5, 2], [3, 4], [5, 7], [3.5, 5], [4.5, 5], [3.5, 4.5], [2, 2], [8, 8], [9, 11]] },
    blobs: { label: 'Three blobs', pts: ML.blobs({ n: 165, spread: .55, seed: 42 }).points },
  };
  let key = 'blobs', kMax = 8, kSel = 3;

  const dCtl = segmented(Object.entries(DATA).map(([k, v]) => ({ label: v.label, value: k })),
    { value: 'blobs', label: 'Dataset', onChange: v => { key = v; refresh(); } });
  const kCtl = slider('k (highlighted)', { min: 1, max: 8, step: 1, value: 3, format: v => String(v), onInput: v => { kSel = v; refresh(); } });
  const out = readout([['TSS (fixed)', 0], ['WCSS at this k', 0], ['BCSS at this k', 0], ['WCSS + BCSS', 0], ['variance explained', 0]]);
  const st = status('');
  right.append(dCtl.root, kCtl.root, out.root, st.root);

  let curve = [];
  function refresh() {
    const pts = DATA[key].pts;
    kMax = Math.min(8, pts.length - 1);
    kCtl.input.max = String(kMax);
    if (kSel > kMax) { kSel = kMax; kCtl.set(kSel); }
    curve = [];
    for (let k = 1; k <= kMax; k++) {
      const seeds = CL.kmeansppSeeds(pts, k, 11).centers;
      const res = CL.lloydFinal(pts, seeds);
      curve.push({ k, ...CL.anova(pts, res.labels, res.centers) });
    }
    const cur = curve[kSel - 1];
    out.set([
      fmt(cur.TSS, 4),
      { html: fmt(cur.WCSS, 4), cls: 'is-ok' },
      fmt(cur.BCSS, 4),
      { html: fmt(cur.WCSS + cur.BCSS, 4), cls: Math.abs(cur.TSS - cur.WCSS - cur.BCSS) < 1e-9 ? 'is-ok' : 'is-warn' },
      `${fmt(100 * cur.BCSS / cur.TSS, 1)}%`,
    ]);
    st.set(
      Math.abs(cur.TSS - cur.WCSS - cur.BCSS) < 1e-9
        ? `${OK}<span>The identity holds exactly: TSS is fixed by the data alone, so every unit of WCSS you remove reappears as BCSS. <strong>Minimising within-cluster scatter is the same thing as maximising between-cluster separation.</strong></span>`
        : `${WARN}<span>Identity violated — that should be impossible.</span>`,
      'ok');
    plot.render();
  }

  plot.onDraw(p => {
    // band below zero for the k labels, band above for the TSS rule and legend
    p.o.xmin = .3; p.o.xmax = kMax + .7;
    p.o.ymin = -curve[0].TSS * .13; p.o.ymax = curve[0].TSS * 1.30;
    p._computeScale();
    p.grid(curve[0].TSS / 6, { color: C.grid });
    const bw = .62;
    curve.forEach(c => {
      const x0 = p.X(c.k - bw / 2), w = p.px(bw);
      const hi = c.k === kSel ? 1 : .45;
      // stacked: WCSS at the bottom, BCSS on top, summing to TSS
      p.ctx.globalAlpha = hi;
      p.ctx.fillStyle = C.c4;
      p.ctx.fillRect(x0, p.Y(c.WCSS), w, p.Y(0) - p.Y(c.WCSS));
      p.ctx.fillStyle = C.c3;
      p.ctx.fillRect(x0, p.Y(c.TSS), w, p.Y(c.WCSS) - p.Y(c.TSS));
      p.ctx.globalAlpha = 1;
      if (c.k === kSel) {
        p.ctx.strokeStyle = C.ink; p.ctx.lineWidth = 1.6;
        p.ctx.strokeRect(x0, p.Y(c.TSS), w, p.Y(0) - p.Y(c.TSS));
      }
      p.text({ px: p.X(c.k), py: p.h - (p.reserveBottom || 16) - 4 }, String(c.k),
        { align: 'center', baseline: 'bottom', size: 11,
          color: c.k === kSel ? C.ink : C.muted, weight: c.k === kSel ? 700 : 500 });
    });
    p.line([.3, curve[0].TSS], [kMax + .7, curve[0].TSS], { color: C.ink, lw: 1.6, dash: [6, 4] });
    p.badge([kMax + .6, curve[0].TSS], `TSS = ${fmt(curve[0].TSS, 1)}`, { color: C.ink, align: 'right', dy: -14 });
    p.axes(); p.ticks(curve[0].TSS / 4);
    p.legend([[C.c4, 'WCSS — within clusters'], [C.c3, 'BCSS — between clusters']], { corner: 'tl' });
    p.xlabel('number of clusters k', { size: 10.5 });
  });

  refresh();

  node.appendChild(note(
    `Each bar is the <em>same height</em> — that is the point. The ANOVA identity ` +
    `<span class="u-mono">TSS = WCSS + BCSS</span> says total scatter about the grand mean splits cleanly ` +
    `into scatter within clusters and scatter between them, and TSS does not depend on the clustering at all. ` +
    `So driving WCSS down is <em>identical</em> to pushing the centroids apart. The cross-terms vanish ` +
    `precisely because each centroid is the mean of its members. Watch where the bars stop shrinking quickly — ` +
    `that knee is the "elbow" heuristic for choosing k.`
  ));
});

/* ============================================================
   3. Initialisation sensitivity and k-means++
   ============================================================ */
defineWidget('kmeans-init', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, { xmin: 0, xmax: 1, ymin: 0, ymax: 1, aspect: 1.5, equal: true, pad: 0 }));

  const pts = ML.blobs({ n: 201, spread: .42, seed: 8 }).points;
  let mode = 'random', seed = 1, k = 3;

  const mCtl = segmented([{ label: 'Random seeding', value: 'random' }, { label: 'k-means++', value: 'pp' }],
    { value: 'random', label: 'Initialisation', onChange: v => { mode = v; refresh(); } });
  const sCtl = slider('Random draw', { min: 1, max: 40, step: 1, value: 1, format: v => `#${v}`, onInput: v => { seed = v; refresh(); } });
  const kCtl = slider('k', { min: 2, max: 6, step: 1, value: 3, format: v => String(v), onInput: v => { k = v; refresh(); } });
  const out = readout([['final WCSS', 0], ['iterations', 0], ['best over 40 draws', 0], ['worst over 40 draws', 0], ['spread', 0]]);
  const st = status('');
  right.append(mCtl.root, sCtl.root, kCtl.root, out.root, st.root);

  let seeds = [], res = null, allW = [], bestW = 0, worstW = 0;
  function refresh() {
    const pick = mode === 'pp' ? CL.kmeansppSeeds(pts, k, seed) : CL.randomSeeds(pts, k, seed);
    seeds = pick.centers;
    res = CL.lloydFinal(pts, seeds);
    allW = [];
    for (let s = 1; s <= 40; s++) {
      const q = mode === 'pp' ? CL.kmeansppSeeds(pts, k, s) : CL.randomSeeds(pts, k, s);
      allW.push(CL.lloydFinal(pts, q.centers).wcss);
    }
    bestW = Math.min(...allW); worstW = Math.max(...allW);
    out.set([
      { html: fmt(res.wcss, 3), cls: res.wcss <= bestW + 1e-6 ? 'is-ok' : '' },
      String(res.iters), fmt(bestW, 3), fmt(worstW, 3),
      { html: `${fmt(100 * (worstW - bestW) / bestW, 1)}%`, cls: (worstW - bestW) / bestW > .1 ? 'is-warn' : 'is-ok' },
    ]);
    st.set(
      res.wcss > bestW * 1.02
        ? `${WARN}<span>This draw landed in a <strong>bad local optimum</strong> — WCSS is ${fmt(100 * (res.wcss / bestW - 1), 1)}% above the best seen. The objective is nonconvex, so the seeds decide which basin you fall into.</span>`
        : `${OK}<span>This draw found the best clustering seen over 40 restarts. ${mode === 'pp' ? 'k-means++ gets here far more often than uniform seeding.' : 'Uniform seeding gets lucky sometimes — scroll the draw slider to see how often it does not.'}</span>`,
      res.wcss > bestW * 1.02 ? 'warn' : 'ok');
    plot.render();
  }

  plot.onDraw(p => {
    fitBounds(p, pts);
    const cols = PAL();
    p.grid(1, { color: C.grid });
    pts.forEach((q, i) => p.dot(q, { r: 3.6, color: withA(cols[res.labels[i]], .75) }));
    seeds.forEach(s => {
      p.ctx.strokeStyle = C.ink; p.ctx.lineWidth = 1.6;
      const [sx, sy] = p.toScreen(s);
      p.ctx.beginPath(); p.ctx.arc(sx, sy, 9, 0, Math.PI * 2); p.ctx.stroke();
    });
    res.centers.forEach((c, j) => cross(p, c, cols[j], 8, 3));
    p.axes(); p.ticks(2);
    p.legend([[C.ink, 'open circle = initial seed'], [C.c1, '× = final centroid']],
      { corner: 'tl', title: `WCSS = ${fmt(res.wcss, 2)}   (best of 40: ${fmt(bestW, 2)})` });
  });

  refresh();

  node.appendChild(note(
    `Drag the draw slider and watch the final answer change. The k-means objective is <strong>nonconvex</strong>, ` +
    `so Lloyd's algorithm only ever finds a local optimum, and which one depends entirely on where you start. ` +
    `<strong>k-means++</strong> replaces uniform seeding with a rule that picks each new centre with probability ` +
    `proportional to D(x)² — the squared distance to the nearest centre already chosen — so seeds spread out ` +
    `across the data instead of clumping. Switch between the two and compare the spread between best and worst ` +
    `over 40 draws: that gap is the variance k-means++ is designed to remove.`
  ));
});

/* ============================================================
   4. Where k-means fails
   ============================================================ */
defineWidget('kmeans-limits', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, { xmin: 0, xmax: 1, ymin: 0, ymax: 1, aspect: 1.5, equal: true, pad: 0 }));

  const SETS = {
    moons: { label: 'Two moons', k: 2, make: () => CL.moons({ n: 160 }) },
    circles: { label: 'Concentric circles', k: 2, make: () => CL.circles({ n: 160 }) },
    aniso: { label: 'Elongated blobs', k: 3, make: () => CL.anisotropic({ n: 180 }) },
    sizes: { label: 'Unequal spread', k: 2, make: () => CL.unequalBlobs({ n: 180 }) },
  };
  let key = 'moons', showTruth = false;
  const sCtl = segmented(Object.entries(SETS).map(([k, v]) => ({ label: v.label, value: k })),
    { value: 'moons', label: 'Dataset', onChange: v => { key = v; refresh(); } });
  const tCtl = toggle('Show the structure k-means should have found', { value: false, onChange: v => { showTruth = v; plot.render(); } });
  const out = readout([['k', 0], ['agreement with the true groups', 0], ['WCSS found', 0], ['WCSS of the true grouping', 0]]);
  const st = status('');
  right.append(sCtl.root, tCtl.root, out.root, st.root);

  let ds = null, res = null, agree = 0;
  function refresh() {
    ds = SETS[key].make();
    const k = SETS[key].k;
    const seeds = CL.kmeansppSeeds(ds.points, k, 5).centers;
    res = CL.lloydFinal(ds.points, seeds);
    // best label permutation for k = 2, direct match for k = 3
    let m = 0;
    if (k === 2) {
      m = Math.max(res.labels.filter((l, i) => l === ds.truth[i]).length,
        res.labels.filter((l, i) => l !== ds.truth[i]).length);
    } else {
      const perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
      m = Math.max(...perms.map(pm => res.labels.filter((l, i) => pm[l] === ds.truth[i]).length));
    }
    agree = m / ds.points.length;
    // WCSS of the ground-truth grouping, for comparison
    const tc = Array.from({ length: k }, (_, j) => {
      const mem = ds.points.filter((_, i) => ds.truth[i] === j);
      return [mem.reduce((s, q) => s + q[0], 0) / mem.length, mem.reduce((s, q) => s + q[1], 0) / mem.length];
    });
    const twcss = ds.points.reduce((s, q, i) => s + CL.sqd(q, tc[ds.truth[i]]), 0);
    out.set([
      String(k),
      { html: `${fmt(agree * 100, 1)}%`, cls: agree > .9 ? 'is-ok' : 'is-warn' },
      fmt(res.wcss, 3),
      { html: fmt(twcss, 3), cls: twcss > res.wcss ? 'is-warn' : '' },
    ]);
    st.set(
      agree > .9
        ? `${OK}<span>k-means handles this one — the groups really are compact and roughly spherical.</span>`
        : `${WARN}<span><strong>k-means found a lower WCSS than the true grouping</strong> (${fmt(res.wcss, 1)} against ${fmt(twcss, 1)}). ` +
          `So this is not an optimisation failure — the algorithm did its job. The <em>objective</em> is what disagrees with the structure you wanted.</span>`,
      agree > .9 ? 'ok' : 'warn');
    plot.render();
  }

  plot.onDraw(p => {
    fitBounds(p, ds.points);
    const cols = PAL();
    p.grid(.5, { color: C.grid });
    const lab = showTruth ? ds.truth : res.labels;
    ds.points.forEach((q, i) => p.dot(q, { r: 3.8, color: withA(cols[lab[i]], .8) }));
    if (!showTruth) {
      res.centers.forEach((c, j) => cross(p, c, cols[j], 9, 3.2));
      // perpendicular bisector between the first two centroids
      if (res.centers.length === 2) {
        const [a, b] = res.centers;
        const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
        const dx = b[0] - a[0], dy = b[1] - a[1];
        const L = Math.hypot(dx, dy) || 1;
        const t = 40;
        p.line([mx - (-dy / L) * t, my - (dx / L) * t], [mx + (-dy / L) * t, my + (dx / L) * t],
          { color: C.ink, lw: 2, dash: [7, 5] });
      }
    }
    p.axes(); p.ticks(1);
    p.legend([[C.ink, showTruth ? 'the grouping you wanted' : 'straight decision boundary']],
      { corner: 'tl', title: showTruth ? 'true structure' : `k-means result — ${fmt(agree * 100, 0)}% agreement` });
  });

  refresh();

  node.appendChild(note(
    `Every failure here has the same cause. The assignment step sends each point to its nearest centroid, so ` +
    `the boundary between any two clusters is the <strong>perpendicular bisector</strong> of the segment joining ` +
    `them — a straight line, always. k-means can therefore only carve space into convex cells. Rings, crescents ` +
    `and sheared blobs have no such description. Toggle the true structure and check the readout: on the failing ` +
    `datasets k-means achieves a <em>lower</em> WCSS than the grouping you actually wanted, which proves the ` +
    `problem is the objective rather than the optimiser.`
  ));
});

/* ============================================================
   5. Means versus medoids
   ============================================================ */
defineWidget('kmedoids-demo', node => {
  const { right, canvas } = split(node, { hint: 'Drag the orange point', wide: true });
  const plot = trackPlot(new Plot(canvas, { xmin: -6, xmax: 14, ymin: -5, ymax: 11, aspect: 1.4, equal: true, pad: 0 }));

  const base = [];
  const r = ML.rng(17);
  for (let i = 0; i < 14; i++) base.push([ML.gauss(r) * 1.05, ML.gauss(r) * 1.05]);
  for (let i = 0; i < 14; i++) base.push([6 + ML.gauss(r) * 1.05, 5 + ML.gauss(r) * 1.05]);
  let outlier = [12, 9];
  let metric = 'euclidean';

  const mCtl = segmented([
    { label: 'Euclidean', value: 'euclidean' },
    { label: 'Manhattan', value: 'manhattan' },
    { label: 'Cosine', value: 'cosine' },
  ], { value: 'euclidean', label: 'Dissimilarity', onChange: v => { metric = v; refresh(); } });
  const acts = el('div', { class: 'pg-actions' },
    button('Outlier nearby', () => { outlier = [7, 6]; refresh(); }),
    button('Outlier far', () => { outlier = [12, 9]; refresh(); }),
    button('Outlier extreme', () => { outlier = [13.5, 10.5]; refresh(); }));
  const out = readout([['mean of cluster 2', 0], ['medoid of cluster 2', 0], ['medoid is a real point', 0], ['mean drift from the bulk', 0]]);
  const st = status('');
  right.append(mCtl.root, acts, out.root, st.root);

  let pts = [], kmRes = null, kmedRes = null;
  function refresh() {
    pts = [...base, outlier];
    kmRes = CL.lloydFinal(pts, [[0, 0], [6, 5]]);
    kmedRes = CL.kmedoids(pts, 2, { seed: 3, metric });
    // the cluster containing the outlier
    const oc = kmRes.labels[pts.length - 1];
    const mem = pts.filter((_, i) => kmRes.labels[i] === oc);
    const bulk = mem.filter(q => q !== outlier);
    const bulkMean = [bulk.reduce((s, q) => s + q[0], 0) / bulk.length, bulk.reduce((s, q) => s + q[1], 0) / bulk.length];
    const mc = kmRes.centers[oc];
    const mo = kmedRes.labels[pts.length - 1];
    const med = pts[kmedRes.medoids[mo]];
    out.set([
      `(${fmt(mc[0], 3)}, ${fmt(mc[1], 3)})`,
      `(${fmt(med[0], 3)}, ${fmt(med[1], 3)})`,
      { html: 'yes — by construction', cls: 'is-ok' },
      { html: fmt(CL.euclid(mc, bulkMean), 3), cls: CL.euclid(mc, bulkMean) > .6 ? 'is-warn' : 'is-ok' },
    ]);
    st.set(
      CL.euclid(mc, bulkMean) > .6
        ? `${WARN}<span>The outlier has dragged the <strong>mean</strong> ${fmt(CL.euclid(mc, bulkMean), 2)} units off the bulk of its cluster. The <strong>medoid</strong> cannot move like that — it has to be one of the actual data points.</span>`
        : `${INFO}<span>With the outlier close by, mean and medoid nearly coincide. Push it further out and watch them separate.</span>`,
      CL.euclid(mc, bulkMean) > .6 ? 'warn' : 'info');
    plot.render();
  }

  plot.onDraw(p => {
    const cols = PAL();
    p.grid(2, { color: C.grid });
    pts.forEach((q, i) => {
      const isOut = i === pts.length - 1;
      p.dot(q, { r: isOut ? 7 : 4.4, color: isOut ? C.c2 : withA(cols[kmRes.labels[i]], .7) });
    });
    kmRes.centers.forEach((c, j) => cross(p, c, cols[j], 9, 3.2));
    kmedRes.medoids.forEach((mi, j) => {
      const [sx, sy] = p.toScreen(pts[mi]);
      p.ctx.strokeStyle = cols[j]; p.ctx.lineWidth = 3;
      p.ctx.beginPath(); p.ctx.arc(sx, sy, 10, 0, Math.PI * 2); p.ctx.stroke();
    });
    p.axes(); p.ticks(2);
    p.legend([[C.c1, '× = mean (k-means)'], [C.c2, '○ = medoid (k-medoids)']],
      { corner: 'br', title: 'the orange point is the outlier' });
  });

  // dragging the outlier
  let dragging = false;
  canvas.style.touchAction = 'none';
  canvas.addEventListener('pointerdown', e => {
    if (CL.euclid(plot.eventWorld(e), outlier) < 1.4) {
      dragging = true;
      canvas.setPointerCapture(e.pointerId);
    }
  });
  canvas.addEventListener('pointermove', e => {
    if (!dragging) return;
    const w = plot.eventWorld(e);
    outlier = [clamp(w[0], -5, 13.5), clamp(w[1], -4, 10.5)];
    refresh();
  });
  canvas.addEventListener('pointerup', e => {
    dragging = false;
    if (canvas.hasPointerCapture?.(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointercancel', () => { dragging = false; });

  refresh();

  node.appendChild(note(
    `Drag the orange point away from the cluster. The <strong>mean</strong> is a weighted average, so it slides ` +
    `smoothly toward the outlier and can end up somewhere no data lives. The <strong>medoid</strong> is required ` +
    `to be an actual observation, so the worst an outlier can do is fail to be chosen. That robustness is one of ` +
    `two reasons to prefer medoids; the other is that the mean needs coordinates to average, while a medoid only ` +
    `needs a dissimilarity — try the cosine setting, which is defined on angles alone.`
  ));
});

/* ============================================================
   6. Kernel k-means and spectral clustering
   ============================================================ */
defineWidget('kernel-kmeans', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, { xmin: 0, xmax: 1, ymin: 0, ymax: 1, aspect: 1.5, equal: true, pad: 0 }));

  const SETS = {
    circles: { label: 'Concentric circles', make: () => CL.circles({ n: 110 }) },
    moons: { label: 'Two moons', make: () => CL.moons({ n: 120 }) },
  };
  let key = 'circles', method = 'kernel', gamma = 20, seed = 1;

  const dCtl = segmented(Object.entries(SETS).map(([k, v]) => ({ label: v.label, value: k })),
    { value: 'circles', label: 'Dataset', onChange: v => { key = v; rebuild(); } });
  const mCtl = segmented([
    { label: 'Plain k-means', value: 'plain' },
    { label: 'Kernel k-means', value: 'kernel' },
    { label: 'Spectral', value: 'spectral' },
  ], { value: 'kernel', label: 'Method', onChange: v => { method = v; refresh(); } });
  const gCtl = slider('RBF γ', { min: 1, max: 100, step: 1, value: 20, format: v => String(v), onInput: v => { gamma = v; refresh(); } });
  const sCtl = slider('Random restart', { min: 1, max: 24, step: 1, value: 1, format: v => `#${v}`, onInput: v => { seed = v; refresh(); } });
  const out = readout([['agreement with the rings', 0], ['method', 0], ['γ', 0], ['runs reaching 95% (of 24)', 0]]);
  const st = status('');
  right.append(dCtl.root, mCtl.root, gCtl.root, sCtl.root, out.root, st.root);

  let ds = null, labels = [], agree = 0, rate = 0;
  const cache = new Map();

  const score = lab => Math.max(
    lab.filter((l, i) => l === ds.truth[i]).length,
    lab.filter((l, i) => l !== ds.truth[i]).length) / ds.points.length;

  function run(m, g, s) {
    const ck = `${key}|${m}|${g}|${s}`;
    if (cache.has(ck)) return cache.get(ck);
    let lab;
    if (m === 'plain') lab = CL.lloydFinal(ds.points, CL.kmeansppSeeds(ds.points, 2, s).centers).labels;
    else if (m === 'kernel') lab = CL.kernelKmeans(ds.points, 2, { gamma: g, seed: s, restarts: 4 }).labels;
    else lab = CL.spectral(ds.points, 2, { gamma: g, seed: s, restarts: 4 }).labels;
    cache.set(ck, lab);
    return lab;
  }
  function rebuild() { ds = SETS[key].make(); cache.clear(); refresh(); }

  function refresh() {
    labels = run(method, gamma, seed);
    agree = score(labels);
    let hit = 0;
    for (let s = 1; s <= 24; s++) if (score(run(method, gamma, s)) >= .95) hit++;
    rate = hit / 24;
    out.set([
      { html: `${fmt(agree * 100, 1)}%`, cls: agree > .9 ? 'is-ok' : 'is-warn' },
      method === 'plain' ? 'k-means' : method === 'kernel' ? 'kernel k-means (RBF)' : 'spectral (RBF affinity)',
      method === 'plain' ? 'n/a' : String(gamma),
      { html: `${hit} / 24 (${fmt(rate * 100, 0)}%)`, cls: rate > .9 ? 'is-ok' : rate > 0 ? '' : 'is-warn' },
    ]);
    if (method === 'plain') {
      st.set(`${WARN}<span>Plain k-means cannot do this at all — its boundaries are straight lines.</span>`, 'warn');
    } else if (method === 'spectral') {
      st.set(rate > .9
        ? `${OK}<span><strong>Spectral clustering succeeds on every restart</strong> at this γ. It embeds using the smallest eigenvectors of the normalised Laplacian, so it keys on <em>connectivity</em> — a ring only has to be connected, not compact.</span>`
        : `${INFO}<span>At γ = ${gamma} the affinity graph is too diffuse (or too fragmented) to separate the groups. Push γ up to about 20 and it locks on.</span>`,
        rate > .9 ? 'ok' : 'info');
    } else {
      st.set(rate >= .9
        ? `${OK}<span>Kernel k-means is working reliably at this γ.</span>`
        : rate > 0
          ? `${WARN}<span><strong>Fragile.</strong> Only ${hit} of 24 restarts recover the true groups at γ = ${gamma}. Kernel k-means still wants clusters that are <em>compact in feature space</em>, and a ring is a curve, not a blob.</span>`
          : `${WARN}<span>No restart out of 24 recovers the true groups at γ = ${gamma}. Try γ near 40 for the rings, or near 8 for the moons — and compare against spectral.</span>`,
        rate >= .9 ? 'ok' : 'warn');
    }
    plot.render();
  }

  plot.onDraw(p => {
    fitBounds(p, ds.points);
    const cols = PAL();
    p.grid(.5, { color: C.grid });
    ds.points.forEach((q, i) => p.dot(q, { r: 4.2, color: withA(cols[labels[i]], .85) }));
    p.axes(); p.ticks(.5);
    p.legend([[cols[0], 'cluster 1'], [cols[1], 'cluster 2']],
      { corner: 'tl', title: `${fmt(agree * 100, 0)}% agreement · ${fmt(rate * 100, 0)}% of restarts succeed` });
  });

  rebuild();

  node.appendChild(note(
    `The kernel trick from the SVM chapter applies here too: expand ` +
    `<span class="u-mono">‖φ(x) − μⱼ‖²</span> and every term is an inner product, so the whole assignment step ` +
    `runs on the kernel matrix without ever forming φ(x). But watch the success-rate readout as you move γ. ` +
    `<strong>Kernel k-means recovers the rings only in a narrow band of γ, and even there only on a minority of ` +
    `restarts</strong> — because it still seeks clusters that are compact in feature space, and two points on ` +
    `opposite sides of the same ring are nearly orthogonal under an RBF kernel. <strong>Spectral clustering</strong>, ` +
    `which the notes describe as a relaxation of kernel k-means, succeeds on every restart for any γ ≳ 20, because ` +
    `it only needs each ring to be <em>connected</em>. Switch between the two methods and compare.`
  ));
});

/* ============================================================
   7. The DBSCAN toy example
   ============================================================ */
defineWidget('dbscan-toy', node => {
  const wrap = el('div');
  node.appendChild(wrap);

  const P = [[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]];
  const NM = ['A', 'B', 'C', 'D', 'E', 'F'];
  let eps = 2, minPts = 3, showF = false;

  const eCtl = slider('ε (radius)', { min: 1, max: 12, step: .5, value: 2, onInput: v => { eps = v; refresh(); } });
  const mCtl = slider('minPts (other points needed)', { min: 1, max: 5, step: 1, value: 3, format: v => String(v), onInput: v => { minPts = v; refresh(); } });
  const fCtl = toggle('Include the far outlier F in the view', { value: false, onChange: v => { showF = v; plot.render(); } });
  const cases = el('div', { class: 'pg-actions' },
    button('Case 1: ε=2, minPts=3', () => { eps = 2; minPts = 3; eCtl.set(2); mCtl.set(3); refresh(); }),
    button('Case 2: ε=2, minPts=2', () => { eps = 2; minPts = 2; eCtl.set(2); mCtl.set(2); refresh(); }),
    button('Case 3: ε=8, minPts=3', () => { eps = 8; minPts = 3; eCtl.set(8); mCtl.set(3); refresh(); }));

  const cv = el('canvas');
  const left = el('div', {}, el('div', { class: 'pg-canvas-wrap' }, cv));
  const right = el('div', { class: 'pg-controls' }, eCtl.root, mCtl.root, fCtl.root, cases);
  const out = readout([['core points', 0], ['border points', 0], ['noise', 0], ['clusters found', 0]]);
  const st = status('');
  const tableHost = el('div', { class: 'table-scroll', style: 'margin-top:1rem' });
  right.append(out.root, st.root);
  wrap.appendChild(el('div', { class: 'pg-split pg-split--wide-ctrl' }, left, right));
  wrap.appendChild(tableHost);

  const plot = trackPlot(new Plot(cv, { xmin: -2, xmax: 14, ymin: -2, ymax: 12, aspect: 1.35, equal: true, pad: 0 }));

  let res = null;
  function refresh() {
    res = CL.dbscan(P, eps, minPts);
    const core = NM.filter((_, i) => res.kind[i] === 'core');
    const border = NM.filter((_, i) => res.kind[i] === 'border');
    const noise = NM.filter((_, i) => res.kind[i] === 'noise');
    out.set([
      core.length ? core.join(', ') : '—',
      border.length ? border.join(', ') : '—',
      noise.length ? noise.join(', ') : '—',
      { html: String(res.clusters), cls: res.clusters ? 'is-ok' : 'is-warn' },
    ]);
    const key = `${eps}/${minPts}`;
    if (key === '2/3') {
      st.set(`${WARN}<span><strong>Case 1.</strong> No point has 3 neighbours within ε = 2, so nothing is core, nothing can seed a cluster, and <em>every</em> point is noise.</span>`, 'warn');
    } else if (key === '2/2') {
      st.set(`${OK}<span><strong>Case 2.</strong> Lowering minPts to 2 makes A, B and C core. They form one cluster; D and E have only one neighbour each, so they stay noise along with F.</span>`, 'ok');
    } else if (key === '8/3') {
      st.set(`${OK}<span><strong>Case 3.</strong> A wider radius makes B, C and D core. A and E are not core but sit inside a core neighbourhood, so they join as <em>border</em> points. The two groups merge into one cluster; only F is noise.</span>`, 'ok');
    } else {
      st.set(`${INFO}<span>${res.clusters} cluster${res.clusters === 1 ? '' : 's'}, ${res.kind.filter(k => k === 'noise').length} noise point${res.kind.filter(k => k === 'noise').length === 1 ? '' : 's'}. Border points join a cluster but never extend it.</span>`, 'info');
    }
    const kd = CL.kDistances(P, minPts);
    tableHost.innerHTML =
      `<table><thead><tr><th>Point</th><th>N<sub>ε</sub>(p) — excluding p itself</th><th>|N<sub>ε</sub>(p)|</th>` +
      `<th>${minPts}-th nearest</th><th>Label</th></tr></thead><tbody>` +
      NM.map((nm, i) => {
        const nb = res.nbrs[i].map(j => NM[j]);
        const badge = res.kind[i] === 'core' ? 'is-ok' : res.kind[i] === 'noise' ? 'is-warn' : '';
        return `<tr><td><strong>${nm}</strong> (${P[i][0]}, ${P[i][1]})</td>` +
          `<td>{${nb.join(', ')}}</td><td>${nb.length}</td>` +
          `<td>${Number.isFinite(kd[i]) ? fmt(kd[i], 2) : '∞'} ${kd[i] <= eps ? '≤' : '>'} ${eps}</td>` +
          `<td class="${badge}">${res.kind[i]}${res.labels[i] >= 0 ? ` — C${res.labels[i] + 1}` : ''}</td></tr>`;
      }).join('') + '</tbody></table>';
    plot.render();
  }

  plot.onDraw(p => {
    const view = showF ? P : P.slice(0, 5);
    fitBounds(p, view, .25);
    p.grid(2, { color: C.grid });
    const cols = PAL();
    P.forEach((q, i) => {
      if (!showF && i === 5) return;
      // eps disc
      p.ctx.beginPath();
      const [sx, sy] = p.toScreen(q);
      const rr = Math.abs(p.px(eps));
      p.ctx.arc(sx, sy, rr, 0, Math.PI * 2);
      p.ctx.fillStyle = withA(res.kind[i] === 'core' ? C.c1 : C.muted, .08);
      p.ctx.fill();
      p.ctx.strokeStyle = withA(res.kind[i] === 'core' ? C.c1 : C.muted, .45);
      p.ctx.lineWidth = 1.2; p.ctx.setLineDash([4, 4]); p.ctx.stroke(); p.ctx.setLineDash([]);
    });
    P.forEach((q, i) => {
      if (!showF && i === 5) return;
      const col = res.labels[i] >= 0 ? cols[res.labels[i]] : C.muted;
      if (res.kind[i] === 'core') {
        p.dot(q, { r: 8, color: col });
      } else if (res.kind[i] === 'border') {
        const [sx, sy] = p.toScreen(q);
        p.ctx.fillStyle = withA(col, .35); p.ctx.beginPath(); p.ctx.arc(sx, sy, 7, 0, Math.PI * 2); p.ctx.fill();
        p.ctx.strokeStyle = col; p.ctx.lineWidth = 2.4; p.ctx.stroke();
      } else {
        const [sx, sy] = p.toScreen(q);
        p.ctx.strokeStyle = C.muted; p.ctx.lineWidth = 2;
        p.ctx.beginPath();
        p.ctx.moveTo(sx - 5, sy - 5); p.ctx.lineTo(sx + 5, sy + 5);
        p.ctx.moveTo(sx + 5, sy - 5); p.ctx.lineTo(sx - 5, sy + 5);
        p.ctx.stroke();
      }
      p.text(q, NM[i], { align: 'center', dy: -15, size: 12, weight: 700, color: C.ink });
    });
    p.axes(); p.ticks(2);
    p.legend([[C.c1, 'core — filled'], [C.c2, 'border — ring'], [C.muted, 'noise — ×']],
      { corner: 'tr', title: `ε = ${eps}, minPts = ${minPts}` });
  });

  refresh();

  node.appendChild(note(
    `The six points from the notes, with every neighbourhood recomputed live. The three preset buttons ` +
    `reproduce the three cases exactly. Notice how differently the two knobs behave: <strong>minPts</strong> ` +
    `changes who qualifies as core without moving a single circle, while <strong>ε</strong> grows the circles ` +
    `and can merge separate groups into one. The asymmetry between core and border is the subtle part — a border ` +
    `point is swept into a cluster because it lies inside a core point's radius, but it is not dense enough to ` +
    `extend the cluster any further. F is 74 units from its nearest neighbour, so no setting short of absurd ` +
    `will ever recruit it.`
  ));
});

/* ============================================================
   8. DBSCAN on a real scene, with the k-distance plot
   ============================================================ */
defineWidget('dbscan-lab', node => {
  const wrap = el('div');
  node.appendChild(wrap);

  const pts = CL.densityScene();
  let eps = .42, minPts = 4;

  const eCtl = slider('ε', { min: .1, max: 1.4, step: .02, value: .42, onInput: v => { eps = v; refresh(); } });
  const mCtl = slider('minPts', { min: 2, max: 12, step: 1, value: 4, format: v => String(v), onInput: v => { minPts = v; refresh(); } });
  const acts = el('div', { class: 'pg-actions' },
    button('Too small (ε=0.16)', () => { eps = .16; eCtl.set(.16); refresh(); }),
    button('Just right (ε=0.42)', () => { eps = .42; eCtl.set(.42); refresh(); }),
    button('Too large (ε=1.0)', () => { eps = 1.0; eCtl.set(1.0); refresh(); }));

  const cv1 = el('canvas'), cv2 = el('canvas');
  const grid = el('div', { style: 'display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:1rem' },
    el('div', {}, el('div', { class: 'matrix-label', html: 'Clustering' }), el('div', { class: 'pg-canvas-wrap' }, cv1)),
    el('div', {}, el('div', { class: 'matrix-label', html: 'Sorted k-distance plot &nbsp;<span style="font-weight:400;color:var(--ink-faint)">(k = minPts)</span>' }), el('div', { class: 'pg-canvas-wrap' }, cv2)));
  const out = readout([['clusters', 0], ['core / border / noise', 0], ['noise fraction', 0], ['suggested ε (knee)', 0]]);
  const st = status('');
  wrap.append(grid, el('div', { class: 'pg-controls', style: 'margin-top:1rem' }, eCtl.root, mCtl.root, acts, out.root, st.root));

  const plot = trackPlot(new Plot(cv1, { xmin: -4, xmax: 4, ymin: -3, ymax: 3, aspect: 1.25, equal: true, pad: 0 }));
  const plot2 = trackPlot(new Plot(cv2, { xmin: 0, xmax: 1, ymin: 0, ymax: 1, aspect: 1.25, equal: false, pad: 0 }));

  let res = null, kd = [], sorted = [], knee = 0;
  function refresh() {
    res = CL.dbscan(pts, eps, minPts);
    kd = CL.kDistances(pts, minPts);
    sorted = [...kd].sort((a, b) => a - b);
    // knee: point of maximum distance from the chord joining the curve's endpoints
    const n = sorted.length;
    const x0 = 0, y0 = sorted[0], x1 = n - 1, y1 = sorted[n - 1];
    let bi = 0, bd = -1;
    for (let i = 0; i < n; i++) {
      const d = Math.abs((y1 - y0) * i - (x1 - x0) * sorted[i] + x1 * y0 - y1 * x0) / Math.hypot(y1 - y0, x1 - x0);
      if (d > bd) { bd = d; bi = i; }
    }
    knee = sorted[bi];
    const nCore = res.kind.filter(k => k === 'core').length;
    const nB = res.kind.filter(k => k === 'border').length;
    const nN = res.kind.filter(k => k === 'noise').length;
    out.set([
      { html: String(res.clusters), cls: res.clusters === 3 ? 'is-ok' : '' },
      `${nCore} / ${nB} / ${nN}`,
      { html: `${fmt(100 * nN / pts.length, 1)}%`, cls: nN / pts.length > .25 ? 'is-warn' : 'is-ok' },
      { html: fmt(knee, 3), cls: 'is-ok' },
    ]);
    st.set(
      nN / pts.length > .3
        ? `${WARN}<span>ε is too small: ${fmt(100 * nN / pts.length, 0)}% of points fail to reach minPts neighbours and are discarded as noise.</span>`
        : res.clusters <= 1
          ? `${WARN}<span>ε is too large: the low-density gaps have been bridged and everything has merged into ${res.clusters} cluster${res.clusters === 1 ? '' : 's'}.</span>`
          : `${OK}<span>${res.clusters} clusters, ${fmt(100 * nN / pts.length, 0)}% noise. DBSCAN found the curved arm too — no shape assumption is involved, only density.</span>`,
      nN / pts.length > .3 || res.clusters <= 1 ? 'warn' : 'ok');
    plot.render(); plot2.render();
  }

  plot.onDraw(p => {
    fitBounds(p, pts, .08);
    const cols = PAL();
    p.grid(1, { color: C.grid });
    pts.forEach((q, i) => {
      if (res.kind[i] === 'noise') {
        const [sx, sy] = p.toScreen(q);
        p.ctx.strokeStyle = withA(C.muted, .8); p.ctx.lineWidth = 1.6;
        p.ctx.beginPath();
        p.ctx.moveTo(sx - 3.5, sy - 3.5); p.ctx.lineTo(sx + 3.5, sy + 3.5);
        p.ctx.moveTo(sx + 3.5, sy - 3.5); p.ctx.lineTo(sx - 3.5, sy + 3.5);
        p.ctx.stroke();
      } else {
        const col = cols[res.labels[i] % cols.length];
        p.dot(q, { r: res.kind[i] === 'core' ? 4.6 : 3.4, color: res.kind[i] === 'core' ? col : withA(col, .45) });
      }
    });
    // show the eps radius once, on the first core point
    const ci = res.kind.indexOf('core');
    if (ci >= 0) {
      const [sx, sy] = p.toScreen(pts[ci]);
      p.ctx.beginPath(); p.ctx.arc(sx, sy, Math.abs(p.px(eps)), 0, Math.PI * 2);
      p.ctx.strokeStyle = C.ink; p.ctx.lineWidth = 1.6; p.ctx.setLineDash([5, 4]); p.ctx.stroke(); p.ctx.setLineDash([]);
    }
    p.axes(); p.ticks(1);
    p.legend([[C.ink, `ε = ${fmt(eps, 2)} (dashed circle)`]], { corner: 'tl', title: `${res.clusters} clusters` });
  });

  plot2.onDraw(p => {
    p.o.xmin = -2; p.o.xmax = sorted.length + 2;
    p.o.ymin = 0; p.o.ymax = Math.max(sorted[sorted.length - 1] * 1.1, eps * 1.2);
    p._computeScale();
    p.grid(p.o.ymax / 6, { color: C.grid });
    p.path(sorted.map((v, i) => [i, v]), { color: C.c1, lw: 2.6 });
    p.line([-2, eps], [sorted.length + 2, eps], { color: C.c2, lw: 2, dash: [6, 4] });
    p.badge([sorted.length, eps], `ε = ${fmt(eps, 2)}`, { color: C.c2, align: 'right', dy: -12 });
    p.line([-2, knee], [sorted.length + 2, knee], { color: C.c3, lw: 1.6, dash: [3, 4] });
    p.badge([2, knee], `knee ≈ ${fmt(knee, 2)}`, { color: C.c3, dy: 14 });
    p.axes(); p.ticks(p.o.ymax / 4);
    p.xlabel('points, sorted by distance to their k-th neighbour', { size: 10 });
  });

  refresh();

  node.appendChild(note(
    `DBSCAN needs no k and makes no shape assumption — it finds the curved arm as happily as the two blobs, and ` +
    `labels stragglers as noise instead of forcing them into a cluster. The right-hand panel is the standard ` +
    `recipe for choosing ε: sort every point by its distance to its k-th nearest neighbour and look for the knee, ` +
    `where the curve turns up as you leave the dense core and enter the outliers. Drag ε onto the knee and compare. ` +
    `The real limitation shows up if you try to capture the sparse blob and the dense blob at once: a single global ` +
    `ε cannot serve both, which is precisely what HDBSCAN was built to fix.`
  ));
});

/* ============================================================
   9. EM for a Gaussian mixture
   ============================================================ */
defineWidget('gmm-em', node => {
  const wrap = el('div');
  node.appendChild(wrap);

  const r = ML.rng(4);
  const pts = [];
  for (let i = 0; i < 60; i++) pts.push([-1.5 + ML.gauss(r) * .75, .4 + ML.gauss(r) * .45]);
  for (let i = 0; i < 55; i++) pts.push([1.4 + ML.gauss(r) * .55, .9 + ML.gauss(r) * .9]);
  for (let i = 0; i < 45; i++) pts.push([.2 + ML.gauss(r) * .9, -1.6 + ML.gauss(r) * .4]);

  let k = 3, step = 0, cov = 'full', hard = false;
  let frames = CL.gmmEM(pts, k, { seed: 6 });

  const kCtl = slider('components k', { min: 2, max: 4, step: 1, value: 3, format: v => String(v), onInput: v => { k = v; rebuild(); } });
  const cCtl = segmented([
    { label: 'Full', value: 'full' }, { label: 'Diagonal', value: 'diagonal' }, { label: 'Spherical', value: 'spherical' },
  ], { value: 'full', label: 'Covariance', onChange: v => { cov = v; rebuild(); } });
  const stepCtl = slider('EM iteration', { min: 0, max: 1, step: 1, value: 0, format: v => String(v + 1), onInput: v => { step = v; sync(); } });
  const hCtl = toggle('Show hard assignments instead of responsibilities', { value: false, onChange: v => { hard = v; plot.render(); } });

  const cv = el('canvas');
  const left = el('div', {}, el('div', { class: 'pg-canvas-wrap' }, cv));
  const right = el('div', { class: 'pg-controls' }, kCtl.root, cCtl.root, stepCtl.root, hCtl.root);
  const out = readout([['log-likelihood', 0], ['change from previous', 0], ['mixing weights π', 0], ['most ambiguous point', 0]]);
  const st = status('');
  right.append(out.root, st.root);
  wrap.appendChild(el('div', { class: 'pg-split pg-split--wide-ctrl' }, left, right));

  const plot = trackPlot(new Plot(cv, { xmin: -4, xmax: 4, ymin: -3, ymax: 3, aspect: 1.35, equal: true, pad: 0 }));

  function rebuild() {
    frames = CL.gmmEM(pts, k, { seed: 6, tied: cov === 'full' ? null : cov });
    stepCtl.input.max = String(frames.length - 1);
    step = Math.min(step, frames.length - 1);
    stepCtl.set(step);
    sync();
  }
  function sync() {
    const f = frames[step];
    const prev = step > 0 ? frames[step - 1].loglik : null;
    // the point whose responsibilities are closest to uniform
    let amb = 0, ae = -1;
    f.gamma.forEach((g, i) => {
      const H = -g.reduce((s, v) => s + (v > 1e-12 ? v * Math.log2(v) : 0), 0);
      if (H > ae) { ae = H; amb = i; }
    });
    out.set([
      { html: fmt(f.loglik, 4), cls: 'is-ok' },
      prev === null ? '— (first E-step)' : `+${fmt(f.loglik - prev, 6)}`,
      f.pi.map(v => fmt(v, 3)).join(', '),
      `x${amb + 1}: ${f.gamma[amb].map(v => fmt(v, 2)).join(' / ')}`,
    ]);
    st.set(
      step === frames.length - 1
        ? `${OK}<span>Converged after ${frames.length} iterations. Every step increased the log-likelihood — EM guarantees that, which is why it always terminates (at a <em>local</em> maximum).</span>`
        : `${INFO}<span>Each point is coloured by its responsibilities γ<sub>ij</sub>, blended. Points in the overlap regions are genuinely shared between components — that information is exactly what hard clustering throws away.</span>`,
      step === frames.length - 1 ? 'ok' : 'info');
    plot.render();
  }

  plot.onDraw(p => {
    fitBounds(p, pts, .1);
    const f = frames[step];
    const cols = PAL();
    p.grid(1, { color: C.grid });
    pts.forEach((q, i) => {
      if (hard) {
        p.dot(q, { r: 4.2, color: withA(cols[f.hard[i]], .85) });
      } else {
        // blend the component colours by responsibility
        let rr = 0, gg = 0, bb = 0;
        f.gamma[i].forEach((w, j) => {
          const hex = cols[j].trim();
          const n3 = hex.length === 4
            ? hex.slice(1).split('').map(c => parseInt(c + c, 16))
            : [1, 3, 5].map(t => parseInt(hex.slice(t, t + 2), 16));
          rr += w * n3[0]; gg += w * n3[1]; bb += w * n3[2];
        });
        const [sx, sy] = p.toScreen(q);
        p.ctx.fillStyle = `rgba(${rr | 0},${gg | 0},${bb | 0},.85)`;
        p.ctx.beginPath(); p.ctx.arc(sx, sy, 4.4, 0, Math.PI * 2); p.ctx.fill();
      }
    });
    f.mu.forEach((m, j) => {
      [1, 2].forEach(lv => {
        const e = CL.covEllipse(f.S[j], lv);
        p.ctx.save();
        const [sx, sy] = p.toScreen(m);
        p.ctx.translate(sx, sy);
        p.ctx.rotate(-e.theta);
        p.ctx.beginPath();
        p.ctx.ellipse(0, 0, Math.abs(p.px(e.rx)), Math.abs(p.px(e.ry)), 0, 0, Math.PI * 2);
        p.ctx.strokeStyle = withA(cols[j], lv === 1 ? .9 : .45);
        p.ctx.lineWidth = lv === 1 ? 2.2 : 1.4;
        p.ctx.stroke();
        p.ctx.restore();
      });
      cross(p, m, cols[j], 8, 3);
    });
    p.axes(); p.ticks(1);
    p.legend(f.mu.map((_, j) => [cols[j], `component ${j + 1} · π = ${fmt(f.pi[j], 2)}`]),
      { corner: 'tl', title: `iteration ${step + 1} · log-likelihood ${fmt(f.loglik, 2)}` });
  });

  rebuild();

  node.appendChild(note(
    `EM alternates two steps that mirror k-means exactly. The <strong>E-step</strong> computes responsibilities ` +
    `γ<sub>ij</sub> — the posterior probability that point i came from component j — which is the soft version of ` +
    `"assign to the nearest centroid". The <strong>M-step</strong> refits each Gaussian using those ` +
    `responsibilities as weights, the soft version of "move to the mean". Point colours are blends, so genuinely ` +
    `ambiguous points come out muddy rather than being forced to pick a side. Restricting the covariance to ` +
    `<em>spherical</em> and imagining σ² → 0 recovers k-means: the responsibilities harden to 0/1 and the ` +
    `ellipses become circles.`
  ));
});

/* ============================================================
   10. Linear autoencoder against PCA
   ============================================================ */
defineWidget('ae-pca', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, { xmin: -3, xmax: 3, ymin: -2, ymax: 2, aspect: 1.5, equal: true, pad: 0 }));

  const data = CL.curveData({ n: 110, bend: 0, noise: .28, seed: 12 });
  let net = CL.makeAE(2, 3, 1, { seed: 4, linear: true });
  let epoch = 0, running = false, raf = null;

  const acts = el('div', { class: 'pg-actions' },
    button('Train 200 steps', () => { for (let i = 0; i < 200; i++) { CL.aeStep(net, data, .04); epoch++; } refresh(); }),
    button('Run', () => { running = !running; if (running) loop(); }),
    button('Reset', () => { net = CL.makeAE(2, 3, 1, { seed: 4 + Math.floor(Math.random() * 900), linear: true }); epoch = 0; refresh(); }));
  const out = readout([['training steps', 0], ['autoencoder MSE', 0], ['PCA (k=1) MSE', 0], ['gap', 0], ['angle between the two subspaces', 0]]);
  const st = status('');
  right.append(acts, out.root, st.root);

  // PCA reference, computed once
  const mean = [data.reduce((s, q) => s + q[0], 0) / data.length, data.reduce((s, q) => s + q[1], 0) / data.length];
  const pcaRes = pca(data.map(q => [q[0] - mean[0], q[1] - mean[1]]));
  const pc1 = pcaRes.components[0];
  const pcaMSE = data.reduce((s, q) => {
    const c = [q[0] - mean[0], q[1] - mean[1]];
    const t = c[0] * pc1[0] + c[1] * pc1[1];
    return s + (c[0] - t * pc1[0]) ** 2 + (c[1] - t * pc1[1]) ** 2;
  }, 0) / data.length;

  function aeDirection() {
    // the decoder column spans the learned 1-D subspace
    const v = [net.W4[0][0] * net.W3[0][0] + net.W4[0][1] * net.W3[1][0] + net.W4[0][2] * net.W3[2][0],
               net.W4[1][0] * net.W3[0][0] + net.W4[1][1] * net.W3[1][0] + net.W4[1][2] * net.W3[2][0]];
    const n2 = Math.hypot(...v) || 1;
    return [v[0] / n2, v[1] / n2];
  }

  function loop() {
    if (!running) return;
    for (let i = 0; i < 25; i++) { CL.aeStep(net, data, .04); epoch++; }
    refresh();
    raf = requestAnimationFrame(loop);
  }
  window.addEventListener('pagehide', () => { running = false; cancelAnimationFrame(raf); });

  function refresh() {
    const mse = CL.aeLoss(net, data);
    const d = aeDirection();
    const cosang = Math.abs(d[0] * pc1[0] + d[1] * pc1[1]);
    const ang = Math.acos(clamp(cosang, -1, 1)) * 180 / Math.PI;
    out.set([
      String(epoch),
      { html: fmt(mse, 5), cls: Math.abs(mse - pcaMSE) < .01 ? 'is-ok' : '' },
      fmt(pcaMSE, 5),
      { html: fmt(mse - pcaMSE, 5), cls: Math.abs(mse - pcaMSE) < .01 ? 'is-ok' : '' },
      { html: `${fmt(ang, 2)}°`, cls: ang < 3 ? 'is-ok' : '' },
    ]);
    st.set(
      epoch === 0
        ? `${INFO}<span>Untrained: the decoder points in a random direction. Press Run and watch it swing onto the principal axis.</span>`
        : Math.abs(mse - pcaMSE) < .01
          ? `${OK}<span><strong>The linear autoencoder has converged to PCA.</strong> Its reconstruction error matches the k=1 PCA error to ${fmt(Math.abs(mse - pcaMSE), 5)}, and the learned subspace sits ${fmt(ang, 2)}° from the first principal component.</span>`
          : `${INFO}<span>Still training — MSE ${fmt(mse, 4)} against PCA's ${fmt(pcaMSE, 4)}. The autoencoder cannot beat PCA here; the best a rank-1 linear map can do <em>is</em> PCA.</span>`,
      Math.abs(mse - pcaMSE) < .01 ? 'ok' : 'info');
    plot.render();
  }

  plot.onDraw(p => {
    fitBounds(p, data, .15);
    p.grid(1, { color: C.grid });
    data.forEach(q => p.dot(q, { r: 3.4, color: withA(C.muted, .55) }));
    data.forEach(q => {
      const { xh } = CL.aeForward(net, q);
      p.line(q, xh, { color: withA(C.c1, .25), lw: 1 });
      p.dot(xh, { r: 3, color: withA(C.c1, .9) });
    });
    const t = 6;
    p.line([mean[0] - pc1[0] * t, mean[1] - pc1[1] * t], [mean[0] + pc1[0] * t, mean[1] + pc1[1] * t],
      { color: C.c3, lw: 2.4, dash: [7, 5] });
    const d = aeDirection();
    p.line([mean[0] - d[0] * t, mean[1] - d[1] * t], [mean[0] + d[0] * t, mean[1] + d[1] * t],
      { color: C.c2, lw: 2 });
    p.axes(); p.ticks(1);
    p.legend([[C.muted, 'data'], [C.c1, 'reconstruction'], [C.c3, 'PCA axis', [7, 5]], [C.c2, 'AE subspace']],
      { corner: 'tl', title: `${epoch} steps · MSE ${fmt(CL.aeLoss(net, data), 4)}` });
  });

  refresh();

  node.appendChild(note(
    `With every activation linear, the autoencoder computes <span class="u-mono">x̂ = W_D W_E x</span> — a rank-k ` +
    `linear map. Minimising squared reconstruction error over all such maps is exactly the problem PCA solves, so ` +
    `the two must reach the same error, and the learned subspace must be the principal subspace. Press Run and ` +
    `watch the solid line rotate onto the dashed one. The autoencoder does <em>not</em> recover the principal ` +
    `components individually — only the subspace they span, since any invertible mixing inside the bottleneck ` +
    `leaves the reconstruction unchanged.`
  ));
});

/* ============================================================
   11. What nonlinearity buys
   ============================================================ */
defineWidget('ae-nonlinear', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, { xmin: -3, xmax: 3, ymin: -2, ymax: 2, aspect: 1.5, equal: true, pad: 0 }));

  let bend = 1, linear = false, epoch = 0, running = false, raf = null;
  let data = CL.curveData({ n: 110, bend, noise: .05, seed: 12 });
  let net = CL.makeAE(2, 8, 1, { seed: 3, linear });

  const bCtl = slider('curvature of the data', { min: 0, max: 1.6, step: .05, value: 1, onInput: v => { bend = v; reset(); } });
  const lCtl = toggle('Force linear activations', { value: false, onChange: v => { linear = v; reset(); } });
  const acts = el('div', { class: 'pg-actions' },
    button('Train 300 steps', () => { for (let i = 0; i < 300; i++) { CL.aeStep(net, data, .05); epoch++; } refresh(); }),
    button('Run', () => { running = !running; if (running) loop(); }),
    button('Reset', () => reset()));
  const out = readout([['training steps', 0], ['reconstruction MSE', 0], ['best possible with a line (PCA)', 0], ['improvement over PCA', 0]]);
  const st = status('');
  right.append(bCtl.root, lCtl.root, acts, out.root, st.root);

  let pcaMSE = 0, mean = [0, 0], pc1 = [1, 0];
  function reset() {
    data = CL.curveData({ n: 110, bend, noise: .05, seed: 12 });
    net = CL.makeAE(2, 8, 1, { seed: 3, linear });
    epoch = 0;
    mean = [data.reduce((s, q) => s + q[0], 0) / data.length, data.reduce((s, q) => s + q[1], 0) / data.length];
    const pr = pca(data.map(q => [q[0] - mean[0], q[1] - mean[1]]));
    pc1 = pr.components[0];
    pcaMSE = data.reduce((s, q) => {
      const c = [q[0] - mean[0], q[1] - mean[1]];
      const t = c[0] * pc1[0] + c[1] * pc1[1];
      return s + (c[0] - t * pc1[0]) ** 2 + (c[1] - t * pc1[1]) ** 2;
    }, 0) / data.length;
    refresh();
  }
  function loop() {
    if (!running) return;
    for (let i = 0; i < 30; i++) { CL.aeStep(net, data, .05); epoch++; }
    refresh();
    raf = requestAnimationFrame(loop);
  }
  window.addEventListener('pagehide', () => { running = false; cancelAnimationFrame(raf); });

  function refresh() {
    const mse = CL.aeLoss(net, data);
    out.set([
      String(epoch),
      { html: fmt(mse, 5), cls: mse < pcaMSE * .6 ? 'is-ok' : '' },
      fmt(pcaMSE, 5),
      { html: mse < pcaMSE ? `${fmt(100 * (1 - mse / pcaMSE), 1)}% lower` : '—', cls: mse < pcaMSE * .6 ? 'is-ok' : '' },
    ]);
    st.set(
      epoch < 50
        ? `${INFO}<span>Press Run. The bottleneck is one number wide, so the network can only ever produce a single curve through the plane — the question is whether that curve can bend.</span>`
        : linear
          ? `${WARN}<span><strong>Linear activations:</strong> the reconstruction is stuck on a straight line and cannot beat PCA's ${fmt(pcaMSE, 4)}, no matter how long you train or how wide the hidden layers are.</span>`
          : mse < pcaMSE * .6
            ? `${OK}<span><strong>The nonlinear autoencoder has learned a curved 1-D manifold</strong> and reconstructs ${fmt(100 * (1 - mse / pcaMSE), 0)}% better than the best possible straight line.</span>`
            : `${INFO}<span>Training — MSE ${fmt(mse, 4)} against PCA's ${fmt(pcaMSE, 4)}.</span>`,
      linear && epoch >= 50 ? 'warn' : (mse < pcaMSE * .6 ? 'ok' : 'info'));
    plot.render();
  }

  plot.onDraw(p => {
    fitBounds(p, data, .15);
    p.grid(1, { color: C.grid });
    data.forEach(q => p.dot(q, { r: 3.2, color: withA(C.muted, .5) }));
    // the manifold the decoder traces as the latent code sweeps its range
    const zs = data.map(q => CL.aeForward(net, q).z[0]);
    const lo = Math.min(...zs), hi = Math.max(...zs);
    const curve = [];
    for (let i = 0; i <= 90; i++) {
      const z = lo + (hi - lo) * i / 90;
      const a3 = linear
        ? net.W3.map((row, j) => row[0] * z + net.b3[j])
        : net.W3.map((row, j) => Math.tanh(row[0] * z + net.b3[j]));
      curve.push(net.W4.map((row, j) => row.reduce((s, w, m) => s + w * a3[m], 0) + net.b4[j]));
    }
    p.path(curve, { color: C.c1, lw: 3 });
    const t = 6;
    p.line([mean[0] - pc1[0] * t, mean[1] - pc1[1] * t], [mean[0] + pc1[0] * t, mean[1] + pc1[1] * t],
      { color: C.c3, lw: 2, dash: [7, 5] });
    data.forEach(q => {
      const { xh } = CL.aeForward(net, q);
      p.line(q, xh, { color: withA(C.c1, .18), lw: 1 });
    });
    p.axes(); p.ticks(1);
    p.legend([[C.muted, 'data'], [C.c1, 'learned 1-D manifold'], [C.c3, 'PCA line', [7, 5]]],
      { corner: 'tl', title: `${linear ? 'linear' : 'tanh'} · ${epoch} steps · MSE ${fmt(CL.aeLoss(net, data), 4)}` });
  });

  reset();

  node.appendChild(note(
    `Both networks have the same shape — 2 → 8 → <strong>1</strong> → 8 → 2 — and the same single-number ` +
    `bottleneck. The only difference is the activation. The linear one can only ever trace a straight line, so ` +
    `it is capped at the PCA error however long you train it. The tanh one traces a <em>curve</em>, and on bent ` +
    `data it reconstructs far better from exactly the same one number. That is the whole argument for ` +
    `autoencoders over PCA: the data lies near a low-dimensional manifold, but that manifold need not be flat. ` +
    `Turn the curvature down to zero and the two become equivalent again.`
  ));
});

/* ============================================================
   12. The sparsity penalty
   ============================================================ */
defineWidget('sparse-kl', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, { xmin: 0, xmax: 1, ymin: 0, ymax: 1, aspect: 1.6, equal: false, pad: 0 }));

  let rho = .05, beta = 3, rhoHat = .3;
  const rCtl = slider('target sparsity ρ', { min: .01, max: .5, step: .01, value: .05, onInput: v => { rho = v; refresh(); } });
  const bCtl = slider('penalty weight β', { min: 0, max: 10, step: .1, value: 3, onInput: v => { beta = v; refresh(); } });
  const hCtl = slider('a neuron\'s average activation ρ̂', { min: .01, max: .95, step: .01, value: .3, onInput: v => { rhoHat = v; refresh(); } });
  const presets = el('div', { class: 'pg-actions' },
    button('ρ = 0.05', () => { rho = .05; rCtl.set(.05); refresh(); }),
    button('ρ = 0.10', () => { rho = .10; rCtl.set(.10); refresh(); }),
    button('ρ = 0.50', () => { rho = .50; rCtl.set(.50); refresh(); }));
  const out = readout([['KL(ρ ‖ ρ̂)', 0], ['β · KL', 0], ['penalty if ρ̂ = ρ', 0], ['penalty if the neuron is always on', 0]]);
  const st = status('');
  right.append(rCtl.root, bCtl.root, hCtl.root, presets, out.root, st.root);

  function refresh() {
    const kl = CL.klBernoulli(rho, rhoHat);
    out.set([
      fmt(kl, 4),
      { html: fmt(beta * kl, 4), cls: beta * kl > 1 ? 'is-warn' : 'is-ok' },
      { html: '0 exactly', cls: 'is-ok' },
      fmt(CL.klBernoulli(rho, .99), 3),
    ]);
    st.set(
      Math.abs(rhoHat - rho) < .015
        ? `${OK}<span>The neuron is firing at exactly its target rate, so it pays <strong>no penalty at all</strong>. KL is zero only here.</span>`
        : `${INFO}<span>This neuron fires ${fmt(rhoHat * 100, 0)}% of the time but should fire ${fmt(rho * 100, 0)}%, so it contributes <strong>${fmt(beta * kl, 3)}</strong> to the loss. Note the asymmetry: being too active costs far more than being too quiet.</span>`,
      Math.abs(rhoHat - rho) < .015 ? 'ok' : 'info');
    plot.render();
  }

  plot.onDraw(p => {
    p.o.xmin = 0; p.o.xmax = 1;
    const ymax = Math.max(2.2, beta * CL.klBernoulli(rho, .95) * .55);
    p.o.ymin = 0; p.o.ymax = ymax;
    p._computeScale();
    p.grid(ymax / 6, { color: C.grid });
    [[.05, C.c3], [.10, C.c5], [.50, C.c4]].forEach(([rr, col]) => {
      const pts = [];
      for (let i = 1; i < 100; i++) {
        const q = i / 100;
        pts.push([q, Math.min(beta * CL.klBernoulli(rr, q), ymax * 1.5)]);
      }
      p.path(pts, { color: withA(col, Math.abs(rr - rho) < .005 ? 1 : .35), lw: Math.abs(rr - rho) < .005 ? 3 : 1.8 });
    });
    const cur = [];
    for (let i = 1; i < 100; i++) {
      const q = i / 100;
      cur.push([q, Math.min(beta * CL.klBernoulli(rho, q), ymax * 1.5)]);
    }
    p.path(cur, { color: C.c1, lw: 3 });
    p.line([rho, 0], [rho, ymax], { color: C.c1, lw: 1.6, dash: [5, 4] });
    p.badge([rho, ymax * .93], `ρ = ${fmt(rho, 2)}`, { color: C.c1, align: 'center' });
    const y = Math.min(beta * CL.klBernoulli(rho, rhoHat), ymax * 1.5);
    p.dot([rhoHat, y], { r: 6, color: C.c2 });
    p.line([rhoHat, 0], [rhoHat, y], { color: C.c2, lw: 1.6, dash: [3, 3] });
    p.badge([rhoHat, y], `β·KL = ${fmt(beta * CL.klBernoulli(rho, rhoHat), 3)}`, { color: C.c2, align: 'center', dy: -16 });
    p.axes(); p.ticks(ymax / 4);
    p.legend([[C.c3, 'ρ = 0.05'], [C.c5, 'ρ = 0.10'], [C.c4, 'ρ = 0.50']], { corner: 'tr', title: 'penalty shape' });
    p.xlabel('average activation ρ̂ of a latent neuron', { size: 10.5 });
  });

  refresh();

  node.appendChild(note(
    `The sparsity term treats each latent neuron's average activation ρ̂<sub>j</sub> as a Bernoulli rate and ` +
    `penalises how far it drifts from a small target ρ. The curve is zero at ρ̂ = ρ and rises steeply on both ` +
    `sides — but not symmetrically: with ρ = 0.05, a neuron firing half the time is punished far harder than one ` +
    `firing 1% of the time. That asymmetry is what drives most units toward silence, leaving a few sharply tuned ` +
    `detectors. β sets how much this matters relative to reconstruction; at β = 0 the constraint vanishes entirely.`
  ));
});

/* ============================================================
   13. The VAE latent space
   ============================================================ */
defineWidget('vae-latent', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, { xmin: -3.4, xmax: 3.4, ymin: -2.6, ymax: 2.6, aspect: 1.4, equal: true, pad: 0 }));

  let sigma = .55, spread = 1.9, showPrior = true;
  const sCtl = slider('encoder σ (per-point spread)', { min: .05, max: 1.5, step: .01, value: .55, onInput: v => { sigma = v; refresh(); } });
  const mCtl = slider('how far the means drift from 0', { min: 0, max: 3, step: .05, value: 1.9, onInput: v => { spread = v; refresh(); } });
  const pCtl = toggle('Show the N(0, I) prior', { value: true, onChange: v => { showPrior = v; plot.render(); } });
  const presets = el('div', { class: 'pg-actions' },
    button('Plain autoencoder', () => { sigma = .05; spread = 2.6; sCtl.set(.05); mCtl.set(2.6); refresh(); }),
    button('Well-behaved VAE', () => { sigma = .55; spread = 1.0; sCtl.set(.55); mCtl.set(1.0); refresh(); }),
    button('Posterior collapse', () => { sigma = 1.0; spread = 0; sCtl.set(1.0); mCtl.set(0); refresh(); }));
  const out = readout([['KL per point', 0], ['coverage of the prior', 0], ['reconstruction quality', 0], ['sampling from N(0,I) lands in data', 0]]);
  const st = status('');
  right.append(sCtl.root, mCtl.root, pCtl.root, presets, out.root, st.root);

  const r = ML.rng(15);
  const base = Array.from({ length: 26 }, (_, i) => {
    const t = 2 * Math.PI * i / 26;
    return [Math.cos(t) * (1 + .25 * Math.cos(3 * t)), Math.sin(t) * (1 + .25 * Math.cos(3 * t))];
  });
  const jitter = base.map(() => [ML.gauss(r) * .18, ML.gauss(r) * .18]);

  let mus = [], kl = 0, cover = 0, hit = 0;
  function refresh() {
    mus = base.map((b, i) => [b[0] * spread / 1.9 + jitter[i][0], b[1] * spread / 1.9 + jitter[i][1]]);
    kl = mus.reduce((s, m) => s + CL.klGaussian(m, [sigma, sigma]), 0) / mus.length;
    // fraction of prior samples that land within sigma of some posterior mean
    const rr = ML.rng(77);
    let inside = 0;
    const N = 400;
    for (let i = 0; i < N; i++) {
      const z = [ML.gauss(rr), ML.gauss(rr)];
      if (mus.some(m => CL.euclid(m, z) < Math.max(sigma, .25) * 1.6)) inside++;
    }
    hit = inside / N;
    cover = hit;
    out.set([
      { html: fmt(kl, 4), cls: kl < 2 ? 'is-ok' : 'is-warn' },
      `${fmt(cover * 100, 0)}%`,
      { html: sigma < .15 ? 'sharp' : sigma > .9 ? 'blurry' : 'good', cls: sigma > .9 ? 'is-warn' : 'is-ok' },
      { html: `${fmt(hit * 100, 0)}%`, cls: hit > .6 ? 'is-ok' : 'is-warn' },
    ]);
    st.set(
      spread < .2
        ? `${WARN}<span><strong>Posterior collapse.</strong> Every input maps to the same distribution, so the KL term is near zero but the latent code carries no information about x at all. Samples are generic.</span>`
        : sigma < .15
          ? `${WARN}<span><strong>This is essentially a plain autoencoder.</strong> With σ ≈ 0 each input maps to a single point, the KL term is large (${fmt(kl, 2)}), and the gaps between encoded points decode to nothing — sampling from N(0, I) only lands in data ${fmt(hit * 100, 0)}% of the time.</span>`
          : hit > .6
            ? `${OK}<span>The posteriors tile the prior: draw z ~ N(0, I) and it lands in a region the decoder has actually been trained on ${fmt(hit * 100, 0)}% of the time. <strong>That is what makes a VAE generative.</strong></span>`
            : `${INFO}<span>The encoded distributions do not yet cover the prior — increase σ or pull the means inward.</span>`,
      spread < .2 || sigma < .15 ? 'warn' : hit > .6 ? 'ok' : 'info');
    plot.render();
  }

  plot.onDraw(p => {
    p.grid(1, { color: C.grid });
    if (showPrior) {
      [1, 2].forEach(lv => {
        const [sx, sy] = p.toScreen([0, 0]);
        p.ctx.beginPath();
        p.ctx.arc(sx, sy, Math.abs(p.px(lv)), 0, Math.PI * 2);
        p.ctx.strokeStyle = withA(C.ink, lv === 1 ? .5 : .28);
        p.ctx.lineWidth = 1.8; p.ctx.setLineDash([6, 5]); p.ctx.stroke(); p.ctx.setLineDash([]);
      });
      p.badge([0, 2], 'prior N(0, I)', { color: C.ink, align: 'center', dy: -13 });
    }
    mus.forEach(m => {
      const [sx, sy] = p.toScreen(m);
      p.ctx.beginPath(); p.ctx.arc(sx, sy, Math.abs(p.px(sigma)), 0, Math.PI * 2);
      p.ctx.fillStyle = withA(C.c1, .16); p.ctx.fill();
      p.ctx.strokeStyle = withA(C.c1, .6); p.ctx.lineWidth = 1.2; p.ctx.stroke();
      p.dot(m, { r: 3.2, color: C.c1 });
    });
    // a handful of prior draws
    const rr = ML.rng(5);
    for (let i = 0; i < 30; i++) {
      const z = [ML.gauss(rr), ML.gauss(rr)];
      const ok = mus.some(m => CL.euclid(m, z) < Math.max(sigma, .25) * 1.6);
      const [sx, sy] = p.toScreen(z);
      p.ctx.strokeStyle = ok ? C.c3 : C.c4; p.ctx.lineWidth = 1.8;
      p.ctx.beginPath();
      p.ctx.moveTo(sx - 4, sy - 4); p.ctx.lineTo(sx + 4, sy + 4);
      p.ctx.moveTo(sx + 4, sy - 4); p.ctx.lineTo(sx - 4, sy + 4);
      p.ctx.stroke();
    }
    p.axes(); p.ticks(1);
    p.legend([[C.c1, 'q(z|x) — one per input'], [C.c3, 'prior draw that decodes well'], [C.c4, 'prior draw in a gap']],
      { corner: 'tl', title: `KL = ${fmt(kl, 2)} per point` });
  });

  refresh();

  node.appendChild(note(
    `A VAE encoder outputs a <em>distribution</em> per input, not a point. The KL term ` +
    `<span class="u-mono">½Σ(μ² + σ² − log σ² − 1)</span> pulls those distributions toward N(0, I), and the ` +
    `reconstruction term pushes them apart so inputs stay distinguishable. The balance is the whole design. ` +
    `Press <strong>Plain autoencoder</strong>: with σ → 0 the codes are isolated points and most prior draws ` +
    `land in empty space, which is why a vanilla autoencoder is not a generative model. Press ` +
    `<strong>Posterior collapse</strong> for the opposite failure — the KL term wins, every input encodes ` +
    `identically, and the latent code carries no information.`
  ));
});
