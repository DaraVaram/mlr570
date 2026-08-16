/* ============================================================
   widgets/prep.js — interactive figures for Data Preprocessing
   ============================================================ */
import {
  Plot, Dragger, C, el, slider, toggle, segmented, button,
  matrixInput, readout, status, defineWidget, canvasHost,
  trackPlot, clamp, fmt, round,
} from '../viz.js';
import * as ML from '../ml.js';

function split(node, { aspect = 1.5, hint, wide = false } = {}) {
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
   1. Why scale at all? — one feature 1000x the other
   ============================================================ */
defineWidget('scaling-demo', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -4, xmax: 4, ymin: -3, ymax: 3, aspect: 1.35, equal: false, pad: 0,
  }));

  let spreadExp = 2;            // feature 1 is 10^spreadExp times feature 2
  let scaling = 'none';
  const base = ML.blobs({ n: 150, spread: 1, seed: 42 });

  const spreadCtl = slider('Feature 1 scale', {
    min: 0, max: 3, step: .1, value: 2,
    format: v => `${fmt(10 ** v, 0)}×`,
    onInput: v => { spreadExp = v; refresh(); },
  });
  const scaleCtl = segmented([
    { label: 'Raw', value: 'none' },
    { label: 'Min–max', value: 'minmax' },
    { label: 'Z-score', value: 'zscore' },
  ], { value: 'none', label: 'Preprocessing', onChange: v => { scaling = v; refresh(); } });
  const out = readout([['feature 1 range', 0], ['feature 2 range', 0], ['clustering accuracy', 0], ['k-means inertia', 0]]);
  const st = status('');
  right.append(spreadCtl.root, scaleCtl.root, out.root, st.root);

  function data() {
    const k = 10 ** spreadExp;
    const raw = base.points.map(([a, b]) => [a * k, b]);
    return scaling === 'none' ? raw : ML.scaleColumns(raw, scaling);
  }

  /** best label permutation accuracy for 3 clusters */
  function accuracy(labels, truth) {
    const perms = [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]];
    let best = 0;
    for (const p of perms) {
      let c = 0;
      for (let i = 0; i < labels.length; i++) if (p[labels[i]] === truth[i]) c++;
      best = Math.max(best, c / labels.length);
    }
    return best;
  }

  let km = null, pts = null;
  function refresh() {
    pts = data();
    km = ML.kmeans(pts, 3, { seed: 7 });
    const f1 = pts.map(p => p[0]), f2 = pts.map(p => p[1]);
    const acc = accuracy(km.labels, base.labels);
    out.set([
      `${fmt(Math.min(...f1), 2)} … ${fmt(Math.max(...f1), 2)}`,
      `${fmt(Math.min(...f2), 2)} … ${fmt(Math.max(...f2), 2)}`,
      { html: `${fmt(acc * 100, 1)}%`, cls: acc > .9 ? 'is-ok' : 'is-warn' },
      fmt(km.inertia, 2),
    ]);
    st.set(
      acc > .9
        ? `${ICON_OK}<span><strong>Clusters recovered.</strong> The three groups are found correctly — the features now carry comparable weight.</span>`
        : `${ICON_WARN}<span><strong>Clustering has failed.</strong> Feature 1 dominates the distance, so k-means slices along it and ignores feature 2 entirely.</span>`,
      acc > .9 ? 'ok' : 'warn');

    // fit the view to the data
    const pad = 0.12;
    const [x0, x1] = [Math.min(...f1), Math.max(...f1)];
    const [y0, y1] = [Math.min(...f2), Math.max(...f2)];
    const dx = (x1 - x0) * pad || 1, dy = (y1 - y0) * pad || 1;
    plot.setBounds({ xmin: x0 - dx, xmax: x1 + dx, ymin: y0 - dy, ymax: y1 + dy });
  }

  plot.onDraw(p => {
    p.grid((p.o.xmax - p.o.xmin) / 8, { color: C.grid });
    p.axes();
    const cols = [C.c1, C.c3, C.c2];
    pts.forEach((q, i) => p.dot(q, { r: 3.4, color: cols[km.labels[i]], alpha: .78 }));
    km.centers.forEach((c, i) => {
      p.dot(c, { r: 8, color: cols[i], ring: true, ringLw: 3 });
      p.dot(c, { r: 2.5, color: C.raised });
    });
    p.text({ px: 12, py: 11 }, 'colour = cluster assigned by k-means · rings = cluster centres',
      { color: C.muted, size: 11 });
    p.text({ px: 12, py: p.h - 5 }, 'feature 1 →', { color: C.muted, size: 11, baseline: 'bottom' });
  });

  refresh();

  node.appendChild(note(
    `The data is unchanged — only the <em>units</em> of feature 1 move. Because Euclidean distance adds the ` +
    `two features' squared differences, a feature measured in thousands drowns out one measured in ones, and ` +
    `k-means ends up cutting along the loud axis. Set the scale to 1000× and watch accuracy collapse, then ` +
    `switch on either scaler to recover it. <strong>This is the entire argument for normalisation</strong>, ` +
    `and it applies to every distance-based method: k-NN, k-means, SVMs with RBF kernels.`
  ));
});

/* ============================================================
   2. Rolling aggregation — mean vs median vs EWMA
   ============================================================ */
defineWidget('rolling-agg', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: 0, xmax: 120, ymin: 0, ymax: 40, aspect: 1.7, equal: false, pad: 0,
  }));

  let w = 10, spikes = 0, showMean = true, showMedian = true, showEwma = false, lambda = .8;

  const wCtl = slider('Window size w', {
    min: 2, max: 40, step: 1, value: 10, format: v => String(v),
    onInput: v => { w = v; refresh(); },
  });
  const spikeCtl = slider('Outlier spikes', {
    min: 0, max: 6, step: 1, value: 0, format: v => String(v),
    onInput: v => { spikes = v; refresh(); },
  });
  const lamCtl = slider('EWMA λ', {
    min: .1, max: .95, step: .01, value: .8,
    onInput: v => { lambda = v; refresh(); },
  });
  const toggles = el('div', { style: 'display:flex;flex-direction:column;gap:.5rem' },
    toggle('Rolling mean', { value: true, onChange: v => { showMean = v; plot.render(); } }).root,
    toggle('Rolling median', { value: true, onChange: v => { showMedian = v; plot.render(); } }).root,
    toggle('EWMA', { value: false, onChange: v => { showEwma = v; plot.render(); } }).root,
  );
  const out = readout([['undefined at start', 0], ['mean · std of residual', 0], ['median · std of residual', 0], ['lag (samples)', 0]]);
  const st = status('');
  right.append(wCtl.root, spikeCtl.root, lamCtl.root, toggles, out.root, st.root);

  let y = [], rm = [], rmed = [], re = [];
  function refresh() {
    y = ML.trendSeries({ n: 120, spikes, seed: 5, noise: 1.1 });
    rm = ML.rollingMean(y, w);
    rmed = ML.rollingMedian(y, w);
    re = ML.ewma(y, lambda);

    const trend = y.map((_, i) => 10 + i * 0.14);
    const resid = (est) => {
      const d = [];
      est.forEach((v, i) => { if (v != null && i >= w - 1) d.push(v - trend[i]); });
      return ML.std(d, 1);
    };
    out.set([
      `${w - 1} points`,
      fmt(resid(rm), 3),
      fmt(resid(rmed), 3),
      `≈ ${fmt((w - 1) / 2, 1)}`,
    ]);
    st.set(
      spikes > 0
        ? `${ICON_INFO}<span>With spikes present the <strong>mean</strong> is yanked off course while the <strong>median</strong> barely moves — bounded versus unbounded influence.</span>`
        : `${ICON_INFO}<span>Without outliers the two are close, and the mean is the slightly more efficient estimator. Add spikes to separate them.</span>`,
      'info');

    const all = y.filter(Number.isFinite);
    const lo = Math.min(...all), hi = Math.max(...all);
    const pad = (hi - lo) * .1 || 1;
    plot.setBounds({ xmin: 0, xmax: y.length, ymin: lo - pad, ymax: hi + pad });
  }

  const line = (arr, opts) => {
    const pts = [];
    arr.forEach((v, i) => { if (v != null) pts.push([i, v]); });
    return pts.length ? plot.path(pts, opts) : null;
  };

  plot.onDraw(p => {
    p.grid((p.o.ymax - p.o.ymin) / 6, { color: C.grid });
    p.axes({ ticks: 20 });
    // the underlying trend the estimators are chasing
    p.fn(x => 10 + x * 0.14, { color: C.muted, lw: 1.3, dash: [5, 5], alpha: .55 });
    line(y, { color: C.c5, lw: 1.3, alpha: .7 });
    if (showMean) line(rm, { color: C.c2, lw: 2.4 });
    if (showMedian) line(rmed, { color: C.c3, lw: 2.4 });
    if (showEwma) line(re, { color: C.c4, lw: 2.2, dash: [6, 4] });

    // shade the region where the window is incomplete
    p.ctx.fillStyle = C.grid;
    p.ctx.globalAlpha = .5;
    p.ctx.fillRect(p.X(0), 0, p.px(w - 1), p.h);
    p.ctx.globalAlpha = 1;
    p.text([Math.max(1, (w - 1) / 2), p.o.ymax - (p.o.ymax - p.o.ymin) * .06], 'NaN',
      { align: 'center', size: 10.5, color: C.muted });

    const key = [
      [C.c5, 'raw series'], [C.muted, 'true trend'],
      ...(showMean ? [[C.c2, 'rolling mean']] : []),
      ...(showMedian ? [[C.c3, 'rolling median']] : []),
      ...(showEwma ? [[C.c4, 'EWMA']] : []),
    ];
    key.forEach(([col, label], i) => {
      p.ctx.strokeStyle = col; p.ctx.lineWidth = 2.4;
      p.ctx.beginPath(); p.ctx.moveTo(14, 14 + i * 15); p.ctx.lineTo(30, 14 + i * 15); p.ctx.stroke();
      p.text({ px: 35, py: 14 + i * 15 }, label, { color: C.muted, size: 10.5, weight: 500 });
    });
  });

  refresh();

  node.appendChild(note(
    `Two things to notice. First, the shaded strip: with a right-aligned window the first <strong>w−1</strong> ` +
    `values are undefined, because there is not yet a full window to average. Second, both smoothers ` +
    `<strong>lag</strong> the trend — they only look backwards, so on a rising series they sit below it. ` +
    `That is the bias–variance trade-off in miniature: larger w means smoother (less variance) but laggier ` +
    `(more bias). Add spikes and the mean's unbounded influence shows itself immediately.`
  ));
});

/* ============================================================
   3. Sampling — imbalance, over- and under-sampling
   ============================================================ */
defineWidget('sampling', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: 0, xmax: 10, ymin: 0, ymax: 6, aspect: 1.6, equal: false, pad: 0,
  }));

  const N = 900;
  let minorityPct = 10, method = 'none';

  const pctCtl = slider('Minority class share', {
    min: 1, max: 50, step: 1, value: 10, format: v => `${v}%`,
    onInput: v => { minorityPct = v; refresh(); },
  });
  const methodCtl = segmented([
    { label: 'Original', value: 'none' },
    { label: 'Oversample', value: 'over' },
    { label: 'Undersample', value: 'under' },
  ], { value: 'none', label: 'Balancing', onChange: v => { method = v; refresh(); } });
  const out = readout([['majority (class 1)', 0], ['minority (class 2)', 0], ['total samples', 0], ['ratio', 0], ['accuracy of "always class 1"', 0]]);
  const st = status('');
  right.append(pctCtl.root, methodCtl.root, out.root, st.root);

  let counts = { maj: 0, min: 0, dupes: 0, dropped: 0 };
  function refresh() {
    const min0 = Math.round(N * minorityPct / 100);
    const maj0 = N - min0;
    let maj = maj0, min = min0, dupes = 0, dropped = 0;
    if (method === 'over') { dupes = maj0 - min0; min = maj0; }
    if (method === 'under') { dropped = maj0 - min0; maj = min0; }
    counts = { maj, min, dupes, dropped, maj0, min0 };

    out.set([
      method === 'under' ? `${maj}  (dropped ${dropped})` : String(maj),
      method === 'over' ? `${min}  (${dupes} synthetic)` : String(min),
      String(maj + min),
      `${fmt(maj / Math.max(1, min), 2)} : 1`,
      { html: `${fmt(100 * maj / (maj + min), 1)}%`,
        cls: maj / (maj + min) > .8 ? 'is-warn' : '' },
    ]);

    const msg = {
      none: minorityPct <= 10
        ? `${ICON_WARN}<span><strong>Severely imbalanced.</strong> A model that always predicts class 1 already scores ${fmt(100 * maj / (maj + min), 1)}% accuracy while learning nothing at all.</span>`
        : `${ICON_INFO}<span>Reasonably balanced. Accuracy is a more trustworthy metric here.</span>`,
      over: `${ICON_INFO}<span><strong>${dupes} synthetic minority samples</strong> added. Balanced — but those points were invented, not observed.</span>`,
      under: `${ICON_WARN}<span><strong>${dropped} majority samples discarded.</strong> Balanced, and every remaining point is real — but you have thrown away ${fmt(100 * dropped / N, 1)}% of your data.</span>`,
    };
    st.set(msg[method], method === 'none' ? (minorityPct <= 10 ? 'warn' : 'info') : (method === 'under' ? 'warn' : 'info'));
    plot.render();
  }

  plot.onDraw(p => {
    const { maj, min, dupes, dropped } = counts;
    const total = Math.max(maj + min, 1);
    const perDot = 10;
    const draw = (count, synth, x0, color, label, sublabel) => {
      const dots = Math.round(count / perDot);
      const cols = 9;
      for (let i = 0; i < dots; i++) {
        const cx = x0 + (i % cols) * .34;
        const cy = 4.6 - Math.floor(i / cols) * .34;
        const isSynth = synth > 0 && i >= Math.round((count - synth) / perDot);
        p.ctx.globalAlpha = isSynth ? .5 : 1;
        p.ctx.fillStyle = color;
        p.ctx.beginPath();
        p.ctx.arc(p.X(cx), p.Y(cy), 4.2, 0, Math.PI * 2);
        p.ctx.fill();
        if (isSynth) {
          p.ctx.globalAlpha = 1; p.ctx.strokeStyle = color; p.ctx.lineWidth = 1.4;
          p.ctx.stroke();
        }
        p.ctx.globalAlpha = 1;
      }
      p.text([x0 + 1.2, 5.4], label, { align: 'center', size: 13, weight: 700, color });
      p.text([x0 + 1.2, 5.05], sublabel, { align: 'center', size: 10.5, color: C.muted });
    };
    draw(maj, 0, .7, C.c1, `Class 1 · ${maj}`, dropped ? `${dropped} dropped` : 'majority');
    draw(min, dupes, 5.6, C.c2, `Class 2 · ${min}`, dupes ? `${dupes} synthetic (hollow)` : 'minority');
    p.text({ px: 12, py: p.h - 5 }, `each dot = ${perDot} samples`, { color: C.muted, size: 10.5, baseline: 'bottom' });
  });

  refresh();

  node.appendChild(note(
    `At a 10% minority share, a model that simply always answers "class 1" scores 90% accuracy while being ` +
    `completely useless — which is why <a href="#metrics">accuracy is the wrong metric</a> on imbalanced data. ` +
    `Both fixes have a cost: <strong>oversampling invents data</strong> (dangerous in medicine, where a ` +
    `synthetic patient may not be a plausible patient), while <strong>undersampling throws real data away</strong>. ` +
    `Neither is free, and the right choice depends on which risk your application can absorb.`
  ));
});

/* ============================================================
   4. Normalisation — four features on wildly different scales
   ============================================================ */
defineWidget('normalization', node => {
  const wrap = el('div');
  node.appendChild(wrap);

  const { rows, names } = ML.peopleTable({ n: 200, seed: 11 });
  let kind = 'none';

  const cv = el('canvas');
  const left = el('div', {}, el('div', { class: 'pg-canvas-wrap' }, cv));
  const right = el('div', { class: 'pg-controls' });
  wrap.appendChild(el('div', { class: 'pg-split pg-split--wide-ctrl' }, left, right));

  const kindCtl = segmented([
    { label: 'Raw', value: 'none' },
    { label: 'Min–max', value: 'minmax' },
    { label: 'Z-score', value: 'zscore' },
    { label: 'Robust', value: 'robust' },
  ], { value: 'none', label: 'Scaler', onChange: v => { kind = v; refresh(); } });

  const out = readout(names.map(n => [n, 0]));
  const st = status('');
  right.append(kindCtl.root, out.root, st.root);

  const plot = trackPlot(new Plot(cv, { xmin: 0, xmax: 4, ymin: -1, ymax: 1, aspect: 1.5, equal: false, pad: 0 }));

  let data = rows;
  function refresh() {
    data = kind === 'none' ? rows : ML.scaleColumns(rows, kind);
    const cols = names.map((_, j) => data.map(r => r[j]));
    out.set(cols.map(c => `${fmt(Math.min(...c), 2)} … ${fmt(Math.max(...c), 2)}`));

    const ranges = cols.map(c => Math.max(...c) - Math.min(...c));
    const worst = Math.max(...ranges) / Math.max(1e-9, Math.min(...ranges));
    st.set(
      kind === 'none'
        ? `${ICON_WARN}<span>The widest feature spans <strong>${fmt(worst, 0)}×</strong> more ground than the narrowest. Any distance or gradient will be dominated by income.</span>`
        : `${ICON_OK}<span>Ranges are now within <strong>${fmt(worst, 1)}×</strong> of each other — every feature gets a comparable say.</span>`,
      kind === 'none' ? 'warn' : 'ok');

    const all = data.flat();
    const lo = Math.min(...all), hi = Math.max(...all);
    const pad = (hi - lo) * .08 || 1;
    plot.setBounds({ xmin: -.35, xmax: 4, ymin: lo - pad, ymax: hi + pad });
  }

  plot.onDraw(p => {
    p.grid((p.o.ymax - p.o.ymin) / 6, { color: C.grid });
    const cols = [C.c1, C.c2, C.c3, C.c5];
    names.forEach((nm, j) => {
      const vals = data.map(r => r[j]);
      const x = j + .5;
      // box plot: whiskers, IQR box, median
      const q1 = ML.quantile(vals, .25), q2 = ML.median(vals), q3 = ML.quantile(vals, .75);
      const lo = Math.min(...vals), hi = Math.max(...vals);
      const w = .17;
      p.line([x, lo], [x, hi], { color: cols[j], lw: 1.4, alpha: .6 });
      p.polygon([[x - w, q1], [x + w, q1], [x + w, q3], [x - w, q3]],
        { color: cols[j], lw: 2, fill: C.fill });
      p.line([x - w, q2], [x + w, q2], { color: cols[j], lw: 3 });
      // jittered points
      vals.forEach((v, i) => {
        const jx = x + ((i * 37 % 100) / 100 - .5) * .22;
        p.dot([jx, v], { r: 1.7, color: cols[j], alpha: .3 });
      });
      p.text([x, p.o.ymin + (p.o.ymax - p.o.ymin) * .02], nm,
        { align: 'center', size: 10.5, color: C.muted, weight: 600 });
    });
    p.ctx.strokeStyle = C.axis; p.ctx.lineWidth = 1.2;
    p.ctx.beginPath();
    p.ctx.moveTo(p.X(-.35), p.Y(0)); p.ctx.lineTo(p.X(4), p.Y(0));
    p.ctx.stroke();
    p.ticks((p.o.ymax - p.o.ymin) / 5);
  });

  refresh();

  node.appendChild(note(
    `Four real-ish features: age in tens, income in thousands, credit score capped at 900, monthly spend in ` +
    `hundreds. On the raw scale income simply <em>is</em> the dataset as far as any distance is concerned. ` +
    `<strong>Min–max</strong> forces everything into [0,1] but is pinned by the single most extreme point; ` +
    `<strong>z-score</strong> centres on the mean with unit variance; <strong>robust</strong> scaling uses the ` +
    `median and IQR, so outliers stop dictating the transformation. Note that scalers must be fitted on the ` +
    `<em>training set only</em>, then applied unchanged to validation and test data.`
  ));
});

/* ============================================================
   5. Encoding — the same column, four ways
   ============================================================ */
defineWidget('encoding', node => {
  const wrap = el('div');
  node.appendChild(wrap);

  const COLORS = ['Red', 'Green', 'Blue', 'Red', 'Green', 'Red'];
  let rows = [...COLORS];
  let method = 'label';

  const methodCtl = segmented([
    { label: 'Label', value: 'label' },
    { label: 'One-hot', value: 'onehot' },
    { label: 'Frequency', value: 'freq' },
    { label: 'Binary', value: 'binary' },
  ], { value: 'label', label: 'Encoding', onChange: v => { method = v; refresh(); } });

  const acts = el('div', { class: 'pg-actions' },
    button('+ Red', () => { rows.push('Red'); refresh(); }),
    button('+ Green', () => { rows.push('Green'); refresh(); }),
    button('+ Blue', () => { rows.push('Blue'); refresh(); }),
    button('+ Purple', () => { rows.push('Purple'); refresh(); }),
    button('Reset', () => { rows = [...COLORS]; refresh(); }),
  );

  const table = el('div', { class: 'table-scroll' });
  const st = status('');
  const blurb = el('div', { style: 'font-size:.88rem;color:var(--ink-muted);line-height:1.55;margin-top:.8rem' });
  wrap.append(methodCtl.root, acts, st.root, table, blurb);

  function refresh() {
    const cats = [...new Set(rows)].sort();
    const freq = {};
    rows.forEach(r => { freq[r] = (freq[r] || 0) + 1; });

    let head = [], body = [];
    if (method === 'label') {
      head = ['Colour', 'Label'];
      body = rows.map(r => [r, String(cats.indexOf(r))]);
    } else if (method === 'onehot') {
      head = ['Colour', ...cats];
      body = rows.map(r => [r, ...cats.map(c => (c === r ? '1' : '0'))]);
    } else if (method === 'freq') {
      head = ['Colour', 'Frequency'];
      body = rows.map(r => [r, String(freq[r])]);
    } else {
      const bits = Math.max(1, Math.ceil(Math.log2(cats.length + 1)));
      head = ['Colour', 'Integer', ...Array.from({ length: bits }, (_, i) => `b${bits - i}`)];
      body = rows.map(r => {
        const n = cats.indexOf(r) + 1;
        const b = n.toString(2).padStart(bits, '0').split('');
        return [r, String(n), ...b];
      });
    }

    const colOf = c => ({ Red: C.c4, Green: C.c3, Blue: C.c5, Purple: C.c6 }[c] || C.muted);
    table.innerHTML =
      `<table><thead><tr>${head.map(h => `<th>${h}</th>`).join('')}</tr></thead><tbody>` +
      body.map(r => `<tr>${r.map((cell, i) =>
        i === 0
          ? `<td style="font-weight:600;color:${colOf(cell)}">${cell}</td>`
          : `<td class="u-mono">${cell}</td>`).join('')}</tr>`).join('') +
      `</tbody></table>`;

    const nCols = { label: 1, onehot: cats.length, freq: 1, binary: Math.max(1, Math.ceil(Math.log2(cats.length + 1))) }[method];
    const msgs = {
      label: `${ICON_WARN}<span><strong>Beware the fake ordering.</strong> Encoding to 0, 1, 2 tells a linear or distance-based model that Blue &gt; Green &gt; Red, and that Red is "twice as far" from Blue as from Green. None of that is true.</span>`,
      onehot: `${ICON_OK}<span><strong>No false ordering</strong>, and every category is equidistant from every other. The cost is width: ${cats.length} categories became ${cats.length} columns.</span>`,
      freq: `${ICON_INFO}<span>Compact — one column regardless of cardinality. But it silently assumes <strong>frequency relates to the target</strong>, and two equally common categories become indistinguishable.</span>`,
      binary: `${ICON_INFO}<span>Only <strong>${nCols} column${nCols > 1 ? 's' : ''}</strong> for ${cats.length} categories, growing like log₂. Far narrower than one-hot, but the columns mean nothing on their own.</span>`,
    };
    st.set(msgs[method], method === 'label' ? 'warn' : method === 'onehot' ? 'ok' : 'info');
    blurb.innerHTML =
      `<strong>${cats.length}</strong> distinct categories → <strong>${nCols}</strong> encoded column${nCols > 1 ? 's' : ''}. ` +
      `Keep adding new colours and watch one-hot grow linearly while binary grows logarithmically.`;
  }

  refresh();

  node.appendChild(note(
    `The choice matters most for <strong>high-cardinality</strong> features. Ten categories is a non-issue; ` +
    `ten thousand postcodes turns one-hot into a very wide, very sparse matrix. Add categories with the ` +
    `buttons and compare how each scheme's width responds — and remember that label encoding's invented ` +
    `ordering is harmless for tree models but actively misleading for linear and distance-based ones.`
  ));
});

/* ============================================================
   6. Discretization — three ways to cut a continuous variable
   ============================================================ */
defineWidget('discretization', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: 0, xmax: 100, ymin: 0, ymax: 1, aspect: 1.7, equal: false, pad: 0,
  }));

  const SCORES = [15, 22, 35, 41, 46, 53, 67, 72, 85, 93];
  let dataset = 'scores', k = 4, method = 'width';
  let values = SCORES;

  const dataCtl = segmented([
    { label: 'Exam scores (10)', value: 'scores' },
    { label: 'Skewed sample', value: 'skew' },
  ], { value: 'scores', label: 'Data', onChange: v => { dataset = v; rebuild(); } });
  const methodCtl = segmented([
    { label: 'Equal width', value: 'width' },
    { label: 'Equal frequency', value: 'freq' },
    { label: 'k-means', value: 'kmeans' },
  ], { value: 'width', label: 'Method', onChange: v => { method = v; refresh(); } });
  const kCtl = slider('Number of bins k', {
    min: 2, max: 6, step: 1, value: 4, format: v => String(v),
    onInput: v => { k = v; refresh(); },
  });
  const out = readout([['bin edges', 0], ['counts per bin', 0], ['bin widths', 0], ['within-bin variance', 0]]);
  const st = status('');
  right.append(dataCtl.root, methodCtl.root, kCtl.root, out.root, st.root);

  function rebuild() {
    if (dataset === 'scores') values = SCORES;
    else {
      const r = ML.rng(3);
      values = Array.from({ length: 60 }, () => Math.round(10 + Math.abs(ML.gauss(r)) ** 2 * 11));
    }
    refresh();
  }

  let binned = null;
  function refresh() {
    binned = method === 'width' ? ML.equalWidthBins(values, k)
           : method === 'freq' ? ML.equalFreqBins(values, k)
           : ML.kmeansBins(values, k);
    const counts = new Array(k).fill(0);
    binned.assign.forEach(a => counts[a]++);
    const widths = [];
    for (let i = 0; i < k; i++) widths.push(binned.edges[i + 1] - binned.edges[i]);

    let wcss = 0;
    for (let b = 0; b < k; b++) {
      const g = values.filter((_, i) => binned.assign[i] === b);
      if (g.length) { const m = ML.mean(g); wcss += g.reduce((s, v) => s + (v - m) ** 2, 0); }
    }

    out.set([
      binned.edges.map(e => fmt(e, 1)).join(', '),
      counts.join(', '),
      widths.map(w => fmt(w, 1)).join(', '),
      { html: fmt(wcss, 1), cls: method === 'kmeans' ? 'is-ok' : '' },
    ]);

    const equalCounts = counts.every(c => c === counts[0]);
    const equalWidths = widths.every(w => Math.abs(w - widths[0]) < 1e-6);
    const msgs = {
      width: `${ICON_INFO}<span><strong>Widths identical</strong> (${fmt(widths[0], 2)} each), counts uneven (${counts.join(', ')}). Simple and interpretable, but a bin can end up nearly empty.</span>`,
      freq: `${ICON_INFO}<span><strong>Counts as even as they can be</strong> (${counts.join(', ')}) — with ${values.length} points and ${k} bins, exactly equal is only possible when k divides ${values.length}. Widths now vary: ${widths.map(w => fmt(w, 1)).join(', ')}.</span>`,
      kmeans: `${ICON_OK}<span><strong>Lowest within-bin variance</strong> of the three (${fmt(wcss, 1)}). Cuts land in the natural gaps in the data rather than at arbitrary positions.</span>`,
    };
    st.set(msgs[method], method === 'kmeans' ? 'ok' : 'info');

    const lo = Math.min(...values), hi = Math.max(...values);
    const pad = (hi - lo) * .08;
    plot.setBounds({ xmin: lo - pad, xmax: hi + pad, ymin: 0, ymax: 1 });
  }

  plot.onDraw(p => {
    const cols = [C.c1, C.c2, C.c3, C.c5, C.c4, C.c6];
    // shaded bins
    for (let b = 0; b < k; b++) {
      const x0 = binned.edges[b], x1 = binned.edges[b + 1];
      p.ctx.fillStyle = cols[b % cols.length];
      p.ctx.globalAlpha = .13;
      p.ctx.fillRect(p.X(x0), p.Y(.95), p.px(x1 - x0), p.Y(0) - p.Y(.95));
      p.ctx.globalAlpha = 1;
      p.ctx.strokeStyle = cols[b % cols.length];
      p.ctx.lineWidth = 1.6;
      p.ctx.setLineDash([4, 4]);
      p.ctx.beginPath();
      p.ctx.moveTo(p.X(x1), p.Y(0)); p.ctx.lineTo(p.X(x1), p.Y(.95));
      p.ctx.stroke();
      p.ctx.setLineDash([]);
      const cnt = binned.assign.filter(a => a === b).length;
      p.text([(x0 + x1) / 2, .88], `bin ${b + 1} · n=${cnt}`,
        { align: 'center', size: 10.5, color: cols[b % cols.length], weight: 650 });
    }
    // the data points themselves, stacked where they collide
    const seen = {};
    values.forEach((v, i) => {
      const key = Math.round(v);
      seen[key] = (seen[key] || 0) + 1;
      const yy = .1 + (seen[key] - 1) * .055;
      p.dot([v, yy], { r: 5, color: cols[binned.assign[i] % cols.length], ring: true, ringLw: 2 });
    });
    if (method === 'kmeans' && binned.centers) {
      binned.centers.forEach((c, i) => {
        p.line([c, 0], [c, .78], { color: cols[i % cols.length], lw: 2 });
        p.badge([c, .8], `μ=${fmt(c, 1)}`, { color: cols[i % cols.length], align: 'center' });
      });
    }
    p.axes({ ticks: Math.max(5, Math.round((p.o.xmax - p.o.xmin) / 8)) });
  });

  rebuild();

  node.appendChild(note(
    `The same ten exam scores, cut three ways. <strong>Equal width</strong> gives tidy round intervals but ` +
    `lets counts drift; <strong>equal frequency</strong> balances the counts but distorts the widths; ` +
    `<strong>k-means</strong> puts the cuts where the data actually has gaps, which is why it achieves the ` +
    `lowest within-bin variance of the three. Switch to the skewed sample to see equal-width binning fail ` +
    `hardest — most points pile into one bin while the rest sit nearly empty.`
  ));
});

/* ============================================================
   7. PCA in 2D — drag the cloud, watch the components
   ============================================================ */
defineWidget('pca-explorer', node => {
  const { right, canvas } = split(node, { hint: 'Drag any point' });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -5, xmax: 5, ymin: -4, ymax: 4, aspect: 1.35, pad: 8,
  }));

  const r = ML.rng(21);
  let pts = Array.from({ length: 40 }, () => {
    const t = ML.gauss(r) * 1.9;
    return [t + ML.gauss(r) * .5, t * .55 + ML.gauss(r) * .5];
  });
  let showProj = true, showRecon = false;

  const projCtl = toggle('Show projections onto PC1', { value: true, onChange: v => { showProj = v; plot.render(); } });
  const reconCtl = toggle('Reconstruct from PC1 only', { value: false, onChange: v => { showRecon = v; plot.render(); } });
  const presets = el('div', { class: 'pg-actions' },
    button('Correlated', () => { regen((g) => { const t = g() * 1.9; return [t + g() * .5, t * .55 + g() * .5]; }); }),
    button('Uncorrelated', () => { regen((g) => [g() * 1.9, g() * 1.1]); }),
    button('Isotropic', () => { regen((g) => [g() * 1.4, g() * 1.4]); }),
  );
  const out = readout([['PC1 direction', 0], ['PC2 direction', 0], ['variance on PC1', 0], ['variance on PC2', 0], ['reconstruction error', 0]]);
  const st = status('');
  right.append(projCtl.root, reconCtl.root, presets, out.root, st.root);

  function regen(fn) {
    const g2 = ML.rng(Math.floor(Math.random() * 1e6));
    pts = Array.from({ length: 40 }, () => fn(() => ML.gauss(g2)));
    refresh();
  }

  const drag = new Dragger(plot);
  pts.forEach((_, i) => drag.add(() => pts[i], p => {
    pts[i] = [clamp(p[0], -4.6, 4.6), clamp(p[1], -3.6, 3.6)];
  }, { r: 11 }));
  drag.onchange = refresh;

  let model = null;
  function refresh() { model = ML.pca(pts); plot.render(); sync(); }

  function sync() {
    const [c1, c2] = model.components;
    const err = model.scores.reduce((s, sc) => s + sc[1] * sc[1], 0);
    out.set([
      `(${fmt(c1[0], 3)}, ${fmt(c1[1], 3)})`,
      c2 ? `(${fmt(c2[0], 3)}, ${fmt(c2[1], 3)})` : '—',
      `${fmt(model.ratio[0] * 100, 1)}%`,
      `${fmt((model.ratio[1] || 0) * 100, 1)}%`,
      fmt(err, 2),
    ]);
    const r0 = model.ratio[0];
    st.set(
      r0 > .93
        ? `${ICON_OK}<span><strong>${fmt(r0 * 100, 1)}%</strong> of the variance lives on one direction — this cloud is essentially one-dimensional, and dropping PC2 costs almost nothing.</span>`
        : r0 > .65
          ? `${ICON_INFO}<span>PC1 carries <strong>${fmt(r0 * 100, 1)}%</strong>. Useful compression, but PC2 still holds real structure.</span>`
          : `${ICON_WARN}<span>PC1 holds only <strong>${fmt(r0 * 100, 1)}%</strong>. The cloud has no dominant direction, so PCA cannot compress it — there is nothing to throw away.</span>`,
      r0 > .93 ? 'ok' : r0 > .65 ? 'info' : 'warn');
  }

  plot.onDraw(p => {
    p.grid(1); p.axes({ ticks: 1 });
    const mu = model.mu;
    const [c1, c2] = model.components;
    const sd1 = Math.sqrt(model.eigenvalues[0]);
    const sd2 = Math.sqrt(model.eigenvalues[1] || 0);

    if (showProj || showRecon) {
      model.scores.forEach((sc, i) => {
        const proj = [mu[0] + c1[0] * sc[0], mu[1] + c1[1] * sc[0]];
        if (showProj) p.line(pts[i], proj, { color: C.c4, lw: 1, dash: [3, 3], alpha: .6 });
        if (showRecon) p.dot(proj, { r: 3.2, color: C.c4, alpha: .9 });
      });
    }
    p.ray(mu, c1, { color: C.c1, lw: 1.3, dash: [7, 5], alpha: .45 });

    pts.forEach(q => p.dot(q, { r: 4, color: C.c5, alpha: .82 }));

    p.arrow(mu, [mu[0] + c1[0] * sd1 * 2, mu[1] + c1[1] * sd1 * 2], { color: C.c1, lw: 3.2 });
    if (c2) p.arrow(mu, [mu[0] + c2[0] * sd2 * 2, mu[1] + c2[1] * sd2 * 2], { color: C.c2, lw: 3.2 });
    p.dot(mu, { r: 5, color: C.c3, ring: true });
    p.badge([mu[0] + c1[0] * sd1 * 2, mu[1] + c1[1] * sd1 * 2],
      `PC1 · ${fmt(model.ratio[0] * 100, 0)}%`, { color: C.c1, align: 'center', dy: -16 });
    if (c2) p.badge([mu[0] + c2[0] * sd2 * 2, mu[1] + c2[1] * sd2 * 2],
      `PC2 · ${fmt((model.ratio[1] || 0) * 100, 0)}%`, { color: C.c2, align: 'center', dy: -16 });
  });

  refresh();

  node.appendChild(note(
    `PCA rotates the axes to line up with the directions the data actually spreads in. The arrows are the ` +
    `principal components, scaled by how much variance each one carries; the teal dot is the mean, which PCA ` +
    `always centres on first. Drag the points into a straight line and PC1 takes essentially 100% — the cloud ` +
    `is one-dimensional wearing two coordinates. Spread them into a circular blob and neither direction wins, ` +
    `which is PCA telling you honestly that there is nothing to compress.`
  ));
});

/* ============================================================
   8. Scree plot — choosing k, and why standardisation matters
   ============================================================ */
defineWidget('pca-scree', node => {
  const wrap = el('div');
  node.appendChild(wrap);

  const { rows, names } = ML.peopleTable({ n: 200, seed: 11 });
  let standardize = false, k = 2;

  const cvScree = el('canvas');
  const cvProj = el('canvas');
  const grid = el('div', { style: 'display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:1rem' },
    el('div', {}, el('div', { class: 'matrix-label', html: 'Explained variance per component' }),
      el('div', { class: 'pg-canvas-wrap' }, cvScree)),
    el('div', {}, el('div', { class: 'matrix-label', html: 'Data projected onto PC1 &amp; PC2' }),
      el('div', { class: 'pg-canvas-wrap' }, cvProj)),
  );

  const stdCtl = toggle('Standardise features first', { value: false, onChange: v => { standardize = v; refresh(); } });
  const kCtl = slider('Components kept k', {
    min: 1, max: 4, step: 1, value: 2, format: v => `${v} / 4`,
    onInput: v => { k = v; refresh(); },
  });
  const out = readout([['PC1', 0], ['PC2', 0], ['PC3', 0], ['PC4', 0], ['cumulative at k', 0]]);
  const st = status('');
  const controls = el('div', { class: 'pg-controls', style: 'margin-top:1.1rem' }, stdCtl.root, kCtl.root, out.root, st.root);
  wrap.append(grid, controls);

  const screePlot = trackPlot(new Plot(cvScree, { xmin: 0, xmax: 4, ymin: 0, ymax: 1.1, aspect: 1.15, equal: false, pad: 0 }));
  const projPlot = trackPlot(new Plot(cvProj, { xmin: -4, xmax: 4, ymin: -4, ymax: 4, aspect: 1.15, pad: 6 }));

  let model = null;
  function refresh() {
    model = ML.pca(rows, { standardize });
    const cum = model.ratio.reduce((acc, v, i) => { acc.push((acc[i - 1] || 0) + v); return acc; }, []);
    out.set([
      ...model.ratio.slice(0, 4).map(v => `${fmt(v * 100, 2)}%`),
      { html: `${fmt(cum[k - 1] * 100, 2)}%`, cls: cum[k - 1] > .9 ? 'is-ok' : '' },
    ]);
    st.set(
      standardize
        ? `${ICON_OK}<span>Standardised: every feature contributes on equal terms, so the components reflect <strong>correlation structure</strong> rather than units.</span>`
        : `${ICON_WARN}<span><strong>Unstandardised.</strong> PC1 takes ${fmt(model.ratio[0] * 100, 1)}% — but that is mostly the income column's sheer numeric size, not real structure. Toggle standardisation and watch the split change completely.</span>`,
      standardize ? 'ok' : 'warn');
    screePlot.render(); projPlot.render();
  }

  screePlot.onDraw(p => {
    p.grid(.25, { color: C.grid });
    const cum = [];
    let acc = 0;
    model.ratio.forEach((v, i) => { acc += v; cum.push([i + 1, acc]); });
    const bw = Math.min(p.px(.5), 40);
    model.ratio.forEach((v, i) => {
      const x = p.X(i + .5);
      p.ctx.fillStyle = i < k ? C.c1 : C.grid;
      p.ctx.fillRect(x - bw / 2, p.Y(v), bw, p.Y(0) - p.Y(v));
      p.text([i + .5, v], `${fmt(v * 100, 1)}%`, { align: 'center', dy: -9, size: 10, color: C.muted });
    });
    p.path([[0, 0], ...cum], { color: C.c2, lw: 2.4 });
    cum.forEach(([x, y]) => p.dot([x, y], { r: 3.4, color: C.c2 }));
    p.line([k, 0], [k, 1.1], { color: C.c4, lw: 1.6, dash: [5, 4] });
    p.line([0, .95], [4, .95], { color: C.c3, lw: 1.2, dash: [3, 4], alpha: .8 });
    p.text([3.9, .97], '95%', { align: 'right', size: 10, color: C.c3 });
    p.axes(); p.ticks(1);
    p.text({ px: 10, py: 11 }, 'bars = individual · line = cumulative', { color: C.muted, size: 10 });
  });

  projPlot.onDraw(p => {
    const sc = model.scores;
    const xs = sc.map(s => s[0]), ys = sc.map(s => s[1]);
    const mx = Math.max(...xs.map(Math.abs)) * 1.15 || 1;
    const my = Math.max(...ys.map(Math.abs)) * 1.15 || 1;
    p.o.xmin = -mx; p.o.xmax = mx; p.o.ymin = -my; p.o.ymax = my;
    p._computeScale();
    p.grid(mx / 4, { color: C.grid });
    p.axes();
    sc.forEach(s => p.dot([s[0], s[1]], { r: 3, color: C.c1, alpha: .6 }));
    p.text({ px: 10, py: 11 }, 'PC1 →', { color: C.muted, size: 10.5 });
    p.text({ px: 10, py: 28 }, '↑ PC2', { color: C.muted, size: 10.5 });
  });

  refresh();

  node.appendChild(note(
    `The scree plot answers "how many components do I keep?". Look for the <strong>elbow</strong> — the point ` +
    `after which bars stop dropping much — or pick k so the cumulative line clears a threshold such as 95%. ` +
    `<strong>Now toggle standardisation.</strong> On raw data PC1 looks overwhelmingly dominant, but that is ` +
    `an artefact of income being measured in thousands while age is measured in tens: PCA maximises variance, ` +
    `and variance is not unit-free. This is exactly why standardising before PCA is the default.`
  ));
});

/* ============================================================
   9. Feature selection — three filter criteria disagree
   ============================================================ */
defineWidget('feature-selection', node => {
  const wrap = el('div');
  node.appendChild(wrap);

  const r = ML.rng(99);
  const n = 240;
  const y = [], feats = { linear: [], nonlinear: [], noise: [], constant: [], redundant: [] };
  for (let i = 0; i < n; i++) {
    const cls = i % 2;
    y.push(cls);
    const lin = cls * 1.7 + ML.gauss(r) * 1.0;
    feats.linear.push(lin);
    feats.nonlinear.push((cls ? 1 : -1) * Math.abs(ML.gauss(r)) * 0 + (cls === 1 ? (ML.gauss(r) > 0 ? 2.4 : -2.4) : 0) + ML.gauss(r) * .45);
    feats.noise.push(ML.gauss(r) * 1.5);
    feats.constant.push(5 + ML.gauss(r) * 0.02);
    feats.redundant.push(lin * 0.98 + ML.gauss(r) * 0.06);
  }
  const names = ['Linear signal', 'Non-linear signal', 'Pure noise', 'Near-constant', 'Redundant copy'];
  const keys = ['linear', 'nonlinear', 'noise', 'constant', 'redundant'];

  let criterion = 'variance';
  const critCtl = segmented([
    { label: 'Variance', value: 'variance' },
    { label: '|Correlation|', value: 'corr' },
    { label: 'Mutual information', value: 'mi' },
  ], { value: 'variance', label: 'Ranking criterion', onChange: v => { criterion = v; refresh(); } });

  const cv = el('canvas');
  const left = el('div', {}, el('div', { class: 'pg-canvas-wrap' }, cv));
  const right = el('div', { class: 'pg-controls' }, critCtl.root);
  const st = status('');
  right.appendChild(st.root);
  wrap.appendChild(el('div', { class: 'pg-split pg-split--wide-ctrl' }, left, right));

  const plot = trackPlot(new Plot(cv, { xmin: 0, xmax: 1, ymin: 0, ymax: 5, aspect: 1.5, equal: false, pad: 0 }));

  let scores = [];
  function refresh() {
    scores = keys.map(kk => {
      const x = feats[kk];
      if (criterion === 'variance') return ML.variance(x, 0);
      if (criterion === 'corr') return Math.abs(ML.pearson(x, y));
      return ML.mutualInfo(x, y, 10);
    });
    const order = scores.map((s, i) => i).sort((a, b) => scores[b] - scores[a]);
    const top = names[order[0]];
    const msgs = {
      variance: `${ICON_WARN}<span>Variance ranks <strong>${top}</strong> first — but it never looks at the label at all. It can only tell you a feature is <em>constant</em>, not that it is <em>useful</em>. Notice pure noise scores well.</span>`,
      corr: `${ICON_INFO}<span>Correlation ranks <strong>${top}</strong> first. It finds the linear signal, but scores the non-linear feature near zero even though that feature perfectly separates the classes.</span>`,
      mi: `${ICON_OK}<span>Mutual information ranks <strong>${top}</strong> first and is the only one of the three that detects the <strong>non-linear</strong> feature, because it makes no assumption of linearity.</span>`,
    };
    st.set(msgs[criterion], criterion === 'mi' ? 'ok' : criterion === 'variance' ? 'warn' : 'info');
    plot.render();
  }

  plot.onDraw(p => {
    const mx = Math.max(...scores, 1e-9);
    p.o.xmin = 0; p.o.xmax = mx * 1.28; p.o.ymin = -.5; p.o.ymax = 5;
    p._computeScale();
    const cols = [C.c1, C.c3, C.c4, C.muted, C.c2];
    scores.forEach((s, i) => {
      const yy = 4.5 - i;
      const h = p.px(0);
      p.ctx.fillStyle = cols[i];
      p.ctx.globalAlpha = .85;
      const x0 = p.X(0), x1 = p.X(s);
      const yTop = p.Y(yy + .32), yBot = p.Y(yy - .32);
      p.ctx.fillRect(x0, yTop, x1 - x0, yBot - yTop);
      p.ctx.globalAlpha = 1;
      p.text({ px: Math.max(4, p.X(0) + 2), py: p.Y(yy + .44) }, names[i],
        { size: 11, weight: 620, color: C.ink, halo: true, haloWidth: 3.5 });
      p.text([s, yy], fmt(s, 3), { size: 11, dx: 8, color: C.muted, mono: true });
    });
    p.text({ px: 10, py: p.h - 4 },
      criterion === 'variance' ? 'variance (ignores the label)'
      : criterion === 'corr' ? '|Pearson correlation| with the label'
      : 'mutual information with the label (bits)',
      { color: C.muted, size: 10.5, baseline: 'bottom' });
  });

  refresh();

  node.appendChild(note(
    `Five features, one label, three criteria — and they rank them differently. ` +
    `<strong>Variance</strong> never sees the label, so pure noise looks excellent and only the near-constant ` +
    `feature is correctly rejected. <strong>Correlation</strong> finds linear relationships and misses the ` +
    `non-linear one entirely. <strong>Mutual information</strong> catches both. None of the three notices ` +
    `that "Redundant copy" duplicates "Linear signal" — filter methods score features ` +
    `<em>one at a time</em>, so redundancy is invisible to them. That blind spot is exactly what wrapper and ` +
    `embedded methods exist to fix.`
  ));
});
