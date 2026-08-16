/* ============================================================
   widgets/eval.js — interactive figures for Model Evaluation
   ============================================================ */
import {
  Plot, Dragger, C, el, slider, toggle, segmented, button,
  matrixInput, readout, status, defineWidget, canvasHost,
  trackPlot, clamp, fmt, round,
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

/* ---------- polynomial least squares, used by the fitting figures ---------- */
function polyfit(xs, ys, deg, ridge = 1e-9) {
  const m = deg + 1;
  const A = Array.from({ length: m }, () => new Array(m).fill(0));
  const b = new Array(m).fill(0);
  for (let i = 0; i < xs.length; i++) {
    const pw = [1];
    for (let d = 1; d < 2 * m; d++) pw.push(pw[d - 1] * xs[i]);
    for (let r = 0; r < m; r++) {
      for (let c = 0; c < m; c++) A[r][c] += pw[r + c];
      b[r] += pw[r] * ys[i];
    }
  }
  for (let r = 0; r < m; r++) A[r][r] += ridge;
  // Gaussian elimination with partial pivoting
  for (let i = 0; i < m; i++) {
    let piv = i;
    for (let r = i + 1; r < m; r++) if (Math.abs(A[r][i]) > Math.abs(A[piv][i])) piv = r;
    if (Math.abs(A[piv][i]) < 1e-14) continue;
    [A[i], A[piv]] = [A[piv], A[i]];
    [b[i], b[piv]] = [b[piv], b[i]];
    for (let r = 0; r < m; r++) {
      if (r === i) continue;
      const f = A[r][i] / A[i][i];
      for (let c = i; c < m; c++) A[r][c] -= f * A[i][c];
      b[r] -= f * b[i];
    }
  }
  return A.map((row, i) => (Math.abs(row[i]) < 1e-14 ? 0 : b[i] / row[i]));
}
const polyval = (coef, x) => coef.reduce((s, c, i) => s + c * x ** i, 0);

/* ============================================================
   1. Train / validation / test split
   ============================================================ */
defineWidget('data-split', node => {
  const wrap = el('div');
  node.appendChild(wrap);

  let train = 70, val = 15;
  const N = 1000;

  const cv = el('canvas');
  const left = el('div', {}, el('div', { class: 'pg-canvas-wrap' }, cv));
  const right = el('div', { class: 'pg-controls' });
  wrap.appendChild(el('div', { class: 'pg-split pg-split--wide-ctrl' }, left, right));

  const trCtl = slider('Training share', {
    min: 30, max: 90, step: 1, value: 70, format: v => `${v}%`,
    onInput: v => { train = v; if (train + val > 95) val = 95 - train; vaCtl.set(val); refresh(); },
  });
  const vaCtl = slider('Validation share', {
    min: 5, max: 40, step: 1, value: 15, format: v => `${v}%`,
    onInput: v => { val = v; if (train + val > 95) { train = 95 - val; trCtl.set(train); } refresh(); },
  });
  const out = readout([['training', 0], ['validation', 0], ['test', 0]]);
  const st = status('');
  right.append(trCtl.root, vaCtl.root, out.root, st.root);

  const plot = trackPlot(new Plot(cv, { xmin: 0, xmax: 100, ymin: 0, ymax: 26, aspect: 2.13, equal: false, pad: 0 }));

  function refresh() {
    const test = 100 - train - val;
    out.set([
      `${train}%  ·  ${Math.round(N * train / 100)} samples`,
      `${val}%  ·  ${Math.round(N * val / 100)} samples`,
      { html: `${test}%  ·  ${Math.round(N * test / 100)} samples`, cls: test < 10 ? 'is-warn' : '' },
    ]);
    st.set(
      test < 10
        ? `${ICON_WARN}<span>Only ${Math.round(N * test / 100)} test samples left. Your final performance estimate will be <strong>very noisy</strong> — a handful of examples either way swings the number.</span>`
        : train < 45
          ? `${ICON_WARN}<span>A small training set limits how much the model can learn, however generous the evaluation split is.</span>`
          : `${ICON_OK}<span>A reasonable balance. The test set is touched <strong>once</strong>, at the very end — reuse it for decisions and it silently becomes a validation set.</span>`,
      test < 10 || train < 45 ? 'warn' : 'ok');
    plot.render();
  }

  plot.onDraw(p => {
    const test = 100 - train - val;
    const segs = [
      { w: train, col: C.c1, label: 'Training', sub: 'fit the parameters' },
      { w: val, col: C.c2, label: 'Validation', sub: 'tune & select' },
      { w: test, col: C.c3, label: 'Test', sub: 'report once' },
    ];
    let x = 0;
    segs.forEach(s => {
      p.ctx.fillStyle = s.col;
      p.ctx.globalAlpha = .88;
      p.ctx.fillRect(p.X(x) + 1.5, p.Y(20), p.px(s.w) - 3, p.Y(9) - p.Y(20));
      p.ctx.globalAlpha = 1;
      if (s.w >= 8) {
        p.text([x + s.w / 2, 16.5], `${s.label}`, { align: 'center', size: 13, weight: 700, color: C.raised });
        p.text([x + s.w / 2, 13.4], `${s.w}%`, { align: 'center', size: 12, weight: 600, color: C.raised, mono: true });
        p.text([x + s.w / 2, 6], s.sub, { align: 'center', size: 10.5, color: C.muted });
      }
      x += s.w;
    });
    p.text({ px: 12, py: 11 }, `${N} samples`, { color: C.muted, size: 11 });
  });

  refresh();

  node.appendChild(note(
    `The split is a budget, and every share you give one set is taken from another. More training data means ` +
    `a better model but a noisier estimate of how good it is; more test data means a sharper estimate of a ` +
    `weaker model. The rule that matters most is not the ratio but the discipline: <strong>the test set is ` +
    `looked at once</strong>. Every time you check it and then change something, it quietly stops being a ` +
    `test set. When data is scarce, <a href="#kfold">k-fold cross-validation</a> gets you out of the trade-off.`
  ));
});

/* ============================================================
   2. Over- and under-fitting
   ============================================================ */
defineWidget('overfitting', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -.3, xmax: 1.3, ymin: -1.9, ymax: 1.9, aspect: 1.11, equal: false, pad: 0,
  }));

  let degree = 3, noise = .22, nTrain = 14;
  const truth = x => Math.sin(2 * Math.PI * x);

  const degCtl = slider('Model complexity (polynomial degree)', {
    min: 1, max: 13, step: 1, value: 3, format: v => String(v),
    onInput: v => { degree = v; refresh(); },
  });
  const noiseCtl = slider('Noise level', {
    min: 0, max: .5, step: .01, value: .22,
    onInput: v => { noise = v; rebuild(); },
  });
  const nCtl = slider('Training points', {
    min: 6, max: 40, step: 1, value: 14, format: v => String(v),
    onInput: v => { nTrain = v; rebuild(); },
  });
  const acts = el('div', { class: 'pg-actions' },
    button('Underfit', () => { degree = 1; degCtl.set(1); refresh(); }),
    button('Just right', () => { degree = 3; degCtl.set(3); refresh(); }),
    button('Overfit', () => { degree = 13; degCtl.set(13); refresh(); }),
    button('Resample', () => rebuild(Math.floor(Math.random() * 1e6))),
  );
  const out = readout([['degree', 0], ['parameters', 0], ['training error (MSE)', 0], ['validation error (MSE)', 0], ['gap', 0]]);
  const st = status('');
  right.append(degCtl.root, noiseCtl.root, nCtl.root, acts, out.root, st.root);

  let train = [], valid = [], coef = [], curveTrain = [], curveVal = [];

  function rebuild(seed = 314) {
    const r = ML.rng(seed);
    train = Array.from({ length: nTrain }, (_, i) => {
      const x = i / (nTrain - 1);
      return [x, truth(x) + ML.gauss(r) * noise];
    });
    valid = Array.from({ length: 40 }, () => {
      const x = r();
      return [x, truth(x) + ML.gauss(r) * noise];
    });
    // error as a function of degree, for the right-hand curve
    curveTrain = []; curveVal = [];
    for (let d = 1; d <= 13; d++) {
      const c = polyfit(train.map(p => p[0]), train.map(p => p[1]), d);
      curveTrain.push([d, mse(train, c)]);
      curveVal.push([d, mse(valid, c)]);
    }
    refresh();
  }
  const mse = (set, c) => set.reduce((s, [x, y]) => s + (y - polyval(c, x)) ** 2, 0) / set.length;

  function refresh() {
    coef = polyfit(train.map(p => p[0]), train.map(p => p[1]), degree);
    const tr = mse(train, coef), va = mse(valid, coef);
    out.set([
      String(degree), String(degree + 1),
      fmt(tr, 4),
      { html: fmt(va, 4), cls: va < .06 ? 'is-ok' : va > .2 ? 'is-warn' : '' },
      { html: fmt(va - tr, 4), cls: va - tr > .1 ? 'is-warn' : '' },
    ]);
    const best = curveVal.reduce((b, c) => (c[1] < b[1] ? c : b), curveVal[0]);
    st.set(
      degree <= 1
        ? `${ICON_WARN}<span><strong>Under-fitting.</strong> Both errors are high — the model is too rigid to represent the curve at all.</span>`
        : va - tr > .12
          ? `${ICON_WARN}<span><strong>Over-fitting.</strong> Training error is tiny (${fmt(tr, 4)}) while validation error is ${fmt(va, 4)}. The model has memorised the noise. Lowest validation error is at degree <strong>${best[0]}</strong>.</span>`
          : `${ICON_OK}<span><strong>Good fit.</strong> Training and validation error are close, so the model is capturing signal rather than noise.</span>`,
      degree <= 1 || va - tr > .12 ? 'warn' : 'ok');
    plot.render();
  }

  plot.onDraw(p => {
    p.grid(.25, { color: C.grid });
    p.axes({ ticks: .5 });
    p.fn(truth, { color: C.muted, lw: 1.8, dash: [6, 4], alpha: .7, from: 0, to: 1 });
    p.fn(x => polyval(coef, x), { color: C.c1, lw: 3, from: -.25, to: 1.25, samples: 500 });
    valid.forEach(([x, y]) => p.dot([x, y], { r: 3, color: C.c3, alpha: .45 }));
    train.forEach(([x, y]) => p.dot([x, y], { r: 4.6, color: C.c2, ring: true, ringLw: 2 }));

    // inset: error vs complexity
    const iw = Math.min(190, p.w * .42), ih = iw * .62;
    const ix = p.w - iw - 12, iy = 12;
    p.ctx.save();
    p.ctx.fillStyle = C.raised; p.ctx.globalAlpha = .93;
    p.ctx.fillRect(ix, iy, iw, ih);
    p.ctx.globalAlpha = 1;
    p.ctx.strokeStyle = C.grid; p.ctx.lineWidth = 1;
    p.ctx.strokeRect(ix, iy, iw, ih);
    const maxE = Math.max(.35, ...curveVal.map(c => Math.min(c[1], 1.2)));
    const px = d => ix + 8 + (iw - 16) * (d - 1) / 12;
    const py = e => iy + ih - 16 - (ih - 26) * Math.min(e, maxE) / maxE;
    const drawSeries = (arr, col) => {
      p.ctx.strokeStyle = col; p.ctx.lineWidth = 2; p.ctx.beginPath();
      arr.forEach(([d, e], i) => { const X = px(d), Y = py(e); i ? p.ctx.lineTo(X, Y) : p.ctx.moveTo(X, Y); });
      p.ctx.stroke();
    };
    drawSeries(curveTrain, C.c2);
    drawSeries(curveVal, C.c4);
    p.ctx.strokeStyle = C.c1; p.ctx.lineWidth = 1.6; p.ctx.setLineDash([4, 3]);
    p.ctx.beginPath(); p.ctx.moveTo(px(degree), iy + 6); p.ctx.lineTo(px(degree), iy + ih - 8); p.ctx.stroke();
    p.ctx.setLineDash([]);
    p.ctx.restore();
    p.text({ px: ix + 8, py: iy + 12 }, 'error vs complexity', { size: 10, color: C.muted });
    p.text({ px: ix + 8, py: iy + ih - 5 }, 'train', { size: 9.5, color: C.c2 });
    p.text({ px: ix + 46, py: iy + ih - 5 }, 'validation', { size: 9.5, color: C.c4 });

    p.text({ px: 12, py: 11 }, 'dashed grey: true function · orange: training points · teal: validation points',
      { color: C.muted, size: 10.5 });
  });

  rebuild();

  node.appendChild(note(
    `Drag the degree up. Training error falls monotonically — a more flexible model can always hug the ` +
    `training points harder — but validation error falls, bottoms out, then climbs. The inset makes that ` +
    `U-shape explicit, and its minimum is the complexity you actually want. ` +
    `<strong>Note what the training error alone would have told you: pick the most complex model.</strong> ` +
    `That is exactly the wrong answer, and it is why model selection is always done on validation error. ` +
    `Now raise the number of training points and watch the overfitting recede — more data buys you the ` +
    `right to use a more complex model.`
  ));
});

/* ============================================================
   3. K-fold cross-validation
   ============================================================ */
defineWidget('kfold', node => {
  const wrap = el('div');
  node.appendChild(wrap);

  let K = 5, activeFold = -1;
  const N = 20;

  const cv = el('canvas');
  const left = el('div', {}, el('div', { class: 'pg-canvas-wrap' }, cv));
  const right = el('div', { class: 'pg-controls' });
  wrap.appendChild(el('div', { class: 'pg-split pg-split--wide-ctrl' }, left, right));

  const kCtl = slider('Number of folds K', {
    min: 2, max: 10, step: 1, value: 5, format: v => String(v),
    onInput: v => { K = v; activeFold = -1; refresh(); },
  });
  const acts = el('div', { class: 'pg-actions' },
    button('▶ Step through folds', () => step()),
    button('Show all', () => { activeFold = -1; refresh(); }),
    button('K = N (leave-one-out)', () => { K = 10; kCtl.set(10); activeFold = -1; refresh(); }),
  );
  const out = readout([['folds', 0], ['train per round', 0], ['validate per round', 0], ['models trained', 0], ['each point validated', 0]]);
  const st = status('');
  right.append(kCtl.root, acts, out.root, st.root);

  const plot = trackPlot(new Plot(cv, { xmin: 0, xmax: 10, ymin: 0, ymax: 10, aspect: 1.23, equal: false, pad: 0 }));

  let timer = null;
  function step() {
    clearInterval(timer);
    activeFold = 0; refresh();
    timer = setInterval(() => {
      activeFold++;
      if (activeFold >= K) { clearInterval(timer); activeFold = -1; }
      refresh();
    }, 900);
  }

  function refresh() {
    const per = N / K;
    out.set([
      String(K),
      `${Math.round(N - per)} of ${N}`,
      `${Math.round(per)} of ${N}`,
      String(K),
      'exactly once',
    ]);
    st.set(
      K >= 10
        ? `${ICON_INFO}<span>Many folds means almost all data trains each model and the estimate has low bias — but you train <strong>${K} models</strong>, and the fold-to-fold results become noisy because each validation set is tiny.</span>`
        : K <= 2
          ? `${ICON_WARN}<span>With K = 2 each model sees only half the data, so every estimate is pessimistic.</span>`
          : `${ICON_OK}<span>K = 5 or 10 is the usual compromise: enough training data per model, enough folds to average over, and a manageable ${K}× training cost.</span>`,
      K >= 10 ? 'info' : K <= 2 ? 'warn' : 'ok');
    plot.render();
  }

  plot.onDraw(p => {
    const rows = K;
    const rowH = 7.5 / rows;          // band reserved above and below for the captions
    const boxW = 9.2 / K;
    for (let round = 0; round < rows; round++) {
      if (activeFold >= 0 && round !== activeFold) p.ctx.globalAlpha = .22;
      const y = 8.9 - round * rowH;
      for (let f = 0; f < K; f++) {
        const isVal = f === round;
        const x = .5 + f * boxW;
        p.ctx.fillStyle = isVal ? C.c2 : C.c1;
        p.ctx.globalAlpha = (activeFold >= 0 && round !== activeFold) ? .18 : (isVal ? .95 : .55);
        p.ctx.fillRect(p.X(x) + 2, p.Y(y), p.px(boxW) - 4, p.py(rowH) - 5);
        p.ctx.globalAlpha = 1;
        if (boxW > .9 && (activeFold < 0 || round === activeFold)) {
          p.text([x + boxW / 2, y - rowH / 2 + .08], isVal ? 'val' : 'train', {
            align: 'center', size: Math.min(11, p.px(boxW) * .16),
            weight: 650, color: isVal ? C.raised : C.raised,
          });
        }
      }
      if (activeFold < 0 || round === activeFold) {
        p.text([.42, y - rowH / 2 + .08], `${round + 1}`, {
          align: 'right', size: 11, weight: 700, color: C.muted,
        });
      }
      p.ctx.globalAlpha = 1;
    }
    p.text({ px: 12, py: 13 },
      activeFold >= 0 ? `round ${activeFold + 1} of ${K}` : `all ${K} rounds — the validation block rotates`,
      { color: C.muted, size: 11, baseline: 'middle' });
    p.text({ px: 12, py: p.h - 5 }, 'final score = average across all rounds', { color: C.muted, size: 10.5, baseline: 'bottom' });
  });

  refresh();

  node.appendChild(note(
    `Every point is used for training in K−1 rounds and for validation in exactly one. That means the ` +
    `performance estimate is averaged over the whole dataset rather than resting on one arbitrary split — ` +
    `which matters enormously when data is scarce and a single unlucky split could mislead you. ` +
    `The cost is that you train <strong>K models instead of one</strong>. Note also that folds should be ` +
    `<em>stratified</em> for classification, so each fold preserves the class balance.`
  ));
});

/* ============================================================
   4. The confusion matrix and everything derived from it
   ============================================================ */
defineWidget('confusion-matrix', node => {
  const wrap = el('div');
  node.appendChild(wrap);

  let M = { TP: 50, FN: 10, FP: 10, TN: 50 };

  const cv = el('canvas');
  const left = el('div', {}, el('div', { class: 'pg-canvas-wrap' }, cv));
  const right = el('div', { class: 'pg-controls' });
  wrap.appendChild(el('div', { class: 'pg-split pg-split--wide-ctrl' }, left, right));

  const inp = matrixInput(2, 2, [[M.TP, M.FN], [M.FP, M.TN]], {
    label: 'Counts &nbsp;<span style="font-weight:400;color:var(--ink-faint)">TP, FN / FP, TN</span>',
    step: 1,
    onInput: m => {
      M = { TP: Math.max(0, m[0][0]), FN: Math.max(0, m[0][1]), FP: Math.max(0, m[1][0]), TN: Math.max(0, m[1][1]) };
      refresh();
    },
  });
  const presets = el('div', { class: 'pg-actions' },
    button('Balanced', () => set(50, 10, 10, 50)),
    button('High precision', () => set(10, 40, 2, 48)),
    button('High recall', () => set(45, 5, 40, 10)),
    button('Always positive', () => set(50, 0, 50, 0)),
  );
  const out = readout([
    ['Accuracy', 0], ['Precision', 0], ['Recall (sensitivity)', 0],
    ['Specificity', 0], ['F₁', 0], ['total', 0],
  ]);
  const st = status('');
  right.append(inp.root, presets, out.root, st.root);

  function set(TP, FN, FP, TN) {
    M = { TP, FN, FP, TN };
    inp.set([[TP, FN], [FP, TN]]);
    refresh();
  }

  const plot = trackPlot(new Plot(cv, { xmin: 0, xmax: 10, ymin: 0, ymax: 8.6, aspect: 1.11, equal: false, pad: 0 }));

  let m = null;
  function refresh() {
    m = ML.classMetrics(M);
    out.set([
      fmt(m.accuracy, 4),
      { html: fmt(m.precision, 4), cls: m.precision > .8 ? 'is-ok' : m.precision < .4 ? 'is-warn' : '' },
      { html: fmt(m.recall, 4), cls: m.recall > .8 ? 'is-ok' : m.recall < .4 ? 'is-warn' : '' },
      fmt(m.specificity, 4),
      { html: fmt(m.f1, 4), cls: m.f1 > .8 ? 'is-ok' : '' },
      String(m.total),
    ]);
    const gap = Math.abs(m.precision - m.recall);
    st.set(
      M.FN === 0 && M.TN === 0
        ? `${ICON_WARN}<span>Predicting positive for everything gives <strong>perfect recall</strong> and tells you nothing. Recall alone is never enough.</span>`
        : gap > .4
          ? `${ICON_WARN}<span>Precision and recall are ${fmt(gap, 2)} apart. F₁ = ${fmt(m.f1, 3)} sits much closer to the <strong>smaller</strong> of the two — that is the harmonic mean doing its job.</span>`
          : `${ICON_OK}<span>Precision and recall are balanced, so F₁ (${fmt(m.f1, 3)}) is a fair summary.</span>`,
      (M.FN === 0 && M.TN === 0) || gap > .4 ? 'warn' : 'ok');
    plot.render();
  }

  plot.onDraw(p => {
    const cells = [
      { r: 0, c: 0, v: M.TP, label: 'True positive', col: C.c3, abbr: 'TP' },
      { r: 0, c: 1, v: M.FN, label: 'False negative', col: C.c4, abbr: 'FN' },
      { r: 1, c: 0, v: M.FP, label: 'False positive', col: C.c4, abbr: 'FP' },
      { r: 1, c: 1, v: M.TN, label: 'True negative', col: C.c3, abbr: 'TN' },
    ];
    const x0 = 2.4, y0 = 6.6, w = 3.2, h = 2.5;
    const mx = Math.max(...cells.map(c => c.v), 1);
    cells.forEach(cell => {
      const X = x0 + cell.c * w, Y = y0 - cell.r * h;
      const t = cell.v / mx;
      p.ctx.fillStyle = cell.col;
      p.ctx.globalAlpha = .16 + t * .62;
      p.ctx.fillRect(p.X(X), p.Y(Y), p.px(w) - 4, p.py(h) - 4);
      p.ctx.globalAlpha = 1;
      p.text([X + w / 2, Y - h / 2 + .42], String(cell.v),
        { align: 'center', size: 22, weight: 750, color: t > .55 ? C.raised : C.ink, mono: true });
      p.text([X + w / 2, Y - h / 2 - .3], cell.label,
        { align: 'center', size: 11, weight: 600, color: t > .55 ? C.raised : C.muted });
      p.text([X + .32, Y - .3], cell.abbr,
        { align: 'left', size: 11, weight: 700, color: t > .55 ? C.raised : C.muted, mono: true });
    });

    p.text([x0 + w, 7.9], 'PREDICTED', { align: 'center', size: 10.5, weight: 700, color: C.muted });
    p.text([x0 + w / 2, 7.4], 'positive', { align: 'center', size: 11, color: C.ink, weight: 600 });
    p.text([x0 + w * 1.5, 7.4], 'negative', { align: 'center', size: 11, color: C.ink, weight: 600 });
    p.ctx.save();
    p.ctx.translate(p.X(1.5), p.Y(y0 - h));
    p.ctx.rotate(-Math.PI / 2);
    p.ctx.font = `700 10.5px ${getComputedStyle(document.documentElement).getPropertyValue('--font-sans')}`;
    p.ctx.fillStyle = C.muted; p.ctx.textAlign = 'center';
    p.ctx.fillText('ACTUAL', 0, 0);
    p.ctx.restore();
    p.text([2.15, y0 - h / 2], 'positive', { align: 'right', size: 11, color: C.ink, weight: 600 });
    p.text([2.15, y0 - h * 1.5], 'negative', { align: 'right', size: 11, color: C.ink, weight: 600 });

    // which cells each metric reads
    p.text([x0 + w, 1.1],
      `Precision reads the left column · Recall reads the top row`,
      { align: 'center', size: 11, color: C.muted });
    p.text([x0 + w, .55],
      `Precision ${fmt(m.precision, 3)}   Recall ${fmt(m.recall, 3)}   F₁ ${fmt(m.f1, 3)}`,
      { align: 'center', size: 12, weight: 700, color: C.ink, mono: true });
  });

  refresh();

  node.appendChild(note(
    `Everything in classification evaluation comes out of these four numbers. <strong>Precision</strong> asks ` +
    `"of the things I flagged, how many were right?" — it reads the predicted-positive column. ` +
    `<strong>Recall</strong> asks "of the things that were actually positive, how many did I catch?" — it ` +
    `reads the actual-positive row. Try "Always positive": recall hits 1.0 and the model is worthless, which ` +
    `is why the two are always quoted together.`
  ));
});

/* ============================================================
   5. The decision threshold — precision/recall trade-off, ROC, PR
   ============================================================ */
defineWidget('threshold-roc', node => {
  const wrap = el('div');
  node.appendChild(wrap);

  let thr = .5, sep = 1.9, prevalence = .5;

  const cvDist = el('canvas'), cvRoc = el('canvas'), cvPr = el('canvas');
  const grid = el('div', { style: 'display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:1rem' },
    el('div', {}, el('div', { class: 'matrix-label', html: 'Score distributions &amp; threshold' }), el('div', { class: 'pg-canvas-wrap' }, cvDist)),
    el('div', {}, el('div', { class: 'matrix-label', html: 'ROC curve' }), el('div', { class: 'pg-canvas-wrap' }, cvRoc)),
    el('div', {}, el('div', { class: 'matrix-label', html: 'Precision–recall curve' }), el('div', { class: 'pg-canvas-wrap' }, cvPr)),
  );

  const thrCtl = slider('Decision threshold', {
    min: 0, max: 1, step: .005, value: .5, format: v => fmt(v, 3),
    onInput: v => { thr = v; refresh(); },
  });
  const sepCtl = slider('Class separability', {
    min: .2, max: 4, step: .05, value: 1.9,
    onInput: v => { sep = v; rebuild(); },
  });
  const prevCtl = slider('Positive class prevalence', {
    min: .05, max: .5, step: .01, value: .5, format: v => `${fmt(v * 100, 0)}%`,
    onInput: v => { prevalence = v; rebuild(); },
  });
  const out = readout([['TP / FP / FN / TN', 0], ['Precision', 0], ['Recall (TPR)', 0], ['False positive rate', 0], ['F₁', 0], ['AUC', 0]]);
  const st = status('');
  const controls = el('div', { class: 'pg-controls', style: 'margin-top:1.1rem' },
    thrCtl.root, sepCtl.root, prevCtl.root, out.root, st.root);
  wrap.append(grid, controls);

  const pDist = trackPlot(new Plot(cvDist, { xmin: 0, xmax: 1, ymin: 0, ymax: 1.1, aspect: 1.1, equal: false, pad: 0 }));
  const pRoc = trackPlot(new Plot(cvRoc, { xmin: -.04, xmax: 1.04, ymin: -.04, ymax: 1.04, aspect: 1.1, equal: false, pad: 0 }));
  const pPr = trackPlot(new Plot(cvPr, { xmin: -.04, xmax: 1.04, ymin: -.04, ymax: 1.04, aspect: 1.1, equal: false, pad: 0 }));

  let scores = [], labels = [], roc = [], pr = [], auc = 0;

  const sigmoid = z => 1 / (1 + Math.exp(-z));

  function rebuild() {
    const r = ML.rng(777);
    const n = 600;
    scores = []; labels = [];
    for (let i = 0; i < n; i++) {
      const pos = r() < prevalence;
      labels.push(pos ? 1 : 0);
      scores.push(clamp(sigmoid((pos ? sep : -sep) / 2 + ML.gauss(r)), 0.001, 0.999));
    }
    // sweep the threshold to build the curves
    roc = []; pr = [];
    const grid2 = [];
    for (let t = 0; t <= 1.0001; t += .005) grid2.push(t);
    grid2.forEach(t => {
      const c = confusionAt(t);
      const tpr = c.TP + c.FN ? c.TP / (c.TP + c.FN) : 0;
      const fpr = c.FP + c.TN ? c.FP / (c.FP + c.TN) : 0;
      const prec = c.TP + c.FP ? c.TP / (c.TP + c.FP) : 1;
      roc.push([fpr, tpr]);
      pr.push([tpr, prec]);
    });
    roc.sort((a, b) => a[0] - b[0]);
    auc = 0;
    for (let i = 1; i < roc.length; i++) {
      auc += (roc[i][0] - roc[i - 1][0]) * (roc[i][1] + roc[i - 1][1]) / 2;
    }
    refresh();
  }

  function confusionAt(t) {
    let TP = 0, FP = 0, FN = 0, TN = 0;
    for (let i = 0; i < scores.length; i++) {
      const pred = scores[i] >= t;
      if (labels[i] === 1) { if (pred) TP++; else FN++; }
      else { if (pred) FP++; else TN++; }
    }
    return { TP, FP, FN, TN };
  }

  function refresh() {
    const c = confusionAt(thr);
    const m = ML.classMetrics(c);
    const fpr = c.FP + c.TN ? c.FP / (c.FP + c.TN) : 0;
    out.set([
      `${c.TP} / ${c.FP} / ${c.FN} / ${c.TN}`,
      { html: fmt(m.precision, 4), cls: m.precision > .8 ? 'is-ok' : '' },
      { html: fmt(m.recall, 4), cls: m.recall > .8 ? 'is-ok' : '' },
      fmt(fpr, 4),
      fmt(m.f1, 4),
      { html: fmt(auc, 4), cls: auc > .85 ? 'is-ok' : auc < .65 ? 'is-warn' : '' },
    ]);
    st.set(
      thr > .85
        ? `${ICON_INFO}<span>A <strong>high threshold</strong> means you only flag cases you are confident about: precision rises, recall falls. Right for spam filters, where a false positive loses real mail.</span>`
        : thr < .15
          ? `${ICON_INFO}<span>A <strong>low threshold</strong> catches almost every positive at the cost of many false alarms. Right for disease screening, where a missed case is far worse than a follow-up test.</span>`
          : `${ICON_INFO}<span>Move the threshold and watch precision and recall move in <strong>opposite directions</strong>. The model has not changed — only where you cut it.</span>`,
      'info');
    pDist.render(); pRoc.render(); pPr.render();
  }

  function hist(vals, bins, lo, hi) {
    const h = new Array(bins).fill(0);
    vals.forEach(v => {
      const b = Math.min(bins - 1, Math.max(0, Math.floor((v - lo) / (hi - lo) * bins)));
      h[b]++;
    });
    return h;
  }

  pDist.onDraw(p => {
    const bins = 34;
    const neg = scores.filter((_, i) => labels[i] === 0);
    const pos = scores.filter((_, i) => labels[i] === 1);
    const hn = hist(neg, bins, 0, 1), hp = hist(pos, bins, 0, 1);
    const mx = Math.max(...hn, ...hp, 1);
    const bw = p.px(1 / bins);
    hn.forEach((cnt, i) => {
      const x = i / bins;
      p.ctx.fillStyle = C.c5; p.ctx.globalAlpha = .62;
      p.ctx.fillRect(p.X(x), p.Y(cnt / mx), bw - .6, p.Y(0) - p.Y(cnt / mx));
    });
    hp.forEach((cnt, i) => {
      const x = i / bins;
      p.ctx.fillStyle = C.c2; p.ctx.globalAlpha = .62;
      p.ctx.fillRect(p.X(x), p.Y(cnt / mx), bw - .6, p.Y(0) - p.Y(cnt / mx));
    });
    p.ctx.globalAlpha = 1;
    p.line([thr, 0], [thr, 1.06], { color: C.c4, lw: 2.4 });
    p.badge([thr, 1.03], fmt(thr, 2), { color: C.c4, align: 'center' });
    p.axes(); p.ticks(.25);
    p.text({ px: 8, py: 12 }, 'negatives', { color: C.c5, size: 10, weight: 650 });
    p.text({ px: 8, py: 25 }, 'positives', { color: C.c2, size: 10, weight: 650 });
    p.text({ px: p.w - 8, py: p.h - 4 }, 'flagged →', { color: C.muted, size: 10, align: 'right', baseline: 'bottom' });
  });

  pRoc.onDraw(p => {
    p.grid(.25, { color: C.grid });
    p.line([0, 0], [1, 1], { color: C.muted, lw: 1.2, dash: [4, 4], alpha: .6 });
    p.path(roc, { color: C.c1, lw: 2.6, fill: C.fill });
    const c = confusionAt(thr);
    const tpr = c.TP + c.FN ? c.TP / (c.TP + c.FN) : 0;
    const fpr = c.FP + c.TN ? c.FP / (c.FP + c.TN) : 0;
    p.dot([fpr, tpr], { r: 6, color: C.c4, ring: true });
    p.axes(); p.ticks(.5);
    p.text({ px: 8, py: 12 }, `AUC = ${fmt(auc, 3)}`, { color: C.c1, size: 11, weight: 700 });
    p.xlabel('false positive rate', { size: 10 });
  });

  pPr.onDraw(p => {
    p.grid(.25, { color: C.grid });
    p.line([0, prevalence], [1, prevalence], { color: C.muted, lw: 1.2, dash: [4, 4], alpha: .7 });
    p.text([.98, prevalence], 'random baseline', { align: 'right', dy: -8, size: 9.5, color: C.muted });
    p.path([...pr].sort((a, b) => a[0] - b[0]), { color: C.c3, lw: 2.6 });
    const c = confusionAt(thr);
    const rec = c.TP + c.FN ? c.TP / (c.TP + c.FN) : 0;
    const prec = c.TP + c.FP ? c.TP / (c.TP + c.FP) : 1;
    p.dot([rec, prec], { r: 6, color: C.c4, ring: true });
    p.axes(); p.ticks(.5);
    p.xlabel('recall', { size: 10 });
    p.text({ px: 8, py: 12 }, 'precision ↑', { size: 10, color: C.muted });
  });

  rebuild();

  node.appendChild(note(
    `A classifier does not output a class — it outputs a <strong>score</strong>. The class only appears once ` +
    `you pick a threshold, and that choice is yours, not the model's. Sliding the threshold traces out the ` +
    `whole ROC and precision–recall curves; <strong>AUC summarises the model across every threshold at once</strong>, ` +
    `which is why it is quoted when the operating point has not been decided yet. ` +
    `Now drop prevalence to 5%: the ROC curve barely moves, while the PR curve collapses. That is why ` +
    `<strong>precision–recall is the honest curve for imbalanced problems</strong> and ROC can flatter a ` +
    `model that is doing badly on the rare class.`
  ));
});

/* ============================================================
   6. Fβ — moving the weight between precision and recall
   ============================================================ */
defineWidget('fbeta', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: 0, xmax: 4.1, ymin: 0, ymax: 1.05, aspect: 1.31, equal: false, pad: 0,
  }));

  let P = .9, R = .5294, beta = 1;

  const pCtl = slider('Precision', { min: .02, max: 1, step: .005, value: .9, format: v => fmt(v, 3), onInput: v => { P = v; refresh(); } });
  const rCtl = slider('Recall', { min: .02, max: 1, step: .005, value: .5294, format: v => fmt(v, 3), onInput: v => { R = v; refresh(); } });
  const bCtl = slider('β', { min: .25, max: 4, step: .05, value: 1, format: v => fmt(v, 2), onInput: v => { beta = v; refresh(); } });
  const presets = el('div', { class: 'pg-actions' },
    button('β = 0.5 (precision)', () => { beta = .5; bCtl.set(.5); refresh(); }),
    button('β = 1 (balanced)', () => { beta = 1; bCtl.set(1); refresh(); }),
    button('β = 2 (recall)', () => { beta = 2; bCtl.set(2); refresh(); }),
    button("Notes' example 4", () => { P = .9; R = 45 / 85; pCtl.set(.9); rCtl.set(45 / 85); beta = 2; bCtl.set(2); refresh(); }),
  );
  const out = readout([
    ['arithmetic mean', 0], ['geometric mean', 0], ['harmonic mean = F₁', 0],
    ['F_β', 0], ['leans toward', 0],
  ]);
  const st = status('');
  right.append(pCtl.root, rCtl.root, bCtl.root, presets, out.root, st.root);

  const fb = (p, r, b) => {
    const b2 = b * b;
    return b2 * p + r ? (1 + b2) * p * r / (b2 * p + r) : 0;
  };

  function refresh() { plot.render(); sync(); }
  function sync() {
    const am = (P + R) / 2, gm = Math.sqrt(P * R), hm = fb(P, R, 1);
    const v = fb(P, R, beta);
    out.set([
      fmt(am, 4), fmt(gm, 4), fmt(hm, 4),
      { html: fmt(v, 4), cls: 'is-ok' },
      beta > 1.02 ? 'recall' : beta < .98 ? 'precision' : 'neither — balanced',
    ]);
    const weaker = P < R ? 'precision' : 'recall';
    const emphasised = beta > 1 ? 'recall' : beta < 1 ? 'precision' : null;
    let msg;
    if (!emphasised) {
      msg = `${ICON_INFO}<span>At β = 1 this is F₁, the harmonic mean — which always sits closer to the <strong>smaller</strong> of the two (${fmt(Math.min(P, R), 3)}) than the arithmetic mean would.</span>`;
    } else if (emphasised === weaker) {
      msg = `${ICON_WARN}<span>β = ${fmt(beta, 2)} weights <strong>${emphasised}</strong>, which is the weaker of your two numbers — so F_β (${fmt(v, 3)}) drops <strong>below</strong> F₁ (${fmt(hm, 3)}).</span>`;
    } else {
      msg = `${ICON_OK}<span>β = ${fmt(beta, 2)} weights <strong>${emphasised}</strong>, the stronger of the two — so F_β (${fmt(v, 3)}) rises above F₁ (${fmt(hm, 3)}).</span>`;
    }
    st.set(msg, !emphasised ? 'info' : emphasised === weaker ? 'warn' : 'ok');
  }

  plot.onDraw(p => {
    p.grid(.25, { color: C.grid });
    // F_beta as a function of beta
    const pts = [];
    for (let b = .25; b <= 4.001; b += .02) pts.push([b, fb(P, R, b)]);
    p.path(pts, { color: C.c1, lw: 3 });

    // asymptotes: as beta -> 0 F_beta -> precision, as beta -> inf -> recall
    p.line([0, P], [4.1, P], { color: C.c2, lw: 1.4, dash: [5, 4], alpha: .85 });
    p.line([0, R], [4.1, R], { color: C.c3, lw: 1.4, dash: [5, 4], alpha: .85 });
    p.badge([4.05, P], `precision ${fmt(P, 3)}`, { color: C.c2, align: 'right', dy: -12 });
    p.badge([4.05, R], `recall ${fmt(R, 3)}`, { color: C.c3, align: 'right', dy: 12 });

    const f1 = fb(P, R, 1);
    p.dot([1, f1], { r: 5, color: C.muted, ring: true });
    // when beta is at 1 the live badge below sits here too, so only draw one
    if (Math.abs(beta - 1) > .06) {
      p.badge([1, f1], `F₁ = ${fmt(f1, 3)}`, { color: C.muted, align: 'center', dy: -18 });
    }

    const v = fb(P, R, beta);
    p.line([beta, 0], [beta, v], { color: C.c4, lw: 1.6, dash: [4, 4] });
    p.dot([beta, v], { r: 7, color: C.c4, ring: true });
    p.badge([beta, v], `F_${fmt(beta, 2)} = ${fmt(v, 3)}`, { color: C.c4, align: 'center', dy: -20 });

    p.axes(); p.ticks(1);
    p.xlabel('β  (β<1 favours precision · β>1 favours recall)', { size: 10.5 });
    p.text({ px: 10, py: 30 }, 'Fβ', { size: 11, color: C.c1, weight: 700 });
  });

  refresh();

  node.appendChild(note(
    `F<sub>β</sub> interpolates between precision and recall: as β → 0 it converges on <strong>precision</strong>, ` +
    `and as β grows it converges on <strong>recall</strong>. The two dashed lines are those limits, and the ` +
    `curve always runs between them. ` +
    `<strong>The direction of travel is the thing to internalise:</strong> raising β pulls F<sub>β</sub> toward ` +
    `recall, so if recall is the <em>weaker</em> of your two numbers, F₂ will be <em>lower</em> than F₁, not ` +
    `higher. Press "Notes' example 4" to load exactly that case — precision 0.9, recall 0.529 — where ` +
    `F₁ = 0.667 and F₂ = 0.577.`
  ));
});

/* ============================================================
   7. MSE vs MAE — what one outlier does
   ============================================================ */
defineWidget('mse-mae', node => {
  const { right, canvas } = split(node, { hint: 'Drag any point', wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -.5, xmax: 5.5, ymin: -1.75, ymax: 15.5, aspect: 1.15, equal: false, pad: 0,
  }));

  const truth = [3, 0.5, 2, 7, 4];
  let pred = [2.5, 0, 2, 8, 4.2];
  let outlierMode = false;

  const acts = el('div', { class: 'pg-actions' },
    button('Reset', () => { pred = [2.5, 0, 2, 8, 4.2]; outlierMode = false; refresh(); }),
    button('Create an outlier', () => { pred = [2.5, 0, 2, 8, 4.2]; pred[4] = -0.5; outlierMode = true; refresh(); }),
  );
  const out = readout([
    ['errors', 0], ['MSE', 0], ['RMSE', 0], ['MAE', 0], ['largest single error', 0], ['its share of MSE', 0],
  ]);
  const st = status('');
  right.append(acts, out.root, st.root);

  const drag = new Dragger(plot);
  pred.forEach((_, i) => drag.add(() => [i, pred[i]], p => {
    pred[i] = clamp(round(p[1], 2), -0.9, 11.5);
  }, { r: 13 }));
  drag.onchange = refresh;

  function refresh() { plot.render(); sync(); }
  function sync() {
    const errs = truth.map((t, i) => pred[i] - t);
    const n = errs.length;
    const mse = errs.reduce((s, e) => s + e * e, 0) / n;
    const mae = errs.reduce((s, e) => s + Math.abs(e), 0) / n;
    const worst = errs.reduce((a, b) => (Math.abs(b) > Math.abs(a) ? b : a), 0);
    const share = mse * n > 1e-12 ? (worst * worst) / (mse * n) : 0;
    out.set([
      errs.map(e => fmt(e, 2)).join(', '),
      { html: fmt(mse, 4), cls: mse > 10 ? 'is-warn' : '' },
      fmt(Math.sqrt(mse), 4),
      fmt(mae, 4),
      fmt(worst, 3),
      { html: `${fmt(share * 100, 1)}%`, cls: share > .7 ? 'is-warn' : '' },
    ]);
    st.set(
      share > .8
        ? `${ICON_WARN}<span>A <strong>single point</strong> now accounts for ${fmt(share * 100, 1)}% of the total squared error. MSE has effectively stopped measuring the other four.</span>`
        : `${ICON_INFO}<span>Drag one prediction far from its target and watch MSE explode while MAE rises only in proportion. Squaring is what makes the difference.</span>`,
      share > .8 ? 'warn' : 'info');
  }

  plot.onDraw(p => {
    p.grid(1, { color: C.grid });
    p.axes(); p.ticks(1, { stepX: 1e9, stepY: 5 });
    truth.forEach((t, i) => {
      const e = pred[i] - t;
      // the squared error, drawn literally as a square
      const side = Math.abs(e);
      if (side > .05) {
        p.ctx.fillStyle = C.c4; p.ctx.globalAlpha = .13;
        const x0 = p.X(i), y0 = p.Y(Math.max(t, pred[i]));
        // deliberately a screen square — the area is what encodes e^2
        p.ctx.fillRect(x0, y0, p.px(side), p.px(side));
        p.ctx.globalAlpha = 1;
        p.ctx.strokeStyle = C.c4; p.ctx.lineWidth = 1; p.ctx.globalAlpha = .5;
        p.ctx.strokeRect(x0, y0, p.px(side), p.px(side));
        p.ctx.globalAlpha = 1;
      }
      p.line([i, t], [i, pred[i]], { color: C.c4, lw: 2, dash: [4, 3] });
      p.dot([i, t], { r: 5.5, color: C.c3, ring: true });
      p.handle([i, pred[i]], { color: C.c1, r: 6 });
      p.text([i, -1.25], `e=${fmt(pred[i] - t, 2)}`,
        { align: 'center', size: 10, color: C.muted, mono: true, halo: true, haloWidth: 3.5 });
    });
    p.title('teal = actual · violet = prediction (drag) · shaded square = squared error', { size: 10.5 });
  });

  refresh();

  node.appendChild(note(
    `The shaded squares are the squared errors, drawn at literal scale — which is the whole story. Double an ` +
    `error and its <em>area</em> quadruples, so one bad prediction can dominate the entire MSE while barely ` +
    `moving the MAE. That sensitivity is a feature when large errors are genuinely costly, and a bug when ` +
    `your data contains outliers you do not want the model chasing. One practical note: MSE is in ` +
    `<strong>squared units</strong>, so comparing its magnitude directly against MAE is meaningless — use ` +
    `<strong>RMSE</strong> if you want a number in the original units.`
  ));
});
