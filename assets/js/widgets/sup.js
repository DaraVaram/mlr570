/* ============================================================
   widgets/sup.js — Supervised learning: linear models & kernels
   ============================================================ */
import {
  Plot, Dragger, C, el, slider, toggle, segmented, button,
  matrixInput, readout, status, defineWidget, canvasHost,
  trackPlot, clamp, fmt, round,
} from '../viz.js';
import * as LA from '../linalg.js';
import * as ML from '../ml.js';

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

/* ---------- shared solvers ---------- */

/** Simplified SMO for a soft-margin SVM. Returns alphas and bias. */
function smo(K, y, Cpen, { tol = 1e-4, passes = 12, maxIter = 4000 } = {}) {
  const n = y.length;
  const a = new Array(n).fill(0);
  let b = 0, passCount = 0, iter = 0;
  const f = i => {
    let s = b;
    for (let j = 0; j < n; j++) if (a[j] !== 0) s += a[j] * y[j] * K[i][j];
    return s;
  };
  while (passCount < passes && iter++ < maxIter) {
    let changed = 0;
    for (let i = 0; i < n; i++) {
      const Ei = f(i) - y[i];
      if ((y[i] * Ei < -tol && a[i] < Cpen) || (y[i] * Ei > tol && a[i] > 0)) {
        const j = (i + 1 + Math.floor((iter * 7 + i * 13) % (n - 1))) % n;
        if (j === i) continue;
        const Ej = f(j) - y[j];
        const ai = a[i], aj = a[j];
        let L, H;
        if (y[i] !== y[j]) { L = Math.max(0, aj - ai); H = Math.min(Cpen, Cpen + aj - ai); }
        else { L = Math.max(0, ai + aj - Cpen); H = Math.min(Cpen, ai + aj); }
        if (L >= H - 1e-12) continue;
        const eta = 2 * K[i][j] - K[i][i] - K[j][j];
        if (eta >= -1e-12) continue;
        let ajNew = aj - y[j] * (Ei - Ej) / eta;
        ajNew = clamp(ajNew, L, H);
        if (Math.abs(ajNew - aj) < 1e-8) continue;
        const aiNew = ai + y[i] * y[j] * (aj - ajNew);
        const b1 = b - Ei - y[i] * (aiNew - ai) * K[i][i] - y[j] * (ajNew - aj) * K[i][j];
        const b2 = b - Ej - y[i] * (aiNew - ai) * K[i][j] - y[j] * (ajNew - aj) * K[j][j];
        a[i] = aiNew; a[j] = ajNew;
        b = (aiNew > 0 && aiNew < Cpen) ? b1 : (ajNew > 0 && ajNew < Cpen) ? b2 : (b1 + b2) / 2;
        changed++;
      }
    }
    passCount = changed === 0 ? passCount + 1 : 0;
  }
  return { a, b };
}

const KERNELS = {
  linear: { name: 'Linear', f: (a, b) => a[0] * b[0] + a[1] * b[1] },
  poly2:  { name: 'Polynomial d=2', f: (a, b) => (1 + a[0] * b[0] + a[1] * b[1]) ** 2 },
  poly3:  { name: 'Polynomial d=3', f: (a, b) => (1 + a[0] * b[0] + a[1] * b[1]) ** 3 },
  rbf:    { name: 'RBF', f: (a, b, g = 1) => Math.exp(-g * ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)) },
};

const DATASETS = {
  xor: () => {
    const pts = [], y = [];
    const r = ML.rng(4);
    for (let i = 0; i < 60; i++) {
      const qx = i % 2 ? 1 : -1, qy = (i >> 1) % 2 ? 1 : -1;
      pts.push([qx * (0.9 + ML.gauss(r) * .3), qy * (0.9 + ML.gauss(r) * .3)]);
      y.push(qx * qy > 0 ? 1 : -1);
    }
    return { pts, y, name: 'XOR' };
  },
  circle: () => {
    const pts = [], y = [];
    const r = ML.rng(9);
    for (let i = 0; i < 70; i++) {
      const inner = i % 2 === 0;
      const rad = inner ? 0.55 + Math.abs(ML.gauss(r)) * .18 : 1.55 + Math.abs(ML.gauss(r)) * .2;
      const t = r() * Math.PI * 2;
      pts.push([Math.cos(t) * rad, Math.sin(t) * rad]);
      y.push(inner ? 1 : -1);
    }
    return { pts, y, name: 'Circle' };
  },
  linear: () => {
    const pts = [], y = [];
    const r = ML.rng(15);
    for (let i = 0; i < 60; i++) {
      const cls = i % 2 ? 1 : -1;
      pts.push([ML.gauss(r) * .5 + cls * .85, ML.gauss(r) * .6 + cls * .55]);
      y.push(cls);
    }
    return { pts, y, name: 'Linearly separable' };
  },
  checker: () => {
    const pts = [], y = [];
    const r = ML.rng(21);
    for (let i = 0; i < 90; i++) {
      const x = (r() * 2 - 1) * 1.9, yy = (r() * 2 - 1) * 1.9;
      pts.push([x, yy]);
      y.push((Math.floor(x + 2) + Math.floor(yy + 2)) % 2 === 0 ? 1 : -1);
    }
    return { pts, y, name: 'Checkerboard' };
  },
};

/** Draw a decision function as a filled field plus its zero contour. */
function drawDecision(p, f, { iso = [0], step = 4 } = {}) {
  const { ctx } = p;
  const W = Math.ceil(p.w / step), H = Math.ceil(p.h / step);
  const vals = new Float64Array(W * H);
  let mx = 1e-9;
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
    const [x, y] = p.toWorld(i * step, j * step);
    const v = f(x, y);
    vals[j * W + i] = v;
    mx = Math.max(mx, Math.abs(v));
  }
  const c1 = hexRgb(C.c1), c4 = hexRgb(C.c4);
  ctx.save();
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
    const v = vals[j * W + i] / mx;
    const t = Math.min(1, Math.abs(v) ** 0.55);
    const col = v >= 0 ? c1 : c4;
    ctx.fillStyle = `rgba(${col[0]},${col[1]},${col[2]},${.06 + t * .3})`;
    ctx.fillRect(i * step, j * step, step + 1, step + 1);
  }
  ctx.restore();
  // zero contour by marching squares on the sampled grid
  ctx.save();
  ctx.strokeStyle = C.ink; ctx.lineWidth = 2.2;
  iso.forEach(level => {
    ctx.beginPath();
    for (let j = 0; j < H - 1; j++) for (let i = 0; i < W - 1; i++) {
      const a = vals[j * W + i], b2 = vals[j * W + i + 1],
            c = vals[(j + 1) * W + i + 1], d = vals[(j + 1) * W + i];
      const idx = (a > level ? 1 : 0) | (b2 > level ? 2 : 0) | (c > level ? 4 : 0) | (d > level ? 8 : 0);
      if (idx === 0 || idx === 15) continue;
      const L = (v0, v1, p0, p1) => p0 + (p1 - p0) * (level - v0) / ((v1 - v0) || 1e-9);
      const x0 = i * step, x1 = (i + 1) * step, y0 = j * step, y1 = (j + 1) * step;
      const bot = [L(a, b2, x0, x1), y0], rgt = [x1, L(b2, c, y0, y1)];
      const top = [L(d, c, x0, x1), y1], lft = [x0, L(a, d, y0, y1)];
      const seg = { 1:[bot,lft],2:[bot,rgt],3:[lft,rgt],4:[rgt,top],5:[bot,rgt],6:[bot,top],
                    7:[lft,top],8:[lft,top],9:[bot,top],10:[lft,bot],11:[rgt,top],12:[lft,rgt],
                    13:[bot,rgt],14:[bot,lft] }[idx];
      if (!seg) continue;
      ctx.moveTo(seg[0][0], seg[0][1]); ctx.lineTo(seg[1][0], seg[1][1]);
    }
    ctx.stroke();
  });
  ctx.restore();
}
function hexRgb(h) {
  h = (h || '').trim();
  if (!h.startsWith('#')) return [90, 55, 212];
  return h.length === 4
    ? h.slice(1).split('').map(c => parseInt(c + c, 16))
    : [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
}

/* ============================================================
   1. Linear separability — is this line a valid separator?
   ============================================================ */
defineWidget('linear-separability', node => {
  const { right, canvas } = split(node, { hint: 'Drag the line handles', wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -4.5, xmax: 4.5, ymin: -4.5, ymax: 3.5, aspect: 1.3, pad: 8,
  }));

  const DATA = [
    { x: [1, 1], y: 1 }, { x: [2, 2], y: 1 },
    { x: [1, -2], y: -1 }, { x: [-1, -3], y: -1 },
  ];
  let w = [1, 1], b = -1;

  const Min = matrixInput(1, 3, [[w[0], w[1], b]], {
    label: 'w₁, w₂, b', step: .5,
    onInput: m => { w = [m[0][0], m[0][1]]; b = m[0][2]; refresh(); },
  });
  const presets = el('div', { class: 'pg-actions' },
    button('x₁+x₂−1', () => set([1, 1], -1)),
    button('Perceptron result x₂=0', () => set([0, 3], 0)),
    button('A bad line', () => set([1, -1], 0)),
  );
  const table = el('div', { class: 'readout' });
  const st = status('');
  right.append(Min.root, presets, table, st.root);

  function set(nw, nb) { w = nw.slice(); b = nb; Min.set([[w[0], w[1], b]]); refresh(); }
  function refresh() { plot.render(); sync(); }

  function sync() {
    const rows = DATA.map(d => {
      const v = w[0] * d.x[0] + w[1] * d.x[1] + b;
      return { ...d, v, m: d.y * v };
    });
    const okAll = rows.every(r => r.m > 0);
    table.innerHTML =
      `<div style="color:var(--ink-faint);margin-bottom:.3em">point &nbsp; y &nbsp; wᵀx+b &nbsp; y(wᵀx+b)</div>` +
      rows.map(r =>
        `<div>(${r.x[0]}, ${r.x[1]})&nbsp; ${r.y > 0 ? '+1' : '−1'} &nbsp; ` +
        `${fmt(r.v, 2).padStart(6)} &nbsp; ` +
        `<span class="${r.m > 0 ? 'is-ok' : 'is-warn'}">${fmt(r.m, 2).padStart(6)}</span></div>`
      ).join('');
    st.set(
      okAll
        ? `${OK}<span>All four satisfy <strong>yᵢ(wᵀxᵢ + b) &gt; 0</strong>, so this line separates the data.</span>`
        : `${WARN}<span><strong>${rows.filter(r => r.m <= 0).length} point(s) on the wrong side.</strong> Separability requires every product to be positive.</span>`,
      okAll ? 'ok' : 'warn');
  }

  const drag = new Dragger(plot);
  const anchor = () => {
    const n2 = w[0] * w[0] + w[1] * w[1];
    return n2 < 1e-9 ? [0, 0] : [-w[0] * b / n2, -w[1] * b / n2];
  };
  drag.add(anchor, p => {
    b = round(-(w[0] * p[0] + w[1] * p[1]), 2);
    Min.set([[w[0], w[1], b]]);
  }, { r: 15 });
  drag.add(() => {
    const a = anchor(), n = Math.hypot(...w) || 1;
    return [a[0] + w[0] / n * 1.6, a[1] + w[1] / n * 1.6];
  }, p => {
    const a = anchor();
    const d = [p[0] - a[0], p[1] - a[1]];
    const n = Math.hypot(...d) || 1;
    const mag = Math.hypot(...w) || 1;
    w = [round(d[0] / n * mag, 2), round(d[1] / n * mag, 2)];
    b = round(-(w[0] * a[0] + w[1] * a[1]), 2);
    Min.set([[w[0], w[1], b]]);
  }, { r: 13 });
  drag.onchange = refresh;

  plot.onDraw(p => {
    drawDecision(p, (x, y) => w[0] * x + w[1] * y + b, { step: 6 });
    p.grid(1, { color: C.grid }); p.axes({ ticks: 1 });
    DATA.forEach(d => {
      const v = w[0] * d.x[0] + w[1] * d.x[1] + b;
      const good = d.y * v > 0;
      p.dot(d.x, { r: 8, color: d.y > 0 ? C.c1 : C.c4, ring: true, ringLw: 2.6 });
      if (!good) {
        p.ctx.strokeStyle = C.warn || C.c4;
        p.ctx.lineWidth = 2.4;
        const [sx, sy] = p.toScreen(d.x);
        p.ctx.beginPath(); p.ctx.arc(sx, sy, 14, 0, Math.PI * 2); p.ctx.stroke();
      }
      p.badge(d.x, `${d.y > 0 ? '+1' : '−1'}`, { color: d.y > 0 ? C.c1 : C.c4, align: 'center', dy: -19 });
    });
    // the weight vector, drawn from the line
    const a = anchor(), n = Math.hypot(...w) || 1;
    p.arrow(a, [a[0] + w[0] / n * 1.6, a[1] + w[1] / n * 1.6], { color: C.c3, lw: 2.6 });
    p.badge([a[0] + w[0] / n * 1.6, a[1] + w[1] / n * 1.6], 'w', { color: C.c3, align: 'center', dy: -16 });
    p.handle(a, { color: C.c2, r: 6 });
  });
  refresh();

  node.appendChild(note(
    `The vector <strong>w is perpendicular to the boundary</strong> — it points toward the +1 side, and ` +
    `<span class="u-mono">b</span> slides the line along it. The sign of <span class="u-mono">wᵀx + b</span> ` +
    `says which side a point is on, so multiplying by the true label <span class="u-mono">y</span> gives a ` +
    `single number that is positive exactly when the point is classified correctly. That is why the whole ` +
    `separability condition compresses to <strong>yᵢ(wᵀxᵢ + b) &gt; 0 for all i</strong>.`
  ));
});

/* ============================================================
   2. Perceptron — step through the updates
   ============================================================ */
defineWidget('perceptron', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -4.5, xmax: 4.5, ymin: -4.5, ymax: 3.5, aspect: 1.3, pad: 8,
  }));

  const NOTES_DATA = [
    { x: [1, 1], y: 1 }, { x: [2, 2], y: 1 },
    { x: [1, -2], y: -1 }, { x: [-1, -3], y: -1 },
  ];
  let data = NOTES_DATA.map(d => ({ ...d, x: d.x.slice() }));
  let eta = 1, trace = [], step = 0;

  const etaCtl = slider('Learning rate η', {
    min: .1, max: 2, step: .1, value: 1, onInput: v => { eta = v; rebuild(); },
  });
  const acts = el('div', { class: 'pg-actions' },
    button('◀', () => { step = Math.max(0, step - 1); refresh(); }),
    button('Step ▶', () => { step = Math.min(trace.length - 1, step + 1); refresh(); }),
    button('Run to end', () => { step = trace.length - 1; refresh(); }),
    button('Restart', () => { step = 0; refresh(); }),
  );
  const presets = el('div', { class: 'pg-actions' },
    button("Notes' dataset", () => { data = NOTES_DATA.map(d => ({ ...d, x: d.x.slice() })); rebuild(); }),
    button('Make it XOR (fails)', () => {
      data = [
        { x: [1, 1], y: 1 }, { x: [-1, -1], y: 1 },
        { x: [1, -1], y: -1 }, { x: [-1, 1], y: -1 },
      ];
      rebuild();
    }),
  );
  const log = el('div', { class: 'readout', style: 'min-height:6em' });
  const st = status('');
  right.append(etaCtl.root, acts, presets, log, st.root);

  function rebuild() {
    trace = [];
    let w = [0, 0], b = 0;
    trace.push({ w: w.slice(), b, desc: 'Initialise w = (0, 0), b = 0', i: -1, m: null, updated: false });
    let updates = 0;
    outer:
    for (let epoch = 0; epoch < 12; epoch++) {
      let clean = true;
      for (let i = 0; i < data.length; i++) {
        const d = data[i];
        const m = d.y * (w[0] * d.x[0] + w[1] * d.x[1] + b);
        if (m <= 0) {
          w = [w[0] + eta * d.y * d.x[0], w[1] + eta * d.y * d.x[1]];
          b = b + eta * d.y;
          clean = false; updates++;
          trace.push({ w: w.slice(), b, i, m, updated: true, epoch,
            desc: `(${d.x[0]}, ${d.x[1]}), y = ${d.y > 0 ? '+1' : '−1'}: margin ${fmt(m, 2)} ≤ 0 → update` });
        } else {
          trace.push({ w: w.slice(), b, i, m, updated: false, epoch,
            desc: `(${d.x[0]}, ${d.x[1]}), y = ${d.y > 0 ? '+1' : '−1'}: margin ${fmt(m, 2)} > 0 → no update` });
        }
        if (trace.length > 90) break outer;
      }
      if (clean) {
        trace.push({ w: w.slice(), b, i: -1, m: null, updated: false, converged: true,
          desc: `Full pass with no updates — converged after ${updates} updates` });
        break;
      }
    }
    step = 0;
    refresh();
  }

  function refresh() { plot.render(); sync(); }
  function sync() {
    const s = trace[step];
    const converged = trace.some(t => t.converged);
    log.innerHTML =
      `<div style="color:var(--ink-faint)">step ${step} / ${trace.length - 1}</div>` +
      `<div style="margin:.3em 0">${s.desc}</div>` +
      `<div>w = (${fmt(s.w[0], 2)}, ${fmt(s.w[1], 2)}) &nbsp; b = ${fmt(s.b, 2)}</div>`;
    if (s.converged || (converged && step === trace.length - 1)) {
      st.set(`${OK}<span><strong>Converged.</strong> Every point now satisfies y(wᵀx + b) &gt; 0.</span>`, 'ok');
    } else if (!converged && step === trace.length - 1) {
      st.set(`${WARN}<span><strong>Not converging.</strong> This data is not linearly separable, so the perceptron cycles forever — it has no stopping condition to reach.</span>`, 'warn');
    } else {
      st.set(`${INFO}<span>The perceptron updates <strong>only</strong> on misclassified points. Correctly classified points leave w and b untouched.</span>`, 'info');
    }
  }

  plot.onDraw(p => {
    const s = trace[step];
    if (Math.hypot(...s.w) > 1e-9) drawDecision(p, (x, y) => s.w[0] * x + s.w[1] * y + s.b, { step: 6 });
    p.grid(1, { color: C.grid }); p.axes({ ticks: 1 });
    data.forEach((d, i) => {
      const active = i === s.i;
      p.dot(d.x, { r: active ? 9 : 7, color: d.y > 0 ? C.c1 : C.c4, ring: true, ringLw: active ? 3.2 : 2 });
      if (active) {
        const [sx, sy] = p.toScreen(d.x);
        p.ctx.strokeStyle = C.c2; p.ctx.lineWidth = 2.4;
        p.ctx.beginPath(); p.ctx.arc(sx, sy, 15, 0, Math.PI * 2); p.ctx.stroke();
      }
    });
    if (Math.hypot(...s.w) > 1e-9) {
      const n = Math.hypot(...s.w);
      const a = [-s.w[0] * s.b / (n * n), -s.w[1] * s.b / (n * n)];
      p.arrow(a, [a[0] + s.w[0] / n * 1.5, a[1] + s.w[1] / n * 1.5], { color: C.c3, lw: 2.4 });
    }
    p.text({ px: 12, py: 11 }, `w = (${fmt(s.w[0], 2)}, ${fmt(s.w[1], 2)}),  b = ${fmt(s.b, 2)}`,
      { color: C.ink, size: 12, weight: 700, mono: true });
  });

  rebuild();

  node.appendChild(note(
    `Step through it. The highlighted point is the one being examined; the boundary only moves when that ` +
    `point is on the wrong side. On the notes' dataset it converges in a single pass to ` +
    `<span class="u-mono">w = (0, 3), b = 0</span> — the line x₂ = 0 — which is a <em>different</em> valid ` +
    `separator from the hand-picked <span class="u-mono">x₁ + x₂ − 1 = 0</span>. ` +
    `<strong>The perceptron has no unique solution</strong>: it stops at the first separator it stumbles ` +
    `into, which depends on initialisation, learning rate and the order of the data. Press "Make it XOR" ` +
    `to watch it fail to terminate at all.`
  ));
});

/* ============================================================
   3. SVM — margin, C, and support vectors
   ============================================================ */
defineWidget('svm-margin', node => {
  const { right, canvas } = split(node, { hint: 'Drag any point', wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -3.4, xmax: 4.2, ymin: -3, ymax: 3.6, aspect: 1.3, pad: 8,
  }));

  const NOTES_POS = [[2.0, 2.0], [3.0, 1.6], [2.6, 2.6], [3.0, 2.6], [2.0, 3.0]];
  const NOTES_NEG = [[-1.0, -1.0], [0.0, -2.0], [-2.0, 0.0], [-1.4, -1.2], [0.6, -1.6]];
  let pts = [...NOTES_POS.map(p => p.slice()), ...NOTES_NEG.map(p => p.slice())];
  let ys = [...NOTES_POS.map(() => 1), ...NOTES_NEG.map(() => -1)];
  let logC = 2;

  const cCtl = slider('Penalty C', {
    min: -2, max: 3, step: .1, value: 2,
    format: v => (10 ** v >= 100 ? fmt(10 ** v, 0) : fmt(10 ** v, 2)),
    onInput: v => { logC = v; refresh(); },
  });
  const acts = el('div', { class: 'pg-actions' },
    button('Reset points', () => {
      pts = [...NOTES_POS.map(p => p.slice()), ...NOTES_NEG.map(p => p.slice())];
      ys = [...NOTES_POS.map(() => 1), ...NOTES_NEG.map(() => -1)];
      rebind(); refresh();
    }),
    button('Add an outlier', () => { pts.push([-0.6, 1.6]); ys.push(-1); rebind(); refresh(); }),
  );
  const out = readout([['w', 0], ['b', 0], ['‖w‖', 0], ['margin 2/‖w‖', 0], ['support vectors', 0], ['margin violations', 0]]);
  const st = status('');
  right.append(cCtl.root, acts, out.root, st.root);

  const drag = new Dragger(plot);
  function rebind() {
    drag.clear();
    pts.forEach((_, i) => drag.add(() => pts[i], p => {
      pts[i] = [clamp(round(p[0], 2), -3.2, 4), clamp(round(p[1], 2), -2.8, 3.4)];
    }, { r: 12 }));
  }
  drag.onchange = refresh;
  rebind();

  let model = null;
  function refresh() {
    const Cp = 10 ** logC;
    const n = pts.length;
    const K = pts.map(a => pts.map(b2 => a[0] * b2[0] + a[1] * b2[1]));
    const { a, b } = smo(K, ys, Cp);
    const w = [0, 0];
    for (let i = 0; i < n; i++) { w[0] += a[i] * ys[i] * pts[i][0]; w[1] += a[i] * ys[i] * pts[i][1]; }
    const nw = Math.hypot(...w);
    const margins = pts.map((x, i) => ys[i] * (w[0] * x[0] + w[1] * x[1] + b));
    const sv = a.map((v, i) => (v > 1e-6 ? i : -1)).filter(i => i >= 0);
    const viol = margins.filter(m => m < 1 - 1e-6).length;
    model = { w, b, a, sv, margins, nw };

    out.set([
      `(${fmt(w[0], 3)}, ${fmt(w[1], 3)})`,
      fmt(b, 3),
      fmt(nw, 4),
      { html: nw > 1e-9 ? fmt(2 / nw, 4) : '∞', cls: 'is-ok' },
      String(sv.length),
      { html: String(viol), cls: viol ? 'is-warn' : 'is-ok' },
    ]);
    st.set(
      viol === 0
        ? `${OK}<span><strong>Hard margin achieved.</strong> Every point sits on or outside its margin. Only the ${sv.length} circled support vectors determine the boundary — drag any other point and nothing moves.</span>`
        : `${INFO}<span><strong>${viol} point(s) inside the margin.</strong> With C = ${fmt(10 ** logC, 2)} the model accepts these violations in exchange for a wider margin. Raise C to push them out.</span>`,
      viol === 0 ? 'ok' : 'info');
    plot.render();
  }

  plot.onDraw(p => {
    const { w, b, sv, margins, nw } = model;
    if (nw > 1e-9) {
      drawDecision(p, (x, y) => w[0] * x + w[1] * y + b, { step: 5 });
      // margin lines
      const dir = [-w[1] / nw, w[0] / nw];
      const base = [-w[0] * b / (nw * nw), -w[1] * b / (nw * nw)];
      [[-1, C.muted], [1, C.muted]].forEach(([off, col]) => {
        const shift = off / nw;
        const c = [base[0] + w[0] / nw * shift, base[1] + w[1] / nw * shift];
        p.ray(c, dir, { color: col, lw: 1.6, dash: [6, 5], alpha: .9 });
      });
      p.ray(base, dir, { color: C.ink, lw: 2.4 });
    }
    p.grid(1, { color: C.grid }); p.axes({ ticks: 1 });
    pts.forEach((x, i) => {
      const isSV = sv.includes(i);
      p.dot(x, { r: 6.5, color: ys[i] > 0 ? C.c1 : C.c4, ring: true, ringLw: 2 });
      if (isSV) {
        const [sx, sy] = p.toScreen(x);
        p.ctx.strokeStyle = C.c2; p.ctx.lineWidth = 2.6;
        p.ctx.beginPath(); p.ctx.arc(sx, sy, 13, 0, Math.PI * 2); p.ctx.stroke();
      }
    });
    p.text({ px: 12, py: 11 }, 'solid: boundary · dashed: ±1 margins · orange rings: support vectors',
      { color: C.muted, size: 10.5 });
  });

  refresh();

  node.appendChild(note(
    `The margin is <span class="u-mono">2/‖w‖</span>, so <em>maximising</em> the margin means ` +
    `<em>minimising</em> ‖w‖ — that is the whole objective. Two things worth doing: drag a point that is ` +
    `<strong>not</strong> circled and watch the boundary refuse to move (only support vectors matter), then ` +
    `press "Add an outlier" and sweep <strong>C</strong>. At large C the boundary contorts to accommodate ` +
    `the outlier; at small C it ignores it and keeps a wide margin. That is the bias–variance dial.`
  ));
});

/* ============================================================
   4. OLS, WLS and heteroscedasticity
   ============================================================ */
defineWidget('ols-wls', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -.3, xmax: 10.3, ymin: -4, ymax: 22, aspect: 1.45, equal: false, pad: 0,
  }));

  let hetero = 1.0, showWLS = true, showResid = false;

  const hCtl = slider('Heteroscedasticity', {
    min: 0, max: 2, step: .05, value: 1,
    format: v => v < .15 ? 'constant' : fmt(v, 2),
    onInput: v => { hetero = v; rebuild(); },
  });
  const wlsCtl = toggle('Show weighted least squares', { value: true, onChange: v => { showWLS = v; plot.render(); } });
  const resCtl = toggle('Show residuals', { value: false, onChange: v => { showResid = v; plot.render(); } });
  const acts = el('div', { class: 'pg-actions' }, button('Resample', () => rebuild(Math.floor(Math.random() * 1e6))));
  const out = readout([['OLS slope', 0], ['OLS intercept', 0], ['WLS slope', 0], ['WLS intercept', 0], ['true slope', 0], ['|OLS − true|', 0]]);
  const st = status('');
  right.append(hCtl.root, wlsCtl.root, resCtl.root, acts, out.root, st.root);

  const TRUE_A = 1.0, TRUE_B = 1.6;
  let data = [], ols = [0, 0], wls = [0, 0];

  function fit(X, y, W) {
    const n = X.length;
    let s00 = 0, s01 = 0, s11 = 0, t0 = 0, t1 = 0;
    for (let i = 0; i < n; i++) {
      const w = W ? W[i] : 1;
      s00 += w; s01 += w * X[i]; s11 += w * X[i] * X[i];
      t0 += w * y[i]; t1 += w * X[i] * y[i];
    }
    const det = s00 * s11 - s01 * s01;
    if (Math.abs(det) < 1e-12) return [0, 0];
    return [(s11 * t0 - s01 * t1) / det, (s00 * t1 - s01 * t0) / det]; // [intercept, slope]
  }

  function rebuild(seed = 77) {
    const r = ML.rng(seed);
    data = [];
    for (let i = 0; i < 60; i++) {
      const x = i / 59 * 10;
      const sd = 0.7 + hetero * x * 0.42;
      data.push({ x, y: TRUE_A + TRUE_B * x + ML.gauss(r) * sd, sd });
    }
    const X = data.map(d => d.x), Y = data.map(d => d.y);
    ols = fit(X, Y, null);
    wls = fit(X, Y, data.map(d => 1 / (d.sd * d.sd)));
    out.set([
      fmt(ols[1], 4), fmt(ols[0], 4),
      fmt(wls[1], 4), fmt(wls[0], 4),
      fmt(TRUE_B, 4),
      { html: fmt(Math.abs(ols[1] - TRUE_B), 4), cls: Math.abs(ols[1] - TRUE_B) > Math.abs(wls[1] - TRUE_B) ? 'is-warn' : '' },
    ]);
    st.set(
      hetero < .15
        ? `${OK}<span>With <strong>constant variance</strong> the two fits coincide — WLS reduces exactly to OLS when every weight is equal. Verify it in the numbers.</span>`
        : `${INFO}<span>Variance grows with x, so the noisiest points at the right dominate the OLS fit. WLS down-weights them by <span class="u-mono">1/σᵢ²</span> and recovers the slope more reliably.</span>`,
      hetero < .15 ? 'ok' : 'info');
    plot.render();
  }

  plot.onDraw(p => {
    p.grid(2, { color: C.grid });
    if (showResid) {
      data.forEach(d => {
        const yh = ols[0] + ols[1] * d.x;
        p.line([d.x, yh], [d.x, d.y], { color: C.c4, lw: 1, alpha: .5 });
      });
    }
    // the true ±2σ envelope
    const up = [], dn = [];
    for (let i = 0; i <= 60; i++) {
      const x = i / 60 * 10, sd = 0.7 + hetero * x * 0.42;
      up.push([x, TRUE_A + TRUE_B * x + 2 * sd]);
      dn.push([x, TRUE_A + TRUE_B * x - 2 * sd]);
    }
    p.path([...up, ...dn.reverse()], { fill: C.fill, stroke: false, close: true });
    data.forEach(d => p.dot([d.x, d.y], { r: 3, color: C.c5, alpha: .7 }));
    p.fn(x => TRUE_A + TRUE_B * x, { color: C.muted, lw: 1.6, dash: [6, 4], from: 0, to: 10 });
    p.fn(x => ols[0] + ols[1] * x, { color: C.c4, lw: 3, from: 0, to: 10 });
    if (showWLS) p.fn(x => wls[0] + wls[1] * x, { color: C.c3, lw: 3, from: 0, to: 10 });
    p.axes({ ticks: 2 });
    const key = [[C.muted, 'true line'], [C.c4, 'OLS'], ...(showWLS ? [[C.c3, 'WLS']] : [])];
    key.forEach(([col, lbl], i) => {
      p.ctx.strokeStyle = col; p.ctx.lineWidth = 2.6;
      p.ctx.beginPath(); p.ctx.moveTo(14, 16 + i * 15); p.ctx.lineTo(32, 16 + i * 15); p.ctx.stroke();
      p.text({ px: 37, py: 16 + i * 15 }, lbl, { color: C.muted, size: 10.5, weight: 600 });
    });
  });

  rebuild();

  node.appendChild(note(
    `OLS gives every observation the same say. When the noise variance grows with x — the shaded band ` +
    `fanning out — the points on the right are the least trustworthy and yet exert the most leverage on the ` +
    `slope. <strong>WLS weights each point by 1/σᵢ²</strong>, so noisy observations count for less. ` +
    `Slide heteroscedasticity to zero and watch the two lines converge exactly: with equal weights, ` +
    `<span class="u-mono">(XᵀWX)⁻¹XᵀWy</span> collapses to <span class="u-mono">(XᵀX)⁻¹Xᵀy</span>.`
  ));
});

/* ============================================================
   5. Ridge — multicollinearity and shrinkage
   ============================================================ */
defineWidget('ridge', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -3.2, xmax: 1.45, ymin: -2, ymax: 3, aspect: 1.5, equal: false, pad: 0,
  }));

  // the notes' worked example
  const X = [[1, 1, 1], [1, 1, 0.9], [1, 0.9, 1], [1, 0.9, 0.9]];
  const y = [2.8, 2.5, 2.6, 2.4];
  let logLam = -3;

  const lamCtl = slider('log₁₀ λ', {
    min: -6, max: 1, step: .05, value: -3,
    format: v => `λ = ${(10 ** v).toExponential(1)}`,
    onInput: v => { logLam = v; refresh(); },
  });
  const presets = el('div', { class: 'pg-actions' },
    button('λ → 0 (OLS)', () => { logLam = -6; lamCtl.set(-6); refresh(); }),
    button('λ = 1', () => { logLam = 0; lamCtl.set(0); refresh(); }),
  );
  const out = readout([['β₁', 0], ['β₂', 0], ['β₃', 0], ['‖β‖₂', 0], ['training RSS', 0], ['cond(XᵀX + λI)', 0]]);
  const st = status('');
  right.append(lamCtl.root, presets, out.root, st.root);

  const XtX = LA.matmul(LA.transpose(X), X);
  const Xty = LA.matvec(LA.transpose(X), y);

  function solveRidge(lam) {
    const A = XtX.map((r, i) => r.map((v, j) => v + (i === j ? lam : 0)));
    return LA.solve(A, Xty) || [0, 0, 0];
  }
  function condOf(lam) {
    const A = XtX.map((r, i) => r.map((v, j) => v + (i === j ? lam : 0)));
    const { s } = LA.svd(A);
    return s[0] / (s[s.length - 1] || 1e-12);
  }

  let beta = [0, 0, 0];
  function refresh() {
    const lam = 10 ** logLam;
    beta = solveRidge(lam);
    const pred = X.map(r => r.reduce((s, v, j) => s + v * beta[j], 0));
    const rss = pred.reduce((s, v, i) => s + (v - y[i]) ** 2, 0);
    out.set([
      fmt(beta[0], 4), fmt(beta[1], 4), fmt(beta[2], 4),
      fmt(Math.hypot(...beta), 4),
      fmt(rss, 6),
      { html: fmt(condOf(lam), 0), cls: condOf(lam) > 1000 ? 'is-warn' : 'is-ok' },
    ]);
    st.set(
      lam < 1e-4
        ? `${WARN}<span><strong>Essentially OLS.</strong> With near-identical predictors, XᵀX is nearly singular (condition number ${fmt(condOf(lam), 0)}) and the coefficients blow up into large values of opposite sign — even though all three predictors carry almost the same information.</span>`
        : `${OK}<span><strong>Shrunk and stabilised.</strong> Ridge pulls the three coefficients toward one another, which is the honest answer when the predictors are nearly interchangeable.</span>`,
      lam < 1e-4 ? 'warn' : 'ok');
    plot.render();
  }

  plot.onDraw(p => {
    p.grid(.5, { color: C.grid });
    // coefficient paths across lambda
    const cols = [C.c1, C.c2, C.c3];
    const paths = [[], [], []];
    for (let L = -6; L <= 1.001; L += .05) {
      const bb = solveRidge(10 ** L);
      for (let k = 0; k < 3; k++) paths[k].push([L, bb[k]]);
    }
    paths.forEach((pa, k) => {
      p.path(pa, { color: cols[k], lw: 2.6 });
      // Label at the low-lambda end, where the three paths are still apart.
      // At high lambda every coefficient is shrunk to ~0 and the labels would
      // land on top of one another. pa[0] is off-plot at log10(lambda) = -6,
      // so take the first point inside the drawn range.
      const end = pa.find(q => q[0] >= p.o.xmin) || pa[0];
      p.text({ px: clamp(p.X(end[0]) + 6, 4, p.w - 20), py: clamp(p.Y(end[1]) - 9, 14, p.h - 16) },
        `β${['₁', '₂', '₃'][k]}`,
        { color: cols[k], size: 11, weight: 700, halo: true, haloWidth: 3.5 });
    });
    p.line([logLam, -2], [logLam, 3], { color: C.c4, lw: 1.8, dash: [5, 4] });
    beta.forEach((b, k) => p.dot([logLam, b], { r: 5, color: cols[k], ring: true }));
    p.line([-3.2, 0], [1.1, 0], { color: C.axis, lw: 1.2 });
    p.axes(); p.ticks(1);
    p.xlabel('log₁₀ λ  (left = no regularisation)', { size: 10.5 });
    p.title('coefficient value', { size: 10.5 });
  });

  refresh();

  node.appendChild(note(
    `This is the notes' worked example: three predictors that are almost identical, so ` +
    `<span class="u-mono">XᵀX</span> is nearly singular. At the far left (λ → 0) the OLS solution is ` +
    `<strong>β ≈ (−1.225, 1.5, 2.5)</strong> — huge, with a sign flip, from data whose targets barely vary. ` +
    `Move λ right and the three paths collapse together: ridge recognises that the predictors carry the same ` +
    `information and splits the credit between them. Note the training RSS <em>rises</em> as λ grows — ` +
    `that is the bias you are buying variance reduction with.`
  ));
});

/* ============================================================
   6. Feature lift — a circle becomes linearly separable
   ============================================================ */
defineWidget('feature-lift', node => {
  const { right, canvas } = split(node, { hint: 'Drag to orbit', wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -2.8, xmax: 2.8, ymin: -2.2, ymax: 2.2, aspect: 1.4, pad: 6,
  }));

  const { pts, y } = DATASETS.circle();
  let lift = 0, yaw = 0.7, pitch = 0.5, plane = 1.2;

  const liftCtl = slider('Lift into the third dimension', {
    min: 0, max: 1, step: .01, value: 0,
    format: v => `${fmt(v * 100, 0)}%`,
    onInput: v => { lift = v; plot.render(); sync(); },
  });
  const planeCtl = slider('Separating plane height θ₃', {
    min: .2, max: 3, step: .05, value: 1.2,
    onInput: v => { plane = v; plot.render(); sync(); },
  });
  const out = readout([['φ(x)', 0], ['separable in ℝ²?', 0], ['separable in ℝ³?', 0], ['misclassified by the plane', 0]]);
  const st = status('');
  right.append(liftCtl.root, planeCtl.root, out.root, st.root);

  let dragging = false, last = null;
  canvas.addEventListener('pointerdown', e => { dragging = true; last = [e.clientX, e.clientY]; canvas.setPointerCapture?.(e.pointerId); });
  canvas.addEventListener('pointermove', e => {
    if (!dragging) return;
    yaw += (e.clientX - last[0]) * .01;
    pitch = clamp(pitch + (e.clientY - last[1]) * .01, -1.3, 1.3);
    last = [e.clientX, e.clientY]; plot.render();
  });
  const stop = e => { dragging = false; canvas.releasePointerCapture?.(e.pointerId); };
  canvas.addEventListener('pointerup', stop);
  canvas.addEventListener('pointercancel', stop);
  canvas.classList.add('is-grabbable');

  const project = ([x, yy, z]) => {
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const X = x * cy - yy * sy, T = x * sy + yy * cy;
    return [X, z * cp - T * sp];
  };

  function sync() {
    const wrong = pts.filter((q, i) => {
      const z = q[0] ** 2 + q[1] ** 2;
      const pred = z < plane ? 1 : -1;
      return pred !== y[i];
    }).length;
    out.set([
      '[x₁, x₂, x₁² + x₂²]',
      { html: 'no', cls: 'is-warn' },
      { html: 'yes', cls: 'is-ok' },
      { html: String(wrong), cls: wrong ? 'is-warn' : 'is-ok' },
    ]);
    st.set(
      lift < .1
        ? `${WARN}<span>Flat in ℝ², no straight line can separate a ring from its centre. Start lifting.</span>`
        : wrong === 0
          ? `${OK}<span>In ℝ³ a <strong>flat plane</strong> separates them perfectly — and projected back down to ℝ² that plane is a <strong>circle</strong>.</span>`
          : `${INFO}<span>${wrong} points on the wrong side — adjust the plane height θ₃.</span>`,
      lift < .1 ? 'warn' : wrong === 0 ? 'ok' : 'info');
  }

  plot.onDraw(p => {
    const scale = 0.62;
    const put = (q) => {
      const z = (q[0] ** 2 + q[1] ** 2) * lift * scale;
      return project([q[0], q[1], z - lift * 1.1]);
    };
    // ground grid
    for (let i = -2; i <= 2; i++) {
      p.line(project([i, -2, -lift * 1.1]), project([i, 2, -lift * 1.1]), { color: C.grid, lw: 1, alpha: .7 });
      p.line(project([-2, i, -lift * 1.1]), project([2, i, -lift * 1.1]), { color: C.grid, lw: 1, alpha: .7 });
    }
    // the separating plane, once lifted
    if (lift > .05) {
      const h = plane * lift * scale - lift * 1.1;
      const corners = [[-2.2, -2.2], [2.2, -2.2], [2.2, 2.2], [-2.2, 2.2]].map(([a, b2]) => project([a, b2, h]));
      p.polygon(corners, { fill: C.fill2, color: C.c2, lw: 1.8, alpha: .9 });
    }
    // vertical lift lines
    if (lift > .02) {
      pts.forEach(q => {
        p.line(project([q[0], q[1], -lift * 1.1]), put(q), { color: C.muted, lw: .8, alpha: .25 });
      });
    }
    pts.forEach((q, i) => p.dot(put(q), { r: 4, color: y[i] > 0 ? C.c1 : C.c4, alpha: .92 }));
    p.text({ px: 12, py: 11 }, 'φ(x) = [x₁, x₂, x₁² + x₂²]', { color: C.ink, size: 12, weight: 700 });
  });

  sync();

  node.appendChild(note(
    `The inner and outer rings cannot be split by any straight line in the plane. But add a third coordinate ` +
    `<span class="u-mono">x₁² + x₂²</span> — squared distance from the origin — and the inner points sink ` +
    `while the outer ones rise. Now a <strong>flat plane</strong> slices cleanly between them, and that plane ` +
    `pushed back down to two dimensions is exactly a circle. This is the entire idea behind kernels: ` +
    `<strong>the same linear machinery, applied in a space where the problem has become linear</strong>.`
  ));
});

/* ============================================================
   7. Kernel playground — the decision boundary each kernel gives
   ============================================================ */
defineWidget('kernel-playground', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -2.4, xmax: 2.4, ymin: -2.4, ymax: 2.4, aspect: 1.1, pad: 0,
  }));

  let dsKey = 'xor', kKey = 'rbf', gamma = 1, logC = 1;
  let ds = DATASETS[dsKey]();

  const dsCtl = segmented(
    Object.entries(DATASETS).map(([k, f]) => ({ label: f().name, value: k })),
    { value: dsKey, label: 'Dataset', onChange: v => { dsKey = v; ds = DATASETS[v](); refresh(); } });
  const kCtl = segmented(
    Object.entries(KERNELS).map(([k, v]) => ({ label: v.name, value: k })),
    { value: kKey, label: 'Kernel', onChange: v => { kKey = v; refresh(); } });
  const gCtl = slider('RBF γ', {
    min: .05, max: 8, step: .05, value: 1, onInput: v => { gamma = v; refresh(); },
  });
  const cCtl = slider('Penalty C', {
    min: -1, max: 3, step: .1, value: 1,
    format: v => fmt(10 ** v, 10 ** v < 10 ? 2 : 0),
    onInput: v => { logC = v; refresh(); },
  });
  const out = readout([['training accuracy', 0], ['support vectors', 0], ['kernel', 0]]);
  const st = status('');
  right.append(dsCtl.root, kCtl.root, gCtl.root, cCtl.root, out.root, st.root);

  let model = null;
  function refresh() {
    const kf = (a, b) => KERNELS[kKey].f(a, b, gamma);
    const K = ds.pts.map(a => ds.pts.map(b => kf(a, b)));
    const { a, b } = smo(K, ds.y, 10 ** logC, { passes: 8, maxIter: 2500 });
    const decide = (x, yv) => {
      let s = b;
      for (let i = 0; i < a.length; i++) if (a[i] > 1e-7) s += a[i] * ds.y[i] * kf(ds.pts[i], [x, yv]);
      return s;
    };
    const correct = ds.pts.filter((q, i) => Math.sign(decide(q[0], q[1])) === ds.y[i]).length;
    const svCount = a.filter(v => v > 1e-6).length;
    model = { decide, a, sv: a.map((v, i) => (v > 1e-6 ? i : -1)).filter(i => i >= 0) };
    const acc = correct / ds.pts.length;
    out.set([
      { html: `${fmt(acc * 100, 1)}%`, cls: acc > .95 ? 'is-ok' : acc < .8 ? 'is-warn' : '' },
      `${svCount} / ${ds.pts.length}`,
      KERNELS[kKey].name,
    ]);
    st.set(
      kKey === 'linear' && dsKey !== 'linear'
        ? `${WARN}<span>A linear kernel can only ever draw a <strong>straight line</strong>. On this dataset that caps accuracy near ${fmt(acc * 100, 0)}% no matter how you tune C.</span>`
        : kKey === 'rbf' && gamma > 5
          ? `${WARN}<span><strong>γ is very large</strong>, so each point's influence is tiny and local. The boundary fragments into islands around individual points — textbook overfitting.</span>`
          : acc > .95
            ? `${OK}<span>Clean separation with ${svCount} support vectors. Only those points shape the boundary.</span>`
            : `${INFO}<span>Try a different kernel, or tune γ and C.</span>`,
      kKey === 'linear' && dsKey !== 'linear' ? 'warn' : (kKey === 'rbf' && gamma > 5) ? 'warn' : acc > .95 ? 'ok' : 'info');
    plot.render();
  }

  plot.onDraw(p => {
    drawDecision(p, model.decide, { step: 4 });
    p.axes();
    ds.pts.forEach((q, i) => {
      p.dot(q, { r: 4.6, color: ds.y[i] > 0 ? C.c1 : C.c4, ring: true, ringLw: 1.6 });
      if (model.sv.includes(i)) {
        const [sx, sy] = p.toScreen(q);
        p.ctx.strokeStyle = C.c2; p.ctx.lineWidth = 1.8;
        p.ctx.beginPath(); p.ctx.arc(sx, sy, 9, 0, Math.PI * 2); p.ctx.stroke();
      }
    });
  });

  refresh();

  node.appendChild(note(
    `Four datasets, four kernels, one algorithm. The <strong>linear</strong> kernel can only draw a straight ` +
    `line and fails on every non-linear set. The <strong>polynomial</strong> kernel bends globally — it fits ` +
    `XOR and the circle, but on the checkerboard it must distort distant regions to fix local ones. The ` +
    `<strong>RBF</strong> kernel is local: each support vector influences only its neighbourhood, which is ` +
    `why it handles the checkerboard. Then push γ above 5 and watch it shatter into islands — locality ` +
    `taken too far is memorisation.`
  ));
});

/* ============================================================
   8. Polynomial kernel expansion
   ============================================================ */
defineWidget('kernel-expansion', node => {
  const wrap = el('div');
  node.appendChild(wrap);

  let n = 2, d = 2;
  const nCtl = slider('Input dimension n', {
    min: 1, max: 100, step: 1, value: 2, format: v => String(v),
    onInput: v => { n = v; refresh(); },
  });
  const dCtl = slider('Degree d', {
    min: 1, max: 6, step: 1, value: 2, format: v => String(v),
    onInput: v => { d = v; refresh(); },
  });
  const expand = el('div', { class: 'readout', style: 'margin-top:.8rem' });
  const out = readout([['features |φ(x)|', 0], ['kernel cost', 0], ['explicit cost', 0], ['speed-up', 0]]);
  const st = status('');
  wrap.append(el('div', { class: 'pg-controls' }, nCtl.root, dCtl.root, out.root, st.root), expand);

  const choose = (a, b) => {
    let r = 1;
    for (let i = 0; i < b; i++) r = r * (a - i) / (i + 1);
    return Math.round(r);
  };

  function refresh() {
    const dim = choose(n + d, d);
    out.set([
      dim.toLocaleString(),
      `O(n) = ${n} multiplications`,
      `O(|φ|) = ${dim.toLocaleString()}`,
      { html: `${fmt(dim / n, 0)}×`, cls: dim / n > 100 ? 'is-ok' : '' },
    ]);
    st.set(
      dim > 5000
        ? `${OK}<span>Building φ(x) explicitly would need <strong>${dim.toLocaleString()} numbers per sample</strong>. The kernel gets the same answer with ${n} multiplications and one power. <em>This is why the kernel trick exists.</em></span>`
        : `${INFO}<span>|φ(x)| = C(n+d, d) = ${dim}. Raise n or d and watch it explode.</span>`,
      dim > 5000 ? 'ok' : 'info');

    if (n === 2 && d === 2) {
      expand.innerHTML =
        `<div style="color:var(--ink-faint);margin-bottom:.35em">K(x, x′) = (1 + x₁x′₁ + x₂x′₂)² expands to</div>` +
        `<div>1 + 2x₁x′₁ + 2x₂x′₂ + 2x₁x₂x′₁x′₂ + x₁²x′₁² + x₂²x′₂²</div>` +
        `<div style="color:var(--ink-faint);margin:.5em 0 .35em">which is exactly φ(x)ᵀφ(x′) with</div>` +
        `<div>φ(x) = [ 1, √2·x₁, √2·x₂, √2·x₁x₂, x₁², x₂² ]ᵀ &nbsp;&nbsp;<span style="color:var(--ok)">← 6 = C(4,2) ✓</span></div>` +
        `<div style="color:var(--ink-faint);margin-top:.5em">A coefficient of 2 splits as √2 on each side, so the inner product reproduces it exactly.</div>`;
    } else {
      expand.innerHTML =
        `<div style="color:var(--ink-faint)">Explicit expansion shown for n = 2, d = 2. ` +
        `Here φ(x) would have <strong>${dim.toLocaleString()}</strong> components — ` +
        `every monomial x₁^{j₁}⋯x_n^{j_n} with total degree ≤ ${d}, each scaled by the square root of its multinomial coefficient.</div>`;
    }
  }
  refresh();

  node.appendChild(note(
    `The kernel <span class="u-mono">(1 + xᵀx′)^d</span> computes an inner product in a space of ` +
    `<span class="u-mono">C(n+d, d)</span> dimensions — <strong>without ever building a vector in that space</strong>. ` +
    `At n = 100, d = 3 that is 176,851 features replaced by 100 multiplications and one cube. The saving is ` +
    `not an optimisation detail; it is what makes the method possible at all.`
  ));
});

/* ============================================================
   9. Kernel ridge regression
   ============================================================ */
defineWidget('kernel-regression', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -.4, xmax: 6.7, ymin: -2.2, ymax: 2.2, aspect: 1.5, equal: false, pad: 0,
  }));

  let kKey = 'rbf', gamma = 1.4, logLam = -2, noise = .18;
  const truth = x => Math.sin(x);

  const kCtl = segmented([
    { label: 'Linear (OLS)', value: 'linear' },
    { label: 'Polynomial d=3', value: 'poly3' },
    { label: 'RBF', value: 'rbf' },
  ], { value: kKey, label: 'Kernel', onChange: v => { kKey = v; refresh(); } });
  const gCtl = slider('RBF γ', { min: .05, max: 8, step: .05, value: 1.4, onInput: v => { gamma = v; refresh(); } });
  const lamCtl = slider('Regularisation log₁₀ λ', {
    min: -6, max: 1, step: .1, value: -2,
    format: v => (10 ** v).toExponential(1),
    onInput: v => { logLam = v; refresh(); },
  });
  const nCtl = slider('Noise', { min: 0, max: .5, step: .01, value: .18, onInput: v => { noise = v; rebuild(); } });
  const out = readout([['training MSE', 0], ['test MSE (held out)', 0], ['λ', 0]]);
  const st = status('');
  right.append(kCtl.root, gCtl.root, lamCtl.root, nCtl.root, out.root, st.root);

  let train = [], test = [], predict = null;

  function rebuild(seed = 31) {
    const r = ML.rng(seed);
    train = []; test = [];
    for (let i = 0; i < 26; i++) {
      const x = i / 25 * 6.3;
      train.push([x, truth(x) + ML.gauss(r) * noise]);
    }
    for (let i = 0; i < 60; i++) {
      const x = r() * 6.3;
      test.push([x, truth(x) + ML.gauss(r) * noise]);
    }
    refresh();
  }

  function refresh() {
    const kf = (a, b) => KERNELS[kKey].f([a, 0], [b, 0], gamma);
    const n = train.length;
    const K = train.map(a => train.map(b => kf(a[0], b[0])));
    const lam = 10 ** logLam;
    const A = K.map((row, i) => row.map((v, j) => v + (i === j ? lam : 0)));
    const alpha = LA.solve(A, train.map(t => t[1])) || new Array(n).fill(0);
    predict = x => train.reduce((s, t, i) => s + alpha[i] * kf(t[0], x), 0);

    const trMse = train.reduce((s, t) => s + (predict(t[0]) - t[1]) ** 2, 0) / n;
    const teMse = test.reduce((s, t) => s + (predict(t[0]) - t[1]) ** 2, 0) / test.length;
    out.set([
      fmt(trMse, 5),
      { html: fmt(teMse, 5), cls: teMse < .06 ? 'is-ok' : teMse > .2 ? 'is-warn' : '' },
      lam.toExponential(1),
    ]);
    st.set(
      kKey === 'linear'
        ? `${WARN}<span>A linear kernel fits a <strong>straight line</strong> to a sine wave. Training and test error are both high — this is under-fitting, and no amount of λ tuning will help.</span>`
        : trMse < 1e-4 && teMse > trMse * 20
          ? `${WARN}<span><strong>Over-fitting.</strong> Training error is near zero while test error is ${fmt(teMse, 3)}. The model is interpolating the noise. Raise λ or lower γ.</span>`
          : teMse < .06
            ? `${OK}<span>Good fit — training and test error are close, so the model has learned the signal rather than the sample.</span>`
            : `${INFO}<span>Adjust γ and λ and watch the training/test gap.</span>`,
      kKey === 'linear' ? 'warn' : (trMse < 1e-4 && teMse > trMse * 20) ? 'warn' : teMse < .06 ? 'ok' : 'info');
    plot.render();
  }

  plot.onDraw(p => {
    p.grid(1, { color: C.grid });
    p.fn(truth, { color: C.muted, lw: 1.8, dash: [6, 4], from: 0, to: 6.4 });
    test.forEach(t => p.dot(t, { r: 2.4, color: C.c3, alpha: .35 }));
    train.forEach(t => p.dot(t, { r: 4.2, color: C.c5, ring: true, ringLw: 1.6 }));
    p.fn(predict, { color: C.c1, lw: 3, from: -.3, to: 6.6, samples: 420 });
    p.axes({ ticks: 1 });
    p.title('dashed: true sin(x) · violet: kernel fit · faint: held-out test points',
      { color: C.muted, size: 10.5 });
  });

  rebuild();

  node.appendChild(note(
    `Kernel ridge regression solves <span class="u-mono">α = (K + λI)⁻¹y</span> and predicts with ` +
    `<span class="u-mono">ŷ = Kα</span> — the same least-squares idea, run in feature space, with λ doing ` +
    `two jobs at once: it controls overfitting <em>and</em> guarantees the matrix is invertible. ` +
    `Set γ high and λ tiny to make it thread every training point exactly, then watch the held-out error ` +
    `climb while the training error goes to zero.`
  ));
});

/* ============================================================
   10. k-NN — k, distance metric, and the elbow
   ============================================================ */
defineWidget('knn', node => {
  const { right, canvas } = split(node, { hint: 'Click to place a query point', wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -2.6, xmax: 2.6, ymin: -2.6, ymax: 2.6, aspect: 1.1, pad: 0,
  }));

  let k = 5, metric = 'l2', weighted = false, showRegions = true;
  let query = [0.4, 0.5];
  const r = ML.rng(66);
  const pts = [], ys = [];
  for (let i = 0; i < 60; i++) {
    const cls = i % 2 ? 1 : -1;
    pts.push([ML.gauss(r) * .72 + cls * .78, ML.gauss(r) * .72 + cls * .62]);
    ys.push(cls);
  }
  // a few label-noise points, so k actually matters
  [4, 17, 33, 48].forEach(i => { ys[i] = -ys[i]; });

  const kCtl = slider('k', { min: 1, max: 25, step: 1, value: 5, format: v => String(v), onInput: v => { k = v; refresh(); } });
  const mCtl = segmented([
    { label: 'Euclidean', value: 'l2' }, { label: 'Manhattan', value: 'l1' },
  ], { value: 'l2', label: 'Distance', onChange: v => { metric = v; refresh(); } });
  const wCtl = toggle('Distance-weighted voting', { value: false, onChange: v => { weighted = v; refresh(); } });
  const rCtl = toggle('Show decision regions', { value: true, onChange: v => { showRegions = v; plot.render(); } });
  const out = readout([['neighbour labels', 0], ['vote', 0], ['prediction', 0], ['leave-one-out error', 0]]);
  const st = status('');
  right.append(kCtl.root, mCtl.root, wCtl.root, rCtl.root, out.root, st.root);

  canvas.addEventListener('pointerdown', e => { query = plot.eventWorld(e); refresh(); });
  canvas.classList.add('is-grabbable');

  const dist = (a, b) => metric === 'l2'
    ? Math.hypot(a[0] - b[0], a[1] - b[1])
    : Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]);

  function classify(q, excl = -1) {
    const order = pts.map((p2, i) => ({ i, d: dist(p2, q) }))
      .filter(o => o.i !== excl)
      .sort((a, b) => a.d - b.d).slice(0, k);
    let score = 0;
    order.forEach(o => { score += (weighted ? 1 / (o.d + 1e-6) : 1) * ys[o.i]; });
    return { pred: score >= 0 ? 1 : -1, order, score };
  }

  function refresh() {
    const { pred, order, score } = classify(query);
    const labels = order.map(o => (ys[o.i] > 0 ? '+' : '−')).join(' ');
    let loo = 0;
    for (let i = 0; i < pts.length; i++) if (classify(pts[i], i).pred !== ys[i]) loo++;
    out.set([
      labels,
      fmt(score, 3),
      { html: pred > 0 ? 'class +1' : 'class −1', cls: pred > 0 ? 'is-ok' : '' },
      { html: `${loo} / ${pts.length}  (${fmt(100 * loo / pts.length, 1)}%)`, cls: loo / pts.length < .12 ? 'is-ok' : '' },
    ]);
    st.set(
      k === 1
        ? `${WARN}<span><strong>k = 1</strong> gives a jagged boundary that wraps around every mislabelled point. Zero training error, high variance — it is memorising.</span>`
        : k > 18
          ? `${WARN}<span><strong>Large k</strong> smooths the boundary toward a straight line and starts ignoring genuine local structure. Low variance, high bias.</span>`
          : `${INFO}<span>Click anywhere to move the query point. Its <strong>${k}</strong> nearest neighbours are ringed, and the majority among them decides the label.</span>`,
      k === 1 || k > 18 ? 'warn' : 'info');
    plot.render();
  }

  plot.onDraw(p => {
    if (showRegions) drawDecision(p, (x, y) => classify([x, y]).score, { step: 7 });
    p.grid(1, { color: C.grid }); p.axes();
    const { order } = classify(query);
    order.forEach(o => p.line(query, pts[o.i], { color: C.c2, lw: 1.2, dash: [3, 3], alpha: .7 }));
    if (order.length) {
      const rad = order[order.length - 1].d;
      if (metric === 'l2') {
        const circ = [];
        for (let i = 0; i <= 90; i++) {
          const t = i / 90 * Math.PI * 2;
          circ.push([query[0] + Math.cos(t) * rad, query[1] + Math.sin(t) * rad]);
        }
        p.path(circ, { color: C.c2, lw: 1.6, dash: [5, 4], close: true, alpha: .8 });
      } else {
        p.polygon([[query[0] + rad, query[1]], [query[0], query[1] + rad],
                   [query[0] - rad, query[1]], [query[0], query[1] - rad]],
          { color: C.c2, lw: 1.6, dash: [5, 4], alpha: .8 });
      }
    }
    pts.forEach((q, i) => {
      const near = order.some(o => o.i === i);
      p.dot(q, { r: near ? 6 : 4.2, color: ys[i] > 0 ? C.c1 : C.c4, ring: near ? C.c2 : true, ringLw: near ? 2.6 : 1.6 });
    });
    p.dot(query, { r: 7, color: C.c2, ring: true, ringLw: 3 });
    p.badge(query, 'query', { color: C.c2, align: 'center', dy: -19 });
  });

  refresh();

  node.appendChild(note(
    `k-NN has no training phase at all — it just stores the data and answers each query by looking around. ` +
    `The dashed shape is the neighbourhood: a <strong>circle</strong> under Euclidean distance, a ` +
    `<strong>diamond</strong> under Manhattan, which is the unit ball of each norm from ` +
    `<a href="01-mathematical-background.html#norms">Chapter 1</a> reappearing. ` +
    `Watch the leave-one-out error as you sweep k: it is high at k = 1 (fitting the four mislabelled points), ` +
    `dips, then rises again as large k blurs the classes together. That dip is the elbow.`
  ));
});
