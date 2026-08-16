/* ============================================================
   widgets/deep.js — CNNs, RNNs and Transformers
   ============================================================ */
import {
  Plot, Dragger, C, css, el, slider, toggle, segmented, button,
  matrixInput, matrixView, readout, status, defineWidget, canvasHost,
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
const grayCell = (p, x, y, w, h, v) => {
  const g = clamp(Math.round(v), 0, 255);
  p.ctx.fillStyle = `rgb(${g},${g},${g})`;
  p.ctx.fillRect(x, y, w, h);
};

/* ============================================================
   1. Convolution — slide a kernel and watch the feature map build
   ============================================================ */
defineWidget('conv-demo', node => {
  const wrap = el('div');
  node.appendChild(wrap);

  const N = 14;
  const KERNELS = {
    edgeV:  { name: 'Vertical edge', k: [[1, 0, -1], [2, 0, -2], [1, 0, -1]] },
    edgeH:  { name: 'Horizontal edge', k: [[1, 2, 1], [0, 0, 0], [-1, -2, -1]] },
    blur:   { name: 'Blur', k: [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]] },
    sharpen:{ name: 'Sharpen', k: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]] },
  };
  let kKey = 'edgeV', stride = 1, pad = 0, pos = 0, relu = false;

  // a simple synthetic image: bright square on dark ground with a diagonal
  const IMG = Array.from({ length: N }, (_, i) =>
    Array.from({ length: N }, (_, j) => {
      let v = 40;
      if (j >= 3 && j <= 7 && i >= 3 && i <= 9) v = 210;
      if (Math.abs(i - j) <= 1 && j > 7) v = 180;
      if (i > 11) v = 90;
      return v;
    }));

  const kIn = matrixInput(3, 3, KERNELS[kKey].k.map(r => r.map(v => round(v, 3))), {
    label: 'Kernel', step: .5, onInput: () => refresh(),
  });
  const kCtl = segmented(Object.entries(KERNELS).map(([k, v]) => ({ label: v.name, value: k })),
    { value: kKey, label: 'Preset', onChange: v => { kKey = v; kIn.set(KERNELS[v].k.map(r => r.map(q => round(q, 3)))); refresh(); } });
  const sCtl = slider('Stride S', { min: 1, max: 4, step: 1, value: 1, format: v => String(v), onInput: v => { stride = v; refresh(); } });
  const pCtl = slider('Padding P', { min: 0, max: 3, step: 1, value: 0, format: v => String(v), onInput: v => { pad = v; refresh(); } });
  const reluCtl = toggle('Apply ReLU to the output', { value: false, onChange: v => { relu = v; refresh(); } });
  const posCtl = slider('Kernel position', { min: 0, max: 100, step: 1, value: 0, format: v => String(v), onInput: v => { pos = v; plot.render(); sync(); } });

  const cv = el('canvas');
  const left = el('div', {}, el('div', { class: 'pg-canvas-wrap' }, cv));
  const right = el('div', { class: 'pg-controls' }, kCtl.root, kIn.root, sCtl.root, pCtl.root, reluCtl.root, posCtl.root);
  const out = readout([['input', 0], ['kernel F', 0], ['output size', 0], ['formula', 0], ['value at this position', 0]]);
  const st = status('');
  right.append(out.root, st.root);
  wrap.appendChild(el('div', { class: 'pg-split pg-split--wide-ctrl' }, left, right));

  const plot = trackPlot(new Plot(cv, { xmin: 0, xmax: 34, ymin: 0, ymax: 20, aspect: 1.39, equal: false, pad: 0 }));

  let padded = [], outMap = [], H2 = 0;
  const F = 3;

  function refresh() {
    const K = kIn.get();
    const M = N + 2 * pad;
    padded = Array.from({ length: M }, (_, i) =>
      Array.from({ length: M }, (_, j) => {
        const ii = i - pad, jj = j - pad;
        return (ii >= 0 && ii < N && jj >= 0 && jj < N) ? IMG[ii][jj] : 0;
      }));
    H2 = Math.floor((N + 2 * pad - F) / stride) + 1;
    outMap = [];
    for (let oi = 0; oi < H2; oi++) {
      const row = [];
      for (let oj = 0; oj < H2; oj++) {
        let s = 0;
        for (let u = 0; u < F; u++) for (let v = 0; v < F; v++) {
          s += K[u][v] * padded[oi * stride + u][oj * stride + v];
        }
        row.push(relu ? Math.max(0, s) : s);
      }
      outMap.push(row);
    }
    posCtl.input.max = String(H2 * H2 - 1);
    if (pos > H2 * H2 - 1) { pos = H2 * H2 - 1; posCtl.set(pos); }
    plot.render(); sync();
  }

  function sync() {
    const oi = Math.floor(pos / H2), oj = pos % H2;
    out.set([
      `${N} × ${N}`,
      `${F} × ${F}`,
      { html: `${H2} × ${H2}`, cls: 'is-ok' },
      `⌊(${N} + 2·${pad} − ${F}) / ${stride}⌋ + 1 = ${H2}`,
      fmt(outMap[oi]?.[oj] ?? 0, 2),
    ]);
    st.set(
      H2 === N
        ? `${OK}<span><strong>Same convolution.</strong> With P = ${pad} and S = 1 the output keeps the input's spatial size.</span>`
        : stride > 1
          ? `${INFO}<span>Stride ${stride} skips positions, so the output is roughly ${stride}× smaller per axis — the cheapest way to downsample.</span>`
          : `${INFO}<span><strong>Valid convolution.</strong> With no padding the output shrinks by F − 1 = ${F - 1} pixels per axis.</span>`,
      H2 === N ? 'ok' : 'info');
  }

  plot.onDraw(p => {
    const M = N + 2 * pad;
    const cell = Math.min(0.62, 12 / M);
    const ox = 1, oy = 18;
    // padded input
    for (let i = 0; i < M; i++) for (let j = 0; j < M; j++) {
      const isPad = i < pad || j < pad || i >= M - pad || j >= M - pad;
      const X = p.X(ox + j * cell), Y = p.Y(oy - i * cell);
      grayCell(p, X, Y, p.px(cell) + 1, p.px(cell) + 1, padded[i][j]);
      if (isPad) {
        p.ctx.fillStyle = withA(C.c2, .35);
        p.ctx.fillRect(X, Y, p.px(cell) + 1, p.py(cell) + 1);
      }
    }
    // kernel window
    const oi = Math.floor(pos / H2), oj = pos % H2;
    const kx = ox + oj * stride * cell, ky = oy - oi * stride * cell;
    p.ctx.strokeStyle = C.c4; p.ctx.lineWidth = 2.6;
    p.ctx.strokeRect(p.X(kx), p.Y(ky), p.px(F * cell), p.py(F * cell));
    p.text([ox + M * cell / 2, oy - M * cell - .7], `input ${N}×${N}${pad ? ` (padded to ${M}×${M})` : ''}`,
      { align: 'center', size: 10.5, color: C.muted });

    // output map
    const ocell = Math.min(0.62, 11 / H2);
    const ox2 = 20, oy2 = 18;
    const vals = outMap.flat();
    const lo = Math.min(...vals), hi = Math.max(...vals);
    for (let i = 0; i < H2; i++) for (let j = 0; j < H2; j++) {
      const t = hi - lo < 1e-9 ? .5 : (outMap[i][j] - lo) / (hi - lo);
      const X = p.X(ox2 + j * ocell), Y = p.Y(oy2 - i * ocell);
      grayCell(p, X, Y, p.px(ocell) + 1, p.px(ocell) + 1, t * 255);
      if (i === oi && j === oj) {
        p.ctx.strokeStyle = C.c4; p.ctx.lineWidth = 2.4;
        p.ctx.strokeRect(X, Y, p.px(ocell), p.py(ocell));
      }
    }
    p.text([ox2 + H2 * ocell / 2, oy2 - H2 * ocell - .7], `feature map ${H2}×${H2}`,
      { align: 'center', size: 10.5, color: C.muted });

    // arrow between
    p.arrow([ox + M * cell + .5, oy - M * cell / 2], [ox2 - .6, oy2 - H2 * ocell / 2],
      { color: C.c1, lw: 2.4, head: 10 });
    p.text([(ox + M * cell + ox2) / 2, oy - M * cell / 2 + .9], 'convolve', { align: 'center', size: 10.5, color: C.c1, weight: 650 });
  });

  refresh();

  node.appendChild(note(
    `Each output pixel is <strong>one dot product</strong> between the kernel and the patch beneath it — ` +
    `slide the position control to watch the window travel and the corresponding output cell light up. ` +
    `Try the two edge kernels: they respond to intensity <em>changes</em>, so flat regions go to zero and ` +
    `boundaries light up. That is a learned feature detector, except here you wrote it by hand. ` +
    `Increase the padding until the output size matches the input — that is "same" convolution.`
  ));
});

/* ============================================================
   2. Output-size arithmetic
   ============================================================ */
defineWidget('conv-arith', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: 0, xmax: 12, ymin: 0, ymax: 8, aspect: 1.39, equal: false, pad: 0,
  }));

  let H = 32, F = 3, S = 1, P = 1, Cin = 3, Cout = 16;

  const hCtl = slider('Input size H', { min: 8, max: 64, step: 1, value: 32, format: v => String(v), onInput: v => { H = v; refresh(); } });
  const fCtl = slider('Kernel F', { min: 1, max: 11, step: 2, value: 3, format: v => String(v), onInput: v => { F = v; refresh(); } });
  const sCtl = slider('Stride S', { min: 1, max: 4, step: 1, value: 1, format: v => String(v), onInput: v => { S = v; refresh(); } });
  const pCtl = slider('Padding P', { min: 0, max: 5, step: 1, value: 1, format: v => String(v), onInput: v => { P = v; refresh(); } });
  const ciCtl = slider('Input channels', { min: 1, max: 512, step: 1, value: 3, format: v => String(v), onInput: v => { Cin = v; refresh(); } });
  const coCtl = slider('Output channels', { min: 1, max: 512, step: 1, value: 16, format: v => String(v), onInput: v => { Cout = v; refresh(); } });
  const presets = el('div', { class: 'pg-actions' },
    button('Same (3×3)', () => { F = 3; P = 1; S = 1; fCtl.set(3); pCtl.set(1); sCtl.set(1); refresh(); }),
    button('AlexNet conv1', () => { H = 227; F = 11; S = 4; P = 0; Cin = 3; Cout = 96; hCtl.input.max = '256'; hCtl.set(227); fCtl.input.max = '11'; fCtl.set(11); sCtl.set(4); pCtl.set(0); ciCtl.set(3); coCtl.set(96); refresh(); }),
  );
  const out = readout([['output H′', 0], ['formula', 0], ['conv parameters', 0], ['if fully connected', 0], ['saving', 0]]);
  const st = status('');
  right.append(hCtl.root, fCtl.root, sCtl.root, pCtl.root, ciCtl.root, coCtl.root, presets, out.root, st.root);

  function refresh() {
    const H2 = Math.floor((H + 2 * P - F) / S) + 1;
    const convP = Cout * (F * F * Cin + 1);
    const fcP = (H * H * Cin) * (H2 * H2 * Cout);
    out.set([
      { html: H2 > 0 ? `${H2} × ${H2}` : 'invalid', cls: H2 > 0 ? 'is-ok' : 'is-warn' },
      `⌊(${H} + 2·${P} − ${F}) / ${S}⌋ + 1`,
      convP.toLocaleString(),
      fcP.toLocaleString(),
      { html: fcP > convP ? `${fmt(fcP / convP, 0)}× fewer` : '—', cls: 'is-ok' },
    ]);
    st.set(
      H2 <= 0
        ? `${WARN}<span>The kernel is larger than the padded input — no valid position exists.</span>`
        : H2 === H
          ? `${OK}<span><strong>Same convolution</strong>: P = ⌊F/2⌋ = ${Math.floor(F / 2)} with S = 1 preserves the spatial size exactly.</span>`
          : `${INFO}<span>Convolution uses <strong>${convP.toLocaleString()}</strong> parameters here. A fully connected layer between the same two tensors would need <strong>${fcP.toLocaleString()}</strong> — that ratio is the whole argument for weight sharing.</span>`,
      H2 <= 0 ? 'warn' : H2 === H ? 'ok' : 'info');
    plot.render();
  }

  plot.onDraw(p => {
    const H2 = Math.max(1, Math.floor((H + 2 * P - F) / S) + 1);
    const scale = 5.2 / Math.max(H + 2 * P, 1);
    // input square
    const iw = H * scale, pw = (H + 2 * P) * scale;
    p.polygon([[1, 6.6], [1 + pw, 6.6], [1 + pw, 6.6 - pw], [1, 6.6 - pw]],
      { color: C.c2, lw: 1.6, dash: [4, 3], fill: withA(C.c2, .12) });
    p.polygon([[1 + P * scale, 6.6 - P * scale], [1 + P * scale + iw, 6.6 - P * scale],
               [1 + P * scale + iw, 6.6 - P * scale - iw], [1 + P * scale, 6.6 - P * scale - iw]],
      { color: C.c1, lw: 2, fill: withA(C.c1, .2) });
    // kernel
    p.polygon([[1 + P * scale, 6.6 - P * scale], [1 + P * scale + F * scale, 6.6 - P * scale],
               [1 + P * scale + F * scale, 6.6 - P * scale - F * scale], [1 + P * scale, 6.6 - P * scale - F * scale]],
      { color: C.c4, lw: 2.2 });
    p.text([1 + pw / 2, 7], `input ${H}×${H}×${Cin}${P ? ` + pad ${P}` : ''}`, { align: 'center', size: 11, color: C.muted });

    // output square
    const ow = H2 * scale;
    p.polygon([[8, 6.6], [8 + ow, 6.6], [8 + ow, 6.6 - ow], [8, 6.6 - ow]],
      { color: C.c3, lw: 2, fill: withA(C.c3, .2) });
    p.text([8 + ow / 2, 7], `output ${H2}×${H2}×${Cout}`, { align: 'center', size: 11, color: C.muted });
    p.arrow([1 + pw + .3, 6.6 - pw / 2], [7.7, 6.6 - ow / 2], { color: C.c1, lw: 2.4, head: 10 });
    p.text([6.5, .6], `F=${F}, S=${S}, P=${P}  →  H′ = ${H2}`, { align: 'center', size: 13, weight: 700, color: C.ink });
  });

  refresh();

  node.appendChild(note(
    `The formula <span class="u-mono">H′ = ⌊(H + 2P − F)/S⌋ + 1</span> is worth internalising — it is the ` +
    `single most common source of shape bugs when building a network. Note the parameter comparison: ` +
    `convolution costs <span class="u-mono">C_out(F²·C_in + 1)</span> weights <strong>regardless of image ` +
    `size</strong>, because the same kernel is reused at every position. A fully connected layer between the ` +
    `same tensors scales with H⁴. That is why nobody feeds raw pixels to a dense layer.`
  ));
});

/* ============================================================
   3. Pooling
   ============================================================ */
defineWidget('pooling', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: 0, xmax: 22, ymin: 0, ymax: 12, aspect: 1.48, equal: false, pad: 0,
  }));

  const N = 8;
  const r = ML.rng(17);
  let GRID = Array.from({ length: N }, () => Array.from({ length: N }, () => Math.round(r() * 9)));
  let mode = 'max', shift = 0;

  const mCtl = segmented([{ label: 'Max pooling', value: 'max' }, { label: 'Average pooling', value: 'avg' }],
    { value: 'max', label: 'Operation', onChange: v => { mode = v; refresh(); } });
  const shiftCtl = slider('Shift the input right by', {
    min: 0, max: 3, step: 1, value: 0, format: v => `${v} px`,
    onInput: v => { shift = v; refresh(); },
  });
  const acts = el('div', { class: 'pg-actions' },
    button('New values', () => {
      const rr = ML.rng(Math.floor(Math.random() * 1e6));
      GRID = Array.from({ length: N }, () => Array.from({ length: N }, () => Math.round(rr() * 9)));
      refresh();
    }));
  const out = readout([['input', 0], ['pool window', 0], ['output', 0], ['unchanged outputs after shift', 0]]);
  const st = status('');
  right.append(mCtl.root, shiftCtl.root, acts, out.root, st.root);

  const shifted = () => GRID.map(row => {
    const nr = new Array(N).fill(0);
    for (let j = 0; j < N; j++) if (j + shift < N) nr[j + shift] = row[j];
    return nr;
  });
  const pool = g => {
    const o = [];
    for (let i = 0; i < N; i += 2) {
      const row = [];
      for (let j = 0; j < N; j += 2) {
        const w = [g[i][j], g[i][j + 1], g[i + 1][j], g[i + 1][j + 1]];
        row.push(mode === 'max' ? Math.max(...w) : w.reduce((a, b) => a + b, 0) / 4);
      }
      o.push(row);
    }
    return o;
  };

  let base = null, cur = null;
  function refresh() {
    base = pool(GRID);
    cur = pool(shifted());
    let same = 0;
    base.forEach((row, i) => row.forEach((v, j) => { if (Math.abs(v - cur[i][j]) < 1e-9) same++; }));
    out.set([`${N} × ${N}`, '2 × 2, stride 2', `${N / 2} × ${N / 2}`,
      { html: `${same} / ${(N / 2) ** 2}`, cls: same > (N / 2) ** 2 * .6 ? 'is-ok' : '' }]);
    st.set(
      shift === 0
        ? `${INFO}<span>${mode === 'max' ? 'Max' : 'Average'} pooling takes the ${mode === 'max' ? 'largest' : 'mean'} value in each 2×2 window, quartering the number of activations.</span>`
        : `${OK}<span>After shifting the input by ${shift} pixel${shift > 1 ? 's' : ''}, <strong>${same} of ${(N / 2) ** 2}</strong> outputs are unchanged. That partial insensitivity to small translations is what pooling buys.</span>`,
      shift === 0 ? 'info' : 'ok');
    plot.render();
  }

  plot.onDraw(p => {
    const g = shifted();
    const cell = 1.05;
    const drawGrid = (M, ox, oy, cs, label, hlWin) => {
      const n = M.length;
      const vals = M.flat();
      const lo = Math.min(...vals), hi = Math.max(...vals);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const t = hi - lo < 1e-9 ? .5 : (M[i][j] - lo) / (hi - lo);
        const X = p.X(ox + j * cs), Y = p.Y(oy - i * cs);
        p.ctx.fillStyle = withA(C.c1, .12 + t * .6);
        p.ctx.fillRect(X + 1, Y + 1, p.px(cs) - 2, p.py(cs) - 2);
        p.text([ox + j * cs + cs / 2, oy - i * cs - cs / 2], fmt(M[i][j], mode === 'avg' && n < N ? 2 : 0),
          { align: 'center', size: 10.5, weight: 600, color: t > .55 ? C.raised : C.ink, mono: true });
      }
      if (hlWin) {
        p.ctx.strokeStyle = C.c4; p.ctx.lineWidth = 2;
        for (let i = 0; i < n; i += 2) for (let j = 0; j < n; j += 2) {
          p.ctx.strokeRect(p.X(ox + j * cs), p.Y(oy - i * cs), p.px(2 * cs), p.py(2 * cs));
        }
      }
      p.text([ox + n * cs / 2, oy + .45], label, { align: 'center', size: 11, color: C.muted, weight: 600 });
    };
    drawGrid(g, .6, 10.4, cell, `input 8×8${shift ? ` (shifted ${shift})` : ''}`, true);
    drawGrid(cur, 14.2, 9.3, cell * 1.1, `${mode === 'max' ? 'max' : 'avg'}-pooled 4×4`, false);
    p.arrow([10, 6], [13.6, 6], { color: C.c1, lw: 2.4, head: 10 });
    p.text([11.8, 6.6], '2×2 / s2', { align: 'center', size: 10.5, color: C.c1, weight: 650 });
  });

  refresh();

  node.appendChild(note(
    `Pooling reduces resolution and buys a little <strong>translation tolerance</strong>: shift the input a ` +
    `pixel or two and many pooled outputs do not change at all. <strong>Max</strong> pooling keeps the ` +
    `strongest response, which suits detecting whether a feature is present anywhere in the window; ` +
    `<strong>average</strong> pooling smooths instead. Modern networks often drop pooling and use strided ` +
    `convolutions, which downsample with learned weights rather than a fixed rule.`
  ));
});

/* ============================================================
   4. Receptive field growth
   ============================================================ */
defineWidget('receptive-field', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: 0, xmax: 40, ymin: 0, ymax: 14, aspect: 1.39, equal: false, pad: 0,
  }));

  let layers = [{ F: 3, S: 1 }, { F: 3, S: 1 }];

  const rebuildCtl = () => {
    ctlHost.innerHTML = '';
    layers.forEach((L, i) => {
      const fs = slider(`Layer ${i + 1} kernel F`, {
        min: 1, max: 7, step: 2, value: L.F, format: v => String(v),
        onInput: v => { layers[i].F = v; refresh(); },
      });
      const ss = slider(`Layer ${i + 1} stride S`, {
        min: 1, max: 3, step: 1, value: L.S, format: v => String(v),
        onInput: v => { layers[i].S = v; refresh(); },
      });
      ctlHost.append(fs.root, ss.root);
    });
  };
  const ctlHost = el('div', { style: 'display:flex;flex-direction:column;gap:.8rem' });
  const acts = el('div', { class: 'pg-actions' },
    button('+ Layer', () => { if (layers.length < 5) { layers.push({ F: 3, S: 1 }); rebuildCtl(); refresh(); } }),
    button('− Layer', () => { if (layers.length > 1) { layers.pop(); rebuildCtl(); refresh(); } }),
    button('Two 3×3 s1', () => { layers = [{ F: 3, S: 1 }, { F: 3, S: 1 }]; rebuildCtl(); refresh(); }),
    button('3×3 s2, then 3×3 s1', () => { layers = [{ F: 3, S: 2 }, { F: 3, S: 1 }]; rebuildCtl(); refresh(); }),
    button('One 5×5', () => { layers = [{ F: 5, S: 1 }]; rebuildCtl(); refresh(); }),
  );
  const out = readout([['receptive field', 0], ['jump Π', 0], ['layers', 0], ['params (C in = C out = C)', 0]]);
  const st = status('');
  right.append(ctlHost, acts, out.root, st.root);
  rebuildCtl();

  let steps = [];
  function refresh() {
    let R = 1, J = 1;
    steps = [{ R, J }];
    layers.forEach(L => { R = R + (L.F - 1) * J; J = J * L.S; steps.push({ R, J }); });
    const params = layers.reduce((s, L) => s + L.F * L.F, 0);
    out.set([
      { html: `${R} × ${R} pixels`, cls: 'is-ok' },
      String(J), String(layers.length),
      `${params}·C²`,
    ]);
    const oneBig = R;
    st.set(
      layers.length >= 2 && layers.every(L => L.F === 3 && L.S === 1)
        ? `${OK}<span>${layers.length} stacked 3×3 layers give the same ${R}×${R} receptive field as one ${oneBig}×${oneBig} kernel, but with <strong>${params}·C² parameters instead of ${oneBig * oneBig}·C²</strong> — and ${layers.length} nonlinearities instead of one. That is the VGG argument.</span>`
        : `${INFO}<span>RF grows by (F−1)·Π at each layer, and a stride multiplies Π for everything above it — which is why strided layers expand the receptive field so aggressively.</span>`,
      layers.length >= 2 && layers.every(L => L.F === 3 && L.S === 1) ? 'ok' : 'info');
    plot.render();
  }

  plot.onDraw(p => {
    const rows = steps.length;
    const rowH = 12 / rows;
    const maxR = steps[steps.length - 1].R;
    const px = Math.min(1.0, 36 / Math.max(maxR, 1));
    steps.forEach((s, li) => {
      const y = 12.6 - li * rowH;
      const cx = 20;
      // draw the pixel span
      for (let k = 0; k < s.R; k++) {
        const x = cx - (s.R * px) / 2 + k * px;
        p.ctx.fillStyle = li === 0 ? withA(C.muted, .35) : withA(C.c1, .25 + .5 * (li / rows));
        p.ctx.fillRect(p.X(x) + .8, p.Y(y), Math.max(1, p.px(px) - 1.6), p.py(rowH * .42));
      }
      const label = li === 0 ? 'input pixel' : `after layer ${li} (F=${layers[li - 1].F}, S=${layers[li - 1].S})`;
      p.text([1, y - rowH * .21], label, { size: 10.5, color: C.muted, weight: 600 });
      p.text([38.5, y - rowH * .21], `RF ${s.R}, Π ${s.J}`, { align: 'right', size: 10.5, color: C.c1, weight: 700, mono: true });
    });
    p.text([20, .5], 'width of the input region a single deep unit can see', { align: 'center', size: 10.5, color: C.muted });
  });

  refresh();

  node.appendChild(note(
    `The recurrence is <span class="u-mono">RF_ℓ = RF_{ℓ−1} + (F_ℓ − 1)·Π_{ℓ−1}</span> with ` +
    `<span class="u-mono">Π_ℓ = Π_{ℓ−1}·S_ℓ</span>, starting from RF₀ = 1 and Π₀ = 1. Two 3×3 layers reach ` +
    `5×5; three reach 7×7 — the same as one 7×7 kernel but with 27C² parameters instead of 49C², and three ` +
    `nonlinearities instead of one. Now set the <em>first</em> layer's stride to 2: the jump doubles, so the ` +
    `second layer adds (3−1)×2 = 4 and the receptive field reaches 7 after only two layers.`
  ));
});

/* ============================================================
   5. RNN unrolled — the H,E,L,L -> O example
   ============================================================ */
defineWidget('rnn-unroll', node => {
  const wrap = el('div');
  node.appendChild(wrap);

  const VOCAB = ['H', 'E', 'L', 'O'];
  const U = [
    [0.2870, 0.8461, 0.5724, 0.4868],
    [0.9029, 0.8715, 0.6911, 0.1899],
    [0.5375, 0.0922, 0.5582, 0.4915],
  ];
  const Wr = [[0.4270, 0, 0], [0, 0.4270, 0], [0, 0, 0.4270]];
  const V = [
    [0.3717, 0.9748, 0.8300],
    [0.3914, 0.2826, 0.6598],
    [0.6499, 0.0982, 0.3343],
    [0.9127, 0.3258, 0.9531],
  ];
  const bs = [0.567001, 0.567001, 0.567001];
  const SEQ = ['H', 'E', 'L', 'L'];
  const TGT = ['E', 'L', 'L', 'O'];
  let step = 4;

  const stepCtl = slider('Time step t', {
    min: 1, max: 4, step: 1, value: 4, format: v => String(v),
    onInput: v => { step = v; refresh(); },
  });
  const cv = el('canvas');
  const left = el('div', {}, el('div', { class: 'pg-canvas-wrap' }, cv));
  const right = el('div', { class: 'pg-controls' }, stepCtl.root);
  const out = readout([['input', 0], ['U xₜ', 0], ['W sₜ₋₁', 0], ['pre-activation', 0], ['sₜ = tanh(·)', 0], ['p̂ₜ', 0], ['prediction', 0], ['loss', 0]]);
  const st = status('');
  right.append(out.root, st.root);
  wrap.appendChild(el('div', { class: 'pg-split pg-split--wide-ctrl' }, left, right));

  const plot = trackPlot(new Plot(cv, { xmin: 0, xmax: 20, ymin: 0, ymax: 11, aspect: 1.39, equal: false, pad: 0 }));

  const mv = (M, v) => M.map(row => row.reduce((s, w, j) => s + w * v[j], 0));
  const softmax = z => { const m = Math.max(...z); const e = z.map(v => Math.exp(v - m)); const s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); };

  // run the whole sequence once
  const trace = [];
  {
    let s = [0, 0, 0];
    SEQ.forEach((ch, i) => {
      const x = VOCAB.map(c => (c === ch ? 1 : 0));
      const Ux = mv(U, x);
      const Ws = mv(Wr, s);
      const pre = Ux.map((v, k) => v + Ws[k] + bs[k]);
      s = pre.map(Math.tanh);
      const y = mv(V, s);
      const p = softmax(y);
      const tgtIdx = VOCAB.indexOf(TGT[i]);
      trace.push({ ch, x, Ux, Ws, pre, s: s.slice(), y, p, pred: VOCAB[p.indexOf(Math.max(...p))],
        loss: -Math.log(p[tgtIdx]), tgt: TGT[i] });
    });
  }

  function refresh() {
    const T = trace[step - 1];
    const v = a => a.map(q => fmt(q, 4)).join(', ');
    out.set([
      `${T.ch} → one-hot [${T.x.join(', ')}]`,
      v(T.Ux), v(T.Ws), v(T.pre), v(T.s),
      T.p.map(q => fmt(q, 4)).join(', '),
      { html: `${T.pred} (target ${T.tgt})`, cls: T.pred === T.tgt ? 'is-ok' : 'is-warn' },
      fmt(T.loss, 6),
    ]);
    if (step === 4) {
      const margin = T.p[3] - T.p[0];
      st.set(`${WARN}<span><strong>O wins by ${fmt(margin, 5)}.</strong> The top two probabilities are ` +
        `${fmt(T.p[3], 4)} (O) and ${fmt(T.p[0], 4)} (H) — a margin of about a quarter of a thousandth. ` +
        `Round any intermediate to three decimals and the answer flips.</span>`, 'warn');
    } else {
      st.set(`${INFO}<span>The prediction here is <strong>${T.pred}</strong> but the target is ` +
        `<strong>${T.tgt}</strong> — this network is only trained to get the final character right.</span>`, 'info');
    }
    plot.render();
  }

  plot.onDraw(p => {
    const xs = [2.5, 6.5, 10.5, 14.5];
    // s_0
    p.ctx.beginPath(); p.ctx.arc(p.X(0.9), p.Y(5.5), 16, 0, Math.PI * 2);
    p.ctx.fillStyle = C.raised; p.ctx.fill();
    p.ctx.strokeStyle = C.muted; p.ctx.lineWidth = 1.8; p.ctx.stroke();
    p.text([0.9, 5.5], 's₀', { align: 'center', size: 11, weight: 700, color: C.muted });

    trace.forEach((T, i) => {
      const x = xs[i];
      const active = i === step - 1;
      const done = i < step;
      // recurrent arrow
      p.arrow([i === 0 ? 1.35 : xs[i - 1] + .75, 5.5], [x - .75, 5.5],
        { color: done ? C.c1 : C.grid, lw: done ? 2.4 : 1.4, head: 9 });
      if (i > 0) p.text([(xs[i - 1] + x) / 2, 6.1], 'W', { align: 'center', size: 10, color: done ? C.c1 : C.muted, weight: 650 });

      // state node
      p.ctx.beginPath(); p.ctx.arc(p.X(x), p.Y(5.5), active ? 24 : 19, 0, Math.PI * 2);
      p.ctx.fillStyle = done ? withA(C.c1, .22) : C.raised; p.ctx.fill();
      p.ctx.strokeStyle = active ? C.c2 : (done ? C.c1 : C.grid);
      p.ctx.lineWidth = active ? 3 : 1.8; p.ctx.stroke();
      p.text([x, 5.75], `s${'₁₂₃₄'[i]}`, { align: 'center', size: 11.5, weight: 700, color: done ? C.c1 : C.muted });
      if (done) p.text([x, 5.15], fmt(T.s[0], 2), { align: 'center', size: 9, color: C.muted, mono: true });

      // input below
      p.arrow([x, 2.6], [x, 4.65], { color: done ? C.c3 : C.grid, lw: done ? 2.2 : 1.3, head: 8 });
      p.ctx.fillStyle = done ? withA(C.c3, .25) : C.raised;
      p.ctx.strokeStyle = done ? C.c3 : C.grid; p.ctx.lineWidth = 1.8;
      p.ctx.beginPath(); p.ctx.roundRect(p.X(x - .55), p.Y(2.4), p.px(1.1), p.py(1.0), 6);
      p.ctx.fill(); p.ctx.stroke();
      p.text([x, 1.9], T.ch, { align: 'center', size: 14, weight: 750, color: done ? C.c3 : C.muted });
      p.text([x, 1.15], `x${'₁₂₃₄'[i]}`, { align: 'center', size: 9.5, color: C.muted });

      // output above
      p.arrow([x, 6.4], [x, 8.3], { color: done ? C.c2 : C.grid, lw: done ? 2.2 : 1.3, head: 8 });
      const best = T.p.indexOf(Math.max(...T.p));
      p.ctx.fillStyle = done ? withA(C.c2, .25) : C.raised;
      p.ctx.strokeStyle = done ? C.c2 : C.grid; p.ctx.lineWidth = 1.8;
      p.ctx.beginPath(); p.ctx.roundRect(p.X(x - .55), p.Y(9.5), p.px(1.1), p.py(1.0), 6);
      p.ctx.fill(); p.ctx.stroke();
      p.text([x, 9.0], done ? VOCAB[best] : '?', { align: 'center', size: 14, weight: 750, color: done ? C.c2 : C.muted });
      p.text([x, 10.1], `target ${T.tgt}`, { align: 'center', size: 9.5, color: C.muted });
    });
    p.text([10, .4], 'the same U, W, V at every step',
      { align: 'center', size: 10.5, color: C.muted });
  });

  refresh();

  node.appendChild(note(
    `Every quantity here is recomputed live from the same U, W, V and bias given in the notes, so you can ` +
    `check any step by hand. Two things worth noticing. First, the <strong>same three matrices are reused ` +
    `at every time step</strong> — that weight sharing across time is exactly analogous to a CNN's sharing ` +
    `across space. Second, the network gets the first three predictions <em>wrong</em>: it was only ever ` +
    `tuned to produce O at t = 4, and it does so by a margin of 0.00025.`
  ));
});

/* ============================================================
   6. Vanishing gradients: vanilla RNN vs GRU carry path
   ============================================================ */
defineWidget('vanishing-gradient', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: 0, xmax: 21, ymin: -9, ymax: 0.6, aspect: 1.31, equal: false, pad: 0,
  }));

  let w = 0.8, alpha = 0.25, z = 0.10, r = 0, wh = 1.0, T = 10;

  const wCtl = slider('Recurrent weight w', { min: .1, max: 2, step: .05, value: .8, onInput: v => { w = v; refresh(); } });
  const aCtl = slider("tanh′ ≈ α", { min: .05, max: 1, step: .01, value: .25, onInput: v => { alpha = v; refresh(); } });
  const zCtl = slider('GRU update gate z', { min: 0, max: 1, step: .01, value: .10, onInput: v => { z = v; refresh(); } });
  const rCtl = slider('GRU reset gate r', { min: 0, max: 1, step: .01, value: 0, onInput: v => { r = v; refresh(); } });
  const tCtl = slider('Sequence length T', { min: 5, max: 40, step: 1, value: 10, format: v => String(v), onInput: v => { T = v; refresh(); } });
  const presets = el('div', { class: 'pg-actions' },
    button('Notes: vanilla', () => { w = .8; alpha = .25; wCtl.set(.8); aCtl.set(.25); refresh(); }),
    button('Notes: GRU case 1', () => { z = .10; r = 0; zCtl.set(.10); rCtl.set(0); refresh(); }),
    button('Notes: GRU case 2', () => { z = .3; r = 1; wh = 1; zCtl.set(.3); rCtl.set(1); refresh(); }),
    button('Exploding (w=2, α=1)', () => { w = 2; alpha = 1; wCtl.set(2); aCtl.set(1); refresh(); }),
  );
  const out = readout([['vanilla per-step  w·α', 0], ['GRU per-step  g', 0], ['vanilla at t = 1', 0], ['GRU at t = 1', 0], ['ratio', 0]]);
  const st = status('');
  right.append(wCtl.root, aCtl.root, zCtl.root, rCtl.root, tCtl.root, presets, out.root, st.root);

  function refresh() {
    const van = w * alpha;
    const g = (1 - z) + z * alpha * wh * r;
    const v1 = van ** (T - 1), g1 = g ** (T - 1);
    out.set([
      fmt(van, 4), fmt(g, 4),
      v1 < 1e-4 ? v1.toExponential(2) : fmt(v1, 6),
      g1 < 1e-4 ? g1.toExponential(2) : fmt(g1, 6),
      { html: g1 / Math.max(v1, 1e-300) > 1 ? `${(g1 / Math.max(v1, 1e-300)).toExponential(1)}× better` : '—', cls: 'is-ok' },
    ]);
    st.set(
      van > 1
        ? `${WARN}<span><strong>Exploding.</strong> w·α = ${fmt(van, 3)} &gt; 1, so the gradient grows by ${fmt(van ** (T - 1), 1)}× over ${T - 1} steps. In practice this is handled by <em>gradient clipping</em>.</span>`
        : v1 < 1e-5
          ? `${WARN}<span><strong>Vanished.</strong> The vanilla gradient reaching t = 1 is ${v1.toExponential(2)} of the terminal gradient — early inputs receive essentially no credit at all.</span>`
          : `${INFO}<span>The vanilla factor is w·α = ${fmt(van, 3)}; the GRU factor is g = ${fmt(g, 3)}. The GRU's carry path <strong>bypasses the tanh entirely</strong>.</span>`,
      van > 1 || v1 < 1e-5 ? 'warn' : 'info');
    plot.render();
  }

  plot.onDraw(p => {
    const van = w * alpha;
    const g = (1 - z) + z * alpha * wh * r;
    const lg = f => Math.log10(Math.max(1e-300, f ** (T - 1)));
    p.o.xmin = 0; p.o.xmax = T + 1;
    // fit both curves, keeping 0 in view and clamping the depth of the well
    const lo = Math.max(-24, Math.min(lg(van), lg(g), 0) - .6);
    p.o.ymin = lo; p.o.ymax = Math.max(.6, lg(van) + .6);
    p._computeScale();
    p.grid(Math.max(1, Math.round((p.o.ymax - p.o.ymin) / 8)), { color: C.grid });

    const series = (factor, col, dash) => {
      const pts = [];
      for (let t = 1; t <= T; t++) pts.push([t, Math.log10(Math.max(1e-300, factor ** (T - t)))]);
      p.path(pts, { color: col, lw: 2.8, dash });
      pts.forEach(q => p.dot(q, { r: 3.4, color: col }));
    };
    series(van, C.c4, null);
    series(g, C.c3, [6, 4]);
    p.line([0, 0], [T + 1, 0], { color: C.muted, lw: 1.2, dash: [4, 4], alpha: .7 });

    p.axes(); p.ticks(Math.max(1, Math.round(T / 8)));
    // exploding curves occupy the top-left, vanishing ones the bottom-left
    p.legend(
      [[C.c4, `vanilla RNN  (w·α = ${fmt(van, 3)})`], [C.c3, `GRU  (g = ${fmt(g, 3)})`, [6, 4]]],
      { corner: van > 1 ? 'bl' : 'br', title: 'log₁₀ gradient, relative to the final step' });
    p.xlabel('time step t', { size: 10.5 });
  });

  refresh();

  node.appendChild(note(
    `The vertical axis is logarithmic, which is the only way to show both curves at once — and that is the ` +
    `whole point. A vanilla RNN multiplies by <span class="u-mono">w·tanh′</span> at every step, so with ` +
    `w = 0.8 and tanh′ ≈ 0.25 the factor is 0.20 and the gradient reaching t = 1 is ` +
    `<strong>5.12 × 10⁻⁷</strong> of the terminal gradient. A GRU with a small update gate has ` +
    `<span class="u-mono">g ≈ 1 − z = 0.90</span>, giving <strong>0.387</strong> — six orders of magnitude ` +
    `better. Push w above 1/α and the problem inverts into <em>exploding</em> gradients.`
  ));
});

/* ============================================================
   7. GRU gates
   ============================================================ */
defineWidget('gru-gates', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: 0, xmax: 21, ymin: -1.35, ymax: 1.35, aspect: 1.39, equal: false, pad: 0,
  }));

  let z = 0.15, r = 0.5, T = 20;
  // a signal to remember: a pulse early, then noise
  const rr = ML.rng(5);
  const xs = Array.from({ length: 21 }, (_, t) => (t === 3 ? 1.0 : ML.gauss(rr) * 0.28));

  const zCtl = slider('Update gate z', { min: 0, max: 1, step: .01, value: .15, onInput: v => { z = v; refresh(); } });
  const rCtl = slider('Reset gate r', { min: 0, max: 1, step: .01, value: .5, onInput: v => { r = v; refresh(); } });
  const presets = el('div', { class: 'pg-actions' },
    button('z ≈ 0 · remember', () => { z = .02; zCtl.set(.02); refresh(); }),
    button('z ≈ 1 · overwrite', () => { z = .98; zCtl.set(.98); refresh(); }),
    button('Balanced', () => { z = .3; r = 1; zCtl.set(.3); rCtl.set(1); refresh(); }),
  );
  const out = readout([['carry weight 1 − z', 0], ['state at t = 20', 0], ['peak retained', 0], ['∂sₜ/∂sₜ₋₁ ≈', 0]]);
  const st = status('');
  right.append(zCtl.root, rCtl.root, presets, out.root, st.root);

  let states = [];
  function refresh() {
    states = [0];
    let s = 0;
    for (let t = 1; t <= T; t++) {
      const cand = Math.tanh(2.2 * xs[t] + 0.9 * (r * s));
      s = (1 - z) * s + z * cand;
      states.push(s);
    }
    const g = (1 - z) + z * 0.25 * 0.9 * r;
    out.set([
      fmt(1 - z, 3),
      fmt(states[T], 4),
      `${fmt(100 * Math.abs(states[T]) / Math.max(1e-9, Math.max(...states.map(Math.abs))), 1)}%`,
      fmt(g, 4),
    ]);
    st.set(
      z < .05
        ? `${OK}<span><strong>z ≈ 0: the state is preserved.</strong> The pulse at t = 3 is still visible at t = 20, and ∂sₜ/∂sₜ₋₁ ≈ 1 — a near-identity path for gradients.</span>`
        : z > .9
          ? `${WARN}<span><strong>z ≈ 1: the state is overwritten every step.</strong> The old value is discarded, the pulse is forgotten immediately, and the unit behaves like a feedforward layer.</span>`
          : `${INFO}<span>With z = ${fmt(z, 2)} the unit keeps ${fmt((1 - z) * 100, 0)}% of its previous state each step. Memory decays with a half-life of about ${fmt(Math.log(0.5) / Math.log(Math.max(1e-9, 1 - z)), 1)} steps.</span>`,
      z < .05 ? 'ok' : z > .9 ? 'warn' : 'info');
    plot.render();
  }

  plot.onDraw(p => {
    p.grid(.5, { color: C.grid });
    // inputs
    for (let t = 1; t <= T; t++) {
      p.line([t, 0], [t, xs[t]], { color: C.c5, lw: 2.2, alpha: .55 });
      p.dot([t, xs[t]], { r: 2.6, color: C.c5, alpha: .7 });
    }
    p.path(states.map((s, t) => [t, s]), { color: C.c1, lw: 3 });
    states.forEach((s, t) => { if (t) p.dot([t, s], { r: 3.2, color: C.c1 }); });
    p.line([3, -1.35], [3, 1.35], { color: C.c2, lw: 1.6, dash: [5, 4] });
    p.badge([3, 1.2], 'signal to remember', { color: C.c2, align: 'center' });
    p.axes(); p.ticks(5);
    p.legend([[C.c5, 'input xₜ'], [C.c1, 'hidden state sₜ']], { corner: 'bl' });
  });

  refresh();

  node.appendChild(note(
    `The update equation <span class="u-mono">sₜ = (1 − z)sₜ₋₁ + z·s̃ₜ</span> is a weighted blend, and z ` +
    `decides the blend. At <strong>z ≈ 0</strong> the unit is almost the identity: the pulse at t = 3 ` +
    `survives to the end of the sequence and gradients flow back nearly undamped. At <strong>z ≈ 1</strong> ` +
    `the past is discarded every step. The crucial detail is that the carry term ` +
    `<span class="u-mono">(1 − z)sₜ₋₁</span> is <strong>linear</strong> — it never passes through a tanh, ` +
    `so it cannot saturate. That is the difference between a GRU and a vanilla RNN.`
  ));
});

/* ============================================================
   8. Positional encoding
   ============================================================ */
defineWidget('positional-encoding', node => {
  const wrap = el('div');
  node.appendChild(wrap);

  let dModel = 32, T = 40, showPair = true, posA = 5;

  const dCtl = slider('d_model', { min: 8, max: 64, step: 2, value: 32, format: v => String(v), onInput: v => { dModel = v; refresh(); } });
  const tCtl = slider('Sequence length T', { min: 10, max: 80, step: 1, value: 40, format: v => String(v), onInput: v => { T = v; refresh(); } });
  const pCtl = slider('Inspect position p', { min: 0, max: 39, step: 1, value: 5, format: v => String(v), onInput: v => { posA = v; refresh(); } });
  const pairCtl = toggle('Show similarity to every other position', { value: true, onChange: v => { showPair = v; plot2.render(); } });

  const cvHeat = el('canvas'), cvSim = el('canvas');
  const grid = el('div', { style: 'display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:1rem' },
    el('div', {}, el('div', { class: 'matrix-label', html: 'PE matrix &nbsp;<span style="font-weight:400;color:var(--ink-faint)">(rows = positions, columns = channels)</span>' }),
      el('div', { class: 'pg-canvas-wrap' }, cvHeat)),
    el('div', {}, el('div', { class: 'matrix-label', html: 'Dot-product similarity to position p' }),
      el('div', { class: 'pg-canvas-wrap' }, cvSim)));
  const out = readout([['channels', 0], ['wavelength, channel 0', 0], ['wavelength, last channel', 0], ['‖PE(p)‖', 0]]);
  const st = status('');
  wrap.append(grid, el('div', { class: 'pg-controls', style: 'margin-top:1rem' }, dCtl.root, tCtl.root, pCtl.root, pairCtl.root, out.root, st.root));

  const plot = trackPlot(new Plot(cvHeat, { xmin: 0, xmax: 1, ymin: 0, ymax: 1, aspect: 1.05, equal: false, pad: 0 }));
  const plot2 = trackPlot(new Plot(cvSim, { xmin: 0, xmax: 1, ymin: -1.1, ymax: 1.1, aspect: 1.05, equal: false, pad: 0 }));

  const PE = (p, i) => {
    const k = Math.floor(i / 2);
    const denom = Math.pow(10000, (2 * k) / dModel);
    return i % 2 === 0 ? Math.sin(p / denom) : Math.cos(p / denom);
  };

  function refresh() {
    pCtl.input.max = String(T - 1);
    if (posA > T - 1) { posA = T - 1; pCtl.set(posA); }
    const norm = Math.sqrt(Array.from({ length: dModel }, (_, i) => PE(posA, i) ** 2).reduce((a, b) => a + b, 0));
    out.set([
      String(dModel),
      `2π ≈ 6.28`,
      `2π·10000^${fmt((dModel - 2) / dModel, 2)} ≈ ${fmt(2 * Math.PI * Math.pow(10000, (dModel - 2) / dModel), 0)}`,
      fmt(norm, 4),
    ]);
    st.set(
      `${INFO}<span>Each channel is a sinusoid of a different wavelength — fast on the left, slow on the ` +
      `right, spanning 2π to about ${fmt(2 * Math.PI * Math.pow(10000, (dModel - 2) / dModel), 0)}. ` +
      `Together they give every position a <strong>unique fingerprint</strong>, and nearby positions get ` +
      `similar ones.</span>`, 'info');
    plot.render(); plot2.render();
  }

  plot.onDraw(p => {
    // Rows read downwards, as a matrix is written: position 0 on top. The
    // bands above and below the matrix hold the two labels so neither is
    // drawn over the colours.
    const padTop = Math.max(1.5, T * .07), padBot = Math.max(1.8, T * .085);
    p.o.xmin = 0; p.o.xmax = dModel; p.o.ymin = -padBot; p.o.ymax = T + padTop;
    p._computeScale();
    const rowY = pos => T - pos;                      // top edge of row `pos`
    const cw = p.px(1), ch = p.py(1);
    for (let pos = 0; pos < T; pos++) for (let i = 0; i < dModel; i++) {
      const v = PE(pos, i);
      p.ctx.fillStyle = v >= 0 ? withA(C.c1, .1 + Math.abs(v) * .8) : withA(C.c4, .1 + Math.abs(v) * .8);
      p.ctx.fillRect(p.X(i), p.Y(rowY(pos)), cw + 1, ch + 1);
    }
    p.ctx.strokeStyle = C.c2; p.ctx.lineWidth = 2.4;
    p.ctx.strokeRect(p.X(0), p.Y(rowY(posA)), p.px(dModel), ch);
    p.text([0, T + padTop * .5], `p = ${posA}`, { color: C.c2, size: 11, weight: 700, dx: 2 });
    p.text([dModel, T + padTop * .5], 'position 0 at the top',
      { color: C.muted, size: 9.5, align: 'right', dx: -2 });
    p.xlabel('channel i →', { size: 10 });
  });

  plot2.onDraw(p => {
    p.o.xmin = 0; p.o.xmax = T; p.o.ymin = -1.15; p.o.ymax = 1.15;
    p._computeScale();
    p.grid(.5, { color: C.grid });
    if (showPair) {
      const a = Array.from({ length: dModel }, (_, i) => PE(posA, i));
      const na = Math.hypot(...a);
      const pts = [];
      for (let q = 0; q < T; q++) {
        const b = Array.from({ length: dModel }, (_, i) => PE(q, i));
        const dot = a.reduce((s, v, i) => s + v * b[i], 0);
        pts.push([q, dot / (na * Math.hypot(...b))]);
      }
      p.path(pts, { color: C.c1, lw: 2.6 });
      pts.forEach(q => p.dot(q, { r: 2.6, color: C.c1, alpha: .8 }));
      p.line([posA, -1.15], [posA, 1.15], { color: C.c2, lw: 1.8, dash: [5, 4] });
      p.badge([posA, 1.05], `p = ${posA}`, { color: C.c2, align: 'center', dy: 16 });
    }
    p.axes(); p.ticks(Math.max(5, Math.round(T / 8)));
    p.text({ px: 8, py: 13 }, 'cosine similarity between PE(p) and PE(q)',
      { color: C.muted, size: 10, halo: true, haloWidth: 4 });
    p.xlabel('position q', { size: 10 });
  });

  refresh();

  node.appendChild(note(
    `Self-attention is <strong>permutation-invariant</strong> — shuffle the tokens and it computes the same ` +
    `thing — so position has to be injected explicitly. The sinusoidal scheme gives channel <em>i</em> a ` +
    `wavelength of <span class="u-mono">2π·10000^(2i/d)</span>, from 2π up to tens of thousands. Look at the ` +
    `similarity curve: it peaks at q = p and falls away smoothly, so the encoding carries a usable notion of ` +
    `<em>nearby</em>. Because the pattern is smooth and multi-frequency rather than a lookup table, it also ` +
    `extends to sequences longer than anything seen in training.`
  ));
});

/* ============================================================
   9. Self-attention on a real sentence
   ============================================================ */
defineWidget('attention', node => {
  const wrap = el('div');
  node.appendChild(wrap);

  const TOKENS = ['The', 'cat', 'sat', 'on', 'the', 'mat'];
  const T = TOKENS.length;
  const dk = 8;
  let query = 1, causal = false, temp = 1, scaled = true;

  // deterministic "learned" embeddings with some structure:
  // 'cat' and 'mat' rhyme-ish/related, 'sat' verb, etc.
  const r = ML.rng(2024);
  const EMB = TOKENS.map(() => Array.from({ length: dk }, () => ML.gauss(r) * 0.8));
  // hand-tie a couple of relationships so attention is interpretable
  EMB[5] = EMB[1].map((v, i) => v * 0.75 + ML.gauss(r) * 0.25);   // mat ~ cat
  EMB[4] = EMB[0].map((v, i) => v * 0.9 + ML.gauss(r) * 0.15);    // the ~ The
  const WQ = Array.from({ length: dk }, () => Array.from({ length: dk }, () => ML.gauss(r) * 0.4));
  const WK = Array.from({ length: dk }, () => Array.from({ length: dk }, () => ML.gauss(r) * 0.4));
  const WV = Array.from({ length: dk }, () => Array.from({ length: dk }, () => ML.gauss(r) * 0.4));

  const qCtl = segmented(TOKENS.map((t, i) => ({ label: t, value: String(i) })),
    { value: '1', label: 'Query token', onChange: v => { query = +v; refresh(); } });
  const causalCtl = toggle('Causal mask (no peeking ahead)', { value: false, onChange: v => { causal = v; refresh(); } });
  const scaleCtl = toggle('Divide by √d_k', { value: true, onChange: v => { scaled = v; refresh(); } });
  const tempCtl = slider('Logit temperature', { min: .2, max: 4, step: .05, value: 1, onInput: v => { temp = v; refresh(); } });

  const cv = el('canvas');
  const left = el('div', {}, el('div', { class: 'pg-canvas-wrap' }, cv));
  const right = el('div', { class: 'pg-controls' }, qCtl.root, causalCtl.root, scaleCtl.root, tempCtl.root);
  const out = readout([['d_k', 0], ['scale', 0], ['max attention weight', 0], ['attends most to', 0], ['entropy of the row', 0]]);
  const st = status('');
  right.append(out.root, st.root);
  wrap.appendChild(el('div', { class: 'pg-split pg-split--wide-ctrl' }, left, right));

  const plot = trackPlot(new Plot(cv, { xmin: 0, xmax: 14, ymin: 0, ymax: 10, aspect: 1.19, equal: false, pad: 0 }));

  const mv = (M, v) => M.map(row => row.reduce((s, w, j) => s + w * v[j], 0));
  const proj = (E, W) => E.map(e => W.map(row => row.reduce((s, w, j) => s + w * e[j], 0)));

  let A = [], E = [];
  function refresh() {
    const Q = proj(EMB, WQ), K = proj(EMB, WK);
    const scale = scaled ? Math.sqrt(dk) : 1;
    E = Q.map((q, i) => K.map((k, j) => {
      let s = q.reduce((acc, v, m) => acc + v * k[m], 0) / scale / temp;
      if (causal && j > i) s = -Infinity;
      return s;
    }));
    A = E.map(row => {
      const m = Math.max(...row.filter(Number.isFinite));
      const e = row.map(v => (Number.isFinite(v) ? Math.exp(v - m) : 0));
      const s = e.reduce((a, b) => a + b, 0);
      return e.map(v => v / s);
    });
    const row = A[query];
    const best = row.indexOf(Math.max(...row));
    const H = -row.reduce((s, v) => s + (v > 1e-12 ? v * Math.log2(v) : 0), 0);
    out.set([
      String(dk),
      scaled ? `1/√${dk} = ${fmt(1 / Math.sqrt(dk), 4)}` : 'none (1)',
      fmt(Math.max(...row), 4),
      `"${TOKENS[best]}"`,
      { html: `${fmt(H, 3)} bits (max ${fmt(Math.log2(T), 3)})`, cls: H < 1 ? 'is-ok' : '' },
    ]);
    st.set(
      !scaled
        ? `${WARN}<span><strong>Unscaled.</strong> The logits grow with d_k, so softmax saturates and the row collapses onto a single token. Entropy has dropped to ${fmt(H, 3)} bits.</span>`
        : causal
          ? `${INFO}<span><strong>Causal mask on.</strong> Token ${query + 1} can only attend to itself and earlier positions — the upper triangle is set to −∞ before the softmax, so those weights become exactly 0.</span>`
          : `${INFO}<span>Row ${query + 1} is a probability distribution over all ${T} tokens, summing to 1. The output for "${TOKENS[query]}" is that weighted average of the value vectors.</span>`,
      !scaled ? 'warn' : 'info');
    plot.render();
  }

  plot.onDraw(p => {
    const cell = 1.15, ox = 3.2, oy = 8.6;
    // matrix
    for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
      const v = A[i][j];
      const X = p.X(ox + j * cell), Y = p.Y(oy - i * cell);
      p.ctx.fillStyle = withA(C.c1, .05 + v * .9);
      p.ctx.fillRect(X + 1, Y + 1, p.px(cell) - 2, p.py(cell) - 2);
      if (i === query) {
        p.ctx.strokeStyle = C.c2; p.ctx.lineWidth = 2;
        p.ctx.strokeRect(X + 1, Y + 1, p.px(cell) - 2, p.py(cell) - 2);
      }
      if (v > 0.005) {
        p.text([ox + j * cell + cell / 2, oy - i * cell - cell / 2], fmt(v, 2),
          { align: 'center', size: 10, weight: 600, mono: true, color: v > .5 ? C.raised : C.ink });
      }
    }
    // labels
    TOKENS.forEach((t, j) => {
      p.text([ox + j * cell + cell / 2, oy + .35], t, { align: 'center', size: 10.5, color: C.muted, weight: 600 });
      p.text([ox - .25, oy - j * cell - cell / 2], t,
        { align: 'right', size: 10.5, color: j === query ? C.c2 : C.muted, weight: j === query ? 700 : 600 });
    });
    p.text([ox + T * cell / 2, oy + .95], 'keys  (what each position offers)', { align: 'center', size: 10.5, color: C.muted });
    p.ctx.save();
    p.ctx.translate(p.X(.55), p.Y(oy - T * cell / 2));
    p.ctx.rotate(-Math.PI / 2);
    p.ctx.font = `600 10.5px ${getComputedStyle(document.documentElement).getPropertyValue('--font-sans')}`;
    p.ctx.fillStyle = C.muted; p.ctx.textAlign = 'center';
    p.ctx.fillText('queries', 0, 0);
    p.ctx.restore();

    // the selected row as a bar chart
    const by = 1.0;
    A[query].forEach((v, j) => {
      const x = ox + j * cell;
      p.ctx.fillStyle = C.c2; p.ctx.globalAlpha = .85;
      p.ctx.fillRect(p.X(x + .12), p.Y(by + v * 1.15), p.px(cell - .24), p.py(v * 1.15));
      p.ctx.globalAlpha = 1;
    });
    p.line([ox, by], [ox + T * cell, by], { color: C.axis, lw: 1.2 });
    p.text([ox + T * cell / 2, .55], `attention weights for "${TOKENS[query]}" — they sum to 1`,
      { align: 'center', size: 10.5, color: C.muted });
  });

  refresh();

  node.appendChild(note(
    `Row <em>i</em> of this matrix says how much token <em>i</em> draws on every other token when building ` +
    `its own representation. It is produced by <span class="u-mono">softmax(QKᵀ/√d_k)</span>, so each row is ` +
    `a probability distribution, and the output is <span class="u-mono">AV</span> — a weighted average of the ` +
    `value vectors. <strong>Switch off the √d_k scaling</strong> and watch the rows collapse onto a single ` +
    `token: without it the logits grow with d_k, softmax saturates, and gradients die. Switch on the ` +
    `<strong>causal mask</strong> and the upper triangle goes to exactly zero — that is what makes ` +
    `autoregressive generation possible.`
  ));
});

/* ============================================================
   10. Why divide by sqrt(d_k)
   ============================================================ */
defineWidget('attention-scaling', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: 0, xmax: 1, ymin: 0, ymax: 1, aspect: 1.31, equal: false, pad: 0,
  }));

  let dk = 64, scaled = true, T = 12;

  const dCtl = slider('d_k', { min: 2, max: 512, step: 2, value: 64, format: v => String(v), onInput: v => { dk = v; refresh(); } });
  const sCtl = toggle('Divide by √d_k', { value: true, onChange: v => { scaled = v; refresh(); } });
  const out = readout([['d_k', 0], ['predicted logit sd', 0], ['measured logit sd', 0], ['max softmax weight', 0], ['row entropy', 0]]);
  const st = status('');
  right.append(dCtl.root, sCtl.root, out.root, st.root);

  let logits = [], weights = [], sd = 0;
  function refresh() {
    const r = ML.rng(808);
    const q = Array.from({ length: dk }, () => ML.gauss(r));
    const ks = Array.from({ length: T }, () => Array.from({ length: dk }, () => ML.gauss(r)));
    logits = ks.map(k => {
      const d = q.reduce((s, v, i) => s + v * k[i], 0);
      return scaled ? d / Math.sqrt(dk) : d;
    });
    const m = Math.max(...logits);
    const e = logits.map(v => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    weights = e.map(v => v / s);
    const mu = logits.reduce((a, b) => a + b, 0) / T;
    sd = Math.sqrt(logits.reduce((a, b) => a + (b - mu) ** 2, 0) / T);
    const H = -weights.reduce((a, v) => a + (v > 1e-12 ? v * Math.log2(v) : 0), 0);
    out.set([
      String(dk),
      scaled ? '1.00 (constant)' : fmt(Math.sqrt(dk), 2),
      fmt(sd, 3),
      { html: fmt(Math.max(...weights), 4), cls: Math.max(...weights) > .9 ? 'is-warn' : 'is-ok' },
      { html: `${fmt(H, 3)} bits`, cls: H < .5 ? 'is-warn' : 'is-ok' },
    ]);
    st.set(
      !scaled && dk > 100
        ? `${WARN}<span><strong>Saturated.</strong> With d_k = ${dk} and no scaling the logit spread is ±${fmt(sd, 1)}, so softmax puts ${fmt(Math.max(...weights) * 100, 1)}% of its mass on one token. Its gradient is then almost exactly zero — the layer stops learning.</span>`
        : scaled
          ? `${OK}<span>Scaling by 1/√d_k keeps the logit standard deviation near <strong>1</strong> whatever d_k is, so the softmax stays in its responsive range.</span>`
          : `${INFO}<span>At small d_k the difference is mild. Slide d_k up to 512 with scaling off and watch the distribution collapse.</span>`,
      (!scaled && dk > 100) ? 'warn' : scaled ? 'ok' : 'info');
    plot.render();
  }

  plot.onDraw(p => {
    // Scale to the data rather than pinning the axis at 1: with a healthy
    // distribution every weight is under 0.2 and a fixed 0–1 axis crushes the
    // bars into a sliver. The band below zero holds the tick labels so they
    // never sit on top of the bars.
    const top = Math.max(1 / T * 1.9, Math.max(...weights) * 1.28, .12);
    const yTop = Math.min(1.1, top);
    p.o.xmin = -.6; p.o.xmax = T; p.o.ymin = -yTop * .2; p.o.ymax = yTop * 1.16;
    p._computeScale();
    p.grid(yTop / 4, { color: C.grid });
    const bw = p.px(.72);
    weights.forEach((w, i) => {
      const x = p.X(i + .14);
      p.ctx.fillStyle = C.c1; p.ctx.globalAlpha = .88;
      p.ctx.fillRect(x, p.Y(Math.min(w, yTop)), bw, p.Y(0) - p.Y(Math.min(w, yTop)));
      p.ctx.globalAlpha = 1;
      if (w > yTop * .06) {
        p.text([i + .5, Math.min(w, yTop)], fmt(w, 2),
          { align: 'center', dy: -9, size: 9.5, color: C.muted, halo: true, haloWidth: 3 });
      }
    });
    p.line([-.6, 1 / T], [T, 1 / T], { color: C.c3, lw: 1.6, dash: [5, 4] });
    p.axes();
    p.ticks(1, { stepX: 2, stepY: yTop / 2, bottom: 17 });
    p.legend([[C.c3, `uniform = ${fmt(1 / T, 3)}`, [5, 4]]],
      { corner: 'tr', title: `softmax weights over ${T} keys` });
    p.xlabel('key index', { size: 10.5 });
  });

  refresh();

  node.appendChild(note(
    `If the query and key entries are independent with variance σ², their dot product over d_k dimensions has ` +
    `variance <span class="u-mono">d_k·σ⁴</span> — it grows with the dimension. Feed those ever-larger logits ` +
    `into a softmax and it saturates: one weight goes to 1, the rest to 0, and since the softmax gradient is ` +
    `proportional to <span class="u-mono">o(1−o)</span>, <strong>the gradient vanishes</strong>. Dividing by ` +
    `√d_k cancels the growth exactly, holding the logit spread near 1 whatever the dimension. Turn the ` +
    `scaling off and push d_k to 512 to see the failure it prevents.`
  ));
});

/* ============================================================
   What you see versus what the model sees
   ============================================================ */
defineWidget('image-numbers', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, { xmin: 0, xmax: 1, ymin: 0, ymax: 1, aspect: 1.23, equal: true, pad: 0 }));

  const N = 16;
  /* A small grayscale glyph, drawn analytically so it needs no asset. */
  const SHAPES = {
    seven: { label: 'A digit', f: (x, y) => {
      const bar = y > .72 && x > .18 && x < .84 ? 1 : 0;
      const dx = x - (.86 - .62 * y);
      const stroke = Math.abs(dx) < .09 && y < .78 ? 1 : 0;
      return Math.max(bar, stroke);
    } },
    edge: { label: 'A hard edge', f: (x) => (x > .5 ? .92 : .12) },
    grad: { label: 'A gradient', f: (x, y) => .1 + .8 * (x * .6 + y * .4) },
    disc: { label: 'A disc', f: (x, y) => (Math.hypot(x - .5, y - .5) < .32 ? .9 : .1) },
  };
  let key = 'seven', mode = 'both', channel = 'gray';

  const sCtl = segmented(Object.entries(SHAPES).map(([k, v]) => ({ label: v.label, value: k })),
    { value: 'seven', label: 'Image', onChange: v => { key = v; refresh(); } });
  const mCtl = segmented([
    { label: 'What you see', value: 'image' },
    { label: 'Both', value: 'both' },
    { label: 'What the model sees', value: 'numbers' },
  ], { value: 'both', label: 'View', onChange: v => { mode = v; plot.render(); } });
  const cCtl = segmented([{ label: 'Grayscale', value: 'gray' }, { label: 'RGB', value: 'rgb' }],
    { value: 'gray', label: 'Channels', onChange: v => { channel = v; refresh(); } });
  const out = readout([['grid', 0], ['numbers in this image', 0], ['as a tensor', 0], ['at 224×224 RGB instead', 0]]);
  const st = status('');
  right.append(sCtl.root, mCtl.root, cCtl.root, out.root, st.root);

  let pix = [];
  function refresh() {
    const f = SHAPES[key].f;
    pix = [];
    for (let r = 0; r < N; r++) {
      const row = [];
      for (let c = 0; c < N; c++) {
        const x = (c + .5) / N, y = 1 - (r + .5) / N;
        row.push(clamp(f(x, y), 0, 1));
      }
      pix.push(row);
    }
    const ch = channel === 'rgb' ? 3 : 1;
    out.set([
      `${N} × ${N}`,
      { html: (N * N * ch).toLocaleString(), cls: 'is-ok' },
      { html: channel === 'rgb' ? `ℝ<sup>${N}×${N}×3</sup>` : `ℝ<sup>${N}×${N}</sup>` },
      { html: (224 * 224 * 3).toLocaleString() + ' numbers', cls: 'is-warn' },
    ]);
    st.set(
      `${INFO}<span>An image is an array of numbers and nothing else. This ${N}×${N} thumbnail is already ` +
      `${(N * N * ch).toLocaleString()} of them; a modest 224×224 colour photograph is <strong>150,528</strong>. ` +
      `Flatten that and hand it to a dense layer of 1000 units and you have committed to over 150 million weights.</span>`,
      'info');
    plot.render();
  }

  plot.onDraw(p => {
    p.o.xmin = 0; p.o.xmax = N; p.o.ymin = 0; p.o.ymax = N;
    p._computeScale();
    p.clear(null);
    const showImg = mode !== 'numbers';
    const showNum = mode !== 'image';
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const v = pix[r][c];
        const [sx, sy] = p.toScreen([c, N - r - 1]);
        const w = Math.abs(p.px(1)), h = Math.abs(p.px(1));
        if (showImg) {
          const g = Math.round(v * 255);
          p.ctx.fillStyle = `rgb(${g},${g},${g})`;
          p.ctx.fillRect(sx, sy - h, w + .5, h + .5);
        } else {
          p.ctx.fillStyle = C.bg;
          p.ctx.fillRect(sx, sy - h, w + .5, h + .5);
        }
        if (showNum) {
          const val = Math.round(v * 255);
          p.ctx.font = `600 ${Math.max(7, Math.min(11, w * .42))}px ${css('--font-mono')}`;
          p.ctx.textAlign = 'center'; p.ctx.textBaseline = 'middle';
          p.ctx.fillStyle = showImg ? (v > .5 ? '#000' : '#fff') : C.ink;
          p.ctx.fillText(String(val), sx + w / 2, sy - h / 2);
        }
        if (!showImg) {
          p.ctx.strokeStyle = withA(C.grid, .8); p.ctx.lineWidth = .6;
          p.ctx.strokeRect(sx, sy - h, w, h);
        }
      }
    }
  });

  refresh();

  node.appendChild(note(
    `Switch between the three views. You see a shape; the model receives a grid of integers in [0, 255] and ` +
    `never sees anything else. Two consequences follow immediately, and they are the entire motivation for ` +
    `convolution. First, <strong>the count explodes</strong> — pixels scale with area, so a dense first layer ` +
    `becomes unaffordable almost at once. Second, and worse, <strong>flattening destroys adjacency</strong>: ` +
    `once this grid is a flat vector, the number at position 37 has no recorded relationship to the one at 53, ` +
    `even though they were neighbours. A dense layer would have to rediscover that from data, separately for ` +
    `every position in the image.`
  ));
});

/* ============================================================
   Valid versus same padding
   ============================================================ */
defineWidget('padding-modes', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, { xmin: 0, xmax: 1, ymin: 0, ymax: 1, aspect: 1.39, equal: true, pad: 0 }));

  let H = 7, F = 3, mode = 'same', layers = 1;
  const hCtl = slider('input size H', { min: 5, max: 12, step: 1, value: 7, format: v => `${v}×${v}`, onInput: v => { H = v; refresh(); } });
  const fCtl = slider('kernel F', { min: 2, max: 5, step: 1, value: 3, format: v => `${v}×${v}`, onInput: v => { F = v; refresh(); } });
  const mCtl = segmented([{ label: 'Valid (P = 0)', value: 'valid' }, { label: 'Same', value: 'same' }],
    { value: 'same', label: 'Padding', onChange: v => { mode = v; refresh(); } });
  const lCtl = slider('stacked layers', { min: 1, max: 10, step: 1, value: 1, format: v => String(v), onInput: v => { layers = v; refresh(); } });
  const out = readout([['padding P', 0], ['output after 1 layer', 0], [`output after ${1} layers`, 0], ['corner pixel used by', 0], ['centre pixel used by', 0]]);
  const st = status('');
  right.append(hCtl.root, fCtl.root, mCtl.root, lCtl.root, out.root, st.root);

  const P = () => (mode === 'same' ? Math.floor(F / 2) : 0);
  const outDim = h => Math.floor((h + 2 * P() - F) / 1) + 1;

  function refresh() {
    let h = H;
    const chain = [h];
    for (let i = 0; i < layers; i++) { h = Math.max(0, outDim(h)); chain.push(h); }
    // how many output positions read the corner vs the centre input pixel (single layer)
    const p = P();
    const countFor = (r, c) => {
      let n = 0;
      const o = outDim(H);
      for (let i = 0; i < o; i++) for (let j = 0; j < o; j++) {
        const r0 = i - p, c0 = j - p;
        if (r >= r0 && r < r0 + F && c >= c0 && c < c0 + F) n++;
      }
      return n;
    };
    out.root.querySelectorAll('dt')[2].textContent = `output after ${layers} layer${layers === 1 ? '' : 's'}`;
    out.set([
      mode === 'same' ? `⌊F/2⌋ = ${p}` : '0',
      `${outDim(H)} × ${outDim(H)}`,
      { html: `${h} × ${h}`, cls: h === 0 ? 'is-warn' : (h === H ? 'is-ok' : '') },
      `${countFor(0, 0)} output${countFor(0, 0) === 1 ? '' : 's'}`,
      `${countFor(Math.floor(H / 2), Math.floor(H / 2))} outputs`,
    ]);
    st.set(
      mode === 'same' && F % 2 === 1
        ? `${OK}<span><strong>Same padding holds the size fixed</strong> at ${H}×${H}, so you can stack as many layers as you like. That is why almost every modern architecture uses it.</span>`
        : mode === 'same'
          ? `${WARN}<span>With an <strong>even</strong> kernel, "same" cannot be exactly symmetric — ⌊F/2⌋ padding gives ${outDim(H)}×${outDim(H)}, not ${H}×${H}. Same padding is only exact for odd F, which is one reason odd kernels dominate.</span>`
          : h === 0
            ? `${WARN}<span><strong>The feature map has vanished.</strong> Valid padding removes F−1 = ${F - 1} pixels per layer, so after ${layers} layers there is nothing left to convolve.</span>`
            : `${INFO}<span>Valid padding shrinks the map by F−1 = ${F - 1} each layer: ${chain.join(' → ')}. The corner pixel is read by only ${countFor(0, 0)} output, against ${countFor(Math.floor(H / 2), Math.floor(H / 2))} for the centre — border information is systematically under-used.</span>`,
      mode === 'same' && F % 2 === 1 ? 'ok' : (h === 0 || (mode === 'same' && F % 2 === 0) ? 'warn' : 'info'));
    plot.render();
  }

  plot.onDraw(p => {
    const pad = P(), tot = H + 2 * pad;
    const o = outDim(H);
    const gap = 1.6;
    const W = tot + gap + Math.max(o, 1);
    p.o.xmin = -.6; p.o.xmax = W + .6;
    p.o.ymin = -1.4; p.o.ymax = Math.max(tot, o) + 1.2;
    p._computeScale();
    p.clear(null);
    const cell = (x, y, fill, stroke, dash) => {
      const [sx, sy] = p.toScreen([x, y + 1]);
      const w = Math.abs(p.px(1));
      if (fill) { p.ctx.fillStyle = fill; p.ctx.fillRect(sx, sy, w, w); }
      p.ctx.strokeStyle = stroke; p.ctx.lineWidth = 1;
      if (dash) p.ctx.setLineDash([3, 3]);
      p.ctx.strokeRect(sx, sy, w, w);
      p.ctx.setLineDash([]);
    };
    // input, with the padding ring
    for (let r = 0; r < tot; r++) {
      for (let c = 0; c < tot; c++) {
        const isPad = r < pad || c < pad || r >= tot - pad || c >= tot - pad;
        cell(c, tot - r - 1,
          isPad ? withA(C.c2, .13) : withA(C.c1, .16),
          isPad ? withA(C.c2, .55) : withA(C.c1, .5),
          isPad);
        if (isPad) {
          const [sx, sy] = p.toScreen([c + .5, tot - r - .5]);
          p.ctx.font = `600 ${Math.max(6, Math.min(10, Math.abs(p.px(1)) * .4))}px ${css('--font-mono')}`;
          p.ctx.textAlign = 'center'; p.ctx.textBaseline = 'middle';
          p.ctx.fillStyle = withA(C.c2, .85);
          p.ctx.fillText('0', sx, sy);
        }
      }
    }
    // kernel footprint at the top-left valid position
    const [kx, ky] = p.toScreen([0, tot]);
    p.ctx.strokeStyle = C.c3; p.ctx.lineWidth = 2.4;
    p.ctx.strokeRect(kx, ky, Math.abs(p.px(F)), Math.abs(p.px(F)));

    p.text([tot / 2, -.5], `input ${H}×${H}${pad ? ` + ${pad} ring` : ''}`, { align: 'center', size: 11, color: C.muted });

    // output
    const ox = tot + gap;
    for (let r = 0; r < o; r++) for (let c = 0; c < o; c++) {
      cell(ox + c, o - r - 1, withA(C.c3, .18), withA(C.c3, .55));
    }
    p.text([ox + o / 2, -.5], `output ${o}×${o}`, { align: 'center', size: 11, color: C.muted });
    // arrow
    const a0 = p.toScreen([tot + .25, Math.max(tot, o) / 2]);
    const a1 = p.toScreen([ox - .25, Math.max(tot, o) / 2]);
    p.ctx.strokeStyle = C.muted; p.ctx.lineWidth = 1.8;
    p.ctx.beginPath(); p.ctx.moveTo(a0[0], a0[1]); p.ctx.lineTo(a1[0], a1[1]); p.ctx.stroke();
    p.ctx.beginPath();
    p.ctx.moveTo(a1[0], a1[1]); p.ctx.lineTo(a1[0] - 7, a1[1] - 5); p.ctx.lineTo(a1[0] - 7, a1[1] + 5);
    p.ctx.fillStyle = C.muted; p.ctx.fill();
    p.legend([[C.c1, 'real pixels'], [C.c2, 'zero padding'], [C.c3, 'kernel / output']], { corner: 'tr' });
  });

  refresh();

  node.appendChild(note(
    `Valid padding uses only positions where the kernel fits entirely inside the image, so each layer eats ` +
    `F−1 pixels and the map shrinks. Push the layer count up with valid padding and watch it disappear ` +
    `altogether — that is the practical reason deep stacks need padding. Same padding adds a ring of ⌊F/2⌋ ` +
    `zeros so the output matches the input exactly, but only when F is <strong>odd</strong>; try F = 2 or 4 and ` +
    `the arithmetic no longer lands. There is a second, subtler reason to pad: without it a corner pixel is ` +
    `read by a single output while a central pixel is read by F² of them, so the network sees the borders far ` +
    `less often than the middle.`
  ));
});

/* ============================================================
   What the layers end up detecting
   ============================================================ */
defineWidget('feature-hierarchy', node => {
  const wrap = el('div');
  node.appendChild(wrap);

  const N = 26;
  /* Stage-1 filters are the classic oriented edges and blobs; stages 2 and 3
     are built by composing stage-1 responses, which is exactly the claim. */
  const K1 = {
    'vertical edge': [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    'horizontal edge': [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    'diagonal ╱': [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
    'diagonal ╲': [[2, 1, 0], [1, 0, -1], [0, -1, -2]],
    'blob / centre-surround': [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
    'low-pass': [[1, 1, 1], [1, 1, 1], [1, 1, 1]].map(r => r.map(v => v / 9)),
  };
  let stage = 1;

  const scene = (x, y) => {
    // a small synthetic "face-like" scene: two dark discs, a bar, a rounded outline
    let v = .82;
    const inD = (cx, cy, r) => Math.hypot(x - cx, y - cy) < r;
    if (Math.hypot((x - .5) / .40, (y - .52) / .46) > 1) v = .30;         // background outside head
    if (inD(.36, .64, .075) || inD(.64, .64, .075)) v = .10;              // eyes
    if (Math.abs(y - .36) < .035 && Math.abs(x - .5) < .16) v = .16;      // mouth
    if (Math.abs(x - .5) < .022 && y > .42 && y < .58) v = .55;           // nose
    return v;
  };

  const img = [];
  for (let r = 0; r < N; r++) {
    const row = [];
    for (let c = 0; c < N; c++) row.push(scene((c + .5) / N, 1 - (r + .5) / N));
    img.push(row);
  }
  function conv(src, k) {
    const n = src.length, f = k.length, off = (f - 1) / 2;
    const out = [];
    for (let r = 0; r < n; r++) {
      const row = [];
      for (let c = 0; c < n; c++) {
        let s = 0;
        for (let u = 0; u < f; u++) for (let v = 0; v < f; v++) {
          const rr = clamp(r + u - off, 0, n - 1), cc = clamp(c + v - off, 0, n - 1);
          s += k[u][v] * src[rr][cc];
        }
        row.push(Math.max(0, s));                              // ReLU
      }
      out.push(row);
    }
    return out;
  }
  const stage1 = Object.fromEntries(Object.entries(K1).map(([k, v]) => [k, conv(img, v)]));
  // stage 2: combine pairs of stage-1 maps -> corners and textures
  const combo = (a, b) => a.map((row, r) => row.map((v, c) => Math.max(0, v * .5 + b[r][c] * .5 - .18)));
  const stage2 = {
    'corner (V + H)': combo(stage1['vertical edge'], stage1['horizontal edge']),
    'texture (╱ + ╲)': combo(stage1['diagonal ╱'], stage1['diagonal ╲']),
    'curve (H + blob)': combo(stage1['horizontal edge'], stage1['blob / centre-surround']),
    'spot cluster': combo(stage1['blob / centre-surround'], stage1['low-pass']),
  };
  const stage3 = {
    'eye-like part': combo(stage2['corner (V + H)'], stage2['spot cluster']),
    'mouth-like part': combo(stage2['curve (H + blob)'], stage1['horizontal edge']),
    'whole-object template': combo(stage2['texture (╱ + ╲)'], stage2['curve (H + blob)']),
  };

  const sCtl = segmented([
    { label: '1 · edges & blobs', value: 1 },
    { label: '2 · corners & textures', value: 2 },
    { label: '3 · parts & objects', value: 3 },
  ], { value: 1, label: 'Layer depth', onChange: v => { stage = Number(v); refresh(); } });
  const st = status('');
  const gridHost = el('div', { style: 'display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:.6rem;margin-top:.7rem' });
  const inputHost = el('div', { style: 'max-width:190px' });

  wrap.appendChild(el('div', { class: 'pg-split pg-split--wide-ctrl' },
    el('div', {}, el('div', { class: 'matrix-label', html: 'Input image' }), inputHost),
    el('div', { class: 'pg-controls' }, sCtl.root, st.root)));
  wrap.appendChild(gridHost);

  function drawMap(cv, m, label) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = cv.clientWidth || 110;
    if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(w * dpr); }
    const ctx = cv.getContext('2d');
    let hi = 1e-6;
    for (const row of m) for (const v of row) if (v > hi) hi = v;
    const tmp = document.createElement('canvas');
    tmp.width = m.length; tmp.height = m.length;
    const id = tmp.getContext('2d').createImageData(m.length, m.length);
    for (let r = 0; r < m.length; r++) for (let c = 0; c < m.length; c++) {
      const g = Math.round(255 * clamp(m[r][c] / hi, 0, 1));
      const k = (r * m.length + c) * 4;
      id.data[k] = g; id.data[k + 1] = g; id.data[k + 2] = g; id.data[k + 3] = 255;
    }
    tmp.getContext('2d').putImageData(id, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(tmp, 0, 0, cv.width, cv.height);
  }

  const inCv = el('canvas', { style: 'width:100%;aspect-ratio:1;display:block;border-radius:8px' });
  inputHost.appendChild(inCv);

  function refresh() {
    const maps = stage === 1 ? stage1 : stage === 2 ? stage2 : stage3;
    gridHost.innerHTML = '';
    Object.entries(maps).forEach(([name, m]) => {
      const cv = el('canvas', { style: 'width:100%;aspect-ratio:1;display:block;border-radius:8px' });
      gridHost.appendChild(el('div', {},
        cv, el('div', { style: 'font-size:.72rem;color:var(--ink-muted);margin-top:.25rem;text-align:center', text: name })));
      requestAnimationFrame(() => drawMap(cv, m));
    });
    requestAnimationFrame(() => drawMap(inCv, img));
    st.set(
      stage === 1
        ? `${INFO}<span><strong>Layer 1 sees only 3×3 patches.</strong> There is nothing else it can detect but oriented edges, blobs and flat regions — and this is what trained networks actually learn here, near-universally.</span>`
        : stage === 2
          ? `${INFO}<span><strong>Layer 2 combines layer-1 maps.</strong> A corner is just a vertical edge <em>and</em> a horizontal edge in the same place; a texture is a mix of orientations. Nothing new is invented — the units are compositions.</span>`
          : `${OK}<span><strong>Layer 3 assembles parts.</strong> By now the receptive field covers most of the image, so a unit can respond to an eye-like or mouth-like configuration rather than any single edge. Nobody designed this hierarchy; it falls out of stacking local operations.</span>`,
      stage === 3 ? 'ok' : 'info');
  }

  refresh();

  node.appendChild(note(
    `These maps are computed live from hand-written kernels, then composed — not photographs of a trained ` +
    `network — but the composition is the honest part of the story. A first-layer unit sees a 3×3 window and ` +
    `so <em>cannot</em> represent anything more than an oriented edge or a blob. Each later stage takes weighted ` +
    `combinations of the stage below, so its units answer questions like "is there a vertical edge and a ` +
    `horizontal edge at the same place?" — a corner. Stack enough of these and the receptive field covers the ` +
    `whole image, at which point a unit can fire for an object part. Compare with ` +
    `<a href="#rf">the receptive-field figure</a>: the hierarchy of <em>features</em> and the growth of the ` +
    `<em>window</em> are the same phenomenon seen from two sides.`
  ));
});

/* ============================================================
   The revolution of depth
   ============================================================ */
defineWidget('arch-evolution', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, { xmin: 0, xmax: 1, ymin: 0, ymax: 1, aspect: 1.27, equal: false, pad: 0 }));

  /* Every number here is from the chapter's own summary table, plus the
     published ImageNet top-5 errors for the winning entries. */
  const MODELS = [
    { name: 'Shallow (2010)', year: 2010, depth: 3, err: 28.2, params: null, note: 'Pre-deep-learning ILSVRC winner — hand-designed features.' },
    { name: 'Shallow (2011)', year: 2011, depth: 4, err: 25.8, params: null, note: 'Still shallow, still hand-designed.' },
    { name: 'AlexNet', year: 2012, depth: 8, err: 16.4, params: 60, note: '5 conv + 3 FC. GPUs, ReLU, dropout. The result that restarted deep learning.' },
    { name: 'ZFNet', year: 2013, depth: 8, err: 11.7, params: 60, note: 'AlexNet with tuned kernel sizes and strides.' },
    { name: 'VGG-16', year: 2014, depth: 16, err: 7.3, params: 138, note: 'Uniform 3×3 stacks. Simple and repeatable, but a huge fully connected tail.' },
    { name: 'GoogLeNet', year: 2014, depth: 22, err: 6.7, params: 6.8, note: 'Inception modules with 1×1 bottlenecks — 20× fewer parameters than AlexNet.' },
    { name: 'ResNet-152', year: 2015, depth: 152, err: 3.57, params: 60, note: 'Residual connections. Depth stopped being a liability.' },
    { name: 'MobileNetV1', year: 2017, depth: 28, err: 10.5, params: 4.2, note: 'Depthwise separable convolutions, engineered for phones.' },
    { name: 'EfficientNet-B0', year: 2019, depth: 19, err: 6.7, params: 5.3, note: 'Compound scaling of depth, width and resolution together.' },
    { name: 'EfficientNet-B7', year: 2019, depth: 64, err: 2.9, params: 66, note: '84.4% top-1. Same principle, scaled up.' },
  ];
  let axis = 'depth', sel = 6;

  const aCtl = segmented([
    { label: 'Depth vs error', value: 'depth' },
    { label: 'Parameters vs error', value: 'params' },
    { label: 'Error over time', value: 'year' },
  ], { value: 'depth', label: 'View', onChange: v => { axis = v; refresh(); } });
  const sCtl = slider('highlight model', {
    min: 0, max: MODELS.length - 1, step: 1, value: 6,
    format: v => MODELS[v].name, onInput: v => { sel = v; refresh(); },
  });
  const out = readout([['model', 0], ['layers with parameters', 0], ['parameters', 0], ['top-5 ImageNet error', 0], ['error per million params', 0]]);
  const st = status('');
  right.append(aCtl.root, sCtl.root, out.root, st.root);

  function refresh() {
    const m = MODELS[sel];
    out.set([
      `${m.name} (${m.year})`,
      String(m.depth),
      m.params === null ? '—' : `≈ ${m.params}M`,
      { html: `${m.err}%`, cls: m.err < 5 ? 'is-ok' : '' },
      m.params === null ? '—' : fmt(m.err / m.params, 3),
    ]);
    st.set(`${INFO}<span><strong>${m.name}.</strong> ${m.note}</span>`, 'info');
    plot.render();
  }

  plot.onDraw(p => {
    const pts = MODELS.filter(m => (axis === 'params' ? m.params !== null : true));
    const xOf = m => (axis === 'depth' ? Math.log10(m.depth) : axis === 'params' ? Math.log10(m.params) : m.year);
    const xs = pts.map(xOf);
    const pad = (Math.max(...xs) - Math.min(...xs)) * .12 + .05;
    p.o.xmin = Math.min(...xs) - pad; p.o.xmax = Math.max(...xs) + pad;
    p.o.ymin = 0; p.o.ymax = 31;
    p._computeScale();
    p.grid(5, { color: C.grid });
    if (axis === 'year') {
      p.path(pts.slice().sort((a, b) => a.year - b.year).map(m => [m.year, m.err]),
        { color: withA(C.c1, .5), lw: 2, dash: [6, 4] });
    }
    pts.forEach(m => {
      const isSel = MODELS[sel] === m;
      const col = m.year <= 2011 ? C.c4 : m.err < 5 ? C.c3 : C.c1;
      p.dot([xOf(m), m.err], { r: isSel ? 9 : 5.5, color: withA(col, isSel ? 1 : .7) });
      if (isSel) p.badge([xOf(m), m.err], `${m.name} · ${m.err}%`, { color: col, align: 'center', dy: -18 });
    });
    p.axes();
    if (axis === 'year') p.ticks(5);
    else {
      p.ticks(5);
      // decade labels on the log axis — placed just above the caption band so
      // they stay inside the canvas (Y(0) is the bottom edge here)
      const band = (p.reserveBottom || 16) + 3;
      for (const v of [1, 10, 100, 1000]) {
        const lx = Math.log10(v);
        if (lx < p.o.xmin || lx > p.o.xmax) continue;
        p.text({ px: p.X(lx), py: p.h - band }, String(v),
          { align: 'center', baseline: 'bottom', size: 10.5, color: C.muted, weight: 500 });
      }
    }
    p.xlabel(axis === 'depth' ? 'layers with learnable parameters (log scale)'
      : axis === 'params' ? 'parameters, millions (log scale)' : 'year');
    p.legend([[C.c4, 'pre-deep-learning'], [C.c1, 'deep CNN'], [C.c3, 'below 5% error']],
      { corner: 'tr', title: 'top-5 ImageNet error (%)' });
  });

  refresh();

  node.appendChild(note(
    `Before 2015, networks stayed under about twenty layers — not from lack of ambition, but because adding ` +
    `layers made the <em>training</em> loss worse, which is an optimisation failure rather than overfitting. ` +
    `ResNet's identity shortcuts removed that barrier and 152 layers reached <strong>3.57%</strong> top-5 error. ` +
    `Then switch to the parameters view and notice that the story is not simply "bigger": GoogLeNet beat VGG-16 ` +
    `with <strong>twenty times fewer parameters</strong>, and EfficientNet-B0 matches it again at 5.3M. Depth and ` +
    `parameter count are different axes, and most of the progress after 2014 came from spending parameters ` +
    `better rather than spending more of them.`
  ));
});

/* ============================================================
   The three sequence supervision patterns
   ============================================================ */
defineWidget('seq-types', node => {
  const { left, right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, { xmin: 0, xmax: 1, ymin: 0, ymax: 1, aspect: 1.39, equal: false, pad: 0 }));

  const PATTERNS = {
    o2m: { label: 'One-to-many', task: 'Image captioning', ins: 1, outs: 5,
      why: 'A single image vector is encoded once, then the decoder emits one word per step until it produces a stop token.' },
    m2o: { label: 'Many-to-one', task: 'Sentiment analysis', ins: 5, outs: 1,
      why: 'The whole review is consumed and only the final hidden state is read out, so the label depends on the entire sequence.' },
    m2m: { label: 'Many-to-many (aligned)', task: 'Part-of-speech tagging', ins: 5, outs: 5,
      why: 'One output per input, emitted as you go. Input and output lengths match exactly.' },
    enc: { label: 'Many-to-many (encoder–decoder)', task: 'Machine translation', ins: 4, outs: 5,
      why: 'Read the whole source first, then generate. Input and output lengths are free to differ — essential when languages do not align word for word.' },
  };
  let key = 'm2m';

  const pCtl = segmented(Object.entries(PATTERNS).map(([k, v]) => ({ label: v.label, value: k })),
    { value: 'm2m', label: 'Pattern', onChange: v => { key = v; refresh(); } });
  const keyRow = el('div', { style: 'display:flex;flex-wrap:wrap;gap:.5rem 1.1rem;padding:.15rem .1rem 0' });
  left.appendChild(keyRow);
  const keyItem = (col, label) => el('span', {
    style: 'display:inline-flex;align-items:center;gap:.42em;font-size:.78rem;color:var(--ink-muted)',
  }, el('span', { style: `width:18px;height:0;border-top:3px solid ${col};border-radius:2px;flex:none` }), label);

  const out = readout([['task', 0], ['inputs', 0], ['outputs', 0], ['lengths must match', 0], ['loss is summed over', 0]]);
  const st = status('');
  right.append(pCtl.root, out.root, st.root);

  function refresh() {
    const P = PATTERNS[key];
    out.set([
      P.task, String(P.ins), String(P.outs),
      { html: key === 'm2m' ? 'yes' : 'no', cls: key === 'm2m' ? 'is-warn' : 'is-ok' },
      key === 'm2o' ? 'the final step only' : `all ${P.outs} output steps`,
    ]);
    st.set(`${INFO}<span><strong>${P.label} — ${P.task}.</strong> ${P.why}</span>`, 'info');
    keyRow.innerHTML = '';
    keyRow.append(keyItem(C.c5, 'input xₜ'), keyItem(C.c1, 'hidden state sₜ'), keyItem(C.c3, 'output ŷₜ'));
    plot.render();
  }

  plot.onDraw(p => {
    const P = PATTERNS[key];
    const T = Math.max(P.ins, P.outs);
    p.o.xmin = -.9; p.o.xmax = T + .4; p.o.ymin = -1.6; p.o.ymax = 4.35;
    p._computeScale();
    p.clear(null);
    const box = (x, y, label, col, filled) => {
      const [sx, sy] = p.toScreen([x, y]);
      const w = Math.abs(p.px(.62)), h = Math.abs(p.py(.52));
      p.ctx.beginPath();
      const r = 5;
      const x0 = sx - w / 2, y0 = sy - h / 2;
      p.ctx.moveTo(x0 + r, y0);
      p.ctx.arcTo(x0 + w, y0, x0 + w, y0 + h, r);
      p.ctx.arcTo(x0 + w, y0 + h, x0, y0 + h, r);
      p.ctx.arcTo(x0, y0 + h, x0, y0, r);
      p.ctx.arcTo(x0, y0, x0 + w, y0, r);
      p.ctx.closePath();
      p.ctx.fillStyle = filled ? withA(col, .2) : C.bg;
      p.ctx.fill();
      p.ctx.strokeStyle = col; p.ctx.lineWidth = 2; p.ctx.stroke();
      p.ctx.font = `700 11px ${css('--font-sans')}`;
      p.ctx.textAlign = 'center'; p.ctx.textBaseline = 'middle';
      p.ctx.fillStyle = C.ink;
      p.ctx.fillText(label, sx, sy);
    };
    const arrow = (a, b, col, dash) => {
      const [x1, y1] = p.toScreen(a), [x2, y2] = p.toScreen(b);
      p.ctx.strokeStyle = col; p.ctx.lineWidth = 1.8;
      if (dash) p.ctx.setLineDash([4, 4]);
      p.ctx.beginPath(); p.ctx.moveTo(x1, y1); p.ctx.lineTo(x2, y2); p.ctx.stroke();
      p.ctx.setLineDash([]);
      const ang = Math.atan2(y2 - y1, x2 - x1);
      p.ctx.beginPath();
      p.ctx.moveTo(x2, y2);
      p.ctx.lineTo(x2 - 7 * Math.cos(ang - .4), y2 - 7 * Math.sin(ang - .4));
      p.ctx.lineTo(x2 - 7 * Math.cos(ang + .4), y2 - 7 * Math.sin(ang + .4));
      p.ctx.fillStyle = col; p.ctx.fill();
    };

    const encEnd = key === 'enc' ? P.ins - 1 : T - 1;
    for (let t = 0; t < T; t++) {
      // hidden state row
      box(t, 1, `s${'₀₁₂₃₄₅₆₇₈₉'[t + 1]}`, C.c1, true);
      if (t > 0) arrow([t - 1 + .34, 1], [t - .34, 1], C.c1);
      // inputs
      const hasIn = key === 'o2m' ? t === 0 : t < P.ins;
      if (hasIn) {
        box(t, -.6, `x${'₀₁₂₃₄₅₆₇₈₉'[t + 1]}`, C.c5, false);
        arrow([t, -.32], [t, .72], C.c5);
      }
      // outputs
      const hasOut = key === 'm2o' ? t === T - 1
        : key === 'o2m' ? true
          : key === 'enc' ? t >= P.ins - 1 && t - (P.ins - 1) < P.outs
            : t < P.outs;
      if (hasOut) {
        box(t, 2.6, `ŷ${'₀₁₂₃₄₅₆₇₈₉'[t + 1]}`, C.c3, false);
        arrow([t, 1.28], [t, 2.32], C.c3);
      }
    }
    if (key === 'enc') {
      const [bx] = p.toScreen([P.ins - 1 + .5, 0]);
      p.ctx.strokeStyle = withA(C.muted, .7); p.ctx.lineWidth = 1.6;
      p.ctx.setLineDash([5, 5]);
      p.ctx.beginPath();
      p.ctx.moveTo(bx, p.Y(3.0)); p.ctx.lineTo(bx, p.Y(-1.3));
      p.ctx.stroke(); p.ctx.setLineDash([]);
      p.text([P.ins / 2 - .6, -1.2], 'encoder', { align: 'center', size: 10.5, color: C.muted });
      p.text([P.ins + 1.4, -1.2], 'decoder', { align: 'center', size: 10.5, color: C.muted });
    }
    p.title(PATTERNS[key].label);
  });

  refresh();

  node.appendChild(note(
    `One recurrence, four supervision regimes. The cell is identical in every case — what changes is only ` +
    `<em>where you attach inputs and read outputs</em>. That flexibility is a large part of why RNNs took over ` +
    `sequence modelling: you do not need a new architecture per task, only a new wiring of the same ` +
    `\\(\\mathbf{U}, \\mathbf{W}, \\mathbf{V}\\). Note the difference between the two many-to-many forms — the ` +
    `aligned version needs input and output lengths to match, which rules out translation, while the ` +
    `encoder–decoder version reads everything before emitting anything and so has no such constraint.`
  ));
});

/* ============================================================
   Rolled, unrolled, and one set of weights everywhere
   ============================================================ */
defineWidget('rnn-sharing', node => {
  const { left, right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, { xmin: 0, xmax: 1, ymin: 0, ymax: 1, aspect: 1.39, equal: false, pad: 0 }));

  let view = 'unrolled', T = 4, hi = null, inside = false;
  const vCtl = segmented([{ label: 'Rolled', value: 'rolled' }, { label: 'Unrolled', value: 'unrolled' }],
    { value: 'unrolled', label: 'View', onChange: v => { view = v; refresh(); } });
  const tCtl = slider('sequence length T', { min: 2, max: 8, step: 1, value: 4, format: v => String(v), onInput: v => { T = v; refresh(); } });
  const iCtl = toggle('Open up the cell', { value: false, onChange: v => { inside = v; refresh(); } });
  const hCtl = segmented([
    { label: 'none', value: '' }, { label: 'U', value: 'U' }, { label: 'W', value: 'W' }, { label: 'V', value: 'V' },
  ], { value: '', label: 'Highlight a shared matrix', onChange: v => { hi = v || null; plot.render(); } });
  const colFor = k => (hi === k ? C.c2 : (k === 'U' ? C.c5 : k === 'W' ? C.c1 : C.c3));

  const keyRow = el('div', { style: 'display:flex;flex-wrap:wrap;gap:.5rem 1.1rem;padding:.15rem .1rem 0' });
  left.appendChild(keyRow);
  const keyItem = (col, label) => el('span', {
    style: 'display:inline-flex;align-items:center;gap:.42em;font-size:.78rem;color:var(--ink-muted)',
  }, el('span', { style: `width:18px;height:0;border-top:3px solid ${col};border-radius:2px;flex:none` }), label);

  const out = readout([['U — reads the input', 0], ['W — carries the state', 0], ['V — reads out', 0], ['total parameters', 0], ['if each step had its own', 0]]);
  const st = status('');
  right.append(vCtl.root, tCtl.root, iCtl.root, hCtl.root, out.root, st.root);

  const d = 4, m = 3;                        // vocab 4, hidden 3 — the chapter's example
  function refresh() {
    const per = d * m + m * m + m * d + m + d;
    out.set([
      `${m} × ${d} = ${m * d}`,
      `${m} × ${m} = ${m * m}`,
      `${d} × ${m} = ${d * m}`,
      { html: `${per} — <strong>independent of T</strong>`, cls: 'is-ok' },
      { html: `${per * T} at T = ${T}`, cls: 'is-warn' },
    ]);
    st.set(
      hi
        ? `${OK}<span><strong>${hi}</strong> is the <em>same matrix</em> at every highlighted position — not ${T} copies. That is weight sharing along time, and it is why an RNN handles any sequence length with a fixed parameter count.</span>`
        : view === 'rolled'
          ? `${INFO}<span>The rolled view is the honest one: there is a single cell with a self-loop. Unroll it to see why backpropagation works.</span>`
          : `${INFO}<span>Unrolled, an RNN is just a deep feedforward network of depth T — whose layers happen to share every weight. Highlight U, W or V to see the sharing.</span>`,
      hi ? 'ok' : 'info');
    keyRow.innerHTML = '';
    keyRow.append(keyItem(colFor('U'), 'U — input → state'),
                  keyItem(colFor('W'), 'W — state → state'),
                  keyItem(colFor('V'), 'V — state → output'));
    plot.render();
  }

  plot.onDraw(p => {
    p.clear(null);
    const drawBox = (x, y, w, h, label, col, filled, lw) => {
      const [sx, sy] = p.toScreen([x, y]);
      const pw = Math.abs(p.px(w)), ph = Math.abs(p.py(h));
      const x0 = sx - pw / 2, y0 = sy - ph / 2, r = 6;
      p.ctx.beginPath();
      p.ctx.moveTo(x0 + r, y0);
      p.ctx.arcTo(x0 + pw, y0, x0 + pw, y0 + ph, r);
      p.ctx.arcTo(x0 + pw, y0 + ph, x0, y0 + ph, r);
      p.ctx.arcTo(x0, y0 + ph, x0, y0, r);
      p.ctx.arcTo(x0, y0, x0 + pw, y0, r);
      p.ctx.closePath();
      p.ctx.fillStyle = filled ? withA(col, .18) : C.bg;
      p.ctx.fill();
      p.ctx.strokeStyle = col; p.ctx.lineWidth = lw || 2; p.ctx.stroke();
      if (label) {
        p.ctx.font = `700 11.5px ${css('--font-sans')}`;
        p.ctx.textAlign = 'center'; p.ctx.textBaseline = 'middle';
        p.ctx.fillStyle = C.ink;
        p.ctx.fillText(label, sx, sy);
      }
    };
    const arrow = (a, b, col, lw, label) => {
      const [x1, y1] = p.toScreen(a), [x2, y2] = p.toScreen(b);
      p.ctx.strokeStyle = col; p.ctx.lineWidth = lw || 1.8;
      p.ctx.beginPath(); p.ctx.moveTo(x1, y1); p.ctx.lineTo(x2, y2); p.ctx.stroke();
      const ang = Math.atan2(y2 - y1, x2 - x1);
      p.ctx.beginPath();
      p.ctx.moveTo(x2, y2);
      p.ctx.lineTo(x2 - 8 * Math.cos(ang - .38), y2 - 8 * Math.sin(ang - .38));
      p.ctx.lineTo(x2 - 8 * Math.cos(ang + .38), y2 - 8 * Math.sin(ang + .38));
      p.ctx.fillStyle = col; p.ctx.fill();
      if (label) {
        p.ctx.font = `800 11px ${css('--font-mono')}`;
        p.ctx.textAlign = 'center'; p.ctx.textBaseline = 'middle';
        const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
        p.ctx.save();
        p.ctx.fillStyle = C.raised;
        p.ctx.beginPath(); p.ctx.arc(mx, my, 9, 0, Math.PI * 2); p.ctx.fill();
        p.ctx.fillStyle = col;
        p.ctx.fillText(label, mx, my);
        p.ctx.restore();
      }
    };
    const lwFor = k => (hi === k ? 3.4 : 1.8);

    if (view === 'rolled') {
      p.o.xmin = -1.6; p.o.xmax = 1.6; p.o.ymin = -1.6; p.o.ymax = 2.75;
      p._computeScale();
      drawBox(0, 0, 1.1, .7, inside ? '' : 'sₜ', C.c1, true, hi === 'W' ? 3.4 : 2);
      if (inside) {
        p.text([0, .16], 'φ(U xₜ + W sₜ₋₁ + b)', { align: 'center', size: 10.5, color: C.ink, weight: 700 });
        p.text([0, -.16], 'tanh', { align: 'center', size: 10, color: C.muted });
      }
      drawBox(0, -1.1, .8, .5, 'xₜ', C.c5, false);
      drawBox(0, 1.15, .8, .5, 'ŷₜ', C.c3, false);
      arrow([0, -.85], [0, -.37], colFor('U'), lwFor('U'), 'U');
      arrow([0, .37], [0, .9], colFor('V'), lwFor('V'), 'V');
      // self loop
      const c = p.toScreen([0, 0]);
      const rr = Math.abs(p.px(.62));
      p.ctx.strokeStyle = colFor('W'); p.ctx.lineWidth = lwFor('W');
      p.ctx.beginPath();
      p.ctx.arc(c[0] + rr * .95, c[1], rr * .58, Math.PI * .62, Math.PI * 1.38, true);
      p.ctx.stroke();
      p.ctx.font = `800 11px ${css('--font-mono')}`;
      p.ctx.textAlign = 'center'; p.ctx.textBaseline = 'middle';
      p.ctx.fillStyle = colFor('W');
      p.ctx.fillText('W', c[0] + rr * 1.75, c[1]);
    } else {
      p.o.xmin = -1.1; p.o.xmax = T + .3; p.o.ymin = -1.6; p.o.ymax = 2.75;
      p._computeScale();
      const sub = '₀₁₂₃₄₅₆₇₈₉';
      drawBox(-.66, 0, .5, .5, 's₀', C.muted, false);
      for (let t = 0; t < T; t++) {
        drawBox(t, 0, .58, .62, inside ? '' : `s${sub[t + 1]}`, C.c1, true, hi === 'W' ? 3.4 : 2);
        if (inside) p.text([t, 0], 'tanh', { align: 'center', size: 9, color: C.muted });
        drawBox(t, -1.1, .58, .46, `x${sub[t + 1]}`, C.c5, false);
        drawBox(t, 1.15, .58, .46, `ŷ${sub[t + 1]}`, C.c3, false);
        arrow([t, -.86], [t, -.34], colFor('U'), lwFor('U'), 'U');
        arrow([t, .34], [t, .9], colFor('V'), lwFor('V'), 'V');
        arrow([t - 1 + (t === 0 ? .62 : .32), 0], [t - .32, 0], colFor('W'), lwFor('W'), 'W');
      }
    }
    p.title(hi ? `${hi} is one matrix, reused at every step`
      : 'the same three matrices at every step');
  });

  refresh();

  node.appendChild(note(
    `The rolled diagram shows what an RNN <em>is</em>: one cell with a loop. The unrolled diagram shows what ` +
    `training <em>does</em> with it — lay out T copies and backpropagate through the lot. The crucial point is ` +
    `that unrolling copies the <strong>computation</strong>, not the <strong>parameters</strong>. Highlight U, ` +
    `W or V and every arrow of that colour lights up at once, because they are all the same matrix. The ` +
    `parameter readout makes the consequence concrete: the count does not contain T anywhere, so one model ` +
    `handles a four-word sentence and a four-hundred-word document alike. It is also why the gradient for W ` +
    `accumulates a contribution from every time step, which is where ` +
    `<a href="#bptt">backpropagation through time</a> begins.`
  ));
});

/* ============================================================
   Inside an LSTM and a GRU cell
   ============================================================ */
defineWidget('gate-cell', node => {
  const { left, right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, { xmin: 0, xmax: 1, ymin: 0, ymax: 1, aspect: 1.23, equal: false, pad: 0 }));

  let kind = 'lstm', f = .9, i = .4, o = .7, z = .3, r = .8, cPrev = .6, cand = .5;
  const kCtl = segmented([{ label: 'LSTM', value: 'lstm' }, { label: 'GRU', value: 'gru' }],
    { value: 'lstm', label: 'Cell', onChange: v => { kind = v; rebuild(); } });
  const g1 = slider('forget gate f', { min: 0, max: 1, step: .02, value: .9, onInput: v => { f = v; refresh(); } });
  const g2 = slider('input gate i', { min: 0, max: 1, step: .02, value: .4, onInput: v => { i = v; refresh(); } });
  const g3 = slider('output gate o', { min: 0, max: 1, step: .02, value: .7, onInput: v => { o = v; refresh(); } });
  const g4 = slider('update gate z', { min: 0, max: 1, step: .02, value: .3, onInput: v => { z = v; refresh(); } });
  const g5 = slider('reset gate r', { min: 0, max: 1, step: .02, value: .8, onInput: v => { r = v; refresh(); } });
  const g6 = slider('previous state', { min: -1, max: 1, step: .05, value: .6, onInput: v => { cPrev = v; refresh(); } });
  const g7 = slider('candidate', { min: -1, max: 1, step: .05, value: .5, onInput: v => { cand = v; refresh(); } });
  const presets = el('div', { class: 'pg-actions' },
    button('Remember', () => { f = 1; i = 0; z = 0; g1.set(1); g2.set(0); g4.set(0); refresh(); }),
    button('Overwrite', () => { f = 0; i = 1; z = 1; g1.set(0); g2.set(1); g4.set(1); refresh(); }),
    button('Blend', () => { f = .6; i = .5; z = .5; g1.set(.6); g2.set(.5); g4.set(.5); refresh(); }));
  const keyRow = el('div', { style: 'display:flex;flex-wrap:wrap;gap:.5rem 1.1rem;padding:.15rem .1rem 0' });
  left.appendChild(keyRow);
  const keyItem = (col, label) => el('span', {
    style: 'display:inline-flex;align-items:center;gap:.42em;font-size:.78rem;color:var(--ink-muted)',
  }, el('span', { style: `width:18px;height:0;border-top:3px solid ${col};border-radius:2px;flex:none` }), label);

  const out = readout([['carried through', 0], ['written in', 0], ['new state', 0], ['∂ new / ∂ old', 0], ['after 20 steps', 0]]);
  const st = status('');
  right.append(kCtl.root, g1.root, g2.root, g3.root, g4.root, g5.root, g6.root, g7.root, presets, out.root, st.root);

  function rebuild() {
    [g1, g2, g3].forEach(s => { s.root.style.display = kind === 'lstm' ? '' : 'none'; });
    [g4, g5].forEach(s => { s.root.style.display = kind === 'gru' ? '' : 'none'; });
    refresh();
  }
  function refresh() {
    let carried, written, next, deriv;
    if (kind === 'lstm') {
      carried = f * cPrev; written = i * cand; next = carried + written; deriv = f;
    } else {
      carried = (1 - z) * cPrev; written = z * cand; next = carried + written; deriv = 1 - z;
    }
    const after = Math.pow(deriv, 20);
    out.set([
      fmt(carried, 4), fmt(written, 4),
      { html: fmt(next, 4), cls: 'is-ok' },
      { html: fmt(deriv, 4), cls: deriv > .8 ? 'is-ok' : deriv < .3 ? 'is-warn' : '' },
      { html: after < 1e-4 ? after.toExponential(2) : fmt(after, 5), cls: after > .01 ? 'is-ok' : 'is-warn' },
    ]);
    st.set(
      deriv > .95
        ? `${OK}<span><strong>The carousel is open.</strong> The derivative along the carry path is ${fmt(deriv, 3)}, so after 20 steps the gradient is still scaled by ${fmt(after, 4)}. A vanilla RNN with tanh′ ≈ 0.25 and w = 0.8 would be at 1.1×10⁻¹⁴.</span>`
        : deriv < .2
          ? `${WARN}<span>The gate is nearly shut: only ${fmt(deriv * 100, 0)}% of the old state survives each step, so information from 20 steps back is scaled by ${after < 1e-4 ? after.toExponential(2) : fmt(after, 5)}. Useful when you <em>want</em> to forget — fatal when you do not.</span>`
          : `${INFO}<span>Partly open. The cell is mixing old state and new candidate; the derivative along the carry path is ${fmt(deriv, 3)}.</span>`,
      deriv > .95 ? 'ok' : deriv < .2 ? 'warn' : 'info');
    keyRow.innerHTML = '';
    keyRow.append(keyItem(C.c3, 'carry path — no weights, no tanh'),
                  keyItem(C.c4, kind === 'lstm' ? 'forget gate' : 'carry gate (1 − z)'),
                  keyItem(C.c1, kind === 'lstm' ? 'input gate' : 'update gate z'),
                  keyItem(C.c5, 'candidate'));
    plot.render();
  }

  plot.onDraw(p => {
    p.o.xmin = -.2; p.o.xmax = 10.2; p.o.ymin = -2.6; p.o.ymax = 2.9;
    p._computeScale();
    p.clear(null);
    const node2 = (x, y, rad, label, col) => {
      const [sx, sy] = p.toScreen([x, y]);
      p.ctx.beginPath(); p.ctx.arc(sx, sy, Math.abs(p.px(rad)), 0, Math.PI * 2);
      p.ctx.fillStyle = C.bg; p.ctx.fill();
      p.ctx.strokeStyle = col; p.ctx.lineWidth = 2.2; p.ctx.stroke();
      p.ctx.font = `700 13px ${css('--font-sans')}`;
      p.ctx.textAlign = 'center'; p.ctx.textBaseline = 'middle';
      p.ctx.fillStyle = col; p.ctx.fillText(label, sx, sy);
    };
    const gateBox = (x, y, label, val, col) => {
      const [sx, sy] = p.toScreen([x, y]);
      const w = Math.abs(p.px(1.15)), h = Math.abs(p.py(.62));
      p.ctx.fillStyle = withA(col, .13 + .5 * val);
      p.ctx.strokeStyle = col; p.ctx.lineWidth = 2;
      p.ctx.beginPath();
      const r2 = 6, x0 = sx - w / 2, y0 = sy - h / 2;
      p.ctx.moveTo(x0 + r2, y0);
      p.ctx.arcTo(x0 + w, y0, x0 + w, y0 + h, r2);
      p.ctx.arcTo(x0 + w, y0 + h, x0, y0 + h, r2);
      p.ctx.arcTo(x0, y0 + h, x0, y0, r2);
      p.ctx.arcTo(x0, y0, x0 + w, y0, r2);
      p.ctx.closePath(); p.ctx.fill(); p.ctx.stroke();
      p.ctx.font = `700 11px ${css('--font-sans')}`;
      p.ctx.textAlign = 'center'; p.ctx.textBaseline = 'middle';
      p.ctx.fillStyle = C.ink;
      p.ctx.fillText(`${label} = ${fmt(val, 2)}`, sx, sy);
    };
    const line = (a, b, col, lw, dash) => {
      const [x1, y1] = p.toScreen(a), [x2, y2] = p.toScreen(b);
      p.ctx.strokeStyle = col; p.ctx.lineWidth = lw;
      if (dash) p.ctx.setLineDash(dash);
      p.ctx.beginPath(); p.ctx.moveTo(x1, y1); p.ctx.lineTo(x2, y2); p.ctx.stroke();
      p.ctx.setLineDash([]);
    };

    const carryVal = kind === 'lstm' ? f : 1 - z;
    // the carry highway across the top — width encodes how open it is
    line([0, 1.7], [10, 1.7], withA(C.c3, .35 + .6 * carryVal), 2 + 7 * carryVal);
    p.text([0.1, 2.25], kind === 'lstm' ? 'cell state cₜ₋₁ → cₜ  (the constant error carousel)' : 'carry path (1 − z)·sₜ₋₁',
      { size: 10.5, color: C.c3, weight: 700 });

    if (kind === 'lstm') {
      gateBox(2.2, .1, 'f', f, C.c4);
      node2(2.2, 1.7, .28, '×', C.c4);
      line([2.2, .41], [2.2, 1.42], C.c4, 2);
      gateBox(4.6, .1, 'i', i, C.c1);
      gateBox(4.6, -1.1, 'c̃', (cand + 1) / 2, C.c5);
      node2(4.6, 1.7, .28, '×', C.c1);
      line([4.6, .41], [4.6, 1.42], C.c1, 2);
      line([4.6, -.79], [4.6, -.21], C.c5, 2);
      node2(6.2, 1.7, .28, '+', C.c3);
      gateBox(8.2, .1, 'o', o, C.c2);
      node2(8.2, 1.7, .28, '×', C.c2);
      line([8.2, .41], [8.2, 1.42], C.c2, 2);
      line([8.2, 1.42], [8.2, -1.9], withA(C.c2, .6), 2, [4, 4]);
      p.text([8.2, -2.2], 'sₜ = o ⊙ tanh(cₜ)', { align: 'center', size: 10.5, color: C.muted });
      p.text([2.2, -.55], 'forget', { align: 'center', size: 9.5, color: C.muted });
      p.text([4.6, -.55], 'input', { align: 'center', size: 9.5, color: C.muted });
      p.text([8.2, -.55], 'output', { align: 'center', size: 9.5, color: C.muted });
    } else {
      gateBox(2.6, .1, '1−z', 1 - z, C.c4);
      node2(2.6, 1.7, .28, '×', C.c4);
      line([2.6, .41], [2.6, 1.42], C.c4, 2);
      gateBox(5.4, .1, 'z', z, C.c1);
      gateBox(5.4, -1.1, 's̃', (cand + 1) / 2, C.c5);
      gateBox(3.9, -2.1, 'r', r, C.c2);
      node2(5.4, 1.7, .28, '×', C.c1);
      line([5.4, .41], [5.4, 1.42], C.c1, 2);
      line([5.4, -.79], [5.4, -.21], C.c5, 2);
      line([4.5, -2.1], [5.4, -1.41], withA(C.c2, .7), 2);
      node2(7.4, 1.7, .28, '+', C.c3);
      p.text([7.4, 2.25], 'sₜ', { align: 'center', size: 11, color: C.c3, weight: 700 });
    }
    p.text([0.1, -2.45], `∂sₜ/∂sₜ₋₁ along the carry path ≈ ${fmt(carryVal, 3)}`,
      { size: 11, color: carryVal > .8 ? C.c3 : C.c4, weight: 700 });
  });

  rebuild();

  node.appendChild(note(
    `Both cells are built from the same two primitives: a <strong>sigmoid gate</strong> producing a number in ` +
    `(0, 1), and an <strong>elementwise multiply</strong> that uses it to scale a vector. The thick line across ` +
    `the top is the part that matters — in the LSTM it is the cell state, updated as ` +
    `<span class="u-mono">cₜ = f ⊙ cₜ₋₁ + i ⊙ c̃ₜ</span>, and in the GRU it is the ` +
    `<span class="u-mono">(1 − z) ⊙ sₜ₋₁</span> term. Either way there is <strong>no weight matrix and no tanh ` +
    `on that path</strong>, so its derivative is just the gate value. Open the gate and the line thickens; the ` +
    `gradient then survives twenty steps essentially intact, against roughly 10⁻¹⁴ for a vanilla RNN. Press ` +
    `<strong>Remember</strong> and <strong>Overwrite</strong> to see the two extremes the network can choose ` +
    `between, per unit and per step.`
  ));
});

/* ============================================================
   Tokens → ids → embeddings
   ============================================================ */
defineWidget('tokenize', node => {
  const wrap = el('div');
  node.appendChild(wrap);

  const SENTENCES = {
    a: 'the cat sat on the mat',
    b: 'unbelievably transformative results',
    c: 'the cats sat on the mats',
  };
  let key = 'a', gran = 'sub', dim = 6;

  const sCtl = segmented([
    { label: '"the cat sat on the mat"', value: 'a' },
    { label: 'rare words', value: 'b' },
    { label: 'plurals', value: 'c' },
  ], { value: 'a', label: 'Sentence', onChange: v => { key = v; refresh(); } });
  const gCtl = segmented([
    { label: 'Whole words', value: 'word' }, { label: 'Subwords', value: 'sub' }, { label: 'Characters', value: 'char' },
  ], { value: 'sub', label: 'Tokenisation', onChange: v => { gran = v; refresh(); } });
  const dCtl = slider('d_model (shown)', { min: 4, max: 10, step: 1, value: 6, format: v => String(v), onInput: v => { dim = v; refresh(); } });
  const out = readout([['tokens', 0], ['vocabulary needed', 0], ['embedding matrix E', 0], ['X, the layer input', 0]]);
  const st = status('');

  const tokHost = el('div', { style: 'margin-top:.6rem' });
  const left = el('div', {}, tokHost);
  const right = el('div', { class: 'pg-controls' }, sCtl.root, gCtl.root, dCtl.root, out.root, st.root);
  wrap.appendChild(el('div', { class: 'pg-split pg-split--wide-ctrl' }, left, right));

  /* A toy subword vocabulary — enough to show the mechanism honestly. */
  const SUBS = ['un', 'believ', 'ably', 'transform', 'ative', 'result', 's', 'the', 'cat', 'sat', 'on', 'mat'];
  function tokenize(text) {
    if (gran === 'char') return [...text.replace(/ /g, '_')];
    const words = text.split(' ');
    if (gran === 'word') return words;
    const out2 = [];
    for (const w of words) {
      let rest = w, guard = 0;
      const parts = [];
      while (rest.length && guard++ < 20) {
        const hit = SUBS.filter(s => rest.startsWith(s)).sort((a, b) => b.length - a.length)[0];
        if (hit) { parts.push(hit); rest = rest.slice(hit.length); }
        else { parts.push(rest[0]); rest = rest.slice(1); }
      }
      out2.push(...parts.map((s, i) => (i ? '##' + s : s)));
    }
    return out2;
  }
  const hash = s => { let h = 2166136261; for (const c of s) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619); } return h >>> 0; };
  const embedOf = (tok, k) => {
    const h = hash(tok + '|' + k);
    return ((h % 2000) / 1000 - 1) * .9;
  };

  function refresh() {
    const toks = tokenize(SENTENCES[key]);
    const uniq = [...new Set(toks)];
    const vocabEst = gran === 'word' ? '~170,000 (every English word form)'
      : gran === 'sub' ? '~30,000–50,000 (typical BPE)'
        : '~100 (letters and punctuation)';
    out.set([
      `${toks.length} tokens, ${uniq.length} distinct`,
      vocabEst,
      `|V| × ${dim}`,
      { html: `${toks.length} × ${dim}`, cls: 'is-ok' },
    ]);
    st.set(
      gran === 'word'
        ? `${WARN}<span><strong>Whole words are brittle.</strong> Any word not in the vocabulary becomes a single unknown token, and "cat"/"cats" are unrelated entries that must each be learned from scratch.</span>`
        : gran === 'char'
          ? `${WARN}<span><strong>Characters never go out of vocabulary</strong>, but sequences get long — and attention costs O(T²), so length is expensive.</span>`
          : `${OK}<span><strong>Subwords are the compromise everyone uses.</strong> Common words stay whole, rare ones split into familiar pieces (note the <span class="u-mono">##</span> continuation marks), so the vocabulary stays bounded and nothing is ever truly unknown.</span>`,
      gran === 'sub' ? 'ok' : 'warn');

    tokHost.innerHTML = '';
    const row = (label, cells, cls) => {
      const r = el('div', { style: 'display:flex;gap:.3rem;align-items:center;margin-bottom:.45rem;flex-wrap:wrap' });
      r.appendChild(el('span', { style: 'font-size:.72rem;color:var(--ink-faint);min-width:74px', text: label }));
      cells.forEach(c => r.appendChild(c));
      tokHost.appendChild(r);
      void cls;
    };
    row('text', toks.map(t => el('span', {
      class: 'pill', style: 'font-family:var(--font-mono);font-size:.76rem', text: t,
    })));
    row('token id', toks.map(t => el('span', {
      style: 'font-family:var(--font-mono);font-size:.72rem;padding:.18rem .4rem;border-radius:5px;' +
             'background:var(--bg-sunken);color:var(--ink-muted)',
      text: String(hash(t) % 30000),
    })));
    const grid = el('div', { style: `display:grid;grid-template-columns:repeat(${dim},minmax(0,1fr));gap:2px;max-width:${dim * 44}px` });
    toks.forEach(t => {
      for (let k = 0; k < dim; k++) {
        const v = embedOf(t, k);
        grid.appendChild(el('span', {
          style: `font-family:var(--font-mono);font-size:.6rem;text-align:center;padding:.2rem 0;border-radius:3px;` +
                 `background:${withA(v >= 0 ? C.c2 : C.c1, .12 + .5 * Math.abs(v))};color:var(--ink)`,
          text: v.toFixed(1),
        }));
      }
    });
    tokHost.appendChild(el('div', { style: 'display:flex;gap:.55rem;align-items:flex-start;margin-top:.5rem' },
      el('span', { style: 'font-size:.72rem;color:var(--ink-faint);min-width:74px;padding-top:.3rem', text: `X (${toks.length}×${dim})` }),
      grid));
  }

  refresh();

  node.appendChild(note(
    `Three steps, and only the last one is learned. <strong>Tokenise</strong> splits the text into units; ` +
    `<strong>look up</strong> maps each unit to an integer id; <strong>embed</strong> maps each id to a row of ` +
    `the matrix \\(\\mathbf{E} \\in \\mathbb{R}^{|\\mathcal{V}| \\times d_{\\text{model}}}\\). Stack those rows and ` +
    `you have \\(\\mathbf{X} \\in \\mathbb{R}^{T \\times d_{\\text{model}}}\\), which is what every later layer ` +
    `actually consumes. Compare the three granularities on the plurals sentence: whole-word tokenisation gives ` +
    `"cat" and "cats" unrelated ids with nothing shared between them, while subwords split off a reusable ` +
    `<span class="u-mono">##s</span>. Unlike one-hot vectors, embeddings are dense and trained, so related ` +
    `tokens can end up near one another — a representation the model builds for itself.`
  ));
});

/* ============================================================
   Building Q, K and V, and the attention that follows
   ============================================================ */
defineWidget('qkv-build', node => {
  const wrap = el('div');
  node.appendChild(wrap);

  const TOKENS = ['the', 'cat', 'sat', 'on', 'the', 'mat'];
  const T = TOKENS.length, dm = 4;
  let dk = 3, step = 3, qi = 1, scaled = true;

  const stCtl = segmented([
    { label: '1 · X', value: 1 }, { label: '2 · Q,K,V', value: 2 },
    { label: '3 · QKᵀ', value: 3 }, { label: '4 · softmax', value: 4 }, { label: '5 · ×V', value: 5 },
  ], { value: 3, label: 'Stage', onChange: v => { step = Number(v); refresh(); } });
  const kCtl = slider('d_k', { min: 2, max: 6, step: 1, value: 3, format: v => String(v), onInput: v => { dk = v; rebuild(); } });
  const qCtl = slider('query token i', { min: 0, max: T - 1, step: 1, value: 1, format: v => `"${TOKENS[v]}"`, onInput: v => { qi = v; refresh(); } });
  const scCtl = toggle('divide by √d_k', { value: true, onChange: v => { scaled = v; refresh(); } });
  const out = readout([['X', 0], ['W^Q, W^K, W^V', 0], ['Q, K, V', 0], ['scores QKᵀ', 0], ['attends most to', 0]]);
  const st = status('');

  const host = el('div', { style: 'display:grid;gap:.75rem;margin-top:.4rem' });
  const left = el('div', {}, host);
  const right = el('div', { class: 'pg-controls' }, stCtl.root, kCtl.root, qCtl.root, scCtl.root, out.root, st.root);
  wrap.appendChild(el('div', { class: 'pg-split pg-split--wide-ctrl' }, left, right));

  const hash = s => { let h = 2166136261; for (const c of s) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619); } return h >>> 0; };
  let X = [], WQ = [], WK = [], WV = [], Q = [], K = [], V = [], E = [], A = [], O = [];

  const mul = (Am, Bm) => Am.map(row => Bm[0].map((_, j) => row.reduce((s, v, k) => s + v * Bm[k][j], 0)));
  function rebuild() {
    X = TOKENS.map((t, i) => Array.from({ length: dm }, (_, k) => ((hash(t + '#' + k + '#' + i) % 2000) / 1000 - 1) * .9));
    const mk = (rows, cols, tag) => Array.from({ length: rows }, (_, a) =>
      Array.from({ length: cols }, (_, b) => ((hash(tag + a + '.' + b) % 2000) / 1000 - 1) * .8));
    WQ = mk(dm, dk, 'Q'); WK = mk(dm, dk, 'K'); WV = mk(dm, dk, 'V');
    refresh();
  }
  function refresh() {
    Q = mul(X, WQ); K = mul(X, WK); V = mul(X, WV);
    const scale = scaled ? 1 / Math.sqrt(dk) : 1;
    E = Q.map(q => K.map(k => q.reduce((s, v, t) => s + v * k[t], 0) * scale));
    A = E.map(row => {
      const m = Math.max(...row);
      const e = row.map(v => Math.exp(v - m));
      const s = e.reduce((a, b) => a + b, 0);
      return e.map(v => v / s);
    });
    O = A.map(a => V[0].map((_, j) => a.reduce((s, w, t) => s + w * V[t][j], 0)));
    const best = A[qi].indexOf(Math.max(...A[qi]));
    out.set([
      `${T} × ${dm}`,
      `${dm} × ${dk} each`,
      `${T} × ${dk} each`,
      { html: `${T} × ${T}`, cls: 'is-warn' },
      { html: `"${TOKENS[best]}" at ${fmt(A[qi][best] * 100, 1)}%`, cls: 'is-ok' },
    ]);
    st.set(
      step === 1 ? `${INFO}<span>Every token is a row of <strong>X</strong>. Nothing here knows about any other token yet.</span>`
        : step === 2 ? `${INFO}<span>Three <em>different</em> linear maps of the same X. A token's <strong>query</strong> is what it is looking for, its <strong>key</strong> is what it offers, its <strong>value</strong> is what it passes on if selected.</span>`
          : step === 3 ? `${INFO}<span>Entry (i, j) is <span class="u-mono">qᵢ·kⱼ</span> — how well token i's query matches token j's key. This is the only place tokens ever see each other, and it costs <strong>T² = ${T * T}</strong> dot products.</span>`
            : step === 4 ? `${OK}<span>Row-wise softmax turns each row into a distribution over all ${T} positions. Row ${qi + 1} sums to <strong>${fmt(A[qi].reduce((s, v) => s + v, 0), 6)}</strong>.</span>`
              : `${OK}<span>Each output row is a weighted average of <em>value</em> vectors, using that row of A as the weights. Token "${TOKENS[qi]}" is now a blend of the whole sentence.</span>`,
      step >= 4 ? 'ok' : 'info');

    host.innerHTML = '';
    const heat = (M, title, opts = {}) => {
      const { rowLab, colLab, hlRow, pct, sym = true } = opts;
      const cols = M[0].length;
      let hi = 1e-9;
      M.forEach(r => r.forEach(v => { hi = Math.max(hi, Math.abs(v)); }));
      const box = el('div');
      box.appendChild(el('div', { class: 'matrix-label', html: title }));
      const g = el('div', { style: `display:grid;grid-template-columns:auto repeat(${cols},minmax(0,1fr));gap:2px;max-width:${90 + cols * 52}px` });
      g.appendChild(el('span'));
      for (let j = 0; j < cols; j++) {
        g.appendChild(el('span', {
          style: 'font-size:.62rem;color:var(--ink-faint);text-align:center',
          text: colLab ? colLab[j] : String(j + 1),
        }));
      }
      M.forEach((row, i) => {
        g.appendChild(el('span', {
          style: `font-size:.66rem;color:${hlRow === i ? 'var(--ink)' : 'var(--ink-faint)'};` +
                 `font-weight:${hlRow === i ? 700 : 400};text-align:right;padding-right:.25rem;white-space:nowrap`,
          text: rowLab ? rowLab[i] : String(i + 1),
        }));
        row.forEach(v => {
          const t = sym ? Math.abs(v) / hi : v / hi;
          const col = sym ? (v >= 0 ? C.c2 : C.c1) : C.c1;
          g.appendChild(el('span', {
            style: `font-family:var(--font-mono);font-size:.6rem;text-align:center;padding:.24rem 0;border-radius:3px;` +
                   `background:${withA(col, .1 + .55 * t)};` +
                   `outline:${hlRow === i ? '1.5px solid ' + C.c3 : 'none'}`,
            text: pct ? (v * 100).toFixed(0) + '%' : v.toFixed(2),
          }));
        });
      });
      box.appendChild(g);
      host.appendChild(box);
    };

    if (step === 1) heat(X, `<strong>X</strong> — token embeddings (${T} × ${dm})`, { rowLab: TOKENS, hlRow: qi });
    if (step === 2) {
      heat(Q, `<strong>Q = X W<sup>Q</sup></strong> (${T} × ${dk})`, { rowLab: TOKENS, hlRow: qi });
      heat(K, `<strong>K = X W<sup>K</sup></strong> (${T} × ${dk})`, { rowLab: TOKENS });
      heat(V, `<strong>V = X W<sup>V</sup></strong> (${T} × ${dk})`, { rowLab: TOKENS });
    }
    if (step === 3) heat(E, `<strong>QK<sup>T</sup>${scaled ? ' / √d_k' : ''}</strong> — raw scores (${T} × ${T})`,
      { rowLab: TOKENS, colLab: TOKENS, hlRow: qi });
    if (step === 4) heat(A, `<strong>A = softmax(QK<sup>T</sup>/√d_k)</strong> — each row sums to 1`,
      { rowLab: TOKENS, colLab: TOKENS, hlRow: qi, pct: true, sym: false });
    if (step === 5) {
      heat(A, `<strong>A</strong> — attention weights`, { rowLab: TOKENS, colLab: TOKENS, hlRow: qi, pct: true, sym: false });
      heat(O, `<strong>A V</strong> — the output, one blended vector per token (${T} × ${dk})`, { rowLab: TOKENS, hlRow: qi });
    }
  }

  rebuild();

  node.appendChild(note(
    `Walk the five stages in order and the whole of self-attention is there. The three projection matrices ` +
    `\\(\\mathbf{W}^Q, \\mathbf{W}^K, \\mathbf{W}^V\\) are the <em>only</em> learned parameters — everything else ` +
    `is fixed arithmetic. Stage 3 is where the quadratic cost lives: a \\(T \\times T\\) score matrix, ` +
    `${T * T} dot products for six tokens and a million for a thousand. Stage 4's softmax is applied ` +
    `<strong>row-wise</strong>, so each token gets its own distribution over the sentence, and stage 5 uses ` +
    `those as blending weights over the value vectors. Turn off the \\(\\sqrt{d_k}\\) division and push ` +
    `\\(d_k\\) up to watch the distribution collapse onto a single token — the failure mode ` +
    `<a href="#scaling">the scaling figure</a> quantifies.`
  ));
});

/* ============================================================
   Multi-head attention
   ============================================================ */
defineWidget('mha-heads', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, { xmin: 0, xmax: 1, ymin: 0, ymax: 1, aspect: 1.27, equal: false, pad: 0 }));

  let dModel = 64, h = 8, showHead = 0;
  const dCtl = slider('d_model', { min: 16, max: 256, step: 16, value: 64, format: v => String(v), onInput: v => { dModel = v; refresh(); } });
  const hCtl = slider('heads h', { min: 1, max: 16, step: 1, value: 8, format: v => String(v), onInput: v => { h = v; refresh(); } });
  const out = readout([['per-head d_k = d_v', 0], ['params per head', 0], ['params, all heads', 0], ['single head at full width', 0], ['scaling uses', 0]]);
  const st = status('');
  right.append(dCtl.root, hCtl.root, out.root, st.root);

  function refresh() {
    const dk = Math.floor(dModel / h);
    const perHead = 3 * dModel * dk;
    const all = perHead * h;
    const single = 3 * dModel * dModel;
    showHead = Math.min(showHead, h - 1);
    out.set([
      { html: `${dModel} / ${h} = ${dk}`, cls: dModel % h === 0 ? 'is-ok' : 'is-warn' },
      `3 · ${dModel} · ${dk} = ${perHead.toLocaleString()}`,
      { html: all.toLocaleString(), cls: 'is-ok' },
      single.toLocaleString(),
      { html: `1/√${dk} = ${fmt(1 / Math.sqrt(dk), 4)}`, cls: 'is-ok' },
    ]);
    st.set(
      dModel % h !== 0
        ? `${WARN}<span>d_model = ${dModel} is not divisible by h = ${h}. Real implementations require it to be — the concatenation has to come back to exactly d_model.</span>`
        : `${OK}<span><strong>${h} heads cost the same as one.</strong> Each head works in ${dk} dimensions instead of ${dModel}, so ${all.toLocaleString()} parameters total against ${single.toLocaleString()} for a single full-width head. You get ${h} different relationships for the price of one — and note the scaling factor is 1/√${dk}, the <em>per-head</em> dimension, not 1/√${dModel}.</span>`,
      dModel % h === 0 ? 'ok' : 'warn');
    plot.render();
  }

  plot.onDraw(p => {
    const dk = Math.max(1, Math.floor(dModel / h));
    p.o.xmin = -.4; p.o.xmax = 10.4; p.o.ymin = -.6; p.o.ymax = Math.max(h, 4) + .8;
    p._computeScale();
    p.clear(null);
    const rect = (x, y, w, ht, col, label, alpha) => {
      const [sx, sy] = p.toScreen([x, y + ht]);
      const pw = Math.abs(p.px(w)), ph = Math.abs(p.py(ht));
      p.ctx.fillStyle = withA(col, alpha ?? .2);
      p.ctx.fillRect(sx, sy, pw, ph);
      p.ctx.strokeStyle = col; p.ctx.lineWidth = 1.4;
      p.ctx.strokeRect(sx, sy, pw, ph);
      if (label && ph > 11) {
        p.ctx.font = `600 ${Math.min(10, ph * .7)}px ${css('--font-sans')}`;
        p.ctx.textAlign = 'center'; p.ctx.textBaseline = 'middle';
        p.ctx.fillStyle = C.ink;
        p.ctx.fillText(label, sx + pw / 2, sy + ph / 2);
      }
    };
    const H = Math.max(h, 1);
    const rowH = Math.min(.82, (Math.max(h, 4) - .2) / H);
    // input X
    rect(0, (Math.max(h, 4) - 1.6) / 2, 1.1, 1.6, C.c5, 'X', .18);
    p.text([.55, -.35], `T × ${dModel}`, { align: 'center', size: 9.5, color: C.muted });
    for (let i = 0; i < H; i++) {
      const y = i * (rowH + .12);
      const isSel = i === showHead;
      rect(2.4, y, 1.5, rowH, isSel ? C.c2 : C.c1, `head ${i + 1}`, isSel ? .32 : .14);
      rect(4.4, y, 1.4, rowH, isSel ? C.c2 : C.c3, `d_k=${dk}`, isSel ? .3 : .12);
      const a = p.toScreen([1.15, (Math.max(h, 4) - 1.6) / 2 + .8]);
      const b = p.toScreen([2.38, y + rowH / 2]);
      p.ctx.strokeStyle = withA(isSel ? C.c2 : C.muted, isSel ? .8 : .3);
      p.ctx.lineWidth = isSel ? 2 : 1;
      p.ctx.beginPath(); p.ctx.moveTo(a[0], a[1]); p.ctx.lineTo(b[0], b[1]); p.ctx.stroke();
      const c = p.toScreen([5.85, y + rowH / 2]);
      const dd = p.toScreen([7.0, (Math.max(h, 4) - Math.min(h * rowH * 1.1, 3)) / 2 + Math.min(h * rowH * 1.1, 3) / 2]);
      p.ctx.strokeStyle = withA(isSel ? C.c2 : C.muted, isSel ? .8 : .3);
      p.ctx.beginPath(); p.ctx.moveTo(c[0], c[1]); p.ctx.lineTo(dd[0], dd[1]); p.ctx.stroke();
    }
    p.text([3.15, Math.max(h, 4) + .45], 'W^Q, W^K, W^V per head', { align: 'center', size: 10, color: C.muted });
    p.text([5.1, Math.max(h, 4) + .45], 'attention', { align: 'center', size: 10, color: C.muted });
    const catH = Math.min(h * rowH * 1.1, 3);
    rect(7.0, (Math.max(h, 4) - catH) / 2, 1.2, catH, C.c3, 'concat', .2);
    p.text([7.6, -.35], `T × ${h * dk}`, { align: 'center', size: 9.5, color: C.muted });
    rect(8.9, (Math.max(h, 4) - 1.6) / 2, 1.1, 1.6, C.c1, 'W^O', .2);
    p.text([9.45, -.35], `${h * dk} × ${dModel}`, { align: 'center', size: 9.5, color: C.muted });
    p.legend([[C.c2, `head ${showHead + 1} highlighted`]], { corner: 'tl', title: `${h} heads × d_k ${dk} = ${h * dk}` });
  });

  canvas.addEventListener('click', () => { showHead = (showHead + 1) % Math.max(h, 1); plot.render(); });
  refresh();

  node.appendChild(note(
    `A single attention operation gives every token exactly one distribution over the sentence, so it can ` +
    `express one kind of relationship. Language has several at once — agreement, coreference, semantic ` +
    `similarity — and multi-head attention runs \\(h\\) of them side by side on <em>narrower</em> projections. ` +
    `That last word is the trick: each head works in \\(d_k = d_{\\text{model}}/h\\) dimensions, so ` +
    `concatenating \\(h\\) of them returns exactly \\(d_{\\text{model}}\\) columns and the total parameter count ` +
    `is unchanged. Move the head slider and watch the per-head width shrink as the count grows while the ` +
    `totals stay put. Click the diagram to step through the heads.`
  ));
});

/* ============================================================
   The transformer block, drawn as a data path rather than a stack
   ============================================================ */
defineWidget('transformer-arch', node => {
  const { left, right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, { xmin: 0, xmax: 100, ymin: 0, ymax: 100, aspect: .82, equal: false, pad: 0 }));

  // The key lives in the DOM rather than on the canvas: it never competes with
  // the diagram for space and stays legible at any figure size.
  const keyItem = (col, label, dashed) => el('span', {
    style: 'display:inline-flex;align-items:center;gap:.42em;font-size:.78rem;color:var(--ink-muted)',
  }, el('span', {
    style: `width:18px;height:0;border-top:${dashed ? '2px dashed' : '3px solid'} ${col};`
         + 'border-radius:2px;flex:none',
  }), label);
  const keyRow = el('div', {
    style: 'display:flex;flex-wrap:wrap;gap:.5rem 1.1rem;padding:.15rem .1rem 0',
  });
  left.appendChild(keyRow);

  /* Each stage carries what it does and, crucially, whether it can see any
     position other than its own — the one structural fact worth taking away. */
  const PARTS = {
    embed: {
      label: 'Token embedding', sub: 'ids → vectors, ℝ^{|V|×d}', kind: 'io', mixes: false,
      why: 'Each token id indexes a row of a learned matrix E. This is the only place discrete symbols enter the model; everything downstream is real-valued.',
    },
    pos: {
      label: 'Positional encoding', sub: 'added, not concatenated', kind: 'io', mixes: false,
      why: 'Attention is permutation-equivariant — shuffle the rows of X and the outputs shuffle with them. Position has to be injected explicitly, and it is added straight onto the embedding so the vector carries both what the token is and where it sits.',
    },
    attn: {
      label: 'Multi-head self-attention', sub: 'h heads · scaled dot-product', kind: 'mix', mixes: true,
      why: 'The only sublayer where positions exchange information at all. Every token forms a query, matches it against every key, and takes a weighted average of the values.',
    },
    mask: {
      label: 'Causal mask', sub: 'inside attention · upper triangle → −∞', kind: 'mix', mixes: true,
      why: 'Not a layer of its own — it is applied to the score matrix inside attention, before the softmax. Setting the upper triangle to −∞ gives those positions exactly zero weight, so a token can never read the future. This single change is what makes the stack autoregressive.',
    },
    add1: {
      label: 'Add & norm', sub: 'x + Sublayer(x), then LayerNorm', kind: 'pos', mixes: false,
      why: 'The residual add is the reason deep stacks train: the gradient reaches earlier layers through the identity path even if the sublayer contributes little. LayerNorm then rescales each position independently.',
    },
    ff: {
      label: 'Feed-forward', sub: 'd → 4d → d, per position', kind: 'pos', mixes: false,
      why: 'A two-layer MLP applied to each position separately and identically — it cannot see any other token. Most of the block\'s parameters live here.',
    },
    add2: {
      label: 'Add & norm', sub: 'second residual', kind: 'pos', mixes: false,
      why: 'Every sublayer in the block is wrapped the same way, so a block is exactly two residual-wrapped operations: one that mixes positions and one that does not.',
    },
    head: {
      label: 'Linear → softmax', sub: 'd → |V|', kind: 'io', mixes: false,
      why: 'Projects each position back to vocabulary size and normalises, giving a distribution over the next token.',
    },
  };

  let sel = 'attn', N = 6, causal = true;

  const nCtl = slider('Blocks stacked, N', { min: 1, max: 12, step: 1, value: 6, format: v => String(v), onInput: v => { N = v; refresh(); } });
  const cCtl = toggle('Causal mask (decoder)', { value: true, onChange: v => { causal = v; if (!causal && sel === 'mask') sel = 'attn'; refresh(); } });
  const out = readout([['component', 0], ['sees other positions', 0], ['inside the repeated block', 0], ['blocks', 0]]);
  const st = status('');
  right.append(nCtl.root, cCtl.root, out.root, st.root);

  /* ---- layout, in diagram units; y grows downward ---- */
  const CX = 44;              // centre of the main column
  const BW = 60;              // box width
  const BH = 8.2;             // box height
  const rows = () => {
    const r = [];
    let y = 12;
    r.push({ key: 'embed', y }); y += 12.6;
    // wide gap here: the plate's top edge and its tab sit in this space
    r.push({ key: 'pos', y, plus: true }); y += 23;
    // the plate must clear the tallest box it contains, and the attention card
    // grows when the mask zone is present
    const attnH = causal ? BH * 1.72 : BH;
    const blockTop = y - attnH / 2 - 5.4;
    r.push({ key: 'attn', y, card: causal }); y += attnH / 2 + 7.6;
    r.push({ key: 'add1', y, plus: true }); y += 13.2;
    r.push({ key: 'ff', y }); y += 12.6;
    r.push({ key: 'add2', y, plus: true }); y += 8.4;
    const blockBot = y;
    y += 6.4;
    r.push({ key: 'head', y });
    return { list: r, blockTop, blockBot, bottom: y + BH / 2 + 5 };
  };

  const colFor = k => (k === 'mix' ? C.c2 : k === 'pos' ? C.c1 : C.c5);

  function refresh() {
    const P = PARTS[sel];
    const inBlock = ['attn', 'mask', 'add1', 'ff', 'add2'].includes(sel);
    out.set([
      P.label,
      { html: P.mixes ? 'yes' : 'no — acts on each position alone', cls: P.mixes ? 'is-ok' : '' },
      { html: inBlock ? `yes — runs ${N}×` : 'no — once, outside', cls: inBlock ? 'is-ok' : '' },
      String(N),
    ]);
    st.set(`${INFO}<span><strong>${P.label}.</strong> ${P.why}</span>`, 'info');
    keyRow.innerHTML = '';
    keyRow.append(
      keyItem(C.c5, 'input / output side'),
      keyItem(C.c2, 'mixes positions'),
      keyItem(C.c1, 'per position only'),
      keyItem(C.c3, 'residual path', true),
    );
    plot.render();
  }

  /* ---- drawing primitives, all in diagram units ---- */
  function roundedPath(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  plot.onDraw(p => {
    const L = rows();
    p.o.xmin = 0; p.o.xmax = 100;
    p.o.ymin = 0; p.o.ymax = Math.max(112, L.bottom + 12);
    p._computeScale();
    p.clear(null);
    const ctx = p.ctx;
    // y grows downward in diagram units
    const Y = v => p.Y(p.o.ymax - v);
    const X = v => p.X(v);
    const U = v => Math.abs(p.px(v));          // horizontal unit → px
    const V = v => Math.abs(p.py(v));          // vertical unit → px

    /* the repeated-block plate, drawn first so everything sits on top */
    const plateX = X(CX - BW / 2 - 5.5), plateW = U(BW + 11);
    const plateY = Y(L.blockTop), plateH = V(L.blockBot - L.blockTop);
    roundedPath(ctx, plateX, plateY, plateW, plateH, 12);
    ctx.fillStyle = withA(C.ink, .035);
    ctx.fill();
    ctx.strokeStyle = withA(C.ink, .22);
    ctx.lineWidth = 1.4;
    ctx.setLineDash([7, 5]);
    ctx.stroke();
    ctx.setLineDash([]);
    // "× N" tab on the plate
    const tabW = U(30), tabH = V(6.4);
    const tabX = plateX + plateW - tabW - U(2), tabY = plateY - tabH / 2;
    roundedPath(ctx, tabX, tabY, tabW, tabH, 7);
    ctx.fillStyle = C.raised; ctx.fill();
    ctx.strokeStyle = withA(C.ink, .35); ctx.lineWidth = 1.4; ctx.stroke();
    ctx.font = `700 ${Math.max(10, V(3.3))}px ${css('--font-sans')}`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillStyle = C.ink;
    ctx.fillText(`${causal ? 'decoder' : 'encoder'} block  × ${N}`, tabX + tabW / 2, tabY + tabH / 2);

    /* connector arrows down the spine, drawn between consecutive boxes */
    const list = L.list;
    for (let i = 0; i < list.length - 1; i++) {
      const a = list[i], b = list[i + 1];
      const aH = a.card ? BH * 1.72 : BH, bH = b.card ? BH * 1.72 : BH;
      const y0 = Y(a.y + aH / 2), y1 = Y(b.y - bH / 2);
      const x = X(CX);
      ctx.strokeStyle = withA(C.ink, .5);
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.moveTo(x, y0 + 1);
      ctx.lineTo(x, y1 - 7);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x, y1 - 1);
      ctx.lineTo(x - 4.6, y1 - 8);
      ctx.lineTo(x + 4.6, y1 - 8);
      ctx.closePath();
      ctx.fillStyle = withA(C.ink, .5);
      ctx.fill();
    }

    /* residual bypasses: leave the spine above a sublayer, run down the left,
       and re-enter at the add node — the path the gradient can take */
    const resid = (fromKey, toKey) => {
      const from = list.find(r => r.key === fromKey);
      const to = list.find(r => r.key === toKey);
      if (!from || !to) return;
      const yA = Y(from.y + BH / 2 + 2.2);
      const yB = Y(to.y);
      const xSpine = X(CX);
      const xOut = X(CX - BW / 2 - 9.5);
      ctx.strokeStyle = C.c3;
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(xSpine, yA);
      ctx.lineTo(xOut + U(2.4), yA);
      ctx.quadraticCurveTo(xOut, yA, xOut, yA + V(2.4));
      ctx.lineTo(xOut, yB - V(2.4));
      ctx.quadraticCurveTo(xOut, yB, xOut + U(2.4), yB);
      ctx.lineTo(X(CX - BW / 2 - 2) - 6, yB);
      ctx.stroke();
      ctx.setLineDash([]);
      const xh = X(CX - BW / 2 - 2);
      ctx.beginPath();
      ctx.moveTo(xh, yB);
      ctx.lineTo(xh - 7, yB - 4.4);
      ctx.lineTo(xh - 7, yB + 4.4);
      ctx.closePath();
      ctx.fillStyle = C.c3; ctx.fill();
      // label it once, on the upper arc
      if (fromKey === 'pos') {
        ctx.save();
        ctx.font = `650 ${Math.max(9, V(2.8))}px ${css('--font-sans')}`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
        ctx.translate(xOut - 4, (yA + yB) / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillStyle = C.c3;
        ctx.fillText('residual', 0, 0);
        ctx.restore();
      }
    };
    resid('pos', 'add1');
    resid('add1', 'add2');

    /* the boxes themselves */
    for (const r of list) {
      const P = PARTS[r.key];
      const isSel = r.key === sel || (r.card && sel === 'mask');
      const col = colFor(P.kind);
      const w = U(BW), h = V(r.card ? BH * 1.72 : BH);
      const x = X(CX) - w / 2, y = Y(r.y) - h / 2;

      if (isSel) {                                   // selection halo
        roundedPath(ctx, x - 4, y - 4, w + 8, h + 8, 12);
        ctx.fillStyle = withA(col, .16); ctx.fill();
      }
      roundedPath(ctx, x, y, w, h, 9);
      ctx.fillStyle = withA(col, isSel ? .28 : .12);
      ctx.fill();
      ctx.strokeStyle = col;
      ctx.lineWidth = isSel ? 2.6 : 1.5;
      ctx.stroke();

      // a thicker leading edge marks the sublayers that mix positions
      if (P.mixes) {
        ctx.save();
        roundedPath(ctx, x, y, w, h, 9);
        ctx.clip();
        ctx.fillStyle = col;
        ctx.fillRect(x, y, Math.max(3, U(1.1)), h);
        ctx.restore();
      }

      const tSize = Math.max(10.5, V(3.5));
      const sSize = Math.max(8.5, V(2.7));
      // a zone is one labelled band of a box; a plain box has exactly one
      const zone = (key, top, height, strong) => {
        const Z = PARTS[key];
        const cy = top + height / 2;
        ctx.font = `${strong ? 700 : 640} ${tSize}px ${css('--font-sans')}`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
        ctx.fillStyle = C.ink;
        ctx.fillText(Z.label, x + w / 2, cy - (Z.sub ? V(.4) : -tSize * .34));
        if (Z.sub) {
          ctx.font = `500 ${sSize}px ${css('--font-sans')}`;
          ctx.fillStyle = C.muted;
          ctx.textBaseline = 'top';
          ctx.fillText(Z.sub, x + w / 2, cy + V(1.0));
        }
      };
      if (r.card) {
        // the mask is drawn inside attention, because that is where it happens
        const hTop = h * .56;
        zone('attn', y, hTop, sel === 'attn');
        ctx.strokeStyle = withA(col, .45);
        ctx.lineWidth = 1.2;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(x + U(3), y + hTop); ctx.lineTo(x + w - U(3), y + hTop);
        ctx.stroke(); ctx.setLineDash([]);
        zone('mask', y + hTop, h - hTop, sel === 'mask');
      } else {
        zone(r.key, y, h, isSel);
      }

      // ⊕ node on the left edge for the two adds and the positional sum
      if (r.plus) {
        const cx = x - U(2), cy = y + h / 2, rr = Math.max(7, V(2.5));
        ctx.beginPath(); ctx.arc(cx, cy, rr, 0, Math.PI * 2);
        ctx.fillStyle = C.bg; ctx.fill();
        ctx.strokeStyle = r.key === 'pos' ? C.c5 : C.c3;
        ctx.lineWidth = 2; ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(cx - rr * .5, cy); ctx.lineTo(cx + rr * .5, cy);
        ctx.moveTo(cx, cy - rr * .5); ctx.lineTo(cx, cy + rr * .5);
        ctx.stroke();
      }

    }

    /* what enters and what leaves */
    const cap = (yv, text) => {
      ctx.font = `600 ${Math.max(9.5, V(3))}px ${css('--font-mono')}`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillStyle = C.muted;
      ctx.fillText(text, X(CX), Y(yv));
    };
    cap(list[0].y - BH / 2 - 4.6, 'token ids   T × 1');
    cap(list[list.length - 1].y + BH / 2 + 5, 'next-token distribution   T × |V|');

  });

  /* click a box to read what it does */
  canvas.addEventListener('click', e => {
    const r = canvas.getBoundingClientRect();
    const py = (e.clientY - r.top) / r.height;
    const L = rows();
    const yv = py * plot.o.ymax;
    let best = null, bd = Infinity, bestRow = null;
    for (const row of L.list) {
      const half = (row.card ? BH * 1.72 : BH) / 2;
      const d = Math.abs(row.y - yv);
      if (d < bd && d < half + 1.5) { bd = d; best = row.key; bestRow = row; }
    }
    if (!best) return;
    // the attention card holds two selectable zones
    if (bestRow.card && yv > bestRow.y - BH * .86 + BH * 1.72 * .56) best = 'mask';
    sel = best;
    refresh();
  });
  canvas.style.cursor = 'pointer';

  refresh();

  node.appendChild(note(
    `Click any stage to read what it does. The structural point the diagram is built around: of ` +
    `everything in the stack, <strong>only attention moves information between positions</strong> — ` +
    `those boxes carry a solid leading edge. The feed-forward layer, both normalisations and the output ` +
    `projection all act on each position separately and identically; shuffle the tokens and they would ` +
    `not notice. That is why positional encoding has to be <em>added</em> at the very bottom, and why ` +
    `the causal mask alone is enough to make the whole stack autoregressive — it is not a layer, it is a ` +
    `change to the score matrix inside attention. The dashed path is the residual: it is what lets the ` +
    `gradient reach the bottom of a deep stack, and it is the same trick as ` +
    `<a href="#architectures">ResNet's skip connection</a>, laid along depth.`
  ));
});
