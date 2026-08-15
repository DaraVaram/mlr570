/* ============================================================
   widgets/deep.js — CNNs, RNNs and Transformers
   ============================================================ */
import {
  Plot, Dragger, C, el, slider, toggle, segmented, button,
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

  const plot = trackPlot(new Plot(cv, { xmin: 0, xmax: 34, ymin: 0, ymax: 20, aspect: 1.7, equal: false, pad: 0 }));

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
        p.ctx.fillRect(X, Y, p.px(cell) + 1, p.px(cell) + 1);
      }
    }
    // kernel window
    const oi = Math.floor(pos / H2), oj = pos % H2;
    const kx = ox + oj * stride * cell, ky = oy - oi * stride * cell;
    p.ctx.strokeStyle = C.c4; p.ctx.lineWidth = 2.6;
    p.ctx.strokeRect(p.X(kx), p.Y(ky), p.px(F * cell), p.px(F * cell));
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
        p.ctx.strokeRect(X, Y, p.px(ocell), p.px(ocell));
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
    xmin: 0, xmax: 12, ymin: 0, ymax: 8, aspect: 1.7, equal: false, pad: 0,
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
    xmin: 0, xmax: 22, ymin: 0, ymax: 12, aspect: 1.8, equal: false, pad: 0,
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
        p.ctx.fillRect(X + 1, Y + 1, p.px(cs) - 2, p.px(cs) - 2);
        p.text([ox + j * cs + cs / 2, oy - i * cs - cs / 2], fmt(M[i][j], mode === 'avg' && n < N ? 2 : 0),
          { align: 'center', size: 10.5, weight: 600, color: t > .55 ? C.raised : C.ink, mono: true });
      }
      if (hlWin) {
        p.ctx.strokeStyle = C.c4; p.ctx.lineWidth = 2;
        for (let i = 0; i < n; i += 2) for (let j = 0; j < n; j += 2) {
          p.ctx.strokeRect(p.X(ox + j * cs), p.Y(oy - i * cs), p.px(2 * cs), p.px(2 * cs));
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
    xmin: 0, xmax: 40, ymin: 0, ymax: 14, aspect: 1.7, equal: false, pad: 0,
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
        p.ctx.fillRect(p.X(x) + .8, p.Y(y), Math.max(1, p.px(px) - 1.6), p.px(rowH * .42));
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

  const plot = trackPlot(new Plot(cv, { xmin: 0, xmax: 20, ymin: 0, ymax: 11, aspect: 1.7, equal: false, pad: 0 }));

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
      p.ctx.beginPath(); p.ctx.roundRect(p.X(x - .55), p.Y(2.4), p.px(1.1), p.px(1.0), 6);
      p.ctx.fill(); p.ctx.stroke();
      p.text([x, 1.9], T.ch, { align: 'center', size: 14, weight: 750, color: done ? C.c3 : C.muted });
      p.text([x, 1.15], `x${'₁₂₃₄'[i]}`, { align: 'center', size: 9.5, color: C.muted });

      // output above
      p.arrow([x, 6.4], [x, 8.3], { color: done ? C.c2 : C.grid, lw: done ? 2.2 : 1.3, head: 8 });
      const best = T.p.indexOf(Math.max(...T.p));
      p.ctx.fillStyle = done ? withA(C.c2, .25) : C.raised;
      p.ctx.strokeStyle = done ? C.c2 : C.grid; p.ctx.lineWidth = 1.8;
      p.ctx.beginPath(); p.ctx.roundRect(p.X(x - .55), p.Y(9.5), p.px(1.1), p.px(1.0), 6);
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
    xmin: 0, xmax: 21, ymin: -9, ymax: 0.6, aspect: 1.6, equal: false, pad: 0,
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
      { corner: van > 1 ? 'bl' : 'tl', title: 'log₁₀ gradient, relative to the final step' });
    p.text({ px: p.w / 2, py: p.h - 5 }, 'time step t', { align: 'center', size: 10.5, color: C.muted });
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
    xmin: 0, xmax: 21, ymin: -1.35, ymax: 1.35, aspect: 1.7, equal: false, pad: 0,
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

  const plot = trackPlot(new Plot(cvHeat, { xmin: 0, xmax: 1, ymin: 0, ymax: 1, aspect: 1.2, equal: false, pad: 0 }));
  const plot2 = trackPlot(new Plot(cvSim, { xmin: 0, xmax: 1, ymin: -1.1, ymax: 1.1, aspect: 1.2, equal: false, pad: 0 }));

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
    p.o.xmin = 0; p.o.xmax = dModel; p.o.ymin = 0; p.o.ymax = T;
    p._computeScale();
    const cw = p.px(1), ch = p.px(1);
    for (let pos = 0; pos < T; pos++) for (let i = 0; i < dModel; i++) {
      const v = PE(pos, i);
      p.ctx.fillStyle = v >= 0 ? withA(C.c1, .1 + Math.abs(v) * .8) : withA(C.c4, .1 + Math.abs(v) * .8);
      p.ctx.fillRect(p.X(i), p.Y(pos + 1), cw + 1, ch + 1);
    }
    p.ctx.strokeStyle = C.c2; p.ctx.lineWidth = 2.4;
    p.ctx.strokeRect(p.X(0), p.Y(posA + 1), p.px(dModel), ch);
    p.text({ px: 8, py: 12 }, `p = ${posA}`, { color: C.c2, size: 11, weight: 700 });
    p.text({ px: p.w / 2, py: p.h - 5 }, 'channel i →', { align: 'center', size: 10, color: C.muted });
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
      p.badge([posA, 1.05], `p = ${posA}`, { color: C.c2, align: 'center' });
    }
    p.axes(); p.ticks(Math.max(5, Math.round(T / 8)));
    p.text({ px: 8, py: 12 }, 'cosine similarity between PE(p) and PE(q)', { color: C.muted, size: 10 });
    p.text({ px: p.w / 2, py: p.h - 5 }, 'position q', { align: 'center', size: 10, color: C.muted });
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

  const plot = trackPlot(new Plot(cv, { xmin: 0, xmax: 14, ymin: 0, ymax: 10, aspect: 1.45, equal: false, pad: 0 }));

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
      p.ctx.fillRect(X + 1, Y + 1, p.px(cell) - 2, p.px(cell) - 2);
      if (i === query) {
        p.ctx.strokeStyle = C.c2; p.ctx.lineWidth = 2;
        p.ctx.strokeRect(X + 1, Y + 1, p.px(cell) - 2, p.px(cell) - 2);
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
      p.ctx.fillRect(p.X(x + .12), p.Y(by + v * 1.15), p.px(cell - .24), p.px(v * 1.15));
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
    xmin: 0, xmax: 1, ymin: 0, ymax: 1, aspect: 1.6, equal: false, pad: 0,
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
    p.o.xmin = -.6; p.o.xmax = T; p.o.ymin = 0; p.o.ymax = 1.12;
    p._computeScale();
    p.grid(.25, { color: C.grid });
    const bw = p.px(.72);
    weights.forEach((w, i) => {
      const x = p.X(i + .14);
      p.ctx.fillStyle = C.c1; p.ctx.globalAlpha = .88;
      p.ctx.fillRect(x, p.Y(w), bw, p.Y(0) - p.Y(w));
      p.ctx.globalAlpha = 1;
      if (w > .03) p.text([i + .5, w], fmt(w, 2), { align: 'center', dy: -9, size: 9.5, color: C.muted });
    });
    p.line([-.6, 1 / T], [T, 1 / T], { color: C.c3, lw: 1.6, dash: [5, 4] });
    p.badge([T - .2, 1 / T], `uniform = ${fmt(1 / T, 3)}`, { color: C.c3, align: 'right', dy: -12 });
    p.axes(); p.ticks(2);
    p.text({ px: 12, py: 16 }, 'softmax weights over 12 keys', { color: C.muted, size: 10.5 });
    p.text({ px: p.w / 2, py: p.h - 5 }, 'key index', { align: 'center', size: 10.5, color: C.muted });
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
