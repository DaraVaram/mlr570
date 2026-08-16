/* ============================================================
   widgets/nets.js — Decision trees and feedforward neural networks
   ============================================================ */
import {
  Plot, Dragger, C, css, el, slider, toggle, segmented, button,
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
const OK = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>`;
const INFO = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16.5v.01" stroke-linecap="round"/></svg>`;
const WARN = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 9v4M12 17v.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/></svg>`;

/** Module-level alpha helper (a widget below defines its own local copy too). */
function withA(hex, a) {
  hex = (hex || '').trim();
  if (!hex.startsWith('#')) return hex;
  const n = hex.length === 4
    ? hex.slice(1).split('').map(c => parseInt(c + c, 16))
    : [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));
  return `rgba(${n.join(',')},${a})`;
}

/* ---------- the Play Tennis dataset, exactly as printed ---------- */
const FEATURES = ['Outlook', 'Temp', 'Humidity', 'Wind'];
const TENNIS = [
  [1, 'Sunny', 'Hot', 'High', 'Weak', 'No'],
  [2, 'Sunny', 'Hot', 'High', 'Strong', 'No'],
  [3, 'Overcast', 'Hot', 'High', 'Weak', 'Yes'],
  [4, 'Rain', 'Mild', 'High', 'Weak', 'Yes'],
  [5, 'Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
  [6, 'Rain', 'Cool', 'Normal', 'Strong', 'No'],
  [7, 'Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
  [8, 'Sunny', 'Mild', 'High', 'Weak', 'No'],
  [9, 'Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
  [10, 'Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
  [11, 'Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
  [12, 'Overcast', 'Mild', 'High', 'Strong', 'Yes'],
  [13, 'Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
  [14, 'Rain', 'Mild', 'High', 'Strong', 'No'],
];

const entropy = labels => {
  const n = labels.length;
  if (!n) return 0;
  const c = {};
  labels.forEach(l => { c[l] = (c[l] || 0) + 1; });
  return -Object.values(c).reduce((s, v) => s + (v / n) * Math.log2(v / n), 0);
};
const giniOf = labels => {
  const n = labels.length;
  if (!n) return 0;
  const c = {};
  labels.forEach(l => { c[l] = (c[l] || 0) + 1; });
  return 1 - Object.values(c).reduce((s, v) => s + (v / n) ** 2, 0);
};
const labelsOf = rows => rows.map(r => r[5]);
const countYN = rows => {
  const y = rows.filter(r => r[5] === 'Yes').length;
  return { yes: y, no: rows.length - y };
};

function splitOn(rows, feat) {
  const idx = FEATURES.indexOf(feat) + 1;
  const groups = {};
  rows.forEach(r => { (groups[r[idx]] ||= []).push(r); });
  return groups;
}

function infoGain(rows, feat, measure = 'entropy') {
  const imp = measure === 'gini' ? giniOf : entropy;
  const before = imp(labelsOf(rows));
  const groups = splitOn(rows, feat);
  let after = 0;
  Object.values(groups).forEach(g => { after += (g.length / rows.length) * imp(labelsOf(g)); });
  return { before, after, gain: before - after, groups };
}

/* ============================================================
   1. Entropy and Gini side by side
   ============================================================ */
defineWidget('entropy-gini', node => {
  const { right, canvas } = split(node, { hint: 'Drag the marker' });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -.05, xmax: 1.05, ymin: -.06, ymax: 1.12, aspect: 1.4, equal: false, pad: 0,
  }));

  let p = 0.3, showScaled = false;
  const pCtl = slider('Proportion p of class 1', {
    min: 0, max: 1, step: .005, value: .3, format: v => fmt(v, 3),
    onInput: v => { p = v; refresh(); },
  });
  const scaleCtl = toggle('Overlay ½·H(p) to compare shapes', {
    value: false, onChange: v => { showScaled = v; plot.render(); },
  });
  const out = readout([['p', 0], ['entropy H(p)', 0], ['Gini 2p(1−p)', 0], ['H(p) / Gini(p)', 0], ['misclassification', 0]]);
  const st = status('');
  right.append(pCtl.root, scaleCtl.root, out.root, st.root);

  const H = q => (q <= 0 || q >= 1) ? 0 : -q * Math.log2(q) - (1 - q) * Math.log2(1 - q);
  const G = q => 2 * q * (1 - q);
  const M = q => Math.min(q, 1 - q);

  const drag = new Dragger(plot);
  drag.add(() => [p, H(p)], q => { p = clamp(round(q[0], 3), 0, 1); pCtl.set(p); });
  drag.onchange = refresh;

  function refresh() { plot.render(); sync(); }
  function sync() {
    const h = H(p), g = G(p);
    out.set([
      fmt(p, 3), fmt(h, 4), fmt(g, 4),
      g > 1e-6 ? fmt(h / g, 4) : '—',
      fmt(M(p), 4),
    ]);
    st.set(
      Math.abs(p - .5) < .01
        ? `${INFO}<span>At p = 0.5 both peak: <strong>H = 1</strong> and <strong>Gini = 0.5</strong>. So H = 2·Gini here — the entropy is the <em>larger</em> of the two, not the smaller.</span>`
        : p < .02 || p > .98
          ? `${OK}<span>Near-pure node: both measures approach 0. Either one will refuse to split further.</span>`
          : `${INFO}<span>H/Gini = <strong>${fmt(h / g, 3)}</strong>. The ratio drifts, but entropy exceeds Gini at every interior p — never the reverse.</span>`,
      p < .02 || p > .98 ? 'ok' : 'info');
  }

  plot.onDraw(pl => {
    pl.grid(.25, { color: C.grid });
    pl.fn(H, { color: C.c1, lw: 3, from: 0, to: 1, samples: 400 });
    pl.fn(G, { color: C.c2, lw: 3, from: 0, to: 1, samples: 400 });
    pl.fn(M, { color: C.c3, lw: 2, dash: [5, 4], from: 0, to: 1, samples: 200 });
    if (showScaled) pl.fn(q => H(q) / 2, { color: C.c1, lw: 2, dash: [4, 4], alpha: .8, from: 0, to: 1, samples: 400 });

    pl.line([p, 0], [p, Math.max(H(p), G(p))], { color: C.c4, lw: 1.6, dash: [4, 4] });
    pl.dot([p, H(p)], { r: 6, color: C.c1, ring: true });
    pl.dot([p, G(p)], { r: 6, color: C.c2, ring: true });
    pl.axes(); pl.ticks(.25);

    const key = [[C.c1, 'entropy H(p)'], [C.c2, 'Gini 2p(1−p)'], [C.c3, 'misclassification']];
    if (showScaled) key.push([C.c1, '½·H(p)  (dashed)']);
    key.forEach(([col, lbl], i) => {
      pl.ctx.strokeStyle = col; pl.ctx.lineWidth = 2.6;
      pl.ctx.beginPath(); pl.ctx.moveTo(14, 16 + i * 15); pl.ctx.lineTo(32, 16 + i * 15); pl.ctx.stroke();
      pl.text({ px: 37, py: 16 + i * 15 }, lbl, { color: C.muted, size: 10.5, weight: 600 });
    });
  });
  refresh();

  node.appendChild(note(
    `Both measures peak at p = 0.5 and vanish at the pure ends, which is why they usually pick the same split. ` +
    `But they are not the same size: <strong>H(0.5) = 1 while Gini(0.5) = 0.5</strong>, so entropy is roughly ` +
    `<em>twice</em> Gini, not half of it. Switch on the ½·H(p) overlay and it lands almost exactly on the Gini ` +
    `curve — that is the correct statement of the relationship. Gini is cheaper because it needs no logarithm, ` +
    `which is why CART and random forests default to it.`
  ));
});

/* ============================================================
   2. Information gain on the Play Tennis dataset
   ============================================================ */
defineWidget('info-gain', node => {
  const wrap = el('div');
  node.appendChild(wrap);

  let branch = 'all', feat = 'Outlook', measure = 'entropy';

  const branchCtl = segmented([
    { label: 'All 14 days', value: 'all' },
    { label: 'Outlook = Sunny', value: 'Sunny' },
    { label: 'Outlook = Overcast', value: 'Overcast' },
    { label: 'Outlook = Rain', value: 'Rain' },
  ], { value: 'all', label: 'Node', onChange: v => { branch = v; refresh(); } });
  const featCtl = segmented(
    FEATURES.map(f => ({ label: f, value: f })),
    { value: 'Outlook', label: 'Split on', onChange: v => { feat = v; refresh(); } });
  const measCtl = segmented([
    { label: 'Entropy', value: 'entropy' }, { label: 'Gini', value: 'gini' },
  ], { value: 'entropy', label: 'Impurity', onChange: v => { measure = v; refresh(); } });

  const cv = el('canvas');
  const left = el('div', {}, el('div', { class: 'pg-canvas-wrap' }, cv));
  const right = el('div', { class: 'pg-controls' }, branchCtl.root, featCtl.root, measCtl.root);
  const out = readout([['node', 0], ['impurity before', 0], ['weighted after', 0], ['gain', 0], ['best feature here', 0]]);
  const st = status('');
  right.append(out.root, st.root);
  wrap.appendChild(el('div', { class: 'pg-split pg-split--wide-ctrl' }, left, right));

  const work = el('div', { class: 'readout', style: 'margin-top:1rem' });
  wrap.appendChild(work);

  const plot = trackPlot(new Plot(cv, { xmin: 0, xmax: 10, ymin: 0, ymax: 7, aspect: 1.5, equal: false, pad: 0 }));

  const rowsFor = () => branch === 'all' ? TENNIS : TENNIS.filter(r => r[1] === branch);

  let res = null;
  function refresh() {
    const rows = rowsFor();
    // Outlook is already used on a branch node
    const avail = branch === 'all' ? FEATURES : FEATURES.filter(f => f !== 'Outlook');
    if (!avail.includes(feat)) { feat = avail[0]; featCtl.set(feat); }
    res = infoGain(rows, feat, measure);

    const all = avail.map(f => ({ f, g: infoGain(rows, f, measure).gain }));
    const best = all.reduce((a, b) => (b.g > a.g ? b : a), all[0]);
    const { yes, no } = countYN(rows);

    out.set([
      `${branch === 'all' ? 'root' : 'Outlook = ' + branch} · ${yes}Y ${no}N (n=${rows.length})`,
      fmt(res.before, 4),
      fmt(res.after, 4),
      { html: fmt(res.gain, 4), cls: Math.abs(res.gain - best.g) < 1e-9 ? 'is-ok' : '' },
      `${best.f} (${fmt(best.g, 4)})`,
    ]);

    const rowsHtml = Object.entries(res.groups).sort().map(([k, g]) => {
      const c = countYN(g);
      const imp = measure === 'gini' ? giniOf(labelsOf(g)) : entropy(labelsOf(g));
      return `<div>${k.padEnd(9)} ${String(c.yes).padStart(2)}Y ${String(c.no).padStart(2)}N ` +
             `&nbsp; n=${g.length} &nbsp; I=${fmt(imp, 4)} &nbsp; ` +
             `<span style="color:var(--ink-faint)">weight ${g.length}/${rows.length}</span></div>`;
    }).join('');
    work.innerHTML =
      `<div style="color:var(--ink-faint);margin-bottom:.35em">Splitting <strong>${feat}</strong> at this node:</div>` +
      rowsHtml +
      `<div style="margin-top:.5em">I_after = ${Object.entries(res.groups).sort().map(([k, g]) => {
        const imp = measure === 'gini' ? giniOf(labelsOf(g)) : entropy(labelsOf(g));
        return `(${g.length}/${rows.length})(${fmt(imp, 3)})`;
      }).join(' + ')} = <em>${fmt(res.after, 4)}</em></div>` +
      `<div>Gain = ${fmt(res.before, 4)} − ${fmt(res.after, 4)} = <em>${fmt(res.gain, 4)}</em></div>`;

    st.set(
      res.gain < 1e-9
        ? `${WARN}<span>Zero gain — this feature tells you nothing at this node.</span>`
        : Math.abs(res.gain - best.g) < 1e-9
          ? `${OK}<span><strong>${feat}</strong> is the best split available here, with gain ${fmt(res.gain, 4)}.</span>`
          : `${INFO}<span><strong>${best.f}</strong> would be better (${fmt(best.g, 4)} vs ${fmt(res.gain, 4)}). Try it.</span>`,
      res.gain < 1e-9 ? 'warn' : Math.abs(res.gain - best.g) < 1e-9 ? 'ok' : 'info');
    plot.render();
  }

  plot.onDraw(pl => {
    const rows = rowsFor();
    const groups = Object.entries(res.groups).sort();
    // parent bar
    const drawBar = (x, y, w, h, g, label) => {
      const c = countYN(g);
      const total = g.length || 1;
      const wy = w * c.yes / total;
      pl.ctx.fillStyle = C.c3; pl.ctx.globalAlpha = .85;
      pl.ctx.fillRect(pl.X(x), pl.Y(y + h), pl.px(wy), pl.py(h));
      pl.ctx.fillStyle = C.c4;
      pl.ctx.fillRect(pl.X(x + wy), pl.Y(y + h), pl.px(w - wy), pl.py(h));
      pl.ctx.globalAlpha = 1;
      pl.text([x + w / 2, y + h / 2], `${c.yes}Y / ${c.no}N`,
        { align: 'center', size: 11.5, weight: 700, color: C.raised });
      pl.text([x + w / 2, y - .3], label, { align: 'center', size: 11, color: C.ink, weight: 620 });
    };
    drawBar(2.5, 5.4, 5, .9, rows, `${branch === 'all' ? 'root' : 'Outlook = ' + branch}  ·  I = ${fmt(res.before, 3)}`);

    const n = groups.length;
    const gap = .35;
    const totalW = 9.4;
    let x = .3;
    groups.forEach(([k, g]) => {
      const w = (totalW - gap * (n - 1)) * (g.length / rows.length);
      pl.line([5, 5.3], [x + w / 2, 3.6], { color: C.muted, lw: 1.4, alpha: .6 });
      const imp = measure === 'gini' ? giniOf(labelsOf(g)) : entropy(labelsOf(g));
      drawBar(x, 2.6, w, .9, g, `${k}`);
      pl.text([x + w / 2, 1.95], `I = ${fmt(imp, 3)}`, { align: 'center', size: 10.5, color: C.muted });
      x += w + gap;
    });
    pl.text([5, 1.3], `${feat}:  gain = ${fmt(res.before, 3)} − ${fmt(res.after, 3)} = ${fmt(res.gain, 4)}`,
      { align: 'center', size: 13, weight: 700, color: C.ink });
    pl.text([5, .6], 'teal = Yes · red = No · bar width ∝ number of samples',
      { align: 'center', size: 10.5, color: C.muted });
  });

  refresh();

  node.appendChild(note(
    `Every number here is computed from the 14-row table, so you can check any of it by hand. ` +
    `At the root <strong>Outlook</strong> wins with gain 0.2468 — it is the only feature producing a pure ` +
    `child (Overcast is 4 Yes, 0 No). Then move to the Sunny branch: <strong>Humidity</strong> splits it ` +
    `perfectly for a gain of 0.971, while Wind manages only 0.020. On the Rain branch it is ` +
    `<strong>Wind</strong> that splits perfectly, and Humidity that scores 0.020. Switching to Gini ` +
    `usually leaves the ranking unchanged — the criteria rarely disagree about which feature to pick.`
  ));
});

/* ============================================================
   3. Greedy tree construction, step by step
   ============================================================ */
defineWidget('tree-builder', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: 0, xmax: 12, ymin: 0, ymax: 8, aspect: 1.6, equal: false, pad: 0,
  }));

  let depthLimit = 0, measure = 'entropy';

  const depthCtl = slider('Grow to depth', {
    min: 0, max: 3, step: 1, value: 0, format: v => String(v),
    onInput: v => { depthLimit = v; refresh(); },
  });
  const measCtl = segmented([
    { label: 'Entropy', value: 'entropy' }, { label: 'Gini', value: 'gini' },
  ], { value: 'entropy', label: 'Criterion', onChange: v => { measure = v; refresh(); } });
  const out = readout([['leaves', 0], ['internal nodes', 0], ['training errors', 0], ['depth', 0]]);
  const st = status('');
  right.append(depthCtl.root, measCtl.root, out.root, st.root);

  function build(rows, used, depth) {
    const { yes, no } = countYN(rows);
    const majority = yes >= no ? 'Yes' : 'No';
    if (!rows.length) return { leaf: true, label: '?', rows, majority: '?' };
    if (yes === 0 || no === 0) return { leaf: true, label: majority, rows, pure: true };
    const avail = FEATURES.filter(f => !used.includes(f));
    if (!avail.length || depth >= depthLimit) {
      return { leaf: true, label: majority, rows, forced: depth >= depthLimit && avail.length > 0 };
    }
    let best = null;
    avail.forEach(f => {
      const r = infoGain(rows, f, measure);
      if (!best || r.gain > best.gain) best = { ...r, feat: f };
    });
    if (best.gain < 1e-9) return { leaf: true, label: majority, rows };
    const children = {};
    Object.entries(best.groups).sort().forEach(([k, g]) => {
      children[k] = build(g, [...used, best.feat], depth + 1);
    });
    return { leaf: false, feat: best.feat, gain: best.gain, children, rows };
  }

  let tree = null;
  function refresh() {
    tree = build(TENNIS, [], 0);
    let leaves = 0, internal = 0, errs = 0, maxD = 0;
    const walk = (n, d) => {
      maxD = Math.max(maxD, d);
      if (n.leaf) {
        leaves++;
        errs += n.rows.filter(r => r[5] !== n.label).length;
      } else {
        internal++;
        Object.values(n.children).forEach(c => walk(c, d + 1));
      }
    };
    walk(tree, 0);
    out.set([String(leaves), String(internal), `${errs} / 14`, String(maxD)]);
    st.set(
      depthLimit === 0
        ? `${WARN}<span>Depth 0 is a single leaf predicting the majority class — <strong>5 of 14 wrong</strong>. This is maximal under-fitting.</span>`
        : errs === 0
          ? `${OK}<span><strong>Zero training errors</strong> with ${leaves} leaves. Every leaf is pure, so the tree has perfectly separated the training set.</span>`
          : `${INFO}<span>${errs} training errors remain. Increase the depth to keep splitting.</span>`,
      depthLimit === 0 ? 'warn' : errs === 0 ? 'ok' : 'info');
    plot.render();
  }

  plot.onDraw(pl => {
    // lay out the tree by counting leaves for horizontal space
    const leafCount = n => n.leaf ? 1 : Object.values(n.children).reduce((s, c) => s + leafCount(c), 0);
    const draw = (n, x0, x1, y, parent) => {
      const cx = (x0 + x1) / 2;
      if (parent) pl.line(parent, [cx, y + .42], { color: C.muted, lw: 1.4, alpha: .65 });
      if (n.leaf) {
        const col = n.label === 'Yes' ? C.c3 : n.label === 'No' ? C.c4 : C.muted;
        pl.ctx.fillStyle = col; pl.ctx.globalAlpha = .9;
        pl.ctx.fillRect(pl.X(cx - .42), pl.Y(y + .34), pl.px(.84), pl.py(.68));
        pl.ctx.globalAlpha = 1;
        pl.text([cx, y], n.label, { align: 'center', size: 12, weight: 750, color: C.raised });
        const c = countYN(n.rows);
        pl.text([cx, y - .55], `${c.yes}/${c.no}`, { align: 'center', size: 9.5, color: C.muted, mono: true });
      } else {
        pl.ctx.fillStyle = C.raised;
        pl.ctx.strokeStyle = C.c1; pl.ctx.lineWidth = 2;
        const w = .58 + n.feat.length * .13;
        pl.ctx.beginPath();
        pl.ctx.roundRect(pl.X(cx - w / 2), pl.Y(y + .34), pl.px(w), pl.py(.68), 6);
        pl.ctx.fill(); pl.ctx.stroke();
        pl.text([cx, y], n.feat, { align: 'center', size: 11.5, weight: 700, color: C.c1 });
        const kids = Object.entries(n.children);
        const total = kids.reduce((s, [, c]) => s + leafCount(c), 0);
        let cur = x0;
        kids.forEach(([k, c]) => {
          const wdt = (x1 - x0) * leafCount(c) / total;
          const kcx = cur + wdt / 2;
          pl.text([(cx + kcx) / 2, y - .9], k, { align: 'center', size: 9.5, color: C.muted });
          draw(c, cur, cur + wdt, y - 2.1, [cx, y - .38]);
          cur += wdt;
        });
      }
    };
    draw(tree, .4, 11.6, 7, null);
  });

  refresh();

  node.appendChild(note(
    `ID3 is <strong>greedy</strong>: at each node it takes the single best split available right now, with no ` +
    `lookahead. Grow it one level at a time. Depth 1 already isolates Overcast as a pure leaf; by depth 2 the ` +
    `tree is exact on all 14 rows. Note that it never uses <strong>Temperature</strong> — once Outlook, ` +
    `Humidity and Wind are in place there is nothing left for it to explain, even though Temperature had a ` +
    `positive gain at the root.`
  ));
});

/* ============================================================
   4. Axis-aligned boundaries and overfitting
   ============================================================ */
defineWidget('tree-boundary', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -2.6, xmax: 2.6, ymin: -2.6, ymax: 2.6, aspect: 1.1, pad: 0,
  }));

  let maxDepth = 3, minLeaf = 1, noise = .12, dsKey = 'spiralish';

  const DS = {
    spiralish: 'Two moons',
    circle: 'Rings',
    diagonal: 'Diagonal',
  };
  let train = [], test = [];

  const dsCtl = segmented(Object.entries(DS).map(([k, v]) => ({ label: v, value: k })),
    { value: 'spiralish', label: 'Dataset', onChange: v => { dsKey = v; rebuild(); } });
  const dCtl = slider('Max depth', {
    min: 1, max: 12, step: 1, value: 3, format: v => String(v),
    onInput: v => { maxDepth = v; refresh(); },
  });
  const mCtl = slider('Min samples per leaf', {
    min: 1, max: 20, step: 1, value: 1, format: v => String(v),
    onInput: v => { minLeaf = v; refresh(); },
  });
  const nCtl = slider('Label noise', {
    min: 0, max: .3, step: .01, value: .12, onInput: v => { noise = v; rebuild(); },
  });
  const out = readout([['leaves', 0], ['training accuracy', 0], ['test accuracy', 0], ['gap', 0]]);
  const st = status('');
  right.append(dsCtl.root, dCtl.root, mCtl.root, nCtl.root, out.root, st.root);

  function gen(seed) {
    const r = ML.rng(seed);
    const pts = [];
    for (let i = 0; i < 220; i++) {
      let x, y, lab;
      if (dsKey === 'circle') {
        const inner = i % 2 === 0;
        const rad = inner ? .5 + Math.abs(ML.gauss(r)) * .28 : 1.5 + Math.abs(ML.gauss(r)) * .3;
        const t = r() * Math.PI * 2;
        x = Math.cos(t) * rad; y = Math.sin(t) * rad; lab = inner ? 1 : -1;
      } else if (dsKey === 'diagonal') {
        x = (r() * 2 - 1) * 2.1; y = (r() * 2 - 1) * 2.1;
        lab = (y > x * .85) ? 1 : -1;
      } else {
        const up = i % 2 === 0;
        const t = r() * Math.PI;
        if (up) { x = Math.cos(t) * 1.35 - .5; y = Math.sin(t) * 1.0 - .3; }
        else { x = -Math.cos(t) * 1.35 + .5; y = -Math.sin(t) * 1.0 + .3; }
        x += ML.gauss(r) * .22; y += ML.gauss(r) * .22;
        lab = up ? 1 : -1;
      }
      if (r() < noise) lab = -lab;
      pts.push([x, y, lab]);
    }
    return pts;
  }

  function buildTree(rows, depth) {
    const lab = rows.map(p => p[2]);
    const pos = lab.filter(l => l > 0).length;
    const majority = pos * 2 >= rows.length ? 1 : -1;
    if (depth >= maxDepth || rows.length < 2 * minLeaf || pos === 0 || pos === rows.length) {
      return { leaf: true, label: majority, n: rows.length };
    }
    let best = null;
    const imp = ls => {
      const n = ls.length; if (!n) return 0;
      const p = ls.filter(l => l > 0).length / n;
      return 2 * p * (1 - p);
    };
    const parentImp = imp(lab);
    for (const dim of [0, 1]) {
      const sorted = [...rows].sort((a, b) => a[dim] - b[dim]);
      for (let i = minLeaf; i <= rows.length - minLeaf; i++) {
        if (i === 0 || i === rows.length) continue;
        if (sorted[i - 1][dim] === sorted[i][dim]) continue;
        const thr = (sorted[i - 1][dim] + sorted[i][dim]) / 2;
        const L = sorted.slice(0, i), R = sorted.slice(i);
        const g = parentImp - (L.length / rows.length) * imp(L.map(p => p[2]))
                            - (R.length / rows.length) * imp(R.map(p => p[2]));
        if (!best || g > best.g) best = { g, dim, thr, L, R };
      }
    }
    if (!best || best.g < 1e-9) return { leaf: true, label: majority, n: rows.length };
    return {
      leaf: false, dim: best.dim, thr: best.thr,
      left: buildTree(best.L, depth + 1), right: buildTree(best.R, depth + 1),
    };
  }
  function predict(t, x, y) {
    if (t.leaf) return t.label;
    const v = t.dim === 0 ? x : y;
    return predict(v <= t.thr ? t.left : t.right, x, y);
  }

  let tree = null;
  function rebuild() {
    const all = gen(2024);
    train = all.slice(0, 140); test = all.slice(140);
    refresh();
  }
  function refresh() {
    tree = buildTree(train, 0);
    let leaves = 0;
    const walk = n => { if (n.leaf) leaves++; else { walk(n.left); walk(n.right); } };
    walk(tree);
    const acc = set => set.filter(p => predict(tree, p[0], p[1]) === p[2]).length / set.length;
    const tr = acc(train), te = acc(test);
    out.set([
      String(leaves),
      { html: `${fmt(tr * 100, 1)}%`, cls: tr > .99 ? 'is-warn' : '' },
      { html: `${fmt(te * 100, 1)}%`, cls: te > .88 ? 'is-ok' : te < .78 ? 'is-warn' : '' },
      { html: `${fmt((tr - te) * 100, 1)} pts`, cls: tr - te > .12 ? 'is-warn' : '' },
    ]);
    st.set(
      maxDepth <= 2
        ? `${WARN}<span><strong>Under-fitting.</strong> Only ${leaves} leaves — too few axis-aligned cuts to describe this shape.</span>`
        : tr - te > .12
          ? `${WARN}<span><strong>Over-fitting.</strong> Training accuracy ${fmt(tr * 100, 1)}% but test only ${fmt(te * 100, 1)}%. Those thin slivers are the tree carving out individual noisy points. Raise "min samples per leaf" to stop it.</span>`
          : `${OK}<span>Training and test accuracy are close — the tree is describing structure, not noise.</span>`,
      maxDepth <= 2 || tr - te > .12 ? 'warn' : 'ok');
    plot.render();
  }

  plot.onDraw(pl => {
    const step = 4;
    const W = Math.ceil(pl.w / step), H = Math.ceil(pl.h / step);
    for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
      const [x, y] = pl.toWorld(i * step, j * step);
      const v = predict(tree, x, y);
      pl.ctx.fillStyle = v > 0 ? C.c1 : C.c4;
      pl.ctx.globalAlpha = .17;
      pl.ctx.fillRect(i * step, j * step, step + 1, step + 1);
    }
    pl.ctx.globalAlpha = 1;
    // draw the actual split lines
    const drawSplits = (n, x0, x1, y0, y1) => {
      if (n.leaf) return;
      if (n.dim === 0) {
        pl.line([n.thr, y0], [n.thr, y1], { color: C.ink, lw: 1.3, alpha: .5 });
        drawSplits(n.left, x0, n.thr, y0, y1);
        drawSplits(n.right, n.thr, x1, y0, y1);
      } else {
        pl.line([x0, n.thr], [x1, n.thr], { color: C.ink, lw: 1.3, alpha: .5 });
        drawSplits(n.left, x0, x1, y0, n.thr);
        drawSplits(n.right, x0, x1, n.thr, y1);
      }
    };
    drawSplits(tree, -2.6, 2.6, -2.6, 2.6);
    test.forEach(p => pl.dot([p[0], p[1]], { r: 2.4, color: p[2] > 0 ? C.c1 : C.c4, alpha: .3 }));
    train.forEach(p => pl.dot([p[0], p[1]], { r: 4, color: p[2] > 0 ? C.c1 : C.c4, ring: true, ringLw: 1.5 }));
    pl.text({ px: 12, py: 16 }, 'solid dots: training · faint: held-out test', { color: C.muted, size: 10.5 });
  });

  rebuild();

  node.appendChild(note(
    `Every cut a tree makes is <strong>perpendicular to one axis</strong> — it can only ever produce ` +
    `rectangles. On the diagonal dataset that is painfully inefficient: the true boundary is a straight line ` +
    `at 45°, and the tree has to approximate it with a staircase. Push the depth to 12 and watch it grow ` +
    `slivers around individual noisy points while test accuracy falls. <strong>Min samples per leaf</strong> ` +
    `is pre-pruning, and it is usually the single most effective control here.`
  ));
});

/* ============================================================
   5. Choosing a threshold for a continuous feature
   ============================================================ */
defineWidget('continuous-split', node => {
  const { right, canvas } = split(node, { hint: 'Drag the threshold', wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: 19.4, xmax: 22.6, ymin: -.34, ymax: 1.34, aspect: 1.8, equal: false, pad: 0,
  }));

  const DATA = [
    { t: 20.0, y: 'Yes' }, { t: 20.6, y: 'Yes' }, { t: 21.1, y: 'Yes' },
    { t: 21.7, y: 'No' }, { t: 22.0, y: 'No' },
  ];
  let thr = 21.4;

  const thrCtl = slider('Threshold θ', {
    min: 19.6, max: 22.4, step: .05, value: 21.4, format: v => `${fmt(v, 2)} °C`,
    onInput: v => { thr = v; refresh(); },
  });
  const out = readout([['left (≤ θ)', 0], ['right (> θ)', 0], ['I_left', 0], ['I_right', 0], ['weighted after', 0], ['gain', 0]]);
  const st = status('');
  right.append(thrCtl.root, out.root, st.root);

  const drag = new Dragger(plot);
  drag.add(() => [thr, .5], p => { thr = clamp(round(p[0], 2), 19.6, 22.4); thrCtl.set(thr); });
  drag.onchange = refresh;

  const gainAt = th => {
    const L = DATA.filter(d => d.t <= th), R = DATA.filter(d => d.t > th);
    const before = entropy(DATA.map(d => d.y));
    const after = (L.length / DATA.length) * entropy(L.map(d => d.y))
                + (R.length / DATA.length) * entropy(R.map(d => d.y));
    return { L, R, before, after, gain: before - after };
  };

  function refresh() { plot.render(); sync(); }
  function sync() {
    const { L, R, before, after, gain } = gainAt(thr);
    const cnt = arr => `${arr.filter(d => d.y === 'Yes').length}Y ${arr.filter(d => d.y === 'No').length}N`;
    out.set([
      `${cnt(L)} (n=${L.length})`, `${cnt(R)} (n=${R.length})`,
      fmt(entropy(L.map(d => d.y)), 4), fmt(entropy(R.map(d => d.y)), 4),
      fmt(after, 4),
      { html: fmt(gain, 4), cls: gain > .99 ? 'is-ok' : '' },
    ]);
    st.set(
      gain > .99
        ? `${OK}<span><strong>Perfect split.</strong> Both children are pure, so the gain equals the full parent entropy of 1 bit.</span>`
        : `${INFO}<span>Gain ${fmt(gain, 3)}. Only thresholds falling <em>between two differently-labelled points</em> can ever be optimal — the rest are wasted evaluations.</span>`,
      gain > .99 ? 'ok' : 'info');
  }

  plot.onDraw(pl => {
    // gain as a function of threshold
    const pts = [];
    for (let t = 19.6; t <= 22.4; t += .01) pts.push([t, gainAt(t).gain]);
    pl.path(pts, { color: C.c2, lw: 2.4 });
    pl.title('information gain as θ sweeps', { color: C.c2, weight: 650 });

    // candidate midpoints
    for (let i = 0; i < DATA.length - 1; i++) {
      const mid = (DATA[i].t + DATA[i + 1].t) / 2;
      const valid = DATA[i].y !== DATA[i + 1].y;
      pl.line([mid, 0], [mid, 1.05], {
        color: valid ? C.c3 : C.muted, lw: valid ? 1.8 : 1,
        dash: [4, 4], alpha: valid ? .95 : .4,
      });
      pl.text([mid, 1.13], valid ? `θ=${fmt(mid, 2)} ✓` : 'skip', {
        align: 'center', size: 9.5, color: valid ? C.c3 : C.muted, weight: valid ? 700 : 500,
        halo: true, haloWidth: 3.5,
      });
    }

    pl.line([thr, -.16], [thr, 1.05], { color: C.c4, lw: 2.4 });
    DATA.forEach(d => {
      pl.dot([d.t, -.07], { r: 7, color: d.y === 'Yes' ? C.c3 : C.c4, ring: true, ringLw: 2 });
      pl.text([d.t, -.15], fmt(d.t, 1),
        { align: 'center', size: 9.5, color: C.muted, baseline: 'top', halo: true, haloWidth: 3.5 });
    });
    pl.axes();
    pl.xlabel('Temperature (°C)');
  });
  refresh();

  node.appendChild(note(
    `Continuous features have infinitely many possible thresholds, but only finitely many <em>useful</em> ones. ` +
    `Sort the samples, and consider only midpoints between adjacent points <strong>whose labels differ</strong> — ` +
    `a threshold between two identically-labelled neighbours cannot change the partition's purity. Here only ` +
    `θ = 21.4 qualifies, and it splits the data perfectly. That reduces an infinite search to at most n−1 ` +
    `candidates per feature, which is what makes trees on continuous data tractable at all.`
  ));
});

/* ============================================================
   6. Activation functions
   ============================================================ */
defineWidget('activations', node => {
  const { right, canvas } = split(node, { hint: 'Drag the marker', wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -5, xmax: 5, ymin: -1.6, ymax: 2.2, aspect: 1.5, equal: false, pad: 0,
  }));

  const ACTS = {
    sigmoid: { name: 'Sigmoid', f: x => 1 / (1 + Math.exp(-x)), d: x => { const s = 1 / (1 + Math.exp(-x)); return s * (1 - s); }, range: '(0, 1)' },
    tanh: { name: 'tanh', f: x => Math.tanh(x), d: x => 1 - Math.tanh(x) ** 2, range: '(−1, 1)' },
    relu: { name: 'ReLU', f: x => Math.max(0, x), d: x => (x > 0 ? 1 : 0), range: '[0, ∞)' },
    leaky: { name: 'Leaky ReLU', f: x => (x > 0 ? x : 0.1 * x), d: x => (x > 0 ? 1 : 0.1), range: '(−∞, ∞)' },
  };
  let key = 'relu', xp = 1.2, showAll = false;

  const pick = segmented(Object.entries(ACTS).map(([k, v]) => ({ label: v.name, value: k })),
    { value: 'relu', label: 'Activation', onChange: v => { key = v; refresh(); } });
  const allCtl = toggle('Overlay all four', { value: false, onChange: v => { showAll = v; plot.render(); } });
  const out = readout([['x', 0], ['f(x)', 0], ["f'(x)", 0], ['range', 0], ["max f'", 0]]);
  const st = status('');
  right.append(pick.root, allCtl.root, out.root, st.root);

  const drag = new Dragger(plot);
  drag.add(() => [xp, ACTS[key].f(xp)], p => { xp = clamp(round(p[0], 2), -5, 5); });
  drag.onchange = refresh;

  function refresh() { plot.render(); sync(); }
  function sync() {
    const a = ACTS[key];
    const maxD = key === 'sigmoid' ? 0.25 : key === 'tanh' ? 1 : 1;
    out.set([
      fmt(xp, 2), fmt(a.f(xp), 4),
      { html: fmt(a.d(xp), 4), cls: Math.abs(a.d(xp)) < .02 ? 'is-warn' : '' },
      a.range, fmt(maxD, 2),
    ]);
    const grad = Math.abs(a.d(xp));
    st.set(
      grad < .02
        ? `${WARN}<span><strong>Gradient ≈ ${fmt(grad, 4)}.</strong> ${key === 'relu' ? 'This unit is dead — a ReLU stuck at x ≤ 0 passes no gradient back and may never recover.' : 'Saturated. Multiply a few of these together through the layers and the signal vanishes entirely.'}</span>`
        : key === 'sigmoid'
          ? `${INFO}<span>Sigmoid's derivative peaks at just <strong>0.25</strong>. Across ten layers that is 0.25¹⁰ ≈ 10⁻⁶ — the vanishing gradient problem in one number.</span>`
          : `${OK}<span>Healthy gradient of ${fmt(grad, 3)} — signal flows back through this unit.</span>`,
      grad < .02 ? 'warn' : key === 'sigmoid' ? 'info' : 'ok');
  }

  plot.onDraw(pl => {
    pl.grid(1, { color: C.grid }); pl.axes({ ticks: 1 });
    if (showAll) {
      Object.entries(ACTS).forEach(([k, a], i) => {
        pl.fn(a.f, { color: [C.c1, C.c2, C.c3, C.c5][i], lw: 1.8, alpha: k === key ? 1 : .45 });
      });
    } else {
      const a = ACTS[key];
      pl.fn(a.f, { color: C.c1, lw: 3, samples: 500 });
      pl.fn(a.d, { color: C.c4, lw: 2.4, dash: [6, 4], samples: 500 });
      // shade the saturated region
      pl.ctx.fillStyle = C.c4; pl.ctx.globalAlpha = .07;
      for (let x = -5; x < 5; x += .05) {
        if (Math.abs(a.d(x)) < .02) pl.ctx.fillRect(pl.X(x), 0, pl.px(.05) + 1, pl.h);
      }
      pl.ctx.globalAlpha = 1;
      pl.dot([xp, a.f(xp)], { r: 6, color: C.c1, ring: true });
      pl.dot([xp, a.d(xp)], { r: 5, color: C.c4, ring: true });
      pl.line([xp, -1.6], [xp, 2.2], { color: C.c4, lw: 1, dash: [3, 3], alpha: .5 });
      const key2 = [[C.c1, 'f(x)'], [C.c4, "f'(x)"]];
      key2.forEach(([col, lbl], i) => {
        pl.ctx.strokeStyle = col; pl.ctx.lineWidth = 2.6;
        pl.ctx.beginPath(); pl.ctx.moveTo(14, 16 + i * 15); pl.ctx.lineTo(32, 16 + i * 15); pl.ctx.stroke();
        pl.text({ px: 37, py: 16 + i * 15 }, lbl, { color: C.muted, size: 10.5, weight: 600 });
      });
      pl.text({ px: 12, py: pl.h - 10 }, 'shaded: gradient below 0.02 (effectively dead)',
        { color: C.muted, size: 10 });
    }
  });
  refresh();

  node.appendChild(note(
    `The dashed curve is what backpropagation actually multiplies by. For <strong>sigmoid</strong> it never ` +
    `exceeds 0.25, so each layer shrinks the gradient by at least a factor of four — stack ten and the signal ` +
    `is gone. <strong>tanh</strong> improves this to a maximum of 1 and is zero-centred, but still saturates. ` +
    `<strong>ReLU</strong> passes gradient unchanged wherever x &gt; 0, which is why it made deep networks ` +
    `trainable — at the cost of passing <em>nothing</em> when x ≤ 0. <strong>Leaky ReLU</strong> keeps a ` +
    `small slope there so a unit can always recover.`
  ));
});

/* ============================================================
   7. Forward pass through a 3 -> 4 -> 2 network
   ============================================================ */
defineWidget('fnn-forward', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: 0, xmax: 10, ymin: 0, ymax: 7, aspect: 1.5, equal: false, pad: 0,
  }));

  const r = ML.rng(31);
  let W1 = Array.from({ length: 4 }, () => Array.from({ length: 3 }, () => round(ML.gauss(r) * .8, 2)));
  let b1 = Array.from({ length: 4 }, () => round(ML.gauss(r) * .3, 2));
  let W2 = Array.from({ length: 2 }, () => Array.from({ length: 4 }, () => round(ML.gauss(r) * .8, 2)));
  let b2 = [round(ML.gauss(r) * .3, 2), round(ML.gauss(r) * .3, 2)];
  let x = [1, -0.5, 0.8];
  let act = 'relu';

  const xIn = matrixInput(1, 3, [x], {
    label: 'Input x', step: .1, onInput: m => { x = m[0]; refresh(); },
  });
  const actCtl = segmented([
    { label: 'ReLU', value: 'relu' }, { label: 'tanh', value: 'tanh' }, { label: 'Sigmoid', value: 'sigmoid' },
  ], { value: 'relu', label: 'Hidden activation', onChange: v => { act = v; refresh(); } });
  const acts = el('div', { class: 'pg-actions' },
    button('Randomise weights', () => {
      const rr = ML.rng(Math.floor(Math.random() * 1e6));
      W1 = W1.map(row => row.map(() => round(ML.gauss(rr) * .8, 2)));
      b1 = b1.map(() => round(ML.gauss(rr) * .3, 2));
      W2 = W2.map(row => row.map(() => round(ML.gauss(rr) * .8, 2)));
      b2 = b2.map(() => round(ML.gauss(rr) * .3, 2));
      refresh();
    }));
  const out = readout([['z⁽¹⁾', 0], ['a⁽¹⁾', 0], ['z⁽²⁾', 0], ['o = softmax(z⁽²⁾)', 0], ['prediction', 0]]);
  const st = status('');
  right.append(xIn.root, actCtl.root, acts, out.root, st.root);

  const F = { relu: v => Math.max(0, v), tanh: v => Math.tanh(v), sigmoid: v => 1 / (1 + Math.exp(-v)) };

  let state = null;
  function refresh() {
    const z1 = W1.map((row, i) => row.reduce((s, w, j) => s + w * x[j], 0) + b1[i]);
    const a1 = z1.map(F[act]);
    const z2 = W2.map((row, i) => row.reduce((s, w, j) => s + w * a1[j], 0) + b2[i]);
    const mx = Math.max(...z2);
    const ex = z2.map(v => Math.exp(v - mx));
    const sum = ex.reduce((s, v) => s + v, 0);
    const o = ex.map(v => v / sum);
    state = { z1, a1, z2, o };
    const v = arr => arr.map(q => fmt(q, 2)).join(', ');
    out.set([v(z1), v(a1), v(z2), v(o),
      { html: `class ${o[0] >= o[1] ? 1 : 2} (${fmt(Math.max(...o) * 100, 1)}%)`, cls: 'is-ok' }]);
    const dead = act === 'relu' ? a1.filter(q => q === 0).length : 0;
    st.set(
      dead > 0
        ? `${WARN}<span><strong>${dead} of 4 hidden units are outputting exactly 0.</strong> Under ReLU those units contribute nothing to this input, and receive no gradient from it either.</span>`
        : `${INFO}<span>Values flow left to right: multiply by weights, add the bias, apply the activation. Softmax at the end turns two raw scores into probabilities that sum to 1.</span>`,
      dead > 0 ? 'warn' : 'info');
    plot.render();
  }

  plot.onDraw(pl => {
    const layers = [
      { n: 3, x: 1.3, vals: x, label: 'input x' },
      { n: 4, x: 5, vals: state.a1, label: `a⁽¹⁾ = ${act}(z⁽¹⁾)` },
      { n: 2, x: 8.7, vals: state.o, label: 'o = softmax' },
    ];
    const ys = (n, i) => 6 - (i + .5) * (5 / n) - (4 - n) * .12;
    // edges, thickness by |weight|
    const drawEdges = (Wm, from, to) => {
      Wm.forEach((row, i) => row.forEach((w, j) => {
        const a = [layers[from].x, ys(layers[from].n, j)];
        const b = [layers[to].x, ys(layers[to].n, i)];
        pl.line(a, b, {
          color: w >= 0 ? C.c1 : C.c4,
          lw: Math.min(4, .4 + Math.abs(w) * 2.2),
          alpha: .38,
        });
      }));
    };
    drawEdges(W1, 0, 1);
    drawEdges(W2, 1, 2);

    layers.forEach((L, li) => {
      for (let i = 0; i < L.n; i++) {
        const y = ys(L.n, i);
        const v = L.vals[i];
        const t = clamp(Math.abs(v) / (li === 2 ? 1 : 2.2), 0, 1);
        pl.ctx.beginPath();
        pl.ctx.arc(pl.X(L.x), pl.Y(y), 22, 0, Math.PI * 2);
        pl.ctx.fillStyle = v === 0 && li === 1 ? C.grid
          : (v >= 0 ? withA(C.c1, .16 + t * .6) : withA(C.c4, .16 + t * .6));
        pl.ctx.fill();
        pl.ctx.strokeStyle = v === 0 && li === 1 ? C.muted : (v >= 0 ? C.c1 : C.c4);
        pl.ctx.lineWidth = 2; pl.ctx.stroke();
        pl.text([L.x, y], fmt(v, 2), { align: 'center', size: 11, weight: 700, mono: true,
          color: t > .55 ? C.raised : C.ink });
      }
      pl.text([L.x, .55], L.label, { align: 'center', size: 11, color: C.muted, weight: 600 });
    });
    pl.text({ px: 12, py: 14 }, 'edge thickness ∝ |weight| · violet positive, red negative',
      { color: C.muted, size: 10.5 });
  });

  function withA(hex, a) {
    hex = (hex || '').trim();
    if (!hex.startsWith('#')) return hex;
    const n = hex.length === 4
      ? hex.slice(1).split('').map(c => parseInt(c + c, 16))
      : [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));
    return `rgba(${n.join(',')},${a})`;
  }

  refresh();

  node.appendChild(note(
    `The whole network is two matrix multiplies with an activation between them: ` +
    `<span class="u-mono">z⁽¹⁾ = W⁽¹⁾x + b⁽¹⁾</span>, <span class="u-mono">a⁽¹⁾ = f(z⁽¹⁾)</span>, ` +
    `<span class="u-mono">z⁽²⁾ = W⁽²⁾a⁽¹⁾ + b⁽²⁾</span>, <span class="u-mono">o = softmax(z⁽²⁾)</span>. ` +
    `Switch the activation to see how differently the same weights behave — and note how often ReLU leaves ` +
    `units at exactly zero. Without a nonlinearity here the two matrices would collapse into one, and the ` +
    `whole network would be a single linear map.`
  ));
});

/* ============================================================
   8. Backpropagation, one step at a time
   ============================================================ */
defineWidget('backprop', node => {
  const wrap = el('div');
  node.appendChild(wrap);

  const r = ML.rng(7);
  const W1 = [[0.5, -0.3, 0.8], [-0.6, 0.4, 0.2], [0.3, 0.7, -0.5], [0.1, -0.2, 0.6]];
  const b1 = [0.1, -0.2, 0.05, 0.3];
  const W2 = [[0.4, -0.5, 0.6, 0.2], [-0.3, 0.7, -0.1, 0.5]];
  const b2 = [0.05, -0.1];
  const x = [1.0, -0.5, 0.8];
  const p = [1, 0];
  let step = 0;

  const STEPS = [
    'Forward: z⁽¹⁾ = W⁽¹⁾x + b⁽¹⁾',
    'Forward: a⁽¹⁾ = ReLU(z⁽¹⁾)',
    'Forward: z⁽²⁾ = W⁽²⁾a⁽¹⁾ + b⁽²⁾',
    'Forward: o = softmax(z⁽²⁾), then the loss',
    'Backward: δ⁽²⁾ = o − p',
    'Backward: ∇W⁽²⁾ = δ⁽²⁾(a⁽¹⁾)ᵀ',
    'Backward: δ⁽¹⁾ = (W⁽²⁾)ᵀδ⁽²⁾ ⊙ ReLU′(z⁽¹⁾)',
    'Backward: ∇W⁽¹⁾ = δ⁽¹⁾xᵀ',
  ];

  const nav = el('div', { class: 'pg-actions' },
    button('◀ Back', () => { step = Math.max(0, step - 1); refresh(); }),
    button('Step ▶', () => { step = Math.min(STEPS.length - 1, step + 1); refresh(); }),
    button('Restart', () => { step = 0; refresh(); }),
  );
  const label = el('div', { style: 'font-size:.95rem;font-weight:640;margin:.7rem 0 .5rem' });
  const body = el('div', { class: 'readout', style: 'min-height:9em' });
  const st = status('');
  wrap.append(nav, label, body, st.root);

  // forward + backward, computed once
  const z1 = W1.map((row, i) => row.reduce((s, w, j) => s + w * x[j], 0) + b1[i]);
  const a1 = z1.map(v => Math.max(0, v));
  const z2 = W2.map((row, i) => row.reduce((s, w, j) => s + w * a1[j], 0) + b2[i]);
  const mx = Math.max(...z2);
  const ex = z2.map(v => Math.exp(v - mx));
  const sum = ex.reduce((s, v) => s + v, 0);
  const o = ex.map(v => v / sum);
  const loss = -p.reduce((s, pi, i) => s + (pi ? pi * Math.log(o[i]) : 0), 0);
  const d2 = o.map((v, i) => v - p[i]);
  const gW2 = d2.map(di => a1.map(aj => di * aj));
  const back = a1.map((_, j) => W2.reduce((s, row, i) => s + row[j] * d2[i], 0));
  const d1 = back.map((v, i) => v * (z1[i] > 0 ? 1 : 0));
  const gW1 = d1.map(di => x.map(xj => di * xj));

  const vec = (a, d = 4) => '[' + a.map(v => fmt(v, d)).join(', ') + ']';
  const mat = (M, d = 4) => M.map(row => '  [' + row.map(v => fmt(v, d)).join(', ') + ']').join('\n');

  function refresh() {
    label.textContent = `Step ${step + 1} / ${STEPS.length}: ${STEPS[step]}`;
    const parts = [];
    if (step >= 0) parts.push(`x  = ${vec(x, 2)}   target p = ${vec(p, 0)}`);
    if (step >= 0) parts.push(`z⁽¹⁾ = ${vec(z1)}`);
    if (step >= 1) parts.push(`a⁽¹⁾ = ${vec(a1)}   ← note unit ${z1.findIndex(v => v <= 0) + 1} is clamped to 0`);
    if (step >= 2) parts.push(`z⁽²⁾ = ${vec(z2)}`);
    if (step >= 3) parts.push(`o   = ${vec(o)}\nloss = −log(o₁) = ${fmt(loss, 4)}`);
    if (step >= 4) parts.push(`δ⁽²⁾ = o − p = ${vec(d2)}`);
    if (step >= 5) parts.push(`∇W⁽²⁾ = δ⁽²⁾(a⁽¹⁾)ᵀ  (2×4)\n${mat(gW2)}`);
    if (step >= 6) parts.push(`(W⁽²⁾)ᵀδ⁽²⁾ = ${vec(back)}\nReLU′(z⁽¹⁾) = ${vec(z1.map(v => (v > 0 ? 1 : 0)), 0)}\nδ⁽¹⁾ = ${vec(d1)}`);
    if (step >= 7) parts.push(`∇W⁽¹⁾ = δ⁽¹⁾xᵀ  (4×3)\n${mat(gW1)}`);
    body.innerHTML = parts.map(t => `<div style="white-space:pre;margin-bottom:.45em">${t}</div>`).join('');

    const msgs = {
      3: `${INFO}<span>The network currently gives the true class a probability of ${fmt(o[0], 3)}, so the loss is ${fmt(loss, 3)}.</span>`,
      4: `${OK}<span><strong>δ⁽²⁾ = o − p.</strong> The whole softmax + cross-entropy derivative collapses to this one subtraction — that cancellation is why the pair is used almost universally.</span>`,
      6: `${WARN}<span>Look at δ⁽¹⁾: the entry for the clamped unit is <strong>exactly 0</strong>. ReLU′ acts as a gate — a unit that was off contributes no gradient at all.</span>`,
      7: `${OK}<span>Both gradients are <strong>outer products</strong>: δ from the layer above times the activations from below. That structure is all backpropagation really is.</span>`,
    };
    st.set(msgs[step] || `${INFO}<span>Values flow forward; derivatives flow back. Step through both directions.</span>`,
      step === 6 ? 'warn' : (step === 4 || step === 7) ? 'ok' : 'info');
  }
  refresh();

  node.appendChild(note(
    `Real numbers, no symbols. The forward pass fills in the values; the backward pass reuses them. Notice ` +
    `that <strong>nothing is recomputed</strong> — <span class="u-mono">a⁽¹⁾</span> from the forward pass is ` +
    `exactly what <span class="u-mono">∇W⁽²⁾</span> needs, which is why the activations must be kept in ` +
    `memory during training and why deep networks are memory-hungry.`
  ));
});

/* ============================================================
   9. Train a network on 2D data
   ============================================================ */
defineWidget('fnn-playground', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -2.4, xmax: 2.4, ymin: -2.4, ymax: 2.4, aspect: 1.1, pad: 0,
  }));

  let hidden = 8, layers = 1, act = 'relu', lr = 0.08, dsKey = 'xor';
  let net = null, epoch = 0, timer = null, lossHist = [];

  const DS = {
    xor: 'XOR', circle: 'Rings', moons: 'Two moons', spiral: 'Spiral',
  };
  let data = [];

  const dsCtl = segmented(Object.entries(DS).map(([k, v]) => ({ label: v, value: k })),
    { value: 'xor', label: 'Dataset', onChange: v => { dsKey = v; reset(); } });
  const hCtl = slider('Hidden units per layer', {
    min: 1, max: 24, step: 1, value: 8, format: v => String(v),
    onInput: v => { hidden = v; reset(); },
  });
  const lCtl = slider('Hidden layers', {
    min: 1, max: 3, step: 1, value: 1, format: v => String(v),
    onInput: v => { layers = v; reset(); },
  });
  const actCtl = segmented([
    { label: 'ReLU', value: 'relu' }, { label: 'tanh', value: 'tanh' }, { label: 'Sigmoid', value: 'sigmoid' },
  ], { value: 'relu', label: 'Activation', onChange: v => { act = v; reset(); } });
  const lrCtl = slider('Learning rate η', {
    min: .005, max: .5, step: .005, value: .08, format: v => fmt(v, 3),
    onInput: v => { lr = v; },
  });
  const acts = el('div', { class: 'pg-actions' },
    button('▶ Train', () => start()),
    button('❚❚ Pause', () => stop()),
    button('Reset', () => reset()),
  );
  const out = readout([['epoch', 0], ['loss', 0], ['training accuracy', 0], ['parameters', 0]]);
  const st = status('');
  right.append(dsCtl.root, hCtl.root, lCtl.root, actCtl.root, lrCtl.root, acts, out.root, st.root);

  const F = {
    relu: { f: v => Math.max(0, v), d: v => (v > 0 ? 1 : 0) },
    tanh: { f: v => Math.tanh(v), d: v => 1 - Math.tanh(v) ** 2 },
    sigmoid: { f: v => 1 / (1 + Math.exp(-v)), d: v => { const s = 1 / (1 + Math.exp(-v)); return s * (1 - s); } },
  };

  function genData() {
    const r = ML.rng(11);
    const pts = [];
    for (let i = 0; i < 160; i++) {
      let x, y, lab;
      if (dsKey === 'xor') {
        x = (r() * 2 - 1) * 1.9; y = (r() * 2 - 1) * 1.9; lab = x * y > 0 ? 1 : 0;
      } else if (dsKey === 'circle') {
        const inner = i % 2 === 0;
        const rad = inner ? .45 + Math.abs(ML.gauss(r)) * .25 : 1.45 + Math.abs(ML.gauss(r)) * .25;
        const t = r() * Math.PI * 2;
        x = Math.cos(t) * rad; y = Math.sin(t) * rad; lab = inner ? 1 : 0;
      } else if (dsKey === 'moons') {
        const up = i % 2 === 0;
        const t = r() * Math.PI;
        if (up) { x = Math.cos(t) * 1.3 - .45; y = Math.sin(t) * .95 - .25; }
        else { x = -Math.cos(t) * 1.3 + .45; y = -Math.sin(t) * .95 + .25; }
        x += ML.gauss(r) * .16; y += ML.gauss(r) * .16;
        lab = up ? 1 : 0;
      } else {
        const arm = i % 2;
        const t = (i / 160) * 3.4 + .4;
        const ang = t * 2.2 + arm * Math.PI;
        x = Math.cos(ang) * t * .55 + ML.gauss(r) * .1;
        y = Math.sin(ang) * t * .55 + ML.gauss(r) * .1;
        lab = arm;
      }
      pts.push([x, y, lab]);
    }
    return pts;
  }

  function initNet() {
    const r = ML.rng(99);
    const sizes = [2, ...Array(layers).fill(hidden), 2];
    const Ws = [], bs = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const fanIn = sizes[l];
      const scale = Math.sqrt(2 / fanIn);
      Ws.push(Array.from({ length: sizes[l + 1] }, () =>
        Array.from({ length: fanIn }, () => ML.gauss(r) * scale)));
      bs.push(Array.from({ length: sizes[l + 1] }, () => 0));
    }
    return { Ws, bs, sizes };
  }

  function forward(n, input) {
    const zs = [], as = [input];
    for (let l = 0; l < n.Ws.length; l++) {
      const z = n.Ws[l].map((row, i) => row.reduce((s, w, j) => s + w * as[l][j], 0) + n.bs[l][i]);
      zs.push(z);
      as.push(l === n.Ws.length - 1 ? softmax(z) : z.map(F[act].f));
    }
    return { zs, as };
  }
  function softmax(z) {
    const m = Math.max(...z);
    const e = z.map(v => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map(v => v / s);
  }

  function trainEpoch() {
    const L = net.Ws.length;
    const gW = net.Ws.map(W => W.map(row => row.map(() => 0)));
    const gb = net.bs.map(b => b.map(() => 0));
    let loss = 0;
    data.forEach(([px, py, lab]) => {
      const { zs, as } = forward(net, [px, py]);
      const o = as[L];
      loss -= Math.log(Math.max(1e-12, o[lab]));
      let delta = o.map((v, i) => v - (i === lab ? 1 : 0));
      for (let l = L - 1; l >= 0; l--) {
        for (let i = 0; i < gW[l].length; i++) {
          gb[l][i] += delta[i];
          for (let j = 0; j < gW[l][i].length; j++) gW[l][i][j] += delta[i] * as[l][j];
        }
        if (l > 0) {
          const back = as[l].map((_, j) => net.Ws[l].reduce((s, row, i) => s + row[j] * delta[i], 0));
          delta = back.map((v, j) => v * F[act].d(zs[l - 1][j]));
        }
      }
    });
    const N = data.length;
    for (let l = 0; l < L; l++) {
      for (let i = 0; i < net.Ws[l].length; i++) {
        net.bs[l][i] -= lr * gb[l][i] / N;
        for (let j = 0; j < net.Ws[l][i].length; j++) net.Ws[l][i][j] -= lr * gW[l][i][j] / N;
      }
    }
    epoch++;
    lossHist.push(loss / N);
    if (lossHist.length > 260) lossHist.shift();
    return loss / N;
  }

  function reset() {
    stop();
    data = genData();
    net = initNet();
    epoch = 0; lossHist = [];
    refresh();
  }
  function start() {
    if (timer) return;
    timer = setInterval(() => { for (let i = 0; i < 4; i++) trainEpoch(); refresh(); }, 40);
  }
  function stop() { clearInterval(timer); timer = null; }

  function refresh() {
    const L = net.Ws.length;
    const acc = data.filter(([px, py, lab]) => {
      const o = forward(net, [px, py]).as[L];
      return (o[1] > o[0] ? 1 : 0) === lab;
    }).length / data.length;
    const nparams = net.Ws.reduce((s, W) => s + W.length * W[0].length, 0)
                  + net.bs.reduce((s, b) => s + b.length, 0);
    out.set([
      String(epoch),
      lossHist.length ? fmt(lossHist[lossHist.length - 1], 4) : '—',
      { html: `${fmt(acc * 100, 1)}%`, cls: acc > .95 ? 'is-ok' : acc < .7 ? 'is-warn' : '' },
      String(nparams),
    ]);
    st.set(
      epoch === 0
        ? `${INFO}<span>Press <strong>Train</strong>. Watch the boundary bend into shape as gradient descent runs.</span>`
        : acc > .96
          ? `${OK}<span>${fmt(acc * 100, 1)}% with ${nparams} parameters. The hidden layer has learned features that make the classes linearly separable at the output.</span>`
          : dsKey === 'spiral' && hidden < 8
            ? `${WARN}<span>Too few hidden units for a spiral — the network does not have the capacity to bend this far. Increase the width.</span>`
            : `${INFO}<span>Training… ${fmt(acc * 100, 1)}%. If it stalls, try more units, another activation, or a different learning rate.</span>`,
      epoch === 0 ? 'info' : acc > .96 ? 'ok' : (dsKey === 'spiral' && hidden < 8) ? 'warn' : 'info');
    plot.render();
  }

  plot.onDraw(pl => {
    const L = net.Ws.length;
    const step = 5;
    const W = Math.ceil(pl.w / step), H = Math.ceil(pl.h / step);
    for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
      const [px, py] = pl.toWorld(i * step, j * step);
      const o = forward(net, [px, py]).as[L];
      const t = Math.abs(o[1] - o[0]);
      pl.ctx.fillStyle = o[1] > o[0] ? C.c1 : C.c4;
      pl.ctx.globalAlpha = .08 + t * .3;
      pl.ctx.fillRect(i * step, j * step, step + 1, step + 1);
    }
    pl.ctx.globalAlpha = 1;
    data.forEach(([px, py, lab]) => pl.dot([px, py], { r: 4, color: lab ? C.c1 : C.c4, ring: true, ringLw: 1.6 }));

    // loss curve inset
    if (lossHist.length > 2) {
      const iw = Math.min(170, pl.w * .4), ih = iw * .42;
      const ix = pl.w - iw - 10, iy = 10;
      pl.ctx.fillStyle = C.raised; pl.ctx.globalAlpha = .92;
      pl.ctx.fillRect(ix, iy, iw, ih); pl.ctx.globalAlpha = 1;
      pl.ctx.strokeStyle = C.grid; pl.ctx.strokeRect(ix, iy, iw, ih);
      const mx = Math.max(...lossHist), mn = Math.min(...lossHist);
      pl.ctx.strokeStyle = C.c2; pl.ctx.lineWidth = 1.8; pl.ctx.beginPath();
      lossHist.forEach((v, i) => {
        const X = ix + 6 + (iw - 12) * i / (lossHist.length - 1);
        const Y = iy + ih - 8 - (ih - 16) * (mx - mn < 1e-9 ? .5 : (v - mn) / (mx - mn));
        i ? pl.ctx.lineTo(X, Y) : pl.ctx.moveTo(X, Y);
      });
      pl.ctx.stroke();
      pl.text({ px: ix + 6, py: iy + 10 }, 'training loss', { size: 9.5, color: C.muted });
    }
  });

  reset();
  node.addEventListener('pagehide', stop);

  node.appendChild(note(
    `A hidden layer's job is to <strong>bend the space until the problem becomes linearly separable</strong>, ` +
    `so the output layer can finish it with a straight cut. Start with XOR and one hidden layer of two units — ` +
    `usually just barely enough. Then try the <strong>spiral</strong> with 4 units and watch it fail outright, ` +
    `before widening it. Switch to <strong>sigmoid</strong> with three layers and notice how much more slowly ` +
    `it learns: that is the vanishing gradient from the activation figure, showing up as wasted epochs.`
  ));
});

/* ============================================================
   10. Universal approximation with ReLU bumps
   ============================================================ */
defineWidget('universal-approx', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -.2, xmax: 6.5, ymin: -1.7, ymax: 1.7, aspect: 1.7, equal: false, pad: 0,
  }));

  let nUnits = 6, showPieces = true, target = 'sine';
  const TARGETS = {
    sine: { name: 'sin(x)', f: x => Math.sin(x) },
    step: { name: 'Step', f: x => (x < 2 ? -0.8 : x < 4 ? 0.9 : -0.4) },
    wiggle: { name: 'Wiggly', f: x => Math.sin(x) * Math.cos(2.1 * x) },
  };

  const nCtl = slider('Hidden ReLU units', {
    min: 1, max: 40, step: 1, value: 6, format: v => String(v),
    onInput: v => { nUnits = v; refresh(); },
  });
  const tCtl = segmented(Object.entries(TARGETS).map(([k, v]) => ({ label: v.name, value: k })),
    { value: 'sine', label: 'Target function', onChange: v => { target = v; refresh(); } });
  const pCtl = toggle('Show the individual units', { value: true, onChange: v => { showPieces = v; plot.render(); } });
  const out = readout([['hidden units', 0], ['parameters', 0], ['max error', 0], ['mean squared error', 0]]);
  const st = status('');
  right.append(nCtl.root, tCtl.root, pCtl.root, out.root, st.root);

  // fit a 1-hidden-layer ReLU net by least squares on fixed, evenly spaced knots
  let model = null;
  function refresh() {
    const f = TARGETS[target].f;
    const knots = Array.from({ length: nUnits }, (_, i) => 0.2 + (6.1 * i) / Math.max(1, nUnits));
    const xs = Array.from({ length: 260 }, (_, i) => i / 259 * 6.3);
    // design matrix: [1, x, relu(x-k1), ..., relu(x-kn)]
    const basis = x => [1, x, ...knots.map(k => Math.max(0, x - k))];
    const m = 2 + nUnits;
    const A = Array.from({ length: m }, () => new Array(m).fill(0));
    const bv = new Array(m).fill(0);
    xs.forEach(x => {
      const phi = basis(x), y = f(x);
      for (let i = 0; i < m; i++) {
        bv[i] += phi[i] * y;
        for (let j = 0; j < m; j++) A[i][j] += phi[i] * phi[j];
      }
    });
    for (let i = 0; i < m; i++) A[i][i] += 1e-7;
    // Gaussian elimination
    const M = A.map((row, i) => [...row, bv[i]]);
    for (let i = 0; i < m; i++) {
      let piv = i;
      for (let r2 = i + 1; r2 < m; r2++) if (Math.abs(M[r2][i]) > Math.abs(M[piv][i])) piv = r2;
      [M[i], M[piv]] = [M[piv], M[i]];
      if (Math.abs(M[i][i]) < 1e-14) continue;
      for (let r2 = 0; r2 < m; r2++) {
        if (r2 === i) continue;
        const fct = M[r2][i] / M[i][i];
        for (let c = i; c <= m; c++) M[r2][c] -= fct * M[i][c];
      }
    }
    const w = M.map((row, i) => (Math.abs(row[i]) < 1e-14 ? 0 : row[m] / row[i]));
    const predict = x => basis(x).reduce((s, v, i) => s + v * w[i], 0);
    let maxErr = 0, mse = 0;
    xs.forEach(x => { const e = Math.abs(predict(x) - f(x)); maxErr = Math.max(maxErr, e); mse += e * e; });
    mse /= xs.length;
    model = { predict, knots, w, f };
    out.set([
      String(nUnits), String(2 + nUnits),
      fmt(maxErr, 4),
      { html: fmt(mse, 6), cls: mse < 1e-3 ? 'is-ok' : mse > .05 ? 'is-warn' : '' },
    ]);
    st.set(
      nUnits <= 2
        ? `${WARN}<span>With ${nUnits} unit${nUnits > 1 ? 's' : ''} the network can only produce ${nUnits + 1} straight segments — nowhere near enough to follow a curve.</span>`
        : mse < 1e-3
          ? `${OK}<span>Mean squared error ${fmt(mse, 6)}. ${nUnits} ReLU units and the fit is essentially exact — this is the universal approximation theorem in practice.</span>`
          : `${INFO}<span>Each extra unit adds one more kink. Keep increasing and watch the error fall.</span>`,
      nUnits <= 2 ? 'warn' : mse < 1e-3 ? 'ok' : 'info');
    plot.render();
  }

  plot.onDraw(pl => {
    pl.grid(1, { color: C.grid });
    if (showPieces) {
      model.knots.forEach((k, i) => {
        const c = model.w[i + 2];
        pl.fn(x => Math.max(0, x - k) * c, { color: C.c3, lw: 1.2, alpha: .45, from: 0, to: 6.3 });
        pl.line([k, -1.7], [k, 1.7], { color: C.c3, lw: .8, dash: [3, 4], alpha: .3 });
      });
    }
    pl.fn(model.f, { color: C.muted, lw: 2.6, dash: [6, 4], from: 0, to: 6.3, samples: 400 });
    pl.fn(model.predict, { color: C.c1, lw: 3, from: 0, to: 6.3, samples: 500 });
    pl.axes({ ticks: 1 });
    const key = [[C.muted, 'target'], [C.c1, 'network output'], ...(showPieces ? [[C.c3, 'individual ReLU units']] : [])];
    key.forEach(([col, lbl], i) => {
      pl.ctx.strokeStyle = col; pl.ctx.lineWidth = 2.6;
      pl.ctx.beginPath(); pl.ctx.moveTo(14, 16 + i * 15); pl.ctx.lineTo(32, 16 + i * 15); pl.ctx.stroke();
      pl.text({ px: 37, py: 16 + i * 15 }, lbl, { color: C.muted, size: 10.5, weight: 600 });
    });
  });
  refresh();

  node.appendChild(note(
    `Each ReLU unit contributes one <strong>kink</strong> — a hinge that switches on at its own threshold. ` +
    `Add enough of them, weighted appropriately, and their sum is a piecewise-linear function that can follow ` +
    `any continuous curve as closely as you like. That is the <strong>universal approximation theorem</strong>, ` +
    `and it is far less mysterious than it sounds: a single hidden layer is just a very flexible way to build ` +
    `a piecewise-linear function. Note what the theorem does <em>not</em> say — it guarantees such a network ` +
    `<em>exists</em>, not that gradient descent will find it, and not that it will generalise.`
  ));
});

/* ============================================================
   Full example: what each layer of a 2→8→8→2 network learns on XOR
   Rebuilds the source chapter's out_xor/ panel study as a live network.
   ============================================================ */
defineWidget('fnn-layers', node => {
  const wrap = el('div');
  node.appendChild(wrap);

  const R = 2.2;                       // input square is [-R, R]^2
  const GRID = 34;                     // resolution of every activation map

  /* ---- data: XOR by quadrant ---- */
  function xorData(n, seed) {
    const r = ML.rng(seed);
    const pts = [];
    for (let i = 0; i < n; i++) {
      const x = (r() * 2 - 1) * 1.75, y = (r() * 2 - 1) * 1.75;
      pts.push([x, y, (x > 0) === (y > 0) ? 0 : 1]);
    }
    return pts;
  }
  let data = xorData(220, 7);

  /* ---- a 2 → 8 → 8 → 2 ReLU network ---- */
  const SIZES = [2, 8, 8, 2];
  function initNet(seed) {
    const r = ML.rng(seed);
    const Ws = [], bs = [];
    for (let l = 0; l < SIZES.length - 1; l++) {
      const scale = Math.sqrt(2 / SIZES[l]);
      Ws.push(Array.from({ length: SIZES[l + 1] }, () =>
        Array.from({ length: SIZES[l] }, () => ML.gauss(r) * scale)));
      bs.push(new Array(SIZES[l + 1]).fill(0));
    }
    return { Ws, bs };
  }
  const relu = v => (v > 0 ? v : 0);
  function softmax(z) {
    const m = Math.max(...z);
    const e = z.map(v => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map(v => v / s);
  }
  function forward(n, input) {
    const a1 = n.Ws[0].map((row, i) => relu(row[0] * input[0] + row[1] * input[1] + n.bs[0][i]));
    const a2 = n.Ws[1].map((row, i) => relu(row.reduce((s, w, j) => s + w * a1[j], 0) + n.bs[1][i]));
    const z3 = n.Ws[2].map((row, i) => row.reduce((s, w, j) => s + w * a2[j], 0) + n.bs[2][i]);
    return { a1, a2, z3, o: softmax(z3) };
  }
  function trainEpoch(n, lr) {
    const gW = n.Ws.map(W => W.map(r2 => r2.map(() => 0)));
    const gb = n.bs.map(b => b.map(() => 0));
    let loss = 0, correct = 0;
    for (const [px, py, lab] of data) {
      const { a1, a2, o } = forward(n, [px, py]);
      loss -= Math.log(Math.max(1e-12, o[lab]));
      if ((o[1] > o[0] ? 1 : 0) === lab) correct++;
      const acts = [[px, py], a1, a2];
      let delta = o.map((v, i) => v - (i === lab ? 1 : 0));
      for (let l = 2; l >= 0; l--) {
        for (let i = 0; i < gW[l].length; i++) {
          gb[l][i] += delta[i];
          for (let j = 0; j < gW[l][i].length; j++) gW[l][i][j] += delta[i] * acts[l][j];
        }
        if (l > 0) {
          const back = acts[l].map((_, j) => n.Ws[l].reduce((s, row, i) => s + row[j] * delta[i], 0));
          delta = back.map((v, j) => (acts[l][j] > 0 ? v : 0));   // ReLU derivative
        }
      }
    }
    const N = data.length;
    for (let l = 0; l < 3; l++) {
      for (let i = 0; i < n.Ws[l].length; i++) {
        n.bs[l][i] -= lr * gb[l][i] / N;
        for (let j = 0; j < n.Ws[l][i].length; j++) n.Ws[l][i][j] -= lr * gW[l][i][j] / N;
      }
    }
    return { loss: loss / N, acc: correct / N };
  }

  /** Loss and accuracy without touching the weights, so epoch 0 reports honestly. */
  function evaluate(n) {
    let loss = 0, correct = 0;
    for (const [px, py, lab] of data) {
      const { o } = forward(n, [px, py]);
      loss -= Math.log(Math.max(1e-12, o[lab]));
      if ((o[1] > o[0] ? 1 : 0) === lab) correct++;
    }
    return { loss: loss / data.length, acc: correct / data.length };
  }

  let net = initNet(11);
  let epoch = 0, running = false, raf = null, lr = .35;
  let stats = evaluate(net);
  let field = null;                   // cached activations over the input grid

  /** One forward pass per grid cell gives every panel at once. */
  function computeField() {
    const a1 = Array.from({ length: 8 }, () => new Float32Array(GRID * GRID));
    const a2 = Array.from({ length: 8 }, () => new Float32Array(GRID * GRID));
    const lg = Array.from({ length: 2 }, () => new Float32Array(GRID * GRID));
    const pr = new Float32Array(GRID * GRID);
    for (let gy = 0; gy < GRID; gy++) {
      const y = R - (2 * R) * (gy / (GRID - 1));
      for (let gx = 0; gx < GRID; gx++) {
        const x = -R + (2 * R) * (gx / (GRID - 1));
        const k = gy * GRID + gx;
        const f = forward(net, [x, y]);
        for (let i = 0; i < 8; i++) { a1[i][k] = f.a1[i]; a2[i][k] = f.a2[i]; }
        lg[0][k] = f.z3[0]; lg[1][k] = f.z3[1];
        pr[k] = f.o[1];
      }
    }
    field = { a1, a2, lg, pr };
  }

  /* ---- panels ---- */
  const panelCv = [];
  function panelGrid(title, count, kind) {
    const host = el('div', { style: 'margin-bottom:.85rem' });
    host.appendChild(el('div', { class: 'matrix-label', html: title }));
    const g = el('div', {
      style: `display:grid;grid-template-columns:repeat(${count === 2 ? 2 : 4},1fr);gap:.4rem`,
    });
    for (let i = 0; i < count; i++) {
      const cv = el('canvas', { style: 'width:100%;aspect-ratio:1;display:block;border-radius:6px' });
      const cell = el('div', { style: 'position:relative' }, cv,
        el('span', {
          style: 'position:absolute;left:4px;top:2px;font-size:.62rem;font-weight:700;' +
                 'color:#fff;text-shadow:0 1px 3px rgba(0,0,0,.65);pointer-events:none',
          text: kind === 'lg' ? `logit ${i + 1}` : `${i + 1}`,
        }));
      g.appendChild(cell);
      panelCv.push({ cv, kind, idx: i });
    }
    host.appendChild(g);
    return host;
  }

  const bigCv = el('canvas');
  const bigHost = el('div', {},
    el('div', { class: 'matrix-label', html: 'Final decision boundary' }),
    el('div', { class: 'pg-canvas-wrap' }, bigCv));

  const h1 = panelGrid('Hidden layer 1 — each neuron is one linear cut (post-ReLU)', 8, 'a1');
  const h2 = panelGrid('Hidden layer 2 — cuts recombined into curved regions', 8, 'a2');
  const lgp = panelGrid('Output logits — one panel per class', 2, 'lg');

  const acts = el('div', { class: 'pg-actions' },
    button('Train 200 epochs', () => { for (let i = 0; i < 200; i++) stats = trainEpoch(net, lr); epoch += 200; refresh(); }),
    button('Run', () => { running = !running; if (running) loop(); }),
    button('Reset', () => { net = initNet(11 + Math.floor(Math.random() * 900)); epoch = 0; stats = evaluate(net); refresh(); }));
  const lrCtl = slider('learning rate', { min: .05, max: 1, step: .05, value: .35, onInput: v => { lr = v; } });
  const out = readout([['epochs', 0], ['loss', 0], ['training accuracy', 0], ['dead ReLU units in layer 1', 0]]);
  const st = status('');

  const left = el('div', {}, bigHost);
  const right = el('div', { class: 'pg-controls' }, acts, lrCtl.root, out.root, st.root);
  wrap.appendChild(el('div', { class: 'pg-split pg-split--wide-ctrl' }, left, right));
  wrap.appendChild(el('div', { style: 'margin-top:1rem' }, h1, h2, lgp));

  const plot = trackPlot(new Plot(bigCv, { xmin: -R, xmax: R, ymin: -R, ymax: R, aspect: 1, equal: true, pad: 0 }));

  function drawPanel(cv, buf, diverging) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = cv.clientWidth || 90;
    if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(w * dpr); }
    const ctx = cv.getContext('2d');
    const img = ctx.createImageData(GRID, GRID);
    let lo = Infinity, hi = -Infinity;
    for (const v of buf) { if (v < lo) lo = v; if (v > hi) hi = v; }
    const span = Math.max(hi - lo, 1e-6);
    const c1 = hexRGB(C.c1), c2 = hexRGB(C.c2), bgc = hexRGB(C.bg);
    for (let k = 0; k < buf.length; k++) {
      let r2, g2, b2;
      if (diverging) {
        const t = (buf[k] - lo) / span * 2 - 1;              // −1 … 1
        const c = t >= 0 ? c2 : c1;
        const a = Math.min(1, Math.abs(t));
        r2 = bgc[0] + (c[0] - bgc[0]) * a;
        g2 = bgc[1] + (c[1] - bgc[1]) * a;
        b2 = bgc[2] + (c[2] - bgc[2]) * a;
      } else {
        const a = (buf[k] - lo) / span;
        r2 = bgc[0] + (c1[0] - bgc[0]) * a;
        g2 = bgc[1] + (c1[1] - bgc[1]) * a;
        b2 = bgc[2] + (c1[2] - bgc[2]) * a;
      }
      img.data[k * 4] = r2; img.data[k * 4 + 1] = g2; img.data[k * 4 + 2] = b2; img.data[k * 4 + 3] = 255;
    }
    const tmp = document.createElement('canvas');
    tmp.width = GRID; tmp.height = GRID;
    tmp.getContext('2d').putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(tmp, 0, 0, cv.width, cv.height);
  }
  function hexRGB(hex) {
    hex = (hex || '#000').trim();
    if (!hex.startsWith('#')) return [128, 128, 128];
    return hex.length === 4
      ? hex.slice(1).split('').map(c => parseInt(c + c, 16))
      : [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));
  }

  function loop() {
    if (!running) return;
    for (let i = 0; i < 12; i++) stats = trainEpoch(net, lr);
    epoch += 12;
    refresh();
    raf = requestAnimationFrame(loop);
  }
  window.addEventListener('pagehide', () => { running = false; cancelAnimationFrame(raf); });

  function refresh() {
    computeField();
    // a unit is dead if it never activates anywhere on the input square
    const dead = field.a1.filter(b => b.every(v => v <= 1e-9)).length;
    out.set([
      String(epoch),
      fmt(stats.loss, 4),
      { html: `${fmt(stats.acc * 100, 1)}%`, cls: stats.acc > .95 ? 'is-ok' : '' },
      { html: `${dead} of 8`, cls: dead > 2 ? 'is-warn' : 'is-ok' },
    ]);
    st.set(
      epoch === 0
        ? `${INFO}<span>Untrained. The layer-1 panels are already straight cuts — that is all a single ReLU unit can ever be — but they are oriented at random.</span>`
        : stats.acc > .95
          ? `${OK}<span><strong>XOR solved.</strong> Look at the three rows: layer 1 holds eight straight cuts, layer 2 recombines them into curved regions, and the two logits are now linearly separable so the final readout is trivial.</span>`
          : `${INFO}<span>Training — accuracy ${fmt(stats.acc * 100, 1)}%. Watch the layer-1 cuts rotate to line up with the quadrant boundaries.</span>`,
      stats.acc > .95 ? 'ok' : 'info');
    panelCv.forEach(({ cv, kind, idx }) => {
      const buf = kind === 'a1' ? field.a1[idx] : kind === 'a2' ? field.a2[idx] : field.lg[idx];
      drawPanel(cv, buf, kind === 'lg');
    });
    plot.render();
  }

  plot.onDraw(p => {
    // decision surface from the cached probabilities
    const cw = p.w, ch = p.h;
    const tmp = document.createElement('canvas');
    tmp.width = GRID; tmp.height = GRID;
    const img = tmp.getContext('2d').createImageData(GRID, GRID);
    const c1 = hexRGB(C.c1), c2 = hexRGB(C.c2), bgc = hexRGB(C.bg);
    for (let k = 0; k < field.pr.length; k++) {
      const t = field.pr[k] * 2 - 1;
      const c = t >= 0 ? c2 : c1;
      const a = Math.min(1, Math.abs(t)) * .62;
      img.data[k * 4] = bgc[0] + (c[0] - bgc[0]) * a;
      img.data[k * 4 + 1] = bgc[1] + (c[1] - bgc[1]) * a;
      img.data[k * 4 + 2] = bgc[2] + (c[2] - bgc[2]) * a;
      img.data[k * 4 + 3] = 255;
    }
    tmp.getContext('2d').putImageData(img, 0, 0);
    p.ctx.imageSmoothingEnabled = true;
    p.ctx.drawImage(tmp, p.X(-R), p.Y(R), cw - 2 * (p.X(-R)), ch - 2 * (p.Y(R)));
    data.forEach(([x, y, lab]) => p.dot([x, y], { r: 3.4, color: lab === 0 ? C.c1 : C.c2 }));
    p.axes(); p.ticks(1);
    p.legend([[C.c1, 'class 0 — same sign'], [C.c2, 'class 1 — opposite sign']],
      { corner: 'tl', title: `${epoch} epochs · accuracy ${fmt(stats.acc * 100, 0)}%` });
  });

  refresh();

  node.appendChild(note(
    `This is the full XOR run-through from the notes, rebuilt as a live network. XOR is the standard example ` +
    `of a problem no linear model can touch, and the three rows of panels show exactly how a network gets ` +
    `around that. Every unit in <strong>layer 1</strong> computes ReLU(w·x + b), so its activation map can ` +
    `only ever be a half-plane — a single straight cut, bright on one side and flat zero on the other. ` +
    `<strong>Layer 2</strong> takes weighted combinations of those eight half-planes, and the maps become ` +
    `selective, cornered, curved. By the <strong>logit</strong> row the two classes have been pulled apart ` +
    `far enough that a plain linear readout separates them. Nonlinearity is not in any single unit; it is in ` +
    `the composition. Watch the dead-unit counter too — ReLU units that start pointing the wrong way can ` +
    `output zero everywhere and never recover, since their gradient is zero as well.`
  ));
});

/* ============================================================
   How different optimisers travel the loss surface
   ============================================================ */
defineWidget('optimizers', node => {
  const { right, canvas } = split(node, { wide: true });
  const plot = trackPlot(new Plot(canvas, { xmin: -2.2, xmax: 2.2, ymin: -1.4, ymax: 1.4, aspect: 1.5, equal: false, pad: 0 }));

  const SURF = {
    ravine: {
      label: 'Ill-conditioned valley',
      f: (x, y) => .5 * (.06 * x * x + 3.2 * y * y),
      g: (x, y) => [.06 * x, 3.2 * y],
      start: [-1.95, 1.1], lr: .35,
    },
    saddle: {
      label: 'Saddle point',
      f: (x, y) => .32 * (x * x * x / 3 - x) + .9 * y * y,
      g: (x, y) => [.32 * (x * x - 1), 1.8 * y],
      start: [-.05, .95], lr: .25,
    },
    bowl: {
      label: 'Well-conditioned bowl',
      f: (x, y) => .5 * (.9 * x * x + 1.1 * y * y),
      g: (x, y) => [.9 * x, 1.1 * y],
      start: [-1.9, 1.05], lr: .35,
    },
  };
  let key = 'ravine', lrScale = 1, mu = .9, steps = 60, showAll = true;

  const sCtl = segmented(Object.entries(SURF).map(([k, v]) => ({ label: v.label, value: k })),
    { value: 'ravine', label: 'Loss surface', onChange: v => { key = v; refresh(); } });
  const lrCtl = slider('learning-rate multiplier', { min: .2, max: 3, step: .1, value: 1, onInput: v => { lrScale = v; refresh(); } });
  const muCtl = slider('momentum μ', { min: 0, max: .98, step: .02, value: .9, onInput: v => { mu = v; refresh(); } });
  const nCtl = slider('steps', { min: 5, max: 200, step: 5, value: 60, format: v => String(v), onInput: v => { steps = v; refresh(); } });
  const aCtl = toggle('Show all four at once', { value: true, onChange: v => { showAll = v; plot.render(); } });
  const out = readout([['GD — final loss', 0], ['Momentum', 0], ['RMSProp', 0], ['Adam', 0], ['fastest to 1% of optimum', 0]]);
  const st = status('');
  right.append(sCtl.root, lrCtl.root, muCtl.root, nCtl.root, aCtl.root, out.root, st.root);

  /* Each optimiser, written exactly as the notes give it. */
  function run(kind) {
    const S = SURF[key];
    const eta = S.lr * lrScale;
    let [x, y] = S.start;
    let vx = 0, vy = 0, sx = 0, sy = 0, mx = 0, my = 0;
    const path = [[x, y]];
    const eps = 1e-8, b1 = .9, b2 = .999;
    for (let t = 1; t <= steps; t++) {
      const [gx, gy] = S.g(x, y);
      if (kind === 'gd') {
        x -= eta * gx; y -= eta * gy;
      } else if (kind === 'mom') {
        vx = mu * vx - eta * gx; vy = mu * vy - eta * gy;
        x += vx; y += vy;
      } else if (kind === 'rms') {
        sx = .9 * sx + .1 * gx * gx; sy = .9 * sy + .1 * gy * gy;
        x -= eta * gx / (Math.sqrt(sx) + eps); y -= eta * gy / (Math.sqrt(sy) + eps);
      } else {
        mx = b1 * mx + (1 - b1) * gx; my = b1 * my + (1 - b1) * gy;
        sx = b2 * sx + (1 - b2) * gx * gx; sy = b2 * sy + (1 - b2) * gy * gy;
        const mhx = mx / (1 - b1 ** t), mhy = my / (1 - b1 ** t);
        const shx = sx / (1 - b2 ** t), shy = sy / (1 - b2 ** t);
        x -= eta * mhx / (Math.sqrt(shx) + eps); y -= eta * mhy / (Math.sqrt(shy) + eps);
      }
      if (!Number.isFinite(x) || !Number.isFinite(y)) break;
      x = clamp(x, -8, 8); y = clamp(y, -8, 8);
      path.push([x, y]);
    }
    return path;
  }

  const KINDS = [['gd', 'GD', () => C.c4], ['mom', 'Momentum', () => C.c1],
                 ['rms', 'RMSProp', () => C.c3], ['adam', 'Adam', () => C.c2]];
  let paths = {};
  function refresh() {
    const S = SURF[key];
    paths = {};
    KINDS.forEach(([k]) => { paths[k] = run(k); });
    const fin = k => { const p = paths[k][paths[k].length - 1]; return S.f(p[0], p[1]); };
    const best = Math.min(...KINDS.map(([k]) => fin(k)));
    // first step within 1% of the best value reached
    const reach = k => {
      const target = best + .01 * Math.abs(best) + 1e-3;
      const i = paths[k].findIndex(p => S.f(p[0], p[1]) <= target);
      return i < 0 ? null : i;
    };
    const times = KINDS.map(([k, lbl]) => [lbl, reach(k)]).filter(t => t[1] !== null);
    times.sort((a, b) => a[1] - b[1]);
    out.set([
      ...KINDS.map(([k]) => ({ html: fmt(fin(k), 5), cls: fin(k) <= best + 1e-9 ? 'is-ok' : '' })),
      times.length ? `${times[0][0]} (${times[0][1]} steps)` : '—',
    ]);
    st.set(
      key === 'ravine'
        ? `${INFO}<span>This valley is <strong>53× steeper across than along</strong>. Plain GD has one step size for both directions, so it must stay small enough not to diverge across the valley — and then crawls along it. Momentum accumulates the consistent along-valley gradient while the oscillating across-valley components cancel. The adaptive methods rescale each coordinate by its own gradient history, which fixes the conditioning directly.</span>`
        : key === 'saddle'
          ? `${INFO}<span>Near a saddle the gradient is tiny but not zero. GD stalls; the methods carrying velocity or per-coordinate scaling escape much sooner.</span>`
          : `${OK}<span>On a well-conditioned bowl everything works and plain GD is competitive. Adaptive optimisers earn their keep on badly scaled problems, not easy ones.</span>`,
      key === 'bowl' ? 'ok' : 'info');
    plot.render();
  }

  plot.onDraw(p => {
    const S = SURF[key];
    p.o.xmin = -2.2; p.o.xmax = 2.2; p.o.ymin = -1.4; p.o.ymax = 1.4;
    p._computeScale();
    // contours
    const levels = [];
    for (let i = 1; i <= 9; i++) levels.push((i / 9) ** 2 * S.f(-2.1, 1.3));
    const N = 90;
    for (const L of levels) {
      p.ctx.beginPath();
      let started = false;
      for (let gy = 0; gy <= N; gy++) {
        for (let gx = 0; gx <= N; gx++) {
          const x = -2.2 + 4.4 * gx / N, y = -1.4 + 2.8 * gy / N;
          if (Math.abs(S.f(x, y) - L) < L * .045 + .006) {
            const [sx2, sy2] = p.toScreen([x, y]);
            if (!started) { p.ctx.moveTo(sx2, sy2); started = true; }
            p.ctx.rect(sx2 - .8, sy2 - .8, 1.6, 1.6);
          }
        }
      }
      p.ctx.fillStyle = withA(C.muted, .30);
      p.ctx.fill();
    }
    p.grid(.5, { color: C.grid });
    p.axes(); p.ticks(.5);
    const shown = showAll ? KINDS : [KINDS[0]];
    shown.forEach(([k, lbl, col]) => {
      const path = paths[k];
      p.path(path, { color: col(), lw: 2.4 });
      path.forEach((q, i) => { if (i % Math.max(1, Math.round(steps / 30)) === 0) p.dot(q, { r: 2.4, color: col() }); });
      p.dot(path[path.length - 1], { r: 5.5, color: col() });
    });
    p.dot([0, 0], { r: 5, color: C.ink });
    p.badge([0, 0], 'optimum', { color: C.ink, align: 'center', dy: 16 });
    p.legend(shown.map(([k, lbl, col]) => [col(), lbl]), { corner: 'tr', title: `${steps} steps from the same start` });
  });

  refresh();

  node.appendChild(note(
    `All four updates are exactly as written in the notes. <strong>GD</strong> takes ` +
    `<span class="u-mono">θ ← θ − η∇L</span>. <strong>Momentum</strong> carries a velocity ` +
    `<span class="u-mono">v_t = μv_{t−1} − η∇L</span>. <strong>RMSProp</strong> divides by a running root-mean-square ` +
    `of the gradient, and <strong>Adam</strong> combines that with a momentum term plus bias correction ` +
    `<span class="u-mono">m̂ = m/(1−β₁ᵗ)</span>, which matters most in the first few steps when the running ` +
    `averages start at zero. Push the learning-rate multiplier up and watch GD blow up across the valley long ` +
    `before the adaptive methods are troubled — that difference in usable step size is most of why Adam is the ` +
    `default.`
  ));
});
