/* ============================================================
   widgets/prob.js — interactive figures for Probability & Statistics
   ============================================================ */
import {
  Plot, Dragger, C, el, slider, toggle, segmented, button,
  matrixInput, readout, status, defineWidget, canvasHost,
  trackPlot, clamp, fmt, round,
} from '../viz.js';

function split(node, { aspect = 1.5, hint, wide = false } = {}) {
  const left = el('div');
  const right = el('div', { class: 'pg-controls' });
  node.appendChild(el('div', { class: 'pg-split' + (wide ? ' pg-split--wide-ctrl' : '') }, left, right));
  const { canvas } = canvasHost(left, { hint });
  return { left, right, canvas, aspect };
}
const note = html => el('div', { class: 'pg-note', html });

/* ---------- deterministic RNG so figures are reproducible ---------- */
function makeRng(seed = 20250913) {
  let s = seed >>> 0;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5;  s >>>= 0;
    return s / 4294967296;
  };
}
const gaussPair = rng => {
  const u = Math.max(1e-12, rng()), v = rng();
  const r = Math.sqrt(-2 * Math.log(u));
  return [r * Math.cos(2 * Math.PI * v), r * Math.sin(2 * Math.PI * v)];
};

/* ---------- distribution definitions ---------- */
const lgamma = x => {
  // Lanczos approximation
  const g = [676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012,
    9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  x -= 1;
  let a = 0.99999999999980993;
  const t = x + 7.5;
  for (let i = 0; i < 8; i++) a += g[i] / (x + i + 1);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
};
const logChoose = (n, k) => lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);

const DISTS = {
  bernoulli: {
    name: 'Bernoulli(p)', discrete: true,
    params: [{ key: 'p', label: 'p', min: 0, max: 1, step: .01, value: .3 }],
    support: () => [0, 1],
    pmf: (k, { p }) => (k === 0 ? 1 - p : k === 1 ? p : 0),
    mean: ({ p }) => p, varr: ({ p }) => p * (1 - p),
    blurb: 'One yes/no trial. The atom every count distribution is built from.',
  },
  binomial: {
    name: 'Binomial(n, p)', discrete: true,
    params: [
      { key: 'n', label: 'n', min: 1, max: 40, step: 1, value: 12 },
      { key: 'p', label: 'p', min: 0, max: 1, step: .01, value: .4 },
    ],
    support: ({ n }) => [0, n],
    pmf: (k, { n, p }) => {
      if (k < 0 || k > n) return 0;
      if (p <= 0) return k === 0 ? 1 : 0;
      if (p >= 1) return k === n ? 1 : 0;
      return Math.exp(logChoose(n, k) + k * Math.log(p) + (n - k) * Math.log(1 - p));
    },
    mean: ({ n, p }) => n * p, varr: ({ n, p }) => n * p * (1 - p),
    blurb: 'How many successes in n independent trials. Defect counts, click-through counts, heads in n flips.',
  },
  poisson: {
    name: 'Poisson(λ)', discrete: true,
    params: [{ key: 'lam', label: 'λ', min: .1, max: 20, step: .1, value: 4 }],
    support: ({ lam }) => [0, Math.max(12, Math.ceil(lam + 4 * Math.sqrt(lam)))],
    pmf: (k, { lam }) => k < 0 ? 0 : Math.exp(-lam + k * Math.log(lam) - lgamma(k + 1)),
    mean: ({ lam }) => lam, varr: ({ lam }) => lam,
    blurb: 'Counts of rare events in a fixed window — arrivals, photon hits, typos per page. Mean and variance coincide.',
  },
  uniform: {
    name: 'Uniform(a, b)', discrete: false,
    params: [
      { key: 'a', label: 'a', min: -5, max: 4, step: .1, value: -1 },
      { key: 'b', label: 'b', min: -4, max: 6, step: .1, value: 3 },
    ],
    range: ({ a, b }) => [Math.min(a, b) - 1.6, Math.max(a, b) + 1.6],
    pdf: (x, { a, b }) => (b <= a ? 0 : (x >= a && x <= b ? 1 / (b - a) : 0)),
    mean: ({ a, b }) => (a + b) / 2, varr: ({ a, b }) => (b - a) ** 2 / 12,
    blurb: 'Every value in the interval equally likely. The default "I know nothing beyond the range" model.',
  },
  normal: {
    name: 'Normal(μ, σ²)', discrete: false,
    params: [
      { key: 'mu', label: 'μ', min: -4, max: 4, step: .05, value: 0 },
      { key: 'sd', label: 'σ', min: .15, max: 3, step: .05, value: 1 },
    ],
    range: ({ mu, sd }) => [mu - 4.4 * sd, mu + 4.4 * sd],
    pdf: (x, { mu, sd }) => Math.exp(-((x - mu) ** 2) / (2 * sd * sd)) / (sd * Math.sqrt(2 * Math.PI)),
    mean: ({ mu }) => mu, varr: ({ sd }) => sd * sd,
    blurb: 'The bell curve. Measurement noise, aggregates of many small effects — and the shape the CLT keeps producing.',
  },
  exponential: {
    name: 'Exponential(λ)', discrete: false,
    params: [{ key: 'lam', label: 'λ', min: .15, max: 4, step: .05, value: 1 }],
    range: ({ lam }) => [-0.6, 7 / lam],
    pdf: (x, { lam }) => (x < 0 ? 0 : lam * Math.exp(-lam * x)),
    mean: ({ lam }) => 1 / lam, varr: ({ lam }) => 1 / (lam * lam),
    blurb: 'Waiting time until the next Poisson event. Memoryless: having waited already tells you nothing.',
  },
};

/* ============================================================
   1. Distribution explorer — pdf/pmf and cdf side by side
   ============================================================ */
defineWidget('distributions', node => {
  let key = 'binomial';
  let P = {};

  const { right, canvas } = split(node, { aspect: 1.27, wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -1, xmax: 13, ymin: 0, ymax: 1, aspect: 1.27, equal: false, pad: 0,
  }));

  const pick = segmented(
    Object.entries(DISTS).map(([k, d]) => ({ label: d.name.split('(')[0], value: k })),
    { value: key, label: 'Distribution', onChange: v => { key = v; buildParams(); refresh(); } }
  );
  const paramHost = el('div', { style: 'display:flex;flex-direction:column;gap:.85rem' });
  const showCdf = toggle('Overlay the CDF', { value: true, onChange: () => plot.render() });
  const out = readout([['distribution', 0], ['E[X]', 0], ['Var(X)', 0], ['SD(X)', 0]]);
  const blurb = el('div', { style: 'font-size:.86rem;color:var(--ink-muted);line-height:1.55' });
  right.append(pick.root, paramHost, showCdf.root, out.root, blurb);

  let ctls = [];
  function buildParams() {
    paramHost.innerHTML = ''; ctls = []; P = {};
    DISTS[key].params.forEach(sp => {
      P[sp.key] = sp.value;
      const c = slider(sp.label, {
        min: sp.min, max: sp.max, step: sp.step, value: sp.value,
        format: v => fmt(v, sp.step >= 1 ? 0 : 2),
        onInput: v => { P[sp.key] = v; refresh(); },
      });
      ctls.push(c); paramHost.appendChild(c.root);
    });
  }

  function refresh() {
    const d = DISTS[key];
    if (d.discrete) {
      const [lo, hi] = d.support(P);
      plot.setBounds({ xmin: lo - .8, xmax: hi + .8, ymin: 0, ymax: 1 });
    } else {
      const [lo, hi] = d.range(P);
      plot.setBounds({ xmin: lo, xmax: hi, ymin: 0, ymax: 1 });
    }
    const m = d.mean(P), v = d.varr(P);
    out.set([d.name, fmt(m, 3), fmt(v, 3), fmt(Math.sqrt(Math.max(0, v)), 3)]);
    blurb.innerHTML = d.blurb;
    plot.render();
  }

  plot.onDraw(p => {
    const d = DISTS[key];
    p.grid(1, { color: C.grid });

    if (d.discrete) {
      const [lo, hi] = d.support(P);
      const vals = [];
      for (let k = lo; k <= hi; k++) vals.push([k, d.pmf(k, P)]);
      const mx = Math.max(...vals.map(v => v[1]), 1e-9);
      const bw = Math.min(p.px(.72), 34);
      let cum = 0;
      const cumPts = [[lo - .8, 0]];
      vals.forEach(([k, pr]) => {
        const h = pr / mx * .92;
        const x = p.X(k), y0 = p.Y(0), y1 = p.Y(h);
        p.ctx.fillStyle = C.c1;
        p.ctx.globalAlpha = .88;
        p.ctx.fillRect(x - bw / 2, y1, bw, y0 - y1);
        p.ctx.globalAlpha = 1;
        if (bw > 17 && pr > .012) {
          p.text([k, h], fmt(pr, 3), { align: 'center', dy: -9, size: 10, color: C.muted });
        }
        cum += pr; cumPts.push([k, cum]);
      });
      cumPts.push([hi + .8, cum]);
      if (showCdf.get()) {
        p.path(cumPts, { color: C.c2, lw: 2.2 });
        p.text({ px: p.w - 10, py: 16 }, 'CDF', { color: C.c2, size: 11, align: 'right' });
      }
      p.axes();
      p.ticks(Math.max(1, Math.round((hi - lo) / 12)));
      // mean marker
      const m = d.mean(P);
      p.line([m, 0], [m, 1], { color: C.c4, lw: 1.6, dash: [5, 4] });
      // sits on the mean's rule, clear of the y tick column at the left
      p.badge([m, .97], `E[X] = ${fmt(m, 2)}`, { color: C.c4, align: 'center', dy: 26 });
    } else {
      const [lo, hi] = d.range(P);
      let mx = 1e-9;
      for (let i = 0; i <= 400; i++) mx = Math.max(mx, d.pdf(lo + (hi - lo) * i / 400, P));
      const pts = [];
      for (let i = 0; i <= 400; i++) {
        const x = lo + (hi - lo) * i / 400;
        pts.push([x, d.pdf(x, P) / mx * .92]);
      }
      p.path([[lo, 0], ...pts, [hi, 0]], { fill: C.fill, color: C.c1, lw: 2.6 });

      if (showCdf.get()) {
        const cum = []; let acc = 0;
        const dx = (hi - lo) / 400;
        for (let i = 0; i <= 400; i++) {
          const x = lo + dx * i;
          acc += d.pdf(x, P) * dx;
          cum.push([x, Math.min(1, acc)]);
        }
        p.path(cum, { color: C.c2, lw: 2.2 });
        p.text({ px: p.w - 10, py: 16 }, 'CDF', { color: C.c2, size: 11, align: 'right' });
      }
      p.axes(); p.ticks(Math.max(.5, round((hi - lo) / 8, 1)));

      const m = d.mean(P), sd = Math.sqrt(Math.max(0, d.varr(P)));
      p.line([m, 0], [m, 1], { color: C.c4, lw: 1.6, dash: [5, 4] });
      p.badge([m, .97], `μ = ${fmt(m, 2)}`, { color: C.c4, align: 'center' });
      if (key === 'normal') {
        [1, 2].forEach(k => {
          [-1, 1].forEach(sgn => {
            p.line([m + sgn * k * sd, 0], [m + sgn * k * sd, .55], { color: C.c3, lw: 1.1, dash: [3, 4], alpha: .7 });
          });
        });
        p.text([m + sd, .58], '±1σ ≈ 68%', { color: C.c3, size: 10.5, align: 'center' });
        p.text([m + 2 * sd, .38], '±2σ ≈ 95%', { color: C.c3, size: 10.5, align: 'center' });
      }
    }
  });

  buildParams(); refresh();

  node.appendChild(note(
    `Discrete distributions place <strong>lumps of probability</strong> on individual values (heights are ` +
    `genuine probabilities, and they sum to 1). Continuous ones spread a <strong>density</strong> — height ` +
    `is not probability, <em>area</em> is, which is why ℙ(X = x) = 0 for any single point. In both cases the ` +
    `CDF climbs from 0 to 1 and never comes back down.`
  ));
});

/* ============================================================
   2. Central Limit Theorem
   ============================================================ */
defineWidget('clt', node => {
  const { right, canvas } = split(node, { aspect: 1.23, wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -1, xmax: 1, ymin: 0, ymax: 1.12, aspect: 1.23, equal: false, pad: 0,
  }));

  const parents = {
    uniform:  { name: 'Uniform', draw: r => r(), mean: .5, varr: 1 / 12 },
    exponential: { name: 'Exponential', draw: r => -Math.log(Math.max(1e-12, r())), mean: 1, varr: 1 },
    bimodal:  { name: 'Bimodal', draw: r => (r() < .5 ? .12 + .1 * r() : .78 + .1 * r()), mean: .5, varr: 0 },
    bernoulli:{ name: 'Coin flip', draw: r => (r() < .35 ? 1 : 0), mean: .35, varr: .35 * .65 },
  };
  let pkey = 'exponential';
  let n = 1, trials = 4000;
  let rng = makeRng();
  let means = [], parentSample = [];

  const pick = segmented(
    Object.entries(parents).map(([k, v]) => ({ label: v.name, value: k })),
    { value: pkey, label: 'Parent distribution', onChange: v => { pkey = v; resample(); } }
  );
  const nCtl = slider('Sample size n', {
    min: 1, max: 80, step: 1, value: 1, format: v => String(v),
    onInput: v => { n = v; resample(); },
  });
  const acts = el('div', { class: 'pg-actions' },
    button('Draw again', () => { rng = makeRng((Math.random() * 1e9) | 0); resample(); }),
    button('n = 1', () => { n = 1; nCtl.set(1); resample(); }),
    button('n = 30', () => { n = 30; nCtl.set(30); resample(); }),
  );
  const showNormal = toggle('Overlay the normal fit', { value: true, onChange: () => plot.render() });
  const out = readout([['parent shape', 0], ['n', 0], ['mean of X̄', 0], ['SD of X̄', 0], ['σ/√n (theory)', 0]]);
  const st = status('');
  right.append(pick.root, nCtl.root, acts, showNormal.root, out.root, st.root);

  function empiricalVar(key) {
    // measured once so "Bimodal" gets a sensible theoretical SD too
    const r = makeRng(777);
    let s = 0, s2 = 0;
    for (let i = 0; i < 20000; i++) { const v = parents[key].draw(r); s += v; s2 += v * v; }
    const m = s / 20000;
    return { mean: m, varr: s2 / 20000 - m * m };
  }

  function resample() {
    const P = parents[pkey];
    means = []; parentSample = [];
    const r = rng;
    for (let t = 0; t < trials; t++) {
      let acc = 0;
      for (let i = 0; i < n; i++) {
        const v = P.draw(r);
        acc += v;
        if (t < 4000 && i === 0) parentSample.push(v);
      }
      means.push(acc / n);
    }
    const stats = empiricalVar(pkey);
    const m = means.reduce((a, b) => a + b, 0) / means.length;
    const sd = Math.sqrt(means.reduce((a, b) => a + (b - m) ** 2, 0) / means.length);
    out.set([
      P.name, String(n), fmt(m, 4), fmt(sd, 4),
      fmt(Math.sqrt(stats.varr / n), 4),
    ]);
    if (n === 1) {
      st.set(`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16.5v.01" stroke-linecap="round"/></svg>` +
        `<span>At n = 1 you are just looking at the parent distribution itself. Now raise n.</span>`, 'info');
    } else if (n >= 25) {
      st.set(`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>` +
        `<span><strong>Bell-shaped</strong>, whatever the parent looked like — and the spread has shrunk by a factor of √${n} ≈ ${fmt(Math.sqrt(n), 2)}.</span>`, 'ok');
    } else {
      st.set(`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16.5v.01" stroke-linecap="round"/></svg>` +
        `<span>Already narrowing and symmetrising. Keep going — by n ≈ 30 the shape is close to normal.</span>`, '');
    }
    plot.render();
  }

  function histogram(data, bins, lo, hi) {
    const h = new Array(bins).fill(0);
    data.forEach(v => {
      const b = Math.floor((v - lo) / (hi - lo) * bins);
      if (b >= 0 && b < bins) h[b]++;
    });
    return h;
  }

  plot.onDraw(p => {
    if (!means.length) return;
    const lo = Math.min(...means), hi = Math.max(...means);
    const padr = (hi - lo) * .18 + 1e-6;
    const L = lo - padr, H = hi + padr;
    // rescale in place: setBounds() would re-enter render()
    p.o.xmin = L; p.o.xmax = H; p.o.ymin = 0; p.o.ymax = 1.12;
    p._computeScale();

    p.grid((H - L) / 6, { color: C.grid });

    const bins = 46;
    const h = histogram(means, bins, L, H);
    const mx = Math.max(...h, 1);
    const bw = p.px((H - L) / bins);
    h.forEach((c, i) => {
      const x0 = L + (H - L) * i / bins;
      const hh = c / mx * .95;
      p.ctx.fillStyle = C.c1; p.ctx.globalAlpha = .82;
      p.ctx.fillRect(p.X(x0), p.Y(hh), Math.max(1, bw - .8), p.Y(0) - p.Y(hh));
      p.ctx.globalAlpha = 1;
    });

    if (showNormal.get()) {
      const m = means.reduce((a, b) => a + b, 0) / means.length;
      const sd = Math.sqrt(means.reduce((a, b) => a + (b - m) ** 2, 0) / means.length) || 1e-6;
      const peak = 1 / (sd * Math.sqrt(2 * Math.PI));
      p.fn(x => Math.exp(-((x - m) ** 2) / (2 * sd * sd)) / (sd * Math.sqrt(2 * Math.PI)) / peak * .95,
        { color: C.c2, lw: 2.6, from: L, to: H });
    }

    p.axes(); p.ticks(round((H - L) / 5, 2));
    p.text({ px: 12, py: 16 },
      `distribution of the sample mean  (n = ${n}, ${trials} repeats)`,
      { color: C.muted, size: 11 });
  });

  resample();

  node.appendChild(note(
    `Pick the <strong>most lopsided parent you can</strong> — the exponential, or the two-humped one — then ` +
    `raise n. The histogram of sample <em>means</em> turns into a bell curve regardless. This is why so much ` +
    `of statistics can assume normality: we almost always work with averages, and averages forget the shape ` +
    `they came from. Note also that the spread shrinks like <span class="u-mono">σ/√n</span>, not σ/n — ` +
    `quadrupling your data only halves your error.`
  ));
});

/* ============================================================
   3. Law of Large Numbers
   ============================================================ */
defineWidget('lln', node => {
  const { right, canvas } = split(node, { aspect: 1.56 });
  const plot = trackPlot(new Plot(canvas, {
    xmin: 0, xmax: 500, ymin: 0, ymax: 1, aspect: 1.56, equal: false, pad: 0,
  }));

  let p0 = .5, N = 500, eps = .08;
  let paths = [];
  let rng = makeRng(4242);

  const pCtl = slider('True mean p', {
    min: .05, max: .95, step: .01, value: .5,
    onInput: v => { p0 = v; regen(); },
  });
  const nCtl = slider('Number of tosses', {
    min: 50, max: 4000, step: 50, value: 500, format: v => String(v),
    onInput: v => { N = v; regen(); },
  });
  const eCtl = slider('Tolerance ε', {
    min: .01, max: .3, step: .005, value: .08,
    onInput: v => { eps = v; plot.render(); sync(); },
  });
  const acts = el('div', { class: 'pg-actions' },
    button('New runs', () => { rng = makeRng((Math.random() * 1e9) | 0); regen(); })
  );
  const out = readout([['runs shown', 0], ['final spread', 0], ['inside ±ε at n', 0]]);
  right.append(pCtl.root, nCtl.root, eCtl.root, acts, out.root);

  function regen() {
    paths = [];
    for (let k = 0; k < 6; k++) {
      const path = []; let s = 0;
      for (let i = 1; i <= N; i++) {
        s += rng() < p0 ? 1 : 0;
        if (i % Math.max(1, Math.floor(N / 600)) === 0 || i < 40) path.push([i, s / i]);
      }
      paths.push(path);
    }
    plot.setBounds({ xmin: 0, xmax: N, ymin: 0, ymax: 1 });
    sync();
  }
  function sync() {
    const finals = paths.map(p => p[p.length - 1][1]);
    const inside = finals.filter(v => Math.abs(v - p0) <= eps).length;
    out.set([
      String(paths.length),
      fmt(Math.max(...finals) - Math.min(...finals), 4),
      `${inside} / ${paths.length}`,
    ]);
    plot.render();
  }

  plot.onDraw(p => {
    p.grid(.25, { color: C.grid });
    // tolerance band
    p.ctx.fillStyle = C.fill2;
    p.ctx.fillRect(p.X(0), p.Y(p0 + eps), p.px(N), p.Y(p0 - eps) - p.Y(p0 + eps));
    p.line([0, p0], [N, p0], { color: C.c4, lw: 2, dash: [6, 4] });

    const cols = [C.c1, C.c3, C.c5, C.c6, C.c2, C.muted];
    paths.forEach((path, i) => p.path(path, { color: cols[i % cols.length], lw: 1.5, alpha: .85 }));

    p.axes(); p.ticks(.25);
    p.badge([N * .35, p0], `p = ${fmt(p0, 2)}`, { color: C.c4, align: 'center', dy: -13 });
    p.text({ px: 12, py: 11 }, 'running average of coin tosses', { color: C.muted, size: 11 });
  });

  regen();

  node.appendChild(note(
    `Each wiggly line is one experiment's running average. Early on they swing wildly; as n grows they are ` +
    `squeezed into the shaded ±ε band around the true p and stay there. That is the Law of Large Numbers — ` +
    `<strong>not</strong> a promise that the average is ever exactly p, but that the probability of missing ` +
    `by more than ε goes to zero. Narrow ε and you simply need more tosses.`
  ));
});

/* ============================================================
   4. Bayes' theorem — the base-rate trap, drawn
   ============================================================ */
defineWidget('bayes', node => {
  const wrap = el('div');
  node.appendChild(wrap);

  let prev = .01, sens = .95, fpr = .05;
  const POP = 1000;

  const cv = el('canvas');
  const left = el('div', {}, el('div', { class: 'pg-canvas-wrap' }, cv));
  const right = el('div', { class: 'pg-controls' });
  wrap.appendChild(el('div', { class: 'pg-split pg-split--wide-ctrl' }, left, right));

  const prevCtl = slider('Prevalence &nbsp;ℙ(D)', {
    min: .001, max: .5, step: .001, value: prev,
    format: v => `${fmt(v * 100, 1)}%`, onInput: v => { prev = v; refresh(); },
  });
  const sensCtl = slider('Sensitivity &nbsp;ℙ(+ | D)', {
    min: .5, max: 1, step: .005, value: sens,
    format: v => `${fmt(v * 100, 1)}%`, onInput: v => { sens = v; refresh(); },
  });
  const fprCtl = slider('False positive rate &nbsp;ℙ(+ | not D)', {
    min: 0, max: .3, step: .002, value: fpr,
    format: v => `${fmt(v * 100, 1)}%`, onInput: v => { fpr = v; refresh(); },
  });
  const out = readout([
    ['ℙ(+)', 0], ['true positives', 0], ['false positives', 0],
    ['ℙ(D | +) posterior', 0], ['ℙ(D) prior', 0],
  ]);
  const st = status('');
  right.append(prevCtl.root, sensCtl.root, fprCtl.root, out.root, st.root);

  const plot = trackPlot(new Plot(cv, { xmin: 0, xmax: 40, ymin: 0, ymax: 25, aspect: 1.31, pad: 6 }));

  function stats() {
    const pPos = sens * prev + fpr * (1 - prev);
    const post = pPos > 0 ? sens * prev / pPos : 0;
    return { pPos, post, tp: sens * prev * POP, fp: fpr * (1 - prev) * POP };
  }

  function refresh() {
    const { pPos, post, tp, fp } = stats();
    out.set([
      `${fmt(pPos * 100, 2)}%`,
      `${fmt(tp, 1)} of ${POP}`,
      `${fmt(fp, 1)} of ${POP}`,
      { html: `${fmt(post * 100, 1)}%`, cls: post < .5 ? 'is-warn' : 'is-ok' },
      `${fmt(prev * 100, 2)}%`,
    ]);
    const ratio = post / prev;
    st.set(
      post < .5
        ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 9v4M12 17v.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/></svg>` +
          `<span>A positive test still leaves you <strong>more likely healthy than sick</strong> (${fmt(post * 100, 1)}%) — because healthy people vastly outnumber sick ones, their few false positives swamp the true ones.</span>`
        : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>` +
          `<span>A positive test now means <strong>${fmt(post * 100, 1)}%</strong> chance of disease — the evidence multiplied your prior by ${fmt(ratio, 1)}×.</span>`,
      post < .5 ? 'warn' : 'ok');
    plot.render();
  }

  plot.onDraw(p => {
    // 1000 people as a 40 x 25 icon array
    const { tp, fp } = stats();
    const sick = Math.round(prev * POP);
    const tpN = Math.round(tp), fpN = Math.round(fp);
    const cell = Math.min(p.px(1), 22);
    const r = cell * .3;

    let idx = 0;
    for (let row = 0; row < 25; row++) {
      for (let col = 0; col < 40; col++, idx++) {
        const isSick = idx < sick;
        const isTP = idx < tpN;
        const isFP = idx >= sick && idx < sick + fpN;
        let color = C.grid, alpha = 1;
        if (isTP)        color = C.c4;
        else if (isSick) color = C.c2;
        else if (isFP)   color = C.c1;
        else             { color = C.muted; alpha = .28; }
        p.ctx.globalAlpha = alpha;
        p.ctx.fillStyle = color;
        p.ctx.beginPath();
        p.ctx.arc(p.X(col + .5), p.Y(24.5 - row), r, 0, Math.PI * 2);
        p.ctx.fill();
      }
    }
    p.ctx.globalAlpha = 1;

    const legend = [
      [C.c4, `sick, tests + (${tpN})`],
      [C.c2, `sick, tests − (${Math.max(0, sick - tpN)})`],
      [C.c1, `healthy, tests + (${fpN})`],
      [C.muted, 'healthy, tests −'],
    ];
    legend.forEach(([col, label], i) => {
      const y = p.h - 10 - (legend.length - 1 - i) * 15;
      p.ctx.fillStyle = col; p.ctx.globalAlpha = col === C.muted ? .4 : 1;
      p.ctx.beginPath(); p.ctx.arc(14, y, 4.2, 0, Math.PI * 2); p.ctx.fill();
      p.ctx.globalAlpha = 1;
      p.text({ px: 24, py: y }, label, { color: C.muted, size: 10.5, weight: 500 });
    });
  });

  refresh();

  node.appendChild(note(
    `Each dot is one person in a population of 1000. Among everyone who tests positive — the red dots plus ` +
    `the violet dots — only the red ones actually have the disease, and ` +
    `<span class="u-mono">ℙ(D | +)</span> is just red ÷ (red + violet). ` +
    `<strong>Set prevalence to 1% and a 95%-accurate test still gives you a ~16% chance of being ill.</strong> ` +
    `Bayes' theorem is the arithmetic; the picture is why it feels so wrong at first.`
  ));
});

/* ============================================================
   5. Joint / marginal / conditional
   ============================================================ */
defineWidget('joint-table', node => {
  const wrap = el('div');
  node.appendChild(wrap);

  let J = [[.1, .1, .1], [.2, .2, .3]];
  let mode = 'joint', selRow = -1, selCol = -1;

  const Min = matrixInput(2, 3, J, {
    label: 'Joint pmf &nbsp;p<sub>X,Y</sub>(x, y)', step: .05,
    onInput: m => { J = m; refresh(); },
  });
  const modeCtl = segmented([
    { label: 'Joint', value: 'joint' },
    { label: 'Marginal X', value: 'mx' },
    { label: 'Marginal Y', value: 'my' },
    { label: 'Conditional', value: 'cond' },
  ], { value: 'joint', label: 'View', onChange: v => { mode = v; refresh(); } });

  const cv = el('canvas');
  const out = readout([['Σ p (must be 1)', 0], ['p_X(0), p_X(1)', 0], ['p_Y(0), p_Y(1), p_Y(2)', 0], ['independent?', 0]]);
  const st = status('');

  const left = el('div', {}, el('div', { class: 'pg-canvas-wrap' }, cv));
  const right = el('div', { class: 'pg-controls' }, Min.root, modeCtl.root, out.root, st.root);
  wrap.appendChild(el('div', { class: 'pg-split pg-split--wide-ctrl' }, left, right));

  Min.cells.forEach((row, i) => row.forEach((c, j) => {
    c.addEventListener('focus', () => { selRow = i; selCol = j; refresh(); });
  }));

  const plot = trackPlot(new Plot(cv, { xmin: 0, xmax: 3, ymin: 0, ymax: 2, aspect: 1.27, pad: 0 }));

  const marginX = () => J.map(r => r.reduce((a, b) => a + b, 0));
  const marginY = () => J[0].map((_, j) => J.reduce((a, r) => a + r[j], 0));

  function refresh() {
    const total = J.flat().reduce((a, b) => a + b, 0);
    const mx = marginX(), my = marginY();
    const indep = J.every((r, i) => r.every((v, j) => Math.abs(v - mx[i] * my[j]) < 5e-3));
    out.set([
      { html: fmt(total, 3), cls: Math.abs(total - 1) < 5e-3 ? 'is-ok' : 'is-warn' },
      mx.map(v => fmt(v, 3)).join(', '),
      my.map(v => fmt(v, 3)).join(', '),
      { html: indep ? 'yes' : 'no', cls: indep ? 'is-ok' : '' },
    ]);

    const msgs = {
      joint: `Every cell is ℙ(X = x and Y = y). They must be non-negative and sum to exactly 1.`,
      mx: `<strong>Marginalising out Y</strong>: sum along each row to get p<sub>X</sub>(x). The information about Y is discarded.`,
      my: `<strong>Marginalising out X</strong>: sum down each column to get p<sub>Y</sub>(y).`,
      cond: selCol >= 0
        ? `<strong>Conditioning on Y = ${selCol}</strong>: take that column and rescale it so it sums to 1. That is the division by p<sub>Y</sub>(y).`
        : `Click a cell in the table to condition on its column.`,
    };
    st.set(msgs[mode], Math.abs(total - 1) < 5e-3 ? 'info' : 'warn');

    Min.highlight(
      mode === 'mx' && selRow >= 0 ? J[selRow].map((_, j) => [selRow, j, 'is-hl'])
      : (mode === 'my' || mode === 'cond') && selCol >= 0 ? J.map((_, i) => [i, selCol, 'is-hl2'])
      : null
    );
    plot.render();
  }

  plot.onDraw(p => {
    const mx = marginX(), my = marginY();
    const maxv = Math.max(...J.flat(), 1e-9);
    for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) {
      let v = J[i][j], denom = maxv, col = C.c1;
      if (mode === 'cond' && selCol >= 0) {
        v = my[selCol] > 1e-9 && j === selCol ? J[i][j] / my[selCol] : (j === selCol ? 0 : J[i][j]);
        denom = j === selCol ? 1 : maxv;
        col = j === selCol ? C.c3 : C.c1;
      }
      const t = clamp(v / denom, 0, 1);
      const dim = (mode === 'cond' && selCol >= 0 && j !== selCol) ? .22
                : (mode === 'mx' && selRow >= 0 && i !== selRow) ? .3
                : (mode === 'my' && selCol >= 0 && j !== selCol) ? .3 : 1;
      p.ctx.globalAlpha = dim;
      p.ctx.fillStyle = shade(col, .1 + t * .82);
      p.ctx.fillRect(p.X(j) + 2, p.Y(2 - i) + 2, p.px(1) - 4, p.px(1) - 4);
      p.ctx.globalAlpha = 1;
      p.text([j + .5, 1.5 - i], fmt(v, 3), {
        align: 'center', size: 12, weight: 700,
        color: t > .5 ? C.raised : C.ink,
      });
      p.text([j + .5, 1.5 - i], `x=${i}, y=${j}`, {
        align: 'center', dy: 16, size: 9.5, weight: 500,
        color: t > .5 ? C.raised : C.muted,
      });
    }
    // marginal bars
    if (mode === 'mx') mx.forEach((v, i) =>
      p.badge([3, 1.5 - i], `p_X(${i}) = ${fmt(v, 3)}`, { color: C.c2, dx: -4, align: 'right' }));
    if (mode === 'my') my.forEach((v, j) =>
      p.badge([j + .5, 0], `${fmt(v, 3)}`, { color: C.c2, align: 'center', dy: 14 }));
  });

  function shade(hex, a) {
    hex = hex.trim();
    if (!hex.startsWith('#')) return hex;
    const n = hex.length === 4
      ? hex.slice(1).split('').map(c => parseInt(c + c, 16))
      : [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));
    return `rgba(${n.join(',')},${a})`;
  }

  refresh();

  node.appendChild(note(
    `The joint table holds everything. <strong>Marginals</strong> come from summing away a variable; ` +
    `<strong>conditionals</strong> come from picking a slice and renormalising it so it is a probability ` +
    `distribution again. Independence is the special case where the joint factorises into the product of ` +
    `its marginals — try editing cells until the "independent?" readout flips to yes.`
  ));
});

/* ============================================================
   6. Covariance and correlation
   ============================================================ */
defineWidget('covariance', node => {
  const { right, canvas } = split(node, { aspect: 1.05 });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -3.4, xmax: 3.4, ymin: -3.4, ymax: 3.4, aspect: 1.05, pad: 6,
  }));

  let rho = .7, sx = 1, sy = 1, n = 320;
  let base = [];
  const rng = makeRng(90210);
  for (let i = 0; i < 600; i++) base.push(gaussPair(rng));

  const rCtl = slider('Correlation ρ', {
    min: -.99, max: .99, step: .01, value: rho, onInput: v => { rho = v; refresh(); },
  });
  const sxCtl = slider('σ<sub>X</sub>', { min: .3, max: 2.2, step: .05, value: 1, onInput: v => { sx = v; refresh(); } });
  const syCtl = slider('σ<sub>Y</sub>', { min: .3, max: 2.2, step: .05, value: 1, onInput: v => { sy = v; refresh(); } });
  const out = readout([['Cov(X, Y)', 0], ['ρ', 0], ['Var(X)', 0], ['Var(Y)', 0]]);
  const st = status('');
  right.append(rCtl.root, sxCtl.root, syCtl.root, out.root, st.root);

  const pts = () => base.slice(0, n).map(([z1, z2]) => [
    z1 * sx,
    (rho * z1 + Math.sqrt(Math.max(0, 1 - rho * rho)) * z2) * sy,
  ]);

  function refresh() {
    const cov = rho * sx * sy;
    out.set([fmt(cov, 3), fmt(rho, 2), fmt(sx * sx, 3), fmt(sy * sy, 3)]);
    st.set(
      Math.abs(rho) < .12
        ? `Almost no <strong>linear</strong> relationship — but that is not the same as independence.`
        : rho > 0
          ? `Positive: above-average X tends to come with above-average Y.`
          : `Negative: above-average X tends to come with below-average Y.`,
      Math.abs(rho) < .12 ? '' : 'info');
    plot.render();
  }

  plot.onDraw(p => {
    p.grid(1); p.axes({ ticks: 1 });
    const P = pts();
    P.forEach(v => p.dot(v, { r: 2.6, color: C.c1, alpha: .55 }));

    // 1σ and 2σ covariance ellipses
    const cov = rho * sx * sy;
    const S = [sx * sx, cov, cov, sy * sy];
    const e = mat2eig(S);
    [1, 2].forEach(k => {
      drawEllipse(p, e, k, k === 1 ? C.c2 : C.c4, k === 1 ? .95 : .5);
    });
    p.text({ px: 12, py: 11 }, '1σ and 2σ covariance ellipses', { color: C.muted, size: 11 });
  });

  function mat2eig([a, b, c, d]) {
    const tr = a + d, dt = a * d - b * c;
    const s = Math.sqrt(Math.max(0, tr * tr / 4 - dt));
    const l1 = tr / 2 + s, l2 = tr / 2 - s;
    let v1 = Math.abs(b) > 1e-9 ? [b, l1 - a] : [1, 0];
    const nn = Math.hypot(...v1) || 1;
    v1 = [v1[0] / nn, v1[1] / nn];
    return { l1: Math.max(0, l1), l2: Math.max(0, l2), v1 };
  }
  function drawEllipse(p, e, k, color, alpha) {
    const a = k * Math.sqrt(e.l1), b = k * Math.sqrt(e.l2);
    const th = Math.atan2(e.v1[1], e.v1[0]);
    const pts = [];
    for (let i = 0; i <= 96; i++) {
      const t = i / 96 * Math.PI * 2;
      const x = a * Math.cos(t), y = b * Math.sin(t);
      pts.push([x * Math.cos(th) - y * Math.sin(th), x * Math.sin(th) + y * Math.cos(th)]);
    }
    p.path(pts, { color, lw: 2, close: true, alpha });
  }

  refresh();

  node.appendChild(note(
    `Covariance measures how two variables move together, but its <em>units</em> are the product of theirs — ` +
    `change σ<sub>X</sub> and the covariance changes even though the cloud's shape does not. ` +
    `<strong>Correlation ρ divides that dependence out</strong>, which is why it stays pinned in [−1, 1]. ` +
    `The ellipses are the level sets of the covariance matrix; its eigenvectors are the axes — the exact ` +
    `object PCA goes looking for.`
  ));
});
