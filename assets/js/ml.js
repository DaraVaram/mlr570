/* ============================================================
   ml.js — small ML primitives shared by the Chapter 2 figures.
   Deterministic where randomness is involved, so figures are stable.
   ============================================================ */
import { svd, transpose, zeros } from './linalg.js';

/* ---------- deterministic RNG ---------- */
export function rng(seed = 12345) {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5;  s >>>= 0;
    return s / 4294967296;
  };
}
export function gauss(r) {
  const u = Math.max(1e-12, r()), v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/* ---------- descriptive statistics ---------- */
export const mean = a => a.reduce((s, v) => s + v, 0) / (a.length || 1);
export function variance(a, ddof = 0) {
  const n = a.length;
  if (n - ddof <= 0) return 0;
  const m = mean(a);
  return a.reduce((s, v) => s + (v - m) ** 2, 0) / (n - ddof);
}
export const std = (a, ddof = 0) => Math.sqrt(variance(a, ddof));

export function median(a) {
  if (!a.length) return NaN;
  const b = [...a].sort((x, y) => x - y);
  const m = b.length >> 1;
  return b.length % 2 ? b[m] : (b[m - 1] + b[m]) / 2;
}
export function quantile(a, q) {
  if (!a.length) return NaN;
  const b = [...a].sort((x, y) => x - y);
  const pos = (b.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return lo === hi ? b[lo] : b[lo] + (b[hi] - b[lo]) * (pos - lo);
}
export const iqr = a => quantile(a, .75) - quantile(a, .25);

export function pearson(a, b) {
  const n = Math.min(a.length, b.length);
  if (n < 2) return NaN;
  const ma = mean(a), mb = mean(b);
  let cov = 0, va = 0, vb = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - ma, db = b[i] - mb;
    cov += da * db; va += da * da; vb += db * db;
  }
  const den = Math.sqrt(va * vb);
  return den < 1e-12 ? 0 : cov / den;
}

export function spearman(a, b) {
  const rank = arr => {
    const idx = arr.map((v, i) => [v, i]).sort((x, y) => x[0] - y[0]);
    const r = new Array(arr.length);
    let i = 0;
    while (i < idx.length) {
      let j = i;
      while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
      const avg = (i + j) / 2 + 1;
      for (let k = i; k <= j; k++) r[idx[k][1]] = avg;
      i = j + 1;
    }
    return r;
  };
  return pearson(rank(a), rank(b));
}

/* ---------- scaling ---------- */
export const scalers = {
  none:   col => col.slice(),
  minmax: col => {
    const lo = Math.min(...col), hi = Math.max(...col);
    const d = hi - lo;
    return d < 1e-12 ? col.map(() => 0) : col.map(v => (v - lo) / d);
  },
  zscore: col => {
    const m = mean(col), s = std(col, 1);
    return s < 1e-12 ? col.map(() => 0) : col.map(v => (v - m) / s);
  },
  robust: col => {
    const md = median(col), r = iqr(col);
    return r < 1e-12 ? col.map(() => 0) : col.map(v => (v - md) / r);
  },
};

/** Apply a scaler column-wise to a row-major matrix. */
export function scaleColumns(X, kind) {
  const f = scalers[kind] || scalers.none;
  const cols = X[0].map((_, j) => f(X.map(r => r[j])));
  return X.map((_, i) => cols.map(c => c[i]));
}

/* ---------- k-means (Lloyd's, k-means++ seeding, deterministic) ---------- */
export function kmeans(points, k, { iters = 100, seed = 7 } = {}) {
  const n = points.length;
  if (!n || k <= 0) return { labels: [], centers: [], inertia: 0, iterations: 0 };
  const dim = points[0].length;
  const r = rng(seed);
  const d2 = (a, b) => { let s = 0; for (let i = 0; i < dim; i++) s += (a[i] - b[i]) ** 2; return s; };

  // k-means++ seeding
  const centers = [points[Math.floor(r() * n) % n].slice()];
  while (centers.length < Math.min(k, n)) {
    const dist = points.map(p => Math.min(...centers.map(c => d2(p, c))));
    const total = dist.reduce((s, v) => s + v, 0);
    let t = r() * total, idx = 0;
    if (total <= 1e-12) {
      idx = Math.floor(r() * n) % n;
    } else {
      for (let i = 0; i < n; i++) { t -= dist[i]; if (t <= 0) { idx = i; break; } }
    }
    centers.push(points[idx].slice());
  }
  while (centers.length < k) centers.push(points[0].slice());

  let labels = new Array(n).fill(0);
  let it = 0;
  for (; it < iters; it++) {
    let moved = false;
    for (let i = 0; i < n; i++) {
      let best = 0, bd = Infinity;
      for (let c = 0; c < k; c++) {
        const dd = d2(points[i], centers[c]);
        if (dd < bd) { bd = dd; best = c; }
      }
      if (labels[i] !== best) { labels[i] = best; moved = true; }
    }
    const sums = Array.from({ length: k }, () => new Array(dim).fill(0));
    const cnt = new Array(k).fill(0);
    for (let i = 0; i < n; i++) {
      cnt[labels[i]]++;
      for (let j = 0; j < dim; j++) sums[labels[i]][j] += points[i][j];
    }
    for (let c = 0; c < k; c++) {
      if (!cnt[c]) continue;
      for (let j = 0; j < dim; j++) centers[c][j] = sums[c][j] / cnt[c];
    }
    if (!moved && it > 0) break;
  }
  let inertia = 0;
  for (let i = 0; i < n; i++) inertia += d2(points[i], centers[labels[i]]);
  return { labels, centers, inertia, iterations: it + 1 };
}

/**
 * Exact optimal 1-D k-means over contiguous partitions.
 * For 1-D data the optimal clustering is always an interval partition, so a
 * small dynamic program gives the true optimum rather than a local one.
 */
export function kmeans1dOptimal(values, k) {
  const x = [...values].sort((a, b) => a - b);
  const n = x.length;
  k = Math.max(1, Math.min(k, n));

  // prefix sums for O(1) segment cost
  const p = [0], p2 = [0];
  for (let i = 0; i < n; i++) { p.push(p[i] + x[i]); p2.push(p2[i] + x[i] * x[i]); }
  const cost = (i, j) => {                       // cost of x[i..j-1]
    const cnt = j - i;
    if (cnt <= 0) return 0;
    const s = p[j] - p[i], s2 = p2[j] - p2[i];
    return Math.max(0, s2 - s * s / cnt);
  };

  const D = Array.from({ length: k + 1 }, () => new Array(n + 1).fill(Infinity));
  const B = Array.from({ length: k + 1 }, () => new Array(n + 1).fill(0));
  D[0][0] = 0;
  for (let c = 1; c <= k; c++) {
    for (let j = 1; j <= n; j++) {
      for (let i = c - 1; i < j; i++) {
        const v = D[c - 1][i] + cost(i, j);
        if (v < D[c][j]) { D[c][j] = v; B[c][j] = i; }
      }
    }
  }
  // walk the split points back
  const edgesIdx = [n];
  let j = n;
  for (let c = k; c >= 1; c--) { j = B[c][j]; edgesIdx.unshift(j); }
  const groups = [];
  for (let c = 0; c < k; c++) groups.push(x.slice(edgesIdx[c], edgesIdx[c + 1]));
  const centers = groups.map(g => (g.length ? g.reduce((s, v) => s + v, 0) / g.length : NaN));
  return { groups, centers, inertia: D[k][n], sorted: x };
}

/* ---------- binning ---------- */
export function equalWidthBins(values, k) {
  const lo = Math.min(...values), hi = Math.max(...values);
  const w = (hi - lo) / k;
  const edges = Array.from({ length: k + 1 }, (_, i) => lo + i * w);
  edges[k] = hi;
  const assign = values.map(v => Math.min(k - 1, Math.max(0, Math.floor((v - lo) / w))));
  return { edges, assign, width: w };
}

export function equalFreqBins(values, k) {
  const sorted = [...values].sort((a, b) => a - b);
  const edges = [sorted[0]];
  for (let i = 1; i < k; i++) edges.push(quantile(sorted, i / k));
  edges.push(sorted[sorted.length - 1]);
  const assign = values.map(v => {
    for (let b = 0; b < k; b++) {
      if (v <= edges[b + 1] || b === k - 1) return b;
    }
    return k - 1;
  });
  return { edges, assign };
}

export function kmeansBins(values, k) {
  const { groups, centers } = kmeans1dOptimal(values, k);
  // midpoints between consecutive cluster boundaries become the edges
  const edges = [Math.min(...values)];
  for (let c = 0; c < groups.length - 1; c++) {
    const a = groups[c][groups[c].length - 1];
    const b = groups[c + 1][0];
    edges.push((a + b) / 2);
  }
  edges.push(Math.max(...values));
  const assign = values.map(v => {
    let best = 0, bd = Infinity;
    centers.forEach((c, i) => { const d = Math.abs(v - c); if (d < bd) { bd = d; best = i; } });
    return best;
  });
  return { edges, assign, centers, groups };
}

/* ---------- PCA ---------- */
/**
 * PCA on a row-major matrix. Centres (and optionally standardises) first,
 * then takes the SVD of the centred matrix — numerically the stable route.
 * Returns components as rows of `components`.
 */
export function pca(X, { standardize = false } = {}) {
  const n = X.length, d = X[0].length;
  const cols = X[0].map((_, j) => X.map(r => r[j]));
  const mu = cols.map(mean);
  const sd = cols.map(c => std(c, 1));
  const Xc = X.map(row => row.map((v, j) => {
    const centred = v - mu[j];
    return standardize ? (sd[j] < 1e-12 ? 0 : centred / sd[j]) : centred;
  }));

  const { U, s, V } = svd(Xc);
  const denom = Math.max(1, n - 1);
  const eigenvalues = s.map(v => (v * v) / denom);
  const total = eigenvalues.reduce((a, b) => a + b, 0) || 1;
  const ratio = eigenvalues.map(v => v / total);
  const components = [];
  for (let c = 0; c < Math.min(d, s.length); c++) components.push(V.map(row => row[c]));

  const scores = Xc.map(row => components.map(comp =>
    comp.reduce((acc, cv, j) => acc + cv * row[j], 0)));

  return { mu, sd, Xc, eigenvalues, ratio, components, scores, singular: s };
}

/* ---------- mutual information (discrete estimate) ---------- */
export function mutualInfo(x, y, bins = 8) {
  const n = x.length;
  const lo = Math.min(...x), hi = Math.max(...x);
  const w = (hi - lo) / bins || 1;
  const bx = x.map(v => Math.min(bins - 1, Math.max(0, Math.floor((v - lo) / w))));
  const classes = [...new Set(y)];
  const joint = new Map();
  const px = new Array(bins).fill(0);
  const py = new Map(classes.map(c => [c, 0]));
  for (let i = 0; i < n; i++) {
    const key = bx[i] + '|' + y[i];
    joint.set(key, (joint.get(key) || 0) + 1);
    px[bx[i]]++;
    py.set(y[i], py.get(y[i]) + 1);
  }
  let mi = 0;
  for (const [key, c] of joint) {
    const [b, cls] = key.split('|');
    const pxy = c / n;
    const a = px[+b] / n, bb = py.get(cls) / n;
    if (pxy > 0 && a > 0 && bb > 0) mi += pxy * Math.log2(pxy / (a * bb));
  }
  return Math.max(0, mi);
}

/* ---------- synthetic datasets ---------- */

/** Three well-separated blobs where feature 1 is `spread` times feature 2. */
export function blobs({ n = 150, spread = 1, seed = 42 } = {}) {
  const r = rng(seed);
  const cent = [[-1.6, -1.2], [0, 1.5], [1.7, -1.0]];
  const pts = [], labels = [];
  for (let i = 0; i < n; i++) {
    const c = i % 3;
    pts.push([
      (cent[c][0] + gauss(r) * 0.42) * spread,
      cent[c][1] + gauss(r) * 0.42,
    ]);
    labels.push(c);
  }
  return { points: pts, labels };
}

/** Mixed-scale tabular data: age, income, credit score, monthly spend. */
export function peopleTable({ n = 200, seed = 11 } = {}) {
  const r = rng(seed);
  const rows = [];
  for (let i = 0; i < n; i++) {
    const age = Math.round(22 + Math.abs(gauss(r)) * 14);
    const income = Math.round(1800 + age * 62 + gauss(r) * 900);
    const credit = Math.round(Math.min(900, Math.max(300, 560 + (income - 4000) * 0.031 + gauss(r) * 70)));
    const spend = Math.round(Math.max(120, income * 0.31 + gauss(r) * 260));
    rows.push([age, income, credit, spend]);
  }
  return { rows, names: ['Age', 'Annual income', 'Credit score', 'Monthly spend'] };
}

/** Noisy upward-trending series, optionally with injected spikes. */
export function trendSeries({ n = 120, spikes = 0, seed = 5, noise = 1 } = {}) {
  const r = rng(seed);
  const y = [];
  for (let i = 0; i < n; i++) y.push(10 + i * 0.14 + gauss(r) * noise);
  if (spikes > 0) {
    const step = Math.floor(n / (spikes + 1));
    for (let s = 1; s <= spikes; s++) {
      const idx = Math.min(n - 1, s * step + 3);
      y[idx] += (s % 2 ? 1 : -1) * (11 + 5 * ((s * 37) % 5));
    }
  }
  return y;
}

/* ---------- rolling aggregations ---------- */
export function rolling(y, w, fn) {
  const out = [];
  for (let t = 0; t < y.length; t++) {
    if (t < w - 1) { out.push(null); continue; }
    out.push(fn(y.slice(t - w + 1, t + 1)));
  }
  return out;
}
export const rollingMean = (y, w) => rolling(y, w, a => mean(a));
export const rollingMedian = (y, w) => rolling(y, w, a => median(a));

/** EWMA in the recursive form  x̄ₜ = (1−λ)xₜ + λ x̄ₜ₋₁. */
export function ewma(y, lambda) {
  const out = [];
  let prev = y[0];
  for (let t = 0; t < y.length; t++) {
    prev = t === 0 ? y[0] : (1 - lambda) * y[t] + lambda * prev;
    out.push(prev);
  }
  return out;
}

/* ---------- classification metric helpers ---------- */
export function classMetrics({ TP, FN, FP, TN }) {
  const tot = TP + FN + FP + TN;
  const accuracy = tot ? (TP + TN) / tot : 0;
  const precision = TP + FP ? TP / (TP + FP) : 0;
  const recall = TP + FN ? TP / (TP + FN) : 0;
  const specificity = TN + FP ? TN / (TN + FP) : 0;
  const f1 = precision + recall ? 2 * precision * recall / (precision + recall) : 0;
  const fbeta = b => {
    const b2 = b * b;
    return b2 * precision + recall ? (1 + b2) * precision * recall / (b2 * precision + recall) : 0;
  };
  return { accuracy, precision, recall, specificity, f1, fbeta, total: tot };
}
