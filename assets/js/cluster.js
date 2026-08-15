/* ============================================================
   cluster.js — clustering and autoencoder numerics, from scratch
   Used by Chapter 6 (Unsupervised Learning).
   ============================================================ */
import { rng, gauss } from './ml.js';
import { eigSym } from './linalg.js';

export const sqd = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return s;
};
export const euclid = (a, b) => Math.sqrt(sqd(a, b));

/* ============================================================
   k-means
   ============================================================ */

/**
 * Lloyd's algorithm recorded step by step — one frame per iteration:
 *   { labels, before, centers, wcssBefore, wcssAfter, moved }
 * `wcssBefore` is measured against the centroids that produced the assignment,
 * which is the convention the notes' worked example uses.
 */
export function lloydTrace(points, initCenters, { iters = 40 } = {}) {
  const dim = points[0].length;
  let centers = initCenters.map(c => c.slice());
  const frames = [];
  let prev = null;
  for (let t = 0; t < iters; t++) {
    const labels = points.map(p => {
      let best = 0, bd = Infinity;
      for (let c = 0; c < centers.length; c++) {
        const d = sqd(p, centers[c]);
        if (d < bd - 1e-12) { bd = d; best = c; }        // ties go to the lower index
      }
      return best;
    });
    const wcssBefore = points.reduce((s, p, i) => s + sqd(p, centers[labels[i]]), 0);
    const before = centers.map(c => c.slice());
    const sums = centers.map(() => new Array(dim).fill(0));
    const cnt = new Array(centers.length).fill(0);
    points.forEach((p, i) => { cnt[labels[i]]++; p.forEach((v, j) => { sums[labels[i]][j] += v; }); });
    centers = centers.map((c, j) => (cnt[j] ? sums[j].map(v => v / cnt[j]) : c.slice()));
    const wcssAfter = points.reduce((s, p, i) => s + sqd(p, centers[labels[i]]), 0);
    const moved = prev ? labels.filter((l, i) => l !== prev[i]).length : labels.length;
    frames.push({ labels, before, centers: centers.map(c => c.slice()), wcssBefore, wcssAfter, moved });
    const stable = prev && labels.every((l, i) => l === prev[i]);
    prev = labels;
    if (stable) break;
  }
  return frames;
}

/** The ANOVA identity: TSS = WCSS + BCSS. */
export function anova(points, labels, centers) {
  const dim = points[0].length;
  const gm = new Array(dim).fill(0);
  points.forEach(p => p.forEach((v, j) => { gm[j] += v / points.length; }));
  const TSS = points.reduce((s, p) => s + sqd(p, gm), 0);
  const WCSS = points.reduce((s, p, i) => s + sqd(p, centers[labels[i]]), 0);
  const cnt = centers.map((_, j) => labels.filter(l => l === j).length);
  const BCSS = centers.reduce((s, c, j) => s + cnt[j] * sqd(c, gm), 0);
  return { TSS, WCSS, BCSS, grand: gm };
}

/** k-means++ seeding, exposed separately so the seeding step itself can be shown. */
export function kmeansppSeeds(points, k, seed = 3) {
  const r = rng(seed);
  const n = points.length;
  const picks = [Math.floor(r() * n) % n];
  const steps = [{ chosen: picks[0], probs: null }];
  while (picks.length < k) {
    const D2 = points.map(p => Math.min(...picks.map(i => sqd(p, points[i]))));
    const tot = D2.reduce((s, v) => s + v, 0);
    const probs = D2.map(v => (tot > 1e-12 ? v / tot : 1 / n));
    const t = r();
    let acc = 0, idx = n - 1;
    for (let i = 0; i < n; i++) { acc += probs[i]; if (t <= acc) { idx = i; break; } }
    picks.push(idx);
    steps.push({ chosen: idx, probs });
  }
  return { indices: picks, centers: picks.map(i => points[i].slice()), steps };
}

/** Uniformly random distinct seeds, for contrast with k-means++. */
export function randomSeeds(points, k, seed = 3) {
  const r = rng(seed);
  const n = points.length;
  const picks = [];
  let guard = 0;
  while (picks.length < k && guard++ < 5000) {
    const c = Math.floor(r() * n) % n;
    if (!picks.includes(c)) picks.push(c);
  }
  return { indices: picks, centers: picks.map(i => points[i].slice()) };
}

/** Run Lloyd's to convergence from given seeds; returns the final WCSS. */
export function lloydFinal(points, centers) {
  const f = lloydTrace(points, centers, { iters: 100 });
  const last = f[f.length - 1];
  return { wcss: last.wcssAfter, labels: last.labels, centers: last.centers, iters: f.length };
}

/* ============================================================
   k-medoids (PAM-style)
   ============================================================ */

export const metrics = {
  euclidean: (a, b) => euclid(a, b),
  manhattan: (a, b) => a.reduce((s, v, i) => s + Math.abs(v - b[i]), 0),
  cosine: (a, b) => {
    const na = Math.hypot(...a), nb = Math.hypot(...b);
    if (!na || !nb) return 1;
    return 1 - a.reduce((s, v, i) => s + v * b[i], 0) / (na * nb);
  },
};

export function kmedoids(points, k, { seed = 5, metric = 'euclidean', iters = 80 } = {}) {
  const n = points.length;
  const D = metrics[metric] || metrics.euclidean;
  const M = points.map(a => points.map(b => D(a, b)));
  const r = rng(seed);
  const med = [];
  let guard = 0;
  while (med.length < k && guard++ < 5000) {
    const c = Math.floor(r() * n) % n;
    if (!med.includes(c)) med.push(c);
  }
  const assign = ms => points.map((_, i) => {
    let best = 0, bd = Infinity;
    ms.forEach((m, j) => { if (M[i][m] < bd) { bd = M[i][m]; best = j; } });
    return best;
  });
  const cost = ms => points.reduce((s, _, i) => s + Math.min(...ms.map(m => M[i][m])), 0);
  let cur = cost(med);
  for (let t = 0; t < iters; t++) {
    let improved = false;
    for (let j = 0; j < k && !improved; j++) {
      for (let c = 0; c < n; c++) {
        if (med.includes(c)) continue;
        const trial = med.slice(); trial[j] = c;
        const tc = cost(trial);
        if (tc < cur - 1e-12) { med[j] = c; cur = tc; improved = true; break; }
      }
    }
    if (!improved) break;
  }
  return { medoids: med, labels: assign(med), cost: cur };
}

/* ============================================================
   Kernel k-means
   ============================================================ */

/**
 * Distances to feature-space means are computed entirely through the kernel
 * matrix, so phi(x) is never formed:
 *   ||phi(x) - mu_j||^2 = K(x,x) - (2/|Cj|) sum_i K(x,xi) + (1/|Cj|^2) sum K(xi,xi')
 */
export function kernelKmeans(points, k, {
  gamma = 1, seed = 4, iters = 80, kernel = 'rbf', degree = 3, coef = 1, restarts = 6,
} = {}) {
  const n = points.length;
  const K = points.map(a => points.map(b => (kernel === 'poly'
    ? (a.reduce((s, v, t) => s + v * b[t], 0) + coef) ** degree
    : Math.exp(-gamma * sqd(a, b)))));
  // squared distance in feature space, straight from the kernel matrix
  const fd2 = (i, j) => K[i][i] - 2 * K[i][j] + K[j][j];

  const runFrom = start => {
    let labels = points.map((_, i) => {
      let best = 0, bd = Infinity;
      start.forEach((s, c) => { const d = fd2(i, s); if (d < bd) { bd = d; best = c; } });
      return best;
    });
    let obj = Infinity;
    for (let t = 0; t < iters; t++) {
      const idx = Array.from({ length: k }, () => []);
      labels.forEach((l, i) => idx[l].push(i));
      const third = idx.map(ix => {
        if (!ix.length) return 0;
        let s = 0;
        for (const a of ix) for (const b of ix) s += K[a][b];
        return s / (ix.length * ix.length);
      });
      const dists = points.map((_, i) => Array.from({ length: k }, (__, c) => {
        const ix = idx[c];
        if (!ix.length) return Infinity;
        let second = 0;
        for (const a of ix) second += K[i][a];
        return K[i][i] - (2 / ix.length) * second + third[c];
      }));
      const next = dists.map(row => row.indexOf(Math.min(...row)));
      obj = dists.reduce((s, row, i) => s + row[next[i]], 0);
      const same = next.every((v, i) => v === labels[i]);
      labels = next;
      if (same) break;
    }
    return { labels, obj };
  };

  // k-means++ seeding *in feature space*: random initial labels get trapped in
  // poor local optima on exactly the nested-ring data this method exists for.
  let best = null;
  for (let attempt = 0; attempt < restarts; attempt++) {
    const r = rng(seed + attempt * 977);
    const start = [Math.floor(r() * n) % n];
    while (start.length < k) {
      const D2 = points.map((_, i) => Math.min(...start.map(s => fd2(i, s))));
      const tot = D2.reduce((s, v) => s + v, 0);
      const t = r() * tot;
      let acc = 0, idx = n - 1;
      for (let i = 0; i < n; i++) { acc += D2[i]; if (t <= acc) { idx = i; break; } }
      start.push(idx);
    }
    const run = runFrom(start);
    if (!best || run.obj < best.obj - 1e-12) best = run;
  }
  return { labels: best.labels, obj: best.obj, K };
}

/**
 * Spectral clustering (Ng–Jordan–Weiss) on an RBF affinity graph.
 * The notes describe this as a continuous relaxation of kernel k-means: rather
 * than assigning in feature space directly, embed with the smallest eigenvectors
 * of the normalised Laplacian and run plain k-means there. It keys on
 * *connectivity* instead of feature-space compactness, which is why it handles
 * nested rings that defeat kernel k-means.
 */
export function spectral(points, k, { gamma = 20, seed = 1, restarts = 8 } = {}) {
  const n = points.length;
  const W = points.map(a => points.map(b => Math.exp(-gamma * sqd(a, b))));
  for (let i = 0; i < n; i++) W[i][i] = 0;
  const deg = W.map(row => row.reduce((s, v) => s + v, 0) || 1e-12);
  const L = W.map((row, i) => row.map((v, j) => (i === j ? 1 : 0) - v / Math.sqrt(deg[i] * deg[j])));

  const { values, vectors } = eigSym(L, { iters: 250 });
  // eigSym sorts descending, so the k smallest eigenvalues sit at the end
  const ord = values.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]).slice(0, k).map(p => p[1]);
  const U = points.map((_, i) => {
    const row = ord.map(c => vectors[c][i]);
    const nn = Math.hypot(...row) || 1;
    return row.map(v => v / nn);                       // row-normalise, per NJW
  });

  const r = rng(seed);
  let best = null;
  for (let a = 0; a < restarts; a++) {
    const seeds = kmeansppSeeds(U, k, seed + a * 131).centers;
    const res = lloydFinal(U, seeds);
    if (!best || res.wcss < best.wcss - 1e-12) best = res;
  }
  void r;
  return { labels: best.labels, embedding: U, eigenvalues: ord.map(i => values[i]) };
}

/* ============================================================
   DBSCAN
   ============================================================ */

/**
 * Neighbourhoods exclude the point itself, matching the convention used in the
 * notes, so minPts counts *other* points within eps.
 */
export function dbscan(points, eps, minPts) {
  const n = points.length;
  const nbrs = points.map((p, i) =>
    points.map((_, j) => j).filter(j => j !== i && euclid(p, points[j]) <= eps + 1e-12));
  const isCore = nbrs.map(nb => nb.length >= minPts);
  const labels = new Array(n).fill(-1);
  let cid = 0;
  for (let i = 0; i < n; i++) {
    if (!isCore[i] || labels[i] !== -1) continue;
    labels[i] = cid;
    const stack = [i];
    while (stack.length) {
      const q = stack.pop();
      for (const s of nbrs[q]) {
        if (labels[s] === -1) {
          labels[s] = cid;
          if (isCore[s]) stack.push(s);
        }
      }
    }
    cid++;
  }
  const kind = points.map((_, i) => (isCore[i] ? 'core' : (labels[i] >= 0 ? 'border' : 'noise')));
  return { labels, isCore, kind, nbrs, clusters: cid };
}

/** Distance from each point to its k-th nearest *other* point. */
export function kDistances(points, k) {
  return points.map((p, i) => {
    const ds = points.map((q, j) => (j === i ? Infinity : euclid(p, q)))
      .filter(Number.isFinite).sort((a, b) => a - b);
    return ds[k - 1] ?? Infinity;
  });
}

/* ============================================================
   Gaussian mixtures / EM
   ============================================================ */

const inv2 = (A, d) => (d === 1
  ? [[1 / A[0][0]]]
  : (() => {
    const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    return [[A[1][1] / det, -A[0][1] / det], [-A[1][0] / det, A[0][0] / det]];
  })());
const det2 = (A, d) => (d === 1 ? A[0][0] : A[0][0] * A[1][1] - A[0][1] * A[1][0]);

export function gaussPdf(x, m, A) {
  const d = x.length;
  const Ai = inv2(A, d), dt = Math.max(det2(A, d), 1e-300);
  let q = 0;
  for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) q += (x[a] - m[a]) * Ai[a][b] * (x[b] - m[b]);
  return Math.exp(-0.5 * q) / Math.sqrt((2 * Math.PI) ** d * dt);
}

/**
 * EM for a Gaussian mixture in 1-D or 2-D, recorded step by step.
 * Each frame holds the parameters *entering* the iteration together with the
 * responsibilities and log-likelihood they produce.
 */
export function gmmEM(points, k, { seed = 9, iters = 80, reg = 1e-6, init = null, tied = null } = {}) {
  const n = points.length, d = points[0].length;
  const r = rng(seed);
  let mu = init ? init.map(c => c.slice())
    : Array.from({ length: k }, () => points[Math.floor(r() * n) % n].slice());
  let S = Array.from({ length: k }, () =>
    Array.from({ length: d }, (_, a) => Array.from({ length: d }, (_, b) => (a === b ? 1 : 0))));
  let pi = new Array(k).fill(1 / k);
  const frames = [];

  for (let t = 0; t < iters; t++) {
    const gamma = [];
    let loglik = 0;
    for (let i = 0; i < n; i++) {
      const w = [];
      for (let j = 0; j < k; j++) w.push(pi[j] * gaussPdf(points[i], mu[j], S[j]));
      const tot = w.reduce((s, v) => s + v, 0) || 1e-300;
      loglik += Math.log(tot);
      gamma.push(w.map(v => v / tot));
    }
    frames.push({
      pi: pi.slice(), mu: mu.map(m => m.slice()), S: S.map(m => m.map(row => row.slice())),
      gamma: gamma.map(g => g.slice()), loglik,
      hard: gamma.map(g => g.indexOf(Math.max(...g))),
    });

    const N = Array.from({ length: k }, (_, j) => gamma.reduce((s, g) => s + g[j], 0));
    pi = N.map(v => v / n);
    mu = Array.from({ length: k }, (_, j) => {
      const m = new Array(d).fill(0);
      points.forEach((p, i) => p.forEach((v, a) => { m[a] += gamma[i][j] * v; }));
      return m.map(v => v / (N[j] || 1e-12));
    });
    S = Array.from({ length: k }, (_, j) => {
      const A = Array.from({ length: d }, () => new Array(d).fill(0));
      points.forEach((p, i) => {
        for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) {
          A[a][b] += gamma[i][j] * (p[a] - mu[j][a]) * (p[b] - mu[j][b]);
        }
      });
      for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) A[a][b] /= (N[j] || 1e-12);
      if (tied === 'spherical') {
        const avg = Array.from({ length: d }, (_, a) => A[a][a]).reduce((s, v) => s + v, 0) / d;
        for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) A[a][b] = (a === b ? avg : 0);
      } else if (tied === 'diagonal') {
        for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) if (a !== b) A[a][b] = 0;
      }
      for (let a = 0; a < d; a++) A[a][a] += reg;
      return A;
    });
    if (t > 1 && Math.abs(loglik - frames[t - 1].loglik) < 1e-10 * (1 + Math.abs(loglik))) break;
  }
  return frames;
}

/** Ellipse (centre + axes + rotation) for a 2x2 covariance at a given sigma level. */
export function covEllipse(S, level = 2) {
  const a = S[0][0], b = S[0][1], c = S[1][1];
  const tr = a + c, dt = a * c - b * b;
  const disc = Math.max(0, tr * tr / 4 - dt);
  const l1 = tr / 2 + Math.sqrt(disc), l2 = tr / 2 - Math.sqrt(disc);
  const theta = Math.abs(b) < 1e-12 ? (a >= c ? 0 : Math.PI / 2) : Math.atan2(l1 - a, b);
  return { rx: level * Math.sqrt(Math.max(l1, 1e-12)), ry: level * Math.sqrt(Math.max(l2, 1e-12)), theta };
}

/* ============================================================
   Autoencoders
   ============================================================ */

/** KL between two Bernoullis — the sparse-autoencoder penalty. */
export function klBernoulli(rho, rhoHat) {
  const e = 1e-9;
  const p = Math.min(Math.max(rho, e), 1 - e);
  const q = Math.min(Math.max(rhoHat, e), 1 - e);
  return p * Math.log(p / q) + (1 - p) * Math.log((1 - p) / (1 - q));
}

/** KL(N(mu, sigma^2) || N(0,1)) summed over dimensions — the VAE regulariser. */
export function klGaussian(mu, sigma) {
  return 0.5 * mu.reduce((s, m, j) => s + m * m + sigma[j] ** 2 - 2 * Math.log(sigma[j]) - 1, 0);
}

/**
 * A small autoencoder: d -> h -> k -> h -> d, tanh hidden units, linear output.
 * Trained by full-batch gradient descent so a figure can step through training.
 * With `linear: true` the hidden layers are identity, which makes the network a
 * plain low-rank map and lets it be compared against PCA.
 */
export function makeAE(d, h, k, { seed = 7, linear = false } = {}) {
  const r = rng(seed);
  const mat = (m, n2, s) => Array.from({ length: m }, () => Array.from({ length: n2 }, () => gauss(r) * s));
  const vec = m => new Array(m).fill(0);
  const s1 = Math.sqrt(1 / d), s2 = Math.sqrt(1 / h), s3 = Math.sqrt(1 / k), s4 = Math.sqrt(1 / h);
  return {
    linear, d, h, k,
    W1: mat(h, d, s1), b1: vec(h),
    W2: mat(k, h, s2), b2: vec(k),
    W3: mat(h, k, s3), b3: vec(h),
    W4: mat(d, h, s4), b4: vec(d),
  };
}

const mv = (M, v, b) => M.map((row, i) => row.reduce((s, w, j) => s + w * v[j], 0) + b[i]);
const act = (v, lin) => (lin ? v.slice() : v.map(Math.tanh));
const dact = (a, lin) => (lin ? a.map(() => 1) : a.map(v => 1 - v * v));

export function aeForward(net, x) {
  const a1 = act(mv(net.W1, x, net.b1), net.linear);
  const z = mv(net.W2, a1, net.b2);                       // latent code, always linear
  const a3 = act(mv(net.W3, z, net.b3), net.linear);
  const xh = mv(net.W4, a3, net.b4);
  return { a1, z, a3, xh };
}

/**
 * One full-batch gradient step on mean squared reconstruction error.
 * The gradient is norm-clipped: a fully linear network has no squashing
 * anywhere, so its gradients compound multiplicatively and will otherwise
 * diverge to NaN at any useful learning rate.
 */
export function aeStep(net, X, lr, { clip = 5 } = {}) {
  const n = X.length;
  const g = {
    W1: net.W1.map(r2 => r2.map(() => 0)), b1: net.b1.map(() => 0),
    W2: net.W2.map(r2 => r2.map(() => 0)), b2: net.b2.map(() => 0),
    W3: net.W3.map(r2 => r2.map(() => 0)), b3: net.b3.map(() => 0),
    W4: net.W4.map(r2 => r2.map(() => 0)), b4: net.b4.map(() => 0),
  };
  let loss = 0;
  for (const x of X) {
    const { a1, z, a3, xh } = aeForward(net, x);
    const e = xh.map((v, i) => v - x[i]);
    loss += e.reduce((s, v) => s + v * v, 0) / n;

    const d4 = e.map(v => (2 * v) / n);
    d4.forEach((dv, i) => { g.b4[i] += dv; a3.forEach((av, j) => { g.W4[i][j] += dv * av; }); });

    const back3 = a3.map((_, j) => d4.reduce((s, dv, i) => s + dv * net.W4[i][j], 0));
    const d3 = back3.map((v, j) => v * dact(a3, net.linear)[j]);
    d3.forEach((dv, i) => { g.b3[i] += dv; z.forEach((zv, j) => { g.W3[i][j] += dv * zv; }); });

    const d2 = z.map((_, j) => d3.reduce((s, dv, i) => s + dv * net.W3[i][j], 0));
    d2.forEach((dv, i) => { g.b2[i] += dv; a1.forEach((av, j) => { g.W2[i][j] += dv * av; }); });

    const back1 = a1.map((_, j) => d2.reduce((s, dv, i) => s + dv * net.W2[i][j], 0));
    const d1 = back1.map((v, j) => v * dact(a1, net.linear)[j]);
    d1.forEach((dv, i) => { g.b1[i] += dv; x.forEach((xv, j) => { g.W1[i][j] += dv * xv; }); });
  }
  // global gradient-norm clipping
  let sq = 0;
  for (const key of ['W1', 'W2', 'W3', 'W4']) for (const row of g[key]) for (const v of row) sq += v * v;
  for (const key of ['b1', 'b2', 'b3', 'b4']) for (const v of g[key]) sq += v * v;
  const gn = Math.sqrt(sq);
  if (!Number.isFinite(gn)) return loss;                 // refuse a poisoned step
  const scale = gn > clip ? clip / gn : 1;

  for (const key of ['W1', 'W2', 'W3', 'W4']) {
    net[key] = net[key].map((row, i) => row.map((v, j) => v - lr * scale * g[key][i][j]));
  }
  for (const key of ['b1', 'b2', 'b3', 'b4']) {
    net[key] = net[key].map((v, i) => v - lr * scale * g[key][i]);
  }
  return loss;
}

export const aeLoss = (net, X) =>
  X.reduce((s, x) => s + sqd(aeForward(net, x).xh, x), 0) / X.length;

/* ---------- datasets ---------- */

export function moons({ n = 160, noise = .07, seed = 21 } = {}) {
  const r = rng(seed), points = [], truth = [];
  const half = Math.ceil(n / 2);
  for (let i = 0; i < n; i++) {
    const top = i < half;
    const t = Math.PI * (i % half) / (half - 1);
    const p = top ? [Math.cos(t), Math.sin(t)] : [1 - Math.cos(t), .5 - Math.sin(t)];
    points.push([p[0] + gauss(r) * noise, p[1] + gauss(r) * noise]);
    truth.push(top ? 0 : 1);
  }
  return { points, truth };
}

export function circles({ n = 160, noise = .055, ratio = .45, seed = 22 } = {}) {
  const r = rng(seed), points = [], truth = [];
  const half = Math.ceil(n / 2);
  for (let i = 0; i < n; i++) {
    const outer = i < half;
    const t = 2 * Math.PI * (i % half) / half;
    const rad = outer ? 1 : ratio;
    points.push([rad * Math.cos(t) + gauss(r) * noise, rad * Math.sin(t) + gauss(r) * noise]);
    truth.push(outer ? 0 : 1);
  }
  return { points, truth };
}

export function anisotropic({ n = 150, seed = 23 } = {}) {
  const r = rng(seed), points = [], truth = [];
  const centres = [[0, 0], [3.4, .8], [1.7, 3.4]];
  const T = [[.62, -.62], [.30, .78]];
  for (let i = 0; i < n; i++) {
    const c = i % 3;
    const v = [gauss(r), gauss(r)];
    points.push([
      centres[c][0] + T[0][0] * v[0] + T[0][1] * v[1],
      centres[c][1] + T[1][0] * v[0] + T[1][1] * v[1],
    ]);
    truth.push(c);
  }
  return { points, truth };
}

export function unequalBlobs({ n = 150, seed = 24 } = {}) {
  const r = rng(seed), points = [], truth = [];
  const spec = [{ c: [-1.4, 0], s: .30 }, { c: [1.8, 0], s: 1.15 }];
  for (let i = 0; i < n; i++) {
    const j = i < n / 2 ? 0 : 1;
    points.push([spec[j].c[0] + gauss(r) * spec[j].s, spec[j].c[1] + gauss(r) * spec[j].s]);
    truth.push(j);
  }
  return { points, truth };
}

/** Two blobs of different density, a curved arm, and scattered noise. */
export function densityScene({ seed = 31 } = {}) {
  const r = rng(seed), out = [];
  for (let i = 0; i < 55; i++) out.push([-1.9 + gauss(r) * .28, 1.1 + gauss(r) * .28]);
  for (let i = 0; i < 50; i++) out.push([1.6 + gauss(r) * .58, 1.2 + gauss(r) * .58]);
  for (let i = 0; i < 42; i++) {
    const t = Math.PI * i / 41;
    out.push([1.7 * Math.cos(t) + gauss(r) * .10, -1.4 + 1.0 * Math.sin(t) + gauss(r) * .10]);
  }
  for (let i = 0; i < 14; i++) out.push([(r() * 2 - 1) * 3.3, (r() * 2 - 1) * 2.7]);
  return out;
}

/** Points along a curve in 2-D, for the autoencoder-vs-PCA figures. */
export function curveData({ n = 120, seed = 33, bend = 1, noise = .06 } = {}) {
  const r = rng(seed), out = [];
  for (let i = 0; i < n; i++) {
    const t = -1.6 + 3.2 * (i / (n - 1));
    out.push([t + gauss(r) * noise, bend * (t * t * .55 - .6) + .35 * t + gauss(r) * noise]);
  }
  return out;
}
