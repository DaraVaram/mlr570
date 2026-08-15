/* ============================================================
   linalg.js — dense numerical linear algebra for the widgets.
   Plain arrays-of-arrays. Small sizes only (< ~200), no BLAS.
   ============================================================ */

export const zeros = (m, n) => Array.from({ length: m }, () => new Array(n).fill(0));
export const eye = n => Array.from({ length: n }, (_, i) =>
  Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
export const clone = A => A.map(r => r.slice());
export const shape = A => [A.length, A[0]?.length ?? 0];

export function matmul(A, B) {
  const [m, k] = shape(A), [k2, n] = shape(B);
  if (k !== k2) throw new Error(`matmul: inner dimensions ${k} and ${k2} disagree`);
  const C = zeros(m, n);
  for (let i = 0; i < m; i++) {
    const Ai = A[i], Ci = C[i];
    for (let p = 0; p < k; p++) {
      const a = Ai[p];
      if (a === 0) continue;
      const Bp = B[p];
      for (let j = 0; j < n; j++) Ci[j] += a * Bp[j];
    }
  }
  return C;
}

export const transpose = A => {
  const [m, n] = shape(A);
  const T = zeros(n, m);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) T[j][i] = A[i][j];
  return T;
};

export const matvec = (A, x) => A.map(r => r.reduce((s, v, j) => s + v * x[j], 0));
export const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
export const norm2 = a => Math.hypot(...a);
export const scale = (A, s) => A.map(r => r.map(v => v * s));
export const frob = A => Math.sqrt(A.flat().reduce((s, v) => s + v * v, 0));

/* ============================================================
   Reduced row echelon form, with a recorded trace of the steps.
   Returns { R, pivots, steps, rank, swaps }
   ============================================================ */
export function rref(Ain, { tol = 1e-10, trace = false } = {}) {
  const R = clone(Ain);
  const [m, n] = shape(R);
  const pivots = [];
  const steps = [];
  let row = 0;

  const snap = (desc, hi) => { if (trace) steps.push({ desc, hi, M: clone(R) }); };
  snap('Starting matrix', null);

  for (let col = 0; col < n && row < m; col++) {
    // partial pivoting for numerical stability
    let piv = row, best = Math.abs(R[row][col]);
    for (let r = row + 1; r < m; r++) {
      const v = Math.abs(R[r][col]);
      if (v > best) { best = v; piv = r; }
    }
    if (best < tol) { for (let r = row; r < m; r++) R[r][col] = 0; continue; }

    if (piv !== row) {
      [R[row], R[piv]] = [R[piv], R[row]];
      snap(`R${row + 1} ↔ R${piv + 1}`, { rows: [row, piv] });
    }

    const p = R[row][col];
    if (Math.abs(p - 1) > tol) {
      for (let j = 0; j < n; j++) R[row][j] /= p;
      snap(`R${row + 1} ← (1/${fmtNum(p)})·R${row + 1}`, { rows: [row] });
    }

    let eliminated = false;
    for (let r = 0; r < m; r++) {
      if (r === row) continue;
      const f = R[r][col];
      if (Math.abs(f) < tol) continue;
      for (let j = 0; j < n; j++) R[r][j] -= f * R[row][j];
      eliminated = true;
    }
    if (eliminated) snap(`Clear column ${col + 1} using R${row + 1}`, { cols: [col] });

    pivots.push(col);
    row++;
  }

  // tidy up -0 and float dust
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) {
    if (Math.abs(R[i][j]) < tol) R[i][j] = 0;
    const near = Math.round(R[i][j]);
    if (Math.abs(R[i][j] - near) < 1e-9) R[i][j] = near;
  }

  return { R, pivots, steps, rank: pivots.length };
}

const fmtNum = v => {
  const r = Math.round(v * 1000) / 1000;
  return Number.isInteger(r) ? r : r.toFixed(3).replace(/0+$/, '');
};

export function rank(A, tol = 1e-9) {
  const { s } = svd(A);
  const mx = s[0] || 0;
  const cut = tol * Math.max(...shape(A)) * mx;
  return s.filter(v => v > cut).length;
}

/**
 * Basis for the null space of A, from the RREF (exact-ish for teaching matrices).
 * Returns an array of basis vectors (each length n). Empty ⇒ trivial null space.
 */
export function nullBasis(A) {
  const [, n] = shape(A);
  const { R, pivots } = rref(A);
  const free = [];
  for (let j = 0; j < n; j++) if (!pivots.includes(j)) free.push(j);
  return free.map(f => {
    const v = new Array(n).fill(0);
    v[f] = 1;
    pivots.forEach((pc, pr) => { v[pc] = -R[pr][f]; });
    return v;
  });
}

/* ============================================================
   SVD — one-sided Jacobi. Robust and compact for small dense A.
   Returns { U (m×k), s (length k, desc), V (n×k) } with k = min(m,n).
   A ≈ U · diag(s) · Vᵀ
   ============================================================ */
export function svd(Ain, { iters = 60, tol = 1e-12 } = {}) {
  const [m, n] = shape(Ain);
  // one-sided Jacobi wants at least as many rows as columns
  if (m < n) {
    const r = svd(transpose(Ain), { iters, tol });
    return { U: r.V, s: r.s, V: r.U };
  }

  const W = clone(Ain);            // m×n, becomes U·Σ
  const V = eye(n);

  for (let sweep = 0; sweep < iters; sweep++) {
    let off = 0;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        let alpha = 0, beta = 0, gamma = 0;
        for (let i = 0; i < m; i++) {
          const wp = W[i][p], wq = W[i][q];
          alpha += wp * wp; beta += wq * wq; gamma += wp * wq;
        }
        if (Math.abs(gamma) <= tol * Math.sqrt(alpha * beta) || gamma === 0) continue;
        off += gamma * gamma;

        const zeta = (beta - alpha) / (2 * gamma);
        const t = Math.sign(zeta || 1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        const c = 1 / Math.sqrt(1 + t * t), s = c * t;

        for (let i = 0; i < m; i++) {
          const wp = W[i][p], wq = W[i][q];
          W[i][p] = c * wp - s * wq;
          W[i][q] = s * wp + c * wq;
        }
        for (let i = 0; i < n; i++) {
          const vp = V[i][p], vq = V[i][q];
          V[i][p] = c * vp - s * vq;
          V[i][q] = s * vp + c * vq;
        }
      }
    }
    if (off < tol) break;
  }

  // singular values are the column norms of W; U is W with columns normalised
  const idx = [];
  const sv = [];
  for (let j = 0; j < n; j++) {
    let sum = 0;
    for (let i = 0; i < m; i++) sum += W[i][j] * W[i][j];
    sv.push(Math.sqrt(sum));
    idx.push(j);
  }
  idx.sort((a, b) => sv[b] - sv[a]);

  const s = idx.map(j => sv[j]);
  const U = zeros(m, n);
  const Vs = zeros(n, n);
  idx.forEach((j, k) => {
    const sk = sv[j];
    for (let i = 0; i < m; i++) U[i][k] = sk > 1e-13 ? W[i][j] / sk : 0;
    for (let i = 0; i < n; i++) Vs[i][k] = V[i][j];
  });

  // fill any zero U columns with something orthonormal-ish so reconstruction is safe
  for (let k = 0; k < n; k++) {
    if (s[k] > 1e-13) continue;
    for (let i = 0; i < m; i++) U[i][k] = 0;
  }

  return { U, s, V: Vs };
}

/** Rank-k reconstruction from a precomputed SVD. */
export function reconstruct({ U, s, V }, k) {
  const m = U.length, n = V.length;
  const out = zeros(m, n);
  const kk = Math.min(k, s.length);
  for (let t = 0; t < kk; t++) {
    const st = s[t];
    if (st <= 1e-13) continue;
    for (let i = 0; i < m; i++) {
      const ui = U[i][t] * st;
      if (ui === 0) continue;
      for (let j = 0; j < n; j++) out[i][j] += ui * V[j][t];
    }
  }
  return out;
}

/* ============================================================
   Symmetric eigendecomposition — cyclic Jacobi.
   Returns { values (desc), vectors (columns) }.
   ============================================================ */
export function eigSym(Ain, { iters = 100, tol = 1e-12 } = {}) {
  const A = clone(Ain);
  const n = A.length;
  let V = eye(n);

  for (let sweep = 0; sweep < iters; sweep++) {
    let off = 0;
    for (let p = 0; p < n - 1; p++)
      for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
    if (off < tol) break;

    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-15) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }

  const pairs = A.map((r, i) => ({ v: r[i], i }))
                 .sort((a, b) => b.v - a.v);
  return {
    values: pairs.map(p => p.v),
    vectors: pairs.map(p => V.map(row => row[p.i])),  // vectors[k] = k-th eigenvector
  };
}

/** Determinant via LU with partial pivoting. */
export function det(Ain) {
  const A = clone(Ain), n = A.length;
  let d = 1;
  for (let i = 0; i < n; i++) {
    let piv = i, best = Math.abs(A[i][i]);
    for (let r = i + 1; r < n; r++) {
      if (Math.abs(A[r][i]) > best) { best = Math.abs(A[r][i]); piv = r; }
    }
    if (best < 1e-14) return 0;
    if (piv !== i) { [A[i], A[piv]] = [A[piv], A[i]]; d = -d; }
    d *= A[i][i];
    for (let r = i + 1; r < n; r++) {
      const f = A[r][i] / A[i][i];
      for (let c = i; c < n; c++) A[r][c] -= f * A[i][c];
    }
  }
  return d;
}

/** Inverse via Gauss–Jordan. Returns null when singular. */
export function inv(Ain) {
  const n = Ain.length;
  const M = Ain.map((r, i) => [...r, ...eye(n)[i]]);
  const { R, rank: rk } = rref(M);
  if (rk < n) return null;
  return R.map(r => r.slice(n));
}

/** Solve Ax = b (square, nonsingular). Returns null when singular. */
export function solve(A, b) {
  const M = A.map((r, i) => [...r, b[i]]);
  const { R, pivots } = rref(M);
  const n = A[0].length;
  if (pivots.length < n || pivots.includes(n)) return null;
  return R.slice(0, n).map(r => r[n]);
}

/** Gram–Schmidt: orthonormal basis for the column space of A. */
export function gramSchmidt(A, tol = 1e-10) {
  const [m, n] = shape(A);
  const Q = [];
  for (let j = 0; j < n; j++) {
    let v = A.map(r => r[j]);
    for (const q of Q) {
      const d = dot(v, q);
      v = v.map((vi, i) => vi - d * q[i]);
    }
    const nv = norm2(v);
    if (nv > tol) Q.push(v.map(vi => vi / nv));
  }
  return Q;   // array of column vectors
}
