/* ============================================================
   viz.js — shared canvas plotting + interaction library
   Theme-aware, DPI-aware, pointer-driven. No dependencies.
   ============================================================ */

/* ---------- theme colour access ---------- */
const _cache = { theme: null, vals: {} };

export function css(name) {
  const theme = document.documentElement.dataset.theme || 'light';
  if (_cache.theme !== theme) { _cache.theme = theme; _cache.vals = {}; }
  if (!(name in _cache.vals)) {
    _cache.vals[name] = getComputedStyle(document.documentElement)
      .getPropertyValue(name).trim();
  }
  return _cache.vals[name];
}
export function invalidateThemeCache() { _cache.theme = null; _cache.vals = {}; }

export const C = {
  get grid()  { return css('--plot-grid'); },
  get axis()  { return css('--plot-axis'); },
  get ink()   { return css('--plot-ink'); },
  get muted() { return css('--plot-muted'); },
  get c1()    { return css('--plot-c1'); },
  get c2()    { return css('--plot-c2'); },
  get c3()    { return css('--plot-c3'); },
  get c4()    { return css('--plot-c4'); },
  get c5()    { return css('--plot-c5'); },
  get c6()    { return css('--plot-c6'); },
  get fill()  { return css('--plot-fill'); },
  get fill2() { return css('--plot-fill2'); },
  get bg()    { return css('--bg-sunken'); },
  get raised(){ return css('--bg-raised'); },
};

/* ---------- small maths helpers ---------- */
export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
export const lerp  = (a, b, t) => a + (b - a) * t;
export const round = (v, d = 2) => {
  const p = 10 ** d;
  const r = Math.round(v * p) / p;
  return Object.is(r, -0) ? 0 : r;
};
export const fmt = (v, d = 2) => {
  const r = round(v, d);
  return Number.isInteger(r) ? String(r) : r.toFixed(d);
};
export const deg = r => r * 180 / Math.PI;
export const rad = d => d * Math.PI / 180;

/* 2x2 matrix helpers ------------------------------------------------ */
export const mat = {
  apply: (m, [x, y]) => [m[0] * x + m[1] * y, m[2] * x + m[3] * y],
  det:   m => m[0] * m[3] - m[1] * m[2],
  mul:   (a, b) => [
    a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3],
    a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[3],
  ],
  transpose: m => [m[0], m[2], m[1], m[3]],
  /* Real eigenvalues/vectors of a 2x2. Returns null when complex. */
  eig(m) {
    const [a, b, c, d] = m;
    const tr = a + d, dt = a * d - b * c;
    const disc = tr * tr / 4 - dt;
    if (disc < -1e-12) return null;                 // complex pair
    const s = Math.sqrt(Math.max(0, disc));
    const l1 = tr / 2 + s, l2 = tr / 2 - s;
    const vecFor = l => {
      // (A - lI)v = 0  →  pick whichever row gives a stable direction
      let v;
      if (Math.abs(b) > 1e-9)       v = [b, l - a];
      else if (Math.abs(c) > 1e-9)  v = [l - d, c];
      else                          v = Math.abs(a - l) < 1e-9 ? [1, 0] : [0, 1];
      const n = Math.hypot(v[0], v[1]);
      return n < 1e-12 ? [1, 0] : [v[0] / n, v[1] / n];
    };
    return { l1, l2, v1: vecFor(l1), v2: vecFor(l2), repeated: s < 1e-7 };
  },
  /* SVD of a 2x2 via eigen-decomposition of AᵀA. */
  svd(m) {
    const [a, b, c, d] = m;
    // AᵀA
    const p = a * a + c * c, q = a * b + c * d, r = b * b + d * d;
    const tr = p + r, dt = p * r - q * q;
    const disc = Math.max(0, tr * tr / 4 - dt);
    const s = Math.sqrt(disc);
    const e1 = Math.max(0, tr / 2 + s), e2 = Math.max(0, tr / 2 - s);
    const s1 = Math.sqrt(e1), s2 = Math.sqrt(e2);
    // right singular vectors = eigenvectors of AᵀA
    let v1;
    if (Math.abs(q) > 1e-10)      v1 = [q, e1 - p];
    else                          v1 = p >= r ? [1, 0] : [0, 1];
    const n1 = Math.hypot(v1[0], v1[1]) || 1;
    v1 = [v1[0] / n1, v1[1] / n1];
    const v2 = [-v1[1], v1[0]];
    // left singular vectors: u = Av/σ
    const av1 = mat.apply(m, v1), av2 = mat.apply(m, v2);
    const u1 = s1 > 1e-10 ? [av1[0] / s1, av1[1] / s1] : [1, 0];
    let u2 = s2 > 1e-10 ? [av2[0] / s2, av2[1] / s2] : [-u1[1], u1[0]];
    return { s1, s2, v1, v2, u1, u2 };
  },
};

/* ============================================================
   Plot — a responsive, DPI-correct 2D canvas with world coords
   ============================================================ */
export class Plot {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {{xmin,xmax,ymin,ymax,aspect?,pad?,equal?}} opts
   */
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.o = Object.assign(
      { xmin: -5, xmax: 5, ymin: -5, ymax: 5, aspect: 1, equal: true, pad: 0 },
      opts
    );
    this.w = 0; this.h = 0; this.dpr = 1;
    this._draw = null;
    this._raf = null;

    this._ro = new ResizeObserver(() => this.resize());
    this._ro.observe(canvas.parentElement || canvas);
    this.resize();
  }

  resize() {
    const host = this.canvas.parentElement || this.canvas;
    const cssW = Math.max(1, host.clientWidth);
    // Height follows the aspect ratio only. Stretching a canvas to fill a tall
    // controls column distorts every figure drawn with equal: false, so the
    // dead space is absorbed by centring the canvas in its column (see
    // .pg-split in components.css) rather than by resizing it.
    const cssH = Math.max(1, Math.round(cssW / this.o.aspect));
    this.dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    this.canvas.style.height = cssH + 'px';
    this.canvas.width  = Math.round(cssW * this.dpr);
    this.canvas.height = Math.round(cssH * this.dpr);
    this.w = cssW; this.h = cssH;
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this._computeScale();
    this.render();
  }

  _computeScale() {
    const { xmin, xmax, ymin, ymax, pad, equal } = this.o;
    const iw = this.w - 2 * pad, ih = this.h - 2 * pad;
    let sx = iw / (xmax - xmin);
    let sy = ih / (ymax - ymin);
    if (equal) { const s = Math.min(sx, sy); sx = sy = s; }
    this.sx = sx; this.sy = sy;
    // centre the world box inside the canvas
    this.ox = pad + (iw - (xmax - xmin) * sx) / 2 - xmin * sx;
    this.oy = pad + (ih - (ymax - ymin) * sy) / 2 + ymax * sy;
  }

  setBounds(b) { Object.assign(this.o, b); this._computeScale(); this.render(); }

  /* coordinate conversion */
  X(x) { return this.ox + x * this.sx; }
  Y(y) { return this.oy - y * this.sy; }
  toScreen([x, y]) { return [this.X(x), this.Y(y)]; }
  toWorld(px, py)  { return [(px - this.ox) / this.sx, (this.oy - py) / this.sy]; }
  /** pointer event → world coords */
  eventWorld(ev) {
    const r = this.canvas.getBoundingClientRect();
    return this.toWorld(ev.clientX - r.left, ev.clientY - r.top);
  }
  /**
   * World length -> pixels.
   * px() uses the horizontal scale and py() the vertical one. They are only
   * interchangeable when the plot was built with equal: true; otherwise
   * sizing a box's height with px() stretches it by sx/sy, which makes
   * tiled cells overlap their neighbours.
   */
  px(len) { return len * this.sx; }
  py(len) { return len * this.sy; }

  /* render plumbing */
  onDraw(fn) { this._draw = fn; this.render(); return this; }
  render() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => {
      this._raf = null;
      if (this._draw) { this.clear(); this._draw(this); }
    });
  }
  renderNow() { if (this._draw) { this.clear(); this._draw(this); } }

  clear(color) {
    const { ctx } = this;
    ctx.save();
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);
    if (color !== null) { ctx.fillStyle = color || C.bg; ctx.fillRect(0, 0, this.w, this.h); }
    ctx.restore();
  }

  destroy() { this._ro.disconnect(); if (this._raf) cancelAnimationFrame(this._raf); }

  /* ---------- drawing primitives ---------- */

  grid(step = 1, opts = {}) {
    const { ctx, o } = this;
    ctx.save();
    ctx.strokeStyle = opts.color || C.grid;
    ctx.lineWidth = opts.lw || 1;
    ctx.beginPath();
    const x0 = Math.ceil(o.xmin / step) * step;
    for (let x = x0; x <= o.xmax + 1e-9; x += step) {
      const px = Math.round(this.X(x)) + .5;
      ctx.moveTo(px, this.Y(o.ymax)); ctx.lineTo(px, this.Y(o.ymin));
    }
    const y0 = Math.ceil(o.ymin / step) * step;
    for (let y = y0; y <= o.ymax + 1e-9; y += step) {
      const py = Math.round(this.Y(y)) + .5;
      ctx.moveTo(this.X(o.xmin), py); ctx.lineTo(this.X(o.xmax), py);
    }
    ctx.stroke();
    ctx.restore();
  }

  axes(opts = {}) {
    const { ctx, o } = this;
    ctx.save();
    ctx.strokeStyle = opts.color || C.axis;
    ctx.lineWidth = opts.lw || 1.4;
    ctx.beginPath();
    if (o.ymin <= 0 && o.ymax >= 0) {
      const py = Math.round(this.Y(0)) + .5;
      ctx.moveTo(this.X(o.xmin), py); ctx.lineTo(this.X(o.xmax), py);
    }
    if (o.xmin <= 0 && o.xmax >= 0) {
      const px = Math.round(this.X(0)) + .5;
      ctx.moveTo(px, this.Y(o.ymin)); ctx.lineTo(px, this.Y(o.ymax));
    }
    ctx.stroke();
    ctx.restore();
    if (opts.ticks) this.ticks(opts.ticks === true ? 1 : opts.ticks, opts);
  }

  /**
   * Axis tick labels.
   *
   * Labels are anchored to the axis, but the axis is often flush with the
   * canvas edge (any plot built with pad: 0 whose range starts at 0), which
   * would push the labels off-canvas entirely. Every label is therefore kept
   * inside the drawing area: it flips to the opposite side of the axis when
   * there is no room, and is clamped along its own axis so the first and last
   * are never half-cut.
   */
  ticks(step = 1, opts = {}) {
    const { ctx, o } = this;
    // x and y often need different spacing (a bar chart's value axis has
    // nothing to do with its category axis), so each may be overridden
    const stepX = opts.stepX ?? step;
    const stepY = opts.stepY ?? step;
    if (!(stepX > 0) || !Number.isFinite(stepX) || !(stepY > 0) || !Number.isFinite(stepY)) return;
    const size = opts.size || 11;
    ctx.save();
    ctx.fillStyle = opts.color || C.muted;
    ctx.font = `500 ${size}px ${css('--font-sans')}`;
    // Tick labels are chrome and frequently land on grid lines, curves or a
    // shaded region. A halo in the panel colour keeps them readable without
    // hiding anything underneath.
    ctx.strokeStyle = C.bg;
    ctx.lineWidth = 3;
    ctx.lineJoin = 'round';
    const put = (t, x, y) => { ctx.strokeText(t, x, y); ctx.fillText(t, x, y); };
    const M = 3;                                   // keep this far from the edge
    // space claimed by a bottom caption, so ticks never sit on top of it
    const B = opts.bottom ?? this.reserveBottom ?? 0;
    const T = opts.top ?? this.reserveTop ?? 0;

    // ---- x labels, normally below the horizontal axis ----
    ctx.textAlign = 'center';
    let lastX = -Infinity, lastY = -Infinity;
    const yBase = clamp(0, o.ymin, o.ymax);
    const axisY = this.Y(yBase);
    const below = axisY + 5 + size <= this.h - M - B;
    // when x labels flip above the axis they must clear the title band too
    ctx.textBaseline = below ? 'top' : 'bottom';
    // flipping above the axis must still clear the caption band, otherwise
    // an axis flush with the bottom edge puts labels straight onto the caption
    const ly = below
      ? Math.min(axisY + 5, this.h - M - B - size)
      : clamp(Math.min(axisY - 5, this.h - B - 4), T + M + size, this.h - M);
    for (let x = Math.ceil(o.xmin / stepX) * stepX; x <= o.xmax + 1e-9; x += stepX) {
      if (Math.abs(x) < 1e-9) continue;
      const label = fmt(x, 2);
      const nat = this.X(x);
      if (nat < -1 || nat > this.w + 1) continue;      // off the plot entirely: drop it
      const half = ctx.measureText(label).width / 2;
      const cx = clamp(nat, M + half, this.w - M - half);
      // never let labels pile up: a step suited to one axis can be far too
      // fine for the other, which would stack hundreds of labels on a pixel
      if (cx - half < lastX + 6) continue;
      lastX = cx + half;
      put(label, cx, ly);
    }

    // ---- y labels, normally left of the vertical axis ----
    const xBase = clamp(0, o.xmin, o.xmax);
    const axisX = this.X(xBase);
    let widest = 0;
    for (let y = Math.ceil(o.ymin / stepY) * stepY; y <= o.ymax + 1e-9; y += stepY) {
      if (Math.abs(y) < 1e-9) continue;
      const ny = this.Y(y);
      if (ny < -1 || ny > this.h + 1) continue;
      widest = Math.max(widest, ctx.measureText(fmt(y, 2)).width);
    }
    const leftFits = axisX - 6 - widest >= M;
    ctx.textAlign = leftFits ? 'right' : 'left';
    ctx.textBaseline = 'middle';
    const lx = leftFits ? Math.max(axisX - 6, M + widest) : Math.min(axisX + 6, this.w - M - widest);
    for (let y = Math.ceil(o.ymin / stepY) * stepY; y <= o.ymax + 1e-9; y += stepY) {
      if (Math.abs(y) < 1e-9) continue;
      const nat = this.Y(y);
      if (nat < -1 || nat > this.h + 1) continue;      // off the plot entirely: drop it
      const cy = clamp(nat, M + size / 2, this.h - M - size / 2);
      if (Math.abs(cy - lastY) < size + 2) continue;
      // When there is no gutter the labels sit inside the plot, where the
      // top-left corner is conventionally used for the figure's caption.
      // Dropping one tick there beats printing two strings on top of each other.
      if (cy - size / 2 < T + 2) continue;
      if (cy + size / 2 > this.h - B - 2) continue;
      if (!leftFits && cy < (opts.topCaption ?? 30)) continue;
      lastY = cy;
      put(fmt(y, 2), lx, cy);
    }
    ctx.restore();
  }

  /**
   * Caption centred along the bottom edge, always fully inside the canvas.
   * Use instead of hand-positioning text at `py: h - 4`, which clips descenders.
   */
  /**
   * Caption along the top edge. Claims a band that ticks() then avoids, which
   * is the only reliable way to stop axis labels landing on it — nudging the
   * bounds just moves a different tick into the same place.
   */
  title(str, opts = {}) {
    const size = opts.size || 11;
    const need = size + 8;
    if (!this.reserveTop || this.reserveTop < need) {
      this.reserveTop = need;
      this.render();
    }
    const align = opts.align || 'left';
    const px = align === 'right' ? this.w - (opts.margin ?? 10)
      : align === 'center' ? this.w / 2 : (opts.margin ?? 10);
    this.text({ px, py: 4 }, str,
      { align, baseline: 'top', size, color: C.muted, weight: 500, ...opts });
  }

  xlabel(str, opts = {}) {
    const need = (opts.size || 10.5) + 6;
    // ticks() runs before this on the same frame, so the very first paint would
    // not yet know to leave room. Claim the space and repaint once.
    if (!this.reserveBottom || this.reserveBottom < need) {
      this.reserveBottom = need;
      this.render();
    }
    this.text({ px: this.w / 2, py: this.h - 3 }, str,
      { align: 'center', baseline: 'bottom', size: 10.5, color: C.muted, weight: 500, ...opts });
  }

  line(a, b, opts = {}) {
    const { ctx } = this;
    ctx.save();
    ctx.strokeStyle = opts.color || C.ink;
    ctx.lineWidth = opts.lw || 2;
    ctx.lineCap = opts.cap || 'round';
    if (opts.dash) ctx.setLineDash(opts.dash);
    if (opts.alpha != null) ctx.globalAlpha = opts.alpha;
    ctx.beginPath();
    ctx.moveTo(...this.toScreen(a));
    ctx.lineTo(...this.toScreen(b));
    ctx.stroke();
    ctx.restore();
  }

  /** Infinite line through point p with direction d, clipped to the view. */
  ray(p, d, opts = {}) {
    const n = Math.hypot(d[0], d[1]) || 1;
    const u = [d[0] / n, d[1] / n];
    const big = (this.o.xmax - this.o.xmin + this.o.ymax - this.o.ymin) * 2;
    this.line(
      [p[0] - u[0] * big, p[1] - u[1] * big],
      [p[0] + u[0] * big, p[1] + u[1] * big],
      opts
    );
  }

  arrow(from, to, opts = {}) {
    const { ctx } = this;
    const [x1, y1] = this.toScreen(from);
    const [x2, y2] = this.toScreen(to);
    const dx = x2 - x1, dy = y2 - y1;
    const len = Math.hypot(dx, dy);
    const color = opts.color || C.c1;
    const lw = opts.lw || 2.6;
    ctx.save();
    ctx.strokeStyle = color; ctx.fillStyle = color;
    ctx.lineWidth = lw; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    if (opts.dash) ctx.setLineDash(opts.dash);
    if (opts.alpha != null) ctx.globalAlpha = opts.alpha;

    if (len < 1.2) { ctx.restore(); return; }
    const head = Math.min(opts.head || 11, len * 0.42);
    const ux = dx / len, uy = dy / len;
    const bx = x2 - ux * head * 0.86, by = y2 - uy * head * 0.86;

    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(bx, by); ctx.stroke();
    ctx.setLineDash([]);
    const wing = head * 0.52;
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(bx - uy * wing, by + ux * wing);
    ctx.lineTo(bx + uy * wing, by - ux * wing);
    ctx.closePath(); ctx.fill();
    ctx.restore();
  }

  dot(p, opts = {}) {
    const { ctx } = this;
    const [x, y] = this.toScreen(p);
    ctx.save();
    if (opts.alpha != null) ctx.globalAlpha = opts.alpha;
    ctx.beginPath();
    ctx.arc(x, y, opts.r || 4, 0, Math.PI * 2);
    ctx.fillStyle = opts.color || C.c1;
    ctx.fill();
    if (opts.ring) {
      ctx.lineWidth = opts.ringLw || 2.5;
      ctx.strokeStyle = opts.ring === true ? C.raised : opts.ring;
      ctx.stroke();
    }
    ctx.restore();
  }

  /** Draggable-looking handle. */
  handle(p, opts = {}) {
    const { ctx } = this;
    const [x, y] = this.toScreen(p);
    const r = opts.r || 7;
    ctx.save();
    if (opts.glow) {
      ctx.beginPath(); ctx.arc(x, y, r + 6, 0, Math.PI * 2);
      ctx.fillStyle = opts.color || C.c1; ctx.globalAlpha = .16; ctx.fill();
      ctx.globalAlpha = 1;
    }
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = opts.color || C.c1; ctx.fill();
    ctx.lineWidth = 2.5; ctx.strokeStyle = C.raised; ctx.stroke();
    ctx.restore();
  }

  path(pts, opts = {}) {
    if (!pts.length) return;
    const { ctx } = this;
    ctx.save();
    ctx.strokeStyle = opts.color || C.c1;
    ctx.lineWidth = opts.lw || 2.2;
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    if (opts.dash) ctx.setLineDash(opts.dash);
    if (opts.alpha != null) ctx.globalAlpha = opts.alpha;
    ctx.beginPath();
    pts.forEach((p, i) => {
      const [x, y] = this.toScreen(p);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    if (opts.close) ctx.closePath();
    if (opts.fill) { ctx.fillStyle = opts.fill; ctx.fill(); }
    if (opts.stroke !== false) ctx.stroke();
    ctx.restore();
  }

  polygon(pts, opts = {}) { this.path(pts, { close: true, ...opts }); }

  /** Plot y=f(x) sampled across the visible range (or [a,b]). */
  fn(f, opts = {}) {
    const a = opts.from ?? this.o.xmin, b = opts.to ?? this.o.xmax;
    const n = opts.samples || 320;
    const pts = [];
    for (let i = 0; i <= n; i++) {
      const x = a + (b - a) * i / n;
      const y = f(x);
      if (Number.isFinite(y)) pts.push([x, y]);
    }
    this.path(pts, opts);
    return pts;
  }

  ellipse(center, rx, ry, rot = 0, opts = {}) {
    const { ctx } = this;
    const [cx, cy] = this.toScreen(center);
    ctx.save();
    ctx.beginPath();
    ctx.ellipse(cx, cy, Math.abs(this.px(rx)), Math.abs(ry * this.sy), -rot, 0, Math.PI * 2);
    if (opts.fill) { ctx.fillStyle = opts.fill; ctx.fill(); }
    if (opts.color) {
      ctx.strokeStyle = opts.color; ctx.lineWidth = opts.lw || 2;
      if (opts.dash) ctx.setLineDash(opts.dash);
      ctx.stroke();
    }
    ctx.restore();
  }

  text(p, str, opts = {}) {
    const { ctx } = this;
    let [x, y] = Array.isArray(p) ? this.toScreen(p) : [p.px, p.py];
    x += opts.dx || 0; y += opts.dy || 0;
    ctx.save();
    ctx.font = `${opts.weight || 600} ${opts.size || 12}px ${opts.mono ? css('--font-mono') : css('--font-sans')}`;
    ctx.textAlign = opts.align || 'left';
    ctx.textBaseline = opts.baseline || 'middle';
    if (opts.halo) {
      ctx.lineWidth = opts.haloWidth || 3.5;
      ctx.strokeStyle = opts.halo === true ? C.bg : opts.halo;
      ctx.lineJoin = 'round';
      ctx.strokeText(str, x, y);
    }
    ctx.fillStyle = opts.color || C.ink;
    ctx.fillText(str, x, y);
    ctx.restore();
  }

  /**
   * Keyed legend on a solid panel, so curves can pass behind it without
   * colliding with the text. `items` is [[colour, label], ...].
   * `corner` is 'tl' | 'tr' | 'bl' | 'br'; `title` is an optional caption line.
   */
  legend(items, opts = {}) {
    const { ctx } = this;
    const size = opts.size || 10.5;
    const rowH = size + 4.5;
    const padX = 9, padY = 7, swatch = 18, gap = 7;
    ctx.save();
    ctx.font = `600 ${size}px ${css('--font-sans')}`;
    let w = 0;
    items.forEach(([, l]) => { w = Math.max(w, ctx.measureText(l).width); });
    let titleW = 0;
    if (opts.title) {
      ctx.font = `500 ${size}px ${css('--font-sans')}`;
      titleW = ctx.measureText(opts.title).width;
    }
    const bw = Math.max(swatch + gap + w, titleW) + padX * 2;
    const bh = items.length * rowH + (opts.title ? rowH + 2 : 0) + padY * 2 - 4.5;
    const m = opts.margin ?? 10;
    const corner = opts.corner || 'tl';
    const bx = corner[1] === 'l' ? m : this.w - bw - m;
    const by = corner[0] === 't' ? m : this.h - bh - m;

    const r = 7;
    ctx.beginPath();
    ctx.moveTo(bx + r, by);
    ctx.arcTo(bx + bw, by, bx + bw, by + bh, r);
    ctx.arcTo(bx + bw, by + bh, bx, by + bh, r);
    ctx.arcTo(bx, by + bh, bx, by, r);
    ctx.arcTo(bx, by, bx + bw, by, r);
    ctx.closePath();
    ctx.fillStyle = opts.bg || C.raised;
    ctx.globalAlpha = opts.alpha ?? .94;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1; ctx.stroke();

    let y = by + padY + size * .5;
    if (opts.title) {
      ctx.font = `500 ${size}px ${css('--font-sans')}`;
      ctx.fillStyle = C.muted; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.fillText(opts.title, bx + padX, y);
      y += rowH + 2;
    }
    ctx.font = `600 ${size}px ${css('--font-sans')}`;
    items.forEach(([col, label, dash]) => {
      ctx.strokeStyle = col; ctx.lineWidth = 2.6;
      ctx.setLineDash(dash || []);
      ctx.beginPath();
      ctx.moveTo(bx + padX, y); ctx.lineTo(bx + padX + swatch, y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = C.muted; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.fillText(label, bx + padX + swatch + gap, y);
      y += rowH;
    });
    ctx.restore();
  }

  /** Small pill-shaped label with background — good for annotations. */
  badge(p, str, opts = {}) {
    const { ctx } = this;
    let [x, y] = this.toScreen(p);
    x += opts.dx || 0; y += opts.dy || 0;
    ctx.save();
    ctx.font = `650 ${opts.size || 11}px ${css('--font-sans')}`;
    const w = ctx.measureText(str).width;
    const padX = 6, padY = 3.5, h = (opts.size || 11) + padY * 2;
    let bx = x, by = y - h / 2;
    if (opts.align === 'center') bx = x - w / 2 - padX;
    else if (opts.align === 'right') bx = x - w - padX * 2;
    const bw = w + padX * 2;
    // A badge anchored to a data point drifts off-canvas as that point moves.
    // Keep the whole pill inside so the label is never half-cut.
    if (opts.clamp !== false) {
      bx = clamp(bx, 2, Math.max(2, this.w - bw - 2));
      by = clamp(by, 2, Math.max(2, this.h - h - 2));
    }
    const r = 5;
    ctx.beginPath();
    ctx.moveTo(bx + r, by);
    ctx.arcTo(bx + bw, by, bx + bw, by + h, r);
    ctx.arcTo(bx + bw, by + h, bx, by + h, r);
    ctx.arcTo(bx, by + h, bx, by, r);
    ctx.arcTo(bx, by, bx + bw, by, r);
    ctx.closePath();
    ctx.fillStyle = opts.bg || C.raised;
    ctx.fill();
    if (opts.border !== false) {
      ctx.strokeStyle = opts.color || C.c1; ctx.lineWidth = 1.2; ctx.stroke();
    }
    ctx.fillStyle = opts.color || C.ink;
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText(str, bx + padX, by + h / 2 + .5);
    ctx.restore();
  }

  /** Right-angle marker at corner b between rays b→a and b→c. */
  rightAngle(a, b, c, opts = {}) {
    const s = opts.size || 11;
    const [bx, by] = this.toScreen(b);
    const [ax, ay] = this.toScreen(a);
    const [cx, cy] = this.toScreen(c);
    const n1 = Math.hypot(ax - bx, ay - by) || 1;
    const n2 = Math.hypot(cx - bx, cy - by) || 1;
    const u = [(ax - bx) / n1 * s, (ay - by) / n1 * s];
    const v = [(cx - bx) / n2 * s, (cy - by) / n2 * s];
    const { ctx } = this;
    ctx.save();
    ctx.strokeStyle = opts.color || C.muted;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(bx + u[0], by + u[1]);
    ctx.lineTo(bx + u[0] + v[0], by + u[1] + v[1]);
    ctx.lineTo(bx + v[0], by + v[1]);
    ctx.stroke();
    ctx.restore();
  }

  /** Draw the image of the unit grid under a 2x2 matrix. */
  transformedGrid(m, opts = {}) {
    const range = opts.range || 6;
    const step = opts.step || 1;
    const seg = opts.seg || 24;
    const color = opts.color || C.c1;
    for (let i = -range; i <= range; i += step) {
      const h = [], v = [];
      for (let k = 0; k <= seg; k++) {
        const t = -range + (2 * range) * k / seg;
        h.push(mat.apply(m, [t, i]));
        v.push(mat.apply(m, [i, t]));
      }
      const major = Math.abs(i) < 1e-9;
      const o = {
        color, lw: major ? 2 : 1.1,
        alpha: major ? .95 : (opts.alpha ?? .45),
      };
      this.path(h, o); this.path(v, o);
    }
  }
}

/* ============================================================
   Dragging — register world-space handles on a Plot
   ============================================================ */
export class Dragger {
  /** @param {Plot} plot */
  constructor(plot) {
    this.plot = plot;
    this.handles = [];        // {get, set, r, id}
    this.active = null;
    this.hover = null;
    this._bind();
  }

  /** get: () => [x,y]  |  set: ([x,y]) => void  */
  add(get, set, opts = {}) {
    const h = { get, set, r: opts.r || 14, id: opts.id, enabled: opts.enabled || (() => true) };
    this.handles.push(h);
    return h;
  }
  clear() { this.handles.length = 0; }

  _pick(ev) {
    const r = this.plot.canvas.getBoundingClientRect();
    const px = ev.clientX - r.left, py = ev.clientY - r.top;
    let best = null, bestD = Infinity;
    for (const h of this.handles) {
      if (!h.enabled()) continue;
      const [hx, hy] = this.plot.toScreen(h.get());
      const d = Math.hypot(px - hx, py - hy);
      if (d < h.r && d < bestD) { best = h; bestD = d; }
    }
    return best;
  }

  _bind() {
    const cv = this.plot.canvas;

    const down = ev => {
      const h = this._pick(ev);
      if (!h) return;
      ev.preventDefault();
      this.active = h;
      cv.setPointerCapture?.(ev.pointerId);
      cv.classList.add('is-grabbing');
      h.set(this.plot.eventWorld(ev), true);
      this.plot.render();
      this.onchange?.();
    };

    const move = ev => {
      if (this.active) {
        ev.preventDefault();
        this.active.set(this.plot.eventWorld(ev), false);
        this.plot.render();
        this.onchange?.();
        return;
      }
      const h = this._pick(ev);
      const hovering = !!h;
      if (hovering !== this.hover) {
        this.hover = hovering;
        cv.classList.toggle('is-grabbable', hovering);
      }
    };

    const up = ev => {
      if (!this.active) return;
      this.active = null;
      cv.releasePointerCapture?.(ev.pointerId);
      cv.classList.remove('is-grabbing');
      this.onend?.();
    };

    cv.addEventListener('pointerdown', down);
    cv.addEventListener('pointermove', move);
    cv.addEventListener('pointerup', up);
    cv.addEventListener('pointercancel', up);
    cv.addEventListener('pointerleave', () => {
      if (!this.active) { this.hover = false; cv.classList.remove('is-grabbable'); }
    });
  }
}

/* ============================================================
   Control builders — sliders, toggles, segmented, matrix inputs
   ============================================================ */

export function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === 'class') n.className = v;
    else if (k === 'html') n.innerHTML = v;
    else if (k === 'text') n.textContent = v;
    else if (k.startsWith('on') && typeof v === 'function') n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) {
    if (kid == null) continue;
    n.appendChild(typeof kid === 'string' ? document.createTextNode(kid) : kid);
  }
  return n;
}

/**
 * Slider control.
 * @returns {{root:HTMLElement, input:HTMLInputElement, get:()=>number, set:(v:number)=>void}}
 */
export function slider(label, { min, max, step = .01, value, format = v => fmt(v, 2), onInput }) {
  const val = el('span', { class: 'ctrl__value', text: format(value) });
  const input = el('input', {
    type: 'range', min, max, step, value,
    // Without this the browser restores the previous session's position on
    // reload, so a shared link would not reproduce the figure it describes.
    autocomplete: 'off',
    'aria-label': label,
  });
  const api = {
    get: () => +input.value,
    set: v => { input.value = v; val.textContent = format(+input.value); },
    input, val,
  };
  input.addEventListener('input', () => {
    val.textContent = format(+input.value);
    onInput?.(+input.value);
  });
  api.root = el('label', { class: 'ctrl' },
    el('span', { class: 'ctrl__label' }, el('span', { html: label }), val),
    input
  );
  return api;
}

export function toggle(label, { value = false, onChange }) {
  const input = el('input', { type: 'checkbox', checked: value });
  input.addEventListener('change', () => onChange?.(input.checked));
  const root = el('label', { class: 'switch-row' },
    el('span', { html: label }), input, el('span', { class: 'switch' })
  );
  return { root, input, get: () => input.checked, set: v => { input.checked = v; } };
}

export function segmented(options, { value, onChange, label } = {}) {
  const btns = options.map(o =>
    el('button', {
      type: 'button', 'aria-pressed': String(o.value === value), 'data-v': o.value,
      html: o.label,
      onclick: () => { api.set(o.value); onChange?.(o.value); },
    })
  );
  const seg = el('div', { class: 'seg', role: 'group', 'aria-label': label || 'options' }, btns);
  const api = {
    root: label ? el('div', { class: 'ctrl' },
            el('span', { class: 'ctrl__label' }, el('span', { html: label })), seg)
          : seg,
    seg,
    get: () => btns.find(b => b.getAttribute('aria-pressed') === 'true')?.dataset.v,
    set: v => btns.forEach(b => b.setAttribute('aria-pressed', String(b.dataset.v === String(v)))),
  };
  return api;
}

export function button(label, onClick, { variant = 'ghost', icon } = {}) {
  return el('button', {
    type: 'button', class: `btn btn--${variant} btn--sm`, onclick: onClick,
    html: (icon || '') + label,
  });
}

/**
 * Editable matrix of numbers.
 * @returns {{root, get:()=>number[][], set:(m)=>void, cells:HTMLInputElement[][]}}
 */
export function matrixInput(rows, cols, values, { onInput, step = .1, label } = {}) {
  const cells = [];
  const grid = el('div', {
    class: 'matrix-input',
    style: `grid-template-columns: repeat(${cols}, auto)`,
  });
  for (let i = 0; i < rows; i++) {
    cells[i] = [];
    for (let j = 0; j < cols; j++) {
      const inp = el('input', {
        type: 'number', step, value: values[i][j],
        autocomplete: 'off',
        'aria-label': `row ${i + 1} column ${j + 1}`,
      });
      inp.addEventListener('input', () => onInput?.(api.get(), i, j));
      cells[i][j] = inp;
      grid.appendChild(inp);
    }
  }
  const api = {
    cells, grid,
    root: label
      ? el('div', {}, el('span', { class: 'matrix-label', html: label }), grid)
      : grid,
    get: () => cells.map(r => r.map(c => {
      const v = parseFloat(c.value);
      return Number.isFinite(v) ? v : 0;
    })),
    set: m => m.forEach((r, i) => r.forEach((v, j) => { cells[i][j].value = round(v, 3); })),
    highlight: (pairs, cls = 'is-hl') => {
      cells.flat().forEach(c => c.classList.remove('is-hl', 'is-hl2'));
      pairs?.forEach(([i, j, c]) => cells[i]?.[j]?.classList.add(c || cls));
    },
  };
  return api;
}

/** Read-only matrix display. */
export function matrixView(values, { label, fmt: f = v => fmt(v, 2) } = {}) {
  const cols = values[0]?.length || 1;
  const grid = el('div', {
    class: 'matrix-view',
    style: `grid-template-columns: repeat(${cols}, auto)`,
  });
  const spans = [];
  values.forEach((r, i) => {
    spans[i] = [];
    r.forEach((v, j) => {
      const s = el('span', { text: f(v) });
      spans[i][j] = s; grid.appendChild(s);
    });
  });
  return {
    root: label ? el('div', {}, el('span', { class: 'matrix-label', html: label }), grid) : grid,
    grid, spans,
    set: m => m.forEach((r, i) => r.forEach((v, j) => { if (spans[i]?.[j]) spans[i][j].textContent = f(v); })),
    highlight: pairs => {
      spans.flat().forEach(s => s.classList.remove('is-hl', 'is-hl2'));
      pairs?.forEach(([i, j, c]) => spans[i]?.[j]?.classList.add(c || 'is-hl'));
    },
  };
}

/** Live readout list: rows = [[label, valueFn|value], ...] */
export function readout(rows) {
  const dl = el('dl');
  const dds = [];
  rows.forEach(([k]) => {
    dl.appendChild(el('dt', { html: k }));
    const dd = el('dd', { text: '' });
    dds.push(dd); dl.appendChild(dd);
  });
  return {
    root: el('div', { class: 'readout' }, dl),
    set: vals => vals.forEach((v, i) => {
      if (dds[i]) {
        if (v && typeof v === 'object' && 'html' in v) dds[i].innerHTML = v.html;
        else dds[i].textContent = v;
        dds[i].className = (v && v.cls) || '';
      }
    }),
  };
}

export function status(initial = '', kind = '') {
  const node = el('div', { class: 'pg-status' + (kind ? ` pg-status--${kind}` : ''), html: initial });
  return {
    root: node,
    set: (html, k) => {
      node.innerHTML = html;
      node.className = 'pg-status' + (k ? ` pg-status--${k}` : '');
    },
  };
}

/* ============================================================
   Widget registration + lazy mount
   ============================================================ */
const registry = new Map();
export function defineWidget(name, mountFn) { registry.set(name, mountFn); }

/** Mount every [data-widget] element once it approaches the viewport. */
export function mountWidgets(root = document) {
  const nodes = root.querySelectorAll('[data-widget]:not([data-mounted])');
  if (!nodes.length) return;
  const io = new IntersectionObserver((entries, obs) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      obs.unobserve(e.target);
      mountOne(e.target);
    }
  }, { rootMargin: '320px 0px' });
  nodes.forEach(n => io.observe(n));
}

function mountOne(node) {
  const name = node.dataset.widget;
  const fn = registry.get(name);
  node.dataset.mounted = '1';
  if (!fn) { console.warn(`[viz] no widget registered as "${name}"`); return; }
  try {
    let opts = {};
    if (node.dataset.opts) { try { opts = JSON.parse(node.dataset.opts); } catch {} }
    fn(node, opts);
  } catch (err) {
    console.error(`[viz] widget "${name}" failed:`, err);
    node.innerHTML =
      `<div class="pg-status pg-status--warn">This interactive figure failed to load. ` +
      `Please refresh, or report it if it keeps happening.</div>`;
  }
}

/* Re-render all live plots when the theme flips. */
const livePlots = new Set();
export function trackPlot(p) { livePlots.add(p); return p; }
window.addEventListener('themechange', () => {
  invalidateThemeCache();
  // renderNow, not render: a queued rAF never fires while the tab is hidden,
  // which would leave the canvas painted in the previous theme on return.
  livePlots.forEach(p => p.renderNow());
});

/* Convenience: build a canvas inside a wrapper with an optional hint. */
export function canvasHost(parent, { aspect = 1.6, hint } = {}) {
  const cv = el('canvas');
  const wrap = el('div', { class: 'pg-canvas-wrap' }, cv);
  if (hint) {
    wrap.appendChild(el('div', { class: 'pg-hint', html:
      `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 11.5V5a1.5 1.5 0 0 1 3 0v6"/><path d="M12 11V4a1.5 1.5 0 0 1 3 0v7"/><path d="M15 11V6a1.5 1.5 0 0 1 3 0v9a6 6 0 0 1-6 6h-2a6 6 0 0 1-6-6v-3a1.5 1.5 0 0 1 3 0v1.5"/></svg>` + hint }));
  }
  parent.appendChild(wrap);
  return { canvas: cv, wrap, aspect };
}
