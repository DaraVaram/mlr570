/* ============================================================
   widgets/calc.js — Calculus, optimisation, and complexity figures
   ============================================================ */
import {
  Plot, Dragger, C, el, slider, toggle, segmented, button,
  matrixInput, readout, status, defineWidget, canvasHost,
  trackPlot, clamp, fmt, round, css,
} from '../viz.js';

function split(node, { aspect = 1.5, hint, wide = false } = {}) {
  const left = el('div');
  const right = el('div', { class: 'pg-controls' });
  node.appendChild(el('div', { class: 'pg-split' + (wide ? ' pg-split--wide-ctrl' : '') }, left, right));
  const { canvas } = canvasHost(left, { hint });
  return { left, right, canvas, aspect };
}
const note = html => el('div', { class: 'pg-note', html });

/* ---------- shared: heatmap + iso-contours for f(x, y) ---------- */
function drawField(p, f, { iso = 9, alpha = .9 } = {}) {
  const { ctx } = p;
  const step = 3;                                   // device-pixel step for speed
  const W = Math.ceil(p.w / step), H = Math.ceil(p.h / step);
  let lo = Infinity, hi = -Infinity;
  const vals = new Float64Array(W * H);
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      const [x, y] = p.toWorld(i * step, j * step);
      const v = f(x, y);
      vals[j * W + i] = v;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  const span = hi - lo || 1;
  const c1 = hexRgb(C.c1), c2 = hexRgb(C.c2);
  ctx.save();
  ctx.globalAlpha = alpha;
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      // sqrt compresses the top end so the basin stays readable
      const t = Math.sqrt(clamp((vals[j * W + i] - lo) / span, 0, 1));
      const r = Math.round(c1[0] + (c2[0] - c1[0]) * t);
      const g = Math.round(c1[1] + (c2[1] - c1[1]) * t);
      const b = Math.round(c1[2] + (c2[2] - c1[2]) * t);
      ctx.fillStyle = `rgba(${r},${g},${b},${.1 + .5 * (1 - t)})`;
      ctx.fillRect(i * step, j * step, step + 1, step + 1);
    }
  }
  ctx.restore();

  // iso-lines by marching squares on a coarser lattice
  const G = 64;
  const grid = [];
  for (let j = 0; j <= G; j++) {
    grid[j] = [];
    for (let i = 0; i <= G; i++) {
      const x = p.o.xmin + (p.o.xmax - p.o.xmin) * i / G;
      const y = p.o.ymin + (p.o.ymax - p.o.ymin) * j / G;
      grid[j][i] = f(x, y);
    }
  }
  ctx.save();
  ctx.strokeStyle = C.muted;
  ctx.globalAlpha = .45;
  ctx.lineWidth = 1;
  for (let k = 1; k <= iso; k++) {
    const level = lo + span * (k / (iso + 1)) ** 2;
    ctx.beginPath();
    for (let j = 0; j < G; j++) {
      for (let i = 0; i < G; i++) {
        const x0 = p.o.xmin + (p.o.xmax - p.o.xmin) * i / G;
        const x1 = p.o.xmin + (p.o.xmax - p.o.xmin) * (i + 1) / G;
        const y0 = p.o.ymin + (p.o.ymax - p.o.ymin) * j / G;
        const y1 = p.o.ymin + (p.o.ymax - p.o.ymin) * (j + 1) / G;
        const a = grid[j][i], b = grid[j][i + 1], c = grid[j + 1][i + 1], d = grid[j + 1][i];
        const idx = (a > level ? 1 : 0) | (b > level ? 2 : 0) | (c > level ? 4 : 0) | (d > level ? 8 : 0);
        if (idx === 0 || idx === 15) continue;
        const L = (v0, v1, p0, p1) => p0 + (p1 - p0) * (level - v0) / ((v1 - v0) || 1e-9);
        const bot = [L(a, b, x0, x1), y0];
        const rgt = [x1, L(b, c, y0, y1)];
        const top = [L(d, c, x0, x1), y1];
        const lft = [x0, L(a, d, y0, y1)];
        const seg = {
          1: [bot, lft], 2: [bot, rgt], 3: [lft, rgt], 4: [rgt, top],
          5: [bot, rgt], 6: [bot, top], 7: [lft, top], 8: [lft, top],
          9: [bot, top], 10: [lft, bot], 11: [rgt, top], 12: [lft, rgt],
          13: [bot, rgt], 14: [bot, lft],
        }[idx];
        if (!seg) continue;
        ctx.moveTo(...p.toScreen(seg[0]));
        ctx.lineTo(...p.toScreen(seg[1]));
      }
    }
    ctx.stroke();
  }
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
   1. The derivative as a limit — watch the secant become tangent
   ============================================================ */
defineWidget('derivative-limit', node => {
  const { right, canvas } = split(node, { aspect: 1.5, hint: 'Drag the point a' });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -1, xmax: 4.2, ymin: -2, ymax: 16.5, aspect: 1.5, equal: false, pad: 0,
  }));

  const FS = {
    quad:  { f: x => 3 * x * x + 2 * x + 1, df: x => 6 * x + 2, label: 'f(x) = 3x² + 2x + 1', d: "f'(x) = 6x + 2" },
    cubic: { f: x => x ** 3 - 3 * x + 2,     df: x => 3 * x * x - 3, label: 'f(x) = x³ − 3x + 2', d: "f'(x) = 3x² − 3" },
    sin:   { f: x => 3 * Math.sin(1.6 * x) + 4, df: x => 4.8 * Math.cos(1.6 * x), label: 'f(x) = 3 sin(1.6x) + 4', d: "f'(x) = 4.8 cos(1.6x)" },
  };
  let key = 'quad', a = 1.2, h = 1.4;

  const pick = segmented(
    [{ label: 'Quadratic', value: 'quad' }, { label: 'Cubic', value: 'cubic' }, { label: 'Sine', value: 'sin' }],
    { value: key, label: 'Function', onChange: v => { key = v; refresh(); } }
  );
  const hCtl = slider('Step size h', {
    min: .002, max: 2, step: .002, value: h,
    format: v => fmt(v, 3), onInput: v => { h = v; refresh(); },
  });
  const acts = el('div', { class: 'pg-actions' },
    button('h → 0', () => animateTo(.002)),
    button('Reset h', () => { h = 1.4; hCtl.set(1.4); refresh(); }),
  );
  const out = readout([
    ['a', 0], ['h', 0], ['f(a)', 0], ['f(a + h)', 0],
    ['difference quotient', 0], ["f'(a) exact", 0], ['error', 0],
  ]);
  const st = status('');
  right.append(pick.root, hCtl.root, acts, out.root, st.root);

  let anim = null;
  function animateTo(target) {
    cancelAnimationFrame(anim);
    const t0 = performance.now(), h0 = h;
    const tick = now => {
      const k = clamp((now - t0) / 1400, 0, 1);
      const e = 1 - (1 - k) ** 3;
      h = h0 + (target - h0) * e;
      hCtl.set(h); refresh();
      if (k < 1) anim = requestAnimationFrame(tick);
    };
    anim = requestAnimationFrame(tick);
  }

  const drag = new Dragger(plot);
  drag.add(() => [a, FS[key].f(a)], p => { a = clamp(round(p[0], 2), -.6, 3.8); });
  drag.onchange = refresh;

  function refresh() { plot.render(); sync(); }
  function sync() {
    const { f, df } = FS[key];
    const q = (f(a + h) - f(a)) / h;
    const exact = df(a);
    out.set([
      fmt(a, 2), fmt(h, 4), fmt(f(a), 3), fmt(f(a + h), 3),
      { html: fmt(q, 4), cls: 'is-ok' }, fmt(exact, 4),
      { html: fmt(Math.abs(q - exact), 5), cls: Math.abs(q - exact) < .02 ? 'is-ok' : '' },
    ]);
    st.set(
      h < .05
        ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>` +
          `<span>The secant is now indistinguishable from the <strong>tangent</strong>. That limit is the derivative.</span>`
        : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16.5v.01" stroke-linecap="round"/></svg>` +
          `<span>The dashed line is a <strong>secant</strong> through two points. Shrink h and watch it pivot into place.</span>`,
      h < .05 ? 'ok' : '');
  }

  plot.onDraw(p => {
    const { f, df, label, d } = FS[key];
    p.grid(1, { color: C.grid }); p.axes({ ticks: 1 });
    p.fn(f, { color: C.c1, lw: 2.6 });

    const fa = f(a), fah = f(a + h);
    // rise / run box
    p.line([a, fa], [a + h, fa], { color: C.c3, lw: 1.6, dash: [4, 3] });
    p.line([a + h, fa], [a + h, fah], { color: C.c4, lw: 1.6, dash: [4, 3] });
    p.badge([a + h / 2, fa], `h = ${fmt(h, 3)}`, { color: C.c3, align: 'center', dy: 15 });
    p.badge([a + h, (fa + fah) / 2], `Δf = ${fmt(fah - fa, 3)}`, { color: C.c4, dx: 8 });

    // secant (extended) and true tangent
    const q = (fah - fa) / h;
    p.ray([a, fa], [1, q], { color: C.c2, lw: 2, dash: [7, 5], alpha: .95 });
    p.ray([a, fa], [1, df(a)], { color: C.c3, lw: 1.6, alpha: .5 });

    p.dot([a + h, fah], { r: 5, color: C.c4, ring: true });
    p.handle([a, fa], { color: C.c1, r: 6 });

    p.title(label, { size: 12, weight: 700, color: C.ink });
    p.text({ px: 12, py: 24 }, d,
      { color: C.muted, size: 11, baseline: 'top', halo: true, haloWidth: 4 });
    p.reserveTop = 38;
  });
  sync();

  node.appendChild(note(
    `The difference quotient <span class="u-mono">(f(a+h) − f(a)) / h</span> is literally ` +
    `<strong>rise over run</strong> for the dashed secant. As h shrinks, the second point slides toward the ` +
    `first and the secant rotates onto the tangent. The derivative is the number that slope converges to — ` +
    `watch the "error" row collapse toward zero.`
  ));
});

/* ============================================================
   2. Gradient descent on a loss surface
   ============================================================ */
defineWidget('gradient-descent', node => {
  const { right, canvas } = split(node, { aspect: 1.25, wide: true, hint: 'Click anywhere to drop a start point' });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -3.2, xmax: 3.2, ymin: -2.6, ymax: 2.6, aspect: 1.25, pad: 0,
  }));

  const SURF = {
    bowl: {
      name: 'Convex bowl',
      f: (x, y, k) => x * x + k * y * y,
      g: (x, y, k) => [2 * x, 2 * k * y],
      blurb: 'A well-behaved quadratic. One minimum, and gradient descent always finds it.',
    },
    valley: {
      name: 'Ill-conditioned',
      f: (x, y, k) => x * x + (k * 12) * y * y,
      g: (x, y, k) => [2 * x, 2 * k * 12 * y],
      blurb: 'The same bowl, stretched. Steep in one direction, nearly flat in the other — descent zig-zags.',
    },
    saddle: {
      name: 'Saddle',
      f: (x, y) => x * x - y * y,
      g: (x, y) => [2 * x, -2 * y],
      blurb: 'A critical point that is a minimum along one axis and a maximum along the other. The gradient vanishes but you are not at a minimum.',
    },
    double: {
      name: 'Two minima',
      f: (x, y) => (x * x - 1.6) ** 2 + 1.4 * y * y + .35 * x,
      g: (x, y) => [4 * x * (x * x - 1.6) + .35, 2.8 * y],
      blurb: 'Non-convex. Which minimum you reach depends entirely on where you start.',
    },
  };
  let key = 'bowl', lr = .12, k = .35, momentum = 0;
  let start = [-2.4, 1.9];
  let path = [], diverged = false;

  const pick = segmented(
    Object.entries(SURF).map(([kk, v]) => ({ label: v.name, value: kk })),
    { value: key, label: 'Loss surface', onChange: v => { key = v; run(); } }
  );
  const lrCtl = slider('Learning rate η', {
    min: .005, max: 1.1, step: .005, value: lr,
    format: v => fmt(v, 3), onInput: v => { lr = v; run(); },
  });
  const momCtl = slider('Momentum β', {
    min: 0, max: .95, step: .01, value: 0,
    format: v => fmt(v, 2), onInput: v => { momentum = v; run(); },
  });
  const kCtl = slider('Curvature ratio', {
    min: .05, max: 1, step: .01, value: k,
    format: v => fmt(v, 2), onInput: v => { k = v; run(); },
  });
  const acts = el('div', { class: 'pg-actions' },
    button('η too big', () => { lr = .95; lrCtl.set(.95); run(); }),
    button('η too small', () => { lr = .02; lrCtl.set(.02); run(); }),
    button('Good η', () => { lr = .18; lrCtl.set(.18); run(); }),
  );
  const out = readout([['steps to converge', 0], ['final loss', 0], ['‖∇f‖ at end', 0], ['status', 0]]);
  const st = status('');
  const blurb = el('div', { style: 'font-size:.86rem;color:var(--ink-muted);line-height:1.55' });
  right.append(pick.root, lrCtl.root, momCtl.root, kCtl.root, acts, out.root, st.root, blurb);

  canvas.addEventListener('pointerdown', e => {
    start = plot.eventWorld(e);
    run();
  });
  canvas.classList.add('is-grabbable');

  const F = (x, y) => SURF[key].f(x, y, k);
  const G = (x, y) => SURF[key].g(x, y, k);

  function run() {
    path = [start.slice()];
    diverged = false;
    let [x, y] = start;
    let vx = 0, vy = 0;
    let steps = 0;
    for (let i = 0; i < 400; i++) {
      const [gx, gy] = G(x, y);
      vx = momentum * vx - lr * gx;
      vy = momentum * vy - lr * gy;
      x += vx; y += vy;
      steps = i + 1;
      if (!Number.isFinite(x) || !Number.isFinite(y) || Math.abs(x) > 60 || Math.abs(y) > 60) {
        diverged = true; break;
      }
      path.push([x, y]);
      if (Math.hypot(gx, gy) < 1e-3 && Math.hypot(vx, vy) < 1e-4) break;
    }
    const last = path[path.length - 1];
    const g = G(last[0], last[1]);
    const gn = Math.hypot(...g);
    out.set([
      diverged ? '—' : String(steps),
      diverged ? 'diverged' : fmt(F(last[0], last[1]), 5),
      diverged ? '—' : fmt(gn, 5),
      { html: diverged ? 'diverged' : (gn < 1e-2 ? 'converged' : 'still moving'),
        cls: diverged ? 'is-warn' : (gn < 1e-2 ? 'is-ok' : '') },
    ]);
    blurb.innerHTML = SURF[key].blurb;

    if (diverged) {
      st.set(`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 9v4M12 17v.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/></svg>` +
        `<span><strong>Diverged.</strong> η is too large — each step overshoots by more than it corrects, so the iterates explode.</span>`, 'warn');
    } else if (key === 'saddle' && gn < 1e-2) {
      st.set(`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16.5v.01" stroke-linecap="round"/></svg>` +
        `<span>The gradient is zero, but this is a <strong>saddle</strong>, not a minimum. ∇f = 0 is necessary, never sufficient.</span>`, 'info');
    } else if (gn < 1e-2) {
      st.set(`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>` +
        `<span><strong>Converged</strong> in ${steps} steps.</span>`, 'ok');
    } else {
      st.set(`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16.5v.01" stroke-linecap="round"/></svg>` +
        `<span>Still crawling after 400 steps — η is too small to make progress.</span>`, '');
    }
    plot.render();
  }

  plot.onDraw(p => {
    drawField(p, F, { iso: 10 });
    p.axes();

    // gradient arrows on a sparse lattice
    for (let i = -3; i <= 3; i++) {
      for (let j = -2; j <= 2; j++) {
        const x = i * .95, y = j * .95;
        const [gx, gy] = G(x, y);
        const n = Math.hypot(gx, gy);
        if (n < 1e-6) continue;
        // fixed-length arrows: direction is the message, magnitude is the contours' job
        const L = 0.3 / n;
        p.arrow([x, y], [x - gx * L, y - gy * L],
          { color: C.muted, lw: 1.4, head: 7, alpha: .6 });
      }
    }

    if (path.length > 1) {
      const vis = path.filter(([x, y]) => Math.abs(x) < 40 && Math.abs(y) < 40);
      p.path(vis, { color: C.c4, lw: 2.4 });
      vis.forEach((q, i) => {
        if (i % Math.max(1, Math.floor(vis.length / 40)) === 0 || i === vis.length - 1) {
          p.dot(q, { r: 3, color: C.c4, alpha: .85 });
        }
      });
    }
    p.handle(start, { color: C.c3, r: 6, glow: true });
    const last = path[path.length - 1];
    if (last && Math.abs(last[0]) < 40) p.dot(last, { r: 6, color: C.c2, ring: true });

    p.text({ px: 12, py: 18 }, 'darker = lower loss', { color: C.muted, size: 11 });
  });

  run();

  node.appendChild(note(
    `Gradient descent takes the step <span class="u-mono">x ← x − η∇f(x)</span> — always downhill, always ` +
    `perpendicular to the contours. Three things to try: crank <strong>η</strong> up until it explodes; ` +
    `switch to <strong>Ill-conditioned</strong> and watch the path zig-zag across the valley instead of ` +
    `running along it (this is exactly why adaptive optimisers exist); then add <strong>momentum</strong> ` +
    `and watch the zig-zag damp out.`
  ));
});

/* ============================================================
   3. Convexity — the chord test
   ============================================================ */
defineWidget('convexity', node => {
  const { right, canvas } = split(node, { aspect: 1.5, hint: 'Drag the two endpoints' });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -3.2, xmax: 3.2, ymin: -3, ymax: 7, aspect: 1.5, equal: false, pad: 0,
  }));

  const FS = {
    quad:  { f: x => x * x, df: x => 2 * x, label: 'f(x) = x²', convex: true },
    exp:   { f: x => Math.exp(.8 * x) - 1, df: x => .8 * Math.exp(.8 * x), label: 'f(x) = e^{0.8x} − 1', convex: true },
    abs:   { f: x => 1.8 * Math.abs(x), df: x => 1.8 * Math.sign(x), label: 'f(x) = 1.8|x|', convex: true },
    cubic: { f: x => .5 * x ** 3 - x, df: x => 1.5 * x * x - 1, label: 'f(x) = 0.5x³ − x', convex: false },
    sin:   { f: x => 2.2 * Math.sin(1.5 * x) + 1, df: x => 3.3 * Math.cos(1.5 * x), label: 'f(x) = 2.2 sin(1.5x) + 1', convex: false },
  };
  let key = 'quad', x1 = -2, x2 = 1.8, theta = .5;
  const showTangent = { on: true };

  const pick = segmented(
    [{ label: 'x²', value: 'quad' }, { label: 'eˣ', value: 'exp' }, { label: '|x|', value: 'abs' },
     { label: 'x³', value: 'cubic' }, { label: 'sin', value: 'sin' }],
    { value: key, label: 'Function', onChange: v => { key = v; refresh(); } }
  );
  const tCtl = slider('θ (position along the chord)', {
    min: 0, max: 1, step: .01, value: .5, onInput: v => { theta = v; refresh(); },
  });
  const tanCtl = toggle('Show first-order condition', {
    value: true, onChange: v => { showTangent.on = v; plot.render(); },
  });
  const out = readout([
    ['θx₁ + (1−θ)x₂', 0], ['f(θx₁ + (1−θ)x₂)', 0], ['θf(x₁) + (1−θ)f(x₂)', 0], ['inequality', 0],
  ]);
  const st = status('');
  right.append(pick.root, tCtl.root, tanCtl.root, out.root, st.root);

  const drag = new Dragger(plot);
  drag.add(() => [x1, FS[key].f(x1)], p => { x1 = clamp(round(p[0], 2), -3, 3); });
  drag.add(() => [x2, FS[key].f(x2)], p => { x2 = clamp(round(p[0], 2), -3, 3); });
  drag.onchange = refresh;

  function refresh() { plot.render(); sync(); }

  function sync() {
    const { f } = FS[key];
    const xm = theta * x1 + (1 - theta) * x2;
    const lhs = f(xm);
    const rhs = theta * f(x1) + (1 - theta) * f(x2);
    const ok = lhs <= rhs + 1e-9;
    out.set([
      fmt(xm, 3), fmt(lhs, 4), fmt(rhs, 4),
      { html: ok ? 'f(mix) ≤ mix of f &nbsp;✓' : 'f(mix) &gt; mix of f &nbsp;✗', cls: ok ? 'is-ok' : 'is-warn' },
    ]);

    // scan the whole interval for a violation, not just this θ
    let violated = false;
    for (let t = 0; t <= 1; t += .01) {
      const xx = t * x1 + (1 - t) * x2;
      if (f(xx) > t * f(x1) + (1 - t) * f(x2) + 1e-6) { violated = true; break; }
    }
    st.set(
      violated
        ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 9v4M12 17v.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/></svg>` +
          `<span><strong>Not convex here.</strong> Part of the curve pokes above the chord — you have found a counterexample.</span>`
        : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>` +
          `<span>The chord stays <strong>on or above</strong> the curve across this whole interval — the convexity inequality holds.</span>`,
      violated ? 'warn' : 'ok');
  }

  plot.onDraw(p => {
    const { f, df, label } = FS[key];
    p.grid(1, { color: C.grid }); p.axes({ ticks: 1 });

    // shade the gap between chord and curve
    const A = [x1, f(x1)], B = [x2, f(x2)];
    const chordY = x => {
      const t = (x - x1) / ((x2 - x1) || 1e-9);
      return f(x1) + t * (f(x2) - f(x1));
    };
    const lo = Math.min(x1, x2), hi = Math.max(x1, x2);
    const region = [];
    for (let i = 0; i <= 80; i++) region.push([lo + (hi - lo) * i / 80, chordY(lo + (hi - lo) * i / 80)]);
    for (let i = 80; i >= 0; i--) region.push([lo + (hi - lo) * i / 80, f(lo + (hi - lo) * i / 80)]);
    p.path(region, { fill: C.fill, stroke: false, close: true });

    p.fn(f, { color: C.c1, lw: 2.8 });
    p.line(A, B, { color: C.c2, lw: 2.4 });

    const xm = theta * x1 + (1 - theta) * x2;
    p.line([xm, f(xm)], [xm, chordY(xm)], { color: C.c4, lw: 2.4 });
    p.dot([xm, f(xm)], { r: 4.5, color: C.c1, ring: true });
    p.dot([xm, chordY(xm)], { r: 4.5, color: C.c2, ring: true });

    if (showTangent.on) {
      p.ray([xm, f(xm)], [1, df(xm)], { color: C.c3, lw: 1.5, dash: [6, 4], alpha: .8 });
      p.badge([xm + 1.4, f(xm) + df(xm) * 1.4], 'tangent', { color: C.c3, align: 'center' });
    }

    p.handle(A, { color: C.c1, r: 6 });
    p.handle(B, { color: C.c1, r: 6 });
    p.text({ px: 12, py: 18 }, label, { color: C.ink, size: 12, weight: 700 });
  });
  sync();

  node.appendChild(note(
    `A function is <strong>convex</strong> when the straight line between any two points on its graph never ` +
    `dips below the graph. Drag the endpoints across the cubic or the sine and you will find a chord that ` +
    `cuts underneath — one counterexample is enough to rule convexity out. The dashed tangent shows the ` +
    `equivalent first-order view: for a convex function the tangent is a <em>global underestimator</em>, ` +
    `which is why a local minimum has to be global.`
  ));
});

/* ============================================================
   4. Quadratic forms, the Hessian, and definiteness
   ============================================================ */
defineWidget('quadratic-form', node => {
  const { right, canvas } = split(node, { aspect: 1.2, wide: true, hint: 'Drag x around the circle' });
  const plot = trackPlot(new Plot(canvas, {
    xmin: -2.6, xmax: 2.6, ymin: -2.6, ymax: 2.6, aspect: 1.2, pad: 0,
  }));

  let A = [[2, 1], [1, 3]];
  let theta = .7;

  const Min = matrixInput(2, 2, A, {
    label: 'Symmetric A', step: .5,
    onInput: m => {
      // keep it symmetric: mirror whichever off-diagonal the user just touched
      A = m;
      if (Math.abs(A[0][1] - A[1][0]) > 1e-9) { A[1][0] = A[0][1]; Min.set(A); }
      refresh();
    },
  });
  const presets = el('div', { class: 'pg-actions' },
    button('Positive definite', () => set([[2, -1], [-1, 2]])),
    button('Positive semidefinite', () => set([[1, -1], [-1, 1]])),
    button('Indefinite', () => set([[1, 0], [0, -1]])),
  );
  const out = readout([['x', 0], ['Q(x) = xᵀAx', 0], ['λ₁, λ₂', 0], ['λ_min‖x‖², λ_max‖x‖²', 0], ['classification', 0]]);
  const st = status('');
  right.append(Min.root, presets, out.root, st.root);

  function set(m) { A = m.map(r => r.slice()); Min.set(A); refresh(); }
  function refresh() { plot.render(); sync(); }

  const X = () => [Math.cos(theta) * 1.5, Math.sin(theta) * 1.5];

  const drag = new Dragger(plot);
  drag.add(X, p => { theta = Math.atan2(p[1], p[0]); });
  drag.onchange = refresh;

  function eig2([[a, b], [, d]]) {
    const tr = a + d, dt = a * d - b * b;
    const s = Math.sqrt(Math.max(0, tr * tr / 4 - dt));
    return [tr / 2 + s, tr / 2 - s];
  }

  function sync() {
    const x = X();
    const Q = A[0][0] * x[0] * x[0] + 2 * A[0][1] * x[0] * x[1] + A[1][1] * x[1] * x[1];
    const [l1, l2] = eig2(A);
    const n2 = x[0] * x[0] + x[1] * x[1];
    const cls = l2 > 1e-9 ? 'positive definite'
              : l2 >= -1e-9 && l1 > 0 ? 'positive semidefinite'
              : l1 < -1e-9 ? 'negative definite'
              : 'indefinite';
    out.set([
      `(${fmt(x[0], 2)}, ${fmt(x[1], 2)})`,
      { html: fmt(Q, 3), cls: Q > 1e-9 ? 'is-ok' : Q < -1e-9 ? 'is-warn' : '' },
      `${fmt(l1, 3)}, ${fmt(l2, 3)}`,
      `${fmt(l2 * n2, 3)} ≤ ${fmt(Q, 3)} ≤ ${fmt(l1 * n2, 3)}`,
      { html: cls, cls: cls.startsWith('positive') ? 'is-ok' : 'is-warn' },
    ]);
    const msg = {
      'positive definite': `All eigenvalues &gt; 0 — the surface is a <strong>bowl</strong>, and a critical point here is a genuine minimum.`,
      'positive semidefinite': `One eigenvalue is 0 — the bowl has a <strong>flat trough</strong>. Q can be zero for nonzero x.`,
      'negative definite': `All eigenvalues &lt; 0 — an upside-down bowl, so a critical point is a <strong>maximum</strong>.`,
      'indefinite': `Mixed signs — a <strong>saddle</strong>. Q is positive in some directions and negative in others.`,
    }[cls];
    st.set(msg, cls.startsWith('positive') ? 'ok' : cls === 'indefinite' ? 'warn' : 'info');
  }

  plot.onDraw(p => {
    // colour the plane by the sign and size of Q
    const f = (x, y) => A[0][0] * x * x + 2 * A[0][1] * x * y + A[1][1] * y * y;
    drawField(p, f, { iso: 9, alpha: .85 });
    p.axes({ ticks: 1 });

    // unit circle and the value of Q along it
    const circ = [], curve = [];
    let mx = 1e-9;
    for (let i = 0; i <= 180; i++) {
      const t = i / 180 * Math.PI * 2;
      const v = [Math.cos(t), Math.sin(t)];
      mx = Math.max(mx, Math.abs(f(v[0], v[1])));
    }
    for (let i = 0; i <= 180; i++) {
      const t = i / 180 * Math.PI * 2;
      const c = Math.cos(t), s = Math.sin(t);
      circ.push([c * 1.5, s * 1.5]);
      const r = 1.5 + f(c, s) / mx * .75;
      curve.push([c * r, s * r]);
    }
    p.path(circ, { color: C.muted, lw: 1.2, dash: [4, 4], alpha: .6, close: true });
    p.path(curve, { color: C.c2, lw: 2.4, close: true });

    // eigen-directions
    const [l1, l2] = eig2(A);
    const b = A[0][1];
    const v1 = Math.abs(b) > 1e-9 ? norm([b, l1 - A[0][0]]) : [1, 0];
    const v2 = [-v1[1], v1[0]];
    [[v1, l1, C.c3], [v2, l2, C.c5]].forEach(([v, l, col]) => {
      p.ray([0, 0], v, { color: col, lw: 1.5, dash: [7, 5], alpha: .8 });
      p.badge([v[0] * 2.15, v[1] * 2.15], `λ = ${fmt(l, 2)}`, { color: col, align: 'center' });
    });

    const x = X();
    p.arrow([0, 0], x, { color: C.c1, lw: 3 });
    p.handle(x, { color: C.c1, r: 6 });
    p.text({ px: 12, py: 18 }, 'Q(x) along the unit circle (orange)', { color: C.muted, size: 11 });
  });
  function norm(v) { const n = Math.hypot(...v) || 1; return [v[0] / n, v[1] / n]; }
  sync();

  node.appendChild(note(
    `A quadratic form <span class="u-mono">Q(x) = xᵀAx</span> is squeezed between its extreme eigenvalues: ` +
    `<span class="u-mono">λ_min‖x‖² ≤ Q(x) ≤ λ_max‖x‖²</span>. The orange curve is Q's value in every ` +
    `direction — it bulges where the eigenvalue is large. This is exactly how the ` +
    `<strong>Hessian</strong> classifies a critical point: all eigenvalues positive means a minimum, all ` +
    `negative means a maximum, mixed signs means a saddle.`
  ));
});

/* ============================================================
   5. Chain rule as a computation graph
   ============================================================ */
defineWidget('chain-rule', node => {
  const { right, canvas } = split(node, { aspect: 2.1 });
  const plot = trackPlot(new Plot(canvas, {
    xmin: 0, xmax: 10, ymin: 0, ymax: 4.6, aspect: 2.1, equal: false, pad: 0,
  }));

  let x = 1.5;
  const xCtl = slider('x', { min: -2.5, max: 2.5, step: .05, value: x, onInput: v => { x = v; refresh(); } });
  const out = readout([
    ['x', 0], ['u = x²', 0], ['v = sin(u)', 0], ['z = 3v + 1', 0],
    ['dz/dv', 0], ['dv/du', 0], ['du/dx', 0], ['dz/dx (product)', 0],
  ]);
  const st = status(
    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16.5v.01" stroke-linecap="round"/></svg>` +
    `Values flow left to right; derivatives multiply back right to left. That backward pass is backpropagation.`, 'info');
  right.append(xCtl.root, out.root, st.root);

  const u = () => x * x;
  const v = () => Math.sin(u());
  const z = () => 3 * v() + 1;
  const dzdv = () => 3;
  const dvdu = () => Math.cos(u());
  const dudx = () => 2 * x;

  function refresh() { plot.render(); sync(); }
  function sync() {
    out.set([
      fmt(x, 3), fmt(u(), 3), fmt(v(), 3), fmt(z(), 3),
      fmt(dzdv(), 3), fmt(dvdu(), 3), fmt(dudx(), 3),
      { html: fmt(dzdv() * dvdu() * dudx(), 4), cls: 'is-ok' },
    ]);
  }

  plot.onDraw(p => {
    const nodes = [
      { x: 1.1, label: 'x',  val: x,   color: C.c3 },
      { x: 3.6, label: 'u',  val: u(), color: C.c1 },
      { x: 6.1, label: 'v',  val: v(), color: C.c1 },
      { x: 8.7, label: 'z',  val: z(), color: C.c2 },
    ];
    const edges = [
      { from: 0, to: 1, fwd: 'u = x²',      back: `du/dx = ${fmt(dudx(), 2)}` },
      { from: 1, to: 2, fwd: 'v = sin(u)',  back: `dv/du = ${fmt(dvdu(), 2)}` },
      { from: 2, to: 3, fwd: 'z = 3v + 1',  back: `dz/dv = ${fmt(dzdv(), 2)}` },
    ];

    edges.forEach(e => {
      const a = nodes[e.from], b = nodes[e.to];
      p.arrow([a.x + .55, 2.9], [b.x - .55, 2.9], { color: C.c1, lw: 2, head: 9 });
      p.text([(a.x + b.x) / 2, 3.35], e.fwd, { align: 'center', size: 11.5, color: C.c1, weight: 650 });
      p.arrow([b.x - .55, 1.75], [a.x + .55, 1.75], { color: C.c4, lw: 2, head: 9, dash: [5, 4] });
      p.text([(a.x + b.x) / 2, 1.28], e.back, { align: 'center', size: 11.5, color: C.c4, weight: 650, mono: true });
    });

    nodes.forEach(n => {
      p.ctx.beginPath();
      p.ctx.arc(p.X(n.x), p.Y(2.32), 27, 0, Math.PI * 2);
      p.ctx.fillStyle = C.raised; p.ctx.fill();
      p.ctx.strokeStyle = n.color; p.ctx.lineWidth = 2.4; p.ctx.stroke();
      p.text([n.x, 2.48], n.label, { align: 'center', size: 14, weight: 750, color: n.color });
      p.text([n.x, 2.1], fmt(n.val, 2), { align: 'center', size: 10.5, color: C.muted, mono: true });
    });

    p.text({ px: 12, py: 11 }, 'forward: values', { color: C.c1, size: 11, weight: 650 });
    p.text({ px: 12, py: p.h - 6 }, 'backward: local derivatives, multiplied', { color: C.c4, size: 11, weight: 650, baseline: 'bottom' });
    p.text([5, .5],
      `dz/dx = ${fmt(dzdv(), 2)} × ${fmt(dvdu(), 2)} × ${fmt(dudx(), 2)} = ${fmt(dzdv() * dvdu() * dudx(), 3)}`,
      { align: 'center', size: 13, weight: 700, color: C.ink });
  });
  sync();

  node.appendChild(note(
    `The chain rule says a composite function's derivative is the <strong>product of the local derivatives</strong> ` +
    `along the path. Compute values forward, then multiply derivatives backward. Scale this up to a network ` +
    `with millions of edges and you have <strong>backpropagation</strong> — nothing more exotic than this ` +
    `picture repeated.`
  ));
});

/* ============================================================
   6. Big-O — how cost scales
   ============================================================ */
defineWidget('bigo', node => {
  const { right, canvas } = split(node, { aspect: 1.55, wide: true });
  const plot = trackPlot(new Plot(canvas, {
    xmin: 0, xmax: 500, ymin: 0, ymax: 1, aspect: 1.55, equal: false, pad: 0,
  }));

  let n = 200, logScale = true;
  const OPS = [
    { key: 'add',  label: 'Vector add — O(n)',           f: n => n,             color: () => C.c3 },
    { key: 'matvec', label: 'Matrix–vector — O(n²)',      f: n => 2 * n * n,     color: () => C.c5 },
    { key: 'matmul', label: 'Matrix–matrix — O(n³)',      f: n => 2 * n ** 3,    color: () => C.c1 },
    { key: 'svd',  label: 'SVD / inverse — O(n³)',        f: n => 12 * n ** 3,   color: () => C.c4 },
  ];

  const nCtl = slider('Matrix size n', {
    min: 10, max: 1000, step: 10, value: n, format: v => String(v),
    onInput: v => { n = v; refresh(); },
  });
  const scaleCtl = toggle('Logarithmic vertical axis', {
    value: true, onChange: v => { logScale = v; refresh(); },
  });
  const out = readout(OPS.map(o => [o.label.split(' — ')[0], 0]));
  const st = status('');
  right.append(nCtl.root, scaleCtl.root, out.root, st.root);

  const human = v => {
    if (v < 1e3) return fmt(v, 0);
    if (v < 1e6) return `${fmt(v / 1e3, 1)}K`;
    if (v < 1e9) return `${fmt(v / 1e6, 1)}M`;
    if (v < 1e12) return `${fmt(v / 1e9, 2)}B`;
    return `${fmt(v / 1e12, 2)}T`;
  };

  function refresh() {
    out.set(OPS.map(o => human(o.f(n))));
    const r = OPS[2].f(2 * n) / OPS[2].f(n);
    st.set(
      `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16.5v.01" stroke-linecap="round"/></svg>` +
      `<span>Doubling n to ${2 * n} multiplies matrix–matrix cost by <strong>${fmt(r, 0)}×</strong> ` +
      `(${human(OPS[2].f(n))} → ${human(OPS[2].f(2 * n))} operations).</span>`, 'info');
    plot.render();
  }

  plot.onDraw(p => {
    const N = 1000;
    const maxv = OPS.reduce((m, o) => Math.max(m, o.f(N)), 1);
    const tf = v => logScale
      ? Math.log10(Math.max(1, v)) / Math.log10(maxv)
      : v / maxv;

    p.o.xmin = 0; p.o.xmax = N; p.o.ymin = 0; p.o.ymax = 1.08;
    p._computeScale();
    p.grid(N / 10, { color: C.grid });

    OPS.forEach(o => {
      const pts = [];
      for (let i = 1; i <= 200; i++) {
        const xx = N * i / 200;
        pts.push([xx, tf(o.f(xx))]);
      }
      p.path(pts, { color: o.color(), lw: 2.4 });
      const last = pts[pts.length - 1];
      p.text([N * .985, last[1]], o.label.split(' — ')[1], {
        align: 'right', size: 10.5, color: o.color(), weight: 650, dy: -9, halo: true,
      });
    });

    p.line([n, 0], [n, 1.08], { color: C.c2, lw: 1.8, dash: [5, 4] });
    p.badge([n, 0], `n = ${n}`, { color: C.c2, align: 'center', dy: -24 });

    p.axes(); p.ticks(N / 5);
    // curve labels live on the right, where the fastest-growing curve ends
    p.title(logScale ? 'operations (log scale)' : 'operations (linear scale)');
  });

  refresh();

  node.appendChild(note(
    `Big-O throws away constants and keeps only the growth rate, because for large n the growth rate is all ` +
    `that matters. On the linear axis the cubic curves make everything else look flat — that is the honest ` +
    `picture of why <strong>matrix multiplication dominates deep learning's cost</strong>, and why so much ` +
    `effort goes into shrinking matrices (low rank) or filling them with zeros (sparsity).`
  ));
});
