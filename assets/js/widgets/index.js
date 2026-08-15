/* ============================================================
   widgets/index.js — register every widget, then mount lazily
   ============================================================ */
import { mountWidgets, defineWidget, Plot, C, trackPlot, mat, el } from '../viz.js';
import './la.js';
import './prob.js';
import './calc.js';
import './prep.js';
import './sim.js';
import './eval.js';
import './sup.js';
import './nets.js';

/* ------------------------------------------------------------
   Landing-page hero: an ambient linear transformation that
   responds to the pointer. Decorative, so it stays cheap.
   ------------------------------------------------------------ */
defineWidget('hero-demo', node => {
  const cv = el('canvas');
  const wrap = el('div', { class: 'pg-canvas-wrap', style: 'border:none;background:transparent' }, cv);
  node.appendChild(wrap);

  const plot = trackPlot(new Plot(cv, {
    xmin: -3.1, xmax: 3.1, ymin: -2.5, ymax: 2.5, aspect: 1.25, pad: 0,
  }));

  let t = 0;
  let target = [0, 0];
  let cur = [0, 0];
  let raf = null;
  let visible = true;

  wrap.addEventListener('pointermove', e => {
    const r = cv.getBoundingClientRect();
    target = [
      ((e.clientX - r.left) / r.width - .5) * 2,
      ((e.clientY - r.top) / r.height - .5) * 2,
    ];
  });
  wrap.addEventListener('pointerleave', () => { target = [0, 0]; });

  const io = new IntersectionObserver(es => { visible = es[0].isIntersecting; });
  io.observe(wrap);

  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function frame() {
    if (visible) {
      t += reduce ? 0 : .006;
      cur = [cur[0] + (target[0] - cur[0]) * .06, cur[1] + (target[1] - cur[1]) * .06];
      plot.renderNow();
    }
    raf = requestAnimationFrame(frame);
  }

  plot.onDraw(p => {
    const a = 1.15 + .32 * Math.sin(t) + cur[0] * .34;
    const b = .48 * Math.cos(t * .8) + cur[1] * .42;
    const c = .34 * Math.sin(t * 1.3) - cur[1] * .26;
    const d = 1.15 + .3 * Math.cos(t * 1.1) - cur[0] * .22;
    const A = [a, b, c, d];

    p.clear(null);
    p.transformedGrid(A, { range: 4, step: .5, color: C.c1, alpha: .16, seg: 16 });

    const circ = [], img = [];
    for (let i = 0; i <= 96; i++) {
      const th = i / 96 * Math.PI * 2;
      const v = [Math.cos(th) * 1.35, Math.sin(th) * 1.35];
      circ.push(v); img.push(mat.apply(A, v));
    }
    p.path(circ, { color: C.muted, lw: 1.1, dash: [4, 5], alpha: .4, close: true });
    p.path(img, { color: C.c2, lw: 2.4, close: true, fill: C.fill2 });

    const sq = [[0, 0], [1, 0], [1, 1], [0, 1]];
    p.polygon(sq.map(v => mat.apply(A, v)), { fill: C.fill, color: C.c1, lw: 1.8 });

    p.arrow([0, 0], [A[0], A[2]], { color: C.c1, lw: 2.8 });
    p.arrow([0, 0], [A[1], A[3]], { color: C.c2, lw: 2.8 });

    const e = mat.eig(A);
    if (e) {
      p.ray([0, 0], e.v1, { color: C.c3, lw: 1.2, dash: [7, 6], alpha: .5 });
      p.ray([0, 0], e.v2, { color: C.c5, lw: 1.2, dash: [7, 6], alpha: .5 });
    }
  });

  frame();
  window.addEventListener('pagehide', () => { cancelAnimationFrame(raf); io.disconnect(); });
});

mountWidgets();
