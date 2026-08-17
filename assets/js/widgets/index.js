/* ============================================================
   widgets/index.js — register every widget, then mount lazily
   ============================================================ */
import { mountWidgets, defineWidget, Plot, C, css, trackPlot, el } from '../viz.js';
import { rng, gauss } from '../ml.js';
import './la.js';
import './prob.js';
import './calc.js';
import './prep.js';
import './sim.js';
import './eval.js';
import './sup.js';
import './nets.js';
import './deep.js';
import './unsup.js';

/* ------------------------------------------------------------
   Landing-page hero: a small network learning, in real time, to
   separate two interleaved spirals — the standard example of a
   problem no linear model can touch. It trains continuously,
   starts over once it has converged, and the pointer rotates the
   data so the boundary has to chase it.
   ------------------------------------------------------------ */
defineWidget('hero-demo', node => {
  const cv = el('canvas', { style: 'cursor:default' });   // beats .pg-canvas-wrap canvas
  // touch-action:auto overrides .pg-canvas-wrap — this figure only reacts to
  // hover, so a finger swiping over it must still scroll the page
  const wrap = el('div', {
    class: 'pg-canvas-wrap',
    style: 'border:none;background:transparent;touch-action:auto',
  }, cv);
  node.appendChild(wrap);

  const R = 1.45;   // the spiral fills ~70% of this, leaving room for the glow to fall off
  const plot = trackPlot(new Plot(cv, {
    xmin: -R, xmax: R, ymin: -R, ymax: R, aspect: 1.08, equal: true, pad: 0,
  }));

  /* ---- two interleaved spirals ---- */
  const NPTS = 170;
  const base = [];
  {
    const r = rng(9);
    for (let i = 0; i < NPTS; i++) {
      const cls = i % 2;
      const t = .22 + 3.1 * (i / NPTS);
      const a = t * 2.1 + cls * Math.PI;
      const rad = t * .30;
      base.push([rad * Math.cos(a) + gauss(r) * .034,
                 rad * Math.sin(a) + gauss(r) * .034, cls]);
    }
  }
  let theta = 0, targetTheta = 0;
  const data = base.map(p => p.slice());
  const rotate = () => {
    const c = Math.cos(theta), s = Math.sin(theta);
    for (let i = 0; i < base.length; i++) {
      data[i][0] = base[i][0] * c - base[i][1] * s;
      data[i][1] = base[i][0] * s + base[i][1] * c;
    }
  };

  /* ---- a 2-16-16-2 tanh network trained by momentum SGD ---- */
  const SIZES = [2, 16, 16, 2];
  let net, vel, epoch = 0, acc = 0, settled = 0;

  function initNet(seed) {
    const r = rng(seed);
    const W = [], b = [];
    for (let l = 0; l < SIZES.length - 1; l++) {
      const sc = Math.sqrt(2 / SIZES[l]);
      W.push(Array.from({ length: SIZES[l + 1] }, () =>
        Array.from({ length: SIZES[l] }, () => gauss(r) * sc)));
      b.push(new Array(SIZES[l + 1]).fill(0));
    }
    return { W, b };
  }
  const zeroLike = n => ({
    W: n.W.map(m => m.map(row => row.map(() => 0))),
    b: n.b.map(v => v.map(() => 0)),
  });
  function reset(seed) {
    net = initNet(seed);
    vel = zeroLike(net);
    epoch = 0; settled = 0;
  }
  reset(3);

  const sigmoidOf = (a, b2) => {
    const m = Math.max(a, b2);
    const e1 = Math.exp(a - m), e2 = Math.exp(b2 - m);
    return e2 / (e1 + e2);
  };
  /* returns P(class 1); fills `acts` when given, for the backward pass */
  function forward(n, x0, x1, acts) {
    let a = [x0, x1];
    if (acts) acts[0] = a;
    for (let l = 0; l < 3; l++) {
      const z = n.W[l].map((row, i) => {
        let s = n.b[l][i];
        for (let j = 0; j < row.length; j++) s += row[j] * a[j];
        return s;
      });
      a = l === 2 ? z : z.map(Math.tanh);
      if (acts) acts[l + 1] = a;
    }
    return sigmoidOf(a[0], a[1]);
  }

  function trainStep(lr, mu) {
    const gW = net.W.map(m => m.map(row => row.map(() => 0)));
    const gb = net.b.map(v => v.map(() => 0));
    let correct = 0;
    const acts = [];
    for (const [px, py, lab] of data) {
      const p1 = forward(net, px, py, acts);
      if ((p1 > .5 ? 1 : 0) === lab) correct++;
      // softmax + cross-entropy collapses to (prediction − target)
      let d = [(1 - p1) - (1 - lab), p1 - lab];
      for (let l = 2; l >= 0; l--) {
        const inp = acts[l];
        for (let i = 0; i < gW[l].length; i++) {
          gb[l][i] += d[i];
          for (let j = 0; j < inp.length; j++) gW[l][i][j] += d[i] * inp[j];
        }
        if (l > 0) {
          const back = inp.map((_, j) => net.W[l].reduce((s, row, i) => s + row[j] * d[i], 0));
          d = back.map((v, j) => v * (1 - inp[j] * inp[j]));   // derivative of tanh
        }
      }
    }
    const n = data.length;
    for (let l = 0; l < 3; l++) {
      for (let i = 0; i < net.W[l].length; i++) {
        vel.b[l][i] = mu * vel.b[l][i] - lr * gb[l][i] / n;
        net.b[l][i] += vel.b[l][i];
        for (let j = 0; j < net.W[l][i].length; j++) {
          vel.W[l][i][j] = mu * vel.W[l][i][j] - lr * gW[l][i][j] / n;
          net.W[l][i][j] += vel.W[l][i][j];
        }
      }
    }
    epoch++;
    acc = correct / n;
  }

  /* ---- the decision field, painted from a small offscreen buffer ---- */
  const G = 52;
  const buf = document.createElement('canvas');
  buf.width = G; buf.height = G;
  const bctx = buf.getContext('2d');
  const img = bctx.createImageData(G, G);
  const hexRGB = hex => {
    hex = (hex || '#888').trim();
    if (!hex.startsWith('#')) return [136, 136, 136];
    return hex.length === 4
      ? hex.slice(1).split('').map(c => parseInt(c + c, 16))
      : [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));
  };

  function paintField() {
    const c1 = hexRGB(C.c1), c2 = hexRGB(C.c2);
    const half = (G - 1) / 2;
    for (let gy = 0; gy < G; gy++) {
      const y = R - 2 * R * (gy / (G - 1));
      for (let gx = 0; gx < G; gx++) {
        const x = -R + 2 * R * (gx / (G - 1));
        const p1 = forward(net, x, y);
        const t = Math.abs(p1 - .5) * 2;                 // how sure the net is
        const c = p1 > .5 ? c2 : c1;
        // fade the field out towards the edge so it reads as a soft disc,
        // not a rectangle pasted onto the page
        const dx = (gx - half) / half, dy = (gy - half) / half;
        const rad = Math.sqrt(dx * dx + dy * dy);
        const vig = rad >= 1 ? 0 : rad <= .74 ? 1 : (1 - (rad - .74) / .26) ** 1.8;
        const k = (gy * G + gx) * 4;
        img.data[k]     = c[0];
        img.data[k + 1] = c[1];
        img.data[k + 2] = c[2];
        img.data[k + 3] = 255 * (.07 + .36 * t * t) * vig;
      }
    }
    bctx.putImageData(img, 0, 0);
  }

  /* ---- animation ---- */
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const LR = .12, MU = .9;         // lr .35 also converges; this pace is watchable
  const EPOCHS_PER_SEC = 55;       // paced by the clock, not by the refresh rate
  const REPAINT_MS = 60;
  let raf = null, visible = true, last = 0, debt = 0, sincePaint = 1e9, held = 0;

  wrap.addEventListener('pointermove', e => {
    const r = cv.getBoundingClientRect();
    targetTheta = ((e.clientX - r.left) / r.width - .5) * 2.4;
  });
  wrap.addEventListener('pointerleave', () => { targetTheta = 0; });

  const io = new IntersectionObserver(es => { visible = es[0].isIntersecting; });
  io.observe(wrap);

  // the field lives in an offscreen buffer, so a theme flip has to repaint it
  // explicitly — trackPlot only re-runs onDraw
  const onTheme = () => { paintField(); plot.renderNow(); };
  window.addEventListener('themechange', onTheme);

  function tick(now) {
    const dt = last ? Math.min(100, now - last) : 16;
    last = now;
    if (visible) {
      if (Math.abs(targetTheta - theta) > 1e-3) {
        theta += (targetTheta - theta) * .07;
        rotate();
      }
      if (!reduce) {
        debt += dt / 1000 * EPOCHS_PER_SEC;
        let steps = Math.min(4, Math.floor(debt));
        debt -= steps;
        while (steps-- > 0) trainStep(LR, MU);
      }

      // hold the finished boundary for a beat, then learn it again from scratch
      held = acc > .995 ? held + dt : 0;
      if (held > 2600 || epoch > 1500) { reset(3 + ((epoch * 7919) % 997)); held = 0; }

      sincePaint += dt;
      if (sincePaint >= REPAINT_MS) { paintField(); sincePaint = 0; }
      plot.renderNow();
    }
    raf = requestAnimationFrame(tick);
  }

  plot.onDraw(p => {
    p.clear(null);
    p.ctx.imageSmoothingEnabled = true;
    p.ctx.globalAlpha = .95;
    p.ctx.drawImage(buf, p.X(-R), p.Y(R), p.px(2 * R), p.py(2 * R));
    p.ctx.globalAlpha = 1;

    for (const [x, y, lab] of data) {
      const [sx, sy] = p.toScreen([x, y]);
      p.ctx.beginPath();
      p.ctx.arc(sx, sy, 3.8, 0, Math.PI * 2);
      p.ctx.fillStyle = lab ? C.c2 : C.c1;
      p.ctx.fill();
      p.ctx.strokeStyle = css('--bg');
      p.ctx.lineWidth = 1.5;
      p.ctx.stroke();
    }

    // a quiet live caption, so it reads as training rather than decoration
    p.ctx.font = `600 11px ${css('--font-mono')}`;
    p.ctx.textAlign = 'left'; p.ctx.textBaseline = 'top';
    p.ctx.fillStyle = C.muted;
    p.ctx.globalAlpha = .8;
    p.ctx.fillText(`epoch ${epoch}`, 10, 10);
    p.ctx.fillText(`train accuracy ${(acc * 100).toFixed(1)}%`, 10, 26);
    p.ctx.globalAlpha = 1;
  });

  rotate();
  paintField();
  if (reduce) { for (let i = 0; i < 900; i++) trainStep(LR, MU); paintField(); }
  plot.renderNow();
  raf = requestAnimationFrame(tick);
  window.addEventListener('pagehide', () => {
    cancelAnimationFrame(raf); io.disconnect();
    window.removeEventListener('themechange', onTheme);
  });
});

mountWidgets();
