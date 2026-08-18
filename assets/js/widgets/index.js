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
   Landing-page hero: a feed-forward network, drawn in full, with
   a live forward pass running through it. The network is really
   trained — 2-8-8-2, tanh, on two interleaved spirals — so every
   activation and every edge you see is the real number. Move the
   pointer and you are moving the input vector: units recolour by
   activation, edges by w·a, and a pulse sweeps left to right
   along whichever layer is currently propagating.
   ------------------------------------------------------------ */
defineWidget('hero-demo', node => {
  const cv = el('canvas', { style: 'cursor:crosshair' });   // beats .pg-canvas-wrap canvas
  // touch-action:auto overrides .pg-canvas-wrap — a finger swiping the
  // full-width mobile hero must still scroll the page
  const wrap = el('div', {
    class: 'pg-canvas-wrap',
    style: 'border:none;background:transparent;touch-action:auto',
  }, cv);
  node.appendChild(wrap);

  // the plot is here for its canvas plumbing (DPI, resize, theme tracking);
  // everything below is laid out in CSS pixels off p.w / p.h
  const plot = trackPlot(new Plot(cv, {
    xmin: 0, xmax: 1, ymin: 0, ymax: 1, aspect: 1.26, equal: false, pad: 0,
  }));

  /* ---------- the task: two interleaved spirals ---------- */
  const NPTS = 170, LIM = 1.06;
  const data = [];
  {
    const r = rng(9);
    for (let i = 0; i < NPTS; i++) {
      const cls = i % 2;
      const t = .22 + 3.1 * (i / NPTS);
      const a = t * 2.1 + cls * Math.PI;
      const rad = t * .30;
      data.push([rad * Math.cos(a) + gauss(r) * .034,
                 rad * Math.sin(a) + gauss(r) * .034, cls]);
    }
  }

  /* ---------- the network ---------- */
  const SIZES = [2, 8, 8, 2], NL = SIZES.length - 1;
  const net = (() => {
    const r = rng(3), W = [], b = [];
    for (let l = 0; l < NL; l++) {
      const sc = Math.sqrt(2 / SIZES[l]);
      W.push(Array.from({ length: SIZES[l + 1] }, () =>
        Array.from({ length: SIZES[l] }, () => gauss(r) * sc)));
      b.push(new Array(SIZES[l + 1]).fill(0));
    }
    return { W, b };
  })();

  const softmax2 = (a, b) => {
    const m = Math.max(a, b);
    const e0 = Math.exp(a - m), e1 = Math.exp(b - m);
    return e1 / (e0 + e1);
  };

  /** Runs the network, leaving every layer's activations in `acts`. */
  const acts = [];
  function forward(x0, x1) {
    let a = [x0, x1];
    acts[0] = a;
    for (let l = 0; l < NL; l++) {
      const z = net.W[l].map((row, i) => {
        let s = net.b[l][i];
        for (let j = 0; j < row.length; j++) s += row[j] * a[j];
        return s;
      });
      a = l === NL - 1 ? z : z.map(Math.tanh);
      acts[l + 1] = a;
    }
    return softmax2(a[0], a[1]);
  }

  // Full-batch momentum SGD. 400 epochs of this costs ~224 ms, which is far too
  // much to block a landing page with, so it runs a slice per frame instead —
  // the network visibly sharpens while the page settles. Seeds 3/100/197/294/391
  // all reach 100% within 212 epochs at this learning rate.
  const EPOCHS = 400, LR = .35, MU = .9;
  const vel = {
    W: net.W.map(m => m.map(row => row.map(() => 0))),
    b: net.b.map(v => v.map(() => 0)),
  };
  let trained = 0, trainAcc = 0;

  function epoch() {
    const gW = net.W.map(m => m.map(row => row.map(() => 0)));
    const gb = net.b.map(v => v.map(() => 0));
    for (const [px, py, lab] of data) {
      const p = forward(px, py);
      // softmax + cross-entropy collapses to (prediction − target)
      let d = [(1 - p) - (1 - lab), p - lab];
      for (let l = NL - 1; l >= 0; l--) {
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
    for (let l = 0; l < NL; l++) {
      for (let i = 0; i < net.W[l].length; i++) {
        vel.b[l][i] = MU * vel.b[l][i] - LR * gb[l][i] / n;
        net.b[l][i] += vel.b[l][i];
        for (let j = 0; j < net.W[l][i].length; j++) {
          vel.W[l][i][j] = MU * vel.W[l][i][j] - LR * gW[l][i][j] / n;
          net.W[l][i][j] += vel.W[l][i][j];
        }
      }
    }
    trained++;
  }

  const evaluate = () => data.reduce((s, [x, y, lab]) =>
    s + ((forward(x, y) > .5 ? 1 : 0) === lab ? 1 : 0), 0) / data.length;

  /* ---------- the input-space inset, cached to an offscreen buffer ---------- */
  const G = 46;
  const field = document.createElement('canvas');
  field.width = G; field.height = G;
  const fctx = field.getContext('2d');
  const fimg = fctx.createImageData(G, G);
  const hexRGB = hex => {
    hex = (hex || '#888').trim();
    if (hex[0] !== '#') return [136, 136, 136];
    return hex.length === 4
      ? hex.slice(1).split('').map(c => parseInt(c + c, 16))
      : [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));
  };
  function paintField() {
    const c1 = hexRGB(C.c1), c2 = hexRGB(C.c2);
    for (let gy = 0; gy < G; gy++) {
      const y = LIM - 2 * LIM * (gy / (G - 1));
      for (let gx = 0; gx < G; gx++) {
        const x = -LIM + 2 * LIM * (gx / (G - 1));
        const p = forward(x, y);
        const conf = Math.abs(p - .5) * 2;
        const c = p > .5 ? c2 : c1;
        const k = (gy * G + gx) * 4;
        fimg.data[k] = c[0]; fimg.data[k + 1] = c[1]; fimg.data[k + 2] = c[2];
        fimg.data[k + 3] = 255 * (.10 + .32 * conf * conf);
      }
    }
    fctx.putImageData(fimg, 0, 0);
  }
  paintField();

  /* ---------- state ---------- */
  let probe = [.62, .30], target = probe.slice(), pOut = .5;
  let hovering = false, clock = 0, pulse = 0;

  const MONO = () => css('--font-mono');
  const roundRect = (ctx, x, y, w, h, r) => {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  };

  /** Column x, node y and radius for the current canvas size. */
  function layout(p) {
    const W = p.w, H = p.h;
    const compact = W < 470;                  // no room for the inset on phones
    const inset = compact ? 0 : Math.max(96, Math.min(132, H * .30));
    const padL = compact ? 18 : 14 + inset + 34;
    const padR = compact ? 56 : 76;           // room for the output probabilities
    const padT = compact ? 44 : 16;
    const capY = H - 20;                      // baseline for the column captions
    const r = Math.max(5.5, Math.min(11, H * .026));
    // size the unit band off its own extent, so the top and bottom margins
    // come out even instead of the stack riding high with the captions pinned
    const bot = capY - 14;
    const gap = Math.min((bot - padT - 2 * r) / (SIZES[1] - 1), 52);
    const cy = (padT + bot) / 2;
    const colX = [0, 1, 2, 3].map(l => padL + (W - padL - padR) * l / 3);
    const rows = SIZES.map(n => {
      const g = n === 2 ? gap * 2.1 : gap;    // spread the 2-unit ends a little
      return Array.from({ length: n }, (_, i) => cy + (i - (n - 1) / 2) * g);
    });
    return { W, H, compact, inset, colX, rows, r, cy, capY };
  }

  /* ---------- drawing ---------- */
  plot.onDraw(p => {
    const ctx = p.ctx;
    const L = layout(p);
    p.clear(null);
    pOut = forward(probe[0], probe[1]);

    // violet/amber mean *class* (inset + output units); signed activations and
    // the signals on the edges get their own diverging pair so the two readings
    // never collide
    const c1 = C.c1, c2 = C.c2;
    const sgn = v => (v >= 0 ? C.c3 : C.c4);

    /* --- edges, coloured by the signal each one carries --- */
    for (let l = 0; l < NL; l++) {
      const a = acts[l];
      let cap = 1e-6;
      for (let i = 0; i < net.W[l].length; i++)
        for (let j = 0; j < a.length; j++)
          cap = Math.max(cap, Math.abs(net.W[l][i][j] * a[j]));

      for (let i = 0; i < net.W[l].length; i++) {
        for (let j = 0; j < a.length; j++) {
          const s = net.W[l][i][j] * a[j];
          const t = Math.min(1, Math.abs(s) / cap);
          ctx.beginPath();
          ctx.moveTo(L.colX[l], L.rows[l][j]);
          ctx.lineTo(L.colX[l + 1], L.rows[l + 1][i]);
          ctx.strokeStyle = sgn(s);
          ctx.globalAlpha = .05 + .40 * t * t;
          ctx.lineWidth = .5 + 1.7 * t * t;
          ctx.stroke();
        }
      }

      /* --- the pulse: one layer at a time, sweeping left to right --- */
      const f = pulse * NL - l;
      if (f > 0 && f < 1) {
        const e = f * f * (3 - 2 * f);        // ease so it does not jerk at the ends
        const glow = Math.sin(Math.PI * f);
        for (let i = 0; i < net.W[l].length; i++) {
          for (let j = 0; j < a.length; j++) {
            const s = net.W[l][i][j] * a[j];
            const t = Math.min(1, Math.abs(s) / cap);
            if (t < .18) continue;
            ctx.beginPath();
            ctx.arc(L.colX[l] + (L.colX[l + 1] - L.colX[l]) * e,
                    L.rows[l][j] + (L.rows[l + 1][i] - L.rows[l][j]) * e,
                    1.1 + 1.5 * t, 0, Math.PI * 2);
            ctx.fillStyle = sgn(s);
            ctx.globalAlpha = .85 * t * glow;
            ctx.fill();
          }
        }
      }
    }
    ctx.globalAlpha = 1;

    /* --- units --- */
    for (let l = 0; l < SIZES.length; l++) {
      // the output layer shows probabilities, every other layer its activations
      const vals = l === NL
        ? [1 - pOut, pOut]
        : acts[l].map(v => (l === 0 ? v / LIM : v));
      for (let i = 0; i < SIZES[l]; i++) {
        const x = L.colX[l], y = L.rows[l][i], v = vals[i];
        const mag = l === NL ? v : Math.min(1, Math.abs(v));   // output: P(class)
        const col = l === NL ? (i ? c2 : c1) : sgn(v);
        ctx.beginPath();
        ctx.arc(x, y, L.r, 0, Math.PI * 2);
        ctx.fillStyle = css('--bg');
        ctx.fill();
        const floor = l === NL ? .22 : .12;
        ctx.globalAlpha = floor + (1 - floor) * mag;
        ctx.fillStyle = col;
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.lineWidth = 1.2;
        ctx.strokeStyle = C.grid;
        ctx.stroke();
      }
    }

    /* --- output probabilities, to the right of the output units --- */
    const win = pOut > .5 ? 1 : 0;
    ctx.font = `600 ${L.compact ? 10 : 11}px ${MONO()}`;
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    for (let i = 0; i < 2; i++) {
      ctx.fillStyle = i === win ? (i ? c2 : c1) : C.muted;
      ctx.globalAlpha = i === win ? 1 : .55;
      ctx.fillText((i ? pOut : 1 - pOut).toFixed(2), L.colX[3] + L.r + 9, L.rows[3][i]);
    }
    ctx.globalAlpha = 1;

    /* --- column captions --- */
    ctx.font = `10px ${MONO()}`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillStyle = C.muted; ctx.globalAlpha = .62;
    ['input', 'tanh', 'tanh', 'softmax'].forEach((s, l) =>
      ctx.fillText(s, L.colX[l], L.capY));
    ctx.globalAlpha = 1;

    /* --- the input-space inset --- */
    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    if (!L.compact) {
      const S = L.inset, ix = 14, iy = L.cy - S / 2;
      ctx.save();
      roundRect(ctx, ix, iy, S, S, 8);
      ctx.clip();
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(field, ix, iy, S, S);
      const toPad = ([x, y]) => [ix + S * (x + LIM) / (2 * LIM), iy + S * (LIM - y) / (2 * LIM)];
      ctx.globalAlpha = .95;
      for (const [x, y, lab] of data) {
        const [sx, sy] = toPad([x, y]);
        ctx.beginPath();
        ctx.arc(sx, sy, 1.6, 0, Math.PI * 2);
        ctx.fillStyle = lab ? c2 : c1;
        ctx.fill();
      }
      ctx.globalAlpha = 1;
      const [hx, hy] = toPad(probe);
      ctx.beginPath(); ctx.arc(hx, hy, 6, 0, Math.PI * 2);
      ctx.strokeStyle = css('--bg'); ctx.lineWidth = 4; ctx.stroke();
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1.8; ctx.stroke();
      ctx.beginPath(); ctx.arc(hx, hy, 1.7, 0, Math.PI * 2);
      ctx.fillStyle = C.ink; ctx.fill();
      ctx.restore();
      roundRect(ctx, ix, iy, S, S, 8);
      ctx.strokeStyle = C.grid; ctx.lineWidth = 1; ctx.stroke();

      ctx.font = `10px ${MONO()}`;
      ctx.fillStyle = C.muted; ctx.globalAlpha = .62;
      ctx.fillText(trained < EPOCHS ? 'learning the spirals…' : 'pointer sets the input',
                   ix, iy + S + 9);
      ctx.globalAlpha = .48;
      ctx.fillText(trained < EPOCHS
        ? `epoch ${trained} / ${EPOCHS}`
        : `2-8-8-2 · ${(trainAcc * 100).toFixed(0)}% on ${NPTS} pts`, ix, iy + S + 24);
      ctx.globalAlpha = 1;
    } else {
      ctx.font = `10px ${MONO()}`;
      ctx.fillStyle = C.muted; ctx.globalAlpha = .6;
      ctx.fillText('feed-forward · 2-8-8-2 · tanh', 16, 12);
      ctx.globalAlpha = .45;
      ctx.fillText(trained < EPOCHS ? `learning the spirals… ${trained}/${EPOCHS}`
                                     : 'pointer sets the input', 16, 26);
      ctx.globalAlpha = 1;
    }
  });

  /* ---------- interaction ---------- */
  wrap.addEventListener('pointermove', e => {
    const r = cv.getBoundingClientRect();
    hovering = true;
    target = [((e.clientX - r.left) / r.width - .5) * 2 * LIM,
              -((e.clientY - r.top) / r.height - .5) * 2 * LIM];
  });
  wrap.addEventListener('pointerleave', () => { hovering = false; });

  /* ---------- animation ---------- */
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const PULSE_MS = 1900;
  let raf = null, visible = true, last = 0;

  const io = new IntersectionObserver(es => { visible = es[0].isIntersecting; });
  io.observe(wrap);

  // the inset field is an offscreen buffer, so a theme flip has to repaint it
  // explicitly — trackPlot only re-runs onDraw
  const onTheme = () => { paintField(); plot.renderNow(); };
  window.addEventListener('themechange', onTheme);

  function tick(now) {
    const dt = last ? Math.min(100, now - last) : 16;
    last = now;
    if (visible) {
      clock += dt;
      if (trained < EPOCHS) {
        // ~7 ms a frame, so the whole 400 epochs land in about half a second
        // without ever blocking long enough to drop one
        const per = reduce ? 60 : 12;
        for (let i = 0; i < per && trained < EPOCHS; i++) epoch();
        if (trained >= EPOCHS) trainAcc = evaluate();
        // repainting the inset costs about as much as three epochs, so do it
        // every third frame while training rather than every frame
        if (trained >= EPOCHS || trained % (per * 3) === 0) paintField();
      }
      if (!reduce) {
        pulse = (clock % PULSE_MS) / PULSE_MS;
        if (!hovering) {
          // drift through the spiral so the network keeps lighting up on its own
          const t = clock / 1000;
          const rad = LIM * (.30 + .58 * Math.abs(Math.sin(t * .19)));
          target = [rad * Math.cos(t * .41), rad * Math.sin(t * .41)];
        }
      }
      probe = [probe[0] + (target[0] - probe[0]) * .12,
               probe[1] + (target[1] - probe[1]) * .12];
      plot.renderNow();
    }
    raf = requestAnimationFrame(tick);
  }
  raf = requestAnimationFrame(tick);

  window.addEventListener('pagehide', () => {
    cancelAnimationFrame(raf); io.disconnect();
    window.removeEventListener('themechange', onTheme);
  });
});

mountWidgets();
