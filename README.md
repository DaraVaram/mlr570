# MLR 570 — Advanced Machine Learning

Interactive course notes for **MLR 570 – Advanced Machine Learning**, Department of Computer Science
and Engineering, American University of Sharjah.

- **Instructor:** Dr. Mohamed I. AlHajri
- **Teaching Assistant / notes author:** Dara Varam

The notes began as LaTeX. This site keeps the mathematics exactly as written and rebuilds the figures
as things students can pick up and move.

## Status

| Chapter | State |
|---|---|
| 1 · Mathematical Background | Live — 26 interactive figures |
| 2 · Data Fundamentals | In conversion |
| 3 · Supervised Learning | In conversion |
| 4 · Unsupervised Learning | In conversion |

## Running it locally

The site is entirely static — no build step, no dependencies to install. It does need to be served over
HTTP rather than opened as a `file://` URL, because the widgets are ES modules.

```bash
python -m http.server 8765
```

Then open <http://localhost:8765>.

## How it is built

Plain HTML, CSS and JavaScript. Mathematics is typeset with [KaTeX](https://katex.org) (loaded from a CDN);
everything else is local.

```
index.html              Landing page
notes.html              Chapter index
syllabus.html           Course description, schedule, assessment
about.html              People, method, errata policy
chapters/
  01-mathematical-background.html
assets/
  css/
    base.css            Design tokens, reset, typography
    layout.css          Nav, sidebar, page grid, footer
    components.css      Cards, callouts, playgrounds, controls
  js/
    site.js             Theme, nav, scrollspy, anchors, KaTeX
    viz.js              Canvas plotting + interaction library
    linalg.js           SVD, eigendecomposition, RREF, rank, projections
    widgets/
      index.js          Registration and lazy mounting
      la.js             Linear algebra figures
      prob.js           Probability and statistics figures
      calc.js           Calculus, optimisation and complexity figures
  img/
```

### Adding a figure

Register a widget, then drop a placeholder into the page. Widgets mount lazily as they approach the
viewport.

```js
// assets/js/widgets/la.js
defineWidget('my-figure', node => {
  const { right, canvas } = split(node, { hint: 'Drag me' });
  const plot = trackPlot(new Plot(canvas, { xmin: -5, xmax: 5, ymin: -5, ymax: 5 }));
  plot.onDraw(p => { p.grid(1); p.axes({ ticks: 1 }); /* ... */ });
});
```

```html
<div class="playground">
  <div class="playground__head"><!-- title, subtitle, badge --></div>
  <div class="playground__body" data-widget="my-figure"></div>
</div>
```

Plot colours are read from CSS custom properties, so figures follow the light/dark theme automatically —
use `C.c1`, `C.c2`, … rather than hard-coded hex values.

### Numerics

`linalg.js` implements the heavier routines from scratch (one-sided Jacobi SVD, cyclic Jacobi symmetric
eigendecomposition, RREF with partial pivoting) so the pages carry no runtime dependencies. Figures compute
results live rather than displaying pre-rendered answers.

## Corrections

Every worked example was recomputed while converting it from the original LaTeX. Corrections are applied to
the text and recorded openly in the [errata section](chapters/01-mathematical-background.html#errata) of
each chapter.

Found a mistake — in the mathematics or in a figure? Please report it to <mialhajri@aus.edu>.
