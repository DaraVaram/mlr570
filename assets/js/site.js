/* ============================================================
   site.js — shell behaviour: theme, nav, TOC scrollspy, anchors
   ============================================================ */

/* ---------- Theme ------------------------------------------------- */
const THEME_KEY = 'mlr570-theme';

function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  const btn = document.getElementById('theme-toggle');
  if (btn) {
    btn.setAttribute('aria-label',
      t === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
    btn.setAttribute('title',
      t === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
  }
  window.dispatchEvent(new CustomEvent('themechange', { detail: t }));
}

function initTheme() {
  const btn = document.getElementById('theme-toggle');
  btn?.addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    try { localStorage.setItem(THEME_KEY, next); } catch {}
    applyTheme(next);
  });
  // sync the aria-label with whatever the inline boot script already set
  applyTheme(document.documentElement.dataset.theme || 'light');

  // follow the OS only while the user has made no explicit choice
  const mq = window.matchMedia('(prefers-color-scheme: dark)');
  mq.addEventListener?.('change', e => {
    let stored = null;
    try { stored = localStorage.getItem(THEME_KEY); } catch {}
    if (!stored) applyTheme(e.matches ? 'dark' : 'light');
  });
}

/* ---------- Mobile nav -------------------------------------------- */
function initNav() {
  const toggle = document.getElementById('nav-toggle');
  const links  = document.getElementById('nav-links');
  toggle?.addEventListener('click', () => {
    const open = links.classList.toggle('is-open');
    toggle.setAttribute('aria-expanded', String(open));
  });
  links?.addEventListener('click', e => {
    if (e.target.closest('a')) {
      links.classList.remove('is-open');
      toggle?.setAttribute('aria-expanded', 'false');
    }
  });

  // Sidebar (chapter pages, small screens)
  const sb = document.getElementById('sidebar');
  const sbToggle = document.getElementById('sidebar-toggle');
  sbToggle?.addEventListener('click', () => {
    const open = sb.classList.toggle('is-open');
    sbToggle.setAttribute('aria-expanded', String(open));
  });
  sb?.addEventListener('click', e => {
    if (e.target.closest('a') && window.innerWidth <= 1040) sb.classList.remove('is-open');
  });
}

/* ---------- Heading anchors + scrollspy ---------------------------- */
function initAnchors() {
  const content = document.querySelector('.content');
  if (!content) return;
  content.querySelectorAll('h2[id], h3[id]').forEach(h => {
    h.classList.add('anchor-h');
    const a = document.createElement('a');
    a.className = 'anchor-link';
    a.href = '#' + h.id;
    a.setAttribute('aria-label', `Link to section: ${h.textContent.trim()}`);
    a.textContent = '#';
    h.prepend(a);
  });
}

function initScrollspy() {
  const links = [...document.querySelectorAll('.toc a[href^="#"]')];
  if (!links.length) return;

  const byId = new Map();
  const targets = [];
  links.forEach(a => {
    const id = decodeURIComponent(a.getAttribute('href').slice(1));
    const t = document.getElementById(id);
    if (t) { byId.set(id, a); targets.push(t); }
  });
  if (!targets.length) return;

  const setActive = id => {
    links.forEach(a => a.classList.remove('is-active'));
    const a = byId.get(id);
    if (!a) return;
    a.classList.add('is-active');
    // keep the active item visible in a long sidebar
    const sb = document.querySelector('.sidebar');
    if (sb && sb.scrollHeight > sb.clientHeight + 10) {
      const ar = a.getBoundingClientRect(), sr = sb.getBoundingClientRect();
      if (ar.top < sr.top + 40 || ar.bottom > sr.bottom - 40) {
        sb.scrollTo({ top: sb.scrollTop + (ar.top - sr.top) - sb.clientHeight / 2.6,
                      behavior: 'smooth' });
      }
    }
  };

  let ticking = false;
  const onScroll = () => {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => {
      ticking = false;
      const line = (parseFloat(getComputedStyle(document.documentElement)
        .getPropertyValue('--nav-h')) || 60) + 60;
      let current = targets[0];
      for (const t of targets) {
        if (t.getBoundingClientRect().top <= line) current = t; else break;
      }
      // at the very bottom, always light up the last entry
      if (window.innerHeight + window.scrollY >= document.body.offsetHeight - 4) {
        current = targets[targets.length - 1];
      }
      setActive(current.id);
    });
  };
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
}

/* ---------- KaTeX ---------------------------------------------------- */
function renderMath() {
  if (!window.renderMathInElement) return;
  window.renderMathInElement(document.body, {
    delimiters: [
      { left: '$$',  right: '$$',  display: true  },
      { left: '\\[', right: '\\]', display: true  },
      { left: '$',   right: '$',   display: false },
      { left: '\\(', right: '\\)', display: false },
    ],
    ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'option'],
    ignoredClasses: ['no-math'],
    throwOnError: false,
    strict: false,
    trust: false,
    macros: {
      '\\R': '\\mathbb{R}',
      '\\E': '\\mathbb{E}',
      '\\P': '\\mathbb{P}',
      '\\Var': '\\operatorname{Var}',
      '\\Cov': '\\operatorname{Cov}',
      '\\rank': '\\operatorname{rank}',
      '\\nullity': '\\operatorname{nullity}',
      '\\Null': '\\operatorname{Null}',
      '\\tr': '\\operatorname{tr}',
      '\\proj': '\\operatorname{proj}',
      '\\T': '^{\\mathsf{T}}',
      '\\vv': '\\mathbf{#1}',
    },
  });
  document.body.classList.add('math-ready');
}

/* ---------- Copy-to-clipboard on code blocks -------------------------- */
function initCodeCopy() {
  document.querySelectorAll('pre').forEach(pre => {
    if (pre.dataset.nocopy != null) return;
    pre.style.position = 'relative';
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'code-copy';
    btn.textContent = 'Copy';
    btn.setAttribute('aria-label', 'Copy code to clipboard');
    btn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(pre.querySelector('code')?.innerText ?? pre.innerText);
        btn.textContent = 'Copied';
        btn.classList.add('is-done');
        setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('is-done'); }, 1600);
      } catch { btn.textContent = 'Press Ctrl+C'; }
    });
    pre.appendChild(btn);
  });
}

/* ---------- Boot ------------------------------------------------------- */
function boot() {
  initTheme();
  initNav();
  initAnchors();
  initScrollspy();
  initCodeCopy();
  renderMath();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}

// KaTeX loads with `defer`, so it may land after DOMContentLoaded.
window.addEventListener('load', () => {
  if (!document.body.classList.contains('math-ready')) renderMath();
});
