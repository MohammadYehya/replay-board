// Typeset LaTeX formulas as crisp transparent PNGs, for the PowerPoint export (app/core/texrender.py).
// MathJax 3 (TeX in, SVG out, no browser) does the typesetting; resvg rasterises the SVG. Run with node; it is
// never imported by the player bundle.
//
// stdin:  {"items": [{"id": str, "latex": str, "display": bool, "color": "#1F2A37", "px_per_em": 120}]}
// stdout: {"results": [{"id", "ok": true, "png": base64, "width_em", "height_em", "depth_em", "px": [w, h]}
//                      | {"id", "ok": false, "error": str}]}
// Sizes are in em of the formula's font size, so the caller can size the picture for any font size. Exit code 3:
// a package is missing (npm install in web/); 2: the input is not valid JSON.
import { existsSync } from 'node:fs';
import { createRequire } from 'node:module';
import { join } from 'node:path';

const require = createRequire(import.meta.url);
let mathjax, TeX, SVG, liteAdaptor, RegisterHTMLHandler, AllPackages, Resvg;
try {
  ({ mathjax } = require('mathjax-full/js/mathjax.js'));
  ({ TeX } = require('mathjax-full/js/input/tex.js'));
  ({ SVG } = require('mathjax-full/js/output/svg.js'));
  ({ liteAdaptor } = require('mathjax-full/js/adaptors/liteAdaptor.js'));
  ({ RegisterHTMLHandler } = require('mathjax-full/js/handlers/html.js'));
  ({ AllPackages } = require('mathjax-full/js/input/tex/AllPackages.js'));
  ({ Resvg } = require('@resvg/resvg-js'));
} catch (e) {
  process.stderr.write(`a package for typesetting equations is missing (run npm install in web/): ${String(e.message).split('\n')[0]}\n`);
  process.exit(3);
}

const PAD = 60; // font units (1000 per em) of transparent margin, so italic overhangs are not clipped
// A formula's picture is at most this big: \hspace{3000em} once asked resvg for 230 GB and killed the whole batch.
const MAX_SIDE = 16000;
const MAX_PIXELS = 24e6;

// Text MathJax has no glyphs for (\text{Å}, µ, é) is drawn with fonts. Loading every system font takes about a
// second per formula; Cambria (Cambria Math is in the same file) and Segoe UI Symbol load in milliseconds and cover
// Latin, Greek, Cyrillic, arrows and maths operators. Other scripts still get the system's fonts.
const winFonts = process.env.WINDIR ? join(process.env.WINDIR, 'Fonts') : '';
const QUICK_FONTS = winFonts ? ['cambria.ttc', 'seguisym.ttf'].map((f) => join(winFonts, f)).filter((f) => existsSync(f)) : [];
const QUICK_CHARS = /^[\u0000-\u052f\u1e00-\u1fff\u2000-\u214f\u2190-\u22ff]*$/u;

function fontOptions(svg) {
  if (!svg.includes('<text')) return { fonts: 'none', font: { loadSystemFonts: false, defaultFontFamily: 'Cambria' } };
  const text = [...svg.matchAll(/<text\b[^>]*>([^<]*)<\/text>/g)].map((m) => m[1]).join('');
  if (QUICK_FONTS.length && QUICK_FONTS[0].endsWith('cambria.ttc') && QUICK_CHARS.test(text)) {
    return { fonts: 'quick', font: { loadSystemFonts: false, fontFiles: QUICK_FONTS, defaultFontFamily: 'Cambria' } };
  }
  return { fonts: 'system', font: { loadSystemFonts: true, defaultFontFamily: 'Cambria' } };
}

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
// noerrors/noundefined would draw a broken formula in red instead of failing it; html/bussproofs are not needed.
const packages = AllPackages.filter((p) => !['noerrors', 'noundefined', 'html', 'bussproofs'].includes(p));
const tex = new TeX({
  packages,
  formatError: (_jax, err) => {
    throw err;
  },
});
const svgOut = new SVG({ fontCache: 'none' });
const doc = mathjax.document('', { InputJax: tex, OutputJax: svgOut });

function readStdin() {
  return new Promise((resolve, reject) => {
    const chunks = [];
    process.stdin.on('data', (c) => chunks.push(c));
    process.stdin.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    process.stdin.on('error', reject);
  });
}

function attr(tag, name) {
  const m = tag.match(new RegExp(`\\s${name}="([^"]*)"`));
  return m ? m[1] : null;
}

function setAttr(tag, name, value) {
  const re = new RegExp(`\\s${name}="[^"]*"`);
  return re.test(tag) ? tag.replace(re, ` ${name}="${value}"`) : tag.replace(/^<svg/, `<svg ${name}="${value}"`);
}

function renderOne(item) {
  const latex = String(item.latex ?? '').trim();
  if (!latex) throw new Error('empty formula');
  const color = /^#[0-9a-fA-F]{6}$/.test(String(item.color)) ? item.color : '#000000';
  const pxPerEm = Math.min(400, Math.max(20, Number(item.px_per_em) || 120));
  const node = doc.convert(latex, { display: Boolean(item.display), em: 16, ex: 8, containerWidth: 1280 });
  let svg = adaptor.outerHTML(adaptor.firstChild(node));
  if (svg.includes('data-mml-node="merror"') || svg.includes('data-mjx-error')) throw new Error('the formula does not typeset');
  const open = svg.match(/^<svg[^>]*>/);
  if (!open) throw new Error('no picture');
  const vb = (attr(open[0], 'viewBox') || '').trim().split(/\s+/).map(Number);
  if (vb.length !== 4 || vb.some((v) => !Number.isFinite(v)) || vb[2] <= 0 || vb[3] <= 0) throw new Error('no size');
  const [x, y, w, h] = [vb[0] - PAD, vb[1] - PAD, vb[2] + 2 * PAD, vb[3] + 2 * PAD];
  const pxW = Math.max(1, Math.ceil((w / 1000) * pxPerEm));
  const pxH = Math.max(1, Math.ceil((h / 1000) * pxPerEm));
  if (pxW > MAX_SIDE || pxH > MAX_SIDE || pxW * pxH > MAX_PIXELS) throw new Error('the formula is too big to draw');
  let head = open[0];
  head = setAttr(head, 'viewBox', `${x} ${y} ${w} ${h}`);
  head = setAttr(head, 'width', String(pxW));
  head = setAttr(head, 'height', String(pxH));
  head = head.replace(/\sstyle="[^"]*"/, '');
  svg = head + svg.slice(open[0].length);
  svg = svg.replace(/currentColor/g, color);
  const { fonts, font } = fontOptions(svg);
  const resvg = new Resvg(svg, { background: 'rgba(0,0,0,0)', fitTo: { mode: 'original' }, font });
  const png = resvg.render().asPng();
  return {
    ok: true,
    png: Buffer.from(png).toString('base64'),
    width_em: w / 1000,
    height_em: h / 1000,
    depth_em: (h + y) / 1000, // below the baseline (the viewBox's y is minus the height above it)
    px: [pxW, pxH],
    fonts, // which fonts drew its text: none, quick or system (for tests)
  };
}

const raw = await readStdin();
let input;
try {
  input = JSON.parse(raw);
} catch {
  process.stderr.write('the input is not JSON\n');
  process.exit(2);
}
const results = [];
for (const item of Array.isArray(input?.items) ? input.items : []) {
  try {
    results.push({ id: item.id, ...renderOne(item) });
  } catch (e) {
    const msg = String(e?.message ?? e).split('\n')[0].slice(0, 200);
    results.push({ id: item?.id, ok: false, error: msg || 'the formula does not typeset' });
  }
}
process.stdout.write(JSON.stringify({ results }));
