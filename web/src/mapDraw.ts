import type { PredictResult } from './types';

// Shared 2D map drawing, used by the app map and the scroll intro. No tiles, works offline.
// k = the local projection (degrees -> metres) the scene was built with; the base-map tiles use the same one so they line up with the route.
export interface Scene { net: Float32Array; gps: Float32Array; clf: Float32Array; brk: Float32Array; hmm: Float32Array; k: { lon0: number; lat0: number; kx: number; ky: number } }
export interface View { s: number; tx: number; ty: number }
// Per-layer alpha (0..1) and how much of each layer is revealed (0..1).
// roadColor / roadW are optional: the scroll intro keeps the original dark thin roads, the app asks for brighter ones.
export interface DrawOpts { net: number; gps: number; clf: number; brk: number; hmm: number; gpsFrac: number; clfFrac: number; hmmFrac: number; roadColor?: string; roadW?: number }
export const FULL: DrawOpts = { net: 1, gps: 1, clf: 1, brk: 1, hmm: 1, gpsFrac: 1, clfFrac: 1, hmmFrac: 1 };

export const GOLD = '#ffaa33', WARN = '#ff6a55', ROAD = '#2a3336', MUTE = '#c4c7c7';

function flat(pairs: number[][], w: number, k: { lon0: number; lat0: number; kx: number; ky: number }) {
  const o = new Float32Array(pairs.length * w);
  pairs.forEach((p, i) => { for (let j = 0; j < w; j += 2) { o[i * w + j] = (p[j] - k.lon0) * k.kx; o[i * w + j + 1] = -(p[j + 1] - k.lat0) * k.ky; } });
  return o;
}

export function buildScene(r: PredictResult): Scene {
  // (a response without GPS points must not turn every coordinate into NaN: fall back to the first road segment)
  const g = r.gps, n = g.length, lon0 = n ? g.reduce((a, p) => a + p[0], 0) / n : (r.network[0]?.[0] ?? 0), lat0 = n ? g.reduce((a, p) => a + p[1], 0) / n : (r.network[0]?.[1] ?? 0);
  const k = { lon0, lat0, kx: 111320 * Math.cos(lat0 * Math.PI / 180), ky: 111132 };
  return { net: flat(r.network, 4, k), gps: flat(g, 2, k), clf: flat(r.classifier_only.arcs, 4, k), brk: flat(r.classifier_only.breaks, 2, k), hmm: flat(r.hmm.route, 2, k), k };
}

// Fit the GPS trace and the HMM route into W x H. Returns null only if the box is too small to be a real layout.
// The margin shrinks on small containers so the map stays large instead of collapsing.
export function fitView(D: Scene, W: number, H: number, pad = 80): View | null {
  if (W < 40 || H < 40) return null;
  const m = Math.min(pad, Math.min(W, H) * 0.12);
  let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  for (const a of [D.gps, D.hmm]) for (let i = 0; i < a.length; i += 2) { x0 = Math.min(x0, a[i]); x1 = Math.max(x1, a[i]); y0 = Math.min(y0, a[i + 1]); y1 = Math.max(y1, a[i + 1]); }
  if (x0 > x1) return null;                       // nothing to fit
  const s = Math.min((W - m * 2) / Math.max(x1 - x0, 150), (H - m * 2) / Math.max(y1 - y0, 150));   // at least 150 m across, so a 1-point trip is not zoomed absurdly far
  return { s, tx: W / 2 - s * (x0 + x1) / 2, ty: H / 2 - s * (y0 + y1) / 2 };
}

export function drawScene(ctx: CanvasRenderingContext2D, D: Scene, v: View, W: number, H: number, dpr: number, o: DrawOpts) {
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
  if (!(v.s > 0) || !isFinite(v.s)) return;
  ctx.setTransform(dpr * v.s, 0, 0, dpr * v.s, dpr * v.tx, dpr * v.ty);
  const px = 1 / v.s;
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  if (o.net > 0) {
    ctx.globalAlpha = o.net; ctx.beginPath();
    for (let i = 0; i < D.net.length; i += 4) { ctx.moveTo(D.net[i], D.net[i + 1]); ctx.lineTo(D.net[i + 2], D.net[i + 3]); }
    ctx.lineWidth = (o.roadW ?? 1.1) * px; ctx.strokeStyle = o.roadColor ?? ROAD; ctx.stroke();
  }
  if (o.gps > 0 && o.gpsFrac > 0) {
    ctx.globalAlpha = o.gps * 0.6; ctx.fillStyle = '#fff'; const r = 1.4 * px, n = Math.ceil(D.gps.length / 2 * o.gpsFrac);
    for (let i = 0; i < n; i++) ctx.fillRect(D.gps[i * 2] - r, D.gps[i * 2 + 1] - r, r * 2, r * 2);
  }
  if (o.clf > 0 && o.clfFrac > 0) {
    ctx.globalAlpha = o.clf; ctx.beginPath(); const n = Math.ceil(D.clf.length / 4 * o.clfFrac);
    for (let i = 0; i < n; i++) { const j = i * 4; ctx.moveTo(D.clf[j], D.clf[j + 1]); ctx.lineTo(D.clf[j + 2], D.clf[j + 3]); }
    ctx.setLineDash([5 * px, 4 * px]); ctx.lineWidth = 1.8 * px; ctx.strokeStyle = MUTE; ctx.stroke(); ctx.setLineDash([]);
  }
  if (o.hmm > 0 && o.hmmFrac > 0 && D.hmm.length) {
    ctx.globalAlpha = o.hmm; const n = Math.max(2, Math.floor(D.hmm.length / 2 * o.hmmFrac));
    ctx.beginPath(); ctx.moveTo(D.hmm[0], D.hmm[1]); for (let i = 1; i < n; i++) ctx.lineTo(D.hmm[i * 2], D.hmm[i * 2 + 1]);
    ctx.shadowColor = GOLD; ctx.shadowBlur = 14; ctx.lineWidth = 3 * px; ctx.strokeStyle = GOLD; ctx.stroke(); ctx.shadowBlur = 0;
    ctx.lineWidth = 1.5 * px; ctx.strokeStyle = '#fff'; ctx.beginPath(); ctx.arc(D.hmm[0], D.hmm[1], 6 * px, 0, 7); ctx.stroke();
    if (o.hmmFrac >= 1) { const e = D.hmm.length - 2; ctx.fillStyle = GOLD; ctx.beginPath(); ctx.arc(D.hmm[e], D.hmm[e + 1], 5 * px, 0, 7); ctx.fill(); }
  }
  if (o.brk > 0 && o.clfFrac > 0) {
    ctx.globalAlpha = o.brk; ctx.lineWidth = 1.2 * px; ctx.strokeStyle = WARN;
    for (let i = 0; i < D.brk.length; i += 2) { ctx.beginPath(); ctx.arc(D.brk[i], D.brk[i + 1], 4 * px, 0, 7); ctx.stroke(); }
  }
  ctx.globalAlpha = 1;
}

// Cheap bloom: shrink the frame into a small canvas, add it back upscaled and blurred.
export function bloom(c: HTMLCanvasElement, ctx: CanvasRenderingContext2D, small: HTMLCanvasElement) {
  const sw = Math.max(1, c.width >> 3), sh = Math.max(1, c.height >> 3);
  small.width = sw; small.height = sh;
  small.getContext('2d')!.drawImage(c, 0, 0, sw, sh);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalCompositeOperation = 'lighter'; ctx.globalAlpha = 0.55; ctx.filter = 'blur(10px)';
  ctx.drawImage(small, 0, 0, c.width, c.height);
  ctx.filter = 'none'; ctx.globalAlpha = 1; ctx.globalCompositeOperation = 'source-over';
}

export function scaleLabel(s: number) {
  const m = 90 / s, p = Math.pow(10, Math.floor(Math.log10(m))), f = m / p, nice = (f >= 5 ? 5 : f >= 2 ? 2 : 1) * p;
  return { px: nice * s, label: nice >= 1000 ? nice / 1000 + ' KM' : nice + ' M' };
}
