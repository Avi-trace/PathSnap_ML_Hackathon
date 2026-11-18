import { useEffect, useRef } from 'react';
import { createFx, type Fx } from './fx';
import { FULL, bloom, buildScene, drawScene, fitView, scaleLabel, type DrawOpts, type Scene, type View } from './mapDraw';
import { TileLayer, type TileStatus } from './tiles';
import type { PredictResult } from './types';

export interface MapHandle { redraw(): void }

interface Props {
  result: PredictResult | null;
  opts?: DrawOpts;                 // layer alphas / reveal fractions; mutate and call handle.redraw() for scrubbing
  interactive?: boolean;           // pan, zoom, double-click fit, route reveal animation
  onScale?: (px: number, label: string) => void;
  handle?: { current: MapHandle | null };
  base?: boolean;                  // draw an online street map under the route (the app turns it on; the scroll intro does not)
  onBase?: (s: TileStatus) => void;
}

// 2D canvas map (no tiles) with an optional WebGL post pass drawn on top (see fx.ts).
export function MapCanvas({ result, opts = FULL, interactive = true, onScale, handle, base = false, onBase }: Props) {
  const cv = useRef<HTMLCanvasElement>(null), gl = useRef<HTMLCanvasElement>(null), tc = useRef<HTMLCanvasElement>(null);
  const st = useRef({ scene: null as Scene | null, v: { s: 1, tx: 0, ty: 0 } as View, reveal: 1, W: 0, H: 0, dpr: 1, opts, raf: 0, queued: 0, small: document.createElement('canvas'), fx: null as Fx | null, onScale, dts: [] as number[], last: 0, tiles: null as TileLayer | null, base, onBase, fitS: 1 });
  st.current.opts = opts; st.current.onScale = onScale; st.current.base = base; st.current.onBase = onBase;

  // Street map under the route: separate canvas, same view transform, created the first time it is wanted.
  const drawTiles = () => {
    const S = st.current, t = tc.current; if (!t) return;
    if (S.base && !S.tiles) { S.tiles = new TileLayer(() => schedule(), s => S.onBase?.(s)); if (!S.tiles.enabled) S.onBase?.('off'); }
    // the light OpenStreetMap tiles are inverted into a dark map so the gold route stays the brightest thing on screen
    if (S.tiles) t.style.filter = S.tiles.darken ? 'invert(1) hue-rotate(180deg) brightness(.82) contrast(.92) saturate(.55)' : '';
    if (S.tiles && S.base && S.scene) S.tiles.draw(t, S.scene, S.v, S.W, S.H, S.dpr); else S.tiles?.clear(t);
  };

  // Whatever goes wrong with the WebGL pass (blank output, render error, lost GPU context, slow frames), fall back to the
  // plain 2D canvas, which is always drawn. The map must never disappear because of a visual effect.
  const disableFx = (why: string) => {
    const S = st.current; if (!S.fx) return;
    try { S.fx.dispose(); } catch { /* context may already be gone */ }
    S.fx = null; if (cv.current) cv.current.style.opacity = '1'; if (gl.current) gl.current.style.display = 'none';
    console.warn('fx pass disabled:', why);
  };
  const renderFx = (c: HTMLCanvasElement) => { const S = st.current; if (!S.fx) return; try { S.fx.render(c); } catch (e) { disableFx((e as Error).message); } };

  const draw = () => {
    const S = st.current, c = cv.current; if (!c) return;
    drawTiles();
    const ctx = c.getContext('2d')!;
    if (!S.scene) { ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, c.width, c.height); renderFx(c); return; }
    const o = { ...S.opts, hmmFrac: S.opts.hmmFrac * S.reveal };
    drawScene(ctx, S.scene, S.v, S.W, S.H, S.dpr, o);
    const sc = scaleLabel(S.v.s); S.onScale?.(sc.px, sc.label);
    bloom(c, ctx, S.small);
    renderFx(c);
  };
  // Pan and zoom events can arrive faster than frames; draw at most once per frame.
  const schedule = () => {
    const S = st.current; if (S.queued) return;
    S.queued = requestAnimationFrame(t => { S.queued = 0; draw(); guard(t); });
  };
  // Frame-rate guard: if continuous pan/zoom frames are slow with the WebGL pass on (weak GPU or software GL),
  // drop the pass and keep the plain 2D canvas.
  const guard = (t: number) => {
    const S = st.current, dt = t - S.last; S.last = t;
    if (!S.fx || dt > 250) { S.dts = []; return; }          // idle gaps are not slow frames
    S.dts.push(dt); if (S.dts.length < 24) return;
    const med = S.dts.slice().sort((a, b) => a - b)[S.dts.length >> 1]; S.dts = [];
    if (med > 70) { disableFx(`slow frames ${med.toFixed(0)} ms`); draw(); }
  };
  const fit = () => { const S = st.current; if (S.scene) { const v = fitView(S.scene, S.W, S.H); if (v) { S.v = v; S.fitS = v.s; } } };

  useEffect(() => {  // canvas size, fx pass, interaction
    const c = cv.current!, g = gl.current!, S = st.current;
    S.fx = createFx(g, () => { disableFx('WebGL context lost'); draw(); });
    c.style.opacity = S.fx ? '0' : '1'; g.style.display = S.fx ? 'block' : 'none';
    if (handle) handle.current = { redraw: schedule };
    const resize = () => {
      const r = c.parentElement!.getBoundingClientRect(); S.dpr = devicePixelRatio || 1; S.W = r.width; S.H = r.height;
      c.width = Math.round(S.W * S.dpr); c.height = Math.round(S.H * S.dpr); c.style.width = g.style.width = S.W + 'px'; c.style.height = g.style.height = S.H + 'px';
      const t = tc.current; if (t) { t.width = c.width; t.height = c.height; t.style.width = S.W + 'px'; t.style.height = S.H + 'px'; }
      fit(); draw();
    };
    const ro = new ResizeObserver(resize); ro.observe(c.parentElement!);
    let drag: { x: number; y: number; tx: number; ty: number } | null = null;
    const down = (e: PointerEvent) => { drag = { x: e.clientX, y: e.clientY, tx: S.v.tx, ty: S.v.ty }; try { c.setPointerCapture(e.pointerId); } catch { /* not an active pointer */ } };
    const move = (e: PointerEvent) => { if (!drag) return; S.v.tx = drag.tx + e.clientX - drag.x; S.v.ty = drag.ty + e.clientY - drag.y; schedule(); };
    const up = () => { drag = null; };
    const wheel = (e: WheelEvent) => {
      if (!S.scene) return; e.preventDefault();
      const r = c.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top, k = Math.exp(-e.deltaY * 0.0015);
      // zoom limits: no further out than a quarter of the fitted view, no further in than 60 px per metre (keeps tiles and scale bar sane)
      const ns = Math.min(Math.max(S.v.s * k, S.fitS * 0.25), 60), kk = ns / S.v.s;
      S.v.s = ns; S.v.tx = mx - (mx - S.v.tx) * kk; S.v.ty = my - (my - S.v.ty) * kk; schedule();
    };
    const dbl = () => { fit(); draw(); };
    if (interactive) {
      c.addEventListener('pointerdown', down); c.addEventListener('pointermove', move); c.addEventListener('pointerup', up);
      c.addEventListener('wheel', wheel, { passive: false }); c.addEventListener('dblclick', dbl);
    }
    return () => {
      ro.disconnect(); cancelAnimationFrame(S.raf); cancelAnimationFrame(S.queued); S.queued = 0; S.fx?.dispose(); S.fx = null;
      if (handle) handle.current = null;
      c.removeEventListener('pointerdown', down); c.removeEventListener('pointermove', move); c.removeEventListener('pointerup', up);
      c.removeEventListener('wheel', wheel); c.removeEventListener('dblclick', dbl);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {  // new result
    const S = st.current; cancelAnimationFrame(S.raf);
    if (!result) { S.scene = null; draw(); return; }
    S.scene = buildScene(result); fit();
    if (!interactive) { S.reveal = 1; draw(); return; }
    const t0 = performance.now();
    const f = (t: number) => { S.reveal = 1 - Math.pow(1 - Math.min(1, (t - t0) / 1400), 3); draw(); if (S.reveal < 1) S.raf = requestAnimationFrame(f); };
    S.raf = requestAnimationFrame(f);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result]);

  useEffect(() => { draw(); }, [opts]);  // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { draw(); }, [base]);  // eslint-disable-line react-hooks/exhaustive-deps

  return (<>
    <canvas ref={tc} className="map tiles" aria-hidden="true" />
    <canvas ref={cv} className="map" aria-label="Map of GPS observations and reconstructed route" />
    <canvas ref={gl} className="map fx" aria-hidden="true" />
  </>);
}
