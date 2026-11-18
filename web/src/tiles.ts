// Online base map drawn UNDER the route, so the route can be read against real streets and place names.
//
// Tiles are standard Web-Mercator XYZ tiles (default: the OpenStreetMap standard map, free, no key; the app darkens it with a CSS
// filter to fit the black UI). They are drawn on their own
// canvas with exactly the same view transform as the route layer. Each tile is placed by its longitude/latitude corners run through
// the SAME local projection as the route (scene.k), so the map and the route cannot drift apart.
//
// Needs internet. If no tile loads, onStatus('fail') fires and the app falls back to the offline vector road network.
//   ?base=0                          turn the base map off
//   ?tiles=https://.../{z}/{x}/{y}.png   use another tile server ({s} subdomain and {r} retina suffix are optional)
import type { Scene, View } from './mapDraw';

export type TileStatus = 'idle' | 'loading' | 'ok' | 'fail' | 'off';   // 'off' = switched off with ?base=0

const DEFAULT = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png';
const SUBS = ['a', 'b', 'c', 'd'];
const MAX_TILES = 60, MAX_CACHE = 400;

interface Entry { img: HTMLImageElement; state: 'loading' | 'ok' | 'fail' }

const tileLon = (x: number, n: number) => x / n * 360 - 180;
const tileLat = (y: number, n: number) => Math.atan(Math.sinh(Math.PI * (1 - 2 * y / n))) * 180 / Math.PI;
const lonToX = (lon: number, n: number) => (lon + 180) / 360 * n;
const latToY = (lat: number, n: number) => { const r = lat * Math.PI / 180; return (1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2 * n; };

export class TileLayer {
  private cache = new Map<string, Entry>();
  private tpl: string | null;
  private ok = 0; private failed = 0; private status: TileStatus = 'idle';

  constructor(private redraw: () => void, private onStatus: (s: TileStatus) => void) {
    const q = new URLSearchParams(location.search);
    this.tpl = q.get('base') === '0' ? null : (q.get('tiles') || DEFAULT);
  }
  get enabled() { return this.tpl !== null; }
  /** The default OpenStreetMap map is light, so the app darkens it; a custom ?tiles= server is shown as it is. */
  get darken() { return !new URLSearchParams(location.search).get('tiles'); }

  private set(s: TileStatus) { if (s !== this.status) { this.status = s; this.onStatus(s); } }

  private get(z: number, x: number, y: number, retina: boolean): Entry {
    const key = `${z}/${x}/${y}/${retina ? 2 : 1}`;
    let e = this.cache.get(key);
    if (e) return e;
    const img = new Image();
    e = { img, state: 'loading' };
    this.cache.set(key, e);
    if (this.cache.size > MAX_CACHE) { const first = this.cache.keys().next().value; if (first) this.cache.delete(first); }
    if (this.status === 'idle') this.set('loading');
    img.onload = () => { e!.state = 'ok'; this.ok++; this.set('ok'); this.redraw(); };
    // a failed tile is forgotten after 5 s so a passing network problem does not leave a permanent hole in the map
    img.onerror = () => { e!.state = 'fail'; this.failed++; if (this.ok === 0 && this.failed >= 6) this.set('fail'); setTimeout(() => { if (this.cache.get(key) === e) this.cache.delete(key); }, 5000); };
    img.src = this.tpl!.replace('{s}', SUBS[(x + y) % SUBS.length]).replace('{z}', String(z)).replace('{x}', String(x)).replace('{y}', String(y)).replace('{r}', retina ? '@2x' : '');
    return e;
  }

  clear(canvas: HTMLCanvasElement) { const c = canvas.getContext('2d'); if (c) { c.setTransform(1, 0, 0, 1, 0, 0); c.clearRect(0, 0, canvas.width, canvas.height); } }

  draw(canvas: HTMLCanvasElement, scene: Scene, v: View, W: number, H: number, dpr: number) {
    this.clear(canvas);
    if (!this.tpl || !(v.s > 0) || !isFinite(v.s) || W < 10 || H < 10) return;
    const ctx = canvas.getContext('2d')!, { lon0, lat0, kx, ky } = scene.k;
    // visible window in metres (screen px -> world), then in degrees
    const west = lon0 + (0 - v.tx) / v.s / kx, east = lon0 + (W - v.tx) / v.s / kx;
    const north = lat0 - (0 - v.ty) / v.s / ky, south = lat0 - (H - v.ty) / v.s / ky;
    // zoom whose tile pixels are about one screen pixel at this scale; fewer tiles if the window is huge
    // (templates without a {r} retina variant get one zoom level more on very sharp screens so they do not look blurry)
    const zBias = !this.tpl.includes('{r}') && dpr >= 2 ? 1 : 0;
    let z = Math.round(Math.log2(156543.03 * Math.cos(lat0 * Math.PI / 180) * v.s)) + zBias;
    z = Math.max(2, Math.min(19, z));
    let xa = 0, xb = 0, ya = 0, yb = 0, n = 1;
    for (;;) {
      n = 2 ** z;
      xa = Math.floor(lonToX(west, n)); xb = Math.min(Math.floor(lonToX(east, n)), xa + n);   // never more than one world width
      ya = Math.floor(latToY(Math.min(north, 85), n)); yb = Math.floor(latToY(Math.max(south, -85), n));
      if ((xb - xa + 1) * (yb - ya + 1) <= MAX_TILES || z <= 2) break;
      z--;
    }
    const retina = this.tpl.includes('{r}') && dpr >= 1.5, eps = 0.6 / v.s;   // overlap tiles slightly to hide seams
    ctx.setTransform(dpr * v.s, 0, 0, dpr * v.s, dpr * v.tx, dpr * v.ty);
    ctx.imageSmoothingEnabled = true; ctx.globalAlpha = 0.95;
    for (let y = Math.max(0, ya); y <= Math.min(n - 1, yb); y++) {
      for (let x = xa; x <= xb; x++) {
        const xm = ((x % n) + n) % n;
        const X = (tileLon(x, n) - lon0) * kx, Y = -(tileLat(y, n) - lat0) * ky;
        const w = (tileLon(x + 1, n) - tileLon(x, n)) * kx, h = (tileLat(y, n) - tileLat(y + 1, n)) * ky;
        const e = this.get(z, xm, y, retina);
        if (e.state === 'ok') { ctx.drawImage(e.img, X - eps / 2, Y - eps / 2, w + eps, h + eps); continue; }
        // not loaded yet: show the quadrant of the coarser parent tile if we already have it (avoids flashing blank while zooming)
        const pe = z > 2 ? this.cache.get(`${z - 1}/${xm >> 1}/${y >> 1}/${retina ? 2 : 1}`) : undefined;
        if (pe && pe.state === 'ok') {
          const iw = pe.img.naturalWidth / 2, ih = pe.img.naturalHeight / 2;
          ctx.drawImage(pe.img, (xm & 1) * iw, (y & 1) * ih, iw, ih, X - eps / 2, Y - eps / 2, w + eps, h + eps);
        }
      }
    }
    ctx.globalAlpha = 1;
  }
}
