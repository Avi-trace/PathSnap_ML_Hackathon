import type { Health, PredictResult, Trip } from './types';

// All data comes from the live PathSnap API. There is no bundled or cached result data in this client.
const params = new URLSearchParams(location.search);
const KEY = 'pathsnap_api';
const store = {
  get: () => { try { return localStorage.getItem(KEY); } catch { return null; } },
  set: (v: string) => { try { localStorage.setItem(KEY, v); } catch { /* ignore */ } },
};

// Accepts "192.168.1.5:8000", " http://192.168.1.5:8000/ " and the like; returns a clean base URL without a trailing slash.
const norm = (v: string) => { const s = v.trim().replace(/\/+$/, ''); return /^https?:\/\//i.test(s) ? s : 'http://' + s; };

// API base: ?api=http://<server-ip>:8000, else the last one used, else localhost. A value given in the address is remembered,
// so reopening the page without ?api= keeps talking to the same server.
const fromUrl = params.get('api');
let base = norm(fromUrl || store.get() || 'http://localhost:8000');
if (fromUrl) store.set(base);
export const getBase = () => base;
export const setBase = (v: string) => { base = norm(v); store.set(base); };

export type LogFn = (line: string, ok: boolean) => void;
let log: LogFn = () => {};
export const onLog = (f: LogFn) => { log = f; };

async function http<T>(method: string, path: string, body?: unknown, timeoutMs = 10_000): Promise<T> {
  const t0 = performance.now();
  const stamp = (s: string | number) => `${new Date().toTimeString().slice(0, 8)} ${method} ${path} ${s} ${((performance.now() - t0) / 1000).toFixed(2)}s`;
  let r: Response;
  try {
    r = await fetch(base + path, { method, headers: body ? { 'Content-Type': 'application/json' } : {}, body: body ? JSON.stringify(body) : undefined, signal: AbortSignal.timeout(timeoutMs) });
  } catch (e) {
    const name = (e as Error).name, timedOut = name === 'TimeoutError' || name === 'AbortError';
    log(stamp(timedOut ? 'TIMEOUT' : 'ERR'), false);
    throw new Error(timedOut ? `no answer from ${base} after ${Math.round(timeoutMs / 1000)} s`
      : `cannot reach ${base} (server not running, wrong address, firewall, or blocked by CORS: open the UI at http://localhost:8080)`);
  }
  log(stamp(r.status), r.ok);
  if (!r.ok) {                                          // show the server's own explanation, e.g. {"detail":"unknown trip_id"}
    const d = await r.json().catch(() => null) as { detail?: unknown } | null;
    const why = d && d.detail !== undefined ? (typeof d.detail === 'string' ? d.detail : JSON.stringify(d.detail)) : (r.statusText || 'error');
    throw new Error(`${r.status}: ${why} (${path})`);
  }
  try { return await r.json() as T; } catch { throw new Error(`invalid JSON from ${base}${path}`); }
}

export const api = {
  health: (): Promise<Health> => http('GET', '/health'),
  async trips(): Promise<Trip[]> {
    const t = await http<Trip[]>('GET', '/trips');
    if (!Array.isArray(t)) throw new Error('invalid /trips response (expected a list)');
    return t;
  },
  predict: (id: string): Promise<PredictResult> => http('POST', '/predict', { trip_id: id }, 90_000),   // large trips take a few seconds
};
