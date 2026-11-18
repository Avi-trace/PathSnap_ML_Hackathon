import type { Health, PredictResult, Trip } from './types';

// All data comes from the live PathSnap API. There is no bundled or cached result data in this client.
const params = new URLSearchParams(location.search);
const KEY = 'pathsnap_api';
const store = {
  get: () => { try { return localStorage.getItem(KEY); } catch { return null; } },
  set: (v: string) => { try { localStorage.setItem(KEY, v); } catch { /* ignore */ } },
};

// API base: ?api=http://<server-ip>:8000, else the last one used, else localhost.
let base = (params.get('api') || store.get() || 'http://localhost:8000').replace(/\/+$/, '');
export const getBase = () => base;
export const setBase = (v: string) => { base = v.trim().replace(/\/+$/, ''); store.set(base); };

export type LogFn = (line: string, ok: boolean) => void;
let log: LogFn = () => {};
export const onLog = (f: LogFn) => { log = f; };

async function http<T>(method: string, path: string, body?: unknown): Promise<T> {
  const t0 = performance.now();
  const stamp = (s: string | number) => `${new Date().toTimeString().slice(0, 8)} ${method} ${path} ${s} ${((performance.now() - t0) / 1000).toFixed(2)}s`;
  let r: Response;
  try {
    r = await fetch(base + path, { method, headers: body ? { 'Content-Type': 'application/json' } : {}, body: body ? JSON.stringify(body) : undefined });
  } catch { log(stamp('ERR'), false); throw new Error('cannot reach ' + base); }
  log(stamp(r.status), r.ok);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} on ${path}`);
  return r.json() as Promise<T>;
}

export const api = {
  health: (): Promise<Health> => http('GET', '/health'),
  trips: (): Promise<Trip[]> => http('GET', '/trips'),
  predict: (id: string): Promise<PredictResult> => http('POST', '/predict', { trip_id: id }),
};
