import type { Health, PredictResult, Trip } from './types';

declare global { interface Window { PATHSNAP_MOCK?: { model: string; trips: Trip[]; results: Record<string, PredictResult> } } }

const params = new URLSearchParams(location.search);
export const MOCK = params.get('mock') === '1';
const KEY = 'pathsnap_api';
const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));
const store = {
  get: () => { try { return localStorage.getItem(KEY); } catch { return null; } },
  set: (v: string) => { try { localStorage.setItem(KEY, v); } catch { /* ignore */ } },
};

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

// Dev only: mock-data.js is synthetic, never experiment output.
export function loadMock(): Promise<void> {
  return new Promise((res, rej) => {
    const s = document.createElement('script');
    s.src = '/mock-data.js'; s.onload = () => res(); s.onerror = () => rej(new Error('mock-data.js not found'));
    document.head.append(s);
  });
}

export const api = {
  async health(): Promise<Health> { if (MOCK) { await sleep(350); return { status: 'ok', model: window.PATHSNAP_MOCK!.model }; } return http('GET', '/health'); },
  async trips(): Promise<Trip[]> { if (MOCK) { await sleep(350); return window.PATHSNAP_MOCK!.trips; } return http('GET', '/trips'); },
  async predict(id: string): Promise<PredictResult> {
    if (MOCK) { await sleep(1400); const r = window.PATHSNAP_MOCK!.results[id]; if (!r) throw new Error('no mock data for ' + id); return r; }
    return http('POST', '/predict', { trip_id: id });
  },
};
