import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, getBase, onLog, setBase } from './api';
import { Cursor } from './Cursor';
import { ErrorBoundary } from './ErrorBoundary';
import { Intro } from './Intro';
import { FULL } from './mapDraw';
import { MapCanvas } from './MapCanvas';
import { Scramble } from './Scramble';
import type { LayerKey, PredictResult, Trip } from './types';

const SKIP_KEY = 'pathsnap_skip_intro';
const skipStored = () => { try { return sessionStorage.getItem(SKIP_KEY) === '1'; } catch { return false; } };
const storeSkip = (v: boolean) => { try { v ? sessionStorage.setItem(SKIP_KEY, '1') : sessionStorage.removeItem(SKIP_KEY); } catch { /* ignore */ } };
const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));
const pct = (v?: number) => (v == null || !Number.isFinite(v) ? '—' : (v * 100).toFixed(1) + '%');   // '—' when the server did not score this trip
const dur = (s: number) => `${Math.floor(s / 60)}M ${String(Math.round(s % 60)).padStart(2, '0')}S`;
const delta = (a?: number, b?: number) => { if (a == null || b == null) return ''; const x = (b - a) * 100; return (x >= 0 ? '+' : '') + x.toFixed(1) + ' PTS WITH HMM'; };

const LAYERS: { k: LayerKey; label: string; sw: string }[] = [
  { k: 'net', label: 'Road network', sw: '#2f3a3d' },
  { k: 'gps', label: 'GPS observations', sw: '#fff' },
  { k: 'clf', label: 'Classifier only', sw: '#c4c7c7' },
  { k: 'hmm', label: 'Classifier + HMM', sw: '#ffaa33' },
  { k: 'brk', label: 'Disconnected jumps', sw: '#ff6a55' },
];

/* ---------------- boot screen (kept from the old UI) ---------------- */
function Boot({ onReady, onEnter, ready }: { onReady: (t: Trip[], model: string, mock: boolean) => void; onEnter: () => void; ready: boolean }) {
  const [pctV, setPctV] = useState(0), [msg, setMsg] = useState('Connecting to API'), [err, setErr] = useState(''), [url, setUrl] = useState(getBase());
  const [tries, setTries] = useState(0);
  useEffect(() => {
    let dead = false;
    const to = async (n: number, m: string) => { setMsg(m); setPctV(n); await sleep(300); };
    (async () => {
      setErr('');
      try {
        await to(15, 'Checking API');
        const h = await api.health();
        await to(55, 'Loading trips');
        const t = await api.trips();
        await to(100, 'Ready');
        if (!dead) onReady(t, h.model ?? '', h.mock === true);
      } catch (e) { if (!dead) { setMsg('Connection failed'); setErr((e as Error).message); } }
    })();
    return () => { dead = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tries]);
  return (
    <div className="boot">
      <div className="boot-in">
        <h1 className="wordmark">PathSnap</h1>
        <div className="pct">{String(pctV).padStart(3, '0')}%</div>
        <div className="label"><Scramble text={msg.toUpperCase()} /></div>
        {err && (
          <div className="boot-err">
            <p>{err}</p>
            <input value={url} onChange={e => setUrl(e.target.value)} spellCheck={false} aria-label="API base URL" />
            <button className="cta" onClick={() => { setBase(url); setTries(n => n + 1); }}>Retry</button>
            <span className="hint">Check the server address and that the PathSnap API is running on that laptop.</span>
          </div>
        )}
        {ready && <button className="cta" onClick={onEnter} autoFocus>Enter</button>}
      </div>
    </div>
  );
}

/* ---------------- app ---------------- */
export function App() {
  const [phase, setPhase] = useState<'boot' | 'intro' | 'app'>('boot'), [ready, setReady] = useState(false);
  const entered = phase === 'app';
  const [trips, setTrips] = useState<Trip[]>([]), [model, setModel] = useState(''), [mock, setMock] = useState(false);
  const [sel, setSel] = useState<string | null>(null), [busy, setBusy] = useState(false);
  const [res, setRes] = useState<PredictResult | null>(null), [error, setError] = useState('');
  const [layers, setLayers] = useState<Record<LayerKey, boolean>>({ net: true, gps: true, clf: true, hmm: true, brk: true });
  const [scale, setScale] = useState({ px: 90, label: '' });
  const [lines, setLines] = useState<{ t: string; ok: boolean }[]>([]);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => { onLog((t, ok) => setLines(l => [...l.slice(-39), { t, ok }])); }, []);
  useEffect(() => { logRef.current?.scrollTo(0, 1e9); }, [lines]);

  const run = useCallback(async () => {
    if (!sel || busy) return;
    setBusy(true); setError('');
    try { setRes(await api.predict(sel)); } catch (e) { setError((e as Error).message); }
    setBusy(false);
  }, [sel, busy]);

  useEffect(() => {  // arrow keys pick a trip, Enter runs
    if (!entered) return;
    const h = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).matches('input')) return;
      const ids = trips.map(t => t.trip_id), i = sel ? ids.indexOf(sel) : -1;
      if (e.key === 'ArrowDown') { e.preventDefault(); setSel(ids[Math.min(ids.length - 1, i + 1)]); }
      if (e.key === 'ArrowUp') { e.preventDefault(); setSel(ids[Math.max(0, i - 1)]); }
      if (e.key === 'Enter' && sel && !(e.target as HTMLElement).matches('button')) run();
    };
    addEventListener('keydown', h); return () => removeEventListener('keydown', h);
  }, [entered, trips, sel, run]);

  const opts = useMemo(() => ({ ...FULL, net: +layers.net, gps: +layers.gps, clf: +layers.clf, hmm: +layers.hmm, brk: +layers.brk }), [layers]);
  const m = res?.metrics;
  const tot = m ? m.latency_ms.candidates + m.latency_ms.classifier + m.latency_ms.hmm : 0;

  return (
    <>
      <Cursor />
      {phase === 'boot' && <Boot ready={ready} onReady={(t, mo, mk) => { setTrips(t); setModel(mo); setMock(mk); setReady(true); }} onEnter={() => setPhase(location.hash === '#app' || skipStored() ? 'app' : 'intro')} />}
      {phase === 'intro' && <Intro trips={trips} onDone={() => { storeSkip(true); setPhase('app'); }} />}
      <div className={'app' + (entered ? ' on' : '')}>
        {mock && <div className="mockbar">MOCK SERVER · RECORDED SAMPLE DATA · NOT THE REAL MODEL · DO NOT QUOTE THESE NUMBERS</div>}
        <header>
          <span className="label tele">000_0_01 · {getBase().replace(/^https?:\/\//, '')} ·<button className="linkbtn" onClick={() => { storeSkip(false); setPhase('intro'); }}>Replay intro</button></span>
          <h2 className="brand">PATHSNAP</h2>
          <span className="label tele">{model || '—'}</span>
        </header>
        <div className="cols">
          <aside className="left">
            <div className="ph"><span className="label"><Scramble text="EVALUATION TRIPS" /></span><span className="label">{String(trips.length).padStart(2, '0')}</span></div>
            <div className="trips">
              {trips.map((t, i) => (
                <button key={t.trip_id} className="trip" aria-current={sel === t.trip_id} onClick={() => setSel(t.trip_id)}>
                  <span className="ix">{String(i + 1).padStart(2, '0')}</span>
                  <span className="id">{t.trip_id}</span>
                  <span className="meta">{t.n_gps} PTS · {dur(t.duration_s)}{t.length_km ? ` · ${t.length_km} KM` : ''}</span>
                </button>
              ))}
            </div>
            <div className="run">
              <button className="cta" disabled={!sel || busy} onClick={run}>Reconstruct route</button>
              <span className="label">{busy ? 'Running…' : sel ? `Trip ${sel} selected` : 'Select a trip'}</span>
            </div>
          </aside>

          <main className="stage">
            <i className="br tl" /><i className="br tr" /><i className="br bl" /><i className="br brr" />
            <ErrorBoundary resetKey={res}><MapCanvas result={res} opts={opts} onScale={(px, label) => setScale({ px, label })} /></ErrorBoundary>
            {!res && !busy && <div className="empty"><div><h2><Scramble text={error ? 'REQUEST FAILED' : 'SELECT A TRIP'} /></h2><span className="label">{error || 'Then reconstruct the route'}</span></div></div>}
            {busy && <div className="empty dim"><div><h2>RECONSTRUCTING</h2><span className="label">Candidates · classifier · Viterbi</span></div></div>}
            {res && (
              <>
                <div className="layers">
                  {LAYERS.map(l => (
                    <label key={l.k} style={{ ['--sw' as string]: l.sw }}>
                      <input type="checkbox" checked={layers[l.k]} onChange={e => setLayers(s => ({ ...s, [l.k]: e.target.checked }))} />{l.label}
                    </label>
                  ))}
                </div>
                <div className="drag">Drag to pan · scroll to zoom · double-click to fit</div>
                <div className="scale"><div style={{ width: scale.px }} />{scale.label}</div>
              </>
            )}
          </main>

          <aside className="right">
            <div className="ph"><span className="label">Results</span><span className="label">{res?.trip_id ?? '—'}</span></div>
            <div className="res">
              {m ? (
                <>
                  <section className="m"><span className="label">Pointwise accuracy</span>
                    <div className="pair"><div><span className="k">Classifier only</span><b className="v mute">{pct(m.pointwise_acc_classifier)}</b></div><div><span className="k">+ HMM / Viterbi</span><b className="v">{pct(m.pointwise_acc_hmm)}</b></div></div>
                    <div className="delta">{delta(m.pointwise_acc_classifier, m.pointwise_acc_hmm)}</div></section>
                  <section className="m"><span className="label">Route continuity</span>
                    <div className="pair"><div><span className="k">Classifier only</span><b className="v mute">{pct(m.route_continuity_classifier)}</b></div><div><span className="k">+ HMM / Viterbi</span><b className="v">{pct(m.route_continuity_hmm)}</b></div></div>
                    <div className="delta">{delta(m.route_continuity_classifier, m.route_continuity_hmm)}</div></section>
                  <section className="m rows"><span className="label">Run details</span>
                    <div><span>GPS observations</span><b>{m.n_obs.toLocaleString()}</b></div>
                    <div><span>Candidate miss rate</span><b>{pct(m.candidate_miss_rate)}</b></div>
                    <div><span>Disconnected jumps (clf)</span><b>{res!.classifier_only.breaks.length}</b></div>
                    <div><span>Model</span><b>{model || '—'}</b></div></section>
                  <section className="m"><span className="label">Latency</span>
                    <div className="rows"><div><span>Total (server)</span><b>{tot} ms</b></div></div>
                    <div className="bar">{([[m.latency_ms.candidates, '#707a7d'], [m.latency_ms.classifier, '#c4c7c7'], [m.latency_ms.hmm, '#ffaa33']] as const).map(([n, c], i) => <i key={i} style={{ width: `${(n / (tot || 1)) * 100}%`, background: c }} />)}</div>
                    <div className="rows"><div><span>Candidates</span><b>{m.latency_ms.candidates} ms</b></div><div><span>Classifier</span><b>{m.latency_ms.classifier} ms</b></div><div><span>HMM decode</span><b>{m.latency_ms.hmm} ms</b></div></div></section>
                  <p className="note">Ground truth stays on the server. This view only receives predictions and aggregate scores.</p>
                </>
              ) : <p className="note" style={{ paddingTop: 20 }}>No result yet.</p>}
            </div>
            <div className="log" ref={logRef}>{lines.map((l, i) => <div key={i} className={l.ok ? 'ok' : 'err'}>{l.t}</div>)}</div>
          </aside>
        </div>
      </div>
    </>
  );
}
