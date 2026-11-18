import { useEffect, useMemo, useRef, useState } from 'react';
import gsap from 'gsap';
import Lenis from 'lenis';
import { api } from './api';
import { ErrorBoundary } from './ErrorBoundary';
import { MapCanvas, type MapHandle } from './MapCanvas';
import { FULL, type DrawOpts } from './mapDraw';
import { Scramble } from './Scramble';
import type { PredictResult, Trip } from './types';

// Scroll-driven intro (about 600vh). One live /predict response is fetched at start and the map layers are
// scrubbed with scroll: raw GPS -> road network -> classifier only -> HMM route -> numbers from `metrics`.
const clamp = (x: number) => Math.min(1, Math.max(0, x));
const pct = (v: number) => (v * 100).toFixed(1) + '%';
const reducedMotion = () => { try { return matchMedia('(prefers-reduced-motion: reduce)').matches; } catch { return false; } };

const STAGES = [
  { n: '01', title: 'RAW GPS', body: 'A trip arrives as a list of GPS readings. Each dot is one reading. Receiver noise and sparse sampling mean the dots do not sit exactly on any road.' },
  { n: '02', title: 'ROAD NETWORK', body: 'PathSnap loads the road segments around the trace. Every reading has to be matched to one of these segments.' },
  { n: '03', title: 'CLASSIFIER ONLY', body: 'A classifier picks a segment for each reading on its own. Each pick is made independently, so neighbouring picks can fail to connect. Red rings mark those jumps.' },
  { n: '04', title: 'HMM + VITERBI', body: 'A hidden Markov model links the readings in time and the Viterbi algorithm finds the single most consistent route through the network. The route stays connected.' },
  { n: '05', title: 'RESULT', body: 'Scores for this trip, computed on the server against the reference route. The figures below come straight from the API response.' },
];
// Scroll progress (0..1) at which each stage's text is centred; every stage owns a fifth of the scroll.
const WIN = 0.2;

function pickTrip(trips: Trip[]): string | null {
  const want = new URLSearchParams(location.search).get('introTrip');
  if (want) return want;
  if (!trips.length) return null;
  return trips.reduce((a, b) => (b.n_gps < a.n_gps ? b : a)).trip_id;
}

export function Intro({ trips, onDone }: { trips: Trip[]; onDone: () => void }) {
  const [res, setRes] = useState<PredictResult | null>(null), [err, setErr] = useState('');
  const reduced = useMemo(reducedMotion, []);
  const scroller = useRef<HTMLDivElement>(null), track = useRef<HTMLDivElement>(null);
  const blocks = useRef<(HTMLDivElement | null)[]>([]), bar = useRef<HTMLDivElement>(null), hint = useRef<HTMLDivElement>(null);
  const opts = useRef<DrawOpts>({ ...FULL, net: 0, clf: 0, brk: 0, hmm: 0, gpsFrac: 0, clfFrac: 0, hmmFrac: 0 });
  const handle = useRef<MapHandle | null>(null);

  useEffect(() => {  // one live fetch at intro start
    let dead = false;
    const id = pickTrip(trips);
    if (!id) { setErr('No trips returned by the API.'); return; }
    api.predict(id).then(r => { if (!dead) setRes(r); }).catch(e => { if (!dead) setErr((e as Error).message); });
    return () => { dead = true; };
  }, [trips]);

  useEffect(() => {  // scroll scrubbing
    if (reduced) { Object.assign(opts.current, FULL); handle.current?.redraw(); return; }
    const sc = scroller.current!, tr = track.current!;
    const lenis = new Lenis({ wrapper: sc, content: tr, smoothWheel: true });
    const target = { p: 0 }, cur = { p: -1 }, shown = { p: 0 };
    lenis.on('scroll', (l: { scroll: number; limit: number }) => { target.p = l.limit > 0 ? clamp(l.scroll / l.limit) : 0; });
    const apply = (p: number) => {
      const o = opts.current;
      o.gps = 1; o.gpsFrac = clamp((p + 0.02) / 0.2);
      o.net = clamp((p - 0.2) / 0.12);
      o.clf = 1 - 0.65 * clamp((p - 0.62) / 0.18); o.clfFrac = clamp((p - 0.4) / 0.16); o.brk = clamp((p - 0.52) / 0.06) * (1 - 0.7 * clamp((p - 0.7) / 0.1));
      o.hmm = 1; o.hmmFrac = clamp((p - 0.62) / 0.18);
      handle.current?.redraw();
      blocks.current.forEach((el, i) => {
        if (!el) return;
        const a = i * WIN, b = a + WIN, f = 0.035;
        const vis = Math.min(i === 0 ? 1 : clamp((p - a) / f), i === 4 ? 1 : clamp((b - p) / f));
        el.style.opacity = String(vis); el.style.transform = `translateY(calc(-50% + ${(1 - vis) * 18}px))`; el.style.pointerEvents = vis > 0.6 ? 'auto' : 'none';
      });
      if (bar.current) bar.current.style.transform = `scaleX(${p})`;
      if (hint.current) hint.current.style.opacity = String(1 - clamp(p / 0.03));
    };
    const tick = (t: number) => {
      lenis.raf(t * 1000);
      shown.p += (target.p - shown.p) * 0.1;               // lerp = the "scrub" lag
      if (Math.abs(shown.p - cur.p) > 0.0004) { cur.p = shown.p; apply(shown.p); }
    };
    gsap.ticker.add(tick); apply(0);
    return () => { gsap.ticker.remove(tick); lenis.destroy(); };
  }, [reduced]);

  useEffect(() => { handle.current?.redraw(); }, [res]);

  const m = res?.metrics;
  const mapBox = (
    <div className="intro-map">
      <i className="br tl" /><i className="br tr" /><i className="br bl" /><i className="br brr" />
      <ErrorBoundary resetKey={res}><MapCanvas result={res} opts={opts.current} interactive={false} handle={handle} /></ErrorBoundary>
      {err && <div className="empty"><div><h2>LIVE DATA UNAVAILABLE</h2><span className="label" style={{ color: 'var(--warn)' }}>{err}</span></div></div>}
    </div>
  );
  const metrics = m ? (
    <div className="intro-metrics">
      <div><span className="k">Trip {res!.trip_id} · {m.n_obs.toLocaleString()} readings</span></div>
      <div className="pair">
        <div><span className="k">Accuracy · classifier</span><b className="v mute">{pct(m.pointwise_acc_classifier)}</b></div>
        <div><span className="k">Accuracy · HMM</span><b className="v">{pct(m.pointwise_acc_hmm)}</b></div>
        <div><span className="k">Continuity · classifier</span><b className="v mute">{pct(m.route_continuity_classifier)}</b></div>
        <div><span className="k">Continuity · HMM</span><b className="v">{pct(m.route_continuity_hmm)}</b></div>
      </div>
    </div>
  ) : <div className="intro-metrics"><span className="label" style={{ color: 'var(--warn)' }}>{err ? 'No live data, so no numbers are shown.' : 'Waiting for the API…'}</span></div>;
  const text = (i: number) => (
    <>
      <span className="label">{STAGES[i].n} / 05</span>
      <h2><Scramble text={STAGES[i].title} /></h2>
      <p>{STAGES[i].body}</p>
      {i === 4 && metrics}
      {i === 4 && <button className="cta" onClick={onDone}>Open trip picker</button>}
    </>
  );

  return (
    <div className={'intro' + (reduced ? ' reduced' : '')} ref={scroller}>
      <div className="intro-top">
        <span className="label">PathSnap · intro</span>
        <button className="skip" onClick={onDone}>Skip intro →</button>
      </div>
      {reduced ? (
        <div className="intro-flat">
          {mapBox}
          {STAGES.map((_, i) => <section key={i} className="stage-block">{text(i)}</section>)}
        </div>
      ) : (
        <div className="intro-track" ref={track}>
          <div className="intro-stage">
            <div className="intro-text">
              {STAGES.map((_, i) => <div key={i} className="stage-block" ref={el => { blocks.current[i] = el; }} style={{ opacity: i === 0 ? 1 : 0 }}>{text(i)}</div>)}
            </div>
            {mapBox}
            <div className="intro-hint label" ref={hint}>Scroll ↓</div>
            <div className="intro-prog"><div ref={bar} /></div>
          </div>
        </div>
      )}
    </div>
  );
}
