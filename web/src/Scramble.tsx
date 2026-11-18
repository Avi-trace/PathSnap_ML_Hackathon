import { useEffect, useState } from 'react';

const GLYPHS = '_^{}=*+<>/\\|#%';

// Character-scramble reveal: random glyphs resolve left to right into `text`.
export function Scramble({ text, ms = 700 }: { text: string; ms?: number }) {
  const [out, setOut] = useState(text);
  useEffect(() => {
    let raf = 0; const t0 = performance.now();
    const step = (t: number) => {
      const k = Math.min(1, (t - t0) / ms), done = Math.floor(k * text.length);
      setOut(text.split('').map((c, i) => (i < done || c === ' ' ? c : GLYPHS[Math.floor(Math.random() * GLYPHS.length)])).join(''));
      if (k < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [text, ms]);
  return <>{out}</>;
}
