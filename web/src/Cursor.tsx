import { useEffect, useRef } from 'react';

// Custom "+" cursor with thin constellation lines to floating nodes that parallax with the mouse.
const NODES = Array.from({ length: 14 }, (_, i) => ({ x: (i * 0.618) % 1, y: (i * 0.377 + 0.13) % 1, z: 0.2 + (i % 4) * 0.2 }));

export function Cursor() {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current!, ctx = cv.getContext('2d')!;
    let W = 0, H = 0, mx = innerWidth / 2, my = innerHeight / 2, raf = 0;
    const resize = () => { const d = devicePixelRatio || 1; W = innerWidth; H = innerHeight; cv.width = W * d; cv.height = H * d; ctx.setTransform(d, 0, 0, d, 0, 0); };
    const move = (e: PointerEvent) => { mx = e.clientX; my = e.clientY; };
    const draw = () => {
      ctx.clearRect(0, 0, W, H);
      const px = mx / W - 0.5, py = my / H - 0.5;
      const pts = NODES.map(n => ({ x: n.x * W - px * 60 * n.z, y: n.y * H - py * 60 * n.z }));
      ctx.lineWidth = 1; ctx.font = '10px "JetBrains Mono", monospace'; ctx.fillStyle = 'rgba(255,255,255,.35)';
      pts.forEach((p, i) => {
        const d = Math.hypot(p.x - mx, p.y - my);
        if (d < 260) { ctx.strokeStyle = `rgba(255,170,51,${0.28 * (1 - d / 260)})`; ctx.beginPath(); ctx.moveTo(mx, my); ctx.lineTo(p.x, p.y); ctx.stroke(); }
        const q = pts[(i + 1) % pts.length], dd = Math.hypot(p.x - q.x, p.y - q.y);
        if (dd < 320) { ctx.strokeStyle = `rgba(255,255,255,${0.07 * (1 - dd / 320)})`; ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke(); }
        ctx.fillText('+', p.x - 3, p.y + 4);
      });
      ctx.strokeStyle = '#fff'; ctx.beginPath(); ctx.moveTo(mx - 9, my); ctx.lineTo(mx + 9, my); ctx.moveTo(mx, my - 9); ctx.lineTo(mx, my + 9); ctx.stroke();
      raf = requestAnimationFrame(draw);
    };
    resize(); draw();
    addEventListener('resize', resize); addEventListener('pointermove', move);
    return () => { cancelAnimationFrame(raf); removeEventListener('resize', resize); removeEventListener('pointermove', move); };
  }, []);
  return <canvas ref={ref} className="cursor-layer" />;
}
