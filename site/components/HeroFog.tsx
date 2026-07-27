"use client";

import { useEffect, useRef } from "react";

/**
 * Hero background — deep-blue fog.
 * Soft overlapping mist particles wander slowly across the hero. The cursor
 * stirs the fog: patches get dragged along the stroke and swivel around it,
 * then settle back. Reduced motion gets a static frame.
 */
type P = {
  hx: number; // home
  hy: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
  a: number;
  c: number; // sprite index
  ph: number; // wander phase
  flow: number; // px per frame of the slow automatic drift
};

const COLORS: [number, number, number][] = [
  [108, 142, 255],  // blue
  [152, 182, 255],  // light blue
  [158, 128, 255],  // violet
];

export default function HeroFog() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const hero = canvas.closest(".hero") as HTMLElement | null;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const LIGHT_COLORS: [number, number, number][] = [
      [47, 86, 224],
      [90, 122, 240],
      [124, 58, 237],
    ];
    const makeSprites = (colors: [number, number, number][]) =>
      colors.map(([r, g, b]) => {
      const s = document.createElement("canvas");
      s.width = s.height = 256;
      const sc = s.getContext("2d")!;
      const grad = sc.createRadialGradient(128, 128, 0, 128, 128, 128);
      grad.addColorStop(0, `rgba(${r},${g},${b},0.95)`);
      grad.addColorStop(0.45, `rgba(${r},${g},${b},0.36)`);
      grad.addColorStop(1, `rgba(${r},${g},${b},0)`);
      sc.fillStyle = grad;
      sc.fillRect(0, 0, 256, 256);
      return s;
      });
    const sprites = makeSprites(COLORS);
    const lightSprites = makeSprites(LIGHT_COLORS);

    let raf = 0;
    let W = 0;
    let H = 0;
    let t = 0;
    let ps: P[] = [];
    const mouse = { x: -1e4, y: -1e4, vx: 0, vy: 0 };

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = Math.round(rect.width);
      H = Math.round(rect.height);
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const n = Math.max(14, Math.min(30, Math.round((W * H) / 52000)));
      ps = Array.from({ length: n }, () => {
        const x = Math.random() * W;
        const y = Math.random() * H;
        return {
          hx: x,
          hy: y,
          x,
          y,
          vx: 0,
          vy: 0,
          r: 170 + Math.random() * 260,
          a: 0.13 + Math.random() * 0.1,
          c: Math.floor(Math.random() * sprites.length),
          ph: Math.random() * Math.PI * 2,
          flow: 0.5 + Math.random() * 0.55,
        };
      });
    };

    const frame = () => {
      t += 0.004;
      mouse.vx *= 0.88;
      mouse.vy *= 0.88;
      ctx.clearRect(0, 0, W, H);
      const isLight = document.documentElement.dataset.theme === "light";
      ctx.globalCompositeOperation = isLight ? "source-over" : "lighter";
      for (const p of ps) {
        // constant, very slow drift across the hero (right to left)
        p.hx -= p.flow;
        if (p.hx < -p.r * 1.4) {
          const shift = W + p.r * 2.8;
          p.hx += shift;
          p.x += shift; // move the patch with its home while fully offscreen
        }
        // slow roaming of the home point
        const hx = p.hx + Math.sin(t * 0.9 + p.ph) * 60;
        const hy = p.hy + Math.cos(t * 0.7 + p.ph * 1.7) * 40;
        // spring toward home
        p.vx += (hx - p.x) * 0.0006;
        p.vy += (hy - p.y) * 0.0006;
        // cursor stirs the fog: dragged along the stroke, swivelling around it
        const dx = p.x - mouse.x;
        const dy = p.y - mouse.y;
        const d = Math.hypot(dx, dy);
        if (d < 360 && d > 0.01) {
          const fall = 1 - d / 360;
          // drag along the cursor's motion
          p.vx += mouse.vx * 0.09 * fall;
          p.vy += mouse.vy * 0.09 * fall;
          // curl around the stroke, direction follows the gesture
          const curl = Math.sign(mouse.vx * dy - mouse.vy * dx) || 1;
          const sp = Math.min(6, Math.hypot(mouse.vx, mouse.vy));
          const sw = 0.05 * sp * fall * curl;
          p.vx += (-dy / d) * sw;
          p.vy += (dx / d) * sw;
        }
        p.vx *= 0.955;
        p.vy *= 0.955;
        p.x += p.vx;
        p.y += p.vy;

        // fog thins right around the cursor (dispersal)
        const thin = d < 240 ? 1 - 0.3 * (1 - d / 240) : 1;
        ctx.globalAlpha = p.a * thin * (isLight ? 0.35 : 1);
        ctx.drawImage((isLight ? lightSprites : sprites)[p.c], p.x - p.r, p.y - p.r, p.r * 2, p.r * 2);
      }
      ctx.globalAlpha = 1;
      ctx.globalCompositeOperation = "source-over";
    };

    const tick = () => {
      frame();
      raf = requestAnimationFrame(tick);
    };

    const onMove = (e: MouseEvent) => {
      const r = canvas.getBoundingClientRect();
      const nx = e.clientX - r.left;
      const ny = e.clientY - r.top;
      if (mouse.x > -1e3) {
        mouse.vx = mouse.vx * 0.5 + (nx - mouse.x) * 0.5;
        mouse.vy = mouse.vy * 0.5 + (ny - mouse.y) * 0.5;
      }
      mouse.x = nx;
      mouse.y = ny;
    };
    const onLeave = () => {
      mouse.x = -1e4;
      mouse.y = -1e4;
    };

    resize();
    window.addEventListener("resize", resize);

    if (reduced) {
      frame(); // static fog, no motion or interaction
    } else {
      raf = requestAnimationFrame(tick);
      hero?.addEventListener("mousemove", onMove);
      hero?.addEventListener("mouseleave", onLeave);
    }

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      hero?.removeEventListener("mousemove", onMove);
      hero?.removeEventListener("mouseleave", onLeave);
    };
  }, []);

  return (
    <div className="hero-fog" aria-hidden>
      <canvas ref={canvasRef} />
    </div>
  );
}
