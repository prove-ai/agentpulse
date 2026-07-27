"use client";

import { useEffect, useRef } from "react";

/**
 * Hero visual — a generalized multi-agent graph with drift.
 * Pulses flow along the edges left-to-right. Every few seconds one agent
 * drifts: it turns red, its downstream pulses go red, downstream agents
 * tinge amber, then everything recovers. Hovering highlights a node;
 * clicking one triggers a drift there. No labels, no product UI —
 * just the idea of drift propagating through a system.
 */
const NODES = [
  { x: 0.10, y: 0.30 },
  { x: 0.13, y: 0.72 },
  { x: 0.40, y: 0.50 },
  { x: 0.66, y: 0.26 },
  { x: 0.68, y: 0.74 },
  { x: 0.92, y: 0.50 },
];
const EDGES: [number, number][] = [
  [0, 2],
  [1, 2],
  [2, 3],
  [2, 4],
  [3, 5],
  [4, 5],
];
// nodes affected downstream of each node (for drift propagation)
const DOWNSTREAM: Record<number, number[]> = {
  0: [2, 3, 4, 5],
  1: [2, 3, 4, 5],
  2: [3, 4, 5],
  3: [5],
  4: [5],
  5: [],
};

type Pulse = { edge: number; t: number; speed: number; red: boolean };

const BLUE = "79,123,255";
const RED = "255,93,99";
const AMBER = "245,181,68";

export default function DriftGraph() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const hero = canvas.closest(".hero") as HTMLElement | null;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let raf = 0;
    let W = 0;
    let H = 0;
    let t = 0;
    const pulses: Pulse[] = [];
    let driftNode = -1;
    let driftPhase = 0; // frames since drift began
    let nextDriftAt = 320; // frames until the next automatic drift
    const mouse = { x: -1e4, y: -1e4 };

    const px = (n: { x: number; y: number }, i: number) => ({
      x: n.x * W,
      y: n.y * H + Math.sin(t * 0.011 + i * 2.1) * H * 0.018,
    });

    const resize = () => {
      const r = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = Math.round(r.width);
      H = Math.round(r.height);
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const edgeCurve = (a: { x: number; y: number }, b: { x: number; y: number }) => {
      const mx = (a.x + b.x) / 2;
      const my = (a.y + b.y) / 2 + (b.x - a.x) * 0.08; // slight sag
      return { mx, my };
    };

    const pointOnEdge = (edge: number, tt: number) => {
      const a = px(NODES[EDGES[edge][0]], EDGES[edge][0]);
      const b = px(NODES[EDGES[edge][1]], EDGES[edge][1]);
      const { mx, my } = edgeCurve(a, b);
      const u = 1 - tt;
      return {
        x: u * u * a.x + 2 * u * tt * mx + tt * tt * b.x,
        y: u * u * a.y + 2 * u * tt * my + tt * tt * b.y,
      };
    };

    const startDrift = (node: number) => {
      if (DOWNSTREAM[node].length === 0) return; // sink can't originate drift
      driftNode = node;
      driftPhase = 0;
    };

    const isDriftEdge = (edge: number) =>
      driftNode >= 0 &&
      (EDGES[edge][0] === driftNode || DOWNSTREAM[driftNode].includes(EDGES[edge][0]));

    const nearestNode = (x: number, y: number) => {
      let best = -1;
      let bd = 60;
      NODES.forEach((n, i) => {
        const p = px(n, i);
        const d = Math.hypot(p.x - x, p.y - y);
        if (d < bd) {
          bd = d;
          best = i;
        }
      });
      return best;
    };

    const frame = () => {
      t++;
      ctx.clearRect(0, 0, W, H);
      const isLight = document.documentElement.dataset.theme === "light";
      const BLUE = isLight ? "47,86,224" : "79,123,255";
      const RED = isLight ? "205,42,50" : "255,93,99";
      const AMBER = isLight ? "170,106,10" : "245,181,68";
      const BODY = isLight ? "#ffffff" : "#0b0f1c";
      const edgeA = isLight ? 0.3 : 0.22;

      // drift lifecycle: ~200 frames red, then recover
      if (driftNode >= 0) {
        driftPhase++;
        if (driftPhase > 230) {
          driftNode = -1;
          nextDriftAt = t + 260 + Math.random() * 240;
        }
      } else if (t > nextDriftAt) {
        startDrift([0, 1, 2, 3, 4][Math.floor(Math.random() * 5)]);
      }

      // edges
      for (let e = 0; e < EDGES.length; e++) {
        const a = px(NODES[EDGES[e][0]], EDGES[e][0]);
        const b = px(NODES[EDGES[e][1]], EDGES[e][1]);
        const { mx, my } = edgeCurve(a, b);
        const hot = isDriftEdge(e);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.quadraticCurveTo(mx, my, b.x, b.y);
        ctx.strokeStyle = hot ? `rgba(${RED},0.4)` : `rgba(${BLUE},${edgeA})`;
        ctx.lineWidth = 1.2;
        ctx.stroke();
      }

      // spawn pulses
      if (t % 34 === 0) {
        const e = Math.floor(Math.random() * EDGES.length);
        pulses.push({ edge: e, t: 0, speed: 0.006 + Math.random() * 0.004, red: false });
      }

      // pulses
      for (let i = pulses.length - 1; i >= 0; i--) {
        const p = pulses[i];
        p.t += p.speed;
        if (p.t >= 1) {
          pulses.splice(i, 1);
          continue;
        }
        p.red = isDriftEdge(p.edge);
        const pos = pointOnEdge(p.edge, p.t);
        const c = p.red ? RED : BLUE;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 2.6, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${c},0.9)`;
        ctx.shadowColor = `rgba(${c},0.8)`;
        ctx.shadowBlur = 8;
        ctx.fill();
        ctx.shadowBlur = 0;
      }

      // nodes
      NODES.forEach((n, i) => {
        const p = px(n, i);
        const isSource = driftNode === i;
        const isDown = driftNode >= 0 && DOWNSTREAM[driftNode].includes(i);
        const hovered = nearestNode(mouse.x, mouse.y) === i;
        const r = 7 + (hovered ? 2.5 : 0) + Math.sin(t * 0.02 + i) * 0.6;

        let ring = `rgba(${BLUE},0.85)`;
        let glow = `rgba(${BLUE},0.55)`;
        if (isSource) {
          ring = `rgba(${RED},0.95)`;
          glow = `rgba(${RED},0.8)`;
        } else if (isDown) {
          ring = `rgba(${AMBER},0.9)`;
          glow = `rgba(${AMBER},0.5)`;
        }

        // halo
        ctx.beginPath();
        ctx.arc(p.x, p.y, r + 6, 0, Math.PI * 2);
        ctx.fillStyle = isSource ? `rgba(${RED},0.12)` : isDown ? `rgba(${AMBER},0.08)` : `rgba(${BLUE},0.07)`;
        ctx.fill();
        // body
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.fillStyle = BODY;
        ctx.strokeStyle = ring;
        ctx.lineWidth = hovered ? 2.2 : 1.6;
        ctx.shadowColor = glow;
        ctx.shadowBlur = isSource ? 22 : hovered ? 14 : 9;
        ctx.fill();
        ctx.stroke();
        ctx.shadowBlur = 0;
        // core dot
        ctx.beginPath();
        ctx.arc(p.x, p.y, 2.2, 0, Math.PI * 2);
        ctx.fillStyle = ring;
        ctx.fill();
      });
    };

    const tick = () => {
      frame();
      raf = requestAnimationFrame(tick);
    };

    const toLocal = (e: MouseEvent) => {
      const r = canvas.getBoundingClientRect();
      return { x: e.clientX - r.left, y: e.clientY - r.top };
    };
    const onMove = (e: MouseEvent) => {
      const p = toLocal(e);
      mouse.x = p.x;
      mouse.y = p.y;
    };
    const onLeave = () => {
      mouse.x = -1e4;
      mouse.y = -1e4;
    };
    const onClick = (e: MouseEvent) => {
      const p = toLocal(e);
      const n = nearestNode(p.x, p.y);
      if (n >= 0) startDrift(n);
    };

    resize();
    window.addEventListener("resize", resize);

    if (reduced) {
      frame(); // static frame
    } else {
      raf = requestAnimationFrame(tick);
      hero?.addEventListener("mousemove", onMove);
      hero?.addEventListener("mouseleave", onLeave);
      hero?.addEventListener("click", onClick);
    }

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      hero?.removeEventListener("mousemove", onMove);
      hero?.removeEventListener("mouseleave", onLeave);
      hero?.removeEventListener("click", onClick);
    };
  }, []);

  return (
    <div className="hero-visual" aria-hidden>
      <canvas ref={canvasRef} />
    </div>
  );
}
