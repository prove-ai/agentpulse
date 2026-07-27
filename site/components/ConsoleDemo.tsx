"use client";

/* eslint-disable @next/next/no-img-element */
import { useEffect, useRef, useState } from "react";

/**
 * 01 // The console — playable tabbed demo.
 * Auto-plays through the dashboard views like a short video once scrolled
 * into view; clicking a tab switches to it and pauses the tour.
 */
const TABS = [
  { key: "overview", label: "Overview", src: "screenshots/light/overview.png" },
  { key: "runs", label: "Run History", src: "screenshots/light/run-history.png" },
  { key: "drift", label: "Drift Investigation", src: "screenshots/light/drift-investigation.png" },
  { key: "metrics", label: "Metrics Explorer", src: "screenshots/light/metrics-explorer.png" },
];

const HOLD_MS = 4200;

export default function ConsoleDemo() {
  const ref = useRef<HTMLDivElement>(null);
  const [active, setActive] = useState(0);
  const [playing, setPlaying] = useState(false);
  const started = useRef(false);

  // start the tour on first view (skip autoplay for reduced motion)
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !started.current) {
          started.current = true;
          setPlaying(true);
        }
      },
      { threshold: 0.35 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  // advance while playing
  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => setActive((a) => (a + 1) % TABS.length), HOLD_MS);
    return () => clearInterval(t);
  }, [playing]);

  const select = (i: number) => {
    setPlaying(false);
    setActive(i);
  };

  return (
    <div className="cdemo" ref={ref}>
      <div className="cdemo-tabs" role="tablist" aria-label="Console views">
        {TABS.map((t, i) => (
          <button
            key={t.key}
            type="button"
            role="tab"
            aria-selected={active === i}
            data-active={active === i}
            onClick={() => select(i)}
          >
            {t.label}
            {active === i && playing && (
              <span className="cdemo-progress" key={`${t.key}-${active}`} style={{ animationDuration: `${HOLD_MS}ms` }} />
            )}
          </button>
        ))}
      </div>
      <div className="cdemo-view">
        {TABS.map((t, i) => (
          <img
            key={t.key}
            src={t.src}
            alt={`AgentPulse console — ${t.label}`}
            data-active={active === i}
            loading={i === 0 ? "eager" : "lazy"}
          />
        ))}
      </div>
    </div>
  );
}
