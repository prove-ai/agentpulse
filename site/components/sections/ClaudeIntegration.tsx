"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/**
 * 04 // Claude integration — a scripted conversation that plays like a short
 * video: auto-plays when scrolled into view, replayable. Tool calls show a
 * typing-dots loader while Claude "works", then collapse when the answer
 * lands. Single centered column: title → lead → bullet row → chat.
 */
type Line =
  | { kind: "user"; text: string; at: number }
  | { kind: "tool"; tool: string; at: number; until: number }
  | { kind: "ai"; text: React.ReactNode; at: number };

const SCRIPT: Line[] = [
  { kind: "user", at: 400, text: "morning — anything break overnight?" },
  { kind: "tool", at: 1200, until: 4400, tool: "get_todays_finding" },
  {
    kind: "ai",
    at: 4400,
    text: (
      <>
        One finding, flagged around run&nbsp;110 in <code>demo</code>:
        <br />
        <code>critic</code> success rate went 92% → 22% and is still low as of
        the latest run. Retries are up 3×, tokens/turn +45% — it&rsquo;s
        retrying and rambling, which usually means it can&rsquo;t parse what
        it&rsquo;s given.
        <br />
        critic&rsquo;s own config is unchanged — same prompt hash and model
        since v1. What did change is the payload it receives from{" "}
        <code>writer</code>, starting at the same run. So critic is probably
        downstream damage, not the source.
      </>
    ),
  },
  { kind: "user", at: 5400, text: "what changed in writer?" },
  { kind: "tool", at: 6200, until: 9400, tool: "get_version_comparison" },
  {
    kind: "ai",
    at: 9400,
    text: (
      <>
        Two things, both at run&nbsp;110: the writer prompt hash changed
        (<code>270339f</code> → <code>a78f0f7</code>) and the model went from{" "}
        <code>gpt-4o-mini</code> to <code>gpt-4-turbo</code>.
        <br />
        After the change, writer responses run ~180% longer and drop the
        section structure critic expects — earlier payloads were structured
        summaries, newer ones read as prose. writer&rsquo;s input from analyst
        is unchanged (same structure, same length distribution), so this
        isn&rsquo;t upstream noise.
      </>
    ),
  },
  { kind: "user", at: 10400, text: "ok, what do I look at first?" },
  { kind: "tool", at: 11200, until: 14400, tool: "get_next_check_steps" },
  {
    kind: "ai",
    at: 14400,
    text: (
      <>
        I&rsquo;d go in this order:
        <ol>
          <li>
            Diff the two writer prompts — check whether the formatting
            constraints got dropped in the rewrite.
          </li>
          <li>
            Pull two or three writer→critic handoffs from each side of
            run&nbsp;110 and compare payload structure.
          </li>
          <li>
            Replay a few pre-change inputs through both writer versions — if
            the old one still produces structured output, that confirms the
            prompt/model change as the cause.
          </li>
        </ol>
        If restoring the format fixes critic, make the fix in writer. Patching
        critic to accept prose would just hide the regression.
      </>
    ),
  },
];

const TOTAL = 15400;

export default function ClaudeIntegration() {
  const ref = useRef<HTMLDivElement>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const [elapsed, setElapsed] = useState(0);
  const [runId, setRunId] = useState(0);
  const started = useRef(false);

  const play = useCallback(() => {
    setElapsed(0);
    setRunId((r) => r + 1);
  }, []);

  // advance a clock while playing; each line shows once elapsed passes its `at`
  useEffect(() => {
    if (runId === 0) return;
    const t0 = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const e = now - t0;
      setElapsed(e);
      if (e < TOTAL) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [runId]);

  // auto-play on first scroll into view (respect reduced motion: show all)
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setElapsed(TOTAL);
      return;
    }
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !started.current) {
          started.current = true;
          play();
        }
      },
      { threshold: 0.3 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [play]);

  const done = elapsed >= TOTAL;
  const visibleCount = SCRIPT.filter((l) => elapsed >= l.at).length;

  // keep the newest message in view as the conversation grows
  useEffect(() => {
    const el = bodyRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [visibleCount]);

  return (
    <section id="claude" className="sect">
      <div className="sect-inner">
        <div className="num">Claude Code</div>
        <h2>Debug agent behavior from Claude Code.</h2>
        <p className="sect-sub sect-sub-wide">
          Ask what changed, trace the failure across agents, and get the next
          debugging steps without leaving your development workflow.
        </p>
        <div className="chat chat-center" ref={ref}>
          <div className="chat-bar">
            <span className="chat-title">Claude Code</span>
            <button
              type="button"
              className="chat-replay"
              onClick={play}
              aria-label="Replay conversation"
            >
              {done ? "↻ replay" : "● live"}
            </button>
          </div>
          <div className="chat-body" ref={bodyRef}>
            {SCRIPT.map((line, i) => {
              if (line.kind === "user") {
                return (
                  <div key={i} className="cmsg cmsg-user" data-show={elapsed >= line.at}>
                    <span className="cavatar cavatar-user">you</span>
                    <div className="cbubble">{line.text}</div>
                  </div>
                );
              }
              if (line.kind === "tool") {
                // visible only while Claude is "working"; collapses when the answer lands
                const active = elapsed >= line.at && elapsed < line.until;
                return (
                  <div key={i} className="ctool" data-show={active} data-gone={elapsed >= line.until}>
                    <span className="ctool-status" />
                    <code>{line.tool}</code>
                    <span className="ctool-src">AgentPulse MCP</span>
                    {active && (
                      <span className="ctool-dots" aria-hidden>
                        <i /><i /><i />
                      </span>
                    )}
                  </div>
                );
              }
              return (
                <div key={i} className="cmsg cmsg-ai" data-show={elapsed >= line.at}>
                  <span className="cavatar cavatar-ai">✳</span>
                  <div className="cbubble">{line.text}</div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
}
