"use client";

import { useEffect, useRef, useState } from "react";

/**
 * 02 // Drift investigation — an animated walkthrough of a real investigation.
 * Pipeline: researcher → analyst → writer → critic, with handoffs between.
 * Steps: 1 detect the breach (critic) → 2 read the handoff payload →
 * 3 find the origin (writer) → 4 clear the upstream agents.
 * Auto-plays on scroll into view; the step pills are clickable.
 */
const STEPS = [
  {
    title: "Spot the symptom",
    text: "Critic success drops from 92% to 22%, while retries and token use climb. The decline persists, so the investigation begins.",
  },
  {
    title: "Inspect the input",
    text: "Critic is receiving a very different payload from Writer: 180% longer and structurally changed. Critic may be the symptom, not the cause.",
  },
  {
    title: "Trace the change",
    text: "Writer’s output changed even though its input from Analyst stayed stable. The drift starts inside Writer.",
  },
  {
    title: "Confirm the source",
    text: "Writer’s prompt and model changed at run 18, exactly where the drift begins. The fix belongs in Writer, not Critic.",
  },
] as const;

const STEP_MS = 3400;

export default function Investigation() {
  const ref = useRef<HTMLDivElement>(null);
  const [step, setStep] = useState(0); // 0 = not started; 1..4 active
  const [auto, setAuto] = useState(true);
  const started = useRef(false);

  // auto-play on first view
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !started.current) {
          started.current = true;
          setStep(1);
        }
      },
      { threshold: 0.35 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  // advance while auto mode is on
  useEffect(() => {
    if (!auto || step === 0 || step >= STEPS.length) return;
    const t = setTimeout(() => setStep((s) => Math.min(s + 1, STEPS.length)), STEP_MS);
    return () => clearTimeout(t);
  }, [auto, step]);

  const goto = (s: number) => {
    setAuto(false);
    setStep(s);
  };

  const cap = STEPS[Math.max(0, step - 1)];

  return (
    <section id="investigation" className="sect">
      <div className="sect-inner">
        <div className="num">Investigation</div>
        <h2>From symptom to source.</h2>
        <p className="sect-sub sect-sub-wide">
          The failing agent may only be the symptom. AgentPulse follows the evidence across agents, handoffs, and changes to find where the failure actually began.
        </p>

        <div className="inv" data-step={step} ref={ref}>
          <div className="inv-steps" role="tablist" aria-label="Investigation steps">
            {STEPS.map((s, i) => (
              <button
                key={s.title}
                type="button"
                role="tab"
                aria-selected={step === i + 1}
                data-active={step === i + 1}
                data-visited={step > i}
                onClick={() => goto(i + 1)}
              >
                <span className="inv-step-n">{i + 1}</span>
                {s.title}
              </button>
            ))}
          </div>

          <div className="inv-cap" key={step}>
            {step === 0 ? (
              <span className="inv-cap-idle">The investigation plays here as it enters view…</span>
            ) : (
              <>
                <span className="inv-cap-title">{cap.title}</span>
                <span className="inv-cap-text">{cap.text}</span>
              </>
            )}
          </div>

          <div className="inv-pipe-scroll">
            <div className="inv-pipe">
              <div className="anode" data-id="researcher">
                <div className="anode-name">Researcher</div>
                <div className="am"><span>success</span><b>98%</b></div>
                <div className="am"><span>latency</span><b>2.1s</b></div>
                <div className="am"><span>input</span><b className="ok">stable</b></div>
              </div>

              <div className="hoff" data-id="h-ra">
                <span className="hoff-line" />
                <span className="hoff-diamond" />
                <span className="hoff-line" />
              </div>

              <div className="anode" data-id="analyst">
                <div className="anode-name">Analyst</div>
                <div className="am"><span>success</span><b>97%</b></div>
                <div className="am"><span>latency</span><b>3.4s</b></div>
                <div className="am"><span>input</span><b className="ok">stable</b></div>
              </div>

              <div className="hoff" data-id="h-aw">
                <span className="hoff-line" />
                <span className="hoff-diamond" />
                <span className="hoff-line" />
              </div>

              <div className="anode" data-id="writer">
                <span className="atag tag-cause">Root cause</span>
                <div className="anode-name">Writer</div>
                <div className="am"><span>input</span><b className="ok">stable</b></div>
                <div className="am"><span>output style</span><b className="bad">drifted</b></div>
                <div className="am"><span>prompt + model</span><b className="warn">changed @ run 18</b></div>
              </div>

              <div className="hoff hoff-payload" data-id="h-wc">
                <div className="hoff-chip">
                  payload length <b className="warn">+180%</b> · structure{" "}
                  <b className="warn">drifted</b>
                </div>
                <span className="hoff-line" />
                <span className="hoff-diamond" />
                <span className="hoff-line" />
              </div>

              <div className="anode" data-id="critic">
                <span className="atag tag-symptom">Symptom</span>
                <div className="anode-name">Critic</div>
                <div className="am"><span>success</span><b className="bad">92% → 22%</b></div>
                <div className="am"><span>retries</span><b className="warn">×3</b></div>
                <div className="am"><span>tokens / turn</span><b className="warn">+45%</b></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
