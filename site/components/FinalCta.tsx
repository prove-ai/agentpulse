export default function FinalCta() {
  return (
    <section id="start" className="final">
      <div className="final-inner">
        <div className="num">Get started</div>
        <h2>Try it on a real production trace</h2>
        <p className="final-sub">
          Bring a failed run or connect AgentPulse to your stack. We'll walk through the investigation with you in 30 minutes.
        </p>
        <div className="final-cta">
          <a className="btn btn-primary" href="https://github.com/prove-ai/agentpulse">
            Get Access
          </a>
          <a className="btn btn-ghost" href="https://calendly.com/nick-proveai/agent-pulse-demo">
            Talk to an engineer
          </a>
        </div>
        <div className="feedback-note">
          <span className="tag-fb">Open experiment</span>
          <p>
            AgentPulse is a reference implementation. We’re learning how teams investigate agent failures, what is missing, and where the workflow breaks.
          </p>
          <a className="btn btn-ghost" href="https://github.com/prove-ai/agentpulse/discussions">
            Share feedback in GitHub Discussions
          </a>
        </div>
      </div>
    </section>
  );
}
