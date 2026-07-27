/**
 * Hero — full-bleed video background (braided river, watermark cropped in
 * encode), copy on the right over a scrim that darkens toward the right.
 */
export default function Hero() {
  return (
    <section className="hero" id="top">
      <div className="hero-media" aria-hidden>
        <video autoPlay muted loop playsInline preload="metadata" poster="hero-poster.jpg">
          <source src="hero.mp4" type="video/mp4" />
        </video>
      </div>
      <div className="hero-scrim" aria-hidden />

      <div className="hero-inner">
        <div className="hero-copy">
          <div className="eyebrow">Drift investigation for multi-agent systems</div>
          <h1>
            <span className="l1">Traces show what happened.</span>
            <br />
            <span className="one-line">We show you <span className="grad-text">where to look</span>.</span>
          </h1>
          <p className="lead">
            Identify which agent drifted, understand why, and know what to check next.
          </p>
          <div className="hero-cta">
            <a className="btn btn-primary" href="https://github.com/prove-ai/agentpulse">
              Get started free <span aria-hidden>→</span>
            </a>
            <a className="btn btn-ghost" href="https://calendly.com/nick-proveai/agent-pulse-demo">
              Chat with our engineers
            </a>
          </div>
          <div className="hero-trust">Open source · MIT License · Works with OpenAI, Anthropic, LangChain</div>
        </div>
      </div>
    </section>
  );
}
