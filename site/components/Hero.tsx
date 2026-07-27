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
          <h1>Find where failures begin</h1>
          <p className="lead">
            Follow changes across agents, handoffs, prompts, and models to the
            most likely source.
          </p>
          <div className="hero-cta">
            <a className="btn btn-primary" href="https://github.com/prove-ai/agentpulse">
              Get started free
            </a>
            <a className="btn btn-ghost" href="https://calendly.com/nick-proveai/agent-pulse-demo">
              Chat with our engineers
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}
