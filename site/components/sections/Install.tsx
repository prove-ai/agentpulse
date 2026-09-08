export default function Install() {
  return (
    <section id="install" className="sect">
      <div className="sect-inner feature-grid grid-install">
        <div className="term">
          <div className="term-head">
            <div className="term-bar">
              <span /><span /><span />
            </div>
          </div>
          <pre>
            <span className="t-dim">$</span> pip install proveai-agentpulse{"\n\n"}
            <span className="t-dim"># launch your app through agentpulse — no code changes</span>{"\n"}
            <span className="t-dim">$</span> agentpulse run python main.py{"\n\n"}
            <span className="t-dim"># then explore the captured runs</span>{"\n"}
            <span className="t-dim">$</span> agentpulse dashboard
          </pre>
        </div>
        <div className="install-side">
          <div className="num">Integration</div>
          <h2>Zero code changes.</h2>
          <p className="sect-sub">
            Launch your app through <code>agentpulse run</code> instead of{" "}
            <code>python</code>. Every LLM call, agent turn, tool call, and
            handoff is captured automatically — your agents&apos;
            implementation stays untouched.
          </p>
        </div>
        <div className="works-with">
          <span className="ww-label">Works with</span>
          <span className="ww-chip">LangChain</span>
          <span className="ww-chip">LangGraph</span>
          <span className="ww-chip">AutoGen</span>
          <span className="ww-chip">OpenAI SDK</span>
          <span className="ww-chip">Anthropic SDK</span>
        </div>
      </div>
    </section>
  );
}
