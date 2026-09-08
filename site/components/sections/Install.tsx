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
            <span className="t-dim"># run your app with this command</span>{"\n"}
            <span className="t-dim">$</span> agentpulse run python main.py{"\n\n"}
            <span className="t-dim"># then explore the captured runs</span>{"\n"}
            <span className="t-dim">$</span> agentpulse dashboard
          </pre>
        </div>
        <div className="install-side">
          <div className="num">Integration</div>
          <h2>One command.</h2>
          <p className="sect-sub">
            Run your app with <code>agentpulse run</code>. Every LLM call,
            agent turn, tool call, and handoff is captured automatically.
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
