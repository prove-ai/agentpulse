"use client";

import { useState } from "react";

const SNIPPETS = {
  python: (
    <pre>
      <span className="t-dim">$</span> git clone https://github.com/prove-ai/agentpulse{"\n\n"}
      <span className="t-dim"># your entrypoint — before any agent imports</span>{"\n"}
      <span className="t-key">from</span> sdk <span className="t-key">import</span> instrument{"\n"}
      instrument(task_type=<span className="t-str">&apos;my-system&apos;</span>, prompt_version=<span className="t-num">1</span>,{"\n"}
      {"           "}db_name=<span className="t-str">&apos;my-system&apos;</span>)
    </pre>
  ),
  node: (
    <pre>
      <span className="t-dim">$</span> git clone https://github.com/prove-ai/agentpulse{"\n\n"}
      <span className="t-dim">{"//"} your entrypoint — before any agent imports</span>{"\n"}
      <span className="t-key">import</span> {"{ instrument }"} <span className="t-key">from</span> <span className="t-str">&quot;agentpulse&quot;</span>;{"\n"}
      instrument({"{"} taskType: <span className="t-str">&quot;my-system&quot;</span>, promptVersion: <span className="t-num">1</span>,{"\n"}
      {"            "}dbName: <span className="t-str">&quot;my-system&quot;</span> {"}"});
    </pre>
  ),
};

export default function Install() {
  const [lang, setLang] = useState<"python" | "node">("python");

  return (
    <section id="install" className="sect">
      <div className="sect-inner feature-grid grid-install">
        <div className="term">
          <div className="term-head">
            <div className="term-bar">
              <span /><span /><span />
            </div>
            <div className="lang-switch" role="tablist" aria-label="SDK language">
              <button
                type="button"
                role="tab"
                aria-selected={lang === "python"}
                data-active={lang === "python"}
                onClick={() => setLang("python")}
              >
                <span className="lang-badge">Py</span>Python
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={lang === "node"}
                data-active={lang === "node"}
                onClick={() => setLang("node")}
              >
                <span className="lang-badge">Js</span>Node
              </button>
            </div>
          </div>
          {SNIPPETS[lang]}
        </div>
        <div className="install-side">
          <div className="num">Integration</div>
          <h2>Instrument once.</h2>
          <p className="sect-sub">
            One call adds OpenTelemetry-native tracing across your agents, tools, 
            and handoffs, without changing their implementation.
          </p>
        </div>
        <div className="works-with">
          <span className="ww-label">Works with</span>
          <span className="ww-chip">LangChain</span>
          <span className="ww-chip">LangGraph</span>
          <span className="ww-chip">OpenAI SDK</span>
          <span className="ww-chip">Anthropic SDK</span>
        </div>
      </div>
    </section>
  );
}
