import DotNav from "@/components/DotNav";
import FinalCta from "@/components/FinalCta";
import Hero from "@/components/Hero";
import ClaudeIntegration from "@/components/sections/ClaudeIntegration";
import Console from "@/components/sections/Console";
import Install from "@/components/sections/Install";
import Investigation from "@/components/sections/Investigation";

// Section map — drives the right-side dot nav.
const SECTIONS = [
  { id: "console", label: "Overview" },
  { id: "investigation", label: "Investigation" },
  { id: "install", label: "Integration" },
  { id: "claude", label: "Claude Code" },
  { id: "start", label: "Get started" },
];

export default function Home() {
  return (
    <>
      <DotNav sections={SECTIONS} />
      <main>
        <Hero />
        <Console />
        <Investigation />
        <Install />
        <ClaudeIntegration />
        <FinalCta />
      </main>
    </>
  );
}
