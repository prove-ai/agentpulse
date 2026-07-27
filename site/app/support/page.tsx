import type { Metadata } from "next";
import Support from "@/components/Support";

export const metadata: Metadata = {
  title: "Support — AgentPulse",
  description: "Get help with AgentPulse: email support, book a call, or join GitHub Discussions.",
};

export default function SupportPage() {
  return (
    <main>
      <Support />
    </main>
  );
}
