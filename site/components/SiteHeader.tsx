"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

export default function SiteHeader() {
  const [scrolled, setScrolled] = useState(false);
  // Only the home page has a dark hero behind the transparent header;
  // every other page starts on a light background.
  const overHero = usePathname() === "/";

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header className="site-header fixed inset-x-0 top-0 z-50" data-scrolled={scrolled || !overHero}>
      <div className="mx-auto grid max-w-[1180px] grid-cols-[1fr_auto_1fr] items-center px-7 py-4">
        <Link href="/" className="flex items-center gap-2.5 justify-self-start font-semibold text-[20px] text-white">
          <svg width="28" height="28" viewBox="0 0 32 32" aria-hidden>
            <rect width="32" height="32" rx="7" fill="#2f56e0" />
            <path
              d="M5 17h5l3-8 4 14 3-6h7"
              fill="none"
              stroke="#fff"
              strokeWidth="2.4"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          AgentPulse
          <span className="rounded-full border border-[#2c3a5c] px-2 py-0.5 font-mono text-[10px] font-medium tracking-[0.08em] text-[#9fb2e8]">
            OPEN SOURCE
          </span>
        </Link>
        <nav className="flex items-center gap-14 justify-self-center text-[14px] font-medium text-[var(--fg-soft)]">
          <Link className="hidden transition-colors hover:text-white sm:block" href="/#console">
            Features
          </Link>
          <a
            className="transition-colors hover:text-white"
            href="https://github.com/prove-ai/agentpulse"
          >
            GitHub
          </a>
          <Link className="hidden transition-colors hover:text-white sm:block" href="/support">
            Support
          </Link>
        </nav>
        <div className="justify-self-end" />
      </div>
    </header>
  );
}
