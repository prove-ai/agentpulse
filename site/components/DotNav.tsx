"use client";

import { useEffect, useState } from "react";

export type DotSection = { id: string; label: string };

/**
 * Right-side diamond dot navigation (Chamber-style).
 * Highlights the section currently in view; click to jump.
 */
export default function DotNav({ sections }: { sections: DotSection[] }) {
  const [active, setActive] = useState(sections[0]?.id);

  useEffect(() => {
    // Next.js App Router drops the #hash scroll on a full-page load and its
    // scroll restoration can reset to top after hydration (e.g. arriving from
    // /support via "/#how") — so re-apply the anchor scroll a few times.
    const timers: ReturnType<typeof setTimeout>[] = [];
    const jumpToHash = () => {
      const hash = window.location.hash.slice(1);
      if (hash) {
        document.getElementById(hash)?.scrollIntoView({ behavior: "instant" as ScrollBehavior });
      }
    };
    if (window.location.hash) {
      requestAnimationFrame(jumpToHash);
      timers.push(setTimeout(jumpToHash, 150), setTimeout(jumpToHash, 450));
    }
    window.addEventListener("hashchange", jumpToHash);

    const observer = new IntersectionObserver(
      (entries) => {
        // pick the visible section closest to the viewport center
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio);
        if (visible[0]) {
          setActive(visible[0].target.id);
          // drives the ambient background hue (see globals.css)
          document.body.dataset.section = visible[0].target.id;
        }
      },
      { rootMargin: "-35% 0px -35% 0px", threshold: [0, 0.25, 0.5, 0.75, 1] },
    );
    for (const s of sections) {
      const el = document.getElementById(s.id);
      if (el) observer.observe(el);
    }
    return () => {
      observer.disconnect();
      timers.forEach(clearTimeout);
      window.removeEventListener("hashchange", jumpToHash);
      delete document.body.dataset.section;
    };
  }, [sections]);

  return (
    <nav className="dotnav" aria-label="Section navigation">
      {sections.map((s) => (
        <a
          key={s.id}
          href={`#${s.id}`}
          data-active={active === s.id}
          data-label={s.label}
          aria-label={s.label}
        />
      ))}
    </nav>
  );
}
