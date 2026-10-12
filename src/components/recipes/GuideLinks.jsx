"use client";

import { useEffect, useRef, useState } from "react";

// Keep Markdown rendering on the server; only navigation and copying need JS.
export function GuideLinks({ children }) {
  const root = useRef(null);
  const timeout = useRef(null);
  const [status, setStatus] = useState("");

  useEffect(() => {
    function revealTarget() {
      let id;
      try {
        id = decodeURIComponent(window.location.hash.slice(1));
      } catch {
        return;
      }
      const target = id && document.getElementById(id);
      if (!target || !root.current?.contains(target)) return;
      const details = target.closest("details");
      if (details) details.open = true;
      target.scrollIntoView({ block: "start" });
    }
    revealTarget();
    window.addEventListener("hashchange", revealTarget);
    return () => {
      window.removeEventListener("hashchange", revealTarget);
      clearTimeout(timeout.current);
    };
  }, []);

  async function copyLink(event) {
    const link = event.target.closest("a[data-guide-link]");
    if (!link || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    const url = new URL(window.location.href);
    url.hash = link.hash;
    window.history.replaceState(window.history.state, "", url);
    window.dispatchEvent(new HashChangeEvent("hashchange"));
    clearTimeout(timeout.current);
    try {
      await navigator.clipboard.writeText(url.href);
      setStatus("Link copied");
    } catch {
      setStatus("Copy the link from the address bar");
    }
    timeout.current = setTimeout(() => setStatus(""), 3000);
  }

  return (
    <div ref={root} onClick={copyLink}>
      {children}
      <div role="status" aria-live="polite" className={status ? "fixed bottom-6 left-1/2 -translate-x-1/2 z-50 rounded-lg border border-border bg-background px-4 py-2 text-sm shadow-lg" : "sr-only"}>
        {status}
      </div>
    </div>
  );
}
