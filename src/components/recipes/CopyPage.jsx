"use client";

// "Copy page" for AI assistants: the recipe as Markdown, rendered for the
// configuration the builder has selected (the `/<hf_id>.md?<query>` route,
// the same text an agent fetching that URL reads), plus links to view it (or
// its JSON) and hand it to Claude / ChatGPT, and to the API docs.

import { useEffect, useRef, useState } from "react";
import { track } from "@vercel/analytics";
import { Braces, Check, ChevronDown, Copy, ExternalLink, FileText } from "lucide-react";
import { useSelection } from "@/lib/selection-store";
import { ClaudeIcon, OpenAIIcon } from "@/components/icons/AssistantLogos";

const COPIED_MS = 2000;

export function CopyPage({ hfId, labels }) {
  const sel = useSelection(hfId);
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState(null); // null | "copied" | "failed"
  const root = useRef(null);

  // Before the builder mounts (or without JS) the recipe default renders.
  const query = sel?.query ? `?${sel.query}` : "";
  const mdPath = `/${hfId}.md${query}`;
  const jsonPath = `/${hfId}.json${query}`;

  useEffect(() => {
    if (!open) return;
    const onDown = (e) => { if (!root.current?.contains(e.target)) setOpen(false); };
    const onKey = (e) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("pointerdown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  useEffect(() => {
    if (!status) return;
    const t = setTimeout(() => setStatus(null), COPIED_MS);
    return () => clearTimeout(t);
  }, [status]);

  async function copyPage() {
    track("copy_page", { action: "copy", hfId });
    const text = fetch(mdPath).then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.text();
    });
    try {
      // Safari only writes the clipboard inside the click: hand it a
      // ClipboardItem whose content is still being fetched.
      if (typeof ClipboardItem !== "undefined" && navigator.clipboard?.write) {
        await navigator.clipboard.write([
          new ClipboardItem({ "text/plain": text.then((t) => new Blob([t], { type: "text/plain" })) }),
        ]);
      } else {
        await navigator.clipboard.writeText(await text);
      }
      setStatus("copied");
    } catch {
      setStatus("failed");
    }
  }

  const absolute = () => `${window.location.origin}${mdPath}`;
  const prompt = () =>
    `Read ${absolute()} — a vLLM serving recipe for ${hfId}, rendered for my configuration — and help me deploy it.`;
  const openIn = (where, url) => {
    track("copy_page", { action: where, hfId });
    window.open(url, "_blank", "noopener,noreferrer");
    setOpen(false);
  };

  const summary = sel && [
    labels.hardware[sel.hwId] || sel.hwId,
    labels.strategies[sel.strategy] || sel.strategy,
    labels.variants[sel.variant] || sel.variant,
  ].filter(Boolean).join(" · ");

  const item = "flex w-full items-center gap-2 px-3 py-2 text-left text-sm hover:bg-muted/60 focus-visible:bg-muted/60 outline-none";

  return (
    <div ref={root} className="relative shrink-0">
      <div className="inline-flex rounded-lg border border-border text-xs font-medium overflow-hidden">
        <button
          type="button"
          onClick={copyPage}
          className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 transition-colors hover:bg-muted/60 ${status === "copied" ? "text-green-600 dark:text-green-400" : ""}`}
          title="Copy this recipe as Markdown, for the configuration selected below"
          aria-label="Copy page"
        >
          {status === "copied" ? <Check size={13} /> : <Copy size={13} />}
          {/* Icon only on phones, where the title needs the width. */}
          <span className="hidden sm:inline">{status === "copied" ? "Copied" : status === "failed" ? "Copy failed" : "Copy page"}</span>
        </button>
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          aria-haspopup="menu"
          aria-expanded={open}
          aria-label="More ways to use this page with AI"
          className="border-l border-border px-1.5 transition-colors hover:bg-muted/60"
        >
          <ChevronDown size={13} className={`transition-transform ${open ? "rotate-180" : ""}`} />
        </button>
      </div>
      {open && (
        <div role="menu" className="absolute right-0 z-40 mt-1.5 w-64 rounded-xl border border-border bg-card py-1 shadow-xl">
          {summary && (
            <div className="px-3 pt-1.5 pb-2 text-[11px] text-muted-foreground border-b border-border mb-1">
              For the selected configuration: <span className="text-foreground">{summary}</span>
            </div>
          )}
          <a role="menuitem" href={mdPath} target="_blank" rel="noopener" className={item}
            onClick={() => { track("copy_page", { action: "view_markdown", hfId }); setOpen(false); }}>
            <FileText size={14} className="text-muted-foreground" /> View as Markdown
            <ExternalLink size={11} className="ml-auto opacity-50" />
          </a>
          <a role="menuitem" href={jsonPath} target="_blank" rel="noopener" className={item}
            onClick={() => { track("copy_page", { action: "view_json", hfId }); setOpen(false); }}>
            <Braces size={14} className="text-muted-foreground" /> View as JSON
            <ExternalLink size={11} className="ml-auto opacity-50" />
          </a>
          <div className="my-1 border-t border-border" />
          <button role="menuitem" type="button" className={item}
            onClick={() => openIn("claude", `https://claude.ai/new?q=${encodeURIComponent(prompt())}`)}>
            <ClaudeIcon /> Open in Claude
            <ExternalLink size={11} className="ml-auto opacity-50" />
          </button>
          <button role="menuitem" type="button" className={item}
            onClick={() => openIn("chatgpt", `https://chatgpt.com/?q=${encodeURIComponent(prompt())}`)}>
            <OpenAIIcon /> Open in ChatGPT
            <ExternalLink size={11} className="ml-auto opacity-50" />
          </button>
          <a role="menuitem" href="/api" className="mt-1 block border-t border-border px-3 pt-2 pb-1.5 text-xs text-muted-foreground hover:text-foreground">
            API docs →
          </a>
        </div>
      )}
    </div>
  );
}
