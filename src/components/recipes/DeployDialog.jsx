"use client";

import { useState, useEffect, useCallback } from "react";
import { createPortal } from "react-dom";
import { ExternalLink, X, Copy, Check, ChevronRight } from "lucide-react";
import { PlatformLogo } from "@/components/icons/PlatformLogos";

function InstallBlock({ code }) {
  const [copied, setCopied] = useState(false);
  const onCopy = useCallback(() => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  }, [code]);
  return (
    <div className="relative rounded-md border border-border bg-[var(--code-block-bg)] text-[var(--code-block-fg)]">
      <button
        onClick={onCopy}
        className={`absolute top-1.5 right-1.5 inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] transition-colors ${
          copied ? "text-green-600 dark:text-green-400" : "text-muted-foreground hover:text-foreground"
        }`}
      >
        {copied ? <><Check size={10} /> Copied</> : <><Copy size={10} /> Copy</>}
      </button>
      <pre className="px-3 py-2.5 text-xs font-mono leading-relaxed whitespace-pre overflow-x-auto">
        {code}
      </pre>
    </div>
  );
}

export function DeployDialog({ platforms, hfId }) {
  const [open, setOpen] = useState(false);
  const [activeId, setActiveId] = useState(platforms?.[0]?.id || null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e) => {
      if (e.key === "Escape") setOpen(false);
    };
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = prev;
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  if (!platforms?.length) return null;

  const active = platforms.find((p) => p.id === activeId) || platforms[0];

  const dialog = open && typeof document !== "undefined" ? createPortal(
    <>
      <div
        className="fixed inset-0 z-40 bg-background/70 backdrop-blur-sm"
        onClick={() => setOpen(false)}
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-label={`Self host vLLM for ${hfId}`}
        className="fixed left-1/2 top-1/2 z-50 -translate-x-1/2 -translate-y-1/2 w-[min(580px,calc(100vw-2rem))] max-h-[calc(100vh-4rem)] overflow-y-auto rounded-xl border border-border bg-card shadow-2xl"
      >
        <div className="flex items-center justify-between px-5 py-3 border-b border-border">
          <div className="min-w-0">
            <h3 className="text-sm font-semibold">Self host vLLM</h3>
            <p className="text-xs text-muted-foreground font-mono truncate">{hfId}</p>
          </div>
          <button
            onClick={() => setOpen(false)}
            className="shrink-0 p-1 rounded-md text-muted-foreground hover:bg-foreground/10 hover:text-foreground transition-colors"
            aria-label="Close"
          >
            <X size={16} />
          </button>
        </div>

        <div role="tablist" className="flex border-b border-border px-2 bg-muted/30">
          {platforms.map((p) => {
            const isActive = p.id === active.id;
            return (
              <button
                key={p.id}
                role="tab"
                aria-selected={isActive}
                onClick={() => setActiveId(p.id)}
                className={`flex items-center gap-2 px-3 py-2.5 text-sm font-medium border-b-2 -mb-px transition-colors ${
                  isActive
                    ? "border-vllm-blue text-foreground"
                    : "border-transparent text-muted-foreground hover:text-foreground"
                }`}
              >
                <PlatformLogo id={p.id} className="w-4 h-4 rounded" />
                {p.name}
              </button>
            );
          })}
        </div>

        <div className="p-5 space-y-3">
          <div className="flex items-start justify-between gap-4">
            {active.blurb && (
              <p className="text-xs text-muted-foreground leading-relaxed">{active.blurb}</p>
            )}
            <div className="shrink-0 flex items-center gap-3 text-xs font-medium whitespace-nowrap">
              {active.script && (
                <a
                  href={active.script}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-vllm-blue hover:underline"
                >
                  View script
                  <ExternalLink size={11} />
                </a>
              )}
              <a
                href={active.url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 text-vllm-blue hover:underline"
              >
                Open guide
                <ExternalLink size={11} />
              </a>
            </div>
          </div>
          {active.install && <InstallBlock code={active.install.trim()} />}
          <p className="text-[11px] text-muted-foreground/70 pt-1">
            Then copy the <code className="font-mono">vllm serve</code> command from the builder above.
          </p>
        </div>
      </div>
    </>,
    document.body
  ) : null;

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        aria-haspopup="dialog"
        className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-vllm-blue transition-colors group"
      >
        <span className="flex -space-x-1">
          {platforms.map((p) => (
            <PlatformLogo
              key={p.id}
              id={p.id}
              className="w-3.5 h-3.5 rounded ring-1 ring-background"
            />
          ))}
        </span>
        Self Host vLLM
        <ChevronRight size={11} className="opacity-60 transition-transform group-hover:translate-x-0.5" />
      </button>
      {dialog}
    </>
  );
}
