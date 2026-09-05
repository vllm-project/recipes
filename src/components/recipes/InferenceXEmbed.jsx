"use client";

import { useEffect, useRef, useState } from "react";
import { ExternalLink } from "lucide-react";
import {
  INFERENCEX_BASE_URL,
  inferencexAgentxUrl,
  inferencexEmbedUrl,
  inferencexModelUrl,
} from "@/lib/inferencex";

const MIN_HEIGHT = 560;
const RESIZE_MESSAGE_TYPE = "inferencex:embed-resize";

function readTheme() {
  return document.documentElement.classList.contains("dark") ? "dark" : "light";
}

/**
 * Live InferenceX benchmark chart for the recipe's model, filtered to vLLM.
 *
 * The iframe follows the site theme (the `dark` class on <html>, toggled by
 * ThemeToggle) and grows to the height the embed reports via postMessage so
 * the chart never scrolls inside the card. Only messages from the InferenceX
 * origin are honoured.
 *
 * The theme is applied client-side after hydration, so the server can't know
 * it. The frame is therefore only rendered once the theme has been read on the
 * client: SSR ships the skeleton alone, and a dark-mode visitor fetches the
 * dark embed once instead of a light one that is immediately thrown away.
 */
export function InferenceXEmbed({ config, title }) {
  const [theme, setTheme] = useState(null);
  const [height, setHeight] = useState(MIN_HEIGHT);
  const [loadedSrc, setLoadedSrc] = useState(null);
  const frameRef = useRef(null);

  useEffect(() => {
    setTheme(readTheme());
    const observer = new MutationObserver(() => setTheme(readTheme()));
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const embedOrigin = new URL(INFERENCEX_BASE_URL).origin;
    const onMessage = (event) => {
      if (event.origin !== embedOrigin) return;
      if (event.source !== frameRef.current?.contentWindow) return;
      const data = event.data;
      if (!data || data.type !== RESIZE_MESSAGE_TYPE) return;
      if (typeof data.height !== "number" || !Number.isFinite(data.height)) return;
      setHeight(Math.max(MIN_HEIGHT, Math.ceil(data.height)));
    };
    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, []);

  const src = theme ? inferencexEmbedUrl(config, theme) : null;
  // Tied to the src rather than a bare boolean so a theme switch (new src,
  // remounted frame) brings the skeleton back until the new embed has loaded.
  const loaded = src !== null && loadedSrc === src;

  return (
    <div className="space-y-3">
      <p className="text-sm text-muted-foreground">
        Throughput vs Interactivity curves on 1Mil Long Context Multi Turn Agentic Coding Workloads (
        <a
          href={inferencexAgentxUrl()}
          target="_blank"
          rel="noopener noreferrer"
          className="text-vllm-blue hover:underline"
        >
          AgentX
        </a>
        ) measured by SemiAnalysis&apos;{" "}
        <a
          href={inferencexModelUrl(config)}
          target="_blank"
          rel="noopener noreferrer"
          className="text-vllm-blue hover:underline"
        >
          InferenceX
        </a>{" "}
        following{" "}
        <a href="https://recipes.vllm.ai" className="text-vllm-blue hover:underline">
          recipes.vllm.ai
        </a>
      </p>
      <a
        href={inferencexModelUrl(config)}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex items-center gap-1.5 text-base font-medium text-vllm-blue hover:underline transition-colors"
      >
        Open the full interactive dashboard on InferenceX
        <ExternalLink size={16} />
      </a>
      <div className="relative rounded-xl overflow-hidden bg-background" style={{ height }}>
        {!loaded && (
          <div
            className="absolute inset-0 animate-pulse bg-muted"
            aria-hidden="true"
          />
        )}
        {src && (
          <iframe
            ref={frameRef}
            key={src}
            src={src}
            title={`InferenceX benchmarks for ${title} on vLLM`}
            loading="lazy"
            onLoad={() => setLoadedSrc(src)}
            className="block h-full w-full border-0"
            sandbox="allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox"
            referrerPolicy="strict-origin-when-cross-origin"
          />
        )}
      </div>
    </div>
  );
}
