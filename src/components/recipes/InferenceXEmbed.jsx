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
  if (typeof document === "undefined") return "light";
  return document.documentElement.classList.contains("dark") ? "dark" : "light";
}

/**
 * Live InferenceX benchmark chart for the recipe's model, filtered to vLLM.
 *
 * The iframe follows the site theme (the `dark` class on <html>, toggled by
 * ThemeToggle) and grows to the height the embed reports via postMessage so
 * the chart never scrolls inside the card. Only messages from the InferenceX
 * origin are honoured.
 */
export function InferenceXEmbed({ config, title }) {
  const [theme, setTheme] = useState("light");
  const [height, setHeight] = useState(MIN_HEIGHT);
  const [loaded, setLoaded] = useState(false);
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

  const src = inferencexEmbedUrl(config, theme);

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
      <div className="relative rounded-xl overflow-hidden bg-background">
        {!loaded && (
          <div
            className="absolute inset-0 animate-pulse bg-muted"
            aria-hidden="true"
          />
        )}
        <iframe
          ref={frameRef}
          key={src}
          src={src}
          title={`InferenceX benchmarks for ${title} on vLLM`}
          loading="lazy"
          onLoad={() => setLoaded(true)}
          style={{ height }}
          className="block w-full border-0"
          sandbox="allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox"
          referrerPolicy="strict-origin-when-cross-origin"
        />
      </div>
    </div>
  );
}
