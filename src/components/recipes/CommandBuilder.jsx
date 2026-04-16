"use client";

import { useState, useMemo, useCallback } from "react";
import { useSearchParams, useRouter, usePathname } from "next/navigation";
import { Badge } from "@/components/ui/badge";
import { Copy, Check, Terminal, Gauge } from "lucide-react";
import { resolveCommand, recommendStrategy, filterHardwareByVram } from "@/lib/command-synthesis";

function CopyButton({ text, className = "" }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [text]);
  return (
    <button
      onClick={handleCopy}
      className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
        copied
          ? "bg-green-500/20 text-green-600 dark:text-green-400"
          : "bg-foreground/10 text-foreground/60 hover:bg-foreground/15 hover:text-foreground/80"
      } ${className}`}
    >
      {copied ? <><Check size={12} /> Copied</> : <><Copy size={12} /> Copy</>}
    </button>
  );
}

function PopoverButton({ label, code }) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-medium bg-foreground/5 text-foreground/60 hover:bg-foreground/10 hover:text-foreground/80 transition-colors"
      >
        {label === "Verify" ? <Terminal size={11} /> : <Gauge size={11} />}
        {label}
      </button>
      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
          <div className="absolute right-0 top-full mt-2 z-50 w-[400px] max-w-[90vw] rounded-xl border border-border bg-card shadow-xl overflow-hidden">
            <div className="flex items-center justify-between px-3 py-2 border-b border-border">
              <span className="text-xs font-medium">{label}</span>
              <button
                onClick={handleCopy}
                className={`text-[11px] flex items-center gap-1 px-2 py-0.5 rounded transition-colors ${
                  copied ? "text-green-600" : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {copied ? <><Check size={10} /> Copied</> : <><Copy size={10} /> Copy</>}
              </button>
            </div>
            <pre className="px-3 py-2.5 text-xs font-mono leading-relaxed whitespace-pre overflow-x-auto bg-[var(--code-block-bg)] text-[var(--code-block-fg)]">
              {code}
            </pre>
          </div>
        </>
      )}
    </div>
  );
}

export function CommandBuilder({ recipe, strategies, taxonomy }) {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();

  const [variant, setVariant] = useState(searchParams.get("variant") || "default");
  // Default hardware: pick smallest compatible profile for the default variant
  const defaultHw = useMemo(() => {
    const defaultVariant = recipe.variants?.default || Object.values(recipe.variants || {})[0] || {};
    const compatible = filterHardwareByVram(taxonomy.hardware_profiles, defaultVariant);
    return compatible[0] || "8x-h100";
  }, [recipe, taxonomy]);
  const [hwId, setHwId] = useState(searchParams.get("hardware") || defaultHw);
  const [strategyOverride, setStrategyOverride] = useState(searchParams.get("strategy") || "");
  const [features, setFeatures] = useState(() => {
    const fp = searchParams.get("features");
    if (fp) return fp.split(",").filter(Boolean);
    return Object.keys(recipe.features || {}).filter((f) => !(recipe.opt_in_features || []).includes(f));
  });

  const currentVariant = recipe.variants?.[variant] || recipe.variants?.default || {};
  const compatibleHw = useMemo(
    () => filterHardwareByVram(taxonomy.hardware_profiles, currentVariant),
    [taxonomy, currentVariant]
  );
  const hwProfile = taxonomy.hardware_profiles?.[hwId] || {};
  const recommended = useMemo(() => recommendStrategy(recipe, hwProfile), [recipe, hwProfile]);
  const activeStrategy = strategyOverride || recommended;

  const compatibleStrategies = useMemo(() => {
    return (recipe.compatible_strategies || []).filter((s) => {
      const strat = strategies[s];
      if (!strat) return false;
      if (hwProfile.multi_node && strat.deploy_type === "single_node") return false;
      if (!hwProfile.multi_node && strat.deploy_type === "multi_node") return false;
      return true;
    });
  }, [recipe, strategies, hwProfile]);

  const result = useMemo(
    () => resolveCommand(recipe, variant, activeStrategy, hwId, features, strategies, taxonomy),
    [recipe, variant, activeStrategy, hwId, features, strategies, taxonomy]
  );

  const syncUrl = useCallback(
    (updates) => {
      const sp = new URLSearchParams(searchParams.toString());
      for (const [k, v] of Object.entries(updates)) {
        if (v && v !== "default" && v !== recommended) sp.set(k, v);
        else sp.delete(k);
      }
      const qs = sp.toString();
      router.replace(qs ? `?${qs}` : pathname, { scroll: false });
    },
    [searchParams, router, recommended]
  );

  const toggleFeature = (f) => {
    const next = features.includes(f) ? features.filter((x) => x !== f) : [...features, f];
    setFeatures(next);
    syncUrl({ features: next.length > 0 ? next.join(",") : "" });
  };

  const isPd = result.deployType === "pd_cluster";

  const modelId = recipe.variants?.[variant]?.model_id || recipe.model?.model_id || "model";
  const verifyCmd = `curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${modelId}",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 32
  }'`;
  const benchCmd = `vllm bench serve \\
  --model ${modelId} \\
  --dataset-name random \\
  --random-input-len 1024 \\
  --random-output-len 1024 \\
  --num-prompts 100 \\
  --max-concurrency 32`;

  return (
    <div className="space-y-4">
      {/* ── Command output (the hero) ── */}
      <div className="rounded-2xl overflow-hidden bg-[var(--command-bg)] border border-border">
        {isPd ? (
          <PdClusterBlock result={result} />
        ) : (
          <div>
            <div className="flex items-center justify-between px-4 pt-3">
              <span className="text-[11px] text-[var(--command-fg)]/40 font-mono">vllm serve</span>
              <div className="flex items-center gap-1.5">
                <CopyButton text={result.command} />
                <PopoverButton label="Verify" code={verifyCmd} />
                <PopoverButton label="Bench" code={benchCmd} />
              </div>
            </div>
            <pre className="px-4 py-3 text-[13px] text-[var(--command-fg)] font-mono leading-relaxed whitespace-pre overflow-x-auto">
              {result.command}
            </pre>
          </div>
        )}
      </div>

      {/* ── Configuration controls ── */}
      <div className="rounded-xl border border-border p-4 space-y-4">
        {/* Row 1: Variant (radio row) */}
        <div>
          <label className="text-[10px] font-semibold text-muted-foreground mb-2 block uppercase tracking-widest">Variant</label>
          <div className="flex flex-wrap gap-2">
            {Object.entries(recipe.variants || {}).map(([key, v]) => (
              <button
                key={key}
                onClick={() => { setVariant(key); syncUrl({ variant: key }); }}
                className={`rounded-lg border px-3 py-2 text-xs font-mono transition-all ${
                  variant === key
                    ? "border-vllm-blue bg-vllm-blue/5 shadow-sm ring-1 ring-vllm-blue/20"
                    : "border-border hover:border-muted-foreground/30"
                }`}
              >
                <span className="font-semibold">{v.precision?.toUpperCase()}</span>
                <span className="text-muted-foreground ml-1.5">{v.vram_minimum_gb} GB</span>
              </button>
            ))}
          </div>
        </div>

        {/* Row 2: Hardware + Strategy + Features in a responsive row */}
        <div className="flex flex-wrap gap-4">
          <div className="min-w-[180px]">
            <label className="text-[10px] font-semibold text-muted-foreground mb-1.5 block uppercase tracking-widest">Hardware</label>
            <select
              value={hwId}
              onChange={(e) => { setHwId(e.target.value); setStrategyOverride(""); syncUrl({ hardware: e.target.value, strategy: "" }); }}
              className="w-full rounded-lg border border-border bg-background px-3 py-2 text-xs focus:outline-none focus:ring-2 focus:ring-vllm-blue/30"
            >
              {compatibleHw.map((id) => {
                const p = taxonomy.hardware_profiles[id];
                return <option key={id} value={id}>{p.display_name} ({p.vram_gb > 0 ? `${p.vram_gb} GB` : "multi"})</option>;
              })}
            </select>
          </div>

          <div className="min-w-[180px]">
            <label className="text-[10px] font-semibold text-muted-foreground mb-1.5 block uppercase tracking-widest">Strategy</label>
            <select
              value={activeStrategy}
              onChange={(e) => { setStrategyOverride(e.target.value); syncUrl({ strategy: e.target.value }); }}
              className="w-full rounded-lg border border-border bg-background px-3 py-2 text-xs focus:outline-none focus:ring-2 focus:ring-vllm-blue/30"
            >
              {compatibleStrategies.map((s) => (
                <option key={s} value={s}>
                  {strategies[s]?.display_name || s}{s === recommended ? " ★" : ""}
                </option>
              ))}
            </select>
          </div>

          {/* Features as inline pills */}
          {Object.keys(recipe.features || {}).length > 0 && (
            <div>
              <label className="text-[10px] font-semibold text-muted-foreground mb-1.5 block uppercase tracking-widest">Features</label>
              <div className="flex flex-wrap gap-1.5">
                {Object.entries(recipe.features || {}).map(([key]) => {
                  const active = features.includes(key);
                  return (
                    <button
                      key={key}
                      onClick={() => toggleFeature(key)}
                      className={`rounded-full px-2.5 py-1 text-[11px] font-medium transition-all ${
                        active
                          ? "bg-vllm-blue/10 text-vllm-blue ring-1 ring-vllm-blue/20"
                          : "bg-secondary text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      {key.replace(/_/g, " ")}
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function PdClusterBlock({ result }) {
  const [tab, setTab] = useState("prefill");
  const tabs = [
    { id: "prefill", label: "Prefill", command: result.prefillCommand },
    { id: "decode", label: "Decode", command: result.decodeCommand },
  ];
  const active = tabs.find((t) => t.id === tab) || tabs[0];

  return (
    <div>
      <div className="flex items-center justify-between px-4 pt-3">
        <div className="flex gap-0">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                tab === t.id ? "bg-foreground/10 text-[var(--command-fg)]" : "text-[var(--command-fg)]/40 hover:text-[var(--command-fg)]/70"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
        <CopyButton text={active.command} />
      </div>
      <pre className="px-4 py-3 text-[13px] text-[var(--command-fg)] font-mono leading-relaxed whitespace-pre overflow-x-auto">
        {active.command}
      </pre>
    </div>
  );
}
