"use client";

import { useState, useMemo, useCallback, useEffect } from "react";
import { useSearchParams, useRouter, usePathname } from "next/navigation";
import { Copy, Check, Terminal, Gauge, Sparkles, ChevronDown, Package } from "lucide-react";
import { resolveCommand, recommendStrategy, isPrecisionCompatible, pickDefaultHardware, pdFitsSingleNode } from "@/lib/command-synthesis";

// Advanced tuning presets — optional tunable flags the user can opt into.
// (vLLM defaults like chunked prefill, prefix caching, CUDA graphs, async
// scheduling are already on — no need to surface them here.)
const ADVANCED_OPTIONS = [
  {
    id: "max_batched_8k",
    label: "max-num-batched-tokens = 8192",
    description: "Tunable batch budget; 8192 is a common sweet spot",
    args: ["--max-num-batched-tokens", "8192"],
  },
  {
    id: "max_num_seqs_256",
    label: "max-num-seqs = 256",
    description: "Max concurrent sequences per batch; lower for latency, higher for throughput",
    args: ["--max-num-seqs", "256"],
  },
  {
    id: "gpu_mem_095",
    label: "gpu-memory-utilization = 0.95",
    description: "Push KV cache further; use with caution on shared GPUs",
    args: ["--gpu-memory-utilization", "0.95"],
  },
  {
    id: "max_model_len_auto",
    label: "max-model-len = auto",
    description: "Auto-size context window to what KV cache can hold on your hardware",
    args: ["--max-model-len", "auto"],
  },
];
const ADVANCED_BY_ID = Object.fromEntries(ADVANCED_OPTIONS.map((o) => [o.id, o]));
import { loadPreferences, savePreference } from "@/lib/preferences";

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
          : "bg-foreground/10 text-foreground/60 hover:bg-foreground/15 hover:text-foreground/90"
      } ${className}`}
    >
      {copied ? <><Check size={12} /> Copied</> : <><Copy size={12} /> Copy</>}
    </button>
  );
}

function PopoverButton({ label, code, icon: Icon }) {
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
        className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-medium bg-foreground/5 text-foreground/60 hover:bg-foreground/10 hover:text-foreground/90 transition-colors"
      >
        <Icon size={11} />
        {label}
      </button>
      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
          <div className="absolute right-0 top-full mt-2 z-50 w-[440px] max-w-[90vw] rounded-xl border border-border bg-card shadow-xl overflow-hidden">
            <div className="flex items-center justify-between px-3 py-2 border-b border-border bg-muted/30">
              <span className="text-xs font-semibold flex items-center gap-1.5">
                <Icon size={12} /> {label}
              </span>
              <button
                onClick={handleCopy}
                className={`text-[11px] flex items-center gap-1 px-2 py-0.5 rounded transition-colors ${
                  copied ? "text-green-600 dark:text-green-400" : "text-muted-foreground hover:text-foreground"
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

  // ── State ──
  const [variant, setVariant] = useState(searchParams.get("variant") || "default");

  // Compute default hardware: URL param > stored preference (if compatible) > smallest compatible profile
  // Smart default hardware: picks a profile compatible with the URL's variant
  // (respecting both VRAM and precision constraints — e.g., NVFP4 → B200).
  const defaultHw = useMemo(() => {
    const urlVariant = searchParams.get("variant") || "default";
    const v = recipe.variants?.[urlVariant] || recipe.variants?.default || {};
    return pickDefaultHardware(taxonomy.hardware_profiles, v);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recipe, taxonomy]);

  const [hwId, setHwId] = useState(searchParams.get("hardware") || defaultHw);

  // After mount: restore an NVIDIA hardware preference from localStorage.
  // (AMD is opt-in per session, never the page-load default — H200 stays canonical.)
  useEffect(() => {
    if (!searchParams.get("hardware")) {
      const prefs = loadPreferences();
      if (prefs.hardware) {
        const v = recipe.variants?.[variant] || recipe.variants?.default || {};
        const prefProfile = taxonomy.hardware_profiles?.[prefs.hardware];
        if (prefProfile?.brand === "NVIDIA" && isPrecisionCompatible(prefProfile, v)) {
          setHwId(prefs.hardware);
        }
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Recipes without a multi_node_* / pd_cluster strategy can't scale beyond one
  // node — force nodeCount to 1 in that case, even if the URL says otherwise.
  const supportsMultiNode = (recipe.compatible_strategies || []).some(
    (s) => s.startsWith("multi_node_") || s === "pd_cluster"
  );
  const [nodeCount, setNodeCount] = useState(() => {
    const n = parseInt(searchParams.get("nodes") || "1", 10);
    if (!supportsMultiNode) return 1;
    return [1, 2].includes(n) ? n : 1;
  });
  const [strategyOverride, setStrategyOverride] = useState(searchParams.get("strategy") || "");
  const [features, setFeatures] = useState(() => {
    const fp = searchParams.get("features");
    if (fp) return fp.split(",").filter(Boolean);
    return Object.keys(recipe.features || {}).filter((f) => !(recipe.opt_in_features || []).includes(f));
  });

  // Advanced tuning flags (defaults off) — toggled independently from features
  const [advanced, setAdvanced] = useState(() => {
    const ap = searchParams.get("advanced");
    return ap ? ap.split(",").filter(Boolean) : [];
  });

  // ── Derived ──
  const currentVariant = recipe.variants?.[variant] || recipe.variants?.default || {};

  // All hardware profiles grouped by brand, sorted by architectural generation
  // within brand (oldest → newest; matches the semianalysis GPU timeline).
  const hwByBrand = useMemo(() => {
    const NVIDIA_ORDER = ["h100", "h200", "b200", "gb200", "b300", "gb300"];
    const AMD_ORDER = ["mi325x", "mi355x"];
    const rankIn = (list, id) => {
      const i = list.indexOf(id);
      return i === -1 ? 9999 : i;
    };
    const groups = {};
    for (const [id, p] of Object.entries(taxonomy.hardware_profiles || {})) {
      const brand = p.brand || "Other";
      if (!groups[brand]) groups[brand] = [];
      groups[brand].push([id, p]);
    }
    for (const [brand, profiles] of Object.entries(groups)) {
      const list = brand === "NVIDIA" ? NVIDIA_ORDER : brand === "AMD" ? AMD_ORDER : [];
      profiles.sort((a, b) => {
        const ra = rankIn(list, a[0]);
        const rb = rankIn(list, b[0]);
        if (ra !== rb) return ra - rb;
        return (a[1].vram_gb || 0) - (b[1].vram_gb || 0);
      });
    }
    const brandOrder = ["NVIDIA", "AMD"];
    return Object.entries(groups).sort(
      ([a], [b]) => {
        const ai = brandOrder.indexOf(a);
        const bi = brandOrder.indexOf(b);
        if (ai === -1 && bi === -1) return a.localeCompare(b);
        if (ai === -1) return 1;
        if (bi === -1) return -1;
        return ai - bi;
      }
    );
  }, [taxonomy]);

  const hwProfile = taxonomy.hardware_profiles?.[hwId] || {};

  const recommended = useMemo(() => recommendStrategy(recipe, hwProfile, nodeCount), [recipe, hwProfile, nodeCount]);

  // If the overridden strategy is pd_cluster in single-node mode but the
  // current hardware can't hold 2× the model, silently fall back to the
  // recommended strategy so the rendered command is always runnable.
  const strategyOverrideValid = useMemo(() => {
    if (!strategyOverride) return true;
    if (strategyOverride === "pd_cluster" && nodeCount === 1) {
      return pdFitsSingleNode(hwProfile, currentVariant);
    }
    return true;
  }, [strategyOverride, nodeCount, hwProfile, currentVariant]);
  const activeStrategy = strategyOverrideValid ? (strategyOverride || recommended) : recommended;

  const compatibleStrategies = useMemo(() => {
    return (recipe.compatible_strategies || []).filter((s) => {
      const strat = strategies[s];
      if (!strat) return false;
      if (nodeCount === 1 && strat.deploy_type === "multi_node") return false;
      if (nodeCount > 1 && strat.deploy_type === "single_node") return false;
      return true;
    });
  }, [recipe, strategies, nodeCount]);

  const result = useMemo(
    () => {
      const advArgs = advanced.flatMap((id) => ADVANCED_BY_ID[id]?.args || []);
      return resolveCommand(recipe, variant, activeStrategy, hwId, features, strategies, taxonomy, advArgs, nodeCount);
    },
    [recipe, variant, activeStrategy, hwId, features, advanced, strategies, taxonomy, nodeCount]
  );

  // Visual feedback when any rendered command changes. Covers single-node
  // (result.command), multi-node (headCommand), and pd_cluster (prefill.command).
  const commandFingerprint = result.command
    || result.headCommand
    || result.prefill?.command
    || "";
  const [changed, setChanged] = useState(false);
  useEffect(() => {
    setChanged(true);
    const t = setTimeout(() => setChanged(false), 600);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [commandFingerprint]);

  // ── URL sync ──
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
    [searchParams, router, recommended, pathname]
  );

  // ── Handlers ──
  const selectVariant = (key) => {
    setVariant(key);
    syncUrl({ variant: key });
    // Only swap hardware when precision demands it (e.g. NVFP4 needs Blackwell).
    // VRAM is not a blocker because multi-node TP/DP can always supply more.
    const v = recipe.variants?.[key] || {};
    const currentProfile = taxonomy.hardware_profiles?.[hwId] || {};
    if (!isPrecisionCompatible(currentProfile, v)) {
      const next = pickDefaultHardware(taxonomy.hardware_profiles, v);
      setHwId(next);
      syncUrl({ hardware: next });
    }
  };

  const selectHardware = (id) => {
    setHwId(id);
    setStrategyOverride("");
    syncUrl({ hardware: id, strategy: "" });
    savePreference("hardware", id);
  };

  const selectStrategy = (s) => {
    setStrategyOverride(s);
    syncUrl({ strategy: s });
  };

  const selectNodes = (n) => {
    setNodeCount(n);
    setStrategyOverride("");
    syncUrl({ nodes: n === 1 ? "" : String(n), strategy: "" });
  };

  const toggleFeature = (f) => {
    const next = features.includes(f) ? features.filter((x) => x !== f) : [...features, f];
    setFeatures(next);
    syncUrl({ features: next.length > 0 ? next.join(",") : "" });
  };

  const toggleAdvanced = (id) => {
    const next = advanced.includes(id) ? advanced.filter((x) => x !== id) : [...advanced, id];
    setAdvanced(next);
    syncUrl({ advanced: next.length > 0 ? next.join(",") : "" });
  };

  const isPd = result.deployType === "pd_cluster";
  const isMultiNode = result.deployType === "multi_node";
  const modelId = recipe.variants?.[variant]?.model_id || recipe.model?.model_id || "model";

  // PD clients hit the router (port 30000), everyone else hits `vllm serve` on 8000.
  const clientPort = isPd ? 30000 : 8000;

  const verifyCmd = `curl http://localhost:${clientPort}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${modelId}",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 32
  }'`;

  const benchCmd = `vllm bench serve \\
  --model ${modelId} \\
  --host localhost \\
  --port ${clientPort} \\
  --dataset-name random \\
  --random-input-len 1024 \\
  --random-output-len 1024 \\
  --num-prompts 100 \\
  --max-concurrency 32`;

  const dependencies = recipe.dependencies || [];

  // Omni models are served via vLLM-Omni (offline Python inference), not `vllm serve`.
  // Skip the command/strategy/feature UI and just show install deps + a pointer to the guide.
  const isOmni = (recipe.meta?.tasks || []).includes("omni");
  if (isOmni) {
    return (
      <div className="space-y-4">
        {dependencies.length > 0 && <DependenciesBlock deps={dependencies} />}
        <div className="rounded-2xl border border-border bg-muted/20 px-5 py-4 text-sm">
          <div className="font-medium mb-1 flex items-center gap-2">
            <Sparkles size={14} className="text-vllm-yellow" />
            Served via vLLM-Omni (offline inference)
          </div>
          <p className="text-muted-foreground text-xs leading-relaxed">
            This model runs as an offline Python workflow, not a long-running <code className="font-mono text-[11px] px-1 py-0.5 rounded bg-foreground/5">vllm serve</code> endpoint.
            See the <strong>Guide</strong> below for the exact inference script and parameters.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* ── Dependencies / extra install ── */}
      {dependencies.length > 0 && <DependenciesBlock deps={dependencies} />}

      {/* ── Command output ── */}
      <div
        className={`rounded-2xl overflow-hidden bg-[var(--command-bg)] border border-border transition-shadow ${
          changed ? "ring-2 ring-vllm-blue/30" : ""
        }`}
      >
        {isPd ? (
          <PdClusterBlock result={result} verifyCmd={verifyCmd} benchCmd={benchCmd} />
        ) : isMultiNode ? (
          <MultiNodeBlock result={result} verifyCmd={verifyCmd} benchCmd={benchCmd} />
        ) : (
          <div>
            <div className="flex items-center justify-between px-4 pt-3">
              <span className="text-[11px] text-[var(--command-fg)]/50 font-mono">vllm serve</span>
              <div className="flex items-center gap-1.5">
                <CopyButton text={result.command} />
                <PopoverButton label="Verify" code={verifyCmd} icon={Terminal} />
                <PopoverButton label="Bench" code={benchCmd} icon={Gauge} />
              </div>
            </div>
            <pre className="px-4 py-3 text-[13px] text-[var(--command-fg)] font-mono leading-relaxed whitespace-pre overflow-x-auto">
              {result.command}
            </pre>
          </div>
        )}
      </div>

      {/* ── Configuration ── */}
      <div className="rounded-xl border border-border divide-y divide-border">
        {/* Hardware (first — user's fixed constraint, grouped by brand) */}
        <ConfigRow label="Hardware">
          <div className="space-y-1.5">
            {hwByBrand.map(([brand, profiles]) => (
              <div key={brand} className="flex flex-wrap items-center gap-2">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/70 w-14 shrink-0">
                  {brand}
                </span>
                <PillGroup>
                  {profiles.map(([id, p]) => {
                    const precisionOk = isPrecisionCompatible(p, currentVariant);
                    // When single-node PD is the active strategy, the hardware must fit
                    // 2× the model (prefill + decode pools share one node).
                    const pdSingleNodeCheck = activeStrategy === "pd_cluster" && nodeCount === 1;
                    const pdOk = !pdSingleNodeCheck || pdFitsSingleNode(p, currentVariant);
                    const disabled = !precisionOk || !pdOk;
                    const reason = !precisionOk
                      ? `${currentVariant.precision?.toUpperCase()} requires NVIDIA Blackwell`
                      : !pdOk
                      ? `Single-node PD needs 2× model VRAM (${2 * (currentVariant.vram_minimum_gb || 0)} GB). Switch to Multi-node or pick a larger GPU.`
                      : p.description;
                    return (
                      <Pill
                        key={id}
                        active={hwId === id}
                        disabled={disabled}
                        onClick={() => !disabled && selectHardware(id)}
                        title={reason}
                      >
                        <span className="font-semibold">{p.display_name}</span>
                        {p.vram_gb > 0 && p.gpu_count > 0 && (
                          <span className="text-muted-foreground ml-1.5 font-mono">
                            {p.gpu_count}×{Math.round(p.vram_gb / p.gpu_count)}G
                          </span>
                        )}
                      </Pill>
                    );
                  })}
                </PillGroup>
              </div>
            ))}
          </div>
        </ConfigRow>

        {/* Variant */}
        <ConfigRow label="Variant">
          <PillGroup>
            {Object.entries(recipe.variants || {}).map(([key, v]) => (
              <Pill
                key={key}
                active={variant === key}
                onClick={() => selectVariant(key)}
                title={`Min ${v.vram_minimum_gb} GB total VRAM — scale out via multi-node if needed`}
              >
                <span className="font-mono font-semibold">{v.precision?.toUpperCase()}</span>
                <span className="text-muted-foreground ml-1.5 font-mono">{v.vram_minimum_gb} GB</span>
              </Pill>
            ))}
          </PillGroup>
        </ConfigRow>

        {/* Strategy */}
        <ConfigRow label="Strategy">
          <PillGroup>
            {compatibleStrategies.map((s) => {
              const isPdSingleNode = s === "pd_cluster" && nodeCount === 1;
              const pdBlocked = isPdSingleNode && !pdFitsSingleNode(hwProfile, currentVariant);
              const disabled = pdBlocked;
              return (
                <Pill
                  key={s}
                  active={activeStrategy === s}
                  disabled={disabled}
                  onClick={() => !disabled && selectStrategy(s)}
                  title={
                    pdBlocked
                      ? `Single-node PD needs 2× model VRAM (${2 * (currentVariant.vram_minimum_gb || 0)} GB). Switch to Multi-node or a larger GPU.`
                      : strategies[s]?.description
                  }
                >
                  <span className="font-semibold">{strategies[s]?.display_name || s}</span>
                  {s === recommended && !disabled && (
                    <Sparkles size={10} className="text-vllm-yellow ml-1" />
                  )}
                </Pill>
              );
            })}
          </PillGroup>
          {strategies[activeStrategy]?.description && (
            <p className="text-[11px] text-muted-foreground mt-2 leading-snug">
              {strategies[activeStrategy].description.split("\n")[0]}
            </p>
          )}
        </ConfigRow>

        {/* Nodes */}
        <ConfigRow label="Nodes">
          <PillGroup>
            {[1, 2].map((n) => {
              // Multi-node pill is disabled when the recipe declares no
              // multi_node_* (or pd_cluster) strategy. Small dense models
              // commonly omit these.
              const disabled = n > 1 && !supportsMultiNode;
              return (
                <Pill
                  key={n}
                  active={nodeCount === n}
                  disabled={disabled}
                  onClick={() => !disabled && selectNodes(n)}
                  title={
                    disabled
                      ? "This recipe does not declare a multi-node strategy. Fits in a single node."
                      : n === 1
                      ? "Single-node deployment (one HGX box)"
                      : `2 nodes × ${hwProfile.gpu_count || 8} GPUs = ${2 * (hwProfile.gpu_count || 8)} GPUs total. Scale further by replicating the worker command with higher --node-rank / --data-parallel-start-rank.`
                  }
                >
                  <span className="font-semibold">{n === 1 ? "Single node" : "Multi-node (example: 2)"}</span>
                  {n > 1 && !disabled && (
                    <span className="text-muted-foreground ml-1.5 font-mono">
                      {2 * (hwProfile.gpu_count || 8)}×GPU
                    </span>
                  )}
                </Pill>
              );
            })}
          </PillGroup>
        </ConfigRow>

        {/* Features */}
        {Object.keys(recipe.features || {}).length > 0 && (
          <ConfigRow label="Features">
            <PillGroup>
              {Object.entries(recipe.features || {}).map(([key]) => (
                <Pill
                  key={key}
                  active={features.includes(key)}
                  onClick={() => toggleFeature(key)}
                >
                  {key.replace(/_/g, " ")}
                </Pill>
              ))}
            </PillGroup>
          </ConfigRow>
        )}

        {/* Advanced (collapsed by default) */}
        <details className="group">
          <summary className="px-4 py-3 cursor-pointer text-[10px] font-semibold text-muted-foreground uppercase tracking-widest hover:bg-muted/30 transition-colors flex items-center gap-2 select-none list-none">
            <span>Advanced</span>
            <span className="text-muted-foreground/50 normal-case tracking-normal font-normal">
              {advanced.length > 0 ? `(${advanced.length} enabled)` : "— performance tuning"}
            </span>
            <ChevronDown size={12} className="ml-auto group-open:rotate-180 transition-transform" />
          </summary>
          <div className="px-4 pb-4 pt-1 border-t border-border/60">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {ADVANCED_OPTIONS.map((opt) => (
                <label
                  key={opt.id}
                  className={`flex items-start gap-2.5 p-2 rounded-lg border cursor-pointer transition-colors ${
                    advanced.includes(opt.id)
                      ? "border-vllm-blue/40 bg-vllm-blue/5"
                      : "border-border hover:bg-muted/30"
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={advanced.includes(opt.id)}
                    onChange={() => toggleAdvanced(opt.id)}
                    className="accent-vllm-blue mt-0.5"
                  />
                  <div className="min-w-0 flex-1">
                    <div className="text-xs font-medium">{opt.label}</div>
                    <div className="text-[10px] text-muted-foreground mt-0.5 leading-snug">{opt.description}</div>
                  </div>
                </label>
              ))}
            </div>
          </div>
        </details>
      </div>
    </div>
  );
}

// ── Sub-components ──

function ConfigRow({ label, children }) {
  return (
    <div className="px-4 py-3 flex flex-col sm:flex-row sm:items-start gap-2 sm:gap-4">
      <div className="text-[10px] font-semibold text-muted-foreground uppercase tracking-widest sm:w-20 sm:pt-1.5 shrink-0">
        {label}
      </div>
      <div className="flex-1 min-w-0">{children}</div>
    </div>
  );
}

function PillGroup({ children }) {
  return <div className="flex flex-wrap gap-1.5">{children}</div>;
}

function Pill({ active, onClick, title, dimmed, disabled, children }) {
  // disabled takes precedence over active — an "active but disabled" pill should
  // clearly look disabled (e.g. PD that was pre-selected but no longer fits).
  const style = disabled
    ? "border-dashed border-border/40 text-muted-foreground/30 cursor-not-allowed bg-muted/20 line-through decoration-muted-foreground/30"
    : active
    ? "border-vllm-blue bg-vllm-blue/5 text-foreground ring-1 ring-vllm-blue/20 shadow-sm"
    : dimmed
    ? "border-dashed border-border/60 text-muted-foreground/50 hover:text-muted-foreground hover:border-muted-foreground/30"
    : "border-border text-muted-foreground hover:text-foreground hover:border-muted-foreground/40 hover:bg-muted/30";
  return (
    <button
      onClick={onClick}
      title={title}
      disabled={disabled}
      aria-disabled={disabled}
      className={`inline-flex items-center rounded-lg border px-2.5 py-1.5 text-xs transition-all ${style}`}
    >
      {children}
    </button>
  );
}

function DependenciesBlock({ deps }) {
  const allCommands = deps.map((d) => d.command).join("\n");
  const requiredCount = deps.filter((d) => !d.optional).length;
  const optionalCount = deps.length - requiredCount;
  return (
    <div className="rounded-2xl overflow-hidden bg-[var(--command-bg)] border border-border">
      <div className="flex items-center justify-between px-4 pt-3">
        <span className="text-[11px] text-[var(--command-fg)]/50 font-mono inline-flex items-center gap-1.5">
          <Package size={11} /> extra install
          {requiredCount > 0 && <span className="text-[var(--command-fg)]/40">· {requiredCount} required</span>}
          {optionalCount > 0 && <span className="text-[var(--command-fg)]/40">· {optionalCount} optional</span>}
        </span>
        <CopyButton text={allCommands} />
      </div>
      <div className="px-4 py-3 text-[13px] font-mono leading-relaxed overflow-x-auto space-y-2">
        {deps.map((d, i) => (
          <div key={i}>
            {d.note && (
              <div className="text-[var(--command-fg)]/45 text-[11px] leading-snug mb-0.5">
                # {d.note}{d.optional ? " (optional)" : ""}
              </div>
            )}
            <div className="text-[var(--command-fg)] whitespace-pre">{d.command}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function MultiNodeBlock({ result, verifyCmd, benchCmd }) {
  const [tab, setTab] = useState("head");
  const tabs = [
    { id: "head", label: "Head", command: result.headCommand },
    { id: "worker", label: "Node 1", command: result.workerCommand },
  ];
  const active = tabs.find((t) => t.id === tab) || tabs[0];
  return (
    <div>
      <div className="flex items-center justify-between px-4 pt-3">
        <div className="flex gap-0.5 bg-foreground/5 rounded-md p-0.5">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-2.5 py-1 text-xs font-medium rounded transition-colors ${
                tab === t.id ? "bg-foreground/10 text-[var(--command-fg)]" : "text-[var(--command-fg)]/50 hover:text-[var(--command-fg)]/80"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-1.5">
          <CopyButton text={active.command} />
          {tab === "head" && (
            <>
              <PopoverButton label="Verify" code={verifyCmd} icon={Terminal} />
              <PopoverButton label="Bench" code={benchCmd} icon={Gauge} />
            </>
          )}
        </div>
      </div>
      <pre className="px-4 py-3 text-[13px] text-[var(--command-fg)] font-mono leading-relaxed whitespace-pre overflow-x-auto">
        {active.command}
      </pre>
      <div className="px-4 pb-3 text-[11px] text-[var(--command-fg)]/45 font-mono leading-snug">
        # Set $HEAD_IP to the rank-0 node's IP before launch. Scale to N nodes by replicating
        # this worker command with --node-rank = i (TP/TEP) or --data-parallel-start-rank = i × local_gpus (DEP).
      </div>
    </div>
  );
}

function PdClusterBlock({ result, verifyCmd, benchCmd }) {
  // Tabs: Prefill · Decode · Router (same shape whether single-node or multi-node;
  // only the CUDA_VISIBLE_DEVICES split and TP size differ).
  const [tab, setTab] = useState("prefill");
  const tabs = [
    { id: "prefill", label: "Prefill", command: result.prefill.command, env: result.prefill.env },
    { id: "decode",  label: "Decode",  command: result.decode.command,  env: result.decode.env },
    { id: "router",  label: "Router",  command: result.router.command, env: {}, install: result.router.install, isRouter: true },
  ];
  const active = tabs.find((t) => t.id === tab) || tabs[0];
  const envLines = Object.entries(active.env || {}).map(([k, v]) => `export ${k}=${v}`).join("\n");
  const fullScript = envLines ? `${envLines}\n\n${active.command}` : active.command;
  return (
    <div>
      <div className="flex items-center justify-between px-4 pt-3 gap-3">
        <div className="flex flex-wrap gap-0.5 bg-foreground/5 rounded-md p-0.5">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-2.5 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap ${
                tab === t.id ? "bg-foreground/10 text-[var(--command-fg)]" : "text-[var(--command-fg)]/50 hover:text-[var(--command-fg)]/80"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          <CopyButton text={fullScript} />
          {active.isRouter && (
            <>
              <PopoverButton label="Verify" code={verifyCmd} icon={Terminal} />
              <PopoverButton label="Bench" code={benchCmd} icon={Gauge} />
            </>
          )}
        </div>
      </div>
      {active.isRouter && active.install && (
        <div className="px-4 pt-3 text-[11px] text-[var(--command-fg)]/50 font-mono leading-snug">
          # Dependency: {active.install}
        </div>
      )}
      {envLines && (
        <pre className="px-4 pt-3 pb-1 text-[12px] text-[var(--command-fg)]/70 font-mono leading-relaxed whitespace-pre overflow-x-auto">
          {envLines}
        </pre>
      )}
      <pre className="px-4 py-3 text-[13px] text-[var(--command-fg)] font-mono leading-relaxed whitespace-pre overflow-x-auto">
        {active.command}
      </pre>
    </div>
  );
}
