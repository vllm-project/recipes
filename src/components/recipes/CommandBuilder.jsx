"use client";

import { useState, useMemo, useCallback, useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { useSearchParams, useRouter, usePathname } from "next/navigation";
import { Copy, Check, Terminal, Gauge, Sparkles, ChevronDown, Package, Info, Zap, Globe, Wrench, Brain } from "lucide-react";
import { resolveCommand, recommendStrategy, isPrecisionCompatible, isHardwareSupported, isVariantHardwareSupported, fitsSingleNode, isHardwareScalable, isKvStoreBrandSupported, variantRunsOnHardware, pickFittingVariant, pickDefaultHardware, resolveSingleNodeTp, computeDockerMeta, buildDockerRun, buildAscendDockerRun, resolveOmniCommand, pdPoolModes, defaultModeFor, isModeSupported, isModeAllowedForVariant, resolveModeKey, isFeatureAllowedForStrategy, isKvOffloadAllowedForStrategy } from "@/lib/command-synthesis";
import { resolveOmniTasks } from "@/lib/omni-tasks";
import { TooltipProvider, InfoTip } from "@/components/ui/tooltip";
import { detectPlaceholdersAll, substitute, substituteEnv, loadEndpoints, saveEndpoint, clearEndpoints } from "@/lib/cluster-endpoints";

// Advanced tuning presets — optional tunable flags the user can opt into.
// (vLLM defaults like chunked prefill, prefix caching, CUDA graphs, async
// scheduling are already on — no need to surface them here.)
// `gatedBy(recipe, activeStrategy)` hides an option when the recipe or current
// strategy can't support it. This keeps casual users from selecting invalid
// combos (e.g. EP on a dense model) while still exposing parallelism knobs.
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
  {
    id: "dcp_8",
    label: "decode-context-parallel-size = 8",
    description:
      "Shard the KV cache at decode time across TP ranks. MLA-attention models only (DeepSeek, Kimi-K2). Max DCP = tensor-parallel-size ÷ num_kv_heads.",
    args: ["--decode-context-parallel-size", "8"],
    gatedBy: (recipe) => recipe?.model?.supports_dcp === true,
  },
  {
    id: "no_flashinfer_autotune",
    label: "no-enable-flashinfer-autotune",
    description: "Skip FlashInfer autotuning at startup (faster startup, may lose autotuned perf)",
    args: ["--no-enable-flashinfer-autotune"],
    gatedBy: (recipe) => recipe?.model?.flashinfer_autotune === true,
  },
  {
    id: "ep_weight_filter",
    label: "enable-ep-weight-filter",
    description: "Skip loading expert weights that don't belong to this EP rank — speeds up weight loading for large MoE models",
    args: ["--enable-ep-weight-filter"],
    gatedBy: (_recipe, activeStrategy) => /(?:^|_)(?:tep|dep)$/.test(activeStrategy || ""),
  },
];
import { loadPreferences, savePreference, loadRecipeState, saveRecipeState } from "@/lib/preferences";

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
      className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${copied
          ? "bg-green-500/20 text-green-600 dark:text-green-400"
          : "bg-foreground/10 text-foreground/60 hover:bg-foreground/15 hover:text-foreground/90"
        } ${className}`}
    >
      {copied ? <><Check size={12} /> Copied</> : <><Copy size={12} /> Copy</>}
    </button>
  );
}

function PopoverButton({ label, code, icon: Icon, disabled, disabledNote }) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [rect, setRect] = useState(null);
  const btnRef = useRef(null);

  // When the popover opens, measure the button position so we can place the
  // popover at a fixed viewport coordinate — the parent command card uses
  // `overflow-hidden` for rounded-corner clipping, which would otherwise
  // crop an absolute-positioned popover.
  useEffect(() => {
    if (!open) return;
    const update = () => {
      if (btnRef.current) setRect(btnRef.current.getBoundingClientRect());
    };
    update();
    const onKey = (e) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("resize", update);
    window.addEventListener("scroll", update, true);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("resize", update);
      window.removeEventListener("scroll", update, true);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Disabled rendering (e.g. cURL/Bench on a tab clients don't connect to):
  // the button stays visible so the reader learns WHERE these actions live
  // instead of watching them vanish; the tooltip names the right tab.
  // (After the hooks above — `disabled` flips as the user switches tabs, and
  // an earlier return would change the hook order between renders.)
  if (disabled) {
    const chip = (
      <span
        tabIndex={0}
        aria-disabled
        aria-label={disabledNote || label}
        className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-medium bg-foreground/5 text-[var(--command-fg)]/25 cursor-not-allowed select-none"
      >
        <Icon size={11} />
        {label}
      </span>
    );
    return disabledNote ? <InfoTip content={disabledNote}>{chip}</InfoTip> : chip;
  }

  const popover = open && rect && typeof document !== "undefined" ? createPortal(
    <>
      <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
      <div
        className="fixed z-50 w-[440px] max-w-[90vw] rounded-xl border border-border bg-card shadow-xl overflow-hidden"
        style={{ top: rect.bottom + 8, left: Math.max(8, rect.right - 440) }}
      >
        <div className="flex items-center justify-between px-3 py-2 border-b border-border bg-muted/30">
          <span className="text-xs font-semibold flex items-center gap-1.5">
            <Icon size={12} /> {label}
          </span>
          <button
            onClick={handleCopy}
            className={`text-[11px] flex items-center gap-1 px-2 py-0.5 rounded transition-colors ${copied ? "text-green-600 dark:text-green-400" : "text-muted-foreground hover:text-foreground"
              }`}
          >
            {copied ? <><Check size={10} /> Copied</> : <><Copy size={10} /> Copy</>}
          </button>
        </div>
        <pre className="px-3 py-2.5 text-xs font-mono leading-relaxed whitespace-pre overflow-x-auto bg-[var(--code-block-bg)] text-[var(--code-block-fg)]">
          {code}
        </pre>
      </div>
    </>,
    document.body
  ) : null;

  return (
    <>
      <button
        ref={btnRef}
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        aria-haspopup="dialog"
        className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-medium bg-foreground/5 text-foreground/60 hover:bg-foreground/10 hover:text-foreground/90 transition-colors"
      >
        <Icon size={11} />
        {label}
      </button>
      {popover}
    </>
  );
}

// Same shell as PopoverButton (portal + click-outside + Escape), but the body
// is a form for editing $VAR / NODE_N substitutions. Lives on each command
// block header next to cURL/Bench so the user finds it where they realize
// "this curl is hitting localhost — I need to point it elsewhere."
function EndpointsPopoverButton({ isPd, isKvStore, isMultiNode, placeholders, endpoints, onChange, onReset }) {
  const [open, setOpen] = useState(false);
  const [rect, setRect] = useState(null);
  const btnRef = useRef(null);

  useEffect(() => {
    if (!open) return;
    const update = () => {
      if (btnRef.current) setRect(btnRef.current.getBoundingClientRect());
    };
    update();
    const onKey = (e) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("resize", update);
    window.addEventListener("scroll", update, true);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("resize", update);
      window.removeEventListener("scroll", update, true);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const clientHostKey = (isPd || isKvStore) ? "ROUTER_HOST" : (isMultiNode ? "HEAD_IP" : "VLLM_HOST");
  const clientPortKey = (isPd || isKvStore) ? "ROUTER_PORT" : "VLLM_PORT";
  const clientHostHint = "localhost";
  const clientPortHint = isPd ? "30000" : isKvStore ? "30080" : "8000";

  const extras = placeholders.filter(
    (p) => !(p.kind === "var" && (p.name === clientHostKey || p.name === clientPortKey)),
  );
  const filledCount = Object.keys(endpoints).length;

  const popover = open && rect && typeof document !== "undefined" ? createPortal(
    <>
      <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
      <div
        className="fixed z-50 w-[480px] max-w-[92vw] rounded-xl border border-border bg-card shadow-xl overflow-hidden"
        style={{ top: rect.bottom + 8, left: Math.max(8, rect.right - 480) }}
      >
        <div className="flex items-center justify-between px-3 py-2 border-b border-border bg-muted/30">
          <span className="text-xs font-semibold flex items-center gap-1.5">
            <Globe size={12} /> Cluster env
          </span>
          {filledCount > 0 && (
            <button
              type="button"
              onClick={onReset}
              className="text-[11px] text-muted-foreground hover:text-foreground transition-colors"
            >
              Clear all
            </button>
          )}
        </div>
        <div className="px-3 py-3 space-y-3 max-h-[60vh] overflow-y-auto">
          <div className="text-[11px] text-muted-foreground leading-snug">
            Saved in your browser and substituted into the rendered command, env exports, curl, and bench. Empty fields stay as <code className="font-mono text-[10px] px-1 py-px rounded bg-foreground/5">$VAR</code>.
          </div>
          <div>
            <div className="text-[10px] font-semibold text-muted-foreground uppercase tracking-widest mb-1.5">
              Curl / bench target
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              <EndpointInput
                label={`$${clientHostKey}`}
                hint={clientHostHint}
                value={endpoints[clientHostKey] || ""}
                onChange={(v) => onChange(clientHostKey, v)}
              />
              <EndpointInput
                label={`$${clientPortKey}`}
                hint={clientPortHint}
                value={endpoints[clientPortKey] || ""}
                onChange={(v) => onChange(clientPortKey, v)}
              />
            </div>
          </div>
          {extras.length > 0 && (
            <div>
              <div className="text-[10px] font-semibold text-muted-foreground uppercase tracking-widest mb-1.5">
                Placeholders in current commands
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {extras.map((p) => (
                  <EndpointInput
                    key={`${p.kind}:${p.name}`}
                    label={p.label}
                    hint={endpointHintFor(p.name)}
                    value={endpoints[p.name] || ""}
                    onChange={(v) => onChange(p.name, v)}
                  />
                ))}
              </div>
              {extras.some((p) => p.name === "IFACE_NAME") && (
                <div className="mt-2 text-[11px] text-muted-foreground leading-snug">
                  Tip: find your inter-node fabric NIC with{" "}
                  <code className="font-mono text-[10px] px-1 py-px rounded bg-foreground/5">
                    ip -o -4 route show to default | awk '{"{print $5; exit}"}'
                  </code>
                  . On HPC clusters, verify against{" "}
                  <code className="font-mono text-[10px] px-1 py-px rounded bg-foreground/5">ibstat</code>
                  {" "}/{" "}
                  <code className="font-mono text-[10px] px-1 py-px rounded bg-foreground/5">ibv_devinfo</code>
                  {" "}— the default route is often the slow management NIC, not the RDMA fabric.
                </div>
              )}
              {extras.some((p) => p.name === "HEAD_IP") && (
                <div className="mt-2 text-[11px] text-muted-foreground leading-snug">
                  Tip: on the rank-0 node, read its IP on that NIC with{" "}
                  <code className="font-mono text-[10px] px-1 py-px rounded bg-foreground/5">
                    ip -o -4 addr show $IFACE_NAME | awk '{"{print $4; exit}"}' | cut -d/ -f1
                  </code>
                  {" "}— make sure every other rank can reach it on that interface.
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </>,
    document.body,
  ) : null;

  return (
    <>
      <button
        ref={btnRef}
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        aria-haspopup="dialog"
        title="Cluster env"
        className={`inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-medium transition-colors ${
          filledCount > 0
            ? "bg-vllm-blue/10 text-vllm-blue hover:bg-vllm-blue/15"
            : "bg-foreground/5 text-foreground/60 hover:bg-foreground/10 hover:text-foreground/90"
        }`}
      >
        <Globe size={11} />
        Env
        {filledCount > 0 && (
          <span className="text-[10px] font-mono tabular-nums opacity-80">{filledCount}</span>
        )}
      </button>
      {popover}
    </>
  );
}

// Framework-level background for the merged Mooncake pill (tooltip + the text
// under the KV Offload row while Mooncake is active). Topology-specific
// wording (embedded vs standalone-store) lives on the Store Topology row.
const MOONCAKE_BACKGROUND =
  "Mooncake is a distributed KV cache layer for LLM serving: vLLM instances share prefix KV blocks through a CPU-DRAM pool coordinated by a lightweight master; at 2+ instances a cache-aware router sends prefix-matched requests to the instance already holding them. Pick the store topology below.";
const MOONCAKE_DOCS_URL =
  "https://docs.vllm.ai/en/stable/features/mooncake_store_connector_usage";
// Where the merged Mooncake pill sorts among taxonomy.kv_offload options
// (their `order` fields are chosen around this): Off · Simple(1) ·
// Mooncake(2) · LMCache(3).
const MOONCAKE_PILL_ORDER = 2;

export function CommandBuilder({ recipe, strategies, taxonomy }) {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();

  // ── Omni tasks ── resolved from the recipe (catalog lookup); declared up here
  // because both the omni and non-omni return paths reference these hooks.
  const omniTasks = useMemo(() => resolveOmniTasks(recipe), [recipe]);

  // ── State ──
  const [variant, setVariant] = useState(searchParams.get("variant") || "default");

  // Active omni task — drives the `vllm serve --omni` model_id swap (Wan2.2's
  // T2V/I2V/TI2V) and the cURL endpoint/body shown in the Try-it popover.
  const [omniTask, setOmniTask] = useState(() => {
    const fromUrl = searchParams.get("task");
    if (fromUrl && omniTasks.some((t) => t.id === fromUrl)) return fromUrl;
    return omniTasks[0]?.id || "";
  });

  // Compute default hardware: URL param > stored preference (if compatible) > smallest compatible profile
  // Smart default hardware: picks a profile compatible with the URL's variant
  // (respecting both VRAM and precision constraints — e.g., NVFP4 → B200).
  const defaultHw = useMemo(() => {
    const urlVariant = searchParams.get("variant") || "default";
    const v = recipe.variants?.[urlVariant] || recipe.variants?.default || {};
    return pickDefaultHardware(taxonomy.hardware_profiles, v, recipe);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recipe, taxonomy]);

  const requestedHwId = searchParams.get("hardware");
  const requestedVariant = recipe.variants?.[searchParams.get("variant") || "default"] || recipe.variants?.default || {};
  const requestedHwProfile = taxonomy.hardware_profiles?.[requestedHwId] || {};
  const requestedHwAllowed = requestedHwId
    && isPrecisionCompatible(requestedHwProfile, requestedVariant)
    && isHardwareSupported(recipe, requestedHwId)
    && isVariantHardwareSupported(requestedVariant, requestedHwId);
  const [hwId, setHwId] = useState(requestedHwAllowed ? requestedHwId : defaultHw);

  // After mount: restore preferences from localStorage in two scopes.
  // URL params always win (explicit > stored).
  //   1. Global (hardware): tracks the user's physical setup, shared across
  //      every recipe. NVIDIA-only — AMD is opt-in per session and H200
  //      stays canonical on first load.
  //   2. Per-recipe (strategy / nodes / features): keyed by hf_id, so each
  //      recipe remembers its own choices independently. Picking TP on
  //      V4-Flash doesn't make V4-Pro default to TP — Pro keeps whatever
  //      was last picked there (or its YAML default if untouched).
  useEffect(() => {
    const prefs = loadPreferences();
    let restoredFitsHw = null;
    // Variant restrictions beat an incompatible hardware query parameter.
    // Normalize the URL immediately so copied links match the rendered state.
    if (requestedHwId && requestedHwId !== hwId) {
      const sp = new URLSearchParams(searchParams.toString());
      sp.set("hardware", hwId);
      router.replace(`?${sp.toString()}`, { scroll: false });
    }
    if (!searchParams.get("hardware") && prefs.hardware) {
      const v = recipe.variants?.[variant] || recipe.variants?.default || {};
      const prefProfile = taxonomy.hardware_profiles?.[prefs.hardware];
      // Mirror the picker filter: a `restricted` profile (DGX Station, TPU)
      // only applies to recipes that explicitly declare it in `meta.hardware`.
      // Without this, selecting DGX on one recipe would leak into the global
      // preference and leave every other recipe with no rendered pill selected.
      const declaredHere = prefs.hardware in (recipe.meta?.hardware || {});
      const restrictedElsewhere = prefProfile?.restricted && !declaredHere;
      if (prefProfile?.brand === "NVIDIA" && !restrictedElsewhere && isPrecisionCompatible(prefProfile, v) && isHardwareSupported(recipe, prefs.hardware) && isVariantHardwareSupported(v, prefs.hardware)) {
        setHwId(prefs.hardware);
        restoredFitsHw = prefProfile;
      }
    }

    const rs = loadRecipeState(recipe.hf_id);
    if (!searchParams.get("strategy") && rs.strategy &&
        (recipe.compatible_strategies || []).includes(rs.strategy) &&
        strategies[rs.strategy]?.deploy_type !== "kv_store_lb") {
      setStrategyOverride(rs.strategy);
    }
    if (!searchParams.get("features") && Array.isArray(rs.features)) {
      const recipeFeatures = Object.keys(recipe.features || {});
      setFeatures(rs.features.filter((f) => recipeFeatures.includes(f)));
    }
    // Resolve the hardware this mount actually settles on (URL > restored pref
    // > default) so the non-scalable fixups below see the right profile.
    const resolvedHwId = restoredFitsHw ? prefs.hardware : hwId;
    const resolvedHw = taxonomy.hardware_profiles?.[resolvedHwId];
    const resolvedScalable = isHardwareScalable(resolvedHw);
    // KV offload restores separately from the serving strategy: "simple" is
    // always valid; a Mooncake pick must also be usable on the hardware this
    // mount settles on (scalable + not opted out) — restoring it onto a
    // non-scalable profile would only trip the downgrade effect and hide the
    // pick until the user returns to supported hardware.
    if (!searchParams.get("kv_offload") && rs.kvOffload &&
        (taxonomy.kv_offload?.[rs.kvOffload] ||
          (strategies[rs.kvOffload]?.deploy_type === "kv_store_lb" &&
            resolvedScalable && isKvStoreBrandSupported(resolvedHw) && isKvStoreSupported(rs.kvOffload, resolvedHwId)))) {
      setKvOffload(rs.kvOffload);
    }
    if (!searchParams.get("kv_instances")) {
      const savedInstances = parseInt(rs.kvInstances, 10);
      if (Number.isFinite(savedInstances) && savedInstances >= 1 && savedInstances <= 16) {
        setKvInstances(savedInstances);
      }
    }
    // Non-scalable hardware can't shard an oversized variant. If we land on one
    // (e.g. ?hardware=dgx_station_gb300, or a restored DGX preference) with a
    // variant that doesn't fit, fall to the largest variant that does. URL
    // ?variant= still wins.
    if (!searchParams.get("variant") && resolvedHw && !resolvedScalable) {
      const v = recipe.variants?.[variant] || recipe.variants?.default || {};
      if (!fitsSingleNode(resolvedHw, v)) {
        const fitting = pickFittingVariant(recipe, resolvedHw, resolvedHwId);
        if (fitting && fitting !== variant) setVariant(fitting);
      }
    }
    // Nodes: prefer the saved value; otherwise auto-bump if the resolved
    // hardware can't fit single-node (mirrors `setHw`'s bump). Non-scalable
    // hardware never bumps — it's locked to one node.
    if (!searchParams.get("nodes") && supportsMultiNode && resolvedScalable) {
      const saved = parseInt(rs.nodes, 10);
      if ([1, 2].includes(saved)) {
        setNodeCount(saved);
      } else if (restoredFitsHw) {
        const v = recipe.variants?.[variant] || recipe.variants?.default || {};
        if (!fitsSingleNode(restoredFitsHw, v)) setNodeCount(2);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Recipes without a multi_node_* / pd_cluster strategy can't scale beyond
  // one node — force nodeCount to 1 in that case, even if the URL says
  // otherwise. (Nodes means nodes PER INSTANCE under Mooncake too; the
  // instance count is a separate axis and doesn't need multi-node support.)
  const supportsMultiNode = (recipe.compatible_strategies || []).some(
    (s) => s.startsWith("multi_node_") || s === "pd_cluster"
  );
  const [nodeCount, setNodeCount] = useState(() => {
    if (!supportsMultiNode) return 1;
    const initialHwId = searchParams.get("hardware") || defaultHw;
    const initialHw = taxonomy.hardware_profiles?.[initialHwId];
    // Non-scalable hardware (single-GPU workstation) is single-node by
    // definition — ignore any ?nodes= pin.
    if (initialHw && !isHardwareScalable(initialHw)) return 1;
    const urlN = searchParams.get("nodes");
    if (urlN) {
      const n = parseInt(urlN, 10);
      return [1, 2].includes(n) ? n : 1;
    }
    // No URL pin: start on multi-node when the initial hardware can't fit
    // single-node. Same fit check the hardware-change handler runs.
    const v = recipe.variants?.[variant] || recipe.variants?.default || {};
    return initialHw && !fitsSingleNode(initialHw, v) ? 2 : 1;
  });
  // PD-specific per-role node counts. Only surfaced when the active strategy
  // is `pd_cluster`; ignored otherwise. Defaults come from the recipe's
  // strategy_overrides block, else 1 for each role. `?prefill_nodes=1&decode_nodes=4`
  // persists in the URL so shares round-trip.
  //
  // `nodes` accepts two shapes: a bare integer (same default for every hw)
  // or an object `{ default, <hwId>… }` for per-hw defaults (e.g. GB200 = 2,
  // GB300 = 1). Unknown hw ids fall through to `default`, then to 1.
  const pdDefaults = useMemo(() => {
    const so = recipe.strategy_overrides?.pd_cluster || {};
    // Each PD role holds a full model copy on its own pool of nodes. If a
    // single node can't fit the model, the role must span >= ceil(model/node)
    // nodes — same logic as the top-level multi-node bump on line 378, applied
    // per role. Recipe-level `strategy_overrides.pd_cluster.<role>.nodes` still
    // wins when set.
    const v = recipe.variants?.[variant] || recipe.variants?.default || {};
    const hw = taxonomy.hardware_profiles?.[hwId];
    const nodeVram = hw?.vram_gb || 0;
    const modelVram = v?.vram_minimum_gb || 0;
    const minNodesPerRole = (modelVram > 0 && nodeVram > 0)
      ? Math.max(1, Math.ceil(modelVram / nodeVram))
      : 1;
    const pickNodes = (role, fallback) => {
      const ov = so[role]?.nodes;
      if (typeof ov === "number") return ov;
      if (ov && typeof ov === "object") {
        if (typeof ov[hwId] === "number") return ov[hwId];
        if (typeof ov.default === "number") return ov.default;
      }
      return fallback;
    };
    return {
      prefill: pickNodes("prefill", minNodesPerRole),
      decode: pickNodes("decode", minNodesPerRole),
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recipe, hwId, variant]);
  const [pdPrefillNodes, setPdPrefillNodes] = useState(() => {
    const n = parseInt(searchParams.get("prefill_nodes") || "", 10);
    return Number.isFinite(n) && n > 0 ? n : pdDefaults.prefill;
  });
  const [pdDecodeNodes, setPdDecodeNodes] = useState(() => {
    const n = parseInt(searchParams.get("decode_nodes") || "", 10);
    return Number.isFinite(n) && n > 0 ? n : pdDefaults.decode;
  });
  // Re-sync per-role node state with hw-specific defaults when hw changes,
  // unless the URL pins a value (shareable links must survive hw switches).
  useEffect(() => {
    if (!searchParams.get("prefill_nodes")) setPdPrefillNodes(pdDefaults.prefill);
    if (!searchParams.get("decode_nodes")) setPdDecodeNodes(pdDefaults.decode);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pdDefaults]);
  // PD-pool parallelism modes offered for this recipe, derived from
  // compatible_strategies (TP / TEP / DEP). A single-mode list (e.g. dense
  // models with only TP) hides the per-pool pills entirely.
  const pdModes = useMemo(() => pdPoolModes(recipe), [recipe]);
  // Default per-role parallelism: recipe strategy_overrides → strategy YAML →
  // "tp", clamped to an offered mode.
  const pdDefaultPar = useMemo(() => {
    const so = recipe.strategy_overrides?.pd_cluster || {};
    const st = strategies?.pd_cluster || {};
    const pick = (role) => {
      const d = so[role]?.parallelism || st[role]?.parallelism || "tp";
      return pdModes.includes(d) ? d : pdModes[0];
    };
    return { prefill: pick("prefill"), decode: pick("decode") };
  }, [recipe, strategies, pdModes]);
  const [pdPrefillPar, setPdPrefillPar] = useState(() => {
    const p = searchParams.get("prefill_mode");
    return p && pdModes.includes(p) ? p : pdDefaultPar.prefill;
  });
  const [pdDecodePar, setPdDecodePar] = useState(() => {
    const p = searchParams.get("decode_mode");
    return p && pdModes.includes(p) ? p : pdDefaultPar.decode;
  });
  // Which DP rank's command to render for each DEP pool. User can bump this
  // to see e.g. rank 7's command — illustrates that each rank differs only in
  // --data-parallel-rank, CUDA_VISIBLE_DEVICES, and the per-host ports.
  const [pdPrefillRank, setPdPrefillRank] = useState(() => {
    const n = parseInt(searchParams.get("prefill_rank") || "0", 10);
    return Number.isFinite(n) && n >= 0 ? n : 0;
  });
  const [pdDecodeRank, setPdDecodeRank] = useState(() => {
    const n = parseInt(searchParams.get("decode_rank") || "0", 10);
    return Number.isFinite(n) && n >= 0 ? n : 0;
  });
  const [strategyOverride, setStrategyOverride] = useState(() => {
    // A kv_store id in ?strategy= must never become the serving strategy —
    // Mooncake selection travels in ?kv_offload= instead.
    const s = searchParams.get("strategy") || "";
    return strategies[s]?.deploy_type === "kv_store_lb" ? "" : s;
  });
  // KV offload: "" = off, a taxonomy option key ("simple"/"lmcache") whose
  // connector args append to the current serving strategy, or a kv_store_lb
  // strategy id (Mooncake) that COMPOSES with it — each instance runs the
  // selected strategy's command wrapped in the router/master deployment shell.
  const [kvOffload, setKvOffload] = useState(searchParams.get("kv_offload") || "");
  // Mooncake instance count — independent vLLM engines sharing the KV pool.
  // Orthogonal to nodeCount (which stays "nodes per instance"). null = follow
  // the active topology's default_instances (distributed: 1, centralized: 2);
  // only read when a Mooncake mode is active.
  const [kvInstances, setKvInstances] = useState(() => {
    const n = parseInt(searchParams.get("kv_instances") || "", 10);
    return Number.isFinite(n) && n >= 1 && n <= 16 ? n : null;
  });
  // Which instance's command the vLLM tab renders (0-based, URL 0 omitted) —
  // same pattern as PD's per-pool node selector. Only the multi-node
  // --master-addr naming differs between instances.
  const [kvInstanceIdx, setKvInstanceIdx] = useState(() => {
    const n = parseInt(searchParams.get("kv_instance") || "0", 10);
    return Number.isFinite(n) && n >= 0 ? n : 0;
  });
  // Default-on features = (all features) − (recipe.opt_in_features) − (recipe.hardware_opt_in_features[hwId]).
  // The per-hw override lets a recipe suppress a feature's default on specific
  // hardware (e.g. GB200's 4-GPU trays make --mm-encoder-tp-mode data unnecessary).
  const defaultFeaturesFor = useCallback(
    (hw, variantKey) => {
      const optIn = new Set(recipe.opt_in_features || []);
      for (const f of recipe.hardware_opt_in_features?.[hw] || []) optIn.add(f);
      // A variant that declares a preferred mode for a feature (`default_modes`)
      // wants that feature on — un-opt-in it for this variant. E.g. selecting
      // the fused DSpark checkpoint auto-enables spec_decoding (default: dspark).
      const forced = new Set(Object.keys(recipe.variants?.[variantKey]?.default_modes || {}));
      return Object.keys(recipe.features || {}).filter((f) => forced.has(f) || !optIn.has(f));
    },
    [recipe]
  );

  // Encode a features array for the URL: returns "" (which syncUrl deletes)
  // when the set matches the YAML default for `hw`+`variantKey`, so links stay
  // clean unless the user actually deviated.
  const featuresToUrl = useCallback(
    (arr, hw, variantKey) => {
      const want = new Set(defaultFeaturesFor(hw, variantKey));
      const got = new Set(arr);
      const same = want.size === got.size && [...want].every((f) => got.has(f));
      return same ? "" : arr.join(",");
    },
    [defaultFeaturesFor]
  );

  const [features, setFeatures] = useState(() => {
    const fp = searchParams.get("features");
    if (fp) return fp.split(",").filter(Boolean);
    const urlHw = searchParams.get("hardware") || defaultHw;
    const urlVariant = searchParams.get("variant") || "default";
    return defaultFeaturesFor(urlHw, urlVariant);
  });

  // Features that are single-select ("pick one of N") rather than boolean —
  // they declare a `modes` map (e.g. spec_decoding → MTP / DFlash / DSpark).
  const featuredModeKeys = useMemo(
    () => Object.keys(recipe.features || {}).filter((k) => recipe.features[k]?.modes),
    [recipe]
  );
  // Default sub-mode per modes-feature for a given variant: a variant can steer
  // it via `default_modes: { <feature>: <mode> }` (e.g. the DSpark checkpoint
  // variant defaults spec_decoding to the dspark method); otherwise it falls
  // back to the feature's own `default_mode`. This keeps the two axes clean —
  // the variant owns model_id (the served checkpoint), the mode owns args (the
  // --speculative-config) — while letting a checkpoint pick its natural method.
  const variantDefaultModes = useCallback(
    (variantKey) => {
      const v = recipe.variants?.[variantKey] || recipe.variants?.default || {};
      const hp = taxonomy.hardware_profiles?.[hwId];
      const out = {};
      for (const k of featuredModeKeys) {
        out[k] = resolveModeKey(recipe.features[k], k, v, variantKey, hp, hwId, undefined);
      }
      return out;
    },
    [recipe, featuredModeKeys, taxonomy, hwId]
  );
  const defaultModes = useMemo(() => variantDefaultModes(variant), [variantDefaultModes, variant]);

  // Selected sub-mode per modes-feature. URL param `fmode` is a comma list of
  // `key:mode` pairs; defaults (for the active variant) are omitted so links stay clean.
  const parseFmode = useCallback((raw) => {
    const out = {};
    if (!raw) return out;
    for (const pair of raw.split(",")) {
      const [k, m] = pair.split(":");
      if (k && m && recipe.features?.[k]?.modes?.[m]) out[k] = m;
    }
    return out;
  }, [recipe]);
  const [featureModes, setFeatureModes] = useState(() => ({
    ...variantDefaultModes(searchParams.get("variant") || "default"),
    ...parseFmode(searchParams.get("fmode")),
  }));
  const featureModesToUrl = useCallback(
    (modes, variantKey) => {
      const defs = variantDefaultModes(variantKey);
      const parts = [];
      for (const k of featuredModeKeys) {
        if (modes[k] && modes[k] !== defs[k]) parts.push(`${k}:${modes[k]}`);
      }
      return parts.join(",");
    },
    [featuredModeKeys, variantDefaultModes]
  );

  // Restore saved sub-mode picks (URL wins). Separate from the main mount
  // effect so it runs after featureModes is declared.
  useEffect(() => {
    if (searchParams.get("fmode") || featuredModeKeys.length === 0) return;
    const rs = loadRecipeState(recipe.hf_id);
    if (rs.featureModes && typeof rs.featureModes === "object") {
      const restored = {};
      for (const k of featuredModeKeys) {
        const m = rs.featureModes[k];
        if (m && recipe.features[k]?.modes?.[m]) restored[k] = m;
      }
      if (Object.keys(restored).length) setFeatureModes((prev) => ({ ...prev, ...restored }));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Advanced tuning flags (defaults off) — toggled independently from features
  const [advanced, setAdvanced] = useState(() => {
    const ap = searchParams.get("advanced");
    return ap ? ap.split(",").filter(Boolean) : [];
  });

  // Cluster endpoints — substitution map for $VAR / NODE_N placeholders that
  // appear in rendered commands. Mirrors localStorage so a user fills once
  // per cluster and every recipe page reuses it. Empty values leave the
  // placeholder as-is.
  const [endpoints, setEndpoints] = useState({});
  useEffect(() => {
    setEndpoints(loadEndpoints());
  }, []);
  const updateEndpoint = useCallback((name, value) => {
    setEndpoints((prev) => {
      const next = { ...prev };
      if (value === undefined || value === null || value === "") delete next[name];
      else next[name] = value;
      return next;
    });
    saveEndpoint(name, value);
  }, []);
  const resetEndpoints = useCallback(() => {
    setEndpoints({});
    clearEndpoints();
  }, []);

  // Per-recipe advanced options declared under top-level `advanced_options:` in
  // the YAML are merged with the global presets so a recipe can surface its
  // own toggles (e.g. model-specific kernel backends) without code changes.
  const advancedOptions = useMemo(
    () => [...ADVANCED_OPTIONS, ...(recipe.advanced_options || [])],
    [recipe]
  );
  const advancedById = useMemo(
    () => Object.fromEntries(advancedOptions.map((o) => [o.id, o])),
    [advancedOptions]
  );

  // Install mode (pip | docker). Drives both the Install block's active tab
  // and the command rendering below: pip mode shows `vllm serve ...`, docker
  // mode wraps the same command in `docker run ...`. Default follows the
  // recipe's `model.install` key order — if `docker` is declared first, it
  // wins; otherwise pip is the default.
  const initialInstallMode = useMemo(() => {
    const install = recipe.model?.install || {};
    if (install.docker === false) return "pip";
    if (install.pip === false) return "docker";
    const keys = Object.keys(install).filter((k) => k === "pip" || k === "docker");
    return keys[0] === "docker" ? "docker" : "pip";
  }, [recipe]);
  const [installMode, setInstallMode] = useState(initialInstallMode);

  // CUDA variant selector for the NVIDIA docker image tag. State holds the
  // raw suffix (`"cu129"` | `"cu130"`) or `"default"` for the upstream base
  // tag (currently CUDA 13). Only surfaced for NVIDIA — AMD / TPU don't ship
  // paired CUDA variants.
  const [dockerCudaVariant, setDockerCudaVariant] = useState("default");

  // ── Derived ──
  const currentVariant = recipe.variants?.[variant] || recipe.variants?.default || {};

  // All hardware profiles grouped by brand, sorted by architectural generation
  // within brand (oldest → newest; matches the semianalysis GPU timeline).
  const hwByBrand = useMemo(() => {
    const NVIDIA_ORDER = ["h100", "h200", "b200", "gb200", "b300", "gb300", "dgx_station_gb300"];
    const AMD_ORDER = ["mi300x", "mi325x", "mi355x"];
    const rankIn = (list, id) => {
      const i = list.indexOf(id);
      return i === -1 ? 9999 : i;
    };
    // `restricted` profiles (e.g. TPU) only appear when the recipe explicitly
    // lists them in `meta.hardware` — keeps specialty hardware out of the
    // picker for recipes that haven't been validated on it.
    const declared = recipe.meta?.hardware || {};
    const groups = {};
    for (const [id, p] of Object.entries(taxonomy.hardware_profiles || {})) {
      if (p.restricted && !(id in declared)) continue;
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
  // Non-scalable hardware (single-GPU workstation, e.g. DGX Station) can't add
  // nodes, so multi-node is off and variants that don't fit are disabled.
  const hwScalable = isHardwareScalable(hwProfile);

  const recommended = useMemo(() => recommendStrategy(recipe, hwProfile, nodeCount), [recipe, hwProfile, nodeCount]);

  const compatibleStrategies = useMemo(() => {
    return (recipe.compatible_strategies || []).filter((s) => {
      const strat = strategies[s];
      if (!strat) return false;
      // kv_store_lb deployments live on the KV Offload row, not here.
      if (strat.deploy_type === "kv_store_lb") return false;
      if (nodeCount === 1 && strat.deploy_type === "multi_node") return false;
      if (nodeCount > 1 && strat.deploy_type === "single_node") return false;
      return true;
    });
  }, [recipe, strategies, nodeCount]);

  // Mooncake KV-store deployments (deploy_type: kv_store_lb) — offered on
  // every recipe by default (omni recipes never reach this row), no
  // compatible_strategies opt-in. Display order = the YAML `order` field
  // (distributed leads, matching the first-click default), alphabetical
  // fallback.
  const compatibleKvStoreStrategies = useMemo(
    () => Object.keys(strategies)
      .filter((s) => strategies[s]?.deploy_type === "kv_store_lb")
      .sort((a, b) =>
        ((strategies[a]?.order ?? 99) - (strategies[b]?.order ?? 99))
        || a.localeCompare(b)
      ),
    [strategies]
  );

  // Mooncake modes follow the repo's fail-open hardware convention: absent =
  // assumed to work; a recipe opts OUT per GPU by marking the strategy × GPU
  // pair `unsupported` under `kv_cache_strategy_hardware`.
  const isKvStoreSupported = useCallback(
    (s, hardwareId = hwId) => recipe.kv_cache_strategy_hardware?.[s]?.[hardwareId] !== "unsupported",
    [recipe, hwId]
  );

  // Serving-strategy per-GPU opt-out, mirroring kv_cache_strategy_hardware but
  // for the Strategy row: a recipe marks a (strategy, GPU) pair `unsupported`
  // under `strategy_hardware` (fail-open — absent = works). The pill renders
  // disabled with a tooltip and the active/recommended resolution skips it.
  const isStrategySupported = useCallback(
    (s, hardwareId = hwId) => recipe.strategy_hardware?.[s]?.[hardwareId] !== "unsupported",
    [recipe, hwId]
  );

  // PD now sizes each pool independently, so the "2× model VRAM on one node"
  // concern that used to invalidate pd_cluster on small GPUs no longer applies.
  const recommendedServingStrategy = (compatibleStrategies.includes(recommended) && isStrategySupported(recommended))
    ? recommended
    : (compatibleStrategies.find((s) => isStrategySupported(s)) || compatibleStrategies[0] || recommended);
  const activeServingStrategy = (compatibleStrategies.includes(strategyOverride) && isStrategySupported(strategyOverride))
    ? strategyOverride
    : recommendedServingStrategy;
  // Downgrade an unusable KV-offload pick instead of rendering a broken
  // command: composing options (taxonomy.kv_offload — Simple, LMCache) can't
  // run under pd_cluster (which owns --kv-transfer-config) or outside their
  // own `strategies` allowlist; Mooncake needs scalable hardware not opted
  // out by the recipe.
  const kvOffloadOptions = taxonomy.kv_offload || {};
  const activeKvOffload =
    kvOffloadOptions[kvOffload]
      ? (isKvOffloadAllowedForStrategy(kvOffloadOptions[kvOffload], activeServingStrategy, strategies[activeServingStrategy]) ? kvOffload : "")
      : compatibleKvStoreStrategies.includes(kvOffload) && hwScalable
          && isKvStoreBrandSupported(hwProfile) && isKvStoreSupported(kvOffload)
        ? kvOffload
        : "";
  const isKvStoreActive = compatibleKvStoreStrategies.includes(activeKvOffload);
  // Mooncake COMPOSES with the serving strategy — parallelism (TP/TEP/DEP,
  // single/multi-node) is orthogonal to the KV layer. Each instance runs the
  // selected strategy's command with the MooncakeStoreConnector appended;
  // synthesis wraps the result in the router/master/(store) deployment shell.
  // Under pd_cluster the per-role MultiConnector path composes instead. The
  // Strategy row therefore stays fully in effect at all times.
  const activeStrategy = activeServingStrategy;
  // Mooncake instance scaling applies (Instances row + per-instance Nodes
  // semantics) on every serving strategy except PD, whose pools size themselves.
  const kvInstancesActive = isKvStoreActive && activeServingStrategy !== "pd_cluster";

  // Keep the Mooncake selection coherent: a pick invalidated by the current
  // hardware (non-scalable or opted out — e.g. a ?hardware= link) is reset in
  // state + URL only. Storage is left alone: deliberate invalidation already
  // cleans it in selectHardware, and wiping it here would destroy a pick
  // saved on supported hardware the moment an unsupported render occurs.
  useEffect(() => {
    if (compatibleKvStoreStrategies.includes(kvOffload) && activeKvOffload !== kvOffload) {
      setKvOffload("");
      syncUrl({ kv_offload: "" });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeKvOffload, kvOffload]);

  // Effective TP under single_node_tp, via the shared resolver so the hint
  // is perfectly in sync with the generated command. The resolver accepts
  // both an explicit `strategy_overrides.single_node_tp.tp` and the
  // auto-fit from `variant.vram_minimum_gb` vs per-GPU VRAM.
  const hwGpuCount = typeof hwProfile.gpu_count === "number" ? hwProfile.gpu_count : 1;
  const effectiveTp = resolveSingleNodeTp(recipe, currentVariant, hwProfile, activeStrategy);
  const showGpuUsageHint =
    nodeCount === 1 && activeStrategy === "single_node_tp" && effectiveTp < hwGpuCount;

  const isSingleNode = nodeCount === 1 && typeof activeStrategy === "string" && activeStrategy.startsWith("single_node_");
  const needGb = currentVariant?.vram_minimum_gb;
  const availGb = hwProfile.vram_gb;
  const vramShortfall =
    isSingleNode && typeof needGb === "number" && typeof availGb === "number" && availGb > 0 && needGb > availGb
      ? { needGb, availGb, gpuCount: hwGpuCount, hwName: hwProfile.display_name || hwId }
      : null;

  const result = useMemo(
    () => {
      const advArgs = advanced.flatMap((id) => advancedById[id]?.args || []);
      const pdNodes = activeStrategy === "pd_cluster"
        ? {
          prefill: { nodes: pdPrefillNodes, rank: pdPrefillRank, parallelism: pdPrefillPar },
          decode: { nodes: pdDecodeNodes, rank: pdDecodeRank, parallelism: pdDecodePar },
        }
        : null;
      return resolveCommand(recipe, variant, activeStrategy, hwId, features, strategies, taxonomy, advArgs, nodeCount, pdNodes, featureModes, activeKvOffload || null, { count: kvInstances ?? undefined, current: kvInstanceIdx });
    },
    [recipe, variant, activeStrategy, hwId, features, featureModes, advanced, advancedById, strategies, taxonomy, nodeCount, pdPrefillNodes, pdDecodeNodes, pdPrefillRank, pdDecodeRank, pdPrefillPar, pdDecodePar, activeKvOffload, kvInstances, kvInstanceIdx]
  );

  // Visual feedback when any rendered command changes. Covers single-node
  // (result.command), multi-node (headCommand), pd_cluster (prefill.command),
  // and omni (resolveCommand's modelId doesn't reflect omni task swaps, so
  // include omniTask explicitly).
  const commandFingerprint = (result.command
    || result.headCommand
    || result.prefill?.command
    || result.vllm?.command
    || "")
    + (omniTask ? `|task:${omniTask}` : "");
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
  const selectOmniTask = (id) => {
    setOmniTask(id);
    // Default task id is omitted from the URL so a fresh page-load lands on
    // the recipe author's intended starting task without a noisy `?task=…`.
    const defaultId = omniTasks[0]?.id;
    syncUrl({ task: id === defaultId ? "" : id });
  };

  const selectVariant = (key) => {
    setVariant(key);
    // Swap hardware when precision or an explicit variant allowlist requires
    // it. VRAM alone is not a blocker because scalable profiles can add nodes.
    const v = recipe.variants?.[key] || {};
    const currentProfile = taxonomy.hardware_profiles?.[hwId] || {};
    const updates = { variant: key };
    if (!isPrecisionCompatible(currentProfile, v) || !isVariantHardwareSupported(v, hwId)) {
      const next = pickDefaultHardware(taxonomy.hardware_profiles, v, recipe);
      setHwId(next);
      updates.hardware = next;
    }
    // Reset spec-decoding mode(s) to the new variant's preferred default so
    // picking the DSpark checkpoint auto-selects the DSpark method (and picking
    // a plain-precision variant falls back to MTP). A manual mode pick after
    // this still wins until the next variant switch.
    if (featuredModeKeys.length) {
      const nextModes = variantDefaultModes(key);
      setFeatureModes(nextModes);
      updates.fmode = featureModesToUrl(nextModes, key);
      saveRecipeState(recipe.hf_id, { featureModes: nextModes });
    }
    // Enable features the new variant forces on (`default_modes`, e.g. the
    // DSpark checkpoint auto-enables spec_decoding), and drop the previous
    // variant's forced-only features unless they're a plain default here.
    const oldForced = new Set(Object.keys(currentVariant?.default_modes || {}));
    const newForced = Object.keys(v.default_modes || {});
    const baseDefault = new Set(defaultFeaturesFor(updates.hardware || hwId, key));
    let nextFeatures = features.filter((f) => !oldForced.has(f) || newForced.includes(f) || baseDefault.has(f));
    for (const f of newForced) if (recipe.features?.[f] && !nextFeatures.includes(f)) nextFeatures.push(f);
    if (nextFeatures.length !== features.length || nextFeatures.some((f, i) => f !== features[i])) {
      setFeatures(nextFeatures);
      updates.features = featuresToUrl(nextFeatures, updates.hardware || hwId, key);
      saveRecipeState(recipe.hf_id, { features: nextFeatures });
    }
    // One syncUrl call — two sequential calls each read the same stale
    // searchParams from this closure, so the second would clobber the first
    // (dropping variant= when selecting a Blackwell-only variant off Hopper).
    syncUrl(updates);
  };

  const selectHardware = (id) => {
    // Apply per-hw opt-in delta: turn off features the new hw opts out of,
    // and turn back on features the old hw opted out of that the new hw doesn't.
    // Features explicitly in recipe.opt_in_features stay off either way.
    const oldHwOptIn = new Set(recipe.hardware_opt_in_features?.[hwId] || []);
    const newHwOptIn = new Set(recipe.hardware_opt_in_features?.[id] || []);
    const baseOptIn = new Set(recipe.opt_in_features || []);
    const next = features.filter((f) => !newHwOptIn.has(f));
    for (const f of oldHwOptIn) {
      if (!newHwOptIn.has(f) && !baseOptIn.has(f) && !next.includes(f)) next.push(f);
    }
    setFeatures(next);
    setHwId(id);
    setStrategyOverride("");
    const newProfile = taxonomy.hardware_profiles?.[id] || {};
    // A selected spec-mode may be gated to specific hardware (e.g. a DFlash
    // draft that only ships for Blackwell). If the new GPU doesn't support it,
    // fall back to the first mode that does so the command stays valid.
    let nextModes = featureModes;
    for (const k of featuredModeKeys) {
      const feat = recipe.features[k];
      const sel = nextModes[k];
      if (sel && feat.modes[sel] && !isModeSupported(feat.modes[sel], newProfile, id)) {
        const firstOk = Object.keys(feat.modes).find((mk) => isModeSupported(feat.modes[mk], newProfile, id));
        if (firstOk) nextModes = { ...nextModes, [k]: firstOk };
      }
    }
    if (nextModes !== featureModes) setFeatureModes(nextModes);
    const newScalable = isHardwareScalable(newProfile);
    // Drop a Mooncake pick when the new GPU is opted out by the recipe or
    // can't be clustered, so the URL doesn't pin a dead config.
    const kvOffloadOk = !compatibleKvStoreStrategies.includes(kvOffload) ||
      (newScalable && isKvStoreBrandSupported(newProfile) && isKvStoreSupported(kvOffload, id));
    if (!kvOffloadOk) {
      setKvOffload("");
      // Full Mooncake sub-state reset, mirroring selectKvOffload("") — a
      // dropped pick must not leave instance count/index behind in state,
      // URL, or storage (they'd resurface on the next enable).
      setKvInstances(null);
      setKvInstanceIdx(0);
    }
    // If the active variant cannot run on the new hardware (explicit allowlist,
    // precision, or a non-scalable VRAM shortfall), fall to the largest variant
    // that can run there.
    let activeVariant = currentVariant;
    // Folded into the single syncUrl below rather than synced here — a separate
    // syncUrl call would read stale searchParams and get clobbered (same footgun
    // as selectVariant). Left undefined when the variant is unchanged so the
    // existing variant= param is preserved, not deleted.
    let variantUpdate;
    if (!variantRunsOnHardware(newProfile, currentVariant, id)) {
      const fitting = pickFittingVariant(recipe, newProfile, id);
      if (fitting && fitting !== variant) {
        setVariant(fitting);
        variantUpdate = fitting;
        activeVariant = recipe.variants?.[fitting] || currentVariant;
      }
    }
    // Bump to multi-node if the new hardware can't fit single-node (otherwise
    // the Single-node pill shows crossed out but the command keeps rendering
    // the invalid single-node config). Bump back DOWN to single-node when the
    // new hardware comfortably fits and the recipe's default is a single-node
    // strategy — without this, switching from GB200 (which bumped to 2 nodes
    // because the model didn't fit a 4-GPU tray) to B300/GB300 would stay at
    // 2 nodes and pick the multi-node sibling. Tied to the click so a
    // deliberate Single-/Multi-node click afterwards still wins. Non-scalable
    // hardware never bumps — it's single-node by definition.
    const fitsNew = fitsSingleNode(newProfile, activeVariant);
    const recipeDefault = recipe.default_strategy;
    const recipeDefaultsSingleNode =
      typeof recipeDefault === "string" && recipeDefault.startsWith("single_node_");
    const shouldBumpNodes = nodeCount === 1 && supportsMultiNode && newScalable && !fitsNew;
    const shouldUnbumpNodes = nodeCount > 1 && (!newScalable || (fitsNew && recipeDefaultsSingleNode));
    if (shouldBumpNodes) setNodeCount(2);
    if (shouldUnbumpNodes) setNodeCount(1);
    syncUrl({
      hardware: id,
      strategy: "",
      nodes: shouldBumpNodes ? "2" : shouldUnbumpNodes ? "" : undefined,
      features: featuresToUrl(next, id, variant),
      ...(nextModes !== featureModes ? { fmode: featureModesToUrl(nextModes, variant) } : {}),
      ...(variantUpdate ? { variant: variantUpdate } : {}),
      ...(kvOffloadOk ? {} : { kv_offload: "", kv_instances: "", kv_instance: "" }),
    });
    savePreference("hardware", id);
    // Mirror the new state to per-recipe storage so a hardware switch
    // doesn't leave stale strategy/nodes/features cached for this recipe.
    saveRecipeState(recipe.hf_id, {
      strategy: undefined,
      nodes: shouldBumpNodes ? 2 : shouldUnbumpNodes ? 1 : nodeCount,
      features: next,
      ...(nextModes !== featureModes ? { featureModes: nextModes } : {}),
      ...(kvOffloadOk ? {} : { kvOffload: undefined, kvInstances: undefined }),
    });
  };

  const selectStrategy = (s) => {
    setStrategyOverride(s);
    syncUrl({ strategy: s });
    // Persisted per-recipe (keyed by hf_id), so picking TP here doesn't
    // affect any other recipe's default. Spec-decoding auto-enable for
    // latency strategies is handled by an effect below so it also fires
    // on initial mount when TP is the default recommendation.
    saveRecipeState(recipe.hf_id, { strategy: s || undefined });
  };

  // "" = off · "simple"/"lmcache" = connector appended to the current serving
  // strategy · kv_store id = Mooncake composed around it (instances each run
  // the strategy's command behind the router). Node count is NOT touched —
  // the Nodes row keeps meaning "nodes per instance"; how many instances sit
  // behind the router is the separate Instances input below.
  const selectKvOffload = (v) => {
    setKvOffload(v);
    // Off resets the Mooncake sub-state (instances count + rendered instance)
    // so the next enable starts from the topology's defaults instead of
    // resurrecting a count picked for a previous topology. Topology switches
    // (v = another kv_store id) deliberately keep an explicit count.
    if (!v) {
      setKvInstances(null);
      setKvInstanceIdx(0);
    }
    syncUrl({ kv_offload: v, ...(v ? {} : { kv_instances: "", kv_instance: "" }) });
    saveRecipeState(recipe.hf_id, { kvOffload: v || undefined, ...(v ? {} : { kvInstances: undefined }) });
  };

  // Mooncake instance count. The active topology's default stays out of the
  // URL and storage so switching topologies keeps following their defaults.
  const selectKvInstances = (n) => {
    const clamped = Math.max(1, Math.min(16, parseInt(n, 10) || 1));
    const topoDefault = strategies[activeKvOffload]?.default_instances || 2;
    setKvInstances(clamped === topoDefault ? null : clamped);
    const idxUpdate = kvInstanceIdx > clamped - 1 ? { kv_instance: "" } : {};
    if (kvInstanceIdx > clamped - 1) setKvInstanceIdx(0);
    syncUrl({ kv_instances: clamped === topoDefault ? "" : String(clamped), ...idxUpdate });
    saveRecipeState(recipe.hf_id, { kvInstances: clamped === topoDefault ? undefined : clamped });
  };

  const selectKvInstanceIdx = (i) => {
    // kvInstances is null while the count follows the topology default
    // (`null - 1` would clamp every pick to instance 1) — clamp against the
    // effective count instead.
    const count = kvInstances ?? (strategies[activeKvOffload]?.default_instances || 2);
    const clamped = Math.max(0, Math.min(count - 1, i));
    setKvInstanceIdx(clamped);
    syncUrl({ kv_instance: clamped === 0 ? "" : String(clamped) });
  };

  const selectNodes = (n) => {
    setNodeCount(n);
    setStrategyOverride("");
    syncUrl({ nodes: n === 1 ? "" : String(n), strategy: "" });
    saveRecipeState(recipe.hf_id, { nodes: n, strategy: undefined });
  };

  const setPdNodes = (role, n) => {
    const clamped = Math.max(1, Math.min(16, parseInt(n, 10) || 1));
    if (role === "prefill") {
      setPdPrefillNodes(clamped);
      setPdPrefillRank(0);
      syncUrl({
        prefill_nodes: clamped === pdDefaults.prefill ? "" : String(clamped),
        prefill_rank: "",
      });
    } else {
      setPdDecodeNodes(clamped);
      setPdDecodeRank(0);
      syncUrl({
        decode_nodes: clamped === pdDefaults.decode ? "" : String(clamped),
        decode_rank: "",
      });
    }
  };

  const setPdRank = (role, r) => {
    const clamped = Math.max(0, parseInt(r, 10) || 0);
    if (role === "prefill") {
      setPdPrefillRank(clamped);
      syncUrl({ prefill_rank: clamped === 0 ? "" : String(clamped) });
    } else {
      setPdDecodeRank(clamped);
      syncUrl({ decode_rank: clamped === 0 ? "" : String(clamped) });
    }
  };

  // Switching a pool's parallelism changes what the rank/node index means
  // (DEP start-rank vs TP node-rank), so reset that pool's rank to 0.
  const setPdPar = (role, mode) => {
    if (!pdModes.includes(mode)) return;
    if (role === "prefill") {
      setPdPrefillPar(mode);
      setPdPrefillRank(0);
      syncUrl({ prefill_mode: mode === pdDefaultPar.prefill ? "" : mode, prefill_rank: "" });
    } else {
      setPdDecodePar(mode);
      setPdDecodeRank(0);
      syncUrl({ decode_mode: mode === pdDefaultPar.decode ? "" : mode, decode_rank: "" });
    }
  };

  const toggleFeature = (f) => {
    // text_only (skip vision encoder) and encoder_parallel (DP the encoder)
    // are mutually exclusive — enabling one clears the other.
    const mutex = { text_only: "encoder_parallel", encoder_parallel: "text_only" };
    const on = !features.includes(f);
    const next = on
      ? [...features.filter((x) => x !== mutex[f]), f]
      : features.filter((x) => x !== f);
    setFeatures(next);
    syncUrl({ features: featuresToUrl(next, hwId, variant) });
    saveRecipeState(recipe.hf_id, { features: next });
  };

  // Pick a sub-mode for a single-select feature (spec_decoding → MTP/DFlash/…).
  const selectFeatureMode = (featureKey, modeKey) => {
    const next = { ...featureModes, [featureKey]: modeKey };
    setFeatureModes(next);
    syncUrl({ fmode: featureModesToUrl(next, variant) });
    saveRecipeState(recipe.hf_id, { featureModes: next });
  };

  const toggleAdvanced = (id) => {
    const next = advanced.includes(id) ? advanced.filter((x) => x !== id) : [...advanced, id];
    setAdvanced(next);
    syncUrl({ advanced: next.length > 0 ? next.join(",") : "" });
  };

  const isPd = result.deployType === "pd_cluster";
  const isMultiNode = result.deployType === "multi_node";
  const isKvStore = result.deployType === "kv_store_lb";
  // Single-instance Mooncake renders no router — clients hit the engine
  // directly, so the curl/bench target falls back to VLLM_HOST:8000.
  const kvHasRouter = isKvStore && !!result.router;
  const modelId = recipe.variants?.[variant]?.model_id || recipe.model?.model_id || "model";

  // PD / KV-store clients hit the router; everyone else hits `vllm serve` on 8000.
  const clientPort = isPd ? 30000 : kvHasRouter ? (result.router?.port || 30080) : 8000;

  // curl/bench target. PD / routed KV-store → router host:port; everyone else
  // (incl. single-instance Mooncake) → the vllm-serve node (head node for
  // multi-node TP). Defaults to localhost so the single-node demo case still
  // works copy-paste; user can fill the Cluster endpoints panel to point at a
  // real cluster.
  const clientHostKey = (isPd || kvHasRouter) ? "ROUTER_HOST" : (isMultiNode ? "HEAD_IP" : "VLLM_HOST");
  const clientPortKey = (isPd || kvHasRouter) ? "ROUTER_PORT" : "VLLM_PORT";
  const clientHost = endpoints[clientHostKey] || "localhost";
  const clientPortStr = endpoints[clientPortKey] || String(clientPort);

  const verifyCmd = `curl http://${clientHost}:${clientPortStr}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${modelId}",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 32
  }'`;

  const benchCmd = `vllm bench serve \\
  --model ${modelId} \\
  --host ${clientHost} \\
  --port ${clientPortStr} \\
  --dataset-name random \\
  --random-input-len 1024 \\
  --random-output-len 1024 \\
  --num-prompts 100 \\
  --max-concurrency 32`;

  // Apply cluster-endpoint substitution to all rendered command/env strings.
  // Detection runs on the *un*substituted commands so the panel can list
  // every placeholder still pending a value.
  const placeholdersInUse = useMemo(() => {
    const texts = [];
    if (result.command) texts.push(result.command);
    for (const c of result.companions || []) texts.push(c.command);
    if (result.headCommand) texts.push(result.headCommand);
    if (result.workerCommand) texts.push(result.workerCommand);
    if (result.prefill?.command) texts.push(result.prefill.command);
    if (result.decode?.command) texts.push(result.decode.command);
    if (result.router?.command) texts.push(result.router.command);
    if (result.vllm?.command) texts.push(result.vllm.command);
    if (result.vllm?.workerCommand) texts.push(result.vllm.workerCommand);
    if (result.master?.command) texts.push(result.master.command);
    if (result.store?.command) texts.push(result.store.command);
    // Mooncake config JSONs carry placeholders too ($MOONCAKE_MASTER_IP,
    // $MOONCAKE_DEVICE_NAME) — scan them so the panel offers those fields.
    if (result.mooncakeConfig) texts.push(JSON.stringify(result.mooncakeConfig));
    if (result.store?.config) texts.push(JSON.stringify(result.store.config));
    // Mooncake composed into PD rides under result.mooncake.
    if (result.mooncake) {
      texts.push(result.mooncake.master.command);
      if (result.mooncake.store?.command) texts.push(result.mooncake.store.command);
      if (result.mooncake.store?.config) texts.push(JSON.stringify(result.mooncake.store.config));
      texts.push(JSON.stringify(result.mooncake.config));
    }
    for (const e of [result.env, result.prefill?.env, result.decode?.env, result.vllm?.env]) {
      if (!e) continue;
      for (const v of Object.values(e)) if (typeof v === "string") texts.push(v);
    }
    return detectPlaceholdersAll(...texts);
  }, [result]);

  // Merge in sensible defaults for the curl/bench/router target so the
  // rendered command is paste-runnable even when the user hasn't filled
  // the Endpoints panel. User values from `endpoints` always win. Per-node
  // IPs ($PREFILL_NODE_N, $HEAD_IP) intentionally have NO default — they're
  // not generally localhost and a wrong default would mislead the reader.
  const effectiveEndpoints = useMemo(() => {
    const defaults = {};
    if (result.deployType === "pd_cluster") {
      defaults.ROUTER_HOST = "localhost";
      defaults.ROUTER_PORT = "30000";
    } else if (result.deployType === "kv_store_lb") {
      if (result.router) {
        defaults.ROUTER_HOST = "localhost";
        defaults.ROUTER_PORT = String(result.router.port || 30080);
      } else {
        // Single instance, no router — clients hit the engine directly.
        defaults.VLLM_HOST = "localhost";
        defaults.VLLM_PORT = "8000";
      }
    } else if (result.deployType === "multi_node") {
      defaults.VLLM_PORT = "8000";
    } else {
      defaults.VLLM_HOST = "localhost";
      defaults.VLLM_PORT = "8000";
    }
    return { ...defaults, ...endpoints };
  }, [result.deployType, result.router?.port, endpoints]);

  const displayedResult = useMemo(() => {
    const sub = (s) => substitute(s, effectiveEndpoints);
    if (result.deployType === "pd_cluster") {
      return {
        ...result,
        prefill: { ...result.prefill, command: sub(result.prefill.command), env: substituteEnv(result.prefill.env, effectiveEndpoints) },
        decode:  { ...result.decode,  command: sub(result.decode.command),  env: substituteEnv(result.decode.env,  effectiveEndpoints) },
        router:  { ...result.router,  command: sub(result.router.command) },
        ...(result.mooncake ? {
          mooncake: {
            ...result.mooncake,
            master: { ...result.mooncake.master, command: sub(result.mooncake.master.command) },
            store: result.mooncake.store
              ? { ...result.mooncake.store, command: sub(result.mooncake.store.command), config: substituteEnv(result.mooncake.store.config, effectiveEndpoints) }
              : null,
            config: substituteEnv(result.mooncake.config, effectiveEndpoints),
          },
        } : {}),
      };
    }
    if (result.deployType === "kv_store_lb") {
      // The config objects get substituted too (flat maps, same shape as
      // env). Placeholders the user leaves unfilled stay literal here — the
      // configs render inside UNQUOTED heredocs, so the paster's shell
      // resolves them (exported value, or "" when unset — the auto-select
      // default for $MOONCAKE_DEVICE_NAME).
      return {
        ...result,
        vllm:   {
          ...result.vllm,
          command: sub(result.vllm.command),
          ...(result.vllm.workerCommand ? { workerCommand: sub(result.vllm.workerCommand) } : {}),
          env: substituteEnv(result.vllm.env, effectiveEndpoints),
        },
        master: { ...result.master, command: sub(result.master.command) },
        store:  result.store
          ? { ...result.store, command: sub(result.store.command), config: substituteEnv(result.store.config, effectiveEndpoints) }
          : null,
        router: result.router
          ? { ...result.router, command: sub(result.router.command) }
          : null,
        mooncakeConfig: substituteEnv(result.mooncakeConfig, effectiveEndpoints),
      };
    }
    if (result.deployType === "multi_node") {
      return {
        ...result,
        headCommand:   sub(result.headCommand),
        workerCommand: sub(result.workerCommand),
        env: substituteEnv(result.env, effectiveEndpoints),
      };
    }
    return {
      ...result,
      command: sub(result.command),
      env: substituteEnv(result.env, effectiveEndpoints),
      ...(result.companions
        ? { companions: result.companions.map((c) => ({ ...c, command: sub(c.command) })) }
        : {}),
    };
  }, [result, effectiveEndpoints]);

  // Brand-filter recipe dependencies against the currently-selected hardware.
  // `brand: NVIDIA | AMD | Intel` (or array) targets a single platform — entries
  // without `brand` are platform-agnostic and always render. Used for cross-
  // platform recipes (e.g. an omni recipe with separate NVIDIA / ROCm wheels)
  // so AMD users don't see CUDA-only steps and vice versa.
  const dependencies = useMemo(() => {
    const all = recipe.dependencies || [];
    const brand = hwProfile?.brand;
    const deps = all.filter((d) => {
      if (!d.brand) return true;
      const allowed = Array.isArray(d.brand) ? d.brand : [d.brand];
      return allowed.includes(brand);
    });
    // The active KV-offload option's runtime dep joins the same extra-install
    // block — present only while selected, and required then (part of
    // Copy-all). Mooncake's install is brand-keyed in the kv_store YAML
    // (CUDA wheel on NVIDIA, non-CUDA build on AMD).
    const kvOpt = kvOffloadOptions[activeKvOffload];
    if (kvOpt?.install) {
      deps.push({
        note: `${kvOpt.display_name || activeKvOffload} — required by the selected KV Offload option`,
        command: kvOpt.install,
      });
    } else if (isKvStoreActive) {
      const raw = strategies[activeKvOffload]?.vllm?.install;
      const cmd = raw && typeof raw === "object"
        ? (raw[brand === "AMD" ? "amd" : "nvidia"] || null)
        : (raw || null);
      if (cmd) {
        deps.push({
          note: "Mooncake transfer engine — required by MooncakeStoreConnector on every vLLM node",
          command: cmd,
        });
      }
    }
    return deps;
  }, [recipe.dependencies, hwProfile?.brand, kvOffloadOptions, activeKvOffload, isKvStoreActive, strategies]);

  // Status caption for the command block header.
  // Only `verified` is a positive signal worth surfacing; anything else
  // falls back to a neutral "vllm serve" label (treat as "assumed to work").
  // Verification covered the plain serving path — any KV Offload pick changes
  // the runtime (extra connector / whole deployment), so the badge yields to
  // the config summary (which names the option) rather than overclaim.
  const hwStatus = recipe.meta?.hardware?.[hwId]; // "verified" | undefined
  const hwFullName = hwProfile?.brand
    ? `${hwProfile.brand} ${hwProfile.display_name || hwId}`
    : (hwProfile?.display_name || hwId);
  const statusHeader = hwStatus === "verified" && !activeKvOffload ? (
    <span className="text-[11px] font-medium text-green-500 inline-flex items-center gap-1.5">
      <span className="inline-block w-1.5 h-1.5 rounded-full bg-green-500" />
      Verified on {hwFullName}
    </span>
  ) : null;

  // Fallback header when hardware isn't verified: a one-line config summary
  // so the reader can eyeball "is this command set for what I want?" without
  // scrolling to the Hardware / Variant / Strategy pills below.
  // Format: `<hw> · <parallelism> · <precision>` — e.g. `H200 · TP=8 · BF16`,
  // `2× H200 · TP=16 · BF16`, `H200 · PD cluster · FP8`.
  const hwDisplay = hwProfile?.display_name || hwId;
  // For a Mooncake deployment the node count is instances × nodes-per-instance
  // (nodeCount alone would understate a 4-instance cluster as "H200").
  const kvTotalNodes = result.deployType === "kv_store_lb"
    ? (result.instances || 1) * (result.nodeCount || 1)
    : null;
  const hwPart = kvTotalNodes
    ? (kvTotalNodes > 1 ? `${kvTotalNodes}× ${hwDisplay}` : hwDisplay)
    : nodeCount > 1 ? `${nodeCount}× ${hwDisplay}` : hwDisplay;
  // The serving strategy always names the parallelism — under Mooncake the
  // instances still run it (composition), so it never disappears from the
  // summary; the KV layer rides along as a suffix.
  const strategyPart = result.deployType === "pd_cluster"
    ? "PD cluster"
    : result.deployType === "multi_node" || (result.deployType === "kv_store_lb" && nodeCount > 1)
      ? (strategies[activeStrategy]?.display_name || activeStrategy)
      : effectiveTp
        ? `TP=${effectiveTp}`
        : (strategies[activeStrategy]?.display_name || activeStrategy);
  const precisionPart = currentVariant.precision?.toUpperCase();
  const configSummary = [hwPart, strategyPart, precisionPart].filter(Boolean).join(" · ")
    + (kvOffloadOptions[activeKvOffload]
        ? ` · ${kvOffloadOptions[activeKvOffload].display_name || activeKvOffload}`
        : isKvStoreActive
          ? ` · Mooncake (${(strategies[activeKvOffload]?.label || "").split(" ")[0] || "KV Store"})`
          : "");

  // Upstream `vllm/vllm-openai:latest` (and recent pinned tags) ship CUDA 13
  // as the base; CUDA 12.9 is the legacy alternate published as a `-cu129`
  // suffix. The recipe's `min_vllm_version` doesn't change which tag the user
  // pulls — `:latest` is always today's base regardless — so the alt suffix
  // is a constant.
  const altCudaSuffix = "cu129";

  const dockerMeta = useMemo(() => {
    const meta = computeDockerMeta(recipe, currentVariant, hwProfile, hwId);
    if (meta.brandKey !== "nvidia") return meta;

    // Explicit CUDA map (e.g. `{cu129: ..., cu130: ...}`) — pick the matching
    // tag and skip auto-suffix. "default" resolves to cu130 (the upstream
    // baseline today). If the chosen variant is missing, fall through to
    // whichever key is present.
    if (meta.cudaMap) {
      const baseCuda = "cu130";
      const wanted = dockerCudaVariant === "default" ? baseCuda : dockerCudaVariant;
      const picked = meta.cudaMap[wanted] || meta.cudaMap[baseCuda] || meta.cudaMap.cu129 || meta.cudaMap.cu130;
      return { ...meta, image: picked || meta.image };
    }

    // Legacy string tag — append the suffix when user picks the alt variant.
    // Nightly tags follow the inverse convention: `:nightly` → `:cu129-nightly`
    // (CUDA prefix, not suffix). Detect the upstream `:nightly` tag and swap
    // accordingly; pinned `:nightly`-bearing tags (e.g. `:myrelease-nightly`)
    // fall back to the suffix path.
    if (dockerCudaVariant === altCudaSuffix) {
      const isUpstreamNightly = /^vllm\/vllm-(openai|openai-rocm|tpu):nightly$/.test(meta.image);
      const next = isUpstreamNightly
        ? meta.image.replace(/:nightly$/, `:${altCudaSuffix}-nightly`)
        : `${meta.image}-${altCudaSuffix}`;
      return { ...meta, image: next };
    }
    return meta;
  }, [recipe, currentVariant, hwProfile, hwId, dockerCudaVariant, altCudaSuffix]);

  // `installMode` carries the user's tab choice; `effectiveInstallMode` folds
  // in constraints that would hide a tab entirely (pip: recipe opt-out or TPU
  // hardware; docker: recipe opt-out). This way switching to TPU flips both
  // the Install tab *and* the rendered command block to docker — they stay
  // in sync without requiring the user to re-click.
  const pipEffectivelyHidden =
    recipe.model?.install?.pip === false || hwProfile?.generation === "tpu";
  const ascendPipHidden = dockerMeta.isAscend;
  const dockerEffectivelyHidden = recipe.model?.install?.docker === false;
  const effectiveInstallMode =
    installMode === "pip" && (pipEffectivelyHidden || ascendPipHidden)
      ? "docker"
      : installMode === "docker" && dockerEffectivelyHidden
        ? "pip"
        : installMode;

  // Omni recipes serve via `vllm serve <model> --omni`. The command shape is
  // simpler than the regular path (no strategy / multi-node / pd), but the
  // surrounding affordances (Install tabs, Hardware pills, Variant pills) all
  // still apply. Plus a Task pill row that swaps endpoint + curl body — and,
  // for multi-checkpoint families like Wan2.2, the served model_id too.
  const isOmni = (recipe.meta?.tasks || []).includes("omni");
  if (isOmni) {
    const omniVariants = Object.entries(recipe.variants || {});
    const showOmniVariants = omniVariants.length > 1;
    const showOmniTaskRow = omniTasks.length > 1;
    const activeTask = omniTasks.find((t) => t.id === omniTask) || omniTasks[0] || null;

    // Render the `vllm serve --omni` command via the shared omni resolver.
    // Falls back to a stub when the recipe has no omni.tasks declared yet
    // (legacy omni-tagged recipes that haven't been migrated).
    const omniRendered = activeTask
      ? resolveOmniCommand(recipe, variant, activeTask, hwProfile)
      : {
          command: `${recipe.omni?.serve_binary || "vllm serve"} ${currentVariant.model_id || recipe.model?.model_id || "model"} --omni`,
          env: {},
          modelId: currentVariant.model_id || recipe.model?.model_id || "model",
        };
    const omniSubbedCommand = substitute(omniRendered.command, effectiveEndpoints);
    const omniSubbedEnv = substituteEnv(omniRendered.env, effectiveEndpoints);

    // Per-task verify command. `activeTask.example` knows the right endpoint
    // (e.g. /v1/images/generations vs /v1/chat/completions multimodal) and
    // payload shape; no chance for the generic /v1/chat/completions curl to
    // mislead the user into hitting the wrong route.
    const omniCurl = activeTask?.example
      ? activeTask.example({
          host: clientHost,
          port: clientPortStr,
          modelId: omniRendered.modelId,
          prompt: undefined,
        })
      : verifyCmd;

    // Recompute placeholders against the omni command set rather than the
    // (unused-here) `result` from resolveCommand.
    const omniPlaceholders = detectPlaceholdersAll(
      omniRendered.command,
      omniCurl,
      ...Object.values(omniRendered.env || {}).filter((v) => typeof v === "string"),
    );

    const omniEndpointsControls = (
      <EndpointsPopoverButton
        isPd={false}
        isKvStore={false}
        isMultiNode={false}
        placeholders={omniPlaceholders}
        endpoints={endpoints}
        onChange={updateEndpoint}
        onReset={resetEndpoints}
      />
    );

    // Config caption: hw · task · precision. Same shape as the non-omni
    // summary so the command-card header reads consistently across recipes.
    const omniConfigSummary = [
      hwProfile?.display_name || hwId,
      activeTask?.label,
      currentVariant.precision?.toUpperCase(),
    ].filter(Boolean).join(" · ");

    return (
      <TooltipProvider>
        <div className="space-y-4">
          <InstallBlock
            recipe={recipe}
            variant={currentVariant}
            dockerMeta={dockerMeta}
            installMode={effectiveInstallMode}
            setInstallMode={setInstallMode}
            dockerCudaVariant={dockerCudaVariant}
            setDockerCudaVariant={setDockerCudaVariant}
            altCudaSuffix={altCudaSuffix}
          />

          {effectiveInstallMode !== "docker" && dependencies.length > 0 && (
            <DependenciesBlock deps={dependencies} />
          )}

          <div
            className={`rounded-2xl overflow-hidden bg-[var(--command-bg)] border border-border transition-shadow ${changed ? "ring-2 ring-vllm-blue/30" : ""}`}
          >
            <SingleCommandBlock
              command={omniSubbedCommand}
              env={omniSubbedEnv}
              verifyCmd={omniCurl}
              benchCmd={benchCmd}
              statusHeader={statusHeader}
              installMode={effectiveInstallMode}
              dockerMeta={dockerMeta}
              configSummary={omniConfigSummary}
              endpointsControls={omniEndpointsControls}
            />
          </div>

          <div className="rounded-xl border border-border divide-y divide-border">
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
                        const variantHardwareOk = isVariantHardwareSupported(currentVariant, id);
                        const status = recipe.meta?.hardware?.[id];
                        const isUnsupported = status === "unsupported";
                        const disabled = !precisionOk || !variantHardwareOk || isUnsupported;
                        const verifiedNote = status === "verified"
                          ? "\n\nVerified — author has tested this hardware end-to-end"
                          : "";
                        const reason = !variantHardwareOk
                          ? `${currentVariant.precision?.toUpperCase()} is only supported on ${(currentVariant.supported_hardware || []).map((hw) => taxonomy.hardware_profiles?.[hw]?.display_name || hw).join(", ")}`
                          : !precisionOk
                          ? `${currentVariant.precision?.toUpperCase()} requires NVIDIA Blackwell`
                          : isUnsupported
                            ? `Not yet supported on ${p.display_name} — this model doesn't run here today, may be enabled in a future release`
                            : `${p.description}${verifiedNote}`;
                        return (
                          <Pill
                            key={id}
                            active={hwId === id}
                            disabled={disabled}
                            onClick={() => !disabled && selectHardware(id)}
                            title={reason}
                          >
                            <HwStatusDot status={status} />
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

            {showOmniTaskRow && (
              <ConfigRow
                label="Task"
                hint="Each task picks a vllm-omni handler endpoint and example payload. For multi-checkpoint families (Wan2.2 T2V/I2V/TI2V) the task also swaps the served model_id."
              >
                <PillGroup>
                  {omniTasks.map((t) => (
                    <Pill
                      key={t.id}
                      active={activeTask?.id === t.id}
                      onClick={() => selectOmniTask(t.id)}
                      title={[t.description, `Endpoint: ${t.endpoint}`].filter(Boolean).join("\n\n")}
                    >
                      <span className="font-semibold">{t.label}</span>
                      {t.vramMinimumGb && (
                        <span className="text-muted-foreground ml-1.5 font-mono">{t.vramMinimumGb} GB</span>
                      )}
                    </Pill>
                  ))}
                </PillGroup>
                {activeTask?.description && (
                  <p className="text-[11px] text-muted-foreground mt-2 leading-snug">
                    {activeTask.description}
                  </p>
                )}
              </ConfigRow>
            )}

            {showOmniVariants && (
              <ConfigRow
                label="Variant"
                hint="VRAM shown is the minimum to LOAD the model (weights + runtime overhead). vLLM-Omni inference may need more for activations and intermediate tensors."
              >
                <PillGroup>
                  {omniVariants.map(([key, v]) => (
                    <Pill
                      key={key}
                      active={variant === key}
                      onClick={() => selectVariant(key)}
                      title={[
                        v.description,
                        `Min ${v.vram_minimum_gb} GB to load.`,
                      ].filter(Boolean).join("\n\n")}
                    >
                      <span className="font-mono font-semibold">{(v.label || v.precision)?.toUpperCase()}</span>
                      <span className="text-muted-foreground ml-1.5 font-mono">{v.vram_minimum_gb} GB</span>
                    </Pill>
                  ))}
                </PillGroup>
              </ConfigRow>
            )}
          </div>
        </div>
      </TooltipProvider>
    );
  }

  return (
    <TooltipProvider>
      <div className="space-y-4">
        {/* ── Install (tabs: pip / docker — mirrors the command-card mode toggle
          so switching from either place keeps them in sync). */}
        <InstallBlock
          recipe={recipe}
            variant={currentVariant}
          dockerMeta={dockerMeta}
          installMode={effectiveInstallMode}
          setInstallMode={setInstallMode}
          dockerCudaVariant={dockerCudaVariant}
          setDockerCudaVariant={setDockerCudaVariant}
          altCudaSuffix={altCudaSuffix}
          extraMinVersion={isKvStoreActive ? strategies[activeKvOffload]?.min_vllm_version : null}
        />

        {/* ── Dependencies / extra install ──
            Hidden in docker mode: today every entry is a host-level
            `pip install` / `bash install_*.sh`, which doesn't apply when
            vLLM ships as a container image. Revisit if a dep ever needs
            to run inside the container. */}
        {effectiveInstallMode !== "docker" && dependencies.length > 0 && (
          <DependenciesBlock deps={dependencies} />
        )}

        {/* ── VRAM shortfall warning (single-node only) ── */}
        {vramShortfall && (
          <div className="rounded-xl border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-[12px] leading-snug flex items-start gap-2">
            <span aria-hidden="true" className="text-amber-500 mt-px">⚠</span>
            <div className="text-foreground/90">
              <span className="font-semibold text-amber-600 dark:text-amber-400">Insufficient VRAM for single-node: </span>
              {vramShortfall.hwName} provides {vramShortfall.gpuCount}×
              {Math.round(vramShortfall.availGb / vramShortfall.gpuCount)}G = {vramShortfall.availGb}GB,
              but this variant needs at least {vramShortfall.needGb}GB for weights alone
              (KV cache requires more).{" "}
              <span className="text-muted-foreground">Switch to a higher-memory GPU, use multi-node TP, or lower <code className="font-mono text-[11px] px-1 py-px rounded bg-muted/50">--max-model-len</code> to shrink the KV cache footprint.</span>
            </div>
          </div>
        )}

        {/* ── Command output ── */}
        {(() => {
          const endpointsControls = (
            <EndpointsPopoverButton
              isPd={isPd}
              isKvStore={kvHasRouter}
              isMultiNode={isMultiNode}
              placeholders={placeholdersInUse}
              endpoints={endpoints}
              onChange={updateEndpoint}
              onReset={resetEndpoints}
            />
          );
          return (
        <div
          className={`rounded-2xl overflow-hidden bg-[var(--command-bg)] border border-border transition-shadow ${changed ? "ring-2 ring-vllm-blue/30" : ""
            }`}
        >
          {isPd ? (
            <PdClusterBlock
              result={displayedResult}
              verifyCmd={verifyCmd}
              benchCmd={benchCmd}
              statusHeader={statusHeader}
              onRankChange={setPdRank}
              installMode={effectiveInstallMode}
              dockerMeta={dockerMeta}
              configSummary={configSummary}
              endpointsControls={endpointsControls}
            />
          ) : isKvStore ? (
            <KvStoreLbBlock
              result={displayedResult}
              verifyCmd={verifyCmd}
              benchCmd={benchCmd}
              statusHeader={statusHeader}
              onInstanceChange={selectKvInstanceIdx}
              installMode={effectiveInstallMode}
              dockerMeta={dockerMeta}
              configSummary={configSummary}
              endpointsControls={endpointsControls}
            />
          ) : isMultiNode ? (
            <MultiNodeBlock
              result={displayedResult}
              verifyCmd={verifyCmd}
              benchCmd={benchCmd}
              statusHeader={statusHeader}
              installMode={effectiveInstallMode}
              dockerMeta={dockerMeta}
              configSummary={configSummary}
              endpointsControls={endpointsControls}
            />
          ) : (
            <SingleCommandBlock
              command={displayedResult.command}
              env={displayedResult.env}
              companions={displayedResult.companions}
              verifyCmd={verifyCmd}
              benchCmd={benchCmd}
              statusHeader={statusHeader}
              installMode={effectiveInstallMode}
              dockerMeta={dockerMeta}
              configSummary={configSummary}
              endpointsControls={endpointsControls}
            />
          )}
        </div>
          );
        })()}

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
                      const variantHardwareOk = isVariantHardwareSupported(currentVariant, id);
                      // Only `verified` carries a label; everything else = silent default.
                      // `unsupported` = author opt-out for this model; disables the pill.
                      const status = recipe.meta?.hardware?.[id];
                      const isUnsupported = status === "unsupported";
                      // Per-role PD now sizes each pool independently, so hardware
                      // only needs to fit 1× model per node (standard precision
                      // check is enough). The old co-located single-node check
                      // (2× model on one node) is no longer the default UX.
                      const disabled = !precisionOk || !variantHardwareOk || isUnsupported;
                      const verifiedNote = status === "verified"
                        ? "\n\nVerified — author has tested this hardware end-to-end"
                        : "";
                      const reason = !variantHardwareOk
                        ? `${currentVariant.precision?.toUpperCase()} is only supported on ${(currentVariant.supported_hardware || []).map((hw) => taxonomy.hardware_profiles?.[hw]?.display_name || hw).join(", ")}`
                        : !precisionOk
                        ? `${currentVariant.precision?.toUpperCase()} requires NVIDIA Blackwell`
                        : isUnsupported
                          ? `Not yet supported on ${p.display_name} — this model doesn't run here today, may be enabled in a future release`
                          : `${p.description}${verifiedNote}`;
                      return (
                        <Pill
                          key={id}
                          active={hwId === id}
                          disabled={disabled}
                          onClick={() => !disabled && selectHardware(id)}
                          title={reason}
                        >
                          <HwStatusDot status={status} />
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
              {showGpuUsageHint && (
                <p className="text-[11px] text-muted-foreground/80 pt-1.5">
                  This recipe runs on {effectiveTp} of {hwGpuCount} GPUs on the selected node — add
                  <code className="font-mono mx-1 px-1 py-0.5 rounded bg-foreground/5 text-[10px]">--tensor-parallel-size N</code>
                  via Advanced to scale up.
                </p>
              )}
            </div>
          </ConfigRow>

          {/* Variant */}
          <ConfigRow
            label="Variant"
            hint="VRAM shown is the minimum to LOAD the model (weights + CUDA/vLLM runtime overhead, ≈ params × bytes × 1.2). It's not a serving budget — long context or large batch typically needs 1.5–2× more for KV cache."
          >
            <PillGroup>
              {Object.entries(recipe.variants || {}).map(([key, v]) => {
                // Disable variants excluded by an exact hardware allowlist,
                // incompatible precision, or a non-scalable VRAM shortfall.
                const disabled = !variantRunsOnHardware(hwProfile, v, hwId);
                const hardwareRestricted = !isVariantHardwareSupported(v, hwId);
                return (
                  <Pill
                    key={key}
                    active={variant === key}
                    disabled={disabled}
                    onClick={() => !disabled && selectVariant(key)}
                    title={
                      disabled
                        ? hardwareRestricted
                          ? `${(v.label || v.precision)?.toUpperCase()} is only supported on ${(v.supported_hardware || []).map((hw) => taxonomy.hardware_profiles?.[hw]?.display_name || hw).join(", ")}`
                          : `${(v.label || v.precision)?.toUpperCase()} needs ${v.vram_minimum_gb} GB but ${hwProfile.display_name || "this hardware"} has ${hwProfile.vram_gb} GB and can't scale out — pick a smaller-footprint variant`
                        : [
                            v.description,
                            `Min ${v.vram_minimum_gb} GB to load — add KV cache for serving. Scale out via multi-node if needed.`,
                          ].filter(Boolean).join("\n\n")
                    }
                  >
                    <span className="font-mono font-semibold">{(v.label || v.precision)?.toUpperCase()}</span>
                    <span className="text-muted-foreground ml-1.5 font-mono">{v.vram_minimum_gb} GB</span>
                  </Pill>
                );
              })}
            </PillGroup>
          </ConfigRow>

          {/* Strategy — always in effect, including under Mooncake: the KV
              layer composes with (never overrides) the parallelism choice.
              Each Mooncake instance runs this strategy's command. */}
          <ConfigRow label="Strategy">
            <PillGroup>
              {compatibleStrategies.map((s) => {
                const supported = isStrategySupported(s);
                return (
                  <Pill
                    key={s}
                    active={activeServingStrategy === s}
                    disabled={!supported}
                    onClick={() => supported && selectStrategy(s)}
                    title={supported
                      ? strategies[s]?.description
                      : `${strategies[s]?.display_name || s} isn't supported on ${hwProfile.display_name || hwId} for this model.`}
                  >
                    <span className="font-semibold">{strategies[s]?.display_name || s}</span>
                    {s === recommended && supported && (
                      <Sparkles size={10} className="text-vllm-yellow ml-1" />
                    )}
                  </Pill>
                );
              })}
            </PillGroup>
            {strategies[activeServingStrategy]?.description && (
              <p className="text-[11px] text-muted-foreground mt-2 leading-snug">
                {strategies[activeServingStrategy].description.split("\n")[0]}
                {isKvStoreActive && activeServingStrategy !== "pd_cluster" && (
                  <span className="text-vllm-blue/80"> Each Mooncake instance runs this strategy.</span>
                )}
              </p>
            )}
            {strategies[activeServingStrategy]?.orientation && (() => {
              const o = strategies[activeServingStrategy].orientation;
              const { label, classes } = o === "latency"
                ? { label: "Latency oriented", classes: "bg-green-500/20 text-green-600 dark:text-green-400" }
                : o === "balanced"
                  ? { label: "Balanced", classes: "bg-blue-500/20 text-blue-600 dark:text-blue-400" }
                  : o === "production"
                    ? { label: "Production deployment", classes: "bg-purple-500/20 text-purple-700 dark:text-purple-400" }
                    : { label: "Throughput oriented", classes: "bg-amber-500/20 text-amber-700 dark:text-amber-400" };
              return (
                <span className={`inline-block text-[10px] font-medium mt-1.5 px-1.5 py-0.5 rounded ${classes}`}>
                  {label}
                </span>
              );
            })()}
          </ConfigRow>

          {/* KV Offload — every option COMPOSES with the serving strategy
              above (the KV layer is orthogonal to parallelism). Simple/LMCache
              append a connector to the current command; Mooncake additionally
              wraps it in a router-fronted instance deployment (kv_store_lb).
              Both Mooncake topologies share one pill here; the Store Topology
              row below picks between them. */}
          <ConfigRow label="KV Offload">
            <PillGroup>
              <Pill
                active={activeKvOffload === ""}
                onClick={() => selectKvOffload("")}
                title="No KV offload — vLLM keeps the KV cache in GPU memory."
              >
                <span className="font-semibold">Off</span>
              </Pill>
              {(() => {
                // Composing options (Simple, LMCache) share the gating helper
                // with synthesis: pd_cluster always excluded, plus the
                // option's own `strategies` allowlist (LMCache = single-node
                // only; its MP server is node-local). The merged Mooncake pill
                // joins the same ordered list at MOONCAKE_PILL_ORDER, so the
                // row reads Off · Simple · Mooncake · LMCache.
                const pills = Object.entries(kvOffloadOptions).map(([key, opt]) => {
                  const allowed = isKvOffloadAllowedForStrategy(opt, activeServingStrategy, strategies[activeServingStrategy]);
                  // Say WHY it's disabled and what would enable it, not just
                  // "not available" — the allowlist gives us the answer.
                  const disabledTitle = activeServingStrategy === "pd_cluster"
                    ? `${opt.display_name || key} can't compose with PD cluster, which owns --kv-transfer-config. (Mooncake composes with PD instead.)`
                    : `${opt.display_name || key} works with: ${(opt.strategies || []).map((s) => strategies[s]?.display_name || s).join(", ")}.`;
                  return {
                    order: opt.order ?? 99,
                    el: (
                      <Pill
                        key={key}
                        active={activeKvOffload === key}
                        disabled={!allowed}
                        onClick={() => allowed && selectKvOffload(key)}
                        title={allowed ? opt.description : disabledTitle}
                      >
                        <span className="font-semibold">{opt.label || opt.display_name || key}</span>
                      </Pill>
                    ),
                  };
                });
                if (compatibleKvStoreStrategies.length > 0) {
                  // One merged pill for the framework; the Store Topology row
                  // below picks centralized vs distributed. Disabled only when
                  // NO topology can run here. First click lands on the leading
                  // topology in display order (distributed — embedded, no
                  // extra store process).
                  const supported = compatibleKvStoreStrategies.filter((s) => isKvStoreSupported(s));
                  const brandOk = isKvStoreBrandSupported(hwProfile);
                  const selectable = hwScalable && brandOk && supported.length > 0;
                  const defaultId = supported[0];
                  const disabledTitle = !brandOk
                    ? `Mooncake's transfer engine ships CUDA and ROCm builds only — not available on ${hwProfile.brand || ""} ${hwProfile.display_name || hwId} backends.`
                    : !hwScalable
                      ? `${hwProfile.display_name || "This hardware"} is a single-GPU workstation and can't run a multi-node KV-store deployment.`
                      : `Mooncake KV store is not supported on ${hwProfile.display_name || hwId}.`;
                  pills.push({
                    order: MOONCAKE_PILL_ORDER,
                    el: (
                      <Pill
                        key="__mooncake"
                        active={isKvStoreActive}
                        disabled={!selectable}
                        onClick={() => selectable && !isKvStoreActive && selectKvOffload(defaultId)}
                        title={selectable
                          // One-liner for the tooltip — the full framework
                          // background renders under the row once active.
                          ? "Mooncake — distributed KV cache: vLLM instances share prefix KV through a coordinated CPU-DRAM pool. Composes with the selected strategy — each instance runs it."
                          : disabledTitle}
                      >
                        <span className="font-semibold">Mooncake</span>
                      </Pill>
                    ),
                  });
                }
                return pills.sort((a, b) => a.order - b.order).map((p) => p.el);
              })()}
            </PillGroup>
            {/* Active option's description below the pills — same pattern as
                the Strategy row's text. Composing options read the taxonomy
                (description + docs_url); Mooncake shows framework-level
                background here, while the topology-specific text lives under
                the Store Topology row. */}
            {(() => {
              const opt = kvOffloadOptions[activeKvOffload];
              const text = opt?.description || (isKvStoreActive ? MOONCAKE_BACKGROUND : null);
              const docs = opt?.docs_url || (isKvStoreActive ? MOONCAKE_DOCS_URL : null);
              return text ? (
                <p className="text-[11px] text-muted-foreground mt-2 leading-snug">
                  {text}
                  {docs && (
                    <>
                      {" "}
                      <a href={docs} target="_blank" rel="noreferrer" className="underline underline-offset-2 hover:text-foreground">
                        Docs ↗
                      </a>
                    </>
                  )}
                </p>
              ) : null;
            })()}
          </ConfigRow>

          {/* Store topology — Mooncake only. Same one-of-N idiom as the
              spec_decoding method row; both YAMLs stay separate deployment
              specs, this row just picks which id kv_offload points at. */}
          {isKvStoreActive && compatibleKvStoreStrategies.length > 1 && (
            <ConfigRow label="Store Topology" nested>
              <PillGroup>
                {compatibleKvStoreStrategies.map((s) => {
                  const supported = isKvStoreSupported(s);
                  return (
                    <Pill
                      key={s}
                      active={activeKvOffload === s}
                      disabled={!supported}
                      onClick={() => supported && selectKvOffload(s)}
                      title={supported
                        ? strategies[s]?.description
                        : `${strategies[s]?.display_name || s} is not supported on ${hwProfile.display_name || hwId}.`}
                    >
                      {strategies[s]?.label || strategies[s]?.display_name || s}
                    </Pill>
                  );
                })}
              </PillGroup>
              {/* Active topology's description — embedded vs standalone-store
                  specifics belong to this row, not the KV Offload row. */}
              {strategies[activeKvOffload]?.description && (
                <p className="text-[11px] text-muted-foreground mt-2 leading-snug">
                  {strategies[activeKvOffload].description.split("\n")[0]}
                </p>
              )}
            </ConfigRow>
          )}

          {/* Instances — Mooncake only: independent vLLM engines behind the
              router. Orthogonal to Nodes, which keeps meaning "nodes per
              instance" (an instance can itself span nodes like multi_node_tp).
              Hidden when Mooncake composes into PD — the PD pools size
              themselves. */}
          {kvInstancesActive && (
            <ConfigRow
              label="Instances"
              nested
              hint="Independent vLLM engines sharing the Mooncake KV pool; a cache-aware router fronts them at 2+ instances. Each instance spans the node count picked in the Nodes row below."
            >
              <div className="inline-flex flex-wrap items-center gap-2 text-sm">
                <input
                  type="number"
                  min={1}
                  max={16}
                  aria-label="Number of vLLM instances"
                  value={result.instances || kvInstances || 1}
                  onChange={(e) => selectKvInstances(e.target.value)}
                  className="w-14 px-2 py-1 text-sm font-mono tabular-nums rounded-md border border-border bg-background focus:outline-none focus:ring-1 focus:ring-vllm-blue/40"
                />
                <span className="text-xs text-muted-foreground tabular-nums">
                  × {nodeCount} node{nodeCount > 1 ? "s" : ""} × {hwProfile.gpu_count || 8} GPUs
                  = {(result.instances || 1) * nodeCount * (hwProfile.gpu_count || 8)} GPUs total
                </span>
              </div>
            </ConfigRow>
          )}

          {/* Nodes — two number inputs for PD (one per pool), pills otherwise */}
          {activeStrategy === "pd_cluster" ? (
            <ConfigRow
              label="Nodes"
              hint={pdModes.length > 1
                ? "Each pool (prefill / decode) sizes and shards independently — pick its parallelism (TP / TEP / DEP) and node count. Total cluster = prefill_nodes + decode_nodes."
                : "Each pool (prefill / decode) sizes independently. Total cluster = prefill_nodes + decode_nodes."}
            >
              <div className="flex flex-col gap-2 text-sm">
                <PdNodeInput
                  label="Prefill"
                  value={pdPrefillNodes}
                  gpuPerNode={hwProfile.gpu_count || 8}
                  onChange={(n) => setPdNodes("prefill", n)}
                  modes={pdModes}
                  parallelism={pdPrefillPar}
                  onParChange={(m) => setPdPar("prefill", m)}
                />
                <PdNodeInput
                  label="Decode"
                  value={pdDecodeNodes}
                  gpuPerNode={hwProfile.gpu_count || 8}
                  onChange={(n) => setPdNodes("decode", n)}
                  modes={pdModes}
                  parallelism={pdDecodePar}
                  onParChange={(m) => setPdPar("decode", m)}
                />
                <span className="text-xs text-muted-foreground tabular-nums">
                  total {(pdPrefillNodes + pdDecodeNodes) * (hwProfile.gpu_count || 8)} GPUs
                  · {pdPrefillNodes + pdDecodeNodes} node{pdPrefillNodes + pdDecodeNodes === 1 ? "" : "s"}
                </span>
              </div>
            </ConfigRow>
          ) : (
            // Under Mooncake the row means "nodes per instance" — the label
            // and tooltips say so, and totals multiply by the instance count
            // (the plain "2 × GPUs" math would understate the deployment).
            <ConfigRow
              label={kvInstancesActive ? "Nodes / Instance" : "Nodes"}
              hint={kvInstancesActive
                ? "Node count of EACH vLLM instance — an instance shards across nodes via the selected strategy's multi-node layout. Total nodes = Instances × this."
                : undefined}
            >
              <PillGroup>
                {[1, 2].map((n) => {
                  // Multi-node pill is disabled when the recipe declares no
                  // multi_node_* (or pd_cluster) strategy (small dense models
                  // commonly omit these), or when the active hardware can't be
                  // clustered (single-GPU workstation, e.g. DGX Station).
                  const noMultiNode = n > 1 && (!supportsMultiNode || !hwScalable);
                  // Single-node pill is disabled when the variant can't fit on
                  // one node of the selected hardware — same struck-through
                  // treatment as unsupported hardware pills. Multi-node still
                  // works because weights shard across nodes.
                  const singleNodeDoesntFit =
                    n === 1 && !fitsSingleNode(hwProfile, currentVariant);
                  const disabled = noMultiNode || singleNodeDoesntFit;
                  return (
                    <Pill
                      key={n}
                      active={nodeCount === n}
                      disabled={disabled}
                      onClick={() => !disabled && selectNodes(n)}
                      title={
                        noMultiNode
                          ? !hwScalable
                            ? `${hwProfile.display_name || "This hardware"} is a single-GPU workstation and can't be clustered into multiple nodes.`
                            : "This recipe does not declare a multi-node strategy. Fits in a single node."
                          : singleNodeDoesntFit
                            ? `Single-node can't fit this variant on ${hwProfile.display_name || "the selected hardware"} (${currentVariant.vram_minimum_gb}GB > ${hwProfile.vram_gb}GB) — use multi-node`
                            : n === 1
                              ? kvInstancesActive
                                ? `Each vLLM instance runs on a single node${(result.instances || 1) > 1 ? ` — ${result.instances} instances = ${result.instances} nodes total (plus master${result.store ? "/store" : ""})` : ""}.`
                                : "Single-node deployment (one HGX box)"
                              : kvInstancesActive
                                ? `Each instance spans 2 nodes × ${hwProfile.gpu_count || 8} GPUs${(result.instances || 1) > 1 ? ` — ${result.instances} instances = ${(result.instances || 1) * 2 * (hwProfile.gpu_count || 8)} GPUs total` : ` = ${2 * (hwProfile.gpu_count || 8)} GPUs`}.`
                                : `2 nodes × ${hwProfile.gpu_count || 8} GPUs = ${2 * (hwProfile.gpu_count || 8)} GPUs total. Scale further by replicating the worker command with higher --node-rank / --data-parallel-start-rank.`
                      }
                    >
                      <span className="font-semibold">{n === 1 ? "Single-node" : "Multi-node (example: 2)"}</span>
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
          )}

          {/* Features */}
          {Object.keys(recipe.features || {}).length > 0 && (
            <ConfigRow label="Features">
              <PillGroup>
                {Object.entries(recipe.features || {}).map(([key, f]) => {
                  // Strategy-gated feature (feature declares `strategies: [...]`):
                  // struck-through + disabled on excluded strategies, same
                  // treatment as unsupported hardware pills. The toggle state is
                  // kept, so switching back to an allowed strategy restores it —
                  // synthesis independently skips gated features, so the emitted
                  // command is correct either way.
                  // Companion-backed features can't render their helper tab
                  // inside the Mooncake deployment shell — synthesis skips
                  // their args too, so the pill disables to match.
                  const kvBlocked = kvInstancesActive && !!f?.companion?.command;
                  const allowed = isFeatureAllowedForStrategy(f, activeStrategy) && !kvBlocked;
                  return (
                  <Pill
                    key={key}
                    active={features.includes(key) && allowed}
                    disabled={!allowed}
                    onClick={() => allowed && toggleFeature(key)}
                    title={!allowed
                      ? kvBlocked
                        ? "Needs its companion process, which isn't rendered inside a Mooncake deployment — set KV Offload to Off to use it."
                        : "Not available with this strategy"
                      : f?.description}
                  >
                    {key === "spec_decoding" && (
                      <Zap size={11} className="inline-block mr-1 -mt-0.5 text-vllm-yellow" fill="currentColor" />
                    )}
                    {key === "tool_calling" && (
                      <Wrench size={11} className="inline-block mr-1 -mt-0.5 text-muted-foreground" />
                    )}
                    {key === "reasoning" && (
                      <Brain size={11} className="inline-block mr-1 -mt-0.5 text-muted-foreground" />
                    )}
                    {f?.label || key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()).replace(/\bKv\b/g, "KV")}
                    {key === "spec_decoding" && (
                      <span className="ml-1.5 text-[11px] text-vllm-yellow font-normal">
                        (for low latency & small batch size)
                      </span>
                    )}
                  </Pill>
                  );
                })}
              </PillGroup>
            </ConfigRow>
          )}

          {/* Sub-mode selector for single-select features (spec_decoding →
              MTP / DFlash / DSpark). Only shown while the parent feature is on. */}
          {featuredModeKeys.map((key) => {
            if (!features.includes(key)) return null;
            const feat = recipe.features[key];
            // Only modes available on the current checkpoint (variant). If a
            // single method is available (e.g. FP8/NVFP4 → MTP only), the row is
            // redundant with the feature toggle, so don't render it — the toggle
            // alone means "MTP on". The DSpark checkpoint exposes {MTP, DSpark}.
            const availEntries = Object.entries(feat.modes).filter(([, m]) => isModeAllowedForVariant(m, variant));
            if (availEntries.length < 2) return null;
            const active = resolveModeKey(feat, key, currentVariant, variant, hwProfile, hwId, featureModes[key]);
            const rowLabel = key === "spec_decoding"
              ? "Spec method"
              : `${feat.label || key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()).replace(/\bKv\b/g, "KV")} method`;
            return (
              <ConfigRow key={`${key}-modes`} label={rowLabel} nested>
                <PillGroup>
                  {availEntries.map(([mk, m]) => {
                    const supported = isModeSupported(m, hwProfile, hwId);
                    return (
                      <Pill
                        key={mk}
                        active={active === mk && supported}
                        disabled={!supported}
                        onClick={() => supported && selectFeatureMode(key, mk)}
                        title={!supported ? "Not supported on this hardware" : m.description}
                      >
                        {m.label || mk.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                      </Pill>
                    );
                  })}
                </PillGroup>
              </ConfigRow>
            );
          })}

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
                {advancedOptions.filter((opt) => !opt.gatedBy || opt.gatedBy(recipe, activeStrategy)).map((opt) => (
                  <label
                    key={opt.id}
                    className={`flex items-start gap-2.5 p-2 rounded-lg border cursor-pointer transition-colors ${advanced.includes(opt.id)
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
    </TooltipProvider>
  );
}

// ── Sub-components ──

function EndpointInput({ label, hint, value, onChange }) {
  return (
    <label className="flex items-center gap-2 min-w-0">
      <span className="text-[11px] font-mono text-muted-foreground shrink-0 w-32 truncate" title={label}>
        {label}
      </span>
      <input
        type="text"
        value={value}
        placeholder={hint || ""}
        onChange={(e) => onChange(e.target.value)}
        spellCheck={false}
        className="flex-1 min-w-0 px-2 py-1 text-xs font-mono rounded-md border border-border bg-background focus:outline-none focus:ring-1 focus:ring-vllm-blue/40"
      />
    </label>
  );
}

// Sensible per-placeholder placeholder hints (shown when the field is empty).
function endpointHintFor(name) {
  if (name.endsWith("_DP_RPC_PORT")) return "12345";
  if (name.endsWith("_PORT")) return "port";
  if (/^(?:PREFILL|DECODE)_NODE_\d+$/.test(name)) return "host";
  if (/^VLLM_INSTANCE_\d+$/.test(name)) return "10.0.0.x";
  if (name === "MOONCAKE_MASTER_IP") return "10.0.0.1";
  if (name.endsWith("_IP")) return "10.0.0.1";
  if (name === "IFACE_NAME") return "bond0";
  if (name === "ROUTER_HOST") return "localhost";
  if (name === "MOONCAKE_CONFIG_PATH") return "/etc/mooncake/mooncake_vllm_config.json";
  if (name === "MOONCAKE_STORE_CONFIG_PATH") return "/etc/mooncake/mooncake_store_config.json";
  if (name.endsWith("DEVICE_NAME")) return "mlx5_0,mlx5_1,… (blank = auto)";
  return "value";
}

const PD_PAR_LABELS = { tp: "TP", tep: "TEP", dep: "DEP" };
const PD_PAR_TIPS = {
  tp: "Tensor parallel — one engine sharded across the pool's GPUs; only the head node serves HTTP.",
  tep: "Tensor + expert parallel — TP layout plus --enable-expert-parallel (MoE models).",
  dep: "Data + expert parallel — one vllm serve per node, DP across nodes with EP (MoE throughput).",
};

function PdNodeInput({ label, value, gpuPerNode, onChange, modes, parallelism, onParChange }) {
  const showPills = Array.isArray(modes) && modes.length > 1 && typeof onParChange === "function";
  return (
    <div className="inline-flex flex-wrap items-center gap-2">
      <span className="text-xs font-medium text-muted-foreground w-12 shrink-0">{label}</span>
      {showPills && (
        <span className="inline-flex gap-1">
          {modes.map((m) => (
            <InfoTip key={m} content={PD_PAR_TIPS[m]}>
              <button
                onClick={() => onParChange(m)}
                aria-label={PD_PAR_TIPS[m]}
                className={`inline-flex items-center rounded-md border px-1.5 py-0.5 text-[11px] font-mono transition-all ${
                  parallelism === m
                    ? "border-vllm-blue bg-vllm-blue/5 text-foreground ring-1 ring-vllm-blue/20"
                    : "border-border text-muted-foreground hover:text-foreground hover:border-muted-foreground/40 hover:bg-muted/30"
                }`}
              >
                {PD_PAR_LABELS[m] || m.toUpperCase()}
              </button>
            </InfoTip>
          ))}
        </span>
      )}
      <input
        type="number"
        min={1}
        max={16}
        aria-label={`${label} node count`}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-14 px-2 py-1 text-sm font-mono tabular-nums rounded-md border border-border bg-background focus:outline-none focus:ring-1 focus:ring-vllm-blue/40"
      />
      <span className="text-xs text-muted-foreground tabular-nums">
        × {gpuPerNode} = {value * gpuPerNode}
      </span>
    </div>
  );
}

function ConfigRow({ label, hint, nested, children }) {
  // `nested` marks a row that only exists as a child of the pick above it
  // (Store Topology / Instances under Mooncake, the spec-method row under
  // Features): a blue left rail + slight tint expresses the ownership that
  // the flat divide-y list otherwise hides.
  return (
    <div className={`px-4 py-3 flex flex-col sm:flex-row sm:items-start gap-2 sm:gap-4 ${nested ? "border-l-2 border-vllm-blue/30 bg-muted/20" : ""}`}>
      <div className="text-[10px] font-semibold text-muted-foreground uppercase tracking-widest sm:w-28 sm:pt-1.5 shrink-0 inline-flex items-center gap-1">
        {label}
        {hint && (
          <InfoTip content={hint}>
            <span
              className="cursor-help text-muted-foreground/60 hover:text-muted-foreground transition-colors"
              aria-label={hint}
            >
              <Info size={11} />
            </span>
          </InfoTip>
        )}
      </div>
      <div className="flex-1 min-w-0">{children}</div>
    </div>
  );
}

function PillGroup({ children }) {
  return <div className="flex flex-wrap gap-1.5">{children}</div>;
}

function HwStatusDot({ status }) {
  // Only `verified` is a meaningful signal. Everything else (including
  // undeclared GPUs) renders nothing — the pill looks clean.
  if (status !== "verified") return null;
  return <span className="inline-block w-1.5 h-1.5 rounded-full mr-1.5 shrink-0 bg-green-500" aria-hidden />;
}

function Pill({ active, onClick, title, dimmed, disabled, children }) {
  // disabled takes precedence over active — an "active but disabled" pill should
  // clearly look disabled (e.g. PD that was pre-selected but no longer fits).
  // dimmed de-emphasizes a selectable pill (dashed border, muted text).
  const style = disabled
    ? "border-dashed border-border/40 text-muted-foreground/30 cursor-not-allowed bg-muted/20 line-through decoration-muted-foreground/30"
    : active
      ? "border-vllm-blue bg-vllm-blue/5 text-foreground ring-1 ring-vllm-blue/20 shadow-sm"
      : dimmed
        ? "border-dashed border-border/60 text-muted-foreground/50 hover:text-muted-foreground hover:border-muted-foreground/30"
        : "border-border text-muted-foreground hover:text-foreground hover:border-muted-foreground/40 hover:bg-muted/30";
  const btn = (
    <button
      onClick={onClick}
      disabled={disabled}
      aria-disabled={disabled}
      aria-label={typeof title === "string" ? title : undefined}
      className={`inline-flex items-center rounded-lg border px-2.5 py-1.5 text-xs transition-all ${style} ${disabled ? "pointer-events-none" : ""}`}
    >
      {children}
    </button>
  );
  // A natively-disabled button swallows pointer events, so Radix tooltips
  // never open on it — exactly the pills whose "why is this disabled"
  // explanation matters most. The span wrapper is the hoverable/focusable
  // trigger (the button itself gets pointer-events-none above).
  const trigger = disabled ? (
    <span tabIndex={0} className="inline-flex rounded-lg cursor-not-allowed">{btn}</span>
  ) : btn;
  return title ? <InfoTip content={title}>{trigger}</InfoTip> : trigger;
}

// One paste-runnable script that provisions Mooncake's config file(s): it
// exports the path vars (respecting values already set in the environment),
// then writes each JSON via unquoted heredoc — sizing/NIC notes render as
// leading # lines. Shared by the KV-store and PD blocks so the heredocs live
// on a single "Mooncake Config" tab instead of repeating on every tab that
// consumes the files.
function buildMooncakeConfigsCommand(config, note, store) {
  const noteLines = (n) =>
    n ? `${String(n).trimEnd().split("\n").map((l) => `# ${l}`).join("\n")}\n` : "";
  const exports = [
    "export MOONCAKE_CONFIG_PATH=${MOONCAKE_CONFIG_PATH:-/etc/mooncake/mooncake_vllm_config.json}",
    ...(store?.config
      ? ["export MOONCAKE_STORE_CONFIG_PATH=${MOONCAKE_STORE_CONFIG_PATH:-/etc/mooncake/mooncake_store_config.json}"]
      : []),
  ].join("\n");
  const parts = [
    exports,
    `${noteLines(note)}cat > $MOONCAKE_CONFIG_PATH <<EOF\n${JSON.stringify(config || {}, null, 2)}\nEOF`,
  ];
  if (store?.config) {
    parts.push(`${noteLines(store.note)}cat > $MOONCAKE_STORE_CONFIG_PATH <<EOF\n${JSON.stringify(store.config, null, 2)}\nEOF`);
  }
  return parts.join("\n\n");
}

function envToExports(env) {
  return Object.entries(env || {})
    .map(([k, v]) => `export ${k}=${v}`)
    .join("\n");
}

// Shared tab strip for the command blocks. `tabs`: [{ id, label, step? }] —
// `step` renders a dimmed launch-order number, making the left-to-right
// sequence explicit instead of implied by tab order. Implements the ARIA tabs
// pattern with roving focus (arrow keys move the selection).
function CommandTabs({ tabs, current, onSelect }) {
  const refs = useRef({});
  const onKeyDown = (e) => {
    const idx = Math.max(0, tabs.findIndex((t) => t.id === current));
    let next = null;
    if (e.key === "ArrowRight") next = tabs[(idx + 1) % tabs.length];
    else if (e.key === "ArrowLeft") next = tabs[(idx - 1 + tabs.length) % tabs.length];
    else if (e.key === "Home") next = tabs[0];
    else if (e.key === "End") next = tabs[tabs.length - 1];
    if (next) {
      e.preventDefault();
      onSelect(next.id);
      refs.current[next.id]?.focus();
    }
  };
  return (
    <div
      role="tablist"
      aria-label="Launch steps"
      onKeyDown={onKeyDown}
      className="flex flex-wrap gap-0.5 bg-foreground/5 rounded-md p-0.5"
    >
      {tabs.map((t) => (
        <button
          key={t.id}
          ref={(el) => { refs.current[t.id] = el; }}
          role="tab"
          aria-selected={current === t.id}
          tabIndex={current === t.id ? 0 : -1}
          onClick={() => onSelect(t.id)}
          className={`px-2.5 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap ${current === t.id ? "bg-foreground/10 text-[var(--command-fg)]" : "text-[var(--command-fg)]/50 hover:text-[var(--command-fg)]/80"
            }`}
        >
          {t.step != null && (
            <span className="font-mono tabular-nums opacity-45 mr-1">{t.step}·</span>
          )}
          {t.label}
        </button>
      ))}
    </div>
  );
}

function SingleCommandBlock({ command, env, companions, verifyCmd, benchCmd, statusHeader, installMode, dockerMeta, configSummary, endpointsControls }) {
  const [tab, setTab] = useState("vllm");
  const isDocker = installMode === "docker";
  // Docker mode: env vars fold into `-e` flags inside the wrapped `docker run`,
  // so there's no separate prelude (the `docker pull` lives in the Install
  // block tabs above). Pip mode: prelude = `export KEY=VAL` lines.
  const prelude = isDocker ? "" : envToExports(env);
  const displayCommand = isDocker
    ? dockerMeta.isAscend
      ? buildAscendDockerRun({ command, env, image: dockerMeta.image, gpuFlags: dockerMeta.gpuFlags })
      : buildDockerRun({ command, env, image: dockerMeta.image, gpuFlags: dockerMeta.gpuFlags })
    : command;
  // A companion process may ride along (`companions[]` from resolveCommand —
  // a feature's `companion:` or the active kv_offload option's, e.g.
  // LMCache's `lmcache server`). When any are active the block grows a
  // PD-style tab bar in LAUNCH ORDER — companions sit LEFT of vLLM Serve
  // because they must be running before it starts. With none, the classic
  // single-command layout renders untouched.
  const hasCompanions = Array.isArray(companions) && companions.length > 0;
  // Falls back to the vLLM view when the selected companion's source was
  // toggled off (stale tab state).
  const activeCompanion = hasCompanions && tab !== "vllm"
    ? companions.find((c) => c.feature === tab) || null
    : null;
  // When a companion (dis)appears — the user toggled its option — jump to the
  // leftmost tab so the launch sequence reads left to right from step 1.
  const companionIds = hasCompanions ? companions.map((c) => c.feature).join(",") : "";
  useEffect(() => {
    setTab(hasCompanions ? companions[0].feature : "vllm");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [companionIds]);
  // Companions are host-side helper binaries (not `vllm serve`), so they get
  // neither the docker-run wrapper nor the env prelude.
  const activePrelude = activeCompanion ? "" : prelude;
  // Leading # lines of a companion command render in the dimmed comment area
  // (same treatment as the description line); the bright pre keeps only the
  // executable body. Copy still grabs comments + body.
  const companionParts = activeCompanion
    ? (() => {
        const lines = String(activeCompanion.command).split("\n");
        let i = 0;
        while (i < lines.length && lines[i].trimStart().startsWith("#")) i++;
        return { comments: lines.slice(0, i).join("\n"), body: lines.slice(i).join("\n") };
      })()
    : null;
  const activeCommand = activeCompanion ? companionParts.body : displayCommand;
  const fullScript = activeCompanion
    ? activeCompanion.command
    : (activePrelude ? `${activePrelude}\n\n${activeCommand}` : activeCommand);
  const actions = (
    <div className="flex items-center gap-1.5 shrink-0">
      <CopyButton text={fullScript} />
      <PopoverButton label="cURL" code={verifyCmd} icon={Terminal} disabled={!!activeCompanion} disabledNote="Clients talk to the vLLM server — cURL & Bench live on the vLLM Serve tab." />
      <PopoverButton label="Bench" code={benchCmd} icon={Gauge} disabled={!!activeCompanion} disabledNote="Clients talk to the vLLM server — cURL & Bench live on the vLLM Serve tab." />
      {endpointsControls}
    </div>
  );
  return (
    <div>
      {hasCompanions ? (
        <>
          <div className="px-4 pt-3 pb-1">
            {statusHeader || (
              <span className="text-[11px] text-[var(--command-fg)]/55 font-mono">
                {configSummary}
              </span>
            )}
          </div>
          <div className="flex items-center justify-between px-4 pt-2 gap-3">
            <CommandTabs
              tabs={[...companions.map((c) => ({ id: c.feature, label: c.label })), { id: "vllm", label: "vLLM Serve" }].map((t, i) => ({ ...t, step: i + 1 }))}
              current={activeCompanion ? activeCompanion.feature : "vllm"}
              onSelect={setTab}
            />
            {actions}
          </div>
        </>
      ) : (
        <div className="flex items-center justify-between px-4 pt-3 gap-3">
          {statusHeader || (
            <span className="text-[11px] text-[var(--command-fg)]/55 font-mono">
              {configSummary}
            </span>
          )}
          {actions}
        </div>
      )}
      {activeCompanion?.description && (
        <div className="px-4 pt-3 text-[11px] text-[var(--command-fg)]/55 font-mono leading-snug">
          # {activeCompanion.description}
        </div>
      )}
      {companionParts?.comments && (
        <div className="px-4 pt-2 text-[11px] text-[var(--command-fg)]/55 font-mono leading-snug whitespace-pre overflow-x-auto">
          {companionParts.comments}
        </div>
      )}
      {activePrelude && (
        <pre className="px-4 pt-3 pb-1 text-[12px] text-[var(--command-fg)]/70 font-mono leading-relaxed whitespace-pre overflow-x-auto">
          {activePrelude}
        </pre>
      )}
      <pre className="px-4 py-3 text-[13px] text-[var(--command-fg)] font-mono leading-relaxed whitespace-pre overflow-x-auto">
        {activeCommand}
      </pre>
    </div>
  );
}

// Higher of two "X.Y.Z" version strings (missing/blank loses). Used to bump the
// displayed min vLLM version when the active variant needs a newer release than
// the recipe baseline (e.g. the DSpark checkpoint requires 0.25.0).
function maxVersion(a, b) {
  if (!a) return b;
  if (!b) return a;
  const pa = String(a).split(".").map((n) => parseInt(n, 10) || 0);
  const pb = String(b).split(".").map((n) => parseInt(n, 10) || 0);
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    const d = (pa[i] || 0) - (pb[i] || 0);
    if (d !== 0) return d > 0 ? a : b;
  }
  return a;
}

function InstallBlock({ recipe, variant, dockerMeta, installMode, setInstallMode, dockerCudaVariant, setDockerCudaVariant, altCudaSuffix, extraMinVersion = null }) {
  // Collapsible install reference. Shows the one-time setup step for the
  // active mode — `uv pip install vllm …` in pip mode, `docker pull <image>`
  // in docker mode. The active tab mirrors the command card's mode toggle
  // and vice versa, so switching from either place keeps them in sync.
  // The actual `docker run` (or `vllm serve`) lives in the command card below.
  // Per-recipe overrides via `model.install.{pip,docker}`:
  //   false              → hide that tab entirely
  //   { command, note }  → override the generated one-liner and/or show a note
  const install = recipe.model?.install || {};
  const pipCfg = install.pip;
  const dockerCfg = install.docker;
  const pipHidden = pipCfg === false;
  const dockerHidden = dockerCfg === false;
  const [open, setOpen] = useState(false);
  const { isAmd, isTpu, isAscend, image: dockerImage, brandKey, cudaMap } = dockerMeta;
  // A variant may require a newer vLLM than the recipe baseline (e.g. the DSpark
  // checkpoint needs 0.25.0, currently nightly). Take the higher version and OR
  // the nightly flag so the Install block reflects the selected checkpoint.
  // extraMinVersion: a deployment-level floor beyond recipe/variant — e.g.
  // the active Mooncake kv_store YAML's min_vllm_version (connector ships
  // in 0.21.0+), so the Install header never understates the requirement.
  const minV = maxVersion(maxVersion(recipe.model?.min_vllm_version, variant?.min_vllm_version), extraMinVersion);
  // Omni recipes are served by vLLM-Omni, a fast-moving companion package that
  // tracks vLLM nightly (Wan2.2 even pins a git commit). Surface it next to the
  // vLLM version so users know the generation path needs nightly wheels.
  const isOmni = recipe.meta?.tasks?.includes("omni");

  // When a recipe's min_vllm_version hasn't shipped yet (cutting-edge models
  // that landed after the last stable release), `model.nightly_required: true`
  // swaps the default pip command to the nightly wheel index and surfaces a
  // pill in the Install header. Manual `install.pip.command` overrides still
  // win — this flag only affects the default.
  const nightlyRequired = recipe.model?.nightly_required === true || variant?.nightly_required === true;
  // Resolve the CUDA tag for pip's nightly wheel index from the same toggle
  // that drives the Docker tag suffix. "default" → cu130 (today's upstream
  // baseline); explicit picks pass through.
  const pipCudaTag = dockerCudaVariant === "default" ? "cu130" : dockerCudaVariant;
  const defaultPipCmd = isAmd
    ? `uv venv --python 3.12
source .venv/bin/activate
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm`
    : nightlyRequired
      ? `uv venv
source .venv/bin/activate
uv pip install -U vllm --pre \\
  --extra-index-url https://wheels.vllm.ai/nightly/${pipCudaTag} \\
  --extra-index-url https://download.pytorch.org/whl/${pipCudaTag} \\
  --index-strategy unsafe-best-match`
      : `uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto`;
  const pipCmd = pipCfg?.command || defaultPipCmd;
  const pipNote =
    pipCfg?.note ||
    (nightlyRequired && !isAmd
      ? `vLLM ${minV} isn't released yet — nightly required. For CUDA 12.9, switch the toggle to cu129.`
      : undefined);

  // Docker install step is just the image pull; the `docker run` that actually
  // serves the model is rendered in the main command block below. A YAML
  // override at `model.install.docker.command` still wins for recipes that
  // need a custom build step. The CUDA-version selector (below, next to Copy)
  // drives the tag suffix for NVIDIA; AMD / TPU pull a single image.
  // Ascend also uses one image selected for the active Atlas generation.
  const defaultDockerCmd = `docker pull ${dockerImage}`;
  const dockerCmd = dockerCfg?.command || defaultDockerCmd;
  const defaultDockerNote = isTpu
    ? "TPU builds are published by vllm-project/tpu-inference. See the Trillium and Ironwood tpu-recipes for pinned image tags and exact deployment flags."
    : isAscend
      ? "Ascend recipes should pin the vLLM Ascend image tested on the selected Atlas system."
    : isAmd
      ? undefined
      : cudaMap
        ? "This recipe ships paired CUDA-tagged images. Pick `cu129` for CUDA 12.9 hosts or `cu130` for CUDA 13."
        : nightlyRequired
          ? "Nightly image ships CUDA 13. Switch to cu129 for the `cu129-nightly` variant if your host is on CUDA 12.9."
          : "Default tag ships CUDA 13. Switch to cu129 for the -cu129 variant if your host is on CUDA 12.9.";
  const dockerNote = dockerCfg?.note || defaultDockerNote;
  // Show the CUDA selector when we're on NVIDIA and the user isn't supplying
  // a full override command (which already bakes in a specific tag). Visible
  // on the docker tab always, and on the pip tab when the command actually
  // varies by CUDA — i.e. nightly recipes whose wheel index URL is explicit.
  // Stable pip uses `--torch-backend auto`, which detects the host CUDA, so
  // a toggle would be inert there.
  const showCudaSelector =
    brandKey === "nvidia" &&
    !dockerCfg?.command &&
    (installMode === "docker" ||
      (installMode === "pip" && nightlyRequired && !pipCfg?.command));

  // TPU has no pip wheel — force-hide the pip tab regardless of recipe overrides.
  const effectivePipHidden = pipHidden || isTpu;
  // Ascend requires a matched vLLM Ascend image in this foundation phase.
  const ascendPipHidden = isAscend;
  const dockerLabel = isTpu ? "Docker (TPU)" : isAscend ? "Docker (Ascend)" : isAmd ? "Docker (ROCm)" : "Docker";
  const tabs = [
    !(effectivePipHidden || ascendPipHidden) && {
      id: "pip",
      label: isAmd ? "pip / uv (ROCm)" : "pip / uv",
      code: pipCmd,
      note: pipNote,
    },
    !dockerHidden && {
      id: "docker",
      label: dockerLabel,
      code: dockerCmd,
      note: dockerNote,
    },
  ].filter(Boolean);
  if (tabs.length === 0) return null;
  const active = tabs.find((t) => t.id === installMode) || tabs[0];
  const summary = tabs.map((t) => (t.id === "pip" ? "pip" : "Docker")).join(" / ");

  return (
    <div className="rounded-2xl overflow-hidden bg-[var(--command-bg)] border border-border">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-3 px-4 py-2.5 text-left hover:bg-white/[0.02] transition-colors"
      >
        <Package size={12} className="text-[var(--command-fg)]/50 shrink-0" />
        <span className="text-[11px] font-semibold text-[var(--command-fg)]/70 uppercase tracking-widest">Install</span>
        <span className="text-[11px] text-[var(--command-fg)]/40 font-mono">
          vLLM {minV}+{isOmni ? " · vLLM-Omni nightly" : ""} · {isTpu ? "TPU" : isAscend ? "Ascend" : isAmd ? "ROCm" : "CUDA"}
        </span>
        {nightlyRequired && (
          <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-400 border border-amber-500/30 uppercase tracking-wider">
            nightly
          </span>
        )}
        <span className="text-[11px] text-[var(--command-fg)]/40 ml-auto">
          {open ? "hide" : summary}
        </span>
        <ChevronDown
          size={14}
          className={`text-[var(--command-fg)]/50 transition-transform ${open ? "rotate-180" : ""}`}
        />
      </button>
      {open && (
        <div className="border-t border-[var(--command-fg)]/10">
          <div className="flex items-center justify-between px-4 pt-3">
            <div className="flex gap-0.5 bg-foreground/5 rounded-md p-0.5">
              {tabs.map((t) => (
                <button
                  key={t.id}
                  onClick={() => setInstallMode(t.id)}
                  className={`px-2.5 py-1 text-xs font-medium rounded transition-colors ${active.id === t.id ? "bg-foreground/10 text-[var(--command-fg)]" : "text-[var(--command-fg)]/50 hover:text-[var(--command-fg)]/80"
                    }`}
                >
                  {t.label}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-2">
              {showCudaSelector && (
                <div className="flex gap-0.5 bg-foreground/5 rounded-md p-0.5">
                  {[
                    { id: "default", label: "default" },
                    { id: altCudaSuffix, label: altCudaSuffix },
                  ].map((v) => (
                    <button
                      key={v.id}
                      onClick={() => setDockerCudaVariant(v.id)}
                      title={
                        v.id === "cu129"
                          ? "Legacy CUDA 12.9 build"
                          : v.id === "cu130"
                            ? "CUDA 13 build"
                            : cudaMap
                              ? "Recipe-recommended tag for this hardware"
                              : "Base tag — CUDA 13"
                      }
                      className={`px-2 py-0.5 text-[11px] font-mono rounded transition-colors ${
                        dockerCudaVariant === v.id
                          ? "bg-foreground/10 text-[var(--command-fg)]"
                          : "text-[var(--command-fg)]/50 hover:text-[var(--command-fg)]/80"
                      }`}
                    >
                      {v.label}
                    </button>
                  ))}
                </div>
              )}
              <CopyButton text={active.code} />
            </div>
          </div>
          {active.note && (
            <div className="px-4 pt-2 text-[11px] text-[var(--command-fg)]/55 leading-snug">
              # {active.note}
            </div>
          )}
          <pre className="px-4 py-3 text-[13px] text-[var(--command-fg)] font-mono leading-relaxed whitespace-pre overflow-x-auto">
            {active.code}
          </pre>
        </div>
      )}
    </div>
  );
}

function DependenciesBlock({ deps }) {
  // Copy-all only includes required entries — optional ones often target a
  // different platform (e.g. AMD ROCm in a recipe with NVIDIA-only kernels),
  // so blindly copy-pasting everything would mix incompatible installs.
  const requiredDeps = deps.filter((d) => !d.optional);
  const requiredCommands = requiredDeps.map((d) => d.command).join("\n");
  const requiredCount = requiredDeps.length;
  const optionalCount = deps.length - requiredCount;
  return (
    <div className="rounded-2xl overflow-hidden bg-[var(--command-bg)] border border-border">
      <div className="flex items-center justify-between px-4 pt-3">
        <span className="text-[11px] text-[var(--command-fg)]/50 font-mono inline-flex items-center gap-1.5">
          <Package size={11} /> extra install
          {requiredCount > 0 && <span className="text-[var(--command-fg)]/40">· {requiredCount} required</span>}
          {optionalCount > 0 && <span className="text-[var(--command-fg)]/40">· {optionalCount} optional</span>}
        </span>
        <CopyButton text={requiredCommands} />
      </div>
      <div className="px-4 py-3 text-[13px] font-mono leading-relaxed overflow-x-auto space-y-2">
        {deps.map((d, i) => (
          <div key={i} className={d.optional ? "opacity-50" : undefined}>
            {d.note && (
              <div className="text-[var(--command-fg)]/45 text-[11px] leading-snug mb-0.5 inline-flex items-center gap-1.5">
                {d.optional && (
                  <span className="inline-flex items-center rounded px-1 py-px text-[9px] font-semibold uppercase tracking-wider bg-foreground/10 text-[var(--command-fg)]/60">
                    Optional
                  </span>
                )}
                <span># {d.note}</span>
              </div>
            )}
            <div className="text-[var(--command-fg)] whitespace-pre">{d.command}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function MultiNodeBlock({ result, verifyCmd, benchCmd, statusHeader, installMode, dockerMeta, configSummary, endpointsControls }) {
  const [tab, setTab] = useState("head");
  const isDocker = installMode === "docker";
  const wrap = (cmd) =>
    isDocker
      ? dockerMeta.isAscend
        ? buildAscendDockerRun({ command: cmd, env: result.env, image: dockerMeta.image, gpuFlags: dockerMeta.gpuFlags })
        : buildDockerRun({ command: cmd, env: result.env, image: dockerMeta.image, gpuFlags: dockerMeta.gpuFlags })
      : cmd;
  const tabs = [
    { id: "head", label: "Head", command: wrap(result.headCommand) },
    { id: "worker", label: "Node 1", command: wrap(result.workerCommand) },
  ];
  const active = tabs.find((t) => t.id === tab) || tabs[0];
  // Docker mode folds env into `-e` flags inside `docker run`, so no
  // separate prelude here. Pip mode keeps the `export KEY=VAL` prelude.
  const prelude = isDocker ? "" : envToExports(result.env);
  const fullScript = prelude ? `${prelude}\n\n${active.command}` : active.command;
  return (
    <div>
      <div className="px-4 pt-3 pb-1">
        {statusHeader || (
          <span className="text-[11px] text-[var(--command-fg)]/55 font-mono">
            {configSummary}
          </span>
        )}
      </div>
      <div className="flex items-center justify-between px-4 pt-2">
        <CommandTabs tabs={tabs} current={active.id} onSelect={setTab} />
        <div className="flex items-center gap-1.5">
          <CopyButton text={fullScript} />
          <PopoverButton label="cURL" code={verifyCmd} icon={Terminal} disabled={active.id !== "head"} disabledNote="Clients connect to the head node — cURL & Bench live on the Head tab." />
          <PopoverButton label="Bench" code={benchCmd} icon={Gauge} disabled={active.id !== "head"} disabledNote="Clients connect to the head node — cURL & Bench live on the Head tab." />
          {endpointsControls}
        </div>
      </div>
      {prelude && (
        <pre className="px-4 pt-3 pb-1 text-[12px] text-[var(--command-fg)]/70 font-mono leading-relaxed whitespace-pre overflow-x-auto">
          {prelude}
        </pre>
      )}
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

function PdClusterBlock({ result, verifyCmd, benchCmd, statusHeader, onRankChange, installMode, dockerMeta, configSummary, endpointsControls }) {
  // Tabs: Prefill · Decode · Router.
  // Each pool (prefill/decode) now carries its own `nodes`, `parallelism`,
  // `dpSize`, `poolGpus` meta — rendered above the command so the reader
  // knows whether the block is a single engine or a rank-0 template that
  // needs to be duplicated for the rest of the DP ranks.
  const [tab, setTab] = useState("prefill");
  const isDocker = installMode === "docker";
  // Prefill/decode are `vllm serve` and get wrapped in `docker run`. The
  // router is `vllm-router` (separate pip package, different entrypoint) —
  // it stays as-is with its pip-install hint regardless of install mode.
  const wrap = (cmd, env) =>
    isDocker
      ? dockerMeta.isAscend
        ? buildAscendDockerRun({ command: cmd, env, image: dockerMeta.image, gpuFlags: dockerMeta.gpuFlags })
        : buildDockerRun({ command: cmd, env, image: dockerMeta.image, gpuFlags: dockerMeta.gpuFlags })
      : cmd;
  // Mooncake composed into PD (result.mooncake): a "Mooncake Config" tab
  // (launch step 0) writes the shared config file(s) once — every
  // prefill/decode node reads them via $MOONCAKE_CONFIG_PATH — then
  // master/(store) tabs follow in launch order. Same conventions as
  // KvStoreLbBlock.
  const mc = result.mooncake;
  // With Mooncake composed in, the tabs become a real launch sequence
  // (config → master → (store) → pools → router), so they get step numbers.
  // Plain PD keeps unnumbered tabs — prefill/decode have no strict order.
  const tabs = [
    ...(mc ? [{
      id: "mc_config", label: "Mooncake Config",
      command: buildMooncakeConfigsCommand(mc.config, mc.configNote, mc.store), env: {},
      description: `Run once on every node (prefill + decode${mc.store ? " + store node" : ""}) before the steps to the right — they read these files via $MOONCAKE_CONFIG_PATH${mc.store ? " / $MOONCAKE_STORE_CONFIG_PATH" : ""}.`,
    }] : []),
    ...(mc ? [{ id: "mc_master", label: "Mooncake Master", command: mc.master.command, env: {}, description: mc.master.description }] : []),
    ...(mc?.store ? [{ id: "mc_store", label: "Mooncake Store", command: mc.store.command, env: {}, description: mc.store.description }] : []),
    { id: "prefill", label: "Prefill", command: wrap(result.prefill.command, result.prefill.env), env: result.prefill.env, meta: result.prefill },
    { id: "decode", label: "Decode", command: wrap(result.decode.command, result.decode.env), env: result.decode.env, meta: result.decode },
    { id: "router", label: "Router", command: result.router.command, env: {}, install: result.router.install, isRouter: true },
  ].map((t, i) => (mc ? { ...t, step: i + 1 } : t));
  // When Mooncake is toggled on/off, jump to the leftmost tab so the launch
  // sequence reads left to right (same behavior as SingleCommandBlock's
  // companion tabs).
  const hasMc = !!mc;
  useEffect(() => {
    setTab(hasMc ? "mc_config" : "prefill");
  }, [hasMc]);
  const active = tabs.find((t) => t.id === tab) || tabs[0];
  // Docker mode folds env into `-e` flags inside `docker run` for prefill /
  // decode, so no prelude there. Router (and pip mode) keep the export-style
  // env lines — router is `vllm-router` regardless of install mode.
  const prelude = isDocker && !active.isRouter ? "" : envToExports(active.env);
  const fullScript = prelude ? `${prelude}\n\n${active.command}` : active.command;
  return (
    <div>
      <div className="px-4 pt-3 pb-1">
        {statusHeader || (
          <span className="text-[11px] text-[var(--command-fg)]/55 font-mono">
            {configSummary}
          </span>
        )}
      </div>
      <div className="flex items-center justify-between px-4 pt-2 gap-3">
        <CommandTabs tabs={tabs} current={active.id} onSelect={setTab} />
        <div className="flex items-center gap-1.5 shrink-0">
          <CopyButton text={fullScript} />
          <PopoverButton label="cURL" code={verifyCmd} icon={Terminal} disabled={!active.isRouter} disabledNote="Clients connect via the Router tab — cURL & Bench live there." />
          <PopoverButton label="Bench" code={benchCmd} icon={Gauge} disabled={!active.isRouter} disabledNote="Clients connect via the Router tab — cURL & Bench live there." />
          {endpointsControls}
        </div>
      </div>
      {active.isRouter && active.install && (
        <div className="px-4 pt-3 text-[11px] text-[var(--command-fg)]/50 font-mono leading-snug">
          # Install: {active.install}
        </div>
      )}
      {active.description && (
        <div className="px-4 pt-3 text-[11px] text-[var(--command-fg)]/55 font-mono leading-snug">
          # {active.description}
        </div>
      )}
      {!active.isRouter && active.meta && (
        <div className="px-4 pt-3 text-[11px] text-[var(--command-fg)]/55 font-mono leading-snug">
          # {active.meta.nodes === 0
            ? `Co-located half-node · ${active.meta.poolGpus} GPU · ${active.meta.parallelism.toUpperCase()}`
            : `${active.meta.nodes} node${active.meta.nodes === 1 ? "" : "s"} × ${(active.meta.poolGpus / Math.max(1, active.meta.nodes))} GPU = ${active.meta.poolGpus} GPU · ${active.meta.parallelism.toUpperCase()}`}
          {active.meta.parallelism === "dep" && active.meta.nodes > 1 && (
            <> {" · DP="}{active.meta.dpSize}{" (dp_local="}{active.meta.dpLocal}{", one vllm serve per node)"}</>
          )}
        </div>
      )}
      {!active.isRouter && active.meta?.nodes > 1 && onRankChange && (
        <div className="px-4 pt-2 pb-0 flex items-center gap-2 text-[11px] text-[var(--command-fg)]/70 flex-wrap">
          <span className="font-mono uppercase tracking-wider text-[var(--command-fg)]/50">node</span>
          {/* Display node index is 1-based; the emitted rank flag stays 0-based
              to match vLLM's convention — --data-parallel-start-rank for DEP,
              --node-rank for TP. */}
          <input
            type="number"
            min={1}
            max={active.meta.nodes}
            aria-label="Node whose command is shown"
            value={(active.meta.currentNode ?? 0) + 1}
            onChange={(e) => {
              const n = parseInt(e.target.value, 10);
              if (Number.isFinite(n)) onRankChange(active.id, n - 1);
            }}
            className="w-14 px-2 py-0.5 text-xs font-mono tabular-nums rounded border border-[var(--command-fg)]/20 bg-transparent text-[var(--command-fg)] focus:outline-none focus:border-vllm-blue/60"
          />
          {active.meta.parallelism === "dep" ? (
            <>
              <span className="text-[var(--command-fg)]/40">
                of 1..{active.meta.nodes} · start_rank = {active.meta.startRank}
              </span>
              <span className="text-[var(--command-fg)]/40 ml-auto">vLLM spawns {active.meta.dpLocal} local DP ranks per node</span>
            </>
          ) : (
            <>
              <span className="text-[var(--command-fg)]/40">
                of 1..{active.meta.nodes} · --node-rank = {active.meta.currentNode ?? 0}
              </span>
              <span className="text-[var(--command-fg)]/40 ml-auto">
                {(active.meta.currentNode ?? 0) === 0 ? "head node — serves HTTP + NIXL" : "follower — runs --headless"}
              </span>
            </>
          )}
        </div>
      )}
      {prelude && (
        <pre className="px-4 pt-3 pb-1 text-[12px] text-[var(--command-fg)]/70 font-mono leading-relaxed whitespace-pre overflow-x-auto">
          {prelude}
        </pre>
      )}
      <pre className="px-4 py-3 text-[13px] text-[var(--command-fg)] font-mono leading-relaxed whitespace-pre overflow-x-auto">
        {active.command}
      </pre>
    </div>
  );
}

function KvStoreLbBlock({ result, verifyCmd, benchCmd, statusHeader, onInstanceChange, installMode, dockerMeta, configSummary, endpointsControls }) {
  // Tabs in launch order — Mooncake Master · (centralized only) Mooncake
  // Store · vLLM Serve (config heredoc + env + serve — one paste-runnable
  // script per instance node; a Worker tab appears when instances span >1
  // node) · Router last (it needs the backends up, mirroring PD). Only the
  // vLLM command gets docker-wrapped; router / master / store are separate
  // binaries and render as-is with their pip-install hints.
  const [tab, setTab] = useState("config");
  const isDocker = installMode === "docker";
  const wrap = (cmd, env) =>
    isDocker
      ? dockerMeta.isAscend
        ? buildAscendDockerRun({ command: cmd, env, image: dockerMeta.image, gpuFlags: dockerMeta.gpuFlags })
        : buildDockerRun({ command: cmd, env, image: dockerMeta.image, gpuFlags: dockerMeta.gpuFlags })
      : cmd;

  const instances = result.instances || 2;
  const nodesPer = result.nodeCount || 1;
  // All config-file writing is centralized on ONE "Mooncake Config" tab
  // (launch step 0): it exports both path vars with sane defaults, then
  // writes the files via UNQUOTED heredocs — any $VAR left unfilled resolves
  // from the paster's shell, and unset vars expand to "" (exactly the
  // auto-select default for $MOONCAKE_DEVICE_NAME). Every other tab just
  // references the files through the env vars instead of repeating heredocs.
  // Sizing/NIC notes from the YAML render as leading # lines.
  const configsCommand = buildMooncakeConfigsCommand(
    result.mooncakeConfig, result.mooncakeConfigNote, result.store,
  );

  const requires = result.vllm.install ? ` Requires: ${result.vllm.install}` : "";
  const currentInstance = result.currentInstance ?? 0;
  // Node-tab naming matches MultiNodeBlock / PD ("Head" / "Node 1") so
  // multi-node reads the same everywhere; single-node instances keep the
  // plain "vLLM Serve" tab. Router (absent at 1 instance) stays last.
  const tabs = [
    {
      id: "config", label: "Mooncake Config", command: configsCommand, env: {},
      description: `Run once on every node (instances${result.store ? " + store node" : ""}) before the steps to the right — they read these files via $MOONCAKE_CONFIG_PATH${result.store ? " / $MOONCAKE_STORE_CONFIG_PATH" : ""}.`,
    },
    { id: "master", label: "Mooncake Master", command: result.master.command, env: {}, description: result.master.description },
    ...(result.store ? [{ id: "store", label: "Mooncake Store", command: result.store.command, env: {}, description: result.store.description }] : []),
    {
      id: "vllm", label: nodesPer > 1 ? "Head" : "vLLM Serve", isVllm: true,
      command: wrap(result.vllm.command, result.vllm.env), env: result.vllm.env,
      description: nodesPer > 1
        ? `Head node of each instance (each instance spans ${nodesPer} nodes).${requires}`
        : instances > 1
          ? `One per instance node.${requires}`
          : `Single instance — clients connect to it directly.${requires}`,
    },
    ...(result.vllm.workerCommand ? [{
      id: "vllm_worker", label: "Node 1", isVllm: true,
      command: wrap(result.vllm.workerCommand, result.vllm.env), env: result.vllm.env,
      description: `Nodes 2..${nodesPer} of each instance (rank > 0, --headless).`,
    }] : []),
    ...(result.router ? [{ id: "router", label: "Router", command: result.router.command, env: {}, description: `LB across ${instances} vLLM instances. Install: ${result.router.install}` }] : []),
  ].map((t, i) => ({ ...t, step: i + 1 }));
  const active = tabs.find((t) => t.id === tab) || tabs[0];
  const prelude = isDocker ? "" : envToExports(active.env);
  const fullScript = prelude ? `${prelude}\n\n${active.command}` : active.command;
  // cURL/Bench live where clients connect: the router tab, or the vLLM tab
  // when a single instance serves directly. On other tabs they render
  // disabled (not hidden) so the reader learns where they went.
  const clientTabId = result.router ? "router" : "vllm";
  // Must match the vLLM tab's actual label — it reads "Head" when each
  // instance spans multiple nodes.
  const clientTabLabel = result.router ? "Router" : nodesPer > 1 ? "Head" : "vLLM Serve";
  const isClientTab = active.id === clientTabId;

  return (
    <div>
      <div className="px-4 pt-3 pb-1">
        {statusHeader || (
          <span className="text-[11px] text-[var(--command-fg)]/55 font-mono">
            {configSummary}
          </span>
        )}
      </div>
      <div className="flex items-center justify-between px-4 pt-2 gap-3">
        <CommandTabs tabs={tabs} current={active.id} onSelect={setTab} />
        <div className="flex items-center gap-1.5 shrink-0">
          <CopyButton text={fullScript} />
          <PopoverButton label="cURL" code={verifyCmd} icon={Terminal} disabled={!isClientTab} disabledNote={`Clients connect via the ${clientTabLabel} tab — cURL & Bench live there.`} />
          <PopoverButton label="Bench" code={benchCmd} icon={Gauge} disabled={!isClientTab} disabledNote={`Clients connect via the ${clientTabLabel} tab — cURL & Bench live there.`} />
          {endpointsControls}
        </div>
      </div>
      {active.description && (
        <div className="px-4 pt-3 text-[11px] text-[var(--command-fg)]/55 font-mono leading-snug">
          # {active.description}
        </div>
      )}
      {active.isVllm && instances > 1 && onInstanceChange && (
        <div className="px-4 pt-2 pb-0 flex items-center gap-2 text-[11px] text-[var(--command-fg)]/70 flex-wrap">
          <span className="font-mono uppercase tracking-wider text-[var(--command-fg)]/50">instance</span>
          {/* Display index is 1-based to match the router's $VLLM_INSTANCE_N. */}
          <input
            type="number"
            min={1}
            max={instances}
            aria-label="Instance whose command is shown"
            value={currentInstance + 1}
            onChange={(e) => {
              const n = parseInt(e.target.value, 10);
              if (Number.isFinite(n)) onInstanceChange(n - 1);
            }}
            className="w-14 px-2 py-0.5 text-xs font-mono tabular-nums rounded border border-[var(--command-fg)]/20 bg-transparent text-[var(--command-fg)] focus:outline-none focus:border-vllm-blue/60"
          />
          <span className="text-[var(--command-fg)]/40">
            of 1..{instances} — runs on $VLLM_INSTANCE_{currentInstance + 1}
          </span>
        </div>
      )}
      {prelude && (
        <pre className="px-4 pt-3 pb-1 text-[12px] text-[var(--command-fg)]/70 font-mono leading-relaxed whitespace-pre overflow-x-auto">
          {prelude}
        </pre>
      )}
      <pre className="px-4 py-3 text-[13px] text-[var(--command-fg)] font-mono leading-relaxed whitespace-pre overflow-x-auto">
        {active.command}
      </pre>
    </div>
  );
}
