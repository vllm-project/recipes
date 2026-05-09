"use client";

import { useState, useMemo, useCallback, useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { useSearchParams, useRouter, usePathname } from "next/navigation";
import { Copy, Check, Terminal, Gauge, Sparkles, ChevronDown, Package, Info, Zap, Globe, Wrench, Brain } from "lucide-react";
import { resolveCommand, recommendStrategy, isPrecisionCompatible, isHardwareSupported, fitsSingleNode, pickDefaultHardware, resolveSingleNodeTp, computeDockerMeta, buildDockerRun } from "@/lib/command-synthesis";
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

function PopoverButton({ label, code, icon: Icon }) {
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
function EndpointsPopoverButton({ isPd, isMultiNode, placeholders, endpoints, onChange, onReset }) {
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

  const clientHostKey = isPd ? "ROUTER_HOST" : (isMultiNode ? "HEAD_IP" : "VLLM_HOST");
  const clientPortKey = isPd ? "ROUTER_PORT" : "VLLM_PORT";
  const clientHostHint = "localhost";
  const clientPortHint = isPd ? "30000" : "8000";

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
    return pickDefaultHardware(taxonomy.hardware_profiles, v, recipe);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recipe, taxonomy]);

  const [hwId, setHwId] = useState(searchParams.get("hardware") || defaultHw);

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
    if (!searchParams.get("hardware") && prefs.hardware) {
      const v = recipe.variants?.[variant] || recipe.variants?.default || {};
      const prefProfile = taxonomy.hardware_profiles?.[prefs.hardware];
      if (prefProfile?.brand === "NVIDIA" && isPrecisionCompatible(prefProfile, v) && isHardwareSupported(recipe, prefs.hardware)) {
        setHwId(prefs.hardware);
        restoredFitsHw = prefProfile;
      }
    }

    const rs = loadRecipeState(recipe.hf_id);
    if (!searchParams.get("strategy") && rs.strategy &&
        (recipe.compatible_strategies || []).includes(rs.strategy)) {
      setStrategyOverride(rs.strategy);
    }
    if (!searchParams.get("features") && Array.isArray(rs.features)) {
      const recipeFeatures = Object.keys(recipe.features || {});
      setFeatures(rs.features.filter((f) => recipeFeatures.includes(f)));
    }
    // Nodes: prefer the saved value; otherwise auto-bump if the resolved
    // hardware can't fit single-node (mirrors `setHw`'s bump).
    if (!searchParams.get("nodes") && supportsMultiNode) {
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

  // Recipes without a multi_node_* / pd_cluster strategy can't scale beyond one
  // node — force nodeCount to 1 in that case, even if the URL says otherwise.
  const supportsMultiNode = (recipe.compatible_strategies || []).some(
    (s) => s.startsWith("multi_node_") || s === "pd_cluster"
  );
  const [nodeCount, setNodeCount] = useState(() => {
    if (!supportsMultiNode) return 1;
    const urlN = searchParams.get("nodes");
    if (urlN) {
      const n = parseInt(urlN, 10);
      return [1, 2].includes(n) ? n : 1;
    }
    // No URL pin: start on multi-node when the initial hardware can't fit
    // single-node. Same fit check the hardware-change handler runs.
    const initialHwId = searchParams.get("hardware") || defaultHw;
    const initialHw = taxonomy.hardware_profiles?.[initialHwId];
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
    const pickNodes = (role, fallback) => {
      const v = so[role]?.nodes;
      if (typeof v === "number") return v;
      if (v && typeof v === "object") {
        if (typeof v[hwId] === "number") return v[hwId];
        if (typeof v.default === "number") return v.default;
      }
      return fallback;
    };
    return {
      prefill: pickNodes("prefill", 1),
      decode: pickNodes("decode", 1),
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recipe, hwId]);
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
  const [strategyOverride, setStrategyOverride] = useState(searchParams.get("strategy") || "");
  // Default-on features = (all features) − (recipe.opt_in_features) − (recipe.hardware_opt_in_features[hwId]).
  // The per-hw override lets a recipe suppress a feature's default on specific
  // hardware (e.g. GB200's 4-GPU trays make --mm-encoder-tp-mode data unnecessary).
  const defaultFeaturesFor = useCallback(
    (hw) => {
      const optIn = new Set(recipe.opt_in_features || []);
      for (const f of recipe.hardware_opt_in_features?.[hw] || []) optIn.add(f);
      return Object.keys(recipe.features || {}).filter((f) => !optIn.has(f));
    },
    [recipe]
  );

  // Encode a features array for the URL: returns "" (which syncUrl deletes)
  // when the set matches the YAML default for `hw`, so links stay clean unless
  // the user actually deviated.
  const featuresToUrl = useCallback(
    (arr, hw) => {
      const want = new Set(defaultFeaturesFor(hw));
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
    return defaultFeaturesFor(urlHw);
  });

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
    const NVIDIA_ORDER = ["h100", "h200", "b200", "gb200", "b300", "gb300"];
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

  const recommended = useMemo(() => recommendStrategy(recipe, hwProfile, nodeCount), [recipe, hwProfile, nodeCount]);

  const compatibleStrategies = useMemo(() => {
    return (recipe.compatible_strategies || []).filter((s) => {
      const strat = strategies[s];
      if (!strat) return false;
      if (nodeCount === 1 && strat.deploy_type === "multi_node") return false;
      if (nodeCount > 1 && strat.deploy_type === "single_node") return false;
      return true;
    });
  }, [recipe, strategies, nodeCount]);

  // PD now sizes each pool independently, so the "2× model VRAM on one node"
  // concern that used to invalidate pd_cluster on small GPUs no longer applies.
  const activeStrategy = strategyOverride || recommended;

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
          prefill: { nodes: pdPrefillNodes, rank: pdPrefillRank },
          decode: { nodes: pdDecodeNodes, rank: pdDecodeRank },
        }
        : null;
      return resolveCommand(recipe, variant, activeStrategy, hwId, features, strategies, taxonomy, advArgs, nodeCount, pdNodes);
    },
    [recipe, variant, activeStrategy, hwId, features, advanced, advancedById, strategies, taxonomy, nodeCount, pdPrefillNodes, pdDecodeNodes, pdPrefillRank, pdDecodeRank]
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
      const next = pickDefaultHardware(taxonomy.hardware_profiles, v, recipe);
      setHwId(next);
      syncUrl({ hardware: next });
    }
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
    // Bump to multi-node if the new hardware can't fit single-node (otherwise
    // the Single-node pill shows crossed out but the command keeps rendering
    // the invalid single-node config). Bump back DOWN to single-node when the
    // new hardware comfortably fits and the recipe's default is a single-node
    // strategy — without this, switching from GB200 (which bumped to 2 nodes
    // because the model didn't fit a 4-GPU tray) to B300/GB300 would stay at
    // 2 nodes and pick the multi-node sibling. Tied to the click so a
    // deliberate Single-/Multi-node click afterwards still wins.
    const newProfile = taxonomy.hardware_profiles?.[id] || {};
    const fitsNew = fitsSingleNode(newProfile, currentVariant);
    const recipeDefault = recipe.default_strategy;
    const recipeDefaultsSingleNode =
      typeof recipeDefault === "string" && recipeDefault.startsWith("single_node_");
    const shouldBumpNodes = nodeCount === 1 && supportsMultiNode && !fitsNew;
    const shouldUnbumpNodes = nodeCount > 1 && fitsNew && recipeDefaultsSingleNode;
    if (shouldBumpNodes) setNodeCount(2);
    if (shouldUnbumpNodes) setNodeCount(1);
    syncUrl({
      hardware: id,
      strategy: "",
      nodes: shouldBumpNodes ? "2" : shouldUnbumpNodes ? "" : undefined,
      features: featuresToUrl(next, id),
    });
    savePreference("hardware", id);
    // Mirror the new state to per-recipe storage so a hardware switch
    // doesn't leave stale strategy/nodes/features cached for this recipe.
    saveRecipeState(recipe.hf_id, {
      strategy: undefined,
      nodes: shouldBumpNodes ? 2 : shouldUnbumpNodes ? 1 : nodeCount,
      features: next,
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

  const toggleFeature = (f) => {
    // text_only (skip vision encoder) and encoder_parallel (DP the encoder)
    // are mutually exclusive — enabling one clears the other.
    const mutex = { text_only: "encoder_parallel", encoder_parallel: "text_only" };
    const on = !features.includes(f);
    const next = on
      ? [...features.filter((x) => x !== mutex[f]), f]
      : features.filter((x) => x !== f);
    setFeatures(next);
    syncUrl({ features: featuresToUrl(next, hwId) });
    saveRecipeState(recipe.hf_id, { features: next });
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

  // curl/bench target. PD → router host:port; everyone else → the vllm-serve
  // node (head node for multi-node TP). Defaults to localhost so the
  // single-node demo case still works copy-paste; user can fill the
  // Cluster endpoints panel to point at a real cluster.
  const clientHostKey = isPd ? "ROUTER_HOST" : (isMultiNode ? "HEAD_IP" : "VLLM_HOST");
  const clientPortKey = isPd ? "ROUTER_PORT" : "VLLM_PORT";
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
    if (result.headCommand) texts.push(result.headCommand);
    if (result.workerCommand) texts.push(result.workerCommand);
    if (result.prefill?.command) texts.push(result.prefill.command);
    if (result.decode?.command) texts.push(result.decode.command);
    if (result.router?.command) texts.push(result.router.command);
    for (const e of [result.env, result.prefill?.env, result.decode?.env]) {
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
    } else if (result.deployType === "multi_node") {
      defaults.VLLM_PORT = "8000";
    } else {
      defaults.VLLM_HOST = "localhost";
      defaults.VLLM_PORT = "8000";
    }
    return { ...defaults, ...endpoints };
  }, [result.deployType, endpoints]);

  const displayedResult = useMemo(() => {
    const sub = (s) => substitute(s, effectiveEndpoints);
    if (result.deployType === "pd_cluster") {
      return {
        ...result,
        prefill: { ...result.prefill, command: sub(result.prefill.command), env: substituteEnv(result.prefill.env, effectiveEndpoints) },
        decode:  { ...result.decode,  command: sub(result.decode.command),  env: substituteEnv(result.decode.env,  effectiveEndpoints) },
        router:  { ...result.router,  command: sub(result.router.command) },
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
    };
  }, [result, effectiveEndpoints]);

  const dependencies = recipe.dependencies || [];

  // Status caption for the command block header.
  // Only `verified` is a positive signal worth surfacing; anything else
  // falls back to a neutral "vllm serve" label (treat as "assumed to work").
  const hwStatus = recipe.meta?.hardware?.[hwId]; // "verified" | undefined
  const hwFullName = hwProfile?.brand
    ? `${hwProfile.brand} ${hwProfile.display_name || hwId}`
    : (hwProfile?.display_name || hwId);
  const statusHeader = hwStatus === "verified" ? (
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
  const hwPart = nodeCount > 1 ? `${nodeCount}× ${hwDisplay}` : hwDisplay;
  const strategyPart = result.deployType === "pd_cluster"
    ? "PD cluster"
    : result.deployType === "multi_node"
      ? (strategies[activeStrategy]?.display_name || activeStrategy)
      : effectiveTp
        ? `TP=${effectiveTp}`
        : (strategies[activeStrategy]?.display_name || activeStrategy);
  const precisionPart = currentVariant.precision?.toUpperCase();
  const configSummary = [hwPart, strategyPart, precisionPart].filter(Boolean).join(" · ");

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

  // Upstream `vllm/vllm-openai:latest` (and recent pinned tags) ship CUDA 13
  // as the base; CUDA 12.9 is the legacy alternate published as a `-cu129`
  // suffix. The recipe's `min_vllm_version` doesn't change which tag the user
  // pulls — `:latest` is always today's base regardless — so the alt suffix
  // is a constant.
  const altCudaSuffix = "cu129";

  const dockerMeta = useMemo(() => {
    const meta = computeDockerMeta(recipe, currentVariant, hwProfile);
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
    if (dockerCudaVariant === altCudaSuffix) {
      return { ...meta, image: `${meta.image}-${altCudaSuffix}` };
    }
    return meta;
  }, [recipe, currentVariant, hwProfile, dockerCudaVariant, altCudaSuffix]);

  // `installMode` carries the user's tab choice; `effectiveInstallMode` folds
  // in constraints that would hide a tab entirely (pip: recipe opt-out or TPU
  // hardware; docker: recipe opt-out). This way switching to TPU flips both
  // the Install tab *and* the rendered command block to docker — they stay
  // in sync without requiring the user to re-click.
  const pipEffectivelyHidden =
    recipe.model?.install?.pip === false || hwProfile?.generation === "tpu";
  const dockerEffectivelyHidden = recipe.model?.install?.docker === false;
  const effectiveInstallMode =
    installMode === "pip" && pipEffectivelyHidden
      ? "docker"
      : installMode === "docker" && dockerEffectivelyHidden
        ? "pip"
        : installMode;

  return (
    <TooltipProvider>
      <div className="space-y-4">
        {/* ── Install (tabs: pip / docker — mirrors the command-card mode toggle
          so switching from either place keeps them in sync). */}
        <InstallBlock
          recipe={recipe}
          dockerMeta={dockerMeta}
          installMode={effectiveInstallMode}
          setInstallMode={setInstallMode}
          dockerCudaVariant={dockerCudaVariant}
          setDockerCudaVariant={setDockerCudaVariant}
          altCudaSuffix={altCudaSuffix}
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
                      // Only `verified` carries a label; everything else = silent default.
                      // `unsupported` = author opt-out for this model; disables the pill.
                      const status = recipe.meta?.hardware?.[id];
                      const isUnsupported = status === "unsupported";
                      // Per-role PD now sizes each pool independently, so hardware
                      // only needs to fit 1× model per node (standard precision
                      // check is enough). The old co-located single-node check
                      // (2× model on one node) is no longer the default UX.
                      const disabled = !precisionOk || isUnsupported;
                      const verifiedNote = status === "verified"
                        ? "\n\nVerified — author has tested this hardware end-to-end"
                        : "";
                      const reason = !precisionOk
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
              {Object.entries(recipe.variants || {}).map(([key, v]) => (
                <Pill
                  key={key}
                  active={variant === key}
                  onClick={() => selectVariant(key)}
                  title={[
                    v.description,
                    `Min ${v.vram_minimum_gb} GB to load — add KV cache for serving. Scale out via multi-node if needed.`,
                  ].filter(Boolean).join("\n\n")}
                >
                  <span className="font-mono font-semibold">{(v.label || v.precision)?.toUpperCase()}</span>
                  <span className="text-muted-foreground ml-1.5 font-mono">{v.vram_minimum_gb} GB</span>
                </Pill>
              ))}
            </PillGroup>
          </ConfigRow>

          {/* Strategy */}
          <ConfigRow label="Strategy">
            <PillGroup>
              {compatibleStrategies.map((s) => {
                return (
                  <Pill
                    key={s}
                    active={activeStrategy === s}
                    onClick={() => selectStrategy(s)}
                    title={strategies[s]?.description}
                  >
                    <span className="font-semibold">{strategies[s]?.display_name || s}</span>
                    {s === recommended && (
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
            {strategies[activeStrategy]?.orientation && (() => {
              const o = strategies[activeStrategy].orientation;
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

          {/* Nodes — two number inputs for PD (one per pool), pills otherwise */}
          {activeStrategy === "pd_cluster" ? (
            <ConfigRow
              label="Nodes"
              hint="Each pool (prefill / decode) sizes independently. Total cluster = prefill_nodes + decode_nodes. For Kimi-K2.5 on GB200 the production pattern is prefill=1, decode=4."
            >
              <div className="flex flex-wrap items-center gap-3 text-sm">
                <PdNodeInput
                  label="Prefill"
                  value={pdPrefillNodes}
                  gpuPerNode={hwProfile.gpu_count || 8}
                  onChange={(n) => setPdNodes("prefill", n)}
                />
                <PdNodeInput
                  label="Decode"
                  value={pdDecodeNodes}
                  gpuPerNode={hwProfile.gpu_count || 8}
                  onChange={(n) => setPdNodes("decode", n)}
                />
                <span className="text-xs text-muted-foreground tabular-nums">
                  total {(pdPrefillNodes + pdDecodeNodes) * (hwProfile.gpu_count || 8)} GPUs
                  · {pdPrefillNodes + pdDecodeNodes} node{pdPrefillNodes + pdDecodeNodes === 1 ? "" : "s"}
                </span>
              </div>
            </ConfigRow>
          ) : (
            <ConfigRow label="Nodes">
              <PillGroup>
                {[1, 2].map((n) => {
                  // Multi-node pill is disabled when the recipe declares no
                  // multi_node_* (or pd_cluster) strategy. Small dense models
                  // commonly omit these.
                  const noMultiNode = n > 1 && !supportsMultiNode;
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
                          ? "This recipe does not declare a multi-node strategy. Fits in a single node."
                          : singleNodeDoesntFit
                            ? `Single-node can't fit this variant on ${hwProfile.display_name || "the selected hardware"} (${currentVariant.vram_minimum_gb}GB > ${hwProfile.vram_gb}GB) — use multi-node`
                            : n === 1
                              ? "Single-node deployment (one HGX box)"
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
                {Object.entries(recipe.features || {}).map(([key, f]) => (
                  <Pill
                    key={key}
                    active={features.includes(key)}
                    onClick={() => toggleFeature(key)}
                    title={f?.description}
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
                    {key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                    {key === "spec_decoding" && (
                      <span className="ml-1.5 text-[11px] text-vllm-yellow font-normal">
                        (for low latency & small batch size)
                      </span>
                    )}
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
  if (name.endsWith("_IP")) return "10.0.0.1";
  return "value";
}

function PdNodeInput({ label, value, gpuPerNode, onChange }) {
  return (
    <label className="inline-flex items-center gap-2">
      <span className="text-xs font-medium text-muted-foreground">{label}</span>
      <input
        type="number"
        min={1}
        max={16}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-14 px-2 py-1 text-sm font-mono tabular-nums rounded-md border border-border bg-background focus:outline-none focus:ring-1 focus:ring-vllm-blue/40"
      />
      <span className="text-xs text-muted-foreground tabular-nums">
        × {gpuPerNode} = {value * gpuPerNode}
      </span>
    </label>
  );
}

function ConfigRow({ label, hint, children }) {
  return (
    <div className="px-4 py-3 flex flex-col sm:flex-row sm:items-start gap-2 sm:gap-4">
      <div className="text-[10px] font-semibold text-muted-foreground uppercase tracking-widest sm:w-20 sm:pt-1.5 shrink-0 inline-flex items-center gap-1">
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
      className={`inline-flex items-center rounded-lg border px-2.5 py-1.5 text-xs transition-all ${style}`}
    >
      {children}
    </button>
  );
  return title ? <InfoTip content={title}>{btn}</InfoTip> : btn;
}

function envToExports(env) {
  return Object.entries(env || {})
    .map(([k, v]) => `export ${k}=${v}`)
    .join("\n");
}

function SingleCommandBlock({ command, env, verifyCmd, benchCmd, statusHeader, installMode, dockerMeta, configSummary, endpointsControls }) {
  const isDocker = installMode === "docker";
  // Docker mode: env vars fold into `-e` flags inside the wrapped `docker run`,
  // so there's no separate prelude (the `docker pull` lives in the Install
  // block tabs above). Pip mode: prelude = `export KEY=VAL` lines.
  const prelude = isDocker ? "" : envToExports(env);
  const displayCommand = isDocker
    ? buildDockerRun({ command, env, image: dockerMeta.image, gpuFlags: dockerMeta.gpuFlags })
    : command;
  const fullScript = prelude ? `${prelude}\n\n${displayCommand}` : displayCommand;
  return (
    <div>
      <div className="flex items-center justify-between px-4 pt-3 gap-3">
        {statusHeader || (
          <span className="text-[11px] text-[var(--command-fg)]/55 font-mono">
            {configSummary}
          </span>
        )}
        <div className="flex items-center gap-1.5">
          <CopyButton text={fullScript} />
          <PopoverButton label="cURL" code={verifyCmd} icon={Terminal} />
          <PopoverButton label="Bench" code={benchCmd} icon={Gauge} />
          {endpointsControls}
        </div>
      </div>
      {prelude && (
        <pre className="px-4 pt-3 pb-1 text-[12px] text-[var(--command-fg)]/70 font-mono leading-relaxed whitespace-pre overflow-x-auto">
          {prelude}
        </pre>
      )}
      <pre className="px-4 py-3 text-[13px] text-[var(--command-fg)] font-mono leading-relaxed whitespace-pre overflow-x-auto">
        {displayCommand}
      </pre>
    </div>
  );
}

function InstallBlock({ recipe, dockerMeta, installMode, setInstallMode, dockerCudaVariant, setDockerCudaVariant, altCudaSuffix }) {
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
  const { isAmd, isTpu, image: dockerImage, brandKey, cudaMap } = dockerMeta;
  const minV = recipe.model?.min_vllm_version;

  // When a recipe's min_vllm_version hasn't shipped yet (cutting-edge models
  // that landed after the last stable release), `model.nightly_required: true`
  // swaps the default pip command to the nightly wheel index and surfaces a
  // pill in the Install header. Manual `install.pip.command` overrides still
  // win — this flag only affects the default.
  const nightlyRequired = recipe.model?.nightly_required === true;
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
  const defaultDockerCmd = `docker pull ${dockerImage}`;
  const dockerCmd = dockerCfg?.command || defaultDockerCmd;
  const defaultDockerNote = isTpu
    ? "TPU builds are published by vllm-project/tpu-inference. See the Trillium and Ironwood tpu-recipes for pinned image tags and exact deployment flags."
    : isAmd
      ? undefined
      : cudaMap
        ? "This recipe ships paired CUDA-tagged images. Pick `cu129` for CUDA 12.9 hosts or `cu130` for CUDA 13."
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
  const dockerLabel = isTpu ? "Docker (TPU)" : isAmd ? "Docker (ROCm)" : "Docker";
  const tabs = [
    !effectivePipHidden && {
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
          vLLM {minV}+ · {isTpu ? "TPU" : isAmd ? "ROCm" : "CUDA"}
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

function MultiNodeBlock({ result, verifyCmd, benchCmd, statusHeader, installMode, dockerMeta, configSummary, endpointsControls }) {
  const [tab, setTab] = useState("head");
  const isDocker = installMode === "docker";
  const wrap = (cmd) =>
    isDocker
      ? buildDockerRun({ command: cmd, env: result.env, image: dockerMeta.image, gpuFlags: dockerMeta.gpuFlags })
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
        <div className="flex gap-0.5 bg-foreground/5 rounded-md p-0.5">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-2.5 py-1 text-xs font-medium rounded transition-colors ${tab === t.id ? "bg-foreground/10 text-[var(--command-fg)]" : "text-[var(--command-fg)]/50 hover:text-[var(--command-fg)]/80"
                }`}
            >
              {t.label}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-1.5">
          <CopyButton text={fullScript} />
          {tab === "head" && (
            <>
              <PopoverButton label="cURL" code={verifyCmd} icon={Terminal} />
              <PopoverButton label="Bench" code={benchCmd} icon={Gauge} />
            </>
          )}
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
      ? buildDockerRun({ command: cmd, env, image: dockerMeta.image, gpuFlags: dockerMeta.gpuFlags })
      : cmd;
  const tabs = [
    { id: "prefill", label: "Prefill", command: wrap(result.prefill.command, result.prefill.env), env: result.prefill.env, meta: result.prefill },
    { id: "decode", label: "Decode", command: wrap(result.decode.command, result.decode.env), env: result.decode.env, meta: result.decode },
    { id: "router", label: "Router", command: result.router.command, env: {}, install: result.router.install, isRouter: true },
  ];
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
        <div className="flex flex-wrap gap-0.5 bg-foreground/5 rounded-md p-0.5">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-2.5 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap ${tab === t.id ? "bg-foreground/10 text-[var(--command-fg)]" : "text-[var(--command-fg)]/50 hover:text-[var(--command-fg)]/80"
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
              <PopoverButton label="cURL" code={verifyCmd} icon={Terminal} />
              <PopoverButton label="Bench" code={benchCmd} icon={Gauge} />
            </>
          )}
          {endpointsControls}
        </div>
      </div>
      {active.isRouter && active.install && (
        <div className="px-4 pt-3 text-[11px] text-[var(--command-fg)]/50 font-mono leading-snug">
          # Dependency: {active.install}
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
      {!active.isRouter && active.meta?.parallelism === "dep" && active.meta.nodes > 1 && onRankChange && (
        <div className="px-4 pt-2 pb-0 flex items-center gap-2 text-[11px] text-[var(--command-fg)]/70 flex-wrap">
          <span className="font-mono uppercase tracking-wider text-[var(--command-fg)]/50">node</span>
          {/* Display node index is 1-based; emitted --data-parallel-start-rank
              stays 0-based to match vLLM's rank convention. */}
          <input
            type="number"
            min={1}
            max={active.meta.nodes}
            value={(active.meta.currentNode ?? 0) + 1}
            onChange={(e) => {
              const n = parseInt(e.target.value, 10);
              if (Number.isFinite(n)) onRankChange(active.id, n - 1);
            }}
            className="w-14 px-2 py-0.5 text-xs font-mono tabular-nums rounded border border-[var(--command-fg)]/20 bg-transparent text-[var(--command-fg)] focus:outline-none focus:border-vllm-blue/60"
          />
          <span className="text-[var(--command-fg)]/40">
            of 1..{active.meta.nodes} · start_rank = {active.meta.startRank}
          </span>
          <span className="text-[var(--command-fg)]/40 ml-auto">vLLM spawns {active.meta.dpLocal} local DP ranks per node</span>
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
