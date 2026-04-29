"use client";

import { useMemo, useCallback } from "react";
import Link from "next/link";
import { useSearchParams, useRouter } from "next/navigation";
import { Layers, Cpu, X, ArrowDownUp, Type, Eye, Sparkles, Hash, SlidersHorizontal, ChevronDown } from "lucide-react";
import { getProviderLogo, getProviderLogoClass, getProviderDisplayName } from "@/lib/providers";

// Per-row decorations: icon (tasks/arch) or colored dot (precision/hardware).
// Color is family-grouped — precision tiers, GPU brands — so the eye can
// scan a row by hue without reading every label.
const TASK_META = {
  text:       { icon: Type,     iconClass: "text-sky-500 dark:text-sky-400" },
  multimodal: { icon: Eye,      iconClass: "text-violet-500 dark:text-violet-400" },
  omni:       { icon: Sparkles, iconClass: "text-amber-500 dark:text-amber-400" },
  embedding:  { icon: Hash,     iconClass: "text-cyan-500 dark:text-cyan-400" },
};
const ARCH_META = {
  moe:   { icon: Layers, iconClass: "text-violet-500 dark:text-violet-400", label: "MoE" },
  dense: { icon: Cpu,    iconClass: "text-sky-500 dark:text-sky-400",       label: "Dense" },
};
// Precision dot — colored by precision tier so `bf16/fp8/fp4/int*` form
// visually distinct families. The three FP4 variants (plain / NVIDIA /
// Microscaling) share a warm-magenta family but hue-shift between
// pink/fuchsia/rose so a quick scan can tell them apart — they were
// indistinguishable when all three sat on the same pink. Same logic for
// INT8 vs INT4 in the emerald family.
const PRECISION_DOT = {
  bf16:  "w-2 h-2 bg-blue-500",
  fp8:   "w-2 h-2 bg-amber-500",
  fp4:   "w-2 h-2 bg-pink-500",
  nvfp4: "w-2 h-2 bg-fuchsia-500",
  mxfp4: "w-2 h-2 bg-rose-500",
  int8:  "w-2 h-2 bg-emerald-500",
  int4:  "w-2 h-2 bg-green-600",
};
// "120B" -> 120, "1.2B" -> 1.2, "1T" -> 1000, "" -> 0.
// `parameter_count` is a free-form string in YAMLs; this normalises to
// billions for sorting and bucket assignment.
function paramsToB(s) {
  if (!s || typeof s !== "string") return 0;
  const m = s.trim().match(/^([\d.]+)\s*([BTM])?$/i);
  if (!m) return 0;
  const n = parseFloat(m[1]);
  const u = (m[2] || "B").toUpperCase();
  if (u === "T") return n * 1000;
  if (u === "M") return n / 1000;
  return n;
}

// Bucket boundaries (in billions). Order matters — first match wins.
// `dot` grows from xs→xl so the chip's leading mark itself tells you the
// bucket size at a glance — bigger model, bigger dot. Saturation also
// climbs so xl reads strongest. All dots stay in the vllm-blue family
// to avoid competing with the precision row's family colors.
const SIZE_BUCKETS = [
  { id: "xs", label: "<10B",     test: (b) => b > 0 && b < 10,           dot: "w-1.5 h-1.5 bg-vllm-blue/40" },
  { id: "s",  label: "10–70B",   test: (b) => b >= 10 && b < 70,         dot: "w-2 h-2 bg-vllm-blue/55" },
  { id: "m",  label: "70–200B",  test: (b) => b >= 70 && b < 200,        dot: "w-2.5 h-2.5 bg-vllm-blue/70" },
  { id: "l",  label: "200B–1T",  test: (b) => b >= 200 && b < 1000,      dot: "w-3 h-3 bg-vllm-blue/85" },
  { id: "xl", label: "≥1T",      test: (b) => b >= 1000,                 dot: "w-3.5 h-3.5 bg-vllm-blue" },
];

const TASK_OPTIONS = ["text", "multimodal", "omni", "embedding"];
const ARCH_OPTIONS = ["moe", "dense"];
const PRECISION_OPTIONS = ["bf16", "fp8", "fp4", "nvfp4", "mxfp4", "int4", "int8"];
// Hardware grouped by brand — rendered as stacked sub-rows under the
// Hardware filter so the visual separation is unambiguous. Single source
// of truth: HARDWARE_BY_ID is derived from this for lookups elsewhere
// (active-filter pills, result rows).
const HW_BRANDS = [
  {
    name: "NVIDIA",
    logo: "/providers/nvidia.png",
    items: [
      { id: "h100", label: "H100" },
      { id: "h200", label: "H200" },
      { id: "b200", label: "B200" },
      { id: "b300", label: "B300" },
      { id: "gb200", label: "GB200" },
      { id: "gb300", label: "GB300" },
    ],
  },
  {
    name: "AMD",
    logo: "/providers/amd.png",
    items: [
      { id: "mi300x", label: "MI300X" },
      { id: "mi325x", label: "MI325X" },
      { id: "mi355x", label: "MI355X" },
    ],
  },
  {
    name: "Google",
    logo: "/providers/Google.png",
    items: [
      { id: "trillium", label: "TPU v6e" },
      { id: "ironwood", label: "TPU v7" },
    ],
  },
];

const HARDWARE_BY_ID = Object.fromEntries(
  HW_BRANDS.flatMap((b) => b.items.map((it) => [it.id, it]))
);

const SORT_OPTIONS = [
  { id: "released", label: "Newest" },
  { id: "updated", label: "Recently updated" },
  { id: "size_desc", label: "Largest" },
  { id: "size_asc", label: "Smallest" },
  { id: "name", label: "Name" },
];

export function BrowseList({ recipes }) {
  const searchParams = useSearchParams();
  const router = useRouter();

  // Single useMemo so the Sets keep stable identity across renders. Without
  // this, downstream useMemos that depend on tasks/archs/etc. invalidate on
  // every render (Set identity changes even if URL didn't).
  const { tasks, archs, sizes, precisions, hardware, provider, sort, q } = useMemo(() => {
    const setOf = (k) => new Set((searchParams.get(k) || "").split(",").filter(Boolean));
    return {
      tasks: setOf("task"),
      archs: setOf("arch"),
      sizes: setOf("size"),
      precisions: setOf("precision"),
      hardware: setOf("hw"),
      provider: searchParams.get("provider") || "",
      sort: searchParams.get("sort") || "released",
      q: (searchParams.get("q") || "").trim().toLowerCase(),
    };
  }, [searchParams]);

  // Free-text match used for the `?q=...` query — same field set as the
  // top-bar SearchBox so handing off from search to browse stays predictable.
  // Verified hardware ids enter the haystack so "h100" / "mi300x" / "b200"
  // find recipes by GPU compatibility; "tpu" is added as a synonym when any
  // TPU profile is verified, since the ids (trillium/ironwood) don't carry it.
  const matchesQ = useCallback(
    (r) => {
      if (!q) return true;
      const hwKeys = Object.entries(r.meta?.hardware || {})
        .filter(([, s]) => s === "verified")
        .map(([h]) => h);
      const hwExtra = hwKeys.some((k) => k === "trillium" || k === "ironwood") ? ["tpu"] : [];
      const hay = [
        r.hf_repo,
        r.hf_org,
        r.meta?.title,
        r.meta?.provider,
        r.meta?.description,
        ...(r.meta?.tasks || []),
        r.model?.architecture,
        r.model?.parameter_count,
        r.variant?.precision,
        ...(r.precisions || []),
        ...hwKeys,
        ...hwExtra,
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return hay.includes(q);
    },
    [q]
  );

  const update = useCallback(
    (patch) => {
      const sp = new URLSearchParams(searchParams.toString());
      for (const [k, v] of Object.entries(patch)) {
        if (v === "" || v == null || (Array.isArray(v) && v.length === 0)) {
          sp.delete(k);
        } else if (Array.isArray(v)) {
          sp.set(k, v.join(","));
        } else {
          sp.set(k, v);
        }
      }
      const qs = sp.toString();
      router.replace(qs ? `?${qs}` : window.location.pathname, { scroll: false });
    },
    [router, searchParams]
  );

  const toggle = useCallback(
    (key, value) => {
      const cur = new Set((searchParams.get(key) || "").split(",").filter(Boolean));
      if (cur.has(value)) cur.delete(value);
      else cur.add(value);
      update({ [key]: [...cur] });
    },
    [searchParams, update]
  );

  const removeOne = useCallback(
    (key, value) => {
      if (key === "provider" || key === "q") {
        update({ [key]: "" });
        return;
      }
      const cur = new Set((searchParams.get(key) || "").split(",").filter(Boolean));
      cur.delete(value);
      update({ [key]: [...cur] });
    },
    [searchParams, update]
  );

  const providers = useMemo(() => {
    const orgs = [...new Set(recipes.map((r) => r.hf_org))];
    return orgs
      .map((o) => ({ id: o, label: getProviderDisplayName(o) }))
      .sort((a, b) => a.label.localeCompare(b.label));
  }, [recipes]);

  // Counts shown next to each chip — based on the *other* active filters,
  // so toggling a chip on the same row doesn't suddenly zero it out.
  const counts = useMemo(() => {
    const matchExcept = (r, exclude) => {
      if (!matchesQ(r)) return false;
      if (exclude !== "task" && tasks.size > 0 && !(r.meta.tasks || []).some((t) => tasks.has(t))) return false;
      if (exclude !== "arch" && archs.size > 0 && !archs.has(r.model.architecture)) return false;
      if (exclude !== "size" && sizes.size > 0) {
        const b = paramsToB(r.model.parameter_count);
        const bucket = SIZE_BUCKETS.find((x) => x.test(b))?.id;
        if (!bucket || !sizes.has(bucket)) return false;
      }
      if (exclude !== "precision" && precisions.size > 0 && !(r.precisions || []).some((p) => precisions.has(p))) return false;
      if (exclude !== "hw" && hardware.size > 0) {
        const hw = r.meta.hardware || {};
        const ok = [...hardware].some((h) => hw[h] === "verified");
        if (!ok) return false;
      }
      if (exclude !== "provider" && provider && r.hf_org !== provider) return false;
      return true;
    };
    const tally = (key, getter) => {
      const out = {};
      for (const r of recipes) {
        if (!matchExcept(r, key)) continue;
        const vals = getter(r);
        for (const v of vals) out[v] = (out[v] || 0) + 1;
      }
      return out;
    };
    return {
      task: tally("task", (r) => r.meta.tasks || []),
      arch: tally("arch", (r) => [r.model.architecture]),
      size: tally("size", (r) => {
        const b = paramsToB(r.model.parameter_count);
        const id = SIZE_BUCKETS.find((x) => x.test(b))?.id;
        return id ? [id] : [];
      }),
      precision: tally("precision", (r) => r.precisions || []),
      hw: tally("hw", (r) => Object.entries(r.meta.hardware || {}).filter(([, s]) => s === "verified").map(([h]) => h)),
    };
  }, [recipes, tasks, archs, sizes, precisions, hardware, provider, matchesQ]);

  const filtered = useMemo(() => {
    const out = recipes.filter((r) => {
      if (!matchesQ(r)) return false;
      if (tasks.size > 0 && !(r.meta.tasks || []).some((t) => tasks.has(t))) return false;
      if (archs.size > 0 && !archs.has(r.model.architecture)) return false;
      if (sizes.size > 0) {
        const b = paramsToB(r.model.parameter_count);
        const id = SIZE_BUCKETS.find((x) => x.test(b))?.id;
        if (!id || !sizes.has(id)) return false;
      }
      if (precisions.size > 0 && !(r.precisions || []).some((p) => precisions.has(p))) return false;
      if (hardware.size > 0) {
        const hw = r.meta.hardware || {};
        const ok = [...hardware].some((h) => hw[h] === "verified");
        if (!ok) return false;
      }
      if (provider && r.hf_org !== provider) return false;
      return true;
    });

    const cmp = {
      released: (a, b) => (new Date(b.hf_released || 0) - new Date(a.hf_released || 0)),
      updated: (a, b) => (new Date(b.meta.date_updated || 0) - new Date(a.meta.date_updated || 0)),
      size_desc: (a, b) => paramsToB(b.model.parameter_count) - paramsToB(a.model.parameter_count),
      size_asc: (a, b) => paramsToB(a.model.parameter_count) - paramsToB(b.model.parameter_count),
      name: (a, b) => a.hf_id.localeCompare(b.hf_id),
    }[sort] || ((a, b) => 0);
    return [...out].sort(cmp);
  }, [recipes, tasks, archs, sizes, precisions, hardware, provider, sort, matchesQ]);

  const activeCount =
    tasks.size + archs.size + sizes.size + precisions.size + hardware.size +
    (provider ? 1 : 0) + (q ? 1 : 0);
  const hasFilters = activeCount > 0;

  // Flat list of currently-applied filters for the inline pills shown when
  // the panel is collapsed. Keeps user oriented without forcing a panel
  // expand to see what's selected.
  const activeFilters = useMemo(() => {
    const out = [];
    if (q) out.push({ key: "q", value: q, label: `"${q}"` });
    for (const t of tasks) out.push({ key: "task", value: t, label: t });
    for (const a of archs) out.push({ key: "arch", value: a, label: ARCH_META[a]?.label || a });
    for (const id of sizes) {
      const b = SIZE_BUCKETS.find((x) => x.id === id);
      if (b) out.push({ key: "size", value: id, label: b.label });
    }
    for (const p of precisions) out.push({ key: "precision", value: p, label: p.toUpperCase() });
    for (const h of hardware) {
      const opt = HARDWARE_BY_ID[h];
      if (opt) out.push({ key: "hw", value: h, label: opt.label });
    }
    if (provider) out.push({ key: "provider", value: provider, label: getProviderDisplayName(provider) });
    return out;
  }, [q, tasks, archs, sizes, precisions, hardware, provider]);

  // Panel state persists in the URL (`?panel=open`) so a refresh keeps
  // whatever the user had. Default is closed — applied filters are
  // surfaced via the inline ActivePill row in the status bar, so a shared
  // filtered link doesn't need to expand the heavy panel to be readable.
  const open = searchParams.get("panel") === "open";
  const setOpen = (next) =>
    update({ panel: (typeof next === "function" ? next(open) : next) ? "open" : "" });

  const clearAll = () => router.replace(window.location.pathname + (sort !== "released" ? `?sort=${sort}` : ""), { scroll: false });

  return (
    <div className="space-y-3">
      {/* Status bar — always visible. Result count + Filter toggle + Sort.
          Collapsed by default; the heavy filter panel only renders when the
          user opts in or arrives via a filtered URL. */}
      <div className="flex flex-wrap items-center gap-2 sm:gap-3 px-1">
        <span className="text-sm text-foreground">
          <span className="font-semibold tabular-nums">{filtered.length}</span>
          {hasFilters && (
            <span className="text-muted-foreground"> of {recipes.length}</span>
          )}
          <span className="text-muted-foreground"> {filtered.length === 1 ? "recipe" : "recipes"}</span>
        </span>

        <button
          onClick={() => setOpen((v) => !v)}
          className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-[13px] transition-colors ${
            open
              ? "border-vllm-blue bg-vllm-blue/10 text-foreground ring-1 ring-vllm-blue/30"
              : "border-foreground/20 text-foreground/80 hover:text-foreground hover:border-foreground/40 hover:bg-muted/40"
          }`}
          aria-expanded={open}
        >
          <SlidersHorizontal size={13} />
          <span className="font-medium">Filter</span>
          {activeCount > 0 && (
            <span className="inline-flex items-center justify-center rounded-full bg-vllm-blue text-white text-[10px] font-semibold px-1.5 min-w-[18px] h-[18px] tabular-nums">
              {activeCount}
            </span>
          )}
          <ChevronDown size={13} className={`transition-transform ${open ? "rotate-180" : ""}`} />
        </button>

        {/* Active filter pills — only when collapsed (the open panel makes
            its own selection visible via highlighted chips). */}
        {!open && hasFilters && (
          <div className="flex flex-wrap items-center gap-1.5">
            {activeFilters.map((f) => (
              <ActivePill key={`${f.key}:${f.value}`} onRemove={() => removeOne(f.key, f.value)}>
                {f.label}
              </ActivePill>
            ))}
          </div>
        )}

        <div className="flex items-center gap-2 ml-auto">
          <ArrowDownUp size={13} className="text-muted-foreground" />
          <select
            value={sort}
            onChange={(e) => update({ sort: e.target.value === "released" ? "" : e.target.value })}
            className="rounded-lg border border-foreground/20 bg-background px-3 py-1.5 text-[13px] focus:outline-none focus:ring-2 focus:ring-vllm-blue/40 hover:border-foreground/40 transition-colors"
          >
            {SORT_OPTIONS.map((s) => (
              <option key={s.id} value={s.id}>{s.label}</option>
            ))}
          </select>
          {hasFilters && (
            <button
              onClick={clearAll}
              className="inline-flex items-center gap-1 rounded-lg border border-foreground/20 px-2.5 py-1.5 text-[13px] text-muted-foreground hover:text-foreground hover:border-foreground/40 hover:bg-muted/40 transition-colors"
            >
              <X size={12} /> Clear
            </button>
          )}
        </div>
      </div>

      {open && (
      <div className="rounded-xl border border-foreground/15 divide-y divide-foreground/10 bg-card/40">
        <FilterRow label="Task">
          <PillGroup>
            {TASK_OPTIONS.map((t) => {
              const m = TASK_META[t] || {};
              return (
                <Chip
                  key={t}
                  icon={m.icon}
                  iconClass={m.iconClass}
                  active={tasks.has(t)}
                  count={counts.task[t]}
                  onClick={() => toggle("task", t)}
                >
                  {t}
                </Chip>
              );
            })}
          </PillGroup>
        </FilterRow>

        <FilterRow label="Arch">
          <PillGroup>
            {ARCH_OPTIONS.map((a) => {
              const m = ARCH_META[a];
              return (
                <Chip
                  key={a}
                  icon={m.icon}
                  iconClass={m.iconClass}
                  active={archs.has(a)}
                  count={counts.arch[a]}
                  onClick={() => toggle("arch", a)}
                >
                  {m.label}
                </Chip>
              );
            })}
          </PillGroup>
        </FilterRow>

        <FilterRow label="Size">
          <PillGroup>
            {SIZE_BUCKETS.map((b) => (
              <Chip
                key={b.id}
                dot={b.dot}
                active={sizes.has(b.id)}
                count={counts.size[b.id]}
                onClick={() => toggle("size", b.id)}
                mono
              >
                {b.label}
              </Chip>
            ))}
          </PillGroup>
        </FilterRow>

        <FilterRow label="Precision">
          <PillGroup>
            {PRECISION_OPTIONS.map((p) => (
              <Chip
                key={p}
                dot={PRECISION_DOT[p]}
                active={precisions.has(p)}
                count={counts.precision[p]}
                onClick={() => toggle("precision", p)}
                mono
              >
                {p}
              </Chip>
            ))}
          </PillGroup>
        </FilterRow>

        <FilterRow label="Hardware">
          <div className="space-y-2.5">
            {HW_BRANDS.map((brand) => (
              <div key={brand.name} className="flex flex-wrap items-center gap-3">
                <span className="inline-flex items-center gap-2 w-28 shrink-0">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={brand.logo} alt="" width={20} height={20} className="rounded shrink-0" aria-hidden />
                  <span className="text-[12px] font-semibold uppercase tracking-wider text-foreground/80">
                    {brand.name}
                  </span>
                </span>
                <PillGroup>
                  {brand.items.map((opt) => (
                    <Chip
                      key={opt.id}
                      active={hardware.has(opt.id)}
                      count={counts.hw[opt.id]}
                      onClick={() => toggle("hw", opt.id)}
                      mono
                    >
                      {opt.label}
                    </Chip>
                  ))}
                </PillGroup>
              </div>
            ))}
          </div>
        </FilterRow>

        <FilterRow label="Provider">
          <select
            value={provider}
            onChange={(e) => update({ provider: e.target.value })}
            className="rounded-lg border border-foreground/20 bg-background px-3 py-2 text-[13px] focus:outline-none focus:ring-2 focus:ring-vllm-blue/40 hover:border-foreground/40 transition-colors"
          >
            <option value="">All providers</option>
            {providers.map((p) => (
              <option key={p.id} value={p.id}>{p.label}</option>
            ))}
          </select>
        </FilterRow>
      </div>
      )}

      {/* List */}
      {filtered.length === 0 ? (
        <div className="rounded-xl border border-dashed border-border p-10 text-center text-sm text-muted-foreground">
          No recipes match these filters. <button onClick={clearAll} className="text-vllm-blue hover:underline">Clear all</button>
        </div>
      ) : (
        <div className="rounded-xl border border-border overflow-hidden">
          {/* Header — desktop only */}
          <div className="hidden md:grid grid-cols-[1fr_72px_64px_72px_140px_1fr] gap-3 px-4 py-2 border-b border-border bg-muted/30 text-[10px] font-semibold text-muted-foreground uppercase tracking-widest">
            <div>Model</div>
            <div>Size</div>
            <div>Arch</div>
            <div>Precision</div>
            <div>Verified on</div>
            <div>Notes</div>
          </div>
          <ul className="divide-y divide-border">
            {filtered.map((r) => <Row key={r.hf_id} recipe={r} />)}
          </ul>
        </div>
      )}
    </div>
  );
}

function FilterRow({ label, children }) {
  // Mirrors ConfigRow in CommandBuilder so the browse page feels like an
  // extension of the recipe page. Label slightly larger here (text-[11px]
  // vs text-[10px]) because this is a dedicated filter surface, not a
  // sidebar packed next to a command card.
  return (
    <div className="px-4 py-3.5 flex flex-col sm:flex-row sm:items-start gap-2 sm:gap-5">
      <div className="text-[11px] font-semibold text-muted-foreground uppercase tracking-widest sm:w-24 sm:pt-2 shrink-0">
        {label}
      </div>
      <div className="flex-1 min-w-0">{children}</div>
    </div>
  );
}

function PillGroup({ children }) {
  return <div className="flex flex-wrap gap-2">{children}</div>;
}

function ActivePill({ children, onRemove }) {
  // Compact pill shown in the status bar when the filter panel is collapsed.
  // Whole pill is the remove affordance — clicking anywhere drops that filter.
  return (
    <button
      type="button"
      onClick={onRemove}
      className="group inline-flex items-center gap-1 rounded-md border border-vllm-blue/40 bg-vllm-blue/10 px-2 py-0.5 text-[12px] text-foreground hover:border-vllm-blue/70 hover:bg-vllm-blue/15 transition-colors"
      aria-label={`Remove ${children} filter`}
    >
      <span className="font-medium">{children}</span>
      <X size={11} className="text-muted-foreground group-hover:text-foreground transition-colors" />
    </button>
  );
}

function Chip({ active, count, onClick, icon: Icon, iconClass, dot, logo, mono, children }) {
  // Borders use foreground/20 instead of --border because dark-mode --border
  // is 10% white, which disappears on the card background.
  // Lead element: icon (Lucide component, color via iconClass), colored
  // dot (Tailwind bg- class), or logo (image URL). Disabled chips desaturate
  // all three so they don't compete with active ones.
  const disabled = !active && !count;
  const style = disabled
    ? "border-dashed border-foreground/15 text-muted-foreground/40 cursor-not-allowed bg-muted/10"
    : active
      ? "border-vllm-blue bg-vllm-blue/10 text-foreground ring-1 ring-vllm-blue/30 shadow-sm"
      : "border-foreground/20 text-foreground/80 hover:text-foreground hover:border-foreground/40 hover:bg-muted/40";
  const leadIconColor = disabled
    ? "text-muted-foreground/30"
    : active
      ? "text-vllm-blue"
      : iconClass || "text-muted-foreground";
  // `dot` is a full Tailwind string (size + color); disabled state forces
  // a neutral gray and a default size to keep alignment consistent.
  const dotClass = disabled ? "w-2 h-2 bg-muted-foreground/30" : dot;
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-2 text-[13px] transition-all ${style}`}
    >
      {Icon && <Icon size={13} className={leadIconColor} aria-hidden />}
      {dot && <span className={`inline-block rounded-full shrink-0 ${dotClass}`} aria-hidden />}
      {logo && (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={logo}
          alt=""
          width={14}
          height={14}
          aria-hidden
          className={`rounded-sm shrink-0 ${disabled ? "opacity-30" : ""}`}
        />
      )}
      <span className={mono ? "font-mono uppercase tracking-wide" : "font-medium capitalize"}>
        {children}
      </span>
      {count != null && (
        <span
          className={`tabular-nums text-[11px] ${
            active ? "text-foreground/60" : "text-muted-foreground/70"
          }`}
        >
          {count}
        </span>
      )}
    </button>
  );
}

function Row({ recipe }) {
  const r = recipe;
  const isMoe = r.model.architecture === "moe";
  const params = r.model.parameter_count || "—";
  const active = r.model.active_parameters;
  const sizeLabel = isMoe && active && active !== params ? `${params}/${active}` : params;
  const logo = getProviderLogo(r.hf_org);
  const verified = Object.entries(r.meta.hardware || {})
    .filter(([, s]) => s === "verified")
    .map(([h]) => h);
  const tasks = r.meta.tasks || [];

  return (
    <li>
      <Link
        href={`/${r.hf_id}`}
        className="group block px-4 py-3 hover:bg-muted/30 transition-colors md:grid md:grid-cols-[1fr_72px_64px_72px_140px_1fr] md:gap-3 md:items-center"
      >
        {/* Model */}
        <div className="flex items-start gap-2.5 min-w-0">
          {logo ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={logo} alt="" width={24} height={24} className={`rounded shrink-0 mt-0.5 ${getProviderLogoClass(r.hf_org)}`} />
          ) : (
            <div className="w-6 h-6 rounded bg-muted flex items-center justify-center text-[10px] font-bold text-muted-foreground shrink-0">
              {r.hf_org.charAt(0)}
            </div>
          )}
          <div className="min-w-0">
            <div className="font-medium text-sm group-hover:text-vllm-blue transition-colors truncate">
              {r.hf_repo}
            </div>
            <div className="font-mono text-[10px] text-muted-foreground truncate">
              {r.hf_org}
              {tasks.length > 0 && <span className="ml-2">· {tasks.join(" · ")}</span>}
            </div>
          </div>
        </div>

        {/* Size */}
        <div className="hidden md:flex items-center gap-1 text-xs">
          {isMoe ? <Layers size={10} className="text-muted-foreground" /> : <Cpu size={10} className="text-muted-foreground" />}
          <span className="font-mono">{sizeLabel}</span>
        </div>

        {/* Arch */}
        <div className="hidden md:block text-xs text-muted-foreground">
          {isMoe ? "MoE" : "Dense"}
        </div>

        {/* Precision */}
        <div className="hidden md:block text-xs font-mono uppercase">
          {r.variant?.precision || "—"}
        </div>

        {/* Verified on */}
        <div className="hidden md:flex flex-wrap gap-1">
          {verified.length === 0 ? (
            <span className="text-[10px] text-muted-foreground/50">—</span>
          ) : (
            verified.map((h) => (
              <span key={h} className="inline-block rounded border border-border px-1 py-0 text-[10px] font-mono text-muted-foreground">
                {HARDWARE_BY_ID[h]?.label || h}
              </span>
            ))
          )}
        </div>

        {/* Notes */}
        <div className="hidden md:block text-xs text-muted-foreground line-clamp-2 leading-snug">
          {r.meta.performance_headline || r.meta.description}
        </div>

        {/* Mobile: spec strip */}
        <div className="md:hidden mt-1.5 flex flex-wrap gap-1.5 text-[11px] text-muted-foreground">
          <span className="font-mono">{sizeLabel}</span>
          <span>·</span>
          <span>{isMoe ? "MoE" : "Dense"}</span>
          {r.variant?.precision && (<><span>·</span><span className="font-mono uppercase">{r.variant.precision}</span></>)}
          {verified.length > 0 && (
            <>
              <span>·</span>
              <span className="font-mono">{verified.map((h) => HARDWARE_BY_ID[h]?.label || h).join(" ")}</span>
            </>
          )}
        </div>
      </Link>
    </li>
  );
}
