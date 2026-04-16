"use client";

import { useMemo, useState } from "react";
import { useSearchParams, useRouter, usePathname } from "next/navigation";
import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { getProviderLogo } from "@/lib/providers";
import { recipeHref } from "@/lib/recipe-utils";
import { ChevronDown, X } from "lucide-react";

export function RecipeCardGrid({ recipes, taxonomy }) {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();

  const taskFilter = searchParams.get("task") || "";
  const providerFilter = searchParams.get("provider") || "";
  const hasFilters = taskFilter || providerFilter;

  const setFilter = (key, value) => {
    const sp = new URLSearchParams(searchParams.toString());
    if (value) sp.set(key, value);
    else sp.delete(key);
    const qs = sp.toString();
    router.replace(qs ? `?${qs}` : pathname, { scroll: false });
  };

  const toggleTask = (t) => setFilter("task", taskFilter === t ? "" : t);

  const taskCounts = useMemo(() => {
    const counts = {};
    for (const r of recipes) {
      for (const t of r.meta.tasks || []) counts[t] = (counts[t] || 0) + 1;
    }
    return Object.entries(counts).sort((a, b) => b[1] - a[1]);
  }, [recipes]);

  const providers = useMemo(
    () => [...new Set(recipes.map((r) => r.meta.provider))].sort(),
    [recipes]
  );

  const filtered = useMemo(() => {
    return recipes.filter((r) => {
      if (providerFilter && r.meta.provider !== providerFilter) return false;
      if (taskFilter && !(r.meta.tasks || []).includes(taskFilter)) return false;
      return true;
    });
  }, [recipes, taskFilter, providerFilter]);

  const byProvider = useMemo(() => {
    const groups = {};
    for (const r of filtered) {
      const p = r.meta.provider;
      if (!groups[p]) groups[p] = [];
      groups[p].push(r);
    }
    return Object.entries(groups).sort((a, b) => b[1].length - a[1].length);
  }, [filtered]);

  return (
    <div>
      {/* Filter bar */}
      <div className="flex flex-wrap items-center gap-2 mb-6 pb-4 border-b border-border">
        {/* Task pills */}
        {taskCounts.map(([task, count]) => (
          <button
            key={task}
            onClick={() => toggleTask(task)}
            className={`rounded-full px-3 py-1 text-xs font-medium transition-all ${
              taskFilter === task
                ? "bg-foreground text-background shadow-sm"
                : "bg-secondary text-secondary-foreground hover:bg-secondary/70"
            }`}
          >
            {task.charAt(0).toUpperCase() + task.slice(1)}
            <span className={`ml-1 tabular-nums ${taskFilter === task ? "opacity-70" : "text-muted-foreground"}`}>{count}</span>
          </button>
        ))}

        <div className="h-4 w-px bg-border mx-1 hidden sm:block" />

        {/* Provider dropdown */}
        <select
          value={providerFilter}
          onChange={(e) => setFilter("provider", e.target.value)}
          className="rounded-full border border-border bg-background px-3 py-1 text-xs focus:outline-none focus:ring-2 focus:ring-vllm-blue/30"
        >
          <option value="">All Providers</option>
          {providers.map((p) => <option key={p} value={p}>{p}</option>)}
        </select>

        {/* Count + clear */}
        <div className="flex items-center gap-2 ml-auto">
          <span className="text-xs text-muted-foreground tabular-nums">
            {hasFilters ? `${filtered.length} / ${recipes.length}` : recipes.length}
          </span>
          {hasFilters && (
            <button
              onClick={() => router.replace(pathname, { scroll: false })}
              className="rounded-full p-0.5 text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
              title="Clear filters"
            >
              <X size={14} />
            </button>
          )}
        </div>
      </div>

      {/* Provider sections */}
      <div className="space-y-3">
        {byProvider.map(([provider, models], i) => (
          <ProviderCard
            key={provider}
            provider={provider}
            models={models}
            defaultOpen={i < 3 || hasFilters}
          />
        ))}
      </div>

      {filtered.length === 0 && (
        <div className="text-center py-20 text-muted-foreground">
          <p className="text-base">No recipes match this combination.</p>
          <a href="https://github.com/vllm-project/recipes/issues" className="text-vllm-blue text-sm hover:underline mt-3 inline-block">
            Request a recipe &rarr;
          </a>
        </div>
      )}
    </div>
  );
}

function ProviderCard({ provider, models, defaultOpen = false }) {
  const [expanded, setExpanded] = useState(defaultOpen);
  const logo = getProviderLogo(provider);

  return (
    <div className={`rounded-xl border border-border overflow-hidden transition-shadow ${expanded ? "shadow-sm" : ""}`}>
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full text-left px-4 py-3 flex items-center gap-3 hover:bg-muted/30 transition-colors"
      >
        {logo ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={logo} alt="" width={28} height={28} className="rounded-lg shrink-0" />
        ) : (
          <div className="w-7 h-7 rounded-lg bg-muted flex items-center justify-center text-xs font-bold text-muted-foreground shrink-0">
            {provider.charAt(0)}
          </div>
        )}
        <span className="font-semibold text-sm flex-1">{provider}</span>
        <span className="text-xs text-muted-foreground tabular-nums">{models.length}</span>
        <ChevronDown
          size={14}
          className={`text-muted-foreground transition-transform duration-200 ${expanded ? "rotate-180" : ""}`}
        />
      </button>

      {expanded && (
        <div className="border-t border-border divide-y divide-border/60">
          {models.map((r) => (
            <ModelRow key={r.meta.slug} recipe={r} />
          ))}
        </div>
      )}
    </div>
  );
}

function ModelRow({ recipe }) {
  const { meta, model, variants, hardware_overrides } = recipe;

  const hwTags = [];
  if (hardware_overrides?.hopper) hwTags.push("Hopper");
  if (hardware_overrides?.blackwell) hwTags.push("Blackwell");
  if (hardware_overrides?.amd) hwTags.push("AMD");

  return (
    <Link
      href={recipeHref(recipe)}
      className="flex flex-wrap items-center gap-x-3 gap-y-1.5 px-4 py-2.5 hover:bg-muted/30 transition-colors group"
    >
      <div className="min-w-[160px] shrink-0">
        <span className="text-sm font-medium group-hover:text-vllm-blue transition-colors">
          {meta.title}
        </span>
        <span className="text-xs text-muted-foreground ml-1.5">
          {model.parameter_count}
          {model.active_parameters && model.active_parameters !== model.parameter_count
            ? ` / ${model.active_parameters}`
            : ""}
        </span>
      </div>

      <Badge variant="outline" className="text-[10px] capitalize">{model.architecture}</Badge>

      <div className="flex gap-1 flex-wrap">
        {Object.entries(variants || {}).map(([name, v]) => (
          <span
            key={name}
            className="inline-flex items-center gap-0.5 rounded bg-muted px-1.5 py-0.5 text-[10px] font-mono"
          >
            {v.precision?.toUpperCase()}
            <span className="text-muted-foreground">{v.vram_minimum_gb}G</span>
          </span>
        ))}
      </div>

      <div className="flex gap-1 flex-wrap">
        {hwTags.map((t) => (
          <Badge key={t} variant="secondary" className="text-[9px]">{t}</Badge>
        ))}
      </div>

      <span className="text-[10px] text-muted-foreground ml-auto shrink-0 tabular-nums">v{model.min_vllm_version}+</span>

      <span className="text-muted-foreground/30 group-hover:text-vllm-blue transition-colors">&rarr;</span>
    </Link>
  );
}
