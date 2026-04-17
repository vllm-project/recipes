"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { getProviderLogo, getProviderDisplayName } from "@/lib/providers";
import { recipeHref } from "@/lib/recipe-utils";
import { ChevronDown } from "lucide-react";

export function RecipeCardGrid({ recipes }) {
  // Group by HF org, sort groups by count desc
  const byOrg = useMemo(() => {
    const groups = {};
    for (const r of recipes) {
      const org = r.hf_org || "unknown";
      if (!groups[org]) groups[org] = [];
      groups[org].push(r);
    }
    return Object.entries(groups).sort((a, b) => b[1].length - a[1].length);
  }, [recipes]);

  return (
    <div>
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-border">
        <div className="flex items-baseline gap-2">
          <span className="text-sm font-semibold tabular-nums">{recipes.length}</span>
          <span className="text-xs text-muted-foreground">recipes</span>
          <span className="text-muted-foreground/50">&middot;</span>
          <span className="text-sm font-semibold tabular-nums">{byOrg.length}</span>
          <span className="text-xs text-muted-foreground">providers</span>
        </div>
      </div>

      <div className="space-y-3">
        {byOrg.map(([org, models], i) => (
          <ProviderCard
            key={org}
            org={org}
            models={models}
            defaultOpen={i < 3}
          />
        ))}
      </div>
    </div>
  );
}

function ProviderCard({ org, models, defaultOpen = false }) {
  const [expanded, setExpanded] = useState(defaultOpen);
  const logo = getProviderLogo(org);
  const displayName = getProviderDisplayName(org);

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
            {displayName.charAt(0)}
          </div>
        )}
        <div className="flex-1 min-w-0 flex items-baseline gap-2">
          <span className="font-semibold text-sm">{displayName}</span>
          <span className="font-mono text-[10px] text-muted-foreground/60 truncate">{org}</span>
        </div>
        <span className="text-xs text-muted-foreground tabular-nums">{models.length}</span>
        <ChevronDown
          size={14}
          className={`text-muted-foreground transition-transform duration-200 ${expanded ? "rotate-180" : ""}`}
        />
      </button>

      {expanded && (
        <div className="border-t border-border divide-y divide-border/60">
          {models.map((r) => (
            <ModelRow key={r.hf_id} recipe={r} />
          ))}
        </div>
      )}
    </div>
  );
}

function ModelRow({ recipe }) {
  const { meta, model, variants, hardware_overrides, hf_repo } = recipe;

  const hwTags = [];
  if (hardware_overrides?.hopper) hwTags.push("Hopper");
  if (hardware_overrides?.blackwell) hwTags.push("Blackwell");
  if (hardware_overrides?.amd) hwTags.push("AMD");

  return (
    <Link
      href={recipeHref(recipe)}
      className="flex flex-wrap items-center gap-x-3 gap-y-1.5 px-4 py-3 hover:bg-muted/40 transition-all group relative"
    >
      <div className="min-w-[220px] shrink-0">
        <div className="text-sm font-semibold group-hover:text-vllm-blue transition-colors font-mono">
          {hf_repo || meta.title}
        </div>
        <span className="text-xs text-muted-foreground font-mono">
          {model.parameter_count}
          {model.active_parameters && model.active_parameters !== model.parameter_count
            ? ` / ${model.active_parameters}`
            : ""}
        </span>
      </div>

      <Badge variant="outline" className="text-[10px] capitalize">{model.architecture}</Badge>

      {meta.tasks?.map((t) => (
        <Badge key={t} variant="secondary" className="text-[10px] capitalize">{t}</Badge>
      ))}

      <div className="flex gap-1 flex-wrap">
        {Object.entries(variants || {}).map(([name, v]) => (
          <span
            key={name}
            className="inline-flex items-center gap-0.5 rounded bg-muted px-1.5 py-0.5 text-[10px] font-mono"
          >
            <span className="font-semibold">{v.precision?.toUpperCase()}</span>
            <span className="text-muted-foreground">{v.vram_minimum_gb}G</span>
          </span>
        ))}
      </div>

      <div className="flex gap-1 flex-wrap">
        {hwTags.map((t) => (
          <Badge key={t} variant="outline" className="text-[9px]">{t}</Badge>
        ))}
      </div>

      <button
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          window.open("https://vllm.ai/#quick-start", "_blank", "noopener,noreferrer");
        }}
        title="Install vLLM"
        className="text-[10px] text-muted-foreground hover:text-vllm-blue ml-auto shrink-0 tabular-nums transition-colors cursor-pointer"
      >
        v{model.min_vllm_version}+
      </button>

      <span className="text-muted-foreground/40 group-hover:text-vllm-blue group-hover:translate-x-0.5 transition-all">&rarr;</span>
    </Link>
  );
}
