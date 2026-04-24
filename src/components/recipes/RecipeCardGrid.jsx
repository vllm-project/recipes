"use client";

import { useMemo } from "react";
import Link from "next/link";
import { Type, Eye, Sparkles, Hash, Layers, Cpu } from "lucide-react";
import { getProviderLogo, getProviderLogoClass, getProviderDisplayName } from "@/lib/providers";

const TASK_ICON = { text: Type, multimodal: Eye, omni: Sparkles, embedding: Hash };

const LATEST_COUNT = 8;

export function RecipeCardGrid({ recipes }) {
  const byOrg = useMemo(() => {
    const groups = {};
    for (const r of recipes) {
      const org = r.hf_org || "unknown";
      if (!groups[org]) groups[org] = [];
      groups[org].push(r);
    }
    return Object.entries(groups).sort((a, b) =>
      a[0].toLowerCase().localeCompare(b[0].toLowerCase())
    );
  }, [recipes]);

  const latest = useMemo(() => {
    // Sort by HF release date — newest models first. Tiebreak on recipe
    // date_updated, then id. Then dedupe by provider so "Latest recipes"
    // surfaces breadth across orgs instead of e.g. eight Qwen rows when
    // one org ships a big collection.
    const sorted = [...recipes].sort((a, b) => {
      const ra = a.hf_released ? new Date(a.hf_released).getTime() : 0;
      const rb = b.hf_released ? new Date(b.hf_released).getTime() : 0;
      if (ra !== rb) return rb - ra;
      const da = a.meta?.date_updated ? new Date(a.meta.date_updated).getTime() : 0;
      const db = b.meta?.date_updated ? new Date(b.meta.date_updated).getTime() : 0;
      if (da !== db) return db - da;
      return (a.hf_id || "").localeCompare(b.hf_id || "");
    });
    const seen = new Set();
    const out = [];
    for (const r of sorted) {
      const org = r.hf_org || "unknown";
      if (seen.has(org)) continue;
      seen.add(org);
      out.push(r);
      if (out.length >= LATEST_COUNT) break;
    }
    return out;
  }, [recipes]);

  return (
    <div className="space-y-10">
      {/* Latest recipes — primary freshness signal */}
      <section>
        <div className="flex items-baseline gap-2 mb-3 pb-2 border-b border-border">
          <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-widest">Latest recipes</span>
          <span className="text-[10px] text-muted-foreground/60">newest {latest.length}</span>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
          {latest.map((r) => <RecipeCard key={r.hf_id} recipe={r} />)}
        </div>
      </section>

      {/* Providers — secondary nav */}
      <section>
        <div className="flex items-baseline gap-2 mb-3 pb-2 border-b border-border">
          <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-widest">Browse by provider</span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
          {byOrg.map(([org, models]) => <ProviderTile key={org} org={org} models={models} />)}
        </div>
      </section>
    </div>
  );
}

function RecipeCard({ recipe }) {
  const v = recipe.variants?.default || {};
  const tasks = recipe.meta?.tasks || [];
  const logo = getProviderLogo(recipe.hf_org);
  const isOmni = tasks.includes("omni");
  const isMoe = recipe.model?.architecture === "moe";
  const ctx = recipe.model?.context_length || 0;
  const ctxLabel = ctx >= 1_000_000 ? `${Math.round(ctx / 1_000_000)}M` : ctx >= 1000 ? `${Math.round(ctx / 1000)}K` : String(ctx);
  const params = recipe.model?.parameter_count || "";
  const active = recipe.model?.active_parameters;
  const paramsLabel = isMoe && active && active !== params ? `${params}/${active}` : params;
  return (
    <Link
      href={`/${recipe.hf_id}`}
      className="group rounded-xl border border-border bg-card hover:border-vllm-blue/40 hover:shadow-sm hover:-translate-y-0.5 transition-all p-3.5 flex flex-col gap-2.5 min-h-[140px]"
    >
      {/* Header: logo + title */}
      <div className="flex items-start gap-2.5">
        {logo ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={logo} alt="" width={28} height={28} className={`rounded-md mt-0.5 shrink-0 ${getProviderLogoClass(recipe.hf_org)}`} />
        ) : (
          <div className="w-7 h-7 rounded-md bg-muted flex items-center justify-center text-xs font-bold text-muted-foreground shrink-0">
            {(recipe.meta?.provider || recipe.hf_org).charAt(0)}
          </div>
        )}
        <div className="min-w-0 flex-1">
          <div className="font-semibold text-sm leading-tight group-hover:text-vllm-blue transition-colors line-clamp-1 break-all">
            {recipe.hf_repo}
          </div>
          <div className="font-mono text-[10px] text-muted-foreground/70 truncate mt-0.5">
            {recipe.hf_org}
          </div>
        </div>
      </div>

      {/* Spec row */}
      <div className="flex flex-wrap gap-1.5 text-[10px]">
        <Badge>
          {isMoe ? <Layers size={9} /> : <Cpu size={9} />}
          <span className="font-mono">{paramsLabel}</span>
        </Badge>
        {v.precision && (
          <Badge>
            <span className="font-mono uppercase">{v.precision}</span>
          </Badge>
        )}
        {ctx > 0 && (
          <Badge>
            <span className="font-mono">{ctxLabel} ctx</span>
          </Badge>
        )}
        {tasks.map((t) => {
          const Icon = TASK_ICON[t] || Hash;
          return (
            <Badge key={t} subtle>
              <Icon size={9} />
              <span className="capitalize">{t}</span>
            </Badge>
          );
        })}
      </div>

      {/* Footer: hint or description */}
      <div className="mt-auto text-[11px] text-muted-foreground line-clamp-2 leading-snug">
        {recipe.meta?.performance_headline || recipe.meta?.description || ""}
      </div>
      {isOmni && (
        <div className="text-[10px] text-vllm-yellow/80 font-medium">via vLLM-Omni</div>
      )}
    </Link>
  );
}

function Badge({ children, subtle }) {
  return (
    <span className={`inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 ${
      subtle
        ? "border-border/60 text-muted-foreground/70"
        : "border-border text-muted-foreground"
    }`}>
      {children}
    </span>
  );
}

function ProviderTile({ org, models }) {
  const logo = getProviderLogo(org);
  const displayName = getProviderDisplayName(org);
  return (
    <Link
      href={`/${org}`}
      className="group rounded-xl border border-border p-3 flex flex-col gap-2 bg-card hover:border-vllm-blue/40 hover:shadow-sm hover:-translate-y-0.5 transition-all"
    >
      <div className="flex items-center gap-2.5">
        {logo ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={logo} alt="" width={28} height={28} className={`rounded-md shrink-0 ${getProviderLogoClass(org)}`} />
        ) : (
          <div className="w-7 h-7 rounded-md bg-muted flex items-center justify-center text-xs font-bold text-muted-foreground shrink-0">
            {displayName.charAt(0)}
          </div>
        )}
        <div className="min-w-0 flex-1">
          <div className="font-semibold text-sm truncate group-hover:text-vllm-blue transition-colors leading-tight">
            {displayName}
          </div>
          <div className="font-mono text-[10px] text-muted-foreground/70 truncate">{org}</div>
        </div>
      </div>
      <div className="flex items-baseline justify-between mt-auto">
        <span className="text-[11px] text-muted-foreground">
          <span className="font-semibold text-foreground tabular-nums">{models.length}</span>{" "}
          {models.length === 1 ? "recipe" : "recipes"}
        </span>
        <span className="text-muted-foreground/40 group-hover:text-vllm-blue group-hover:translate-x-0.5 transition-all text-xs">→</span>
      </div>
    </Link>
  );
}
