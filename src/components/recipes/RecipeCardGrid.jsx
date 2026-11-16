"use client";

import { useMemo } from "react";
import Link from "next/link";
import { Layers, Cpu } from "lucide-react";
import { getProviderLogo, getProviderLogoClass, getProviderDisplayName } from "@/lib/providers";
import { compareRecipesByDateAdded, primaryRecipeTask } from "@/lib/recipe-utils";
import { TASK_ICON, TASK_LABEL, OUTPUT_ICON, OUTPUT_LABEL, kindTooltip } from "@/components/recipes/task-icons";
import { InfoTip } from "@/components/ui/tooltip";

// The kind label on a card: an icon + muted text, no fill or border, so it
// reads as metadata beside the title rather than competing with it. Only the
// icon carries color (the /browse Task filter's hues; omni's vLLM-Omni ring
// brings its own), enough to tell kinds apart at a glance.
const KIND_ICON_CLASS = {
  text:       "text-muted-foreground/70",
  multimodal: "text-violet-500 dark:text-violet-400",
  embedding:  "text-cyan-600 dark:text-cyan-400",
  omni:       "",
};

const LATEST_COUNT = 8;
// At most 2 per provider: one org landing several recipes on the same day
// shouldn't take the whole row, but a cap of 1 hid genuinely new recipes behind
// a sibling that landed a day later, and back-filled the row with much older
// ones from other orgs.
const LATEST_PER_ORG = 2;

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
    // "Latest recipes" means newly added to this catalog, not the HF repo's
    // createdAt (a model repo may exist privately long before launch).
    const sorted = [...recipes].sort(compareRecipesByDateAdded);
    const perOrg = new Map();
    const out = [];
    for (const r of sorted) {
      const org = r.hf_org || "unknown";
      const taken = perOrg.get(org) || 0;
      if (taken >= LATEST_PER_ORG) continue;
      perOrg.set(org, taken + 1);
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
          <Link
            href="/browse"
            className="ml-auto text-[11px] text-muted-foreground hover:text-vllm-blue transition-colors"
          >
            View all {recipes.length} →
          </Link>
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
  const kind = primaryRecipeTask(recipe);
  const isOmni = kind === "omni";
  const logo = getProviderLogo(recipe.hf_org);
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
          {/* The kind sits on the org line, not beside the title: there it took
              ~80px from the title row and truncated 27-31 character names. */}
          <div className="flex items-center gap-1.5 mt-0.5 min-w-0">
            <span className="font-mono text-[10px] text-muted-foreground/70 truncate">{recipe.hf_org}</span>
            <span className="text-[10px] text-muted-foreground/40" aria-hidden>·</span>
            <KindChip kind={kind} tasks={recipe.meta?.tasks} outputs={recipe.omni_outputs} />
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
        {/* What an omni model generates, where an LLM shows its context length. */}
        {isOmni && (recipe.omni_outputs || []).map((o) => {
          const Icon = OUTPUT_ICON[o];
          return (
            <Badge key={o}>
              {Icon && <Icon size={9} />}
              <span>{OUTPUT_LABEL[o] || o}</span>
            </Badge>
          );
        })}
        {/* Context length means nothing for image / video / audio generation. */}
        {ctx > 0 && !isOmni && (
          <Badge>
            <span className="font-mono">{ctxLabel} ctx</span>
          </Badge>
        )}
      </div>

      {/* Footer: hint or description */}
      <div className="mt-auto text-[11px] text-muted-foreground line-clamp-2 leading-snug">
        {recipe.meta?.description || recipe.meta?.performance_headline || ""}
      </div>
    </Link>
  );
}

function KindChip({ kind, tasks, outputs }) {
  if (!(kind in KIND_ICON_CLASS)) return null;
  const Icon = TASK_ICON[kind];
  const label = TASK_LABEL[kind] || kind;
  const tip = kindTooltip(kind, { tasks, omniOutputs: outputs });
  return (
    <InfoTip
      side="top"
      content={tip && (
        <span className="block">
          <span className="block font-semibold mb-0.5">{tip.title}</span>
          {tip.body}
        </span>
      )}
    >
      <span
        className="inline-flex items-center gap-1 text-[10px] text-muted-foreground whitespace-nowrap shrink-0 cursor-help"
      >
        {Icon && <Icon size={kind === "omni" ? 11 : 10} className={KIND_ICON_CLASS[kind]} />}
        {label}
      </span>
    </InfoTip>
  );
}

function Badge({ children }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-md border border-border px-1.5 py-0.5 text-muted-foreground">
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
