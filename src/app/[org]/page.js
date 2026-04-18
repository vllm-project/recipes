import Link from "next/link";
import { notFound } from "next/navigation";
import { getAllRecipes } from "@/lib/recipes";
import { getProviderLogo, getProviderDisplayName } from "@/lib/providers";
import { recipeHref } from "@/lib/recipe-utils";
import { Badge } from "@/components/ui/badge";
import { ExternalLink, Type, Eye, Sparkles, Cpu, Hash } from "lucide-react";

export async function generateStaticParams() {
  const recipes = getAllRecipes();
  const orgs = [...new Set(recipes.map((r) => r.hf_org))];
  return orgs.map((org) => ({ org }));
}

export async function generateMetadata({ params }) {
  const { org } = await params;
  const displayName = getProviderDisplayName(org);
  return {
    title: displayName,
    description: `vLLM recipes for ${displayName} models`,
  };
}

// Task → icon + display label
const TASK_META = {
  text:       { icon: Type,     label: "Text" },
  multimodal: { icon: Eye,      label: "Multimodal" },
  omni:       { icon: Sparkles, label: "Omni" },
  embedding:  { icon: Hash,     label: "Embedding" },
};
const TASK_ORDER = ["text", "multimodal", "omni", "embedding"];

export default async function OrgPage({ params }) {
  const { org } = await params;
  const all = getAllRecipes();
  const models = all.filter((r) => r.hf_org === org);
  if (models.length === 0) notFound();

  const logo = getProviderLogo(org);
  const displayName = getProviderDisplayName(org);

  // Group by primary task (first in meta.tasks, or "other" if empty)
  const groups = {};
  for (const m of models) {
    const task = (m.meta.tasks || [])[0] || "other";
    if (!groups[task]) groups[task] = [];
    groups[task].push(m);
  }
  const orderedGroups = TASK_ORDER
    .filter((t) => groups[t])
    .map((t) => [t, groups[t]]);
  // Any task not in TASK_ORDER gets appended at the end
  for (const t of Object.keys(groups)) {
    if (!TASK_ORDER.includes(t)) orderedGroups.push([t, groups[t]]);
  }

  return (
    <main className="py-8">
      <header className="mb-8 flex items-center gap-4">
        {logo ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={logo} alt="" width={56} height={56} className="rounded-xl shrink-0" />
        ) : (
          <div className="w-14 h-14 rounded-xl bg-muted flex items-center justify-center text-2xl font-bold text-muted-foreground">
            {displayName.charAt(0)}
          </div>
        )}
        <div>
          <h1 className="text-2xl font-bold tracking-tight">{displayName}</h1>
          <div className="flex items-center gap-3 mt-1 text-sm text-muted-foreground">
            <span className="font-mono text-xs">{org}</span>
            <span>·</span>
            <span>{models.length} recipe{models.length > 1 ? "s" : ""}</span>
            <a
              href={`https://huggingface.co/${org}`}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-xs hover:text-foreground transition-colors"
            >
              HuggingFace
              <ExternalLink size={10} />
            </a>
          </div>
        </div>
      </header>

      <div className="space-y-8">
        {orderedGroups.map(([task, list]) => {
          const tmeta = TASK_META[task];
          const Icon = tmeta?.icon || Cpu;
          return (
            <section key={task}>
              <div className="flex items-center gap-2 mb-3">
                <Icon size={14} className="text-muted-foreground" />
                <h2 className="text-sm font-semibold uppercase tracking-widest text-muted-foreground">
                  {tmeta?.label || task}
                </h2>
                <span className="text-xs text-muted-foreground/60 tabular-nums">{list.length}</span>
              </div>
              <div className="border border-border rounded-xl divide-y divide-border/60 overflow-hidden">
                {list.map((r) => (
                  <ModelRow key={r.hf_id} recipe={r} />
                ))}
              </div>
            </section>
          );
        })}
      </div>
    </main>
  );
}

function ModelRow({ recipe }) {
  const { meta, model, variants, hf_repo } = recipe;

  return (
    <Link
      href={recipeHref(recipe)}
      className="flex flex-wrap items-center gap-x-3 gap-y-1.5 px-4 py-3 hover:bg-muted/40 transition-all group"
    >
      <div className="min-w-[220px] shrink-0">
        <div className="text-sm font-semibold font-mono group-hover:text-vllm-blue transition-colors">
          {hf_repo || meta.title}
        </div>
        <div className="text-xs text-muted-foreground font-mono">
          {model.parameter_count}
          {model.active_parameters && model.active_parameters !== model.parameter_count
            ? ` / ${model.active_parameters}`
            : ""}
        </div>
      </div>

      <Badge variant="outline" className="text-[10px] capitalize">{model.architecture}</Badge>

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

      <span className="text-[10px] text-muted-foreground ml-auto shrink-0 tabular-nums">v{model.min_vllm_version}+</span>
      <span className="text-muted-foreground/40 group-hover:text-vllm-blue group-hover:translate-x-0.5 transition-all">&rarr;</span>
    </Link>
  );
}
