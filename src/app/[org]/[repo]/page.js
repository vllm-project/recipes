import { Suspense } from "react";
import Link from "next/link";
import { notFound } from "next/navigation";
import { getAllRecipes, getRecipeByHfId } from "@/lib/recipes";
import { recipeHref } from "@/lib/recipe-utils";
import { loadStrategies } from "@/lib/strategies";
import { loadTaxonomy } from "@/lib/taxonomy";
import { getProviderLogo } from "@/lib/providers";
import { CommandBuilder } from "@/components/recipes/CommandBuilder";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeSlug from "rehype-slug";
import { Badge } from "@/components/ui/badge";
import { Cpu, Layers, Pencil, Bug, ExternalLink } from "lucide-react";

export async function generateStaticParams() {
  return getAllRecipes().map((r) => ({
    org: r.hf_org,
    repo: r.hf_repo,
  }));
}

export async function generateMetadata({ params }) {
  const { org, repo } = await params;
  const recipe = getRecipeByHfId(org, repo);
  if (!recipe) return {};
  return { title: recipe.meta.title, description: recipe.meta.description };
}

export default async function RecipePage({ params }) {
  const { org, repo } = await params;
  const recipe = getRecipeByHfId(org, repo);
  if (!recipe) notFound();

  const strategies = loadStrategies();
  const taxonomy = loadTaxonomy();
  const guide = recipe.guide || "";
  const logo = getProviderLogo(recipe.hf_org);

  const configRows = Object.entries(recipe.variants || {}).map(([key, v]) => ({
    name: key === "default" ? "Default" : key.toUpperCase(),
    precision: v.precision?.toUpperCase() || "—",
    vram: `${v.vram_minimum_gb} GB`,
    description: v.description || "",
  }));

  const allRecipes = getAllRecipes();
  // related_recipes can be either "org/repo" HF id or the old slug format
  const related = (recipe.meta.related_recipes || [])
    .map((s) => allRecipes.find((r) => r.hf_id === s || r.meta.slug === s))
    .filter(Boolean);

  return (
    <main className="py-6 w-full min-w-0">
      {/* ── Model header ── */}
      <header className="mb-8">
        <div className="flex items-start gap-4 mb-4">
          {logo && (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={logo} alt={recipe.meta.provider} width={44} height={44} className="rounded-xl mt-0.5 shrink-0" />
          )}
          <div className="min-w-0 flex-1">
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight font-mono break-all">
              <span className="text-muted-foreground font-normal">{recipe.hf_org}/</span>
              {recipe.hf_repo}
            </h1>
            <p className="text-sm text-muted-foreground mt-1 line-clamp-2">{recipe.meta.description}</p>
            <a
              href={`https://huggingface.co/${recipe.hf_id}`}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-vllm-blue transition-colors mt-2"
            >
              View on HuggingFace
              <ExternalLink size={10} />
            </a>
          </div>
        </div>

        {/* Tags — single row, no duplication */}
        <div className="flex flex-wrap gap-1.5">
          <Badge variant="outline" className="gap-1 text-xs capitalize">
            {recipe.model.architecture === "moe" ? <Layers size={11} /> : <Cpu size={11} />}
            {recipe.model.architecture}
          </Badge>
          <Badge variant="outline" className="text-xs">
            {recipe.model.parameter_count}
            {recipe.model.active_parameters && recipe.model.active_parameters !== recipe.model.parameter_count
              ? ` / ${recipe.model.active_parameters}`
              : ""}
          </Badge>
          <Badge variant="outline" className="text-xs">{(recipe.model.context_length || 0).toLocaleString()} ctx</Badge>
          <a
            href="https://vllm.ai/#quick-start"
            target="_blank"
            rel="noopener noreferrer"
            title="Install vLLM"
            className="inline-flex items-center rounded-md border border-border px-2 py-0.5 text-xs font-medium w-fit whitespace-nowrap transition-colors hover:border-vllm-blue/50 hover:text-vllm-blue"
          >
            vLLM {recipe.model.min_vllm_version}+
            <ExternalLink size={10} className="ml-1 opacity-50" />
          </a>
          {recipe.meta.tasks?.map((t) => (
            <Badge key={t} variant="secondary" className="text-xs capitalize">{t}</Badge>
          ))}
        </div>
      </header>

      {/* ── Command Builder ── */}
      <section className="mb-10">
        <Suspense fallback={<div className="h-40 rounded-2xl bg-muted animate-pulse" />}>
          <CommandBuilder recipe={recipe} strategies={strategies} taxonomy={taxonomy} />
        </Suspense>
      </section>

      {/* ── Reference sections ── */}
      <section className="space-y-2">
        {guide && (
          <Accordion title="Guide" defaultOpen>
            <div className="guide-content">
              <Markdown
                remarkPlugins={[remarkGfm]}
                rehypePlugins={[rehypeSlug]}
              >
                {guide}
              </Markdown>
            </div>
          </Accordion>
        )}

        {configRows.length > 1 && (
          <Accordion title="Configuration Matrix">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-left text-muted-foreground text-xs">
                    <th className="pb-2 pr-6 font-medium">Variant</th>
                    <th className="pb-2 pr-6 font-medium">Precision</th>
                    <th className="pb-2 pr-6 font-medium">Min VRAM</th>
                    <th className="pb-2 font-medium">Notes</th>
                  </tr>
                </thead>
                <tbody>
                  {configRows.map((row) => (
                    <tr key={row.name} className="border-b border-border/40">
                      <td className="py-2 pr-6 font-mono text-xs">{row.name}</td>
                      <td className="py-2 pr-6 text-xs">{row.precision}</td>
                      <td className="py-2 pr-6 text-xs tabular-nums">{row.vram}</td>
                      <td className="py-2 text-xs text-muted-foreground">{row.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Accordion>
        )}
      </section>

      {/* ── Footer ── */}
      <footer className="mt-10 pt-4 border-t border-border text-sm text-muted-foreground flex flex-wrap gap-x-4 gap-y-2">
        <span>Updated {recipe.meta.date_updated}</span>
        <a
          href={`https://github.com/vllm-project/recipes/edit/main/models/${recipe.hf_org}/${recipe.hf_repo}.yaml`}
          className="inline-flex items-center gap-1 hover:text-foreground transition-colors"
        >
          <Pencil size={12} /> Edit recipe
        </a>
        <a
          href="https://github.com/vllm-project/recipes/issues"
          className="inline-flex items-center gap-1 hover:text-foreground transition-colors"
        >
          <Bug size={12} /> Report issue
        </a>
        {related.length > 0 && (
          <span className="basis-full mt-1">
            Related:{" "}
            {related.map((r, i) => (
              <span key={r.meta.slug}>
                {i > 0 && " · "}
                <Link href={recipeHref(r)} className="text-vllm-blue hover:underline">{r.meta.title}</Link>
              </span>
            ))}
          </span>
        )}
      </footer>
    </main>
  );
}

function Accordion({ title, children, defaultOpen = false }) {
  return (
    <details className="group rounded-xl border border-border overflow-hidden" open={defaultOpen || undefined}>
      <summary className="px-5 py-3 cursor-pointer text-sm font-semibold select-none hover:bg-muted/20 transition-colors flex items-center justify-between">
        {title}
        <ChevronIcon />
      </summary>
      <div className="px-5 pb-5 border-t border-border/40 pt-4">{children}</div>
    </details>
  );
}

function ChevronIcon() {
  return (
    <svg
      className="w-4 h-4 text-muted-foreground transition-transform duration-200 group-open:rotate-180"
      fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
    >
      <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
    </svg>
  );
}
