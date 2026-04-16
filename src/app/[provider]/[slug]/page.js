import { Suspense } from "react";
import Link from "next/link";
import { notFound } from "next/navigation";
import { getAllRecipes, getRecipe } from "@/lib/recipes";
import { recipeHref } from "@/lib/recipe-utils";
import { loadStrategies } from "@/lib/strategies";
import { loadTaxonomy } from "@/lib/taxonomy";
import { getProviderLogo } from "@/lib/providers";
import { CommandBuilder } from "@/components/recipes/CommandBuilder";
import { MDXRemote } from "next-mdx-remote/rsc";
import remarkGfm from "remark-gfm";
import rehypeSlug from "rehype-slug";
import rehypePrettyCode from "rehype-pretty-code";
import rehypeRaw from "rehype-raw";
import { Badge } from "@/components/ui/badge";
import { Cpu, Layers, Pencil, Bug } from "lucide-react";

export async function generateStaticParams() {
  return getAllRecipes().map((r) => ({
    provider: r.meta.provider.toLowerCase().replace(/\s+/g, "-"),
    slug: r.meta.slug,
  }));
}

export async function generateMetadata({ params }) {
  const { slug } = await params;
  const recipe = getRecipe(slug);
  if (!recipe) return {};
  return { title: recipe.meta.title, description: recipe.meta.description };
}

export default async function RecipePage({ params }) {
  const { slug } = await params;
  const recipe = getRecipe(slug);
  if (!recipe) notFound();

  const strategies = loadStrategies();
  const taxonomy = loadTaxonomy();
  const guide = recipe.guide || "";
  const logo = getProviderLogo(recipe.meta.provider);

  const hwTags = Object.keys(recipe.hardware_overrides || {}).map(
    (k) => k.charAt(0).toUpperCase() + k.slice(1)
  );

  const configRows = Object.entries(recipe.variants || {}).map(([key, v]) => ({
    name: key === "default" ? "Default" : key.toUpperCase(),
    precision: v.precision?.toUpperCase() || "—",
    vram: `${v.vram_minimum_gb} GB`,
    description: v.description || "",
  }));

  const allRecipes = getAllRecipes();
  const related = (recipe.meta.related_recipes || [])
    .map((s) => allRecipes.find((r) => r.meta.slug === s))
    .filter(Boolean);

  return (
    <main className="py-6 max-w-4xl">
      {/* ── Model header ── */}
      <header className="mb-8">
        <div className="flex items-start gap-4 mb-4">
          {logo && (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={logo} alt={recipe.meta.provider} width={44} height={44} className="rounded-xl mt-0.5 shrink-0" />
          )}
          <div className="min-w-0">
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">{recipe.meta.title}</h1>
            <p className="text-sm text-muted-foreground mt-1 line-clamp-2">{recipe.meta.description}</p>
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
          <Badge variant="outline" className="text-xs">vLLM {recipe.model.min_vllm_version}+</Badge>
          {hwTags.map((t) => (
            <Badge key={t} variant="secondary" className="text-xs">{t}</Badge>
          ))}
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
              <MDXRemote
                source={guide}
                options={{
                  mdxOptions: {
                    remarkPlugins: [remarkGfm],
                    rehypePlugins: [rehypeRaw, rehypeSlug, [rehypePrettyCode, { theme: "github-dark-default", keepBackground: false }]],
                  },
                }}
              />
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
          href={`https://github.com/vllm-project/recipes/edit/main/models/${recipe.meta.slug}.yaml`}
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
