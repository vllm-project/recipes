import { Suspense } from "react";
import Link from "next/link";
import { notFound, redirect } from "next/navigation";
import { getAllRecipes, getAllRoutablePairs, findVariantRedirect, getRecipeByHfId } from "@/lib/recipes";
import { recipeHref } from "@/lib/recipe-utils";
import { siteUrl } from "@/lib/site-url";
import { loadStrategies } from "@/lib/strategies";
import { loadTaxonomy } from "@/lib/taxonomy";
import { getProviderLogo, getProviderLogoClass } from "@/lib/providers";
import { CommandBuilder } from "@/components/recipes/CommandBuilder";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeSlug from "rehype-slug";
import { Badge } from "@/components/ui/badge";
import { Cpu, Layers, Pencil, Bug, ExternalLink } from "lucide-react";

export async function generateStaticParams() {
  return getAllRoutablePairs();
}

export async function generateMetadata({ params }) {
  const { org, repo } = await params;
  const recipe = getRecipeByHfId(org, repo);
  if (!recipe) return {};
  const { meta, model } = recipe;
  const title = `${org}/${repo}`;
  const description = meta.description || meta.title;
  const paramStr = model.parameter_count
    ? model.active_parameters && model.active_parameters !== model.parameter_count
      ? `${model.parameter_count} / ${model.active_parameters} active`
      : model.parameter_count
    : "";
  const ctxStr = model.context_length ? `${(model.context_length / 1024).toFixed(0)}K ctx` : "";
  const metaLine = [paramStr, (model.architecture || "").toUpperCase(), ctxStr]
    .filter(Boolean)
    .join(" · ");
  const versionStr = model.min_vllm_version ? `vLLM ${model.min_vllm_version}+` : "";
  // Provider subtitle is redundant with the "org/" prefix in the title, so
  // the recipe OG uses: title + meta (spec strip) + version pill.
  const ogUrl = `/og?title=${encodeURIComponent(title)}&meta=${encodeURIComponent(
    metaLine
  )}&version=${encodeURIComponent(versionStr)}&path=${encodeURIComponent(`/${org}/${repo}`)}`;
  // og:title — aim for 50–60 chars for optimal preview width. Skip the
  // " on vLLM" suffix since og:site_name already provides it.
  const ogTitle = metaLine ? `${title} — ${metaLine}` : title;
  // Page <title> uses the canonical hf_id (e.g. "deepseek-ai/DeepSeek-V3.2")
  // — that's what people search for. The layout template appends " | vLLM
  // Recipes" so the final tab reads "deepseek-ai/DeepSeek-V3.2 | vLLM Recipes".
  return {
    title,
    description,
    openGraph: {
      type: "article",
      title: ogTitle,
      description,
      url: `/${org}/${repo}`,
      images: [{ url: ogUrl, width: 1200, height: 630, alt: title }],
    },
    twitter: {
      card: "summary_large_image",
      title: ogTitle,
      description,
      images: [ogUrl],
    },
    alternates: { canonical: `/${org}/${repo}` },
  };
}

export default async function RecipePage({ params }) {
  const { org, repo } = await params;
  const recipe = getRecipeByHfId(org, repo);
  if (!recipe) {
    const v = findVariantRedirect(org, repo);
    if (v) redirect(`/${v.parent.hf_org}/${v.parent.hf_repo}?variant=${encodeURIComponent(v.variantKey)}`);
    notFound();
  }

  const strategies = loadStrategies();
  const taxonomy = loadTaxonomy();
  const guide = recipe.guide || "";
  const logo = getProviderLogo(recipe.hf_org);

  const allRecipes = getAllRecipes();
  // related_recipes can be either "org/repo" HF id or the old slug format
  const related = (recipe.meta.related_recipes || [])
    .map((s) => allRecipes.find((r) => r.hf_id === s || r.meta.slug === s))
    .filter(Boolean);

  const recipeUrl = `${siteUrl}/${recipe.hf_org}/${recipe.hf_repo}`;
  const jsonLd = [
    {
      "@context": "https://schema.org",
      "@type": "SoftwareApplication",
      name: recipe.hf_id,
      alternateName: recipe.meta.title,
      description: recipe.meta.description || recipe.meta.title,
      applicationCategory: "DeveloperApplication",
      operatingSystem: "Linux",
      url: recipeUrl,
      softwareRequirements: recipe.model.min_vllm_version
        ? `vLLM ${recipe.model.min_vllm_version}+`
        : "vLLM",
      creator: { "@type": "Organization", name: recipe.meta.provider, url: `https://huggingface.co/${recipe.hf_org}` },
      sameAs: [`https://huggingface.co/${recipe.hf_id}`],
    },
    {
      "@context": "https://schema.org",
      "@type": "BreadcrumbList",
      itemListElement: [
        { "@type": "ListItem", position: 1, name: "vLLM Recipes", item: siteUrl },
        { "@type": "ListItem", position: 2, name: recipe.meta.provider, item: `${siteUrl}/${recipe.hf_org}` },
        { "@type": "ListItem", position: 3, name: recipe.hf_repo, item: recipeUrl },
      ],
    },
  ];

  return (
    <main className="py-6 w-full min-w-0">
      <script
        type="application/ld+json"
        // eslint-disable-next-line react/no-danger
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      {/* ── Model header ── */}
      <header className="mb-8">
        <div className="flex items-start gap-4 mb-4">
          {logo && (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={logo} alt={recipe.meta.provider} width={44} height={44} className={`rounded-xl mt-0.5 shrink-0 ${getProviderLogoClass(recipe.hf_org)}`} />
          )}
          <div className="min-w-0 flex-1">
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight font-mono break-all">
              <span className="text-muted-foreground font-normal">{recipe.hf_org}/</span>
              {recipe.hf_repo}
            </h1>
            <p className="text-sm text-muted-foreground mt-1 line-clamp-2">{recipe.meta.description}</p>
            {recipe.meta.performance_headline && (
              <p className="text-xs text-muted-foreground/70 italic mt-1 leading-snug">{recipe.meta.performance_headline}</p>
            )}
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
              <span key={r.hf_id}>
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
  // Stronger border than bare `--border` (10% in dark mode) so the outline is
  // visible on both themes. `foreground/15` stays subtle but catches enough
  // contrast to define the card.
  return (
    <details
      className="group rounded-xl border border-foreground/15 bg-card/40 overflow-hidden"
      open={defaultOpen || undefined}
    >
      <summary className="px-5 py-3 cursor-pointer text-sm font-semibold select-none hover:bg-foreground/[0.04] transition-colors flex items-center justify-between">
        {title}
        <ChevronIcon />
      </summary>
      <div className="px-5 pb-5 border-t border-foreground/10 pt-4">{children}</div>
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
