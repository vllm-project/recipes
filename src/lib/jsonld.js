// Centralised schema.org JSON-LD builders for recipes.vllm.ai.
//
// Every page emits a single `<script type="application/ld+json">` rooted in
// an `@graph` so we can reference shared entities (the vLLM Organization, the
// site WebSite) by `@id` instead of inlining them on each schema.
//
// Schemas wired here:
//   - Organization        (sitewide, by @id)
//   - WebSite             (sitewide, with SearchAction)
//   - TechArticle         (per recipe — primary type per PROD-8 spec)
//   - SoftwareApplication (per recipe — kept alongside TechArticle so the
//                          existing SoftwareApplication signal isn't lost)
//   - BreadcrumbList      (per page, mirroring URL hierarchy)
//
// JSON-LD is rendered inside a `<script>` tag, so a stray `</script>` inside
// a stringified value would break the document. `jsonLdScript` defends with
// the standard `<` → `<` substitution.

import { siteUrl } from "@/lib/site-url";

const ORG_NAME = "vLLM";
const ORG_LEGAL_NAME = "vLLM Project";
const ORG_DESCRIPTION =
  "vLLM is a high-throughput, memory-efficient inference and serving engine for large language models.";
// Use vllm.ai as the brand anchor so all three properties (vllm.ai,
// docs.vllm.ai, recipes.vllm.ai) reference the same Organization @id.
// This is the single most important architectural choice in the SEO
// bundle — it stitches three subdomains into one authority cluster.
const VLLM_BRAND_ID = "https://vllm.ai/#organization";

export function buildOrganizationLd() {
  return {
    "@type": "Organization",
    "@id": VLLM_BRAND_ID,
    name: ORG_NAME,
    legalName: ORG_LEGAL_NAME,
    url: "https://vllm.ai",
    description: ORG_DESCRIPTION,
    logo: { "@type": "ImageObject", url: "https://vllm.ai/vLLM-Logo.png" },
    sameAs: [
      "https://github.com/vllm-project/vllm",
      "https://x.com/vllm_project",
      "https://huggingface.co/vllm-project",
    ],
  };
}

export function buildWebSiteLd() {
  return {
    "@type": "WebSite",
    "@id": `${siteUrl}/#website`,
    name: "vLLM Recipes",
    url: siteUrl,
    description:
      "Per-model vLLM serving recipes: hardware-tuned vllm serve commands, flag explanations, and known pitfalls.",
    publisher: { "@id": VLLM_BRAND_ID },
    inLanguage: "en",
  };
}

export function buildBreadcrumbLd(crumbs) {
  return {
    "@type": "BreadcrumbList",
    itemListElement: crumbs.map((crumb, index) => ({
      "@type": "ListItem",
      position: index + 1,
      name: crumb.name,
      item: crumb.url.startsWith("http") ? crumb.url : `${siteUrl}${crumb.url}`,
    })),
  };
}

// Render the list of verified GPUs into a human-readable hardware string.
// Used in both metadata templates and TechArticle.keywords.
export function hardwareLabel(recipe) {
  const hw = recipe?.meta?.hardware;
  if (!hw || typeof hw !== "object") return "";
  const verified = Object.entries(hw)
    .filter(([, status]) => status === "verified")
    .map(([gpu]) => gpu);
  if (verified.length === 0) return "";
  return verified.join(", ");
}

export function buildTechArticleLd(recipe) {
  const url = `${siteUrl}/${recipe.hf_org}/${recipe.hf_repo}`;
  const hwList = hardwareLabel(recipe);
  const datePublished = recipe.hf_released || recipe.meta?.date_updated || undefined;
  const dateModified = recipe.meta?.date_updated || datePublished;

  const keywords = [
    "vllm",
    recipe.hf_repo,
    recipe.meta?.provider,
    ...(recipe.meta?.tasks || []),
    recipe.model?.architecture,
    hwList,
  ]
    .filter(Boolean)
    .join(", ");

  return {
    "@type": "TechArticle",
    "@id": `${url}#article`,
    headline: `${recipe.hf_repo} on vLLM`,
    name: `${recipe.hf_repo} on vLLM`,
    description: recipe.meta?.description || `${recipe.hf_id} serving recipe for vLLM.`,
    url,
    mainEntityOfPage: { "@type": "WebPage", "@id": url },
    inLanguage: "en",
    isAccessibleForFree: true,
    proficiencyLevel: "Expert",
    keywords,
    ...(datePublished && { datePublished }),
    ...(dateModified && { dateModified }),
    about: {
      "@type": "Thing",
      name: recipe.hf_id,
      url: `https://huggingface.co/${recipe.hf_id}`,
    },
    author: {
      "@type": "Organization",
      name: recipe.meta?.provider || "vLLM",
      ...(recipe.hf_org && { url: `https://huggingface.co/${recipe.hf_org}` }),
    },
    publisher: { "@id": VLLM_BRAND_ID },
    isPartOf: { "@id": `${siteUrl}/#website` },
  };
}

export function buildSoftwareApplicationLd(recipe) {
  const url = `${siteUrl}/${recipe.hf_org}/${recipe.hf_repo}`;
  return {
    "@type": "SoftwareApplication",
    name: recipe.hf_id,
    alternateName: recipe.meta?.title,
    description: recipe.meta?.description || recipe.meta?.title,
    applicationCategory: "DeveloperApplication",
    operatingSystem: "Linux",
    url,
    softwareRequirements: recipe.model?.min_vllm_version
      ? `vLLM ${recipe.model.min_vllm_version}+`
      : "vLLM",
    creator: {
      "@type": "Organization",
      name: recipe.meta?.provider,
      url: `https://huggingface.co/${recipe.hf_org}`,
    },
    sameAs: [`https://huggingface.co/${recipe.hf_id}`],
  };
}

export function jsonLdScript(data) {
  const json = JSON.stringify(data).replace(/</g, "\\u003c");
  return { __html: json };
}
