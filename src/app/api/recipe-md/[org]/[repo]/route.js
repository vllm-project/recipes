// /api/recipe-md/<org>/<repo> — internal markdown shadow for a recipe.
// Reached via middleware-driven rewrite when someone requests
// `/<org>/<repo>.md`; never linked publicly.
//
// PROD-10 — why .md shadow URLs matter: AI assistants (Claude, ChatGPT,
// Perplexity) often fetch URLs the user pastes. The recipes site is
// command-builder heavy; the rendered HTML page works fine in a browser
// but is awkward to retrieve cleanly because the command itself is
// client-composed from variant + hardware + strategy state. Serving a
// canonical markdown view fixes that:
//
//   - One representative `vllm serve` command (default variant, default
//     hardware) that an assistant can quote verbatim.
//   - The recipe metadata, variant list, verified hardware, and the
//     full guide body — every piece an assistant needs to answer
//     "how do I run X with vLLM?".
//   - YAML-like frontmatter at the top so retrievers can extract
//     structured fields without parsing markdown.
//
// We do NOT use User-Agent sniffing (cloaking risk) or Accept-header
// negotiation (operationally fragile through CDNs). The `.md` suffix
// is the explicit signal.

import { getAllRoutablePairs, getRecipeByHfId } from "@/lib/recipes";
import { siteUrl } from "@/lib/site-url";

// Render the recipe's verified hardware map into a comma-separated string.
// Inlined here so this PR has no cross-branch dependency on jsonld.js.
function hardwareLabel(recipe) {
  const hw = recipe?.meta?.hardware;
  if (!hw || typeof hw !== "object") return "";
  const verified = Object.entries(hw)
    .filter(([, status]) => status === "verified")
    .map(([gpu]) => gpu);
  return verified.join(", ");
}

export const dynamic = "force-static";

export async function generateStaticParams() {
  return getAllRoutablePairs();
}

function renderRecipeMarkdown(recipe) {
  const canonical = `${siteUrl}/${recipe.hf_org}/${recipe.hf_repo}`;
  const hw = hardwareLabel(recipe);
  const provider = recipe.meta?.provider || recipe.hf_org;
  const tasks = (recipe.meta?.tasks || []).join(", ");

  const frontmatter = [
    "---",
    `title: ${JSON.stringify(`${recipe.hf_id} on vLLM`)}`,
    `hf_id: ${recipe.hf_id}`,
    `provider: ${JSON.stringify(provider)}`,
    recipe.hf_released ? `released: ${recipe.hf_released}` : null,
    recipe.meta?.date_updated ? `last_updated: ${recipe.meta.date_updated}` : null,
    recipe.model?.architecture ? `architecture: ${recipe.model.architecture}` : null,
    recipe.model?.parameter_count
      ? `parameters: ${JSON.stringify(recipe.model.parameter_count)}`
      : null,
    recipe.model?.context_length ? `context_length: ${recipe.model.context_length}` : null,
    recipe.model?.min_vllm_version ? `min_vllm_version: ${recipe.model.min_vllm_version}` : null,
    tasks ? `tasks: [${tasks}]` : null,
    hw ? `verified_hardware: ${JSON.stringify(hw)}` : null,
    `canonical_url: ${canonical}`,
    "---",
    "",
  ]
    .filter((l) => l !== null)
    .join("\n");

  const sections = [];
  sections.push(`# ${recipe.hf_id} on vLLM`);
  if (recipe.meta?.description) {
    sections.push("");
    sections.push(recipe.meta.description);
  }
  if (recipe.meta?.performance_headline) {
    sections.push("");
    sections.push(`> ${recipe.meta.performance_headline}`);
  }

  // Model facts as a compact reference block. Assistants tend to lift
  // these into responses verbatim.
  const facts = [];
  if (recipe.model?.parameter_count) facts.push(`- Parameters: ${recipe.model.parameter_count}`);
  if (recipe.model?.active_parameters)
    facts.push(`- Active parameters: ${recipe.model.active_parameters}`);
  if (recipe.model?.context_length)
    facts.push(`- Context length: ${recipe.model.context_length.toLocaleString()} tokens`);
  if (recipe.model?.architecture) facts.push(`- Architecture: ${recipe.model.architecture}`);
  if (recipe.model?.min_vllm_version)
    facts.push(`- Minimum vLLM version: ${recipe.model.min_vllm_version}`);
  if (hw) facts.push(`- Verified hardware: ${hw}`);
  if (facts.length > 0) {
    sections.push("");
    sections.push("## Model");
    sections.push("");
    sections.push(facts.join("\n"));
  }

  // Variants — important because quantised variants change the model_id
  // and the VRAM requirement, both things an assistant needs.
  if (recipe.variants && Object.keys(recipe.variants).length > 0) {
    sections.push("");
    sections.push("## Variants");
    sections.push("");
    for (const [name, v] of Object.entries(recipe.variants)) {
      const bits = [];
      if (v.precision) bits.push(`precision=${v.precision}`);
      if (v.vram_minimum_gb) bits.push(`vram_minimum_gb=${v.vram_minimum_gb}`);
      if (v.model_id) bits.push(`model_id=${v.model_id}`);
      sections.push(`- **${name}** — ${bits.join(", ") || "(default)"}`);
      if (v.description) sections.push(`  ${v.description}`);
    }
  }

  // A canonical `vllm serve` command from base_args. The site composes a
  // richer command client-side based on user choices; this is the
  // assistant-friendly baseline.
  const baseArgs = recipe.model?.base_args || [];
  const baseEnv = recipe.model?.base_env || {};
  if (baseArgs.length > 0 || Object.keys(baseEnv).length > 0) {
    sections.push("");
    sections.push("## Recommended baseline command");
    sections.push("");
    if (baseArgs.length > 0) {
      sections.push("```bash");
      sections.push(`vllm serve ${recipe.model?.model_id || recipe.hf_id} \\`);
      for (let i = 0; i < baseArgs.length; i++) {
        const trailing = i < baseArgs.length - 1 ? " \\" : "";
        sections.push(`  ${baseArgs[i]}${trailing}`);
      }
      sections.push("```");
    }
    if (Object.keys(baseEnv).length > 0) {
      sections.push("");
      sections.push("Required environment:");
      sections.push("");
      sections.push("```bash");
      for (const [k, v] of Object.entries(baseEnv)) {
        sections.push(`export ${k}=${v}`);
      }
      sections.push("```");
    }
  }

  // Extra install steps (DeepGEMM pins, transformers commits, etc).
  if (recipe.dependencies && recipe.dependencies.length > 0) {
    sections.push("");
    sections.push("## Additional dependencies");
    sections.push("");
    for (const dep of recipe.dependencies) {
      if (!dep?.command) continue;
      sections.push("```bash");
      sections.push(dep.command);
      sections.push("```");
      if (dep.note) sections.push(`> ${dep.note}`);
    }
  }

  // The hand-written guide is the most important section — pass it
  // through verbatim.
  if (recipe.guide && recipe.guide.trim()) {
    sections.push("");
    sections.push("## Guide");
    sections.push("");
    sections.push(recipe.guide.trim());
  }

  sections.push("");
  sections.push("## Source");
  sections.push("");
  sections.push(`- Canonical URL: ${canonical}`);
  sections.push(
    `- HuggingFace: https://huggingface.co/${recipe.hf_id}`,
  );
  sections.push(
    `- Edit on GitHub: https://github.com/vllm-project/recipes/edit/main/models/${recipe.hf_org}/${recipe.hf_repo}.yaml`,
  );

  return frontmatter + sections.join("\n") + "\n";
}

export async function GET(_request, { params }) {
  const { org, repo } = await params;
  const recipe = getRecipeByHfId(org, repo);
  if (!recipe) {
    return new Response("Not found", {
      status: 404,
      headers: { "Content-Type": "text/plain; charset=utf-8" },
    });
  }
  const body = renderRecipeMarkdown(recipe);
  const canonical = `${siteUrl}/${recipe.hf_org}/${recipe.hf_repo}`;
  return new Response(body, {
    status: 200,
    headers: {
      "Content-Type": "text/markdown; charset=utf-8",
      "Cache-Control": "public, max-age=3600, s-maxage=3600",
      Link: `<${canonical}>; rel="canonical"; type="text/html"`,
    },
  });
}
