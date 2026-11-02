import Markdown from "react-markdown";
import { getRecipeByHfId, getAllRecipes } from "@/lib/recipes";
import { siteUrl } from "@/lib/site-url";
import { guideRehypePluginsFor, guideRemarkPlugins } from "@/lib/guide-markdown";

export const metadata = {
  title: "API",
  description: "Every vLLM recipe as JSON or Markdown, for scripts and AI agents: any recipe page URL plus .json or .md renders that configuration.",
  alternates: { canonical: "/api" },
};

// The one place the JSON API is explained for people; the endpoint list in
// scripts/build-recipes-api.mjs is the source it follows.
function apiMarkdown(example) {
  const page = `${siteUrl}/${example}`;
  const q = "?hardware=b200&strategy=single_node_dep";
  return `
Every recipe on this site is also available to scripts and AI agents — no key, the same data the pages render.

## Any page, as Markdown or JSON

Take a recipe page URL, configured in the builder however you like, and add \`.md\` or \`.json\`. The query string carries the configuration, so the response is the command for exactly what the page shows:

\`\`\`text
${page}${q}         ← the page
${page}.md${q}      ← that configuration as Markdown
${page}.json${q}    ← that configuration as JSON
\`\`\`

Adding the extension to the end of the whole URL (\`…&strategy=single_node_dep.md\`) works too. The **Copy page** menu on every recipe page links both forms for the current selection.

- **Markdown** leads with the rendered command (marked authoritative), then the notes and known issues that apply to this configuration, the tuned preset in effect, and the guide as background — the form to hand to an AI assistant.
- **JSON** has the same shape as the static per-hardware renderings below, plus \`selection\` (what was rendered), \`page_url\` and \`markdown_url\`.

\`\`\`bash
# The serve command for DeepSeek-R1 on 8× B200, data + expert parallel
curl -s "${page}.json${q}" | jq -r .command

# Multi-node: one command per node — never replay one on every node
curl -s "${page}.json?hardware=h100&strategy=multi_node_tp&nodes=2" | jq -r '.head_command, .worker_commands[]'
\`\`\`

### Query parameters

The same ones the page URL uses. Values the recipe can't run fall back to its defaults, and \`selection.adjusted\` (JSON) or a note under the Configuration line (Markdown) says what changed and why.

| Parameter | Value |
|---|---|
| \`hardware\` | A hardware profile id, e.g. \`h200\`, \`b200\`, \`gb300\`, \`mi355x\` ([taxonomy](${siteUrl}/taxonomy.json)) |
| \`variant\` | A variant key of the recipe (\`default\`, \`nvfp4\`, …) |
| \`strategy\` | A serving strategy id ([strategies](${siteUrl}/strategies.json)); multi-node ones also need \`nodes\` |
| \`nodes\` | Node count for a multi-node strategy (nodes per instance under Mooncake) |
| \`features\` | Comma-separated feature keys to turn on (\`tool_calling,reasoning\`, …); absent = the recipe default |
| \`preset\`, \`point\` | A tuned preset and its operating point; \`preset=baseline\` drops a default preset |
| \`kv_offload\` | A KV offload option (\`simple\`, \`lmcache\`, … — \`kv_offload\` in the [taxonomy](${siteUrl}/taxonomy.json)) or a Mooncake topology (\`kv_store_distributed_mooncake\`, \`kv_store_centralized_mooncake\`) |
| \`prefill_nodes\`, \`decode_nodes\`, \`prefill_mode\`, \`decode_mode\`, \`pd_router\`, \`pd_connector\` | Prefill/decode disaggregation (\`strategy=pd_cluster\`) |

Each recipe's Markdown lists the values it accepts under **Other configurations**.

## Static files

Built with the site; cheap to fetch in bulk.

| Path | Contents |
|---|---|
| [\`/models.json\`](${siteUrl}/models.json) | Every recipe: id, title, provider, page and JSON links |
| \`/<hf_id>.json\` | One recipe: metadata, variants, features, guide, and \`recommended_command\` with a \`by_hardware\` index |
| \`/<hf_id>/hw/<hardware>.json\` | The recommended rendering on one hardware profile, with \`alternatives\` per strategy |
| \`/<hf_id>/hw/<hardware>/strategies/<strategy>.json\` | One hardware × strategy rendering |
| \`/<hf_id>/presets/<preset>.json\` | A tuned preset, rendered at each of its points |
| [\`/strategies.json\`](${siteUrl}/strategies.json), [\`/taxonomy.json\`](${siteUrl}/taxonomy.json) | Serving strategies; hardware profiles and the rest of the vocabulary |
| [\`/recipe.schema.json\`](${siteUrl}/recipe.schema.json) | JSON Schema of a recipe YAML |

## For AI assistants

- [\`/llms.txt\`](${siteUrl}/llms.txt) — an index of every recipe, linking its Markdown form.
- [\`/llms-full.txt\`](${siteUrl}/llms-full.txt) — the full text of every recipe in one file.
`;
}

export default function ApiPage() {
  const example = getRecipeByHfId("deepseek-ai", "DeepSeek-R1")?.hf_id || getAllRecipes()[0]?.hf_id;
  return (
    <main className="max-w-3xl mx-auto px-4 sm:px-6 py-8">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold tracking-tight">Recipes API</h1>
        <p className="text-sm text-muted-foreground mt-1">JSON and Markdown for scripts and AI agents.</p>
      </header>
      <div className="guide-content">
        <Markdown remarkPlugins={guideRemarkPlugins} rehypePlugins={guideRehypePluginsFor("")}>
          {apiMarkdown(example)}
        </Markdown>
      </div>
    </main>
  );
}
