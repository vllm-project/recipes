"use client";

import { usePathname } from "next/navigation";
import Link from "next/link";
import { FileQuestion, GitPullRequest, Search, ExternalLink, BookOpen, Sparkles } from "lucide-react";

// A missed URL here is almost always a model that isn't in the repo yet, not
// a typo. The page is designed around that — it surfaces the attempted
// `org/repo` back at the user, prefills a GitHub issue with it, and points to
// CONTRIBUTING.md for anyone who'd rather open a PR themselves.
export default function NotFound() {
  const pathname = usePathname() || "";
  const trimmed = pathname.replace(/^\/+|\/+$/g, "");
  const parts = trimmed.split("/").filter(Boolean);
  // Treat `/<org>/<repo>` as a probable model request. Ignore anything that
  // contains reserved segments or is clearly not a recipe URL.
  const looksLikeModel =
    parts.length === 2 && !["_next", "api", "og", "providers"].includes(parts[0]);
  const hfId = looksLikeModel ? `${parts[0]}/${parts[1]}` : "";

  const issueTitle = hfId
    ? `Recipe request: ${hfId}`
    : "Recipe request: <org>/<repo>";
  const issueBody = hfId
    ? `### Model\n\nhttps://huggingface.co/${hfId}\n\n### Hardware I'm targeting\n\n<!-- e.g. 8x H200, 4x MI355X, 2-node TP+PP -->\n\n### Precision / variant\n\n<!-- bf16, fp8, nvfp4, ... -->\n\n### Notes\n\n<!-- anything else you want the recipe to cover -->`
    : "### Model\n\n<HuggingFace URL>\n\n### Hardware\n\n### Precision / variant\n\n### Notes";
  const issueUrl = `https://github.com/vllm-project/recipes/issues/new?title=${encodeURIComponent(
    issueTitle
  )}&body=${encodeURIComponent(issueBody)}&labels=recipe-request`;
  const hfUrl = hfId ? `https://huggingface.co/${hfId}` : null;

  return (
    <main className="max-w-3xl mx-auto px-4 sm:px-6 py-16">
      {/* Header */}
      <div className="flex items-start gap-4 mb-8">
        <div className="w-12 h-12 rounded-xl bg-muted flex items-center justify-center shrink-0">
          <FileQuestion size={22} className="text-muted-foreground" />
        </div>
        <div>
          <p className="text-xs font-mono uppercase tracking-widest text-muted-foreground mb-1">
            404
          </p>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">
            {hfId ? "No recipe yet for this model" : "Page not found"}
          </h1>
          {hfId && (
            <p className="text-sm text-muted-foreground mt-2">
              You were looking for{" "}
              <code className="font-mono text-[12px] bg-muted/60 px-1.5 py-0.5 rounded">
                {hfId}
              </code>
              . Help the community — request it below or contribute one yourself.
            </p>
          )}
          {!hfId && pathname && (
            <p className="text-sm text-muted-foreground mt-2">
              <code className="font-mono text-[12px] bg-muted/60 px-1.5 py-0.5 rounded">
                {pathname}
              </code>{" "}
              doesn&apos;t match any recipe. Recipes live at{" "}
              <code className="font-mono text-[12px] bg-muted/60 px-1.5 py-0.5 rounded">
                /&lt;hf_org&gt;/&lt;hf_repo&gt;
              </code>
              , mirroring HuggingFace.
            </p>
          )}
        </div>
      </div>

      {/* CTAs */}
      <div className="grid sm:grid-cols-2 gap-3">
        <a
          href={issueUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="group flex flex-col gap-2 rounded-xl border border-border bg-card/40 hover:bg-card hover:border-vllm-blue/40 transition-all p-4"
        >
          <div className="flex items-center gap-2">
            <GitPullRequest size={14} className="text-vllm-blue" />
            <span className="text-sm font-semibold">
              {hfId ? "Request this recipe" : "Request a recipe"}
            </span>
            <ExternalLink size={11} className="ml-auto text-muted-foreground/50" />
          </div>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Opens a prefilled issue{hfId ? ` for ${hfId}` : ""}. Takes 30 seconds — maintainers
            pick up requests based on hardware availability.
          </p>
        </a>

        {/* Contribute card is a <div> (not <a>) because it hosts two distinct
            links — the header→CONTRIBUTING.md and an inline →SKILL.md. Nested
            anchors would be invalid HTML. */}
        <div className="group flex flex-col gap-2 rounded-xl border border-border bg-card/40 hover:bg-card hover:border-vllm-blue/40 transition-all p-4">
          <a
            href="https://github.com/vllm-project/recipes/blob/main/CONTRIBUTING.md"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2"
          >
            <BookOpen size={14} className="text-vllm-blue" />
            <span className="text-sm font-semibold hover:text-vllm-blue transition-colors">
              Contribute one yourself
            </span>
            <ExternalLink size={11} className="ml-auto text-muted-foreground/50" />
          </a>
          <p className="text-xs text-muted-foreground leading-relaxed">
            One YAML at{" "}
            <code className="font-mono text-[11px]">models/&lt;org&gt;/&lt;repo&gt;.yaml</code>
            .{" "}
            <a
              href="https://github.com/vllm-project/recipes/blob/main/CONTRIBUTING.md"
              target="_blank"
              rel="noopener noreferrer"
              className="underline underline-offset-2 decoration-muted-foreground/30 hover:decoration-foreground hover:text-foreground transition-colors"
            >
              CONTRIBUTING.md
            </a>{" "}
            has the schema, VRAM formula, and validation steps.
          </p>
          <p className="text-xs text-muted-foreground/80 leading-relaxed inline-flex items-center gap-1.5 mt-0.5">
            <Sparkles size={11} className="text-vllm-blue shrink-0" />
            <span>
              Using Claude Code? The{" "}
              <a
                href="https://github.com/vllm-project/recipes/blob/main/.claude/skills/add-recipe/SKILL.md"
                target="_blank"
                rel="noopener noreferrer"
                className="font-mono text-[11px] text-vllm-blue underline underline-offset-2 decoration-vllm-blue/40 hover:decoration-vllm-blue"
              >
                add-recipe
              </a>{" "}
              skill walks the whole process.
            </span>
          </p>
        </div>

        {hfUrl && (
          <a
            href={hfUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="group flex flex-col gap-2 rounded-xl border border-border bg-card/40 hover:bg-card hover:border-vllm-blue/40 transition-all p-4"
          >
            <div className="flex items-center gap-2">
              <ExternalLink size={14} className="text-vllm-blue" />
              <span className="text-sm font-semibold">Open on HuggingFace</span>
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Check the model card for vLLM support status and recommended hardware.
            </p>
          </a>
        )}

        <Link
          href="/"
          className="group flex flex-col gap-2 rounded-xl border border-border bg-card/40 hover:bg-card hover:border-vllm-blue/40 transition-all p-4"
        >
          <div className="flex items-center gap-2">
            <Search size={14} className="text-vllm-blue" />
            <span className="text-sm font-semibold">Browse all recipes</span>
          </div>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Search the full index by model name, provider, or task. Maybe the recipe
            exists under a slightly different path.
          </p>
        </Link>
      </div>
    </main>
  );
}
