"use client";

import { useEffect, useState } from "react";

// Set at build time from public/preview.json, which only preview branches
// (scripts/preview-prs.mjs) carry; empty on the production site.
const preview = process.env.RECIPES_PREVIEW ? JSON.parse(process.env.RECIPES_PREVIEW) : null;

const short = (sha) => sha.slice(0, 7);

function statusText(pull) {
  if (!pull) return null;
  if (pull.merged_at) return "The PR is merged.";
  if (pull.state === "closed") return "The PR is closed.";
  if (pull.head.sha === preview.sha) return "Up to date with the PR's latest commit.";
  return `The PR has newer commits (${short(pull.head.sha)}); this preview rebuilds within about 10–20 minutes.`;
}

export function PreviewBanner() {
  const [pull, setPull] = useState(null);

  useEffect(() => {
    if (!preview) return;
    // Unauthenticated API (60 requests/hour per IP); on failure the banner
    // just omits the status.
    fetch(`https://api.github.com/repos/${preview.repo}/pulls/${preview.pr}`)
      .then((res) => (res.ok ? res.json() : null))
      .then(setPull)
      .catch(() => {});
  }, []);

  if (!preview) return null;
  const prUrl = `https://github.com/${preview.repo}/pull/${preview.pr}`;
  const outdated = pull && !pull.merged_at && pull.state === "open" && pull.head.sha !== preview.sha;

  return (
    <div className="border-b border-amber-500/40 bg-amber-500/10 text-[13px] leading-snug">
      <div className="max-w-[1480px] mx-auto px-4 sm:px-6 py-2 flex flex-wrap items-baseline gap-x-2 gap-y-1">
        <span className="font-semibold text-amber-600 dark:text-amber-400">Preview</span>
        <span>
          of <a href={prUrl} className="underline hover:text-vllm-blue">{preview.repo}#{preview.pr}</a> at{" "}
          <a href={`${prUrl}/commits/${preview.sha}`} className="font-mono underline hover:text-vllm-blue">
            {short(preview.sha)}
          </a>
          .
        </span>
        {pull && (
          <span className={outdated ? "font-semibold text-amber-600 dark:text-amber-400" : "text-muted-foreground"}>
            {statusText(pull)}
          </span>
        )}
      </div>
    </div>
  );
}
