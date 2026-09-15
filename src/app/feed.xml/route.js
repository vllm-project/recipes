// /feed.xml — Atom 1.0 changelog for recipes.vllm.ai
//
// Why Atom and not RSS 2.0: RSS dates are RFC-822, which is locale- and
// timezone-sensitive and a recurring parsing footgun. Atom uses
// RFC-3339 / ISO-8601 natively, which both readers and AI-assistant
// fetchers handle cleanly. Either format would satisfy the PROD-14
// spec; Atom has fewer ways to be subtly wrong.
//
// Entries are derived from two git signals against the repo root:
//
//   1. New recipes — first commit (--diff-filter=A) touching a
//      models/<org>/<repo>.yaml file in the last 180 days.
//   2. Significantly-updated recipes — most recent commit touching a
//      models/<org>/<repo>.yaml file in the last 60 days, deduped by
//      file (one entry per recipe).
//
// Same git-history-only derivation as the docs hook on
// vllm-project/vllm — no extra deps, no network calls.
//
// Statically generated at build time via `force-static`.

import { execFileSync } from "child_process";
import { getRecipeByHfId, getAllRecipes } from "@/lib/recipes";
import { siteUrl } from "@/lib/site-url";

export const dynamic = "force-static";

const PAGE_WINDOW_DAYS = 60;
const NEW_RECIPE_WINDOW_DAYS = 180;
const MAX_ENTRIES = 40;

function repoRoot() {
  return process.cwd();
}

function runGit(args) {
  try {
    return execFileSync("git", args, {
      cwd: repoRoot(),
      encoding: "utf-8",
      stdio: ["ignore", "pipe", "ignore"],
      timeout: 60_000,
    });
  } catch {
    return "";
  }
}

function sinceDate(days) {
  const d = new Date(Date.now() - days * 24 * 60 * 60 * 1000);
  return d.toISOString().slice(0, 10);
}

function escapeXml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function escapeAttr(s) {
  return escapeXml(s).replace(/"/g, "&quot;");
}

// Parse `git log --name-only --diff-filter=A|M --pretty=format:<sentinel>`
// into a list of { hash, isoDate, subject, paths[] } entries.
function parseGitLog(raw) {
  const entries = [];
  let cur = null;
  for (const line of raw.split("\n")) {
    if (line.startsWith("__COMMIT__")) {
      if (cur) entries.push(cur);
      const [hash, iso, ...subjectParts] = line.slice("__COMMIT__".length).split("\x00");
      cur = { hash, isoDate: iso, subject: subjectParts.join("\x00"), paths: [] };
      continue;
    }
    if (cur && line.trim()) {
      cur.paths.push(line.trim());
    }
  }
  if (cur) entries.push(cur);
  return entries;
}

function gatherUpdatedRecipes() {
  const raw = runGit([
    "log",
    `--since=${sinceDate(PAGE_WINDOW_DAYS)}`,
    "--diff-filter=AM",
    "--name-only",
    "--pretty=format:__COMMIT__%H%x00%cI%x00%s",
    "--",
    "models/",
  ]);
  const commits = parseGitLog(raw);
  const seen = new Map();
  for (const c of commits) {
    for (const p of c.paths) {
      if (!p.startsWith("models/") || !p.endsWith(".yaml")) continue;
      if (seen.has(p)) continue;
      // models/<org>/<repo>.yaml → <org>, <repo>
      const rel = p.slice("models/".length, -".yaml".length);
      const parts = rel.split("/");
      if (parts.length !== 2) continue;
      const [org, repo] = parts;
      const recipe = getRecipeByHfId(org, repo);
      if (!recipe) continue;
      seen.set(p, {
        kind: "update",
        org,
        repo,
        url: `${siteUrl}/${org}/${repo}`,
        title: `${org}/${repo} — recipe updated`,
        summary: c.subject || `Recipe ${org}/${repo} updated.`,
        id: `${siteUrl}/feed-update-${c.hash}-${org}-${repo}`,
        updated: c.isoDate,
      });
    }
  }
  return [...seen.values()];
}

function gatherNewRecipes() {
  const raw = runGit([
    "log",
    `--since=${sinceDate(NEW_RECIPE_WINDOW_DAYS)}`,
    "--diff-filter=A",
    "--name-only",
    "--pretty=format:__COMMIT__%H%x00%cI%x00%s",
    "--",
    "models/",
  ]);
  const commits = parseGitLog(raw);
  const out = [];
  const seen = new Set();
  for (const c of commits) {
    for (const p of c.paths) {
      if (!p.startsWith("models/") || !p.endsWith(".yaml")) continue;
      if (seen.has(p)) continue;
      seen.add(p);
      const rel = p.slice("models/".length, -".yaml".length);
      const parts = rel.split("/");
      if (parts.length !== 2) continue;
      const [org, repo] = parts;
      const recipe = getRecipeByHfId(org, repo);
      if (!recipe) continue;
      out.push({
        kind: "new",
        org,
        repo,
        url: `${siteUrl}/${org}/${repo}`,
        title: `${org}/${repo} — new recipe`,
        summary:
          recipe.meta?.description || c.subject || `New recipe for ${org}/${repo}.`,
        id: `${siteUrl}/feed-new-${c.hash}-${org}-${repo}`,
        updated: c.isoDate,
      });
    }
  }
  return out;
}

function renderEntry(entry) {
  return [
    "  <entry>",
    `    <title>${escapeXml(entry.title)}</title>`,
    `    <link href="${escapeAttr(entry.url)}" rel="alternate" type="text/html"/>`,
    `    <id>${escapeXml(entry.id)}</id>`,
    `    <updated>${escapeXml(entry.updated)}</updated>`,
    `    <category term="${escapeAttr(entry.kind)}"/>`,
    `    <summary type="html">${escapeXml(entry.summary)}</summary>`,
    "  </entry>",
    "",
  ].join("\n");
}

function fallbackEntries() {
  // When git history isn't available at build (shallow clone, etc.),
  // fall back to a feed seeded from the recipe metadata so /feed.xml
  // never 404s. Surface the 40 most-recently-updated recipes per their
  // own date_updated stamp.
  const recipes = getAllRecipes().slice(0, MAX_ENTRIES);
  return recipes
    .filter((r) => r.meta?.date_updated)
    .map((r) => ({
      kind: "update",
      org: r.hf_org,
      repo: r.hf_repo,
      url: `${siteUrl}/${r.hf_org}/${r.hf_repo}`,
      title: `${r.hf_id} — recipe updated`,
      summary: r.meta?.description || `Recipe ${r.hf_id} updated.`,
      id: `${siteUrl}/feed-meta-${r.hf_org}-${r.hf_repo}-${r.meta.date_updated}`,
      // ISO-8601 with timezone-aware noon UTC so Atom is happy.
      updated: `${r.meta.date_updated}T12:00:00Z`,
    }));
}

export async function GET() {
  const updated = gatherUpdatedRecipes();
  const created = gatherNewRecipes();

  // De-dupe: a recipe that appears in `created` shouldn't also show
  // up as an `update` for the same commit.
  const createdKeys = new Set(created.map((e) => `${e.org}/${e.repo}`));
  const dedupedUpdates = updated.filter(
    (e) => !createdKeys.has(`${e.org}/${e.repo}`),
  );

  let entries = [...created, ...dedupedUpdates];
  if (entries.length === 0) {
    entries = fallbackEntries();
  }

  entries.sort((a, b) => (a.updated < b.updated ? 1 : a.updated > b.updated ? -1 : 0));
  entries = entries.slice(0, MAX_ENTRIES);

  const feedUrl = `${siteUrl}/feed.xml`;
  const lastUpdated =
    entries[0]?.updated || new Date().toISOString().replace(/\.\d{3}Z$/, "Z");

  const xml = [
    `<?xml version="1.0" encoding="UTF-8"?>`,
    `<feed xmlns="http://www.w3.org/2005/Atom">`,
    `  <title>vLLM Recipes Changelog</title>`,
    `  <subtitle>New and updated vLLM serving recipes.</subtitle>`,
    `  <link href="${escapeAttr(feedUrl)}" rel="self" type="application/atom+xml"/>`,
    `  <link href="${escapeAttr(siteUrl + "/")}" rel="alternate" type="text/html"/>`,
    `  <id>${escapeXml(feedUrl)}</id>`,
    `  <updated>${escapeXml(lastUpdated)}</updated>`,
    `  <generator uri="https://github.com/vllm-project/recipes">recipes.vllm.ai feed</generator>`,
    `  <author><name>vLLM Project</name><uri>https://vllm.ai</uri></author>`,
    "",
    ...entries.map(renderEntry),
    `</feed>`,
    "",
  ].join("\n");

  return new Response(xml, {
    status: 200,
    headers: {
      "Content-Type": "application/atom+xml; charset=utf-8",
      "Cache-Control": "public, max-age=3600, s-maxage=3600",
    },
  });
}
