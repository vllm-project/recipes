/**
 * Describe how sources.lock.json differs from HEAD, as a commit message
 * (used by .github/workflows/bump-sources.yml). Prints nothing when the lock
 * is unchanged. For each moved source it lists the upstream commits that
 * touched the recipe directory and the recipes they changed; if the GitHub
 * API is unavailable it falls back to the commit range alone.
 *
 *   Sync recipes from vllm-project/vllm@612e928016e2
 *
 *   vllm-project/vllm@main:recipes 7b569ec8953c → 612e928016e2
 *   https://github.com/vllm-project/vllm/compare/<old>...<new>
 *
 *   Commits:
 *   - vllm-project/vllm@56e67b46a3dd [Doc] Sync Kimi-K3 … (vllm-project/vllm#62761)
 *
 *   Recipes:
 *   - moonshotai/Kimi-K3 (modified): https://recipes.vllm.ai/moonshotai/Kimi-K3
 */

import fs from "fs";
import path from "path";
import { execFileSync } from "child_process";
import { LOCK_FILE, ROOT } from "./lib/content-sources.mjs";

const SITE = "https://recipes.vllm.ai";
const MAX_COMMIT_PAGES = 5;

function headLock() {
  try {
    const rel = path.relative(ROOT, LOCK_FILE);
    return JSON.parse(execFileSync("git", ["show", `HEAD:${rel}`], { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }));
  } catch {
    return {};
  }
}

async function gh(apiPath) {
  const headers = { Accept: "application/vnd.github+json", "User-Agent": "vllm-recipes-sync" };
  if (process.env.GITHUB_TOKEN) headers.Authorization = `Bearer ${process.env.GITHUB_TOKEN}`;
  const res = await fetch(`https://api.github.com${apiPath}`, { headers });
  if (!res.ok) throw new Error(`GitHub API ${apiPath}: ${res.status}`);
  return res.json();
}

// Commits in (prev, next] that touched the recipe directory, newest first.
// `prev` itself rarely touches it, so the walk is bounded by its date. Only
// the trailing "(#123)" is this repo's pull request; it is qualified so it
// doesn't link to the recipes repo. Other references are left as written.
async function recipeCommits(repo, dir, prev, next) {
  const since = (await gh(`/repos/${repo}/commits/${prev}`)).commit.committer.date;
  const out = [];
  for (let page = 1; page <= MAX_COMMIT_PAGES; page++) {
    const batch = await gh(`/repos/${repo}/commits?sha=${next}&path=${encodeURIComponent(dir)}&since=${since}&per_page=100&page=${page}`);
    for (const c of batch) {
      if (c.sha === prev) return out;
      const title = c.commit.message.split("\n")[0].replace(/\(#(\d+)\)$/, `(${repo}#$1)`);
      out.push(`- ${repo}@${c.sha.slice(0, 12)} ${title}`);
    }
    if (batch.length < 100) break;
  }
  return out;
}

// Recipe files that differ between two trees of the recipe directory.
async function changedRecipes(repo, prevTree, nextTree) {
  const blobs = async (tree) => new Map(
    (await gh(`/repos/${repo}/git/trees/${tree}?recursive=1`)).tree
      .filter((e) => e.type === "blob" && /^(models|overlays)\/.+\.ya?ml$/.test(e.path))
      .map((e) => [e.path, e.sha]),
  );
  const [before, after] = await Promise.all([blobs(prevTree), blobs(nextTree)]);
  const lines = [];
  for (const file of [...new Set([...before.keys(), ...after.keys()])].sort()) {
    const was = before.get(file);
    const is = after.get(file);
    if (was === is) continue;
    const kind = !was ? "added" : !is ? "removed" : "modified";
    const [dir, ...rest] = file.split("/");
    const id = rest.join("/").replace(/\.ya?ml$/, "");
    const label = dir === "overlays" ? `${id} overlay` : id;
    lines.push(`- ${label} (${kind})${is ? `: ${SITE}/${id}` : ""}`);
  }
  return lines;
}

const before = headLock();
const after = fs.existsSync(LOCK_FILE) ? JSON.parse(fs.readFileSync(LOCK_FILE, "utf8")) : {};
const subjects = [];
const sections = [];

for (const [channel, sources] of Object.entries(after)) {
  for (const [name, entry] of Object.entries(sources)) {
    const prev = before[channel]?.[name];
    const pinned = (e) => e && JSON.stringify({ ...e, resolved_at: undefined });
    if (pinned(prev) === pinned(entry)) continue;
    const short = entry.sha.slice(0, 12);
    subjects.push(`${entry.repo}@${short}`);
    const from = prev ? prev.sha.slice(0, 12) : "(new)";
    const moved = prev && (prev.path !== entry.path || prev.repo !== entry.repo || prev.ref !== entry.ref)
      ? ` (was ${prev.repo}@${prev.ref}:${prev.path})` : "";
    const lines = [`${entry.repo}@${entry.ref}:${entry.path} ${from} → ${short}${moved} [${channel}.${name}]`];
    const sameRepo = prev && prev.repo === entry.repo && prev.path === entry.path && prev.sha !== entry.sha;
    lines.push(sameRepo
      ? `https://github.com/${entry.repo}/compare/${prev.sha}...${entry.sha}`
      : `https://github.com/${entry.repo}/tree/${entry.sha}/${entry.path}`);
    if (sameRepo) {
      try {
        const commits = await recipeCommits(entry.repo, entry.path, prev.sha, entry.sha);
        if (commits.length) lines.push("", "Commits:", ...commits);
        if (prev.tree_sha && entry.tree_sha && prev.tree_sha !== entry.tree_sha) {
          const recipes = await changedRecipes(entry.repo, prev.tree_sha, entry.tree_sha);
          if (recipes.length) lines.push("", "Recipes:", ...recipes);
        }
      } catch (err) {
        console.error(`lock-changes: ${err.message}; listing the range only`);
      }
    }
    sections.push(lines.join("\n"));
  }
}
for (const [channel, sources] of Object.entries(before)) {
  for (const [name, entry] of Object.entries(sources)) {
    if (after[channel]?.[name]) continue;
    subjects.push(`removed ${channel}.${name}`);
    sections.push(`${entry.repo}@${entry.ref}:${entry.path} removed [${channel}.${name}]`);
  }
}

if (subjects.length) {
  console.log(`Sync recipes from ${subjects.join(", ")}\n\n${sections.join("\n\n")}`);
}
