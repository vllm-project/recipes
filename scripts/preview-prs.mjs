/**
 * Keep one preview branch per open source-repo pull request that changes
 * recipes, so Vercel deploys each one as a branch preview.
 *
 *   node scripts/preview-prs.mjs --source vllm [--label recipe] [--max 20] [--dry-run]
 *   node scripts/preview-prs.mjs --source vllm --pr 63244 [--dry-run]   # just these PRs
 *
 * A preview branch `preview/<source>-pr-<N>` is this checkout's HEAD plus a lock
 * that pins the source to the PR head (`ref` is unchanged, so sync-sources
 * fetches the pinned commit as usual) and `public/preview.json`. It is created
 * only with `--pr`: the source repo dispatches that for PRs from authors its CI
 * trusts, or a maintainer runs it. Without `--pr`, open PRs carrying `--label`
 * get their existing branch refreshed, and branches of closed or unlabeled PRs
 * are deleted. A branch is rewritten only when the PR's recipe directory
 * (tree_sha) changed. The source repo links the preview from its PR:
 *   https://vllm-recipes-git-preview-<source>-pr-<N>-inferact-inc.vercel.app
 *
 * Only the PR's commit and tree ids are handled here; nothing from the PR is
 * run. Needs GITHUB_TOKEN and a checkout that can push to this repository.
 */

import fs from "fs";
import path from "path";
import { execFileSync } from "child_process";
import { ROOT, loadLock, loadSources, parseArgs, writeLock } from "./lib/content-sources.mjs";

const args = parseArgs(process.argv.slice(2));
const source = args.source;
if (!source) {
  console.error("usage: preview-prs.mjs --source <name> [--label <label>] [--max <n>] [--pr <n,n>] [--dry-run]");
  process.exit(2);
}
const label = args.label || "recipe";
const max = Number(args.max || 20);
const dryRun = Boolean(args["dry-run"]);
const selfRepo = process.env.GITHUB_REPOSITORY || "vllm-project/recipes";
const prefix = `preview/${source}-pr-`;
const MARKER = path.join(ROOT, "public", "preview.json");

const { defaultChannel, channels } = loadSources();
const spec = channels[defaultChannel]?.[source];
if (!spec?.repo) throw new Error(`sources.yaml: ${defaultChannel}.${source} is not a remote source`);
const subPath = path.posix.normalize(spec.path || ".").replace(/\/$/, "");

async function gh(apiPath, { method = "GET", raw = false, allow404 = false } = {}) {
  const headers = {
    Accept: raw ? "application/vnd.github.raw+json" : "application/vnd.github+json",
    "User-Agent": "vllm-recipes-preview",
  };
  if (process.env.GITHUB_TOKEN) headers.Authorization = `Bearer ${process.env.GITHUB_TOKEN}`;
  const res = await fetch(`https://api.github.com${apiPath}`, { method, headers });
  if (allow404 && res.status === 404) return null;
  if (!res.ok) throw new Error(`GitHub API ${method} ${apiPath}: ${res.status} ${await res.text()}`);
  return raw ? res.text() : res.json();
}

function git(...argv) {
  return execFileSync("git", argv, { cwd: ROOT, encoding: "utf8", stdio: ["ignore", "pipe", "inherit"] }).trim();
}

async function labeledPrs() {
  const numbers = [];
  const q = encodeURIComponent(`repo:${spec.repo} is:pr is:open label:"${label}"`);
  for (let page = 1; ; page++) {
    const res = await gh(`/search/issues?q=${q}&per_page=100&page=${page}`);
    numbers.push(...res.items.map((item) => item.number));
    if (res.items.length < 100) return numbers;
  }
}

async function recipeTree(sha) {
  const parent = path.posix.dirname(subPath);
  const entries = await gh(`/repos/${spec.repo}/contents/${parent === "." ? "" : parent}?ref=${sha}`);
  return entries.find((e) => e.name === path.posix.basename(subPath) && e.type === "dir")?.sha || null;
}

async function previewedTree(branch) {
  const marker = await gh(`/repos/${selfRepo}/contents/public/preview.json?ref=${encodeURIComponent(branch)}`,
    { raw: true, allow404: true });
  return marker ? JSON.parse(marker).tree_sha : null;
}

const base = git("rev-parse", "HEAD");

function pushPreview(pr, sha, treeSha) {
  const branch = `${prefix}${pr}`;
  git("checkout", "-q", "--force", "--detach", base);
  const lock = loadLock();
  const entry = lock[defaultChannel]?.[source];
  if (!entry) throw new Error(`sources.lock.json has no ${defaultChannel}.${source} entry`);
  lock[defaultChannel][source] = { ...entry, sha, tree_sha: treeSha, resolved_at: new Date().toISOString() };
  writeLock(lock);
  fs.writeFileSync(MARKER, JSON.stringify({ repo: spec.repo, pr, sha, tree_sha: treeSha }, null, 2) + "\n");
  git("add", "sources.lock.json", MARKER);
  git("commit", "-q", "-s", "-m", `Preview ${spec.repo}#${pr} at ${sha.slice(0, 12)}`);
  git("push", "-q", "--force", "origin", `HEAD:refs/heads/${branch}`);
  git("checkout", "-q", "--force", "--detach", base);
}

const existing = new Set(
  (await gh(`/repos/${selfRepo}/git/matching-refs/heads/${prefix}`))
    .map((ref) => Number(ref.ref.slice(`refs/heads/${prefix}`.length))),
);
// A dispatch names one PR; cap a list so one request can't fan out widely.
const MAX_PRS = 10;
const only = args.pr ? [...new Set(String(args.pr).split(",").map(Number))] : null;
if (only && (only.length > MAX_PRS || !only.every(Number.isInteger))) {
  throw new Error(`--pr takes up to ${MAX_PRS} PR numbers, got "${args.pr}"`);
}
const wanted = only || await labeledPrs();
const keep = new Set(wanted);
let rebuilt = 0;

for (const pr of wanted) {
  const pull = await gh(`/repos/${spec.repo}/pulls/${pr}`);
  if (pull.state !== "open") {
    keep.delete(pr);
    if (!existing.has(pr)) {
      console.log(`  - #${pr}: closed, no preview`);
      continue;
    }
    console.log(`  ✗ #${pr}: closed, deleting ${prefix}${pr}`);
    if (!dryRun) git("push", "-q", "origin", "--delete", `refs/heads/${prefix}${pr}`);
    continue;
  }
  if (!only && !existing.has(pr)) {
    console.log(`  - #${pr}: no preview requested yet`);
    continue;
  }
  const sha = pull.head.sha;
  const treeSha = await recipeTree(sha);
  if (!treeSha) {
    console.log(`  - #${pr}: no ${subPath}/ at ${sha.slice(0, 12)}`);
    keep.delete(pr);
    continue;
  }
  if (existing.has(pr) && await previewedTree(`${prefix}${pr}`) === treeSha) {
    console.log(`  ✓ #${pr}: ${subPath}/ unchanged`);
    continue;
  }
  if (rebuilt >= max) {
    console.log(`  … #${pr}: over --max ${max}, next run`);
    continue;
  }
  console.log(`  ↑ #${pr} → ${sha.slice(0, 12)} (${subPath}/ ${treeSha.slice(0, 12)})`);
  if (!dryRun) pushPreview(pr, sha, treeSha);
  rebuilt++;
}

if (!only) {
  for (const pr of existing) {
    if (keep.has(pr)) continue;
    console.log(`  ✗ #${pr}: closed or unlabeled, deleting ${prefix}${pr}`);
    if (!dryRun) git("push", "-q", "origin", "--delete", `refs/heads/${prefix}${pr}`);
  }
}
