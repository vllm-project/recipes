/**
 * Fetch the remote recipe sources pinned in sources.lock.json into .sources/.
 *
 *   node scripts/sync-sources.mjs [--channel <name> | --all]   # fetch what the lock pins
 *   node scripts/sync-sources.mjs --update [--channel ...]     # re-resolve refs, rewrite the lock
 *
 * The lock records the commit (`sha`) and the tree hash of the source's
 * recipe directory (`tree_sha`). `--update` only moves `sha` when `tree_sha`
 * changed, so commits that don't touch the recipe directory never bump the
 * lock. Fetching is a shallow, blobless, sparse git checkout of just `path`.
 * Sources with `local:` or a developer override are skipped.
 *
 * GITHUB_TOKEN is optional (raises the API rate limit).
 */

import fs from "fs";
import path from "path";
import { execFileSync } from "child_process";
import {
  CACHE_DIR, cacheDirFor, isRemote, loadLock, loadOverrides, loadSources, parseArgs, writeLock,
} from "./lib/content-sources.mjs";

const args = parseArgs(process.argv.slice(2));
const { defaultChannel, channels } = loadSources();
const overrides = loadOverrides(args.override);
const targets = args.all ? Object.keys(channels) : [args.channel || process.env.RECIPES_CHANNEL || defaultChannel];

async function gh(apiPath) {
  const headers = { Accept: "application/vnd.github+json", "User-Agent": "vllm-recipes-sync" };
  if (process.env.GITHUB_TOKEN) headers.Authorization = `Bearer ${process.env.GITHUB_TOKEN}`;
  const res = await fetch(`https://api.github.com${apiPath}`, { headers });
  if (!res.ok) throw new Error(`GitHub API ${apiPath}: ${res.status} ${await res.text()}`);
  return res.json();
}

function normalizedPath(p) {
  const n = path.posix.normalize(p || ".").replace(/\/$/, "");
  return n === "." ? "" : n;
}

// → { sha, tree_sha } for `ref`, where tree_sha identifies the recipe dir.
async function resolveRef(repo, ref, subPath) {
  const commit = await gh(`/repos/${repo}/commits/${encodeURIComponent(ref)}`);
  const sha = commit.sha;
  const sub = normalizedPath(subPath);
  if (!sub) return { sha, tree_sha: commit.commit.tree.sha };
  const parent = path.posix.dirname(sub);
  const listing = await gh(
    `/repos/${repo}/contents/${parent === "." ? "" : parent}?ref=${sha}`
  );
  const entry = Array.isArray(listing)
    ? listing.find((e) => e.name === path.posix.basename(sub) && e.type === "dir")
    : null;
  if (!entry) throw new Error(`${repo}@${sha.slice(0, 12)} has no directory ${sub}/`);
  return { sha, tree_sha: entry.sha };
}

// True when `ref` was force-pushed past the locked commit: the recipe dir is
// unchanged, but the old commit is no longer in the branch's history and may
// be garbage-collected, so the lock must move even though tree_sha didn't.
async function rewritten(repo, oldSha, newSha) {
  if (oldSha === newSha) return false;
  try {
    const cmp = await gh(`/repos/${repo}/compare/${oldSha}...${newSha}`);
    return cmp.status !== "ahead" && cmp.status !== "identical";
  } catch {
    return true;
  }
}

function git(cwd, ...argv) {
  execFileSync("git", argv, { cwd, stdio: ["ignore", "ignore", "pipe"] });
}

function fetchSource(repo, sha, subPath) {
  const dest = cacheDirFor(repo, sha, subPath);
  if (fs.existsSync(dest)) return false;
  fs.mkdirSync(CACHE_DIR, { recursive: true });
  const tmp = fs.mkdtempSync(path.join(CACHE_DIR, ".tmp-"));
  try {
    const sub = normalizedPath(subPath);
    git(tmp, "init", "-q");
    git(tmp, "remote", "add", "origin", `https://github.com/${repo}.git`);
    if (sub) git(tmp, "sparse-checkout", "set", "--no-cone", `/${sub}/`);
    git(tmp, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", sha);
    git(tmp, "checkout", "-q", "FETCH_HEAD");
    fs.rmSync(path.join(tmp, ".git"), { recursive: true, force: true });
    fs.renameSync(sub ? path.join(tmp, sub) : tmp, dest);
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
  return true;
}

const lock = loadLock();
let lockChanged = false;

for (const channel of targets) {
  const sources = channels[channel];
  if (!sources) throw new Error(`sources.yaml has no channel "${channel}"`);
  for (const [name, spec] of Object.entries(sources)) {
    if (!isRemote(spec, name, overrides)) continue;
    const label = `${channel}.${name}`;
    let entry = lock[channel]?.[name];
    const stale = !entry || entry.repo !== spec.repo || entry.ref !== spec.ref
      || normalizedPath(entry.path) !== normalizedPath(spec.path);
    if (stale && !args.update) {
      // Never float: a build must deploy exactly what a lock bump validated.
      throw new Error(`${label}: sources.lock.json has no pin for ${spec.repo}@${spec.ref}:${spec.path || "."} `
        + `(new source, or its repo/ref/path changed) — run: node scripts/sync-sources.mjs --update --channel ${channel}`);
    }
    if (args.update || stale) {
      const resolved = await resolveRef(spec.repo, spec.ref, spec.path);
      if (stale || resolved.tree_sha !== entry.tree_sha || await rewritten(spec.repo, entry.sha, resolved.sha)) {
        entry = {
          repo: spec.repo, ref: spec.ref, path: spec.path || ".",
          ...resolved, resolved_at: new Date().toISOString(),
        };
        lock[channel] = { ...lock[channel], [name]: entry };
        lockChanged = true;
        console.log(`  ↑ ${label} → ${spec.repo}@${entry.sha.slice(0, 12)}`);
      }
    }
    const fetched = fetchSource(spec.repo, entry.sha, spec.path);
    console.log(`  ${fetched ? "↓" : "✓"} ${label} ${spec.repo}@${entry.sha.slice(0, 12)}`);
  }
  // Drop lock entries for sources removed from (or made local in) sources.yaml.
  for (const name of Object.keys(lock[channel] || {})) {
    if (!sources[name] || !isRemote(sources[name], name, {})) {
      delete lock[channel][name];
      lockChanged = true;
    }
  }
}

if (lockChanged) {
  writeLock(lock);
  console.log("✓ sources.lock.json updated");
} else {
  console.log("✓ sources in sync");
}
