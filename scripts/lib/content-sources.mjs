// Shared by sync-sources.mjs and assemble-content.mjs: reads sources.yaml,
// sources.lock.json and developer overrides, and resolves each source of a
// channel to a local directory.

import fs from "fs";
import os from "os";
import path from "path";
import yaml from "js-yaml";

export const ROOT = process.cwd();
export const SOURCES_FILE = path.resolve(ROOT, process.env.RECIPES_SOURCES || "sources.yaml");
export const LOCK_FILE = path.join(path.dirname(SOURCES_FILE), "sources.lock.json");
// Lock and developer overrides sit beside whichever sources file is in use.
export const LOCAL_FILE = path.join(path.dirname(SOURCES_FILE), "sources.local.yaml");
export const CACHE_DIR = path.join(ROOT, ".sources");

const ROLES = new Set(["core", "owner", "overlay"]);

export function loadSources() {
  const cfg = yaml.load(fs.readFileSync(SOURCES_FILE, "utf8")) || {};
  const channels = cfg.channels || {};
  for (const [channel, sources] of Object.entries(channels)) {
    for (const [name, spec] of Object.entries(sources || {})) {
      if (!ROLES.has(spec.role)) {
        throw new Error(`sources.yaml: ${channel}.${name}.role must be one of ${[...ROLES].join("|")}`);
      }
      if (!spec.local && !(spec.repo && spec.ref)) {
        throw new Error(`sources.yaml: ${channel}.${name} needs repo + ref (or local)`);
      }
    }
  }
  return { defaultChannel: cfg.default_channel || "latest", channels };
}

export function loadLock() {
  try {
    return JSON.parse(fs.readFileSync(LOCK_FILE, "utf8"));
  } catch {
    return {};
  }
}

export function writeLock(lock) {
  fs.writeFileSync(LOCK_FILE, JSON.stringify(lock, null, 2) + "\n");
}

function expandHome(p) {
  return p.startsWith("~/") ? path.join(os.homedir(), p.slice(2)) : p;
}

// sources.local.yaml `overrides: { <source>: <dir> }`, then the
// RECIPES_OVERRIDES env var ("name=dir,name=dir" — what validate-source.mjs
// passes to child scripts), then `--override name=dir` CLI flags; later wins.
export function loadOverrides(cliOverrides = []) {
  const out = {};
  if (fs.existsSync(LOCAL_FILE)) {
    const local = yaml.load(fs.readFileSync(LOCAL_FILE, "utf8")) || {};
    for (const [name, dir] of Object.entries(local.overrides || {})) {
      out[name] = path.resolve(ROOT, expandHome(String(dir)));
    }
  }
  const fromEnv = (process.env.RECIPES_OVERRIDES || "").split(",").map((x) => x.trim()).filter(Boolean);
  for (const flag of [...fromEnv, ...cliOverrides]) {
    const i = flag.indexOf("=");
    if (i <= 0) throw new Error(`--override expects name=dir, got "${flag}"`);
    out[flag.slice(0, i)] = path.resolve(ROOT, expandHome(flag.slice(i + 1)));
  }
  return out;
}

// One cache dir per (repo, commit, recipe dir): two sources may read
// different directories of the same commit.
export function cacheDirFor(repo, sha, subPath = ".") {
  const sub = path.posix.normalize(subPath || ".").replace(/\/$/, "");
  const suffix = sub === "." ? "" : `__${sub.replace(/[^A-Za-z0-9._-]+/g, "_")}`;
  return path.join(CACHE_DIR, `${repo.replace("/", "__")}@${sha}${suffix}`);
}

export function isRemote(spec, name, overrides) {
  return !overrides[name] && !spec.local;
}

// Local directory holding a source's models/ + overlays/, and the commit it
// reflects (null for local checkouts).
export function resolveSourceDir(channel, name, spec, lock, overrides) {
  if (overrides[name]) return { dir: overrides[name], sha: null, local: true };
  if (spec.local) return { dir: path.resolve(ROOT, spec.local), sha: null, local: true };
  const entry = lock[channel]?.[name];
  if (!entry?.sha) {
    throw new Error(`${channel}.${name} is not locked — run: node scripts/sync-sources.mjs --channel ${channel} --update`);
  }
  const dir = cacheDirFor(spec.repo, entry.sha, spec.path);
  if (!fs.existsSync(dir)) {
    throw new Error(`${channel}.${name}@${entry.sha.slice(0, 12)} is not fetched — run: node scripts/sync-sources.mjs --channel ${channel}`);
  }
  return { dir, sha: entry.sha, local: false };
}

// Minimal argv parser: --flag, --key value, repeatable --override.
export function parseArgs(argv) {
  const args = { override: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) continue;
    const key = a.slice(2);
    const next = argv[i + 1];
    const hasValue = next !== undefined && !next.startsWith("--");
    if (key === "override") {
      if (hasValue) args.override.push(argv[++i]);
    } else {
      args[key] = hasValue ? argv[++i] : true;
    }
  }
  return args;
}
