/**
 * Assemble a channel's recipe sources into .content/<channel>/ — the tree the
 * site loaders and build-recipes-api read (same layout the repo root had):
 *
 *   models/<org>/<repo>.yaml   recipes from core/owner sources, overlays merged
 *   taxonomy.yaml, platforms.yaml, strategies/, kv_store/, quickstart/
 *                              renderer-owned vocabulary (this repo)
 *   provenance.json            hf_id → source repo/sha/path (+ overlays)
 *
 * Unmerged files are symlinked (copied on CI), so edits to a local source show
 * up without re-assembling; only adding/removing files needs a re-run.
 *
 *   node scripts/assemble-content.mjs [--channel <name> | --all] [--strict] [--override name=dir] [--watch]
 *
 * --watch re-assembles whenever a local source's models/ or overlays/ changes
 * (for previewing a checkout alongside `pnpm dev`).
 *
 * --strict also fails on overlays whose target recipe doesn't exist (CI for
 * overlay repos); scope violations and ownership conflicts always fail.
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import yaml from "js-yaml";
import { applyOverlay, hasGuideMarkers, resolveScope, stripGuideMarkers, variantInScope } from "../src/lib/overlay-merge.js";
import { ROOT, loadLock, loadOverrides, loadSources, parseArgs, resolveSourceDir } from "./lib/content-sources.mjs";

// Highest recipe schema this renderer understands (source manifest.yaml).
export const SUPPORTED_SCHEMA_VERSION = 1;

const VOCABULARY = ["taxonomy.yaml", "platforms.yaml", "strategies", "kv_store", "quickstart"];

function findYamlFiles(dir) {
  if (!fs.existsSync(dir)) return [];
  const out = [];
  for (const e of fs.readdirSync(dir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...findYamlFiles(full));
    else if (/\.ya?ml$/.test(e.name)) out.push(full);
  }
  return out;
}

function hfIdOf(baseDir, file) {
  const parts = path.relative(baseDir, file).split(path.sep);
  if (parts.length < 2) return null;
  return `${parts[0]}/${parts[parts.length - 1].replace(/\.ya?ml$/, "")}`;
}

function editUrl(spec, relPath) {
  if (!spec.repo) return null;
  const p = path.posix.join(spec.path || ".", relPath.split(path.sep).join("/"));
  return `https://github.com/${spec.repo}/edit/${spec.ref || "main"}/${p}`;
}

// Parse a source file, reporting a syntax error with its source and path
// (and continuing, so one run lists every broken file).
function loadYaml(file, label, rel, errors) {
  try {
    return yaml.load(fs.readFileSync(file, "utf8"), { filename: rel });
  } catch (e) {
    errors.push(`${label} ${rel}: invalid YAML — ${e.message.split("\n")[0]}`);
    return undefined;
  }
}

function readManifest(dir, label) {
  const file = path.join(dir, "manifest.yaml");
  if (!fs.existsSync(file)) return {};
  const m = yaml.load(fs.readFileSync(file, "utf8")) || {};
  if ((m.schema_version ?? 1) > SUPPORTED_SCHEMA_VERSION) {
    throw new Error(`${label}: schema_version ${m.schema_version} is newer than this renderer supports (${SUPPORTED_SCHEMA_VERSION})`);
  }
  return m;
}

// Symlink locally (live edits); copy on CI/Vercel so the tree is
// self-contained for Next's output file tracing.
const COPY = !!(process.env.CI || process.env.VERCEL);

function link(target, dest) {
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  if (COPY) fs.cpSync(target, dest, { recursive: true, dereference: true });
  else fs.symlinkSync(path.relative(path.dirname(dest), target), dest);
}

export function assembleChannel(channel, { strict = false, overrides = {}, quiet = false } = {}) {
  const { channels } = loadSources();
  const sources = channels[channel];
  if (!sources) throw new Error(`sources.yaml has no channel "${channel}"`);
  const lock = loadLock();
  const taxonomy = yaml.load(fs.readFileSync(path.join(ROOT, "taxonomy.yaml"), "utf8"));
  const out = path.join(ROOT, ".content", channel);
  // Build beside the live tree and swap it in, so a running dev server never
  // reads a half-written tree. Same depth as `out`, so relative symlinks hold.
  const tmp = path.join(ROOT, ".content", `.tmp-${channel}`);
  const errors = [];
  const warnings = [];

  // ── Collect recipes and overlays from every source ──
  const owners = new Map(); // hf_id → { name, spec, file, rel, sha }
  const overlays = new Map(); // hf_id → [{ name, spec, file, rel, sha, ctx }]
  const sourceInfo = {};
  const localDirs = [];
  for (const [name, spec] of Object.entries(sources)) {
    const { dir, sha, local } = resolveSourceDir(channel, name, spec, lock, overrides);
    const label = `${channel}.${name}`;
    if (local) localDirs.push(dir);
    readManifest(dir, label);
    sourceInfo[name] = { repo: spec.repo || null, ref: spec.ref || null, sha, path: spec.path || ".", role: spec.role, local };
    const ctx = resolveScope(spec.scope, taxonomy);

    const modelsDir = path.join(dir, "models");
    for (const file of findYamlFiles(modelsDir)) {
      const hfId = hfIdOf(modelsDir, file);
      if (!hfId) continue;
      const rel = path.relative(dir, file);
      const prev = owners.get(hfId);
      if (prev) {
        errors.push(`${hfId}: shipped by both "${prev.name}" (${prev.rel}) and "${name}" (${rel}) — one source must own it`);
        continue;
      }
      const parsed = loadYaml(file, label, rel, errors);
      if (parsed === undefined) continue;
      const doc = parsed && typeof parsed === "object" ? parsed : {};
      const meta = doc.meta || {};
      const tasks = Array.isArray(meta.tasks) ? meta.tasks : [];
      if (spec.scope?.tasks && !spec.scope.tasks.some((t) => tasks.includes(t))) {
        errors.push(`${label} ${rel}: source scope requires meta.tasks to include ${spec.scope.tasks.join("|")}`);
      }
      if (spec.scope?.exclude_tasks?.some((t) => tasks.includes(t))) {
        errors.push(`${label} ${rel}: source scope excludes meta.tasks ${spec.scope.exclude_tasks.join("|")}`);
      }
      if (spec.role === "overlay") {
        // An overlay repo may own whole recipes only for models that run
        // exclusively on its hardware: every declared GPU and every variant's
        // supported_hardware allowlist must be in scope (a variant without
        // one would render on every GPU).
        const hw = Object.keys(meta.hardware || {});
        const variants = Object.entries(doc.variants || {});
        if (hw.length === 0 || !hw.every((k) => ctx.isHwKey(k))
          || variants.length === 0 || !variants.every(([, v]) => variantInScope(v, ctx))) {
          errors.push(`${label} ${rel}: an overlay source may only own recipes whose meta.hardware and every variant's supported_hardware are in scope`);
        }
      }
      owners.set(hfId, { name, spec, file, rel, sha });
    }

    const overlaysDir = path.join(dir, "overlays");
    const overlayFiles = findYamlFiles(overlaysDir);
    if (overlayFiles.length && spec.role !== "overlay") {
      errors.push(`${label}: has overlays/ but role is "${spec.role}" (needs role: overlay)`);
      continue;
    }
    for (const file of overlayFiles) {
      const hfId = hfIdOf(overlaysDir, file);
      if (!hfId) continue;
      if (!overlays.has(hfId)) overlays.set(hfId, []);
      overlays.get(hfId).push({ name, spec, file, rel: path.relative(dir, file), sha, ctx });
    }
  }

  // ── Write the tree ──
  fs.rmSync(tmp, { recursive: true, force: true });
  // models/ always exists, even for a channel with no recipes yet.
  fs.mkdirSync(path.join(tmp, "models"), { recursive: true });
  for (const entry of VOCABULARY) {
    const src = path.join(ROOT, entry);
    if (fs.existsSync(src)) link(src, path.join(tmp, entry));
  }

  const provenance = {};
  let merged = 0;
  for (const [hfId, owner] of [...owners].sort(([a], [b]) => a.localeCompare(b))) {
    const dest = path.join(tmp, "models", `${hfId}.yaml`);
    const ovs = overlays.get(hfId) || [];
    const raw = fs.readFileSync(owner.file, "utf8");
    const ownerLabel = `${channel}.${owner.name}`;
    provenance[hfId] = {
      source: owner.name,
      repo: owner.spec.repo || null,
      ref: owner.spec.ref || null,
      sha: owner.sha,
      path: path.posix.join(owner.spec.path || ".", owner.rel.split(path.sep).join("/")),
      edit_url: editUrl(owner.spec, owner.rel),
      overlays: ovs.map((o) => ({
        source: o.name,
        repo: o.spec.repo || null,
        ref: o.spec.ref || null,
        sha: o.sha,
        path: path.posix.join(o.spec.path || ".", o.rel.split(path.sep).join("/")),
        edit_url: editUrl(o.spec, o.rel),
      })),
    };
    if (ovs.length === 0 && !hasGuideMarkers(raw)) {
      link(owner.file, dest);
      continue;
    }
    let recipe = loadYaml(owner.file, ownerLabel, owner.rel, errors);
    if (recipe === undefined) continue;
    if (!recipe || typeof recipe !== "object" || Array.isArray(recipe)) {
      errors.push(`${owner.name} ${owner.rel}: recipe is empty or not a map`);
      continue;
    }
    for (const o of ovs) {
      const ov = loadYaml(o.file, `${channel}.${o.name}`, o.rel, errors);
      if (ov === undefined) continue;
      const res = applyOverlay(recipe, ov || {}, { ctx: o.ctx, source: o.name });
      for (const e of res.errors) errors.push(`${channel}.${o.name} ${o.rel}: ${e}`);
      recipe = res.recipe;
    }
    recipe.guide = stripGuideMarkers(recipe.guide);
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    fs.writeFileSync(
      dest,
      `# Assembled from ${provenance[hfId].path}${ovs.map((o) => ` + ${o.name}:${o.rel}`).join("")} — do not edit\n`
        + yaml.dump(recipe, { lineWidth: -1, noRefs: true }),
    );
    merged++;
  }

  for (const [hfId, ovs] of overlays) {
    if (owners.has(hfId)) continue;
    for (const o of ovs) {
      (strict ? errors : warnings).push(`${channel}.${o.name} ${o.rel}: no recipe ${hfId} in this channel — overlay skipped`);
    }
  }

  fs.writeFileSync(
    path.join(tmp, "provenance.json"),
    JSON.stringify({ channel, sources: sourceInfo, recipes: provenance }, null, 2) + "\n",
  );

  for (const w of warnings) console.warn(`  ⚠ ${w}`);
  if (errors.length) {
    fs.rmSync(tmp, { recursive: true, force: true });
    throw new Error(`assemble ${channel} failed:\n  - ${errors.join("\n  - ")}`);
  }
  // Two renames back to back keep the window without a tree to microseconds.
  const old = path.join(ROOT, ".content", `.old-${channel}`);
  fs.rmSync(old, { recursive: true, force: true });
  if (fs.existsSync(out)) fs.renameSync(out, old);
  fs.renameSync(tmp, out);
  fs.rmSync(old, { recursive: true, force: true });
  if (!quiet) {
    console.log(`✓ Content ${channel}: ${owners.size} recipes from ${Object.keys(sources).length} source(s), ${merged} with overlays → .content/${channel}`);
  }
  return { channel, dir: out, recipes: owners.size, merged, localDirs };
}

// CLI
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = parseArgs(process.argv.slice(2));
  const { defaultChannel, channels } = loadSources();
  const overrides = loadOverrides(args.override);
  const targets = args.all ? Object.keys(channels) : [args.channel || process.env.RECIPES_CHANNEL || defaultChannel];
  const run = () => targets.flatMap((channel) => assembleChannel(channel, { strict: !!args.strict, overrides }).localDirs);
  let dirs;
  try {
    dirs = run();
  } catch (e) {
    console.error(e.message);
    if (!args.watch) process.exit(1);
    dirs = Object.values(overrides);
  }
  if (args.watch) {
    let timer = null;
    const rerun = () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        try { run(); } catch (e) { console.error(e.message); }
      }, 200);
    };
    const watched = [...new Set(dirs)].flatMap((d) => ["models", "overlays"].map((s) => path.join(d, s)))
      .filter((d) => fs.existsSync(d));
    for (const d of watched) fs.watch(d, { recursive: true }, rerun);
    console.log(`  watching ${watched.map((d) => path.relative(ROOT, d) || ".").join(", ")}`);
  }
}
