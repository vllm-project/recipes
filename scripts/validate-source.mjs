/**
 * Validate one source repo's recipe directory against this renderer — what a
 * source repo's CI runs on pull requests (.github/workflows/validate-recipes.yml)
 * and what an author can run locally before opening one.
 *
 *   node scripts/validate-source.mjs --source vllm --dir ~/vllm/recipes
 *
 * Builds an isolated `validate` channel in a temp dir and runs the checks a
 * lock bump runs on it: schema + strict assemble (ownership, overlay scope,
 * orphaned overlays) + the full JSON API build. Exits non-zero on failure.
 *
 * - If --source is in the default channel, the channel is copied with that
 *   source read from --dir (same role, scope and pins for the others), so
 *   cross-source rules are checked too.
 * - Otherwise (a repo not wired into sources.yaml yet) --dir is validated on
 *   its own, as a core source.
 *
 * Nothing outside the temp dir changes: .content/latest, public/ and your
 * sources.local.yaml are left alone.
 */

import fs from "fs";
import os from "os";
import path from "path";
import { spawnSync } from "child_process";
import yaml from "js-yaml";
import { ROOT, loadLock, loadSources, parseArgs } from "./lib/content-sources.mjs";

const args = parseArgs(process.argv.slice(2));
if (!args.source || !args.dir) {
  console.error("usage: validate-source.mjs --source <name> --dir <recipe directory>");
  process.exit(2);
}
const source = String(args.source);
const dir = path.resolve(String(args.dir).replace(/^~(?=\/)/, os.homedir()));
if (!fs.existsSync(path.join(dir, "models")) && !fs.existsSync(path.join(dir, "overlays"))) {
  console.error(`${dir} has neither models/ nor overlays/`);
  process.exit(2);
}

const { defaultChannel, channels } = loadSources();
const current = channels[defaultChannel];
const CHANNEL = "validate";
let sources;
let lock = {};
if (current[source]) {
  // Keep repo/ref/path for provenance and role/scope for the rules; read the
  // files from --dir. Paths of other local sources resolve from this repo.
  sources = Object.fromEntries(Object.entries(current).map(([name, spec]) => [
    name,
    name === source ? { ...spec, local: dir }
      : spec.local ? { ...spec, local: path.resolve(ROOT, spec.local) } : spec,
  ]));
  lock = { [CHANNEL]: loadLock()[defaultChannel] || {} };
  console.log(`Validating "${source}" from ${dir} within channel ${defaultChannel}`);
} else {
  sources = { [source]: { local: dir, role: "core" } };
  console.log(`"${source}" is not in sources.yaml yet — validating ${dir} on its own`);
}

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "recipes-validate-"));
fs.writeFileSync(path.join(tmp, "sources.yaml"),
  yaml.dump({ default_channel: CHANNEL, channels: { [CHANNEL]: sources } }));
fs.writeFileSync(path.join(tmp, "sources.lock.json"), JSON.stringify(lock, null, 2));

// The caller's RECIPES_OVERRIDES would replace --dir (overrides win over
// `local`), so the children never see it.
const { RECIPES_OVERRIDES: _ignored, ...inherited } = process.env;
const env = {
  ...inherited,
  RECIPES_SOURCES: path.join(tmp, "sources.yaml"),
  RECIPES_CHANNEL: CHANNEL,
  RECIPES_PUBLIC_DIR: path.join(tmp, "public"),
};
const steps = [
  ["Fetch the other pinned sources", ["scripts/sync-sources.mjs"]],
  ["Assemble strictly", ["scripts/assemble-content.mjs", "--strict"]],
  // Reuse the strict tree instead of assembling again.
  ["Check the schema and build the JSON API", ["scripts/build-recipes-api.mjs"],
    { RECIPES_CONTENT_DIR: path.join(ROOT, ".content", CHANNEL) }],
];

function report(msg) {
  console.log(`\n${msg}`);
  if (process.env.GITHUB_STEP_SUMMARY) fs.appendFileSync(process.env.GITHUB_STEP_SUMMARY, `${msg}\n`);
}

let failed = null;
for (const [label, argv, extraEnv = {}] of steps) {
  console.log(`\n▶ ${label}`);
  const res = spawnSync(process.execPath, argv, { stdio: "inherit", env: { ...env, ...extraEnv }, cwd: ROOT });
  if (res.status !== 0) {
    failed = { label, status: res.status || 1 };
    break;
  }
}
fs.rmSync(tmp, { recursive: true, force: true });
fs.rmSync(path.join(ROOT, ".content", CHANNEL), { recursive: true, force: true });

if (failed) {
  report(`✗ ${source} recipes failed: ${failed.label} (see the log above)`);
  process.exit(failed.status);
}
report(`✓ ${source} recipes from ${dir} are valid against the renderer`);
