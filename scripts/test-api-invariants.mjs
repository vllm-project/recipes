// Invariants every published JSON file must hold, checked against the built
// API rather than against a rendering this file constructs. The point is to
// catch a build that writes self-contradictory output: a command serving a
// checkpoint the recipe excludes from that hardware, a preset whose declared
// `variant` disagrees with the checkpoint in its own command, a TP wider than
// the node can hold.
//
// `test:render` compares resolveSelection() against these files, which catches
// a *disagreement* between the two paths. This file checks each file on its
// own terms, so it fails even when both paths are wrong in the same way.
//
// Run after `node scripts/build-recipes-api.mjs` (reads public/).

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { getAllRecipes } from "../src/lib/recipes.js";
import { loadTaxonomy } from "../src/lib/taxonomy.js";
import { loadStrategies } from "../src/lib/strategies.js";
import { resolveSelection } from "../src/lib/recipe-render.js";
import { variantRunsOnHardware } from "../src/lib/command-synthesis.js";

const PUBLIC = process.env.RECIPES_PUBLIC_DIR ? path.resolve(process.env.RECIPES_PUBLIC_DIR) : path.join(process.cwd(), "public");
const taxonomy = loadTaxonomy();
const strategies = loadStrategies();
const profiles = taxonomy.hardware_profiles || {};

const recipes = getAllRecipes();

function jsonFiles(dir, out = []) {
  if (!fs.existsSync(dir)) return out;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) jsonFiles(full, out);
    else if (entry.name.endsWith(".json")) out.push(full);
  }
  return out;
}

const readJson = (file) => JSON.parse(fs.readFileSync(file, "utf8"));

// Every rendering a recipe publishes: the per-hardware files, their strategy
// alternatives, and each preset's points. Omni recipes carry a different shape
// (`{ command }`, no argv) and are covered by their own checks.
function* publishedRenderings(recipe) {
  for (const file of jsonFiles(path.join(PUBLIC, recipe.hf_id))) {
    const doc = readJson(file);
    const rel = path.relative(PUBLIC, file);
    if (doc?.points) {
      for (const [point, rendering] of Object.entries(doc.points)) {
        yield { rel: `${rel}#${point}`, rendering, preset: doc };
      }
    } else if (doc?.argv || doc?.head_argv || doc?.prefill?.argv) {
      yield { rel, rendering: doc, preset: null };
    }
  }
}

const flagValue = (argv, flag) => {
  const at = argv.indexOf(flag);
  return at >= 0 ? argv[at + 1] : undefined;
};

// The argv a rendering launches with: `argv` for a single-node deployment,
// `head_argv` for multi-node, the prefill pool's for a PD cluster (both pools
// serve the same checkpoint, so either answers the question).
function launchArgv(rendering) {
  return rendering.argv || rendering.head_argv || rendering.prefill?.argv || [];
}

// Every command a multi-node rendering launches: the head plus each follower
// rank. Rank-dependent flags (`--data-parallel-start-rank`, `--node-rank`)
// live only on the workers, so a check reading `head_argv` alone cannot see
// them.
function launchArgvs(rendering) {
  const workers = rendering.worker_argvs?.length
    ? rendering.worker_argvs
    : rendering.worker_argv ? [rendering.worker_argv] : [];
  return [
    { role: "head", argv: launchArgv(rendering) },
    ...workers.map((argv, i) => ({ role: `worker ${i + 1}`, argv })),
  ].filter(({ argv }) => argv?.length);
}

// The checkpoint a rendering serves. `vllm serve <id>` and Dynamo's
// `python -m dynamo.vllm --model <id>` are the two launch forms published here.
function servedCheckpoint(rendering) {
  const argv = launchArgv(rendering);
  const afterServe = argv[argv.indexOf("serve") + 1];
  if (argv.includes("serve") && afterServe) return afterServe;
  return flagValue(argv, "--model");
}

test("every published rendering runs the variant it declares, on the hardware it declares", () => {
  const wrong = [];
  let checked = 0;
  for (const recipe of recipes) {
    for (const { rel, rendering } of publishedRenderings(recipe)) {
      const { hardware, variant } = rendering;
      const profile = profiles[hardware];
      if (!profile) continue;
      const cfg = recipe.variants?.[variant];
      checked++;
      // A published file naming a variant the recipe doesn't define is worse
      // than one naming an incompatible variant: there is no checkpoint, no
      // vram budget and no precision to check it against. Don't fall back to
      // the `default` config — that would test a variant nobody published.
      if (!cfg) wrong.push(`${rel}: declares variant "${variant}", which ${recipe.hf_id} does not define`);
      else if (!variantRunsOnHardware(profile, cfg, hardware)) {
        wrong.push(`${rel}: variant "${variant}" does not run on ${hardware} by the recipe's own rules`);
      }
    }
  }
  assert.ok(checked > 500, `only ${checked} renderings checked`);
  assert.deepEqual(wrong, [], `${wrong.length}/${checked} renderings contradict their recipe:\n  ${wrong.slice(0, 15).join("\n  ")}`);
});

test("a published rendering serves the checkpoint its variant names", () => {
  const wrong = [];
  let checked = 0;
  for (const recipe of recipes) {
    for (const { rel, rendering } of publishedRenderings(recipe)) {
      const cfg = recipe.variants?.[rendering.variant];
      const expected = cfg?.model_id || recipe.model?.model_id;
      const served = servedCheckpoint(rendering);
      if (!expected || !served) continue;
      checked++;
      if (served !== expected) wrong.push(`${rel}: declares variant "${rendering.variant}" (${expected}) but serves ${served}`);
    }
  }
  assert.ok(checked > 500, `only ${checked} renderings checked`);
  assert.deepEqual(wrong, [], `${wrong.length}/${checked} serve a different checkpoint than they declare:\n  ${wrong.slice(0, 15).join("\n  ")}`);
});

// Multi-node parallelism comes in three dimensions, and a strategy picks
// which ones it uses: TP alone (`multi_node_tp`/`_tep`), TP+PP
// (`multi_node_tp_pp`, TP per node, PP across), TP+DP (`multi_node_tp_dp`),
// and DEP (`multi_node_dep`, DP across nodes, TP inside). Checking only
// `--tensor-parallel-size` leaves the other two unread — a DEP rendering could
// ask for 32 DP ranks on 4-GPU nodes and pass.
//
// Each dimension is bounded by the layer it is spread over, not by their
// product: in vLLM mp a TP group may cross nodes, PP stages map to nodes, and
// `--data-parallel-size` counts DP replicas while `--data-parallel-size-local`
// says how many land per node — so the published flags do not fix a single
// rank count to multiply. Comparing each against its own bound is what the
// flags do determine, and each catches its own class of mistake: PP wider than
// the cluster, DP ranks that no node runs, a TP group larger than the whole
// fabric.
const DIMENSION_BOUNDS = [
  // [flag, bound description, how many units the flag may span]
  ["--pipeline-parallel-size", "PP stages", (gpus, nodes) => nodes],
  ["--data-parallel-size", "DP ranks, at --data-parallel-size-local per node", (gpus, nodes, dpLocal) => nodes * (dpLocal || gpus)],
  ["--data-parallel-size-local", "DP ranks on one node", (gpus) => gpus],
  ["--tensor-parallel-size", "TP ranks across the cluster", (gpus, nodes) => gpus * nodes],
];

test("a multi-node rendering never asks for a parallel dimension its fabric cannot hold", () => {
  const wrong = [];
  const checked = new Set();
  for (const recipe of recipes) {
    for (const { rel, rendering } of publishedRenderings(recipe)) {
      if (rendering.deploy_type !== "multi_node") continue;
      const gpus = profiles[rendering.hardware]?.gpu_count;
      const nodes = rendering.node_count || 1;
      if (!gpus) continue;
      for (const { role, argv } of launchArgvs(rendering)) {
        const at = `${rel} (${role})`;
        const dpLocal = parseInt(flagValue(argv, "--data-parallel-size-local"), 10);
        for (const [flag, what, bound] of DIMENSION_BOUNDS) {
          const value = parseInt(flagValue(argv, flag), 10);
          // Absent or non-numeric: the strategy doesn't use this dimension here.
          if (!Number.isFinite(value)) continue;
          checked.add(flag);
          const limit = bound(gpus, nodes, Number.isFinite(dpLocal) ? dpLocal : null);
          if (value > limit) wrong.push(`${at}: ${flag} ${value} exceeds ${limit} (${what})`);
        }
        // --data-parallel-start-rank places a worker's first DP rank; one past
        // the last rank means it would run nothing.
        const startRank = parseInt(flagValue(argv, "--data-parallel-start-rank"), 10);
        const dp = parseInt(flagValue(argv, "--data-parallel-size"), 10);
        if (Number.isFinite(startRank) && Number.isFinite(dp)) {
          checked.add("--data-parallel-start-rank");
          if (startRank >= dp) wrong.push(`${at}: --data-parallel-start-rank ${startRank} is past the last of ${dp} DP ranks`);
        }
      }
    }
  }
  // Every dimension the catalog actually uses must have been exercised, or a
  // typo in a flag name above would silently pass.
  assert.ok(checked.has("--tensor-parallel-size") && checked.has("--data-parallel-size") && checked.has("--pipeline-parallel-size"),
    `dimensions not all exercised: saw ${[...checked].join(", ")}`);
  assert.deepEqual(wrong, [], `${wrong.length} over-subscribed dimensions:\n  ${wrong.slice(0, 15).join("\n  ")}`);
});

test("each preset file's declared variant matches the checkpoint in its own commands", () => {
  const wrong = [];
  let checked = 0;
  for (const recipe of recipes) {
    for (const { rel, rendering, preset } of publishedRenderings(recipe)) {
      if (!preset || !launchArgv(rendering).length) continue;
      const cfg = recipe.variants?.[rendering.variant];
      const expected = cfg?.model_id || recipe.model?.model_id;
      const served = servedCheckpoint(rendering);
      if (!expected || !served) continue;
      checked++;
      // The file's top-level `variant` is what a reader trusts; every point
      // below it must serve that same checkpoint.
      if (preset.variant !== rendering.variant) {
        wrong.push(`${rel}: file declares variant "${preset.variant}", point renders "${rendering.variant}"`);
      } else if (served !== expected) {
        wrong.push(`${rel}: declares variant "${rendering.variant}" (${expected}) but serves ${served}`);
      }
    }
  }
  assert.ok(checked > 100, `only ${checked} preset points checked`);
  assert.deepEqual(wrong, [], `${wrong.length}/${checked} preset points disagree with their file:\n  ${wrong.slice(0, 15).join("\n  ")}`);
});

test("a per-strategy file renders the deployment it names from the page query", () => {
  const wrong = [];
  let checked = 0;
  for (const recipe of recipes) {
    const base = path.join(PUBLIC, recipe.hf_id);
    if (!fs.existsSync(base)) continue;
    for (const file of jsonFiles(base)) {
      const match = path.relative(base, file).match(/^hw[/\\]([^/\\]+)[/\\]strategies[/\\](.+)\.json$/);
      if (!match) continue;
      const [, hardware, strategy] = match;
      const expected = readJson(file);
      if (!expected?.head_argv) continue;
      checked++;
      const query = new URLSearchParams({ hardware, strategy });
      if (expected.variant && expected.variant !== "default") query.set("variant", expected.variant);
      if (expected.node_count > 1) query.set("nodes", String(expected.node_count));
      const got = resolveSelection(recipe, query, strategies, taxonomy)?.rendering;
      if (!got) wrong.push(`${path.relative(PUBLIC, file)}: the page query renders nothing`);
      else if (got.deploy_type !== expected.deploy_type) {
        wrong.push(`${path.relative(PUBLIC, file)}: file says ${expected.deploy_type}, the query renders ${got.deploy_type}`);
      }
    }
  }
  assert.ok(checked > 500, `only ${checked} per-strategy files checked`);
  assert.deepEqual(wrong, [], `${wrong.length}/${checked} per-strategy files are unreachable or mis-typed:\n  ${wrong.slice(0, 15).join("\n  ")}`);
});
