import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import yaml from "js-yaml";
import { attachPoolProfiles, loadPoolProfiles } from "../src/lib/content-root.js";
import { cacheDirFor } from "./lib/content-sources.mjs";
import { validatePoolDeployment, validatePoolProfile } from "../src/lib/pool-profile-validation.js";

const root = fileURLToPath(new URL("../", import.meta.url));
const repo = "vllm-project/vllm-ascend";
const profileId = "vllm-ascend/mooncake";
const profile = {
  schema_version: 1, label: "Mooncake SSD", backend: "mooncake",
  inputs: { master: { type: "host", scope: "cluster", label: "Master address" } },
  store: { prefill: { backend: "mooncake" }, decode: { backend: "mooncake" } },
  steps: [{ id: "check", title: "Check master", stage: "control", target: "containers", check: "The master answers." }],
};
const fixture = (t) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "recipes-pool-source-"));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  fs.copyFileSync(path.join(root, "taxonomy.yaml"), path.join(dir, "taxonomy.yaml"));
  const put = (file, data) => {
    fs.mkdirSync(path.dirname(path.join(dir, file)), { recursive: true });
    fs.writeFileSync(path.join(dir, file), yaml.dump(data));
  };
  const src = { repo, ref: "main", path: "recipes", role: "overlay", scope: { brands: ["Huawei"] } };
  const sources = (artifacts = []) => put("sources.yaml", { default_channel: "test", channels: { test: {
    base: { local: path.join(dir, "base"), role: "owner" },
    "vllm-ascend": { ...src, ...(artifacts.length ? { extra_artifacts: artifacts } : {}) },
  } } });
  put("base/models/org/model.yaml", { meta: { hardware: {} }, variants: {} });
  const pin = (sha) => {
    fs.writeFileSync(path.join(dir, "sources.lock.json"), JSON.stringify({ test: { "vllm-ascend": { ...src, sha } } }));
    const cached = path.join(".sources", path.basename(cacheDirFor(repo, sha, "recipes")));
    fs.mkdirSync(path.join(dir, cached), { recursive: true });
    return cached;
  };
  const writeOverlay = (cached, kvOffload = profileId, hardware = "ascend_910c") => put(
    path.join(cached, "overlays/org/model.yaml"),
    { presets: { pooled: { label: "Pooled", hardware, strategy: "pd_cluster", select: { kv_offload: kvOffload } } } },
  );
  const assemble = () => spawnSync(process.execPath, [path.join(root, "scripts/assemble-content.mjs"), "--strict"], {
    cwd: dir, encoding: "utf8",
    env: { ...process.env, CI: "1", RECIPES_SOURCES: path.join(dir, "sources.yaml"), RECIPES_CHANNEL: "test", RECIPES_OVERRIDES: "" },
  });
  return { dir, put, pin, sources, writeOverlay, assemble, content: path.join(dir, ".content/test") };
};

test("old locks assemble without profiles; a profile-only revision updates rendering input and provenance", (t) => {
  const f = fixture(t);
  f.sources();
  f.pin("old");
  let run = f.assemble();
  assert.equal(run.status, 0, run.stderr);
  assert.deepEqual(loadPoolProfiles(f.content), {});

  f.sources(["pool_profiles"]);
  for (const [sha, label] of [["first", "First guide"], ["second", "Revised guide"]]) {
    const cached = f.pin(sha);
    f.writeOverlay(cached);
    f.put(path.join(cached, "pool_profiles/mooncake.yaml"), { ...profile, label });
    run = f.assemble();
    assert.equal(run.status, 0, run.stderr);
    const model = yaml.load(fs.readFileSync(path.join(f.content, "models/org/model.yaml"), "utf8"));
    const catalog = loadPoolProfiles(f.content);
    const provenance = JSON.parse(fs.readFileSync(path.join(f.content, "provenance.json"), "utf8"));
    assert.equal(catalog[profileId].label, label);
    assert.equal(catalog[profileId].source.sha, sha);
    assert.equal(catalog[profileId].source.path, "recipes/pool_profiles/mooncake.yaml");
    assert.deepEqual(provenance.pool_profiles[profileId], catalog[profileId].source);
    assert.deepEqual(provenance.recipes["org/model"].pool_profiles, provenance.pool_profiles);
    assert.deepEqual(attachPoolProfiles(model, { ...catalog, "unrelated/profile": {} })._poolProfiles, catalog);
    assert.equal(attachPoolProfiles({}, catalog)._poolProfiles, undefined);
  }
});

test("source checks reject missing, malformed, duplicate, cross-source or out-of-scope pool profiles", (t) => {
  const f = fixture(t);
  const cached = f.pin("local-sha");
  const file = path.join(cached, "pool_profiles/mooncake.yaml");
  f.sources(["pool_profiles"]);
  f.writeOverlay(cached);
  const fails = (pattern) => {
    const run = f.assemble();
    assert.notEqual(run.status, 0);
    assert.match(run.stderr, pattern);
  };
  fails(/presets\.pooled\.select\.kv_offload: missing pool profile/);
  f.put(file, { ...profile, schema_version: 99 });
  fails(/pool_profiles\/mooncake\.yaml.*version/i);
  f.put(file, profile);
  f.put(path.join(cached, "pool_profiles/unused.yaml"), { ...profile, source: { sha: "forged" } });
  fails(/unused\.yaml.*additional properties.*source/);
  fs.unlinkSync(path.join(f.dir, cached, "pool_profiles/unused.yaml"));
  f.writeOverlay(cached, "other/mooncake");
  fails(/belongs to another source/);
  f.writeOverlay(cached, profileId, "h200");
  fails(/outside the source's granted scope/);
  f.writeOverlay(cached);
  f.put(path.join(cached, "pool_profiles/mooncake.yml"), profile);
  fails(/duplicate pool profile/);
  f.sources();
  fails(/missing pool profile/);
});

test("reuse and new each require runnable roles; actions and references must work on every selected placement", () => {
  const service = (role) => ({ role, container: "model", launch: {
    argv: [role], ...(role !== "proxy" && { transfer: { kv_connector: "MooncakeConnectorV1" } }),
  } });
  const d = { hosts: ["h"], containers: { model: { host: "h" } },
    placements: ["prefill", "decode", "proxy"].map(service),
    steps: [{ id: "launch", title: "Launch", target: "decode", stage: "serve", action: "launch", check: "Ready" }],
  };
  assert.deepEqual(validatePoolDeployment(profile, d), []);
  const check = (change, expected) => {
    const layout = structuredClone(d);
    change(layout);
    assert.match(validatePoolDeployment(profile, layout).join("\n"), expected);
  };
  check((layout) => { layout.placements[0].mode = "new"; }, /Missing prefill placement in reuse mode/);
  check((layout) => { layout.steps[0].target = "hosts"; }, /launch actions must target placements/);
  check((layout) => {
    layout.placements.push({ role: "meta", container: "model", mode: "new" });
    layout.steps[0].target = "control";
    layout.steps[0].mode = "new";
  }, /missing launch.argv on meta/);
  check((layout) => { layout.placements[1].launch.store = { backend: "memcache" }; }, /backend must match/);
  check((layout) => { layout.placements[0].launch.argv.push("{{placement.prt}}"); }, /unknown placement variable placement.prt/);
  check((layout) => { layout.steps[0].check = "{{placement.port}}"; }, /unknown placement variable placement.port/);
  check((layout) => {
    layout.inputs = { new_path: { type: "path", scope: "host", mode: "new", label: "Path" } };
    layout.steps[0].check = "{{new_path}}";
  }, /only available in new mode/);
  d.placements[1].vars = { port: 37000 };
  d.steps[0].check = "Verify {{placement.port}} for {{placement.id}}";
  assert.deepEqual(validatePoolDeployment(profile, d), []);
});

test("capacity references must be numeric; per-host counts must come from host inputs", () => {
  const inputs = { segment: { scope: "host", type: "integer", label: "Segment GiB" },
    cluster_count: { scope: "cluster", type: "integer", label: "Cluster count" } };
  const budget = { label: "RAM", unit: "GiB", input: "segment", count: "workers" };
  const check = (extra) => validatePoolProfile({ ...profile, inputs, capacity: [{ ...budget, ...extra }] });
  assert.deepEqual(check({}), []);
  assert.deepEqual(check({ count: "segment" }), []);
  assert.match(check({ input: "missing" }).join("\n"), /capacity.input must name an integer input/);
  assert.match(check({ count: "cluster_count" }).join("\n"), /capacity.count must be workers or a per-host integer/);
});
