import test from "node:test";
import assert from "node:assert/strict";
import { buildDockerArgv, buildDockerRun, computeDockerMeta, isPipInstallAvailable } from "../src/lib/command-synthesis.js";
import { loadTaxonomy } from "../src/lib/taxonomy.js";

test("Zhenwu profiles are restricted and do not offer a generic pip wheel", () => {
  const profiles = loadTaxonomy().hardware_profiles;
  for (const id of ["810e", "m890"]) {
    assert.equal(profiles[id].brand, "Zhenwu");
    assert.equal(profiles[id].generation, "ppu");
    assert.equal(profiles[id].restricted, true);
    assert.equal(isPipInstallAvailable({ model: {} }, {}, profiles[id], id), false);
  }
});

test("PPU does not fall back to a NVIDIA container", () => {
  const profile = { brand: "Zhenwu", generation: "ppu", gpu_count: 16, vram_gb: 1536 };
  const meta = computeDockerMeta({ model: { docker_image: "example/nvidia:tag" } }, {}, profile, "810e");
  assert.equal(meta.isPpu, true);
  assert.equal(meta.brandKey, "ppu");
  assert.equal(meta.dockerEnabled, false);
  assert.equal(meta.image, null);
});

test("a PPU hardware profile can provide the container image", () => {
  const profile = { brand: "Zhenwu", generation: "ppu", docker_image: "example/ppu:tag" };
  const meta = computeDockerMeta({ model: {} }, {}, profile, "810e");
  assert.equal(meta.dockerEnabled, true);
  assert.equal(meta.image, "example/ppu:tag");
});

test("an explicitly pinned PPU image uses privileged mode and host networking", () => {
  const profile = { brand: "Zhenwu", generation: "ppu", gpu_count: 16, vram_gb: 1536 };
  const variant = { hardware_overrides: { "810e": { docker_image: "example/ppu:tag", docker_serve_command: "vllm serve" } } };
  const meta = computeDockerMeta({ model: {} }, variant, profile, "810e");
  const argv = buildDockerArgv({ argv: ["vllm", "serve", "org/model"], env: {}, meta });
  assert.equal(meta.dockerEnabled, true);
  assert.equal(meta.image, "example/ppu:tag");
  assert.ok(argv.includes("--privileged"));
  assert.ok(argv.includes("--init"));
  assert.deepEqual(argv.slice(argv.indexOf("--ulimit"), argv.indexOf("--ulimit") + 2), ["--ulimit", "memlock=-1:-1"]);
  assert.deepEqual(argv.slice(argv.indexOf("--security-opt"), argv.indexOf("--security-opt") + 2), ["--security-opt", "seccomp=unconfined"]);
  assert.equal(argv.some((arg) => arg.startsWith("--device")), false);
  assert.deepEqual(argv.slice(argv.indexOf("--network"), argv.indexOf("--network") + 2), ["--network", "host"]);
  assert.deepEqual(argv.slice(-4), ["example/ppu:tag", "vllm", "serve", "org/model"]);
  assert.equal(argv.includes("--entrypoint"), false);
  assert.equal(argv.includes("--gpus"), false);
  const shell = buildDockerRun({ command: "vllm serve org/model", env: {}, ...meta });
  assert.match(shell, /^docker run --network host --ipc host --privileged --init --ulimit memlock=-1:-1 --security-opt seccomp=unconfined/);
  assert.match(shell, /example\/ppu:tag \\\n  vllm serve org\/model/);
  assert.doesNotMatch(shell, /--device|--entrypoint/);
});
