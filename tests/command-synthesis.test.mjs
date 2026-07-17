import assert from "node:assert/strict";
import test from "node:test";

import {
  buildAscendDockerArgv,
  buildAscendDockerRun,
  buildDockerArgv,
  buildDockerRun,
  computeDockerMeta,
  resolveSingleNodeTp,
} from "../src/lib/command-synthesis.js";

const recipe = { model: {} };
const variant = {};

test("Atlas A2 uses the Ascend image and exposes eight NPU devices", () => {
  const meta = computeDockerMeta(
    recipe,
    variant,
    { brand: "Huawei", generation: "ascend", gpu_count: 8 },
    "atlas_800i_a2",
  );

  assert.equal(meta.brandKey, "ascend");
  assert.equal(meta.isAscend, true);
  assert.equal(meta.image, "quay.io/ascend/vllm-ascend:main");
  assert.match(meta.gpuFlags, /--device=\/dev\/davinci0(?:\s|\\)/);
  assert.match(meta.gpuFlags, /--device=\/dev\/davinci7(?:\s|\\)/);
  assert.doesNotMatch(meta.gpuFlags, /--device=\/dev\/davinci8(?:\s|\\)/);
  assert.match(meta.gpuFlags, /--device=\/dev\/davinci_manager/);
  assert.match(meta.gpuFlags, /\/usr\/local\/dcmi:\/usr\/local\/dcmi/);
});

test("Atlas A3 uses the A3 image and exposes sixteen NPU devices", () => {
  const meta = computeDockerMeta(
    recipe,
    variant,
    { brand: "Huawei", generation: "ascend", gpu_count: 16 },
    "atlas_800i_a3",
  );

  assert.equal(meta.image, "quay.io/ascend/vllm-ascend:main-a3");
  assert.match(meta.gpuFlags, /--device=\/dev\/davinci15(?:\s|\\)/);
  assert.doesNotMatch(meta.gpuFlags, /--device=\/dev\/davinci16(?:\s|\\)/);

  const nightly = computeDockerMeta(
    { model: { nightly_required: true } },
    variant,
    { brand: "Huawei", generation: "ascend", gpu_count: 16 },
    "atlas_800i_a3",
  );
  assert.equal(nightly.image, "quay.io/ascend/vllm-ascend:nightly-main-a3");
});

test("Atlas A3 enforces two NPUs without changing other hardware", () => {
  const smallVariant = { vram_minimum_gb: 20 };
  const a3 = { brand: "Huawei", generation: "ascend", gpu_count: 16, vram_gb: 1024, minimum_tp: 2 };
  const a2 = { brand: "Huawei", generation: "ascend", gpu_count: 8, vram_gb: 512 };
  const nvidia = { brand: "NVIDIA", generation: "hopper", gpu_count: 8, vram_gb: 640 };
  const amd = { brand: "AMD", generation: "amd", gpu_count: 8, vram_gb: 1536 };
  const tpu = { brand: "Google", generation: "tpu", gpu_count: 8, vram_gb: 256 };

  assert.equal(resolveSingleNodeTp(recipe, smallVariant, a3), 2);
  assert.equal(resolveSingleNodeTp(recipe, smallVariant, a2), 1);
  assert.equal(resolveSingleNodeTp(recipe, smallVariant, nvidia), 1);
  assert.equal(resolveSingleNodeTp(recipe, smallVariant, amd), 1);
  assert.equal(resolveSingleNodeTp(recipe, smallVariant, tpu), 1);
});

test("Ascend accepts brand and exact hardware image pins", () => {
  const brandPinned = computeDockerMeta(
    { model: { docker_image: { ascend: "quay.io/ascend/vllm-ascend:v0.18.0" } } },
    variant,
    { brand: "Huawei", generation: "ascend", gpu_count: 8 },
    "atlas_800i_a2",
  );
  assert.equal(brandPinned.image, "quay.io/ascend/vllm-ascend:v0.18.0");

  const hardwarePinned = computeDockerMeta(
    recipe,
    {
      hardware_overrides: {
        atlas_800i_a3: {
          docker_image: "quay.io/ascend/vllm-ascend:v0.18.0-a3",
        },
      },
    },
    { brand: "Huawei", generation: "ascend", gpu_count: 16 },
    "atlas_800i_a3",
  );
  assert.equal(hardwarePinned.image, "quay.io/ascend/vllm-ascend:v0.18.0-a3");
});

test("Ascend docker commands set the vllm entrypoint and preserve serve", () => {
  const meta = computeDockerMeta(
    recipe,
    variant,
    { brand: "Huawei", generation: "ascend", gpu_count: 8 },
    "atlas_800i_a2",
  );
  const argv = buildAscendDockerArgv({
    argv: ["vllm", "serve", "org/model", "--max-model-len", "4096"],
    env: {},
    meta,
  });

  const shell = buildAscendDockerRun({
    command: "vllm serve org/model \\\n  --max-model-len 4096",
    env: {},
    image: meta.image,
    gpuFlags: meta.gpuFlags,
  });

  assert.ok(argv.includes("--device=/dev/davinci0"));
  assert.ok(argv.includes("--device=/dev/davinci7"));
  assert.ok(!argv.includes("--device=/dev/davinci8"));
  assert.ok(argv.includes("--device=/dev/davinci_manager"));
  assert.ok(argv.includes("/usr/local/dcmi:/usr/local/dcmi"));
  const imageIndex = argv.indexOf(meta.image);
  assert.deepEqual(argv.slice(imageIndex - 2, imageIndex + 3), [
    "--entrypoint", "vllm", meta.image, "serve", "org/model",
  ]);
  assert.match(shell, /--entrypoint vllm/);
  assert.match(shell, /quay\.io\/ascend\/vllm-ascend:main serve org\/model/);
});

test("existing Docker brands keep their current defaults", () => {
  const nvidia = computeDockerMeta(recipe, variant, { brand: "NVIDIA" });
  assert.equal(nvidia.brandKey, "nvidia");
  assert.equal(nvidia.image, "vllm/vllm-openai:latest");
  assert.equal(nvidia.gpuFlags, "--gpus all");
  const nvidiaShell = buildDockerRun({
    command: "vllm serve org/model",
    env: {},
    image: nvidia.image,
    gpuFlags: nvidia.gpuFlags,
  });
  assert.doesNotMatch(nvidiaShell, /--entrypoint/);
  assert.match(nvidiaShell, /vllm\/vllm-openai:latest org\/model/);
  const nvidiaArgv = buildDockerArgv({
    argv: ["vllm", "serve", "org/model"],
    env: {},
    meta: nvidia,
  });
  assert.deepEqual(nvidiaArgv.slice(-2), [nvidia.image, "org/model"]);

  const amd = computeDockerMeta(recipe, variant, { brand: "AMD" });
  assert.equal(amd.brandKey, "amd");
  assert.equal(amd.image, "vllm/vllm-openai-rocm:latest");
  assert.match(amd.gpuFlags, /--device=\/dev\/kfd/);

  const tpu = computeDockerMeta(recipe, variant, {
    brand: "Google",
    generation: "tpu",
  });
  assert.equal(tpu.brandKey, "tpu");
  assert.equal(tpu.image, "vllm/vllm-tpu:latest");
  assert.match(tpu.gpuFlags, /--privileged --network host/);
});
