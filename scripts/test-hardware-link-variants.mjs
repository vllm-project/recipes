import assert from "node:assert/strict";
import { test } from "node:test";
import { pickLinkedVariant } from "../src/lib/command-synthesis.js";

const profiles = {
  h100: {
    brand: "NVIDIA", generation: "hopper", compute_arch: "sm90",
    gpu_count: 1, vram_gb: 80, scalable: false,
  },
  ascend_310p3: {
    brand: "Huawei", generation: "npu", gpu_count: 1,
    vram_gb: 48, scalable: false, restricted: true,
  },
  ascend_910b3: {
    brand: "Huawei", generation: "npu", gpu_count: 1,
    vram_gb: 64, scalable: false, restricted: true,
  },
  ascend_910b4: {
    brand: "Huawei", generation: "npu", gpu_count: 1,
    vram_gb: 32, scalable: false, restricted: true,
  },
};

const recipe = {
  meta: {
    hardware: {
      ascend_310p3: "supported",
      ascend_910b3: "supported",
      ascend_910b4: "supported",
    },
  },
  variants: {
    default: {
      precision: "bf16", vram_minimum_gb: 2, supported_hardware: ["arch:sm90"],
    },
    ascend_fp16: {
      precision: "fp16", vram_minimum_gb: 3, supported_hardware: ["ascend_310p3"],
    },
    ascend_bf16: {
      precision: "bf16", vram_minimum_gb: 3,
      supported_hardware: ["ascend_910b3", "ascend_910b4"],
    },
  },
};

test("hardware-only Ascend 310P link selects FP16, A2 links select BF16", () => {
  assert.equal(pickLinkedVariant(recipe, null, "ascend_310p3", profiles), "ascend_fp16");
  assert.equal(pickLinkedVariant(recipe, null, "ascend_910b3", profiles), "ascend_bf16");
  assert.equal(pickLinkedVariant(recipe, null, "ascend_910b4", profiles), "ascend_bf16");
});

test("explicit variant selection is never silently replaced", () => {
  assert.equal(pickLinkedVariant(recipe, "default", "ascend_310p3", profiles), "default");
  assert.equal(pickLinkedVariant(recipe, "ascend_fp16", "ascend_910b4", profiles), "ascend_fp16");
});

test("GPU and absent/invalid links keep their normal default", () => {
  assert.equal(pickLinkedVariant(recipe, null, "h100", profiles), "default");
  assert.equal(pickLinkedVariant(recipe, null, "missing-device", profiles), "default");
  assert.equal(pickLinkedVariant(recipe, null, null, profiles), "default");
});

test("restricted hardware cannot be enabled without recipe opt-in", () => {
  const undeclared = {
    ...recipe,
    meta: { hardware: { ascend_910b3: "supported" } },
  };
  assert.equal(pickLinkedVariant(undeclared, null, "ascend_310p3", profiles), "default");
  assert.equal(pickLinkedVariant(undeclared, null, "ascend_910b4", profiles), "default");
});
