import assert from "node:assert/strict";
import { test } from "node:test";
import { pickLinkedVariant } from "../src/lib/command-synthesis.js";

const profiles = {
  h100: {
    brand: "NVIDIA", generation: "hopper", compute_arch: "sm90",
    gpu_count: 1, vram_gb: 80, scalable: false,
  },
  example_npu_fp16: {
    brand: "Huawei", generation: "npu", gpu_count: 1,
    vram_gb: 48, scalable: false, restricted: true,
  },
  ascend_910b: {
    brand: "Huawei", generation: "npu", gpu_count: 1,
    vram_gb: 64, scalable: false, restricted: true,
  },
  example_npu_bf16: {
    brand: "Huawei", generation: "npu", gpu_count: 1,
    vram_gb: 32, scalable: false, restricted: true,
  },
};

const recipe = {
  meta: {
    hardware: {
      example_npu_fp16: "supported",
      ascend_910b: "supported",
      example_npu_bf16: "supported",
    },
  },
  variants: {
    default: {
      // Mirrors the real PaddleOCR-VL core recipe: BF16 is generic (no allowlist).
      precision: "bf16", vram_minimum_gb: 2,
    },
    ascend_fp16: {
      precision: "fp16", vram_minimum_gb: 3, supported_hardware: ["example_npu_fp16"],
    },
    ascend_fp16_eager: {
      precision: "fp16", vram_minimum_gb: 3, supported_hardware: ["example_npu_fp16"],
    },
    ascend_bf16: {
      precision: "bf16", vram_minimum_gb: 3,
      supported_hardware: ["ascend_910b", "example_npu_bf16"],
    },
  },
};

test("generic BF16 default yields to exact NPU FP16 and BF16 variants", () => {
  assert.equal(pickLinkedVariant(recipe, null, "example_npu_fp16", profiles), "ascend_fp16");
  assert.equal(pickLinkedVariant(recipe, null, "ascend_910b", profiles), "ascend_bf16");
  assert.equal(pickLinkedVariant(recipe, null, "example_npu_bf16", profiles), "ascend_bf16");
});

test("explicit variant selection is never silently replaced", () => {
  assert.equal(pickLinkedVariant(recipe, "default", "example_npu_fp16", profiles), "default");
  assert.equal(pickLinkedVariant(recipe, "ascend_fp16", "example_npu_bf16", profiles), "ascend_fp16");
});

test("GPU and absent/invalid links keep their normal default", () => {
  assert.equal(pickLinkedVariant(recipe, null, "h100", profiles), "default");
  assert.equal(pickLinkedVariant(recipe, null, "missing-device", profiles), "default");
  assert.equal(pickLinkedVariant(recipe, null, null, profiles), "default");
});

test("restricted hardware cannot be enabled without recipe opt-in", () => {
  const undeclared = {
    ...recipe,
    meta: { hardware: { ascend_910b: "supported" } },
  };
  assert.equal(pickLinkedVariant(undeclared, null, "example_npu_fp16", profiles), "default");
  assert.equal(pickLinkedVariant(undeclared, null, "example_npu_bf16", profiles), "default");
});

test("author default_variant may override automatic exact-hardware selection", () => {
  const preferred = {
    ...recipe,
    meta: {
      ...recipe.meta,
      default_variant: { example_npu_fp16: "ascend_fp16_eager" },
    },
  };
  assert.equal(pickLinkedVariant(preferred, null, "example_npu_fp16", profiles), "ascend_fp16_eager");
});

test("a generic BF16 model without an exact hardware variant stays on default", () => {
  const generic = { ...recipe, variants: { default: recipe.variants.default } };
  assert.equal(pickLinkedVariant(generic, null, "example_npu_fp16", profiles), "default");
});
