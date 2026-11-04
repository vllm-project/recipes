import assert from "node:assert/strict";
import { test } from "node:test";
import { isVariantHardwareSupported } from "../src/lib/command-synthesis.js";

test("exact hardware denial takes precedence over an allowlist", () => {
  const variant = { supported_hardware: ["ascend_910c", "h200"], hardware_overrides: { ascend_910c: { support: "unsupported" } } };
  assert.equal(isVariantHardwareSupported(variant, "ascend_910c"), false);
  assert.equal(isVariantHardwareSupported(variant, "h200"), true);
});

test("default variants can be denied without restricting other hardware", () => {
  const variant = { hardware_overrides: { ascend_950dt: { support: "unsupported" } } };
  assert.equal(isVariantHardwareSupported(variant, "ascend_950dt"), false);
  assert.equal(isVariantHardwareSupported(variant, "mi350x"), true);
  assert.equal(isVariantHardwareSupported({}, "ascend_950dt"), true);
});

test("supported does not bypass an existing allowlist", () => {
  assert.equal(isVariantHardwareSupported({ supported_hardware: ["h200"], hardware_overrides: { ascend_910c: { support: "supported" } } }, "ascend_910c"), false);
});
