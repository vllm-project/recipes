/**
 * Build-time: fetch HF org avatars to public/providers/<hf_org>.<ext>.
 *
 * Two sources feed this:
 *  1. src/lib/providers.js — model providers (DeepSeek, Qwen, …). Each has a
 *     logo path like "/providers/<hf_org>.png".
 *  2. HARDWARE_LOGOS below — hardware vendors shown in the Hardware filter row
 *     of BrowseList.jsx. These aren't model providers, so they're not in
 *     PROVIDERS. NVIDIA and Google ride along because they're also providers.
 *
 * Populates these files at build time so Vercel's CDN serves them globally.
 * public/ is gitignored, so anything not fetched here 404s in production.
 * Skips files already on disk (in case of incremental builds).
 */

import fs from "fs";
import path from "path";
import { PROVIDERS } from "../src/lib/providers.js";

// Hardware vendors referenced by BrowseList.jsx that have no entry in
// PROVIDERS. Ascend's official community favicon is checked in so the
// hardware logo stays available without a build-time request to its site.
// Source: https://www.hiascend.com/_static3/favicon.ico
const HARDWARE_LOGOS = {
  amd: { logo: "/providers/amd.png" },
  Intel: { logo: "/providers/intel.png" },
  Huawei: { logo: "/providers/ascend.png", source: "assets/hardware/ascend.png" },
};

const OUT = "public/providers";
fs.mkdirSync(OUT, { recursive: true });

async function fetchOrgAvatarUrl(org) {
  const res = await fetch(`https://huggingface.co/${org}`, {
    headers: { "User-Agent": "vllm-recipes-build/1.0" },
  });
  const html = await res.text();
  const match = html.match(/cdn-avatars\.huggingface\.co\/[^"&]+/);
  return match ? `https://${match[0]}` : null;
}

let fetched = 0;
let skipped = 0;
let failed = 0;

// Merge both sources; PROVIDERS wins if an org appears in both.
const targets = { ...HARDWARE_LOGOS, ...PROVIDERS };

for (const [org, meta] of Object.entries(targets)) {
  if (!meta.logo) continue;
  const localPath = path.join("public", meta.logo);
  if (meta.source) {
    fs.copyFileSync(meta.source, localPath);
    console.log(`✓ ${org} (local asset)`);
    fetched++;
    continue;
  }
  if (fs.existsSync(localPath)) {
    skipped++;
    continue;
  }
  try {
    const avatarUrl = await fetchOrgAvatarUrl(org);
    if (!avatarUrl) {
      console.warn(`⚠ no avatar found for ${org}`);
      failed++;
      continue;
    }
    const imgRes = await fetch(avatarUrl);
    const buffer = Buffer.from(await imgRes.arrayBuffer());
    fs.writeFileSync(localPath, buffer);
    console.log(`✓ ${org} (${buffer.length}B)`);
    fetched++;
  } catch (e) {
    console.warn(`✗ ${org}: ${e.message}`);
    failed++;
  }
}

console.log(`\n${fetched} fetched, ${skipped} cached, ${failed} failed`);
if (failed > 0) {
  console.warn("Some logos failed — continuing build anyway");
}
