/**
 * Build-time: fetch HF org avatars to public/providers/<hf_org>.<ext>.
 *
 * Driven by src/lib/providers.js — it has an explicit list of HF orgs.
 * Each org has a logo path like "/providers/<hf_org>.png" — this script
 * populates that file at build time so Vercel's CDN serves them globally.
 *
 * Skips orgs already downloaded (in case of incremental builds).
 */

import fs from "fs";
import path from "path";
import { PROVIDERS } from "../src/lib/providers.js";

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

for (const [org, meta] of Object.entries(PROVIDERS)) {
  if (!meta.logo) continue;
  const localPath = path.join("public", meta.logo);
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
