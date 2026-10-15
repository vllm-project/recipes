/**
 * Describe how sources.lock.json differs from HEAD, as a commit message
 * (used by .github/workflows/bump-sources.yml). Prints nothing when the lock
 * is unchanged.
 */

import fs from "fs";
import path from "path";
import { execFileSync } from "child_process";
import { LOCK_FILE, ROOT } from "./lib/content-sources.mjs";

function headLock() {
  try {
    const rel = path.relative(ROOT, LOCK_FILE);
    return JSON.parse(execFileSync("git", ["show", `HEAD:${rel}`], { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }));
  } catch {
    return {};
  }
}

const before = headLock();
const after = fs.existsSync(LOCK_FILE) ? JSON.parse(fs.readFileSync(LOCK_FILE, "utf8")) : {};
const names = [];
const lines = [];

for (const [channel, sources] of Object.entries(after)) {
  for (const [name, entry] of Object.entries(sources)) {
    const prev = before[channel]?.[name];
    const pinned = (e) => e && JSON.stringify({ ...e, resolved_at: undefined });
    if (pinned(prev) === pinned(entry)) continue;
    names.push(`${channel}.${name}`);
    const from = prev ? prev.sha.slice(0, 12) : "(new)";
    const moved = prev && (prev.path !== entry.path || prev.repo !== entry.repo || prev.ref !== entry.ref)
      ? ` (was ${prev.repo}@${prev.ref}:${prev.path})` : "";
    lines.push(`- ${channel}.${name}: ${entry.repo}@${entry.ref}:${entry.path} ${from} → ${entry.sha.slice(0, 12)}${moved}`);
    if (prev && prev.repo === entry.repo && prev.sha !== entry.sha) {
      lines.push(`  https://github.com/${entry.repo}/compare/${prev.sha}...${entry.sha}`);
    }
  }
}
for (const [channel, sources] of Object.entries(before)) {
  for (const name of Object.keys(sources)) {
    if (after[channel]?.[name]) continue;
    names.push(`${channel}.${name}`);
    lines.push(`- ${channel}.${name}: removed`);
  }
}

if (names.length) {
  console.log(`Bump recipe sources: ${names.join(", ")}\n\n${lines.join("\n")}`);
}
