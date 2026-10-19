/**
 * Credit the people who wrote each recipe: refresh recipe-contributors.json
 * (hf_id → { contributors: [{ login, commits }], history: [url] }) from the git
 * history of the recipe's YAML. The recipe page shows both, and the JSON API
 * publishes the contributors.
 *
 *   GITHUB_TOKEN=… node scripts/fetch-contributors.mjs
 *
 * Reads the assembled channel (run sync-sources + assemble-content first) for
 * where each recipe lives, then asks the GitHub GraphQL API for the commits
 * that touched it, in its owning repo, in any overlay, and at its old path in
 * vllm-project/recipes (models/<hf_id>.yaml) from before it moved. Commit
 * authors and Co-authored-by trailers both count, matched to GitHub accounts
 * by email. Not counted:
 *   - bots and AI assistants (co-author trailers such as Claude's)
 *   - authors whose email isn't linked to a GitHub account
 *   - commits that touched more than BULK_COMMIT_RECIPES recipes — catalog-wide
 *     maintenance (field renames, version bumps), not authorship of each one —
 *     except the commit that created the file, which always counts
 *   - commits that copy recipes between repos (the move to vllm and the syncs
 *     that followed, titled "[Doc] Sync … vllm-project/recipes…"); the change
 *     is credited where it was first made
 * Run by .github/workflows/recipe-contributors.yml, which commits the result.
 */

import fs from "fs";
import path from "path";
import { loadProvenance } from "../src/lib/content-root.js";

const OUT = path.join(process.cwd(), "recipe-contributors.json");
const LEGACY_REPO = "vllm-project/recipes";
const BULK_COMMIT_RECIPES = 20;
const COPY_COMMIT = [
  /^\[Doc\] Host model recipes in recipes\//,
  /^\[Doc\] Sync .*vllm-project\/recipes/,
  /^\[Infra\] Remove the copies of the vLLM recipes/,
];
// Coding agents that sign commits as co-authors; they are not contributors.
const AI_LOGINS = new Set(["claude", "codex", "copilot", "copilot-swe-agent", "cursoragent", "devin-ai-integration", "mistral-vibe"]);
const BATCH = 20;

const token = process.env.GITHUB_TOKEN;
if (!token) {
  console.error("GITHUB_TOKEN is required (the GraphQL API has no anonymous access)");
  process.exit(2);
}

async function graphql(query) {
  for (let attempt = 1; ; attempt++) {
    const res = await fetch("https://api.github.com/graphql", {
      method: "POST",
      headers: { authorization: `bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const body = await res.json().catch(() => ({}));
    if (res.ok && !body.errors) return body.data;
    if (attempt >= 3) throw new Error(`GraphQL ${res.status}: ${JSON.stringify(body.errors || body).slice(0, 500)}`);
    await new Promise((r) => setTimeout(r, 2000 * attempt));
  }
}

const HISTORY_FIELDS = `pageInfo { hasNextPage endCursor }
  nodes { oid messageHeadline committedDate
    authors(first: 10) { nodes { email user { login } } } }`;

// One history query per (repo, ref, path), batched as aliases.
async function fetchHistories(targets) {
  const out = new Map(targets.map((t) => [t.key, []]));
  let pending = targets.map((t) => ({ ...t, cursor: null }));
  while (pending.length) {
    const next = [];
    for (let i = 0; i < pending.length; i += BATCH) {
      const batch = pending.slice(i, i + BATCH);
      const query = `query {\n${batch.map((t, j) => {
        const [owner, name] = t.repo.split("/");
        const after = t.cursor ? `, after: ${JSON.stringify(t.cursor)}` : "";
        return `t${j}: repository(owner: ${JSON.stringify(owner)}, name: ${JSON.stringify(name)}) {
          object(expression: ${JSON.stringify(t.ref)}) { ... on Commit {
            history(first: 100, path: ${JSON.stringify(t.path)}${after}) { ${HISTORY_FIELDS} } } } }`;
      }).join("\n")}\n}`;
      const data = await graphql(query);
      batch.forEach((t, j) => {
        const history = data[`t${j}`]?.object?.history;
        if (!history) throw new Error(`${t.repo}@${t.ref}: no commit history for ${t.path}`);
        out.get(t.key).push(...history.nodes);
        if (history.pageInfo.hasNextPage) next.push({ ...t, cursor: history.pageInfo.endCursor });
      });
    }
    pending = next;
  }
  return out;
}

function isPerson(login, email) {
  if (!login) return false;                                 // email not linked to an account
  const l = login.toLowerCase();
  return !l.endsWith("[bot]") && !AI_LOGINS.has(l) && !/@anthropic\.com$/i.test(email || "");
}

const provenance = loadProvenance();
const hfIds = Object.keys(provenance).sort();
if (!hfIds.length) {
  console.error("No assembled recipes — run sync-sources.mjs and assemble-content.mjs first");
  process.exit(2);
}

// Where each recipe's history lives: its owner, its overlays, and its old path
// in this repo. The same (repo, path) is fetched once.
const targets = new Map();
const recipeTargets = {};
for (const hfId of hfIds) {
  const p = provenance[hfId];
  const places = [p, ...(p.overlays || [])]
    .filter((s) => s.repo)
    .map((s) => ({ repo: s.repo, ref: s.sha || s.ref || "main", branch: s.ref || "main", path: s.path }));
  places.push({ repo: LEGACY_REPO, ref: "main", branch: "main", path: `models/${hfId}.yaml` });
  recipeTargets[hfId] = [];
  for (const t of places) {
    const key = `${t.repo}:${t.path}`;
    if (!targets.has(key)) targets.set(key, { ...t, key });
    if (!recipeTargets[hfId].includes(key)) recipeTargets[hfId].push(key);
  }
}

const histories = await fetchHistories([...targets.values()]);

// How many recipes each commit touched, from the histories themselves.
const commitsByRecipe = {};
const recipesPerCommit = new Map();
for (const hfId of hfIds) {
  const commits = new Map();
  for (const key of recipeTargets[hfId]) {
    const history = histories.get(key); // newest first
    history.forEach((c, i) => commits.set(c.oid, { ...c, created: i === history.length - 1 }));
  }
  commitsByRecipe[hfId] = [...commits.values()];
  for (const oid of commits.keys()) recipesPerCommit.set(oid, (recipesPerCommit.get(oid) || 0) + 1);
}

const result = {};
let skippedBulk = 0;
for (const hfId of hfIds) {
  const people = new Map(); // login → { commits, first }
  for (const c of commitsByRecipe[hfId]) {
    if (COPY_COMMIT.some((re) => re.test(c.messageHeadline))) continue;
    if (!c.created && recipesPerCommit.get(c.oid) > BULK_COMMIT_RECIPES) { skippedBulk++; continue; }
    const logins = new Set(c.authors.nodes.filter((a) => isPerson(a.user?.login, a.email)).map((a) => a.user.login));
    for (const login of logins) {
      const p = people.get(login) || { commits: 0, first: c.committedDate };
      p.commits++;
      if (c.committedDate < p.first) p.first = c.committedDate;
      people.set(login, p);
    }
  }
  result[hfId] = {
    // Most commits first; ties by who contributed first.
    contributors: [...people]
      .sort(([la, a], [lb, b]) => b.commits - a.commits || a.first.localeCompare(b.first) || la.localeCompare(lb))
      .map(([login, { commits }]) => ({ login, commits })),
    // GitHub history pages that have commits: the owning repo, then the old path.
    history: recipeTargets[hfId]
      .filter((key) => histories.get(key).length)
      .map((key) => targets.get(key))
      .map((t) => `https://github.com/${t.repo}/commits/${t.branch}/${t.path}`),
  };
}

fs.writeFileSync(OUT, JSON.stringify(result, null, 2) + "\n");
const people = new Set(Object.values(result).flatMap((r) => r.contributors.map((c) => c.login)));
console.log(`✓ recipe-contributors.json: ${hfIds.length} recipes, ${people.size} contributors (${targets.size} file histories; ${skippedBulk} bulk-commit credits skipped)`);
