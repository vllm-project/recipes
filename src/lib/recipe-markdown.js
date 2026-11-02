// Markdown rendering of one recipe selection — what `/<hf_id>.md?<query>`
// serves, for AI assistants and agents. The rendered command leads and is
// marked authoritative; the guide comes last as background, since its
// hand-written examples may target another configuration.

import { whenMatches } from "./recipe-context.js";
import { SELECTION_PARAMS, defaultFeaturesFor, synthesizeInstall } from "./recipe-render.js";

const fence = (code, lang = "bash") => {
  const ticks = /```/.test(code) ? "````" : "```";
  return `${ticks}${lang}\n${code.trimEnd()}\n${ticks}`;
};

const envExports = (env) => Object.entries(env || {}).map(([k, v]) => `export ${k}=${v}`).join("\n");

const withEnv = (env, command) => [envExports(env), command].filter(Boolean).join("\n");

function labelVariant(recipe, key) {
  const v = recipe.variants?.[key] || {};
  return v.label || (v.precision || key).toUpperCase();
}

function labelFeature(recipe, key) {
  return recipe.features?.[key]?.label || key.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
}

function configLine(recipe, resolved, strategies) {
  const { selection, hwProfile, rendering } = resolved;
  const hw = `${hwProfile.gpu_count ? `${hwProfile.gpu_count}× ` : ""}${hwProfile.display_name || selection.hardware}`;
  const parts = [`hardware \`${selection.hardware}\` (${hw})`];
  if (!resolved.omni) {
    parts.push(`strategy \`${selection.strategy}\` (${strategies[selection.strategy]?.display_name || selection.strategy})`);
    if (rendering?.deploy_type === "multi_node" || rendering?.deploy_type === "kv_store_lb") parts.push(`${rendering.node_count} node${rendering.node_count === 1 ? "" : "s"}${rendering.deploy_type === "kv_store_lb" ? " per instance" : ""}`);
  }
  parts.push(`variant \`${selection.variant}\` (${labelVariant(recipe, selection.variant)})`);
  if (selection.task) parts.push(`task \`${selection.task}\``);
  if (selection.features?.length) parts.push(`features ${selection.features.map((f) => `\`${f}\``).join(", ")}`);
  if (selection.kv_offload) parts.push(`KV offload \`${selection.kv_offload}\``);
  if (selection.pd_router && selection.pd_router !== "vllm") parts.push(`PD router \`${selection.pd_router}\``);
  if (selection.preset) parts.push(`preset \`${selection.preset}\`${selection.point ? ` at point \`${selection.point}\`` : ""}`);
  if (selection.frontend && selection.frontend !== "python") parts.push(`frontend \`${selection.frontend}\``);
  return parts.join(" · ");
}

function serveSection(r) {
  const out = [];
  const dockerNote = (cmd) => cmd ? ["", "Docker equivalent:", "", fence(cmd)] : [];
  if (r.deploy_type === "multi_node") {
    const workers = r.worker_commands || [r.worker_command];
    out.push(`Run one command per node: the head (node rank 0) plus ${workers.length} worker${workers.length === 1 ? "" : "s"}. Each node's command differs — don't replay one on every node.`);
    out.push("", "### Head (node 0)", "", fence(withEnv(r.env, r.head_command)));
    workers.forEach((cmd, i) => out.push("", `### Node ${i + 1}`, "", fence(withEnv(r.env, cmd))));
    out.push(...dockerNote(r.head_docker_command && `# Head\n${r.head_docker_command}`));
    return out;
  }
  if (r.deploy_type === "pd_cluster") {
    const mc = r.mooncake;
    out.push(`Disaggregated prefill/decode${r.pd_connector ? ` (KV transfer: \`${r.pd_connector}\`)` : ""}. ${mc || r.infra ? "Start in this order" : "Start both pools, then the router"}.`);
    const step = (title, description, command) => {
      out.push("", `### ${title}`, "");
      if (description) out.push(description, "");
      out.push(fence(command));
    };
    if (mc) {
      if (mc.install) out.push("", "Mooncake on every prefill/decode node:", "", fence(mc.install));
      step("Mooncake config (on every prefill/decode node)", mc.configNote,
        [`cat > $MOONCAKE_CONFIG_PATH <<EOF\n${JSON.stringify(mc.config, null, 2)}\nEOF`,
          mc.store?.config && `cat > $MOONCAKE_STORE_CONFIG_PATH <<EOF\n${JSON.stringify(mc.store.config, null, 2)}\nEOF`].filter(Boolean).join("\n\n"));
      step("Mooncake master", mc.master.description, mc.master.command);
      if (mc.store) step("Mooncake store", mc.store.description, mc.store.command);
    }
    if (r.infra) step(r.infra.label || "Control plane", r.infra.description, r.infra.command);
    for (const i of r.orchestrator_install || []) out.push("", i.note ? `${i.note}:` : "Install:", "", fence(i.command));
    if (r.worker_install) out.push("", "On every prefill/decode node:", "", fence(r.worker_install));
    for (const role of ["prefill", "decode"]) {
      const p = r[role];
      if (!p?.command) continue;
      const nodes = p.nodes ?? 1;
      const scope = nodes > 1
        ? ` — pool spans ${nodes} nodes; this is node ${p.currentNode ?? 0}'s command (add \`${role}_rank=<n>\` to the query for another)`
        : "";
      out.push("", `### ${role === "prefill" ? "Prefill" : "Decode"}${scope}`, "", fence(withEnv(p.env, p.command)));
    }
    if (r.router?.command) {
      out.push("", `### ${r.router.label || "Router"}`, "");
      if (r.router.install && !(r.orchestrator_install || []).some((i) => i.command === r.router.install)) out.push(fence(r.router.install), "");
      out.push(fence(withEnv(r.router.env, r.router.command)));
    }
    if (r.registration) step(r.registration.label || "Register workers", r.registration.description, r.registration.command);
    return out;
  }
  if (r.deploy_type === "kv_store_lb") {
    out.push(`Mooncake KV store around ${r.instances ?? 1} vLLM instance${r.instances === 1 ? "" : "s"}, each running \`${r.serving_strategy}\`. Start in this order.`);
    let step = 1;
    out.push("", `### ${step++}. Mooncake master`, "", fence(r.master?.command || ""));
    if (r.store?.command) out.push("", `### ${step++}. Mooncake store`, "", fence(r.store.command));
    if (r.mooncake_config) {
      out.push("", `### ${step++}. Mooncake config (on every vLLM node)`, "");
      if (r.mooncake_config_note) out.push(r.mooncake_config_note, "");
      out.push(fence(`cat > $MOONCAKE_CONFIG_PATH <<EOF\n${JSON.stringify(r.mooncake_config, null, 2)}\nEOF`));
    }
    out.push("", `### ${step++}. vLLM instance${r.vllm_worker_commands?.length ? " (head)" : ""}`, "");
    if (r.vllm_install) out.push(fence(r.vllm_install), "");
    out.push(fence(withEnv(r.env, r.vllm_command)));
    (r.vllm_worker_commands || []).forEach((cmd, i) => out.push("", `#### Instance node ${i + 1}`, "", fence(withEnv(r.env, cmd))));
    if (r.router?.command && (r.instances ?? 1) > 1) out.push("", `### ${step++}. Router`, "", fence(r.router.command));
    return out;
  }
  out.push(fence(withEnv(r.env, r.command)));
  for (const c of r.companions || []) {
    out.push("", `### ${c.label}${c.after ? " (start after vLLM)" : ""}`, "");
    if (c.description) out.push(c.description, "");
    out.push(fence(c.command));
  }
  out.push(...dockerNote(r.docker_command));
  return out;
}

function installSection(recipe, resolved) {
  const { hwProfile, rendering } = resolved;
  const out = [];
  const min = recipe.model?.min_vllm_version;
  if (min) out.push(`Requires vLLM >= ${min}${recipe.model?.nightly_required ? " (not released yet — nightly wheel)" : ""}.`, "");
  const install = synthesizeInstall(recipe);
  if (hwProfile.brand === "NVIDIA" && install?.pip) {
    if (install.pip.note) out.push(install.pip.note, "");
    out.push(fence(install.pip.command), "");
  }
  if (rendering?.docker_image) out.push(`Docker image: \`${rendering.docker_image}\``, "");
  const deps = (recipe.dependencies || []).filter((d) => {
    if (!d.brand) return true;
    return (Array.isArray(d.brand) ? d.brand : [d.brand]).includes(hwProfile.brand);
  });
  if (deps.length) {
    out.push("Extra dependencies:", "");
    for (const d of deps) {
      out.push(`- ${d.note || "Dependency"}${d.optional ? " (optional)" : ""}`, "", fence(d.command), "");
    }
  }
  return out;
}

function contextEntries(recipe, resolved, kind, hardwareProfiles) {
  const { selection } = resolved;
  const sel = { hwId: selection.hardware, strategy: selection.strategy, variant: selection.variant, features: selection.features };
  return (recipe[kind] || []).filter((e) => whenMatches(e.when, sel, hardwareProfiles));
}

// Nest the guide under "## Guide": every heading one level down, leaving
// fenced code (where `# ` is a shell comment) alone.
function demoteHeadings(md) {
  let fenceMark = null;
  return md.split("\n").map((line) => {
    const f = line.match(/^\s*(`{3,}|~{3,})/);
    if (f) {
      if (!fenceMark) fenceMark = f[1];
      else if (f[1][0] === fenceMark[0] && f[1].length >= fenceMark.length) fenceMark = null;
      return line;
    }
    return !fenceMark && /^#{1,5} /.test(line) ? `#${line}` : line;
  }).join("\n");
}

// Page-only anchors are dead links here: `#preset-<key>` (another preset's
// card) points to that preset's Markdown, `#issue-<slug>` to the page. Other
// anchors name headings this document carries too.
function pageLinks(md, recipe, origin) {
  return md.replace(/\]\(#((preset|issue)-[^)\s]+)\)/g, (_, id, kind) => {
    const key = kind === "preset" ? id.slice(7) : null;
    return recipe.presets?.[key]
      ? `](${origin}/${recipe.hf_id}.md?${presetQuery(key, recipe.presets[key])})`
      : `](${origin}/${recipe.hf_id}#${id})`;
  });
}

// The query that selects a preset, as the Presets tab's "Use this preset"
// link builds it: a variant-scoped preset needs its variant too.
function presetQuery(key, p) {
  const q = [`hardware=${p.hardware}`];
  if (p.variants?.[0] && p.variants[0] !== "default") q.push(`variant=${p.variants[0]}`);
  if (p.strategy) q.push(`strategy=${p.strategy}`);
  q.push(`preset=${key}`);
  return q.join("&");
}

const selectText = (sel) => Object.entries(sel || {}).flatMap(([k, v]) =>
  k === "features" && v && typeof v === "object"
    ? Object.entries(v).map(([f, on]) => `\`${f}\` ${on === false ? "off" : on === true ? "on" : `on (\`${on}\` mode)`}`)
    : [`\`${k}=${v === false ? "off" : v}\``]);

function imageLines(image) {
  if (!image) return [];
  const img = typeof image === "string" ? { vllm: image } : image;
  return [
    img.vllm && `- vLLM image: \`${img.vllm}\` (the Docker command above uses it)`,
    img.dynamo && `- Dynamo: \`${img.dynamo}\``,
    ...(img.pip || []).map((p) => `- pip: \`${p}\``),
    img.model_revision && `- Model revision: \`${img.model_revision}\``,
  ].filter(Boolean);
}

// The active tuned preset, as its card in the Presets tab shows it: what it
// is, its evidence, what it pins and implies, its points, and the runbook.
// Every tuned preset with the query that selects it — across all hardware, so
// a lane on another GPU is discoverable from any configuration.
function presetOptions(recipe, strategies) {
  const presets = Object.entries(recipe.presets || {});
  if (!presets.length) return [];
  return [
    "- `preset`: a tuned, evidence-backed configuration (`baseline` opts out of a default one). Each sets its own hardware and strategy:",
    ...presets.map(([k, p]) => `  - \`${k}\` — ${p.label || k} (${[p.hardware, p.strategy && (strategies[p.strategy]?.short_label || p.strategy), p.workload, p.default ? "default" : "opt-in"].filter(Boolean).join(" · ")}): \`${presetQuery(k, p)}\``),
  ];
}

function presetSection(recipe, rendering, strategies, origin) {
  const key = rendering?.preset;
  const p = key && recipe.presets?.[key];
  if (!p) return [];
  const out = [`## Tuned preset: ${p.label || key}`, ""];
  if (p.description) out.push(p.description.trim(), "");
  const scope = [`hardware \`${p.hardware}\``];
  if (p.strategy) scope.push(`strategy \`${p.strategy}\` (${strategies[p.strategy]?.display_name || p.strategy})`);
  if (p.variants?.length) scope.push(`variants ${p.variants.map((v) => `\`${v}\``).join(", ")}`);
  out.push(
    `- Key: \`${key}\` · ${p.default ? "applied by default on this hardware and strategy (\`preset=baseline\` drops it)" : "opt-in"}`,
    `- Qualified on: ${scope.join(" · ")}`,
  );
  if (p.workload) out.push(`- Workload: ${p.workload}`);
  if (p.source) {
    const extra = [p.source.key && `lane \`${p.source.key}\``, p.source.revision && `revision \`${String(p.source.revision).slice(0, 12)}\``].filter(Boolean);
    out.push(`- Source: ${p.source.href ? `[${p.source.label || p.source.href}](${p.source.href})` : p.source.label}${extra.length ? ` (${extra.join(", ")})` : ""}`);
  }
  out.push(...imageLines(p.image));
  const implied = selectText(p.select);
  if (implied.length) out.push(`- Builder choices it implies (already in the command above): ${implied.join(", ")}`);
  out.push(`- Every point rendered as JSON: ${origin}/${recipe.hf_id}/presets/${key}.json`, "");

  const points = Object.entries(p.points || {});
  if (points.length) {
    const byConcurrency = points.every(([, pt]) => pt.concurrency != null);
    const active = rendering.preset_point;
    out.push(`### ${byConcurrency ? "Concurrency points" : "Points"}`, "",
      `Each point is a delta on the preset; select one with \`point=<key>\`.`, "",
      `| Point | ${byConcurrency ? "Concurrency | " : ""}Select |`,
      `|---|${byConcurrency ? "---|" : ""}---|`);
    const defaultPoint = (points.find(([, pt]) => pt.default) || points[0])[0];
    for (const [pk, pt] of points) {
      const mark = [pk === active && "**in use**", pk === defaultPoint && "default"].filter(Boolean).join(", ");
      out.push(`| ${pt.label || pk}${mark ? ` (${mark})` : ""}${pt.description ? ` — ${pt.description.replace(/\s+/g, " ").trim()}` : ""} | ${byConcurrency ? `${pt.concurrency} | ` : ""}\`point=${pk}\` |`);
    }
    out.push("");
  }
  if (p.runbook?.trim()) out.push("### Runbook", "", pageLinks(demoteHeadings(demoteHeadings(p.runbook.trim())), recipe, origin), "");
  return out;
}

// resolved: resolveSelection's result; urls: { page, md, json } (absolute).
export function selectionMarkdown(recipe, resolved, { strategies, taxonomy, urls }) {
  const { rendering, options, selection } = resolved;
  const hardwareProfiles = taxonomy.hardware_profiles || {};
  const origin = new URL(urls.page).origin;
  const defaultFeatures = defaultFeaturesFor(recipe, selection.hardware, selection.variant);
  const lines = [
    `# ${recipe.hf_id} — vLLM serving recipe`,
    "",
    recipe.meta?.description ? `> ${recipe.meta.description.replace(/\s+/g, " ").trim()}` : null,
    recipe.meta?.description ? "" : null,
    `Configuration: ${configLine(recipe, resolved, strategies)}`,
    "",
    ...(selection.adjusted?.length ? [
      ...selection.adjusted.map((a) => `> **Note:** requested \`${a.param}=${a.requested}\` is ${a.reason}; rendered \`${a.used}\` instead.`),
      "",
    ] : []),
    `- Interactive page: ${urls.page}`,
    `- This configuration as JSON: ${urls.json}`,
    "",
    "The commands below were rendered for exactly this configuration and take precedence over any example in the guide.",
    "",
    "## Other configurations",
    "",
    `Change the query string (the same parameters as the page URL) and fetch this URL again. Unknown or incompatible values fall back to the recipe default, so check the Configuration line.`,
    "",
    `- \`hardware\`: ${options.hardware.map((h) => `\`${h}\``).join(", ")}`,
    `- \`variant\`: ${options.variants.map((v) => `\`${v}\``).join(", ")}`,
    options.strategies.length ? `- \`strategy\`: ${options.strategies.map((s) => `\`${s}\``).join(", ")} (multi-node strategies also need \`nodes=<n>\`)` : null,
    options.tasks?.length ? `- \`task\`: ${options.tasks.map((t) => `\`${t}\``).join(", ")}` : null,
    options.features.length ? `- \`features\`: comma-separated list of ${options.features.map((f) => `\`${f}\``).join(", ")} (default: ${defaultFeatures.map((f) => `\`${f}\``).join(", ") || "none"})` : null,
    ...presetOptions(recipe, strategies),
    "",
    "## Install",
    "",
    ...installSection(recipe, resolved),
    "## Serve",
    "",
  ].filter((l) => l !== null);

  if (!rendering) {
    lines.push("This configuration couldn't be rendered. Try another `hardware` or `strategy`.", "");
  } else if (resolved.omni) {
    lines.push("vLLM-Omni recipe — the generation endpoints and request examples are in the guide below.", "", fence(withEnv(rendering.env, rendering.command)), "");
  } else {
    lines.push(...serveSection(rendering), "");
  }
  lines.push(...presetSection(recipe, rendering, strategies, origin));

  const notes = contextEntries(recipe, resolved, "notes", hardwareProfiles);
  if (notes.length) {
    lines.push("## Notes for this configuration", "");
    for (const n of notes) {
      if (n.title) lines.push(`### ${n.title}`, "");
      lines.push(pageLinks(n.body.trim(), recipe, origin), "");
    }
  }

  const matching = contextEntries(recipe, resolved, "troubleshooting", hardwareProfiles);
  if (matching.length) {
    lines.push("## Known issues", "");
    for (const t of matching) lines.push(`### ${t.title}`, "", pageLinks(t.body.trim(), recipe, origin), "");
  }

  if (recipe.guide?.trim()) {
    lines.push(
      "## Guide",
      "",
      "> Background reference. Commands in it are examples and may target another configuration — use the Serve section above.",
      "",
      pageLinks(demoteHeadings(recipe.guide.trim()), recipe, origin),
      "",
    );
  }
  return lines.join("\n");
}

// The `.md` / `.json` twin of a page URL: same path + extension, same query
// (selection params only). A bare `/<hf_id>.json` is the static whole-recipe
// file, so the JSON link always names the hardware to get the rendering.
export function selectionUrls(origin, hfId, query, hardware) {
  const q = new URLSearchParams();
  for (const k of SELECTION_PARAMS) if (query.get(k) != null) q.set(k, query.get(k));
  const qs = q.toString() ? `?${q}` : "";
  const jq = new URLSearchParams(q);
  if (!jq.has("hardware") && hardware) jq.set("hardware", hardware);
  return {
    page: `${origin}/${hfId}${qs}`,
    md: `${origin}/${hfId}.md${qs}`,
    json: `${origin}/${hfId}.json${jq.toString() ? `?${jq}` : ""}`,
  };
}
