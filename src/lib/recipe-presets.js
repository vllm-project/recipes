// Tuned presets: named, evidence-backed configurations for one hardware
// profile (and optionally one serving strategy and some variants), e.g. the
// InferenceX AgentX lane a vendor team qualified on GB200. A preset is a delta
// on top of the recipe's baseline command, so it never restates shared flags:
//
//   presets:
//     gb200_agentx_pd:
//       label: "AgentX 1P1D"
//       hardware: gb200            # exact profile id (required)
//       strategy: pd_cluster       # optional; absent = every strategy
//       variants: [default]        # optional; absent = every variant
//       workload: "Agentic"        # optional free-text tag
//       default: true              # applied unless the user picks Baseline
//       source: { label, href, key, revision }
//       image: "vllm/vllm-openai:…"  # or { vllm, dynamo, pip: [], model_revision }
//       tp: 4                      # single_node_tp only: TP size on this GPU
//       select:                    # builder choices the preset implies
//         nodes: 2                 # multi_node_* only: node count
//         pd_router: dynamo
//         pd_connector: moriio     # pd_cluster.yaml `connectors` key
//         frontend: python
//         features: { tool_calling: false, spec_decoding: dspark }
//         dcp: 16
//         kv_offload: simple
//       args: [...]                # every node / both PD roles
//       env: {...}
//       remove_args: [--block-size]          # drop inherited flags
//       json_args:                           # edit keys inside JSON-valued flags
//         --speculative-config: { num_speculative_tokens: 4 }   # null deletes a key
//       roles:                     # pd_cluster only; per role: args, env,
//         decode: { nodes: 4, parallelism: tp, args: [...] }   # remove_args, json_args
//       points:                    # named operating points (e.g. per concurrency)
//         c16: { label: "C16", concurrency: 16, default: true, roles: {...} }
//         c32: { label: "C32", concurrency: 32, args: [...], select: {...} }
//       runbook: |                 # prose only: evidence, reproduction, limits
//
// A point has the preset's own delta fields (args, env, remove_args,
// json_args, roles, select, image, tp) and layers on top of it, so the preset
// holds what every point shares. A point can't change the hardware, strategy
// or variants: that is another preset.
//
// The active preset is mounted into the exact-GPU strategy and role layers
// (`strategy_overrides.<s>.hardware_overrides.<hw>` and
// `strategy_overrides.<s>.<role>.hardware_overrides.<hw>`), which synthesis
// already applies last and additively; a role's `nodes` / `parallelism` become
// the `<hw>` entry of that role's hardware-keyed map. `remove_args` and
// `json_args` ride along as `_presetOps`, applied after every arg layer
// (features included). A recipe that moves such blocks into a `default: true`
// preset therefore renders exactly as before, and Baseline drops the lane's
// topology together with its flags.

export const BASELINE = "baseline";
const ROLES = ["prefill", "decode"];

const presetApplies = (p, hwId, strategyName, variantKey) =>
  p?.hardware === hwId
  && (!p.strategy || p.strategy === strategyName)
  && (!p.variants?.length || variantKey == null || p.variants.includes(variantKey));

/** Presets that apply to this hardware + strategy (+ variant), strategy-specific first. */
export function matchingPresets(recipe, hwId, strategyName, variantKey) {
  const all = Object.entries(recipe?.presets || {})
    .filter(([, p]) => presetApplies(p, hwId, strategyName, variantKey));
  return [
    ...all.filter(([, p]) => p.strategy),
    ...all.filter(([, p]) => !p.strategy),
  ].map(([key, p]) => ({ key, ...p }));
}

/**
 * The preset key in effect: an explicit matching pick, BASELINE when the user
 * opted out, else the matching default (strategy-specific before
 * hardware-wide), else BASELINE.
 */
export function resolvePresetKey(recipe, hwId, strategyName, pick, variantKey) {
  const options = matchingPresets(recipe, hwId, strategyName, variantKey);
  if (pick === BASELINE) return BASELINE;
  if (pick && options.some((p) => p.key === pick)) return pick;
  return options.find((p) => p.default)?.key ?? BASELINE;
}

/** A preset's points in YAML order, as `{ key, ...point }`. */
export function presetPoints(preset) {
  return Object.entries(preset?.points || {}).map(([key, pt]) => ({ key, ...pt }));
}

/** The point in effect: a valid pick, else the `default: true` point, else the first. */
export function resolvePointKey(preset, pick) {
  const points = presetPoints(preset);
  if (!points.length) return null;
  if (pick && points.some((pt) => pt.key === pick)) return pick;
  return (points.find((pt) => pt.default) || points[0]).key;
}

const concat = (a, b) => (b?.length ? [...(a || []), ...b] : a);
const assign = (a, b) => (b && Object.keys(b).length ? { ...(a || {}), ...b } : a);
const isObj = (v) => v && typeof v === "object" && !Array.isArray(v);

// Deep merge for JSON edits and `select`: objects merge, anything else
// (including null, which json_args reads as "delete") replaces.
function deepMerge(a, b) {
  if (!isObj(a) || !isObj(b)) return b === undefined ? a : b;
  const out = { ...a };
  for (const [k, v] of Object.entries(b)) out[k] = deepMerge(a[k], v);
  return out;
}

// Delta fields shared by a preset, its roles and its points.
function mergeDelta(a = {}, b = {}) {
  const out = { ...a };
  const args = concat(a.args, b.args);
  const env = assign(a.env, b.env);
  const remove = concat(a.remove_args, b.remove_args);
  const json = b.json_args ? deepMerge(a.json_args || {}, b.json_args) : a.json_args;
  if (args) out.args = args;
  if (env) out.env = env;
  if (remove) out.remove_args = remove;
  if (json) out.json_args = json;
  for (const k of ["nodes", "parallelism", "tp"]) if (b[k] !== undefined) out[k] = b[k];
  return out;
}

/** The preset with point `pointKey` (or its default point) layered on. */
export function effectivePreset(preset, pointKey) {
  if (!preset) return null;
  const key = resolvePointKey(preset, pointKey);
  const pt = key ? preset.points[key] : null;
  if (!pt) return preset;
  const out = { ...mergeDelta(preset, pt), point: key };
  out.select = pt.select ? deepMerge(preset.select || {}, pt.select) : preset.select;
  // A point's image fields layer over the preset's (a point may pin only a
  // different Dynamo commit and keep the preset's vLLM image).
  if (pt.image) out.image = { ...(presetImage(preset.image) || {}), ...presetImage(pt.image) };
  if (preset.roles || pt.roles) {
    out.roles = {};
    for (const role of ROLES) {
      const merged = mergeDelta(preset.roles?.[role], pt.roles?.[role]);
      if (Object.keys(merged).length) out.roles[role] = merged;
    }
  }
  return out;
}

/** `image` as an object: { vllm?, dynamo?, pip?, model_revision? }. */
export function presetImage(image) {
  if (!image) return null;
  return typeof image === "string" ? { vllm: image } : image;
}

// Add an exact-GPU entry to a value that may be a scalar or a hardware-keyed map.
function mountKeyed(raw, hwId, value) {
  if (value === undefined) return raw;
  const map = raw && typeof raw === "object" ? { ...raw } : raw === undefined ? {} : { default: raw };
  map[hwId] = value;
  return map;
}

function mountLayer(block, delta = {}) {
  delta = delta || {};
  const next = { ...(block || {}) };
  const args = concat(next.extra_args, delta.args);
  const env = assign(next.extra_env, delta.env);
  if (args) next.extra_args = args;
  if (env) next.extra_env = env;
  return next;
}

const argOps = (d) => (d?.remove_args?.length || d?.json_args ? { remove_args: d.remove_args || [], json_args: d.json_args || {} } : null);

/**
 * The recipe with preset `key` (at `opts.point`) mounted for `strategyName`.
 * Returns the recipe unchanged for BASELINE or a preset that doesn't match.
 */
export function withPreset(recipe, key, hwId, strategyName, opts = {}) {
  const raw = key && key !== BASELINE ? recipe?.presets?.[key] : null;
  if (!raw || !presetApplies(raw, hwId, strategyName, opts.variantKey)) return recipe;
  const p = effectivePreset(raw, opts.point);
  const so = { ...(recipe.strategy_overrides?.[strategyName] || {}) };
  so.hardware_overrides = { ...(so.hardware_overrides || {}) };
  so.hardware_overrides[hwId] = mountLayer(so.hardware_overrides[hwId], p);
  const ops = { all: argOps(p), roles: {} };
  for (const [role, delta] of Object.entries(p.roles || {})) {
    const r = { ...(so[role] || {}) };
    r.hardware_overrides = { ...(r.hardware_overrides || {}) };
    r.hardware_overrides[hwId] = mountLayer(r.hardware_overrides[hwId], delta);
    for (const k of ["nodes", "parallelism"]) {
      const v = mountKeyed(r[k], hwId, delta?.[k]);
      if (v !== undefined) r[k] = v;
    }
    so[role] = r;
    const roleOps = argOps(delta);
    if (roleOps) ops.roles[role] = roleOps;
  }
  const out = { ...recipe, strategy_overrides: { ...(recipe.strategy_overrides || {}), [strategyName]: so } };
  if (ops.all || Object.keys(ops.roles).length) out._presetOps = ops;
  // The preset's engine image pins the docker command on this GPU, for the
  // variants it covers (exact variant × GPU image, which computeDockerMeta
  // already prefers over model- and brand-wide images).
  // The preset's single-node TP size becomes the exact-GPU entry of the
  // variant `tp` map, which resolveSingleNodeTp already prefers.
  const image = presetImage(p.image)?.vllm;
  const tp = strategyName === "single_node_tp" && Number.isInteger(p.tp) ? p.tp : null;
  if (image || tp) {
    out.variants = { ...(recipe.variants || {}) };
    for (const vk of raw.variants?.length ? raw.variants : Object.keys(out.variants)) {
      let v = out.variants[vk];
      if (!v) continue;
      if (image) v = { ...v, hardware_overrides: { ...(v.hardware_overrides || {}), [hwId]: { ...(v.hardware_overrides?.[hwId] || {}), docker_image: image } } };
      if (tp) v = { ...v, tp: mountKeyed(v.tp, hwId, tp) };
      out.variants[vk] = v;
    }
  }
  return out;
}

/** The builder choices preset `key` at `point` implies (merged with the point's). */
export function presetSelect(recipe, key, point) {
  const raw = key && key !== BASELINE ? recipe?.presets?.[key] : null;
  return raw ? effectivePreset(raw, point)?.select || null : null;
}

// ── Arg ops (synthesis) ─────────────────────────────────────────────────────

const normFlag = (f) => (typeof f === "string" && f.startsWith("--") ? `--${f.slice(2).replace(/_/g, "-")}` : f);
const takesValue = (next) => next !== undefined && !(typeof next === "string" && next.startsWith("-"));

// A KV connector's name (`MoRIIOConnector`, `SimpleCPUOffloadConnector`) as a
// key of a --kv-transfer-config edit addresses that connector: the value itself
// when it is that connector, else the matching child of a MultiConnector, so a
// lane can tune one connector without restating the composed list.
const CONNECTOR_KEY = /^[A-Z][A-Za-z0-9]*Connector[A-Za-z0-9]*$/;
function mergeEdit(base, edit, flag) {
  if (!isObj(edit)) return deepMerge(base, edit);
  let out = base;
  const rest = {};
  for (const [k, v] of Object.entries(edit)) {
    if (!CONNECTOR_KEY.test(k)) { rest[k] = v; continue; }
    if (out?.kv_connector === k) { out = deepMerge(out, v); continue; }
    const children = out?.kv_connector_extra_config?.connectors;
    const i = Array.isArray(children) ? children.findIndex((c) => c?.kv_connector === k) : -1;
    if (i < 0) throw new Error(`json_args: ${flag} has no ${k}`);
    const next = [...children];
    next[i] = deepMerge(children[i], v);
    out = { ...out, kv_connector_extra_config: { ...out.kv_connector_extra_config, connectors: next } };
  }
  return deepMerge(out, rest);
}

function applyJsonEdit(value, edit, flag) {
  let base = {};
  if (value !== undefined) {
    try {
      base = JSON.parse(value);
    } catch {
      base = undefined;
    }
    // Only a JSON object can take key edits (`128` or `"fp8"` parse as JSON too).
    if (!isObj(base)) throw new Error(`json_args: ${flag} value is not a JSON object: ${value}`);
  }
  const prune = (o) => {
    if (Array.isArray(o)) return o.map(prune);
    if (!isObj(o)) return o;
    const out = {};
    for (const [k, v] of Object.entries(o)) if (v !== null) out[k] = prune(v);
    return out;
  };
  return JSON.stringify(prune(mergeEdit(base, edit, flag)));
}

/**
 * Flags a toggle owns: every flag a recipe feature (args, modes, hardware
 * overrides) or a KV Offload option emits. Whether such a flag is on the
 * command is the toggle's call, so a preset only edits it (`select` is how a
 * preset turns the toggle on).
 */
export function toggleOwnedFlags(recipe, taxonomy) {
  const flags = new Set();
  const add = (list) => (list || []).forEach((t) => typeof t === "string" && t.startsWith("--") && flags.add(normFlag(t)));
  const addHo = (ho) => Object.values(ho || {}).forEach((o) => add(o?.args));
  for (const f of Object.values(recipe?.features || {})) {
    add(f?.args);
    addHo(f?.hardware_overrides);
    for (const m of Object.values(f?.modes || {})) { add(m?.args); addHo(m?.hardware_overrides); }
  }
  for (const o of Object.values(taxonomy?.kv_offload || {})) add(o?.args);
  return flags;
}

/**
 * Apply a preset's `remove_args` and `json_args` to a raw arg list. Flags match
 * with `_` and `-` treated alike. `json_args` edits the LAST occurrence of the
 * flag (the one last-wins dedupe keeps); a flag that isn't there yet is added
 * with the edit as its value, unless it is in `editOnly` (toggleOwnedFlags):
 * spec decoding switched off must not leave a `--speculative-config` holding
 * only the preset's tweak.
 */
export function applyPresetArgOps(args, ops, editOnly) {
  if (!ops) return args;
  let out = args;
  if (ops.remove_args?.length) {
    const drop = new Set(ops.remove_args.map(normFlag));
    const kept = [];
    for (let i = 0; i < out.length; i++) {
      if (!drop.has(normFlag(out[i]))) { kept.push(out[i]); continue; }
      if (takesValue(out[i + 1])) i++;
    }
    out = kept;
  }
  for (const [flag, edit] of Object.entries(ops.json_args || {})) {
    const want = normFlag(flag);
    let at = -1;
    for (let i = 0; i < out.length; i++) if (normFlag(out[i]) === want && takesValue(out[i + 1])) at = i;
    if (at >= 0) {
      out = [...out];
      out[at + 1] = applyJsonEdit(out[at + 1], edit, flag);
    } else if (!editOnly?.has(want)) {
      out = [...out, flag, applyJsonEdit(undefined, edit, flag)];
    }
  }
  return out;
}

// ── Validation ──────────────────────────────────────────────────────────────

/** Build-time checks the JSON Schema can't express. Returns error strings. */
// `pdConnectors` is pd_cluster.yaml's `connectors` map (select.pd_connector).
export function validatePresets(recipe, taxonomy, strategyIds, kvStoreIds = [], pdConnectors = {}) {
  const errors = [];
  const seenDefault = new Map();
  const frontends = Object.keys(taxonomy?.frontends || {});
  for (const [key, p] of Object.entries(recipe?.presets || {})) {
    const where = `presets.${key}`;
    if (key === BASELINE) errors.push(`${where}: "${BASELINE}" is reserved`);
    if (!taxonomy?.hardware_profiles?.[p.hardware]) errors.push(`${where}.hardware: unknown hardware profile "${p.hardware}"`);
    if (p.strategy && !strategyIds.includes(p.strategy)) {
      errors.push(`${where}.strategy: "${p.strategy}" is not offered by this recipe`);
    }
    for (const v of p.variants || []) {
      if (!recipe.variants?.[v]) errors.push(`${where}.variants: no variant "${v}"`);
    }
    const isPd = p.strategy === "pd_cluster";
    if (p.roles && !isPd) errors.push(`${where}.roles: only a pd_cluster preset has roles`);
    if (p.default) {
      const slot = `${p.hardware}|${p.strategy || "*"}|${(p.variants || []).join(",") || "*"}`;
      if (seenDefault.has(slot)) errors.push(`${where}: ${seenDefault.get(slot)} is already the default for this hardware + strategy`);
      else seenDefault.set(slot, key);
    }
    const checkDelta = (d, at) => {
      for (const f of d?.remove_args || []) {
        if (!String(f).startsWith("--")) errors.push(`${at}.remove_args: "${f}" is not a --flag`);
        if ((d.args || []).map(normFlag).includes(normFlag(f))) errors.push(`${at}: ${f} is both added and removed`);
      }
      for (const [f, edit] of Object.entries(d?.json_args || {})) {
        if (!f.startsWith("--")) errors.push(`${at}.json_args: "${f}" is not a --flag`);
        if (!isObj(edit)) errors.push(`${at}.json_args.${f}: must be an object of keys to set (null deletes)`);
      }
    };
    const checkSelect = (s, at) => {
      if (!s) return;
      if (s.pd_router !== undefined && !isPd) errors.push(`${at}.pd_router: only a pd_cluster preset picks a PD router`);
      if (s.pd_connector !== undefined) {
        const c = pdConnectors[s.pd_connector];
        const brand = taxonomy?.hardware_profiles?.[p.hardware]?.brand;
        if (!isPd) errors.push(`${at}.pd_connector: only a pd_cluster preset picks a KV transfer connector`);
        else if (!c) errors.push(`${at}.pd_connector: unknown connector "${s.pd_connector}"`);
        else if (c.brands?.length && !c.brands.includes(brand)) errors.push(`${at}.pd_connector: ${s.pd_connector} doesn't run on ${p.hardware} (${c.brands.join(" / ")} only)`);
        else if (s.pd_router && !(c.routers || ["vllm"]).includes(s.pd_router)) errors.push(`${at}.pd_connector: ${s.pd_connector} doesn't work behind pd_router ${s.pd_router}`);
      }
      if (s.frontend !== undefined) {
        if (!frontends.includes(s.frontend)) errors.push(`${at}.frontend: unknown frontend "${s.frontend}"`);
        else if (recipe.model?.[`${s.frontend}_frontend`] === "unsupported") errors.push(`${at}.frontend: ${s.frontend} is unsupported by this recipe`);
      }
      for (const [f, v] of Object.entries(s.features || {})) {
        const feat = recipe.features?.[f];
        if (!feat) errors.push(`${at}.features: no feature "${f}"`);
        else if (typeof v === "string" && !feat.modes?.[v]) errors.push(`${at}.features.${f}: no mode "${v}"`);
      }
      if (s.kv_offload !== undefined && s.kv_offload !== false && s.kv_offload !== "off"
          && !taxonomy?.kv_offload?.[s.kv_offload] && !kvStoreIds.includes(s.kv_offload)) {
        errors.push(`${at}.kv_offload: unknown option "${s.kv_offload}"`);
      }
    };
    const gpus = taxonomy?.hardware_profiles?.[p.hardware]?.gpu_count;
    const checkTopology = (d, at) => {
      if (d?.tp !== undefined) {
        if (p.strategy !== "single_node_tp") errors.push(`${at}.tp: only a single_node_tp preset sets tp`);
        else if (gpus && d.tp > gpus) errors.push(`${at}.tp: ${d.tp} exceeds ${p.hardware}'s ${gpus} GPUs`);
      }
      if (d?.select?.nodes !== undefined && !String(p.strategy || "").startsWith("multi_node_")) {
        errors.push(`${at}.select.nodes: only a multi_node_* preset sets the node count`);
      }
    };
    checkDelta(p, where);
    checkTopology(p, where);
    checkSelect(p.select, `${where}.select`);
    for (const [role, d] of Object.entries(p.roles || {})) checkDelta(d, `${where}.roles.${role}`);
    const points = presetPoints(p);
    if (points.filter((pt) => pt.default).length > 1) errors.push(`${where}.points: more than one default point`);
    for (const pt of points) {
      const at = `${where}.points.${pt.key}`;
      if (pt.roles && !isPd) errors.push(`${at}.roles: only a pd_cluster preset has roles`);
      checkDelta(pt, at);
      checkTopology(pt, at);
      checkSelect(pt.select, `${at}.select`);
      for (const [role, d] of Object.entries(pt.roles || {})) checkDelta(d, `${at}.roles.${role}`);
    }
  }
  return errors;
}
