// Tuned profiles: named, evidence-backed configurations for one hardware
// profile (and optionally one serving strategy), e.g. the InferenceX AgentX
// lane a vendor team qualified on GB200. A profile is a delta on top of the
// recipe's baseline command, so it never has to restate shared flags:
//
//   profiles:
//     gb200_agentx_pd:
//       label: "AgentX 1P1D"
//       hardware: gb200            # exact profile id (required)
//       strategy: pd_cluster       # optional; absent = every strategy
//       workload: "Agentic"        # optional free-text tag
//       default: true              # applied unless the user picks Baseline
//       args: [...]                # every node / both PD roles
//       env: {...}
//       roles:                     # pd_cluster only
//         decode: { args, env, nodes: 4, parallelism: tp }
//       source: { label, href }
//       runbook: |                 # markdown rendered under "Tuned profiles"
//
// The active profile is mounted into the exact-GPU strategy and role layers
// (`strategy_overrides.<s>.hardware_overrides.<hw>` and
// `strategy_overrides.<s>.<role>.hardware_overrides.<hw>`), which synthesis
// already applies last and additively; a role's `nodes` / `parallelism` become
// the `<hw>` entry of that role's hardware-keyed map. A recipe that moves such
// blocks into a `default: true` profile therefore renders exactly as before,
// and Baseline drops the lane's topology together with its flags.

export const BASELINE = "baseline";

/** Profiles that apply to this hardware + strategy, strategy-specific first. */
export function matchingProfiles(recipe, hwId, strategyName) {
  const all = Object.entries(recipe?.profiles || {})
    .filter(([, p]) => p?.hardware === hwId && (!p.strategy || p.strategy === strategyName));
  return [
    ...all.filter(([, p]) => p.strategy),
    ...all.filter(([, p]) => !p.strategy),
  ].map(([key, p]) => ({ key, ...p }));
}

/**
 * The profile key in effect: an explicit matching pick, BASELINE when the user
 * opted out, else the matching default (strategy-specific before
 * hardware-wide), else BASELINE.
 */
export function resolveProfileKey(recipe, hwId, strategyName, pick) {
  const options = matchingProfiles(recipe, hwId, strategyName);
  if (pick === BASELINE) return BASELINE;
  if (pick && options.some((p) => p.key === pick)) return pick;
  return options.find((p) => p.default)?.key ?? BASELINE;
}

const concat = (a, b) => (b?.length ? [...(a || []), ...b] : a);
const assign = (a, b) => (b && Object.keys(b).length ? { ...(a || {}), ...b } : a);

// Add an exact-GPU entry to a value that may be a scalar or a hardware-keyed map.
function mountKeyed(raw, hwId, value) {
  if (value === undefined) return raw;
  const map = raw && typeof raw === "object" ? { ...raw } : raw === undefined ? {} : { default: raw };
  map[hwId] = value;
  return map;
}

function mountLayer(block, delta) {
  const next = { ...(block || {}) };
  const args = concat(next.extra_args, delta.args);
  const env = assign(next.extra_env, delta.env);
  if (args) next.extra_args = args;
  if (env) next.extra_env = env;
  return next;
}

/**
 * The recipe with profile `key` mounted for `strategyName`. Returns the recipe
 * unchanged for BASELINE or a profile that doesn't match.
 */
export function withProfile(recipe, key, hwId, strategyName) {
  const p = key && key !== BASELINE ? recipe?.profiles?.[key] : null;
  if (!p || p.hardware !== hwId || (p.strategy && p.strategy !== strategyName)) return recipe;
  const so = { ...(recipe.strategy_overrides?.[strategyName] || {}) };
  so.hardware_overrides = { ...(so.hardware_overrides || {}) };
  so.hardware_overrides[hwId] = mountLayer(so.hardware_overrides[hwId], p);
  for (const [role, delta] of Object.entries(p.roles || {})) {
    const r = { ...(so[role] || {}) };
    r.hardware_overrides = { ...(r.hardware_overrides || {}) };
    r.hardware_overrides[hwId] = mountLayer(r.hardware_overrides[hwId], delta);
    for (const k of ["nodes", "parallelism"]) {
      const v = mountKeyed(r[k], hwId, delta[k]);
      if (v !== undefined) r[k] = v;
    }
    so[role] = r;
  }
  return { ...recipe, strategy_overrides: { ...(recipe.strategy_overrides || {}), [strategyName]: so } };
}

/** Build-time checks the JSON Schema can't express. Returns error strings. */
export function validateProfiles(recipe, taxonomy, strategyIds) {
  const errors = [];
  const seenDefault = new Map();
  for (const [key, p] of Object.entries(recipe?.profiles || {})) {
    const where = `profiles.${key}`;
    if (key === BASELINE) errors.push(`${where}: "${BASELINE}" is reserved`);
    if (!taxonomy?.hardware_profiles?.[p.hardware]) errors.push(`${where}.hardware: unknown hardware profile "${p.hardware}"`);
    if (p.strategy && !strategyIds.includes(p.strategy)) {
      errors.push(`${where}.strategy: "${p.strategy}" is not offered by this recipe`);
    }
    if (p.roles && p.strategy !== "pd_cluster") errors.push(`${where}.roles: only a pd_cluster profile has roles`);
    if (p.default) {
      const slot = `${p.hardware}|${p.strategy || "*"}`;
      if (seenDefault.has(slot)) errors.push(`${where}: ${seenDefault.get(slot)} is already the default for ${slot.replace("|*", "")}`);
      else seenDefault.set(slot, key);
    }
  }
  return errors;
}
