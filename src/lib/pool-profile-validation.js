// Shared by strict source assembly and the API build. Check the references in
// the combined layout, since a backend cannot know a model's placement IDs.
import { identifier, inputError, inputTypes, fieldReference } from "./pool-inputs.js";

export const stages = ["storage", "container", "config", "control", "serve", "verify"];
const modes = ["new", "reuse"];
const roles = ["prefill", "decode", "master", "meta", "proxy"];
const targets = ["hosts", "containers", "control", "prefill", "decode", "proxy"];
const record = (value) => value !== null && typeof value === "object" && !Array.isArray(value);

export function placementsFor(target, placements, mode) {
  return placements.filter((p) => p && (!mode || !p.mode || p.mode === mode) && (
    target === "control" ? p.role === "master" || p.role === "meta"
      : target.startsWith("placement:") ? (p.id ?? p.role) === target.slice(10) : p.role === target
  ));
}

function fields(spec, errors) {
  for (const [key, value] of Object.entries(spec || {})) {
    if (!identifier.test(key) || !value?.label || !inputTypes.includes(value.type) || !["host", "cluster"].includes(value.scope))
      errors.push("Invalid input: " + key);
    else if (value.default != null && inputError(value.type, value.default))
      errors.push("Invalid default for " + key + ": " + value.default);
    if (value?.mode && !modes.includes(value.mode)) errors.push("Unknown mode for " + key);
  }
}

function validateSteps(steps, errors) {
  if (!Array.isArray(steps)) { errors.push("steps must be an array"); return; }
  const ids = new Set();
  for (const s of steps) {
    if (!s || !identifier.test(s.id) || !s.title || !stages.includes(s.stage) || typeof s.check !== "string" || !s.check)
      errors.push("Invalid step " + (s?.id ?? "(missing ID)"));
    if (!s) continue;
    if (ids.has(s.id)) errors.push("Duplicate step " + s.id);
    ids.add(s.id);
    if (!targets.includes(s.target) && !/^(host|placement):[\w-]+$/.test(s.target)) errors.push("Invalid target for " + s.id);
    if (s.mode && !modes.includes(s.mode)) errors.push("Unknown mode for " + s.id);
    if (s.action && !["container", "launch"].includes(s.action)) errors.push("Unknown action for " + s.id);
    if (s.action && s.command) errors.push(s.id + " must select an action or a command");
    if (s.action === "container" && s.target !== "containers") errors.push(s.id + ": container actions must target containers");
    if (s.action === "launch" && (s.target === "hosts" || s.target === "containers" || s.target?.startsWith("host:")))
      errors.push(s.id + ": launch actions must target placements with launch.argv");
    if (s.depends_on && !Array.isArray(s.depends_on)) errors.push("Invalid dependencies for " + s.id);
  }
}

function validateBindings(owner, errors) {
  for (const kind of ["mounts", "devices"]) {
    const bindings = owner[kind] ?? [];
    if (!Array.isArray(bindings) || bindings.some((b) =>
      !record(b) || typeof b.source !== "string" || (kind === "mounts" && typeof b.target !== "string"))) {
      errors.push("Invalid " + kind + " mappings");
    }
  }
}

function validateBudgets(capacity, specs, errors) {
  if (capacity == null) return;
  if (!Array.isArray(capacity)) { errors.push("capacity must be an array"); return; }
  for (const b of capacity) {
    if (!record(b)) { errors.push("Expected a capacity entry"); continue; }
    if (b.input != null && specs?.[b.input]?.type !== "integer") errors.push(b.label + ": capacity.input must name an integer input: " + b.input);
    const count = specs?.[b.count];
    if (b.count !== "workers" && (count?.scope !== "host" || count.type !== "integer"))
      errors.push(b.label + ": capacity.count must be workers or a per-host integer input: " + b.count);
  }
}

export function validatePoolProfile(profile) {
  if (!record(profile)) return ["Expected a pool profile object"];
  const errors = [];
  if (profile.schema_version !== 1) errors.push("Unsupported pool profile schema_version");
  if (!profile.label || !identifier.test(profile.backend || "")) errors.push("Expected label and backend");
  fields(profile.inputs, errors);
  validateBudgets(profile.capacity, profile.inputs, errors);
  validateBindings(profile, errors);
  for (const role of ["prefill", "decode"])
    if (!record(profile.store?.[role]) || profile.store[role].backend !== profile.backend) errors.push("Missing store." + role + " for " + profile.backend);
  validateSteps(profile.steps, errors);
  return errors;
}

// Stable stage/dependency order; independent peers keep their authored order.
export function orderSteps(steps) {
  const byId = new Map(steps.map((s) => [s.id, s]));
  const seen = new Set(), active = new Set(), ordered = [];
  function visit(s) {
    if (active.has(s.id)) throw new Error("Step dependency cycle at " + s.id);
    if (seen.has(s.id)) return;
    active.add(s.id);
    for (const dep of s.depends_on || []) {
      const before = byId.get(dep);
      if (!before) throw new Error(s.id + ": unknown prerequisite " + dep);
      if (stages.indexOf(before.stage) > stages.indexOf(s.stage)) throw new Error(s.id + ": prerequisite in a later stage: " + dep);
      if (s.mode && before.mode && before.mode !== s.mode) throw new Error(s.id + ": prerequisite " + dep + " is unavailable in this mode");
      visit(before);
    }
    active.delete(s.id);
    seen.add(s.id);
    ordered.push(s);
  }
  for (const stage of stages) for (const step of steps) if (step.stage === stage) visit(step);
  return ordered;
}

export function validatePoolDeployment(profile, d, hardware) {
  if (!record(d)) return ["Expected a deployment layout"];
  const errors = [];
  if (!Array.isArray(d.hosts) || !d.hosts.length || !record(d.containers) || !Array.isArray(d.placements))
    return ["Expected hosts, containers and placements"];
  const hosts = new Set(d.hosts);
  if (hosts.size !== d.hosts.length || d.hosts.some((h) => typeof h !== "string" || !identifier.test(h))) errors.push("Invalid or duplicate host ID");
  const containers = Object.entries(d.containers);
  if (!containers.length) errors.push("Expected at least one container");
  for (const [id, c] of containers) {
    if (!identifier.test(id) || !hosts.has(c?.host)) errors.push("Invalid container or unknown host: " + id);
  }
  fields(d.inputs, errors);
  validateBindings(d, errors);
  const specs = { ...profile.inputs, ...d.inputs };
  const constants = d.constants || {};
  for (const [key, value] of Object.entries(constants)) {
    if (!identifier.test(key) || Object.hasOwn(specs, key) || ["host_id", "container_id", "pool_device_paths"].includes(key))
      errors.push("Invalid or conflicting deployment constant " + key);
    if (!["string", "number", "boolean"].includes(typeof value) || inputError("text", value) || String(value).includes("{{"))
      errors.push("Expected a single-line literal for deployment constant " + key);
  }
  if (d.inputs) validateBudgets(profile.capacity, specs, errors);
  for (const [key, value] of Object.entries(d.defaults || {})) {
    const parts = key.split(".");
    const spec = specs[parts.length === 1 ? key : parts[2]];
    if (!spec || (parts.length > 1 && (parts.length !== 3 || parts[0] !== "hosts" || !hosts.has(parts[1]) || spec.scope !== "host")))
      errors.push("Unknown default field " + key);
    else if (inputError(spec.type, value)) errors.push("Invalid deployment default " + key);
  }
  const ids = new Set(), used = new Map();
  // Check each reference against the context that will actually interpolate it.
  // An unqualified input resolves on the current host; a host-qualified input
  // may also point to another host. Placement-local variables have no fallback.
  function checkRefs(value, where, placement = null, hasContainer = true, mode = placement?.mode) {
    for (const [, ref] of (JSON.stringify(value) || "").matchAll(fieldReference)) {
      if (Object.hasOwn(constants, ref)) continue;
      if (ref === "host_id" || ref === "pool_device_paths" || (ref === "container_id" && hasContainer)) continue;
      if (ref.startsWith("placement.")) {
        const key = ref.slice("placement.".length);
        if (!placement || !(key === "id" || key === "rank" || Object.hasOwn(placement.vars || {}, key)))
          errors.push(where + ": unknown placement variable " + ref);
        continue;
      }
      const parts = ref.split(".");
      let spec;
      if (parts.length === 1 && Object.hasOwn(specs, ref)) spec = specs[ref];
      if (parts.length === 3 && parts[0] === "hosts" && hosts.has(parts[1]) && specs[parts[2]]?.scope === "host")
        spec = specs[parts[2]];
      if (!spec) errors.push(where + ": Unknown input reference: " + ref);
      else if (spec.mode && mode !== spec.mode) errors.push(where + ": " + ref + " is only available in " + spec.mode + " mode");
    }
  }
  checkRefs([d.containers, profile.mounts, profile.devices, d.mounts, d.devices], "containers");
  for (const p of d.placements) {
    if (!record(p)) { errors.push("Invalid placement"); continue; }
    const id = p.id ?? p.role, c = d.containers[p.container];
    if (!roles.includes(p.role) || !identifier.test(id) || ids.has(id) || !c) errors.push("Invalid or duplicate placement: " + id);
    ids.add(id);
    if (p.mode && !modes.includes(p.mode)) errors.push("Invalid placement mode: " + id);
    for (const [key, value] of Object.entries(p.vars || {})) {
      if (typeof value === "string" && value.match(fieldReference))
        errors.push(id + ": placement." + key + " must be literal; reference inputs directly in the launch or step");
    }
    if (["prefill", "decode", "proxy"].includes(p.role)) {
      if (!Array.isArray(p.launch?.argv) || !p.launch.argv.length) errors.push("Missing launch.argv for " + id);
      if (p.role !== "proxy") {
        if (!(p.launch?.transfer ?? p.transfer)?.kv_connector) errors.push("Missing transfer connector for " + id);
        if (p.launch?.argv?.some((a) => /^--kv[-_]transfer[-_]config(?:=|$)/.test(a))) errors.push("Duplicate --kv-transfer-config in " + id);
        const store = p.launch?.store ?? p.store;
        if (store?.backend !== undefined && store.backend !== profile.backend)
          errors.push(id + ": store backend must match pool profile " + profile.backend);
        checkRefs(profile.store?.[p.role], id, p);
      }
      if (p.devices && !Array.isArray(p.devices)) errors.push("Invalid device list for " + id);
      for (const device of Array.isArray(p.devices) ? p.devices : []) {
        const key = c?.host + ":" + device;
        if (!Number.isInteger(device) || device < 0 || device >= (hardware?.gpu_count ?? Infinity) || used.has(key))
          errors.push("Out-of-range or overlapping device " + key + " in " + id);
        used.set(key, id);
      }
    }
    checkRefs([p.launch, p.transfer, p.store], id, p);
  }
  for (const mode of modes) {
    for (const role of ["prefill", "decode", "proxy"]) {
      const active = placementsFor(role, d.placements, mode);
      if (!active.length) errors.push("Missing " + role + " placement in " + mode + " mode");
      if (role === "proxy" && active.length > 1) errors.push("Expected one proxy placement in " + mode + " mode");
    }
  }
  const steps = [...(profile.steps || []), ...(d.steps || [])];
  validateSteps(steps, errors);
  for (const s of steps) {
    if (!s || typeof s.target !== "string") continue;
    if (s.target?.startsWith("placement:") && !ids.has(s.target.slice(10))) errors.push("Unknown placement target in " + s.id);
    if (s.target?.startsWith("host:") && !hosts.has(s.target.slice(5))) errors.push("Unknown host target in " + s.id);
    const names = [s.command, s.check];
    if (s.target === "hosts" || s.target.startsWith("host:") || s.target === "containers") {
      checkRefs(names, s.id, null, s.target === "containers", s.mode);
      continue;
    }
    const placements = placementsFor(s.target, d.placements, s.mode);
    if (!placements.length) errors.push(s.id + ": no active placement for " + s.target + (s.mode ? " in " + s.mode + " mode" : ""));
    for (const p of placements) {
      if (s.action === "launch" && !p.launch?.argv?.length) errors.push(s.id + ": missing launch.argv on " + (p.id ?? p.role));
      checkRefs(names, s.id + "/" + (p.id ?? p.role), p, true, s.mode ?? p.mode);
    }
  }
  try { orderSteps(steps); } catch (e) { errors.push(e.message); }
  return [...new Set(errors)];
}
