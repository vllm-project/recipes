// Overlay fragments: a source with `role: overlay` (e.g. vllm-ascend) ships
// `overlays/<org>/<repo>.yaml` files that merge into the recipe another
// source owns. The merge is ADD-ONLY and SCOPED: an overlay may only add keys
// for hardware its scope grants (sources.yaml `scope.brands`), so vllm-ascend
// can attach Ascend profiles, variants and overrides to a core recipe but can
// never change what the NVIDIA/AMD pills render.
//
// `applyOverlay` is the merge; `splitOverlay` is its inverse (pull every
// in-scope key out of a full recipe) — the migration tool for moving content
// into an overlay repo, and the round-trip check that keeps the two in sync.

// ── Scope ──

// Resolve a `scope` from sources.yaml against the taxonomy into the sets of
// hardware keys an overlay may write: exact profile ids, generations used only
// by in-scope profiles (`npu`), lowercase brand keys, and `arch:` selectors
// whose every profile is in scope.
export function resolveScope(scope, taxonomy) {
  const brands = new Set((scope?.brands || []).map((b) => b.toLowerCase()));
  const profiles = taxonomy?.hardware_profiles || {};
  const ids = new Set();
  const genIn = new Set();
  const genOut = new Set();
  const archIn = new Set();
  const archOut = new Set();
  for (const [id, p] of Object.entries(profiles)) {
    const inScope = brands.has(String(p.brand || "").toLowerCase());
    if (inScope) ids.add(id);
    if (p.generation) (inScope ? genIn : genOut).add(p.generation);
    if (p.compute_arch) (inScope ? archIn : archOut).add(p.compute_arch);
  }
  const generations = new Set([...genIn].filter((g) => !genOut.has(g)));
  const archs = new Set([...archIn].filter((a) => !archOut.has(a)));
  const isHwKey = (key) =>
    ids.has(key) || generations.has(key) || brands.has(String(key).toLowerCase())
    || (String(key).startsWith("arch:") && archs.has(String(key).slice(5)));
  return { brands, ids, generations, isHwKey, empty: brands.size === 0 };
}

function brandInScope(brand, ctx) {
  const list = Array.isArray(brand) ? brand : [brand];
  return list.length > 0 && list.every((b) => b && ctx.brands.has(String(b).toLowerCase()));
}

// A variant belongs to the overlay when it can only ever run on in-scope
// hardware (e.g. Kimi-K3's `ascend_w4a8`, `supported_hardware: [ascend_910c]`).
export function variantInScope(variant, ctx) {
  const hw = variant?.supported_hardware;
  return Array.isArray(hw) && hw.length > 0 && hw.every((k) => ctx.isHwKey(k));
}

// A tuned preset belongs to the overlay when its hardware is in scope.
export function presetInScope(preset, ctx) {
  return typeof preset?.hardware === "string" && ctx.isHwKey(preset.hardware);
}

// ── Rule table ──
//
// Path segments: a literal key, "*" (any key), "@hw" (an in-scope hardware
// key — the whole subtree below it belongs to the overlay), "@variant" (a
// variant absent from the core recipe that passes variantInScope), "@preset"
// (likewise, a preset that passes presetInScope), "[task]"
// (an omni task in the core `omni.tasks` array, addressed by `key ?? id`).
// `union: true` = an array the overlay may append in-scope hardware ids to.
// When the schema gains a new hardware-keyed field, add it here.
export const OVERLAY_RULES = [
  ["meta", "hardware", "@hw"],
  ["hardware_overrides", "@hw"],
  ["hardware_notes", "@hw"],
  ["hardware_opt_in_features", "@hw"],
  ["strategy_hardware", "*", "@hw"],
  ["strategy_min_gpus", "@hw"],
  ["kv_cache_strategy_hardware", "*", "@hw"],
  ["strategy_overrides", "*", "hardware_overrides", "@hw"],
  ["features", "*", "hardware_overrides", "@hw"],
  ["features", "*", "modes", "*", "hardware_overrides", "@hw"],
  ["features", "*", "modes", "*", "hardware", "@hw"],
  ["variants", "@variant"],
  ["presets", "@preset"],
  ["variants", "*", "hardware_overrides", "@hw"],
  ["variants", "*", "tp", "@hw"],
  { path: ["variants", "*", "supported_hardware"], union: true },
  ["omni", "tasks", "[task]", "hardware_overrides", "@hw"],
].map((r) => (Array.isArray(r) ? { path: r } : r));

const isObj = (v) => v !== null && typeof v === "object" && !Array.isArray(v) && !(v instanceof Date);
// Keys that would reach Object.prototype through a merge; never valid in a recipe.
const RESERVED = new Set(["__proto__", "constructor", "prototype"]);
const own = (o, k) => isObj(o) && Object.hasOwn(o, k);
const taskKey = (t) => (typeof t === "string" ? t : t?.key ?? t?.id);

function segmentMatches(seg, key, coreParent, ctx) {
  if (seg === "*") return true;
  if (seg === "@hw") return ctx.isHwKey(key);
  if (seg === "@variant" || seg === "@preset") return !own(coreParent, key);
  if (seg === "[task]") return true;
  return seg === key;
}

// Child of a core node, resolving "[task]" against the omni tasks array.
function coreChild(node, key, seg) {
  if (seg === "[task]") return Array.isArray(node) ? node.find((t) => taskKey(t) === key) : undefined;
  return own(node, key) ? node[key] : undefined;
}

// ── Merge ──

const GUIDE_MARKER = /^[ \t]*<!--\s*overlay:([\w.-]+)(?::([\w.-]+))?\s*-->[ \t]*\n?/gm;

// Merge `overlay` into a deep copy of `core`. Returns { recipe, errors };
// callers fail the build on any error.
export function applyOverlay(core, overlay, { ctx, source }) {
  const recipe = structuredClone(core);
  const errors = [];
  const err = (p, msg) => errors.push(`${p.join(".") || "<root>"}: ${msg}`);
  if (ctx.empty) {
    errors.push(`source "${source}" has no scope.brands in sources.yaml — overlays need a granted scope`);
    return { recipe, errors };
  }

  for (const [key, value] of Object.entries(overlay || {})) {
    if (RESERVED.has(key)) { err([key], "reserved key"); continue; }
    if (key === "dependencies") {
      if (!Array.isArray(value)) { err([key], "must be a list"); continue; }
      // Platform installs (vllm-ascend builds) must precede the model's own
      // deps, so entries go first unless marked `position: end`.
      const start = [];
      const end = [];
      for (const [i, dep] of value.entries()) {
        if (!isObj(dep)) { err([key, i], "must be a map"); continue; }
        if (!brandInScope(dep.brand, ctx)) err([key, i], `needs a brand in scope (${[...ctx.brands].join(", ")})`);
        const { position, ...rest } = structuredClone(dep);
        (position === "end" ? end : start).push(rest);
      }
      recipe.dependencies = [...start, ...(recipe.dependencies || []), ...end];
    } else if (key === "guide_sections") {
      if (!isObj(value)) { err([key], "must be a map of section name → markdown"); continue; }
      recipe.guide = insertGuideSections(recipe.guide || "", source, value);
    } else if (key === "meta" && isObj(value) && "date_updated" in value) {
      const { date_updated, ...rest } = value;
      const iso = (d) => (d instanceof Date ? d.toISOString().slice(0, 10) : String(d));
      if (iso(date_updated) > iso(recipe.meta?.date_updated ?? "")) recipe.meta.date_updated = date_updated;
      mergeNode(recipe, "meta", rest, [key], OVERLAY_RULES, ctx, err);
    } else {
      mergeNode(recipe, key, value, [key], OVERLAY_RULES, ctx, err);
    }
  }
  return { recipe, errors };
}

// Merge overlay value `ov` into `parent[key]`, where `rules` are the rules
// whose prefix matched so far and `p` is the path of `key`.
function mergeNode(parent, key, ov, p, rules, ctx, err, seg = null) {
  const depth = p.length - 1;
  if (RESERVED.has(key)) return err(p, "reserved key");
  const coreVal = seg === "[task]" ? coreChild(parent, key, seg) : own(parent, key) ? parent[key] : undefined;
  const live = rules.filter((r) => r.path.length > depth && segmentMatches(r.path[depth], key, parent, ctx));
  if (live.length === 0) return err(p, "not writable by an overlay (outside its hardware scope)");

  // A rule ending here: the overlay owns this key (or, for `union`, may extend
  // the array with in-scope ids).
  const done = live.find((r) => r.path.length === depth + 1);
  if (done) {
    if (done.union) {
      if (!Array.isArray(ov) || !ov.every((k) => ctx.isHwKey(k))) return err(p, "may only append in-scope hardware ids");
      if (coreVal !== undefined && !Array.isArray(coreVal)) return err(p, "core value is not a list");
      // An absent list means "no allowlist" — creating one would narrow it.
      if (coreVal === undefined) return err(p, "core variant has no supported_hardware list to extend");
      parent[key] = [...coreVal, ...ov.filter((k) => !coreVal.includes(k))];
      return;
    }
    if (done.path[depth] === "@variant" && !variantInScope(ov, ctx)) {
      return err(p, "new variant must list only in-scope hardware in supported_hardware");
    }
    if (done.path[depth] === "@preset" && !presetInScope(ov, ctx)) {
      return err(p, "new preset must target in-scope hardware");
    }
    if (coreVal !== undefined) return err(p, "already set by the core recipe (overlays are add-only)");
    setChild(parent, key, structuredClone(ov), seg);
    return;
  }

  // Intermediate node: recurse, creating the container if the core lacks it.
  if (!isObj(ov)) return err(p, "expected a map");
  let node = coreVal;
  if (node === undefined) {
    // "[task]" targets must already exist in the core omni.tasks list.
    if (seg === "[task]") return err(p, `no omni task "${key}" in the core recipe`);
    node = {};
    setChild(parent, key, node, seg);
  } else if (seg === "[task]" && typeof node === "string") {
    // Promote a bare task id to its object form so it can carry overrides.
    node = { id: node };
    setChild(parent, key, node, seg);
  } else if (!isObj(node) && !(Array.isArray(node) && live.some((r) => r.path[depth + 1] === "[task]"))) {
    return err(p, "core value is not a map");
  }
  const nextRules = live;
  for (const [k, v] of Object.entries(ov)) {
    const nextSeg = nextRules.some((r) => r.path[depth + 1] === "[task]") ? "[task]" : null;
    mergeNode(node, k, v, [...p, k], nextRules, ctx, err, nextSeg);
  }
}

function setChild(parent, key, value, seg) {
  if (seg === "[task]") {
    const i = parent.findIndex((t) => taskKey(t) === key);
    parent[i] = value;
  } else {
    parent[key] = value;
  }
}

// Insert each named section at `<!-- overlay:<source>:<name> -->` (or the
// bare `<!-- overlay:<source> -->` for a section named "default"); sections
// without a marker append to the end of the guide.
function insertGuideSections(guide, source, sections) {
  const placed = new Set();
  let out = guide.replace(GUIDE_MARKER, (m, src, name = "default") => {
    if (src !== source || !Object.hasOwn(sections, name)) return m;
    placed.add(name);
    return ensureTrailingNewline(sections[name]);
  });
  for (const [name, md] of Object.entries(sections)) {
    if (!placed.has(name)) out = `${ensureTrailingNewline(out)}\n${ensureTrailingNewline(md)}`;
  }
  return out;
}

const ensureTrailingNewline = (s) => (String(s).endsWith("\n") ? String(s) : `${s}\n`);

export function hasGuideMarkers(text) {
  GUIDE_MARKER.lastIndex = 0;
  return GUIDE_MARKER.test(text);
}

// Drop markers left unfilled (no overlay in this channel supplied them).
export function stripGuideMarkers(guide) {
  return typeof guide === "string" ? guide.replace(GUIDE_MARKER, "") : guide;
}

// ── Split (inverse of applyOverlay) ──

// Pull every in-scope key out of `recipe` → { core, overlay }. Guide sections
// whose heading matches `guideHeading` (a RegExp) move into
// `guide_sections`, leaving a marker in the core guide.
export function splitOverlay(recipe, { ctx, source, guideHeading = null }) {
  const core = structuredClone(recipe);
  const overlay = {};
  const put = (p, value) => {
    let node = overlay;
    for (const k of p.slice(0, -1)) node = node[k] ??= {};
    node[p[p.length - 1]] = value;
  };

  // New variants first, so the per-variant rules below skip them.
  const movedVariants = new Set();
  for (const [vk, v] of Object.entries(core.variants || {})) {
    if (vk !== "default" && variantInScope(v, ctx)) {
      put(["variants", vk], v);
      delete core.variants[vk];
      movedVariants.add(vk);
    }
  }

  for (const [pk, prof] of Object.entries(core.presets || {})) {
    if (!presetInScope(prof, ctx)) continue;
    put(["presets", pk], prof);
    delete core.presets[pk];
  }
  if (core.presets && !Object.keys(core.presets).length) delete core.presets;

  for (const rule of OVERLAY_RULES) {
    if (rule.path.includes("@variant") || rule.path.includes("@preset")) continue;
    extract(core, rule, 0, [], ctx, put);
  }

  if (Array.isArray(core.dependencies)) {
    const keep = [];
    const move = [];
    for (const dep of core.dependencies) {
      if (!brandInScope(dep?.brand, ctx)) keep.push(dep);
      // Entries after every kept one stay at the end; the rest go first.
      else move.push(keep.length && core.dependencies.slice(core.dependencies.indexOf(dep)).every((d) => brandInScope(d?.brand, ctx))
        ? { ...dep, position: "end" } : dep);
    }
    if (move.length) {
      put(["dependencies"], move);
      if (keep.length) core.dependencies = keep;
      else delete core.dependencies;
    }
  }

  if (guideHeading && typeof core.guide === "string") {
    const { guide, sections } = extractGuideSections(core.guide, guideHeading, source);
    if (Object.keys(sections).length) {
      core.guide = guide;
      overlay.guide_sections = sections;
    }
  }

  pruneEmpty(core, recipe);
  return { core, overlay, movedVariants: [...movedVariants] };
}

function extract(node, rule, depth, p, ctx, put) {
  if (node === undefined || node === null) return;
  const seg = rule.path[depth];
  const last = depth === rule.path.length - 1;
  const keys = seg === "[task]"
    ? (Array.isArray(node) ? node.map(taskKey) : [])
    : isObj(node) ? Object.keys(node) : [];
  for (const key of keys) {
    if (seg === "@hw" ? !ctx.isHwKey(key) : seg !== "*" && seg !== "[task]" && seg !== key) continue;
    const child = coreChild(node, key, seg);
    if (last) {
      if (rule.union) {
        if (!Array.isArray(child)) continue;
        const moved = child.filter((k) => ctx.isHwKey(k));
        // A list of only in-scope ids is an in-scope variant (moved whole).
        if (moved.length && moved.length < child.length) {
          put([...p, key], moved);
          node[key] = child.filter((k) => !ctx.isHwKey(k));
        }
      } else {
        put([...p, key], child);
        delete node[key];
      }
    } else {
      extract(child, rule, depth + 1, [...p, key], ctx, put);
    }
  }
}

// Remove containers the split emptied (but keep ones empty in the original).
function pruneEmpty(node, original) {
  if (!isObj(node)) {
    if (Array.isArray(node)) node.forEach((c, i) => pruneEmpty(c, original?.[i]));
    return;
  }
  for (const [k, v] of Object.entries(node)) {
    pruneEmpty(v, original?.[k]);
    const wasEmpty = isObj(original?.[k]) && Object.keys(original[k]).length === 0;
    if (isObj(v) && Object.keys(v).length === 0 && !wasEmpty) delete node[k];
  }
}

// Split markdown into heading-delimited sections; move each section whose
// heading matches (plus its subsections) out. Adjacent moved sections share
// one marker.
function extractGuideSections(guide, heading, source) {
  const lines = guide.split("\n");
  const sections = {};
  const out = [];
  let fence = false;
  let movingLevel = 0;
  let current = null;
  let lastWasMarker = false;
  let n = 0;
  for (const line of lines) {
    if (/^\s*(```|~~~)/.test(line)) fence = !fence;
    const h = !fence && line.match(/^(#{1,6})\s+(.*)$/);
    if (h && movingLevel && h[1].length <= movingLevel) movingLevel = 0;
    if (h && !movingLevel && heading.test(h[2])) {
      movingLevel = h[1].length;
      if (!lastWasMarker) {
        current = `section${++n}`;
        sections[current] = [];
        out.push(`<!-- overlay:${source}:${current} -->`);
      }
    }
    if (movingLevel) {
      sections[current].push(line);
      lastWasMarker = true;
    } else {
      out.push(line);
      if (line.trim() !== "") lastWasMarker = false;
    }
  }
  return {
    guide: out.join("\n"),
    // Trailing "\n" terminates the last line, so merging the section back in
    // place of its marker line reproduces the original text exactly.
    sections: Object.fromEntries(Object.entries(sections).map(([k, v]) => [k, `${v.join("\n")}\n`])),
  };
}
