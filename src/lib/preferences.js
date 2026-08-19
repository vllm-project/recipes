/**
 * User preferences persisted across sessions via localStorage.
 *
 * Two scopes:
 *   1. Global (`vllm-recipes-prefs`): user's physical setup that's the same
 *      across every recipe — currently just `hardware`.
 *   2. Per-recipe (`vllm-recipes-recipe-state`, keyed by `hf_id`): selection
 *      state that's specific to a model — strategy, nodes, features. So
 *      picking TP on V4-Flash sticks for V4-Flash, while V4-Pro keeps its
 *      own choice independently.
 */

const KEY = "vllm-recipes-prefs";
const RECIPE_STATE_KEY = "vllm-recipes-recipe-state";

export function loadPreferences() {
  if (typeof window === "undefined") return {};
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

export function savePreference(key, value) {
  if (typeof window === "undefined") return;
  try {
    const prefs = loadPreferences();
    if (value === undefined || value === null || value === "") {
      delete prefs[key];
    } else {
      prefs[key] = value;
    }
    localStorage.setItem(KEY, JSON.stringify(prefs));
  } catch {}
}

export function loadRecipeState(hfId) {
  if (typeof window === "undefined" || !hfId) return {};
  try {
    const raw = localStorage.getItem(RECIPE_STATE_KEY);
    const all = raw ? JSON.parse(raw) : {};
    return all[hfId] || {};
  } catch {
    return {};
  }
}

// Merge `partial` into the recipe's state. Keys with empty/undefined values
// are removed; if the recipe ends up with no state, drop its entry entirely
// so storage stays tight as users browse.
export function saveRecipeState(hfId, partial) {
  if (typeof window === "undefined" || !hfId) return;
  try {
    const raw = localStorage.getItem(RECIPE_STATE_KEY);
    const all = raw ? JSON.parse(raw) : {};
    const merged = { ...(all[hfId] || {}), ...partial };
    for (const k of Object.keys(merged)) {
      const v = merged[k];
      const empty = v === undefined || v === null || v === "" ||
        (Array.isArray(v) && v.length === 0);
      if (empty) delete merged[k];
    }
    if (Object.keys(merged).length === 0) {
      delete all[hfId];
    } else {
      all[hfId] = merged;
    }
    localStorage.setItem(RECIPE_STATE_KEY, JSON.stringify(all));
  } catch {}
}
