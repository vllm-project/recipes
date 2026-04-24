/**
 * User preferences persisted across sessions via localStorage.
 * Used by CommandBuilder to remember the user's preferred hardware.
 */

const KEY = "vllm-recipes-prefs";

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
