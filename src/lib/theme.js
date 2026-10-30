// Color theme: "light", "dark", or "system" (follow the OS, the default).
// localStorage `theme` holds an explicit pick; no entry means system. The
// root layout runs THEME_SCRIPT in <head> so the `dark` class is set before
// first paint — applied after hydration, every dark-mode load flashed light.

export const THEME_KEY = "theme";
export const DARK_QUERY = "(prefers-color-scheme: dark)";

export const THEME_SCRIPT = `(function(){try{var t=localStorage.getItem("${THEME_KEY}");var d=t==="dark"||(t!=="light"&&matchMedia("${DARK_QUERY}").matches);document.documentElement.classList.toggle("dark",d)}catch(e){}})()`;

export function readThemePref() {
  try {
    const t = localStorage.getItem(THEME_KEY);
    return t === "light" || t === "dark" ? t : "system";
  } catch {
    return "system";
  }
}

export function applyTheme(pref) {
  const dark = pref === "dark" || (pref === "system" && window.matchMedia(DARK_QUERY).matches);
  document.documentElement.classList.toggle("dark", dark);
}
