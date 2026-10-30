"use client";

import { useEffect, useRef, useState } from "react";
import { Sun, Moon, Monitor, Check } from "lucide-react";
import { DARK_QUERY, THEME_KEY, applyTheme, readThemePref } from "@/lib/theme";

const OPTIONS = [
  { id: "light", label: "Light", Icon: Sun },
  { id: "dark", label: "Dark", Icon: Moon },
  { id: "system", label: "System", Icon: Monitor },
];

export function ThemeToggle() {
  const [pref, setPref] = useState(null); // null until mounted: the server can't know it
  const [open, setOpen] = useState(false);
  const root = useRef(null);

  useEffect(() => {
    setPref(readThemePref());
    // Another tab picked a theme.
    const onStorage = (e) => {
      if (e.key !== THEME_KEY) return;
      const next = readThemePref();
      setPref(next);
      applyTheme(next);
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  // Following the system: track OS changes (e.g. automatic dark mode at dusk).
  useEffect(() => {
    if (pref !== "system") return;
    const mq = window.matchMedia(DARK_QUERY);
    const onChange = () => applyTheme("system");
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, [pref]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e) => { if (!root.current?.contains(e.target)) setOpen(false); };
    const onKey = (e) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("pointerdown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const choose = (next) => {
    try {
      if (next === "system") localStorage.removeItem(THEME_KEY);
      else localStorage.setItem(THEME_KEY, next);
    } catch {}
    setPref(next);
    applyTheme(next);
    setOpen(false);
  };

  const current = OPTIONS.find((o) => o.id === pref);
  return (
    <div ref={root} className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={current ? `Theme: ${current.label}` : "Theme"}
        className="p-2 rounded-lg hover:bg-muted transition-colors cursor-pointer"
      >
        {current ? <current.Icon className="w-4 h-4 text-foreground" /> : <div className="w-4 h-4" />}
      </button>
      {open && (
        <div role="menu" className="absolute right-0 z-40 mt-1.5 w-36 rounded-xl border border-border bg-card py-1 shadow-xl">
          {OPTIONS.map(({ id, label, Icon }) => (
            <button
              key={id}
              type="button"
              role="menuitemradio"
              aria-checked={pref === id}
              onClick={() => choose(id)}
              className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm text-foreground hover:bg-muted/60 focus-visible:bg-muted/60 outline-none"
            >
              <Icon className="w-4 h-4 text-muted-foreground" /> {label}
              {pref === id && <Check className="ml-auto w-3.5 h-3.5 text-vllm-blue" />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
