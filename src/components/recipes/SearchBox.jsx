"use client";

import { useState, useRef, useEffect, useMemo } from "react";
import { useRouter } from "next/navigation";
import { Search, ArrowRight } from "lucide-react";
import { recipeHref } from "@/lib/recipe-utils";

export function SearchBox({ recipes }) {
  const [query, setQuery] = useState("");
  const [focused, setFocused] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef(null);
  const router = useRouter();

  // Simple search: match title, provider, tasks, description
  const results = useMemo(() => {
    if (!query.trim()) return [];
    const q = query.toLowerCase();
    return recipes
      .filter((r) => {
        const hay = [
          r.meta.title,
          r.meta.provider,
          r.meta.description,
          ...(r.meta.tasks || []),
          r.model.architecture,
          r.model.parameter_count,
        ]
          .join(" ")
          .toLowerCase();
        return hay.includes(q);
      })
      .slice(0, 8);
  }, [query, recipes]);

  // Keyboard shortcuts
  useEffect(() => {
    function onKeyDown(e) {
      // Cmd+K or / to focus search
      if ((e.metaKey && e.key === "k") || (e.key === "/" && document.activeElement === document.body)) {
        e.preventDefault();
        inputRef.current?.focus();
      }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, []);

  const handleKeyDown = (e) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIndex((i) => Math.min(i + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIndex((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter" && results[selectedIndex]) {
      router.push(recipeHref(results[selectedIndex]));
      setQuery("");
      inputRef.current?.blur();
    } else if (e.key === "Escape") {
      setQuery("");
      inputRef.current?.blur();
    }
  };

  const showDropdown = focused && query.trim() && results.length > 0;

  return (
    <div className="relative w-full max-w-md">
      <div className="relative">
        <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground pointer-events-none" />
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => { setQuery(e.target.value); setSelectedIndex(0); }}
          onFocus={() => setFocused(true)}
          onBlur={() => setTimeout(() => setFocused(false), 200)}
          onKeyDown={handleKeyDown}
          placeholder="Search models...  ⌘K"
          className="w-full rounded-lg border border-border bg-background pl-9 pr-3 py-2 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-vllm-blue/40"
        />
      </div>

      {showDropdown && (
        <div className="absolute top-full mt-1 left-0 right-0 bg-card border border-border rounded-lg shadow-lg overflow-hidden z-50">
          {results.map((r, i) => (
            <button
              key={r.meta.slug}
              onMouseDown={() => router.push(recipeHref(r))}
              className={`w-full text-left px-3 py-2.5 flex items-center gap-3 text-sm transition-colors ${
                i === selectedIndex ? "bg-muted" : "hover:bg-muted/50"
              }`}
            >
              <div className="flex-1 min-w-0">
                <div className="font-medium truncate">{r.meta.title}</div>
                <div className="text-xs text-muted-foreground truncate">
                  {r.meta.provider} &middot; {r.model.parameter_count} &middot; {r.model.architecture}
                </div>
              </div>
              <ArrowRight size={14} className="text-muted-foreground shrink-0" />
            </button>
          ))}
        </div>
      )}

      {focused && query.trim() && results.length === 0 && (
        <div className="absolute top-full mt-1 left-0 right-0 bg-card border border-border rounded-lg shadow-lg p-4 text-sm text-muted-foreground z-50">
          No recipes found for &ldquo;{query}&rdquo;
        </div>
      )}
    </div>
  );
}
