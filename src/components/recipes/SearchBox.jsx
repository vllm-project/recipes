"use client";

import { useState, useRef, useEffect, useMemo } from "react";
import { useRouter } from "next/navigation";
import { Search, ArrowRight, Building2 } from "lucide-react";
import { recipeHref } from "@/lib/recipe-utils";
import { getProviderLogo, getProviderLogoClass, getProviderDisplayName, PROVIDERS } from "@/lib/providers";

export function SearchBox({ recipes }) {
  const [query, setQuery] = useState("");
  const [focused, setFocused] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef(null);
  const router = useRouter();

  // Build a unified results list: first matching providers, then matching recipes
  const results = useMemo(() => {
    if (!query.trim()) return [];
    const q = query.toLowerCase();

    // Count recipes per org (for display on provider entry)
    const orgCount = {};
    for (const r of recipes) orgCount[r.hf_org] = (orgCount[r.hf_org] || 0) + 1;

    // Find matching providers (by hf_org or display_name)
    const providerMatches = Object.entries(PROVIDERS)
      .filter(([org, meta]) => {
        if (!orgCount[org]) return false; // only providers that have recipes
        const hay = `${org} ${meta.display_name}`.toLowerCase();
        return hay.includes(q);
      })
      // Dedupe case variants (e.g. "google" and "Google" both match)
      .filter((entry, i, arr) => arr.findIndex((e) => e[1].display_name === entry[1].display_name) === i)
      .slice(0, 3)
      .map(([org, meta]) => ({
        type: "provider",
        org,
        displayName: meta.display_name,
        count: orgCount[org],
        href: `/${org}`,
      }));

    // Find matching recipes
    const recipeMatches = recipes
      .filter((r) => {
        const hay = [
          r.meta.title,
          r.hf_repo,
          r.hf_org,
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
      .slice(0, 6)
      .map((r) => ({ type: "recipe", recipe: r, href: recipeHref(r) }));

    return [...providerMatches, ...recipeMatches];
  }, [query, recipes]);

  useEffect(() => {
    function onKeyDown(e) {
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
      router.push(results[selectedIndex].href);
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
          placeholder="Search models or providers...  ⌘K"
          className="w-full rounded-lg border border-border bg-background pl-9 pr-3 py-2 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-vllm-blue/40"
        />
      </div>

      {showDropdown && (
        <div className="absolute top-full mt-1 left-0 right-0 bg-card border border-border rounded-lg shadow-lg overflow-hidden z-50">
          {results.map((r, i) => {
            const active = i === selectedIndex;
            return r.type === "provider" ? (
              <ProviderResult key={`p-${r.org}`} entry={r} active={active} onClick={() => router.push(r.href)} />
            ) : (
              <RecipeResult key={`r-${r.recipe.hf_id}`} entry={r} active={active} onClick={() => router.push(r.href)} />
            );
          })}
        </div>
      )}

      {focused && query.trim() && results.length === 0 && (
        <div className="absolute top-full mt-1 left-0 right-0 bg-card border border-border rounded-lg shadow-lg p-4 text-sm text-muted-foreground z-50">
          No matches for &ldquo;{query}&rdquo;
        </div>
      )}
    </div>
  );
}

function ProviderResult({ entry, active, onClick }) {
  const logo = getProviderLogo(entry.org);
  return (
    <button
      onMouseDown={onClick}
      className={`w-full text-left px-3 py-2.5 flex items-center gap-3 text-sm transition-colors border-l-2 ${
        active ? "bg-muted border-vllm-blue" : "border-transparent hover:bg-muted/50"
      }`}
    >
      {logo ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={logo} alt="" width={22} height={22} className={`rounded shrink-0 ${getProviderLogoClass(entry.org)}`} />
      ) : (
        <div className="w-[22px] h-[22px] rounded bg-muted flex items-center justify-center">
          <Building2 size={12} className="text-muted-foreground" />
        </div>
      )}
      <div className="flex-1 min-w-0">
        <div className="font-medium truncate flex items-center gap-2">
          {entry.displayName}
          <span className="text-[10px] font-normal text-muted-foreground font-mono">{entry.org}</span>
        </div>
        <div className="text-xs text-muted-foreground">
          Provider · {entry.count} recipe{entry.count !== 1 ? "s" : ""}
        </div>
      </div>
      <ArrowRight size={14} className="text-muted-foreground shrink-0" />
    </button>
  );
}

function RecipeResult({ entry, active, onClick }) {
  const r = entry.recipe;
  return (
    <button
      onMouseDown={onClick}
      className={`w-full text-left px-3 py-2.5 flex items-center gap-3 text-sm transition-colors border-l-2 ${
        active ? "bg-muted border-vllm-blue" : "border-transparent hover:bg-muted/50"
      }`}
    >
      <div className="flex-1 min-w-0">
        <div className="font-medium truncate font-mono text-[13px]">
          {r.hf_repo || r.meta.title}
        </div>
        <div className="text-xs text-muted-foreground truncate">
          {r.meta.provider} · {r.model.parameter_count} · {r.model.architecture}
        </div>
      </div>
      <ArrowRight size={14} className="text-muted-foreground shrink-0" />
    </button>
  );
}
