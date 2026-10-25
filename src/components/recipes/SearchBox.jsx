"use client";

import { useState, useRef, useEffect, useMemo } from "react";
import { useRouter } from "next/navigation";
import { Search, ArrowRight, Building2, SlidersHorizontal } from "lucide-react";
import { compareRecipesByDateAdded, recipeHref } from "@/lib/recipe-utils";
import { getProviderLogo, getProviderLogoClass, getProviderDisplayName, PROVIDERS } from "@/lib/providers";
import { searchRecipes, searchProviders } from "@/lib/search";

export function SearchBox({ recipes }) {
  const [query, setQuery] = useState("");
  const [focused, setFocused] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef(null);
  const router = useRouter();

  // Build a unified results list: first matching providers, then matching recipes
  const results = useMemo(() => {
    if (!query.trim()) return [];

    // Count recipes per org (for display on provider entry)
    const orgCount = {};
    for (const r of recipes) orgCount[r.hf_org] = (orgCount[r.hf_org] || 0) + 1;

    // Find matching providers (by hf_org or display_name)
    const providerList = Object.entries(PROVIDERS).filter(([org]) => orgCount[org]);
    const providerMatches = searchProviders(providerList, query)
      .filter((entry, i, arr) => arr.findIndex((e) => e[1].display_name === entry[1].display_name) === i)
      .slice(0, 3)
      .map(([org, meta]) => ({
        type: "provider",
        org,
        displayName: meta.display_name,
        count: orgCount[org],
        href: `/${org}`,
      }));

    const allRecipeMatches = searchRecipes(recipes, query);
    const recipeMatches = allRecipeMatches
      .slice(0, 6)
      .map((r) => ({ type: "recipe", recipe: r, href: recipeHref(r) }));

    // Footer that hands off to /browse. When there are matches, scope the
    // browse view to this query so user lands in a pre-filtered state. With
    // zero matches, drop the q param and offer "Browse all" as a fallback.
    const browseEntry = allRecipeMatches.length > 0
      ? {
          type: "browse",
          count: allRecipeMatches.length,
          href: `/browse?q=${encodeURIComponent(query.trim())}`,
        }
      : {
          type: "browse",
          count: recipes.length,
          href: "/browse",
          fallback: true,
        };

    return [...providerMatches, ...recipeMatches, browseEntry];
  }, [query, recipes]);

  // Empty-state default — shown when the box is focused but no query yet.
  // Top 5 recipes most recently added to the catalog so ⌘K is immediately
  // useful for "what just landed?", plus the same Browse all footer.
  const defaultResults = useMemo(() => {
    if (query.trim()) return [];
    const sorted = [...recipes].sort(compareRecipesByDateAdded);
    return [
      ...sorted.slice(0, 5).map((r) => ({ type: "recipe", recipe: r, href: recipeHref(r) })),
      { type: "browse", count: recipes.length, href: "/browse", fallback: true },
    ];
  }, [query, recipes]);

  // Unified list the dropdown actually renders + arrow keys navigate.
  const list = query.trim() ? results : defaultResults;

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

  // Selecting a result needs to close the dropdown synchronously: otherwise
  // `setQuery("")` flips `list` from the matched results to the "Latest 5"
  // default while the route is still loading and the onBlur timeout (200ms)
  // hasn't run yet, causing a visible flash from results → Latest → close.
  const selectResult = (href) => {
    setFocused(false);
    setQuery("");
    setSelectedIndex(0);
    inputRef.current?.blur();
    router.push(href);
  };

  const handleKeyDown = (e) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIndex((i) => Math.min(i + 1, list.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIndex((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter" && list[selectedIndex]) {
      e.preventDefault();
      selectResult(list[selectedIndex].href);
    } else if (e.key === "Escape") {
      setFocused(false);
      setQuery("");
      setSelectedIndex(0);
      inputRef.current?.blur();
    }
  };

  const showDropdown = focused && list.length > 0;

  return (
    <div className="relative w-full max-w-md">
      <div className="relative">
        <Search size={16} className="absolute left-2.5 sm:left-3 top-1/2 -translate-y-1/2 text-muted-foreground pointer-events-none" />
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => { setQuery(e.target.value); setSelectedIndex(0); }}
          onFocus={() => setFocused(true)}
          onBlur={() => setTimeout(() => setFocused(false), 200)}
          onKeyDown={handleKeyDown}
          aria-label="Search models or providers"
          className="w-full rounded-lg border border-border bg-background pl-8 sm:pl-9 pr-2 sm:pr-3 py-2 text-base sm:text-sm focus:outline-none focus:ring-2 focus:ring-vllm-blue/40"
        />
        {/* Responsive placeholder: a native `placeholder` can't change with the
            viewport, and on a phone the header leaves room for one word.
            16px text below `sm`: iOS Safari zooms the page into any input
            under 16px on focus. */}
        {!query && (
          <span className="absolute left-8 sm:left-9 right-2 sm:right-3 top-1/2 -translate-y-1/2 flex items-center gap-2 text-base sm:text-sm text-muted-foreground pointer-events-none whitespace-nowrap overflow-hidden">
            <span className="sm:hidden">Search</span>
            <span className="hidden sm:inline truncate">Search models or providers...</span>
            <kbd className="hidden sm:inline ml-auto font-sans text-xs">⌘K</kbd>
          </span>
        )}
      </div>

      {showDropdown && (
        // On a phone the input is a narrow slot in the header, so the results
        // span the screen just below the (sticky) header instead of the input.
        <div className="fixed inset-x-4 top-[4.25rem] max-h-[calc(100dvh-5rem)] overflow-y-auto sm:absolute sm:inset-x-0 sm:top-full sm:mt-1 sm:max-h-none sm:overflow-hidden bg-card border border-border rounded-lg shadow-lg z-50">
          {!query.trim() && (
            <div className="px-3 py-1.5 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground bg-muted/30 border-b border-border">
              Latest
            </div>
          )}
          {list.map((r, i) => {
            const active = i === selectedIndex;
            const onClick = () => selectResult(r.href);
            if (r.type === "provider") {
              return <ProviderResult key={`p-${r.org}`} entry={r} active={active} onClick={onClick} />;
            }
            if (r.type === "recipe") {
              return <RecipeResult key={`r-${r.recipe.hf_id}`} entry={r} active={active} onClick={onClick} />;
            }
            return <BrowseAllResult key="browse" entry={r} active={active} onClick={onClick} query={query} />;
          })}
        </div>
      )}
    </div>
  );
}

function ProviderLogo({ org }) {
  const logo = getProviderLogo(org);
  return logo ? (
    // eslint-disable-next-line @next/next/no-img-element
    <img src={logo} alt="" width={22} height={22} className={`rounded shrink-0 ${getProviderLogoClass(org)}`} />
  ) : (
    <div className="w-[22px] h-[22px] rounded bg-muted flex items-center justify-center shrink-0">
      <Building2 size={12} className="text-muted-foreground" />
    </div>
  );
}

function ProviderResult({ entry, active, onClick }) {
  return (
    <button
      onMouseDown={onClick}
      className={`w-full text-left px-3 py-2.5 flex items-center gap-3 text-sm transition-colors border-l-2 ${
        active ? "bg-muted border-vllm-blue" : "border-transparent hover:bg-muted/50"
      }`}
    >
      <ProviderLogo org={entry.org} />
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

function BrowseAllResult({ entry, active, onClick, query }) {
  // Footer row in the search dropdown. Two voices:
  //   matches:  "Browse N matching recipes →" (link to /browse?q=...)
  //   none:     "Browse all N recipes →" (link to /browse, fallback)
  const label = entry.fallback
    ? `Browse all ${entry.count} recipes`
    : `Browse ${entry.count} matching recipe${entry.count === 1 ? "" : "s"}`;
  return (
    <button
      onMouseDown={onClick}
      className={`w-full text-left px-3 py-2.5 flex items-center gap-3 text-sm transition-colors border-l-2 border-t border-t-border ${
        active ? "bg-muted border-vllm-blue" : "border-transparent hover:bg-muted/50"
      }`}
    >
      <div className="w-[22px] h-[22px] rounded bg-muted flex items-center justify-center shrink-0">
        <SlidersHorizontal size={12} className="text-muted-foreground" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="font-medium truncate">
          {label}
          {!entry.fallback && (
            <span className="text-[11px] text-muted-foreground font-mono ml-2">q = &ldquo;{query}&rdquo;</span>
          )}
        </div>
        <div className="text-xs text-muted-foreground">
          {entry.fallback ? "Open the full filter view" : "Open in browse with filters available"}
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
      <ProviderLogo org={r.hf_org} />
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
