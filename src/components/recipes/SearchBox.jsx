"use client";

import { useState, useRef, useEffect, useMemo } from "react";
import { useRouter } from "next/navigation";
import { Search, ArrowRight, Building2, SlidersHorizontal } from "lucide-react";
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

    // Find matching recipes — compute total match count first, then slice
    // for display. The total drives the "Browse N matching recipes" footer
    // so users have an exit when there are more matches than fit.
    const allRecipeMatches = recipes.filter((r) => {
      // Verified hardware ids land in the haystack so "h100", "mi300x",
      // "b200" etc. find recipes by GPU compatibility. "tpu" is added as
      // a synonym whenever any TPU profile is verified, since the ids
      // (trillium / ironwood) don't contain that string.
      const hwKeys = Object.entries(r.meta?.hardware || {})
        .filter(([, s]) => s === "verified")
        .map(([h]) => h);
      const hwExtra = hwKeys.some((k) => k === "trillium" || k === "ironwood") ? ["tpu"] : [];
      const hay = [
        r.meta.title,
        r.hf_repo,
        r.hf_org,
        r.meta.provider,
        r.meta.description,
        ...(r.meta.tasks || []),
        r.model.architecture,
        r.model.parameter_count,
        ...hwKeys,
        ...hwExtra,
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return hay.includes(q);
    });
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
  // Top 5 newest recipes by HF release date so ⌘K is immediately useful
  // for "what just landed?", plus the same Browse all footer.
  const defaultResults = useMemo(() => {
    if (query.trim()) return [];
    const sorted = [...recipes].sort((a, b) => {
      const ra = a.hf_released ? new Date(a.hf_released).getTime() : 0;
      const rb = b.hf_released ? new Date(b.hf_released).getTime() : 0;
      return rb - ra;
    });
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

  const handleKeyDown = (e) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIndex((i) => Math.min(i + 1, list.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIndex((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter" && list[selectedIndex]) {
      router.push(list[selectedIndex].href);
      setQuery("");
      inputRef.current?.blur();
    } else if (e.key === "Escape") {
      setQuery("");
      inputRef.current?.blur();
    }
  };

  const showDropdown = focused && list.length > 0;

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
          {!query.trim() && (
            <div className="px-3 py-1.5 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground bg-muted/30 border-b border-border">
              Latest
            </div>
          )}
          {list.map((r, i) => {
            const active = i === selectedIndex;
            const onClick = () => { router.push(r.href); setQuery(""); inputRef.current?.blur(); };
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
