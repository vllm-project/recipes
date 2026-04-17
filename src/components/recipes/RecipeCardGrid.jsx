"use client";

import { useMemo } from "react";
import Link from "next/link";
import { getProviderLogo, getProviderDisplayName } from "@/lib/providers";

export function RecipeCardGrid({ recipes }) {
  const byOrg = useMemo(() => {
    const groups = {};
    for (const r of recipes) {
      const org = r.hf_org || "unknown";
      if (!groups[org]) groups[org] = [];
      groups[org].push(r);
    }
    // Sort alphabetically by HF org (case-insensitive)
    return Object.entries(groups).sort((a, b) =>
      a[0].toLowerCase().localeCompare(b[0].toLowerCase())
    );
  }, [recipes]);

  return (
    <div>
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-border">
        <div className="flex items-baseline gap-2">
          <span className="text-sm font-semibold tabular-nums">{recipes.length}</span>
          <span className="text-xs text-muted-foreground">recipes</span>
          <span className="text-muted-foreground/50">&middot;</span>
          <span className="text-sm font-semibold tabular-nums">{byOrg.length}</span>
          <span className="text-xs text-muted-foreground">providers</span>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
        {byOrg.map(([org, models]) => (
          <ProviderTile key={org} org={org} models={models} />
        ))}
      </div>
    </div>
  );
}

function ProviderTile({ org, models }) {
  const logo = getProviderLogo(org);
  const displayName = getProviderDisplayName(org);

  return (
    <Link
      href={`/${org}`}
      className="group rounded-xl border border-border p-4 flex flex-col gap-2 hover:border-vllm-blue/40 hover:shadow-sm hover:-translate-y-0.5 transition-all bg-card"
    >
      <div className="flex items-center gap-2.5">
        {logo ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={logo} alt="" width={32} height={32} className="rounded-lg shrink-0" />
        ) : (
          <div className="w-8 h-8 rounded-lg bg-muted flex items-center justify-center text-sm font-bold text-muted-foreground shrink-0">
            {displayName.charAt(0)}
          </div>
        )}
        <div className="min-w-0 flex-1">
          <div className="font-semibold text-sm truncate group-hover:text-vllm-blue transition-colors">
            {displayName}
          </div>
          <div className="font-mono text-[10px] text-muted-foreground/70 truncate">{org}</div>
        </div>
      </div>

      <div className="flex items-baseline justify-between mt-auto pt-1">
        <span className="text-xs text-muted-foreground">
          <span className="font-semibold text-foreground tabular-nums">{models.length}</span>{" "}
          {models.length === 1 ? "recipe" : "recipes"}
        </span>
        <span className="text-muted-foreground/40 group-hover:text-vllm-blue group-hover:translate-x-0.5 transition-all text-xs">&rarr;</span>
      </div>
    </Link>
  );
}
