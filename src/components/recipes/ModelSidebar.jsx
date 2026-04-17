"use client";

import { useState, useEffect } from "react";
import { usePathname } from "next/navigation";
import Link from "next/link";
import { getProviderLogo, getProviderDisplayName } from "@/lib/providers";
import { recipeHref } from "@/lib/recipe-utils";
import { ChevronRight } from "lucide-react";

export function ModelSidebar({ recipesByOrg }) {
  const pathname = usePathname();
  // URL: /<org>/<repo>
  const parts = pathname.split("/").filter(Boolean);
  const currentOrg = parts[0] || "";
  const currentRepo = parts[1] || "";

  const [expanded, setExpanded] = useState(null);

  useEffect(() => {
    const found = recipesByOrg.find(([org]) => org === currentOrg);
    if (found) setExpanded(found[0]);
  }, [currentOrg, recipesByOrg]);

  const toggle = (org) => setExpanded(expanded === org ? null : org);

  return (
    <nav className="w-52 shrink-0 hidden lg:block">
      <div className="sticky top-16 pt-4 space-y-px max-h-[calc(100vh-5rem)] overflow-y-auto scrollbar-thin">
        <div className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground px-3 pb-2">
          Models
        </div>
        {recipesByOrg.map(([org, models]) => {
          const logo = getProviderLogo(org);
          const displayName = getProviderDisplayName(org);
          const isExpanded = expanded === org;
          const hasActive = org === currentOrg;

          return (
            <div key={org}>
              <button
                onClick={() => toggle(org)}
                className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-left transition-colors ${
                  hasActive
                    ? "bg-muted text-foreground"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                }`}
              >
                {logo ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={logo} alt="" width={16} height={16} className="rounded shrink-0" />
                ) : (
                  <div className="w-4 h-4 rounded bg-muted flex items-center justify-center text-[9px] font-bold text-muted-foreground shrink-0">
                    {displayName.charAt(0)}
                  </div>
                )}
                <span className="flex-1 text-xs font-medium truncate">{displayName}</span>
                <ChevronRight
                  size={11}
                  className={`shrink-0 text-muted-foreground/40 transition-transform duration-200 ${isExpanded ? "rotate-90" : ""}`}
                />
              </button>

              {isExpanded && (
                <div className="ml-[27px] border-l border-border/60 pl-2.5 py-0.5 space-y-px">
                  {models.map((m) => {
                    const isActive = m.hf_repo === currentRepo && m.hf_org === currentOrg;
                    return (
                      <Link
                        key={m.hf_id}
                        href={recipeHref(m)}
                        className={`block px-2.5 py-1.5 rounded-md text-[11px] font-mono transition-colors ${
                          isActive
                            ? "bg-vllm-blue/10 text-vllm-blue font-semibold"
                            : "text-muted-foreground hover:text-foreground hover:bg-muted/30"
                        }`}
                      >
                        {m.hf_repo || m.meta.title}
                      </Link>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </nav>
  );
}
