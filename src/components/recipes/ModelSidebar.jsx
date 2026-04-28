"use client";

import { useState, useEffect, useRef } from "react";
import { usePathname } from "next/navigation";
import Link from "next/link";
import { getProviderLogo, getProviderLogoClass, getProviderDisplayName } from "@/lib/providers";
import { recipeHref } from "@/lib/recipe-utils";
import { ChevronRight, Type, Eye, Sparkles, Hash, Cpu } from "lucide-react";
import { TooltipProvider, InfoTip } from "@/components/ui/tooltip";

const TASK_ICON = {
  text:       Type,
  multimodal: Eye,
  omni:       Sparkles,
  embedding:  Hash,
};

export function ModelSidebar({ recipesByOrg }) {
  const pathname = usePathname();
  const parts = pathname.split("/").filter(Boolean);
  const currentOrg = parts[0] || "";
  const currentRepo = parts[1] || "";

  // Compute the initial expanded org synchronously so SSR emits the recipe
  // links inside the current org — without this the server HTML only contains
  // org-level links, hurting crawl signal.
  const [expanded, setExpanded] = useState(() => {
    const found = recipesByOrg.find(([org]) => org === currentOrg);
    return found ? found[0] : null;
  });

  useEffect(() => {
    const found = recipesByOrg.find(([org]) => org === currentOrg);
    if (found) setExpanded(found[0]);
  }, [currentOrg, recipesByOrg]);

  const activeRef = useRef(null);
  useEffect(() => {
    if (!activeRef.current) return;
    const link = activeRef.current;
    const scroller = link.closest("[data-sidebar-scroll]");
    if (!scroller) return;
    const linkTop = link.offsetTop;
    const linkBottom = linkTop + link.offsetHeight;
    const viewTop = scroller.scrollTop;
    const viewBottom = viewTop + scroller.clientHeight;
    if (linkTop < viewTop || linkBottom > viewBottom) {
      scroller.scrollTop = linkTop - scroller.clientHeight / 2 + link.offsetHeight / 2;
    }
  }, [currentOrg, currentRepo, expanded]);

  const toggle = (org) => setExpanded(expanded === org ? null : org);

  return (
    <TooltipProvider>
    <nav className="w-64 shrink-0 hidden lg:block">
      <div data-sidebar-scroll className="sticky top-16 pt-4 space-y-0.5 max-h-[calc(100vh-5rem)] overflow-y-auto scrollbar-thin">
        <div className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground px-3 pb-2">
          Providers
        </div>
        {recipesByOrg.map(([org, models]) => {
          const logo = getProviderLogo(org);
          const displayName = getProviderDisplayName(org);
          const isExpanded = expanded === org;
          const isOrgPage = org === currentOrg && !currentRepo;
          const hasActiveModel = org === currentOrg && currentRepo;

          return (
            <div key={org}>
              <div
                className={`flex items-center gap-1 rounded-lg transition-colors ${
                  isOrgPage
                    ? "bg-vllm-blue/10 text-vllm-blue"
                    : hasActiveModel
                    ? "bg-muted/60 text-foreground"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted/40"
                }`}
              >
                <Link
                  href={`/${org}`}
                  className="flex items-center gap-2.5 px-3 py-2 flex-1 min-w-0"
                >
                  {logo ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={logo} alt="" width={18} height={18} className={`rounded shrink-0 ${getProviderLogoClass(org)}`} />
                  ) : (
                    <div className="w-[18px] h-[18px] rounded bg-muted flex items-center justify-center text-[10px] font-bold text-muted-foreground shrink-0">
                      {displayName.charAt(0)}
                    </div>
                  )}
                  <span className={`flex-1 text-sm truncate ${isOrgPage ? "font-semibold" : "font-medium"}`}>
                    {displayName}
                  </span>
                </Link>
                <button
                  onClick={() => toggle(org)}
                  aria-label={isExpanded ? "Collapse" : "Expand"}
                  className="px-2 py-2 hover:bg-muted/60 rounded-r-lg transition-colors shrink-0"
                >
                  <ChevronRight
                    size={12}
                    className={`text-muted-foreground/60 transition-transform duration-200 ${isExpanded ? "rotate-90" : ""}`}
                  />
                </button>
              </div>

              {isExpanded && (
                <div className="ml-[29px] border-l border-border/60 pl-2.5 py-0.5 space-y-px">
                  {models.map((m) => {
                    const isActive = m.hf_repo === currentRepo && m.hf_org === currentOrg;
                    const primaryTask = (m.meta.tasks || [])[0];
                    const Icon = TASK_ICON[primaryTask] || Cpu;
                    return (
                      <InfoTip key={m.hf_id} content={primaryTask}>
                        <Link
                          ref={isActive ? activeRef : null}
                          href={recipeHref(m)}
                          className={`flex items-center gap-1.5 px-2 py-1.5 rounded-md text-[11px] font-mono transition-colors ${
                            isActive
                              ? "bg-vllm-blue/10 text-vllm-blue font-semibold"
                              : "text-muted-foreground hover:text-foreground hover:bg-muted/30"
                          }`}
                        >
                          <Icon size={10} className="shrink-0 opacity-60" />
                          <span className="truncate">{m.hf_repo || m.meta.title}</span>
                        </Link>
                      </InfoTip>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </nav>
    </TooltipProvider>
  );
}
