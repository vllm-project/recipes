"use client";

// Sections below the command builder that follow its selection: a "For this
// configuration" panel, then the reference tabs (Guide · Presets ·
// Troubleshooting). Content (Markdown) is rendered on the server and passed in
// as nodes; these only decide placement. Before the builder publishes a
// selection (server HTML, pre-hydration) everything renders unfiltered, and
// every tab panel stays in the DOM, so no content is lost to search or to
// find-in-page.

import { useCallback, useEffect, useRef, useState } from "react";
import { ExternalLink, Link2 } from "lucide-react";
import { useSelection } from "@/lib/selection-store";
import { whenMatches } from "@/lib/recipe-context";

function HeadingLink({ id }) {
  return (
    <a href={`#${id}`} data-guide-link className="guide-heading-link" aria-label="Copy link to this section" title="Copy link to this section">
      <Link2 size={16} aria-hidden="true" />
    </a>
  );
}

function Chevron({ className = "w-4 h-4 group-open:rotate-180" }) {
  return (
    <svg className={`${className} shrink-0 text-muted-foreground transition-transform duration-200`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
    </svg>
  );
}

function Chips({ labels }) {
  if (!labels?.length) return null;
  const MAX = 3;
  const chip = "rounded border border-foreground/15 px-1.5 py-px font-mono text-[11px] font-normal text-muted-foreground";
  return (
    <span className="flex flex-wrap gap-1" title={labels.length > MAX ? labels.join(", ") : undefined}>
      {labels.slice(0, MAX).map((l) => <span key={l} className={chip}>{l}</span>)}
      {labels.length > MAX && <span className={chip}>+{labels.length - MAX}</span>}
    </span>
  );
}

function SourceLink({ source }) {
  if (!source?.href) return null;
  return (
    <a href={source.href} className="inline-flex items-center gap-1 text-vllm-blue hover:underline" target="_blank" rel="noreferrer">
      {source.label || "Source"}<ExternalLink size={11} aria-hidden="true" />
    </a>
  );
}

function Runbook({ preset }) {
  return (
    <details className="group/rb mt-3 rounded-lg border border-foreground/10">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-4 py-2.5 text-sm hover:bg-foreground/[0.03] [&::-webkit-details-marker]:hidden">
        <span className="min-w-0">
          <span className="font-medium">Runbook</span>
          {preset.toc?.length > 0 && (
            <span className="ml-2 text-xs text-muted-foreground">{preset.toc.map((t) => t.text).join(" · ")}</span>
          )}
        </span>
        <Chevron className="w-3.5 h-3.5 group-open/rb:rotate-180" />
      </summary>
      <div className="guide-content border-t border-foreground/10 px-4 pt-3 pb-4">{preset.runbook}</div>
    </details>
  );
}

// What applies to the command just built: the active preset and its runbook,
// notes whose `when` matches, and links to matching known issues. Hidden when
// nothing matches, so the tabs below never shift.
export function ForThisConfiguration({ hfId, hardware, strategyLabels, presets, notes, troubleshooting }) {
  const sel = useSelection(hfId);
  useRevealHashOnSelection(!!sel);
  if (!sel) return null;
  const preset = presets.find((p) => p.key === sel.preset);
  const myNotes = notes.filter((n) => whenMatches(n.when, sel, hardware));
  const myIssues = troubleshooting.filter((t) => t.when && whenMatches(t.when, sel, hardware));
  if (!preset && !myNotes.length && !myIssues.length) return null;

  const scope = [hardware[sel.hwId]?.display_name || sel.hwId, strategyLabels[sel.strategy] || sel.strategy].filter(Boolean).join(" · ");
  return (
    <section id="this-configuration" aria-labelledby="this-configuration-title" className="rounded-xl border border-foreground/15 bg-card/40 scroll-mt-24">
      <div className="flex items-center justify-between gap-3 border-b border-foreground/10 px-5 py-3">
        <h2 id="this-configuration-title" className="flex items-center gap-2 text-sm font-semibold">
          For this configuration<HeadingLink id="this-configuration" />
        </h2>
        <span className="truncate text-xs text-muted-foreground">{scope}</span>
      </div>
      <div className="space-y-4 px-5 pt-4 pb-5">
        {preset && (
          <div>
            <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm">
              <span className="text-muted-foreground">Preset</span>
              <span className="font-semibold">{preset.label}</span>
              {preset.workload && <span className="text-xs text-muted-foreground">· {preset.workload}</span>}
            </div>
            {preset.description && <p className="mt-1 text-sm">{preset.description}</p>}
            {preset.source?.href && <p className="mt-1 text-xs"><SourceLink source={preset.source} /></p>}
            {preset.runbook && <Runbook preset={preset} />}
          </div>
        )}
        {myNotes.map((n, i) => (
          <div key={i} className="guide-content">
            {n.title && <p className="!mb-1 font-semibold">{n.title}</p>}
            {n.body}
          </div>
        ))}
        {myIssues.length > 0 && (
          <div className="text-sm">
            <span className="text-muted-foreground">Troubleshooting for this configuration: </span>
            {myIssues.map((t, i) => (
              <span key={t.id}>{i > 0 && " · "}<a href={`#${t.id}`} className="text-vllm-blue hover:underline">{t.title}</a></span>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

function PresetCard({ p, hardware, strategyLabels, active }) {
  const id = `preset-${p.key}`;
  const builderHref = `?hardware=${p.hardware}${p.strategy ? `&strategy=${p.strategy}` : ""}&preset=${p.key}#recipe-builder`;
  return (
    <div className={`rounded-lg border px-4 py-3 ${active ? "border-vllm-blue/50 bg-vllm-blue/[0.04]" : "border-foreground/10"}`}>
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <span className="rounded-md border border-foreground/15 px-1.5 py-0.5 font-mono text-xs text-muted-foreground">{hardware[p.hardware]?.display_name || p.hardware}</span>
        <span id={id} className="scroll-mt-24 font-semibold">{p.label}</span>
        {p.default && <span className="rounded-md bg-muted px-1.5 py-0.5 text-[11px] font-medium">Default</span>}
        {active && <span className="rounded-md bg-vllm-blue/15 px-1.5 py-0.5 text-[11px] font-medium text-vllm-blue">In use</span>}
        <HeadingLink id={id} />
      </div>
      <div className="mt-1 text-xs text-muted-foreground">
        {[strategyLabels[p.strategy] || p.strategy, p.workload].filter(Boolean).join(" · ")}
      </div>
      {p.description && <p className="mt-1.5 text-sm">{p.description}</p>}
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs">
        {active
          ? <a href="#this-configuration" className="text-vllm-blue hover:underline">Runbook in For this configuration ↑</a>
          : <a href={builderHref} className="text-vllm-blue hover:underline">Use in builder</a>}
        <SourceLink source={p.source} />
      </div>
      {!active && p.runbook && <Runbook preset={p} />}
    </div>
  );
}

function Issue({ t }) {
  return (
    <details className="group/issue rounded-lg border border-foreground/10">
      <summary className="flex cursor-pointer list-none items-start justify-between gap-3 px-4 py-2.5 hover:bg-foreground/[0.03] [&::-webkit-details-marker]:hidden">
        <span className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
          <span id={t.id} className="scroll-mt-24 text-sm font-medium">{t.title}</span>
          <Chips labels={t.labels} />
        </span>
        <Chevron className="mt-0.5 w-3.5 h-3.5 group-open/issue:rotate-180" />
      </summary>
      <div className="guide-content border-t border-foreground/10 px-4 pt-3 pb-4">{t.body}</div>
    </details>
  );
}

// Open every <details> around `el` and bring it into view.
function reveal(el) {
  for (let d = el.closest("details"); d; d = d.parentElement?.closest("details")) d.open = true;
  el.scrollIntoView({ block: "start" });
}

// The selection arrives after hydration and can add the For this configuration
// panel above a #hash target the browser already scrolled to (or be the
// target itself), so scroll to it once more when it does.
function useRevealHashOnSelection(ready, within) {
  const done = useRef(false);
  useEffect(() => {
    if (!ready || done.current) return;
    done.current = true;
    let id;
    try { id = decodeURIComponent(window.location.hash.slice(1)); } catch { return; }
    const target = id && document.getElementById(id);
    if (!target || (within && !within.current?.contains(target))) return;
    requestAnimationFrame(() => requestAnimationFrame(() => reveal(target)));
  }, [ready, within]);
}

// Guide · Presets · Troubleshooting as tabs. Empty tabs are dropped; with a
// single tab left there is no tab bar. Inactive panels use
// hidden="until-found", so find-in-page still reaches them and switches the
// tab (beforematch). A #hash inside a panel selects its tab first.
export function ReferenceTabs({ hfId, hardware, strategyLabels, presets, troubleshooting, guide }) {
  const sel = useSelection(hfId);
  const mine = sel ? troubleshooting.filter((t) => whenMatches(t.when, sel, hardware)) : troubleshooting;
  const other = sel ? troubleshooting.filter((t) => !whenMatches(t.when, sel, hardware)) : [];
  const flagged = sel && troubleshooting.some((t) => t.when && whenMatches(t.when, sel, hardware));

  const tabs = [
    guide && { id: "recipe-guide", label: "Guide" },
    presets.length && { id: "presets", label: "Presets", count: presets.length },
    troubleshooting.length && { id: "troubleshooting", label: "Troubleshooting", count: sel ? mine.length : troubleshooting.length, dot: flagged },
  ].filter(Boolean);
  const [current, setCurrent] = useState(tabs[0]?.id);
  const active = tabs.some((t) => t.id === current) ? current : tabs[0]?.id;
  const root = useRef(null);
  const tabRefs = useRef({});
  useRevealHashOnSelection(!!sel, root);

  const select = useCallback((id, { focus = false, updateHash = false } = {}) => {
    setCurrent(id);
    if (focus) tabRefs.current[id]?.focus();
    if (updateHash) {
      const url = new URL(window.location.href);
      url.hash = id;
      window.history.replaceState(window.history.state, "", url);
    }
  }, []);

  // Inactive panels: hidden="until-found" (React writes a boolean `hidden`).
  useEffect(() => {
    for (const el of root.current?.querySelectorAll("[data-tab-panel]") || []) {
      if (el.dataset.tabPanel === active) el.removeAttribute("hidden");
      else el.setAttribute("hidden", "until-found");
    }
  }, [active, sel]);

  useEffect(() => {
    const node = root.current;
    if (!node) return;
    const onMatch = (e) => {
      const panel = e.target.closest?.("[data-tab-panel]");
      if (panel) setCurrent(panel.dataset.tabPanel);
    };
    const onHash = () => {
      let id;
      try { id = decodeURIComponent(window.location.hash.slice(1)); } catch { return; }
      const target = id && document.getElementById(id);
      if (!target || !node.contains(target)) return;
      const tab = target.dataset?.tabPanel || target.closest("[data-tab-panel]")?.dataset.tabPanel;
      if (tab) setCurrent(tab);
      // After the tab switch commits and the panel is unhidden.
      requestAnimationFrame(() => requestAnimationFrame(() => reveal(target)));
    };
    node.addEventListener("beforematch", onMatch, true);
    window.addEventListener("hashchange", onHash);
    onHash();
    return () => {
      node.removeEventListener("beforematch", onMatch, true);
      window.removeEventListener("hashchange", onHash);
    };
  }, []);

  if (!tabs.length) return null;

  const onKeyDown = (e) => {
    const idx = Math.max(0, tabs.findIndex((t) => t.id === active));
    const next = e.key === "ArrowRight" ? tabs[(idx + 1) % tabs.length]
      : e.key === "ArrowLeft" ? tabs[(idx - 1 + tabs.length) % tabs.length]
      : e.key === "Home" ? tabs[0]
      : e.key === "End" ? tabs[tabs.length - 1] : null;
    if (next) { e.preventDefault(); select(next.id, { focus: true, updateHash: true }); }
  };

  const panelProps = (id) => ({
    id, "data-tab-panel": id, role: tabs.length > 1 ? "tabpanel" : undefined,
    "aria-labelledby": tabs.length > 1 ? `tab-${id}` : undefined,
    // Padding lives on an inner wrapper: a hidden="until-found" panel keeps its
    // box (content-visibility: hidden), so its own padding would show as a gap.
    hidden: id !== active, className: "scroll-mt-24",
  });

  return (
    <div ref={root} className="rounded-xl border border-foreground/15 bg-card/40">
      {tabs.length > 1 ? (
        <div role="tablist" aria-label="Recipe reference" onKeyDown={onKeyDown} className="flex overflow-x-auto border-b border-foreground/10 px-3">
          {tabs.map((t) => (
            <button
              key={t.id}
              id={`tab-${t.id}`}
              ref={(el) => { tabRefs.current[t.id] = el; }}
              role="tab"
              aria-selected={active === t.id}
              aria-controls={t.id}
              tabIndex={active === t.id ? 0 : -1}
              onClick={() => select(t.id, { updateHash: true })}
              className={`-mb-px flex shrink-0 items-center gap-1.5 whitespace-nowrap border-b-2 px-3 py-3 text-sm transition-colors ${active === t.id ? "border-vllm-blue font-semibold text-foreground" : "border-transparent text-muted-foreground hover:text-foreground"}`}
            >
              {t.label}
              {t.count != null && <span className="text-xs font-normal text-muted-foreground tabular-nums">{t.count}</span>}
              {t.dot && <span className="h-1.5 w-1.5 rounded-full bg-amber-500" aria-label="matches this configuration" />}
            </button>
          ))}
        </div>
      ) : (
        <div className="flex items-center gap-2 border-b border-foreground/10 px-5 py-3 text-sm font-semibold">
          {tabs[0].label}<HeadingLink id={tabs[0].id} />
        </div>
      )}

      {guide && <div {...panelProps("recipe-guide")}><div className="guide-content px-5 pt-4 pb-5 [&>:first-child]:!mt-0">{guide}</div></div>}

      {presets.length > 0 && (
        <div {...panelProps("presets")}><div className="px-5 pt-4 pb-5">
          <p className="mb-3 text-sm text-muted-foreground">
            Configurations qualified on specific hardware, layered on the baseline command. A default preset applies automatically in the builder on its hardware; pick Baseline there to drop it.
          </p>
          <div className="space-y-2">
            {(sel ? [...presets].sort((a, b) => (b.hardware === sel.hwId) - (a.hardware === sel.hwId)) : presets).map((p) => (
              <PresetCard key={p.key} p={p} hardware={hardware} strategyLabels={strategyLabels} active={sel?.preset === p.key} />
            ))}
          </div>
        </div></div>
      )}

      {troubleshooting.length > 0 && (
        <div {...panelProps("troubleshooting")}><div className="px-5 pt-4 pb-5">
          {sel && <p className="mb-3 text-sm text-muted-foreground">{mine.length ? `${mine.length} apply to this configuration.` : "None apply to this configuration."}</p>}
          <div className="space-y-2">{mine.map((t) => <Issue key={t.id} t={t} />)}</div>
          {other.length > 0 && (
            <details className="group/other mt-3">
              <summary className="flex cursor-pointer list-none items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground [&::-webkit-details-marker]:hidden">
                <Chevron className="w-3.5 h-3.5 -rotate-90 group-open/other:rotate-0" />
                Other configurations ({other.length})
              </summary>
              <div className="mt-2 space-y-2">{other.map((t) => <Issue key={t.id} t={t} />)}</div>
            </details>
          )}
        </div></div>
      )}
    </div>
  );
}
