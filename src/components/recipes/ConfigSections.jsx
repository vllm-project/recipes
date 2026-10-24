"use client";

// Sections below the command builder that follow its selection. Content
// (Markdown) is rendered on the server and passed in as nodes; these only
// decide placement. Before the builder publishes a selection (server HTML,
// pre-hydration) everything renders unfiltered, so no content is lost.

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

// Same frame as the page's Accordion.
function Panel({ id, title, aside, children }) {
  return (
    <details id={id} className="group rounded-xl border border-foreground/15 bg-card/40 overflow-hidden scroll-mt-24" open>
      <summary className="px-5 py-3 cursor-pointer text-sm font-semibold select-none hover:bg-foreground/[0.04] transition-colors flex items-center justify-between gap-3 list-none [&::-webkit-details-marker]:hidden">
        <span className="flex min-w-0 items-center gap-2">{title}<HeadingLink id={id} /></span>
        <span className="flex items-center gap-3">
          {aside && <span className="hidden sm:inline truncate text-xs font-normal text-muted-foreground">{aside}</span>}
          <Chevron />
        </span>
      </summary>
      <div className="px-5 pb-5 border-t border-foreground/10 pt-4">{children}</div>
    </details>
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

function Runbook({ profile, label = "Runbook" }) {
  return (
    <details className="group/rb mt-3 rounded-lg border border-foreground/10">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-4 py-2.5 text-sm hover:bg-foreground/[0.03] [&::-webkit-details-marker]:hidden">
        <span className="min-w-0">
          <span className="font-medium">{label}</span>
          {profile.toc?.length > 0 && (
            <span className="ml-2 text-xs text-muted-foreground">{profile.toc.map((t) => t.text).join(" · ")}</span>
          )}
        </span>
        <Chevron className="w-3.5 h-3.5 group-open/rb:rotate-180" />
      </summary>
      <div className="guide-content border-t border-foreground/10 px-4 pt-3 pb-4">{profile.runbook}</div>
    </details>
  );
}

// ① What applies to the command just built: the active tuned profile and its
// runbook, notes whose `when` matches, and a pointer to matching known issues.
export function ThisConfiguration({ hfId, hardware, strategyLabels, profiles, notes, troubleshooting }) {
  const sel = useSelection(hfId);
  if (!sel) return null;
  const profile = profiles.find((p) => p.key === sel.profile);
  const myNotes = notes.filter((n) => whenMatches(n.when, sel, hardware));
  const myIssues = troubleshooting.filter((t) => t.when && whenMatches(t.when, sel, hardware));
  if (!profile && !myNotes.length && !myIssues.length) return null;

  const aside = [hardware[sel.hwId]?.display_name || sel.hwId, strategyLabels[sel.strategy] || sel.strategy].filter(Boolean).join(" · ");
  return (
    <Panel id="this-configuration" title="This configuration" aside={aside}>
      <div className="space-y-4">
        {profile && (
          <div>
            <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm">
              <span className="text-muted-foreground">Tuned profile</span>
              <span className="font-semibold">{profile.label}</span>
              {profile.workload && <span className="text-xs text-muted-foreground">· {profile.workload}</span>}
            </div>
            {profile.description && <p className="mt-1 text-sm">{profile.description}</p>}
            {profile.source?.href && <p className="mt-1 text-xs"><SourceLink source={profile.source} /></p>}
            {profile.runbook && <Runbook profile={profile} />}
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
            <span className="text-muted-foreground">Known issues for this configuration: </span>
            {myIssues.map((t, i) => (
              <span key={t.id}>{i > 0 && " · "}<a href={`#${t.id}`} className="text-vllm-blue hover:underline">{t.title}</a></span>
            ))}
          </div>
        )}
      </div>
    </Panel>
  );
}

function ProfileCard({ p, hardware, strategyLabels, active }) {
  const id = `profile-${p.key}`;
  const builderHref = `?hardware=${p.hardware}${p.strategy ? `&strategy=${p.strategy}` : ""}&profile=${p.key}#recipe-builder`;
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
          ? <a href="#this-configuration" className="text-vllm-blue hover:underline">Runbook in This configuration ↑</a>
          : <a href={builderHref} className="text-vllm-blue hover:underline">Use in builder</a>}
        <SourceLink source={p.source} />
      </div>
      {!active && p.runbook && <Runbook profile={p} />}
    </div>
  );
}

// ④ Every tuned profile, the selected hardware's first. The active one's
// runbook renders in This configuration instead (each runbook renders once,
// so heading ids stay unique).
export function TunedProfiles({ hfId, hardware, strategyLabels, profiles }) {
  const sel = useSelection(hfId);
  if (!profiles.length) return null;
  const ordered = sel ? [...profiles].sort((a, b) => (b.hardware === sel.hwId) - (a.hardware === sel.hwId)) : profiles;
  return (
    <Panel id="tuned-profiles" title={<>Tuned profiles<span className="font-normal text-muted-foreground">{profiles.length}</span></>}>
      <p className="text-sm text-muted-foreground mb-3">
        Configurations qualified on specific hardware, layered on the baseline command. A default profile applies automatically in the builder on its hardware; pick Baseline there to drop it.
      </p>
      <div className="space-y-2">
        {ordered.map((p) => (
          <ProfileCard key={p.key} p={p} hardware={hardware} strategyLabels={strategyLabels} active={sel?.profile === p.key} />
        ))}
      </div>
    </Panel>
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

// ⑤ Known issues: those matching the selection (or unconditional) first, the
// rest folded under "Other configurations".
export function Troubleshooting({ hfId, hardware, troubleshooting }) {
  const sel = useSelection(hfId);
  if (!troubleshooting.length) return null;
  const mine = sel ? troubleshooting.filter((t) => whenMatches(t.when, sel, hardware)) : troubleshooting;
  const other = sel ? troubleshooting.filter((t) => !whenMatches(t.when, sel, hardware)) : [];
  return (
    <Panel id="troubleshooting" title="Troubleshooting" aside={sel ? `${mine.length} for this configuration` : null}>
      <div className="space-y-2">
        {mine.map((t) => <Issue key={t.id} t={t} />)}
        {!mine.length && <p className="text-sm text-muted-foreground">No known issues for this configuration.</p>}
      </div>
      {other.length > 0 && (
        <details className="group/other mt-3">
          <summary className="flex cursor-pointer list-none items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground [&::-webkit-details-marker]:hidden">
            <Chevron className="w-3.5 h-3.5 -rotate-90 group-open/other:rotate-0" />
            Other configurations ({other.length})
          </summary>
          <div className="mt-2 space-y-2">{other.map((t) => <Issue key={t.id} t={t} />)}</div>
        </details>
      )}
    </Panel>
  );
}
