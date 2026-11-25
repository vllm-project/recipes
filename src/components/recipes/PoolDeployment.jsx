"use client";

import { useEffect, useRef, useState } from "react";
import { poolDeploymentMarkdown, poolPlacementLabel } from "@/lib/recipe-markdown";

const control = "rounded-md border border-border bg-background px-2.5 py-1.5 text-xs hover:bg-muted/60 disabled:opacity-40";
const link = "text-vllm-blue underline underline-offset-2";
const stepAnchor = (id) => "pool-step-" + id;

function CopyAction({ getText, disabled, children }) {
  const [message, setMessage] = useState("");
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(getText());
      setMessage("Copied");
    } catch {
      setMessage("Copy failed");
    }
  };
  useEffect(() => {
    if (!message) return;
    const timeout = setTimeout(() => setMessage(""), 2000);
    return () => clearTimeout(timeout);
  }, [message]);
  return <button type="button" className={control} onClick={copy} disabled={disabled} aria-live="polite">{message || children}</button>;
}

function InputGroup({ title, fields, poolInputs, changePoolInput }) {
  if (!fields.length) return null;
  const pending = fields.filter((f) => f.missing || f.error).length;
  return <details className="min-w-0 rounded-md border border-border">
    <summary className="cursor-pointer px-3 py-2 text-sm font-medium">
      {title}<span className="ml-2 text-xs font-normal text-muted-foreground">{pending ? pending + " to complete" : "Ready"}</span>
    </summary>
    <div className="grid min-w-0 gap-3 border-t border-border p-3 sm:grid-cols-2">
      {fields.map((field) => <label key={field.key} className="flex min-w-0 flex-col gap-1 text-xs">
        <span>{field.label}</span>
        {field.type === "boolean" ? <select className={control}
          value={String(poolInputs?.[field.key] ?? (field.missing ? "" : field.value))}
          aria-invalid={Boolean(field.error)} onChange={(e) => changePoolInput(field.key, e.target.value)}>
          <option value="">Choose</option><option value="false">False</option><option value="true">True</option>
        </select> : <input className="min-w-0 rounded-md border border-border bg-background px-2 py-1.5 font-mono"
          type="text" autoComplete="off" spellCheck={false} aria-invalid={Boolean(field.error)}
          placeholder={field.key} value={poolInputs?.[field.key] ?? (field.missing ? "" : field.value ?? "")}
          onChange={(e) => changePoolInput(field.key, e.target.value)} />}
        {field.error && <span role="alert" className="text-amber-700">{field.error}</span>}
      </label>)}
    </div>
  </details>;
}

export function PoolConfiguration({ plan, image, poolInputs, changePoolInput, changePoolMode }) {
  if (!plan) return null;
  return <div className="min-w-0 space-y-3">
    <fieldset><legend className="mb-1.5 text-sm font-semibold">Pool mode</legend>
      <div className="flex flex-wrap gap-2">{[["new", "Create a pool"], ["reuse", "Use an existing pool"]].map(([value, label]) =>
        <button type="button" key={value} aria-pressed={plan.mode === value} className={control + " aria-pressed:border-vllm-blue aria-pressed:text-vllm-blue"}
          onClick={() => changePoolMode(value)}>{label}</button>)}</div>
    </fieldset>
    <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
      <span>{plan.hosts.length} {plan.hosts.length === 1 ? "host" : "hosts"}</span>
      {plan.ascend_version && <span>vLLM-Ascend {plan.ascend_version}</span>}
      {/^(https?:\/\/)/i.test(plan.tutorial || "") && <a className={link} href={plan.tutorial}>Model tutorial</a>}
    </div>
    <details className="text-xs">
      <summary className="cursor-pointer font-medium">Topology and image</summary>
      {image && <p className="mt-2 break-all font-mono">{image}</p>}
      <ul className="mt-2 space-y-2">{plan.hosts.map((id) => {
        const containers = Object.entries(plan.containers).filter(([, c]) => c.host === id);
        const budgets = (plan.capacity || []).filter((row) => row.host === id);
        return <li key={id}><strong>Host {id}</strong>
          {containers.map(([name]) => <p key={name} className="mt-0.5 break-words">
            Container {name} · {plan.placements.filter((p) => p.container === name).map(poolPlacementLabel).join(", ")}
          </p>)}
          {budgets.map((b) => <p key={b.label} className="text-muted-foreground">{b.label}: {b.total} {b.unit} ({b.count} × {b.per_instance} {b.unit})</p>)}
        </li>;
      })}</ul>
    </details>
    {plan.mode ? <div className="space-y-2">
      <InputGroup title="Cluster inputs" fields={plan.inputs.filter((f) => !f.host)} poolInputs={poolInputs} changePoolInput={changePoolInput} />
      {plan.hosts.map((id) => <InputGroup key={id} title={"Inputs for " + id} fields={plan.inputs.filter((f) => f.host === id)} poolInputs={poolInputs} changePoolInput={changePoolInput} />)}
      <a href="#pool-commands" className={"inline-block text-sm font-medium " + link}>View deployment commands</a>
    </div> : <p className="text-xs text-muted-foreground">Choose a mode to enter site values. The template export includes both modes.</p>}
  </div>;
}

export function PoolCommands({ plan }) {
  return <section id="pool-commands" className="min-w-0 scroll-mt-24 space-y-3 p-3 text-[var(--command-fg)] sm:p-4" aria-labelledby="pool-commands-title">
    <h3 id="pool-commands-title" className="text-sm font-semibold">Pool deployment commands</h3>
    {!plan ? <p role="alert" className="text-sm">The selected pool deployment could not be rendered.</p> :
      !plan.mode ? <a className={"text-sm " + link} href="#deployment">Choose a pool mode to view commands</a> :
      <HostCommands plan={plan} />}
  </section>;
}

function HostCommands({ plan }) {
  const [active, setActive] = useState({ host: "", stepId: "" });
  const [checked, setChecked] = useState({});
  const previous = useRef(new Map());
  const currentSteps = useRef(plan.steps);

  useEffect(() => {
    currentSteps.current = plan.steps;
    const signatures = new Map();
    const changed = new Set();
    for (const step of plan.steps) {
      const value = JSON.stringify([step.command, step.check, step.missing, step.errors, step.depends_on]);
      signatures.set(step.id, value);
      if (value !== previous.current.get(step.id) || step.depends_on?.some((id) => changed.has(id))) changed.add(step.id);
    }
    previous.current = signatures;
    if (changed.size) setChecked((before) =>
      Object.fromEntries(Object.entries(before).filter(([id]) => signatures.has(id) && !changed.has(id))));
  }, [plan.steps]);

  useEffect(() => {
    let frame;
    function followHash() {
      cancelAnimationFrame(frame);
      let id;
      try { id = decodeURIComponent(window.location.hash.slice(1)); } catch { return; }
      const target = currentSteps.current.find((s) => stepAnchor(s.id) === id);
      if (!target) return;
      setActive({ host: target.host, stepId: target.id });
      // The target can belong to another host and enters the DOM after selection.
      frame = requestAnimationFrame(() => {
        frame = requestAnimationFrame(() => document.getElementById(id)?.scrollIntoView({ block: "start" }));
      });
    }
    followHash();
    window.addEventListener("hashchange", followHash);
    return () => { window.removeEventListener("hashchange", followHash); cancelAnimationFrame(frame); };
  }, []);

  const visible = plan.steps.filter((s) => !active.host || s.host === active.host);
  const index = Math.max(0, visible.findIndex((s) => s.id === active.stepId));
  const step = visible[index];
  const number = new Map(plan.steps.map((s, i) => [s.id, i + 1]));
  const incomplete = Boolean(step?.missing?.length || step?.errors?.length);

  function choose(host, stepId) {
    setActive({ host, stepId });
    // Change the shareable anchor without moving focus out of the selector.
    const url = new URL(window.location.href);
    url.hash = stepId ? stepAnchor(stepId) : "pool-commands";
    window.history.replaceState(window.history.state, "", url);
  }

  return <>
    <div className="grid min-w-0 gap-2 sm:grid-cols-[minmax(7rem,1fr)_minmax(0,3fr)]">
      <label className="grid min-w-0 gap-1 text-xs font-medium">Host
        <select className={control + " w-full min-w-0"} value={active.host} onChange={(e) => {
          const host = e.target.value;
          choose(host, plan.steps.find((s) => !host || s.host === host)?.id || "");
        }}>
          <option value="">All hosts</option>
          {plan.hosts.map((id) => <option key={id} value={id}>{id}</option>)}
        </select>
      </label>
      <label className="grid min-w-0 gap-1 text-xs font-medium">Step
        <select className={control + " w-full min-w-0"} value={step?.id || ""} onChange={(e) => choose(active.host, e.target.value)}>
          {visible.map((s) => <option key={s.id} value={s.id}>{number.get(s.id)}. {s.title} · {s.host}{checked[s.id] ? " ✓" : ""}</option>)}
        </select>
      </label>
    </div>
    <div className="flex flex-wrap items-center gap-2">
      <button type="button" className={control} disabled={index === 0} onClick={() => choose(active.host, visible[index - 1].id)}>Previous step</button>
      <button type="button" className={control} disabled={index + 1 >= visible.length} onClick={() => choose(active.host, visible[index + 1].id)}>Next step</button>
      {active.host && <CopyAction key={active.host} getText={() => poolDeploymentMarkdown(plan, active.host)}>Copy this host’s steps</CopyAction>}
    </div>
    {step && <article id={stepAnchor(step.id)} className="min-w-0 scroll-mt-24 rounded-lg border border-border p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h4 className="min-w-0 break-words text-sm font-semibold">{number.get(step.id)}. {step.title}</h4>
        <label className="flex items-center gap-1.5 text-xs"><input type="checkbox" disabled={incomplete}
          checked={!!checked[step.id]} onChange={(e) => {
            const value = e.target.checked;
            setChecked((before) => ({ ...before, [step.id]: value }));
          }}/>Checked</label>
      </div>
      <p className="mt-1 text-xs text-muted-foreground">{[step.stage, "host " + step.host,
        step.container && "container " + step.container, step.role].filter(Boolean).join(" · ")}</p>
      {!!step.depends_on?.length && <div className="mt-1 flex flex-wrap items-baseline gap-x-1 text-xs">
        <span>After steps</span>{step.depends_on.map((id) => <a key={id} className={link} href={"#" + stepAnchor(id)}>{number.get(id) ?? id}</a>)}
      </div>}
      {step.command && <pre className="my-2 max-h-72 max-w-full overflow-auto rounded-md bg-muted/50 p-3 text-xs"><code>{step.command}</code></pre>}
      {step.missing?.length > 0 && <p className="my-2 break-words text-xs text-amber-700">Fill in {step.missing.join(", ")}. <a href="#deployment" className={link}>Edit pool inputs</a></p>}
      {step.errors?.map((msg) => <p key={msg} className="my-2 break-words text-xs text-amber-700">{msg}</p>)}
      {step.command && <CopyAction key={step.id} getText={() => step.command} disabled={incomplete}>Copy command</CopyAction>}
      {step.check && <p className="mt-2 break-words text-sm"><strong>Check</strong> {step.check}</p>}
    </article>}
  </>;
}
