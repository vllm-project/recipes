"use client";

import { useEffect, useRef, useState } from "react";
import { poolDeploymentMarkdown, poolPlacementLabel } from "@/lib/recipe-markdown";

const control = "rounded-md border border-border px-2.5 py-1.5 text-xs hover:bg-muted/60 disabled:opacity-40";

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
  return <button type="button" className={control} onClick={copy} disabled={disabled}>{message || children}</button>;
}

function InputGroup({ title, fields, selection }) {
  if (!fields.length) return null;
  return <fieldset className="grid gap-3 sm:grid-cols-2">
    <legend className="mb-2 text-sm font-semibold">{title}</legend>
    {fields.map((field) => <label key={field.key} className="flex min-w-0 flex-col gap-1 text-xs">
      <span>{field.label}</span>
      {field.type === "boolean" ? <select className="rounded-md border border-border bg-background px-2 py-1.5"
        value={String(selection.poolInputs?.[field.key] ?? (field.missing ? "" : field.value))}
        onChange={(e) => selection.changePoolInput(field.key, e.target.value)}>
        <option value="">Choose</option><option value="false">False</option><option value="true">True</option>
      </select> : <input className="rounded-md border border-border bg-background px-2 py-1.5 font-mono"
        type="text" autoComplete="off" spellCheck={false} aria-invalid={Boolean(field.error)}
        placeholder={field.key} value={selection.poolInputs?.[field.key] ?? (field.missing ? "" : field.value ?? "")}
        onChange={(e) => selection.changePoolInput(field.key, e.target.value)} />}
      {field.error && <span role="alert" className="text-amber-700">{field.error}</span>}
    </label>)}
  </fieldset>;
}

export function PoolDeployment({ selection }) {
  const plan = selection.poolPlan;
  const [host, setHost] = useState("");
  const [checked, setChecked] = useState({});
  const previous = useRef(new Map());
  useEffect(() => {
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
  }, [plan]);
  const number = new Map(plan.steps.map((step, index) => [step.id, index + 1]));
  return <div className="space-y-5 px-5 py-4">
    <div>
      <h3 className="text-base font-semibold">{plan.label}</h3>
      {plan.ascend_version && <p className="mt-1 text-xs text-muted-foreground">vLLM-Ascend {plan.ascend_version}
        {/^(https?:\/\/)/i.test(plan.tutorial || "") && <> · <a className="text-vllm-blue underline" href={plan.tutorial}>Model tutorial</a></>}
      </p>}
      {selection.poolImage && <p className="mt-1 break-all font-mono text-xs">{selection.poolImage}</p>}
      <ul className="mt-2 space-y-1 text-sm">{plan.hosts.map((id) => {
        const containers = Object.entries(plan.containers).filter(([, c]) => c.host === id);
        const budgets = (plan.capacity || []).filter((row) => row.host === id);
        return <li key={id}><strong>{id}</strong> {containers.map(([c]) => <span key={c} className="ml-2">
          {c} · {plan.placements.filter((p) => p.container === c).map(poolPlacementLabel).join(", ")}
        </span>)}
          {budgets.length > 0 && <ul className="ml-4 list-disc text-xs text-muted-foreground">
            {budgets.map((b) => <li key={b.label}>{b.label}: {b.total} {b.unit} ({b.count} × {b.per_instance} {b.unit})</li>)}
          </ul>}
        </li>;
      })}</ul>
    </div>
    <fieldset><legend className="mb-2 text-sm font-semibold">Pool mode</legend>
      <div className="flex gap-2">{[["new", "Create a pool"], ["reuse", "Use an existing pool"]].map(([value, label]) =>
        <button type="button" key={value} aria-pressed={plan.mode === value} className={`${control} aria-pressed:border-vllm-blue aria-pressed:text-vllm-blue`}
          onClick={() => selection.changePoolMode(value)}>{label}</button>)}</div>
    </fieldset>
    {!plan.mode ? <p className="text-sm text-muted-foreground">Choose a mode to enter site values and follow the steps. The template export includes both modes.</p> : <>
      <InputGroup title="Cluster inputs" fields={plan.inputs.filter((f) => !f.host)} selection={selection} />
      {plan.hosts.map((id) => <InputGroup key={id} title={`Inputs for ${id}`} fields={plan.inputs.filter((f) => f.host === id)} selection={selection} />)}
      <div className="flex flex-wrap items-center gap-2">
        <label className="text-sm">Steps for <select className={control} value={host} onChange={(e) => setHost(e.target.value)}>
          <option value="">All hosts</option>{plan.hosts.map((id) => <option key={id}>{id}</option>)}
        </select></label>
        {host && <CopyAction getText={() => poolDeploymentMarkdown(plan, host)}>Copy this host’s steps</CopyAction>}
      </div>
      <ol className="space-y-4">{plan.steps.filter((s) => !host || s.host === host).map((step) => {
        const incomplete = !!(step.missing?.length || step.errors?.length);
        return <li key={step.id} id={`pool-step-${step.id}`} className="scroll-mt-24 rounded-lg border border-border p-3">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <h4 className="font-semibold">{number.get(step.id)}. {step.title}</h4>
            <label className="flex items-center gap-1.5 text-xs"><input type="checkbox" disabled={incomplete}
              checked={!!checked[step.id]} onChange={(e) => setChecked({ ...checked, [step.id]: e.target.checked })}/>Checked</label>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">{[step.stage, step.host, step.container, step.role].filter(Boolean).join(" · ")}</p>
          {!!step.depends_on?.length && <p className="mt-1 text-xs">After steps {step.depends_on.map((id) => number.get(id) ?? id).join(", ")}</p>}
          {step.command && <pre className="my-2 overflow-x-auto rounded-md bg-muted/50 p-3 text-xs"><code>{step.command}</code></pre>}
          {step.missing?.length > 0 && <p className="my-2 text-xs text-amber-700">Fill in {step.missing.join(", ")}</p>}
          {step.errors?.map((msg) => <p key={msg} className="my-2 text-xs text-amber-700">{msg}</p>)}
          {step.command && <CopyAction getText={() => step.command} disabled={incomplete}>Copy command</CopyAction>}
          {step.check && <p className="mt-2 text-sm"><strong>Check</strong> {step.check}</p>}
        </li>;
      })}</ol>
    </>}
  </div>;
}
