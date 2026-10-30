// Server-rendered stand-in for the command builder until it hydrates. The
// builder reads the query string, so it renders on the client only — without
// this, the static HTML (all a non-JS reader such as an AI agent's fetch sees)
// had no command, and the guide's hand-written examples were the only ones
// left to copy. This shows the recipe's default configuration and says where
// any other configuration's command can be read.

function primaryCommand(r) {
  if (!r) return null;
  if (r.deploy_type === "multi_node") return r.head_command;
  if (r.deploy_type === "pd_cluster") return r.prefill?.command;
  if (r.deploy_type === "kv_store_lb") return r.vllm_command;
  return r.command;
}

export function CommandFallback({ hfId, summary, rendering }) {
  const command = primaryCommand(rendering);
  const env = Object.entries(rendering?.env || {}).map(([k, v]) => `export ${k}=${v}`).join("\n");
  return (
    <div className="rounded-2xl border border-border overflow-hidden bg-[var(--command-bg)]">
      <div className="px-4 pt-3 text-xs text-muted-foreground">
        Default configuration: {summary}
      </div>
      {command && (
        <pre className="px-4 py-3 text-[13px] text-[var(--command-fg)] font-mono leading-relaxed whitespace-pre overflow-x-auto">
          {[env, command].filter(Boolean).join("\n")}
        </pre>
      )}
      <p className="px-4 pb-3 text-xs text-muted-foreground">
        The interactive builder loads with JavaScript. Every configuration&apos;s command is also available as
        Markdown at <a className="underline" href={`/${hfId}.md`}>/{hfId}.md</a> — keep the page&apos;s
        query string (e.g. <code className="font-mono">?hardware=b200</code>) to render that configuration.
      </p>
    </div>
  );
}
