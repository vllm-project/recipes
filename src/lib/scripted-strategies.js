// Fixed, documented deployments are separate from the tunable synthesizer.
// Named scripts travel with the recipe YAML through source sync, overlays and
// assembly. Prose is not a command source. Keep the old fence reader only for
// recipes authored against the earlier section/block contract.
export function guideCodeBlock(guide, section, index) {
  const lines = String(guide || "").split("\n");
  let active = false, level = 0, fence = null, body = [], blocks = [];
  for (const line of lines) {
    const marker = line.match(/^( *)```(\w*)\s*$/);
    if (fence !== null) {
      if (marker && marker[1] === fence && !marker[2]) {
        if (active) blocks.push(body.join("\n"));
        fence = null; body = [];
      } else body.push(line.startsWith(fence) ? line.slice(fence.length) : line);
      continue;
    }
    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      if (active && heading[1].length <= level) break;
      if (heading[2] === section) { active = true; level = heading[1].length; }
    }
    if (marker) { fence = marker[1]; body = []; }
  }
  if (blocks[index] === undefined) throw new Error(`Missing script ${index} in guide section: ${section}`);
  return blocks[index].trimEnd();
}

export function resolveDeploymentScript(recipe, hardware, ref) {
  if (ref?.script !== undefined) {
    const script = recipe.deployment_scripts?.[hardware]?.[ref.script];
    if (typeof script !== "string" || !script.trim()) {
      throw new Error(`Missing deployment script: ${hardware}/${ref.script}`);
    }
    return script.trimEnd();
  }
  return guideCodeBlock(recipe.guide, ref?.section, ref?.block);
}

export function resolveLaunchScript(recipe, hardware, step) {
  if (step.launch_script !== undefined) return resolveDeploymentScript(recipe, hardware, { script: step.launch_script });
  return step.launch_block === undefined ? "" : guideCodeBlock(recipe.guide, step.section, step.launch_block);
}

export function resolveContainerSetup(recipe, hardware) {
  const setup = recipe.container_setup?.[hardware];
  if (!setup) return null;
  if (typeof setup === "string") return { guideHref: setup };
  return {
    description: setup.description,
    command: resolveDeploymentScript(recipe, hardware, setup),
    ...(setup.container_script ? { containerCommand: resolveDeploymentScript(recipe, hardware, { script: setup.container_script }) } : {}),
  };
}

export function resolveScriptedStrategy(recipe, strategy, hardware) {
  const preset = recipe.scripted_strategies?.[strategy]?.[hardware];
  if (!preset) return null;
  const exports = Object.entries(preset.env || {}).map(([key, value]) =>
    `export ${key}='${String(value).replaceAll("'", "'\\''")}'`).join("\n");
  return {
    ...preset,
    deploy_type: "scripted",
    strategy,
    hardware,
    steps: preset.steps.map(step => {
      const script = resolveDeploymentScript(recipe, hardware, step);
      const launch = resolveLaunchScript(recipe, hardware, step);
      if (script.split("\n").includes("RECIPE_SCRIPT_EOF")) throw new Error("Reserved script delimiter");
      return {
        label: step.label,
        command: (exports ? `${exports}\n\n` : "") + (step.filename
          ? `cat > ${step.filename} <<'RECIPE_SCRIPT_EOF'\n${script}\nRECIPE_SCRIPT_EOF${launch ? `\n\n${launch}` : ""}`
          : script),
      };
    }),
  };
}
