// `meta.hardware.<id>` is a status — verified | supported | unsupported — or,
// to record a verified run, an object:
//   { status: verified, date?, vllm?, image?, note? }
// Every reader compares the status string, so loaders split the object form:
// `meta.hardware` keeps the status and `meta.verification` holds the details.
// Who maintains a configuration is a separate field, `meta.maintained_by`
// (src/lib/maintainers.js).

export function splitHardwareStatus(recipe) {
  const hw = recipe?.meta?.hardware;
  if (!hw || typeof hw !== "object") return recipe;
  const verification = {};
  for (const [id, value] of Object.entries(hw)) {
    if (!value || typeof value !== "object") continue;
    const { status, ...details } = value;
    hw[id] = status;
    if (details.date instanceof Date) details.date = details.date.toISOString().slice(0, 10);
    if (Object.keys(details).length) verification[id] = details;
  }
  if (Object.keys(verification).length) recipe.meta.verification = verification;
  return recipe;
}

export function verificationFor(recipe, hwId) {
  return recipe?.meta?.verification?.[hwId] || null;
}

// Tooltip text: "Verified on 2026-10-15 (vllm/vllm-omni:…)" plus the note.
export function verificationNote(recipe, hwId) {
  const v = verificationFor(recipe, hwId);
  if (!v) return "Verified — author has tested this hardware end-to-end";
  const env = [v.vllm && `vLLM ${v.vllm}`, v.image].filter(Boolean).join(", ");
  const line = `Verified${v.date ? ` on ${v.date}` : ""}${env ? ` (${env})` : ""}`;
  return v.note ? `${line}\n${v.note}` : line;
}
