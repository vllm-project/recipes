// One icon + label per recipe task, shared by the homepage cards, /browse,
// the org page and the sidebar. Omni recipes carry the vLLM-Omni ring, the
// only colored mark among them, since vLLM-Omni serves them.
import { Type, Eye, Hash, Image, Film, AudioLines } from "lucide-react";
import { OmniMark } from "@/components/icons/OmniMark";

export const TASK_ICON = { text: Type, multimodal: Eye, omni: OmniMark, embedding: Hash };

export const TASK_LABEL = {
  text: "Text",
  multimodal: "Multimodal",
  omni: "Omni",
  embedding: "Embedding",
};

/** What an omni recipe generates (`omniOutputs`): a tag beside size / precision on cards. */
export const OUTPUT_LABEL = { image: "Image", video: "Video", audio: "Audio" };
export const OUTPUT_ICON = { image: Image, video: Film, audio: AudioLines };

/** "Image · Video": what an omni recipe generates, or "" when unknown. */
export function outputsLabel(omniOutputs = []) {
  return omniOutputs.map((o) => OUTPUT_LABEL[o] || o).join(" · ");
}

/** A list row's kind: its primary task, plus what an omni recipe generates ("Omni · Image"). */
export function kindLabel(kind, omniOutputs = []) {
  const label = TASK_LABEL[kind] || kind;
  const outputs = kind === "omni" ? outputsLabel(omniOutputs) : "";
  return outputs ? `${label} · ${outputs}` : label;
}
