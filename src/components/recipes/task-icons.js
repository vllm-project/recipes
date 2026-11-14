// One icon + label per recipe task, shared by the homepage cards, /browse,
// the org page and the sidebar. Omni recipes carry the vLLM-Omni ring, the
// only colored mark among them, since vLLM-Omni serves them.
import { Type, Eye, Hash } from "lucide-react";
import { OmniMark } from "@/components/icons/OmniMark";

export const TASK_ICON = { text: Type, multimodal: Eye, omni: OmniMark, embedding: Hash };

export const TASK_LABEL = {
  text: "Text",
  multimodal: "Multimodal",
  omni: "vLLM-Omni",
  embedding: "Embedding",
};

const OUTPUT_LABEL = { image: "Image", video: "Video", audio: "Audio" };

/**
 * What a card or list row says a recipe is: its primary task, and for omni
 * recipes what it generates ("Image · Video") instead of the word "Omni".
 */
export function kindLabel(kind, omniOutputs = []) {
  if (kind === "omni" && omniOutputs.length) {
    return omniOutputs.map((o) => OUTPUT_LABEL[o] || o).join(" · ");
  }
  return TASK_LABEL[kind] || kind;
}
