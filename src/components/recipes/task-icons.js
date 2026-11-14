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

const OUTPUT_NOUN = { image: "images", video: "video", audio: "audio" };

function joinAnd(items) {
  if (items.length <= 1) return items.join("");
  return `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
}

/**
 * Tooltip for a card's kind chip: what the kind means for someone about to
 * serve it, i.e. what goes in, what comes out, and which endpoint or binary.
 * `tasks` / `omniOutputs` make it specific to the recipe where the data allows.
 */
export function kindTooltip(kind, { tasks = [], omniOutputs = [] } = {}) {
  if (kind === "text") {
    return {
      title: "Text",
      body: "Text in, text out: chat, reasoning, code and tool calling through /v1/chat/completions.",
    };
  }
  if (kind === "multimodal") {
    return {
      title: "Multimodal",
      body: "Takes images (and on some models video or audio) together with text, and answers in text through /v1/chat/completions.",
    };
  }
  if (kind === "embedding") {
    const alsoImages = tasks.includes("multimodal") ? " Accepts images as well as text." : "";
    return {
      title: "Embedding",
      body: `Returns vectors or scores rather than generated text: embedding, reranking or classification, through the pooling endpoints (/v1/embeddings, /v1/rerank, /score, /classify), not chat.${alsoImages}`,
    };
  }
  if (kind === "omni") {
    const nouns = omniOutputs.map((o) => OUTPUT_NOUN[o] || o);
    const makes = nouns.length ? `Generates ${joinAnd(nouns)}. ` : "";
    return {
      title: "Omni",
      body: `${makes}Served by vLLM-Omni, vLLM's companion for generation models (vllm serve --omni).`,
    };
  }
  return null;
}
