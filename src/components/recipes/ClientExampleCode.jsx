"use client";
import { useSelection } from "@/lib/selection-store";
import { fillClientModel } from "@/lib/client-model";

export default function ClientExampleCode({ hfId, defaultModel, className, children }) {
  const selection = useSelection(hfId);
  return <code className={className}>{fillClientModel(String(children), selection?.clientModel || defaultModel)}</code>;
}
