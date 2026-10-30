import { highlightShell } from "@/lib/shell-highlight";

// Renders shell text with the restrained `sh-*` token classes (see globals.css).
// No hooks, so it works in both the client CommandBuilder and the server-
// rendered guide markdown.
export default function ShellCode({ children }) {
  const text = Array.isArray(children) ? children.join("") : String(children ?? "");
  return highlightShell(text).map((seg, i) =>
    seg.kind === "plain"
      ? seg.text
      : <span key={i} className={`sh-${seg.kind}`}>{seg.text}</span>
  );
}
