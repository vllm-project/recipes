// Restrained shell highlighting for the command blocks and the guide's bash
// fences. Pure functions, no DOM — safe in server and client components.
//
// The palette is deliberately small: flags, strings and $VARs get a quiet tint,
// comments dim, command words gain weight, and only the two things a reader
// scans for — the served model id and the container image — get the accent.
// The output is a flat list of `{ text, kind }` segments whose texts concatenate
// back to the input byte-for-byte, so copy/paste is never affected.

// docker-run flags that take no value; every other flag without `=` consumes
// the next token, which is how the image is told apart from flag values.
const DOCKER_BOOL_FLAGS = new Set([
  "--rm", "-it", "-ti", "-i", "-t", "-d", "--privileged", "--init",
  "--detach", "--interactive", "--tty", "--read-only",
]);

// Flags whose value is a model id, on any command.
const MODEL_FLAGS = new Set(["--model", "--tokenizer", "--served-model-name"]);

const SEPARATORS = new Set(["&&", "||", "|", ";"]);

const MODEL_LIKE = /^[\w.-]+\/[\w.-]+$|^\/[\w./-]+$/;

// Split into whitespace / comment / word tokens. A word runs to the next
// unquoted whitespace, so `--x='{"a": 1}'` and `KEY="a b"` stay one word.
function lex(src) {
  const toks = [];
  let i = 0;
  const n = src.length;
  while (i < n) {
    const c = src[i];
    if (c === " " || c === "\t" || c === "\n" || c === "\r") {
      let j = i;
      while (j < n && " \t\n\r".includes(src[j])) j++;
      toks.push({ type: "ws", text: src.slice(i, j) });
      i = j;
      continue;
    }
    if (c === "#") {
      let j = src.indexOf("\n", i);
      if (j < 0) j = n;
      toks.push({ type: "comment", text: src.slice(i, j) });
      i = j;
      continue;
    }
    let j = i;
    while (j < n && !" \t\n\r".includes(src[j])) {
      const q = src[j];
      if (q === "'" || q === '"') {
        const end = src.indexOf(q, j + 1);
        j = end < 0 ? n : end + 1;
      } else if (q === "\\" && j + 1 < n && src[j + 1] !== "\n") {
        j += 2;
      } else {
        j++;
      }
    }
    toks.push({ type: "word", text: src.slice(i, j) });
    i = j;
  }
  return toks;
}

// Split a word's text into string / var / plain pieces; `base` colors the
// unquoted, non-variable parts. Inside strings, a JSON `"model": "<id>"` value
// (curl payloads, --speculative-config) is lifted to the model accent.
function splitWord(text, base, out) {
  const re = /'[^']*'?|"(?:[^"\\]|\\.)*"?|\$\{[^}]*\}|\$\w+/g;
  let last = 0;
  let m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push({ text: text.slice(last, m.index), kind: base });
    const piece = m[0];
    if (piece[0] === "$") out.push({ text: piece, kind: "var" });
    else splitString(piece, out);
    last = m.index + piece.length;
  }
  if (last < text.length) out.push({ text: text.slice(last), kind: base });
}

function splitString(piece, out) {
  const re = /("model"\s*:\s*")([^"]+)(")/g;
  let last = 0;
  let m;
  while ((m = re.exec(piece))) {
    const valueStart = m.index + m[1].length;
    out.push({ text: piece.slice(last, valueStart), kind: "string" });
    out.push({ text: m[2], kind: "model" });
    last = valueStart + m[2].length;
  }
  out.push({ text: piece.slice(last), kind: "string" });
}

function stripQuotes(s) {
  return s.replace(/^(['"])(.*)\1$/, "$2");
}

export function highlightShell(src) {
  const out = [];
  // Per simple-command state: `pos` counts words since the command started.
  let cmd = null;
  let atStart = true;
  let expect = null; // what the next word is: "value" | "model" | "image" | null
  let afterImage = false;
  let continued = false; // last word was a `\` line continuation

  const reset = () => {
    atStart = true;
    cmd = null;
    expect = null;
    afterImage = false;
  };

  for (const tok of lex(String(src ?? ""))) {
    if (tok.type === "ws") {
      out.push({ text: tok.text, kind: "plain" });
      if (tok.text.includes("\n") && !continued) reset();
      if (tok.text.includes("\n")) continued = false;
      continue;
    }
    if (tok.type === "comment") {
      out.push({ text: tok.text, kind: "comment" });
      continue;
    }
    const w = tok.text;
    continued = w === "\\";

    if (continued) {
      out.push({ text: w, kind: "punct" });
      continue;
    }
    if (SEPARATORS.has(w)) {
      out.push({ text: w, kind: "punct" });
      reset();
      continue;
    }

    if (atStart) {
      // Leading `KEY=value` assignments and `export`/`sudo` keep us at the
      // command position; the first other word is the command name.
      if (/^[A-Za-z_]\w*=/.test(w)) {
        const eq = w.indexOf("=");
        out.push({ text: w.slice(0, eq), kind: "var" });
        out.push({ text: "=", kind: "punct" });
        splitWord(w.slice(eq + 1), "plain", out);
        continue;
      }
      out.push({ text: w, kind: "command" });
      if (w !== "export" && w !== "sudo" && w !== "env") {
        cmd = { name: w, sub: null, pos: 0 };
        atStart = false;
      }
      continue;
    }

    cmd.pos++;
    const sub = cmd.sub;

    // Subcommands that read as part of the command name.
    if (cmd.pos === 1 && !w.startsWith("-")) {
      const two = `${cmd.name} ${w}`;
      if (["vllm serve", "vllm bench", "docker run", "docker pull", "hf download",
        "huggingface-cli download", "vllm-omni serve"].includes(two)) {
        cmd.sub = w;
        out.push({ text: w, kind: "command" });
        expect = two === "docker run" ? "image" : (two === "docker pull" ? "image-now" : (w === "bench" ? null : "model-now"));
        continue;
      }
    }

    if (w.startsWith("-") && w.length > 1 && !/^-\d/.test(w)) {
      const eq = w.indexOf("=");
      const name = eq < 0 ? w : w.slice(0, eq);
      out.push({ text: name, kind: "flag" });
      if (eq >= 0) {
        out.push({ text: "=", kind: "flag" });
        const value = w.slice(eq + 1);
        if (MODEL_FLAGS.has(name) && MODEL_LIKE.test(stripQuotes(value))) out.push({ text: value, kind: "model" });
        else splitWord(value, "plain", out);
      } else if (MODEL_FLAGS.has(name)) {
        expect = expect === "image" ? "image" : "model-flag";
      } else if (cmd.name === "docker" && sub === "run" && expect === "image" && !DOCKER_BOOL_FLAGS.has(name)) {
        expect = "image-value";
      }
      continue;
    }

    // Positional word.
    if (expect === "image-value") {
      expect = "image";
      splitWord(w, "plain", out);
      continue;
    }
    if (expect === "image" || expect === "image-now") {
      out.push({ text: w, kind: "image" });
      expect = null;
      afterImage = true;
      continue;
    }
    if (afterImage) {
      // The vLLM image's entrypoint is `vllm serve`, so the word right after
      // the image is the model.
      afterImage = false;
      if (MODEL_LIKE.test(w)) {
        out.push({ text: w, kind: "model" });
        continue;
      }
    }
    if ((expect === "model-now" || expect === "model-flag") && MODEL_LIKE.test(stripQuotes(w))) {
      out.push({ text: w, kind: "model" });
      expect = null;
      continue;
    }
    expect = null;
    splitWord(w, "plain", out);
  }
  return out;
}

// Fence languages the guide treats as shell.
export const SHELL_LANGS = new Set(["bash", "sh", "shell", "zsh", "console"]);
