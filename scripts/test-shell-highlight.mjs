import assert from "node:assert/strict";
import { test } from "node:test";
import { highlightShell, SHELL_LANGS } from "../src/lib/shell-highlight.js";

// The module's one hard promise: the segment texts concatenate back to the
// input byte-for-byte, so copy/paste off a highlighted block is unaffected.
const concat = (segs) => segs.map((s) => s.text).join("");
const textOf = (segs, kind) => segs.filter((s) => s.kind === kind).map((s) => s.text);

// Representative commands spanning every lexer and highlighter path: plain
// flags, quoted args, env assignments, pipes/separators, comments, $VARs,
// JSON payloads, line continuations and a multi-line `vllm serve`.
const CORPUS = [
  "",
  "   ",
  "\n\t \n",
  "ls -la",
  "sleep -5",
  "vllm serve meta-llama/Llama-3.1-8B",
  "vllm serve --model meta-llama/Llama-3.1-8B --tensor-parallel-size 2",
  "vllm serve --model=meta-llama/Llama-3.1-8B",
  "HF_TOKEN=hf_abc vllm serve org/Model-X",
  "export HF_HUB_ENABLE_HF_TRANSFER=1",
  "KEY=\"a b\" cmd --flag='{\"a\": 1}' positional",
  "cat file | grep needle && echo done; echo next || true",
  "vllm serve model  # trailing comment",
  "# a whole-line comment",
  "echo $HOME ${HF_HOME}/cache",
  "docker run --rm -it vllm/vllm-openai:latest meta-llama/Llama-3.1-8B",
  "docker run --rm -v /data:/data vllm/vllm-openai:latest org/Model",
  "docker pull vllm/vllm-openai:latest",
  "hf download meta-llama/Llama-3.1-8B",
  "curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\": \"org/Model-X\", \"messages\": []}'",
  "echo \"unterminated string",
  "echo a\\ b",
  [
    "vllm serve meta-llama/Llama-3.1-8B \\",
    "  --tensor-parallel-size 2 \\",
    "  --max-model-len 4096",
  ].join("\n"),
];

test("segment texts reconstruct the input byte-for-byte", () => {
  for (const src of CORPUS) {
    for (const jsonModel of [true, false]) {
      const out = highlightShell(src, { jsonModel });
      assert.equal(concat(out), src, `concat mismatch (jsonModel=${jsonModel}): ${JSON.stringify(src)}`);
      for (const seg of out) {
        assert.equal(typeof seg.text, "string", `non-string text in ${JSON.stringify(src)}`);
        assert.equal(typeof seg.kind, "string", `non-string kind in ${JSON.stringify(src)}`);
      }
    }
  }
});

test("empty and nullish input produce no segments", () => {
  assert.deepEqual(highlightShell(""), []);
  assert.deepEqual(highlightShell(null), []);
  assert.deepEqual(highlightShell(undefined), []);
  // Whitespace is preserved as a single plain run, never dropped.
  assert.deepEqual(highlightShell("  \n "), [{ text: "  \n ", kind: "plain" }]);
});

test("command words, flags and positionals get distinct kinds", () => {
  const out = highlightShell("ls -la file.txt");
  assert.deepEqual(out, [
    { text: "ls", kind: "command" },
    { text: " ", kind: "plain" },
    { text: "-la", kind: "flag" },
    { text: " ", kind: "plain" },
    { text: "file.txt", kind: "plain" },
  ]);
});

test("a leading dash followed by a digit is a value, not a flag", () => {
  // `-5` matches the -\d guard, so it stays a plain positional.
  const out = highlightShell("sleep -5");
  assert.deepEqual(textOf(out, "flag"), []);
  assert.ok(out.some((s) => s.text === "-5" && s.kind === "plain"));
});

test("the served model id is accented after --model, --model=, and bare `vllm serve`", () => {
  const id = "meta-llama/Llama-3.1-8B";
  assert.deepEqual(textOf(highlightShell(`vllm serve ${id}`), "model"), [id]);
  assert.deepEqual(textOf(highlightShell(`vllm serve --model ${id}`), "model"), [id]);
  assert.deepEqual(textOf(highlightShell(`vllm serve --model=${id}`), "model"), [id]);
  // --served-model-name and --tokenizer are model flags too.
  assert.deepEqual(textOf(highlightShell(`vllm serve --served-model-name ${id}`), "model"), [id]);
});

test("docker run marks the image, then the model that follows it", () => {
  const out = highlightShell("docker run --rm -it vllm/vllm-openai:latest meta-llama/Llama-3.1-8B");
  assert.deepEqual(textOf(out, "image"), ["vllm/vllm-openai:latest"]);
  assert.deepEqual(textOf(out, "model"), ["meta-llama/Llama-3.1-8B"]);
  // A non-boolean flag before the image consumes the next token as its value,
  // so the mount path is NOT mistaken for the image.
  const mounted = highlightShell("docker run --rm -v /data:/data vllm/vllm-openai:latest org/Model");
  assert.deepEqual(textOf(mounted, "image"), ["vllm/vllm-openai:latest"]);
  assert.ok(mounted.some((s) => s.text === "/data:/data" && s.kind !== "image"));
  // docker pull takes an image with no trailing model.
  assert.deepEqual(textOf(highlightShell("docker pull vllm/vllm-openai:latest"), "image"), ["vllm/vllm-openai:latest"]);
});

test("env assignments, separators, comments and $VARs are tinted but preserved", () => {
  const out = highlightShell("HF_TOKEN=hf_abc vllm serve org/Model  # go");
  assert.ok(out.some((s) => s.text === "HF_TOKEN" && s.kind === "var"));
  assert.ok(out.some((s) => s.text === "=" && s.kind === "punct"));
  assert.ok(out.some((s) => s.text === "vllm" && s.kind === "command"));
  assert.ok(out.some((s) => s.text === "# go" && s.kind === "comment"));

  const seps = highlightShell("a | b && c ; d || e");
  assert.deepEqual(textOf(seps, "punct"), ["|", "&&", ";", "||"]);
  // Each separator resets command state, so b/c/d/e are command words again.
  assert.deepEqual(textOf(seps, "command"), ["a", "b", "c", "d", "e"]);

  const vars = highlightShell("echo $HOME ${HF_HOME}/cache");
  assert.deepEqual(textOf(vars, "var"), ["$HOME", "${HF_HOME}"]);
});

test("line continuation keeps one logical command; a bare newline resets it", () => {
  const cont = highlightShell("vllm serve org/Model \\\n  --tensor-parallel-size 2");
  assert.ok(cont.some((s) => s.text === "\\" && s.kind === "punct"));
  // The backslash joins the lines, so `--tensor-parallel-size` is still a flag
  // of the same command (not a new command word).
  assert.ok(cont.some((s) => s.text === "--tensor-parallel-size" && s.kind === "flag"));
  assert.deepEqual(textOf(cont, "command"), ["vllm", "serve"]);

  // Without the backslash, the newline ends the command.
  const reset = highlightShell("vllm serve org/Model\ndocker pull img/x:1");
  assert.deepEqual(textOf(reset, "command"), ["vllm", "serve", "docker", "pull"]);
});

test("a JSON \"model\" value is accented only when jsonModel is on", () => {
  const cmd = "curl http://localhost:8000/v1/chat/completions -d '{\"model\": \"org/Model-X\", \"messages\": []}'";
  const on = highlightShell(cmd);
  assert.ok(on.some((s) => s.text === "org/Model-X" && s.kind === "model"));

  const off = highlightShell(cmd, { jsonModel: false });
  assert.ok(off.some((s) => s.text === "org/Model-X" && s.kind === "string"));
  assert.equal(off.some((s) => s.kind === "model"), false);
  // No `json-model` placeholder kind ever leaks to callers.
  assert.equal(on.some((s) => s.kind === "json-model"), false);
  assert.equal(off.some((s) => s.kind === "json-model"), false);
});

test("SHELL_LANGS covers the fence languages treated as shell", () => {
  for (const lang of ["bash", "sh", "shell", "zsh", "console"]) {
    assert.ok(SHELL_LANGS.has(lang), `missing ${lang}`);
  }
  assert.equal(SHELL_LANGS.has("python"), false);
});
