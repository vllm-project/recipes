// Guides from source-repo PRs render unreviewed on preview URLs, so the guide
// renderer must never emit markup or URLs a guide author controls beyond plain
// Markdown. Runs the page's react-markdown configuration (src/lib/guide-markdown.js).
import assert from "node:assert/strict";
import { test } from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import Markdown from "react-markdown";
import { guideRehypePlugins, guideRemarkPlugins } from "../src/lib/guide-markdown.js";

const render = (guide) => renderToStaticMarkup(
  createElement(Markdown, { remarkPlugins: guideRemarkPlugins, rehypePlugins: guideRehypePlugins }, guide),
);

const ATTACKS = {
  "script tag": "<script>alert(1)</script>",
  "inline event handler": '<img src="x" onerror="alert(1)">',
  "iframe": '<iframe src="https://example.com"></iframe>',
  "style tag": "<style>body{display:none}</style>",
  "raw anchor": '<a href="javascript:alert(1)">x</a>',
  "javascript: link": "[x](javascript:alert(1))",
  "mixed-case javascript: link": "[x](JaVaScRiPt:alert(1))",
  "javascript: image": "![x](javascript:alert(1))",
  "data: link": "[x](data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==)",
  "autolink": "<javascript:alert(1)>",
  "html in a table cell": "| a |\n|---|\n| <script>alert(1)</script> |",
};

for (const [name, guide] of Object.entries(ATTACKS)) {
  test(`guide markup stays data: ${name}`, () => {
    const html = render(guide);
    assert.doesNotMatch(html, /<(script|iframe|style|object|embed)\b/i, html);
    // Raw HTML may come out as escaped text (&lt;img …&gt;); only real tags matter.
    for (const tag of html.match(/<[a-z][^>]*>/gi) || []) {
      assert.doesNotMatch(tag, /\son[a-z]+=/i, html);
    }
    assert.doesNotMatch(html, /(href|src)="\s*(javascript|data|vbscript):/i, html);
  });
}

test("plain Markdown still renders", () => {
  const html = render("## Setup\n\n**Run** [the docs](https://docs.vllm.ai) and `vllm serve`.");
  assert.match(html, /<h2 id="setup">Setup<\/h2>/);
  assert.match(html, /<strong>Run<\/strong>/);
  assert.match(html, /<a href="https:\/\/docs\.vllm\.ai">the docs<\/a>/);
  assert.match(html, /<code>vllm serve<\/code>/);
});
