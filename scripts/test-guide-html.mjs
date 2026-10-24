// Guides from source-repo PRs render unreviewed on preview URLs, so the guide
// renderer must never emit markup or URLs a guide author controls beyond plain
// Markdown. Runs the page's react-markdown configuration (src/lib/guide-markdown.js),
// which also renders preset runbooks, notes and troubleshooting bodies (with
// prefixed heading ids).
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import Markdown from "react-markdown";
import { guideRehypePlugins, guideRehypePluginsFor, guideRemarkPlugins } from "../src/lib/guide-markdown.js";

const render = (guide, rehypePlugins = guideRehypePlugins) => renderToStaticMarkup(
  createElement(Markdown, { remarkPlugins: guideRemarkPlugins, rehypePlugins }, guide),
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

for (const [name, guide] of Object.entries(ATTACKS)) for (const prefix of ["", "preset-x-"]) {
  test(`guide markup stays data: ${name}${prefix ? " (prefixed ids)" : ""}`, () => {
    const html = render(guide, guideRehypePluginsFor(prefix));
    assert.doesNotMatch(html, /<(script|iframe|style|object|embed)\b/i, html);
    // Raw HTML may come out as escaped text (&lt;img …&gt;); only real tags matter.
    for (const tag of html.match(/<[a-z][^>]*>/gi) || []) {
      assert.doesNotMatch(tag, /\son[a-z]+=/i, html);
    }
    // Browsers drop whitespace and control characters inside a URL scheme.
    for (const [, url] of html.matchAll(/\s(?:href|src)="([^"]*)"/gi)) {
      const scheme = decodeEntities(url).replace(/[\u0000-\u0020\u007f]/g, "").toLowerCase();
      assert.doesNotMatch(scheme, /^(javascript|data|vbscript):/, html);
    }
  });
}

function decodeEntities(s) {
  return s
    .replace(/&#x([0-9a-f]+);?/gi, (_, h) => String.fromCodePoint(parseInt(h, 16)))
    .replace(/&#(\d+);?/g, (_, d) => String.fromCodePoint(Number(d)))
    .replace(/&(amp|quot|apos|lt|gt|tab|newline);/gi, (_, n) => ({
      amp: "&", quot: '"', apos: "'", lt: "<", gt: ">", tab: "\t", newline: "\n",
    })[n.toLowerCase()]);
}

// Control characters inside a scheme: micromark percent-encodes them
// (java%09script:), and react-markdown's defaultUrlTransform then blanks the
// URL. This pins the second layer, which the page relies on.
const CONTROL_CHAR_SCHEMES = {
  "tab": "[x](<java\tscript:alert(1)>)",
  "entity tab": "[x](java&#9;script:alert(1))",
  "entity newline": "[x](java&#x0A;script:alert(1))",
  "named entity tab": "[x](java&Tab;script:alert(1))",
};

for (const [name, guide] of Object.entries(CONTROL_CHAR_SCHEMES)) {
  test(`defaultUrlTransform blanks a javascript: scheme split by control characters (${name})`, () => {
    assert.equal(render(guide), '<p><a href="">x</a></p>');
  });
}

test("every Markdown renderer in src/ uses the guide configuration", () => {
  const files = fs.readdirSync("src", { recursive: true })
    .filter((f) => /\.(jsx?|mjs|tsx?)$/.test(f))
    .map((f) => path.join("src", f));
  const users = files.filter((f) => fs.readFileSync(f, "utf8").includes("react-markdown"));
  assert.ok(users.length > 0, "no react-markdown user found; update this test");
  for (const f of users.filter((f) => !f.endsWith(path.join("lib", "guide-markdown.js")))) {
    assert.match(fs.readFileSync(f, "utf8"), /from "@\/lib\/guide-markdown"/, `${f} renders Markdown without src/lib/guide-markdown.js`);
  }
});

test("prefixed heading ids for runbooks, notes and troubleshooting", () => {
  assert.match(render("## Services", guideRehypePluginsFor("preset-gb200-")), /<h2 id="preset-gb200-services">Services<\/h2>/);
});

test("plain Markdown still renders", () => {
  const html = render("## Setup\n\n**Run** [the docs](https://docs.vllm.ai) and `vllm serve`.");
  assert.match(html, /<h2 id="setup">Setup<\/h2>/);
  assert.match(html, /<strong>Run<\/strong>/);
  assert.match(html, /<a href="https:\/\/docs\.vllm\.ai">the docs<\/a>/);
  assert.match(html, /<code>vllm serve<\/code>/);
});
