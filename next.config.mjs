import fs from "fs";
import { SELECTION_PARAMS } from "./src/lib/selection-params.js";

// Preview branches of source-repo PRs (scripts/preview-prs.mjs) carry
// public/preview.json; PreviewBanner shows which PR commit the site renders.
const previewFile = "public/preview.json";
const preview = fs.existsSync(previewFile) ? fs.readFileSync(previewFile, "utf8") : "";

/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    RECIPES_PREVIEW: preview,
  },
  images: {
    unoptimized: true,
  },
  // Recipe pages that render on demand (dynamicParams) read the assembled
  // content tree at runtime; its path is computed (src/lib/content-root.js),
  // so the tracer can't discover it on its own.
  outputFileTracingIncludes: {
    "/**/*": ["./.content/**/*", "./sources.yaml"],
  },
  outputFileTracingExcludes: {
    "/**/*": ["./.content/.tmp-*/**", "./.content/.old-*/**", "./.content/validate/**"],
  },
  // A recipe page URL's agent-readable twins, rendered on demand for the same
  // query string: `/<org>/<repo>.md?…`, and `/<org>/<repo>.json?…` when the
  // query selects a configuration (a bare `.json` stays the static
  // whole-recipe file in public/, so those rules run before the filesystem).
  async rewrites() {
    const render = (format) => `/api/render/${format}/:org/:repo`;
    return {
      beforeFiles: SELECTION_PARAMS.map((key) => ({
        source: "/:org/:repo.json",
        has: [{ type: "query", key }],
        destination: render("json"),
      })),
      afterFiles: [{ source: "/:org/:repo.md", destination: render("md") }],
    };
  },
};

export default nextConfig;
