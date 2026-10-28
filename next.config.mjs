import fs from "fs";

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
};

export default nextConfig;
