/** @type {import('next').NextConfig} */
const nextConfig = {
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
