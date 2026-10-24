import remarkGfm from "remark-gfm";
import rehypeSlug from "rehype-slug";

// Guides render unreviewed on source-PR previews, so they must stay data: no
// rehype-raw (raw HTML is dropped) and react-markdown's default URL filter.
// scripts/test-guide-html.mjs checks this configuration.
export const guideRemarkPlugins = [remarkGfm];
export const guideRehypePlugins = [rehypeSlug];
