import remarkGfm from "remark-gfm";
import rehypeSlug from "rehype-slug";

// Guides render unreviewed on source-PR previews, so they must stay data: no
// rehype-raw (raw HTML is dropped) and react-markdown's default URL filter.
// scripts/test-guide-html.mjs checks this configuration.
export const guideRemarkPlugins = [remarkGfm];
export const guideRehypePlugins = [rehypeSlug];

// The same configuration with heading ids prefixed, for Markdown that renders
// beside the guide on one page (preset runbooks, notes, troubleshooting) so
// their heading ids can't collide with the guide's.
export function guideRehypePluginsFor(prefix) {
  return prefix ? [[rehypeSlug, { prefix }]] : guideRehypePlugins;
}
