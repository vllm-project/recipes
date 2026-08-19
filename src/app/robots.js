import { siteUrl } from "@/lib/site-url";

// Internal-only paths we never want indexed. `/og` is the dynamic OG image
// endpoint (never an SEO destination); `/api/` is the Next.js internal route
// bucket; `/_next/` and `/static/` are framework build artefacts.
const COMMON_DISALLOW = ["/og", "/api/", "/_next/", "/static/"];

// Default policy is "allow everything to all crawlers" via `User-Agent: *`.
// We still list the major AI crawlers explicitly so:
//   1. Intent is visible to anyone auditing this file (it answers the
//      common "is GPTBot allowed?" question without inference).
//   2. Future changes to the `User-Agent: *` rule cannot accidentally drop
//      AI-assistant traffic.
// Note: there is no functional difference vs `User-Agent: *` today — these
// stanzas are documentation that happens to be machine-readable.
const AI_USER_AGENTS = [
  // OpenAI
  "GPTBot",
  "ChatGPT-User",
  "OAI-SearchBot",
  // Anthropic
  "ClaudeBot",
  "Claude-Web",
  "anthropic-ai",
  // Perplexity
  "PerplexityBot",
  "Perplexity-User",
  // Google's AI training opt-out token (distinct from Googlebot)
  "Google-Extended",
  // Other major engines / AI-search surfaces
  "Bingbot",
  "Applebot",
  "Applebot-Extended",
  "Amazonbot",
  "Bytespider",
  "CCBot",
  "DuckAssistBot",
  "YouBot",
  "Mistral-AI-User",
  "cohere-ai",
  "Meta-ExternalAgent",
  "Meta-ExternalFetcher",
  "FacebookBot",
];

export default function robots() {
  return {
    rules: [
      { userAgent: "*", allow: "/", disallow: COMMON_DISALLOW },
      ...AI_USER_AGENTS.map((userAgent) => ({
        userAgent,
        allow: "/",
        disallow: COMMON_DISALLOW,
      })),
    ],
    sitemap: `${siteUrl}/sitemap.xml`,
    host: siteUrl,
  };
}
