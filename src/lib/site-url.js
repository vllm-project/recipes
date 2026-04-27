// Canonical site URL used for metadataBase, sitemap, robots, JSON-LD, OG.
//
// On Vercel, `VERCEL_URL` is set in BOTH production and preview, and it's
// always the unique per-deploy hostname (vllm-recipes-XXX.vercel.app), never
// the prod alias. So we only honour it on preview — on prod we use the
// hardcoded canonical domain unless `NEXT_PUBLIC_SITE_URL` overrides it.
export const siteUrl =
  process.env.NEXT_PUBLIC_SITE_URL ||
  (process.env.VERCEL_ENV === "preview" && process.env.VERCEL_URL
    ? `https://${process.env.VERCEL_URL}`
    : "https://recipes.vllm.ai");
