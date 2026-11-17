// /search-index.json — what the header search box searches, fetched the first
// time the box is used. Inlined into the root layout it rode along in every
// page's HTML and every <Link> prefetch (~150 KB raw each), and in every 404.
// Static: built once per deployment and served from the CDN.

import { getAllRecipes } from "@/lib/recipes";
import { toSearchEntry } from "@/lib/recipe-utils";

export const dynamic = "force-static";

export function GET() {
  return Response.json(getAllRecipes().map(toSearchEntry));
}
