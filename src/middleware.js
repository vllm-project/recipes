import { NextResponse } from "next/server";

// PROD-10 — `.md` shadow URLs for recipes.
//
// Anyone requesting `/<org>/<repo>.md` gets the markdown twin of the
// rendered recipe page. This is the "fetch by URL" path AI assistants
// take when a user pastes a link into Claude/ChatGPT/Perplexity. We
// rewrite (server-side, transparent) rather than redirect because the
// user should see the `.md` URL in their address bar — that's the
// explicit "give me markdown" signal.
//
// Design notes:
//   - No User-Agent sniffing. Google classifies "different bytes for the
//     same URL based on UA" as cloaking; SEO penalty risk.
//   - No Accept-header negotiation. Operationally fragile in caches and
//     CDNs. The `.md` suffix is the explicit signal.
//   - Only `/<org>/<repo>.md` matches — top-level `/<thing>.md` would
//     conflict with future static markdown files (llms.txt, robots.txt,
//     etc. already have their own handlers).

export function middleware(request) {
  const { pathname } = request.nextUrl;
  if (!pathname.endsWith(".md")) return NextResponse.next();

  // Expect exactly `/<org>/<repo>.md` (two segments, no extra slashes).
  const stripped = pathname.slice(1, -".md".length);
  const parts = stripped.split("/");
  if (parts.length !== 2 || !parts[0] || !parts[1]) {
    return NextResponse.next();
  }
  const [org, repo] = parts;

  const url = request.nextUrl.clone();
  url.pathname = `/api/recipe-md/${org}/${repo}`;
  return NextResponse.rewrite(url);
}

export const config = {
  // Match anything that ends in `.md` but skip framework internals so
  // static markdown files (llms.txt is text/plain, not affected) and
  // Next's own assets don't get touched.
  matcher: ["/((?!_next/|api/|og$|robots.txt$|sitemap.xml$|llms.txt$|llms-full.txt$).*\\.md)"],
};
