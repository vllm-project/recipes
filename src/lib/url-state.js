// Write a page's query string in place.
//
// Next.js wires the native History API into its router, so useSearchParams /
// usePathname readers re-render after this exactly as they do after
// router.replace. router.replace also asks the server for the route's RSC
// payload on every call, even for a statically generated page that ignores
// its query: on a recipe page that was ~80 KB per builder click, all of it
// discarded. Use this for client-only state such as builder or filter
// selections; keep router.push/replace for real navigation.
export function replaceQuery(qs) {
  const url = `${window.location.pathname}${qs ? `?${qs}` : ""}${window.location.hash}`;
  window.history.replaceState(window.history.state, "", url);
}
