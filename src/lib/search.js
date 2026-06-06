import { matchSorter } from "match-sorter";

const strip = (s) => String(s || "").replace(/[-_.]/g, "");

const RECIPE_KEYS = [
  { key: (r) => strip(r.hf_repo) },
  { key: (r) => strip(r.hf_org) },
  { key: (r) => strip(r.meta?.title) },
  { key: (r) => strip(r.meta?.provider) },
  { key: (r) => strip(r.meta?.description) },
  { key: (r) => (r.meta?.tasks || []).map(strip).join(" ") },
  { key: (r) => strip(r.model?.architecture) },
  { key: (r) => strip(r.model?.parameter_count) },
  { key: (r) => strip(r.variant?.precision) },
  { key: (r) => (r.precisions || []).map(strip).join(" ") },
  {
    key: (r) => {
      const hw = r.meta?.hardware || {};
      const verified = Object.entries(hw)
        .filter(([, s]) => s === "verified")
        .map(([h]) => h);
      const synonyms = [
        ...(verified.some((k) => k === "trillium" || k === "ironwood") ? ["tpu"] : []),
        ...(verified.some((k) => k === "xeon6" || k === "xeon5") ? ["intel", "xeon", "cpu", "x86"] : []),
      ];
      return [...verified, ...synonyms].map(strip).join(" ");
    },
  },
];

export function searchRecipes(recipes, query) {
  const q = query.trim();
  if (!q) return [];
  return q.split(/\s+/).reduceRight(
    (items, word) =>
      matchSorter(items, strip(word), {
        keys: RECIPE_KEYS,
        threshold: matchSorter.rankings.CONTAINS,
      }),
    recipes,
  );
}

export function searchProviders(providerEntries, query) {
  const q = query.trim();
  if (!q) return [];
  const keys = [
    { key: (entry) => strip(entry[0]) },
    { key: (entry) => strip(entry[1]?.display_name) },
  ];
  return q.split(/\s+/).reduceRight(
    (items, word) =>
      matchSorter(items, strip(word), {
        keys,
        threshold: matchSorter.rankings.CONTAINS,
      }),
    providerEntries,
  );
}
