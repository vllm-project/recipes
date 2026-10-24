"use client";

// The command builder's current selection, published for the sections below
// it (This configuration, Tuned profiles, Troubleshooting) without lifting the
// builder's state. null until the builder mounts, so server HTML renders the
// unfiltered content.

import { useSyncExternalStore } from "react";

let selection = null;
const listeners = new Set();

export function publishSelection(next) {
  selection = next;
  for (const l of listeners) l();
}

function subscribe(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

// The selection for this recipe, or null (a stale one from a previous page
// is ignored).
export function useSelection(hfId) {
  const sel = useSyncExternalStore(subscribe, () => selection, () => null);
  return sel && sel.hfId === hfId ? sel : null;
}
