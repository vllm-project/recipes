"use client";

// The command builder's current selection, published for the reference tabs
// below it (Presets, Troubleshooting) without lifting the builder's state.
// null until the builder mounts, so server HTML renders the unfiltered
// content. The other direction, a preset card asking the builder to apply a
// preset, goes through requestPreset / onPresetRequest.

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

const presetRequests = new Set();

// Ask the builder to apply a preset: it switches hardware and strategy as
// needed. Returns false when no builder is listening (the caller can fall
// back to navigating).
export function requestPreset(hfId, key) {
  for (const l of presetRequests) l({ hfId, key });
  return presetRequests.size > 0;
}

export function onPresetRequest(listener) {
  presetRequests.add(listener);
  return () => presetRequests.delete(listener);
}
