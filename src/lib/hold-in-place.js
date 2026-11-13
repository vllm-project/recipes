"use client";

// Keep `el` at its current position on screen while the layout above it
// changes. A pick re-renders the command block, which sits above the config
// rows and the reference tabs, and can change its height by hundreds of
// pixels (a preset or strategy swaps a whole PD / multi-node command). Native
// scroll anchoring doesn't catch it, and Safari has none.
//
// Corrections run in a microtask (after React's synchronous flush of a
// discrete event, before paint) and then before every frame for `ms`, which
// also covers follow-up renders (a hardware switch that picks its preset in an
// effect, router.replace's transition). The reader's own scrolling ends it.
export function holdInPlace(el, ms = 700) {
  if (!el) return;
  const before = el.getBoundingClientRect().top;
  const end = performance.now() + ms;
  let stopped = false;
  const stop = () => { stopped = true; };
  const opts = { once: true, passive: true };
  window.addEventListener("wheel", stop, opts);
  window.addEventListener("touchmove", stop, opts);
  window.addEventListener("keydown", stop, opts);
  const settle = () => {
    if (stopped || !el.isConnected) return;
    const delta = el.getBoundingClientRect().top - before;
    if (Math.abs(delta) > 1) window.scrollBy(0, delta);
  };
  queueMicrotask(settle);
  const tick = () => {
    settle();
    if (!stopped && performance.now() < end) requestAnimationFrame(tick);
    else {
      window.removeEventListener("wheel", stop);
      window.removeEventListener("touchmove", stop);
      window.removeEventListener("keydown", stop);
    }
  };
  requestAnimationFrame(tick);
}
