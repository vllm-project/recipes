// The "O" ring of the vLLM-Omni wordmark (vllm-project/vllm-omni,
// docs/source/logos/vllm-omni-logo.png): the part of that brand which isn't
// vLLM's own "v". Used wherever a recipe is served by vLLM-Omni.
//
// Drawn as short solid-colored arcs instead of an SVG gradient: a gradient
// needs an `id`, every instance on the page would share it, and a `url(#id)`
// that resolves to an instance inside a hidden subtree (the collapsed mobile
// sidebar) renders nothing in Chrome and Firefox.
//
// Stops sampled clockwise from 12 o'clock around the wordmark's ring.
const STOPS = [
  "#c6945d", "#fd9626", "#fc7e2e", "#e34957", "#b12397", "#a017a7",
  "#a215ae", "#911eb0", "#5f34bd", "#2955d3", "#0777e0", "#4987b3",
];
const SEGMENTS = 24; // at icon sizes (≤16px) more segments look identical
const RX = 7.8; // the wordmark's O is taller than wide (98 : 113)
const RY = 9;
const C = 12;

function hexToRgb(h) {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function colorAt(t) {
  const pos = t * STOPS.length;
  const i = Math.floor(pos) % STOPS.length;
  const f = pos - Math.floor(pos);
  const a = hexToRgb(STOPS[i]);
  const b = hexToRgb(STOPS[(i + 1) % STOPS.length]);
  const mix = a.map((v, k) => Math.round(v + (b[k] - v) * f));
  return `rgb(${mix.join(",")})`;
}

function point(deg) {
  const a = (deg * Math.PI) / 180;
  return [C + RX * Math.sin(a), C - RY * Math.cos(a)];
}

// Each arc overlaps the next by a degree so antialiasing leaves no seams.
const ARCS = Array.from({ length: SEGMENTS }, (_, i) => {
  const step = 360 / SEGMENTS;
  const [x0, y0] = point(i * step);
  const [x1, y1] = point((i + 1) * step + 1);
  const f = (n) => n.toFixed(3);
  return {
    d: `M${f(x0)} ${f(y0)}A${RX} ${RY} 0 0 1 ${f(x1)} ${f(y1)}`,
    color: colorAt((i + 0.5) / SEGMENTS),
  };
});

/** Same call shape as a lucide icon: `size`, `className`, other SVG props. */
export function OmniMark({ size = 24, className = "", strokeWidth = 2.8, ...props }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
      className={`shrink-0 ${className}`}
      {...props}
    >
      {ARCS.map((a, i) => (
        <path key={i} d={a.d} stroke={a.color} strokeWidth={strokeWidth} />
      ))}
    </svg>
  );
}
