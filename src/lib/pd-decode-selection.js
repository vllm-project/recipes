import { IncompatibleDecodeTp } from "./command-synthesis.js";

// Only an interactive Ascend PCP selection can substitute a Decode mode. The
// strict renderer checks the final connector and the actual TP at each pool size.
export function selectPdDecode(pools, options, render) {
  if (!options) return { pools, result: render(pools) };
  const { modes, defaultMode, nodesFor, listModes } = options;
  const requested = pools.decode.parallelism;
  const available = new Set();
  let selected, widthError;
  for (const mode of new Set([requested, defaultMode, ...modes])) {
    if (mode !== requested && !modes.includes(mode)) continue;
    const candidate = mode === requested ? pools
      : { ...pools, decode: { ...pools.decode, parallelism: mode, nodes: nodesFor(mode) } };
    try {
      const result = render(candidate);
      selected ??= { pools: candidate, result };
      available.add(mode);
      if (!listModes) break; // A shared link only needs its first usable mode.
    } catch (error) {
      if (!(error instanceof RangeError)) throw error;
      if (mode === requested) {
        if (!(error instanceof IncompatibleDecodeTp)) throw error;
        widthError = error;
      }
    }
  }
  // The requested mode was tried first. If no alternative worked, preserve its error.
  if (!selected) throw widthError;
  return { ...selected, modes: modes.filter((mode) => available.has(mode)), reason: widthError?.message };
}
