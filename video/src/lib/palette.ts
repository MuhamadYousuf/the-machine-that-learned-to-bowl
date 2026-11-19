// One palette for every clip (matches the Python analysis figures).
export const palette = {
  bg: '#060a13',
  bgSoft: '#0b1220',
  outfield: '#0c2a1d',
  pitch: '#c7b182',
  pitchDark: '#a8915f',
  crease: '#f4f1e8',
  stumps: '#f2e6c9',
  ink: '#e8f1fb',
  muted: '#93a7bd',
  teal: '#49debe',
  pink: '#fb7185',
  amber: '#fbbf24',
  orange: '#fb923c',
  violet: '#a78bfa',
  green: '#5ee0a0',
  ball: '#ff3b3b',
  trailPre: '#6fe7ff',
  trailPost: '#ffd166',
  wicket: '#ff4d6d',
  runs: '#5ee0a0',
  dot: '#c8d3e0',
};

export function outcomeColor(result: string, runs: number): string {
  if (['bowled', 'lbw', 'caught_behind', 'caught_field'].includes(result)) return palette.wicket;
  if (result === 'wide' || result === 'no_ball') return palette.amber;
  if (runs >= 4) return palette.orange;
  if (runs > 0) return palette.runs;
  return palette.dot;
}

/** Map t in [0,1] from a dull blue (low fitness) to bright teal-white (high fitness). */
export function fitnessColor(t: number): string {
  const c = Math.max(0, Math.min(1, t));
  const stops: [number, number, number][] = [
    [64, 70, 120],
    [167, 139, 250],
    [73, 222, 190],
    [230, 255, 245],
  ];
  const seg = Math.min(stops.length - 2, Math.floor(c * (stops.length - 1)));
  const f = c * (stops.length - 1) - seg;
  const a = stops[seg];
  const b = stops[seg + 1];
  const mix = a.map((v, i) => Math.round(v + (b[i] - v) * f));
  return `rgb(${mix[0]},${mix[1]},${mix[2]})`;
}

export const ease = {
  inOutCubic: (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
  outCubic: (t: number) => 1 - Math.pow(1 - t, 3),
  inOutSine: (t: number) => -(Math.cos(Math.PI * t) - 1) / 2,
};

export const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
