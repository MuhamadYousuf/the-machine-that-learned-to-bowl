import {useEffect, useState} from 'react';
import {continueRender, delayRender, staticFile} from 'remotion';

export type Delivery = {
  t: number[];
  x: number[];
  y: number[];
  z: number[];
  bounce: [number, number, number] | null;
  tb: number | null;
  tc: number | null;
  ts: number | null;
  result: string;
  runs: number;
  wicket: boolean;
  legal: boolean;
  shot: string | null;
  shotAngle: number | null;
  shotDist: number | null;
  length: number | null;
  line: number | null;
  category: string;
  hitsStumps: boolean;
  wide: boolean;
  beamer: boolean;
  speed: number;
  elev: number;
  azim: number;
  swing: number;
  batter: string;
  genome?: number;
  reward?: number;
  fitness?: number;
};

export type OversFile = {
  run: string;
  label: string;
  generation: number;
  archetype: string;
  overs: {seed: number; deliveries: Delivery[]}[];
};

export type SwarmFile = {run: string; generation: number; deliveries: Delivery[]};

const cache = new Map<string, Promise<unknown>>();

export function fetchJson<T>(name: string): Promise<T> {
  if (!cache.has(name)) {
    cache.set(
      name,
      fetch(staticFile(`data/${name}`)).then((r) => {
        if (!r.ok) throw new Error(`failed to load ${name}: ${r.status}`);
        return r.json();
      }),
    );
  }
  return cache.get(name) as Promise<T>;
}

/** Load JSON files from public/data, blocking the render until they arrive. */
export function useJsonFiles<T>(names: string[]): T[] | null {
  const key = names.join('|');
  const [data, setData] = useState<T[] | null>(null);
  const [handle] = useState(() => delayRender(`load ${key}`));
  useEffect(() => {
    Promise.all(names.map((n) => fetchJson<T>(n)))
      .then((values) => {
        setData(values);
        continueRender(handle);
      })
      .catch((err) => {
        console.error(err);
        throw err;
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return data;
}

/** Simulation coordinates (x along pitch, y off side, z up) to three.js (X, Y up, Z). */
export function toScene(x: number, y: number, z: number): [number, number, number] {
  return [y, z, x];
}

/** Ball position at time t (seconds since release), linear between 4 ms samples. */
export function positionAt(d: Delivery, t: number): [number, number, number] {
  const ts = d.t;
  if (t <= ts[0]) return toScene(d.x[0], d.y[0], d.z[0]);
  const last = ts.length - 1;
  if (t >= ts[last]) return toScene(d.x[last], d.y[last], d.z[last]);
  let lo = 0;
  let hi = last;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (ts[mid] <= t) lo = mid;
    else hi = mid;
  }
  const a = (t - ts[lo]) / (ts[hi] - ts[lo]);
  return toScene(
    d.x[lo] + (d.x[hi] - d.x[lo]) * a,
    d.y[lo] + (d.y[hi] - d.y[lo]) * a,
    d.z[lo] + (d.z[hi] - d.z[lo]) * a,
  );
}

/** Points along the path from release up to time t (inclusive), in scene coordinates. */
export function pathUntil(d: Delivery, t: number, from = 0): [number, number, number][] {
  const out: [number, number, number][] = [];
  for (let i = 0; i < d.t.length; i++) {
    if (d.t[i] < from) continue;
    if (d.t[i] > t) break;
    out.push(toScene(d.x[i], d.y[i], d.z[i]));
  }
  out.push(positionAt(d, t));
  return out;
}

/** The time at which the ball's visible flight ends (bat contact or the stumps). */
export function flightEnd(d: Delivery): number {
  const end = d.t[d.t.length - 1];
  if (reachesStumps(d)) return Math.min(d.ts ?? end, end);
  return Math.min(d.tc ?? end, end);
}

/** Seconds (ball time) the struck ball takes to reach its landing point; 0 if nothing was hit for runs. */
export function shotTime(d: Delivery): number {
  if (!d.runs || d.shotAngle === null || d.shotDist === null || reachesStumps(d)) return 0;
  return Math.min(2.4, d.shotDist / (d.runs >= 6 ? 30 : 34));
}

/** Scene position of the struck ball at fraction u (0..1) of its shot, starting at `from` (scene coords). */
export function shotPoint(d: Delivery, from: [number, number, number], u: number): [number, number, number] {
  const a = ((d.shotAngle ?? 0) * Math.PI) / 180;
  const six = d.runs >= 6;
  const along = six ? u : 1 - (1 - u) * (1 - u); // fours slow down as they run along the ground
  const dist = (d.shotDist ?? 0) * along;
  // Sim: x towards the batter, y off side. Angle 0 = straight back past the bowler. Scene = (y, z, x).
  const x = from[0] + Math.sin(a) * dist;
  const z = from[2] - Math.cos(a) * dist;
  const y = six ? from[1] + (22 - from[1]) * 4 * u * (1 - u) * 1.0 - from[1] * u * u
               : Math.max(0.04, from[1] * (1 - u) * Math.abs(Math.cos(u * Math.PI * 2.5)));
  return [x, Math.max(0.04, y), z];
}

/** Bowled and LBW balls carry on to the stumps; everything else meets the bat or pad. */
export function reachesStumps(d: Delivery): boolean {
  return d.result === 'bowled' || d.result === 'lbw' || d.wide || (!d.legal && d.result !== 'no_ball');
}
