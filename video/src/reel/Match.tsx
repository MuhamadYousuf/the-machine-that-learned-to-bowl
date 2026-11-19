import React from 'react';
import {BallFlight} from '../hawkeye/BallFlight';
import {Delivery, flightEnd, positionAt} from '../lib/data';
import {bowlerPose, Batter, Bowler, FIELDERS, Fielder, Keeper} from './Mannequin';
import {Ground, L, Look, Pitch, Pose, Stage} from './World';
import {clamp01, lerp} from './timeline';

export type Data = {
  slog: OversLike;
  tech: OversLike;
  tail: OversLike;
  v1tech: OversLike;
  swarms: Record<string, {deliveries: Delivery[]}>;
  net: NetFile;
};
export type OversLike = {overs: {deliveries: Delivery[]}[]};
export type NetFile = {inputs: string[]; outputs: string[]; sizes: number[]; champion: {w: number[][]; b: number[][]}; frames: {gen: number; w: number[][]; b: number[][]}[]};

export const isWicket = (d: Delivery) => ['bowled', 'lbw', 'caught_behind', 'caught_field'].includes(d.result);

/** Where the bowler is (for cameras that follow him). */
export function bowlerAt(u: number): [number, number, number] {
  return bowlerPose(u).root;
}

/**
 * One delivery in the stadium. s = sim seconds relative to release (negative = run-up).
 * ghosts = earlier deliveries drawn as faded full paths.
 */
export const Match: React.FC<{
  d: Delivery;
  s: number;
  pose: Pose;
  look?: Look;
  ghosts?: Delivery[];
  fielders?: boolean;
  trail?: boolean;
  ballScale?: number;
  children?: React.ReactNode;
}> = ({d, s, pose, look, ghosts = [], fielders = true, trail = true, ballScale = 2.2, children}) => {
  const end = flightEnd(d);
  const hit = d.result === 'bowled' || d.result === 'lbw' && d.hitsStumps ? clamp01((s - end) / 0.28) : 0;
  const catchUp = d.result === 'caught_behind' ? clamp01((s - (d.tc ?? end) - 0.02) / 0.12) : 0;
  return (
    <Stage pose={pose} look={look}>
      <Ground />
      <Pitch batterHit={d.result === 'bowled' ? hit : 0} />
      <Bowler u={s} />
      <Batter t={s} tc={d.tc ?? end} shot={d.shot ?? 'defend'} />
      <Keeper catchUp={catchUp} />
      {fielders ? FIELDERS.map(([x, z], i) => <Fielder key={i} x={x} z={z} t={s} />) : null}
      {ghosts.map((g, i) => (
        <BallFlight key={i} d={g} t={flightEnd(g)} style={{ghost: true, showBall: false, trailOpacity: 0.55, trailRadius: 0.018, showFlash: false, showRing: false, dotRadius: 0.07, dotOpacity: 0.9}} />
      ))}
      {s >= 0 ? (
        <BallFlight d={d} t={s} style={{ballScale, trailRadius: trail ? 0.02 : 0.0001, trailOpacity: trail ? 0.95 : 0, colorPre: '#ffffff', colorPost: '#ffe14d'}} />
      ) : null}
      {children}
    </Stage>
  );
};

/* ------------------------------------------------------------------ cameras */

export const cams = {
  /** Behind the bowler, following his run (like a chase cam behind a car). */
  chase(u: number, aspect: number): Pose {
    const b = bowlerAt(Math.min(u, 0.25));
    const tall = aspect < 0.7;
    return {position: [b[0] + 0.9, 2.1, b[2] - (tall ? 5.2 : 6.5)], target: [0, 1.2, b[2] + 14], fov: tall ? 54 : 46};
  },
  /** Ground-level stump camera at the batter's end, looking back up the pitch. */
  stump(aspect: number): Pose {
    return {position: [3.1, 0.55, L - 2.2], target: [-0.15, 0.5, L + 0.25], fov: aspect < 0.7 ? 46 : 38};
  },
  broadcast(aspect: number): Pose {
    return aspect < 0.7 ? {position: [0.6, 6.5, -12.5], target: [0, 0.4, L - 5], fov: 34} : {position: [0.6, 6.5, -14], target: [0, 0.4, L - 4], fov: 27};
  },
  side(aspect: number, zc = L - 7): Pose {
    return {position: [aspect < 0.7 ? 15 : 17, 1.5, zc], target: [0, 1.0, zc], fov: aspect < 0.7 ? 40 : 34};
  },
  keeper(aspect: number): Pose {
    return {position: [0.5, 1.9, L + 7.5], target: [0, 0.9, 4], fov: aspect < 0.7 ? 40 : 32};
  },
  orbit(a: number, radius = 62, height = 30, fov = 42): Pose {
    return {position: [Math.sin(a) * radius, height, L / 2 - Math.cos(a) * radius], target: [0, 0, L / 2], fov};
  },
};

export function mix(a: Pose, b: Pose, k: number): Pose {
  const v = (p: [number, number, number], q: [number, number, number]) => p.map((x, i) => lerp(x, q[i], k)) as [number, number, number];
  return {position: v(a.position, b.position), target: v(a.target, b.target), fov: lerp(a.fov, b.fov, k), up: a.up && b.up ? v(a.up, b.up) : undefined};
}

/** Small handheld drift so static shots breathe. */
export function drift(p: Pose, t: number, amt = 0.08): Pose {
  return {
    ...p,
    position: [p.position[0] + Math.sin(t * 0.7) * amt, p.position[1] + Math.sin(t * 0.53 + 1) * amt * 0.6, p.position[2] + Math.cos(t * 0.41) * amt],
  };
}

export {positionAt, flightEnd};
