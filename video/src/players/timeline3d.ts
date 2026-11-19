import {Delivery, flightEnd, shotTime} from '../lib/data';
import {FPS} from '../hawkeye/timeline';

/** Shot timeline for the 3D player scenes (shared by Match3D and the Blender baker). */
export const RUNUP = 2.3;
export const AFTER = 1.5;

/** Ball-time τ (seconds since release) as a function of real seconds into the ball's segment, with an optional slow-mo ramp. */
export type Segment = {d: Delivery; index: number; start: number; frames: number; tau: (real: number) => number; release: number; end: number};

export function buildSegments(ds: Delivery[], props: {slowmo?: {ball: number; factor: number}; runup?: number}, fps = FPS): Segment[] {
  const runup = props.runup ?? RUNUP;
  let cursor = 0;
  return ds.map((d, index) => {
    const ballEnd = flightEnd(d) + shotTime(d) + AFTER;
    const slow = props.slowmo && props.slowmo.ball === index ? props.slowmo.factor : 1;
    // slow-mo window in ball time: from 0.35 s before contact to 0.9 s after
    const contact = d.tc ?? flightEnd(d);
    const w0 = contact - 0.35;
    const w1 = contact + 0.9;
    // real time to cover ball time [-runup, ballEnd]
    const realOf = (tau: number) => {
      const a = Math.min(tau, w0) - -runup;
      const b = Math.max(0, Math.min(tau, w1) - w0) / slow;
      const c = Math.max(0, tau - w1);
      return Math.max(0, a) + b + c;
    };
    const total = realOf(ballEnd);
    const tau = (real: number) => {
      const r0 = realOf(w0);
      const r1 = realOf(w1);
      if (real <= r0) return -runup + real;
      if (real <= r1) return w0 + (real - r0) * slow;
      return w1 + (real - r1);
    };
    const frames = Math.round(total * fps);
    const seg = {d, index, start: cursor, frames, tau, release: cursor + Math.round(realOf(0) * fps), end: cursor + Math.round(realOf(flightEnd(d)) * fps)};
    cursor += frames;
    return seg;
  });
}

