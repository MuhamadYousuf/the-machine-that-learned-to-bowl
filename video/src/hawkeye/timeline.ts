import {Delivery, flightEnd, reachesStumps, shotTime} from '../lib/data';

export const FPS = 60;
export const LEAD_S = 0.6; // pause before each release (real seconds)
export const TAIL_S = 1.1; // linger after the ball finishes

export type Slot = {index: number; start: number; flight: number; end: number; d: Delivery};

/** Real-time length of one ball's on-screen flight, including the outcome beat. */
export function ballDuration(d: Delivery, speed: number): number {
  const flight = flightEnd(d) + (reachesStumps(d) ? 0.15 : 0.1);
  return LEAD_S + (flight + shotTime(d)) / speed + TAIL_S;
}

/** Back-to-back schedule (in frames) for a list of deliveries at a playback speed. */
export function schedule(deliveries: Delivery[], speed: number): {slots: Slot[]; frames: number} {
  let cursor = 0;
  const slots = deliveries.map((d, index) => {
    const total = ballDuration(d, speed);
    const start = cursor;
    cursor += Math.round(total * FPS);
    return {index, start, flight: Math.round(LEAD_S * FPS), end: cursor, d};
  });
  return {slots, frames: cursor};
}

/** Ball time (seconds since release) for a frame inside a slot. */
export function ballTime(frame: number, slot: Slot, speed: number): number {
  const local = frame - slot.start - slot.flight;
  return (local / FPS) * speed;
}

export type BallEvent = {
  ball: number;
  startFrame: number;
  releaseFrame: number;
  bounceFrame: number | null;
  creaseFrame: number | null;
  endFrame: number;
  result: string;
  runs: number;
  wicket: boolean;
  category: string;
  speedKmh: number;
  batter: string;
};

/** Frame-exact event list for a scheduled sequence (frames include the opening hold `offset`). */
export function ballEvents(deliveries: Delivery[], speed: number, offset: number): BallEvent[] {
  const {slots} = schedule(deliveries, speed);
  const at = (slot: Slot, seconds: number | null) =>
    seconds === null ? null : offset + slot.start + slot.flight + Math.round((seconds / speed) * FPS);
  return slots.map((slot) => ({
    ball: slot.index,
    startFrame: offset + slot.start,
    releaseFrame: offset + slot.start + slot.flight,
    bounceFrame: at(slot, slot.d.tb),
    creaseFrame: at(slot, slot.d.tc),
    endFrame: at(slot, flightEnd(slot.d)) as number,
    result: slot.d.result,
    runs: slot.d.runs,
    wicket: slot.d.wicket,
    category: slot.d.category,
    speedKmh: slot.d.speed,
    batter: slot.d.batter,
  }));
}
