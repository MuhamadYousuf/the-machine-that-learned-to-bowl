import {Delivery, OversFile} from '../lib/data';

/** The deliveries for an over (or chosen balls of it). */
export function pickDeliveries(data: OversFile, overIndex: number, balls: number[], overCount = 1): Delivery[] {
  if (balls.length) {
    const all = data.overs[overIndex].deliveries;
    return balls.map((i) => all[i]);
  }
  return data.overs.slice(overIndex, overIndex + overCount).flatMap((o) => o.deliveries);
}
