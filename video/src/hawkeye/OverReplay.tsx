import React from 'react';
import {AbsoluteFill, useCurrentFrame, useVideoConfig} from 'remotion';
import {Delivery, OversFile, useJsonFiles} from '../lib/data';
import {BallFlight} from './BallFlight';
import {CameraName, cameraPose} from './cameras';
import {Scene} from './Scene';
import {Stadium} from './Stadium';
import {ballTime, schedule} from './timeline';
import {HOLD} from './hold';

export type OverReplayProps = {
  file: string;
  overIndex: number;
  /** Number of consecutive overs to play from overIndex (default 1). */
  overCount?: number;
  /** Optional subset of ball indices (0-based) within the first over; empty = whole overs. */
  balls: number[];
  camera: CameraName;
  speed: number;
  accumulate: boolean;
  goodLengthBand: boolean;
};

export {pickDeliveries} from './pick';
import {pickDeliveries} from './pick';

/** Wicket accents that follow the active delivery. */
export function wicketAccents(d: Delivery | undefined, t: number) {
  if (!d || !d.wicket) return {tilt: 0, bail: 0, zone: 0};
  const end = d.ts ?? d.t[d.t.length - 1];
  const since = t - end;
  if (d.result === 'bowled' && since > 0) {
    const k = Math.min(1, since / 0.5);
    return {tilt: 0.35 * k, bail: 0.5 * k, zone: 0};
  }
  if (d.result === 'lbw' && since > 0) {
    return {tilt: 0, bail: 0, zone: 0.6 + 0.4 * Math.sin(since * 14)};
  }
  return {tilt: 0, bail: 0, zone: 0};
}

/** One stage showing a sequence of deliveries at local frame f (after the opening hold). */
export const OverStage: React.FC<{
  deliveries: Delivery[];
  f: number;
  camera: CameraName;
  speed: number;
  accumulate: boolean;
  goodLengthBand: boolean;
  width: number;
  height: number;
  /** Keep at most this many earlier balls on screen (oldest fade first). */
  keep?: number;
}> = ({deliveries, f, camera, speed, accumulate, goodLengthBand, width, height, keep = 12}) => {
  const {slots} = schedule(deliveries, speed);
  const activeIndex = slots.findIndex((s) => f < s.end);
  const active = activeIndex === -1 ? slots[slots.length - 1] : slots[activeIndex];
  const t = ballTime(Math.min(Math.max(0, f), active.end - 1), active, speed);
  const acc = wicketAccents(active.d, t);
  const flight = active.d.tc ?? active.d.t[active.d.t.length - 1];
  const pose = cameraPose(camera, width / height, t / flight);
  const mapView = camera === 'pitchMap';
  const style = {ballScale: pose.ballScale, trailRadius: 0.018 * pose.trailScale, ...(mapView ? {trailOpacity: 0.55, dotOpacity: 1, dotRadius: 0.11} : {})};
  const earlier = accumulate ? slots.filter((s) => s.index < active.index && s.index >= active.index - keep) : [];
  return (
    <Scene pose={pose} width={width} height={height}>
      <Stadium goodLengthBand={goodLengthBand} batterStumpsTilt={acc.tilt} bailLift={acc.bail} wicketZone={acc.zone} />
      {earlier.map((s) => (
        <BallFlight key={s.index} d={s.d} t={99} style={{...style, ghost: true}} />
      ))}
      <BallFlight d={active.d} t={t} style={style} />
    </Scene>
  );
};

export const OverReplay: React.FC<OverReplayProps> = (props) => {
  const frame = useCurrentFrame();
  const {width, height} = useVideoConfig();
  const files = useJsonFiles<OversFile>([props.file]);
  if (!files) return <AbsoluteFill />;
  const deliveries = pickDeliveries(files[0], props.overIndex, props.balls, props.overCount ?? 1);
  return (
    <OverStage
      deliveries={deliveries}
      f={frame - HOLD}
      camera={props.camera}
      speed={props.speed}
      accumulate={props.accumulate}
      goodLengthBand={props.goodLengthBand}
      width={width}
      height={height}
    />
  );
};
