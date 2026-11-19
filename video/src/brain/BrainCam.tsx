import React, {useMemo} from 'react';
import {AbsoluteFill, useCurrentFrame, useVideoConfig} from 'remotion';
import {useJsonFiles} from '../lib/data';
import {ease, palette} from '../lib/palette';
import {HOLD} from '../hawkeye/hold';
import {FPS} from '../hawkeye/timeline';
import {NetFile, stableEdges} from './net';
import {NetworkView} from './NetworkView';
import {Gauge, PitchDot, PitchMap2D} from './widgets';

export type CamBall = {
  inputs: number[];
  h1: number[];
  h2: number[];
  raw: number[];
  out: number[];
  release: {speed_kmh: number; elevation_deg: number; azimuth_deg: number; swing: number};
  length: number;
  line: number;
  category: string;
  result: string;
  runs: number;
  wicket: boolean;
  bounce: [number, number] | null;
  times: {release_s?: number; bounce_s?: number | null; crease_s?: number | null; stumps_s?: number | null};
};
export type CamFile = {run: string; archetype: string; seed: number; generation: number; balls: CamBall[]};

export const THINK_S = 1.0;
export const GAUGE_S = [0.7, 1.1];
export const FLIGHT_START_S = 1.1;
export const BALL_HOLD_S = 0.75;

export function camBallSeconds(b: CamBall, speed: number) {
  const flight = b.times.crease_s ?? b.times.stumps_s ?? 0.65;
  return FLIGHT_START_S + flight / speed + BALL_HOLD_S;
}

export function camSchedule(balls: CamBall[], speed: number) {
  let cursor = 0;
  return balls.map((b, index) => {
    const start = cursor;
    cursor += camBallSeconds(b, speed);
    const bounce = b.times.bounce_s == null ? null : start + FLIGHT_START_S + b.times.bounce_s / speed;
    return {index, start, end: cursor, bounce, b};
  });
}

export function camEvents(balls: CamBall[], speed: number) {
  return camSchedule(balls, speed).map((s) => ({
    ball: s.index,
    thinkStartFrame: HOLD + Math.round(s.start * FPS),
    decisionFrame: HOLD + Math.round((s.start + GAUGE_S[1]) * FPS),
    releaseFrame: HOLD + Math.round((s.start + FLIGHT_START_S) * FPS),
    bounceFrame: s.bounce == null ? null : HOLD + Math.round(s.bounce * FPS),
    endFrame: HOLD + Math.round(s.end * FPS),
    result: s.b.result,
    runs: s.b.runs,
    wicket: s.b.wicket,
    category: s.b.category,
    speedKmh: s.b.release.speed_kmh,
  }));
}

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

/** State of the brain at local time `t` (seconds) — activations blend layer by layer from the previous ball. */
export function camState(balls: CamBall[], speed: number, t: number) {
  const slots = camSchedule(balls, speed);
  const found = slots.findIndex((s) => t < s.end);
  const slot = found === -1 ? slots[slots.length - 1] : slots[found];
  const local = t - slot.start;
  const prev = slot.index > 0 ? balls[slot.index - 1] : null;
  const cur = slot.b;
  const layers = [cur.inputs, cur.h1, cur.h2, cur.out];
  const prevLayers = prev ? [prev.inputs, prev.h1, prev.h2, prev.out] : layers.map((l) => l.map(() => 0));
  const step = THINK_S / 4;
  const activations = layers.map((layer, l) => {
    const k = ease.inOutCubic(clamp01((local - l * step) / step));
    return layer.map((v, i) => lerp(prevLayers[l][i], v, k));
  });
  // Pulse: which layer of edges is "firing" right now.
  const pulse = [0, 1, 2].map((l) => {
    const c = (l + 0.5) * step + step / 2;
    return 0.35 + 0.65 * Math.exp(-Math.pow((local - c) / (step * 0.9), 2));
  });
  const g = ease.inOutCubic(clamp01((local - GAUGE_S[0]) / (GAUGE_S[1] - GAUGE_S[0])));
  const prevOut = prev ? prev.out : [0, 0, 0, 0];
  const gauges = cur.out.map((v, i) => lerp(prevOut[i], v, g));
  const dots: PitchDot[] = [];
  for (const s of slots) {
    if (s.bounce == null || s.b.bounce == null || s.b.length == null) continue;
    if (t < s.bounce) continue;
    dots.push({length: s.b.length, line: s.b.line, result: s.b.result, runs: s.b.runs, age: t - s.bounce});
  }
  return {slot, activations, pulse, gauges, dots};
}

export type BrainCamProps = {run: string; archetype: string; speed: number};

/** A2: watch the brain think, ball by ball, and see where each decision lands. */
export const BrainCam: React.FC<BrainCamProps> = ({run, archetype, speed}) => {
  const frame = useCurrentFrame();
  const {width, height} = useVideoConfig();
  const files = useJsonFiles<NetFile | CamFile>([`brain_net_${run}.json`, `brain_cam_${run}_${archetype}.json`]);
  const net = files?.[0] as NetFile | undefined;
  const cam = files?.[1] as CamFile | undefined;
  const edges = useMemo(() => (net ? stableEdges([net.champion], net.sizes, [40, 40, 16]) : []), [net]);
  if (!net || !cam) return <AbsoluteFill />;
  const t = Math.max(0, (frame - HOLD) / FPS);
  const state = camState(cam.balls, speed, t);
  const vertical = height > width;
  const netBox = vertical ? {x: 0, y: 0, w: width, h: 1040} : {x: 0, y: 0, w: 1180, h: height};
  const gaugeSize = vertical ? 250 : 330;
  const gaugeBox = vertical ? {x: 20, y: 1040, w: width - 40, h: 250} : {x: 1180, y: 40, w: 740, h: 600};
  const pitchBox = vertical ? {x: 30, y: 1320, w: width - 60, h: 560} : {x: 1200, y: 640, w: 700, h: 410};
  const gaugeColors = [palette.teal, palette.violet, palette.amber, palette.pink];
  return (
    <AbsoluteFill style={{backgroundColor: palette.bg}}>
      <div style={{position: 'absolute', left: netBox.x, top: netBox.y, width: netBox.w, height: netBox.h}}>
        <NetworkView
          net={net.champion}
          sizes={net.sizes}
          edges={edges}
          width={netBox.w}
          height={netBox.h}
          vertical={vertical}
          activations={state.activations}
          layerPulse={state.pulse}
          edgeScale={1.4}
          nodeRadius={vertical ? 15 : 15}
          margin={vertical ? 90 : 90}
        />
      </div>
      <div
        style={{
          position: 'absolute',
          left: gaugeBox.x,
          top: gaugeBox.y,
          width: gaugeBox.w,
          height: gaugeBox.h,
          display: 'flex',
          flexWrap: 'wrap',
          justifyContent: 'space-around',
          alignContent: 'center',
        }}
      >
        {state.gauges.map((v, i) => (
          <div key={i} style={{width: vertical ? gaugeSize : gaugeSize, display: 'flex', justifyContent: 'center'}}>
            <Gauge value={v} size={gaugeSize} color={gaugeColors[i]} />
          </div>
        ))}
      </div>
      <div style={{position: 'absolute', left: pitchBox.x, top: pitchBox.y, width: pitchBox.w, height: pitchBox.h}}>
        <PitchMap2D dots={state.dots} width={pitchBox.w} height={pitchBox.h} vertical={false} />
      </div>
    </AbsoluteFill>
  );
};
