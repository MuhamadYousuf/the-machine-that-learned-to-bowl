import React from 'react';
import {AbsoluteFill, CalculateMetadataFunction, useCurrentFrame, useVideoConfig} from 'remotion';
import {Delivery, fetchJson, flightEnd, OversFile, shotTime, useJsonFiles} from '../lib/data';
import {BallFlight} from '../hawkeye/BallFlight';
import {CameraPose} from '../hawkeye/cameras';
import {HOLD} from '../hawkeye/hold';
import {pickDeliveries, wicketAccents} from '../hawkeye/OverReplay';
import {Scene} from '../hawkeye/Scene';
import {Stadium} from '../hawkeye/Stadium';
import {BallEvent, FPS} from '../hawkeye/timeline';
import {BatterBall, BatterHudLayer} from '../edit/BatterHud';
import {Batter, BATTER_POS} from './Batter';
import {Bowler} from './Bowler';
import {Kit, KIT_BLUE} from './Character';
import {PlayerLights} from './PlayerLights';
import {buildSegments, RUNUP} from './timeline3d';

export type Shot3D = 'broadcast' | 'wide' | 'bowlerSide' | 'release' | 'batterSide' | 'batterFront' | 'stumps' | 'square';
export const KITS: Record<string, Kit> = {
  slogger: {top: '#b81d2c', bottom: '#8d1420', topTexture: false},
  technician: KIT_BLUE,
  tailender: {top: '#e0a61a', bottom: '#b9840d', topTexture: false},
};

export type Match3DProps = {
  file: string;
  batterFile?: string;
  overIndex: number;
  balls: number[];
  cameras: Shot3D[]; // one per ball (cycled)
  slowmo?: {ball: number; factor: number}; // slow the moment of truth on one ball
  hud: boolean;
  archetype: string;
  runup?: number; // seconds of run-up shown before each release
};

function pose(name: Shot3D, aspect: number, tau: number): CameraPose {
  const portrait = aspect < 1;
  const L = 20.12;
  const p = Math.max(0, Math.min(1, (tau + 1.5) / 2.2));
  switch (name) {
    case 'wide':
      return portrait
        ? {position: [0, 6.5, -16], target: [0, 0.6, 10], fov: 44, trailScale: 1.8, ballScale: 2.2}
        : {position: [0, 5.5, -17], target: [0, 0.7, 12], fov: 30, trailScale: 1.8, ballScale: 2.2};
    case 'bowlerSide': {
      // tracks the bowler's run-up from square, ending at the crease
      const z = -9 + 8 * p;
      return {position: [-7.5, 1.5, z - 1], target: [0.5, 1.2, z], fov: portrait ? 44 : 32, trailScale: 1, ballScale: 1.4};
    }
    case 'release':
      return portrait
        ? {position: [-0.8, 1.6, 7.5], target: [0.4, 1.6, -1.2], fov: 38, trailScale: 1, ballScale: 1.3}
        : {position: [-0.8, 1.6, 8.5], target: [0.4, 1.6, -1.2], fov: 28, trailScale: 1, ballScale: 1.3};
    case 'batterSide':
      return portrait
        ? {position: [5.2, 1.2, L - 0.5], target: [-0.4, 1.0, L - 0.9], fov: 46, trailScale: 1, ballScale: 1.2}
        : {position: [5.8, 1.25, L - 0.6], target: [-0.4, 1.0, L - 1.0], fov: 30, trailScale: 1, ballScale: 1.2};
    case 'batterFront':
      return portrait
        ? {position: [0.4, 1.5, L - 7.5], target: [-0.3, 1.0, L - 0.8], fov: 34, trailScale: 1, ballScale: 1.2}
        : {position: [0.5, 1.6, L - 9], target: [-0.3, 1.0, L - 0.8], fov: 22, trailScale: 1, ballScale: 1.2};
    case 'stumps':
      return portrait
        ? {position: [0.15, 0.45, L + 2.6], target: [0, 0.7, 8], fov: 42, trailScale: 0.9, ballScale: 1.2}
        : {position: [0.4, 0.42, L + 2.4], target: [0, 0.7, 8], fov: 28, trailScale: 0.9, ballScale: 1.2};
    case 'square':
      return {position: [-11, 2.2, 10], target: [0, 1.0, 10], fov: portrait ? 70 : 52, trailScale: 1.3, ballScale: 1.6};
    case 'broadcast':
    default:
      return portrait
        ? {position: [0, 7.5, -10], target: [0, 0.4, 13.5], fov: 34, trailScale: 1.6, ballScale: 2.0}
        : {position: [0, 6.2, -12.5], target: [0, 0.7, 15.5], fov: 21, trailScale: 1.6, ballScale: 2.0};
  }
}

export const match3dMetadata: CalculateMetadataFunction<Match3DProps & {sidecar?: unknown}> = async ({props}) => {
  const data = await fetchJson<OversFile>(props.file);
  const ds = pickDeliveries(data, props.overIndex, props.balls, 1);
  const segs = buildSegments(ds, props);
  const frames = segs.reduce((a, s) => a + s.frames, 0);
  return {
    durationInFrames: frames + 2 * HOLD,
    props: {...props, sidecar: {fps: FPS, holdFrames: HOLD, balls: segs.map((s) => ({ball: s.index, startFrame: HOLD + s.start, releaseFrame: HOLD + s.release, endFrame: HOLD + s.end, result: s.d.result, runs: s.d.runs}))}},
  };
};

export const Match3D: React.FC<Match3DProps> = (props) => {
  const frame = useCurrentFrame();
  const {width, height} = useVideoConfig();
  const names = [props.file, ...(props.batterFile ? [props.batterFile] : [])];
  const files = useJsonFiles<unknown>(names);
  if (!files) return <AbsoluteFill />;
  const data = files[0] as OversFile;
  const ds = pickDeliveries(data, props.overIndex, props.balls, 1);
  const segs = buildSegments(ds, props);
  const f = Math.max(0, frame - HOLD);
  let seg = segs[segs.length - 1];
  for (const s of segs) if (f < s.start + s.frames) { seg = s; break; }
  const local = Math.min(f - seg.start, seg.frames - 1) / FPS;
  const tau = frame < HOLD ? -(props.runup ?? RUNUP) : seg.tau(local);
  const acc = wicketAccents(seg.d, Math.max(0, tau));
  const cam = props.cameras[seg.index % props.cameras.length];
  const kit = KITS[props.archetype] ?? KIT_BLUE;

  let hud: React.ReactNode = null;
  if (props.hud && files[1]) {
    const bf = files[1] as {overs: {over_index: number; balls: BatterBall[]}[]};
    const over = bf.overs.find((o) => o.over_index === props.overIndex);
    if (over) {
      const balls = props.balls.length ? props.balls.map((i) => over.balls[i]) : over.balls;
      const events = segs.map((s) => ({startFrame: HOLD + s.start, endFrame: HOLD + s.end} as BallEvent));
      hud = <BatterHudLayer balls={balls} events={events} archetype={props.archetype} />;
    }
  }
  return (
    <AbsoluteFill>
      <Scene pose={pose(cam, width / height, tau)} bloom={0.7} shadows>
        <Stadium goodLengthBand={false} batterStumpsTilt={acc.tilt} bailLift={acc.bail} wicketZone={acc.zone} />
        <PlayerLights />
        <Bowler sinceRelease={tau} />
        <Batter d={seg.d} t={tau} kit={kit} />
        {tau >= 0 ? <BallFlight d={seg.d} t={tau} style={{ballScale: 1.25, trailRadius: 0.016}} /> : null}
      </Scene>
      {hud}
    </AbsoluteFill>
  );
};

export {BATTER_POS};
