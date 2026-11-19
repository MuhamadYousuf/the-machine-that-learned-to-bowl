import React from 'react';
import {AbsoluteFill, useCurrentFrame, useVideoConfig} from 'remotion';
import {OversFile, useJsonFiles} from '../lib/data';
import {BallFlight} from '../hawkeye/BallFlight';
import {CameraPose} from '../hawkeye/cameras';
import {Scene} from '../hawkeye/Scene';
import {Stadium} from '../hawkeye/Stadium';
import {Bowler} from './Bowler';
import {Batter} from './Batter';
import {PlayerLights} from './PlayerLights';

const POSES: Record<string, CameraPose> = {
  side: {position: [-9, 1.6, -1.5], target: [0.3, 1.2, -1.5], fov: 34, trailScale: 1, ballScale: 1.4},
  behind: {position: [1.2, 2.4, -9], target: [0.2, 1.4, 4], fov: 40, trailScale: 1, ballScale: 1.4},
  front: {position: [-0.6, 1.5, 9], target: [0.3, 1.4, -1.5], fov: 30, trailScale: 1, ballScale: 1.4},
  overview: {position: [40, 30, -20], target: [0, 0, -15], fov: 50, trailScale: 3, ballScale: 3},
  bat_side: {position: [5.5, 1.3, 19.6], target: [-0.4, 1.0, 19.3], fov: 32, trailScale: 1, ballScale: 1.2},
  bat_front: {position: [0.6, 1.7, 11], target: [-0.3, 1.0, 19.2], fov: 22, trailScale: 1, ballScale: 1.2},
  broadcast: {position: [0, 4.2, -11], target: [0, 0.8, 17], fov: 26, trailScale: 1.4, ballScale: 1.6},
  armSide: {position: [-5.5, 1.9, -1.0], target: [0.4, 1.6, -1.0], fov: 30, trailScale: 1, ballScale: 1.2},
  armFront: {position: [0.6, 1.9, 5.5], target: [0.4, 1.6, -1.2], fov: 30, trailScale: 1, ballScale: 1.2},
  batClose: {position: [3.6, 1.15, 19.0], target: [-0.35, 0.95, 19.1], fov: 34, trailScale: 0.8, ballScale: 1.0},
  batBowler: {position: [0.2, 1.35, 15.2], target: [-0.35, 0.95, 19.2], fov: 32, trailScale: 0.8, ballScale: 1.0},
  probe: {position: [5, 1.5, -18.6], target: [0.5, 1, -18.6], fov: 45, trailScale: 1, ballScale: 1},
};

/** Check the bowler model against a real simulated delivery: lead-in, release, flight. */
export const PlayerTest: React.FC<{file: string; overIndex: number; ball: number; camera: string}> = ({file, overIndex, ball, camera}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const data = useJsonFiles<OversFile>([file]);
  if (!data) return <AbsoluteFill />;
  const d = data[0].overs[overIndex].deliveries[ball];
  const t = frame / fps - 3.0; // release at 3 s
  return (
    <Scene pose={POSES[camera]} bloom={0.6} shadows>
      <Stadium goodLengthBand={false} />
      <PlayerLights />
      <Bowler sinceRelease={t} />
      <Batter d={d} t={t} />
      {camera === 'overview' ? <mesh position={[0.51, 1, -41.77]}><boxGeometry args={[1, 2, 1]} /><meshBasicMaterial color="red" /></mesh> : null}
      {t >= 0 ? <BallFlight d={d} t={t} /> : null}
    </Scene>
  );
};
