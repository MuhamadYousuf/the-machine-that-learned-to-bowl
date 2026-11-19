import React from 'react';
import {useCurrentFrame, useVideoConfig} from 'remotion';
import {Batter, Bowler, FIELDERS, Fielder, Keeper} from './Mannequin';
import {Ground, L, Pitch, Pose, Stage} from './World';

export type ReelTestProps = {cam: 'side' | 'chase' | 'wide' | 'batter'};

export const ReelTest: React.FC<ReelTestProps> = ({cam}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const u = frame / fps - 1.5; // release at 1.5 s
  const poses: Record<string, Pose> = {
    side: {position: [9, 1.6, -2.5], target: [0, 1.2, -2.5], fov: 45},
    chase: {position: [1.2, 2.4, -9], target: [0, 1, 8], fov: 50},
    wide: {position: [0, 14, -24], target: [0, 0, 8], fov: 40},
    batter: {position: [5, 1.4, L - 1.3], target: [-0.4, 1.1, L - 1.3], fov: 45},
  };
  return (
    <Stage pose={poses[cam]}>
      <Ground />
      <Pitch />
      <Bowler u={u} />
      <Batter t={u} tc={0.56} shot={cam === 'batter' ? 'drive' : 'defend'} />
      <Keeper />
      {FIELDERS.map(([x, z], i) => (
        <Fielder key={i} x={x} z={z} t={u} />
      ))}
    </Stage>
  );
};
