import React from 'react';
import {AbsoluteFill, useCurrentFrame, useVideoConfig} from 'remotion';
import {OversFile, toScene, useJsonFiles} from '../lib/data';
import {outcomeColor, palette} from '../lib/palette';
import {cameraPose} from './cameras';
import {HOLD} from './hold';
import {Scene} from './Scene';
import {Stadium} from './Stadium';
import {FPS} from './timeline';

export type PitchBuildProps = {file: string; overs: number; ballsPerSecond: number; colorBy: 'outcome' | 'single'};

export function pitchBuildFrames(count: number, ballsPerSecond: number) {
  return Math.round((count / ballsPerSecond) * FPS) + FPS;
}

/** Top-down pitch map: landing spots of many deliveries accumulate into the bowler's plan. */
export const PitchBuild: React.FC<PitchBuildProps> = ({file, overs, ballsPerSecond, colorBy}) => {
  const frame = useCurrentFrame();
  const {width, height} = useVideoConfig();
  const files = useJsonFiles<OversFile>([file]);
  if (!files) return <AbsoluteFill />;
  const deliveries = files[0].overs.slice(0, overs).flatMap((o) => o.deliveries).filter((d) => d.bounce);
  const elapsed = Math.max(0, frame - HOLD) / FPS;
  const shown = elapsed * ballsPerSecond;
  const pose = cameraPose('pitchMap', width / height);
  return (
    <Scene pose={pose} bloom={1.2}>
      <Stadium goodLengthBand />
      {deliveries.map((d, i) => {
        const age = shown - i;
        if (age < 0) return null;
        const pop = Math.min(1, age / 0.6);
        const scale = 1 + 1.6 * (1 - pop) * (1 - pop);
        const [x, , z] = toScene(d.bounce![0], d.bounce![1], 0);
        const colour = colorBy === 'outcome' ? outcomeColor(d.result, d.runs) : palette.trailPost;
        return (
          <group key={i} position={[x, 0.008 + i * 0.00002, z]} rotation={[-Math.PI / 2, 0, 0]}>
            <mesh scale={scale}>
              <circleGeometry args={[0.075, 28]} />
              <meshBasicMaterial color={colour} transparent opacity={0.9} toneMapped={false} />
            </mesh>
            {pop < 1 ? (
              <mesh>
                <ringGeometry args={[0.09 + pop * 0.35, 0.11 + pop * 0.37, 40]} />
                <meshBasicMaterial color={colour} transparent opacity={(1 - pop) * 0.8} toneMapped={false} />
              </mesh>
            ) : null}
          </group>
        );
      })}
    </Scene>
  );
};
