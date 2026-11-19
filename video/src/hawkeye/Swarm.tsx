import React from 'react';
import {AbsoluteFill, useCurrentFrame, useVideoConfig} from 'remotion';
import {Delivery, flightEnd, SwarmFile, useJsonFiles} from '../lib/data';
import {fitnessColor} from '../lib/palette';
import {BallFlight} from './BallFlight';
import {CameraName, cameraPose} from './cameras';
import {HOLD} from './hold';
import {Scene} from './Scene';
import {Stadium} from './Stadium';
import {FPS} from './timeline';

export type SwarmProps = {
  generations: number[];
  archetype: string;
  camera: CameraName;
  speed: number;
  /** Slowly orbit the camera across the whole clip (uses the orbit camera). */
  orbit: boolean;
};

export const SWARM_LEAD = 0.5;
export const SWARM_LINGER = 1.8;
export const SWARM_FADE = 0.5;

export function swarmFile(generation: number) {
  return `swarm_ga2_${String(generation).padStart(4, '0')}.json`;
}

export function segmentFrames(deliveries: Delivery[], speed: number): number {
  const longest = Math.max(...deliveries.map(flightEnd));
  return Math.round((SWARM_LEAD + longest / speed + SWARM_LINGER + SWARM_FADE) * FPS);
}

/** Fitness on the normalized scale (0 random, 1 expert) mapped to the swarm colour ramp. */
const colourFor = (fitness: number | undefined) => fitnessColor(((fitness ?? 0) + 0.15) / 1.4);

/**
 * Every genome in a saved population bowls the same first ball at once.
 * Generation 1 sprays everywhere; by generation 1500 the swarm is a tight bunch.
 */
export const Swarm: React.FC<SwarmProps> = ({generations, archetype, camera, speed, orbit}) => {
  const frame = useCurrentFrame();
  const {width, height, durationInFrames} = useVideoConfig();
  const files = useJsonFiles<SwarmFile>(generations.map(swarmFile));
  if (!files) return <AbsoluteFill />;
  const segments = files.map((file) => file.deliveries.filter((d) => d.batter === archetype || !d.batter));
  const lengths = segments.map((s) => segmentFrames(s, speed));
  const f = Math.max(0, frame - HOLD);
  let index = 0;
  let start = 0;
  while (index < segments.length - 1 && f >= start + lengths[index]) {
    start += lengths[index];
    index++;
  }
  const local = (f - start) / FPS;
  const isLast = index === segments.length - 1;
  const fadeStart = (lengths[index] / FPS) - SWARM_FADE;
  const opacity = !isLast && local > fadeStart ? Math.max(0, 1 - (local - fadeStart) / SWARM_FADE) : 1;
  const t = Math.max(0, (local - SWARM_LEAD) * speed);
  const aspect = width / height;
  const pose = orbit ? cameraPose('orbit', aspect, 0, frame / durationInFrames) : cameraPose(camera, aspect);
  const deliveries = segments[index];
  return (
    <Scene pose={pose} bloom={1.6}>
      <Stadium />
      {deliveries.map((d, i) => {
        const colour = colourFor(d.fitness);
        return (
          <BallFlight
            key={`${index}-${i}`}
            d={d}
            t={t}
            style={{
              ballScale: pose.ballScale * 0.7,
              trailRadius: 0.0065 * pose.trailScale,
              trailOpacity: 0.85 * opacity,
              colorPre: colour,
              colorPost: colour,
              showShadow: false,
              showBall: opacity > 0.5,
              showFlash: false,
              showRing: false,
              dotOpacity: 0.25,
            }}
          />
        );
      })}
    </Scene>
  );
};
