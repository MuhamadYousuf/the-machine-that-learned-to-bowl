import React from 'react';
import {CalculateMetadataFunction} from 'remotion';
import {fetchJson} from '../lib/data';
import {HOLD} from '../hawkeye/hold';
import {FPS} from '../hawkeye/timeline';
import {BrainCam, BrainCamProps, CamFile, camEvents, camSchedule} from './BrainCam';
import {BrainMorph} from './BrainMorph';
import {BehaviourSwarm, DecisionMap, FamilyTree, LearningCurve, RetestBars, V1VsV2, V1V2Props, WeightHeatmaps} from './EvolutionClips';

/* eslint-disable @typescript-eslint/no-explicit-any */
type Entry = {id: string; component: React.FC<any>; defaults: any; metadata: CalculateMetadataFunction<any>};

/** Clips with a fixed length: `seconds` of animation between the 2 s holds. */
const fixed: CalculateMetadataFunction<{seconds: number}> = async ({props}) => ({
  durationInFrames: Math.round(props.seconds * FPS) + 2 * HOLD,
  props: {...props, sidecar: {fps: FPS, holdFrames: HOLD, animationFrames: Math.round(props.seconds * FPS)}},
});

const camMetadata: CalculateMetadataFunction<BrainCamProps & {sidecar?: unknown}> = async ({props}) => {
  const cam = await fetchJson<CamFile>(`brain_cam_${props.run}_${props.archetype}.json`);
  const slots = camSchedule(cam.balls, props.speed);
  return {
    durationInFrames: Math.round(slots[slots.length - 1].end * FPS) + 2 * HOLD,
    props: {...props, sidecar: {fps: FPS, holdFrames: HOLD, generation: cam.generation, seed: cam.seed, balls: camEvents(cam.balls, props.speed)}},
  };
};

const v1v2Metadata: CalculateMetadataFunction<V1V2Props & {sidecar?: unknown}> = async ({props}) => {
  const cams = await Promise.all(['ga1500', 'ga2'].map((r) => fetchJson<CamFile>(`brain_cam_${r}_${props.archetype}.json`)));
  const ends = cams.map((c) => camSchedule(c.balls, props.speed).slice(-1)[0].end);
  return {
    durationInFrames: Math.round(Math.max(...ends) * FPS) + 2 * HOLD,
    props: {...props, sidecar: {fps: FPS, holdFrames: HOLD, v1: camEvents(cams[0].balls, props.speed), v2: camEvents(cams[1].balls, props.speed)}},
  };
};

export const brainEntries: Entry[] = [
  {id: 'BrainMorph', component: BrainMorph, defaults: {run: 'ga2', mode: 'morph', seconds: 20}, metadata: fixed},
  {id: 'BrainCam', component: BrainCam, defaults: {run: 'ga2', archetype: 'technician', speed: 1}, metadata: camMetadata},
  {id: 'BehaviourSwarm', component: BehaviourSwarm, defaults: {seconds: 22}, metadata: fixed},
  {id: 'Heatmaps', component: WeightHeatmaps, defaults: {run: 'ga2', seconds: 20}, metadata: fixed},
  {id: 'Decision', component: DecisionMap, defaults: {output: 'length_category', seconds: 14}, metadata: fixed},
  {id: 'Tree', component: FamilyTree, defaults: {seconds: 14}, metadata: fixed},
  {id: 'Curve', component: LearningCurve, defaults: {run: 'ga2', seconds: 14}, metadata: fixed},
  {id: 'Retest', component: RetestBars, defaults: {seconds: 6}, metadata: fixed},
  {id: 'V1V2', component: V1VsV2, defaults: {archetype: 'technician', speed: 1}, metadata: v1v2Metadata},
];
