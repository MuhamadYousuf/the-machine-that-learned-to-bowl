import React from 'react';
import {CalculateMetadataFunction, Composition} from 'remotion';
import {fetchJson, flightEnd, OversFile, SwarmFile} from './lib/data';
import {HOLD} from './hawkeye/hold';
import {OverReplay, OverReplayProps, pickDeliveries} from './hawkeye/OverReplay';
import {ballEvents, FPS, schedule} from './hawkeye/timeline';
import {segmentFrames, Swarm, SWARM_LEAD, swarmFile, SwarmProps} from './hawkeye/Swarm';
import {Compare, CompareProps} from './hawkeye/Compare';
import {PitchBuild, pitchBuildFrames, PitchBuildProps} from './hawkeye/PitchBuild';
import {brainEntries} from './brain/register';
import {Film, filmMetadata} from './edit/Film';
import {Duel, duelMetadata} from './edit/BatterHud';
import {PlayerTest} from './players/PlayerTest';
import {Match3D, match3dMetadata} from './players/Match3D';
import {ReelTest} from './reel/ReelTest';
import {Reel, reelMetadata} from './reel/Reel';

const FORMATS = {
  '16x9': {width: 1920, height: 1080},
  '9x16': {width: 1080, height: 1920},
} as const;

// Every calculateMetadata also returns a `sidecar` prop: frame-exact events that
// render.mjs writes next to the MP4 so captions, sound effects and cuts line up.
type WithSidecar<T> = T & {sidecar?: unknown};

const overMetadata: CalculateMetadataFunction<WithSidecar<OverReplayProps>> = async ({props}) => {
  const data = await fetchJson<OversFile>(props.file);
  const deliveries = pickDeliveries(data, props.overIndex, props.balls, props.overCount ?? 1);
  const {frames} = schedule(deliveries, props.speed);
  return {
    durationInFrames: frames + 2 * HOLD,
    props: {...props, sidecar: {fps: FPS, holdFrames: HOLD, run: data.run, generation: data.generation, balls: ballEvents(deliveries, props.speed, HOLD)}},
  };
};

const swarmMetadata: CalculateMetadataFunction<WithSidecar<SwarmProps>> = async ({props}) => {
  const files = await Promise.all(props.generations.map((g) => fetchJson<SwarmFile>(swarmFile(g))));
  let cursor = HOLD;
  const segments = files.map((f) => {
    const ds = f.deliveries.filter((d) => d.batter === props.archetype || !d.batter);
    const length = segmentFrames(ds, props.speed);
    const start = cursor;
    cursor += length;
    const release = start + Math.round(SWARM_LEAD * FPS);
    return {
      generation: f.generation,
      startFrame: start,
      releaseFrame: release,
      lastBallEndFrame: release + Math.round((Math.max(...ds.map(flightEnd)) / props.speed) * FPS),
      endFrame: start + length,
      balls: ds.length,
    };
  });
  return {durationInFrames: cursor + HOLD, props: {...props, sidecar: {fps: FPS, holdFrames: HOLD, segments}}};
};

const compareMetadata: CalculateMetadataFunction<WithSidecar<CompareProps>> = async ({props}) => {
  const sides = await Promise.all([props.left, props.right].map((s) => fetchJson<OversFile>(s.file)));
  const lists = [props.left, props.right].map((s, i) => pickDeliveries(sides[i], s.overIndex, [], props.overCount));
  const frames = Math.max(...lists.map((ds) => schedule(ds, props.speed).frames));
  return {
    durationInFrames: frames + 2 * HOLD,
    props: {
      ...props,
      sidecar: {fps: FPS, holdFrames: HOLD, left: ballEvents(lists[0], props.speed, HOLD), right: ballEvents(lists[1], props.speed, HOLD)},
    },
  };
};

const pitchBuildMetadata: CalculateMetadataFunction<WithSidecar<PitchBuildProps>> = async ({props}) => {
  const data = await fetchJson<OversFile>(props.file);
  const count = data.overs.slice(0, props.overs).flatMap((o) => o.deliveries).filter((d) => d.bounce).length;
  return {
    durationInFrames: pitchBuildFrames(count, props.ballsPerSecond) + 2 * HOLD,
    props: {...props, sidecar: {fps: FPS, holdFrames: HOLD, balls: count, framesPerBall: FPS / props.ballsPerSecond}},
  };
};

const overDefaults: OverReplayProps = {
  file: 'overs_ga2_champion_technician.json',
  overIndex: 0,
  balls: [],
  camera: 'broadcast',
  speed: 1,
  accumulate: true,
  goodLengthBand: false,
};
const swarmDefaults: SwarmProps = {generations: [1, 1500], archetype: 'technician', camera: 'broadcast', speed: 0.5, orbit: false};
const compareDefaults: CompareProps = {
  left: {file: 'overs_ga1500_champion_technician.json', overIndex: 0},
  right: {file: 'overs_ga2_champion_technician.json', overIndex: 0},
  overCount: 1,
  camera: 'pitchMap',
  speed: 1,
  goodLengthBand: true,
};
const pitchBuildDefaults: PitchBuildProps = {file: 'overs_ga2_champion_technician.json', overs: 30, ballsPerSecond: 8, colorBy: 'outcome'};

/* eslint-disable @typescript-eslint/no-explicit-any */
const entries: {id: string; component: React.FC<any>; defaults: any; metadata: CalculateMetadataFunction<any>}[] = [
  {id: 'Over', component: OverReplay, defaults: overDefaults, metadata: overMetadata},
  {id: 'Swarm', component: Swarm, defaults: swarmDefaults, metadata: swarmMetadata},
  {id: 'Compare', component: Compare, defaults: compareDefaults, metadata: compareMetadata},
  {id: 'PitchBuild', component: PitchBuild, defaults: pitchBuildDefaults, metadata: pitchBuildMetadata},
  {id: 'Match3D', component: Match3D, defaults: {file: 'overs_ga2_champion_slogger.json', batterFile: 'batter_ga2_champion_slogger.json', overIndex: 17, balls: [], cameras: ['broadcast'], hud: true, archetype: 'slogger'}, metadata: match3dMetadata},
  {id: 'Duel', component: Duel, defaults: {...overDefaults, batterFile: 'batter_ga2_champion_technician.json'}, metadata: duelMetadata},
  ...brainEntries,
];

export const RemotionRoot: React.FC = () => (
  <>
    {entries.flatMap((entry) =>
      Object.entries(FORMATS).map(([name, size]) => (
        <Composition
          key={`${entry.id}-${name}`}
          id={`${entry.id}-${name}`}
          component={entry.component}
          fps={FPS}
          width={size.width}
          height={size.height}
          durationInFrames={600}
          defaultProps={entry.defaults}
          calculateMetadata={entry.metadata}
        />
      )),
    )}
    {([['short', '9x16'], ['short', '16x9'], ['long', '16x9']] as const).map(([cut, f]) => (
      <Composition key={`Doc-${cut}-${f}`} id={`Doc-${cut}-${f}`} component={Film} fps={60} width={FORMATS[f].width} height={FORMATS[f].height}
        durationInFrames={600} defaultProps={{format: f, audio: true, cut}} calculateMetadata={filmMetadata} />
    ))}
    <Composition id="PlayerTest" component={PlayerTest} fps={60} width={1920} height={1080} durationInFrames={360}
      defaultProps={{file: 'overs_ga2_champion_slogger.json', overIndex: 17, ball: 5, camera: 'side'}} />
    {(['9x16', '16x9'] as const).map((f) => (
      <Composition
        key={`Film-${f}`}
        id={`Film-${f}`}
        component={Film}
        fps={60}
        width={FORMATS[f].width}
        height={FORMATS[f].height}
        durationInFrames={600}
        defaultProps={{format: f, audio: true}}
        calculateMetadata={filmMetadata}
      />
    ))}
    {([['9x16', 1920], ['4x5', 1350]] as const).map(([f, hgt]) => (
      <Composition key={`Reel-${f}`} id={`Reel-${f}`} component={Reel} fps={30} width={1080} height={hgt} durationInFrames={300}
        defaultProps={{audio: true}} calculateMetadata={reelMetadata} />
    ))}
    <Composition id="ReelTest" component={ReelTest} fps={30} width={1080} height={1920} durationInFrames={120} defaultProps={{cam: 'chase' as const}} />
  </>
);
