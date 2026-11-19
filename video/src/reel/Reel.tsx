import React, {useMemo} from 'react';
import {AbsoluteFill, Audio, CalculateMetadataFunction, Sequence, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {useJsonFiles} from '../lib/data';
import {Brain, Hook, Sees} from './ActOne';
import {Cheat, Fix, Gen1, Rise} from './ActTwo';
import {Outro, Plans} from './ActThree';
import {Data} from './Match';
import {SCENES, SceneId, TL, Timeline} from './timeline';
import {Captions} from './ui';

export type ReelProps = {timeline?: Timeline; audio: boolean; only?: SceneId};

export const FPS = 30;

export const reelMetadata: CalculateMetadataFunction<ReelProps> = async ({props}) => {
  const timeline: Timeline = await fetch(staticFile('reel/vo_timeline.json')).then((r) => r.json());
  const tl = new TL(timeline);
  const sc = tl.scenes();
  const [a, b] = props.only ? sc[props.only] : [0, sc.outro[1]];
  return {durationInFrames: Math.ceil((b - a) * FPS), fps: FPS, props: {...props, timeline}};
};

const FILES = [
  'overs_ga2_champion_slogger.json',
  'overs_ga2_champion_technician.json',
  'overs_ga2_champion_tailender.json',
  'overs_ga1500_champion_technician.json',
  'swarm_ga2_0001.json',
  'swarm_ga2_0050.json',
  'swarm_ga2_0250.json',
  'swarm_ga2_1500.json',
  'brain_net_ga2.json',
];

const SceneAt: React.FC<{id: SceneId; tl: TL; data: Data; t0: number; t1: number; offset: number}> = ({id, tl, data, t0, t1, offset}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const t = t0 + frame / fps;
  const p = {tl, t, data, t0, t1};
  switch (id) {
    case 'hook':
      return <Hook {...p} />;
    case 'sees':
      return <Sees {...p} />;
    case 'brain':
      return <Brain {...p} />;
    case 'gen1':
      return <Gen1 {...p} />;
    case 'cheat':
      return <Cheat {...p} />;
    case 'fix':
      return <Fix {...p} />;
    case 'rise':
      return <Rise {...p} />;
    case 'plans':
      return <Plans {...p} />;
    case 'outro':
      return <Outro {...p} />;
  }
};

export const Reel: React.FC<ReelProps> = ({timeline, audio, only}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const files = useJsonFiles<unknown>(FILES);
  const tl = useMemo(() => (timeline ? new TL(timeline) : null), [timeline]);
  const data = useMemo<Data | null>(() => {
    if (!files) return null;
    const [slog, tech, tail, v1tech, s1, s50, s250, s1500, net] = files as never[];
    return {slog, tech, tail, v1tech, swarms: {'0001': s1, '0050': s50, '0250': s250, '1500': s1500}, net};
  }, [files]);
  if (!tl || !data) return null;
  const sc = tl.scenes();
  const base = only ? sc[only][0] : 0;
  const t = base + frame / fps;
  const titleHook = tl.end('h3') + 0.1;
  return (
    <AbsoluteFill style={{background: '#000'}}>
      {SCENES.filter((s) => !only || s === only).map((id) => {
        const [a, b] = sc[id];
        return (
          <Sequence key={id} from={Math.round((a - base) * fps)} durationInFrames={Math.max(1, Math.round((b - a) * fps))} layout="none">
            <AbsoluteFill>
              <SceneAt id={id} tl={tl} data={data} t0={a} t1={b} offset={base} />
            </AbsoluteFill>
          </Sequence>
        );
      })}
      <Captions lines={tl.tl.lines} t={t} hidden={(x) => (x > titleHook && x < sc.sees[0] + 0.2) || x > tl.end('o3') + 0.2} />
      {audio ? <Audio src={staticFile('reel/mix.wav')} trimBefore={Math.round(base * fps)} /> : null}
    </AbsoluteFill>
  );
};
