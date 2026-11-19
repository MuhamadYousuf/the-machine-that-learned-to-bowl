import React from 'react';
import {AbsoluteFill, Audio, CalculateMetadataFunction, interpolate, OffthreadVideo, Sequence, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {palette} from '../lib/palette';
import {Captions} from './Captions';
import {EndCard} from './EndCard';
import {Overlays} from './Overlays';
import {ResultBars} from './ResultBars';
import {Plan, Shot} from './types';

export type FilmProps = {format: '9x16' | '16x9'; plan?: Plan; audio: boolean; cut?: string};

export const filmMetadata: CalculateMetadataFunction<FilmProps> = async ({props}) => {
  const base = props.cut ? `edit/${props.cut}/` : 'edit/';
  const plan: Plan = await fetch(staticFile(`${base}plan_${props.format}.json`)).then((r) => r.json());
  return {durationInFrames: plan.durationInFrames, fps: plan.fps, props: {...props, plan}};
};

const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;

/** Camera shake for the first moments of each wicket / stamp / boom. */
function shakeAt(plan: Plan, t: number, fps: number): [number, number, number] {
  let x = 0;
  let y = 0;
  let r = 0;
  const hits = [
    ...plan.overlays.filter((o) => o.type === 'wicket' || o.type === 'stamp').map((o) => ({t: o.start, a: o.type === 'stamp' ? 16 : 22})),
    ...plan.sfx.filter((s) => s.type === 'boom').map((s) => ({t: s.t, a: 26})),
    ...plan.sfx.filter((s) => s.type === 'hit').map((s) => ({t: s.t, a: 9})),
  ];
  for (const h of hits) {
    const dt = t - h.t;
    if (dt < 0 || dt > 0.6) continue;
    const decay = Math.exp(-dt * 7) * h.a;
    const f = Math.floor(dt * fps);
    x += Math.sin(f * 2.1 + h.t * 13) * decay;
    y += Math.cos(f * 2.9 + h.t * 7) * decay * 0.8;
    r += Math.sin(f * 1.7) * decay * 0.03;
  }
  return [x, y, r];
}

const ShotView: React.FC<{shot: Shot; plan: Plan; index: number; prev?: Shot}> = ({shot, plan, index, prev}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const dur = Math.max(1, Math.round((shot.end - shot.start) * fps));
  const p = frame / dur;
  const sceneStart = index === 0 || plan.shots[index - 1].scene !== shot.scene;
  const [z0, z1] = shot.zoom;
  const ease = (v: number) => -(Math.cos(Math.PI * v) - 1) / 2;
  const zoom = z0 + (z1 - z0) * ease(p);
  // Scene changes punch in out of a zoom-blur; cuts inside a scene get a short exposure pop.
  const inT = interpolate(frame, [0, sceneStart ? 12 : 6], [0, 1], clamp);
  const punch = sceneStart ? 1 + 0.14 * (1 - ease(inT)) : 1 + 0.03 * (1 - inT);
  const blur = sceneStart ? 16 * (1 - inT) : 0;
  const pop = sceneStart ? 1 + 0.6 * (1 - inT) : 1 + 0.25 * (1 - inT);
  const grade =
    shot.grade === 'red'
      ? 'saturate(1.15) contrast(1.08) sepia(0.25) hue-rotate(-18deg)'
      : shot.grade === 'dim'
        ? 'brightness(0.42) blur(7px) saturate(0.7)'
        : '';
  const filter = `${grade} brightness(${pop}) ${blur > 0.1 ? `blur(${blur}px)` : ''}`;
  let body: React.ReactNode;
  if (shot.clip === '@bars') {
    body = <ResultBars plan={plan} shotStart={shot.start} />;
  } else if (shot.clip === '@endcard') {
    body = <EndCard plan={plan} bg={prev} shotStart={shot.start} />;
  } else {
    body = (
      <OffthreadVideo
        src={staticFile(shot.file!)}
        trimBefore={Math.round((shot.src ?? 0) * fps)}
        playbackRate={shot.rate}
        muted
        style={{width: '100%', height: '100%', objectFit: 'cover'}}
      />
    );
  }
  return (
    <AbsoluteFill style={{transform: `scale(${zoom * punch})`, filter}}>
      {body}
      {shot.grade === 'red' ? <AbsoluteFill style={{background: 'radial-gradient(ellipse at center, rgba(255,40,60,0) 40%, rgba(255,30,60,0.28) 100%)'}} /> : null}
    </AbsoluteFill>
  );
};

/** White/teal flash at scene changes, red at wickets. */
const Flashes: React.FC<{plan: Plan}> = ({plan}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const t = frame / fps;
  let white = 0;
  let red = 0;
  for (const s of plan.sections.slice(1)) {
    const dt = t - s.start;
    if (dt >= 0 && dt < 0.25) white = Math.max(white, 0.22 * (1 - dt / 0.25));
  }
  for (const o of plan.overlays) {
    if (o.type !== 'wicket' && o.type !== 'stamp') continue;
    const dt = t - o.start;
    if (dt >= 0 && dt < 0.45) red = Math.max(red, 0.45 * (1 - dt / 0.45));
  }
  for (const s of plan.sfx) {
    if (s.type !== 'boom') continue;
    const dt = t - s.t;
    if (dt >= 0 && dt < 0.4) white = Math.max(white, 0.6 * (1 - dt / 0.4));
  }
  return (
    <>
      {white > 0 ? <AbsoluteFill style={{background: '#eafff9', opacity: white, mixBlendMode: 'screen'}} /> : null}
      {red > 0 ? <AbsoluteFill style={{background: palette.wicket, opacity: red, mixBlendMode: 'screen'}} /> : null}
    </>
  );
};

/** Animated film grain + vignette so every source shares one look. */
const Finish: React.FC = () => {
  const frame = useCurrentFrame();
  const seed = frame % 7;
  return (
    <>
      <AbsoluteFill style={{background: 'radial-gradient(ellipse at center, rgba(0,0,0,0) 55%, rgba(0,0,0,0.5) 100%)', pointerEvents: 'none'}} />
      <AbsoluteFill style={{opacity: 0.07, mixBlendMode: 'overlay'}}>
        <svg width="100%" height="100%">
          <filter id={`grain${seed}`}>
            <feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves={2} seed={seed} stitchTiles="stitch" />
          </filter>
          <rect width="100%" height="100%" filter={`url(#grain${seed})`} />
        </svg>
      </AbsoluteFill>
    </>
  );
};

const Progress: React.FC<{plan: Plan}> = ({plan}) => {
  const frame = useCurrentFrame();
  const p = frame / plan.durationInFrames;
  return (
    <div style={{position: 'absolute', left: 0, top: 0, height: 6, width: `${p * 100}%`, background: `linear-gradient(90deg, ${palette.teal}, #b9fff0)`, boxShadow: `0 0 14px ${palette.teal}`, opacity: 0.85}} />
  );
};

export const Film: React.FC<FilmProps> = ({plan, audio, cut}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  if (!plan) return null;
  const [sx, sy, sr] = shakeAt(plan, frame / fps, fps);
  return (
    <AbsoluteFill style={{background: palette.bg}}>
      <AbsoluteFill style={{transform: `translate(${sx}px, ${sy}px) rotate(${sr}deg) scale(${sx || sy ? 1.03 : 1})`}}>
        {plan.shots.map((shot, i) => {
          const from = Math.round(shot.start * fps);
          const dur = Math.max(1, Math.round(shot.end * fps) - from);
          return (
            <Sequence key={i} from={from} durationInFrames={dur} layout="none">
              <AbsoluteFill>
                <ShotView shot={shot} plan={plan} index={i} prev={plan.shots[i - 1]} />
              </AbsoluteFill>
            </Sequence>
          );
        })}
        <Flashes plan={plan} />
      </AbsoluteFill>
      <Finish />
      <Overlays plan={plan} />
      <Captions plan={plan} />
      <Progress plan={plan} />
      <AbsoluteFill style={{background: '#000', opacity: interpolate(frame, [plan.durationInFrames - 50, plan.durationInFrames - 2], [0, 1], clamp)}} />
      {audio ? <Audio src={staticFile(cut ? `edit/${cut}/mix.wav` : 'edit/mix.wav')} /> : null}
    </AbsoluteFill>
  );
};
