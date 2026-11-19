import React from 'react';
import {AbsoluteFill, OffthreadVideo, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {palette} from '../lib/palette';
import {display, mono} from './fonts';
import {Plan, Shot} from './types';

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
const outCubic = (v: number) => 1 - Math.pow(1 - clamp01(v), 3);

/** Closing card over the still-orbiting final swarm. */
export const EndCard: React.FC<{plan: Plan; bg?: Shot; shotStart: number}> = ({bg, shotStart}) => {
  const frame = useCurrentFrame();
  const {fps, width, height} = useVideoConfig();
  const portrait = height > width;
  const t = frame / fps;
  // Continue the previous clip exactly where it left off, then blur it back.
  const bgSrc = bg && bg.file ? (bg.src ?? 0) + (shotStart - bg.start) * bg.rate : 0;
  const blur = 4 + 14 * outCubic(t / 1.2);
  const lines = ['AN AI THAT', 'TAUGHT ITSELF', 'TO BOWL'];
  const facts = ['NEUROEVOLUTION', '192 BRAINS × 1,500 GENERATIONS', '~83 MILLION DELIVERIES'];
  return (
    <AbsoluteFill>
      {bg && bg.file ? (
        <AbsoluteFill style={{filter: `blur(${blur}px) brightness(${1 - 0.45 * outCubic(t / 1.2)})`, transform: `scale(${1.08 + 0.04 * t / 5})`}}>
          <OffthreadVideo src={staticFile(bg.file)} trimBefore={Math.round(bgSrc * fps)} playbackRate={0.25} muted style={{width: '100%', height: '100%', objectFit: 'cover'}} />
        </AbsoluteFill>
      ) : null}
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', flexDirection: 'column'}}>
        {lines.map((l, i) => {
          const p = clamp01((t - 0.15 - i * 0.18) / 0.45);
          return (
            <div
              key={l}
              style={{
                fontFamily: display,
                fontWeight: 900,
                fontSize: portrait ? 118 : 120,
                lineHeight: 1.02,
                color: i === 2 ? palette.teal : '#ffffff',
                opacity: p,
                letterSpacing: `${0.12 * (1 - outCubic(p))}em`,
                transform: `translateY(${(1 - outCubic(p)) * 40}px)`,
                textShadow: i === 2 ? `0 0 60px ${palette.teal}cc` : '0 0 40px rgba(0,0,0,0.8)',
              }}
            >
              {l}
            </div>
          );
        })}
        <div style={{height: portrait ? 70 : 50}} />
        <div style={{width: `${(portrait ? 60 : 30) * outCubic((t - 0.9) / 0.6)}%`, height: 3, background: palette.teal, boxShadow: `0 0 16px ${palette.teal}`}} />
        <div style={{height: portrait ? 50 : 36}} />
        {facts.map((f, i) => {
          const p = clamp01((t - 1.2 - i * 0.2) / 0.4);
          return (
            <div key={f} style={{fontFamily: mono, fontWeight: 700, fontSize: portrait ? 36 : 32, letterSpacing: '0.18em', color: palette.muted, opacity: p, marginTop: 10, transform: `translateY(${(1 - outCubic(p)) * 14}px)`}}>
              {f}
            </div>
          );
        })}
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
