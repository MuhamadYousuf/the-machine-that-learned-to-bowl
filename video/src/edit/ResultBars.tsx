import React from 'react';
import {AbsoluteFill, useCurrentFrame, useVideoConfig} from 'remotion';
import {palette} from '../lib/palette';
import {display, mono} from './fonts';
import {Plan} from './types';

// Held-out re-test, normalised so a specialist (a brain trained on that batter alone) = 1.0.
const DATA = [
  {name: 'SLOGGER', v1: 0.62, v2: 1.251},
  {name: 'TECHNICIAN', v1: 0.651, v2: 1.162},
  {name: 'TAILENDER', v1: 0.664, v2: 1.179},
];

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
const outBack = (v: number) => {
  const x = clamp01(v);
  return 1 + 2.4 * Math.pow(x - 1, 3) + 1.4 * Math.pow(x - 1, 2);
};

/** Final scoreboard: v2 bars rise past the specialist line, then the v1 "cheater" bars fall short. */
export const ResultBars: React.FC<{plan: Plan; shotStart: number}> = ({plan}) => {
  const frame = useCurrentFrame();
  const {fps, width, height} = useVideoConfig();
  const portrait = height > width;
  const t = frame / fps + plan.shots.find((s) => s.clip === '@bars')!.start;
  const {v1: tV1, v2: tV2, close: tClose} = plan.result;

  const chartW = portrait ? width * 0.86 : width * 0.62;
  const chartH = portrait ? height * 0.36 : height * 0.5;
  const left = (width - chartW) / 2;
  const top = portrait ? height * 0.2 : height * 0.2;
  const maxV = 1.4;
  const y = (v: number) => top + chartH - (v / maxV) * chartH;
  const group = chartW / 3;
  const barW = group * (portrait ? 0.3 : 0.24);
  const glow = clamp01((t - tClose) / 0.3) * clamp01(1 - (t - tClose - 1.2) / 1.0) * 0.8 + 0.2;
  const lineA = clamp01((t - (tV2 - 1.2)) / 0.5);

  return (
    <AbsoluteFill style={{background: `radial-gradient(ellipse at 50% 35%, #0e1a2c 0%, ${palette.bg} 70%)`}}>
      <svg width={width} height={height} style={{position: 'absolute'}}>
        <defs>
          <linearGradient id="gv2" x1="0" y1="1" x2="0" y2="0">
            <stop offset="0" stopColor="#1f9e86" />
            <stop offset="1" stopColor="#7dffe0" />
          </linearGradient>
          <linearGradient id="gv1" x1="0" y1="1" x2="0" y2="0">
            <stop offset="0" stopColor="#a8501c" />
            <stop offset="1" stopColor="#ffab6b" />
          </linearGradient>
          <filter id="bglow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="18" />
          </filter>
        </defs>
        <line x1={left} x2={left + chartW} y1={y(0)} y2={y(0)} stroke={palette.muted} strokeOpacity={0.5} strokeWidth={3} />
        <line x1={left} x2={left + chartW * lineA} y1={y(1)} y2={y(1)} stroke={palette.amber} strokeWidth={4} strokeDasharray="16 12" />
        {DATA.map((d, i) => {
          const cx = left + group * (i + 0.5);
          const p2 = outBack((t - tV2 - i * 0.12) / 0.6);
          const p1 = outBack((t - tV1 - i * 0.1) / 0.5);
          const h2 = (d.v2 * p2 * chartH) / maxV;
          const h1 = (d.v1 * p1 * chartH) / maxV;
          const x2 = cx + barW * 0.08;
          const x1 = cx - barW * 1.08;
          return (
            <g key={d.name}>
              <rect x={x2} y={y(0) - h2} width={barW} height={Math.max(0, h2)} rx={10} fill="url(#gv2)" opacity={glow} filter="url(#bglow)" />
              <rect x={x2} y={y(0) - h2} width={barW} height={Math.max(0, h2)} rx={10} fill="url(#gv2)" />
              <rect x={x1} y={y(0) - h1} width={barW} height={Math.max(0, h1)} rx={10} fill="url(#gv1)" />
            </g>
          );
        })}
      </svg>
      {/* text layer */}
      <div style={{position: 'absolute', left: left - (portrait ? 0 : 230), top: y(1) + (portrait ? -46 : -18), opacity: lineA, fontFamily: mono, fontWeight: 700, fontSize: portrait ? 26 : 26, color: palette.amber, whiteSpace: 'nowrap', textAlign: 'left'}}>
        SPECIALIST = 1.0
      </div>
      {DATA.map((d, i) => {
        const cx = left + group * (i + 0.5);
        const p2 = outBack((t - tV2 - i * 0.12) / 0.6);
        const p1 = outBack((t - tV1 - i * 0.1) / 0.5);
        const fs = portrait ? 40 : 38;
        return (
          <React.Fragment key={d.name}>
            <div style={{position: 'absolute', left: cx - group / 2, width: group, top: y(0) + 18, textAlign: 'center', fontFamily: mono, fontWeight: 700, fontSize: portrait ? 28 : 28, letterSpacing: '0.12em', color: '#dbe8f5'}}>
              {d.name}
            </div>
            <div style={{position: 'absolute', left: cx + barW * 0.08, width: barW, top: y(d.v2 * p2) - fs - 10, textAlign: 'center', fontFamily: display, fontWeight: 900, fontSize: fs, color: '#c9fff2', opacity: clamp01(p2)}}>
              {(d.v2 * Math.min(1, p2)).toFixed(2)}
            </div>
            <div style={{position: 'absolute', left: cx - barW * 1.08, width: barW, top: y(d.v1 * p1) - fs - 10, textAlign: 'center', fontFamily: display, fontWeight: 900, fontSize: fs * 0.85, color: '#ffc29a', opacity: clamp01(p1)}}>
              {(d.v1 * Math.min(1, p1)).toFixed(2)}
            </div>
          </React.Fragment>
        );
      })}
      {/* legend */}
      <div style={{position: 'absolute', top: portrait ? height * 0.1 : height * 0.07, width: '100%', display: 'flex', justifyContent: 'center', gap: 40, fontFamily: mono, fontWeight: 700, fontSize: portrait ? 32 : 30}}>
        <div style={{display: 'flex', alignItems: 'center', gap: 12, color: palette.teal, opacity: clamp01((t - tV2 + 0.6) / 0.4)}}>
          <div style={{width: 26, height: 26, borderRadius: 6, background: palette.teal}} /> v2 · FINAL BRAIN
        </div>
        <div style={{display: 'flex', alignItems: 'center', gap: 12, color: palette.orange, opacity: clamp01((t - tV1 + 0.3) / 0.4)}}>
          <div style={{width: 26, height: 26, borderRadius: 6, background: palette.orange}} /> v1 · CHEATER
        </div>
      </div>
    </AbsoluteFill>
  );
};
