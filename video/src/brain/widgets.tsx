import React from 'react';
import {outcomeColor, palette} from '../lib/palette';

/** Radial gauge for one action output in [-1, 1]; the needle swings to each decision. */
export const Gauge: React.FC<{value: number; size: number; color?: string}> = ({value, size, color = palette.teal}) => {
  const v = Math.max(-1, Math.min(1, value));
  const cx = size / 2;
  const cy = size * 0.58;
  const r = size * 0.4;
  const sweep = 240;
  const start = -90 - sweep / 2;
  const toXY = (deg: number, rr = r) => [cx + rr * Math.cos((deg * Math.PI) / 180), cy + rr * Math.sin((deg * Math.PI) / 180)];
  const angleOf = (x: number) => start + ((x + 1) / 2) * sweep;
  const arc = (a0: number, a1: number, rr = r) => {
    const [x0, y0] = toXY(a0, rr);
    const [x1, y1] = toXY(a1, rr);
    const large = Math.abs(a1 - a0) > 180 ? 1 : 0;
    const sweepFlag = a1 > a0 ? 1 : 0;
    return `M ${x0} ${y0} A ${rr} ${rr} 0 ${large} ${sweepFlag} ${x1} ${y1}`;
  };
  const a = angleOf(v);
  const mid = angleOf(0);
  const [nx, ny] = toXY(a, r * 0.92);
  const pinned = Math.abs(v) > 0.95;
  const fill = pinned ? palette.orange : color;
  return (
    <svg width={size} height={size * 0.9} viewBox={`0 0 ${size} ${size * 0.9}`}>
      <defs>
        <filter id="gglow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation={size * 0.025} />
        </filter>
      </defs>
      <path d={arc(start, start + sweep)} stroke="#1c2a3d" strokeWidth={size * 0.075} fill="none" strokeLinecap="round" />
      {[-1, 0, 1].map((t) => {
        const [x0, y0] = toXY(angleOf(t), r * 1.13);
        const [x1, y1] = toXY(angleOf(t), r * 1.24);
        return <line key={t} x1={x0} y1={y0} x2={x1} y2={y1} stroke={palette.muted} strokeWidth={size * 0.012} strokeOpacity={0.7} />;
      })}
      {Math.abs(a - mid) > 0.5 ? (
        <>
          <path d={arc(Math.min(mid, a), Math.max(mid, a))} stroke={fill} strokeWidth={size * 0.075} fill="none" strokeLinecap="round" filter="url(#gglow)" opacity={0.8} />
          <path d={arc(Math.min(mid, a), Math.max(mid, a))} stroke={fill} strokeWidth={size * 0.075} fill="none" strokeLinecap="round" />
        </>
      ) : null}
      <line x1={cx} y1={cy} x2={nx} y2={ny} stroke="#f2f6fb" strokeWidth={size * 0.022} strokeLinecap="round" />
      <circle cx={cx} cy={cy} r={size * 0.045} fill="#f2f6fb" />
    </svg>
  );
};

export type PitchDot = {length: number; line: number; result: string; runs: number; age: number};

/**
 * Top view of the batter's half: stumps, creases, the good-length band (6–8 m)
 * and landing dots. `vertical` puts the batter at the bottom of a tall panel.
 */
export const PitchMap2D: React.FC<{dots: PitchDot[]; width: number; height: number; vertical: boolean; dotColor?: string}> = ({
  dots,
  width,
  height,
  vertical,
  dotColor,
}) => {
  const maxLen = 14;
  const halfWidth = 1.6;
  // Map (length from stumps, line) to screen. Pitch strip is 3.05 m wide.
  const pad = 0.06;
  const map = (len: number, line: number): [number, number] => {
    const u = Math.max(-1.0, Math.min(maxLen, len)) / maxLen; // 0 at stumps
    const s = line / halfWidth; // -1..1
    if (vertical) {
      const h = height * (1 - 2 * pad);
      const w = Math.min(width * 0.9, h * ((2 * halfWidth) / (maxLen + 1.5)) * 2.2);
      return [width / 2 - s * w * 0.5, height * (1 - pad) - u * h * 0.9];
    }
    const w = width * (1 - 2 * pad);
    const hh = Math.min(height * 0.9, w * ((2 * halfWidth) / (maxLen + 1.5)) * 2.2);
    return [width * (1 - pad) - u * w * 0.9, height / 2 - s * hh * 0.5];
  };
  const rect = (l0: number, l1: number, w0: number, w1: number) => {
    const [xa, ya] = map(l0, w0);
    const [xb, yb] = map(l1, w1);
    return {x: Math.min(xa, xb), y: Math.min(ya, yb), width: Math.abs(xb - xa), height: Math.abs(yb - ya)};
  };
  const strip = rect(-1.0, maxLen, -1.525, 1.525);
  const band = rect(6, 8, -1.525, 1.525);
  const crease = rect(1.2, 1.24, -1.83, 1.83);
  const stumpsA = map(0, -0.114);
  const stumpsB = map(0, 0.114);
  const unit = vertical ? width : height;
  return (
    <svg width={width} height={height} style={{position: 'absolute', left: 0, top: 0}}>
      <defs>
        <filter id="dglow" x="-100%" y="-100%" width="300%" height="300%">
          <feGaussianBlur stdDeviation={unit * 0.012} />
        </filter>
      </defs>
      <rect {...strip} rx={6} fill="#a8936a" opacity={0.92} />
      <rect {...band} fill={palette.teal} opacity={0.18} />
      <rect {...crease} fill="#f4f1e8" />
      <line x1={stumpsA[0]} y1={stumpsA[1]} x2={stumpsB[0]} y2={stumpsB[1]} stroke="#f4f1e8" strokeWidth={unit * 0.012} strokeLinecap="round" />
      {dots.map((d, i) => {
        const [x, y] = map(d.length, d.line);
        const pop = Math.min(1, d.age / 0.5);
        const colour = dotColor ?? outcomeColor(d.result, d.runs);
        const r = unit * 0.034 * (1 + 0.8 * (1 - pop) * (1 - pop));
        return (
          <g key={i}>
            <circle cx={x} cy={y} r={r * 1.8} fill={colour} opacity={0.45} filter="url(#dglow)" />
            <circle cx={x} cy={y} r={r} fill={colour} stroke="#0b1220" strokeWidth={unit * 0.004} />
            {pop < 1 ? <circle cx={x} cy={y} r={r * (1.2 + 3 * pop)} fill="none" stroke={colour} strokeWidth={unit * 0.006} opacity={1 - pop} /> : null}
          </g>
        );
      })}
    </svg>
  );
};

/** Release-speed histogram (115–150 km/h) that fills in as balls are bowled. */
export const SpeedHistogram: React.FC<{speeds: number[]; width: number; height: number; color: string}> = ({speeds, width, height, color}) => {
  const bins = 8;
  const counts = new Array(bins).fill(0);
  for (const s of speeds) counts[Math.min(bins - 1, Math.max(0, Math.floor(((s - 115) / 35) * bins)))]++;
  const maxCount = Math.max(4, ...counts);
  const gap = width * 0.02;
  const bw = (width - gap * (bins - 1)) / bins;
  return (
    <svg width={width} height={height}>
      <line x1={0} y1={height - 2} x2={width} y2={height - 2} stroke={palette.muted} strokeOpacity={0.5} strokeWidth={3} />
      {counts.map((c, i) => {
        const h = (c / maxCount) * (height - 8);
        return <rect key={i} x={i * (bw + gap)} y={height - 4 - h} width={bw} height={h} rx={4} fill={color} opacity={0.9} />;
      })}
    </svg>
  );
};
