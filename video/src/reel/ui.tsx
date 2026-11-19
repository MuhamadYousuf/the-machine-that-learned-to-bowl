import React from 'react';
import {AbsoluteFill, useVideoConfig} from 'remotion';
import {display, mono} from '../edit/fonts';
import {Line, clamp01, outBack, ramp} from './timeline';

export const Y = '#ffd23f';
export const INK = '#ffffff';
export const PANEL = 'rgba(14,18,28,0.82)';
export const RED = '#ff4d5e';
export const GREEN = '#3ee08f';
export const TEAL = '#2fd6c0';
export const ORANGE = '#ff9a3c';
export const BLUE = '#4aa8ff';

/** Layout helpers for the two vertical formats (1080x1920 reels, 1080x1350 LinkedIn). */
export function useFmt() {
  const {width, height} = useVideoConfig();
  const tall = height / width > 1.5;
  return {w: width, h: height, tall, aspect: width / height};
}

const shadow = '0 4px 0 rgba(0,0,0,0.35), 0 0 24px rgba(0,0,0,0.35)';

/* ------------------------------------------------------------------ captions */

type Chunk = {words: Line['words']; start: number; end: number};

function chunks(line: Line, max = 4): Chunk[] {
  const groups: Line['words'][] = [];
  let cur: Line['words'] = [];
  line.words.forEach((w, i) => {
    cur.push(w);
    const punct = /[.,?!:;]$/.test(w.w);
    const next = line.words[i + 1];
    const gap = next ? next.s - w.e : 0;
    if (cur.length >= max || punct || gap > 0.35) {
      groups.push(cur);
      cur = [];
    }
  });
  if (cur.length) groups.push(cur);
  return groups.map((g, k) => ({words: g, start: g[0].s - 0.05, end: groups[k + 1] ? groups[k + 1][0].s - 0.05 : line.end + 0.25}));
}

const KEY = new Set(['cheat', 'yorker', 'yorkers', 'plan', 'learned', 'carnage', 'illegal', 'unplayable', 'nothing', 'zero', 'weights', 'random', 'parents', 'trick', 'eighteen', 'four', 'better', 'three', 'stumps', 'repeat', 'idea']);

export const Captions: React.FC<{lines: Line[]; t: number; hidden?: (t: number) => boolean}> = ({lines, t, hidden}) => {
  const {tall, h} = useFmt();
  if (hidden?.(t)) return null;
  const line = lines.find((l) => t >= l.start - 0.1 && t < l.end + 0.3);
  if (!line || !line.words.length) return null;
  const cs = chunks(line);
  const c = cs.find((x) => t >= x.start && t < x.end);
  if (!c) return null;
  const pop = outBack((t - c.start) / 0.18);
  const size = tall ? 66 : 58;
  return (
    <AbsoluteFill style={{justifyContent: 'flex-end', alignItems: 'center', paddingBottom: tall ? h * 0.2 : h * 0.13, pointerEvents: 'none'}}>
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          justifyContent: 'center',
          gap: '0 16px',
          maxWidth: '88%',
          transform: `scale(${0.85 + 0.15 * pop})`,
          opacity: clamp01(pop * 1.4),
        }}
      >
        {c.words.map((w, i) => {
          const active = t >= w.s - 0.03 && t < w.e + 0.1;
          const key = KEY.has(w.w.toLowerCase().replace(/[^a-z]/g, ''));
          const hl = key && t >= w.s - 0.03;
          return (
            <span
              key={i}
              style={{
                fontFamily: display,
                fontWeight: 900,
                fontSize: size,
                lineHeight: 1.2,
                color: hl ? '#111' : active ? Y : INK,
                background: hl ? Y : 'rgba(10,12,18,0.72)',
                padding: '2px 14px 6px',
                borderRadius: 14,
                textShadow: hl ? 'none' : '0 3px 0 rgba(0,0,0,0.5)',
                transform: active ? 'translateY(-4px)' : 'none',
                marginBottom: 10,
              }}
            >
              {w.w.replace(/^[^\w']+/, '')}
            </span>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

/* ------------------------------------------------------------------ widgets */

export const Panel: React.FC<{style?: React.CSSProperties; children: React.ReactNode; a?: number}> = ({style, children, a = 1}) => (
  <div
    style={{
      background: PANEL,
      borderRadius: 22,
      border: '2px solid rgba(255,255,255,0.12)',
      padding: '22px 28px',
      color: INK,
      boxShadow: '0 18px 40px rgba(0,0,0,0.35)',
      opacity: a,
      transform: `translateY(${(1 - a) * 30}px) scale(${0.94 + 0.06 * a})`,
      ...style,
    }}
  >
    {children}
  </div>
);

export const Label: React.FC<{children: React.ReactNode; size?: number; color?: string; style?: React.CSSProperties}> = ({children, size = 26, color = 'rgba(255,255,255,0.65)', style}) => (
  <div style={{fontFamily: mono, fontWeight: 700, fontSize: size, letterSpacing: 2, color, textTransform: 'uppercase', ...style}}>{children}</div>
);

export const Big: React.FC<{children: React.ReactNode; size?: number; color?: string; style?: React.CSSProperties}> = ({children, size = 64, color = INK, style}) => (
  <div style={{fontFamily: display, fontWeight: 900, fontSize: size, color, lineHeight: 1.0, textShadow: shadow, ...style}}>{children}</div>
);

/** "GEN 1" badge, top-left, like a race HUD. */
export const GenBadge: React.FC<{gen: number | string; sub?: string; a?: number; color?: string}> = ({gen, sub, a = 1, color = Y}) => {
  const {tall} = useFmt();
  return (
    <div style={{position: 'absolute', left: 54, top: tall ? 150 : 70, opacity: a, transform: `translateX(${(1 - a) * -40}px)`}}>
      <div style={{fontFamily: display, fontWeight: 900, fontSize: 96, color, textShadow: shadow, lineHeight: 0.95}}>GEN {gen}</div>
      {sub ? <Label size={28} color="rgba(255,255,255,0.9)" style={{textShadow: shadow, marginTop: 8}}>{sub}</Label> : null}
    </div>
  );
};

/** Slammed stamp text (rotated, with impact scale). */
export const Stamp: React.FC<{text: string; t: number; at: number; color?: string; y?: number; size?: number; rot?: number; until?: number}> = ({
  text,
  t,
  at,
  color = RED,
  y = 0.36,
  size = 120,
  rot = -6,
  until = Infinity,
}) => {
  const {h} = useFmt();
  if (t < at || t > until + 0.3) return null;
  const k = clamp01((t - at) / 0.14);
  const sc = 2.2 - 1.2 * k;
  const out = 1 - ramp(t, until, until + 0.3);
  return (
    <AbsoluteFill style={{alignItems: 'center', pointerEvents: 'none'}}>
      <div
        style={{
          position: 'absolute',
          top: h * y,
          transform: `translateY(-50%) rotate(${rot}deg) scale(${sc})`,
          opacity: k * out,
          fontFamily: display,
          fontWeight: 900,
          fontSize: size,
          color,
          border: `10px solid ${color}`,
          borderRadius: 18,
          padding: '4px 30px 10px',
          background: 'rgba(0,0,0,0.25)',
          textShadow: shadow,
          whiteSpace: 'nowrap',
        }}
      >
        {text}
      </div>
    </AbsoluteFill>
  );
};

/** A horizontal meter (like gas / steer bars). v in [-1,1] (centred) or [0,1]. */
export const Meter: React.FC<{label: string; value: string; v: number; centred?: boolean; color?: string; a?: number}> = ({label, value, v, centred, color = Y, a = 1}) => {
  const w = 300;
  return (
    <div style={{display: 'flex', alignItems: 'center', gap: 18, opacity: a, marginBottom: 14}}>
      <Label size={24} style={{width: 120}}>{label}</Label>
      <div style={{position: 'relative', width: w, height: 18, background: 'rgba(255,255,255,0.14)', borderRadius: 9}}>
        {centred ? <div style={{position: 'absolute', left: w / 2 - 1, top: -4, width: 2, height: 26, background: 'rgba(255,255,255,0.5)'}} /> : null}
        <div
          style={{
            position: 'absolute',
            top: 0,
            height: 18,
            borderRadius: 9,
            background: color,
            left: centred ? (v < 0 ? w / 2 + (v * w) / 2 : w / 2) : 0,
            width: centred ? (Math.abs(v) * w) / 2 : clamp01(v) * w,
            boxShadow: `0 0 16px ${color}`,
          }}
        />
      </div>
      <div style={{fontFamily: mono, fontWeight: 700, fontSize: 30, color: INK, minWidth: 150}}>{value}</div>
    </div>
  );
};

/** Full-frame flash. */
export const Flash: React.FC<{t: number; at: number; color?: string; dur?: number; peak?: number}> = ({t, at, color = '#ffffff', dur = 0.35, peak = 0.7}) => {
  const d = t - at;
  if (d < 0 || d > dur) return null;
  return <AbsoluteFill style={{background: color, opacity: peak * (1 - d / dur), mixBlendMode: 'screen'}} />;
};

export const Fade: React.FC<{a: number; color?: string}> = ({a, color = '#000'}) => (a > 0.001 ? <AbsoluteFill style={{background: color, opacity: a}} /> : null);

/** Title card. */
export const Title: React.FC<{a: number; sub?: string}> = ({a, sub}) => (
  <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', opacity: a}}>
    <div style={{textAlign: 'center', transform: `scale(${0.92 + 0.08 * a})`}}>
      <Label size={30} color={Y} style={{letterSpacing: 10, marginBottom: 22, textShadow: shadow}}>{sub ?? 'A neuroevolution story'}</Label>
      <Big size={104} style={{letterSpacing: -1}}>THE MACHINE</Big>
      <Big size={64} color="rgba(255,255,255,0.85)" style={{margin: '10px 0'}}>THAT LEARNED</Big>
      <Big size={140} color={Y}>TO BOWL</Big>
    </div>
  </AbsoluteFill>
);
