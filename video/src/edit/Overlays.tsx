import React from 'react';
import {AbsoluteFill, useCurrentFrame, useVideoConfig} from 'remotion';
import {palette} from '../lib/palette';
import {display, mono} from './fonts';
import {Overlay, Plan} from './types';

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
const outCubic = (v: number) => 1 - Math.pow(1 - clamp01(v), 3);
const outBack = (v: number) => {
  const c1 = 1.70158;
  const c3 = c1 + 1;
  const x = clamp01(v);
  return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2);
};
const fmt = (n: number) => Math.round(n).toLocaleString('en-US');
const accent = (c?: unknown) => (c === 'orange' ? palette.orange : c === 'red' ? palette.wicket : palette.teal);

type Ctx = {t: number; portrait: boolean; w: number; h: number};

/** In/out envelope for an overlay: 0 → 1 over `inS`, back to 0 over `outS` before the end. */
function life(o: Overlay, t: number, inS = 0.22, outS = 0.18) {
  return {a: outCubic((t - o.start) / inS) * clamp01((o.end - t) / outS), p: clamp01((t - o.start) / inS), age: t - o.start};
}

const Glass: React.FC<{children: React.ReactNode; style?: React.CSSProperties; color?: string}> = ({children, style, color = palette.teal}) => (
  <div
    style={{
      background: 'rgba(6,10,19,0.62)',
      border: `2px solid ${color}55`,
      boxShadow: `0 0 40px ${color}22, inset 0 0 30px rgba(255,255,255,0.03)`,
      borderRadius: 22,
      backdropFilter: 'blur(10px)',
      ...style,
    }}
  >
    {children}
  </div>
);

function yFor(pos: unknown, c: Ctx) {
  if (pos === 'bottom') return c.portrait ? '76%' : '50%';
  return c.portrait ? '9%' : '7%';
}

const Gen: React.FC<{o: Overlay; c: Ctx}> = ({o, c}) => {
  const {a, age} = life(o, c.t, 0.25, 0.15);
  const from = o.from as number;
  const to = o.to as number;
  const dur = (o.dur as number) ?? 0.0001;
  let value = from + (to - from) * outCubic(age / dur);
  let glitch = 0;
  if (o.rewind) {
    // Roll back from 1500 to 0 with a glitchy jitter.
    const p = clamp01(age / 0.55);
    value = Math.round(1500 * (1 - p) * (1 - p));
    glitch = (1 - p) * 10;
  }
  const big = c.portrait ? 150 : 120;
  const rolling = age < dur + 0.05 && to !== from;
  return (
    <div style={{position: 'absolute', top: c.portrait ? '7%' : '6%', width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', opacity: a}}>
      <div style={{fontFamily: mono, fontWeight: 700, fontSize: big * 0.2, letterSpacing: '0.42em', color: palette.muted, marginBottom: -4, paddingLeft: '0.42em'}}>GENERATION</div>
      <div
        style={{
          fontFamily: display,
          fontWeight: 900,
          fontSize: big,
          lineHeight: 1,
          color: '#f2fbff',
          fontVariantNumeric: 'tabular-nums',
          textShadow: `${glitch}px 0 0 rgba(255,60,90,0.8), ${-glitch}px 0 0 rgba(80,240,255,0.8), 0 0 40px ${palette.teal}66`,
          transform: `scale(${rolling ? 1.06 : 1})`,
        }}
      >
        {fmt(value)}
      </div>
    </div>
  );
};

const Stat: React.FC<{o: Overlay; c: Ctx; counter?: boolean}> = ({o, c, counter}) => {
  const {a, p, age} = life(o, c.t, 0.28, 0.2);
  const big = c.portrait ? 170 : 130;
  const value = counter ? fmt((o.to as number) * outCubic(age / ((o.dur as number) ?? 1.5))) : (o.big as string);
  // Landscape curves fill the top of frame: counters sit in the empty lower half instead.
  const top = counter && !c.portrait ? (o.pos === 'bottom' ? '54%' : '30%') : yFor(o.pos, c);
  return (
    <div style={{position: 'absolute', top, width: '100%', display: 'flex', justifyContent: 'center', opacity: a}}>
      <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center', transform: `translateY(${(1 - outBack(p)) * 40}px) scale(${0.9 + 0.1 * outBack(p)})`}}>
        <div style={{fontFamily: display, fontWeight: 900, fontSize: big, lineHeight: 0.95, color: '#ffffff', fontVariantNumeric: 'tabular-nums', textShadow: `0 0 50px ${palette.teal}88, 0 6px 0 rgba(0,0,0,0.5)`}}>{value}</div>
        <div style={{fontFamily: mono, fontWeight: 700, fontSize: big * 0.2, letterSpacing: '0.35em', color: palette.teal, marginTop: 6, paddingLeft: '0.35em'}}>{o.small as string}</div>
        {o.sub ? <div style={{fontFamily: mono, fontWeight: 500, fontSize: big * 0.14, color: palette.muted, marginTop: 6}}>{o.sub as string}</div> : null}
      </div>
    </div>
  );
};

const Chips: React.FC<{o: Overlay; c: Ctx}> = ({o, c}) => {
  const out = clamp01((o.end - c.t) / 0.18);
  const items = o.items as [string, number][];
  const colors = [palette.teal, palette.amber, palette.violet, palette.pink];
  return (
    <div style={{position: 'absolute', top: c.portrait ? '86%' : '69%', width: '100%', display: 'flex', justifyContent: 'center', gap: c.portrait ? 16 : 22, opacity: out}}>
      {items.map(([label, at], i) => {
        const p = clamp01((c.t - at + 0.03) / 0.2);
        return (
          <div
            key={label}
            style={{
              fontFamily: mono,
              fontWeight: 700,
              fontSize: c.portrait ? 34 : 34,
              padding: c.portrait ? '12px 20px' : '12px 26px',
              borderRadius: 999,
              color: colors[i % 4],
              border: `3px solid ${colors[i % 4]}`,
              background: 'rgba(6,10,19,0.7)',
              boxShadow: `0 0 26px ${colors[i % 4]}55`,
              opacity: p,
              transform: `translateY(${(1 - outBack(p)) * 30}px) scale(${0.7 + 0.3 * outBack(p)})`,
            }}
          >
            {label}
          </div>
        );
      })}
    </div>
  );
};

const Label: React.FC<{o: Overlay; c: Ctx}> = ({o, c}) => {
  const {a, p} = life(o, c.t, 0.25, 0.2);
  const col = accent(o.color);
  const text = o.text as string;
  const shown = Math.round(text.length * outCubic(p * 1.4));
  return (
    <div style={{position: 'absolute', top: o.pos === 'bottom' ? yFor('bottom', c) : c.portrait ? '5.5%' : '6%', width: '100%', display: 'flex', justifyContent: 'center', opacity: a}}>
      <Glass color={col} style={{padding: c.portrait ? '16px 28px' : '14px 30px', display: 'flex', alignItems: 'center', gap: 16}}>
        <div style={{width: 14, height: 14, borderRadius: 7, background: col, boxShadow: `0 0 16px ${col}`}} />
        <div style={{fontFamily: mono, fontWeight: 700, fontSize: c.portrait ? 34 : 32, letterSpacing: '0.12em', color: '#eaf6ff', whiteSpace: 'nowrap'}}>
          {text.slice(0, shown)}
          <span style={{opacity: shown < text.length ? 1 : 0, color: col}}>▌</span>
        </div>
      </Glass>
    </div>
  );
};

const Tag: React.FC<{o: Overlay; c: Ctx}> = ({o, c}) => {
  const {a, p} = life(o, c.t, 0.3, 0.2);
  const col = accent(o.color);
  return (
    <div style={{position: 'absolute', top: c.portrait ? '4%' : '5%', right: c.portrait ? '5%' : '3.5%', opacity: a, transform: `translateX(${(1 - outCubic(p)) * 60}px)`}}>
      <div style={{fontFamily: mono, fontWeight: 700, fontSize: c.portrait ? 30 : 28, letterSpacing: '0.2em', color: '#06101a', background: col, padding: '10px 18px', borderRadius: 10, boxShadow: `0 0 30px ${col}88`}}>
        {o.text as string}
      </div>
    </div>
  );
};

const Flash: React.FC<{o: Overlay; c: Ctx}> = ({o, c}) => {
  const age = c.t - o.start;
  if (age < 0 || age > 0.75) return null;
  const p = clamp01(age / 0.1);
  const a = clamp01((0.75 - age) / 0.25);
  const top = o.side === 'a' ? (c.portrait ? '14%' : '10%') : c.portrait ? '38%' : '36%';
  const col = o.text === 'BOUNCER' ? palette.orange : palette.pink;
  return (
    <div style={{position: 'absolute', top, width: '100%', display: 'flex', justifyContent: 'center', opacity: a}}>
      <div
        style={{
          fontFamily: display,
          fontWeight: 900,
          fontSize: c.portrait ? 150 : 150,
          color: col,
          fontStyle: 'italic',
          transform: `scale(${1.5 - 0.5 * p}) skewX(-6deg)`,
          textShadow: `0 0 60px ${col}aa, 6px 6px 0 rgba(0,0,0,0.6)`,
        }}
      >
        {o.text as string}
      </div>
    </div>
  );
};

const Stamp: React.FC<{o: Overlay; c: Ctx}> = ({o, c}) => {
  const {a, age} = life(o, c.t, 0.08, 0.2);
  const p = clamp01(age / 0.14);
  const jitter = age < 0.5 ? Math.sin(age * 90) * (0.5 - age) * 12 : 0;
  const glitchOn = Math.floor(age * 20) % 7 === 0 && age < 1.2;
  return (
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', opacity: a}}>
      <div
        style={{
          fontFamily: display,
          fontWeight: 900,
          fontSize: c.portrait ? 120 : 140,
          lineHeight: 1,
          textAlign: 'center',
          color: palette.wicket,
          border: `10px solid ${palette.wicket}`,
          borderRadius: 18,
          padding: '18px 36px',
          background: 'rgba(20,0,6,0.55)',
          transform: `rotate(-7deg) scale(${2.2 - 1.2 * outCubic(p)}) translateX(${jitter}px)`,
          textShadow: glitchOn ? `8px 0 0 rgba(80,240,255,0.8), -8px 0 0 rgba(255,220,0,0.6)` : `0 0 50px ${palette.wicket}`,
          boxShadow: `0 0 80px ${palette.wicket}88`,
          maxWidth: c.portrait ? '84%' : '70%',
        }}
      >
        {(o.text as string).split(' ').map((w, i) => (
          <div key={i}>{w}</div>
        ))}
      </div>
    </AbsoluteFill>
  );
};

const Rule: React.FC<{o: Overlay; c: Ctx}> = ({o, c}) => {
  const {a, p} = life(o, c.t, 0.35, 0.2);
  const top = o.pos === 'bottom' ? (c.portrait ? '30%' : '40%') : c.portrait ? '14%' : '14%';
  return (
    <div style={{position: 'absolute', top, width: '100%', display: 'flex', justifyContent: 'center', opacity: a}}>
      <Glass style={{padding: c.portrait ? '30px 34px' : '28px 40px', display: 'flex', alignItems: 'center', gap: 28, width: c.portrait ? '84%' : '58%', transform: `translateX(${(1 - outCubic(p)) * -120}px)`}}>
        <div style={{fontFamily: mono, fontWeight: 700, fontSize: c.portrait ? 64 : 60, color: palette.teal, textShadow: `0 0 30px ${palette.teal}`}}>{o.n as string}</div>
        <div style={{fontFamily: display, fontWeight: 800, fontSize: c.portrait ? 50 : 48, color: '#f2fbff', lineHeight: 1.1}}>{o.text as string}</div>
      </Glass>
    </div>
  );
};

const Wicket: React.FC<{o: Overlay; c: Ctx}> = ({o, c}) => {
  const {a, age} = life(o, c.t, 0.06, 0.25);
  const p = clamp01(age / 0.16);
  const sub = clamp01((age - 0.18) / 0.25);
  return (
    <div style={{position: 'absolute', top: c.portrait ? '4.5%' : '3%', width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', opacity: a}}>
      <div
        style={{
          fontFamily: display,
          fontWeight: 900,
          fontSize: c.portrait ? 190 : 150,
          lineHeight: 1,
          color: '#fff',
          letterSpacing: `${0.02 + 0.2 * (1 - outCubic(p))}em`,
          transform: `scale(${1.6 - 0.6 * outCubic(p)})`,
          textShadow: `0 0 70px ${palette.wicket}, 0 0 20px ${palette.wicket}, 0 8px 0 rgba(0,0,0,0.5)`,
        }}
      >
        WICKET
      </div>
      <div style={{marginTop: 10, fontFamily: mono, fontWeight: 700, fontSize: c.portrait ? 44 : 40, letterSpacing: '0.3em', color: palette.wicket, opacity: sub, transform: `translateY(${(1 - sub) * 20}px)`, paddingLeft: '0.3em'}}>
        {o.text as string}
      </div>
    </div>
  );
};

const Title: React.FC<{o: Overlay; c: Ctx}> = ({o, c}) => {
  const {a, age} = life(o, c.t, 0.05, 0.15);
  const words = (o.text as string).split(' ');
  return (
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', opacity: a}}>
      <div style={{display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: '0 0.3em', width: c.portrait ? '86%' : '70%', fontFamily: display, fontWeight: 900, fontSize: c.portrait ? 130 : 130, lineHeight: 1, color: '#fff', textAlign: 'center'}}>
        {words.map((w, i) => {
          const p = clamp01((age - i * 0.07) / 0.16);
          return (
            <span key={i} style={{display: 'inline-block', opacity: p, transform: `translateY(${(1 - outBack(p)) * 60}px)`, textShadow: `0 0 60px ${palette.teal}aa, 0 8px 0 rgba(0,0,0,0.5)`, color: w === 'BOWL' ? palette.teal : '#fff'}}>
              {w}
            </span>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

/** Chapter card: a full-frame number and title over a darkened shot. */
const Chapter: React.FC<{o: Overlay; c: Ctx}> = ({o, c}) => {
  const {a, p} = life(o, c.t, 0.35, 0.45);
  const line = outCubic((c.t - o.start - 0.15) / 0.6);
  return (
    <AbsoluteFill style={{background: `rgba(6,10,19,${0.72 * a})`, justifyContent: 'center', alignItems: 'center', opacity: a}}>
      <div style={{fontFamily: mono, fontWeight: 700, fontSize: c.portrait ? 44 : 40, letterSpacing: '0.5em', color: palette.teal, paddingLeft: '0.5em',
        transform: `translateY(${(1 - outCubic(p)) * 20}px)`}}>CHAPTER {o.n as string}</div>
      <div style={{width: `${(c.portrait ? 50 : 22) * line}%`, height: 3, background: palette.teal, margin: '22px 0', boxShadow: `0 0 18px ${palette.teal}`}} />
      <div style={{fontFamily: display, fontWeight: 900, fontSize: c.portrait ? 110 : 120, color: '#fff', textAlign: 'center', lineHeight: 1,
        letterSpacing: `${0.08 * (1 - outCubic(p))}em`, textShadow: '0 8px 30px rgba(0,0,0,0.6)'}}>{o.title as string}</div>
    </AbsoluteFill>
  );
};

/** Batter character card: name, colour bar and three traits typed in line by line. */
const Card: React.FC<{o: Overlay; c: Ctx}> = ({o, c}) => {
  const {a, p, age} = life(o, c.t, 0.35, 0.25);
  const col = {red: '#e0394a', blue: '#3d7bff', yellow: '#f2b51c'}[o.colour as string] ?? palette.teal;
  const lines = o.lines as string[];
  return (
    <div style={{position: 'absolute', left: c.portrait ? '6%' : '4%', bottom: c.portrait ? '6%' : '20%', opacity: a,
      transform: `translateX(${(1 - outCubic(p)) * -80}px)`}}>
      <Glass color={col} style={{padding: c.portrait ? '26px 34px' : '24px 32px', minWidth: c.portrait ? 640 : 560, borderLeft: `10px solid ${col}`}}>
        <div style={{fontFamily: mono, fontWeight: 700, fontSize: c.portrait ? 30 : 26, letterSpacing: '0.3em', color: palette.muted}}>BATTER</div>
        <div style={{fontFamily: display, fontWeight: 900, fontSize: c.portrait ? 76 : 66, color: '#fff', lineHeight: 1.05, margin: '6px 0 14px'}}>{o.name as string}</div>
        {lines.map((l, i) => {
          const q = clamp01((age - 0.35 - i * 0.28) / 0.3);
          return (
            <div key={i} style={{fontFamily: display, fontWeight: 600, fontSize: c.portrait ? 38 : 32, color: i === lines.length - 1 ? col : '#dbe6f2', opacity: q,
              transform: `translateX(${(1 - q) * 20}px)`, marginTop: 6}}>{l}</div>
          );
        })}
      </Glass>
    </div>
  );
};

export const Overlays: React.FC<{plan: Plan}> = ({plan}) => {
  const frame = useCurrentFrame();
  const {fps, width, height} = useVideoConfig();
  const t = frame / fps;
  const c: Ctx = {t, portrait: height > width, w: width, h: height};
  return (
    <AbsoluteFill style={{pointerEvents: 'none'}}>
      {plan.overlays.map((o, i) => {
        const end = o.type === 'flash' ? o.start + 0.8 : o.end;
        if (t < o.start - 0.05 || t > end) return null;
        switch (o.type) {
          case 'gen':
            return <Gen key={i} o={o} c={c} />;
          case 'stat':
            return <Stat key={i} o={o} c={c} />;
          case 'counter':
            return <Stat key={i} o={o} c={c} counter />;
          case 'chips':
            return <Chips key={i} o={o} c={c} />;
          case 'label':
            return <Label key={i} o={o} c={c} />;
          case 'tag':
            return <Tag key={i} o={o} c={c} />;
          case 'flash':
            return <Flash key={i} o={o} c={c} />;
          case 'stamp':
            return <Stamp key={i} o={o} c={c} />;
          case 'rule':
            return <Rule key={i} o={o} c={c} />;
          case 'wicket':
            return <Wicket key={i} o={o} c={c} />;
          case 'title':
            return <Title key={i} o={o} c={c} />;
          case 'chapter':
            return <Chapter key={i} o={o} c={c} />;
          case 'card':
            return <Card key={i} o={o} c={c} />;
          default:
            return null;
        }
      })}
    </AbsoluteFill>
  );
};
