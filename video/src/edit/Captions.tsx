import React from 'react';
import {AbsoluteFill, useCurrentFrame, useVideoConfig} from 'remotion';
import {palette} from '../lib/palette';
import {display} from './fonts';
import {CaptionLine, Plan, Word} from './types';

type Chunk = {words: Word[]; start: number; end: number; emph: Set<number>};

const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, '');

/** Split a line into short phrases (≤ maxWords), breaking after punctuation where possible. */
function chunkLine(line: CaptionLine, maxWords: number): Chunk[] {
  const emphTokens = line.emph.map((e) => e.split(/\s+/).map(norm));
  const emphIdx = new Set<number>();
  line.words.forEach((_, i) => {
    for (const toks of emphTokens) {
      if (toks.every((tok, k) => line.words[i + k] && norm(line.words[i + k].w) === tok)) toks.forEach((_, k) => emphIdx.add(i + k));
    }
  });
  const groups: number[][] = [];
  let cur: number[] = [];
  line.words.forEach((w, i) => {
    cur.push(i);
    const punct = /[.,?!:;]$/.test(w.w) || /^[.,?!:;]/.test(line.words[i + 1]?.w ?? '');
    if (cur.length >= maxWords || (punct && cur.length >= 1)) {
      groups.push(cur);
      cur = [];
    }
  });
  if (cur.length) groups.push(cur);
  return groups.map((g, k) => {
    const next = groups[k + 1];
    const start = line.words[g[0]].s - 0.06;
    const end = next ? line.words[next[0]].s - 0.06 : line.end + 0.35;
    return {words: g.map((i) => line.words[i]), start, end, emph: new Set(g.filter((i) => emphIdx.has(i)).map((i) => i - g[0]))};
  });
}

export const Captions: React.FC<{plan: Plan}> = ({plan}) => {
  const frame = useCurrentFrame();
  const {fps, width, height} = useVideoConfig();
  const t = frame / fps;
  const portrait = height > width;
  const chunks = React.useMemo(() => plan.captions.flatMap((l) => chunkLine(l, portrait ? 3 : 5)), [plan, portrait]);
  const chunk = chunks.find((c) => t >= c.start && t < c.end);
  if (!chunk) return null;
  // Big flash words (BOUNCER / YORKER) already say it; don't double up.
  if (plan.overlays.some((o) => o.type === 'flash' && t >= o.start - 0.05 && t < o.start + 0.75)) return null;
  // Brain-cam layouts keep gauges mid-frame in portrait: drop the captions onto the pitch map instead.
  const line = plan.captions.find((l) => t >= l.start - 0.1 && t < l.end + 0.4);
  const low = portrait && line && ['cheat', 'think'].includes(line.scene) && !['L12'].includes(line.id);
  const size = portrait ? 84 : 68;
  const age = t - chunk.start;
  const leave = chunk.end - t;
  const groupIn = Math.min(1, age / 0.12);
  const groupOut = Math.min(1, leave / 0.1);
  return (
    <AbsoluteFill style={{justifyContent: 'flex-start', alignItems: 'center', pointerEvents: 'none'}}>
      <div
        style={{
          position: 'absolute',
          top: low ? '80%' : portrait ? '62%' : '80%',
          left: '50%',
          width: 'max-content',
          maxWidth: portrait ? '88%' : '80%',
          display: 'flex',
          flexWrap: 'wrap',
          justifyContent: 'center',
          gap: `0 ${size * 0.36}px`,
          fontFamily: display,
          fontWeight: 900,
          fontSize: size,
          lineHeight: 1.12,
          letterSpacing: '-0.01em',
          textTransform: 'uppercase',
          opacity: groupOut,
          padding: portrait ? '14px 26px' : '10px 30px',
          borderRadius: 28,
          background: 'radial-gradient(ellipse at center, rgba(4,8,16,0.62) 0%, rgba(4,8,16,0.38) 60%, rgba(4,8,16,0) 100%)',
          transform: `translateX(-50%) translateY(${(1 - groupIn) * 18}px)`,
        }}
      >
        {chunk.words.map((w, i) => {
          const wa = t - w.s;
          const shown = wa > -0.04;
          const pop = Math.max(0, Math.min(1, (wa + 0.04) / 0.11));
          const overshoot = pop < 1 ? 0.8 + 0.24 * Math.sin((pop * Math.PI) / 1.2) : 1;
          const active = t >= w.s - 0.04 && t < w.e + 0.05;
          const emph = chunk.emph.has(i);
          const colour = emph ? palette.amber : active ? '#ffffff' : 'rgba(235,245,255,0.92)';
          return (
            <span
              key={i}
              style={{
                display: 'inline-block',
                opacity: shown ? 1 : 0,
                color: colour,
                transform: `scale(${shown ? overshoot : 0.6}) translateY(${active ? -3 : 0}px)`,
                textShadow: emph
                  ? `0 0 26px rgba(251,191,36,0.55), 0 4px 0 rgba(0,0,0,0.55), 0 0 4px rgba(0,0,0,0.9)`
                  : `0 4px 0 rgba(0,0,0,0.55), 0 0 18px rgba(0,0,0,0.85), 0 0 4px rgba(0,0,0,0.9)`,
                WebkitTextStroke: '2px rgba(6,10,19,0.55)',
              }}
            >
              {w.w.replace(/^[^A-Za-z0-9']+|[^A-Za-z0-9'?!]+$/g, '')}
            </span>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
