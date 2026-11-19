import React, {useMemo} from 'react';
import {AbsoluteFill, useCurrentFrame, useVideoConfig} from 'remotion';
import {useJsonFiles} from '../lib/data';
import {clamp01, ease, fitnessColor, palette} from '../lib/palette';
import {HOLD} from '../hawkeye/hold';
import {FPS} from '../hawkeye/timeline';
import {CamFile, camSchedule, camState} from './BrainCam';
import {logGeneration, NetFile, netAtGeneration} from './net';
import {Gauge, PitchMap2D, SpeedHistogram} from './widgets';

/** Progress 0..1 between the opening and closing holds. */
function useProgress() {
  const frame = useCurrentFrame();
  const {durationInFrames} = useVideoConfig();
  return clamp01((frame - HOLD) / (durationInFrames - 2 * HOLD));
}

const diverging = (v: number, max: number) => {
  const t = Math.max(-1, Math.min(1, v / max));
  const target = t >= 0 ? [73, 222, 190] : [251, 113, 133];
  const base = [14, 22, 36];
  const k = Math.pow(Math.abs(t), 0.75);
  const c = base.map((b, i) => Math.round(b + (target[i] - b) * k));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
};

/* ---------------------------------------------------------------- A3 behaviour swarm */

type SwarmFrame = {gen: number; points: [number, number][]; fitness: number[]};
export type BehaviourSwarmProps = {seconds: number};

/** A3: every genome as a dot at (mean bounce length, mean release speed). Chaos at gen 1 → a tight bunch. */
export const BehaviourSwarm: React.FC<BehaviourSwarmProps> = () => {
  const p = useProgress();
  const {width, height} = useVideoConfig();
  const files = useJsonFiles<{frames: SwarmFrame[]}>(['brain_swarm_behaviour.json']);
  if (!files) return <AbsoluteFill />;
  const frames = files[0].frames;
  // Spend more time on early generations (log scale), then hold on the final bunch.
  const gen = logGeneration(Math.min(1, p / 0.85));
  let hi = frames.findIndex((f) => f.gen >= gen);
  if (hi === -1) hi = frames.length - 1;
  const lo = Math.max(0, hi - 1);
  const a = frames[lo];
  const b = frames[hi];
  const k = b.gen === a.gen ? 1 : ease.inOutCubic(clamp01((gen - a.gen) / (b.gen - a.gen)));
  const vertical = height > width;
  const pad = vertical ? {l: 90, r: 90, t: 260, b: 260} : {l: 150, r: 150, t: 110, b: 110};
  const lengthRange: [number, number] = [-0.5, 13];
  const speedRange: [number, number] = [116, 150];
  const X = (len: number) => pad.l + ((len - lengthRange[0]) / (lengthRange[1] - lengthRange[0])) * (width - pad.l - pad.r);
  const Y = (s: number) => height - pad.b - ((s - speedRange[0]) / (speedRange[1] - speedRange[0])) * (height - pad.t - pad.b);
  const champTrail: [number, number][] = frames.filter((f) => f.gen <= gen).map((f) => f.points[0]);
  const r = vertical ? 11 : 10;
  return (
    <AbsoluteFill style={{backgroundColor: palette.bg}}>
      <svg width={width} height={height}>
        <defs>
          <filter id="sglow" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation={8} />
          </filter>
        </defs>
        <rect x={X(6)} y={pad.t} width={X(8) - X(6)} height={height - pad.t - pad.b} fill={palette.teal} opacity={0.1} />
        <line x1={pad.l} y1={height - pad.b} x2={width - pad.r} y2={height - pad.b} stroke={palette.muted} strokeOpacity={0.45} strokeWidth={3} />
        <line x1={pad.l} y1={pad.t} x2={pad.l} y2={height - pad.b} stroke={palette.muted} strokeOpacity={0.45} strokeWidth={3} />
        {[0, 2, 6, 8, 10.5].map((m) => (
          <line key={m} x1={X(m)} y1={height - pad.b} x2={X(m)} y2={height - pad.b + 18} stroke={palette.muted} strokeOpacity={0.6} strokeWidth={3} />
        ))}
        {champTrail.length > 1 ? (
          <polyline
            points={champTrail.map(([l, s]) => `${X(l)},${Y(s)}`).join(' ')}
            fill="none"
            stroke={palette.amber}
            strokeOpacity={0.45}
            strokeWidth={4}
            strokeLinejoin="round"
          />
        ) : null}
        {a.points.map((pa, i) => {
          const pb = b.points[i] ?? pa;
          const l = pa[0] + (pb[0] - pa[0]) * k;
          const s = pa[1] + (pb[1] - pa[1]) * k;
          const f = a.fitness[i] + ((b.fitness[i] ?? a.fitness[i]) - a.fitness[i]) * k;
          const colour = fitnessColor((f + 0.15) / 1.4);
          return (
            <g key={i}>
              <circle cx={X(l)} cy={Y(s)} r={r * 1.9} fill={colour} opacity={0.35} filter="url(#sglow)" />
              <circle cx={X(l)} cy={Y(s)} r={r} fill={colour} opacity={0.92} />
            </g>
          );
        })}
        {(() => {
          const pa = a.points[0];
          const pb = b.points[0];
          const l = pa[0] + (pb[0] - pa[0]) * k;
          const s = pa[1] + (pb[1] - pa[1]) * k;
          return <circle cx={X(l)} cy={Y(s)} r={r * 2.3} fill="none" stroke={palette.amber} strokeWidth={5} />;
        })()}
      </svg>
    </AbsoluteFill>
  );
};

/* ---------------------------------------------------------------- A4 weight heatmaps */

export type HeatmapsProps = {run: string; seconds: number};

/** A4: the three weight matrices morphing from random noise (gen 1) to structure (gen 1500). */
export const WeightHeatmaps: React.FC<HeatmapsProps> = ({run}) => {
  const p = useProgress();
  const {width, height} = useVideoConfig();
  const files = useJsonFiles<NetFile>([`brain_net_${run}.json`]);
  if (!files) return <AbsoluteFill />;
  const file = files[0];
  const net = netAtGeneration(file, logGeneration(ease.inOutSine(Math.min(1, p / 0.9))));
  const s = file.sizes;
  const mats = [0, 1, 2].map((l) => ({rows: s[l], cols: s[l + 1], w: net.w[l]}));
  const vertical = height > width;
  // Cell size so the matrices fill the frame.
  const cell = vertical ? Math.min((width - 160) / 24, (height - 300) / (18 + 24 + 4 + 2)) : Math.min((height - 160) / 24, (width - 240) / (24 + 24 + 4));
  const gap = cell * 1.6;
  let cursor = (vertical ? height : width) / 2 - (cell * (24 + 24 + 4) + 2 * gap) / 2;
  return (
    <AbsoluteFill style={{backgroundColor: palette.bg}}>
      <svg width={width} height={height}>
        {mats.map((m, l) => {
          // Vertical: draw transposed (cols as rows) so each matrix is wide and short.
          const rows = vertical ? m.cols : m.rows;
          const cols = vertical ? m.rows : m.cols;
          const ox = vertical ? (width - cols * cell) / 2 : cursor;
          const oy = vertical ? cursor : (height - rows * cell) / 2;
          cursor += (vertical ? rows : cols) * cell + gap;
          const cells = [];
          for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
              const i = vertical ? c : r;
              const j = vertical ? r : c;
              const v = m.w[i * m.cols + j];
              cells.push(<rect key={`${r}-${c}`} x={ox + c * cell} y={oy + r * cell} width={cell - 2} height={cell - 2} rx={3} fill={diverging(v, 2.2)} />);
            }
          }
          return <g key={l}>{cells}</g>;
        })}
      </svg>
    </AbsoluteFill>
  );
};

/* ---------------------------------------------------------------- A5 decision map */

type DecisionGrid = {x: string; y: string; values: number[][]};
export type DecisionProps = {output: 'length_category' | 'speed'; seconds: number};

const CATEGORY_COLOURS: Record<string, string> = {
  '-3': '#fbbf24', // full toss
  '-2': '#f472b6', // yorker
  '-1': '#fb923c', // full
  '1': '#49debe', // good
  '2': '#a3e635', // short
  '3': '#a78bfa', // bouncer
};

const DecisionGridView: React.FC<{grid: DecisionGrid; output: string; size: number; x: number; y: number; opacity?: number}> = ({grid, output, size, x, y, opacity = 1}) => {
  const n = grid.values.length;
  const c = size / n;
  return (
    <g opacity={opacity}>
      {grid.values.map((row, r) =>
        row.map((v, k) => (
          <rect
            key={`${r}-${k}`}
            x={x + k * c}
            y={y + (n - 1 - r) * c}
            width={c + 0.5}
            height={c + 0.5}
            fill={output === 'speed' ? fitnessColor((v - 115) / 35) : CATEGORY_COLOURS[String(Math.round(v))] ?? '#334155'}
          />
        )),
      )}
      <rect x={x} y={y} width={size} height={size} fill="none" stroke={palette.muted} strokeOpacity={0.5} strokeWidth={3} rx={6} />
    </g>
  );
};

/** A5: what the brain would bowl across a sweep of two inputs — gen 1 noise vs the champion's structure. */
export const DecisionMap: React.FC<DecisionProps> = ({output}) => {
  const p = useProgress();
  const {width, height} = useVideoConfig();
  const files = useJsonFiles<Record<string, DecisionGrid>>(['brain_decision.json']);
  if (!files) return <AbsoluteFill />;
  const g1 = files[0][`gen1_${output}`];
  const ch = files[0][`champion_${output}`];
  const vertical = height > width;
  // Phase 1 (0–0.35): gen 1 alone, big. Phase 2 (0.35–0.55): crossfade to champion. Phase 3: side by side.
  const big = Math.min(width, height) * 0.82;
  const cross = ease.inOutCubic(clamp01((p - 0.35) / 0.2));
  const split = ease.inOutCubic(clamp01((p - 0.62) / 0.15));
  const small = vertical ? Math.min(width * 0.82, height * 0.4) : Math.min(height * 0.8, width * 0.42);
  const size = big + (small - big) * split;
  const centre = {x: (width - size) / 2, y: (height - size) / 2};
  const offset = vertical ? {x: 0, y: (small / 2 + 40) * split} : {x: (small / 2 + 50) * split, y: 0};
  return (
    <AbsoluteFill style={{backgroundColor: palette.bg}}>
      <svg width={width} height={height}>
        {split > 0 ? <DecisionGridView grid={g1} output={output} size={size} x={centre.x - offset.x} y={centre.y - offset.y} /> : null}
        {split === 0 ? <DecisionGridView grid={g1} output={output} size={size} x={centre.x} y={centre.y} opacity={1 - cross} /> : null}
        <DecisionGridView grid={ch} output={output} size={size} x={centre.x + offset.x} y={centre.y + offset.y} opacity={split > 0 ? 1 : cross} />
      </svg>
    </AbsoluteFill>
  );
};

/* ---------------------------------------------------------------- A6 family tree */

type TreeNode = {id: number; generation: number; fitness: number; was_elite: boolean; crossover_used: boolean; donor?: boolean};
type TreeFile = {nodes: TreeNode[]; edges: {from: number; to: number; kind: string}[]; champion_generation: number};
export type TreeProps = {seconds: number};

/** A6: the champion's ancestry, generation by generation, grown left → right (top → bottom in portrait). */
export const FamilyTree: React.FC<TreeProps> = () => {
  const p = useProgress();
  const {width, height} = useVideoConfig();
  const files = useJsonFiles<TreeFile>(['brain_tree_ga2.json']);
  const data = files?.[0];
  const prepared = useMemo(() => {
    if (!data) return null;
    const byId = new Map(data.nodes.map((n) => [n.id, n]));
    const primary = data.nodes.filter((n) => !n.donor).sort((a, b) => a.generation - b.generation);
    return {byId, primary};
  }, [data]);
  if (!data || !prepared) return <AbsoluteFill />;
  const vertical = height > width;
  const maxGen = data.champion_generation;
  const shownGen = logGeneration(Math.min(1, p / 0.9), 1, maxGen + 1) - 1;
  const fits = prepared.primary.map((n) => n.fitness);
  const fMin = Math.min(...fits, 0);
  const fMax = Math.max(...fits, 1.25);
  const pad = vertical ? 110 : 120;
  // Generation axis uses a log-ish scale so the busy early ancestry gets room.
  const gScale = (g: number) => Math.log(1 + g) / Math.log(1 + maxGen);
  const pos = (n: TreeNode): [number, number] => {
    const a = gScale(n.generation);
    const b = (n.fitness - fMin) / (fMax - fMin);
    return vertical
      ? [pad + b * (width - 2 * pad), pad + a * (height - 2 * pad)]
      : [pad + a * (width - 2 * pad), height - pad - b * (height - 2 * pad)];
  };
  const visible = prepared.primary.filter((n) => n.generation <= shownGen);
  const donors = data.edges.filter((e) => e.kind === 'crossover');
  return (
    <AbsoluteFill style={{backgroundColor: palette.bg}}>
      <svg width={width} height={height}>
        <defs>
          <filter id="tglow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation={7} />
          </filter>
        </defs>
        {donors.map((e, i) => {
          const child = prepared.byId.get(e.to);
          const donor = prepared.byId.get(e.from);
          if (!child || !donor || child.generation > shownGen) return null;
          const [x1, y1] = pos(donor);
          const [x2, y2] = pos(child);
          return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={palette.violet} strokeOpacity={0.35} strokeWidth={2.5} />;
        })}
        {visible.length > 1 ? (
          <>
            <polyline points={visible.map((n) => pos(n).join(',')).join(' ')} fill="none" stroke={palette.teal} strokeWidth={14} strokeOpacity={0.5} filter="url(#tglow)" />
            {visible.slice(1).map((n, i) => {
              const [x1, y1] = pos(visible[i]);
              const [x2, y2] = pos(n);
              return <line key={n.id} x1={x1} y1={y1} x2={x2} y2={y2} stroke={fitnessColor((n.fitness + 0.15) / 1.4)} strokeWidth={6} strokeLinecap="round" />;
            })}
          </>
        ) : null}
        {visible
          .filter((n) => n.crossover_used)
          .map((n) => {
            const [x, y] = pos(n);
            return <circle key={`c${n.id}`} cx={x} cy={y} r={5} fill={palette.violet} />;
          })}
        {visible.length ? (
          (() => {
            const [x, y] = pos(visible[visible.length - 1]);
            return (
              <g>
                <circle cx={x} cy={y} r={26} fill={palette.amber} opacity={0.35} filter="url(#tglow)" />
                <circle cx={x} cy={y} r={12} fill={palette.amber} />
              </g>
            );
          })()
        ) : null}
      </svg>
    </AbsoluteFill>
  );
};

/* ---------------------------------------------------------------- A7 learning curves */

type Curves = Record<string, {gen: number[]; best: number[]; mean: number[]; validation: [number, number][]}> & {
  retest: {v1: Record<string, number>; v2: Record<string, number>};
};
export type CurveProps = {run: string; seconds: number};

/** A7: the learning curve drawing itself: 0 = random bowling, 1 = the expert plan. */
export const LearningCurve: React.FC<CurveProps> = ({run}) => {
  const p = useProgress();
  const {width, height} = useVideoConfig();
  const files = useJsonFiles<Curves>(['brain_curves.json']);
  if (!files) return <AbsoluteFill />;
  const c = files[0][run];
  const vertical = height > width;
  const pad = vertical ? {l: 80, r: 80, t: 420, b: 420} : {l: 130, r: 130, t: 110, b: 110};
  const yMin = -0.1;
  const yMax = 1.35;
  const maxGen = c.gen[c.gen.length - 1];
  const X = (g: number) => pad.l + (g / maxGen) * (width - pad.l - pad.r);
  const Y = (v: number) => height - pad.b - ((v - yMin) / (yMax - yMin)) * (height - pad.t - pad.b);
  const reveal = ease.inOutSine(Math.min(1, p / 0.9)) * maxGen;
  const best = c.gen.map((g, i) => [g, c.best[i]] as const).filter(([g]) => g <= reveal);
  const smooth = (arr: readonly (readonly [number, number])[], w = 15) =>
    arr.map((_, i) => {
      const s = arr.slice(Math.max(0, i - w), i + 1);
      return [arr[i][0], s.reduce((a, b) => a + b[1], 0) / s.length] as const;
    });
  const mean = smooth(c.gen.map((g, i) => [g, c.mean[i]] as const).filter(([g]) => g <= reveal));
  const val = c.validation.filter(([g]) => g <= reveal);
  const line = (pts: readonly (readonly [number, number])[]) => pts.map(([g, v]) => `${X(g)},${Y(v)}`).join(' ');
  return (
    <AbsoluteFill style={{backgroundColor: palette.bg}}>
      <svg width={width} height={height}>
        <defs>
          <filter id="cglow" x="-10%" y="-10%" width="120%" height="120%">
            <feGaussianBlur stdDeviation={9} />
          </filter>
        </defs>
        {[0, 1].map((v) => (
          <line key={v} x1={pad.l} x2={width - pad.r} y1={Y(v)} y2={Y(v)} stroke={v === 1 ? palette.amber : palette.muted} strokeOpacity={0.7} strokeWidth={4} strokeDasharray="18 14" />
        ))}
        <line x1={pad.l} y1={Y(yMin)} x2={pad.l} y2={Y(yMax)} stroke={palette.muted} strokeOpacity={0.35} strokeWidth={3} />
        {best.length > 1 ? <polyline points={line(best)} fill="none" stroke={palette.teal} strokeOpacity={0.22} strokeWidth={3} /> : null}
        {mean.length > 1 ? <polyline points={line(mean)} fill="none" stroke={palette.violet} strokeOpacity={0.85} strokeWidth={6} strokeLinejoin="round" /> : null}
        {val.length > 1 ? (
          <>
            <polyline points={line(val)} fill="none" stroke={palette.teal} strokeWidth={16} strokeOpacity={0.55} filter="url(#cglow)" />
            <polyline points={line(val)} fill="none" stroke={palette.teal} strokeWidth={8} strokeLinejoin="round" strokeLinecap="round" />
            <circle cx={X(val[val.length - 1][0])} cy={Y(val[val.length - 1][1])} r={14} fill="#ffffff" />
          </>
        ) : null}
      </svg>
    </AbsoluteFill>
  );
};

/* ---------------------------------------------------------------- A9 re-test bars */

export type RetestProps = {seconds: number};

/** A9: v1 re-tested in the fixed game vs v2, per batter; dashed line = expert plan (1.0). */
export const RetestBars: React.FC<RetestProps> = () => {
  const p = useProgress();
  const {width, height} = useVideoConfig();
  const files = useJsonFiles<Curves>(['brain_curves.json']);
  if (!files) return <AbsoluteFill />;
  const {v1, v2} = files[0].retest;
  const arch = ['slogger', 'technician', 'tailender'];
  const vertical = height > width;
  const pad = vertical ? {l: 110, r: 110, t: 420, b: 420} : {l: 220, r: 220, t: 130, b: 130};
  const yMax = 1.4;
  const H = height - pad.t - pad.b;
  const Y = (v: number) => height - pad.b - (v / yMax) * H;
  const groupW = (width - pad.l - pad.r) / 3;
  const barW = groupW * 0.32;
  const grow = (i: number, which: number) => ease.outCubic(clamp01((p - 0.08 - i * 0.12 - which * 0.06) / 0.3));
  return (
    <AbsoluteFill style={{backgroundColor: palette.bg}}>
      <svg width={width} height={height}>
        <line x1={pad.l - 30} x2={width - pad.r + 30} y1={Y(0)} y2={Y(0)} stroke={palette.muted} strokeOpacity={0.6} strokeWidth={4} />
        <line x1={pad.l - 30} x2={width - pad.r + 30} y1={Y(1)} y2={Y(1)} stroke={palette.amber} strokeOpacity={0.8} strokeWidth={4} strokeDasharray="18 14" />
        {arch.map((a, i) => {
          const x0 = pad.l + i * groupW + groupW / 2;
          return [v1[a], v2[a]].map((v, which) => {
            const h = (v / yMax) * H * grow(i, which);
            const x = which === 0 ? x0 - barW - 8 : x0 + 8;
            return (
              <rect key={`${a}${which}`} x={x} y={Y(0) - h} width={barW} height={h} rx={10} fill={which === 0 ? palette.orange : palette.teal} />
            );
          });
        })}
      </svg>
    </AbsoluteFill>
  );
};

/* ---------------------------------------------------------------- A8 v1 vs v2 brains */

export type V1V2Props = {archetype: string; speed: number};

/** A8: the cheating brain vs the real bowler — output gauges, landing spots and speeds, side by side. */
export const V1VsV2: React.FC<V1V2Props> = ({archetype, speed}) => {
  const frame = useCurrentFrame();
  const {width, height} = useVideoConfig();
  const files = useJsonFiles<CamFile>([`brain_cam_ga1500_${archetype}.json`, `brain_cam_ga2_${archetype}.json`]);
  if (!files) return <AbsoluteFill />;
  const t = Math.max(0, (frame - HOLD) / FPS);
  const vertical = height > width;
  const halfW = vertical ? width : width / 2;
  const halfH = vertical ? height / 2 : height;
  const sides = files.map((cam, i) => {
    const state = camState(cam.balls, speed, t);
    const speeds = camSchedule(cam.balls, speed)
      .filter((s) => t >= s.start + 1.1)
      .map((s) => s.b.release.speed_kmh);
    return {state, speeds, colour: i === 0 ? palette.orange : palette.teal};
  });
  const gauge = vertical ? 220 : 200;
  return (
    <AbsoluteFill style={{backgroundColor: palette.bg, display: 'flex', flexDirection: vertical ? 'column' : 'row'}}>
      {sides.map((side, i) => (
        <div key={i} style={{position: 'relative', width: halfW, height: halfH, borderLeft: !vertical && i === 1 ? `4px solid #1c2a3d` : undefined, borderTop: vertical && i === 1 ? `4px solid #1c2a3d` : undefined}}>
          <div style={{position: 'absolute', left: 0, top: vertical ? 20 : 60, width: halfW, display: 'flex', justifyContent: 'space-around'}}>
            {side.state.gauges.map((v, k) => (
              <Gauge key={k} value={v} size={gauge} color={side.colour} />
            ))}
          </div>
          <div style={{position: 'absolute', left: 30, top: vertical ? 240 : 300, width: halfW - 60, height: vertical ? 420 : 460}}>
            <PitchMap2D dots={side.state.dots} width={halfW - 60} height={vertical ? 420 : 460} vertical={false} dotColor={side.colour} />
          </div>
          <div style={{position: 'absolute', left: 80, top: vertical ? 690 : 820, width: halfW - 160}}>
            <SpeedHistogram speeds={side.speeds} width={halfW - 160} height={vertical ? 230 : 200} color={side.colour} />
          </div>
        </div>
      ))}
    </AbsoluteFill>
  );
};
