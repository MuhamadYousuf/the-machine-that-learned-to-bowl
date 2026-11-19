import React from 'react';
import {AbsoluteFill, CalculateMetadataFunction, useCurrentFrame, useVideoConfig} from 'remotion';
import {fetchJson, OversFile, useJsonFiles} from '../lib/data';
import {outcomeColor, palette} from '../lib/palette';
import {HOLD} from '../hawkeye/hold';
import {OverReplay, OverReplayProps, pickDeliveries} from '../hawkeye/OverReplay';
import {BallEvent, ballEvents, FPS, schedule} from '../hawkeye/timeline';
import {display, mono} from './fonts';

/** One ball of batter-side data (scripts/export_batter_data.py). */
export type BatterBall = {
  ball: number;
  legal: boolean;
  result: string;
  runs: number;
  shot: string;
  batter: string;
  confidence_before: number;
  confidence_after: number | null;
  pattern_before: number;
  pattern_after: number | null;
  surprise: number;
  wicket_prob: number;
};
type BatterFile = {archetype: string; overs: {over_index: number; balls: BatterBall[]}[]};

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
const outCubic = (v: number) => 1 - Math.pow(1 - clamp01(v), 3);
const WICKETS = ['bowled', 'lbw', 'caught_behind', 'caught_field'];
const NAMES: Record<string, string> = {slogger: 'THE SLOGGER', technician: 'THE TECHNICIAN', tailender: 'THE TAILENDER'};

function label(b: BatterBall): string {
  if (b.result === 'bowled') return 'BOWLED';
  if (b.result === 'lbw') return 'LBW';
  if (b.result === 'caught_behind') return 'CAUGHT BEHIND';
  if (b.result === 'caught_field') return 'CAUGHT';
  if (b.result === 'wide') return 'WIDE';
  if (b.result === 'no_ball') return 'NO BALL';
  if (b.runs === 4) return 'FOUR';
  if (b.runs === 6) return 'SIX';
  if (b.runs > 0) return `${b.runs} RUN${b.runs > 1 ? 'S' : ''}`;
  return b.shot === 'leave' ? 'LEFT ALONE' : 'DOT BALL';
}

function chip(b: BatterBall): string {
  if (WICKETS.includes(b.result)) return 'W';
  if (b.result === 'wide') return 'wd';
  if (b.result === 'no_ball') return 'nb';
  return b.runs ? String(b.runs) : '•';
}

/**
 * Batter's-eye overlay for one over: name card, confidence meter (eases from before→after at
 * each ball's end), the ball-by-ball over tracker, the shot/result slam, and an optional
 * "pattern read" meter. Times are frame-exact with the Hawk-Eye replay underneath.
 */
export const BatterHudLayer: React.FC<{balls: BatterBall[]; events: BallEvent[]; archetype: string; showPattern?: boolean}> = ({
  balls,
  events,
  archetype,
  showPattern = false,
}) => {
  const frame = useCurrentFrame();
  const {width, height} = useVideoConfig();
  const portrait = height > width;
  const s = portrait ? (width / 1080) * 1.3 : height / 1080;
  // Which ball are we in, and has it finished?
  let current = 0;
  for (let i = 0; i < events.length; i++) if (frame >= events[i].startFrame) current = i;
  const ev = events[current];
  const b = balls[current];
  if (!b || !ev) return null;
  const done = frame >= ev.endFrame;
  const settle = outCubic((frame - ev.endFrame) / 24);
  const conf = done && b.confidence_after !== null ? b.confidence_before + (b.confidence_after - b.confidence_before) * settle : b.confidence_before;
  const pat = done && b.pattern_after !== null ? b.pattern_before + (b.pattern_after - b.pattern_before) * settle : b.pattern_before;
  const wicket = WICKETS.includes(b.result);
  const intro = outCubic((frame - (events[0]?.startFrame ?? 0) + 30) / 24);
  const resultAge = frame - ev.endFrame;
  const showResult = done && resultAge < 80;
  const pop = outCubic(resultAge / 8);
  const colour = outcomeColor(b.result, b.runs);
  const confColour = conf > 0.45 ? palette.orange : conf > 0.25 ? palette.amber : palette.teal;

  const panel: React.CSSProperties = {
    background: 'rgba(6,10,19,0.72)',
    border: `${2 * s}px solid rgba(147,167,189,0.25)`,
    borderRadius: 18 * s,
    backdropFilter: 'blur(8px)',
  };

  return (
    <AbsoluteFill style={{pointerEvents: 'none', opacity: intro}}>
      {/* name + confidence */}
      <div style={{position: 'absolute', left: (portrait ? 40 : 56) * s, top: (portrait ? 150 : 56) * s, width: (portrait ? 420 : 400) * s, padding: 22 * s, ...panel}}>
        <div style={{fontFamily: mono, fontWeight: 700, fontSize: 22 * s, letterSpacing: '0.3em', color: palette.muted}}>BATTER</div>
        <div style={{fontFamily: display, fontWeight: 900, fontSize: 44 * s, color: '#fff', lineHeight: 1.1, marginTop: 4 * s}}>{NAMES[archetype] ?? archetype.toUpperCase()}</div>
        <div style={{display: 'flex', justifyContent: 'space-between', marginTop: 18 * s, fontFamily: mono, fontWeight: 700, fontSize: 22 * s, color: palette.muted, letterSpacing: '0.15em'}}>
          <span>CONFIDENCE</span>
          <span style={{color: confColour}}>{Math.round(conf * 100)}</span>
        </div>
        <div style={{height: 16 * s, borderRadius: 8 * s, background: '#1c2a3d', marginTop: 8 * s, overflow: 'hidden'}}>
          <div style={{height: '100%', width: `${Math.min(100, (conf / 0.6) * 100)}%`, background: `linear-gradient(90deg, ${palette.teal}, ${confColour})`, boxShadow: `0 0 ${18 * s}px ${confColour}`}} />
        </div>
        {showPattern ? (
          <>
            <div style={{display: 'flex', justifyContent: 'space-between', marginTop: 16 * s, fontFamily: mono, fontWeight: 700, fontSize: 22 * s, color: palette.muted, letterSpacing: '0.15em'}}>
              <span>PATTERN READ</span>
              <span style={{color: pat > 0.8 ? palette.pink : palette.violet}}>{Math.round(pat * 100)}%</span>
            </div>
            <div style={{height: 16 * s, borderRadius: 8 * s, background: '#1c2a3d', marginTop: 8 * s, overflow: 'hidden'}}>
              <div style={{height: '100%', width: `${pat * 100}%`, background: pat > 0.8 ? palette.pink : palette.violet, boxShadow: `0 0 ${18 * s}px ${pat > 0.8 ? palette.pink : palette.violet}`}} />
            </div>
          </>
        ) : null}
      </div>

      {/* over tracker */}
      <div style={{position: 'absolute', right: (portrait ? 40 : 56) * s, top: (portrait ? 150 : 56) * s, padding: `${16 * s}px ${20 * s}px`, display: 'flex', gap: 12 * s, ...panel}}>
        {balls.map((bb, i) => {
          const finished = i < current || (i === current && done);
          const c = outcomeColor(bb.result, bb.runs);
          return (
            <div
              key={i}
              style={{
                width: 54 * s,
                height: 54 * s,
                borderRadius: 27 * s,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontFamily: display,
                fontWeight: 900,
                fontSize: 26 * s,
                color: finished ? '#06101a' : palette.muted,
                background: finished ? c : 'transparent',
                border: `${3 * s}px solid ${i === current && !done ? palette.ink : finished ? c : '#2a3a52'}`,
                boxShadow: finished ? `0 0 ${14 * s}px ${c}` : 'none',
                transform: `scale(${i === current && done ? 0.8 + 0.2 * pop : 1})`,
              }}
            >
              {finished ? chip(bb) : ''}
            </div>
          );
        })}
      </div>

      {/* result slam */}
      {showResult && !wicket ? (
        <div style={{position: 'absolute', top: portrait ? '30%' : '11%', width: '100%', display: 'flex', justifyContent: 'center', opacity: clamp01((80 - resultAge) / 14)}}>
          <div
            style={{
              fontFamily: display,
              fontWeight: 900,
              fontSize: (b.runs >= 4 ? 150 : 84) * s,
              color: b.runs >= 4 ? colour : '#eaf2fb',
              transform: `scale(${1.35 - 0.35 * pop})`,
              textShadow: `0 0 ${50 * s}px ${colour}aa, 0 ${6 * s}px 0 rgba(0,0,0,0.55)`,
              fontStyle: b.runs >= 4 ? 'italic' : 'normal',
            }}
          >
            {label(b)}
          </div>
        </div>
      ) : null}
    </AbsoluteFill>
  );
};

export type DuelProps = OverReplayProps & {batterFile: string; showPattern?: boolean};

export const duelMetadata: CalculateMetadataFunction<DuelProps & {sidecar?: unknown}> = async ({props}) => {
  const data = await fetchJson<OversFile>(props.file);
  const deliveries = pickDeliveries(data, props.overIndex, props.balls, props.overCount ?? 1);
  const {frames} = schedule(deliveries, props.speed);
  return {
    durationInFrames: frames + 2 * HOLD,
    props: {...props, sidecar: {fps: FPS, holdFrames: HOLD, balls: ballEvents(deliveries, props.speed, HOLD)}},
  };
};

/** The Hawk-Eye over replay with the batter's-eye overlay on top. */
export const Duel: React.FC<DuelProps> = (props) => {
  const files = useJsonFiles<OversFile | BatterFile>([props.file, props.batterFile]);
  if (!files) return null;
  const [overs, batter] = files as [OversFile, BatterFile];
  const deliveries = pickDeliveries(overs, props.overIndex, props.balls, props.overCount ?? 1);
  const events = ballEvents(deliveries, props.speed, HOLD);
  const over = batter.overs.find((o) => o.over_index === props.overIndex);
  return (
    <AbsoluteFill>
      <OverReplay {...props} />
      {over ? <BatterHudLayer balls={props.balls.length ? props.balls.map((i) => over.balls[i]) : over.balls} events={events} archetype={batter.archetype} showPattern={props.showPattern} /> : null}
    </AbsoluteFill>
  );
};
