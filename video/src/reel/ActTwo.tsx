import React, {useMemo} from 'react';
import {AbsoluteFill} from 'remotion';
import {display, mono} from '../edit/fonts';
import {Delivery, flightEnd} from '../lib/data';
import {cams, Data, drift, Match, mix} from './Match';
import {Lane, LaneTag, Nets, lerpPose, pickLanes, topDown} from './Nets';
import {TL, clamp01, lerp, ramp} from './timeline';
import {Big, Fade, Flash, GREEN, GenBadge, Label, ORANGE, Panel, RED, Stamp, Y, useFmt} from './ui';
import {L, Pose} from './World';

type P = {tl: TL; t: number; data: Data; t0: number; t1: number};

const COLS_TALL = 8;

function useLanes(data: Data, gen: string, seed = 1) {
  const {tall} = useFmt();
  return useMemo(() => pickLanes(data.swarms[gen].deliveries, tall ? 16 : 14, tall ? COLS_TALL : 7, 3, seed), [data, gen, tall, seed]);
}

const runsPerOver = (ds: Delivery[]) => (ds.reduce((a, d) => a + (d.runs ?? 0), 0) / ds.length) * 6;
const stumpsPct = (ds: Delivery[]) => (ds.filter((d) => d.hitsStumps).length / ds.length) * 100;

/* ================================================================== GEN 1 */

export const Gen1: React.FC<P> = ({tl, t, data, t0}) => {
  const {aspect, tall} = useFmt();
  const lanes = useLanes(data, '0001', 3);
  const top = topDown(lanes, aspect);
  // crane: start low behind the first lane, rise to the overview
  const first = lanes[lanes.length - 1];
  const low: Pose = {position: [first.x + 0.8, 2.2, first.z - 14], target: [first.x, 1, first.z + 12], fov: 50};
  const k = ramp(t, t0 + 0.3, t0 + 3.0);
  const pose = lerpPose(low, top, k * k);
  const best = [...lanes].sort((a, b) => b.fitness - a.fitness).slice(0, 3);
  const hlAt = tl.word('g3', 'few');
  const parentsAt = tl.word('g4', 'parents');
  const childAt = tl.word('g4', 'children');
  const g5 = tl.at('g5');
  const gens = [tl.word('g5', 'worse'), tl.word('g5', 'better'), tl.word('g5', 'best'), tl.word('g5', 'repeat')];
  const genNow = 1 + gens.filter((x) => t >= x).length + (t >= childAt ? 1 : 0);
  const childK = ramp(t, childAt, childAt + 0.4);
  const hl = ramp(t, hlAt, hlAt + 0.4) * (1 - ramp(t, g5 - 0.2, g5 + 0.2));
  const rpo = runsPerOver(data.swarms['0001'].deliveries);
  const carnage = tl.word('g2', 'carnage');
  // during "repeat" montage, the brains flicker into new generations quickly
  const kitShift = t >= childAt ? 70 + gens.filter((x) => t >= x).length * 53 : 0;
  return (
    <AbsoluteFill>
      <Nets
        lanes={lanes}
        t={t - t0}
        pose={drift(pose, t, 0.04)}
        kitShift={kitShift}
        looks={(lane) => {
          const isBest = best.includes(lane);
          return {box: isBest ? hl : 0, dim: !isBest ? hl * 0.9 : 0};
        }}
      />
      {best.map((lane, i) => (
        <LaneTag key={i} pose={pose} lane={lane} a={hl * (t > parentsAt ? 1 : 0.0) + (t < parentsAt ? hl : 0)} text={t > parentsAt ? 'PARENT' : `#${i + 1}`} />
      ))}
      <GenBadge gen={genNow} sub={`192 random brains`} a={ramp(t, tl.word('g1', 'random') - 0.2, tl.word('g1', 'random') + 0.3)} />
      {t > carnage - 0.1 && t < hlAt ? (
        <div style={{position: 'absolute', right: 50, top: tall ? 160 : 70, textAlign: 'right', opacity: ramp(t, carnage - 0.1, carnage + 0.2)}}>
          <Big size={110} color={RED}>{rpo.toFixed(1)}</Big>
          <Label size={26} color="#fff" style={{textShadow: '0 2px 6px #000'}}>runs / over</Label>
        </div>
      ) : null}
      <Stamp t={t} at={carnage} until={carnage + 1.3} text="CARNAGE" color={RED} y={0.42} />
      {t > childAt - 0.2 && t < g5 + 0.4 ? <MutationCard t={t} at={childAt} /> : null}
      <Flash t={t} at={childAt} peak={0.35} />
      {gens.map((g, i) => (
        <Flash key={i} t={t} at={g} peak={0.25} dur={0.25} />
      ))}
    </AbsoluteFill>
  );
};

/** A parent's weights copied to a child, a few cells nudged at random. */
const MutationCard: React.FC<{t: number; at: number}> = ({t, at}) => {
  const {tall, h} = useFmt();
  const n = 12 * 6;
  const a = ramp(t, at - 0.2, at + 0.2);
  const cells = Array.from({length: n}, (_, i) => {
    const base = Math.sin(i * 12.9898) * 0.5 + 0.5;
    const mutated = (i * 37) % 11 === 0;
    const flick = mutated ? Math.sin((t - at) * 12 + i) * 0.5 * ramp(t, at + 0.3, at + 0.6) : 0;
    return {v: Math.max(0, Math.min(1, base + flick)), mutated: mutated && t > at + 0.3};
  });
  return (
    <div style={{position: 'absolute', right: 40, top: tall ? h * 0.52 : h * 0.5}}>
      <Panel a={a} style={{padding: '14px 18px'}}>
        <Label size={20} color={Y} style={{marginBottom: 8}}>child = parent + tiny changes</Label>
        <div style={{display: 'grid', gridTemplateColumns: 'repeat(12, 22px)', gap: 4}}>
          {cells.map((c, i) => (
            <div key={i} style={{width: 22, height: 22, borderRadius: 4, background: `rgba(${Math.round(255 * c.v)},${Math.round(150 * c.v + 60)},${Math.round(255 * (1 - c.v))},0.9)`, outline: c.mutated ? `3px solid ${Y}` : 'none'}} />
          ))}
        </div>
      </Panel>
    </div>
  );
};

/* ================================================================== CHEAT */

export const Cheat: React.FC<P> = ({tl, t, data, t0}) => {
  const {aspect, tall, h} = useFmt();
  const over = data.v1tech.overs[1].deliveries; // bouncer, yorker, bouncer, yorker, bouncer, yorker
  const CYCLE = 2.05;
  const start = t0 + 0.2;
  const redAt = tl.word('c4', 'cheat') - 0.8;
  const slow = ramp(t, tl.at('c4') - 0.3, tl.at('c4') + 0.3);
  // after the over, keep replaying the last bouncer in slow motion
  const idx = Math.min(over.length - 1, Math.max(0, Math.floor((t - start) / CYCLE)));
  const local = t - start - idx * CYCLE;
  const overDone = t - start > over.length * CYCLE;
  const d = overDone ? over[4] : over[idx];
  const s = overDone ? -0.6 + ((t - start - over.length * CYCLE) % 5) * 0.35 : -0.75 + local;
  const ghosts = over.slice(0, overDone ? over.length : idx);
  const bouncers = over.slice(0, overDone ? over.length : idx + (s > flightEnd(d) * 0.5 ? 1 : 0)).filter((x) => x.category === 'bouncer').length;
  const red = ramp(t, redAt, redAt + 0.8);
  const pose = slow > 0 ? mix(cams.broadcast(aspect), {position: [1.4, 2.4, -6.5], target: [0.3, 1.5, 0], fov: 38}, slow) : drift(cams.broadcast(aspect), t, 0.05);
  const diagram = ramp(t, tl.at('c5') - 0.1, tl.at('c5') + 0.4);
  const showSpeed = s > 0 && s < 1.4;
  return (
    <AbsoluteFill>
      <AbsoluteFill style={{filter: diagram > 0 ? `brightness(${1 - diagram * 0.55}) blur(${diagram * 5}px)` : undefined}}>
        <Match d={d} s={s} pose={pose} ghosts={ghosts} look={{red: red * 0.85, bloom: 0.45 + red * 0.3}} />
      </AbsoluteFill>
      <GenBadge gen={1500} sub="first run" a={ramp(t, t0, t0 + 0.5) * (1 - diagram)} color={red > 0.5 ? RED : Y} />
      {showSpeed && !overDone ? (
        <div style={{position: 'absolute', right: 50, top: tall ? 170 : 70, textAlign: 'right', opacity: ramp(s, 0, 0.12) * (1 - ramp(s, 1.1, 1.4))}}>
          <Big size={120} color={d.category === 'bouncer' ? ORANGE : Y}>{Math.round(d.speed ?? 0)}</Big>
          <Label size={30} color="#fff" style={{textShadow: '0 2px 6px #000'}}>km/h · {d.category}</Label>
        </div>
      ) : null}
      {t > tl.at('c3') - 0.3 && t < tl.at('c4') ? (
        <div style={{position: 'absolute', left: 50, top: tall ? h * 0.36 : h * 0.33}}>
          <Panel a={ramp(t, tl.at('c3') - 0.3, tl.at('c3'))}>
            <Label size={24}>bouncers this over</Label>
            <div style={{display: 'flex', gap: 12, marginTop: 10}}>
              {[0, 1, 2].map((i) => (
                <div key={i} style={{width: 64, height: 64, borderRadius: 32, background: i < Math.max(bouncers, 3 * ramp(t, tl.at('c3'), tl.at('c3') + 0.6)) ? (i >= 2 ? RED : ORANGE) : 'rgba(255,255,255,0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: display, fontWeight: 900, fontSize: 34, color: '#111'}}>
                  {i + 1}
                </div>
              ))}
            </div>
            <Label size={20} color="rgba(255,255,255,0.6)" style={{marginTop: 10}}>the limit is 2</Label>
          </Panel>
        </div>
      ) : null}
      <Stamp t={t} at={tl.word('c3', 'legal')} until={tl.at('c4') - 0.2} text="ILLEGAL" />
      <Stamp t={t} at={tl.word('c4', 'cheat')} until={tl.at('c5') - 0.1} text="CHEAT" size={170} rot={-8} />
      <Flash t={t} at={tl.word('c4', 'cheat')} color="#ff2040" peak={0.5} />
      {diagram > 0 ? <SurpriseDiagram t={t} a={diagram} at={tl.at('c5')} tl={tl} /> : null}
    </AbsoluteFill>
  );
};

/** "Surprise" was just distance from what the batter expected, so extremes always win. */
const SurpriseDiagram: React.FC<{t: number; a: number; at: number; tl: TL}> = ({t, a, at, tl}) => {
  const {w, h, tall} = useFmt();
  const x0 = w * 0.1;
  const x1 = w * 0.9;
  const y = tall ? h * 0.42 : h * 0.4;
  const pulse = (k: number) => 1 + 0.15 * Math.sin((t - at) * 6 + k);
  const ex = ramp(t, tl.word('c5', 'extremes') - 0.2, tl.word('c5', 'extremes') + 0.2);
  return (
    <AbsoluteFill style={{opacity: a}}>
      <div style={{position: 'absolute', top: y - (tall ? 420 : 330), left: 0, right: 0, textAlign: 'center'}}>
        <Label size={30} color={Y}>how the batters judged surprise</Label>
      </div>
      <svg width={w} height={h} style={{position: 'absolute', inset: 0}}>
        <line x1={x0} y1={y} x2={x1} y2={y} stroke="#fff" strokeWidth={6} strokeLinecap="round" />
        {/* surprise curve: V shape, zero at expectation */}
        <path d={`M ${x0} ${y - 220} L ${(x0 + x1) / 2} ${y - 10} L ${x1} ${y - 220}`} stroke={RED} strokeWidth={8} fill="none" opacity={ramp(t, at + 0.3, at + 0.9)} />
        <circle cx={(x0 + x1) / 2} cy={y} r={18} fill={Y} />
        <circle cx={x0} cy={y - 220} r={30 * pulse(0) * ex} fill={RED} />
        <circle cx={x1} cy={y - 220} r={30 * pulse(1) * ex} fill={RED} />
      </svg>
      <div style={{position: 'absolute', top: y + 24, left: x0 - 70, width: 160, textAlign: 'center'}}><Label size={24} color="#fff">yorker</Label></div>
      <div style={{position: 'absolute', top: y + 24, left: (x0 + x1) / 2 - 110, width: 220, textAlign: 'center'}}><Label size={24} color={Y}>expected</Label></div>
      <div style={{position: 'absolute', top: y + 24, left: x1 - 90, width: 180, textAlign: 'center'}}><Label size={24} color="#fff">bouncer</Label></div>
      <div style={{position: 'absolute', top: y - 280, left: x0, opacity: ramp(t, at + 0.3, at + 0.9)}}>
        <Label size={24} color={RED}>high surprise</Label>
      </div>
    </AbsoluteFill>
  );
};

/* ================================================================== FIX */

export const Fix: React.FC<P> = ({tl, t, data, t0, t1}) => {
  const {aspect, tall, h} = useFmt();
  const againAt = tl.word('f2', 'again');
  const nothingAt = tl.word('f2', 'nothing');
  const rewind = ramp(t, againAt, nothingAt);
  const gen = Math.round(lerp(1500, 0, rewind * rewind));
  const rules = [
    {at: tl.word('f1', 'patterns') - 0.6, text: 'Batters read patterns'},
    {at: tl.word('f1', 'bouncers') - 0.3, text: 'Max 2 bouncers per over'},
  ];
  const s = 2.5 + (t - t0) * 0.2;
  return (
    <AbsoluteFill>
      <AbsoluteFill style={{filter: `blur(${4 + rewind * 10}px) brightness(${0.55 - rewind * 0.35}) saturate(${1 - rewind})`}}>
        <Match d={data.tech.overs[6].deliveries[0]} s={-1.2} pose={cams.orbit(2.4 + (t - t0) * 0.08 - rewind * 3, 40, 18, 42)} fielders trail={false} />
      </AbsoluteFill>
      {t < againAt + 0.2 ? (
        <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
          <Panel a={ramp(t, t0, t0 + 0.4) * (1 - ramp(t, againAt - 0.2, againAt + 0.2))} style={{padding: '30px 40px', minWidth: 700}}>
            <Label size={30} color={Y} style={{marginBottom: 18}}>rule change</Label>
            {rules.map((r, i) => {
              const a = ramp(t, r.at, r.at + 0.3);
              const chars = Math.round(r.text.length * ramp(t, r.at, r.at + 0.6));
              return (
                <div key={i} style={{display: 'flex', alignItems: 'center', gap: 18, opacity: a, marginBottom: 14}}>
                  <div style={{width: 46, height: 46, borderRadius: 23, background: GREEN, color: '#111', display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: display, fontWeight: 900, fontSize: 30}}>✓</div>
                  <div style={{fontFamily: display, fontWeight: 800, fontSize: 44, color: '#fff'}}>{r.text.slice(0, chars)}</div>
                </div>
              );
            })}
          </Panel>
        </AbsoluteFill>
      ) : null}
      {t > againAt - 0.2 ? (
        <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
          <div style={{textAlign: 'center', opacity: ramp(t, againAt - 0.2, againAt + 0.2), transform: `skewX(${Math.sin(t * 40) * 6 * (rewind > 0 && rewind < 1 ? 1 : 0)}deg)`}}>
            <Big size={200} color={gen === 0 ? '#fff' : Y}>GEN {gen}</Big>
            <Label size={30} color="#fff" style={{marginTop: 10}}>{gen === 0 ? 'from nothing' : 'rewinding'}</Label>
          </div>
        </AbsoluteFill>
      ) : null}
      <Flash t={t} at={nothingAt + 0.3} peak={0.5} />
      <Fade a={ramp(t, t1 - 0.25, t1) * 0.6} />
    </AbsoluteFill>
  );
};

/* ================================================================== RISE */

export const Rise: React.FC<P> = ({tl, t, data, t0}) => {
  const {aspect, tall} = useFmt();
  const steps = [
    {at: t0, gen: '0001', label: 1},
    {at: tl.at('r2') - 0.15, gen: '0050', label: 50},
    {at: tl.at('r3') - 0.15, gen: '0250', label: 250},
    {at: tl.at('r4') - 0.15, gen: '1500', label: 1500},
  ];
  const cur = [...steps].reverse().find((x) => t >= x.at)!;
  const lanes = useLanes(data, cur.gen, 5);
  const ds = data.swarms[cur.gen].deliveries;
  const rpo = runsPerOver(ds);
  const pct = stumpsPct(ds);
  const pose = topDown(lanes, aspect, 1 + (t - t0) * 0.012);
  const since = t - cur.at;
  const good = clamp01((12 - rpo) / 11.5);
  const colr = good > 0.85 ? GREEN : good > 0.5 ? Y : RED;
  return (
    <AbsoluteFill>
      <Nets lanes={lanes} t={since + 0.6} pose={pose} period={2.3} kitShift={cur.label * 7} />
      <GenBadge gen={cur.label.toLocaleString('en-US')} sub={`${pct.toFixed(0)}% of balls hit the stumps`} a={ramp(since, 0, 0.15)} />
      <div style={{position: 'absolute', right: 50, top: tall ? 160 : 70, textAlign: 'right', transform: `scale(${1 + 0.15 * (1 - ramp(since, 0, 0.25))})`, transformOrigin: 'right top'}}>
        <Big size={120} color={colr}>{rpo.toFixed(1)}</Big>
        <Label size={26} color="#fff" style={{textShadow: '0 2px 6px #000'}}>runs / over</Label>
      </div>
      {steps.slice(1).map((s, i) => (
        <Flash key={i} t={t} at={s.at} peak={0.3} dur={0.25} />
      ))}
    </AbsoluteFill>
  );
};
