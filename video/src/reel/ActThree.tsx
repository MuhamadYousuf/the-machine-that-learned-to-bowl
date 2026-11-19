import React from 'react';
import {AbsoluteFill} from 'remotion';
import {display, mono} from '../edit/fonts';
import {Delivery, flightEnd} from '../lib/data';
import {cams, Data, drift, Match, mix} from './Match';
import {TL, clamp01, lerp, ramp} from './timeline';
import {BLUE, Big, Fade, Flash, GREEN, Label, ORANGE, Panel, Stamp, Title, Y, useFmt} from './ui';
import {L, Pose} from './World';

type P = {tl: TL; t: number; data: Data; t0: number; t1: number};

/** Play deliveries back to back; returns the current one and its sim time. */
function sequence(ds: Delivery[], t: number, start: number, lead = 0.8, tail = 0.9) {
  let acc = start;
  for (let i = 0; i < ds.length; i++) {
    const dur = lead + flightEnd(ds[i]) + tail;
    if (t < acc + dur || i === ds.length - 1) return {i, d: ds[i], s: t - acc - lead, ghosts: ds.slice(0, i)};
    acc += dur;
  }
  return {i: 0, d: ds[0], s: 0, ghosts: []};
}

const StatCard: React.FC<{a: number; big: string; small: string; color?: string; top?: number}> = ({a, big, small, color = Y, top}) => {
  const {tall, h} = useFmt();
  return (
    <div style={{position: 'absolute', left: 50, top: top ?? (tall ? 170 : 70), opacity: a, transform: `translateX(${(1 - a) * -40}px)`}}>
      <Big size={110} color={color}>{big}</Big>
      <Label size={28} color="#fff" style={{textShadow: '0 2px 6px #000', marginTop: 6}}>{small}</Label>
    </div>
  );
};

const VsCard: React.FC<{name: string; a: number}> = ({name, a}) => {
  const {tall} = useFmt();
  return (
    <div style={{position: 'absolute', right: 50, top: tall ? 175 : 75, opacity: a, textAlign: 'right'}}>
      <Label size={24} color="rgba(255,255,255,0.8)" style={{textShadow: '0 2px 6px #000'}}>vs</Label>
      <Big size={58}>{name}</Big>
    </div>
  );
};

export const Plans: React.FC<P> = ({tl, t, data, t0}) => {
  const {aspect, tall, h, w} = useFmt();
  const p2 = tl.at('p2') - 0.2;
  const p3 = tl.at('p3') - 0.2;
  const p4 = tl.at('p4') - 0.2;
  const p5 = tl.at('p5') - 0.2;
  let body: React.ReactNode = null;
  let cards: React.ReactNode = null;
  if (t < p2) {
    // "It didn't find a trick. It found a plan." — sweeping aerial into the bowler at his mark
    const k = ramp(t, t0, p2);
    const pose = mix(cams.orbit(-0.6, 70, 34, 40), {position: [3, 3.5, -16], target: [0, 1.2, 4], fov: 46}, k * k);
    body = <Match d={data.slog.overs[17].deliveries[5]} s={-1.35 + Math.max(0, t - (p2 - 0.4))} pose={pose} />;
    cards = (
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        <div style={{opacity: ramp(t, tl.word('p1', 'plan', 0) - 0.1, tl.word('p1', 'plan', 0) + 0.3) * (1 - ramp(t, p2 - 0.4, p2)), textAlign: 'center'}}>
          <Label size={30} color="#fff" style={{textShadow: '0 2px 8px #000'}}>not a trick</Label>
          <Big size={150} color={Y}>A PLAN</Big>
        </div>
      </AbsoluteFill>
    );
  } else if (t < p3) {
    // Slogger: full and straight, bowled.
    const d = data.slog.overs[17].deliveries[5];
    const relAt = p2 + 1.0;
    const s = t - relAt;
    const pose = s < 0.05 ? drift(cams.chase(s, aspect), t, 0.03) : mix(cams.stump(aspect), {...cams.stump(aspect), fov: cams.stump(aspect).fov - 8}, ramp(s, 0, 2));
    const slowS = s < flightEnd(d) ? s : flightEnd(d) + (s - flightEnd(d)) * 0.25;
    body = <Match d={d} s={slowS} pose={pose} />;
    const hitT = relAt + flightEnd(d);
    cards = (
      <>
        <VsCard name="THE SLOGGER" a={ramp(t, p2, p2 + 0.3)} />
        <Flash t={t} at={hitT} />
        <Stamp t={t} at={hitT + 0.1} until={p3 - 0.3} text="BOWLED" color={Y} y={0.26} />
        <StatCard a={ramp(t, hitT + 0.6, hitT + 0.9) * (1 - ramp(t, p3 - 0.3, p3))} big="34%" small="yorkers · 51% full" top={tall ? h * 0.5 : h * 0.45} />
      </>
    );
  } else if (t < p4) {
    // Technician: good length, again and again, until he edges one.
    const over = data.tech.overs[6].deliveries;
    const ds = [over[0], over[2], over[3], over[4]];
    const seq = sequence(ds, t, p3, 0.55, 0.55);
    const last = seq.i === ds.length - 1;
    const pose = last && seq.s > 0 ? mix(cams.broadcast(aspect), cams.keeper(aspect), ramp(seq.s, 0.1, 0.5)) : drift(cams.broadcast(aspect), t, 0.04);
    const slowS = last && seq.s > (seq.d.tc ?? 0.5) - 0.05 ? (seq.d.tc ?? 0.5) - 0.05 + (seq.s - (seq.d.tc ?? 0.5) + 0.05) * 0.3 : seq.s;
    body = <Match d={seq.d} s={slowS} pose={pose} ghosts={seq.ghosts} />;
    const catchAt = p3 + ds.slice(0, 3).reduce((a, d) => a + 0.55 + flightEnd(d) + 0.55, 0) + 0.55 + (ds[3].tc ?? 0.5) - 0.05 + 0.45;
    cards = (
      <>
        <VsCard name="THE TECHNICIAN" a={ramp(t, p3, p3 + 0.3)} />
        <div style={{position: 'absolute', left: 50, top: tall ? 170 : 70, opacity: ramp(t, p3 + 0.5, p3 + 0.8)}}>
          <Label size={26} color="#fff" style={{textShadow: '0 2px 6px #000'}}>good length · outside off</Label>
          <div style={{display: 'flex', gap: 10, marginTop: 10}}>
            {ds.map((_, i) => (
              <div key={i} style={{width: 46, height: 46, borderRadius: 23, background: i < seq.i || (i === seq.i && seq.s > 0.5) ? (i === 3 ? GREEN : 'rgba(255,255,255,0.85)') : 'rgba(255,255,255,0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: display, fontWeight: 900, fontSize: 20, color: '#111'}}>
                {i === 3 && seq.s > 0.5 ? 'W' : '•'}
              </div>
            ))}
          </div>
        </div>
        <Stamp t={t} at={catchAt} until={p4 - 0.3} text="CAUGHT BEHIND" color={GREEN} size={96} y={0.26} />
        <StatCard a={ramp(t, catchAt + 0.6, catchAt + 0.9) * (1 - ramp(t, p4 - 0.3, p4))} big="73%" small="good length" top={tall ? h * 0.5 : h * 0.45} />
      </>
    );
  } else if (t < p5) {
    // Tailender: full and straight, zero bouncers.
    const d = data.tail.overs[9].deliveries[1];
    const relAt = p4 + 0.9;
    const s = t - relAt;
    const pose = drift(mix({position: [7, 1.3, L - 9], target: [0, 0.9, L - 3], fov: 40}, {position: [5.5, 1.0, L - 3.5], target: [-0.2, 0.6, L - 0.5], fov: 34}, ramp(s, -0.2, flightEnd(d) + 0.3)), t, 0.02);
    const slowS = s < flightEnd(d) ? s : flightEnd(d) + (s - flightEnd(d)) * 0.3;
    body = <Match d={d} s={slowS} pose={pose} />;
    const hitT = relAt + flightEnd(d);
    const zeroAt = tl.word('p4', 'zero');
    const twoAt = tl.word('p4', 'two');
    cards = (
      <>
        <VsCard name="THE TAILENDER" a={ramp(t, p4, p4 + 0.3)} />
        <Flash t={t} at={hitT} />
        <div style={{position: 'absolute', left: 50, top: tall ? h * 0.36 : h * 0.34, display: 'flex', gap: 24}}>
          <Panel a={ramp(t, zeroAt - 0.1, zeroAt + 0.2)}>
            <Label size={22}>its bouncers</Label>
            <Big size={110} color={GREEN}>0</Big>
          </Panel>
          <Panel a={ramp(t, twoAt - 0.1, twoAt + 0.2)}>
            <Label size={22}>my plan</Label>
            <Big size={110} color={ORANGE}>2</Big>
          </Panel>
        </div>
      </>
    );
  } else {
    // Scoreboard: against my own hand-written plans.
    const k = ramp(t, p5, p5 + 0.3);
    body = (
      <AbsoluteFill style={{filter: 'blur(8px) brightness(0.4)'}}>
        <Match d={data.slog.overs[17].deliveries[5]} s={-1.5} pose={cams.orbit(0.6 + (t - p5) * 0.05, 45, 20, 40)} fielders />
      </AbsoluteFill>
    );
    const rows = [
      {name: 'Slogger', v: 1.25},
      {name: 'Technician', v: 1.16},
      {name: 'Tailender', v: 1.18},
    ];
    const barW = w * 0.62;
    const scale = barW / 1.35;
    const beatAt = tl.word('p5', 'beat');
    cards = (
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', opacity: k}}>
        <div style={{width: w * 0.86}}>
          <Label size={28} color={Y} style={{marginBottom: 26}}>score vs my hand-written plans</Label>
          <div style={{position: 'relative'}}>
            <div style={{position: 'absolute', left: 230 + scale * 1.0, top: -16, bottom: -16, width: 4, background: '#fff', opacity: 0.8}} />
            <div style={{position: 'absolute', left: 230 + scale * 1.0 + 10, top: -50, fontFamily: mono, fontWeight: 700, fontSize: 22, color: '#fff'}}>MY PLAN = 1.00</div>
            {rows.map((r, i) => {
              const g = ramp(t, p5 + 0.3 + i * 0.25, beatAt + i * 0.15);
              const v = r.v * g;
              return (
                <div key={i} style={{display: 'flex', alignItems: 'center', marginBottom: 26}}>
                  <div style={{width: 230, fontFamily: display, fontWeight: 800, fontSize: 36, color: '#fff'}}>{r.name}</div>
                  <div style={{height: 54, width: scale * v, background: v > 1 ? GREEN : BLUE, borderRadius: 10, boxShadow: v > 1 ? `0 0 24px ${GREEN}` : 'none'}} />
                  <div style={{marginLeft: 16, fontFamily: mono, fontWeight: 700, fontSize: 36, color: v > 1 ? GREEN : '#fff'}}>{v.toFixed(2)}</div>
                </div>
              );
            })}
          </div>
        </div>
        <Stamp t={t} at={beatAt + 0.5} text="BEAT ALL THREE" color={GREEN} size={92} y={tall ? 0.7 : 0.78} rot={-4} />
      </AbsoluteFill>
    );
  }
  return (
    <AbsoluteFill>
      {body}
      {cards}
    </AbsoluteFill>
  );
};

/* ================================================================== OUTRO */

export const Outro: React.FC<P> = ({tl, t, data, t0, t1}) => {
  const {tall, h} = useFmt();
  const s = t - t0;
  const pose = cams.orbit(-1.2 + s * 0.07, 78 - s * 1.4, 42 - s * 0.9, 42);
  const stats = [
    {at: tl.word('o1', 'random'), big: '192', small: 'random brains'},
    {at: tl.word('o1', 'generations'), big: '1,500', small: 'generations'},
    {at: tl.word('o1', 'deliveries'), big: '83,000,000', small: 'deliveries'},
  ];
  const learnedAt = tl.word('o3', 'learned');
  const black = ramp(t, tl.end('o3') + 0.25, tl.end('o3') + 0.9);
  const title = ramp(t, tl.end('o3') + 0.7, tl.end('o3') + 1.4);
  const statsOut = 1 - ramp(t, tl.at('o2') - 0.2, tl.at('o2') + 0.3);
  return (
    <AbsoluteFill>
      <Match d={data.slog.overs[17].deliveries[5]} s={-1.6 + (s % 3.2) * 0} pose={pose} look={{warm: 0.9, sun: 0.95, bloom: 0.55}} />
      <div style={{position: 'absolute', left: 0, right: 0, top: tall ? h * 0.2 : h * 0.12, textAlign: 'center'}}>
        {stats.map((x, i) => {
          const a = ramp(t, x.at - 0.15, x.at + 0.2) * statsOut;
          return (
            <div key={i} style={{opacity: a, transform: `translateY(${(1 - a) * 30}px)`, marginBottom: 26}}>
              <Big size={i === 2 ? 104 : 120} color={Y}>{x.big}</Big>
              <Label size={28} color="#fff" style={{textShadow: '0 2px 8px #000'}}>{x.small}</Label>
            </div>
          );
        })}
      </div>
      <Flash t={t} at={learnedAt} peak={0.35} dur={0.6} />
      <Fade a={black} />
      <Title a={title} sub="built from scratch · neuroevolution" />
      <Fade a={ramp(t, t1 - 0.6, t1)} />
    </AbsoluteFill>
  );
};
