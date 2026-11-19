import React, {useMemo} from 'react';
import {AbsoluteFill} from 'remotion';
import {display, mono} from '../edit/fonts';
import {Delivery, flightEnd, positionAt} from '../lib/data';
import {bowlerPose} from './Mannequin';
import {cams, Data, drift, Match, mix} from './Match';
import {project} from './Nets';
import {TL, clamp01, lerp, ramp} from './timeline';
import {BLUE, Big, Fade, Flash, GREEN, Label, Meter, ORANGE, Panel, Stamp, Title, Y, useFmt} from './ui';
import {L, rng} from './World';

type P = {tl: TL; t: number; data: Data};

const cat = (d: Delivery) => (d.category ?? '').replace('_', ' ').toUpperCase();

/* ================================================================== HOOK */

export const Hook: React.FC<P> = ({tl, t, data}) => {
  const {aspect} = useFmt();
  const d = data.slog.overs[9].deliveries[2]; // 117 km/h yorker, bowled
  const ts = flightEnd(d);
  const hitAt = tl.at('h3') - 0.15;
  const r1 = 0.3;
  const r2 = 0.1;
  const s = t < hitAt ? ts + (t - hitAt) * r1 : ts + (t - hitAt) * r2;
  const release = hitAt - ts / r1;
  const titleAt = tl.end('h3') + 0.2;
  let pose;
  if (t < release - 0.1) pose = drift(cams.chase(s, aspect), t, 0.05);
  else pose = drift(mix(cams.stump(aspect), {...cams.stump(aspect), position: [2.5, 0.45, L - 1.6], fov: cams.stump(aspect).fov - 4}, ramp(t, release, hitAt + 1)), t, 0.02);
  const dimTitle = ramp(t, titleAt - 0.2, titleAt + 0.6);
  return (
    <AbsoluteFill>
      <AbsoluteFill style={{filter: `brightness(${1 - dimTitle * 0.55}) blur(${dimTitle * 6}px)`}}>
        <Match d={d} s={s} pose={pose} look={{warm: 0.35, bloom: 0.5}} />
      </AbsoluteFill>
      <Flash t={t} at={hitAt} peak={0.8} dur={0.5} />
      <Stamp t={t} at={hitAt + 0.25} until={titleAt - 0.2} text="YORKER" color={Y} y={0.3} size={130} />
      {t > hitAt + 0.5 && t < titleAt ? (
        <AbsoluteFill style={{alignItems: 'center'}}>
          <Label size={36} color="#fff" style={{position: 'absolute', top: '39%', opacity: ramp(t, hitAt + 0.5, hitAt + 0.8), textShadow: '0 3px 8px #000'}}>
            {Math.round(d.speed ?? 0)} km/h · stumps
          </Label>
        </AbsoluteFill>
      ) : null}
      <SlowMo a={ramp(t, 0.8, 1.2) * (1 - dimTitle)} />
      <Title a={dimTitle} />
      <Fade a={1 - ramp(t, 0, 0.9)} />
    </AbsoluteFill>
  );
};

/* ================================================================== SEES */

export const Sees: React.FC<P & {t0: number}> = ({tl, t, data, t0}) => {
  const {aspect, w, h, tall} = useFmt();
  const over = data.slog.overs[11].deliveries;
  const d = over[4];
  const last = over[3];
  const before = over[2];
  const runsSoFar = over.slice(0, 4).reduce((a, x) => a + (x.runs ?? 0), 0);
  const relAt = tl.word('s5', 'speed');
  // Slow, "machine time" run-up until release, then the ball at half speed.
  const rate0 = 1.6 / Math.max(1, relAt - t0);
  const s = t < relAt ? -1.6 + (t - t0) * rate0 : (t - relAt) * 0.5;
  const ts = flightEnd(d);
  const hitT = relAt + ts / 0.5;
  // Camera: from behind the bowler, craning up to a high three-quarter view that shows the batter.
  const k = ramp(t, t0, tl.at('s3') + 1.0);
  const high = {position: [-7.5, 7.5, -9] as [number, number, number], target: [0, 0.8, 9] as [number, number, number], fov: tall ? 48 : 42};
  let pose = mix(cams.chase(Math.min(s, -0.3), aspect), high, k);
  pose = mix(pose, cams.chase(-0.1, aspect), ramp(t, tl.at('s4') - 0.2, relAt - 0.2));
  if (t > relAt - 0.2) pose = mix(cams.chase(-0.1, aspect), cams.stump(aspect), ramp(t, relAt, relAt + 0.35));
  const head: [number, number, number] = (() => {
    const r = bowlerPose(Math.min(s, 0)).root;
    return [r[0], r[1] + 0.75, r[2]];
  })();
  const rays: {at: number; to: [number, number, number]; label: string; value: string}[] = [
    {at: tl.word('s2', 'numbers'), to: [0, 0.1, L * 0.55], label: 'over', value: `ball ${5} · ${runsSoFar} runs`},
    {at: tl.word('s3', 'batting'), to: [-0.42, 1.0, L - 1.35], label: 'batter', value: 'SLOGGER'},
    {at: tl.word('s3', 'confident'), to: [-0.42, 1.75, L - 1.35], label: 'confidence', value: '21%'},
    {at: tl.word('s3', 'last'), to: positionAt(last, flightEnd(last) * 0.92), label: 'last ball', value: `${cat(last)} ${Math.round(last.speed ?? 0)}`},
    {at: tl.word('s3', 'before'), to: positionAt(before, flightEnd(before) * 0.92), label: 'ball before', value: `${cat(before)} ${Math.round(before.speed ?? 0)}`},
  ];
  const raysOn = t < tl.at('s4') + 0.4;
  const raysA = 1 - ramp(t, tl.at('s4') - 0.1, tl.at('s4') + 0.4);
  const inputsAt = tl.at('s4');
  const outAt = tl.at('s5');
  const outs = [
    {at: tl.word('s5', 'speed'), label: 'SPEED', value: `${Math.round(d.speed ?? 0)} km/h`, v: ((d.speed ?? 130) - 110) / 45},
    {at: tl.word('s5', 'length'), label: 'LENGTH', value: `${(d.length ?? 0).toFixed(1)} m · ${cat(d)}`, v: 1 - (d.length ?? 0) / 12, color: GREEN},
    {at: tl.word('s5', 'line'), label: 'LINE', value: 'stumps', v: -(d.line ?? 0) / 0.6, centred: true, color: BLUE},
    {at: tl.word('s5', 'swing'), label: 'SWING', value: 'away', v: (d.swing ?? -1) * 1, centred: true, color: ORANGE},
  ];
  const nums: [string, string][] = [
    ['is slogger', '1'], ['is technician', '0'], ['is tailender', '0'], ['confidence', '0.21'], ['balls faced', '70'], ['balls this over', '4'], ['runs this over', `${runsSoFar}`],
    ['last length', `${(last.length ?? 0).toFixed(1)}`], ['last line', `${(last.line ?? 0).toFixed(2)}`], ['last swing', `${(last.swing ?? 0).toFixed(1)}`], ['last speed', `${Math.round(last.speed ?? 0)}`], ['last valid', '1'],
    ['before length', `${(before.length ?? 0).toFixed(1)}`], ['before line', `${(before.line ?? 0).toFixed(2)}`], ['before swing', `${(before.swing ?? 0).toFixed(1)}`], ['before speed', `${Math.round(before.speed ?? 0)}`], ['before valid', '1'], ['bouncers used', '0'],
  ];
  return (
    <AbsoluteFill>
      <Match d={d} s={s} pose={pose} ghosts={raysOn && t > rays[3].at ? [last, ...(t > rays[4].at ? [before] : [])] : []} look={{bloom: 0.4}} />
      {raysOn ? (
        <svg width={w} height={h} style={{position: 'absolute', inset: 0, opacity: raysA}}>
          {rays.map((r, i) => {
            const a = ramp(t, r.at - 0.05, r.at + 0.3);
            if (a <= 0) return null;
            const [x0, y0] = project(pose, w, h, head);
            const [x1, y1] = project(pose, w, h, r.to);
            const xe = lerp(x0, x1, a);
            const ye = lerp(y0, y1, a);
            return (
              <g key={i}>
                <line x1={x0} y1={y0} x2={xe} y2={ye} stroke={Y} strokeWidth={4} strokeDasharray="14 8" opacity={0.95} />
                <circle cx={xe} cy={ye} r={9} fill={Y} />
              </g>
            );
          })}
        </svg>
      ) : null}
      {raysOn
        ? rays.map((r, i) => {
            const a = ramp(t, r.at + 0.15, r.at + 0.45) * raysA;
            if (a <= 0) return null;
            const [x1, y1] = project(pose, w, h, r.to);
            return (
              <div key={i} style={{position: 'absolute', left: Math.min(w - 260, Math.max(20, x1 + 18)), top: y1 - 70, opacity: a, transform: `scale(${0.9 + 0.1 * a})`}}>
                <div style={{background: 'rgba(12,16,24,0.85)', borderRadius: 12, padding: '6px 14px', border: `2px solid ${Y}`}}>
                  <Label size={20} color={Y}>{r.label}</Label>
                  <div style={{fontFamily: mono, fontWeight: 700, fontSize: 30, color: '#fff'}}>{r.value}</div>
                </div>
              </div>
            );
          })
        : null}
      {t > inputsAt - 0.1 && t < hitT + 1.2 ? (
        <div style={{position: 'absolute', left: 40, top: tall ? 140 : 50, display: 'flex', gap: 22, alignItems: 'flex-start'}}>
          <Panel a={ramp(t, inputsAt - 0.1, inputsAt + 0.3)} style={{padding: '14px 18px'}}>
            <Label size={20} color={Y} style={{marginBottom: 6}}>18 inputs</Label>
            {nums.map(([k, v], i) => {
              const a = ramp(t, inputsAt + i * 0.05, inputsAt + i * 0.05 + 0.2);
              return (
                <div key={i} style={{display: 'flex', justifyContent: 'space-between', gap: 18, opacity: a, fontFamily: mono, fontSize: tall ? 22 : 17, lineHeight: tall ? 1.45 : 1.3}}>
                  <span style={{color: 'rgba(255,255,255,0.6)'}}>{k}</span>
                  <span style={{color: '#fff', fontWeight: 700}}>{v}</span>
                </div>
              );
            })}
          </Panel>
          <div style={{marginTop: tall ? 260 : 170, opacity: ramp(t, outAt - 0.3, outAt)}}>
            <div style={{width: 110, height: 110, borderRadius: 24, background: Y, display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: 'serif', fontStyle: 'italic', fontWeight: 700, fontSize: 70, color: '#111', boxShadow: `0 0 40px ${Y}`}}>
              f
            </div>
          </div>
        </div>
      ) : null}
      {t > outAt - 0.1 && t < hitT + 1.2 ? (
        <div style={{position: 'absolute', left: 40, top: tall ? h * 0.56 : h * 0.6}}>
          <Panel a={ramp(t, outAt - 0.1, outAt + 0.3)} style={{padding: '18px 22px'}}>
            <Label size={20} color={Y} style={{marginBottom: 10}}>4 outputs</Label>
            {outs.map((o, i) => (
              <Meter key={i} label={o.label} value={o.value} v={o.v * ramp(t, o.at - 0.05, o.at + 0.3)} centred={o.centred} color={o.color ?? Y} a={ramp(t, o.at - 0.1, o.at + 0.15) * 0.85 + 0.15} />
            ))}
          </Panel>
        </div>
      ) : null}
      <SlowMo a={(t < relAt ? 1 : 0) * ramp(t, t0 + 0.3, t0 + 0.6)} text="machine time" />
      <Flash t={t} at={hitT} peak={0.5} />
    </AbsoluteFill>
  );
};

const SlowMo: React.FC<{a: number; text?: string}> = ({a, text = 'slow motion'}) => {
  const {tall} = useFmt();
  if (a <= 0) return null;
  return (
    <div style={{position: 'absolute', right: 44, top: tall ? 120 : 40, opacity: a * 0.9}}>
      <Label size={22} color="#fff" style={{textShadow: '0 2px 6px #000'}}>{text}</Label>
    </div>
  );
};

/* ================================================================== BRAIN */

export const Brain: React.FC<P & {t0: number}> = ({tl, t, data, t0}) => {
  const {w, h, aspect, tall} = useFmt();
  const net = data.net;
  const sizes = net.sizes; // 18, 24, 24, 4
  const W = net.champion.w; // flattened [in*out]
  const random = net.frames[0].w;
  const changeAt = tl.word('b2', 'change');
  // weight morph: champion -> random generation-1 brain, bouncing a few times while "change the weights" plays
  const morphs = [changeAt, changeAt + 0.6, changeAt + 1.2].map((x) => ramp(t, x, x + 0.35));
  const morph = (morphs[0] - morphs[1] + morphs[2]) * (1 - ramp(t, tl.at('b3'), tl.at('b3') + 0.6));
  const top = tall ? 400 : 230;
  const bottom = tall ? h * 0.66 : h * 0.72;
  const rowY = (i: number) => lerp(bottom, top, i / (sizes.length - 1));
  const nodeX = (i: number, n: number) => w * 0.08 + ((w * 0.84) * (i + 0.5)) / n;
  const build = (layer: number) => {
    const a = tl.at('b1');
    const e = tl.word('b1', 'weights');
    return ramp(t, lerp(a, e, layer / 4), lerp(a, e, (layer + 1) / 4));
  };
  const qmark = ramp(t, tl.at('b4') - 0.1, tl.at('b4') + 0.5);
  const netA = (1 - qmark * 0.85) * ramp(t, t0, t0 + 0.4);
  const r = rng(9);
  const s = t - t0;
  const pose = mix(cams.orbit(0.9 + s * 0.04, 16, 4, 40), cams.orbit(1.2 + s * 0.04, 14, 3.5, 40), 0.5);
  pose.target = [0, 1.2, 0];
  return (
    <AbsoluteFill>
      <AbsoluteFill style={{filter: 'blur(7px) brightness(0.45) saturate(0.8)'}}>
        <Match d={data.slog.overs[11].deliveries[4]} s={0.9} pose={{...pose, position: [pose.position[0], pose.position[1], pose.position[2] - 2]}} fielders={false} trail={false} />
      </AbsoluteFill>
      <svg width={w} height={h} style={{position: 'absolute', inset: 0, opacity: netA}}>
        {[0, 1, 2].map((li) => {
          const nIn = sizes[li];
          const nOut = sizes[li + 1];
          const a = build(li + 1);
          if (a <= 0) return null;
          const lines: React.ReactNode[] = [];
          for (let i = 0; i < nIn; i++) {
            for (let j = 0; j < nOut; j++) {
              const idx = i * nOut + j;
              const wv = lerp(W[li][idx] ?? 0, random[li][idx] ?? 0, morph);
              const op = Math.min(0.9, Math.abs(wv) * 0.9) * a;
              if (op < 0.05) continue;
              const prog = clamp01(a * 1.3 - ((i + j) / (nIn + nOut)) * 0.3);
              const x1 = nodeX(i, nIn);
              const y1 = rowY(li);
              const x2 = lerp(x1, nodeX(j, nOut), prog);
              const y2 = lerp(y1, rowY(li + 1), prog);
              lines.push(<line key={idx} x1={x1} y1={y1} x2={x2} y2={y2} stroke={wv > 0 ? ORANGE : BLUE} strokeWidth={1.4} opacity={op} />);
            }
          }
          return <g key={li}>{lines}</g>;
        })}
        {sizes.map((n, li) => {
          const a = build(li);
          return Array.from({length: n}, (_, i) => (
            <circle key={`${li}-${i}`} cx={nodeX(i, n)} cy={rowY(li)} r={(li === 3 ? 22 : 11) * clamp01(a * 1.5 - i / n / 2)} fill={li === 0 ? '#fff' : li === 3 ? Y : '#2b3445'} stroke="#fff" strokeWidth={2} />
          ));
        })}
      </svg>
      {/* labels */}
      <div style={{position: 'absolute', left: 0, right: 0, top: rowY(3) - 85, display: 'flex', justifyContent: 'space-around', padding: `0 ${w * 0.08}px`, opacity: build(3) * netA}}>
        {['SPEED', 'LENGTH', 'LINE', 'SWING'].map((x) => (
          <Label key={x} size={24} color={Y} style={{textShadow: '0 2px 6px #000'}}>{x}</Label>
        ))}
      </div>
      <Label size={22} style={{position: 'absolute', left: 0, right: 0, textAlign: 'center', top: rowY(0) + 26, opacity: build(0) * netA}}>18 numbers in</Label>
      {/* equation */}
      <div style={{position: 'absolute', left: 0, right: 0, top: tall ? 150 : 40, textAlign: 'center', opacity: ramp(t, tl.word('b1', 'weights') - 0.6, tl.word('b1', 'weights')) * (1 - qmark)}}>
        <div style={{display: 'inline-block', background: 'rgba(10,12,18,0.8)', borderRadius: 16, padding: '10px 22px'}}>
          <span style={{fontFamily: mono, fontWeight: 700, fontSize: tall ? 30 : 26, color: Y}}>{Math.round(1156 * ramp(t, tl.word('b1', 'thousand') - 0.2, tl.word('b1', 'weights'))).toLocaleString('en-US')} weights</span>
          <span style={{fontFamily: mono, fontSize: tall ? 22 : 19, color: 'rgba(255,255,255,0.75)'}}>{'  '}= 18×24+24 + 24×24+24 + 24×4+4</span>
        </div>
      </div>
      {/* question marks */}
      {t > tl.at('b3') - 0.2
        ? Array.from({length: 26}, (_, i) => {
            const x = r() * w;
            const y = top + r() * (bottom - top);
            const a = ramp(t, tl.at('b3') + i * 0.04, tl.at('b3') + i * 0.04 + 0.3) * (1 - qmark);
            return (
              <div key={i} style={{position: 'absolute', left: x, top: y, fontFamily: display, fontWeight: 900, fontSize: 40 + r() * 30, color: Y, opacity: a, transform: `translate(-50%,-50%) rotate(${(r() - 0.5) * 30}deg)`}}>
                ?
              </div>
            );
          })
        : null}
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', opacity: qmark}}>
        <Big size={420} color={Y} style={{transform: `scale(${0.8 + 0.2 * qmark}) rotate(${(1 - qmark) * 10}deg)`}}>?</Big>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
