import React, {useMemo} from 'react';
import * as THREE from 'three';
import {AbsoluteFill} from 'remotion';
import {display, mono} from '../edit/fonts';
import {Delivery, flightEnd, positionAt} from '../lib/data';
import {Batter, Bowler} from './Mannequin';
import {GREEN, ORANGE, Y, useFmt} from './ui';
import {L, Pose, Stage, Stumps, col, rng} from './World';
import {clamp01, lerp, ramp} from './timeline';

export const LANE_W = 4.0;
export const ROW_DZ = 35;
const Z_MIN = -9;
const Z_MAX = 24;
export const PERIOD = 2.7;
const RELEASE_IN_CYCLE = 1.3;

export type Lane = {genome: number; fitness: number; deliveries: Delivery[]; x: number; z: number; hue: number};

/** Pick `n` brains from a generation: the best `best` plus an even spread of the rest, shuffled into lanes. */
export function pickLanes(ds: Delivery[], n: number, cols: number, best = 3, seed = 1): Lane[] {
  const by = new Map<number, Delivery[]>();
  ds.forEach((d) => {
    const g = (d as unknown as {genome: number}).genome;
    if (!by.has(g)) by.set(g, []);
    by.get(g)!.push(d);
  });
  const genomes = [...by.entries()].map(([g, list]) => ({g, list, f: (list[0] as unknown as {fitness: number}).fitness}));
  genomes.sort((a, b) => b.f - a.f);
  const chosen = genomes.slice(0, best);
  const rest = genomes.slice(best);
  const r = rng(seed);
  for (let i = 0; chosen.length < n && i < rest.length; i++) {
    const j = Math.floor((i * rest.length) / (n - best));
    if (rest[j] && !chosen.includes(rest[j])) chosen.push(rest[j]);
  }
  // shuffle lane order deterministically
  const order = chosen.map((c) => ({c, k: r()})).sort((a, b) => a.k - b.k).map((o) => o.c);
  const rows = Math.ceil(n / cols);
  return order.map((c, i) => {
    const row = Math.floor(i / cols);
    const colI = i % cols;
    return {
      genome: c.g,
      fitness: c.f,
      deliveries: c.list,
      x: (colI - (cols - 1) / 2) * LANE_W,
      z: (row - (rows - 1) / 2) * ROW_DZ,
      hue: (c.g * 47) % 360,
    };
  });
}

export type LaneState = {u: number; d: Delivery; k: number; cycleStart: number};

export function laneState(lane: Lane, idx: number, t: number, period = PERIOD): LaneState {
  const off = ((idx * 0.618) % 1) * 1.1;
  const tau = Math.max(0, t + 10 * period - off);
  const k = Math.floor(tau / period);
  const d = lane.deliveries[k % lane.deliveries.length];
  const u = tau - k * period - RELEASE_IN_CYCLE;
  return {u, d, k, cycleStart: t - (tau - k * period)};
}

const isWicket = (d: Delivery) => ['bowled', 'lbw', 'caught_behind', 'caught_field'].includes(d.result);

/** World position of the ball in a lane (lane-local), including a struck ball flying back into the net. */
function ballPos(d: Delivery, u: number): [number, number, number] | null {
  if (u < 0) return null;
  const end = flightEnd(d);
  if (u <= end) return positionAt(d, u);
  const after = u - end;
  const p = positionAt(d, end);
  if (d.runs > 0 && !isWicket(d)) {
    const k = clamp01(after / 0.75);
    const ang = (((d.shotAngle ?? 0) * Math.PI) / 180) * 0.5;
    const tz = Z_MIN + 2;
    const tx = Math.max(-1.6, Math.min(1.6, Math.sin(ang) * 6));
    return [lerp(p[0], tx, k), Math.max(0.05, p[1] + (d.runs >= 6 ? 7 : 2.2) * 4 * k * (1 - k)), lerp(p[2], tz, k)];
  }
  if (d.result === 'caught_behind') {
    const k = clamp01(after / 0.15);
    return [lerp(p[0], 0.25, k), lerp(p[1], 0.9, k), lerp(p[2], L + 1.7, k)];
  }
  if (after > 0.6) return null;
  return [p[0], Math.max(0.05, p[1] * (1 - after * 2)), p[2] + after * 1.2];
}

const Trail: React.FC<{d: Delivery; u: number; color: string; radius: number}> = ({d, u, color, radius}) => {
  const geo = useMemo(() => {
    const pts: THREE.Vector3[] = [];
    for (let i = 10; i >= 0; i--) {
      const p = ballPos(d, u - i * 0.018);
      if (p) pts.push(new THREE.Vector3(...p));
    }
    const clean = pts.filter((v, i) => i === 0 || v.distanceToSquared(pts[i - 1]) > 1e-6);
    if (clean.length < 2) return null;
    return new THREE.TubeGeometry(new THREE.CatmullRomCurve3(clean), 16, radius, 6, false);
  }, [d, u, radius]);
  if (!geo) return null;
  return (
    <mesh geometry={geo}>
      <meshBasicMaterial color={color} transparent opacity={0.85} toneMapped={false} />
    </mesh>
  );
};

const NetWalls: React.FC = () => {
  const len = Z_MAX - Z_MIN;
  const zc = (Z_MAX + Z_MIN) / 2;
  return (
    <group>
      {[-1, 1].map((s) => (
        <group key={s}>
          <mesh position={[s * (LANE_W / 2 - 0.1), 1.6, zc]}>
            <boxGeometry args={[0.04, 3.2, len]} />
            <meshStandardMaterial color="#1b2a22" transparent opacity={0.28} />
          </mesh>
          <mesh position={[s * (LANE_W / 2 - 0.1), 3.2, zc]}>
            <boxGeometry args={[0.09, 0.09, len]} />
            <meshStandardMaterial color="#e8eef2" />
          </mesh>
        </group>
      ))}
      <mesh position={[0, 3.2, Z_MAX]}>
        <boxGeometry args={[LANE_W - 0.2, 0.09, 0.09]} />
        <meshStandardMaterial color="#e8eef2" />
      </mesh>
    </group>
  );
};

export type LaneLook = {dim?: number; glow?: number; glowColor?: string; box?: number};

const LaneView: React.FC<{lane: Lane; idx: number; t: number; look: LaneLook; ballScale: number; period: number; kitShift?: number}> = ({lane, idx, t, look, ballScale, period, kitShift = 0}) => {
  const {u, d} = laneState(lane, idx, t, period);
  const pos = ballPos(d, u);
  const end = flightEnd(d);
  const after = u - end;
  const hit = d.result === 'bowled' || d.result === 'lbw' ? clamp01(after / 0.5) : 0;
  const flash = after > 0 && after < 0.9 ? 1 - after / 0.9 : 0;
  const res = isWicket(d) ? GREEN : d.runs >= 4 ? ORANGE : d.runs > 0 ? '#ffe08a' : '#ffffff';
  const tint = flash * (isWicket(d) ? 0.55 : d.runs >= 4 ? 0.45 : d.runs > 0 ? 0.15 : 0);
  const shirt = new THREE.Color().setHSL(((lane.hue + kitShift) % 360) / 360, 0.65, 0.5).getStyle();
  return (
    <group position={[lane.x, 0, lane.z]}>
      <mesh position={[0, 0.003, (Z_MIN + Z_MAX) / 2]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[LANE_W - 0.25, Z_MAX - Z_MIN]} />
        <meshStandardMaterial color="#3f8f3c" roughness={1} />
      </mesh>
      <mesh position={[0, 0.035, (-2.44 + L) / 2]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[2.4, L + 2.44 + 2]} />
        <meshStandardMaterial color={col.pitch} roughness={1} />
      </mesh>
      {tint > 0 ? (
        <mesh position={[0, 0.07, (Z_MIN + Z_MAX) / 2]} rotation={[-Math.PI / 2, 0, 0]}>
          <planeGeometry args={[LANE_W - 0.25, Z_MAX - Z_MIN]} />
          <meshBasicMaterial color={res} transparent opacity={tint} toneMapped={false} />
        </mesh>
      ) : null}
      <NetWalls />
      <Stumps z={-2.44} dir={-1} />
      <Stumps z={L} hit={hit} />
      <Bowler u={Math.max(-1.3, u)} kit={{shirt, cap: shirt}} />
      <Batter t={u} tc={d.tc ?? 0.6} shot={d.shot ?? 'defend'} />
      {pos ? (
        <>
          <mesh position={pos}>
            <sphereGeometry args={[0.036 * ballScale, 14, 10]} />
            <meshBasicMaterial color={col.ball} toneMapped={false} />
          </mesh>
          <Trail d={d} u={u} color={after > 0 && d.runs > 0 ? ORANGE : '#ffffff'} radius={0.02 * ballScale} />
        </>
      ) : null}
      {(look.box ?? 0) > 0 ? <Box a={look.box!} /> : null}
      {(look.dim ?? 0) > 0 ? (
        <mesh position={[0, 4.5, (Z_MIN + Z_MAX) / 2]} rotation={[-Math.PI / 2, 0, 0]}>
          <planeGeometry args={[LANE_W, Z_MAX - Z_MIN + 1]} />
          <meshBasicMaterial color="#0b0f16" transparent opacity={look.dim! * 0.62} />
        </mesh>
      ) : null}
    </group>
  );
};

const Box: React.FC<{a: number}> = ({a}) => {
  const len = Z_MAX - Z_MIN + 0.6;
  const zc = (Z_MAX + Z_MIN) / 2;
  const w = LANE_W - 0.1;
  const th = 0.22;
  return (
    <group position={[0, 4.6, zc]} scale={[1 + (1 - a) * 0.3, 1, 1 + (1 - a) * 0.1]}>
      {[
        [0, -len / 2, w, th],
        [0, len / 2, w, th],
        [-w / 2, 0, th, len],
        [w / 2, 0, th, len],
      ].map(([x, z, sx, sz], i) => (
        <mesh key={i} position={[x, 0, z]}>
          <boxGeometry args={[sx, 0.05, sz]} />
          <meshBasicMaterial color={Y} transparent opacity={a} toneMapped={false} />
        </mesh>
      ))}
    </group>
  );
};

/** Camera that frames all lanes from straight above (bowlers at the bottom). */
export function topDown(lanes: Lane[], aspect: number, zoom = 1): Pose {
  const xs = lanes.map((l) => l.x);
  const zs = lanes.map((l) => l.z);
  const w = Math.max(...xs) - Math.min(...xs) + LANE_W + 1;
  const zc = (Math.max(...zs) + Math.min(...zs)) / 2 + (Z_MIN + Z_MAX) / 2;
  const hgt = Math.max(...zs) - Math.min(...zs) + (Z_MAX - Z_MIN) + 2;
  const fov = 34;
  const needH = Math.max(hgt, w / aspect) / zoom;
  const dist = needH / 2 / Math.tan(((fov / 2) * Math.PI) / 180);
  return {position: [0, dist, zc + 0.001], target: [0, 0, zc], fov, up: [0, 0, 1]};
}

export function lerpPose(a: Pose, b: Pose, k: number): Pose {
  const v = (p: [number, number, number], q: [number, number, number]) => p.map((x, i) => lerp(x, q[i], k)) as [number, number, number];
  return {position: v(a.position, b.position), target: v(a.target, b.target), fov: lerp(a.fov, b.fov, k), up: v(a.up ?? [0, 1, 0], b.up ?? [0, 1, 0])};
}

/** Screen position (px) of a world point for a pose. */
export function project(pose: Pose, w: number, h: number, p: [number, number, number]): [number, number] {
  const cam = new THREE.PerspectiveCamera(pose.fov, w / h, 0.05, 900);
  cam.position.set(...pose.position);
  cam.up.set(...(pose.up ?? [0, 1, 0]));
  cam.lookAt(...pose.target);
  cam.updateMatrixWorld();
  const v = new THREE.Vector3(...p).project(cam);
  return [((v.x + 1) / 2) * w, ((1 - v.y) / 2) * h];
}

export type NetsProps = {
  lanes: Lane[];
  t: number;
  pose: Pose;
  looks?: (lane: Lane, i: number) => LaneLook;
  ballScale?: number;
  period?: number;
  kitShift?: number;
  labels?: boolean;
  lookStage?: {dim?: number; red?: number};
};

export const Nets: React.FC<NetsProps> = ({lanes, t, pose, looks, ballScale = 4, period = PERIOD, kitShift = 0, labels = true, lookStage}) => {
  const {w, h} = useFmt();
  return (
    <AbsoluteFill>
      <Stage pose={pose} look={{bloom: 0.25, ...lookStage}}>
        <mesh position={[0, -0.08, 20]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
          <planeGeometry args={[400, 400]} />
          <meshStandardMaterial color={col.sand} roughness={1} />
        </mesh>
        {lanes.map((lane, i) => (
          <LaneView key={i} lane={lane} idx={i} t={t} look={looks?.(lane, i) ?? {}} ballScale={ballScale} period={period} kitShift={kitShift} />
        ))}
      </Stage>
      {labels
        ? lanes.map((lane, i) => {
            const {u, d} = laneState(lane, i, t, period);
            const after = u - flightEnd(d);
            if (after < 0.05 || after > 1.0) return null;
            const wk = isWicket(d);
            if (!wk && d.runs < 4) return null;
            const [sx, sy] = project(pose, w, h, [lane.x, 0, lane.z + (wk ? L : 4)]);
            const a = ramp(after, 0.05, 0.2) * (1 - ramp(after, 0.75, 1.0));
            return (
              <div
                key={i}
                style={{
                  position: 'absolute',
                  left: sx,
                  top: sy - after * 30,
                  transform: `translate(-50%,-50%) scale(${0.8 + 0.2 * a})`,
                  opacity: a,
                  fontFamily: display,
                  fontWeight: 900,
                  fontSize: 34,
                  color: '#111',
                  background: wk ? GREEN : ORANGE,
                  padding: '2px 12px 4px',
                  borderRadius: 10,
                  whiteSpace: 'nowrap',
                }}
              >
                {wk ? 'OUT' : d.runs}
              </div>
            );
          })
        : null}
    </AbsoluteFill>
  );
};

export const LaneTag: React.FC<{pose: Pose; lane: Lane; text: string; a: number; color?: string}> = ({pose, lane, text, a, color = Y}) => {
  const {w, h} = useFmt();
  const [sx, sy] = project(pose, w, h, [lane.x, 4.6, lane.z + Z_MAX + 0.4]);
  if (a <= 0) return null;
  return (
    <div style={{position: 'absolute', left: sx, top: sy, transform: `translate(-50%,-110%) scale(${0.8 + 0.2 * a})`, opacity: a, fontFamily: mono, fontWeight: 700, fontSize: 24, color: '#111', background: color, padding: '3px 10px', borderRadius: 8, whiteSpace: 'nowrap'}}>
      {text}
    </div>
  );
};
