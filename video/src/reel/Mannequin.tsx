import React from 'react';
import * as THREE from 'three';
import {BOWL_CREASE, col, L} from './World';

/**
 * Low-poly mannequin players, posed procedurally.
 * Local frame: facing +Z, up +Y, character's right = -X.
 * Limb angles are rotations about the local X axis: positive swings a hanging limb backwards.
 */
export type Pose = {
  root: [number, number, number];
  yaw: number;
  lean: number;
  twist: number;
  head?: number;
  lSh: [number, number];
  lEl: number;
  rSh: [number, number];
  rEl: number;
  lHip: number;
  lKnee: number;
  rHip: number;
  rKnee: number;
  /** Optional hand targets in the torso frame (two-bone IK). */
  lHand?: [number, number, number];
  rHand?: [number, number, number];
};

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
const smooth = (v: number) => {
  const t = clamp01(v);
  return t * t * (3 - 2 * t);
};
const seg = (u: number, a: number, b: number) => smooth((u - a) / (b - a));

const Limb: React.FC<{len: number; r: number; color: string}> = ({len, r, color}) => (
  <mesh position={[0, -len / 2, 0]} castShadow>
    <capsuleGeometry args={[r, Math.max(0.01, len - 2 * r), 4, 8]} />
    <meshStandardMaterial color={color} roughness={0.7} flatShading />
  </mesh>
);

const UP = 0.31;
const FORE = 0.31;
/** Two-bone IK: shoulder at s, hand target t (torso frame). Elbows point down/back. */
export function solveArm(s: [number, number, number], t: [number, number, number]): {q: THREE.Quaternion; el: number} {
  const S = new THREE.Vector3(...s);
  const d = new THREE.Vector3(...t).sub(S);
  const D = Math.min(UP + FORE - 1e-3, Math.max(0.08, d.length()));
  const dh = d.clone().normalize();
  const cosA = (UP * UP + D * D - FORE * FORE) / (2 * UP * D);
  const A = Math.acos(Math.max(-1, Math.min(1, cosA)));
  const pole = new THREE.Vector3(0, -0.6, -1);
  const ph = pole.sub(dh.clone().multiplyScalar(pole.dot(dh))).normalize();
  const u1 = dh.clone().multiplyScalar(Math.cos(A)).add(ph.clone().multiplyScalar(Math.sin(A)));
  const E = u1.clone().multiplyScalar(UP);
  const u2 = dh.clone().multiplyScalar(D).sub(E).normalize();
  const y = u1.clone().negate();
  let z = u2.clone().sub(u1.clone().multiplyScalar(u2.dot(u1)));
  if (z.lengthSq() < 1e-8) z = new THREE.Vector3(0, 0, 1);
  z.normalize();
  const x = new THREE.Vector3().crossVectors(y, z);
  const q = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, z));
  const bend = Math.acos(Math.max(-1, Math.min(1, u1.dot(u2))));
  return {q, el: -bend};
}

export type Kit = {shirt: string; pants: string; skin?: string; helmet?: string; pads?: boolean; gloves?: string; cap?: string; glow?: number};

export const Mannequin: React.FC<{pose: Pose; kit: Kit; bat?: React.ReactNode; scale?: number; ballInHand?: boolean}> = ({pose, kit, bat, scale = 1, ballInHand}) => {
  const skin = kit.skin ?? col.skin;
  const legCol = kit.pants;
  const hand = (
    <mesh position={[0, -0.31, 0]} castShadow>
      <sphereGeometry args={[0.055, 8, 6]} />
      <meshStandardMaterial color={kit.gloves ?? skin} roughness={0.7} flatShading />
    </mesh>
  );
  const arm = (side: 1 | -1, sh: [number, number], el: number, target?: [number, number, number], extra?: React.ReactNode) => {
    if (target) {
      const ik = solveArm([side * 0.21, 0.5, 0], target);
      return (
        <group position={[side * 0.21, 0.5, 0]} quaternion={ik.q}>
          <Limb len={0.31} r={0.055} color={kit.shirt} />
          <group position={[0, -0.31, 0]} rotation={[ik.el, 0, 0]}>
            <Limb len={0.29} r={0.045} color={skin} />
            {hand}
            {extra}
          </group>
        </group>
      );
    }
    return (
      <group position={[side * 0.21, 0.5, 0]} rotation={[sh[0], 0, side * sh[1]]}>
        <Limb len={0.31} r={0.055} color={kit.shirt} />
        <group position={[0, -0.31, 0]} rotation={[el, 0, 0]}>
          <Limb len={0.29} r={0.045} color={skin} />
          {hand}
          {extra}
        </group>
      </group>
    );
  };
  const leg = (side: 1 | -1, hip: number, knee: number) => (
    <group position={[side * 0.1, 0, 0]} rotation={[hip, 0, 0]}>
      <Limb len={0.46} r={kit.pads ? 0.085 : 0.07} color={legCol} />
      <group position={[0, -0.46, 0]} rotation={[knee, 0, 0]}>
        <Limb len={0.44} r={kit.pads ? 0.085 : 0.06} color={kit.pads ? col.batterPad : legCol} />
        <mesh position={[0, -0.46, 0.06]} castShadow>
          <boxGeometry args={[0.1, 0.06, 0.24]} />
          <meshStandardMaterial color="#26282c" roughness={0.8} />
        </mesh>
      </group>
    </group>
  );
  return (
    <group position={pose.root} rotation={[0, pose.yaw, 0]} scale={scale}>
      <group rotation={[pose.lean, 0, 0]}>
        <mesh position={[0, 0.02, 0]} castShadow>
          <boxGeometry args={[0.3, 0.16, 0.18]} />
          <meshStandardMaterial color={legCol} roughness={0.8} flatShading />
        </mesh>
        <group rotation={[0, pose.twist, 0]}>
          <mesh position={[0, 0.3, 0]} castShadow>
            <capsuleGeometry args={[0.15, 0.32, 4, 8]} />
            <meshStandardMaterial color={kit.shirt} roughness={0.6} flatShading emissive={kit.shirt} emissiveIntensity={kit.glow ?? 0} />
          </mesh>
          <group position={[0, 0.72, 0]} rotation={[0, pose.head ?? 0, 0]}>
            <mesh castShadow>
              <sphereGeometry args={[0.11, 12, 10]} />
              <meshStandardMaterial color={skin} roughness={0.7} flatShading />
            </mesh>
            {kit.helmet ? (
              <>
                <mesh position={[0, 0.03, -0.01]} castShadow>
                  <sphereGeometry args={[0.125, 12, 8, 0, Math.PI * 2, 0, Math.PI / 1.8]} />
                  <meshStandardMaterial color={kit.helmet} roughness={0.4} />
                </mesh>
                <mesh position={[0, -0.02, 0.1]}>
                  <boxGeometry args={[0.2, 0.12, 0.02]} />
                  <meshStandardMaterial color="#9aa3ad" wireframe />
                </mesh>
              </>
            ) : null}
            {kit.cap ? (
              <>
                <mesh position={[0, 0.04, 0]}>
                  <sphereGeometry args={[0.118, 12, 6, 0, Math.PI * 2, 0, Math.PI / 2.2]} />
                  <meshStandardMaterial color={kit.cap} />
                </mesh>
                <mesh position={[0, 0.05, 0.11]} rotation={[0.2, 0, 0]}>
                  <boxGeometry args={[0.16, 0.012, 0.1]} />
                  <meshStandardMaterial color={kit.cap} />
                </mesh>
              </>
            ) : null}
          </group>
          {arm(-1, pose.rSh, pose.rEl, pose.rHand, ballInHand ? (
            <mesh position={[0, -0.36, 0.02]}>
              <sphereGeometry args={[0.05, 12, 10]} />
              <meshStandardMaterial color={col.ball} emissive={col.ball} emissiveIntensity={0.4} />
            </mesh>
          ) : null)}
          {arm(1, pose.lSh, pose.lEl, pose.lHand)}
          {bat}
        </group>
        {leg(-1, pose.rHip, pose.rKnee)}
        {leg(1, pose.lHip, pose.lKnee)}
      </group>
    </group>
  );
};

/* ------------------------------------------------------------------ bowler */

/** Where the bowler's body is at release (hand ends up near the sim's release point). */
const Z0 = BOWL_CREASE - 0.32;

/**
 * Bowler pose at u seconds relative to release. Run-up, bound, delivery stride,
 * windmill arm (release just past vertical), follow-through.
 */
export function bowlerPose(u: number, x0 = 0.5): Pose {
  const RUN = 6.4;
  let z: number;
  let y = 0.95;
  let lean = 0.22;
  let yaw = 0;
  let twist = 0;
  let lHip = 0;
  let rHip = 0;
  let lKnee = 0.2;
  let rKnee = 0.2;
  let rSh: [number, number] = [0, 0.1];
  let lSh: [number, number] = [0, 0.1];
  let rEl = -0.3;
  let lEl = -0.4;
  if (u < -0.62) {
    z = Z0 - 2.7 + RUN * (u + 0.62);
    const ph = u * Math.PI * 2 * 2.7;
    lHip = -0.75 * Math.sin(ph);
    rHip = 0.75 * Math.sin(ph);
    lKnee = 0.25 + 0.9 * Math.max(0, Math.sin(ph + 1.2));
    rKnee = 0.25 + 0.9 * Math.max(0, -Math.sin(ph + 1.2));
    lSh = [0.8 * Math.sin(ph), 0.12];
    rSh = [-0.8 * Math.sin(ph), 0.12];
    lEl = -1.3;
    rEl = -1.3;
    y = 0.93 + 0.06 * Math.abs(Math.cos(ph));
    lean = 0.28;
  } else if (u < -0.3) {
    const p = (u + 0.62) / 0.32;
    z = Z0 - 2.7 + (2.05 * p);
    y = 0.95 + 0.32 * Math.sin(Math.PI * p);
    lean = lerp(0.2, -0.12, p);
    yaw = lerp(0, 0.9, smooth(p));
    lHip = lerp(-0.4, -1.1, p);
    lKnee = lerp(0.6, 1.3, p);
    rHip = lerp(0.3, -0.2, p);
    rKnee = lerp(0.4, 1.5, p);
    lSh = [lerp(0, -2.8, smooth(p)), 0.1];
    lEl = lerp(-0.8, -0.1, p);
    rSh = [lerp(0, 0.5, p), 0.15];
    rEl = lerp(-0.6, -0.9, p);
  } else if (u < 0) {
    const p = (u + 0.3) / 0.3;
    z = Z0 - 0.65 + 0.65 * smooth(p);
    y = lerp(0.95, 0.88, p);
    lean = lerp(-0.15, 0.32, smooth(p));
    yaw = lerp(0.9, 0.1, smooth(p));
    twist = lerp(0.2, -0.15, p);
    lHip = lerp(-1.1, -0.75, p);
    lKnee = lerp(1.0, 0.05, smooth(p * 1.4));
    rHip = lerp(-0.2, 0.55, p);
    rKnee = lerp(1.2, 0.5, p);
    lSh = [lerp(-2.8, 0.4, smooth(p)), 0.12];
    lEl = lerp(-0.1, -0.9, p);
    rSh = [lerp(0.5, Math.PI + 0.22, smooth(p)), 0.04];
    rEl = lerp(-0.6, 0, smooth(p * 1.3));
  } else {
    const p = Math.min(1, u / 0.35);
    const settle = Math.min(1, u / 1.4);
    z = Z0 + 2.6 * (1 - Math.exp(-u * 1.9));
    y = 0.88 + 0.06 * settle;
    lean = lerp(0.32, 0.7, smooth(p)) - 0.35 * smooth((u - 0.35) / 0.9);
    yaw = lerp(0.1, -0.35, smooth(p)) * (1 - settle * 0.6);
    twist = lerp(-0.15, -0.5, smooth(p)) * (1 - settle * 0.7);
    rSh = [lerp(Math.PI + 0.22, 2 * Math.PI + 0.35, smooth(p)), lerp(0.04, -0.4, p)];
    rEl = -0.2;
    lSh = [lerp(0.4, 0.9, p), 0.2];
    lEl = -0.6;
    const ph = u * Math.PI * 2 * 1.6;
    const step = smooth((u - 0.25) / 0.3);
    lHip = lerp(-0.75, -0.4 * Math.sin(ph), step);
    rHip = lerp(0.55, 0.4 * Math.sin(ph), step);
    lKnee = lerp(0.1, 0.3 + 0.4 * Math.max(0, Math.sin(ph + 1)), step);
    rKnee = lerp(0.5, 0.3 + 0.4 * Math.max(0, -Math.sin(ph + 1)), step);
  }
  return {root: [x0, y, z], yaw, lean, twist, lSh, lEl, rSh, rEl, lHip, lKnee, rHip, rKnee};
}

export const Bowler: React.FC<{u: number; x0?: number; kit?: Partial<Kit>; scale?: number}> = ({u, x0, kit, scale}) => (
  <Mannequin pose={bowlerPose(u, x0)} kit={{shirt: col.bowler, pants: '#f4f6f8', cap: col.bowlerDark, ...kit}} scale={scale} ballInHand={u < 0} />
);

/* ------------------------------------------------------------------ batter */

export type Shot = 'defend' | 'miss' | 'slog' | 'drive' | 'leave' | 'pull' | 'edge' | 'fend' | 'cut' | string;
type V3 = [number, number, number];
type Key = {k: number; g: V3; d: V3};

/*
 * Bat keyframes in the batter's torso frame: +X = towards the bowler, +Z = chest (off side), -Z = leg side.
 * g = top-hand grip, d = direction from grip to the toe of the bat. k = seconds relative to the ball reaching the crease.
 */
const STANCE: Key = {k: -0.6, g: [-0.02, 0.08, 0.3], d: [-0.12, -1, 0.12]};
const LIFT: Key = {k: -0.24, g: [-0.22, 0.38, 0.22], d: [-0.65, 0.72, 0.05]};
const SHOTS: Record<string, Key[]> = {
  defend: [STANCE, {k: -0.3, g: [-0.15, 0.3, 0.25], d: [-0.5, 0.5, 0.1]}, {k: -0.04, g: [0.3, 0.16, 0.36], d: [0.35, -0.95, 0.05]}, {k: 0.5, g: [0.3, 0.16, 0.36], d: [0.35, -0.95, 0.05]}],
  edge: [STANCE, {k: -0.3, g: [-0.15, 0.3, 0.25], d: [-0.5, 0.5, 0.1]}, {k: -0.04, g: [0.3, 0.2, 0.42], d: [0.4, -0.85, 0.3]}, {k: 0.5, g: [0.28, 0.2, 0.4], d: [0.4, -0.85, 0.3]}],
  fend: [STANCE, {k: -0.3, g: [-0.1, 0.35, 0.25], d: [-0.4, 0.6, 0.1]}, {k: -0.04, g: [0.18, 0.55, 0.38], d: [0.3, -0.8, 0.25]}, {k: 0.5, g: [0.18, 0.55, 0.38], d: [0.3, -0.8, 0.25]}],
  drive: [STANCE, LIFT, {k: -0.1, g: [0.0, 0.25, 0.35], d: [-0.4, -0.9, 0.1]}, {k: 0.0, g: [0.32, 0.08, 0.38], d: [0.15, -1, 0.1]}, {k: 0.18, g: [0.35, 0.45, 0.25], d: [0.75, 0.55, -0.1]}, {k: 0.45, g: [0.2, 0.62, 0.1], d: [0.2, 0.9, -0.4]}],
  slog: [STANCE, LIFT, {k: -0.1, g: [0.0, 0.25, 0.35], d: [-0.4, -0.9, 0.2]}, {k: 0.0, g: [0.32, 0.1, 0.32], d: [0.2, -0.95, -0.2]}, {k: 0.16, g: [0.2, 0.45, 0.0], d: [0.3, 0.3, -0.9]}, {k: 0.45, g: [-0.05, 0.62, -0.1], d: [-0.4, 0.6, -0.7]}],
  miss: [STANCE, LIFT, {k: -0.04, g: [0.0, 0.25, 0.35], d: [-0.4, -0.9, 0.1]}, {k: 0.08, g: [0.32, 0.1, 0.38], d: [0.2, -1, 0.1]}, {k: 0.35, g: [0.3, 0.45, 0.2], d: [0.7, 0.6, -0.2]}],
  pull: [STANCE, LIFT, {k: -0.08, g: [0.1, 0.45, 0.32], d: [-0.1, 0.15, 1]}, {k: 0.0, g: [0.2, 0.45, 0.2], d: [0.95, 0.1, 0.2]}, {k: 0.2, g: [0.05, 0.5, -0.05], d: [0.2, 0.2, -1]}, {k: 0.45, g: [-0.1, 0.55, -0.1], d: [-0.4, 0.5, -0.8]}],
  leave: [STANCE, LIFT, {k: -0.1, g: [-0.15, 0.62, 0.12], d: [-0.35, 0.92, -0.1]}, {k: 0.5, g: [-0.15, 0.62, 0.12], d: [-0.35, 0.92, -0.1]}],
};
SHOTS.cut = SHOTS.pull;

function sampleKeys(keys: Key[], k: number): {g: V3; d: V3} {
  if (k <= keys[0].k) return keys[0];
  for (let i = 0; i < keys.length - 1; i++) {
    const a = keys[i];
    const b = keys[i + 1];
    if (k <= b.k) {
      const t = smooth((k - a.k) / (b.k - a.k));
      const g = a.g.map((v, j) => lerp(v, b.g[j], t)) as V3;
      const d = new THREE.Vector3(...a.d).normalize().lerp(new THREE.Vector3(...b.d).normalize(), t).normalize();
      return {g, d: [d.x, d.y, d.z]};
    }
  }
  return keys[keys.length - 1];
}

/** Batter (right-handed) at time t after release; tc = time the ball reaches the crease. */
export function batterPose(t: number, tc: number, shot: Shot): {pose: Pose; grip: V3; dir: V3} {
  const k = t - tc;
  const keys = SHOTS[shot] ?? SHOTS.defend;
  const {g, d} = sampleKeys(keys, k);
  const stride = ['defend', 'edge', 'drive', 'slog', 'miss'].includes(shot) ? seg(k, -0.32, -0.06) : 0;
  const back = seg(k, -0.55, -0.22);
  // torso-frame grip -> mannequin local frame: local +Z = chest (off side) = frame +Z; towards bowler = local... see Bat
  const dd = new THREE.Vector3(...d);
  const top: V3 = [g[0], g[1], g[2]];
  const bottom: V3 = [g[0] + dd.x * 0.11, g[1] + dd.y * 0.11, g[2] + dd.z * 0.11];
  const toLocal = (v: V3): V3 => [v[0], v[1], v[2]];
  const pose: Pose = {
    root: [-0.42, 0.9 - 0.06 * (1 - back), L - 1.35 - stride * 0.25],
    yaw: Math.PI / 2,
    lean: 0.2 + stride * 0.1,
    twist: 0,
    head: -1.2,
    lSh: [0, 0],
    lEl: 0,
    rSh: [0, 0],
    rEl: 0,
    lHand: toLocal(top),
    rHand: toLocal(bottom),
    lHip: -0.2 - stride * 0.5,
    lKnee: 0.25 + stride * 0.25,
    rHip: 0.15 + stride * 0.2,
    rKnee: 0.28,
  };
  return {pose, grip: toLocal(top), dir: toLocal(d)};
}

/** Bat with its handle at `grip`, pointing along `dir` (torso frame). */
export const Bat: React.FC<{grip: V3; dir: V3}> = ({grip, dir}) => {
  const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, -1, 0), new THREE.Vector3(...dir).normalize());
  return (
    <group position={grip} quaternion={q}>
      <mesh position={[0, 0.0, 0]} castShadow>
        <cylinderGeometry args={[0.018, 0.018, 0.3, 8]} />
        <meshStandardMaterial color="#222" />
      </mesh>
      <mesh position={[0, -0.5, 0]} castShadow>
        <boxGeometry args={[0.11, 0.6, 0.05]} />
        <meshStandardMaterial color={col.bat} roughness={0.6} flatShading />
      </mesh>
    </group>
  );
};

export const Batter: React.FC<{t: number; tc: number; shot: Shot; kit?: Partial<Kit>; scale?: number; root?: V3}> = ({t, tc, shot, kit, scale}) => {
  const {pose, grip, dir} = batterPose(t, tc, shot);
  return <Mannequin pose={pose} kit={{shirt: col.batter, pants: col.batter, helmet: col.helmet, pads: true, gloves: '#ffffff', ...kit}} bat={<Bat grip={grip} dir={dir} />} scale={scale} />;
};

/** Wicketkeeper crouched behind the stumps; `catchUp` raises the gloves for a take. */
export const Keeper: React.FC<{catchUp?: number}> = ({catchUp = 0}) => (
  <Mannequin
    pose={{
      root: [0.25, 0.62 + catchUp * 0.25, L + 1.9],
      yaw: Math.PI,
      lean: 0.5 - catchUp * 0.3,
      twist: 0,
      lSh: [-1.2 - catchUp * 0.6, 0.15],
      lEl: -0.3,
      rSh: [-1.2 - catchUp * 0.6, 0.15],
      rEl: -0.3,
      lHip: -1.4 + catchUp * 0.5,
      lKnee: 2.2 - catchUp * 0.9,
      rHip: -1.4 + catchUp * 0.5,
      rKnee: 2.2 - catchUp * 0.9,
    }}
    kit={{shirt: '#e9edf2', pants: '#e9edf2', helmet: col.helmet, pads: true, gloves: '#f5d742'}}
  />
);

/** A standing fielder, gently swaying. */
export const Fielder: React.FC<{x: number; z: number; t: number; face?: number}> = ({x, z, t, face}) => {
  const sway = Math.sin(t * 1.3 + x) * 0.05;
  const yaw = face ?? Math.atan2(0 - x, L - 2 - z);
  return (
    <Mannequin
      pose={{root: [x, 0.93, z], yaw, lean: 0.25 + sway, twist: 0, lSh: [-0.4, 0.2], lEl: -0.6, rSh: [-0.4, 0.2], rEl: -0.6, lHip: -0.25, lKnee: 0.35, rHip: 0.1, rKnee: 0.35}}
      kit={{shirt: col.bowler, pants: '#f4f6f8', cap: col.bowlerDark}}
    />
  );
};

export const FIELDERS: [number, number][] = [
  [6, L + 3.5],
  [4.6, L + 4.4],
  [12, L - 2],
  [-14, L - 6],
  [20, 8],
  [-22, 10],
  [-8, -14],
  [9, -16],
];
