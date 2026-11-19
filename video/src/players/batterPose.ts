import * as THREE from 'three';
import {RoundedBoxGeometry} from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
import {Delivery, flightEnd, positionAt} from '../lib/data';
import {poseClip, Rig} from './rig';

/**
 * The batter, posed procedurally on the realistic cricketer (the asset pack has no batting motion capture).
 *
 * Each frame: restore the base pose → hips (crouch, shift, turn) → two-bone IK puts the feet on their
 * stance/stride targets → spine leans and twists → a limited look-at turns neck and head towards the bowler,
 * then the ball → the bat follows the shot's keyframes, with its sweet spot driven to where the simulated ball
 * is at contact → two-bone IK puts both hands on the handle with fixed elbow directions (no flipping) →
 * gloves and pads ride on the bones.
 *
 * Batter-local frame (centimetres, before scaling): +Z = facing (off side), +X = towards the bowler, +Y = up.
 * Timing is relative to contact (rel = ball time − contact time).
 */

const v = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);
const clamp01 = (x: number) => Math.max(0, Math.min(1, x));
const smooth = (x: number) => {
  const c = clamp01(x);
  return c * c * (3 - 2 * c);
};

// ---------------------------------------------------------------- keyframed body plan

type Key = {
  t: number;
  hips: THREE.Vector3; // offset of the pelvis (cm)
  yaw: number; // pelvis turn about up (+ = chest opens towards the bowler)
  lean: number; // forward lean (towards the facing direction)
  side: number; // lean towards the bowler
  front: THREE.Vector3; // front (left) foot target
  back: THREE.Vector3; // back (right) foot target
  g: THREE.Vector3; // top-hand grip point on the handle
  d: THREE.Vector3; // bat direction, handle → toe
};

const SWEET = 60; // cm from top grip to the middle of the blade
const STANCE_FRONT = v(24, 9, 6);
const STANCE_BACK = v(-26, 9, -2);
const STANCE: Omit<Key, 't'> = {
  hips: v(2, -13, 4), yaw: 0.1, lean: 0.3, side: 0.04,
  front: STANCE_FRONT.clone(), back: STANCE_BACK.clone(),
  g: v(-6, 84, 24), d: v(-0.22, -0.96, -0.18).normalize(),
};
const BACKLIFT: Partial<Key> = {g: v(-16, 116, 8), d: v(-0.62, 0.66, -0.42).normalize(), lean: 0.28};

function k(t: number, base: Omit<Key, 't'>, over: Partial<Key> = {}): Key {
  const o = {...base, ...over} as Key;
  return {
    t, hips: o.hips.clone(), yaw: o.yaw, lean: o.lean, side: o.side,
    front: o.front.clone(), back: o.back.clone(), g: o.g.clone(), d: o.d.clone().normalize(),
  };
}

/** Grip such that the bat's sweet spot sits at `p` with direction `d`. */
const gripFor = (p: THREE.Vector3, d: THREE.Vector3) => p.clone().sub(d.clone().normalize().multiplyScalar(SWEET));

function planShot(d: Delivery, c: THREE.Vector3): Key[] {
  const shot = d.shot ?? 'defend';
  const short = (d.length ?? 0) > 8.5;
  const line = Math.max(-20, Math.min(30, c.z - 30)); // how far across towards the ball the front foot goes
  const S = STANCE;
  const stride = {
    hips: v(22, -19, 8), lean: 0.45, side: 0.16,
    front: v(64, 9, 12 + line * 0.5), back: STANCE_BACK.clone().add(v(4, 2, 0)),
  };
  const lift = (front: THREE.Vector3) => front.clone().add(v(-12, 12, 0)); // foot in the air mid-stride

  if (shot === 'pull' || shot === 'fend' || (short && shot !== 'leave' && shot !== 'defend' && shot !== 'edge')) {
    const back = v(-34, 9, 16);
    const contactD = v(0.18, 0.06, 0.98);
    const hit = {...S, hips: v(-6, -12, 10), yaw: 0.9, lean: 0.18, side: -0.05, back, front: v(20, 9, 0), d: contactD, g: gripFor(c, contactD)};
    const finish = {...hit, yaw: 1.25, g: v(4, 138, -24), d: v(-0.4, 0.32, -0.86)};
    return [k(-0.6, S), k(-0.3, S, BACKLIFT), k(-0.12, S, {...BACKLIFT, back, hips: v(-4, -12, 8), yaw: 0.3}), k(0, hit), k(0.3, finish), k(1.0, finish), k(1.8, S)];
  }
  if (shot === 'slog') {
    const front = v(34, 9, -22);
    const contactD = v(0.32, -0.74, -0.59);
    const big = {g: v(-22, 136, 0), d: v(-0.52, 0.8, -0.22), lean: 0.22};
    const hit = {...S, ...stride, front, hips: v(10, -16, -4), yaw: 0.65, lean: 0.25, side: 0.05, d: contactD, g: gripFor(c, contactD)};
    const finish = {...hit, yaw: 0.95, g: v(12, 152, -16), d: v(-0.5, 0.66, -0.56)};
    return [k(-0.6, S), k(-0.32, S, big), k(-0.12, S, {...big, front: lift(front), hips: v(6, -15, 0)}), k(0, hit), k(0.32, finish), k(1.1, finish), k(2.0, S)];
  }
  if (shot === 'leave') {
    const up = {g: v(-4, 146, 8), d: v(-0.18, 0.96, -0.2), lean: 0.24};
    const step = {...S, ...stride, front: v(46, 9, 14), hips: v(14, -16, 8), ...up};
    return [k(-0.6, S), k(-0.3, S, BACKLIFT), k(-0.12, S, {...BACKLIFT, front: lift(v(46, 9, 14))}), k(0.02, step), k(0.9, step), k(1.8, S)];
  }
  if (shot === 'miss') {
    // the yorker/fast ball beats the bat: front foot half forward, bat arrives late and above the ball
    const contactD = v(0.0, -0.99, 0.12);
    const late = {...S, ...stride, front: v(46, 9, 8), hips: v(14, -17, 6), d: contactD, g: gripFor(c.clone().add(v(-10, 20, 0)), contactD)};
    const after = {...late, g: v(18, 112, 28), d: v(0.45, -0.85, 0.27), lean: 0.38};
    return [k(-0.6, S), k(-0.28, S, BACKLIFT), k(-0.1, S, {...BACKLIFT, front: lift(v(46, 9, 8))}), k(0.04, late), k(0.4, after), k(1.4, after), k(2.2, S)];
  }
  if (shot === 'drive') {
    const contactD = v(0.08, -0.97, 0.22);
    const hit = {...S, ...stride, d: contactD, g: gripFor(c, contactD)};
    const finish = {...hit, g: v(22, 148, 22), d: v(-0.3, 0.82, -0.48), lean: 0.36};
    return [k(-0.6, S), k(-0.28, S, BACKLIFT), k(-0.1, S, {...BACKLIFT, front: lift(stride.front)}), k(0, hit), k(0.32, finish), k(1.1, finish), k(2.0, S)];
  }
  // forward defence / edge: soft hands, handle ahead of the blade, bat beside the front pad
  const contactD = v(-0.26, -0.94, 0.2);
  const target = shot === 'edge' ? c.clone().add(v(0, 0, -7)) : c;
  const block = {...S, ...stride, d: contactD, g: gripFor(target, contactD)};
  return [k(-0.6, S), k(-0.26, S, {g: v(-12, 106, 14), d: v(-0.4, 0.5, -0.3), lean: 0.3}), k(-0.1, S, {front: lift(stride.front), g: v(-8, 100, 18), d: v(-0.3, 0.2, -0.2)}),
    k(0, block), k(0.35, block), k(1.0, block), k(1.8, S)];
}

function sample(keys: Key[], t: number): Omit<Key, 't'> {
  if (t <= keys[0].t) return keys[0];
  for (let i = 0; i < keys.length - 1; i++) {
    const a = keys[i];
    const b = keys[i + 1];
    if (t <= b.t) {
      const u = smooth((t - a.t) / (b.t - a.t));
      return {
        hips: a.hips.clone().lerp(b.hips, u), yaw: a.yaw + (b.yaw - a.yaw) * u, lean: a.lean + (b.lean - a.lean) * u,
        side: a.side + (b.side - a.side) * u, front: a.front.clone().lerp(b.front, u), back: a.back.clone().lerp(b.back, u),
        g: a.g.clone().lerp(b.g, u), d: a.d.clone().lerp(b.d, u).normalize(),
      };
    }
  }
  return keys[keys.length - 1];
}

// ---------------------------------------------------------------- skeleton helpers (world space)

const wp = (o: THREE.Object3D) => o.getWorldPosition(new THREE.Vector3());

function rotateWorld(bone: THREE.Object3D, q: THREE.Quaternion): void {
  const world = bone.getWorldQuaternion(new THREE.Quaternion());
  const parent = bone.parent ? bone.parent.getWorldQuaternion(new THREE.Quaternion()) : new THREE.Quaternion();
  bone.quaternion.copy(parent.invert().multiply(q.clone().multiply(world)));
  bone.updateMatrixWorld(true);
}

function aim(bone: THREE.Object3D, from: THREE.Vector3, to: THREE.Vector3, amount = 1): void {
  const p = wp(bone);
  const a = from.clone().sub(p).normalize();
  const b = to.clone().sub(p).normalize();
  if (a.lengthSq() < 1e-9 || b.lengthSq() < 1e-9) return;
  const q = new THREE.Quaternion().setFromUnitVectors(a, b);
  if (amount < 1) q.slerp(new THREE.Quaternion(), 1 - amount);
  rotateWorld(bone, q);
}

function moveWorld(bone: THREE.Object3D, delta: THREE.Vector3): void {
  const p = wp(bone);
  const parent = bone.parent!;
  const a = parent.worldToLocal(p.clone());
  const b = parent.worldToLocal(p.clone().add(delta));
  bone.position.add(b.sub(a));
  bone.updateMatrixWorld(true);
}

/** Analytic two-bone IK: put `end` on `target`, bending towards `pole` (world space). */
function twoBone(upper: THREE.Object3D, mid: THREE.Object3D, end: THREE.Object3D, target: THREE.Vector3, pole: THREE.Vector3): void {
  const A = wp(upper);
  const B = wp(mid);
  const C = wp(end);
  const la = A.distanceTo(B);
  const lb = B.distanceTo(C);
  const toT = target.clone().sub(A);
  const d = Math.min(Math.max(toT.length(), Math.abs(la - lb) + 1e-3), (la + lb) * 0.999);
  const dir = toT.normalize();
  const cosA = (la * la + d * d - lb * lb) / (2 * la * d);
  const sinA = Math.sqrt(Math.max(0, 1 - cosA * cosA));
  const toPole = pole.clone().sub(A);
  const perp = toPole.sub(dir.clone().multiplyScalar(toPole.dot(dir)));
  if (perp.lengthSq() < 1e-9) return;
  perp.normalize();
  const B2 = A.clone().add(dir.clone().multiplyScalar(la * cosA)).add(perp.multiplyScalar(la * sinA));
  aim(upper, B, B2);
  aim(mid, wp(end), A.clone().add(dir.multiplyScalar(d)));
}

/** Turn the bone's local axis `axis` (e.g. head forward) towards `target`, by at most `maxAngle`. */
function look(bone: THREE.Object3D, axis: THREE.Vector3, target: THREE.Vector3, amount: number, maxAngle: number): void {
  const q = bone.getWorldQuaternion(new THREE.Quaternion());
  const fwd = axis.clone().applyQuaternion(q).normalize();
  const want = target.clone().sub(wp(bone)).normalize();
  let r = new THREE.Quaternion().setFromUnitVectors(fwd, want);
  const angle = 2 * Math.acos(Math.min(1, Math.abs(r.w)));
  const scale = Math.min(amount, angle > 1e-6 ? maxAngle / angle : 1);
  r = new THREE.Quaternion().slerp(r, scale);
  rotateWorld(bone, r);
}

// ---------------------------------------------------------------- kit

export type BatterKit = {bat: THREE.Group};

export function buildKit(rig: Rig): BatterKit {
  const wood = new THREE.MeshStandardMaterial({name: 'BatWillow', color: '#dcc08c', roughness: 0.5});
  const grip = new THREE.MeshStandardMaterial({name: 'BatGrip', color: '#202634', roughness: 0.85});
  const white = new THREE.MeshStandardMaterial({name: 'PadWhite', color: '#eef0f2', roughness: 0.62, side: THREE.DoubleSide});
  const glove = new THREE.MeshStandardMaterial({name: 'GloveWhite', color: '#f2f3f5', roughness: 0.55});
  const navy = new THREE.MeshStandardMaterial({name: 'HelmetNavy', color: '#13254d', roughness: 0.32, metalness: 0.15});
  const steel = new THREE.MeshStandardMaterial({name: 'GrilleSteel', color: '#b9c2cf', roughness: 0.3, metalness: 0.85});
  rig.root.updateMatrixWorld(true);
  const rootScale = rig.root.getWorldScale(new THREE.Vector3()).x;
  const fit = (m: THREE.Object3D, bone: THREE.Object3D) => {
    const kk = rootScale / bone.getWorldScale(new THREE.Vector3()).x;
    m.scale.multiplyScalar(kk);
    m.position.multiplyScalar(kk);
  };
  // Bat (cm), built along -Y from the top grip: 33 cm handle, then a 56 cm blade with a raised spine.
  const bat = new THREE.Group();
  bat.name = 'Bat';
  const handle = new THREE.Mesh(new THREE.CylinderGeometry(1.6, 1.75, 34, 18), grip);
  handle.position.y = -12;
  const blade = new THREE.Mesh(new RoundedBoxGeometry(10.8, 56, 3.6, 3, 1.2), wood);
  blade.position.y = -29 - 28;
  const spine = new THREE.Mesh(new RoundedBoxGeometry(5.5, 46, 3.2, 3, 1.4), wood);
  spine.position.set(0, -29 - 31, -2.4);
  const toe = new THREE.Mesh(new RoundedBoxGeometry(10.8, 3, 4.6, 2, 1.2), wood);
  toe.position.set(0, -29 - 55, -0.6);
  bat.add(handle, blade, spine, toe);
  rig.root.add(bat);
  // Pads: curved shells wrapping the front of each shin, with cane ridges, a knee roll and side wings.
  const pad = (bone: THREE.Bone) => {
    const gp = new THREE.Group();
    const shell = new THREE.Mesh(new THREE.CylinderGeometry(8.4, 7.2, 46, 24, 1, true, -Math.PI * 0.62, Math.PI * 1.24), white);
    shell.position.set(0, 25, 1.5);
    gp.add(shell);
    for (const a of [-0.55, -0.18, 0.18, 0.55]) {
      const ridge = new THREE.Mesh(new THREE.CapsuleGeometry(1.15, 38, 4, 8), white);
      ridge.position.set(Math.sin(a) * 8.6, 26, 1.5 + Math.cos(a) * 8.6);
      gp.add(ridge);
    }
    const knee = new THREE.Mesh(new THREE.TorusGeometry(7.5, 3.6, 10, 20, Math.PI * 1.2), white);
    knee.rotation.set(Math.PI / 2, 0, -Math.PI * 0.1 + Math.PI);
    knee.position.set(0, 1, 2);
    gp.add(knee);
    fit(gp, bone);
    bone.add(gp);
  };
  pad(rig.bones.LeftLeg);
  pad(rig.bones.RightLeg);
  // Batting gloves along each hand bone (finger direction is the bone's +Y).
  for (const h of [rig.bones.LeftHand, rig.bones.RightHand]) {
    const gl = new THREE.Group();
    const palm = new THREE.Mesh(new RoundedBoxGeometry(9.5, 10, 6.5, 4, 2.6), glove);
    palm.position.set(0, 5, 0);
    const fingers = new THREE.Mesh(new RoundedBoxGeometry(9, 8, 5.5, 4, 2.4), glove);
    fingers.position.set(0, 12, 1.2);
    fingers.rotation.x = 0.5;
    const cuff = new THREE.Mesh(new THREE.CylinderGeometry(4.6, 4.8, 6, 16), glove);
    cuff.position.set(0, -2, 0);
    gl.add(palm, fingers, cuff);
    fit(gl, h);
    h.add(gl);
  }
  // Helmet with peak and grille.
  const helmet = new THREE.Group();
  const shell = new THREE.Mesh(new THREE.SphereGeometry(13.5, 28, 20, 0, Math.PI * 2, 0, Math.PI * 0.6), navy);
  shell.position.set(0, 9, -1);
  shell.scale.set(1, 1.05, 1.15);
  const peak = new THREE.Mesh(new RoundedBoxGeometry(20, 1.4, 7, 2, 0.6), navy);
  peak.position.set(0, 9, 14);
  peak.rotation.x = -0.15;
  const grille = new THREE.Group();
  for (let i = 0; i < 4; i++) {
    const bar = new THREE.Mesh(new THREE.TorusGeometry(11, 0.42, 6, 24, Math.PI), steel);
    bar.rotation.z = Math.PI;
    bar.position.set(0, 4 - i * 4.0, 11.5);
    bar.scale.set(1, 0.55, 1);
    grille.add(bar);
  }
  helmet.add(shell, peak, grille);
  fit(helmet, rig.bones.Head);
  rig.bones.Head.add(helmet);
  return {bat};
}

// ---------------------------------------------------------------- pose

const REST = new WeakMap<Rig, [THREE.Object3D, THREE.Vector3, THREE.Quaternion][]>();
function restPose(rig: Rig) {
  let r = REST.get(rig);
  if (!r) {
    poseClip(rig, 'RightArmFastBowler', 0.0);
    r = [];
    rig.root.traverse((o) => {
      if ((o as THREE.Bone).isBone) r!.push([o, o.position.clone(), o.quaternion.clone()]);
    });
    REST.set(rig, r);
  }
  return r;
}

/** Where the batter stands (scene coords): just in front of his stumps, on the leg side, facing the off side. */
export const BATTER_POS: [number, number, number] = [-0.42, 0, 19.15];

/** Pose the batter for ball time `t` (seconds since release). `group` positions him on the pitch. */
export function poseBatter(rig: Rig, kit: BatterKit, group: THREE.Group, d: Delivery | undefined, t: number): void {
  if (rig.root.parent !== group) group.add(rig.root);
  group.position.set(...BATTER_POS);
  group.rotation.set(0, Math.PI / 2, 0);
  group.updateMatrixWorld(true);

  for (const [bone, pos, quat] of restPose(rig)) {
    bone.position.copy(pos);
    bone.quaternion.copy(quat);
  }
  poseClip(rig, 'RightArmFastBowler', 0.0);
  rig.root.updateMatrixWorld(true);
  const L = (p: THREE.Vector3) => rig.root.localToWorld(p.clone()); // batter-local cm → world
  const Ldir = (p: THREE.Vector3) => L(p).sub(L(v(0, 0, 0))).normalize();
  const up = Ldir(v(0, 1, 0));
  const side = Ldir(v(1, 0, 0)); // towards the bowler
  const fwd = Ldir(v(0, 0, 1)); // facing (off side)

  const contactT = d ? (d.tc ?? flightEnd(d)) : 0;
  const contactLocal = d ? rig.root.worldToLocal(new THREE.Vector3(...positionAt(d, contactT))) : v(40, 50, 30);
  const keys = d ? planShot(d, contactLocal) : [k(0, STANCE)];
  const rel = t - contactT;
  const s = sample(keys, rel);

  // 1. pelvis: crouch/shift, then turn and lean.
  const hips = rig.bones.Hips;
  moveWorld(hips, L(s.hips).sub(L(v(0, 0, 0))));
  rotateWorld(hips, new THREE.Quaternion().setFromAxisAngle(up, s.yaw * 0.5));

  // 2. feet planted on their targets; knees bend forward and slightly out.
  twoBone(rig.bones.LeftUpLeg, rig.bones.LeftLeg, rig.bones.LeftFoot, L(s.front), L(s.front.clone().add(v(20, 40, 60))));
  twoBone(rig.bones.RightUpLeg, rig.bones.RightLeg, rig.bones.RightFoot, L(s.back), L(s.back.clone().add(v(-10, 40, 60))));
  // keep feet flat-ish: point the toes along the crease (front) and square (back)
  aim(rig.bones.LeftFoot, wp(rig.bones.LeftToeBase), L(s.front.clone().add(v(10, -6, 12))), 0.7);
  aim(rig.bones.RightFoot, wp(rig.bones.RightToeBase), L(s.back.clone().add(v(2, -6, 14))), 0.7);

  // 3. spine: forward lean over the bat, a little towards the bowler, chest turned with the hips.
  const leanQ = new THREE.Quaternion().setFromAxisAngle(side, s.lean * 0.5).multiply(new THREE.Quaternion().setFromAxisAngle(fwd, -s.side * 0.5));
  rotateWorld(rig.bones.Spine, leanQ);
  rotateWorld(rig.bones.Spine1, new THREE.Quaternion().setFromAxisAngle(side, s.lean * 0.35));
  rotateWorld(rig.bones.Spine2, new THREE.Quaternion().setFromAxisAngle(up, s.yaw * 0.5 + 0.18));

  // 4. eyes: on the bowler before release, on the ball in flight, then follow the ball away.
  let lookAt = L(v(2000, 170, 30));
  if (d && t > -0.15) {
    const tb = Math.max(0, Math.min(t, contactT + 0.25));
    lookAt = new THREE.Vector3(...positionAt(d, Math.min(tb, flightEnd(d))));
  }
  const headFwd = v(0, 0, 1);
  look(rig.bones.Neck, headFwd, lookAt, 0.45, 0.75);
  look(rig.bones.Head, headFwd, lookAt, 1.0, 0.95);

  // 5. bat: handle at the grip, blade along d, face towards the bowler (+X) as far as d allows.
  const dir = s.d.clone().normalize();
  const faceWant = v(1, 0, 0.25);
  const face = faceWant.clone().sub(dir.clone().multiplyScalar(faceWant.dot(dir))).normalize();
  const xAxis = new THREE.Vector3().crossVectors(dir.clone().negate(), face).normalize(); // bat local X
  const m = new THREE.Matrix4().makeBasis(xAxis, dir.clone().negate(), face);
  kit.bat.position.copy(s.g);
  kit.bat.quaternion.setFromRotationMatrix(m);

  // 6. hands on the handle: top (left) hand at the grip, bottom (right) hand just below; elbows fixed out/down.
  const top = s.g.clone();
  const bottom = s.g.clone().add(dir.clone().multiplyScalar(11));
  const lShoulder = rig.root.worldToLocal(wp(rig.bones.LeftArm));
  const rShoulder = rig.root.worldToLocal(wp(rig.bones.RightArm));
  twoBone(rig.bones.LeftArm, rig.bones.LeftForeArm, rig.bones.LeftHand, L(top), L(lShoulder.add(v(38, -26, 22))));
  twoBone(rig.bones.RightArm, rig.bones.RightForeArm, rig.bones.RightHand, L(bottom), L(rShoulder.add(v(-14, -44, 26))));
  // wrap the hands around the handle: fingers point across the handle, towards the facing side
  for (const [hand, grip] of [[rig.bones.LeftHand, top], [rig.bones.RightHand, bottom]] as const) {
    const across = fwd.clone().sub(L(dir).sub(L(v(0, 0, 0))).normalize().multiplyScalar(fwd.dot(L(dir).sub(L(v(0, 0, 0))).normalize()))).normalize();
    const finger = hand.children.find((c) => /Middle1$/.test(c.name)) ?? hand.children[0];
    if (finger) aim(hand, wp(finger), wp(hand).add(across.multiplyScalar(wp(finger).distanceTo(wp(hand)))), 0.85);
    void grip;
  }
}
