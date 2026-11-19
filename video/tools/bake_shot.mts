/**
 * Bake one 3D shot for Blender.
 *
 *   npx tsx tools/bake_shot.mts <shot.json>
 *
 * shot.json: {file, overIndex, balls?: number[], slowmo?: {ball, factor}, runup?: number, fps?: 30, out: "blender/shots/<name>"}
 *
 * Writes into <out>/:
 *   batter.glb   the batter (same realistic cricketer, kit, bat) with a baked per-frame animation from the
 *                shared batting solver (src/players/batterPose.ts): exactly what the browser preview shows
 *   shot.json    per frame: ball time, bowler clip time, ball position (Blender coords), stump tilt/bail lift;
 *                plus per-ball events (frames) for the edit
 * Coordinates in shot.json are Blender's: X = -sim y, Y = sim x, Z = sim z.
 */
import fs from 'node:fs';
import path from 'node:path';

// ---- minimal browser shims for three's FBXLoader / GLTFExporter in Node
const g = globalThis as any;
g.self = g;
g.window = g;
g.addEventListener = () => {};
g.removeEventListener = () => {};
g.dispatchEvent = () => true;
g.document = {createElementNS: () => ({style: {}, addEventListener() {}, removeEventListener() {}, setAttribute() {}, getContext: () => null})};
g.FileReader = class {
  result: unknown = null;
  onload: null | (() => void) = null;
  onloadend: null | (() => void) = null;
  readAsArrayBuffer(blob: Blob) {
    blob.arrayBuffer().then((b) => {
      this.result = b;
      this.onload?.();
      this.onloadend?.();
    });
  }
  readAsDataURL(blob: Blob) {
    blob.arrayBuffer().then((b) => {
      this.result = `data:${blob.type};base64,${Buffer.from(b).toString('base64')}`;
      this.onload?.();
      this.onloadend?.();
    });
  }
};

const THREE = await import('three');
const {FBXLoader} = await import('three/examples/jsm/loaders/FBXLoader.js');
const {GLTFExporter} = await import('three/examples/jsm/exporters/GLTFExporter.js');
const SkeletonUtils = await import('three/examples/jsm/utils/SkeletonUtils.js');
THREE.TextureLoader.prototype.load = () => new THREE.Texture();

const {buildKit, poseBatter} = await import('../src/players/batterPose');
const {buildSegments} = await import('../src/players/timeline3d');
const {positionAt, flightEnd, shotPoint, shotTime} = await import('../src/lib/data');
const {pickDeliveries, } = await import('../src/hawkeye/pick');

import {fileURLToPath} from 'node:url';
const VIDEO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const spec = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const fps = spec.fps ?? 30;
const out = path.resolve(VIDEO, spec.out);
fs.mkdirSync(out, {recursive: true});

const data = JSON.parse(fs.readFileSync(path.join(VIDEO, 'public', 'data', spec.file), 'utf8'));
const deliveries = pickDeliveries(data, spec.overIndex, spec.balls ?? [], 1);
const segs = buildSegments(deliveries, {slowmo: spec.slowmo, runup: spec.runup}, fps);
const total = segs.reduce((a, s) => a + s.frames, 0);

// ---- the batter rig, exactly as the preview builds it
const buf = fs.readFileSync(path.join(VIDEO, 'public', 'models', 'bowler', 'CricketBowlingAnimations.fbx'));
const src = new FBXLoader().parse(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), '');
const root = SkeletonUtils.clone(src) as THREE.Group;
root.scale.setScalar(0.01 * 1.073);
const bones: Record<string, THREE.Bone> = {};
root.traverse((o) => {
  if ((o as THREE.Bone).isBone && !bones[o.name]) bones[o.name] = o as THREE.Bone;
  if ((o as THREE.Mesh).isMesh) {
    // plain named materials; Blender re-applies the real textures by mesh name
    (o as THREE.Mesh).material = new THREE.MeshStandardMaterial({name: o.name, color: 0x888888});
  }
});
const rig = {root, mixer: new THREE.AnimationMixer(root), clips: src.animations, bones};
const kit = buildKit(rig as any);
kit.bat.name = 'Bat';
const group = new THREE.Group();
group.name = 'BatterRoot';

// ---- sample every frame
const animated = Object.values(bones);
const times: number[] = [];
const q: Record<string, number[]> = {};
const p: Record<string, number[]> = {};
for (const b of animated) {
  q[b.name] = [];
  p[b.name] = [];
}
const bat = {q: [] as number[], p: [] as number[]};
const frames: unknown[] = [];
for (let f = 0; f < total; f++) {
  let seg = segs[segs.length - 1];
  for (const s of segs) if (f < s.start + s.frames) { seg = s; break; }
  const local = Math.min(f - seg.start, seg.frames - 1) / fps;
  const tau = seg.tau(local);
  poseBatter(rig as any, kit, group, seg.d, tau);
  times.push(f / fps);
  for (const b of animated) {
    q[b.name].push(...b.quaternion.toArray());
    p[b.name].push(...b.position.toArray());
  }
  bat.q.push(...kit.bat.quaternion.toArray());
  bat.p.push(...kit.bat.position.toArray());
  const d = seg.d;
  const end = flightEnd(d);
  let ball: number[] | null = null;
  const toBlender = (p: number[]) => [-p[0], p[2], p[1]]; // scene (X=y, Y=z, Z=x) → Blender (-y, x, z)
  const contact = positionAt(d, end);
  const st = shotTime(d);
  if (tau >= 0 && tau <= end) {
    ball = toBlender(positionAt(d, tau));
  } else if (tau > end && st > 0 && tau <= end + st) {
    ball = toBlender(shotPoint(d, contact, (tau - end) / st)); // struck for runs
  } else if (tau > end && d.result === 'caught_behind' && tau <= end + 0.6) {
    const u = (tau - end) / 0.6; // thin edge flying on towards the slips/keeper
    ball = toBlender([contact[0] + 0.9 * u, contact[1] + 0.35 * u, contact[2] + 14 * u]);
  } else if (tau > end && d.result === 'bowled' && tau <= end + 0.9) {
    // the ball carries on through the stumps, losing most of its pace, and drops to the turf
    const s = tau - end;
    const k = (1 - Math.exp(-s * 3.5)) / 3.5;
    ball = toBlender([contact[0] * (1 - 0.2 * s), Math.max(0.036, contact[1] + 1.2 * s - 4.9 * s * s), contact[2] + 9 * k]);
  } else if (tau > end && tau <= end + 0.08) {
    ball = toBlender(contact);
  }
  let tilt = 0;
  let bail = 0;
  if (d.result === 'bowled' && d.wicket) {
    const since = tau - (d.ts ?? d.t[d.t.length - 1]);
    if (since > 0) {
      const k = Math.min(1, since / 0.5);
      tilt = 0.35 * k;
      bail = since;
    }
  }
  frames.push({f, seg: seg.index, tau: +tau.toFixed(5), bowler: +tau.toFixed(5), ball, tilt, bail: +bail.toFixed(4)});
}

// ---- animation clip → glb
const tracks: THREE.KeyframeTrack[] = [];
for (const b of animated) {
  tracks.push(new THREE.QuaternionKeyframeTrack(`${b.name}.quaternion`, times, q[b.name]));
  tracks.push(new THREE.VectorKeyframeTrack(`${b.name}.position`, times, p[b.name]));
}
tracks.push(new THREE.QuaternionKeyframeTrack('Bat.quaternion', times, bat.q));
tracks.push(new THREE.VectorKeyframeTrack('Bat.position', times, bat.p));
const clip = new THREE.AnimationClip('Bake', times[times.length - 1] ?? 0, tracks);
group.updateMatrixWorld(true);
const exporter = new GLTFExporter();
const glb = await exporter.parseAsync(group, {binary: true, animations: [clip], onlyVisible: false});
fs.writeFileSync(path.join(out, 'batter.glb'), Buffer.from(glb as ArrayBuffer));

const events = segs.map((s) => ({ball: s.index, startFrame: s.start, releaseFrame: s.release, endFrame: s.end, result: s.d.result, runs: s.d.runs, shot: s.d.shot}));
fs.writeFileSync(path.join(out, 'shot.json'), JSON.stringify({fps, frames: total, batterPos: group.position.toArray(), events, perFrame: frames}));
console.log(`baked ${total} frames (${(total / fps).toFixed(1)} s) → ${out}`);
