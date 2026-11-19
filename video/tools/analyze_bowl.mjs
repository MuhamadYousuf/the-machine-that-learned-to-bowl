import fs from 'node:fs';
globalThis.self = globalThis; globalThis.window = globalThis;
globalThis.document = {createElementNS: () => ({style: {}, addEventListener() {}, removeEventListener() {}, setAttribute() {}})};
const THREE = await import('three');
const {FBXLoader} = await import('three/examples/jsm/loaders/FBXLoader.js');
THREE.TextureLoader.prototype.load = () => new THREE.Texture();
const b = fs.readFileSync('public/models/bowler/CricketBowlingAnimations.fbx');
const obj = new FBXLoader().parse(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength), '');
const clipName = process.argv[2] || 'Armature|RightArmFastBowler';
const clip = obj.animations.find((c) => c.name === clipName);
const mixer = new THREE.AnimationMixer(obj);
mixer.clipAction(clip).play();
const find = (n) => { let r; obj.traverse((o) => { if (o.isBone && o.name === n && !r) r = o; }); return r; };
const hand = find('RightHand'), hips = find('Hips'), head = find('Head'), lfoot = find('LeftFoot');
const v = new THREE.Vector3();
const rows = [];
for (let t = 0; t <= clip.duration; t += 1 / 30) {
  mixer.setTime(t); obj.updateMatrixWorld(true);
  const h = hand.getWorldPosition(new THREE.Vector3()), p = hips.getWorldPosition(new THREE.Vector3()), hd = head.getWorldPosition(new THREE.Vector3()), lf = lfoot.getWorldPosition(new THREE.Vector3());
  rows.push({t, h, p, hd, lf});
}
const top = rows.reduce((a, r) => (r.h.y > a.h.y ? r : a));
const fmt = (q) => q.toArray().map((x) => x.toFixed(0)).join(',');
for (const r of rows.filter((_, i) => i % 6 === 0)) console.log(`t ${r.t.toFixed(2)} hand ${fmt(r.h)} hips ${fmt(r.p)} head ${fmt(r.hd)} lfoot ${fmt(r.lf)}`);
console.log(`highest hand at t=${top.t.toFixed(3)}: hand ${fmt(top.h)} hips ${fmt(top.p)} head ${fmt(top.hd)}`);
// hand speed peak (release is when the arm is fastest, near the top)
let best = {s: 0};
for (let i = 1; i < rows.length; i++) { const s = rows[i].h.distanceTo(rows[i - 1].h) * 30; if (s > best.s) best = {s, t: rows[i].t, h: rows[i].h}; }
console.log(`fastest hand at t=${best.t.toFixed(3)}: ${best.s.toFixed(0)} cm/s at ${fmt(best.h)}`);
