import fs from 'node:fs';
globalThis.self = globalThis; globalThis.window = globalThis;
globalThis.document = {createElementNS: () => ({style: {}, addEventListener() {}, removeEventListener() {}, setAttribute() {}})};
const THREE = await import('three');
const {FBXLoader} = await import('three/examples/jsm/loaders/FBXLoader.js');
THREE.TextureLoader.prototype.load = () => new THREE.Texture();
const b = fs.readFileSync('public/models/bowler/CricketBowlingAnimations.fbx');
const obj = new FBXLoader().parse(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength), '');
const clip = obj.animations.find((c) => c.name.endsWith('RightArmFastBowler'));
const mixer = new THREE.AnimationMixer(obj); mixer.clipAction(clip).play();
const bones = {}; obj.traverse((o) => { if (o.isBone && !bones[o.name]) bones[o.name] = o; });
let prev = null;
for (let t = 6.54; t <= 6.6; t += 1 / 480) {
  mixer.setTime(t); obj.updateMatrixWorld(true);
  const sh = bones.RightArm.getWorldPosition(new THREE.Vector3());
  const h = bones.RightHand.getWorldPosition(new THREE.Vector3());
  const arm = h.clone().sub(sh); const ang = Math.sign(arm.z) * arm.angleTo(new THREE.Vector3(0,1,0)) * 180 / Math.PI; // 0 = straight up, + = towards the batter
  const sp = prev ? h.distanceTo(prev) * 480 / 100 : 0; const hp = h;
  prev = h;
  console.log(`hand ${h.x.toFixed(0)},${h.y.toFixed(0)},${h.z.toFixed(0)} sh ${sh.y.toFixed(0)} t ${t.toFixed(4)}  arm ${ang.toFixed(0).padStart(5)}°  hand height ${(h.y / 100).toFixed(2)} m  speed ${sp.toFixed(1)} m/s`);
}
