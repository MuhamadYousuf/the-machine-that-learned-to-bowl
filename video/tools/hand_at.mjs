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
for (let t = 6.44; t <= 6.54; t += 0.01) {
  mixer.setTime(t); obj.updateMatrixWorld(true);
  const h = bones.RightHand.getWorldPosition(new THREE.Vector3());
  const sh = bones.RightArm.getWorldPosition(new THREE.Vector3());
  const head = bones.Head.getWorldPosition(new THREE.Vector3());
  console.log(`t ${t.toFixed(2)} hand ${h.toArray().map((x) => x.toFixed(0)).join(',')}  hand-head z ${(h.z - head.z).toFixed(0)}  above shoulder ${(h.y - sh.y).toFixed(0)}`);
}
