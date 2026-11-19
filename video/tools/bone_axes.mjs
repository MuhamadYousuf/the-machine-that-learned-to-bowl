import fs from 'node:fs';
globalThis.self = globalThis; globalThis.window = globalThis;
globalThis.document = {createElementNS: () => ({style: {}, addEventListener() {}, removeEventListener() {}, setAttribute() {}})};
const THREE = await import('three');
const {FBXLoader} = await import('three/examples/jsm/loaders/FBXLoader.js');
THREE.TextureLoader.prototype.load = () => new THREE.Texture();
const b = fs.readFileSync('public/models/bowler/CricketBowlingAnimations.fbx');
const obj = new FBXLoader().parse(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength), '');
const mixer = new THREE.AnimationMixer(obj);
mixer.clipAction(obj.animations.find((c) => c.name.endsWith('RightArmFastBowler'))).play();
mixer.setTime(Number(process.argv[2] ?? 0)); obj.updateMatrixWorld(true);
const bones = {}; obj.traverse((o) => { if (o.isBone && !bones[o.name]) bones[o.name] = o; });
const f = (v) => v.toArray().map((x) => x.toFixed(2)).join(',');
for (const n of ['Hips','Spine','Spine2','Neck','Head','LeftShoulder','LeftArm','LeftForeArm','LeftHand','RightArm','RightForeArm','RightHand','LeftUpLeg','LeftLeg','LeftFoot','RightUpLeg','RightLeg','RightFoot']) {
  const bn = bones[n]; const p = bn.getWorldPosition(new THREE.Vector3());
  const q = bn.getWorldQuaternion(new THREE.Quaternion());
  const ax = (x,y,z) => new THREE.Vector3(x,y,z).applyQuaternion(q);
  const child = bn.children.find((c) => c.isBone);
  const len = child ? child.getWorldPosition(new THREE.Vector3()).distanceTo(p) : 0;
  const dir = child ? child.getWorldPosition(new THREE.Vector3()).sub(p).normalize() : new THREE.Vector3();
  console.log(`${n.padEnd(13)} pos ${f(p)}  len ${len.toFixed(1)}  toChild ${f(dir)}  localY ${f(ax(0,1,0))}  localZ ${f(ax(0,0,1))}`);
}
