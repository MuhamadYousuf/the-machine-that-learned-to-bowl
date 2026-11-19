// Inspect an FBX: animation clips (name, length, tracks) and skeleton bones. Textures are skipped.
import fs from 'node:fs';
globalThis.self = globalThis;
globalThis.window = globalThis;
globalThis.document = {createElementNS: () => ({style: {}, addEventListener() {}, removeEventListener() {}, setAttribute() {}})};
const THREE = await import('three');
const {FBXLoader} = await import('three/examples/jsm/loaders/FBXLoader.js');
THREE.TextureLoader.prototype.load = () => new THREE.Texture();
for (const file of process.argv.slice(2)) {
  const buf = fs.readFileSync(file);
  const obj = new FBXLoader().parse(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), '');
  const bones = [], meshes = [];
  obj.traverse((o) => { if (o.isBone) bones.push(o.name); if (o.isMesh) meshes.push(`${o.name}${o.isSkinnedMesh ? ' (skinned)' : ''}`); });
  const box = new THREE.Box3().setFromObject(obj);
  console.log(`\n== ${file.split('/').pop()}`);
  console.log(`size ${box.getSize(new THREE.Vector3()).toArray().map((v) => v.toFixed(1)).join(' x ')}, meshes: ${meshes.length} [${meshes.slice(0, 12).join(', ')}]`);
  console.log(`bones (${bones.length}): ${bones.slice(0, 70).join(', ')}`);
  for (const c of obj.animations) console.log(`clip "${c.name}" ${c.duration.toFixed(2)}s, ${c.tracks.length} tracks`);
}
