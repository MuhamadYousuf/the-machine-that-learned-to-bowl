import React, {useMemo, useState} from 'react';
import {continueRender, delayRender, staticFile} from 'remotion';
import {useThree} from '@react-three/fiber';
import * as THREE from 'three';
import {FBXLoader} from 'three/examples/jsm/loaders/FBXLoader.js';
import * as SkeletonUtils from 'three/examples/jsm/utils/SkeletonUtils.js';

/**
 * The realistic cricketer from the "Cricket Bowling Animations Pack" (Unity Asset Store, HK / Hamza Khan).
 * Licensed for use in this video; the files live in public/models/ and are git-ignored (not redistributable).
 *
 * The FBX holds the skinned character plus 8 bowling clips (root motion along +Z, centimetres).
 * Its embedded texture paths point at the author's machine, so materials are rebuilt from the pack's PNGs.
 */
export const MODEL = 'models/bowler/CricketBowlingAnimations.fbx';
export const SCALE = 0.01 * 1.073; // cm → m, and 1.07× so the release hand sits at the sim's 2.20 m

export type Kit = {top: string; bottom: string; topTexture: boolean};
export const KIT_GREEN: Kit = {top: '#ffffff', bottom: '#007300', topTexture: true}; // the pack's own kit
export const KIT_BLUE: Kit = {top: '#1e4fa8', bottom: '#1b3f86', topTexture: false};

const BLANK = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=';
/** The FBX's embedded texture paths point at the author's machine; answer them with a blank pixel (real maps are applied after). */
function quietManager(): THREE.LoadingManager {
  const m = new THREE.LoadingManager();
  m.setURLModifier((url) => (/\.(jpe?g|png|tga)$/i.test(url) ? BLANK : url));
  return m;
}

type Asset = {scene: THREE.Group; clips: THREE.AnimationClip[]; textures: Record<string, THREE.Texture>};
let assetPromise: Promise<Asset> | null = null;

function loadTexture(name: string): Promise<THREE.Texture> {
  return new Promise((resolve, reject) =>
    new THREE.TextureLoader().load(staticFile(`models/bowler/${name}`), (t) => {
      t.colorSpace = name.includes('Normal') ? THREE.NoColorSpace : THREE.SRGBColorSpace;
      t.flipY = true;
      resolve(t);
    }, undefined, reject),
  );
}

export function loadCricketer(): Promise<Asset> {
  if (!assetPromise) {
    const names = ['BodyDiffuse.png', 'SkinDiffuse.png', 'OutfitTopDifffuse.png', 'FootWearDiffuse.png', 'HairAndBeardDiffuse.png',
      'EyeDiffuse.png', 'TeethDiffuse.png', 'OutfitTopNormal.png', 'OutfitBottomNormal.png', 'FootWearNormal.png', 'BodyNormal.png', 'HairNormal.png'];
    assetPromise = Promise.all([
      new FBXLoader(quietManager()).loadAsync(staticFile(MODEL)),
      ...names.map(loadTexture),
    ]).then(([scene, ...tex]) => {
      const textures: Record<string, THREE.Texture> = {};
      names.forEach((n, i) => (textures[n] = tex[i] as THREE.Texture));
      return {scene: scene as THREE.Group, clips: (scene as THREE.Group).animations, textures};
    });
  }
  return assetPromise;
}

function materialFor(mesh: string, tex: Record<string, THREE.Texture>, kit: Kit): THREE.Material {
  const std = (o: THREE.MeshStandardMaterialParameters) => new THREE.MeshStandardMaterial({roughness: 0.7, metalness: 0.0, ...o});
  if (mesh.startsWith('Body')) return std({map: tex['SkinDiffuse.png'], normalMap: tex['BodyNormal.png'], roughness: 0.6});
  if (mesh.startsWith('Head')) return std({map: tex['SkinDiffuse.png'], roughness: 0.55});
  if (mesh.startsWith('OutfitTop') || mesh.startsWith('OufitTop'))
    return std(kit.topTexture ? {map: tex['OutfitTopDifffuse.png'], normalMap: tex['OutfitTopNormal.png']} : {color: kit.top, normalMap: tex['OutfitTopNormal.png']});
  if (mesh.startsWith('OutfitBottom') || mesh.startsWith('OufitBottom')) return std({color: kit.bottom, normalMap: tex['OutfitBottomNormal.png']});
  if (mesh.startsWith('FootWear')) return std({map: tex['FootWearDiffuse.png'], normalMap: tex['FootWearNormal.png']});
  if (mesh.startsWith('Hair') || mesh.startsWith('Beard')) return std({color: '#141414', normalMap: tex['HairNormal.png'], roughness: 0.9});
  if (mesh.startsWith('Eye')) return std({map: tex['EyeDiffuse.png'], roughness: 0.2});
  if (mesh.startsWith('Teeth')) return std({map: tex['TeethDiffuse.png']});
  return std({color: '#888888'});
}

import type {Rig} from './rig';
export type {Rig} from './rig';

/** A fresh, independently animatable copy of the cricketer with a kit applied. */
export function useCricketer(kit: Kit): Rig | null {
  const advance = useThree((st) => st.advance);
  const [asset, setAsset] = useState<Asset | null>(null);
  const [handle] = useState(() => delayRender('load cricketer'));
  React.useEffect(() => {
    loadCricketer().then(setAsset);
  }, []);
  const rig = useMemo(() => {
    if (!asset) return null;
    const root = SkeletonUtils.clone(asset.scene) as THREE.Group;
    const bones: Record<string, THREE.Bone> = {};
    root.traverse((o) => {
      if ((o as THREE.Mesh).isMesh) {
        const m = o as THREE.SkinnedMesh;
        m.material = materialFor(m.name, asset.textures, kit);
        m.frustumCulled = false;
        m.castShadow = true;
      }
      if ((o as THREE.Bone).isBone && !bones[o.name]) bones[o.name] = o as THREE.Bone;
    });
    root.scale.setScalar(SCALE);
    return {root, mixer: new THREE.AnimationMixer(root), clips: asset.clips, bones};
  }, [asset, kit]);
  // Release the frame only after the rig is in the scene and the canvas has drawn it (two animation frames).
  React.useEffect(() => {
    if (!rig) return;
    let a = 0;
    let b = 0;
    a = requestAnimationFrame(() => {
      advance(performance.now()); // the canvas only redraws on frame changes; draw the new rig now
      b = requestAnimationFrame(() => continueRender(handle));
    });
    return () => {
      cancelAnimationFrame(a);
      cancelAnimationFrame(b);
    };
  }, [rig, handle, advance]);
  return rig;
}

export {poseClip} from './rig';
