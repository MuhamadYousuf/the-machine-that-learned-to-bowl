import React, {useMemo} from 'react';
import * as THREE from 'three';
import {Delivery} from '../lib/data';
import {Kit, KIT_BLUE, useCricketer} from './Character';
import {buildKit, poseBatter} from './batterPose';

export {BATTER_POS} from './batterPose';

/** The batter, posed by the shared solver in batterPose.ts (also used to bake Blender animation). */
export const Batter: React.FC<{d?: Delivery; t: number; kit?: Kit}> = ({d, t, kit: kitColours = KIT_BLUE}) => {
  const rig = useCricketer(kitColours);
  const kit = useMemo(() => (rig ? buildKit(rig) : null), [rig]);
  const group = useMemo(() => new THREE.Group(), []);
  if (!rig || !kit) return null;
  poseBatter(rig, kit, group, d, t);
  return <primitive object={group} />;
};
