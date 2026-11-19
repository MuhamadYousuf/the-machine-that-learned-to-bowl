import React, {useMemo} from 'react';
import * as THREE from 'three';

/**
 * Floodlight-style lighting for the 3D players: a shadow-casting key light from high behind the bowler's
 * off side, a cool rim light from the batter's end, and an invisible shadow catcher over the field so the
 * players' shadows land on the (unlit) grass and pitch.
 */
export const PlayerLights: React.FC<{focusZ?: number; intensity?: number}> = ({focusZ = 9, intensity = 1}) => {
  const key = useMemo(() => {
    const l = new THREE.DirectionalLight('#fff4e0', 2.1 * intensity);
    l.position.set(9, 26, focusZ - 16);
    l.target.position.set(0, 0, focusZ);
    l.castShadow = true;
    l.shadow.mapSize.set(4096, 4096);
    const c = l.shadow.camera;
    c.left = -18;
    c.right = 18;
    c.top = 26;
    c.bottom = -26;
    c.near = 1;
    c.far = 90;
    l.shadow.bias = -0.0004;
    l.shadow.normalBias = 0.02;
    return l;
  }, [focusZ, intensity]);
  return (
    <>
      <primitive object={key} />
      <primitive object={key.target} />
      <directionalLight position={[-7, 9, focusZ + 22]} intensity={1.1 * intensity} color="#b9d4ff" />
      <hemisphereLight args={['#dfe8ff', '#1f2a1c', 0.55 * intensity]} />
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.012, focusZ]} receiveShadow>
        <planeGeometry args={[70, 90]} />
        <shadowMaterial transparent opacity={0.42} />
      </mesh>
    </>
  );
};
