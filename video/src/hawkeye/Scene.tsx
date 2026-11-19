import React, {useLayoutEffect} from 'react';
import {useThree} from '@react-three/fiber';
import {ThreeCanvas} from '@remotion/three';
import {Bloom, EffectComposer, Vignette} from '@react-three/postprocessing';
import * as THREE from 'three';
import {useVideoConfig} from 'remotion';
import {palette} from '../lib/palette';
import {CameraPose} from './cameras';

const CameraRig: React.FC<{pose: CameraPose}> = ({pose}) => {
  const {camera, size} = useThree();
  useLayoutEffect(() => {
    const cam = camera as THREE.PerspectiveCamera;
    cam.fov = pose.fov;
    cam.aspect = size.width / size.height;
    cam.near = 0.05;
    cam.far = 400;
    cam.position.set(...pose.position);
    cam.up.set(...(pose.up ?? [0, 1, 0]));
    cam.lookAt(...pose.target);
    cam.updateProjectionMatrix();
  }, [camera, size, pose.fov, pose.position[0], pose.position[1], pose.position[2], pose.target[0], pose.target[1], pose.target[2], pose.up?.[0], pose.up?.[2]]);
  return null;
};

/** A 3D stage that fills the composition, with a camera pose and soft bloom. */
export const Scene: React.FC<{pose: CameraPose; bloom?: number; width?: number; height?: number; shadows?: boolean; children: React.ReactNode}> = ({
  pose,
  bloom = 1.4,
  shadows = false,
  width: w,
  height: h,
  children,
}) => {
  const config = useVideoConfig();
  const width = w ?? config.width;
  const height = h ?? config.height;
  return (
    <div style={{position: 'relative', width, height, backgroundColor: palette.bg, overflow: 'hidden'}}>
      <ThreeCanvas
        width={width}
        height={height}
        gl={{antialias: true, toneMapping: THREE.ACESFilmicToneMapping, outputColorSpace: THREE.SRGBColorSpace, preserveDrawingBuffer: true}}
        dpr={1}
        shadows={shadows ? {type: THREE.PCFSoftShadowMap} : false}
      >
        <color attach="background" args={[palette.bg]} />
        <fog attach="fog" args={[palette.bg, 45, 110]} />
        <CameraRig pose={pose} />
        {children}
        <EffectComposer multisampling={8}>
          <Bloom intensity={bloom} luminanceThreshold={0.35} luminanceSmoothing={0.3} mipmapBlur radius={0.7} />
          <Vignette offset={0.3} darkness={0.55} />
        </EffectComposer>
      </ThreeCanvas>
    </div>
  );
};
