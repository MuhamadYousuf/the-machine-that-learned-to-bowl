import React, {useMemo} from 'react';
import * as THREE from 'three';
import {palette} from '../lib/palette';

export const PITCH_LENGTH = 20.12;
export const POPPING = 1.22;
const STUMP_HALF_WIDTH = 0.2286 / 2;
const STUMP_RADIUS = 0.019;
const STUMP_HEIGHT = 0.711;

/** Deterministic pseudo-random generator so every render gets identical textures. */
function mulberry32(seed: number) {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pitchTexture(): THREE.Texture {
  const w = 512;
  const h = 4096;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d')!;
  const grad = ctx.createLinearGradient(0, 0, w, 0);
  grad.addColorStop(0, '#b29c6c');
  grad.addColorStop(0.5, palette.pitch);
  grad.addColorStop(1, '#b29c6c');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);
  const rand = mulberry32(7);
  // Fine grain and a few worn patches near both ends.
  for (let i = 0; i < 60000; i++) {
    const x = rand() * w;
    const y = rand() * h;
    const v = rand();
    ctx.fillStyle = v < 0.5 ? `rgba(90,70,40,${0.05 + rand() * 0.07})` : `rgba(255,245,220,${0.04 + rand() * 0.06})`;
    ctx.fillRect(x, y, 1 + rand() * 2, 1 + rand() * 3);
  }
  for (const centre of [0.1, 0.9]) {
    for (let i = 0; i < 40; i++) {
      const x = w * (0.3 + rand() * 0.4);
      const y = h * (centre + (rand() - 0.5) * 0.08);
      const r = 10 + rand() * 40;
      const g = ctx.createRadialGradient(x, y, 0, x, y, r);
      g.addColorStop(0, 'rgba(120,95,55,0.18)');
      g.addColorStop(1, 'rgba(120,95,55,0)');
      ctx.fillStyle = g;
      ctx.fillRect(x - r, y - r, 2 * r, 2 * r);
    }
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  return texture;
}

function outfieldTexture(): THREE.Texture {
  const size = 2048;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, '#15442e');
  g.addColorStop(0.35, '#0f3524');
  g.addColorStop(0.75, '#0a2016');
  g.addColorStop(1, palette.bg);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  // Mowing stripes, very subtle.
  // Mowing stripes run along the pitch, very subtle.
  ctx.globalAlpha = 0.035;
  for (let i = 0; i < 64; i++) {
    ctx.fillStyle = i % 2 ? '#ffffff' : '#000000';
    ctx.fillRect((i * size) / 64, 0, size / 64, size);
  }
  ctx.globalAlpha = 1;
  const mask = ctx.createRadialGradient(size / 2, size / 2, size * 0.3, size / 2, size / 2, size / 2);
  mask.addColorStop(0, 'rgba(6,10,19,0)');
  mask.addColorStop(1, 'rgba(6,10,19,1)');
  ctx.fillStyle = mask;
  ctx.fillRect(0, 0, size, size);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

const Line: React.FC<{x: number; z: number; w: number; d: number; opacity?: number}> = ({x, z, w, d, opacity = 0.95}) => (
  <mesh position={[x, 0.004, z]} rotation={[-Math.PI / 2, 0, 0]}>
    <planeGeometry args={[w, d]} />
    <meshBasicMaterial color={palette.crease} transparent opacity={opacity} toneMapped={false} />
  </mesh>
);

export const Stumps: React.FC<{z: number; tilt?: number; bailLift?: number}> = ({z, tilt = 0, bailLift = 0}) => {
  const xs = [-STUMP_HALF_WIDTH + STUMP_RADIUS, 0, STUMP_HALF_WIDTH - STUMP_RADIUS];
  return (
    <group position={[0, 0, z]}>
      {xs.map((x, i) => (
        <group key={i} position={[x, 0, 0]} rotation={[tilt * (1 - 0.25 * Math.abs(i - 1)), 0, tilt * 0.3 * (i - 1)]}>
          <mesh position={[0, STUMP_HEIGHT / 2, 0]}>
            <cylinderGeometry args={[STUMP_RADIUS, STUMP_RADIUS, STUMP_HEIGHT, 16]} />
            <meshStandardMaterial color={palette.stumps} roughness={0.5} />
          </mesh>
        </group>
      ))}
      {[-1, 1].map((side) => (
        <mesh
          key={side}
          position={[side * 0.057, STUMP_HEIGHT + 0.012 + bailLift * (1 + 0.3 * side), side * bailLift * 0.4]}
          rotation={[bailLift * 3 * side, bailLift * 5, Math.PI / 2 + bailLift * 2 * side]}
        >
          <cylinderGeometry args={[0.008, 0.008, 0.11, 8]} />
          <meshStandardMaterial color={palette.stumps} roughness={0.5} />
        </mesh>
      ))}
    </group>
  );
};

export const Stadium: React.FC<{goodLengthBand?: boolean; wicketZone?: number; batterStumpsTilt?: number; bailLift?: number}> = ({
  goodLengthBand = false,
  wicketZone = 0,
  batterStumpsTilt = 0,
  bailLift = 0,
}) => {
  const pitchTex = useMemo(pitchTexture, []);
  const fieldTex = useMemo(outfieldTexture, []);
  const mid = PITCH_LENGTH / 2;
  return (
    <group>
      <ambientLight intensity={0.45} />
      <directionalLight position={[-6, 18, -8]} intensity={1.25} />
      <directionalLight position={[8, 10, 30]} intensity={0.5} />
      <mesh position={[0, -0.01, mid]} rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[75, 128]} />
        <meshBasicMaterial map={fieldTex} toneMapped={false} />
      </mesh>
      <mesh position={[0, 0, mid]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[3.05, PITCH_LENGTH + 2 * 2.0]} />
        <meshStandardMaterial map={pitchTex} roughness={0.95} />
      </mesh>
      {goodLengthBand ? (
        <mesh position={[0, 0.003, PITCH_LENGTH - 7]} rotation={[-Math.PI / 2, 0, 0]}>
          <planeGeometry args={[3.05, 2]} />
          <meshBasicMaterial color={palette.teal} transparent opacity={0.13} toneMapped={false} />
        </mesh>
      ) : null}
      {[0, PITCH_LENGTH].map((z) => (
        <Line key={`b${z}`} x={0} z={z} w={2.64} d={0.05} />
      ))}
      {[POPPING, PITCH_LENGTH - POPPING].map((z) => (
        <Line key={`p${z}`} x={0} z={z - (z < mid ? 0 : 0)} w={3.66} d={0.05} />
      ))}
      {[-1.32, 1.32].flatMap((x) => [
        <Line key={`r0${x}`} x={x} z={(POPPING - 1.22) / 2} w={0.05} d={POPPING + 1.22} />,
        <Line key={`r1${x}`} x={x} z={PITCH_LENGTH - (POPPING - 1.22) / 2} w={0.05} d={POPPING + 1.22} />,
      ])}
      <Stumps z={0} />
      <Stumps z={PITCH_LENGTH} tilt={batterStumpsTilt} bailLift={bailLift} />
      {wicketZone > 0 ? (
        <mesh position={[0, STUMP_HEIGHT / 2 + 0.02, PITCH_LENGTH]}>
          <boxGeometry args={[STUMP_HALF_WIDTH * 2 + 0.072, STUMP_HEIGHT + 0.04, 0.02]} />
          <meshBasicMaterial color={palette.wicket} transparent opacity={0.18 * wicketZone} toneMapped={false} />
        </mesh>
      ) : null}
    </group>
  );
};
