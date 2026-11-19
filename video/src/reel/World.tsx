import React, {useLayoutEffect, useMemo} from 'react';
import {useThree} from '@react-three/fiber';
import {ThreeCanvas} from '@remotion/three';
import {Bloom, EffectComposer, Vignette} from '@react-three/postprocessing';
import * as THREE from 'three';
import {useVideoConfig} from 'remotion';

/**
 * A bright, toy-like simulation world (low-poly stadium in daylight).
 * Scene axes match the Hawk-Eye clips: X = off side, Y = up, Z = down the pitch.
 * The simulator releases the ball at z = -1.22, so the bowler's popping crease is there
 * and the bowler's stumps sit at z = -2.44. The batter's stumps are at z = 20.12.
 */
export const L = 20.12;
export const BOWL_STUMPS = -2.44;
export const BOWL_CREASE = -1.22;
export const STUMP_H = 0.711;

export const col = {
  sky: '#9fd3ff',
  skyLow: '#e6f3ff',
  grass: '#58b04f',
  grassDark: '#4c9d45',
  sand: '#e2cfa0',
  pitch: '#dcc690',
  crease: '#ffffff',
  stumps: '#f7e7c0',
  bowler: '#1fb5a0',
  bowlerDark: '#127566',
  batter: '#e8eef5',
  batterPad: '#ffffff',
  skin: '#c98e62',
  helmet: '#1d3f8f',
  bat: '#e8c98a',
  keeper: '#f2f2f2',
  ball: '#ff2d3a',
  hud: '#ffd23f',
};

export function rng(seed: number) {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export type Pose = {position: [number, number, number]; target: [number, number, number]; fov: number; up?: [number, number, number]};

const CameraRig: React.FC<{pose: Pose}> = ({pose}) => {
  const {camera, size} = useThree();
  useLayoutEffect(() => {
    const cam = camera as THREE.PerspectiveCamera;
    cam.fov = pose.fov;
    cam.aspect = size.width / size.height;
    const dist = Math.hypot(pose.position[0] - pose.target[0], pose.position[1] - pose.target[1], pose.position[2] - pose.target[2]);
    cam.near = Math.max(0.05, dist * 0.015);
    cam.far = 900;
    cam.position.set(...pose.position);
    cam.up.set(...(pose.up ?? [0, 1, 0]));
    cam.lookAt(...pose.target);
    cam.updateProjectionMatrix();
  });
  return null;
};

export type Look = {sun?: number; warm?: number; dim?: number; red?: number; bloom?: number};

/** The 3D stage: sky, sun, soft shadows, light bloom. */
export const Stage: React.FC<{pose: Pose; look?: Look; children: React.ReactNode}> = ({pose, look = {}, children}) => {
  const {width, height} = useVideoConfig();
  const {sun = 1, warm = 0, dim = 0, red = 0, bloom = 0.35} = look;
  const sky = new THREE.Color(col.sky).lerp(new THREE.Color('#ffb27a'), warm * 0.7).lerp(new THREE.Color('#2a0b12'), red * 0.75).multiplyScalar(1 - dim * 0.7);
  const sunCol = new THREE.Color('#fff4e0').lerp(new THREE.Color('#ffb070'), warm).lerp(new THREE.Color('#ff5a5a'), red * 0.6);
  return (
    <ThreeCanvas
      width={width}
      height={height}
      gl={{antialias: true, toneMapping: THREE.ACESFilmicToneMapping, outputColorSpace: THREE.SRGBColorSpace, preserveDrawingBuffer: true}}
      dpr={1}
      shadows={{type: THREE.PCFSoftShadowMap}}
    >
      <color attach="background" args={[sky]} />
      <fog attach="fog" args={[sky, 140, 420]} />
      <CameraRig pose={pose} />
      <hemisphereLight args={[sky, new THREE.Color('#5d8f45'), 1.05 * sun * (1 - dim * 0.6)]} />
      <directionalLight
        position={[-30 + warm * -20, 55 - warm * 30, -20]}
        intensity={2.4 * sun * (1 - dim * 0.6)}
        color={sunCol}
        castShadow
        shadow-mapSize-width={4096}
        shadow-mapSize-height={4096}
        shadow-camera-left={-40}
        shadow-camera-right={40}
        shadow-camera-top={40}
        shadow-camera-bottom={-40}
        shadow-camera-near={1}
        shadow-camera-far={200}
        shadow-bias={-0.0004}
        shadow-normalBias={0.02}
      />
      {children}
      <EffectComposer multisampling={8}>
        <Bloom intensity={bloom} luminanceThreshold={0.82} luminanceSmoothing={0.2} mipmapBlur radius={0.6} />
        <Vignette offset={0.35} darkness={0.42 + red * 0.3} />
      </EffectComposer>
    </ThreeCanvas>
  );
};

function grassTexture(): THREE.Texture {
  const size = 2048;
  const c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  const ctx = c.getContext('2d')!;
  ctx.fillStyle = col.grass;
  ctx.fillRect(0, 0, size, size);
  const n = 26;
  for (let i = 0; i < n; i++) {
    if (i % 2) continue;
    ctx.fillStyle = col.grassDark;
    ctx.fillRect((i * size) / n, 0, size / n, size);
  }
  const r = rng(3);
  for (let i = 0; i < 40000; i++) {
    ctx.fillStyle = r() < 0.5 ? 'rgba(20,60,20,0.06)' : 'rgba(220,255,200,0.05)';
    ctx.fillRect(r() * size, r() * size, 2, 3);
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

const Flat: React.FC<{x: number; z: number; w: number; d: number; color?: string; y?: number; opacity?: number}> = ({x, z, w, d, color = col.crease, y = 0.006, opacity = 1}) => (
  <mesh position={[x, y, z]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
    <planeGeometry args={[w, d]} />
    <meshStandardMaterial color={color} roughness={0.9} transparent={opacity < 1} opacity={opacity} />
  </mesh>
);

/** Three stumps and two bails. `hit` 0..1 animates a bowled wicket. */
export const Stumps: React.FC<{z: number; hit?: number; dir?: number}> = ({z, hit = 0, dir = 1}) => {
  const xs = [-0.0953, 0, 0.0953];
  const e = Math.min(1, hit * 2.2);
  return (
    <group position={[0, 0, z]}>
      {xs.map((x, i) => (
        <group key={i} position={[x, 0, 0]} rotation={[dir * e * (0.75 - 0.2 * Math.abs(i - 1)), 0, e * 0.35 * (i - 1)]}>
          <mesh position={[0, STUMP_H / 2, 0]} castShadow>
            <cylinderGeometry args={[0.022, 0.022, STUMP_H, 10]} />
            <meshStandardMaterial color={col.stumps} roughness={0.5} />
          </mesh>
        </group>
      ))}
      {[-1, 1].map((s) => {
        const fly = hit;
        return (
          <mesh
            key={s}
            castShadow
            position={[s * 0.05 + s * fly * 0.9, STUMP_H + 0.015 + fly * 2.2 * (1 - fly) * 1.6 + fly * 0.02, dir * fly * (1.4 + s * 0.3)]}
            rotation={[fly * 9 * s, fly * 6, Math.PI / 2 + fly * 4 * s]}
          >
            <cylinderGeometry args={[0.011, 0.011, 0.11, 8]} />
            <meshStandardMaterial color={col.stumps} roughness={0.5} />
          </mesh>
        );
      })}
    </group>
  );
};

/** The playing strip with creases. */
export const Pitch: React.FC<{batterHit?: number; band?: {z0: number; z1: number; color: string; opacity: number}}> = ({batterHit = 0, band}) => (
  <group>
    <mesh position={[0, 0.003, (BOWL_STUMPS + L) / 2]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
      <planeGeometry args={[3.05, L - BOWL_STUMPS + 3]} />
      <meshStandardMaterial color={col.pitch} roughness={1} />
    </mesh>
    {band ? <Flat x={0} z={(band.z0 + band.z1) / 2} w={3.05} d={band.z1 - band.z0} color={band.color} y={0.007} opacity={band.opacity} /> : null}
    {[BOWL_STUMPS, L].map((z) => (
      <Flat key={`b${z}`} x={0} z={z} w={2.64} d={0.05} />
    ))}
    {[BOWL_CREASE, L - 1.22].map((z) => (
      <Flat key={`p${z}`} x={0} z={z} w={3.66} d={0.05} />
    ))}
    {[-1.32, 1.32].flatMap((x) => [
      <Flat key={`r0${x}`} x={x} z={BOWL_STUMPS + 0.61} w={0.05} d={2.44} />,
      <Flat key={`r1${x}`} x={x} z={L - 0.61} w={0.05} d={2.44} />,
    ])}
    <Stumps z={BOWL_STUMPS} dir={-1} />
    <Stumps z={L} hit={batterHit} />
  </group>
);

/** Instanced low-poly crowd in tiered stands, trees beyond, floodlight towers. */
const Stands: React.FC = () => {
  const {seats, crowd, colors} = useMemo(() => {
    const r = rng(11);
    const seatsM: THREE.Matrix4[] = [];
    const crowdM: THREE.Matrix4[] = [];
    const cols: THREE.Color[] = [];
    const palette = ['#ff5a5f', '#ffd23f', '#3ec1d3', '#2a9d8f', '#ffffff', '#f4a261', '#7b61ff', '#1d3f8f', '#e76f51', '#9be564'];
    const segs = 120;
    for (let tier = 0; tier < 7; tier++) {
      const rad = 78 + tier * 2.4;
      const y = 0.6 + tier * 1.4;
      for (let i = 0; i < segs; i++) {
        const a = (i / segs) * Math.PI * 2;
        const m = new THREE.Matrix4().compose(
          new THREE.Vector3(Math.cos(a) * rad, y, L / 2 + Math.sin(a) * rad),
          new THREE.Quaternion().setFromEuler(new THREE.Euler(0, -a + Math.PI / 2, 0)),
          new THREE.Vector3(rad * 2 * Math.PI / segs + 0.05, 1.3, 2.4),
        );
        seatsM.push(m);
        const people = 6;
        for (let k = 0; k < people; k++) {
          if (r() < 0.12) continue;
          const aa = a + ((k + 0.5) / people - 0.5) * (2 * Math.PI / segs);
          const h = 0.7 + r() * 0.25;
          crowdM.push(
            new THREE.Matrix4().compose(
              new THREE.Vector3(Math.cos(aa) * (rad + 0.3), y + 0.65 + h / 2, L / 2 + Math.sin(aa) * (rad + 0.3)),
              new THREE.Quaternion(),
              new THREE.Vector3(0.45, h, 0.45),
            ),
          );
          cols.push(new THREE.Color(palette[Math.floor(r() * palette.length)]));
        }
      }
    }
    return {seats: seatsM, crowd: crowdM, colors: cols};
  }, []);
  const seatRef = React.useRef<THREE.InstancedMesh>(null);
  const crowdRef = React.useRef<THREE.InstancedMesh>(null);
  useLayoutEffect(() => {
    seats.forEach((m, i) => seatRef.current!.setMatrixAt(i, m));
    seatRef.current!.instanceMatrix.needsUpdate = true;
    crowd.forEach((m, i) => {
      crowdRef.current!.setMatrixAt(i, m);
      crowdRef.current!.setColorAt(i, colors[i]);
    });
    crowdRef.current!.instanceMatrix.needsUpdate = true;
    if (crowdRef.current!.instanceColor) crowdRef.current!.instanceColor.needsUpdate = true;
  }, [seats, crowd, colors]);
  return (
    <group>
      <instancedMesh ref={seatRef} args={[undefined, undefined, seats.length]} receiveShadow>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color="#c9d3df" roughness={0.9} />
      </instancedMesh>
      <instancedMesh ref={crowdRef} args={[undefined, undefined, crowd.length]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial roughness={0.8} />
      </instancedMesh>
      {/* roof ring */}
      <mesh position={[0, 15.5, L / 2]} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[88, 1.2, 6, 96]} />
        <meshStandardMaterial color="#e8edf3" roughness={0.6} />
      </mesh>
      {[45, 135, 225, 315].map((deg) => {
        const a = (deg * Math.PI) / 180;
        const x = Math.cos(a) * 92;
        const z = L / 2 + Math.sin(a) * 92;
        return (
          <group key={deg} position={[x, 0, z]} rotation={[0, -a + Math.PI / 2, 0]}>
            <mesh position={[0, 20, 0]} castShadow>
              <cylinderGeometry args={[0.5, 0.9, 40, 8]} />
              <meshStandardMaterial color="#d7dde5" />
            </mesh>
            <mesh position={[0, 41, 0]}>
              <boxGeometry args={[9, 4, 1]} />
              <meshStandardMaterial color="#ffffff" emissive="#fff9e0" emissiveIntensity={0.6} />
            </mesh>
          </group>
        );
      })}
    </group>
  );
};

const Trees: React.FC = () => {
  const items = useMemo(() => {
    const r = rng(5);
    const out: {x: number; z: number; s: number; c: string}[] = [];
    for (let i = 0; i < 110; i++) {
      const a = r() * Math.PI * 2;
      const rad = 100 + r() * 60;
      out.push({x: Math.cos(a) * rad, z: L / 2 + Math.sin(a) * rad, s: 3 + r() * 3.5, c: ['#5fae4e', '#4f9a44', '#77c25f'][Math.floor(r() * 3)]});
    }
    return out;
  }, []);
  return (
    <group>
      {items.map((t, i) => (
        <group key={i} position={[t.x, 0, t.z]}>
          <mesh position={[0, t.s * 0.6, 0]} castShadow>
            <cylinderGeometry args={[t.s * 0.08, t.s * 0.12, t.s * 1.2, 6]} />
            <meshStandardMaterial color="#7a5636" flatShading />
          </mesh>
          <mesh position={[0, t.s * 1.5, 0]} castShadow>
            <icosahedronGeometry args={[t.s * 0.75, 0]} />
            <meshStandardMaterial color={t.c} flatShading />
          </mesh>
        </group>
      ))}
    </group>
  );
};

/** Whole ground: outfield, rope, sightscreens, stands and trees. */
export const Ground: React.FC<{stands?: boolean}> = ({stands = true}) => {
  const tex = useMemo(grassTexture, []);
  return (
    <group>
      <mesh position={[0, -0.05, L / 2]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <circleGeometry args={[600, 64]} />
        <meshStandardMaterial color={col.sand} roughness={1} />
      </mesh>
      <mesh position={[0, 0, L / 2]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <circleGeometry args={[76, 128]} />
        <meshStandardMaterial map={tex} roughness={1} />
      </mesh>
      <mesh position={[0, 0.06, L / 2]} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[66, 0.08, 6, 160]} />
        <meshStandardMaterial color="#ffffff" />
      </mesh>
      {/* 30-yard circle, faint */}
      <mesh position={[0, 0.004, L / 2]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[27.3, 27.45, 128]} />
        <meshBasicMaterial color="#ffffff" transparent opacity={0.45} />
      </mesh>
      {[-1, 1].map((s) => (
        <mesh key={s} position={[0, 3.2, L / 2 + s * 72]} castShadow>
          <boxGeometry args={[14, 6.4, 0.4]} />
          <meshStandardMaterial color="#ffffff" />
        </mesh>
      ))}
      {stands ? <Stands /> : null}
      <Trees />
    </group>
  );
};
