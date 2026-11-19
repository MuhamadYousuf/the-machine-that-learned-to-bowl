import {PITCH_LENGTH} from './Stadium';

export type Vec3 = [number, number, number];
export type CameraPose = {position: Vec3; target: Vec3; fov: number; trailScale: number; ballScale: number; up?: Vec3};
export type CameraName = 'broadcast' | 'batterEnd' | 'sideOn' | 'sideHalf' | 'topDown' | 'pitchMap' | 'stumpCam' | 'orbit';

const DEG = 180 / Math.PI;

/** Vertical field of view (degrees) that shows `width` metres across at distance `d`. */
function fovForWidth(width: number, d: number, aspect: number): number {
  return 2 * Math.atan(width / aspect / 2 / d) * DEG;
}

/**
 * Camera poses. `aspect` = width / height of the stage, so split screens and
 * both video formats frame correctly. Scene axes: X = off side (+), Y = up,
 * Z = down the pitch (batter's stumps at 20.12).
 *
 * `progress` runs 0 → 1 over a ball's flight (used by following cameras);
 * `orbitT` runs 0 → 1 over an orbit move.
 */
export function cameraPose(name: CameraName, aspect: number, progress = 0, orbitT = 0): CameraPose {
  const L = PITCH_LENGTH;
  const portrait = aspect < 1;
  const p = Math.max(0, Math.min(1, progress));
  const smooth = p * p * (3 - 2 * p);
  switch (name) {
    case 'broadcast':
      return portrait
        ? {position: [0, 9.5, -12], target: [0, 0, L - 7], fov: 30, trailScale: 1.6, ballScale: 2.0}
        : {position: [0, 9, -17], target: [0, 0, L - 3.5], fov: 17, trailScale: 1.6, ballScale: 2.0};
    case 'batterEnd':
      return portrait
        ? {position: [0.9, 1.6, L + 6], target: [0, 0.9, 5], fov: 32, trailScale: 1.3, ballScale: 1.6}
        : {position: [0.9, 1.6, L + 7.5], target: [0, 0.9, 5], fov: 22, trailScale: 1.3, ballScale: 1.6};
    case 'sideOn': {
      // Landscape sees the whole pitch; portrait pans with the ball from release to the batter.
      if (!portrait) return {position: [21, 1.7, L * 0.46], target: [0, 1.0, L * 0.46], fov: 37, trailScale: 1.4, ballScale: 2.0};
      const z = 3 + (L - 4.5) * smooth;
      return {position: [13, 1.4, z], target: [0, 0.9, z], fov: 34, trailScale: 1.1, ballScale: 1.6};
    }
    case 'sideHalf': {
      // Side view of the batter's half only: bounce points and heights read clearly.
      const zc = L - 6.5;
      const d = 13;
      const width = portrait ? 9 : 15.5;
      return {position: [d, 1.1, zc], target: [0, 1.0, zc], fov: fovForWidth(width, d, aspect), trailScale: 1.2, ballScale: 1.7};
    }
    case 'topDown':
      return portrait
        ? {position: [0, 22, L - 6.5 + 0.01], target: [0, 0, L - 6.5], fov: 38, trailScale: 1.1, ballScale: 1.5}
        : {position: [0, 30, L * 0.5 + 0.01], target: [0, 0, L * 0.5], fov: 17, trailScale: 1.4, ballScale: 2.0};
    case 'pitchMap': {
      // Straight down on the batter's half. Portrait: bowler at the top. Landscape: pitch runs left → right.
      const zc = L - 6.8;
      const d = 24;
      if (portrait) {
        return {position: [0, d, zc], target: [0, 0, zc], up: [0, 0, -1], fov: 2 * Math.atan(15 / 2 / d) * DEG, trailScale: 1.2, ballScale: 1.6};
      }
      return {position: [0, d, zc], target: [0, 0, zc], up: [1, 0, 0], fov: fovForWidth(15.5, d, aspect), trailScale: 1.2, ballScale: 1.6};
    }
    case 'stumpCam':
      return portrait
        ? {position: [0.1, 0.34, L + 2.4], target: [0, 0.6, 7], fov: 40, trailScale: 0.9, ballScale: 1.2}
        : {position: [0.45, 0.32, L + 2.2], target: [0, 0.55, 7], fov: 26, trailScale: 0.9, ballScale: 1.2};
    case 'orbit': {
      const a = -Math.PI * 0.85 + orbitT * Math.PI * 0.7;
      const radius = portrait ? 27 : 22;
      return {
        position: [Math.sin(a) * radius, 6 + 4 * Math.sin(orbitT * Math.PI), L / 2 + Math.cos(a) * radius],
        target: [0, 0.6, L / 2],
        fov: portrait ? 46 : 32,
        trailScale: 2.0,
        ballScale: 2.4,
      };
    }
  }
}
