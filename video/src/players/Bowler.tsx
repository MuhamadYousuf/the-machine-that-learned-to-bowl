import React from 'react';
import {KIT_GREEN, poseClip, SCALE, useCricketer} from './Character';

/** Right-arm fast bowler clip: highest point of the bowling hand (the release), measured with tools/analyze_bowl.mjs. */
export const BOWL_CLIP = 'RightArmFastBowler';
export const RELEASE_CLIP_T = 6.5; // arm whipping over the top (not the earlier hand-highest point, where the arm is still loading)
const HAND_AT_RELEASE_CM = [-37, 184, 3935]; // clip-local position of RightHand at release
const SIM_RELEASE = {X: 0.3, Y: 2.2, Z: -1.22}; // scene coords of the simulator's release point (y off side, z up, x along)

/**
 * The bowler, placed so his hand is exactly at the simulator's release point at the moment of release.
 * `sinceRelease` is seconds relative to release (negative = run-up, positive = follow-through).
 */
export const Bowler: React.FC<{sinceRelease: number; visible?: boolean}> = ({sinceRelease, visible = true}) => {
  const rig = useCricketer(KIT_GREEN);
  if (!rig) return null;
  poseClip(rig, BOWL_CLIP, RELEASE_CLIP_T + sinceRelease);
  const offset: [number, number, number] = [
    SIM_RELEASE.X - HAND_AT_RELEASE_CM[0] * SCALE,
    0,
    SIM_RELEASE.Z - HAND_AT_RELEASE_CM[2] * SCALE,
  ];
  return (
    <group position={offset} visible={visible}>
      <primitive object={rig.root} />
    </group>
  );
};
