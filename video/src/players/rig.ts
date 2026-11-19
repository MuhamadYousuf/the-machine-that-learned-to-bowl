import * as THREE from 'three';

export type Rig = {root: THREE.Group; mixer: THREE.AnimationMixer; clips: THREE.AnimationClip[]; bones: Record<string, THREE.Bone>};

/** Pose a rig at time `t` of the named clip (absolute, so renders are frame-exact). */
export function poseClip(rig: Rig, clipName: string, t: number): void {
  const clip = rig.clips.find((c) => c.name.endsWith(clipName));
  if (!clip) throw new Error(`no clip ${clipName}`);
  const action = rig.mixer.clipAction(clip);
  if (!action.isRunning()) {
    rig.mixer.stopAllAction();
    action.play();
  }
  rig.mixer.setTime(Math.max(0, Math.min(t, clip.duration - 1e-3)));
}
