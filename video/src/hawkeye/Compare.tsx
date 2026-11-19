import React from 'react';
import {AbsoluteFill, useCurrentFrame, useVideoConfig} from 'remotion';
import {OversFile, useJsonFiles} from '../lib/data';
import {palette} from '../lib/palette';
import {CameraName} from './cameras';
import {HOLD} from './hold';
import {OverStage, pickDeliveries} from './OverReplay';

export type CompareProps = {
  left: {file: string; overIndex: number};
  right: {file: string; overIndex: number};
  overCount: number;
  camera: CameraName;
  speed: number;
  goodLengthBand: boolean;
};

/**
 * Split screen: left/top = first source (v1, the loophole run), right/bottom = second (v2).
 * Both play the same number of balls in lock-step so the contrast is visible ball by ball.
 */
export const Compare: React.FC<CompareProps> = (props) => {
  const frame = useCurrentFrame();
  const {width, height} = useVideoConfig();
  const files = useJsonFiles<OversFile>([props.left.file, props.right.file]);
  if (!files) return <AbsoluteFill />;
  const portrait = height > width;
  const gap = 6;
  const w = portrait ? width : (width - gap) / 2;
  const h = portrait ? (height - gap) / 2 : height;
  const sides = [props.left, props.right].map((side, i) =>
    pickDeliveries(files[i], side.overIndex, [], props.overCount),
  );
  return (
    <AbsoluteFill style={{backgroundColor: palette.bg, display: 'flex', flexDirection: portrait ? 'column' : 'row', gap}}>
      {sides.map((deliveries, i) => (
        <OverStage
          key={i}
          deliveries={deliveries}
          f={frame - HOLD}
          camera={props.camera}
          speed={props.speed}
          accumulate
          goodLengthBand={props.goodLengthBand}
          width={Math.round(w)}
          height={Math.round(h)}
        />
      ))}
    </AbsoluteFill>
  );
};
