import React, {useMemo} from 'react';
import {AbsoluteFill, useCurrentFrame, useVideoConfig} from 'remotion';
import {useJsonFiles} from '../lib/data';
import {ease, palette} from '../lib/palette';
import {HOLD} from '../hawkeye/hold';
import {logGeneration, NetFile, netAtGeneration, stableEdges} from './net';
import {NetworkView} from './NetworkView';

export type BrainMorphProps = {run: string; mode: 'morph' | 'milestones'; seconds: number};

const MILESTONES = [1, 100, 500, 1500];

/** Generation shown at progress p for each mode. */
function generationAt(mode: BrainMorphProps['mode'], p: number): number {
  if (mode === 'morph') return logGeneration(ease.inOutSine(p));
  // Milestones: hold on each, with eased transitions between them.
  const n = MILESTONES.length;
  const segment = Math.min(n - 1, Math.floor(p * n));
  const local = p * n - segment;
  const from = MILESTONES[segment];
  const to = MILESTONES[Math.min(n - 1, segment + 1)];
  const k = segment === n - 1 ? 0 : ease.inOutCubic(Math.max(0, (local - 0.7) / 0.3));
  return Math.exp(Math.log(from) + (Math.log(to) - Math.log(from)) * k);
}

/** A1: the bowler's brain rewiring itself from generation 1 to 1500. */
export const BrainMorph: React.FC<BrainMorphProps> = ({run, mode}) => {
  const frame = useCurrentFrame();
  const {width, height, durationInFrames} = useVideoConfig();
  const files = useJsonFiles<NetFile>([`brain_net_${run}.json`]);
  const file = files?.[0];
  const edges = useMemo(() => {
    if (!file) return [];
    const keyNets = [1, 100, 500, 1500].map((g) => netAtGeneration(file, g));
    return stableEdges(keyNets, file.sizes, [26, 30, 14]);
  }, [file]);
  if (!file) return <AbsoluteFill />;
  const p = Math.max(0, Math.min(1, (frame - HOLD) / (durationInFrames - 2 * HOLD)));
  const net = netAtGeneration(file, generationAt(mode, p));
  const vertical = height > width;
  const nodeValues = [new Array(file.sizes[0]).fill(0.15), ...net.b.map((b) => b.map((v) => Math.tanh(v * 2)))];
  return (
    <AbsoluteFill style={{backgroundColor: palette.bg}}>
      <NetworkView
        net={net}
        sizes={file.sizes}
        edges={edges}
        width={width}
        height={height}
        vertical={vertical}
        nodeValues={nodeValues}
        edgeScale={2.4}
        nodeRadius={vertical ? 16 : 15}
        margin={vertical ? 110 : 120}
      />
    </AbsoluteFill>
  );
};
