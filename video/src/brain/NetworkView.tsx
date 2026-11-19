import React from 'react';
import {palette} from '../lib/palette';
import {Edge, weight} from './net';

export type NetworkViewProps = {
  net: {w: number[][]; b: number[][]};
  sizes: number[];
  edges: Edge[];
  width: number;
  height: number;
  vertical: boolean;
  /** Activations per layer [inputs, h1, h2, outputs]; when given, edges glow by |w × source activation|. */
  activations?: number[][];
  /** Node colours without driving edges (e.g. tanh(bias) during a morph). */
  nodeValues?: number[][];
  /** Normalisation for edge strength (|w| or |w·a| at which an edge is drawn at full strength). */
  edgeScale?: number;
  /** Extra emphasis for layer → layer flow during brain-cam pulses: 0..1 per layer of edges. */
  layerPulse?: number[];
  nodeRadius?: number;
  margin?: number;
};

const mixHex = (a: string, b: string, t: number) => {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16));
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16));
  const m = pa.map((v, i) => Math.round(v + (pb[i] - v) * Math.max(0, Math.min(1, t))));
  return `rgb(${m[0]},${m[1]},${m[2]})`;
};

export function nodePositions(sizes: number[], width: number, height: number, vertical: boolean, margin: number) {
  return sizes.map((n, l) => {
    const along = margin + (l / (sizes.length - 1)) * ((vertical ? height : width) - 2 * margin);
    const crossSpan = (vertical ? width : height) - 2 * margin;
    // Narrow layers (outputs) sit closer to the centre so the shape reads as a funnel.
    const spread = n <= 4 ? 0.62 : 1;
    return Array.from({length: n}, (_, i) => {
      const c = n === 1 ? 0.5 : i / (n - 1);
      const cross = margin + crossSpan * (0.5 + (c - 0.5) * spread);
      return vertical ? [cross, along] : [along, cross];
    });
  });
}

/** A neural network drawn as glowing nodes and edges; colour = sign (teal +, pink −). */
export const NetworkView: React.FC<NetworkViewProps> = ({
  net,
  sizes,
  edges,
  width,
  height,
  vertical,
  activations,
  nodeValues,
  edgeScale = 2.2,
  layerPulse = [1, 1, 1],
  nodeRadius = 13,
  margin = 70,
}) => {
  const pos = nodePositions(sizes, width, height, vertical, margin);
  const drawn = edges.map((e) => {
    const w = weight(net, sizes, e.layer, e.i, e.j);
    const value = activations ? w * activations[e.layer][e.i] : w;
    const mag = Math.min(1, Math.abs(value) / edgeScale);
    return {e, value, mag};
  });
  drawn.sort((a, b) => a.mag - b.mag);
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{position: 'absolute', left: 0, top: 0}}>
      <defs>
        <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="6" />
        </filter>
        <filter id="nodeGlow" x="-100%" y="-100%" width="300%" height="300%">
          <feGaussianBlur stdDeviation="9" />
        </filter>
      </defs>
      <g filter="url(#glow)" opacity={0.9}>
        {drawn
          .filter((d) => d.mag > 0.35)
          .map(({e, value, mag}) => {
            const [x1, y1] = pos[e.layer][e.i];
            const [x2, y2] = pos[e.layer + 1][e.j];
            return (
              <line
                key={`g${e.key}`}
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke={value >= 0 ? palette.teal : palette.pink}
                strokeWidth={3 + 8 * mag}
                strokeOpacity={mag * layerPulse[e.layer]}
                strokeLinecap="round"
              />
            );
          })}
      </g>
      {drawn.map(({e, value, mag}) => {
        const [x1, y1] = pos[e.layer][e.i];
        const [x2, y2] = pos[e.layer + 1][e.j];
        return (
          <line
            key={e.key}
            x1={x1}
            y1={y1}
            x2={x2}
            y2={y2}
            stroke={value >= 0 ? palette.teal : palette.pink}
            strokeWidth={1.6 + 5.5 * mag}
            strokeOpacity={(0.12 + 0.88 * mag) * (0.35 + 0.65 * layerPulse[e.layer])}
            strokeLinecap="round"
          />
        );
      })}
      {pos.map((layer, l) =>
        layer.map(([x, y], i) => {
          const a = activations ? activations[l][i] : nodeValues ? nodeValues[l][i] : null;
          const strength = a === null ? 0.5 : Math.min(1, Math.abs(a));
          const base = a === null ? '#d6e2ef' : mixHex('#1a2638', a >= 0 ? palette.teal : palette.pink, Math.pow(strength, 0.7));
          const r = nodeRadius * (l === 3 ? 1.35 : 1);
          return (
            <g key={`n${l}-${i}`}>
              {a !== null && strength > 0.25 ? (
                <circle cx={x} cy={y} r={r * (1.6 + strength)} fill={base} opacity={0.55 * strength} filter="url(#nodeGlow)" />
              ) : null}
              <circle cx={x} cy={y} r={r} fill={base} stroke="#e8f1fb" strokeOpacity={0.55} strokeWidth={2} />
            </g>
          );
        }),
      )}
    </svg>
  );
};
