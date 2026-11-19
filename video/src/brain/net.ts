// Network data helpers for the 2D brain clips.

export type NetFrame = {gen: number; w: number[][]; b: number[][]};
export type NetFile = {
  run: string;
  inputs: string[];
  outputs: string[];
  sizes: number[]; // [inputs, hidden1, hidden2, outputs]
  championGeneration: number;
  champion: {w: number[][]; b: number[][]};
  frames: NetFrame[];
};

export type Edge = {layer: number; i: number; j: number; key: string};

/** Weight of edge (i → j) in layer `layer` (matrices are stored row-major as [in, out]). */
export function weight(net: {w: number[][]}, sizes: number[], layer: number, i: number, j: number): number {
  return net.w[layer][i * sizes[layer + 1] + j];
}

/** Weights interpolated between saved generations (eased), for smooth morphs. */
export function netAtGeneration(file: NetFile, generation: number): {w: number[][]; b: number[][]} {
  const frames = file.frames;
  if (generation <= frames[0].gen) return frames[0];
  if (generation >= frames[frames.length - 1].gen) return frames[frames.length - 1];
  let hi = frames.findIndex((f) => f.gen >= generation);
  const lo = Math.max(0, hi - 1);
  const a = frames[lo];
  const b = frames[hi];
  const raw = b.gen === a.gen ? 0 : (generation - a.gen) / (b.gen - a.gen);
  const t = raw * raw * (3 - 2 * raw);
  return {
    w: a.w.map((m, l) => m.map((v, k) => v + (b.w[l][k] - v) * t)),
    b: a.b.map((m, l) => m.map((v, k) => v + (b.b[l][k] - v) * t)),
  };
}

/** Stable edge set: the union of each listed network's top-K edges per layer (no flicker during morphs). */
export function stableEdges(nets: {w: number[][]}[], sizes: number[], perLayer: number[]): Edge[] {
  const keys = new Set<string>();
  for (const net of nets) {
    for (let layer = 0; layer < 3; layer++) {
      const all: [number, number, number][] = [];
      for (let i = 0; i < sizes[layer]; i++) {
        for (let j = 0; j < sizes[layer + 1]; j++) all.push([Math.abs(weight(net, sizes, layer, i, j)), i, j]);
      }
      all.sort((x, y) => y[0] - x[0]);
      for (const [, i, j] of all.slice(0, perLayer[layer])) keys.add(`${layer}:${i}:${j}`);
    }
  }
  return [...keys].map((key) => {
    const [layer, i, j] = key.split(':').map(Number);
    return {layer, i, j, key};
  });
}

/** Forward pass (tanh MLP), returning activations of each layer. */
export function forward(net: {w: number[][]; b: number[][]}, sizes: number[], input: number[]) {
  const layers: number[][] = [input];
  let x = input;
  for (let l = 0; l < 3; l++) {
    const y: number[] = [];
    for (let j = 0; j < sizes[l + 1]; j++) {
      let s = net.b[l][j];
      for (let i = 0; i < sizes[l]; i++) s += x[i] * net.w[l][i * sizes[l + 1] + j];
      y.push(Math.tanh(s));
    }
    layers.push(y);
    x = y;
  }
  return layers;
}

/** Generation shown at progress p ∈ [0,1] on a log scale (evolution changes fastest early on). */
export function logGeneration(p: number, from = 1, to = 1500): number {
  const c = Math.max(0, Math.min(1, p));
  return Math.exp(Math.log(from) + (Math.log(to) - Math.log(from)) * c);
}
