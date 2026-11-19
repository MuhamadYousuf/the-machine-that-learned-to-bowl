export type Word = {w: string; s: number; e: number};
export type Line = {id: string; scene: string; text: string; start: number; end: number; words: Word[]};
export type Timeline = {duration: number; lines: Line[]};

export const SCENES = ['hook', 'sees', 'brain', 'gen1', 'cheat', 'fix', 'rise', 'plans', 'outro'] as const;
export type SceneId = (typeof SCENES)[number];

/** Seconds of title card after the last word. */
export const TAIL = 3.2;

export class TL {
  constructor(public tl: Timeline) {}
  line(id: string): Line {
    const l = this.tl.lines.find((x) => x.id === id);
    if (!l) throw new Error(`no line ${id}`);
    return l;
  }
  at(id: string): number {
    return this.line(id).start;
  }
  end(id: string): number {
    return this.line(id).end;
  }
  /** Start time of the first word in line `id` that matches `w` (case/punctuation-insensitive). */
  word(id: string, w: string, nth = 0): number {
    const n = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, '');
    const hits = this.line(id).words.filter((x) => n(x.w) === n(w));
    const hit = hits[Math.min(nth, hits.length - 1)];
    return hit ? hit.s : this.at(id);
  }
  /** [start, end) of each scene. */
  scenes(): Record<SceneId, [number, number]> {
    const starts = SCENES.map((s, i) => {
      if (i === 0) return 0;
      const ls = this.tl.lines.filter((l) => l.scene === s);
      return Math.max(0, ls[0].start - 0.3);
    });
    const total = this.tl.duration + TAIL;
    return Object.fromEntries(SCENES.map((s, i) => [s, [starts[i], i + 1 < starts.length ? starts[i + 1] : total]])) as Record<SceneId, [number, number]>;
  }
}

export const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
export const smooth = (v: number) => {
  const t = clamp01(v);
  return t * t * (3 - 2 * t);
};
/** 0 → 1 between times a and b (smoothstep). */
export const ramp = (t: number, a: number, b: number) => smooth((t - a) / (b - a));
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
export const outBack = (v: number) => {
  const t = clamp01(v);
  const c1 = 1.70158;
  const c3 = c1 + 1;
  return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
};
