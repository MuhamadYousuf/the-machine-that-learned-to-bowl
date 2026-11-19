export type Word = {w: string; s: number; e: number};
export type CaptionLine = {id: string; start: number; end: number; emph: string[]; words: Word[]; scene: string};
export type Shot = {
  scene: string;
  clip: string;
  start: number;
  end: number;
  rate: number;
  zoom: [number, number];
  grade?: 'red' | 'dim' | null;
  src?: number;
  file?: string;
};
export type Overlay = {type: string; start: number; end: number; [key: string]: unknown};
export type Plan = {
  format: '9x16' | '16x9';
  fps: number;
  duration: number;
  durationInFrames: number;
  sections: {scene: string; mood: string; start: number; end: number}[];
  shots: Shot[];
  overlays: Overlay[];
  sfx: {t: number; type: string}[];
  result: {v1: number; v2: number; close: number};
  captions: CaptionLine[];
};
