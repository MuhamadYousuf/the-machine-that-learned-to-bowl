"""Original score + sound design + final mix, all synthesised from the edit plan.

Reads  public/edit/plan_9x16.json (timing is identical for both formats) and public/edit/vo.wav
Writes public/edit/music.wav, sfx.wav, mix.wav (48 kHz stereo; mix loudness-normalised to -14 LUFS)

The music follows the story: an uneasy hook, a scrappy "chaos" pulse, a build while the
brains evolve, a hard stop when the cheat is discovered, a dark phrygian pulse for the
cheat, a reset, then the main theme for the comeback and a quiet piano outro.

Run:  audio/.venv/Scripts/python.exe audio/make_audio.py
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt

ROOT = Path(__file__).resolve().parents[1]
EDIT = ROOT / "public" / "edit"
SR = 48000
BPM = 96
BEAT = 60 / BPM
BAR = 4 * BEAT
RNG = np.random.default_rng(7)


def mtof(m):
    return 440.0 * 2 ** ((np.asarray(m, dtype=float) - 69) / 12)


def lp(x, f, order=2):
    return sosfilt(butter(order, min(f, SR * 0.45), "lowpass", fs=SR, output="sos"), x, axis=0)


def hp(x, f, order=2):
    return sosfilt(butter(order, f, "highpass", fs=SR, output="sos"), x, axis=0)


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, min(hi, SR * 0.45)], "bandpass", fs=SR, output="sos"), x, axis=0)


def tvec(dur):
    return np.arange(int(dur * SR)) / SR


def saw(f, dur, phase=0.0):
    t = tvec(dur)
    return 2 * ((f * t + phase) % 1.0) - 1


def adsr(n, a, d, s, r):
    a_n, d_n, r_n = int(a * SR), int(d * SR), int(r * SR)
    e = np.full(n, s, dtype=float)
    a_n = min(a_n, n)
    e[:a_n] = np.linspace(0, 1, a_n, endpoint=False) if a_n else e[:a_n]
    d_end = min(n, a_n + d_n)
    if d_end > a_n:
        e[a_n:d_end] = np.linspace(1, s, d_end - a_n)
    if r_n and n > r_n:
        e[-r_n:] *= np.linspace(1, 0, r_n)
    return e


class Track:
    """Stereo buffer you can drop mono/stereo sounds into at a time, with pan and gain."""

    def __init__(self, dur):
        self.buf = np.zeros((int(dur * SR) + SR * 4, 2))

    def add(self, t, x, gain=1.0, pan=0.0):
        i = int(round(t * SR))
        if i < 0:
            x, i = x[-i:], 0
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            left, right = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
            x = np.stack([x * left, x * right], axis=1) * np.sqrt(2)
        n = min(len(x), len(self.buf) - i)
        if n > 0:
            self.buf[i:i + n] += x[:n] * gain


def reverb(x, seconds=2.6, tone=5000, seed=1):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    decay = np.exp(-6.9 * t / seconds)
    ir = np.stack([lp(rng.standard_normal(n), tone) * decay, lp(rng.standard_normal(n), tone) * decay], axis=1)
    ir[: int(0.012 * SR)] = 0  # pre-delay
    ir /= np.sqrt((ir ** 2).sum(axis=0, keepdims=True))
    wet = np.stack([fftconvolve(x[:, c], ir[:, c])[: len(x)] for c in range(2)], axis=1)
    return wet


# ---------------------------------------------------------------- instruments

def pad(notes, dur, cutoff=1800, detune=0.11, attack=0.6, release=0.8):
    n = int(dur * SR)
    out = np.zeros((n, 2))
    for k, m in enumerate(notes):
        f = float(mtof(m))
        for j, cents in enumerate((-detune, 0.0, detune)):
            ff = f * 2 ** (cents / 12)
            v = saw(ff, dur, phase=RNG.random())
            side = (j - 1) * 0.6
            out[:, 0] += v * (1 - side) * 0.5
            out[:, 1] += v * (1 + side) * 0.5
    out = lp(out, cutoff, order=2)
    out *= adsr(n, attack, 0.4, 0.85, release)[:, None]
    return out / max(1, len(notes)) * 0.5


def pluck(m, dur=0.35, bright=3500, decay=6.0):
    f = float(mtof(m))
    t = tvec(dur)
    x = 0.6 * saw(f, dur) + 0.4 * np.sign(np.sin(2 * np.pi * f * t))
    x = lp(x * np.exp(-t * decay), bright)
    x[: 64] *= np.linspace(0, 1, 64)
    return x * 0.5


def keys(m, dur=2.5):
    """Soft electric-piano/bell tone: a few decaying sine partials."""
    f = float(mtof(m))
    t = tvec(dur)
    x = np.zeros_like(t)
    for h, (amp, dec) in enumerate(((1.0, 1.6), (0.45, 3.0), (0.22, 4.5), (0.12, 7.0), (0.08, 9.0)), start=1):
        x += amp * np.sin(2 * np.pi * f * h * (1 + 0.0007 * h * h) * t) * np.exp(-t * dec)
    x[: 96] *= np.linspace(0, 1, 96)
    return x * 0.35


def bass(m, dur, drive=1.6):
    f = float(mtof(m))
    t = tvec(dur)
    x = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)
    x = np.tanh(drive * x) * adsr(len(t), 0.005, 0.12, 0.7, 0.05)
    return x * 0.5


def kick(level=1.0):
    t = tvec(0.45)
    f = 45 + 95 * np.exp(-t * 28)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7.5)
    x[: 120] += RNG.standard_normal(120) * 0.25 * np.linspace(1, 0, 120)
    return np.tanh(1.4 * x) * 0.9 * level


def snare():
    t = tvec(0.3)
    tone = np.sin(2 * np.pi * 185 * t) * np.exp(-t * 22)
    noise = bp(RNG.standard_normal(len(t)), 1500, 9000) * np.exp(-t * 16)
    return (0.5 * tone + 0.9 * noise) * 0.55


def hat(open_=False):
    t = tvec(0.25 if open_ else 0.06)
    x = hp(RNG.standard_normal(len(t)), 7500) * np.exp(-t * (14 if open_ else 70))
    return x * 0.22


def clap():
    t = tvec(0.25)
    env = np.zeros_like(t)
    for d in (0.0, 0.011, 0.022):
        env += np.where(t >= d, np.exp(-(t - d) * 60), 0)
    env += np.exp(-t * 14) * 0.4
    return bp(RNG.standard_normal(len(t)), 900, 6000) * env * 0.35


# ---------------------------------------------------------------- sections

AM, F, C, G, DM, E = [57, 60, 64, 71], [53, 57, 60, 64], [55, 60, 64, 67], [55, 59, 62, 67], [53, 57, 62, 65], [56, 59, 64, 68]
ROOTS = {id(AM): 45, id(F): 41, id(C): 48, id(G): 43, id(DM): 50, id(E): 40}


def chord_root(ch):
    return ROOTS[id(ch)]


def grid(t0, t1, step):
    n = int(np.floor((t1 - t0) / step + 1e-6))
    return [t0 + i * step for i in range(n)]


def sec_hook(tr, t0, t1, cues):
    # Uneasy suspended drone, a heartbeat that speeds up into the stumps breaking.
    boom = min((c["t"] for c in cues if c["type"] == "boom"), default=t1)
    tr.add(t0, pad([45, 52, 59, 64], boom - t0 + 0.6, cutoff=900, attack=1.6, release=0.6), 0.9)
    t, gap = t0 + 0.4, 0.95
    while t < boom - 0.25:
        tr.add(t, kick(0.55), 0.8)
        tr.add(t + 0.17, kick(0.35), 0.6)
        t += gap
        gap = max(0.42, gap * 0.9)
    # after the stumps go: a long, wide minor chord swelling out of the boom
    tr.add(boom, pad([45, 57, 60, 64, 69], t1 - boom + 1.2, cutoff=2400, attack=0.05, release=1.2), 1.0)
    tr.add(boom, bass(33, min(3.0, t1 - boom), drive=1.0) * np.exp(-tvec(min(3.0, t1 - boom)) * 1.2), 0.9)


def sec_chaos(tr, t0, t1):
    # Scrappy, off-kilter: random pentatonic plucks, a lopsided beat.
    scale = [57, 60, 62, 64, 67, 69, 72, 74]
    for i, t in enumerate(grid(t0, t1, BEAT / 2)):
        if RNG.random() < 0.8:
            m = int(RNG.choice(scale))
            tr.add(t + RNG.normal(0, 0.012), pluck(m, 0.22, bright=2500 + 2500 * RNG.random(), decay=10), 0.55, pan=RNG.uniform(-0.6, 0.6))
        if i % 4 == 0 or (i % 8 == 5):
            tr.add(t, kick(0.7), 0.7)
        if i % 8 == 4:
            tr.add(t, clap(), 0.8)
        tr.add(t, hat(), 0.6, pan=0.3)
    tr.add(t0, pad([45, 57, 60, 63], t1 - t0, cutoff=1100, attack=0.3), 0.45)


def chords_over(tr, t0, t1, prog, cutoff0, cutoff1, gain=0.8):
    bars = grid(t0, t1, BAR) or [t0]
    for k, tb in enumerate(bars):
        ch = prog[k % len(prog)]
        end = min(t1, tb + BAR)
        cut = cutoff0 + (cutoff1 - cutoff0) * (k / max(1, len(bars) - 1))
        tr.add(tb, pad(ch, end - tb + 0.5, cutoff=cut, attack=0.25, release=0.5), gain)
    return bars


def arp(tr, t0, t1, prog, step=BEAT / 4, bright0=1800, bright1=5000, gain=0.45, octave=12):
    ts = grid(t0, t1, step)
    pattern = [0, 1, 2, 3, 2, 1, 2, 3]
    for i, t in enumerate(ts):
        bar = int((t - t0) // BAR)
        ch = prog[bar % len(prog)]
        m = ch[pattern[i % len(pattern)] % len(ch)] + octave
        frac = (t - t0) / max(1e-6, t1 - t0)
        tr.add(t, pluck(m, 0.3, bright=bright0 + (bright1 - bright0) * frac, decay=9), gain, pan=0.35 * np.sin(i * 0.7))


def drums(tr, t0, t1, style="full", level=1.0):
    for i, t in enumerate(grid(t0, t1, BEAT / 2)):
        beat = i // 2
        on = i % 2 == 0
        if style == "half":
            if on and beat % 4 in (0, 2):
                tr.add(t, kick(), 0.8 * level)
            if not on:
                tr.add(t, hat(), 0.5 * level, pan=0.25)
        else:
            if on:
                tr.add(t, kick(), 0.9 * level)
            if on and beat % 2 == 1:
                tr.add(t, snare(), 0.75 * level)
                tr.add(t, clap(), 0.4 * level)
            tr.add(t, hat(open_=not on and beat % 4 == 3), 0.55 * level, pan=0.25)


def bassline(tr, t0, t1, prog, step=BEAT / 2, gain=0.7):
    for t in grid(t0, t1, step):
        bar = int((t - t0) // BAR)
        tr.add(t, bass(chord_root(prog[bar % len(prog)]), step * 0.92), gain)


def sec_build(tr, t0, t1, intensity):
    prog = [AM, F, C, G]
    chords_over(tr, t0, t1, prog, 900 + 600 * intensity, 2200 + 1800 * intensity, 0.7)
    arp(tr, t0, t1, prog, bright0=1500 + 1500 * intensity, bright1=3500 + 2500 * intensity, gain=0.38 + 0.1 * intensity)
    drums(tr, t0, t1, "half" if intensity < 1 else "full", 0.7 + 0.3 * intensity)
    if intensity >= 1:
        bassline(tr, t0, t1, prog)


def sec_twist(tr, t0, t1):
    # The music stops; a cold, dissonant drone and a clock.
    dur = t1 - t0
    tr.add(t0 + 0.15, pad([33, 45, 51, 52], dur + 0.4, cutoff=700, attack=1.2, release=0.4), 1.0)
    for t in grid(t0 + 0.4, t1, 0.5):
        tick = hp(RNG.standard_normal(int(0.012 * SR)), 3000) * 0.25
        tr.add(t, tick, 0.7, pan=0.4 if int((t - t0) / 0.5) % 2 else -0.4)


def sec_cheat(tr, t0, t1):
    # Dark A phrygian pulse with Bb stabs.
    for i, t in enumerate(grid(t0, t1, BEAT / 2)):
        m = 33 if (i // 8) % 2 == 0 else 34
        tr.add(t, bass(m, BEAT / 2 * 0.9, drive=3.0), 0.75)
        if i % 2 == 0:
            tr.add(t, kick(), 0.85)
        if i % 4 == 2:
            tr.add(t, snare(), 0.5)
        tr.add(t, hat(), 0.4, pan=-0.3)
    tr.add(t0, pad([45, 46, 52, 58], t1 - t0, cutoff=1300, attack=0.4), 0.6)


def sec_fix(tr, t0, t1, cues):
    # Hopeful but restrained; everything sucks out on the rewind.
    rew = min((c["t"] for c in cues if c["type"] == "rewind"), default=t1)
    stop = rew + 0.7
    prog = [DM, F, C, G]
    chords_over(tr, t0, stop, prog, 1200, 2600, 0.7)
    arp(tr, t0, stop, prog, step=BEAT / 2, bright0=2000, bright1=3500, gain=0.35)


def sec_theme(tr, t0, t1, level=1.0, drums_on=True):
    prog = [C, G, AM, F]
    chords_over(tr, t0, t1, prog, 2200, 4200, 0.75)
    arp(tr, t0, t1, prog, bright0=3500, bright1=6000, gain=0.42)
    bassline(tr, t0, t1, prog)
    if drums_on:
        drums(tr, t0, t1, "full", level)
    # Bell motif on the first bar of every two.
    motif = [(0, 76), (0.75, 74), (1.5, 72), (2.0, 74), (2.5, 79), (3.25, 76)]
    for tb in grid(t0, t1, 2 * BAR):
        for off, m in motif:
            if tb + off * BEAT < t1:
                tr.add(tb + off * BEAT, keys(m, 1.6), 0.55, pan=0.2)


def sec_think(tr, t0, t1):
    prog = [AM, F, C, G]
    chords_over(tr, t0, t1, prog, 1400, 2000, 0.75)
    arp(tr, t0, t1, prog, step=BEAT / 2, bright0=2500, bright1=2500, gain=0.3)


def sec_result(tr, t0, t1, cues):
    close = min((c["t"] for c in cues if c["type"] == "boom" and c["t"] >= t0), default=t1)
    prog = [F, G, AM, AM]
    chords_over(tr, t0, close, prog, 1500, 4000, 0.75)
    arp(tr, t0, close, prog, step=BEAT / 4, bright0=2000, bright1=6500, gain=0.4)
    for t in grid(t0, close, BEAT):  # snare roll building into "not even close"
        frac = (t - t0) / max(1e-6, close - t0)
        if frac > 0.45:
            tr.add(t, kick(), 0.8)
    roll_start = max(t0, close - 2 * BEAT)
    for k, t in enumerate(grid(roll_start, close, BEAT / 4)):
        tr.add(t, snare(), 0.25 + 0.5 * k / 8)
    sec_theme(tr, close, t1, level=1.0)


def sec_outro(tr, t0, t1, cues):
    end_hit = max((c["t"] for c in cues if c["type"] == "boom"), default=t1)
    prog = [F, C, G, AM]
    chords_over(tr, t0, t1 + 1.5, prog, 1500, 1200, 0.7)
    notes = [64, 67, 72, 71, 67, 64, 62, 60]
    for i, t in enumerate(grid(t0, t1, BEAT)):
        bar = int((t - t0) // BAR)
        if i % 2 == 0 or t > end_hit:
            tr.add(t, keys(notes[i % len(notes)] + (12 if t > end_hit else 0), 2.4), 0.6, pan=0.25 * np.sin(i))
        if i % 4 == 0:
            tr.add(t, keys(chord_root(prog[bar % 4]) + 12, 3.0), 0.5)
    tr.add(end_hit, pad([48, 55, 60, 64, 67, 72], t1 - end_hit + 2.0, cutoff=3000, attack=0.05, release=2.0), 0.9)


def music(plan):
    dur = plan["duration"]
    tr = Track(dur)
    cues = plan["sfx"]
    # Merge consecutive scenes with the same mood so the groove runs through cuts.
    merged = []
    for s in plan["sections"]:
        if merged and merged[-1]["mood"] == s["mood"]:
            merged[-1]["end"] = s["end"]
        else:
            merged.append(dict(s))
    for s in merged:
        t0, t1, mood = s["start"], s["end"], s["mood"]
        local = [c for c in cues if t0 <= c["t"] < t1]
        if mood == "hook":
            sec_hook(tr, t0, t1, local)
        elif mood == "chaos":
            sec_chaos(tr, t0, t1)
        elif mood == "build1":
            sec_build(tr, t0, t1, 0.5)
        elif mood == "build2":
            sec_build(tr, t0, t1, 1.0)
        elif mood == "twist":
            sec_twist(tr, t0, t1)
        elif mood == "cheat":
            sec_cheat(tr, t0, t1)
        elif mood == "fix":
            sec_fix(tr, t0, t1, local)
        elif mood in ("rise", "hero"):
            sec_theme(tr, t0, t1, level=0.85 if mood == "rise" else 1.0, drums_on=True)
        elif mood == "think":
            sec_think(tr, t0, t1)
        elif mood == "result":
            sec_result(tr, t0, t1, local)
        elif mood == "outro":
            sec_outro(tr, t0, t1, local)
    # Hard cuts between sections (short fades so nothing clicks), except into the outro tail.
    x = tr.buf
    x = x + 0.28 * reverb(x, 2.4, 4500)
    # Music-bus EQ: clear room for the voice. Rumble cut, ~-6 dB low shelf, gentle air lift.
    x = hp(x, 32)
    x = x - 0.5 * lp(x, 140)
    x = x + 0.15 * hp(x, 7000)
    n_total = int((dur + 1.0) * SR)
    x = x[:n_total]
    fade = int(3.5 * SR)
    x[-fade:] *= np.linspace(1, 0, fade)[:, None] ** 2
    return x


# ---------------------------------------------------------------- sound effects

def noise(dur):
    return RNG.standard_normal(int(dur * SR))


def sweep_noise(dur, f0, f1, q=0.35):
    """Band-limited noise whose centre frequency glides f0→f1 (processed in short blocks)."""
    n = int(dur * SR)
    x = noise(dur)
    out = np.zeros(n)
    block = 1024
    for i in range(0, n, block):
        frac = i / n
        fc = f0 * (f1 / f0) ** frac
        seg = x[max(0, i - 2048): i + block]
        y = bp(seg, fc * (1 - q), fc * (1 + q))
        out[i: i + block] = y[-len(out[i: i + block]):]
    return out


def sfx_sound(kind):
    if kind in ("whoosh", "whoosh_soft"):
        d = 0.55 if kind == "whoosh" else 0.4
        t = tvec(d)
        env = np.sin(np.pi * np.clip(t / d, 0, 1)) ** 2
        x = sweep_noise(d, 400, 5000) * env
        return x * (0.5 if kind == "whoosh" else 0.25), 0.0, True
    if kind == "riser":
        d = 1.6
        t = tvec(d)
        env = (t / d) ** 2.2
        x = sweep_noise(d, 300, 7000, 0.25) * env + 0.25 * saw(110, d) * env * np.sin(2 * np.pi * (40 + 400 * (t / d) ** 2) * t)
        return x * 0.55, 0.0, True
    if kind in ("boom", "impact", "rise_hit"):
        d = 2.8 if kind == "boom" else 1.6
        t = tvec(d)
        f = 32 + 60 * np.exp(-t * 9)
        sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * (1.6 if kind == "boom" else 3.0))
        crack = lp(noise(d), 3500) * np.exp(-t * 18)
        x = np.tanh(1.3 * sub) * 0.9 + crack * 0.5
        return x * (0.9 if kind == "boom" else 0.7), 0.0, True
    if kind == "hit":
        t = tvec(0.5)
        x = np.tanh(3 * np.sin(2 * np.pi * (55 + 120 * np.exp(-t * 30)) * t)) * np.exp(-t * 9) + bp(noise(0.5), 800, 5000) * np.exp(-t * 30) * 0.6
        return x * 0.55, 0.0, True
    if kind == "stamp":
        t = tvec(0.9)
        x = np.tanh(2 * np.sin(2 * np.pi * (48 + 60 * np.exp(-t * 20)) * t)) * np.exp(-t * 5) + bp(noise(0.9), 200, 2500) * np.exp(-t * 14)
        # bit-crushed glitch tail
        x = np.round(x * 12) / 12
        return x * 0.6, 0.0, True
    if kind == "tick":
        t = tvec(0.12)
        x = np.sin(2 * np.pi * 1800 * t) * np.exp(-t * 60) + 0.3 * np.sin(2 * np.pi * 3600 * t) * np.exp(-t * 90)
        return x * 0.22, 0.0, False
    if kind == "blip":
        t = tvec(0.14)
        x = np.sin(2 * np.pi * (900 + 900 * t / 0.14) * t) * np.exp(-t * 35)
        return x * 0.2, RNG.uniform(-0.4, 0.4), False
    if kind == "rewind":
        d = 0.8
        t = tvec(d)
        f = 1400 * np.exp(-t * 3.2) + 60
        x = saw(1, d) * 0  # placeholder for shape
        x = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.4 + sweep_noise(d, 6000, 200) * 0.6
        x *= np.minimum(1, t / 0.05) * np.minimum(1, (d - t) / 0.1)
        return x * 0.5, 0.0, True
    if kind == "drop":
        # tape-stop style downward glide
        d = 0.6
        t = tvec(d)
        f = 220 * np.exp(-t * 5)
        x = np.tanh(2 * np.sin(2 * np.pi * np.cumsum(f) / SR)) * np.exp(-t * 3)
        return x * 0.5, 0.0, True
    if kind in ("bounce", "bounce_slow"):
        d = 0.18 if kind == "bounce" else 0.45
        t = tvec(d)
        x = np.sin(2 * np.pi * (140 if kind == "bounce" else 80) * t) * np.exp(-t * (40 if kind == "bounce" else 12))
        x += lp(noise(d), 1200) * np.exp(-t * 60) * 0.5
        return x * 0.35, 0.0, False
    if kind == "bat":
        t = tvec(0.35)
        x = np.sin(2 * np.pi * 1150 * t) * np.exp(-t * 35) + np.sin(2 * np.pi * 620 * t) * np.exp(-t * 25) * 0.6
        x += bp(noise(0.35), 1500, 7000) * np.exp(-t * 70) * 0.8
        return x * 0.45, 0.0, True
    if kind == "edge":
        t = tvec(0.3)
        x = np.sin(2 * np.pi * 1900 * t) * np.exp(-t * 60) * 0.6 + bp(noise(0.3), 2500, 9000) * np.exp(-t * 90)
        glove = lp(noise(0.3), 600) * np.exp(-(t - 0.12).clip(0) * 40) * (t > 0.12)
        return (x + glove * 0.9) * 0.6, 0.0, True
    if kind == "pad":
        t = tvec(0.4)
        x = lp(noise(0.4), 500) * np.exp(-t * 30) + np.sin(2 * np.pi * 95 * t) * np.exp(-t * 25)
        return x * 0.7, 0.0, True
    if kind == "crowd":
        # stadium roar: band-limited noise swelling and fading, with a few higher "whistle" partials
        d = 3.2
        t = tvec(d)
        env = np.minimum(1, t / 0.35) * np.exp(-np.maximum(0, t - 0.6) * 1.1)
        x = bp(noise(d), 300, 3500) * env + 0.25 * bp(noise(d), 2500, 6000) * env
        return x * 0.55, 0.0, True
    if kind == "stumps":
        t = tvec(1.2)
        x = np.zeros_like(t)
        for k, (f, d0) in enumerate(((780, 0.0), (1040, 0.03), (1320, 0.07), (2300, 0.12), (2900, 0.2))):
            tt = (t - d0).clip(0)
            x += np.sin(2 * np.pi * f * tt) * np.exp(-tt * (14 + 4 * k)) * (t >= d0)
        x += bp(noise(1.2), 1500, 8000) * np.exp(-t * 25) * 0.6
        return x * 0.5, 0.0, True
    raise KeyError(kind)


def sfx(plan):
    tr = Track(plan["duration"])
    rev_send = Track(plan["duration"])
    for cue in plan["sfx"]:
        x, pan, wet = sfx_sound(cue["type"])
        tr.add(cue["t"], x, 1.0, pan)
        if wet:
            rev_send.add(cue["t"], x, 1.0, pan)
    out = tr.buf + 0.35 * reverb(rev_send.buf, 1.8, 6000, seed=3)
    return out


# ---------------------------------------------------------------- mix

def env_follow(x, ms=120):
    e = np.abs(x)
    return lp(e, 1000 / ms, order=1)


def main():
    import sys
    global EDIT
    cut = sys.argv[1] if len(sys.argv) > 1 else None  # "short" / "long": public/edit/<cut>/
    if cut:
        EDIT = EDIT / cut
    plan_path = next(p for p in (EDIT / "plan_9x16.json", EDIT / "plan_16x9.json") if p.exists())
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    vo, sr = sf.read(EDIT / "vo.wav")
    assert sr == SR
    n = int((plan["duration"] + 1.0) * SR)
    vo = np.pad(vo, (0, max(0, n - len(vo))))[:n]
    vo_st = np.stack([vo, vo], axis=1)

    m = music(plan)
    m = np.pad(m, ((0, max(0, n - len(m))), (0, 0)))[:n]
    s = sfx(plan)[:n]
    s = np.pad(s, ((0, max(0, n - len(s))), (0, 0)))[:n]

    # Sidechain: music dips under the voice (fast attack, slower release).
    env = env_follow(vo, 60)
    env = env / (np.percentile(env[env > 1e-4], 95) + 1e-9)
    duck = 1 - 0.5 * np.clip(env, 0, 1)
    duck = lp(duck, 4, order=1)
    m_norm = m / (np.sqrt(np.mean(m ** 2)) + 1e-9)
    s_norm = s / (np.abs(s).max() + 1e-9)
    vo_rms = np.sqrt(np.mean(vo[vo != 0] ** 2))
    mix = vo_st * 1.0 + m_norm * vo_rms * 0.42 * duck[:, None] + s_norm * 0.42
    mix = np.tanh(mix * 1.1) / 1.1

    sf.write(EDIT / "music.wav", (m_norm * 0.2).astype(np.float32), SR)
    sf.write(EDIT / "sfx.wav", (s_norm * 0.5).astype(np.float32), SR)
    raw = EDIT / "mix_raw.wav"
    sf.write(raw, mix.astype(np.float32), SR)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(raw), "-af", "loudnorm=I=-14:TP=-1.0:LRA=11",
                    "-ar", str(SR), "-c:a", "pcm_s24le", str(EDIT / "mix.wav")], check=True)
    raw.unlink()
    print(f"mix {n / SR:.1f} s written")


if __name__ == "__main__":
    main()
