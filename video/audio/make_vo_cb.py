"""Expressive voiceover with Chatterbox TTS (per-line emotion), auto-checked with Whisper.

For every line of reel/script.json:
  * generate several takes with the line's exaggeration / pacing settings,
  * transcribe each take with faster-whisper (word timestamps),
  * keep the take whose words match the script best (garbled or skipped words are rejected).
Then lay the lines out with their pauses, add a cinematic voice chain, and write
  public/reel/vo.wav, public/reel/vo_timeline.json (line + word timings), and out/reel/vo_takes/ for review.

Run:  audio/.venv_cb/Scripts/python.exe audio/make_vo_cb.py [--takes 3] [--only h1,h2]
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt, fftconvolve

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "reel" / "script.json"
OUT = ROOT / "public" / "reel"
TAKES = ROOT / "out" / "reel" / "vo_takes"
REF = ROOT / "reel" / "voice_ref.wav"
SR = 48000

NUM = {"0": "zero"}


def norm_words(s: str) -> list[str]:
    s = s.lower().replace("-", " ")
    return [w for w in re.sub(r"[^a-z0-9' ]", " ", s).split() if w]


def wer(ref: list[str], hyp: list[str]) -> float:
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)
    d[:, 0] = np.arange(len(ref) + 1)
    d[0, :] = np.arange(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    return d[-1, -1] / max(1, len(ref))


def spoken(s: str) -> list[str]:
    """Normalise whisper output (digits -> words) for comparison."""
    words = []
    for w in norm_words(s):
        if w.isdigit():
            try:
                from num2words import num2words

                words += norm_words(num2words(int(w)).replace(" and ", " and "))
                continue
            except Exception:
                pass
        words.append(w)
    return words


def polish(x: np.ndarray) -> np.ndarray:
    """Cinematic narration chain: rumble cut, warmth, presence, compression, a touch of room."""
    x = sosfilt(butter(2, 70, "highpass", fs=SR, output="sos"), x)
    warmth = sosfilt(butter(2, [120, 300], "bandpass", fs=SR, output="sos"), x)
    presence = sosfilt(butter(2, [2500, 6000], "bandpass", fs=SR, output="sos"), x)
    x = x + 0.18 * warmth + 0.22 * presence
    env = np.sqrt(sosfilt(butter(1, 10, fs=SR, output="sos"), x * x).clip(1e-9))
    thresh = 0.1
    x = x * np.where(env > thresh, (thresh / env) ** 0.5, 1.0)
    # short, dark room reverb, low in the mix
    n = int(1.1 * SR)
    rng = np.random.default_rng(3)
    ir = rng.standard_normal(n) * np.exp(-np.linspace(0, 7.5, n))
    ir = sosfilt(butter(2, 3800, fs=SR, output="sos"), ir)
    wet = fftconvolve(x, ir)[: len(x)]
    wet = wet / (np.abs(wet).max() + 1e-9) * np.abs(x).max()
    x = x + 0.11 * wet
    return x / (np.abs(x).max() + 1e-9) * 0.9


def trim(audio: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    loud = np.flatnonzero(np.abs(audio) > 0.012)
    if len(loud) == 0:
        return audio, 0.0
    head = max(0, loud[0] - int(0.03 * sr))
    tail = min(len(audio), loud[-1] + int(0.12 * sr))
    return audio[head:tail], head / sr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--takes", type=int, default=3)
    ap.add_argument("--only", default="")
    ap.add_argument("--ref", default=str(REF))
    args = ap.parse_args()

    import torch
    from chatterbox.tts import ChatterboxTTS
    from faster_whisper import WhisperModel

    script = json.loads(SCRIPT.read_text(encoding="utf-8"))
    only = set(filter(None, args.only.split(",")))
    TAKES.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    tts = ChatterboxTTS.from_pretrained(device="cuda")
    ref = args.ref if Path(args.ref).exists() else None
    if ref:
        tts.prepare_conditionals(ref, exaggeration=0.5)
    asr = WhisperModel("small.en", device="cuda", compute_type="float16")

    picked = {}
    for line in script["lines"]:
        lid = line["id"]
        best_file = TAKES / f"{lid}.wav"
        if only and lid not in only and best_file.exists():
            continue
        target = norm_words(line["text"])
        results = []
        for k in range(args.takes):
            torch.manual_seed(1000 + 17 * k + sum(map(ord, lid)) % 997)
            wav = tts.generate(
                line["text"],
                audio_prompt_path=None,
                exaggeration=line["exag"],
                cfg_weight=line["cfg"],
                temperature=0.75,
            )
            audio = wav.squeeze().cpu().numpy().astype(np.float32)
            audio, _ = trim(audio, tts.sr)
            path = TAKES / f"{lid}_take{k}.wav"
            sf.write(path, audio, tts.sr)
            a16 = resample_poly(audio, 16000, tts.sr).astype(np.float32)
            segs, _ = asr.transcribe(a16, language="en", word_timestamps=True, beam_size=5)
            words = [w for s in segs for w in (s.words or [])]
            hyp = spoken(" ".join(w.word for w in words))
            score = wer(target, hyp)
            dur = len(audio) / tts.sr
            # words per second: penalise rushed / dragged takes a little
            rate = len(target) / max(dur, 0.1)
            results.append({"k": k, "wer": score, "dur": dur, "rate": rate, "path": path, "words": [(w.word.strip(), w.start, w.end) for w in words]})
            print(f"  {lid} take{k}: wer={score:.2f} dur={dur:.2f}s rate={rate:.2f}w/s  '{' '.join(hyp)}'")
            if score == 0 and k >= 1:
                break
        results.sort(key=lambda r: (round(r["wer"], 2), abs(r["rate"] - 2.6)))
        best = results[0]
        audio, sr = sf.read(best["path"])
        sf.write(best_file, audio, sr)
        (TAKES / f"{lid}.json").write_text(json.dumps({"wer": best["wer"], "words": best["words"], "sr": sr}), encoding="utf-8")
        picked[lid] = best["wer"]
        print(f"{lid}: picked take{best['k']} wer={best['wer']:.2f}")

    # Assemble every line (re-using picked takes on disk).
    chunks, lines = [np.zeros(int(0.5 * SR))], []
    cursor = 0.5
    for line in script["lines"]:
        lid = line["id"]
        audio, sr = sf.read(TAKES / f"{lid}.wav")
        meta = json.loads((TAKES / f"{lid}.json").read_text(encoding="utf-8"))
        up = resample_poly(audio, SR, sr)
        dur = len(up) / SR
        words = []
        for w, s, e in meta["words"]:
            if not any(c.isalnum() for c in w):
                continue
            words.append({"w": w, "s": round(cursor + s, 3), "e": round(cursor + e, 3)})
        lines.append({"id": lid, "scene": line["scene"], "text": line["text"], "start": round(cursor, 3), "end": round(cursor + dur, 3), "wer": meta["wer"], "words": words})
        chunks += [up, np.zeros(int(line["pause"] * SR))]
        cursor += dur + line["pause"]
    vo = polish(np.concatenate(chunks))
    sf.write(OUT / "vo.wav", vo.astype(np.float32), SR, subtype="PCM_24")
    (OUT / "vo_timeline.json").write_text(json.dumps({"duration": round(len(vo) / SR, 3), "lines": lines}, indent=1), encoding="utf-8")
    bad = [l["id"] for l in lines if l["wer"] > 0.1]
    print(f"total {len(vo) / SR:.1f}s; lines with wer>0.1: {bad}")


if __name__ == "__main__":
    main()
