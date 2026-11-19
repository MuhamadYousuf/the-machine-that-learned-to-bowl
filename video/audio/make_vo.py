"""Voiceover: Kokoro TTS for every line of edit/script.json.

Writes
  public/edit/vo.wav            the full narration (48 kHz mono), lines laid end to end with their pauses
  public/edit/vo_timeline.json  line and word timings in seconds (word timings come from Kokoro itself)

Run:  audio/.venv/Scripts/python.exe audio/make_vo.py [--voice af_heart]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "edit" / "script.json"
OUT = ROOT / "public" / "edit"
SR_TTS = 24000
SR = 48000


def polish(x: np.ndarray) -> np.ndarray:
    """Broadcast-style voice chain: rumble cut, gentle presence lift, soft compression."""
    x = sosfilt(butter(2, 80, "highpass", fs=SR, output="sos"), x)
    presence = sosfilt(butter(2, [2500, 6000], "bandpass", fs=SR, output="sos"), x)
    x = x + 0.25 * presence
    # Soft-knee compression on a smoothed envelope.
    env = np.sqrt(sosfilt(butter(1, 12, fs=SR, output="sos"), x * x).clip(1e-9))
    thresh = 0.12
    gain = np.where(env > thresh, (thresh / env) ** 0.45, 1.0)
    x = x * gain
    return x / (np.abs(x).max() + 1e-9) * 0.89


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice")
    parser.add_argument("--script", default=str(SCRIPT), help="narration JSON")
    parser.add_argument("--out", default=str(OUT), help="output folder (vo.wav, vo_timeline.json)")
    args = parser.parse_args()
    from kokoro import KPipeline

    script = json.loads(Path(args.script).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    voice = args.voice or script["voice"]
    pipe = KPipeline(lang_code=voice[0], repo_id="hexgrad/Kokoro-82M")
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks, lines = [], []
    cursor = 0.4  # short breath of silence before the first line
    chunks.append(np.zeros(int(cursor * SR)))
    for line in script["lines"]:
        audio_parts, words, offset = [], [], 0.0
        for result in pipe(line["text"], voice=voice, speed=line["speed"]):
            audio = result.audio.cpu().numpy() if hasattr(result.audio, "cpu") else np.asarray(result.audio)
            for tok in result.tokens or []:
                if not tok.text.strip():
                    continue
                if not any(c.isalnum() for c in tok.text):
                    if words:  # punctuation sticks to the previous word (captions break on it)
                        words[-1]["w"] += tok.text.strip()
                    continue
                if tok.start_ts is None:
                    continue
                words.append({"w": tok.text, "s": round(cursor + offset + tok.start_ts, 3), "e": round(cursor + offset + (tok.end_ts or tok.start_ts + 0.2), 3)})
            audio_parts.append(audio)
            offset += len(audio) / SR_TTS
        audio = np.concatenate(audio_parts)
        # Trim leading/trailing silence so pauses are exactly what the script says.
        loud = np.flatnonzero(np.abs(audio) > 0.01)
        head = max(0, loud[0] - int(0.03 * SR_TTS))
        tail = min(len(audio), loud[-1] + int(0.08 * SR_TTS))
        shift = head / SR_TTS
        audio = audio[head:tail]
        for w in words:
            w["s"] = round(max(cursor, w["s"] - shift), 3)
            w["e"] = round(max(w["s"] + 0.05, w["e"] - shift), 3)
        up = resample_poly(audio, SR, SR_TTS)
        duration = len(up) / SR
        lines.append({**{k: line.get(k, []) for k in ("id", "scene", "text", "emph")}, "start": round(cursor, 3), "end": round(cursor + duration, 3), "words": words})
        chunks.append(up)
        chunks.append(np.zeros(int(line["pause"] * SR)))
        cursor += duration + line["pause"]
        print(f"{line['id']} {cursor:6.2f}s  {line['text']}")

    vo = polish(np.concatenate(chunks))
    sf.write(out_dir / "vo.wav", vo.astype(np.float32), SR, subtype="PCM_24")
    (out_dir / "vo_timeline.json").write_text(json.dumps({"voice": voice, "duration": round(len(vo) / SR, 3), "lines": lines}, indent=1), encoding="utf-8")
    print(f"total {len(vo) / SR:.1f} s, voice {voice}")


if __name__ == "__main__":
    main()
