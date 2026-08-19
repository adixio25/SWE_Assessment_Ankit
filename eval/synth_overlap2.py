"""Second overlap training set: two-speaker turn-taking, realistic hard negatives.

The main dev set's `ovlp_` group builds its negatives from a *single* speaker,
so a frame-level detector trained on it never sees the one thing every real
call is made of: two different voices alternating in turns without overlapping.
Measured consequence (see TECHNICAL_MEMO.md): the first frame-level head scored
false-positive overlap on call_001 — a clean turn-taking call — because a
speaker change was the closest thing to "two voices" it had ever seen labelled
negative.

This generator fixes the training distribution, not the model:

  * Every clip lays out utterances from TWO different sources alternating in
    turns (agent/customer shaped), with digitally-silent gaps like the real
    calls. That makes turn-taking itself an explicit hard negative.
  * Half the clips then receive genuine overlap events — a third voice or a
    pitch-shifted utterance mixed over active speech, spans recorded exactly
    as the main generator now does — including short backchannel-style
    interjections (0.4-1.2s), the overlap type real calls actually contain.
  * Half of all clips are band-limited to telephony bandwidth (3.4kHz), so
    the head cannot rely on wideband cues that 8kHz call audio does not have.

Output is a separate directory (default `data/devset_ovlp2`) so the main dev
set, and every baseline already measured on it, stays byte-identical.
`eval/train_overlap_frames.py --extra` consumes this alongside the main
`ovlp_` group.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from eval.synth import SR, SpeechSource, _lowpass, _pitch_shift, apply_gain, extract_speech


@dataclass
class Ovlp2Spec:
    name: str
    source: str  # primary source group for GroupKFold — the sources are held apart
    duration_sec: float
    speaker_overlap_present: bool
    overlap_sec: float
    overlap_spans: list
    narrowband: bool


def _lay_out_two_speakers(
    a: list[np.ndarray], b: list[np.ndarray], duration_sec: float, rng: np.random.Generator
) -> np.ndarray:
    """Alternate utterances from two speakers with silent gaps, no overlap."""
    out = np.zeros(int(duration_sec * SR), dtype=np.float32)
    cursor = int(rng.uniform(0.2, 1.0) * SR)
    turn = 0
    while cursor < len(out) - SR:
        pool = a if turn % 2 == 0 else b
        utt = pool[rng.integers(0, len(pool))]
        # The second voice is pitch-shifted up a little so the two "talkers"
        # do not share a fundamental even when sources are similar voices.
        if turn % 2 == 1:
            utt = _pitch_shift(utt, float(rng.uniform(2.0, 5.0)))
        end = min(cursor + len(utt), len(out))
        out[cursor:end] += utt[: end - cursor]
        cursor = end + int(rng.uniform(0.15, 1.2) * SR)
        turn += 1
    return np.clip(out, -1.0, 1.0)


def _add_overlap_events(
    track: np.ndarray,
    voices: list[np.ndarray],
    n_events: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, list[list[float]]]:
    """Mix genuine overlap events over active speech, recording spans."""
    out = track.copy()
    active = np.flatnonzero(np.abs(track) > 0.02)
    spans: list[list[float]] = []
    placed = 0.0
    attempts = 0
    while len(spans) < n_events and attempts < 60:
        attempts += 1
        utt = voices[rng.integers(0, len(voices))]
        shifted = _pitch_shift(utt, float(rng.uniform(-6.0, -3.0)))
        # Backchannel-to-interjection lengths: 0.4-1.6s.
        length = min(len(shifted), int(rng.uniform(0.4, 1.6) * SR))
        if active.size == 0:
            break
        start = int(active[rng.integers(0, active.size)])
        if start + length >= len(out):
            continue
        if np.mean(np.abs(track[start : start + length]) > 0.02) < 0.7:
            continue
        out[start : start + length] += shifted[:length] * 0.55
        spans.append([round(start / SR, 3), round((start + length) / SR, 3)])
        placed += length / SR
    return np.clip(out, -1.0, 1.0), placed, spans


def generate(out_dir: Path, sources: list[SpeechSource], n_clips: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs: list[Ovlp2Spec] = []

    for i in range(n_clips):
        src_a = sources[i % len(sources)]
        src_b = sources[(i + 1) % len(sources)]
        src_c = sources[(i + 2) % len(sources)]
        duration = float(rng.uniform(20.0, 40.0))
        track = _lay_out_two_speakers(src_a.utterances, src_b.utterances, duration, rng)

        want = rng.random() < 0.5
        if want:
            n_events = int(rng.integers(1, 5))
            mixed, placed, spans = _add_overlap_events(track, src_c.utterances, n_events, rng)
        else:
            mixed, placed, spans = track, 0.0, []

        narrowband = rng.random() < 0.5
        if narrowband:
            mixed = _lowpass(mixed, float(rng.uniform(3200.0, 4000.0)))

        audio = apply_gain(mixed.astype(np.float32), -20.0)
        name = f"ovlp2_{i + 1:04d}.wav"
        sf.write(out_dir / name, audio, SR, subtype="PCM_16")
        specs.append(
            Ovlp2Spec(
                name=name,
                source=src_a.name,
                duration_sec=duration,
                speaker_overlap_present=placed >= 0.5,
                overlap_sec=placed,
                overlap_spans=spans,
                narrowband=narrowband,
            )
        )

    (out_dir / "manifest.json").write_text(json.dumps([asdict(s) for s in specs], indent=1))
    pos = sum(s.speaker_overlap_present for s in specs)
    print(f"generated {len(specs)} two-speaker clips into {out_dir} ({pos} overlap-positive)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="requirements", type=Path)
    parser.add_argument("--out", default=Path("data/devset_ovlp2"), type=Path)
    parser.add_argument("--n", default=150, type=int)
    parser.add_argument("--seed", default=20250809, type=int)
    args = parser.parse_args()

    paths = sorted(
        p for p in args.source_dir.iterdir()
        if p.suffix.lower() in {".ogg", ".wav", ".mp3", ".opus", ".m4a", ".flac"}
    )
    sources = extract_speech(paths)
    generate(args.out, sources, args.n, args.seed)


if __name__ == "__main__":
    main()
