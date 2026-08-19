"""Procedural generation of a labelled dev set.

Three labelled clips cannot support cross-validation, and the brief forbids
reporting accuracy from the training set alone. So we build a dev set where the
ground truth is known by construction: we control the mixing, therefore we know
the answer.

Everything here is generated from clean speech taken out of the provided calls
plus synthesised noise. Nothing is downloaded, no external corpus is required,
and no production audio is transmitted anywhere. Noise sources are synthesised
rather than sampled because the acoustic signatures we need to separate — hiss,
hum, rumble, transients, babble — are all well defined in the frequency domain.

The generator deliberately reproduces two properties of the provided calls:

  * gaps are digitally silent on clean material, so detectors cannot rely on a
    convenient continuous room tone; and
  * noise is injected episodically over spans, not laid under the whole clip.

Speech sources are held apart between splits. A clip built from call_001's
speech never appears in both halves of an evaluation split, which is the leakage
the brief warns about.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from app.audio.features import build_frames
from app.audio.io import load_clip
from app.audio.noise import analyse_noise
from app.audio.vad import detect_voice
from app.config import Thresholds

SR = 16_000


# --------------------------------------------------------------------------
# speech pool
# --------------------------------------------------------------------------

@dataclass
class SpeechSource:
    name: str
    utterances: list[np.ndarray]


def extract_speech(paths: list[Path], min_sec: float = 0.6) -> list[SpeechSource]:
    """Pull *clean* utterances out of the provided calls.

    Only the speech itself is kept; the silence between utterances is discarded
    and regenerated later, so a synthesised clip's gap structure is under our
    control rather than inherited.

    Crucially, utterances overlapping a span the noise detector flags are
    dropped. Two of the three provided calls carry background noise over part of
    their length, and building "no noise" negatives out of those spans would
    label noisy audio as clean — the detector would then be measured against a
    reference that disagrees with itself. This is the leakage the brief warns
    about, arriving through the generator rather than through the split.
    """
    th = Thresholds()
    sources: list[SpeechSource] = []
    for path in paths:
        clip = load_clip(path)
        fb = build_frames(clip.samples, SR)
        vad = detect_voice(fb, th)

        noise = analyse_noise(fb, vad, th)
        dirty = [(w.start, w.end) for w in noise.windows if w.noisy]

        def is_clean(start: float, end: float) -> bool:
            return not any(start < d_end and end > d_start for d_start, d_end in dirty)

        utterances = []
        for seg in vad.speech:
            if seg.duration < min_sec or not is_clean(seg.start, seg.end):
                continue
            lo, hi = int(seg.start * SR), int(seg.end * SR)
            chunk = clip.samples[lo:hi]
            peak = float(np.max(np.abs(chunk)))
            if peak > 1e-3:
                utterances.append((chunk / peak * 0.5).astype(np.float32))
        if utterances:
            sources.append(SpeechSource(name=path.stem, utterances=utterances))
    return sources


def _pitch_shift(x: np.ndarray, semitones: float) -> np.ndarray:
    """Crude resampling shift, used to make a second distinct-sounding talker.

    Resampling changes duration as well as pitch, which is acceptable here: we
    need a voice with a different fundamental, not a faithful transposition.
    """
    factor = 2.0 ** (semitones / 12.0)
    n = int(len(x) / factor)
    if n < 8:
        return x
    idx = np.linspace(0, len(x) - 1, n)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


# --------------------------------------------------------------------------
# noise synthesis
# --------------------------------------------------------------------------

def _white(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n).astype(np.float32)


def _lowpass(x: np.ndarray, cutoff_hz: float) -> np.ndarray:
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, 1 / SR)
    # Steep, like a real narrowband codec. A gentle roll-off does not actually
    # remove the consonant band and would label audible speech as muffled.
    spec *= 1.0 / (1.0 + (freqs / max(cutoff_hz, 1.0)) ** 12)
    return np.fft.irfft(spec, n=x.size).astype(np.float32)


def _highpass(x: np.ndarray, cutoff_hz: float) -> np.ndarray:
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, 1 / SR)
    ratio = (freqs / max(cutoff_hz, 1.0)) ** 12
    spec *= ratio / (1.0 + ratio)
    return np.fft.irfft(spec, n=x.size).astype(np.float32)


def _bandpass(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return _lowpass(_highpass(x, lo), hi)


def shaped_noise(psd: np.ndarray, freqs: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Noise shaped to a measured power spectrum.

    Used to reproduce the *actual* background signatures found in the provided
    calls rather than a guess at what television or static sound like. Training
    a classifier on invented signatures was measurably worse on real audio: the
    first version scored 0.69 macro-F1 on synthetic noise and got both real
    noisy clips wrong, because its idea of "TV" was not AutoAce's.
    """
    noise = _white(n, rng)
    spec = np.fft.rfft(noise)
    target_freqs = np.fft.rfftfreq(n, 1 / SR)
    gain = np.interp(target_freqs, freqs, np.sqrt(np.maximum(psd, 1e-12)))
    shaped = np.fft.irfft(spec * gain, n=n).astype(np.float32)
    peak = float(np.max(np.abs(shaped)))
    return (shaped / peak).astype(np.float32) if peak > 0 else shaped


def extract_noise_templates(paths: list[Path]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Measure the real noise spectrum from each labelled noisy call.

    Returns {label: (psd, freqs)} keyed by the reference label for that call, so
    the generator can produce more examples of genuinely the same background.
    """
    import csv

    th = Thresholds()
    templates: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    labels_path = paths[0].parent / "labels.csv"
    if not labels_path.exists():
        return templates

    reference = {}
    for row in csv.DictReader(labels_path.open()):
        reference[row["name"]] = json.loads(row["result_json"])

    for path in paths:
        meta = reference.get(path.name)
        if not meta or not meta.get("background_noise_present"):
            continue
        label = str(meta.get("background_noise_type") or "").strip()
        if not label:
            continue
        clip = load_clip(path)
        fb = build_frames(clip.samples, SR)
        vad = detect_voice(fb, th)
        noise = analyse_noise(fb, vad, th)
        noisy = [w for w in noise.windows if w.noisy]
        if not noisy:
            continue
        psd = np.mean([w.noise_psd for w in noisy], axis=0)
        templates[label] = (psd, fb.freqs.copy())
    return templates


def make_noise(
    kind: str, n: int, rng: np.random.Generator,
    templates: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> np.ndarray:
    """Synthesise one of the canonical background noises."""
    if templates and kind in templates:
        psd, freqs = templates[kind]
        return shaped_noise(psd, freqs, n, rng)

    t = np.arange(n) / SR

    if kind == "sharp static":
        # Broadband hiss with occasional crackle, tilted toward high frequency.
        base = _highpass(_white(n, rng), 1200.0)
        crackle = np.zeros(n, dtype=np.float32)
        for _ in range(max(1, n // (SR * 2))):
            pos = rng.integers(0, max(n - 200, 1))
            crackle[pos:pos + 60] += rng.standard_normal(60).astype(np.float32) * 3.0
        out = base + crackle

    elif kind == "TV":
        # Speech-like band with music-ish tonal content and broadcast dynamics.
        babble = _bandpass(_white(n, rng), 250.0, 3800.0)
        envelope = 1.0 + 0.9 * np.sin(2 * np.pi * 3.5 * t + rng.random() * 6.28)
        tones = sum(
            math.sin(2 * math.pi * f * 1.0) * np.sin(2 * np.pi * f * t)
            for f in (220.0, 330.0, 440.0)
        )
        out = babble * envelope + 0.25 * tones.astype(np.float32)

    elif kind == "office chatter":
        # Several overlapping voice-band streams, no broadcast sheen.
        out = np.zeros(n, dtype=np.float32)
        for _ in range(4):
            stream = _bandpass(_white(n, rng), 300.0, 3200.0)
            rate = 2.5 + rng.random() * 2.5
            out += stream * (1.0 + 0.8 * np.sin(2 * np.pi * rate * t + rng.random() * 6.28))
        out /= 4.0

    elif kind == "music":
        # Sustained triads over a slow chord progression: strongly tonal.
        out = np.zeros(n, dtype=np.float32)
        roots = [196.0, 220.0, 262.0, 294.0]
        seg = max(n // 4, 1)
        for i, root in enumerate(roots):
            lo, hi = i * seg, min((i + 1) * seg, n)
            tt = t[lo:hi]
            for ratio, gain in ((1.0, 1.0), (1.25, 0.7), (1.5, 0.6), (2.0, 0.4)):
                out[lo:hi] += gain * np.sin(2 * np.pi * root * ratio * tt).astype(np.float32)
        out += 0.05 * _bandpass(_white(n, rng), 100.0, 6000.0)

    elif kind == "road noise":
        # Steady low-mid rumble with slow amplitude drift.
        out = _lowpass(_white(n, rng), 900.0) * (1.0 + 0.15 * np.sin(2 * np.pi * 0.3 * t))

    elif kind == "wind":
        # Turbulent very-low-frequency energy with gusting.
        gust = 1.0 + 0.8 * np.sin(2 * np.pi * 0.22 * t + rng.random() * 6.28)
        out = _lowpass(_white(n, rng), 320.0) * gust

    elif kind == "keyboard typing":
        # Sparse bright transients with fast decay.
        out = np.zeros(n, dtype=np.float32)
        strokes = max(2, int(n / SR * 5))
        for _ in range(strokes):
            pos = rng.integers(0, max(n - 900, 1))
            length = 400
            click = _highpass(_white(length, rng), 2500.0)
            click *= np.exp(-np.arange(length) / 60.0).astype(np.float32)
            out[pos:pos + length] += click * 4.0

    elif kind == "mechanical hum":
        # Mains fundamental plus harmonics, very narrowband.
        out = np.zeros(n, dtype=np.float32)
        for k, gain in ((1, 1.0), (2, 0.5), (3, 0.3), (4, 0.15)):
            out += gain * np.sin(2 * np.pi * 50.0 * k * t).astype(np.float32)
        out += 0.03 * _lowpass(_white(n, rng), 400.0)

    else:
        raise ValueError(f"unknown noise kind: {kind}")

    peak = float(np.max(np.abs(out)))
    return (out / peak).astype(np.float32) if peak > 0 else out


# --------------------------------------------------------------------------
# degradations
# --------------------------------------------------------------------------

def apply_clipping(x: np.ndarray, drive: float) -> np.ndarray:
    return np.clip(x * drive, -1.0, 1.0).astype(np.float32)


def apply_muffle(x: np.ndarray, cutoff_hz: float) -> np.ndarray:
    return _lowpass(x, cutoff_hz)


def apply_dropouts(x: np.ndarray, rate_per_sec: float, rng: np.random.Generator) -> np.ndarray:
    """Punch short holes straight through the signal, as packet loss does."""
    out = x.copy()
    count = max(1, int(len(x) / SR * rate_per_sec))
    for _ in range(count):
        length = int(rng.uniform(0.08, 0.18) * SR)
        pos = rng.integers(0, max(len(x) - length, 1))
        out[pos:pos + length] = 0.0
    return out


def apply_reverb(x: np.ndarray, rt60_sec: float, rng: np.random.Generator) -> np.ndarray:
    """Exponentially decaying noise impulse response."""
    n = int(rt60_sec * SR)
    if n < 8:
        return x
    ir = _white(n, rng) * np.exp(-6.9 * np.arange(n) / n).astype(np.float32)
    ir[0] += 3.0
    ir /= np.sum(np.abs(ir)) + 1e-9
    wet = np.convolve(x, ir, mode="full")[: len(x)]
    peak = float(np.max(np.abs(wet)))
    return (wet / peak * float(np.max(np.abs(x)))).astype(np.float32) if peak > 0 else x


def apply_gain(x: np.ndarray, target_dbfs: float) -> np.ndarray:
    rms = float(np.sqrt((x**2).mean()))
    if rms < 1e-9:
        return x
    return (x * (10 ** (target_dbfs / 20.0) / rms)).astype(np.float32)


# --------------------------------------------------------------------------
# clip assembly
# --------------------------------------------------------------------------

@dataclass
class SynthSpec:
    name: str
    source: str                # speech source, held apart across splits
    duration_sec: float
    background_noise_present: bool
    background_noise_type: str
    background_noise_severity: str
    audio_quality: str
    speaker_overlap_present: bool
    long_silence_present: bool
    noise_snr_db: float
    overlap_sec: float
    longest_silence_sec: float
    # (start_sec, end_sec) of each mixed-in overlap event; empty for every group
    # except ovlp_. Ground truth by construction for frame-level overlap training.
    overlap_spans: list = None


def _lay_out_speech(
    utterances: list[np.ndarray],
    duration_sec: float,
    rng: np.random.Generator,
    long_silence_sec: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Build a conversation track: utterances separated by digital silence."""
    total = int(duration_sec * SR)
    track = np.zeros(total, dtype=np.float32)
    pos = int(rng.uniform(0.1, 0.5) * SR)
    longest_gap = 0.0
    inserted_long = long_silence_sec <= 0

    while pos < total - SR:
        utt = utterances[rng.integers(0, len(utterances))]
        end = min(pos + len(utt), total)
        track[pos:end] += utt[: end - pos]
        pos = end

        if not inserted_long and pos < total - int(long_silence_sec * SR) - SR:
            gap = long_silence_sec
            inserted_long = True
        else:
            gap = float(rng.uniform(0.25, 1.4))
        longest_gap = max(longest_gap, gap if pos < total - SR else 0.0)
        pos += int(gap * SR)

    return track, longest_gap


def _inject_noise(
    track: np.ndarray, noise: np.ndarray, snr_db: float,
    coverage: float, rng: np.random.Generator,
) -> np.ndarray:
    """Mix noise into spans covering `coverage` of the clip.

    Episodic by design: the provided calls carry noise over parts of the call
    rather than throughout, and a detector tuned on continuous noise beds will
    not find it.
    """
    out = track.copy()
    speech_rms = float(np.sqrt((track[np.abs(track) > 1e-4] ** 2).mean())) if np.any(np.abs(track) > 1e-4) else 0.05
    noise_gain = speech_rms / (10 ** (snr_db / 20.0))

    total = len(track)
    remaining = int(coverage * total)
    while remaining > SR // 2:
        span = min(int(rng.uniform(2.0, 6.0) * SR), remaining)
        start = int(rng.integers(0, max(total - span, 1)))
        chunk = noise[:span] if len(noise) >= span else np.tile(noise, span // len(noise) + 1)[:span]
        # Fade the edges so the injection is not itself a click.
        fade = min(int(0.05 * SR), span // 4)
        if fade > 1:
            ramp = np.linspace(0, 1, fade, dtype=np.float32)
            chunk = chunk.copy()
            chunk[:fade] *= ramp
            chunk[-fade:] *= ramp[::-1]
        out[start:start + span] += chunk * noise_gain
        remaining -= span
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _add_overlap(
    track: np.ndarray, other: list[np.ndarray], seconds: float, rng: np.random.Generator
) -> tuple[np.ndarray, float, list[tuple[float, float]]]:
    """Mix in a second, pitch-shifted talker over active speech.

    Also returns the placed spans as (start_sec, end_sec) pairs. Recording where
    each overlap event landed adds no RNG draws, so audio generated with a given
    seed is byte-identical to what this function produced before spans were
    recorded — the manifest just carries strictly more ground truth (needed for
    frame-level overlap training, eval/train_overlap_frames.py).
    """
    out = track.copy()
    active = np.flatnonzero(np.abs(track) > 0.02)
    if active.size == 0 or seconds <= 0:
        return out, 0.0, []

    placed = 0.0
    attempts = 0
    spans: list[tuple[float, float]] = []
    while placed < seconds and attempts < 40:
        attempts += 1
        utt = other[rng.integers(0, len(other))]
        shifted = _pitch_shift(utt, float(rng.uniform(-6.0, -3.0)))
        length = min(len(shifted), int(1.6 * SR))
        start = int(active[rng.integers(0, active.size)])
        if start + length >= len(out):
            continue
        # Only counts as overlap where the primary talker is actually speaking.
        if np.mean(np.abs(track[start:start + length]) > 0.02) < 0.7:
            continue
        out[start:start + length] += shifted[:length] * 0.55
        placed += length / SR
        spans.append((start / SR, (start + length) / SR))
    return np.clip(out, -1.0, 1.0).astype(np.float32), placed, spans


def generate(
    out_dir: Path,
    sources: list[SpeechSource],
    n_per_group: int = 60,
    seed: int = 20250808,
    templates: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> list[SynthSpec]:
    """Generate the full dev set and its manifest."""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs: list[SynthSpec] = []

    # A plain list, deliberately not a set: string-set iteration order is
    # hash-randomized per process, and this list's order feeds the shared RNG —
    # a set here makes every run of the generator produce different audio even
    # with the same seed.
    noise_kinds = [
        "sharp static", "TV", "office chatter", "music",
        "road noise", "wind", "keyboard typing", "mechanical hum",
    ]

    def emit(audio: np.ndarray, spec: SynthSpec) -> None:
        sf.write(out_dir / spec.name, audio, SR, subtype="PCM_16")
        specs.append(spec)

    index = 0

    def next_name(tag: str) -> str:
        nonlocal index
        index += 1
        return f"{tag}_{index:04d}.wav"

    # ---- group 1: background noise present/type/severity ----
    for i in range(n_per_group):
        src = sources[i % len(sources)]
        duration = float(rng.uniform(20.0, 45.0))
        track, _ = _lay_out_speech(src.utterances, duration, rng)

        if rng.random() < 0.30:
            audio = apply_gain(track, -20.0)
            emit(audio, SynthSpec(
                name=next_name("noise"), source=src.name, duration_sec=duration,
                background_noise_present=False, background_noise_type="",
                background_noise_severity="none", audio_quality="clear",
                speaker_overlap_present=False, long_silence_present=False,
                noise_snr_db=99.0, overlap_sec=0.0, longest_silence_sec=0.0,
            ))
            continue

        kind = noise_kinds[int(rng.integers(0, len(noise_kinds)))]
        # Severity follows coverage and SNR together, matching how the field is
        # defined: how much the noise actually interferes with the call.
        severity = ["low", "medium", "high"][int(rng.integers(0, 3))]
        snr, coverage = {
            "low": (float(rng.uniform(16, 23)), float(rng.uniform(0.18, 0.35))),
            "medium": (float(rng.uniform(9, 16)), float(rng.uniform(0.35, 0.65))),
            "high": (float(rng.uniform(3, 10)), float(rng.uniform(0.60, 0.95))),
        }[severity]

        noise = make_noise(kind, int(duration * SR), rng, templates)
        audio = apply_gain(_inject_noise(track, noise, snr, coverage, rng), -20.0)
        emit(audio, SynthSpec(
            name=next_name("noise"), source=src.name, duration_sec=duration,
            background_noise_present=True, background_noise_type=kind,
            background_noise_severity=severity, audio_quality="clear",
            speaker_overlap_present=False, long_silence_present=False,
            noise_snr_db=snr, overlap_sec=0.0, longest_silence_sec=0.0,
        ))

    # ---- group 2: audio quality ----
    for i in range(n_per_group):
        src = sources[i % len(sources)]
        duration = float(rng.uniform(20.0, 40.0))
        track, _ = _lay_out_speech(src.utterances, duration, rng)
        roll = rng.random()

        if roll < 0.34:
            audio = apply_gain(track, -20.0)
            quality = "clear"
        elif roll < 0.67:
            choice = int(rng.integers(0, 5))
            if choice == 0:
                audio = apply_gain(apply_muffle(track, float(rng.uniform(2000, 2800))), -20.0)
            elif choice == 1:
                audio = apply_clipping(apply_gain(track, -14.0), float(rng.uniform(2.2, 3.2)))
            elif choice == 2:
                audio = apply_gain(apply_dropouts(track, 0.6, rng), -20.0)
            elif choice == 3:
                audio = apply_gain(track, float(rng.uniform(-42.0, -36.0)))
            else:
                audio = apply_gain(apply_reverb(track, float(rng.uniform(0.35, 0.50)), rng), -20.0)
            quality = "slightly_impaired"
        else:
            choice = int(rng.integers(0, 5))
            if choice == 0:
                audio = apply_gain(apply_muffle(track, float(rng.uniform(900, 1500))), -20.0)
            elif choice == 1:
                audio = apply_clipping(apply_gain(track, -8.0), float(rng.uniform(6.0, 12.0)))
            elif choice == 2:
                audio = apply_gain(apply_dropouts(track, 3.0, rng), -20.0)
            elif choice == 3:
                audio = apply_gain(track, float(rng.uniform(-56.0, -50.0)))
            else:
                audio = apply_gain(apply_reverb(track, float(rng.uniform(0.7, 1.0)), rng), -20.0)
            quality = "severely_impaired"

        emit(audio, SynthSpec(
            name=next_name("qual"), source=src.name, duration_sec=duration,
            background_noise_present=False, background_noise_type="",
            background_noise_severity="none", audio_quality=quality,
            speaker_overlap_present=False, long_silence_present=False,
            noise_snr_db=99.0, overlap_sec=0.0, longest_silence_sec=0.0,
        ))

    # ---- group 3: speaker overlap ----
    for i in range(n_per_group):
        src = sources[i % len(sources)]
        other = sources[(i + 1) % len(sources)]
        duration = float(rng.uniform(20.0, 40.0))
        track, _ = _lay_out_speech(src.utterances, duration, rng)

        want = rng.random() < 0.5
        target = float(rng.uniform(1.5, 6.0)) if want else 0.0
        mixed, placed, spans = _add_overlap(track, other.utterances, target, rng)
        audio = apply_gain(mixed, -20.0)

        emit(audio, SynthSpec(
            name=next_name("ovlp"), source=src.name, duration_sec=duration,
            background_noise_present=False, background_noise_type="",
            background_noise_severity="none", audio_quality="clear",
            speaker_overlap_present=placed >= 0.5, long_silence_present=False,
            noise_snr_db=99.0, overlap_sec=placed, longest_silence_sec=0.0,
            overlap_spans=[[round(s, 3), round(e, 3)] for s, e in spans],
        ))

    # ---- group 4: long silence ----
    for i in range(n_per_group):
        src = sources[i % len(sources)]
        duration = float(rng.uniform(30.0, 60.0))
        want = rng.random() < 0.5
        gap = float(rng.uniform(9.0, 16.0)) if want else float(rng.uniform(1.0, 6.0))
        track, longest = _lay_out_speech(src.utterances, duration, rng, long_silence_sec=gap)
        audio = apply_gain(track, -20.0)

        emit(audio, SynthSpec(
            name=next_name("sil"), source=src.name, duration_sec=duration,
            background_noise_present=False, background_noise_type="",
            background_noise_severity="none", audio_quality="clear",
            speaker_overlap_present=False, long_silence_present=longest >= 8.0,
            noise_snr_db=99.0, overlap_sec=0.0, longest_silence_sec=longest,
        ))

    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps([asdict(s) for s in specs], indent=2))
    return specs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate the synthetic dev set")
    parser.add_argument("--out", default="data/devset", type=Path)
    parser.add_argument("--source-dir", default="requirements", type=Path)
    parser.add_argument("--per-group", default=60, type=int)
    parser.add_argument("--seed", default=20250808, type=int)
    args = parser.parse_args()

    audio_paths = sorted(
        p for p in args.source_dir.iterdir()
        if p.suffix.lower() in {".ogg", ".wav", ".mp3", ".opus", ".m4a", ".flac"}
    )
    speech = extract_speech(audio_paths)
    print(f"speech sources: {[(s.name, len(s.utterances)) for s in speech]}")

    noise_templates = extract_noise_templates(audio_paths)
    print(f"real noise templates: {sorted(noise_templates)}")

    generated = generate(
        args.out, speech, n_per_group=args.per_group, seed=args.seed,
        templates=noise_templates,
    )
    print(f"generated {len(generated)} clips into {args.out}")
