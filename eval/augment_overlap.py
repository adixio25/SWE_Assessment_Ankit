"""Acoustic-domain augmentation for the overlap training sets.

Measured motivation (eval/compare_overlap_real.py, recorded in
TECHNICAL_MEMO.md): the frame-level WavLM head's *ranking* transfers to real
audio — AMI 2-speaker AUC 0.860 against the cepstral detector's 0.429 — but
its probability *calibration* does not. Real meeting and telephony audio
produce systematically lower frame probabilities than the clean synthetic
training clips, so a fixed cutoff tuned on dev data starves recall exactly
where the detector is otherwise strongest.

That is a training-distribution gap, not a model or decision-rule gap: the
head has mostly seen clean wideband speech, while real calls are band-limited
codec audio (Harper Valley is 8kHz PCM) and real rooms are reverberant (AMI).
This script closes the gap at the data level — each training clip gets one
augmented copy with a random combination of:

  * telephone channel: 300-3400Hz bandpass + a 16k->8k->16k resample round
    trip (kills wideband cues 8kHz call audio never had);
  * synthetic reverb: convolution with an exponentially-decaying noise RIR
    (T60 0.25-0.7s), mixed under the dry signal;
  * additive pink noise at 10-22dB SNR.

Overlap spans carry over unchanged — all three transforms are effectively
time-invariant at the 20ms frame scale (reverb smears tails by ~100-300ms,
accepted as minor label noise rather than corrected, since the smeared frames
genuinely do still contain both voices).

Usage:
    python -m eval.augment_overlap --sources data/devset data/devset_ovlp2 \
        --out data/devset_ovlp_aug
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

from eval.synth import SR

TEL_BAND = (300.0, 3400.0)


def _bandpass(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, 1 / SR)
    spec[(freqs < lo) | (freqs > hi)] = 0.0
    return np.fft.irfft(spec, n=x.size).astype(np.float32)


def _telephone(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = _bandpass(x, TEL_BAND[0], TEL_BAND[1])
    y = resample_poly(y, 1, 2)  # 16k -> 8k
    return resample_poly(y, 2, 1).astype(np.float32)[: x.size]


def _reverb(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    t60 = float(rng.uniform(0.25, 0.7))
    n = int(t60 * SR)
    t = np.arange(n) / SR
    rir = rng.standard_normal(n).astype(np.float32) * np.exp(-6.9 * t / t60)
    rir[0] = 1.0
    wet = fftconvolve(x, rir)[: x.size].astype(np.float32)
    wet /= max(np.max(np.abs(wet)), 1e-6)
    dry_gain = float(rng.uniform(0.5, 0.8))
    out = dry_gain * x + (1 - dry_gain) * wet * max(np.max(np.abs(x)), 1e-6)
    return out.astype(np.float32)


def _pink_noise(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    snr_db = float(rng.uniform(10.0, 22.0))
    white = rng.standard_normal(x.size).astype(np.float32)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(x.size, 1 / SR)
    spec[1:] /= np.sqrt(freqs[1:])
    pink = np.fft.irfft(spec, n=x.size).astype(np.float32)
    sig_rms = float(np.sqrt(np.mean(x**2)) + 1e-9)
    pink_rms = float(np.sqrt(np.mean(pink**2)) + 1e-9)
    pink *= sig_rms / pink_rms / (10 ** (snr_db / 20.0))
    return (x + pink).astype(np.float32)


def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    transforms = []
    if rng.random() < 0.6:
        transforms.append(_telephone)
    if rng.random() < 0.5:
        transforms.append(_reverb)
    if rng.random() < 0.4:
        transforms.append(_pink_noise)
    if not transforms:
        transforms.append(_telephone)
    y = x
    for t in transforms:
        y = t(y, rng)
    return np.clip(y, -1.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", type=Path,
                        default=[Path("data/devset"), Path("data/devset_ovlp2")])
    parser.add_argument("--out", default=Path("data/devset_ovlp_aug"), type=Path)
    parser.add_argument("--seed", default=20250810, type=int)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    out_specs = []
    for src_dir in args.sources:
        manifest = json.loads((src_dir / "manifest.json").read_text())
        for spec in manifest:
            if not spec["name"].startswith(("ovlp_", "ovlp2_")):
                continue
            x, sr = sf.read(src_dir / spec["name"], dtype="float32")
            assert sr == SR
            y = augment(x, rng)
            name = f"aug_{spec['name']}"
            sf.write(args.out / name, y, SR, subtype="PCM_16")
            out_specs.append({**spec, "name": name})

    (args.out / "manifest.json").write_text(json.dumps(out_specs, indent=1))
    pos = sum(bool(s["speaker_overlap_present"]) for s in out_specs)
    print(f"augmented {len(out_specs)} clips into {args.out} ({pos} overlap-positive)")


if __name__ == "__main__":
    main()
