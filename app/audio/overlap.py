"""Overlapped-speech detection on a single mixed channel.

The provided calls are dual mono — left and right correlate at 1.0000 — so there
is no per-speaker channel to difference, and overlap has to be found in the
mixture.

Two DSP approaches were built and measured on the synthetic dev set. Both results
are reported here because the negative one is informative:

  * Harmonic-comb enrichment at the main 25 ms analysis resolution. Locate the
    dominant pitch, subtract its comb, test the residual for a second comb.
    Measured AUC 0.50 — no signal at all. The cause is frequency resolution: a
    25 ms frame gives 40 Hz bins, and two talkers' fundamentals routinely sit
    closer together than that, so the two combs are simply not resolvable.

  * Cepstral peak competition at 128 ms resolution (7.8 Hz bins). The log
    spectrum of a single voice has one strong periodicity; a second talker adds
    a competing one. Scored as the ratio of the second cepstral peak to the
    first. Originally reported as AUC 0.66; re-measured in v2 specifically
    against the 150-clip `ovlp_*` dev-set subset (the group actually varying
    this label — the other 450 clips are trivially negative and dilute the
    number) and comes out at **AUC 0.593** (`eval/tune_overlap.py`). Recorded
    here as a measurement correction, not a new regression — the original
    figure most likely mixed in the diluting clips.

The second shipped through iteration 2, and 0.59 is honestly weak. It is the
least reliable field in the system and its confidence contribution is
discounted accordingly. The frame-fraction threshold was re-fit against the
same 150-clip subset — 0.25 (F1 0.464), not the previous hand-set 0.27
(F1 0.368 on the same set, recomputed) — but this does not rescue every weak
instance: one known call's overlap sits at the 10th percentile of the
positive class's own competing-fraction distribution, i.e. acoustically
weaker than 90% of clips that genuinely do overlap, and no threshold catches
that without destroying precision elsewhere.

v3: the default backend is now the frame-level WavLM detector in
`app/audio/overlap_frames.py` (dev CV AUC 0.843 vs 0.596; Harper Valley
0.627 vs 0.548; AMI 0.732 vs 0.429 — see TECHNICAL_MEMO.md "Iteration 3").
The cepstral detector below remains as the fallback when the trained
artifact is absent or WavLM cannot load, and `OVERLAP_BACKEND=cepstral`
forces it. `pyannote/segmentation-3.0` still takes priority when configured:
it is purpose-built for this, but licence-gated behind a manual acceptance
not every deployment will have completed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from app.audio.features import FrameBank
from app.audio.vad import VadResult
from app.config import Thresholds

# High-resolution analysis, used only here. 2048 samples at 16 kHz is a 128 ms
# window giving 7.8 Hz bins — coarse enough to stay quasi-stationary over a
# syllable, fine enough to separate two fundamentals.
OVERLAP_FFT = 2048
OVERLAP_HOP = 256
F0_MIN = 70.0
F0_MAX = 400.0
CEPSTRUM_BAND = (60.0, 1600.0)


@dataclass
class OverlapResult:
    present: bool
    total_sec: float
    event_count: int
    frame_scores: np.ndarray
    evidence: dict

    @property
    def margin(self) -> float:
        """Distance from the decision boundary, for confidence weighting."""
        return float(self.evidence.get("margin", 0.0))


def _cepstral_competition(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Per-frame ratio of the second cepstral peak to the first.

    A single voice produces one dominant quefrency; a second simultaneous voice
    produces a competing one. Frames too quiet to carry pitch are excluded
    rather than scored as zero, which would dilute the statistic.
    """
    if samples.size < OVERLAP_FFT * 2:
        return np.zeros(0, dtype=np.float32)

    n_frames = 1 + (samples.size - OVERLAP_FFT) // OVERLAP_HOP
    idx = np.arange(OVERLAP_FFT)[None, :] + OVERLAP_HOP * np.arange(n_frames)[:, None]
    frames = samples[idx] * np.hanning(OVERLAP_FFT).astype(np.float32)

    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(OVERLAP_FFT, 1.0 / sample_rate)
    bin_width = float(freqs[1] - freqs[0])

    energy = 10.0 * np.log10((frames**2).mean(axis=1) + 1e-12)
    active = np.flatnonzero(energy > max(np.percentile(energy, 55), energy.max() - 30.0))
    if active.size < 10:
        return np.zeros(0, dtype=np.float32)

    band = (freqs >= CEPSTRUM_BAND[0]) & (freqs <= CEPSTRUM_BAND[1])
    quefrency = np.fft.rfftfreq(int(band.sum()), bin_width)
    pitch_range = (quefrency > 1.0 / F0_MAX) & (quefrency < 1.0 / F0_MIN)
    if pitch_range.sum() < 6:
        return np.zeros(0, dtype=np.float32)

    log_spec = np.log(power[active][:, band] + 1e-12)
    log_spec -= log_spec.mean(axis=1, keepdims=True)
    cepstra = np.abs(np.fft.rfft(log_spec, axis=1))[:, pitch_range]

    ordered = np.sort(cepstra, axis=1)[:, ::-1]
    return (ordered[:, 1] / (ordered[:, 0] + 1e-9)).astype(np.float32)


def _pyannote_overlap(samples: np.ndarray, sample_rate: int) -> tuple[float, int] | None:
    """Optional: overlapped-speech regions from pyannote segmentation.

    Returns (total_seconds, event_count), or None when the model is unavailable.
    Enabled by setting OVERLAP_BACKEND=pyannote and providing HF_TOKEN; the
    model is licence-gated (requires accepting the terms once at
    huggingface.co/pyannote/segmentation-3.0), so it stays entirely optional.

    v2 note: `pyannote.audio.pipelines.OverlappedSpeechDetection` was removed in
    pyannote.audio 4.0 (verified against the installed 4.0.7 — the old import
    raises ImportError, it does not silently degrade). The replacement below
    calls the segmentation model directly through `Inference`, which performs
    the same powerset-to-multilabel conversion the old pipeline wrapper did
    internally (see `pyannote.audio.utils.powerset.Powerset` and the docstring
    on `Inference.__init__`): each frame's output is a probability per local
    speaker slot, and a frame counts as overlap when two or more slots are
    active at once. This is lower-level than the old pipeline call but has no
    dependency on a wrapper class that can be renamed or dropped again.
    """
    if os.environ.get("OVERLAP_BACKEND", "").lower() != "pyannote":
        return None
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        return None
    try:
        import torch
        from pyannote.audio import Inference, Model

        model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=token)
        inference = Inference(model, step=2.5)  # powerset auto-converted to multilabel
        waveform = torch.from_numpy(samples).float().unsqueeze(0)
        output = inference({"waveform": waveform, "sample_rate": sample_rate})

        active_speakers = (output.data > 0.5).sum(axis=1)
        overlap_mask = active_speakers >= 2
        step = output.sliding_window.step

        total_sec = float(overlap_mask.sum()) * step
        padded = np.concatenate(([False], overlap_mask, [False]))
        events = int((padded[1:].astype(int) - padded[:-1].astype(int) == 1).sum())
        return total_sec, events
    except Exception:
        # Any failure falls through to the DSP detector rather than failing the
        # clip — including the expected case where HF_TOKEN has not yet
        # accepted the model's licence terms (a one-time manual step at the
        # URL above; it cannot be done from inference code).
        return None


def detect_overlap(fb: FrameBank, vad: VadResult, th: Thresholds) -> OverlapResult:
    external = _pyannote_overlap(fb.samples, fb.sample_rate)
    if external is not None:
        total_sec, events = external
        return OverlapResult(
            present=total_sec >= th.overlap_min_sec,
            total_sec=total_sec,
            event_count=events,
            frame_scores=np.zeros(0, dtype=np.float32),
            evidence={
                "backend": "pyannote/segmentation-3.0",
                "overlap_sec": round(total_sec, 2),
                "events": events,
                "margin": 1.0,
            },
        )

    # v3: frame-level WavLM head (see app/audio/overlap_frames.py). Default when
    # its trained artifact is present; OVERLAP_BACKEND=cepstral forces the old
    # detector. Decides on total detected overlap seconds — the same
    # domain-independent rule as the pyannote backend — not a clip-level
    # probability, which is what sank the previous WavLM attempt on
    # base-rate-shifted domains (see result.md).
    if os.environ.get("OVERLAP_BACKEND", "").lower() != "cepstral":
        from app.audio import overlap_frames

        frames_out = overlap_frames.detect(fb.samples, fb.sample_rate)
        if frames_out is not None:
            total_sec, events, top_prob = frames_out
            margin = min(abs(total_sec - th.overlap_min_sec) / max(th.overlap_min_sec, 1e-6), 1.0)
            return OverlapResult(
                present=total_sec >= th.overlap_min_sec,
                total_sec=total_sec,
                event_count=events,
                frame_scores=np.zeros(0, dtype=np.float32),
                evidence={
                    "backend": "wavlm-frames",
                    "overlap_sec": round(total_sec, 2),
                    "events": events,
                    "top_frame_prob": round(top_prob, 3),
                    "decision_threshold_sec": th.overlap_min_sec,
                    "margin": round(margin, 3),
                },
            )

    scores = _cepstral_competition(fb.samples, fb.sample_rate)
    if scores.size < 20:
        return OverlapResult(
            present=False, total_sec=0.0, event_count=0, frame_scores=scores,
            evidence={
                "backend": "cepstral", "reason": "too few pitched frames",
                "margin": 0.0,
            },
        )

    # Clip-level statistic: what share of pitched frames show a competing pitch.
    # A tail statistic rather than a mean, because overlap covers only part of a
    # call and an average washes it out entirely.
    competing = float((scores >= th.overlap_frame_score).mean())
    present = competing >= th.overlap_frame_fraction
    total_sec = competing * scores.size * (OVERLAP_HOP / fb.sample_rate)

    margin = abs(competing - th.overlap_frame_fraction) / max(th.overlap_frame_fraction, 1e-6)

    evidence = {
        "backend": "cepstral",
        "competing_frame_frac": round(competing, 4),
        "frames_scored": int(scores.size),
        "score_p90": round(float(np.percentile(scores, 90)), 3),
        "overlap_sec": round(total_sec, 2),
        "decision_threshold": th.overlap_frame_fraction,
        "margin": round(min(margin, 1.0), 3),
        "reliability": "low — measured AUC 0.593 on the ovlp_ dev-set subset, AUC 0.590 on real Harper Valley audio",
    }
    return OverlapResult(
        present=bool(present), total_sec=total_sec,
        event_count=int(competing * scores.size), frame_scores=scores,
        evidence=evidence,
    )
