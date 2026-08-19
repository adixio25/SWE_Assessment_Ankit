"""Frame-level WavLM overlapped-speech detection — inference side.

The shipped cepstral detector measures AUC ~0.59 on every dataset it has been
tested on (synthetic dev set, Harper Valley, AMI x2) — barely above chance,
the weakest field in the system. The published fix for OSD is a pre-trained
speech encoder with a frame-level classification head (Lebourdais et al.,
Interspeech 2022 — WavLM features; Sun et al., Interspeech 2025), and a
previous clip-level WavLM attempt in this repo already showed WavLM's ranking
beats cepstral on every dataset (see `result.md`). What sank that attempt was
the *decision rule*: a clip-level probability threshold calibrated on one
domain's base rate. This module classifies every 20ms frame instead, groups
frame decisions into segments, and reports *total overlap seconds* — the same
domain-independent quantity the pyannote backend outputs, decided by the same
`overlap_min_sec` rule. Seconds mean the same thing at any base rate.

Training lives in `eval/train_overlap_frames.py`; ground-truth frame labels
come from the dev-set generator's recorded overlap spans (truth by
construction, no circularity). The trained head is a logistic regression over
one WavLM-base hidden layer — the artifact at `app/models/overlap_frames_wavlm.joblib`
is a few KB; WavLM-base itself (95M params, ~380MB fp32) loads lazily on
first use and is shared across clips.

Validated before shipping (see `eval/compare_overlap_real.py` output recorded
in TECHNICAL_MEMO.md): grouped-CV on the dev set plus two real-audio domains
(Harper Valley telephony, AMI 2-speaker meetings), against the cepstral
detector on identical clips with identical truth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# WavLM-base conv frontend stride: 320 samples = 20ms per frame at 16kHz.
FRAME_SEC = 0.02
# Median smoothing over 5 frames (100ms); segments under 10 frames (200ms)
# are discarded as detector noise rather than counted as overlap. Segments
# separated by under 10 frames (200ms) are merged first — standard OSD
# post-processing: a sustained event fragmented by a momentary probability dip
# is one event, not several sub-threshold ones.
SMOOTH_FRAMES = 5
MIN_SEG_FRAMES = 10
MERGE_GAP_FRAMES = 10
# Inference chunking keeps peak memory flat on long clips.
CHUNK_SEC = 30.0

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "overlap_frames_wavlm.joblib"

_BACKEND = None  # lazy singleton: (extractor, wavlm, head_payload)


def _load_wavlm():
    from transformers import Wav2Vec2FeatureExtractor, WavLMModel

    extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")
    model = WavLMModel.from_pretrained("microsoft/wavlm-base", output_hidden_states=True)
    model.eval()
    return extractor, model


def wavlm_frame_features(
    extractor, model, samples: np.ndarray, sample_rate: int, layers
) -> dict[int, np.ndarray]:
    """Per-frame hidden states for the requested layers, chunked for memory.

    Returns {layer: (n_frames, 768) float32}. Frame i covers roughly
    [i*20ms, i*20ms+25ms) of audio.
    """
    import torch

    if sample_rate != 16000:
        raise ValueError("WavLM expects 16kHz input; the pipeline already resamples to it")

    chunk = int(CHUNK_SEC * sample_rate)
    outs: dict[int, list[np.ndarray]] = {l: [] for l in layers}
    for start in range(0, len(samples), chunk):
        piece = samples[start : start + chunk]
        if len(piece) < 640:  # under two frames of audio; nothing to score
            continue
        inputs = extractor(piece, sampling_rate=sample_rate, return_tensors="pt")
        with torch.no_grad():
            hidden = model(**inputs).hidden_states
        for l in layers:
            outs[l].append(hidden[l].squeeze(0).numpy().astype(np.float32))
    return {
        l: np.concatenate(v, axis=0) if v else np.zeros((0, 768), np.float32)
        for l, v in outs.items()
    }


def smooth_and_segment(prob: np.ndarray, cutoff: float) -> tuple[float, int]:
    """Median-smooth frame probabilities, threshold, drop tiny segments.

    Returns (total_overlap_sec, event_count).
    """
    if prob.size == 0:
        return 0.0, 0
    k = SMOOTH_FRAMES
    padded = np.pad(prob, (k // 2, k // 2), mode="edge")
    sliding = np.lib.stride_tricks.sliding_window_view(padded, k)
    smoothed = np.median(sliding, axis=1)

    mask = smoothed >= cutoff
    padded_mask = np.concatenate(([False], mask, [False])).astype(int)
    edges = np.diff(padded_mask)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    # Merge segments separated by a sub-200ms dip before the duration filter.
    merged: list[list[int]] = []
    for s, e in zip(starts, ends):
        if merged and s - merged[-1][1] <= MERGE_GAP_FRAMES:
            merged[-1][1] = e
        else:
            merged.append([int(s), int(e)])

    total_frames = 0
    events = 0
    for s, e in merged:
        if e - s >= MIN_SEG_FRAMES:
            total_frames += e - s
            events += 1
    return total_frames * FRAME_SEC, events


def available() -> bool:
    """True when the trained head artifact is present."""
    return MODEL_PATH.exists()


def detect(samples: np.ndarray, sample_rate: int) -> tuple[float, int, float] | None:
    """Total overlap seconds via the frame-level WavLM head.

    Returns (total_sec, event_count, mean_top_prob) or None when the backend
    cannot run (missing artifact or model download failure) — callers fall
    back to the cepstral detector, mirroring how the pyannote backend degrades.
    """
    global _BACKEND
    if not available():
        return None
    try:
        import joblib

        if _BACKEND is None:
            extractor, model = _load_wavlm()
            payload = joblib.load(MODEL_PATH)
            _BACKEND = (extractor, model, payload)
        extractor, model, payload = _BACKEND

        layer = payload["layer"]
        feats = wavlm_frame_features(extractor, model, samples, sample_rate, layers=(layer,))[layer]
        if len(feats) == 0:
            return 0.0, 0, 0.0
        prob = payload["model"].predict_proba(feats)[:, 1]
        total_sec, events = smooth_and_segment(prob, cutoff=payload["prob_cutoff"])
        # Mean of the top decile of frame probabilities: a strength signal for
        # confidence weighting that is robust to clip length.
        top = np.sort(prob)[-max(1, len(prob) // 10):]
        return total_sec, events, float(top.mean())
    except Exception:
        # Same degradation contract as the pyannote backend: any failure here
        # (e.g. first-run model download on an offline box) must not fail the
        # clip — the cepstral detector still answers.
        return None
