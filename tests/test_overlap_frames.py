"""Offline tests for the frame-level overlap post-processing.

These cover the deterministic decision plumbing only — no WavLM download, no
model artifact — mirroring how the rest of this suite tests logic without
network access.
"""

from __future__ import annotations

import numpy as np

from app.audio import overlap_frames
from app.audio.overlap_frames import (
    FRAME_SEC,
    MERGE_GAP_FRAMES,
    MIN_SEG_FRAMES,
    smooth_and_segment,
)


def test_empty_probabilities_yield_nothing():
    assert smooth_and_segment(np.zeros(0), cutoff=0.9) == (0.0, 0)


def test_sustained_segment_is_counted():
    prob = np.zeros(500)
    prob[100:150] = 0.99  # 50 frames = 1s
    sec, events = smooth_and_segment(prob, cutoff=0.9)
    assert events == 1
    assert abs(sec - 50 * FRAME_SEC) < 0.1


def test_blip_below_min_duration_is_dropped():
    prob = np.zeros(500)
    prob[100 : 100 + MIN_SEG_FRAMES - 4] = 0.99  # shorter than the floor
    sec, events = smooth_and_segment(prob, cutoff=0.9)
    assert events == 0
    assert sec == 0.0


def test_fragmented_event_is_merged_across_a_short_dip():
    """A sustained event split by a sub-200ms dip is one event, not two
    sub-threshold fragments."""
    prob = np.zeros(500)
    prob[100:108] = 0.99                      # 8 frames — alone, below the floor
    dip_end = 108 + MERGE_GAP_FRAMES - 2      # dip shorter than the merge gap
    prob[dip_end : dip_end + 8] = 0.99        # another 8 frames
    sec, events = smooth_and_segment(prob, cutoff=0.9)
    assert events == 1
    assert sec > 0.0


def test_distant_blips_are_not_merged():
    prob = np.zeros(1000)
    prob[100:112] = 0.99
    prob[600:612] = 0.99  # far beyond the merge gap
    sec, events = smooth_and_segment(prob, cutoff=0.9)
    assert events == 2


def test_missing_artifact_degrades_to_none(monkeypatch, tmp_path):
    """No trained head on disk -> detect() answers None so the cepstral
    fallback still runs — the same degradation contract as pyannote."""
    monkeypatch.setattr(overlap_frames, "MODEL_PATH", tmp_path / "absent.joblib")
    assert overlap_frames.detect(np.zeros(16000, dtype=np.float32), 16000) is None
