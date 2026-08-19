"""OR-ensemble check: frame-level WavLM head + cepstral detector, dev CV.

The two detectors fail in complementary ways on the known calls (WavLM-frames
misses call_002's weak sustained overlap; cepstral misses call_003's) — an OR
of the two scores 3/3 there. n=3 proves nothing on its own, so this script
measures the same OR rule on the 300-clip overlap training set under the same
GroupKFold used everywhere else: WavLM probabilities are strictly out-of-fold,
cepstral needs no training. Reported per cutoff so the operating point is
chosen from grouped-CV evidence, not from the three known calls.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from app.audio.features import build_frames
from app.audio.io import load_clip
from app.audio.overlap import _cepstral_competition
from app.audio.overlap_frames import smooth_and_segment
from app.config import Thresholds
from eval.train_overlap_frames import _clip_feature_cache, _fit_head


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devset", default=Path("data/devset"), type=Path)
    parser.add_argument("--extra", default=Path("data/devset_ovlp2"), type=Path)
    parser.add_argument("--cache", default=Path("data/wavlm_frame_cache"), type=Path)
    parser.add_argument("--layer", default=4, type=int)
    parser.add_argument("--min-sec", default=0.35, type=float)
    parser.add_argument("--seed", default=20250808, type=int)
    args = parser.parse_args()

    th = Thresholds()
    rows = _clip_feature_cache(args.devset, args.cache)
    rows += _clip_feature_cache(args.extra, args.cache, prefix="ovlp2_")
    clip_y = np.array([r["clip_y"] for r in rows])

    # Out-of-fold WavLM frame probabilities.
    groups = np.array([r["source"] for r in rows])
    splitter = GroupKFold(n_splits=len(set(groups)))
    probs: list[np.ndarray | None] = [None] * len(rows)
    for train_idx, test_idx in splitter.split(np.zeros(len(rows)), groups=groups):
        Xtr = np.concatenate([rows[i]["feats"][args.layer][::2] for i in train_idx])
        ytr = np.concatenate([rows[i]["frame_y"][::2] for i in train_idx])
        clf = _fit_head(Xtr, ytr, args.seed)
        for i in test_idx:
            probs[i] = clf.predict_proba(rows[i]["feats"][args.layer])[:, 1]

    # Cepstral competing-frame fraction per clip (no training, no folds needed).
    print("computing cepstral fractions over both clip sets...")
    ceps = np.zeros(len(rows))
    for i, r in enumerate(rows):
        base = args.devset if r["name"].startswith("ovlp_") else args.extra
        clip = load_clip(base / r["name"])
        fb = build_frames(clip.samples, clip.sample_rate)
        scores = _cepstral_competition(fb.samples, fb.sample_rate)
        ceps[i] = float((scores >= th.overlap_frame_score).mean()) if scores.size >= 20 else 0.0
    ceps_pred = ceps >= th.overlap_frame_fraction

    def line(pred, label):
        tp = int((pred & clip_y).sum()); fp = int((pred & ~clip_y).sum())
        fn = int((~pred & clip_y).sum()); tn = int((~pred & ~clip_y).sum())
        acc = (tp + tn) / len(rows)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        print(f"{label:<28} acc {acc:.3f}  P {prec:.3f}  R {rec:.3f}  F1 {f1:.3f}  "
              f"(tp={tp} fp={fp} fn={fn} tn={tn})")

    print(f"\n{len(rows)} clips, {clip_y.sum()} positive")
    line(ceps_pred, "cepstral alone")
    for cutoff in (0.9, 0.95, 0.97, 0.98, 0.99):
        secs = np.array([smooth_and_segment(p, cutoff)[0] for p in probs])
        wavlm_pred = secs >= args.min_sec
        line(wavlm_pred, f"wavlm-frames @{cutoff}")
        line(wavlm_pred | ceps_pred, f"  OR cepstral @{cutoff}")
        line(wavlm_pred & ceps_pred, f"  AND cepstral @{cutoff}")


if __name__ == "__main__":
    main()
