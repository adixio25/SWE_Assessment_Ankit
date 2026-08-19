"""Frame-level WavLM overlapped-speech detection — the v3 overlap experiment.

Why this exists, given `eval/train_overlap_wavlm.py` already tried WavLM and
was rejected: that attempt mean-pooled the whole clip into one embedding and
thresholded a clip-level *probability*. Two structural problems followed:

  1. Mean-pooling destroys localization — a 2-second overlap in a 40-second
     clip contributes 5% of the pooled embedding, so the classifier is asked
     to detect a signal that pooling has already diluted 20x.
  2. A probability threshold calibrated on one domain's base rate collapses
     when the base rate shifts. That is exactly what happened on the AMI
     2-speaker set (84% positive vs the dev set's ~27%): dev-calibrated
     thresholds scored 10-13/50 against the cepstral detector's 34/50.

The published approach to OSD — WavLM features with a *frame-level*
classification head (Lebourdais et al., Interspeech 2022; Sun et al.,
Interspeech 2025) — fixes both at once. Each 20ms frame is classified
independently, frame decisions are smoothed and grouped into segments, and
the clip-level answer is derived from *total overlap seconds*, the same
domain-independent decision rule the pyannote backend already uses
(`total_sec >= overlap_min_sec`). Seconds of detected overlap mean the same
thing on a 27%-positive dev set and an 84%-positive meeting corpus; a
probability cutoff does not.

Training labels come from the dev-set manifest's `overlap_spans` — the exact
(start, end) of every mixed-in overlap event, recorded by the generator, so
frame labels are ground truth by construction, not derived from any detector
in this repo (no circularity).

Validation is GroupKFold by speech source, the same grouping every other
classifier in this repo uses. Clip-level metrics are reported from
out-of-fold predictions only.

Usage:
    python -m eval.train_overlap_frames --devset data/devset
    python -m eval.train_overlap_frames --devset data/devset --save app/models/overlap_frames_wavlm.joblib
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from app.audio.io import load_clip
from app.audio.overlap_frames import (
    FRAME_SEC,
    _load_wavlm,
    smooth_and_segment,
    wavlm_frame_features,
)

# Candidate transformer layers for the head. Published OSD probes find middle
# layers carry more "who is speaking" structure than the last layer, which
# drifts toward phone identity; we let grouped CV pick instead of assuming.
CANDIDATE_LAYERS = (4, 8, 12)

_CACHE_VERSION = "v1"


def frame_labels(n_frames: int, spans: list[list[float]]) -> np.ndarray:
    """Boolean per-frame labels from ground-truth overlap spans."""
    t = np.arange(n_frames) * FRAME_SEC + FRAME_SEC / 2
    y = np.zeros(n_frames, dtype=bool)
    for s, e in spans:
        y |= (t >= s) & (t <= e)
    return y


def _clip_feature_cache(devset: Path, cache_dir: Path, prefix="ovlp_") -> list[dict]:
    """Extract (or load cached) WavLM frame features + labels for overlap clips."""
    manifest = json.loads((devset / "manifest.json").read_text())
    specs = [s for s in manifest if s["name"].startswith(prefix)]
    cache_dir.mkdir(parents=True, exist_ok=True)

    extractor = model = None
    rows = []
    for spec in specs:
        cpath = cache_dir / f"{_CACHE_VERSION}_{spec['name']}.npz"
        if cpath.exists():
            z = np.load(cpath)
            feats = {l: z[f"layer_{l}"].astype(np.float32) for l in CANDIDATE_LAYERS}
        else:
            if model is None:
                extractor, model = _load_wavlm()
            clip = load_clip(devset / spec["name"])
            feats = wavlm_frame_features(
                extractor, model, clip.samples, clip.sample_rate, layers=CANDIDATE_LAYERS
            )
            np.savez_compressed(
                cpath, **{f"layer_{l}": v.astype(np.float16) for l, v in feats.items()}
            )
        n = len(next(iter(feats.values())))
        rows.append(
            {
                "name": spec["name"],
                "source": spec["source"],
                "feats": feats,
                "frame_y": frame_labels(n, spec.get("overlap_spans") or []),
                "clip_y": bool(spec["speaker_overlap_present"]),
                "overlap_sec": float(spec["overlap_sec"]),
            }
        )
        if len(rows) % 25 == 0:
            print(f"  features: {len(rows)}/{len(specs)} clips")
    return rows


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def _fit_head(X: np.ndarray, y: np.ndarray, seed: int) -> LogisticRegression:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1, random_state=seed)
        clf.fit(X, y)
    return clf


def evaluate_layer(rows: list[dict], layer: int, min_sec: float, seed: int) -> dict:
    """Grouped CV: train frame head on train sources, score held-out clips."""
    groups = np.array([r["source"] for r in rows])
    splitter = GroupKFold(n_splits=len(set(groups)))

    clip_secs = np.full(len(rows), np.nan)
    clip_frame_auc = []
    for train_idx, test_idx in splitter.split(np.zeros(len(rows)), groups=groups):
        Xtr = np.concatenate([rows[i]["feats"][layer][::2] for i in train_idx])
        ytr = np.concatenate([rows[i]["frame_y"][::2] for i in train_idx])
        clf = _fit_head(Xtr, ytr, seed)

        for i in test_idx:
            prob = clf.predict_proba(rows[i]["feats"][layer])[:, 1]
            clip_secs[i], _ = smooth_and_segment(prob, cutoff=0.5)
            if rows[i]["frame_y"].any() and not rows[i]["frame_y"].all():
                clip_frame_auc.append(_auc(prob, rows[i]["frame_y"]))

    clip_y = np.array([r["clip_y"] for r in rows])
    pred = clip_secs >= min_sec
    tp = int((pred & clip_y).sum()); fp = int((pred & ~clip_y).sum())
    fn = int((~pred & clip_y).sum()); tn = int((~pred & ~clip_y).sum())
    return {
        "layer": layer,
        "clip_auc": _auc(clip_secs, clip_y),
        "acc": (tp + tn) / len(rows),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        "mean_frame_auc": float(np.mean(clip_frame_auc)) if clip_frame_auc else float("nan"),
        "oof_secs": clip_secs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devset", default="data/devset", type=Path)
    parser.add_argument("--extra", default=None, type=Path, nargs="*",
                        help="optional additional training dirs (eval/synth_overlap2.py "
                             "two-speaker set, eval/augment_overlap.py augmented set)")
    parser.add_argument("--cache", default=Path("data/wavlm_frame_cache"), type=Path)
    parser.add_argument("--save", default=None, type=Path,
                        help="fit on all clips and save the deployable head here")
    parser.add_argument("--save-cutoff", default=None, type=float,
                        help="override the saved prob_cutoff (default: best dev-CV F1). "
                             "The shipped 0.9 was chosen as the best worst-case accuracy "
                             "across the real-audio domains among dev-viable candidates — "
                             "see eval/compare_overlap_real.py output in TECHNICAL_MEMO.md")
    parser.add_argument("--min-sec", default=0.35, type=float)
    parser.add_argument("--seed", default=20250808, type=int)
    args = parser.parse_args()

    print("extracting WavLM frame features over the ovlp_ dev-set subset...")
    rows = _clip_feature_cache(args.devset, args.cache)
    for extra in args.extra or []:
        print(f"adding clips from {extra}...")
        rows += _clip_feature_cache(extra, args.cache, prefix=("ovlp2_", "aug_"))
    clip_y = np.array([r["clip_y"] for r in rows])
    print(f"dataset: {len(rows)} clips, {clip_y.sum()} positive, "
          f"{sum(len(r['frame_y']) for r in rows)} frames "
          f"({sum(r['frame_y'].sum() for r in rows)} overlap frames)")

    results = []
    for layer in CANDIDATE_LAYERS:
        r = evaluate_layer(rows, layer, args.min_sec, args.seed)
        results.append(r)
        print(f"layer {layer:>2}: clip AUC {r['clip_auc']:.3f}  acc {r['acc']:.3f}  "
              f"P {r['precision']:.3f}  R {r['recall']:.3f}  F1 {r['f1']:.3f}  "
              f"(mean within-clip frame AUC {r['mean_frame_auc']:.3f})")

    best = max(results, key=lambda r: r["clip_auc"])
    print(f"\nbest layer by grouped-CV clip AUC: {best['layer']} at {best['clip_auc']:.3f} "
          f"(shipped cepstral detector on this same subset: see eval/tune_overlap.py)")

    # Operating-point sweep for the winning layer. class_weight='balanced'
    # training inflates frame probabilities (overlap frames are ~1.7% of the
    # set), so 0.5 is far too permissive a cutoff. Swept here under the same
    # grouped CV — chosen from dev data only, then held fixed everywhere else.
    print(f"\noperating-point sweep, layer {best['layer']} (decision: total_sec >= {args.min_sec}):")
    print(f"{'cutoff':>8} {'acc':>7} {'precision':>10} {'recall':>8} {'f1':>7} {'auc':>7}")
    groups = np.array([r["source"] for r in rows])
    splitter = GroupKFold(n_splits=len(set(groups)))
    fold_probs: list[np.ndarray | None] = [None] * len(rows)
    for train_idx, test_idx in splitter.split(np.zeros(len(rows)), groups=groups):
        Xtr = np.concatenate([rows[i]["feats"][best["layer"]][::2] for i in train_idx])
        ytr = np.concatenate([rows[i]["frame_y"][::2] for i in train_idx])
        clf = _fit_head(Xtr, ytr, args.seed)
        for i in test_idx:
            fold_probs[i] = clf.predict_proba(rows[i]["feats"][best["layer"]])[:, 1]

    best_op = (0.5, -1.0)
    for cutoff in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99):
        secs = np.array([smooth_and_segment(p, cutoff)[0] for p in fold_probs])
        pred = secs >= args.min_sec
        tp = int((pred & clip_y).sum()); fp = int((pred & ~clip_y).sum())
        fn = int((~pred & clip_y).sum()); tn = int((~pred & ~clip_y).sum())
        acc = (tp + tn) / len(rows)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        auc = _auc(secs, clip_y)
        print(f"{cutoff:>8} {acc:>7.3f} {prec:>10.3f} {rec:>8.3f} {f1:>7.3f} {auc:>7.3f}")
        if f1 > best_op[1]:
            best_op = (cutoff, f1)
    print(f"best F1 cutoff on grouped-CV dev predictions: {best_op[0]} (F1 {best_op[1]:.3f})")

    if args.save:
        layer = best["layer"]
        X = np.concatenate([r["feats"][layer][::2] for r in rows])
        y = np.concatenate([r["frame_y"][::2] for r in rows])
        clf = _fit_head(X, y, args.seed)
        payload = {
            "model": clf,
            "layer": layer,
            "frame_sec": FRAME_SEC,
            "prob_cutoff": args.save_cutoff if args.save_cutoff is not None else best_op[0],
            "trained_on": f"{len(rows)} ovlp_ dev clips, grouped-CV clip AUC {best['clip_auc']:.3f}",
        }
        args.save.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, args.save)
        print(f"saved frame-level head -> {args.save}")


if __name__ == "__main__":
    main()
