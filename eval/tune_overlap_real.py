"""Sweep the cepstral overlap threshold against real Harper Valley audio.

`eval/tune_overlap.py` fit 0.25 against the synthetic `ovlp_*` dev set
(AUC 0.593). `eval/harper_valley_eval.py` then measured only 0.52 accuracy
at that threshold on real calls — close to chance. Before concluding the
detector itself is the problem (and pyannote the only fix), check the
narrower question: is 0.25 simply the wrong cutoff for real audio's
`competing_frame_frac` distribution, which may sit somewhere different than
the synthetic set's?
"""

from __future__ import annotations

import json
import random
import warnings
import os
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf

from app.audio.features import build_frames
from app.audio.overlap import _cepstral_competition
from app.audio.vad import detect_voice
from app.config import Thresholds

REPO = Path(os.environ.get(
    "HARPER_VALLEY_DIR",
    "data/external/harper_valley_repo/data",
))


def mix_channels(sid: str) -> tuple[np.ndarray, int]:
    agent, sr_a = sf.read(REPO / "audio" / "agent" / f"{sid}.wav", dtype="float32")
    caller, sr_c = sf.read(REPO / "audio" / "caller" / f"{sid}.wav", dtype="float32")
    n = max(len(agent), len(caller))
    agent = np.pad(agent, (0, n - len(agent)))
    caller = np.pad(caller, (0, n - len(caller)))
    return np.clip(agent + caller, -1.0, 1.0), sr_a


def derive_overlap_truth(transcript: list[dict], min_overlap_ms: float = 350) -> bool:
    caller_spans = [(s["offset_ms"], s["offset_ms"] + s["duration_ms"])
                     for s in transcript if s["speaker_role"] == "caller"]
    agent_spans = [(s["offset_ms"], s["offset_ms"] + s["duration_ms"])
                    for s in transcript if s["speaker_role"] == "agent"]
    total = 0.0
    for cs, ce in caller_spans:
        for a_s, a_e in agent_spans:
            total += max(0.0, min(ce, a_e) - max(cs, a_s))
    return total >= min_overlap_ms


def main(n: int, seed: int) -> None:
    th = Thresholds()
    transcripts = sorted((REPO / "transcript").glob("*.json"))
    random.seed(seed)
    sids = [p.stem for p in random.sample(transcripts, n)]

    fracs, labels = [], []
    for i, sid in enumerate(sids):
        transcript = json.loads((REPO / "transcript" / f"{sid}.json").read_text())
        try:
            mixed, sr = mix_channels(sid)
        except FileNotFoundError:
            continue
        fb = build_frames(mixed, sr)
        vad = detect_voice(fb, th)
        scores = _cepstral_competition(fb.samples, fb.sample_rate)
        if scores.size < 20:
            continue
        frac = float((scores >= th.overlap_frame_score).mean())
        fracs.append(frac)
        labels.append(derive_overlap_truth(transcript))
        print(f"  [{i+1}/{n}] {sid}: frac={frac:.3f} truth={labels[-1]}")

    fracs = np.array(fracs)
    labels = np.array(labels)
    print(f"\ndataset: {len(labels)} real calls, {labels.sum()} positive "
          f"(0.35s-minimum overlap standard, matching config.py)")

    pos = fracs[labels]
    neg = fracs[~labels]
    print(f"positive-class competing_frac: mean={pos.mean():.3f} median={np.median(pos):.3f}")
    print(f"negative-class competing_frac: mean={neg.mean():.3f} median={np.median(neg):.3f}")

    if len(pos) and len(neg):
        ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
        r_pos = ranks[:len(pos)].sum()
        auc = (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
        print(f"AUC on real audio: {auc:.3f} (synthetic dev-set AUC was 0.593)")

    print(f"\n{'threshold':>10} {'accuracy':>10} {'precision':>10} {'recall':>10} {'f1':>10}")
    best = (0.0, -1.0)
    for cutoff in np.arange(0.10, 0.60, 0.02):
        pred = fracs >= cutoff
        tp = int((pred & labels).sum())
        fp = int((pred & ~labels).sum())
        fn = int((~pred & labels).sum())
        tn = int((~pred & ~labels).sum())
        acc = (tp + tn) / len(labels)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best[1]:
            best = (cutoff, f1)
        print(f"{cutoff:>10.2f} {acc:>10.3f} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f}")

    print(f"\nbest F1 threshold on real audio: {best[0]:.2f} (F1={best[1]:.3f}) "
          f"vs shipped 0.25 (fit on synthetic)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    main(args.n, args.seed)
