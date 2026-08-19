"""Head-to-head overlap comparison on real audio: cepstral vs frame-level WavLM.

The last WavLM attempt (`eval/train_overlap_wavlm.py`) was rejected because a
dev-calibrated probability threshold collapsed on AMI 2-speaker's 84% positive
base rate. The frame-level detector (`eval/train_overlap_frames.py`) decides
from *total detected overlap seconds* instead — the same domain-independent
rule the pyannote backend uses — so the decision this script tests is the one
a deployment would actually run, on both real domains at once:

  * Harper Valley (target domain: real 8kHz bank-support telephony, truth
    derived from separately-recorded agent/caller channel timing at this
    system's own 0.35s standard);
  * Trelis/ami-2speaker-test (out-of-scope stress domain: real meeting audio,
    84% positive — the set that killed the previous attempt).

Both detectors are scored on identical clips with identical truth. Every
number is printed next to its trivial-baseline counterpart, per this repo's
own reporting discipline.

Usage:
    python -m eval.compare_overlap_real --model app/models/overlap_frames_wavlm.joblib \
        [--hv-n 60] [--ami-n 50]
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from app.audio.features import build_frames
from app.audio.overlap import _cepstral_competition
from app.config import Thresholds
from app.audio.overlap_frames import _load_wavlm, smooth_and_segment, wavlm_frame_features

HV_REPO = Path(os.environ.get("HARPER_VALLEY_DIR", "data/external/harper_valley_repo/data"))


def _metrics(pred: np.ndarray, truth: np.ndarray) -> str:
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum()); tn = int((~pred & ~truth).sum())
    acc = (tp + tn) / len(truth)
    return f"acc {acc:.3f} ({tp+tn}/{len(truth)})  tp={tp} fp={fp} fn={fn} tn={tn}"


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


SWEEP_CUTOFFS = (0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99)


class FrameDetector:
    def __init__(self, model_path: Path):
        payload = joblib.load(model_path)
        self.clf = payload["model"]
        self.layer = payload["layer"]
        self.cutoff = payload["prob_cutoff"]
        self.extractor, self.wavlm = _load_wavlm()

    def frame_probs(self, samples_16k: np.ndarray) -> np.ndarray:
        feats = wavlm_frame_features(
            self.extractor, self.wavlm, samples_16k, 16000, layers=(self.layer,)
        )[self.layer]
        if len(feats) == 0:
            return np.zeros(0, dtype=np.float32)
        return self.clf.predict_proba(feats)[:, 1]


def _to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return x.astype(np.float32)
    return resample_poly(x.astype(np.float32), 16000, sr).astype(np.float32)


# ---------------------------------------------------------------- Harper Valley

def hv_clips(n: int, seed: int):
    transcripts = sorted((HV_REPO / "transcript").glob("*.json"))
    random.seed(seed)
    sids = [p.stem for p in random.sample(transcripts, n)]
    for sid in sids:
        agent, sr_a = sf.read(HV_REPO / "audio" / "agent" / f"{sid}.wav", dtype="float32")
        caller, _ = sf.read(HV_REPO / "audio" / "caller" / f"{sid}.wav", dtype="float32")
        m = max(len(agent), len(caller))
        agent = np.pad(agent, (0, m - len(agent)))
        caller = np.pad(caller, (0, m - len(caller)))
        mixed = np.clip(agent + caller, -1.0, 1.0)

        segs = json.loads((HV_REPO / "transcript" / f"{sid}.json").read_text())
        total_ms = 0.0
        caller_spans = [(s["offset_ms"], s["offset_ms"] + s["duration_ms"])
                        for s in segs if s["speaker_role"] == "caller"]
        agent_spans = [(s["offset_ms"], s["offset_ms"] + s["duration_ms"])
                       for s in segs if s["speaker_role"] == "agent"]
        for cs, ce in caller_spans:
            for a_s, a_e in agent_spans:
                total_ms += max(0.0, min(ce, a_e) - max(cs, a_s))
        yield sid, mixed, sr_a, total_ms / 1000.0


# ---------------------------------------------------------------- AMI 2-speaker

def ami_clips(n: int):
    from datasets import load_dataset

    ds = load_dataset("Trelis/ami-2speaker-test", split="train")
    for i in range(min(n, len(ds))):
        row = ds[i]
        samples = row["audio"].get_all_samples()
        data = samples.data.numpy()
        if data.ndim > 1:
            data = data[0]
        sr = samples.sample_rate
        duration = samples.duration_seconds
        yield f"clip_{i:02d}", data, sr, row["overlap_ratio"] * duration


# ---------------------------------------------------------------------- driver

def run_domain(name: str, clips, detector: FrameDetector, th: Thresholds, min_sec: float):
    ceps_frac, probs, truth_sec, ids = [], [], [], []
    for cid, audio, sr, true_overlap_sec in clips:
        x16 = _to_16k(audio, sr)
        fb = build_frames(x16, 16000)
        scores = _cepstral_competition(fb.samples, fb.sample_rate)
        frac = float((scores >= th.overlap_frame_score).mean()) if scores.size >= 20 else 0.0
        ceps_frac.append(frac)
        probs.append(detector.frame_probs(x16))
        truth_sec.append(true_overlap_sec)
        ids.append(cid)
        if len(ids) % 10 == 0:
            print(f"  {name}: {len(ids)} clips scored")

    ceps_frac = np.array(ceps_frac)
    truth = np.array(truth_sec) >= min_sec

    base = max(truth.mean(), 1 - truth.mean())
    print(f"\n=== {name}: {len(truth)} clips, {truth.sum()} positive "
          f"(trivial constant baseline {base:.3f}) ===")
    ceps_pred = ceps_frac >= th.overlap_frame_fraction
    print(f"cepstral (shipped):        AUC {_auc(ceps_frac, truth):.3f}  {_metrics(ceps_pred, truth)}")

    for cutoff in SWEEP_CUTOFFS:
        secs = np.array([smooth_and_segment(p, cutoff)[0] for p in probs])
        pred = secs >= min_sec
        tag = " <- shipped cutoff" if abs(cutoff - detector.cutoff) < 1e-9 else ""
        print(f"wavlm-frames @{cutoff:<5}:     AUC {_auc(secs, truth):.3f}  {_metrics(pred, truth)}{tag}")

    secs = np.array([smooth_and_segment(p, detector.cutoff)[0] for p in probs])
    wavlm_pred = secs >= min_sec
    print(f"ensemble OR  (@shipped):   {_metrics(ceps_pred | wavlm_pred, truth)}")
    print(f"ensemble AND (@shipped):   {_metrics(ceps_pred & wavlm_pred, truth)}")
    return {"ids": ids, "truth": truth, "ceps": ceps_frac, "probs": probs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=Path("app/models/overlap_frames_wavlm.joblib"), type=Path)
    parser.add_argument("--hv-n", default=60, type=int)
    parser.add_argument("--ami-n", default=50, type=int)
    parser.add_argument("--seed", default=20250808, type=int)
    args = parser.parse_args()

    th = Thresholds()
    detector = FrameDetector(args.model)
    print(f"loaded frame head: layer {detector.layer}, cutoff {detector.cutoff}")

    if HV_REPO.exists():
        run_domain("Harper Valley (real 8kHz telephony)",
                   hv_clips(args.hv_n, args.seed), detector, th, th.overlap_min_sec)
    else:
        print(f"Harper Valley repo not found at {HV_REPO} — skipping")

    run_domain("AMI 2-speaker (real meetings, 84% positive)",
               ami_clips(args.ami_n), detector, th, th.overlap_min_sec)


if __name__ == "__main__":
    main()
