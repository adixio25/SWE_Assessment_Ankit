"""Validate NVIDIA NeMo Sortformer as an overlap-detection backend.

Found while researching alternatives to the licence-gated pyannote backend:
`nvidia/diar_sortformer_4spk-v1` is Apache-2.0, ungated, CPU-fast (~1-6s per
clip observed), and correctly answered all three known calls on first try —
which is exactly the kind of result this session has learned not to trust
without a larger check. It also visibly over-segments 2-speaker calls into 3
"speakers" (observed on call_001/002, both real 2-participant calls), which
could inflate a derived overlap signal from spurious extra speaker labels
rather than real simultaneous speech. Validated here against both the
150-clip synthetic `ovlp_*` dev set (same evidence base as the cepstral
detector) and a real Harper Valley sample, before any decision to wire it
into `app/audio/overlap.py`.
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


def overlap_seconds_from_segments(segments: list[str]) -> tuple[float, int]:
    parsed = []
    for seg in segments:
        s, e, spk = seg.split()
        parsed.append((float(s), float(e), spk))
    total = 0.0
    n_speakers = len({p[2] for p in parsed})
    for i in range(len(parsed)):
        for j in range(i + 1, len(parsed)):
            s1, e1, spk1 = parsed[i]
            s2, e2, spk2 = parsed[j]
            if spk1 != spk2:
                total += max(0.0, min(e1, e2) - max(s1, s2))
    return total, n_speakers


def eval_synthetic_devset(model, devset: Path, min_overlap_sec: float = 0.35):
    manifest = json.loads((devset / "manifest.json").read_text())
    ovlp_clips = [s for s in manifest if s["name"].startswith("ovlp_")]

    fracs, labels = [], []
    for i, spec in enumerate(ovlp_clips):
        path = devset / spec["name"]
        result = model.diarize(audio=[str(path)], batch_size=1, verbose=False)
        overlap_sec, n_spk = overlap_seconds_from_segments(result[0])
        fracs.append(overlap_sec)
        labels.append(bool(spec["speaker_overlap_present"]))
        if (i + 1) % 25 == 0:
            print(f"  synthetic [{i+1}/{len(ovlp_clips)}]")

    fracs = np.array(fracs)
    labels = np.array(labels)
    pred = fracs >= min_overlap_sec
    acc = (pred == labels).mean()
    tp = int((pred & labels).sum())
    fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    pos, neg = fracs[labels], fracs[~labels]
    if len(pos) and len(neg):
        ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
        r_pos = ranks[:len(pos)].sum()
        auc = (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    else:
        auc = float("nan")

    print(f"\nsynthetic ovlp_ dev set ({len(labels)} clips): "
          f"accuracy={acc:.3f} precision={precision:.3f} recall={recall:.3f} "
          f"f1={f1:.3f} AUC={auc:.3f}")
    print(f"  (cepstral fallback on the same set: AUC 0.593, accuracy 0.633 at 0.25)")
    return acc, auc


REPO = Path(os.environ.get(
    "HARPER_VALLEY_DIR",
    "data/external/harper_valley_repo/data",
))


def derive_hv_truth(transcript: list[dict], min_overlap_ms: float = 350) -> bool:
    caller_spans = [(s["offset_ms"], s["offset_ms"] + s["duration_ms"])
                     for s in transcript if s["speaker_role"] == "caller"]
    agent_spans = [(s["offset_ms"], s["offset_ms"] + s["duration_ms"])
                    for s in transcript if s["speaker_role"] == "agent"]
    total = 0.0
    for cs, ce in caller_spans:
        for a_s, a_e in agent_spans:
            total += max(0.0, min(ce, a_e) - max(cs, a_s))
    return total >= min_overlap_ms


def eval_harper_valley(model, n: int, seed: int, min_overlap_sec: float = 0.35):
    transcripts = sorted((REPO / "transcript").glob("*.json"))
    random.seed(seed)
    sids = [p.stem for p in random.sample(transcripts, n)]

    fracs, labels = [], []
    for i, sid in enumerate(sids):
        transcript = json.loads((REPO / "transcript" / f"{sid}.json").read_text())
        agent, sr_a = sf.read(REPO / "audio" / "agent" / f"{sid}.wav", dtype="float32")
        caller, sr_c = sf.read(REPO / "audio" / "caller" / f"{sid}.wav", dtype="float32")
        n_samp = max(len(agent), len(caller))
        agent = np.pad(agent, (0, n_samp - len(agent)))
        caller = np.pad(caller, (0, n_samp - len(caller)))
        mixed = np.clip(agent + caller, -1.0, 1.0)
        wav_path = f"/tmp/hv_sortformer_{sid}.wav"
        sf.write(wav_path, mixed, sr_a)

        result = model.diarize(audio=[wav_path], batch_size=1, verbose=False)
        overlap_sec, n_spk = overlap_seconds_from_segments(result[0])
        fracs.append(overlap_sec)
        labels.append(derive_hv_truth(transcript))
        if (i + 1) % 10 == 0:
            print(f"  harper valley [{i+1}/{n}]")

    fracs = np.array(fracs)
    labels = np.array(labels)
    pred = fracs >= min_overlap_sec
    acc = (pred == labels).mean()
    tp = int((pred & labels).sum())
    fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    tn = int((~pred & ~labels).sum())
    print(f"\nreal Harper Valley ({len(labels)} calls): accuracy={acc:.3f} "
          f"(cepstral fallback on a comparable sample: 0.52, AUC 0.59)")
    print(f"  confusion: tp={tp} fp={fp} fn={fn} tn={tn} | "
          f"positives_in_truth={int(labels.sum())} positives_predicted={int(pred.sum())}")
    print(f"  fracs (predicted overlap seconds) — positive class: {fracs[labels].tolist()}")
    print(f"  fracs (predicted overlap seconds) — negative class: {fracs[~labels].tolist()}")
    return acc


def main(n_devset: int, n_hv: int, seed: int) -> None:
    from nemo.collections.asr.models import SortformerEncLabelModel
    print("loading nvidia/diar_sortformer_4spk-v1 ...")
    model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1")
    model.eval()
    print("loaded.\n")

    devset = Path("data/devset")
    eval_synthetic_devset(model, devset)
    eval_harper_valley(model, n_hv, seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-hv", type=int, default=30)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    main(150, args.n_hv, args.seed)
