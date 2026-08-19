"""Validate against real (not synthetic) contact-center audio.

Every other eval script in this repo runs against the 600-clip synthetic dev
set (built from RAVDESS/CREMA-D, injected noise/overlap/defects) or the
3 provided known calls. Neither is real, unscripted contact-center speech.
The Gridspace-Stanford Harper Valley dataset (github.com/cricketclub/
gridspace-stanford-harper-valley) is: 1446 simulated bank-support calls,
each with separate agent/caller audio tracks and human-corrected
transcripts with per-segment timing and a 3-class emotion softmax.

What this can and cannot validate, stated up front rather than discovered
mid-analysis:

  * `speaker_overlap_present` — YES, genuinely. The two channels are
    separately recorded, so overlap can be derived directly from segment
    timing (any caller segment's [start, end] intersecting any agent
    segment's) without touching this system's own detection code — an
    independent ground truth, not a self-check.
  * `emotional_tone` — PARTIALLY. Harper Valley's emotion label is a
    3-class softmax (neutral/negative/positive), not this system's 5-class
    tone. Compared at the coarse polarity level only (satisfied->positive,
    frustrated/upset/distressed->negative, neutral->neutral) — a real check,
    but not a claim of 5-class accuracy.
  * `audio_quality` — PARTIALLY. `caller_mos` is an intelligibility MOS
    (3-5 in this dataset, simulated studio conditions, not real telephony
    degradation), bucketed to this system's 3-class scale. A weak proxy for
    a system that measures clipping/dropout/reverb specifically, not overall
    intelligibility — reported as such.
  * `background_noise_present/type/severity` — NOT VALIDATABLE. Harper
    Valley has no noise annotation, and the calls were recorded in clean
    conditions. Not attempted.
  * `long_silence_present` — NOT VALIDATABLE independently. Any "ground
    truth" derived from the same segment-gap logic this system already uses
    would be circular, not a real check. Not attempted.

Audio caveat: agent.wav and caller.wav for the same call are not exactly
equal length (observed ~2s difference on a sample clip) — both start from
call connection but may end at slightly different disconnect times. Mixed
by zero-padding the shorter to the longer and summing, assuming t=0
alignment. Not verified sample-accurate; a real limitation of this adapter,
not of the dataset.
"""

from __future__ import annotations

import asyncio
import json
import random
import warnings
import os
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf

from app.pipeline import analyse_clip
from app.schema import EmotionalTone

REPO = Path(os.environ.get(
    "HARPER_VALLEY_DIR",
    "data/external/harper_valley_repo/data",
))
WORKDIR = Path("/tmp/hv_mixed")

TONE_POLARITY = {
    EmotionalTone.SATISFIED: "positive",
    EmotionalTone.NEUTRAL: "neutral",
    EmotionalTone.FRUSTRATED: "negative",
    EmotionalTone.UPSET: "negative",
    EmotionalTone.DISTRESSED: "negative",
}


def mix_channels(sid: str) -> tuple[np.ndarray, int]:
    agent, sr_a = sf.read(REPO / "audio" / "agent" / f"{sid}.wav", dtype="float32")
    caller, sr_c = sf.read(REPO / "audio" / "caller" / f"{sid}.wav", dtype="float32")
    assert sr_a == sr_c, f"sample rate mismatch for {sid}"
    n = max(len(agent), len(caller))
    agent = np.pad(agent, (0, n - len(agent)))
    caller = np.pad(caller, (0, n - len(caller)))
    mixed = np.clip(agent + caller, -1.0, 1.0)
    return mixed, sr_a


def derive_overlap_truth(transcript: list[dict], min_overlap_ms: float = 350) -> bool:
    """Total overlapping duration must clear a minimum, not just any intersection.

    Matches `config.py`'s own `overlap_min_sec = 0.35` — the threshold this
    codebase already uses to decide a pyannote-detected overlap is "real"
    rather than a sub-second backchannel ("mm-hmm", a brief interruption).
    Applying a different, looser standard to the ground truth than the
    system is held to would make this an unfair comparison, not a real one.
    """
    caller_spans = [
        (s["offset_ms"], s["offset_ms"] + s["duration_ms"])
        for s in transcript if s["speaker_role"] == "caller"
    ]
    agent_spans = [
        (s["offset_ms"], s["offset_ms"] + s["duration_ms"])
        for s in transcript if s["speaker_role"] == "agent"
    ]
    total_overlap_ms = 0.0
    for cs, ce in caller_spans:
        for a_s, a_e in agent_spans:
            total_overlap_ms += max(0.0, min(ce, a_e) - max(cs, a_s))
    return total_overlap_ms >= min_overlap_ms


def derive_tone_truth(transcript: list[dict]) -> str | None:
    caller_segs = [s for s in transcript if s["speaker_role"] == "caller" and s.get("emotion")]
    if not caller_segs:
        return None
    total_dur = sum(s["duration_ms"] for s in caller_segs)
    if total_dur == 0:
        return None
    agg = {"neutral": 0.0, "negative": 0.0, "positive": 0.0}
    for s in caller_segs:
        w = s["duration_ms"] / total_dur
        for k in agg:
            agg[k] += w * s["emotion"].get(k, 0.0)
    return max(agg, key=agg.get)


def derive_quality_truth(mos: float | None) -> str | None:
    if mos is None:
        return None
    if mos >= 4.0:
        return "clear"
    if mos >= 2.5:
        return "slightly_impaired"
    return "severely_impaired"


def pick_subset(n: int, seed: int) -> list[str]:
    transcripts = sorted((REPO / "transcript").glob("*.json"))
    random.seed(seed)
    chosen = random.sample(transcripts, n)
    return [p.stem for p in chosen]


async def main(n: int, seed: int) -> None:
    WORKDIR.mkdir(exist_ok=True)
    sids = pick_subset(n, seed)
    print(f"testing {len(sids)} real Harper Valley calls (seed={seed})")

    overlap_hits = overlap_total = 0
    tone_hits = tone_total = 0
    quality_hits = quality_total = 0
    rows = []

    for sid in sids:
        transcript = json.loads((REPO / "transcript" / f"{sid}.json").read_text())
        metadata = json.loads((REPO / "metadata" / f"{sid}.json").read_text())

        try:
            mixed, sr = mix_channels(sid)
        except FileNotFoundError:
            continue
        wav_path = WORKDIR / f"{sid}.wav"
        sf.write(wav_path, mixed, sr)

        report = await analyse_clip(wav_path, use_cache=False)
        if report.status != "ok":
            print(f"  {sid}: pipeline FAILED — {report.error}")
            continue
        result = report.result

        overlap_truth = derive_overlap_truth(transcript)
        overlap_pred = result.speaker_overlap_present
        overlap_hits += int(overlap_truth == overlap_pred)
        overlap_total += 1

        tone_truth = derive_tone_truth(transcript)
        tone_pred = TONE_POLARITY[result.emotional_tone]
        if tone_truth is not None:
            tone_hits += int(tone_truth == tone_pred)
            tone_total += 1

        mos = metadata.get("labels", {}).get("caller_mos")
        quality_truth = derive_quality_truth(mos)
        if quality_truth is not None:
            quality_hits += int(quality_truth == result.audio_quality.value)
            quality_total += 1

        rows.append((sid, overlap_truth, overlap_pred, tone_truth, tone_pred,
                      quality_truth, result.audio_quality.value))
        print(f"  {sid}: overlap truth={overlap_truth} pred={overlap_pred} | "
              f"tone truth={tone_truth} pred={tone_pred} | "
              f"quality truth={quality_truth} pred={result.audio_quality.value}")

    print("\n=== SUMMARY (real Harper Valley calls, not synthetic) ===")
    print(f"speaker_overlap_present: {overlap_hits}/{overlap_total} = "
          f"{overlap_hits/max(overlap_total,1):.3f}")
    print(f"emotional_tone (coarse polarity only): {tone_hits}/{tone_total} = "
          f"{tone_hits/max(tone_total,1):.3f}")
    print(f"audio_quality (MOS-bucketed proxy): {quality_hits}/{quality_total} = "
          f"{quality_hits/max(quality_total,1):.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    asyncio.run(main(args.n, args.seed))
