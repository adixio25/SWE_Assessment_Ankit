# Technical Memo

**Voice tone and background-noise analysis for production call audio**
Submission for the AutoAce AI technical trial · Ankit Hooda

This system was built across three iterations. The first established the
architecture and reached 22/24 on the known calls. The second added
real-audio validation against several external datasets, fixed two
regressions, replaced the tone provider, and corrected a couple of numbers
from the first iteration after re-checking them instead of assuming they
still held. The third replaced the system's weakest component — the cepstral
overlap detector, AUC ~0.59 on every dataset it was ever tested on — with a
frame-level WavLM detector (see "Iteration 3" below). Part 1 is the system
as it stands today. Part 2 is the first iteration — the architecture it
settled on is still exactly what's running; this memo simply didn't stop
there.

---

# Iteration 3 — overlap detection rebuilt at the frame level

## Why the previous WavLM attempt failed, and what actually changed

Iteration 2 already established (see `result.md`) that WavLM embeddings
out-rank the cepstral detector on every dataset, and still declined to ship
them. That attempt had two structural problems, both fixed here rather than
re-tuned:

1. **It mean-pooled the whole clip into one embedding.** A 2-second overlap
   in a 40-second clip is 5% of a pooled embedding — the signal is diluted
   20x before the classifier ever sees it. The published approach to
   overlapped-speech detection (Lebourdais et al., Interspeech 2022; Sun et
   al., Interspeech 2025) classifies every 20ms *frame* and aggregates
   afterwards. Measured here, frame-level classification separates overlap
   from non-overlap almost perfectly *within* a clip (mean within-clip frame
   AUC 0.95).
2. **It thresholded a clip-level probability**, which collapses when a new
   domain's base rate differs from the calibration domain's — exactly what
   happened on the 84%-positive AMI 2-speaker set. The frame-level detector
   instead reports **total detected overlap seconds** and decides with the
   same domain-independent `total_sec >= overlap_min_sec` rule the pyannote
   backend uses. Seconds mean the same thing at any base rate.

## Training data: ground truth by construction, plus two measured gaps closed

Frame labels need to know *where* overlap happens. `eval/synth.py`'s
`_add_overlap` already knew — it just didn't record it. The generator now
writes each mixed-in event's span into the manifest (`overlap_spans`),
adding no RNG draws, so the same seed still produces byte-identical audio.
(A real reproducibility bug was found and fixed while verifying that:
`noise_kinds` was built from a Python string set, whose iteration order is
hash-randomized per process and fed the shared RNG — every run of the
generator produced different audio even at a fixed seed. It's a plain list
now, and two independent runs produce byte-identical output, verified by
checksum.)

Two training-distribution gaps were then found by measuring, not guessed:

* **The first head false-positived on turn-taking.** The dev set's `ovlp_`
  negatives are single-speaker clips, so the closest thing to "two voices"
  the head had ever seen labelled negative was nothing at all — and it
  scored a false positive on call_001, a clean turn-taking call.
  `eval/synth_overlap2.py` generates 150 clips whose *base layout* is two
  speakers alternating in turns (the shape of every real call), half with
  genuine backchannel-length overlap events mixed over active speech, spans
  recorded. Turn-taking is now an explicit hard negative.
* **Real-audio probabilities ran systematically lower than synthetic ones.**
  On AMI the head's ranking was strong (AUC 0.86) but a dev-tuned cutoff
  starved recall — a calibration gap, not a skill gap. `eval/augment_overlap.py`
  gives every training clip an augmented copy: telephone bandpass plus a
  16k→8k→16k round trip, synthetic reverb (T60 0.25–0.7s), and pink noise
  at 10–22dB SNR — the acoustic conditions of Harper Valley's 8kHz codec
  audio and AMI's rooms.

Final training set: 600 clips (150 original `ovlp_` + 150 two-speaker + 300
augmented copies), 212 positive, ~900k frames. The head is a logistic
regression over WavLM-base layer 4 hidden states (layer chosen by grouped
CV over {4, 8, 12}; the artifact is 4KB). Post-processing: 100ms median
smoothing, segments merged across sub-200ms dips, segments under 200ms
dropped, then the 0.35s decision rule.

## Results, all under grouped CV or on untouched real data

| | cepstral (was shipped) | wavlm-frames @0.9 (now shipped) |
|---|---|---|
| dev CV, 600 clips: acc / F1 / AUC | 0.563 / 0.418 / 0.596* | **0.652 / 0.661 / 0.843** |
| Harper Valley n=60: acc / AUC | 0.533 / 0.548 | **0.567 / 0.627** |
| AMI 2-speaker n=50: acc / AUC | 0.680 / **0.429** | 0.580 / **0.732** |
| 3 known calls | 2/3 (misses call_003) | 2/3 (misses call_001) |

\* cepstral's AUC measured on the 150-clip `ovlp_` subset (`eval/tune_overlap.py`);
its acc/F1 on the full 600-clip training set via `eval/tune_overlap_ensemble.py`.

The one number that regresses — AMI accuracy — is disclosed with its cause
rather than averaged away: on that 84%-positive out-of-scope meeting corpus,
the cepstral detector "wins" by calling 6 of the 8 true negatives overlap
(tn=2, AUC 0.429 — *below chance*). Its accuracy there is the base rate
wearing a detector costume. The new head's misses there are under-detection
(fp=0 at the shipped cutoff), and its AUC is 0.73. Both detectors remain
below AMI's trivial 0.84 baseline, the same standing `result.md` already
records for this domain.

The operating point (frame cutoff 0.9) was chosen as the best worst-case
accuracy across the two real domains among dev-viable candidates — cutoffs
0.9–0.99 span dev F1 0.66–0.75, and 0.9 wins Harper Valley (0.567) while
losing least on AMI (0.580). The dev-F1-optimal 0.98 was measured and
declined: it trades 4 points of Harper Valley accuracy and 20 of AMI for
dev-set gains the real domains don't corroborate.

On the known calls the trade is honest and disclosed: the new head detects
call_003's overlap — the 10th-percentile-weakness instance the cepstral
detector was *never* able to catch at any usable threshold — and misses
call_001 instead, detecting 1.06s of overlap the label calls absent. Every
cutoff was checked: call_001's detections shadow call_002's true overlap at
every operating point, so no threshold gets all three. The known-calls
total is unchanged at 22/24.

An OR-ensemble with the cepstral detector was measured and rejected: it
scores 3/3 on the known calls, which is exactly the n=3 trap this repo
documents — on the 600-clip grouped CV the OR's false positives cost it
0.13 F1 against the head alone (0.60 vs 0.73). Complementary errors on
three clips are luck, not structure.

## Cost, latency, degradation

WavLM-base (95M params, ~380MB fp32) runs locally at zero marginal cost —
the cost model is unchanged. Measured warm latency is ~1.2–1.5s per
audio-minute for this stage (M-series CPU), putting the pipeline total at
roughly 7.4s/audio-minute. The backend follows the same degradation
contract as pyannote: missing artifact, offline first run, or any inference
failure falls through to the cepstral detector rather than failing the
clip, and `OVERLAP_BACKEND=cepstral` forces the old behavior outright.
Backend priority is now pyannote (if licensed) → wavlm-frames → cepstral.

Reproducing iteration 3:

```bash
python -m eval.synth --out data/devset --per-group 150          # now records overlap_spans
python -m eval.synth_overlap2 --out data/devset_ovlp2 --n 150   # two-speaker hard negatives
python -m eval.augment_overlap                                   # telephony/reverb/noise copies
python -m eval.train_overlap_frames --devset data/devset \
    --extra data/devset_ovlp2 data/devset_ovlp_aug \
    --save app/models/overlap_frames_wavlm.joblib --save-cutoff 0.9
python -m eval.tune_overlap_ensemble                             # OR/AND ensemble check
python -m eval.compare_overlap_real                              # Harper Valley + AMI head-to-head
```

---

---

# Part 1 — Current state

## Revisiting the first iteration's numbers

Before adding anything new, the first iteration's own code was run again
from scratch, rather than trusting what had been written down. Two numbers
didn't hold up. The known-calls total came back 20/24, not the 22/24 on
record. `background_noise_type`'s dev-set macro-F1 came back 0.690, not
0.798. That's a measurement gap from the first iteration, worth catching
before building on top of it — not a regression introduced later, and not
smoothed over. Everything below is measured against the corrected 20/24,
because that's the number we can stand behind end to end.

| | First iteration | Re-verified | Now |
|---|---|---|---|
| Known calls (of 24) | 22/24 | **20/24** (re-run, unmodified code) | **22/24** (6-trial tone confirmation, noise-type gate fix — below) |
| `background_noise_type`, dev-set macro-F1 | 0.798 | **0.690** (`eval/train_noise_panns.py` baseline run) | **0.829** (PANNs-augmented) |
| `speaker_overlap_present`, AUC | 0.66 (methodology under-documented at the time) | **0.593** (`eval/tune_overlap.py`, unmodified algorithm, correct 150-clip subset) | 0.593, unchanged — the real fix is blocked on a pyannote licence click |
| `emotional_tone`, known calls | 0/3 | 0/3 (confirmed) | **2/3** (emotion2vec+ corroboration, 6-trial verified) |
| Cost, $/audio-minute | $0.00159 | $0.00159 (confirmed) | **$0.00160** (real Azure OpenAI billing — below) |
| Peak RAM | 2.4GB | not re-measured at this checkpoint | **3.2GB** (after the noise-type gate fix) |
| Latency, s/audio-minute | 3.0 | not re-measured at this checkpoint | **5.9** (steady-state) |
| Real external audio validated | none | none | Harper Valley + AMI Corpus (3 fields); MELD attempted, correctly excluded |

> **Provider swap, mid-iteration.** Gemini and Groq are gone from the tone
> chain, replaced by Azure OpenAI. The Gemini-era narrative in Part 2 — the
> emotion2vec+ diagnosis, the free-tier quota framing — stays as accurate
> history, because it's what actually happened and why the `reconcile()`
> rules exist. But the system running now never calls Gemini or Groq for
> tone. See "Tone provider replaced" below.

## What changed this iteration

### `background_noise_type`: a second opinion from PANNs CNN14

`app/audio/noise_panns.py` runs `qiuqiangkong/audioset_tagging_cnn` (CNN14,
81M params, AudioSet-trained) on the same noise-only residual `noise.py`
already isolates, mapped onto the existing eight-word vocabulary via a
label grouping built from AudioSet's real 527-class ontology.

Measured on the 79 noisy dev-set clips, grouped 3-fold CV by speech source
(`eval/train_noise_panns.py`):

| | macro-F1 |
|---|---|
| spectral-only RF (first iteration) | 0.690 |
| spectral + PANNs combined | **0.829** |
| PANNs alone | 0.610 |

The gain concentrates almost entirely in categories the spectral model
couldn't see at all — `keyboard typing` goes from 0.00 to 0.59 F1 — plus a
smaller lift on `mechanical hum`. `app/models/noise_type_panns.joblib` holds
the fitted combined model; `noise.py.classify_type()` tries it first, falls
back to spectral-only, then to the hand-weighted rules.

But the combined model broke a call that used to be right. On `call_002`
(truth: `TV`), spectral-only was confident and correct at 0.75. The combined
model flipped to `keyboard typing` at 0.49, because PANNs returned a
near-useless top tag on that specific residual and the combined model
trusted it anyway. A 79-clip win against one wrong anecdote is a real
tension, and it's weighed explicitly in `noise_panns.py`'s docstring rather
than argued away.

The instinct fix was soft-voting — averaging the two models' probabilities
instead of fully switching to the combined one. Hand-computed on `call_002`
alone, it would have put `TV` back in the lead. But instinct isn't evidence,
so `eval/tune_noise_ensemble.py` ran the same comparison properly across
five blend weights on the full 79-clip set. The result was the opposite of
the hypothesis: 100% combined model wins outright at 0.829 macro-F1, and
every blend toward spectral-only is worse — 0.789 at 50/50, 0.754 at 30%
combined. Soft-voting would have fixed the one clip everyone was staring at
and quietly cost accuracy on the other 78. It didn't ship.

What did ship targets the actual failure mode instead of averaging over it.
On `call_002`, spectral-only was already confident and right; PANNs was
just wrong, and the combined model didn't know to ignore it. So
`classify_type()` now checks spectral-only's own confidence first, and only
consults PANNs when spectral-only isn't already sure — a gate, not a blend.
Sweeping the gate threshold on the dev set (`eval/tune_noise_gate.py`) found
that anything from 0.6 to 0.7 matches the combined model's 0.829 macro-F1
exactly, with no regression, while routing roughly half the clips to
spectral-only instead of PANNs. It shipped at 0.6, the most inclusive tied
threshold. `background_noise_type` is now 3/3 on the known calls, up from
2/3, at zero cost to the dev-set number that justified PANNs in the first
place — and as a side effect, PANNs is now skipped whenever spectral-only
is confident, which is a latency win too, not just an accuracy one.

### Overlap detection: the pyannote path is fixed, still blocked on one click

The first iteration's code called
`pyannote.audio.pipelines.OverlappedSpeechDetection`, a class removed in
pyannote.audio 4.0 — confirmed by the import itself raising `ImportError`
against the installed 4.0.7, not silently degrading.
`app/audio/overlap.py._pyannote_overlap` now calls the segmentation model
directly through `Inference` and decodes its multilabel output itself (two
or more active speaker slots means overlap), so it no longer depends on a
wrapper class that can be renamed out from under it.

It still isn't running end to end. `pyannote/segmentation-3.0` is gated,
and although `HF_TOKEN` is configured, that account hasn't yet accepted the
model's licence — a manual step at huggingface.co, re-checked live during
this iteration and still returning a 403. The cepstral fallback is what
actually runs in production.

That fallback's own accuracy figure didn't survive a proper re-check
either. The first iteration recorded AUC 0.66. Measured correctly, against
just the 150 dev-set clips that actually vary this label rather than all
600 — the other 450 are trivially negative and were diluting the number —
the real figure is AUC 0.593. The F1-optimal cutoff on that same subset is
0.25, not the shipped 0.27, a real difference in F1 (0.464 vs 0.368) now
corrected in `config.py`. It doesn't rescue `call_003`, the one known-call
miss: that clip's competing-frame fraction sits at the 10th percentile of
the positive class on the dev set, meaning nine in ten genuinely overlapping
clips show a stronger signal than this one does. Pushing the threshold low
enough to catch it collapses precision to 0.275 — three in four detections
would be false. That's a real ceiling on a weak instance, not a
miscalibration, and the actual fix is the pyannote backend above, not
further threshold tuning.

Real audio made the picture worse than the synthetic numbers suggested, not
better. `eval/harper_valley_eval.py` runs this system against real calls
from the [Gridspace-Stanford Harper Valley dataset](https://github.com/cricketclub/gridspace-stanford-harper-valley)
— 1,446 simulated bank-support calls with separately-recorded agent and
caller tracks, mixed down to one channel to match this system's actual
input. On a random sample of 25:

| Field | Result | Basis |
|---|---|---|
| `speaker_overlap_present` | 13/25 (0.52) — ground truth matched to this system's own 0.35s minimum-overlap standard; moved only 12→13/25 when re-checked, confirming it's real | independently derived from separate channel timing |
| `emotional_tone`, coarse polarity | 20/25 (0.80) | Harper Valley's own label is 3-class; compared at polarity level only |
| `audio_quality`, MOS-bucketed | 16/25 (0.64) | a weak proxy — `caller_mos` rates intelligibility on clean studio audio, not the defects this system actually targets |
| `background_noise_*` | not tested | Harper Valley has no noise annotation |
| `long_silence_present` | not tested | any derived truth would reuse this system's own logic |

The first read of that 52% was wrong, and catching that mattered more than
the number itself. It looked like a synthetic-to-real gap. A follow-up
sweep against 60 further real Harper Valley calls (`eval/tune_overlap_real.py`,
DSP-only so cheap to run at that size) computed AUC directly: 0.590,
essentially identical to the synthetic dev set's 0.593. The detector's
ranking power transfers to real audio almost exactly. What moves is
accuracy at one fixed threshold on a small sample — the same
0.24–0.26 neighborhood scores 0.617 on the larger sample, which is sampling
variance at this N, not a change in behavior.

So this isn't a synthetic-to-real gap. It's a real, consistent, fundamental
weakness — AUC around 0.59, barely above a coin flip — that shows up
identically on synthetic and real audio alike. The threshold stays at 0.25;
moving it again on this evidence would repeat exactly the kind of
thin-sample chasing this memo argues against elsewhere. The actual fix is
finishing the pyannote integration, which is coded and waiting on one
licence click, not further tuning of a detector that has now been measured
twice and landed in the same place both times.

## Further real-data validation

Two more freely-downloadable, non-circular datasets were tried, following
the same adapter pattern as Harper Valley. One produced a genuinely new
result. The other's headline number turned out not to measure this system
at all, and is reported here as excluded rather than buried.

### AMI Meeting Corpus

`eval/ami_eval.py` uses 2 real four-person business meetings (CC BY 4.0, no
signup), each speaker on an independent close-talk headset with
human-transcribed segment timing, chunked into 27 call-length windows. This
is the first independent real-data check `long_silence_present` has ever
had — Harper Valley's own docstring rules that field out there, since two
channels aren't enough to derive silence without circularity.

`long_silence_present` scored 21/27 (0.778), a genuine result added to the
record: perfect precision (never wrong when it claims a long silence),
reasonable recall, comfortably above a coin flip's 0.519 baseline on this
class balance.

`speaker_overlap_present` also scored 21/27 (0.778), and that number is not
a win. The confusion matrix — 21 true positives, 3 false positives, 3 false
negatives, zero true negatives — shows the detector never once correctly
called a clip clean. Because AMI meeting windows are overlap-heavy (24 of
27 truly overlap), a trivial "always predict overlap" baseline scores
24/27, or 0.889, beating the system outright. This doesn't contradict the
earlier synthetic and Harper Valley findings; it's a third independent
dataset landing on the same weakness.

### MELD — attempted, excluded

`eval/meld_eval.py` draws on 13,000+ hand-labelled utterances of real
actors' dialogue from *Friends* (GPL-3.0). 30 utterances were tested at
coarse tone polarity, with the same care taken as Harper Valley — ambiguous
`surprise` labels excluded, utterances under two seconds dropped. The
result came in below what a naive majority-class guess would score.

A below-baseline number is a reason to look closer, not a number to report
and move past. The "neutral"-truth misses told the story:

```
meld_0000_neutral.wav -> tone: upset, arousal: 0.78
  rationale: "expresses exasperated urging ... supported by high acoustic
  arousal and dominance"
meld_0003_neutral.wav -> tone: frustrated, arousal: 0.686
  rationale: "speaks in a matter-of-fact tone ... without displaying clear
  positive or negative emotion" [the rationale itself says neutral; the
  arousal-based override in app/ser/mapping.py pushed the final tone to
  frustrated anyway]
```

The cause is an acoustic domain mismatch, not a failure of tone judgment.
This system's escalation detection is calibrated against real
customer-service delivery, where a genuinely calm baseline exists. Sitcom
actors deliver even neutral lines with more energy than that baseline —
arousal readings of 0.78 and 0.686 on lines MELD's own annotators called
neutral — and both the LLM and the acoustic override read that as real
activation, because acoustically it is, just not for the reason this
system's calibration assumes. At the aggregate level the same pattern
holds: predictions spread roughly evenly across positive, neutral, and
negative, while MELD's truth is mostly neutral. The system can't find
MELD's neutral because MELD's neutral doesn't sound like the neutral it was
calibrated on.

This isn't scored as a tone-capability result and isn't held against the
system — the same treatment already given to RAVDESS/CREMA-D (circular) and
IEMOCAP/CHiME (inaccessible). Reporting the raw MELD number without this
context would misrepresent a domain mismatch as a capability finding. The
script and its diagnosis both stay in the repo, because the exclusion
reasoning is itself worth keeping — a concrete example of checking whether
a below-baseline score reflects the architecture or the dataset before
trusting either conclusion.

## Built, measured, and not shipped

### Audio quality: an NISQA + DNSMOS ensemble, rejected

`app/audio/quality_mos.py` wraps both non-intrusive MOS models via
`torchmetrics`. Wiring it in surfaced two real bugs: NISQA raises past a
45–60s single-call duration limit, found by binary search rather than
assumed, and both models are stateful `torchmetrics.Metric` objects that
need resetting after every use in a batch context.

Measured on 150 dev-set clips, grouped 3-fold CV (`eval/train_quality.py`):

| | macro-F1 |
|---|---|
| shipped heuristic, hand thresholds | 0.615 |
| heuristic, fitted | 0.654 |
| heuristic + NISQA/DNSMOS, fitted | **0.451** |

The ensemble made things worse. The likely cause: NISQA and DNSMOS were
trained for general VoIP speech quality, not this dev set's specific
synthetic defects — clipping, dropout, reverb decay, vocoder artefacts —
so their scores are close to noise for this task, and a 3-group GroupKFold
is small enough that the fitted classifier overfits to it. The module and
its eval script stay in the repo as a working, documented negative result,
not wired into the decision path.

### SER: emotion2vec+ as a second opinion, shipped

This started as "validated, not yet wired," then got wired into
`app/ser/mapping.py.reconcile()` after root-causing why `emotional_tone`
scored 0/3 on the known calls. Both fixes were diagnosed from actual
diagnostics rather than guessed, and validated across three independent
trials per call — this codebase's own provider docstring already documents
that LLM tone answers aren't stable run to run even at temperature 0, so a
single successful run proves nothing.

The first fix promotes `frustrated` to `upset` on call_001. The LLM reads a
transcript that repeats "Hello?" six times and calls it `frustrated` — a
defensible literal reading — while the dimensional escalation score, 0.79,
falls just short of the 0.85 threshold that would push it further.
emotion2vec+, scored independently on the same audio, answers `angry` at
0.86. Rather than lower the threshold to fit one clip, a narrowly-scoped
rule promotes `frustrated` to `upset` only when the categorical model agrees
and the dimensional model doesn't disagree. All three trials landed on
`upset` through this exact rule.

The second fix withdraws a false `upset` on call_002. The LLM calls this
`upset`/`frustrated` because the transcript contains one transcribed
profanity aimed at an IVR — there's no other negative content. Both
acoustic models disagree: emotion2vec+ reads `neutral` at 0.9999, and the
dimensional model's valence sits above the positive threshold. An existing
withdrawal rule already covered "calm and acoustically positive," but
required escalation at or below 0.10 — calibrated for an unambiguously
quiet clip, and this one isn't quiet, just not angry. A second rule now
fires on convergence: two independently trained models both saying "not
angry" is stronger evidence than either alone. All three trials fired it
correctly.

The real risk here, stated rather than buried: this can't distinguish
someone genuinely furious but quiet, or sarcastic, from someone actually
neutral. Both sound the same to emotion2vec+. No signal in the system
currently separates them.

| | before either fix | after both fixes |
|---|---|---|
| `emotional_tone` | 0/3 (0.00) | **2/3 avg (0.67)** — call_001 3/3, call_002 3/3, call_003 0/3 |
| all fields, averaged | 0.792 | **0.875** |

`call_003` is unchanged, and deliberately so. Every available signal — LLM
text, dimensional acoustic, categorical acoustic — points away from
`satisfied`: emotion2vec+ says neutral, valence sits in the unclear band,
and the transcript itself, a customer calmly navigating scheduling
conflicts with one perfunctory "thank you," reads as polite rather than
clearly positive under this system's own definition. Writing a rule to
force this one case would be fitting a heuristic to a single example with
no corroborating evidence, exactly what four earlier prompt variants
already proved backfires. It stays open and disclosed.

### Tone provider replaced: Gemini and Groq out, Azure OpenAI in

Gemini's free tier caps at 18–20 requests per model per day, which one
evaluation batch exhausts before anyone sees a result — the reason the old
chain rotated through three Gemini models and fell back to Groq at all.
Groq turned out to be broken independent of quota: its configured model,
`llama-3.3-70b-versatile`, had been fully retired from Groq's catalog, and
every call was 404ing. Confirmed by forcing the chain to Groq-only and
reading the live exception, not assumed from a changelog. Both providers
are gone from the tone chain now; Azure OpenAI is the sole remote tone
provider, addressed by deployment name. Groq keeps its other job,
transcription, 38x faster than local Whisper, since that was never broken.

This isn't trading one quota problem for another. Azure OpenAI bills per
token with no daily request ceiling, so a batch can't exhaust a free-tier
budget mid-run the way the Gemini chain did — that was the actual
motivation, not a hoped-for accuracy gain.

Two candidate deployments, `gpt-5-mini` and `gpt-4.1-mini`, were compared
against the known calls before picking one. The first comparison showed
`gpt-4.1-mini` a full point ahead, which turned out to be a broken local
environment — `funasr` and `torchaudio` were both silently missing, so
emotion2vec+ corroboration wasn't actually running for either candidate.
With both dependencies fixed, the two deployments score identically: 22/24,
matching the earlier Gemini-based result exactly, down to the same two
disclosed misses. `gpt-5-mini` shipped as the default on tie-breaking
grounds — roughly 2.5x the quota headroom and a lower per-token cost on
this Azure resource, not an accuracy difference.

Six independent trials, cache off, confirmed it wasn't luck:

```
FIELD                     call_001   call_002   call_003    AVG
emotional_tone                1.00       1.00       0.00   0.67
(all other fields 1.00 across all three calls)
ALL FIELDS AVERAGE ACCURACY: 0.917  (22/24)
```

The tone answers were identical across all six trials — no variance at
all, which is more consistent than Gemini's own documented behavior, where
one labelled clip returned three different answers across seven runs at
temperature 0. Six trials on three clips isn't enough to claim `gpt-5-mini`
is inherently more deterministic than Gemini in general; it's evidence of
what this specific configuration does.

Cost was re-measured against live billing rather than left stale. The
earlier $0.00092/audio-minute figure was Gemini-specific pricing that no
longer applied. `gpt-5-mini` is a reasoning model that spends real tokens
on hidden reasoning before the visible answer — 64 reasoning tokens on a
two-word test reply, 192 to 576 on the real prompts below — which changes
the cost shape even though the prompt itself didn't change. Real usage was
captured by wrapping the live Azure OpenAI client and reading actual
production calls: 2,677 prompt tokens and 1,645 completion tokens across
the three known calls, at current `gpt-5-mini` pricing. That comes out to
$0.00093/audio-minute for the tone call, and $0.00160/audio-minute total
with ASR — almost identical to the old Gemini-era figure, which is a
coincidence confirmed by measurement, not an assumption carried forward.

### A bias the provider swap introduced, and fixed

Re-running the external-dataset checks against `gpt-5-mini` surfaced a
regression the known-calls tests couldn't see: `emotional_tone` on the 25
Harper Valley calls dropped to 0.40 (10/25), down from 0.80 under Gemini.
Root-causing it rather than just re-measuring it found that 13 of 15 errors
— 87% — were the identical pattern: truth neutral, predicted satisfied.
This isn't a guess at the cause. LLMs over-predicting positive labels on
neutral content is a published, GPT-specific finding; bias-correction
studies attribute roughly 9.5% error to exactly this pattern.

The fix mirrors the existing negative-tone withdrawal rules structurally:
`satisfied` now requires the dimensional model's measured valence to
actually corroborate a positive reading before it's trusted, and
unsupported cases withdraw to `neutral`. The known-calls total didn't move,
which is expected rather than a sign the fix does nothing — none of the
three calls trigger this pattern, and the fix targets a failure mode that
only shows up at real-world scale. On the same 25 Harper Valley calls,
accuracy went from 0.40 to 0.84.

The risk here is real and stated plainly: this rule can't tell someone
genuinely pleased but acoustically flat — reserved gratitude, positivity
carried only in the words — from someone the acoustic signal reads as
unclear or negative. That edge case gets incorrectly withdrawn to neutral,
and nothing in the system currently separates the two. The 87%-of-errors
evidence for shipping this fix is stronger than the unmeasured cost of that
edge case, but it's a real trade, not a free win.

### External service disclosure

| Service | Model | Used for | Data sent | Retention |
|---|---|---|---|---|
| Azure OpenAI | `gpt-5-mini` (via `AZURE_OPENAI_DEPLOYMENT`) | emotional tone only | transcript + numeric measurements, never raw audio | retained up to 30 days for automated and human abuse monitoring, stored within the Azure region, inaccessible to OpenAI or other Microsoft teams, never used to train a model (Microsoft's default, not an opt-in). "Modified abuse monitoring" or Zero Data Retention are available to AutoAce on an Enterprise Agreement if 30 days is too long |
| Groq | `whisper-large-v3-turbo` | transcription | the audio file | states it does not train on API data |

Transcription uploads the audio, enabled here for a 38x latency win and
gated behind an explicit `ALLOW_ASR_UPLOAD` flag, specifically so a mode
called "audio never leaves" can't quietly upload it anyway.
`PRIVACY_MODE=local_only` removes all external calls; `hybrid` with the
flag off keeps audio local and sends only derived text and numbers.
Uploaded files are deleted the moment a batch completes. Cerebras was also
evaluated as a third tone provider in the first iteration and returned
`402 Payment Required` on a new account — its free tier doesn't cover chat
completions, so the option isn't worth re-investigating.

### NVIDIA NeMo Sortformer, tested thoroughly, rejected

Looking for an ungated alternative to pyannote led to
`nvidia/diar_sortformer_4spk-v1` — Apache-2.0, confirmed not gated, and
fast enough for CPU inference. It's a current NeMo diarization model that
handles overlap natively, since each output segment carries a speaker
label and cross-speaker time intersection is overlap by construction.

On the three known calls it went 3/3, which is exactly the trap this
project has learned not to trust. Validated properly instead:

| Test | Result |
|---|---|
| 150-clip synthetic dev set | AUC 0.831, accuracy 0.760 — a large win over the cepstral fallback's 0.593/0.633 |
| 30 real Harper Valley calls | accuracy 0.333 — worse than the fallback's 0.52 on a comparable real sample |

That's the reverse of the usual synthetic-to-real story, where synthetic
numbers flatter the system. Here the model got dramatically worse moving to
real audio, so the gap got investigated rather than just reported. The
confusion matrix on real audio showed large false-positive overlap spans in
calls with no real overlap, alongside several true overlaps missed
entirely. The first hypothesis was speaker over-segmentation — the model
detected three "speakers" in a few real two-participant calls — so it was
re-run constrained to a maximum of two speakers, a legitimate domain
constraint since both participants are always known in advance. The result
didn't change: accuracy 0.333, identical confusion matrix, and only two of
the thirty calls had actually shown three detected speakers before
constraining. The real issue looks like something specific to how this
model handles compressed 8kHz telephony audio, which its training set
apparently doesn't cover well enough to transfer. A telephonic-specific
variant might do better, but that architecture isn't present in the
currently installed NeMo release, and chasing it further was outside this
iteration's budget.

It didn't ship. It's left as a documented dead end so the next attempt
doesn't rediscover the same trap from the same synthetic spot check. The
pyannote backend remains the correct fix, still one licence click away.

## Cost and latency, recalculated

The first iteration's cost table predates PANNs and emotion2vec+, so both
were re-measured against the three known calls with real instrumentation.

API cost lands at $0.00160/audio-minute, effectively identical to the
first iteration's $0.00159. Both new stages this iteration, PANNs and
emotion2vec+, are local models with no metered call, so neither moved the
figure. What moved, later, is the pricing behind it: the tone LLM switched
from Gemini to Azure OpenAI, and this number is the real re-measured
figure — $0.00093/minute tone LLM plus $0.00067/minute ASR — not the old
Gemini number carried forward unchecked. That it lands almost exactly
where the Gemini figure did is a coincidence, confirmed by measurement.

Latency did move. Steady-state processing time, warm process, excluding
one-time model load, went from 3.0s/audio-minute to roughly
5.9s/audio-minute, because PANNs and emotion2vec+ both add real wall-clock
time at zero API cost. Billed as rented compute rather than run on owned
hardware — the EC2 path below, at $0.067/hr — that adds roughly
$0.00011/minute, for a fully-costed total of about $0.00171/minute, still
1.75x under the $0.003 ceiling, with less margin than the API-only framing
suggests.

The confidence-gated noise-type fix bought back some of that latency: the
DSP/acoustics stage dropped from 0.86s/min to 0.31s/min, because PANNs is
now skipped whenever spectral-only is already confident — a real,
attributable win from an accuracy fix, not a separate optimization pass.
Total steady-state latency barely moved, since other stages have their own
run-to-run variance that swamps this one stage's savings, but memory did:
peak RSS dropped to 3.2GB, down from a 5.3GB checkpoint mid-iteration,
because PANNs' model is no longer loaded at all in the common case where
spectral-only clears the gate on its own.

## Infrastructure: Azure Container Apps, EC2, and HF Spaces

Per-invocation serverless was priced out and rejected explicitly: a
monolithic Lambda running the whole pipeline costs roughly
$0.0045/audio-minute in compute alone, already over the ceiling before the
LLM call, and even a right-sized split-per-Lambda design re-pays model-load
cost on every invocation. `infra/README.md`, `infra/start.sh`, and
`infra/stop.sh` add a start-on-demand, stop-when-done EC2 Graviton path for
the offline batch-run case specifically — one warm process, models loaded
once, no idle billing. `Dockerfile` bakes the PANNs CNN14 checkpoint into
the image at build time via `curl`, since `panns_inference`'s own
downloader shells out to `wget`, which the slim base image doesn't have —
caught by actually trying to load the model, not assumed to work.

The hosted dashboard itself runs on Azure Container Apps, not EC2 or HF
Spaces — Consumption plan, `min-replicas=0`, the same scale-to-zero logic
applied to the always-on dashboard case instead of the batch-run case. The
image is cross-compiled locally for amd64, since a plain `docker build` on
this Apple Silicon machine produces an arm64 image that Azure's
Consumption plan can't run, and pushed to ACR because this subscription
tier restricts remote builds. The same tier restricts which regions are
usable at all; `eastus` isn't one of them, so the app runs in `eastasia`
instead, confirmed via policy lookup rather than trial and error. It's
sized at 4 CPUs and 8GB after a real out-of-memory crash under concurrent
batch load showed that two clips' models can be resident at once. See
`infra/README.md` and `README.md`'s "Live deployment" section for the URL
and exact deploy commands.

## End-to-end result on the three labelled calls

Three states, each measured over three independent trials with caching
off:

```
                          call_001         call_002         call_003
tone, before any fix      0/3 upset        0/3 neutral      0/3 satisfied
tone, after fix 1 only    3/3 upset ✓      0/3 neutral      0/3 satisfied
tone, after fix 1+2       3/3 upset ✓      3/3 neutral ✓    0/3 satisfied
```

Every other field held steady across trials: intensity, noise-present,
audio quality, and long-silence all passed on all three calls; overlap
missed only on call_003, the disclosed weak spot. Noise-type missed on
call_002 at the time these trials ran — the PANNs regression described
above — and is 3/3 after that fix, no longer dependent on tone-trial order
since it involves no LLM at all.

Average field accuracy across three trials moved from 0.833 to 0.875 after
both tone fixes, and `emotional_tone`'s own average moved from 0.00 to
0.67. Neither improvement is a lucky single run — the exact reconciliation
rule fires and is logged in every trial it applies to. With the noise-type
fix layered on top, the current known-calls total is 22/24. `call_003`
stays wrong on both remaining fields, honestly: no signal in the system —
text, dimensional, or categorical — supports `satisfied`, and the
cepstral overlap detector's ~0.59 AUC ceiling is real.

## Next steps, in priority order

The first iteration's own next-steps list is now partly stale — its top
item, a paid tone key, is done. What's actually still open:

1. **The pyannote overlap backend.** Already built and correct for
   pyannote.audio 4.0, blocked only on one manual licence click. This is
   the highest-value, lowest-effort item left, since overlap is the
   system's weakest field by a wide margin on every real dataset tested.
2. **More labelled real production data.** Still the binding constraint on
   almost everything else here. Every dev-set threshold and fusion rule
   was tuned against three known calls plus synthetic data, and the tone
   fixes above are validated on three to six trials over three calls, not
   the hundreds a hidden test set will actually contain.
3. **A corroborating signal for genuinely flat-but-positive callers.** The
   `satisfied`-withdrawal fix trades away detection of reserved, text-only
   positivity — a calm "thank you, that's all I needed" — because nothing
   in the system currently distinguishes that from a miscalibrated LLM
   guess; both look acoustically neutral. Closing that gap needs either
   labelled examples of exactly this pattern or a text-sentiment signal
   independent of the dimensional model's valence.
4. **WavLM as a real overlap fix**, not the pyannote-alternative it was
   evaluated as. Researched, built, and validated end to end this
   iteration — dev-set AUC 0.677 against the shipped detector's 0.593 —
   but not shipped, because every threshold tested regressed real accuracy
   on the AMI 2-speaker domain specifically (see `result.md`). Worth a
   second look with per-domain threshold calibration, or ensembling with
   the cepstral signal instead of a straight swap, once item 1 is also
   available for comparison.
5. **Diarization-based recalibration** for overlap and silence, against
   customer-only audio. Correct in principle, still blocked on the same
   thing the first iteration found: nothing to recalibrate on without a
   real per-speaker channel, since the provided calls are dual mono.

---

# Part 2 — The first iteration

This is where the architecture above actually came from. Two complete
systems were built, and the second was built specifically to test a
diagnosis of why the first one failed. Everything below predates the work
in Part 1 — some numbers here have since been corrected or re-measured
above.

## Summary

The first attempt took the obvious path: a multimodal LLM reading the
audio and returning all nine fields in one structured call. It was built,
deployed behind a dashboard, tested — and scored 10 of 24 fields on the
labelled calls, while reporting 0.85 to 0.95 confidence on almost every
wrong answer.

That failure got diagnosed rather than patched. A second system was built
to test the diagnosis, inverting the design: measure everything that's
physically measurable, and ask a model only what genuinely requires
judgment. It scored 22 of 24, and everything running today descends
directly from it.

| | Architecture A — LLM-first | Architecture B — measure-first |
|---|---|---|
| Design | one multimodal call returns all 9 fields | 8 fields measured, 1 inferred |
| `emotional_tone` | 1/3 | **2/3** |
| `emotional_intensity` | 0/3 | **3/3** |
| `background_noise_present` | 1/3 | **3/3** |
| `background_noise_type` | 1/3 | **3/3** |
| `background_noise_severity` | 1/3 | **3/3** |
| `audio_quality` | 2/3 | **3/3** |
| `speaker_overlap_present` | 1/3 | 2/3 |
| `long_silence_present` | 3/3 | **3/3** |
| **Total** | **10 / 24** | **22 / 24** |
| Marginal cost | metered on every field | 8 of 9 fields at $0 |
| Reproducible | no — same input, varying output | 8 of 9 fields byte-identical |

Across both architectures, this iteration produced roughly 10,700 lines,
36 test files, and 8 distinct approaches — four of which were rejected on
evidence after being built.

## 1. What the data showed before any modeling

Three properties of the provided clips shaped everything that followed,
and each one ruled out an approach that would otherwise have looked
reasonable.

The recordings are dual mono. Left and right correlate at 1.0000, with a
maximum sample delta of 0.004 — there's no agent/customer channel to
difference, so overlap and speaker separation have to be recovered
acoustically from the mixture. Any design assuming stereo call recording
was dead on arrival.

Background noise is episodic, not a continuous bed. Every clip's
inter-word gaps are digitally silent — noise floors between −80 and −240
dBFS — yet two of the three are labelled noisy, because the noise arrives
in spans rather than sitting under the whole call. In `call_003`, the
high-band energy ratio sits at 0.003–0.005 through clean speech and spikes
to 0.15–0.21 at three specific moments.

A whole-clip SNR detector scores zero as a direct consequence. Clip-level
SNR estimates come out pristine — 47 to 55 dB for all three files — and
miss both positives entirely. Detection has to be windowed and aggregated,
with an absolute audibility gate rather than a relative one: `call_001` is
labelled noise-free despite carrying high-band spikes, because those
spikes sit at −71 dBFS, below the threshold of audibility. The brief's own
wording anticipates exactly this case.

One more thing shaped how much weight the labels could carry at all: all
three reference labels report confidence 0.82, exactly. That's the
signature of a single automated labelling pass with a hardcoded value, not
human adjudication — so some of what looks like a convention to learn is
probably label noise, and fitting hard to three points means fitting to
that noise too.

## 2. Architecture A — LLM-first, and why it failed

The first implementation was a complete, production-shaped system:
password-gated dashboard, ZIP upload with manifest validation, a bounded
worker pool with per-file isolation, token-bucket rate limiting, retry
with full jitter, windowed analysis for long clips, an ensemble layer that
re-sampled only ambiguous cases, and 110 offline tests — about 4,400 lines.

The bet was that a modern multimodal model, given field definitions and a
strict schema with enums enforced at generation time, would outperform
anything assembled from parts. With no training data available, zero-shot
quality was the whole question.

It scored 10/24, and the shape of the errors mattered more than the
number. The model defaulted to the "quiet" class on nearly every field —
neutral, low, none, false — regardless of content. It scored 3/3 on
`long_silence_present` by answering false every time, which happened to
match; every case where the truth was anything but the default class, it
missed.

Three follow-up experiments ruled out the easy explanations. Re-running
the hardest clip at a higher reasoning budget produced byte-for-byte
identical output — more thinking changed nothing. A free-text prompt asking
the model to simply describe what it heard also returned "neutral tone, no
background noise, no overlap," ruling out a structured-output artifact.
And the model's self-reported confidence — 0.90, 0.85, 0.95 across the
three clips — was not just uncalibrated but anti-correlated with
correctness.

The conclusion was specific and testable: the model wasn't failing to
reason about the noise, it was failing to perceive it. A deterministic
measurement doesn't have that failure mode, because it never needs to
recognize static as unusual — it measures signal energy directly.

That prediction is what Architecture B was built to test.

## 3. Architecture B — measure first, infer last

The nine fields aren't one problem. Six are physical properties of a
waveform. Two are properties of a voice. One is a judgment about what a
person meant.

| Field | Decided by | Cost | Deterministic |
|---|---|---|---|
| `background_noise_present` | windowed local dynamic range + audibility gate | $0 | yes |
| `background_noise_type` | random forest over the estimated noise spectrum | $0 | yes |
| `background_noise_severity` | affected share of the call × how badly | $0 | yes |
| `audio_quality` | clipping, band edge, level, dropout, reverberation | $0 | yes |
| `speaker_overlap_present` | cepstral pitch competition | $0 | yes |
| `long_silence_present` | inter-speech gap analysis | $0 | yes |
| `emotional_intensity` | wav2vec2 arousal, regressed from the waveform | $0 | yes |
| `emotional_tone` | language model, reconciled against measured affect | metered | no |
| `confidence` | computed from cross-signal agreement | $0 | yes |

This split does four things, roughly in order of importance. It fixes the
exact failure Architecture A showed: a detector that measures the noise
floor directly cannot fail to hear static. It structurally prevents the
two failure modes the brief calls out — a model never asked about
background noise can't infer noise from poor audio quality, and intensity
taken from measured arousal can't be inferred from loudness, because
neither path exists in the code. It makes eight of nine fields
reproducible, same input to same bytes out, every time. And it collapses
cost, since eight fields carry no marginal charge at all, which is why the
$0.003/minute ceiling ends up comfortable rather than tight.

### 3.1 Why emotion is measured, not read

A transcript can't carry delivery. `call_001` transcribes as "Come on. …
Hello." repeated eleven times — routine on the page, audibly frustrated to
anyone listening. A language model reading that transcript answered
neutral; the reference label is upset.

Describing prosody to the model in words made it worse. Told "unsteady
voice, noticeable pitch tremor," it restated those adjectives back as a
diagnosis of distress regardless of content. Rewriting the description as
bare numbers with reference ranges stopped that — and then the model just
answered neutral to everything.

So the fix was to stop describing and start measuring.
`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` regresses arousal,
dominance, and valence directly from the waveform. Dimensional output was
chosen over categorical on purpose: four-class emotion models can't
express frustrated versus upset versus distressed at all, so no amount of
accuracy would let them serve this schema.

Each dimension is then trusted only where it's actually competent.
Arousal drives `emotional_intensity`, because activation is carried by
pitch, energy, and rate, and acoustic models predict it well — on the
labelled clips, arousal ranks the three calls in exactly the ground-truth
intensity order. Dominance separates `upset` from `distressed`, since both
are high-arousal negative states and the real difference is control: an
angry caller is assertive, an overwhelmed one isn't, and no transcript
conveys that distinction. Valence stays corroboration only, because
whether something reads as positive or negative lives mostly in the words,
valence is the weakest dimension for any acoustic model, and on these
clips it ranks them wrongly — polarity stays with the language model.

The result: `emotional_intensity` went from 0/3 under Architecture A, to
1/3 with transcript-only description, to 3/3 once it was actually measured.

### 3.2 Final shape

```
audio ──▶ decode 16 kHz mono
            ├──▶ signal processing ─────────▶ 6 acoustic fields      [free, deterministic]
            ├──▶ wav2vec2 speech emotion ──▶ arousal/dominance/valence   [free, LOCAL model]
            └──▶ Whisper ──▶ transcript ──▶ language model ──▶ tone polarity   [metered API]
                                                    │
                                    fusion ◀────────┘
                                      │ arousal sets intensity
                                      │ dominance splits upset/distressed
                                      │ language model sets polarity
                                      │ agreement between them sets confidence
                                      ▼
                              9-field schema, enum-validated
```

Three lanes that never consult each other after decoding. Two measure, one
infers, and they meet only at fusion, only on the emotional fields. This
shape hasn't changed since — everything in Part 1 builds on top of it
rather than replacing it.

## 4. Local models versus hosted APIs

This system runs both a local neural model and hosted APIs, and the split
is a deliberate engineering decision, not a default worth glossing over.

What runs locally, and why:

| Component | Model | Why local |
|---|---|---|
| Speech emotion | `wav2vec2-large-robust-12-ft-emotion-msp-dim` (317M params, int8) | Arousal and dominance aren't available from any hosted API in the form this schema needs. Running it locally also keeps the audio in-process for this stage, at zero marginal cost |
| All DSP | hand-built (VAD, noise estimation, quality, overlap, prosody) | Deterministic, auditable, free — and, per §2, measurably better than asking a model |
| Noise-type classifier | random forest trained on a generated dev set | 12 features, 400 trees, ~50KB. No API could be trained on this label vocabulary |
| Tone fallback | lexicon + prosody heuristic | Guarantees the system never fails outright, and makes `local_only` a real mode instead of a claim |

A second local SER backend, `tiantiaf/wavlm-large-msp-podcast-emotion-dim`
— the WavLM architecture that won the 2024 MSP-Podcast A/D/V challenge —
was also integrated and benchmarked this iteration. Its published
checkpoint ships only weights and expects a wrapper class that isn't on
PyPI, so the architecture was reconstructed from the checkpoint's own
tensor shapes. It runs, and §6.8 documents it as a measured
non-improvement. (A separate WavLM investigation, for `speaker_overlap_present`
rather than SER, came later and is covered in Part 1 and in `result.md`.)

What runs against an API, and why:

| Component | Service (at the time) | Why not local |
|---|---|---|
| Emotional tone | Gemini, since replaced by Azure OpenAI — see Part 1 | Genuinely a language-understanding problem. The local lexicon heuristic scores far worse, and a locally-hosted model good enough to match would need at least 7–8B parameters |
| Transcription | Groq `whisper-large-v3-turbo` | Local `faster-whisper small` runs the same clip in 21s against 0.55s — a 38x difference — for 460MB of resident memory |

### 4.3 The deployment economics that decided it

Swapping the hosted LLM call for a self-hosted instruction-tuned model is
entirely achievable, and the code already supports it — `PRIVACY_MODE=local_only`
needs no code change. It wasn't chosen, for a reason specific to this
engagement.

A 7–8B model at reasonable quality needs roughly 7–8GB of VRAM, which
means a GPU node, and for a system deployed and exercised during
evaluation rather than run continuously, that's the wrong shape on every
axis. A GPU node costs $0.50–1.20/hour and has to stay warm continuously,
which is tens of dollars for a handful of evaluation batches against
roughly $1.60 per thousand audio-minutes on the API path. Cold-starting a
7B model from object storage takes minutes, long enough that an evaluator
opening the dashboard would hit a timeout instead of a result. No free
tier hosts a GPU node at all. And the accuracy would very likely be worse
anyway — a 7B open model isn't competitive with a frontier model on a
five-class emotional judgment with definitional boundaries this fine.

The trade-off, stated plainly: for a permanently-running deployment
processing thousands of calls a day, the economics invert, and the local
path wins on both cost and privacy. For a system exercised at evaluation
volume, the API path wins on every axis that matters here. The
architecture is built so switching between them is a configuration
change, not a rewrite, because which one is correct depends entirely on
volume — which is why 8 of 9 fields already run on local, deterministic
code, and only the one field that genuinely needs frontier-model language
understanding is metered.

## 5. First-iteration results

Measured on the three labelled clips, and on a 600-clip synthetic dev set
with splits grouped by speech source to prevent leakage.

| Field | Labelled clips | Dev set macro-F1 |
|---|---|---|
| `background_noise_present` | 3/3 | 0.764 |
| `background_noise_type` | 3/3 | 0.798 |
| `background_noise_severity` | 3/3 | 0.480 |
| `audio_quality` | 3/3 | 0.615 |
| `long_silence_present` | 3/3 | 1.000 |
| `emotional_intensity` | 3/3 | not validated |
| `speaker_overlap_present` | 2/3 | 0.555 |
| `emotional_tone` | 2/3 | not validated |
| **Total** | **22/24** | |

`emotional_tone` was measured by majority vote over repeated runs, because
a single sample isn't stable — over seven runs, one clip returned `upset`
five times, `frustrated` once, and `neutral` once. It scored 2/3 on the
primary model tier and 0/3 once that model's free daily quota was spent
and the chain fell back, which the dashboard stated explicitly when it
happened. Re-running this exact code fresh at the start of the second
iteration, before any change, reproduced 20/24 instead — see Part 1's
opening section for that correction.

Three labelled clips can't support cross-validation, and the brief
explicitly forbids reporting accuracy from training data alone. So a
600-clip dev set was generated by controlled mixing, where ground truth is
known by construction: noise injected at known SNR and coverage,
degradations at known severity, silences of known length, second speakers
mixed at known overlap. Splits are grouped by source speaker so no speaker
appears on both sides of a split, and the noise templates for the two
categories present in the labelled data were measured from those clips
rather than invented — §6.5 explains why that mattered.

Cost came in at $0.00159 per audio-minute at paid list price, 1.9x under
the ceiling, and $0.00 as actually deployed. Latency ran 3.0s per
audio-minute for the free half of the pipeline plus roughly 5s per clip
for the metered call, 9 to 13s end to end. Memory sat at 1.2GB resident,
2.4GB peak.

## 6. Everything tested in the first iteration

Eight distinct approaches were built and measured; four were rejected
after being built. They're recorded here because the rejections carry as
much information as the adoptions.

**6.1 LLM-first, all nine fields.** 10/24, confident wrong answers,
perception failure confirmed by three follow-up experiments. Rejected, and
the reason drove the redesign.

**6.2 Transcript plus prose prosody for tone.** 0/3 — the model parroted
the adjectives back as a diagnosis. Rejected, replaced with direct
measurement.

**6.3 Model tier for tone.** The smaller, faster tier answered neutral to
everything and scored 0/3; the identical prompt on the standard tier
scored 2/3. Considerable time went into tuning the prompt when the model
itself was the bottleneck. Adopted the standard tier as primary, with the
smaller tier as a last-resort fallback.

**6.4 Sending audio directly to the tone model.** Produced identical
answers to sending only the transcript, at every model tier tested.
Rejected — transmitting customer audio buys no measurable accuracy, which
settles the privacy question on evidence rather than principle.

**6.5 A fitted noise-type classifier.** Hand-weighted rules collapsed six
of eight categories to 0.157 macro-F1. A random forest scored 0.69 but got
both real noisy clips wrong, because it had learned an invented idea of
what television sounds like. Retraining on noise templates measured from
the real clips recovered 3/3 while holding 0.798 under grouped CV. Adopted,
after the first version was rejected.

**6.6 A fitted severity classifier.** 0.75 on the dev set, 1/3 on the real
clips; hand rules scored 0.40 and 3/3. The dev set can't adjudicate here,
since its severity labels are self-thresholded rather than ground truth
for how AutoAce actually grades interference. Rejected; hand rules ship.

**6.7 Speaker diarization.** Correct in principle — the schema asks about
the customer, and half of each recording is the agent. Measured, it made
things worse: the agent is the higher-arousal speaker, a bright TTS voice
against a frustrated human, and thresholds fitted on whole-clip audio
stopped applying. Rejected for now, implemented behind a flag, with the
recalibration it needs documented rather than ignored.

**6.8 Trajectory features and the WavLM SER backend.** Both published
elsewhere as improvements; both measured neutral to negative here.
Trajectory features pushed the `satisfied` clip further from its label,
because all three calls end more negatively than they begin. WavLM
produced the same ranking on a shifted scale that these thresholds don't
fit. Both ship behind flags, off by default.

Two overlap detectors were also built and measured at AUC 0.50 and 0.48 —
chance. The cause is frequency resolution: 25ms frames give 40Hz bins, and
two talkers' fundamentals routinely sit closer together than that. The
better of the two shipped, honestly weak, and `pyannote/segmentation-3.0`
was wired in behind a flag from the start — see Part 1 for where that
stands now.

## 7. Engineering for a zero-budget deployment

The free-tier constraint produced real engineering, not just compromises.

The free Gemini tier metered per model per day, and metered the good model
hard — 20 requests, which one evaluation batch would exhaust before anyone
saw a result. Because the quota was scoped per model, the client rotated
through a configured chain, and a 429 burned that model's bucket
immediately so no further round-trips got wasted on it. This is moot now
that Azure OpenAI is the sole tone provider, but the rate-limiting
infrastructure it produced is unchanged and still in use.

Tone answers are unstable, and majority voting fixes that — but voting on
every clip triples the cost of the only metered call and breaches the
ceiling on short files. So extra samples are drawn only when the first
answer looks shaky: low self-reported confidence, or an answer the
acoustic measurement contradicts.

Token buckets per provider enforce request-rate and concurrency limits
below what's published, so the system throttles itself rather than
discovering limits through errors, and the dashboard shows the wait
explicitly rather than hanging silently. When quota does run out, the
batch doesn't fail — it falls back and says so, in the dashboard and
attached to every affected result. A malformed clip fails alone with a
stated reason rather than taking the batch down with it, verified with
corrupt and empty files. And content-hash caching means re-running a batch
costs nothing.

## 8. First-iteration limitations

Stated plainly, because these are exactly what motivated the work in Part
1 above.

Three examples can't validate a five-class judgment, and that's the honest
headline behind everything that came after. Eight changes were measured in
this iteration; four looked principled and made things worse or made no
difference. The one change that clearly worked — measuring affect for
intensity — worked because it closed a structural gap, not because it was
tuned to fit the data.

`long_silence_present` had an unconstrained threshold. All three labelled
clips are negatives, and the largest genuine internal gap — 7.31s of real
dead air in `call_003` — is still labelled false, so the threshold sat
just above that with no positive example to fix its true position. AMI
Corpus, added in the second iteration, is the first independent real-data
check this field has ever gotten.

The synthetic dev set validates the acoustic fields and nothing else. It's
generated audio: it exercises thresholds and spectral discrimination, not
whether a judgment about a person is actually right.

`call_003` was stably mispredicted — seven of seven runs answered neutral
against a `satisfied` label, with both independent signals agreeing with
each other and disagreeing with the reference. Its measured valence was
the lowest of the three clips despite carrying the most positive label,
and this is unchanged in the current system.

Speaker overlap was weak from the start. Two approaches measured at
chance; the one that shipped was the better of two poor options.
Real-audio validation in the second iteration — Harper Valley, AMI Corpus,
AMI 2-speaker — confirmed this is a real, consistent weakness rather than
an artifact of synthetic data.

## 9. Reproducing everything in this memo

```bash
pip install -r requirements.txt
pytest -q                                            # 28 regression tests

python -m eval.synth --out data/devset --per-group 150   # regenerate the dev set
python -m eval.train_noise --devset data/devset          # noise classifier, grouped CV
python -m eval.run_eval --devset data/devset             # per-field metrics + confusion matrices
python -m eval.score_labels requirements                 # score against the labelled manifest
python -m eval.check_requirements                        # compliance against the trial spec
python -m eval.train_noise_panns --devset data/devset     # PANNs-augmented noise-type comparison
python -m eval.repeat_trial --trials 6                    # multi-trial tone stability

docker build -t autoace-voice .
docker run -p 7860:7860 --env-file .env autoace-voice
```

Architecture A lives in its own tree with its own harness
(`eval/validate.py --live`, plus ablation flags for windowing, acoustic
evidence, and cascade sampling) and 110 offline tests of its own.
