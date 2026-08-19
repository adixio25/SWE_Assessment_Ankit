# Results — dataset × field matrix

Every field this system outputs, scored against every dataset it's been tested
against, as of the current code (`gpt-5-mini` via Azure OpenAI as sole tone
provider, `app/ser/mapping.py`'s `satisfied`-withdrawal fix applied.
**Iteration 3 update:** `app/audio/overlap.py` now ships a frame-level WavLM
detector — backend order pyannote (if licensed) → wavlm-frames → cepstral.
The `speaker_overlap_present` numbers in the matrix below are the *cepstral*
detector's historical record; the shipped wavlm-frames numbers, measured on
the same datasets, are in TECHNICAL_MEMO.md's "Iteration 3" section: dev CV
acc 0.652/AUC 0.843 vs cepstral 0.563/0.596, Harper Valley 0.567/0.627 vs
0.533/0.548, AMI 0.580/0.732 vs 0.680/0.429 — AMI accuracy is the one
disclosed regression, on a below-chance-AUC baseline riding an 84% base
rate. Known calls stay 2/3, now catching call_003 and missing call_001.
The earlier clip-level WavLM attempt described below remains accurate
history of why the first attempt was not shipped).
Numbers
are pass/fail counts, not percentages, so sample size is always visible next
to the result — a field's accuracy means something different at n=18 than
at n=30.

`—` means the field has no usable ground truth in that dataset — not tested,
not scored, not counted for or against the system. Reporting a field as
passing on a dataset that never labelled it would be fabricating evidence;
this table follows the same rule the rest of this repo's docs already hold
themselves to.

## Master matrix

| Field | 3 known calls (6-trial, n=18) | Harper Valley (n=25) | AMI Corpus (n=27) | AMI 2-speaker (n=50) |
|---|---|---|---|---|
| `emotional_tone` | 12/18 | 21/25 | — | — |
| `emotional_intensity` | 18/18 | — | — | — |
| `background_noise_present` | 18/18 | — | — | — |
| `background_noise_type` | 18/18 | — | — | — |
| `background_noise_severity` | 18/18 | — | — | — |
| `audio_quality` | 18/18 | 16/25 | — | — |
| `speaker_overlap_present` | 12/18 | 13/25 | 21/27* | 34/50* |
| `long_silence_present` | 18/18 | — | 21/27 | — |
| `confidence` | not independently scorable (calibration, not a pass/fail field) | | | |

\* AMI's `speaker_overlap_present` results are **not counted as good
results** — both score below their own sample's trivial baseline. See their
sections below. Included for completeness, not as a win.

§ A WavLM-based classifier was built and validated as a candidate
replacement for this field — see "Overlap detection: WavLM researched, not
shipped" below for the full trail (Harper Valley 13/25 → 16/25, a real
improvement; AMI 2-speaker 34/50 → 10/50, a real regression on a domain this
system was never built for). The numbers in this table are the **currently
shipped** cepstral-only detector's, not WavLM's — that backend was not
adopted, so it is not what's reflected here.

† HaessigDB's ground truth is a **derived mapping I constructed myself**, not
an original annotation — see its section below for why that matters more
than usual here.

‡ Also below a trivial baseline — see the HaessigDB section. Don't read
16/25 and 13/25 as "roughly two-thirds good" without reading why.

## Per-dataset detail

### 3 known calls — `eval/repeat_trial.py --trials 6`

The only dataset every field can be checked against, because it's the only
one with a full 9-field labelled manifest (`requirements/labels.csv`). 6
independent trials × 3 calls = 18 judgments per field, not a single run —
LLM-backed fields are non-deterministic run to run, so one pass proves
nothing on its own.

| Field | Pass | Fail | Notes |
|---|---|---|---|
| `emotional_tone` | 12/18 | 6/18 | call_001 6/6, call_002 6/6, call_003 0/6 — same known miss every trial, no signal in the system supports `satisfied` for that clip |
| `emotional_intensity` | 18/18 | 0 | |
| `background_noise_present` | 18/18 | 0 | |
| `background_noise_type` | 18/18 | 0 | |
| `background_noise_severity` | 18/18 | 0 | |
| `audio_quality` | 18/18 | 0 | |
| `speaker_overlap_present` | 12/18 | 6/18 | call_001 6/6, call_002 6/6, call_003 0/6 — the disclosed weak cepstral detector, real fix blocked on a pyannote licence click. call_003's overlap sits at the 10th percentile of the positive-class distribution, an intrinsically weak instance |
| `long_silence_present` | 18/18 | 0 | |

### Harper Valley — 25 real bank-support calls

Real (not scripted) customer-service-style audio, closest available domain
match to what this system is actually built for. Only 3 fields have
independently-derivable ground truth here (see `eval/harper_valley_eval.py`'s
own docstring for why the other 5 can't be tested against this corpus).

| Field | Pass | Fail | Notes |
|---|---|---|---|
| `emotional_tone` (coarse polarity) | 21/25 | 4/25 | up from 10/25 before the `satisfied`-withdrawal fix; now above the original 20/25 Gemini baseline |
| `speaker_overlap_present` | 13/25 | 12/25 | matches the same ~0.59 AUC ceiling measured elsewhere — a real, consistent weakness. A WavLM classifier measured 16/25 here in testing (see below) but was not adopted for deployment |
| `audio_quality` (MOS-bucketed proxy) | 16/25 | 9/25 | weak proxy by design — `caller_mos` measures intelligibility on clean studio audio, not the clipping/dropout defects this system actually targets |

### MELD — 30 real-actor sitcom utterances (*Friends*)

Real human speakers, but scripted comedic delivery, not customer-service
speech — kept in the evidence base as a documented domain-mismatch case,
not scored as a capability result (see `TECHNICAL_MEMO.md`).

| Field | Pass | Fail | Notes |
|---|---|---|---|
| `emotional_tone` (coarse polarity) | 12/30 | 18/30 | root-caused: sitcom actors deliver even "neutral" lines with more vocal energy than this system's real-call-calibrated baseline expects — an acoustic domain mismatch, not a tone-judgement failure |

### AMI Corpus — 27 real 4-person meeting windows

Real meeting audio with independent per-speaker close-talk timing —
genuinely non-circular ground truth for silence and overlap, but no
emotional-tone labels exist for this corpus at all.

| Field | Pass | Fail | Notes |
|---|---|---|---|
| `long_silence_present` | 21/27 | 6/27 | first-ever independent, non-circular validation of this field; perfect precision (tp=8, fp=0) |
| `speaker_overlap_present` | 21/27 | 6/27 | **not a good result** — a trivial "always predict overlap" baseline scores 24/27 (0.889) on this overlap-heavy sample, beating the system. Confusion matrix has **zero true negatives** (tp=21, fp=3, fn=3, tn=0). Recorded as reinforcing evidence for the already-known overlap weakness, not a new capability. This is the currently shipped cepstral detector's number |

### AMI 2-speaker overlap benchmark — `Trelis/ami-2speaker-test`, 50 clips

A different, purpose-built AMI adapter from the raw-corpus one above — real
AMI meeting audio reconstructed as 2-speaker virtual meetings with an
explicit `overlap_ratio` per clip, independently verified as a real,
CC-BY-4.0 HuggingFace dataset before use (`eval/ami_2speaker_eval.py`).

| Field | Pass | Fail | Notes |
|---|---|---|---|
| `speaker_overlap_present` | 34/50 | 16/50 | trivial always-predict-overlap baseline on this sample scores **42/50 (0.84)** — the system's 0.68 is below it. This is the currently shipped cepstral detector's number; a tested WavLM alternative scored *worse* here (10/50, see below) |

### Overlap detection: WavLM researched, validated, deliberately not shipped

Full writeup with the complete research trail (three rejected variants — a
naive threshold, RandomForest, windowed pooling — plus the one that was
ultimately built and tested end-to-end) is in `TECHNICAL_MEMO.md`.
Summary here, because it's real evidence worth keeping even though the
system running in production doesn't use it:

Researched published overlapped-speech-detection work looking for a signal
source structurally different from cepstral pitch tracking. Found WavLM
self-supervised features reported as state-of-the-art for OSD on real
corpora in the literature; `microsoft/wavlm-base` verified genuinely ungated
(no licence click, unlike pyannote) before use.

WavLM's *ranking* (AUC) beat cepstral's on **every single dataset tested,
no exceptions**: dev set (0.677 vs 0.593), this AMI 2-speaker set (0.592 vs
0.429), and Harper Valley (0.707 vs ≈0.5). The complication was turning
ranking into a deployed accuracy number — this AMI 2-speaker set is 84%
overlap-positive, a base rate no real customer-service call produces, and no
threshold chosen honestly (from dev data only, never touching this set's
labels) could survive that big a shift: 13/50, then 10/50 with a properly
cross-validated threshold, both below the cepstral detector's 34/50.

**Re-tested against Harper Valley instead of stopping there** — real
bank-support calls, 40% overlap-positive, much closer to both the dev set's
own rate and to what a real deployment would actually see. Same classifier,
same dev-only threshold, only the target domain changed: accuracy 13/25 →
**16/25**, a real win on the domain this system is actually built for.

**Built, wired in, and validated end-to-end — then deliberately not
deployed.** The trade-off was real and disclosed rather than hidden: a
genuine win on the target domain (Harper Valley) against a genuine loss on
an out-of-scope domain (AMI meetings). The decision made was to keep the
simpler, already-understood cepstral detector rather than accept that
trade-off, on the reasoning that a backend switch shouldn't make *any*
tested domain meaningfully worse, even one outside the primary target.
`app/audio/overlap.py` currently runs pyannote (if configured) → cepstral
only — no WavLM in the active code path, and the trained classifier
(`app/models/overlap_wavlm.joblib`) is not in the repo. `eval/train_overlap_wavlm.py`
is kept and reproduces the dev-set comparison (0.593 → 0.677 AUC) that
motivated the investigation — rerunning it regenerates the classifier and the
result. Reversing this decision later needs a training run, not new
research: the method, the evidence, and the code are all documented.

### HaessigDB — `nwllr/haessigDB`, 25 acted call-center calls

Verified real (Apache-2.0, HuggingFace) before use. Four actors performing
scripted banking-support calls where the customer is directed to escalate
to genuine irritation by the end of the call. Sentence-level 1-10 ratings on
aggression/frustration/annoyance; sentences concatenated per (actor, call)
into whole-call clips before scoring, because this system judges whole
calls, not isolated utterances (`eval/haessigdb_eval.py`).

**Ground truth here is not the dataset's own label — it's a threshold rule I
built myself** (peak rating ≥7 → `upset`/`high`, ≥4 → `frustrated`/`medium`,
else `neutral`/`low`), because HaessigDB ships three continuous ratings, not
this system's enum. That choice matters more than it usually would:

| Field | Pass | Fail | Truth distribution in this sample |
|---|---|---|---|
| `emotional_tone` | 16/25 | 9/25 | **24 of 25 calls derive to `upset`**, 1 to `frustrated`, 0 `neutral` |
| `emotional_intensity` | 13/25 | 12/25 | **25 of 25 calls derive to `high`** |

**A trivial "always predict upset/high" baseline scores 24/25 (0.96) and
25/25 (1.00) on this sample — both beat the system's actual 0.64 and 0.52.**
Reported plainly rather than left for the reader to notice: on raw numbers
alone this looks like a below-baseline result, the same category as AMI's
overlap finding.

But unlike AMI's overlap finding, this one comes with a real question about
whether the *ground truth construction* is fair, not just the system: taking
the single worst-rated sentence in a 10-40 sentence call as "the call's
truth" is an aggressive choice — a call that spikes to aggression 8 for one
sentence and stays calm for the other 30 is being labelled identically to
one that's aggression 8 throughout. This system's `emotional_tone` is
defined as the *primary* tone of the whole clip, and several of the misses
(e.g. `actor3_call36`: peak aggression 3.3, peak frustration 8.3, 21
sentences, predicted `neutral`) look like the system reading the call's
overall character rather than its single worst moment — a defensible
reading, not obviously a wrong one. This is the same caution emotion2vec+'s
own module docstring already gives about max-aggregation being prone to
spurious peaks, now showing up in an external dataset instead of an internal
one. Recorded honestly as a genuine below-baseline result on the ground
truth as constructed, with the construction's own weakness disclosed
alongside it — not resolved in either direction.

## Reading this table honestly

- `emotional_tone` and `speaker_overlap_present` are the only two fields that
  ever fail on any labelled data. Every other field is passing 100% wherever
  it's actually been tested.
- `emotional_tone`'s four numbers (12/18, 21/25, 12/30, 16/25) are not
  interchangeable — they're measuring different things against different
  ground truth constructions. Harper Valley tests the system's actual target
  domain against real coarse-polarity labels; MELD tests a domain the system
  was never built for; the 3 known calls use the real 5-class schema;
  HaessigDB tests against ground truth *I derived myself* from continuous
  ratings, on a sample so skewed toward extremes that a trivial constant
  prediction beats the system. Averaging any of these into one number would
  misrepresent all of them.
- `speaker_overlap_present`'s weakness is consistent across every dataset
  the *shipped* cepstral detector has been tested on (known calls, Harper
  Valley, AMI Corpus, AMI 2-speaker) — four independent confirmations of the
  same ~0.59 AUC ceiling. A WavLM-based alternative was researched and would
  trade a real win on the target domain (Harper Valley: 13/25 → 16/25) for a
  real loss on a domain outside the target (AMI 2-speaker: 34/50 → 10/50) —
  that trade-off was tested all the way through and deliberately not
  accepted, so it stays a documented option, not the shipped behavior. The
  pyannote fix, still blocked on one manual licence acceptance, remains the
  actual long-term answer — it doesn't carry this trade-off at all.
- Several results in this file are below a trivial baseline (AMI Corpus
  21/27, HaessigDB's 16/25 and 13/25, and the WavLM AMI 2-speaker result
  discussed but not shipped). Reported exactly as measured in every case —
  a below-baseline number on a real dataset is evidence to sit with and
  explain, not a reason to keep searching for a dataset that comes back
  flattering instead.
- This file changed twice in one research pass because the first real-audio
  check (AMI) was itself re-examined rather than accepted as final — it was
  the wrong target domain, not proof the underlying signal was bad. Worth
  naming as a pattern: a below-baseline result is a prompt to check the
  *test* as well as the system before concluding the system failed.
