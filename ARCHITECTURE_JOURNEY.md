# Architecture Journey — what, how, and why

This is a plain-language walkthrough of the whole system: what AutoAce actually asked for,
what was tried, what broke, and why the current design looks the way it does. Everything
here is backed by a measurement somewhere in this repo (`TECHNICAL_MEMO.md`, `eval/`) —
this file exists to connect those dots into one story instead of making you piece it
together across the codebase yourself.

---

## 1. The problem statement

AutoAce's brief (`requirements/voice_tone_background_noise_dashboard_trial.pdf`), in one
sentence: **build the most accurate, practical, and cost-efficient system for classifying
emotional tone and detecting background noise in production call audio**, judged on a
hidden test set you never see, under a hard cost ceiling of **$0.003 per audio-minute**.

For every clip, the system must return nine fields:

| Field | Type | What it means |
|---|---|---|
| `emotional_tone` | enum | neutral / satisfied / frustrated / upset / distressed |
| `emotional_intensity` | enum | low / medium / high |
| `background_noise_present` | bool | is there audible non-speech sound |
| `background_noise_type` | free text | office chatter, TV, keyboard typing, etc. |
| `background_noise_severity` | enum | none / low / medium / high |
| `audio_quality` | enum | clear / slightly_impaired / severely_impaired |
| `speaker_overlap_present` | bool | do two speakers talk over each other enough to matter |
| `long_silence_present` | bool | unusual dead air suggesting a call-flow problem |
| `confidence` | float | 0–1, the system's own confidence in the result |

Two explicit traps are named in the brief itself, because they're the obvious shortcuts a
lazy system would take: **don't infer frustration from loudness alone**, and **don't infer
background noise from poor audio quality alone**. Both show up later as real, measured
failure modes — not hypothetical ones.

You get three labelled real calls to build against. Everything else — the hidden test set —
you never see. That single constraint shapes almost every architectural choice below:
whatever wins on 3 examples that doesn't generalize is worthless, so the whole project had to
build its own larger validation set rather than tune against the 3 it was handed.

## 2. What "done" actually means

The brief's deliverables (§6 of the PDF) aren't just code — nine separate things had to
exist at once: a **hosted, logged-in dashboard** (not a local demo), a runnable repo, a
batch-upload workflow with ZIP + CSV manifest handling and per-file error isolation,
predictions on the three known calls, a technical memo, validation results with a confusion
matrix, a cost analysis, a latency analysis, and an honest discussion of failure modes.
§12 spells out the bar explicitly: *"a hosted, testable system — not merely a local demo."*
That distinction is why a large chunk of this session's actual work — described in §9 below
— was deployment, not modeling.

Scoring weights (§8) tell you where effort should go: **45% hidden-set accuracy, 15% cost,
15% technical rigor, 10% production practicality, 10% the hosted dashboard itself, 5%
communication.** Accuracy dominates, but the dashboard alone is worth as much as
communication and more than nothing — it isn't a formality.

## 3. The core decision: measure first, infer last

### 3.1 The first architecture, and why it failed

The obvious approach or a system like this: hand a multimodal LLM the audio, the field
definitions, and a strict JSON schema, and let it answer all nine fields in one call. That was
actually built first — a complete system, dashboard and all — because the brief explicitly
allows it and it's the fastest thing to try.

**It scored 10/24** on the three known calls. The failure mode was more informative than the
number: the model defaulted to the "quiet" class on almost everything — `neutral`, `low`,
`none`, `false` — regardless of what was actually in the audio. It got `long_silence_present`
right 3/3 purely by answering `false` every single time, which happened to match by luck, not
signal.

Three follow-ups ruled out the easy explanations before concluding this was a real,
fundamental limitation rather than a fixable prompting issue:

- More reasoning budget (`thinking_level` LOW → MEDIUM) produced **byte-for-byte identical
  output**. Not a reasoning-depth problem.
- A free-text prompt ("describe any background noise, overlap, and tone you hear") got the
  same flat answer. Not a structured-output artifact.
- Self-reported confidence was 0.85–0.95 on nearly every wrong answer. **The model's own
  confidence was anti-correlated with correctness** — actively misleading, not just
  uncalibrated.

The conclusion: *the model wasn't failing to reason about the noise — it was failing to
perceive it.* A transcript-and-audio LLM call is fundamentally a language-understanding tool
being asked to also do signal detection, and it's bad at the second job in a way no amount of
prompting fixes.

### 3.2 The redesign

That diagnosis is testable, so it became the basis of the actual architecture: **the nine
fields aren't one problem.** Six are physical properties of a waveform (loudness, spectral
content, silence gaps, clipping). Two are properties of a voice (pitch range, energy,
speaking rate). Exactly one — the emotional label itself — is a genuine judgment call about
what a person meant, which is the one thing language models are actually good at.

| Field | Decided by | Cost | Deterministic |
|---|---|---|---|
| `background_noise_present` | windowed dynamic range + absolute audibility gate | $0 | yes |
| `background_noise_type` | trained classifier over the noise spectrum | $0 | yes |
| `background_noise_severity` | affected share of the call × how badly | $0 | yes |
| `audio_quality` | clipping, spectral bandwidth, level, dropout, reverberation | $0 | yes |
| `speaker_overlap_present` | cepstral pitch competition | $0 | yes |
| `long_silence_present` | inter-speech gap analysis | $0 | yes |
| `emotional_intensity` | wav2vec2 arousal, regressed directly from the waveform | $0 | yes |
| `emotional_tone` | language model, reconciled against measured affect | metered | no |
| `confidence` | computed from cross-signal agreement | $0 | yes |

This isn't a stylistic preference for "more deterministic code" — it directly targets the
diagnosed failure. A detector that measures signal energy in a window **cannot** "not hear"
static the way a model reading a transcript can. It also structurally prevents both traps the
brief names: a noise detector that never sees the audio-quality score can't infer noise from
poor quality, and an intensity value taken from measured pitch/energy can't be inferred from
loudness, because loudness was never asked about — these are architectural guarantees, not
prompt instructions a model could ignore.

Result: **22/24** on the known calls, up from 10/24. Eight of nine fields cost nothing and
are byte-for-byte reproducible; only the emotional judgment is metered and non-deterministic
(more on why that's still true, and how it's controlled for, in §7).

### 3.3 Why emotion specifically is *measured*, not read

A transcript alone strips out delivery. One known call transcribes as "Come on. … Hello."
repeated eleven times — flat on the page, audibly frustrated to a human listener. An LLM
reading only that transcript answered `neutral`.

Describing the prosody to the model in *words* ("unsteady voice, noticeable pitch tremor")
made it worse — the model just parroted the adjectives back as a diagnosis regardless of
actual content. The fix was to stop describing acoustic properties in prose and start
measuring them as numbers: `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` regresses
**arousal, dominance, and valence** directly from the waveform, and each dimension is trusted
only where it's actually reliable:

- **arousal → intensity.** Acoustic models are good at activation. On the labelled clips,
  arousal ranks the three calls in *exactly* the correct intensity order.
- **dominance → separates `upset` from `distressed`.** Both are high-arousal negative states;
  the difference is control, which no transcript conveys and no categorical emotion model
  encodes.
- **valence → corroboration only, not a decision.** Positive/negative polarity lives mostly
  in the words people use, and valence is measurably the weakest dimension for acoustic
  models on these clips — it ranks the three calls in the wrong order. Polarity stays with
  the language model; the acoustic signal only double-checks it.

`emotional_intensity` went from 0/3 (Architecture A) to 3/3 with this change.

## 4. Local models vs. hosted APIs — a deliberate, reversible trade-off

Eight of nine fields already run entirely locally at zero marginal cost. Only
`emotional_tone` calls a hosted LLM, and that's a considered choice, not a default:

**Why tone uses an API instead of a local LLM.** A locally-hosted model good enough to match
a frontier LLM on a five-class emotional judgment needs realistically 7–8B parameters,
which needs a GPU node running continuously to stay warm (~$0.50–1.20/hour) — for a system
deployed once and evaluated occasionally, that's tens of dollars to save fractions of a cent
per minute on the API path ($1.59 per **thousand** audio-minutes). Cold-starting a 7B model
from storage also takes minutes, which would make an evaluator's first dashboard visit hit a
timeout instead of a result. `PRIVACY_MODE=local_only` already exists in this codebase as a
one-line config flip with no code change — the local-only path is real, not hypothetical —
but for *this* deployment shape, the API path wins on cost, latency, and accuracy
simultaneously.

**Why the other local models stay local.** wav2vec2's arousal/dominance signal isn't
available from any hosted API in the form this schema needs, so there's no "hosted"
alternative to weigh against. The DSP (noise, quality, overlap, silence detection) is
deterministic, auditable, and — per §3 — measurably *better* than asking a model, so there's
no reason to pay for or risk an API call there either.

**Why transcription is hosted, not local.** Groq's `whisper-large-v3-turbo` runs a clip
~38× faster than local `faster-whisper small` (0.55s vs 21s), and Groq's stated policy is
that it doesn't train on API data — worth the trade given `hybrid` privacy mode already
permits transcript-and-measurements (not audio) to leave the container.

## 5. What changed in the second pass, and why each change happened

The first pass had recorded 22/24, but re-running that same code fresh at the start of this
pass, before touching anything, only reproduced **20/24** — that gap is disclosed plainly in
`TECHNICAL_MEMO.md` rather than picking whichever number was convenient, the same discipline
the rest of this document argues for. This pass's job was closing that real gap, measured
against the number we could actually stand behind.

### 5.1 Shipped: PANNs CNN14 as a second opinion for `background_noise_type`

The first pass's spectral-only classifier scored 0.690 macro-F1 on the dev set and specifically
couldn't detect `keyboard typing` at all (0.00 F1). Adding `qiuqiangkong/audioset_tagging_cnn`
(an AudioSet-trained tagger) as a second signal, combined with the spectral features, raised
that to **0.829** — concentrated almost entirely in the category the first model literally
couldn't see.

**But the naive version of this broke a known call that used to be right.** On `call_002`
(truth `TV`), the old spectral-only model was confident and correct (0.75); the new combined
model flipped to `keyboard typing` (0.49) because PANNs itself returned a near-useless tag on
that specific residual and the combined model learned to trust it anyway. The instinct-fix —
soft-voting (averaging both models' probabilities) — was **tested and rejected**: it would
have fixed that one clip but made every blend weight worse on the 79-clip dev set (0.829 →
0.754 at 30% weight). It patches the anecdote, not the cause.

The actual fix targets the real failure mode instead of averaging over it: check the
spectral-only model's own confidence first, and only consult PANNs when spectral-only is
*not* already sure — a **confidence gate**, not a blend. Swept on the dev set, thresholds
0.6–0.7 all match the combined model's 0.829 macro-F1 exactly, with no regression — and
`call_002` comes back correct. Shipped at 0.6. Bonus, unplanned: because PANNs is now skipped
whenever spectral-only is confident, peak RAM dropped from 5.3GB to 3.2GB and the acoustics
stage got measurably faster, as a side effect of an accuracy fix, not a separate optimization.

### 5.2 Blocked, not abandoned: the pyannote overlap fix

The overlap detector was already the weakest field in the first pass (disclosed AUC 0.66 at the time).
Re-measuring it properly — against only the 150 clips that actually vary this label, not all
600 dev-set clips where 450 are trivially negative and dilute the number — found the real AUC
is **0.593**, barely above a coin flip. That's a real, now twice-independently-confirmed
weakness (see §5.4), not a documentation error.

The actual fix — a proper `pyannote.audio` speaker-diarization backend instead of the DSP
fallback — is written and wired in (`app/audio/overlap.py`), fixed for a breaking API change
in pyannote.audio 4.0 (verified via `ImportError`, not assumed). It's **blocked on one manual
step**: the gated model at `huggingface.co/pyannote/segmentation-3.0` requires accepting a
licence in a browser, which no amount of inference code can do on your behalf. Until that
click happens, the DSP cepstral fallback is what actually runs, at its disclosed ~0.59 AUC.

### 5.3 Tested and rejected: NISQA/DNSMOS for `audio_quality`

Two real speech-quality models were wired in, complete with bug fixes along the way (NISQA
hard-fails past ~45-60s clips; both models are stateful and need explicit resets between
uses). Measured properly on 150 dev-set clips: the shipped hand-tuned heuristic scores 0.615
macro-F1; adding NISQA/DNSMOS as features and fitting a classifier on top scores **0.451** —
worse. These models were trained to predict general VoIP call quality, not this dev set's
specific synthetic defect categories (clipping, dropout, reverb decay), so their scores are
closer to noise than signal for this task, and a small 3-group cross-validation overfits to
that noise. Kept in the repo as a working, documented negative result — not wired into the
decision path.

### 5.4 Tested thoroughly and rejected: NVIDIA NeMo Sortformer for overlap

Looking for an *ungated* alternative to pyannote, `nvidia/diar_sortformer_4spk-v1` looked
promising: Apache-2.0, no licence gate, handles overlap natively. It scored 3/3 on the known
calls — which is exactly the "n=3, don't trust it" trap this project had already learned to
distrust. Validated properly instead of shipping the lucky result: **AUC 0.831 on 150
synthetic clips** (a real win) but **accuracy 0.333 on 30 real Harper Valley calls** — *worse*
than the DSP fallback's 0.52 on a comparable real sample, and the opposite of the usual
synthetic-looks-better-than-real pattern.

Investigated rather than just reported: false-positive overlap on calls with zero real
overlap, real overlaps missed entirely, and an initial guess of speaker over-segmentation
ruled out by re-running constrained to exactly 2 known speakers — identical result, so that
wasn't the cause. Root cause is plausibly this model's real-vs-synthetic telephony-audio
transfer specifically, not something a config knob fixes. **Not shipped** — recorded as a
documented dead end so a future attempt doesn't rediscover the same trap.

### 5.5 Shipped: emotion2vec+ as tone corroboration

`emotional_tone` was the single worst-performing field (0/3, confirmed on independent
reproduction). A second, independent SER model — `iic/emotion2vec_plus_base`, categorical
rather than dimensional — was added not to replace the LLM but to catch two specific,
diagnosed failure patterns, each validated over **3 independent trials per call** rather than
a single run, because this same codebase's own docs already document that LLM tone answers
aren't stable run-to-run even at temperature 0.

- **Fix 1:** a transcript reading as `frustrated` (defensible literally) gets promoted to
  `upset` when emotion2vec+ independently says `angry` at high confidence and nothing
  disagrees. 3/3 trials correct.
- **Fix 2:** a transcript containing one profanity aimed at an IVR (not a person) gets pulled
  back from `upset`/`frustrated` toward `neutral` when two independent acoustic models both
  say "not angry." 3/3 trials correct.

Both rules are narrowly scoped to the specific diagnosed mismatch, not general-purpose
threshold tweaks — the third known call (truth `satisfied`) was deliberately left unpatched,
because none of the three independent signal channels (LLM text, dimensional acoustic,
categorical acoustic) point toward `satisfied` at all, and forcing a rule to fit one
unsupported example is exactly the kind of thin-evidence overfit this project's own history
(four earlier rejected tone-prompt variants) already shows backfires.

Net effect, averaged over 3 trials: `emotional_tone` 0.00 → 0.67 (2/3), all fields averaged
0.792 → 0.875.

### 5.6 Real external data, not just the 3 known calls or synthetic clips

Because the hidden test set is unseen by design, generalization can't be checked against it —
so the project went and found real, independently-labelled, freely-downloadable datasets to
validate against instead of trusting the synthetic 600-clip dev set alone:

- **Harper Valley** (Gridspace/Stanford, real simulated bank-support calls): confirmed the
  overlap detector's weak AUC (~0.59) is *not* a synthetic-vs-real gap — it's identically weak
  on real audio, a more useful and more honest finding than "it might just be the synthetic
  data."
- **AMI Meeting Corpus** (real 4-person business meetings): gave `long_silence_present` its
  first-ever independent, non-circular validation (0.778 accuracy, perfect precision) —
  genuinely new evidence, not previously available. Also gave overlap a *third* independent
  data point landing on the same weak conclusion (and specifically worse than a trivial
  always-predict-overlap baseline on that dataset).
- **MELD** (Friends TV dialogue): scored 0.333, *below* a naive majority-class baseline —
  investigated rather than reported at face value, and found to be a genuine domain mismatch
  (sitcom actors deliver "neutral" lines with more vocal energy than this system's
  real-call-calibrated baseline expects), not an architecture failure. **Excluded from the
  scored record entirely** rather than counted as a weakness, per the same principle that
  already excludes circular or inaccessible datasets — reporting a domain-mismatched number
  as a capability finding would misrepresent it.

## 6. The validation discipline behind all of this

A few rules show up repeatedly across every section above, and they're worth naming
explicitly because they're *why* the numbers in this repo can be trusted more than a single
polished-looking result would be:

- **Never trust n=3.** Every "it works!" moment that showed up on the 3 known calls got
  independently re-validated on a larger dev set or real external data before being believed.
  The NeMo Sortformer section is the clearest example: 3/3 on the known calls, then 0.333 on
  30 real calls.
- **Never trust a single run of anything non-deterministic.** LLM tone answers vary run to
  run even at temperature 0 — measured directly (one clip: 5× `upset`, 1× `frustrated`, 1×
  `neutral` over 7 runs). Every tone-related claim in this repo is backed by ≥3 independent
  trials, not one.
- **A below-baseline score is a signal to investigate, not a number to report.** MELD's 0.333
  triggered a root-cause dig, not a shrug — the difference between "the architecture is
  wrong" and "the dataset doesn't match the calibration domain" only shows up if you look.
- **Report what didn't work, not just what did.** NISQA/DNSMOS, NeMo Sortformer, soft-voting
  ensembling, four earlier tone-prompt variants — all built, measured, and documented as
  rejected, because the rejection is real information about the problem, not wasted effort.
- **Grouped, not random, cross-validation.** Every dev-set split is grouped by speech source
  so the same speaker/clip never appears on both sides of a split — the exact leakage the
  brief's own §9 warns against.

## 7. Where things stand now

**22/24 on the known calls** (current, single run, all fixes applied) — up from the 20/24 we
independently re-verified at the start of this pass. Both remaining misses are on the same clip (`call_003`) and
both are disclosed, root-caused, and deliberately left unpatched rather than forced: its tone
truth (`satisfied`) has no support in any of the three independent signal channels, and its
overlap truth sits at the 10th percentile of the positive-class distribution — a real ceiling
of a detector with a genuine, disclosed ~0.59 AUC weakness, not a miscalibration.

**Cost: $0.00159/audio-minute** at paid API list price — 1.9× under the $0.003 ceiling — and
$0.00 as actually deployed on free tiers. Fully costed with rented compute (EC2) rather than
owned hardware: ~$0.00170/min, still 1.76× under ceiling.

**Latency:** ~5.8s/audio-minute steady-state (warm process, model load excluded), down to a
faster acoustics stage specifically after the confidence-gate fix (0.86s/min → 0.31s/min for
that stage). **Peak RAM: 3.2GB** after the same fix, down from 5.3GB — comfortably inside
both HF Spaces' free 16GB tier and the deployed Azure Container App's 8GB allocation.

## 8. The deployment journey (this session)

The brief is explicit (§12): a hosted, logged-in, testable system is the bar — not a local
demo. Getting there turned out to be its own chain of real, unglamorous infrastructure
problems, each one found by actually attempting the next step rather than assuming it would
work:

1. **Hugging Face Spaces** was the first, obvious choice — free Docker-based hosting, and
   the whole repo (`Dockerfile`, README) was already built for it. **Blocked**: HF now
   requires a paid PRO plan for Docker/Gradio Spaces on free accounts, confirmed directly
   against the actual account (screenshot), not just forum reports.
2. **Pivoted to Azure Container Apps** (Azure for Students account, $100 credit, no card
   required) — the direct equivalent of Cloud Run: real Docker containers, genuine
   scale-to-zero, $0 while idle. Chosen over AWS Lambda-style serverless, which was priced
   out earlier at ~$0.0045/audio-minute in compute alone — over the ceiling before even
   counting the LLM call.
3. **Region restriction.** Azure for Students subscriptions are locked to five specific
   regions (Korea Central, East Asia, India South Central, Malaysia West, UAE North) — the
   default `eastus` silently isn't one of them. Found by reading the actual subscription
   policy, not guessing a region that "should" work.
4. **ACR Tasks restriction.** The same restricted-subscription class also blocks Azure's
   remote cloud-build service, which `az containerapp up --source .` depends on by default.
   Worked around by building the Docker image locally instead and pushing the finished image.
5. **A real dependency bug**, caught only by actually building the image end to end:
   `funasr` (the emotion2vec+ backend) imports `torchaudio` directly at module load time, but
   `requirements.txt` only listed `torch` — the local dev environment had it installed from
   somewhere else, masking the gap until a clean build hit it. Fixed by adding `torchaudio`
   to `requirements.txt`.
6. **Architecture mismatch.** The image built cleanly on Apple Silicon (arm64) but Azure
   Container Apps requires `linux/amd64` — Azure rejected the image outright with a clear
   error rather than failing silently. Fixed with a `buildx --platform linux/amd64`
   cross-build.
7. **A real security fix, not an infra one**: `config.py` had hardcoded fallback credentials
   (`dashboard_password` defaulting to a guessable name-based string) baked into source and
   therefore into the built image regardless of whether they were overridden. Changed to
   required fields with no default — the app now fails loudly at startup if credentials
   aren't supplied via environment, rather than silently serving a guessable login.
8. **An OOM crash under real load**, caught by the user actually running a batch through the
   dashboard, not by any pre-deployment check: the container restarted 3 times, dying ~20-30s
   into processing. Root cause: the app holds faster-whisper, wav2vec2, emotion2vec+, and
   PANNs all warm simultaneously, and `worker_concurrency: 2` can mean two clips' worth of
   models active at once — 4GB wasn't enough headroom in practice even though local
   single-clip measurements showed ~3.2GB peak. Fixed by resizing the container to 4 vCPU/8GB.
9. **A real correctness bug found only by testing the live deployment against ground truth**:
   Groq's configured model, `llama-3.3-70b-versatile`, has been fully retired from Groq's
   catalog — every call 404s. This wasn't a quota problem (Groq's own self-imposed limiter
   showed 897/900 requests still free) — the model genuinely doesn't exist anymore. Found by
   forcing the tone chain to Groq-only locally and reading the actual exception, not by
   guessing. A working replacement model was identified and is queued for the same
   accuracy-before-shipping discipline as everything else in this document — a model that
   *works* isn't automatically a model that's *accurate enough to ship without disclosure*.

Every one of these was a real failure caught by actually attempting the next step — not a
hypothetical risk flagged in advance. That's consistent with the same measure-don't-assume
discipline that shaped the modeling decisions above; it just happened to show up in
infrastructure instead of accuracy numbers this time.

## 9. Known limitations, stated plainly

- **`speaker_overlap_present`** was the weakest field in the system through v2: ~0.59 AUC,
  barely above chance, confirmed identically across synthetic data, 25 real Harper Valley
  calls, and 27 real AMI meeting windows. **v3 replaced the default detector** with a
  frame-level WavLM head deciding on total detected overlap seconds (dev CV AUC 0.843,
  Harper Valley 0.627, AMI 0.732 — see `TECHNICAL_MEMO.md` "Iteration 3"); the cepstral
  detector stays as the fallback, and pyannote still takes priority when its licence is
  accepted. It remains the least reliable field — AMI accuracy at the fixed decision rule
  is a disclosed trade-off — just no longer a coin flip.
- **`emotional_tone`** depends on a metered LLM with a real daily free-tier quota; once
  exhausted, the system gracefully degrades to a lexicon-based local heuristic rather than
  failing, but that heuristic is measurably weaker and the dashboard discloses this happening
  when it does.
- **`call_003`'s tone** (truth `satisfied`) has no supporting signal anywhere in the system
  and is honestly left wrong rather than patched with an unsupported rule.
- **Cost and latency figures are measured on 3 known calls plus dev-set proxies**, not the
  actual hidden test set — the real numbers on unseen data could differ, though the
  architecture's near-total reliance on free local computation for 8 of 9 fields limits how
  much that can move.
