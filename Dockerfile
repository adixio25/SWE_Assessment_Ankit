# Hugging Face Spaces (Docker SDK), free CPU tier: 2 vCPU / 16 GB RAM / 50 GB disk.
# Also runs as-is on the EC2 path described in infra/README.md (a single
# `t4g.large`, 2 vCPU / 8 GB — everything below fits with headroom on the
# lighter model tier this image ships).
#
# Sized deliberately for that box:
#   * CPU-only torch, saving ~2.3 GB of unused CUDA libraries
#   * every model baked into the image, because a free Space (and a
#     start/stop EC2 box, which reverts to a fresh root volume each time it
#     is recreated) has no durable place to cache a download between runs
#   * thread counts pinned to the 2 vCPU actually available, since letting
#     torch and OpenMP each assume more cores makes CPU inference slower
#
# v2 adds PANNs CNN14 (~330MB checkpoint) for background_noise_type and
# emotion2vec+ base (~1.1GB) for a second SER opinion; total image RAM
# footprint with every model warm is ~7-9GB, still inside either target's
# ceiling but with less headroom than v1 — see TECHNICAL_MEMO_V2.md. NISQA
# and DNSMOS are deliberately NOT baked in here despite having a working,
# tested module (app/audio/quality_mos.py) — measured to make quality
# classification worse, not better, so there is no reason to pay their RAM.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser
ENV HOME=/home/appuser \
    PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    XDG_CACHE_HOME=/home/appuser/.cache \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false \
    PORT=7860

WORKDIR /home/appuser/app
USER appuser

COPY --chown=appuser:appuser requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=appuser:appuser app ./app

# Warm the model caches at build time so the first request is not a download.
# v3 adds microsoft/wavlm-base for the frame-level overlap detector
# (app/audio/overlap_frames.py) — without this, a cold container's first
# overlap decision would silently fall back to the cepstral detector while
# ~380MB downloads.
RUN python -c "\
from faster_whisper import WhisperModel; \
WhisperModel('small', device='cpu', compute_type='int8')" \
 && python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim', \
                  allow_patterns=['*.json','*.safetensors'])" \
 && python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('microsoft/wavlm-base', \
                  allow_patterns=['*.json','*.bin','*.safetensors'])"

# v2 addition: PANNs CNN14, used by app/audio/noise_panns.py and required for
# app/models/noise_type_panns.joblib (measured +0.14 macro-F1 over the
# spectral-only classifier on the dev set — see eval/train_noise_panns.py).
# panns_inference's own downloader shells out to `wget`, which the slim base
# image does not have; curl is used directly instead, at build time, so a
# cold container never blocks a request on a 320MB download.
RUN mkdir -p /home/appuser/panns_data \
 && curl -sL --retry 5 --retry-delay 3 --retry-all-errors -o /home/appuser/panns_data/class_labels_indices.csv \
      "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv" \
 && curl -sL --retry 5 --retry-delay 3 --retry-all-errors -o "/home/appuser/panns_data/Cnn14_mAP=0.431.pth" \
      "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"

# v2 addition: emotion2vec+ base (~1.1GB via modelscope), the second SER
# opinion in app/ser/emotion2vec_backend.py. Warmed at build time for the
# same reason as everything else in this file — no request should pay for a
# cold download. Skippable by building with --build-arg SER_ENSEMBLE=false
# if the extra ~1.1GB image size / ~1GB RAM is not wanted; the app degrades
# to v1 behaviour automatically (see app/config.py's SER_ENSEMBLE_ENABLED).
ARG SER_ENSEMBLE=true
RUN if [ "$SER_ENSEMBLE" = "true" ]; then \
      python -c "from funasr import AutoModel; AutoModel(model='iic/emotion2vec_plus_base', disable_update=True)"; \
    fi

EXPOSE 7860
HEALTHCHECK --interval=60s --timeout=10s --start-period=90s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:7860/health')"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
