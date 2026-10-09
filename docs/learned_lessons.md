# Learned Lessons

Tracking mistakes, problems encountered, and how they were resolved.
This file helps avoid repeating the same mistakes and serves as a knowledge base.

---

## How to Use This File

When you encounter a problem during development:

1. **Add an entry** under the relevant category when a mistake/problem is identified
2. **Document the symptoms**: what went wrong, error messages, unexpected behavior
3. **Document the root cause**: why it happened
4. **Document the fix**: exactly what was done to resolve it
5. **Add prevention notes**: how to avoid this in the future

### Entry Template

```
### [Short title]: YYYY-MM-DD

**Status**: Resolved / In Progress / Investigating

**Symptoms**:
- What happened, error messages, unexpected behavior

**Root Cause**:
- Why it happened

**Fix**:
- What was done to resolve it

**Prevention**:
- How to avoid this in the future
```

---

## Categories

- [Build & Dependencies](#build--dependencies)
- [Frontend](#frontend)
- [Backend](#backend)
- [Docker](#docker)
- [Models & AI](#models--ai)
- [Audio Processing](#audio-processing)
- [Configuration](#configuration)

---

## Build & Dependencies

### WSL2 migration: 6 bugs from Windows-first development: 2026-02-27

**Status**: Resolved

**Symptoms**:
- `pnpm dev` fails: "powershell not found"
- `uv sync` fails: Python 3.10 too old (requires 3.11+)
- pnpm resolves to Windows binary at `/mnt/c/Program Files/nodejs/pnpm`
- Docker health checks and EXPOSE directives reference wrong ports

**Root Cause**:
- Project was initially developed on Windows with PowerShell
- Root package.json `"dev"` script called `dev.ps1` directly
- WSL2 inherits Windows PATH, picking up Windows pnpm binary
- Dockerfiles had leftover default ports (8000, 3000) instead of project ports (36300, 36301)
- api.Dockerfile health endpoint assumed /api/health prefix but health router mounts at /health

**Fix**:
- Created `dev.sh` (bash equivalent of dev.ps1)
- Changed package.json: `"dev": "bash dev.sh"`, added `"dev:win"` for PowerShell
- Installed Python 3.12 via `uv python install 3.12` and pinned it
- Enabled native pnpm via `corepack enable pnpm`
- Fixed api.Dockerfile: EXPOSE 36300, HEALTHCHECK on :36300/health
- Fixed web.Dockerfile: EXPOSE 36301, CMD `-p 36301`, HEALTHCHECK on :36301

**Prevention**:
- Always test on both Windows and Linux when project targets both
- Docker health checks must match the actual app port and endpoint
- Use `corepack enable` to manage pnpm version consistently across platforms

---

## Frontend

_(No entries yet)_

---

## Backend

_(No entries yet)_

---

## Docker

### Dockerfile ports must match app configuration: 2026-02-27

**Status**: Resolved

**Symptoms**:
- Docker HEALTHCHECK would always fail after build
- Container appears unhealthy despite app running fine

**Root Cause**:
- api.Dockerfile had `EXPOSE 8000` and health check on `:8000/api/health`
- Actual API runs on port 36300, health endpoint is `/health` (no /api prefix)
- web.Dockerfile had `EXPOSE 3000` and no port flag in CMD
- Actual web app needs port 36301

**Fix**:
- Updated both Dockerfiles to use correct ports (36300, 36301)
- Corrected health endpoint path from `/api/health` to `/health`

**Prevention**:
- When changing app ports, grep for the old port across all Docker files
- Health check URL must match the actual route: check router prefix

---

## Models & AI

### Silero VAD requires torchcodec with torchaudio 2.10+: 2026-02-27

**Status**: Resolved

**Symptoms**:
- `test_silero_vad` fails with: `RuntimeError: torchaudio version 2.10.0+cu128 requires torchcodec for audio I/O`

**Root Cause**:
- torchaudio 2.10 (from PyTorch CUDA 12.8 index) dropped its own audio I/O backends
- Now requires `torchcodec` package for audio loading
- Silero VAD's `read_audio()` function internally uses `torchaudio.load()`

**Fix**:
- Use `librosa.load()` + `torch.from_numpy()` instead of `read_audio()` for loading WAV files
- Pass `sampling_rate=16000` to `get_speech_timestamps()` since we bypass `read_audio()`

**Prevention**:
- When using torchaudio >= 2.10, either install `torchcodec` or load audio with librosa/soundfile

### wav2small model requires manual weight loading: 2026-02-27

**Status**: Resolved

**Symptoms**:
- `AutoFeatureExtractor.from_pretrained("audeering/wav2small")` fails
- `Wav2Vec2PreTrainedModel` approach fails with meta device tensor errors on newer PyTorch

**Root Cause**:
- wav2small is a custom architecture (Vgg7 + spectrogram) that doesn't have a standard transformers feature extractor
- The model uses custom Conv/Spectrogram/LogmelFilterBank layers incompatible with the transformers auto API

**Fix**:
- Manually define model architecture (Vgg7, Spectrogram, LogmelFilterBank, Wav2Small classes)
- Download weights via `hf_hub_download("audeering/wav2small", "model.safetensors")`
- Load with `load_file()` + `model.load_state_dict(state_dict, strict=False)`
- Use `.copy_()` instead of `.data = ` for tensor assignment (avoids meta device issues)

### Parakeet TDT 1.1B outputs lowercase only: 2026-02-27

**Status**: Known behavior

**Symptoms**:
- Parakeet TDT 1.1B transcription has no capitalization or punctuation
- Output: "the quick brown fox jumps over the lazy dog" (all lowercase, no periods)

**Root Cause**:
- This is by design: the 1.1B model was trained on lowercase-normalized text
- The 0.6B v2 model was trained with proper casing and punctuation

**Workaround**:
- Use Parakeet TDT 0.6B v2 when formatting matters (has caps + punctuation)
- Use 1.1B only when raw accuracy on challenging audio is the priority
- Post-processing (truecasing) could be added but adds latency

**Prevention**:
- Always check model output format before choosing a model for production
- Larger model ≠ better for all use cases

---

## Audio Processing

### Recording quality was degraded by premature downsampling: 2026-02-27

**Status**: Resolved

**Symptoms**:
- Voice recordings from the mic sounded bad/low quality
- Playback quality noticeably poor

**Root Cause**:
- The frontend (`useRecorder.ts`) was requesting `sampleRate: 16000` from the browser mic
- This forced the mic to capture at 16kHz from the start, losing audio fidelity
- 16kHz is only needed for AI model input (Whisper, emotion2vec, etc.), not for recording/playback

**Fix**:
- Changed default recording sample rate to 48kHz (HD quality)
- Added configurable sample rate selector in mic settings (16k / 44.1k / 48k)
- The backend `audio.py` already had `prepare_for_model()` which downsamples to 16kHz only when feeding AI models: no backend changes needed
- Set `AudioContext({ sampleRate })` to match the requested rate

**Prevention**:
- Record at the highest practical quality; only downsample at the point of model consumption
- The rule is: "record high, process low": keep source quality, resample on demand

---

## Configuration

### Hardcoded Windows paths in test files: 2026-02-27

**Status**: Resolved

**Symptoms**:
- All 5 test files in `api/` failed with `FileNotFoundError: [Errno 2] No such file or directory: 'e:\\dev\\GitHub\\Speechos\\samples'`
- Tests worked on Windows but failed in WSL2

**Root Cause**:
- `SAMPLES_DIR` was hardcoded as `r"e:\dev\GitHub\Speechos\samples"` in all test files
- `test_integration.py` also had a hardcoded absolute path for `test_tone.wav`

**Fix**:
- Replaced all hardcoded paths with: `os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")`
- This resolves relative to the project root regardless of platform

**Prevention**:
- Never hardcode absolute paths in test files
- Use `__file__`-relative paths or a shared conftest.py fixture for project paths

---

## Docker

### Docker TTS engine API configs must be verified against actual images: 2026-02-28

**Status**: Resolved

**Symptoms**:
- orpheus: 404 on `/api/synthesize`, wrong port
- melotts: 404 on `/tts/convert/tts`
- fish-speech: image tag `latest-webui-cuda` doesn't exist
- cosyvoice: image `catcto/cosyvoice:latest` doesn't exist at all

**Root Cause**:
- API endpoints were guessed from project READMEs without verifying actual Docker image behavior
- Image tags were assumed without checking Docker Hub for available tags
- Some images are WebUI-only (Gradio) with no REST API endpoint

**Fix**:
- orpheus: endpoint is `/api/generate` (not `/api/synthesize`), internal port is 8899 (not 8080)
- melotts: endpoint is `/convert/tts` (not `/tts/convert/tts`)
- fish-speech: changed to `server-cuda` image (REST API), port 8080, endpoint `/v1/tts`
- cosyvoice: removed from options (image doesn't exist on Docker Hub)
- parler: added `response_type: wav` (defaults to MP3)

**Prevention**:
- Always check Docker Hub for available image tags before adding to compose
- Always verify API endpoints by either: (a) starting the container and checking `/docs`, or (b) reading the Dockerfile/source
- WebUI images (Gradio) are NOT the same as API server images
- Document the verified API in comments next to the compose service

---

## Models & AI

### HuggingFace models may have untrained classifier heads: 2026-02-28

**Status**: Resolved

**Symptoms**:
- wav2vec2-ser models load without errors but produce random/uniform scores (~0.13 each)
- Warning: "Some weights were not initialized... You should probably TRAIN this model"

**Root Cause**:
- Many HuggingFace wav2vec2 models are base models meant for fine-tuning
- The transformers `audio-classification` pipeline adds a new classifier head
- If the uploaded checkpoint doesn't include classifier weights (just base model weights), the classifier is randomly initialized
- Models `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` and `r-f/wav2vec-english-speech-emotion-recognition` both have this issue

**Fix**:
- Removed all 3 wav2vec2-ser models from the emotion options dropdown
- Also removed: WavLM SER (model not found), wav2small (regression model, not classification)
- Kept: emotion2vec (works perfectly), HuBERT SER (works with 4 emotions)

**Prevention**:
- Always test model inference before adding to options: check that scores are differentiated, not uniform
- Watch for "should probably TRAIN" warnings from transformers
- Verify model card says it's fine-tuned for the target task, not just a base model

### Browser records WebM/Opus, Docker engines need WAV: 2026-02-28

**Status**: Resolved

**Symptoms**:
- Docker STT engines return 500: "file does not start with RIFF id"
- Works fine with native engines (faster-whisper accepts any format via prepare_for_model)

**Root Cause**:
- MediaRecorder API records in WebM/Opus format by default
- Docker STT containers expect WAV (RIFF) format
- Native engines use `prepare_for_model()` which decodes any format via librosa

**Fix**:
- Added WAV conversion in `transcribe.py` before sending to Docker engines:
  `read_audio(data)` → `resample_to_16k(audio_np, sr)` → `float32_to_wav_bytes(audio_16k)`

**Prevention**:
- Any external API that accepts "audio files" should be assumed to only accept WAV unless documented otherwise
- Always convert browser audio to WAV before sending to external services

---

### PyAV float audio incorrectly normalized as int16: 2026-02-28

**Status**: Resolved

**Symptoms**:
- Docker STT engines (Parakeet TDT 0.6B v2) receive audio but return empty transcription
- Only happens with WebM/Opus input (WAV input works fine)
- Audio RMS after conversion is ~0.000005 (essentially silence)

**Root Cause**:
- `_decode_with_av()` in `audio.py` had normalization logic: `if audio.max() > 1.0: audio = audio / 32768.0`
- Opus codec outputs float audio (format `fltp`) in [-1, 1] range
- But decoded values can slightly exceed 1.0 (e.g., max=1.013621: normal for lossy codecs)
- This triggered the int16 normalization path, dividing already-normalized float audio by 32768
- Result: signal amplitude crushed to ~0.00003 of original: inaudible

**Fix**:
- Check `stream.codec_context.format.name` to determine if audio is float format (`fltp`, `flt`, `dblp`, `dbl`)
- Float formats: clip to [-1, 1] instead of dividing
- Integer formats: normalize by actual peak value, not hardcoded 32768

**Prevention**:
- Never assume audio values > 1.0 means integer format: float audio from lossy codecs routinely exceeds [-1, 1] slightly
- Always check the codec format (int vs float) before deciding normalization strategy
- Test WebM/Opus roundtrip in automated tests to catch amplitude issues
