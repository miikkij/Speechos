# Engine Audit: 2026-02-28

Systematic test of every engine option. Each engine tested for:
- Can it load/start?
- What input does it need?
- What response format does it return?
- Is the integration code correct?

## Legend
- [x] = Working, integrated, tested
- [~] = Works but has caveats
- [ ] = Not working / not integrated
- [N/A] = Not installed, cannot test
- [BROKEN] = Broken: needs removal or fix

---

## 1. STT Engines: Native

### 1.1 faster-whisper (CTranslate2)
- **Status**: [x] Working
- **Package**: `faster-whisper` (installed)
- **Models**: tiny, base, small, medium, turbo, large-v3, distil-whisper
- **Input**: numpy float32 array at 16kHz
- **API**: `stt.transcribe(audio, language=..., beam_size=5, vad_filter=True)`
- **Output**: generator of segments (start, end, text) + info (language, language_probability)
- **Integration**: `transcribe.py:74-99`: iterates segments, builds response
- **Test result**: Transcribes correctly, returns timestamps + language detection
- **Device**: CPU or CUDA

### 1.2 vosk (Kaldi)
- **Status**: [~] Installed, needs model download
- **Package**: `vosk` (installed)
- **Models**: vosk-model-small-en-us-0.15 (40MB), vosk-model-en-us-0.22 (1.8GB)
- **Input**: WAV bytes via KaldiRecognizer (requires wave module, 16kHz PCM)
- **API**: `KaldiRecognizer(model, framerate)` → `rec.AcceptWaveform(chunk)` → `rec.FinalResult()`
- **Output**: JSON string: `{"text": "...", "result": [{"word": "...", "start": ..., "end": ...}]}`
- **Integration**: `transcribe.py:101-128`: converts float32→WAV, feeds chunks, parses JSON
- **Note**: Models auto-download on first use via `models.py:341-356`
- **Device**: CPU only

### 1.3 whisperx
- **Status**: [N/A] Not installed
- **Package**: `whisperx` (not installed)
- **Install**: `pip install whisperx`
- **Input**: numpy array + language
- **API**: `stt.transcribe(audio, language=language, batch_size=16)`
- **Output**: dict with `{"segments": [...], "language": "..."}`
- **Integration**: `transcribe.py:162-182`: code present, untested
- **Device**: CPU or CUDA

### 1.4 moonshine
- **Status**: [N/A] Not installed
- **Package**: `moonshine` (not installed)
- **Install**: `pip install moonshine`
- **Input**: numpy array
- **API**: `stt.transcribe(audio)`
- **Output**: string or list
- **Integration**: `transcribe.py:184-194`: code present, untested
- **Device**: CPU

### 1.5 nemo (NVIDIA NeMo)
- **Status**: [N/A] Not installed
- **Package**: `nemo_toolkit[asr]` (not installed)
- **Install**: `pip install nemo_toolkit[asr]`
- **Input**: file path (temporary WAV file)
- **API**: `stt.transcribe([tmp_path])`
- **Output**: list of strings or Hypothesis objects
- **Integration**: `transcribe.py:130-160`: writes temp WAV, transcribes, cleans up
- **Device**: CUDA only (raises ValueError on CPU)

### 1.6 wav2vec2-stt (Transformers)
- **Status**: [x] Working
- **Package**: `transformers` (installed)
- **Models**: facebook/wav2vec2-base-960h (95M), facebook/wav2vec2-large-960h (317M)
- **Input**: dict `{"raw": audio, "sampling_rate": 16000}`
- **API**: `stt({"raw": audio, "sampling_rate": 16000})`
- **Output**: dict `{"text": "ALL CAPS TEXT"}`
- **Integration**: `transcribe.py:196-207`
- **Test result**: Works. Outputs ALL CAPS (no punctuation, no lowercase)
- **Caveat**: Output is uppercase-only: may want to add `.lower()` post-processing
- **Device**: CPU or CUDA

---

## 2. STT Engines: Docker

### 2.1 speaches
- **Status**: [x] Working
- **Image**: `ghcr.io/speaches-ai/speaches:latest-cuda` (5.6GB, pulled)
- **Port**: 36320
- **Health**: `GET /health`
- **API**: `POST /v1/audio/transcriptions` (OpenAI-compatible)
- **File field**: `file`
- **Extra fields**: `model=Systran/faster-whisper-small`
- **Response**: JSON `{"text": "..."}`
- **Parser**: `_parse_text_response`
- **Test result**: 9s startup. Caps + punctuation. Accurate.
- **Startup time**: ~9s

### 2.2 whisper-asr
- **Status**: [x] Working
- **Image**: `onerahmet/openai-whisper-asr-webservice:latest-gpu` (14.6GB, pulled)
- **Port**: 36321
- **Health**: `GET /`
- **API**: `POST /asr`
- **File field**: `audio_file`
- **Response**: plain text
- **Parser**: `_parse_text_response`
- **Test result**: 51s startup. Accurate but slow to start.
- **Startup time**: ~51s

### 2.3 whisper-cpp: REMOVED
- **Reason**: Requires manual model download (`ggml-large-v3.bin`). Not auto-startable.

### 2.4 linto-nemo (Parakeet TDT 0.6B v2)
- **Status**: [x] Working
- **Image**: `lintoai/linto-stt-nemo:latest` (8.9GB, pulled)
- **Port**: 36323
- **Health**: `GET /healthcheck`
- **API**: `POST /transcribe` with `Accept: application/json`
- **File field**: `file`
- **Response**: Double-encoded JSON. Has `text`, `words` (word-level timestamps)
- **Parser**: `_parse_linto_response` (handles double-encoding + "words" field)
- **Test result**: 57s startup. Caps + punctuation. Word timestamps. Best balanced.
- **Startup time**: ~57s

### 2.5 linto-whisper
- **Status**: [x] Working
- **Image**: `lintoai/linto-stt-whisper:latest` (2.6GB, pulled)
- **Port**: 36324
- **Health**: `GET /healthcheck`
- **API**: `POST /transcribe` with `Accept: application/json`
- **File field**: `file`
- **Response**: Double-encoded JSON with word timestamps + confidence
- **Parser**: `_parse_linto_response`
- **Test result**: 42s startup. Works correctly.
- **Note**: Image was initially wrong (`-gpu` suffix). Fixed to `lintoai/linto-stt-whisper:latest`.
- **Startup time**: ~42s

### 2.6 whisperx-api: REMOVED
- **Reason**: Upstream bug: VAD model download URL returns HTTP 301, container crashes on startup.

### 2.7 linto-nemo-1.1b (Parakeet TDT 1.1B)
- **Status**: [~] Working with caveat
- **Image**: `lintoai/linto-stt-nemo:latest` (same image, different model config)
- **Port**: 36327
- **Health**: `GET /healthcheck`
- **API**: `POST /transcribe` with `Accept: application/json`
- **File field**: `file`
- **Response**: Double-encoded JSON. Lowercase only (no caps/punctuation)
- **Parser**: `_parse_linto_response`
- **Test result**: 75s startup. Best raw accuracy but lowercase-only output.
- **Caveat**: No capitalization or punctuation
- **Startup time**: ~75s

---

## 3. TTS Engines: Native

### 3.1 piper (ONNX)
- **Status**: [x] Working
- **Package**: `piper` (installed)
- **Models**: en_US-lessac-medium/high, en_US-amy-medium, en_US-ryan-medium, en_US-arctic-medium, en_US-libritts-high, en_GB-alan-medium
- **Input**: text string
- **API**: `tts.synthesize(text)` returns generator of chunks with `.audio_float_array` and `.sample_rate`
- **Output**: float32 audio chunks at model's native sample rate
- **Integration**: `synthesize.py:70-78`: concatenates chunks, converts to int16
- **Test result**: Works. Ultra-fast. Auto-downloads models from HuggingFace.
- **Device**: CPU (ONNX)

### 3.2 kokoro
- **Status**: [~] Installed, model download issue
- **Package**: `kokoro` (installed)
- **Input**: text + voice name
- **API**: `tts(text, voice=voice)` returns generator of `(graphemes, phonemes, audio_tensor)`
- **Output**: audio tensors at 24kHz
- **Integration**: `synthesize.py:80-90`: concatenates tensor chunks
- **Note**: Model download hangs during first use (network/timeout issue). Once cached, should work.
- **Device**: CPU or CUDA

### 3.3 chatterbox
- **Status**: [N/A] Not installed
- **Package**: `chatterbox-tts` (not installed)
- **Install**: `pip install chatterbox-tts`
- **Input**: text
- **API**: `tts.generate(text)` returns tensor [1, samples] at 24kHz
- **Integration**: `synthesize.py:92-97`: squeeze + convert to int16
- **Device**: CPU or CUDA

### 3.4 bark
- **Status**: [~] Installed via transformers, untested live
- **Package**: `transformers` (installed, includes bark support)
- **Models**: suno/bark, suno/bark-small
- **Input**: text
- **API**: `tts(text)` returns `{"audio": ndarray, "sampling_rate": int}`
- **Output**: float32 audio + sample rate
- **Integration**: `synthesize.py:125-130`: squeeze + convert to int16
- **Note**: Slow on CPU. GPU recommended. Supports music/effects via text tokens.
- **Device**: CPU or CUDA

### 3.5 espeak (eSpeak-NG)
- **Status**: [x] Working
- **Package**: `espeakng_loader` (installed, provides DLL + data)
- **Input**: text (UTF-8 encoded)
- **API**: ctypes calls to `espeak_Synth()` with callback collecting PCM chunks
- **Output**: int16 PCM audio at espeak's native sample rate
- **Integration**: `synthesize.py:178-224`: DLL path via espeakng_loader, callback-based synthesis
- **Test result**: Works. Robotic but instant. 100+ languages.
- **Device**: CPU (rule-based, no GPU)

---

## 4. TTS Engines: Docker

### 4.1 qwen3-tts
- **Status**: [x] Working, tested
- **Image**: `speechos-qwen3-tts:latest` (11.9GB, built locally)
- **Port**: 36316
- **Health**: `GET /health`
- **API**: `POST /v1/audio/speech` (OpenAI-compatible)
- **Params**: JSON `{"input": text, "voice": "serena", "response_format": "wav"}`
- **Response**: WAV bytes
- **Integration**: `docker_tts.py:78-90`
- **Test result**: 39s startup. SOTA quality. 6 voices. Emotion control via text.
- **Startup time**: ~39s
- **VRAM**: ~3.9GB

### 4.2 xtts (Coqui TTS)
- **Status**: [~] Config correct, not live-tested (image not pulled)
- **Image**: `ghcr.io/coqui-ai/tts` (NOT pulled locally)
- **Port**: 36310 → internal 5002
- **Health**: `GET /`
- **API**: `GET /api/tts?text=...&language_id=en`
- **Params**: query string: text, language_id, speaker_id, speaker_wav
- **Response**: WAV bytes
- **Integration**: `docker_tts.py:20-28`: uses GET with params. CORRECT.
- **Note**: 17 languages, voice cloning support. Need to pull ~5GB image.

### 4.3 chattts
- **Status**: [ ] UNVERIFIED: endpoint and port uncertain
- **Image**: `yikchunnnn/chattts-dockerized:latest` (NOT pulled, ~15GB)
- **Port**: 36311 → internal 9000
- **Health**: `GET /`
- **API**: `POST /generate`: UNVERIFIED, may be wrong
- **Params**: JSON `{"text": "..."}`
- **Response**: Unknown (likely MP3, not WAV)
- **Integration**: `docker_tts.py:29-37`: config present but unverified
- **Action needed**: Pull image, start container, check `/docs` for actual API
- **Risk**: Endpoint and port may be wrong. Response might be MP3 not WAV.

### 4.4 melotts
- **Status**: [BROKEN] Wrong API endpoint
- **Image**: `sensejworld/melotts:v0.0.4` (NOT pulled, ~9.2GB)
- **Port**: 36312 → internal 8888
- **Health**: `GET /`
- **API configured**: `POST /tts/convert/tts`: WRONG
- **API actual**: `POST /convert/tts` (no `/tts` prefix)
- **Params**: JSON `{"text": "...", "language": "EN", "speaker_id": "EN-US"}`
- **Response**: WAV bytes
- **Integration**: `docker_tts.py:38-50`: endpoint URL wrong
- **Fix**: Change synth_url from `/tts/convert/tts` to `/convert/tts`

### 4.5 orpheus
- **Status**: [BROKEN] Wrong endpoint AND wrong port
- **Image**: `neosun/orpheus-tts:v1.5.0-allinone` (NOT pulled)
- **Port configured**: 36313 → internal 8080: WRONG
- **Port actual**: internal 8899
- **Health**: `GET /`
- **API configured**: `POST /api/synthesize`: WRONG
- **API actual**: `POST /api/generate`
- **Params**: JSON `{"text": "...", "voice": "...", "model_size": "..."}`
- **Response**: WAV bytes
- **Integration**: `docker_tts.py:51-59`: endpoint AND port wrong
- **Fix**: (1) Change compose port to `36313:8899`, (2) Change synth_url to `/api/generate`

### 4.6 fish-speech
- **Status**: [BROKEN] Wrong image tag, WebUI-only (not REST API)
- **Image configured**: `fishaudio/fish-speech:latest-webui-cuda`: TAG DOESN'T EXIST
- **Image actual**: `fishaudio/fish-speech:webui-cuda` (Gradio only) or `server-cuda` (REST API)
- **Port**: 36314 → internal 7860 (Gradio) or 8080 (API server)
- **Health**: `GET /`
- **API (server image)**: `POST /v1/tts`
- **Params**: JSON `{"text": "..."}`
- **Response**: Chunked audio stream
- **Integration**: `docker_tts.py:60-68`: completely wrong
- **Fix**: (1) Change image to `fishaudio/fish-speech:server-cuda`, (2) Change port to `36314:8080`, (3) Change endpoint to `/v1/tts`, (4) Handle streaming response

### 4.7 cosyvoice
- **Status**: [BROKEN] Docker image doesn't exist on Docker Hub
- **Image configured**: `catcto/cosyvoice:latest`: DOES NOT EXIST
- **Alternatives**: `lucferreira/cosyvoice`, `harryliu888/cosyvoice`, or build from https://github.com/catcto/CosyVoiceDocker
- **Port**: 36315 → internal 8080
- **API**: `POST /v1/tts` (form data: text + spk)
- **Integration**: `docker_tts.py:69-77`: image doesn't exist, endpoint likely wrong
- **Fix**: Need to find/build a working image, then verify API

### 4.8 parler
- **Status**: [~] Config correct, not live-tested (image not pulled)
- **Image**: `fedirz/parler-tts-server:latest` (NOT pulled, ~6.8GB)
- **Port**: 36317 → internal 8000
- **Health**: `GET /health`
- **API**: `POST /v1/audio/speech` (OpenAI-compatible)
- **Params**: JSON `{"input": text, "voice": "description"}`
- **Response**: MP3 by default (supports WAV via `response_type` param)
- **Integration**: `docker_tts.py:91-102`: CORRECT endpoint
- **Note**: Response is MP3 by default. May need to add `response_type: "wav"` to params.
- **Risk**: Response format mismatch: returns MP3, code expects WAV.

---

## 5. Emotion Engines

### 5.1 emotion2vec base (FunASR)
- **Status**: [x] Working
- **Package**: `funasr` (installed, 1.3.1)
- **Model**: `iic/emotion2vec_base_finetuned` (1.05GB, downloads from ModelScope)
- **Input**: file path (temp WAV written to disk)
- **API**: `model.generate(path, granularity="utterance", extract_embedding=False)`
- **Output**: list of results with `labels` (bilingual: "生气/angry") and `scores` (float list)
- **Labels**: angry, disgusted, fearful, happy, neutral, other, sad, surprised, \<unk\> (9 classes)
- **Integration**: `analysis.py:116-168`: handles bilingual labels with LABEL_MAP
- **Test result**: Works. Returned happy=0.39, surprised=0.19, angry=0.19 for calm speech.
- **Device**: CPU or CUDA

### 5.2 emotion2vec+ large
- **Status**: [~] Same API as base, not separately tested
- **Model**: `iic/emotion2vec_plus_large`
- **Note**: Larger model, same API. Expected to work identically with better accuracy.

### 5.3 wav2vec2-ser (ehcalabres)
- **Status**: [BROKEN] Classifier weights randomly initialized
- **Model**: `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`
- **Issue**: Model loads but emits warning "weights not initialized... You should probably TRAIN this model". All scores are ~0.13 (near-random). The model checkpoint's classifier layer names don't match the transformers pipeline expectation.
- **Test result**: neutral=0.12, sad=0.13, calm=0.13: effectively random
- **Action**: REMOVE from options

### 5.4 wav2vec2-ser (Dpngtm)
- **Status**: [BROKEN] Model removed from HuggingFace
- **Model**: `Dpngtm/wav2vec2-emotion`
- **Issue**: `OSError: not a valid model identifier listed on HuggingFace`
- **Action**: REMOVE from options

### 5.5 wav2vec2-ser (r-f)
- **Status**: [BROKEN] Classifier weights randomly initialized
- **Model**: `r-f/wav2vec-english-speech-emotion-recognition`
- **Issue**: Same as ehcalabres: untrained classifier heads. Scores near-random (0.14-0.16 for all emotions). Also tries to load kenlm decoder (not installed).
- **Test result**: neutral=0.16, fear=0.14, surprise=0.14: effectively random
- **Action**: REMOVE from options

### 5.6 HuBERT SER (SUPERB)
- **Status**: [x] Working
- **Package**: `transformers` (installed)
- **Model**: `superb/hubert-large-superb-er`
- **Input**: dict `{"raw": audio, "sampling_rate": 16000}`
- **API**: `pipeline("audio-classification", model=...)` then `pipe(audio_dict)`
- **Output**: list of `{"label": "hap", "score": 0.875}` sorted by score
- **Labels**: hap, sad, neu, ang (4 emotions, abbreviated)
- **Integration**: `analysis.py:101-113` via `_predict_emotion_pipeline`
- **Test result**: Works well. Returned hap=0.875 for calm/pleasant speech.
- **Caveat**: Only 4 emotion classes, abbreviated labels (not full words)
- **Device**: CPU or CUDA

### 5.7 WavLM SER
- **Status**: [BROKEN] Model not found on HuggingFace
- **Model**: `3loi/SER-Odyssey-Baseline-WavLM-Multi-Fused`
- **Issue**: `OSError: not a valid model identifier`: model appears to be private or removed
- **Action**: REMOVE from options

### 5.8 wav2small
- **Status**: [BROKEN] Incompatible with transformers audio-classification pipeline
- **Model**: `audeering/wav2small`
- **Issue**: No `preprocessor_config.json`: can't be loaded as standard pipeline. `AutoModelForAudioClassification` loads but has only 2 labels (LABEL_0, LABEL_1) with randomly initialized classifier. This is actually a regression model for arousal/dominance/valence, not a classification model.
- **Action**: REMOVE from options (would need custom loading code)

---

## 6. Diarization Engines

### 6.1 pyannote (3.1)
- **Status**: [N/A] Not installed
- **Package**: `pyannote.audio` (not installed)
- **Install**: `pip install pyannote.audio` + HuggingFace token for gated models
- **Model**: `pyannote/speaker-diarization-3.1`
- **Input**: file-like object (BytesIO with WAV) + uri
- **API**: `pipeline({"uri": "audio", "audio": buf})` → `diarization.itertracks(yield_label=True)`
- **Output**: iterable of (turn, _, speaker) with turn.start, turn.end
- **Integration**: `analysis.py:195-226`: code present, correct pattern
- **Requirements**: Accept terms at huggingface.co for pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0
- **Device**: CPU or CUDA

### 6.2 speechbrain (ECAPA-TDNN)
- **Status**: [N/A] Not installed
- **Package**: `speechbrain` (not installed)
- **Install**: `pip install speechbrain`
- **Model**: `speechbrain/spkrec-ecapa-voxceleb`
- **Input**: torch tensor chunks
- **API**: `model.encode_batch(tensor)` → embeddings → spectral clustering
- **Output**: embeddings for clustering (not end-to-end diarization)
- **Integration**: `analysis.py:229-302`: windowed embedding extraction + clustering
- **Device**: CPU or CUDA

### 6.3 resemblyzer
- **Status**: [x] Working
- **Package**: `resemblyzer` (installed)
- **Input**: preprocessed WAV (float32 at 16kHz)
- **API**: `encoder.embed_utterance(chunk)` → 256-dim d-vector embeddings → spectral clustering
- **Output**: embeddings for clustering, then merged segments
- **Integration**: `analysis.py:337-429`: windowed embeddings + cosine similarity + spectral clustering + merge
- **Test result**: Loads in 0.03s. Embedding shape (256,). Uses sklearn SpectralClustering.
- **Dependencies**: `resemblyzer`, `sklearn` (both installed)
- **Device**: CPU or CUDA

---

## 7. VAD Engines

### 7.1 silero (v5)
- **Status**: [x] Working
- **Package**: `torch` (installed, loaded from torch.hub)
- **Input**: torch tensor of audio samples
- **API**: `get_speech_timestamps(tensor, model, sampling_rate=sr)`
- **Output**: list of `{"start": sample_idx, "end": sample_idx}`
- **Integration**: `models.py:264-271` loads model + utils via torch.hub
- **Test result**: Found 1 speech segment (0.29s - 3.45s) in 5s test file.
- **Note**: Not directly used in transcription pipeline yet: available for future VAD filtering
- **Device**: CPU

### 7.2 pyannote-vad
- **Status**: [N/A] Not installed
- **Package**: `pyannote.audio` (not installed)
- **Install**: `pip install pyannote.audio` + HuggingFace token
- **Model**: `pyannote/segmentation-3.0`
- **Integration**: `models.py:272-285`: code present
- **Requirements**: Accept terms at huggingface.co for pyannote/segmentation-3.0

---

## Summary Matrix

### Working & Tested
| Engine | Category | Type | Status |
|--------|----------|------|--------|
| faster-whisper | STT | Native | [x] Production ready |
| wav2vec2-stt | STT | Native | [x] Works (ALL CAPS output) |
| speaches | STT | Docker | [x] Tested, 9s startup |
| whisper-asr | STT | Docker | [x] Tested, 51s startup |
| linto-nemo | STT | Docker | [x] Tested, 57s startup |
| linto-whisper | STT | Docker | [x] Tested, 42s startup |
| linto-nemo-1.1b | STT | Docker | [~] Works, lowercase only |
| piper | TTS | Native | [x] Production ready |
| espeak | TTS | Native | [x] Works (robotic) |
| qwen3-tts | TTS | Docker | [x] Tested, best quality |
| emotion2vec | Emotion | Native | [x] Works, 9 emotions |
| HuBERT SER | Emotion | Native | [x] Works, 4 emotions |
| resemblyzer | Diarization | Native | [x] Works |
| silero | VAD | Native | [x] Works |

### Installed But Not Fully Tested
| Engine | Category | Issue |
|--------|----------|-------|
| vosk | STT | Model needs download on first use |
| kokoro | TTS | Model download hangs |
| bark | TTS | Installed, needs live test |
| emotion2vec+ large | Emotion | Same API, larger model |
| xtts (Docker) | TTS | Image not pulled, config correct |
| parler (Docker) | TTS | Image not pulled, config correct |

### Not Installed
| Engine | Category | Install command |
|--------|----------|----------------|
| whisperx | STT | `pip install whisperx` |
| moonshine | STT | `pip install moonshine` |
| nemo | STT | `pip install nemo_toolkit[asr]` |
| chatterbox | TTS | `pip install chatterbox-tts` |
| pyannote | Diarization | `pip install pyannote.audio` + HF token |
| speechbrain | Diarization | `pip install speechbrain` |
| pyannote-vad | VAD | `pip install pyannote.audio` + HF token |

### BROKEN: Needs Fix or Removal
| Engine | Category | Issue | Action |
|--------|----------|-------|--------|
| wav2vec2-ser (ehcalabres) | Emotion | Untrained classifier, random scores | REMOVE |
| wav2vec2-ser (Dpngtm) | Emotion | Model removed from HuggingFace | REMOVE |
| wav2vec2-ser (r-f) | Emotion | Untrained classifier, random scores | REMOVE |
| WavLM SER (3loi) | Emotion | Model not found/private | REMOVE |
| wav2small | Emotion | Wrong model type for pipeline | REMOVE |
| chattts (Docker) | TTS | Endpoint/port unverified | VERIFY |
| melotts (Docker) | TTS | Wrong endpoint (`/tts/convert/tts` → `/convert/tts`) | FIX |
| orpheus (Docker) | TTS | Wrong endpoint (`/api/synthesize` → `/api/generate`) + wrong port (8080 → 8899) | FIX |
| fish-speech (Docker) | TTS | Image tag doesn't exist + WebUI-only (no REST API) | FIX image to `server-cuda` |
| cosyvoice (Docker) | TTS | Image doesn't exist on Docker Hub | BUILD or find alternative |

---

## Recommended Actions

### Priority 1: Remove broken engines from dropdown
1. Remove all 3 wav2vec2-ser models from MODEL_OPTIONS (broken classifiers)
2. Remove WavLM SER from MODEL_OPTIONS (model not found)
3. Remove wav2small from MODEL_OPTIONS (wrong model type)
4. Remove cosyvoice from MODEL_OPTIONS until working image found

### Priority 2: Fix Docker TTS configs
1. **melotts**: Change synth_url to `http://localhost:36312/convert/tts`
2. **orpheus**: Change port in compose to `36313:8899`, synth_url to `http://localhost:36313/api/generate`
3. **fish-speech**: Change image to `server-cuda`, port to `36314:8080`, synth_url to `http://localhost:36314/v1/tts`
4. **parler**: Add `response_type: "wav"` to params (returns MP3 by default)

### Priority 3: Pull and test remaining Docker images
1. Pull xtts image: `docker pull ghcr.io/coqui-ai/tts`
2. Pull parler image: `docker pull fedirz/parler-tts-server:latest`
3. Pull chattts image and verify API: `docker pull yikchunnnn/chattts-dockerized:latest`
4. Pull melotts image: `docker pull sensejworld/melotts:v0.0.4`
