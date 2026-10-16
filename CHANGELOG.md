# Changelog

All notable changes to Speechos are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-16

First release: a local-first speech workspace that records, transcribes, analyzes and synthesizes speech on your own
machine, with Finnish as a first-class language.

### Added

- **Web UI** with four views (Next.js 16, React 19, Tailwind 4):
  - **Capture**: record from any microphone with a live level meter, or drop in an audio file.
  - **Analyze**: recordings library with search, waveform with speaker and speech-activity lanes, a transcript that
    seeks the audio, insights (summary, emotion, speakers, pitch, speaking rate, loudness) and export to TXT, SRT
    or JSON.
  - **Synthesize**: voice cards filtered by language with benchmark notes, speed control, waveform preview and
    WAV download.
  - **Models**: hardware tier and per-category model choice with install status.
- **Finnish speech recognition** with Finnish-NLP/whisper-large-finnish-v3 for `language=fi` on GPU tiers
  (FLEURS-fi WER 6.2 % vs 7.5 % for large-v3), enabled when its CTranslate2 conversion is on disk.
- **Chatterbox-Finnish** as an optional text-to-speech engine (`--extra tts-fi`): CER 3.3 %, 0.43× real time on an
  RTX 4090 with CUDA-graph decoding, voice cloning from a Finnish reference clip.
- **Finnish text normalization** for TTS input (numbers, dates, money, clock times, phone numbers, abbreviations).
- **Finnish STT/TTS benchmark** (`api/bench/`): WER/CER/RTF on FLEURS-fi, Common Voice and VoxPopuli; TTS round-trip
  CER over several seeds, UTMOS and a blinded listening page.
- Hardware tiers with Finnish-first defaults, plus overrides through `config.yaml` and `SPEECHOS_*` variables.
- pytest suite that needs no GPU, torch or model weights, and GitHub Actions CI for the API and the web app.
- Docker images for GPU (CUDA 12.8, with pyannote and Chatterbox-Finnish) and CPU, `dev.ps1` / `dev.sh` launchers.
- MIT license.

### Changed

- Model inference and audio decoding run off the event loop with a lock per model, so the API stays responsive.
- Uploaded and recorded audio is always stored as 16 kHz mono WAV, whatever the browser sends.
- Speaker analysis uses pyannote community-1 when `HF_TOKEN` is set and its terms are accepted, and falls back to
  Resemblyzer automatically otherwise.
- Dependencies updated: PyTorch 2.11 (CUDA 12.8), FastAPI 0.143 / Starlette 1.7, NumPy 2.5, librosa 1.0.

### Fixed

- Speaker turns from Resemblyzer were compressed into the first two thirds of a recording.
- Loudness dynamic range reported values like 187 dB because of digital silence.
- The `/models` web route was excluded by `.gitignore`.
- Shell scripts are kept on LF line endings so they run on Linux and macOS.

### Known limitations

- Piper's Finnish voice garbles some short replies (for example "Ymmärrän."); Chatterbox-Finnish reads them correctly.
- Streaming transcription and synthesis over WebSocket are not available yet.
- Models stay in memory once loaded; several large models together need a 24 GB GPU.

[0.1.0]: https://github.com/miikkij/Speechos/releases/tag/v0.1.0
