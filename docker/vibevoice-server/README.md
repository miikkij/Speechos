# VibeVoice TTS server

Long-form, multi-speaker ("podcast") TTS for Speechos, served behind an
OpenAI-compatible HTTP API.

- **Model**: [`vibevoice-community/VibeVoice`](https://github.com/vibevoice-community/VibeVoice)
  (community fork of Microsoft VibeVoice; MIT). Default weights `vibevoice/VibeVoice-1.5B`
  (~7–8 GB VRAM). Set `VIBEVOICE_MODEL` to `vibevoice/VibeVoice-7B` for the larger model
  (~24 GB, or ~8 GB with a 4-bit quant).
- **API**: `POST /v1/audio/speech` with
  `{"model": "...", "input": "...", "voice": "Frank", "response_format": "wav"}`.
- **Multi-speaker**: prefix lines in `input` with `Speaker 0:` / `Speaker 1:` (up to 4).
- **Output**: 24 kHz WAV. **GPU-only** (CUDA + PyTorch).

## Run via Speechos

The API auto-manages this container. Selecting the `vibevoice` TTS engine
(`POST /system/switch-model`) starts it on demand. Manual control:

```bash
# from PowerShell (WSL2 docker)
bash -c "docker compose -f /mnt/e/dev/GitHub/Speechos/docker/tts-engines.yml up -d --build vibevoice"
bash -c "docker compose -f /mnt/e/dev/GitHub/Speechos/docker/tts-engines.yml stop vibevoice"
```

## Notes

- First request downloads the model from HuggingFace into the `tts-vibevoice-models` volume.
- Only one GPU-heavy TTS container runs at a time (Speechos stops the others on switch).
- Microsoft pulled the original repo in Sep 2025; this uses the MIT-licensed community fork.
  The model card asks for "research only" use — treat that as an advisory compliance flag.
