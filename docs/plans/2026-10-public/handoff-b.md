# Handoff B → A / merge: suomen mallit (2026-10-09)

Session B touched no file under `api/src/**`. These are the steps that belong to the A side or to the merge.
Do not change any defaults until the listening review (`report-b.md`, "Open decisions").

## 1. STT: Finnish-NLP/whisper-large-finnish-v3 (won the FLEURS comparison) — config only, no code change

Result (FLEURS-fi test n=300): WER 6.2 % vs large-v3 7.5 % (95 % CI of the difference −2.1…−0.5 pp), CER
not significantly different, same speed (RTF 0.044). Finnish only — do not use it as the multilingual default.

**Model files:** the CT2 float16 model is in the main checkout (gitignored):
`E:\dev\GitHub\Speechos\models\whisper-ct2\whisper-large-finnish-v3-ct2-f16\`
(`model.bin` 3.1 GB, `config.json`, `tokenizer.json`, `preprocessor_config.json`, `vocabulary.json`,
`CONVERSION.txt`). Reproduce on another machine:

```bash
huggingface-cli download Finnish-NLP/whisper-large-finnish-v3 --revision b23deb0b3855c829ffe04cb1c6709757ff16d49c
ct2-transformers-converter --model <snapshot-dir> \
  --output_dir models/whisper-ct2/whisper-large-finnish-v3-ct2-f16 \
  --quantization float16 --copy_files tokenizer.json preprocessor_config.json
```

**Use today without code changes:** `ModelManager._load_stt` passes `cfg.model` straight to
`faster_whisper.WhisperModel`, which accepts a directory path. In `config.yaml`:

```yaml
stt:
  engine: faster-whisper
  model: ../models/whisper-ct2/whisper-large-finnish-v3-ct2-f16   # relative to the API's cwd (api/)
  device: cuda
  compute_type: float16
```

**Recommended small A change (optional, after the listening/user decision):** a language-specific STT slot,
so Finnish requests use the Finnish model and everything else uses large-v3:

- `api/src/config.py`: new optional field `stt_fi: ModelConfig | None = None`, add `"stt_fi"` to the key list
  in `_apply_yaml_overrides` (the same loop as `stt`, `tts_fast`, …).
- `api/src/models.py`: `get_stt(language: str | None = None)` → if `language == "fi"` and `config.stt_fi`,
  `_get_or_load("stt_fi", config.stt_fi, self._load_stt)`.
- `api/src/routers/transcribe.py`: pass `language` to `get_stt(...)`.
- Config key: `stt_fi: {engine: faster-whisper, model: <the path above>, device: cuda, compute_type: float16}`.
- VRAM: +~3.1 GB when both are loaded (fp16). Test: `uv run pytest` + a Finnish transcription through the
  API, compared against `api/bench/results/stt_fi_details_lock300.json`.

## 2. TTS: Chatterbox-Finnish — NOT registered (automatic criteria not met)

| Criterion | Requirement | Measured (chatterbox-finnish-ref, 3 seeds) | |
|---|---|---|---|
| CER mean | ≤ Piper + 1 pp = 5.65 % | 3.29 % | ✅ |
| Run-to-run sd | ≤ 1 pp | 0.26 pp | ✅ |
| RTF (GPU) | ≤ 0.5 | 1.28 (RTX 4090) | ❌ |

So `api/src/engines_fi/chatterbox_fi.py` was not created and there is nothing to register in `models.py`.
If the speed issue gets solved (see `report-b.md` "Open decisions"), the loader already exists:
`api/bench/tts_engines.py: load_chatterbox_finnish()` + `import_chatterbox_vendor()` (the exact model-card
inference path, revisions pinned) — move it to `engines_fi` and register it then.

## 3. Observations for the A side (no action required from B)

1. **Production Chatterbox loading cannot work in the current lock:** `models.py` imports `chatterbox.mtl_tts`
   (PyPI `chatterbox-tts`), whose latest version 0.1.7 pins `torch==2.6.0` and `transformers==5.2.0` → it does not
   install alongside torch 2.10 / transformers<5. The same code also comes bundled with Chatterbox-Finnish
   (MIT); the bench uses that one. If A wants Chatterbox in production, the cleanest path is the bundled code +
   the `tts-fi` extra (B's `pyproject` addition).
2. **The alignment analyzer requires eager attention** on transformers ≥ 4.47 (otherwise
   `ValueError: output_attentions … sdpa`). The bench sets
   `chatterbox_.models.t3.llama_configs.LLAMA_CONFIGS[*]["attn_implementation"] = "eager"` before building T3.
3. **The `nemo` STT engine** (`config.example.yaml`, Parakeet/Canary) cannot be installed into the API
   environment: nemo-toolkit 2.4–2.6 pin transformers~=4.53, 2.7.x needs setuptools≥79 (core pins `<75`),
   3.0 pins transformers 5.x. The bench runs it in an isolated uv project (`api/bench/nemo/`). If A wants NeMo
   in production: either drop the `setuptools<75` pin (then nemo 2.7.x fits transformers 4.57) or use a
   separate worker process.
4. **Piper licences:** `piper-tts` 1.4.1 is GPL-3.0-or-later. The voice `fi_FI-harri-medium` uses CC0 data
   but was fine-tuned from the `lessac` voice, whose data is under the Blizzard 2013 licence (research use).
   Worth checking if Speechos is ever distributed commercially.
5. **Piper fails on very short turns:** "Joo, ymmärrän." → ~6 s of babble in every run (duration predictor).
   If Piper stays the default, a guard is worth adding to the synthesis route (duration ≫ expected for the
   phoneme count → retry with a different `length_scale`/noise, or pad with a filler word).

## 4. Merge

- `api/pyproject.toml`: B added only the extras `bench-qwen`, `bench-tts`, `tts-fi` + one line in
  `[tool.uv.sources]` (`utmosv2` git rev, needed because hatchling does not allow direct references in extras).
- `api/uv.lock`: regenerate with `uv lock` at merge time as agreed.
- `api/bench/nemo/uv.lock` is B's own separate lock; it does not affect the API.
