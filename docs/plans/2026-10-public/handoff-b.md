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

**Recommended small A change (optional, after the user's decision):** a Finnish STT slot, so Finnish
requests use the Finnish model and everything else uses large-v3 (written against the `resolve()`/`run()` API
that A introduced; `main` @ 7788a94):

- `api/src/config.py`: new optional field `stt_fi: ModelConfig | None = None` and `"stt_fi"` in the key list in
  `_apply_yaml_overrides` (the same loop as `stt`, `tts_fast`, …).
- `api/src/models.py` `ModelManager.resolve()`: before `if category not in TTS_SLOTS:` add
  `if category == "stt" and base_language(language) == "fi" and self.config.stt_fi: return "stt:fi", self.config.stt_fi`.
  The cache key `stt:fi` makes `run_sync` pick `_loader_for("stt")` (it splits on `:`), so `_load_stt` needs no
  change (faster-whisper accepts a directory path).
- `api/src/routers/transcribe.py` line ~46: `get_models().run("stt", …, language=lang)`.
- Config key: `stt_fi: {engine: faster-whisper, model: <the path above>, device: cuda, compute_type: float16}`.
- Test (no torch): like `tests/test_transcribe.py`, fake `_load_stt` and assert that `language=fi` resolves to
  the `stt:fi` key and `language=en` to `stt`.
- VRAM: +~3.1 GB when both are loaded (fp16).

## 2. TTS: Chatterbox-Finnish — criteria met after the speedup → instructions to register as an *optional* engine

| Criterion | Requirement | Bundled loop | CUDA-graph decoder (`engines_fi/chatterbox_graph.py`) |
|---|---|---|---|
| CER mean (38 sentences × 3 seeds) | ≤ Piper + 1 pp = 5.65 % | 3.29 % ✅ | 3.29 % ✅ (bit-identical output) |
| Run-to-run sd | ≤ 1 pp | 0.26 pp ✅ | 0.26 pp ✅ |
| RTF (RTX 4090) | ≤ 0.5 | 1.28 ❌ | **0.43 ✅** (first audio 1.5 s) |

Module: `api/src/engines_fi/chatterbox_fi.py` (loading, pinned download, sentence-level synthesis, 24 kHz,
~4.4 GB VRAM — details in the docstring). Do not change the default: Piper stays the default until the listening
review (report-b.md). Registration following A's handoff (`handoff-a.md` §1) — engine name `chatterbox-fi`:

1. **Load** — `api/src/models.py` `ModelManager._load_tts`, a new branch (import inside the branch):
   ```python
   if cfg.engine == "chatterbox-fi":
       from src.engines_fi.chatterbox_fi import ChatterboxFinnish
       # cfg.model: "sample" (repo reference voice) or a path to a Finnish reference WAV; "" = built-in voice
       return ChatterboxFinnish.load(self.model_dir, device=cfg.device, ref_wav=cfg.model or None)
   ```
2. **Single language** — `ModelManager._resolve_tts`: before the `_ENGLISH_ONLY_TTS` check,
   ```python
   if cfg.engine == "chatterbox-fi" and lang not in (None, "fi"):
       raise UnsupportedLanguageError("Chatterbox-Finnish only speaks Finnish ('fi').")
   ```
3. **Inference** — `api/src/routers/synthesize.py` `synthesize_wav()`, a new branch:
   ```python
   if engine == "chatterbox-fi":
       if inp.voice and inp.voice != tts.ref_wav:
           tts.set_voice(inp.voice)          # path to a reference WAV, like the chatterbox branch
       audio, sr = tts.synthesize(text)       # numbers/abbreviations normalized inside (normalize_fi)
       return _wav_bytes(_to_int16(audio), sr)
   ```
   `inp.speed` is not supported (the model has no tempo control) — ignore it, or resample afterwards.
4. **Menu** — `api/src/routers/system.py`: in `MODEL_OPTIONS["tts"]`
   `{"engine": "chatterbox-fi", "model": "sample", "label": "Chatterbox-Finnish (fi, voice cloning, MIT, GPU)"}`;
   in `_check_engine_installed` `elif engine == "chatterbox-fi": import perth, s3tokenizer, num2words  # noqa: F401`;
   add `"chatterbox-fi"` to `GPU_CAPABLE_TTS`.
5. **Dependencies** — the `tts-fi` extra (already in `pyproject.toml`). Docker GPU image: `uv sync --extra tts-fi`.
   The bundled code downloads the pinned snapshot to `<model_dir>/chatterbox-finnish/` on first load (~1 GB) +
   the HF cache (~3.2 GB).
6. **Test** (no torch) — like `tests/test_synthesize.py::test_chatterbox_multilingual_gets_language_id`: fake
   `_load_tts` → an object with `ref_wav`, `set_voice()`, `synthesize()` returning `(np.zeros(2400), 24000)`;
   assert 200 for `language=fi` and 400 for `language=en`.
7. **Ruff** — `src/engines_fi` passes `ruff check` (verified with `--config "extend-exclude=[]"`); remove it
   from `[tool.ruff] extend-exclude`. `bench/` stays excluded.

## 2b. Finnish text normalization in front of Piper (recommended, small)

`api/src/engines_fi/text_norm_fi.py: normalize_fi(text)` spells out numbers, dates, money, percentages, clock
times, phone numbers, abbreviations and acronyms. Targeted test (10 number sentences × 3 seeds,
`api/bench/results/tts_fi_number_items.md`): Piper key items correct 80 % → 91 %, Chatterbox-Finnish 40 % → 99 %.
In `api/src/routers/synthesize.py` `synthesize_wav()`, Piper branch, before synthesis (extend the existing
`from src.lang import UnsupportedLanguageError` with `base_language, piper_voice_language`):

```python
if base_language(inp.language) == "fi" or piper_voice_language(cfg.model) == "fi":
    from src.engines_fi.text_norm_fi import normalize_fi   # needs num2words (tts-fi extra; also a kokoro dependency)
    text = normalize_fi(text)
```

The Chatterbox-Finnish module already normalizes internally. Tests: `api/bench/test_text_norm_fi.py` (18 tests,
no torch) — move them to `api/tests/` if you want them in CI (they need `num2words`).

## 3. Observations for the A side (no action required from B)

1. **Production Chatterbox loading cannot work in the current lock:** `models.py` imports `chatterbox.mtl_tts`
   (PyPI `chatterbox-tts`), whose latest version 0.1.7 pins `torch==2.6.0` and `transformers==5.2.0` → it does not
   install alongside torch 2.10 / transformers<5. The same code also comes bundled with Chatterbox-Finnish
   (MIT); the bench uses that one. If A wants Chatterbox in production, the cleanest path is the bundled code +
   the `tts-fi` extra (B's `pyproject` addition).
2. **The alignment analyzer requires eager attention** on transformers ≥ 4.47 (otherwise
   `ValueError: output_attentions … sdpa`). `engines_fi.chatterbox_fi.import_vendor()` sets the bundled
   `LLAMA_CONFIGS[*]["attn_implementation"] = "eager"`; the graph decoder switches all but the analyzer's layers to
   SDPA. The same decoder (`enable_cuda_graph(model)`) also works for the existing `chatterbox` engine if it is moved
   to the bundled code.
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
