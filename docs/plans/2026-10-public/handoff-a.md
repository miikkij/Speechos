# Handoff A → B / merge (2026-10-09)

Session A (branch `next/a-core`) did not touch any of B's files. This note lists what
changed in A's files that B's work and the merge depend on.

## 1. `api/src/models.py` and engine registration (merge step)

`ModelManager` keeps its public API (`get_stt()`, `get_tts()`, `unload()`,
`config.tts = ...`), so `api/bench/tts_fi.py` works unchanged. New pieces:

- `ModelManager.run(category, fn, language=None)` — async; resolves the cache key,
  then in a worker thread under that model's lock loads the model and calls
  `fn(model, cfg)`. Routers use only this.
- `ModelManager.resolve(category, language)` → `(cache_key, ModelConfig)`; TTS slots get
  per-language variants (`tts:piper:fi_FI-harri-medium`, `tts:kokoro:f`).
- Loaders are unchanged in shape: `_load_stt(cfg)`, `_load_tts(cfg)`, … return the model
  object. `ModelConfig` has a new optional `language` field (set by `resolve`, used by
  Kokoro).

To register a new engine from `api/src/engines_fi/` at merge time:

| Kind | Load | Inference | Menu |
|---|---|---|---|
| STT | branch in `ModelManager._load_stt` | branch in `transcribe_array()` in `api/src/routers/transcribe.py` — return `{text, segments, language, language_probability}` | `MODEL_OPTIONS["stt"]` + `_check_engine_installed` in `api/src/routers/system.py` |
| TTS | branch in `ModelManager._load_tts` | branch in `synthesize_wav()` in `api/src/routers/synthesize.py` — return mono 16-bit WAV bytes (use `_wav_bytes(_to_int16(audio), sr)`); `inp.language`, `inp.speed`, `inp.voice` are available | same, category `tts` |

Import `src.engines_fi.*` inside the loader branch, not at module top: CI imports
`src.models` without torch or model packages.

If the engine is single-language, add it to `_ENGLISH_ONLY_TTS`-style handling in
`ModelManager._resolve_tts` (or raise `UnsupportedLanguageError`) so a wrong `language`
gives a clear 400. Add a unit test with a fake model in `api/tests/` (pattern:
`tests/test_synthesize.py::test_chatterbox_multilingual_gets_language_id`) — tests must
not need torch or weights; real-model tests get `@pytest.mark.models`.

## 2. Tier defaults

All tiers now default TTS to Piper `fi_FI-harri-medium` (B's locked result). If B's
benchmark promotes another Finnish STT/TTS model, change `TIER_CONFIGS` /
`_PIPER_FI` in `api/src/config.py` and the matrix in
`docs/architecture/HARDWARE-TIERS.md`; `tests/test_config.py::test_every_tier_defaults_to_finnish_piper`
asserts the current default.

## 3. `api/pyproject.toml` / `api/uv.lock`

A added `ruff>=0.8.0` to the `dev` extra plus `[tool.pytest.ini_options]` and
`[tool.ruff]` sections (ruff excludes `bench/`). `uv lock` was regenerated on
`next/a-core`; per the README, regenerate again after merging B.

Suggestion for whoever owns the `gpu` extra: pyannote community-1 needs
`pyannote.audio>=4.0` (the extra says `>=3.3.0`). A's code handles both 3.x and 4.x
outputs.

## 4. CI

`.github/workflows/ci.yml` installs `api/requirements-ci.txt` (no torch) and runs
`pytest` on `api/tests/` only and `ruff check src tests`; ruff excludes `bench/` and
`src/engines_fi/` (`[tool.ruff] extend-exclude`) so B's code cannot turn CI red at merge —
remove that exclude once it passes. B's `api/bench/` and
`api/src/engines_fi/` are not imported by tests, so nothing breaks; if B adds tests
under `api/tests/`, they must run without torch or be marked `models`.
