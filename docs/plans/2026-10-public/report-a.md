# Sessio A — raportti (9.10.2026)

Branch `next/a-core` (worktree `../Speechos-a`), 12 committia `main`in (`a8ad49a`) päälle.
Ei pusheja, tageja eikä PR:iä. B:n tiedostoihin ei koskettu; B:lle ja mergeen liittyvät
huomiot ovat tiedostossa [handoff-a.md](handoff-a.md).

## Mitä tehtiin

### 1. Testisuite (`api/tests/`, 110 testiä + 6 `models`-testiä)
- `conftest.py`: generoidut 16 kHz mono -WAVit, WebM/Opus-generaattori (PyAV, kuten
  Chromen MediaRecorder), fake-Whisper/Piper, `create_app()`-pohjainen TestClient ja
  httpx `ASGITransport` -asiakas.
- Kattaa: tier-tunnistus (psutil ja torch.cuda mockattu `sys.modules`-tasolla), rikkinäinen
  CUDA, config-overridet (env, YAML, precedence), CORS, audiokonversio, kaikki routerit,
  `language`-kentän läpivienti (transcribe + synthesize), Piperin speed, Kokoro/Kitten-virheet,
  Chatterboxin `language_id`, slotit, Docker-TTS:n async-subprocessit, `/health` kesken
  pitkän inferenssin, mallikohtainen lukko (rinnakkaisuus = 1).
- `@pytest.mark.models` (oletuksena ohitettu, `-m models`): oikea Piper fi/en, Whisper
  suomi-litterointi ja kielentunnistus, Silero VAD, Piperin speed. Siirretty talteen
  vanhoista skripteistä (`test_integration.py`, `test_vad_stt.py`, `test_all_models.py`).
- Poistettu: `api/test_*.py` (5 kpl), `api/check_tts.py`, `api/tmp_options.json`
  (grep: ei viittauksia), `api/asset/` (ChatTTS-tokenizer; ChatTTS on vain
  Docker-moottori — `is_docker_engine` ohittaa natiivilatauksen, eikä ChatTTS-pakettia ole
  asennettu).
- `pyproject.toml`: `ruff` dev-extraan, pytest- ja ruff-konfiguraatio.

### 2. CI (`.github/workflows/ci.yml`)
- api: Python 3.12, `uv venv` + `uv pip install -r api/requirements-ci.txt` (versiot
  uv.lockista, ei torchia/CUDAa/mallikirjastoja), `ruff check src tests`, `pytest`.
- web: pnpm (versio `packageManager`-kentästä), `install --frozen-lockfile`, `lint`,
  `typecheck`, `build`.
- `next lint` vaihdettu ESLint flat configiin (`web/eslint.config.mjs`): vanha komento oli
  interaktiivinen ilman konfiguraatiota eikä olisi toiminut CI:ssä.

### 3. Event loop vapaaksi
- `ModelManager.run(category, fn)`: lataus + inferenssi worker-säikeessä
  (`asyncio.to_thread`) mallikohtaisen `threading.Lock`in alla (lukko otetaan säikeessä, joten
  se pitää myös jos pyyntö peruuntuu). Audion dekoodaus ja librosa-piirteet myös säikeissä.
- `/api/analyze` ajaa piirteet, emootion, diarisoinnin ja VAD:n rinnakkain.
- `docker_tts`: `asyncio.create_subprocess_exec` (timeout + kill), async readiness-poll,
  `shlex.quote`; fallback säikeeseen, jos event loop ei tue subprocesseja (Windows
  SelectorEventLoop, `workers > 1`).

### 4. Suomi päästä päähän
- Web: kielivalitsin (Suomi / English / Auto), oletus `fi`, muistetaan localStorageen;
  kieli kulkee transcribe- ja synthesize-kutsuihin. TTS:ään nopeusvalinta.
- Piper: pyynnön kieli valitsee äänen (`fi` → `fi_FI-harri-medium`, `en` →
  `en_US-lessac-medium`, myös sv/de), välimuistissa omalla avaimella. `speed` →
  `SynthesisConfig.length_scale = 1/speed`.
- Kokoro: oma `KPipeline` per kielikoodi; suomi → 400 + ohje käyttää Piperiä. Kitten/Parler:
  muut kuin englanti → 400.

### 5. Oikea WAV
- `POST /api/recordings` dekoodaa ja koodaa aina 16 kHz mono PCM_16 -WAViksi
  (`to_wav_16k_mono`). PyAV-dekoodaus resamplaa nyt packed float monoksi (vanha koodi
  keskiarvoisti väärin pakatuille formaateille ja arvasi skaalauksen).
- Testit: WebM/Opus-upload → tallennettu tiedosto on RIFF/WAVE 16 kHz mono PCM_16.

### 6. Tierit ja config
- Kaikkien tierien TTS = Piper `fi_FI-harri-medium`; cpu-2gb STT Vosk (englanti) →
  Whisper tiny.
- GPU ≥ 12 GB -tierit: pyannote community-1, kun `HF_TOKEN` on asetettu ja
  `pyannote.audio` on asennettu, muuten Resemblyzer. Diarisointi tukee pyannote 4:n
  `DiarizeOutput`ia ja in-memory-waveformia; dispatch moottorinimen mukaan (vanha
  tyyppinimitarkistus ei olisi tunnistanut pyannote 4:n pipelinea).
- Dokumentoidut env-overridet toteutettu: `SPEECHOS_<SLOT>_{ENGINE,MODEL,DEVICE,COMPUTE_TYPE}`
  + `SPEECHOS_WORKERS`, `_RECORDINGS_DIR`, `_CORS_ORIGINS`, `_MODEL_LOADING`, `_CONFIG`.
  YAML:n `tier`/`compute_mode`/`storage.samples_dir` luetaan; tierin nimi validoidaan.
- CORS luetaan configista (`create_app()`-tehdas; `python -m src` käyttää sitä).
- `tts_fast`/`tts_expressive`/`tts_clone` käytettävissä `slot`-kentällä.
- `get_vad`/`torch.hub` korvattu pinnatulla `silero-vad`-paketilla; VAD on nyt käytössä
  `/api/analyze`n `speech`-osiossa.
- `model_loading: eager` toteutettu (taustalataus).
- Korjattu löydetty bugi: `CUDA_VISIBLE_DEVICES=""` kaatoi käynnistyksen
  (`is_available()` True mutta laite 0 virheellinen) → CPU-fallback.

### 7. Docker
- Healthcheck `:36300/health`, `CMD python -m src` (vanha `python -m src.server` ei
  käynnistänyt mitään), CUDA-base `12.8.1-runtime-ubuntu22.04`, Python 3.12 uv:lla,
  `uv sync --frozen --no-install-project`, build-essential deps-vaiheeseen (webrtcvad),
  recordings `/recordings`-volumeen, web-image portissa 36301 ilman puuttuvaa `public/`ia,
  `NEXT_PUBLIC_API_URL` build-argumenttina, per-Dockerfile `.dockerignore`.

### 8. Dokumentit
- `ARCHITECTURE.md`, `HARDWARE-TIERS.md` kirjoitettu koodin mukaisiksi, toteuttamattomat
  Roadmap-osioon. README, CLAUDE.md, config.example.yaml, start.sh/start.ps1 (dev-tilan
  `python -m src.server` → `python -m src`) päivitetty.
- Lisätty `samples/fi_harri_sample.wav` (7,9 s, Piperillä tuotettu).

## Verifiointi (komennot ja tulokset)

| Komento | Tulos |
|---|---|
| `cd api && uv run pytest -q` (Python 3.13, täysi env) | **110 passed, 6 deselected** |
| sama Python 3.12 -venvissä pelkällä `requirements-ci.txt`:llä (`import torch` → ModuleNotFoundError), `uv run --no-sync pytest -q` | **110 passed, 6 deselected** |
| `uv run ruff check src tests` | All checks passed |
| Mutaatiotesti: `ModelManager.run` muutettu synkroniseksi → `test_health_answers_during_long_inference` | **failasi** (testi mittaa oikeaa asiaa), palautettu |
| `CUDA_VISIBLE_DEVICES="" SPEECHOS_MODEL_DIR=<main>/api/models SPEECHOS_TEST_STT_MODEL=medium uv run pytest -m models` | **6 passed** (CPU) |
| `cd web && pnpm lint && pnpm typecheck && pnpm build` | OK (`/` 7.95 kB) |
| actionlint 1.7.7 (Docker) `.github/workflows/ci.yml` | exit 0 |
| `docker compose -f … config -q` gpu / cpu / tts-engines | kaikki OK |
| `docker build` web → ajo | healthy, HTTP 200 portissa 36301 |
| `docker build` api-cpu → ajo | healthy, `/api/system/info` cpu-32gb (image 9,65 GB) |
| `docker build` api (GPU) → ajo **ilman** `--gpus` | healthy, Python 3.12.13, torch 2.10.0+cu128, CPU-tila (image 13 GB) |
| Live: API portissa **36500** (`SPEECHOS_COMPUTE_MODE=cpu`, `SPEECHOS_TIER=cpu-32gb`, `CUDA_VISIBLE_DEVICES=""`) | ks. alla |

Live-ajo (`scratchpad/live_check.py` httpx:llä):
- `/health` → `ok`; tier `cpu-32gb`, STT Whisper medium int8 (cpu), TTS Piper harri.
- `POST /api/synthesize {language: fi}` → 200, `X-TTS-Model: fi_FI-harri-medium`, 2,3 s
  → tallennettu `samples/fi_harri_sample.wav` (16 kHz mono PCM_16, 7,94 s).
- Piper speed 0.75 vs 1.5: 1,59 s vs 0,84 s. `language: en` → `en_US-lessac-medium`.
- **`POST /api/transcribe language=fi`** → 200, 15,7 s, `language: fi`:
  "Hyvää huomenta! Tänä on suomenkielinen testinäyte SPKOS-palvelulle. Puheen tunnistus
  toimii nyt myös Suomeksi." (Whisper medium CPU:lla; virheet "Tänä", "SPKOS" ovat mallin.)
- `/health` pollattiin litteroinnin aikana: 77 vastausta, **max 127 ms**.
- `language=auto` → tunnistettu `fi` (p = 0,986).
- `POST /api/recordings` → tallentui 16 kHz WAViksi, poistettiin.
- Palvelin sammutettiin: `taskkill /PID 81272 /T /F` (3 prosessia), portti 36500 vapaa.

GPU:ta ei käytetty: kaikki testit, live-ajo ja Docker-ajot olivat CPU:lla.

## Mitä jäi tekemättä ja miksi

- **VRAM-eviction / `vram.*`-asetukset**: ei toteutettu (vaatisi VRAM-budjetoinnin ja
  GPU-testauksen, GPU varattu B:lle). Poistettu config.example.yamlista, Roadmapissa.
- **WebSocket-streaming, `/api/voices`, erilliset analyysiendpointit**: eivät kuuluneet
  tehtävälistaan; siirretty dokumenteissa Roadmapiin.
- **CPU-only torch Docker-imageen**: lockfile pinnaa cu128-torchin kaikille alustoille;
  CPU-image toimii mutta on 9,65 GB. Vaatisi erillisen lockin/indeksin.
- **pyannote community-1:tä ei ajettu oikeasti**: pyannote.audio ei ole asennettuna
  (`gpu`-extra) eikä HF_TOKENia ollut; polku on testattu vain mockeilla (valinta) ja
  koodi tukee sekä 3.x- että 4.x-tulostetta.
- **GPU-imagea ei ajettu GPU:lla** (vain CPU-tilassa ilman `--gpus`).
- **UI:ta ei testattu selaimessa**; lint, typecheck ja build menevät läpi, API-kutsujen
  muoto on testattu API-päässä.

## Avoimet päätökset omistajalle

1. **`samples/fi_harri_sample.wav`**: Piper harri -mallin tuottama. Harrin datasetti on CC0,
   mutta malli on hienosäädetty lessac-äänestä (Blizzard 2011 -datasetti, rajoitetumpi
   lisenssi). Pidetäänkö näyte repossa vai generoidaanko se testissä lennossa
   (`test_models_integration.py` tekee jo niin, jos tiedostoa ei ole)?
2. **pyannote-versio**: `gpu`-extra sanoo `pyannote.audio>=3.3.0`; community-1 vaatii 4.x.
   Extra ei ole A:n eikä B:n omistama — nostetaanko `>=4.0`?
3. **cpu-2gb STT**: vaihdettu Voskista Whisper tinyyn suomen takia. Jos 2 GB:n koneilla
   halutaan Vosk, tarvitaan suomenkielinen Vosk-malli.
4. **Kokoro + suomi**: nyt selkeä 400-virhe. Vaihtoehto olisi automaattinen fallback Piperiin.
5. **Rivinvaihdot**: repo on enimmäkseen CRLF (ei `.gitattributes`ia). Muokatut tiedostot
   palautettiin CRLF:iksi; uudet tiedostot (tests, workflow, lang.py, …) ovat LF.
   `.gitattributes` kannattaisi lisätä (ei A:n omistama tiedosto).
6. **`origin/main` vs paikallinen `main`**: ei kosketty, kuten ohjeistettiin.
7. Docker-verifioinnin lopuksi ajoin `docker builder prune -f`, joka poisti myös muiden
   projektien käyttämättömän build-cachen WSL:n Dockerista (ei imageja, kontteja eikä
   volumeja). Cache rakentuu uudelleen seuraavassa buildissa.

## Merge-muistiinpanot

- Mergen jälkeen: `cd api && uv lock && uv run pytest`, `cd web && pnpm build`.
- B:n moottorien rekisteröinti: ks. [handoff-a.md](handoff-a.md) (lataus `_load_*`,
  inferenssi `transcribe_array()` / `synthesize_wav()`, valikko `MODEL_OPTIONS`).
