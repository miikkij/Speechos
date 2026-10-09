# Sessio A — Ydin-API, web UI, testit, CI ja Docker

Olet itsenäinen Claude Code -sessio Speechos-repossa (`E:\dev\GitHub\Speechos`, Windows 11, PowerShell + Git Bash).
Speechos on local-first puheanalyysi- ja synteesialusta: `api/` (FastAPI, uv) ja `web/` (Next.js 15, pnpm).
Lue ensin `CLAUDE.md` ja `docs/plans/2026-10-public/README.md`.

## Säännöt

- Työskentele omassa worktreessä: `git worktree add ../Speechos-a -b next/a-core main` ja tee kaikki muutokset sinne.
  Sessio B ajaa rinnakkain omassa worktreessään.
- Muokkaa vain omistamiasi tiedostoja (lista alla). Jos tarvitset muutoksen B:n tiedostoon, kirjoita se
  `docs/plans/2026-10-public/handoff-a.md`:hen äläkä tee sitä itse.
- Riippuvuudet: uv (Python) ja pnpm (web). Lisää Python-riippuvuudet vain lisäyksinä. Aja `uv lock` omassa worktreessäsi
  ja commitoi; älä yritä sovittaa B:n lockfile-muutoksia.
- Commitoi pieninä conventional commit -committeina (`feat(api): …`, `fix(web): …`, `test: …`, `ci: …`) omalle branchillesi.
  ÄLÄ pushaa, älä tee tageja, älä luo PR:iä, älä koske `main`iin.
- Päächeckoutissa (`E:\dev\GitHub\Speechos`) voi olla seurattamattomia tiedostoja, jotka eivät kuulu tähän työhön.
  Älä koskaan aja `git add -A` / `git add .` päächeckoutissa; työskentele vain worktreessäsi ja lisää tiedostot nimeltä.
- Windows-prosessit: älä käynnistä palvelinta `while true` -taustasilmukassa. Sammuta itse käynnistämäsi prosessit
  (koko prosessipuu, `taskkill /T /F`). Käytä porttia **36500**.
- GPU on varattu sessio B:n benchmarkeille. Testisi eivät saa vaatia GPU:ta.
- Älä arvaa: tarkista kirjastojen API ja versiot asennetuista paketeista tai dokumentaatiosta.
- Lopuksi kirjoita `docs/plans/2026-10-public/report-a.md`: mitä tehtiin, mitä verifioitiin (komennot + tulokset),
  mikä jäi tekemättä ja miksi, sekä avoimet päätökset.

## Omistamasi tiedostot

`api/src/**` (paitsi `api/src/engines_fi/**`, joka on B:n), `api/tests/**` (uusi), `api/test_*.py`, `api/check_tts.py`,
`api/tmp_options.json`, `api/asset/**`, `api/pyproject.toml` (`dev`-extra ja ydinriippuvuudet), `web/**`, `docker/**`,
`.github/**` (uusi), `docs/architecture/**`, `README.md`, `CLAUDE.md`, `start.ps1`, `start.sh`, `dev.ps1`, `package.json`,
`config.example.yaml`. Älä koske: `api/bench/**`, `docs/research/**`.

## Lähtötilanne (analysoitu 9.10.2026)

- Viisi `api/test_*.py`:tä ovat käsin ajettavia skriptejä: ajautuvat import-hetkellä, ei asserteja, kovakoodatut polut.
  `.github/`-kansiota ei ole, joten CI:tä ei ole.
- Raskas inferenssi (STT, TTS, analyysi) ajetaan synkronisesti `async`-handlereissa ja blokkaa event loopin.
  `api/src/docker_tts.py` tekee blokkaavia subprocess-kutsuja.
- API tukee jo `language`-kenttää (transcribe Form-kenttä, synthesize JSON) ja `web/src/lib/api.ts` osaa lähettää sen,
  mutta UI (`web/src/app/page.tsx` ja komponentit) ei koskaan välitä kieltä: Chatterbox putoaa englantiin ja STT arvaa kielen.
- Tallennin tuottaa webm/opusta mutta tallentaa sen `*.wav`-nimellä. Piper ohittaa `speed`-parametrin.
  Kokoro on kovakoodattu englanniksi (`KPipeline(lang_code="a")`).
- `api/src/config.py`: kaikkien tierien TTS on Piper `en_US-lessac-medium`; `tts_fast`/`tts_expressive`/`tts_clone`-slotteja
  ei lueta; dokumentoidut `SPEECHOS_STT_*`/`SPEECHOS_TTS_*`/`SPEECHOS_EMOTION_*`-envit eivät toimi; VRAM-eviction ei toimi;
  `server.py` kovakoodaa CORSin eikä lue `config.cors_origins`ia. Diarisointi on kaikilla tiereillä Resemblyzer,
  vaikka `pyannote-audio 4.0.4` ja `pyannote/speaker-diarization-community-1` ovat saatavilla.
- `ModelManager._load_vad` lataa Silerosta lukitsemattoman `torch.hub`-version, eikä `get_vad()`ia kutsuta mistään.
- Docker: healthcheck osoittaa `:8000/api/health`, mutta sovellus palvelee `/health` portissa 36300.
  `docker/api.Dockerfile` käyttää CUDA 12.4.1 -baseimagea, vaikka torch on cu128.
- `docs/architecture/ARCHITECTURE.md` ja `HARDWARE-TIERS.md` (27.2.2026) lupaavat endpointteja ja tier-malleja, joita ei ole.
- Huom: `origin/main` ja paikallinen `main` eivät jaa historiaa. Älä yritä korjata sitä; se on omistajan päätös.

## Tehtävät prioriteettijärjestyksessä

1. **Testisuite.** Luo `api/tests/` (pytest, `conftest.py`, fixturet jotka generoivat pieniä 16 kHz mono -WAVeja).
   Testaa ilman oikeita malleja: tier-tunnistus (mockaa psutil ja torch.cuda), audiokonversio, routerit FastAPI
   TestClientilla ModelManager mockattuna, config-overridet, `language`-kentän läpivienti. Merkitse oikeita malleja
   vaativat testit `@pytest.mark.models`-merkillä ja ohita ne oletuksena. Siirrä vanhoista `api/test_*.py`-skripteistä
   talteen se mikä on hyödyllistä, poista loput. Poista `api/tmp_options.json` ja `api/check_tts.py`, jos mikään ei käytä
   niitä (tarkista grepillä). `api/asset/` (ChatTTS) poistetaan vain, jos varmistat ettei mikään tarvitse sitä.
2. **CI.** `.github/workflows/ci.yml`: api (uv, Python 3.12, kevyt asennus ilman CUDA-torchia — käytä CPU-indeksiä tai
   järjestä testit niin ettei torchia tarvita) → `uv run pytest`; ruff lint; web → `pnpm install --frozen-lockfile`,
   lint, `tsc --noEmit`, `pnpm build`. Validoi YAML (actionlint, jos saatavilla). Workflow ajaa vasta kun omistaja pushaa.
3. **Event loop vapaaksi.** Siirrä raskas inferenssi `asyncio.to_thread`/`run_in_threadpool`-kutsuihin. Lisää mallikohtainen
   lukko, jottei samaa mallia ajeta rinnakkain säieturvattomasti. Vaihda `docker_tts`:n subprocess-kutsut ei-blokkaaviksi.
   Testaa: `/health` vastaa kesken pitkän inferenssin.
4. **Suomi päästä päähän.** Web UI:hin kielivalitsin (auto / fi / en; oletus fi, valinta muistetaan localStorageen).
   UI välittää kielen transcribe- ja synthesize-kutsuihin. Piperille suomen ääni `fi_FI-harri-medium`, kun `language=fi`.
   Kokorolle selkeä virhe tai fallback, kun kieltä ei tueta. Korjaa Piperin `speed`.
5. **Oikea WAV.** Palvelin muuntaa jokaisen ladatun tai nauhoitetun tiedoston 16 kHz mono PCM -WAViksi ennen tallennusta
   (älä luota selaimen formaattiin). Testi webm-syötteelle.
6. **Tierit ja config.** Suomi-oletukset tiereihin (TTS fi = Piper harri-medium). GPU ≥ 12 GB -tiereillä diarisointi on
   pyannote community-1, kun `HF_TOKEN` on asetettu, muuten Resemblyzer. Toteuta dokumentoidut env-overridet
   (`SPEECHOS_STT_MODEL` jne.) jos toteutus on pieni, muuten poista ne dokumenteista. Lue CORS configista.
   Poista kuollut `get_vad`/`torch.hub`-polku tai korvaa se pinnatulla `silero-vad`-paketilla.
7. **Docker.** Korjaa healthcheckit (oikea portti ja polku). Nosta `api.Dockerfile`:n CUDA-base 12.8:aan, jotta se vastaa
   torch cu128 -buildia. Varmista että `docker compose config` validoituu kaikille compose-tiedostoille.
8. **Dokumentit.** Päivitä `ARCHITECTURE.md` ja `HARDWARE-TIERS.md` vastaamaan koodia; toteuttamattomat asiat
   "Roadmap"-osioon. Päivitä `CLAUDE.md`:n ja `README.md`:n komennot (esim. pytest-ajo, bench-extra).

## Valmis kun

- `cd api && uv run pytest` on vihreä ilman GPU:ta ja ilman mallilatauksia.
- `cd web && pnpm lint && pnpm build` menee läpi.
- API käynnistyy portissa 36500, `/health` vastaa ja suomenkielinen näyte (`samples/`) litteroituu `language=fi`:llä.
  Sammutat palvelimen itse lopuksi.
- `report-a.md` on kirjoitettu ja kaikki on commitoitu branchille `next/a-core`.
