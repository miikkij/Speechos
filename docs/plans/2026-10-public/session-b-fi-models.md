# Sessio B — FI-benchmark 2.0: uudet suomen STT- ja TTS-mallit

Olet itsenäinen Claude Code -sessio Speechos-repossa (`E:\dev\GitHub\Speechos`, Windows 11, PowerShell + Git Bash, NVIDIA-GPU).
Speechosissa on suomen kielen laatubenchmark `api/bench/` (`stt_fi.py`, `tts_fi.py`, `make_ref.py`, `results/`).
Lue ensin `CLAUDE.md`, `docs/plans/2026-10-public/README.md` ja `api/bench/README.md`.

## Säännöt

- Työskentele omassa worktreessä: `git worktree add ../Speechos-b -b next/b-fi-models main` ja tee kaikki muutokset sinne.
  Sessio A ajaa rinnakkain omassa worktreessään.
- Muokkaa vain omistamiasi tiedostoja (lista alla). Jos tarvitset muutoksen A:n tiedostoon (esim. moottorin rekisteröinti
  `api/src/models.py`:hin), kirjoita tarkka ohje `docs/plans/2026-10-public/handoff-b.md`:hen äläkä tee sitä itse.
- Riippuvuudet: uv. Lisää riippuvuudet vain omiin extroihisi (`bench`, uudet `bench-nemo`, `bench-qwen`, `tts-fi` tms.).
  Aja `uv lock` omassa worktreessäsi ja commitoi; älä yritä sovittaa A:n lockfile-muutoksia. Raskaat tai Windowsilla
  hankalat paketit (NeMo, vLLM) erillisiin extroihin; jos asennus ei onnistu, ohita malli hallitusti ja kirjaa syy.
- Commitoi pieninä conventional commit -committeina (`feat(bench): …`, `docs(research): …`) omalle branchillesi.
  ÄLÄ pushaa, älä tee tageja, älä luo PR:iä, älä koske `main`iin.
- Päächeckoutissa (`E:\dev\GitHub\Speechos`) voi olla seurattamattomia tiedostoja, jotka eivät kuulu tähän työhön.
  Älä koskaan aja `git add -A` / `git add .` päächeckoutissa; työskentele vain worktreessäsi ja lisää tiedostot nimeltä.
- Windows-prosessit: älä käynnistä mitään `while true` -taustasilmukassa; sammuta itse käynnistämäsi prosessit.
- **Olet ainoa raskas GPU-käyttäjä.** Aja silti mallit yksi kerrallaan ja vapauta VRAM (`del` + `torch.cuda.empty_cache()`)
  mallien välissä.
- Worktreessä ei ole gitignoroituja tiedostoja. Käytä päächeckoutin jo ladattuja malleja absoluuttisella polulla
  (`E:\dev\GitHub\Speechos\models` ja `E:\dev\GitHub\Speechos\api\models`, esim. ympäristömuuttujalla tai bench-skriptin
  argumentilla) äläkä lataa niitä uudelleen. `HF_TOKEN` on tiedostossa `E:\dev\GitHub\Speechos\.env`; lue se sieltä,
  älä kopioi sitä worktreehen äläkä tulosta sitä. Kunnioita gated-mallien
  ja -datasettien ehtoja: jos pääsy puuttuu, kirjaa se raporttiin äläkä kierrä sitä.
- Älä arvaa: tarkista mallikortit ja kirjastojen API ennen käyttöä. Tallenna jokaisen mallin tarkka HF-revisio tuloksiin.
- Lopuksi kirjoita `docs/plans/2026-10-public/report-b.md`: mitä tehtiin, mitä verifioitiin (komennot + tulokset),
  mikä jäi tekemättä ja miksi, sekä avoimet päätökset.

## Omistamasi tiedostot

`api/bench/**`, `docs/research/**`, `api/src/engines_fi/**` (uusi), `api/pyproject.toml`:n `bench*`- ja uudet mallikohtaiset
extrat (vain lisäyksiä). Älä koske muuhun `api/src/**`:iin, `web/**`:iin, `docker/**`:iin tai `.github/**`:iin.

## Lähtötilanne (lukitut tulokset 19.6.2026)

- STT: faster-whisper large-v3, FLEURS-fi n=100: WER 7,9 %, CER 1,46 %, RTF 0,058, mediaanilatenssi 0,66 s.
  Common Voice -fi ja 300 klipin lukitusajo ajamatta. Muita STT-malleja ei verrattu.
- TTS round-trip (n=30): Piper `fi_FI-harri-medium` CER 3,5 % (lukittu valinta), perus-Chatterbox Multilingual
  CER 4,9–16,5 % ajojen välillä (epävakaa, ~30× hitaampi), eSpeak 10,3 %. SQUIM-MOS osoittautui epäluotettavaksi.
  UTMOS on TODO (`tts_fi.py`). edge-tts:ää ei ole benchmarkattu.

## Uudet kandidaatit (haettu 9.10.2026, varmista mallikortista ennen käyttöä)

STT:
- `Finnish-NLP/whisper-large-finnish-v3` (Apache-2.0): raportoitu FLEURS WER 9,63 → 8,21, Common Voice 10,82 → 8,23.
  Muunna CTranslate2-muotoon (`ct2-transformers-converter`, float16) faster-whisperia varten ja kirjaa komento.
- faster-whisper `large-v3-turbo` (nopeus vs. tarkkuus).
- `nvidia/parakeet-tdt-0.6b-v3` (CC BY 4.0, 25 kieltä sis. fi, FLEURS fi WER 13,21 %, striimaus NeMolla). Kiinnostava
  nopeutensa vuoksi.
- `Qwen/Qwen3-ASR-1.7B` (Apache-2.0, fi tuettu; transformers-backend). Valinnainen.

TTS:
- `Finnish-NLP/Chatterbox-Finnish` (MIT; Chatterbox-hienosäätö suomelle): raportoitu WER 28,94 % → 2,76 %,
  MOS 4,34 (Gemini-arvio, ei ihmisraati). Tärkein kandidaatti: paikallinen, luonnollinen suomen ääni äänikloonauksella.
  Mallikortin suositusasetukset: repetition_penalty 1.2, temperature 0.8, exaggeration 0.6.
- edge-tts `fi-FI-NooraNeural` / `HarriNeural` — pilvireferenssi, johon paikallisia verrataan (ei paikallinen vaihtoehto).
- `facebook/mms-tts-fin` (CC-BY-NC 4.0, vain arviointiin).
- `TheBurgstall/ZipVoice-Suomi-12k` ja `ArttuPakarinen/sesame-csm-FIN-parlament-full-finetune` (koulutettu
  CC-BY-NC-ND Nord-Parl-TTS-datalla, vain arviointiin). Valinnaisia, jos ajettavissa kohtuudella.
- Perus-Chatterbox Multilingual uudelleen vertailukohdaksi; Piper-fi baseline.

## Tehtävät prioriteettijärjestyksessä

1. **TTS-vakausmittaus.** Laajenna `tts_fi.py`: jokainen moottori ajetaan ≥ 3 kertaa eri seedeillä samoilla lauseilla,
   raportoi CER/WER keskiarvo ja keskihajonta, RTF, time-to-first-audio lauseelle sekä lisenssi-sarake.
   Lisää lyhyitä keskustelulauseita ("Joo, ymmärrän.", "Miltä se tuntui?").
2. **UTMOS.** Toteuta UTMOS (esim. `speechmos`/UTMOSv2; tarkista mikä toimii) suhteelliseksi luonnollisuusmittariksi.
   Kirjaa, että se on koulutettu pääosin englannilla eikä korvaa kuuntelua.
3. **TTS-ajot.** Aja kaikki TTS-kandidaatit. Tallenna 10 klippiä per moottori `api/bench/listen/`-kansioon (gitignored)
   ja tee siitä yksinkertainen HTML-kuuntelusivu (sokkoutettu järjestys, avain erillisessä tiedostossa).
4. **STT-ajot.** Lisää kandidaatit `stt_fi.py`:hin. Aja FLEURS-fi ja Common Voice -fi n=100, sitten 300 klipin lukitusajo
   kahdelle parhaalle. Lisää erillinen mittaus lyhyille klipeille (1–5 s), koska keskustelukäytössä vuorot ovat lyhyitä.
5. **Moottori (ehdollinen).** Jos Chatterbox-Finnish täyttää automaattikriteerit — keskimääräinen CER ≤ Piper + 1 pp,
   hajonta ajojen välillä ≤ 1 pp ja RTF ≤ 0,5 GPU:lla — toteuta se itsenäisenä moduulina `api/src/engines_fi/chatterbox_fi.py`
   (lataus, lausetason synteesi, näytetaajuus, VRAM-tarve docstringissä) ja kirjoita `handoff-b.md`:hen tarkka ohje sen
   rekisteröimiseksi `api/src/models.py`:hin ja moottorivalikkoon. Sama STT:lle, jos suomi-Whisper voittaa
   (CT2-mallin polku ja config-avain). Älä vaihda oletuksia: se päätetään kuuntelun jälkeen.
6. **Dokumentit.** Päivitä `api/bench/README.md`, `api/bench/results/*.md`, `docs/research/speech-to-text/STT-MODEL-COMPARISON.md`
   ja `docs/research/text-to-speech/TTS-MODEL-COMPARISON.md` suomi-osiolla (päivämäärä, ajetut revisiot, luvut, lisenssit).

## Valmis kun

- Tulostaulukot (STT ja TTS) ovat `api/bench/results/`-kansiossa json- ja md-muodossa, ajoparametrit ja revisiot mukana.
- Kuuntelusivu on valmis ja `report-b.md` antaa selkeän suosituksen: pysyykö Piper-fi, tuleeko Chatterbox-Finnish
  valinnaiseksi tai oletukseksi, ja vaihdetaanko STT-malli. Perustelut lukuina.
- Kaikki on commitoitu branchille `next/b-fi-models` (ei audiotiedostoja eikä malleja gitiin).
