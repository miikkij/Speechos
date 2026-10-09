# Raportti B — FI-benchmark 2.0 (2026-10-09)

Branch `next/b-fi-models` (worktree `../Speechos-b`), ei pushattu. Kaikki ajot RTX 4090 / Windows 11,
torch 2.10 cu128, transformers 4.57.6. Mallien tarkat revisiot ovat tulos-JSONeissa ja alla.

## Suositus lyhyesti

| Päätös | Suositus | Perustelu (luvut) |
|---|---|---|
| **TTS-oletus** | **Piper-fi pysyy oletuksena** — lisää eteen `normalize_fi` (handoff §2b) ja lyhyiden vuorojen suoja | Ainoa paikallinen reaaliaikainen vaihtoehto: RTF 0,036, TTFA 0,11 s. CER 4,65 % ± 1,33 (ei 3,5 % kuten kesäkuun yksittäisajo) ja hajoaa lyhyissä vuoroissa ("Joo, ymmärrän." → ~6 s höpinää joka ajossa). |
| **Chatterbox-Finnish** | **Valinnaiseksi moottoriksi (ei oletukseksi ennen kuuntelua).** Nopeutuksen jälkeen kaikki automaattikriteerit täyttyvät; moduuli `api/src/engines_fi/chatterbox_fi.py` + rekisteröintiohje `handoff-b.md` §2. | CER 3,29 % ± 0,26 (paras paikallinen, vakain autoregressiivinen), UTMOS22 3,72 (lähimpänä pilviääniä 3,9–4,0). CUDA-graph-dekooderilla RTF 1,28 → **0,43** ja TTFA 4,6 → **1,5 s** bitilleen samalla äänellä. Kriteerit: CER ✅, hajonta ✅, RTF ✅. |
| **STT** | **Suomi-Whisper (Finnish-NLP/whisper-large-finnish-v3) suomenkielisille pyynnöille**, large-v3 pysyy yleisoletuksena. Ei vaihdeta ennen käyttäjän päätöstä. | FLEURS-fi n=300: WER 6,2 % vs 7,5 % (ero −1,3 pp, 95 % CI −2,1…−0,5), CER 1,43 vs 1,53 % (ero ei merkitsevä), sama nopeus (RTF 0,044), Apache-2.0. Konfiguraatio-ohje: `handoff-b.md` §1. |

Muut: **large-v3-turbo** on 3× nopeampi (RTF 0,018, latenssi 0,21 s) +0,6 pp WER FLEURSilla, mutta selvästi heikompi
lyhyillä klipeillä (16,3 vs 13,5 %). **Parakeet** (WER 12,6 %) ja **Qwen3-ASR** (27,3 %) eivät kilpaile suomessa.

## Jatko (2026-10-09 ilta): nopeutus, moottorimoduuli, tekstinormalisointi

Käyttäjä pyysi toteuttamaan suositukset. Tehty omistamiini tiedostoihin (`api/src/engines_fi/**`, `api/bench/**`):

**1. Chatterbox-Finnishin nopeutus** — profiili: T3:n tokenigenerointi 84 % ajasta, 30 tok/s, ja pullonkaula oli
Pythonin/kernel-käynnistysten yleiskustannus (~31 000 pientä `linear`-kutsua lauseessa), ei laskenta.

| Kokeilu (10 lausetta, sama konfiguraatio) | T3 tok/s | RTF | Tulos |
|---|---:|---:|---|
| Alkuperäinen silmukka (eager kaikissa kerroksissa) | 30 | 0,99 | lähtötaso |
| SDPA muille kuin analysaattorin kerroksille 9/12/13 | 34 | 0,92 | pieni hyöty → hylätty |
| bf16-autocast | 29,5 | 1,05 | hitaampi → hylätty |
| **CUDA graph + staattinen KV-välimuisti + staattinen 4D-maski** (`chatterbox_graph.py`) | **78** | **0,49** | identtiset tokenit |
| CUDA graph + bf16-runko (Llama-painot bf16) | 129 | 0,35 | ei bitti-identtinen |

Täysi ajo (38 lausetta × 3 seediä, `results/tts_fi_results_v2-graph.md`):

| Variantti | CER ± sd | WER | lyhyet | katastr./ajo | RTF | TTFA | UTMOS22 |
|---|---:|---:|---:|---:|---:|---:|---:|
| alkuperäinen | 3,29 % ± 0,26 | 8,2 % | 4,3 % | 1,0 | 1,28 | 4,64 s | 3,72 |
| **CUDA graph fp32** | **3,29 % ± 0,26** | 8,2 % | 4,3 % | 1,0 | **0,43** | **1,54 s** | 3,72 |
| CUDA graph bf16 | 2,73 % ± 0,40 | 7,9 % | 2,8 % | 0,3 | 0,32 | 1,16 s | 3,71 |

fp32-graafin kaikki kolme seediä tuottavat samat luvut kuin alkuperäinen, ja 10 kuunteluklippiä ovat näyte
näytteeltä identtiset (max-ero 0,0). bf16 on nopeampi eikä huonompi, mutta ei identtinen → jätetty valinnaiseksi
(`bf16=True`). Oletus moduulissa: graph fp32.

**2. Moottorimoduuli** `api/src/engines_fi/chatterbox_fi.py`: pinnattu lataus, mallikortin latauspolku,
lausetason `synthesize()`, 24 kHz, `set_voice()` kloonaukseen, normalisointi sisällä, ~4,4 GB VRAM. Rekisteröinti
on A:n tiedostoissa (`models.py`, `synthesize.py`, `system.py`) → tarkka ohje `handoff-b.md` §2 (A:n
`handoff-a.md`:n mallin mukaan). `ruff check src/engines_fi` puhdas → exclude voidaan poistaa.

**3. Suomen tekstinormalisointi** `api/src/engines_fi/text_norm_fi.py` (num2words, sijamuotoheuristiikat):
numerot, päivämäärät (essiivi), rahat, prosentit, kellonajat, puhelinnumerot (numero kerrallaan), lyhenteet,
akronyymit, sähköpostit. 18 testiä (`bench/test_text_norm_fi.py`). Kohdennettu mittaus 10 numero-/lyhennelauseella
(`bench/sentences_numbers_fi.txt`, 3 seediä): avainkohtien osuus (`results/tts_fi_number_items.md`,
`bench/number_items.py`; round-trip-CER ei sovi tähän, koska oikea luenta palaa tuomarilta numeroiden ja sanojen
sekoituksena):

| Moottori | ilman normalisointia | `normalize_fi` | lauseet täysin oikein |
|---|---:|---:|---:|
| Chatterbox-Finnish (graph) | 40 % | **99 %** | 9/30 → **29/30** |
| Piper fi_FI-harri-medium | 80 % | **91 %** | 16/30 → 23/30 |

Chatterbox-Finnish sotkee numerot ilman normalisointia ("7.45 raiteelta 9" → "7 millä menee miisaraitelta 1");
normalisoituna lähes kaikki menevät oikein. Piper lukee numerot osin itse (eSpeakin foneemitus), mutta hyötyy myös
(osoite, puhelinnumero, lyhenteet). → Normalisointi kuuluu molempien eteen.

**4. `main` oli edennyt** (käyttäjä mergesi B:n, A:n työ, riippuvuuspäivitys torch 2.11 / piper-tts 1.8 /
ctranslate2 4.8 / librosa 1.0). Branch kelattiin `main`iin (fast-forward, ei uudelleenkirjoitusta), `uv lock`
uudelleen, A:n 114 testiä + 18 normalisointitestiä menevät läpi, moottorimoduuli toimii torch 2.11:llä.
**Huom:** illan aikana koneella pyöri muita raskaita prosesseja (CPU 60–96 %, GPU 15,7 GB varattuna, mm. toinen
Gradio-sovellus ja päächeckoutin API), joten torch 2.11 -pinon nopeuslukuja ei mitattu luotettavasti; yllä olevat
RTF-luvut on mitattu hiljaisella koneella torch 2.10:llä. CER-tulokset eivät riipu kuormasta.

## Mitä tehtiin

1. **TTS-vakausmittaus** (`api/bench/tts_fi.py` uusiksi + `tts_engines.py`): jokainen moottori 3 seediä samoilla
   38 lauseella (30 vanhaa + 8 uutta lyhyttä keskusteluvuoroa), CER/WER keskiarvo ± sd, lyhyiden vuorojen CER,
   katastrofaaliset lauseet/ajo (CER > 30 %), RTF, TTFA, torch-VRAM, lisenssi- ja revisiosarake.
2. **UTMOS** (`utmos.py`): UTMOS22-strong (SpeechMOS torch.hub v1.2.0) + UTMOSv2 (sarulab-speech/UTMOSv2
   @cc2700d). Kirjattu kaikkialle: englanti/japani-koulutettu → vain suhteellinen järjestys. SQUIM poistettu.
3. **TTS-ajot**: 10 moottoria × 3 seediä (+ CSM-FIN 1 ajo, ZipVoice ohitettu). 10 klippiä/moottori
   `api/bench/listen/` (gitignored) + sokkoutettu kuuntelusivu (`make_listen_page.py`, avain `key.json`).
4. **STT-ajot** (`stt_fi.py` uusiksi): faster-whisper (large-v3, turbo, suomi-Whisper CT2-muunnettuna), NeMo
   Parakeet (oma uv-projekti `api/bench/nemo/`), Qwen3-ASR (extra `bench-qwen`). FLEURS-fi + VoxPopuli-fi +
   VoxPopuli 1–5 s -klipit n=100, sitten 300 klipin lukitusajo (suomi-Whisper, large-v3, turbo).
5. **Moottori (ehdollinen)**: Chatterbox-Finnish ei täyttänyt RTF-kriteeriä → ei moduulia, ei rekisteröintiä.
   STT:lle konfiguraatio-ohje (ei koodimuutosta tarvita) `handoff-b.md`:ssä.
6. **Dokumentit**: `api/bench/README.md`, `api/bench/results/*`, suomi-osiot `STT-MODEL-COMPARISON.md` ja
   `TTS-MODEL-COMPARISON.md`, `handoff-b.md`.

## TTS-tulokset (38 lausetta × 3 seediä, tuomari faster-whisper large-v3 @edaa852)

| Moottori | CER ± sd | WER | lyhyet CER | katastr./ajo | RTF | TTFA | UTMOS22 | UTMOSv2 | Lisenssi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| edge Noora (pilvi, vertailu) | 1,95 % ± 0,04 | 3,4 % | 0,0 % | 0 | 0,20 | 0,50 s | 3,88 | 3,02 | MS-palvelu |
| edge Harri (pilvi, vertailu) | 2,17 % ± 0,04 | 4,1 % | 0,0 % | 0 | 0,28 | 0,63 s | 4,00 | 3,28 | MS-palvelu |
| **Chatterbox-Finnish cp986 + fi-ref** | **3,29 % ± 0,26** | 8,2 % | 4,3 % | 1,0 | 1,28 | 4,64 s | 3,72 | 3,01 | MIT |
| MMS-TTS fin | 3,41 % ± 0,78 | 13,9 % | 2,8 % | 0 | 0,02 | 0,07 s | 3,31 | 3,17 | CC-BY-NC (vain arviointi) |
| Chatterbox ML + fi-ref | 3,95 % ± 1,21 | 6,7 % | 0,9 % | 1,0 | 1,00 | 4,43 s | 3,46 | 2,81 | MIT |
| **Piper fi_FI-harri-medium** | 4,65 % ± 1,33 | 11,5 % | 15,0 % | 1,7 | 0,036 | 0,11 s | 3,38 | 2,84 | GPL-3.0 moottori; ääni CC0-data (lessac-pohja) |
| Chatterbox-Finnish cp1060 + fi-ref | 5,14 % ± 2,49 | 8,9 % | 24,5 % | 1,3 | 0,83 | 3,02 s | 3,65 | 3,02 | MIT |
| Chatterbox-Finnish cp986, oletusääni | 5,34 % ± 1,33 | 13,6 % | 9,5 % | 2,0 | 1,01 | 2,31 s | 3,71 | 3,09 | MIT |
| eSpeak-NG 1.52 | 9,94 % ± 0,00 | 31,4 % | 3,7 % | 3,0 | 0,02 | 0,04 s | 1,77 | 1,73 | GPL-3.0 |
| Chatterbox ML, oletusääni | 16,89 % ± 8,74 | 23,5 % | 59,9 % | 4,3 | 1,09 | 5,35 s | 3,06 | 2,75 | MIT |
| CSM-FIN (1 ajo, 10 kuuntelulausetta) | 55,7 % | 65,4 % | 281,5 % | 5 / 10 | 4,45 | 44,7 s | 2,52 | 2,45 | painot Apache-2.0, data CC-BY-NC-ND |
| ZipVoice-Suomi-12k | ei ajettu: k2:lla ei Windows-buildia | | | | | | | | CC-BY-ND-4.0 |

Havainnot:
- **Chatterbox-Finnish tarvitsee suomalaisen referenssiäänen**: oletusäänellä (englantilainen `conds.pt`) CER
  5,34 % ± 1,33. Sen virheet ovat lähinnä numeroita ("12 A" → "kupsitoista a", "24." ), ei höpinää.
  Referenssinä käytettiin mallin repon `samples/reference_finnish.wav`.
- **cp1060 "best_naturalness"** on epävakaampi kuin cp986 (sd 2,49 pp, lyhyet vuorot 24,5 %) → cp986.
- **Piper**: katastrofit ovat lyhyitä vuoroja ("Joo, ymmärrän." → "joon pinnata saralään", "Kerro lisää.") ja
  yksi katkeaminen ("Syksyn lehdet kah…"). ONNX-kohinaa ei voi seedata → ajojen välinen vaihtelu on todellista.
- **Perus-Chatterbox ML oletusäänellä** on edelleen käyttökelvoton suomeksi (kesäkuun epävakaus toistui).
- **Numeroiden luku** on kaikkien paikallisten moottoreiden yhteinen heikkous → suomen tekstinormalisointi
  (numerot sanoiksi) TTS:n eteen on halvin parannus kaikille.
- UTMOS-järjestys on looginen (eSpeak 1,8 → Piper/MMS 3,3–3,4 → Chatterbox-Finnish 3,7 → pilvi 3,9–4,0),
  toisin kuin kesäkuun SQUIM.

Chatterbox-Finnishin nopeusprofiili (10 lausetta, sama konfiguraatio, ilman tuomaria/UTMOSia): RTF 0,99, josta
**T3 (autoregressiiviset puhetokenit) 84 %**, S3Gen 16 %, Perth-vesileima 1 %. T3 tuottaa 30,9 tokenia/s
(25 tokenia = 1 s audiota) → RTF ≤ 0,5 vaatii T3:n ~2,5× nopeutuksen. Huippu-VRAM 3,6 GB.

## STT-tulokset

**FLEURS-fi test n=300** (ainoa kaikille puhdas testi):

| Malli | WER | CER | RTF | latenssi med | Lisenssi |
|---|---:|---:|---:|---:|---|
| **suomi-Whisper v3** (`Finnish-NLP/whisper-large-finnish-v3@b23deb0` → CT2 fp16) | **6,2 %** | 1,43 % | 0,044 | 0,54 s | Apache-2.0 |
| large-v3 (`Systran/faster-whisper-large-v3@edaa852`) | 7,5 % | 1,53 % | 0,044 | 0,53 s | MIT |
| large-v3-turbo (`mobiuslabsgmbh/faster-whisper-large-v3-turbo`) | 8,1 % | 1,65 % | 0,018 | 0,21 s | MIT |

Pariutettu bootstrap (5000×): suomi-Whisper − large-v3 WER −1,27 pp [−2,08; −0,53], CER −0,09 pp [−0,32; +0,15];
suomi-Whisper − turbo WER −1,89 pp [−2,75; −1,05].

**Kaikki kandidaatit n=100** (`results/stt_fi_results_n100.md`):

| Malli | FLEURS WER / CER | VoxPopuli WER | lyhyet 1–5 s WER | RTF | latenssi med |
|---|---:|---:|---:|---:|---:|
| suomi-Whisper v3 | 6,4 % / 1,47 % | ⚠ 6,1 % (vuoto) | ⚠ 3,5 % (vuoto) | 0,048 | 0,54 s |
| large-v3 | 7,9 % / 1,46 % | 11,5 % | 13,5 % | 0,052 | 0,58 s |
| large-v3-turbo | 8,0 % / 1,56 % | 11,6 % | 16,3 % | 0,018 | 0,21 s |
| Parakeet TDT 0.6B v3 (`@541d1f9`) | 12,6 % / 2,29 % | 14,5 % | 14,8 % | 0,008 | 0,09 s |
| Qwen3-ASR 1.7B (`@7278e1e`) | 27,3 % / 6,12 % | 29,1 % | 31,7 % | 0,22 | 2,45 s |

300 klipin VoxPopuli: large-v3 10,9 %, turbo 11,7 % (suomi-Whisper vuotanut).

## Mitä verifioitiin (komennot + tulokset)

| Komento (api/, kaikki bench-extrat) | Tulos |
|---|---|
| `uv lock` + `uv sync --extra bench --extra bench-qwen --extra bench-tts --extra tts-fi` | OK; ydinpaketit ennallaan (transformers 4.57.6, torch 2.10.0+cu128) |
| `uv lock && uv sync` (`bench/nemo/`) | OK, nemo-toolkit 3.0.0 + torch 2.10 cu128; Parakeet transkriboi suomea |
| `stt_fi.py --models large-v3 … --n 5 --tag smoke` | kaikki 4 taustaa toimivat (tulokset poistettu) |
| `stt_fi.py --models large-v3 large-v3-turbo fi-whisper-v3 parakeet-tdt-0.6b-v3 qwen3-asr-1.7b --datasets fleurs voxpopuli voxpopuli-short --n 100 --tag n100` | `results/stt_fi_results_n100.*`; large-v3 FLEURS toistaa kesäkuun 7,9 % / 1,46 % tarkasti → sama klippivalinta |
| `stt_fi.py --models fi-whisper-v3 large-v3 large-v3-turbo --datasets fleurs voxpopuli --n 300 --tag lock300` | `results/stt_fi_results_lock300.*` |
| `tts_fi.py --engines … (10) --seeds 0 1 2 --tag v2 --save-audio bench/listen` | `results/tts_fi_results_v2.*`, 0 kaatumista; klipit 10/moottori |
| `tts_fi.py --engines csm-fin --seeds 0 --sentences-file <10 kuuntelulausetta> --tag v2-csm` | `results/tts_fi_results_v2-csm.*` |
| `make_listen_page.py` | `bench/listen/index.html` + `key.json` |
| Vuototarkistus: `Finnish-NLP/voxpopuli_fi_audio_processed` 9074 riviä vs. `facebook/voxpopuli` fi 7878+718+478 | täsmää → VoxPopuli-testi vuotanut suomi-Whisperille |

## Mitä jäi tekemättä ja miksi

- **Common Voice -fi**: Mozilla jakaa Common Voicen lokakuusta 2025 vain Mozilla Data Collectivessa; HF-repo
  `mozilla-foundation/common_voice_17_0` on tyhjä README. Kolmannen osapuolen peiliä (`fsicoli/common_voice_17_0`)
  ei käytetty, koska se kiertäisi virallisen jakelukanavan ehdot. Skripti tukee `--cv-path`-paikallista korpusta.
  Korvaajana VoxPopuli-fi (CC0, puhdas large-v3:lle, Parakeetille mahdollinen päällekkäisyys, suomi-Whisperille vuoto).
- **Puhdas lyhyiden klippien testi suomi-Whisperille** puuttuu (VoxPopuli vuotanut; FLEURSissa ei juuri < 5 s klippejä).
- **ZipVoice-Suomi**: vaatii `k2`:n, jolle ei ole Windows-buildia → ohitettu hallitusti (ajettavissa WSL:ssä).
- **CSM-FIN**: generointi CPU-sidonnainen (GPU ~34 %), ~3 min/lause → täysi 3×38 olisi vienyt ~5 h. Ajettiin
  1 seed 10 kuuntelulauseella; jo savutestissä CER > 100 % (parafrasoi/keksii tekstiä) ja RTF ≈ 4 → hylätty.
- **NeMo extraksi**: ei mahdollista ilman ydinriippuvuuksien muutosta (ks. handoff §3.3) → oma uv-projekti.
- **Kesäkuun `chatterbox-fi-clone` käyttäjän omalla äänellä (`bench/ref_fi.wav`)** ei ajettu uudelleen; `--ref-wav`
  tukee sitä. Vertailu tehtiin mallin repon julkisella referenssillä, jotta tulos on toistettavissa.
- **Kuuntelu** on käyttäjän tehtävä (sivu valmis).

## Avoimet päätökset (käyttäjälle)

1. **Kuuntele** `E:\dev\GitHub\Speechos-b\api\bench\listen\index.html` (11 moottoria × 10 lausetta, sokkoutettu),
   lataa `ratings.json` ja aja `uv run python bench/make_listen_page.py --unblind <ratings.json>`. Erityisesti:
   kuulostaako Chatterbox-Finnish niin paljon paremmalta kuin Piper, että nopeutustyö kannattaa?
2. **Chatterbox-Finnish valinnaiseksi moottoriksi?** Nopeutus tehty (RTF 0,43, TTFA 1,5 s), moduuli valmis;
   rekisteröinti A:n tiedostoihin `handoff-b.md` §2:n mukaan — kuka tekee ja milloin? Lisänopeutus mahdollinen:
   bf16-runko (RTF 0,32) ja S3Genin striimaus (TTFA < 1 s).
3. **STT**: otetaanko suomi-Whisper käyttöön suomenkielisille pyynnöille (`stt_fi`-slotti, handoff §1)?
   Ennen oletukseksi tekemistä kannattaisi ajaa puhdas lyhyiden klippien testi: Common Voice -fi MDC:stä
   (`--cv-path`) tai omat ~50 nauhoitettua keskusteluvuoroa.
4. **Piperin lyhyiden vuorojen suoja** (A:n koodia): kesto ≫ odotettu → uusi yritys toisella `length_scale`-arvolla.
5. **Suomen tekstinormalisointi Piperin eteen** (A:n `synthesize_wav`): `normalize_fi(text)` ennen Piper-synteesiä
   — moduuli valmis, ks. mittaus yllä. Chatterbox-Finnish-moduuli normalisoi jo itse.
6. **Lisenssit**: piper-tts 1.4 on GPL-3.0; harri-ääni on hienosäädetty lessacista (Blizzard 2013 -datalisenssi).
   MMS ja CSM-FIN vain arviointiin (NC/ND).
