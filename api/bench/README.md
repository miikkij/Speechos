# Speechos benchmarks — suomen kielen STT/TTS-laatu

Skriptit suomen STT- ja TTS-mallien vertailuun ja teknologialukituksen tueksi. Versio 2 (2026-10-09):
useita STT-taustoja, TTS-vakausmittaus usealla seedillä, UTMOS ja sokkoutettu kuuntelusivu.
Yhteenveto ja suositukset: [`docs/plans/2026-10-public/report-b.md`](../../docs/plans/2026-10-public/report-b.md).

| Tiedosto | Mitä |
|---|---|
| `stt_fi.py` | STT: WER / CER / RTF / latenssi, useita malleja × datasetit, revisiot tuloksiin |
| `nemo/` | Erillinen uv-projekti NeMo-workerille (Parakeet) — ks. alla miksi |
| `tts_fi.py` | TTS: round-trip CER/WER ≥ 3 seedillä, vakaus (sd), RTF, TTFA, UTMOS, lisenssi |
| `tts_engines.py` | TTS-moottorien adapterit (Piper, eSpeak, edge-tts, MMS, Chatterbox ML/Finnish, CSM) |
| `utmos.py` | UTMOS22-strong + UTMOSv2 -luonnollisuusennusteet |
| `make_listen_page.py` | Sokkoutettu HTML-kuuntelusivu + avaimen purku |
| `make_ref.py` | Referenssi-WAV:n nauhoitus selaimessa (16k mono) |
| `results/` | Tulokset json + md (commitoidaan). `listen/` = klipit, gitignored |

## Asennus

Kaikki ajetaan `api/`-kansiosta. Anna `uv run`:lle aina kaikki bench-extrat, muuten uv karsii toisen
ajon paketit pois ympäristöstä.

```bash
cd api
uv sync --extra bench --extra bench-qwen --extra bench-tts --extra tts-fi
# NeMo (Parakeet) asentuu omaan ympäristöönsä ensimmäisellä ajolla: uv run --project bench/nemo …
```

| Extra | Sisältö |
|---|---|
| `bench` | jiwer, datasets, soundfile, librosa, torchcodec (CPU) |
| `bench-qwen` | `qwen-asr` (pinnaa transformers==4.57.6 = ydinlukko; tuo gradion → starlette 1.x vain tässä extrassa) |
| `bench-tts` | edge-tts, UTMOSv2 (git, pinnattu `[tool.uv.sources]`:ssa) |
| `tts-fi` | Chatterbox-Finnishin mukana tulevan Chatterbox-koodin ajonaikaiset riippuvuudet |

**Miksi NeMo ei ole extra:** jokainen `nemo-toolkit`-julkaisu on ristiriidassa ydinlukon kanssa
(2.4–2.6: transformers~=4.53; 2.7: setuptools>=79 vs. ydin `setuptools<75`; 3.0: transformers 5.x).
`bench/nemo/` on oma uv-projekti (nemo-toolkit 3.0.0, torch 2.10 cu128), jota `stt_fi.py` kutsuu
aliprosessina; klipit välitetään npz:nä.

**Miksi ei PyPI:n `chatterbox-tts`:** se pinnaa torch==2.6 ja transformers==5.2. Sekä perus-Chatterbox
Multilingual että Chatterbox-Finnish ajetaan Chatterbox-Finnish-repon mukana tulevalla koodilla
(`src/chatterbox_`, MIT). T3 rakennetaan eager-attentionilla, koska alignment-analysaattori tarvitsee
`output_attentions`-tiedon ja transformers 4.57 ei enää putoa automaattisesti SDPA:sta eageriin.

### Mallit ja polut (ei ladata worktreehen uudelleen)

| Lippu / env | Oletus | Sisältö |
|---|---|---|
| `--models-root` / `SPEECHOS_MODELS_ROOT` | `<repo>/models` | `whisper-ct2/whisper-large-finnish-v3-ct2-f16`, `chatterbox-finnish/` |
| `--whisper-cache` / `SPEECHOS_WHISPER_CACHE` | `<repo>/api/models/whisper` | faster-whisper-mallit |
| `--api-models` / `SPEECHOS_MODEL_DIR` | `<repo>/api/models` | Piper-äänet (`piper/fi_FI-harri-medium.onnx`) |

Muut HF-mallit (Chatterbox-perus, MMS, CSM, Parakeet, Qwen) tulevat normaalista HF-välimuistista
pinnatulla revisiolla. Worktreestä ajettaessa osoita päächeckoutiin, esim.
`--models-root E:\dev\GitHub\Speechos\models --api-models E:\dev\GitHub\Speechos\api\models`.

Mallien lataus:
```bash
# Suomi-Whisper → CTranslate2 float16 (kirjattu myös models/whisper-ct2/*/CONVERSION.txt)
huggingface-cli download Finnish-NLP/whisper-large-finnish-v3 --revision b23deb0b3855c829ffe04cb1c6709757ff16d49c
ct2-transformers-converter --model <HF-snapshot> \
  --output_dir ../models/whisper-ct2/whisper-large-finnish-v3-ct2-f16 \
  --quantization float16 --copy_files tokenizer.json preprocessor_config.json

# Chatterbox-Finnish (koodi + hienosäädetyt T3-painot + näytereferenssi)
python -c "from huggingface_hub import snapshot_download as s; s('Finnish-NLP/Chatterbox-Finnish', \
  revision='d15775e1788055e67f49dfc6da402021e51bd0f0', local_dir='../models/chatterbox-finnish', \
  allow_patterns=['src/chatterbox_/*.py','src/chatterbox_/**/*.py','models/*.safetensors','samples/reference_finnish.wav'])"
```

---

## STT — `stt_fi.py`

```bash
# Iterointi: kaikki mallit, kolme datasettiä, n=100
uv run --extra bench --extra bench-qwen --extra bench-tts --extra tts-fi python bench/stt_fi.py \
  --models large-v3 large-v3-turbo fi-whisper-v3 parakeet-tdt-0.6b-v3 qwen3-asr-1.7b \
  --datasets fleurs voxpopuli voxpopuli-short --n 100 --tag n100

# Lukitusajo: 300 klippiä
uv run … python bench/stt_fi.py --models fi-whisper-v3 large-v3 large-v3-turbo \
  --datasets fleurs voxpopuli --n 300 --tag lock300

# Pelkkä md uudelleen JSONista (päivittää train-overlap-merkinnät)
uv run python bench/stt_fi.py --render bench/results/stt_fi_results_n100.json
```

| Malli-avain | Tausta | Lisenssi |
|---|---|---|
| `large-v3`, `large-v3-turbo`, `medium` | faster-whisper (CT2) | MIT |
| `fi-whisper-v3` | Finnish-NLP/whisper-large-finnish-v3 → CT2 float16 | Apache-2.0 |
| `parakeet-tdt-0.6b-v3` | NeMo 3.0, aliprosessi `bench/nemo/` | CC-BY-4.0 |
| `qwen3-asr-1.7b` | qwen-asr, transformers-tausta, bf16 | Apache-2.0 |

### Datasetit (raportoidaan aina erikseen)
- **`fleurs`** — google/fleurs fi_fi test (luettu puhe). HF:n parquet-konversio pinnattuna; sama
  valinta kuin 2026-06-ajossa (large-v3 toistaa WER 7,9 %). **Ainoa kaikille malleille puhdas testi.**
- **`voxpopuli`** — facebook/voxpopuli fi test (Euroopan parlamentti, CC0). Sisältää kohdistusvirheitä
  (klipin puhe ≠ transkriptio), jotka nostavat kaikkien mallien CER:ää.
- **`voxpopuli-short`** — VoxPopuli fi test+validation, klipit 1–5 s (keskustelun lyhyiden vuorojen proxy).
- **`commonvoice`** — vain paikallisesta korpuksesta (`--cv-path`, odottaa `test.tsv` + `clips/`).
  Common Voice jaetaan lokakuusta 2025 alkaen vain Mozilla Data Collectivesta; HF-repo on tyhjä tynkä.

**⚠ Koulutusdatan päällekkäisyys** (`TRAIN_OVERLAP` skriptissä, näkyy tulostaulukossa):
Finnish-NLP:n Whisper on koulutettu *kaikilla* VoxPopuli-fi-spliteillä (9074 riviä = train 7878 +
validation 718 + test 478) → sen VoxPopuli-rivit ovat vuotaneita eivätkä kelpaa vertailuun. FLEURS-
ja CV-testit ovat sille in-domain (train-splitit mukana), mutta testiklipit eivät vuoda.
Parakeetin Granary/MOSEL sisältää VoxPopuli-audiota pseudolabeleilla (päällekkäisyys mahdollinen).

### Tulokset
`results/stt_fi_results_<tag>.{json,md}` (+ `stt_fi_details_<tag>.json`: jokaisen klipin ref/hyp/aika).

| Mittari | Tavoite | Minimi |
|---|---|---|
| WER (fi) | ≤ 12 % | ≤ 18 % |
| CER (fi) | ≤ 5 % | ≤ 8 % |
| RTF (GPU) | ≤ 0,3 | ≤ 1,0 |

Tulokset 2026-10-09: ks. `results/stt_fi_results_n100.md`, `results/stt_fi_results_lock300.md` ja
yhteenveto alla.

---

## TTS — `tts_fi.py`

Round-trip: lause → TTS → STT-tuomari (faster-whisper large-v3, pinnattu) → CER/WER. Jokainen moottori
ajetaan samoilla 38 lauseella (30 vanhaa + 8 lyhyttä keskusteluvuoroa) **vähintään kolmella seedillä**.

```bash
uv run --extra bench --extra bench-qwen --extra bench-tts --extra tts-fi python bench/tts_fi.py \
  --engines piper-fi espeak-fi edge-noora edge-harri mms-tts-fin chatterbox-finnish-ref \
            chatterbox-finnish-preset chatterbox-finnish-nat-ref chatterbox-ml-ref chatterbox-ml-preset \
  --seeds 0 1 2 --tag v2 --save-audio bench/listen
uv run python bench/make_listen_page.py            # → bench/listen/index.html + key.json
```

| Moottori | Mikä | Lisenssi |
|---|---|---|
| `piper-fi` | Piper `fi_FI-harri-medium` (tuotannon oletus) | moottori GPL-3.0 (piper-tts 1.4); ääni CC0-data, hienosäädetty lessacista (Blizzard-2013-lisenssi) |
| `espeak-fi` | eSpeak NG | GPL-3.0 |
| `edge-noora`, `edge-harri` | Microsoft Edge -neuraaliäänet (pilvi, **vain vertailukohta**) | palvelun ehdot |
| `mms-tts-fin` | facebook/mms-tts-fin (VITS) | CC-BY-NC-4.0 — vain arviointiin |
| `chatterbox-finnish-ref` | Finnish-NLP/Chatterbox-Finnish cp986, mallikortin asetukset, suomalainen referenssiääni | MIT |
| `chatterbox-finnish-preset` | sama, oletusääni `conds.pt` (englantilainen) | MIT |
| `chatterbox-finnish-nat-ref` | "best_naturalness" cp1060 | MIT |
| `chatterbox-ml-ref`, `-preset` | perus-Chatterbox Multilingual (t3_mtl23ls_v2), `language_id="fi"` | MIT |
| `csm-fin` | Sesame CSM-1B, suomen parlamenttihienosäätö (prosessori hienosäätörepon omista tiedostoista) | painot Apache-2.0, data CC-BY-NC-ND |
| `zipvoice-suomi` | kirjataan ohitetuksi: k2:lla ei ole Windows-buildia | CC-BY-ND-4.0 |

Referenssiääni `*-ref`-moottoreille: oletuksena Chatterbox-Finnish-repon `samples/reference_finnish.wav`
(`--ref-wav` vaihtaa). Chatterbox-asetukset: Finnish = mallikortti (repetition_penalty 1.2,
temperature 0.8, exaggeration 0.6); ML = kirjaston oletukset (rp 2.0, cfg 0.5, exaggeration 0.5).

### Mittarit
- **CER/WER mean ± sd** — korpustason CER per ajo, keskiarvo ja keskihajonta seedien yli (vakaus).
- **short CER** — 8 lyhyen keskusteluvuoron CER ("Joo, ymmärrän.", "Miltä se tuntui?", …).
- **catastr./run** — lauseet, joiden CER > 30 % (toisto, katkeaminen, höpinä).
- **RTF** = synteesiaika / audion kesto. **TTFA** = aika ensimmäiseen audiopalaan moottorin
  API:n kautta (ei-striimaavilla = koko synteesiaika; Piper per lause; edge-tts verkkostriimi).
- **UTMOS22 / UTMOSv2** — neuraaliset luonnollisuusennusteet (asteikko 1–5). Koulutettu pääosin
  englannin- ja japaninkielisillä kuuntelukokeilla: **vain suhteellinen järjestys**, ei korvaa
  kuuntelua. (SQUIM-MOS poistettu: se antoi kesäkuussa eSpeakille korkeimman arvon.)
- **VRAM** — torchin huippuvaraus moottorin aikana (sisältää UTMOS-mallit ~1,7 GB, ei CT2-tuomaria).

### Kuuntelusivu
`--save-audio bench/listen` tallentaa 10 kiinteää lausetta per moottori (seed 0). `make_listen_page.py`
tekee `listen/index.html`:n, jossa moottorit ovat joka lauseessa satunnaisjärjestyksessä (A, B, …),
arviointi 1–5 tallentuu selaimeen ja "Lataa ratings.json" vie arviot tiedostoon. Avain on
`listen/key.json` — älä avaa ennen arviointia. Purku:
```bash
uv run python bench/make_listen_page.py --unblind bench/listen/ratings.json   # → listen/unblinded.md
```

### Tulokset
`results/tts_fi_results_v2.{json,md}` (kaikki moottorit, 3 seediä) ja `results/tts_fi_results_v2-csm.*`
(CSM-FIN, 1 seed). JSON sisältää jokaisen lauseen transkription, CER:n, keston ja UTMOS-arvot.

---

## Tulokset 2026-10-09 (tiivistelmä)

Suositukset ja perustelut: `docs/plans/2026-10-public/report-b.md`.

**STT, FLEURS-fi n=300** (puhdas vertailu): suomi-Whisper v3 WER **6,2 %** / CER 1,43 % · large-v3 7,5 % / 1,53 % ·
large-v3-turbo 8,1 % / 1,65 % (RTF 0,018, 3× nopein). n=100: Parakeet 12,6 %, Qwen3-ASR 27,3 %.
Lyhyet 1–5 s klipit (VoxPopuli): large-v3 13,5 %, turbo 16,3 %, Parakeet 14,8 % (suomi-Whisper vuotanut).

**TTS, 38 lausetta × 3 seediä** (CER ± sd · RTF · UTMOS22):

| Moottori | CER ± sd | RTF | TTFA | UTMOS22 |
|---|---:|---:|---:|---:|
| edge Noora / Harri (pilvi) | 1,95 / 2,17 % | 0,20 / 0,28 | 0,5–0,6 s | 3,88 / 4,00 |
| Chatterbox-Finnish cp986 + fi-ref | 3,29 % ± 0,26 | 1,28 | 4,6 s | 3,72 |
| MMS-TTS fin (NC) | 3,41 % ± 0,78 | 0,02 | 0,07 s | 3,31 |
| Chatterbox ML + fi-ref | 3,95 % ± 1,21 | 1,00 | 4,4 s | 3,46 |
| Piper fi_FI-harri-medium | 4,65 % ± 1,33 | 0,036 | 0,11 s | 3,38 |
| eSpeak-NG | 9,94 % ± 0,00 | 0,02 | 0,04 s | 1,77 |
| Chatterbox ML, oletusääni | 16,89 % ± 8,74 | 1,09 | 5,4 s | 3,06 |

Piper hajoaa lyhyissä vuoroissa ("Joo, ymmärrän." → ~6 s höpinää joka ajossa). Chatterbox-Finnish on paras
paikallinen ja vakaa, mutta reaaliaikaa hitaampi (T3-tokenigenerointi 84 % ajasta, 30,9 tok/s).

## Referenssi-WAV:n teko (klonaus)
`make_ref.py` (selain, ffmpeg PATH:ssa) tai `ffmpeg -i klippi.mp3 -ac 1 -ar 16000 bench/ref_fi.wav`.

## Historia
2026-06-19 (v1): large-v3 FLEURS-fi n=100 WER 7,9 %; TTS n=30 yksi ajo: Piper 3,5 %, Chatterbox ML
4,9–16,5 %, eSpeak 10,3 %. Tiedostot `results/stt_fi_results.*` ja `results/tts_fi_results.*`
(v1-formaatti, ei ylikirjoitettu).
