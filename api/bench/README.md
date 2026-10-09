# Speechos benchmarks — suomen kielen STT/TTS-laatu

Skriptit suomen STT- ja TTS-mallien vertailuun ja teknologialukituksen tueksi.

## STT — `stt_fi.py` (WER / CER / RTF, faster-whisper, suomi)

Mittaa suomen STT-laadun vakiodudatasetillä. **Aja WSL2:ssa GPU:lla.**

### Asennus
```bash
cd api
uv sync --extra bench        # jiwer + datasets + soundfile + librosa
# tai: uv pip install jiwer datasets
```

### Datasetit
- **FLEURS-fi** (`google/fleurs`, `fi_fi`): avoin, latautuu automaattisesti. **Toimii heti.**
- **Common Voice -fi**: gated HuggingFacessa → vaihtoehdot:
  - `huggingface-cli login` + hyväksy ehdot datasetin sivulla, sitten `--datasets commonvoice`, **tai**
  - lataa korpus osoitteesta https://commonvoice.mozilla.org/fi/datasets, pura, ja
    `--datasets commonvoice --cv-path /polku/cv-corpus-XX-fi` (odottaa `test.tsv` + `clips/`).

### Ajo
```bash
# Nopea iterointi: vain FLEURS, 100 klippiä, large-v3
uv run python bench/stt_fi.py --models large-v3 --datasets fleurs --n 100

# Mallivertailu, molemmat datasetit (CV paikallisesta hakemistosta)
uv run python bench/stt_fi.py \
  --models large-v3 medium turbo Systran/faster-distil-whisper-large-v3 \
  --datasets fleurs commonvoice --cv-path /data/cv-corpus-fi --n 100

# Lopullinen lukitusajo: 300 klippiä/datasetti
uv run python bench/stt_fi.py --models large-v3 --datasets fleurs commonvoice --cv-path /data/cv-fi --n 300
```

### Parametrit
| Lippu | Oletus | Selitys |
|---|---|---|
| `--models` | `large-v3` | faster-whisper-malli(t) |
| `--datasets` | `fleurs` | `fleurs` ja/tai `commonvoice` |
| `--n` | `100` | klippiä/datasetti (100 iterointi → 300 lukitus) |
| `--seed` | `1234` | satunnaissiemen (toistettavuus) |
| `--device` | `cuda` | `cuda`/`cpu` |
| `--compute-type` | `float16` | `float16`/`int8_float16`/`int8` |
| `--cv-path` | – | paikallinen Common Voice -hakemisto |

### Tulokset
- `bench/results/stt_fi_results.json` — koneluettava, per (malli × datasetti)
- `bench/results/stt_fi_results.md` — markdown-taulukko (WER/CER/RTF/latenssi)

Mittarit raportoidaan **per datasetti erikseen** (ei niputettuna). Teksti normalisoidaan ennen WER/CER:iä
(pienaakkoset, välimerkit pois, ä/ö/å säilytetään). RTF = prosessointiaika / audiokesto. Kylmälatenssi
(ensimmäinen klippi mallin latauksen jälkeen) raportoidaan erikseen.

### STT-kynnykset
| Mittari | Tavoite | Minimi |
|---|---|---|
| WER (fi, large-v3) | ≤ 12 % | ≤ 18 % |
| CER (fi) | ≤ 5 % | ≤ 8 % |
| RTF (GPU) | ≤ 0,3 | ≤ 1,0 |

**Tulos 2026-06-19** (large-v3, FLEURS-fi, n=100, GPU/float16): WER **7,9 %**, CER **1,5 %**, RTF **0,058** → ylittää kaikki kynnykset. Common Voice -fi + 300 klipin lukitusajo vielä ajamatta.

---

## TTS — `tts_fi.py` (round-trip ymmärrettävyys: TTS → STT → CER/WER)

Karsintamittari suomi-TTS-moottoreille: syntetisoi lause → transkriboi takaisin large-v3:lla → mittaa
montako sanaa selvisi (CER/WER) + TTS:n RTF. Moottorit ladataan **tuotannon ModelManagerin kautta** (sama
koodipolku kuin API), joten Piper lataa fi-äänen itse ja Chatterbox käyttää monikielistä luokkaa.

### Lisäasennukset
- **Chatterbox:** `uv pip install chatterbox-tts` (GPU)
- **eSpeak:** `sudo apt-get install espeak-ng`
- Piper-fi toimii ilman lisäasennuksia (CPU, lataa äänen HuggingFacesta).

### Ajo
```bash
# CPU-baselinet ilman Chatterboxia:
uv run python bench/tts_fi.py --engines piper-fi espeak-fi

# + Chatterbox preset-ääni (GPU):
uv run python bench/tts_fi.py --engines piper-fi espeak-fi chatterbox-fi-preset

# Chatterbox klonatulla äänellä (vaatii referenssi-WAV 16k mono):
uv run python bench/tts_fi.py --engines chatterbox-fi-clone --ref-wav /data/ref_fi.wav

# Omat lauseet (yksi per rivi):
uv run python bench/tts_fi.py --engines piper-fi --sentences-file my_fi_sentences.txt
```

Moottorit: `piper-fi`, `espeak-fi`, `chatterbox-fi-preset`, `chatterbox-fi-clone`. Oletuslauseet (~30) ovat
skriptissä (foneettisesti vaihtelevia + keskustelureplikoja + numeroita/lyhenteitä/vieraskielisiä nimiä).

### Tulokset
- `bench/results/tts_fi_results.{json,md}` — round-trip WER/CER, TTS RTF, synteesilatenssi per moottori.

**Karsinta = round-trip CER.** Luonnollisuus = neuraalinen MOS (torchaudio SQUIM, ei lisäriippuvuutta):
```bash
# Tarvitsee puhtaan referenssipuheklipin (mikä tahansa puhuja) SQUIM:n non-matching-referenssiksi:
uv run python bench/tts_fi.py --engines piper-fi chatterbox-fi-preset --mos --mos-ref /data/clean_fi.wav
```
Round-trip yksin ei näe robottimaisuutta (eSpeak voi saada hyvän CER:n mutta kuulostaa huonolta), joten
lukitusta ei tehdä pelkän CER:n perusteella — MOS + voittajan **10 klipin omakuuntelu**.

### Referenssi-WAV:n teko (klonaus + MOS)
Klonaus (`--ref-wav`) ja MOS-NMR (`--mos-ref`) tarvitsevat puhtaan suomalaisen puheklipin 16k monona.

**Helpoin — selainpohjainen työkalu `make_ref.py`** (nauhoita mikistä tai raahaa tiedosto → ajaa ffmpegin):
```bash
uv run python bench/make_ref.py          # avaa http://localhost:36399
# Nauhoita ~10–20 s suomea TAI valitse äänitiedosto → "Lähetä" → bench/ref_fi.wav (16k mono)
```
(Vain Python-stdlib, mutta ffmpeg pitää olla PATH:ssa.)

**Tai komentoriviltä** suoraan olemassa olevasta tiedostosta:
```bash
ffmpeg -i jokin_klippi.mp3 -ac 1 -ar 16000 bench/ref_fi.wav
```

### Tulos 2026-06-19 (n=30, round-trip, GPU, täysi ajo)
| Moottori | CER | WER | RTF | synth/lause | Huom |
|---|---:|---:|---:|---:|---|
| **piper-fi** | **3,5 %** | 10,9 % | 0,03 | 0,10 s | paras ymmärrettävyys, ~30× nopein, vakaa |
| chatterbox-fi-preset | 4,9 % | 13,7 % | 0,72 | 3,1 s | epävakaa (toisto/EOS) |
| chatterbox-fi-clone | 6,1 % | 13,1 % | 0,84 | 3,3 s | klonaus ei parantanut; epävakaa |
| espeak-fi | 10,3 % | 33,7 % | 0,01 | 0,04 s | robotti-lattia |

**⚠️ SQUIM-MOS osoittautui epäluotettavaksi** (antoi eSpeakille korkeimman, Piperille matalimman — väärin
päin). Naturalness ratkaistaan **ihmis-kuuntelulla**: aja `--save-audio out_dir` ja kuuntele klipit.
```bash
uv run python bench/tts_fi.py --engines piper-fi chatterbox-fi-preset \
  --save-audio bench/listen --ref-wav bench/ref_fi.wav
# → bench/listen/piper-fi/00.wav … , bench/listen/chatterbox-fi-preset/00.wav …
```

**Suositus: Piper-fi** (paras CER, ~30× nopein, vakaa). Chatterbox parkkiin (ekspressiivisyys/klonaus ei
lunastunut suomeksi + latenssi/epävakaus). Varmistus pikakuuntelulla.
</content>
