# Finnish TTS benchmark — v2

Date 2026-10-09 · 38 sentences (8 short) × seeds [0, 1, 2] · judge Systran/faster-whisper-large-v3@edaa852ec7e145841d8ffdb056a99866b5f0a478 (beam 5, VAD) · device cuda · ref voice `reference_finnish.wav`

| Engine | runs | CER mean ± sd | WER mean ± sd | short CER | catastr./run | RTF | TTFA med (s) | TTFA short (s) | utmos22 | utmosv2 | VRAM (GB) | local | License |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| piper-fi | 3 | 4.65% ± 1.33% | 11.5% ± 1.6% | 15.0% | 1.7 | 0.036 | 0.11 | 0.06 | 3.38 | 2.84 | 1.7 | yes | engine GPL-3.0 (piper-tts 1.4); voice data CC0, fine-tuned from lessac (Blizzard-2013 data licence) |
| espeak-fi | 3 | 9.94% ± 0.00% | 31.4% ± 0.0% | 3.7% | 3.0 | 0.022 | 0.04 | 0.04 | 1.77 | 1.73 | 1.7 | yes | GPL-3.0 |
| edge-noora | 3 | 1.95% ± 0.04% | 3.4% ± 0.3% | 0.0% | 0.0 | 0.200 | 0.50 | 0.48 | 3.88 | 3.02 | 1.7 | no (cloud) | Microsoft online service (proprietary, unofficial API) |
| edge-harri | 3 | 2.17% ± 0.04% | 4.1% ± 0.0% | 0.0% | 0.0 | 0.278 | 0.63 | 0.54 | 4.00 | 3.28 | 1.7 | no (cloud) | Microsoft online service (proprietary, unofficial API) |
| mms-tts-fin | 3 | 3.41% ± 0.78% | 13.9% ± 2.4% | 2.8% | 0.0 | 0.021 | 0.07 | 0.07 | 3.31 | 3.17 | 1.9 | yes | CC-BY-NC-4.0 (evaluation only) |
| chatterbox-finnish-ref | 3 | 3.29% ± 0.26% | 8.2% ± 1.4% | 4.3% | 1.0 | 1.281 | 4.64 | 2.97 | 3.72 | 3.01 | 5.2 | yes | MIT (code+weights); training data CV-15 CC0, CC-BY, Parliament CLARIN PUB+BY+PRIV |
| chatterbox-finnish-preset | 3 | 5.34% ± 1.33% | 13.6% ± 1.6% | 9.5% | 2.0 | 1.014 | 2.31 | 1.42 | 3.71 | 3.09 | 5.0 | yes | MIT (code+weights); training data CV-15 CC0, CC-BY, Parliament CLARIN PUB+BY+PRIV |
| chatterbox-finnish-nat-ref | 3 | 5.14% ± 2.49% | 8.9% ± 2.4% | 24.5% | 1.3 | 0.826 | 3.02 | 1.96 | 3.65 | 3.02 | 5.2 | yes | MIT (code+weights); training data CV-15 CC0, CC-BY, Parliament CLARIN PUB+BY+PRIV |
| chatterbox-ml-ref | 3 | 3.95% ± 1.21% | 6.7% ± 1.4% | 0.9% | 1.0 | 0.997 | 4.43 | 2.96 | 3.46 | 2.81 | 5.2 | yes | MIT |
| chatterbox-ml-preset | 3 | 16.89% ± 8.74% | 23.5% ± 7.8% | 59.9% | 4.3 | 1.087 | 5.35 | 4.95 | 3.06 | 2.75 | 5.8 | yes | MIT |
| zipvoice-suomi | — | skipped: RuntimeError: needs k2-fsa/ZipVoice + k2 (no Windows build); run under Linux/WSL | | | | | | | | | | | | CC-BY-ND-4.0 (data Nord-Parl-TTS CC-BY-NC-ND) |

- CER/WER: round-trip TTS → STT judge, corpus-level per run; ± = sd across seeds (stability).
- catastr./run: sentences with CER > 30% (repetition, truncation, babble).
- TTFA = time to first audio chunk via the engine API (= full synthesis for non-streaming).
- UTMOS22/UTMOSv2: English-trained naturalness predictors — relative ranking only.
- VRAM: peak torch allocation while the engine ran (includes UTMOS activations, excludes the CT2 judge).
- Revisions and per-sentence transcripts: see the JSON.
