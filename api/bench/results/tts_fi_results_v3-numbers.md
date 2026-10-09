# Finnish TTS benchmark — v3-numbers

Date 2026-10-09 · 10 sentences (0 short) × seeds [0, 1, 2] · judge Systran/faster-whisper-large-v3@edaa852ec7e145841d8ffdb056a99866b5f0a478 (beam 5, VAD) · device cuda · ref voice `reference_finnish.wav`

| Engine | runs | CER mean ± sd | WER mean ± sd | short CER | catastr./run | RTF | TTFA med (s) | TTFA short (s) | utmos22 | VRAM (GB) | local | License |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| piper-fi | 3 | 12.50% ± 1.79% | 21.9% ± 4.1% | 0.0% | 1.0 | 0.034 | 0.15 | 0.00 | 3.34 | 0.6 | yes | engine GPL-3.0 (piper-tts 1.4); voice data CC0, fine-tuned from lessac (Blizzard-2013 data licence) |
| chatterbox-finnish-ref-graph | 3 | 22.79% ± 8.84% | 38.0% ± 2.4% | 0.0% | 3.0 | 0.812 | 3.78 | 0.00 | 3.73 | 4.9 | yes | MIT (code+weights); training data CV-15 CC0, CC-BY, Parliament CLARIN PUB+BY+PRIV |

- CER/WER: round-trip TTS → STT judge, corpus-level per run; ± = sd across seeds (stability).
- catastr./run: sentences with CER > 30% (repetition, truncation, babble).
- TTFA = time to first audio chunk via the engine API (= full synthesis for non-streaming).
- UTMOS22/UTMOSv2: English-trained naturalness predictors — relative ranking only.
- VRAM: peak torch allocation while the engine ran (includes UTMOS activations, excludes the CT2 judge).
- Revisions and per-sentence transcripts: see the JSON.
