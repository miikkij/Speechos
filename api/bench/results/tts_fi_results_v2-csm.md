# Finnish TTS benchmark — v2-csm

Date 2026-10-09 · 10 sentences (2 short) × seeds [0] · judge Systran/faster-whisper-large-v3@edaa852ec7e145841d8ffdb056a99866b5f0a478 (beam 5, VAD) · device cuda · ref voice `reference_finnish.wav`

| Engine | runs | CER mean ± sd | WER mean ± sd | short CER | catastr./run | RTF | TTFA med (s) | TTFA short (s) | utmos22 | utmosv2 | VRAM (GB) | local | License |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| csm-fin | 1 | 55.69% ± 0.00% | 65.4% ± 0.0% | 281.5% | 5.0 | 4.446 | 44.66 | 43.68 | 2.52 | 2.45 | 5.0 | yes | weights Apache-2.0, data Nord-Parl-TTS CC-BY-NC-ND-4.0 (evaluation only) |

- CER/WER: round-trip TTS → STT judge, corpus-level per run; ± = sd across seeds (stability).
- catastr./run: sentences with CER > 30% (repetition, truncation, babble).
- TTFA = time to first audio chunk via the engine API (= full synthesis for non-streaming).
- UTMOS22/UTMOSv2: English-trained naturalness predictors — relative ranking only.
- VRAM: peak torch allocation while the engine ran (includes UTMOS activations, excludes the CT2 judge).
- Revisions and per-sentence transcripts: see the JSON.
