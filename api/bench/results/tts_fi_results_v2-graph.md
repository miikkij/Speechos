# Finnish TTS benchmark — v2-graph

Date 2026-10-09 · 38 sentences (8 short) × seeds [0, 1, 2] · judge Systran/faster-whisper-large-v3@edaa852ec7e145841d8ffdb056a99866b5f0a478 (beam 5, VAD) · device cuda · ref voice `reference_finnish.wav`

| Engine | runs | CER mean ± sd | WER mean ± sd | short CER | catastr./run | RTF | TTFA med (s) | TTFA short (s) | utmos22 | utmosv2 | VRAM (GB) | local | License |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| chatterbox-finnish-ref-graph | 3 | 3.29% ± 0.26% | 8.2% ± 1.4% | 4.3% | 1.0 | 0.430 | 1.54 | 1.10 | 3.72 | 3.01 | 6.0 | yes | MIT (code+weights); training data CV-15 CC0, CC-BY, Parliament CLARIN PUB+BY+PRIV |
| chatterbox-finnish-ref-graph-bf16 | 3 | 2.73% ± 0.40% | 7.9% ± 1.7% | 2.8% | 0.3 | 0.316 | 1.16 | 0.93 | 3.71 | 3.01 | 4.6 | yes | MIT (code+weights); training data CV-15 CC0, CC-BY, Parliament CLARIN PUB+BY+PRIV |

- CER/WER: round-trip TTS → STT judge, corpus-level per run; ± = sd across seeds (stability).
- catastr./run: sentences with CER > 30% (repetition, truncation, babble).
- TTFA = time to first audio chunk via the engine API (= full synthesis for non-streaming).
- UTMOS22/UTMOSv2: English-trained naturalness predictors — relative ranking only.
- VRAM: peak torch allocation while the engine ran (includes UTMOS activations, excludes the CT2 judge).
- Revisions and per-sentence transcripts: see the JSON.
