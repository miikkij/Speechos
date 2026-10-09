# Finnish STT benchmark — lock300

Date 2026-10-09 · seed=1234 · n≤300/dataset · device=cuda · faster-whisper compute=float16, beam=5, vad_filter=True

| Model | Dataset | n | clip avg (s) | WER | CER | RTF | lat.med (s) | lat.p95 (s) | cold (s) | License | Train overlap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| fi-whisper-v3 | fleurs | 300 | 12.7 | 6.2% | 1.43% | 0.044 | 0.54 | 0.84 | 1.01 | Apache-2.0 | in-domain: FLEURS-fi train split in training data (test clean) |
| fi-whisper-v3 | voxpopuli | 300 | 9.8 | 5.3% | 3.70% | 0.056 | 0.52 | 1.01 | 0.31 | Apache-2.0 | ⚠ LEAK: VoxPopuli-fi train+val+test in training data |
| large-v3 | fleurs | 300 | 12.7 | 7.5% | 1.53% | 0.044 | 0.53 | 0.88 | 0.89 | MIT |  |
| large-v3 | voxpopuli | 300 | 9.8 | 10.9% | 5.33% | 0.061 | 0.56 | 1.13 | 0.32 | MIT |  |
| large-v3-turbo | fleurs | 300 | 12.7 | 8.1% | 1.65% | 0.018 | 0.21 | 0.34 | 0.40 | MIT |  |
| large-v3-turbo | voxpopuli | 300 | 9.8 | 11.7% | 5.82% | 0.023 | 0.20 | 0.39 | 0.14 | MIT |  |

Model and dataset revisions: `model_revision` / `dataset_revision` in the JSON. WER/CER on normalized text. ⚠ LEAK rows: the eval split was in that model's training data — not a valid comparison.
