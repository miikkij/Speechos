# Finnish STT benchmark — n100

Date 2026-10-09 · seed=1234 · n≤100/dataset · device=cuda · faster-whisper compute=float16, beam=5, vad_filter=True

| Model | Dataset | n | clip avg (s) | WER | CER | RTF | lat.med (s) | lat.p95 (s) | cold (s) | License | Train overlap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| large-v3 | fleurs | 100 | 11.9 | 7.9% | 1.46% | 0.052 | 0.58 | 1.02 | 1.40 | MIT |  |
| large-v3 | voxpopuli | 100 | 10.0 | 11.5% | 5.84% | 0.070 | 0.64 | 1.27 | 0.38 | MIT |  |
| large-v3 | voxpopuli-short | 100 | 3.5 | 13.5% | 6.40% | 0.089 | 0.32 | 0.42 | 0.28 | MIT |  |
| large-v3-turbo | fleurs | 100 | 11.9 | 8.0% | 1.56% | 0.018 | 0.21 | 0.30 | 0.23 | MIT |  |
| large-v3-turbo | voxpopuli | 100 | 10.0 | 11.6% | 5.83% | 0.022 | 0.21 | 0.37 | 0.14 | MIT |  |
| large-v3-turbo | voxpopuli-short | 100 | 3.5 | 16.3% | 8.03% | 0.036 | 0.13 | 0.15 | 0.12 | MIT |  |
| fi-whisper-v3 | fleurs | 100 | 11.9 | 6.4% | 1.47% | 0.048 | 0.54 | 0.89 | 0.89 | Apache-2.0 | in-domain: FLEURS-fi train split in training data (test clean) |
| fi-whisper-v3 | voxpopuli | 100 | 10.0 | 6.1% | 4.70% | 0.069 | 0.64 | 1.34 | 0.34 | Apache-2.0 | ⚠ LEAK: VoxPopuli-fi train+val+test in training data |
| fi-whisper-v3 | voxpopuli-short | 100 | 3.5 | 3.5% | 1.95% | 0.099 | 0.35 | 0.49 | 0.30 | Apache-2.0 | ⚠ LEAK: VoxPopuli-fi train+val+test in training data |
| parakeet-tdt-0.6b-v3 | fleurs | 100 | 11.9 | 12.6% | 2.29% | 0.008 | 0.09 | 0.11 | 1.11 | CC-BY-4.0 |  |
| parakeet-tdt-0.6b-v3 | voxpopuli | 100 | 10.0 | 14.5% | 6.30% | 0.012 | 0.11 | 0.15 | 1.27 | CC-BY-4.0 | possible: Granary/MOSEL pseudo-labels VoxPopuli audio |
| parakeet-tdt-0.6b-v3 | voxpopuli-short | 100 | 3.5 | 14.8% | 5.83% | 0.029 | 0.09 | 0.12 | 1.10 | CC-BY-4.0 | possible: Granary/MOSEL pseudo-labels VoxPopuli audio |
| qwen3-asr-1.7b | fleurs | 100 | 11.9 | 27.3% | 6.12% | 0.220 | 2.45 | 4.57 | 5.07 | Apache-2.0 |  |
| qwen3-asr-1.7b | voxpopuli | 100 | 10.0 | 29.1% | 9.38% | 0.236 | 2.19 | 4.16 | 1.45 | Apache-2.0 |  |
| qwen3-asr-1.7b | voxpopuli-short | 100 | 3.5 | 31.7% | 11.13% | 0.262 | 0.91 | 1.50 | 0.90 | Apache-2.0 |  |

Model and dataset revisions: `model_revision` / `dataset_revision` in the JSON. WER/CER on normalized text. ⚠ LEAK rows: the eval split was in that model's training data — not a valid comparison.
