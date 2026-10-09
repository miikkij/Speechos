# Number/abbreviation reading — key-item accuracy (2026-10-09)

10 sentences (`bench/sentences_numbers_fi.txt`) × 3 seeds, judge faster-whisper large-v3, torch 2.11 stack. Produced by `bench/number_items.py` from `tts_fi_results_v3-numbers{,-norm}.json`. Round-trip CER is not used here: a correct verbalization comes back as a mix of digits and words that matches neither written form.

| Results | Engine | key items | sentences fully right |
|---|---|---:|---:|
| raw text | piper-fi | 72/90 = 80% | 16/30 |
| raw text | chatterbox-finnish-ref-graph | 36/90 = 40% | 9/30 |
| normalize_fi | piper-fi | 82/90 = 91% | 23/30 |
| normalize_fi | chatterbox-finnish-ref-graph | 89/90 = 99% | 29/30 |
