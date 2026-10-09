"""NeMo ASR worker for bench/stt_fi.py (runs in the isolated bench/nemo uv environment).

Reads clips from an .npz written by stt_fi.py (`audio_<i>` float32 16 kHz mono arrays), transcribes
them one by one (batch_size=1, so per-clip latency is comparable with the other backends) and
writes {"hyps": [...], "times": [...], "revision": ..., "load_s": ...} as JSON.

    uv run --project bench/nemo python bench/nemo/worker.py \
        --model nvidia/parakeet-tdt-0.6b-v3 --revision <sha> --clips clips.npz --out hyps.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from huggingface_hub import hf_hub_download
    import nemo.collections.asr as nemo_asr

    t0 = time.perf_counter()
    nemo_name = args.model.split("/")[-1] + ".nemo"
    path = hf_hub_download(args.model, nemo_name, revision=args.revision)
    model = nemo_asr.models.ASRModel.restore_from(path, map_location=args.device)
    model = model.to(args.device).eval()
    load_s = time.perf_counter() - t0

    data = np.load(args.clips)
    n = len([k for k in data.files if k.startswith("audio_")])
    hyps: list[str] = []
    times: list[float] = []
    with torch.inference_mode():
        for i in range(n):
            audio = data[f"audio_{i}"].astype(np.float32)
            t = time.perf_counter()
            out = model.transcribe([audio], batch_size=1, verbose=False)
            if args.device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
            hyp = out[0].text if hasattr(out[0], "text") else str(out[0])
            hyps.append(hyp)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"hyps": hyps, "times": times, "revision": args.revision, "load_s": load_s,
                   "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if args.device == "cuda" else 0.0},
                  f, ensure_ascii=False)


if __name__ == "__main__":
    main()
