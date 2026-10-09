"""Finnish STT quality benchmark — WER / CER / RTF for faster-whisper.

Measures Finnish speech-to-text quality on standardized datasets:
  - FLEURS (google/fleurs, fi_fi)  — ungated, downloads automatically
  - Common Voice (Finnish)         — gated on HF (needs token) OR pass a local --cv-path

Metrics (reported PER DATASET, not pooled):
  - WER, CER (jiwer), on normalized text (lowercase, punctuation stripped, ä/ö/å kept)
  - RTF  = total_processing_time / total_audio_duration
  - latency: per-clip median + p95; cold (first clip after load) reported separately

Decisions baked in:
  - both datasets, 100 clips/dataset for iteration → 300 for the locking run
  - fixed seed for reproducibility

Run (from api/, inside WSL2 with GPU):
  uv run python bench/stt_fi.py --models large-v3 --datasets fleurs --n 100
  uv run python bench/stt_fi.py --models large-v3 medium turbo --datasets fleurs commonvoice --n 100
  uv run python bench/stt_fi.py --datasets commonvoice --cv-path /data/cv-fi --n 300

Install deps:  uv sync --extra bench   (or: uv pip install jiwer datasets)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SR = 16_000


# ── Text normalization ────────────────────────────────────────────────
# Lowercase, drop punctuation, collapse whitespace. \w under Unicode keeps
# Finnish ä/ö/å, so [^\w\s] removes only punctuation/symbols.
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = text.lower()
    text = _PUNCT.sub(" ", text)
    text = _WS.sub(" ", text)
    return text.strip()


# ── Dataset loaders → list[(audio_float32_16k, reference_text)] ────────


def _decode_16k(sample_audio: dict) -> np.ndarray:
    """datasets Audio (already cast to 16k) → float32 mono array."""
    arr = np.asarray(sample_audio["array"], dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1).astype(np.float32)
    return arr


def load_fleurs(n: int, seed: int, split: str = "test") -> list[tuple[np.ndarray, str]]:
    from datasets import Audio, load_dataset

    try:
        ds = load_dataset("google/fleurs", "fi_fi", split=split)
    except Exception:
        ds = load_dataset("google/fleurs", "fi_fi", split=split, trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=SR))
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    out: list[tuple[np.ndarray, str]] = []
    for s in ds:
        ref = (s.get("raw_transcription") or s.get("transcription") or "").strip()
        if not ref:
            continue
        out.append((_decode_16k(s["audio"]), ref))
    return out


def load_commonvoice_hf(
    n: int, seed: int, hf_name: str = "mozilla-foundation/common_voice_17_0", split: str = "test"
) -> list[tuple[np.ndarray, str]]:
    from datasets import Audio, load_dataset

    # Gated dataset — requires `huggingface-cli login` + accepting terms on the dataset page.
    ds = load_dataset(hf_name, "fi", split=split)
    ds = ds.cast_column("audio", Audio(sampling_rate=SR))
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    out: list[tuple[np.ndarray, str]] = []
    for s in ds:
        ref = (s.get("sentence") or "").strip()
        if not ref:
            continue
        out.append((_decode_16k(s["audio"]), ref))
    return out


def load_commonvoice_local(
    cv_path: str, n: int, seed: int, tsv: str = "test.tsv"
) -> list[tuple[np.ndarray, str]]:
    """Local Common Voice corpus: <cv_path>/<tsv> + <cv_path>/clips/*.mp3."""
    import librosa

    root = Path(cv_path)
    tsv_path = root / tsv
    if not tsv_path.exists():
        raise FileNotFoundError(f"Common Voice TSV not found: {tsv_path}")
    clips_dir = root / "clips"

    rows: list[tuple[str, str]] = []
    with open(tsv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            path = row.get("path") or ""
            sentence = (row.get("sentence") or "").strip()
            if path and sentence:
                rows.append((path, sentence))

    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    rows = rows[:n]

    out: list[tuple[np.ndarray, str]] = []
    for path, sentence in rows:
        audio_path = clips_dir / path if not Path(path).is_absolute() else Path(path)
        if not audio_path.exists():
            audio_path = root / path  # some exports omit clips/
        y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
        out.append((y.astype(np.float32), sentence))
    return out


# ── Benchmark core ─────────────────────────────────────────────────────


@dataclass
class Result:
    model: str
    dataset: str
    device: str
    compute_type: str
    n_clips: int
    seed: int
    wer: float
    cer: float
    total_audio_s: float
    total_proc_s: float
    rtf: float
    latency_median_s: float
    latency_p95_s: float
    cold_first_s: float


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round(0.95 * (len(s) - 1))))
    return s[idx]


def run_model_on_dataset(
    model, model_name: str, dataset_name: str, clips: list[tuple[np.ndarray, str]],
    device: str, compute_type: str, seed: int,
) -> Result:
    import jiwer

    refs: list[str] = []
    hyps: list[str] = []
    per_clip_time: list[float] = []
    total_audio = 0.0

    for i, (audio, ref) in enumerate(clips):
        t0 = time.perf_counter()
        segments, _info = model.transcribe(audio, language="fi", beam_size=5, vad_filter=True)
        hyp = " ".join(seg.text.strip() for seg in segments)  # generator → consume = compute
        dt = time.perf_counter() - t0

        per_clip_time.append(dt)
        total_audio += len(audio) / SR
        refs.append(normalize(ref))
        hyps.append(normalize(hyp))
        if (i + 1) % 10 == 0:
            print(f"    [{dataset_name}/{model_name}] {i + 1}/{len(clips)} clips", flush=True)

    # jiwer ignores empty refs poorly; filter pairs with empty normalized ref
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    refs_f = [r for r, _ in pairs]
    hyps_f = [h for _, h in pairs]

    wer = float(jiwer.wer(refs_f, hyps_f))
    cer = float(jiwer.cer(refs_f, hyps_f))
    total_proc = sum(per_clip_time)

    return Result(
        model=model_name,
        dataset=dataset_name,
        device=device,
        compute_type=compute_type,
        n_clips=len(pairs),
        seed=seed,
        wer=round(wer, 4),
        cer=round(cer, 4),
        total_audio_s=round(total_audio, 1),
        total_proc_s=round(total_proc, 1),
        rtf=round(total_proc / total_audio, 3) if total_audio else 0.0,
        latency_median_s=round(statistics.median(per_clip_time), 3) if per_clip_time else 0.0,
        latency_p95_s=round(_p95(per_clip_time), 3),
        cold_first_s=round(per_clip_time[0], 3) if per_clip_time else 0.0,
    )


def load_dataset_clips(name: str, args) -> list[tuple[np.ndarray, str]]:
    if name == "fleurs":
        return load_fleurs(args.n, args.seed)
    if name == "commonvoice":
        if args.cv_path:
            return load_commonvoice_local(args.cv_path, args.n, args.seed, tsv=args.cv_tsv)
        return load_commonvoice_hf(args.n, args.seed, hf_name=args.cv_hf_name)
    raise ValueError(f"Unknown dataset: {name}")


def markdown_table(results: list[Result]) -> str:
    head = (
        "| Model | Dataset | n | WER | CER | RTF | lat.med (s) | lat.p95 (s) | cold (s) |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    for r in results:
        rows.append(
            f"| {r.model} | {r.dataset} | {r.n_clips} | {r.wer:.1%} | {r.cer:.1%} | "
            f"{r.rtf:.2f} | {r.latency_median_s:.2f} | {r.latency_p95_s:.2f} | {r.cold_first_s:.2f} |"
        )
    return head + "\n".join(rows) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Finnish STT WER/CER/RTF benchmark (faster-whisper)")
    ap.add_argument("--models", nargs="+", default=["large-v3"],
                    help="faster-whisper model ids, e.g. large-v3 medium turbo "
                         "Systran/faster-distil-whisper-large-v3")
    ap.add_argument("--datasets", nargs="+", default=["fleurs"],
                    choices=["fleurs", "commonvoice"])
    ap.add_argument("--n", type=int, default=100, help="clips per dataset (100 iterate / 300 lock)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--compute-type", default="float16",
                    help="float16 (GPU), int8_float16, or int8 (CPU)")
    ap.add_argument("--cv-path", default=None, help="local Common Voice dir (clips/ + test.tsv)")
    ap.add_argument("--cv-tsv", default="test.tsv")
    ap.add_argument("--cv-hf-name", default="mozilla-foundation/common_voice_17_0")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets once (shared across models for fair comparison)
    print(f"Loading datasets {args.datasets} (n={args.n}, seed={args.seed}) ...", flush=True)
    clips_by_ds = {name: load_dataset_clips(name, args) for name in args.datasets}
    for name, clips in clips_by_ds.items():
        print(f"  {name}: {len(clips)} clips", flush=True)

    results: list[Result] = []
    for model_name in args.models:
        print(f"\nLoading model: {model_name} ({args.device}/{args.compute_type}) ...", flush=True)
        model = WhisperModel(model_name, device=args.device, compute_type=args.compute_type,
                             download_root="./models/whisper")
        for ds_name in args.datasets:
            print(f"  Benchmarking {model_name} on {ds_name} ...", flush=True)
            res = run_model_on_dataset(
                model, model_name, ds_name, clips_by_ds[ds_name],
                args.device, args.compute_type, args.seed,
            )
            results.append(res)
            print(f"    → WER={res.wer:.1%} CER={res.cer:.1%} RTF={res.rtf:.2f}", flush=True)

    # Write outputs
    json_path = out_dir / "stt_fi_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
    md = markdown_table(results)
    md_path = out_dir / "stt_fi_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Finnish STT benchmark (seed={args.seed}, n={args.n}/dataset)\n\n{md}")

    print("\n" + "=" * 70)
    print(md)
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")


if __name__ == "__main__":
    main()
