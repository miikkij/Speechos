"""Finnish STT quality benchmark — WER / CER / RTF / latency across STT backends.

Datasets (reported PER DATASET, never pooled):
  - fleurs            google/fleurs fi_fi test — read speech, ungated (HF parquet conversion, pinned)
  - voxpopuli         facebook/voxpopuli fi test — European Parliament speech, CC0, ungated (pinned)
  - voxpopuli-short   voxpopuli fi test+validation clips of 1–5 s — short conversational-turn proxy
  - commonvoice       Common Voice fi from a LOCAL corpus (--cv-path). Since Oct 2025 Common Voice is
                      distributed only via Mozilla Data Collective; the HF repo is an empty stub.

Models (see MODELS below; anything else is passed to faster-whisper as an alias/repo/path):
  - large-v3, large-v3-turbo, medium      faster-whisper (CTranslate2)
  - fi-whisper-v3                         Finnish-NLP/whisper-large-finnish-v3 converted to CT2 float16
  - parakeet-tdt-0.6b-v3                  NVIDIA NeMo, run in the isolated bench/nemo uv project
  - qwen3-asr-1.7b                        Qwen3-ASR via qwen-asr (transformers backend, extra bench-qwen)

Metrics: WER/CER (jiwer) on normalized text (lowercase, punctuation stripped, ä/ö/å kept);
RTF = processing time / audio duration; per-clip latency median + p95; cold = first clip.
Every result row records the exact model + dataset revision.

Run (from api/):
  uv run --extra bench python bench/stt_fi.py --models large-v3 --datasets fleurs --n 100
  uv run --extra bench --extra bench-qwen python bench/stt_fi.py \
      --models large-v3 large-v3-turbo fi-whisper-v3 parakeet-tdt-0.6b-v3 qwen3-asr-1.7b \
      --datasets fleurs voxpopuli --n 100 --tag n100
  uv run --extra bench python bench/stt_fi.py --models large-v3 --datasets voxpopuli-short --n 300
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

SR = 16_000
BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent.parent


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


# ── Model registry ─────────────────────────────────────────────────────
# revision = exact HF commit used for the 2026-10 runs (pinned so reruns are reproducible).
MODELS: dict[str, dict] = {
    "large-v3": {"backend": "faster-whisper", "repo": "Systran/faster-whisper-large-v3",
                 "revision": "edaa852ec7e145841d8ffdb056a99866b5f0a478", "license": "MIT"},
    "large-v3-turbo": {"backend": "faster-whisper", "repo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
                       "revision": None, "license": "MIT"},
    "medium": {"backend": "faster-whisper", "repo": "Systran/faster-whisper-medium",
               "revision": None, "license": "MIT"},
    # Finnish fine-tune of large-v3, converted locally (see models/whisper-ct2/*/CONVERSION.txt):
    #   ct2-transformers-converter --model Finnish-NLP/whisper-large-finnish-v3 \
    #     --output_dir <models-root>/whisper-ct2/whisper-large-finnish-v3-ct2-f16 \
    #     --quantization float16 --copy_files tokenizer.json preprocessor_config.json
    "fi-whisper-v3": {"backend": "faster-whisper", "path": "whisper-ct2/whisper-large-finnish-v3-ct2-f16",
                      "repo": "Finnish-NLP/whisper-large-finnish-v3",
                      "revision": "b23deb0b3855c829ffe04cb1c6709757ff16d49c", "license": "Apache-2.0"},
    "parakeet-tdt-0.6b-v3": {"backend": "nemo", "repo": "nvidia/parakeet-tdt-0.6b-v3",
                             "revision": "541d1f99c6b0c3cd0b11a95167540bb8edefd82b", "license": "CC-BY-4.0"},
    "qwen3-asr-1.7b": {"backend": "qwen-asr", "repo": "Qwen/Qwen3-ASR-1.7B",
                       "revision": "7278e1e70fe206f11671096ffdd38061171dd6e5", "license": "Apache-2.0"},
}

# HF parquet conversions (refs/convert/parquet) — pinned commits.
DATASETS = {
    "fleurs": {"repo": "google/fleurs", "config": "fi_fi",
               "revision": "168de341b3db6859a9bac1c50a2ef5e3b47647e0", "license": "CC-BY-4.0"},
    "voxpopuli": {"repo": "facebook/voxpopuli", "config": "fi",
                  "revision": "db0a600a0776f910769e9dc1fdd7f218e8f78a45", "license": "CC0-1.0"},
}


# ── Dataset loaders → list[(audio_float32_16k, reference_text)] ────────


def _decode_bytes_16k(raw: bytes) -> np.ndarray:
    import soundfile as sf

    arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != SR:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _parquet_split(name: str, split: str):
    """HF Dataset for a pinned parquet split; audio left as undecoded bytes."""
    from datasets import Audio, load_dataset
    from huggingface_hub import hf_hub_download

    spec = DATASETS[name]
    path = hf_hub_download(spec["repo"], f"{spec['config']}/{split}/0000.parquet",
                           repo_type="dataset", revision=spec["revision"])
    ds = load_dataset("parquet", data_files=path, split="train")
    return ds.cast_column("audio", Audio(decode=False))


def load_fleurs(n: int, seed: int, split: str = "test") -> list[tuple[np.ndarray, str]]:
    # Same row order as load_dataset("google/fleurs", "fi_fi") → same shuffle(seed) selection
    # as the 2026-06 runs.
    ds = _parquet_split("fleurs", split)
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    out: list[tuple[np.ndarray, str]] = []
    for s in ds:
        ref = (s.get("raw_transcription") or s.get("transcription") or "").strip()
        if ref:
            out.append((_decode_bytes_16k(s["audio"]["bytes"]), ref))
    return out


def load_voxpopuli(n: int, seed: int, splits: tuple[str, ...] = ("test",),
                   min_dur: float = 0.0, max_dur: float = 1e9) -> list[tuple[np.ndarray, str]]:
    import soundfile as sf
    from datasets import concatenate_datasets

    ds = concatenate_datasets([_parquet_split("voxpopuli", sp) for sp in splits])
    ds = ds.shuffle(seed=seed)
    out: list[tuple[np.ndarray, str]] = []
    for s in ds:
        ref = (s.get("raw_text") or "").strip()
        if not ref:
            continue
        raw = s["audio"]["bytes"]
        if min_dur > 0 or max_dur < 1e9:
            dur = sf.info(io.BytesIO(raw)).duration
            if not (min_dur <= dur <= max_dur):
                continue
        out.append((_decode_bytes_16k(raw), ref))
        if len(out) >= n:
            break
    return out


def load_commonvoice_hf(
    n: int, seed: int, hf_name: str, split: str = "test"
) -> list[tuple[np.ndarray, str]]:
    from datasets import Audio, load_dataset

    ds = load_dataset(hf_name, "fi", split=split)
    ds = ds.cast_column("audio", Audio(sampling_rate=SR))
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    out: list[tuple[np.ndarray, str]] = []
    for s in ds:
        ref = (s.get("sentence") or "").strip()
        if ref:
            out.append((np.asarray(s["audio"]["array"], dtype=np.float32), ref))
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
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
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


def load_dataset_clips(name: str, args) -> list[tuple[np.ndarray, str]]:
    if name == "fleurs":
        return load_fleurs(args.n, args.seed)
    if name == "voxpopuli":
        return load_voxpopuli(args.n, args.seed)
    if name == "voxpopuli-short":
        return load_voxpopuli(args.n, args.seed, splits=("test", "validation"), min_dur=1.0, max_dur=5.0)
    if name == "commonvoice":
        if args.cv_path:
            return load_commonvoice_local(args.cv_path, args.n, args.seed, tsv=args.cv_tsv)
        return load_commonvoice_hf(args.n, args.seed, hf_name=args.cv_hf_name)
    raise ValueError(f"Unknown dataset: {name}")


def dataset_revision(name: str, args) -> str:
    if name.startswith("voxpopuli"):
        return f"{DATASETS['voxpopuli']['repo']}@{DATASETS['voxpopuli']['revision']}"
    if name == "fleurs":
        return f"{DATASETS['fleurs']['repo']}@{DATASETS['fleurs']['revision']}"
    if name == "commonvoice":
        return f"local:{args.cv_path}/{args.cv_tsv}" if args.cv_path else args.cv_hf_name
    return "?"


# ── STT backends ───────────────────────────────────────────────────────


class FasterWhisperBackend:
    def __init__(self, key: str, spec: dict, args) -> None:
        from faster_whisper import WhisperModel
        from faster_whisper.utils import download_model

        if "path" in spec:
            path = Path(args.models_root) / spec["path"]
            if not path.exists():
                raise FileNotFoundError(f"{key}: CT2 model not found at {path} (see MODELS comment)")
            model_path = str(path)
            self.revision = f"{spec['repo']}@{spec['revision']} → CT2 float16 ({path.name})"
        else:
            repo = spec.get("repo", key)
            model_path = download_model(repo, cache_dir=args.whisper_cache, revision=spec.get("revision"))
            # snapshot dir name is the commit sha
            self.revision = f"{repo}@{Path(model_path).name}"
        self.model = WhisperModel(model_path, device=args.device, compute_type=args.compute_type)
        self.beam_size = args.beam_size

    def transcribe_all(self, clips: list[np.ndarray]) -> tuple[list[str], list[float]]:
        hyps, times = [], []
        for i, audio in enumerate(clips):
            t0 = time.perf_counter()
            segments, _ = self.model.transcribe(audio, language="fi", beam_size=self.beam_size,
                                                vad_filter=True)
            hyps.append(" ".join(seg.text.strip() for seg in segments))  # generator → consume = compute
            times.append(time.perf_counter() - t0)
            _progress(i, len(clips))
        return hyps, times

    def close(self) -> None:
        del self.model


class QwenAsrBackend:
    def __init__(self, key: str, spec: dict, args) -> None:
        import torch
        from huggingface_hub import snapshot_download

        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as e:
            raise ImportError("qwen-asr missing — run with: uv run --extra bench --extra bench-qwen") from e
        path = snapshot_download(spec["repo"], revision=spec["revision"])
        self.model = Qwen3ASRModel.from_pretrained(
            path, dtype=torch.bfloat16, device_map=f"{args.device}:0" if args.device == "cuda" else "cpu",
            max_inference_batch_size=1, max_new_tokens=256,
        )
        self.revision = f"{spec['repo']}@{spec['revision']}"
        self.device = args.device

    def transcribe_all(self, clips: list[np.ndarray]) -> tuple[list[str], list[float]]:
        import torch

        hyps, times = [], []
        for i, audio in enumerate(clips):
            t0 = time.perf_counter()
            res = self.model.transcribe(audio=(audio, SR), language="Finnish")
            if self.device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            hyps.append(res[0].text)
            _progress(i, len(clips))
        return hyps, times

    def close(self) -> None:
        del self.model


class NemoWorkerBackend:
    """Runs bench/nemo/worker.py in the isolated bench/nemo uv environment (subprocess)."""

    def __init__(self, key: str, spec: dict, args) -> None:
        self.spec = spec
        self.device = args.device
        self.revision = f"{spec['repo']}@{spec['revision']}"

    def transcribe_all(self, clips: list[np.ndarray]) -> tuple[list[str], list[float]]:
        with tempfile.TemporaryDirectory() as td:
            npz = Path(td) / "clips.npz"
            out = Path(td) / "hyps.json"
            np.savez(npz, **{f"audio_{i}": a for i, a in enumerate(clips)})
            cmd = ["uv", "run", "--project", str(BENCH_DIR / "nemo"), "python",
                   str(BENCH_DIR / "nemo" / "worker.py"), "--model", self.spec["repo"],
                   "--revision", self.spec["revision"], "--clips", str(npz), "--out", str(out),
                   "--device", self.device]
            print(f"    [nemo] {' '.join(cmd[:6])} ... ({len(clips)} clips)", flush=True)
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if proc.returncode != 0 or not out.exists():
                tail = "\n".join((proc.stderr or "").splitlines()[-20:])
                raise RuntimeError(f"NeMo worker failed (exit {proc.returncode}):\n{tail}")
            data = json.loads(out.read_text(encoding="utf-8"))
        return data["hyps"], data["times"]

    def close(self) -> None:
        pass


BACKENDS = {"faster-whisper": FasterWhisperBackend, "qwen-asr": QwenAsrBackend, "nemo": NemoWorkerBackend}


def _progress(i: int, n: int) -> None:
    if (i + 1) % 25 == 0 or i + 1 == n:
        print(f"      {i + 1}/{n}", flush=True)


def load_backend(key: str, args):
    spec = MODELS.get(key, {"backend": "faster-whisper", "repo": key, "license": "?"})
    return BACKENDS[spec["backend"]](key, spec, args), spec


def free_gpu() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


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
    model_revision: str = ""
    dataset_revision: str = ""
    license: str = ""
    mean_clip_s: float = 0.0
    date: str = ""
    extra: dict = field(default_factory=dict)


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round(0.95 * (len(s) - 1))))
    return s[idx]


def score(model_name: str, dataset_name: str, clips: list[tuple[np.ndarray, str]],
          hyps: list[str], times: list[float], args) -> tuple[Result, list[dict]]:
    import jiwer

    refs = [normalize(r) for _, r in clips]
    hyps_n = [normalize(h) for h in hyps]
    pairs = [(r, h) for r, h in zip(refs, hyps_n) if r]
    total_audio = sum(len(a) / SR for a, _ in clips)
    total_proc = sum(times)
    res = Result(
        model=model_name, dataset=dataset_name, device=args.device, compute_type=args.compute_type,
        n_clips=len(pairs), seed=args.seed,
        wer=round(float(jiwer.wer([r for r, _ in pairs], [h for _, h in pairs])), 4),
        cer=round(float(jiwer.cer([r for r, _ in pairs], [h for _, h in pairs])), 4),
        total_audio_s=round(total_audio, 1), total_proc_s=round(total_proc, 1),
        rtf=round(total_proc / total_audio, 3) if total_audio else 0.0,
        latency_median_s=round(statistics.median(times), 3) if times else 0.0,
        latency_p95_s=round(_p95(times), 3),
        cold_first_s=round(times[0], 3) if times else 0.0,
        mean_clip_s=round(total_audio / len(clips), 2) if clips else 0.0,
        date=time.strftime("%Y-%m-%d"),
    )
    details = [{"ref": r, "hyp": h, "t": round(t, 3), "dur": round(len(a) / SR, 2)}
               for (a, _), r, h, t in zip(clips, refs, hyps_n, times)]
    return res, details


def markdown_table(results: list[Result]) -> str:
    head = (
        "| Model | Dataset | n | clip avg (s) | WER | CER | RTF | lat.med (s) | lat.p95 (s) | cold (s) | License |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
    )
    rows = [
        f"| {r.model} | {r.dataset} | {r.n_clips} | {r.mean_clip_s:.1f} | {r.wer:.1%} | {r.cer:.2%} | "
        f"{r.rtf:.3f} | {r.latency_median_s:.2f} | {r.latency_p95_s:.2f} | {r.cold_first_s:.2f} | {r.license} |"
        for r in results
    ]
    return head + "\n".join(rows) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Finnish STT WER/CER/RTF benchmark")
    ap.add_argument("--models", nargs="+", default=["large-v3"],
                    help=f"keys {list(MODELS)} or any faster-whisper alias/repo/path")
    ap.add_argument("--datasets", nargs="+", default=["fleurs"],
                    choices=["fleurs", "voxpopuli", "voxpopuli-short", "commonvoice"])
    ap.add_argument("--n", type=int, default=100, help="clips per dataset (100 iterate / 300 lock)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--compute-type", default="float16",
                    help="faster-whisper: float16 (GPU), int8_float16, or int8 (CPU)")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--models-root", default=os.environ.get("SPEECHOS_MODELS_ROOT", str(REPO_ROOT / "models")),
                    help="root for locally converted models (fi-whisper-v3); env SPEECHOS_MODELS_ROOT")
    ap.add_argument("--whisper-cache", default=os.environ.get(
        "SPEECHOS_WHISPER_CACHE", str(REPO_ROOT / "api" / "models" / "whisper")),
        help="faster-whisper download cache; env SPEECHOS_WHISPER_CACHE")
    ap.add_argument("--cv-path", default=None, help="local Common Voice dir (clips/ + test.tsv)")
    ap.add_argument("--cv-tsv", default="test.tsv")
    ap.add_argument("--cv-hf-name", default="mozilla-foundation/common_voice_17_0")
    ap.add_argument("--tag", default="", help="suffix for result files: stt_fi_results[_<tag>].{json,md}")
    ap.add_argument("--out", default=str(BENCH_DIR / "results"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    print(f"Loading datasets {args.datasets} (n={args.n}, seed={args.seed}) ...", flush=True)
    clips_by_ds = {name: load_dataset_clips(name, args) for name in args.datasets}
    for name, clips in clips_by_ds.items():
        dur = sum(len(a) for a, _ in clips) / SR
        print(f"  {name}: {len(clips)} clips, {dur / 60:.1f} min", flush=True)

    results: list[Result] = []
    all_details: dict[str, list[dict]] = {}
    for model_name in args.models:
        print(f"\nLoading model: {model_name} ...", flush=True)
        try:
            backend, spec = load_backend(model_name, args)
        except Exception as e:  # missing optional dep / model → skip, keep the run going
            print(f"  SKIP {model_name}: {type(e).__name__}: {e}", flush=True)
            continue
        try:
            for ds_name in args.datasets:
                print(f"  Benchmarking {model_name} on {ds_name} ...", flush=True)
                clips = clips_by_ds[ds_name]
                hyps, times = backend.transcribe_all([a for a, _ in clips])
                res, details = score(model_name, ds_name, clips, hyps, times, args)
                res.model_revision = backend.revision
                res.dataset_revision = dataset_revision(ds_name, args)
                res.license = spec.get("license", "?")
                results.append(res)
                all_details[f"{model_name}|{ds_name}"] = details
                print(f"    → WER={res.wer:.1%} CER={res.cer:.2%} RTF={res.rtf:.3f} "
                      f"lat.med={res.latency_median_s:.2f}s", flush=True)
        except Exception as e:
            print(f"  FAILED {model_name}: {type(e).__name__}: {e}", flush=True)
        finally:
            backend.close()
            del backend
            free_gpu()

    params = {k: v for k, v in vars(args).items() if k not in ("out",)}
    json_path = out_dir / f"stt_fi_results{suffix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"params": params, "results": [asdict(r) for r in results]}, f, indent=2, ensure_ascii=False)
    with open(out_dir / f"stt_fi_details{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(all_details, f, indent=1, ensure_ascii=False)
    md = markdown_table(results)
    md_path = out_dir / f"stt_fi_results{suffix}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Finnish STT benchmark{(' — ' + args.tag) if args.tag else ''}\n\n"
                f"Date {time.strftime('%Y-%m-%d')} · seed={args.seed} · n≤{args.n}/dataset · "
                f"device={args.device} · faster-whisper compute={args.compute_type}, beam={args.beam_size}, "
                f"vad_filter=True\n\n{md}\n"
                "Model revisions: see `model_revision` in the JSON. WER/CER on normalized text.\n")

    print("\n" + "=" * 70)
    print(md)
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
