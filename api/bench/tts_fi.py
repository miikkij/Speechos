"""Finnish TTS quality benchmark — round-trip intelligibility (TTS → STT → CER/WER).

Screening metric for Finnish TTS engines: synthesize a Finnish sentence, transcribe
the audio back with faster-whisper large-v3 (the STT judge, validated separately), and
measure how well the words survived (CER/WER vs. the input text). Also reports TTS RTF.

Per locked decisions (see api/bench/README.md):
  - candidates: Piper-fi, eSpeak-fi, Chatterbox-fi (preset + cloned voice)
  - round-trip CER = screening; UTMOS naturalness ranking is a follow-up (TODO, §below)

Engines are loaded through the production ModelManager (same code path as the API),
so Piper auto-downloads its fi voice and Chatterbox uses the multilingual class.

Run (from api/, WSL2 with GPU):
  uv run python bench/tts_fi.py --engines piper-fi espeak-fi
  uv run python bench/tts_fi.py --engines piper-fi espeak-fi chatterbox-fi-preset
  uv run python bench/tts_fi.py --engines chatterbox-fi-clone --ref-wav /data/ref_fi.wav

Deps:  uv sync --extra bench   (Chatterbox also needs: uv pip install chatterbox-tts;
       eSpeak needs the system binary: apt-get install espeak-ng)
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from stt_fi import normalize  # reuse the same Finnish text normalizer

SR_STT = 16_000


# ── Default Finnish evaluation sentences ───────────────────────────────
# Phonetically varied + conversational (voice-assistant style) + the usual TTS
# stumbling blocks: numbers, abbreviations, foreign names.
DEFAULT_SENTENCES = [
    "Hei, miten voit tänään?",
    "Voisitko kertoa lisää siitä, mikä sinua painaa juuri nyt?",
    "Ymmärrän, että tilanne tuntuu raskaalta.",
    "Otetaan rauhassa, ei ole kiire mihinkään.",
    "Mikä sai sinut tuntemaan noin?",
    "Kuulostaa siltä, että olet ollut todella väsynyt viime aikoina.",
    "Haluatko, että puhutaan tästä tarkemmin?",
    "Joskus pienikin askel eteenpäin riittää.",
    "Olet tehnyt parhaasi, ja se on tärkeää.",
    "Kerro, jos haluat että vaihdamme aihetta.",
    "Syksyn lehdet kahisivat jalkojen alla.",
    "Pöllö huhuili kaukana metsän siimeksessä.",
    "Äidin yövalvojaiset äityivät pitkiksi.",
    "Hän pyöräili töihin sateesta huolimatta.",
    "Kahvi maistui erityisen hyvältä tänä aamuna.",
    "Junat olivat myöhässä lumimyrskyn takia.",
    "Kello on varttia vaille kolme iltapäivällä.",
    "Tapaaminen on sovittu tiistaina 24. maaliskuuta.",
    "Lämpötila laski yöllä viiteentoista pakkasasteeseen.",
    "Maksu oli yhteensä 49,90 euroa.",
    "Soitan sinulle takaisin noin kymmenen minuutin kuluttua.",
    "Osoite on Mannerheimintie 12 A, Helsinki.",
    "Esim. tämä lyhenne saattaa olla TTS-mallille hankala.",
    "Tohtori Virtanen suositteli lepoa ja rauhoittumista.",
    "New York ja Tokio ovat eri aikavyöhykkeillä.",
    "Sähköposti lähetettiin osoitteeseen asiakaspalvelu@example.fi.",
    "Hän kuunteli Sibeliuksen sinfoniaa keskittyneesti.",
    "Älä huoli, kaikki järjestyy vielä.",
    "Mietitäänkö yhdessä, mitä voisit tehdä huomenna?",
    "Kiitos, että jaoit tämän kanssani.",
]


# ── Engine catalogue ───────────────────────────────────────────────────
ENGINES = {
    "piper-fi": {"engine": "piper", "model": "fi_FI-harri-medium", "device": "cpu"},
    "espeak-fi": {"engine": "espeak", "model": "fi", "device": "cpu"},
    "chatterbox-fi-preset": {"engine": "chatterbox", "model": "multilingual",
                             "device": "cuda", "language": "fi"},
    "chatterbox-fi-clone": {"engine": "chatterbox", "model": "multilingual",
                            "device": "cuda", "language": "fi", "clone": True},
}


# ── Synthesis (mirrors src/routers/synthesize.py per engine) ───────────


def _espeak_synth(text: str, lang: str = "fi") -> tuple[np.ndarray, int]:
    """Synthesize via the system espeak-ng binary → float32 mono."""
    proc = subprocess.run(
        ["espeak-ng", "--stdout", "-v", lang, text],
        capture_output=True, check=True,
    )
    with wave.open(io.BytesIO(proc.stdout), "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr


def synth(engine: str, tts_obj, text: str, language: str | None,
          ref_wav: str | None) -> tuple[np.ndarray, int]:
    """Return (float32 mono audio, sample_rate) for one engine."""
    if engine == "espeak":
        return _espeak_synth(text, language or "fi")

    if engine == "piper":
        chunks = []
        sr = 22050
        for ch in tts_obj.synthesize(text):
            sr = ch.sample_rate
            chunks.append(ch.audio_float_array)
        return np.concatenate(chunks).astype(np.float32), sr

    if engine == "chatterbox":
        cb = tts_obj["model"]
        sr = tts_obj.get("sr", 24000)
        kwargs: dict = {}
        if tts_obj.get("multilingual"):
            kwargs["language_id"] = language or "en"
        if ref_wav:
            kwargs["audio_prompt_path"] = ref_wav
        wav = cb.generate(text, **kwargs)
        return wav.squeeze().cpu().numpy().astype(np.float32), sr

    raise ValueError(f"Unsupported bench engine: {engine}")


# ── Benchmark core ─────────────────────────────────────────────────────


@dataclass
class Result:
    engine_key: str
    n: int
    wer: float
    cer: float
    tts_rtf: float
    synth_median_s: float
    total_audio_s: float
    failures: int
    mos_mean: float = 0.0  # neural naturalness MOS (torchaudio SQUIM), 0 = not measured


def _resample_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR_STT:
        return audio
    import librosa
    return librosa.resample(audio, orig_sr=sr, target_sr=SR_STT).astype(np.float32)


def run_engine(engine_key: str, sentences: list[str], whisper, ref_wav: str | None,
               mos_model=None, nmr=None, save_dir: str | None = None) -> Result:
    import jiwer
    import statistics

    spec = ENGINES[engine_key]
    engine = spec["engine"]
    language = spec.get("language")
    use_ref = ref_wav if spec.get("clone") else None

    # Load engine via the production ModelManager (handles downloads + multilingual)
    tts_obj = None
    if engine != "espeak":
        from src.config import ModelConfig, load_config
        from src.models import ModelManager

        mgr = getattr(run_engine, "_mgr", None)
        if mgr is None:
            mgr = ModelManager(load_config())
            run_engine._mgr = mgr  # type: ignore[attr-defined]
        mgr.config.tts = ModelConfig(engine=engine, model=spec["model"], device=spec["device"])
        mgr.unload("tts")
        tts_obj = mgr.get_tts()

    refs, hyps, synth_times, mos_scores = [], [], [], []
    total_audio = 0.0
    failures = 0

    for i, sent in enumerate(sentences):
        try:
            t0 = time.perf_counter()
            audio, sr = synth(engine, tts_obj, sent, language, use_ref)
            synth_times.append(time.perf_counter() - t0)
            total_audio += len(audio) / sr

            if save_dir:
                import soundfile as sf
                d = Path(save_dir) / engine_key
                d.mkdir(parents=True, exist_ok=True)
                sf.write(str(d / f"{i:02d}.wav"), audio, sr)

            audio16 = _resample_16k(audio, sr)
            segments, _ = whisper.transcribe(audio16, language="fi", beam_size=5, vad_filter=True)
            hyp = " ".join(s.text.strip() for s in segments)

            if mos_model is not None and nmr is not None:
                import torch
                test = torch.from_numpy(audio16).unsqueeze(0)
                with torch.no_grad():
                    mos_scores.append(float(mos_model(test, nmr)[0].item()))
        except Exception as e:  # one bad sentence shouldn't kill the run
            failures += 1
            print(f"    [{engine_key}] sentence {i} FAILED: {type(e).__name__}: {e}", flush=True)
            continue

        refs.append(normalize(sent))
        hyps.append(normalize(hyp))
        if (i + 1) % 10 == 0:
            print(f"    [{engine_key}] {i + 1}/{len(sentences)}", flush=True)

    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    refs_f = [r for r, _ in pairs]
    hyps_f = [h for _, h in pairs]
    wer = float(jiwer.wer(refs_f, hyps_f)) if pairs else 1.0
    cer = float(jiwer.cer(refs_f, hyps_f)) if pairs else 1.0
    total_synth = sum(synth_times)

    return Result(
        engine_key=engine_key,
        n=len(pairs),
        wer=round(wer, 4),
        cer=round(cer, 4),
        tts_rtf=round(total_synth / total_audio, 3) if total_audio else 0.0,
        synth_median_s=round(statistics.median(synth_times), 3) if synth_times else 0.0,
        total_audio_s=round(total_audio, 1),
        failures=failures,
        mos_mean=round(statistics.mean(mos_scores), 2) if mos_scores else 0.0,
    )


def markdown_table(results: list[Result]) -> str:
    head = (
        "| Engine | n | round-trip WER | round-trip CER | MOS | TTS RTF | synth med (s) | fails |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = [
        f"| {r.engine_key} | {r.n} | {r.wer:.1%} | {r.cer:.1%} | "
        f"{(f'{r.mos_mean:.2f}' if r.mos_mean else '—')} | {r.tts_rtf:.2f} | "
        f"{r.synth_median_s:.2f} | {r.failures} |"
        for r in results
    ]
    return head + "\n".join(rows) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Finnish TTS round-trip benchmark (TTS->STT->CER)")
    ap.add_argument("--engines", nargs="+", default=["piper-fi", "espeak-fi"],
                    choices=list(ENGINES.keys()))
    ap.add_argument("--sentences-file", default=None,
                    help="UTF-8 text file, one Finnish sentence per line (default: built-in set)")
    ap.add_argument("--ref-wav", default=None, help="reference WAV (16k mono) for chatterbox-fi-clone")
    ap.add_argument("--judge-model", default="large-v3", help="faster-whisper STT judge model")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--mos", action="store_true",
                    help="also score neural naturalness MOS via torchaudio SQUIM (needs --mos-ref)")
    ap.add_argument("--mos-ref", default=None,
                    help="clean speech WAV (any speaker) used as SQUIM non-matching reference")
    ap.add_argument("--save-audio", default=None,
                    help="dir to save each engine's synthesized WAVs (for the human listening test)")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()

    # ── Fail fast on bad inputs (before loading any heavy model) ──
    if "chatterbox-fi-clone" in args.engines and not args.ref_wav:
        ap.error("chatterbox-fi-clone requires --ref-wav <path to 16k mono reference>")
    if args.ref_wav and not Path(args.ref_wav).exists():
        ap.error(f"--ref-wav not found: {args.ref_wav}")
    if args.mos and not args.mos_ref:
        ap.error("--mos requires --mos-ref <clean speech WAV>")
    if args.mos_ref and not Path(args.mos_ref).exists():
        ap.error(f"--mos-ref not found: {args.mos_ref}")

    sentences = DEFAULT_SENTENCES
    if args.sentences_file:
        sentences = [ln.strip() for ln in Path(args.sentences_file).read_text(encoding="utf-8").splitlines()
                     if ln.strip()]

    from faster_whisper import WhisperModel
    print(f"Loading STT judge: {args.judge_model} ({args.device}/{args.compute_type}) ...", flush=True)
    whisper = WhisperModel(args.judge_model, device=args.device, compute_type=args.compute_type,
                           download_root="./models/whisper")

    # Optional neural naturalness MOS (torchaudio SQUIM subjective, CPU, no extra dep)
    mos_model = None
    nmr = None
    if args.mos:
        import torch
        import soundfile as sf
        from torchaudio.pipelines import SQUIM_SUBJECTIVE
        print("Loading SQUIM subjective MOS model ...", flush=True)
        mos_model = SQUIM_SUBJECTIVE.get_model()
        # Load NMR via soundfile — torchaudio.load() routes through torchcodec,
        # whose DLLs fail to load on Windows.
        data, sr = sf.read(args.mos_ref, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != SR_STT:
            data = _resample_16k(data.astype(np.float32), sr)
        nmr = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32)).unsqueeze(0)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[Result] = []
    for key in args.engines:
        print(f"\nEngine: {key} ({len(sentences)} sentences) ...", flush=True)
        res = run_engine(key, sentences, whisper, args.ref_wav, mos_model=mos_model, nmr=nmr,
                         save_dir=args.save_audio)
        results.append(res)
        print(f"  → round-trip WER={res.wer:.1%} CER={res.cer:.1%} TTS RTF={res.tts_rtf:.2f}"
              f" ({res.failures} failures)", flush=True)

    json_path = out_dir / "tts_fi_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
    md = markdown_table(results)
    md_path = out_dir / "tts_fi_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Finnish TTS round-trip benchmark (n={len(sentences)} sentences)\n\n{md}\n"
                "> Screening metric = round-trip CER (intelligibility). UTMOS naturalness "
                "ranking + 10-clip listening test are the follow-up step.\n")

    print("\n" + "=" * 70)
    print(md)
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")
    print("\nNEXT: UTMOS/NISQA naturalness ranking + 10-clip manual listen for the winner.")


if __name__ == "__main__":
    main()
