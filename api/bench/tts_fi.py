"""Finnish TTS benchmark — round-trip intelligibility + stability + latency + UTMOS.

For every engine and every seed (≥3 runs, same sentences): synthesize → transcribe back with the
STT judge (faster-whisper large-v3, the validated Finnish STT) → CER/WER vs. the input text.

Reported per engine (mean ± sd over runs):
  - round-trip CER / WER (corpus-level per run), CER on the short conversational subset
  - catastrophic sentences per run (sentence CER > 30 %: repetition, truncation, babble)
  - TTS RTF (synth time / audio duration) and time-to-first-audio (TTFA, median per sentence;
    = full synthesis time for non-streaming engines, see tts_engines.py)
  - UTMOS22 / UTMOSv2 naturalness (relative only — English-trained predictors, see utmos.py)
  - licence, exact revision, local/cloud, peak torch VRAM

Run (from api/; all bench extras so uv does not prune packages between runs):
  uv run --extra bench --extra bench-tts --extra tts-fi python bench/tts_fi.py \
      --engines piper-fi espeak-fi chatterbox-finnish-ref --seeds 0 1 2 --tag v2
Clips for the listening test: --save-audio bench/listen (10 fixed sentences, seed-0 run), then
  uv run python bench/make_listen_page.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from stt_fi import MODELS as STT_MODELS
from stt_fi import normalize  # same Finnish text normalizer as the STT bench
from tts_engines import ENGINE_KEYS, build_engine, free_gpu
from utmos import load_scorers

SR_STT = 16_000
BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent.parent
CATASTROPHIC_CER = 0.30


# ── Evaluation sentences ───────────────────────────────────────────────
# Phonetically varied + conversational (voice-assistant style) + the usual TTS
# stumbling blocks: numbers, abbreviations, foreign names. Unchanged since 2026-06.
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
# Short conversational turns (added 2026-10): a dialogue system speaks many of these, and
# autoregressive TTS fails differently on very short inputs (cut-offs, babble, long tails).
SHORT_SENTENCES = [
    "Joo, ymmärrän.",
    "Miltä se tuntui?",
    "Kerro lisää.",
    "Aivan, totta.",
    "Selvä, jatketaan.",
    "Mitä sitten tapahtui?",
    "Hyvä kysymys.",
    "Ei se haittaa.",
]
# 10 fixed sentences saved for the blinded listening page (mix of conversational, hard, short).
LISTEN_SENTENCES = [
    "Hei, miten voit tänään?",
    "Ymmärrän, että tilanne tuntuu raskaalta.",
    "Kuulostaa siltä, että olet ollut todella väsynyt viime aikoina.",
    "Olet tehnyt parhaasi, ja se on tärkeää.",
    "Äidin yövalvojaiset äityivät pitkiksi.",
    "Tapaaminen on sovittu tiistaina 24. maaliskuuta.",
    "Maksu oli yhteensä 49,90 euroa.",
    "Mietitäänkö yhdessä, mitä voisit tehdä huomenna?",
    "Joo, ymmärrän.",
    "Miltä se tuntui?",
]


def _resample_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR_STT:
        return audio
    import librosa
    return librosa.resample(audio, orig_sr=sr, target_sr=SR_STT).astype(np.float32)


def _cer(ref: str, hyp: str) -> float:
    import jiwer
    if not ref:
        return 0.0
    return float(jiwer.cer(ref, hyp)) if hyp else 1.0


def _ms(values: list[float]) -> tuple[float, float]:
    """mean, sample sd (0 for a single value)."""
    if not values:
        return 0.0, 0.0
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def run_engine(key: str, sentences: list[str], short_set: set[str], seeds: list[int], whisper, scorers,
               cfg, save_dir: Path | None, text_fn=None) -> dict:
    import jiwer
    import torch

    engine = build_engine(key, cfg)
    info = asdict(engine.info)
    print(f"\nEngine: {key}  [{info['revision']}]", flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_load = time.perf_counter()
    try:
        engine.load()
    except Exception as e:
        print(f"  SKIP {key}: {type(e).__name__}: {e}", flush=True)
        return {"engine": key, "info": info, "skipped": f"{type(e).__name__}: {e}"}
    load_s = time.perf_counter() - t_load

    runs, details = [], []
    try:
        for seed in seeds:
            refs, hyps, refs_short, hyps_short = [], [], [], []
            synth_t, audio_t, ttfas, ttfas_short, catastrophic, failures = 0.0, 0.0, [], [], 0, 0
            mos: dict[str, list[float]] = {s.name: [] for s in scorers}
            for i, sent in enumerate(sentences):
                try:
                    t0 = time.perf_counter()
                    audio, sr, ttfa = engine.synth(text_fn(sent) if text_fn else sent, seed * 10_000 + i)
                    dt = time.perf_counter() - t0
                except Exception as e:  # one bad sentence shouldn't kill the run
                    failures += 1
                    print(f"    [{key} s{seed}] sentence {i} FAILED: {type(e).__name__}: {e}", flush=True)
                    continue
                dur = len(audio) / sr
                synth_t += dt
                audio_t += dur
                ttfas.append(ttfa)
                if sent in short_set:
                    ttfas_short.append(ttfa)

                audio16 = _resample_16k(audio, sr)
                segs, _ = whisper.transcribe(audio16, language="fi", beam_size=5, vad_filter=True)
                hyp = normalize(" ".join(s.text.strip() for s in segs))
                ref = normalize(sent)
                c = _cer(ref, hyp)
                catastrophic += c > CATASTROPHIC_CER
                refs.append(ref)
                hyps.append(hyp)
                if sent in short_set:
                    refs_short.append(ref)
                    hyps_short.append(hyp)
                scores = {}
                for s in scorers:
                    try:
                        scores[s.name] = s.score(audio, sr)
                        mos[s.name].append(scores[s.name])
                    except Exception as e:
                        print(f"    {s.name} failed on sentence {i}: {e}", flush=True)
                details.append({"seed": seed, "i": i, "ref": ref, "hyp": hyp, "cer": round(c, 4),
                                "dur_s": round(dur, 2), "synth_s": round(dt, 3), "ttfa_s": round(ttfa, 3),
                                **{k: round(v, 3) for k, v in scores.items()}})

                if save_dir is not None and seed == seeds[0] and sent in LISTEN_SENTENCES:
                    import soundfile as sf
                    d = save_dir / key
                    d.mkdir(parents=True, exist_ok=True)
                    sf.write(str(d / f"{LISTEN_SENTENCES.index(sent):02d}.wav"), audio, sr)

                if (i + 1) % 10 == 0:
                    print(f"    [{key} s{seed}] {i + 1}/{len(sentences)}", flush=True)

            run = {
                "seed": seed, "n": len(refs), "failures": failures,
                "cer": float(jiwer.cer(refs, hyps)) if refs else 1.0,
                "wer": float(jiwer.wer(refs, hyps)) if refs else 1.0,
                "cer_short": float(jiwer.cer(refs_short, hyps_short)) if refs_short else None,
                "catastrophic": int(catastrophic),
                "rtf": synth_t / audio_t if audio_t else 0.0,
                "ttfa_median_s": statistics.median(ttfas) if ttfas else 0.0,
                "ttfa_short_median_s": statistics.median(ttfas_short) if ttfas_short else 0.0,
                "audio_s": audio_t,
                **{f"{k}_mean": (statistics.mean(v) if v else None) for k, v in mos.items()},
            }
            runs.append(run)
            print(f"  seed {seed}: CER={run['cer']:.2%} WER={run['wer']:.1%} short-CER="
                  f"{(run['cer_short'] or 0):.1%} cata={run['catastrophic']} RTF={run['rtf']:.3f} "
                  f"TTFA={run['ttfa_median_s']:.2f}s " +
                  " ".join(f"{k}={run[f'{k}_mean']:.2f}" for k in mos if run.get(f"{k}_mean")), flush=True)
    finally:
        peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        engine.close()
        del engine
        free_gpu()

    agg = {}
    for m in ["cer", "wer", "cer_short", "catastrophic", "rtf", "ttfa_median_s", "ttfa_short_median_s"] + \
             [f"{s.name}_mean" for s in scorers]:
        vals = [r[m] for r in runs if r.get(m) is not None]
        agg[m], agg[f"{m}_sd"] = _ms(vals)
    return {"engine": key, "info": info, "load_s": round(load_s, 1), "peak_vram_gb": round(peak, 2),
            "runs": runs, "agg": agg, "details": details}


def markdown_table(results: list[dict], scorer_names: list[str]) -> str:
    mos_cols = "".join(f" {n} |" for n in scorer_names)
    head = ("| Engine | runs | CER mean ± sd | WER mean ± sd | short CER | catastr./run | RTF | TTFA med (s) "
            f"| TTFA short (s) |{mos_cols} VRAM (GB) | local | License |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|" + "---:|" * len(scorer_names) + "---:|---|---|\n")
    rows = []
    for r in results:
        info = r["info"]
        if "skipped" in r:
            rows.append(f"| {r['engine']} | — | skipped: {r['skipped'][:90]} |" + " |" * (9 + len(scorer_names))
                        + f" {info['license']} |")
            continue
        a = r["agg"]
        mos = "".join(f" {a[f'{n}_mean']:.2f} |" if a.get(f"{n}_mean") else " — |" for n in scorer_names)
        rows.append(
            f"| {r['engine']} | {len(r['runs'])} | {a['cer']:.2%} ± {a['cer_sd']:.2%} | "
            f"{a['wer']:.1%} ± {a['wer_sd']:.1%} | {a['cer_short']:.1%} | {a['catastrophic']:.1f} | "
            f"{a['rtf']:.3f} | {a['ttfa_median_s']:.2f} | {a['ttfa_short_median_s']:.2f} |{mos} "
            f"{r['peak_vram_gb']:.1f} | {'yes' if info['local'] else 'no (cloud)'} | {info['license']} |")
    return head + "\n".join(rows) + "\n"


def main() -> None:
    # Windows consoles default to cp1252; help text and progress lines contain non-ASCII (≥, →, ä)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Finnish TTS round-trip + stability benchmark")
    ap.add_argument("--engines", nargs="+", default=["piper-fi", "espeak-fi"], choices=ENGINE_KEYS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="one full run per seed (≥3)")
    ap.add_argument("--sentences-file", default=None,
                    help="UTF-8 file, one sentence per line (default: built-in 30 + 8 short)")
    ap.add_argument("--models-root", default=os.environ.get("SPEECHOS_MODELS_ROOT", str(REPO_ROOT / "models")))
    ap.add_argument("--api-models", default=os.environ.get("SPEECHOS_MODEL_DIR", str(REPO_ROOT / "api" / "models")))
    ap.add_argument("--ref-wav", default=None,
                    help="voice reference for *-ref engines (default: Chatterbox-Finnish samples/reference_finnish.wav)")
    ap.add_argument("--judge-model", default="large-v3", help="faster-whisper STT judge (key in stt_fi.MODELS)")
    ap.add_argument("--whisper-cache", default=os.environ.get(
        "SPEECHOS_WHISPER_CACHE", str(REPO_ROOT / "api" / "models" / "whisper")))
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--utmos", nargs="*", default=["utmos22", "utmosv2"], help="naturalness predictors ('' = none)")
    ap.add_argument("--save-audio", default=None, help="dir for the 10 listening clips per engine (seed-0 run)")
    ap.add_argument("--normalize-fi", action="store_true",
                    help="spell out numbers/abbreviations with src/engines_fi/text_norm_fi.py before synthesis "
                         "(the round-trip reference stays the original sentence)")
    ap.add_argument("--tag", default="", help="suffix for result files: tts_fi_results[_<tag>].{json,md}")
    ap.add_argument("--out", default=str(BENCH_DIR / "results"))
    args = ap.parse_args()

    if len(args.seeds) < 3:
        print("NOTE: fewer than 3 seeds — the stability (sd) column is not meaningful.", flush=True)
    ref_wav = args.ref_wav or str(Path(args.models_root) / "chatterbox-finnish" / "samples" / "reference_finnish.wav")
    if any(k.endswith("-ref") for k in args.engines) and not Path(ref_wav).exists():
        ap.error(f"reference WAV not found: {ref_wav}")

    sentences = DEFAULT_SENTENCES + SHORT_SENTENCES
    if args.sentences_file:
        sentences = [ln.strip() for ln in Path(args.sentences_file).read_text(encoding="utf-8").splitlines()
                     if ln.strip()]
    short_set = set(SHORT_SENTENCES)

    from faster_whisper import WhisperModel
    from faster_whisper.utils import download_model
    jspec = STT_MODELS[args.judge_model]
    judge_path = download_model(jspec["repo"], cache_dir=args.whisper_cache, revision=jspec.get("revision"))
    judge_rev = f"{jspec['repo']}@{Path(judge_path).name}"
    print(f"Loading STT judge: {judge_rev} ({args.device}/{args.compute_type}) ...", flush=True)
    whisper = WhisperModel(judge_path, device=args.device, compute_type=args.compute_type)
    scorers = load_scorers([u for u in args.utmos if u], args.device)

    cfg = SimpleNamespace(device=args.device, models_root=args.models_root, api_models=args.api_models,
                          ref_wav=ref_wav)
    save_dir = Path(args.save_audio) if args.save_audio else None
    text_fn = None
    if args.normalize_fi:
        sys.path.insert(0, str(BENCH_DIR.parent))
        from src.engines_fi.text_norm_fi import normalize_fi as text_fn
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "sentences.json").write_text(json.dumps(LISTEN_SENTENCES, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    json_path = out_dir / f"tts_fi_results{suffix}.json"
    md_path = out_dir / f"tts_fi_results{suffix}.md"
    scorer_names = [s.name for s in scorers]
    params = {"engines": args.engines, "seeds": args.seeds, "n_sentences": len(sentences),
              "n_short": len(short_set & set(sentences)), "judge": judge_rev, "judge_beam": 5,
              "judge_vad_filter": True, "ref_wav": Path(ref_wav).name, "device": args.device,
              "utmos": {s.name: s.revision for s in scorers}, "catastrophic_cer": CATASTROPHIC_CER,
              "normalize_fi": bool(args.normalize_fi),
              "date": time.strftime("%Y-%m-%d")}

    results: list[dict] = []
    for key in args.engines:
        try:
            results.append(run_engine(key, sentences, short_set, args.seeds, whisper, scorers, cfg, save_dir,
                                      text_fn=text_fn))
        except Exception as e:
            traceback.print_exc()
            results.append({"engine": key, "info": {"license": "?", "revision": "?", "local": True},
                            "skipped": f"crashed: {type(e).__name__}: {e}"})
        # checkpoint after every engine so a later crash keeps earlier numbers
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"params": params, "sentences": sentences, "results": results}, f, indent=1, ensure_ascii=False)

    md = markdown_table(results, scorer_names)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(
            f"# Finnish TTS benchmark{(' — ' + args.tag) if args.tag else ''}\n\n"
            f"Date {params['date']} · {len(sentences)} sentences ({params['n_short']} short) × seeds {args.seeds} · "
            f"judge {judge_rev} (beam 5, VAD) · device {args.device} · ref voice `{params['ref_wav']}`\n\n{md}\n"
            "- CER/WER: round-trip TTS → STT judge, corpus-level per run; ± = sd across seeds (stability).\n"
            f"- catastr./run: sentences with CER > {CATASTROPHIC_CER:.0%} (repetition, truncation, babble).\n"
            "- TTFA = time to first audio chunk via the engine API (= full synthesis for non-streaming).\n"
            "- UTMOS22/UTMOSv2: English-trained naturalness predictors — relative ranking only.\n"
            "- VRAM: peak torch allocation while the engine ran (includes UTMOS activations, excludes the CT2 judge).\n"
            "- Revisions and per-sentence transcripts: see the JSON.\n")
    print("\n" + "=" * 70)
    print(md)
    print(f"JSON: {json_path}\nMD:   {md_path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
