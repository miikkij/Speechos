"""Chatterbox-Finnish TTS engine (Finnish-NLP/Chatterbox-Finnish) — standalone module, not yet registered.

What it is
    ResembleAI Chatterbox (multilingual base t3_mtl23ls_v2 + S3Gen vocoder) with the Finnish fine-tuned T3
    checkpoint `best_finnish_multilingual_cp986`, loaded exactly like the model card's inference example
    (ChatterboxTTS + EnTokenizer over the multilingual grapheme vocab, fine-tuned T3 injected with
    strict=False). Zero-shot voice: give it a clean Finnish reference clip (≥ 6 s, mono). Without one it
    falls back to the base model's built-in English voice, which is clearly worse in Finnish.

Loading
    `ChatterboxFinnish.load(model_dir, device="cuda")`. Downloads on first use (pinned revisions):
      - <model_dir>/chatterbox-finnish/   code (src/chatterbox_, MIT) + fine-tuned T3 + sample reference voice
                                           from Finnish-NLP/Chatterbox-Finnish @ CHATTERBOX_FI_REVISION
      - HF cache                           ResembleAI/chatterbox @ CHATTERBOX_BASE_REVISION
                                           (ve, t3_mtl23ls_v2, s3gen, tokenizer, conds — ~3.2 GB)
    Python deps: the `tts-fi` extra (resemble-perth, s3tokenizer, conformer, diffusers<0.31, omegaconf,
    einops, safetensors, num2words). The PyPI `chatterbox-tts` package is NOT used (it pins torch 2.6).

Synthesis
    `synthesize(text, seed=None) -> (float32 mono np.ndarray, 24000)` — one call per sentence (the model is
    trained on single utterances; split paragraphs first). Text is run through `normalize_fi`
    (numbers/abbreviations → words) unless normalize=False. Model-card sampling settings:
    repetition_penalty 1.2, temperature 0.8, exaggeration 0.6 (cfg_weight 0.5, min_p 0.05, top_p 1.0).
    Output carries Resemble's Perth implicit watermark (as upstream).

Sample rate
    24 000 Hz (S3GEN_SR). Resample to 16 kHz for analysis models.

Speed / VRAM (RTX 4090, 38 Finnish sentences × 3 seeds, api/bench/results/tts_fi_results_v2-graph.md)
    graph=True (default, CUDA only): T3 decoding through a captured CUDA graph + static KV cache — identical
    tokens to the bundled loop, RTF ≈ 0.43, time-to-first-audio ≈ 1.5 s, peak ≈ 4.4 GB VRAM (fp32).
    graph=False: bundled loop, RTF ≈ 1.3, first audio ≈ 4.6 s, peak ≈ 3.6 GB.
    bf16=True: Llama backbone in bf16 — faster (RTF ≈ 0.35 on 10 sentences) but not bit-identical; see bench.
    CPU works (graph disabled) but is far slower than real time.
"""

from __future__ import annotations

import random
import sys
import threading
from pathlib import Path

import numpy as np

CHATTERBOX_FI_REPO = "Finnish-NLP/Chatterbox-Finnish"
CHATTERBOX_FI_REVISION = "d15775e1788055e67f49dfc6da402021e51bd0f0"
CHATTERBOX_FI_CHECKPOINT = "models/best_finnish_multilingual_cp986.safetensors"
CHATTERBOX_BASE_REPO = "ResembleAI/chatterbox"
CHATTERBOX_BASE_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
SAMPLE_RATE = 24_000

# Model-card settings (README "Running Inference" / inference_example.py).
CARD_PARAMS = {"repetition_penalty": 1.2, "temperature": 0.8, "exaggeration": 0.6}

_SNAPSHOT_PATTERNS = ["src/chatterbox_/*.py", "src/chatterbox_/**/*.py", CHATTERBOX_FI_CHECKPOINT,
                      "samples/reference_finnish.wav", "README.md"]
_BASE_PATTERNS = ["ve.safetensors", "t3_mtl23ls_v2.safetensors", "s3gen.safetensors",
                  "grapheme_mtl_merged_expanded_v1.json", "conds.pt"]


def ensure_snapshot(model_dir: str | Path) -> Path:
    """Return <model_dir>/chatterbox-finnish, downloading the pinned snapshot if it is missing."""
    root = Path(model_dir) / "chatterbox-finnish"
    if not (root / CHATTERBOX_FI_CHECKPOINT).exists() or not (root / "src" / "chatterbox_").exists():
        from huggingface_hub import snapshot_download
        snapshot_download(CHATTERBOX_FI_REPO, revision=CHATTERBOX_FI_REVISION, local_dir=str(root),
                          allow_patterns=_SNAPSHOT_PATTERNS)
    return root


def base_snapshot() -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(CHATTERBOX_BASE_REPO, revision=CHATTERBOX_BASE_REVISION,
                                  allow_patterns=_BASE_PATTERNS))


def import_vendor(snapshot_dir: str | Path):
    """Import the `chatterbox_` package bundled in the Chatterbox-Finnish snapshot.

    The alignment-stream analyzer (active because the vocab is multilingual) needs attention weights;
    transformers ≥ 4.47 no longer falls back from SDPA to eager on its own, so T3 is built with eager
    attention (the graphed decoder then switches all but the analyzer's layers to SDPA).
    """
    src = Path(snapshot_dir) / "src"
    if not (src / "chatterbox_").exists():
        raise FileNotFoundError(f"bundled Chatterbox code missing: {src / 'chatterbox_'} (see ensure_snapshot)")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import chatterbox_
    from chatterbox_.models.t3 import llama_configs

    for cfg in llama_configs.LLAMA_CONFIGS.values():
        if isinstance(cfg, dict) and "attn_implementation" in cfg:
            cfg["attn_implementation"] = "eager"
    return chatterbox_


def load_model(base_dir: Path, finetune: Path, device: str):
    """The card's `ChatterboxTTS.from_local('pretrained_models')` + T3 injection, reading the base files
    straight from the HF cache (no renamed copies). Requires import_vendor() first."""
    import torch
    from chatterbox_.models.s3gen import S3Gen
    from chatterbox_.models.t3 import T3
    from chatterbox_.models.t3.modules.t3_config import T3Config
    from chatterbox_.models.tokenizers import EnTokenizer
    from chatterbox_.models.voice_encoder import VoiceEncoder
    from chatterbox_.tts import ChatterboxTTS, Conditionals
    from safetensors.torch import load_file

    ve = VoiceEncoder()
    ve.load_state_dict(load_file(base_dir / "ve.safetensors"))
    ve.to(device).eval()

    t3_state = load_file(base_dir / "t3_mtl23ls_v2.safetensors")
    config = T3Config.english_only()
    config.text_tokens_dict_size = t3_state["text_emb.weight"].shape[0]  # 2454 (multilingual vocab)
    t3 = T3(hp=config)
    t3.load_state_dict(t3_state, strict=False)
    ft = load_file(str(finetune))
    t3.load_state_dict({k[3:] if k.startswith("t3.") else k: v for k, v in ft.items()}, strict=False)
    t3.to(device).eval()

    s3gen = S3Gen()
    s3gen.load_state_dict(load_file(base_dir / "s3gen.safetensors"), strict=False)
    s3gen.to(device).eval()

    tokenizer = EnTokenizer(str(base_dir / "grapheme_mtl_merged_expanded_v1.json"))
    conds = Conditionals.load(base_dir / "conds.pt", map_location=None if device == "cuda" else "cpu").to(device)
    del t3_state, ft
    if device == "cuda":
        torch.cuda.empty_cache()
    return ChatterboxTTS(t3, s3gen, ve, tokenizer, device, conds=conds)


class ChatterboxFinnish:
    """Loaded Chatterbox-Finnish engine. Thread-safe: synthesis calls are serialized (one GPU model)."""

    sample_rate = SAMPLE_RATE

    def __init__(self, model, ref_wav: str | None, params: dict, normalize: bool) -> None:
        self.model = model
        self.ref_wav = ref_wav
        self.params = params
        self.normalize = normalize
        self._lock = threading.Lock()

    @classmethod
    def load(cls, model_dir: str | Path, device: str = "cuda", ref_wav: str | None = "sample",
             graph: bool = True, bf16: bool = False, normalize: bool = True) -> ChatterboxFinnish:
        """ref_wav: path to a Finnish reference clip, "sample" for the repo's reference_finnish.wav, or None
        for the built-in (English) voice. graph/bf16 apply on CUDA only."""
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        snap = ensure_snapshot(model_dir)
        import_vendor(snap)
        model = load_model(base_snapshot(), snap / CHATTERBOX_FI_CHECKPOINT, device)
        if graph and device == "cuda":
            from src.engines_fi.chatterbox_graph import enable_cuda_graph
            enable_cuda_graph(model, bf16=bf16)
        if ref_wav == "sample":
            ref_wav = str(snap / "samples" / "reference_finnish.wav")
        if ref_wav:
            model.prepare_conditionals(ref_wav, exaggeration=CARD_PARAMS["exaggeration"])
        return cls(model, ref_wav, dict(CARD_PARAMS), normalize)

    def set_voice(self, ref_wav: str) -> None:
        """Switch the cloned voice (computes speaker conditionals once; ~0.3 s)."""
        with self._lock:
            self.model.prepare_conditionals(ref_wav, exaggeration=self.params["exaggeration"])
            self.ref_wav = ref_wav

    def synthesize(self, text: str, seed: int | None = None) -> tuple[np.ndarray, int]:
        import torch

        if self.normalize:
            from src.engines_fi.text_norm_fi import normalize_fi
            text = normalize_fi(text)
        with self._lock:
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed % 2**32)
                torch.manual_seed(seed)
            wav = self.model.generate(text, **self.params)
        return wav.squeeze().cpu().numpy().astype(np.float32), SAMPLE_RATE

    def unload(self) -> None:
        import gc

        import torch

        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
