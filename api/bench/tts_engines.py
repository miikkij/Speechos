"""Finnish TTS engine adapters for bench/tts_fi.py.

Each adapter: load() → synth(text, seed) → (float32 mono audio, sample_rate, ttfa_s) → close().
ttfa_s = time-to-first-audio: seconds until the first audio chunk is available to the caller through
the engine's native API. For non-streaming engines (Chatterbox, MMS, CSM) it equals the full synthesis
time of the sentence; Piper yields per-sentence chunks; edge-tts streams MP3 chunks over the network.

Model locations (never re-downloaded into the worktree):
  --models-root  (env SPEECHOS_MODELS_ROOT, default <repo>/models)      chatterbox-finnish snapshot
  --api-models   (env SPEECHOS_MODEL_DIR,   default <repo>/api/models)  piper voices
HF-hub models (Chatterbox base, MMS, CSM) live in the normal HF cache, pinned by revision.
"""

from __future__ import annotations

import asyncio
import gc
import io
import random
import subprocess
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CHATTERBOX_BASE = ("ResembleAI/chatterbox", "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18")
CHATTERBOX_FI = ("Finnish-NLP/Chatterbox-Finnish", "d15775e1788055e67f49dfc6da402021e51bd0f0")
MMS_FIN = ("facebook/mms-tts-fin", "0b8a1a27e4d22423e67381bd16ff13b9e6215c71")
CSM_FIN = ("ArttuPakarinen/sesame-csm-FIN-parlament-full-finetune", "f297d601e7cb05ce621206e1b18e3260ef206e6e")
PIPER_VOICES_REV = "v1.0.0 (375a0fe641dea077c2a47b4e9a056d6da521eed3)"


@dataclass
class EngineInfo:
    key: str
    license: str
    revision: str
    local: bool = True
    device: str = "cpu"
    notes: str = ""
    params: dict = field(default_factory=dict)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def free_gpu() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _sync_cuda() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


# ── CPU / cloud engines ────────────────────────────────────────────────


class PiperEngine:
    def __init__(self, cfg) -> None:
        self.voice_name = "fi_FI-harri-medium"
        self.path = Path(cfg.api_models) / "piper" / f"{self.voice_name}.onnx"
        self.info = EngineInfo(
            "piper-fi", license="engine GPL-3.0 (piper-tts 1.4); voice data CC0, fine-tuned from lessac "
            "(Blizzard-2013 data licence)", revision=f"rhasspy/piper-voices {PIPER_VOICES_REV} {self.voice_name}",
            notes="ONNX noise not seedable")

    def load(self) -> None:
        from piper import PiperVoice
        if not self.path.exists():
            raise FileNotFoundError(f"Piper voice missing: {self.path}")
        self.voice = PiperVoice.load(str(self.path))

    def synth(self, text: str, seed: int) -> tuple[np.ndarray, int, float]:
        t0 = time.perf_counter()
        ttfa = None
        chunks, sr = [], 22050
        for ch in self.voice.synthesize(text):
            if ttfa is None:
                ttfa = time.perf_counter() - t0
            sr = ch.sample_rate
            chunks.append(ch.audio_float_array)
        return np.concatenate(chunks).astype(np.float32), sr, ttfa or 0.0

    def close(self) -> None:
        self.voice = None


class EspeakEngine:
    def __init__(self, cfg) -> None:
        self.info = EngineInfo("espeak-fi", license="GPL-3.0", revision=_espeak_version())

    def load(self) -> None:
        pass

    def synth(self, text: str, seed: int) -> tuple[np.ndarray, int, float]:
        t0 = time.perf_counter()
        proc = subprocess.run(["espeak-ng", "--stdout", "-v", "fi", text], capture_output=True, check=True)
        ttfa = time.perf_counter() - t0
        with wave.open(io.BytesIO(proc.stdout), "rb") as wf:
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
        return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0, sr, ttfa

    def close(self) -> None:
        pass


def _espeak_version() -> str:
    try:
        out = subprocess.run(["espeak-ng", "--version"], capture_output=True, text=True).stdout
        return out.strip().split(" Data at")[0]
    except OSError:
        return "espeak-ng (not found)"


class EdgeEngine:
    """Microsoft Edge neural voices (cloud). Reference point only — not a local option."""

    def __init__(self, cfg, key: str, voice: str) -> None:
        import importlib.metadata as md
        self.voice = voice
        self.info = EngineInfo(key, license="Microsoft online service (proprietary, unofficial API)",
                               revision=f"edge-tts {md.version('edge-tts')} voice {voice}", local=False,
                               device="cloud")

    def load(self) -> None:
        pass

    async def _stream(self, text: str) -> tuple[bytes, float]:
        import edge_tts
        t0 = time.perf_counter()
        ttfa = None
        buf = bytearray()
        async for chunk in edge_tts.Communicate(text, self.voice).stream():
            if chunk["type"] == "audio":
                if ttfa is None:
                    ttfa = time.perf_counter() - t0
                buf.extend(chunk["data"])
        return bytes(buf), ttfa or 0.0

    def synth(self, text: str, seed: int) -> tuple[np.ndarray, int, float]:
        import soundfile as sf
        data, ttfa = asyncio.run(self._stream(text))
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32), sr, ttfa

    def close(self) -> None:
        pass


# ── Chatterbox (bundled code from Finnish-NLP/Chatterbox-Finnish) ──────


def import_chatterbox_vendor(models_root: str):
    """Import the `chatterbox_` package shipped in the Chatterbox-Finnish snapshot.

    PyPI chatterbox-tts pins torch==2.6 / transformers==5.2 and cannot share the API lock, so both the
    base multilingual model and the Finnish fine-tune run on this bundled copy (same upstream code).
    """
    src = Path(models_root) / "chatterbox-finnish" / "src"
    if not (src / "chatterbox_").exists():
        raise FileNotFoundError(
            f"{src / 'chatterbox_'} missing — snapshot_download('{CHATTERBOX_FI[0]}', revision='{CHATTERBOX_FI[1]}', "
            f"local_dir='<models-root>/chatterbox-finnish')")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import chatterbox_  # noqa: F401
    from chatterbox_.models.t3 import llama_configs

    # The alignment-stream analyzer (multilingual vocab → always on) needs output_attentions=True.
    # transformers 4.46 (what the bundled code was written for) silently fell back from SDPA to eager;
    # 4.57 raises instead, so build T3 with eager attention up front. In-memory only.
    for cfg in llama_configs.LLAMA_CONFIGS.values():
        if isinstance(cfg, dict) and "attn_implementation" in cfg:
            cfg["attn_implementation"] = "eager"
    return chatterbox_


def _base_snapshot() -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(
        CHATTERBOX_BASE[0], revision=CHATTERBOX_BASE[1],
        allow_patterns=["ve.pt", "ve.safetensors", "t3_mtl23ls_v2.safetensors", "s3gen.pt", "s3gen.safetensors",
                        "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"]))


class ChatterboxMultilingualEngine:
    """Base Chatterbox Multilingual (t3_mtl23ls_v2) with language_id='fi'."""

    def __init__(self, cfg, key: str, ref_wav: str | None) -> None:
        self.cfg = cfg
        self.ref_wav = ref_wav
        self.params = {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8, "repetition_penalty": 2.0}
        self.info = EngineInfo(key, license="MIT", revision=f"{CHATTERBOX_BASE[0]}@{CHATTERBOX_BASE[1]} t3_mtl23ls_v2",
                               device=cfg.device, params={**self.params, "ref": Path(ref_wav).name if ref_wav else "conds.pt"},
                               notes="Perth watermark applied")

    def load(self) -> None:
        cb = import_chatterbox_vendor(self.cfg.models_root)
        self.model = cb.mtl_tts.ChatterboxMultilingualTTS.from_local(_base_snapshot(), self.cfg.device)
        if self.ref_wav:
            self.model.prepare_conditionals(self.ref_wav, exaggeration=self.params["exaggeration"])

    def synth(self, text: str, seed: int) -> tuple[np.ndarray, int, float]:
        seed_everything(seed)
        t0 = time.perf_counter()
        wav = self.model.generate(text, language_id="fi", **self.params)
        _sync_cuda()
        dt = time.perf_counter() - t0
        return wav.squeeze().cpu().numpy().astype(np.float32), self.model.sr, dt

    def close(self) -> None:
        del self.model
        free_gpu()


class ChatterboxFinnishEngine:
    """Finnish-NLP/Chatterbox-Finnish: base Chatterbox + fine-tuned T3, exactly as in the model card.

    Card inference path: ChatterboxTTS (EnTokenizer over the multilingual grapheme vocab, no [fi] token)
    with t3_mtl23ls_v2 base weights, then the fine-tuned T3 state dict loaded on top (strict=False).
    """

    CHECKPOINTS = {"cp986": "best_finnish_multilingual_cp986.safetensors",
                   "cp1060": "best_naturalness_cp1060.safetensors"}

    def __init__(self, cfg, key: str, ref_wav: str | None, checkpoint: str = "cp986") -> None:
        self.cfg = cfg
        self.ref_wav = ref_wav
        self.checkpoint = checkpoint
        # Model-card settings (README "Running Inference" + inference_example.py).
        self.params = {"repetition_penalty": 1.2, "temperature": 0.8, "exaggeration": 0.6}
        self.info = EngineInfo(
            key, license="MIT (code+weights); training data CV-15 CC0, CC-BY, Parliament CLARIN PUB+BY+PRIV",
            revision=f"{CHATTERBOX_FI[0]}@{CHATTERBOX_FI[1]} {self.CHECKPOINTS[checkpoint]} + "
                     f"{CHATTERBOX_BASE[0]}@{CHATTERBOX_BASE[1][:10]}",
            device=cfg.device, params={**self.params, "ref": Path(ref_wav).name if ref_wav else "conds.pt"},
            notes="Perth watermark applied")

    def load(self) -> None:
        from safetensors.torch import load_file

        cb = import_chatterbox_vendor(self.cfg.models_root)
        self.model = load_chatterbox_finnish(cb, _base_snapshot(),
                                             Path(self.cfg.models_root) / "chatterbox-finnish" / "models" /
                                             self.CHECKPOINTS[self.checkpoint], self.cfg.device, load_file)
        if self.ref_wav:
            self.model.prepare_conditionals(self.ref_wav, exaggeration=self.params["exaggeration"])

    def synth(self, text: str, seed: int) -> tuple[np.ndarray, int, float]:
        seed_everything(seed)
        t0 = time.perf_counter()
        wav = self.model.generate(text, **self.params)
        _sync_cuda()
        dt = time.perf_counter() - t0
        return wav.squeeze().cpu().numpy().astype(np.float32), self.model.sr, dt

    def close(self) -> None:
        del self.model
        free_gpu()


def load_chatterbox_finnish(cb, base_dir: Path, finetune: Path, device: str, load_file):
    """Mirror of the card's ChatterboxTTS.from_local('pretrained_models') + T3 injection, reading the
    base files straight from the HF cache (no renamed copies)."""
    import torch
    from chatterbox_.models.s3gen import S3Gen
    from chatterbox_.models.t3 import T3
    from chatterbox_.models.t3.modules.t3_config import T3Config
    from chatterbox_.models.tokenizers import EnTokenizer
    from chatterbox_.models.voice_encoder import VoiceEncoder
    from chatterbox_.tts import ChatterboxTTS, Conditionals

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
    torch.cuda.empty_cache() if device == "cuda" else None
    return ChatterboxTTS(t3, s3gen, ve, tokenizer, device, conds=conds)


# ── transformers engines (evaluation only, non-commercial) ─────────────


class MmsEngine:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.info = EngineInfo("mms-tts-fin", license="CC-BY-NC-4.0 (evaluation only)",
                               revision=f"{MMS_FIN[0]}@{MMS_FIN[1]}", device=cfg.device)

    def load(self) -> None:
        from transformers import AutoTokenizer, VitsModel
        self.tok = AutoTokenizer.from_pretrained(MMS_FIN[0], revision=MMS_FIN[1])
        self.model = VitsModel.from_pretrained(MMS_FIN[0], revision=MMS_FIN[1]).to(self.cfg.device).eval()

    def synth(self, text: str, seed: int) -> tuple[np.ndarray, int, float]:
        import torch
        seed_everything(seed)
        t0 = time.perf_counter()
        inputs = self.tok(text, return_tensors="pt").to(self.cfg.device)
        with torch.no_grad():
            wav = self.model(**inputs).waveform
        _sync_cuda()
        dt = time.perf_counter() - t0
        return wav.squeeze().float().cpu().numpy(), self.model.config.sampling_rate, dt

    def close(self) -> None:
        del self.model
        free_gpu()


class CsmFinEngine:
    """Sesame CSM-1B fine-tuned on Finnish parliament speech (Nord-Parl-TTS, CC-BY-NC-ND data).

    The processor files ship in the fine-tune repo itself (ungated); the gated base sesame/csm-1b is
    not accessed. Settings from the model card test script.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 600}
        self.info = EngineInfo("csm-fin", license="weights Apache-2.0, data Nord-Parl-TTS CC-BY-NC-ND-4.0 "
                               "(evaluation only)", revision=f"{CSM_FIN[0]}@{CSM_FIN[1]}", device=cfg.device,
                               params=self.params)

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, CsmForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(CSM_FIN[0], revision=CSM_FIN[1])
        self.model = CsmForConditionalGeneration.from_pretrained(
            CSM_FIN[0], revision=CSM_FIN[1], torch_dtype=torch.bfloat16 if self.cfg.device == "cuda" else "auto",
            attn_implementation="eager").to(self.cfg.device).eval()

    def synth(self, text: str, seed: int) -> tuple[np.ndarray, int, float]:
        import torch
        seed_everything(seed)
        t0 = time.perf_counter()
        conv = [{"role": "0", "content": [{"type": "text", "text": text}]}]
        raw = self.processor.apply_chat_template(conv, tokenize=True, add_generation_prompt=True,
                                                 return_tensors="pt", return_dict=True)
        inputs = {k: v.to(self.cfg.device) for k, v in raw.items()}
        with torch.no_grad():
            audio = self.model.generate(**inputs, output_audio=True, do_sample=True, **self.params)
        _sync_cuda()
        dt = time.perf_counter() - t0
        wav = audio[0] if isinstance(audio, (list, tuple)) else audio
        return wav.squeeze().float().cpu().numpy(), 24000, dt

    def close(self) -> None:
        del self.model
        free_gpu()


class SkippedEngine:
    """Placeholder for candidates that cannot run here; recorded in the results with the reason."""

    def __init__(self, key: str, license: str, revision: str, reason: str) -> None:
        self.info = EngineInfo(key, license=license, revision=revision, notes=reason)
        self.reason = reason

    def load(self) -> None:
        raise RuntimeError(self.reason)


def build_engine(key: str, cfg):
    """Engine catalogue. cfg needs: device, models_root, api_models, ref_wav (path or None)."""
    ref = cfg.ref_wav
    if key == "piper-fi":
        return PiperEngine(cfg)
    if key == "espeak-fi":
        return EspeakEngine(cfg)
    if key == "edge-noora":
        return EdgeEngine(cfg, key, "fi-FI-NooraNeural")
    if key == "edge-harri":
        return EdgeEngine(cfg, key, "fi-FI-HarriNeural")
    if key == "chatterbox-ml-preset":
        return ChatterboxMultilingualEngine(cfg, key, None)
    if key == "chatterbox-ml-ref":
        return ChatterboxMultilingualEngine(cfg, key, ref)
    if key == "chatterbox-finnish-ref":
        return ChatterboxFinnishEngine(cfg, key, ref)
    if key == "chatterbox-finnish-preset":
        return ChatterboxFinnishEngine(cfg, key, None)
    if key == "chatterbox-finnish-nat-ref":
        return ChatterboxFinnishEngine(cfg, key, ref, checkpoint="cp1060")
    if key == "mms-tts-fin":
        return MmsEngine(cfg)
    if key == "csm-fin":
        return CsmFinEngine(cfg)
    if key == "zipvoice-suomi":
        return SkippedEngine(
            key, "CC-BY-ND-4.0 (data Nord-Parl-TTS CC-BY-NC-ND)", "TheBurgstall/ZipVoice-Suomi-12k@6b031a602a16",
            "needs k2-fsa/ZipVoice + k2 (no Windows build); run under Linux/WSL")
    raise ValueError(f"Unknown TTS engine: {key}")


ENGINE_KEYS = ["piper-fi", "espeak-fi", "edge-noora", "edge-harri", "chatterbox-ml-preset", "chatterbox-ml-ref",
               "chatterbox-finnish-ref", "chatterbox-finnish-preset", "chatterbox-finnish-nat-ref", "mms-tts-fin",
               "csm-fin", "zipvoice-suomi"]
