"""UTMOS naturalness predictors for the Finnish TTS bench (relative ranking only).

  - utmos22: UTMOS22 "strong learner" (VoiceMOS 2022 winner) via torch.hub tarepan/SpeechMOS:v1.2.0
             — no extra packages, weights from the SpeechMOS GitHub release.
  - utmosv2: UTMOSv2 (VoiceMOS 2024) from sarulab-speech/UTMOSv2 (extra `bench-tts`, git-pinned).

CAVEAT: both are trained on (mostly English / Japanese) listening-test data. On Finnish they are a
*relative* naturalness signal between engines, not an absolute MOS, and they do not replace listening.
Scores are on a 1–5 scale. Input: float32 mono at any sample rate (resampled to 16 kHz internally).
"""

from __future__ import annotations

import numpy as np

SPEECHMOS_REF = "tarepan/SpeechMOS:v1.2.0"  # tag → commit ed25eacbfa
UTMOSV2_REV = "sarulab-speech/UTMOSv2@cc2700db57bb83ee13dc31ebe1b868c254e15d09"


class Utmos22:
    name = "utmos22"
    revision = SPEECHMOS_REF

    def __init__(self, device: str = "cuda") -> None:
        import torch
        self.device = device
        self.model = torch.hub.load(SPEECHMOS_REF, "utmos22_strong", trust_repo=True).to(device).eval()

    def score(self, audio: np.ndarray, sr: int) -> float:
        import torch
        wave = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return float(self.model(wave, sr).item())


class UtmosV2:
    name = "utmosv2"
    revision = UTMOSV2_REV

    def __init__(self, device: str = "cuda") -> None:
        import utmosv2
        self.device = device
        self.model = utmosv2.create_model(pretrained=True, device=device)

    def score(self, audio: np.ndarray, sr: int) -> float:
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
        # num_workers=0: DataLoader worker processes (spawn) are fragile on Windows.
        out = self.model.predict(data=np.ascontiguousarray(audio, dtype=np.float32), sr=16000,
                                 device=self.device, num_workers=0, verbose=False)
        return float(np.asarray(out).reshape(-1)[0])


def load_scorers(names: list[str], device: str) -> list:
    scorers = []
    for n in names:
        try:
            scorers.append({"utmos22": Utmos22, "utmosv2": UtmosV2}[n](device))
        except Exception as e:  # optional metric — never kill the run
            print(f"  UTMOS scorer {n} unavailable: {type(e).__name__}: {e}", flush=True)
    return scorers
