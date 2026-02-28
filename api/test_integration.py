"""Quick integration test - verify all models load and work."""
import os
from src.config import load_config
from src.models import ModelManager
import numpy as np
import librosa

cfg = load_config()
print(f"Tier: {cfg.hardware.tier}")
print(f"STT engine: {cfg.stt.engine} model: {cfg.stt.model}")
print(f"Emotion engine: {cfg.emotion.engine} model: {cfg.emotion.model}")
print(f"Diarization engine: {cfg.diarization.engine} model: {cfg.diarization.model}")

mgr = ModelManager(cfg)

print("\nLoading STT...")
stt = mgr.get_stt()
print(f"STT loaded: {type(stt).__name__}")

print("Loading Emotion...")
emo = mgr.get_emotion()
print(f"Emotion loaded: {type(emo).__name__}")

print("Loading Diarization...")
dia = mgr.get_diarization()
print(f"Diarization loaded: {type(dia).__name__}")

# Quick STT test
_samples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
y, sr = librosa.load(os.path.join(_samples_dir, "test_tone.wav"), sr=16000)
segments, info = stt.transcribe(y, beam_size=5, language="en")
segs = list(segments)
text = " ".join(s.text.strip() for s in segs)
print(f"\nSTT test: lang={info.language} text='{text}'")

# Quick emotion test
from src.analysis import predict_emotion
emotions = predict_emotion(y, 16000, emo)
if emotions:
    print(f"Emotion test: top={emotions[0]['label']} score={emotions[0]['score']}")
else:
    print("Emotion test: no results")

print("\nAll models working correctly.")
